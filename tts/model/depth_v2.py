"""Depth transformer (DepthV2): predicts codebook levels 1..31 of each frame.

Geometry: 4 layers x 1024 x 16 heads, FFN 3072 (GELU). Eleven weight GROUPS:
levels 1-8 each get a PRIVATE trunk; levels 9-16, 17-24, 25-31 share three
band trunks. Speaker conditioning enters HERE ONLY: each reference embedding
(CAM++-1024 or ECAPA-192, mutually exclusive per example) is projected by a
source-specific Linear to one 1024-d K/V memory token; every layer carries a
cross-attention over that memory, SHARED across groups (the reconcile budget:
~560M all-in including the speaker path). Per-level code embeddings are
rank-128 factorized; 31 independent 1024->2048 heads.

Level indexing follows depth.py: level 0 (cb0) is the backbone's; this module
generates levels 1..31 (or 1..q-1 under level-dropout).

Null conditioning (matched joint dropout, 10%): spk_mem is replaced by a
learned null token — the SAME code path, so CFG at inference is just a second
forward with the null memory.

GROUPED teacher-forced forward (OPT-IN): a prefix position is
embedded from its SLOT alone (see _prefix) and the trunk self-attention is
causal, so within one weight group a single pass over the group's longest
prefix already contains, at position L, the state the level-L pass would
produce at its last position. Levels 9-16 / 17-24 / 25-31 share groups
8/9/10, so those three bands collapse to one pass each: 118 positions per
frame instead of 527 at q=32.

The maths is equivalent but the KERNEL ORDER is not, so the grouped logits
are NOT bit-for-bit the loop's: measured max |dlogit| 1.9e-06 in fp32,
i.e. equality only within GROUPED_FP32_ATOL. It is therefore OPT-IN —
`grouped` defaults to False, so the default forward is S2's per-level loop
kernel for kernel, and the control configuration and the zero-init
warm-start equivalence check are bitwise S2. Enable it per configuration
(config key `model.depth_grouped` -> DepthV2(..., grouped=True)) only after
that configuration has passed a short bf16 tolerance test (2 GPUs,
q16/q24/q32, GROUPED_BF16_SMOKE_ATOL).
"""
import functools
import math

import torch
import torch.nn as nn

from ..tokens import N_Q, N_CODES

GROUP_OF_LEVEL = [None] + [i for i in range(8)] + [8] * 8 + [9] * 8 + [10] * 7
N_GROUPS = 11
D, H, FFN, LAYERS = 1024, 16, 3072, 4
RANK = 128
# Selective activation checkpointing: checkpoint levels >= this during
# training (levels 1-8 have short prefixes and run direct; see _forward_loop).
# NOT an environment variable any more: the value is a per-RUN recipe
# knob, so it belongs in the closed config schema as `model.depth_ckpt_from`
# and in the dumped <run_dir>/config.yaml, not in whatever the launcher
# happened to export. scripts/s3_train.py REFUSES to start if the old
# `S2_CKPT_FROM` is still exported (a silent env override of a config key
# is exactly what this refusal prevents).
CKPT_FROM = 9              # loop-path default = S2, bitwise
GROUPED_CKPT_FROM = 1      # grouped-path default: see default_ckpt_from()

# Tolerances of record for the grouped path. The grouped and the
# loop path compute the same function in a DIFFERENT KERNEL ORDER, so they
# agree to floating point, never bitwise: measured max |dlogit| 1.88e-06 in
# fp32 at q32 against a logit scale of 2.65.
# The tests assert allclose(atol=GROUPED_FP32_ATOL) — ~5x headroom over that
# measurement — and torch.equal ONLY against the default (loop) path.
GROUPED_FP32_ATOL = 1e-5
# Pre-registered gate for turning grouped=True on for a training config: in
# TRAINING bf16 on 2 GPUs, grouped vs loop at q16/q24/q32 within
# atol=rtol=GROUPED_BF16_SMOKE_ATOL and a depth-loss delta below
# GROUPED_BF16_LOSS_DELTA. Provisional — to be tightened from a measurement
# before a full-scale run.
GROUPED_BF16_SMOKE_ATOL = 2e-2
GROUPED_BF16_LOSS_DELTA = 1e-3


def _check_ckpt_from(v):
    """Validate an explicit `ckpt_from` (1..31). 1 = checkpoint every level."""
    v = int(v)
    if not 1 <= v <= N_Q - 1:
        raise ValueError(f"ckpt_from={v}: must be 1..{N_Q - 1} (1 = every "
                         f"level checkpointed, {N_Q - 1} = only the last)")
    return v


def default_ckpt_from(grouped: bool) -> int:
    """Effective selective-checkpoint threshold when the config names none.

    LOOP path -> CKPT_FROM (9), which is S2 bit-for-bit: levels 1-8 run direct
    and 9-31 recompute.

    GROUPED path -> 1, i.e. every group pass is checkpointed. WHY the default
    differs: the grouped schedule keeps the eight private levels 1-8 as
    44 of its 118 positions, and at threshold 9 those are the ONLY positions
    whose activations are stored — so grouped saves trunk FLOPs but almost no
    activation memory (~6.9 vs the loop's ~7.4 GiB at 1/16 coverage, estimated).
    Checkpointing every pass drops that to ~3.0 GiB for +9% depth FLOPs, which
    is the memory training needs to afford 1/8 and 1/4 depth coverage. An
    explicit `model.depth_ckpt_from` overrides this in either mode.
    """
    return GROUPED_CKPT_FROM if grouped else CKPT_FROM


def sample_codes(logits, temp=1.0, top_k=0):
    """Draw one code per row from `logits` (N, n_codes) -> (N,) long.

    WHY it exists: serving samples the depth
    stack at temp 0.7 with top-k 64 (generate() below), while S2's roll-in
    sampled the FULL softmax at temp 1 — so the corrupted inputs roll-in fed
    the model were drawn from a distribution it never meets at serve time.
    One helper now spells the rule for the roll-in path.

    The defaults are S2 bit-for-bit: `p / 1.0` is the identity in IEEE-754
    and `top_k=0` skips the mask, leaving exactly
    torch.multinomial(logits.detach().float().softmax(-1), 1).squeeze(-1).
    """
    p = logits.detach().float()
    if temp != 1.0:
        p = p / temp
    if top_k:
        kth = p.topk(min(int(top_k), p.shape[-1]), -1).values[:, -1:]
        p = p.masked_fill(p < kth, -float("inf"))
    return torch.multinomial(p.softmax(-1), 1).squeeze(-1)


@functools.cache
def group_schedule(q):
    """Passes for levels 1..q-1 as ((group, (levels...)), ...), ascending.

    Levels 1-8 own groups 0-7, so each is its own pass (prefix lengths 2..9).
    Levels 9-16, 17-24 and 25-31 share groups 8, 9 and 10, so each band is ONE
    pass over its deepest prefix (lengths 17 / 25 / 32, clipped by q_eff).
    Cached: the schedule depends only on q, and q takes 3 values in training.
    """
    buckets = {}
    for level in range(1, q):
        buckets.setdefault(GROUP_OF_LEVEL[level], []).append(level)
    for g, lv in buckets.items():
        if lv != list(range(lv[0], lv[-1] + 1)):
            raise ValueError(
                f"GROUP_OF_LEVEL group {g} is not one contiguous run of "
                f"levels ({lv}) — the shared pass would be wrong")
    return tuple((g, tuple(lv)) for g, lv in buckets.items())


class _SelfBlock(nn.Module):
    """One private-trunk layer: causal self-attn + GELU FFN (pre-norm)."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.f1, self.f2 = nn.Linear(D, FFN, bias=False), nn.Linear(FFN, D, bias=False)

    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)

        def sh(t):
            return t.view(B, T, H, D // H).transpose(1, 2)
        a = nn.functional.scaled_dot_product_attention(sh(q), sh(k), sh(v),
                                                       attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.f2(nn.functional.gelu(self.f1(self.n2(x))))


class _CrossBlock(nn.Module):
    """Speaker cross-attention, shared across groups at one layer index.
    kmask (B, S) bool, True = real reference token (padding excluded:
    repeat-padding would double a ref's attention mass)."""

    def __init__(self):
        super().__init__()
        self.n = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)

    def forward(self, x, mem, kmask=None):
        B, T, _ = x.shape
        S = mem.shape[1]
        q = self.q(self.n(x)).view(B, T, H, D // H).transpose(1, 2)
        k, v = self.kv(mem).chunk(2, -1)
        k = k.view(B, S, H, D // H).transpose(1, 2)
        v = v.view(B, S, H, D // H).transpose(1, 2)
        am = kmask[:, None, None, :] if kmask is not None else None
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return x + self.o(a.transpose(1, 2).reshape(B, T, D))


class DepthV2(nn.Module):
    def __init__(self, d_backbone=2048, n_q=N_Q, n_codes=N_CODES,
                 grouped: bool = False, ckpt_from: int | None = None):
        """grouped: the DEFAULT path taken by forward() when its per-call
        `grouped` kwarg is None. False = S2's per-level loop, kernel for
        kernel (the strict S2 equivalence check needs a bitwise S2 default);
        True = the grouped passes, which agree only to
        GROUPED_FP32_ATOL. Wired from config key `model.depth_grouped`
        (default false) by TTSModelV2. Deliberately a PLAIN attribute, not a
        buffer: the state_dict — and hence the S2 checkpoint contract — is
        unchanged, and the flag is a property of the RUN, not of the weights.

        ckpt_from: selective activation-checkpointing threshold, 1..31 (levels
        >= it recompute during training). None = default_ckpt_from(grouped) —
        9 on the loop path (S2 bitwise), 1 on the grouped path (the memory
        lever). Wired from config key `model.depth_ckpt_from`; also a
        plain attribute, and also a property of the RUN, not of the weights.
        """
        super().__init__()
        self.grouped = bool(grouped)
        # Selective-checkpoint threshold, per RUN (config `model.depth_ckpt_from`).
        # None = the mode's default: see default_ckpt_from(). Like
        # `grouped` this is a plain attribute — it must not enter state_dict.
        self.ckpt_from = default_ckpt_from(self.grouped) if ckpt_from is None \
            else _check_ckpt_from(ckpt_from)
        self.n_q, self.n_codes = n_q, n_codes
        self.in_proj = nn.Linear(d_backbone, D, bias=False)
        # factorized per-level code embeddings: (codes -> rank) private per
        # level, (rank -> D) shared
        self.code_lo = nn.Embedding((n_q - 1) * n_codes, RANK)
        self.code_up = nn.Linear(RANK, D, bias=False)
        self.level_emb = nn.Embedding(n_q, D)
        # 11 private trunks x 4 layers; cross-attn shared across groups per layer
        self.groups = nn.ModuleList(
            nn.ModuleList(_SelfBlock() for _ in range(LAYERS))
            for _ in range(N_GROUPS))
        self.cross = nn.ModuleList(_CrossBlock() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.heads = nn.ModuleList(
            nn.Linear(D, n_codes, bias=False) for _ in range(n_q - 1))
        # speaker memory: source-specific projections + learned null token
        self.spk_proj_campp = nn.Linear(1024, D, bias=False)
        self.spk_proj_ecapa = nn.Linear(192, D, bias=False)
        self.null_mem = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.null_mem, std=0.02)
        nn.init.normal_(self.code_lo.weight, std=0.02)
        nn.init.normal_(self.level_emb.weight, std=0.02)

    def spk_memory(self, refs, source, null_mask=None):
        """refs: (B, R, dim) L2-normalized reference embeddings (dim 1024 for
        'campp', 192 for 'ecapa'); source: str or (B,) list of str;
        null_mask: (B,) bool — replace that row's memory with the null token
        (matched joint dropout / CFG negative). Returns (B, R, D)."""
        B, R, _ = refs.shape
        if isinstance(source, str):
            proj = self.spk_proj_campp if source == "campp" else self.spk_proj_ecapa
            mem = proj(refs)
        else:
            mem = torch.zeros(B, R, D, dtype=refs.dtype, device=refs.device)
            for i, s in enumerate(source):
                proj = self.spk_proj_campp if s == "campp" else self.spk_proj_ecapa
                mem[i] = proj(refs[i, :, : proj.in_features])
        if null_mask is not None and null_mask.any():
            mem[null_mask] = self.null_mem.to(mem.dtype).expand(-1, R, -1)
        return mem

    def _prefix(self, h_frame, codes_prev, level):
        """Embedded prefix for predicting `level`: [proj(h)+lvl0] +
        [emb(cb_k, slot k) + lvl_{k+1} for k=0..level-1] -> (B, level+1, D).
        codes_prev columns are cb0..cb30 — cb0 IS in the within-frame
        factorization; generate() builds the identical
        prefix from the sampled cb0 + its own sampled deeper codes."""
        parts = [self.in_proj(h_frame)[:, None] + self.level_emb.weight[:1]]
        ks = torch.arange(level, device=h_frame.device)
        flat = codes_prev[:, :level] + ks * self.n_codes
        e = self.code_up(self.code_lo(flat)) + self.level_emb(ks + 1)[None]
        parts.append(e)
        return torch.cat(parts, 1)

    def _trunk(self, x, group, mem, kmask=None):
        """One group's 4 causal self-attn layers interleaved with the 4 SHARED
        speaker cross-attn layers, over the whole prefix; all positions, unnormed."""
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        for li in range(LAYERS):
            x = self.groups[group][li](x, mask)
            x = self.cross[li](x, mem, kmask)
        return x

    def _run(self, x, group, mem, kmask=None):
        """Last position only — the per-level path and generate() use this."""
        return self.norm(self._trunk(x, group, mem, kmask)[:, -1])

    def _run_at(self, x, group, mem, kmask=None, take=(-1,)):
        """Same trunk, read out at several positions -> (B, len(take), D).

        WHY this is exact: _prefix embeds position k from slot k
        alone — never from the level being predicted — and the self-attention
        is causal, so position L of a length-(L'+1) prefix (L <= L') sees the
        same tokens under the same weights as the level-L pass sees at ITS
        last position. Every layer after that (cross-attn over the speaker
        memory, FFN, LayerNorm) is position-wise. `take` is a plain tuple so
        torch.utils.checkpoint passes it through untouched.
        """
        return self.norm(self._trunk(x, group, mem, kmask)[:, list(take)])

    def _forward_loop(self, h_frames, inputs, spk_mem, ref_mask, q,
                      rollin_mask=None, rollin_temp=1.0, rollin_top_k=0):
        """S2's per-level path: 31 passes over prefixes 2..32 = 527 positions.

        Kept (a) as the reference implementation the grouped path is tested
        against and (b) as the ONLY path for roll-in rows: such a row's
        level-L input is SAMPLED from this same forward's level-(L-1) logits,
        so its prefix does not exist until the previous level has run and the
        band passes cannot be shared. `inputs` is mutated in place for roll-in
        rows — every caller hands over a clone.
        """
        out = []
        for level in range(1, q):
            x = self._prefix(h_frames, inputs, level)
            if (self.training and torch.is_grad_enabled()
                    and level >= self.ckpt_from):
                # selective activation checkpointing:
                # the un-checkpointed 31-level loop held ~25GB and OOM'd the
                # first launch; levels 1-8 (short prefixes) stay direct —
                # 35/80 GB peak leaves the headroom — and levels 9-31
                # recompute, cutting the recompute tax on the cheap levels
                h = torch.utils.checkpoint.checkpoint(
                    self._run, x, GROUP_OF_LEVEL[level], spk_mem, ref_mask,
                    use_reentrant=False)
            else:
                h = self._run(x, GROUP_OF_LEVEL[level], spk_mem, ref_mask)
            logits = self.heads[level - 1](h)
            out.append(logits)
            if rollin_mask is not None and level < q - 1:
                # roll-in replaces cb1.. only; column 0 is substituted once,
                # before the loop, from rollin_cb0 if the caller sampled it
                with torch.no_grad():
                    samp = sample_codes(logits, rollin_temp, rollin_top_k)
                    inputs[rollin_mask, level] = samp[rollin_mask]
        return out

    def _forward_grouped(self, h_frames, inputs, spk_mem, ref_mask, q):
        """One causal pass per weight GROUP instead of one per level.

        COST MODEL — positions processed per frame, summed over passes:
          loop     sum_{L=1..q-1} (L+1)                = 527 (q32) 299 (q24) 135 (q16)
          grouped  (2+..+9) + 17 + 25 + 32             = 118 (q32)  85 (q24)  60 (q16)
        i.e. 22.4% of the trunk work at q32 for identical outputs. The saving
        is trunk-only: the 31 head matmuls and the prefix embeddings are
        unchanged, and the grouped path builds ONE prefix per group instead of
        one per level (11 instead of 31 at q32).
        """
        out = [None] * (q - 1)
        for g, lv in group_schedule(q):
            x = self._prefix(h_frames, inputs, lv[-1])
            if (self.training and torch.is_grad_enabled()
                    and lv[-1] >= self.ckpt_from):
                # SAME threshold convention, applied per PASS: a pass is
                # checkpointed iff any level it serves would have been (its
                # deepest level lv[-1] is the test). At the LOOP default
                # ckpt_from=9 that is exactly the three shared bands, which
                # are also the memory-heavy passes (prefixes 17/25/32 held
                # for 4 self- + 4 cross-attn layers); the eight private
                # levels 1-8 stay direct, as in S2 — which is why the GROUPED
                # default is ckpt_from=1: with 9, those 44 of 118
                # positions are the entire activation bill and the grouped
                # path saves trunk FLOPs but almost no memory.
                hs = torch.utils.checkpoint.checkpoint(
                    self._run_at, x, g, spk_mem, ref_mask, lv,
                    use_reentrant=False)
            else:
                hs = self._run_at(x, g, spk_mem, ref_mask, lv)
            for j, level in enumerate(lv):
                out[level - 1] = self.heads[level - 1](hs[:, j])
        return out

    def _forward_split(self, h_frames, inputs, spk_mem, ref_mask, q,
                       rollin_mask, rollin_temp, rollin_top_k, sample=True):
        """Mixed batch: roll-in rows take the sequential loop, the rest take
        the group passes; the two results are re-merged row-wise.

        `sample=False` (set by forward() when the caller supplied
        `rollin_codes`) means the roll-in inputs are already in `inputs` and
        must NOT be re-drawn. The rows still take the sequential loop, so the
        split/merge machinery is exercised exactly as in a real training step —
        only the draw is gone.

        COST MODEL with a roll-in row fraction r at q32:
          r*527 + (1-r)*118 positions, vs 527 for the pure loop. The training
          recipe ramps roll-in to 0.30 of the depth frames, so the
          steady state is ~0.30*527 + 0.70*118 = 241 positions = 46% of S2.
          The extra cost of splitting is one more prefix build and two
          index_copy per level; both are O(N*D), negligible against a trunk
          pass. Row order is preserved by index_copy on sorted index vectors.
        """
        ridx = rollin_mask.nonzero(as_tuple=True)[0]
        nidx = (~rollin_mask).nonzero(as_tuple=True)[0]
        r_out = self._forward_loop(
            h_frames[ridx], inputs[ridx], spk_mem[ridx],
            None if ref_mask is None else ref_mask[ridx], q,
            rollin_mask.new_ones(ridx.numel()) if sample else None,
            rollin_temp, rollin_top_k)
        if nidx.numel() == 0:
            return r_out                      # ridx == arange(N): order kept
        n_out = self._forward_grouped(
            h_frames[nidx], inputs[nidx], spk_mem[nidx],
            None if ref_mask is None else ref_mask[nidx], q)
        n_rows = h_frames.shape[0]
        return [a.new_zeros(n_rows, a.shape[-1])
                .index_copy(0, nidx, a).index_copy(0, ridx, b)
                for a, b in zip(n_out, r_out)]

    def forward(self, h_frames, targets, refs=None, null_mask=None, q_eff=None,
                rollin_mask=None, source="ecapa", spk_mem=None, ref_mask=None,
                grouped=None, rollin_temp=1.0, rollin_top_k=0, rollin_cb0=None,
                rollin_codes=None):
        """Training (teacher-forced). h_frames (N, d_backbone); targets
        (N, n_q) ground-truth codes incl. level 0; refs (N, R, dim) reference
        embeddings projected HERE (inside Module.__call__, so FSDP2's
        all-gather hook covers the speaker projections); null_mask (N,) bool.
        rollin_mask (N,) bool: those frames read STOP-GRADIENT sampled codes
        (from this forward's own logits) instead of teacher codes at levels
        >=2 — scheduled sampling for the depth stack (roll-in).
        q_eff: level-dropout Q. Returns list of logits per level 1..q_eff-1
        (each (N, n_codes)) — same contract as depth.py. Loss targets are
        ALWAYS the true codes; roll-in only corrupts inputs.

        S3 kwargs (all default to S2 behaviour bit-for-bit):
        grouped: None (default) = use the module attribute `self.grouped`,
          set from the constructor kwarg / config key `model.depth_grouped`
          and FALSE unless that configuration opted in. True runs one causal
          pass per weight group instead of one per level: equivalent
          maths, different KERNEL ORDER, hence equal to the loop only within
          GROUPED_FP32_ATOL (1.9e-06 measured in fp32) — which is why it is
          opt-in and why the equivalence check runs the default.
          The per-call kwarg overrides the attribute and exists so the tests
          (and a bf16 tolerance test) can drive both paths on one module. See _run_at
          for WHY it is exact in exact arithmetic and
          _forward_grouped/_forward_split for the cost model.
        rollin_temp / rollin_top_k: sampling rule for the roll-in inputs
          Serving uses 0.7 / 64; S2 roll-in used 1.0 / 0,
          which is the default here so nothing moves until a caller asks.
        rollin_cb0 (N,) long or None: for rows in rollin_mask, the level-1
          INPUT (prefix slot 0, and hence every deeper prefix of that row —
          exactly what generate() does with its own sampled cb0) uses this
          code instead of the true cb0. The caller (tts_model_v2) samples it
          from the backbone's cb0 logits for the selected frames. TARGETS
          NEVER CHANGE: the loss still scores against the true codes. Codes
          must be valid cb0 classes; AEOS/invalid must be resolved by the
          caller, same contract as generate().
        rollin_codes (N, n_q) long or None: TEACHER-FORCED roll-in. When
          given, the roll-in rows read columns 1..q-2 of this tensor as their
          level inputs and NOTHING is sampled inside this forward; column 0 is
          untouched (it stays governed by rollin_cb0 / the true cb0) and
          columns >= q-1 are never read as inputs at all. Default None = the
          sampled roll-in, bitwise: the only added work is one Python `is not None` test.

          WHY IT EXISTS (found in a bf16 test run). The loop path draws its roll-in
          samples over the FULL batch and _forward_split draws them over the
          roll-in SUB-batch, so the two consume different RNG and, in bf16,
          different reduction tilings flip near-tie argmaxes: the sampled codes
          diverge and the paths stop computing the same function. Measured on a
          checkpoint at update 84000: `noroll` is EXACT at every q while the roll-in case reaches
          max |dlogit| ~10-19 purely from that divergence. Injecting the loop's
          own draws makes both paths see identical inputs at every level, which
          is the only way the grouped/split machinery can be gated on the
          roll-in path at all. TARGETS ARE NEVER TOUCHED — the loss still
          scores against the true codes.
        """
        if grouped is None:
            grouped = self.grouped        # config default; see __init__
        q = q_eff or self.n_q
        if spk_mem is None:
            spk_mem = self.spk_memory(refs, source, null_mask)
        if ref_mask is not None and null_mask is not None:
            # null rows attend over all R copies of the null token
            ref_mask = ref_mask | null_mask[:, None]
        inputs = targets                      # columns cb0..cb31; :31 consumed
        roll = rollin_mask is not None and bool(rollin_mask.any())
        if roll:
            inputs = inputs.clone()
            if rollin_cb0 is not None:
                if rollin_cb0.shape != (h_frames.shape[0],):
                    raise ValueError(
                        f"rollin_cb0 must be ({h_frames.shape[0]},), got "
                        f"{tuple(rollin_cb0.shape)}")
                sub = rollin_cb0[rollin_mask]
                if bool(((sub < 0) | (sub >= self.n_codes)).any()):
                    raise ValueError(
                        "rollin_cb0 carries aeos/invalid cb0 classes; the "
                        "caller must resolve them (as generate() requires)")
                inputs[rollin_mask, 0] = sub.to(inputs.dtype)
        force = rollin_codes is not None
        if force:
            if not roll:
                raise ValueError(
                    "rollin_codes needs a rollin_mask with at least one row: "
                    "there is nothing to force otherwise")
            if rollin_codes.shape != (h_frames.shape[0], self.n_q):
                raise ValueError(
                    f"rollin_codes must be ({h_frames.shape[0]}, {self.n_q}), "
                    f"got {tuple(rollin_codes.shape)}")
            hi = max(1, q - 1)          # columns 1..q-2 are what sampling writes
            sub = rollin_codes[rollin_mask, 1:hi]
            if sub.numel() and bool(((sub < 0) | (sub >= self.n_codes)).any()):
                raise ValueError(
                    "rollin_codes carries aeos/invalid classes; the caller "
                    "must resolve them (as rollin_cb0 requires)")
            inputs[rollin_mask, 1:hi] = sub.to(inputs.dtype)
        if not grouped:
            out = self._forward_loop(h_frames, inputs, spk_mem, ref_mask, q,
                                     rollin_mask if (roll and not force) else None,
                                     rollin_temp, rollin_top_k)
        elif roll:
            out = self._forward_split(h_frames, inputs, spk_mem, ref_mask, q,
                                      rollin_mask, rollin_temp, rollin_top_k,
                                      sample=not force)
        else:
            out = self._forward_grouped(h_frames, inputs, spk_mem, ref_mask, q)
        # gradient-symmetry anchor: params whose use varies with the batch
        # (null token on null rows; the dormant CAM++ projection; the ECAPA
        # projection when every row is null) must produce grads on EVERY rank
        # every audio step, or FSDP2's reduce-scatter set diverges (S1 lesson)
        anchor = 0.0 * (self.null_mem.sum() + self.spk_proj_campp.weight.sum()
                        + self.spk_proj_ecapa.weight.sum())
        out[0] = out[0] + anchor.to(out[0].dtype)
        return out

    @torch.inference_mode()
    def generate(self, h_frame, cb0, spk_mem, q_serve=None, temp=0.7,
                 top_k=64, cfg_scale=0.0, null_mem=None, ref_mask=None):
        """Serve one frame. cfg_scale>0 applies depth-logit CFG:
        logits = null + scale*(cond - null) (depth only, not
        cb0). null_mem defaults to the learned null token. The prefix uses
        the SAMPLED cb0 + this frame's own sampled deeper codes — identical
        construction to teacher forcing."""
        from ..tokens import AEOS_CLASS
        if torch.any(cb0 >= AEOS_CLASS) or torch.any(cb0 < 0):
            raise ValueError("aeos/invalid cb0 must be handled by the caller")
        q = q_serve or self.n_q
        B = h_frame.shape[0]
        if cfg_scale > 0 and null_mem is None:
            null_mem = self.null_mem.to(spk_mem.dtype).expand(B, spk_mem.shape[1], -1)
        outs = [cb0]
        for level in range(1, q):
            codes_prev = torch.stack(outs, 1)                 # cb0..cb_{l-1}
            x = self._prefix(h_frame, codes_prev, level)
            h = self._run(x, GROUP_OF_LEVEL[level], spk_mem, ref_mask)
            logits = self.heads[level - 1](h).float()
            if cfg_scale > 0:
                hn = self._run(x, GROUP_OF_LEVEL[level], null_mem)
                ln = self.heads[level - 1](hn).float()
                logits = ln + cfg_scale * (logits - ln)
            logits = logits / temp
            if top_k:
                kth = logits.topk(top_k, -1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            outs.append(torch.multinomial(logits.softmax(-1), 1).squeeze(-1))
        return torch.stack(outs, -1)                          # (B, q)


if __name__ == "__main__":
    m = DepthV2()
    n = sum(p.numel() for p in m.parameters())
    parts = {
        "group_trunks": sum(p.numel() for p in m.groups.parameters()),
        "cross": sum(p.numel() for p in m.cross.parameters()),
        "heads": sum(p.numel() for p in m.heads.parameters()),
        "code_emb": m.code_lo.weight.numel() + m.code_up.weight.numel(),
        "spk_path": (m.spk_proj_campp.weight.numel()
                     + m.spk_proj_ecapa.weight.numel() + m.null_mem.numel()),
    }
    print({k: f"{v/1e6:.1f}M" for k, v in parts.items()}, f"TOTAL {n/1e6:.1f}M")
    B = 3
    h = torch.randn(B, 2048)
    tgt = torch.randint(0, 2048, (B, 32))
    refs = torch.nn.functional.normalize(torch.randn(B, 3, 1024), dim=-1)
    outs = m(h, tgt, refs=refs, source="campp",
             null_mask=torch.tensor([False, True, False]), q_eff=16)
    assert len(outs) == 15 and outs[0].shape == (B, 2048)
    er = torch.nn.functional.normalize(torch.randn(B, 3, 192), dim=-1)
    outs = m(h, tgt, refs=er, null_mask=torch.zeros(B, dtype=torch.bool),
             q_eff=16, rollin_mask=torch.tensor([True, False, True]),
             ref_mask=torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]]).bool())
    assert len(outs) == 15
    (sum(o.float().pow(2).mean() for o in outs)).backward()
    for nm in ("null_mem", "spk_proj_campp.weight", "spk_proj_ecapa.weight"):
        g = dict(m.named_parameters())[nm].grad
        assert g is not None, f"symmetry anchor failed for {nm}"
    # acceptance: cb0 must steer level-1; cb_k steers level k+1 but never
    # levels <= k (within-frame causality)
    with torch.no_grad():
        nm = torch.zeros(B, dtype=torch.bool)
        o1 = m(h, tgt, refs=er, null_mask=nm, q_eff=8)
        t2 = tgt.clone(); t2[:, 0] = (t2[:, 0] + 7) % 2048
        o2 = m(h, t2, refs=er, null_mask=nm, q_eff=8)
        assert not torch.allclose(o1[0], o2[0]), "cb0 does not reach level 1"
        t3 = tgt.clone(); t3[:, 2] = (t3[:, 2] + 7) % 2048
        o3 = m(h, t3, refs=er, null_mask=nm, q_eff=8)
        assert torch.allclose(o1[0], o3[0]) and torch.allclose(o1[1], o3[1]), \
            "cb2 leaked into levels <= 2"
        assert not torch.allclose(o1[2], o3[2]), "cb2 does not reach level 3"
    # the grouped path reproduces the per-level loop to
    # GROUPED_FP32_ATOL (NOT bitwise — kernel order differs), and the DEFAULT
    # path is the loop itself, bit for bit
    assert m.grouped is False, "grouped must default to OFF"
    with torch.no_grad():
        nm = torch.zeros(B, dtype=torch.bool)
        for qq in (16, 24, 32):
            og = m(h, tgt, refs=er, null_mask=nm, q_eff=qq, grouped=True)
            ol = m(h, tgt, refs=er, null_mask=nm, q_eff=qq, grouped=False)
            dflt = m(h, tgt, refs=er, null_mask=nm, q_eff=qq)
            assert all(torch.allclose(a, b, atol=GROUPED_FP32_ATOL)
                       for a, b in zip(og, ol)), f"grouped != loop at q={qq}"
            assert all(torch.equal(a, b) for a, b in zip(dflt, ol)), \
                f"the default forward is not the S2 loop at q={qq}"
    # teacher-forced vs generation prefix identity for a fixed code sequence
    cp = tgt[:1, :5]
    x_tf = m._prefix(h[:1], cp, 5)
    x_gen = m._prefix(h[:1], torch.stack([cp[:, j] for j in range(5)], 1), 5)
    assert torch.equal(x_tf, x_gen), "teacher/gen prefix construction differs"
    g = m.generate(torch.randn(1, 2048), torch.tensor([7]),
                   m.spk_memory(torch.nn.functional.normalize(
                       torch.randn(1, 2, 192), dim=-1), "ecapa"),
                   q_serve=16, cfg_scale=2.0)
    assert g.shape == (1, 16)
    print("depth_v2 self-test OK")
