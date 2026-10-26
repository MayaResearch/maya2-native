"""HISTORY CORRUPTION of the backbone's audio input.

WHY: S2 was trained with complete temporal
teacher forcing, so the backbone always saw ground-truth history and learned to
COPY the acoustic condition of that history (level, hiss) into the next frame.
At serving its own imperfect output is the history and the condition snowballs
— the measured drift lives in the COARSE codes, i.e. in what the backbone
reads, not in the deep codebooks. The fix is to corrupt what the backbone reads
as history for TARGET frames while every LABEL stays the clean truth, so the
model must predict the clean next frame from a history it cannot trust.

What is NEVER touched (the correctness contract, tested in
tests/test_anchor_corrupt.py):
  * in-context REFERENCE frames — they are the voice prompt; corrupting them
    would teach the model to ignore the reference;
  * `batch["codes"]` itself — cb0 targets, AEOS targets, depth targets AND the
    depth teacher-forced prefix all read the ORIGINAL tensor
    (tts_model_v2.py forward), so this module returns a CLONE;
  * levels >= q_use (BOTH kinds; the first
    version copied all 32 codebooks on a swap, which `audio_in` happened not to
    read), and levels < first_level for level_noise.

Two corruption kinds, mixed by weight:
  span_swap  (.625) a contiguous 1-4 s window of the target span's history is
                    replaced by a same-length window of a DONOR span in the
                    same batch: same language, same q_use, and — when
                    `require_other_speaker` is set — a DIFFERENT SPEAKER.
                    Realistic-but-wrong acoustics.

DONOR SPEAKER. The first version required only a
different BATCH ROW and called that "a different speaker group". It is not: a
batch is up to 96 rows drawn from ONE shard, so several rows are routinely the
same (video, speaker) group. A same-speaker donor swaps temporal CONTENT while
leaving the history ACOUSTICS (level, hiss, timbre) exactly as they were — i.e.
it does not break the acoustic-copy shortcut described above, which is the whole
point of the treatment. With `require_other_speaker: true` the donor must have
a KNOWN, DIFFERENT `batch["speaker_id"]`; a row with no eligible donor falls
back to level_noise for that window (it is never left uncorrupted, and never
swapped from its own speaker). Two counts quantify the gap: the realised
same-speaker swap frames, and the frames the unconstrained row-only rule would have taken
from the same speaker in expectation (`swap_same_spk_expected_frames`) — the
measurement of how weak an unconstrained arm's treatment really was.
  level_noise(.375) the same kind of window keeps levels < first_level (8) and
                    randomises levels in [first_level, q_use). Coarse structure
                    survives, the fine condition does not.

Ramp: the caller passes the current max frame fraction (0 -> 0.30 over 10k
updates). Per span we draw ONE window length W and the NUMBER
of windows k so that E[corrupted target frames] == frac * target frames; the
realised counts come back in `stats` for telemetry.

DEVICE / STALLS. The first version read four GPU
tensors back with `.tolist()`, drew every decision as a python scalar in a loop
over spans and windows, and copied the noise block CPU->GPU inside the step —
several synchronisations plus a host round-trip per update. This version builds
the corrupted tensor ENTIRELY ON THE CODES' DEVICE with fixed-shape vectorised
ops: no `.tolist()`, no `.item()`, no python loop over the batch, no H2D of
generated values, and no data-dependent shape (which would itself sync, because
allocating a masked_select output needs the count on the host). "Corrupt on the
CPU before H2D" was rejected: the loop hands the model a batch that is already
on the GPU (tts/train/loop.py:200), so a CPU build would mean a full D2H+H2D of
`codes` every step.

Consequences of that decision, both REQUIRED of the caller:
  * `gen` must be a torch.Generator ON `batch["codes"].device` (a CUDA
    generator for GPU training: `torch.Generator(device="cuda")`); a CPU
    generator with GPU codes raises here instead of silently syncing.
  * every count in `stats` is a 0-dim int64 tensor on that device, plus a
    stacked 1-D `stats["counts"]` in `stats["count_keys"]` order. Accumulate
    the TENSORS across updates and divide ONCE at logging time
    (Sigma corrupted_frames / Sigma target_frames) — reading them per step with
    float() re-introduces the stall, and AVERAGING per-batch `realised_frac`
    over updates is the wrong estimator (it weights a 2-frame batch like a
    2000-frame one and counts text-only batches as 0).

Window PLACEMENT is a random-phase tiling: the k windows of a span sit on the
grid `phase + i*W` for k distinct slots i, phase ~ U[0, L - floor(L/W)*W].
That makes windows non-overlapping BY CONSTRUCTION (the first version drew
uniform starts and retried up to 4 times, so overlaps still happened and the
realised fraction sagged), which is what makes E[corrupted] = frac * L exact.
"""
import math
from dataclasses import dataclass

import torch

from ..tokens import FRAME_HZ, N_CODES, N_Q

#: `stats["counts"]` layout — a 1-D int64 tensor the caller can accumulate on
#: device without a sync. FRAME counts are the ones telemetry must divide.
#: New keys are APPENDED, never inserted: scripts/s3_train.py all-reduces this
#: vector by index and its logged names are a run-to-run contract.
COUNT_KEYS = ("target_frames", "corrupted_frames", "swap_frames",
              "noise_frames", "noise_noop_frames", "windows", "swap_draws",
              "noise_draws", "donor_fallbacks",
              # ---- donor speaker constraint, all FRAME-weighted ----
              # swap draws that found NO eligible donor and became level_noise
              # (the window-weighted form of `donor_fallbacks`); these frames
              # are also inside `noise_frames`.
              "swap_fallback_frames",
              # frames REALLY swapped from a donor with the same KNOWN speaker.
              # Identically 0 when require_other_speaker is on — the invariant
              # the fix exists for. Unconstrained, this is exactly how much of
              # the treatment was a no-op.
              "swap_same_spk_frames",
              # frames the UNCONSTRAINED (different-row-only) rule would have
              # taken from the same speaker, in EXPECTATION (donors are drawn
              # uniformly over the eligible set, so this is n_same/n_eligible
              # per window x its length, rounded once per batch). Informational
              # counterfactual; equals swap_same_spk_frames in expectation when
              # the constraint is off.
              "swap_same_spk_expected_frames",
              # frames swapped while NO speaker constraint was in force (knob
              # off, or a batch with no speaker_id at all — old callers)
              "swap_unconstrained_frames")


@dataclass(frozen=True)
class CorruptConfig:
    """Parsed `history_corruption` config block. `from_dict(None)` and a block with
    `enabled: false` both return None = feature OFF = S2 behaviour."""
    swap_w: float = 0.625
    noise_w: float = 0.375
    dur_lo: float = 1.0                 # seconds
    dur_hi: float = 4.0
    first_level: int = 8
    #: span_swap donors must be a DIFFERENT, KNOWN speaker. Default False
    #: = the original behaviour (different batch row only), bit-for-bit.
    require_other_speaker: bool = False

    @classmethod
    def from_dict(cls, d):
        if not d or not d.get("enabled", False):
            return None
        sw = d.get("span_swap") or {}
        nz = d.get("level_noise") or {}
        dur = list(sw.get("duration_s", (1.0, 4.0)))
        c = cls(swap_w=float(sw.get("weight", 0.625)),
                noise_w=float(nz.get("weight", 0.375)),
                dur_lo=float(dur[0]), dur_hi=float(dur[1]),
                first_level=int(nz.get("first_level", 8)),
                require_other_speaker=bool(
                    d.get("require_other_speaker", False)))
        if c.swap_w < 0 or c.noise_w < 0 or c.swap_w + c.noise_w <= 0:
            raise ValueError(f"history_corruption weights must be >=0 and sum "
                             f">0, got {c.swap_w}/{c.noise_w}")
        if not 0 < c.dur_lo <= c.dur_hi:
            raise ValueError(f"span_swap.duration_s must be 0 < lo <= hi, got {dur}")
        if not 1 <= c.first_level < N_Q:
            raise ValueError(f"level_noise.first_level must be in [1,{N_Q}), "
                             f"got {c.first_level}")
        return c

    @property
    def p_swap(self):
        return self.swap_w / (self.swap_w + self.noise_w)


def _span_geometry(batch, dev):
    """The four (S,) span tensors collate_v2._add_span_geometry emits
    (packing_v2.py), moved to `dev` only if the caller left them elsewhere."""
    need = ("span_batch_row", "span_start", "span_len", "span_is_target")
    if any(k not in batch for k in need):
        raise ValueError("history corruption needs collate_v2's span geometry "
                         f"{need} (see packing_v2._add_span_geometry)")
    return tuple(batch[k].to(dev) for k in need)


def _lang_ids(batch, dev):
    """Per-ROW language id. `batch["langs"]` is a python list of strings (it is
    not a tensor, so tts/train/loop.py:200 leaves it on the host); one small
    B-element H2D per step is the whole cost, and it buys a device-side
    same-language donor test."""
    langs = batch.get("langs")
    if langs is None:
        raise ValueError("history corruption needs batch['langs'] (per-row "
                         "language from collate_v2, packing_v2.py:158)")
    seen = {}
    ids = [seen.setdefault(lg, len(seen)) for lg in langs]
    return torch.tensor(ids, dtype=torch.long, device=dev)


def _speaker_ids(batch, cfg, dev):
    """Per-ROW speaker id, or None when the batch carries no identity.

    `collate_v2` always emits `speaker_id` (packing_v2._speaker_ids); a batch
    that does not have it is an OLD caller (a fixture from before the constraint, a hand-built
    serving batch). Those keep the historical unconstrained donor rule, which
    is why the key is optional — but a config that ASKED for the constraint and
    then silently ran the weak treatment for 300k updates is exactly the defect
    the constraint exists to remove, so that combination is fatal (no silent
    fallbacks)."""
    spk = batch.get("speaker_id")
    if spk is None:
        if cfg.require_other_speaker:
            raise ValueError(
                "history_corruption.require_other_speaker is set but the batch "
                "has no `speaker_id` — collate_v2 emits it from each example's "
                "`speaker_key` (tts/data/packing_v2.py); a hand-built batch "
                "must supply a (B,) long tensor, -1 where the speaker is "
                "unknown")
        return None
    return torch.as_tensor(spk, device=dev).reshape(-1).long()


def _zero_stats(frac, dev):
    z = torch.zeros(len(COUNT_KEYS), dtype=torch.long, device=dev)
    return _pack_stats(frac, z, dev)


def _pack_stats(frac, counts, dev):
    st = {"requested_frac": float(frac), "counts": counts,
          "count_keys": COUNT_KEYS}
    st.update({k: counts[i] for i, k in enumerate(COUNT_KEYS)})
    # per-BATCH only; telemetry must use Sigma corrupted / Sigma target instead
    # (module docstring)
    st["realised_frac"] = (counts[1].double()
                           / counts[0].clamp(min=1).double())
    return st


def corrupt_history(batch, cfg, frac, gen=None, trace=False):
    """Return (codes_for_the_backbone_input, stats).

    `codes_for_the_backbone_input` is `batch["codes"]` ITSELF when nothing can
    be corrupted (no clone, no cost) and a clone otherwise — the caller must
    treat it as read-only and keep using `batch["codes"]` for every label.

    frac: current max frame fraction from the ramp (0..0.30). cfg: CorruptConfig
    or None (OFF). gen: torch.Generator on `batch["codes"].device` (see the
    module docstring). trace=True materialises a per-window record for
    fixed-seed tests — it is the ONLY path that syncs, and it is test-only.
    """
    codes = batch["codes"]
    dev = codes.device
    if cfg is None or float(frac) <= 0.0:
        return codes, _zero_stats(frac, dev)
    if gen is not None and gen.device.type != dev.type:
        raise ValueError(
            f"corrupt_history: generator is on {gen.device} but codes are on "
            f"{dev}; pass a per-rank torch.Generator(device='{dev.type}') — "
            "every draw is a device tensor op")

    rows, starts, lens, is_tgt = _span_geometry(batch, dev)
    S = int(rows.shape[0])                      # tensor SHAPES never sync
    B, T = int(codes.shape[0]), int(codes.shape[1])
    if S == 0:                                  # text-replay-only batch
        return codes, _zero_stats(frac, dev)
    lang = _lang_ids(batch, dev)[rows]          # (S,) language of each span
    spk_row = _speaker_ids(batch, cfg, dev)     # (B,) or None
    spk = None if spk_row is None else spk_row[rows]        # (S,)
    # python bool fixed by the CONFIG (a missing speaker_id already raised
    # above), so every rank takes the same branch — no data-dependent control
    # flow in the step.
    constrained = bool(cfg.require_other_speaker)
    # q_use stays a TENSOR: turning it into a python int here would sync, and
    # it is only ever needed as a level mask.
    q_use = torch.as_tensor(batch["q_use"], device=dev).reshape(())
    lvl = torch.arange(N_Q, device=dev)
    in_q = lvl < q_use                                    # (N_Q,)
    noise_lvl = in_q[cfg.first_level:]                    # (N_Q-first_level,)
    lens_f = lens.to(torch.float32)

    # ---- per-span draws: window length W, window count k ------------------
    u = torch.rand(S, generator=gen, device=dev)
    W = torch.round((cfg.dur_lo + (cfg.dur_hi - cfg.dur_lo) * u) * FRAME_HZ)
    W = W.long().clamp(min=1)
    W = torch.minimum(W, lens.clamp(min=1))               # window fits the span
    n_slots = torch.div(lens, W, rounding_mode="floor").clamp(min=1)
    want = frac * lens_f / W.to(torch.float32)            # E[k] for E[kW]=fL
    k = want.floor()
    k = k + (torch.rand(S, generator=gen, device=dev) < (want - k)).to(k.dtype)
    k = torch.minimum(k.long(), n_slots)
    k = torch.where(is_tgt, k, torch.zeros_like(k))       # refs are never touched

    # ---- which slots of the random-phase tiling are used ------------------
    # NS bounds floor(L/W) for every span: L <= T and W >= min(W_min, L).
    w_min = max(1, math.floor(cfg.dur_lo * FRAME_HZ))
    NS = max(1, T // w_min)
    slot = torch.arange(NS, device=dev)
    slot_ok = slot[None, :] < n_slots[:, None]            # (S, NS)
    key = torch.rand(S, NS, generator=gen, device=dev)
    key = torch.where(slot_ok, key, torch.full_like(key, 2.0))
    rank = torch.argsort(torch.argsort(key, dim=1, stable=True), dim=1)
    chosen = (rank < k[:, None]) & slot_ok
    phase = (torch.rand(S, generator=gen, device=dev)
             * (lens - n_slots * W + 1).to(torch.float32)).long()
    start = starts[:, None] + phase[:, None] + slot[None, :] * W[:, None]

    # ---- kind per window + donor choice -----------------------------------
    want_swap = torch.rand(S, NS, generator=gen, device=dev) < cfg.p_swap
    # donor eligibility is a per-SPAN property (W is per span): a donor must be
    # a different row (= different speaker group), the same language, and at
    # least as long as the window. Same q_use holds by construction — collate_v2
    # draws ONE q per BATCH (packing_v2.py:162).
    elig_row = ((lang[:, None] == lang[None, :])
                & (rows[:, None] != rows[None, :])
                & (lens[None, :] >= W[:, None]))          # (S_win, S_donor)
    if spk is None:
        same_spk = None
    else:
        known = spk >= 0
        # -1 is UNKNOWN, not "the unknown speaker": two unknown rows are never
        # certified as the same person, and (under the constraint) never as
        # different people either — an unknown row is simply not a donor.
        same_spk = (spk[:, None] == spk[None, :]) & known[:, None] & known[None, :]
    elig = elig_row
    if constrained:
        elig = elig_row & known[:, None] & known[None, :] & ~same_spk
    n_elig = elig.sum(1)
    has_donor = n_elig > 0
    is_swap = chosen & want_swap & has_donor[:, None]
    fallback = chosen & want_swap & ~has_donor[:, None]   # -> becomes noise
    is_noise = chosen & ~is_swap
    # pick the p-th eligible donor without materialising the eligible list:
    # cumsum jumps by 1 exactly at eligible columns, so the first column whose
    # cumsum reaches p+1 IS that donor.
    pick = (torch.rand(S, NS, generator=gen, device=dev)
            * n_elig[:, None].to(torch.float32)).long()
    pick = torch.minimum(pick, (n_elig - 1).clamp(min=0)[:, None])
    donor = torch.searchsorted(elig.long().cumsum(1).contiguous(),
                               (pick + 1).contiguous()).clamp(max=S - 1)
    d_room = (lens[donor] - W[:, None] + 1).clamp(min=1).to(torch.float32)
    d_off = (torch.rand(S, NS, generator=gen, device=dev) * d_room).long()
    d_off = torch.minimum(d_off, (lens[donor] - W[:, None]).clamp(min=0))

    # ---- expand windows to frames (fixed shape, invalid entries parked) ----
    Wm = max(1, min(math.ceil(cfg.dur_hi * FRAME_HZ), T))
    off = torch.arange(Wm, device=dev)
    inwin = off[None, None, :] < W[:, None, None]         # (S, NS, Wm)
    dst_row = rows[:, None, None].expand(S, NS, Wm)
    dst_col = start[:, :, None] + off
    src_row = rows[donor][:, :, None].expand(S, NS, Wm)
    src_col = (starts[donor] + d_off)[:, :, None] + off
    # column T is a SCRATCH column: every masked-off entry is written there and
    # the helper planes are sliced back to T before use. This is what keeps the
    # shapes static — a boolean select would need its size on the host.
    swap_at = torch.where(inwin & is_swap[:, :, None], dst_col, T)
    noise_at = torch.where(inwin & is_noise[:, :, None], dst_col, T)

    m_swap = torch.zeros(B, T + 1, dtype=torch.bool, device=dev)
    m_swap[dst_row, swap_at] = True
    gr = torch.arange(B, device=dev)[:, None].expand(B, T + 1).clone()
    gc = torch.arange(T + 1, device=dev)[None, :].expand(B, T + 1).clone()
    gr[dst_row, swap_at] = src_row                        # identity elsewhere
    gc[dst_row, swap_at] = src_col
    m_noise = torch.zeros(B, T + 1, dtype=torch.bool, device=dev)
    m_noise[dst_row, noise_at] = True

    m_swap, m_noise = m_swap[:, :T], m_noise[:, :T]
    gathered = codes[gr[:, :T], gc[:, :T].clamp(max=T - 1)]
    out = torch.where(m_swap[:, :, None] & in_q, gathered, codes)
    seg = out[..., cfg.first_level:]
    noise = torch.randint(0, N_CODES, seg.shape, generator=gen,
                          dtype=codes.dtype, device=dev)
    out[..., cfg.first_level:] = torch.where(
        m_noise[:, :, None] & noise_lvl, noise, seg)

    # ---- frame-weighted telemetry (device tensors; see the docstring) ------
    wf = W[:, None]
    swap_frames = (is_swap * wf).sum()
    noise_frames_raw = (is_noise * wf).sum()
    live = noise_lvl.any().long()          # 0 iff q_use <= first_level: the
    noise_frames = noise_frames_raw * live  # window would write nothing
    # ---- donor-speaker bookkeeping (device only, no RNG: the OFF path is
    # bit-identical to the unconstrained module because nothing here is drawn) ------
    zero = swap_frames * 0
    if same_spk is None:
        swap_same = swap_same_exp = zero
    else:
        # REALISED: the donor actually used, per window
        swap_same = ((is_swap & torch.gather(same_spk, 1, donor)) * wf).sum()
        # COUNTERFACTUAL for the unconstrained rule: donors are uniform over the eligible
        # set, so E[same-speaker frames] = W * n_same_eligible / n_eligible.
        # Computed from `elig_row` (the UNCONSTRAINED matrix) so it is the same
        # number whether or not the constraint is on. Rounded once per batch to
        # keep `counts` a single int64 vector the caller can accumulate.
        n_row = elig_row.sum(1)
        p_same = ((elig_row & same_spk).sum(1).double()
                  / n_row.clamp(min=1).double())
        swap_same_exp = (((is_swap * wf).double() * p_same[:, None]).sum()
                         .round().long())
    counts = torch.stack([
        (lens * is_tgt).sum(),                      # target_frames
        swap_frames + noise_frames,                 # corrupted_frames
        swap_frames,                                # swap_frames
        noise_frames,                               # noise_frames
        noise_frames_raw * (1 - live),              # noise_noop_frames
        chosen.sum(),                               # windows
        is_swap.sum(),                              # swap_draws
        is_noise.sum(),                             # noise_draws
        fallback.sum(),                             # donor_fallbacks
        (fallback * wf).sum(),                      # swap_fallback_frames
        swap_same,                                  # swap_same_spk_frames
        swap_same_exp,                     # swap_same_spk_expected_frames
        zero if constrained else swap_frames,  # swap_unconstrained_frames
    ]).long()
    stats = _pack_stats(frac, counts, dev)
    if trace:
        stats["trace"] = _trace(chosen, is_swap, rows, start, W, donor,
                                starts[donor] + d_off)
    return out, stats


def _trace(chosen, is_swap, rows, start, W, donor, d_start):
    """Per-window records for the fixed-seed determinism test. TEST ONLY: this
    is the one path that copies decisions back to the host."""
    ch = chosen.tolist()
    sw = is_swap.tolist()
    rw, st, wl = rows.tolist(), start.tolist(), W.tolist()
    dn, ds = donor.tolist(), d_start.tolist()
    recs = []
    for j, row in enumerate(ch):
        for i, on in enumerate(row):
            if not on:
                continue
            swap = sw[j][i]
            recs.append({"span": j, "row": rw[j], "start": st[j][i],
                         "frames": wl[j], "kind": "swap" if swap else "noise",
                         "donor_span": dn[j][i] if swap else None,
                         "donor_start": ds[j][i] if swap else None})
    return recs
