"""S2 packing — conditioning contract.

One packed row = ONE same-speaker group only (the S1 mode-A fix: unrelated
cuts NEVER share a row, so no attention reset machinery is needed). Four
modes decide the two conditioning channels independently:

  icl_spk    (30%): ref utterances in context  + ECAPA depth memory
  icl_only   ( 5%): ref utterances in context  + null memory
  spk_only   (55%): no refs in context         + ECAPA depth memory
  joint_null (10%): no refs in context         + null memory  (CFG negative;
              matched dropout — same target units as the conditioned arms)

Layout (lang tag once, on the first span; no <|spk|> slots, no emo slot):
  [ref: <|lang|> text <|audio|> frames AEOS] x R   (icl modes; cont forced off)
  [target: (<|cont|>) (<|lang|> if first span) text <|audio|> frames AEOS] x N

Ref frames carry loss (real same-speaker audio; they also feed the ledger).
Throughput comes from token-budget BATCHING of short single-group rows
(TokenBudgetBatcher), not from co-packing strangers.

Span geometry (additive, never changes an existing tensor): collate_v2 also emits
the SPAN GEOMETRY the S3 model needs — where each span's <|audio|> sentinel
sits (the speaker anchor is added there, no new token/position/kind) and which
audio frames are TARGET vs in-context REFERENCE (history corruption touches
target frames only). See `_add_span_geometry`.
"""
import numpy as np
import torch

from ..tokens import AEOS_CLASS, CONTRACT, N_Q

KIND_TEXT, KIND_AUDIO, KIND_AEOS = 0, 1, 3
MODES2 = ("icl_spk", "icl_only", "spk_only", "joint_null")
D_SPK = 192
R_MAX = 3


class BadUnit(Exception):
    """An ENUMERATED data reject: the only exception
    class a loader worker may skip. Everything else is a defect and fails loud."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class PackerV2:
    def __init__(self, tokenizer, token_ids, max_tokens=8192, rng=None):
        self.tok = tokenizer
        self.ids = token_ids
        self.max_tokens = max_tokens
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _span(self, cut, first, cont):
        """Numpy span builder: the original per-token python lists
        ([[-1]*32] per text token, codes.tolist()) dominated the GIL-bound
        loader path. Output is byte-identical to the list version (verified
        by an identity test against it)."""
        head = []
        untrained = 0
        if cont:
            head.append(self.ids[CONTRACT.cont]); untrained += 1
        if first:
            head.append(self.ids[f"<|lang:{cut['lang_final']}|>"])
            untrained += 1
        text = cut["text_final"]
        if not isinstance(text, str) or not text.strip():
            raise BadUnit("empty_text")             # explicit bad-text condition
        try:
            head += self.tok(text, add_special_tokens=False)["input_ids"]
        except UnicodeError as e:                   # the ONLY tokenizer reject
            raise BadUnit("unicode") from e         # treated as bad data; any
                                                    # other failure propagates
        head.append(self.ids[CONTRACT.audio])
        n_text = len(head)
        codes = cut["codes"]                      # (n_q, F) reader-validated
        F_ = codes.shape[1]
        L = n_text + F_ + 1
        t = np.zeros(L, np.int64); t[:n_text] = head
        k = np.zeros(L, np.int8)
        k[n_text:n_text + F_] = KIND_AUDIO; k[-1] = KIND_AEOS
        c = np.full((L, N_Q), -1, np.int64)
        c[n_text:n_text + F_] = codes.T
        c[-1, 0] = AEOS_CLASS
        m = np.ones(L, bool); m[:untrained] = False
        return t, k, c, m

    def build(self, unit, refs, mode):
        """unit: target cut dicts (a chain or a single cut); refs: dict from
        ShardSourceV2.refs_for or None; mode: one of MODES2. Returns example
        dict or None (nothing packed)."""
        use_icl = mode in ("icl_spk", "icl_only")
        use_mem = mode in ("icl_spk", "spk_only") and refs is not None
        # one language per packed row: the single lang
        # tag must name every span in the row
        lang = unit[0]["lang_final"]
        assert all(c["lang_final"] == lang for c in unit), "mixed-lang unit"
        # speaker identity: a packed row is ONE (video, speaker)
        # group by construction (`ShardSourceV2._units` iterates groups), so a
        # unit that is language-homogeneous must be speaker-homogeneous too.
        # Assert it rather than trust it: history corruption picks swap donors
        # by this identity, and a mixed-speaker row would make the donor rule
        # silently wrong. None (a hand-built unit with no identity) is legal
        # and collates to speaker_id = -1 — see collate_v2.
        speaker_key = unit[0].get("speaker_key")
        assert all(c.get("speaker_key") == speaker_key for c in unit), \
            "mixed-speaker unit"
        if use_icl:
            assert all(rc["lang_final"] == lang for rc in refs["cuts"]), \
                "ref language differs from target language"
        spans, ref_rows, total = [], [], 0
        if use_icl:
            for i, rc in enumerate(refs["cuts"]):
                s = self._span(rc, first=(i == 0), cont=False)
                if total + len(s[0]) > self.max_tokens:
                    break
                spans.append(s); total += len(s[0])
                ref_rows.append(refs["rows"][i])
        target_start = total                # first token of the target spans
        packed_rows, packed = [], 0
        for j, cut in enumerate(unit):
            # <|cont|> only when the actual predecessor is IN the row:
            # unit cuts after the first carry their predecessor by construction
            s = self._span(cut, first=(total == 0),
                           cont=(j > 0 and cut.get("cont", False)))
            if total + len(s[0]) > self.max_tokens:
                break                              # whole-span preflight
            spans.append(s); total += len(s[0])
            packed_rows.append(cut["row"]); packed += 1
        if packed == 0:
            return None
        # examples carry NUMPY (not torch): they cross a multiprocessing
        # queue as plain pickled bytes — torch tensors would ship one shared-
        # memory file descriptor EACH and exhaust the fd limit
        if use_mem:
            embs = np.asarray(refs["embs"], dtype=np.float32)
        else:
            embs = np.zeros((1, D_SPK), np.float32)
        return {
            "tokens": np.concatenate([s[0] for s in spans]),
            "kind": np.concatenate([s[1] for s in spans]),
            "codes": np.concatenate([s[2] for s in spans]),
            "loss_mask": np.concatenate([s[3] for s in spans]),
            "spk_refs": embs,
            "spk_null": not use_mem,
            "mode": mode, "n_cuts": packed,
            "target_rows": packed_rows,
            "ref_rows": ref_rows if use_icl else [],
            "target_start": target_start,   # evals score the target span only;
                                            # training may exclude refs from
                                            # the depth loss
            "lang": lang,                   # trained-language telemetry
            # identity carried to collate_v2 -> batch["speaker_id"], the
            # donor constraint of tts/model/corrupt.py span_swap. None when the
            # source did not supply one.
            "speaker_key": speaker_key,
            # stable per-row ids of the TARGET cuts that actually fit, for the
            # paired-arm order digest
            "sample_keys": [c.get("sample_key") for c in unit[:packed]],
        }


def collate_v2(examples, q_choices=(8, 16, 24, 32), q_rng=None, pad_id=0):
    """Variable-B batch of one-group rows. q_rng MUST be the rank-shared
    lockstep generator (FSDP reduce-scatter shape contract — S1 lesson).
    Text-replay examples (no spk keys) collate as null-memory rows."""
    q_rng = q_rng if q_rng is not None else np.random.default_rng(0)
    B = len(examples)
    T = max(e["tokens"].shape[0] for e in examples)
    out = {
        "tokens": torch.full((B, T), pad_id, dtype=torch.long),
        "kind": torch.full((B, T), KIND_TEXT, dtype=torch.int8),
        "codes": torch.full((B, T, N_Q), -1, dtype=torch.long),
        "loss_mask": torch.zeros(B, T, dtype=torch.bool),
        "attention_mask": torch.zeros(B, T, dtype=torch.long),
        "spk_refs": torch.zeros(B, R_MAX, D_SPK),
        "spk_ref_mask": torch.zeros(B, R_MAX, dtype=torch.bool),
        "spk_null": torch.ones(B, dtype=torch.bool),
        "q_use": torch.tensor(int(q_rng.choice(q_choices))),
        "modes": [e.get("mode", "text") for e in examples],
        "langs": [e.get("lang", "text") for e in examples],
        # per-ROW speaker identity as a dense
        # BATCH-LOCAL id, -1 = unknown (text-replay rows, and any example whose
        # source did not carry a `speaker_key`). Only equality within the batch
        # matters — tts/model/corrupt.py uses it to keep a span_swap donor off
        # the row's OWN speaker, and a same-speaker donor leaves the history
        # acoustics intact, i.e. does not break the acoustic-copy shortcut at all.
        # -1 never matches -1 there: unknown is not "same speaker".
        "speaker_id": _speaker_ids(examples),
        # first TARGET position per row (0 for non-ICL rows and text rows)
        "target_start": torch.tensor([int(e.get("target_start", 0))
                                      for e in examples], dtype=torch.long),
        "credited_s": torch.tensor([float(e.get("credited_s", 0.0))
                                    for e in examples]),
    }
    for b, e in enumerate(examples):
        L = e["tokens"].shape[0]
        # examples may carry numpy (audio path, mp-queue-friendly) or torch
        # (text replay) — as_tensor handles both without copying torch inputs
        out["tokens"][b, :L] = torch.as_tensor(e["tokens"])
        out["kind"][b, :L] = torch.as_tensor(e["kind"])
        out["codes"][b, :L] = torch.as_tensor(e["codes"])
        out["loss_mask"][b, :L] = torch.as_tensor(e["loss_mask"])
        out["attention_mask"][b, :L] = 1
        r = e.get("spk_refs")
        if r is not None and not bool(e.get("spk_null", True)):
            R = r.shape[0]
            out["spk_refs"][b, :R] = torch.as_tensor(r)
            out["spk_ref_mask"][b, :R] = True      # zero-pad + mask: padding
            out["spk_null"][b] = False             # never absorbs attention
    _add_span_geometry(out)                        # additive span geometry
    return out


def _speaker_ids(examples):
    """(B,) long tensor of batch-local speaker ids; -1 where unknown.

    Ids are assigned in first-appearance order, so the tensor is a pure
    function of the example list (no RNG, no global state) and two ranks that
    see the same rows produce the same ids."""
    seen, ids = {}, []
    for e in examples:
        key = e.get("speaker_key")
        ids.append(-1 if key is None else seen.setdefault(key, len(seen)))
    return torch.tensor(ids, dtype=torch.long)


def _add_span_geometry(out):
    """Add per-span <|audio|> sentinel positions + per-position
    target/reference/row/span ids to an already-collated batch.

    WHY derived here instead of emitted by PackerV2.build: only the batch knows
    the padded (B, T) geometry, and text-replay examples never pass through
    build() — deriving from `kind` is ONE code path for every example source and
    leaves every pre-existing tensor byte-identical (proved by
    tests/test_anchor_corrupt.py::test_collate_byte_identical_to_baseline).

    Layout invariant relied on (`PackerV2._span` above): a span is
    [ ...text..., <|audio|>, F audio frames, AEOS ], so a MAXIMAL run of
    KIND_AUDIO positions is exactly one span's frames, and the position right
    before the run holds that span's <|audio|> sentinel. A frame is a TARGET
    frame iff its column >= the row's `target_start` — the same rule the model
    already uses to keep refs out of the depth loss (tts_model_v2.py:62-68).

    Added keys (S = number of audio spans in the batch, may be 0):
      span_batch_row   (S,)  long  batch row b of the span
      span_start       (S,)  long  column of the span's FIRST audio frame
      span_len         (S,)  long  number of audio frames in the span
      span_sentinel    (S,)  long  column of the span's <|audio|> token
      span_is_target   (S,)  bool  False = in-context reference span
      frame_span_id    (B,T) long  global span id at audio frames, -1 elsewhere
      frame_batch_row  (B,T) long  b at audio frames, -1 elsewhere
      frame_target_mask(B,T) bool  audio frame of a TARGET span
      frame_ref_mask   (B,T) bool  audio frame of a REFERENCE span
    """
    if "target_start" not in out:
        raise ValueError("span geometry needs `target_start` (the first TARGET "
                         "column of every row); collate_v2 always sets it, a "
                         "hand-built batch must too")
    aud = out["kind"] == KIND_AUDIO
    B, T = aud.shape
    dev = aud.device                     # every derived tensor stays on the
                                         # batch's device: serving builds its
                                         # prompt batches straight on the GPU
    prv = torch.zeros_like(aud); prv[:, 1:] = aud[:, :-1]
    nxt = torch.zeros_like(aud); nxt[:, :-1] = aud[:, 1:]
    starts, ends = aud & ~prv, aud & ~nxt
    rows_s, pos_s = starts.nonzero(as_tuple=True)      # row-major -> pairs 1:1
    rows_e, pos_e = ends.nonzero(as_tuple=True)        # with the run ends
    # the 1:1 pairing is what makes span_len correct for the SECOND and later
    # span of a row; assert it instead of trusting it silently (multi-span
    # rows are where a mispairing would go unnoticed)
    if not torch.equal(rows_s, rows_e):
        raise ValueError("audio run starts and ends do not pair up row-wise — "
                         "packer layout violated")
    if pos_s.numel() and int(pos_s.min()) == 0:
        raise ValueError("an audio run starts at column 0 — no <|audio|> "
                         "sentinel precedes it; packer layout violated")
    pos = torch.arange(T, device=dev)[None, :]
    is_tgt = pos >= out["target_start"].to(dev)[:, None]       # (B, T)
    # global span id at every audio position: count span starts up to here
    sid = (starts.reshape(-1).long().cumsum(0).view(B, T) - 1)
    rid = torch.arange(B, device=dev)[:, None].expand(B, T)
    out["span_batch_row"] = rows_s
    out["span_start"] = pos_s
    out["span_len"] = pos_e - pos_s + 1
    out["span_sentinel"] = pos_s - 1
    out["span_is_target"] = is_tgt[rows_s, pos_s]
    out["frame_span_id"] = torch.where(aud, sid, torch.full_like(sid, -1))
    out["frame_batch_row"] = torch.where(aud, rid, torch.full_like(rid, -1))
    out["frame_target_mask"] = aud & is_tgt
    out["frame_ref_mask"] = aud & ~is_tgt
    return out


class TokenBudgetBatcher:
    """Pool -> length-sort -> token-budget batches that cover EVERY pooled row
    -> shuffled emission. Padded size B*max_len stays under pack_tokens, B
    under max_rows.

    Replaces the reservoir that re-sorted after every refill and emitted only
    down to n//2 rows: on an infinite source that kept the ~n/2 shortest rows
    back until an even shorter row displaced them (credited-but-untrained
    sink) and emitted every cycle long->short.
    Here every pooled row is emitted before the next pool is read; batch
    order inside a pool is shuffled so consecutive updates are not
    length/mode-correlated. Within-batch padding is unchanged (batches are
    contiguous length ranges of a sorted pool)."""

    def __init__(self, gen, pack_tokens=13000, max_rows=48, reservoir=64,
                 stats=None, rng=None):
        self.gen = gen
        self.budget = pack_tokens
        self.max_rows = max_rows
        self.n = reservoir
        self.stats = stats if stats is not None else {}
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _batches(self, pool):
        pool.sort(key=lambda e: e["tokens"].shape[0], reverse=True)
        out, i = [], 0
        while i < len(pool):
            top = pool[i]["tokens"].shape[0]
            take = max(1, min(self.max_rows, len(pool) - i,
                              self.budget // max(1, top)))
            out.append(pool[i:i + take])
            i += take
        self.rng.shuffle(out)                 # in-place list shuffle
        self.stats["pools"] = self.stats.get("pools", 0) + 1
        self.stats["rows"] = self.stats.get("rows", 0) + len(pool)
        return out

    def __iter__(self):
        pool = []
        for ex in self.gen:
            pool.append(ex)
            if len(pool) >= self.n:
                for b in self._batches(pool):
                    yield b
                pool = []
        if pool:                              # finite source (tests): drain
            for b in self._batches(pool):
                yield b
