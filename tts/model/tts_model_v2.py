"""S2 assembly: backbone + DepthV2 (speaker memory lives in the depth stack).

Differences from v1 (tts_model.py):
- no SpeakerConditioner, no KIND_SPK slots (packer v2 never emits them — a
  slot reaching compose is a data bug and raises);
- no EmotionHead (emotion arm deferred; <|emo|> stays reserved in the vocab);
- depth is DepthV2: speaker refs enter as cross-attention K/V memory INSIDE
  depth.forward (never via a bare method call — FSDP2 all-gather hooks only
  wrap Module.__call__, the audio_io lesson);
- scheduled roll-in: batch["rollin_p"] converts that fraction of the depth
  frames to stop-gradient sampled prefixes (exposure to own errors).

S3 additions — ALL DEFAULT OFF and
bit-identical to S2 when off:
- `speaker_anchor`: zero-init MLP added to every span's <|audio|> sentinel
  embedding, so the BACKBONE finally knows who is speaking;
- `history_corruption`: TARGET-only corruption of the codes that feed the
  backbone as history, labels untouched (against copying the
  history's acoustic condition);
- `depth_grad_scale`: scales the acoustic gradient flowing from depth into the
  backbone (forward is an identity for any g);
- `rollin_include_cb0`: roll-in frames read a sampled cb0 from the backbone's
  own logits, matching serving;
- `chunked_lm_head`: the 194,548-way text CE is computed in row chunks under
  activation checkpointing instead of materialising ~30 GiB of logits.

Batch contract (packing_v2.collate_v2): tokens, kind (0=text 1=audio 3=aeos),
codes, loss_mask, attention_mask, spk_refs (B, R, 192), spk_null (B,), q_use,
optional rollin_p / sel_generator / anchor_generator / arm_id; S3 span geometry
span_batch_row / span_start / span_len / span_sentinel / span_is_target /
frame_* masks.

RNG: depth-frame SELECTION and anchor DROPOUT are two
independent stochastic streams. `forward(sel_generator=..., anchor_generator=
...)` (or the same keys on the batch, which the eval scripts use) supplies a
tagged torch.Generator for each; scripts/s3_train.py:tagged_generator() derives
them from (run.seed, tag, rank, update). With them supplied, an arm that has an
anchor and an arm that does not select the SAME frames of the same batch at the
same update, because neither draw walks the global torch stream. With them
absent both fall back to the global stream and the anchor's dropout draw shifts
selection — the S2 behaviour, kept only so old call sites still run.
"""
import inspect

import torch
from torch import nn

from ..tokens import AEOS_CLASS
from .anchors import D_SPK_ECAPA, SpeakerAnchor
from .audio_io import AudioInputEmbed
from .backbone import Backbone
from .chunked_ce import chunked_ce_sum
from .corrupt import CorruptConfig, corrupt_history
from .depth_v2 import DepthV2

KIND_TEXT, KIND_AUDIO, KIND_SPK, KIND_AEOS = 0, 1, 2, 3


class TTSModelV2(nn.Module):
    def __init__(self, backbone="qwen", attn="sdpa", amortize_frac=0.125,
                 speaker_anchor=None, history_corruption=None,
                 depth_grad_scale=1.0, depth_grouped=False,
                 chunked_lm_head=0, depth_ckpt_from=None):
        """speaker_anchor / history_corruption are the config dicts of
        configs/s3_path_a.yaml (None or {"enabled": false} = OFF):
          speaker_anchor: {enabled, input_dim, dropout, hidden}
          history_corruption: {enabled, span_swap: {weight, duration_s:[lo,hi]},
                               level_noise: {weight, first_level}}

        depth_grouped selects DepthV2's grouped teacher-forced forward.
        It is a CONSTRUCTOR decision, never a per-call one: the flag has to
        be identical on every rank for the whole run, and `forward` must not be
        able to flip an FSDP2 collective pattern per batch. Default False =
        the level-by-level loop = bitwise S2.

        chunked_lm_head (config key `model.chunked_lm_head`, rows per chunk)
        is a MEMORY knob with no numeric intent: 0 = off = the unchunked path, which
        materialises the whole (positions, 194,548) logits tensor, bitwise
        unchanged. Any positive value computes the same fp32 CE sum in chunks
        under checkpointing (tts/model/chunked_ce.py). Like
        depth_grouped it is a CONSTRUCTOR decision so no batch can flip the
        checkpoint/collective pattern per rank.

        depth_ckpt_from is DepthV2's selective activation-checkpointing
        threshold (config `model.depth_ckpt_from`, 1..31). None = the mode
        default: 9 on the loop path (S2 bitwise), 1 when depth_grouped is on,
        where checkpointing every group pass is the activation-memory
        lever. A constructor decision for the same FSDP2 reason.
        """
        super().__init__()
        self.backbone = Backbone(backbone, attn=attn)
        d = self.backbone.d_model
        self.depth = DepthV2(d_backbone=d, grouped=depth_grouped,
                             ckpt_from=depth_ckpt_from)
        self.audio_in = AudioInputEmbed(d)          # root-owned: FSDP-safe
        self._configure(d, amortize_frac, speaker_anchor, history_corruption,
                        depth_grad_scale, chunked_lm_head)

    def _configure(self, d_model, amortize_frac=0.125, speaker_anchor=None,
                   history_corruption=None, depth_grad_scale=1.0,
                   chunked_lm_head=0):
        """Config-only half of __init__. Split out so tests can drive the REAL
        compose_inputs/forward with stand-in backbone/depth submodules — the
        real ones need the Qwen weights and 555M depth params, which no CPU
        test can afford."""
        self.amortize_frac = amortize_frac
        self.depth_grad_scale = float(depth_grad_scale)
        # plain attributes, never buffers: properties of the RUN, not of the
        # weights, so the state_dict / checkpoint contract is unchanged
        self.chunked_lm_head = int(chunked_lm_head or 0)
        if self.chunked_lm_head < 0:
            raise ValueError(f"model.chunked_lm_head must be >= 0 (0 = off), "
                             f"got {chunked_lm_head}")
        self.speaker_anchor = None
        if speaker_anchor and speaker_anchor.get("enabled", False):
            self.speaker_anchor = SpeakerAnchor(
                d_model=d_model,
                input_dim=int(speaker_anchor.get("input_dim", D_SPK_ECAPA)),
                hidden=int(speaker_anchor.get("hidden", 512)),
                dropout_p=float(speaker_anchor.get("dropout", 0.20)))
        self.corrupt_cfg = CorruptConfig.from_dict(history_corruption)
        # rollin_cb0/rollin_temp/rollin_top_k are passed to DepthV2.forward
        # only if its signature has them, so an older depth module degrades
        # to the plain roll-in instead of raising.
        self._depth_kwargs = frozenset(
            inspect.signature(self.depth.forward).parameters)

    def compose_inputs(self, batch, codes=None, anchor_gen=None):
        """`codes` overrides batch["codes"] on the INPUT path only (S3 history
        corruption); every LABEL keeps reading batch["codes"]. Default None =
        exactly the S2 path. `anchor_gen` is the tagged generator for the
        anchor's row dropout (falls back to batch["anchor_generator"])."""
        if bool((batch["kind"] == KIND_SPK).any()):
            raise ValueError("KIND_SPK reached the v2 model — v1 packer output?")
        tok_emb = self.backbone.embed_text(batch["tokens"])
        q_use = int(batch["q_use"])
        codes = batch["codes"] if codes is None else codes
        audio = batch["kind"] == KIND_AUDIO
        if audio.any():
            fb = self.audio_in.frames(codes[audio], q_use)
            tok_emb = tok_emb.masked_scatter(audio[..., None], fb.to(tok_emb.dtype))
        aeos = batch["kind"] == KIND_AEOS
        if aeos.any():
            tok_emb = tok_emb.masked_scatter(
                aeos[..., None], self.audio_in.aeos(int(aeos.sum())).to(tok_emb.dtype))
        if self.speaker_anchor is not None:
            tok_emb = self._add_speaker_anchor(batch, tok_emb, anchor_gen)
        return tok_emb

    def _add_speaker_anchor(self, batch, tok_emb, gen=None):
        """Add the row's speaker vector at EVERY span's <|audio|>
        sentinel (packing_v2 span geometry). One vector per ROW — the batch
        carries one spk_refs block per row, not per span."""
        for k in ("span_batch_row", "span_sentinel"):
            if k not in batch:
                raise ValueError(f"speaker_anchor needs collate_v2's {k} "
                                 "(see packing_v2._add_span_geometry)")
        a = self.speaker_anchor(batch["spk_refs"],
                                ref_mask=batch.get("spk_ref_mask"),
                                null_mask=batch.get("spk_null"),
                                gen=(gen if gen is not None
                                     else batch.get("anchor_generator")))
        rows, cols = batch["span_batch_row"], batch["span_sentinel"]
        if rows.numel():
            tok_emb = tok_emb.index_put((rows, cols), a[rows].to(tok_emb.dtype),
                                        accumulate=True)
        # gradient symmetry: exactly-zero term so the
        # anchor params are in the graph even on a batch with no spans / all
        # rows null — added unconditionally, never inside the `if`.
        return tok_emb + self.speaker_anchor.sym_term().to(tok_emb.dtype)

    def _rollin_cb0(self, h_sel, true_cb0, rmask, temp, top_k):
        """Roll-in cb0: roll-in frames feed DEPTH a cb0 SAMPLED from
        the backbone's own logits for that frame instead of the true one, so
        the depth stack sees the cb0 distribution it meets at serving (S2
        roll-in never sampled cb0). Stop-gradient throughout; frames
        outside the roll-in mask keep the TRUE cb0, so the returned (N,) tensor
        is safe whether depth substitutes everywhere or only on masked rows.

        SUPPORT == SERVING'S. scripts/
        s2_serve_controls.py:283-289 divides the FULL (N_CODES+1)-way logits by
        the temperature and takes top-k over ALL classes INCLUDING AEOS; only
        then, when it decides to continue, does it zero AEOS out of that
        distribution and renormalise (s2_serve_controls.py:330-333). Dropping
        AEOS BEFORE the top-k (the first version of this method) admits the
        65th-ranked class whenever AEOS is inside the top 64, so training would
        sample from a support serving never uses. Masking the AEOS LOGIT to
        -inf after the top-k is exactly serving's "q[AEOS]=0; q/=q.sum()".

        `h_sel` must be the RAW backbone hidden state, not an fp32 copy: under
        FSDP2 mixed precision `cb0_head` is bf16, and serving likewise
        feeds `cb0_logits` the model-dtype hidden state (s2_serve_controls.py:
        283) while passing `h.float()` to depth (:335)."""
        neg_inf = -float("inf")
        with torch.no_grad():
            lg = self.backbone.cb0_logits(h_sel.detach()).float()
            if temp != 1.0:
                lg = lg / max(float(temp), 1e-6)
            k = int(top_k)
            if k:
                kth = lg.topk(min(k, lg.shape[-1]), -1).values[:, -1:]
                sup = lg.masked_fill(lg < kth, neg_inf)      # fresh tensor
            else:
                sup = lg.clone()                             # never alias `lg`
            sup[:, AEOS_CLASS] = neg_inf
            if k == 1:
                # top_k == 1 with AEOS as the argmax leaves an all -inf row
                # (NaN in softmax). Serving cannot reach it — the controlled
                # stop breaks on `p.argmax() == AEOS_CLASS` first
                # (s2_serve_controls.py:321) — but training has no stop, so fall
                # back to the un-top-k'd, AEOS-free support. k >= 2 keeps at
                # least two classes, so at most one of them can be AEOS.
                lg[:, AEOS_CLASS] = neg_inf
                sup = torch.where(torch.isfinite(sup).any(-1, keepdim=True),
                                  sup, lg)
            samp = torch.multinomial(sup.softmax(-1), 1).squeeze(-1)
        return torch.where(rmask, samp, true_cb0)

    def forward(self, batch, corrupt_frac=0.0, corrupt_gen=None,
                rollin_temp=1.0, rollin_top_k=0, rollin_include_cb0=False,
                sel_generator=None, anchor_generator=None):
        """corrupt_frac: current history-corruption ramp value (0..0.30), only
        meaningful when the model was built with history_corruption enabled;
        corrupt_gen: the loop's seeded torch.Generator. rollin_* configure the
        within-frame roll-in sampling. sel_generator /
        anchor_generator are the tagged per-update streams for depth-frame
        selection and anchor dropout (module docstring); each
        falls back to the same-named batch key and then to the global torch
        stream. S2 call sites — model(batch) — are unaffected: every S3 knob
        defaults to the S2 behaviour."""
        codes_in = batch["codes"]
        if self.corrupt_cfg is not None and float(corrupt_frac) > 0.0:
            codes_in, cstats = corrupt_history(batch, self.corrupt_cfg,
                                               float(corrupt_frac), corrupt_gen)
        else:
            cstats = None
        x = self.compose_inputs(batch, codes=codes_in,
                                anchor_gen=anchor_generator)
        hidden, _ = self.backbone(x, attention_mask=batch.get("attention_mask"))
        out = {"hidden": hidden}
        if cstats is not None:
            out["corrupt_stats"] = cstats            # telemetry for the loop
        nk = batch["kind"][:, 1:]
        lm = batch["loss_mask"][:, 1:]
        h = hidden[:, :-1]
        pred_text = (nk == KIND_TEXT) & lm
        pred_cb0 = ((nk == KIND_AUDIO) | (nk == KIND_AEOS)) & lm
        pred_depth = (nk == KIND_AUDIO) & lm
        if not batch.get("depth_on_refs", True) and "target_start" in batch:
            # exclude in-context REFERENCE frames from the depth loss: the
            # row's speaker memory holds those utterances' own embeddings,
            # so predicting them is a self-conditioning shortcut.
            # cb0 loss on refs is unaffected.
            pos = torch.arange(1, nk.shape[1] + 1, device=nk.device)[None, :]
            pred_depth = pred_depth & (pos >= batch["target_start"][:, None])
        # the 194,548-way text head is skipped when this arm's text loss
        # weight is 0 (audio arms) — the flag is a rank-lockstep Python bool
        # set by the loop from the arm; the tied matrix still trains as the
        # input embedding on those arms
        if pred_text.any() and batch.get("compute_text_logits", True):
            ht = h[pred_text]
            tt = batch["tokens"][:, 1:][pred_text]
            if self.chunked_lm_head:
                # ~30 GiB of transient 194,548-way logits on the replay
                # arm, ~11.9 GiB of it retained until backward. The chunked
                # form computes the SAME fp32 CE sum tts/train/losses.py:93-96
                # would (sum of per-chunk sums) and hands it over ready-made;
                # `text_targets` still travels, so the denominator, telemetry
                # and every other consumer are unchanged.
                out["text_loss_sum"] = chunked_ce_sum(
                    self.backbone.text_logits, ht, tt, self.chunked_lm_head)
            else:
                # key insertion order kept as it was, so `out` is identical
                out["text_logits"] = self.backbone.text_logits(ht)
            out["text_targets"] = tt
        if pred_cb0.any():
            out["cb0_logits"] = self.backbone.cb0_logits(h[pred_cb0])
            out["cb0_targets"] = batch["codes"][:, 1:, 0][pred_cb0]
        if pred_depth.any():
            q_use = int(batch["q_use"])
            tgt = batch["codes"][:, 1:][pred_depth]          # (N, n_q) ORIGINAL
            row_of = torch.nonzero(pred_depth)[:, 0]         # frame -> batch row
            n = tgt.shape[0]
            k = max(1, int(n * self.amortize_frac))
            gen = (sel_generator if sel_generator is not None
                   else batch.get("sel_generator"))
            sel = torch.randperm(n, device=tgt.device, generator=gen)[:k]
            rows = row_of[sel]
            p = float(batch.get("rollin_p", 0.0))
            rmask = (torch.rand(k, device=tgt.device) < p) if p > 0 else None
            # the REALIZED roll-in count, not the
            # expected p*k the telemetry used to record. A device scalar that
            # is never read here — s3_train sums it ON DEVICE and reads it once
            # per log window — so this costs one reduction and no sync, and it
            # is the count the depth FLOP model must be charged for (rolled
            # rows take _forward_split's sequential path).
            out["rollin_count"] = (
                rmask.sum() if rmask is not None
                else torch.zeros((), dtype=torch.long, device=tgt.device))
            # RAW backbone dtype for the cb0 head (bf16 under FSDP2 mixed
            # precision: an fp32 hidden state into
            # the root-owned bf16 cb0_head aborts the forward), fp32 for depth.
            # Serving splits the same way: cb0_logits(h) vs depth h.float()
            # (s2_serve_controls.py:283 / :335).
            h_raw = h[pred_depth][sel]
            kw = {}
            if rollin_include_cb0 and rmask is not None \
                    and "rollin_cb0" in self._depth_kwargs:
                kw["rollin_cb0"] = self._rollin_cb0(
                    h_raw, tgt[sel][:, 0], rmask, rollin_temp, rollin_top_k)
            h_sel = h_raw.float()
            for name, val in (("rollin_temp", rollin_temp),
                              ("rollin_top_k", rollin_top_k)):
                if name in self._depth_kwargs:
                    kw[name] = val
            g = self.depth_grad_scale
            if g != 1.0:
                # depth_grad_scale: forward is an exact identity for any g; the
                # gradient reaching the BACKBONE through depth is scaled by g,
                # depth's own parameter gradients are untouched.
                h_sel = h_sel.detach() + g * (h_sel - h_sel.detach())
            out["depth_logits"] = self.depth(
                h_sel, tgt[sel],
                refs=batch["spk_refs"][rows],
                null_mask=batch["spk_null"][rows],
                ref_mask=batch["spk_ref_mask"][rows],
                q_eff=q_use, rollin_mask=rmask, **kw)
            out["depth_targets"] = tgt[sel][:, 1:q_use]
        return out
