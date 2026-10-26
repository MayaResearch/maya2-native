"""The backbone-side SPEAKER ANCHOR.

WHY: in S2 the speaker reaches the DEPTH stack only, so the backbone predicts
cb0 with no idea who is speaking — measured, spk_only cb0 NLL equals joint_null
exactly. This module injects one speaker vector into the
BACKBONE at a position that already exists: the span's `<|audio|>` sentinel.
No new token, no new position, no KIND_SPK — the frozen
50-token contract and the packed geometry are untouched.

WHY the final layer is zero-initialised: S3 warm-starts from a trained S2
checkpoint. A zero output layer makes the step-0
forward bit-identical to S2's, so the warm restart cannot damage the model
before the anchor has learned anything.

WHY a learned null INPUT (not a zero input): rows with no usable speaker
embedding (joint_null / icl_only / text replay) and rows whose anchor is
dropped must still traverse the same MLP, so the null case is a point the
network can place anywhere in input space instead of a hard-coded origin —
the same trick DepthV2 uses for its speaker memory (depth_v2.py:103).
"""
import torch
from torch import nn

D_SPK_ECAPA = 192          # ECAPA embedding width carried by collate_v2


class SpeakerAnchor(nn.Module):
    """2-layer MLP: mean-pooled ECAPA-192 -> d_model, ZERO-INIT output layer.

    `d_model` is first because it has no sensible default (the natural spelling
    `SpeakerAnchor(input_dim=192, d_model, ...)` is not valid Python);
    every caller passes keywords, so the names are what matter.

    dropout_p is the PER-ROW anchor dropout probability (20% by default,
    independent of the depth memory's dropout) — it is deliberately NOT an
    nn.Dropout inside the MLP: dropping the whole anchor for a row is what
    teaches the backbone to work without one, and it must stay independent of
    DepthV2's null_mask so the two conditioning channels can be ablated apart.
    """

    def __init__(self, d_model, input_dim=D_SPK_ECAPA, hidden=512,
                 dropout_p=0.20):
        super().__init__()
        self.input_dim, self.dropout_p = int(input_dim), float(dropout_p)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.null_vec = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.null_vec, std=0.02)
        nn.init.zeros_(self.fc2.weight)          # zero-init OUTPUT layer:
        nn.init.zeros_(self.fc2.bias)            # step-0 forward == S2's

    def pool(self, refs, ref_mask=None):
        """Mean-pool a row's reference embeddings -> (B, input_dim), in fp32.

        The batch carries ONE `spk_refs` block per ROW, not per span
        (packing_v2.py:120-128 stores `refs["embs"]` on the example; collate_v2
        pads it to (B, R_MAX, 192) with `spk_ref_mask`), so every span of a row
        receives the SAME anchor vector. Padding rows are excluded by the mask
        instead of being averaged in as zeros.

        The POOLING ARITHMETIC stays fp32 (a masked mean of up to 3 vectors is
        cheap and bf16 would round the divisor); `forward` casts the result to
        the LINEAR's parameter dtype before fc1 — see the dtype note there.
        """
        refs = refs[..., :self.input_dim].float()
        if ref_mask is None:
            return refs.mean(1)
        m = ref_mask[..., None].to(refs.dtype)
        return (refs * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, refs, ref_mask=None, null_mask=None, gen=None):
        """refs (B, R, >=input_dim); ref_mask (B, R) bool True = real ref;
        null_mask (B,) bool True = row has no speaker embedding. Returns
        (B, d_model), zero at init.

        `gen` is the optional torch.Generator for the row-dropout draw, and it
        is not a convenience: without it the draw comes out
        of the GLOBAL torch stream, and because it happens inside
        compose_inputs — i.e. BEFORE TTSModelV2.forward's depth-frame randperm
        — an arm with an anchor selects different frames from an arm without
        one on the very same batch. The driver passes a tagged per-rank,
        per-update generator (scripts/s3_train.py:tagged_generator, TAG_ANCHOR)
        so the two streams are independent and the causal arms stay paired.
        Must live on the same device as the draw; `u` is moved to the pooled
        device afterwards so a CPU generator still works in tests."""
        pooled = self.pool(refs, ref_mask)
        B = pooled.shape[0]
        use_null = torch.zeros(B, dtype=torch.bool, device=pooled.device)
        if null_mask is not None:
            use_null |= null_mask.to(pooled.device)
        if ref_mask is not None:
            use_null |= ref_mask.sum(1) == 0          # nothing to pool
        if self.training and self.dropout_p > 0:
            # per-ROW dropout; ranks may drop different rows (that is fine —
            # the parameter SET touched per step is what FSDP2 needs to match,
            # and sym_term() below guarantees it)
            u = torch.rand(B, generator=gen,
                           device=gen.device if gen is not None else pooled.device)
            use_null |= (u.to(pooled.device) < self.dropout_p)
        x = torch.where(use_null[:, None], self.null_vec.to(pooled.dtype)[None, :],
                        pooled)
        # DTYPE: under FSDP2 mixed precision the
        # root-owned parameters are all-gathered as bf16 while `spk_refs` stays
        # fp32, so an fp32 `x` into a bf16 fc1 aborts the very first forward
        # with "mat1 and mat2 must have the same dtype". Take the dtype FROM the
        # weight (not from a config flag) so the module is correct in fp32 CPU
        # tests, in bf16 training and in fp16 alike.
        x = x.to(self.fc1.weight.dtype)
        return self.fc2(nn.functional.gelu(self.fc1(x)))

    def sym_term(self):
        """FSDP2 gradient symmetry (the same anchor
        DepthV2.forward:191-197 carries): a parameter whose use is
        data-dependent must be pulled into the graph on EVERY step of EVERY
        rank or the reduce-scatter sets diverge. `null_vec` is used only when
        some row is null/dropped, and NO anchor parameter is used at all on a
        batch with no audio spans (text replay) — so the caller adds this
        exactly-zero scalar to its output unconditionally."""
        return 0.0 * sum(p.sum() for p in self.parameters())
