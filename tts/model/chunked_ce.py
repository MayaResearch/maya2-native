"""Chunked, recompute-on-backward cross-entropy for the LM head.

THE PROBLEM. The text
replay arm (10% of updates, `data.text_rows` x `data.max_tokens`) sends every
text position through the backbone's 194,548-way LM head at once:
tts/model/tts_model_v2.py materialises `out["text_logits"]` and
tts/train/losses.py then calls `F.cross_entropy(logits.float(), ...)` on it. At
~16,384 positions that is bf16 logits 5.9 GiB + the `.float()` copy 11.9 GiB +
cross_entropy's internal log-softmax 11.9 GiB ~= 30 GiB of transient, of which
the ~11.9 GiB log-softmax is RETAINED until backward. This is very probably
the training memory peak, and therefore the cheapest way to buy 1/4 depth
coverage, larger packs, and selective backbone checkpointing.

THE FIX. Process the positions in row chunks: for each chunk compute its logits,
take the fp32 CE SUM, add it to a running fp32 total, and wrap the chunk in
non-reentrant `torch.utils.checkpoint` so nothing but the (chunk, d) hidden
slice — which is a VIEW of a tensor that was already alive — survives the
forward. Transient falls to 30 GiB / ceil(n/chunk) and the retained log-softmax
to zero; the cost is one extra LM-head matmul per chunk during backward, the
same trade tts/train/loop.py:183-187 already makes for the 28 backbone layers.

WHY THE MATHS IS THE SAME. `reduction="sum"` is additive over rows, and CE is
computed per row: sum_i f(x_i) == sum_chunks sum_{i in chunk} f(x_i) in exact
arithmetic, and in fp32 the two differ only in SUMMATION ORDER (measured below
1e-7 relative in tests/test_chunked_ce.py). Every cast is the one the unchunked
path makes, in the same place: `head(h)` in the module dtype, `.float()`, then
`F.cross_entropy(..., reduction="sum")`. `weight` and `ignore_index` are
forwarded untouched, so a weighted head (the AEOS-weighted cb0 CE,
tts/train/losses.py:97-107) decomposes identically — with `reduction="sum"`
PyTorch returns sum_i w[t_i] * loss_i and applies no weight normalisation, which
is exactly what makes the split legal.

WHY NOT A `nn.Module`. It holds no state. Keeping it a function means the
state_dict, and hence the S2 checkpoint contract, is untouched, and the caller
stays inside `TTSModelV2.forward` so the sharded LM head
is only ever entered through `Module.__call__` while the root module's forward
is live — the FSDP2 rule the audio_io lesson is named after.
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


def chunked_ce_sum(head, hidden, targets, chunk, *, weight=None,
                   ignore_index=-100):
    """CE SUM over `hidden` in `chunk`-row pieces. -> () fp32 tensor.

    head            callable (rows, d) -> (rows, vocab); the module call whose
                    parameters must still receive gradients (e.g.
                    `Backbone.text_logits`).
    hidden          (N, d) hidden states, model dtype.
    targets         (N,) long class indices.
    chunk           rows per piece; must be > 0 (0 means "off" and is the
                    caller's decision, not this function's).
    weight          per-class weights, forwarded to F.cross_entropy unchanged.
    ignore_index    forwarded unchanged (default -100, F.cross_entropy's).

    An empty `hidden` returns a 0.0 that is still connected to `head`'s
    parameters ONLY if the caller made it so; it does not fabricate a graph.
    Callers that need gradient symmetry on an empty batch must use their own
    zero term (the pattern in DepthV2.forward), because this function must not
    invent a collective the other ranks do not issue.
    """
    if int(chunk) <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk} (0 = off is the "
                         f"caller's branch, not this function's)")
    chunk = int(chunk)
    n = int(hidden.shape[0])
    if targets.shape[0] != n:
        raise ValueError(f"hidden has {n} rows, targets {targets.shape[0]}")

    def _ce(h_chunk, t_chunk):
        return F.cross_entropy(head(h_chunk).float(), t_chunk, weight=weight,
                               ignore_index=ignore_index, reduction="sum")

    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for i in range(0, n, chunk):
        h_c, t_c = hidden[i:i + chunk], targets[i:i + chunk]
        if torch.is_grad_enabled():
            # preserve_rng_state=False: the chunk is a matmul + CE, with no RNG
            # of any kind, and the default True does a device RNG snapshot per
            # chunk — a host-device sync we would pay ~8x per text update.
            part = ckpt.checkpoint(_ce, h_c, t_c, use_reentrant=False,
                                   preserve_rng_state=False)
        else:
            part = _ce(h_c, t_c)      # eval/no_grad: checkpointing saves nothing
        total = total + part
    return total
