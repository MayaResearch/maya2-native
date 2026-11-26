"""GPU unit test: tiny_attention (serve/fast_depth_kernels.py) vs F.scaled_dot_product_attention on every attention shape
FastDepth produces, built with the callers' strides.  Error of record: max |kernel - fp32 SDPA| against the same-dtype SDPA's
own error vs fp32 SDPA.  Also: fallback to SDPA outside the kernel's domain, and replay inside a CUDA graph.
Usage: test_tiny_attn.py   (no weights needed)"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not torch.cuda.is_available():
    print("SKIP: needs CUDA"); sys.exit(0)
from serve.fast_depth_kernels import tiny_attention
from tts.model.depth_v2 import GROUP_OF_LEVEL, LAYERS, D, H
HEAD, N_LEVELS = D // H, 32
torch.manual_seed(0)


def shapes():
    """(kind, T, N, causal) for every attention call of one FastDepth frame (mirrors _compute / _compute_fused)."""
    out, previous = set(), None
    for level in range(1, N_LEVELS):
        group, n = GROUP_OF_LEVEL[level], level + 1
        for layer in range(LAYERS):
            if group == previous:
                out.add(("incremental", 1, n, False))
            elif layer == LAYERS - 1:
                out.add(("last", 1, n, False))
            else:
                out.add(("causal", n, n, True))
        previous = group
    return sorted(out, key=lambda s: (s[0], s[2]))


def inputs(kind, B, T, N, dt):
    """q/k/v laid out exactly as FastDepth passes them (views of the packed qkv / the per-layer KV buffer)."""
    if kind == "incremental":
        qkv = torch.randn(B, 1, 3, H, HEAD, device="cuda").to(dt)
        kv = torch.randn(B, N_LEVELS, 2, H, HEAD, device="cuda").to(dt)
        return qkv[:, :, 0].transpose(1, 2), kv[:, :N, 0].transpose(1, 2), kv[:, :N, 1].transpose(1, 2)
    qkv = torch.randn(B, N, 3, H, HEAD, device="cuda").to(dt)
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
    return (q[:, :, -1:] if kind == "last" else q), k, v


S = shapes()
assert max(n for _, _, n, _ in S) <= 32, "depth prefix longer than the kernel's 32 keys"
print(f"{len(S)} distinct (kind, T, N, causal) shapes, N in [{min(s[2] for s in S)}, {max(s[2] for s in S)}]")
worst = {}
with torch.inference_mode():
    for dt in (torch.float16, torch.bfloat16):
        for B in (1, 2, 7, 64, 256, 384):
            for kind, T, N, causal in S:
                q, k, v = inputs(kind, B, T, N, dt)
                ref32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
                sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                out = tiny_attention(q, k, v, is_causal=causal)
                assert out.shape == sdpa.shape and out.dtype == dt
                e_k = (out.float() - ref32).abs().max().item()
                e_s = (sdpa.float() - ref32).abs().max().item()
                r_k = ((out.float() - ref32).norm() / ref32.norm()).item()
                key = (str(dt).split(".")[-1], kind)
                w = worst.setdefault(key, [0.0, 0.0, 0.0])
                w[0] = max(w[0], e_k); w[1] = max(w[1], e_s); w[2] = max(w[2], r_k)
                # fp32 softmax and P @ V: only the output rounding remains, so no worse than SDPA's (+ one ulp of slack)
                assert e_k <= e_s + (8e-3 if dt == torch.bfloat16 else 1e-3), (key, B, T, N, e_k, e_s)
    for (dt, kind), (e_k, e_s, r_k) in sorted(worst.items()):
        print(f"{dt:9s} {kind:12s} max|kernel-fp32| {e_k:.2e}  max|sdpa-fp32| {e_s:.2e}  kernel rel L2 {r_k:.2e}")

    # outside the kernel's domain it must defer to SDPA (fp32, > 32 keys, causal with T < N)
    q, k, v = (torch.randn(2, H, 8, HEAD, device="cuda") for _ in range(3))
    assert torch.equal(tiny_attention(q, k, v, is_causal=True), F.scaled_dot_product_attention(q, k, v, is_causal=True))
    q, k, v = (torch.randn(2, H, 40, HEAD, device="cuda", dtype=torch.float16) for _ in range(3))
    assert torch.equal(tiny_attention(q, k, v), F.scaled_dot_product_attention(q, k, v))
    q = torch.randn(2, H, 5, HEAD, device="cuda", dtype=torch.float16); k, v = (torch.randn(2, H, 9, HEAD, device="cuda", dtype=torch.float16) for _ in range(2))
    e = (tiny_attention(q, k, v, is_causal=True).float() - F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)).abs().max().item()
    assert e < 2e-3, e   # top-left causal alignment, as SDPA

    # CUDA graph: capture once, replay on new data in the same buffers
    q, k, v = inputs("causal", 64, 9, 9, torch.float16)
    q, k, v = q.clone(), k.clone(), v.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tiny_attention(q, k, v, is_causal=True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = tiny_attention(q, k, v, is_causal=True)
    for _ in range(3):
        for t in (q, k, v): t.copy_(torch.randn_like(t))
        g.replay()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
        assert (o.float() - ref).abs().max().item() < 2e-3
print("TINY_ATTN_OK")
