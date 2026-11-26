"""GPU unit test: _decode_attn_gqa vs _decode_attn (the per-head kernel it replaces) and an fp32 reference, on the static
cache layout [max_slots+1, kv_heads, capacity, head_dim] with slot indirection and per-row positions (block edges, 0,
capacity-1, inline-prefill rows of one slot).  Also: the dynamic loop bound inside a replayed CUDA graph.
Usage: test_decode_attn_gqa.py   (no weights needed)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not torch.cuda.is_available():
    print("SKIP: needs CUDA"); sys.exit(0)
import triton
from serve.fast_backbone import _decode_attn, _decode_attn_gqa
torch.manual_seed(0)
H, KV, D = 16, 8, 128          # Qwen3 backbone geometry (backbone/config.json)
G = H // KV
scale = D ** -0.5


def old(q, kc, vc, slots, pos, ctx):
    o = torch.empty_like(q); block = 256
    _decode_attn[(q.shape[0], H)](q, kc, vc, o, slots, pos, scale, q.stride(0), q.stride(1), kc.stride(0), kc.stride(1), kc.stride(2),
                                  o.stride(0), o.stride(1), G=G, D=D, BLOCK=block, NBLOCKS=(ctx + block - 1) // block)
    return o


def new(q, kc, vc, slots, pos, o=None):
    o = torch.empty_like(q) if o is None else o
    _decode_attn_gqa[(q.shape[0], KV)](q, kc, vc, o, slots, pos, scale, q.stride(0), q.stride(1), kc.stride(0), kc.stride(1),
                                       kc.stride(2), o.stride(0), o.stride(1), G=G, GP=max(16, triton.next_power_of_2(G)), D=D,
                                       BLOCK=32, num_warps=4, num_stages=2)
    return o


def ref32(q, kc, vc, slots, pos):
    out = []
    for i in range(q.shape[0]):
        k = kc[slots[i], :, :pos[i] + 1].float(); v = vc[slots[i], :, :pos[i] + 1].float()        # [KV, t, D]
        s = torch.einsum("kgd,ktd->kgt", q[i].float().view(KV, G, D), k) * scale
        out.append(torch.einsum("kgt,ktd->kgd", s.softmax(-1), v).reshape(H, D))
    return torch.stack(out)


worst = {}
with torch.inference_mode():
    for dt in (torch.bfloat16, torch.float16):
        for CAP in (1024, 1536):
            edges = [0, 1, 63, 64, 65, 255, 256, 257, CAP // 2, CAP - 2, CAP - 1]
            for B in (1, 2, 7, 64, 192, 256, 384, 512):
                S = B + 1                                                     # last slot = the dummy padding slot
                kc = torch.randn(S, KV, CAP, D, device="cuda").to(dt); vc = torch.randn_like(kc)
                kc[:, :, :, :8] *= 4                                          # peakier scores than iid noise
                q = torch.randn(B, H, D, device="cuda").to(dt)
                slots = torch.randperm(S, device="cuda")[:B]
                pos = torch.randint(0, CAP, (B,), device="cuda")
                pos[:min(B, len(edges))] = torch.tensor(edges[:B], device="cuda")
                if B >= 64:                                                   # inline prefill: 8 rows of one slot at consecutive positions
                    slots[-8:] = slots[-9]; pos[-8:] = torch.arange(100, 108, device="cuda")
                a, b = old(q, kc, vc, slots, pos, CAP), new(q, kc, vc, slots, pos)
                rows = torch.arange(B, device="cuda") if B <= 64 else torch.randperm(B, device="cuda")[:64]
                r = ref32(q[rows], kc, vc, slots[rows], pos[rows].tolist())
                for i in range(min(B, len(edges))):                           # the edge rows are always checked
                    if i not in rows.tolist():
                        rows = torch.cat([rows, torch.tensor([i], device="cuda")])
                        r = torch.cat([r, ref32(q[i:i + 1], kc, vc, slots[i:i + 1], pos[i:i + 1].tolist())])
                e_new = (b[rows].float() - r).abs().max().item(); e_old = (a[rows].float() - r).abs().max().item()
                l_new = ((b[rows].float() - r).norm() / r.norm()).item(); l_old = ((a[rows].float() - r).norm() / r.norm()).item()
                w = worst.setdefault(str(dt).split(".")[-1], [0, 0, 0, 0])
                w[0] = max(w[0], e_new); w[1] = max(w[1], e_old); w[2] = max(w[2], l_new); w[3] = max(w[3], l_old)
                # same fp32 softmax and P @ V as the old kernel: only summation order differs
                assert e_new <= e_old * 1.05 + 1e-6, (dt, CAP, B, e_new, e_old)
                assert l_new <= l_old * 1.05 + 1e-6, (dt, CAP, B, l_new, l_old)
                assert torch.isfinite(b).all()
    for dt, (e_new, e_old, l_new, l_old) in worst.items():
        print(f"{dt:9s} max|new-fp32| {e_new:.2e}  max|old-fp32| {e_old:.2e}  rel L2 new {l_new:.2e}  old {l_old:.2e}")

    # CUDA graph: the loop bound comes from Pos at replay time, so changing positions between replays must be honoured
    CAP, B = 1536, 64
    kc = torch.randn(B + 1, KV, CAP, D, device="cuda", dtype=torch.bfloat16); vc = torch.randn_like(kc)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(B, device="cuda"); pos = torch.zeros(B, dtype=torch.long, device="cuda"); o = torch.empty_like(q)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        new(q, kc, vc, slots, pos, o)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        new(q, kc, vc, slots, pos, o)
    for p in (0, 64, 700, CAP - 1):
        pos.copy_(torch.randint(0, p + 1, (B,), device="cuda")); pos[0] = p
        g.replay()
        r = ref32(q, kc, vc, slots, pos.tolist())
        assert (o.float() - r).abs().max().item() < 2e-2, p
print("DECODE_ATTN_GQA_OK")
