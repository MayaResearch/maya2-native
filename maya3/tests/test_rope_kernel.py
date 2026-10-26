"""GPU unit test: _qk_norm_rope_cache vs HF Qwen3 q_norm/k_norm + rotate_half RoPE (one layer)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from serve.fast_backbone import _qk_norm_rope_cache, _rotate_half
torch.manual_seed(0)
B, H, KV, D, CAP = 5, 16, 8, 128, 512
dt = torch.bfloat16
qkv = (torch.randn(B, (H + 2 * KV) * D, device="cuda") * 2).to(dt)
wq = (1 + 0.1 * torch.randn(D, device="cuda")).to(dt); wk = (1 + 0.1 * torch.randn(D, device="cuda")).to(dt)
pos = torch.tensor([0, 7, 100, 300, 511], device="cuda"); slots = torch.tensor([0, 1, 2, 3, 4], device="cuda")
inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
fr = torch.arange(CAP, device="cuda").float()[:, None] * inv[None]
emb = torch.cat([fr, fr], -1); cos, sin = emb.cos().to(dt).contiguous(), emb.sin().to(dt).contiguous()
def rms(x, w, eps=1e-6):
    xf = x.float(); return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dt)
q_, k_, v_ = qkv.split([H * D, KV * D, KV * D], -1)
c, s = cos[pos][:, None], sin[pos][:, None]
q = rms(q_.view(B, H, D), wq); k = rms(k_.view(B, KV, D), wk)
qr = q * c + _rotate_half(q) * s; kr = k * c + _rotate_half(k) * s
K = torch.zeros(6, KV, CAP, D, device="cuda", dtype=dt); V = torch.zeros_like(K); qo = torch.empty(B, H, D, device="cuda", dtype=dt)
_qk_norm_rope_cache[(B, H + KV)](qkv, qo, K, V, slots, pos, cos, sin, wq, wk, 1e-6, qkv.stride(0), qo.stride(0), qo.stride(1),
                                 K.stride(0), K.stride(1), K.stride(2), cos.stride(0), H=H, KVH=KV, D=D)
kk = K[slots, :, pos]; vv = V[slots, :, pos]
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
print("q rel", rel(qo, qr), "k rel", rel(kk, kr), "v exact", torch.equal(vv, v_.view(B, KV, D)))
assert rel(qo, qr) < 1e-2 and rel(kk, kr) < 1e-2 and torch.equal(vv, v_.view(B, KV, D))
print("ROPE_KERNEL_OK")
