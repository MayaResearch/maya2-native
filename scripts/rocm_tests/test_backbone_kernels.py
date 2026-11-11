from __future__ import annotations
import unittest
import torch
import triton
from ._common import require_rocm, assert_finite, max_abs_error
from serve.fast_backbone import _decode_attn, _qk_norm_rope_cache, _add_rms

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _rms_bf16(x, weight, eps):
    inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = (x.float() * inv).to(torch.bfloat16).float()
    return (normalized * weight.float()).to(torch.bfloat16)

class BackboneKernelTests(unittest.TestCase):
    def test_decode_attention_fp16_bf16(self):
        device = require_rocm(self)
        torch.manual_seed(1701)
        cases = [(2, 4, 2, 64, 128, [37, 81]), (1, 8, 2, 64, 512, [319])]
        for dtype in (torch.float16, torch.bfloat16):
            for B, H, KV, D, CTX, positions in cases:
                with self.subTest(dtype=str(dtype), B=B, H=H, CTX=CTX):
                    G = H // KV
                    q = torch.randn(B, H, D, device=device, dtype=dtype)
                    k = torch.randn(4, KV, CTX, D, device=device, dtype=dtype)
                    v = torch.randn_like(k)
                    slots = torch.tensor([1, 3][:B], device=device, dtype=torch.long)
                    pos = torch.tensor(positions, device=device, dtype=torch.long)
                    out = torch.empty_like(q)
                    _decode_attn[(B, H)](q, k, v, out, slots, pos, D ** -0.5,
                        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                        out.stride(0), out.stride(1), G=G, D=D, BLOCK=256, NBLOCKS=triton.cdiv(CTX, 256))
                    torch.cuda.synchronize(device)
                    qr = q.float().view(B, KV, G, D)
                    kr, vr = k.index_select(0, slots).float(), v.index_select(0, slots).float()
                    scores = torch.einsum("bkgd,bktd->bkgt", qr, kr) * (D ** -0.5)
                    mask = torch.arange(CTX, device=device)[None, None, None, :] <= pos[:, None, None, None]
                    scores = scores.masked_fill(~mask, float("-inf"))
                    expected = torch.einsum("bkgt,bktd->bkgd", scores.softmax(-1), vr).reshape(B, H, D)
                    err = max_abs_error(out, expected)
                    tolerance = 0.04 if dtype == torch.bfloat16 else 0.005
                    assert_finite(self, out, "_decode_attn output")
                    self.assertLess(err, tolerance, f"{dtype=} {B=} {CTX=} max_abs_error={err}")

    def test_qk_norm_rope_and_kv_cache_bf16(self):
        device = require_rocm(self)
        torch.manual_seed(1702)
        B, H, KV, D, CAP = 2, 4, 2, 64, 128
        slots = torch.tensor([1, 3], device=device, dtype=torch.long)
        pos = torch.tensor([7, 19], device=device, dtype=torch.long)
        qkv = torch.randn(B, (H + 2 * KV) * D, device=device, dtype=torch.bfloat16)
        qout = torch.empty(B, H, D, device=device, dtype=torch.bfloat16)
        k_cache = torch.randn(5, KV, CAP, D, device=device, dtype=torch.bfloat16)
        v_cache = torch.randn_like(k_cache)
        k_expected, v_expected = k_cache.clone(), v_cache.clone()
        wq = torch.randn(D, device=device, dtype=torch.bfloat16)
        wk = torch.randn(D, device=device, dtype=torch.bfloat16)
        eps = 1e-6
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=device).float() / D))
        angles = torch.arange(CAP, device=device).float()[:, None] * inv_freq[None, :]
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16).contiguous()
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16).contiguous()
        _qk_norm_rope_cache[(B, H + KV)](qkv, qout, k_cache, v_cache, slots, pos, cos, sin, wq, wk, eps,
            qkv.stride(0), qout.stride(0), qout.stride(1), k_cache.stride(0), k_cache.stride(1),
            k_cache.stride(2), cos.stride(0), H=H, KVH=KV, D=D)
        torch.cuda.synchronize(device)
        parts = qkv.view(B, H + 2 * KV, D)
        q_norm = _rms_bf16(parts[:, :H], wq, eps)
        k_norm = _rms_bf16(parts[:, H:H + KV], wk, eps)
        q_expected = (q_norm.float() * cos[pos, None].float() + _rotate_half(q_norm).float() * sin[pos, None].float()).to(torch.bfloat16)
        k_rot = k_norm.float() * cos[pos, None].float() + _rotate_half(k_norm).float() * sin[pos, None].float()
        v_values = parts[:, H + KV:]
        for row in range(B):
            slot, p = int(slots[row].item()), int(pos[row].item())
            k_expected[slot, :, p, :] = k_rot[row].to(torch.bfloat16)
            v_expected[slot, :, p, :] = v_values[row]
        assert_finite(self, qout, "_qk_norm_rope_cache Q output")
        self.assertLess(max_abs_error(qout, q_expected), 0.02, "Q output mismatch")
        self.assertLess(max_abs_error(k_cache, k_expected), 0.02, "K cache write mismatch")
        self.assertLess(max_abs_error(v_cache, v_expected), 1e-7, "V cache write mismatch")

    def test_add_rms_bf16(self):
        device = require_rocm(self)
        torch.manual_seed(1703)
        B, N, eps = 3, 128, 1e-6
        h = torch.randn(B, N, device=device, dtype=torch.bfloat16)
        a = torch.randn_like(h)
        weight = torch.randn(N, device=device, dtype=torch.bfloat16)
        h_out, y = torch.empty_like(h), torch.empty_like(h)
        _add_rms[(B,)](h, a, h_out, y, weight, eps, h.stride(0), a.stride(0), h_out.stride(0), y.stride(0), N=N)
        torch.cuda.synchronize(device)
        summed = (h.float() + a.float()).to(torch.bfloat16)
        expected_y = ((summed.float() * torch.rsqrt(summed.float().pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16).float() * weight.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(h_out, summed), "residual output differs")
        self.assertLess(max_abs_error(y, expected_y), 0.02, f"RMS output differs; max_abs_error={max_abs_error(y, expected_y)}")

    def test_decode_attention_kernel_graph_capture(self):
        device = require_rocm(self)
        B, H, KV, D, CTX = 1, 4, 2, 64, 256
        q = torch.randn(B, H, D, device=device, dtype=torch.float16)
        k = torch.randn(2, KV, CTX, D, device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        slots = torch.tensor([1], device=device, dtype=torch.long)
        pos = torch.tensor([127], device=device, dtype=torch.long)
        out = torch.empty_like(q)
        def launch():
            _decode_attn[(B, H)](q, k, v, out, slots, pos, D ** -0.5,
                q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                out.stride(0), out.stride(1), G=H // KV, D=D, BLOCK=256, NBLOCKS=1)
        launch()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        torch.cuda.synchronize(device)
        assert_finite(self, out, "graph-captured _decode_attn")
        qr = q.float().view(B, KV, H // KV, D)
        scores = torch.einsum("bkgd,bktd->bkgt", qr, k.index_select(0, slots).float()) * D ** -0.5
        mask = torch.arange(CTX, device=device)[None, None, None, :] <= pos[:, None, None, None]
        expected = torch.einsum("bkgt,bktd->bkgd", scores.masked_fill(~mask, -float("inf")).softmax(-1), v.index_select(0, slots).float()).reshape(B, H, D)
        self.assertLess(max_abs_error(out, expected), 0.005)

if __name__ == "__main__":
    unittest.main()
