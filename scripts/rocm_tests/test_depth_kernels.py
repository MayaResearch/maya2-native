from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
from ._common import require_rocm, assert_finite, max_abs_error
from serve.fast_depth_kernels import add_ln, topk_sample, row_exponential

class DepthKernelTests(unittest.TestCase):
    def test_add_layernorm_fp16_bf16_with_and_without_cross(self):
        device = require_rocm(self)
        torch.manual_seed(2701)
        B, T, D = 2, 3, 128
        for dtype in (torch.float16, torch.bfloat16):
            for use_cross in (False, True):
                with self.subTest(dtype=str(dtype), use_cross=use_cross):
                    x = torch.randn(B, T, D, device=device, dtype=dtype)
                    a = torch.randn_like(x)
                    c = torch.randn(B, 1, D, device=device, dtype=dtype) if use_cross else None
                    ln = torch.nn.LayerNorm(D, device=device, dtype=dtype)
                    xo, y = add_ln(x, a, ln, c)
                    expected_x = (x.float() + a.float()).to(dtype)
                    if c is not None:
                        expected_x = (expected_x.float() + c.float()).to(dtype)
                    expected_y = F.layer_norm(expected_x, (D,), ln.weight, ln.bias, ln.eps)
                    assert_finite(self, xo, "_add_ln residual")
                    assert_finite(self, y, "_add_ln LayerNorm")
                    self.assertTrue(torch.equal(xo, expected_x), "_add_ln residual rounding mismatch")
                    tolerance = 0.04 if dtype == torch.bfloat16 else 0.01
                    err = max_abs_error(y, expected_y)
                    self.assertLess(err, tolerance, f"_add_ln {dtype=} {use_cross=} max_abs_error={err}")

    def test_topk_sample_greedy(self):
        device = require_rocm(self)
        torch.manual_seed(2702)
        B, V, K = 8, 1024, 32
        logits = torch.randn(B, V, device=device, dtype=torch.float32)
        seeds = torch.arange(B, device=device, dtype=torch.int32) + 11
        actual = topk_sample(logits, seeds, level=3, temp=0.8, top_k=K, greedy=True, ctr_t=torch.zeros_like(seeds))
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(actual, expected), "greedy top-k sampler differs from argmax")

    def test_topk_sample_stochastic_stays_inside_topk_and_repeats(self):
        device = require_rocm(self)
        torch.manual_seed(2703)
        B, V, K = 16, 1024, 64
        logits = torch.randn(B, V, device=device, dtype=torch.float32)
        seeds = torch.arange(B, device=device, dtype=torch.int32) + 101
        counters = torch.arange(B, device=device, dtype=torch.int32) + 7
        a = topk_sample(logits, seeds, level=5, temp=0.75, top_k=K, greedy=False, ctr_t=counters)
        b = topk_sample(logits, seeds, level=5, temp=0.75, top_k=K, greedy=False, ctr_t=counters)
        self.assertTrue(torch.equal(a, b), "same Philox seeds/counters should reproduce the same samples")
        allowed = torch.zeros(B, V, device=device, dtype=torch.bool)
        allowed.scatter_(1, (logits / 0.75).topk(K, dim=-1).indices, True)
        self.assertTrue(bool(allowed.gather(1, a[:, None]).all().item()), "sampler selected an index outside top-k")

    def test_row_exponential_finite_positive_and_deterministic(self):
        device = require_rocm(self)
        B, V = 4, 1024
        seeds = torch.tensor([3, 5, 7, 11], device=device, dtype=torch.int32)
        counters = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        a = row_exponential(B, V, seeds, counters, level=2)
        b = row_exponential(B, V, seeds, counters, level=2)
        assert_finite(self, a, "row_exponential")
        self.assertTrue(bool((a >= 0).all().item()), "exponential samples must be non-negative")
        self.assertTrue(torch.equal(a, b), "same Philox seeds/counters should reproduce the same values")

if __name__ == "__main__":
    unittest.main()
