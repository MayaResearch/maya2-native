from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
from ._common import require_rocm, assert_finite

class BackendPrimitiveTests(unittest.TestCase):
    def test_matmul_fp16_bf16(self):
        device = require_rocm(self)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                a = torch.randn(1024, 1024, device=device, dtype=dtype)
                b = torch.randn(1024, 1024, device=device, dtype=dtype)
                y = a @ b
                assert_finite(self, y, f"matmul {dtype}")
                self.assertEqual(tuple(y.shape), (1024, 1024))

    def test_sdpa_fp16_bf16(self):
        device = require_rocm(self)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                q = torch.randn(2, 4, 128, 64, device=device, dtype=dtype)
                k, v = torch.randn_like(q), torch.randn_like(q)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                assert_finite(self, y, f"SDPA {dtype}")
                self.assertEqual(tuple(y.shape), tuple(q.shape))

    def test_cuda_graph_matmul_capture_replay(self):
        device = require_rocm(self)
        a = torch.randn(512, 512, device=device, dtype=torch.float16)
        b = torch.randn(512, 512, device=device, dtype=torch.float16)
        eager = a @ b
        out = torch.empty_like(eager)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(a @ b)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_finite(self, out, "graph matmul")
        torch.testing.assert_close(out, eager, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    unittest.main()
