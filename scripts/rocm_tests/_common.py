from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch

def require_rocm(testcase):
    if not torch.cuda.is_available():
        testcase.skipTest("PyTorch accelerator is unavailable")
    if not getattr(torch.version, "hip", None):
        testcase.skipTest("This suite requires a ROCm/HIP PyTorch build")
    return torch.device("cuda", torch.cuda.current_device())

def max_abs_error(actual, expected):
    return float((actual.float() - expected.float()).abs().max().item())

def assert_finite(testcase, tensor, label):
    testcase.assertTrue(bool(torch.isfinite(tensor).all().item()), f"{label} contains NaN or Inf")
