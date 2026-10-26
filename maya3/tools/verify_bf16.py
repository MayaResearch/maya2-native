"""Check that the bf16 copy loads to bit-identical engine weights: load_model_for_eval(dtype=bf16) on both files (CPU).
Usage: verify_bf16.py <fp32 model.pt> <bf16 model.pt> <backbone_dir>"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tts.model.load_s3 import load_model_for_eval
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cfg = {"model": {"backbone": sys.argv[3], "attn": "sdpa"}, "data": {"max_tokens": 8192}}
def load(p):
    m, _ = load_model_for_eval(p, device=None, dtype=torch.bfloat16, cfg=cfg, config=f"{ROOT}/configs/s3_fresh_v1.yaml",
                               amortize_frac=1.0, require_anchor=True, verbose=False)
    return m.state_dict()
a, b = load(sys.argv[1]), load(sys.argv[2])
bad = [k for k in a if not torch.equal(a[k], b[k])]
print(f"{len(a)} tensors, {len(bad)} differ", bad[:3]); assert not bad
print("BF16_LOSSLESS_OK")
