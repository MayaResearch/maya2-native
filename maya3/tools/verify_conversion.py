"""Verify a safetensors release export against its source `model.pt`, on CPU.

    CUDA_VISIBLE_DEVICES= python -m maya3.tools.verify_conversion <model.pt> <out_dir> [--backbone-dir DIR]

Checks, in order (any failure exits non-zero):
  1. raw tensors: the export (aliases restored) has exactly the source's names, and every tensor has the same
     dtype, shape and bytes (bitwise; also torch.equal);
  2. both load through `tts.model.load_s3.load_model_for_eval` (the source with its run config / the engine's
     default config, the export with its own maya3_config.json) into the same model: same anchor config,
     depth_grouped, tensor count and parameter count;
  3. every state-dict tensor of the two loaded models is bit-identical (`torch.equal` + bitwise).
The backbone is built from <out_dir>/backbone (config + tokenizer only) for both loads unless --backbone-dir is given.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts.model import load_s3 as L  # noqa: E402
from maya3.tools.convert_to_safetensors import ENGINE_DEFAULT_CONFIG, bits_equal  # noqa: E402


def compare(a, b, what):
    """-> list of mismatch strings."""
    errs = []
    if set(a) != set(b):
        errs.append(f"{what}: key sets differ: only in source {sorted(set(a) - set(b))[:8]}, "
                    f"only in export {sorted(set(b) - set(a))[:8]}")
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            errs.append(f"{what}: {k}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}")
        elif not torch.equal(x, y) and not (torch.isnan(x).any() and bits_equal(x, y)):
            errs.append(f"{what}: {k}: values differ")
        elif not bits_equal(x, y):
            errs.append(f"{what}: {k}: bit patterns differ (e.g. signed zeros)")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--backbone-dir", help="default: <out_dir>/backbone")
    ap.add_argument("--model-config", help="config for the SOURCE load (default: the run config next to it, "
                                           "else configs/s3_fresh_v1.yaml)")
    ap.add_argument("--skip-models", action="store_true", help="only the raw tensor comparison (step 1)")
    A = ap.parse_args(argv)
    t0 = time.time()
    flat = L._flat_path(os.path.abspath(A.ckpt))
    out = os.path.abspath(A.out_dir)
    bb = os.path.abspath(A.backbone_dir or os.path.join(out, "backbone"))
    errs = []

    # 1. raw tensors
    src = torch.load(flat, map_location="cpu", weights_only=True, mmap=True)
    st = L._safetensors_files(out)
    if st is None:
        sys.exit(f"{out}: no model.safetensors[.index.json]")
    exp = L._load_safetensors(st)
    errs += compare(src, exp, "raw")
    print(f"[1] raw: {len(src)} source tensors vs {len(exp)} export tensors ({len(st[1])} file(s), "
          f"aliases {st[2]}): {'OK' if not errs else 'FAIL'}  ({time.time() - t0:.0f}s)", flush=True)
    del src, exp
    gc.collect()
    if A.skip_models or errs:
        return finish(errs)

    # 2./3. through load_model_for_eval
    cfg = {"model": {"backbone": bb, "attn": "sdpa"}}
    mc = A.model_config
    if mc is None and L.find_run_config(flat) is None:
        mc = str(ENGINE_DEFAULT_CONFIG)
    m1, p1 = L.load_model_for_eval(flat, cfg=cfg, config=mc, require_anchor=None)
    sd1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}
    del m1
    gc.collect()
    m2, p2 = L.load_model_for_eval(out, cfg=cfg, require_anchor=None)
    sd2 = m2.state_dict()
    for key in ("anchor", "anchor_cfg", "depth_grouped", "depth_ckpt_from", "n_tensors", "n_params", "update"):
        if p1.get(key) != p2.get(key):
            errs.append(f"provenance {key}: source {p1.get(key)!r} vs export {p2.get(key)!r}")
    e = compare(sd1, sd2, "state_dict")
    errs += e
    print(f"[2] provenance: source format={p1['format']} cfg={p1['config_source']} | export format={p2['format']} "
          f"cfg={p2['config_source']}", flush=True)
    print(f"[3] state_dict: {len(sd1)} vs {len(sd2)} tensors, "
          f"dtypes {sorted({str(v.dtype) for v in sd2.values()})}: {'OK' if not e else 'FAIL'}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return finish(errs, {"n_state_dict": len(sd1), "n_params": p2["n_params"], "anchor_cfg": p2["anchor_cfg"],
                         "depth_grouped": p2["depth_grouped"]})


def finish(errs, extra=None):
    if errs:
        for e in errs[:50]:
            print("MISMATCH " + e)
        print(f"VERIFY FAIL ({len(errs)} mismatches)")
        sys.exit(1)
    print("VERIFY OK " + json.dumps(extra or {}, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
