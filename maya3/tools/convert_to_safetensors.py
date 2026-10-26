"""Convert a flat Maya3 checkpoint (`model.pt`, keys `model.<name>`) into the safetensors release format.

    python -m maya3.tools.convert_to_safetensors <model.pt> <out_dir> [--backbone-src DIR] [--model-config YAML]

Writes into <out_dir>:
  model.safetensors                      (one file when the stored bytes are <= --max-shard-gb, default 5)
  model-0000k-of-0000N.safetensors       (otherwise) + model.safetensors.index.json
  maya3_config.json                      the resolved model config `tts.model.load_s3.load_model_for_eval` needs
                                         (speaker anchor, depth_grouped, ...), the format version and provenance
  backbone/                              with --backbone-src: config.json + tokenizer files only (never weights)

Guarantees:
  * every tensor keeps its stored dtype and values exactly (no casts; non-contiguous tensors are made contiguous);
  * tensors that share storage (tied weights: lm_head <-> embed_tokens) and tensors that are exact copies of each
    other (same dtype/shape, bit-identical) are stored ONCE; the other names are recorded as aliases in the
    safetensors metadata (`maya3_aliases`) and in maya3_config.json, and the loader restores them;
  * keys are stored without the training-namespace `model.` prefix (the loader adds it back).

Check the result with `python -m maya3.tools.verify_conversion <model.pt> <out_dir>`.
CPU only; needs about 2x the checkpoint size in RAM and the checkpoint size in free disk.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts.model import load_s3 as L  # noqa: E402

FORMAT = "maya3-safetensors"
FORMAT_VERSION = 1
ENGINE_DEFAULT_CONFIG = ROOT / "configs" / "s3_fresh_v1.yaml"   # what maya3.engine.Engine passes as model_config
LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
# small backbone metadata the architecture (AutoConfig) and tokenizer (AutoTokenizer) need; never weights
BACKBONE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _gb(n):
    return n / 1e9


def _free_bytes(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def bits_equal(a, b):
    """Bitwise equality (torch.equal treats -0.0 == 0.0 and NaN != NaN; aliasing must be exact)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def _alias_preference(name):
    # keep the input embedding as the stored copy of a tied pair (the HF convention); otherwise the first name
    return ("lm_head" in name, name)


def plan_storage(sd, dedupe_equal=True):
    """-> (stored {name: tensor}, aliases {alias: stored name}).

    Shared storage: identical views -> alias; any other overlap -> an independent contiguous copy (values exact).
    Bit-identical independent tensors (same dtype, shape, bytes) -> alias as well (lossless dedupe)."""
    by_storage = collections.defaultdict(list)
    for k, v in sd.items():
        by_storage[(v.device.type, v.untyped_storage().data_ptr())].append(k)
    stored, aliases = {}, {}
    for names in by_storage.values():
        names = sorted(names, key=_alias_preference)
        views = {}
        for k in names:
            v = sd[k]
            sig = (v.dtype, tuple(v.shape), tuple(v.stride()), v.storage_offset())
            if sig in views:
                aliases[k] = views[sig]
            else:
                views[sig] = k
                stored[k] = v
        if len(views) > 1:                   # partially-overlapping views of one storage: unshare them
            for k in views.values():
                stored[k] = stored[k].contiguous().clone()
    # exact-copy dedupe among independent tensors
    groups = collections.defaultdict(list)
    for k, v in stored.items():
        groups[(v.dtype, tuple(v.shape))].append(k)
    for names in (groups.values() if dedupe_equal else ()):
        if len(names) < 2:
            continue
        names = sorted(names, key=_alias_preference)
        keep = []
        for k in names:
            hit = next((c for c in keep if bits_equal(stored[c], stored[k])), None)
            if hit is None:
                keep.append(k)
            else:
                aliases[k] = hit
        for k in names:
            if k in aliases:
                stored.pop(k)
    # an alias must point at a stored name (chains cannot form: targets are always stored names)
    assert all(t in stored for t in aliases.values())
    stored = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in stored.items()}
    return stored, aliases


def shard(stored, max_bytes):
    """Greedy, in name order -> [[names], ...]."""
    shards, cur, size = [], [], 0
    for k in sorted(stored):
        n = stored[k].numel() * stored[k].element_size()
        if cur and size + n > max_bytes:
            shards.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += n
    if cur:
        shards.append(cur)
    return shards


def resolve_release_model_cfg(ckpt, shapes, model_config=None):
    """The inference-relevant model block, reconciled against the checkpoint's own tensors (as load_s3 does)."""
    if model_config is not None:
        mc, src = L.resolve_model_cfg(ckpt, config=str(model_config))
    else:
        mc, src = L.resolve_model_cfg(ckpt)
        if src == "none":
            mc, src = L.resolve_model_cfg(ckpt, config=str(ENGINE_DEFAULT_CONFIG))
    anchor = L._reconcile_anchor(mc.get("speaker_anchor"), L.anchor_cfg_from_shapes(shapes))
    dcf = mc.get("depth_ckpt_from")
    out = {
        "backbone": "backbone",            # relative to the release dir
        "attn": "sdpa",
        "speaker_anchor": anchor,
        "depth_grouped": bool(mc.get("depth_grouped", False)),
        "depth_ckpt_from": int(dcf) if dcf is not None else None,
    }
    # the engine loads with configs/s3_fresh_v1.yaml as an explicit override: make sure it builds the same model
    eng, _ = L.resolve_model_cfg(ckpt, config=str(ENGINE_DEFAULT_CONFIG))
    eng_anchor = L._reconcile_anchor(eng.get("speaker_anchor"), L.anchor_cfg_from_shapes(shapes))
    agrees = eng_anchor == anchor and bool(eng.get("depth_grouped", False)) == out["depth_grouped"]
    return out, src, agrees


def copy_backbone(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in BACKBONE_FILES:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            copied.append(name)
    if "config.json" not in copied or "tokenizer.json" not in copied:
        raise FileNotFoundError(f"{src} lacks config.json / tokenizer.json")
    bad = [p.name for p in dst.iterdir() if p.name.endswith(WEIGHT_SUFFIXES)]
    if bad:
        raise RuntimeError(f"weight files in {dst}: {bad} (the backbone dir must hold none)")
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", help="flat model.pt (keys model.<name>) or a dir containing one")
    ap.add_argument("out_dir")
    ap.add_argument("--backbone-src", help="backbone dir to take config.json + tokenizer files from (no weights copied)")
    ap.add_argument("--model-config", help="the run's resolved config yaml (default: found next to the checkpoint, "
                                           "else configs/s3_fresh_v1.yaml)")
    ap.add_argument("--max-shard-gb", type=float, default=5.0)
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="refuse to write when the target fs has less free")
    ap.add_argument("--source-name", help="label recorded in maya3_config.json (default: <run>/<step>)")
    ap.add_argument("--no-dedupe", action="store_true", help="store equal-valued (untied) tensors separately")
    A = ap.parse_args(argv)

    flat = L._flat_path(os.path.abspath(A.ckpt))
    if flat is None:
        sys.exit(f"{A.ckpt}: not a flat model.pt export")
    free = _free_bytes(A.out_dir)
    need = os.path.getsize(flat)
    if free < A.min_free_gb * 1e9 or free < need * 1.05:
        sys.exit(f"refusing to write: {_gb(free):.1f} GB free at {A.out_dir}, need >= {A.min_free_gb:.0f} GB "
                 f"and >= {_gb(need) * 1.05:.1f} GB")

    print(f"loading {flat} ({_gb(need):.2f} GB, mmap)", flush=True)
    sd = torch.load(flat, map_location="cpu", weights_only=True, mmap=True)
    bad = [k for k in sd if not k.startswith(L.MODEL_PREFIX)]
    if bad:
        sys.exit(f"non-model keys in {flat}: {bad[:4]}")
    sd = {k[len(L.MODEL_PREFIX):]: v for k, v in sd.items()}
    shapes = {k: v.shape for k, v in sd.items()}
    model_cfg, cfg_src, engine_agrees = resolve_release_model_cfg(flat, shapes, A.model_config)
    print(f"model config from {cfg_src}: {json.dumps(model_cfg)}  (engine default config agrees: {engine_agrees})")
    if not engine_agrees:
        print("WARNING: configs/s3_fresh_v1.yaml (maya3.engine's model_config default) would build a different "
              "model; pass model_config=<release>/maya3_config.json to the Engine", file=sys.stderr)

    stored, aliases = plan_storage(sd, dedupe_equal=not A.no_dedupe)
    stored_bytes = sum(v.numel() * v.element_size() for v in stored.values())
    print(f"{len(sd)} tensors -> {len(stored)} stored + {len(aliases)} aliases {aliases}; "
          f"{_gb(stored_bytes):.2f} GB", flush=True)

    out = Path(A.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in list(out.glob("model*.safetensors")) + list(out.glob("model.safetensors.index.json")):
        p.unlink()                                   # stale shards of a previous conversion
    from safetensors.torch import save_file
    groups = shard(stored, int(A.max_shard_gb * 1e9))
    meta = {"format": "pt", L.ALIASES_KEY: json.dumps(aliases, sort_keys=True),
            "maya3_format": FORMAT, "maya3_format_version": str(FORMAT_VERSION)}
    weight_map = {}
    names = ([L.SAFETENSORS_NAME] if len(groups) == 1 else
             [f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors" for i in range(len(groups))])
    for fname, keys in zip(names, groups):
        tmp = out / (fname + ".tmp")
        save_file({k: stored[k] for k in keys}, str(tmp), metadata=meta)
        os.chmod(tmp, 0o644)
        os.replace(tmp, out / fname)
        weight_map.update({k: fname for k in keys})
        print(f"  wrote {fname}: {len(keys)} tensors, {_gb(os.path.getsize(out / fname)):.2f} GB", flush=True)
    if len(groups) > 1:
        index = {"metadata": {"total_size": stored_bytes, L.ALIASES_KEY: aliases,
                              "maya3_format": FORMAT, "maya3_format_version": FORMAT_VERSION},
                 "weight_map": dict(sorted(weight_map.items()))}
        (out / L.SAFETENSORS_INDEX).write_text(json.dumps(index, indent=1) + "\n")

    prov = {}
    pj = Path(flat).parent / "provenance.json"
    if pj.is_file():
        prov = json.loads(pj.read_text())
    step_dir = Path(flat).parent
    dtypes = collections.Counter(str(v.dtype).replace("torch.", "") for v in sd.values())
    cfg = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_type": "maya3",
        "model": model_cfg,
        "key_prefix": L.MODEL_PREFIX,           # stripped in the files; load_s3 adds it back
        "aliases": aliases,
        "tensors": {"count": len(sd), "stored": len(stored), "dtypes": dict(dtypes),
                    "stored_bytes": stored_bytes,
                    "n_params_unique": int(sum(v.numel() for v in stored.values()))},
        "architecture": {"backbone": "Qwen3-1.7B (Qwen3ForCausalLM, built from backbone/config.json)",
                         "depth_decoder": "DepthV2, 31 levels", "speaker_anchor": "ECAPA-192 -> MLP"},
        "audio": {"sample_rate": 24000, "frame_rate_hz": 12.5, "n_codebooks": 32, "codebook_size": 2048,
                  "codec": "kyutai/mimi"},
        "speaker_encoder": {"repo": "speechbrain/spkrec-ecapa-voxceleb", "dim": 192},
        "languages": LANGS,
        "source": {"name": A.source_name or f"{step_dir.parent.parent.name}/{step_dir.name}",
                   "update": prov.get("source_update", L.checkpoint_update(flat)),
                   "format": prov.get("format"), "sha256_total": prov.get("total_hash"),
                   "hash_alg": prov.get("hash_alg"), "n_params": prov.get("n_params")},
        "converted_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "converter": {"torch": torch.__version__,
                      "safetensors": __import__("safetensors").__version__},
    }
    (out / L.RELEASE_CONFIG).write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"  wrote {L.RELEASE_CONFIG}")
    if A.backbone_src:
        copied = copy_backbone(A.backbone_src, out / "backbone")
        print(f"  backbone/: {copied}")
    print("CONVERTED " + json.dumps({"out_dir": str(out), "files": names, "stored_gb": round(_gb(stored_bytes), 3),
                                     "aliases": aliases}))


if __name__ == "__main__":
    main()
