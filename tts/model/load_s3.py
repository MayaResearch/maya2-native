"""The ONE checkpoint loader every eval/serving script uses.

WHY this module exists: a loader that builds an anchor-free `TTSModelV2`, asks DCP for exactly
the tensors that model happens to have, and throws the rest away will, pointed at an S3
checkpoint, load a *different model* — the speaker anchor is silently dropped — while printing
nothing. Every comparison made through such a path is invalid, and nothing in the output
says so. This module makes the two failure modes impossible:

  * the checkpoint carries `speaker_anchor.*` tensors and the constructed model has no
    anchor  -> RuntimeError, never a quiet anchor-free eval;
  * the constructed model has an anchor and the checkpoint has none                 -> RuntimeError;
  * any other missing/unexpected/mis-shaped model tensor                            -> RuntimeError.

It also keeps working for an S2 baseline checkpoint (no anchor keys, S2 config), because S2
and S3 checkpoints MUST evaluate through the same code path or the comparison is between two
loaders as much as between two checkpoints.

Accepted checkpoint sources:
  <run>/ckpt/<7-digit>/          DCP training checkpoint (`__k_0.distcp` + `extra.pt`),
                                 tts/train/loop.py:54 `save_ckpt`
  <dir>/model.pt or a .pt file   flat model-only export from
                                 scripts/s3_warmstart_extract.py (keys `model.<name>`,
                                 `provenance.json` beside it)
  <dir>/model.safetensors        release export from maya3/tools/convert_to_safetensors.py:
  <dir>/model.safetensors.index.json  one file or index-sharded, keys WITHOUT the `model.`
                                 prefix, tied tensors stored once (aliases in the metadata),
                                 `maya3_config.json` beside it (the resolved model config,
                                 used when no explicit `config=` is given)

Model kwargs (which anchor, which backbone) are resolved in this order, and the answer is
reported in the provenance dict so a log line always says what was built:
  1. `config=`            an explicit --model-config path (the operator overrides everything);
  2. `<run>/config.yaml`  the run's own resolved config, if the trainer wrote one next to the
                          checkpoint (not every trainer writes one; the lookup is here
                          so it works whenever one is present);
  3. `cfg=`              the eval config the script already loads (--config).
`model.backbone`/`model.attn` are ALWAYS taken from the caller's eval config when it has them:
an eval machine usually points the backbone at a local path in its own eval config, so a run
config's absolute training path would not exist there. Everything else (the anchor block) comes
from the resolution above, and is cross-checked against the checkpoint's own tensor shapes.

Training-only knobs are forced OFF for evaluation: `history_corruption` (it rewrites the
backbone's input history — a serving prompt must never be corrupted) and `depth_grad_scale`
(a gradient-only knob; its forward is the identity, tts_model_v2.py:208-211).

`model.depth_grouped` is NOT such a knob and is reconstructed from the run's config. It selects
DepthV2's grouped teacher-forced forward, which is a
CONSTRUCTOR decision (depth_v2.py:158) and only agrees with the per-level loop to
GROUPED_FP32_ATOL — so scoring a checkpoint trained with `depth_grouped: true` through the loop
measures a slightly different function than the one that was trained. It matters ONLY for
TEACHER-FORCED evaluation — the depth NLL / cb0 NLL numbers, e.g. clean-history
scoring, which goes through `TTSModelV2.forward` — because `DepthV2.generate()` has no grouped
path at all (it is autoregressive over levels by construction, depth_v2.py `generate`), so a
free-running/serving run is bit-identical either way. The flag is reported in the provenance
dict and therefore in the `CKPT_LOADED` line, which is the line to check
before reading any number.
"""
import json
import os

import torch

MODEL_PREFIX = "model."
ANCHOR_PREFIX = "speaker_anchor."
# nearest-first search for the run's resolved config, relative to the checkpoint dir:
# <ckpt>/, <ckpt>/../ (= <run>/ckpt/), <ckpt>/../../ (= <run>/, the documented home)
RUN_CONFIG_NAMES = ("config.yaml", "config.resolved.yaml")
RUN_CONFIG_DEPTH = 3
# 1-D per-span tables and (B, T) per-frame maps added by packing_v2._add_span_geometry
SPAN_KEYS = ("span_batch_row", "span_start", "span_len", "span_sentinel",
             "span_is_target")
FRAME_KEYS = ("frame_span_id", "frame_batch_row", "frame_target_mask",
              "frame_ref_mask")


# --------------------------------------------------------------------- checkpoint inspection
def _dcp_dir(path):
    """True if `path` is a directory holding DCP shards (`*.distcp`)."""
    return os.path.isdir(path) and any(f.endswith(".distcp")
                                       for f in os.listdir(path))


def _flat_path(path):
    """The flat model-only export at `path`, or None."""
    if os.path.isfile(path) and path.endswith(".pt"):
        return path
    cand = os.path.join(path, "model.pt")
    return cand if os.path.isfile(cand) else None


# ------------------------------------------------------------------ safetensors release format
SAFETENSORS_NAME = "model.safetensors"
SAFETENSORS_INDEX = "model.safetensors.index.json"
RELEASE_CONFIG = "maya3_config.json"
ALIASES_KEY = "maya3_aliases"      # metadata entry: json {alias name: stored name} for tied tensors


def _safetensors_files(path):
    """(release dir, [shard files], aliases) for a safetensors release export at `path`, or None.
    `path` is the release dir, its `model.safetensors`, or its `model.safetensors.index.json`."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isfile(path) and not path.endswith((".safetensors", ".safetensors.index.json")):
        return None
    index = os.path.join(d, SAFETENSORS_INDEX)
    single = os.path.join(d, SAFETENSORS_NAME)
    if os.path.isfile(index):
        with open(index) as fh:
            idx = json.load(fh)
        files = sorted(set(idx["weight_map"].values()))
        meta = idx.get("metadata") or {}
        aliases = meta.get(ALIASES_KEY) or {}
        if isinstance(aliases, str):
            aliases = json.loads(aliases)
        return d, [os.path.join(d, f) for f in files], dict(aliases)
    if os.path.isfile(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as fh:
            meta = fh.metadata() or {}
        return d, [single], dict(json.loads(meta.get(ALIASES_KEY, "{}")))
    return None


def _load_safetensors(st):
    """Flat `model.<name>` state dict from a safetensors release export (CPU; dtypes as stored).
    Aliased (tied) names point at the SAME tensor object as their stored name, as torch.load
    would give for a tied flat export."""
    from safetensors.torch import load_file
    _, files, aliases = st
    sd = {}
    for f in files:
        part = load_file(f, device="cpu")
        dup = set(part) & set(sd)
        if dup:
            raise RuntimeError(f"tensor(s) stored in more than one shard: {sorted(dup)[:4]}")
        sd.update(part)
    for alias, target in aliases.items():
        if alias in sd:
            raise RuntimeError(f"aliased tensor {alias} is also stored")
        if target not in sd:
            raise RuntimeError(f"alias {alias} -> {target}: target not in the export")
        sd[alias] = sd[target]
    return {MODEL_PREFIX + k: v for k, v in sd.items()}


def _release_config_path(path):
    """`maya3_config.json` of a safetensors release export at `path`, or None."""
    if not os.path.exists(path) or _dcp_dir(path) or _flat_path(path) is not None:
        return None                        # DCP / flat .pt keep their own resolution
    st = _safetensors_files(path)
    if st is None:
        return None
    p = os.path.join(st[0], RELEASE_CONFIG)
    return p if os.path.isfile(p) else None


def _inspect(path):
    """(kind, shapes, payload). `payload` is the loaded flat state dict for a `.pt` export
    (loaded ONCE — the file is ~9 GiB) and None for DCP, whose tensor bytes are not touched
    until the model exists."""
    if _dcp_dir(path):
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata
        md = FileSystemReader(path).read_metadata().state_dict_metadata
        shapes = {}
        for k, m in md.items():
            if not k.startswith(MODEL_PREFIX):
                continue                       # optim.* etc: not our business
            if not isinstance(m, TensorStorageMetadata):
                raise TypeError(f"non-tensor entry under model.* in {path}: "
                                f"{k} ({type(m).__name__})")
            shapes[k[len(MODEL_PREFIX):]] = torch.Size(m.size)
        if not shapes:
            raise ValueError(f"no {MODEL_PREFIX}* tensors in DCP checkpoint {path}")
        return "dcp", shapes, None
    flat = _flat_path(path)
    if flat is None:
        st = _safetensors_files(path)
        if st is not None:
            sd = _load_safetensors(st)
            return "safetensors", {k[len(MODEL_PREFIX):]: v.shape for k, v in sd.items()}, sd
        raise FileNotFoundError(
            f"{path} is neither a DCP checkpoint dir (*.distcp), a flat "
            f"model export (*.pt / <dir>/model.pt) nor a safetensors release export "
            f"(<dir>/{SAFETENSORS_NAME} or <dir>/{SAFETENSORS_INDEX})")
    sd = torch.load(flat, map_location="cpu", weights_only=True)
    bad = [k for k in sd if not k.startswith(MODEL_PREFIX)]
    if bad:
        raise RuntimeError(
            f"{flat} holds non-model keys ({bad[:4]}) — a flat warm-start export "
            f"carries `model.*` tensors only (scripts/s3_warmstart_extract.py)")
    return "flat", {k[len(MODEL_PREFIX):]: v.shape for k, v in sd.items()}, sd


def checkpoint_shapes(path):
    """(kind, {tensor name without `model.` -> torch.Size}) for a checkpoint.

    DCP reads METADATA only (no tensor bytes), so the anchor decision can be made before the
    2.90B-parameter model is constructed (measured on an S2 checkpoint: 727 tensors,
    2,899,683,328 params, namespaces backbone/depth/audio_in only)."""
    kind, shapes, _ = _inspect(path)
    return kind, shapes


def checkpoint_update(path):
    """Update number of the checkpoint, or None. DCP: the 7-digit directory name, which
    loop.py:55 writes and commits LATEST for last, so it is authoritative (s3_warmstart_extract
    .read_update cross-checks it against extra.pt at bank time). Flat: provenance.json's
    source_update."""
    base = os.path.basename(os.path.normpath(path))
    if len(base) == 7 and base.isdigit():
        return int(base)
    flat = _flat_path(path) if not _dcp_dir(path) else None
    if flat is None and not _dcp_dir(path):
        rc = _release_config_path(path)
        if rc is not None:
            with open(rc) as fh:
                return (json.load(fh).get("source") or {}).get("update")
    if flat is not None:
        prov = os.path.join(os.path.dirname(os.path.abspath(flat)),
                            "provenance.json")
        if os.path.isfile(prov):
            with open(prov) as fh:
                return json.load(fh).get("source_update")
    return None


def anchor_cfg_from_shapes(shapes):
    """Derive the SpeakerAnchor constructor kwargs from the checkpoint's own tensors, or None.

    The tensors define the module exactly — `fc1.weight` is (hidden, input_dim) — so a
    checkpoint can always be loaded correctly even where no config survived. `dropout` is
    training-only (anchors.py:80 guards on self.training) and irrelevant under .eval().
    """
    w = shapes.get(ANCHOR_PREFIX + "fc1.weight")
    if w is None:
        return None
    if len(w) != 2:
        raise RuntimeError(f"{ANCHOR_PREFIX}fc1.weight has shape {tuple(w)}, expected 2-D")
    return {"enabled": True, "hidden": int(w[0]), "input_dim": int(w[1])}


# ------------------------------------------------------------------------- config resolution
def find_run_config(ckpt_dir):
    """The run's resolved config next to a checkpoint, or None (nearest first)."""
    d = os.path.abspath(ckpt_dir if os.path.isdir(ckpt_dir)
                        else os.path.dirname(ckpt_dir))
    for _ in range(RUN_CONFIG_DEPTH):
        for name in RUN_CONFIG_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_model_cfg(ckpt_dir, cfg=None, config=None):
    """-> (model_cfg dict, source string). See the module docstring for the precedence."""
    if config is not None:
        mc = config if isinstance(config, dict) else _load_yaml(config)
        src = "override" if isinstance(config, dict) else f"override:{config}"
        return dict(mc.get("model", mc)), src
    rel = _release_config_path(ckpt_dir)
    if rel is not None:
        # a safetensors release export carries its own resolved config; a relative
        # `backbone` is relative to the release dir (`backbone/` in the HF repo layout)
        with open(rel) as fh:
            mc = dict(json.load(fh).get("model", {}))
        bb = mc.get("backbone")
        if bb and not os.path.isabs(bb):
            mc["backbone"] = os.path.join(os.path.dirname(rel), bb)
        return mc, f"release:{rel}"
    run = find_run_config(ckpt_dir)
    if run is not None:
        return dict((_load_yaml(run) or {}).get("model", {})), f"run:{run}"
    if cfg is not None:
        return dict(cfg.get("model", cfg)), "eval-config"
    return {}, "none"


def _reconcile_anchor(cfg_anchor, ckpt_anchor, require_anchor=None):
    """Decide the anchor kwargs, or raise. This is the whole point of the module.

    cfg_anchor  = the `model.speaker_anchor` block of the resolved config (None if absent).
    ckpt_anchor = derived from the checkpoint's tensors (None if it has no anchor).
    """
    cfg_on = bool((cfg_anchor or {}).get("enabled", False))
    ckpt_on = ckpt_anchor is not None
    if cfg_anchor is not None and cfg_on != ckpt_on:
        raise RuntimeError(
            f"speaker anchor mismatch: the resolved config says enabled={cfg_on} but the "
            f"checkpoint {'HAS' if ckpt_on else 'has NO'} {ANCHOR_PREFIX}* tensors. "
            f"Evaluating either way would compare a different model than the one that was "
            f"trained. Pass the run's own config with --model-config.")
    if require_anchor is not None and bool(require_anchor) != ckpt_on:
        raise RuntimeError(
            f"--require-anchor={require_anchor} but the checkpoint "
            f"{'HAS' if ckpt_on else 'has NO'} {ANCHOR_PREFIX}* tensors")
    if not ckpt_on:
        return None
    out = dict(ckpt_anchor)                       # geometry from the tensors (authoritative)
    if cfg_anchor:
        for k in ("input_dim", "hidden"):
            if k in cfg_anchor and int(cfg_anchor[k]) != int(out[k]):
                raise RuntimeError(
                    f"speaker_anchor.{k}: config says {cfg_anchor[k]}, the checkpoint's "
                    f"tensors say {out[k]}")
        out["dropout"] = float(cfg_anchor.get("dropout", 0.20))
    return out


# ----------------------------------------------------------------------------- construction
def build_eval_model(model_cfg, anchor_cfg, amortize_frac, depth_grouped=False):
    """Construct the eval-mode TTSModelV2. Kept as a module-level function so CPU tests can
    substitute a stand-in: the real backbone needs the Qwen weights and DepthV2 is 555M
    parameters, neither of which fits a <60 s CPU test.

    `depth_grouped` is the RUN's flag (config key `model.depth_grouped`), not an eval choice:
    DepthV2 takes it in the constructor precisely so no forward can flip it (depth_v2.py:158).
    """
    from .tts_model_v2 import TTSModelV2
    return TTSModelV2(backbone=model_cfg["backbone"],
                      attn=model_cfg.get("attn", "sdpa"),
                      amortize_frac=amortize_frac,
                      speaker_anchor=anchor_cfg,
                      history_corruption=None,     # training-only (see docstring)
                      depth_grad_scale=1.0,        # gradient-only knob
                      depth_grouped=depth_grouped)


def _strict_diff(model, ckpt_shapes, source):
    """Every model tensor must be in the checkpoint, every checkpoint tensor in the model, at
    the same shape. S2's loader asked for the model's keys only, so extra checkpoint tensors
    (the S3 anchor) vanished without a word."""
    have = {k: v.shape for k, v in model.state_dict().items()}
    missing = sorted(set(have) - set(ckpt_shapes))
    unexpected = sorted(set(ckpt_shapes) - set(have))
    bad = sorted(k for k in set(have) & set(ckpt_shapes)
                 if tuple(have[k]) != tuple(ckpt_shapes[k]))
    if missing or unexpected or bad:
        raise RuntimeError(
            f"checkpoint {source} does not match the constructed model:\n"
            f"  missing from the checkpoint ({len(missing)}): {missing[:8]}\n"
            f"  not in the model ({len(unexpected)}): {unexpected[:8]}\n"
            f"  shape mismatch ({len(bad)}): "
            f"{[(k, tuple(ckpt_shapes[k]), tuple(have[k])) for k in bad[:4]]}")


def load_model_for_eval(ckpt_dir, device=None, dtype=None, *, cfg=None,
                        config=None, amortize_frac=1.0, require_anchor=None,
                        verbose=True):
    """Load a checkpoint STRICTLY into an eval-mode TTSModelV2. -> (model, provenance).

    ckpt_dir       DCP checkpoint dir, a flat model.pt, or a dir containing one, or a
                   safetensors release dir (model.safetensors[.index.json] + maya3_config.json).
    device/dtype   applied after loading (None = leave on CPU / as constructed).
    cfg            the eval config dict the script already loaded (--config).
    config         explicit override: a path to the run's resolved config, or a dict.
    amortize_frac  1.0 = score every frame (the eval setting); irrelevant for generation.
    require_anchor True/False asserts the arm's expectation before anything is built.

    provenance: {source, format, update, config_source, backbone, anchor, anchor_cfg,
                 depth_grouped, depth_ckpt_from, n_tensors, n_params, amortize_frac}

    `depth_grouped` is taken from the resolved config's `model.depth_grouped` (the same
    precedence as the anchor block: --model-config > <run>/config.yaml > --config) and
    reported so a teacher-forced number can be attributed to the path that produced it —
    see the module docstring for why generation is unaffected.
    """
    src = os.path.abspath(ckpt_dir)
    kind, shapes, payload = _inspect(src)
    model_cfg, cfg_source = resolve_model_cfg(src, cfg=cfg, config=config)
    # eval machines point model.backbone at a local copy, so the CALLER's config wins for
    # the two path-like keys and the run config only supplies the feature blocks
    eval_model = (cfg or {}).get("model", {}) if cfg else {}
    for k in ("backbone", "attn"):
        if k in eval_model:
            model_cfg[k] = eval_model[k]
    if "backbone" not in model_cfg:
        raise RuntimeError(f"no model.backbone in the resolved config ({cfg_source}) — "
                           f"pass --config or --model-config")
    anchor_cfg = _reconcile_anchor(model_cfg.get("speaker_anchor"),
                                   anchor_cfg_from_shapes(shapes),
                                   require_anchor=require_anchor)
    # the run's depth path, not an eval choice: a run trained with
    # model.depth_grouped=true would be scored through the LOOP without this line
    depth_grouped = bool(model_cfg.get("depth_grouped", False))
    # `model.depth_ckpt_from` is reported but NOT applied: activation
    # checkpointing only exists in training (`self.training and
    # torch.is_grad_enabled()`, depth_v2.py:311/347), so it cannot change an
    # eval number. It is in the provenance so a scored checkpoint records the
    # memory recipe it was trained under.
    depth_ckpt_from = model_cfg.get("depth_ckpt_from")
    model = build_eval_model(model_cfg, anchor_cfg, amortize_frac, depth_grouped)
    _strict_diff(model, shapes, src)

    if kind == "dcp":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        # the model's OWN tensors are the load targets (S2's proven no-dist read path: the
        # 24 rank shards are reassembled by the reader), so nothing is left uninitialised
        sd = {MODEL_PREFIX + k: v for k, v in model.state_dict().items()}
        dcp.load(sd, storage_reader=FileSystemReader(src))
        payload = sd
    flat = {k[len(MODEL_PREFIX):]: v for k, v in payload.items()}
    model.load_state_dict(flat, strict=True)       # belt and braces after _strict_diff

    model.eval()
    if device is not None or dtype is not None:
        model.to(**({"device": device} if device is not None else {}),
                 **({"dtype": dtype} if dtype is not None else {}))
    prov = {"source": src, "format": kind, "update": checkpoint_update(src),
            "config_source": cfg_source, "backbone": model_cfg["backbone"],
            "anchor": anchor_cfg is not None, "anchor_cfg": anchor_cfg,
            "depth_grouped": depth_grouped,
            "depth_ckpt_from": (int(depth_ckpt_from)
                                if depth_ckpt_from is not None else None),
            "n_tensors": len(shapes),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "amortize_frac": amortize_frac}
    if verbose:
        print("CKPT_LOADED " + json.dumps(prov, sort_keys=True), flush=True)
    return model, prov


# ------------------------------------------------------------------- serving prompt geometry
def truncate_prompt_batch(b, cut, require_target_span=False):
    """Cut a collated batch at column `cut` and SLICE THE SPAN GEOMETRY WITH IT.

    WHY: the serving/eval prompt is one collate_v2 batch cut at the
    target's first audio column. The old inline loop kept every tensor whose dim-1 was T and
    passed everything else through untouched — which left the 1-D `span_*` tables describing
    the FULL example: spans that were cut away still appeared, and `span_start`/`span_len`
    still counted frames that are no longer in the prompt. `compose_inputs` would then add the
    speaker anchor at sentinel columns outside the prompt (index_put out of range, or silently
    the wrong column once padding shifts).

    Spans are kept iff their `<|audio|>` SENTINEL survives the cut, not their frames: the
    target span's sentinel sits at `cut - 1` (it is the last prompt position, the one whose
    hidden state generates frame 0) while all of its frames are cut away. Dropping it would
    serve the target span with no anchor at all — exactly the training/serving mismatch the
    anchor exists to remove. `span_len` is clipped to the frames that remain.

    A batch with no geometry (a collate from before span geometry was added) passes through unchanged.
    """
    T = int(b["kind"].shape[1])
    cut = int(cut)
    if not 0 < cut <= T:
        raise ValueError(f"cut {cut} outside (0, {T}]")
    pb = {}
    for k, v in b.items():
        if k in SPAN_KEYS:
            continue
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == T:
            pb[k] = v[:, :cut].clone()
        else:
            pb[k] = v
    if "span_sentinel" not in b:
        return pb
    sent = b["span_sentinel"]
    idx = ((sent >= 0) & (sent < cut)).nonzero().flatten()
    start = b["span_start"][idx].clamp(max=cut)
    end = (b["span_start"][idx] + b["span_len"][idx]).clamp(max=cut)
    pb["span_batch_row"] = b["span_batch_row"][idx].clone()
    pb["span_sentinel"] = sent[idx].clone()
    pb["span_start"] = start
    pb["span_len"] = (end - start).clamp(min=0)
    pb["span_is_target"] = b["span_is_target"][idx].clone()
    if "frame_span_id" in pb:
        # a surviving frame can only belong to a surviving span (sentinel < start <= frame),
        # so the remap is total; -1 stays -1
        remap = torch.full((int(sent.numel()),), -1, dtype=torch.long)
        remap[idx] = torch.arange(idx.numel(), dtype=torch.long)
        f = pb["frame_span_id"]
        pb["frame_span_id"] = torch.where(f >= 0, remap[f.clamp(min=0)], f)
    if require_target_span and not bool(pb["span_is_target"].any()):
        raise RuntimeError(
            "the truncated prompt keeps no TARGET span sentinel — the speaker anchor would "
            "never reach the position that generates frame 0 (packing/target_start bug)")
    return pb
