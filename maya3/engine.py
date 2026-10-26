"""Maya3 streaming engine: one GPU-owner thread, continuous batching over seats, everything hot CUDA-graphed.

Per tick (one 80 ms Mimi frame for every active seat):
  1. admit   : new requests get a slot; the voice's cached prefix KV (reference span, constant per voice) is copied into
               the slot and only the target text is prefilled -- INSIDE the same graphed backbone step as the decode
               rows (inline prefill, serve/fast_backbone.StaticBackbone.step_mixed).  Plain requests (no prefix) are
               prefilled entirely inline.  So a new request's first frame is produced in its admission tick.
  2. cb0     : controlled stop (tau 0.35 / hazard 0.6 / argmax AEOS, min frames) + cb0 sampling on the GPU, with the
               per-seat AEOS bias of the EOS-boost loop guard.
  3. depth   : 31 levels in one CUDA graph (serve/fast_depth.FastDepth), fp16 by default (closer to fp32 greedy than bf16).
  4. mimi    : one-frame static-state Mimi decode graph (serve/fast_codec.StaticMimi) -> int16 PCM on the GPU.
  5. ONE device->host copy (stop codes, cb0, PCM) and ONE sync; then guards + delivery on the host.

Differences from the offline reference decoder used for the quality evals (FastBatch): prefix-KV reuse + inline
prefill instead of a full eager HF prefill per request, vectorised loop guard, one sync per tick, PCM as int16.
"""
from __future__ import annotations

import collections
import dataclasses
import gc
import itertools
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from maya3 import guards as G  # noqa: E402
from maya3.prompt import DEFAULT_BUILDER, MODES, Voice, VoiceRegistry  # noqa: E402

SR = 24000
FRAME = 1920
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)
ROW_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512)
REASONS = {1: "aeos", 2: "aeos_argmax", 3: "aeos_tau", 4: "aeos_hazard"}


@dataclasses.dataclass
class EngineConfig:
    max_seats: int = 16
    capacity: int = 2048                 # backbone KV tokens per seat (prompt + generated frames)
    depth_dtype: str = "fp16"            # fp16 | bf16 | fp32
    temp: float = 0.7                    # depth sampler
    top_k: int = 64
    # cb0 sampler defaults = best setting of a sampler sweep (presence penalty 0.8 over the last 12 frames +
    # temperature 0.9; best stability objective, raw-text no-stop rate 0.79 -> 0.07 %)
    cb0_temp: float = 0.9
    cb0_top_k: int = 64
    cb0_top_p: float = 1.0
    cb0_rep: float = 0.8                 # presence penalty: logit -= rep for every code seen in the last cb0_rep_win frames
    cb0_rep_win: int = 12
    hard_cap: int = 1500                 # frames (120 s): absolute per-chunk ceiling
    max_inline_rows: int = 384           # prefill rows folded into one tick (protects running seats' tick time)
    max_admit_per_tick: int = 8
    busy_frac: float = 0.75              # above this seat occupancy, admit at most busy_admit per tick (smooths tick time)
    busy_admit: int = 4
    seed: int = 20261024
    guards: G.GuardConfig = dataclasses.field(default_factory=G.GuardConfig)
    gap_s: float = 0.15                  # silence between the text pieces of one request
    chunk_s: float = 20.0                # long text is split into pieces of about this many seconds
    use_graphs: bool = True
    fuse_proj: bool = False              # backbone: one GEMM for q|k|v and one for gate|up per layer
    fuse_rope: bool = True               # backbone: fused q/k RMSNorm + RoPE + KV-cache write (Triton; implies fuse_proj)
    fused_norm: bool = False             # backbone: F.rms_norm instead of the HF RMSNorm module
    fused_depth: bool = True             # depth: Triton add+LayerNorm and one-kernel top-k sampler
    compile_depth: bool = False          # Inductor-fuse the depth frame body before capture (measured: no gain; off)
    free_reference_depth: bool = True    # drop the fp32 reference depth stack once FastDepth holds its own copy (-2.2 GB)
    verify_prefix: bool = False          # debug: assert the cached prefix embedding equals the recomputed one


class Request:
    """One synthesis request.  on_event(kind, payload) is called from the engine thread:
         ('audio', np.int16 [n])            PCM 24 kHz mono, one or more frames
         ('end', dict)                      reason, frames, audio_s, ttfa_s, pieces, stops
         ('error', dict)                    code, error
    It must be cheap and non-blocking (put into a queue)."""
    _ids = itertools.count()

    def __init__(self, pieces, lang, voice: Voice, on_event, max_frames=None, rid=None, sampling=None):
        self.id = rid or f"r{next(self._ids)}"
        self.sampling = dict(sampling or {})
        self.pieces = list(pieces)
        self.lang = lang
        self.voice = voice
        self.on_event = on_event
        self.max_frames = max_frames
        self.cancelled = False
        self.t_submit = time.perf_counter()
        self.t_first = None
        self.k = 0                 # current piece
        self.frames = 0
        self.stops = []
        self.done = False

    def cancel(self):
        self.cancelled = True


def _check_sampling(sp):
    if not sp:
        return {}
    allowed = {"seed", "temperature", "top_k", "top_p", "repetition_penalty"}
    bad = set(sp) - allowed
    if bad:
        raise ValueError(f"unknown sampling keys {sorted(bad)}; allowed {sorted(allowed)}")
    out = {k: v for k, v in sp.items() if v is not None}
    if "temperature" in out and not 0.05 <= float(out["temperature"]) <= 2.0:
        raise ValueError("temperature must be in [0.05, 2.0]")
    if "top_k" in out and not 1 <= int(out["top_k"]) <= 2049:
        raise ValueError("top_k must be in [1, 2049]")
    if "top_p" in out and not 0.0 < float(out["top_p"]) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if "repetition_penalty" in out and not 0.0 <= float(out["repetition_penalty"]) <= 5.0:
        raise ValueError("repetition_penalty must be in [0, 5]")
    if "seed" in out:
        out["seed"] = int(out["seed"])
    return out


class _Seat:
    __slots__ = ("req", "slot", "frames", "cap", "loose_cap", "minimum", "suspend", "boost_at", "n_boost", "hist", "new")

    def __init__(self, req, slot, cap, loose_cap, minimum, suspend, hist):
        self.req, self.slot, self.cap, self.loose_cap, self.minimum, self.suspend = req, slot, cap, loose_cap, minimum, suspend
        self.frames = 0
        self.boost_at = None
        self.n_boost = 0
        self.hist = np.full(hist, -1, np.int64)
        self.new = True


class Engine:
    def __init__(self, ckpt, backbone_dir, mimi_dir, config: EngineConfig = None, device="cuda", registry=None,
                 builder=DEFAULT_BUILDER, model_config=None, log=print):
        from transformers import MimiModel
        from tts.data.packing_v2 import PackerV2
        from tts.model.load_s3 import load_model_for_eval
        from scripts import s2_serve_controls as S
        self.S = S
        self.cfg = cfg = config or EngineConfig()
        self.device = device
        self.log = log
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        t0 = time.perf_counter()
        mcfg = {"model": {"backbone": str(backbone_dir), "attn": "sdpa"}, "data": {"max_tokens": 8192}}
        self.model, self.provenance = load_model_for_eval(str(ckpt), device=device, dtype=torch.bfloat16, cfg=mcfg,
                                                          config=str(model_config or ROOT / "configs/s3_fresh_v1.yaml"),
                                                          amortize_frac=1.0, require_anchor=True, verbose=False)
        self.model.depth.float()
        self.packer = PackerV2(self.model.backbone.tokenizer, self.model.backbone.token_ids, max_tokens=8192)
        self.prior = S.load_prior(str(ROOT / "configs/text_frames_prior.json"))
        self.dur = G.DurationPrior()
        self.mimi = MimiModel.from_pretrained(str(mimi_dir)).to(device).eval()
        self.load_s = time.perf_counter() - t0
        self.registry = registry or VoiceRegistry()
        self.builder = builder
        self._build_fast()
        self._pending = collections.deque()
        self._calls = collections.deque()     # callables run on the engine thread between ticks (voice prep, encodes)
        self._cv = threading.Condition()
        self._stop = False
        self._thread = None
        self.stats = {"ticks": 0, "admitted": 0, "completed": 0, "prefix_hits": 0, "prefix_misses": 0, "fallback_prefill": 0}
        self.tick_ms = collections.deque(maxlen=20000)
        self.phase_ms = {}
        self.failed = None

    # ------------------------------------------------------------------ fast components
    def _build_fast(self):
        from serve.fast_backbone import StaticBackbone
        from serve.fast_depth import FastDepth
        from serve.fast_codec import StaticMimi
        cfg, dev = self.cfg, self.device
        n = cfg.max_seats
        buckets = tuple(b for b in BUCKETS if b < n) + (next(b for b in BUCKETS if b >= n),)
        rows = tuple(b for b in ROW_BUCKETS if b <= buckets[-1] + cfg.max_inline_rows)
        rows = rows + tuple(b for b in ROW_BUCKETS if b > rows[-1])[:1]
        self.buckets, self.row_buckets, self.max_rows = buckets, rows, rows[-1]
        ctx = tuple(c for c in (256, 512, 768, 1024, 1536, 2048, 3072) if c < cfg.capacity) + (cfg.capacity,)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.depth_dtype]
        self.backbone = StaticBackbone(self.model.backbone, n, capacity=cfg.capacity, batch_buckets=rows, ctx_buckets=ctx,
                                       attention="triton", use_graphs=cfg.use_graphs,
                                       fuse_proj=cfg.fuse_proj, fused_norm=cfg.fused_norm, fuse_rope=cfg.fuse_rope)
        if cfg.compile_depth:
            import torch._dynamo as _dyn
            _dyn.config.cache_size_limit = max(64, _dyn.config.cache_size_limit)
        self.depth = FastDepth(self.model, n, dtype=dt, batch_buckets=buckets, seed=cfg.seed + 1, temp=cfg.temp, top_k=cfg.top_k,
                               compile=cfg.compile_depth, fused=cfg.fused_depth)
        self.codec = StaticMimi(self.mimi, n, batch_buckets=buckets)
        if cfg.free_reference_depth and getattr(self.model.depth, "groups", None) is not None:
            d = self.model.depth   # only spk_proj_* / null_mem are used after this (speaker memory); FastDepth has its own copy
            for name in ("groups", "cross", "heads", "code_lo", "code_up", "in_proj", "level_emb", "norm"):
                if hasattr(d, name):
                    setattr(d, name, None)
            torch.cuda.empty_cache()
        m = n + 1
        D = self.model.backbone.d_model
        self.h = torch.zeros(m, D, device=dev, dtype=torch.bfloat16)
        self.feedback = torch.zeros_like(self.h)
        self.frames_t = torch.zeros(m, device=dev, dtype=torch.long)
        self.min_t = torch.zeros(m, device=dev, dtype=torch.long)
        self.hazard_t = torch.zeros(m, device=dev, dtype=torch.float32)
        self.aeos_bias = torch.zeros(m, device=dev, dtype=torch.float32)
        # per-slot sampler settings and RNG stream (per-request seed => reproducible output at a fixed batch composition)
        self.temp_t = torch.full((m,), cfg.cb0_temp, device=dev, dtype=torch.float32)
        self.topk_t = torch.full((m,), cfg.cb0_top_k, device=dev, dtype=torch.long)
        self.topp_t = torch.full((m,), cfg.cb0_top_p, device=dev, dtype=torch.float32)
        self.rep_t = torch.full((m,), cfg.cb0_rep, device=dev, dtype=torch.float32)
        self.seed_t = torch.zeros(m, device=dev, dtype=torch.int32)
        self.ctr_t = torch.zeros(m, device=dev, dtype=torch.int32)
        self.hist_t = torch.full((m, max(1, cfg.cb0_rep_win)), -1, device=dev, dtype=torch.long)
        import random as _random
        self._seed_rng = _random.Random(cfg.seed)
        self.generator = torch.Generator(device=dev).manual_seed(cfg.seed)
        self.free = list(range(n))[::-1]
        self.active = []
        self._any_rep = False      # some active request overrides the presence penalty (host-side flag, no sync)
        self._any_top_p = cfg.cb0_top_p < 1.0
        # one pinned host buffer per tick: [B, 2 + FRAME/4] int64 view = stop code, cb0, PCM int16 packed
        self._host = torch.empty(n, 2 + FRAME // 4, dtype=torch.int64).pin_memory()
        self._evt = torch.cuda.Event()
        self.warm = False

    def warmup(self):
        t0 = time.perf_counter()
        self.backbone.warmup()
        self.depth.warmup()
        self.codec.warmup()
        torch.cuda.synchronize()
        # exercise the admission paths once (prefix KV build, inline prefill, cb0 kernels)
        v = self.registry.get("plain")
        for _ in range(2):
            done = threading.Event()
            r = Request(["Hello, this is a warmup."], "en", v, lambda k, p, d=done: d.set() if k != "audio" else None, max_frames=3)
            self._pending.append(r)
            while self._pending or self.active:
                self._tick()
        gc.collect()
        gc.freeze()                       # long-lived objects (model, graphs) leave the GC's scan set: no multi-ms pauses
        g0, g1, g2 = gc.get_threshold()
        gc.set_threshold(max(g0, 50000), g1, g2)
        self.warm = True
        self.warmup_s = time.perf_counter() - t0
        return self.warmup_s

    # ------------------------------------------------------------------ voices
    @torch.inference_mode()
    def prepare_voice(self, voice: Voice):
        """Pack the voice's prefix, compute speaker memory and the prefix backbone KV (once per voice)."""
        with voice.lock:
            if voice.mem is not None:
                return voice
            use_icl, use_mem = MODES[voice.mode]
            builder = voice.builder or self.builder
            voice.prefix = builder.prefix(self.packer, voice)
            voice.spk_refs = (np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None
                              else np.zeros((1, 192), np.float32))
            refs_t = torch.as_tensor(voice.spk_refs, device=self.device)[None]
            voice.mem = self.model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not use_mem], device=self.device))
            voice.rmask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
            voice.ts = int(voice.prefix["tokens"].shape[0]) if voice.prefix is not None else 0
            if voice.ts:
                # embed a dummy request to get the prefix embeddings exactly as the full prompt composes them
                ex = builder.example(self.packer, voice, "a", voice.ref_lang or "en")
                prompt, n = self.S.prompt_from_example(ex, self.model, self.device, torch.bfloat16, 32)
                emb = prompt[:, :voice.ts].to(torch.bfloat16)
                h, cache = self.model.backbone(emb, use_cache=True)
                voice.kv = [(l.keys[:, :, :voice.ts].clone(), l.values[:, :, :voice.ts].clone()) for l in cache.layers]
                voice.emb = emb.clone()
                del cache
            return voice

    # ------------------------------------------------------------------ public (thread-safe)
    def split(self, text, lang):
        return self.S.split_text(text, lang, self.prior, self.cfg.chunk_s) if self.cfg.chunk_s > 0 else [text]

    def submit(self, text, lang, voice="plain", on_event=None, max_frames=None, rid=None, split=True, sampling=None):
        """sampling: optional {seed, temperature, top_k, top_p, repetition_penalty} for codebook 0 (the depth decoder keeps the
        engine's temperature/top-k but draws from the request's seeded stream). Same seed + same text + same voice => same audio
        when the request runs alone (batched bf16 numerics may differ slightly with batch composition)."""
        sampling = _check_sampling(sampling)
        if isinstance(voice, str):
            voice = self.registry.get(voice)
        if not isinstance(text, (list, tuple)):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            text = self.split(text.strip(), lang) if split else [text.strip()]
        req = Request(text, lang, voice, on_event or (lambda k, p: None), max_frames=max_frames, rid=rid, sampling=sampling)
        with self._cv:
            if self.failed:
                raise RuntimeError(f"engine failed: {self.failed}")
            self._pending.append(req)
            self._cv.notify()
        return req

    def call(self, fn, timeout=120):
        """Run fn() on the engine thread (between ticks) and return its result; direct call if the loop is not running."""
        if self._thread is None or threading.current_thread() is self._thread:
            with torch.inference_mode():
                return fn()
        box = {}
        ev = threading.Event()
        with self._cv:
            self._calls.append((fn, box, ev))
            self._cv.notify()
        if not ev.wait(timeout):
            raise TimeoutError("engine call timed out")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _run_calls(self):
        while self._calls:
            with self._cv:
                fn, box, ev = self._calls.popleft()
            try:
                box["result"] = fn()
            except Exception as exc:
                box["error"] = exc
            ev.set()

    def start(self):
        if self._thread is None:
            if not self.warm:
                self.warmup()
            self._thread = threading.Thread(target=self._loop, name="maya3-engine", daemon=True)
            self._thread.start()
        return self

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def health(self):
        recent = list(self.tick_ms)[-2000:]
        rs = sorted(recent)
        return {"ready": self.warm and self.failed is None, "failed": self.failed, "active": len(self.active),
                "pending": len(self._pending), "max_seats": self.cfg.max_seats, **self.stats,
                "tick_ms_p50": rs[len(rs) // 2] if rs else None, "tick_ms_p99": rs[int(len(rs) * .99)] if rs else None,
                "phase_ms": self.phase_ms, **self._mem()}

    def _mem(self):
        try:
            return {"mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2), "mem_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)}
        except Exception:
            return {}

    # ------------------------------------------------------------------ loop
    def _loop(self):
        try:
            with torch.inference_mode():
                while True:
                    with self._cv:
                        while not self._stop and not self._pending and not self.active and not self._calls:
                            self._cv.wait(0.5)
                        if self._stop:
                            break
                    self._run_calls()
                    self._tick()
        except BaseException as exc:  # pragma: no cover
            import traceback
            traceback.print_exc()
            self.failed = f"{type(exc).__name__}: {exc}"
        finally:
            with self._cv:
                jobs = list(self._pending) + [s.req for s in self.active]
                self._pending.clear()
            for r in jobs:
                self._emit(r, "error", {"code": "unavailable", "error": self.failed or "engine stopped"})
            self.active = []

    def _emit(self, req, kind, payload):
        try:
            req.on_event(kind, payload)
        except Exception:  # a broken consumer must not kill the engine
            req.cancelled = True

    # ------------------------------------------------------------------ admission
    def _plan(self, req):
        """Build the prompt for req's current piece. Returns (prompt_emb [1,T,D] bf16, ts, cap, loose_cap, minimum, suspend)."""
        S, cfg = self.S, self.cfg
        voice = req.voice
        if voice.mem is None:
            self.prepare_voice(voice)
        text = req.pieces[req.k]
        builder = voice.builder or self.builder
        ex = builder.example(self.packer, voice, text, req.lang)
        prompt, n = S.prompt_from_example(ex, self.model, self.device, torch.bfloat16, 32)
        prompt = prompt.to(torch.bfloat16)
        ef, er = S.expected_frames(self.prior, req.lang, len(text))
        loose = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        cap = loose
        g = cfg.guards
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, req.lang, g.dur_k, g.dur_margin_s)))
        if req.max_frames:   # total over all pieces of the request
            cap = min(cap, max(1, int(req.max_frames) - req.frames))
        minimum = max(4, int(0.5 * ef))
        if g.enabled and g.loop and g.aware_v2:
            suspend = G.suspend_windows_v2(text, req.lang, self.dur)
        else:
            suspend = G.tag_windows(text, req.lang, self.dur, g.tag_hold_s) if (g.enabled and g.loop and g.tag_aware and "[" in text) else ()
        return prompt, int(n), cap, loose, minimum, suspend

    def _bind(self, req, cap, loose, minimum, suspend):
        slot = self.free.pop()
        v = req.voice
        self.depth.set_speaker(slot, v.mem, v.rmask)
        self.codec.reset(slot)
        self.frames_t[slot] = 0
        self.min_t[slot] = minimum
        self.hazard_t[slot] = 0
        self.aeos_bias[slot] = 0
        sp, cfg = req.sampling, self.cfg
        if "seed" in sp and sp["seed"] is not None:   # each piece of a multi-piece request gets its own derived stream
            seed = (int(sp["seed"]) * 1000003 + req.k * 7919) % (2 ** 31 - 1)
        else:
            seed = self._seed_rng.randrange(1, 2 ** 31 - 1)
        self.seed_t[slot] = seed
        self.ctr_t[slot] = 0
        self.depth.row_seed[slot] = (seed * 69069 + 1) % (2 ** 31 - 1)
        self.depth.row_ctr[slot] = 0
        self.temp_t[slot] = float(sp.get("temperature") or cfg.cb0_temp)
        self.topk_t[slot] = int(sp.get("top_k") or cfg.cb0_top_k)
        self.topp_t[slot] = float(sp.get("top_p") or cfg.cb0_top_p)
        self.rep_t[slot] = float(sp["repetition_penalty"] if sp.get("repetition_penalty") is not None else cfg.cb0_rep)
        self.hist_t[slot] = -1
        if float(sp.get("top_p") or 1.0) < 1.0:
            self._any_top_p = True
        if sp.get("repetition_penalty"):
            self._any_rep = True
        return _Seat(req, slot, cap, loose, minimum, suspend, self.cfg.guards.hist)

    def _admit(self):
        """Pop pending requests into seats. Returns (new seats, inline prefills [(slot, x)])."""
        new, prefills, rows = [], [], len(self.active)
        cap_admit = self.cfg.max_admit_per_tick
        if len(self.active) > self.cfg.busy_frac * self.cfg.max_seats:
            cap_admit = min(cap_admit, self.cfg.busy_admit)
        while self._pending and self.free and len(new) < cap_admit:
            with self._cv:
                req = self._pending[0]
                if req.cancelled:
                    self._pending.popleft()
                    self._emit(req, "error", {"code": "cancelled", "error": "cancelled"})
                    continue
            try:
                prompt, n, cap, loose, minimum, suspend = self._plan(req)
            except Exception as exc:
                with self._cv:
                    self._pending.popleft()
                self._emit(req, "error", {"code": "invalid_request", "error": f"{type(exc).__name__}: {exc}"})
                continue
            if n + cap + 2 > self.cfg.capacity:
                cap = self.cfg.capacity - n - 2
                if cap < 8:
                    with self._cv:
                        self._pending.popleft()
                    self._emit(req, "error", {"code": "invalid_request",
                                              "error": f"prompt of {n} tokens leaves no room in KV capacity {self.cfg.capacity}"})
                    continue
            v = req.voice
            ts = v.ts
            tokens = n - ts
            inline = 0 < tokens and rows + tokens <= self.max_rows
            if not inline and new:
                break  # row budget for this tick is used; admit next tick
            with self._cv:
                self._pending.popleft()
            seat = self._bind(req, cap, loose, minimum, suspend)
            if self.cfg.verify_prefix and ts:
                assert torch.equal(prompt[:, :ts], v.emb), "cached prefix embedding differs from the recomputed prompt"
            if inline:
                if ts:
                    self.backbone.load_prefix(seat.slot, v.kv, ts)
                    self.stats["prefix_hits"] += 1
                else:
                    self.backbone.lengths[seat.slot] = 0
                    self.backbone.lengths_cpu[seat.slot] = 0
                prefills.append((seat.slot, prompt[0, ts:]))
                rows += tokens
            else:   # prompt longer than one tick's row budget: eager HF prefill (from the cached prefix if any)
                from serve.cache import cache_from_owned_tensors
                if ts:
                    h, cache = self.model.backbone(prompt[:, ts:], past_key_values=cache_from_owned_tensors(v.kv), use_cache=True)
                else:
                    h, cache = self.model.backbone(prompt, use_cache=True)
                self.backbone.load_slot(seat.slot, cache, n)
                self.h[seat.slot] = h[0, -1]
                del cache
                self.stats["fallback_prefill"] += 1
                seat.new = "eager"   # already has its hidden state; it decodes this tick without a backbone row
            new.append(seat)
            self.stats["admitted"] += 1
        return new, prefills

    # ------------------------------------------------------------------ one tick
    @torch.inference_mode()
    def _tick(self):
        t0 = time.perf_counter()
        cfg = self.cfg
        # cancelled seats leave before any work
        keep = []
        for s in self.active:
            if s.req.cancelled:
                self._release(s)
                self._emit(s.req, "error", {"code": "cancelled", "error": "cancelled"})
            else:
                keep.append(s)
        self.active = keep
        if self._pending and any(r.cancelled for r in self._pending):
            with self._cv:
                gone = [r for r in self._pending if r.cancelled]
                self._pending = collections.deque(r for r in self._pending if not r.cancelled)
            for r in gone:
                self._emit(r, "error", {"code": "cancelled", "error": "cancelled"})
        new, prefills = self._admit()
        t1 = time.perf_counter()
        dec = [s.slot for s in self.active]
        if dec or prefills:
            idx = torch.tensor(dec, device=self.device, dtype=torch.long) if dec else None
            if prefills:
                h, last = self.backbone.step_mixed(dec, self.feedback[idx] if dec else None, prefills)
                for (slot, _), hid in zip(prefills, last):
                    self.h[slot] = hid
            else:
                h = self.backbone.step(dec, self.feedback[idx])
            if dec:
                self.h[idx] = h
        self.active = self.active + new
        if not self.active:
            return
        t2 = time.perf_counter()
        seats = self.active
        slots = [s.slot for s in seats]
        B = len(slots)
        idx = torch.tensor(slots, device=self.device, dtype=torch.long)
        h = self.h[idx]
        code, cb0 = self._cb0(h, idx)
        codes, fb = self.depth(slots, h, cb0)
        self.feedback[idx] = fb.to(self.feedback.dtype)
        pcm = self.codec.decode(slots, codes)
        pcm16 = (pcm.clamp(-1, 1) * 32767).round().to(torch.int16)
        packed = torch.cat([code[:, None], cb0[:, None], pcm16.view(torch.int64).view(B, FRAME // 4)], 1)
        host = self._host[:B]
        host.copy_(packed, non_blocking=True)
        self._evt.record()
        self._evt.synchronize()
        t3 = time.perf_counter()
        hv = host.numpy()
        stop_l = hv[:, 0].tolist()
        cb0_l = hv[:, 1]
        pcm_np = hv[:, 2:].view(np.int16)
        now = time.perf_counter()
        finished = []
        g = cfg.guards
        check = []
        for i, s in enumerate(seats):
            req = s.req
            s.new = False
            if stop_l[i]:
                finished.append((s, REASONS[stop_l[i]]))
                continue
            s.frames += 1
            req.frames += 1
            if req.t_first is None:
                req.t_first = now
            self._emit(req, "audio", pcm_np[i].copy())
            s.hist[:-1] = s.hist[1:]
            s.hist[-1] = cb0_l[i]
            if s.frames >= s.cap:
                if s.req.max_frames and s.req.frames >= s.req.max_frames:
                    reason = "max_frames"          # the caller's limit, not a model failure
                elif g.enabled and s.cap < s.loose_cap:
                    reason = "cap_guard"
                else:
                    reason = "cap"
                finished.append((s, reason))
            elif g.enabled and g.loop and s.frames >= g.start and s.frames % g.every == 0 and \
                    not any(a <= s.frames <= b for a, b in s.suspend):
                check.append(i)
        if check:
            W = np.stack([seats[i].hist for i in check])
            n = np.array([seats[i].frames for i in check])
            hits = G.loop_detect_batch(W, n, g)
            fin_ids = {id(f[0]) for f in finished}
            for i, hit in zip(check, hits):
                s = seats[i]
                if id(s) in fin_ids:
                    continue
                if hit is not None and g.mode == "boost":
                    if s.boost_at is None:
                        s.boost_at = s.frames
                        s.n_boost += 1
                        self.aeos_bias[s.slot] = g.boost
                    if s.frames - s.boost_at >= g.backstop:
                        finished.append((s, "loop_backstop"))
                elif hit is not None:
                    finished.append((s, "loop_" + hit.split(":")[0]))
                elif s.boost_at is not None:
                    self.aeos_bias[s.slot] = 0.0
                    s.boost_at = None
        if finished:
            gone = set()
            for s, reason in finished:
                gone.add(id(s))
                self._release(s)
                self._finish_piece(s, reason)
            self.active = [s for s in self.active if id(s) not in gone]
        t4 = time.perf_counter()
        self.stats["ticks"] += 1
        self.tick_ms.append((t4 - t0) * 1e3)
        self.phase_ms = {"admit": round((t1 - t0) * 1e3, 2), "backbone": round((t2 - t1) * 1e3, 2),
                         "frame": round((t3 - t2) * 1e3, 2), "host": round((t4 - t3) * 1e3, 2), "batch": B,
                         "prefill_rows": sum(int(x.shape[0]) for _, x in prefills)}

    def _release(self, s):
        self.backbone.free(s.slot)
        self.aeos_bias[s.slot] = 0
        self.free.append(s.slot)

    def _finish_piece(self, s, reason):
        req = s.req
        req.stops.append({"reason": reason, "frames": s.frames, "cap": s.cap, "n_boost": s.n_boost})
        req.k += 1
        if req.k < len(req.pieces) and not req.cancelled and not (req.max_frames and req.frames >= req.max_frames):
            self._emit(req, "audio", np.zeros(int(self.cfg.gap_s * SR), np.int16))
            with self._cv:
                self._pending.appendleft(req)   # next piece goes first: the stream continues without queueing
            return
        req.done = True
        self.stats["completed"] += 1
        self._emit(req, "end", {"reason": req.stops[-1]["reason"], "frames": req.frames, "audio_s": req.frames * FRAME / SR,
                                "ttfa_s": (req.t_first - req.t_submit) if req.t_first else None,
                                "wall_s": time.perf_counter() - req.t_submit, "pieces": len(req.pieces), "stops": req.stops})

    def _cb0(self, h, idx):
        """Controlled stop + cb0 sampling (same rule as the reference decoder and ServeSession.speak(stop='controlled')) for all seats, with
        per-seat temperature / top-k / top-p / presence penalty and a per-request Philox stream."""
        from tts.tokens import AEOS_CLASS
        from serve.fast_depth_kernels import row_exponential
        logits = self.model.backbone.cb0_logits(h).float() / self.temp_t[idx, None]
        logits[:, AEOS_CLASS] += self.aeos_bias[idx]
        rep = self.rep_t[idx]
        if self.cfg.cb0_rep_win > 0 and bool((self.cfg.cb0_rep > 0) or self._any_rep):
            hist = self.hist_t[idx]
            pen = torch.zeros_like(logits)
            pen.scatter_add_(1, hist.clamp_min(0), (hist >= 0).float())
            pen[:, AEOS_CLASS] = 0
            logits = logits - rep[:, None] * (pen > 0).float()
        srt = logits.sort(-1, descending=True).values
        kth = srt.gather(1, (self.topk_t[idx] - 1).clamp(0, logits.shape[1] - 1)[:, None])
        logits = logits.masked_fill(logits < kth, -float("inf"))
        if self._any_top_p:
            sp_, si = logits.softmax(-1).sort(-1, descending=True)
            drop = (sp_.cumsum(-1) - sp_) > self.topp_t[idx, None]
            logits = logits.masked_fill(torch.zeros_like(drop).scatter(1, si, drop), -float("inf"))
        probs = logits.softmax(-1)
        pa = probs[:, AEOS_CLASS]
        arg = probs.argmax(-1)
        frames = self.frames_t[idx]
        can = frames >= self.min_t[idx]
        hz = self.hazard_t[idx]
        hz = torch.where(can, 1 - (1 - hz) * (1 - pa), hz)
        code = torch.zeros_like(arg)
        code = torch.where(can & (hz >= .6), torch.full_like(code, 4), code)
        code = torch.where(can & (pa >= .35), torch.full_like(code, 3), code)
        code = torch.where(can & (arg == AEOS_CLASS), torch.full_like(code, 2), code)
        q = probs.clone()
        q[:, AEOS_CLASS] = 0
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        E = row_exponential(q.shape[0], q.shape[1], self.seed_t[idx], self.ctr_t[idx], level=0)
        cb0 = (q / E).argmax(-1).clamp_max(AEOS_CLASS - 1)
        self.ctr_t[idx] += 1
        if self.cfg.cb0_rep_win > 0:
            self.hist_t[idx] = torch.cat([self.hist_t[idx, 1:], cb0[:, None]], 1)
        self.hazard_t[idx] = hz
        self.frames_t[idx] = frames + 1
        return code, cb0
