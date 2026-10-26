"""CPU-only stand-ins for maya3.engine.Engine / maya3.api.Maya3TTS with the same event contract.

FakeEngine runs one thread that "ticks" every `frame_s` seconds: it admits pending requests into at most `max_seats`
seats and emits one 80 ms frame (1920 int16 samples) per active seat per tick, then ('end', {...}).  Frame k of a
request is filled with the value k + 1 so tests can check order and completeness of the PCM.
Text directives (for tests): "#frames=N" fixes the frame count, "#error=K" emits ('error', internal) after K frames.
"""
from __future__ import annotations

import collections
import re
import threading
import time
from types import SimpleNamespace

import numpy as np

FRAME = 1920
SR = 24000


class FakeVoice:
    def __init__(self, vid, mode="plain", ref_lang=None, meta=None, builder=None):
        self.id, self.mode, self.ref_lang = vid, mode, ref_lang
        self.meta, self.builder = dict(meta or {}), builder


class NamedSpeakerBuilder:
    """Stand-in for maya3.prompt.NamedSpeakerBuilder (the server recognises named voices by mode 'named' or by this
    builder class name)."""

    def __init__(self, template="{{{name}}}: {text}"):
        self.template = template


def named_voice(name, mode="named", **meta):
    return FakeVoice(name, mode, None, meta={"name": name, **meta}, builder=NamedSpeakerBuilder())


class FakeRegistry:
    def __init__(self):
        self._v = {"plain": FakeVoice("plain")}
        self._lock = threading.Lock()

    def add(self, v, replace=True):
        with self._lock:
            if not replace and v.id in self._v:
                raise KeyError(f"voice {v.id} exists")
            self._v[v.id] = v
        return v

    def get(self, vid):
        with self._lock:
            if vid not in self._v:
                raise KeyError(f"unknown voice {vid!r}")
            return self._v[vid]

    def remove(self, vid):
        with self._lock:
            if vid == "plain":
                raise KeyError("cannot remove the plain voice")
            return self._v.pop(vid, None)

    def ids(self):
        with self._lock:
            return list(self._v)

    def __contains__(self, vid):
        with self._lock:
            return vid in self._v


class FakeRequest:
    def __init__(self, rid, text, lang, voice, on_event, frames, error_at, sampling=None):
        self.id, self.text, self.lang, self.voice, self.on_event = rid, text, lang, voice, on_event
        self.sampling = sampling           # the sampling= kwarg as received (None when not passed)
        self.n_frames, self.error_at = frames, error_at
        self.cancelled = False
        self.frames = 0
        self.t_submit = time.perf_counter()
        self.t_first = None
        self.events = []          # kinds emitted, for assertions
        self.done = threading.Event()

    def cancel(self):
        self.cancelled = True


class FakeEngine:
    def __init__(self, max_seats=4, frame_s=0.005, frames_per_char=0.5, min_frames=3):
        self.cfg = SimpleNamespace(max_seats=max_seats)
        self.frame_s, self.frames_per_char, self.min_frames = frame_s, frames_per_char, min_frames
        self.registry = FakeRegistry()
        self.device = "cpu"
        self.failed = None
        self.requests = []
        self._pending = collections.deque()
        self._active = []
        self._cv = threading.Condition()
        self._stop = False
        self._n = 0
        self._thread = threading.Thread(target=self._loop, name="fake-engine", daemon=True)
        self._thread.start()

    def submit(self, text, lang, voice="plain", on_event=None, max_frames=None, rid=None, split=True, sampling=None):
        if isinstance(voice, str):
            voice = self.registry.get(voice)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if self.failed:
            raise RuntimeError(f"engine failed: {self.failed}")
        if sampling is not None:           # same contract as Engine._check_sampling
            bad = set(sampling) - {"seed", "temperature", "top_k", "top_p", "repetition_penalty"}
            if bad:
                raise ValueError(f"unknown sampling keys {sorted(bad)}")
        m = re.search(r"#frames=(\d+)", text)
        n = int(m.group(1)) if m else max(self.min_frames, int(len(text) * self.frames_per_char))
        if max_frames:
            n = min(n, int(max_frames))
        m = re.search(r"#error=(\d+)", text)
        self._n += 1
        req = FakeRequest(rid or f"r{self._n}", text, lang, voice, on_event or (lambda k, p: None), n,
                          int(m.group(1)) if m else None, None if sampling is None else dict(sampling))
        with self._cv:
            self.requests.append(req)
            self._pending.append(req)
            self._cv.notify()
        return req

    def _emit(self, req, kind, payload):
        req.events.append(kind)
        try:
            req.on_event(kind, payload)
        except Exception:
            req.cancelled = True
        if kind != "audio":
            req.done.set()

    def _loop(self):
        while True:
            with self._cv:
                while not self._stop and not self._pending and not self._active:
                    self._cv.wait(0.5)
                if self._stop:
                    break
                # cancelled requests leave first (pending ones too, like the real engine's admission)
                for r in [r for r in self._pending if r.cancelled]:
                    self._pending.remove(r)
                    self._emit(r, "error", {"code": "cancelled", "error": "cancelled"})
                while self._pending and len(self._active) < self.cfg.max_seats:
                    self._active.append(self._pending.popleft())
            keep = []
            for r in self._active:
                if r.cancelled:
                    self._emit(r, "error", {"code": "cancelled", "error": "cancelled"})
                    continue
                if r.error_at is not None and r.frames >= r.error_at:
                    self._emit(r, "error", {"code": "internal", "error": "synthetic failure"})
                    continue
                if r.t_first is None:
                    r.t_first = time.perf_counter()
                r.frames += 1
                self._emit(r, "audio", np.full(FRAME, r.frames, np.int16))
                if r.frames >= r.n_frames:
                    self._emit(r, "end", {"reason": "aeos", "frames": r.frames, "audio_s": r.frames * FRAME / SR,
                                          "ttfa_s": r.t_first - r.t_submit, "wall_s": time.perf_counter() - r.t_submit,
                                          "pieces": 1, "stops": [{"reason": "aeos", "frames": r.frames}]})
                else:
                    keep.append(r)
            self._active = keep
            time.sleep(self.frame_s)
        for r in list(self._pending) + self._active:
            self._emit(r, "error", {"code": "unavailable", "error": "engine stopped"})

    def health(self):
        return {"ready": self.failed is None, "failed": self.failed, "active": len(self._active),
                "pending": len(self._pending), "max_seats": self.cfg.max_seats}

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=5)


class FakeTTS:
    """Maya3TTS surface used by the server: engine, voices, register_voice, remove_voice, close."""

    def __init__(self, engine=None, register_s=0.0, **engine_kw):
        self.engine = engine or FakeEngine(**engine_kw)
        self.register_s = register_s
        self.registered = []

    @property
    def voices(self):
        return self.engine.registry.ids()

    def register_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", **kw):
        if mode in ("icl", "icl_spk") and (not ref_text or not lang):
            raise ValueError("ref_text and lang are required for cloning")
        if mode != "plain" and ref_audio is None:
            raise ValueError("reference audio required")
        if ref_audio is not None:
            w, sr = ref_audio
            assert isinstance(w, np.ndarray) and w.dtype == np.float32 and w.ndim == 1
        time.sleep(self.register_s)
        self.registered.append({"voice_id": voice_id, "mode": mode, "lang": lang, "ref_text": ref_text,
                                "ref_sr": ref_audio[1] if ref_audio is not None else None,
                                "ref_len": len(ref_audio[0]) if ref_audio is not None else 0})
        self.engine.registry.add(FakeVoice(voice_id, mode, lang if mode in ("icl", "icl_spk") else None,
                                           meta=kw.get("meta")))
        return voice_id

    def remove_voice(self, voice_id):
        return self.engine.registry.remove(voice_id) is not None

    def close(self):
        self.engine.close()
