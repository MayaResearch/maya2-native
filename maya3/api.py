"""Public Python API.

    from maya3 import Maya3TTS
    tts = Maya3TTS.from_pretrained("path/to/model.pt", backbone_dir=..., mimi_dir=..., ecapa_dir=...)
    wav = tts.generate("नमस्ते, आप कैसे हैं?", lang="hi")                                  # plain voice
    tts.register_voice("asha", ref_audio="asha.wav", ref_text="...", lang="hi")            # clone once, reuse
    wav = tts.generate("Hello there.", lang="en", voice="asha")                             # cached prefix KV + ECAPA
    wav = tts.generate("Hello there.", lang="en", ref_audio="x.wav", ref_text="...")        # one-off clone
    for pcm in tts.stream("Long text ...", lang="te", voice="asha"): ...                    # int16 chunks, 24 kHz
    async for pcm in tts.astream(...): ...                                                  # asyncio

Audio is 24 kHz mono.  generate() returns float32 in [-1, 1]; stream()/astream() yield int16 numpy chunks.
Concurrency: any number of threads / coroutines may call generate/stream at once; the engine batches them.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch

from maya3.engine import SR, Engine, EngineConfig
from maya3.prompt import Voice, norm_mode


class GenerationError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def load_audio(x, sr=None):
    """path | (array, sr) | array (assumed 24 kHz) -> float32 mono 24 kHz."""
    if isinstance(x, (str, os.PathLike)):
        import soundfile as sf
        w, sr = sf.read(str(x), dtype="float32")
    elif isinstance(x, tuple):
        w, sr = x
    else:
        w = x
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1) if w.shape[1] < w.shape[0] else w.mean(0)
    sr = sr or SR
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        w = resample_poly(w, SR // g, int(sr) // g).astype(np.float32)
    return w


class Ecapa:
    """ECAPA-TDNN speaker embedding, corpus recipe (speechbrain spkrec-ecapa-voxceleb; 16 kHz fbank-80, sentence norm)."""

    def __init__(self, ecapa_dir, device="cuda"):
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.fbank = Fbank(n_mels=80).eval().to(device)
        self.norm = InputNormalization(norm_type="sentence", std_norm=False).eval().to(device)
        emb = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                         dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192).eval()
        emb.load_state_dict(torch.load(Path(ecapa_dir) / "embedding_model.ckpt", map_location="cpu", weights_only=True), strict=True)
        self.emb = emb.to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, w24):
        from scipy.signal import resample_poly
        if w24 is None or len(w24) < 2400:
            raise ValueError("reference audio is too short (need >= 0.1 s; 3-10 s recommended)")
        w16 = torch.from_numpy(resample_poly(np.asarray(w24, np.float32), 2, 3).astype(np.float32)).to(self.device)[None]
        L = torch.ones(1, device=self.device)
        e = self.emb(self.norm(self.fbank(w16), L), L).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


class Maya3TTS:
    def __init__(self, engine: Engine, ecapa_dir=None):
        self.engine = engine
        self.ecapa_dir = ecapa_dir
        self._ecapa = None
        self._oneoff = {}
        engine.start()

    @classmethod
    def from_pretrained(cls, ckpt, backbone_dir=None, mimi_dir=None, ecapa_dir=None, device="cuda", voices=True, **config):
        """from_pretrained("<release dir>")  -- a Maya3 release directory (model.safetensors + backbone/ + voices/), resolved by
        maya3.hub.resolve (Mimi / ECAPA from the Hugging Face cache); or explicit (ckpt, backbone_dir, mimi_dir, ecapa_dir).
        voices: register the release's named voices (voices/voices.json). config: EngineConfig fields."""
        named = None
        if backbone_dir is None:
            from maya3 import hub
            r = hub.resolve(ckpt)
            ckpt, backbone_dir = r["ckpt"], r["backbone_dir"]
            mimi_dir, ecapa_dir = mimi_dir or r["mimi_dir"], ecapa_dir or r.get("ecapa_dir")
            named = r.get("voices_json")
        cfg = EngineConfig(**config)
        tts = cls(Engine(ckpt, backbone_dir, mimi_dir, cfg, device=device), ecapa_dir=ecapa_dir)
        if voices and named and os.path.exists(named):
            tts.load_named_voices(named)
        return tts

    def load_named_voices(self, voices_json, **kw):
        """Register the named (trained) voices listed in voices.json; see maya3.prompt.load_named_voices."""
        from maya3.prompt import load_named_voices
        ids = load_named_voices(self.engine.registry, voices_json, **kw)
        for vid in ids:
            v = self.engine.registry.get(vid)
            self.engine.call(lambda v=v: self.engine.prepare_voice(v))
        return ids

    # ------------------------------------------------------------------ voices
    @property
    def voices(self):
        return self.engine.registry.ids()

    def _embed(self, w):
        if self._ecapa is None:
            if self.ecapa_dir is None:
                raise ValueError("ecapa_dir is required for reference-audio voices")
            self._ecapa = Ecapa(self.ecapa_dir, self.engine.device)
        return self._ecapa(w)

    def make_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", ref_codes=None, ecapa=None,
                   builder=None, meta=None):
        """Build a Voice from a reference clip (audio + its transcript + its language) or precomputed codes/ECAPA."""
        mode = norm_mode(mode)
        clips = []
        if ref_audio is not None or ref_codes is not None:
            if ref_codes is None:
                w = load_audio(ref_audio)
                if len(w) > 30 * SR:
                    raise ValueError("reference audio longer than 30 s; 3-10 s works best")

                def enc():
                    x = torch.as_tensor(w, device=self.engine.device)[None, None]
                    return self.engine.mimi.encode(x, num_quantizers=32).audio_codes[0].cpu().numpy().astype(np.int64)
                ref_codes = self.engine.call(enc)
                if ecapa is None and mode in ("spk", "icl_spk"):
                    ecapa = self.engine.call(lambda: self._embed(w))
            if mode in ("icl", "icl_spk"):
                if not ref_text or not lang:
                    raise ValueError("ref_text and lang (the reference clip's transcript and language) are required for cloning")
                clips = [{"text": ref_text, "codes": np.asarray(ref_codes, np.int64), "lang": lang}]
        v = Voice(voice_id, mode, clips=clips, ecapa=ecapa, builder=builder, meta=meta or {})
        self.engine.call(lambda: self.engine.prepare_voice(v))
        return v

    def register_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", **kw):
        """Register (or replace) a named voice; its prefix KV and speaker memory are computed now and reused."""
        v = self.make_voice(voice_id, ref_audio, ref_text, lang, mode, **kw)
        self.engine.registry.add(v)
        return voice_id

    def remove_voice(self, voice_id):
        return self.engine.registry.remove(voice_id) is not None

    def _voice(self, voice, ref_audio, ref_text, ref_lang, mode):
        if ref_audio is None:
            return self.engine.registry.get(voice or "plain")
        if voice is not None and voice != "plain":
            raise ValueError("pass either voice=<registered/named voice> or ref_audio=, not both (a named voice with a reference "
                             "clip was never trained)")
        h = hashlib.sha1()
        if isinstance(ref_audio, (str, os.PathLike)):
            st = os.stat(ref_audio)
            h.update(f"{os.path.abspath(ref_audio)}|{st.st_size}|{st.st_mtime_ns}".encode())
        else:
            h.update(np.asarray(load_audio(ref_audio)).tobytes())
        h.update(f"|{ref_text}|{ref_lang}|{mode}".encode())
        key = h.hexdigest()
        if key not in self._oneoff:
            if len(self._oneoff) > 64:
                self._oneoff.pop(next(iter(self._oneoff)))
            self._oneoff[key] = self.make_voice(f"oneoff-{key[:12]}", ref_audio, ref_text, ref_lang, mode)
        return self._oneoff[key]

    # ------------------------------------------------------------------ synthesis
    SAMPLING_KEYS = ("seed", "temperature", "top_k", "top_p", "repetition_penalty")

    def _prep_text(self, text, lang, normalize):
        if not normalize:
            return text
        from maya3.text import normalise_text      # the bundled normaliser (digits, currency, URLs, ...)
        if isinstance(text, (list, tuple)):
            return [normalise_text(t, lang) for t in text]
        return normalise_text(text, lang)

    def stream(self, text, lang="en", voice=None, ref_audio=None, ref_text=None, ref_lang=None, mode="icl_spk",
               max_frames=None, timeout=300, normalize=False, info=None, seed=None, temperature=None, top_k=None,
               top_p=None, repetition_penalty=None):
        """Yield int16 PCM chunks (24 kHz) as they are generated.
        seed/temperature/top_k/top_p/repetition_penalty: per-request sampler settings (None = engine defaults).
        normalize: run the bundled text normaliser first (digits, currency, dates, URLs, ALL-CAPS names).
        info: optional dict that receives the end payload (reason, frames, ttfa_s, stops, ...)."""
        v = self._voice(voice, ref_audio, ref_text, ref_lang or lang, mode)
        sampling = {"seed": seed, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                    "repetition_penalty": repetition_penalty}
        q = queue.Queue()
        req = self.engine.submit(self._prep_text(text, lang, normalize), lang, v, on_event=lambda k, p: q.put((k, p)),
                                 max_frames=max_frames, sampling=sampling)
        try:
            while True:
                kind, payload = q.get(timeout=timeout)
                if kind == "audio":
                    yield payload
                elif kind == "end":
                    if info is not None:
                        info.update(payload)
                    self.last_info = payload
                    return
                else:
                    raise GenerationError(payload.get("code"), payload.get("error"))
        finally:
            req.cancel()

    def generate(self, text, lang="en", voice=None, ref_audio=None, ref_text=None, ref_lang=None, mode="icl_spk",
                 max_frames=None, return_info=False, **kw):
        """Synthesise text -> float32 waveform (24 kHz). kw: normalize, seed, temperature, top_k, top_p, repetition_penalty."""
        info = {}
        chunks = list(self.stream(text, lang, voice, ref_audio, ref_text, ref_lang, mode, max_frames, info=info, **kw))
        wav = (np.concatenate(chunks).astype(np.float32) / 32767.0) if chunks else np.zeros(0, np.float32)
        return (wav, info) if return_info else wav

    async def astream(self, text, lang="en", voice=None, ref_audio=None, ref_text=None, ref_lang=None, mode="icl_spk",
                      max_frames=None, info=None, normalize=False, **sampling):
        """asyncio variant of stream(); `info` (a dict) receives the end payload."""
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        if ref_audio is not None:
            v = await asyncio.to_thread(self._voice, voice, ref_audio, ref_text, ref_lang or lang, mode)
        else:
            v = self.engine.registry.get(voice or "plain")
        req = self.engine.submit(self._prep_text(text, lang, normalize), lang, v,
                                 on_event=lambda k, p: loop.call_soon_threadsafe(q.put_nowait, (k, p)),
                                 max_frames=max_frames, sampling=sampling)
        try:
            while True:
                kind, payload = await q.get()
                if kind == "audio":
                    yield payload
                elif kind == "end":
                    if info is not None:
                        info.update(payload)
                    return
                else:
                    raise GenerationError(payload.get("code"), payload.get("error"))
        finally:
            req.cancel()

    def close(self):
        self.engine.close()
