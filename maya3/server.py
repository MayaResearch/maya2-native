"""Maya3 TTS server: OpenAI-compatible HTTP speech endpoint + low-latency WebSocket streaming (aiohttp).

Run
---
    python -m maya3.server --ckpt model.pt --backbone-dir BB --mimi-dir MIMI --ecapa-dir ECAPA \\
        --host 127.0.0.1 --port 8080 --max-seats 64 --max-pending 64 --queue-deadline-ms 2000 [--voices voices.json] \\
        [--no-normalise]

The port opens immediately; `GET /health` answers 503 while the model loads and warms up and 200 once it is ready.
`--voices` is a JSON list (or {"voices": [...]}) of {"voice_id", "ref_audio" (path, relative to the json file),
"ref_text", "lang", "mode", "meta"} that is registered before the server reports ready.
The model's audio is 24 kHz, mono; PCM is signed 16-bit little-endian.

Text normalisation
------------------
Every request's text goes through the bundled normaliser (maya3.text.normalise_text: numbers, currency, dates,
times, OTPs / phone numbers, URLs / e-mails -> words in the request language; ALL-CAPS names title-cased; inline
[tags] such as [laughs] untouched) unless the server runs with --no-normalise or the request sets "normalize": false
(HTTP and WS).  Only the user's text is normalised; a named voice's "{Name}: " prefix is added by the engine.

Sampling (HTTP and WS, all optional; absent = engine defaults)
--------------------------------------------------------------
    seed (int, 0..2^63-1)  temperature (0.1..1.5)  top_k (int, 1..2048)  top_p (0 < p <= 1)  repetition_penalty (0..5)
Only the keys given are passed to the engine (Engine.submit(..., sampling={...})); invalid values -> 400 / WS
invalid_request naming the field.  Same seed + text + voice gives the same audio when the request runs alone.

HTTP (OpenAI-compatible)
------------------------
    POST /v1/audio/speech   {"model": "maya3", "input": "नमस्ते", "voice": "plain", "response_format": "wav",
                             "language": "hi", "speed": 1.0, "stream_format": "audio", "normalize": true, "seed": 7}
    curl -sN localhost:8080/v1/audio/speech -H 'content-type: application/json' \\
         -d '{"input":"Hello there.","language":"en","response_format":"pcm"}' | ffplay -f s16le -ar 24000 -ch_layout mono -

  * response_format (Content-Type):
        wav  (audio/wav)              default; streamed, RIFF header with sizes 0xFFFFFFFF, then s16le PCM
        pcm  (audio/pcm)              streamed raw s16le, 24 kHz mono
        mp3  (audio/mpeg)             streamed CBR MPEG-2 layer III, 24 kHz mono, ~80 kbit/s (libsndfile/LAME); bare
                                      frames, no Xing/Info header; decoders add ~50 ms of encoder delay/padding
        opus (audio/ogg; codecs=opus) streamed Ogg/Opus at 24 kHz (libsndfile/libopus); the Ogg headers are sent at
                                      once, audio in ~1 s Ogg pages
        flac (audio/flac)             lossless, sent when the audio is complete (Content-Length set)
    aac -> 400 (no AAC encoder is bundled).  mp3/opus need libsndfile >= 1.1 (the soundfile >= 0.12 wheels); when it
    is missing they are refused with 400.  GET /health lists the usable formats under server.formats.
  * stream_format: "audio" (default, chunked body) or "sse" (OpenAI events `speech.audio.delta` carrying base64
    bytes of the chosen format, then `speech.audio.done`; wav/pcm/mp3/opus, not flac).
  * language / lang: as bn en gu hi kn ml mr or pa ta te; when absent it is guessed from the script of the input
    (Devanagari -> hi, Bengali script -> bn, Latin -> en, ...). Reported in the X-Maya3-Language header.
  * voice: a registered voice id ("plain" by default); named voices match case-insensitively ("asha" -> "Asha").
    OpenAI voice names (alloy, nova, ...) map to "plain" unless a voice with that id is registered.
    speed must be 1.0.  Extension: max_frames (80 ms frames, 1..1500).
  * normalize: true (default) / false; sampling fields as above.
  * Response headers: X-Sample-Rate, X-Channels, X-Encoding (pcm_s16le | mp3 | opus | flac), X-Maya3-Language,
    X-Maya3-Voice (the resolved id), X-Maya3-Normalized (1/0), X-Request-Id.
  * Errors are JSON {"error": {"message", "type", "code", "param"}}: 400 invalid input, 503 overloaded / not ready
    (with Retry-After), sent before any audio.  A failure after audio started aborts the connection (the chunked body
    has no terminating chunk), so a truncated utterance never looks like a successful one.

WebSocket
---------
    WS /v1/tts/stream
  server -> {"type": "metadata", "sample_rate": 24000, "encoding": "pcm_s16le", "channels": 1, "max_contexts": 8,
             "languages": [...]}
  client -> {"type": "speak", "context_id": "c1", "text": "...", "language": "hi", "voice": "plain",
             "max_frames": 400, "binary": false, "normalize": true, "seed": 7, "temperature": 0.8, ...}
            (voice, language, normalize and the sampling fields behave as on HTTP)
  server -> {"type": "audio", "context_id": "c1", "audio": "<base64 s16le>"}            (repeated)
            or, with "binary": true, binary frames: uint32-LE len(context_id utf-8) | context_id | PCM s16le
  server -> {"type": "end", "context_id": "c1", "reason": "aeos", "frames": 52, "audio_s": 4.16, "ttfa_ms": 61.0}
  client -> {"type": "cancel", "context_id": "c1"}   -> the context ends with {"type": "end", "reason": "cancelled"}
  client -> {"type": "ping"}                         -> {"type": "pong"}
  errors -> {"type": "error", "context_id": "c1" | null, "code": "...", "error": "..."}
  Every speak gets exactly one terminal message (end or error) and no audio after it.  Several contexts may run on
  one socket at once (up to max_contexts; more -> error "too_many_contexts").  Error codes: invalid_request,
  overloaded (queue full or queue deadline passed; sent before any audio), slow_consumer (the client fell more than
  --ws-max-lag-s of audio behind; the request is cancelled), unavailable, too_many_contexts, internal.
  Closing the socket cancels its requests.  Audio on the WebSocket is always PCM s16le 24 kHz mono (no
  response_format); use HTTP for wav/mp3/opus/flac.

Voices
------
    GET    /v1/voices                     {"voices": [{"id", "mode", "ref_lang"}]}
                        mode: plain | spk | icl | icl_spk | named; named voices also carry "meta" (the registry's
                        Voice.meta: name / display name, gender, age, style, langs, ...); a name-template voice
                        whose own mode is not "named" (e.g. icl_spk) reports that as "base_mode"
    POST   /v1/voices   JSON {"voice_id", "ref_audio_b64" (wav/flac bytes, base64), "ref_text", "lang",
                              "mode": "icl_spk", "replace": false}
                        or multipart/form-data with the same fields and the clip as file field "ref_audio"
                        -> 201 {"id", "mode", "ref_lang", "register_s"}; 409 if it exists and replace is false
    DELETE /v1/voices/{id}

Health / misc
-------------
    GET /health      engine.health() + {"server": {...}}; 503 until ready
    GET /v1/models   OpenAI-style model list

Admission control: a request that would make more than --max-pending requests wait for their first audio, or that
waits longer than --queue-deadline-ms before its first audio, is refused (HTTP 503 / WS "overloaded") and cancelled
in the engine.  A client disconnect cancels its engine request.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import collections
import io
import json
import logging
import math
import threading
import re
import struct
import time
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web

SR = 24000
FRAME = 1920
LANGS = ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")
MAX_TEXT_CHARS = 4000
MAX_FRAMES = 1500
MODES = ("plain", "spk", "icl", "icl_spk")
MODE_ALIASES = {"joint_null": "plain", "spk_only": "spk", "icl_only": "icl", "clone": "icl_spk"}
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse",
                 "marin", "cedar"}
FORMATS = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg", "opus": "audio/ogg; codecs=opus",
           "flac": "audio/flac"}
STREAMED_FORMATS = ("wav", "pcm", "mp3", "opus")      # flac is encoded once the utterance is complete
SF_CODECS = {"mp3": ("MP3", "MPEG_LAYER_III"), "opus": ("OGG", "OPUS")}   # libsndfile >= 1.1 (soundfile >= 0.12 wheels)
UNSUPPORTED_FORMATS = {"aac": "no AAC encoder is bundled"}
MP3_LEVEL = 0.5                                       # libsndfile compression level; CBR ~80 kbit/s at 24 kHz mono
NORMALISE_INLINE_CHARS = 256                          # longer inputs are normalised on a worker thread
SEED_MAX = 2 ** 63 - 1
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
# Unicode blocks -> language, used only when the caller gives no language.
SCRIPTS = ((0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
           (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
           (0x0D00, 0x0D7F, "ml"))

log = logging.getLogger("maya3.server")


# ---------------------------------------------------------------------------------------------------- helpers
class ApiError(Exception):
    def __init__(self, status, code, message, param=None, retry_after=None):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.retry_after = status, code, message, param, retry_after

    def response(self):
        kind = "invalid_request_error" if self.status < 500 else "server_error"
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after else None
        return web.json_response({"error": {"message": self.message, "type": kind, "code": self.code,
                                            "param": self.param}}, status=self.status, headers=headers)


def bad(message, param=None, code="invalid_request"):
    return ApiError(400, code, message, param)


def detect_lang(text):
    counts = collections.Counter()
    for ch in text:
        o = ord(ch)
        if o < 0x80:
            if ch.isalpha():
                counts["en"] += 1
            continue
        for lo, hi, lang in SCRIPTS:
            if lo <= o <= hi:
                counts[lang] += 1
                break
    return counts.most_common(1)[0][0] if counts else "en"


def wav_header(data_bytes=0xFFFFFFFF, sr=SR, channels=1, bits=16):
    """RIFF/WAVE header; the default 0xFFFFFFFF sizes mark a stream of unknown length."""
    riff = 0xFFFFFFFF if data_bytes == 0xFFFFFFFF else 36 + data_bytes
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, channels, sr,
                       sr * channels * bits // 8, channels * bits // 8, bits, b"data", data_bytes)


def ws_binary_frame(context_id, pcm):
    cid = context_id.encode()
    return struct.pack("<I", len(cid)) + cid + pcm


def parse_lang(body, text):
    lang = body.get("language", body.get("lang"))
    if lang is None or lang == "auto":
        return detect_lang(text)
    if not isinstance(lang, str) or lang.lower() not in LANGS:
        raise bad(f"unsupported language {lang!r}; supported: {', '.join(LANGS)}", "language")
    return lang.lower()


def parse_text(text, field):
    if not isinstance(text, str) or not text.strip():
        raise bad(f"{field} must be a non-empty string", field)
    if len(text) > MAX_TEXT_CHARS:
        raise bad(f"{field} is {len(text)} characters; the limit is {MAX_TEXT_CHARS}", field)
    return text.strip()


def parse_max_frames(v):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_FRAMES:
        raise bad(f"max_frames must be an integer in 1..{MAX_FRAMES} (80 ms frames)", "max_frames")
    return v


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_sampling(body):
    """Per-request sampling overrides -> dict of the keys that were given (None when none were)."""
    out = {}
    v = body.get("seed")
    if v is not None:
        if not _is_int(v) or not 0 <= v <= SEED_MAX:
            raise bad("seed must be an integer in 0..2^63-1", "seed")
        out["seed"] = int(v)
    v = body.get("temperature")
    if v is not None:
        if not _is_num(v) or not 0.1 <= v <= 1.5:
            raise bad("temperature must be a number in 0.1..1.5", "temperature")
        out["temperature"] = float(v)
    v = body.get("top_k")
    if v is not None:
        if not _is_int(v) or not 1 <= v <= 2048:
            raise bad("top_k must be an integer in 1..2048", "top_k")
        out["top_k"] = int(v)
    v = body.get("top_p")
    if v is not None:
        if not _is_num(v) or not 0 < v <= 1:
            raise bad("top_p must be a number with 0 < top_p <= 1", "top_p")
        out["top_p"] = float(v)
    v = body.get("repetition_penalty")
    if v is not None:
        if not _is_num(v) or not 0 <= v <= 5:
            raise bad("repetition_penalty must be a number in 0..5", "repetition_penalty")
        out["repetition_penalty"] = float(v)
    return out or None


def parse_normalize(body):
    key = "normalize" if "normalize" in body else "normalise"
    v = body.get(key, True)
    if v is None:
        return True
    if not isinstance(v, bool):
        raise bad(f"{key} must be true or false", key)
    return v


def is_named(voice):
    """Named speakers (selected by a "{Name}: text" template): mode 'named' or a NamedSpeakerBuilder builder."""
    return getattr(voice, "mode", None) == "named" or type(getattr(voice, "builder", None)).__name__ == "NamedSpeakerBuilder"


def _jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return {}


def voice_info(vid, v):
    ref_lang = getattr(v, "ref_lang", None)
    if ref_lang is None:
        clips = getattr(v, "clips", None) or []
        ref_lang = clips[0].get("lang") if clips and isinstance(clips[0], dict) else None
    mode = getattr(v, "mode", None)
    out = {"id": vid, "mode": mode, "ref_lang": ref_lang}
    if is_named(v):
        out["mode"] = "named"
        if mode != "named":
            out["base_mode"] = mode          # conditioning under the name template (plain/spk/icl/icl_spk)
        out["meta"] = _jsonable(dict(getattr(v, "meta", None) or {}))
    return out


def _normaliser():
    from maya3.text import normalise_text
    return normalise_text


# ---------------------------------------------------------------------------------------------------- jobs
class Overloaded(Exception):
    pass


class _Bridge:
    """Engine thread -> event loop: events are queued and ONE call_soon_threadsafe wakes the loop per batch (not one per
    80 ms frame per request, which costs a self-pipe write + wakeup each and adds host jitter at 100+ seats)."""

    def __init__(self, loop):
        self.loop, self.q, self.scheduled = loop, collections.deque(), False

    def post(self, job, kind, payload):    # engine thread
        self.q.append((job, kind, payload))
        if not self.scheduled:
            self.scheduled = True
            self.loop.call_soon_threadsafe(self.flush)

    def flush(self):                        # event loop
        self.scheduled = False
        q = self.q
        while q:
            job, kind, payload = q.popleft()
            job._put(kind, payload)


_BRIDGES = {}


def _bridge(loop):
    b = _BRIDGES.get(id(loop))
    if b is None or b.loop is not loop:
        b = _BRIDGES[id(loop)] = _Bridge(loop)
    return b


class Job:
    """One engine request seen from the event loop.  Engine events arrive through a coalescing _Bridge."""

    def __init__(self, srv, rid, loop, max_lag_s, on_overflow=None):
        self.srv, self.rid, self.loop = srv, rid, loop
        self.items = collections.deque()
        self.ev = asyncio.Event()
        self.req = None
        self.lag = 0                       # samples received from the engine but not yet consumed
        self.max_lag = int(max_lag_s * SR) if max_lag_s else None
        self.on_overflow = on_overflow
        self.overflowed = False
        self.closed = False
        self.t_submit = time.perf_counter()
        self.t_first = None
        self.bridge = _bridge(loop)

    def on_event(self, kind, payload):     # engine thread
        try:
            self.bridge.post(self, kind, payload)
        except RuntimeError:               # loop closed: the request is orphaned
            if self.req is not None:
                self.req.cancel()

    def _put(self, kind, payload):
        if self.closed:
            return
        self.srv._waiting.discard(self)
        if kind == "audio":
            if self.t_first is None:
                self.t_first = time.perf_counter()
            self.lag += len(payload)
            if self.max_lag is not None and self.lag > self.max_lag and not self.overflowed:
                self.overflowed = True
                if self.req is not None:
                    self.req.cancel()
                if self.on_overflow is not None:
                    self.on_overflow()
        self.items.append((kind, payload))
        self.ev.set()

    def interrupt(self):
        self.items.append(("interrupt", None))
        self.ev.set()

    async def get(self, timeout=None):
        while not self.items:
            self.ev.clear()
            if timeout is None:
                await self.ev.wait()
            else:
                await asyncio.wait_for(self.ev.wait(), timeout)
        kind, payload = self.items.popleft()
        if kind == "audio":
            self.lag -= len(payload)
        return kind, payload

    def take_audio(self, first, max_samples):
        """Coalesce `first` with audio already queued (no waiting) up to max_samples."""
        parts, n = [first.tobytes()], len(first)
        while n < max_samples and self.items and self.items[0][0] == "audio":
            _, p = self.items.popleft()
            self.lag -= len(p)
            parts.append(p.tobytes())
            n += len(p)
        return b"".join(parts), n

    def close(self):
        if not self.closed:
            self.closed = True
            if self.req is not None:
                self.req.cancel()
            self.srv._waiting.discard(self)
            self.srv._jobs.discard(self)


# ---------------------------------------------------------------------------------------------------- server
class Maya3Server:
    def __init__(self, tts=None, *, max_pending=64, queue_deadline_ms=2000, ws_max_contexts=8, ws_max_lag_s=20.0,
                 http_max_lag_s=120.0, ws_send_queue=64, ws_send_timeout_s=15.0, coalesce_ms=480, loader=None,
                 owns_tts=False, normalise=True):
        self.tts = tts
        self.normalise = normalise         # bundled text normaliser (maya3.text); per request: "normalize": false
        self.loader = loader
        self.owns_tts = owns_tts or loader is not None
        self.load_error = None
        self.max_pending = max_pending
        self.deadline = queue_deadline_ms / 1000.0
        self.ws_max_contexts = ws_max_contexts
        self.ws_max_lag_s = ws_max_lag_s
        self.http_max_lag_s = http_max_lag_s
        self.ws_send_queue = ws_send_queue
        self.ws_send_timeout_s = ws_send_timeout_s
        self.coalesce = max(FRAME, int(coalesce_ms * SR / 1000))
        self._waiting = set()              # jobs without their first event yet (admission control)
        self._jobs = set()
        self._voice_lock = None
        self.sockets = 0
        self.stats = collections.Counter()

    # ------------------------------------------------------------------ app
    def app(self):
        app = web.Application(client_max_size=24 * 1024 * 1024)
        app["maya3"] = self
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        r = app.router
        r.add_get("/health", self.health)
        r.add_get("/v1/models", self.models)
        r.add_post("/v1/audio/speech", self.speech)
        r.add_get("/v1/tts/stream", self.ws)
        r.add_get("/v1/voices", self.list_voices)
        r.add_post("/v1/voices", self.add_voice)
        r.add_delete("/v1/voices/{voice_id}", self.delete_voice)
        return app

    async def _startup(self, app):
        self._voice_lock = asyncio.Lock()
        if self.normalise:
            try:                           # import + warm the tables off the loop (first call ~10 ms)
                await asyncio.to_thread(_normaliser().__call__, "1", "en")
            except Exception as exc:
                log.error("text normaliser unavailable (%s: %s); serving without normalisation", type(exc).__name__, exc)
                self.normalise = False
        if self.tts is None and self.loader is not None:
            self._load_task = asyncio.create_task(self._load())

    async def _load(self):
        t0 = time.perf_counter()
        try:
            self.tts = await asyncio.to_thread(self.loader)
            log.info("model ready in %.1f s", time.perf_counter() - t0)
        except Exception as exc:   # surfaced by /health
            log.exception("model load failed")
            self.load_error = f"{type(exc).__name__}: {exc}"

    async def _cleanup(self, app):
        for job in list(self._jobs):
            job.close()
        if self.owns_tts and self.tts is not None:
            await asyncio.to_thread(self.tts.close)

    @property
    def engine(self):
        return self.tts.engine

    def ready(self):
        if self.tts is None:
            raise ApiError(503, "unavailable", self.load_error or "model is loading", retry_after=5)
        failed = getattr(self.engine, "failed", None)
        if failed:
            raise ApiError(503, "unavailable", f"engine failed: {failed}")

    def resolve_voice(self, voice):
        if voice is None or voice == "":
            return "plain"
        if not isinstance(voice, str):
            raise bad("voice must be a string (a registered voice id)", "voice")
        reg = self.engine.registry
        if voice in reg:
            return voice
        low = voice.lower()
        for vid in reg.ids():              # named voices match case-insensitively ("asha" -> "Asha")
            if vid.lower() == low:
                try:
                    if is_named(reg.get(vid)):
                        return vid
                except KeyError:
                    pass
        if low in OPENAI_VOICES:
            return "plain"
        raise bad(f"unknown voice {voice!r}; registered voices: {', '.join(reg.ids())}", "voice")

    # ------------------------------------------------------------------ admission
    async def prepare_text(self, text, lang, normalize=True):
        """Apply the bundled normaliser (unless disabled globally or for this request).  Only the user's text is
        normalised: a named voice's "{Name}: " prefix is added by the engine."""
        if not (self.normalise and normalize):
            return text
        fn = _normaliser()
        out = fn(text, lang) if len(text) <= NORMALISE_INLINE_CHARS else await asyncio.to_thread(fn, text, lang)
        self.stats["normalised"] += 1
        return out if out.strip() else text

    def submit(self, text, lang, voice, max_frames, rid, max_lag_s, on_overflow=None, sampling=None):
        """Admit and submit a request.  Raises ApiError (overloaded / unavailable / invalid)."""
        if len(self._waiting) >= self.max_pending:
            self.stats["rejected_pending"] += 1
            raise ApiError(503, "overloaded", f"server overloaded: {len(self._waiting)} requests already waiting "
                                              f"for audio (max_pending={self.max_pending}); retry later", retry_after=1)
        job = Job(self, rid, asyncio.get_running_loop(), max_lag_s, on_overflow)
        try:
            kw = {"sampling": sampling} if sampling else {}
            job.req = self.engine.submit(text, lang, voice, on_event=job.on_event, max_frames=max_frames, rid=rid, **kw)
        except KeyError as exc:
            raise bad(str(exc).strip("'\""), "voice")
        except ValueError as exc:
            raise bad(str(exc))
        except RuntimeError as exc:
            raise ApiError(503, "unavailable", str(exc))
        self._waiting.add(job)
        self._jobs.add(job)
        self.stats["submitted"] += 1
        return job

    async def first_event(self, job, gone=None):
        """Wait for the job's first event within the queue deadline; on expiry cancel it and raise Overloaded."""
        limit = job.t_submit + self.deadline
        while True:
            remaining = limit - time.perf_counter()
            if remaining <= 0:
                job.close()
                self.stats["rejected_deadline"] += 1
                raise Overloaded(f"no audio within the queue deadline of {self.deadline * 1000:.0f} ms; retry later")
            try:
                return await job.get(timeout=min(remaining, 0.2))
            except asyncio.TimeoutError:
                if gone is not None and gone():
                    raise ConnectionResetError("client went away")

    # ------------------------------------------------------------------ HTTP
    async def health(self, request):
        srv = {"waiting": len(self._waiting), "jobs": len(self._jobs), "ws_sockets": self.sockets,
               "max_pending": self.max_pending, "queue_deadline_ms": self.deadline * 1000, "normalise": self.normalise,
               "formats": [f for f in FORMATS if codec_support().get(f) is None], **self.stats}
        if self.tts is None:
            return web.json_response({"ready": False, "loading": self.load_error is None, "failed": self.load_error,
                                      "server": srv}, status=503)
        try:
            state = dict(self.engine.health())
        except Exception as exc:
            state = {"ready": False, "failed": f"health: {type(exc).__name__}: {exc}"}
        state["server"] = srv
        return web.json_response(state, status=200 if state.get("ready") else 503)

    async def models(self, request):
        return web.json_response({"object": "list", "data": [{"id": "maya3", "object": "model", "owned_by": "maya"}]})

    async def _json(self, request):
        try:
            body = await request.json()
        except Exception:
            raise bad("request body must be a JSON object")
        if not isinstance(body, dict):
            raise bad("request body must be a JSON object")
        return body

    async def speech(self, request):
        try:
            return await self._speech(request)
        except ApiError as exc:
            return exc.response()

    async def _speech(self, request):
        self.ready()
        body = await self._json(request)
        text = parse_text(body.get("input"), "input")
        lang = parse_lang(body, text)
        fmt = body.get("response_format", "wav") or "wav"
        usable = [f for f in FORMATS if codec_support().get(f) is None]
        if not isinstance(fmt, str) or fmt.lower() not in FORMATS:
            why = UNSUPPORTED_FORMATS.get(fmt.lower()) if isinstance(fmt, str) else None
            hint = f" is not supported ({why})" if why else ""
            raise bad(f"response_format {fmt!r}{hint}; use one of: {', '.join(usable)}", "response_format")
        fmt = fmt.lower()
        if codec_support().get(fmt):
            raise bad(f"response_format {fmt!r} is not available on this server ({codec_support()[fmt]}); use one of: "
                      f"{', '.join(usable)}", "response_format")
        speed = body.get("speed", 1.0)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or float(speed) != 1.0:
            raise bad("speed must be 1.0 (speed control is not supported)", "speed")
        sfmt = body.get("stream_format", "audio") or "audio"
        if sfmt not in ("audio", "sse"):
            raise bad("stream_format must be 'audio' or 'sse'", "stream_format")
        if sfmt == "sse" and fmt not in STREAMED_FORMATS:
            raise bad(f"stream_format 'sse' supports response_format {', '.join(STREAMED_FORMATS)}", "stream_format")
        voice = self.resolve_voice(body.get("voice"))
        max_frames = parse_max_frames(body.get("max_frames"))
        sampling = parse_sampling(body)
        normalize = parse_normalize(body)
        rid = f"h-{uuid.uuid4().hex[:16]}"
        n_chars = len(text)
        text = await self.prepare_text(text, lang, normalize)
        enc = StreamEncoder(fmt) if fmt in STREAMED_FORMATS else None

        def abort():
            if request.transport is not None:
                request.transport.abort()
        job = self.submit(text, lang, voice, max_frames, rid, self.http_max_lag_s, on_overflow=abort, sampling=sampling)
        gone = lambda: request.transport is None or request.transport.is_closing()
        resp = None
        try:
            try:
                kind, payload = await self.first_event(job, gone)
            except Overloaded as exc:
                raise ApiError(503, "overloaded", str(exc), retry_after=1)
            if kind == "error":
                code = payload.get("code")
                status = 400 if code == "invalid_request" else 503
                raise ApiError(status, code or "unavailable", payload.get("error") or "generation failed")
            headers = {"Content-Type": FORMATS[fmt], "X-Sample-Rate": str(SR), "X-Channels": "1",
                       "X-Encoding": "pcm_s16le" if fmt in ("wav", "pcm") else fmt, "X-Maya3-Language": lang,
                       "X-Maya3-Voice": voice, "X-Maya3-Normalized": "1" if (self.normalise and normalize) else "0",
                       "X-Request-Id": rid, "Cache-Control": "no-store"}
            if fmt == "flac":
                pcm = bytearray()
                while kind == "audio":
                    pcm += payload.tobytes()
                    kind, payload = await job.get()
                if kind != "end":
                    raise ApiError(503, payload.get("code") or "unavailable", payload.get("error") or "generation failed")
                data = await asyncio.to_thread(_encode_flac, bytes(pcm))
                self._log_end(rid, "http", lang, voice, n_chars, job, payload)
                return web.Response(body=data, headers=headers)
            sse = sfmt == "sse"
            if sse:
                headers["Content-Type"] = "text/event-stream"

            async def encode(fn, *a):
                return await asyncio.to_thread(fn, *a) if enc.heavy else fn(*a)
            try:
                head = await encode(enc.start)
            except Exception as exc:
                log.exception("%s: %s encoder failed to start", rid, fmt)
                raise ApiError(503, "unavailable", f"{fmt} encoder failed: {type(exc).__name__}: {exc}")
            resp = web.StreamResponse(headers=headers)
            resp.enable_chunked_encoding()
            await resp.prepare(request)

            async def out(data):
                if not data:
                    return
                if sse:
                    data = b"data: " + json.dumps({"type": "speech.audio.delta",
                                                   "audio": base64.b64encode(data).decode()}).encode() + b"\n\n"
                await resp.write(data)
            await out(head)
            while True:
                if kind == "audio":
                    data, _ = job.take_audio(payload, self.coalesce)
                    await out(await encode(enc.feed, data))
                elif kind == "end":
                    await out(await encode(enc.finish))
                    if sse:
                        await resp.write(b"data: " + json.dumps({"type": "speech.audio.done", "usage": {
                            "audio_s": payload.get("audio_s"), "frames": payload.get("frames"),
                            "reason": payload.get("reason")}}).encode() + b"\n\n")
                    await resp.write_eof()
                    self._log_end(rid, "http", lang, voice, n_chars, job, payload)
                    return resp
                elif kind == "error":
                    # Never turn a failed utterance into a well-formed end of stream.
                    log.warning("%s: error after audio started: %s", rid, payload)
                    self.stats["aborted_midstream"] += 1
                    abort()
                    return resp
                kind, payload = await job.get()
        except asyncio.CancelledError:
            self.stats["client_gone"] += 1
            raise
        except ConnectionResetError:        # client went away (or was dropped as a slow consumer)
            self.stats["client_gone"] += 1
            return resp if resp is not None else web.Response(status=499)
        except ApiError:
            if resp is not None:   # headers already sent: abort instead of a JSON body
                abort()
                return resp
            raise
        finally:
            if job.overflowed:
                self.stats["slow_consumer"] += 1
            job.close()
            if enc is not None:
                enc.close()

    def _log_end(self, rid, via, lang, voice, chars, job, info):
        self.stats["completed"] += 1
        ttfa = (job.t_first - job.t_submit) * 1000 if job.t_first else None
        log.info("%s %s lang=%s voice=%s chars=%d ttfa_ms=%s audio_s=%.2f reason=%s", rid, via, lang, voice, chars,
                 f"{ttfa:.0f}" if ttfa is not None else "-", info.get("audio_s") or 0.0, info.get("reason"))

    # ------------------------------------------------------------------ voices
    async def list_voices(self, request):
        try:
            self.ready()
        except ApiError as exc:
            return exc.response()
        reg = self.engine.registry
        out = []
        for vid in reg.ids():
            try:
                v = reg.get(vid)
            except KeyError:
                continue
            out.append(voice_info(vid, v))
        return web.json_response({"voices": out})

    async def add_voice(self, request):
        try:
            self.ready()
            fields, audio = await self._voice_fields(request)
            vid = fields.get("voice_id") or fields.get("id")
            if not isinstance(vid, str) or not VOICE_ID_RE.match(vid):
                raise bad("voice_id must be 1-64 characters of letters, digits, '_', '.', '-' (starting with a "
                          "letter or digit)", "voice_id")
            if vid == "plain":
                raise bad("'plain' is built in and cannot be replaced", "voice_id")
            replace = fields.get("replace") in (True, "true", "1", 1)
            if vid in self.engine.registry and not replace:
                raise ApiError(409, "voice_exists", f"voice {vid!r} already exists; pass replace=true to overwrite",
                               "voice_id")
            mode = fields.get("mode") or "icl_spk"
            mode = MODE_ALIASES.get(mode, mode)
            if mode not in MODES:
                raise bad(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}", "mode")
            lang = fields.get("lang") or fields.get("language")
            if lang is not None and lang not in LANGS:
                raise bad(f"unsupported lang {lang!r}; supported: {', '.join(LANGS)}", "lang")
            ref_text = fields.get("ref_text")
            if mode in ("icl", "icl_spk") and (not isinstance(ref_text, str) or not ref_text.strip() or not lang):
                raise bad(f"mode {mode} needs ref_text (the transcript of the clip) and lang (the clip's language)",
                          "ref_text")
            if mode != "plain" and audio is None:
                raise bad(f"mode {mode} needs reference audio (ref_audio_b64 or a multipart file 'ref_audio')",
                          "ref_audio_b64")
            ref = await asyncio.to_thread(_decode_audio, audio) if audio is not None else None
            t0 = time.perf_counter()
            async with self._voice_lock:
                try:
                    await asyncio.to_thread(self.tts.register_voice, vid, ref_audio=ref,
                                            ref_text=ref_text.strip() if isinstance(ref_text, str) else None,
                                            lang=lang, mode=mode)
                except (ValueError, KeyError) as exc:
                    raise bad(f"could not register voice: {exc}")
            self.stats["voices_registered"] += 1
            info = voice_info(vid, self.engine.registry.get(vid))
            info["ref_lang"] = info["ref_lang"] or (lang if mode in ("icl", "icl_spk") else None)
            return web.json_response({**info, "register_s": round(time.perf_counter() - t0, 3)}, status=201)
        except ApiError as exc:
            return exc.response()

    async def _voice_fields(self, request):
        if request.content_type.startswith("multipart/"):
            fields, audio = {}, None
            reader = await request.multipart()
            async for part in reader:
                if part.name == "ref_audio":
                    audio = bytes(await part.read())
                elif part.name:
                    fields[part.name] = await part.text()
            return fields, audio
        body = await self._json(request)
        audio = None
        b64 = body.get("ref_audio_b64")
        if b64 is not None:
            if not isinstance(b64, str):
                raise bad("ref_audio_b64 must be a base64 string of a wav/flac file", "ref_audio_b64")
            try:
                audio = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                raise bad("ref_audio_b64 is not valid base64", "ref_audio_b64")
        return body, audio

    async def delete_voice(self, request):
        try:
            self.ready()
            vid = request.match_info["voice_id"]
            if vid == "plain":
                raise bad("the plain voice cannot be removed", "voice_id")
            if vid not in self.engine.registry:
                raise ApiError(404, "voice_not_found", f"unknown voice {vid!r}", "voice_id")
            async with self._voice_lock:
                await asyncio.to_thread(self.tts.remove_voice, vid)
            return web.json_response({"id": vid, "deleted": True})
        except ApiError as exc:
            return exc.response()

    # ------------------------------------------------------------------ WebSocket
    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=64 * 1024)
        await ws.prepare(request)
        sock = _Socket(self, ws, request)
        self.sockets += 1
        sock.sender = asyncio.create_task(sock.send_loop())
        sock.put({"type": "metadata", "sample_rate": SR, "encoding": "pcm_s16le", "channels": 1,
                  "max_contexts": self.ws_max_contexts, "languages": list(LANGS)})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    sock.handle(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    sock.error(None, "invalid_request", "send JSON text messages (speak / cancel / ping)")
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.sockets -= 1
            await sock.shutdown()
        return ws


class _Context:
    __slots__ = ("cid", "job", "task", "binary", "stop", "samples")

    def __init__(self, cid, binary):
        self.cid, self.binary = cid, binary
        self.job = self.task = self.stop = None
        self.samples = 0


class _Socket:
    """One WS connection: a reader (the handler), one pump task per context, one sender with a bounded queue."""

    def __init__(self, srv, ws, request):
        self.srv, self.ws, self.request = srv, ws, request
        self.contexts = {}
        self.out = collections.deque()     # (cid or None, kind, data)
        self.wake = asyncio.Event()
        self.room = asyncio.Event()
        self.room.set()
        self.closed = False
        self.sender = None

    # -- sending
    def put(self, msg, cid=None, audio=False):
        self.out.append((cid, audio, msg))
        self.wake.set()

    async def put_audio(self, ctx, msg):
        while len(self.out) >= self.srv.ws_send_queue and not self.closed and ctx.stop is None:
            self.room.clear()
            await self.room.wait()
        if self.closed or ctx.stop is not None:
            return False
        self.put(msg, ctx.cid, True)
        return True

    def error(self, cid, code, message):
        self.put({"type": "error", "context_id": cid, "code": code, "error": message})

    async def send_loop(self):
        ws = self.ws
        try:
            while True:
                while not self.out:
                    self.wake.clear()
                    await self.wake.wait()
                _, _, msg = self.out.popleft()
                if len(self.out) < self.srv.ws_send_queue:
                    self.room.set()
                if ws.closed:
                    continue
                coro = ws.send_bytes(msg) if isinstance(msg, bytes) else ws.send_str(json.dumps(msg))
                try:
                    await asyncio.wait_for(coro, self.srv.ws_send_timeout_s)
                except asyncio.TimeoutError:
                    log.warning("ws send blocked > %.0f s; dropping the connection", self.srv.ws_send_timeout_s)
                    self.abort()
                    return
                except (ConnectionResetError, RuntimeError):
                    self.abort()
                    return
        except asyncio.CancelledError:
            pass

    def abort(self):
        self.closed = True
        self.room.set()
        for ctx in self.contexts.values():
            self._stop(ctx, "disconnected")
        if self.request.transport is not None:
            self.request.transport.abort()

    # -- receiving
    def handle(self, data):
        cid = None
        try:
            try:
                msg = json.loads(data)
            except ValueError:
                raise bad("message is not valid JSON")
            if not isinstance(msg, dict):
                raise bad("message must be a JSON object")
            cid = msg.get("context_id")
            if cid is not None and (not isinstance(cid, str) or not 1 <= len(cid) <= 128):
                cid = None
                raise bad("context_id must be a string of 1-128 characters", "context_id")
            kind = msg.get("type")
            if kind == "ping":
                self.put({"type": "pong", **({"context_id": cid} if cid else {})})
            elif kind == "cancel":
                ctx = self.contexts.get(cid)
                if ctx is not None:        # unknown / finished contexts: idempotent no-op
                    self._stop(ctx, "cancelled")
            elif kind == "speak":
                self._speak(cid, msg)
            else:
                raise bad("unknown message type; use speak, cancel or ping", "type")
        except ApiError as exc:
            self.error(cid, exc.code, exc.message)

    def _speak(self, cid, msg):
        srv = self.srv
        if cid is None:
            raise bad("speak needs a context_id", "context_id")
        if cid in self.contexts:
            raise bad(f"context {cid!r} is still active; use a new context_id", "context_id")
        if len(self.contexts) >= srv.ws_max_contexts:
            raise ApiError(429, "too_many_contexts",
                           f"at most {srv.ws_max_contexts} concurrent contexts per connection")
        srv.ready()
        text = parse_text(msg.get("text"), "text")
        lang = parse_lang(msg, text)
        voice = srv.resolve_voice(msg.get("voice"))
        max_frames = parse_max_frames(msg.get("max_frames"))
        binary = msg.get("binary", False)
        if not isinstance(binary, bool):
            raise bad("binary must be true or false", "binary")
        sampling = parse_sampling(msg)
        normalize = parse_normalize(msg)
        ctx = _Context(cid, binary)
        chars = len(text)
        deferred = None
        if srv.normalise and normalize and len(text) > NORMALISE_INLINE_CHARS:
            deferred = (text, sampling, max_frames)        # normalised on a worker thread, then submitted by _pump
        else:
            if srv.normalise and normalize:
                text = _normaliser()(text, lang) or text
                srv.stats["normalised"] += 1
            self._submit(ctx, text, lang, voice, max_frames, sampling)
        self.contexts[cid] = ctx
        ctx.task = asyncio.create_task(self._pump(ctx, lang, voice, chars, deferred))

    def _submit(self, ctx, text, lang, voice, max_frames, sampling):
        ctx.job = self.srv.submit(text, lang, voice, max_frames, f"w-{uuid.uuid4().hex[:12]}", self.srv.ws_max_lag_s,
                                  on_overflow=lambda: self._stop(ctx, "slow_consumer"), sampling=sampling)

    def _stop(self, ctx, reason):
        if ctx.stop is None:
            ctx.stop = reason
            if ctx.job is not None:
                ctx.job.close()
                ctx.job.interrupt()
            # drop audio of this context that has not been sent yet
            if any(c == ctx.cid and a for c, a, _ in self.out):
                self.out = collections.deque(x for x in self.out if not (x[0] == ctx.cid and x[1]))
            self.room.set()

    async def _pump(self, ctx, lang, voice, chars, deferred=None):
        srv, cid = self.srv, ctx.cid
        job = None
        try:
            if deferred is not None:
                text, sampling, max_frames = deferred
                text = await srv.prepare_text(text, lang)
                if ctx.stop is not None:
                    if ctx.stop == "cancelled":
                        self.put({"type": "end", "context_id": cid, "reason": "cancelled", "frames": 0, "audio_s": 0.0,
                                  "ttfa_ms": None})
                        srv.stats["cancelled"] += 1
                    return
                try:
                    self._submit(ctx, text, lang, voice, max_frames, sampling)
                except ApiError as exc:
                    self.error(cid, exc.code, exc.message)
                    return
            job = ctx.job
            try:
                kind, payload = await srv.first_event(job)
            except Overloaded as exc:
                self.error(cid, "overloaded", str(exc))
                return
            while ctx.stop is None:
                if kind == "audio":
                    pcm, n = job.take_audio(payload, srv.coalesce)
                    msg = ws_binary_frame(cid, pcm) if ctx.binary else \
                        {"type": "audio", "context_id": cid, "audio": base64.b64encode(pcm).decode()}
                    if await self.put_audio(ctx, msg):
                        ctx.samples += n
                elif kind == "end":
                    self.put({"type": "end", "context_id": cid, "reason": payload.get("reason"),
                              "frames": payload.get("frames"), "audio_s": payload.get("audio_s"),
                              "ttfa_ms": self._ttfa(job, payload)})
                    srv._log_end(job.rid, "ws", lang, voice, chars, job, payload)
                    return
                elif kind == "error":
                    code = payload.get("code") or "internal"
                    self.error(cid, code, payload.get("error") or code)
                    return
                kind, payload = await job.get()
            if ctx.stop == "cancelled":
                self.put({"type": "end", "context_id": cid, "reason": "cancelled", "frames": ctx.samples // FRAME,
                          "audio_s": round(ctx.samples / SR, 3), "ttfa_ms": self._ttfa(job, {})})
                srv.stats["cancelled"] += 1
            elif ctx.stop == "slow_consumer":
                srv.stats["slow_consumer"] += 1
                self.error(cid, "slow_consumer", f"client fell more than {srv.ws_max_lag_s:.0f} s of audio behind; "
                                                 "request cancelled")
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("ws context %s failed", cid)
            self.error(cid, "internal", f"{type(exc).__name__}: {exc}")
        finally:
            if job is not None:
                job.close()
            if self.contexts.get(cid) is ctx:
                del self.contexts[cid]

    @staticmethod
    def _ttfa(job, info):
        if info.get("ttfa_s") is not None:
            return round(info["ttfa_s"] * 1000, 1)
        return round((job.t_first - job.t_submit) * 1000, 1) if job.t_first else None

    async def shutdown(self):
        self.closed = True
        self.room.set()
        ctxs = list(self.contexts.values())
        for ctx in ctxs:
            self._stop(ctx, "disconnected")
            ctx.task.cancel()
        await asyncio.gather(*(c.task for c in ctxs), return_exceptions=True)
        if self.sender is not None:
            self.sender.cancel()
            await asyncio.gather(self.sender, return_exceptions=True)


# ---------------------------------------------------------------------------------------------------- audio io
_CODECS = None


def codec_support():
    """{format: None (ok) | reason} for the formats encoded by libsndfile (checked once)."""
    global _CODECS
    if _CODECS is None:
        out = {}
        try:
            import soundfile as sf
            formats = sf.available_formats()
            for fmt, (major, sub) in SF_CODECS.items():
                ok = major in formats and sub in sf.available_subtypes(major)
                out[fmt] = None if ok else f"this libsndfile ({sf.__libsndfile_version__}) cannot write {major}/{sub}"
            out["flac"] = None if "FLAC" in formats else "this libsndfile cannot write FLAC"
        except Exception as exc:   # soundfile missing / broken
            out = {f: f"soundfile unavailable ({type(exc).__name__}: {exc})" for f in ("mp3", "opus", "flac")}
        _CODECS = out
    return _CODECS


class _AppendSink(io.RawIOBase):
    """Write-only file object for libsndfile that hands out bytes as soon as they are appended.  libsndfile seeks back
    at close to patch headers it already wrote (the MP3 LAME/Info frame); bytes already handed out cannot change, so
    such rewrites are dropped.  For MP3 the blank Info placeholder (the first write) is withheld entirely: a stream of
    plain CBR frames is valid MP3, while a never-filled Info frame misleads decoders about the length."""

    def __init__(self, skip_blank_first=False):
        self.buf = bytearray()      # bytes not yet taken
        self.base = 0               # stream offset of buf[0]
        self.pos = 0
        self.skip = skip_blank_first
        self.n_writes = 0

    def writable(self):
        return True

    def seekable(self):
        return True

    def readable(self):
        return False

    def write(self, b):
        b = bytes(b)
        n = len(b)
        self.n_writes += 1
        end = self.base + len(self.buf)
        if self.n_writes == 1 and self.skip and self.pos == 0 and n > 4 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0 \
                and not any(b[4:]):
            self.base = n           # withheld placeholder frame
        elif self.pos >= end:
            if self.pos > end:      # never happens for append-only encoders; keep the stream well-formed anyway
                self.buf.extend(b"\0" * (self.pos - end))
            self.buf.extend(b)
        elif self.pos + n > end:    # partial overlap with handed-out bytes: keep only the new tail
            self.buf.extend(b[end - self.pos:])
        self.pos += n
        return n

    def seek(self, offset, whence=0):
        end = self.base + len(self.buf)
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else end + offset)
        return self.pos

    def tell(self):
        return self.pos

    def take(self):
        out = bytes(self.buf)
        self.base += len(self.buf)
        self.buf.clear()
        return out


class StreamEncoder:
    """Incremental encoder for one HTTP response: start() -> header bytes, feed(pcm s16le) -> bytes, finish() -> tail.
    wav/pcm are pass-through (cheap, called on the event loop); mp3/opus run libsndfile (call feed/finish off-loop;
    `heavy` is True)."""

    def __init__(self, fmt):
        self.fmt = fmt
        self.heavy = fmt in SF_CODECS
        self.sf = self.sink = None
        self.lock = threading.Lock()
        self.closed = False

    def start(self):
        if self.fmt == "wav":
            return wav_header()
        if not self.heavy:
            return b""
        import soundfile as sf
        major, sub = SF_CODECS[self.fmt]
        self.sink = _AppendSink(skip_blank_first=self.fmt == "mp3")
        kw = {"bitrate_mode": "CONSTANT", "compression_level": MP3_LEVEL} if self.fmt == "mp3" else {}
        self.sf = sf.SoundFile(self.sink, "w", samplerate=SR, channels=1, format=major, subtype=sub, **kw)
        return self.sink.take()

    def feed(self, pcm):
        if not self.heavy:
            return pcm
        import numpy as np
        with self.lock:
            if self.closed:
                return b""
            self.sf.write(np.frombuffer(pcm, np.int16))
            return self.sink.take()

    def finish(self):
        if not self.heavy:
            return b""
        with self.lock:
            if self.closed:
                return b""
            self.closed = True
            self.sf.close()
            return self.sink.take()

    def close(self):
        if self.heavy and self.sf is not None:
            with self.lock:
                if not self.closed:
                    self.closed = True
                    try:
                        self.sf.close()
                    except Exception:
                        pass


def _encode_flac(pcm):
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(pcm, np.int16), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _decode_audio(data):
    """wav/flac/ogg bytes -> (float32 mono array, sample rate); Maya3TTS resamples to 24 kHz."""
    import numpy as np
    import soundfile as sf
    try:
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise bad(f"could not decode reference audio ({exc}); send a wav or flac file", "ref_audio_b64")
    w = w.mean(1).astype(np.float32)
    dur = len(w) / sr if sr else 0
    if dur < 0.5:
        raise bad(f"reference audio is {dur:.2f} s; need at least 0.5 s (3-10 s works best)", "ref_audio_b64")
    if dur > 30:
        raise bad(f"reference audio is {dur:.1f} s; the limit is 30 s (3-10 s works best)", "ref_audio_b64")
    return (w, int(sr))


# ---------------------------------------------------------------------------------------------------- CLI
def load_voices_file(tts, path):
    path = Path(path)
    spec = json.loads(path.read_text())
    items = spec["voices"] if isinstance(spec, dict) else spec
    for it in items:
        ref = it.get("ref_audio")
        if ref is not None and not Path(ref).is_absolute():
            ref = str(path.parent / ref)
        kw = {"meta": it["meta"]} if isinstance(it.get("meta"), dict) else {}
        tts.register_voice(it["voice_id"], ref_audio=ref, ref_text=it.get("ref_text"), lang=it.get("lang"),
                           mode=it.get("mode", "icl_spk"), **kw)
        log.info("registered voice %s (%s)", it["voice_id"], it.get("mode", "icl_spk"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Maya3 TTS server (OpenAI-compatible HTTP + WebSocket streaming)")
    ap.add_argument("--ckpt", required=True, help="model checkpoint, or a Maya3 release directory (then --backbone-dir "
                                                  "and --mimi-dir may be omitted)")
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--mimi-dir", default=None)
    ap.add_argument("--ecapa-dir", default=None, help="needed to register reference-audio voices")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seats", type=int, default=64, help="concurrent generations batched on the GPU")
    ap.add_argument("--capacity", type=int, default=None, help="EngineConfig.capacity (KV tokens per seat)")
    ap.add_argument("--depth-dtype", choices=["fp16", "bf16", "fp32"], default=None)
    ap.add_argument("--max-pending", type=int, default=None,
                    help="max requests waiting for their first audio (default: max-seats)")
    ap.add_argument("--queue-deadline-ms", type=float, default=2000)
    ap.add_argument("--voices", default=None, help="voices.json to register at startup")
    ap.add_argument("--ws-max-contexts", type=int, default=8, help="concurrent contexts per WebSocket")
    ap.add_argument("--ws-max-lag-s", type=float, default=20.0,
                    help="cancel a WS request when this much generated audio is waiting to be sent")
    ap.add_argument("--http-max-lag-s", type=float, default=120.0)
    ap.add_argument("--busy-frac", type=float, default=None, help="EngineConfig.busy_frac (admission smoothing under load)")
    ap.add_argument("--busy-admit", type=int, default=None, help="EngineConfig.busy_admit (prefills per tick when busy)")
    ap.add_argument("--no-normalise", "--no-normalize", dest="no_normalise", action="store_true",
                    help="turn the bundled text normaliser off for every request (default: on; per request: "
                         "\"normalize\": false)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if args.max_seats < 1 or args.queue_deadline_ms <= 0 or (args.max_pending is not None and args.max_pending < 1):
        ap.error("max-seats and max-pending must be >= 1 and queue-deadline-ms > 0")
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def loader():
        from maya3.api import Maya3TTS
        cfg = {"max_seats": args.max_seats}
        if args.capacity:
            cfg["capacity"] = args.capacity
        if args.depth_dtype:
            cfg["depth_dtype"] = args.depth_dtype
        if args.busy_frac is not None:
            cfg["busy_frac"] = args.busy_frac
        if args.busy_admit is not None:
            cfg["busy_admit"] = args.busy_admit
        tts = Maya3TTS.from_pretrained(args.ckpt, args.backbone_dir, args.mimi_dir, ecapa_dir=args.ecapa_dir,
                                       device=args.device, **cfg)
        if args.voices:
            load_voices_file(tts, args.voices)
        return tts

    srv = Maya3Server(loader=loader, max_pending=args.max_pending or args.max_seats,
                      queue_deadline_ms=args.queue_deadline_ms, ws_max_contexts=args.ws_max_contexts,
                      ws_max_lag_s=args.ws_max_lag_s, http_max_lag_s=args.http_max_lag_s,
                      normalise=not args.no_normalise)
    web.run_app(srv.app(), host=args.host, port=args.port, access_log=None, handler_cancellation=True,
                print=lambda *a: log.info("listening on http://%s:%d", args.host, args.port))


if __name__ == "__main__":
    main()
