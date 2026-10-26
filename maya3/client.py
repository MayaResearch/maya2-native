"""Small clients for the Maya3 server (maya3/server.py).  Audio is 24 kHz mono; WS audio is always s16le PCM, HTTP
returns wav / pcm / mp3 / opus (Ogg) / flac.  Extra keyword arguments are sent as request fields: seed, temperature,
top_k, top_p, repetition_penalty, normalize (default true on the server).

Sync (standard library only):
    from maya3.client import Maya3Client
    c = Maya3Client("http://127.0.0.1:8080")
    c.speech("नमस्ते, आप कैसे हैं?", "out.wav", language="hi")          # streamed to the file
    c.speech("Hello there.", "out.mp3", language="en", seed=7)          # format from the suffix (.ogg/.opus -> opus)
    for pcm in c.iter_speech("Hello there.", language="en"): ...       # raw PCM bytes as they arrive
    c.register_voice("asha", "asha.wav", ref_text="...", lang="hi"); c.voices(); c.remove_voice("asha")

Async (aiohttp):
    from maya3.client import Maya3Stream, aspeech
    async with Maya3Stream("ws://127.0.0.1:8080/v1/tts/stream") as s:
        async for pcm in s.speak("Hello there.", language="en", voice="plain"): ...
        # several s.speak(...) iterators may run concurrently on one socket; s.last_end[context_id] has the end info
    await aspeech("Hello", "out.wav", base_url="http://127.0.0.1:8080", language="en")
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SR = 24000


class Maya3Error(RuntimeError):
    def __init__(self, code, message, status=None):
        super().__init__(f"{code}: {message}" + (f" (HTTP {status})" if status else ""))
        self.code, self.message, self.status = code, message, status


def _speech_body(text, language, voice, response_format, max_frames, extra):
    body = {"model": "maya3", "input": text, "voice": voice, "response_format": response_format}
    if language:
        body["language"] = language
    if max_frames:
        body["max_frames"] = max_frames
    body.update(extra)
    return body


SUFFIX_FORMATS = {"wav": "wav", "pcm": "pcm", "raw": "pcm", "mp3": "mp3", "opus": "opus", "ogg": "opus", "flac": "flac"}


def _format_for(out, response_format):
    if response_format:
        return response_format
    if not out:
        return "wav"
    return SUFFIX_FORMATS.get(Path(out).suffix.lstrip(".").lower(), "wav")


def _http_error(status, raw):
    try:
        err = json.loads(raw)["error"]
        return Maya3Error(err.get("code"), err.get("message"), status)
    except Exception:
        return Maya3Error("http_error", raw[:300].decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw), status)


# ---------------------------------------------------------------------------------------------------- sync
class Maya3Client:
    def __init__(self, base_url="http://127.0.0.1:8080", timeout=300):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _open(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        hdrs = {"Content-Type": "application/json"} if isinstance(body, dict) else {}
        hdrs.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise _http_error(e.code, e.read()) from None

    def _json(self, method, path, body=None):
        with self._open(method, path, body) as r:
            return json.loads(r.read())

    def iter_speech(self, text, language=None, voice="plain", response_format="pcm", max_frames=None,
                    chunk_bytes=3840, **extra):
        """Yield the response body (pcm: raw s16le) in chunks as it streams."""
        r = self._open("POST", "/v1/audio/speech", _speech_body(text, language, voice, response_format, max_frames, extra))
        with r:
            while True:
                b = r.read1(chunk_bytes) if hasattr(r, "read1") else r.read(chunk_bytes)
                if not b:
                    return
                yield b

    def speech(self, text, out=None, language=None, voice="plain", response_format=None, max_frames=None, **extra):
        """Synthesize to `out` (path; format from its suffix: .wav/.pcm/.mp3/.opus/.ogg/.flac, else wav) or return the
        bytes when out is None."""
        fmt = _format_for(out, response_format)
        chunks = self.iter_speech(text, language, voice, fmt, max_frames, chunk_bytes=65536, **extra)
        if out is None:
            return b"".join(chunks)
        with open(out, "wb") as f:
            for b in chunks:
                f.write(b)
        return out

    def health(self):
        try:
            return self._json("GET", "/health")
        except Maya3Error as e:
            if e.status == 503:
                return {"ready": False, "error": e.message}
            raise

    def voices(self):
        return self._json("GET", "/v1/voices")["voices"]

    def register_voice(self, voice_id, ref_audio, ref_text=None, lang=None, mode="icl_spk", replace=False):
        """ref_audio: path or wav/flac bytes."""
        data = ref_audio if isinstance(ref_audio, (bytes, bytearray)) else Path(ref_audio).read_bytes()
        return self._json("POST", "/v1/voices", {"voice_id": voice_id, "ref_audio_b64": base64.b64encode(data).decode(),
                                                 "ref_text": ref_text, "lang": lang, "mode": mode, "replace": replace})

    def remove_voice(self, voice_id):
        return self._json("DELETE", f"/v1/voices/{voice_id}")


# ---------------------------------------------------------------------------------------------------- async
async def aspeech(text, out=None, base_url="http://127.0.0.1:8080", language=None, voice="plain", response_format=None,
                  max_frames=None, session=None, **extra):
    """Async HTTP synthesis to a file (streamed) or to bytes."""
    import aiohttp
    fmt = _format_for(out, response_format)
    own = session is None
    session = session or aiohttp.ClientSession()
    try:
        async with session.post(base_url.rstrip("/") + "/v1/audio/speech",
                                json=_speech_body(text, language, voice, fmt, max_frames, extra)) as r:
            if r.status != 200:
                raise _http_error(r.status, await r.read())
            if out is None:
                return await r.read()
            with open(out, "wb") as f:
                async for b in r.content.iter_any():
                    f.write(b)
            return out
    finally:
        if own:
            await session.close()


def parse_binary_frame(data):
    """WS binary audio frame -> (context_id, pcm bytes)."""
    (n,) = struct.unpack_from("<I", data)
    return data[4:4 + n].decode(), data[4 + n:]


class Maya3Stream:
    """WebSocket client; demultiplexes concurrent contexts on one socket."""

    def __init__(self, url="ws://127.0.0.1:8080/v1/tts/stream", binary=False, session=None):
        self.url, self.binary = url, binary
        self._session, self._own = session, session is None
        self.ws = None
        self.metadata = None
        self.last_end = {}
        self._queues = {}
        self._reader = None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *exc):
        await self.close()

    async def connect(self):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self.ws = await self._session.ws_connect(self.url, max_msg_size=16 * 1024 * 1024)
        self.metadata = await self.ws.receive_json()
        self._reader = asyncio.create_task(self._read())
        return self

    async def _read(self):
        import aiohttp
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    cid, pcm = parse_binary_frame(msg.data)
                    item = ("audio", pcm)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    cid = m.get("context_id")
                    if m.get("type") == "audio":
                        item = ("audio", base64.b64decode(m["audio"]))
                    else:
                        item = (m.get("type"), m)
                else:
                    break
                q = self._queues.get(cid)
                if q is not None:
                    q.put_nowait(item)
                elif cid is None and item[0] == "error":   # socket-level error: fail every waiting context
                    for q in self._queues.values():
                        q.put_nowait(item)
        finally:
            for q in self._queues.values():
                q.put_nowait(("closed", None))

    async def speak(self, text, language=None, voice=None, context_id=None, max_frames=None, **extra):
        """Async iterator of PCM bytes for one utterance; breaking out early cancels it on the server.
        extra: seed, temperature, top_k, top_p, repetition_penalty, normalize."""
        cid = context_id or uuid.uuid4().hex[:12]
        q = self._queues[cid] = asyncio.Queue()
        msg = {"type": "speak", "context_id": cid, "text": text, "binary": self.binary}
        msg.update({k: v for k, v in extra.items() if v is not None})
        if language:
            msg["language"] = language
        if voice:
            msg["voice"] = voice
        if max_frames:
            msg["max_frames"] = max_frames
        done = False
        try:
            await self.ws.send_json(msg)
            while True:
                kind, payload = await q.get()
                if kind == "audio":
                    yield payload
                elif kind == "end":
                    done = True
                    self.last_end[cid] = payload
                    return
                elif kind == "error":
                    done = True
                    raise Maya3Error(payload.get("code"), payload.get("error"))
                elif kind == "closed":
                    done = True
                    raise Maya3Error("closed", "connection closed before the end of the utterance")
        finally:
            if not done and not self.ws.closed:
                try:
                    await self.ws.send_json({"type": "cancel", "context_id": cid})
                except Exception:
                    pass
            self._queues.pop(cid, None)

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
        if self._reader is not None:
            await asyncio.gather(self._reader, return_exceptions=True)
        if self._own and self._session is not None:
            await self._session.close()


async def stream_ws(text, url="ws://127.0.0.1:8080/v1/tts/stream", language=None, voice=None, max_frames=None, **extra):
    """One-shot WS streaming iterator of PCM bytes."""
    async with Maya3Stream(url) as s:
        async for pcm in s.speak(text, language, voice, max_frames=max_frames, **extra):
            yield pcm
