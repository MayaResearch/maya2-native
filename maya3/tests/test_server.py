"""Server/client/bench tests against FakeEngine (CPU only, ephemeral ports).

    python -m pytest maya3/tests/test_server.py -q
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import time

import aiohttp
import numpy as np
import pytest
import soundfile as sf

from maya3 import server as S
from maya3.client import Maya3Client, Maya3Error, Maya3Stream, aspeech, parse_binary_frame
from maya3.tests.fakes import FRAME, FakeTTS, FakeVoice, named_voice

pytestmark = pytest.mark.asyncio


def frames_of(pcm: bytes):
    a = np.frombuffer(pcm, np.int16)
    assert len(a) % FRAME == 0
    return [int(x) for x in a.reshape(-1, FRAME)[:, 0]]


def wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def wait_until(pred, timeout=3.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return pred()


@pytest.fixture
def make(aiohttp_client):
    made = []

    async def _make(engine=None, **server_kw):
        # FakeEngine reads "#frames=N" directives from the text, which the normaliser would spell out: tests opt in.
        server_kw.setdefault("normalise", False)
        tts = FakeTTS(**(engine or {}))
        srv = S.Maya3Server(tts, **server_kw)
        client = await aiohttp_client(srv.app())
        made.append(tts)
        return client, srv, tts
    yield _make
    for tts in made:
        tts.close()


async def speech(client, **body):
    body.setdefault("input", "hello world")
    return await client.post("/v1/audio/speech", json=body)


# ------------------------------------------------------------------------------------------- health / misc
async def test_health_ready_and_loading(make, aiohttp_client):
    client, srv, tts = await make()
    r = await client.get("/health")
    assert r.status == 200
    h = await r.json()
    assert h["ready"] and h["max_seats"] == 4 and "server" in h
    loading = await aiohttp_client(S.Maya3Server(None).app())
    r = await loading.get("/health")
    assert r.status == 503 and (await r.json())["ready"] is False
    r = await speech(loading)
    assert r.status == 503 and (await r.json())["error"]["code"] == "unavailable"
    r = await client.get("/v1/models")
    assert (await r.json())["data"][0]["id"] == "maya3"


async def test_background_loader(aiohttp_client):
    tts = FakeTTS()
    srv = S.Maya3Server(loader=lambda: (time.sleep(0.2), tts)[1])
    client = await aiohttp_client(srv.app())
    assert (await client.get("/health")).status == 503
    assert await wait_until(lambda: srv.tts is not None)
    assert (await client.get("/health")).status == 200


# ------------------------------------------------------------------------------------------- OpenAI endpoint
async def test_speech_pcm_streams(make):
    client, srv, tts = await make(engine={"frame_s": 0.02})
    t0 = time.perf_counter()
    r = await speech(client, input="stream me #frames=30", response_format="pcm", language="en")
    assert r.status == 200
    assert r.headers["Content-Type"] == "audio/pcm" and r.headers["X-Sample-Rate"] == "24000"
    assert r.headers["X-Maya3-Language"] == "en" and r.headers["Transfer-Encoding"] == "chunked"
    first = await r.content.readexactly(2 * FRAME)
    t_first = time.perf_counter() - t0
    rest = await r.read()
    t_all = time.perf_counter() - t0
    assert frames_of(first + rest) == list(range(1, 31))
    assert t_first < t_all / 2, (t_first, t_all)   # audio arrives while generating, not at the end
    req = tts.engine.requests[-1]
    assert req.lang == "en" and req.voice.id == "plain" and req.id == r.headers["X-Request-Id"]


async def test_speech_wav_streaming_header(make):
    client, srv, tts = await make()
    r = await speech(client, input="wav #frames=7", response_format="wav", language="hi")
    assert r.status == 200 and r.headers["Content-Type"] == "audio/wav"
    body = await r.read()
    riff, size, wave, fmt, _, pcmfmt, ch, sr, _, _, bits, data, dsize = struct.unpack("<4sI4s4sIHHIIHH4sI", body[:44])
    assert (riff, wave, fmt, data) == (b"RIFF", b"WAVE", b"fmt ", b"data")
    assert size == dsize == 0xFFFFFFFF and (pcmfmt, ch, sr, bits) == (1, 1, 24000, 16)
    assert frames_of(body[44:]) == list(range(1, 8))


async def test_speech_default_format_is_wav_and_flac(make):
    client, srv, tts = await make()
    r = await speech(client, input="x #frames=4")
    assert r.status == 200 and r.headers["Content-Type"] == "audio/wav"
    r = await speech(client, input="x #frames=5", response_format="flac")
    assert r.status == 200 and r.headers["Content-Type"] == "audio/flac"
    w, sr = sf.read(io.BytesIO(await r.read()), dtype="int16")
    assert sr == 24000 and len(w) == 5 * FRAME and int(w[-1]) == 5


async def test_speech_sse(make):
    client, srv, tts = await make()
    r = await speech(client, input="x #frames=6", response_format="pcm", stream_format="sse")
    assert r.status == 200 and r.headers["Content-Type"].startswith("text/event-stream")
    events = [json.loads(l[6:]) for l in (await r.text()).splitlines() if l.startswith("data: ")]
    assert events[-1]["type"] == "speech.audio.done" and events[-1]["usage"]["frames"] == 6
    pcm = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"] == "speech.audio.delta")
    assert frames_of(pcm) == list(range(1, 7))


async def test_openai_voice_alias_and_language_detection(make):
    client, srv, tts = await make()
    r = await speech(client, input="नमस्ते दुनिया", voice="alloy", response_format="pcm", model="tts-1")
    assert r.status == 200 and r.headers["X-Maya3-Voice"] == "plain" and r.headers["X-Maya3-Language"] == "hi"
    await r.read()
    assert tts.engine.requests[-1].lang == "hi"
    r = await speech(client, input="வணக்கம்", response_format="pcm")
    assert r.headers["X-Maya3-Language"] == "ta"
    await r.read()


@pytest.mark.parametrize("body,param,frag", [
    ({"input": ""}, "input", "non-empty"),
    ({"input": "   "}, "input", "non-empty"),
    ({"input": 5}, "input", "non-empty"),
    ({"input": "x" * 4001}, "input", "4000"),
    ({"input": "hi", "language": "fr"}, "language", "unsupported language"),
    ({"input": "hi", "voice": "nobody"}, "voice", "unknown voice"),
    ({"input": "hi", "response_format": "aac"}, "response_format", "not supported"),
    ({"input": "hi", "response_format": "ogg"}, "response_format", "wav, pcm, mp3, opus, flac"),
    ({"input": "hi", "response_format": 3}, "response_format", "use one of"),
    ({"input": "hi", "speed": 1.5}, "speed", "1.0"),
    ({"input": "hi", "max_frames": 0}, "max_frames", "max_frames"),
    ({"input": "hi", "stream_format": "sse", "response_format": "flac"}, "stream_format", "sse"),
])
async def test_speech_validation(make, body, param, frag):
    client, srv, tts = await make()
    r = await client.post("/v1/audio/speech", json=body)
    assert r.status == 400
    err = (await r.json())["error"]
    assert err["param"] == param and frag in err["message"] and err["type"] == "invalid_request_error"
    assert not tts.engine.requests


async def test_speech_bad_json(make):
    client, srv, tts = await make()
    r = await client.post("/v1/audio/speech", data=b"{nope", headers={"Content-Type": "application/json"})
    assert r.status == 400 and "JSON" in (await r.json())["error"]["message"]
    r = await client.post("/v1/audio/speech", json=[1, 2])
    assert r.status == 400


async def test_speech_speed_one_accepted(make):
    client, srv, tts = await make()
    r = await speech(client, input="x #frames=3", speed=1, response_format="pcm")
    assert r.status == 200 and len(await r.read()) == 3 * FRAME * 2


async def test_midstream_error_aborts_body(make):
    client, srv, tts = await make()
    r = await speech(client, input="boom #frames=50 #error=3", response_format="pcm")
    assert r.status == 200
    with pytest.raises(aiohttp.ClientPayloadError):
        await r.read()


# ------------------------------------------------------------------------------------------- admission
async def test_overload_max_pending_503(make):
    client, srv, tts = await make(engine={"max_seats": 1, "frame_s": 0.02}, max_pending=1, queue_deadline_ms=5000)
    r1 = await speech(client, input="long #frames=500", response_format="pcm")
    assert r1.status == 200                       # holds the only seat
    t2 = asyncio.create_task(speech(client, input="second", response_format="pcm"))
    assert await wait_until(lambda: len(srv._waiting) == 1)
    r3 = await speech(client, input="third", response_format="pcm")
    assert r3.status == 503 and r3.headers.get("Retry-After") == "1"
    assert (await r3.json())["error"]["code"] == "overloaded"
    assert len(tts.engine.requests) == 2          # the rejected one never reached the engine
    r1.close()
    r2 = await t2
    assert r2.status == 200
    await r2.read()


async def test_overload_queue_deadline_503_and_cancel(make):
    client, srv, tts = await make(engine={"max_seats": 1, "frame_s": 0.02}, max_pending=8, queue_deadline_ms=200)
    r1 = await speech(client, input="long #frames=500", response_format="pcm")
    assert r1.status == 200
    t0 = time.perf_counter()
    r2 = await speech(client, input="waits", response_format="wav")
    waited = time.perf_counter() - t0
    assert r2.status == 503 and (await r2.json())["error"]["code"] == "overloaded"
    assert 0.15 < waited < 1.5
    assert await wait_until(lambda: tts.engine.requests[1].cancelled)
    assert srv.stats["rejected_deadline"] == 1 and not srv._waiting
    r1.close()


async def test_http_disconnect_cancels_engine_request(make):
    client, srv, tts = await make(engine={"frame_s": 0.01})
    r = await speech(client, input="forever #frames=100000", response_format="pcm")
    await r.content.readexactly(2 * FRAME)
    req = tts.engine.requests[-1]
    r.close()
    assert await wait_until(lambda: req.cancelled and req.done.is_set())
    assert await wait_until(lambda: not srv._jobs)


async def test_http_disconnect_while_queued_cancels(make):
    client, srv, tts = await make(engine={"max_seats": 1, "frame_s": 0.02}, queue_deadline_ms=10000)
    r1 = await speech(client, input="long #frames=500", response_format="pcm")
    task = asyncio.create_task(speech(client, input="queued", response_format="pcm"))
    assert await wait_until(lambda: len(tts.engine.requests) == 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert await wait_until(lambda: tts.engine.requests[1].cancelled)
    r1.close()


async def test_engine_failure_is_503(make):
    client, srv, tts = await make()
    tts.engine.failed = "boom"
    r = await speech(client)
    assert r.status == 503
    assert (await client.get("/health")).status == 503


# ------------------------------------------------------------------------------------------- WebSocket
async def ws_open(client):
    ws = await client.ws_connect("/v1/tts/stream")
    meta = await ws.receive_json()
    assert meta == {"type": "metadata", "sample_rate": 24000, "encoding": "pcm_s16le", "channels": 1,
                    "max_contexts": meta["max_contexts"], "languages": list(S.LANGS)}
    return ws


async def ws_collect(ws, until, timeout=10):
    """Read until `until(contexts)` is true; returns {cid: {"pcm": bytes, "msgs": [...]}}."""
    ctx = {}
    async with asyncio.timeout(timeout):
        while not until(ctx):
            m = await ws.receive()
            if m.type == aiohttp.WSMsgType.BINARY:
                cid, pcm = parse_binary_frame(m.data)
                c = ctx.setdefault(cid, {"pcm": b"", "msgs": []})
                c["pcm"] += pcm
                c["msgs"].append({"type": "audio", "binary": True})
                continue
            assert m.type == aiohttp.WSMsgType.TEXT, m
            f = json.loads(m.data)
            c = ctx.setdefault(f.get("context_id"), {"pcm": b"", "msgs": []})
            if f["type"] == "audio":
                c["pcm"] += base64.b64decode(f["audio"])
            c["msgs"].append(f)
    return ctx


def terminal(c):
    return bool(c["msgs"]) and c["msgs"][-1]["type"] in ("end", "error")


async def test_ws_multi_context(make):
    client, srv, tts = await make(engine={"frame_s": 0.005})
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "a", "text": "one #frames=20", "language": "en"})
    await ws.send_json({"type": "speak", "context_id": "b", "text": "दो #frames=11", "language": "hi", "binary": True})
    await ws.send_json({"type": "speak", "context_id": "c", "text": "three #frames=5", "lang": "te", "max_frames": 3})
    ctx = await ws_collect(ws, lambda c: all(k in c and terminal(c[k]) for k in "abc"))
    for cid, n in (("a", 20), ("b", 11), ("c", 3)):
        end = ctx[cid]["msgs"][-1]
        assert end["type"] == "end" and end["reason"] == "aeos" and end["frames"] == n
        assert end["ttfa_ms"] is not None and abs(end["audio_s"] - n * 0.08) < 1e-6
        assert frames_of(ctx[cid]["pcm"]) == list(range(1, n + 1))
        assert all(m["type"] == "audio" for m in ctx[cid]["msgs"][:-1])
    assert all(m.get("binary") for m in ctx["b"]["msgs"][:-1])
    langs = {r.text.split()[0]: r.lang for r in tts.engine.requests}
    assert langs == {"one": "en", "दो": "hi", "three": "te"}
    # a finished context id may be reused
    await ws.send_json({"type": "speak", "context_id": "a", "text": "again #frames=2"})
    ctx = await ws_collect(ws, lambda c: "a" in c and terminal(c["a"]))
    assert ctx["a"]["msgs"][-1]["frames"] == 2
    await ws.close()


async def test_ws_cancel(make):
    client, srv, tts = await make(engine={"frame_s": 0.01})
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "long", "text": "x #frames=100000"})
    await ws.send_json({"type": "speak", "context_id": "short", "text": "y #frames=30"})
    await ws_collect(ws, lambda c: "long" in c and c["long"]["pcm"])
    await ws.send_json({"type": "cancel", "context_id": "long"})
    await ws.send_json({"type": "cancel", "context_id": "never-existed"})      # ignored
    ctx = await ws_collect(ws, lambda c: all(k in c and terminal(c[k]) for k in ("long", "short")))
    end = ctx["long"]["msgs"][-1]
    assert end["type"] == "end" and end["reason"] == "cancelled"
    assert ctx["short"]["msgs"][-1]["reason"] == "aeos" and len(frames_of(ctx["short"]["pcm"])) <= 30
    long_req = next(r for r in tts.engine.requests if "100000" in r.text)
    assert await wait_until(lambda: long_req.cancelled and long_req.done.is_set())
    # nothing for 'long' after its end
    await ws.send_json({"type": "ping"})
    ctx = await ws_collect(ws, lambda c: None in c and any(m["type"] == "pong" for m in c[None]["msgs"]))
    assert "long" not in ctx
    assert srv.stats["cancelled"] == 1
    await ws.close()


async def test_ws_errors(make):
    client, srv, tts = await make(ws_max_contexts=2, engine={"frame_s": 0.02})
    ws = await ws_open(client)

    async def err(msg, cid=None):
        await (ws.send_str(msg) if isinstance(msg, str) else ws.send_json(msg))
        ctx = await ws_collect(ws, lambda c: cid in c and terminal(c[cid]))
        e = ctx[cid]["msgs"][-1]
        assert e["type"] == "error", e
        return e
    assert (await err("not json"))["code"] == "invalid_request"
    assert "context_id" in (await err({"type": "speak", "text": "hi"}))["error"]
    assert "non-empty" in (await err({"type": "speak", "context_id": "e1", "text": ""}, "e1"))["error"]
    assert "4000" in (await err({"type": "speak", "context_id": "e2", "text": "x" * 4001}, "e2"))["error"]
    assert "language" in (await err({"type": "speak", "context_id": "e3", "text": "hi", "language": "xx"}, "e3"))["error"]
    assert "unknown voice" in (await err({"type": "speak", "context_id": "e4", "text": "hi", "voice": "zz"}, "e4"))["error"]
    assert "unknown message type" in (await err({"type": "nope", "context_id": "e5"}, "e5"))["error"]
    await ws.send_bytes(b"\x00\x01")
    ctx = await ws_collect(ws, lambda c: None in c and terminal(c[None]))
    assert ctx[None]["msgs"][-1]["code"] == "invalid_request"
    assert not tts.engine.requests
    # duplicate active context and the per-socket context limit
    await ws.send_json({"type": "speak", "context_id": "k1", "text": "a #frames=1000"})
    await ws.send_json({"type": "speak", "context_id": "k1", "text": "b"})
    ctx = await ws_collect(ws, lambda c: "k1" in c and any(m["type"] == "error" for m in c["k1"]["msgs"]))
    assert "still active" in [m for m in ctx["k1"]["msgs"] if m["type"] == "error"][0]["error"]
    await ws.send_json({"type": "speak", "context_id": "k2", "text": "a #frames=1000"})
    e = await err({"type": "speak", "context_id": "k3", "text": "c"}, "k3")
    assert e["code"] == "too_many_contexts"
    await ws.close()


async def test_ws_overloaded(make):
    client, srv, tts = await make(engine={"max_seats": 1, "frame_s": 0.02}, queue_deadline_ms=150)
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "hog", "text": "x #frames=100000"})
    await ws_collect(ws, lambda c: "hog" in c and c["hog"]["pcm"])
    await ws.send_json({"type": "speak", "context_id": "late", "text": "y"})
    ctx = await ws_collect(ws, lambda c: "late" in c and terminal(c["late"]))
    assert ctx["late"]["msgs"] == [ctx["late"]["msgs"][-1]] and ctx["late"]["msgs"][-1]["code"] == "overloaded"
    assert await wait_until(lambda: tts.engine.requests[1].cancelled)
    # max_pending path
    srv.max_pending = 0
    await ws.send_json({"type": "speak", "context_id": "late2", "text": "y"})
    ctx = await ws_collect(ws, lambda c: "late2" in c and terminal(c["late2"]))
    assert ctx["late2"]["msgs"][-1]["code"] == "overloaded" and len(tts.engine.requests) == 2
    await ws.close()


async def test_ws_disconnect_cancels(make):
    client, srv, tts = await make(engine={"frame_s": 0.01})
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "a", "text": "x #frames=100000"})
    await ws.send_json({"type": "speak", "context_id": "b", "text": "y #frames=100000"})
    await ws_collect(ws, lambda c: all(k in c and c[k]["pcm"] for k in "ab"))
    await ws.close()
    assert await wait_until(lambda: all(r.cancelled and r.done.is_set() for r in tts.engine.requests))
    assert await wait_until(lambda: not srv._jobs and srv.sockets == 0)


async def test_ws_slow_consumer_cancelled(make):
    # Engine much faster than the client reads: with binary frames the socket buffers fill, the bounded send queue
    # fills, generated audio piles up past ws_max_lag_s and the request is cancelled.
    client, srv, tts = await make(engine={"frame_s": 0.0005}, ws_max_lag_s=2.0, ws_send_queue=4, coalesce_ms=80)
    ws = await client.ws_connect("/v1/tts/stream", max_msg_size=0)
    await ws.receive_json()
    await ws.send_json({"type": "speak", "context_id": "s", "text": "x #frames=200000", "binary": True})
    assert await wait_until(lambda: tts.engine.requests and tts.engine.requests[0].cancelled, timeout=20)
    ctx = await ws_collect(ws, lambda c: "s" in c and terminal(c["s"]), timeout=30)
    assert ctx["s"]["msgs"][-1]["code"] == "slow_consumer"
    assert srv.stats["slow_consumer"] == 1
    await ws.close()


# ------------------------------------------------------------------------------------------- voices
async def test_voice_registration_json(make):
    client, srv, tts = await make()
    body = {"voice_id": "asha", "ref_audio_b64": base64.b64encode(wav_bytes(1.5, 16000)).decode(),
            "ref_text": "a reference", "lang": "hi", "mode": "icl_spk"}
    r = await client.post("/v1/voices", json=body)
    assert r.status == 201, await r.text()
    out = await r.json()
    assert out["id"] == "asha" and out["mode"] == "icl_spk" and out["ref_lang"] == "hi"
    reg = tts.registered[-1]
    assert reg["ref_sr"] == 16000 and reg["ref_len"] == 24000 and reg["ref_text"] == "a reference"
    voices = {v["id"]: v for v in (await (await client.get("/v1/voices")).json())["voices"]}
    assert set(voices) == {"plain", "asha"}
    # use it
    r = await speech(client, input="x #frames=3", voice="asha", response_format="pcm")
    assert r.status == 200 and r.headers["X-Maya3-Voice"] == "asha"
    await r.read()
    assert tts.engine.requests[-1].voice.id == "asha"
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "v", "text": "y #frames=2", "voice": "asha"})
    ctx = await ws_collect(ws, lambda c: "v" in c and terminal(c["v"]))
    assert ctx["v"]["msgs"][-1]["type"] == "end" and tts.engine.requests[-1].voice.id == "asha"
    await ws.close()
    # exists -> 409 unless replace
    assert (await client.post("/v1/voices", json=body)).status == 409
    assert (await client.post("/v1/voices", json={**body, "replace": True})).status == 201
    # an OpenAI name can be registered and then wins over the alias
    assert (await client.post("/v1/voices", json={**body, "voice_id": "alloy"})).status == 201
    r = await speech(client, input="z #frames=2", voice="alloy", response_format="pcm")
    assert r.headers["X-Maya3-Voice"] == "alloy"
    await r.read()
    # delete
    assert (await client.delete("/v1/voices/asha")).status == 200
    assert (await client.delete("/v1/voices/asha")).status == 404
    assert (await client.delete("/v1/voices/plain")).status == 400
    r = await speech(client, input="x", voice="asha")
    assert r.status == 400 and "unknown voice" in (await r.json())["error"]["message"]


async def test_voice_registration_multipart(make):
    client, srv, tts = await make()
    form = aiohttp.FormData()
    form.add_field("voice_id", "ravi")
    form.add_field("ref_text", "namaskara")
    form.add_field("lang", "kn")
    form.add_field("ref_audio", wav_bytes(2.0, 24000), filename="ravi.wav", content_type="audio/wav")
    r = await client.post("/v1/voices", data=form)
    assert r.status == 201, await r.text()
    assert tts.registered[-1]["lang"] == "kn" and tts.registered[-1]["ref_len"] == 48000


@pytest.mark.parametrize("patch,frag", [
    ({"voice_id": "bad id!"}, "voice_id"),
    ({"voice_id": "plain"}, "built in"),
    ({"ref_text": None}, "ref_text"),
    ({"lang": "xx"}, "unsupported lang"),
    ({"mode": "weird"}, "unknown mode"),
    ({"ref_audio_b64": "@@@"}, "base64"),
    ({"ref_audio_b64": base64.b64encode(b"not audio").decode()}, "decode"),
    ({"ref_audio_b64": base64.b64encode(wav_bytes(0.2)).decode()}, "at least"),
    ({"ref_audio_b64": base64.b64encode(wav_bytes(31.0, 8000)).decode()}, "30 s"),
    ({"ref_audio_b64": None}, "reference audio"),
])
async def test_voice_registration_errors(make, patch, frag):
    client, srv, tts = await make()
    body = {"voice_id": "v1", "ref_audio_b64": base64.b64encode(wav_bytes()).decode(), "ref_text": "t", "lang": "en"}
    body.update(patch)
    r = await client.post("/v1/voices", json=body)
    assert r.status == 400, await r.text()
    assert frag in (await r.json())["error"]["message"]
    assert not tts.registered


async def test_register_does_not_block_streaming(make):
    client, srv, tts = await make(engine={"frame_s": 0.005})
    tts.register_s = 0.5
    body = {"voice_id": "slow", "ref_audio_b64": base64.b64encode(wav_bytes()).decode(), "ref_text": "t", "lang": "en"}
    reg = asyncio.create_task(client.post("/v1/voices", json=body))
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    r = await speech(client, input="x #frames=5", response_format="pcm")
    await r.read()
    assert time.perf_counter() - t0 < 0.4
    assert (await reg).status == 201


# ------------------------------------------------------------------------------------------- client + bench
async def test_clients(make, tmp_path):
    client, srv, tts = await make()
    base = str(client.make_url("")).rstrip("/")
    ws_url = base.replace("http://", "ws://") + "/v1/tts/stream"
    # async HTTP
    await aspeech("x #frames=4", tmp_path / "a.wav", base_url=base, language="en")
    raw = (tmp_path / "a.wav").read_bytes()
    assert raw[:4] == b"RIFF" and frames_of(raw[44:]) == [1, 2, 3, 4]
    # async WS, two concurrent contexts, base64 and binary
    for binary in (False, True):
        async with Maya3Stream(ws_url, binary=binary) as s:
            async def one(text):
                return b"".join([p async for p in s.speak(text, language="en")])
            a, b = await asyncio.gather(one("a #frames=6"), one("b #frames=9"))
            assert frames_of(a) == list(range(1, 7)) and frames_of(b) == list(range(1, 10))
            assert len(s.last_end) == 2
            with pytest.raises(Maya3Error) as ei:
                async for _ in s.speak("x", language="fr"):
                    pass
            assert ei.value.code == "invalid_request"
            # breaking early cancels on the server
            async for _ in s.speak("long #frames=100000"):
                break
            req = tts.engine.requests[-1]
            assert await wait_until(lambda: req.cancelled)
    # sync client (urllib) in a thread
    c = Maya3Client(base)

    def sync_part():
        pcm = c.speech("x #frames=3", response_format="pcm", language="en")
        assert frames_of(pcm) == [1, 2, 3]
        c.speech("y #frames=2", str(tmp_path / "b.flac"))
        chunks = list(c.iter_speech("z #frames=5"))
        assert frames_of(b"".join(chunks)) == [1, 2, 3, 4, 5]
        c.register_voice("cv", wav_bytes(), ref_text="t", lang="en")
        assert "cv" in [v["id"] for v in c.voices()]
        c.remove_voice("cv")
        try:
            c.speech("x", language="fr")
        except Maya3Error as e:
            assert e.status == 400 and e.code == "invalid_request"
        else:
            raise AssertionError("expected 400")
        return c.health()
    h = await asyncio.to_thread(sync_part)
    assert h["ready"]
    # formats from the file suffix, sampling / normalize as extra fields (HTTP and WS)
    await aspeech("x #frames=5", tmp_path / "c.ogg", base_url=base, seed=3)
    assert tts.engine.requests[-1].sampling == {"seed": 3}
    assert abs(sf.info(str(tmp_path / "c.ogg")).duration - 0.4) < 0.01
    await asyncio.to_thread(c.speech, "x #frames=5", str(tmp_path / "c.mp3"), top_k=5, normalize=False)
    assert tts.engine.requests[-1].sampling == {"top_k": 5}
    assert (tmp_path / "c.mp3").read_bytes()[:1] == b"\xff"
    async with Maya3Stream(ws_url) as s:
        pcm = b"".join([p async for p in s.speak("w #frames=3", temperature=0.5, normalize=False)])
    assert frames_of(pcm) == [1, 2, 3] and tts.engine.requests[-1].sampling == {"temperature": 0.5}
    w, sr = sf.read(str(tmp_path / "b.flac"), dtype="int16")
    assert sr == 24000 and len(w) == 2 * FRAME


async def test_load_ws_bench_smoke(make, tmp_path):
    from maya3.bench import load_ws
    client, srv, tts = await make(engine={"frame_s": 0.01, "max_seats": 8})
    texts = tmp_path / "t.jsonl"
    texts.write_text("\n".join(json.dumps({"id": f"r{i}", "lang": l, "pieces": [f"text {i} #frames={5 + i}"]})
                               for i, l in enumerate(["en", "hi", "te", "en"])) + "\n")
    url = str(client.make_url("/v1/tts/stream")).replace("http://", "ws://")
    for binary in ([], ["--binary"]):
        args = load_ws.build_parser().parse_args(["--url", url, "--texts", str(texts), "--lanes", "3",
                                                  "--duration", "1.0", "--per-request"] + binary)
        s = await load_ws.run(args)
        assert s["completed"] >= 6 and s["errors"] == {}, s["errors"]
        assert s["ttfa_ms"]["p50"] is not None and s["rtf"]["p90"] is not None and s["audio_s_per_s"] > 0
        assert s["stalls"] == 0        # the fake engine runs ~8x faster than real time
        assert set(s["by_lang"]) <= {"en", "hi", "te"} and s["health_before"]["ready"]
        assert all(r["frames"] == round(r["audio_s"] / 0.08) for r in s["per_request"] if r["status"] == "ok")


async def test_playback_stall_accounting():
    from maya3.bench.load_ws import Playback
    pb = Playback(sent=0.0)
    b80 = 2 * FRAME
    pb.add(0.10, b80)            # playback 0.10..0.18
    pb.add(0.15, b80)            # buffered, 0.18..0.26
    pb.add(0.30, b80)            # dry at 0.26: gap 40 ms, no stall
    assert pb.stalls == 0
    pb.add(0.60, b80)            # dry at 0.38: gap 220 ms -> stall
    assert pb.stalls == 1 and abs(pb.stall_s - 0.22) < 1e-9 and abs(pb.audio_s - 0.32) < 1e-9
    pre = Playback(sent=0.0, prebuffer_s=0.3)
    pre.add(0.1, b80)
    pre.add(0.5, b80)            # playback starts 0.4, dry at 0.48 -> 20 ms gap
    assert pre.stalls == 0


async def test_wav_header_and_lang_detection():
    h = S.wav_header(100)
    assert len(h) == 44 and struct.unpack_from("<I", h, 4)[0] == 136 and struct.unpack_from("<I", h, 40)[0] == 100
    assert S.detect_lang("ನಮಸ್ಕಾರ") == "kn" and S.detect_lang("123 ...") == "en"
    assert S.detect_lang("সুপ্ৰভাত") == "bn" and S.detect_lang("ਸਤ ਸ੍ਰੀ") == "pa" and S.detect_lang("હેલો") == "gu"


# ------------------------------------------------------------------------------------------- output formats
def decoded_s(data):
    w, sr = sf.read(io.BytesIO(data), dtype="int16")
    assert sr == 24000
    return len(w) / sr


@pytest.mark.parametrize("fmt,ctype,tol", [("mp3", "audio/mpeg", 0.1), ("opus", "audio/ogg; codecs=opus", 0.01)])
async def test_speech_mp3_opus_streamed(make, fmt, ctype, tol):
    assert S.codec_support()[fmt] is None, S.codec_support()
    client, srv, tts = await make(engine={"frame_s": 0.02})
    t0 = time.perf_counter()
    r = await speech(client, input="x #frames=40", response_format=fmt)     # 3.2 s
    assert r.status == 200 and r.headers["Content-Type"] == ctype and r.headers["Transfer-Encoding"] == "chunked"
    assert r.headers["X-Encoding"] == fmt
    chunks, t_chunks = [], []
    async for b in r.content.iter_any():
        chunks.append(b)
        t_chunks.append(time.perf_counter() - t0)
    t_all = time.perf_counter() - t0
    data = b"".join(chunks)
    assert abs(decoded_s(data) - 40 * 0.08) <= tol, decoded_s(data)
    # encoded audio leaves while the utterance is still being generated (fake engine: 40 frames x 20 ms = 0.8 s)
    n_early = sum(1 for t in t_chunks if t < t_all * 0.75)
    assert n_early >= 2, (t_chunks, t_all)
    if fmt == "mp3":
        assert data[:2] == b"\xff\xf3" or data[:2] == b"\xff\xf2"      # bare MPEG-2 layer III frames
    else:
        assert data[:4] == b"OggS" and b"OpusHead" in data[:64]


@pytest.mark.parametrize("frames", [1, 6, 75])
@pytest.mark.parametrize("fmt", ["mp3", "opus", "flac", "wav"])
async def test_formats_duration(make, fmt, frames):
    client, srv, tts = await make(engine={"frame_s": 0.001})
    r = await speech(client, input=f"x #frames={frames}", response_format=fmt)
    assert r.status == 200
    data = await r.read()
    want = frames * 0.08
    if fmt == "wav":   # streamed header has unknown sizes; check the PCM payload
        assert len(data) - 44 == frames * FRAME * 2
    else:
        assert abs(decoded_s(data) - want) <= (0.1 if fmt == "mp3" else 0.001), (fmt, decoded_s(data), want)


async def test_sse_mp3(make):
    client, srv, tts = await make()
    r = await speech(client, input="x #frames=20", response_format="mp3", stream_format="sse")
    assert r.status == 200
    events = [json.loads(l[6:]) for l in (await r.text()).splitlines() if l.startswith("data: ")]
    assert events[-1]["type"] == "speech.audio.done"
    data = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"] == "speech.audio.delta")
    assert abs(decoded_s(data) - 1.6) <= 0.1


async def test_unavailable_codec_is_400(make, monkeypatch):
    client, srv, tts = await make()
    monkeypatch.setattr(S, "_CODECS", {"mp3": "no MP3 here", "opus": None, "flac": None})
    r = await speech(client, input="x", response_format="mp3")
    assert r.status == 400 and "no MP3 here" in (await r.json())["error"]["message"]
    assert not tts.engine.requests


async def test_mp3_midstream_error_aborts(make):
    client, srv, tts = await make()
    r = await speech(client, input="boom #frames=50 #error=12", response_format="mp3")
    assert r.status == 200
    with pytest.raises(aiohttp.ClientPayloadError):
        await r.read()


async def test_mp3_client_disconnect_cancels(make):
    client, srv, tts = await make(engine={"frame_s": 0.01})
    r = await speech(client, input="forever #frames=100000", response_format="opus")
    await r.content.readany()
    req = tts.engine.requests[-1]
    r.close()
    assert await wait_until(lambda: req.cancelled and req.done.is_set())
    assert await wait_until(lambda: not srv._jobs)


# ------------------------------------------------------------------------------------------- sampling
SAMPLING = {"seed": 1234, "temperature": 0.7, "top_k": 50, "top_p": 0.9}


async def test_sampling_pass_through_http(make):
    client, srv, tts = await make()
    r = await speech(client, input="x #frames=2", response_format="pcm", **SAMPLING)
    assert r.status == 200
    await r.read()
    assert tts.engine.requests[-1].sampling == SAMPLING
    r = await speech(client, input="x #frames=2", response_format="pcm", seed=0, temperature=1)
    await r.read()
    got = tts.engine.requests[-1].sampling
    assert got == {"seed": 0, "temperature": 1.0} and isinstance(got["temperature"], float)
    r = await speech(client, input="x #frames=2", response_format="pcm", top_p=None)     # null = absent
    await r.read()
    assert tts.engine.requests[-1].sampling is None                     # nothing given -> no sampling kwarg


async def test_sampling_pass_through_ws(make):
    client, srv, tts = await make()
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "s", "text": "y #frames=2", **SAMPLING})
    await ws.send_json({"type": "speak", "context_id": "t", "text": "z #frames=2", "top_k": 1})
    ctx = await ws_collect(ws, lambda c: all(k in c and terminal(c[k]) for k in "st"))
    assert ctx["s"]["msgs"][-1]["type"] == "end"
    by_text = {r.text.split()[0]: r.sampling for r in tts.engine.requests}
    assert by_text == {"y": SAMPLING, "z": {"top_k": 1}}
    await ws.send_json({"type": "speak", "context_id": "bad", "text": "z", "temperature": 3})
    ctx = await ws_collect(ws, lambda c: "bad" in c and terminal(c["bad"]))
    e = ctx["bad"]["msgs"][-1]
    assert e["type"] == "error" and e["code"] == "invalid_request" and "temperature" in e["error"]
    assert len(tts.engine.requests) == 2
    await ws.close()


@pytest.mark.parametrize("patch,param", [
    ({"seed": -1}, "seed"), ({"seed": "7"}, "seed"), ({"seed": True}, "seed"), ({"seed": 1.5}, "seed"),
    ({"seed": 2 ** 63}, "seed"),
    ({"temperature": 0.05}, "temperature"), ({"temperature": 1.6}, "temperature"), ({"temperature": "hot"}, "temperature"),
    ({"temperature": False}, "temperature"),
    ({"top_k": 0}, "top_k"), ({"top_k": 2049}, "top_k"), ({"top_k": 5.0}, "top_k"),
    ({"top_p": 0}, "top_p"), ({"top_p": 1.01}, "top_p"), ({"top_p": -0.5}, "top_p"),
    ({"repetition_penalty": 6}, "repetition_penalty"),
    ({"normalize": "no"}, "normalize"),
])
async def test_sampling_validation(make, patch, param):
    client, srv, tts = await make()
    r = await speech(client, **patch)
    assert r.status == 400
    err = (await r.json())["error"]
    assert err["param"] == param and param in err["message"]
    assert not tts.engine.requests


async def test_sampling_nan_rejected(make):
    client, srv, tts = await make()
    r = await client.post("/v1/audio/speech", data=b'{"input": "hi", "top_p": NaN}',
                          headers={"Content-Type": "application/json"})
    assert r.status == 400 and (await r.json())["error"]["param"] == "top_p"


async def test_sampling_bounds_accepted(make):
    client, srv, tts = await make()
    for sp in ({"temperature": 0.1, "top_k": 1, "top_p": 1}, {"temperature": 1.5, "top_k": 2048, "top_p": 1e-6},
               {"seed": 2 ** 63 - 1}):
        r = await speech(client, input="x #frames=1", response_format="pcm", **sp)
        assert r.status == 200, await r.text()
        await r.read()
        assert tts.engine.requests[-1].sampling == {k: (float(v) if k in ("temperature", "top_p") else v)
                                                    for k, v in sp.items()}


# ------------------------------------------------------------------------------------------- normaliser
async def test_normaliser_default_on_http(make):
    assert S.Maya3Server(None).normalise is True                    # on by default
    client, srv, tts = await make(normalise=True)
    r = await speech(client, input="I have 25 apples [laughs]", language="en", response_format="pcm")
    assert r.status == 200 and r.headers["X-Maya3-Normalized"] == "1"
    await r.read()
    assert tts.engine.requests[-1].text == "I have twenty five apples [laughs]"
    r = await speech(client, input="Your OTP is 482913.", language="en", response_format="pcm")
    await r.read()
    assert tts.engine.requests[-1].text == "Your OTP is four, eight, two, nine, one, three."
    r = await speech(client, input="मेरे पास 25 रुपये हैं", response_format="pcm")      # language detected from raw
    assert r.headers["X-Maya3-Language"] == "hi"
    await r.read()
    assert tts.engine.requests[-1].text == "मेरे पास पच्चीस रुपये हैं"
    r = await speech(client, input="నా దగ్గర 25 రూపాయలు ఉన్నాయి", language="te", response_format="pcm")
    await r.read()
    assert tts.engine.requests[-1].text == "నా దగ్గర ఇరవై ఐదు రూపాయలు ఉన్నాయి"
    r = await speech(client, input="Visit https://www.example.com/abc today", language="en", response_format="pcm")
    await r.read()
    assert tts.engine.requests[-1].text == "Visit W W W dot example dot com slash abc today"
    # per request off
    r = await speech(client, input="I have 25 apples", language="en", response_format="pcm", normalize=False)
    assert r.headers["X-Maya3-Normalized"] == "0"
    await r.read()
    assert tts.engine.requests[-1].text == "I have 25 apples"
    assert srv.stats["normalised"] == 5
    h = await (await client.get("/health")).json()
    assert h["server"]["normalise"] is True and {"wav", "pcm", "mp3", "opus", "flac"} <= set(h["server"]["formats"])


async def test_normaliser_long_input_off_loop(make):
    client, srv, tts = await make(normalise=True)
    text = "Pay 25 rupees now. " * 30                     # > NORMALISE_INLINE_CHARS: normalised on a worker thread
    r = await speech(client, input=text, language="en", response_format="pcm", max_frames=3)
    assert r.status == 200
    await r.read()
    assert tts.engine.requests[-1].text == ("Pay twenty five rupees now. " * 30).strip()


async def test_normaliser_globally_off(make):
    client, srv, tts = await make(normalise=False)
    r = await speech(client, input="I have 25 apples", language="en", response_format="pcm", normalize=True)
    assert r.headers["X-Maya3-Normalized"] == "0"
    await r.read()
    assert tts.engine.requests[-1].text == "I have 25 apples"


async def test_normaliser_ws(make):
    client, srv, tts = await make(normalise=True)
    ws = await ws_open(client)
    long = "Pay 25 rupees now. " * 30
    await ws.send_json({"type": "speak", "context_id": "a", "text": "I have 3 cats [laughs]", "language": "en"})
    await ws.send_json({"type": "speak", "context_id": "b", "text": "I have 3 cats", "normalize": False})
    await ws.send_json({"type": "speak", "context_id": "c", "text": long, "language": "en", "max_frames": 2})
    ctx = await ws_collect(ws, lambda c: all(k in c and terminal(c[k]) for k in "abc"))
    assert all(ctx[k]["msgs"][-1]["type"] == "end" for k in "abc")
    texts = sorted(r.text for r in tts.engine.requests)
    assert texts == sorted(["I have three cats [laughs]", "I have 3 cats", ("Pay twenty five rupees now. " * 30).strip()])
    # cancelling a long context while it is being normalised still ends it exactly once
    await ws.send_json({"type": "speak", "context_id": "d", "text": long, "language": "en"})
    await ws.send_json({"type": "cancel", "context_id": "d"})
    ctx = await ws_collect(ws, lambda c: "d" in c and terminal(c["d"]))
    assert ctx["d"]["msgs"][-1]["type"] == "end" and ctx["d"]["msgs"][-1]["reason"] == "cancelled"
    await ws.send_json({"type": "ping"})
    ctx = await ws_collect(ws, lambda c: None in c and any(m["type"] == "pong" for m in c[None]["msgs"]))
    assert "d" not in ctx
    await ws.close()
    assert await wait_until(lambda: not srv._jobs)


async def test_named_voice_text_is_normalised_without_prefix(make):
    client, srv, tts = await make(normalise=True)
    tts.engine.registry.add(named_voice("Asha", gender="female"))
    r = await speech(client, input="I have 25 apples", voice="asha", language="en", response_format="pcm")
    assert r.status == 200 and r.headers["X-Maya3-Voice"] == "Asha"
    await r.read()
    req = tts.engine.requests[-1]
    assert req.voice.id == "Asha" and req.text == "I have twenty five apples"     # the engine adds "{Asha}: "


# ------------------------------------------------------------------------------------------- voices (named)
async def test_voices_list_named(make):
    client, srv, tts = await make()
    meta = {"display_name": "Asha", "gender": "female", "age": "adult", "style": "warm", "langs": ["hi", "en"]}
    reg = tts.engine.registry
    reg.add(named_voice("Asha", **meta))
    reg.add(named_voice("Nova", gender="female"))                       # shadows the OpenAI alias "nova"
    reg.add(named_voice("Ravi", mode="icl_spk"))                        # name template over a cloned reference
    reg.add(FakeVoice("bob", "icl_spk", "en"))                          # a clone: case-sensitive id
    voices = {v["id"]: v for v in (await (await client.get("/v1/voices")).json())["voices"]}
    assert voices["plain"] == {"id": "plain", "mode": "plain", "ref_lang": None}
    assert voices["bob"] == {"id": "bob", "mode": "icl_spk", "ref_lang": "en"}
    assert voices["Asha"] == {"id": "Asha", "mode": "named", "ref_lang": None, "meta": {"name": "Asha", **meta}}
    assert voices["Ravi"]["mode"] == "named" and voices["Ravi"]["base_mode"] == "icl_spk"
    for asked, want in (("asha", "Asha"), ("ASHA", "Asha"), ("Asha", "Asha"), ("nova", "Nova"), ("alloy", "plain")):
        r = await speech(client, input="x #frames=1", voice=asked, response_format="pcm")
        assert r.status == 200 and r.headers["X-Maya3-Voice"] == want, (asked, await r.text())
        await r.read()
        assert tts.engine.requests[-1].voice.id == want
    r = await speech(client, input="x", voice="BOB")
    assert r.status == 400 and "unknown voice" in (await r.json())["error"]["message"]
    ws = await ws_open(client)
    await ws.send_json({"type": "speak", "context_id": "n", "text": "y #frames=1", "voice": "aSHA"})
    ctx = await ws_collect(ws, lambda c: "n" in c and terminal(c["n"]))
    assert ctx["n"]["msgs"][-1]["type"] == "end" and tts.engine.requests[-1].voice.id == "Asha"
    await ws.close()


async def test_voices_file_meta(tmp_path):
    tts = FakeTTS()
    wav = tmp_path / "a.wav"
    wav.write_bytes(wav_bytes(1.0))
    spec = tmp_path / "voices.json"
    spec.write_text(json.dumps({"voices": [{"voice_id": "asha", "ref_audio": "a.wav", "ref_text": "t", "lang": "hi",
                                            "mode": "plain", "meta": {"gender": "female"}}]}))
    tts.register_voice = lambda vid, **kw: tts.registered.append((vid, kw))
    S.load_voices_file(tts, spec)
    vid, kw = tts.registered[0]
    assert vid == "asha" and kw["meta"] == {"gender": "female"} and kw["ref_audio"] == str(wav)
    tts.close()
