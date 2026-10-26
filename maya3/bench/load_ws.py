"""Closed-loop WebSocket load client for maya3/server.py.

    python -m maya3.bench.load_ws --url ws://127.0.0.1:8080/v1/tts/stream --texts sets/infer300.jsonl \\
        --lanes 32 --duration 180 --ramp-s 5 --out /tmp/load_32.json

Each lane holds one WebSocket and keeps exactly one utterance in flight (send speak -> wait for end/error -> next).
Texts come from a jsonl (field `pieces[0]`, else `text`; language from `lang`), shuffled with --seed.
Per request: TTFA (speak sent -> first audio received), playback stalls (playback starts when the first audio
arrives (+ --prebuffer-ms); a chunk arriving more than 100 ms after the playback buffer ran dry is a stall),
RTF = (end received - sent) / audio seconds, audio_s, server-reported ttfa/reason.
Requests are started only inside the --duration window; in-flight ones are drained and counted.
audio_s_per_s counts audio received inside the window.  Output: JSON summary (p50/p90/p99) to stdout / --out.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import random
import struct
import sys
import time
from pathlib import Path

SR = 24000
STALL_S = 0.100


def pct(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}

    def at(p):
        i = (len(v) - 1) * p
        lo = int(i)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)
    return {"n": len(v), "p50": round(at(.5), 4), "p90": round(at(.9), 4), "p99": round(at(.99), 4),
            "max": round(v[-1], 4), "mean": round(sum(v) / len(v), 4)}


class Playback:
    """Real-time playback simulation of one utterance."""

    def __init__(self, sent, prebuffer_s=0.0):
        self.sent, self.prebuffer = sent, prebuffer_s
        self.first = None
        self.dry = None          # time the playback buffer runs dry
        self.audio_s = 0.0
        self.stalls = 0
        self.stall_s = 0.0
        self.max_gap_s = 0.0
        self.arrivals = []       # (t_rel_to_run_start, audio_s) for the audio-in-window metric

    def add(self, t, nbytes):
        dur = nbytes / (2 * SR)
        if self.first is None:
            self.first = t
            self.dry = t + self.prebuffer
        gap = t - self.dry
        if gap > 0:
            self.max_gap_s = max(self.max_gap_s, gap)
            if gap > STALL_S:
                self.stalls += 1
                self.stall_s += gap
        self.dry = max(t, self.dry) + dur
        self.audio_s += dur
        return dur


def load_texts(path, langs=None, field="pieces"):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lang = r.get("lang")
        if langs and lang not in langs:
            continue
        t = r.get(field)
        if isinstance(t, list):
            t = t[0] if t else None
        t = t or r.get("text")
        if isinstance(t, str) and t.strip():
            rows.append({"id": r.get("id"), "lang": lang, "text": t.strip()})
    if not rows:
        raise SystemExit(f"no usable rows in {path} (langs={langs})")
    return rows


def parse_binary(data):
    (n,) = struct.unpack_from("<I", data)
    return data[4:4 + n].decode(), len(data) - 4 - n


async def one_request(ws, row, args, lane, seq, run_start):
    import aiohttp
    cid = f"l{lane}-{seq}"
    msg = {"type": "speak", "context_id": cid, "text": row["text"], "language": row["lang"], "binary": args.binary}
    if args.voice:
        msg["voice"] = args.voice
    if args.max_frames:
        msg["max_frames"] = args.max_frames
    sent = time.perf_counter()
    pb = Playback(sent, args.prebuffer_ms / 1000)
    rec = {"lane": lane, "seq": seq, "id": row["id"], "lang": row["lang"], "chars": len(row["text"]),
           "sent_s": round(sent - run_start, 4), "status": "error", "code": None}
    try:
        await ws.send_json(msg)
        async with asyncio.timeout(args.timeout):
            while True:
                m = await ws.receive()
                t = time.perf_counter()
                if m.type == aiohttp.WSMsgType.BINARY:
                    c, n = parse_binary(m.data)
                    if c != cid:
                        raise RuntimeError(f"audio for unexpected context {c}")
                    pb.arrivals.append((t - run_start, pb.add(t, n)))
                    continue
                if m.type != aiohttp.WSMsgType.TEXT:
                    raise ConnectionError(f"socket closed ({m.type.name})")
                f = json.loads(m.data)
                if f.get("context_id") not in (cid, None):
                    raise RuntimeError(f"frame for unexpected context {f.get('context_id')}")
                kind = f.get("type")
                if kind == "audio":
                    n = len(base64.b64decode(f["audio"]))
                    pb.arrivals.append((t - run_start, pb.add(t, n)))
                elif kind == "end":
                    rec.update(status="ok" if pb.audio_s > 0 else "error", code=None if pb.audio_s > 0 else "no_audio",
                               reason=f.get("reason"), server_ttfa_ms=f.get("ttfa_ms"), frames=f.get("frames"))
                    break
                elif kind == "error":
                    rec.update(code=f.get("code"), error=f.get("error"))
                    break
                elif kind == "pong":
                    continue
    except TimeoutError:
        rec.update(code="client_timeout")
        try:
            await ws.send_json({"type": "cancel", "context_id": cid})
        except Exception:
            pass
    except (ConnectionError, RuntimeError) as exc:
        rec.update(code="connection", error=str(exc))
    end = time.perf_counter()
    rec.update(ttfa_ms=round((pb.first - sent) * 1000, 2) if pb.first else None,
               e2e_ms=round((end - sent) * 1000, 2), audio_s=round(pb.audio_s, 4),
               rtf=round((end - sent) / pb.audio_s, 4) if pb.audio_s else None,
               stalls=pb.stalls, stall_ms=round(pb.stall_s * 1000, 2), max_gap_ms=round(pb.max_gap_s * 1000, 2))
    return rec, pb.arrivals


async def lane_loop(session, lane, args, rows, run_start, stop_at, out, arrivals):
    if args.ramp_s > 0 and args.lanes > 1:
        await asyncio.sleep(args.ramp_s * lane / args.lanes)
    rng = random.Random(args.seed * 1000 + lane)
    order = list(range(len(rows)))
    rng.shuffle(order)
    seq = 0
    ws = None
    while time.perf_counter() < stop_at:
        if ws is None or ws.closed:
            try:
                ws = await session.ws_connect(args.url, max_msg_size=64 * 1024 * 1024, heartbeat=None)
                meta = await ws.receive_json(timeout=30)
                if meta.get("type") != "metadata" or meta.get("sample_rate") != SR:
                    raise RuntimeError(f"bad metadata {meta}")
            except Exception as exc:
                out.append({"lane": lane, "seq": seq, "status": "error", "code": "connect", "error": str(exc),
                            "sent_s": round(time.perf_counter() - run_start, 4)})
                ws = None
                await asyncio.sleep(1.0)
                continue
        row = rows[order[seq % len(order)]]
        rec, arr = await one_request(ws, row, args, lane, seq, run_start)
        out.append(rec)
        arrivals.extend(arr)
        seq += 1
        if rec["status"] != "ok":
            if rec["code"] in ("connection", "client_timeout"):
                await ws.close()
                ws = None
            await asyncio.sleep(args.error_backoff_ms / 1000)
    if ws is not None:
        await ws.close()


async def run(args):
    import aiohttp
    rows = load_texts(args.texts, set(args.langs.split(",")) if args.langs else None)
    http = args.url.replace("ws://", "http://").replace("wss://", "https://").split("/v1/")[0]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        async def health():
            try:
                async with session.get(http + "/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    return await r.json()
            except Exception as exc:
                return {"error": str(exc)}
        h0 = await health()
        out, arrivals = [], []
        run_start = time.perf_counter()
        stop_at = run_start + args.duration
        await asyncio.gather(*(lane_loop(session, i, args, rows, run_start, stop_at, out, arrivals)
                               for i in range(args.lanes)))
        elapsed = time.perf_counter() - run_start
        h1 = await health()
    window = min(args.duration, elapsed)
    ok = [r for r in out if r["status"] == "ok"]
    errors = collections.Counter(r["code"] for r in out if r["status"] != "ok")
    audio_win = sum(a for t, a in arrivals if t <= window)
    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "duration_s": round(window, 3), "elapsed_with_drain_s": round(elapsed, 3),
        "requests": len(out), "completed": len(ok), "errors": dict(errors),
        "error_pct": round(100 * (len(out) - len(ok)) / max(1, len(out)), 3),
        "req_per_s": round(len(ok) / window, 4) if window else None,
        "audio_s_per_s": round(audio_win / window, 4) if window else None,
        "audio_s_per_s_with_drain": round(sum(r.get("audio_s") or 0 for r in out) / elapsed, 4) if elapsed else None,
        "ttfa_ms": pct(r["ttfa_ms"] for r in ok),
        "server_ttfa_ms": pct(r.get("server_ttfa_ms") for r in ok),
        "rtf": pct(r["rtf"] for r in ok),
        "e2e_ms": pct(r["e2e_ms"] for r in ok),
        "audio_s": pct(r["audio_s"] for r in ok),
        "stalls": sum(r.get("stalls", 0) for r in out),
        "stalled_request_pct": round(100 * sum(r.get("stalls", 0) > 0 for r in ok) / max(1, len(ok)), 3),
        "stall_ms_per_audio_min": round(60 * sum(r.get("stall_ms", 0) for r in ok) / max(1e-9, sum(r["audio_s"] for r in ok)), 3)
        if ok else None,
        "max_gap_ms": pct(r.get("max_gap_ms") for r in ok),
        "reasons": dict(collections.Counter(r.get("reason") for r in ok)),
        "by_lang": {lang: {"n": len(rs), "ttfa_ms_p90": pct(r["ttfa_ms"] for r in rs)["p90"],
                           "rtf_p90": pct(r["rtf"] for r in rs)["p90"], "stalls": sum(r["stalls"] for r in rs)}
                    for lang, rs in sorted(_group(ok, "lang").items())},
        "health_before": h0, "health_after": h1,
    }
    if args.per_request:
        summary["per_request"] = out
    return summary


def _group(rows, key):
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return g


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="ws://127.0.0.1:8080/v1/tts/stream")
    ap.add_argument("--texts", required=True, type=Path, help="jsonl with pieces[0] (or text) and lang")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds during which new requests are started")
    ap.add_argument("--ramp-s", type=float, default=0.0, help="stagger lane starts over this many seconds")
    ap.add_argument("--langs", default=None, help="comma list filter, e.g. hi,en,te")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--binary", action="store_true", help="binary audio frames instead of base64 JSON")
    ap.add_argument("--prebuffer-ms", type=float, default=0.0, help="client playback prebuffer before starting playback")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout (s)")
    ap.add_argument("--error-backoff-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-request", action="store_true", help="include every request record in the JSON")
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = asyncio.run(run(args))
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    brief = {k: summary[k] for k in ("requests", "completed", "errors", "audio_s_per_s", "stalls", "stalled_request_pct")}
    brief.update(ttfa_ms=summary["ttfa_ms"], rtf=summary["rtf"])
    print(json.dumps(brief, indent=1) if args.out else text, file=sys.stdout)


if __name__ == "__main__":
    main()
