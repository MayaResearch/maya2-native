#!/usr/bin/env python3
"""Stream from the WebSocket endpoint and report time to first audio. Two contexts run concurrently on one socket.

    python -m maya3.server --ckpt <repo or dir> --port 8080 &
    python examples/ws_stream.py --url ws://127.0.0.1:8080/v1/tts/stream
"""
import argparse
import asyncio
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maya3.client import Maya3Stream  # noqa: E402


async def speak(s, text, lang, voice, out):
    t0, first, buf = time.perf_counter(), None, bytearray()
    async for pcm in s.speak(text, language=lang, voice=voice):
        first = first or time.perf_counter()
        buf += pcm
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(bytes(buf))
    print(f"{out}: TTFA {1e3 * (first - t0):.0f} ms, {len(buf) / 48000:.2f} s audio")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080/v1/tts/stream")
    a = ap.parse_args()
    async with Maya3Stream(a.url, binary=True) as s:
        await asyncio.gather(
            speak(s, "आपका ऑर्डर आज शाम तक पहुँच जाएगा।", "hi", "Ahana", "ws_hi.wav"),
            speak(s, "Your appointment is confirmed for 10:30 AM tomorrow.", "en", "Siddhant", "ws_en.wav"))


if __name__ == "__main__":
    asyncio.run(main())
