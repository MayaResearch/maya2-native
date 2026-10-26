#!/usr/bin/env python3
"""Command-line synthesis with Maya3.

    python examples/synthesize.py "नमस्ते, आप कैसे हैं?" --lang hi --voice Ahana -o out.wav
    python examples/synthesize.py "Hello!" --lang en --ref ref.wav --ref-text "<exact transcript>" --ref-lang hi -o clone.wav
    python examples/synthesize.py --list-voices

--model is a Hugging Face repo id or a local release directory (default: $MAYA3_MODEL).
"""
import argparse
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maya3 import Maya3TTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--model", default=os.environ.get("MAYA3_MODEL"), help="HF repo id or local release dir")
    ap.add_argument("--lang", default="en", help="as bn en gu hi kn ml mr or pa ta te")
    ap.add_argument("--voice", default=None, help="named voice (see --list-voices) or 'plain'")
    ap.add_argument("--ref", default=None, help="reference wav for zero-shot cloning (3-10 s)")
    ap.add_argument("--ref-text", default=None, help="exact transcript of --ref")
    ap.add_argument("--ref-lang", default=None, help="language of --ref (default: --lang)")
    ap.add_argument("--no-normalize", action="store_true", help="skip the bundled text normaliser")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model (or $MAYA3_MODEL) is required")
    tts = Maya3TTS.from_pretrained(a.model, max_seats=1)
    if a.list_voices:
        print("\n".join(sorted(tts.voices)))
        return
    if not a.text:
        ap.error("text is required")
    t = time.time()
    wav = tts.generate(a.text, lang=a.lang, voice=a.voice, ref_audio=a.ref, ref_text=a.ref_text,
                       ref_lang=a.ref_lang or a.lang if a.ref else None, normalize=not a.no_normalize, seed=a.seed)
    sf.write(a.out, wav, 24000)
    print(f"wrote {a.out}: {len(wav) / 24000:.2f} s of audio in {time.time() - t:.2f} s")


if __name__ == "__main__":
    main()
