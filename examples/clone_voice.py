#!/usr/bin/env python3
"""Zero-shot voice cloning, in Python and over HTTP.

Only clone voices you own or have the speaker's documented consent to use.

    python examples/clone_voice.py --model <repo or dir> --ref ref.wav --ref-text "<exact transcript>" --ref-lang hi
    python examples/clone_voice.py --server http://127.0.0.1:8080 --ref ref.wav --ref-text "..." --ref-lang hi
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXTS = [("hi", "नमस्ते, यह मेरी आवाज़ की एक प्रति है।"), ("en", "Hello, this is a cloned voice."),
         ("ta", "வணக்கம், இது ஒரு குளோன் செய்யப்பட்ட குரல்.")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MAYA3_MODEL"))
    ap.add_argument("--server", default=None, help="use a running maya3.server instead of loading the model")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--ref-lang", required=True)
    ap.add_argument("--mode", default="icl_spk", choices=["icl_spk", "icl", "spk"])
    ap.add_argument("--out-dir", default="clone_out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.server:
        from maya3.client import Maya3Client
        c = Maya3Client(a.server)
        c.register_voice("my_voice", a.ref, ref_text=a.ref_text, lang=a.ref_lang, mode=a.mode, replace=True)
        for lang, text in TEXTS:
            c.speech(text, f"{a.out_dir}/{lang}.wav", language=lang, voice="my_voice")
        c.remove_voice("my_voice")
    else:
        import soundfile as sf
        from maya3 import Maya3TTS
        tts = Maya3TTS.from_pretrained(a.model, max_seats=4)
        tts.register_voice("my_voice", ref_audio=a.ref, ref_text=a.ref_text, lang=a.ref_lang, mode=a.mode)
        for lang, text in TEXTS:
            sf.write(f"{a.out_dir}/{lang}.wav", tts.generate(text, lang=lang, voice="my_voice"), 24000)
    print(f"wrote {len(TEXTS)} clips to {a.out_dir}/")


if __name__ == "__main__":
    main()
