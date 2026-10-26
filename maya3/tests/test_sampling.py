"""GPU: per-request sampling. Same seed alone -> identical audio; different seed -> different audio; temperature / top_k /
top_p / repetition_penalty accepted and effective; invalid values rejected. Usage: test_sampling.py <model.pt>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maya3 import Maya3TTS
from maya3.bench import common as C

tts = Maya3TTS.from_pretrained(sys.argv[1], C.BACKBONE, C.MIMI, ecapa_dir=C.ECAPA, max_seats=4)
T = "The quick brown fox jumps over the lazy dog, and then it runs away."
a = tts.generate(T, "en", seed=123)
b = tts.generate(T, "en", seed=123)
c = tts.generate(T, "en", seed=124)
print("same seed identical:", len(a) == len(b) and np.array_equal(a, b), "| other seed differs:", not (len(a) == len(c) and np.array_equal(a, c)))
assert len(a) == len(b) and np.array_equal(a, b)
assert not (len(a) == len(c) and np.array_equal(a, c))
for kw in ({"temperature": 0.5}, {"top_k": 8}, {"top_p": 0.8}, {"repetition_penalty": 0.0}, {"temperature": 1.2, "seed": 7}):
    w, info = tts.generate(T, "en", return_info=True, **kw)
    print(kw, round(len(w) / 24000, 2), "s", info["stops"][-1]["reason"])
    assert len(w) > 24000
g1 = tts.generate(T, "en", seed=5, top_k=1)      # top_k=1 on cb0: deterministic cb0 path
g2 = tts.generate(T, "en", seed=6, top_k=1)
print("top_k=1 different seeds same length:", len(g1) == len(g2))
for bad in ({"temperature": 0}, {"top_k": 0}, {"top_p": 1.5}, {"foo": 1}):
    try:
        tts.generate("x", "en", **bad) if "foo" not in bad else tts.engine.submit("x", "en", sampling=bad)
        raise SystemExit(f"accepted invalid {bad}")
    except (ValueError, TypeError):
        pass
print("SAMPLING_OK")
tts.close()
