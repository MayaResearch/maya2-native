"""CPU: vectorised loop guard == a scalar reference loop_detect (_ref_loop_detect below) on random / periodic / stalled cb0 streams."""
import os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from maya3 import guards as G


def _ref_loop_detect(c0, min_frames=12, stall=32, period_win=64, period_thr=0.7, pmin=2, pmax=16):
    n = len(c0)
    if n < max(min_frames, stall):
        return None
    tail = np.asarray(c0[-stall:])
    if (tail == tail[0]).all():
        return "stall"
    if n >= period_win:
        w = np.asarray(c0[-period_win:])
        for p in range(pmin, pmax + 1):
            if (w[p:] == w[:-p]).mean() >= period_thr:
                return f"period:{p}"
    return None


def test_loop_detect_batch_matches_reference():
    rng = np.random.default_rng(0)
    cfg = G.GuardConfig()
    Ws, ns, refs = [], [], []
    for t in range(3000):
        n = int(rng.integers(30, 200)); kind = t % 3
        if kind == 0:
            c = rng.integers(0, 50, n)
        elif kind == 1:
            p = int(rng.integers(2, 17)); base = rng.integers(0, 30, p)
            c = np.array([base[i % p] for i in range(n)]); c[rng.random(n) < rng.random() * 0.5] = rng.integers(0, 30)
        else:
            c = rng.integers(0, 3, n); c[-int(rng.integers(20, 40)):] = 7
        refs.append(_ref_loop_detect(c[-96:]))
        W = np.full(96, -1); tail = c[-96:]; W[-len(tail):] = tail
        Ws.append(W); ns.append(n)
    got = G.loop_detect_batch(np.stack(Ws), np.array(ns), cfg)
    assert got == refs


def test_duration_prior_and_tags():
    dp = G.DurationPrior()
    assert dp.cap_frames("Hello there, how are you?", "en") > 20
    w = G.tag_windows("Hello [laughs] there", "en", dp)
    assert len(w) == 1 and w[0][0] <= w[0][1]


if __name__ == "__main__":
    test_loop_detect_batch_matches_reference(); test_duration_prior_and_tags(); print("test_guards OK")


# ---- loop guard v2 windows: fixtures = the actual loop_backstop cuts of an earlier checkpoint on normalised eval text
def _cuts():
    import json
    return json.load(open(os.path.join(os.path.dirname(__file__), "data_backstop_cuts.json")))


def test_v2_windows_cover_tag_filler_speaker_cuts():
    P = G.DurationPrior()
    cuts = _cuts()
    cov = lambda c, **kw: any(a <= c["cut"] - 24 <= b for a, b in G.suspend_windows_v2(c["text"], c["lang"], P, **kw))
    for c in cuts:            # v1 tag windows covered none of the real cuts (loop onset = cut - 24-frame backstop)
        assert not any(a <= c["cut"] - 24 <= b for a, b in G.tag_windows(c["text"], c["lang"], P)), c["id"]
    tag = [c for c in cuts if c["kind"] == "tag"]
    sp = [c for c in cuts if c["kind"] == "speaker"]
    fill = [c for c in cuts if c["kind"] == "filler"]
    assert len(tag) == 25 and all(cov(c, style_hold_s=None) for c in tag)      # first v2 (style tags to end): all
    assert sum(cov(c) for c in tag) >= 12                                         # v2b default: bounded style windows
    assert all(cov(c) for c in sp) and len(sp) == 3
    assert sum(cov(c) for c in fill) >= 3


def test_v2_keeps_guard_on_unmarked_text():
    P = G.DurationPrior()
    for c in _cuts():
        if c["kind"] == "none":   # incl. kn-0147, a real loop (ASR heard 217 words for a 37-word text): must stay guarded
            assert G.suspend_windows_v2(c["text"], c["lang"], P) == (), c["id"]


def test_v2_window_shapes():
    P = G.DurationPrior()
    w = G.suspend_windows_v2("[whispers] arre sun, maine tujhe paise bhej diye hain", "hi", P)
    assert w[0][0] == 0 and w[0][1] == int(4.0 / G.FRAME_S)       # style tag at t=0: bounded window (v2b default)
    w = G.suspend_windows_v2("[whispers] arre sun, maine tujhe paise bhej diye hain", "hi", P, style_hold_s=None)
    assert w[0][1] >= 10 ** 6                                    # first v2: to the end of the piece
    w = G.suspend_windows_v2("I checked it, [laughs] and it is fine now.", "en", P)
    assert len(w) == 1 and w[0][1] < 10 ** 6
    w = G.suspend_windows_v2("ನಮಸ್ಕಾರ, ನಾನು ರಮೇಶ್. ಹ್ಮ್, ನಿಮ್ಮ ಕಾರ್ಡ್ ನಂಬರ್", "kn", P)
    assert len(w) == 1                                           # filler
    w = G.suspend_windows_v2("Rohan: are you coming? Priya: yes, I am.", "en", P)
    assert len(w) == 1                                           # speaker switch (the leading name is not a switch)
    assert G.GuardConfig().aware_v2 is (os.environ.get("MAYA3_GUARD_V2", "1") == "1")   # default ON (v2b, validated 19:12Z)


def test_v2_identical_to_eval_guards():
    ev = os.environ.get("MAYA3_REF_GUARDS", "")
    if not ev or not os.path.exists(ev):
        pytest.skip("eval guards not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_eval_guards_v2", ev)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    Pm, Pe = G.DurationPrior(), E.DurationPrior()
    for c in _cuts():
        assert G.suspend_windows_v2(c["text"], c["lang"], Pm) == E.suspend_windows_v2(c["text"], c["lang"], Pe), c["id"]
