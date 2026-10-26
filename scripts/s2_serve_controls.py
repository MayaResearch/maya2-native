"""Serving controls: the decode-side fixes for stop behaviour, long-form drift and output level.

Everything here is decode-side; the checkpoint is untouched. Three controls, each switchable so
ablations can be run with and without each one:

  --stop {sampled,controlled}
      sampled     = the trained behaviour: <aeos> is one class in the multinomial cb0 draw (a
                    well-fit model spreads the stop hazard over trailing-silence frames, so a
                    non-aeos draw keeps talking = babble tails).
      controlled  = <aeos> is masked out of the sampled draw; the stop fires when, after a
                    text-conditioned minimum length, argmax == <aeos>, or p(<aeos>) >= --tau, or the
                    cumulative stop hazard 1 - prod(1 - p_t) >= --hazard. The cap is the same
                    per-language text -> frames prior at its p99 residual instead of a flat 1500.
  --chunk-s S
      chunked generation with RE-PROMPTING (against long-rollout drift): the text is split at sentence terminals into pieces
      of about S seconds (prior-estimated). Chunk 1 uses the requested arm. Chunk k > 1 is generated
      in a FRESH backbone context whose in-context reference is the previous chunk's text plus the
      last --ref-tail-frames of its generated codes (the same speaker memory throughout). Drift
      accumulated over long rollouts resets at each chunk boundary. 0 disables.
  --level {none,peaknorm,limiter}
      post-decode level control (against loudness ramps): peaknorm = the corpus's own consumer rule (attenuate so the
      peak sits at --ceiling-db, never boost); limiter = look-ahead peak limiter at --ceiling-db
      (5 ms look-ahead, 80 ms release) so a loudness ramp cannot flat-top.

Rails are computed on the RAW decode (so the ramp is visible) and on the processed output: clip rate
(|x| >= 0.999), flat-top runs (>= 3 consecutive pinned samples), pinned fraction, RMS dBFS, per-second
RMS slope (dB/s) and end-minus-start (last 3 s vs first 3 s), HF (> 6 kHz) energy fraction, silent
(>= 3 s of 1-s windows under RMS 0.005), frames / cap / aeos, duration vs the prior's expectation.

Targets: by default a LONG text per row is built by concatenating the target texts of other eval-set
rows of the same language until --long-chars characters (the corpus targets are ~10 s; drift and
loudness ramps need 25-40 s to show). --targets orig uses each row's own text instead.

  python scripts/s2_serve_controls.py --config <eval_config.yaml> --set <eval_set.pt> \
      --ckpt <run>/ckpt/<step> --out-dir <out_dir> --langs hi,te --per-lang 2 \
      --views spk_only,icl_spk --configs baseline,controlled [--device cuda:0] [--q-serve 16]

The checkpoint is loaded by the SHARED strict loader tts/model/load_s3.py, so an
S3 checkpoint builds and loads its speaker anchor and an S2 checkpoint does not — a disagreement
raises instead of silently serving a different model. --model-config <yaml> names the run config
that built the checkpoint (default: auto-discovered next to it, else --config);
--require-anchor {yes,no} asserts the arm before anything is built. The loaded model is echoed as
`CKPT_LOADED {...}` and stored in <out-dir>/run.json under "ckpt_provenance".

Configs (--configs): baseline = sampled stop, no chunking, no level control (the plain trained decode);
controlled = controlled stop + chunk-s (re-prompt on the ORIGINAL reference) + limiter; stop_only /
chunk_only / level_only ablations.

Further controls (added after listening tests reported "static / double voice from the start, worse by
the end, pure static once the transcript is done"):
  --greedy-from L      depth levels >= L are taken by argmax instead of sampled (the fine codebooks
                       sampled at temperature decode as broadband hiss on EVERY frame, and that hiss is
                       fed back into the model's context). Configs greedy9 / greedy17 isolate it.
  --deep-temp T        levels >= 9 sampled at temperature T (levels 1-8 keep --temp).
  --reprompt {orig,own}  chunk k > 1 re-prompts on the row's ORIGINAL reference block (orig, default:
                       clean human audio; for spk_only a fresh embedding-only context) or on the
                       model's own previous chunk (own, the original behaviour, which carries the drift
                       forward — te icl went 0.5% -> 8.4% HF by the last chunk in a short test).
  --static-stop H      every --static-check frames after the minimum length, decode the last 3 s and
                       stop when the HF fraction of the last 2 s exceeds H twice in a row (the
                       "transcript is over, only static remains" state), or when it is silent
                       (RMS < -45 dBFS). 0 disables.
  config "full" = controlled stop + chunking (orig) + limiter + greedy-from 9 + static-stop 0.05.
"""

import argparse
import collections
import json
import os
import re
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml

KIND_TEXT, KIND_AUDIO = 0, 1
SEED0 = 20260901
SR = 24000
FPS = 12.5
# per-language text -> frames prior (frames = a + b * chars), fitted on 40 shards of the
# training corpus (the --prior file, configs/text_frames_prior.json by default). resid_p99 = 99th percentile of the
# residual, used as the cap margin. Fallback when the JSON is not present.
PRIOR_DEFAULT = {"a": 2.0, "b": 0.72, "resid_p99": 90.0}
TERMINALS = "।.!?;।॥"


def load_prior(path):
    prior = {}
    if path and os.path.exists(path):
        for lang, d in json.load(open(path)).items():
            prior[lang] = {"a": float(d["affine_a"]), "b": float(d["affine_b"]),
                           "resid_p99": float(d["resid_p99"])}
    return prior


def expected_frames(prior, lang, nchar):
    p = prior.get(lang, PRIOR_DEFAULT)
    return p["a"] + p["b"] * nchar, p["resid_p99"]


# ----------------------------------------------------------------------------- text
def split_text(text, lang, prior, chunk_s):
    """Split at sentence terminals into pieces of about chunk_s seconds (prior-estimated).
    Falls back to commas, then whitespace, for run-on text."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * FPS
    # sentence pieces keep their terminal punctuation
    pieces = [p.strip() for p in re.split(f"(?<=[{re.escape(TERMINALS)}])\\s+", text) if p.strip()]
    out, cur = [], ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and expected_frames(prior, lang, len(cand))[0] > target:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    # a single piece longer than the budget: split at commas / whitespace
    final = []
    for piece in out:
        if expected_frames(prior, lang, len(piece))[0] <= target * 1.3:
            final.append(piece)
            continue
        words = re.split(r"(?<=[,،])\s+|\s+", piece)
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if cur and expected_frames(prior, lang, len(cand))[0] > target:
                final.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final


def row_text(ex, tok, ctl_ids):
    """Target-span text of an eval-set view, decoded from its text tokens without control tokens."""
    t, k = np.asarray(ex["tokens"]), np.asarray(ex["kind"])
    ts = int(ex.get("target_start", 0))
    ids = [int(x) for x in t[ts:][k[ts:] == KIND_TEXT] if int(x) not in ctl_ids]
    return tok.decode(ids).strip()


# ----------------------------------------------------------------------------- prompts
def make_example(packer, prefix, text, lang, spk_refs, spk_null):
    """Pack a prompt: an optional reference prefix (tokens/kind/codes/loss_mask arrays, already
    packed) followed by the target span for `text` (one dummy audio frame; the prompt is truncated
    at it)."""
    from tts.tokens import N_Q

    cut = {"text_final": text, "lang_final": lang, "codes": np.zeros((N_Q, 1), np.int64), "row": 0}
    first = prefix is None
    t, k, c, m = packer._span(cut, first=first, cont=False)
    if prefix is not None:
        t = np.concatenate([prefix["tokens"], t])
        k = np.concatenate([prefix["kind"], k])
        c = np.concatenate([prefix["codes"], c])
        m = np.concatenate([prefix["loss_mask"], m])
        ts = int(prefix["tokens"].shape[0])
    else:
        ts = 0
    return {"tokens": t, "kind": k, "codes": c, "loss_mask": m, "spk_refs": spk_refs,
            "spk_null": spk_null, "mode": "serve", "n_cuts": 1, "target_rows": [0],
            "ref_rows": [], "target_start": ts, "lang": lang}


def ref_prefix_from_view(ex):
    """The packed reference block of an icl view (everything before target_start)."""
    ts = int(ex.get("target_start", 0))
    if ts == 0:
        return None
    return {key: np.asarray(ex[key])[:ts] for key in ("tokens", "kind", "codes", "loss_mask")}


def ref_prefix_from_generated(packer, text, lang, codes_qF, q_serve, tail_frames):
    """Re-prompting reference: the previous chunk's text + the tail of its generated codes,
    packed as a first (lang-tagged) reference span. Levels >= q_serve are zero; the prompt is
    composed with q_use = q_serve so they are never read."""
    from tts.tokens import N_Q

    codes = codes_qF[:, -tail_frames:] if tail_frames > 0 else codes_qF
    full = np.zeros((N_Q, codes.shape[1]), np.int64)
    full[: codes.shape[0]] = codes.cpu().numpy() if torch.is_tensor(codes) else codes
    cut = {"text_final": text, "lang_final": lang, "codes": full, "row": 0}
    t, k, c, m = packer._span(cut, first=True, cont=False)
    return {"tokens": t, "kind": k, "codes": c, "loss_mask": m}


def prompt_from_example(ex, model, dev, dt, q_serve):
    """Collate one example with q_use = q_serve, cut at the target's first audio position and
    compose the backbone inputs. Returns (prompt_emb, n_prompt_tokens).

    The prompt goes through collate_v2 (PackerV2._span -> collate_v2), so the S3 per-span
    geometry is built by the SAME code as in training and is only sliced here — never
    hand-made. The speaker vector the anchor reads is collate_v2's `spk_refs`/`spk_ref_mask`
    /`spk_null` block, i.e. the same mean-pooled ECAPA-192 input as in training
    (tts/model/anchors.py:52-65)."""
    from tts.data.packing_v2 import collate_v2
    from tts.model.load_s3 import truncate_prompt_batch

    b = collate_v2([ex], q_choices=(q_serve,), q_rng=np.random.default_rng(0))
    kind = b["kind"][0]
    ts = int(ex.get("target_start", 0))
    a = (kind == KIND_AUDIO).nonzero().flatten()
    a = a[a >= ts]
    assert a.numel(), "prompt has no target audio position"
    a0 = int(a[0])
    # slice the span tables with the columns. The target span's <|audio|>
    # sentinel is at a0-1 (the last prompt position, whose hidden state generates frame 0)
    # and MUST survive the cut, or the anchor never reaches the generating position.
    pb = truncate_prompt_batch(b, a0,
                               require_target_span=model.speaker_anchor is not None)
    pb = {k: (v.to(dev, dtype=dt) if torch.is_tensor(v) and v.is_floating_point()
              else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in pb.items()}
    return model.compose_inputs(pb), a0


# ----------------------------------------------------------------------------- depth sampling
@torch.inference_mode()
def depth_generate(depth, h_frame, cb0, spk_mem, q_serve, temp, top_k, cfg_scale, ref_mask,
                   greedy_from=None, deep_temp=None, deep_from=9):
    """DepthV2.generate with per-level control: levels >= greedy_from by argmax; levels >= deep_from
    at deep_temp (else temp). Same prefix construction and CFG as the model's own generate."""
    from tts.model.depth_v2 import GROUP_OF_LEVEL
    from tts.tokens import AEOS_CLASS

    if torch.any(cb0 >= AEOS_CLASS) or torch.any(cb0 < 0):
        raise ValueError("aeos/invalid cb0 must be handled by the caller")
    B = h_frame.shape[0]
    null_mem = None
    if cfg_scale > 0:
        null_mem = depth.null_mem.to(spk_mem.dtype).expand(B, spk_mem.shape[1], -1)
    outs = [cb0]
    for level in range(1, q_serve):
        codes_prev = torch.stack(outs, 1)
        x = depth._prefix(h_frame, codes_prev, level)
        h = depth._run(x, GROUP_OF_LEVEL[level], spk_mem, ref_mask)
        logits = depth.heads[level - 1](h).float()
        if cfg_scale > 0:
            hn = depth._run(x, GROUP_OF_LEVEL[level], null_mem)
            ln = depth.heads[level - 1](hn).float()
            logits = ln + cfg_scale * (logits - ln)
        if greedy_from is not None and level >= greedy_from:
            outs.append(logits.argmax(-1))
            continue
        t = deep_temp if (deep_temp is not None and level >= deep_from) else temp
        logits = logits / t
        if top_k:
            kth = logits.topk(top_k, -1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, -float("inf"))
        outs.append(torch.multinomial(logits.softmax(-1), 1).squeeze(-1))
    return torch.stack(outs, -1)


def hf_of(w, sr=SR):
    if w.size < 256:
        return 0.0
    sp = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), 1 / sr)
    return float(sp[f >= 6000].sum() / (sp.sum() + 1e-12))


# ----------------------------------------------------------------------------- session
class ServeSession:
    """Backbone KV session + DepthV2 frame decoding with speaker memory, plus the stop policy."""

    def __init__(self, model, spk_mem, ref_mask, q_serve=16, temp=0.7, top_k=64, cfg_scale=0.0,
                 device="cuda:0", greedy_from=None, deep_temp=None, mimi=None, static_stop=0.0,
                 static_check=25):
        self.m, self.dev = model, device
        self.mem, self.rmask = spk_mem, ref_mask
        self.q, self.temp, self.top_k, self.cfg = q_serve, temp, top_k, cfg_scale
        self.greedy_from, self.deep_temp = greedy_from, deep_temp
        self.mimi, self.static_stop, self.static_check = mimi, static_stop, static_check
        self.past = None

    def _static_state(self, out):
        """Decode the last 3 s of generated codes; return (hf of the last 2 s, rms dBFS)."""
        codes = torch.stack(out[-38:], -1)[None].to(self.dev).long()
        w = self.mimi.decode(codes)[0][0, 0].float().cpu().numpy()
        w = w[-2 * SR:]
        rms = 20 * np.log10(np.sqrt((w ** 2).mean()) + 1e-9) if w.size else -99.0
        return hf_of(w), float(rms)

    def _step(self, emb):
        h, self.past = self.m.backbone(emb, past_key_values=self.past, use_cache=True)
        return h[:, -1]

    def _cb0_probs(self, h):
        logits = self.m.backbone.cb0_logits(h).float() / self.temp
        if self.top_k:
            kth = logits.topk(self.top_k, -1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, -float("inf"))
        return logits.softmax(-1)[0]

    @torch.inference_mode()
    def speak(self, prompt_emb, cap, stop="sampled", min_frames=0, tau=0.35, hazard=0.6):
        from tts.tokens import AEOS_CLASS

        h = self._step(prompt_emb.to(self.dev))
        out, p_trace, H = [], [], 0.0
        reason = "cap"
        static_hits = 0
        for t in range(cap):
            if (self.static_stop > 0 and self.mimi is not None and t >= max(min_frames, 38)
                    and t % self.static_check == 0):
                hf, rms = self._static_state(out)
                if rms < -45.0:
                    reason = "silence"
                    break
                static_hits = static_hits + 1 if hf >= self.static_stop else 0
                if static_hits >= 2:
                    reason = "static"
                    break
            p = self._cb0_probs(h)
            p_aeos = float(p[AEOS_CLASS])
            p_trace.append(p_aeos)
            if stop == "sampled":
                cb0 = int(torch.multinomial(p, 1))
                if cb0 == AEOS_CLASS:
                    reason = "aeos"
                    break
            else:
                if t >= min_frames:
                    H = 1.0 - (1.0 - H) * (1.0 - p_aeos)
                    if int(p.argmax()) == AEOS_CLASS:
                        reason = "aeos_argmax"
                        break
                    if p_aeos >= tau:
                        reason = "aeos_tau"
                        break
                    if H >= hazard:
                        reason = "aeos_hazard"
                        break
                q = p.clone()
                q[AEOS_CLASS] = 0.0
                q = q / q.sum()
                cb0 = int(torch.multinomial(q, 1))
            frame = depth_generate(
                self.m.depth, h.float(), torch.tensor([cb0], device=self.dev), self.mem, self.q,
                self.temp, self.top_k, self.cfg, self.rmask, greedy_from=self.greedy_from,
                deep_temp=self.deep_temp)[0]
            out.append(frame)
            fb = self.m.audio_in.frames(frame[None, :], frame.shape[0])
            h = self._step(fb[None].to(prompt_emb.dtype))
        if reason.startswith("aeos"):
            self._step(self.m.audio_in.aeos(1)[None].to(prompt_emb.dtype))
        codes = (torch.stack(out, -1) if out
                 else torch.zeros(self.q, 0, dtype=torch.long, device=self.dev))
        return codes, reason, p_trace


# ----------------------------------------------------------------------------- level + rails
def peak_norm(x, ceiling_db=-1.0):
    ceil = 10 ** (ceiling_db / 20)
    peak = float(np.abs(x).max()) if x.size else 0.0
    g = min(1.0, ceil / peak) if peak > 0 else 1.0
    return (x * g).astype(np.float32), g


def limiter(x, ceiling_db=-1.0, lookahead_s=0.005, release_s=0.08, sr=SR):
    """Look-ahead peak limiter: gain drops instantly (within the look-ahead window) to keep the
    windowed peak at the ceiling and recovers with an exponential release."""
    from scipy.ndimage import maximum_filter1d

    if x.size == 0:
        return x.astype(np.float32), 1.0
    ceil = 10 ** (ceiling_db / 20)
    L = max(1, int(lookahead_s * sr))
    env = maximum_filter1d(np.abs(x), size=2 * L + 1, mode="nearest")
    g = np.minimum(1.0, ceil / np.maximum(env, 1e-9))
    rel = float(np.exp(-1.0 / (release_s * sr)))
    out_g = np.empty_like(g)
    cur = 1.0
    for n in range(len(g)):
        gn = g[n]
        cur = gn if gn < cur else gn + (cur - gn) * rel
        out_g[n] = cur
    y = np.clip(x * out_g, -1.0, 1.0).astype(np.float32)
    return y, float(out_g.min())


def rails(w, sr=SR):
    a = np.abs(w)
    n = len(w)
    pinned = a >= 0.999
    runs = 0
    if n:
        d = np.diff(np.concatenate([[0], pinned.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        runs = int(((ends - starts) >= 3).sum())
    sec = w[: (n // sr) * sr].reshape(-1, sr) if n >= sr else np.zeros((0, sr))
    rms_s = 20 * np.log10(np.sqrt((sec ** 2).mean(1)) + 1e-9) if len(sec) else np.array([])
    slope = float(np.polyfit(np.arange(len(rms_s)), rms_s, 1)[0]) if len(rms_s) >= 4 else None
    end_start = (float(rms_s[-3:].mean() - rms_s[:3].mean()) if len(rms_s) >= 6 else None)
    if n:
        sp = np.abs(np.fft.rfft(w)) ** 2
        f = np.fft.rfftfreq(n, 1 / sr)
        hf = float(sp[f >= 6000].sum() / (sp.sum() + 1e-12))
    else:
        hf = None
    thirds = None
    if len(sec) >= 3:
        hfs = [hf_of(x) for x in sec]
        t3 = len(hfs) // 3
        thirds = [round(float(np.mean(hfs[:t3])), 4), round(float(np.mean(hfs[t3:2 * t3])), 4),
                  round(float(np.mean(hfs[2 * t3:])), 4)]
    silent_run = 0
    best = 0
    for r in rms_s:
        silent_run = silent_run + 1 if r < 20 * np.log10(0.005) else 0
        best = max(best, silent_run)
    return {
        "dur_s": round(n / sr, 2),
        "peak_dbfs": round(float(20 * np.log10(a.max() + 1e-9)), 2) if n else None,
        "rms_dbfs": round(float(20 * np.log10(np.sqrt((w ** 2).mean()) + 1e-9)), 2) if n else None,
        "clip_frac": round(float(pinned.mean()), 5) if n else None,
        "flat_top_runs": runs,
        "rms_slope_db_s": None if slope is None else round(slope, 3),
        "rms_end_minus_start_db": None if end_start is None else round(end_start, 2),
        "hf_frac_gt6k": None if hf is None else round(hf, 4),
        "hf_thirds": thirds,
        "silent_ge3s": bool(best >= 3),
    }


def save_wav(p, w):
    with wave.open(p, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes())


CONFIGS = {
    # stop / chunk / level are the base controls; greedy_from / deep_temp / reprompt / static_stop the further ones
    "baseline":   {"stop": "sampled",    "chunk": False, "level": "none"},
    "controlled": {"stop": "controlled", "chunk": True,  "level": "limiter", "reprompt": "orig"},
    "stop_only":  {"stop": "controlled", "chunk": False, "level": "none"},
    "chunk_only": {"stop": "sampled",    "chunk": True,  "level": "none", "reprompt": "orig"},
    "chunk_own":  {"stop": "sampled",    "chunk": True,  "level": "none", "reprompt": "own"},
    "level_only": {"stop": "sampled",    "chunk": False, "level": "limiter"},
    "greedy17":   {"stop": "sampled",    "chunk": False, "level": "none", "greedy_from": 17},
    "greedy9":    {"stop": "sampled",    "chunk": False, "level": "none", "greedy_from": 9},
    "deep05":     {"stop": "sampled",    "chunk": False, "level": "none", "deep_temp": 0.5},
    "full":       {"stop": "controlled", "chunk": True,  "level": "limiter", "reprompt": "orig",
                   "greedy_from": 9, "static_stop": 0.05},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="set_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--langs", default="hi,te")
    ap.add_argument("--per-lang", type=int, default=2)
    ap.add_argument("--views", default="spk_only,icl_spk")
    ap.add_argument("--configs", default="baseline,controlled")
    ap.add_argument("--targets", choices=["long", "orig"], default="long")
    ap.add_argument("--long-chars", type=int, default=450)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--q-serve", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--hard-cap", type=int, default=1500)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--hazard", type=float, default=0.6)
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="minimum length as a fraction of the prior's expected frames")
    ap.add_argument("--chunk-s", type=float, default=16.0)
    ap.add_argument("--ref-tail-frames", type=int, default=150)
    ap.add_argument("--level", default=None, help="override the config's level control")
    ap.add_argument("--ceiling-db", type=float, default=-1.0)
    ap.add_argument("--prior", default=os.path.join(os.path.dirname(__file__), "..", "configs",
                                                     "text_frames_prior.json"))
    ap.add_argument("--greedy-from", type=int, default=None, help="override: levels >= L by argmax")
    ap.add_argument("--deep-temp", type=float, default=None, help="override: levels >= 9 at this temp")
    ap.add_argument("--reprompt", choices=["orig", "own"], default=None, help="override chunk re-prompt source")
    ap.add_argument("--static-stop", type=float, default=None, help="override: HF fraction that stops (0 off)")
    ap.add_argument("--static-check", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED0)
    # the run config that BUILT the checkpoint (speaker anchor etc). Default =
    # auto-discovered next to the checkpoint, else --config. --require-anchor asserts the arm.
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--require-anchor", choices=["yes", "no"], default=None)
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    cfg = yaml.safe_load(open(A.config))
    dev = A.device
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    os.makedirs(A.out_dir + "/wav", exist_ok=True)
    prior = load_prior(A.prior)
    if not prior:
        print(f"WARNING: no prior at {A.prior}; using the default {PRIOR_DEFAULT}", flush=True)
    views = A.views.split(",")
    configs = A.configs.split(",")
    langs = A.langs.split(",")

    from transformers import AutoTokenizer

    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    from tts.tokens import extend_tokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    tok, ids = extend_tokenizer(tok)
    ctl_ids = set(int(v) for v in ids.values())
    packer = PackerV2(tok, ids, max_tokens=int(cfg["data"].get("max_tokens", 8192)))
    if dev.startswith("cuda"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    t0 = time.time()
    # shared STRICT loader (tts/model/load_s3.py). A loader that always built an anchor-free
    # model and never asked for the checkpoint's extra tensors would silently serve a different
    # model from an anchored checkpoint.
    model, prov = load_model_for_eval(
        A.ckpt, device=dev, dtype=dt, cfg=cfg, config=A.model_config,
        amortize_frac=1.0,          # unused here (generation never calls forward())
        require_anchor=None if A.require_anchor is None
        else A.require_anchor == "yes")
    model.depth.float()
    print(f"model loaded in {time.time() - t0:.0f}s on {dev} "
          f"(anchor={prov['anchor']}, update={prov['update']})", flush=True)

    from transformers import MimiModel

    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    data = torch.load(A.set_path, weights_only=False)
    by_lang = collections.defaultdict(list)
    for i, row in enumerate(data["rows"]):
        by_lang[row["lang"]].append((i, row))
    picked = []
    for lang in langs:
        rows = by_lang.get(lang, [])
        picked += [(lang, i, r) for i, r in rows[: A.per_lang]]
    # long targets: concatenate OTHER rows' target texts of the same language
    texts = {lang: [row_text(r["views"]["spk_only"], tok, ctl_ids) for _, r in by_lang[lang]]
             for lang in langs}

    recs = []
    for lang, i, row in picked:
        own = row_text(row["views"]["spk_only"], tok, ctl_ids)
        if A.targets == "long":
            pool = [t for t in texts[lang] if t != own] or [own]
            text, k = own, 0
            while len(text) < A.long_chars:
                text = text + " " + pool[k % len(pool)]
                k += 1
        else:
            text = own
        exp_frames, _resid = expected_frames(prior, lang, len(text))
        for view in views:
            ex0 = row["views"][view]
            spk_refs = np.asarray(ex0["spk_refs"], np.float32)
            spk_null = bool(ex0.get("spk_null", True))
            refs_t = torch.as_tensor(spk_refs, dtype=torch.float32, device=dev)[None]
            null_t = torch.tensor([spk_null], device=dev)
            mem = model.depth.spk_memory(refs_t, "ecapa", null_mask=null_t)
            rmask = torch.ones(1, refs_t.shape[1], dtype=torch.bool, device=dev)
            icl_prefix = ref_prefix_from_view(ex0) if view.startswith("icl") else None
            for cname in configs:
                c = dict(CONFIGS[cname])
                if A.level is not None:
                    c["level"] = A.level
                greedy_from = A.greedy_from if A.greedy_from is not None else c.get("greedy_from")
                deep_temp = A.deep_temp if A.deep_temp is not None else c.get("deep_temp")
                reprompt = A.reprompt or c.get("reprompt", "orig")
                static_stop = A.static_stop if A.static_stop is not None else c.get("static_stop", 0.0)
                chunks = split_text(text, lang, prior, A.chunk_s if c["chunk"] else 0)
                torch.manual_seed(A.seed + 7919 * i + 101 * views.index(view))
                gen_codes, reasons, frames_per_chunk, caps = [], [], [], []
                prev_text, prev_codes = None, None
                tg = time.time()
                for ci, ctext in enumerate(chunks):
                    if ci == 0 or reprompt == "orig":
                        prefix = icl_prefix
                    else:
                        prefix = ref_prefix_from_generated(packer, prev_text, lang, prev_codes,
                                                           A.q_serve, A.ref_tail_frames)
                    ex = make_example(packer, prefix, ctext, lang, spk_refs, spk_null)
                    emb, _ = prompt_from_example(ex, model, dev, dt, A.q_serve)
                    ef, er = expected_frames(prior, lang, len(ctext))
                    if c["stop"] == "controlled":
                        cap = min(A.hard_cap, int(ef + er + 10))
                        min_frames = max(4, int(A.min_frac * ef))
                    else:
                        cap = A.hard_cap
                        min_frames = 0
                    sess = ServeSession(model, mem, rmask, q_serve=A.q_serve, temp=A.temp,
                                        top_k=A.top_k, cfg_scale=A.cfg if not spk_null else 0.0,
                                        device=dev, greedy_from=greedy_from, deep_temp=deep_temp,
                                        mimi=mimi, static_stop=static_stop,
                                        static_check=A.static_check)
                    codes, reason, _ = sess.speak(emb.to(dt), cap, stop=c["stop"],
                                                  min_frames=min_frames, tau=A.tau,
                                                  hazard=A.hazard)
                    gen_codes.append(codes)
                    reasons.append(reason)
                    frames_per_chunk.append(int(codes.shape[-1]))
                    caps.append(cap)
                    prev_text, prev_codes = ctext, codes
                    print(f"  [{lang} r{i} {view} {cname}] chunk {ci + 1}/{len(chunks)} "
                          f"{codes.shape[-1]}f cap {cap} stop={reason} "
                          f"({time.time() - tg:.0f}s)", flush=True)
                codes = torch.cat(gen_codes, -1) if gen_codes else None
                recs.append({"lang": lang, "row": i, "view": view, "config": cname, "text": text,
                             "nchar": len(text), "exp_frames": round(exp_frames, 1),
                             "chunks": len(chunks), "frames": frames_per_chunk, "caps": caps,
                             "stops": reasons, "gen_frames": int(codes.shape[-1]),
                             "level": c["level"], "greedy_from": greedy_from,
                             "deep_temp": deep_temp, "reprompt": reprompt if c["chunk"] else None,
                             "static_stop": static_stop, "codes": codes.cpu()})
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    summary = collections.defaultdict(list)
    with torch.inference_mode(), open(A.out_dir + "/rails.jsonl", "w") as fh:
        for n, rec in enumerate(recs):
            codes = rec.pop("codes")
            if codes.shape[-1] == 0:
                w = np.zeros(0, np.float32)
            else:
                w = mimi.decode(codes[None].to(dev).long())[0][0, 0].float().cpu().numpy()
            raw = rails(w)
            if rec["level"] == "peaknorm":
                y, g = peak_norm(w, A.ceiling_db)
            elif rec["level"] == "limiter":
                y, g = limiter(w, A.ceiling_db)
            else:
                y, g = np.clip(w, -1, 1).astype(np.float32), 1.0
            proc = rails(y)
            base = f"{A.out_dir}/wav/{n:03d}_{rec['lang']}_r{rec['row']}_{rec['view']}_{rec['config']}"
            save_wav(base + "_raw.wav", w)
            if rec["level"] != "none":
                save_wav(base + "_proc.wav", y)
            rec.update({"raw": raw, "proc": proc, "min_gain": round(float(g), 3),
                        "dur_ratio_vs_prior": round(rec["gen_frames"] / max(1.0, rec["exp_frames"]), 3)})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            key = (rec["view"], rec["config"])
            summary[key].append(rec)
            print(f"{n:03d} {rec['lang']} r{rec['row']} {rec['view']:8s} {rec['config']:10s} "
                  f"{rec['gen_frames']:4d}f/{rec['exp_frames']:.0f} exp, chunks {rec['chunks']}, "
                  f"stops {','.join(rec['stops'])} | RAW peak {raw['peak_dbfs']} rms {raw['rms_dbfs']} "
                  f"clip {raw['clip_frac']} runs {raw['flat_top_runs']} slope {raw['rms_slope_db_s']} "
                  f"hf {raw['hf_frac_gt6k']} | PROC peak {proc['peak_dbfs']} clip {proc['clip_frac']} "
                  f"runs {proc['flat_top_runs']}", flush=True)
    print("\n=== SUMMARY (means over rows; RAW decode unless marked proc) ===")
    print(f"{'view':9s}{'config':11s}{'n':>3s}{'frames':>8s}{'dur/exp':>8s}{'peak':>7s}{'rms':>7s}"
          f"{'clip%':>7s}{'runs':>6s}{'slope':>7s}{'end-st':>7s}{'hf%':>6s}{'procclip%':>10s}{'procruns':>9s}")
    for (view, cname), rs in summary.items():
        def mean(key, sub="raw", rs=rs):
            vals = [r[sub][key] for r in rs if r[sub][key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"{view:9s}{cname:11s}{len(rs):3d}{sum(r['gen_frames'] for r in rs) / len(rs):8.0f}"
              f"{sum(r['dur_ratio_vs_prior'] for r in rs) / len(rs):8.2f}{mean('peak_dbfs'):7.2f}"
              f"{mean('rms_dbfs'):7.2f}{100 * mean('clip_frac'):7.3f}{mean('flat_top_runs'):6.0f}"
              f"{mean('rms_slope_db_s'):7.3f}{mean('rms_end_minus_start_db'):7.2f}"
              f"{100 * mean('hf_frac_gt6k'):6.2f}{100 * mean('clip_frac', 'proc'):10.3f}"
              f"{mean('flat_top_runs', 'proc'):9.0f}")
    json.dump({"args": vars(A), "n": len(recs), "ckpt_provenance": prov},
              open(A.out_dir + "/run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
