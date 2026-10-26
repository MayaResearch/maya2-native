"""Voices and prompt building for the Maya3 engine.

A *voice* is everything about a request that does not depend on the target text: the conditioning mode, the packed
reference prefix (reference text + its 32-codebook Mimi codes), the ECAPA-192 speaker embedding and the depth speaker
memory computed from it.  Because the prompt layout tags the first reference span with the REFERENCE language
(not the target language), the packed prefix -- and therefore its backbone KV -- is a constant of the voice and is
computed once and cached (see engine.py).

Prompt geometry is the training packer's (tts.data.packing_v2.PackerV2._span + collate_v2, via
scripts/s2_serve_controls); this module only decides WHICH spans to pack, exactly as the training-time example builder
does (layout "v5"; checked token for token against it):

    plain   : <|lang:B|> target <|audio|>                         (speaker memory nulled)
    spk     : same tokens, ECAPA anchor + depth speaker memory
    icl     : <|lang:A|> ref <|audio|> codes AEOS [<|lang:B|> if A != B] target <|audio|>   (speaker memory nulled)
    icl_spk : icl + ECAPA (the full zero-shot clone; the default for a reference clip)

Prompt builders are pluggable: a Voice carries a `builder` (default V5Builder).  Named speakers
("{Name}: text"-style prefixes, format not frozen) become registry entries with their own builder and a cached
prefix; the engine needs no new path (see NamedSpeakerBuilder).
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Callable, Optional

import numpy as np
import torch

MODES = {"plain": (False, False), "spk": (False, True), "icl": (True, False), "icl_spk": (True, True),
         "named": (False, False)}  # (use_icl, use_mem); named = a trained voice selected by its name in the text, memory nulled
MODE_ALIASES = {"joint_null": "plain", "spk_only": "spk", "icl_only": "icl", "clone": "icl_spk"}


def norm_mode(m):
    m = MODE_ALIASES.get(m, m)
    if m not in MODES:
        raise ValueError(f"unknown conditioning mode {m!r}; choose one of {sorted(MODES)}")
    return m


@dataclasses.dataclass
class Voice:
    """A registered voice.  `clips`: list of {text, codes (32,F) int, lang}; `ecapa`: (192,) L2-normalised or None."""
    id: str
    mode: str = "plain"
    clips: list = dataclasses.field(default_factory=list)
    ecapa: Optional[np.ndarray] = None
    builder: Optional["PromptBuilder"] = None
    meta: dict = dataclasses.field(default_factory=dict)
    # ---- cached, filled lazily by the engine (never by users)
    prefix: Optional[dict] = None          # packed prefix arrays {tokens, kind, codes, loss_mask}
    spk_refs: Optional[np.ndarray] = None  # (1,192) float32
    mem: Optional[torch.Tensor] = None     # (1,1,D) depth speaker memory
    rmask: Optional[torch.Tensor] = None   # (1,1) bool
    kv: Optional[list] = None              # [(k, v)] per layer, [1, KV, ts, Dh] (backbone prefix KV)
    emb: Optional[torch.Tensor] = None     # [1, ts, D] prefix input embeddings (for the exactness check)
    ts: int = 0                            # prefix length in tokens (0 = nothing to cache)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.mode = norm_mode(self.mode)
        use_icl, use_mem = MODES[self.mode]
        if use_icl and not self.clips:
            raise ValueError(f"voice {self.id}: mode {self.mode} needs at least one reference clip")
        if use_mem and self.ecapa is None:
            raise ValueError(f"voice {self.id}: mode {self.mode} needs an ECAPA embedding")
        if self.ecapa is not None:
            e = np.asarray(self.ecapa, np.float32).reshape(-1, 192)
            if e.shape[0] != 1:
                raise ValueError("the fast depth path supports exactly one ECAPA reference embedding per voice")
            self.ecapa = e[0]

    @property
    def ref_lang(self):
        return self.clips[0]["lang"] if self.clips else None


class PromptBuilder:
    """Interface: prefix(packer, voice) -> packed prefix dict or None (constant per voice; cached);
    target(packer, voice, text, lang) -> example dict for s2_serve_controls.prompt_from_example."""

    def prefix(self, packer, voice):
        raise NotImplementedError

    def example(self, packer, voice, text, lang):
        raise NotImplementedError


class V5Builder(PromptBuilder):
    """The training layout (layout='v5' in the training-time example builder)."""

    def prefix(self, packer, voice):
        use_icl, _ = MODES[voice.mode]
        if not use_icl:
            return None
        spans = []
        for i, cl in enumerate(voice.clips):
            cut = {"text_final": cl["text"], "lang_final": cl["lang"], "codes": np.asarray(cl["codes"], np.int64), "row": 0}
            spans.append(packer._span(cut, first=(i == 0), cont=False))
        return {key: np.concatenate([sp[j] for sp in spans]) for j, key in enumerate(("tokens", "kind", "codes", "loss_mask"))}

    def example(self, packer, voice, text, lang):
        from scripts import s2_serve_controls as S
        from tts.tokens import N_Q
        _, use_mem = MODES[voice.mode]
        prefix = voice.prefix
        xling = prefix is not None and any(cl["lang"] != lang for cl in voice.clips)
        ex = S.make_example(packer, prefix, text, lang, voice.spk_refs, not use_mem)
        if xling:  # the target's first span gets its own language tag when the references are in another language
            cut = {"text_final": text, "lang_final": lang, "codes": np.zeros((N_Q, 1), np.int64), "row": 0}
            t, k, c, m = packer._span(cut, first=True, cont=False)
            ts = ex["target_start"]
            for key, arr in (("tokens", t), ("kind", k), ("codes", c), ("loss_mask", m)):
                ex[key] = np.concatenate([prefix[key], arr])
            assert ex["target_start"] == ts
        ex["xling"] = xling
        return ex


class NamedSpeakerBuilder(V5Builder):
    """Named speakers: the voice is selected by its name in the text conditioning,
        <|lang:xx|>{template.format(name=Name, text=text)}<|audio|>
    with no reference audio and a nulled speaker memory (like plain). The template follows the trainer's
    named-speaker prompt spec; it was checked token for token against the
    trainer's packer. Nothing is cached per voice (the language tag precedes the name), the whole prompt is short."""

    def __init__(self, template="{{{name}}}: {text}"):
        self.template = template

    def render(self, voice, text):
        return self.template.format(name=voice.meta.get("name", voice.id), text=text.strip())

    def prefix(self, packer, voice):
        return None

    def example(self, packer, voice, text, lang):
        return super().example(packer, voice, self.render(voice, text), lang)


def load_named_voices(registry, voices_json, template=None, replace=True):
    """Register the trained named voices of voices.json (list, or {"voices": [...]}); each entry needs a name
    ("name"/"id"/"display_name"); every other field is kept in Voice.meta (gender, age, style, langs, provenance...).
    Returns the registered ids. Lookup is case-insensitive (VoiceRegistry.get)."""
    import json
    d = json.load(open(voices_json))
    items = d.get("voices", d) if isinstance(d, dict) else d
    tmpl = template or (d.get("template") if isinstance(d, dict) else None)
    builder = NamedSpeakerBuilder(tmpl) if tmpl else NamedSpeakerBuilder()
    import os
    import numpy as np
    base = os.path.dirname(os.path.abspath(voices_json))
    ids = []
    for it in items:
        name = it.get("name") or it.get("display_name") or it.get("id")
        if not name:
            raise ValueError(f"voice entry without a name: {it}")
        meta = {k: v for k, v in it.items() if k not in ("ecapa", "centroid", "ref_clips")}
        meta["name"] = name
        mode = norm_mode(it.get("mode", "named"))
        if mode == "named":
            v = Voice(str(name), "named", builder=builder, meta=meta)
        else:
            # per-voice fallback: the voice's stored ECAPA centroid (spk) and/or curated reference clip(s) (icl / icl_spk)
            ecapa = np.load(os.path.join(base, it["ecapa"])).astype(np.float32).reshape(-1) if it.get("ecapa") else None
            clips = []
            if MODES[mode][0]:
                for c in (it.get("ref_clips") or [])[: int(it.get("max_ref_clips", 1))]:
                    clips.append({"text": c["text"], "lang": c["lang"],
                                  "codes": np.load(os.path.join(base, c["codes"])).astype(np.int64)})
            v = Voice(str(name), mode, clips=clips, ecapa=ecapa, meta=meta)
        registry.add(v, replace=replace)
        ids.append(str(name))
    return ids


DEFAULT_BUILDER = V5Builder()


class VoiceRegistry:
    """Thread-safe id -> Voice map.  `plain` is always present (no reference, speaker memory nulled)."""

    def __init__(self):
        self._v = {"plain": Voice("plain", "plain")}
        self._lock = threading.Lock()

    def add(self, voice: Voice, replace=True):
        with self._lock:
            if not replace and voice.id in self._v:
                raise KeyError(f"voice {voice.id} exists")
            self._v[voice.id] = voice
        return voice

    def get(self, vid) -> Voice:
        with self._lock:
            if vid in self._v:
                return self._v[vid]
            low = str(vid).lower()     # case-insensitive fallback (named voices: "asha" == "Asha")
            hits = [v for k, v in self._v.items() if k.lower() == low]
            if len(hits) == 1:
                return hits[0]
            raise KeyError(f"unknown voice {vid!r}")

    def remove(self, vid):
        with self._lock:
            if vid == "plain":
                raise KeyError("cannot remove the plain voice")
            return self._v.pop(vid, None)

    def ids(self):
        with self._lock:
            return list(self._v)

    def __contains__(self, vid):
        with self._lock:
            return vid in self._v
