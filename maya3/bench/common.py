"""Shared bench helpers: default asset paths (env-overridable), row loading, pack voices."""
import json, os
import numpy as np
H = os.path.expanduser("~")
ASSETS = os.environ.get("M3_ASSETS", "assets")              # bench/test assets dir (backbone/, pack/, ...)
HF = os.environ.get("HF_HOME", f"{H}/.cache/huggingface")
BACKBONE = os.environ.get("M3_BACKBONE", f"{ASSETS}/backbone")
MIMI = f"{HF}/hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7"
ECAPA = f"{HF}/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ROWS = os.environ.get("INFER_ROWS", "rows.jsonl")


def load_pack(pack_dir=f"{ASSETS}/pack", codes_dir=f"{ASSETS}/speaker_pack_v1"):
    refs = json.load(open(f"{pack_dir}/refs.json"))
    ec = np.load(f"{pack_dir}/ecapa.npz")
    return {r["pack_uid"]: dict(r, codes=np.load(f"{codes_dir}/{r['code_file']}").astype(np.int64),
                                ecapa=ec[r["pack_uid"]].astype(np.float32), text=r["text_train"]) for r in refs}


def rows(path=ROWS):
    return [json.loads(l) for l in open(path) if l.strip()]


def register_pack_voices(engine, pack, rows_):
    """One registry voice per (ref_uid, mode) used by the rows; returns row -> voice id."""
    from maya3.prompt import Voice
    out = {}
    for r in rows_:
        if r.get("voice"):                   # named (trained) voice selected by name
            out[r["id"]] = r["voice"]; continue
        if not r.get("ref_uid") or r["mode"] == "plain":
            out[r["id"]] = "plain"; continue
        vid = f"{r['ref_uid']}|{r['mode']}"
        if vid not in engine.registry:
            p = pack[r["ref_uid"]]
            engine.registry.add(Voice(vid, r["mode"], clips=[{"text": p["text"], "codes": p["codes"], "lang": p["lang_final"]}],
                                      ecapa=p["ecapa"]))
        out[r["id"]] = vid
    return out
