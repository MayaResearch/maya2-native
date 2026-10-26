"""Write the PUBLIC voices/ dir of a release: voices.json (name, prefix, gender, age band, style, trained languages, hours,
per-voice mode) + the ECAPA centroids / reference-clip codes the chosen modes need. Drops provider ids, source speaker ids,
pack names, catalogue uids and paths.
Usage: make_release_voices.py <internal voices.json> <out voices dir> [--mode named|spk|icl_spk] [--fallback-dir DIR]
       [--mode-per-voice Name=mode,...]"""
import argparse, json, os, shutil
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--mode", default="named"); ap.add_argument("--fallback-dir", default=None)
ap.add_argument("--mode-per-voice", default="")
A = ap.parse_args()
per = dict(x.split("=") for x in A.mode_per_voice.split(",") if x)
d = json.load(open(A.src)); os.makedirs(A.out, exist_ok=True)
fb = {}
if A.fallback_dir:
    fb = {v["name"]: v for v in json.load(open(os.path.join(A.fallback_dir, "voices_icl_spk.json")))["voices"]}
pub = []
for v in d["voices"]:
    n = v["name"]; mode = per.get(n, A.mode)
    e = {"name": n, "prompt_prefix": v.get("prompt_prefix", "{" + n + "}: "), "mode": mode, "gender": v.get("gender"),
         "age_band": v.get("age_band"), "style": v.get("style"), "langs": v.get("langs_ge1h") or v.get("langs"), "hours": v.get("hours")}
    if mode in ("spk", "icl_spk", "icl"):
        f = fb[n]
        shutil.copy(os.path.join(A.fallback_dir, f["ecapa"]), os.path.join(A.out, f["ecapa"])); e["ecapa"] = f["ecapa"]
        if mode != "spk":
            e["ref_clips"] = []
            for c in f["ref_clips"][:1]:
                shutil.copy(os.path.join(A.fallback_dir, c["codes"]), os.path.join(A.out, c["codes"]))
                e["ref_clips"].append({"codes": c["codes"], "text": c["text"], "lang": c["lang"], "seconds": c.get("seconds")})
    pub.append(e)
json.dump({"schema": "maya3_voices_v1", "prompt_format": "{Name}: <text>",
           "note": "Select with voice=<name> (case-insensitive). mode: named = trained name prompt; spk = stored speaker embedding; "
                   "icl_spk = embedding + reference clip. Never combine a named voice with your own reference audio.",
           "voices": pub}, open(os.path.join(A.out, "voices.json"), "w"), indent=1, ensure_ascii=False)
print(f"wrote {len(pub)} voices ({sorted(set(x['mode'] for x in pub))}) -> {A.out}")
