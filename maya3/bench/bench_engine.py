"""maya3.Engine latency/throughput bench (same protocol as baseline_fastbatch.py, through the public submit() path).

  B=1 : one request at a time; TTFA = submit -> first PCM chunk delivered to the callback (prompt build, prefix-KV copy,
        inline prefill, first frame, Mimi, D2H all included); RTF = wall / audio.
  N>1 : closed loop, N requests in flight; a finished request immediately submits the next; reps*N requests total.
        TTFA = submit -> first chunk (queue wait included).  Tick p50/p99 from the engine.
Usage: bench_engine.py <model.pt> <out.json> [--seats 1,8,32,64,100] [--depth fp16] [--no-guards]
"""
import argparse, json, os, sys, threading, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maya3.engine import Engine, EngineConfig, SR
from maya3.bench import common as C

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--seats", default="1,8,32,64,100"); ap.add_argument("--depth", default="fp16")
ap.add_argument("--no-guards", action="store_true"); ap.add_argument("--b1-n", type=int, default=40)
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--capacity", type=int, default=2304)
ap.add_argument("--max-inline-rows", type=int, default=384); ap.add_argument("--max-admit", type=int, default=8)
ap.add_argument("--label", default="maya3"); ap.add_argument("--cb0-rep", type=float, default=None); ap.add_argument("--ramp-s", type=float, default=2.0)
A = ap.parse_args()
q = lambda x, p: round(float(np.quantile(x, p)), 4) if len(x) else None
rows = C.rows()
pack = C.load_pack()
res = {"ckpt": A.ckpt, "label": A.label, "depth": A.depth, "guards": not A.no_guards, "gpu": torch.cuda.get_device_name(), "runs": {}}
E = None
for N in [int(x) for x in A.seats.split(",")]:
    cfg = EngineConfig(max_seats=N, capacity=A.capacity, depth_dtype=A.depth, max_inline_rows=A.max_inline_rows,
                       max_admit_per_tick=A.max_admit, free_reference_depth=False,
                       **({} if A.cb0_rep is None else {"cb0_rep": A.cb0_rep}))   # rebuilt per N below
    cfg.guards.enabled = not A.no_guards
    if E is None:
        E = Engine(A.ckpt, C.BACKBONE, C.MIMI, cfg)
        model, mimi, registry = E.model, E.mimi, E.registry
    else:   # rebuild the fast components only (same weights)
        E.cfg = cfg; E.backbone = E.depth = E.codec = None; torch.cuda.empty_cache(); E._build_fast()
    vmap = C.register_pack_voices(E, pack, rows)
    t0 = time.time(); E.warmup(); warm = time.time() - t0
    for v in set(vmap.values()):
        E.prepare_voice(E.registry.get(v))
    E.start()
    reqs = [(r["pieces"][0], r["lang"], vmap[r["id"]], r["mode"]) for r in rows]
    lock = threading.Lock()
    rec = {"warmup_s": round(warm, 1)}
    # extra warm-up through the live loop
    ev = threading.Event(); E.submit(reqs[0][0], reqs[0][1], reqs[0][2], on_event=lambda k, p: ev.set() if k != "audio" else None); ev.wait(60)
    torch.cuda.reset_peak_memory_stats(); E.tick_ms.clear()
    if N == 1:
        ttfa, rtf, audio = {"plain": [], "icl_spk": []}, {"plain": [], "icl_spk": []}, 0.0
        tw = time.time()
        for i, (text, lang, vid, mode) in enumerate(reqs[:A.b1_n]):
            done = threading.Event(); info = {}
            t_sub = time.perf_counter()
            first = []
            def cb(k, p, first=first, info=info, done=done):
                if k == "audio":
                    if not first: first.append(time.perf_counter())
                else:
                    info.update(p if isinstance(p, dict) else {}); info["kind"] = k; done.set()
            E.submit(text, lang, vid, on_event=cb, split=False)
            done.wait(120)
            if info.get("kind") != "end" or not first: continue
            ttfa[mode].append(first[0] - t_sub); rtf[mode].append(info["wall_s"] / max(info["audio_s"], 1e-3)); audio += info["audio_s"]
        wall = time.time() - tw
        rec.update({"n_req": A.b1_n, "audio_s_per_s": round(audio / wall, 2)})
        for m in ttfa:
            rec[m] = {"n": len(ttfa[m]), "ttfa_ms_p50": round(q(ttfa[m], .5) * 1e3, 1), "ttfa_ms_p90": round(q(ttfa[m], .9) * 1e3, 1),
                      "ttfa_ms_p99": round(q(ttfa[m], .99) * 1e3, 1), "rtf_p50": q(rtf[m], .5), "rtf_p90": q(rtf[m], .9)}
    else:
        total = A.reps * N
        state = {"sent": 0, "done": 0, "audio": 0.0}
        tt, rt, reasons, steady = [], [], {}, []
        alldone = threading.Event()
        def send():
            with lock:
                if state["sent"] >= total: return
                i = state["sent"]; state["sent"] += 1
            text, lang, vid, mode = reqs[i % len(reqs)]
            t_sub = time.perf_counter(); first = []
            def cb(k, p):
                if k == "audio":
                    if not first: first.append(time.perf_counter())
                    return
                with lock:
                    state["done"] += 1
                    if k == "end":
                        tt.append(first[0] - t_sub if first else None); rt.append(p["wall_s"] / max(p["audio_s"], 1e-3))
                        if t_sub - tw0[0] > A.ramp_s + 1.0 and first: steady.append((first[0] - t_sub, rt[-1]))
                        state["audio"] += p["audio_s"]; reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
                    else:
                        reasons[p.get("code")] = reasons.get(p.get("code"), 0) + 1
                    fin = state["done"] >= total
                if fin: alldone.set()
                else: send()
            E.submit(text, lang, vid, on_event=cb, split=False)
        tw = time.time(); tw0 = [time.perf_counter()]
        for i in range(N): threading.Timer(A.ramp_s * i / N, send).start()
        alldone.wait(3600)
        wall = time.time() - tw
        tt = [x for x in tt if x is not None]
        tk = sorted(E.tick_ms)
        rec.update({"n_req": total, "wall_s": round(wall, 1), "audio_s": round(state["audio"], 1),
                    "audio_s_per_s": round(state["audio"] / wall, 2), "ttfa_ms_p50": round(q(tt, .5) * 1e3, 1),
                    "ttfa_ms_p99": round(q(tt, .99) * 1e3, 1), "rtf_p50": q(rt, .5), "rtf_p99": q(rt, .99), "reasons": reasons, "ramp_s": A.ramp_s,
                    "steady": {"n": len(steady), "ttfa_ms_p50": round(q([a for a, _ in steady], .5) * 1e3, 1) if steady else None,
                               "ttfa_ms_p99": round(q([a for a, _ in steady], .99) * 1e3, 1) if steady else None,
                               "rtf_p99": q([b for _, b in steady], .99) if steady else None},
                    "tick_ms_p50": round(tk[len(tk) // 2], 2), "tick_ms_p99": round(tk[int(len(tk) * .99)], 2)})
    rec["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rec["phase_ms_last"] = E.phase_ms
    rec["stats"] = dict(E.stats)
    E.close(); E._thread = None; E._stop = False
    res["runs"][f"N{N}"] = rec
    print(f"N={N}", json.dumps(rec), flush=True)
    json.dump(res, open(A.out, "w"), indent=1)
print("BENCH_DONE")
