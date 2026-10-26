"""GPU: StaticBackbone fused paths (fuse_proj, fuse_rope) vs the reference unfused path on identical inputs.
Usage: test_backbone_fused.py <model.pt>"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maya3.engine import Engine, EngineConfig
from maya3.bench import common as C
from serve.fast_backbone import StaticBackbone
E = Engine(sys.argv[1], C.BACKBONE, C.MIMI, EngineConfig(max_seats=8, capacity=1024))
bb = E.model.backbone
torch.manual_seed(0)
with torch.inference_mode():
    mk = lambda **kw: StaticBackbone(bb, 8, capacity=1024, batch_buckets=(1, 2, 4, 8, 64, 384), ctx_buckets=(512, 1024), attention="triton", **kw)
    ref, fp, fr, solo = mk(), mk(fuse_proj=True), mk(fuse_rope=True), mk()
    slots = [0, 3, 5]
    pre = torch.randn(1, 200, bb.d_model, device="cuda", dtype=torch.bfloat16) * 0.05
    for S in (ref, fp, fr, solo):
        for s in slots:
            S.load_prefix(s, [(torch.zeros(1, S.kv_heads, 0, S.head_dim, device="cuda", dtype=torch.bfloat16),) * 2 for _ in S.layers], 0)
    outs = {n: [] for n in ("ref", "proj", "rope", "solo")}
    # inline prefill of 200 rows for slot 0 + 3 + 5 (different lengths), then 20 decode steps
    for n, S in (("ref", ref), ("proj", fp), ("rope", fr)):
        _, last = S.step_mixed([], None, [(0, pre[0]), (3, pre[0, :120]), (5, pre[0, :37])])
        outs[n].append(torch.stack(last))
        torch.manual_seed(1)
        for t in range(20):
            x = torch.randn(3, bb.d_model, device="cuda", dtype=torch.bfloat16) * 0.05
            outs[n].append(S.step(slots, x))
    # noise floor: the SAME unfused path, but every slot prefilled and decoded alone (different batch composition)
    last = [solo.step_mixed([], None, [(sl, t)])[1][0] for sl, t in ((0, pre[0]), (3, pre[0, :120]), (5, pre[0, :37]))]
    outs["solo"].append(torch.stack(last)); torch.manual_seed(1)
    for t in range(20):
        x = torch.randn(3, bb.d_model, device="cuda", dtype=torch.bfloat16) * 0.05
        outs["solo"].append(torch.cat([solo.step([sl], x[j:j + 1]) for j, sl in enumerate(slots)]))
    floor = None
    for n in ("solo", "proj", "rope"):
        worst = 0
        if n == "solo":
            print("solo per-step rel:", [round(((a.float() - b.float()).norm() / a.float().norm()).item(), 4) for a, b in zip(outs["ref"], outs[n])])
            print("solo per-row rel step0:", [round(((a.float() - b.float()).norm() / a.float().norm()).item(), 4) for a, b in zip(outs["ref"][0], outs[n][0])])
        for a, b in zip(outs["ref"], outs[n]):
            cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), -1).min().item()
            rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
            worst = max(worst, rel)
        print(f"{n}: worst rel L2 {worst:.2e}, last min cosine {cos:.6f}")
        if n == "solo": floor = worst
        else: assert worst <= max(2 * floor, 1e-3), (n, worst, floor)
print("BACKBONE_FUSED_OK")
