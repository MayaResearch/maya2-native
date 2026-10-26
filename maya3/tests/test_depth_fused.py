"""GPU: FastDepth fused body vs reference body (greedy code agreement vs the fp16 noise floor = fp16 vs fp32 reference),
and the fused top-k sampler's distribution vs the exact top-k softmax.  Usage: test_depth_fused.py <model.pt>"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maya3.engine import Engine, EngineConfig
from maya3.bench import common as C
from serve.fast_depth import FastDepth
from serve.fast_depth_kernels import topk_sample
E = Engine(sys.argv[1], C.BACKBONE, C.MIMI, EngineConfig(max_seats=4, capacity=1024, free_reference_depth=False))
torch.manual_seed(0)
with torch.inference_mode():
    # sampler distribution: 200k draws from one fixed logit row
    lg = torch.randn(1, 2048, device="cuda") * 3
    seed = torch.zeros(1, device="cuda", dtype=torch.int32)
    draws = []
    for i in range(4000):
        draws.append(topk_sample(lg.expand(50, -1).contiguous(), seed, 3, 0.7, 64)); seed += 1
    d = torch.cat(draws)
    emp = torch.bincount(d, minlength=2048).float() / d.numel()
    l = lg[0] / 0.7; kth = l.topk(64).values[-1]; p = torch.where(l >= kth, l, torch.tensor(-float("inf"), device="cuda")).softmax(-1)
    tv = 0.5 * (emp - p).abs().sum().item()
    print(f"sampler total-variation vs exact top-k softmax: {tv:.4f} (200k draws; sampling noise ~0.01)")
    assert tv < 0.03 and emp[p == 0].sum() == 0
    # greedy agreement on real hidden states from a prefill of real prompts
    v = E.registry.get("plain"); E.prepare_voice(v)
    ex = E.builder.example(E.packer, v, "नमस्ते, आज मौसम बहुत अच्छा है और हम बाहर घूमने जा रहे हैं।", "hi")
    prompt, n = E.S.prompt_from_example(ex, E.model, "cuda", torch.bfloat16, 32)
    hs, _ = E.model.backbone(prompt.to(torch.bfloat16)); h = hs[0, -16:].contiguous()   # 16 real-ish states
    B = h.shape[0]; slots = list(range(B))
    mk = lambda dt, fused: FastDepth(E.model, B, dtype=dt, batch_buckets=(B,), greedy=True, fused=fused)
    ref32, ref16, fus16 = mk(torch.float32, False), mk(torch.float16, False), mk(torch.float16, True)
    for D_ in (ref32, ref16, fus16):
        for s_ in slots: D_.set_speaker(s_, v.mem, v.rmask)
    cb0 = E.model.backbone.cb0_logits(h).float()[:, :2048].argmax(-1)
    out = {k: D_._fn(torch.tensor(slots, device="cuda"), h.to(D_.dtype), cb0)[0] for k, D_ in
           (("fp32", ref32), ("fp16", ref16), ("fused", fus16))}
    agree = lambda a, b: (out[a] == out[b]).float().mean().item()
    print(f"greedy code agreement: fp16 vs fp32 {agree('fp16','fp32'):.3f} | fused vs fp32 {agree('fused','fp32'):.3f} | fused vs fp16 {agree('fused','fp16'):.3f}")
    assert agree("fused", "fp32") >= agree("fp16", "fp32") - 0.05
print("DEPTH_FUSED_OK")
