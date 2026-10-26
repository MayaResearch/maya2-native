"""GPU: engine loaded with the backbone built from config (no backbone safetensors read) == normal load: identical
state dict, identical RoPE buffers and identical prompt hidden states. Usage: test_backbone_from_config.py <model.pt>"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maya3.engine import Engine, EngineConfig
from maya3.bench import common as C
def load(flag):
    os.environ["MAYA3_BACKBONE_FROM_CONFIG"] = flag
    t = time.time(); E = Engine(sys.argv[1], C.BACKBONE, C.MIMI, EngineConfig(max_seats=2, capacity=512)); return E, time.time() - t
A, ta = load("0"); B, tb = load("1")
sa, sb = A.model.state_dict(), B.model.state_dict()
diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
ra = [b for n, b in A.model.backbone.lm.named_buffers()]; rb = [b for n, b in B.model.backbone.lm.named_buffers()]
print("load_s normal / from_config:", round(ta, 1), round(tb, 1), "| state diffs:", len(diff), diff[:3], "| buffers equal:", all(torch.equal(x, y) for x, y in zip(ra, rb)))
with torch.inference_mode():
    v = A.registry.get("plain"); A.prepare_voice(v)
    ex = A.builder.example(A.packer, v, "Hello there, how are you?", "en")
    p, _ = A.S.prompt_from_example(ex, A.model, "cuda", torch.bfloat16, 32)
    ha, _ = A.model.backbone(p.to(torch.bfloat16)); hb, _ = B.model.backbone(p.to(torch.bfloat16))
    print("hidden equal:", torch.equal(ha, hb))
    assert not diff and torch.equal(ha, hb)
print("FROM_CONFIG_OK")
