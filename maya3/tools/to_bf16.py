"""Cast a flat Maya3 model.pt to bf16 (half the size). Lossless for inference: load_model_for_eval(dtype=bf16) casts every
tensor to bf16 anyway (the engine then upcasts the depth stack from those bf16 values), so the engine sees bit-identical
weights. Usage: to_bf16.py <model.pt> <out model.pt>"""
import sys, torch
src, out = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
payload = sd.get("model", sd) if isinstance(sd, dict) else sd
n, memo = 0, {}
for k, v in list(payload.items()):
    if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16:
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
        if key not in memo:          # tied tensors (lm_head = embed_tokens) stay shared
            memo[key] = v.to(torch.bfloat16); n += 1
        payload[k] = memo[key]
torch.save(sd, out)
print(f"cast {n} tensors -> {out}")
