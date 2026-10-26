"""Whole-frame depth decoder in one CUDA graph per batch bucket (bf16 by default).

One replay = 31 depth levels + in-graph sampling + the backbone feedback
embedding. Differences from WholeDepthGraph/IncrementalDepth, all of them the
same real-valued function (kernel order differs, so not bit-exact):

* a private lower-precision deep copy of `model.depth` (the original is untouched);
* speaker cross-attention hoisted to a per-slot constant table (single unmasked
  reference => softmax is exactly one => output is o(v), independent of the query),
  gathered inside the graph from a static `slots_idx`, so admission needs no recapture;
* ONE engine-wide CUDA generator registered on every graph and one batched
  `exponential_` per level (no per-seat Philox state copies);
* prefix embeddings computed once per position (position k depends on slot k alone)
  from a fused `code_up(code_lo)+level_emb` table, written into a static buffer;
* preallocated within-group KV (one strided write per layer instead of two cats);
* `is_causal=True` instead of a rebuilt boolean mask; no `_assert_async`;
* last trunk layer evaluates only the final position (its other outputs are never
  read: later levels of a shared group reuse this layer's K/V, which come from its input).
"""
import copy
import math

import torch
import torch.nn.functional as F

from tts.model.depth_v2 import GROUP_OF_LEVEL, LAYERS, D, H

HEAD = D // H
N_LEVELS = 32


class FastDepth:
    def __init__(self, model, max_slots, dtype=torch.bfloat16, temp=0.7, top_k=64,
                 batch_buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96), seed=1234, greedy=False, compile=False, fused=False):
        if model.depth.n_q != N_LEVELS:
            raise ValueError('FastDepth expects the 32-codebook depth stack')
        self.max_slots = int(max_slots)
        self.dtype = dtype
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.greedy = bool(greedy)  # debug: argmax of the same fp32 probabilities
        self.batch_buckets = tuple(sorted(set(int(b) for b in batch_buckets)))
        with torch.inference_mode():
            d = copy.deepcopy(model.depth).to(dtype).eval().requires_grad_(False)
            self.depth = d
            self.device = d.in_proj.weight.device
            self.n_codes = d.n_codes
            # position k+1 of the prefix = code_up(code_lo[cb_k + k*n_codes]) + level_emb[k+1]
            table = d.code_up(d.code_lo.weight).view(N_LEVELS - 1, self.n_codes, D)
            self.embed = (table + d.level_emb.weight[1:N_LEVELS, None]).contiguous()
            self.level0 = d.level_emb.weight[0].clone()
            self.audio_in = model.audio_in  # shared, read-only (engine dtype, bf16 in serving)
            self.feedback_offsets = torch.arange(N_LEVELS, device=self.device) * model.audio_in.n_codes
            self.feedback_scale = math.sqrt(N_LEVELS)
            # row max_slots is the padding slot (zeros)
            self.cross_const = torch.zeros(self.max_slots + 1, LAYERS, D, device=self.device, dtype=dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.pool = None
        self.graphs = {}
        # optional: Inductor-fused frame body (fixed shape per bucket), still replayed as one CUDA graph
        # fused=True: Triton add+LayerNorm and one-kernel top-k sampling (serve/fast_depth_kernels.py); RNG = Philox keyed by
        # a device seed the graph increments each replay (same distribution, different random stream than torch's)
        self.fused = bool(fused)
        self.seed_t = torch.full((1,), int(seed) % (2 ** 31 - 1), device=self.device, dtype=torch.int32)
        # per-slot sampling streams (fused path): seed of the slot's request and its frame counter (incremented in-graph)
        self.row_seed = torch.zeros(self.max_slots + 1, device=self.device, dtype=torch.int32)
        self.row_ctr = torch.zeros(self.max_slots + 1, device=self.device, dtype=torch.int32)
        self.per_row = True
        body = self._compute_fused if self.fused else self._compute
        self._fn = torch.compile(body, dynamic=False) if compile else body

    @torch.inference_mode()
    def set_speaker(self, slot, mem, ref_mask):
        if mem.ndim != 3 or mem.shape[0] != 1 or mem.shape[1] != 1 or tuple(ref_mask.shape) != (1, 1):
            raise ValueError(f'FastDepth serves exactly one reference per slot; got mem {tuple(mem.shape)}')
        if not 0 <= slot < self.max_slots:
            raise IndexError(f'slot {slot} outside 0..{self.max_slots - 1}')
        mem = mem.to(self.device, self.dtype)
        keep = ref_mask.to(self.device)[:, :, None]
        for layer, block in enumerate(self.depth.cross):
            _, v = block.kv(mem).chunk(2, -1)
            self.cross_const[slot, layer].copy_(block.o(v * keep)[0, 0])

    def _compute(self, slots_idx, h, cb0):
        d = self.depth
        B = h.shape[0]
        cross = self.cross_const[slots_idx]  # [B, LAYERS, D]
        cross = [cross[:, layer, None] for layer in range(LAYERS)]
        prefix = torch.empty(B, N_LEVELS, D, device=h.device, dtype=self.dtype)
        prefix[:, 0] = d.in_proj(h) + self.level0
        prefix[:, 1] = F.embedding(cb0, self.embed[0])
        # [layer][B, position, k|v, head, dim]: one strided write per layer and level
        kv = [torch.empty(B, N_LEVELS, 2, H, HEAD, device=h.device, dtype=self.dtype) for _ in range(LAYERS)]
        codes = [cb0]
        previous = None
        for level in range(1, N_LEVELS):
            group = GROUP_OF_LEVEL[level]
            n = level + 1
            incremental = group == previous
            reused_later = level + 1 < N_LEVELS and GROUP_OF_LEVEL[level + 1] == group
            x = prefix[:, level:n] if incremental else prefix[:, :n]
            for layer, block in enumerate(d.groups[group]):
                T = x.shape[1]
                qkv = block.qkv(block.n1(x)).view(B, T, 3, H, HEAD)
                q = qkv[:, :, 0].transpose(1, 2)
                if incremental:
                    kv[layer][:, level:n].copy_(qkv[:, :, 1:])
                    k = kv[layer][:, :n, 0].transpose(1, 2)
                    v = kv[layer][:, :n, 1].transpose(1, 2)
                    a = F.scaled_dot_product_attention(q, k, v)
                else:
                    if reused_later:
                        kv[layer][:, :n].copy_(qkv[:, :, 1:])
                    k = qkv[:, :, 1].transpose(1, 2)
                    v = qkv[:, :, 2].transpose(1, 2)
                    if layer == LAYERS - 1:
                        # only the final position is read after the last layer; it sees every key
                        a = F.scaled_dot_product_attention(q[:, :, -1:], k, v)
                        x = x[:, -1:]
                    else:
                        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + block.o(a.transpose(1, 2).reshape(B, x.shape[1], D))
                x = x + block.f2(F.gelu(block.f1(block.n2(x))))
                x = x + cross[layer]
            logits = d.heads[level - 1](d.norm(x[:, -1])).float() / self.temp
            kth = logits.topk(self.top_k, -1).values[:, -1:]
            probs = logits.masked_fill(logits < kth, -float('inf')).softmax(-1)
            if self.greedy:
                code = probs.argmax(-1)
            else:
                race = torch.empty_like(probs).exponential_(1, generator=self.generator)
                code = (probs / race).argmax(-1)
            codes.append(code)
            if n < N_LEVELS:
                prefix[:, n] = F.embedding(code, self.embed[level])
            previous = group
        codes = torch.stack(codes, -1)
        # == model.audio_in.frames(codes, 32) without its host-synchronizing range check
        e = self.audio_in.table(codes + self.feedback_offsets)
        feedback = self.audio_in.norm(e.sum(1) / self.feedback_scale)
        return codes, feedback

    def _compute_fused(self, slots_idx, h, cb0):
        from serve.fast_depth_kernels import add_ln, topk_sample
        d = self.depth
        B = h.shape[0]
        cross = self.cross_const[slots_idx]  # [B, LAYERS, D]
        cross = [cross[:, layer, None] for layer in range(LAYERS)]
        prefix = torch.empty(B, N_LEVELS, D, device=h.device, dtype=self.dtype)
        prefix[:, 0] = d.in_proj(h) + self.level0
        prefix[:, 1] = F.embedding(cb0, self.embed[0])
        kv = [torch.empty(B, N_LEVELS, 2, H, HEAD, device=h.device, dtype=self.dtype) for _ in range(LAYERS)]
        codes = [cb0]
        previous = None
        row_seed = self.row_seed[slots_idx]
        row_ctr = self.row_ctr[slots_idx]
        for level in range(1, N_LEVELS):
            group = GROUP_OF_LEVEL[level]
            n = level + 1
            incremental = group == previous
            reused_later = level + 1 < N_LEVELS and GROUP_OF_LEVEL[level + 1] == group
            x = prefix[:, level:n] if incremental else prefix[:, :n]
            blocks = d.groups[group]
            y = blocks[0].n1(x)
            for layer, block in enumerate(blocks):
                T = x.shape[1]
                qkv = block.qkv(y).view(B, T, 3, H, HEAD)
                q = qkv[:, :, 0].transpose(1, 2)
                if incremental:
                    kv[layer][:, level:n].copy_(qkv[:, :, 1:])
                    k = kv[layer][:, :n, 0].transpose(1, 2)
                    v = kv[layer][:, :n, 1].transpose(1, 2)
                    a = F.scaled_dot_product_attention(q, k, v)
                else:
                    if reused_later:
                        kv[layer][:, :n].copy_(qkv[:, :, 1:])
                    k = qkv[:, :, 1].transpose(1, 2)
                    v = qkv[:, :, 2].transpose(1, 2)
                    if layer == LAYERS - 1:
                        a = F.scaled_dot_product_attention(q[:, :, -1:], k, v)
                        x = x[:, -1:]
                    else:
                        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                T = x.shape[1]
                x, y2 = add_ln(x, block.o(a.transpose(1, 2).reshape(B, T, D)), block.n2)
                f = block.f2(F.gelu(block.f1(y2)))
                nxt = blocks[layer + 1].n1 if layer + 1 < LAYERS else d.norm
                x, y = add_ln(x, f, nxt, cross[layer])
            # y = d.norm(x) of the final position (x is [B,1,D] after the last layer)
            if self.per_row:
                code = topk_sample(d.heads[level - 1](y[:, -1]), row_seed, level, self.temp, self.top_k, self.greedy, ctr_t=row_ctr)
            else:
                code = topk_sample(d.heads[level - 1](y[:, -1]), self.seed_t, level, self.temp, self.top_k, self.greedy)
            codes.append(code)
            if n < N_LEVELS:
                prefix[:, n] = F.embedding(code, self.embed[level])
            previous = group
        self.seed_t.add_(1)
        self.row_ctr.index_add_(0, slots_idx, torch.ones_like(row_ctr))
        codes = torch.stack(codes, -1)
        e = self.audio_in.table(codes + self.feedback_offsets)
        feedback = self.audio_in.norm(e.sum(1) / self.feedback_scale)
        return codes, feedback

    @torch.inference_mode()
    def _capture(self, bucket):
        if bucket in self.graphs:
            return self.graphs[bucket]
        s = {'slots_idx': torch.full((bucket,), self.max_slots, device=self.device, dtype=torch.long),
             'h': torch.zeros(bucket, self.depth.in_proj.in_features, device=self.device, dtype=self.dtype),
             'cb0': torch.zeros(bucket, device=self.device, dtype=torch.long),
             'last_slots': None}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._fn(s['slots_idx'], s['h'], s['cb0'])
        torch.cuda.current_stream().wait_stream(stream)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self.generator)
        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
            s['codes'], s['feedback'] = self._fn(s['slots_idx'], s['h'], s['cb0'])
        s['graph'] = graph
        self.graphs[bucket] = s
        return s

    def warmup(self, batch_buckets=None):
        # largest first so smaller captures reuse the shared pool's blocks
        for bucket in sorted(set(batch_buckets or self.batch_buckets), reverse=True):
            self._capture(int(bucket))

    def bucket_for(self, batch):
        for bucket in self.batch_buckets:
            if bucket >= batch:
                return bucket
        raise ValueError(f'batch {batch} exceeds the largest bucket {self.batch_buckets[-1]}')

    @torch.inference_mode()
    def __call__(self, slots, h, cb0):
        B = len(slots)
        if h.shape[0] != B or cb0.shape[0] != B:
            raise ValueError('slots, h and cb0 must have the same batch size')
        s = self._capture(self.bucket_for(B))
        key = tuple(slots)
        if key != s['last_slots']:
            if any(not 0 <= slot < self.max_slots for slot in key):
                raise IndexError(f'slot outside 0..{self.max_slots - 1}')
            # pageable source: staged before copy_ returns, so a later call cannot race it
            padded = list(key) + [self.max_slots] * (s['slots_idx'].shape[0] - B)
            s['slots_idx'].copy_(torch.tensor(padded, dtype=torch.long))
            s['last_slots'] = key
        s['h'][:B].copy_(h)
        s['cb0'][:B].copy_(cb0)
        s['graph'].replay()
        return s['codes'][:B].clone(), s['feedback'][:B].clone()
