"""Slot-based, static-state, CUDA-graphed one-frame Mimi decoder.

Arithmetic mirrors serve/codec.py::StreamingMimi (the verified reference): causal
convolution history, transposed-convolution overlap tails (bias removed from the
stored tail), and sliding-window attention with absolute RoPE positions. The
difference is where state lives: every stateful layer owns ONE static tensor with
a row per slot (row ``max_slots`` is a dummy used to pad a batch to its bucket),
and the decoder transformer runs against static per-slot KV ring buffers, so one
decode is a single CUDA-graph replay with no per-seat Python work.

No upstream module or parameter is modified; HF sub-modules are called for their
weights. Global side effect: warmup() sets torch.backends.cudnn.benchmark=False so
convolution algorithm choice is settled before capture. Matmul/cuDNN TF32 flags
are read at capture time and baked into the graphs.

Sliding-window semantics (HF create_sliding_window_causal_mask, reproduced by
StreamingMimi keeping sliding_window-1 cached tokens plus the new ones): a query
at absolute position p sees keys with  p - sliding_window < key_pos <= p.
"""
import torch
import torch.nn.functional as F
from transformers.models.mimi.modeling_mimi import (MimiConv1d, MimiConvTranspose1d, MimiResnetBlock,
                                                    apply_rotary_pos_emb, repeat_kv)

NEVER = -(1 << 40)  # key_pos of an empty ring entry; always outside every window


class _Bucket:
    __slots__ = ('graph', 'slots', 'codes', 'out', 'pinned', 'events', 'turn')


class StaticMimi:
    def __init__(self, mimi, max_slots, batch_buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96), ring=256):
        cfg = mimi.config
        if not cfg.use_causal_conv or cfg.pad_mode != 'constant' or cfg.trim_right_ratio != 1:
            raise ValueError('Static decoder requires causal constant padding and trim_right_ratio=1')
        if mimi.upsample is None:
            raise ValueError('Static decoder expects the Mimi upsample layer')
        self.m = mimi
        self.max_slots = int(max_slots)
        self.dummy = self.max_slots
        self.batch_buckets = tuple(sorted(set(int(b) for b in batch_buckets)))
        self.window = int(cfg.sliding_window)
        self.frame_tokens = int(mimi.upsample.conv.stride[0])
        self.ring = int(ring)
        if self.ring < self.window + self.frame_tokens:
            raise ValueError('ring must hold sliding_window + frame_tokens entries')
        parameter = next(mimi.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.codebooks = int(cfg.num_quantizers)
        with torch.inference_mode():
            # Convolution history / overlap tails: one flat row per slot, one view per stateful layer.
            self._spec = {}
            self._plan(mimi.upsample, 'upsample')
            for i, layer in enumerate(mimi.decoder.layers):
                self._plan(layer, f'decoder.{i}')
            total = sum(c * p for c, p in self._spec.values())
            rows = self.max_slots + 1
            self.conv_flat = torch.zeros(rows, total, device=self.device, dtype=self.dtype)
            self.state, offset = {}, 0
            for key, (channels, pad) in self._spec.items():
                self.state[key] = self.conv_flat[:, offset:offset + channels * pad].view(rows, channels, pad)
                offset += channels * pad
            layers = mimi.decoder_transformer.layers
            attn = layers[0].self_attn
            self.kv_heads, self.head_dim, self.kv_groups = attn.num_key_value_heads, attn.head_dim, attn.num_key_value_groups
            # [layer, k/v, slot, heads, ring, head_dim]; keys are stored post-RoPE.
            self.kv = torch.zeros(len(layers), 2, rows, self.kv_heads, self.ring, self.head_dim,
                                  device=self.device, dtype=self.dtype)
            self.key_pos = torch.full((rows, self.ring), NEVER, device=self.device, dtype=torch.long)
            self.tokens = torch.zeros(rows, device=self.device, dtype=torch.long)
            self._steps = torch.arange(self.frame_tokens, device=self.device)
            rotary = attn.rotary_emb
            self._inv_freq = rotary.inv_freq.detach().float().clone()
            self._rope_scale = float(rotary.attention_scaling)
        self._pool = None
        self._buckets = {}
        self.samples_per_frame = None

    # ---- state layout -------------------------------------------------------------------------
    def _plan(self, module, key):
        if isinstance(module, MimiConv1d):
            if module.conv.stride != (1,):
                raise ValueError('Only stride-one decoder convolutions are supported')
            padding = (module.conv.kernel_size[0] - 1) * module.conv.dilation[0]
            if padding:
                self._spec[key] = (module.conv.in_channels, padding)
        elif isinstance(module, MimiConvTranspose1d):
            if module.padding_left != 0 or module.conv.dilation != (1,):
                raise ValueError('Unsupported transpose-convolution geometry')
            if module.padding_right:
                self._spec[key] = (module.conv.out_channels, module.padding_right)
        elif isinstance(module, MimiResnetBlock):
            self._plan(module.shortcut, key + '.shortcut')
            for i, child in enumerate(module.block):
                self._plan(child, key + f'.block.{i}')

    def state_bytes_per_slot(self):
        return {'conv': self.conv_flat[0].numel() * self.conv_flat.element_size(),
                'attention_kv': self.kv[:, :, 0].numel() * self.kv.element_size(),
                'attention_index': self.key_pos[0].numel() * 8 + 8}

    # ---- graph body ---------------------------------------------------------------------------
    def _layer(self, module, x, slots, key):
        if isinstance(module, MimiConv1d):
            if key in self.state:
                state = self.state[key]
                joined = torch.cat((state.index_select(0, slots), x), dim=-1)
                state.index_copy_(0, slots, joined[..., -state.shape[-1]:])
                x = joined
            return module.conv(x)
        if isinstance(module, MimiConvTranspose1d):
            y = module.conv(x)
            produced = x.shape[-1] * module.conv.stride[0]
            if key in self.state:
                state = self.state[key]
                overlap = state.shape[-1]
                if overlap > produced:
                    raise ValueError('Chunk too short for transpose-convolution overlap')
                previous = state.index_select(0, slots)
                tail = y[..., produced:]
                if module.conv.bias is not None:
                    tail = tail - module.conv.bias[None, :, None]
                state.index_copy_(0, slots, tail)
                y[..., :overlap] += previous          # disjoint from the tail because overlap <= produced
            return y[..., :produced]
        if isinstance(module, MimiResnetBlock):
            shortcut = self._layer(module.shortcut, x, slots, key + '.shortcut')
            for i, child in enumerate(module.block):
                x = self._layer(child, x, slots, key + f'.block.{i}')
            return shortcut + x
        return module(x)

    @staticmethod
    def _rvq(rvq, codes):
        total = None
        for i in range(codes.shape[1]):
            e = F.embedding(codes[:, i], rvq.layers[i].codebook.embed).permute(0, 2, 1)
            total = e if total is None else total + e
        return rvq.output_proj(total) if rvq.output_proj is not None else total

    def _quantizer(self, codes):
        # Same arithmetic as MimiSplitResidualVectorQuantizer.decode, without its in-graph torch.tensor(0.0).
        q = self.m.quantizer
        n = q.num_semantic_quantizers
        out = self._rvq(q.semantic_residual_vector_quantizer, codes[:, :n])
        if codes.shape[1] > n:
            out = out + self._rvq(q.acoustic_residual_vector_quantizer, codes[:, n:])
        return out

    def _transformer(self, h, slots):
        batch, count, _ = h.shape
        start = self.tokens.index_select(0, slots)
        positions = start[:, None] + self._steps[None]                      # [B,T] absolute
        ring_index = positions % self.ring
        rows = slots[:, None]
        self.key_pos.index_put_((rows, ring_index), positions)
        key_pos = self.key_pos.index_select(0, slots)[:, None, :]            # [B,1,R]
        query_pos = positions[:, :, None]                                    # [B,T,1]
        mask = ((key_pos <= query_pos) & (key_pos > query_pos - self.window))[:, None]   # [B,1,T,R]
        freqs = positions[:, :, None].float() * self._inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self._rope_scale).to(h.dtype)
        sin = (emb.sin() * self._rope_scale).to(h.dtype)
        for li, layer in enumerate(self.m.decoder_transformer.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(batch, count, attn.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            keys, values = self.kv[li, 0], self.kv[li, 1]
            # Advanced indices on dims 0 and 2 put the indexed dims first: value layout [B,T,heads,dim].
            keys[rows, :, ring_index] = k.transpose(1, 2)
            values[rows, :, ring_index] = v.transpose(1, 2)
            k_all = repeat_kv(keys.index_select(0, slots), self.kv_groups)
            v_all = repeat_kv(values.index_select(0, slots), self.kv_groups)
            a = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
            a = attn.o_proj(a.transpose(1, 2).reshape(batch, count, -1))
            h = h + layer.self_attn_layer_scale(a)
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        self.tokens.index_copy_(0, slots, start + count)
        self.tokens[self.dummy].zero_()
        return h

    def _forward(self, slots, codes):
        x = self._quantizer(codes)
        x = self._layer(self.m.upsample, x, slots, 'upsample')
        x = self._transformer(x.transpose(1, 2), slots).transpose(1, 2)
        for i, layer in enumerate(self.m.decoder.layers):
            x = self._layer(layer, x, slots, f'decoder.{i}')
        return x

    # ---- capture ------------------------------------------------------------------------------
    def _capture(self, size):
        bucket = _Bucket()
        bucket.slots = torch.full((size,), self.dummy, device=self.device, dtype=torch.long)
        bucket.codes = torch.zeros(size, self.codebooks, 1, device=self.device, dtype=torch.long)
        bucket.pinned = [torch.full((size,), self.dummy, dtype=torch.long).pin_memory() for _ in range(8)]
        bucket.events = [None] * len(bucket.pinned)
        bucket.turn = 0
        # Warmup and capture touch only the dummy slot, so live seats are never disturbed.
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward(bucket.slots, bucket.codes)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        bucket.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bucket.graph, pool=self._pool):
            bucket.out = self._forward(bucket.slots, bucket.codes)
        if self.samples_per_frame is None:
            self.samples_per_frame = bucket.out.shape[-1]
        self._reset_row(self.dummy)
        self._buckets[size] = bucket
        return bucket

    @torch.inference_mode()
    def warmup(self, batch_buckets=None):
        torch.backends.cudnn.benchmark = False
        sizes = self.batch_buckets if batch_buckets is None else tuple(sorted(set(int(b) for b in batch_buckets)))
        for size in sorted(sizes, reverse=True):      # largest first so the shared pool is sized once
            if size not in self._buckets:
                self._capture(size)
        self.batch_buckets = tuple(sorted(set(self.batch_buckets) | set(sizes)))

    # ---- public -------------------------------------------------------------------------------
    def _reset_row(self, row):
        self.conv_flat[row].zero_()
        self.key_pos[row].fill_(NEVER)
        self.tokens[row].zero_()

    @torch.inference_mode()
    def reset(self, slot, zero_kv=False):
        """Three fill kernels, no host sync. KV rows are unreachable once key_pos is NEVER
        (masked weights are exactly zero); zero_kv=True additionally clears them."""
        if not 0 <= slot < self.max_slots:
            raise IndexError(f'slot {slot} outside [0,{self.max_slots})')
        self._reset_row(slot)
        if zero_kv:
            self.kv[:, :, slot].zero_()

    @torch.inference_mode()
    def decode(self, slots, codes):
        """slots: distinct ints in [0,max_slots) (or a CUDA long tensor of them); codes: [B,K] long CUDA,
        one frame per seat. Returns an owned float32 [B, samples_per_frame] tensor."""
        count = len(slots)
        if codes.shape[0] != count or codes.dim() != 2 or codes.shape[1] != self.codebooks:
            raise ValueError(f'codes must be [{count},{self.codebooks}], got {tuple(codes.shape)}')
        size = next((b for b in self.batch_buckets if b >= count), None)
        if size is None:
            raise ValueError(f'batch {count} exceeds largest bucket {self.batch_buckets[-1]}')
        bucket = self._buckets.get(size) or self._capture(size)
        if torch.is_tensor(slots):
            bucket.slots[:count].copy_(slots)
            bucket.slots[count:].fill_(self.dummy)
        else:
            if len(set(slots)) != count or min(slots) < 0 or max(slots) >= self.max_slots:
                raise ValueError('slots must be distinct and inside [0,max_slots)')
            # Rotate pinned staging rows so an async H2D copy is never overwritten before it runs.
            turn = bucket.turn
            bucket.turn = (turn + 1) % len(bucket.pinned)
            if bucket.events[turn] is not None:
                bucket.events[turn].synchronize()
            staging = bucket.pinned[turn]
            staging[:count] = torch.tensor(slots, dtype=torch.long)
            staging[count:] = self.dummy
            bucket.slots.copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            bucket.events[turn] = event
        bucket.codes[:count, :, 0].copy_(codes)
        bucket.graph.replay()
        return bucket.out[:count, 0].float().clone() if bucket.out.dtype != torch.float32 else bucket.out[:count, 0].clone()
