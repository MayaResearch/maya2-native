"""Static-KV, CUDA-graph single-token decode for the Qwen3 backbone with per-seat positions.

KV lives in [max_slots+1, kv_heads, capacity, head_dim] tensors per layer (last row is a
dummy slot used to pad a batch up to its bucket; its length stays 0).  One CUDA graph per
(batch bucket, context bucket); all graphs share one memory pool.  HF module weights are
used in place (no copies).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'serve/scratch/triton'))
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton=None


def _rotate_half(x):
    half=x.shape[-1]//2
    return torch.cat((-x[...,half:],x[...,:half]),-1)


if triton is not None:
    @triton.jit
    def _decode_attn(Q,K,V,Out,Slots,Pos,scale,
                     sq_b,sq_h,sk_s,sk_h,sk_t,so_b,so_h,
                     G:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr,NBLOCKS:tl.constexpr):
        """One program per (row, q head).  Reads the static cache in place for keys 0..pos."""
        row=tl.program_id(0);head=tl.program_id(1)
        slot=tl.load(Slots+row);pos=tl.load(Pos+row)
        d=tl.arange(0,D)
        q=tl.load(Q+row*sq_b+head*sq_h+d).to(tl.float32)
        kbase=K+slot*sk_s+(head//G)*sk_h
        vbase=V+slot*sk_s+(head//G)*sk_h
        m=tl.full((),-float('inf'),tl.float32)
        den=tl.zeros((),tl.float32)
        acc=tl.zeros((D,),tl.float32)
        for blk in range(0,NBLOCKS):
            t=blk*BLOCK+tl.arange(0,BLOCK)
            valid=t<=pos
            k=tl.load(kbase+t[:,None]*sk_t+d[None,:],mask=valid[:,None],other=0.).to(tl.float32)
            s=tl.sum(k*q[None,:],1)*scale
            s=tl.where(valid,s,-float('inf'))
            bm=tl.maximum(m,tl.max(s,0))
            w=tl.exp(s-bm)
            v=tl.load(vbase+t[:,None]*sk_t+d[None,:],mask=valid[:,None],other=0.).to(tl.float32)
            r=tl.exp(m-bm)
            acc=acc*r+tl.sum(w[:,None]*v,0)
            den=den*r+tl.sum(w,0)
            m=bm
        tl.store(Out+row*so_b+head*so_h+d,(acc/den).to(Out.dtype.element_ty))


    @triton.jit
    def _qk_norm_rope_cache(QKV,Qout,K,V,Slots,Pos,COS,SIN,WQ,WK,eps,
                            s_qkv,s_qo_b,s_qo_h,sk_s,sk_h,sk_t,s_cos,
                            H:tl.constexpr,KVH:tl.constexpr,D:tl.constexpr):
        """Fused Qwen3 q_norm/k_norm (RMSNorm) + RoPE + K/V cache write for one (row, head). Program j < H: query head;
        j >= H: key head j-H (also copies its value head into the cache)."""
        row=tl.program_id(0);j=tl.program_id(1)
        half=tl.arange(0,D//2)
        pos=tl.load(Pos+row)
        c1=tl.load(COS+pos*s_cos+half).to(tl.float32);c2=tl.load(COS+pos*s_cos+D//2+half).to(tl.float32)
        n1s=tl.load(SIN+pos*s_cos+half).to(tl.float32);n2s=tl.load(SIN+pos*s_cos+D//2+half).to(tl.float32)
        if j<H:
            base=QKV+row*s_qkv+j*D;w1=tl.load(WQ+half).to(tl.float32);w2=tl.load(WQ+D//2+half).to(tl.float32)
        else:
            base=QKV+row*s_qkv+(H+j-H)*D;w1=tl.load(WK+half).to(tl.float32);w2=tl.load(WK+D//2+half).to(tl.float32)
        x1=tl.load(base+half).to(tl.float32);x2=tl.load(base+D//2+half).to(tl.float32)
        inv=1.0/tl.sqrt((tl.sum(x1*x1,0)+tl.sum(x2*x2,0))/D+eps)
        # HF: weight * (x*inv).to(bf16), in bf16
        y1=(x1*inv).to(tl.bfloat16).to(tl.float32)*w1;y2=(x2*inv).to(tl.bfloat16).to(tl.float32)*w2
        y1=y1.to(tl.bfloat16).to(tl.float32);y2=y2.to(tl.bfloat16).to(tl.float32)
        o1=y1*c1-y2*n1s;o2=y2*c2+y1*n2s
        if j<H:
            ob=Qout+row*s_qo_b+j*s_qo_h
            tl.store(ob+half,o1.to(Qout.dtype.element_ty));tl.store(ob+D//2+half,o2.to(Qout.dtype.element_ty))
        else:
            kh=j-H;slot=tl.load(Slots+row)
            kb=K+slot*sk_s+kh*sk_h+pos*sk_t
            tl.store(kb+half,o1.to(K.dtype.element_ty));tl.store(kb+D//2+half,o2.to(K.dtype.element_ty))
            vb=QKV+row*s_qkv+(H+KVH+kh)*D
            vv1=tl.load(vb+half);vv2=tl.load(vb+D//2+half)
            vcb=V+slot*sk_s+kh*sk_h+pos*sk_t
            tl.store(vcb+half,vv1);tl.store(vcb+D//2+half,vv2)


    @triton.jit
    def _add_rms(Hin,A,Hout,Y,W,eps,s_h,s_a,s_ho,s_y,N:tl.constexpr):
        """h' = h + a (bf16 rounding, as torch) ; y = W * (h' * rsqrt(mean(h'^2) + eps)).to(bf16)  (HF Qwen3RMSNorm)."""
        row=tl.program_id(0);c=tl.arange(0,N)
        hs=(tl.load(Hin+row*s_h+c).to(tl.float32)+tl.load(A+row*s_a+c).to(tl.float32)).to(tl.bfloat16)
        tl.store(Hout+row*s_ho+c,hs)
        x=hs.to(tl.float32)
        inv=1.0/tl.sqrt(tl.sum(x*x,0)/N+eps)
        y=(x*inv).to(tl.bfloat16).to(tl.float32)*tl.load(W+c).to(tl.float32)
        tl.store(Y+row*s_y+c,y.to(tl.bfloat16))


class _Graph:
    __slots__=('graph','slots','slots_cpu','x','real','real_cpu','out','last','offset','offset_cpu','offset_dirty','evt')


class StaticBackbone:
    def __init__(self,backbone,max_slots,capacity=1536,
                 batch_buckets=(1,2,4,8,16,24,32,48,64,80,96),ctx_buckets=(512,768,1024,1536),
                 attention='gather',fused_norm=False,use_graphs=True,fuse_proj=False,fuse_rope=False):
        """attention: 'gather' (index_select + matmul; the simple path) or 'triton' (in-place kernel)."""
        if attention not in ('gather','triton'):raise ValueError('attention must be gather or triton')
        if attention=='triton' and triton is None:raise RuntimeError('triton is not importable')
        with torch.inference_mode():
            self.backbone=backbone;self.lm=backbone.lm.model;cfg=backbone.lm.config
            self.layers=list(self.lm.layers);self.norm=self.lm.norm
            p=next(self.lm.parameters());self.dtype=p.dtype;self.device=p.device
            self.heads=cfg.num_attention_heads;self.kv_heads=cfg.num_key_value_heads
            self.head_dim=getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)
            self.groups=self.heads//self.kv_heads;self.hidden=cfg.hidden_size
            self.scaling=self.head_dim**-0.5;self.eps=cfg.rms_norm_eps
            self.max_slots=max_slots;self.capacity=capacity;self.dummy=max_slots
            self.attention=attention;self.fused_norm=fused_norm;self.use_graphs=use_graphs
            # fuse_proj: one GEMM for q|k|v and one for gate|up per layer; the HF modules' weights become views of the
            # concatenated tensors (no extra memory; the HF path still works)
            self.fuse_proj=fuse_proj or fuse_rope;self.fuse_rope=fuse_rope and attention=='triton';self.wqkv=[];self.wgu=[]
            if self.fuse_proj:
                for layer in self.layers:
                    a=layer.self_attn;m=layer.mlp
                    if a.q_proj.bias is not None or getattr(m.gate_proj,'bias',None) is not None:raise ValueError('fuse_proj expects bias-free projections')
                    w=torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]).contiguous()
                    nq,nk=a.q_proj.weight.shape[0],a.k_proj.weight.shape[0]
                    a.q_proj.weight.data=w[:nq];a.k_proj.weight.data=w[nq:nq+nk];a.v_proj.weight.data=w[nq+nk:]
                    self.wqkv.append(w)
                    g=torch.cat([m.gate_proj.weight,m.up_proj.weight]).contiguous();ng=m.gate_proj.weight.shape[0]
                    m.gate_proj.weight.data=g[:ng];m.up_proj.weight.data=g[ng:]
                    self.wgu.append(g)
            # Row buckets may exceed max_slots: inline prefill adds several rows for one slot.
            self.batch_buckets=tuple(sorted(set(batch_buckets)))
            if self.batch_buckets[-1]<max_slots:self.batch_buckets+=(max_slots,)
            self.ctx_buckets=tuple(sorted({min(c,capacity) for c in ctx_buckets}|{capacity}))
            shape=(max_slots+1,self.kv_heads,capacity,self.head_dim)
            self.k=[torch.zeros(shape,dtype=self.dtype,device=self.device) for _ in self.layers]
            self.v=[torch.zeros(shape,dtype=self.dtype,device=self.device) for _ in self.layers]
            self.lengths=torch.zeros(max_slots+1,dtype=torch.long,device=self.device)
            self.lengths_cpu=[0]*(max_slots+1)
            positions=torch.arange(capacity,device=self.device)[None]
            cos,sin=self.lm.rotary_emb(torch.zeros(1,1,self.hidden,dtype=self.dtype,device=self.device),positions)
            self.cos=cos[0].contiguous();self.sin=sin[0].contiguous()      # [capacity, head_dim], model dtype (as HF)
            self.graphs={};self.pool=None

    # ---------------------------------------------------------------- slots
    @torch.inference_mode()
    def load_slot(self,slot,hf_cache,length):
        if not 0<=slot<self.max_slots:raise IndexError(slot)
        if length>self.capacity:raise ValueError(f'length {length} exceeds capacity {self.capacity}')
        for i in range(len(self.layers)):
            layer=hf_cache.layers[i]
            self.k[i][slot,:,:length].copy_(layer.keys[0,:,:length])
            self.v[i][slot,:,:length].copy_(layer.values[0,:,:length])
        self.lengths[slot]=length;self.lengths_cpu[slot]=length

    @torch.inference_mode()
    def free(self,slot):
        self.lengths[slot]=0;self.lengths_cpu[slot]=0

    # ---------------------------------------------------------------- math
    def _rms(self,module,x):
        if self.fused_norm:return F.rms_norm(x,(x.shape[-1],),module.weight,self.eps)
        return module(x)

    def _forward(self,slots,x,real,ctx,offset=None):
        B=x.shape[0];H=self.heads;KV=self.kv_heads;D=self.head_dim;G=self.groups
        pos=self.lengths[slots]                                            # [B]
        if offset is not None:pos=pos+offset   # inline prefill: several consecutive rows of one slot
        cos=self.cos[pos][:,None];sin=self.sin[pos][:,None]                # [B,1,D]
        if self.attention=='gather':
            bias=torch.zeros(B,ctx,dtype=torch.float32,device=x.device).masked_fill_(
                torch.arange(ctx,device=x.device)[None]>pos[:,None],float('-inf')).view(B,1,1,ctx)
        h=x;pend=None
        for i,layer in enumerate(self.layers):
            attn=layer.self_attn
            if self.fuse_rope and pend is not None:
                h,y=self._add_norm(h,pend,layer.input_layernorm.weight);pend=None
            else:
                y=self._rms(layer.input_layernorm,h)
            kc=self.k[i];vc=self.v[i]
            if self.fuse_rope:
                qkv=F.linear(y,self.wqkv[i]).contiguous()
                q=torch.empty(B,H,D,dtype=qkv.dtype,device=qkv.device)
                _qk_norm_rope_cache[(B,H+KV)](qkv,q,kc,vc,slots,pos,self.cos,self.sin,attn.q_norm.weight,attn.k_norm.weight,
                    self.eps,qkv.stride(0),q.stride(0),q.stride(1),kc.stride(0),kc.stride(1),kc.stride(2),self.cos.stride(0),
                    H=H,KVH=KV,D=D)
                o=torch.empty_like(q)
                self._launch(q,kc,vc,o,slots,pos,ctx)
                h,y2=self._add_norm(h,attn.o_proj(o.view(B,H*D)),layer.post_attention_layernorm.weight)
                gu=F.linear(y2,self.wgu[i]);g_,u_=gu.chunk(2,-1)
                pend=layer.mlp.down_proj(F.silu(g_)*u_)
                continue
            if self.fuse_proj:
                qkv=F.linear(y,self.wqkv[i]);q_,k_,v_=qkv.split([H*D,KV*D,KV*D],-1)
                q=self._rms(attn.q_norm,q_.view(B,H,D));k=self._rms(attn.k_norm,k_.view(B,KV,D));v=v_.view(B,KV,D)
            else:
                q=self._rms(attn.q_norm,attn.q_proj(y).view(B,H,D))
                k=self._rms(attn.k_norm,attn.k_proj(y).view(B,KV,D))
                v=attn.v_proj(y).view(B,KV,D)
            q=q*cos+_rotate_half(q)*sin
            k=k*cos+_rotate_half(k)*sin
            kc=self.k[i];vc=self.v[i]
            kc[slots,:,pos]=k;vc[slots,:,pos]=v
            if self.attention=='gather':
                K=kc[:,:,:ctx].index_select(0,slots);V=vc[:,:,:ctx].index_select(0,slots)   # [B,KV,ctx,D]
                scores=torch.matmul(q.view(B,KV,G,D),K.transpose(-1,-2)).float()*self.scaling+bias
                o=torch.matmul(scores.softmax(-1).to(q.dtype),V).reshape(B,H*D)
            else:
                q=q.contiguous();o=torch.empty_like(q)
                self._launch(q,kc,vc,o,slots,pos,ctx)
                o=o.view(B,H*D)
            h=h+attn.o_proj(o)
            if self.fuse_proj:
                gu=F.linear(self._rms(layer.post_attention_layernorm,h),self.wgu[i]);g_,u_=gu.chunk(2,-1)
                h=h+layer.mlp.down_proj(F.silu(g_)*u_)
            else:
                h=h+layer.mlp(self._rms(layer.post_attention_layernorm,h))
        if pend is not None:
            h,out=self._add_norm(h,pend,self.norm.weight)
        else:
            out=self._rms(self.norm,h)
        self.lengths.index_add_(0,slots,real)
        return out

    def _add_norm(self,h,a,w):
        if h.dtype!=torch.bfloat16:raise ValueError('fused add+norm kernel is bf16-only')
        a=a.contiguous();ho=torch.empty_like(h);y=torch.empty_like(h)
        _add_rms[(h.shape[0],)](h,a,ho,y,w,self.eps,h.stride(0),a.stride(0),ho.stride(0),y.stride(0),N=h.shape[-1])
        return ho,y

    def _launch(self,q,kc,vc,o,slots,pos,ctx):
        block=256
        _decode_attn[(q.shape[0],self.heads)](q,kc,vc,o,slots,pos,self.scaling,
            q.stride(0),q.stride(1),kc.stride(0),kc.stride(1),kc.stride(2),o.stride(0),o.stride(1),
            G=self.groups,D=self.head_dim,BLOCK=block,NBLOCKS=(ctx+block-1)//block)
        return o

    # ---------------------------------------------------------------- graphs
    def _bucket(self,values,need,what):
        for b in values:
            if b>=need:return b
        raise ValueError(f'{what} {need} exceeds largest bucket {values[-1]}')

    @torch.inference_mode()
    def _capture(self,bb,cb):
        g=_Graph()
        g.slots=torch.full((bb,),self.dummy,dtype=torch.long,device=self.device)
        g.real=torch.zeros(bb,dtype=torch.long,device=self.device)
        g.offset=torch.zeros(bb,dtype=torch.long,device=self.device)
        g.offset_cpu=torch.zeros(bb,dtype=torch.long).pin_memory();g.offset_dirty=False
        g.x=torch.zeros(bb,self.hidden,dtype=self.dtype,device=self.device)
        g.slots_cpu=torch.full((bb,),self.dummy,dtype=torch.long).pin_memory()
        g.real_cpu=torch.zeros(bb,dtype=torch.long).pin_memory()
        g.last=None;g.graph=None;g.evt=None
        if self.use_graphs:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self._forward(g.slots,g.x,g.real,cb,g.offset)   # all-dummy rows: touches only the dummy slot
            torch.cuda.current_stream().wait_stream(stream)
            if self.pool is None:self.pool=torch.cuda.graph_pool_handle()
            g.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g.graph,pool=self.pool):
                g.out=self._forward(g.slots,g.x,g.real,cb,g.offset)
        self.graphs[bb,cb]=g
        return g

    @torch.inference_mode()
    def warmup(self,batch_buckets=None,ctx_buckets=None):
        # Largest first so later (smaller) captures reuse the shared pool's blocks.
        for bb in sorted(batch_buckets or self.batch_buckets,reverse=True):
            for cb in sorted(ctx_buckets or self.ctx_buckets,reverse=True):
                cb=min(cb,self.capacity)
                if (bb,cb) not in self.graphs:self._capture(bb,cb)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def step(self,slots,x):
        B=len(slots)
        if B==0:return x.new_zeros(0,self.hidden)
        if len(set(slots))!=B:raise ValueError('duplicate slots in one step')
        need=0
        for s in slots:
            if not 0<=s<self.max_slots:raise IndexError(s)
            need=max(need,self.lengths_cpu[s]+1)
        if need>self.capacity:raise ValueError(f'context {need} exceeds capacity {self.capacity}')
        bb=self._bucket(self.batch_buckets,B,'batch');cb=self._bucket(self.ctx_buckets,need,'context')
        g=self.graphs.get((bb,cb)) or self._capture(bb,cb)
        key=tuple(slots)
        if g.offset_dirty:g.offset.zero_();g.offset_dirty=False;g.last=None
        if g.last!=key:
            if g.evt is not None:g.evt.synchronize()   # pinned staging may still feed an earlier async copy
            g.slots_cpu[:B]=torch.tensor(slots,dtype=torch.long);g.slots_cpu[B:]=self.dummy
            g.real_cpu[:B]=1;g.real_cpu[B:]=0
            g.slots.copy_(g.slots_cpu,non_blocking=True);g.real.copy_(g.real_cpu,non_blocking=True)
            g.evt=torch.cuda.Event();g.evt.record()
            g.last=key
        g.x[:B].copy_(x)
        if g.graph is not None:
            g.graph.replay();out=g.out[:B].clone()
        else:
            out=self._forward(g.slots,g.x,g.real,cb,g.offset)[:B]
        for s in slots:self.lengths_cpu[s]+=1
        return out

    @torch.inference_mode()
    def load_prefix(self,slot,kv,length):
        """Install a stored reference-prefix KV (list of per-layer (k,v) [1,KV,length,D]) into a slot."""
        if not 0<=slot<self.max_slots:raise IndexError(slot)
        for i,(k,v) in enumerate(kv):
            self.k[i][slot,:,:length].copy_(k[0]);self.v[i][slot,:,:length].copy_(v[0])
        self.lengths[slot]=length;self.lengths_cpu[slot]=length

    @torch.inference_mode()
    def step_mixed(self,slots,x,prefills):
        """One graphed step for decoding seats plus inline prefills.

        prefills: list of (slot, x_tokens [T,hidden]); the slot must already hold its prefix.
        Each prefill contributes T rows of the same slot at consecutive positions: every layer
        writes all rows' K/V before attention, and row i only reads keys <= its own position,
        so this is an exact causal prefill. Returns (decode hidden [B,hidden], [last hidden per prefill]).
        """
        B=len(slots);rows=B+sum(t.shape[0] for _,t in prefills)
        if rows==0:return x.new_zeros(0,self.hidden),[]
        if len(set(slots)|{s for s,_ in prefills})!=B+len(prefills):raise ValueError('duplicate slots in one step')
        need=max([self.lengths_cpu[s]+1 for s in slots]+[self.lengths_cpu[s]+t.shape[0] for s,t in prefills])
        if need>self.capacity:raise ValueError(f'context {need} exceeds capacity {self.capacity}')
        bb=self._bucket(self.batch_buckets,rows,'batch');cb=self._bucket(self.ctx_buckets,need,'context')
        g=self.graphs.get((bb,cb)) or self._capture(bb,cb)
        if g.evt is not None:g.evt.synchronize()
        index=list(slots);offsets=[0]*B
        for s,t in prefills:index+=[s]*t.shape[0];offsets+=range(t.shape[0])
        g.slots_cpu[:rows]=torch.tensor(index,dtype=torch.long);g.slots_cpu[rows:]=self.dummy
        g.offset_cpu[:rows]=torch.tensor(offsets,dtype=torch.long);g.offset_cpu[rows:]=0
        g.real_cpu[:rows]=1;g.real_cpu[rows:]=0
        g.slots.copy_(g.slots_cpu,non_blocking=True);g.real.copy_(g.real_cpu,non_blocking=True)
        g.offset.copy_(g.offset_cpu,non_blocking=True);g.last=None;g.offset_dirty=bool(prefills)
        g.evt=torch.cuda.Event();g.evt.record()
        if B:g.x[:B].copy_(x)
        at=B
        for _,t in prefills:g.x[at:at+t.shape[0]].copy_(t);at+=t.shape[0]
        if g.graph is not None:g.graph.replay();out=g.out
        else:out=self._forward(g.slots,g.x,g.real,cb,g.offset)
        decoded=out[:B].clone();last=[];at=B
        for s,t in prefills:
            at+=t.shape[0];last.append(out[at-1].clone());self.lengths_cpu[s]+=t.shape[0]
        for s in slots:self.lengths_cpu[s]+=1
        return decoded,last

    def memory_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.k+self.v)
