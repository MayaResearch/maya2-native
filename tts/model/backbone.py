"""Backbone wrapper: a HF causal LM consumed as a 12.5 Hz sequence model.

Candidate backbones load through the same wrapper so comparing them is a
config switch, not a code fork:
    qwen:   Qwen/Qwen3-4B-Base     (Apache-2.0, full attention)

The wrapper owns three things the raw LM doesn't have:
- vocab extension for the token contract (resize + mean-init of new rows);
- an input-composition layer: text positions use the LM's embedding table,
  audio positions use the depth decoder's summed-code feedback vector, the
  speaker slot uses the conditioning projector's output — all merged into one
  (B, T, d) inputs_embeds tensor (audio never enters the text vocab);
- the cb0 head: n_codes+1 classes (class 2048 = <aeos>), separate from lm_head.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..tokens import CONTRACT, N_CODES, AEOS_CLASS, extend_tokenizer

BACKBONES = {
    "qwen": "Qwen/Qwen3-4B-Base",
    "llama": "meta-llama/Llama-3.2-3B",
}


def _from_config(path):
    import glob, os
    if os.environ.get("MAYA3_BACKBONE_FROM_CONFIG") == "1":
        return True
    return os.path.isdir(path) and not glob.glob(os.path.join(path, "*.safetensors")) and not glob.glob(os.path.join(path, "*.bin"))


def _lm_from_config(path, dtype, attn):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation=attn)
    lm = lm.to_empty(device="cpu")
    for m in lm.modules():   # non-persistent buffers (RoPE inv_freq) were never materialised: recompute them
        if hasattr(m, "inv_freq") and hasattr(m, "rope_type"):
            if getattr(m, "rope_init_fn", None) is not None:
                fn = m.rope_init_fn
            elif m.rope_type == "default":
                fn = m.compute_default_rope_parameters
            else:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                fn = ROPE_INIT_FUNCTIONS[m.rope_type]
            inv, scale = fn(m.config, torch.device("cpu"))
            m.inv_freq = inv
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = inv
            if hasattr(m, "attention_scaling"):
                m.attention_scaling = scale
    lm.tie_weights()
    return lm


class Backbone(nn.Module):
    def __init__(self, name="qwen", dtype=torch.float32, attn="flash_attention_2"):
        # fp32 parameters by default: FSDP2's MixedPrecisionPolicy casts to bf16
        # for compute but the SHARDED source params are what AdamW updates — load
        # them bf16 and there are no fp32 masters.
        super().__init__()
        path = BACKBONES.get(name, name)
        if _from_config(path):
            # inference: every backbone weight comes from the Maya3 checkpoint loaded strictly afterwards, so build the
            # architecture without reading (or even needing) the multi-GB backbone safetensors
            self.lm = _lm_from_config(path, dtype, attn)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=dtype,
                                                           attn_implementation=attn)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer, self.token_ids = extend_tokenizer(self.tokenizer)
        old = self.lm.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > old:
            self.lm.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                emb = self.lm.get_input_embeddings().weight
                emb[old:] = emb[:old].mean(0, keepdim=True) + 0.02 * torch.randn_like(emb[old:])
        self.d_model = self.lm.config.hidden_size
        # cb0 + <aeos>: the only audio-side output the backbone owns
        self.cb0_head = nn.Linear(self.d_model, N_CODES + 1, bias=False)

    def embed_text(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def forward(self, inputs_embeds, attention_mask=None, past_key_values=None,
                use_cache=False):
        out = self.lm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            past_key_values=past_key_values, use_cache=use_cache)
        return out.last_hidden_state, getattr(out, "past_key_values", None)

    def text_logits(self, hidden):
        return self.lm.lm_head(hidden)

    def cb0_logits(self, hidden):
        return self.cb0_head(hidden)
