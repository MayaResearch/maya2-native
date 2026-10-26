"""Reference ragged-cache batching; positions remain local to each seat."""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


def cache_from_owned_tensors(layers):
    """Install freshly owned tensors without concatenating them to empty caches.

    DynamicCache(ddp_cache_data=...) calls update(), which copies every tensor
    once more. These inputs are already owned by this cache, so initialization
    can install them directly using the pinned DynamicLayer representation.
    """
    cache=DynamicCache()
    for keys,values in layers:
        layer=DynamicLayer()
        layer.keys,layer.values=keys,values
        layer.dtype,layer.device=keys.dtype,keys.device
        layer.is_initialized=True
        cache.layers.append(layer)
    return cache


def pack_caches(caches, new_tokens, device):
    lengths=[c.layers[0].keys.shape[-2] if c is not None else 0 for c in caches]
    maximum=max(lengths)
    if maximum==0:
        return None,None,lengths
    exemplar=next(c for c in caches if c is not None)
    layers=[]
    for li in range(len(exemplar.layers)):
        kv=[]
        for name in ('keys','values'):
            sample=getattr(exemplar.layers[li],name)
            stacked=sample.new_zeros(len(caches),sample.shape[1],maximum,sample.shape[-1])
            for i,(c,length) in enumerate(zip(caches,lengths)):
                if length:
                    stacked[i:i+1,:,-length:]=getattr(c.layers[li],name)
            kv.append(stacked)
        layers.append(tuple(kv))
    mask=torch.arange(maximum+new_tokens,device=device)[None]>=torch.tensor(
        [maximum-l for l in lengths],device=device)[:,None]
    return cache_from_owned_tensors(layers),mask,lengths


def split_cache(cache,lengths,keep=None,*,copy=True):
    """Split cache rows, optionally borrowing immutable views for experiments.

    The default gives independent mutable storage. copy=False permits only
    DynamicLayer append/crop operations, which replace tensors rather than
    writing existing storage. A borrowed row retains the whole packed batch
    until all views die; measure memory before using it in a serving path.
    """
    def take(tensor,i,length):
        view=tensor[i:i+1,:,-length:]
        return view.clone(memory_format=torch.contiguous_format) if copy else view
    result=[]
    for i,length in enumerate(lengths):
        length=min(length,keep) if keep is not None else length
        result.append(cache_from_owned_tensors([
            (take(l.keys,i,length),take(l.values,i,length)) for l in cache.layers]))
    return result
