# Experimental AMD ROCm status

Maya3's upstream fast serving engine was developed and benchmarked on NVIDIA CUDA.
The current ROCm work is an experimental compatibility effort, not an upstream-
certified AMD backend.

## Known status

The following operations have been independently smoke-tested by the maintainer
on an AMD Radeon RX 9070 (gfx1201), ROCm 7.1 and PyTorch 2.8.0:

- FP16 and BF16 matrix multiplication
- FP16 and BF16 `torch.nn.functional.scaled_dot_product_attention`
- A standalone Triton 3.4 JIT elementwise-add kernel: compilation, GPU launch,
  and numerical comparison passed
- A standalone `torch.cuda.CUDAGraph` matrix-multiplication capture and replay
- Finite outputs for the tested shapes

These are maintainer-run smoke tests on the stated hardware/software stack.
They validate the tested primitives, not the complete Maya execution paths.
In particular, the standalone Triton test does **not** validate Maya's custom
attention, fused RoPE, add+RMSNorm, depth sampling, or Mimi operations. The
standalone graph test does **not** validate capture/replay of the full backbone,
depth decoder, or Mimi decoder. Checkpoint loading, speaker conditioning,
streaming, and end-to-end audio generation remain unverified.

## Why the complete engine is not yet ROCm-ready

The current engine has several NVIDIA-oriented fast paths:

- `serve/fast_backbone.py` contains custom Triton attention, fused RoPE/cache,
  and add+RMSNorm kernels, plus CUDA graph/stream capture paths. The generic
  Triton and graph smoke tests passed, but these project-specific kernels and
  capture bodies still need independent validation.
- `serve/fast_depth_kernels.py` contains custom Triton sampling and normalization
  kernels that have not yet been tested on the RX 9070.
- `serve/fast_depth.py` and `serve/fast_codec.py` capture CUDA graphs; the
  complete captured bodies remain unvalidated on ROCm.
- `maya3/engine.py` creates CUDA events and uses CUDA backend controls.

PyTorch on ROCm intentionally exposes much of the `torch.cuda` namespace. A
successful `torch.cuda.is_available()` check therefore does not prove that a
CUDA-specific feature or custom kernel is supported on AMD.

## Environment isolation

Keep the working ROCm environment separate from any upstream CUDA environment.
Do not install the upstream CUDA-pinned dependency set into a validated ROCm
virtual environment.

The upstream `pyproject.toml` pins torch/torchaudio 2.8.0 and its README
installation command selects CUDA 12.8 wheels. Install dependencies in a fresh
environment and select AMD's wheel source explicitly, following the AMD
installation instructions for the exact OS, ROCm and Python combination.

## Read-only diagnostic

From the repository root, run:

```bash
python scripts/rocm_diagnostics.py
```

The report distinguishes PyTorch's HIP and CUDA build versions. The
`cuda_graph_api_present` field reports only symbol presence; it is not a
capability or correctness test.

## Porting sequence and acceptance gates

1. Load the checkpoint and run the unoptimized Hugging Face/PyTorch backbone.
2. Compare eager backbone outputs against the reference backend on fixed inputs.
3. Validate speaker encoder and reference conditioning.
4. Validate depth decoder eager operations and sampling.
5. Validate Mimi decoder eager frame decoding and audio output.
6. Validate the existing Maya Triton kernels individually on ROCm, including
   attention, fused RoPE/cache, add+RMSNorm, depth add+LayerNorm, and sampling.
7. Validate graph capture/replay for each complete component (backbone, depth,
   Mimi), not just a standalone matrix multiplication.
8. Keep optimized paths behind explicit capability/configuration checks and
   provide an eager PyTorch fallback where a project-specific operation fails.

A complete AMD acceptance run must include end-to-end Tamil and Indian English
voice-cloning samples, numerical checks, real-time factor, first-audio latency,
peak VRAM/RAM, and subjective speaker-identity evaluation.
