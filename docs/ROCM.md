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
- Maya's custom Triton `_decode_attn` kernel: FP16 and BF16 compilation,
  execution, finite-output checks, and comparison against an independent
  PyTorch attention reference all passed
- Finite outputs for the tested shapes

For the Maya `_decode_attn` test, the maximum absolute error against the
FP32 PyTorch reference was 0.000243 for FP16 and 0.001951 for BF16, within the
test tolerances of 0.005 and 0.04 respectively. The test used B=2, H=4,
KV heads=2, head dimension=64, and cache length=128, with causal positions
37 and 81.

These are maintainer-run smoke tests on the stated hardware/software stack.
They validate the tested primitives and the tested `_decode_attn` configuration,
not the complete Maya execution paths. The standalone graph test does **not**
validate capture/replay of the full backbone, depth decoder, or Mimi decoder.
The checkpoint-backed component runner and the CLI synthesis run now add
higher-level evidence; see `docs/ROCM_COMPONENT_TESTS.md` and
`docs/ROCM_END_TO_END_TTS.md`. Voice cloning, reference-audio conditioning,
quality evaluation, sustained streaming, and production readiness remain
separate acceptance gates.

## CUDA backward-compatibility boundary

ROCm enablement is additive. Existing NVIDIA CUDA inference remains the
reference/default path and must not be rewritten, silently reconfigured, or
made dependent on ROCm diagnostics.

- Do not import or execute ROCm test runners during normal package import or
  CUDA inference startup.
- Do not change CUDA kernels, graph/stream behavior, default inference
  configuration, dependency pins, or environment variables as part of ROCm
  enablement.
- Keep ROCm-specific checks and test commands opt-in.
- Backend identity is derived from the PyTorch build marker (`torch.version.hip`
  or `torch.version.cuda`), separately from runtime device availability.
- `torch.cuda` API presence and `CUDAGraph` symbol presence are not proof of
  NVIDIA CUDA hardware or feature support.
- CUDA behavior cannot be declared regression-free until the CUDA acceptance
  suite in `docs/CUDA_BACKWARD_COMPATIBILITY.md` has been run on supported
  NVIDIA hardware. The current maintainer does not have CUDA hardware, so this
  gate remains explicitly unverified.

## Why the complete engine is not yet ROCm-ready

The current engine has several NVIDIA-oriented fast paths:

- `serve/fast_backbone.py` contains custom Triton attention, fused RoPE/cache,
  and add+RMSNorm kernels, plus CUDA graph/stream capture paths. The
  `_decode_attn` kernel passed an isolated FP16/BF16 correctness test on the
  RX 9070. Fused RoPE/cache, add+RMSNorm, and the complete backbone capture
  body still need independent validation.
- `serve/fast_depth_kernels.py` contains custom Triton sampling and normalization
  kernels that have been tested in the isolated suite on the RX 9070.
- `serve/fast_depth.py` and `serve/fast_codec.py` capture CUDA graphs; the
  complete captured bodies were exercised by the checkpoint-backed component
  run, but reference-engine numerical parity remains a separate gate.
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

## Automated kernel validation suite

Run the suite from the repository root with the already validated ROCm Python
environment. The runner writes a JSON report and returns a non-zero exit code
when a test fails or errors:

```bash
python scripts/run_rocm_tests.py --json /tmp/maya-rocm-test-results.json
```

The isolated kernel suite covers generic PyTorch primitives, standalone Triton JIT and graph capture, Maya backbone
kernels (`_decode_attn`, `_qk_norm_rope_cache`, `_add_rms`), depth kernels
(`_add_ln`, `_topk_sample`, `_row_exponential`), and graph capture of the
Maya decode-attention kernel. Tests skipped because ROCm is unavailable are
not compatibility passes. This isolated suite does not instantiate full
model-backed components; use the checkpoint-backed runner below for that.

See `docs/ROCM_TEST_PLAN.md` for verified historical results, test coverage,
run instructions, and the JSON report contract.

## Checkpoint-backed component validation

The full-component runner is:

```bash
PYTHON=~/code/maya/.venv/bin/python bash scripts/run_rocm_component_tests.sh \
  --ckpt /absolute/path/to/checkpoint-or-release \
  --backbone-dir /absolute/path/to/backbone \
  --mimi-dir /absolute/path/to/mimi \
  --report /tmp/maya-rocm-component-results.json
```

It validates graph/eager consistency for StaticBackbone, all 32 FastDepth
codebooks, StaticMimi frame-state/reset behavior, and Engine warmup,
synthesis, cancellation, and slot reuse. It requires local checkpoint assets
and does not download model weights. The revised runner completed all four
gates on the target RX 9070; see `docs/ROCM_COMPONENT_TESTS.md` for the
recorded results and limitations.

## Read-only diagnostic

From the repository root, run:

```bash
python scripts/rocm_diagnostics.py
```

The report distinguishes PyTorch's HIP and CUDA build versions. The
`cuda_graph_api_present` field reports only symbol presence; it is not a
capability or correctness test. Diagnostic device-metadata failures are
reported as unavailable metadata and do not change backend identity.

## Porting sequence and acceptance gates

1. Load the checkpoint and run the unoptimized Hugging Face/PyTorch backbone.
2. Compare eager backbone outputs against the reference backend on fixed inputs.
3. Validate speaker encoder and reference conditioning.
4. Validate depth decoder eager operations and sampling.
5. Validate Mimi decoder eager frame decoding and audio output.
6. Validate the remaining Maya Triton kernels individually on ROCm, including
   fused RoPE/cache, add+RMSNorm, depth add+LayerNorm, and sampling. The
   `_decode_attn` kernel has passed its isolated FP16/BF16 correctness test.
7. Validate graph capture/replay for each complete component (backbone, depth,
   Mimi), not just a standalone matrix multiplication.
8. Keep optimized paths behind explicit capability/configuration checks and
   provide an eager PyTorch fallback where a project-specific operation fails.

A complete AMD acceptance run must include end-to-end Tamil and Indian English
voice-cloning samples, numerical checks, real-time factor, first-audio latency,
peak VRAM/RAM, and subjective speaker-identity evaluation.

## End-to-end CLI synthesis evidence

A user-run Tamil command-line synthesis completed successfully on the RX 9070 ROCm environment and wrote `maya-rocm-demo-ta.wav`. The run used `examples/synthesize.py`, the cached `maya-research/maya2-native` checkpoint, `--lang ta`, `--voice Kayal`, and a fixed seed. The script writes 24 kHz mono WAV output. The user first encountered a missing `soundfile` Python module; installing it into the existing ROCm environment with `python -m pip install soundfile` resolved the import issue. Do not reinstall the upstream CUDA-pinned dependency set to resolve this package omission.

This establishes a successful user-observed CLI-to-WAV synthesis path for that sample. No audio-quality score, real-time factor, peak-memory measurement, voice-cloning result, or sustained-streaming result was supplied, so none is claimed. See `docs/ROCM_END_TO_END_TTS.md` for the reproducible commands and remaining acceptance gates.
