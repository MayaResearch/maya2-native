# ROCm validation test plan

This plan covers the experimental AMD backend in this fork. It separates tests
already executed by the maintainer from tests added here but not yet executed.
A test is not considered passed until output is collected from the RX 9070.

## Target environment recorded so far

| Item | Recorded value |
|---|---|
| GPU | AMD Radeon RX 9070 (gfx1201) |
| ROCm / HIP | 7.1.25424-4179531dcd / 7.1 |
| PyTorch | 2.8.0+rocm7.1.0.git7a520360 |
| Triton | 3.4.0+rocm7.1.0.gitf9e5bf54 |
| Python | 3.12 |
| OS context | Ubuntu 26.04 under WSL2 |
| Project virtualenv | ~/code/maya/.venv |
| Repository checkout | ~/code/maya2-native-rocm |

Confirm these values from the JSON report each time; do not assume the agent
uses the same interpreter or checkout.

## Verified results before this test suite

These results were executed by the maintainer and reported in the project
conversation. They apply to the stated hardware/software stack.

| Test | Result | Evidence |
|---|---|---|
| Backend detection | PASS | HIP 7.1, AMD Radeon RX 9070, Triton importable; graph API symbol present |
| FP16 4096x4096 matmul | PASS | Correct output shape, finite output; approximately 0.09 GB allocated |
| FP16/BF16 SDPA smoke test | PASS | Finite output and expected shape |
| Standalone Triton JIT add | PASS | File-based JIT compile, launch and numerical comparison |
| Standalone torch.cuda.CUDAGraph matmul | PASS | Capture/replay succeeded; finite output |
| Maya _decode_attn | PASS | FP16 max abs error 0.00024277 (tol 0.005); BF16 max abs error 0.00195140 (tol 0.04); finite outputs |

The _decode_attn test used B=2, H=4, KV heads=2, D=64, cache length=128,
and causal positions 37 and 81. The repository test adds another batch/head/
context configuration and graph-captured invocation.

The compiler warning about _POSIX_C_SOURCE being redefined appeared during
Triton/HIP helper compilation. It was non-fatal in the successful runs. Do not
treat this warning alone as a test failure.

## Test inventory added here

| Test module | Coverage | Status before first suite run |
|---|---|---|
| test_backend_primitives.py | FP16/BF16 matmul, standalone Triton JIT add, SDPA, generic graph capture/replay | Pending in suite; prior smoke tests exist |
| test_backbone_kernels.py | _decode_attn, _qk_norm_rope_cache, _add_rms, _decode_attn graph capture | _decode_attn previously passed; expanded cases and other kernels pending |
| test_depth_kernels.py | _add_ln with/without cross residual, greedy/stochastic _topk_sample, _row_exponential | Pending |
| scripts/run_rocm_tests.py | Runs suite and emits machine-readable JSON | Pending |

These tests exercise individual kernels and primitives. They do not instantiate
the full checkpoint-backed StaticBackbone, FastDepth, StaticMimi, or Engine.
Full model/component graph capture and end-to-end synthesis remain separate
acceptance gates.

## Run instructions

Run from the repository root with the known-good ROCm virtual environment.
Do not install or upgrade packages as part of the test run.

    cd ~/code/maya2-native-rocm
    pwd
    ~/code/maya/.venv/bin/python -c 'import torch, triton; print("torch", torch.__version__); print("HIP", torch.version.hip); print("triton", triton.__version__); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
    ~/code/maya/.venv/bin/python scripts/run_rocm_tests.py --json /tmp/maya-rocm-test-results.json

The runner exits with code 0 only when there are no failures or errors. Tests
are skipped if the active PyTorch build is not ROCm or no accelerator is
available. A skipped test is not a pass.

## Individual test commands

    ~/code/maya/.venv/bin/python -m unittest scripts.rocm_tests.test_backend_primitives -v
    ~/code/maya/.venv/bin/python -m unittest scripts.rocm_tests.test_backbone_kernels -v
    ~/code/maya/.venv/bin/python -m unittest scripts.rocm_tests.test_depth_kernels -v

## Instructions for the local AI agent

1. Read this document and docs/ROCM.md before changing source code.
2. Confirm pwd, Git branch, Git status, Python executable, PyTorch, HIP,
   Triton and GPU identity. Expected branch: feature/rocm-backend.
3. Do not modify kernels, dependencies, environment variables, compiler flags,
   or test tolerances before the baseline run.
4. Run the complete suite exactly as shown above.
5. Preserve stdout/stderr. Report compiler warnings separately from exceptions
   and failed assertions.
6. Return /tmp/maya-rocm-test-results.json and the complete console output.
7. If a test fails, reproduce it alone and report the first traceback. Do not
   silently patch code or weaken tolerances. Propose a minimal fix and wait for
   approval before modifying the implementation.

## Result interpretation

- PASS: test completed and all assertions passed.
- FAIL: an assertion detected a numerical or behavioral mismatch.
- ERROR: import, compilation, launch, runtime, or unexpected exception.
- SKIP: ROCm accelerator/runtime unavailable; this is not evidence of support.
- WARNING: compiler/runtime warning without a failing test. Include it, but do
  not label the test failed solely because of it.

## Remaining acceptance gates (not covered by this suite)

1. Compare StaticBackbone eager execution against the reference Hugging Face
   backbone on identical model weights and fixed inputs.
2. Validate complete backbone graph capture/replay for real model shapes,
   slot reuse, batch buckets, and context buckets.
3. Compare FastDepth eager and fused execution against the reference depth
   decoder, including all 32 codebook levels and per-request sampling.
4. Validate complete FastDepth graph capture/replay.
5. Compare StaticMimi eager frame decode against reference Mimi; validate state
   reset, multi-frame continuity, and complete graph capture/replay.
6. Validate Engine.warmup(), voice preparation, admission, streaming,
   cancellation, slot reuse, and end-to-end audio generation.
7. Measure real-time factor, first-audio latency, throughput, peak VRAM/RAM,
   and subjective voice identity for Tamil and Indian English.

These component tests require matching model checkpoints, backbone and Mimi
assets, and known-good reference outputs. Add them as opt-in tests when those
local asset paths and fixtures are supplied; this suite does not guess paths
or download model weights.
