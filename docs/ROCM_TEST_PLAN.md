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
| scripts/run_rocm_tests.py | Runs isolated suite and emits machine-readable JSON | User-reported run: 12 passed, 0 failed/errors/skips |
| scripts/run_rocm_component_tests.py | Checkpoint-backed StaticBackbone, FastDepth, StaticMimi, and Engine lifecycle checks | Revised RX 9070 run: all four gates PASS. Fused-versus-unfused StaticBackbone numerical parity remains a separate open diagnostic |

The isolated tests exercise individual kernels and primitives. The aggregate
run was reported by the user as 12 tests passed, 0 failures, 0 errors, and 0
skips (4.072 seconds). The machine-readable JSON report was not attached to the
repository, so per-test records and its environment snapshot are not archived.

A second, checkpoint-backed runner now exercises the first four component
gates. It is opt-in because it requires local checkpoint, backbone, and Mimi
assets. It compares StaticBackbone graph/eager decode, FastDepth graph/eager
32-codebook output, StaticMimi eager/graph frame continuity and reset, then
Engine warmup, synthesis, cancellation, and slot reuse. See
`docs/ROCM_COMPONENT_TESTS.md`. This runner is implemented but has not yet been
executed on the target machine; no component-level PASS is claimed.

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

## Remaining acceptance gates after the component runner

1. Execute `scripts/run_rocm_component_tests.sh` on the RX 9070 and archive its
   JSON report and complete console output. Fix failures without weakening
   tolerances or converting errors into skips.
2. Compare StaticBackbone outputs against the reference Hugging Face backbone
   on identical weights, prompt tokens, and fixed inputs (the current component
   runner compares graph and eager implementations of StaticBackbone itself).
3. Compare FastDepth eager/fused results against the reference depth decoder,
   including non-greedy per-request sampling and numerical/code agreement.
4. Compare StaticMimi output against the reference Mimi decoder, not only its
   own eager implementation; add waveform-level and multi-frame quality checks.
5. Validate cloned-speaker conditioning and reference-audio preparation with
   the local ECAPA assets.
6. Run longer streaming/soak and multi-seat concurrency tests; measure real-time
   factor, first-audio latency, throughput, peak VRAM/RAM, and error recovery.
7. Evaluate Tamil and Indian English pronunciation, naturalness, and speaker
   identity using recorded reference samples and an objective/subjective rubric.

These component tests require matching model checkpoints, backbone and Mimi
assets, and known-good reference outputs. Add them as opt-in tests when those
local asset paths and fixtures are supplied; this suite does not guess paths
or download model weights.


## Revised checkpoint-backed component run

The revised runner passed all four gates on the RX 9070: StaticBackbone graph versus same-configuration eager (two steps, max absolute error 0.0), FastDepth (32 codebooks, feedback max absolute error 0.0), StaticMimi (two frames, graph/eager max absolute error 0.0, reset reproducible), and Engine lifecycle (warmup, synthesis, cancellation, slot reuse, queue drain).

The run reported 0.919 s warmup, approximately 36 ms first-audio latency, approximately 22.36 ms frame phase, and approximately 8.10 GB allocated after requests. These are short-run measurements only.

A separate StaticBackbone diagnostic still shows fused-eager versus unfused-eager maximum absolute errors of 0.2265625 (step 1) and 1.375 (step 2). Graph replay parity is established for the tested configuration; fused/unfused reference parity is not. Further investigation is required.


## End-to-end synthesis milestone (user-confirmed)

The user subsequently ran `examples/synthesize.py` against the local checkpoint on the RX 9070 ROCm environment and confirmed successful creation of `maya-rocm-demo-ta.wav` using Tamil and the built-in `Kayal` voice. `soundfile` was initially absent from the active Python environment; installing it with the active ROCm interpreter's `python -m pip install soundfile` resolved the issue. This is a successful CLI-to-WAV smoke test. No audio-quality rubric, RTF, peak VRAM/RAM, long-duration test, or voice-cloning result has yet been recorded. Keep the fused-eager versus unfused-eager StaticBackbone mismatch (max absolute errors 0.2265625 and 1.375 over two decode steps) as an unresolved diagnostic.


## CUDA backward-compatibility gate (not run)

The maintainer does not have NVIDIA CUDA hardware. Therefore this PR cannot
claim that CUDA runtime behavior or performance has been regression-tested.
The production diff is scoped to new ROCm/backend diagnostic and test files
plus documentation; this is static inspection, not runtime proof.

The CPU-only backend tests verify classification, including the important
case where a CUDA-built PyTorch has no available GPU. They do not test CUDA
kernels or inference. The required NVIDIA acceptance matrix and evidence
requirements are documented in `docs/CUDA_BACKWARD_COMPATIBILITY.md`.
Keep this gate marked **NOT RUN** until an NVIDIA system executes it.
