# Full-stack ROCm component validation

This suite extends the isolated kernel tests with checkpoint-backed validation of the first four full-stack acceptance gates. It is designed for the ROCm PyTorch environment used for the Radeon RX 9070 port.

## Coverage

| Gate | Component | Validation |
|---|---|---|
| 1 | `serve.fast_backbone.StaticBackbone` | Two consecutive decode steps; graph path compared with same-configuration eager execution; fused-versus-unfused differences are recorded separately as diagnostics |
| 2 | `serve.fast_depth.FastDepth` | Full 32-codebook frame; graph replay compared with the eager frame body; greedy mode makes the comparison deterministic; compares feedback embedding |
| 3 | `serve.fast_codec.StaticMimi` | Eager versus graph decode for two consecutive frames; reset reproducibility; verifies frame sample count |
| 4 | `maya3.engine.Engine` | Full graph warmup; plain-voice synthesis; cancellation after first audio; subsequent request verifies released-slot reuse and queue drain |

The comparisons deliberately use the same model weights and component math. They establish graph/eager consistency and lifecycle correctness; they are not a substitute for a separately generated Hugging Face/reference-engine quality evaluation.

## Prerequisites

- The existing ROCm virtual environment, with PyTorch built for ROCm/HIP and Triton available.
- A local Maya checkpoint or release directory.
- The matching local Qwen backbone directory.
- A local Mimi model directory.
- No checkpoint or model download is initiated by this script.
- Run on the same branch and environment as the isolated ROCm suite.

The engine uses the plain voice for this smoke test, so an ECAPA directory and reference recording are not required. Voice cloning quality is a separate acceptance gate.

## Run all four gates

From the repository root:

```bash
cd ~/code/maya2-native-rocm
git status --short --branch
git pull --ff-only origin feature/rocm-backend

PYTHON=~/code/maya/.venv/bin/python bash scripts/run_rocm_component_tests.sh \
  --ckpt /absolute/path/to/maya-checkpoint-or-release \
  --backbone-dir /absolute/path/to/backbone \
  --mimi-dir /absolute/path/to/mimi \
  --max-seats 1 \
  --capacity 512 \
  --max-inline-rows 0 \
  --depth-dtype fp16 \
  --report /tmp/maya-rocm-component-results.json
```

If the ROCm environment is elsewhere, set `PYTHON` to its Python executable. The conservative defaults are one engine seat and zero inline-prefill rows to limit graph-memory pressure on a 16 GB GPU; increase them only after the single-seat run passes. The runner defaults to `cuda` as the PyTorch device name because ROCm exposes its accelerator through PyTorch's `torch.cuda` API; it separately verifies that `torch.version.hip` is populated.

For a lower-risk run that performs all three component tests and engine graph warmup but does not start the request/cancellation lifecycle:

```bash
PYTHON=~/code/maya/.venv/bin/python bash scripts/run_rocm_component_tests.sh \
  --ckpt /absolute/path/to/maya-checkpoint-or-release \
  --backbone-dir /absolute/path/to/backbone \
  --mimi-dir /absolute/path/to/mimi \
  --skip-engine-stream
```

## Outputs and exit codes

The JSON report records:
- Python, PyTorch, HIP, Triton and GPU identity;
- arguments and model-load duration;
- PASS/FAIL for each of the four gates;
- per-test duration and relevant numeric details;
- full traceback for failures;
- aggregate result.

Exit code:
- `0`: all four gates passed;
- `1`: one or more gates failed, or setup failed after ROCm validation;
- `2`: ROCm accelerator or ROCm PyTorch build is unavailable.

A skipped engine stream is intentionally not considered a complete pass: use it only to isolate component warmup and graph behavior.

## Important interpretation

1. **StaticBackbone** first compares graph replay with eager execution using identical fusion flags, weights, inputs, and two sequential positions. It separately records fused-eager versus unfused-eager numerical differences; those diagnostics must not be confused with graph replay correctness. Neither comparison independently establishes parity against a separately instantiated upstream Hugging Face model.
2. **FastDepth** checks all 32 generated codebook IDs and the feedback embedding. Greedy sampling is enabled only for deterministic validation; the engine's configured sampling mode is restored afterward.
3. **StaticMimi** compares eager and graph executions of the same static-state decoder under inference mode, checks two-frame state continuity, and verifies reset reproducibility. A separate reference Mimi decoder comparison and audio-quality evaluation remain desirable.
4. **Engine** uses the plain voice. The cancellation test requires the request to terminate with the engine's `cancelled` error; it does not accept natural completion as a cancellation pass.
5. This suite does not establish subjective voice similarity, Tamil/Indian-English pronunciation quality, long-duration stability, streaming SLA, multi-seat throughput, or production readiness.
6. The script exercises the existing ROCm-compatible PyTorch/HIP `torch.cuda` graph API. It does not change model weights, kernel math, package versions, or dependency pins.

Run the isolated kernel suite separately as well:

```bash
~/code/maya/.venv/bin/python scripts/run_rocm_tests.py \
  --json /tmp/maya-rocm-kernel-results.json
```


## Initial RX 9070 component run

The first user-reported run on the target environment completed all four gates:
- FastDepth: PASS (32 codebooks; graph/eager feedback maximum absolute error 0).
- Engine lifecycle: PASS (warmup, plain-voice audio generation, cancellation, slot reuse, and queue drain).
- StaticBackbone: FAIL at graph-versus-unfused-reference step 1. The original test mixed graph replay and fusion differences in one assertion.
- StaticMimi: FAIL because the eager test call mutated inference tensors outside `torch.inference_mode()`.

A follow-up test-only change separates StaticBackbone graph-versus-same-configuration-eager parity from fused-versus-unfused diagnostics, and runs the StaticMimi test under inference mode. Commit `0e5f19dbb8cbee2a0367017b1f71a2c449ceda86`. These changes are awaiting a rerun; they are not yet validated.


## Revised RX 9070 component run — PASS

The revised runner completed all four gates on the user's Radeon RX 9070 (gfx1201), ROCm 7.1, PyTorch 2.8.0+rocm7.1.0, and Triton 3.4.0 environment.

| Gate | Result | Evidence |
|---|---|---|
| StaticBackbone | PASS | Graph versus same-configuration eager maximum absolute error 0.0 for both decode steps |
| FastDepth | PASS | 32 codebooks; fused kernel path; feedback maximum absolute error 0.0 |
| StaticMimi | PASS | 32 codebooks; 1,920 samples/frame; two-frame graph/eager maximum absolute error 0.0; reset reproducible |
| Engine lifecycle | PASS | Warmup, plain-voice synthesis, cancellation, slot reuse, and queue drain |

Engine measurements from this short run: warmup 0.919 s; first-audio latency 0.0360 s for the normal request; reported frame phase approximately 22.36 ms; memory allocation after requests approximately 8.10 GB (reserved approximately 8.23 GB). These are smoke-test observations, not sustained performance or SLA claims.

**Important separate diagnostic:** fused eager versus unfused eager StaticBackbone outputs differed by maximum absolute error 0.2265625 at step 1 and 1.375 at step 2 (mean absolute errors 0.0054241 and 0.0664032). This does not fail the graph-versus-same-configuration-eager criterion, which was exactly zero, but fused/unfused numerical parity remains unresolved. Do not claim parity against the unfused reference. Investigate fused QKV projection/view layout, fused Q/K normalization and RoPE, KV-cache writes/reads, and accumulated sequential-step effects before changing kernels or tolerances.

The machine-readable report supplied by the user was `/tmp/maya-rocm-component-results-v2.json`; it is not checked into the repository.


## End-to-end CLI synthesis follow-up

After the four component gates passed, the user ran the standalone CLI synthesis example on the same RX 9070 ROCm environment and confirmed that `maya-rocm-demo-ta.wav` was created successfully. The successful sample used Tamil (`--lang ta`) and the built-in `Kayal` voice. The CLI depends on `soundfile` for WAV writing; the missing module was resolved by installing `soundfile` into the existing ROCm virtual environment using `python -m pip install soundfile`. This is a user-confirmed WAV-generation result, not a voice-cloning or subjective quality evaluation. See `docs/ROCM_END_TO_END_TTS.md`.


## Evidence status and open numerical diagnostic

The revised RX 9070 run reported all four component gates as PASS. This
supersedes the earlier initial-run failure and pending-rerun notes above:
those entries are retained as historical chronology, not current status.

Graph replay versus same-configuration eager execution matched for the two
tested StaticBackbone decode steps. This does not establish fused-versus-
unfused parity. That separate comparison remains unresolved (maximum
absolute error 0.2265625 at step 1 and 1.375 at step 2). Do not relax
tolerances or describe the unfused reference as matched.

CUDA backward compatibility is a separate, unrun gate because the maintainer
does not have NVIDIA hardware. See
`docs/CUDA_BACKWARD_COMPATIBILITY.md`; ROCm PASS results do not imply CUDA
regression coverage.
