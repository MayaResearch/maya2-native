# Full-stack ROCm component validation

This suite extends the isolated kernel tests with checkpoint-backed validation of the first four full-stack acceptance gates. It is designed for the ROCm PyTorch environment used for the Radeon RX 9070 port.

## Coverage

| Gate | Component | Validation |
|---|---|---|
| 1 | `serve.fast_backbone.StaticBackbone` | Two consecutive decode steps; fused CUDA-graph path compared with an unfused eager StaticBackbone reference; verifies output shape and finite values |
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

1. **StaticBackbone** compares the fused graph path against an unfused eager StaticBackbone reference using identical weights and inputs for two sequential positions. It does not independently establish parity against a separately instantiated upstream Hugging Face model.
2. **FastDepth** checks all 32 generated codebook IDs and the feedback embedding. Greedy sampling is enabled only for deterministic validation; the engine's configured sampling mode is restored afterward.
3. **StaticMimi** compares eager and graph executions of the same static-state decoder, checks two-frame state continuity, and verifies reset reproducibility. A separate reference Mimi decoder comparison and audio-quality evaluation remain desirable.
4. **Engine** uses the plain voice. The cancellation test requires the request to terminate with the engine's `cancelled` error; it does not accept natural completion as a cancellation pass.
5. This suite does not establish subjective voice similarity, Tamil/Indian-English pronunciation quality, long-duration stability, streaming SLA, multi-seat throughput, or production readiness.
6. The script exercises the existing ROCm-compatible PyTorch/HIP `torch.cuda` graph API. It does not change model weights, kernel math, package versions, or dependency pins.

Run the isolated kernel suite separately as well:

```bash
~/code/maya/.venv/bin/python scripts/run_rocm_tests.py \
  --json /tmp/maya-rocm-kernel-results.json
```
