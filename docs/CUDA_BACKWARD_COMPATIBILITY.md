# CUDA backward-compatibility acceptance

## Purpose

ROCm support is an additive backend effort. It must not alter the existing
NVIDIA CUDA inference implementation, defaults, dependencies, or runtime
behavior. This document defines the CUDA regression gate independently from
the ROCm test suites.

## Current verification status

**Not run: NVIDIA hardware is unavailable to the maintainer.**

The repository changes in this PR do not directly modify the existing
inference implementation modules or CUDA dependency pins. This is static
scope evidence only; it is not proof of runtime compatibility, numerical
parity, or unchanged performance. Do not mark the CUDA gate as passed until
the commands below have been executed on a supported NVIDIA system and the
environment/report have been attached to the PR.

Mocked backend-detection tests can validate classification logic on CPU.
They cannot validate CUDA kernels, graph capture, memory behavior, inference
outputs, or performance.

## Required NVIDIA run

Use a clean environment matching the project's supported CUDA/PyTorch/Triton
pins. Do not install ROCm wheels or change dependency pins to run this gate.
Record GPU model, driver, CUDA runtime, PyTorch, Triton, Python, checkpoint
revision, and the exact command/configuration.

Run the existing project test suite first, then the following acceptance
checks using the same checkpoint, prompts, seeds, sampling parameters, and
configuration as the recorded CUDA baseline:

| Area | Required check | Acceptance |
|---|---|---|
| Backend detection | Real CUDA build and available GPU | `name=cuda`, `available=true` |
| FP16/BF16 | Existing inference smoke tests | No new failures; BF16 only where supported |
| Attention | Existing Triton attention path | Correctness against established baseline |
| KV cache | Prefill plus multiple decode steps | No changed cache contents or token behavior beyond established tolerance |
| CUDA graphs | Existing capture/replay paths | Capture, replay, and lifecycle succeed |
| FastDepth | Existing depth-decoder path | Same output behavior under fixed inputs/seeds |
| Mimi | Existing codec decode path | Same frame/state behavior under fixed inputs |
| End-to-end | CLI synthesis | Successful output and baseline audio/numerical comparison |
| Performance | Representative baseline workload | No unexplained regression; report throughput, latency, and peak memory |

Do not invent universal numerical tolerances. Reuse the project's established
CUDA test tolerances and baseline artifacts. If no baseline exists for a
component, record that gap rather than declaring parity.

## Isolation rules

- ROCm diagnostics and ROCm test runners are opt-in commands; they must not be
  imported or executed as a side effect of normal inference startup.
- Do not mutate process-wide environment variables from backend detection or
  diagnostic code.
- Do not change CUDA kernels, CUDA graph/stream code, default device selection,
  sampling behavior, dependency pins, or CUDA installation instructions as
  part of the ROCm-only change.
- Backend identity is based on the PyTorch build marker; device availability
  is a separate field.
- ROCm's `torch.cuda` compatibility namespace must never be interpreted as
  proof of NVIDIA CUDA.

## Evidence to attach

Attach the full test output and machine-readable report, including environment
metadata, test names, pass/fail/skip status, and any performance comparison.
A skipped or mocked test is not a substitute for this NVIDIA hardware run.
