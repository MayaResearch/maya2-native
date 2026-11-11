# Maya3 ROCm Port — Handoff Context

**Repository:** https://github.com/tmasabari/maya2-native  
**Working branch:** `feature/rocm-backend`  
**Target:** AMD Radeon RX 9070 (`gfx1201`)  
**Last updated:** 2026-11-11

## Environment

- Ubuntu 26.04 / Linux kernel 7.0.0-43-generic
- ROCm/HIP 7.1.25424-4179531dcd / 7.1
- PyTorch `2.8.0+rocm7.1.0.git7a520360`
- Triton `3.4.0+rocm7.1.0.gitf9e5bf54`
- Python 3.12.16; Transformers 5.6.2; Tokenizers 0.22.2
- ROCm virtual environment: `~/code/maya/.venv`
- Repository checkout: `~/code/maya2-native-rocm`
- Checkpoint: `~/.cache/huggingface/hub/models--maya-research--maya2-native/snapshots/87a83efdd6551e038246b6b37836a6264713eec3`

## Verified evidence

1. Isolated ROCm kernel suite: 12 passed, 0 failed, 0 errors, 0 skipped (user-reported).
2. Checkpoint-backed component suite: all four gates passed — StaticBackbone graph/eager same-configuration comparison, FastDepth 32-codebook path, StaticMimi two-frame graph/eager and reset behavior, and Engine lifecycle (warmup, synthesis, cancellation, slot reuse, queue drain).
3. Standalone CLI synthesis: user confirmed successful creation of `maya-rocm-demo-ta.wav` using Tamil (`--lang ta`) and built-in Kayal voice. Output is written as 24 kHz mono WAV by `examples/synthesize.py`.
4. The CLI initially failed because `soundfile` was absent in the active environment. Installing with `~/code/maya/.venv/bin/python -m pip install soundfile` resolved the dependency issue. Do not install the upstream CUDA-pinned dependency set into this ROCm environment.

## Reproduce the Tamil CLI smoke test

```bash
cd ~/code/maya2-native-rocm
source ~/code/maya/.venv/bin/activate
export HIP_VISIBLE_DEVICES=0
CKPT="$HOME/.cache/huggingface/hub/models--maya-research--maya2-native/snapshots/87a83efdd6551e038246b6b37836a6264713eec3"

python examples/synthesize.py \\
  "வணக்கம்! இது AMD Radeon RX 9070 GPU பயன்படுத்தி Maya2 மூலம் உருவாக்கப்பட்ட தமிழ் குரல் சோதனை." \\
  --model "$CKPT" --lang ta --voice Kayal --seed 42 \\
  --out maya-rocm-demo-ta.wav
```

Playback: `aplay maya-rocm-demo-ta.wav`.

Indian English demo:

```bash
python examples/synthesize.py \\
  "Hello! This is a voice generation test running on an AMD Radeon RX 9070 GPU." \\
  --model "$CKPT" --lang en --voice Siddhant --seed 42 \\
  --out maya-rocm-demo-en.wav
```

## Important unresolved issue

StaticBackbone graph replay matched same-configuration eager execution exactly (max absolute error 0.0 for both tested decode steps). However, fused-eager versus unfused-eager differed by max absolute error 0.2265625 at step 1 and 1.375 at step 2. This is not a graph replay failure, but fused/unfused numerical parity remains unresolved. Investigate projection layout, fused Q/K normalization and RoPE, KV-cache writes/reads, and sequential error accumulation. Do not weaken tolerances or claim parity until isolated.

## Not yet validated

- Subjective Tamil/Indian-English pronunciation and naturalness rubric
- Voice cloning with reference audio / ECAPA speaker conditioning
- Reference-engine parity for backbone, depth and Mimi outputs
- Sustained streaming, long-duration stability, concurrency and production SLA
- Measured RTF, sustained throughput, peak VRAM/RAM during standalone CLI generation

## Evidence discipline

The successful WAV creation is user-confirmed functional evidence for one CLI sample. It is not a performance benchmark or production-readiness claim. Keep user-observed results, automated component tests, and external upstream performance claims clearly separated.


## CUDA backward-compatibility status

No NVIDIA CUDA hardware is available to the maintainer, so runtime CUDA
regression testing has not been performed. The PR's production source diff
does not directly modify the existing inference engine or CUDA kernel files,
but that static observation is not a runtime guarantee. The CUDA acceptance
matrix is documented in `docs/CUDA_BACKWARD_COMPATIBILITY.md` and must be
run on NVIDIA hardware before claiming CUDA backward compatibility.
