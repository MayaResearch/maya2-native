# Maya3 engine internals

Maya3 is a Qwen3-1.7B backbone plus a 555M depth decoder.
- It predicts 32 Mimi codebooks at 12.5 Hz, one 80 ms frame per step, and the audio is 24 kHz mono.
- It conditions on an ECAPA-192 speaker embedding.
- It speaks 12 languages: `as bn en gu hi kn ml mr or pa ta te`.

This package is the inference stack:
- a continuous-batching engine where every hot step is a CUDA graph;
- a voice registry that caches each voice's prompt prefix;
- guards inside the engine, on by default;
- a Python API, an OpenAI-compatible HTTP endpoint and a streaming WebSocket.

## Install

You need Python 3.10-3.12 and an NVIDIA GPU with a CUDA 12.8 driver. torch 2.8.0+cu128 comes from the PyTorch index.

```bash
git clone https://github.com/MayaResearch/maya2-native maya3 && cd maya3
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server]" --extra-index-url https://download.pytorch.org/whl/cu128
# extras: [server] = aiohttp (HTTP/WS server + client); [dev] = server deps + pytest, pytest-asyncio, pytest-aiohttp
```

CPU-only checks need no GPU (see `run_tests.sh` at the repo root):

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q maya3/tests/test_text.py maya3/tests/test_guards.py maya3/tests/test_server.py
```

Pinned runtime versions:

| package | version |
|---|---|
| torch | 2.8.0, with triton 3.4.0 |
| torchaudio | 2.8.0 |
| transformers | 5.6.2 |
| speechbrain | 1.1.2 |

numpy, scipy, soundfile, huggingface_hub, safetensors and pyyaml are also required.

The install provides the packages `maya3`, `tts` (model code), `serve` (CUDA-graph components) and `scripts` (prompt and stop controls), plus `configs/`. An editable install (`-e`) is recommended. A plain `pip install .` also works, because the two `configs/` files the engine reads are packaged, but then Triton caches its kernels inside the installed `serve/` package unless `TRITON_CACHE_DIR` is set.

## Quick start (Python)

```python
from maya3 import Maya3TTS
import soundfile as sf

tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=16)   # HF repo id or a local release dir

wav = tts.generate("नमस्ते, आज मौसम बहुत अच्छा है।", lang="hi")          # float32, 24 kHz mono
sf.write("out.wav", wav, 24000)

wav = tts.generate("Your order has shipped.", lang="en", voice="Ahana")  # a named voice from voices/voices.json

tts.register_voice("asha", ref_audio="asha.wav", ref_text="<exact transcript of asha.wav>", lang="hi")
wav = tts.generate("Your order has shipped.", lang="en", voice="asha")   # cross-lingual clone, cached prefix

for pcm in tts.stream("Long text ...", lang="te", voice="asha"):         # int16 chunks as they are generated
    ...

wav = tts.generate("Pay ₹1,250 by 5 pm.", lang="en", normalize=True)     # bundled normaliser: digits, currency, URLs
```

**Release directory.** `from_pretrained(dir)` resolves the directory with `maya3.hub.resolve`. Mimi and ECAPA are taken from the Hugging Face cache, or downloaded on first use. The directory holds:
- `model.safetensors`, or the shards plus `model.safetensors.index.json`;
- `maya3_config.json`;
- `backbone/`: the Qwen3 config and tokenizer only;
- `voices/voices.json`.

```python
from maya3.hub import resolve
resolve("/path/to/release/maya3")
# {"ckpt": ".../maya3", "backbone_dir": ".../maya3/backbone", "mimi_dir": "<kyutai/mimi snapshot>",
#  "ecapa_dir": "<speechbrain/spkrec-ecapa-voxceleb snapshot>", "voices_json": ... or None}
```

- `HF_HUB_OFFLINE=1` never touches the network.
- `MAYA3_MIMI_DIR` and `MAYA3_ECAPA_DIR` (or `resolve(..., mimi=, ecapa=)`) point at local copies.
- The explicit form still works: `Maya3TTS.from_pretrained("model.pt", backbone_dir=..., mimi_dir=..., ecapa_dir=...)`.

**Calls and voices.**
- `generate`, `stream` and `astream` are thread- and asyncio-safe. Concurrent calls are batched on the GPU.
- `ref_audio=`, `ref_text=` and `ref_lang=` on a call clone a voice once and cache it for the session. A registered voice is better.
- The reference clip should be 3-10 s long, with its exact transcript and language. Its language may differ from the target's, which gives cross-lingual cloning.

**Conditioning modes.**
- `icl_spk` (default with a reference): reference audio plus ECAPA;
- `icl`;
- `spk` (ECAPA only; no transcript needed);
- `plain` (no reference);
- `named` (a trained voice selected by `voice="Name"`, with no reference audio).

**Text.**
- Inline tags such as `[laughs]`, `[sighs]` and `[whispers]` are passed through.
- Long text is split at sentence boundaries into pieces of about 20 s. The pieces are streamed back-to-back with 150 ms gaps.

**Sampling per call.** The options are `seed`, `temperature`, `top_k`, `top_p` and `repetition_penalty`. With the same seed, text and voice, the audio is the same when the request runs alone.

### Converting a checkpoint into a release directory

```bash
python -m maya3.tools.convert_to_safetensors <model.pt> <out_dir> --backbone-src <backbone dir with config.json + tokenizer>
CUDA_VISIBLE_DEVICES= python -m maya3.tools.verify_conversion <model.pt> <out_dir>   # bit-identical check, CPU
```

- Dtypes are kept exactly as stored.
- The output is sharded at 5 GB.
- Tied tensors (`lm_head` and `embed_tokens`) are stored once and recorded as aliases.
- `maya3_config.json` records the resolved model config: speaker anchor, `depth_grouped` and the format version.
- `tts.model.load_s3.load_model_for_eval` loads the directory directly.

The backbone's own `model.safetensors` (7 GB) is never needed. When the backbone directory holds no weight files, the architecture is built from `config.json` and every weight comes from the Maya3 checkpoint. The result is bit-identical (`maya3/tests/test_backbone_from_config.py`). `MAYA3_BACKBONE_FROM_CONFIG=1` forces this path.

## Server

```bash
python -m maya3.server --ckpt /path/to/release/maya3 --port 8080 --max-seats 64
# or explicitly: --ckpt model.pt --backbone-dir BB --mimi-dir MIMI --ecapa-dir ECAPA [--voices clones.json]
curl -sN localhost:8080/v1/audio/speech -H 'content-type: application/json' \
     -d '{"input":"Hello there.","language":"en","voice":"plain","response_format":"wav"}' > out.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(model="maya3", voice="plain", input="Hello there.",
                                                         response_format="pcm") as r:   # s16le, 24 kHz mono
    for chunk in r.iter_bytes():
        ...
```

- `POST /v1/audio/speech` is OpenAI-compatible.
  - It streams wav, pcm, mp3 or opus. flac is sent once the audio is complete.
  - Text is normalised by default. Turn this off with `"normalize": false` on a request, or `--no-normalise` for the whole server.
- `WS /v1/tts/stream` streams with multiple contexts per socket and supports cancel.
- `GET/POST/DELETE /v1/voices` manages voices, and `GET /health` reports status.
- `--voices` takes a JSON file of reference-clip voices (`voice_id`, `ref_audio`, `ref_text`, `lang`, `mode`). Named voices come from the release's `voices/voices.json`.
- **Per-voice mode.** Each entry of `voices/voices.json` has a `mode`:
  - `named`: the trained `{Name}: text` prompt.
  - `spk`: the voice's stored ECAPA centroid, given by `ecapa` (an .npy path relative to the file).
  - `icl_spk`: the centroid plus a curated reference clip. `ref_clips` holds `[{codes: <.npy (32,F) Mimi codes>, text, lang}]`.
  - Callers use `voice="Name"` in every mode; the mode only changes how the voice is conditioned.
- `maya3.client` has a Python client for HTTP and WebSocket.
- The full wire contract is in the docstring of `maya3/server.py`.

## Hardware and memory

Measured with bf16 backbone, fp16 depth, KV capacity 2048 tokens per seat, warmed and serving requests. Values are from the torch allocator; add about 0.5 GB of CUDA context.

| max_seats | static weights | KV | steady GPU memory | peak during load |
|---:|---:|---:|---:|---:|
| 1 | 5.6 GB | 0.47 GB | 7.7 GB | 9.2 GB |
| 8 | 5.6 GB | 2.1 GB | 9.7 GB | 10.9 GB |
| 16 | 5.6 GB | 4.0 GB | 11.7 GB | 12.8 GB |
| 32 | 5.6 GB | 7.8 GB | 15.8 GB | 16.5 GB |

| GPU | measured streams at the SLA (TTFA p99 <= 300 ms, 0 playback stalls) | B=1 TTFA / RTF |
|---|---|---|
| H100 80 GB | 224 (capacity 1536) | 14 ms / 0.137 |
| RTX 4090 24 GB | 48 at capacity 2048 (TTFA p99 66 ms); 96 at capacity 1024 (TTFA p99 104 ms, 188 audio-s/s) | 15 ms / 0.154 |
| 16 GB card | about 16 at capacity 2048 (estimate from the memory table) | - |
| 12 GB card | 1-4 (estimate) | - |

- Everything runs in plain bf16; no quantisation is needed.
- KV is 112 KiB per token per seat, so `capacity=1024` halves the KV memory.
- The fp32 reference depth decoder (2.2 GB) is freed once the fp16 fast copy is built.

## Knobs (`EngineConfig`, passed as keyword arguments to `from_pretrained`)

| knob | default | meaning |
|---|---|---|
| `max_seats` | 16 | concurrent generations batched on the GPU. The CUDA graphs and KV are sized for it, so memory follows the table above. |
| `capacity` | 2048 | backbone KV tokens per seat (prompt plus generated frames, at 12.5 frames/s) |
| `depth_dtype` | `fp16` | depth decoder precision: `fp16`, `bf16` or `fp32`. fp16 is the closest to fp32 greedy. |
| `cb0_temp`, `cb0_top_k`, `cb0_top_p`, `cb0_rep`, `cb0_rep_win` | engine defaults | cb0 sampler: temperature, top-k, top-p, presence penalty and its window |
| `temp`, `top_k` | 0.7, 64 | depth sampler |
| `chunk_s`, `gap_s` | 20 s, 0.15 s | long-text piece length and the silence between pieces |
| `hard_cap` | 1500 | absolute frame ceiling per piece (120 s) |
| `busy_frac`, `busy_admit` | 0.75, 4 | admission smoothing under load. `--busy-frac 0.6` gave 0 stalls in a 30-min soak at 224 seats. |
| `guards` | on | `GuardConfig`: the duration cap and loop guard below |
| `seed` | fixed | engine RNG seed; per-request `seed=` overrides it |

## Engine design (`maya3/engine.py`)

Each tick produces one frame for every active seat:
1. **Admission.**
   - A new request's voice prefix (reference text plus codes) is constant per voice, so its backbone KV is computed once and copied into the request's slot.
   - Only the target text is prefilled. This happens *inside* the same graphed backbone step as the decode rows, so the first frame comes out in the admission tick.
2. **Generation.** The cb0 sampler and controlled stop run on the GPU. Then come the depth decoder (31 levels, one CUDA graph, fp16) and the Mimi one-frame decoder (one CUDA graph).
3. **Readout.** One device-to-host copy and one sync per tick. Guards and delivery then run on the host.

The prompt geometry comes from the training packer (`tts.data.packing_v2`). It was checked token for token against the reference eval harness.

Prompt builders are pluggable (`maya3/prompt.py`). Named speakers use `NamedSpeakerBuilder`, which renders `<|lang:xx|>{Name}: text<|audio|>`.

## Guards (on by default, zero latency, no configuration): `maya3/guards.py`

- **Duration cap.** Generation stops at 2.2 × the expected duration + 1.2 s. The expected duration comes from a digit-, pause- and language-aware prior.
- **Loop guard.** It watches for a cb0 stall or a periodic loop.
  - On a hit it applies an EOS boost of +4 on the AEOS logit, so the model ends the phrase itself.
  - If the loop persists 24 frames after the boost, a backstop stop follows.
  - It is tag-aware: it is suspended around inline `[tags]`.

## Measured (one H100 NVL, bf16 backbone, fp16 depth, guards on)

These engine measurements were taken on a pre-release checkpoint with the same architecture; speed does not depend on the weights. The release-1.0 numbers are in the README.

- **B=1:** TTFA 14 ms for plain voice and 15 ms for a cloned voice. This covers the prompt build, prefill, first frame, Mimi decode and the host copy. RTF is 0.137, about 7x faster than real time.
- **One frame:** 10.5 ms at B=1 (backbone 2.0, cb0 sampler 0.35, depth 6.5, Mimi 1.8), 23.7 ms at B=100 and 30 ms at B=160.
- **Server (WebSocket, closed loop, 0 playback stalls):**

  | lanes | frames | TTFA p99 | RTF p99 | throughput |
  |---:|---|---:|---:|---:|
  | 224 | base64 | 137 ms | 0.85 | 329 audio-s/s |
  | 256 | binary | 150 ms | 0.92 | 348 audio-s/s |

- **Server, 224 lanes, binary frames, by voice type:**
  - plain: TTFA p99 166 ms, 0 stalls, 342.6 audio-s/s;
  - cloned voice: TTFA p99 185 ms, 0 stalls, 317.0 audio-s/s.
  - Seats at SLA (TTFA p99 ≤ 300 ms, 0 stalls): **224 per H100**.
- **Quality:** ASR CER and clone similarity match the reference eval engine within seed noise.

## Benchmarks (`maya3/bench/`)

| script | what it does |
|---|---|
| `bench_engine.py` | latency and closed-loop throughput of the engine |
| `load_ws.py` | WebSocket load generator: client-side TTFA, stalls and RTF through the server |
