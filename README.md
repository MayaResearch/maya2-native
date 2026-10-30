# Maya2 TTS

Streaming multilingual text-to-speech for **11 Indic languages and English**, with 20 built-in voices and
zero-shot voice cloning. This repository is the **inference engine**: a continuous-batching CUDA-graph engine, a
Python API, an OpenAI-compatible HTTP server and a WebSocket streaming server. The weights are on Hugging Face at
**[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native)**.

**[Samples](https://huggingface.co/maya-research/maya2-native#samples) · [Model card](https://huggingface.co/maya-research/maya2-native) · [Engine docs](docs/ENGINE.md) · [Text normaliser](maya3/text/README.md)**

> The Python package and model are called **Maya3**; `maya2-native` is the repository name.

- **Languages:** as bn en gu hi kn ml mr or pa ta te.
- **Streaming:** about 14 ms to first audio at B=1 on an H100. One H100 NVL serves **224 concurrent streams** at **323 s of audio per second**, with steady-state TTFA p99 **242 ms** and 0 playback stalls.
- **Voices:** 20 named voices (`voice="Ahana"`), or clone a voice from 3-10 s of audio.
- **Text normaliser (bundled):** numbers, currency, dates, URLs and ALL-CAPS names are spoken correctly in every language.

## Install

You need Python 3.10-3.12, Linux and an NVIDIA GPU with a CUDA 12.8 driver (8 GB or more of GPU memory).

```bash
git clone https://github.com/MayaResearch/maya2-native maya3 && cd maya3
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server]" --extra-index-url https://download.pytorch.org/whl/cu128
```

The pinned runtime is torch 2.8.0 (cu128, triton 3.4.0), transformers 5.6.2 and speechbrain 1.1.2. Weights, the
Mimi codec and the ECAPA speaker encoder are downloaded from Hugging Face on first use. Set `HF_HUB_OFFLINE=1` to
use only the local cache.

## Quickstart

### Python

```python
from maya3 import Maya3TTS
import soundfile as sf

tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=16)

wav = tts.generate("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।", lang="hi", voice="Ahana")
sf.write("out.wav", wav, 24000)                       # float32, 24 kHz mono

for pcm in tts.stream("Your OTP is 482913.", lang="en", voice="Siddhant", normalize=True):
    ...                                                # int16 PCM chunks, as they are generated
```

`generate`, `stream` and `astream` are thread-safe and asyncio-safe. Concurrent calls are batched on the GPU.

### Command line

```bash
python examples/synthesize.py "నమస్కారం, మీరు ఎలా ఉన్నారు?" --lang te --voice Sreekar -o out.wav
python examples/synthesize.py --list-voices
```

### HTTP server (OpenAI-compatible) and WebSocket streaming

```bash
python -m maya3.server --ckpt maya-research/maya2-native --host 0.0.0.0 --port 8080 --max-seats 64
```

```bash
curl -sN localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Your order has shipped.","language":"en","voice":"Ahana","response_format":"wav"}' > out.wav
```

```python
from openai import OpenAI          # any OpenAI-compatible client
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
        model="maya3", voice="Ahana", input="Hello from Maya3.", response_format="pcm") as r:   # s16le, 24 kHz
    for chunk in r.iter_bytes():
        ...
```

- `POST /v1/audio/speech` streams `wav`, `pcm`, `mp3` or `opus`; `flac` is sent once the audio is complete.
- `WS /v1/tts/stream` carries several concurrent contexts per socket, with cancel and binary frames. See `examples/ws_stream.py` and the protocol in the docstring of `maya3/server.py`.
- `GET/POST/DELETE /v1/voices` manages cloned voices, and `GET /health` reports status.
- Text is normalised by default. Turn this off with `"normalize": false` per request, or `--no-normalise` for the whole server.
- `maya3.client` has a small Python client for HTTP and WebSocket.

## Preview text normalization

Want to inspect how numbers, dates or currency will be spoken before generating
audio? [Preview text normalization locally](maya3/text/README.md#preview-normalization-without-a-gpu)
with `python -m maya3.text --lang hi --text 'मेरे पास 25 रुपये हैं'`.
It needs only Python, with no GPU or weights.

## Voices

| voice | gender | age | languages (>= 1 h trained) | description |
|---|---|---|---|---|
| Ahana | female | young | bn en gu hi kn ml mr or pa ta te | Young female voice, bright, youthful |
| Vaidehi | female | adult | bn en gu hi kn ml mr or pa ta te | Adult female voice, calm, gentle |
| Malini | female | mature | bn en gu hi kn ml mr or pa ta te | Mature female voice, steady narrator |
| Charvi | female | young | bn en gu hi kn ml mr or pa ta te | Young female voice, energetic, upbeat |
| Jahnavi | female | adult | bn en gu hi kn ml mr or ta te | Adult female voice, warm, friendly |
| Anirban | male | adult | bn en gu hi kn ml mr or pa ta te | Adult male voice, steady narrator |
| Nakul | male | young | bn en gu hi kn ml mr or pa ta te | Young male voice, bright, youthful |
| Vasant | male | mature | bn en gu hi kn ml mr or ta te | Mature male voice, steady narrator |
| Siddhant | male | adult | bn en gu hi kn ml mr or pa ta te | Adult male voice, clear, professional |
| Satvik | male | young | bn en gu hi kn ml mr or pa ta te | Young male voice, warm, friendly |
| Abhay | male | adult | en hi | Adult male voice, natural conversational style; trained on English, Hindi |
| Nivedita | female | adult | en hi | Adult female voice, natural conversational style; trained on English, Hindi |
| Ishwari | female | adult | hi | Adult female voice, natural conversational style; trained on Hindi |
| Ritvik | male | adult | hi | Adult male voice, natural conversational style; trained on Hindi |
| Vrinda | female | young | hi | Young female voice, natural conversational style; trained on Hindi |
| Sreekar | male | adult | en hi te | Adult male voice, natural conversational style; trained on English, Hindi, Telugu |
| Kayal | female | adult | en ta | Adult female voice, natural conversational style; trained on English, Tamil |
| Shaunak | male | young | en hi | Young male voice, natural conversational style; trained on English, Hindi |
| Akshat | male | adult | en hi | Adult male voice, natural conversational style; trained on English, Hindi |
| Tanushree | female | adult | en hi | Adult female voice, natural conversational style; trained on English, Hindi |

- Every voice can speak all 12 languages. Languages outside its trained list are spoken cross-lingually.
- Names are case-insensitive. `tts.voices` (a property) and `GET /v1/voices` list them.
- The voices are synthetic personas and must not be presented as real people.

**Named-voice prompt format.** A named voice is selected by its name in the prompt: the engine renders
`<|lang:xx|>{Name}: <text><|audio|>`. Send only your text with `voice="Name"` (Python) or `"voice": "Name"` (HTTP/WS).
Do not type `{Name}: ` yourself, and do not combine a named voice with a reference clip. Each voice's conditioning
mode is in `voices/voices.json`:
- `named` uses the trained name prompt;
- `spk` uses a stored speaker embedding.

## Voice cloning

```python
# 3-10 s of clean speech + its exact transcript and language. Only clone voices you have consent to use.
tts.register_voice("my_voice", ref_audio="ref.wav", ref_text="<exact transcript of ref.wav>", lang="hi")
wav = tts.generate("வணக்கம், எப்படி இருக்கிறீர்கள்?", lang="ta", voice="my_voice")   # cross-lingual

# one-off (cached for the session):
wav = tts.generate("Hello!", lang="en", ref_audio="ref.wav", ref_text="...", ref_lang="hi")
```

There are three conditioning modes for a reference:
- `icl_spk` (default): the reference audio as an in-context prompt, plus its ECAPA embedding;
- `icl`: the reference audio only;
- `spk`: the ECAPA embedding only, so no transcript is needed.

The reference language can differ from the target language. Over HTTP, `POST /v1/voices` registers a clone (see
`examples/clone_voice.py`). Zero-shot clones are recognisably close to the reference but not identical.

## Languages

| code | language | CER % (Qwen3-ASR, release bench) |
|---|---|---:|
| `as` | Assamese | 6.59 |
| `bn` | Bengali | 2.83 |
| `en` | English | 1.74 |
| `gu` | Gujarati | 3.76 |
| `hi` | Hindi | 2.05 |
| `kn` | Kannada | 2.72 |
| `ml` | Malayalam | 4.20 |
| `mr` | Marathi | 2.09 |
| `or` | Odia | 4.33 |
| `pa` | Punjabi | 3.32 |
| `ta` | Tamil | 4.55 |
| `te` | Telugu | 2.64 |

Overall CER on the 6,000-clip release bench is 3.40 % (Qwen3-ASR). Real failures, where two ASRs agree, are
0.15 %. The model card describes the method in full.

## Streaming performance

The whole release bench (6000 texts, 20.3 h of audio) was served by `maya3.server` with its
defaults, as 224 concurrent WebSocket streams on **one H100 NVL**:

| metric | value |
|---|---|
| throughput | 323 audio-s/s |
| TTFA, steady state, p50 / p90 / p99 | 111 / 134 / 242 ms |
| TTFA, whole run incl. ramp and drain, p50 / p99 | 115 / 774 ms |
| playback stalls / errors | 0 / 0 |
| B=1 (single stream, H100) | TTFA about 14 ms, RTF about 0.13 |

GPU memory, with a bf16 backbone, an fp16 depth decoder and 2048 KV tokens per seat (add about 0.5 GB of CUDA context):

| `max_seats` | steady | peak |
|---:|---:|---:|
| 1 | 7.7 GB | 9.2 GB |
| 8 | 9.7 GB | 10.9 GB |
| 16 | 11.7 GB | 12.8 GB |
| 32 | 15.8 GB | 16.5 GB |

An RTX 4090 (24 GB) serves 48 streams at `capacity=2048` (TTFA p99 66 ms), or 96 at `capacity=1024`. Engine knobs,
design notes and the conversion tools are in [docs/ENGINE.md](docs/ENGINE.md). The text normaliser is described in
[maya3/text/README.md](maya3/text/README.md).

## Limitations

- **Assamese is the weakest language**, with the highest CER and the least stable voice identity. No voice has much Assamese training audio.
- **Very short prompts (1-2 words)** can drift in voice identity. Prefer a slightly longer phrase for short agent turns.
- **Code-mixed and mixed-script text** has the highest error rates, and English words inside Indic sentences vary by voice.
- **URLs, symbols and long digit strings** need the normaliser (on by default in the server, `normalize=True` in Python).
- **Clone similarity** is recognisably close but not identical. Noisy or very short references reduce it.
- **Inline emotion tags** (`[laughs]`, `[whispers]`, ...) are best effort.
- **Sampling.** Output is sampled. The same seed is reproducible only when the request runs alone.

## Responsible use

Do not use Maya3 to impersonate real people, to clone a voice without the speaker's consent, for fraud, or to defeat
voice authentication. Tell listeners when speech is synthetic. To report misuse or a security problem, see
[SECURITY.md](SECURITY.md) (contact: TBD).

## Repository layout

| path | content |
|---|---|
| `maya3/` | public API (`api.py`), streaming engine (`engine.py`), HTTP/WS server (`server.py`), client, guards, text normaliser, tools, tests |
| `serve/` | CUDA-graph components: fused backbone step, fp16 depth decoder, one-frame Mimi decoder, KV cache |
| `tts/` | model definition and the training-time prompt packer the engine reproduces |
| `scripts/` | prompt and stop-control helpers shared with the model code |
| `configs/` | model config and the duration prior read by the engine |
| `examples/` | command-line synthesis, OpenAI client, WebSocket streaming, voice cloning |
| `docs/ENGINE.md` | engine internals, knobs, memory and latency measurements, checkpoint conversion |

## Tests

The CPU tests need no GPU and no weights:

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu128
./run_tests.sh            # text, normalization CLI, guards, and fake-engine server checks
```

The GPU checks (`maya3/tests/test_backbone_*.py`, `test_depth_fused.py`, `test_rope_kernel.py`, `test_sampling.py`)
are scripts that take a checkpoint path. See their docstrings.

## License

Apache-2.0. See `LICENSE`. The model weights have their own license on the Hugging Face model card. Maya3 builds on
Qwen3, Mimi (Kyutai) and SpeechBrain ECAPA-TDNN; see their licences.

## Citation

```bibtex
@misc{maya3tts,
  title        = {Maya3: Streaming Multilingual Text-to-Speech for Indic Languages},
  author       = {Maya Research},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/maya-research/maya2-native}}
}
```

GitHub's "Cite this repository" button reads `CITATION.cff`. Changes are listed in [CHANGELOG.md](CHANGELOG.md).
