# Changelog

## 1.0 (2026-10-25)

First public release.

- Maya3 weights on Hugging Face: [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native) (bf16 safetensors).
- 12 languages: as bn en gu hi kn ml mr or pa ta te.
- 20 named voices, plus zero-shot and cross-lingual voice cloning.
- Streaming engine: continuous batching, CUDA graphs, cached voice prefixes, built-in duration and loop guards.
- Python API (`generate`, `stream`, `astream`), an OpenAI-compatible HTTP server and a multi-context WebSocket server.
- Bundled text normaliser for numbers, currency, dates, times, URLs and e-mail addresses in every language.
