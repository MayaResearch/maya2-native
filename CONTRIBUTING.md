# Contributing

Thanks for helping improve Maya3.

- **Bugs and feature requests:** open a GitHub issue. Include the language, the voice or cloning mode, the exact text, the seed, the GPU, and the `maya3` commit. A short audio clip helps.
- **Pull requests:** keep them small and focused, and describe what changed and how you tested it.
  - Run the CPU tests before sending: `./run_tests.sh`.
  - Changes to the engine's hot path (`maya3/engine.py`, `serve/fast_*.py`) should come with before/after numbers from `maya3/bench/bench_engine.py` on a GPU.
- **Text normaliser:** `maya3/text/tn/` is a verbatim vendored copy, pinned by sha256 in `test_text.py`. Put fixes in `maya3/text/rules.py` or `maya3/text/lexicon.py`, with a test case.
- **Style:** Python 3.10+, standard library first, no new heavy dependencies without discussion.
- **Security / misuse reports:** do not open a public issue. See [SECURITY.md](SECURITY.md).
- By contributing you agree that your contribution is licensed under the repository's license.
