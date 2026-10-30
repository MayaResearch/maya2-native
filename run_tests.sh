#!/usr/bin/env bash
# CPU test suite: no GPU, no weights, no network. Needs the [dev] extras.
set -euo pipefail
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES= exec "${PYTHON:-python}" -m pytest -q -p no:cacheprovider \
  maya3/tests/test_text.py maya3/tests/test_text_cli.py maya3/tests/test_guards.py maya3/tests/test_server.py "$@"
