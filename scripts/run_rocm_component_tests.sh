#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/code/maya/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set PYTHON=/absolute/path/to/your/ROCm/venv/bin/python" >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/scripts/run_rocm_component_tests.py" "$@"
