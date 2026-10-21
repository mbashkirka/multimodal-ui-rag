#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This bootstrap script is intended for macOS."
  exit 1
fi

choose_python() {
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)' >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10-3.12 was not found."
  echo "Recommended: brew install python@3.12"
  exit 1
fi

echo "Using $PYTHON_BIN: $($PYTHON_BIN --version)"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv"
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'

# Do not import FAISS and PyTorch in the same diagnostic interpreter on Apple Silicon.
python -c 'import platform, open_clip, torch; print("Python:", platform.python_version()); print("PyTorch:", torch.__version__); print("OpenCLIP:", getattr(open_clip, "__version__", "unknown")); print("MPS available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))'
python -c 'import faiss; print("FAISS:", getattr(faiss, "__version__", "unknown"))'
