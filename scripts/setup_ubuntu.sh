#!/usr/bin/env bash
# Create .venv and install the pinned environment on Ubuntu.
# Usage: bash scripts/setup_ubuntu.sh <CUDA_TAG|cpu>
#   CUDA_TAG: the tag from https://pytorch.org/get-started/locally/ (e.g. cu126, cu128); use 'cpu' without a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "usage: bash scripts/setup_ubuntu.sh <CUDA_TAG|cpu>   (see docs/INSTALL.md)" >&2
  exit 2
fi
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "$PY not found (sudo apt install python3.12 python3.12-venv)" >&2; exit 1; }

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.15.0 torchvision==0.29.2 --index-url "https://download.pytorch.org/whl/${TAG}"
pip install -r requirements/base.txt
pip install -e .
corelog env-report
echo
echo "Done. Next: source .venv/bin/activate && pytest && corelog smoke-test"
