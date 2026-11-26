#!/usr/bin/env bash
# Create the project environment in ./.venv (uv if available, else python -m venv + pip).
#
#   environment/setup_env.sh            # CPU / Apple-silicon (MPS) build of PyTorch
#   environment/setup_env.sh --cuda 12.4  # NVIDIA GPU machine: CUDA wheels from download.pytorch.org
#
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA=""
PY="${PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
TORCH_INDEX=()
if [[ -n "$CUDA" ]]; then
  TORCH_INDEX=(--index-url "https://download.pytorch.org/whl/cu${CUDA//./}")
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY" .venv
  uv pip install --python .venv/bin/python "${TORCH_INDEX[@]}" torch
  uv pip install --python .venv/bin/python -e ".[ml,dev]"
else
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install "${TORCH_INDEX[@]}" torch
  .venv/bin/pip install -e ".[ml,dev]"
fi

.venv/bin/python - <<'EOF'
import torch, cardiomag
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"cardiomag ready: torch {torch.__version__}, device {dev}")
EOF
