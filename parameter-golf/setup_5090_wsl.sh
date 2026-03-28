#!/usr/bin/env bash
set -euo pipefail

# One-time setup for local WSL training on a single RTX 5090.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
DATA_VARIANT="${DATA_VARIANT:-sp1024}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install NVIDIA drivers + WSL CUDA support first."
  exit 1
fi

echo "[1/4] Creating virtual environment at ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[2/4] Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Reinstall torch from the official CUDA wheels to avoid CPU-only builds.
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

echo "[3/4] Checking CUDA visibility from PyTorch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA. Check WSL GPU setup.")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo "[4/4] Downloading FineWeb cache (variant=${DATA_VARIANT}, train-shards=${TRAIN_SHARDS})"
python data/cached_challenge_fineweb.py --variant "${DATA_VARIANT}" --train-shards "${TRAIN_SHARDS}"

cat <<'EOF'

Setup complete.
Next run:
  source .venv/bin/activate
  ./run_5090_wsl.sh
EOF
