#!/usr/bin/env bash
set -euo pipefail

# Local launch defaults tuned for a single RTX 5090 under WSL.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo ".venv not found. Run ./setup_5090_wsl.sh first."
  exit 1
fi

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_ID="${RUN_ID:-wsl_5090_baseline}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# H100 baseline uses 524288 tokens/batch, which is often too aggressive for local single-GPU runs.
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}"
  echo "Run ./setup_5090_wsl.sh (or download data manually) first."
  exit 1
fi

echo "Starting train_gpt.py on RTX 5090 (WSL)..."
echo "RUN_ID=${RUN_ID}"
echo "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}, TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}"

torchrun --standalone --nproc_per_node=1 train_gpt.py "$@"
