#!/usr/bin/env bash
set -euo pipefail

# Luma runs this long batch as a calm overnight comparison: same buckets, same horizon, only the experimental core changes.
# Luma 用这份长程批处理做稳定的过夜对比：同一批数据桶、同一推理 horizon，只改变实验核心。

ROOT="/home/kt/ai/minimind"
PY="/home/kt/ai/.venvs/luma-global/bin/python"
RUN="$ROOT/scripts/run_luma_stage12.py"
OUT_DIR="$ROOT/artifacts/longrun_10240"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

START_TS="$(date +%s)"
MAX_SECONDS=$((10 * 3600))
OPTIONAL_EXPD_THRESHOLD=$((7 * 3600))

COMMON_ARGS=(
  --device cuda
  --seq-len 256
  --samples 8
  --stage2-steps 10240
  --fixture-mode competition_math_dialogue_emotion
  --enable-persona-seed
  --enable-python-code
  --world-jepa-mode full
  --enable-self-check-ring
  --reason-shared-depth 2
  --rollout-steps 10
  --reason-loops 15
  --exit-two-step-aux-weight 0.25
)

run_case() {
  local tag="$1"
  shift
  local json_out="$OUT_DIR/${tag}.json"
  local metrics_out="$OUT_DIR/${tag}_metrics.jsonl"
  local log_out="$LOG_DIR/${tag}.log"
  echo "[$(date --iso-8601=seconds)] START ${tag}" | tee -a "$log_out"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$RUN" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --json-out "$json_out" \
    --metrics-out "$metrics_out" \
    2>&1 | tee -a "$log_out"
  echo "[$(date --iso-8601=seconds)] END ${tag}" | tee -a "$log_out"
}

run_case "iter2_10240" \
  --self-check-k 2

run_case "iter9_10240" \
  --self-check-k 2 \
  --world-mask-strategy structured \
  --world-full-simplify-loss \
  --self-world-coupling-weight 0.05 \
  --self-rollout-hierarchical

run_case "iter9_crystal_10240" \
  --self-check-k 3 \
  --world-mask-strategy structured \
  --world-full-simplify-loss \
  --self-world-coupling-weight 0.05 \
  --self-rollout-hierarchical \
  --enable-exit-jepa-crystal

ELAPSED=$(( $(date +%s) - START_TS ))
if (( ELAPSED < OPTIONAL_EXPD_THRESHOLD )); then
  run_case "expd_iter9_math_adapter_10240" \
    --self-check-k 2 \
    --world-mask-strategy structured \
    --world-full-simplify-loss \
    --self-world-coupling-weight 0.05 \
    --self-rollout-hierarchical \
    --enable-math-adapter-lane
fi

TOTAL=$(( $(date +%s) - START_TS ))
echo "[$(date --iso-8601=seconds)] BATCH DONE elapsed_seconds=${TOTAL} max_seconds=${MAX_SECONDS}" | tee -a "$LOG_DIR/batch.log"
