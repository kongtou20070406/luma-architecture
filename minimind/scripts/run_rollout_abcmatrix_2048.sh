#!/usr/bin/env bash
set -euo pipefail
PY=/home/kt/ai/.venvs/luma-global/bin/python
ROOT=/home/kt/ai/minimind
OUT=$ROOT/artifacts/rollout_abcmatrix_2048
LOG=$ROOT/logs/rollout_abcmatrix_2048
mkdir -p "$OUT" "$LOG"
COMMON=(
  --device cuda
  --fixture-mode competition_math_dialogue_emotion
  --enable-persona-seed
  --enable-python-code
  --seq-len 256
  --samples 8
  --stage2-steps 2048
  --world-jepa-mode full
  --enable-self-check-ring
  --self-check-k 2
  --reason-shared-depth 2
  --rollout-steps 10
  --reason-loops 15
  --exit-two-step-aux-weight 0.25
  --self-loop-awareness-mode predictor_progress
  --self-progress-shape-weight 0.10
  --self-progress-trend-weight 0.05
  --self-progress-plateau-weight 0.02
)
run_case() {
  local tag="$1"; shift
  echo "[$(date --iso-8601=seconds)] START $tag"
  "$PY" "$ROOT/scripts/run_luma_stage12.py" \
    "${COMMON[@]}" "$@" \
    --json-out "$OUT/${tag}.json" \
    --metrics-out "$OUT/${tag}_metrics.jsonl" \
    > "$LOG/${tag}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE $tag"
}
run_case baseline_progress_full_2048
run_case horizon3_2048 --rollout-steps 3 --self-rollout-supervision-horizon 3
run_case horizon4_2048 --rollout-steps 4 --self-rollout-supervision-horizon 4
run_case near3_weighted_2048 --self-rollout-weighting-mode near3
run_case self_span_mask_2048 --self-feature-span-mask-ratio 0.10
