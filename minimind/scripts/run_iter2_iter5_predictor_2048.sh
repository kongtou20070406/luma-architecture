#!/usr/bin/env bash
set -euo pipefail
PY=/home/kt/ai/.venvs/luma-global/bin/python
SCRIPT=/home/kt/ai/minimind/scripts/run_luma_stage12.py
OUTDIR=/home/kt/ai/minimind/artifacts/iter2_iter5_predictor_2048
LOGDIR=/home/kt/ai/minimind/logs/iter2_iter5_predictor_2048
mkdir -p "$OUTDIR" "$LOGDIR"
COMMON=(
  --device cuda
  --seq-len 256
  --samples 8
  --stage2-steps 2048
  --fixture-mode competition_math_dialogue_emotion
  --enable-persona-seed
  --enable-python-code
  --world-jepa-mode full
  --enable-self-check-ring
  --self-check-k 2
  --reason-shared-depth 2
  --rollout-steps 10
  --reason-loops 15
  --exit-two-step-aux-weight 0.25
)
run_one() {
  local tag="$1"; shift
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" "$SCRIPT" "${COMMON[@]}" "$@" \
    --json-out "$OUTDIR/${tag}.json" \
    --metrics-out "$OUTDIR/${tag}_metrics.jsonl" \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "DONE $tag $(date --iso-8601=seconds)"
}
run_one iter2_2048 --self-loop-awareness-mode none
run_one iter5_2048 --self-loop-awareness-mode none --enable-math-adapter-lane
run_one iter2_predictor_progress_2048 --self-loop-awareness-mode predictor_progress
run_one iter5_predictor_progress_2048 --self-loop-awareness-mode predictor_progress --enable-math-adapter-lane
