#!/usr/bin/env bash
set -euo pipefail
PY=/home/kt/ai/.venvs/luma-global/bin/python
SCRIPT=/home/kt/ai/minimind/scripts/run_luma_stage12.py
OUTDIR=/home/kt/ai/minimind/artifacts/loopaware_2048
LOGDIR=/home/kt/ai/minimind/logs/loopaware_2048
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
  local mode="$1"
  local tag="$2"
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" "$SCRIPT" "${COMMON[@]}" \
    --self-loop-awareness-mode "$mode" \
    --json-out "$OUTDIR/${tag}.json" \
    --metrics-out "$OUTDIR/${tag}_metrics.jsonl" \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "DONE $tag $(date --iso-8601=seconds)"
}
run_one none loopaware_none_2048
run_one ct_progress loopaware_ct_progress_2048
run_one predictor_progress loopaware_predictor_progress_2048
run_one dual_phase loopaware_dual_phase_2048
