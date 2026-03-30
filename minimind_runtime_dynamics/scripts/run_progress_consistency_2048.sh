#!/usr/bin/env bash
set -euo pipefail
PY=/home/kt/ai/.venvs/luma-global/bin/python
SCRIPT=/home/kt/ai/minimind/scripts/run_luma_stage12.py
OUTDIR=/home/kt/ai/minimind/artifacts/progress_consistency_2048
LOGDIR=/home/kt/ai/minimind/logs/progress_consistency_2048
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
  --self-loop-awareness-mode predictor_progress
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
run_one baseline_predictor_2048
run_one progress_improve_2048 --self-progress-shape-weight 0.10
run_one progress_full_2048 --self-progress-shape-weight 0.10 --self-progress-trend-weight 0.05 --self-progress-plateau-weight 0.02
run_one local_smooth_2048 --self-local-delta-consistency-weight 0.05
run_one local_curvature_2048 --self-local-delta-consistency-weight 0.05 --self-local-curvature-weight 0.02
