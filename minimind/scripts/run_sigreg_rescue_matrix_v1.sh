#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kt/ai/luma-architecture/minimind"
PY="/home/kt/ai/.venvs/luma-global/bin/python"
PROGRAM="$ROOT/luma_stage0/dynamics_autoresearch_program.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/artifacts/autoresearch_sigreg_rescue_v1_${STAMP}"

mkdir -p "$OUTDIR"/{reports,metrics,summaries,logs}
RUNTIME="$OUTDIR/rescue-runtime.json"
RESULTS="$OUTDIR/rescue-results.tsv"

cat > "$RESULTS" <<'TSV'
case	candidate	status	summary_path	pod_rank	pod_top1	dmd_radius	forcing_top_abs_corr	guard_all_ok	score	mixed_self_tail	arc_agi_self_tail
TSV

write_runtime() {
  local status="$1"
  local case_name="${2:-}"
  local candidate="${3:-}"
  cat > "$RUNTIME" <<JSON
{"status":"$status","case":"$case_name","candidate":"$candidate","ts":$(date +%s),"outdir":"$OUTDIR"}
JSON
}

append_result_from_summary() {
  local case_name="$1"
  local candidate="$2"
  local summary="$3"
  "$PY" - <<PY >> "$RESULTS"
import json, pathlib
s_path = pathlib.Path("$summary")
if not s_path.exists():
    print("\\t".join(["$case_name","$candidate","missing_summary",str(s_path),"","","","","","","",""]))
else:
    s = json.loads(s_path.read_text())
    layer2 = s.get("layer2", {})
    guard = s.get("guard", {})
    bucket = s.get("bucket_scores", {})
    mixed_self = bucket.get("mixed", {}).get("self_loss_tail")
    arc_self = bucket.get("arc_agi", {}).get("self_loss_tail")
    print("\\t".join([
        "$case_name",
        "$candidate",
        "ok",
        str(s_path),
        str(layer2.get("pod_effective_rank")),
        str(layer2.get("pod_top1_energy_ratio")),
        str(layer2.get("dmd_spectral_radius")),
        str(layer2.get("forcing_top_abs_corr")),
        str(guard.get("all_ok")),
        str(s.get("score")),
        str(mixed_self),
        str(arc_self),
    ]))
PY
}

run_case() {
  local case_name="$1"
  local candidate="$2"
  shift 2
  local extra_args=("$@")

  local base="rescue_v1__${case_name}"
  local report="$OUTDIR/reports/${base}.json"
  local metrics="$OUTDIR/metrics/${base}.jsonl"
  local summary="$OUTDIR/summaries/${base}.json"
  local log="$OUTDIR/logs/${base}.log"
  local l2json="$OUTDIR/reports/${base}.layer2.json"
  local l2md="$OUTDIR/reports/${base}.layer2.md"
  local l2csv="$OUTDIR/reports/${base}.layer2"

  write_runtime "running" "$case_name" "$candidate"
  echo "[$(date)] RUN $case_name $candidate" | tee -a "$log"

  cmd=(
    "$PY" "$ROOT/scripts/run_dynamics_candidate_eval.py"
    --program "$PROGRAM"
    --candidate "$candidate"
    --stage2-steps 2048
    --json-out "$report"
    --metrics-out "$metrics"
    --summary-out "$summary"
    --layer2-json-out "$l2json"
    --layer2-md-out "$l2md"
    --layer2-csv-prefix "$l2csv"
  )

  for arg in "${extra_args[@]}"; do
    cmd+=(--extra-arg "$arg")
  done

  # Keep memory conservative so this can run alongside the main chain.
  cmd+=(--extra-arg "--seq-len 256")
  cmd+=(--extra-arg "--samples 4")

  if "${cmd[@]}" >> "$log" 2>&1; then
    append_result_from_summary "$case_name" "$candidate" "$summary"
    echo "[$(date)] DONE $case_name" | tee -a "$log"
  else
    append_result_from_summary "$case_name" "$candidate" "$summary"
    echo "[$(date)] FAIL $case_name" | tee -a "$log"
  fi
}

# R1: reduce over-regularization pressure; keep med sigreg on encoder source.
run_case \
  "r1_balanced_med" \
  "A2-progress_shape_v1-h3+progress_exit_readout+m1_full_regularizers_from_a4e_sigreg_med" \
  "--self-jepa-weight 0.4" \
  "--self-rollout-weight 0.2" \
  "--exit-aux-weight 0.003" \
  "--rollout-zone-weight 0.005" \
  "--routing-tier-entropy-weight 0.002" \
  "--routing-min-local-share-weight 0.002" \
  "--trajectory-vitality-weight 0.005"

# R2: keep self+rollout, disable exit/extra regularizers to test decoupled training pressure.
run_case \
  "r2_self_rollout_lite" \
  "A2-progress_shape_v1-h3+progress_exit_readout+m1_self_rollout_from_a4e" \
  "--self-jepa-weight 0.6" \
  "--self-rollout-weight 0.2" \
  "--exit-aux-weight 0.0" \
  "--rollout-zone-weight 0.0" \
  "--routing-tier-entropy-weight 0.0" \
  "--routing-min-local-share-weight 0.0" \
  "--trajectory-vitality-weight 0.0"

# R3: same as R1 but gentler world sigreg + longer warmup.
run_case \
  "r3_med_warmup_long" \
  "A2-progress_shape_v1-h3+progress_exit_readout+m1_full_regularizers_from_a4e_sigreg_med" \
  "--world-sigreg-weight 0.015" \
  "--sigreg-world-warmup-steps 1024" \
  "--self-jepa-weight 0.4" \
  "--self-rollout-weight 0.2" \
  "--exit-aux-weight 0.003" \
  "--rollout-zone-weight 0.005" \
  "--routing-tier-entropy-weight 0.002" \
  "--routing-min-local-share-weight 0.002" \
  "--trajectory-vitality-weight 0.005"

# R4: world-only reference with encoder latent sigreg, to verify rank recovery baseline.
run_case \
  "r4_world_only_encoder" \
  "A2-progress_shape_v1-h3+progress_exit_readout+m0_a2_cosine_sigreg_encoder" \
  "--world-sigreg-weight 0.015" \
  "--sigreg-world-warmup-steps 1024"

write_runtime "done"
echo "Rescue v1 done. Results: $RESULTS"

