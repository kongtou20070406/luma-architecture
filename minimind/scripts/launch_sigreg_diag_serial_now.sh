#!/usr/bin/env bash
set -u

ROOT="/home/kt/ai/luma-architecture/minimind"
PROGRAM="$ROOT/luma_stage0/dynamics_autoresearch_program.json"
PY="/home/kt/ai/.venvs/luma-global/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/artifacts/autoresearch_sigreg_diag_serial_now_${STAMP}"

mkdir -p "$OUTDIR"/{reports,metrics,summaries,logs}
RUNTIME="$OUTDIR/diag-runtime.json"
STATE="$OUTDIR/diag-state.json"
RESULTS="$OUTDIR/diag-results.tsv"
MASTER_LOG="$OUTDIR/diag-master.log"

echo -e "case\tcandidate\tstatus\tsummary_path\treport_path\tmetrics_path\tfirst_nonfinite_summary\tfirst_nonfinite_report" > "$RESULTS"
echo "{\"status\":\"running\",\"ts\":$(date +%s),\"outdir\":\"$OUTDIR\"}" > "$STATE"
echo "[$(date)] start serial diag run" >> "$MASTER_LOG"

run_case() {
  local case_name="$1"
  local candidate="$2"
  local extra1="${3:-}"
  local extra2="${4:-}"

  local base="sigreg_diag__${case_name}"
  local report="$OUTDIR/reports/${base}.json"
  local metrics="$OUTDIR/metrics/${base}.jsonl"
  local summary="$OUTDIR/summaries/${base}.json"
  local l2json="$OUTDIR/reports/${base}.layer2.json"
  local l2md="$OUTDIR/reports/${base}.layer2.md"
  local l2csv="$OUTDIR/reports/${base}.layer2"
  local log="$OUTDIR/logs/${base}.log"

  echo "{\"status\":\"running\",\"case\":\"$case_name\",\"candidate\":\"$candidate\",\"ts\":$(date +%s)}" > "$RUNTIME"
  echo "[$(date)] RUN $case_name $candidate" | tee -a "$MASTER_LOG" >> "$log"

  if [[ -n "$extra1" && -n "$extra2" ]]; then
    "$PY" "$ROOT/scripts/run_dynamics_candidate_eval.py" \
      --program "$PROGRAM" \
      --candidate "$candidate" \
      --stage2-steps 4096 \
      --json-out "$report" \
      --metrics-out "$metrics" \
      --summary-out "$summary" \
      --layer2-json-out "$l2json" \
      --layer2-md-out "$l2md" \
      --layer2-csv-prefix "$l2csv" \
      --extra-arg "$extra1" \
      --extra-arg "$extra2" >> "$log" 2>&1
  elif [[ -n "$extra1" ]]; then
    "$PY" "$ROOT/scripts/run_dynamics_candidate_eval.py" \
      --program "$PROGRAM" \
      --candidate "$candidate" \
      --stage2-steps 4096 \
      --json-out "$report" \
      --metrics-out "$metrics" \
      --summary-out "$summary" \
      --layer2-json-out "$l2json" \
      --layer2-md-out "$l2md" \
      --layer2-csv-prefix "$l2csv" \
      --extra-arg "$extra1" >> "$log" 2>&1
  else
    "$PY" "$ROOT/scripts/run_dynamics_candidate_eval.py" \
      --program "$PROGRAM" \
      --candidate "$candidate" \
      --stage2-steps 4096 \
      --json-out "$report" \
      --metrics-out "$metrics" \
      --summary-out "$summary" \
      --layer2-json-out "$l2json" \
      --layer2-md-out "$l2md" \
      --layer2-csv-prefix "$l2csv" >> "$log" 2>&1
  fi

  local rc=$?
  "$PY" - <<PY >> "$log" 2>&1
import json, pathlib
summary = pathlib.Path("$summary")
report = pathlib.Path("$report")
s = json.loads(summary.read_text()) if summary.exists() else {}
r = json.loads(report.read_text()) if report.exists() else {}
s_v = s.get("first_nonfinite_step")
r_v = r.get("stage2", {}).get("first_nonfinite_step")
status = "ok" if (summary.exists() and report.exists()) else "missing_artifact"
print("\\t".join([
  "$case_name",
  "$candidate",
  status,
  str(summary),
  str(report),
  "$metrics",
  repr(s_v),
  repr(r_v),
]))
PY
  tail -n 1 "$log" >> "$RESULTS"
  echo "[$(date)] DONE $case_name rc=$rc" | tee -a "$MASTER_LOG" >> "$log"
}

run_case "m0_a2_online" "A2-progress_shape_v1-h3+progress_exit_readout+m0_a2_cosine_sigreg_online" "--seq-len 1024"
run_case "m0_a2_encoder" "A2-progress_shape_v1-h3+progress_exit_readout+m0_a2_cosine_sigreg_encoder" "--seq-len 1024"
run_case "m0_a2_online_fp32off" "A2-progress_shape_v1-h3+progress_exit_readout+m0_a2_cosine_sigreg_online" "--seq-len 1024" "--no-sigreg-world-fp32-only"

echo "{\"status\":\"done\",\"ts\":$(date +%s),\"outdir\":\"$OUTDIR\"}" > "$STATE"
echo "{\"status\":\"done\",\"ts\":$(date +%s),\"outdir\":\"$OUTDIR\"}" > "$RUNTIME"
echo "[$(date)] serial diag complete" >> "$MASTER_LOG"

