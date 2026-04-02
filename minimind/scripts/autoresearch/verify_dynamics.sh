#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
exec /home/kt/ai/.venvs/luma-global/bin/python scripts/run_dynamics_candidate_eval.py \
  --json-out artifacts/autoresearch_dynamics/latest.json \
  --metrics-out artifacts/autoresearch_dynamics/latest_metrics.jsonl \
  --summary-out artifacts/autoresearch_dynamics/latest_summary.json
