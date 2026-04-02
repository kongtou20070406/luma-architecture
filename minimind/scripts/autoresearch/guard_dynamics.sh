#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
exec /home/kt/ai/.venvs/luma-global/bin/python scripts/check_dynamics_guard.py \
  --summary artifacts/autoresearch_dynamics/latest_summary.json
