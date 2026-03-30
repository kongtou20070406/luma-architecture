#!/usr/bin/env bash
set -euo pipefail
cd /home/kt/ai/minimind
exec /home/kt/ai/.venvs/luma-global/bin/python scripts/check_dynamics_guard.py \
  --summary artifacts/autoresearch_dynamics/latest_summary.json
