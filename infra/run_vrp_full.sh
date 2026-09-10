#!/usr/bin/env bash
# OFA-VRP: Full walk-forward dispatch (all windows, no smoke test).
#
# Invokes modal_vrp.py via `modal run` (local_entrypoint / dispatch mode).
#
# Usage:
#   bash infra/run_vrp_full.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG=/tmp/modal_run.log

: > "$LOG"

echo "=== run_vrp_full.sh: starting full VRP dispatch ==="

cd "$REPO_ROOT"
modal run "$SCRIPT_DIR/modal_vrp.py::dispatch" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== /tmp/modal_run.log ==="
cat "$LOG" || true
echo "=== end of log ==="
echo ""

echo "=== run_vrp_full.sh: done (exit=$EXIT_CODE) ==="
exit "$EXIT_CODE"
