#!/usr/bin/env bash
# OFA-VRP: noop_window smoke sentinel.
# Verifies image build, sys.path, and wf_runner imports inside the container.
# No actual compute is performed.
#
# Usage:
#   bash infra/run_vrp_noop_smoke.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG=/tmp/modal_vrp_noop_smoke.log

: > "$LOG"

echo "=== run_vrp_noop_smoke.sh: starting noop_window smoke sentinel ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "    entrypoint : infra/modal_vrp_noop_smoke.py"
echo "    log        : $LOG"
echo ""

cd "$REPO_ROOT"
modal run "$SCRIPT_DIR/modal_vrp_noop_smoke.py" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "ERROR: modal run failed (exit=$EXIT_CODE). Log: $LOG" >&2
fi

echo "=== run_vrp_noop_smoke.sh: done (exit=$EXIT_CODE) — sentinel written to current/options_vrp/results/options_vrp/regime_db.jsonl ==="
exit "$EXIT_CODE"
