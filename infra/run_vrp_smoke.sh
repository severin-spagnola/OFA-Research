#!/usr/bin/env bash
# OFA-VRP: Smoke-test dispatch (2 real volume windows, iter3-ctx).
#
# Invokes modal_vrp_iter3_ctx.py::smoke via `modal run`, which calls
# dispatch(smoke=True) — capped at 2 windows from real volume data.
# Output is written to current/options_vrp/results/options_vrp/smoke_test.jsonl.
# Logs merged stdout+stderr to /tmp/modal_vrp_smoke_real.log and tees to terminal.
# Exits non-zero if modal run fails.
#
# Usage:
#   bash infra/run_vrp_smoke.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG=/tmp/modal_vrp_smoke_real.log

: > "$LOG"

echo "=== run_vrp_smoke.sh: starting smoke dispatch (iter3-ctx) ==="

cd "$REPO_ROOT"
modal run "$SCRIPT_DIR/modal_vrp_iter3_ctx.py::smoke" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "ERROR: modal run failed (exit=$EXIT_CODE). Log: $LOG" >&2
fi

echo ""
echo "=== run_vrp_smoke.sh: done (exit=$EXIT_CODE) — output: current/options_vrp/results/options_vrp/smoke_test.jsonl ==="
exit "$EXIT_CODE"
