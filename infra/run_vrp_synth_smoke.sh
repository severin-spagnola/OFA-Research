#!/usr/bin/env bash
# OFA-VRP: synthetic end-to-end smoke.
# Generates synthetic parquet chain files inside the container, calls
# evaluate_window(), and asserts ok=True. No Volume data required.
#
# Usage:
#   bash infra/run_vrp_synth_smoke.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG=/tmp/modal_vrp_synth_smoke.log

: > "$LOG"

echo "=== run_vrp_synth_smoke.sh: starting synthetic smoke ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "    entrypoint : infra/modal_vrp_synth_smoke.py"
echo "    log        : $LOG"
echo ""

cd "$REPO_ROOT"
modal run "$SCRIPT_DIR/modal_vrp_synth_smoke.py" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "ERROR: modal run failed (exit=$EXIT_CODE). Log: $LOG" >&2
fi

echo "=== run_vrp_synth_smoke.sh: done (exit=$EXIT_CODE) ==="
exit "$EXIT_CODE"
