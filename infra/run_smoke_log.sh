#!/usr/bin/env bash
# OFA-VRP: Run modal smoke entrypoint; capture all stdout+stderr to logs/smoke_run.log.
#
# Usage:
#   bash infra/run_smoke_log.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
LOG="$LOG_DIR/smoke_run.log"

mkdir -p "$LOG_DIR"
: > "$LOG"

echo "=== run_smoke_log.sh: starting modal smoke ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ===" | tee -a "$LOG"
echo "    entrypoint : infra/modal_vrp_iter3_ctx.py::smoke" | tee -a "$LOG"
echo "    log        : $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

cd "$REPO_ROOT"
modal run "$SCRIPT_DIR/modal_vrp_iter3_ctx.py::smoke" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"
echo "=== run_smoke_log.sh: done (exit=$EXIT_CODE) — log: $LOG ===" | tee -a "$LOG"

exit "$EXIT_CODE"
