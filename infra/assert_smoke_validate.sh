#!/usr/bin/env bash
# assert_smoke_validate.sh
#
# Asserts smoke_test.jsonl has at least 1 non-empty line, then runs
# validate_smoke_test.py and asserts it prints PASSED and exits 0.
#
# Usage:
#   bash infra/assert_smoke_validate.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JSONL="$REPO_ROOT/current/options_vrp/results/options_vrp/smoke_test.jsonl"

# ── 1. Assert smoke_test.jsonl exists ────────────────────────────────────────
if [ ! -f "$JSONL" ]; then
    echo "FAILED: smoke_test.jsonl not found: $JSONL" >&2
    exit 1
fi

# ── 2. Assert at least 1 non-empty line ──────────────────────────────────────
NON_EMPTY=$(grep -c '[^[:space:]]' "$JSONL" || true)
if [ "$NON_EMPTY" -lt 1 ]; then
    echo "FAILED: expected at least 1 non-empty line in smoke_test.jsonl, got $NON_EMPTY" >&2
    exit 1
fi
echo "ASSERT OK: smoke_test.jsonl has $NON_EMPTY non-empty line(s)"

# ── 3. Run validate_smoke_test.py and capture output + exit code ──────────────
VALIDATE_OUT=$(python3 "$SCRIPT_DIR/validate_smoke_test.py" 2>&1)
VALIDATE_EXIT=$?

echo "$VALIDATE_OUT"

if [ "$VALIDATE_EXIT" -ne 0 ]; then
    echo "FAILED: validate_smoke_test.py exited $VALIDATE_EXIT" >&2
    exit 1
fi

# ── 4. Assert output contains PASSED ─────────────────────────────────────────
if ! echo "$VALIDATE_OUT" | grep -q "PASSED"; then
    echo "FAILED: validate_smoke_test.py output did not contain 'PASSED'" >&2
    exit 1
fi

echo ""
echo "=== assert_smoke_validate.sh: ALL ASSERTIONS PASSED ==="
exit 0
