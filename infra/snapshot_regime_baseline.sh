#!/usr/bin/env bash
# Snapshot pre-run baseline of regime_db.jsonl.
# Writes two lines to /tmp/regime_baseline.txt:
#   count=N
#   max_window_id=YYYY-MM-DD
#
# max_window_id is the val_end date of the record with the highest window_id
# (window_id is an integer; val_end is the corresponding YYYY-MM-DD date string).
#
# Usage:
#   bash infra/snapshot_regime_baseline.sh

set -euo pipefail

JSONL="current/options_vrp/results/options_vrp/regime_db.jsonl"

python3 - <<'EOF' | tee /tmp/regime_baseline.txt
import json

path = "current/options_vrp/results/options_vrp/regime_db.jsonl"
records = [json.loads(line) for line in open(path)]
count = len(records)
max_rec = max(records, key=lambda r: r["window_id"])
max_window_id = max_rec["val_end"]
print(f"count={count}")
print(f"max_window_id={max_window_id}")
EOF
