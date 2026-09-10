#!/usr/bin/env bash
# Completion hook for wf_runner PID 40153 (vrp_full_run2, 44 windows)
WATCH_PID=40153
RESULTS_DIR="/Users/severinspagnola/Desktop/OFA-Research/current/options_vrp/results/options_vrp"
LOG_FILE="/Users/severinspagnola/Desktop/OFA-Research/current/options_vrp/vrp_full_run2.log"
ATLAS_STATE="/Users/severinspagnola/Desktop/ATLAS/atlas/state/OFA-VRP"
DISCORD_WEBHOOK="https://discord.com/api/webhooks/1483594072005541900/UxUEWxoOtauGF9jHo1uB6mNzLjpA4uvwimA6nfZew2Fm_ioWIhR63D2Hm4jQ1ic-v7aN"
EXPECTED_WINDOWS=44

echo "[hook] Monitoring PID $WATCH_PID for $EXPECTED_WINDOWS-window walk-forward..."
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 30
done

echo "[hook] PID $WATCH_PID has exited. Running completion steps."

# Copy files to ATLAS state dir
mkdir -p "$ATLAS_STATE"
cp "$RESULTS_DIR/regime_db.jsonl" "$ATLAS_STATE/regime_db.jsonl" 2>/dev/null && echo "[hook] Copied regime_db.jsonl" || echo "[hook] WARNING: regime_db.jsonl not found"
cp "$LOG_FILE" "$ATLAS_STATE/vrp_full_run2.log" 2>/dev/null && echo "[hook] Copied vrp_full_run2.log"

# Parse regime_db.jsonl with Python and write summary CSV + Discord notification
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 - <<PYEOF
import json
import csv
import os
import urllib.request

REGIME_DB = "$RESULTS_DIR/regime_db.jsonl"
CSV_OUT = "$ATLAS_STATE/vrp_walkforward_summary.csv"
DISCORD_WEBHOOK = "$DISCORD_WEBHOOK"
EXPECTED_WINDOWS = $EXPECTED_WINDOWS

records = []
if os.path.exists(REGIME_DB):
    with open(REGIME_DB) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"[hook] Skipping bad line: {e}")

# Deduplicate by window_id (keep last)
by_window = {}
for r in records:
    wid = r.get("window_id")
    if wid is not None:
        by_window[wid] = r

unique_windows = sorted(by_window.keys())
n_unique = len(unique_windows)
print(f"[hook] regime_db.jsonl: {len(records)} lines, {n_unique} unique window_ids (expected {EXPECTED_WINDOWS})")

fwd_profitable_count = 0
rows = []
for wid in unique_windows:
    r = by_window[wid]
    fwd_profitable = r.get("fwd_profitable", 0)
    if fwd_profitable:
        fwd_profitable_count += 1
    rows.append({
        "window_id": wid,
        "best_gene": str(r.get("best_gene", "")),
        "fitness": r.get("fitness", ""),
        "val_sharpe": r.get("val_sharpe", ""),
        "fwd_profitable": fwd_profitable,
        "n_val_trades": r.get("n_val_trades", r.get("val_trades", "")),
    })

# Write CSV
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["window_id", "best_gene", "fitness", "val_sharpe", "fwd_profitable", "n_val_trades"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[hook] Wrote CSV with {n_unique} windows: {CSV_OUT}")

# Build Discord message
if n_unique < EXPECTED_WINDOWS:
    msg = (
        f":warning: **OFA-VRP Walk-Forward COMPLETE (WARNING)**\n"
        f"Expected {EXPECTED_WINDOWS} windows but only {n_unique} unique window_ids in regime_db.jsonl.\n"
        f"Forward profitable windows: {fwd_profitable_count}/{n_unique}\n"
        f"CSV written with available data. Manual inspection recommended."
    )
else:
    msg = (
        f":white_check_mark: **OFA-VRP Walk-Forward COMPLETE**\n"
        f"Total windows: {n_unique}/{EXPECTED_WINDOWS}\n"
        f"Forward profitable: {fwd_profitable_count}/{n_unique}\n"
        f"Results copied to ATLAS state. CSV written."
    )

payload = json.dumps({"content": msg}).encode("utf-8")
req = urllib.request.Request(
    DISCORD_WEBHOOK,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"[hook] Discord POST status: {resp.status}")
except Exception as e:
    print(f"[hook] Discord POST failed: {e}")
PYEOF

echo "[hook] All completion steps done."
