"""
VRP Walk-Forward Completion Hook
Waits for PID 18801 to finish, then:
  1. Copies result files to ATLAS state dir
  2. Generates summary CSV
  3. POSTs Discord notification
"""
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error

PID = 18801
REGIME_DB = "/Users/severinspagnola/Desktop/OFA-Research/current/options_vrp/results/options_vrp/regime_db.jsonl"
LOG_FILE = "/Users/severinspagnola/Desktop/OFA-Research/current/options_vrp/vrp_full_run.log"
ATLAS_STATE = "/Users/severinspagnola/Desktop/ATLAS/atlas/state/OFA-VRP"
SUMMARY_CSV = os.path.join(ATLAS_STATE, "vrp_walkforward_summary.csv")
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1483594072005541900/UxUEWxoOtauGF9jHo1uB6mNzLjpA4uvwimA6nfZew2Fm_ioWIhR63D2Hm4jQ1ic-v7aN"

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it

def wait_for_pid(pid, poll_interval=10):
    print(f"[hook] Waiting for PID {pid} to finish...", flush=True)
    while is_running(pid):
        time.sleep(poll_interval)
    print(f"[hook] PID {pid} has exited.", flush=True)

def copy_results():
    os.makedirs(ATLAS_STATE, exist_ok=True)
    copied = []
    for src in [REGIME_DB, LOG_FILE]:
        if os.path.exists(src):
            dst = os.path.join(ATLAS_STATE, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"[hook] Copied {src} -> {dst}", flush=True)
        else:
            print(f"[hook] WARNING: {src} not found, skipping copy.", flush=True)
    return copied

def load_records():
    records = []
    if not os.path.exists(REGIME_DB):
        print(f"[hook] ERROR: {REGIME_DB} not found.", flush=True)
        return records
    with open(REGIME_DB) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[hook] WARN: skipping bad JSON line: {e}", flush=True)
    return records

def generate_summary_csv(records):
    os.makedirs(ATLAS_STATE, exist_ok=True)
    rows = []
    for r in records:
        train_fitness = r.get("train_fitness", {})
        fitness = train_fitness.get("fitness", -99.0) if isinstance(train_fitness, dict) else -99.0
        rows.append({
            "window_id": r.get("window_id", ""),
            "best_gene": r.get("genes_desc", ""),
            "fitness": fitness,
            "val_sharpe": r.get("val_sharpe", ""),
            "fwd_profitable": r.get("forward_profitable", ""),
            "n_val_trades": r.get("val_n_trades", ""),
        })
    rows.sort(key=lambda x: (x["window_id"] if isinstance(x["window_id"], int) else -1))
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "best_gene", "fitness", "val_sharpe", "fwd_profitable", "n_val_trades"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hook] Wrote summary CSV: {SUMMARY_CSV} ({len(rows)} rows)", flush=True)
    return rows

def post_discord(n_total, n_profitable, results_path):
    msg = (
        f"**OFA-VRP Walk-Forward Run COMPLETE** \n"
        f"Windows: {n_total} total, **{n_profitable} forward-profitable**\n"
        f"Results: `{results_path}`"
    )
    payload = json.dumps({"content": msg}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[hook] Discord notification sent (HTTP {resp.status})", flush=True)
    except urllib.error.URLError as e:
        print(f"[hook] WARNING: Discord POST failed: {e}", flush=True)

def main():
    wait_for_pid(PID)
    copy_results()
    records = load_records()
    print(f"[hook] Loaded {len(records)} records from regime_db.jsonl", flush=True)
    rows = generate_summary_csv(records)
    n_total = len(rows)
    n_profitable = sum(1 for r in rows if r.get("fwd_profitable") is True or r.get("fwd_profitable") == "True")
    post_discord(n_total, n_profitable, ATLAS_STATE)
    print(f"[hook] Done. {n_total} windows, {n_profitable} profitable.", flush=True)

if __name__ == "__main__":
    main()
