#!/usr/bin/env python3
"""
OFA-Research Orchestrator — Overnight Strategy Search
======================================================
Submits overnight walk-forward jobs to RunPod, polls results.

Usage:
    python infra/orchestrator.py                         # defaults
    python infra/orchestrator.py --max-batches 1 --workers 1  # single test job
    python infra/orchestrator.py --dry-run               # preview without submitting

Requires:
    RUNPOD_API_KEY env var (or in .env)
    RUNPOD_ENDPOINT_ID env var (set after creating endpoint)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent

# ─── Config ─────────────────────────────────────────────────────────────────

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RESULTS_DIR = _REPO_ROOT / "results"

# Walk-forward start dates — overnight sessions exist from the start of
# futures data, no dependency on GEX/options chain availability.
START_DATES = [
    # 2022 — MES data starts 2022-02-14
    "2022-03-01",
    "2022-03-28",
    "2022-04-25",
    "2022-05-23",
    "2022-06-20",
    "2022-07-18",
    "2022-08-15",
    "2022-09-12",
    "2022-10-10",
    "2022-11-07",
    "2022-12-05",
    # 2023
    "2023-01-03",
    "2023-01-30",
    "2023-02-27",
    "2023-03-27",
    "2023-04-24",
    "2023-05-22",
    "2023-06-19",
    "2023-07-17",
    "2023-08-14",
    "2023-09-11",
    "2023-10-09",
    "2023-11-06",
    "2023-12-04",
    # 2024
    "2024-01-02",
    "2024-01-29",
    "2024-02-26",
    "2024-03-25",
    "2024-04-22",
    "2024-05-20",
    "2024-06-17",
    "2024-07-15",
    "2024-08-12",
    "2024-09-09",
    "2024-10-07",
    "2024-11-04",
    "2024-12-02",
    # 2025
    "2025-01-06",
    "2025-02-03",
    "2025-03-03",
    "2025-04-07",
    "2025-05-05",
    "2025-06-02",
    "2025-06-30",
    "2025-07-28",
    "2025-08-25",
    "2025-09-22",
    "2025-10-20",
    "2025-11-03",
]

POLL_INTERVAL_SEC = 15
MAX_POLL_DURATION_SEC = 7200  # 2 hour max wait per job

# Render progress endpoint
RENDER_URL = os.environ.get("RENDER_DATA_URL", "https://gap-autotrader-cxt8.onrender.com")


# ─── RunPod API ─────────────────────────────────────────────────────────────

def submit_job(start_date: str, meta_config: dict | None = None,
               asset: str = "MES") -> str:
    """Submit an overnight walk-forward job to RunPod. Returns job ID."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"

    payload = {
        "input": {
            "job": "llm_walkforward",
            "asset": asset,
            "wf_window": 60,
            "wf_max_regime": 180,
            "wf_start": start_date,
            "n_strategies": 10,
            "n_refinements": 10,
            "max_dd_dollars": 2000,
            "use_seeds": False,
            "use_genes": True,
            "max_regimes": 6,
        }
    }

    if meta_config:
        payload["input"]["meta_config"] = meta_config

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    job_id = result.get("id", "")
    print(f"  Submitted overnight job: {asset} start={start_date} → {job_id}")
    return job_id


def poll_job(job_id: str) -> tuple[str, dict | None]:
    """Check job status. Returns (status, output_or_none)."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    status = result.get("status", "UNKNOWN")
    output = result.get("output") if status == "COMPLETED" else None
    return status, output


def cancel_job(job_id: str) -> bool:
    """Cancel a running RunPod job."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/cancel/{job_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        return result.get("status") == "CANCELLED"
    except Exception as e:
        print(f"  [cancel] Failed to cancel {job_id[:8]}: {e}")
        return False


def query_regime_progress(job_id: str) -> dict | None:
    """Query Render for regime-level progress of a job."""
    url = f"{RENDER_URL}/regime-progress?job_id={job_id}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def save_result(job_id: str, output: dict) -> Path:
    """Save job result to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"overnight_{job_id}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


# ─── Orchestration Loop ────────────────────────────────────────────────────

def orchestrate(
    max_batches: int = 10,
    workers_per_batch: int = 3,
    dry_run: bool = False,
    start_index: int = 0,
    asset: str = "MES",
    start_date: str | None = None,
    dates: list[str] | None = None,
):
    """Streaming orchestration — keeps workers_per_batch jobs in flight."""
    if not RUNPOD_API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        return
    if not RUNPOD_ENDPOINT_ID:
        print("ERROR: Set RUNPOD_ENDPOINT_ID environment variable")
        return

    max_jobs = max_batches * workers_per_batch
    pool_size = workers_per_batch

    print(f"Starting OVERNIGHT orchestrator [{asset}]: "
          f"max_jobs={max_jobs}, pool_size={pool_size}")

    # Load meta-config (static JSON, no meta-optimizer)
    cfg_path = _REPO_ROOT / "results" / "meta_config.json"
    meta_dict = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            meta_dict = json.load(f)
        print(f"  Loaded meta-config from {cfg_path}")
    else:
        print(f"  No meta_config.json — jobs will use handler defaults")

    # Date selection
    date_list = dates or []
    date_list_idx = 0
    date_idx = start_index

    if date_list:
        max_jobs = len(date_list)
        print(f"  Using explicit date list: {len(date_list)} dates")

    def next_date():
        nonlocal date_idx, date_list_idx
        if date_list:
            if date_list_idx < len(date_list):
                sd = date_list[date_list_idx]
                date_list_idx += 1
                return sd
            return date_list[0]
        if start_date:
            return start_date
        if date_idx < start_index + len(START_DATES):
            sd = START_DATES[date_idx % len(START_DATES)]
            date_idx += 1
            return sd
        # Shuffle after first pass
        sd = random.choice(START_DATES)
        return sd

    # Submit initial pool
    in_flight = {}
    total_submitted = 0
    total_completed = 0

    def submit_next():
        nonlocal total_submitted
        if total_submitted >= max_jobs:
            return
        sd = next_date()
        if dry_run:
            print(f"  [DRY RUN] Would submit overnight {sd}")
            total_submitted += 1
            return
        try:
            job_id = submit_job(sd, meta_dict, asset=asset)
            in_flight[job_id] = {"start_date": sd, "submitted_at": time.time()}
            total_submitted += 1
        except Exception as e:
            print(f"  FAILED to submit {sd}: {e}")
            total_submitted += 1

    print(f"\n{'='*60}")
    print(f"Filling pool with {pool_size} initial overnight jobs")
    print(f"{'='*60}")
    for _ in range(pool_size):
        submit_next()

    if dry_run:
        while total_submitted < max_jobs:
            submit_next()
        return

    # Poll loop
    overall_start = time.time()
    while in_flight or total_submitted < max_jobs:
        if not in_flight:
            for _ in range(min(pool_size, max_jobs - total_submitted)):
                submit_next()
            if not in_flight:
                break

        time.sleep(POLL_INTERVAL_SEC)
        ts = datetime.now().strftime("%H:%M:%S")

        completed_this_round = []
        for job_id in list(in_flight.keys()):
            info = in_flight[job_id]
            elapsed = time.time() - info["submitted_at"]

            if elapsed > MAX_POLL_DURATION_SEC:
                print(f"  {ts} TIMEOUT {info['start_date']} after {elapsed/60:.0f}min")
                cancel_job(job_id)
                del in_flight[job_id]
                continue

            try:
                status, output = poll_job(job_id)
            except Exception as e:
                print(f"  {ts} Poll error for {job_id[:8]}: {e}")
                continue

            if status == "COMPLETED" and output:
                if output.get("error"):
                    print(f"  {ts} ERROR {info['start_date']}: {output['error']}")
                    if output.get("traceback"):
                        for line in output["traceback"].strip().split("\n")[-3:]:
                            print(f"    {line}")
                    del in_flight[job_id]
                    continue

                path = save_result(job_id, output)
                pnl = output.get("total_pnl", "?")
                wr = output.get("aggregate_win_rate", "?")
                n_reg = output.get("total_regimes", "?")
                n_trades = output.get("total_trades", "?")
                job_min = elapsed / 60
                print(f"  {ts} DONE {info['start_date']} ({job_min:.0f}min): "
                      f"P&L=${pnl} WR={wr}% regimes={n_reg} trades={n_trades}")

                completed_this_round.append(path)
                del in_flight[job_id]
                total_completed += 1

            elif status == "FAILED":
                error = output.get("error", "unknown") if output else "unknown"
                print(f"  {ts} FAILED {info['start_date']}: {error}")
                del in_flight[job_id]

            else:
                mins = elapsed / 60
                progress = query_regime_progress(job_id)
                regime_info = ""
                if progress and progress.get("regimes_done", 0) > 0:
                    regime_info = f" [{progress['regimes_done']} regimes done]"
                print(f"  {ts} {info['start_date']}: {status} ({mins:.0f}min){regime_info}")

        # Backfill pool
        while len(in_flight) < pool_size and total_submitted < max_jobs:
            submit_next()

    total_min = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"OVERNIGHT Completed [{asset}]: {total_completed} jobs in {total_min:.0f}min")
    print(f"  Submitted: {total_submitted}")
    print(f"{'='*60}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OFA-Research Overnight Orchestrator")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--asset", type=str, default="MES")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override: all jobs use this start date")
    parser.add_argument("--dates", type=str, default=None,
                        help="Comma-separated dates (overrides max-batches)")
    args = parser.parse_args()

    date_list = [d.strip() for d in args.dates.split(",")] if args.dates else None

    orchestrate(
        max_batches=args.max_batches,
        workers_per_batch=args.workers,
        dry_run=args.dry_run,
        start_index=args.start_index,
        asset=args.asset,
        start_date=args.start_date,
        dates=date_list,
    )


if __name__ == "__main__":
    main()
