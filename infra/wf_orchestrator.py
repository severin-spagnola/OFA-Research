"""
Walk-Forward Orchestrator
==========================
Dispatches one RunPod job per walk-forward window, polls for completion,
and aggregates results into results/options_wf/regime_db.jsonl.

Each job gets a specific (train_days, val_days, fwd_days) list. The forward
period is all remaining trading days after val — kill conditions on the worker
determine when the forward test ends.

Usage:
    python infra/wf_orchestrator.py
    python infra/wf_orchestrator.py --train-days 45 --val-days 15 --candidates 500
    python infra/wf_orchestrator.py --diagnostic
    python infra/wf_orchestrator.py --max-concurrent 8
    python infra/wf_orchestrator.py --start-window 15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time as tm
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load env from Strat repo
_STRAT_ENV = Path.home() / "Desktop" / "Strat" / ".env"
if _STRAT_ENV.exists():
    load_dotenv(_STRAT_ENV)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
WF_ENDPOINT_ID = os.environ.get("WF_ENDPOINT_ID",
                                 os.environ.get("OPTIONS_ENDPOINT_ID", "5vpdn34inq2tty"))


DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1483594072005541900/UxUEWxoOtauGF9jHo1uB6mNzLjpA4uvwimA6nfZew2Fm_ioWIhR63D2Hm4jQ1ic-v7aN"


def _send_discord(content: str) -> None:
    """Fire-and-forget Discord webhook notification."""
    try:
        import urllib.request
        data = json.dumps({"content": content[:2000], "username": "ATLAS"}).encode()
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL, data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[orch] Discord webhook failed: {e}", flush=True)


def submit_job(endpoint_id: str, payload: dict) -> dict:
    """Submit a job to RunPod serverless endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json={"input": payload}, headers=headers, timeout=60)
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt < 2:
                print(f"    (timeout on submit, retry {attempt+1}/2)")
                tm.sleep(5)
            else:
                raise


def check_job(endpoint_id: str, job_id: str) -> dict:
    """Check job status."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=30)
    return resp.json()


def generate_windows_from_days(
    all_days: list[date],
    train_size: int,
    val_size: int,
    step_size: int,
) -> list[dict]:
    """Generate walk-forward windows as serializable dicts.

    Forward days = everything after val. Kill conditions on the worker
    determine the actual forward period length per strategy.
    """
    windows = []
    min_needed = train_size + val_size
    min_fwd_days = 5
    i = 0
    window_id = 0

    while i + min_needed <= len(all_days):
        train_days = [d.isoformat() for d in all_days[i : i + train_size]]
        val_days = [d.isoformat() for d in all_days[i + train_size : i + min_needed]]
        fwd_days = [d.isoformat() for d in all_days[i + min_needed :]]

        if len(fwd_days) < min_fwd_days:
            break

        windows.append({
            "window_id": window_id,
            "train_days": train_days,
            "val_days": val_days,
            "fwd_days": fwd_days,
        })
        window_id += 1
        i += step_size

    return windows


def get_available_days_local(data_dir: Path, chain_subdir: str = "options_5dte") -> list[date]:
    """Get available trading days from local data."""
    chain_dir = data_dir / chain_subdir
    und_dir = data_dir / "underlying"

    if chain_dir.exists() and und_dir.exists():
        chain_dates = set()
        for f in chain_dir.glob("*.parquet"):
            name = f.stem
            if name.endswith("_meta"):
                continue
            parts = name.split("_")
            for p in parts:
                try:
                    chain_dates.add(date.fromisoformat(p))
                except ValueError:
                    continue

        und_dates = set()
        for f in und_dir.glob("*.parquet"):
            parts = f.stem.split("_")
            for p in parts:
                try:
                    und_dates.add(date.fromisoformat(p))
                except ValueError:
                    continue

        common = sorted(chain_dates & und_dates)
        if len(common) >= 50:
            return common

    print("[orch] Local data not found — submitting diagnostic job to get available days...",
          flush=True)
    return []


def main():
    parser = argparse.ArgumentParser(description="Walk-forward orchestrator for RunPod")
    parser.add_argument("--train-days", type=int, default=45)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: random)")
    parser.add_argument("--endpoint", default=WF_ENDPOINT_ID)
    parser.add_argument("--max-concurrent", type=int, default=0,
                        help="Max jobs in flight (0 = submit all at once)")
    parser.add_argument("--start-window", type=int, default=0,
                        help="Skip windows before this ID (for resuming)")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local data dir for window generation")
    parser.add_argument("--output", default="results/options_wf/regime_db.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date-list", type=str, default=None,
                        help="Path to JSON file with list of available dates")
    parser.add_argument("--skip-all-kills", action="store_true",
                        help="Disable all kill conditions (forward runs to data_end)")
    args = parser.parse_args()

    # Auto-generate seed if not specified
    if args.seed is None:
        args.seed = random.randint(10000, 99999)
        print(f"[orch] Auto-generated seed: {args.seed}")

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    endpoint = args.endpoint

    if args.diagnostic:
        print("Submitting diagnostic job...", flush=True)
        result = submit_job(endpoint, {"job": "diagnostic"})
        if "id" in result:
            _poll_single(endpoint, result["id"])
        else:
            print(json.dumps(result, indent=2))
        return

    # Get available days for window generation
    all_days = []
    if args.date_list:
        with open(args.date_list) as f:
            all_days = sorted(date.fromisoformat(s) for s in json.load(f))
    elif args.data_dir:
        all_days = get_available_days_local(Path(args.data_dir))

    if not all_days:
        print("[orch] Fetching available dates via diagnostic job...", flush=True)
        diag_result = submit_job(endpoint, {"job": "diagnostic"})
        if "id" not in diag_result:
            print(f"ERROR: {diag_result}")
            sys.exit(1)

        diag_output = _poll_single(endpoint, diag_result["id"])
        if not diag_output:
            print("ERROR: Diagnostic job returned no output")
            sys.exit(1)

        date_range = diag_output.get("date_range", [])
        if len(date_range) == 2:
            from datetime import timedelta
            start = date.fromisoformat(date_range[0])
            end = date.fromisoformat(date_range[1])
            d = start
            while d <= end:
                if d.weekday() < 5:
                    all_days.append(d)
                d += timedelta(days=1)
            n_expected = diag_output.get("available_days", 0)
            print(f"[orch] Generated {len(all_days)} weekdays from range "
                  f"(server has {n_expected} actual trading days)", flush=True)
            print(f"[orch] NOTE: Using weekday approximation. Holidays will produce "
                  f"empty results on server (harmless).", flush=True)
        else:
            print("ERROR: Could not determine date range from diagnostic")
            sys.exit(1)

    # Generate windows
    windows = generate_windows_from_days(
        all_days, args.train_days, args.val_days, args.step,
    )

    # Apply start_window filter
    windows = [w for w in windows if w["window_id"] >= args.start_window]

    if not windows:
        print("No windows to evaluate")
        return

    print(f"\nWalk-Forward Orchestrator")
    print(f"  Endpoint:       {endpoint}")
    print(f"  Train window:   {args.train_days} days")
    print(f"  Val window:     {args.val_days} days")
    print(f"  Forward:        kill-gated (all remaining days)")
    print(f"  Step size:      {args.step} days")
    print(f"  Total windows:  {len(windows)}")
    print(f"  Candidates:     {args.candidates}/window")
    print(f"  Top N:          {args.top_n}")
    print(f"  Min trades:     {args.min_trades}")
    print(f"  Seed base:      {args.seed}")
    print(f"  Output:         {args.output}")
    print(f"  First window:   train [{windows[0]['train_days'][0]}→{windows[0]['train_days'][-1]}] "
          f"val [{windows[0]['val_days'][0]}→{windows[0]['val_days'][-1]}] "
          f"fwd {len(windows[0]['fwd_days'])}d avail")
    print(f"  Last window:    train [{windows[-1]['train_days'][0]}→{windows[-1]['train_days'][-1]}] "
          f"val [{windows[-1]['val_days'][0]}→{windows[-1]['val_days'][-1]}] "
          f"fwd {len(windows[-1]['fwd_days'])}d avail")
    print()

    # Submit and poll
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        print(f"[orch] --overwrite: clearing {output_path}", flush=True)
        output_path.unlink()

    # Dedup
    existing_wids = set()
    if output_path.exists() and not args.overwrite:
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_wids.add(json.loads(line).get("window_id", -1))
                    except json.JSONDecodeError:
                        pass
        if existing_wids:
            pre_count = len(windows)
            windows = [w for w in windows if w["window_id"] not in existing_wids]
            print(f"[orch] Dedup: {pre_count - len(windows)} windows already in output, "
                  f"{len(windows)} remaining", flush=True)

    if not windows:
        print("All windows already processed. Use --overwrite to re-run.")
        return

    if args.max_concurrent > 0:
        _run_throttled(endpoint, windows, args, output_path)
    else:
        _run_all_at_once(endpoint, windows, args, output_path)

    # Count final results and notify Discord
    n_regimes = 0
    if output_path.exists():
        with open(output_path) as f:
            n_regimes = sum(1 for line in f if line.strip())
    output_name = output_path.name
    _send_discord(
        f"**[OFA] wf_orchestrator done** — `{output_name}`\n"
        f"{n_regimes} regimes written. step={args.step}, candidates={args.candidates}. "
        f"Ready for pool refresh."
    )


def _build_payload(window: dict, args) -> dict:
    """Build RunPod job payload for a single window."""
    payload = {
        "job": "vrp_walkforward",
        "window_id": window["window_id"],
        "train_days": window["train_days"],
        "val_days": window["val_days"],
        "fwd_days": window["fwd_days"],
        "n_candidates": args.candidates,
        "top_n": args.top_n,
        "min_trades": args.min_trades,
        "seed": args.seed + window["window_id"] * 137,
        "chain_subdir": "options_5dte",
    }
    if getattr(args, "skip_all_kills", False):
        payload["skip_all_kills"] = True
    return payload


def _run_all_at_once(endpoint: str, windows: list[dict], args, output_path: Path):
    """Submit all windows at once, poll until all complete."""
    jobs = []
    print(f"Submitting {len(windows)} jobs...", flush=True)

    for w in windows:
        payload = _build_payload(w, args)
        result = submit_job(endpoint, payload)
        job_id = result.get("id", "?")
        jobs.append({
            "id": job_id,
            "window_id": w["window_id"],
            "status": "submitted",
            "train_range": f"{w['train_days'][0]}→{w['train_days'][-1]}",
        })
        if len(jobs) % 10 == 0:
            print(f"  Submitted {len(jobs)}/{len(windows)}", flush=True)

    print(f"\n  {len(jobs)} jobs submitted. Polling...\n", flush=True)
    _poll_jobs(endpoint, jobs, output_path)


def _run_throttled(endpoint: str, windows: list[dict], args, output_path: Path):
    """Submit windows with concurrency limit."""
    max_concurrent = args.max_concurrent
    pending_windows = list(windows)
    active_jobs = []
    completed = 0
    total = len(windows)

    print(f"Running {total} windows with max {max_concurrent} concurrent...\n", flush=True)

    processed_job_ids: set[str] = set()
    while pending_windows or active_jobs:
        while pending_windows and len(active_jobs) < max_concurrent:
            w = pending_windows.pop(0)
            payload = _build_payload(w, args)
            result = submit_job(endpoint, payload)
            job_id = result.get("id", "?")
            active_jobs.append({
                "id": job_id,
                "window_id": w["window_id"],
                "train_range": f"{w['train_days'][0]}→{w['train_days'][-1]}",
            })
            print(f"  Submitted window {w['window_id']} (job={job_id})", flush=True)

        if not active_jobs:
            break

        tm.sleep(15)
        still_active = []
        for job in active_jobs:
            result = check_job(endpoint, job["id"])
            status = result.get("status", "unknown")

            if status == "COMPLETED":
                if job["id"] in processed_job_ids:
                    continue
                processed_job_ids.add(job["id"])
                completed += 1
                output = result.get("output", {})
                _process_completed_window(job, output, output_path, completed, total)
            elif status in ("FAILED", "CANCELLED"):
                completed += 1
                error = result.get("output", {}).get("error", result.get("error", status))
                print(f"  {status} window {job['window_id']}: {error}", flush=True)
            else:
                still_active.append(job)

        active_jobs = still_active
        print(f"  {len(active_jobs)} active, {completed}/{total} done, "
              f"{len(pending_windows)} queued", end="\r", flush=True)

    print(f"\n\nDone! {completed}/{total} windows processed.")
    _print_summary(output_path)


def _poll_jobs(endpoint: str, jobs: list[dict], output_path: Path):
    """Poll all jobs until completion."""
    pending = set(j["id"] for j in jobs if j["id"] != "?")
    completed = 0
    total = len(jobs)
    processed_job_ids: set[str] = set()

    while pending:
        tm.sleep(15)
        for job_id in list(pending):
            try:
                result = check_job(endpoint, job_id)
            except Exception as e:
                print(f"  (poll error: {e})", flush=True)
                continue

            status = result.get("status", "unknown")

            if status == "COMPLETED":
                pending.discard(job_id)
                if job_id not in processed_job_ids:
                    processed_job_ids.add(job_id)
                    completed += 1
                    job = next((j for j in jobs if j["id"] == job_id), {})
                    output = result.get("output", {})
                    _process_completed_window(job, output, output_path, completed, total)

            elif status in ("FAILED", "CANCELLED"):
                pending.discard(job_id)
                completed += 1
                job = next((j for j in jobs if j["id"] == job_id), {})
                error = result.get("output", {}).get("error", result.get("error", status))
                print(f"  {status} window {job.get('window_id', '?')}: {error}", flush=True)

        print(f"  {len(pending)} pending, {completed}/{total} done", end="\r", flush=True)

    print(f"\n\nDone! {completed}/{total} windows processed.")
    _print_summary(output_path)


def _process_completed_window(job: dict, output: dict, output_path: Path,
                               completed: int, total: int):
    """Process a completed window job — append regime records to JSONL."""
    wid = job.get("window_id", output.get("window_id", "?"))
    n_regimes = output.get("n_regimes", 0)
    n_profitable = output.get("n_profitable", 0)
    eval_sec = output.get("eval_sec", 0)

    regimes = output.get("regimes", [])
    if regimes:
        with open(output_path, "a") as f:
            for r in regimes:
                f.write(json.dumps(r) + "\n")

    print(f"\n  [{completed}/{total}] Window {wid} ({job.get('train_range', '?')}): "
          f"{n_regimes} regimes, {n_profitable} profitable ({eval_sec:.0f}s)", flush=True)


def _print_summary(output_path: Path):
    """Print summary from accumulated JSONL."""
    if not output_path.exists():
        print("No results file found.")
        return

    records = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No regime records generated.")
        return

    n = len(records)
    n_pos = sum(1 for r in records if r.get("forward_profitable", False))
    n_windows = len(set(r.get("window_id", 0) for r in records))

    # Forward PnL stats
    fwd_pnls = [r.get("fwd_cum_pnl", 0) for r in records]
    total_fwd_pnl = sum(fwd_pnls)

    # Exit reason breakdown
    exit_reasons: dict[str, int] = {}
    for r in records:
        reason = r.get("forward_exit_reason", "unknown")
        key = reason.split(" ")[0] if reason != "data_end" else "data_end"
        exit_reasons[key] = exit_reasons.get(key, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"WALK-FORWARD SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total regime records:  {n}")
    print(f"  Windows with data:     {n_windows}")
    print(f"  Forward profitable:    {n_pos} ({n_pos/n*100:.1f}%)")
    print(f"  Total fwd cum PnL:     ${total_fwd_pnl:+,.0f}")
    print(f"  Exit reasons:          {dict(sorted(exit_reasons.items(), key=lambda x: -x[1]))}")
    print(f"  Results:               {output_path}")


def _poll_single(endpoint: str, job_id: str) -> dict | None:
    """Poll a single job until completion, return output."""
    while True:
        tm.sleep(10)
        result = check_job(endpoint, job_id)
        status = result.get("status", "unknown")
        print(f"  Status: {status}", end="\r", flush=True)
        if status == "COMPLETED":
            output = result.get("output", {})
            print(f"\n{json.dumps(output, indent=2)}")
            return output
        elif status == "FAILED":
            error = result.get("output", {}).get("error", result.get("error", "unknown"))
            print(f"\n  FAILED: {error}")
            return None


if __name__ == "__main__":
    main()
