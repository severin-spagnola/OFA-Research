"""
Options Gene Search Orchestrator
=================================
Submits batch jobs to RunPod serverless endpoint for options gene search.

Usage:
    python options_orchestrator.py                    # 4 workers, 5K candidates each
    python options_orchestrator.py --workers 8        # 8 workers
    python options_orchestrator.py --candidates 10000 # 10K per worker
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as tm
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load env from Strat repo
_STRAT_ENV = Path.home() / "Desktop" / "Strat" / ".env"
if _STRAT_ENV.exists():
    load_dotenv(_STRAT_ENV)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
# Options endpoint — update this after creating the RunPod endpoint
OPTIONS_ENDPOINT_ID = os.environ.get("OPTIONS_ENDPOINT_ID", "5vpdn34inq2tty")


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


def main():
    parser = argparse.ArgumentParser(description="Submit options gene search batch to RunPod")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--candidates", type=int, default=5000, help="Candidates per worker")
    parser.add_argument("--val-split", type=float, default=0.30)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--endpoint", default=OPTIONS_ENDPOINT_ID)
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic job only")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward simulation")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--max-dd", type=float, default=None, help="DD kill threshold (uses calibrated default if omitted)")
    parser.add_argument("--start", type=str, default=None, help="Start date for walkforward")
    parser.add_argument("--wf-windows", type=int, default=None, help="Total regime windows (auto-computed if omitted)")
    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    endpoint = args.endpoint
    print(f"Options Gene Search Orchestrator")
    print(f"  Endpoint: {endpoint}")
    print(f"  Workers: {args.workers}")
    print(f"  Candidates/worker: {args.candidates}")
    print(f"  Total candidates: {args.workers * args.candidates:,}")
    print()

    if args.diagnostic:
        print("Submitting diagnostic job...")
        result = submit_job(endpoint, {"job": "diagnostic"})
        print(json.dumps(result, indent=2))
        if "id" in result:
            _poll_job(endpoint, result["id"])
        return

    if args.walkforward:
        n_workers = args.workers

        # Estimate total windows if not provided
        # ~471 data days, subtract train+val overhead, divide by step
        if args.wf_windows:
            total_windows = args.wf_windows
        else:
            # Conservative estimate — will be exact on server
            approx_data_days = 471
            total_windows = max(1, (approx_data_days - args.train_days - args.val_days) // args.step_days)

        # Split windows across workers
        chunk_size = (total_windows + n_workers - 1) // n_workers
        print(f"Walk-forward: ~{total_windows} windows across {n_workers} workers ({chunk_size}/worker)")
        print()

        jobs = []
        for i in range(n_workers):
            c_start = i * chunk_size
            c_end = min((i + 1) * chunk_size, total_windows)
            if c_start >= total_windows:
                break

            payload = {
                "job": "walkforward",
                "train_days": args.train_days,
                "val_days": args.val_days,
                "step_days": args.step_days,
                "candidates": args.candidates,
                "chunk_start": c_start,
                "chunk_end": c_end,
                "regime_id_offset": c_start,
            }
            if args.max_dd is not None:
                payload["max_dd"] = args.max_dd
            if args.start:
                payload["start"] = args.start

            result = submit_job(endpoint, payload)
            job_id = result.get("id", "?")
            jobs.append({"id": job_id, "chunk": f"{c_start}-{c_end}", "status": "submitted"})
            print(f"  Worker {i+1}/{n_workers}: job={job_id} windows=[{c_start}:{c_end}]")

        print(f"\n  {len(jobs)} jobs submitted. Polling...")
        _poll_walkforward_chunks(endpoint, jobs)
        return

    # Submit workers with different seeds
    jobs = []
    for i in range(args.workers):
        payload = {
            "candidates": args.candidates,
            "seed": 1000 + i * 1000,
            "val_split": args.val_split,
            "top_n": args.top_n,
            "min_trades": args.min_trades,
        }
        result = submit_job(endpoint, payload)
        job_id = result.get("id", "?")
        jobs.append({"id": job_id, "seed": payload["seed"], "status": "submitted"})
        print(f"  Worker {i+1}/{args.workers}: job={job_id} seed={payload['seed']}")

    print(f"\n  {len(jobs)} jobs submitted. Polling...")
    _poll_jobs(endpoint, jobs)


def _poll_job(endpoint: str, job_id: str):
    """Poll a single job until completion."""
    while True:
        tm.sleep(10)
        result = check_job(endpoint, job_id)
        status = result.get("status", "unknown")
        print(f"  Status: {status}")
        if status in ("COMPLETED", "FAILED"):
            output = result.get("output", {})
            print(json.dumps(output, indent=2))
            return


def _poll_walkforward(endpoint: str, job_id: str):
    """Poll a walk-forward job until completion, save results."""
    output_dir = Path("results/options")
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        tm.sleep(30)
        result = check_job(endpoint, job_id)
        status = result.get("status", "unknown")
        print(f"  Status: {status}", end="\r")

        if status == "COMPLETED":
            output = result.get("output", {})
            n_regimes = output.get("n_regimes", 0)
            n_positive = output.get("n_positive", 0)
            total_pnl = output.get("total_forward_pnl", 0)
            print(f"\n  Walk-forward complete: {n_regimes} regimes, "
                  f"{n_positive} positive, ${total_pnl:+,.0f} total forward PnL")

            # Save full results
            out_path = output_dir / "regime_db.json"
            with open(out_path, "w") as f:
                json.dump(output.get("regimes", []), f, indent=2)
            print(f"  Saved to {out_path}")

            # Print regime summary
            for r in output.get("regimes", []):
                status_icon = "+" if r.get("forward_pnl", 0) > 0 else "-"
                print(f"    [{status_icon}] {r.get('train_start','?')}→{r.get('forward_end','?')} "
                      f"pnl=${r.get('forward_pnl',0):+.0f} "
                      f"WR={r.get('forward_wr',0)*100:.0f}% "
                      f"{r.get('forward_trades',0)}t/{r.get('forward_days',0)}d "
                      f"death={r.get('death_reason','?')}")
            return

        elif status == "FAILED":
            error = result.get("output", {}).get("error", result.get("error", "unknown"))
            print(f"\n  FAILED: {error}")
            return


def _poll_jobs(endpoint: str, jobs: list[dict]):
    """Poll all jobs until completion."""
    output_dir = Path("results/options")
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = set(j["id"] for j in jobs if j["id"] != "?")
    completed = 0
    total_survivors = 0

    while pending:
        tm.sleep(15)
        for job_id in list(pending):
            result = check_job(endpoint, job_id)
            status = result.get("status", "unknown")

            if status == "COMPLETED":
                pending.discard(job_id)
                completed += 1
                output = result.get("output", {})

                # Find matching job info
                job_info = next((j for j in jobs if j["id"] == job_id), {})
                seed = job_info.get("seed", 0)

                n_viable = output.get("n_viable", 0)
                n_pass = output.get("n_val_pass", 0)
                total_survivors += n_pass

                print(f"\n  [{completed}/{len(jobs)}] seed={seed}: "
                      f"viable={n_viable} val_pass={n_pass}")

                # Print top results
                for i, r in enumerate(output.get("top_results", [])[:3]):
                    vf = r.get("val_fitness", {})
                    print(f"    #{i+1}: fitness={vf.get('fitness', 0):.4f} "
                          f"pnl=${vf.get('cum_pnl', 0):+.0f} "
                          f"WR={vf.get('win_rate', 0)*100:.0f}% "
                          f"{r.get('genes_desc', '')}")

                # Save result
                out_path = output_dir / f"options_seed{seed}.json"
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2)

            elif status == "FAILED":
                pending.discard(job_id)
                completed += 1
                error = result.get("output", {}).get("error", result.get("error", "unknown"))
                print(f"\n  FAILED (seed={next((j.get('seed') for j in jobs if j['id']==job_id), '?')}): {error}")

        print(f"  ... {len(pending)} pending, {completed}/{len(jobs)} done, "
              f"{total_survivors} total survivors", end="\r")

    print(f"\n\nDone! {total_survivors} total survivors across {len(jobs)} workers.")
    print(f"Results saved in {output_dir}/")


def _poll_walkforward_chunks(endpoint: str, jobs: list[dict]):
    """Poll parallel walk-forward chunk jobs, merge results."""
    output_dir = Path("results/options")
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = set(j["id"] for j in jobs if j["id"] != "?")
    completed = 0
    all_regimes = []

    while pending:
        tm.sleep(30)
        for job_id in list(pending):
            result = check_job(endpoint, job_id)
            status = result.get("status", "unknown")

            if status == "COMPLETED":
                pending.discard(job_id)
                completed += 1
                output = result.get("output", {})

                job_info = next((j for j in jobs if j["id"] == job_id), {})
                chunk = job_info.get("chunk", "?")
                n_regimes = output.get("n_regimes", 0)
                n_positive = output.get("n_positive", 0)
                chunk_pnl = output.get("total_forward_pnl", 0)

                regimes = output.get("regimes", [])
                all_regimes.extend(regimes)

                print(f"\n  [{completed}/{len(jobs)}] chunk={chunk}: "
                      f"{n_regimes} regimes, {n_positive} positive, ${chunk_pnl:+,.0f}")

                # Save individual chunk
                out_path = output_dir / f"wf_chunk_{chunk.replace('-', '_')}.json"
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2)

            elif status == "FAILED":
                pending.discard(job_id)
                completed += 1
                error = result.get("output", {}).get("error", result.get("error", "unknown"))
                job_info = next((j for j in jobs if j["id"] == job_id), {})
                print(f"\n  FAILED (chunk={job_info.get('chunk', '?')}): {error}")

        print(f"  ... {len(pending)} pending, {completed}/{len(jobs)} done", end="\r")

    # Merge and save
    all_regimes.sort(key=lambda r: r.get("regime_id", 0))

    out_path = output_dir / "regime_db.json"
    with open(out_path, "w") as f:
        json.dump(all_regimes, f, indent=2)

    n_total = len(all_regimes)
    n_positive = sum(1 for r in all_regimes if r.get("forward_pnl", 0) > 0)
    total_pnl = sum(r.get("forward_pnl", 0) for r in all_regimes)

    print(f"\n\nWalk-forward complete: {n_total} regimes, "
          f"{n_positive} positive ({n_positive/n_total*100:.0f}%), "
          f"${total_pnl:+,.0f} total forward PnL")
    print(f"Saved merged results to {out_path}")

    # Print regime summary
    for r in all_regimes:
        icon = "+" if r.get("forward_pnl", 0) > 0 else "-"
        print(f"  [{icon}] R{r.get('regime_id','?')} "
              f"{r.get('train_start','?')}→{r.get('forward_end','?')} "
              f"pnl=${r.get('forward_pnl',0):+.0f} "
              f"WR={r.get('forward_wr',0)*100:.0f}% "
              f"{r.get('forward_trades',0)}t/{r.get('forward_days',0)}d "
              f"death={r.get('death_reason','?')}")


if __name__ == "__main__":
    main()
