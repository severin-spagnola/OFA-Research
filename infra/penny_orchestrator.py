#!/usr/bin/env python3
"""
Penny Gapper Orchestrator — Parallel Gene Search on RunPod
============================================================
Submits multiple penny search jobs (different seeds) to RunPod workers,
polls for completion, merges top candidates across all workers.

Usage:
    python infra/penny_orchestrator.py                             # defaults
    python infra/penny_orchestrator.py --workers 5                 # 5 parallel workers
    python infra/penny_orchestrator.py --profile nasdaq_highvol    # specific profile
    python infra/penny_orchestrator.py --candidates 10000          # per worker
    python infra/penny_orchestrator.py --profile all               # all profiles
    python infra/penny_orchestrator.py --dry-run                   # preview

Each worker gets a different RNG seed and evaluates --candidates genes.
Results are merged and ranked by val fitness to find the overall best.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
RESULTS_DIR = _REPO_ROOT / "results" / "penny"

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("PENNY_RUNPOD_ENDPOINT_ID",
                                     os.environ.get("RUNPOD_ENDPOINT_ID", ""))

PROFILES = ["listed_highvol", "nasdaq_highvol", "otc_highvol", "smallcap_volatile", "allcap_volatile"]

POLL_INTERVAL_SEC = 10
MAX_POLL_DURATION_SEC = 3600  # 1 hour max per job (penny is fast)


# ─── RunPod API ─────────────────────────────────────────────────────────────

def submit_job(profile: str, candidates: int, seed: int,
               val_split: float = 0.30, top_n: int = 50) -> str:
    """Submit a penny search job. Returns job ID."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"

    payload = {
        "input": {
            "job": "penny_search",
            "profile": profile,
            "candidates": candidates,
            "seed": seed,
            "val_split": val_split,
            "top_n": top_n,
        }
    }

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
    print(f"  Submitted: profile={profile} candidates={candidates} seed={seed} → {job_id}")
    return job_id


def poll_job(job_id: str) -> tuple[str, dict | None]:
    """Check job status. Returns (status, full_response_or_none)."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    status = result.get("status", "UNKNOWN")
    if status in ("COMPLETED", "FAILED"):
        return status, result
    return status, None


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
            json.loads(resp.read().decode())
        return True
    except Exception as e:
        print(f"  [cancel] Failed: {e}")
        return False


# ─── Log saving ───────────────────────────────────────────────────────────

LOGS_DIR = _REPO_ROOT / "results" / "penny" / "logs"


def _save_job_log(job_id: str, seed: int, full_response: dict):
    """Save full RunPod job response to a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = full_response.get("status", "UNKNOWN")
    log_path = LOGS_DIR / f"job_{ts_str}_seed{seed}_{status}.json"
    with open(log_path, "w") as f:
        json.dump(full_response, f, indent=2)
    print(f"  [log] Saved → {log_path.name}")


# ─── Orchestration ──────────────────────────────────────────────────────────

def run_search(
    profile: str,
    n_workers: int = 3,
    candidates_per_worker: int = 5000,
    val_split: float = 0.30,
    base_seed: int = 1000,
    dry_run: bool = False,
):
    """Run parallel penny search for a single profile."""
    if not RUNPOD_API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        return
    if not RUNPOD_ENDPOINT_ID:
        print("ERROR: Set RUNPOD_ENDPOINT_ID or PENNY_RUNPOD_ENDPOINT_ID")
        return

    total_candidates = n_workers * candidates_per_worker
    print(f"\n{'='*70}")
    print(f"PENNY GENE SEARCH — {profile}")
    print(f"{'='*70}")
    print(f"  Workers: {n_workers}")
    print(f"  Candidates/worker: {candidates_per_worker}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Val split: {val_split}")

    if dry_run:
        for i in range(n_workers):
            print(f"  [DRY RUN] Would submit: profile={profile} "
                  f"candidates={candidates_per_worker} seed={base_seed + i}")
        return

    # Submit all workers
    in_flight = {}
    for i in range(n_workers):
        seed = base_seed + i
        try:
            job_id = submit_job(profile, candidates_per_worker, seed, val_split)
            in_flight[job_id] = {
                "seed": seed,
                "submitted_at": time.time(),
            }
        except Exception as e:
            print(f"  FAILED to submit seed={seed}: {e}")

    if not in_flight:
        print("  No jobs submitted!")
        return

    # Poll loop
    completed_results = []
    overall_start = time.time()

    while in_flight:
        time.sleep(POLL_INTERVAL_SEC)
        ts = datetime.now().strftime("%H:%M:%S")

        for job_id in list(in_flight.keys()):
            info = in_flight[job_id]
            elapsed = time.time() - info["submitted_at"]

            if elapsed > MAX_POLL_DURATION_SEC:
                print(f"  {ts} TIMEOUT seed={info['seed']} after {elapsed/60:.0f}min")
                cancel_job(job_id)
                del in_flight[job_id]
                continue

            try:
                status, output = poll_job(job_id)
            except Exception as e:
                print(f"  {ts} Poll error for {job_id[:8]}: {e}")
                continue

            if status == "COMPLETED" and output:
                resp_output = output.get("output", {})
                if resp_output.get("error"):
                    print(f"  {ts} ERROR seed={info['seed']}: {resp_output['error']}")
                    if resp_output.get("traceback"):
                        for line in resp_output["traceback"].strip().split("\n")[-3:]:
                            print(f"    {line}")
                    _save_job_log(job_id, info["seed"], output)
                    del in_flight[job_id]
                    continue

                n_viable = resp_output.get("n_viable", 0)
                n_val_pass = resp_output.get("n_val_pass", 0)
                train_sec = resp_output.get("train_eval_sec", 0)
                val_sec = resp_output.get("val_eval_sec", 0)
                mins = elapsed / 60

                print(f"  {ts} DONE seed={info['seed']} ({mins:.1f}min): "
                      f"viable={n_viable} val_pass={n_val_pass} "
                      f"(train={train_sec:.0f}s val={val_sec:.0f}s)")

                completed_results.append(resp_output)
                _save_job_log(job_id, info["seed"], output)
                del in_flight[job_id]

            elif status == "FAILED":
                resp_output = output.get("output", {}) if output else {}
                error = output.get("error", "") or resp_output.get("error", "unknown") if output else "unknown"
                print(f"  {ts} FAILED seed={info['seed']}: {error}")
                if resp_output.get("traceback"):
                    for line in resp_output["traceback"].strip().split("\n")[-3:]:
                        print(f"    {line}")
                _save_job_log(job_id, info["seed"], output)
                del in_flight[job_id]

            else:
                mins = elapsed / 60
                print(f"  {ts} seed={info['seed']}: {status} ({mins:.0f}min)")

    total_min = (time.time() - overall_start) / 60

    # Merge results
    if not completed_results:
        print(f"\n  No completed results!")
        return

    all_top = []
    for result in completed_results:
        for r in result.get("top_results", []):
            all_top.append(r)

    # Rank by val fitness
    all_top.sort(key=lambda x: -x["val_fitness"]["fitness"])

    # Deduplicate by genes description (same strategy from different seeds)
    seen = set()
    unique_top = []
    for r in all_top:
        desc = r["genes_desc"]
        if desc not in seen:
            seen.add(desc)
            unique_top.append(r)

    print(f"\n{'='*70}")
    print(f"MERGED RESULTS — {profile}")
    print(f"{'='*70}")
    print(f"  Workers completed: {len(completed_results)}/{n_workers}")
    print(f"  Total candidates: {sum(r.get('n_candidates', 0) for r in completed_results)}")
    print(f"  Total viable: {sum(r.get('n_viable', 0) for r in completed_results)}")
    print(f"  Val-pass: {sum(r.get('n_val_pass', 0) for r in completed_results)}")
    print(f"  Unique top strategies: {len(unique_top)}")
    print(f"  Total time: {total_min:.1f}min")

    # Print top 10
    print(f"\n  Top 10 (by val fitness):")
    print(f"  {'#':>3} {'ValFit':>7} {'TrFit':>7} {'TrN':>5} {'VlN':>5} "
          f"{'TrWR':>5} {'VlWR':>5} {'TrCum':>7} {'VlCum':>7}  Genes")
    for i, r in enumerate(unique_top[:10]):
        tf = r["train_fitness"]
        vf = r["val_fitness"]
        print(f"  {i+1:>3} {vf['fitness']:>7.4f} {tf['fitness']:>7.4f} "
              f"{tf['n_trades']:>5} {vf['n_trades']:>5} "
              f"{tf['win_rate']*100:>4.0f}% {vf['win_rate']*100:>4.0f}% "
              f"{tf['cum_r']:>+6.1f} {vf['cum_r']:>+6.1f}  "
              f"{r['genes_desc']}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"penny_search_{profile}_{ts_str}.json"

    save_data = {
        "profile": profile,
        "n_workers": n_workers,
        "candidates_per_worker": candidates_per_worker,
        "total_candidates": sum(r.get("n_candidates", 0) for r in completed_results),
        "total_viable": sum(r.get("n_viable", 0) for r in completed_results),
        "total_val_pass": sum(r.get("n_val_pass", 0) for r in completed_results),
        "total_time_min": round(total_min, 1),
        "top_20": unique_top[:20],
        "worker_results": [
            {
                "seed": r.get("seed"),
                "n_candidates": r.get("n_candidates"),
                "n_viable": r.get("n_viable"),
                "n_val_pass": r.get("n_val_pass"),
                "train_eval_sec": r.get("train_eval_sec"),
                "val_eval_sec": r.get("val_eval_sec"),
            }
            for r in completed_results
        ],
    }

    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved → {result_path}")

    # Print best candidate details
    if unique_top:
        best = unique_top[0]
        print(f"\n  {'='*60}")
        print(f"  BEST CANDIDATE")
        print(f"  {'='*60}")
        print(f"  {best['genes_desc']}")
        print(f"  Val fitness:  {best['val_fitness']['fitness']:.4f}")
        print(f"  Train: {best['train_fitness']['n_trades']} trades, "
              f"WR={best['train_fitness']['win_rate']*100:.1f}%, "
              f"avg={best['train_fitness']['avg_r']:+.3f}R, "
              f"cum={best['train_fitness']['cum_r']:+.1f}R, "
              f"PF={best['train_fitness']['profit_factor']:.2f}")
        print(f"  Val:   {best['val_fitness']['n_trades']} trades, "
              f"WR={best['val_fitness']['win_rate']*100:.1f}%, "
              f"avg={best['val_fitness']['avg_r']:+.3f}R, "
              f"cum={best['val_fitness']['cum_r']:+.1f}R, "
              f"PF={best['val_fitness']['profit_factor']:.2f}")
        print(f"\n  Full genes:")
        print(f"  {json.dumps(best['genes'], indent=4)}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Penny Gapper Gene Search Orchestrator")
    parser.add_argument("--profile", type=str, default="listed_highvol",
                        help="Profile name or 'all' for all profiles")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel RunPod workers")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Gene candidates per worker")
    parser.add_argument("--val-split", type=float, default=0.30)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles = PROFILES if args.profile == "all" else [args.profile]

    for profile in profiles:
        run_search(
            profile=profile,
            n_workers=args.workers,
            candidates_per_worker=args.candidates,
            val_split=args.val_split,
            base_seed=args.base_seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
