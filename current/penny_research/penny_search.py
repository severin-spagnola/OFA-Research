"""
Penny Gapper Gene Search Engine
=================================
Runs gene combinator search over pooled penny gapper data.
Evaluates candidates with temporal train/validation split.

Profiles define which gappers to pool:
  - listed_highvol: NASDAQ/NYSE/AMEX, vol_ratio >= 5x
  - nasdaq_highvol: NASDAQ only, vol_ratio >= 10x
  - otc_highvol: OTC, vol_ratio >= 20x

Usage:
    python penny_search.py                           # full search, default profile
    python penny_search.py --profile nasdaq_highvol  # specific profile
    python penny_search.py --candidates 10000        # more candidates
    python penny_search.py --profile all             # run all profiles
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time as tm
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from penny_genes import PennyGenes, sample_candidates, describe_genes
from penny_backtest import evaluate_pooled, evaluate_ticker_day, preload_bar_data, BacktestResult, TradeResult, CostModel
from penny_fitness import compute_fitness, FitnessResult

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
BAR_CACHE_DIR = DATA_DIR / "bar_cache"
RESULTS_DIR = SCRIPT_DIR / "results"


# ═════════════════════════════════════════════════════════════════════════════
# PROFILES
# ═════════════════════════════════════════════════════════════════════════════

PROFILES = {
    "listed_highvol": {
        "description": "Listed exchanges (non-OTC), gap >= 30%, prev_volume >= 100k",
        "filters": {
            "exchange_exclude": ["OTC Link"],
            "min_prev_volume": 100_000,
            "min_gap_pct": 30.0,
            "gap_direction": "up",
        },
    },
    "nasdaq_highvol": {
        "description": "NASDAQ only, gap >= 30%, prev_volume >= 200k",
        "filters": {
            "exchange_include": ["XNAS"],
            "min_prev_volume": 200_000,
            "min_gap_pct": 30.0,
            "gap_direction": "up",
        },
    },
    "otc_highvol": {
        "description": "OTC Link, gap >= 30%, prev_volume >= 50k",
        "filters": {
            "exchange_include": ["OTC Link"],
            "min_prev_volume": 50_000,
            "min_gap_pct": 30.0,
            "gap_direction": "up",
        },
    },
    "smallcap_volatile": {
        "description": "Listed $3-10, gap >= 15%, prev_volume >= 50k — higher-priced volatile small-caps",
        "filters": {
            "exchange_exclude": ["OTC Link"],
            "min_prev_close": 3.0,
            "max_prev_close": 10.0,
            "min_prev_volume": 50_000,
            "min_gap_pct": 15.0,
            "gap_direction": "up",
        },
    },
    "allcap_volatile": {
        "description": "Listed under $10, gap >= 15%, prev_volume >= 50k — full range volatile",
        "filters": {
            "exchange_exclude": ["OTC Link"],
            "max_prev_close": 10.0,
            "min_prev_volume": 50_000,
            "min_gap_pct": 15.0,
            "gap_direction": "up",
        },
    },
}


def load_ticker_days(profile_name: str) -> list[dict]:
    """Load and filter ticker-days for a profile."""
    data_path = DATA_DIR / "phase3_full.parquet"
    if not data_path.exists():
        data_path = DATA_DIR / "phase2_enriched.parquet"
    if not data_path.exists():
        data_path = DATA_DIR / "phase1_gappers.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"No survey data found. Run penny_gapper_survey.py first.")

    df = pd.read_parquet(data_path)
    profile = PROFILES[profile_name]
    filters = profile["filters"]

    # Apply filters
    if "exchange_exclude" in filters:
        df = df[~df["primary_exchange"].isin(filters["exchange_exclude"])]
    if "exchange_include" in filters:
        df = df[df["primary_exchange"].isin(filters["exchange_include"])]
    if "min_prev_volume" in filters:
        df = df[df["prev_volume"] >= filters["min_prev_volume"]]
    if "min_prev_close" in filters:
        df = df[df["prev_close"] >= filters["min_prev_close"]]
    if "max_prev_close" in filters:
        df = df[df["prev_close"] <= filters["max_prev_close"]]
    if "min_gap_pct" in filters:
        if filters.get("gap_direction") == "up":
            df = df[df["gap_pct"] >= filters["min_gap_pct"]]
        else:
            df = df[df["gap_pct"].abs() >= filters["min_gap_pct"]]

    # Convert to list of dicts
    ticker_days = []
    for _, row in df.iterrows():
        cache_file = BAR_CACHE_DIR / f"{row['ticker']}_{row['date']}.parquet"
        if not cache_file.exists():
            continue
        ticker_days.append({
            "ticker": row["ticker"],
            "date": row["date"],
            "gap_pct": row.get("gap_pct", 0),
            "prev_close": row.get("prev_close", 0),
            "vol_ratio": row.get("vol_ratio", 0),
            "primary_exchange": row.get("primary_exchange", ""),
        })

    # Sort by date for temporal split
    ticker_days.sort(key=lambda x: x["date"])
    return ticker_days


# ═════════════════════════════════════════════════════════════════════════════
# PARALLEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

# Global for fork-based worker pool
_WORKER_TICKER_DAYS: list[dict] = []
_WORKER_BAR_CACHE_DIR: str = ""
_WORKER_BAR_DATA: dict = {}
_WORKER_CANDIDATES: list[PennyGenes] = []
_WORKER_COST: CostModel | None = None


def _init_worker(ticker_days, bar_cache_dir, bar_data, candidates, cost=None):
    global _WORKER_TICKER_DAYS, _WORKER_BAR_CACHE_DIR, _WORKER_BAR_DATA, _WORKER_CANDIDATES, _WORKER_COST
    _WORKER_TICKER_DAYS = ticker_days
    _WORKER_BAR_CACHE_DIR = bar_cache_dir
    _WORKER_BAR_DATA = bar_data
    _WORKER_CANDIDATES = candidates
    _WORKER_COST = cost


def _eval_candidate(idx: int) -> tuple[int, dict, dict]:
    """Evaluate candidate at index. Returns (idx, fitness_dict, summary_dict)."""
    genes = _WORKER_CANDIDATES[idx]
    result = evaluate_pooled(
        genes, _WORKER_TICKER_DAYS, _WORKER_BAR_CACHE_DIR,
        bar_data=_WORKER_BAR_DATA if _WORKER_BAR_DATA else None,
        cost=_WORKER_COST,
    )
    fitness = compute_fitness(result)
    return idx, asdict(fitness), result.summary_dict()


def parallel_eval(
    candidates: list[PennyGenes],
    ticker_days: list[dict],
    bar_cache_dir: str | Path,
    n_workers: int = 0,
    bar_data: dict | None = None,
    cost: CostModel | None = None,
) -> list[tuple[int, FitnessResult, dict]]:
    """Evaluate all candidates in parallel.

    If bar_data is provided (from preload_bar_data), uses in-memory data
    instead of reading parquet files from disk each time.
    If cost is provided, uses that cost model instead of the default.
    """

    if n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)

    bar_cache_str = str(bar_cache_dir)

    if n_workers <= 1 or len(candidates) <= 10:
        # Sequential
        results = []
        for i in range(len(candidates)):
            result = evaluate_pooled(candidates[i], ticker_days, bar_cache_dir, bar_data=bar_data, cost=cost)
            fitness = compute_fitness(result)
            results.append((i, fitness, result.summary_dict()))
        return results

    # Parallel with fork
    global _WORKER_TICKER_DAYS, _WORKER_BAR_CACHE_DIR, _WORKER_BAR_DATA, _WORKER_CANDIDATES, _WORKER_COST
    _WORKER_TICKER_DAYS = ticker_days
    _WORKER_BAR_CACHE_DIR = bar_cache_str
    _WORKER_BAR_DATA = bar_data or {}
    _WORKER_CANDIDATES = candidates
    _WORKER_COST = cost

    ctx = mp.get_context("fork")
    with ctx.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(ticker_days, bar_cache_str, bar_data or {}, candidates, cost),
    ) as pool:
        raw_results = pool.map(_eval_candidate, range(len(candidates)))

    # Convert back to FitnessResult objects
    results = []
    for idx, fit_dict, summary in raw_results:
        fr = FitnessResult(**fit_dict)
        results.append((idx, fr, summary))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# GENE SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def run_gene_search(
    profile_name: str = "listed_highvol",
    max_candidates: int = 5000,
    seed: int = 42,
    n_workers: int = 0,
    val_split: float = 0.30,
) -> dict:
    """Run full gene search with temporal train/val split.

    Returns dict with best genes, fitness, and diagnostics.
    """
    print(f"\n{'='*70}")
    print(f"PENNY GENE SEARCH — Profile: {profile_name}")
    print(f"{'='*70}")

    # ── Load data ──
    ticker_days = load_ticker_days(profile_name)
    print(f"  Loaded {len(ticker_days)} ticker-days with cached 1m bars")
    if len(ticker_days) < 30:
        print(f"  ERROR: Need at least 30 ticker-days for meaningful search")
        return {}

    # ── Temporal split (by unique date boundary, not row index) ──
    unique_dates = sorted(set(td["date"] for td in ticker_days))
    split_date_idx = int(len(unique_dates) * (1 - val_split))
    train_cutoff_date = unique_dates[split_date_idx]  # first val date

    train_days = [td for td in ticker_days if td["date"] < train_cutoff_date]
    val_days = [td for td in ticker_days if td["date"] >= train_cutoff_date]

    if not train_days or not val_days:
        print(f"  ERROR: Split produced empty train or val set")
        return {}

    print(f"  Train: {len(train_days)} ticker-days "
          f"({train_days[0]['date']} to {train_days[-1]['date']})")
    print(f"  Val:   {len(val_days)} ticker-days "
          f"({val_days[0]['date']} to {val_days[-1]['date']})")

    # ── Sample candidates ──
    print(f"\n  Sampling {max_candidates} gene candidates...")
    candidates = sample_candidates(n=max_candidates, seed=seed)

    # Count distributions
    n_long = sum(1 for c in candidates if c.direction == "long")
    n_short = sum(1 for c in candidates if c.direction == "short")
    n_trail = sum(1 for c in candidates if c.trail_type != "none")
    n_partial = sum(1 for c in candidates if c.partial_exit)
    print(f"  Long: {n_long}, Short: {n_short}")
    print(f"  With trail: {n_trail}, With partial: {n_partial}")

    # ── Pre-load bar data into memory ──
    t0 = tm.time()
    bar_data = preload_bar_data(ticker_days, BAR_CACHE_DIR)
    mem_mb = sum(df.memory_usage(deep=True).sum() for df in bar_data.values()) / 1024 / 1024
    print(f"  Pre-loaded {len(bar_data)} bar files ({mem_mb:.0f}MB, {tm.time()-t0:.1f}s)")

    # ── Phase 1: Train evaluation ──
    print(f"\n  Phase 1: Evaluating {max_candidates} candidates on train set...")
    t0 = tm.time()
    train_results = parallel_eval(candidates, train_days, BAR_CACHE_DIR, n_workers, bar_data=bar_data)
    t1 = tm.time()
    print(f"  Train eval: {t1-t0:.1f}s ({(t1-t0)/max_candidates*1000:.1f}ms/candidate)")

    # Filter viable
    viable = [(idx, fr, sm) for idx, fr, sm in train_results if fr.fitness > 0]
    print(f"  Viable candidates: {len(viable)}/{max_candidates} "
          f"({len(viable)/max_candidates*100:.1f}%)")

    if not viable:
        print("  No viable candidates found. Try different profile or more candidates.")
        return {}

    # Sort by fitness
    viable.sort(key=lambda x: -x[1].fitness)

    # Print top 10 train
    print(f"\n  Top 10 (train):")
    print(f"  {'Rank':>4} {'Fitness':>8} {'Trades':>7} {'WR%':>6} {'AvgR':>7} "
          f"{'CumR':>7} {'PF':>6} {'DD':>6} {'Genes'}")
    for rank, (idx, fr, sm) in enumerate(viable[:10]):
        g = candidates[idx]
        print(f"  {rank+1:>4} {fr.fitness:>8.4f} {fr.n_trades:>7} {fr.win_rate*100:>5.1f}% "
              f"{fr.avg_r:>+6.3f} {fr.cum_r:>+6.1f} {fr.profit_factor:>5.2f} "
              f"{fr.max_dd_r:>5.1f} {describe_genes(g)}")

    # ── Phase 2: Validation on top 50 ──
    top_n = min(50, len(viable))
    top_indices = [idx for idx, _, _ in viable[:top_n]]
    top_candidates = [candidates[i] for i in top_indices]

    print(f"\n  Phase 2: Validating top {top_n} on val set...")
    t0 = tm.time()
    val_results = parallel_eval(top_candidates, val_days, BAR_CACHE_DIR, n_workers, bar_data=bar_data)
    t1 = tm.time()
    print(f"  Val eval: {t1-t0:.1f}s")

    # Rank by val fitness only (train was just a prefilter)
    combined = []
    for j, (val_idx, val_fr, val_sm) in enumerate(val_results):
        train_idx, train_fr, train_sm = viable[j]
        combined.append((train_idx, train_fr, val_fr, val_fr.fitness, candidates[train_idx]))

    # Filter: must be viable on val too
    combined = [c for c in combined if c[3] > 0]
    combined.sort(key=lambda x: -x[3])

    # Print top 10 by val fitness
    print(f"\n  Top 10 (ranked by val fitness, train-prefiltered):")
    print(f"  {'Rank':>4} {'ValFit':>8} {'TrFit':>7} {'TrTrd':>6} "
          f"{'VlTrd':>6} {'TrWR':>5} {'VlWR':>5} {'TrCumR':>7} {'VlCumR':>7}")
    for rank, (idx, tfr, vfr, vf, g) in enumerate(combined[:10]):
        print(f"  {rank+1:>4} {vf:>8.4f} {tfr.fitness:>7.4f} "
              f"{tfr.n_trades:>6} {vfr.n_trades:>6} "
              f"{tfr.win_rate*100:>4.0f}% {vfr.win_rate*100:>4.0f}% "
              f"{tfr.cum_r:>+6.1f} {vfr.cum_r:>+6.1f}")

    # ── Best candidate ──
    if combined:
        best_idx, best_train, best_val, best_val_fit, best_genes = combined[0]

        print(f"\n  {'='*60}")
        print(f"  BEST CANDIDATE")
        print(f"  {'='*60}")
        print(f"  Genes: {describe_genes(best_genes)}")
        print(f"  Val fitness: {best_val_fit:.4f}")
        print(f"  Train: {best_train.n_trades} trades, "
              f"WR={best_train.win_rate*100:.1f}%, "
              f"avg={best_train.avg_r:+.3f}R, "
              f"cum={best_train.cum_r:+.1f}R, "
              f"PF={best_train.profit_factor:.2f}")
        print(f"  Val:   {best_val.n_trades} trades, "
              f"WR={best_val.win_rate*100:.1f}%, "
              f"avg={best_val.avg_r:+.3f}R, "
              f"cum={best_val.cum_r:+.1f}R, "
              f"PF={best_val.profit_factor:.2f}")

        # Full gene dump
        print(f"\n  Full gene config:")
        for k, v in best_genes.to_dict().items():
            print(f"    {k}: {v}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"search_{profile_name}.json"
        result_data = {
            "profile": profile_name,
            "n_candidates": max_candidates,
            "n_viable": len(viable),
            "train_ticker_days": len(train_days),
            "val_ticker_days": len(val_days),
            "train_date_range": [train_days[0]["date"], train_days[-1]["date"]],
            "val_date_range": [val_days[0]["date"], val_days[-1]["date"]],
            "best_genes": best_genes.to_dict(),
            "best_val_fitness_score": best_val_fit,
            "best_train_fitness": asdict(best_train),
            "best_val_fitness": asdict(best_val),
            "top_10": [
                {
                    "rank": rank + 1,
                    "genes": g.to_dict(),
                    "val_fitness_score": vf,
                    "train_fitness": asdict(tfr),
                    "val_fitness": asdict(vfr),
                }
                for rank, (_, tfr, vfr, vf, g) in enumerate(combined[:10])
            ],
        }
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n  Saved results -> {result_path}")

        return result_data

    return {}


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Penny Gapper Gene Search")
    parser.add_argument("--profile", default="listed_highvol",
                        choices=list(PROFILES.keys()) + ["all"],
                        help="Gapper profile to search")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Number of gene candidates (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers (0=auto)")
    parser.add_argument("--val-split", type=float, default=0.30,
                        help="Validation split fraction (default: 0.30)")
    args = parser.parse_args()

    if args.profile == "all":
        for profile in PROFILES:
            run_gene_search(
                profile_name=profile,
                max_candidates=args.candidates,
                seed=args.seed,
                n_workers=args.workers,
                val_split=args.val_split,
            )
    else:
        run_gene_search(
            profile_name=args.profile,
            max_candidates=args.candidates,
            seed=args.seed,
            n_workers=args.workers,
            val_split=args.val_split,
        )


if __name__ == "__main__":
    main()
