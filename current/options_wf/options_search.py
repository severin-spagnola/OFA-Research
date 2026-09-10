"""
Options Gene Search — Parallel Evaluation Engine
=================================================
Distributes gene candidate evaluation across multiple workers.
Pre-loads all data into memory once, then evaluates candidates in parallel.

Used by the RunPod handler to process batches of candidates.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time as tm
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

from options_backtest import (
    evaluate_day, evaluate_day_overnight,
    compute_fitness, compute_extended_analytics, FitnessResult, OptionsTrade,
    load_underlying, load_chain, load_chain_meta, get_available_days,
    CostModel, precompute_day_context, precompute_chain,
)
from options_genes import OptionsGenes

# Default data directory — set by RunPod handler
DATA_DIR = Path(os.environ.get("OPTIONS_DATA_DIR", "/data"))
CHAIN_SUBDIR = os.environ.get("OPTIONS_CHAIN_SUBDIR", "options_5dte")

# Max subsequent days to look ahead for multi-day holds
_MAX_HOLD_DAYS = 5


def parallel_eval(
    candidates: list[OptionsGenes],
    days: list[date],
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    n_workers: int | None = None,
    preloaded_und: dict | None = None,
    preloaded_chains: dict | None = None,
    preloaded_metas: dict | None = None,
    cost: CostModel | None = None,
    min_trades: int = 50,
    preloaded_ctxs: dict | None = None,
    allow_overnight: bool = False,
) -> list[tuple[int, FitnessResult, dict]]:
    """Evaluate candidates in parallel across workers.

    Args:
        allow_overnight: if True, use evaluate_day_overnight for candidates
            with eod_exit=False, passing subsequent days' chain data.

    Returns list of (candidate_index, FitnessResult, summary_dict).
    """
    if n_workers is None:
        n_workers = int(os.environ.get("OPTIONS_WORKERS", mp.cpu_count()))

    # If data not preloaded, load now (serial — only happens once)
    if preloaded_und is None:
        preloaded_und = {}
        for d in days:
            und = load_underlying(data_dir, d)
            if not und.empty:
                preloaded_und[d] = und

    if preloaded_chains is None:
        preloaded_chains = {}
        for d in days:
            chain = load_chain(data_dir, d, chain_subdir)
            if not chain.empty:
                preloaded_chains[d] = chain

    if preloaded_metas is None:
        preloaded_metas = {}
        for d in days:
            meta = load_chain_meta(data_dir, d, chain_subdir)
            if meta:
                preloaded_metas[d] = meta

    # Add prev day stats to metas (for gap and regime filters)
    sorted_days = sorted(preloaded_und.keys())
    for i, d in enumerate(sorted_days):
        if d in preloaded_metas:
            if i > 0:
                prev_d = sorted_days[i - 1]
                if prev_d in preloaded_und:
                    prev_und = preloaded_und[prev_d]
                    preloaded_metas[d]["prev_close"] = float(prev_und.iloc[-1]["c"])
                    preloaded_metas[d]["prev_high"] = float(prev_und["h"].max())
                    preloaded_metas[d]["prev_low"] = float(prev_und["l"].min())

    # Pre-compute chain _minute columns (must happen before ctx enrichment)
    for d in list(preloaded_chains.keys()):
        preloaded_chains[d] = precompute_chain(preloaded_chains[d])

    # Pre-compute day contexts with chain-derived signals
    if preloaded_ctxs is None:
        preloaded_ctxs = {}
        for d in days:
            if d in preloaded_und:
                chain_for_ctx = preloaded_chains.get(d)
                meta_for_ctx = preloaded_metas.get(d)
                ctx = precompute_day_context(preloaded_und[d], chain_for_ctx, meta_for_ctx)
                if ctx is not None:
                    preloaded_ctxs[d] = ctx

    # Filter to days with all data available
    valid_days = [d for d in days
                  if d in preloaded_und and d in preloaded_chains
                  and d in preloaded_metas and d in preloaded_ctxs]

    # For overnight holds: precompute subsequent chain lookups per day
    # Maps day -> list of subsequent chains (up to _MAX_HOLD_DAYS ahead)
    # IMPORTANT: Only use days within the current split (valid_days) to prevent
    # train/val data leakage — a train-day trade must NOT hold into val days.
    subsequent_chains_map: dict[date, list[pd.DataFrame]] = {}
    if allow_overnight:
        valid_days_set = set(valid_days)
        sorted_valid = sorted(valid_days)
        for i, d in enumerate(sorted_valid):
            subsequent = []
            for j in range(1, _MAX_HOLD_DAYS + 1):
                if i + j < len(sorted_valid):
                    next_d = sorted_valid[i + j]
                    if next_d in preloaded_chains:
                        subsequent.append(preloaded_chains[next_d])
            subsequent_chains_map[d] = subsequent

    n_total = len(candidates)
    log_interval = max(1, n_total // 4)  # 25% steps
    t0 = tm.time()

    results = []
    for idx, genes in enumerate(candidates):
        try:
            gene_dict = genes.to_dict()
            trades = []

            # Check if this candidate needs overnight eval
            needs_overnight = (allow_overnight and
                               not gene_dict.get("exit_rules", {}).get("eod_exit", True))

            for d in valid_days:
                if needs_overnight:
                    trade = evaluate_day_overnight(
                        preloaded_und[d],
                        preloaded_chains[d],
                        preloaded_metas[d],
                        gene_dict,
                        subsequent_chains=subsequent_chains_map.get(d, []),
                        cost=cost,
                        _ctx=preloaded_ctxs[d],
                    )
                else:
                    trade = evaluate_day(
                        preloaded_und[d],
                        preloaded_chains[d],
                        preloaded_metas[d],
                        gene_dict,
                        cost,
                        _ctx=preloaded_ctxs[d],
                    )
                if trade is not None:
                    trades.append(trade)

            fr = compute_fitness(trades, min_trades=min_trades)
            summary = {
                "n_trades": len(trades),
                "n_wins": sum(1 for t in trades if t.result == "win"),
                "cum_pnl": round(sum(t.pnl_net for t in trades), 2),
                "avg_pnl": round(sum(t.pnl_net for t in trades) / len(trades), 2) if trades else 0,
            }
            if needs_overnight:
                summary["overnight_hold"] = True

            # Extended analytics for viable candidates only (avoids 10K MC on garbage)
            if fr.fitness > 0 and trades:
                summary["extended"] = compute_extended_analytics(trades)

                # Cost sensitivity: re-evaluate at 2x and 3x friction
                for mult, label in [(2, "cost_2x"), (3, "cost_3x")]:
                    stressed_cost = CostModel(
                        slippage_per_leg=(cost or CostModel()).slippage_per_leg * mult,
                        commission_per_leg=(cost or CostModel()).commission_per_leg * mult,
                    )
                    stressed_pnls = []
                    for t in trades:
                        entry_cost = stressed_cost.entry_friction
                        exit_cost = stressed_cost.exit_friction
                        if t.spread_type and "naked" in t.spread_type:
                            entry_cost = stressed_cost.slippage_per_leg * 100 + stressed_cost.commission_per_leg
                            exit_cost = stressed_cost.commission_per_leg
                        stressed_pnl = t.pnl_per_contract - entry_cost - exit_cost
                        stressed_pnls.append(stressed_pnl)
                    s_cum = round(sum(stressed_pnls), 2)
                    s_wins = sum(1 for p in stressed_pnls if p > 0)
                    summary["extended"][label] = {
                        "cum_pnl": s_cum,
                        "win_rate": round(s_wins / len(stressed_pnls), 4) if stressed_pnls else 0,
                        "avg_pnl": round(s_cum / len(stressed_pnls), 2) if stressed_pnls else 0,
                        "profitable": s_cum > 0,
                    }

            results.append((idx, fr, summary))
        except Exception as e:
            results.append((idx, FitnessResult(fitness=-99.0), {"error": str(e)}))

        # Progress logging at 25% intervals
        if (idx + 1) % log_interval == 0 or idx == n_total - 1:
            elapsed = tm.time() - t0
            pct = (idx + 1) / n_total * 100
            viable = sum(1 for _, fr, _ in results if fr.fitness > 0)
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f"    [{pct:.0f}%] {idx+1}/{n_total} evaluated | "
                  f"{viable} viable | {elapsed:.0f}s elapsed | "
                  f"~{eta:.0f}s remaining", flush=True)

    return results
