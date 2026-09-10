"""
Options Walk-Forward Regime Simulator
======================================
Simulates the full OFA lifecycle on historical data:
  1. Train: search best strategy on rolling N-day window
  2. Validate: check on holdout period
  3. Forward-test: walk day-by-day until kill conditions fire
  4. Record: save regime record (train stats, forward stats, death reason)

Used to:
  - Generate regime records for classifier training
  - Calibrate kill conditions (DD thresholds, WR floors, etc.)
  - Validate the strategy lifecycle before going live

Usage:
    python options_walkforward.py                  # full historical simulation
    python options_walkforward.py --start 2024-06  # start from specific month
"""
from __future__ import annotations

# Version marker — increment when kill logic changes to detect stale RunPod builds
_WALKFORWARD_VERSION = "2026-03-16-killfix"

import json
import os
import sys
import time as tm
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from options_backtest import (
    evaluate_day, evaluate_day_multi, compute_fitness, FitnessResult, OptionsTrade,
    load_underlying, load_chain, load_chain_meta, get_available_days,
    CostModel, precompute_day_context, precompute_chain,
)
from options_genes import sample_candidates, OptionsGenes
from options_search import parallel_eval


# ─── Kill Conditions ─────────────────────────────────────────────────────────

@dataclass
class KillConfig:
    """Tunable kill condition thresholds.

    Calibrated from 60-regime walk-forward on 2024-03 → 2026-03 data.
    """
    max_dd_dollars: float = 500.0        # fixed DD limit per strategy
    max_single_trade_loss: float = 300.0 # kill if any single trade loses more than this (per-contract)
    scale_contracts: bool = True         # scale position size with profit
    scale_every: float = 750.0           # +1 contract per this much cum_pnl
    emergency_wr_floor: float = 0.20     # kill if WR < 20% after emergency_min_trades
    emergency_min_trades: int = 5        # early emergency check (before standard early WR)
    early_wr_floor: float = 0.35         # kill if WR < 35% after min trades
    early_min_trades: int = 10           # min trades before early WR check
    wr_lookback: int = 20                # rolling WR window
    wr_floor: float = 0.40              # kill if rolling WR < 40%
    max_consec_loss_days: int = 7        # kill after N consecutive red days
    flat_regime_pnl: float = 150.0       # |cum_pnl| < this after N trades = flat
    flat_regime_trades: int = 15         # min trades for flat check
    avg_pnl_floor: float = 3.0           # kill if avg PnL/trade < this after flat_regime_trades
    negative_trajectory_pnl: float = -500.0  # cum_pnl < this after N trades
    negative_trajectory_trades: int = 15
    max_regime_days: int = 365           # force kill after N calendar days
    live_score: bool = True              # reward proven strategies with wider thresholds
    live_score_min_trades: int = 10      # min trades before score can boost thresholds


def live_score_multiplier(
    cum_pnl: float,
    max_dd: float,
    n_trades: int,
    days_alive: int,
    min_trades: int = 10,
) -> float:
    """Compute a multiplier (1.0–2.0) that rewards proven live performance.

    Strategies that perform well earn wider DD/STL allowances.
    Three signals:
      - Return:DD ratio (capital efficiency)
      - Avg PnL per trade (edge quality)
      - Days alive (survival = signal)
    """
    if cum_pnl <= 0 or n_trades < min_trades:
        return 1.0

    ret_dd = cum_pnl / max(max_dd, 1.0)
    avg_pnl = cum_pnl / n_trades
    day_factor = min(days_alive / 30.0, 2.0)

    score = (
        0.4 * min(ret_dd, 3.0) / 3.0
        + 0.3 * min(avg_pnl, 50.0) / 50.0
        + 0.3 * day_factor / 2.0
    )

    return 1.0 + score  # 1.0 → 2.0


def get_contracts(cum_pnl: float, config: KillConfig) -> int:
    """Compute number of contracts based on cumulative PnL.

    Starts at 1 contract, scales up by 1 for every `scale_every` dollars of profit.
    Never scales below 1 contract.
    """
    if not config.scale_contracts or cum_pnl <= 0:
        return 1
    return 1 + int(cum_pnl / config.scale_every)


def check_kill(
    trades: list[OptionsTrade],
    config: KillConfig,
    start_date: date,
    current_date: date,
    contracts_history: list[int] | None = None,
    check_stl_from: int = 0,
) -> tuple[bool, str]:
    """Check if the regime should be killed.

    Proven strategies earn wider DD/STL allowances via live_score_multiplier.
    The `contracts_history` list tracks how many contracts each trade was executed
    with, so scaled PnL is used for DD calculations.

    Args:
        check_stl_from: check STL on all trades from this index onward.
            Covers multi-entry where multiple trades are appended between
            kill checks. Default 0 = check all trades.

    Returns (should_kill, reason_string).
    """
    n = len(trades)
    if n == 0:
        # Check time limit even with no trades
        days_elapsed = (current_date - start_date).days
        if days_elapsed >= config.max_regime_days:
            return True, f"time_limit ({days_elapsed}d, 0 trades)"
        return False, ""

    days_alive = (current_date - start_date).days

    # Compute scaled PnL series (each trade's pnl * its contract count)
    if contracts_history and len(contracts_history) == n:
        scaled_pnls = [t.pnl_net * c for t, c in zip(trades, contracts_history)]
    else:
        # Length mismatch or missing — fall back to unscaled
        if contracts_history and len(contracts_history) != n:
            print(f"  WARNING: contracts_history len {len(contracts_history)} != trades len {n}, using unscaled")
        scaled_pnls = [t.pnl_net for t in trades]

    cum_pnl = sum(scaled_pnls)

    # Drawdown from peak (on scaled PnL curve) — computed early for live score
    running_pnl = np.concatenate(([0.0], np.cumsum(scaled_pnls)))
    peak = np.maximum.accumulate(running_pnl)
    dd = peak - running_pnl
    max_dd = float(dd.max())

    # Live performance score — reward proven strategies with wider thresholds
    if config.live_score:
        mult = live_score_multiplier(
            cum_pnl, max_dd, n, days_alive, config.live_score_min_trades,
        )
    else:
        mult = 1.0

    # Contract count for DD scaling — use current equity, not stale history
    cur_contracts = get_contracts(cum_pnl, config)

    eff_dd_limit = config.max_dd_dollars * mult * cur_contracts
    eff_stl_limit = config.max_single_trade_loss * mult  # per-contract, no scaling

    # Per-trade loss cap — check ALL trades from check_stl_from onward
    # (catches multi-entry trades that were appended since last kill check)
    for i in range(max(check_stl_from, 0), n):
        trade_pnl = trades[i].pnl_net
        if trade_pnl < -eff_stl_limit:
            return True, (
                f"single_trade_loss ${trade_pnl:.0f}/ct"
                f" < -${eff_stl_limit:.0f} (base ${config.max_single_trade_loss:.0f} x{mult:.2f})"
            )

    # Drawdown check (on scaled PnL curve, DD allowance scales with contracts)
    if max_dd >= eff_dd_limit:
        return True, (
            f"drawdown ${max_dd:.0f} >= ${eff_dd_limit:.0f}"
            f" (base ${config.max_dd_dollars:.0f} x{mult:.2f} x{cur_contracts}ct) after {n} trades"
        )

    # Emergency early WR check (5 trades)
    if config.emergency_min_trades <= n < config.early_min_trades:
        wr = sum(1 for t in trades if t.result == "win") / n
        if wr < config.emergency_wr_floor:
            return True, f"emergency_wr {wr:.1%} < {config.emergency_wr_floor:.0%} after {n} trades"

    # Early WR check (10 trades)
    if n >= config.early_min_trades:
        wr = sum(1 for t in trades if t.result == "win") / n
        if wr < config.early_wr_floor:
            return True, f"early_wr {wr:.1%} < {config.early_wr_floor:.0%} after {n} trades"

    # Rolling WR check
    if n >= config.wr_lookback:
        recent = trades[-config.wr_lookback:]
        recent_wr = sum(1 for t in recent if t.result == "win") / len(recent)
        if recent_wr < config.wr_floor:
            return True, f"rolling_wr {recent_wr:.1%} < {config.wr_floor:.0%} (last {config.wr_lookback})"

    # Consecutive losing days
    daily_pnl = {}
    for t in trades:
        d = t.trade_date
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl_net
    sorted_days = sorted(daily_pnl.keys())
    if len(sorted_days) >= config.max_consec_loss_days:
        consec = 0
        max_consec = 0
        for d in sorted_days:
            if daily_pnl[d] < 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        if max_consec >= config.max_consec_loss_days:
            return True, f"consecutive_loss_days {max_consec} >= {config.max_consec_loss_days}"

    # Negative trajectory
    if n >= config.negative_trajectory_trades and cum_pnl < config.negative_trajectory_pnl:
        return True, f"negative_trajectory ${cum_pnl:.0f} after {n} trades"

    # Flat regime (tighter: 15 trades, $150)
    if n >= config.flat_regime_trades and abs(cum_pnl) < config.flat_regime_pnl:
        return True, f"flat_regime ${cum_pnl:.0f} after {n} trades"

    # Avg PnL/trade floor — catches slow bleeders
    if n >= config.flat_regime_trades:
        avg_pnl = cum_pnl / n
        if avg_pnl < config.avg_pnl_floor:
            return True, f"low_avg_pnl ${avg_pnl:.1f}/trade < ${config.avg_pnl_floor:.0f} after {n} trades"

    # Time limit
    days_elapsed = (current_date - start_date).days
    if days_elapsed >= config.max_regime_days:
        return True, f"time_limit ({days_elapsed}d, {n} trades, ${cum_pnl:.0f})"

    return False, ""


def kill_state_snapshot(
    trades: list[OptionsTrade],
    config: KillConfig,
    start_date: date,
    current_date: date,
    contracts_history: list[int] | None = None,
) -> dict:
    """Return a diagnostic snapshot of all kill condition metrics.

    Does NOT trigger a kill — just computes and returns the current state
    so it can be logged for debugging. Call this every day for every active
    strategy to understand why kills are (or aren't) firing.
    """
    n = len(trades)
    days_alive = (current_date - start_date).days

    if n == 0:
        return {
            "n_trades": 0,
            "days_alive": days_alive,
            "cum_pnl": 0.0,
            "max_dd": 0.0,
            "eff_dd_limit": config.max_dd_dollars,
            "dd_headroom": config.max_dd_dollars,
            "live_score_mult": 1.0,
            "cur_contracts": 1,
            "wr_all": 0.0,
            "wr_rolling_20": 0.0,
            "avg_pnl_per_trade": 0.0,
            "max_consec_loss_days": 0,
            "worst_single_trade": 0.0,
            "eff_stl_limit": config.max_single_trade_loss,
            "nearest_kill": "none (0 trades)",
        }

    # Compute scaled PnL
    if contracts_history and len(contracts_history) == n:
        scaled_pnls = [t.pnl_net * c for t, c in zip(trades, contracts_history)]
    else:
        scaled_pnls = [t.pnl_net for t in trades]

    cum_pnl = sum(scaled_pnls)
    running_pnl = np.concatenate(([0.0], np.cumsum(scaled_pnls)))
    peak = np.maximum.accumulate(running_pnl)
    dd = peak - running_pnl
    max_dd = float(dd.max())

    if config.live_score:
        mult = live_score_multiplier(
            cum_pnl, max_dd, n, days_alive, config.live_score_min_trades,
        )
    else:
        mult = 1.0

    cur_contracts = get_contracts(cum_pnl, config)
    eff_dd_limit = config.max_dd_dollars * mult * cur_contracts
    eff_stl_limit = config.max_single_trade_loss * mult

    # Win rate (all trades)
    wr_all = sum(1 for t in trades if t.result == "win") / n

    # Rolling WR (last 20)
    if n >= config.wr_lookback:
        recent = trades[-config.wr_lookback:]
        wr_rolling = sum(1 for t in recent if t.result == "win") / len(recent)
    else:
        wr_rolling = wr_all  # not enough for rolling, show all-time

    avg_pnl = cum_pnl / n

    # Consecutive losing days
    daily_pnl_map: dict[str, float] = {}
    for t in trades:
        d = t.trade_date
        daily_pnl_map[d] = daily_pnl_map.get(d, 0) + t.pnl_net
    sorted_trade_days = sorted(daily_pnl_map.keys())
    consec = 0
    max_consec_losses = 0
    for td in sorted_trade_days:
        if daily_pnl_map[td] < 0:
            consec += 1
            max_consec_losses = max(max_consec_losses, consec)
        else:
            consec = 0

    # Worst single trade (per-contract, unscaled)
    worst_single = min(t.pnl_net for t in trades)

    # Determine which kill condition is closest to firing
    nearest_kill = _nearest_kill_condition(
        n, cum_pnl, max_dd, eff_dd_limit, eff_stl_limit,
        wr_all, wr_rolling, avg_pnl, max_consec_losses, worst_single,
        days_alive, config,
    )

    return {
        "n_trades": n,
        "days_alive": days_alive,
        "cum_pnl": round(cum_pnl, 2),
        "max_dd": round(max_dd, 2),
        "eff_dd_limit": round(eff_dd_limit, 2),
        "dd_headroom": round(eff_dd_limit - max_dd, 2),
        "live_score_mult": round(mult, 3),
        "cur_contracts": cur_contracts,
        "wr_all": round(wr_all, 4),
        "wr_rolling_20": round(wr_rolling, 4),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "max_consec_loss_days": max_consec_losses,
        "worst_single_trade": round(worst_single, 2),
        "eff_stl_limit": round(eff_stl_limit, 2),
        "nearest_kill": nearest_kill,
    }


def _nearest_kill_condition(
    n: int, cum_pnl: float, max_dd: float,
    eff_dd_limit: float, eff_stl_limit: float,
    wr_all: float, wr_rolling: float, avg_pnl: float,
    max_consec_losses: int, worst_single: float,
    days_alive: int, config: KillConfig,
) -> str:
    """Identify which kill condition is closest to triggering."""
    candidates = []

    # DD proximity (percentage of limit used)
    dd_pct = max_dd / max(eff_dd_limit, 1.0)
    candidates.append((dd_pct, f"dd {dd_pct:.0%} of limit (${max_dd:.0f}/${eff_dd_limit:.0f})"))

    # STL proximity
    if worst_single < 0:
        stl_pct = abs(worst_single) / max(eff_stl_limit, 1.0)
        candidates.append((stl_pct, f"stl {stl_pct:.0%} of limit (${worst_single:.0f}/${eff_stl_limit:.0f})"))

    # WR checks
    if config.emergency_min_trades <= n < config.early_min_trades:
        wr_gap = config.emergency_wr_floor - wr_all
        if wr_gap > 0:
            candidates.append((1.0, f"emergency_wr {wr_all:.0%} BELOW {config.emergency_wr_floor:.0%}"))
        else:
            wr_margin = wr_all - config.emergency_wr_floor
            candidates.append((1.0 - wr_margin, f"emergency_wr {wr_all:.0%} margin={wr_margin:.0%}"))
    elif n >= config.early_min_trades:
        wr_gap = config.early_wr_floor - wr_all
        if wr_gap > 0:
            candidates.append((1.0, f"early_wr {wr_all:.0%} BELOW {config.early_wr_floor:.0%}"))
        else:
            wr_margin = wr_all - config.early_wr_floor
            candidates.append((1.0 - wr_margin, f"early_wr {wr_all:.0%} margin={wr_margin:.0%}"))

    if n >= config.wr_lookback:
        rolling_gap = config.wr_floor - wr_rolling
        if rolling_gap > 0:
            candidates.append((1.0, f"rolling_wr {wr_rolling:.0%} BELOW {config.wr_floor:.0%}"))
        else:
            rolling_margin = wr_rolling - config.wr_floor
            candidates.append((1.0 - rolling_margin, f"rolling_wr {wr_rolling:.0%} margin={rolling_margin:.0%}"))

    # Flat regime
    if n >= config.flat_regime_trades:
        if abs(cum_pnl) < config.flat_regime_pnl:
            candidates.append((1.0, f"flat_regime |${cum_pnl:.0f}| < ${config.flat_regime_pnl:.0f}"))
        if avg_pnl < config.avg_pnl_floor:
            candidates.append((1.0, f"avg_pnl ${avg_pnl:.1f} < ${config.avg_pnl_floor:.0f}"))

    # Consec losses
    consec_pct = max_consec_losses / config.max_consec_loss_days
    candidates.append((consec_pct, f"consec_loss {max_consec_losses}/{config.max_consec_loss_days}"))

    # Time limit
    time_pct = days_alive / config.max_regime_days
    candidates.append((time_pct, f"time {days_alive}/{config.max_regime_days}d"))

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] if candidates else "unknown"


# ─── Regime Record ───────────────────────────────────────────────────────────

@dataclass
class RegimeRecord:
    """Complete record of a strategy regime lifecycle."""
    regime_id: int
    train_start: str
    train_end: str
    forward_start: str
    forward_end: str
    death_reason: str

    # Strategy info
    genes_desc: str
    genes: dict

    # Training stats
    train_fitness: float
    train_pnl: float
    train_wr: float
    train_sharpe: float
    train_pf: float
    train_n_trades: int

    # Forward stats
    forward_trades: int
    forward_pnl: float
    forward_wr: float
    forward_max_dd: float
    forward_days: int
    forward_avg_pnl: float


# ─── Walk-Forward Simulation ─────────────────────────────────────────────────

def run_walkforward(
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    train_days: int = 60,
    val_days: int = 20,
    step_days: int = 5,
    n_candidates: int = 5000,
    kill_config: KillConfig | None = None,
    cost: CostModel | None = None,
    min_trades: int = 30,
    start_from: date | None = None,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
    regime_id_offset: int = 0,
) -> list[RegimeRecord]:
    """Run full walk-forward simulation on historical data.

    Slides a training window forward, for each window:
    1. Search for best strategy on train period
    2. Validate on val period
    3. Walk forward day-by-day until kill or data runs out
    4. Record regime

    Args:
        data_dir: path to data directory
        train_days: training window size
        val_days: validation holdout size
        step_days: how many days to step forward between regimes
        n_candidates: candidates per search
        kill_config: kill condition thresholds
        start_from: skip to this date for train start
        chunk_start: first window index to process (for parallelization)
        chunk_end: last window index (exclusive) to process
        regime_id_offset: offset for regime IDs (so chunks don't collide)

    Returns:
        List of RegimeRecord for all simulated regimes.
    """
    if kill_config is None:
        kill_config = KillConfig()
    if cost is None:
        cost = CostModel()

    all_days = get_available_days(data_dir, chain_subdir)
    if not all_days:
        print("No data available!")
        return []

    print(f"Walk-forward simulation")
    print(f"  Data: {len(all_days)} days ({all_days[0]} to {all_days[-1]})")
    print(f"  Train: {train_days}d, Val: {val_days}d, Step: {step_days}d")
    print(f"  Candidates: {n_candidates}")
    print(f"  Kill: DD=${kill_config.max_dd_dollars:.0f}, "
          f"WR floor={kill_config.wr_floor:.0%}, "
          f"consec loss={kill_config.max_consec_loss_days}d")
    print()

    # Preload all data once
    print("  Loading data...", end=" ", flush=True)
    t0 = tm.time()
    und_cache = {}
    chain_cache = {}
    meta_cache = {}
    for d in all_days:
        und = load_underlying(data_dir, d)
        if not und.empty:
            und_cache[d] = und
        chain = load_chain(data_dir, d, chain_subdir)
        if not chain.empty:
            chain_cache[d] = chain
        meta = load_chain_meta(data_dir, d, chain_subdir)
        if meta:
            meta_cache[d] = meta

    # Add prev day stats to metas
    sorted_days = sorted(und_cache.keys())
    for i, d in enumerate(sorted_days):
        if d in meta_cache and i > 0:
            prev_d = sorted_days[i - 1]
            if prev_d in und_cache:
                prev_und = und_cache[prev_d]
                meta_cache[d]["prev_close"] = float(prev_und.iloc[-1]["c"])
                meta_cache[d]["prev_high"] = float(prev_und["h"].max())
                meta_cache[d]["prev_low"] = float(prev_und["l"].min())

    # Precompute chains
    for d in list(chain_cache.keys()):
        chain_cache[d] = precompute_chain(chain_cache[d])

    # Precompute contexts
    ctx_cache = {}
    for d in all_days:
        if d in und_cache:
            chain_for_ctx = chain_cache.get(d)
            meta_for_ctx = meta_cache.get(d)
            ctx = precompute_day_context(und_cache[d], chain_for_ctx, meta_for_ctx)
            if ctx is not None:
                ctx_cache[d] = ctx

    valid_days = [d for d in all_days
                  if d in und_cache and d in chain_cache
                  and d in meta_cache and d in ctx_cache]

    print(f"{len(valid_days)} valid days in {tm.time()-t0:.1f}s")

    # Compute all possible window start indices
    base_start = 0
    if start_from:
        for i, d in enumerate(valid_days):
            if d >= start_from:
                base_start = i
                break

    all_window_starts = []
    idx = base_start
    while idx + train_days + val_days < len(valid_days):
        all_window_starts.append(idx)
        idx += step_days

    total_windows = len(all_window_starts)
    print(f"  Total regime windows: {total_windows}")

    # Apply chunk bounds for parallelization
    if chunk_start is not None:
        c_start = max(0, chunk_start)
        c_end = min(total_windows, chunk_end) if chunk_end is not None else total_windows
        all_window_starts = all_window_starts[c_start:c_end]
        print(f"  Chunk: windows [{c_start}:{c_end}] = {len(all_window_starts)} windows")

    # Walk forward
    regimes = []
    regime_id = regime_id_offset

    for wi, window_start_idx in enumerate(all_window_starts):
        train_end_idx = window_start_idx + train_days
        val_end_idx = train_end_idx + val_days
        forward_start_idx = val_end_idx

        if forward_start_idx >= len(valid_days):
            break

        train_period = valid_days[window_start_idx:train_end_idx]
        val_period = valid_days[train_end_idx:val_end_idx]
        forward_days_avail = valid_days[forward_start_idx:]

        if len(train_period) < train_days // 2 or len(val_period) < val_days // 2:
            window_start_idx += step_days
            continue

        regime_id += 1
        print(f"\n{'='*60}")
        print(f"  Regime {regime_id}: train {train_period[0]}→{train_period[-1]} "
              f"({len(train_period)}d), val {val_period[0]}→{val_period[-1]} "
              f"({len(val_period)}d)")

        # Phase 1: Gene search on training period
        t1 = tm.time()
        candidates = sample_candidates(n_candidates, seed=regime_id * 1000)

        results = parallel_eval(
            candidates, train_period, data_dir, chain_subdir,
            preloaded_und=und_cache, preloaded_chains=chain_cache,
            preloaded_metas=meta_cache, preloaded_ctxs=ctx_cache,
            cost=cost, min_trades=min_trades,
        )

        # Find best by fitness
        viable = [(idx, fr, s) for idx, fr, s in results if fr.fitness > 0]
        if not viable:
            print(f"    No viable strategies (0/{n_candidates}). Stepping forward.")
            window_start_idx += step_days
            continue

        viable.sort(key=lambda x: x[1].fitness, reverse=True)
        best_idx, best_fr, best_summary = viable[0]
        best_genes = candidates[best_idx]
        gene_dict = best_genes.to_dict()

        print(f"    Search: {len(viable)} viable in {tm.time()-t1:.0f}s")
        print(f"    Winner: {best_genes.describe()}")
        print(f"    Train: fitness={best_fr.fitness:.3f} pnl=${best_fr.cum_pnl:+.0f} "
              f"WR={best_fr.win_rate:.0%} sharpe={best_fr.sharpe:.2f}")

        # Phase 2: Validate on holdout
        val_trades = []
        for d in val_period:
            if d in und_cache and d in chain_cache and d in meta_cache:
                trade = evaluate_day(
                    und_cache[d], chain_cache[d], meta_cache[d],
                    gene_dict, cost, _ctx=ctx_cache.get(d),
                )
                if trade is not None:
                    val_trades.append(trade)

        val_fitness = compute_fitness(val_trades, min_trades=5)
        val_pnl = sum(t.pnl_net for t in val_trades)
        val_wr = sum(1 for t in val_trades if t.result == "win") / len(val_trades) if val_trades else 0

        print(f"    Val: {len(val_trades)} trades, pnl=${val_pnl:+.0f}, "
              f"WR={val_wr:.0%}, fitness={val_fitness.fitness:.3f}")

        if val_fitness.fitness <= 0 or val_pnl <= 0:
            print(f"    FAILED validation. Stepping forward.")
            window_start_idx += step_days
            continue

        # Phase 3: Forward test — walk day-by-day until kill
        print(f"    Forward testing from {forward_days_avail[0]}...")
        forward_trades = []
        death_reason = "alive"
        forward_end = forward_days_avail[0]

        for d in forward_days_avail:
            if d in und_cache and d in chain_cache and d in meta_cache:
                trade = evaluate_day(
                    und_cache[d], chain_cache[d], meta_cache[d],
                    gene_dict, cost, _ctx=ctx_cache.get(d),
                )
                if trade is not None:
                    forward_trades.append(trade)

            forward_end = d

            # Check kill conditions
            killed, reason = check_kill(
                forward_trades, kill_config,
                forward_days_avail[0], d,
            )
            if killed:
                death_reason = reason
                break

        # Record regime
        fwd_pnl = sum(t.pnl_net for t in forward_trades)
        fwd_wr = (sum(1 for t in forward_trades if t.result == "win") / len(forward_trades)
                  if forward_trades else 0)
        fwd_dd = 0.0
        if forward_trades:
            running = np.concatenate(([0.0], np.cumsum([t.pnl_net for t in forward_trades])))
            peak = np.maximum.accumulate(running)
            fwd_dd = float((peak - running).max())
        fwd_days = (forward_end - forward_days_avail[0]).days if forward_days_avail else 0

        regime = RegimeRecord(
            regime_id=regime_id,
            train_start=str(train_period[0]),
            train_end=str(train_period[-1]),
            forward_start=str(forward_days_avail[0]) if forward_days_avail else "",
            forward_end=str(forward_end),
            death_reason=death_reason,
            genes_desc=best_genes.describe(),
            genes=gene_dict,
            train_fitness=best_fr.fitness,
            train_pnl=best_fr.cum_pnl,
            train_wr=best_fr.win_rate,
            train_sharpe=best_fr.sharpe,
            train_pf=best_fr.profit_factor,
            train_n_trades=best_summary["n_trades"],
            forward_trades=len(forward_trades),
            forward_pnl=round(fwd_pnl, 2),
            forward_wr=round(fwd_wr, 4),
            forward_max_dd=round(fwd_dd, 2),
            forward_days=fwd_days,
            forward_avg_pnl=round(fwd_pnl / len(forward_trades), 2) if forward_trades else 0,
        )
        regimes.append(regime)

        status = "SURVIVED" if death_reason == "alive" else f"KILLED: {death_reason}"
        print(f"    Forward: {len(forward_trades)} trades over {fwd_days}d, "
              f"pnl=${fwd_pnl:+.0f}, WR={fwd_wr:.0%}, DD=${fwd_dd:.0f}")
        print(f"    {status}")

        # Step forward — use step_days or jump past the death date
        window_start_idx += step_days

    # Summary
    print(f"\n{'='*60}")
    print(f"Walk-forward complete: {len(regimes)} regimes")
    if regimes:
        positive = sum(1 for r in regimes if r.forward_pnl > 0)
        total_pnl = sum(r.forward_pnl for r in regimes)
        avg_days = sum(r.forward_days for r in regimes) / len(regimes)
        print(f"  Positive: {positive}/{len(regimes)} ({positive/len(regimes):.0%})")
        print(f"  Total forward PnL: ${total_pnl:+,.0f}")
        print(f"  Avg regime lifespan: {avg_days:.0f} days")

        # Death reason breakdown
        reasons = {}
        for r in regimes:
            key = r.death_reason.split(" ")[0] if r.death_reason != "alive" else "alive"
            reasons[key] = reasons.get(key, 0) + 1
        print(f"  Death reasons: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")

    return regimes


# ─── Live Portfolio Simulation ──────────────────────────────────────────────

@dataclass
class ActiveStrategy:
    """A live strategy with its trade history for kill tracking."""
    strategy_id: int
    genes: dict
    genes_desc: str
    start_date: date
    trades: list  # list[OptionsTrade]
    contracts_history: list  # list[int] — contracts per trade
    train_fitness: float
    train_pnl: float
    train_wr: float
    trades_at_last_kill_check: int = 0  # index for STL scanning


@dataclass
class DayResult:
    """Portfolio-level result for a single trading day."""
    date: str
    n_active: int
    trades_taken: int
    day_pnl: float
    cum_pnl: float
    portfolio_dd: float          # current portfolio drawdown from peak
    max_portfolio_dd: float      # running max portfolio DD so far
    strategy_dds: dict           # {strategy_id: current_dd} for each active strategy
    strategies_added: int
    strategies_killed: int
    kill_reasons: list


def run_live_sim(
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    train_days: int = 60,
    val_days: int = 20,
    retrain_every: int = 5,
    n_candidates: int = 5000,
    max_concurrent: int = 0,
    kill_config: KillConfig | None = None,
    cost: CostModel | None = None,
    min_trades: int = 30,
    start_from: date | None = None,
    multi_entry: bool = False,
    force_trade_type: dict | None = None,
    force_contracts: int | None = None,
    allowed_trade_types: list[dict] | None = None,
    classifier_fn: "Callable[[dict], float] | None" = None,
    classifier_threshold: float = 0.5,
    verbose_kill_log: bool = True,
) -> dict:
    """Simulate live portfolio deployment over full history.

    Unlike run_walkforward which tests regimes independently, this walks
    through every day chronologically maintaining a portfolio of active
    strategies — exactly as live deployment would work.

    Every `retrain_every` days:
      1. Train on prior `train_days`, validate on prior `val_days`
      2. If strategy passes validation, add it (no pool cap; portfolio DD is the risk control)

    Every day:
      1. Check kill conditions on all active strategies
      2. Trade all surviving strategies (each can produce 0-1 trades)
      3. Record portfolio-level PnL and per-strategy drawdowns

    Args:
        max_concurrent: 0 = unlimited (default). >0 = cap active strategies.
        verbose_kill_log: if True, log kill condition state for every active
            strategy every day. Essential for debugging kill failures.

    Returns dict with daily equity curve and regime lifecycle records.
    """
    if kill_config is None:
        kill_config = KillConfig()
    if cost is None:
        cost = CostModel()

    # ── Load all data ──
    all_days = get_available_days(data_dir, chain_subdir)
    if not all_days:
        print("No data available!")
        return {"error": "no data"}

    print(f"Live portfolio simulation (walkforward version: {_WALKFORWARD_VERSION})")
    print(f"  Data: {len(all_days)} days ({all_days[0]} to {all_days[-1]})")
    print(f"  Train: {train_days}d, Val: {val_days}d, Retrain every: {retrain_every}d")
    print(f"  Max concurrent strategies: {'unlimited' if max_concurrent == 0 else max_concurrent}")
    print(f"  Contract scaling: {'ON (every ${:.0f})'.format(kill_config.scale_every) if kill_config.scale_contracts else 'OFF'}")
    print(f"  Multi-entry: {'ON' if multi_entry else 'OFF'}")
    print(f"  Kill: DD=${kill_config.max_dd_dollars:.0f}, STL=${kill_config.max_single_trade_loss:.0f}")
    print(f"  Candidates per search: {n_candidates}")
    if classifier_fn is not None:
        print(f"  Classifier gate: ON (threshold={classifier_threshold:.2f})")
    else:
        print(f"  Classifier gate: OFF")
    print()

    print("  Loading data...", end=" ", flush=True)
    t0 = tm.time()
    und_cache = {}
    chain_cache = {}
    meta_cache = {}
    for d in all_days:
        und = load_underlying(data_dir, d)
        if not und.empty:
            und_cache[d] = und
        chain = load_chain(data_dir, d, chain_subdir)
        if not chain.empty:
            chain_cache[d] = chain
        meta = load_chain_meta(data_dir, d, chain_subdir)
        if meta:
            meta_cache[d] = meta

    sorted_all = sorted(und_cache.keys())
    for i, d in enumerate(sorted_all):
        if d in meta_cache and i > 0:
            prev_d = sorted_all[i - 1]
            if prev_d in und_cache:
                prev_und = und_cache[prev_d]
                meta_cache[d]["prev_close"] = float(prev_und.iloc[-1]["c"])
                meta_cache[d]["prev_high"] = float(prev_und["h"].max())
                meta_cache[d]["prev_low"] = float(prev_und["l"].min())

    for d in list(chain_cache.keys()):
        chain_cache[d] = precompute_chain(chain_cache[d])

    ctx_cache = {}
    for d in all_days:
        if d in und_cache:
            ctx = precompute_day_context(
                und_cache[d], chain_cache.get(d), meta_cache.get(d)
            )
            if ctx is not None:
                ctx_cache[d] = ctx

    valid_days = [d for d in all_days
                  if d in und_cache and d in chain_cache
                  and d in meta_cache and d in ctx_cache]

    print(f"{len(valid_days)} valid days in {tm.time()-t0:.1f}s")

    # First tradeable day is after train + val warmup
    warmup = train_days + val_days

    # If start_from is set, find the warmup offset so first trade day >= start_from
    if start_from is not None:
        # We need warmup days of data BEFORE start_from for training
        # Find the index of the first valid day >= start_from
        start_idx = None
        for i, d in enumerate(valid_days):
            if d >= start_from:
                start_idx = i
                break
        if start_idx is not None and start_idx >= warmup:
            warmup = start_idx
            print(f"  start_from={start_from}: first trade day at index {warmup} ({valid_days[warmup]})")

    if len(valid_days) <= warmup:
        return {"error": f"Not enough data: {len(valid_days)} days, need >{warmup}"}

    # ── Main simulation loop ──
    active: list[ActiveStrategy] = []
    all_day_results: list[DayResult] = []
    all_regime_records: list[RegimeRecord] = []
    strategy_counter = 0
    cum_pnl = 0.0
    portfolio_peak = 0.0
    max_portfolio_dd = 0.0
    days_since_retrain = retrain_every  # trigger retrain on first eligible day

    print(f"\n  Starting live sim from day {warmup} ({valid_days[warmup]})...")

    for day_idx in range(warmup, len(valid_days)):
        d = valid_days[day_idx]
        day_pnl = 0.0
        trades_taken = 0
        killed_today = []
        added_today = 0

        # ── Step 1: Check kills on all active strategies ──
        surviving = []
        for strat in active:
            # Verbose kill diagnostics — log state BEFORE the kill decision
            if verbose_kill_log and strat.trades:
                snap = kill_state_snapshot(
                    strat.trades, kill_config, strat.start_date, d,
                    contracts_history=strat.contracts_history,
                )
                print(
                    f"    [KILL-CHECK {d}] strat={strat.strategy_id} | "
                    f"{snap['n_trades']}t {snap['days_alive']}d | "
                    f"pnl=${snap['cum_pnl']:+.0f} dd=${snap['max_dd']:.0f}/"
                    f"${snap['eff_dd_limit']:.0f} | "
                    f"WR={snap['wr_all']:.0%} roll={snap['wr_rolling_20']:.0%} | "
                    f"avg=${snap['avg_pnl_per_trade']:.1f} | "
                    f"mult={snap['live_score_mult']:.2f} {snap['cur_contracts']}ct | "
                    f"nearest: {snap['nearest_kill']}",
                    flush=True,
                )

            killed, reason = check_kill(
                strat.trades, kill_config, strat.start_date, d,
                contracts_history=strat.contracts_history,
                check_stl_from=strat.trades_at_last_kill_check,
            )
            strat.trades_at_last_kill_check = len(strat.trades)
            if killed:
                killed_today.append(reason)
                # Record completed regime (using scaled PnL)
                fwd_trades = strat.trades
                scaled_pnls = [t.pnl_net * c for t, c in zip(fwd_trades, strat.contracts_history)] if fwd_trades else []
                fwd_pnl = sum(scaled_pnls)
                fwd_wr = (sum(1 for t in fwd_trades if t.result == "win")
                          / len(fwd_trades)) if fwd_trades else 0
                fwd_dd = 0.0
                if scaled_pnls:
                    running = np.concatenate(([0.0], np.cumsum(scaled_pnls)))
                    peak = np.maximum.accumulate(running)
                    fwd_dd = float((peak - running).max())
                fwd_days = (d - strat.start_date).days
                max_contracts = max(strat.contracts_history) if strat.contracts_history else 1

                regime = RegimeRecord(
                    regime_id=strat.strategy_id,
                    train_start="",
                    train_end="",
                    forward_start=str(strat.start_date),
                    forward_end=str(d),
                    death_reason=reason,
                    genes_desc=strat.genes_desc,
                    genes=strat.genes,
                    train_fitness=strat.train_fitness,
                    train_pnl=strat.train_pnl,
                    train_wr=strat.train_wr,
                    train_sharpe=0.0,
                    train_pf=0.0,
                    train_n_trades=0,
                    forward_trades=len(fwd_trades),
                    forward_pnl=round(fwd_pnl, 2),
                    forward_wr=round(fwd_wr, 4),
                    forward_max_dd=round(fwd_dd, 2),
                    forward_days=fwd_days,
                    forward_avg_pnl=round(fwd_pnl / len(fwd_trades), 2) if fwd_trades else 0,
                )
                all_regime_records.append(regime)
                print(f"    [{d}] KILLED strategy {strat.strategy_id}: {reason} "
                      f"({len(fwd_trades)} trades, ${fwd_pnl:+.0f}, max {max_contracts}ct)")
            else:
                surviving.append(strat)
        active = surviving

        # ── Step 2: Retrain if it's time ──
        days_since_retrain += 1
        has_room = max_concurrent == 0 or len(active) < max_concurrent
        if days_since_retrain >= retrain_every and has_room:
            days_since_retrain = 0

            # Training window: last train_days before val window
            # Val window: last val_days before today
            val_end_idx = day_idx
            val_start_idx = max(0, val_end_idx - val_days)
            train_end_idx = val_start_idx
            train_start_idx = max(0, train_end_idx - train_days)

            train_period = valid_days[train_start_idx:train_end_idx]
            val_period = valid_days[val_start_idx:val_end_idx]

            if len(train_period) >= train_days // 2 and len(val_period) >= val_days // 2:
                t1 = tm.time()
                strategy_counter += 1
                candidates = sample_candidates(
                    n_candidates, seed=strategy_counter * 1000 + day_idx,
                    allowed_trade_types=allowed_trade_types,
                )

                results = parallel_eval(
                    candidates, train_period, data_dir, chain_subdir,
                    preloaded_und=und_cache, preloaded_chains=chain_cache,
                    preloaded_metas=meta_cache, preloaded_ctxs=ctx_cache,
                    cost=cost, min_trades=min_trades,
                )

                viable = [(idx, fr, s) for idx, fr, s in results if fr.fitness > 0]
                if viable:
                    viable.sort(key=lambda x: x[1].fitness, reverse=True)
                    best_idx, best_fr, best_summary = viable[0]
                    best_genes = candidates[best_idx]
                    gene_dict = best_genes.to_dict()

                    # Override trade type if forced (e.g., spreads instead of naked)
                    if force_trade_type is not None:
                        gene_dict["trade_type"] = force_trade_type

                    # Validate
                    val_trades = []
                    for vd in val_period:
                        if vd in und_cache and vd in chain_cache and vd in meta_cache:
                            trade = evaluate_day(
                                und_cache[vd], chain_cache[vd], meta_cache[vd],
                                gene_dict, cost, _ctx=ctx_cache.get(vd),
                            )
                            if trade is not None:
                                val_trades.append(trade)

                    val_fitness = compute_fitness(val_trades, min_trades=5)
                    val_pnl = sum(t.pnl_net for t in val_trades)

                    if val_fitness.fitness > 0 and val_pnl > 0:
                        # ── Classifier gate ──
                        _passed_classifier = True
                        _clf_prob = None
                        if classifier_fn is not None:
                            _train_fr_dict = {
                                "fitness": best_fr.fitness,
                                "sharpe": best_fr.sharpe,
                                "profit_factor": best_fr.profit_factor,
                                "win_rate": best_fr.win_rate,
                                "n_trades": best_fr.n_trades,
                                "avg_r": best_fr.avg_r,
                                "cum_r": best_fr.cum_r,
                                "max_dd_r": best_fr.max_dd_r,
                                "avg_pnl": best_fr.avg_pnl,
                                "cum_pnl": best_fr.cum_pnl,
                                "avg_win": best_fr.avg_win,
                                "avg_loss": best_fr.avg_loss,
                                "max_drawdown": best_fr.max_drawdown,
                            }
                            _val_fr_dict = {
                                "fitness": val_fitness.fitness,
                                "sharpe": val_fitness.sharpe,
                                "profit_factor": val_fitness.profit_factor,
                                "win_rate": val_fitness.win_rate,
                                "n_trades": val_fitness.n_trades,
                                "avg_pnl": val_fitness.avg_pnl,
                            }
                            _rob = {}
                            try:
                                _overfit_dir = Path(__file__).parent.parent / "overfit_finder"
                                if str(_overfit_dir) not in sys.path:
                                    sys.path.insert(0, str(_overfit_dir))
                                from robustness_metrics import compute_robustness_metrics
                                _train_pnls = [t.pnl_net for t in val_trades]
                                # Re-eval train trades for robustness
                                _train_trades_rob = []
                                for _td in train_period:
                                    if _td in und_cache and _td in chain_cache and _td in meta_cache:
                                        _t = evaluate_day(
                                            und_cache[_td], chain_cache[_td], meta_cache[_td],
                                            gene_dict, cost, _ctx=ctx_cache.get(_td),
                                        )
                                        if _t is not None:
                                            _train_trades_rob.append(_t)
                                _train_pnls = [t.pnl_net for t in _train_trades_rob]
                                _rob_result = compute_robustness_metrics(
                                    _train_pnls, window_days=len(train_period),
                                )
                                if _rob_result is not None:
                                    _rob = _rob_result.to_dict()
                            except Exception:
                                pass

                            _regime_dict = {
                                "train_fitness": _train_fr_dict,
                                "val_fitness": _val_fr_dict,
                                "val_cum_pnl": val_pnl,
                                "genes": gene_dict,
                                "archetype": best_genes.archetype,
                                "direction": best_genes.direction,
                                "robustness": _rob,
                            }
                            _clf_prob = classifier_fn(_regime_dict)
                            _passed_classifier = _clf_prob >= classifier_threshold

                        if _passed_classifier:
                            new_strat = ActiveStrategy(
                                strategy_id=strategy_counter,
                                genes=gene_dict,
                                genes_desc=best_genes.describe(),
                                start_date=d,
                                trades=[],
                                contracts_history=[],
                                train_fitness=best_fr.fitness,
                                train_pnl=best_fr.cum_pnl,
                                train_wr=best_fr.win_rate,
                            )
                            active.append(new_strat)
                            added_today = 1
                            _prob_str = f"P={_clf_prob:.3f}, " if _clf_prob is not None else ""
                            print(f"    [{d}] ADDED strategy {strategy_counter}: "
                                  f"{best_genes.describe()} "
                                  f"({_prob_str}train fit={best_fr.fitness:.3f}, "
                                  f"val pnl=${val_pnl:+.0f}) "
                                  f"[{tm.time()-t1:.0f}s]")
                        else:
                            print(f"    [{d}] REJECTED by classifier: "
                                  f"{best_genes.describe()} "
                                  f"(P={_clf_prob:.3f} < {classifier_threshold:.2f}, "
                                  f"train fit={best_fr.fitness:.3f}, "
                                  f"val pnl=${val_pnl:+.0f}) "
                                  f"[{tm.time()-t1:.0f}s]")
                    else:
                        print(f"    [{d}] Retrained but failed validation "
                              f"(val fit={val_fitness.fitness:.3f}, "
                              f"val pnl=${val_pnl:+.0f})")

        # ── Step 3: Trade all active strategies ──
        for strat in active:
            if d in und_cache and d in chain_cache and d in meta_cache:
                if multi_entry:
                    day_trades = evaluate_day_multi(
                        und_cache[d], chain_cache[d], meta_cache[d],
                        strat.genes, cost, _ctx=ctx_cache.get(d),
                    )
                else:
                    single = evaluate_day(
                        und_cache[d], chain_cache[d], meta_cache[d],
                        strat.genes, cost, _ctx=ctx_cache.get(d),
                    )
                    day_trades = [single] if single is not None else []

                for trade in day_trades:
                    # Compute contracts based on strategy's scaled cum_pnl
                    strat_cum = sum(
                        t.pnl_net * c
                        for t, c in zip(strat.trades, strat.contracts_history)
                    ) if strat.trades else 0.0
                    n_contracts = force_contracts if force_contracts else get_contracts(strat_cum, kill_config)
                    strat.trades.append(trade)
                    strat.contracts_history.append(n_contracts)
                    scaled_pnl = trade.pnl_net * n_contracts
                    day_pnl += scaled_pnl
                    trades_taken += 1

        cum_pnl += day_pnl

        # Portfolio DD tracking
        portfolio_peak = max(portfolio_peak, cum_pnl)
        current_dd = portfolio_peak - cum_pnl
        max_portfolio_dd = max(max_portfolio_dd, current_dd)

        # Per-strategy DD snapshots (using scaled PnL)
        strat_dds = {}
        for strat in active:
            if strat.trades:
                scaled = [t.pnl_net * c for t, c in zip(strat.trades, strat.contracts_history)]
                s_running = np.concatenate(([0.0], np.cumsum(scaled)))
                s_peak = np.maximum.accumulate(s_running)
                strat_dds[strat.strategy_id] = round(float((s_peak - s_running)[-1]), 2)
            else:
                strat_dds[strat.strategy_id] = 0.0

        day_result = DayResult(
            date=str(d),
            n_active=len(active),
            trades_taken=trades_taken,
            day_pnl=round(day_pnl, 2),
            cum_pnl=round(cum_pnl, 2),
            portfolio_dd=round(current_dd, 2),
            max_portfolio_dd=round(max_portfolio_dd, 2),
            strategy_dds=strat_dds,
            strategies_added=added_today,
            strategies_killed=len(killed_today),
            kill_reasons=killed_today,
        )
        all_day_results.append(day_result)

        # Daily progress logging
        print(f"  [{d}] {len(active)} active, "
              f"day=${day_pnl:+,.0f}, cum=${cum_pnl:+,.0f}, "
              f"dd=${current_dd:,.0f}, maxDD=${max_portfolio_dd:,.0f}"
              f"{' | killed: ' + '; '.join(killed_today) if killed_today else ''}")

    # ── Record any strategies still alive at end ──
    # Also compute kill state snapshot for every surviving strategy
    alive_kill_states = []
    for strat in active:
        fwd_trades = strat.trades
        scaled_pnls = [t.pnl_net * c for t, c in zip(fwd_trades, strat.contracts_history)] if fwd_trades else []
        fwd_pnl = sum(scaled_pnls)
        fwd_wr = (sum(1 for t in fwd_trades if t.result == "win")
                  / len(fwd_trades)) if fwd_trades else 0
        fwd_dd = 0.0
        if scaled_pnls:
            running = np.concatenate(([0.0], np.cumsum(scaled_pnls)))
            peak = np.maximum.accumulate(running)
            fwd_dd = float((peak - running).max())
        fwd_days = (valid_days[-1] - strat.start_date).days

        regime = RegimeRecord(
            regime_id=strat.strategy_id,
            train_start="",
            train_end="",
            forward_start=str(strat.start_date),
            forward_end=str(valid_days[-1]),
            death_reason="alive",
            genes_desc=strat.genes_desc,
            genes=strat.genes,
            train_fitness=strat.train_fitness,
            train_pnl=strat.train_pnl,
            train_wr=strat.train_wr,
            train_sharpe=0.0,
            train_pf=0.0,
            train_n_trades=0,
            forward_trades=len(fwd_trades),
            forward_pnl=round(fwd_pnl, 2),
            forward_wr=round(fwd_wr, 4),
            forward_max_dd=round(fwd_dd, 2),
            forward_days=fwd_days,
            forward_avg_pnl=round(fwd_pnl / len(fwd_trades), 2) if fwd_trades else 0,
        )
        all_regime_records.append(regime)

        # Kill state for alive strategies — diagnostic
        snap = kill_state_snapshot(
            strat.trades, kill_config, strat.start_date, valid_days[-1],
            contracts_history=strat.contracts_history,
        )
        snap["strategy_id"] = strat.strategy_id
        alive_kill_states.append(snap)

    # ── Summary ──
    total_trades = sum(dr.trades_taken for dr in all_day_results)
    trading_days = sum(1 for dr in all_day_results if dr.trades_taken > 0)
    total_regimes = len(all_regime_records)
    positive_regimes = sum(1 for r in all_regime_records if r.forward_pnl > 0)
    peak_concurrent = max((dr.n_active for dr in all_day_results), default=0)
    total_kills = sum(dr.strategies_killed for dr in all_day_results)

    print(f"\n{'='*60}")
    print(f"Live simulation complete")
    print(f"  Period: {all_day_results[0].date} → {all_day_results[-1].date} "
          f"({len(all_day_results)} days)")
    print(f"  Total trades: {total_trades} across {trading_days} trading days")
    print(f"  Total PnL: ${cum_pnl:+,.0f}")
    print(f"  Max portfolio DD: ${max_portfolio_dd:,.0f}")
    print(f"  Peak concurrent strategies: {peak_concurrent}")
    print(f"  Regimes: {total_regimes} total, {positive_regimes} positive "
          f"({positive_regimes/total_regimes:.0%})" if total_regimes else "")
    print(f"  Total kills: {total_kills}")
    print(f"  Still alive: {len(active)}")
    print(f"  Avg daily PnL: ${cum_pnl / len(all_day_results):+.0f}")
    if trading_days > 0:
        print(f"  Avg PnL per trading day: ${cum_pnl / trading_days:+.0f}")

    # ── Kill condition diagnostic ──
    zero_trade_strats = sum(1 for s in alive_kill_states if s['n_trades'] == 0)
    if zero_trade_strats > 0:
        print(f"\n  *** WARNING: {zero_trade_strats}/{len(alive_kill_states)} surviving "
              f"strategies have ZERO trades ***")
        print(f"  *** This means evaluate_day returned None for every day — "
              f"check gene compatibility with chain data ***", flush=True)

    if len(active) > 0 and total_kills == 0 and total_regimes > 5:
        print(f"\n  *** WARNING: ZERO kills fired across {len(all_day_results)} days "
              f"with {total_regimes} deployed strategies ***")
        print(f"  *** This is almost certainly a bug. Kill state for each surviving strategy: ***")
        for snap in alive_kill_states:
            print(f"    strat={snap['strategy_id']} | "
                  f"{snap['n_trades']}t {snap['days_alive']}d | "
                  f"pnl=${snap['cum_pnl']:+.0f} dd=${snap['max_dd']:.0f}/"
                  f"${snap['eff_dd_limit']:.0f} | "
                  f"WR={snap['wr_all']:.0%} roll={snap['wr_rolling_20']:.0%} | "
                  f"avg=${snap['avg_pnl_per_trade']:.1f} | "
                  f"nearest: {snap['nearest_kill']}", flush=True)

    return {
        "status": "ok",
        "walkforward_version": _WALKFORWARD_VERSION,
        "summary": {
            "period_start": all_day_results[0].date if all_day_results else "",
            "period_end": all_day_results[-1].date if all_day_results else "",
            "total_days": len(all_day_results),
            "trading_days": trading_days,
            "total_trades": total_trades,
            "cum_pnl": round(cum_pnl, 2),
            "max_portfolio_dd": round(max_portfolio_dd, 2),
            "total_regimes": total_regimes,
            "positive_regimes": positive_regimes,
            "strategies_generated": strategy_counter,
            "peak_concurrent": peak_concurrent,
            "avg_daily_pnl": round(cum_pnl / len(all_day_results), 2) if all_day_results else 0,
            "total_kills": total_kills,
            "still_alive": len(active),
        },
        "daily_equity": [asdict(dr) for dr in all_day_results],
        "regimes": [asdict(r) for r in all_regime_records],
        "alive_kill_states": alive_kill_states,
        "kill_config": asdict(kill_config),
    }


def save_regimes(regimes: list[RegimeRecord], path: Path) -> None:
    """Save regime records to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in regimes]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(regimes)} regimes to {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options walk-forward regime simulator")
    parser.add_argument("--data-dir", type=str, default="/data")
    parser.add_argument("--chain-subdir", default="options_5dte")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--max-dd", type=float, default=3000.0)
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="results/options/regime_db.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    kill_config = KillConfig(max_dd_dollars=args.max_dd)
    start_from = date.fromisoformat(args.start) if args.start else None

    regimes = run_walkforward(
        data_dir=data_dir,
        chain_subdir=args.chain_subdir,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        n_candidates=args.candidates,
        kill_config=kill_config,
        start_from=start_from,
    )

    save_regimes(regimes, Path(args.output))
