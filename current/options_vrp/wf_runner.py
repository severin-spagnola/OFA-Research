"""
Walk-Forward Regime Runner (OFA-VRP)
======================================
Walk-forward pipeline for the credit spread VRP strategy.
Three-phase evaluation:
  - Train (45d): strategy search & fitness ranking
  - Val (15d): filter noise strategies (Sharpe, trade count gates)
  - Forward (kill-gated): day-by-day simulation with kill conditions until
    death or data exhaustion → forward_profitable label

The forward period is NOT fixed-length. It runs until kill conditions fire.

Output: results/options_vrp/regime_db.jsonl — one regime record per line.

Usage:
    python current/options_vrp/wf_runner.py --data-dir /path/to/options_5dte
    python current/options_vrp/wf_runner.py --train-days 45 --val-days 15 --candidates 2000
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import random
import sys
import time as tm
import dataclasses
from dataclasses import asdict, dataclass, fields as dc_fields
from datetime import date
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from options_backtest import evaluate_day, FitnessResult
from options_genes import VRPGenes, SEARCH_SPACE, all_genes, random_genes

# Robustness metrics from the MES pipeline — wired in for classifier features
_OVERFIT_DIR = _SCRIPT_DIR.parent / "overfit_finder"
if str(_OVERFIT_DIR) not in sys.path:
    sys.path.insert(0, str(_OVERFIT_DIR))
try:
    from robustness_metrics import compute_robustness_metrics
    _HAS_ROBUSTNESS = True
except ImportError:
    _HAS_ROBUSTNESS = False


# ─── Per-day record (used by kill conditions) ─────────────────────────────────

@dataclass
class DailyRecord:
    """Minimal per-day record compatible with kill condition logic."""
    pnl_net: float
    result: str     # "win" | "loss"
    trade_date: str  # YYYY-MM-DD


# ─── Kill Conditions ──────────────────────────────────────────────────────────
# These MUST stay identical to the livesim kill logic. If you change one,
# change the other. Divergence means regime_db labels won't match live behavior.

@dataclass
class KillConfig:
    """Tunable kill condition thresholds."""
    max_dd_dollars: float = 500.0
    max_single_trade_loss: float = 300.0
    scale_contracts: bool = True
    scale_every: float = 750.0
    emergency_wr_floor: float = 0.20
    emergency_min_trades: int = 5
    early_wr_floor: float = 0.35
    early_min_trades: int = 10
    wr_lookback: int = 20
    wr_floor: float = 0.40
    max_consec_loss_days: int = 7
    flat_regime_pnl: float = 150.0
    flat_regime_trades: int = 15
    avg_pnl_floor: float = 0.5
    negative_trajectory_pnl: float = -500.0
    negative_trajectory_trades: int = 15
    max_regime_days: int = 365
    live_score: bool = True
    live_score_min_trades: int = 10
    skip_all_kills: bool = False


def _live_score_multiplier(
    cum_pnl: float,
    max_dd: float,
    n_trades: int,
    days_alive: int,
    min_trades: int = 10,
) -> float:
    """Compute a multiplier (1.0–2.0) that rewards proven live performance."""
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
    return 1.0 + score


def _get_contracts(cum_pnl: float, config: KillConfig) -> int:
    """Compute number of contracts based on cumulative PnL."""
    if not config.scale_contracts or cum_pnl <= 0:
        return 1
    return 1 + int(cum_pnl / config.scale_every)


def check_kill(
    trades: list[DailyRecord],
    config: KillConfig,
    start_date: date,
    current_date: date,
) -> tuple[bool, str]:
    """Check if the regime should be killed.

    Returns (should_kill, reason_string).
    """
    if config.skip_all_kills:
        return False, ""

    n = len(trades)
    if n == 0:
        days_elapsed = (current_date - start_date).days
        if days_elapsed >= config.max_regime_days:
            return True, f"time_limit ({days_elapsed}d, 0 trades)"
        return False, ""

    days_alive = (current_date - start_date).days
    pnls = [t.pnl_net for t in trades]
    cum_pnl = sum(pnls)

    # Drawdown from peak
    running_pnl = np.concatenate(([0.0], np.cumsum(pnls)))
    peak = np.maximum.accumulate(running_pnl)
    dd = peak - running_pnl
    max_dd = float(dd.max())

    # Live performance score
    if config.live_score:
        mult = _live_score_multiplier(
            cum_pnl, max_dd, n, days_alive, config.live_score_min_trades,
        )
    else:
        mult = 1.0

    cur_contracts = _get_contracts(cum_pnl, config)
    eff_dd_limit = config.max_dd_dollars * mult * cur_contracts
    eff_stl_limit = config.max_single_trade_loss * mult

    # Per-trade loss cap
    for t in trades:
        if t.pnl_net < -eff_stl_limit:
            return True, (
                f"single_trade_loss ${t.pnl_net:.0f}/ct"
                f" < -${eff_stl_limit:.0f} (base ${config.max_single_trade_loss:.0f} x{mult:.2f})"
            )

    # Drawdown check
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
    daily_pnl: dict[str, float] = {}
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

    # Flat regime — DISABLED for VRP (credit spreads thrive in calm/low-vol markets)
    # if n >= config.flat_regime_trades and abs(cum_pnl) < config.flat_regime_pnl:
    #     return True, f"flat_regime ${cum_pnl:.0f} after {n} trades"

    # Avg PnL/trade floor — DISABLED: absolute dollar threshold killed profitable records
    # (e.g. $191.6 cumPnL at $2.99/trade avg was killed just under the $3 threshold)
    # if n >= config.flat_regime_trades:
    #     avg_pnl = cum_pnl / n
    #     if avg_pnl < config.avg_pnl_floor:
    #         return True, f"low_avg_pnl ${avg_pnl:.1f}/trade < ${config.avg_pnl_floor:.0f} after {n} trades"

    # Time limit
    days_elapsed = (current_date - start_date).days
    if days_elapsed >= config.max_regime_days:
        return True, f"time_limit ({days_elapsed}d, {n} trades, ${cum_pnl:.0f})"

    return False, ""


def kill_state_snapshot(
    trades: list[DailyRecord],
    config: KillConfig,
    start_date: date,
    current_date: date,
) -> dict:
    """Return a diagnostic snapshot of kill condition metrics."""
    n = len(trades)
    days_alive = (current_date - start_date).days

    if n == 0:
        return {"n_trades": 0, "days_alive": days_alive, "cum_pnl": 0.0,
                "max_dd": 0.0, "wr_all": 0.0, "nearest_kill": "none (0 trades)"}

    pnls = [t.pnl_net for t in trades]
    cum_pnl = sum(pnls)
    running_pnl = np.concatenate(([0.0], np.cumsum(pnls)))
    peak = np.maximum.accumulate(running_pnl)
    dd = peak - running_pnl
    max_dd = float(dd.max())

    if config.live_score:
        mult = _live_score_multiplier(
            cum_pnl, max_dd, n, days_alive, config.live_score_min_trades,
        )
    else:
        mult = 1.0

    cur_contracts = _get_contracts(cum_pnl, config)
    eff_dd_limit = config.max_dd_dollars * mult * cur_contracts
    wr_all = sum(1 for t in trades if t.result == "win") / n
    avg_pnl = cum_pnl / n

    if n >= config.wr_lookback:
        recent = trades[-config.wr_lookback:]
        wr_rolling = sum(1 for t in recent if t.result == "win") / len(recent)
    else:
        wr_rolling = wr_all

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
    }


# ─── Walk-forward window generation ──────────────────────────────────────────

@dataclass
class WFWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    train_days: list[date]
    val_days: list[date]
    fwd_days: list[date]  # all remaining days after val — kill-gated, not fixed


def generate_windows(
    all_days: list[date],
    train_size: int = 45,
    val_size: int = 15,
    step_size: int = 10,
) -> list[WFWindow]:
    """Generate walk-forward windows: train / val / forward (all remaining).

    A window is only generated if at least 5 trading days exist after val.
    """
    windows = []
    min_needed = train_size + val_size
    min_fwd_days = 5
    i = 0
    window_id = 0

    while i + min_needed <= len(all_days):
        train_days = all_days[i : i + train_size]
        val_days = all_days[i + train_size : i + min_needed]
        fwd_days = all_days[i + min_needed :]

        if len(fwd_days) < min_fwd_days:
            break

        windows.append(WFWindow(
            window_id=window_id,
            train_start=train_days[0],
            train_end=train_days[-1],
            val_start=val_days[0],
            val_end=val_days[-1],
            train_days=train_days,
            val_days=val_days,
            fwd_days=fwd_days,
        ))
        window_id += 1
        i += step_size

    return windows


# ─── Data availability ────────────────────────────────────────────────────────

def get_available_days(data_dir: str | Path) -> list[date]:
    """Return sorted list of dates that have both chain + meta files."""
    base = Path(data_dir)
    days = []
    for p in base.glob("chain_*_meta.json"):
        stem = p.stem  # e.g. chain_2024-03-13_meta
        day_str = stem[len("chain_"):-len("_meta")]
        chain_p = base / f"chain_{day_str}.parquet"
        if chain_p.exists():
            try:
                days.append(date.fromisoformat(day_str))
            except ValueError:
                pass
    return sorted(days)


# ─── Aggregate fitness ────────────────────────────────────────────────────────

def compute_period_fitness(records: list[DailyRecord], min_trades: int = 10) -> dict:
    """Aggregate per-day DailyRecords into period-level fitness metrics."""
    pnls = [r.pnl_net for r in records]
    n = len(pnls)
    if n == 0:
        return {"fitness": -99.0, "cum_pnl": 0.0, "win_rate": 0.0, "n_trades": 0, "sharpe": 0.0}
    wins = sum(1 for r in records if r.result == "win")
    cum_pnl = sum(pnls)
    win_rate = wins / n
    arr = np.array(pnls, dtype=float)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(arr.mean() / std) * np.sqrt(252) if std > 0 else 0.0
    fitness = sharpe if n >= min_trades else -99.0
    return {
        "fitness": fitness,
        "cum_pnl": round(cum_pnl, 2),
        "win_rate": round(win_rate, 4),
        "n_trades": n,
        "sharpe": round(sharpe, 4),
    }


def _describe_genes(genes: VRPGenes) -> str:
    return (
        f"{genes.spread_type} ofs={genes.short_strike_offset} "
        f"ww={genes.wing_width} tp={genes.tp_pct:.0%} sl={genes.sl_multiple}x"
    )


# ─── GA operators ─────────────────────────────────────────────────────────────

def _crossover(parent1: VRPGenes, parent2: VRPGenes) -> VRPGenes:
    """Uniform crossover: independently pick each field from one of the two parents."""
    vals = {
        f.name: (getattr(parent1, f.name) if random.random() < 0.5 else getattr(parent2, f.name))
        for f in dc_fields(VRPGenes)
    }
    return VRPGenes(**vals)


def _mutate(genes: VRPGenes, mutation_rate: float = 0.15) -> VRPGenes:
    """Per-field mutation: with probability mutation_rate re-sample from SEARCH_SPACE."""
    vals = {
        f.name: (random.choice(SEARCH_SPACE[f.name]) if random.random() < mutation_rate
                 else getattr(genes, f.name))
        for f in dc_fields(VRPGenes)
    }
    return VRPGenes(**vals)


# ─── Val pass criteria ───────────────────────────────────────────────────────
VAL_MIN_TRADES = 2  # 15-day val window at 5-DTE yields at most 3 trades; 5 was structurally unreachable
VAL_MIN_SHARPE = 0.0


# ─── Single window evaluation ─────────────────────────────────────────────────

def _evaluate_population(
    population: list[VRPGenes],
    train_days: list[date],
    data_dir_str: str,
    min_train_trades: int,
) -> list[tuple]:
    """Evaluate every candidate in *population* over *train_days*.

    Returns a list of (genes, metrics_dict, records_list) sorted by fitness
    descending.
    """
    results = []
    for genes in population:
        records: list[DailyRecord] = []
        for d in train_days:
            try:
                fr = evaluate_day(genes, d.isoformat(), data_dir_str)
                if fr.num_trades > 0:
                    result = "win" if fr.win_rate > 0 else "loss"
                    records.append(DailyRecord(fr.total_pnl, result, d.isoformat()))
            except Exception:
                pass
        metrics = compute_period_fitness(records, min_trades=min_train_trades)
        results.append((genes, metrics, records))
    results.sort(key=lambda x: -x[1]["fitness"])
    return results


def evaluate_window(
    window: WFWindow,
    data_dir: str | Path,
    n_candidates: int = 100,
    n_generations: int = 1,
    top_n: int = 20,
    min_train_trades: int = 5,
    seed: int = 42,
    kill_config: KillConfig | None = None,
    verbose: bool = False,
    spy_context: dict = None,
) -> list[dict]:
    """Evaluate a single walk-forward window (train → val → kill-gated forward).

    Train phase:
      - If n_candidates <= 0 (exhaustive mode): evaluate ALL gene combinations
        from all_genes() exactly once — no generations, no sampling.
      - If n_candidates > 0 (GA mode): randomly sample n_candidates candidates,
        then run n_generations of truncation selection + crossover + mutation.

    After train phase, take top top_n by fitness (fitness > 0) for val.

    Returns:
        list of regime record dicts (one per val-passing strategy)
    """
    if kill_config is None:
        kill_config = KillConfig()

    data_dir_str = str(data_dir)

    # ── Phase 1: Train search ────────────────────────────────────────────────
    random.seed(seed)

    if n_candidates <= 0:
        # Exhaustive mode: evaluate all gene combinations exactly once
        population: list[VRPGenes] = all_genes()
        if verbose:
            print(
                f"  [W{window.window_id}] exhaustive mode: {len(population)} candidates",
                flush=True,
            )
        train_results = _evaluate_population(
            population, window.train_days, data_dir_str, min_train_trades
        )
    else:
        # GA mode: random sampling + generations
        n_generations = max(1, n_generations)
        population = [random_genes() for _ in range(n_candidates)]

        pop_results: list[tuple] = []
        for gen in range(n_generations):
            pop_results = _evaluate_population(
                population, window.train_days, data_dir_str, min_train_trades
            )

            if gen == n_generations - 1:
                break  # last generation — no need to breed

            # Truncation selection: keep top 50%
            n_survivors = max(2, len(pop_results) // 2)
            survivor_genes = [g for g, _, _ in pop_results[:n_survivors]]

            # Breed next generation (elitism: preserve best unchanged)
            new_population: list[VRPGenes] = [survivor_genes[0]]
            while len(new_population) < n_candidates:
                p1, p2 = random.choices(survivor_genes, k=2)
                child = _crossover(p1, p2)
                child = _mutate(child)
                new_population.append(child)
            population = new_population

            if verbose:
                best_fitness = pop_results[0][1]["fitness"] if pop_results else -99.0
                print(
                    f"  [W{window.window_id}] gen {gen + 1}/{n_generations} "
                    f"best_fitness={best_fitness:.4f}",
                    flush=True,
                )

        train_results = pop_results  # already sorted by fitness desc

    # Sort by fitness, take top N with positive fitness
    viable = [(g, m, r) for g, m, r in train_results if m["fitness"] > 0]
    viable.sort(key=lambda x: -x[1]["fitness"])
    top_viable = viable[:top_n]

    # Diagnostic: always log training-phase outcome so we can diagnose 0 val-tested runs
    best_fitness = train_results[0][1]["fitness"] if train_results else -99.0
    best_n_trades = train_results[0][1]["n_trades"] if train_results else 0
    print(
        f"  [W{window.window_id}] train: {len(train_results)} candidates, "
        f"best_fitness={best_fitness:.3f} best_n_trades={best_n_trades} "
        f"viable={len(viable)} min_train_trades={min_train_trades}",
        flush=True,
    )

    if not top_viable:
        return []

    if verbose:
        print(f"  [W{window.window_id}] {len(viable)} viable → top {len(top_viable)} to val", flush=True)

    # Phase 2: Val evaluation (filter gate only)
    regime_records = []
    for genes, train_metrics, train_records in top_viable:
        val_records: list[DailyRecord] = []
        for d in window.val_days:
            try:
                fr = evaluate_day(genes, d.isoformat(), data_dir_str)
                if fr.num_trades > 0:
                    result = "win" if fr.win_rate > 0 else "loss"
                    val_records.append(DailyRecord(fr.total_pnl, result, d.isoformat()))
            except Exception as e:
                logger.error(f'Val exception: {e}', exc_info=True)

        val_metrics = compute_period_fitness(val_records, min_trades=1)

        if val_metrics["n_trades"] < VAL_MIN_TRADES:
            continue
        if val_metrics["sharpe"] < VAL_MIN_SHARPE:
            continue

        # Robustness metrics on train pnls
        robustness_dict: dict = {}
        if _HAS_ROBUSTNESS:
            train_pnls = [r.pnl_net for r in train_records]
            robustness = compute_robustness_metrics(train_pnls, window_days=len(window.train_days))
            robustness_dict = robustness.to_dict() if robustness is not None else {}

        # Phase 3: Kill-gated forward evaluation
        fwd_records: list[DailyRecord] = []
        fwd_exit_reason = "data_end"
        fwd_last_day = window.fwd_days[0]

        for d in window.fwd_days:
            try:
                fr = evaluate_day(genes, d.isoformat(), data_dir_str)
                if fr.num_trades > 0:
                    result = "win" if fr.win_rate > 0 else "loss"
                    fwd_records.append(DailyRecord(fr.total_pnl, result, d.isoformat()))
            except Exception:
                pass

            fwd_last_day = d

            killed, reason = check_kill(fwd_records, kill_config, window.fwd_days[0], d)
            if killed:
                fwd_exit_reason = reason
                break

        fwd_cum_pnl = sum(r.pnl_net for r in fwd_records)
        fwd_n_trades = len(fwd_records)
        fwd_days_elapsed = (fwd_last_day - window.fwd_days[0]).days
        fwd_wr = (sum(1 for r in fwd_records if r.result == "win") / fwd_n_trades
                  if fwd_n_trades > 0 else 0.0)
        fwd_dd = 0.0
        if fwd_records:
            running = np.concatenate(([0.0], np.cumsum([r.pnl_net for r in fwd_records])))
            peak_arr = np.maximum.accumulate(running)
            fwd_dd = float((peak_arr - running).max())

        forward_profitable = fwd_cum_pnl > 0

        gene_dict = asdict(genes)

        raw_trades = [
            {
                "trade_num": i + 1,
                "trade_date": r.trade_date,
                "pnl_net": round(r.pnl_net, 2),
                "result": r.result,
            }
            for i, r in enumerate(fwd_records)
        ]

        spy_bull_pct = (
            float(np.mean([spy_context[d.isoformat()] for d in window.train_days if d.isoformat() in spy_context]))
            if spy_context else 0.5
        )

        record = {
            "window_id": window.window_id,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "val_start": window.val_start.isoformat(),
            "val_end": window.val_end.isoformat(),
            "n_train_days": len(window.train_days),
            "n_val_days": len(window.val_days),
            "genes": gene_dict,
            "genes_desc": _describe_genes(genes),
            "spread_type": genes.spread_type,
            "train_fitness": train_metrics,
            "val_fitness": val_metrics,
            "forward_profitable": forward_profitable,
            "forward_source": "kill_gated",
            "fwd_cum_pnl": round(fwd_cum_pnl, 2),
            "fwd_n_trades": fwd_n_trades,
            "fwd_win_rate": round(fwd_wr, 4),
            "fwd_sharpe": None,
            "fwd_max_dd": round(fwd_dd, 2),
            "forward_days": fwd_days_elapsed,
            "forward_exit_reason": fwd_exit_reason,
            "val_cum_pnl": round(val_metrics["cum_pnl"], 2),
            "val_n_trades": val_metrics["n_trades"],
            "val_win_rate": val_metrics["win_rate"],
            "val_sharpe": min(val_metrics.get("sharpe", 0), 3.0),
            "robustness": robustness_dict,
            "raw_trades": raw_trades,
            "spy_bull_pct": round(spy_bull_pct, 4),
        }
        regime_records.append(record)

    return regime_records


# ─── Full walk-forward run ─────────────────────────────────────────────────────

def _load_existing_keys(output_path: Path) -> set[int]:
    """Load window_id values from existing regime_db.jsonl for dedup."""
    keys: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        keys.add(r.get("window_id", -1))
                    except json.JSONDecodeError:
                        continue
    return keys


def run_walkforward(
    data_dir: str | Path,
    train_size: int = 45,
    val_size: int = 15,
    step_size: int = 10,
    n_candidates: int = 100,
    n_generations: int = 10,
    top_n: int = 20,
    min_train_trades: int = 5,
    seed: int = 42,
    kill_config: KillConfig | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Run the full walk-forward across all available data."""
    if output_path is None:
        output_path = Path("results/options_vrp/regime_db.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if kill_config is None:
        kill_config = KillConfig()

    if overwrite and output_path.exists() and not dry_run:
        print(f"[wf] --overwrite: clearing {output_path}", flush=True)
        output_path.unlink()
        existing_window_ids: set[int] = set()
    else:
        existing_window_ids = _load_existing_keys(output_path)
        if existing_window_ids:
            print(f"[wf] Found {len(existing_window_ids)} existing window(s) — "
                  f"will skip duplicates", flush=True)

    data_dir = Path(data_dir)
    print("[wf] Loading available days...", flush=True)
    all_days = get_available_days(data_dir)
    if not all_days:
        print("[wf] No data found.", flush=True)
        return []
    print(f"[wf] {len(all_days)} trading days: {all_days[0]} to {all_days[-1]}", flush=True)

    windows = generate_windows(all_days, train_size, val_size, step_size)
    print(f"[wf] Generated {len(windows)} walk-forward windows "
          f"(train={train_size}, val={val_size}, step={step_size}, "
          f"forward=kill-gated)", flush=True)
    if not windows:
        print("[wf] No windows possible with this data/config", flush=True)
        return []

    w0, wN = windows[0], windows[-1]
    print(f"[wf] First window: train {w0.train_start}→{w0.train_end}, "
          f"val {w0.val_start}→{w0.val_end}, fwd {len(w0.fwd_days)}d avail", flush=True)
    print(f"[wf] Last window:  train {wN.train_start}→{wN.train_end}, "
          f"val {wN.val_start}→{wN.val_end}, fwd {len(wN.fwd_days)}d avail", flush=True)

    all_records = []
    t_start = tm.time()

    for w in windows:
        if w.window_id in existing_window_ids:
            print(f"[WARN] Skipping duplicate: window_id={w.window_id}", flush=True)
            continue

        t_w = tm.time()
        w_seed = seed + w.window_id * 137

        records = evaluate_window(
            window=w,
            data_dir=data_dir,
            n_candidates=n_candidates,
            n_generations=n_generations,
            top_n=top_n,
            min_train_trades=min_train_trades,
            seed=w_seed,
            kill_config=kill_config,
            verbose=verbose,
        )

        if dry_run:
            for r in records:
                print(f"[DRY-RUN] window={r['window_id']} "
                      f"{r['genes_desc'][:50]} fwd=${r['fwd_cum_pnl']:.0f} "
                      f"exit={r['forward_exit_reason'][:30]}", flush=True)
        else:
            with open(output_path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        all_records.extend(records)

        n_viable = len(records)
        n_profitable = sum(1 for r in records if r["forward_profitable"])
        elapsed = tm.time() - t_start
        w_elapsed = tm.time() - t_w
        remaining = (elapsed / (w.window_id + 1)) * (len(windows) - w.window_id - 1)

        print(
            f"[wf] Window {w.window_id + 1}/{len(windows)} | "
            f"train {w.train_start}→{w.train_end} | "
            f"{n_viable} val-tested, {n_profitable} fwd-profitable | "
            f"{w_elapsed:.0f}s | ~{remaining:.0f}s remaining",
            flush=True,
        )

    total = len(all_records)
    profitable = sum(1 for r in all_records if r["forward_profitable"])
    rate = profitable / total * 100 if total else 0
    print(f"\n[wf] DONE: {total} regime records, {profitable} forward-profitable "
          f"({rate:.1f}%)", flush=True)
    if all_records:
        exit_reasons: dict[str, int] = {}
        for r in all_records:
            key = r["forward_exit_reason"].split(" ")[0] if r["forward_exit_reason"] != "data_end" else "data_end"
            exit_reasons[key] = exit_reasons.get(key, 0) + 1
        print(f"[wf] Exit reasons: {dict(sorted(exit_reasons.items(), key=lambda x: -x[1]))}", flush=True)
    print(f"[wf] Saved to {output_path}", flush=True)

    return all_records


def run_walk_forward(
    data_dir: str | None = None,
    n_generations: int = 10,
    population_size: int = 100,
    train_size: int = 45,
    val_size: int = 15,
    step_size: int = 10,
    top_n: int = 20,
    min_train_trades: int = 5,
    seed: int = 42,
    kill_config: KillConfig | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Convenience wrapper with population_size / n_generations naming.

    population_size = initial population of random candidates per window.
    n_generations   = number of GA generations (evaluate → select → breed).
    """
    if data_dir is None:
        data_dir = os.environ.get("OPTIONS_DATA_DIR", "/data")
    return run_walkforward(
        data_dir=data_dir,
        train_size=train_size,
        val_size=val_size,
        step_size=step_size,
        n_candidates=population_size,
        n_generations=n_generations,
        top_n=top_n,
        min_train_trades=min_train_trades,
        seed=seed,
        kill_config=kill_config,
        output_path=output_path,
        overwrite=overwrite,
        dry_run=dry_run,
        verbose=verbose,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward options regime runner (VRP)")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("OPTIONS_DATA_DIR", "/data"))
    parser.add_argument("--train-days", type=int, default=45)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: random)")
    parser.add_argument("--output", type=str, default="results/options_vrp/regime_db.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-all-kills", action="store_true",
                        help="Disable all kill conditions (forward runs to data_end)")
    args = parser.parse_args()

    # Run-guard: prevent concurrent wf_runner processes from corrupting regime_db.jsonl
    _lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wf_runner.lock')
    _lock_fh = open(_lock_path, 'w')
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('[ERROR] Another wf_runner is already running. Exiting to prevent corruption.')
        raise SystemExit(1)

    if args.seed is None:
        args.seed = random.randint(10000, 99999)
        print(f"[wf_runner] Auto-generated seed: {args.seed}")

    kill_cfg = KillConfig(skip_all_kills=args.skip_all_kills) if args.skip_all_kills else None

    run_walkforward(
        data_dir=Path(args.data_dir),
        train_size=args.train_days,
        val_size=args.val_days,
        step_size=args.step,
        n_candidates=args.candidates,
        top_n=args.top_n,
        min_train_trades=args.min_trades,
        seed=args.seed,
        kill_config=kill_cfg,
        output_path=Path(args.output),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
