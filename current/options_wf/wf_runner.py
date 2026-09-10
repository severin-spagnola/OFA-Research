"""
Walk-Forward Regime Runner (kill-gated forward)
=================================================
OFA-style walk-forward with 3-phase evaluation:
  - Train (45d): strategy search & fitness ranking
  - Val (15d): filter noise strategies (Sharpe, trade count gates)
  - Forward (kill-gated): day-by-day simulation with kill conditions until
    death or data exhaustion → forward_profitable label

The forward period is NOT fixed-length. It runs until kill conditions fire,
mirroring exactly what live deployment does. This produces honest labels.

Output: results/options_wf/regime_db.jsonl — one regime record per line.

Usage:
    python current/options_wf/wf_runner.py --train-days 45 --val-days 15 --candidates 2000

For parallel execution on RunPod, use infra/wf_orchestrator.py instead.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time as tm
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

# Ensure options_wf is on the path
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from options_backtest import (
    evaluate_day,
    compute_fitness,
    FitnessResult,
    OptionsTrade,
    load_underlying,
    load_chain,
    load_chain_meta,
    get_available_days,
    CostModel,
    precompute_day_context,
    precompute_chain,
)
from options_genes import sample_candidates, describe_genes, OptionsGenes

# Robustness metrics from the MES pipeline — wired in for classifier features
_OVERFIT_DIR = _SCRIPT_DIR.parent / "overfit_finder"
if str(_OVERFIT_DIR) not in sys.path:
    sys.path.insert(0, str(_OVERFIT_DIR))
from robustness_metrics import compute_robustness_metrics


# ─── Kill Conditions (ported from options_research/options_walkforward.py) ────
# These MUST stay identical to the livesim kill logic. If you change one,
# change the other. Divergence means regime_db labels won't match live behavior.

@dataclass
class KillConfig:
    """Tunable kill condition thresholds.

    Calibrated from 60-regime walk-forward on 2024-03 → 2026-03 data.
    """
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
    avg_pnl_floor: float = 3.0
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
    trades: list[OptionsTrade],
    config: KillConfig,
    start_date: date,
    current_date: date,
) -> tuple[bool, str]:
    """Check if the regime should be killed.

    Identical to options_walkforward.check_kill (minus contracts_history and
    check_stl_from which are livesim-only features for multi-entry).

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

    # Flat regime
    if n >= config.flat_regime_trades and abs(cum_pnl) < config.flat_regime_pnl:
        return True, f"flat_regime ${cum_pnl:.0f} after {n} trades"

    # Avg PnL/trade floor
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
) -> dict:
    """Return a diagnostic snapshot of kill condition metrics (wf_runner version).

    Does NOT trigger a kill. Used for logging/debugging.
    """
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
    """A single walk-forward window.

    Forward days are ALL available trading days after val_end. The forward
    period is not fixed-length — it runs until kill conditions fire.
    """
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

    The forward period is every available day after val ends. Each strategy
    walks forward until kill conditions fire, so the forward period is
    variable-length per strategy within a window.

    A window is only generated if at least 5 trading days exist after val
    for forward testing. Windows that don't meet this are skipped.

    Args:
        all_days: sorted list of available trading dates
        train_size: number of trading days in train window
        val_size: number of trading days in val window
        step_size: number of trading days to step forward between windows
    """
    windows = []
    min_needed = train_size + val_size
    min_fwd_days = 5  # minimum forward days to be usable
    i = 0
    window_id = 0

    while i + min_needed <= len(all_days):
        train_days = all_days[i : i + train_size]
        val_days = all_days[i + train_size : i + min_needed]
        fwd_days = all_days[i + min_needed :]  # everything after val

        if len(fwd_days) < min_fwd_days:
            # Not enough forward data — skip this and all subsequent windows
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


# ─── Single window evaluation ────────────────────────────────────────────────

def _fr_to_dict(fr: FitnessResult) -> dict:
    return {
        "fitness": fr.fitness,
        "sharpe": fr.sharpe,
        "profit_factor": fr.profit_factor,
        "win_rate": fr.win_rate,
        "n_trades": fr.n_trades,
        "avg_r": fr.avg_r,
        "cum_r": fr.cum_r,
        "max_dd_r": fr.max_dd_r,
        "avg_pnl": fr.avg_pnl,
        "cum_pnl": fr.cum_pnl,
        "avg_win": fr.avg_win,
        "avg_loss": fr.avg_loss,
        "max_drawdown": fr.max_drawdown,
    }


# Number of extra days to load before the first train day, for prev_close /
# prev_range context.
N_CONTEXT_DAYS = 5

# ─── Val pass criteria ───────────────────────────────────────────────────────
VAL_MIN_TRADES = 10
VAL_MIN_SHARPE = 0.5


def evaluate_window(
    window: WFWindow,
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    n_candidates: int = 2000,
    top_n: int = 20,
    min_trades: int = 30,
    seed: int = 42,
    cost: CostModel | None = None,
    kill_config: KillConfig | None = None,
    preloaded_und: dict | None = None,
    preloaded_chains: dict | None = None,
    preloaded_metas: dict | None = None,
    context_days: list[date] | None = None,
) -> list[dict]:
    """Evaluate a single walk-forward window (train → val → kill-gated forward).

    1. Sample n_candidates gene combinations
    2. Evaluate all on train_days → compute robustness metrics
    3. Take top_n by train fitness
    4. Filter on val_days (Sharpe, trade count gates)
    5. Walk survivors through fwd_days day-by-day with kill conditions

    Returns:
        list of regime record dicts (one per val-passing strategy)
    """
    if cost is None:
        cost = CostModel()
    if kill_config is None:
        kill_config = KillConfig()

    all_window_days = window.train_days + window.val_days + window.fwd_days

    # Determine context days for prev_close enrichment
    if context_days is None:
        context_days = []
    context_day_set = set(context_days)

    assert not context_day_set & set(window.train_days), \
        "Context days must not overlap with train days"
    assert not context_day_set & set(window.val_days), \
        "Context days must not overlap with val days"

    # Days to load: context + train + val + forward
    all_load_days = sorted(context_days) + all_window_days

    # Load data if not preloaded
    if preloaded_und is None:
        preloaded_und = {}
        if not context_days and data_dir:
            available = get_available_days(data_dir, chain_subdir)
            if available:
                first_train = window.train_days[0]
                prior_days = [d for d in available if d < first_train]
                context_days = prior_days[-N_CONTEXT_DAYS:]
                context_day_set = set(context_days)
                all_load_days = sorted(context_days) + all_window_days

        for d in all_load_days:
            und = load_underlying(data_dir, d)
            if not und.empty:
                preloaded_und[d] = und

    if preloaded_chains is None:
        preloaded_chains = {}
        for d in all_load_days:
            chain = load_chain(data_dir, d, chain_subdir)
            if not chain.empty:
                preloaded_chains[d] = chain

    if preloaded_metas is None:
        preloaded_metas = {}
        for d in all_load_days:
            meta = load_chain_meta(data_dir, d, chain_subdir)
            if meta:
                preloaded_metas[d] = meta

    # Add prev_close to metas
    sorted_und_days = sorted(preloaded_und.keys())
    for i, d in enumerate(sorted_und_days):
        if d in preloaded_metas and i > 0:
            prev_d = sorted_und_days[i - 1]
            if prev_d in preloaded_und:
                prev_und = preloaded_und[prev_d]
                preloaded_metas[d]["prev_close"] = float(prev_und.iloc[-1]["c"])
                preloaded_metas[d]["prev_high"] = float(prev_und["h"].max())
                preloaded_metas[d]["prev_low"] = float(prev_und["l"].min())

    # Pre-compute chain minute columns (skip context-only days)
    for d in list(preloaded_chains.keys()):
        if d not in context_day_set:
            preloaded_chains[d] = precompute_chain(preloaded_chains[d])

    # Pre-compute day contexts
    preloaded_ctxs = {}
    for d in all_window_days:
        if d in preloaded_und:
            ctx = precompute_day_context(
                preloaded_und[d],
                preloaded_chains.get(d),
                preloaded_metas.get(d),
            )
            if ctx is not None:
                preloaded_ctxs[d] = ctx

    # Filter to days with complete data
    valid_train = [d for d in window.train_days
                   if d in preloaded_und and d in preloaded_chains
                   and d in preloaded_metas and d in preloaded_ctxs]
    valid_val = [d for d in window.val_days
                 if d in preloaded_und and d in preloaded_chains
                 and d in preloaded_metas and d in preloaded_ctxs]
    valid_fwd = [d for d in window.fwd_days
                 if d in preloaded_und and d in preloaded_chains
                 and d in preloaded_metas and d in preloaded_ctxs]

    if len(valid_train) < 20 or len(valid_val) < 5:
        return []

    if len(valid_fwd) < 5:
        print(f"  [WARN] Window {window.window_id}: only {len(valid_fwd)} forward days "
              f"(<5), skipping", flush=True)
        return []

    # Sample candidates
    candidates = sample_candidates(n=n_candidates, seed=seed)

    # Phase 1: Train evaluation
    train_results = []
    for idx, genes in enumerate(candidates):
        try:
            gene_dict = genes.to_dict()
            trades = []
            for d in valid_train:
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
            train_results.append((idx, fr, trades))
        except Exception:
            train_results.append((idx, FitnessResult(fitness=-99.0), []))

    # Sort by fitness, take top N with positive fitness
    viable = [(idx, fr, trades) for idx, fr, trades in train_results if fr.fitness > 0]
    viable.sort(key=lambda x: -x[1].fitness)
    top_viable = viable[:top_n]

    if not top_viable:
        return []

    # Phase 2: Val evaluation (filter gate only)
    regime_records = []
    for idx, train_fr, train_trades in top_viable:
        genes = candidates[idx]
        gene_dict = genes.to_dict()

        val_trades = []
        for d in valid_val:
            try:
                trade = evaluate_day(
                    preloaded_und[d],
                    preloaded_chains[d],
                    preloaded_metas[d],
                    gene_dict,
                    cost,
                    _ctx=preloaded_ctxs[d],
                )
                if trade is not None:
                    val_trades.append(trade)
            except Exception:
                continue

        val_fr = compute_fitness(val_trades, min_trades=1)

        if val_fr.n_trades < VAL_MIN_TRADES:
            continue
        val_cum_pnl = sum(t.pnl_net for t in val_trades)
        if val_fr.sharpe < VAL_MIN_SHARPE:
            continue

        # Robustness metrics on train trades
        train_pnls = [t.pnl_net for t in train_trades]
        robustness = compute_robustness_metrics(train_pnls, window_days=len(valid_train))
        robustness_dict = robustness.to_dict() if robustness is not None else {}

        # ── Phase 3: Kill-gated forward evaluation ──
        # Walk day-by-day, check kill conditions after each day.
        # Identical to options_walkforward.py forward loop.
        fwd_trades: list[OptionsTrade] = []
        fwd_exit_reason = "data_end"
        fwd_last_day = valid_fwd[0]

        for d in valid_fwd:
            try:
                trade = evaluate_day(
                    preloaded_und[d],
                    preloaded_chains[d],
                    preloaded_metas[d],
                    gene_dict,
                    cost,
                    _ctx=preloaded_ctxs[d],
                )
                if trade is not None:
                    fwd_trades.append(trade)
            except Exception:
                pass

            fwd_last_day = d

            # Check kill conditions
            killed, reason = check_kill(
                fwd_trades, kill_config,
                valid_fwd[0], d,
            )
            if killed:
                fwd_exit_reason = reason
                break

        fwd_cum_pnl = sum(t.pnl_net for t in fwd_trades) if fwd_trades else 0.0
        fwd_n_trades = len(fwd_trades)
        fwd_days_elapsed = (fwd_last_day - valid_fwd[0]).days if fwd_trades else 0
        fwd_wr = (sum(1 for t in fwd_trades if t.result == "win") / fwd_n_trades
                  if fwd_n_trades > 0 else 0.0)
        fwd_dd = 0.0
        if fwd_trades:
            running = np.concatenate(([0.0], np.cumsum([t.pnl_net for t in fwd_trades])))
            peak_arr = np.maximum.accumulate(running)
            fwd_dd = float((peak_arr - running).max())

        forward_profitable = fwd_cum_pnl > 0

        raw_trades = [
            {
                "trade_num": i + 1,
                "trade_date": t.trade_date,
                "direction": t.direction,
                "pnl_net": round(t.pnl_net, 2),
                "result": t.result,
                "exit_reason": t.exit_reason,
                "mfe_pct": round(t.mfe_pct, 4),
                "mae_pct": round(t.mae_pct, 4),
                "long_strike": t.long_strike,
                "short_strike": t.short_strike,
            }
            for i, t in enumerate(fwd_trades)
        ]

        record = {
            "window_id": window.window_id,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "val_start": window.val_start.isoformat(),
            "val_end": window.val_end.isoformat(),
            "n_train_days": len(valid_train),
            "n_val_days": len(valid_val),
            "genes": gene_dict,
            "genes_desc": describe_genes(genes),
            "archetype": genes.archetype,
            "direction": genes.direction,
            "trade_type": gene_dict.get("trade_type", {}),
            "train_fitness": _fr_to_dict(train_fr),
            "val_fitness": _fr_to_dict(val_fr),
            # Kill-gated forward test
            "forward_profitable": forward_profitable,
            "forward_source": "kill_gated",
            "fwd_cum_pnl": round(fwd_cum_pnl, 2),
            "fwd_n_trades": fwd_n_trades,
            "fwd_win_rate": round(fwd_wr, 4),
            "fwd_sharpe": None,  # not meaningful for variable-length periods
            "fwd_max_dd": round(fwd_dd, 2),
            "forward_days": fwd_days_elapsed,
            "forward_exit_reason": fwd_exit_reason,
            # Val metrics (for reference)
            "val_cum_pnl": round(val_cum_pnl, 2),
            "val_n_trades": val_fr.n_trades,
            "val_win_rate": val_fr.win_rate,
            "val_sharpe": val_fr.sharpe,
            # Robustness metrics from train trades
            "robustness": robustness_dict,
            # Per-trade raw data for post-hoc analysis
            "raw_trades": raw_trades,
        }
        regime_records.append(record)

    return regime_records


# ─── Full walk-forward run ────────────────────────────────────────────────────

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
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    train_size: int = 45,
    val_size: int = 15,
    step_size: int = 10,
    n_candidates: int = 2000,
    top_n: int = 20,
    min_trades: int = 30,
    seed: int = 42,
    kill_config: KillConfig | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Run the full walk-forward across all available data.

    Forward period is kill-gated (not fixed-length).
    """
    if output_path is None:
        output_path = Path("results/options_wf/regime_db.jsonl")
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

    cost = CostModel()

    print("[wf] Loading available days...", flush=True)
    all_days = get_available_days(data_dir, chain_subdir)
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

    # Pre-load ALL data once
    print("[wf] Pre-loading all data...", flush=True)
    t0 = tm.time()
    preloaded_und = {}
    preloaded_chains = {}
    preloaded_metas = {}

    for d in all_days:
        und = load_underlying(data_dir, d)
        if not und.empty:
            preloaded_und[d] = und
        chain = load_chain(data_dir, d, chain_subdir)
        if not chain.empty:
            preloaded_chains[d] = chain
        meta = load_chain_meta(data_dir, d, chain_subdir)
        if meta:
            preloaded_metas[d] = meta

    preload_sec = tm.time() - t0
    print(f"[wf] Pre-loaded {len(preloaded_und)} und, {len(preloaded_chains)} chains "
          f"in {preload_sec:.1f}s", flush=True)

    # Run each window
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
            chain_subdir=chain_subdir,
            n_candidates=n_candidates,
            top_n=top_n,
            min_trades=min_trades,
            seed=w_seed,
            cost=cost,
            kill_config=kill_config,
            preloaded_und=preloaded_und,
            preloaded_chains=preloaded_chains,
            preloaded_metas=preloaded_metas,
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

    # Summary
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward options regime runner")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("OPTIONS_DATA_DIR", "/data"))
    parser.add_argument("--chain-subdir", default="options_5dte")
    parser.add_argument("--train-days", type=int, default=45)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: random)")
    parser.add_argument("--output", type=str, default="results/options_wf/regime_db.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-all-kills", action="store_true",
                        help="Disable all kill conditions (forward runs to data_end)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(10000, 99999)
        print(f"[wf_runner] Auto-generated seed: {args.seed}")

    kill_cfg = KillConfig(skip_all_kills=args.skip_all_kills) if args.skip_all_kills else None

    run_walkforward(
        data_dir=Path(args.data_dir),
        chain_subdir=args.chain_subdir,
        train_size=args.train_days,
        val_size=args.val_days,
        step_size=args.step,
        n_candidates=args.candidates,
        top_n=args.top_n,
        min_trades=args.min_trades,
        seed=args.seed,
        kill_config=kill_cfg,
        output_path=Path(args.output),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
