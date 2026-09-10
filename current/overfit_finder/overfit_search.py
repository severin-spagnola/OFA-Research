"""
Overfit Finder — Parameterized Search Engine
=============================================
Deliberately overfits the MES friend strategy to the most recent N-day window.
Runs grid search + optional LLM-guided mutations over the parameter space,
ranks candidates by a fitness function that rewards recent performance,
and outputs the best candidate with full diagnostics.

The philosophy: overfit HARD to the last 60-90 days, trade it until it breaks
(2-3 consecutive losing days = regime death), then generate a new one.

Usage:
    python overfit_search.py                    # grid search, last 90 days
    python overfit_search.py --window 60        # last 60 days
    python overfit_search.py --llm              # add LLM structural mutations
    python overfit_search.py --top 5            # show top 5 candidates
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import io
import time as tm
from contextlib import redirect_stdout
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import multiprocessing as mp
import numpy as np
import pandas as pd

# Add the fvg_gap directory so we can import the strategy
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_FVG_DIR = _REPO_ROOT / "archived" / "fvg_gap"
sys.path.insert(0, str(_FVG_DIR))

import mes_backtest as mb
from mes_friend_strategy_base import (
    VariantConfig, VariantResult, run_variant, build_candles,
    equity_df, summarize, run_backtest_silent,
)
from strategy_dsl import StrategyDefinition, run_strategy
from regime_filter import score_strategy as _regime_filter_score, should_deploy as _regime_filter_deploy
from robustness_metrics import compute_robustness_metrics
from asset_config import AssetConfig, get_asset_config, DEFAULT_ASSET
from strategy_generator import (
    generate_strategies, refine_strategy, refine_seed,
    build_market_context,
)
from strategy_seeds import STRATEGY_SEEDS, get_seeds, get_seed_archetype, scale_seeds_for_asset
from macro_gate import MacroGate, load_macro_config

ET = ZoneInfo("US/Eastern")
PT = ZoneInfo("America/Los_Angeles")

# ─── Data path ────────────────────────────────────────────────────────────────
DATA_PATH = Path(os.environ.get("MES_DATA_PATH",
                                str(_REPO_ROOT / "data" / "mes" / "mes_1m.csv")))

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = _SCRIPT_DIR / "results"
UNCENSORED_LOG_PATH = Path(os.environ.get(
    "TRADER_DATA_DIR", str(_REPO_ROOT))) / "results" / "uncensored_mfe_mae.jsonl"

# Uncensored MFE/MAE collection — always on
_COLLECT_UNCENSORED = True
_UNCENSORED_RECORDS: list[dict] = []  # in-memory buffer, returned in job response
_REJECTED_RECORDS: list[dict] = []    # classifier rejections, returned in job response
_AUDIT_TRADES: list[dict] = []        # per-trade records for audit mode, returned in job response
_AUDIT_MODE: bool = False             # set by runpod_handler when audit_mode=True


# ─── Uncensored MFE/MAE Collection ──────────────────────────────────────────

def simulate_trade_uncensored(
    candles_1m: list,
    ts_index_1m: list,
    entry_time,
    entry_price: float,
    direction: str,
    max_bars: int = 390,
) -> dict:
    """Run trade with no SL/TP to collect true uncensored MFE/MAE.
    Never used for strategy selection — data collection only."""
    mfe = 0.0
    mae = 0.0
    mfe_bar = 0
    mae_bar = 0
    path = []

    lo = mb.bisect_left(ts_index_1m, entry_time)
    hi = min(lo + max_bars + 1, len(candles_1m))
    future = candles_1m[lo:hi]

    for i, bar in enumerate(future):
        if direction == "long":
            favorable = bar.high - entry_price
            adverse = entry_price - bar.low
        else:
            favorable = entry_price - bar.low
            adverse = bar.high - entry_price

        if favorable > mfe:
            mfe = favorable
            mfe_bar = i
        if adverse > mae:
            mae = adverse
            mae_bar = i

        path.append(round(favorable - adverse, 2))

        # Stop at EOD
        if bar.timestamp.time() >= time(16, 0):
            break

    return {
        "mfe_uncensored": round(mfe, 2),
        "mae_uncensored": round(mae, 2),
        "mfe_bar_uncensored": mfe_bar,
        "mae_bar_uncensored": mae_bar,
        "path": path[:50],
    }


def log_uncensored_candidate(candidate_id: str, archetype: str,
                              direction: str, entry_price: float,
                              uncensored: dict):
    """Append one uncensored MFE/MAE record to the in-memory buffer + disk log."""
    record = {
        "candidate_id": candidate_id,
        "archetype": archetype,
        "direction": direction,
        "entry_price": entry_price,
        "log_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        **{k: v for k, v in uncensored.items() if k != "path"},  # skip path array (too large for payload)
    }
    _UNCENSORED_RECORDS.append(record)
    # Also write to disk (useful for pod-mode runs with persistent /data)
    UNCENSORED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(UNCENSORED_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # silent — never block main pipeline


# ─── Meta-Configuration (tunable by orchestrator) ──────────────────────────

@dataclass
class MetaConfig:
    """All meta-parameters that control the walk-forward pipeline.

    These are the knobs the orchestrator tunes between batches.
    Defaults match the current hardcoded values.
    """
    # Archetype scores for the scoring gate
    archetype_scores: dict = field(default_factory=lambda: {
        "prev_close": 1.10, "smc_sweep": 1.33, "on_breakout": 0.50,
        "day_open": 0.0, "mean_reversion": -0.33,
        "gap_fill_fade": -0.82, "first_bar": -1.29,
        "gex_mean_reversion": 0.0, "gex_momentum": 0.0,
        "vwap_fade": 0.0, "vol_regime": 0.0, "earnings_catalyst": 0.0,
    })

    # Scoring gate
    gate_threshold: float = 0.0        # reject if gate_score < this
    fitness_bonus_high: float = 3.0    # fitness >= this → +0.5 to gate score
    fitness_bonus_low: float = 0.5     # fitness < this → -0.5 to gate score

    # LightGBM classifier gate (online trainer)
    classifier_confidence_threshold: float = 0.40  # reject if classifier P(winner) < this

    # Early regime death — effectively disabled (min_trades=999)
    early_death_wr_floor: float = 20.0   # kill if WR below this %
    early_death_min_trades: int = 999    # check early death after N trades (999 = disabled)
    max_dd_dollars: float = 2000.0       # max drawdown before kill

    # Regime death thresholds — effectively disabled (extreme values)
    negative_trajectory_pnl: float = -99999.0   # cum P&L < this after N trades (disabled)
    negative_trajectory_trades: int = 999
    flat_regime_pnl: float = 0.0         # |cum P&L| < this after N trades (disabled)
    flat_regime_trades: int = 999

    # Fitness composite weights
    sharpe_weight: float = 0.20
    pf_weight: float = 0.20
    trade_count_weight: float = 0.50
    return_dd_weight: float = 0.10

    # Overfit penalty params — penalize extreme training metrics
    sharpe_penalty_above: float = 5.0    # penalize Sharpe above this
    pf_penalty_above: float = 5.0        # penalize PF above this
    hard_sharpe_ceiling: float = 10.0    # hard reject above this
    hard_pf_ceiling: float = 8.0         # hard reject above this
    min_trade_count: int = 8             # hard reject below this

    # Trajectory scoring: DISABLED — losers have higher trajectory scores than winners
    trajectory_weight: float = 0.0

    # Bonus/penalty params for fitness scoring
    tsl_be_bonus: float = 0.15           # bonus for TSL/BE exit strategies
    frontload_penalty: float = 0.0       # DISABLED — no predictive power (43% vs 41%)
    sweet_spot_bonus: float = 0.20       # bonus for sweet-spot metrics

    # Recency bias: boost strategies that improved in the latter half of training.
    # Final fitness = (1 - recency_weight) * full_fitness + recency_weight * latter_half_fitness
    recency_weight: float = 0.40

    # MFE/MAE ratio gate — penalize overfit entries with extreme favorable excursion
    mfe_mae_ratio_ceiling: float = 5.0      # penalize ratio above this
    mfe_mae_ceiling_penalty: float = 0.20   # flat penalty when ratio exceeds ceiling

    # Bayesian edge scoring — reward strategies statistically unlikely to be noise
    bayesian_assumed_edge_wr: float = 0.70   # assumed WR if strategy has real edge
    bayesian_prior_edge: float = 0.15        # prior probability strategy has edge
    bayesian_weight: float = 0.20            # weight of Bayesian score in fitness
    bayesian_kill_threshold: float = 0.10    # kill forward regime when posterior drops below (effectively disabled)

    # Hard WR circuit breaker — catches catastrophic WR collapse faster than Bayesian
    wr_circuit_breaker_threshold: float = 0.25
    wr_circuit_breaker_min_trades: int = 8

    # mfe_before_mae entry quality signal
    mfe_before_mae_bonus_high: float = 0.10
    mfe_before_mae_bonus_mid: float = 0.05
    mfe_before_mae_penalty: float = -0.10
    mfe_before_mae_high_threshold: float = 0.70
    mfe_before_mae_mid_threshold: float = 0.50
    mfe_before_mae_low_threshold: float = 0.35

    # SL/TP ranges — sampled from by gene combinator and LLM
    sl_range_min: float = 8.0
    sl_range_max: float = 15.0
    tp_range_min: float = 10.0
    tp_range_max: float = 20.0

    # TSL trail distance ranges for gene sampling
    tsl_min_trail_pts: float = 4.0
    tsl_max_trail_pts: float = 6.0
    tsl_be_min_trigger_pts: float = 10.0
    tsl_be_max_trigger_pts: float = 13.0

    # Evolution confidence system
    evolution_base_threshold: float = 0.60   # base confidence needed to fire evolution
    evolution_cost_per: float = 0.15         # threshold increase per prior evolution
    evolution_min_trades: int = 5            # min trades for observation confidence
    evolution_stress_weight: float = 0.30    # how much stress lowers threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetaConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# Default config — load learned meta_config.json if it exists, else use defaults.
def _load_default_meta() -> MetaConfig:
    """Load meta_config.json from results dir (written by meta_optimizer).
    Falls back to MetaConfig() defaults if not found or invalid."""
    _meta_path = Path(__file__).resolve().parent.parent.parent / "results" / "meta_config.json"
    if _meta_path.exists():
        try:
            with open(_meta_path) as f:
                d = json.load(f)
            mc = MetaConfig.from_dict(d)
            print(f"[meta] Loaded learned config from {_meta_path.name} "
                  f"(gate={mc.gate_threshold}, recency={mc.recency_weight})")
            return mc
        except Exception:
            pass
    return MetaConfig()

DEFAULT_META_CONFIG = _load_default_meta()


# ─── Parameter Space ─────────────────────────────────────────────────────────

@dataclass
class ParamSet:
    """All tunable strategy parameters for one search candidate."""
    # Gap filtering
    m1_extrema_distance: float = 15.0
    gap_max_age_days: float = 7.0
    min_gap_size: float = 4.0
    oversized_gap: float = 20.0

    # Stop loss
    m1_max_sl_pts: float = 18.0
    sl_buffer: float = 3.0
    min_sl_pts: float = 8.0
    absolute_max_sl: float = 32.0

    # Take profit / partials
    large_stop: float = 18.0
    partial_profit: float = 14.0
    runner_r: float = 1.5

    # Penetration
    penetration: float = 3.0

    # Time
    m1_gap_cutoff_pt_hour: int = 5
    m1_gap_cutoff_pt_min: int = 0

    # Entry style
    entry_style: str = "wick"
    immediate_entry: bool = True

    # M2
    m2_breakout_min: float = 13.0
    enable_m2: bool = True

    # M3
    m3_start_pt_hour: int = 7
    enable_m3: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        """Short human-readable label for this param set."""
        parts = []
        if self.m1_extrema_distance != 15.0:
            parts.append(f"ext{self.m1_extrema_distance:.0f}")
        if self.gap_max_age_days != 7.0:
            parts.append(f"age{self.gap_max_age_days:.0f}")
        if self.m1_max_sl_pts != 18.0:
            parts.append(f"sl{self.m1_max_sl_pts:.0f}")
        if self.partial_profit != 14.0:
            parts.append(f"tp{self.partial_profit:.0f}")
        if self.runner_r != 1.5:
            parts.append(f"rr{self.runner_r:.1f}")
        if self.penetration != 3.0:
            parts.append(f"pen{self.penetration:.1f}")
        if self.entry_style != "wick":
            parts.append(self.entry_style)
        if not self.enable_m2:
            parts.append("noM2")
        if not self.enable_m3:
            parts.append("noM3")
        return "_".join(parts) if parts else "baseline"


# ─── Grid definition ─────────────────────────────────────────────────────────

GRID = {
    "m1_extrema_distance": [12.0, 15.0, 18.0, 20.0],
    "gap_max_age_days": [3.0, 5.0, 7.0, 10.0],
    "m1_max_sl_pts": [14.0, 18.0, 22.0],
    "partial_profit": [10.0, 14.0, 18.0],
    "runner_r": [1.0, 1.5, 2.0],
    "penetration": [2.0, 3.0, 4.0],
    "entry_style": ["wick", "body"],
    "enable_m2": [True, False],
    "enable_m3": [True, False],
}

# Reduced grid for quick runs — only structural params, tight bounds.
# Removed noise-fitting knobs: m1_extrema_distance (fixed at 15),
# gap_max_age_days (fixed at 5), runner_r capped at 1.5,
# m1_max_sl_pts capped at 14. These always overfit in walk-forward.
GRID_FAST = {
    "m1_max_sl_pts": [12.0, 14.0],
    "partial_profit": [8.0, 10.0, 14.0],
    "runner_r": [1.0, 1.5],
    "penetration": [2.0, 3.0],
}


# ─── Fitness function ─────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    """Fitness evaluation for one candidate."""
    fitness: float
    sharpe: float
    profit_factor: float
    total_pnl: float
    max_dd: float
    win_rate: float
    n_trades: int
    avg_trade_pnl: float
    # Penalties
    trade_count_penalty: float = 0.0
    dd_penalty: float = 0.0
    concentration_penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _raw_fitness(pnls: list[float], window_days: int,
                 meta: "MetaConfig", archetype: str = "",
                 has_tsl_be: bool = False) -> float:
    """Compute raw composite fitness with overfit penalties.

    Factored out so we can call it on full trade list AND on latter-half subset.
    Returns raw fitness float (not rounded, not clamped to FitnessResult).

    Mirrors compute_fitness() logic: caps Sharpe/PF contributions, penalizes
    extreme values, rewards moderate "sweet spot" metrics.
    """
    n = len(pnls)
    if n < 3:
        return -999.0

    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Sharpe
    if n >= 5:
        wd = max(int(window_days), 1)
        trades_per_day = n / wd
        annual_factor = np.sqrt(252 * max(trades_per_day, 0.1))
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
        std_floor = max(abs(mean_pnl) * 0.1, 1e-9)
        if not np.isfinite(std_pnl):
            std_pnl = std_floor
        else:
            std_pnl = max(std_pnl, std_floor)
        raw_sharpe = (mean_pnl / std_pnl * annual_factor) if std_pnl > 0 else 0.0
        sharpe = float(np.clip(raw_sharpe, -10.0, 10.0)) if np.isfinite(raw_sharpe) else 0.0
    else:
        sharpe = 0.0

    # Profit factor
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0

    # Hard rejection (match compute_fitness)
    hard_sharpe_ceil = getattr(meta, 'hard_sharpe_ceiling', 10.0)
    hard_pf_ceil = getattr(meta, 'hard_pf_ceiling', 8.0)
    min_tc = getattr(meta, 'min_trade_count', 8)
    # Archetype-aware: GEX/vol/IV archetypes have fewer trade opportunities
    _LOW_FREQ_ARCHETYPES = {"gex_momentum", "gex_mean_reversion", "vol_regime", "vwap_fade"}
    if archetype in _LOW_FREQ_ARCHETYPES:
        min_tc = max(3, min_tc // 2)
    if sharpe > hard_sharpe_ceil or profit_factor > hard_pf_ceil or n < min_tc:
        return -1.0

    # Overfit penalties
    sharpe_pen_above = getattr(meta, 'sharpe_penalty_above', 5.0)
    pf_pen_above = getattr(meta, 'pf_penalty_above', 5.0)
    sharpe_penalty = max(0, sharpe - sharpe_pen_above) * 0.3
    pf_penalty = max(0, profit_factor - pf_pen_above) * 0.2

    # Trade count bonus (scales to 1.0 at target)
    trade_target = 10.0 if archetype in _LOW_FREQ_ARCHETYPES else 20.0
    trade_bonus = min(n / trade_target, 1.0)

    # Sweet spot bonus: empirically optimal zone
    # TSL/BE strategies structurally compress Sharpe and PF (BE exits are ~$0 P&L),
    # so relax thresholds to avoid penalizing protective exit mechanics.
    if has_tsl_be:
        in_sweet_spot = (3 <= sharpe < 9 and
                         2 <= profit_factor < 5 and
                         5 <= n <= 12)
    else:
        in_sweet_spot = (4 <= sharpe < 9 and
                         3 <= profit_factor < 5 and
                         5 <= n <= 12)
    sweet_spot_bonus = meta.sweet_spot_bonus if in_sweet_spot else 0.0
    tsl_be_bonus = meta.tsl_be_bonus if has_tsl_be else 0.0

    # Return:DD from cumulative PnL series (no equity curve needed)
    return_dd_score = 0.0
    if total_pnl > 0:
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0
        if max_dd > 0:
            return_dd_score = min((total_pnl / max_dd) / 3.0, 1.0)

    return (
        min(sharpe, sharpe_pen_above) * meta.sharpe_weight
        + min(profit_factor, pf_pen_above) * meta.pf_weight
        + trade_bonus * meta.trade_count_weight
        + return_dd_score * meta.return_dd_weight
        + sweet_spot_bonus
        + tsl_be_bonus
        - sharpe_penalty
        - pf_penalty
    )


# ─── Structural scoring helpers ─────────────────────────────────────────────
# Based on empirical firing rates from 6,626 candidate analysis.
# These predict forward validation success better than in-sample metrics.

def _condition_type_score(strategy) -> float:
    """Score based on entry condition types and their empirical firing rates."""
    entry = getattr(strategy, 'entry', {}) or {}
    conditions = entry.get('conditions', []) if isinstance(entry, dict) else []
    score = 0.0
    for c in conditions:
        ct = c.get('type', '') if isinstance(c, dict) else ''
        if ct == 'price_broke':
            score += 0.08        # 48% fire rate, best
        elif ct in ('gex_zscore_above', 'gex_zscore_below'):
            score += 0.06        # 44-45% fire rate
        elif ct == 'realized_vol_below':
            score += 0.05        # 43% fire rate
        elif ct == 'iv_above':
            score -= 0.15        # 12% fire rate, nearly useless
        elif ct == 'realized_vol_above':
            score -= 0.10        # 21% fire rate
        elif ct == 'earnings_nearby':
            score -= 0.08        # 27% fire rate
        elif ct == 'pullback_to':
            score -= 0.05        # 31% fire rate
    return score


def _exit_type_score(strategy) -> float:
    """Penalize candle_wick SL (30-34% fire rate) vs fixed_pts (39-47%)."""
    exit_dict = getattr(strategy, 'exit', {}) or {}
    sl = exit_dict.get('sl', exit_dict.get('stop_loss', {}))
    sl_type = sl.get('type', '') if isinstance(sl, dict) else ''
    if sl_type == 'candle_wick':
        return -0.10
    elif sl_type == 'fixed_pts':
        return 0.05
    return 0.0


def _direction_score(strategy) -> float:
    """Short strategies fire at 44% vs long at 38%."""
    entry = getattr(strategy, 'entry', {}) or {}
    direction = entry.get('direction', '') if isinstance(entry, dict) else ''
    if direction == 'short':
        return 0.05
    elif direction == 'long':
        return -0.03
    return 0.0


def _time_window_score(strategy) -> float:
    """Afternoon strategies (start >= 12:00) fire at 36% vs morning 39-42%."""
    entry = getattr(strategy, 'entry', {}) or {}
    tw = entry.get('time_window', {}) if isinstance(entry, dict) else {}
    start = tw.get('start', '') if isinstance(tw, dict) else ''
    if start >= '12:00':
        return -0.08
    elif start and start <= '09:45':
        return 0.06
    return 0.0


# ─── Bayesian Edge Scoring ────────────────────────────────────────────────

from scipy.stats import binom as binom_dist


def bayesian_edge_probability(
    n_wins: int,
    n_trades: int,
    assumed_edge_wr: float,
    prior_edge: float,
) -> float:
    """Compute P(strategy has real edge | observed results) using Bayes theorem.

    Compares two hypotheses:
      H1: Strategy has real edge (fires at assumed_edge_wr win rate)
      H0: Strategy is noise (fires at 50/50)

    Returns posterior probability of H1 given observed n_wins/n_trades.
    """
    if n_trades < 3:
        return prior_edge  # insufficient data, return prior unchanged

    # Likelihood of observed results under each hypothesis
    p_given_edge = binom_dist.pmf(n_wins, n_trades, assumed_edge_wr)
    p_given_noise = binom_dist.pmf(n_wins, n_trades, 0.50)

    # Bayes theorem
    numerator = p_given_edge * prior_edge
    denominator = numerator + p_given_noise * (1.0 - prior_edge)

    if denominator < 1e-10:
        return prior_edge  # avoid division by zero

    return numerator / denominator


def compute_adaptive_dd_limit(
    train_pnls: list[float],
    n_forward_trades: int = 60,
    confidence: float = 0.95,
    n_sims: int = 2000,
    floor: float = 500.0,
    ceiling: float = 2000.0,
) -> float:
    """Compute strategy-specific DD kill threshold from training P&L distribution.

    Simulates n_sims equity paths by resampling from train_pnls (with replacement),
    computes max drawdown of each path, and returns the `confidence`-th percentile.
    The result is clamped to [floor, ceiling].

    This gives each strategy a personalized DD limit: tight-distribution strategies
    get a tight limit, wide-distribution strategies get a wider one.
    """
    import numpy as np
    if len(train_pnls) < 5:
        return ceiling

    pnls = np.array(train_pnls, dtype=float)
    rng = np.random.default_rng(42)

    max_dds = np.zeros(n_sims)
    for i in range(n_sims):
        # Resample n_forward_trades from training distribution
        sim_trades = rng.choice(pnls, size=n_forward_trades, replace=True)
        cumsum = np.cumsum(sim_trades)
        peak = np.maximum.accumulate(cumsum)
        max_dds[i] = float((peak - cumsum).max())

    dd_threshold = float(np.percentile(max_dds, confidence * 100))
    dd_threshold = max(floor, min(ceiling, dd_threshold))
    return round(dd_threshold, 0)


def compute_fitness(trades: list, equity: pd.DataFrame,
                    window_days: int,
                    meta: MetaConfig | None = None,
                    archetype: str = "",
                    has_tsl_be: bool = False,
                    strategy=None,
                    mfe_mae_ratio: float | None = None,
                    mfe_before_mae_pct: float | None = None) -> FitnessResult:
    """Compute fitness score for a set of trades.

    Rewards: high Sharpe, high profit factor, many trades
    Penalizes: large drawdowns, concentrated returns, too few trades

    Recency bias: blends full-window fitness with latter-half-only fitness.
    A strategy that was bad early but found a regime scores higher than one
    that was uniformly mediocre — because we care about CURRENT edge, not
    historical consistency (the forward test handles that).
    """
    if not trades or equity.empty:
        return FitnessResult(
            fitness=-999, sharpe=0, profit_factor=0, total_pnl=0,
            max_dd=0, win_rate=0, n_trades=0, avg_trade_pnl=0,
        )

    pnls = [float(t.pnl_dollars or 0) for t in trades]
    if not all(np.isfinite(p) for p in pnls):
        return FitnessResult(
            fitness=-999, sharpe=0, profit_factor=0, total_pnl=0,
            max_dd=0, win_rate=0, n_trades=0, avg_trade_pnl=0,
        )
    n = len(pnls)
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / n * 100 if n else 0
    avg_pnl = total_pnl / n if n else 0

    _m = meta or DEFAULT_META_CONFIG

    # Sharpe (annualized from daily-ish trades)
    if n >= 5:
        window_days = max(int(window_days), 1)
        trades_per_day = n / window_days
        annual_factor = np.sqrt(252 * max(trades_per_day, 0.1))
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
        std_floor = max(abs(mean_pnl) * 0.1, 1e-9)
        if not np.isfinite(std_pnl):
            std_pnl = std_floor
        else:
            std_pnl = max(std_pnl, std_floor)
        raw_sharpe = (mean_pnl / std_pnl * annual_factor) if std_pnl > 0 else 0.0
        if not np.isfinite(raw_sharpe):
            sharpe = 0.0
        else:
            sharpe = float(np.clip(raw_sharpe, -10.0, 10.0))
    else:
        sharpe = 0

    # Profit factor
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    max_dd = float(equity["drawdown_dollars"].max()) if not equity.empty else 0

    # ── Hard rejection: extreme overfit signals ──
    hard_sharpe_ceil = getattr(_m, 'hard_sharpe_ceiling', 10.0)
    hard_pf_ceil = getattr(_m, 'hard_pf_ceiling', 8.0)
    min_tc = getattr(_m, 'min_trade_count', 8)
    # Archetype-aware: GEX/vol/IV archetypes have fewer trade opportunities
    _LOW_FREQ_ARCHETYPES = {"gex_momentum", "gex_mean_reversion", "vol_regime", "vwap_fade"}
    if archetype in _LOW_FREQ_ARCHETYPES:
        min_tc = max(3, min_tc // 2)  # halve minimum for low-frequency archetypes
    if sharpe > hard_sharpe_ceil or profit_factor > hard_pf_ceil or n < min_tc:
        return FitnessResult(
            fitness=-1.0, sharpe=round(sharpe, 3),
            profit_factor=round(profit_factor, 3),
            total_pnl=round(total_pnl, 2), max_dd=round(max_dd, 2),
            win_rate=round(win_rate, 2), n_trades=n,
            avg_trade_pnl=round(avg_pnl, 2),
        )

    # ── Overfit penalties: extreme training metrics = overfit noise ──
    sharpe_pen_above = getattr(_m, 'sharpe_penalty_above', 5.0)
    pf_pen_above = getattr(_m, 'pf_penalty_above', 5.0)
    sharpe_penalty = max(0, sharpe - sharpe_pen_above) * 0.3
    pf_penalty = max(0, profit_factor - pf_pen_above) * 0.2

    # ── Trade count bonus (scales to 1.0 at target trade count) ──
    # Low-frequency archetypes get a lower target (10 instead of 20)
    trade_target = 10.0 if archetype in _LOW_FREQ_ARCHETYPES else 20.0
    trade_bonus = min(n / trade_target, 1.0)
    trade_count_penalty = round(1.0 - trade_bonus, 4)  # penalty = how much bonus was lost

    # ── Sweet spot bonus: empirically optimal zone = real signal ──
    # Sharpe [4-9), PF [3-5), Trades [5-12) → 62-65% forward WR historically
    # TSL/BE strategies structurally compress Sharpe and PF (BE exits are ~$0 P&L),
    # so relax thresholds to avoid penalizing protective exit mechanics.
    if has_tsl_be:
        in_sweet_spot = (3 <= sharpe < 9 and
                         2 <= profit_factor < 5 and
                         5 <= n <= 12)
    else:
        in_sweet_spot = (4 <= sharpe < 9 and
                         3 <= profit_factor < 5 and
                         5 <= n <= 12)
    sweet_spot_bonus = _m.sweet_spot_bonus if in_sweet_spot else 0.0
    tsl_be_bonus = _m.tsl_be_bonus if has_tsl_be else 0.0

    # ── Penalties (existing) ──

    # Drawdown relative to gains
    dd_penalty = 0.0
    if total_pnl > 0 and max_dd > total_pnl * 0.5:
        denom = max(total_pnl, 1.0)
        dd_penalty = (max_dd / denom - 0.5) * 1.0
        dd_penalty = min(dd_penalty, 5.0)

    # Concentration: top trades shouldn't dominate
    concentration_penalty = 0.0
    if n >= 10 and total_pnl > 0:
        sorted_pnls = sorted(pnls, reverse=True)
        top5_pnl = sum(sorted_pnls[:max(1, n // 20)])
        denom = max(total_pnl, 1.0)
        top5_ratio = top5_pnl / denom
        if top5_ratio > 0.5:
            concentration_penalty = (top5_ratio - 0.5) * 2.0
            concentration_penalty = min(concentration_penalty, 5.0)

    # Return:DD reward — incentivize high profit relative to drawdown
    return_dd_score = 0.0
    if total_pnl > 0 and max_dd > 0:
        return_dd_ratio = total_pnl / max_dd
        return_dd_score = min(return_dd_ratio / 3.0, 1.0)  # 3:1 ratio = perfect score

    # ── Full-window composite fitness ──
    # Sharpe and PF contributions are CAPPED — no reward for extremes
    full_fitness = (
        min(sharpe, sharpe_pen_above) * _m.sharpe_weight
        + min(profit_factor, pf_pen_above) * _m.pf_weight
        + trade_bonus * _m.trade_count_weight
        + return_dd_score * _m.return_dd_weight
        + sweet_spot_bonus
        + tsl_be_bonus
        - sharpe_penalty
        - pf_penalty
        - dd_penalty
        - concentration_penalty
    )

    # ── Structural scoring: predict forward firing from strategy structure ──
    if strategy is not None:
        structural_score = (
            _condition_type_score(strategy)
            + _exit_type_score(strategy)
            + _direction_score(strategy)
            + _time_window_score(strategy)
        )
        full_fitness += structural_score

    # ── MFE/MAE ratio gate — penalize overfit entries ──
    # High ratio (>5x) means price ran perfectly in training direction
    # but these entries collapse forward (empirical: 35% pass rate vs 60% for ratio<1)
    mfe_mae_penalty = 0.0
    if mfe_mae_ratio is not None and mfe_mae_ratio > _m.mfe_mae_ratio_ceiling:
        mfe_mae_penalty = _m.mfe_mae_ceiling_penalty
    full_fitness -= mfe_mae_penalty

    # ── mfe_before_mae quality signal — reward trades that go favorable first ──
    mfbm_bonus = 0.0
    if mfe_before_mae_pct is not None:
        if mfe_before_mae_pct >= _m.mfe_before_mae_high_threshold:
            mfbm_bonus = _m.mfe_before_mae_bonus_high
        elif mfe_before_mae_pct >= _m.mfe_before_mae_mid_threshold:
            mfbm_bonus = _m.mfe_before_mae_bonus_mid
        elif mfe_before_mae_pct < _m.mfe_before_mae_low_threshold:
            mfbm_bonus = _m.mfe_before_mae_penalty
    full_fitness += mfbm_bonus

    # ── Bayesian edge probability — reward strategies unlikely to be noise ──
    n_wins = len(wins)
    bayes_score = bayesian_edge_probability(
        n_wins=n_wins,
        n_trades=n,
        assumed_edge_wr=_m.bayesian_assumed_edge_wr,
        prior_edge=_m.bayesian_prior_edge,
    )
    full_fitness += bayes_score * _m.bayesian_weight

    # ── Recency bias: score the latter half of trades separately ──
    # A strategy that caught a regime switch mid-training should be boosted,
    # not killed by aggregate stats that dilute the recent edge.
    recency_w = _m.recency_weight
    if recency_w > 0 and n >= 6:
        mid = n // 2
        latter_pnls = pnls[mid:]
        # Use half the window for the latter subset (approximate)
        latter_window = max(window_days // 2, 1)
        latter_fitness = _raw_fitness(latter_pnls, latter_window, _m, archetype=archetype,
                                     has_tsl_be=has_tsl_be)
        if np.isfinite(latter_fitness):
            fitness = (1.0 - recency_w) * full_fitness + recency_w * latter_fitness
        else:
            fitness = full_fitness
    else:
        fitness = full_fitness

    if not np.isfinite(fitness):
        fitness = -999.0

    return FitnessResult(
        fitness=round(fitness, 4),
        sharpe=round(sharpe, 3),
        profit_factor=round(profit_factor, 3),
        total_pnl=round(total_pnl, 2),
        max_dd=round(max_dd, 2),
        win_rate=round(win_rate, 2),
        n_trades=n,
        avg_trade_pnl=round(avg_pnl, 2),
        trade_count_penalty=round(trade_count_penalty, 4),
        dd_penalty=round(dd_penalty, 4),
        concentration_penalty=round(concentration_penalty, 4),
    )


# ─── Backtest runner ──────────────────────────────────────────────────────────

def _resolve_data_path(asset: str) -> str:
    """Resolve the data file path for a given asset."""
    if asset.upper() == "MES":
        return str(DATA_PATH)
    # Check for env var override: F_DATA_PATH, BAC_DATA_PATH, etc.
    env_key = f"{asset.upper()}_DATA_PATH"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    # Default: /data/{ticker}/{ticker}_1m.csv (RunPod) or local data dir
    runpod_path = Path(f"/data/{asset.lower()}/{asset.lower()}_1m.csv")
    if runpod_path.exists():
        return str(runpod_path)
    local_path = _REPO_ROOT / "data" / asset.lower() / f"{asset.lower()}_1m.csv"
    return str(local_path)


class DataCache:
    """Load instrument data once, slice for different windows."""

    def __init__(self, data_path: str = None, asset: str = "MES"):
        self.asset_config = get_asset_config(asset)
        if data_path is None:
            data_path = _resolve_data_path(asset)

        print(f"Loading {self.asset_config.ticker} 1m data from {data_path}...")
        t0 = tm.time()

        if self.asset_config.data_loader == "databento":
            self.candles_1m = mb.load_databento_data(data_path)
        else:
            from equity_data import load_equity_data
            self.candles_1m = load_equity_data(data_path)

        self.candles_15m = mb.build_15m_candles(self.candles_1m)
        self.candles_1h = build_candles(self.candles_1m, 60)
        elapsed = tm.time() - t0
        print(f"  Loaded {len(self.candles_1m):,} 1m candles in {elapsed:.1f}s")

        # Build timestamp index for slicing
        self._ts_1m = [c.timestamp for c in self.candles_1m]
        self._ts_15m = [c.timestamp for c in self.candles_15m]
        self._ts_1h = [c.timestamp for c in self.candles_1h]

        # Load GEX cache (SPY options → applies to MES and S&P-linked assets)
        self.gex_cache = None
        try:
            from gex_features import GexCache
            gc = GexCache()
            first, last = self.date_range
            gc.precompute(start_date=first, end_date=last)
            if gc.stats.get("n_snapshots", 0) > 0:
                self.gex_cache = gc
                print(f"  GEX: {gc.stats}")
        except Exception as e:
            print(f"  GEX cache not available: {e}")

        # Load DailyCache (VWAP, realized vol, earnings)
        self.daily_cache = None
        try:
            from gex_features import DailyCache
            dc = DailyCache()
            first, last = self.date_range
            dc.precompute(start_date=first, end_date=last)
            if dc.stats.get("n_days", 0) > 0:
                self.daily_cache = dc
                print(f"  Daily: {dc.stats}")
        except Exception as e:
            print(f"  Daily cache not available: {e}")

    def slice_window(self, start_date: str, end_date: str):
        """Return candle lists filtered to [start, end] date range."""
        start = pd.Timestamp(start_date, tz=ET)
        end = pd.Timestamp(end_date, tz=ET) + pd.Timedelta(days=1)

        # We need some lookback before start for gap detection
        lookback_start = start - pd.Timedelta(days=30)

        c1m = [c for c in self.candles_1m if lookback_start <= c.timestamp < end]
        c15m = [c for c in self.candles_15m if lookback_start <= c.timestamp < end]
        c1h = [c for c in self.candles_1h if lookback_start <= c.timestamp < end]

        return c1m, c15m, c1h

    @property
    def date_range(self) -> tuple[str, str]:
        first = self.candles_1m[0].timestamp.strftime("%Y-%m-%d")
        last = self.candles_1m[-1].timestamp.strftime("%Y-%m-%d")
        return first, last


def apply_params(params: ParamSet):
    """Monkey-patch the strategy module's global constants with param values.

    This is ugly but effective — the strategy reads globals at runtime,
    so we just overwrite them before each backtest call.
    """
    import mes_friend_strategy_base as strat

    strat.M1_EXTREMA_DISTANCE = params.m1_extrema_distance
    strat.M1_POTENTIAL_DISTANCE = params.m1_extrema_distance
    strat.GAP_MAX_AGE_DAYS = params.gap_max_age_days
    strat.MIN_GAP_SIZE = params.min_gap_size
    strat.OVERSIZED_GAP = params.oversized_gap
    strat.M1_MAX_SL_PTS = params.m1_max_sl_pts
    strat.SL_BUFFER = params.sl_buffer
    strat.MIN_SL_PTS = params.min_sl_pts
    strat.ABSOLUTE_MAX_SL = params.absolute_max_sl
    strat.LARGE_STOP = params.large_stop
    strat.PARTIAL_PROFIT = params.partial_profit
    strat.RUNNER_R = params.runner_r
    strat.PENETRATION = params.penetration
    strat.M2_BREAKOUT_MIN = params.m2_breakout_min
    strat.M3_START_PT = time(params.m3_start_pt_hour, 0)


def run_candidate(params: ParamSet, c1m, c15m, c1h,
                  window_days: int,
                  precomputed: dict = None,
                  return_trades: bool = False):
    """Run one backtest candidate and return fitness + summary.

    precomputed: optional dict with gap_snapshots_15m, gap_snapshots_15m_bff,
                 gap_snapshots_1h, gap_ts_15m, gap_ts_1h for fast gap lookups.
    return_trades: if True, returns (fitness, summary, trades, equity) instead
                   of (fitness, summary).
    """
    pre = precomputed or {}

    apply_params(params)
    config = VariantConfig(
        name=params.label,
        m1_gap_cutoff_pt=time(params.m1_gap_cutoff_pt_hour,
                              params.m1_gap_cutoff_pt_min),
        entry_style=params.entry_style,
        immediate_entry=params.immediate_entry,
        priority_mode="legacy",
        enable_m2_rewrite=False,
    )
    result = run_backtest_silent(run_variant, config, c1m, c15m, c1h, **pre)

    fitness = compute_fitness(result.trades, result.equity, window_days)
    summary = result.summary

    if return_trades:
        return fitness, summary, result.trades, result.equity
    return fitness, summary


# ─── Parallel candidate evaluation ──────────────────────────────────────────

# Module-level globals for multiprocessing workers (set by _init_worker)
_worker_c1m = None
_worker_c15m = None
_worker_c1h = None
_worker_precomputed = None
_worker_window_days = None


def _init_worker(c1m, c15m, c1h, precomputed, window_days):
    """Initialize worker process with shared data (called once per worker)."""
    global _worker_c1m, _worker_c15m, _worker_c1h, _worker_precomputed, _worker_window_days
    _worker_c1m = c1m
    _worker_c15m = c15m
    _worker_c1h = c1h
    _worker_precomputed = precomputed
    _worker_window_days = window_days


def _eval_candidate(params: ParamSet) -> SearchResult | None:
    """Evaluate a single candidate in a worker process."""
    try:
        fitness, summary = run_candidate(
            params, _worker_c1m, _worker_c15m, _worker_c1h,
            _worker_window_days, precomputed=_worker_precomputed,
        )
        return SearchResult(rank=0, params=params, fitness=fitness, summary=summary)
    except Exception:
        return None


def run_search_parallel(
    candidates: list[ParamSet],
    c1m, c15m, c1h,
    precomputed: dict,
    window_days: int,
    n_workers: int = 0,
) -> list[SearchResult]:
    """Evaluate candidates in parallel using multiprocessing.

    Args:
        n_workers: Number of worker processes. 0 = auto (cpu_count).
    """
    if n_workers <= 0:
        n_workers = min(mp.cpu_count(), len(candidates))

    # For small candidate counts or single core, run sequentially
    if n_workers <= 1 or len(candidates) <= 4:
        results = []
        for i, params in enumerate(candidates):
            try:
                fitness, summary = run_candidate(
                    params, c1m, c15m, c1h, window_days,
                    precomputed=precomputed,
                )
                results.append(SearchResult(
                    rank=0, params=params, fitness=fitness, summary=summary,
                ))
            except Exception:
                continue
            if (i + 1) % 25 == 0 or (i + 1) == len(candidates):
                print(f"  [{i+1}/{len(candidates)}] "
                      f"best so far: {max(r.fitness.fitness for r in results):.3f}"
                      if results else f"  [{i+1}/{len(candidates)}]")
        return results

    print(f"  Parallel: {n_workers} workers, {len(candidates)} candidates")
    t0 = tm.time()

    # Use fork-based start method for copy-on-write memory sharing
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(c1m, c15m, c1h, precomputed, window_days),
    ) as pool:
        raw_results = pool.map(_eval_candidate, candidates, chunksize=4)

    results = [r for r in raw_results if r is not None]
    elapsed = tm.time() - t0
    rate = len(candidates) / elapsed if elapsed > 0 else 0
    print(f"  [{len(candidates)}/{len(candidates)}] {rate:.1f}/sec parallel | "
          f"{elapsed:.1f}s total")

    return results


# ─── Search strategies ────────────────────────────────────────────────────────

def generate_grid(grid: dict = None, fast: bool = False) -> list[ParamSet]:
    """Generate all parameter combinations from the grid."""
    if grid is None:
        grid = GRID_FAST if fast else GRID

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    param_sets = []
    for combo in combos:
        kwargs = dict(zip(keys, combo))
        param_sets.append(ParamSet(**kwargs))

    return param_sets


def generate_random(n: int = 50, seed: int = 42) -> list[ParamSet]:
    """Generate random parameter sets (Latin hypercube style)."""
    rng = np.random.default_rng(seed)
    param_sets = []

    for _ in range(n):
        ps = ParamSet(
            m1_extrema_distance=rng.choice([10, 12, 15, 18, 20, 25]),
            gap_max_age_days=rng.choice([2, 3, 5, 7, 10, 14]),
            m1_max_sl_pts=rng.choice([12, 14, 16, 18, 20, 24]),
            partial_profit=rng.choice([8, 10, 12, 14, 16, 18, 20]),
            runner_r=rng.choice([0.8, 1.0, 1.2, 1.5, 2.0, 2.5]),
            penetration=rng.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            entry_style=rng.choice(["wick", "body"]),
            enable_m2=bool(rng.choice([True, False])),
            enable_m3=bool(rng.choice([True, False])),
            sl_buffer=rng.choice([2.0, 3.0, 4.0]),
            min_sl_pts=rng.choice([6.0, 8.0, 10.0]),
            m2_breakout_min=rng.choice([10.0, 13.0, 16.0]),
        )
        param_sets.append(ps)

    return param_sets


# ─── Main search loop ─────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    rank: int
    params: ParamSet
    fitness: FitnessResult
    summary: dict


def run_search(
    window_days: int = 90,
    fast: bool = True,
    n_random: int = 0,
    top_n: int = 10,
    custom_params: list[ParamSet] = None,
) -> list[SearchResult]:
    """Run the full overfit search.

    Args:
        window_days: How many recent days to overfit to
        fast: Use reduced grid (faster, fewer combos)
        n_random: Additional random candidates to try
        top_n: Return this many top candidates
        custom_params: If provided, use these instead of grid
    """
    # Load data
    cache = DataCache()
    first_date, last_date = cache.date_range

    # Compute window
    end = pd.Timestamp(last_date)
    start = end - pd.Timedelta(days=window_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"\nOverfit window: {start_str} to {end_str} ({window_days} days)")
    print(f"Data range: {first_date} to {last_date}")

    # Slice data to window (+ lookback for gap detection)
    c1m, c15m, c1h = cache.slice_window(start_str, end_str)
    print(f"Window candles: {len(c1m):,} 1m, {len(c15m):,} 15m, {len(c1h):,} 1h")

    # Precompute gap snapshots
    print("Precomputing gap snapshots...")
    t0_pre = tm.time()
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    snaps_15m_bff, _ = mb.precompute_gap_index(c15m, body_fill_first=True)
    snaps_1h, ts_1h_idx = mb.precompute_gap_index(c1h, body_fill_first=False)
    precomputed = {
        "gap_snapshots_15m": snaps_15m,
        "gap_snapshots_15m_bff": snaps_15m_bff,
        "gap_snapshots_1h": snaps_1h,
        "gap_ts_15m": ts_15m_idx,
        "gap_ts_1h": ts_1h_idx,
    }
    print(f"  Precomputed in {tm.time() - t0_pre:.1f}s "
          f"({len(snaps_15m):,} 15m + {len(snaps_1h):,} 1h snapshots)")

    # Generate candidates
    if custom_params:
        candidates = custom_params
    else:
        candidates = generate_grid(fast=fast)
        if n_random > 0:
            candidates.extend(generate_random(n=n_random))

    n_workers = int(os.environ.get("OVERFIT_WORKERS", "0"))
    print(f"\nSearching {len(candidates)} candidates...")
    t0 = tm.time()

    results = run_search_parallel(
        candidates, c1m, c15m, c1h, precomputed, window_days,
        n_workers=n_workers,
    )

    # Rank by fitness
    results.sort(key=lambda r: r.fitness.fitness, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    elapsed = tm.time() - t0
    print(f"\nSearch complete: {len(results)} candidates in {elapsed:.1f}s")

    return results[:top_n]


# ─── Windowed search (for walk-forward) ──────────────────────────────────────

def run_search_windowed(
    cache: DataCache,
    start_date: str,
    end_date: str,
    window_days: int = 90,
    fast: bool = True,
    top_n: int = 1,
) -> list[SearchResult]:
    """Run overfit search on a specific date window using pre-loaded data.

    Like run_search() but takes a pre-loaded DataCache and explicit dates
    instead of computing "last N days from data end."
    """
    c1m, c15m, c1h = cache.slice_window(start_date, end_date)

    # Precompute gap snapshots
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    snaps_15m_bff, _ = mb.precompute_gap_index(c15m, body_fill_first=True)
    snaps_1h, ts_1h_idx = mb.precompute_gap_index(c1h, body_fill_first=False)
    precomputed = {
        "gap_snapshots_15m": snaps_15m,
        "gap_snapshots_15m_bff": snaps_15m_bff,
        "gap_snapshots_1h": snaps_1h,
        "gap_ts_15m": ts_15m_idx,
        "gap_ts_1h": ts_1h_idx,
    }

    # Generate candidates
    candidates = generate_grid(fast=fast)

    n_workers = int(os.environ.get("OVERFIT_WORKERS", "0"))
    results = run_search_parallel(
        candidates, c1m, c15m, c1h, precomputed, window_days,
        n_workers=n_workers,
    )

    results.sort(key=lambda r: r.fitness.fitness, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results[:top_n]


def run_forward_test(
    cache: DataCache,
    params: ParamSet,
    start_date: str,
    end_date: str,
) -> tuple[list, pd.DataFrame, dict]:
    """Run a single backtest with specific params on a forward date range.

    Returns (trades, equity_df, summary).
    """
    c1m, c15m, c1h = cache.slice_window(start_date, end_date)
    window_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days

    # Precompute gap snapshots
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    snaps_15m_bff, _ = mb.precompute_gap_index(c15m, body_fill_first=True)
    snaps_1h, ts_1h_idx = mb.precompute_gap_index(c1h, body_fill_first=False)
    precomputed = {
        "gap_snapshots_15m": snaps_15m,
        "gap_snapshots_15m_bff": snaps_15m_bff,
        "gap_snapshots_1h": snaps_1h,
        "gap_ts_15m": ts_15m_idx,
        "gap_ts_1h": ts_1h_idx,
    }

    fitness, summary, trades, equity = run_candidate(
        params, c1m, c15m, c1h, window_days,
        precomputed=precomputed, return_trades=True,
    )

    # Filter trades to only those with entry on/after start_date
    # (slice_window includes 30-day lookback for gap context, so backtest
    # may produce trades before the actual forward period)
    start_ts = pd.Timestamp(start_date, tz=ET)
    trades = [t for t in trades
              if t.entry_time and t.entry_time >= start_ts]

    return trades, equity, summary


# ─── LLM Strategy Search ────────────────────────────────────────────────────

def run_strategy_candidate(
    strategy: StrategyDefinition,
    c1m, c15m, c1h,
    window_days: int,
    precomputed: tuple = None,
    start_date: str = None,
    end_date: str = None,
    return_trades: bool = False,
    asset_config: AssetConfig | None = None,
    gex_cache=None,
    daily_cache=None,
    archetype: str = "",
    overnight: bool = False,
):
    """Run one LLM-generated strategy and return fitness + summary.

    Like run_candidate() but takes a StrategyDefinition instead of ParamSet.
    """
    _meta = DEFAULT_META_CONFIG
    trades = run_strategy(
        strategy, c1m, c15m, c1h,
        precomputed=precomputed,
        start_date=start_date,
        end_date=end_date,
        asset_config=asset_config,
        gex_cache=gex_cache,
        daily_cache=daily_cache,
        collect_uncensored=True,
        overnight=overnight,
    )

    # Build equity DataFrame for fitness computation
    eq = equity_df(trades) if trades else pd.DataFrame()

    # Detect TSL/BE exit params for fitness compensation
    _exit = getattr(strategy, 'exit', {}) or {}
    _has_tsl_be = bool(_exit.get('be_trigger_pts') or _exit.get('trail_distance_pts'))

    # Compute MFE/MAE before fitness so we can pass ratio into the gate
    n = len(trades)
    pnls = [float(t.pnl_dollars or 0) for t in trades]
    mfe_vals = [t.mfe_pts for t in trades if hasattr(t, 'mfe_pts') and t.mfe_pts > 0]
    mae_vals = [t.mae_pts for t in trades if hasattr(t, 'mae_pts') and t.mae_pts > 0]
    sorted_mfe = sorted(mfe_vals) if mfe_vals else []
    sorted_mae = sorted(mae_vals) if mae_vals else []
    _mfe_mae_ratio = (
        round(sorted_mfe[len(sorted_mfe) // 2] / sorted_mae[len(sorted_mae) // 2], 2)
        if sorted_mfe and sorted_mae and sorted_mae[len(sorted_mae) // 2] > 0 else None
    )

    # Enhanced instrumentation: bars_to_mfe/mae, mfe_before_mae
    bars_to_mfe_vals = [t.bars_to_mfe for t in trades
                        if hasattr(t, 'bars_to_mfe') and t.bars_to_mfe > 0]
    bars_to_mae_vals = [t.bars_to_mae for t in trades
                        if hasattr(t, 'bars_to_mae') and t.bars_to_mae > 0]
    mfbm_vals = [t.mfe_before_mae for t in trades
                 if hasattr(t, 'mfe_before_mae') and t.mfe_before_mae is not None]
    _bars_mfe_med = sorted(bars_to_mfe_vals)[len(bars_to_mfe_vals) // 2] if bars_to_mfe_vals else None
    _bars_mae_med = sorted(bars_to_mae_vals)[len(bars_to_mae_vals) // 2] if bars_to_mae_vals else None
    _mfbm_pct = round(sum(1.0 for v in mfbm_vals if v) / len(mfbm_vals), 3) if mfbm_vals else None

    fitness = compute_fitness(trades, eq, window_days, archetype=archetype,
                              has_tsl_be=_has_tsl_be, strategy=strategy,
                              mfe_mae_ratio=_mfe_mae_ratio,
                              mfe_before_mae_pct=_mfbm_pct)

    # Build summary dict
    summary = {
        "n_trades": n,
        "total_pnl": sum(pnls),
        "win_rate": (sum(1 for p in pnls if p > 0) / n * 100) if n else 0,
        "strategy_name": strategy.name,
        "pnls_array": [round(p, 2) for p in pnls],
        "mfe_array": [round(t.mfe_pts, 2) for t in trades if hasattr(t, 'mfe_pts')],
        "mae_array": [round(t.mae_pts, 2) for t in trades if hasattr(t, 'mae_pts')],
        "mfe_p50": round(sorted_mfe[len(sorted_mfe) // 2], 2) if sorted_mfe else None,
        "mfe_p75": round(sorted_mfe[int(len(sorted_mfe) * 0.75)], 2) if sorted_mfe else None,
        "mae_p50": round(sorted_mae[len(sorted_mae) // 2], 2) if sorted_mae else None,
        "mae_p75": round(sorted_mae[int(len(sorted_mae) * 0.75)], 2) if sorted_mae else None,
        "mfe_mae_ratio": _mfe_mae_ratio,
        "bars_to_mfe_median": _bars_mfe_med,
        "bars_to_mae_median": _bars_mae_med,
        "mfe_before_mae_pct": _mfbm_pct,
        "mfe_before_mae_raw": mfbm_vals,
    }

    # Compute robustness metrics (observational only — no scoring impact)
    rob = compute_robustness_metrics(pnls, window_days)
    if rob is not None:
        summary["robustness"] = rob.to_dict()

    if return_trades:
        return fitness, summary, trades, eq
    return fitness, summary


# Module-level globals for LLM strategy worker
_worker_strategy_list = None
_worker_start_date = None
_worker_end_date = None


def _init_strategy_worker(c1m, c15m, c1h, precomputed, window_days,
                           start_date, end_date, strategies, asset_config,
                           gex_cache=None, daily_cache=None, overnight=False):
    """Initialize worker for LLM strategy evaluation."""
    global _worker_c1m, _worker_c15m, _worker_c1h, _worker_precomputed
    global _worker_window_days, _worker_strategy_list
    global _worker_start_date, _worker_end_date, _worker_asset_config
    global _worker_gex_cache, _worker_daily_cache, _worker_overnight
    _worker_c1m = c1m
    _worker_c15m = c15m
    _worker_c1h = c1h
    _worker_precomputed = precomputed
    _worker_window_days = window_days
    _worker_strategy_list = strategies
    _worker_start_date = start_date
    _worker_end_date = end_date
    _worker_asset_config = asset_config
    _worker_gex_cache = gex_cache
    _worker_daily_cache = daily_cache
    _worker_overnight = overnight


def _eval_strategy(idx: int) -> tuple[int, FitnessResult, dict] | None:
    """Evaluate a single LLM strategy in a worker process."""
    try:
        strategy = _worker_strategy_list[idx]
        fitness, summary = run_strategy_candidate(
            strategy, _worker_c1m, _worker_c15m, _worker_c1h,
            _worker_window_days, precomputed=_worker_precomputed,
            start_date=_worker_start_date, end_date=_worker_end_date,
            asset_config=_worker_asset_config,
            gex_cache=_worker_gex_cache,
            daily_cache=_worker_daily_cache,
            archetype=getattr(strategy, 'archetype', ''),
            overnight=_worker_overnight,
        )
        return idx, fitness, summary
    except Exception as e:
        print(f"    Strategy #{idx} failed: {e}")
        return None


def _parallel_eval_strategies(
    strategies: list,
    c1m, c15m, c1h,
    precomputed: tuple,
    window_days: int,
    start_date: str,
    end_date: str,
    asset_config: AssetConfig | None = None,
    gex_cache=None,
    daily_cache=None,
    overnight: bool = False,
) -> list[tuple[int, FitnessResult, dict]]:
    """Evaluate a list of StrategyDefinition objects in parallel."""
    n = len(strategies)
    if n == 0:
        return []

    n_workers = min(mp.cpu_count(), n)
    if n_workers <= 1 or n <= 2:
        # Sequential fallback for tiny batches
        results = []
        for i, strat in enumerate(strategies):
            try:
                fitness, summary = run_strategy_candidate(
                    strat, c1m, c15m, c1h, window_days,
                    precomputed=precomputed,
                    start_date=start_date, end_date=end_date,
                    asset_config=asset_config,
                    gex_cache=gex_cache,
                    daily_cache=daily_cache,
                    archetype=getattr(strat, 'archetype', ''),
                    overnight=overnight,
                )
                results.append((i, fitness, summary))
            except Exception:
                continue
        return results

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_strategy_worker,
        initargs=(c1m, c15m, c1h, precomputed, window_days,
                  start_date, end_date, strategies, asset_config,
                  gex_cache, daily_cache, overnight),
    ) as pool:
        raw = pool.map(_eval_strategy, range(n), chunksize=max(1, n // n_workers))

    return [r for r in raw if r is not None]


def run_llm_search(
    cache: DataCache,
    start_date: str,
    end_date: str,
    window_days: int = 90,
    n_strategies: int = 10,
    n_refinements: int = 5,
    prior_results: list[dict] | None = None,
    top_n: int = 1,
) -> list[SearchResult]:
    """Generate LLM strategies, evaluate them, refine winners.

    1. Build market context from candle data
    2. Generate n_strategies initial candidates via LLM
    3. Backtest all candidates
    4. Take top 3, refine each into n_refinements variants
    5. Backtest refinements, return ranked results
    """
    ac = cache.asset_config
    c1m, c15m, c1h = cache.slice_window(start_date, end_date)

    # Precompute gap index for strategy DSL
    precomputed = None
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    precomputed = (snaps_15m, ts_15m_idx)

    # Build market context
    print(f"  Building market context for {start_date} to {end_date}...")
    context = build_market_context(c1m, start_date, end_date, asset_config=ac)

    # Generate initial strategies
    strategies = generate_strategies(context, n=n_strategies,
                                      prior_results=prior_results,
                                      asset_config=ac)
    if not strategies:
        print("  No strategies generated. Falling back to grid search.")
        return run_search_windowed(cache, start_date, end_date,
                                    window_days=window_days, fast=True, top_n=top_n)

    # Evaluate initial strategies
    print(f"  Evaluating {len(strategies)} initial strategies...")
    results = []
    for i, strat in enumerate(strategies):
        try:
            fitness, summary = run_strategy_candidate(
                strat, c1m, c15m, c1h, window_days,
                precomputed=precomputed,
                start_date=start_date, end_date=end_date,
                asset_config=ac,
                gex_cache=cache.gex_cache,
                daily_cache=cache.daily_cache,
            )
            sr = SearchResult(
                rank=0,
                params=ParamSet(),  # placeholder — strategy stored in summary
                fitness=fitness,
                summary=summary,
            )
            # Store strategy definition in summary for later retrieval
            sr.summary["strategy_def"] = strat.to_dict()
            results.append(sr)
            status = f"${fitness.total_pnl:>8,.0f} | {fitness.win_rate:>5.1f}% WR | " \
                     f"{fitness.n_trades} trades | fit={fitness.fitness:.3f}"
            print(f"    [{i+1}/{len(strategies)}] {strat.name}: {status}")
        except Exception as e:
            print(f"    [{i+1}/{len(strategies)}] {strat.name}: FAILED ({e})")
            continue

    if not results:
        print("  All strategies failed evaluation.")
        return []

    # Sort by fitness
    results.sort(key=lambda r: r.fitness.fitness, reverse=True)

    # Refine top 3 winners
    top3 = results[:min(3, len(results))]
    if n_refinements > 0 and top3[0].fitness.fitness > 0:
        print(f"\n  Refining top {len(top3)} strategies...")
        refinement_results = []

        for sr in top3:
            strat_dict = sr.summary.get("strategy_def", {})
            strat = StrategyDefinition.from_dict(strat_dict)
            variants = refine_strategy(
                strat, sr.fitness.to_dict(), context,
                n_variants=n_refinements,
                asset_config=ac,
            )

            for j, var in enumerate(variants):
                try:
                    fitness, summary = run_strategy_candidate(
                        var, c1m, c15m, c1h, window_days,
                        precomputed=precomputed,
                        start_date=start_date, end_date=end_date,
                        asset_config=ac,
                        gex_cache=cache.gex_cache,
                        daily_cache=cache.daily_cache,
                    )
                    ref_sr = SearchResult(
                        rank=0, params=ParamSet(),
                        fitness=fitness, summary=summary,
                    )
                    ref_sr.summary["strategy_def"] = var.to_dict()
                    refinement_results.append(ref_sr)
                    print(f"      Refinement: {var.name} → fit={fitness.fitness:.3f}")
                except Exception:
                    continue

        results.extend(refinement_results)

    # Final ranking
    results.sort(key=lambda r: r.fitness.fitness, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results[:top_n]


def run_forward_test_llm(
    cache: DataCache,
    strategy: StrategyDefinition,
    start_date: str,
    end_date: str,
    overnight: bool = False,
) -> tuple[list, pd.DataFrame, dict]:
    """Forward-test an LLM strategy on a date range.

    Returns (trades, equity_df, summary).
    """
    ac = cache.asset_config
    c1m, c15m, c1h = cache.slice_window(start_date, end_date)

    # Precompute gap index
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    precomputed = (snaps_15m, ts_15m_idx)

    trades = run_strategy(
        strategy, c1m, c15m, c1h,
        precomputed=precomputed,
        start_date=start_date, end_date=end_date,
        asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
        overnight=overnight,
    )

    # Filter trades to only those on/after start_date BEFORE computing summary
    start_ts = pd.Timestamp(start_date, tz=ET)
    trades = [t for t in trades if t.entry_time and t.entry_time >= start_ts]

    eq = equity_df(trades) if trades else pd.DataFrame()

    n = len(trades)
    pnls = [float(t.pnl_dollars or 0) for t in trades]
    summary = {
        "n_trades": n,
        "total_pnl": sum(pnls),
        "win_rate": (sum(1 for p in pnls if p > 0) / n * 100) if n else 0,
        "strategy_name": strategy.name,
    }

    return trades, eq, summary


# ─── Seed-based search ───────────────────────────────────────────────────────

def run_seed_search(
    cache: DataCache,
    start_date: str,
    end_date: str,
    window_days: int = 30,
    n_refinements: int = 20,
    prior_results: list[dict] | None = None,
    top_n: int = 1,
    failed_archetypes: list[str] | None = None,
) -> list[SearchResult]:
    """Evaluate all strategy seeds, refine top performers via LLM.

    All backtests are parallelized across available CPUs.

    1. Quick-eval all seeds on the TRAIN split (first 2/3) — PARALLEL
    2. Take top 2 with fitness > 0
    3. LLM refines each into n_refinements parameter variants
    4. Backtest all refinements on TRAIN split — PARALLEL
    5. Validate ALL candidates on the VALIDATION split (last 1/3) — PARALLEL
    6. Reject OOS losers, penalize failed archetypes
    7. If no candidates pass validation, use best train-fit seed (no LLM fallback)
    """
    ac = cache.asset_config

    # Train/validation split: 2/3 train, 1/3 validate (no boundary overlap)
    train_days = int(window_days * 2 / 3)
    train_start = start_date
    train_end_ts = pd.Timestamp(start_date) + pd.Timedelta(days=train_days)
    train_end = train_end_ts.strftime("%Y-%m-%d")
    val_start_ts = train_end_ts + pd.Timedelta(days=1)
    val_start = val_start_ts.strftime("%Y-%m-%d")
    val_end = end_date
    val_days = max(1, window_days - train_days - 1)

    print(f"  Train/validate split: {train_start}→{train_end} (train) | "
          f"{val_start}→{val_end} (validate)")

    c1m, c15m, c1h = cache.slice_window(start_date, end_date)
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    precomputed = (snaps_15m, ts_15m_idx)

    if failed_archetypes is None:
        failed_archetypes = []

    # 1. Quick-eval all seeds on TRAIN split — PARALLEL
    # Scale seeds for non-MES assets
    if ac.ticker != "MES":
        seed_dicts = scale_seeds_for_asset(ac)
        seeds = [StrategyDefinition.from_dict(s) for s in seed_dicts]
    else:
        seeds = get_seeds()
        seed_dicts = STRATEGY_SEEDS
    _t_phase = tm.time()
    print(f"  Evaluating {len(seeds)} seeds in parallel...")

    seed_evals = _parallel_eval_strategies(
        seeds, c1m, c15m, c1h, precomputed, train_days,
        train_start, train_end, asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
    )

    results = []
    for idx, fitness, summary in seed_evals:
        sr = SearchResult(rank=0, params=ParamSet(), fitness=fitness, summary=summary)
        sr.summary["strategy_def"] = seeds[idx].to_dict()
        sr.summary["archetype"] = get_seed_archetype(seed_dicts[idx])
        results.append(sr)
        print(f"    {seeds[idx].name}: "
              f"${fitness.total_pnl:>8,.0f} | {fitness.win_rate:>5.1f}% WR | "
              f"{fitness.n_trades} trades | fit={fitness.fitness:.3f}")

    print(f"  [TIMER] Seed eval: {tm.time() - _t_phase:.1f}s")

    if not results:
        print("  All seeds failed. Using best raw seed.")
        # Return empty — caller will handle
        return []

    results.sort(key=lambda r: r.fitness.fitness, reverse=True)

    # 2. Take top 2 with positive fitness
    top_seeds = [r for r in results if r.fitness.fitness > 0][:2]
    if not top_seeds:
        # No positive fitness seeds — return best anyway (no LLM fallback)
        print("  No seeds with positive fitness. Using least-bad seed.")
        results[0].rank = 1
        return results[:top_n]

    print(f"\n  Top {len(top_seeds)} seeds → LLM tuning ({n_refinements} variants each)...")

    # 3. Build market context + generate all refinements (LLM calls)
    context = build_market_context(c1m, start_date, end_date, asset_config=ac)

    _t_phase = tm.time()
    all_variants = []  # (StrategyDefinition, archetype) pairs

    # Build regime context for LLM prompt enrichment
    _regime_ctx = {}
    if cache.gex_cache and start_date:
        try:
            from datetime import date as _date, time as _time
            _opt_d = _date.fromisoformat(start_date)
            _snap = cache.gex_cache.get(_opt_d, _time(10, 0))
            if _snap:
                _regime_ctx["gex_regime"] = _snap.gex_regime
        except Exception:
            pass
    # Top 3 archetypes by fitness from current window
    _arch_fitness = {}
    for r in results:
        a = r.summary.get("archetype", "unknown")
        af = r.summary.get("adjusted_fitness", r.fitness.fitness)
        if a not in _arch_fitness or af > _arch_fitness[a]:
            _arch_fitness[a] = af
    _regime_ctx["top_archetypes"] = sorted(
        [{"archetype": a, "fitness": f} for a, f in _arch_fitness.items()],
        key=lambda x: x["fitness"], reverse=True
    )[:3]

    def _refine_one_seed(sr):
        strat_dict = sr.summary.get("strategy_def", {})
        archetype = sr.summary.get("archetype", "unknown")
        strat = StrategyDefinition.from_dict(strat_dict)
        # Compute MFE/MAE context for LLM prompt enrichment
        mfe_mae_ctx = compute_regime_mfe_mae_context(results, archetype,
                                                      uncensored_records=_UNCENSORED_RECORDS)
        _t_llm = tm.time()
        variants = refine_seed(
            strat, sr.fitness.to_dict(), context,
            archetype=archetype,
            n_variants=n_refinements,
            prior_results=prior_results,
            asset_config=ac,
            mfe_mae_context=mfe_mae_ctx,
            regime_context=_regime_ctx,
        )
        print(f"    [TIMER] LLM refine '{strat.name}': {tm.time() - _t_llm:.1f}s → {len(variants)} variants"
              f" (MFE/MAE ctx: {'yes' if mfe_mae_ctx else 'no'})")
        return [(var, archetype) for var in variants]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(top_seeds)) as pool:
        futures = [pool.submit(_refine_one_seed, sr) for sr in top_seeds]
        for fut in futures:
            all_variants.extend(fut.result())

    print(f"  [TIMER] All LLM refinements: {tm.time() - _t_phase:.1f}s total")

    if not all_variants:
        print("  WARNING: All LLM refinements returned empty! "
              "Check VLLM_BASE_URL / API connectivity.")
        print("  Falling back to raw seed results only.")

    _t_phase = tm.time()
    print(f"  Backtesting {len(all_variants)} refinements in parallel...")

    # 4. Backtest all refinements on TRAIN split — PARALLEL
    variant_strats = [v[0] for v in all_variants]
    ref_evals = _parallel_eval_strategies(
        variant_strats, c1m, c15m, c1h, precomputed, train_days,
        train_start, train_end, asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
    )

    for idx, fitness, summary in ref_evals:
        ref_sr = SearchResult(rank=0, params=ParamSet(), fitness=fitness, summary=summary)
        ref_sr.summary["strategy_def"] = all_variants[idx][0].to_dict()
        ref_sr.summary["archetype"] = all_variants[idx][1]
        results.append(ref_sr)
        print(f"      {all_variants[idx][0].name} → fit={fitness.fitness:.3f}, "
              f"${fitness.total_pnl:,.0f}")

    print(f"  [TIMER] Refinement backtests: {tm.time() - _t_phase:.1f}s")

    # Sort all candidates by train fitness
    results.sort(key=lambda r: r.fitness.fitness, reverse=True)

    # 5. Validate ALL candidates on holdout split — PARALLEL
    val_strats = []
    val_map = []  # index into results
    for i, r in enumerate(results):
        strat_dict = r.summary.get("strategy_def", {})
        try:
            val_strats.append(StrategyDefinition.from_dict(strat_dict))
            val_map.append(i)
        except Exception:
            continue

    _t_phase = tm.time()
    print(f"\n  Validating {len(val_strats)} candidates on holdout in parallel...")
    val_evals = _parallel_eval_strategies(
        val_strats, c1m, c15m, c1h, precomputed, val_days,
        val_start, val_end, asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
    )

    # Build validation lookup: results_idx → val_fitness
    val_lookup = {}
    for idx, val_fitness, _ in val_evals:
        results_idx = val_map[idx]
        val_lookup[results_idx] = val_fitness

    print(f"  [TIMER] Validation backtests: {tm.time() - _t_phase:.1f}s")

    # 6. Score candidates: reject OOS losers, penalize failed archetypes
    validated = []
    for i, r in enumerate(results):
        val_fitness = val_lookup.get(i)
        if val_fitness is None:
            continue
        archetype = r.summary.get("archetype", "unknown")
        strat_name = r.summary.get("strategy_name", "?")

        if val_fitness.total_pnl <= 0:
            print(f"    REJECTED (OOS): {strat_name} [{archetype}] "
                  f"val P&L=${val_fitness.total_pnl:,.0f}")
            continue

        # Hard gate: minimum val trades (4 for overnight, 2 for RTH)
        _min_val_trades = 4 if getattr(_mc, '_overnight', False) else 2
        if val_fitness.n_trades < _min_val_trades:
            print(f"    REJECTED (min trades): {strat_name} [{archetype}] "
                  f"val trades={val_fitness.n_trades} < {_min_val_trades}")
            continue

        # Expectancy gate: avg P&L per trade must be positive
        val_expectancy = val_fitness.total_pnl / val_fitness.n_trades
        if val_expectancy < 10:  # at least $10/trade avg to filter noise
            print(f"    REJECTED (expectancy): {strat_name} [{archetype}] "
                  f"val expectancy=${val_expectancy:,.1f}/trade")
            continue

        # Profit factor gate: val PF must show real edge
        if val_fitness.profit_factor < 1.1:
            print(f"    REJECTED (PF): {strat_name} [{archetype}] "
                  f"val PF={val_fitness.profit_factor:.2f} < 1.1")
            continue

        archetype_penalty = 0.0
        fail_count = failed_archetypes.count(archetype)
        if fail_count > 0:
            archetype_penalty = fail_count * 0.5

        r.summary["val_pnl"] = val_fitness.total_pnl
        r.summary["val_fitness"] = val_fitness.fitness
        r.summary["adjusted_fitness"] = (
            r.fitness.fitness + val_fitness.fitness * 0.3 - archetype_penalty
        )
        validated.append(r)
        print(f"    PASSED: {strat_name} [{archetype}] "
              f"train={r.fitness.fitness:.3f} val=${val_fitness.total_pnl:,.0f} "
              f"adj={r.summary['adjusted_fitness']:.3f}")

    # 7. If nothing passed validation, return empty — skip this regime
    if not validated:
        print("  No candidates passed validation. Skipping regime (no unvalidated fallback).")
        _log_seed_candidates(results, [], start_date, end_date)
        return []

    validated.sort(key=lambda r: r.summary.get("adjusted_fitness", 0), reverse=True)
    for i, r in enumerate(validated):
        r.rank = i + 1

    _log_seed_candidates(results, validated, start_date, end_date)
    # Attach candidates for pipeline transport
    _attach_candidates_to_winner(results, validated, validated[0])
    return validated[:top_n]


def _attach_candidates_to_winner(
    all_results: list[SearchResult],
    validated: list[SearchResult],
    winner: SearchResult,
) -> None:
    """Attach top-50 candidate dicts to winner's summary for pipeline transport."""
    validated_set = {id(r) for r in validated}
    top50 = []
    seen = set()
    for r in all_results[:50]:
        sd = r.summary.get("strategy_def", {})
        exit_d = sd.get("exit", {})
        top50.append({
            "rank": r.rank,
            "archetype": r.summary.get("archetype", "unknown"),
            "strategy_name": r.summary.get("strategy_name", "?"),
            "passed_validation": id(r) in validated_set,
            "train_fitness": r.fitness.fitness,
            "train_sharpe": r.fitness.sharpe,
            "train_pf": r.fitness.profit_factor,
            "train_pnl": r.fitness.total_pnl,
            "train_n_trades": r.fitness.n_trades,
            "train_wr": r.fitness.win_rate,
            "val_pnl": r.summary.get("val_pnl"),
            "adjusted_fitness": r.summary.get("adjusted_fitness"),
            "has_tsl_be": bool(exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts")),
            "exit_type": ("tsl_be" if (exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts"))
                          else "fixed_sl_tp"),
            "strategy_def": sd,
        })
        seen.add(id(r))
    for r in validated:
        if id(r) not in seen:
            sd = r.summary.get("strategy_def", {})
            exit_d = sd.get("exit", {})
            top50.append({
                "rank": r.rank,
                "archetype": r.summary.get("archetype", "unknown"),
                "strategy_name": r.summary.get("strategy_name", "?"),
                "passed_validation": True,
                "train_fitness": r.fitness.fitness,
                "train_sharpe": r.fitness.sharpe,
                "train_pf": r.fitness.profit_factor,
                "train_pnl": r.fitness.total_pnl,
                "train_n_trades": r.fitness.n_trades,
                "train_wr": r.fitness.win_rate,
                "val_pnl": r.summary.get("val_pnl"),
                "adjusted_fitness": r.summary.get("adjusted_fitness"),
                "has_tsl_be": bool(exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts")),
                "exit_type": ("tsl_be" if (exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts"))
                              else "fixed_sl_tp"),
                "strategy_def": sd,
            })
    winner.summary["_candidates_top50"] = top50


def compute_regime_mfe_mae_context(
    all_results: list,
    archetype: str,
    uncensored_records: list = None,
) -> dict:
    """Compute MFE/MAE context for LLM prompt injection.

    Prefers uncensored distributions when available (more accurate).
    Falls back to censored candidate data.
    """
    # --- Uncensored path (preferred) ---
    if uncensored_records:
        arch_unc = [r for r in uncensored_records
                    if r.get("archetype") == archetype
                    and r.get("mfe_uncensored") and r.get("mae_uncensored")]

        if len(arch_unc) >= 5:
            mfes = sorted([r["mfe_uncensored"] for r in arch_unc])
            maes = sorted([r["mae_uncensored"] for r in arch_unc])
            n = len(arch_unc)

            ctx = {
                "source": "uncensored",
                "n_candidates": n,
                "mfe_p25": round(mfes[n // 4], 1),
                "mfe_p50": round(mfes[n // 2], 1),
                "mfe_p75": round(mfes[3 * n // 4], 1),
                "mae_p50": round(maes[n // 2], 1),
                "mae_p75": round(maes[3 * n // 4], 1),
                "mfe_before_mae_pct": None,
                "bars_to_mfe_median": None,
                "bars_to_mae_median": None,
            }
            print(f"    [MFE/MAE ctx] archetype='{archetype}': source=uncensored, n={n}, "
                  f"mfe_p50={ctx['mfe_p50']}pts, mae_p75={ctx['mae_p75']}pts")
            return ctx

    # --- Censored fallback ---
    arch_candidates = [
        r.summary for r in all_results
        if r.summary.get("archetype", getattr(r, "archetype", "")) == archetype
        or getattr(r.summary.get("strategy_def", {}), "archetype", "") == archetype
    ]
    if not arch_candidates:
        arch_candidates = [
            r.summary for r in all_results
            if r.summary.get("strategy_def", {}).get("archetype") == archetype
        ]
    n_arch = len(arch_candidates)
    arch_candidates = [c for c in arch_candidates
                       if c.get("mfe_p50") and c.get("mae_p50")]
    n_with_data = len(arch_candidates)
    if n_with_data < 3:
        print(f"    [MFE/MAE ctx] archetype='{archetype}': {n_arch} arch matches, "
              f"{n_with_data} have mfe/mae data (need >=3) → skipped")
        return {}

    def _med(lst):
        s = sorted(lst)
        return s[len(s) // 2] if s else None

    mfe_p50s = sorted([c["mfe_p50"] for c in arch_candidates])
    mae_p50s = sorted([c["mae_p50"] for c in arch_candidates])
    mae_p75s = sorted([c.get("mae_p75", 0) for c in arch_candidates if c.get("mae_p75")])
    mfbm = [c["mfe_before_mae_pct"] for c in arch_candidates
            if c.get("mfe_before_mae_pct") is not None]
    btmfe = [c["bars_to_mfe_median"] for c in arch_candidates
             if c.get("bars_to_mfe_median")]
    btmae = [c["bars_to_mae_median"] for c in arch_candidates
             if c.get("bars_to_mae_median")]
    n = len(arch_candidates)

    ctx = {
        "source": "censored",
        "n_candidates": n,
        "mfe_p25": round(mfe_p50s[n // 4], 1) if n > 4 else None,
        "mfe_p50": round(mfe_p50s[n // 2], 1),
        "mfe_p75": round(mfe_p50s[3 * n // 4], 1) if n > 4 else None,
        "mae_p50": round(mae_p50s[n // 2], 1),
        "mae_p75": round(_med(mae_p75s), 1) if mae_p75s else None,
        "mfe_before_mae_pct": round(sum(mfbm) / len(mfbm) * 100) if mfbm else None,
        "bars_to_mfe_median": round(_med(btmfe)) if btmfe else None,
        "bars_to_mae_median": round(_med(btmae)) if btmae else None,
    }
    print(f"    [MFE/MAE ctx] archetype='{archetype}': source=censored, n={ctx['n_candidates']}, "
          f"mfe_p50={ctx['mfe_p50']}pts, mae_p75={ctx['mae_p75']}pts")
    return ctx


def _log_seed_candidates(
    all_results: list[SearchResult],
    validated: list[SearchResult],
    start_date: str,
    end_date: str,
) -> None:
    """Append seed search candidates (winners + losers) to candidate_logs.jsonl."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    validated_set = {id(r) for r in validated}

    entries = []
    for r in all_results:
        sd = r.summary.get("strategy_def", {})
        exit_d = sd.get("exit", {})
        pnls = r.summary.get("pnls_array", [])
        # Trajectory fields
        traj = {}
        if pnls and len(pnls) >= 6:
            n3 = max(len(pnls) // 3, 1)
            early_pnl = round(sum(pnls[:n3]), 2)
            late_pnl = round(sum(pnls[2 * n3:]), 2)
            activity = abs(early_pnl) + abs(late_pnl) + 1.0
            trajectory = (late_pnl - early_pnl) / activity
            traj = {
                "early_pnl": early_pnl,
                "late_pnl": late_pnl,
                "trajectory_score": round(max(trajectory, 0.0), 4),
                "frontloaded": bool(early_pnl > late_pnl and early_pnl > 0),
            }
        else:
            traj = {"early_pnl": None, "late_pnl": None,
                    "trajectory_score": None, "frontloaded": None}
        rob = r.summary.get("robustness", {}) or {}
        entries.append({
            "rank": r.rank,
            "archetype": r.summary.get("archetype", "unknown"),
            "strategy_name": r.summary.get("strategy_name", "?"),
            "passed_validation": id(r) in validated_set,
            "train_fitness": r.fitness.fitness,
            "train_sharpe": r.fitness.sharpe,
            "train_pf": r.fitness.profit_factor,
            "train_pnl": r.fitness.total_pnl,
            "train_max_dd": r.fitness.max_dd,
            "train_wr": r.fitness.win_rate,
            "train_n_trades": r.fitness.n_trades,
            "train_avg_pnl": r.fitness.avg_trade_pnl,
            "val_pnl": r.summary.get("val_pnl"),
            "val_fitness": r.summary.get("val_fitness"),
            "adjusted_fitness": r.summary.get("adjusted_fitness"),
            "has_tsl_be": bool(exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts")),
            "exit_type": ("tsl_be" if (exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts"))
                          else "fixed_sl_tp"),
            "pnls_array": pnls,
            "mfe_array": r.summary.get("mfe_array", []),
            "mae_array": r.summary.get("mae_array", []),
            "mfe_p50": r.summary.get("mfe_p50"),
            "mfe_p75": r.summary.get("mfe_p75"),
            "mae_p50": r.summary.get("mae_p50"),
            "mae_p75": r.summary.get("mae_p75"),
            "mfe_mae_ratio": r.summary.get("mfe_mae_ratio"),
            "bars_to_mfe_median": r.summary.get("bars_to_mfe_median"),
            "bars_to_mae_median": r.summary.get("bars_to_mae_median"),
            "mfe_before_mae_pct": r.summary.get("mfe_before_mae_pct"),
            "mfe_before_mae_raw": r.summary.get("mfe_before_mae_raw", []),
            **traj,
            # Robustness metrics (previously missing from candidate_logs)
            "win_loss_ratio": rob.get("win_loss_ratio"),
            "expectancy_per_trade": rob.get("expectancy_per_trade"),
            "kelly_fraction": rob.get("kelly_fraction"),
            "trades_per_day": rob.get("trades_per_day"),
            "mc_p_value": rob.get("mc_p_value"),
            "ttest_p_value": rob.get("ttest_p_value"),
            "pf_stability_ratio": rob.get("pf_stability_ratio"),
            "top_trade_pct": rob.get("top_trade_pct"),
            "remove_best_still_positive": rob.get("remove_best_still_positive"),
            "wr_stability_ratio": rob.get("wr_stability_ratio"),
            "payoff_consistency": rob.get("payoff_consistency"),
            "strategy_def": sd,
        })

    jsonl_path = OUTPUT_DIR / "candidate_logs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(jsonl_path, "a") as f:
            for entry in entries:
                record = {
                    "log_timestamp": ts,
                    "optimize_start": start_date,
                    "search_mode": "seed",
                    **entry,
                }
                f.write(json.dumps(record, default=str) + "\n")
        print(f"  [LOG] Appended {len(entries)} candidates to candidate_logs.jsonl")
    except Exception as e:
        print(f"  [LOG] Failed to append candidates: {e}")


# ─── Mid-run strategy evolution ──────────────────────────────────────────────

def _trigger_strength(trigger_type: str, observed_value: float, baseline_value: float) -> float:
    """Normalize trigger signal to 0.0-1.0 strength.

    Higher = more deviation from baseline = stronger signal for evolution.
    """
    if baseline_value == 0:
        return 0.0
    if trigger_type in ('low_wr', 'negative_pnl'):
        deviation = max(baseline_value - observed_value, 0.0) / baseline_value
    elif trigger_type in ('avg_loss_too_big', 'high_dd'):
        deviation = max(observed_value - baseline_value, 0.0) / baseline_value
    else:
        deviation = abs(observed_value - baseline_value) / baseline_value
    return min(deviation, 1.0)


def _evolution_confidence(
    trigger_signal_strength: float,
    n_trades_observed: int,
    n_prior_evolutions: int,
    current_dd: float,
    max_dd: float,
    meta: MetaConfig,
) -> tuple[float, float]:
    """Returns (confidence, required_threshold).

    Evolution fires only if confidence >= required_threshold.

    trigger_signal_strength: 0.0-1.0, how far outside normal the trigger signal is
    n_trades_observed: trades seen since last evolution (or since start)
    n_prior_evolutions: total evolutions fired in this regime's lifetime
    current_dd: current drawdown in dollars (positive number)
    max_dd: max DD threshold in dollars
    """
    # Observation confidence — need enough trades to trust the signal
    observation_confidence = min(n_trades_observed / (meta.evolution_min_trades * 2.0), 1.0)

    # Raw confidence = signal strength × observation backing
    raw_confidence = trigger_signal_strength * observation_confidence

    # Required threshold rises with each prior evolution
    base_required = meta.evolution_base_threshold + (n_prior_evolutions * meta.evolution_cost_per)

    # Stress adjustment — strategy near death gets lower threshold
    stress_level = min(current_dd / max_dd, 1.0) if max_dd > 0 else 0.0
    stress_adjustment = (0.5 - stress_level) * meta.evolution_stress_weight

    required_threshold = base_required + stress_adjustment

    return raw_confidence, required_threshold


def evolve_strategy(
    strategy: StrategyDefinition,
    recent_trades: list,
    train_wr: float = 0.50,
    train_avg_loss: float = 50.0,
    n_prior_evolutions: int = 0,
    trades_since_last_evolution: int = 0,
    current_dd: float = 0.0,
    max_dd: float = 3000.0,
    meta: MetaConfig | None = None,
) -> tuple[StrategyDefinition, dict | None]:
    """Deterministic micro-mutation gated by evolution confidence system.

    Analyzes recent trades, detects triggers, computes confidence vs threshold.
    Only fires evolution if confidence >= required_threshold.

    Returns (strategy, evolution_record_or_None).
    evolution_record contains type, confidence, required, fired, etc.
    """
    _m = meta or MetaConfig()

    if len(recent_trades) < _m.evolution_min_trades:
        return strategy, None

    pnls = [float(t.pnl_dollars or 0) for t in recent_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    wr = len(wins) / n if n else 0.5
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    avg_pnl = sum(pnls) / n

    # Detect triggers and compute signal strength
    triggers = []

    # Rule 1: avg loss too big → tighten SL
    if avg_loss > 0 and avg_win > 0 and avg_loss > 1.5 * avg_win:
        strength = _trigger_strength('avg_loss_too_big', avg_loss, avg_win)
        triggers.append(('avg_loss_too_big', strength, avg_loss, avg_win))

    # Rule 2: low win rate → widen time window
    if wr < 0.35:
        strength = _trigger_strength('low_wr', wr, train_wr)
        triggers.append(('low_wr', strength, wr, train_wr))

    # Rule 3: high win rate → tighten SL slightly
    if wr > 0.65:
        strength = _trigger_strength('high_wr', wr, train_wr)
        triggers.append(('high_wr', strength, wr, train_wr))

    # Rule 4: negative avg P&L → shift time window later
    if avg_pnl < -100:
        strength = _trigger_strength('negative_pnl', avg_pnl, 0.0)
        # Use absolute scale since baseline is 0
        strength = min(abs(avg_pnl) / 200.0, 1.0)
        triggers.append(('negative_pnl', strength, avg_pnl, 0.0))

    if not triggers:
        return strategy, None

    # Use strongest trigger
    best_trigger = max(triggers, key=lambda t: t[1])
    evolution_reason, trigger_strength, observed, baseline = best_trigger

    confidence, required = _evolution_confidence(
        trigger_signal_strength=trigger_strength,
        n_trades_observed=trades_since_last_evolution,
        n_prior_evolutions=n_prior_evolutions,
        current_dd=current_dd,
        max_dd=max_dd,
        meta=_m,
    )

    stress_level = min(current_dd / max_dd, 1.0) if max_dd > 0 else 0.0

    evolution_record = {
        'type': evolution_reason,
        'confidence': round(confidence, 3),
        'required': round(required, 3),
        'fired': confidence >= required,
        'trades_observed': trades_since_last_evolution,
        'stress_level': round(stress_level, 3),
        'trigger_strength': round(trigger_strength, 3),
        'observed': round(observed, 3),
        'baseline': round(baseline, 3),
    }

    if confidence < required:
        print(f"    Evolution SKIPPED: {evolution_reason} "
              f"(confidence={confidence:.2f} < required={required:.2f}, "
              f"prior_evolutions={n_prior_evolutions})")
        return strategy, evolution_record

    # Fire the evolution
    import copy
    new_def = copy.deepcopy(strategy.to_dict())

    sl_cfg = new_def.get("exit", {}).get("stop_loss", {})
    tw = new_def.get("entry", {}).get("time_window", {})

    if evolution_reason == 'avg_loss_too_big':
        if sl_cfg.get("type") == "fixed_pts":
            old_val = sl_cfg.get("value", 12)
            new_val = max(4, old_val - 2)
            if new_val != old_val:
                sl_cfg["value"] = new_val
                print(f"    Evolution FIRED: SL {old_val} → {new_val} "
                      f"(confidence={confidence:.2f} >= {required:.2f})")
            else:
                return strategy, evolution_record

    elif evolution_reason == 'low_wr':
        if tw:
            end_h, end_m = map(int, tw.get("end", "11:00").split(":"))
            end_total = end_h * 60 + end_m + 30
            if end_total <= 15 * 60 + 45:
                tw["end"] = f"{end_total // 60:02d}:{end_total % 60:02d}"
                print(f"    Evolution FIRED: time window end → {tw['end']} "
                      f"(confidence={confidence:.2f} >= {required:.2f})")
            else:
                return strategy, evolution_record
        else:
            return strategy, evolution_record

    elif evolution_reason == 'high_wr':
        if sl_cfg.get("type") == "fixed_pts":
            old_val = sl_cfg.get("value", 12)
            new_val = max(4, old_val - 1)
            if new_val != old_val:
                sl_cfg["value"] = new_val
                print(f"    Evolution FIRED: SL {old_val} → {new_val} "
                      f"(confidence={confidence:.2f} >= {required:.2f})")
            else:
                return strategy, evolution_record

    elif evolution_reason == 'negative_pnl':
        if tw:
            start_h, start_m = map(int, tw.get("start", "09:30").split(":"))
            start_total = start_h * 60 + start_m + 15
            if start_total <= 11 * 60:
                tw["start"] = f"{start_total // 60:02d}:{start_total % 60:02d}"
                print(f"    Evolution FIRED: time window start → {tw['start']} "
                      f"(confidence={confidence:.2f} >= {required:.2f})")
            else:
                return strategy, evolution_record
        else:
            return strategy, evolution_record

    return StrategyDefinition.from_dict(new_def), evolution_record


def run_forward_test_chunked(
    cache: DataCache,
    strategy: StrategyDefinition,
    start_date: str,
    end_date: str,
    evolution_interval_days: int = 14,
    min_trades_for_evolution: int = 10,
    train_wr: float = 0.50,
    train_avg_loss: float = 50.0,
    meta: MetaConfig | None = None,
    overnight: bool = False,
) -> tuple[list, pd.DataFrame, dict, int]:
    """Forward-test with periodic mid-run evolution gated by confidence system.

    Runs the strategy in 2-week chunks. At each checkpoint, evaluates
    evolution triggers and fires only if confidence exceeds threshold.

    Returns (trades, equity_df, summary, n_evolutions).
    """
    _m = meta or MetaConfig()
    current_strategy = strategy
    all_trades = []
    n_evolutions = 0
    trades_since_last_evolution = 0
    evolution_history = []

    chunk_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)

    while chunk_start < final_end:
        chunk_end = min(chunk_start + pd.Timedelta(days=evolution_interval_days),
                        final_end)

        chunk_start_str = chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        # Run strategy on this chunk
        trades, eq, summary = run_forward_test_llm(
            cache, current_strategy, chunk_start_str, chunk_end_str,
            overnight=overnight,
        )
        all_trades.extend(trades)
        trades_since_last_evolution += len(trades)

        # Evolution checkpoint
        if (len(all_trades) >= min_trades_for_evolution
                and chunk_end < final_end):
            recent = all_trades[-min_trades_for_evolution:]

            # Compute current drawdown for stress assessment
            all_pnls = [float(t.pnl_dollars or 0) for t in all_trades]
            cum = np.cumsum(all_pnls)
            running_max = np.maximum.accumulate(cum)
            current_dd = float((running_max[-1] - cum[-1])) if len(cum) else 0.0

            evolved, ev_record = evolve_strategy(
                current_strategy, recent,
                train_wr=train_wr,
                train_avg_loss=train_avg_loss,
                n_prior_evolutions=n_evolutions,
                trades_since_last_evolution=trades_since_last_evolution,
                current_dd=current_dd,
                max_dd=_m.max_dd_dollars,
                meta=_m,
            )
            if ev_record:
                evolution_history.append(ev_record)
            if evolved is not current_strategy:
                n_evolutions += 1
                trades_since_last_evolution = 0
                current_strategy = evolved

        chunk_start = chunk_end

    # Build aggregate equity
    eq = equity_df(all_trades) if all_trades else pd.DataFrame()
    n = len(all_trades)
    pnls = [float(t.pnl_dollars or 0) for t in all_trades]
    summary = {
        "n_trades": n,
        "total_pnl": sum(pnls),
        "win_rate": (sum(1 for p in pnls if p > 0) / n * 100) if n else 0,
        "strategy_name": strategy.name,
        "n_evolutions": n_evolutions,
        "evolution_log": evolution_history,
    }

    return all_trades, eq, summary, n_evolutions


# ─── Walk-forward regime backtest ────────────────────────────────────────────

def trades_to_regime_dicts(trades: list) -> list[dict]:
    """Convert mb.Trade objects to dicts compatible with regime_monitor."""
    out = []
    for t in trades:
        ts = t.exit_time if t.exit_time else t.entry_time
        d = {
            "date": ts.strftime("%Y-%m-%d") if ts else "",
            "exit_time": str(ts) if ts else "",
            "pnl_dollars": float(t.pnl_dollars or 0),
            "result": t.result,
        }
        out.append(d)
    return out


@dataclass
class RegimeRecord:
    """One optimization-then-trade regime period."""
    regime_id: int
    optimize_start: str
    optimize_end: str
    forward_start: str
    forward_end: str
    death_reason: str
    winner_params: dict
    winner_fitness: float
    # Forward test results
    forward_trades: int
    forward_pnl: float
    forward_win_rate: float
    forward_max_dd: float
    forward_days: int
    optimization_time_sec: float
    # Fitness component breakdown (for meta-optimizer weight tuning)
    train_sharpe: float = 0.0
    train_pf: float = 0.0
    train_n_trades: int = 0
    # Robustness metrics (observational only — no scoring impact)
    train_robustness: Optional[dict] = None    # metrics on training trades
    forward_robustness: Optional[dict] = None  # metrics on forward trades
    # Top-50 candidates from search (for meta-analysis / negative class data)
    candidates_top50: Optional[list] = None
    # Bayesian kill tracking
    kill_posterior: Optional[float] = None
    # Death context — market conditions when regime died
    death_gex_regime: Optional[str] = None     # bullish/bearish/neutral at death
    death_rv: Optional[float] = None           # realized vol at death
    # Optimization context — market conditions at optimization start (deployment-time feature)
    optimize_gex_regime: Optional[str] = None  # bullish/bearish/neutral at optimize start
    optimize_rv: Optional[float] = None        # realized vol at optimize start
    # Market snapshot at deployment (forward_start) — for macro regime gating
    market_snapshot: Optional[dict] = None
    # Evolution tracking
    n_evolutions: int = 0
    # LLM backend tracking
    llm_backend: str = "anthropic_claude"
    llm_model: str = "claude-sonnet-4-20250514"

    def to_dict(self) -> dict:
        return asdict(self)


def compute_market_snapshot(
    candles_1m: list,
    snapshot_date: str,
    asset_config: AssetConfig | None = None,
) -> dict:
    """Compute market conditions at a point in time for macro regime gating.

    Uses the 20 trading days BEFORE snapshot_date to characterize:
    - Volatility (realized vol from daily ranges)
    - Overnight range (avg ON range, normalized)
    - Trend (10d/20d returns)
    - Range regime (5d vs 20d range ratio — compression/expansion)
    - Gap frequency (% of days with significant gaps)
    - Directional consistency (% of days closing in majority direction)

    All computed from 1-minute candles — no external data needed.
    """
    from datetime import time as dtime
    snap_ts = pd.Timestamp(snapshot_date)
    if hasattr(candles_1m[0].timestamp, 'tz') and candles_1m[0].timestamp.tz:
        snap_ts = snap_ts.tz_localize(candles_1m[0].timestamp.tz)

    # Get RTH bars in the 30 calendar days before snapshot
    lookback_start = snap_ts - pd.Timedelta(days=40)
    window = [c for c in candles_1m
              if lookback_start <= c.timestamp < snap_ts
              and dtime(9, 30) <= c.timestamp.time() < dtime(16, 0)]

    if not window:
        return {"error": "no_data"}

    # Group by trading day
    by_day = {}
    for c in window:
        d = c.timestamp.normalize()
        by_day.setdefault(d, []).append(c)

    trading_days = sorted(by_day.keys())
    if len(trading_days) < 5:
        return {"error": "insufficient_days", "n_days": len(trading_days)}

    # Take last 20 trading days
    trading_days = trading_days[-20:]

    # Daily OHLC
    daily_highs = []
    daily_lows = []
    daily_opens = []
    daily_closes = []
    daily_ranges = []
    for d in trading_days:
        bars = by_day[d]
        h = max(c.high for c in bars)
        l = min(c.low for c in bars)
        daily_highs.append(h)
        daily_lows.append(l)
        daily_opens.append(bars[0].open)
        daily_closes.append(bars[-1].close)
        daily_ranges.append(h - l)

    n_days = len(trading_days)
    price_level = (daily_opens[0] + daily_closes[-1]) / 2

    # --- Realized volatility (daily range as % of price) ---
    avg_range_20d = sum(daily_ranges) / n_days
    vol_pct = avg_range_20d / price_level * 100

    # --- Overnight ranges ---
    ts_index = [c.timestamp for c in candles_1m]
    use_prev_rth = asset_config and asset_config.overnight_session == "prev_rth"
    on_ranges = []
    for d in trading_days:
        if use_prev_rth:
            from equity_data import get_prev_day_high_low
            on_h, on_l = get_prev_day_high_low(candles_1m, d, ts_index)
        else:
            on_h, on_l = mb.get_overnight_high_low(candles_1m, d, ts_index)
        if on_h is not None and on_l is not None and on_h > on_l:
            on_ranges.append(on_h - on_l)
    avg_on_range = sum(on_ranges) / len(on_ranges) if on_ranges else 0
    on_range_pct = avg_on_range / price_level * 100

    # --- Trend (10d and 20d) ---
    ret_20d = (daily_closes[-1] - daily_opens[0]) / daily_opens[0] * 100
    mid = max(n_days // 2, 1)
    ret_10d = (daily_closes[-1] - daily_opens[mid]) / daily_opens[mid] * 100

    # --- Range regime (5d vs 20d — compression/expansion) ---
    avg_range_5d = sum(daily_ranges[-5:]) / min(5, len(daily_ranges[-5:]))
    range_ratio = avg_range_5d / avg_range_20d if avg_range_20d > 0 else 1.0

    # --- Gap frequency ---
    gap_threshold = 3.0 if not asset_config else asset_config.default_sl / 4
    n_gap_up = 0
    n_gap_down = 0
    for i in range(1, n_days):
        diff = daily_opens[i] - daily_closes[i - 1]
        if diff > gap_threshold:
            n_gap_up += 1
        elif diff < -gap_threshold:
            n_gap_down += 1
    gap_pct = (n_gap_up + n_gap_down) / max(n_days - 1, 1) * 100

    # --- Directional consistency ---
    up_days = sum(1 for i in range(n_days) if daily_closes[i] > daily_opens[i])
    down_days = n_days - up_days
    dir_consistency = max(up_days, down_days) / n_days * 100

    # --- Consecutive same-direction days (streak at snapshot) ---
    streak = 1
    last_dir = 1 if daily_closes[-1] > daily_opens[-1] else -1
    for i in range(n_days - 2, -1, -1):
        d_dir = 1 if daily_closes[i] > daily_opens[i] else -1
        if d_dir == last_dir:
            streak += 1
        else:
            break

    # --- Max daily range in last 5 days (event spike detection) ---
    max_range_5d = max(daily_ranges[-5:])
    max_range_5d_pct = max_range_5d / price_level * 100

    return {
        "snapshot_date": snapshot_date,
        "n_days": n_days,
        "price_level": round(price_level, 2),
        # Volatility
        "avg_daily_range_pts": round(avg_range_20d, 2),
        "daily_vol_pct": round(vol_pct, 4),
        "avg_on_range_pts": round(avg_on_range, 2),
        "on_range_pct": round(on_range_pct, 4),
        # Trend
        "ret_10d_pct": round(ret_10d, 4),
        "ret_20d_pct": round(ret_20d, 4),
        # Range regime
        "range_ratio_5d_20d": round(range_ratio, 4),
        "max_range_5d_pct": round(max_range_5d_pct, 4),
        # Gaps & direction
        "gap_pct": round(gap_pct, 2),
        "gap_up_count": n_gap_up,
        "gap_down_count": n_gap_down,
        "dir_consistency_pct": round(dir_consistency, 2),
        "streak_days": streak,
        "streak_direction": "up" if last_dir == 1 else "down",
    }


@dataclass
class WalkForwardSummary:
    """Aggregate results across all regime periods."""
    data_start: str
    data_end: str
    window_days: int
    total_regimes: int
    total_trades: int
    total_pnl: float
    total_forward_days: int
    aggregate_win_rate: float
    aggregate_max_dd: float
    avg_regime_lifespan_days: float
    median_regime_lifespan_days: float
    avg_regime_pnl: float
    total_optimization_time_sec: float
    regimes: list[RegimeRecord] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    meta_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _equity_size_scaler(peak_equity: float, current_equity: float,
                        max_dd: float) -> float:
    """Progressive position size scaler based on equity drawdown.

    Returns a multiplier in [0.25, 1.0]:
    - At peak equity: 1.0 (full size)
    - 25% drawdown from peak: 0.75
    - 50% drawdown from peak: 0.50
    - 75%+ drawdown from peak: 0.25

    The idea: bleed less capital while a regime is dying, rather than
    trading full size all the way to the kill threshold.
    """
    if peak_equity <= 0 or max_dd <= 0:
        return 1.0
    dd = peak_equity - current_equity
    if dd <= 0:
        return 1.0
    dd_pct = dd / max_dd  # fraction of max DD consumed
    # Linear scale: 1.0 at 0% DD → 0.25 at 100% DD
    return max(0.25, 1.0 - 0.75 * dd_pct)


def find_regime_death(
    forward_trades: list,
    forward_start: str,
    max_days: int = 180,
    max_consecutive_losing_days: int = 5,
    max_dd_dollars: float = 3500,
    min_wr_floor: float = 50.0,
    wr_lookback: int = 20,
    min_trades_for_check: int = 20,
    meta: MetaConfig | None = None,
    progressive_sizing: bool = True,
    train_wr: float = 0.0,
) -> tuple[str | None, str, list]:
    """Find the date the regime dies within the forward trades.

    Kill conditions (checked after min_trades_for_check trades):
    - Standard regime_status checks (DD, WR floor, consecutive losing days)
    - Negative P&L trajectory: cumulative P&L < -$500 after 20+ trades
    - Flat regime: |cumulative P&L| < $200 after 25+ trades (not alpha)

    When progressive_sizing=True, trade P&Ls are scaled down as the
    equity curve draws down from peak. This simulates reducing position
    size when the regime starts dying, reducing total bleed.

    Returns (death_date_str, death_reason, trades_before_death).
    If the regime never dies: (None, "alive", all_trades).
    """
    from regime_monitor import regime_status

    if not forward_trades:
        return None, "no_trades", []

    # Group trades by date
    from collections import OrderedDict
    by_date = OrderedDict()
    for t in forward_trades:
        ts = t.exit_time if t.exit_time else t.entry_time
        if not ts:
            continue
        d = ts.strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(t)

    # Walk day-by-day, checking regime status after each day
    accumulated = []
    start_ts = pd.Timestamp(forward_start)

    # Progressive sizing state
    running_equity = 0.0
    peak_equity = 0.0

    for day_str, day_trades in by_date.items():
        # Skip trades before forward start
        day_ts = pd.Timestamp(day_str)
        if day_ts < start_ts:
            continue

        # Apply progressive sizing: scale today's trade P&Ls by equity scaler
        if progressive_sizing and peak_equity > 0:
            scaler = _equity_size_scaler(peak_equity, running_equity, max_dd_dollars)
            if scaler < 1.0:
                import copy
                scaled_trades = []
                for t in day_trades:
                    t_copy = copy.copy(t)
                    if t_copy.pnl_dollars is not None:
                        t_copy.pnl_dollars = round(t_copy.pnl_dollars * scaler, 2)
                    if t_copy.pnl is not None:
                        t_copy.pnl = round(t_copy.pnl * scaler, 4)
                    scaled_trades.append(t_copy)
                day_trades = scaled_trades

        accumulated.extend(day_trades)
        n_acc = len(accumulated)

        # Update equity tracking
        for t in day_trades:
            running_equity += float(t.pnl_dollars or 0)
            peak_equity = max(peak_equity, running_equity)

        # Check if we've exceeded max regime days
        elapsed_days = (day_ts - start_ts).days
        if elapsed_days > max_days:
            return day_str, "max_period", accumulated

        _m = meta or DEFAULT_META_CONFIG

        # DD check from trade 1 — never let a strategy blow past the DD limit
        if n_acc >= 1:
            dd = peak_equity - running_equity
            if dd >= max_dd_dollars:
                return (day_str,
                        f"drawdown ${dd:,.0f} exceeds ${max_dd_dollars:,.0f} limit after {n_acc} trades",
                        accumulated)

        # Early kill: WR below floor after N trades — don't let bad strategies bleed
        _early_min = _m.early_death_min_trades
        _early_wr = _m.early_death_wr_floor
        if _early_min <= n_acc < min_trades_for_check:
            early_wins = sum(1 for t in accumulated if (t.pnl_dollars or 0) > 0)
            early_wr = early_wins / n_acc * 100
            if early_wr < _early_wr:
                return (day_str,
                        f"early kill: WR {early_wr:.0f}% < {_early_wr:.0f}% after {n_acc} trades",
                        accumulated)

        # Hard WR circuit breaker — catches catastrophic WR collapse fast
        # Bayesian kill is elegant but slow when WR collapses to 20-35%
        if (n_acc >= _m.wr_circuit_breaker_min_trades):
            fwd_wins = sum(1 for t in accumulated if (t.pnl_dollars or 0) > 0)
            forward_wr = fwd_wins / n_acc
            if forward_wr < _m.wr_circuit_breaker_threshold:
                return (day_str,
                        f"wr_circuit_breaker: {forward_wr:.1%} WR after {n_acc} trades "
                        f"(threshold: {_m.wr_circuit_breaker_threshold:.0%})",
                        accumulated)

        # Bayesian edge decay kill: posterior probability of real edge too low
        if n_acc >= 5 and train_wr > 0:
            fwd_wins = sum(1 for t in accumulated if (t.pnl_dollars or 0) > 0)
            posterior = bayesian_edge_probability(
                n_wins=fwd_wins,
                n_trades=n_acc,
                assumed_edge_wr=train_wr,
                prior_edge=0.70,  # high deployment prior — passed train + validation
            )
            if posterior < _m.bayesian_kill_threshold:
                return (day_str,
                        f"bayesian_edge_decay: posterior={posterior:.3f} < {_m.bayesian_kill_threshold}",
                        accumulated)

        # Need enough trades before checking regime status
        if n_acc < min_trades_for_check:
            continue

        # Check regime health (standard checks)
        trade_dicts = trades_to_regime_dicts(accumulated)
        status = regime_status(
            trade_dicts,
            max_consecutive_losing_days=max_consecutive_losing_days,
            max_dd_dollars=max_dd_dollars,
            min_wr_floor=min_wr_floor,
            wr_lookback=wr_lookback,
        )

        if status["kill"]:
            reason = "; ".join(status["reasons"])
            return day_str, reason, accumulated

        # Negative trajectory kill: losing money after enough trades
        if n_acc >= _m.negative_trajectory_trades and running_equity < _m.negative_trajectory_pnl:
            return (day_str,
                    f"negative trajectory: ${running_equity:,.0f} after {n_acc} trades",
                    accumulated)

        # Flat regime kill: not making money = no alpha
        if n_acc >= _m.flat_regime_trades and abs(running_equity) < _m.flat_regime_pnl:
            return (day_str,
                    f"flat regime: ${running_equity:,.0f} after {n_acc} trades",
                    accumulated)

        # Late-death kill: was profitable but gave it all back
        # If we were up > $500 at some point but now <= $0, the edge is gone
        if n_acc >= _m.negative_trajectory_trades and peak_equity >= 500 and running_equity <= 0:
            return (day_str,
                    f"late death: peaked at ${peak_equity:,.0f}, now ${running_equity:,.0f} after {n_acc} trades",
                    accumulated)

    return None, "alive", accumulated


# ─── LightGBM Classifier Gate ───────────────────────────────────────────────

# Feature columns must match online_trainer._FEATURE_COLS exactly
_CLASSIFIER_FEATURE_COLS = [
    "adjusted_fitness", "train_sharpe", "train_pf", "train_wr",
    "train_n_trades", "train_pnl", "has_tsl_be", "exit_type_encoded",
    "win_loss_ratio", "expectancy_per_trade", "kelly_fraction",
    "trades_per_day", "mc_p_value", "ttest_p_value", "pf_stability_ratio",
    "top_trade_pct", "remove_best_still_positive", "wr_stability_ratio",
    "payoff_consistency",
]

_CLASSIFIER_MODEL = None  # loaded once per job


def _load_classifier():
    """Load the LightGBM classifier from /data/models/classifier_latest.pkl."""
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL

    model_path = Path("/data/models/classifier_latest.pkl")
    if not model_path.exists():
        return None

    try:
        import joblib
        _CLASSIFIER_MODEL = joblib.load(model_path)
        print(f"[CLASSIFIER] Loaded model from {model_path}")
        return _CLASSIFIER_MODEL
    except Exception as e:
        print(f"[CLASSIFIER] Failed to load model: {e}")
        return None


def _classifier_score_winner(winner, mc) -> float | None:
    """Score a winner strategy with the LightGBM classifier.

    Returns P(winner) confidence or None if classifier unavailable.
    """
    model = _load_classifier()
    if model is None:
        return None

    # Build feature vector matching _FEATURE_COLS order
    rob = winner.summary.get("robustness", {}) or {}
    sd = winner.summary.get("strategy_def", {})
    exit_d = sd.get("exit", {})
    has_tsl_be = 1.0 if (exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts")) else 0.0
    exit_type = "tsl_be" if has_tsl_be else "fixed_sl_tp"
    exit_map = {"fixed_pts": 0, "candle_wick": 1, "level": 2, "risk_multiple": 3,
                "tsl_be": 4, "fixed_sl_tp": 0}

    feats = [
        float(winner.summary.get("adjusted_fitness") or winner.fitness.fitness),
        float(winner.fitness.sharpe),
        float(winner.fitness.profit_factor),
        float(winner.fitness.win_rate),
        float(winner.fitness.n_trades),
        float(winner.fitness.total_pnl),
        has_tsl_be,
        float(exit_map.get(exit_type, -1)),
        float(rob.get("win_loss_ratio") or 0),
        float(rob.get("expectancy_per_trade") or 0),
        float(rob.get("kelly_fraction") or 0),
        float(rob.get("trades_per_day") or 0),
        float(rob.get("mc_p_value") or 0),
        float(rob.get("ttest_p_value") or 0),
        float(rob.get("pf_stability_ratio") or 0),
        float(rob.get("top_trade_pct") or 0),
        float(rob.get("remove_best_still_positive") or 0),
        float(rob.get("wr_stability_ratio") or 0),
        float(rob.get("payoff_consistency") or 0),
    ]

    try:
        X = np.array([feats], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = float(model.predict(X)[0])
        return confidence
    except Exception as e:
        print(f"[CLASSIFIER] Scoring failed: {e}")
        return None


def run_walkforward_llm(
    window_days: int = 30,
    max_regime_days: int = 180,
    overlap_days: int = 0,
    n_strategies: int = 10,
    n_refinements: int = 20,
    start_date: str = None,
    max_dd_dollars: float = 3500,
    use_seeds: bool = True,
    use_genes: bool = False,
    max_regimes: int = 0,
    meta: MetaConfig | None = None,
    asset: str = "MES",
    on_regime_done: callable = None,
    overnight: bool = False,
) -> WalkForwardSummary:
    """Run walk-forward regime simulation using LLM-generated strategies.

    When use_genes=True, uses gene combinator search (systematic exploration
    of all DSL condition combinations). Overrides use_seeds.

    When use_seeds=True (default), evaluates curated strategy seeds first,
    then has the LLM tune the best seeds' parameters. Mid-run evolution
    adjusts strategy parameters every 2 weeks during forward testing.

    When use_seeds=False, falls back to from-scratch LLM generation.
    max_regimes: stop after this many regimes (0 = no limit).
    """
    global _UNCENSORED_RECORDS, _REJECTED_RECORDS, _AUDIT_TRADES
    _UNCENSORED_RECORDS = []  # reset buffer for this job
    _REJECTED_RECORDS = []    # reset rejection buffer for this job
    _AUDIT_TRADES = []        # reset audit trades buffer for this job
    print("[uncensored] MFE/MAE collection active (always-on)")

    # Job-level random seed — ensures different candidate space explored
    # on each run even for identical start dates
    import time as _time_mod
    import random as _random_mod
    job_seed = int(_time_mod.time() * 1000) % (2**31)
    _random_mod.seed(job_seed)
    np.random.seed(job_seed)
    print(f"[SEED] Job random seed: {job_seed}")

    _mc = meta or DEFAULT_META_CONFIG
    # Override meta-config DD with explicit parameter (CLI or job payload)
    if max_dd_dollars != _mc.max_dd_dollars:
        import copy
        _mc = copy.copy(_mc)
        _mc.max_dd_dollars = max_dd_dollars
    cache = DataCache(asset=asset)
    first_date, last_date = cache.date_range
    data_start = pd.Timestamp(first_date)
    data_end = pd.Timestamp(last_date)

    if start_date:
        optimize_start = pd.Timestamp(start_date)
    else:
        optimize_start = data_start + pd.Timedelta(days=120)

    if use_genes:
        mode_str = "GENE COMBINATOR"
    elif use_seeds:
        mode_str = "SEED + LLM TUNING"
    else:
        mode_str = "LLM FROM-SCRATCH"
    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD REGIME SIMULATION [{mode_str}] — {asset}")
    print(f"{'=' * 70}")
    print(f"Asset: {cache.asset_config.name} ({asset})")
    print(f"Data: {first_date} to {last_date}")
    print(f"Optimization window: {window_days} days")
    print(f"Max regime lifespan: {max_regime_days} days")
    print(f"Max DD threshold: ${_mc.max_dd_dollars:,.0f}")
    if max_regimes:
        print(f"Max regimes: {max_regimes}")
    print(f"First optimization: {optimize_start.strftime('%Y-%m-%d')}")
    if use_genes:
        print(f"Mode: Gene combinator (5000 candidates + LLM refinement of top 3)")
    elif use_seeds:
        print(f"Seeds: {len(STRATEGY_SEEDS)} archetypes, {n_refinements} LLM refinements each")
    else:
        print(f"Strategies per regime: {n_strategies} + {n_refinements} refinements each")

    regimes: list[RegimeRecord] = []
    all_forward_trades = []
    regime_id = 0
    total_opt_time = 0
    prior_results = []  # feedback loop for LLM
    failed_archetypes = []  # track archetypes that failed OOS for diversity penalty
    consecutive_gate_rejects = 0  # force-deploy after too many rejections
    MAX_CONSECUTIVE_GATE_REJECTS = 3

    while True:
        # Max regimes cap
        if max_regimes and regime_id >= max_regimes:
            print(f"\n  Reached max regimes cap ({max_regimes}). Done.")
            break

        optimize_end = optimize_start + pd.Timedelta(days=window_days)

        if optimize_end + pd.Timedelta(days=5) > data_end:
            print(f"\n  Not enough data for regime #{regime_id + 1} "
                  f"(need past {optimize_end.strftime('%Y-%m-%d')}). Done.")
            break

        opt_start_str = optimize_start.strftime("%Y-%m-%d")
        opt_end_str = optimize_end.strftime("%Y-%m-%d")

        regime_id += 1
        regime_t0 = tm.time()
        print(f"\n--- Regime #{regime_id}: optimize {opt_start_str} to {opt_end_str} ---")

        # 0. Macro gate — check if environment is suitable for trading
        try:
            _macro_cfg = load_macro_config()
            _macro_gate = MacroGate(_macro_cfg)
            _macro_snap = compute_market_snapshot(
                cache.candles_1m, opt_end_str, asset_config=cache.asset_config)
            _macro_decision = _macro_gate.evaluate(_macro_snap)
            if _macro_decision.blocked:
                print(f"  [MACRO GATE] BLOCKED: {', '.join(_macro_decision.reasons)}")
                print(f"  [MACRO GATE] score={_macro_decision.score:.3f} — OBSERVE ONLY, continuing anyway")
                # Observe-only: log but don't skip
            elif _macro_decision.reduced:
                print(f"  [MACRO GATE] REDUCED: score={_macro_decision.score:.3f} "
                      f"size_scalar={_macro_decision.size_scalar:.2f}")
            else:
                print(f"  [MACRO GATE] PASS: score={_macro_decision.score:.3f}")
        except Exception as _mg_err:
            _macro_decision = None
            print(f"  [MACRO GATE] Error: {_mg_err}")

        # 1. Search (gene combinator, seed-based, or from-scratch)
        t0 = tm.time()
        if use_genes:
            from gene_combinator import run_gene_search
            # Gene search already tests ~3800 combos — light LLM refinement only
            gene_refinements = min(n_refinements, 3)
            results = run_gene_search(
                cache, opt_start_str, opt_end_str,
                window_days=window_days,
                max_candidates=3000,
                top_n=1,
                n_refinements=gene_refinements,
                prior_results=prior_results if prior_results else None,
                failed_archetypes=failed_archetypes,
                meta=_mc,
                overnight=overnight,
            )
        elif use_seeds:
            results = run_seed_search(
                cache, opt_start_str, opt_end_str,
                window_days=window_days,
                n_refinements=n_refinements,
                prior_results=prior_results if prior_results else None,
                top_n=1,
                failed_archetypes=failed_archetypes,
            )
        else:
            results = run_llm_search(
                cache, opt_start_str, opt_end_str,
                window_days=window_days,
                n_strategies=n_strategies,
                n_refinements=n_refinements,
                prior_results=prior_results if prior_results else None,
                top_n=1,
            )
        opt_time = tm.time() - t0
        total_opt_time += opt_time

        if not results or results[0].fitness.fitness < 0:
            best = f"{results[0].fitness.fitness:.3f}" if results else "N/A"
            print(f"  No viable winner (best fitness: {best}). "
                  f"Skipping, advancing 30 days.")
            optimize_start += pd.Timedelta(days=30)
            continue

        winner = results[0]
        winner_strat_dict = winner.summary.get("strategy_def", {})
        winner_strat = StrategyDefinition.from_dict(winner_strat_dict)
        winner_name = winner_strat.name
        winner_archetype = winner.summary.get("archetype", "unknown")
        # Persist archetype in winner_params for output
        winner_strat_dict["archetype"] = winner_archetype

        # ── Scoring gate: reject strategies unlikely to be forward-profitable ──
        gate_score = _mc.archetype_scores.get(winner_archetype, 0.0)
        if winner.fitness.fitness >= _mc.fitness_bonus_high:
            gate_score += 0.5
        elif winner.fitness.fitness < _mc.fitness_bonus_low:
            gate_score -= 0.5
        train_rob = winner.summary.get("robustness", {})
        kelly = train_rob.get("kelly_fraction") if train_rob else None
        if kelly is not None:
            if kelly < 0:
                gate_score -= 0.5
            elif kelly > 0.5:
                gate_score += 0.5

        if gate_score < _mc.gate_threshold:
            consecutive_gate_rejects += 1
            if consecutive_gate_rejects < MAX_CONSECUTIVE_GATE_REJECTS:
                print(f"  GATE REJECT: '{winner_name}' [{winner_archetype}] "
                      f"score={gate_score:.2f} < {_mc.gate_threshold:.2f} | "
                      f"fitness={winner.fitness.fitness:.3f} "
                      f"({consecutive_gate_rejects}/{MAX_CONSECUTIVE_GATE_REJECTS} rejects)")
                optimize_start += pd.Timedelta(days=window_days)
                continue
            else:
                print(f"  GATE OVERRIDE ({consecutive_gate_rejects} consecutive rejects): "
                      f"force-deploying '{winner_name}' [{winner_archetype}] "
                      f"score={gate_score:.2f} | fitness={winner.fitness.fitness:.3f}")
                consecutive_gate_rejects = 0

        else:
            consecutive_gate_rejects = 0

        # ── Deterministic regime filter gate ──
        _train_rob = winner.summary.get("robustness") or {}
        _filter_score = _regime_filter_score(
            train_fitness=winner.fitness.fitness,
            train_robustness=_train_rob,
            val_pnl=winner.summary.get("val_pnl"),
        )
        _filter_deploy = _regime_filter_deploy(_filter_score)
        _filter_str = (f"score={_filter_score.total_score:.3f} "
                       f"decision={_filter_score.decision}")
        if _filter_score.hard_reject:
            print(f"  [REGIME FILTER] HARD REJECT: '{winner_name}' [{winner_archetype}] "
                  f"{_filter_str} — {_filter_score.hard_reject_reason}")
            optimize_start += pd.Timedelta(days=window_days)
            continue
        elif not _filter_deploy:
            print(f"  [REGIME FILTER] SOFT REJECT: '{winner_name}' [{winner_archetype}] "
                  f"{_filter_str} | {_filter_score.reasons}")
            # Don't hard-skip on soft reject for now — log it and continue deploying
            # Once we have calibrated thresholds, uncomment the skip:
            # optimize_start += pd.Timedelta(days=window_days)
            # continue
        else:
            print(f"  [REGIME FILTER] PASS: '{winner_name}' [{winner_archetype}] "
                  f"{_filter_str}")

        # ── LightGBM classifier gate ──
        _classifier_override = os.environ.get("CLASSIFIER_OVERRIDE", "")
        _clf_confidence = _classifier_score_winner(winner, _mc)
        if _clf_confidence is not None:
            if _classifier_override == "explore":
                print(f"  [CLASSIFIER] confidence={_clf_confidence:.3f} "
                      f"(threshold={_mc.classifier_confidence_threshold:.2f}, "
                      f"BYPASSED for exploration)")
            elif _clf_confidence < _mc.classifier_confidence_threshold:
                print(f"  [CLASSIFIER] REJECT: '{winner_name}' [{winner_archetype}] "
                      f"confidence={_clf_confidence:.3f} < "
                      f"{_mc.classifier_confidence_threshold:.2f}")
                # Log rejection for future training data
                try:
                    _rob = winner.summary.get("robustness", {}) or {}
                    _sd = winner.summary.get("strategy_def", {})
                    _exit_d = _sd.get("exit", {})
                    _has_tsl = 1.0 if (_exit_d.get("be_trigger_pts") or _exit_d.get("trail_distance_pts")) else 0.0
                    _etype = "tsl_be" if _has_tsl else "fixed_sl_tp"
                    _emap = {"fixed_pts": 0, "candle_wick": 1, "level": 2,
                             "risk_multiple": 3, "tsl_be": 4, "fixed_sl_tp": 0}
                    _rej = {
                        "timestamp": datetime.now().isoformat(),
                        "strategy_name": winner_name,
                        "start_date": opt_start_str,
                        "optimize_window": window_days,
                        "archetype": winner_archetype,
                        "classifier_confidence": round(_clf_confidence, 5),
                        "threshold": _mc.classifier_confidence_threshold,
                        "adjusted_fitness": float(winner.summary.get("adjusted_fitness") or winner.fitness.fitness),
                        "train_sharpe": float(winner.fitness.sharpe),
                        "train_pf": float(winner.fitness.profit_factor),
                        "train_wr": float(winner.fitness.win_rate),
                        "train_n_trades": int(winner.fitness.n_trades),
                        "train_pnl": float(winner.fitness.total_pnl),
                        "has_tsl_be": _has_tsl,
                        "exit_type_encoded": float(_emap.get(_etype, -1)),
                        "win_loss_ratio": float(_rob.get("win_loss_ratio") or 0),
                        "expectancy_per_trade": float(_rob.get("expectancy_per_trade") or 0),
                        "kelly_fraction": float(_rob.get("kelly_fraction") or 0),
                        "trades_per_day": float(_rob.get("trades_per_day") or 0),
                        "mc_p_value": float(_rob.get("mc_p_value") or 0),
                        "ttest_p_value": float(_rob.get("ttest_p_value") or 0),
                        "pf_stability_ratio": float(_rob.get("pf_stability_ratio") or 0),
                        "top_trade_pct": float(_rob.get("top_trade_pct") or 0),
                        "remove_best_still_positive": float(_rob.get("remove_best_still_positive") or 0),
                        "wr_stability_ratio": float(_rob.get("wr_stability_ratio") or 0),
                        "payoff_consistency": float(_rob.get("payoff_consistency") or 0),
                        "val_pnl": float(winner.summary.get("val_pnl") or 0),
                        "train_fitness": float(winner.fitness.fitness),
                    }
                    _rej_path = Path(__file__).resolve().parent.parent.parent / "results" / "rejected_regimes.jsonl"
                    with open(_rej_path, "a") as _rf:
                        _rf.write(json.dumps(_rej) + "\n")
                    _REJECTED_RECORDS.append(_rej)
                except Exception as _re:
                    print(f"  [CLASSIFIER] Rejection log failed: {_re}")
                optimize_start += pd.Timedelta(days=window_days)
                continue
            else:
                print(f"  [CLASSIFIER] confidence={_clf_confidence:.3f} "
                      f"(threshold={_mc.classifier_confidence_threshold:.2f}) ✓")

        _clf_str = f" | clf={_clf_confidence:.3f}" if _clf_confidence is not None else ""
        print(f"  Winner: '{winner_name}' [{winner_archetype}] | "
              f"fitness={winner.fitness.fitness:.3f} | "
              f"Sharpe={winner.fitness.sharpe:.2f} | "
              f"P&L=${winner.fitness.total_pnl:,.0f} | "
              f"trades={winner.fitness.n_trades} | "
              f"score={gate_score:.2f}{_clf_str} | "
              f"({opt_time:.1f}s)")

        # 2. Forward test (chunked with mid-run evolution)
        forward_start_str = opt_end_str
        forward_end_str = last_date
        print(f"  Forward testing {forward_start_str} to {forward_end_str} "
              f"(chunked, evolution every 14d)...")

        fwd_t0 = tm.time()
        # Compute train baselines for evolution confidence
        _train_wr = winner.fitness.win_rate / 100.0 if winner.fitness.win_rate > 1 else winner.fitness.win_rate
        _train_avg_loss = abs(winner.fitness.total_pnl / max(winner.fitness.n_trades, 1)) if winner.fitness.total_pnl < 0 else 50.0

        trades, equity, fwd_summary, n_evolutions = run_forward_test_chunked(
            cache, winner_strat, forward_start_str, forward_end_str,
            train_wr=_train_wr,
            train_avg_loss=_train_avg_loss,
            meta=_mc,
            overnight=overnight,
        )
        print(f"  [TIMER] Forward test: {tm.time() - fwd_t0:.1f}s")

        if n_evolutions > 0:
            print(f"  Mid-run evolutions applied: {n_evolutions}")

        # 3. Find regime death — adaptive DD limit from training P&L distribution
        _train_pnls = winner.summary.get("pnls_array", [])
        if _train_pnls and len(_train_pnls) >= 5:
            _adaptive_dd = compute_adaptive_dd_limit(
                _train_pnls,
                n_forward_trades=60,
                confidence=0.95,
                floor=500.0,
                ceiling=_mc.max_dd_dollars,
            )
            print(f"  Adaptive DD limit: ${_adaptive_dd:,.0f} "
                  f"(from {len(_train_pnls)} train trades, "
                  f"ceiling=${_mc.max_dd_dollars:,.0f})")
        else:
            _adaptive_dd = _mc.max_dd_dollars
            print(f"  Using static DD limit: ${_adaptive_dd:,.0f} (insufficient train PnLs)")

        death_date, death_reason, trades_before_death = find_regime_death(
            trades, forward_start_str, max_days=max_regime_days,
            max_dd_dollars=_adaptive_dd,
            meta=_mc,
            train_wr=_train_wr,
        )

        if death_date:
            actual_trades = trades_before_death
            actual_end = death_date
        else:
            actual_trades = trades
            actual_end = forward_end_str
            death_reason = "data_end"

        # Compute forward metrics
        fwd_n = len(actual_trades)
        fwd_pnl = sum(float(t.pnl_dollars or 0) for t in actual_trades)
        fwd_wins = sum(1 for t in actual_trades if (t.pnl_dollars or 0) > 0)
        fwd_wr = (fwd_wins / fwd_n * 100) if fwd_n else 0
        fwd_days = (pd.Timestamp(actual_end) - pd.Timestamp(forward_start_str)).days

        fwd_max_dd = 0.0
        if actual_trades:
            running = 0.0
            peak = 0.0
            for t in actual_trades:
                running += float(t.pnl_dollars or 0)
                peak = max(peak, running)
                fwd_max_dd = max(fwd_max_dd, peak - running)

        # Compute final Bayesian posterior at death
        _kill_posterior = None
        if fwd_n >= 5 and _train_wr > 0:
            _kill_posterior = round(bayesian_edge_probability(
                n_wins=fwd_wins,
                n_trades=fwd_n,
                assumed_edge_wr=_train_wr,
                prior_edge=0.70,
            ), 4)

        print(f"  Forward: {fwd_n} trades, ${fwd_pnl:,.0f} P&L, "
              f"{fwd_wr:.1f}% WR, {fwd_days}d | death: {death_reason}"
              f"{f' | posterior={_kill_posterior:.3f}' if _kill_posterior is not None else ''}")

        # Compute robustness metrics on both training and forward trades
        train_rob = winner.summary.get("robustness")  # already computed during search
        fwd_pnls = [float(t.pnl_dollars or 0) for t in actual_trades]
        fwd_rob_obj = compute_robustness_metrics(fwd_pnls, fwd_days) if fwd_days > 0 else None
        fwd_rob = fwd_rob_obj.to_dict() if fwd_rob_obj else None

        # 4. Record regime
        # Extract candidates from winner's summary (attached by search functions)
        _candidates = winner.summary.pop("_candidates_top50", None)

        # Death context — market conditions at regime death
        _death_gex_regime = None
        _death_rv = None
        from datetime import date as _date, time as _time
        _t10 = _time(10, 0)
        if cache.gex_cache and actual_end:
            try:
                death_d = _date.fromisoformat(actual_end) if isinstance(actual_end, str) else actual_end
                snap = cache.gex_cache.get(death_d, _t10)
                if snap:
                    _death_gex_regime = snap.gex_regime  # "positive"/"negative"/"neutral"
            except Exception:
                pass
        if cache.daily_cache and actual_end:
            try:
                death_d = _date.fromisoformat(actual_end) if isinstance(actual_end, str) else actual_end
                dc_day = cache.daily_cache.get(death_d)
                if dc_day and hasattr(dc_day, 'realized_vol_5m'):
                    _death_rv = round(dc_day.realized_vol_5m, 4) if dc_day.realized_vol_5m else None
            except Exception:
                pass

        # Optimization context — market conditions at optimize start (deployment-time feature)
        _opt_gex_regime = None
        _opt_rv = None
        if cache.gex_cache and opt_start_str:
            try:
                opt_d = _date.fromisoformat(opt_start_str)
                snap = cache.gex_cache.get(opt_d, _t10)
                if snap:
                    _opt_gex_regime = snap.gex_regime  # "positive"/"negative"/"neutral"
            except Exception:
                pass
        if cache.daily_cache and opt_start_str:
            try:
                opt_d = _date.fromisoformat(opt_start_str)
                dc_day = cache.daily_cache.get(opt_d)
                if dc_day and hasattr(dc_day, 'realized_vol_5m'):
                    _opt_rv = round(dc_day.realized_vol_5m, 4) if dc_day.realized_vol_5m else None
            except Exception:
                pass

        # Market snapshot at deployment time (forward_start)
        _market_snap = None
        try:
            _market_snap = compute_market_snapshot(
                cache.candles_1m, forward_start_str, asset_config=cache.asset_config)
            if _market_snap and "error" not in _market_snap:
                print(f"  [SNAPSHOT] vol={_market_snap['daily_vol_pct']:.3f}% "
                      f"ON_range={_market_snap['avg_on_range_pts']:.1f}pts "
                      f"ret10d={_market_snap['ret_10d_pct']:+.2f}% "
                      f"range_ratio={_market_snap['range_ratio_5d_20d']:.2f} "
                      f"gaps={_market_snap['gap_pct']:.0f}%")
        except Exception as _snap_err:
            print(f"  [SNAPSHOT] Failed: {_snap_err}")

        record = RegimeRecord(
            regime_id=regime_id,
            optimize_start=opt_start_str,
            optimize_end=opt_end_str,
            forward_start=forward_start_str,
            forward_end=actual_end,
            death_reason=death_reason,
            winner_params=winner_strat_dict,
            winner_fitness=winner.fitness.fitness,
            forward_trades=fwd_n,
            forward_pnl=round(fwd_pnl, 2),
            forward_win_rate=round(fwd_wr, 2),
            forward_max_dd=round(fwd_max_dd, 2),
            forward_days=fwd_days,
            optimization_time_sec=round(opt_time, 1),
            train_sharpe=round(winner.fitness.sharpe, 3),
            train_pf=round(winner.fitness.profit_factor, 3),
            train_n_trades=winner.fitness.n_trades,
            train_robustness=train_rob,
            forward_robustness=fwd_rob,
            candidates_top50=_candidates,
            kill_posterior=_kill_posterior,
            death_gex_regime=_death_gex_regime,
            death_rv=_death_rv,
            optimize_gex_regime=_opt_gex_regime,
            optimize_rv=_opt_rv,
            market_snapshot=_market_snap,
            n_evolutions=n_evolutions,
        )
        regimes.append(record)
        all_forward_trades.extend(actual_trades)

        # Audit mode: collect per-trade records for portfolio simulation
        if _AUDIT_MODE and actual_trades:
            _strat_dir = winner_strat_dict.get("entry", {}).get("direction", "long")
            for _ti, _t in enumerate(actual_trades):
                _entry_ts = str(_t.entry_time) if _t.entry_time else ""
                _exit_ts = str(_t.exit_time) if _t.exit_time else ""
                # Compute bars held from entry/exit timestamps (1-min bars)
                _bars = 0
                if _t.entry_time and _t.exit_time:
                    _td = (_t.exit_time - _t.entry_time).total_seconds()
                    _bars = max(1, int(_td / 60))
                _AUDIT_TRADES.append({
                    "trade_id": f"r{regime_id}_t{_ti}",
                    "regime_id": str(regime_id),
                    "strategy_name": winner_name,
                    "archetype": winner_archetype,
                    "optimize_start": opt_start_str,
                    "entry_time": _entry_ts,
                    "exit_time": _exit_ts,
                    "direction": _strat_dir if _strat_dir in ("long", "short") else ("long" if (_t.pnl or 0) >= 0 and _t.entry_price and _t.exit_price and _t.exit_price > _t.entry_price else "short" if _t.entry_price and _t.exit_price else "long"),
                    "entry_price": float(_t.entry_price) if _t.entry_price else 0.0,
                    "exit_price": float(_t.exit_price) if _t.exit_price else 0.0,
                    "pnl": float(_t.pnl_dollars or 0),
                    "bars_held": _bars,
                    "exit_reason": _t.result or "unknown",
                    "death_reason": death_reason if _t == actual_trades[-1] else None,
                    "classifier_confidence": round(float(_clf_confidence), 5) if _clf_confidence is not None else None,
                    "train_fitness": round(float(winner.fitness.fitness), 5),
                })
            print(f"  [audit] Collected {len(actual_trades)} trades for regime #{regime_id}")

        regime_elapsed = tm.time() - regime_t0
        print(f"  [TIMER] ═══ Regime #{regime_id} total: {regime_elapsed:.1f}s "
              f"(search={opt_time:.1f}s + forward={tm.time() - fwd_t0:.1f}s) ═══")

        # Report regime progress to orchestrator (via Render callback)
        if on_regime_done:
            try:
                on_regime_done(
                    regime_index=regime_id,
                    regime_elapsed_sec=regime_elapsed,
                    regime_record=record.to_dict() if hasattr(record, 'to_dict') else None,
                )
            except Exception as e:
                print(f"  [progress] Callback error: {e}")

        # 5. Feed back results to next LLM call (with archetype tracking)
        prior_results.append({
            "strategy_name": winner_name,
            "archetype": winner_archetype,
            "fitness": winner.fitness.fitness,
            "death_reason": death_reason,
            "forward_pnl": fwd_pnl,
            "forward_win_rate": fwd_wr,
            "forward_days": fwd_days,
            "n_evolutions": n_evolutions,
            "evolution_log": fwd_summary.get("evolution_log", []),
        })

        # Track failed archetypes for diversity penalty (keep last 3)
        if fwd_pnl < 0:
            failed_archetypes.append(winner_archetype)
        if len(failed_archetypes) > 3:
            failed_archetypes = failed_archetypes[-3:]

        # 6. Advance optimization start
        if death_date:
            optimize_start = pd.Timestamp(death_date) - pd.Timedelta(days=overlap_days)
        else:
            break

    # ── Build aggregate summary ──
    total_trades = len(all_forward_trades)
    total_pnl = sum(float(t.pnl_dollars or 0) for t in all_forward_trades)
    total_wins = sum(1 for t in all_forward_trades if (t.pnl_dollars or 0) > 0)
    total_wr = (total_wins / total_trades * 100) if total_trades else 0
    total_fwd_days = sum(r.forward_days for r in regimes)
    lifespans = [r.forward_days for r in regimes]
    avg_lifespan = np.mean(lifespans) if lifespans else 0
    median_lifespan = float(np.median(lifespans)) if lifespans else 0
    avg_pnl = total_pnl / len(regimes) if regimes else 0

    agg_max_dd = 0.0
    running = 0.0
    peak = 0.0
    for t in all_forward_trades:
        running += float(t.pnl_dollars or 0)
        peak = max(peak, running)
        agg_max_dd = max(agg_max_dd, peak - running)

    equity_curve = []
    running = 0.0
    for t in all_forward_trades:
        running += float(t.pnl_dollars or 0)
        ts = t.exit_time if t.exit_time else t.entry_time
        equity_curve.append({
            "date": ts.strftime("%Y-%m-%d") if ts else "",
            "equity": round(running, 2),
        })

    summary = WalkForwardSummary(
        data_start=first_date,
        data_end=last_date,
        window_days=window_days,
        total_regimes=len(regimes),
        total_trades=total_trades,
        total_pnl=round(total_pnl, 2),
        total_forward_days=total_fwd_days,
        aggregate_win_rate=round(total_wr, 2),
        aggregate_max_dd=round(agg_max_dd, 2),
        avg_regime_lifespan_days=round(avg_lifespan, 1),
        median_regime_lifespan_days=round(median_lifespan, 1),
        avg_regime_pnl=round(avg_pnl, 2),
        total_optimization_time_sec=round(total_opt_time, 1),
        regimes=regimes,
        equity_curve=equity_curve,
        meta_config=_mc.to_dict(),
    )

    return summary


def print_walkforward_summary(summary: WalkForwardSummary):
    """Pretty-print walk-forward results."""
    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD RESULTS")
    print(f"{'=' * 70}")
    print(f"  Data range: {summary.data_start} to {summary.data_end}")
    print(f"  Optimization window: {summary.window_days} days")
    print(f"  Total regimes: {summary.total_regimes}")
    print(f"  Total forward-test trades: {summary.total_trades}")
    print(f"  Total P&L: ${summary.total_pnl:,.2f}")
    print(f"  Aggregate win rate: {summary.aggregate_win_rate:.1f}%")
    print(f"  Aggregate max DD: ${summary.aggregate_max_dd:,.2f}")
    print(f"  Avg regime lifespan: {summary.avg_regime_lifespan_days:.0f} days "
          f"(median: {summary.median_regime_lifespan_days:.0f})")
    print(f"  Avg regime P&L: ${summary.avg_regime_pnl:,.2f}")
    print(f"  Total optimization time: {summary.total_optimization_time_sec:.0f}s")

    print(f"\n  Per-regime breakdown:")
    print(f"  {'#':>3} {'Opt Window':<25} {'Fwd Days':>8} {'Trades':>7} "
          f"{'P&L':>10} {'WR':>6} {'MaxDD':>10} {'Death Reason'}")
    print(f"  {'-' * 100}")

    for r in summary.regimes:
        print(f"  {r.regime_id:>3} {r.optimize_start} → {r.optimize_end}  "
              f"{r.forward_days:>7}d {r.forward_trades:>7} "
              f"${r.forward_pnl:>9,.0f} {r.forward_win_rate:>5.1f}% "
              f"${r.forward_max_dd:>9,.0f}  {r.death_reason}")

    print()


def save_walkforward_results(summary: WalkForwardSummary, output_dir: Path) -> Path:
    """Save walk-forward results as comprehensive JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = output_dir / f"walkforward_{ts}.json"
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)

    print(f"Walk-forward results saved: {path}")
    return path


# ─── Output ───────────────────────────────────────────────────────────────────

def print_results(results: list[SearchResult]):
    """Pretty-print top candidates."""
    print("\n" + "=" * 90)
    print("TOP CANDIDATES — Overfit Search Results")
    print("=" * 90)

    for r in results:
        f = r.fitness
        s = r.summary
        print(f"\n#{r.rank} | fitness={f.fitness:.3f} | {r.params.label}")
        print(f"  Sharpe={f.sharpe:.2f} | PF={f.profit_factor:.2f} | "
              f"WR={f.win_rate:.1f}% | Trades={f.n_trades}")
        print(f"  P&L=${f.total_pnl:,.0f} | MaxDD=${f.max_dd:,.0f} | "
              f"Avg=${f.avg_trade_pnl:.0f}/trade")
        if any([f.trade_count_penalty, f.dd_penalty, f.concentration_penalty]):
            penalties = []
            if f.trade_count_penalty:
                penalties.append(f"trades={f.trade_count_penalty:.2f}")
            if f.dd_penalty:
                penalties.append(f"dd={f.dd_penalty:.2f}")
            if f.concentration_penalty:
                penalties.append(f"conc={f.concentration_penalty:.2f}")
            print(f"  Penalties: {', '.join(penalties)}")

        # Key params that differ from baseline
        p = r.params
        diffs = []
        if p.m1_extrema_distance != 15.0:
            diffs.append(f"extrema_dist={p.m1_extrema_distance}")
        if p.gap_max_age_days != 7.0:
            diffs.append(f"gap_age={p.gap_max_age_days}d")
        if p.m1_max_sl_pts != 18.0:
            diffs.append(f"max_sl={p.m1_max_sl_pts}")
        if p.partial_profit != 14.0:
            diffs.append(f"partial_tp={p.partial_profit}")
        if p.runner_r != 1.5:
            diffs.append(f"runner_r={p.runner_r}")
        if p.penetration != 3.0:
            diffs.append(f"penetration={p.penetration}")
        if not p.enable_m2:
            diffs.append("M2=OFF")
        if not p.enable_m3:
            diffs.append("M3=OFF")
        if diffs:
            print(f"  Params: {', '.join(diffs)}")


def save_results(results: list[SearchResult], window_days: int):
    """Save results to JSON + CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON with full details
    out = {
        "search_time": ts,
        "window_days": window_days,
        "n_candidates_evaluated": len(results),
        "candidates": [
            {
                "rank": r.rank,
                "params": r.params.to_dict(),
                "fitness": r.fitness.to_dict(),
                "summary": r.summary,
            }
            for r in results
        ],
    }
    json_path = OUTPUT_DIR / f"search_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # CSV summary
    rows = []
    for r in results:
        row = {"rank": r.rank, **r.fitness.to_dict(), **r.params.to_dict()}
        rows.append(row)
    csv_path = OUTPUT_DIR / f"search_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nResults saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    return json_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def save_deploy_config(winner: SearchResult, window_days: int):
    """Write winner config as deployable JSON + .env format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    p = winner.params
    deploy = {
        "generated_at": datetime.now().isoformat(),
        "window_days": window_days,
        "fitness": winner.fitness.fitness,
        "sharpe": winner.fitness.sharpe,
        "profit_factor": winner.fitness.profit_factor,
        "total_pnl": winner.fitness.total_pnl,
        "n_trades": winner.fitness.n_trades,
        "params": p.to_dict(),
    }
    json_path = OUTPUT_DIR / "deploy_config.json"
    with open(json_path, "w") as f:
        json.dump(deploy, f, indent=2, default=str)

    env_lines = [
        f"MES_M1_EXTREMA_DISTANCE={p.m1_extrema_distance}",
        f"MES_GAP_MAX_AGE_DAYS={p.gap_max_age_days}",
        f"MES_MIN_GAP_SIZE={p.min_gap_size}",
        f"MES_OVERSIZED_GAP={p.oversized_gap}",
        f"MES_M1_MAX_SL_PTS={p.m1_max_sl_pts}",
        f"MES_SL_BUFFER={p.sl_buffer}",
        f"MES_MIN_SL_PTS={p.min_sl_pts}",
        f"MES_LARGE_STOP={p.large_stop}",
        f"MES_PARTIAL_PROFIT={p.partial_profit}",
        f"MES_RUNNER_R={p.runner_r}",
        f"MES_PENETRATION={p.penetration}",
        f"MES_ENTRY_STYLE={p.entry_style}",
        f"MES_ENABLE_M2={str(p.enable_m2).lower()}",
        f"MES_ENABLE_M3={str(p.enable_m3).lower()}",
        f"MES_M2_BREAKOUT_MIN={p.m2_breakout_min}",
    ]
    env_path = OUTPUT_DIR / "deploy.env"
    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")

    print(f"\nDeploy config written:")
    print(f"  {json_path}")
    print(f"  {env_path}")


def main():
    parser = argparse.ArgumentParser(description="Overfit Finder — MES Strategy Search")
    parser.add_argument("--window", type=int, default=90,
                        help="Lookback window in days (default: 90)")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Use reduced grid (default: True)")
    parser.add_argument("--full", action="store_true",
                        help="Use full grid (slower, more combos)")
    parser.add_argument("--random", type=int, default=0,
                        help="Additional random candidates to try")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top candidates to show")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM to generate structural variants (requires llm_variant.py)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 3 candidates, 14-day window, validates full flow")
    parser.add_argument("--deploy", action="store_true",
                        help="Write winner config as deploy_config.json + deploy.env")
    parser.add_argument("--audit", action="store_true",
                        help="Run LLM realism audit on top candidates (requires OPENAI_API_KEY)")
    parser.add_argument("--audit-top", type=int, default=3,
                        help="Number of top candidates to audit (default: 3)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward regime simulation across full data history")
    parser.add_argument("--wf-window", type=int, default=45,
                        help="Walk-forward optimization window in calendar days (default: 45)")
    parser.add_argument("--wf-max-regime", type=int, default=180,
                        help="Max regime lifespan before forced regeneration (default: 180)")
    parser.add_argument("--wf-overlap", type=int, default=0,
                        help="Days of overlap when starting next optimization (default: 0)")
    parser.add_argument("--wf-start", type=str, default=None,
                        help="Walk-forward start date (YYYY-MM-DD). Default: earliest viable date.")
    parser.add_argument("--n-strategies", type=int, default=10,
                        help="Number of LLM strategies to generate per regime (default: 10)")
    parser.add_argument("--n-refinements", type=int, default=20,
                        help="Number of refinement variants per top strategy (default: 20)")
    parser.add_argument("--seeds", action="store_true", default=True,
                        help="Use strategy seeds + LLM tuning (default: True)")
    parser.add_argument("--no-seeds", action="store_true",
                        help="Disable seeds, use from-scratch LLM generation")
    parser.add_argument("--genes", action="store_true",
                        help="Use gene combinator search (systematic exploration of DSL condition combos)")
    parser.add_argument("--wf-max-dd", type=float, default=3500,
                        help="Max drawdown dollars for regime death (default: 3500)")
    parser.add_argument("--max-regimes", type=int, default=0,
                        help="Max number of regimes to run (0=unlimited, default: 0)")
    parser.add_argument("--asset", type=str, default="MES",
                        help="Asset ticker to run (MES, F, BAC, SOFI, SNAP). Default: MES")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, 1=sequential)")
    parser.add_argument("--overnight", action="store_true",
                        help="Run overnight session search (18:00-09:30 ET, no GEX/earnings genes)")
    args = parser.parse_args()

    # Set worker count for parallel evaluation
    if args.workers:
        os.environ["OVERFIT_WORKERS"] = str(args.workers)

    # Walk-forward mode: its own pipeline
    if args.walkforward:
        use_seeds = args.seeds and not args.no_seeds
        wf_summary = run_walkforward_llm(
            window_days=args.wf_window,
            max_regime_days=args.wf_max_regime,
            overlap_days=args.wf_overlap,
            n_strategies=args.n_strategies,
            n_refinements=args.n_refinements,
            start_date=args.wf_start,
            max_dd_dollars=args.wf_max_dd,
            use_seeds=use_seeds,
            use_genes=args.genes,
            max_regimes=args.max_regimes,
            asset=args.asset,
            overnight=args.overnight,
        )
        print_walkforward_summary(wf_summary)
        save_walkforward_results(wf_summary, OUTPUT_DIR)
        return

    # Test mode: minimal grid, short window
    if args.test:
        print("=== TEST MODE ===")
        custom = [
            ParamSet(),                                              # baseline
            ParamSet(penetration=2.0, runner_r=2.0, gap_max_age_days=5.0),  # current winner
            ParamSet(m1_extrema_distance=20.0, gap_max_age_days=3.0),       # divergent
        ]
        results = run_search(
            window_days=14,
            custom_params=custom,
            top_n=3,
        )
        if results:
            print_results(results)
            save_results(results, 14)
            print("\n=== TEST PASSED ===")
        else:
            print("\n=== TEST FAILED: no results ===")
        return

    fast = not args.full

    # Run grid search
    results = run_search(
        window_days=args.window,
        fast=fast,
        n_random=args.random,
        top_n=args.top,
    )

    if not results:
        print("No valid candidates found!")
        return

    # Optionally run LLM mutations on top candidates
    if args.llm:
        try:
            from llm_variant import generate_llm_variants
            print("\n--- LLM Structural Mutations ---")
            top3 = results[:3]
            llm_params = generate_llm_variants(
                [r.params for r in top3],
                [r.fitness for r in top3],
                n_variants=10,
            )
            if llm_params:
                print(f"LLM generated {len(llm_params)} structural variants, backtesting...")
                llm_results = run_search(
                    window_days=args.window,
                    custom_params=llm_params,
                    top_n=args.top,
                )
                # Merge and re-rank
                all_results = results + llm_results
                all_results.sort(key=lambda r: r.fitness.fitness, reverse=True)
                for i, r in enumerate(all_results):
                    r.rank = i + 1
                results = all_results[:args.top]
        except ImportError:
            print("llm_variant.py not found — skipping LLM mutations")

    print_results(results)
    save_results(results, args.window)

    # Optionally run LLM realism audit
    if args.audit:
        try:
            from realism_audit import audit_candidates, print_audit_results, save_audit_results
            print("\n--- LLM Realism Audit ---")
            reports = audit_candidates(results, top_n=args.audit_top)
            print_audit_results(reports)
            save_audit_results(reports, OUTPUT_DIR)
        except ImportError:
            print("realism_audit.py not found — skipping audit")
        except RuntimeError as e:
            print(f"Audit failed: {e}")

    # Print the winner's full config for easy copy-paste
    winner = results[0]
    print("\n" + "=" * 90)
    print("WINNER — Copy this config to deploy:")
    print("=" * 90)
    print(json.dumps(winner.params.to_dict(), indent=2))

    # Optionally write deploy files
    if args.deploy:
        save_deploy_config(winner, args.window)


if __name__ == "__main__":
    main()
