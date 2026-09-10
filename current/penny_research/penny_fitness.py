"""
Penny Gapper Fitness Scoring
==============================
Composite fitness function for evaluating gene candidates.
Mirrors the MES fitness logic but tuned for penny intraday dynamics.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from penny_backtest import BacktestResult


@dataclass
class FitnessResult:
    fitness: float
    sharpe: float
    profit_factor: float
    win_rate: float
    n_trades: int
    avg_r: float
    cum_r: float
    max_dd_r: float
    # Breakdown
    sharpe_score: float = 0
    pf_score: float = 0
    trade_count_score: float = 0
    return_dd_score: float = 0
    penalties: float = 0
    bonuses: float = 0


def compute_fitness(
    result: BacktestResult,
    min_trades: int = 50,
    target_trades: int = 100,
    hard_sharpe_ceiling: float = 8.0,
    hard_pf_ceiling: float = 8.0,
) -> FitnessResult:
    """Compute composite fitness score for a gene candidate.

    Returns FitnessResult with fitness >= 0 for viable candidates,
    fitness = -1 for hard rejections.
    """
    n = result.n_trades
    wr = result.win_rate
    avg_r = result.avg_r
    cum_r = result.cum_r
    pf = result.profit_factor
    dd = result.max_dd_r
    sharpe = result.sharpe

    # ── Hard rejection ──
    if n < min_trades:
        return FitnessResult(fitness=-1.0, sharpe=sharpe, profit_factor=pf,
                             win_rate=wr, n_trades=n, avg_r=avg_r,
                             cum_r=cum_r, max_dd_r=dd)

    if sharpe > hard_sharpe_ceiling or pf > hard_pf_ceiling:
        return FitnessResult(fitness=-1.0, sharpe=sharpe, profit_factor=pf,
                             win_rate=wr, n_trades=n, avg_r=avg_r,
                             cum_r=cum_r, max_dd_r=dd,
                             penalties=1.0)

    # ── Component scores ──
    # Sharpe (capped at 5)
    sharpe_score = min(abs(sharpe), 5.0) / 5.0 * 0.20
    if sharpe < 0:
        sharpe_score = 0

    # Profit factor (capped at 5)
    pf_score = min(pf, 5.0) / 5.0 * 0.20
    if pf < 1:
        pf_score *= 0.3  # heavy penalty for PF < 1

    # Trade count (ramp to target)
    trade_count_score = min(n / target_trades, 1.0) * 0.25

    # Return:DD ratio
    return_dd_score = 0
    if dd > 0 and cum_r > 0:
        ratio = cum_r / dd
        return_dd_score = min(ratio / 3.0, 1.0) * 0.15
    elif cum_r > 0:
        return_dd_score = 0.15  # no drawdown = perfect

    # ── Bonuses ──
    bonuses = 0

    # Win rate bonus: penny strategies can have high WR
    if wr >= 0.50:
        bonuses += 0.10
    if wr >= 0.60:
        bonuses += 0.05

    # Consistency bonus: positive avg R with low DD
    if avg_r > 0 and dd < cum_r * 0.5:
        bonuses += 0.05

    # ── Penalties ──
    penalties = 0

    # Sharpe too high = likely overfit
    if sharpe > 5.0:
        penalties += (sharpe - 5.0) * 0.05

    # PF too high
    if pf > 5.0:
        penalties += (pf - 5.0) * 0.03

    # Concentration: if top trades dominate
    if n >= 5:
        rs = sorted([t.pnl_r for t in result.trades], reverse=True)
        top_pnl = rs[0]
        top3_pnl = sum(rs[:3])
        if cum_r > 0:
            if top_pnl > cum_r * 0.40:
                penalties += 0.15  # single trade carries >40% of profit
            if top3_pnl > cum_r * 0.70:
                penalties += 0.10  # top 3 trades carry >70% of profit

    # High drawdown relative to profits
    if cum_r > 0 and dd > cum_r * 0.8:
        penalties += 0.05

    # Negative P&L
    if cum_r <= 0:
        penalties += 0.30

    # ── Composite ──
    fitness = sharpe_score + pf_score + trade_count_score + return_dd_score + bonuses - penalties
    fitness = max(fitness, 0.0)

    return FitnessResult(
        fitness=round(fitness, 4),
        sharpe=round(sharpe, 3),
        profit_factor=round(pf, 3),
        win_rate=round(wr, 4),
        n_trades=n,
        avg_r=round(avg_r, 4),
        cum_r=round(cum_r, 2),
        max_dd_r=round(dd, 2),
        sharpe_score=round(sharpe_score, 4),
        pf_score=round(pf_score, 4),
        trade_count_score=round(trade_count_score, 4),
        return_dd_score=round(return_dd_score, 4),
        penalties=round(penalties, 4),
        bonuses=round(bonuses, 4),
    )
