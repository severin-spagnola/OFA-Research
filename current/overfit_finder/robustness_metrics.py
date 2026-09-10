"""
Robustness Metrics — Statistical Tests for Strategy Quality
============================================================
Collects data on whether a strategy's performance is statistically
distinguishable from luck. These metrics are OBSERVATIONAL ONLY —
they do not affect fitness scores or strategy selection.

All metrics are cheap to compute (<100ms total per strategy).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RobustnessMetrics:
    """Full battery of robustness diagnostics for one strategy's trades."""

    # ── Monte Carlo permutation test ──
    mc_p_value: float               # p-value: probability random shuffle beats real P&L
    mc_percentile: float            # where real P&L sits in shuffled distribution (0-100)
    mc_n_shuffles: int              # number of shuffles used

    # ── t-test on mean trade P&L ──
    ttest_p_value: float            # two-sided p-value: is mean trade P&L != 0?
    ttest_t_statistic: float        # t-statistic
    mean_trade_pnl: float           # mean trade P&L
    std_trade_pnl: float            # std dev of trade P&L

    # ── Profit factor stability (first half vs second half) ──
    pf_first_half: float            # profit factor of first half of trades
    pf_second_half: float           # profit factor of second half of trades
    pf_stability_ratio: float       # min(pf1,pf2) / max(pf1,pf2) — 1.0 = perfectly stable

    # ── Trade clustering / fragility ──
    top_trade_pct: float            # % of total P&L from single best trade
    top3_trade_pct: float           # % of total P&L from top 3 trades
    remove_best_pnl: float          # total P&L with best trade removed
    remove_best_still_positive: bool  # is strategy still profitable without best trade?

    # ── Win rate stability ──
    wr_first_half: float            # win rate of first half of trades
    wr_second_half: float           # win rate of second half of trades
    wr_stability_ratio: float       # min/max ratio — 1.0 = perfectly stable

    # ── Streak analysis ──
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_win_streak: float
    avg_loss_streak: float

    # ── Risk/reward consistency ──
    avg_win_dollars: float
    avg_loss_dollars: float
    win_loss_ratio: float           # avg_win / abs(avg_loss)
    payoff_consistency: float       # std(wins) / mean(wins) — lower = more consistent

    # ── Trade frequency ──
    trades_per_day: float           # n_trades / window_days
    trades_per_week: float          # n_trades / (window_days / 7)

    # ── Expectancy ──
    expectancy_per_trade: float     # (WR * avg_win) - ((1-WR) * avg_loss)
    kelly_fraction: float           # optimal Kelly bet fraction

    def to_dict(self) -> dict:
        return asdict(self)


def compute_robustness_metrics(
    pnls: list[float],
    window_days: int,
    n_shuffles: int = 1000,
) -> Optional[RobustnessMetrics]:
    """Compute full robustness battery on a list of trade P&Ls.

    Returns None if fewer than 5 trades (not enough data for any test).
    """
    n = len(pnls)
    if n < 5:
        return None

    pnls_arr = np.array(pnls, dtype=float)
    total_pnl = float(pnls_arr.sum())
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]

    # ── 1. Monte Carlo permutation test ──
    # Shuffle trade order, recompute max drawdown-adjusted P&L
    # If real P&L is in the top 5% of shuffled P&Ls, ordering matters
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    shuffled_pnls = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(pnls_arr)
        shuffled_pnls[i] = float(shuffled.sum())  # sum is order-invariant for total,
        # but we use max equity as the test statistic
        cumsum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cumsum)
        max_dd = float((peak - cumsum).max())
        # Risk-adjusted: P&L / (1 + max_dd)
        shuffled_pnls[i] = float(cumsum[-1]) / (1.0 + max_dd)

    # Real risk-adjusted metric
    real_cumsum = np.cumsum(pnls_arr)
    real_peak = np.maximum.accumulate(real_cumsum)
    real_max_dd = float((real_peak - real_cumsum).max())
    real_metric = total_pnl / (1.0 + real_max_dd)

    mc_percentile = float(np.mean(shuffled_pnls < real_metric) * 100)
    mc_p_value = 1.0 - mc_percentile / 100.0

    # ── 2. t-test on mean trade P&L ──
    mean_pnl = float(pnls_arr.mean())
    std_pnl = float(pnls_arr.std(ddof=1)) if n > 1 else 1.0
    t_stat = (mean_pnl / (std_pnl / np.sqrt(n))) if std_pnl > 0 else 0.0
    # Approximate p-value using normal distribution (good enough for n >= 10)
    # For small n, this is conservative
    from math import erfc, sqrt
    ttest_p = float(erfc(abs(t_stat) / sqrt(2)))

    # ── 3. Profit factor stability ──
    mid = n // 2
    first_half = pnls_arr[:mid]
    second_half = pnls_arr[mid:]

    def _pf(arr):
        w = arr[arr > 0].sum()
        l_abs = abs(arr[arr < 0].sum())
        return float(w / l_abs) if l_abs > 0.01 else 10.0

    pf1 = _pf(first_half)
    pf2 = _pf(second_half)
    pf_max = max(pf1, pf2, 0.01)
    pf_stability = min(pf1, pf2) / pf_max if pf_max > 0 else 0

    # ── 4. Trade clustering / fragility ──
    sorted_pnls = np.sort(pnls_arr)[::-1]
    if total_pnl > 0:
        top1_pct = float(sorted_pnls[0] / total_pnl * 100) if total_pnl != 0 else 0
        top3_pct = float(sorted_pnls[:3].sum() / total_pnl * 100) if total_pnl != 0 else 0
    else:
        # Negative total — clustering is meaningless, but record raw values
        top1_pct = 0.0
        top3_pct = 0.0
    remove_best = float(total_pnl - sorted_pnls[0])
    remove_best_positive = remove_best > 0

    # ── 5. Win rate stability ──
    wr1 = float(np.mean(first_half > 0) * 100)
    wr2 = float(np.mean(second_half > 0) * 100)
    wr_max = max(wr1, wr2, 0.01)
    wr_stability = min(wr1, wr2) / wr_max if wr_max > 0 else 0

    # ── 6. Streak analysis ──
    def _streaks(results):
        """Compute max and avg streak lengths for wins (True) and losses (False)."""
        if len(results) == 0:
            return 0, 0, 0.0, 0.0
        win_streaks = []
        loss_streaks = []
        current_streak = 1
        for i in range(1, len(results)):
            if results[i] == results[i - 1]:
                current_streak += 1
            else:
                if results[i - 1]:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_streak = 1
        # Don't forget last streak
        if results[-1]:
            win_streaks.append(current_streak)
        else:
            loss_streaks.append(current_streak)

        max_w = max(win_streaks) if win_streaks else 0
        max_l = max(loss_streaks) if loss_streaks else 0
        avg_w = float(np.mean(win_streaks)) if win_streaks else 0.0
        avg_l = float(np.mean(loss_streaks)) if loss_streaks else 0.0
        return max_w, max_l, avg_w, avg_l

    trade_results = [p > 0 for p in pnls]
    max_w, max_l, avg_w, avg_l = _streaks(trade_results)

    # ── 7. Risk/reward consistency ──
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 10.0
    payoff_consistency = float(wins.std() / wins.mean()) if len(wins) > 1 and wins.mean() > 0 else 0.0

    # ── 8. Trade frequency ──
    tpd = n / window_days if window_days > 0 else 0
    tpw = n / (window_days / 7) if window_days > 0 else 0

    # ── 9. Expectancy & Kelly ──
    wr_decimal = len(wins) / n if n > 0 else 0
    expectancy = (wr_decimal * avg_win) + ((1 - wr_decimal) * avg_loss)
    # Kelly: f* = (bp - q) / b where b = avg_win/abs(avg_loss), p = WR, q = 1-WR
    if avg_loss != 0 and abs(avg_loss) > 0:
        b = abs(avg_win / avg_loss)
        kelly = (b * wr_decimal - (1 - wr_decimal)) / b if b > 0 else 0
    else:
        kelly = 0.0

    return RobustnessMetrics(
        mc_p_value=round(mc_p_value, 4),
        mc_percentile=round(mc_percentile, 2),
        mc_n_shuffles=n_shuffles,
        ttest_p_value=round(ttest_p, 6),
        ttest_t_statistic=round(t_stat, 4),
        mean_trade_pnl=round(mean_pnl, 2),
        std_trade_pnl=round(std_pnl, 2),
        pf_first_half=round(pf1, 3),
        pf_second_half=round(pf2, 3),
        pf_stability_ratio=round(pf_stability, 3),
        top_trade_pct=round(top1_pct, 2),
        top3_trade_pct=round(top3_pct, 2),
        remove_best_pnl=round(remove_best, 2),
        remove_best_still_positive=remove_best_positive,
        wr_first_half=round(wr1, 2),
        wr_second_half=round(wr2, 2),
        wr_stability_ratio=round(wr_stability, 3),
        max_consecutive_wins=max_w,
        max_consecutive_losses=max_l,
        avg_win_streak=round(avg_w, 2),
        avg_loss_streak=round(avg_l, 2),
        avg_win_dollars=round(avg_win, 2),
        avg_loss_dollars=round(avg_loss, 2),
        win_loss_ratio=round(win_loss_ratio, 3),
        payoff_consistency=round(payoff_consistency, 3),
        trades_per_day=round(tpd, 3),
        trades_per_week=round(tpw, 3),
        expectancy_per_trade=round(expectancy, 2),
        kelly_fraction=round(kelly, 4),
    )
