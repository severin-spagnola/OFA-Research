"""
Penny Gapper Backtester — Pooled Ticker-Day Evaluation
=======================================================
Evaluates a PennyGenes candidate across pooled 1m bar data from
multiple ticker-days matching a profile. Each ticker-day is an
independent "setup" — the strategy either triggers or doesn't.

Supports 18 entry types, 4 SL types, 4 trail types, and 8 confirmations.

No forward-looking bias:
  - Entry on bar AFTER trigger confirmation (next bar open)
  - Stop/TP/trail checked on closed bars only
  - Indicators use only bars [0..current], never future bars
  - EOD forced exit at 15:55 ET
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from penny_genes import PennyGenes

RTH_START_TIME = pd.Timestamp("09:30").time()
RTH_END_TIME = pd.Timestamp("16:00").time()
EOD_EXIT_TIME = pd.Timestamp("15:55").time()


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION COST MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """Slippage, spread, and commission model for penny stocks.

    Costs are applied adversely:
      - Entry: price moves against you (buy higher, sell lower)
      - Exit: same adverse direction
      - Stops: additional gap-through slippage on top of adverse move
    """
    spread_pct: float = 0.15        # half-spread as % of price (applied each side)
    slippage_pct: float = 0.10      # market impact / fill slippage as % of price
    stop_extra_slip_pct: float = 0.20  # extra slippage on stop fills (gap-through)
    commission_per_share: float = 0.0  # per-share commission (0 for most brokers)
    min_commission: float = 0.0      # minimum commission per trade
    sec_fee_rate: float = 0.0        # SEC fee — broker-absorbed, not modeled
    taf_fee_per_share: float = 0.0    # FINRA TAF — broker-absorbed, not modeled


# Default conservative model for penny stocks
DEFAULT_COST_MODEL = CostModel()


def apply_entry_cost(price: float, is_long: bool, cost: CostModel) -> float:
    """Apply adverse slippage + spread to entry price."""
    slip = price * (cost.spread_pct + cost.slippage_pct) / 100
    return price + slip if is_long else price - slip


def apply_exit_cost(price: float, is_long: bool, cost: CostModel,
                    is_stop: bool = False) -> float:
    """Apply adverse slippage + spread to exit price.

    Stop exits get extra slippage to model gap-through.
    """
    slip_pct = cost.spread_pct + cost.slippage_pct
    if is_stop:
        slip_pct += cost.stop_extra_slip_pct
    slip = price * slip_pct / 100
    # Exit is opposite direction: long sells lower, short covers higher
    return price - slip if is_long else price + slip


def compute_trade_fees(entry_price: float, exit_price: float,
                       shares: float, is_long: bool,
                       cost: CostModel) -> float:
    """Compute total fees (commission + SEC + TAF) for a round-trip."""
    comm = max(shares * cost.commission_per_share * 2, cost.min_commission)
    # SEC fee on sell side only
    sell_value = exit_price * shares if is_long else entry_price * shares
    sec = sell_value * cost.sec_fee_rate
    taf = shares * cost.taf_fee_per_share
    return comm + sec + taf


# ═════════════════════════════════════════════════════════════════════════════
# TRADE RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    ticker: str
    date: str
    direction: str
    entry_type: str

    entry_price: float
    entry_time: str
    entry_bar_idx: int

    exit_price: float
    exit_time: str
    exit_bar_idx: int
    exit_reason: str

    risk: float
    risk_pct: float
    pnl_dollars: float
    pnl_r: float
    pnl_pct: float

    hold_bars: int
    hold_minutes: int

    mfe_pct: float
    mae_pct: float
    mfe_r: float
    mae_r: float

    gap_pct: float
    trigger_bar_idx: int  # bar where the entry condition triggered


@dataclass
class BacktestResult:
    """Aggregate result across all ticker-days for a gene config."""
    genes: PennyGenes
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_r > 0)

    @property
    def win_rate(self) -> float:
        return self.n_wins / max(self.n_trades, 1)

    @property
    def avg_r(self) -> float:
        return float(np.mean([t.pnl_r for t in self.trades])) if self.trades else 0

    @property
    def cum_r(self) -> float:
        return sum(t.pnl_r for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_r for t in self.trades if t.pnl_r > 0)
        gross_loss = abs(sum(t.pnl_r for t in self.trades if t.pnl_r < 0))
        return gross_win / max(gross_loss, 0.001)

    @property
    def max_dd_r(self) -> float:
        if not self.trades:
            return 0
        equity = np.cumsum([t.pnl_r for t in self.trades])
        peak = np.maximum.accumulate(equity)
        return float((peak - equity).max())

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 3:
            return 0
        rs = [t.pnl_r for t in self.trades]
        std = np.std(rs)
        if std < 0.001:
            return 0
        return float(np.mean(rs) / std * np.sqrt(len(rs)))

    def summary_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": round(self.win_rate * 100, 1),
            "avg_r": round(self.avg_r, 4),
            "cum_r": round(self.cum_r, 2),
            "profit_factor": round(self.profit_factor, 3),
            "max_dd_r": round(self.max_dd_r, 2),
            "sharpe": round(self.sharpe, 3),
        }


# ═════════════════════════════════════════════════════════════════════════════
# INDICATORS (computed incrementally, no lookahead)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Output[i] uses only closes[0..i]."""
    ema = np.empty_like(closes, dtype=float)
    ema[0] = closes[0]
    mult = 2.0 / (period + 1)
    for i in range(1, len(closes)):
        ema[i] = closes[i] * mult + ema[i - 1] * (1 - mult)
    return ema


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI. Output[i] uses only closes[0..i]. First `period` values are 50."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def compute_macd(closes: np.ndarray, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram. No lookahead."""
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_vwap(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Session VWAP (cumulative). No lookahead — uses typical price."""
    typical = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    return vwap


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Average True Range. Output[i] uses only bars[0..i]."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.empty(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ═════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE ALL INDICATORS FOR A TICKER-DAY
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BarContext:
    """Precomputed indicators and raw arrays for a ticker-day."""
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    timestamps: np.ndarray
    n: int

    # Indicators (lazily computed)
    vwap: np.ndarray = None
    atr_5: np.ndarray = None
    atr_10: np.ndarray = None
    atr_14: np.ndarray = None
    ema_5: np.ndarray = None
    ema_8: np.ndarray = None
    ema_13: np.ndarray = None
    ema_21: np.ndarray = None
    ema_34: np.ndarray = None
    rsi_5: np.ndarray = None
    rsi_7: np.ndarray = None
    rsi_14: np.ndarray = None
    macd_lines: dict = None  # keyed by (fast, slow, signal)


def build_context(rth: pd.DataFrame) -> BarContext:
    """Build BarContext with all precomputed indicators."""
    opens = rth["open"].values.astype(float)
    highs = rth["high"].values.astype(float)
    lows = rth["low"].values.astype(float)
    closes = rth["close"].values.astype(float)
    volumes = rth["volume"].values.astype(float)
    timestamps = rth["timestamp"].values

    ctx = BarContext(
        opens=opens, highs=highs, lows=lows, closes=closes,
        volumes=volumes, timestamps=timestamps, n=len(rth),
    )

    ctx.vwap = compute_vwap(opens, highs, lows, closes, volumes)
    ctx.atr_5 = compute_atr(highs, lows, closes, 5)
    ctx.atr_10 = compute_atr(highs, lows, closes, 10)
    ctx.atr_14 = compute_atr(highs, lows, closes, 14)

    ctx.ema_5 = compute_ema(closes, 5)
    ctx.ema_8 = compute_ema(closes, 8)
    ctx.ema_13 = compute_ema(closes, 13)
    ctx.ema_21 = compute_ema(closes, 21)
    ctx.ema_34 = compute_ema(closes, 34)

    ctx.rsi_5 = compute_rsi(closes, 5)
    ctx.rsi_7 = compute_rsi(closes, 7)
    ctx.rsi_14 = compute_rsi(closes, 14)

    ctx.macd_lines = {}
    for fast, slow, sig in [(5, 13, 5), (8, 21, 5), (12, 26, 9), (5, 13, 9)]:
        ml, sl, hist = compute_macd(closes, fast, slow, sig)
        ctx.macd_lines[(fast, slow, sig)] = (ml, sl, hist)

    return ctx


def get_ema(ctx: BarContext, period: int) -> np.ndarray:
    """Get EMA for a period, using precomputed or computing on-the-fly."""
    mapping = {5: ctx.ema_5, 8: ctx.ema_8, 13: ctx.ema_13,
               21: ctx.ema_21, 34: ctx.ema_34}
    if period in mapping:
        return mapping[period]
    return compute_ema(ctx.closes, period)


def get_rsi(ctx: BarContext, period: int) -> np.ndarray:
    mapping = {5: ctx.rsi_5, 7: ctx.rsi_7, 14: ctx.rsi_14}
    if period in mapping:
        return mapping[period]
    return compute_rsi(ctx.closes, period)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY TYPE DETECTORS
# Each returns (trigger_bar_idx, entry_bar_idx, stop_price, info_dict) or None
# trigger_bar_idx = bar where condition was confirmed (closed bar)
# entry_bar_idx = trigger_bar_idx + 1 + delay (next bar open)
# ═════════════════════════════════════════════════════════════════════════════

def _detect_ramp(ctx: BarContext, params: dict, genes: PennyGenes,
                 search_start: int) -> tuple | None:
    """Consecutive green bars from open."""
    min_bars = params.get("min_ramp_bars", 2)
    min_body = params.get("min_ramp_body_pct", 1.0)
    max_wick = params.get("max_wick_ratio", 1.0)

    ramp_start = search_start
    ramp_bars = 0
    body_total = 0.0
    wick_total = 0.0
    ramp_high = -np.inf
    ramp_low = np.inf

    for i in range(search_start, min(search_start + 20, ctx.n)):
        body = ctx.closes[i] - ctx.opens[i]
        if body <= 0:
            if ramp_bars > 0:
                break
            ramp_start = i + 1
            continue
        ramp_bars += 1
        body_total += body
        bar_range = ctx.highs[i] - ctx.lows[i]
        wick_total += max(bar_range - body, 0)
        ramp_high = max(ramp_high, ctx.highs[i])
        ramp_low = min(ramp_low, ctx.lows[i])

    if ramp_bars < min_bars:
        return None

    price = ctx.opens[ramp_start]
    if price <= 0:
        return None
    body_pct = body_total / price * 100
    wick_ratio = wick_total / max(body_total, 0.001)

    if body_pct < min_body or wick_ratio > max_wick:
        return None

    trigger_idx = ramp_start + ramp_bars - 1
    entry_idx = trigger_idx + 1 + genes.entry_delay_bars
    if entry_idx >= ctx.n:
        return None

    # SL anchor
    if genes.direction == "long":
        sl = ramp_low - price * genes.sl_buffer_pct / 100
    else:
        sl = ramp_high + price * genes.sl_buffer_pct / 100

    return trigger_idx, entry_idx, sl, {"ramp_bars": ramp_bars, "ramp_body_pct": body_pct}


def _detect_orb(ctx: BarContext, params: dict, genes: PennyGenes,
                search_start: int) -> tuple | None:
    """Opening Range Breakout."""
    orb_min = params.get("orb_minutes", 5)
    buffer_pct = params.get("orb_buffer_pct", 0)

    if ctx.n < orb_min + 2:
        return None

    # ORB range = first N bars
    orb_high = np.max(ctx.highs[:orb_min])
    orb_low = np.min(ctx.lows[:orb_min])
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None

    buffer = orb_high * buffer_pct / 100

    # Scan for breakout after ORB window
    for i in range(max(orb_min, search_start), ctx.n - 1):
        if genes.direction == "long" and ctx.closes[i] > orb_high + buffer:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = orb_low - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"orb_high": orb_high, "orb_low": orb_low}

        elif genes.direction == "short" and ctx.closes[i] < orb_low - buffer:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = orb_high + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"orb_high": orb_high, "orb_low": orb_low}

    return None


def _detect_vwap_cross(ctx: BarContext, params: dict, genes: PennyGenes,
                       search_start: int) -> tuple | None:
    """Price crosses VWAP."""
    cross_dir = params.get("cross_direction", "above")
    min_dist = params.get("min_distance_pct", 0)

    for i in range(max(1, search_start), ctx.n - 1):
        vwap = ctx.vwap[i]
        if vwap <= 0:
            continue
        dist_pct = (ctx.closes[i] - vwap) / vwap * 100

        if cross_dir == "above":
            # Was below VWAP, now above by min_distance
            if ctx.closes[i - 1] <= ctx.vwap[i - 1] and dist_pct >= min_dist:
                entry_idx = i + 1 + genes.entry_delay_bars
                if entry_idx >= ctx.n:
                    return None
                sl = vwap - ctx.opens[0] * genes.sl_buffer_pct / 100
                return i, entry_idx, sl, {"vwap": vwap}
        else:
            if ctx.closes[i - 1] >= ctx.vwap[i - 1] and dist_pct <= -min_dist:
                entry_idx = i + 1 + genes.entry_delay_bars
                if entry_idx >= ctx.n:
                    return None
                sl = vwap + ctx.opens[0] * genes.sl_buffer_pct / 100
                return i, entry_idx, sl, {"vwap": vwap}

    return None


def _detect_vwap_bounce(ctx: BarContext, params: dict, genes: PennyGenes,
                        search_start: int) -> tuple | None:
    """Price touches VWAP and bounces."""
    bounce_dir = params.get("bounce_direction", "long_from_below")
    prox = params.get("proximity_pct", 0.5)
    min_bounce = params.get("min_bounce_bars", 1)

    for i in range(max(2, search_start), ctx.n - 1):
        vwap = ctx.vwap[i]
        if vwap <= 0:
            continue
        dist = abs(ctx.lows[i] - vwap) / vwap * 100 if bounce_dir == "long_from_below" \
            else abs(ctx.highs[i] - vwap) / vwap * 100

        if dist > prox:
            continue

        # Check bounce: N bars closing in the right direction
        bounce_count = 0
        for j in range(max(0, i - min_bounce), i + 1):
            if bounce_dir == "long_from_below" and ctx.closes[j] > ctx.opens[j]:
                bounce_count += 1
            elif bounce_dir == "short_from_above" and ctx.closes[j] < ctx.opens[j]:
                bounce_count += 1

        if bounce_count >= min_bounce:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            if bounce_dir == "long_from_below":
                sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            else:
                sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"vwap": vwap}

    return None


def _detect_ema_cross(ctx: BarContext, params: dict, genes: PennyGenes,
                      search_start: int) -> tuple | None:
    """Fast EMA crosses slow EMA."""
    fast = get_ema(ctx, params.get("fast_period", 5))
    slow = get_ema(ctx, params.get("slow_period", 21))

    for i in range(max(1, search_start), ctx.n - 1):
        if genes.direction == "long":
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                entry_idx = i + 1 + genes.entry_delay_bars
                if entry_idx >= ctx.n:
                    return None
                sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
                return i, entry_idx, sl, {}
        else:
            if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
                entry_idx = i + 1 + genes.entry_delay_bars
                if entry_idx >= ctx.n:
                    return None
                sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
                return i, entry_idx, sl, {}

    return None


def _detect_ema_trend(ctx: BarContext, params: dict, genes: PennyGenes,
                      search_start: int) -> tuple | None:
    """Price above/below EMA with distance threshold."""
    ema = get_ema(ctx, params.get("ema_period", 13))
    pv = params.get("price_vs_ema", "above")
    min_dist = params.get("min_distance_pct", 0)

    for i in range(max(1, search_start), ctx.n - 1):
        if ema[i] <= 0:
            continue
        dist = (ctx.closes[i] - ema[i]) / ema[i] * 100

        if pv == "above" and dist >= min_dist:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = ema[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"ema": ema[i]}
        elif pv == "below" and dist <= -min_dist:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = ema[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"ema": ema[i]}

    return None


def _detect_pullback(ctx: BarContext, params: dict, genes: PennyGenes,
                     search_start: int) -> tuple | None:
    """Fib-style pullback after initial thrust."""
    min_move = params.get("min_initial_move_pct", 5.0)
    pb_depth = params.get("pullback_depth_pct", 50)
    pb_tol = params.get("pullback_tolerance_pct", 10)

    open_price = ctx.opens[0]
    if open_price <= 0:
        return None

    # Find initial thrust
    thrust_high_idx = -1
    thrust_high = open_price
    for i in range(search_start, ctx.n):
        if ctx.highs[i] > thrust_high:
            thrust_high = ctx.highs[i]
            thrust_high_idx = i
        move_pct = (thrust_high - open_price) / open_price * 100
        if move_pct >= min_move and thrust_high_idx >= 0:
            break
    else:
        return None

    if thrust_high_idx < 0:
        return None

    thrust_range = thrust_high - open_price
    if thrust_range <= 0:
        return None

    # Find pullback to target depth
    target_level = thrust_high - thrust_range * pb_depth / 100
    tol = thrust_range * pb_tol / 100

    for i in range(thrust_high_idx + 1, ctx.n - 1):
        if abs(ctx.lows[i] - target_level) <= tol or ctx.lows[i] <= target_level:
            # Check for bounce (close above open = green bar)
            if ctx.closes[i] > ctx.opens[i]:
                entry_idx = i + 1 + genes.entry_delay_bars
                if entry_idx >= ctx.n:
                    return None
                sl = target_level - tol - ctx.opens[0] * genes.sl_buffer_pct / 100
                return i, entry_idx, sl, {"thrust_high": thrust_high}

    return None


def _detect_macd_cross(ctx: BarContext, params: dict, genes: PennyGenes,
                       search_start: int) -> tuple | None:
    """MACD line crosses signal line."""
    fast = params.get("fast_period", 12)
    slow = params.get("slow_period", 26)
    sig = params.get("signal_period", 9)
    cross_type = params.get("cross_type", "bullish")

    key = (fast, slow, sig)
    if key in ctx.macd_lines:
        ml, sl, _ = ctx.macd_lines[key]
    else:
        ml, sl, _ = compute_macd(ctx.closes, fast, slow, sig)

    for i in range(max(1, search_start), ctx.n - 1):
        if cross_type == "bullish" and ml[i - 1] <= sl[i - 1] and ml[i] > sl[i]:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            stop = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, stop, {}
        elif cross_type == "bearish" and ml[i - 1] >= sl[i - 1] and ml[i] < sl[i]:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            stop = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, stop, {}

    return None


def _detect_rsi_extreme(ctx: BarContext, params: dict, genes: PennyGenes,
                        search_start: int) -> tuple | None:
    """RSI hits extreme level."""
    period = params.get("rsi_period", 14)
    entry_on = params.get("entry_on", "oversold")
    os_level = params.get("oversold_level", 30)
    ob_level = params.get("overbought_level", 70)

    rsi = get_rsi(ctx, period)

    for i in range(max(period + 1, search_start), ctx.n - 1):
        if entry_on == "oversold" and rsi[i] <= os_level:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"rsi": rsi[i]}
        elif entry_on == "overbought" and rsi[i] >= ob_level:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"rsi": rsi[i]}

    return None


def _detect_candle_pattern(ctx: BarContext, params: dict, genes: PennyGenes,
                           search_start: int) -> tuple | None:
    """Candlestick pattern recognition."""
    pattern = params.get("pattern", "engulfing_bull")
    min_body = params.get("min_body_pct", 0.5)

    for i in range(max(2, search_start), ctx.n - 1):
        price = ctx.opens[i]
        if price <= 0:
            continue
        body = abs(ctx.closes[i] - ctx.opens[i])
        body_pct = body / price * 100
        if body_pct < min_body and pattern not in ("doji_reversal", "inside_bar_break"):
            continue

        triggered = False
        prev_body = abs(ctx.closes[i - 1] - ctx.opens[i - 1])

        if pattern == "engulfing_bull":
            triggered = (ctx.closes[i - 1] < ctx.opens[i - 1] and  # prev red
                         ctx.closes[i] > ctx.opens[i] and           # current green
                         body > prev_body)                            # engulfs
        elif pattern == "engulfing_bear":
            triggered = (ctx.closes[i - 1] > ctx.opens[i - 1] and
                         ctx.closes[i] < ctx.opens[i] and
                         body > prev_body)
        elif pattern == "hammer":
            bar_range = ctx.highs[i] - ctx.lows[i]
            lower_wick = min(ctx.opens[i], ctx.closes[i]) - ctx.lows[i]
            triggered = bar_range > 0 and lower_wick / bar_range > 0.6 and body / bar_range < 0.3
        elif pattern == "shooting_star":
            bar_range = ctx.highs[i] - ctx.lows[i]
            upper_wick = ctx.highs[i] - max(ctx.opens[i], ctx.closes[i])
            triggered = bar_range > 0 and upper_wick / bar_range > 0.6 and body / bar_range < 0.3
        elif pattern == "doji_reversal":
            bar_range = ctx.highs[i] - ctx.lows[i]
            triggered = bar_range > 0 and body / bar_range < 0.1
        elif pattern == "three_soldiers":
            if i >= 2:
                triggered = all(ctx.closes[i - j] > ctx.opens[i - j] for j in range(3))
        elif pattern == "three_crows":
            if i >= 2:
                triggered = all(ctx.closes[i - j] < ctx.opens[i - j] for j in range(3))
        elif pattern == "inside_bar_break":
            if i >= 1:
                inside = (ctx.highs[i - 1] <= ctx.highs[i - 2] and
                          ctx.lows[i - 1] >= ctx.lows[i - 2]) if i >= 2 else False
                if inside:
                    if genes.direction == "long":
                        triggered = ctx.closes[i] > ctx.highs[i - 1]
                    else:
                        triggered = ctx.closes[i] < ctx.lows[i - 1]

        if triggered:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            if genes.direction == "long":
                sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            else:
                sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"pattern": pattern}

    return None


def _detect_volume_climax(ctx: BarContext, params: dict, genes: PennyGenes,
                          search_start: int) -> tuple | None:
    """Extreme volume spike."""
    mult = params.get("vol_multiple", 5.0)
    lookback = params.get("lookback_bars", 10)
    interp = params.get("interpretation", "momentum")

    for i in range(max(lookback, search_start), ctx.n - 1):
        avg_vol = np.mean(ctx.volumes[max(0, i - lookback):i])
        if avg_vol <= 0:
            continue
        ratio = ctx.volumes[i] / avg_vol
        if ratio < mult:
            continue

        entry_idx = i + 1 + genes.entry_delay_bars
        if entry_idx >= ctx.n:
            return None

        if interp == "momentum":
            # Trade in direction of the climax bar
            if ctx.closes[i] > ctx.opens[i]:  # green climax
                sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100
            else:
                sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
        else:  # exhaustion — fade
            if ctx.closes[i] > ctx.opens[i]:
                sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            else:
                sl = ctx.lows[i] - ctx.opens[0] * genes.sl_buffer_pct / 100

        return i, entry_idx, sl, {"vol_ratio": ratio}

    return None


def _detect_range_breakout(ctx: BarContext, params: dict, genes: PennyGenes,
                           search_start: int) -> tuple | None:
    """Break out of consolidation range."""
    lookback = params.get("lookback_bars", 10)
    break_dir = params.get("break_direction", "high")
    min_consol = params.get("min_consolidation_bars", 5)
    max_range_pct = params.get("max_range_pct", 5.0)

    for i in range(max(lookback + min_consol, search_start), ctx.n - 1):
        # Check if prior N bars were consolidation
        consol_high = np.max(ctx.highs[i - lookback:i])
        consol_low = np.min(ctx.lows[i - lookback:i])
        consol_range = consol_high - consol_low
        if ctx.opens[0] <= 0:
            continue
        range_pct = consol_range / ctx.opens[0] * 100
        if range_pct > max_range_pct or range_pct <= 0:
            continue

        if break_dir == "high" and ctx.closes[i] > consol_high:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = consol_low - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"consol_high": consol_high}
        elif break_dir == "low" and ctx.closes[i] < consol_low:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = consol_high + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"consol_low": consol_low}

    return None


def _detect_gap_fill_entry(ctx: BarContext, params: dict, genes: PennyGenes,
                           search_start: int, prev_close: float) -> tuple | None:
    """Fade gap back toward prev close."""
    min_gap = params.get("min_gap_pct", 50)
    wait_bars = params.get("wait_bars", 3)
    require_red = params.get("require_red_bar", False)

    if prev_close <= 0 or ctx.opens[0] <= 0:
        return None
    gap_pct = (ctx.opens[0] - prev_close) / prev_close * 100
    if gap_pct < min_gap:
        return None

    check_start = max(wait_bars, search_start)
    for i in range(check_start, ctx.n - 1):
        if require_red and ctx.closes[i] >= ctx.opens[i]:
            continue
        # Need a red bar (selling pressure) to confirm fade
        if ctx.closes[i] < ctx.opens[i]:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = ctx.highs[i] + ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"gap_pct": gap_pct, "prev_close": prev_close}

    return None


def _detect_hod_break(ctx: BarContext, params: dict, genes: PennyGenes,
                      search_start: int) -> tuple | None:
    """New high of day after HOD has aged N bars."""
    min_age = params.get("min_hod_age_bars", 10)
    margin = params.get("break_margin_pct", 0)

    hod = ctx.highs[0]
    hod_bar = 0

    for i in range(1, ctx.n - 1):
        if ctx.highs[i] > hod:
            age = i - hod_bar
            if age >= min_age and i >= search_start:
                buffer = hod * margin / 100
                if ctx.closes[i] > hod + buffer:
                    entry_idx = i + 1 + genes.entry_delay_bars
                    if entry_idx >= ctx.n:
                        return None
                    # Find recent swing low for SL
                    recent_low = np.min(ctx.lows[max(0, i - min_age):i + 1])
                    sl = recent_low - ctx.opens[0] * genes.sl_buffer_pct / 100
                    return i, entry_idx, sl, {"old_hod": hod, "new_hod": ctx.highs[i]}
            hod = ctx.highs[i]
            hod_bar = i

    return None


def _detect_first_red_fade(ctx: BarContext, params: dict, genes: PennyGenes,
                           search_start: int) -> tuple | None:
    """First red bar after N consecutive green bars."""
    min_run = params.get("min_green_run", 5)
    min_run_pct = params.get("min_run_pct", 5.0)

    green_count = 0
    run_start_price = ctx.opens[search_start] if search_start < ctx.n else 0

    for i in range(search_start, ctx.n - 1):
        if ctx.closes[i] > ctx.opens[i]:
            if green_count == 0:
                run_start_price = ctx.opens[i]
            green_count += 1
        else:
            if green_count >= min_run and run_start_price > 0:
                run_pct = (ctx.highs[i - 1] - run_start_price) / run_start_price * 100
                if run_pct >= min_run_pct:
                    entry_idx = i + 1 + genes.entry_delay_bars
                    if entry_idx >= ctx.n:
                        return None
                    sl = ctx.highs[i - 1] + ctx.opens[0] * genes.sl_buffer_pct / 100
                    return i, entry_idx, sl, {"green_run": green_count, "run_pct": run_pct}
            green_count = 0

    return None


def _detect_price_level(ctx: BarContext, params: dict, genes: PennyGenes,
                        search_start: int, prev_close: float) -> tuple | None:
    """Interaction with key price levels."""
    level_type = params.get("level_type", "whole_dollar")
    interaction = params.get("interaction", "break_above")
    prox = params.get("proximity_pct", 0.5)

    for i in range(max(1, search_start), ctx.n - 1):
        price = ctx.closes[i]
        if price <= 0:
            continue

        # Determine the reference level
        if level_type == "whole_dollar":
            level = round(price)
        elif level_type == "half_dollar":
            level = round(price * 2) / 2
        elif level_type == "prev_close":
            level = prev_close
        elif level_type == "day_open":
            level = ctx.opens[0]
        elif level_type == "premarket_high":
            # Approximate: highest price in first 2 bars
            level = max(ctx.highs[0], ctx.highs[1]) if ctx.n > 1 else ctx.highs[0]
        else:
            continue

        if level <= 0:
            continue
        dist = abs(price - level) / level * 100

        triggered = False
        if interaction == "break_above" and price > level and ctx.closes[i - 1] <= level:
            triggered = True
        elif interaction == "break_below" and price < level and ctx.closes[i - 1] >= level:
            triggered = True
        elif interaction == "bounce_from" and dist <= prox:
            # Check for bounce direction
            if genes.direction == "long" and ctx.closes[i] > ctx.opens[i]:
                triggered = True
            elif genes.direction == "short" and ctx.closes[i] < ctx.opens[i]:
                triggered = True

        if triggered:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            if genes.direction == "long":
                sl = level - ctx.opens[0] * (genes.sl_buffer_pct + prox) / 100
            else:
                sl = level + ctx.opens[0] * (genes.sl_buffer_pct + prox) / 100
            return i, entry_idx, sl, {"level": level, "level_type": level_type}

    return None


def _detect_momentum_accel(ctx: BarContext, params: dict, genes: PennyGenes,
                           search_start: int) -> tuple | None:
    """Momentum accelerating over consecutive windows."""
    window = params.get("window_bars", 5)
    min_windows = params.get("min_windows", 2)
    min_accel = params.get("min_accel_pct", 1.0)

    min_bars = window * (min_windows + 1)
    for i in range(max(min_bars, search_start), ctx.n - 1):
        # Compute returns for the last N windows
        rets = []
        for w in range(min_windows + 1):
            end = i - w * window
            start = end - window
            if start < 0:
                break
            ret = (ctx.closes[end] - ctx.closes[start]) / max(ctx.closes[start], 0.001) * 100
            rets.append(ret)

        if len(rets) < min_windows + 1:
            continue

        rets.reverse()  # oldest first

        # Check acceleration
        accel = True
        for j in range(1, len(rets)):
            if rets[j] - rets[j - 1] < min_accel:
                accel = False
                break

        if accel:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            sl = np.min(ctx.lows[i - window:i + 1]) - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"accel_windows": len(rets)}

    return None


def _detect_sustained_volume(ctx: BarContext, params: dict, genes: PennyGenes,
                             search_start: int) -> tuple | None:
    """Volume remains elevated for N consecutive bars."""
    min_ratio = params.get("min_avg_vol_ratio", 3.0)
    min_bars = params.get("min_sustained_bars", 10)

    # Baseline: average volume of first few bars (or we'd need prev day data)
    if ctx.n < min_bars + 5:
        return None

    # Use bar 0 volume as baseline reference (it's the open bar, usually high)
    # Better: use prev_day average but we don't have it per-bar
    baseline = ctx.volumes[0] if ctx.volumes[0] > 0 else 1

    consec = 0
    for i in range(search_start, ctx.n - 1):
        if ctx.volumes[i] >= baseline * min_ratio:
            consec += 1
        else:
            consec = 0

        if consec >= min_bars:
            entry_idx = i + 1 + genes.entry_delay_bars
            if entry_idx >= ctx.n:
                return None
            recent_low = np.min(ctx.lows[i - min_bars:i + 1])
            sl = recent_low - ctx.opens[0] * genes.sl_buffer_pct / 100
            return i, entry_idx, sl, {"sustained_bars": consec}

    return None


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY TYPE DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

ENTRY_DETECTORS = {
    "ramp": _detect_ramp,
    "orb": _detect_orb,
    "vwap_cross": _detect_vwap_cross,
    "vwap_bounce": _detect_vwap_bounce,
    "ema_cross": _detect_ema_cross,
    "ema_trend": _detect_ema_trend,
    "pullback": _detect_pullback,
    "macd_cross": _detect_macd_cross,
    "rsi_extreme": _detect_rsi_extreme,
    "candle_pattern": _detect_candle_pattern,
    "volume_climax": _detect_volume_climax,
    "range_breakout": _detect_range_breakout,
    "hod_break": _detect_hod_break,
    "first_red_fade": _detect_first_red_fade,
    "price_level": _detect_price_level,
    "momentum_accel": _detect_momentum_accel,
    "sustained_volume": _detect_sustained_volume,
}

# These detectors need prev_close as an extra arg
_NEEDS_PREV_CLOSE = {"gap_fill_entry", "price_level"}


def detect_entry(ctx: BarContext, genes: PennyGenes,
                 search_start: int, prev_close: float) -> tuple | None:
    """Dispatch to the appropriate entry detector."""
    et = genes.entry_type
    params = genes.entry_params

    if et == "gap_fill_entry":
        return _detect_gap_fill_entry(ctx, params, genes, search_start, prev_close)
    elif et == "price_level":
        return _detect_price_level(ctx, params, genes, search_start, prev_close)
    elif et in ENTRY_DETECTORS:
        return ENTRY_DETECTORS[et](ctx, params, genes, search_start)
    else:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CONFIRMATION CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_confirmation(ctx: BarContext, genes: PennyGenes,
                       trigger_idx: int) -> bool:
    """Check if confirmation condition is met at trigger bar."""
    ct = genes.confirmation_type
    cp = genes.confirmation_params

    if ct == "none":
        return True
    if trigger_idx < 1 or trigger_idx >= ctx.n:
        return False

    if ct == "green_bar":
        return ctx.closes[trigger_idx] > ctx.opens[trigger_idx]
    elif ct == "red_bar":
        return ctx.closes[trigger_idx] < ctx.opens[trigger_idx]
    elif ct == "volume_above":
        threshold = cp.get("vol_threshold", 3.0)
        lookback = min(trigger_idx, 10)
        if lookback < 1:
            return True
        avg_vol = np.mean(ctx.volumes[trigger_idx - lookback:trigger_idx])
        return ctx.volumes[trigger_idx] >= avg_vol * threshold
    elif ct == "ema_aligned":
        period = cp.get("ema_period", 13)
        alignment = cp.get("alignment", "price_above")
        ema = get_ema(ctx, period)
        if alignment == "price_above":
            return ctx.closes[trigger_idx] > ema[trigger_idx]
        else:
            return ctx.closes[trigger_idx] < ema[trigger_idx]
    elif ct == "volume_declining":
        bars = cp.get("decline_bars", 5)
        if trigger_idx < bars:
            return False
        vols = ctx.volumes[trigger_idx - bars:trigger_idx + 1]
        return all(vols[i] <= vols[i - 1] * 1.1 for i in range(1, len(vols)))
    elif ct == "higher_lows":
        bars = cp.get("lookback_bars", 3)
        if trigger_idx < bars:
            return False
        for j in range(trigger_idx - bars + 1, trigger_idx + 1):
            if ctx.lows[j] < ctx.lows[j - 1]:
                return False
        return True
    elif ct == "lower_highs":
        bars = cp.get("lookback_bars", 3)
        if trigger_idx < bars:
            return False
        for j in range(trigger_idx - bars + 1, trigger_idx + 1):
            if ctx.highs[j] > ctx.highs[j - 1]:
                return False
        return True

    return True


# ═════════════════════════════════════════════════════════════════════════════
# STOP LOSS COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_stop(ctx: BarContext, genes: PennyGenes,
                 entry_idx: int, entry_price: float,
                 detector_sl: float, trigger_idx: int = 0) -> float:
    """Compute stop price based on SL type gene.

    trigger_idx is the last bar whose data is known at decision time.
    ATR and swing lookback use trigger_idx to avoid lookahead.
    """
    sl_type = genes.sl_type
    is_long = genes.direction == "long"

    if sl_type == "ramp_low":
        # Use the detector's SL (already computed with ramp_low logic)
        return detector_sl

    elif sl_type == "fixed_pct":
        if is_long:
            return entry_price * (1 - genes.sl_fixed_pct / 100)
        else:
            return entry_price * (1 + genes.sl_fixed_pct / 100)

    elif sl_type == "atr_mult":
        # Use trigger_idx (last known bar) to avoid lookahead into entry bar
        atr_ref = trigger_idx if trigger_idx > 0 else max(entry_idx - 1, 0)
        # Use gene-specified ATR period
        atr_period = getattr(genes, "sl_atr_period", 14)
        if atr_period <= 5:
            atr_arr = ctx.atr_5
        elif atr_period <= 10:
            atr_arr = ctx.atr_10
        else:
            atr_arr = ctx.atr_14
        atr_val = atr_arr[atr_ref] if not np.isnan(atr_arr[atr_ref]) else 0
        if atr_val <= 0:
            # Fallback to fixed
            return entry_price * (1 - 0.05) if is_long else entry_price * (1 + 0.05)
        if is_long:
            return entry_price - atr_val * genes.sl_atr_mult
        else:
            return entry_price + atr_val * genes.sl_atr_mult

    elif sl_type == "swing_low":
        # Look back from trigger_idx (last known bar) to avoid entry bar lookahead
        ref = trigger_idx if trigger_idx > 0 else max(entry_idx - 1, 0)
        lookback = min(ref, 10)
        if lookback < 1:
            return detector_sl
        if is_long:
            swing = np.min(ctx.lows[ref - lookback:ref + 1])
            return swing - entry_price * genes.sl_buffer_pct / 100
        else:
            swing = np.max(ctx.highs[ref - lookback:ref + 1])
            return swing + entry_price * genes.sl_buffer_pct / 100

    return detector_sl


# ═════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION (handles all trail types)
# ═════════════════════════════════════════════════════════════════════════════

def simulate_trade(
    ctx: BarContext,
    genes: PennyGenes,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    trigger_idx: int,
    ticker: str,
    date: str,
    gap_pct: float,
    cost: CostModel | None = None,
) -> TradeResult | None:
    """Simulate trade from entry_idx forward with execution costs."""
    if cost is None:
        cost = DEFAULT_COST_MODEL

    is_long = genes.direction == "long"

    # Apply entry slippage
    entry_price = apply_entry_cost(entry_price, is_long, cost)

    if is_long:
        risk = entry_price - stop_price
    else:
        risk = stop_price - entry_price

    if risk <= 0 or entry_price <= 0:
        return None

    risk_pct = risk / entry_price * 100
    tp_price = entry_price + risk * genes.tp_r if is_long else entry_price - risk * genes.tp_r

    remaining_pct = 1.0
    partial_pnl_r = 0.0
    partial_done = False
    trail_active = False
    trail_stop = 0.0

    mfe_price = entry_price
    mae_price = entry_price

    exit_price = None
    exit_reason = None
    exit_idx = None

    for i in range(entry_idx, ctx.n):
        bar_ts = pd.Timestamp(ctx.timestamps[i])

        # MFE/MAE
        if is_long:
            mfe_price = max(mfe_price, ctx.highs[i])
            mae_price = min(mae_price, ctx.lows[i])
        else:
            mfe_price = min(mfe_price, ctx.lows[i])
            mae_price = max(mae_price, ctx.highs[i])

        # Stop loss (check FIRST — conservative, before EOD/time/TP)
        effective_stop = trail_stop if trail_active else stop_price
        if is_long and ctx.lows[i] <= effective_stop:
            # Gap-through: if bar opens below stop, fill at open (worse than stop)
            raw_fill = min(effective_stop, ctx.opens[i]) if ctx.opens[i] < effective_stop else effective_stop
            raw_fill = max(raw_fill, ctx.lows[i])  # can't fill below bar low
            exit_price = apply_exit_cost(raw_fill, is_long, cost, is_stop=True)
            exit_reason = "trail" if trail_active else "sl"
            exit_idx = i
            break
        elif not is_long and ctx.highs[i] >= effective_stop:
            raw_fill = max(effective_stop, ctx.opens[i]) if ctx.opens[i] > effective_stop else effective_stop
            raw_fill = min(raw_fill, ctx.highs[i])
            exit_price = apply_exit_cost(raw_fill, is_long, cost, is_stop=True)
            exit_reason = "trail" if trail_active else "sl"
            exit_idx = i
            break

        # EOD (after SL check — if stop hit on EOD bar, take the stop)
        if bar_ts.time() >= EOD_EXIT_TIME:
            exit_price = apply_exit_cost(ctx.closes[i], is_long, cost)
            exit_reason = "eod"
            exit_idx = i
            break

        # Time exit
        if (i - entry_idx) >= genes.max_hold_minutes:
            exit_price = apply_exit_cost(ctx.closes[i], is_long, cost)
            exit_reason = "time"
            exit_idx = i
            break

        # Partial exit (apply slippage to partial fill)
        if genes.partial_exit and not partial_done:
            partial_price = entry_price + risk * genes.partial_target_r if is_long \
                else entry_price - risk * genes.partial_target_r
            if (is_long and ctx.highs[i] >= partial_price) or \
               (not is_long and ctx.lows[i] <= partial_price):
                filled_partial = apply_exit_cost(partial_price, is_long, cost)
                partial_r = (filled_partial - entry_price) / risk if is_long \
                    else (entry_price - filled_partial) / risk
                partial_pnl_r = partial_r * genes.partial_pct
                remaining_pct -= genes.partial_pct
                partial_done = True

        # Take profit (apply slippage — favorable fills still get cost)
        if is_long and ctx.highs[i] >= tp_price:
            exit_price = apply_exit_cost(tp_price, is_long, cost)
            exit_reason = "tp"
            exit_idx = i
            break
        elif not is_long and ctx.lows[i] <= tp_price:
            exit_price = apply_exit_cost(tp_price, is_long, cost)
            exit_reason = "tp"
            exit_idx = i
            break

        # Trailing stop update
        if genes.trail_type != "none":
            current_r = ((ctx.closes[i] - entry_price) / risk) if is_long \
                else ((entry_price - ctx.closes[i]) / risk)

            if current_r >= genes.trail_activation_r:
                new_trail = None

                if genes.trail_type == "fixed_pct":
                    if is_long:
                        new_trail = ctx.closes[i] * (1 - genes.trail_distance_pct)
                    else:
                        new_trail = ctx.closes[i] * (1 + genes.trail_distance_pct)

                elif genes.trail_type == "bar_low":
                    new_trail = ctx.lows[i] if is_long else ctx.highs[i]

                elif genes.trail_type == "ema_trail":
                    ema = get_ema(ctx, genes.trail_ema_period)
                    new_trail = ema[i]

                if new_trail is not None:
                    if is_long:
                        if not trail_active or new_trail > trail_stop:
                            trail_stop = new_trail
                            trail_active = True
                    else:
                        if not trail_active or new_trail < trail_stop:
                            trail_stop = new_trail
                            trail_active = True

    if exit_price is None:
        exit_price = apply_exit_cost(ctx.closes[-1], is_long, cost)
        exit_reason = "eod"
        exit_idx = ctx.n - 1

    # P&L (entry_price already includes entry slippage from apply_entry_cost)
    raw_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)

    # Deduct round-trip fees (assume 100 share baseline for fee calc)
    fees = compute_trade_fees(entry_price, exit_price, 100, is_long, cost)
    fee_per_share = fees / 100
    raw_pnl -= fee_per_share

    if partial_done:
        total_pnl_r = partial_pnl_r + (raw_pnl / risk) * remaining_pct
    else:
        total_pnl_r = raw_pnl / risk

    pnl_pct = raw_pnl / entry_price * 100

    if is_long:
        mfe_pct = (mfe_price - entry_price) / entry_price * 100
        mae_pct = (mae_price - entry_price) / entry_price * 100
    else:
        mfe_pct = (entry_price - mfe_price) / entry_price * 100
        mae_pct = (entry_price - mae_price) / entry_price * 100

    return TradeResult(
        ticker=ticker, date=date, direction=genes.direction,
        entry_type=genes.entry_type,
        entry_price=round(entry_price, 4),
        entry_time=str(pd.Timestamp(ctx.timestamps[entry_idx])),
        entry_bar_idx=entry_idx,
        exit_price=round(exit_price, 4),
        exit_time=str(pd.Timestamp(ctx.timestamps[exit_idx])),
        exit_bar_idx=exit_idx,
        exit_reason=exit_reason,
        risk=round(risk, 4), risk_pct=round(risk_pct, 2),
        pnl_dollars=round(raw_pnl, 4), pnl_r=round(total_pnl_r, 4),
        pnl_pct=round(pnl_pct, 2),
        hold_bars=exit_idx - entry_idx, hold_minutes=exit_idx - entry_idx,
        mfe_pct=round(mfe_pct, 2), mae_pct=round(mae_pct, 2),
        mfe_r=round(mfe_pct / max(risk_pct, 0.001), 2),
        mae_r=round(mae_pct / max(risk_pct, 0.001), 2),
        gap_pct=round(gap_pct, 1),
        trigger_bar_idx=trigger_idx,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER-DAY EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ticker_day(
    bars: pd.DataFrame,
    genes: PennyGenes,
    ticker: str,
    date: str,
    gap_pct: float,
    prev_close: float,
    cost: CostModel | None = None,
) -> TradeResult | None:
    """Evaluate a gene config on a single ticker-day."""
    if bars.empty or len(bars) < 10:
        return None

    if bars["timestamp"].dt.tz is None:
        bars = bars.copy()
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("US/Eastern")

    rth = bars[
        (bars["timestamp"].dt.time >= RTH_START_TIME) &
        (bars["timestamp"].dt.time < RTH_END_TIME)
    ].copy().reset_index(drop=True)

    if len(rth) < 15:
        return None

    # Build context with all indicators
    ctx = build_context(rth)

    # Determine search start from timing genes
    search_start = 0
    if genes.earliest_entry_min > 0:
        first_ts = pd.Timestamp(ctx.timestamps[0])
        rth_open = first_ts.replace(hour=9, minute=30, second=0)
        target_time = rth_open + pd.Timedelta(minutes=genes.earliest_entry_min)
        for j in range(ctx.n):
            if pd.Timestamp(ctx.timestamps[j]) >= target_time:
                search_start = j
                break

    # Detect entry
    result = detect_entry(ctx, genes, search_start, prev_close)
    if result is None:
        return None

    trigger_idx, entry_idx, detector_sl, info = result

    # Check latest entry time
    entry_ts = pd.Timestamp(ctx.timestamps[entry_idx])
    first_ts = pd.Timestamp(ctx.timestamps[0])
    rth_open = first_ts.replace(hour=9, minute=30, second=0)
    max_entry_time = rth_open + pd.Timedelta(minutes=genes.latest_entry_min)
    if entry_ts > max_entry_time:
        return None

    # Check volume filter
    if genes.min_vol_spike > 1.0:
        lookback = min(trigger_idx, 10)
        if lookback > 0:
            avg_vol = np.mean(ctx.volumes[max(0, trigger_idx - lookback):trigger_idx])
            if avg_vol > 0 and ctx.volumes[trigger_idx] / avg_vol < genes.min_vol_spike:
                return None

    # Check volume acceleration filter
    if genes.require_vol_accel:
        if trigger_idx < 3:
            return None  # not enough bars to verify acceleration
        recent_vols = ctx.volumes[trigger_idx - 2:trigger_idx + 1]
        if not (recent_vols[1] >= recent_vols[0] and recent_vols[2] >= recent_vols[1]):
            return None

    # Check first bar return filter
    if genes.min_first_bar_ret_pct > 0 and ctx.n > 0:
        first_bar_ret = (ctx.closes[0] - ctx.opens[0]) / max(ctx.opens[0], 0.001) * 100
        if first_bar_ret < genes.min_first_bar_ret_pct:
            return None

    # Check spread filter
    if genes.max_spread_pct < 999:
        bar_range_pct = (ctx.highs[trigger_idx] - ctx.lows[trigger_idx]) / max(ctx.opens[trigger_idx], 0.001) * 100
        if bar_range_pct > genes.max_spread_pct:
            return None

    # Check confirmation
    if not check_confirmation(ctx, genes, trigger_idx):
        return None

    # Entry price
    entry_price = ctx.opens[entry_idx]
    if entry_price <= 0:
        return None

    # Compute final stop price
    stop_price = compute_stop(ctx, genes, entry_idx, entry_price, detector_sl, trigger_idx)
    if stop_price <= 0:
        return None

    # Simulate trade
    return simulate_trade(
        ctx=ctx, genes=genes,
        entry_idx=entry_idx, entry_price=entry_price,
        stop_price=stop_price, trigger_idx=trigger_idx,
        ticker=ticker, date=date, gap_pct=gap_pct,
        cost=cost,
    )


# ═════════════════════════════════════════════════════════════════════════════
# POOLED EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def preload_bar_data(
    ticker_days: list[dict],
    bar_cache_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load all bar cache parquet files into memory once.

    Returns dict keyed by "{ticker}_{date}" -> DataFrame.
    This avoids re-reading parquet files for every candidate evaluation.
    """
    bar_cache_dir = Path(bar_cache_dir)
    bar_data = {}
    n_missing = 0
    n_error = 0
    n_empty = 0
    first_error = None
    for td in ticker_days:
        key = f"{td['ticker']}_{td['date']}"
        cache_file = bar_cache_dir / f"{key}.parquet"
        if not cache_file.exists():
            n_missing += 1
            continue
        try:
            bars = pd.read_parquet(cache_file)
            if not bars.empty:
                bar_data[key] = bars
            else:
                n_empty += 1
        except Exception as e:
            n_error += 1
            if first_error is None:
                first_error = f"{key}: {e}"
            continue
    if n_missing or n_error or n_empty:
        print(f"[preload] missing={n_missing} errors={n_error} empty={n_empty} "
              f"loaded={len(bar_data)} first_error={first_error}", flush=True)
    return bar_data


def evaluate_pooled(
    genes: PennyGenes,
    ticker_days: list[dict],
    bar_cache_dir: str | Path,
    bar_data: dict[str, pd.DataFrame] | None = None,
    cost: CostModel | None = None,
) -> BacktestResult:
    """Evaluate a gene config across all ticker-days in a pool.

    If bar_data is provided (from preload_bar_data), uses in-memory data
    instead of reading parquet files from disk each time.
    """
    result = BacktestResult(genes=genes)
    bar_cache_dir = Path(bar_cache_dir)

    for td in ticker_days:
        ticker = td["ticker"]
        date = td["date"]
        key = f"{ticker}_{date}"

        if bar_data is not None:
            bars = bar_data.get(key)
            if bars is None:
                continue
        else:
            cache_file = bar_cache_dir / f"{key}.parquet"
            if not cache_file.exists():
                continue
            try:
                bars = pd.read_parquet(cache_file)
            except Exception:
                continue
            if bars.empty:
                continue

        trade = evaluate_ticker_day(
            bars=bars, genes=genes, ticker=ticker, date=date,
            gap_pct=td.get("gap_pct", 0), prev_close=td.get("prev_close", 0),
            cost=cost,
        )

        if trade is not None:
            result.trades.append(trade)

    return result
