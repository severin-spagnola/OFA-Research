"""
Strategy DSL — Rule-Based Strategy Evaluator
=============================================
Defines a simple JSON-serializable strategy format that the LLM generates
and the backtester evaluates. Strategies are data (rules), not code.

A strategy definition specifies:
  - Entry conditions (time window, price/candle/gap conditions)
  - Exit rules (SL, TP, time stop)
  - Filters (ON range, day-of-week)
  - Position sizing

The evaluator converts these rules into Trade objects using 1m candle data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import time, timedelta
from typing import Optional, TYPE_CHECKING

import pandas as pd

# ─── Backtester imports ───────────────────────────────────────────────────────
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "archived" / "fvg_gap"))

import mes_backtest as mb

if TYPE_CHECKING:
    from asset_config import AssetConfig
    from gex_features import GexContext, DailyContext


# ─── Constants (MES defaults, used when no AssetConfig provided) ─────────────

SLIPPAGE_PER_SIDE_PTS = 0.125
ROUND_TRIP_FEE_PER_CONTRACT = 1.24
MAX_CONTRACTS = 100
RTH_START = time(9, 30)
RTH_END = time(16, 0)


def _session_vwap(ctx) -> float:
    """Compute rolling VWAP from session start's 1m candles up to current 15m bar.

    Uses ctx.candles_1m / ctx.ts_index_1m. Returns 0.0 if insufficient data.
    For overnight sessions, uses the session_start from ctx if available.
    """
    cur_ts = ctx.current_15m.timestamp
    if hasattr(ctx, 'session_start_ts') and ctx.session_start_ts is not None:
        day_start = ctx.session_start_ts
    else:
        day_start = cur_ts.normalize().replace(hour=9, minute=30)
    lo = mb.bisect_left(ctx.ts_index_1m, day_start)
    # Include all 1m bars within current 15m candle (exclusive of next interval start)
    bar_end = cur_ts + pd.Timedelta(minutes=15)
    hi = mb.bisect_left(ctx.ts_index_1m, bar_end)
    bars = ctx.candles_1m[lo:hi]
    if not bars:
        return 0.0
    cum_pv = 0.0
    cum_v = 0.0
    for b in bars:
        cum_pv += b.close * b.volume
        cum_v += b.volume
    return cum_pv / cum_v if cum_v > 0 else 0.0


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class StrategyDefinition:
    """A complete strategy defined as JSON-serializable rules."""
    name: str
    description: str
    entry: dict          # {direction, time_window, conditions: [...]}
    exit: dict           # {stop_loss, take_profit, time_stop, partial}
    filters: list[dict] = field(default_factory=list)
    position_size: dict = field(default_factory=lambda: {"risk_dollars": 600})
    archetype: str = ""  # strategy archetype (e.g. "gex_momentum", "smc_sweep")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> StrategyDefinition:
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            entry=d.get("entry", {}),
            exit=d.get("exit", {}),
            filters=d.get("filters", []),
            position_size=d.get("position_size", {"risk_dollars": 600}),
            archetype=d.get("archetype", ""),
        )


@dataclass
class CandleContext:
    """Pre-computed context available to strategy rules at each 15m candle."""
    # Overnight levels
    on_high: Optional[float]
    on_low: Optional[float]
    on_range: float

    # Session context
    day_open: Optional[float]     # first RTH 15m bar open
    prev_close: Optional[float]   # previous day's last close

    # Current candle
    current_15m: mb.Candle
    bar_index: int                # 0-based within RTH session

    # Lookback
    recent_15m: list[mb.Candle]   # last N 15m candles (newest last)

    # Gaps
    gaps: list[mb.Gap]

    # 1m data for entry fill detection
    candles_1m: list[mb.Candle]
    ts_index_1m: list[pd.Timestamp]

    # Time
    time_et: time

    # Breakout tracking (set during evaluation)
    breakout_candle: Optional[mb.Candle] = None

    # GEX/options flow context (None for assets without chain data)
    gex: Optional["GexContext"] = None

    # Daily market context — VWAP, realized vol, earnings (None if unavailable)
    daily: Optional["DailyContext"] = None

    # Overnight session tracking (None for RTH)
    session_start_ts: Optional[pd.Timestamp] = None


# ─── Level Resolution ─────────────────────────────────────────────────────────

def _resolve_level(level_spec, ctx: CandleContext) -> Optional[float]:
    """Resolve a level reference to a float price."""
    if isinstance(level_spec, (int, float)):
        return float(level_spec)
    if level_spec == "on_high":
        return ctx.on_high
    if level_spec == "on_low":
        return ctx.on_low
    if level_spec == "day_open":
        return ctx.day_open
    if level_spec == "prev_close":
        return ctx.prev_close
    if level_spec == "on_mid":
        if ctx.on_high is not None and ctx.on_low is not None:
            return (ctx.on_high + ctx.on_low) / 2
        return None
    if level_spec == "current_open":
        return ctx.current_15m.open
    if level_spec == "current_close":
        return ctx.current_15m.close
    if level_spec == "current_high":
        return ctx.current_15m.high
    if level_spec == "current_low":
        return ctx.current_15m.low
    if level_spec == "breakout_candle_body_bottom":
        if ctx.breakout_candle:
            return ctx.breakout_candle.body_bottom
        return None
    if level_spec == "breakout_candle_body_top":
        if ctx.breakout_candle:
            return ctx.breakout_candle.body_top
        return None
    if level_spec == "prev_vwap":
        if ctx.daily is not None:
            return ctx.daily.prev_vwap
        return None
    return None


# ─── Condition Evaluation ─────────────────────────────────────────────────────

def evaluate_condition(cond: dict, ctx: CandleContext) -> bool:
    """Evaluate a single entry condition against the current context."""
    ctype = cond.get("type", "")

    if ctype == "price_above":
        level = _resolve_level(cond.get("level"), ctx)
        if level is None:
            return False
        return ctx.current_15m.close > level

    if ctype == "price_below":
        level = _resolve_level(cond.get("level"), ctx)
        if level is None:
            return False
        return ctx.current_15m.close < level

    if ctype == "price_broke":
        level = _resolve_level(cond.get("level"), ctx)
        if level is None:
            return False
        lookback = cond.get("lookback_bars", 2)
        candles = ctx.recent_15m[-lookback:] if ctx.recent_15m else []
        direction = cond.get("direction", "above")
        for c in candles:
            if direction == "above" and c.high > level:
                # Track breakout candle for pullback references
                ctx.breakout_candle = c
                return True
            if direction == "below" and c.low < level:
                ctx.breakout_candle = c
                return True
        # Auto-detect direction from level
        if direction not in ("above", "below"):
            for c in candles:
                if c.high > level:
                    ctx.breakout_candle = c
                    return True
                if c.low < level:
                    ctx.breakout_candle = c
                    return True
        return False

    if ctype == "pullback_to":
        ref = _resolve_level(cond.get("reference"), ctx)
        if ref is None:
            return False
        buffer = cond.get("buffer_pts", 3.0)
        price = ctx.current_15m.close
        return abs(price - ref) <= buffer

    if ctype == "candle_pattern":
        c = ctx.current_15m
        pattern = cond.get("pattern", "")
        if pattern == "is_bullish":
            return c.is_bullish
        if pattern == "is_bearish":
            return not c.is_bullish
        if pattern == "min_body_size":
            return c.body_size >= cond.get("size_pts", 5.0)
        if pattern == "max_body_size":
            return c.body_size <= cond.get("size_pts", 20.0)
        if pattern == "large_wick_up":
            wick = c.high - c.body_top
            return wick >= cond.get("size_pts", 5.0)
        if pattern == "large_wick_down":
            wick = c.body_bottom - c.low
            return wick >= cond.get("size_pts", 5.0)
        return False

    if ctype == "min_candle_size":
        return ctx.current_15m.body_size >= cond.get("size_pts", 5.0)

    if ctype == "gap_exists":
        location = cond.get("location", "any")
        max_dist = cond.get("max_distance_pts", 15.0)
        min_size = cond.get("min_size_pts", 4.0)
        for gap in ctx.gaps:
            if gap.size < min_size:
                continue
            if location == "any":
                return True
            gap_mid = (gap.gap_top + gap.gap_bottom) / 2
            if location == "near_on_high" and ctx.on_high is not None:
                if abs(gap_mid - ctx.on_high) <= max_dist:
                    return True
            if location == "near_on_low" and ctx.on_low is not None:
                if abs(gap_mid - ctx.on_low) <= max_dist:
                    return True
        return False

    if ctype == "time_is":
        # Check if current time is within a range
        after = cond.get("after", "00:00")
        before = cond.get("before", "23:59")
        t = ctx.time_et
        after_t = time(*map(int, after.split(":")))
        before_t = time(*map(int, before.split(":")))
        return after_t <= t <= before_t

    if ctype == "bar_index":
        idx = cond.get("value", 0)
        op = cond.get("op", "eq")
        if op == "eq":
            return ctx.bar_index == idx
        if op == "gte":
            return ctx.bar_index >= idx
        if op == "lte":
            return ctx.bar_index <= idx
        return False

    if ctype == "on_range_between":
        min_r = cond.get("min", 0)
        max_r = cond.get("max", 999)
        return min_r <= ctx.on_range <= max_r

    # ─── GEX / Options Flow Conditions ─────────────────────────────────────
    # All return False when ctx.gex is None (graceful degradation for
    # assets without chain data).

    if ctype == "gex_regime_is":
        if ctx.gex is None:
            return False
        return ctx.gex.gex_regime == cond.get("regime", "positive")

    if ctype == "price_near_wall":
        if ctx.gex is None:
            return False
        wall = cond.get("wall", "put")
        within_pts = cond.get("within_pts", 8.0)
        price = ctx.current_15m.close
        if wall == "call":
            return abs(price - ctx.gex.call_wall_strike) <= within_pts
        if wall == "put":
            return abs(price - ctx.gex.put_wall_strike) <= within_pts
        return False

    if ctype == "gex_zscore_above":
        if ctx.gex is None:
            return False
        return ctx.gex.gex_zscore >= cond.get("value", 1.0)

    if ctype == "gex_zscore_below":
        if ctx.gex is None:
            return False
        return ctx.gex.gex_zscore <= cond.get("value", -1.0)

    if ctype == "iv_above":
        if ctx.gex is None:
            return False
        return ctx.gex.atm_iv >= cond.get("value", 0.20)

    if ctype == "pc_ratio_above":
        if ctx.gex is None:
            return False
        return ctx.gex.pc_volume_ratio >= cond.get("value", 1.5)

    # ─── Daily Stats / Earnings Conditions ─────────────────────────────────
    # All return False when ctx.daily is None (graceful degradation).

    if ctype == "price_above_vwap":
        if ctx.daily is None or ctx.daily.prev_vwap <= 0:
            return False
        return ctx.current_15m.close > ctx.daily.prev_vwap

    if ctype == "price_below_vwap":
        if ctx.daily is None or ctx.daily.prev_vwap <= 0:
            return False
        return ctx.current_15m.close < ctx.daily.prev_vwap

    # ── Current-session VWAP fade: price extended away from today's rolling VWAP ──
    if ctype == "price_extended_above_vwap":
        vwap = _session_vwap(ctx)
        if vwap <= 0:
            return False
        threshold = cond.get("pts", 5.0)
        return (ctx.current_15m.close - vwap) >= threshold

    if ctype == "price_extended_below_vwap":
        vwap = _session_vwap(ctx)
        if vwap <= 0:
            return False
        threshold = cond.get("pts", 5.0)
        return (vwap - ctx.current_15m.close) >= threshold

    if ctype == "realized_vol_above":
        if ctx.daily is None:
            return False
        return ctx.daily.realized_vol_5m >= cond.get("value", 0.15)

    if ctype == "realized_vol_below":
        if ctx.daily is None:
            return False
        return ctx.daily.realized_vol_5m <= cond.get("value", 0.10)

    if ctype == "earnings_nearby":
        if ctx.daily is None:
            return False
        min_tickers = cond.get("min_tickers", 1)
        return len(ctx.daily.earnings_nearby) >= min_tickers

    if ctype == "no_earnings_nearby":
        if ctx.daily is None:
            return True  # no data = assume no earnings
        return len(ctx.daily.earnings_nearby) == 0

    # ─── Overnight-Specific Conditions ───────────────────────────────────

    if ctype == "session_vwap_cross":
        # Price crossed session VWAP (computed from session_start_ts)
        vwap = _session_vwap(ctx)
        if vwap <= 0:
            return False
        direction = cond.get("direction", "above")
        if direction == "above":
            return ctx.current_15m.close > vwap and ctx.current_15m.open <= vwap
        else:  # below
            return ctx.current_15m.close < vwap and ctx.current_15m.open >= vwap

    if ctype == "orb_breakout":
        # Opening range breakout — price broke the high/low of first N 15m bars
        n_bars = cond.get("n_bars", 2)  # first N 15m bars = first N*15 min
        if ctx.bar_index < n_bars:
            return False  # still within the opening range
        if len(ctx.recent_15m) < n_bars + 1:
            return False
        # recent_15m is newest-last; session bars start at -(bar_index+1)
        orb_start = max(0, len(ctx.recent_15m) - ctx.bar_index - 1)
        orb_bars = ctx.recent_15m[orb_start:orb_start + n_bars]
        if len(orb_bars) < n_bars:
            return False
        orb_high = max(b.high for b in orb_bars)
        orb_low = min(b.low for b in orb_bars)
        direction = cond.get("direction", "above")
        if direction == "above":
            return ctx.current_15m.close > orb_high
        else:
            return ctx.current_15m.close < orb_low

    if ctype == "reversion_to_session_open":
        # Price returned to session open after moving away by min_excursion
        if ctx.day_open is None:
            return False
        buffer = cond.get("buffer_pts", 2.0)
        min_excursion = cond.get("min_excursion_pts", 4.0)
        price = ctx.current_15m.close
        if len(ctx.recent_15m) < 3:
            return False
        max_high = max(b.high for b in ctx.recent_15m[:-1])
        min_low = min(b.low for b in ctx.recent_15m[:-1])
        moved_up = (max_high - ctx.day_open) >= min_excursion
        moved_down = (ctx.day_open - min_low) >= min_excursion
        near_open = abs(price - ctx.day_open) <= buffer
        return near_open and (moved_up or moved_down)

    if ctype == "consecutive_candles":
        # N consecutive same-direction candles (strict: close > open for bull, close < open for bear)
        n = cond.get("n", 3)
        direction = cond.get("direction", "bullish")
        # Require enough session-local bars (bar_index is 0-based within session)
        if ctx.bar_index + 1 < n:
            return False
        if len(ctx.recent_15m) < n:
            return False
        last_n = ctx.recent_15m[-n:]
        if direction == "bullish":
            return all(c.close > c.open for c in last_n)  # strict: excludes doji
        else:
            return all(c.close < c.open for c in last_n)  # strict: excludes doji

    if ctype == "body_range_ratio":
        # Candle body-to-range ratio (doji detection via low ratio, marubozu via high)
        c = ctx.current_15m
        candle_range = c.high - c.low
        if candle_range < 0.25:
            return False
        ratio = c.body_size / candle_range
        op = cond.get("op", "below")
        threshold = cond.get("threshold", 0.3)
        if op == "below":
            return ratio <= threshold
        else:
            return ratio >= threshold

    if ctype == "extended_from_session_extreme":
        # Price far from session running high/low (mean-reversion trigger)
        if len(ctx.recent_15m) < 3:
            return False
        session_start_idx = max(0, len(ctx.recent_15m) - ctx.bar_index - 1)
        session_bars = ctx.recent_15m[session_start_idx:]
        if len(session_bars) < 2:
            return False
        sess_high = max(b.high for b in session_bars)
        sess_low = min(b.low for b in session_bars)
        min_dist = cond.get("min_dist_pts", 3.0)
        extreme = cond.get("extreme", "high")
        price = ctx.current_15m.close
        if extreme == "high":
            return (sess_high - price) >= min_dist
        else:
            return (price - sess_low) >= min_dist

    if ctype == "bar_range_compression":
        # N consecutive narrow-range bars followed by expansion (squeeze → breakout)
        n = cond.get("n", 4)
        max_range_pts = cond.get("max_range_pts", 2.0)
        # Require enough session-local bars
        if ctx.bar_index < n:
            return False
        if len(ctx.recent_15m) < n + 1:
            return False
        narrow_bars = ctx.recent_15m[-(n + 1):-1]
        if len(narrow_bars) < n:
            return False
        all_narrow = all((b.high - b.low) <= max_range_pts for b in narrow_bars)
        current_range = ctx.current_15m.high - ctx.current_15m.low
        return all_narrow and current_range > max_range_pts

    if ctype == "rth_close_bias":
        # RTH closed near its high or low — overnight directional bias
        if ctx.prev_close is None or ctx.on_high is None or ctx.on_low is None:
            return False
        rth_range = ctx.on_high - ctx.on_low
        if rth_range < 1.0:
            return False
        close_pct = (ctx.prev_close - ctx.on_low) / rth_range
        bias = cond.get("bias", "bullish")
        threshold = cond.get("threshold", 0.75)
        if bias == "bullish":
            return close_pct >= threshold
        else:
            return close_pct <= (1.0 - threshold)

    # Unknown condition type — reject (don't silently pass)
    return False


# ─── Entry Computation ────────────────────────────────────────────────────────

def _parse_time(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


def _determine_direction(strategy: StrategyDefinition, ctx: CandleContext) -> Optional[str]:
    """Determine trade direction from strategy or context."""
    direction = strategy.entry.get("direction", "adaptive")

    if direction in ("long", "short"):
        return direction

    # Adaptive: infer from conditions
    conditions = strategy.entry.get("conditions", [])
    for cond in conditions:
        if cond.get("type") == "price_broke":
            level_name = cond.get("level", "")
            if level_name == "on_high":
                return "long"
            if level_name == "on_low":
                return "short"
        if cond.get("type") == "gap_exists":
            loc = cond.get("location", "")
            if loc == "near_on_high":
                return "short"  # fade gap near resistance
            if loc == "near_on_low":
                return "long"   # fade gap near support

    # Fallback: use price vs ON midpoint
    if ctx.on_high is not None and ctx.on_low is not None:
        mid = (ctx.on_high + ctx.on_low) / 2
        return "long" if ctx.current_15m.close > mid else "short"

    return None


def _compute_sl_tp(
    strategy: StrategyDefinition,
    direction: str,
    entry_price: float,
    ctx: CandleContext,
    asset_config: AssetConfig | None = None,
) -> tuple[float, float]:
    """Compute stop loss and take profit from exit rules."""
    # SL clamp bounds from asset config (defaults = MES)
    sl_min = asset_config.sl_min if asset_config else 4.0
    sl_max = asset_config.sl_max if asset_config else 32.0
    default_sl = asset_config.default_sl if asset_config else 12.0

    exit_rules = strategy.exit
    sl_spec = exit_rules.get("stop_loss", {"type": "fixed_pts", "value": default_sl})
    tp_spec = exit_rules.get("take_profit", {"type": "risk_multiple", "value": 1.0})

    # Stop loss
    sl_type = sl_spec.get("type", "fixed_pts")
    if sl_type == "fixed_pts":
        sl_pts = sl_spec.get("value", default_sl)
        sl = entry_price - sl_pts if direction == "long" else entry_price + sl_pts
    elif sl_type == "candle_wick":
        buffer = sl_spec.get("buffer_pts", 3.0)
        if direction == "long":
            sl = ctx.current_15m.low - buffer
        else:
            sl = ctx.current_15m.high + buffer
    elif sl_type == "level":
        level = _resolve_level(sl_spec.get("value"), ctx)
        buffer = sl_spec.get("buffer_pts", 2.0)
        if level is None:
            sl = entry_price - default_sl if direction == "long" else entry_price + default_sl
        else:
            sl = level - buffer if direction == "long" else level + buffer
    else:
        sl = entry_price - default_sl if direction == "long" else entry_price + default_sl

    risk = abs(entry_price - sl)
    # Enforce minimum SL distance
    if risk < sl_min:
        fallback_sl = max(sl_min * 2, default_sl)
        sl = entry_price - fallback_sl if direction == "long" else entry_price + fallback_sl
        risk = fallback_sl
    # Cap maximum SL
    if risk > sl_max:
        sl = entry_price - sl_max if direction == "long" else entry_price + sl_max
        risk = sl_max

    # Take profit
    tp_type = tp_spec.get("type", "risk_multiple")
    if tp_type == "risk_multiple":
        mult = tp_spec.get("value", 1.0)
        tp = entry_price + risk * mult if direction == "long" else entry_price - risk * mult
    elif tp_type == "fixed_pts":
        tp_pts = tp_spec.get("value", 12)
        tp = entry_price + tp_pts if direction == "long" else entry_price - tp_pts
    elif tp_type == "level":
        level = _resolve_level(tp_spec.get("value"), ctx)
        if level is None:
            tp = entry_price + risk if direction == "long" else entry_price - risk
        else:
            tp = level
    else:
        tp = entry_price + risk if direction == "long" else entry_price - risk

    return sl, tp


def compute_entry(
    strategy: StrategyDefinition,
    ctx: CandleContext,
    asset_config: AssetConfig | None = None,
) -> Optional[mb.Trade]:
    """Evaluate strategy rules against context. Returns Trade or None."""
    entry = strategy.entry

    # Check time window (handles overnight windows that cross midnight)
    tw = entry.get("time_window", {})
    tw_start = _parse_time(tw.get("start", "09:30"))
    tw_end = _parse_time(tw.get("end", "15:45"))

    # Also enforce time_stop as entry cutoff — no new entries at or after time_stop
    # (entry at time_stop bar would fill next bar, potentially outside session)
    time_stop_str = strategy.exit.get("time_stop")
    entry_cutoff = _parse_time(time_stop_str) if time_stop_str else tw_end

    if tw_start <= tw_end:
        # Normal window (e.g. 09:30-15:45)
        if ctx.time_et < tw_start or ctx.time_et >= entry_cutoff:
            return None
    else:
        # Overnight window crossing midnight (e.g. 18:00-02:00)
        if ctx.time_et < tw_start and ctx.time_et >= entry_cutoff:
            return None

    # Evaluate all conditions (AND logic)
    conditions = entry.get("conditions", [])
    for cond in conditions:
        if not evaluate_condition(cond, ctx):
            return None

    # Determine direction
    direction = _determine_direction(strategy, ctx)
    if direction is None:
        return None

    # Signal on current 15m candle close → fill at NEXT 15m candle's first 1m bar.
    # This avoids look-ahead bias: we decide based on the completed 15m candle,
    # then enter at the open of the first 1m bar in the following 15m window.
    next_candle_start = ctx.current_15m.timestamp + timedelta(minutes=15)
    next_candle_end = next_candle_start + timedelta(minutes=15)
    lo = mb.bisect_left(ctx.ts_index_1m, next_candle_start)
    hi = mb.bisect_left(ctx.ts_index_1m, next_candle_end)
    bars = ctx.candles_1m[lo:hi]

    if not bars:
        return None

    # Fill at first 1m bar open in the next 15m window
    entry_price = bars[0].open
    entry_time = bars[0].timestamp

    # Compute SL and TP based on actual entry price
    sl, tp = _compute_sl_tp(strategy, direction, entry_price, ctx, asset_config)

    # Minimum risk threshold (scaled by tick size)
    min_risk = asset_config.tick_size * 4 if asset_config else 1.0
    risk = abs(entry_price - sl)
    if risk < min_risk:
        return None

    return mb.Trade(
        model=0,  # 0 = LLM-generated strategy
        direction=direction,
        entry_price=entry_price,
        stop_loss=sl,
        take_profit=tp,
        entry_time=entry_time,
        notes=f"[{strategy.name}]",
    )


# ─── Trade Simulation ─────────────────────────────────────────────────────────

def simulate_trade_simple(
    trade: mb.Trade,
    candles_1m: list[mb.Candle],
    ts_index_1m: list[pd.Timestamp],
    asset_config: AssetConfig | None = None,
    exit_params: dict | None = None,
    session_end_time: time | None = None,
) -> mb.Trade:
    """Simulate trade exit with costs. Simpler than the friend strategy version —
    no partial fills, just SL/TP/EOD exit.
    exit_params optionally contains be_trigger_pts and trail_distance_pts for
    breakeven / trailing stop logic.
    session_end_time overrides the default 16:00 EOD exit (used for overnight sessions)."""
    # Asset-specific constants (defaults = MES)
    multiplier = asset_config.multiplier if asset_config else mb.MES_MULTIPLIER
    slippage = asset_config.slippage_per_side if asset_config else SLIPPAGE_PER_SIDE_PTS
    fee = asset_config.round_trip_fee if asset_config else ROUND_TRIP_FEE_PER_CONTRACT
    max_pos = asset_config.max_contracts if asset_config else MAX_CONTRACTS
    risk_budget = asset_config.risk_budget if asset_config else 600.0

    risk = abs(trade.entry_price - trade.stop_loss)
    contracts = min(max_pos, max(1, int(risk_budget / (risk * multiplier))))
    trade.num_contracts = contracts

    # TSL / Breakeven params (0 = disabled)
    be_trigger = exit_params.get("be_trigger_pts", 0) if exit_params else 0
    trail_dist = exit_params.get("trail_distance_pts", 0) if exit_params else 0
    watermark = trade.entry_price
    be_moved = False

    # Include entry bar in exit simulation — adverse moves on the entry
    # minute should be modeled (use bisect_left, not bisect_right)
    lo = mb.bisect_left(ts_index_1m, trade.entry_time)

    # Compute absolute session end timestamp for overnight-safe horizon
    _eod_time_val = session_end_time or time(16, 0)
    if session_end_time and session_end_time < time(12, 0):
        # Overnight session ending after midnight (e.g. 01:45, 07:45, 09:15)
        # Session end is on the NEXT calendar day from entry
        session_end_ts = (trade.entry_time.normalize() + timedelta(days=1)).replace(
            hour=_eod_time_val.hour, minute=_eod_time_val.minute)
        # Extend horizon to cover full overnight session (entry day + next day)
        eod = trade.entry_time.normalize() + timedelta(days=2)
    else:
        # RTH or same-day session (e.g. 22:45 for early Asia)
        session_end_ts = trade.entry_time.normalize().replace(
            hour=_eod_time_val.hour, minute=_eod_time_val.minute)
        eod = trade.entry_time.normalize() + timedelta(days=1)
    hi = mb.bisect_left(ts_index_1m, eod)
    future = candles_1m[lo:hi]

    # Enhanced instrumentation init
    _mfe_bar = 0
    _mae_bar = 0
    _mfbm_determined = False
    _mfe_before_mae = None

    for bar_idx, c in enumerate(future):
        # ── TSL / Breakeven adjustment (before SL/TP checks) ──
        if be_trigger:
            if trade.direction == "long":
                watermark = max(watermark, c.high)
                fav = watermark - trade.entry_price
                if fav >= be_trigger:
                    if not be_moved:
                        be_level = trade.entry_price + 1.0
                        trade.stop_loss = max(trade.stop_loss, be_level)
                        be_moved = True
                        trade.notes += " [BE]"
                    if trail_dist:
                        trail_level = watermark - trail_dist
                        if trail_level > trade.stop_loss:
                            trade.stop_loss = trail_level
                            if "[TSL]" not in trade.notes:
                                trade.notes += " [TSL]"
            else:  # short
                watermark = min(watermark, c.low)
                fav = trade.entry_price - watermark
                if fav >= be_trigger:
                    if not be_moved:
                        be_level = trade.entry_price - 1.0
                        trade.stop_loss = min(trade.stop_loss, be_level)
                        be_moved = True
                        trade.notes += " [BE]"
                    if trail_dist:
                        trail_level = watermark + trail_dist
                        if trail_level < trade.stop_loss:
                            trade.stop_loss = trail_level
                            if "[TSL]" not in trade.notes:
                                trade.notes += " [TSL]"

        # ── MFE/MAE tracking (passive instrumentation) ──
        if trade.direction == "long":
            favorable = c.high - trade.entry_price
            adverse = trade.entry_price - c.low
        else:
            favorable = trade.entry_price - c.low
            adverse = c.high - trade.entry_price

        if favorable > trade.mfe_pts:
            trade.mfe_pts = favorable
            _mfe_bar = bar_idx
        if adverse > trade.mae_pts:
            trade.mae_pts = adverse
            _mae_bar = bar_idx

        # Determine mfe_before_mae — which came first past 1pt threshold
        if not _mfbm_determined:
            if favorable >= 1.0 and trade.mfe_pts > trade.mae_pts:
                _mfe_before_mae = True
                _mfbm_determined = True
            elif adverse >= 1.0 and trade.mae_pts > trade.mfe_pts:
                _mfe_before_mae = False
                _mfbm_determined = True

        if trade.direction == "long":
            if c.low <= trade.stop_loss:
                # Gap-through: fill at worse of stop or bar open
                trade.exit_price = min(trade.stop_loss, c.open)
                trade.exit_time = c.timestamp
                break
            if c.high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = c.timestamp
                break
        else:
            if c.high >= trade.stop_loss:
                # Gap-through: fill at worse of stop or bar open
                trade.exit_price = max(trade.stop_loss, c.open)
                trade.exit_time = c.timestamp
                break
            if c.low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = c.timestamp
                break

        # EOD / session-end exit (uses absolute timestamp for overnight safety)
        if c.timestamp >= session_end_ts:
            trade.exit_price = c.close
            trade.exit_time = c.timestamp
            trade.notes += " [EOD exit]"
            break

    # Finalize enhanced instrumentation
    trade.bars_to_mfe = _mfe_bar
    trade.bars_to_mae = _mae_bar
    trade.mfe_before_mae = _mfe_before_mae

    # Data end exit
    if trade.exit_price is None and future:
        last = future[-1]
        trade.exit_price = last.close
        trade.exit_time = last.timestamp
        trade.notes += " [data end exit]"

    if trade.exit_price is None:
        trade.exit_price = trade.entry_price
        trade.exit_time = trade.entry_time
        trade.pnl_dollars = 0.0
        trade.pnl = 0.0
        trade.result = "no_fill"
        return trade

    # P&L calculation
    if trade.direction == "long":
        pts = trade.exit_price - trade.entry_price
    else:
        pts = trade.entry_price - trade.exit_price

    gross_dollars = pts * contracts * multiplier
    slippage_dollars = contracts * (2.0 * slippage * multiplier)
    fee_dollars = contracts * fee
    net_dollars = gross_dollars - slippage_dollars - fee_dollars

    trade.pnl_dollars = round(net_dollars, 2)
    trade.pnl = round(pts, 4)
    trade.result = "win" if trade.pnl_dollars > 0 else "loss"
    return trade


# ─── Check Filters ────────────────────────────────────────────────────────────

def check_filters(
    filters: list[dict],
    on_high: Optional[float],
    on_low: Optional[float],
    trade_date: pd.Timestamp,
) -> bool:
    """Check per-day filters. Returns True if day passes all filters."""
    on_range = (on_high - on_low) if (on_high is not None and on_low is not None) else 0

    for f in filters:
        ftype = f.get("type", "")

        if ftype == "min_on_range":
            if on_range < f.get("value", 0):
                return False

        elif ftype == "max_on_range":
            if on_range > f.get("value", 999):
                return False

        elif ftype == "day_of_week":
            allowed = f.get("days", [0, 1, 2, 3, 4])  # Mon=0, Fri=4
            if trade_date.weekday() not in allowed:
                return False

    return True


# ─── Main Strategy Runner ─────────────────────────────────────────────────────

def _build_overnight_sessions(
    candles_15m: list[mb.Candle],
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build overnight session boundaries from 15m candle data.

    An overnight session runs from 18:00 ET (day N) to 09:30 ET (day N+1).
    Returns list of (session_start, session_end) timestamps.
    """
    tz = candles_15m[0].timestamp.tzinfo if candles_15m else None

    # Find all unique calendar dates with overnight data (18:00+ or before 09:30)
    overnight_dates = sorted(set(
        c.timestamp.normalize() for c in candles_15m
        if c.timestamp.time() >= time(18, 0) or c.timestamp.time() < time(9, 30)
    ))

    if start_date and tz:
        sd = pd.Timestamp(start_date, tz=tz)
        overnight_dates = [d for d in overnight_dates if d >= sd]
    if end_date and tz:
        ed = pd.Timestamp(end_date, tz=tz)
        overnight_dates = [d for d in overnight_dates if d < ed]

    # Group into sessions: each session starts at 18:00 on a date that has
    # evening candles and ends at 09:30 the next morning
    sessions = []
    seen = set()
    for d in overnight_dates:
        session_start = d.replace(hour=18, minute=0)
        if session_start in seen:
            continue
        seen.add(session_start)
        session_end = (d + pd.Timedelta(days=1)).replace(hour=9, minute=30)
        # Verify there's actually data in this session
        has_data = any(
            session_start <= c.timestamp < session_end
            for c in candles_15m
            if abs((c.timestamp - session_start).total_seconds()) < 16 * 3600
        )
        if has_data:
            sessions.append((session_start, session_end))

    return sessions


def run_strategy(
    strategy: StrategyDefinition,
    candles_1m: list[mb.Candle],
    candles_15m: list[mb.Candle],
    candles_1h: list[mb.Candle],
    ts_index_1m: list[pd.Timestamp] = None,
    precomputed: tuple = None,
    start_date: str = None,
    end_date: str = None,
    asset_config: AssetConfig | None = None,
    gex_cache=None,
    daily_cache=None,
    collect_uncensored: bool = False,
    overnight: bool = False,
) -> list[mb.Trade]:
    """Run a strategy definition against candle data, producing Trade objects.

    This is the main entry point — equivalent to run_variant() in the old system
    but driven by JSON rules instead of hardcoded M1/M2/M3 logic.

    When overnight=True, iterates over overnight sessions (18:00→09:30) instead
    of RTH sessions (09:30→16:00). Reference levels are redefined:
    - day_open = first candle of overnight session
    - prev_close = prior RTH session's last close
    - on_high/on_low = prior RTH session's high/low (the "daytime range")
    """
    if ts_index_1m is None:
        ts_index_1m = [c.timestamp for c in candles_1m]

    if overnight:
        return _run_strategy_overnight(
            strategy, candles_1m, candles_15m, candles_1h,
            ts_index_1m, precomputed, start_date, end_date,
            asset_config, collect_uncensored,
        )

    # Build trading days from 15m candles (weekdays only, RTH)
    trading_days = sorted(set(
        c.timestamp.normalize() for c in candles_15m
        if c.timestamp.time() >= RTH_START
    ))

    if start_date:
        sd = pd.Timestamp(start_date, tz=candles_15m[0].timestamp.tzinfo)
        trading_days = [d for d in trading_days if d >= sd]
    if end_date:
        ed = pd.Timestamp(end_date, tz=candles_15m[0].timestamp.tzinfo)
        trading_days = [d for d in trading_days if d < ed]

    all_trades = []
    prev_close = None
    max_trades_per_day = 1  # prop firm rule
    _collect_uncensored = collect_uncensored

    for day in trading_days:
        # Get overnight H/L (equities use previous day's RTH high/low)
        if asset_config and asset_config.overnight_session == "prev_rth":
            from equity_data import get_prev_day_high_low
            on_high, on_low = get_prev_day_high_low(candles_1m, day, ts_index_1m)
        else:
            on_high, on_low = mb.get_overnight_high_low(candles_1m, day, ts_index_1m)
        on_range = (on_high - on_low) if (on_high is not None and on_low is not None) else 0

        # Check filters
        if not check_filters(strategy.filters, on_high, on_low, day):
            continue

        # Get 15m candles for this day (RTH only)
        day_start = day.replace(hour=RTH_START.hour, minute=RTH_START.minute)
        day_end = day.replace(hour=RTH_END.hour, minute=RTH_END.minute)
        day_candles = [
            c for c in candles_15m
            if day_start <= c.timestamp < day_end
        ]
        if not day_candles:
            continue

        day_open = day_candles[0].open
        trades_today = 0
        in_trade = False

        for bar_idx, candle in enumerate(day_candles):
            if trades_today >= max_trades_per_day:
                break
            if in_trade:
                continue

            # Get gaps at this timestamp
            gaps = []
            if precomputed:
                gaps = mb.find_gaps_at_time_fast(
                    precomputed[0], precomputed[1], candle.timestamp,
                    lookback_days=7,
                )

            # Build recent 15m candles for lookback
            candle_idx = candles_15m.index(candle) if candle in candles_15m else -1
            lookback_n = 10
            if candle_idx >= 0:
                recent = candles_15m[max(0, candle_idx - lookback_n):candle_idx + 1]
            else:
                recent = [candle]

            # GEX context (None for assets without chain data)
            gex_ctx = None
            if gex_cache is not None:
                gex_ctx = gex_cache.get(day.date(), candle.timestamp.time())

            # Daily context (VWAP, realized vol, earnings)
            daily_ctx = None
            if daily_cache is not None:
                daily_ctx = daily_cache.get(day.date())

            # Build context
            ctx = CandleContext(
                on_high=on_high,
                on_low=on_low,
                on_range=on_range,
                day_open=day_open,
                prev_close=prev_close,
                current_15m=candle,
                bar_index=bar_idx,
                recent_15m=recent,
                gaps=gaps,
                candles_1m=candles_1m,
                ts_index_1m=ts_index_1m,
                time_et=candle.timestamp.time(),
                gex=gex_ctx,
                daily=daily_ctx,
            )

            # Try entry
            trade = compute_entry(strategy, ctx, asset_config)
            if trade is None:
                continue

            # Simulate exit
            trade = simulate_trade_simple(trade, candles_1m, ts_index_1m, asset_config,
                                           exit_params=strategy.exit)

            if trade.result != "no_fill":
                all_trades.append(trade)
                trades_today += 1
                in_trade = True  # only 1 position at a time

                # Uncensored MFE/MAE collection (gated by caller)
                if _collect_uncensored:
                    from overfit_search import simulate_trade_uncensored, log_uncensored_candidate
                    unc = simulate_trade_uncensored(
                        candles_1m, ts_index_1m,
                        trade.entry_time, trade.entry_price, trade.direction,
                    )
                    log_uncensored_candidate(
                        candidate_id=strategy.name,
                        archetype=strategy.archetype,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        uncensored=unc,
                    )

        # Track prev close for next day
        if day_candles:
            prev_close = day_candles[-1].close

    return all_trades


def _run_strategy_overnight(
    strategy: StrategyDefinition,
    candles_1m: list[mb.Candle],
    candles_15m: list[mb.Candle],
    candles_1h: list[mb.Candle],
    ts_index_1m: list[pd.Timestamp],
    precomputed: tuple,
    start_date: str | None,
    end_date: str | None,
    asset_config: AssetConfig | None,
    collect_uncensored: bool,
) -> list[mb.Trade]:
    """Run strategy over overnight sessions (18:00 ET → 09:30 ET).

    Reference levels are inverted from RTH:
    - on_high/on_low = prior RTH session high/low (the "daytime range" acts as
      reference levels for overnight, just like overnight range is reference for RTH)
    - day_open = first candle of the overnight session
    - prev_close = last RTH close
    """
    sessions = _build_overnight_sessions(candles_15m, start_date, end_date)

    all_trades = []
    prev_rth_close = None
    max_trades_per_session = 1
    _collect_uncensored = collect_uncensored

    # Parse time_stop from strategy exit for session-end exit
    time_stop_str = strategy.exit.get("time_stop", "09:15")
    _session_end_time = _parse_time(time_stop_str)

    for session_start, session_end in sessions:
        # Get session candles (15m bars within this overnight window)
        session_candles = [
            c for c in candles_15m
            if session_start <= c.timestamp < session_end
        ]
        if not session_candles:
            continue

        # Reference levels: most recent completed RTH session's high/low
        # Search backwards from session_start to find the last RTH session
        # (handles weekends/holidays where same-day RTH doesn't exist)
        rth_candles = None
        for days_back in range(0, 5):  # look back up to 5 calendar days
            check_date = session_start.normalize() - pd.Timedelta(days=days_back)
            rth_start = check_date.replace(hour=9, minute=30)
            rth_end = check_date.replace(hour=16, minute=0)
            candidate_rth = [
                c for c in candles_15m
                if rth_start <= c.timestamp < rth_end
            ]
            if candidate_rth:
                rth_candles = candidate_rth
                break

        if rth_candles:
            on_high = max(c.high for c in rth_candles)   # daytime high = overnight's reference high
            on_low = min(c.low for c in rth_candles)     # daytime low = overnight's reference low
            prev_rth_close = rth_candles[-1].close
        else:
            on_high = None
            on_low = None
            prev_rth_close = None  # no stale data — skip this session

        on_range = (on_high - on_low) if (on_high is not None and on_low is not None) else 0

        # Skip sessions with no RTH reference data (no levels to trade against)
        if on_high is None:
            continue

        # Check filters using the "daytime range" as ON range
        trade_date = session_start.normalize()
        if not check_filters(strategy.filters, on_high, on_low, trade_date):
            continue

        session_open = session_candles[0].open
        trades_this_session = 0
        in_trade = False

        for bar_idx, candle in enumerate(session_candles):
            if trades_this_session >= max_trades_per_session:
                break
            if in_trade:
                continue

            # Gaps
            gaps = []
            if precomputed:
                gaps = mb.find_gaps_at_time_fast(
                    precomputed[0], precomputed[1], candle.timestamp,
                    lookback_days=7,
                )

            # Lookback — session-scoped (only bars from THIS overnight session)
            recent = session_candles[:bar_idx + 1]

            # Build context — no GEX/daily for overnight
            ctx = CandleContext(
                on_high=on_high,
                on_low=on_low,
                on_range=on_range,
                day_open=session_open,
                prev_close=prev_rth_close,
                current_15m=candle,
                bar_index=bar_idx,
                recent_15m=recent,
                gaps=gaps,
                candles_1m=candles_1m,
                ts_index_1m=ts_index_1m,
                time_et=candle.timestamp.time(),
                gex=None,
                daily=None,
                session_start_ts=session_start,
            )

            trade = compute_entry(strategy, ctx, asset_config)
            if trade is None:
                continue

            # Simulate exit with session-end time instead of RTH EOD
            trade = simulate_trade_simple(
                trade, candles_1m, ts_index_1m, asset_config,
                exit_params=strategy.exit,
                session_end_time=_session_end_time,
            )

            if trade.result != "no_fill":
                all_trades.append(trade)
                trades_this_session += 1
                in_trade = True

                if _collect_uncensored:
                    from overfit_search import simulate_trade_uncensored, log_uncensored_candidate
                    unc = simulate_trade_uncensored(
                        candles_1m, ts_index_1m,
                        trade.entry_time, trade.entry_price, trade.direction,
                    )
                    log_uncensored_candidate(
                        candidate_id=strategy.name,
                        archetype=strategy.archetype,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        uncensored=unc,
                    )

    return all_trades
