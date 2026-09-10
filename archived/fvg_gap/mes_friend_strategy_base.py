"""
MES Friend Strategy (Base) - Backtest Prototype
================================================
New file that leaves existing mes_backtest.py unchanged.

Implements requested base behavior:
  - Model hierarchy is whole-day: M1 -> M2 -> M3.
  - Single position only.
  - No lunch skip.
  - M1 gaps must be within 20 pts of overnight H/L.
  - M1 cutoff comparison: gap created after 05:00 PT vs 06:00 PT.
  - Oversized gaps (>=20 pts): require 15m/1h gap alignment.
  - Gap age invalid at >=7 days.
  - Significance marker is 3 pts (penetration + breakout significance).
  - M2 breakout min size 14 pts (open→breakout wick + quarter counter-wick).
  - M3 enabled only after 07:00 PT, uses overnight midpoint and wick entry.
  - Large stop (>=17 pts): take 50% at +14 pts, move runner stop to breakeven,
    runner target at 1.5R.
  - Tracks win rate / EV by 30-minute bucket of gap creation time in PT.

Realism defaults (configurable constants below):
  - Fees and slippage included.
  - Contract cap included.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Lazy import matplotlib — only needed for charting, not backtesting
plt = None

def _get_plt():
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt

import mes_backtest as mb


ET = ZoneInfo("US/Eastern")
PT = ZoneInfo("America/Los_Angeles")

# ---- Run scope ----
START_DATE = "2022-02-14"
END_DATE = "2026-02-15"
# 1yr only: START_DATE = "2025-02-15"

# ---- Strategy constants ----
PENETRATION = 3.0
MIN_GAP_SIZE = 4.0
M1_EXTREMA_DISTANCE = 15.0  # exclusive: dist < 15 (max 14pts)
BTB_EXTREMA_DISTANCE = 26.0  # wider threshold for back-to-back gapper pairs
M1_POTENTIAL_DISTANCE = 15.0
M2_BREAKOUT_MIN = 13.0      # min breakout candle size (body + quarter far wick)
M2_BREAKOUT_SIG = 3.0       # close must be 3+ pts past ON H/L
M3_START_PT = time(7, 0)
GAP_MAX_AGE_DAYS = 7.0      # true gap > 7 days is no longer valid
OVERSIZED_GAP = 20.0
LARGE_STOP = 18.0            # partials at 18+ pt stops
PARTIAL_PROFIT = 14.0
RUNNER_R = 1.5
SL_BUFFER = 3.0              # Place SL 3 pts past the wick
MIN_SL_PTS = 8.0             # Minimum SL distance; snap SL to 8pts if raw risk < 8

# Model-specific SL caps
M1_MAX_SL_PTS = 18.0         # do NOT take 18+ pt stops for M1
M2_MAX_SL_BASE = 20.0        # M2 default max SL
M2_MAX_SL_EXTENDED = 28.0    # M2 max SL for breakout candle > 30pts
ABSOLUTE_MAX_SL = 32.0       # hard cap for any trade

# M2 breakout retracement constants
M2_RETRACEMENT_WINDOW = 2    # max 15m candles after breakout to retrace
M2_BREAKOUT_TIMES_ET = [time(9, 30), time(9, 45)]  # valid breakout times

# M4 "Fade ON H/L" constants (kept for reference, not used in main cascade)
M4_SL_BUFFER = 3.0
M4_WINDOW_START_ET = time(9, 30)
M4_WINDOW_END_ET = time(10, 30)
M4_MIN_ON_RANGE = 20.0
M4_MAX_SL_PTS = 15.0

# Whole-day scan starts at 05:00 PT (08:00 ET) and runs to close.
TRADE_WINDOW_START = time(8, 0)
TRADE_WINDOW_END = time(16, 0)

# ---- Realism knobs ----
MAX_CONTRACTS = 100
SLIPPAGE_PER_SIDE_PTS = 0.125
ROUND_TRIP_FEE_PER_CONTRACT = 1.24


@dataclass
class VariantConfig:
    name: str
    m1_gap_cutoff_pt: time
    entry_style: str = "wick"   # "wick" or "body"
    body_fill_first: bool = False
    immediate_entry: bool = False
    m2_sl_mode: str = "candle_wick"      # "candle_wick" or "on_level"
    priority_mode: str = "first_fill"     # "first_fill" or "m1_pre_rth"
    enable_m2_rewrite: bool = True        # use new M2 breakout retracement


@dataclass
class VariantResult:
    config: VariantConfig
    trades: list[mb.Trade]
    equity: pd.DataFrame
    bucket_stats: pd.DataFrame
    summary: dict


def build_candles(candles: list[mb.Candle], bucket_minutes: int) -> list[mb.Candle]:
    """Aggregate 1m candles to arbitrary minute bucket."""
    buckets: dict[pd.Timestamp, list[mb.Candle]] = {}
    for c in candles:
        minute = (c.timestamp.minute // bucket_minutes) * bucket_minutes
        ts = c.timestamp.replace(minute=minute, second=0, microsecond=0)
        buckets.setdefault(ts, []).append(c)
    out: list[mb.Candle] = []
    for ts in sorted(buckets.keys()):
        bars = buckets[ts]
        out.append(
            mb.Candle(
                timestamp=ts,
                open=bars[0].open,
                high=max(b.high for b in bars),
                low=min(b.low for b in bars),
                close=bars[-1].close,
                volume=sum(b.volume for b in bars),
            )
        )
    return out


def run_backtest_silent(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink):
        return fn(*args, **kwargs)


def overnight_window_for_day(trade_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Overnight range for MES: previous session 18:00 ET to current day 09:30 ET.
    (15:00 PT to 06:30 PT overnight session.)"""
    prev_day = trade_date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    on_start = prev_day.replace(hour=18, minute=0, second=0, microsecond=0)
    on_end = trade_date.replace(hour=9, minute=30, second=0, microsecond=0)
    return on_start, on_end


def overnight_midpoint_prev_close_to_5am_pt(candles_1m: list[mb.Candle],
                                            ts_index: list[pd.Timestamp],
                                            trade_date: pd.Timestamp) -> Optional[float]:
    """Midpoint of overnight move from previous close to 05:00 PT."""
    on_start, _ = overnight_window_for_day(trade_date)
    end_pt = trade_date.tz_convert(PT).replace(hour=5, minute=0, second=0, microsecond=0)
    on_end = end_pt.tz_convert(ET)

    lo = mb.bisect_left(ts_index, on_start)
    hi = mb.bisect_left(ts_index, on_end)
    segment = candles_1m[lo:hi]
    if not segment:
        return None
    high = max(c.high for c in segment)
    low = min(c.low for c in segment)
    return (high + low) / 2.0


def gap_age_days(gap: mb.Gap, at_time: pd.Timestamp) -> float:
    return (at_time - gap.created_at).total_seconds() / 86400.0


def gap_created_after_cutoff_pt(gap: mb.Gap, cutoff_pt: time) -> bool:
    return gap.created_at.tz_convert(PT).time() >= cutoff_pt


def gap_is_from_current_overnight(gap: mb.Gap, trade_date: pd.Timestamp) -> bool:
    """True if gap was created during the overnight session leading into trade_date.
    Overnight = previous weekday 18:00 ET to before 09:30 ET (exclusive)."""
    on_start, on_end = overnight_window_for_day(trade_date)
    return on_start <= gap.created_at < on_end


def gap_is_same_trading_week(gap: mb.Gap, trade_time: pd.Timestamp) -> bool:
    """Check if gap was created in the same trading week as trade_time.
    Trading week = Monday through Friday.
    Weekend gaps (Sat/Sun) belong to the NEXT Monday-Friday week.
    Gaps from prior week expire when new week starts (Monday)."""
    gap_dt = gap.created_at
    trade_monday = trade_time.normalize() - timedelta(days=trade_time.weekday())
    gap_monday = gap_dt.normalize() - timedelta(days=gap_dt.weekday())
    if gap_dt.weekday() >= 5:
        gap_monday = gap_dt.normalize() + timedelta(days=(7 - gap_dt.weekday()))
    return gap_monday == trade_monday


def distance_to_overnight_extrema(gap: mb.Gap, overnight_high: float, overnight_low: float) -> float:
    return min(
        abs(gap.gap_top - overnight_high),
        abs(gap.gap_bottom - overnight_high),
        abs(gap.gap_top - overnight_low),
        abs(gap.gap_bottom - overnight_low),
    )


def determine_direction(gap: mb.Gap, candle_15m: mb.Candle) -> str:
    """Direction based on current 15m open relative to gap range."""
    if candle_15m.open >= gap.gap_top:
        return "long"
    if candle_15m.open <= gap.gap_bottom:
        return "short"
    mid = (gap.gap_top + gap.gap_bottom) / 2.0
    return "long" if candle_15m.open >= mid else "short"


def ranges_overlap(a_bot: float, a_top: float, b_bot: float, b_top: float) -> bool:
    return min(a_top, b_top) > max(a_bot, b_bot)


def has_1h_alignment(gap_15m: mb.Gap,
                     candle_15m: mb.Candle,
                     gaps_1h: list[mb.Gap],
                     intended_direction: str) -> bool:
    """Require a same-direction overlapping unfilled 1h gap."""
    for g1h in gaps_1h:
        if not ranges_overlap(gap_15m.gap_bottom, gap_15m.gap_top, g1h.gap_bottom, g1h.gap_top):
            continue
        d1h = determine_direction(g1h, candle_15m)
        if d1h == intended_direction:
            return True
    return False


def try_m4_fade_on_hl(
    candles_1m: list[mb.Candle],
    ts_1m: list,
    candle_15m: mb.Candle,
    overnight_high: float,
    overnight_low: float,
    trade_date,
) -> Optional[mb.Trade]:
    """M4: Fade ON H/L touch during first 60min of RTH.

    Scans 1m bars within this 15m candle for price actually reaching ON H or
    ON L.  Entry is a limit order AT the ON level — the 1m bar must trade
    through the level (high >= ON_HIGH for short, low <= ON_LOW for long).
    No tolerance: price must actually touch.

    SL is placed behind the ON extreme + buffer.  NO lookahead — we only use
    data available at entry time (the overnight range).
    """
    ct = candle_15m.timestamp
    if not hasattr(ct, 'hour'):
        return None
    et_time = ct.time() if ct.tzinfo is None else ct.astimezone(ZoneInfo("US/Eastern")).time()
    if et_time < M4_WINDOW_START_ET or et_time >= M4_WINDOW_END_ET:
        return None

    on_range = overnight_high - overnight_low
    if on_range < M4_MIN_ON_RANGE:
        return None

    # Scan 1m bars within this 15m bucket
    start = candle_15m.timestamp
    end = start + timedelta(minutes=15)
    lo = mb.bisect_left(ts_1m, start)
    hi = mb.bisect_left(ts_1m, end)
    bars = candles_1m[lo:hi]

    for bar in bars:
        # Check touch of ON HIGH → short fade
        # Bar must actually trade AT or ABOVE overnight high (limit fill)
        if bar.high >= overnight_high:
            entry = round(overnight_high * 4) / 4  # limit at ON H
            # SL behind ON high + buffer — no lookahead, only ON range
            sl = overnight_high + M4_SL_BUFFER
            risk = abs(entry - sl)
            if risk < MIN_SL_PTS:
                sl = entry + MIN_SL_PTS
                risk = MIN_SL_PTS
            if risk > M4_MAX_SL_PTS:
                continue
            tp = entry - risk  # 1RR
            return mb.Trade(
                model=4, direction="short",
                entry_price=entry, stop_loss=sl, take_profit=tp,
                entry_time=bar.timestamp,
                gap=None,
                notes=f"M4 fade ON_HIGH touch ({risk:.1f}r)",
            )

        # Check touch of ON LOW → long fade
        # Bar must actually trade AT or BELOW overnight low (limit fill)
        if bar.low <= overnight_low:
            entry = round(overnight_low * 4) / 4  # limit at ON L
            # SL behind ON low - buffer — no lookahead
            sl = overnight_low - M4_SL_BUFFER
            risk = abs(entry - sl)
            if risk < MIN_SL_PTS:
                sl = entry - MIN_SL_PTS
                risk = MIN_SL_PTS
            if risk > M4_MAX_SL_PTS:
                continue
            tp = entry + risk  # 1RR
            return mb.Trade(
                model=4, direction="long",
                entry_price=entry, stop_loss=sl, take_profit=tp,
                entry_time=bar.timestamp,
                gap=None,
                notes=f"M4 fade ON_LOW touch ({risk:.1f}r)",
            )

    return None


def m2_breakout_qualifies_candle(
    candle: mb.Candle,
    overnight_high: float,
    overnight_low: float,
) -> Optional[str]:
    """Check if a 15m candle is an M2 breakout. Returns direction or None.

    Breakout requires:
    - Candle wick breaks ON H/L (high > ON_HIGH or low < ON_LOW)
    - Close 3+ pts past ON H/L (significance)
    - Directional move + quarter far wick >= 13pts (size)
    - Candle time must be 09:30 or 09:45 ET

    Returns "long" for bull breakout (buy momentum), "short" for bear breakout.
    """
    ct = candle.timestamp
    et_time = ct.time() if ct.tzinfo is None else ct.astimezone(ZoneInfo("US/Eastern")).time()
    if et_time not in (time(9, 30), time(9, 45)):
        return None

    c = candle
    # Bull breakout: wick breaks above ON HIGH, close confirms
    bull_size = (c.high - c.open) + (c.open - c.low) / 4.0
    bull_break = (
        c.high > overnight_high
        and c.close > overnight_high + M2_BREAKOUT_SIG
        and bull_size >= M2_BREAKOUT_MIN
    )
    # Bear breakout: wick breaks below ON LOW, close confirms
    bear_size = (c.open - c.low) + (c.high - c.open) / 4.0
    bear_break = (
        c.low < overnight_low
        and c.close < overnight_low - M2_BREAKOUT_SIG
        and bear_size >= M2_BREAKOUT_MIN
    )

    if bull_break:
        return "long"
    if bear_break:
        return "short"
    return None


def m2_combine_candles(c1: mb.Candle, c2: mb.Candle) -> mb.Candle:
    """Combine two consecutive 15m candles into a synthetic candle for M2."""
    return mb.Candle(
        timestamp=c1.timestamp,
        open=c1.open,
        high=max(c1.high, c2.high),
        low=min(c1.low, c2.low),
        close=c2.close,
        volume=c1.volume + c2.volume,
    )


def try_m2_breakout_entry(
    breakout_candle: mb.Candle,
    breakout_dir: str,
    retracement_candle: mb.Candle,
    candles_1m: list[mb.Candle],
    ts_1m: list[pd.Timestamp],
    overnight_high: float,
    overnight_low: float,
    sl_mode: str = "candle_wick",
) -> Optional[mb.Trade]:
    """M2 breakout retracement entry — no gap required.

    After a breakout candle breaks ON H/L, enter on retracement back into the
    breakout candle body. Entry at body edge + PENETRATION into the body.
    SL behind candle wick or ON H/L depending on sl_mode.
    """
    ct = retracement_candle.timestamp.time()
    if ct < TRADE_WINDOW_START or ct >= TRADE_WINDOW_END:
        return None

    # Entry: retracement back INTO the breakout candle body
    # Bear breakout (short momentum) → we go SHORT when price retraces UP
    #   Entry = body_top - PENETRATION (3pts into body from top, price approaches from below)
    # Bull breakout (long momentum) → we go LONG when price retraces DOWN
    #   Entry = body_bottom + PENETRATION (3pts into body from bottom, price approaches from above)
    if breakout_dir == "short":
        # Bear breakout: short on retracement up into candle body
        direction = "short"
        entry_price = breakout_candle.body_top - PENETRATION
        if sl_mode == "on_level":
            stop_loss = overnight_low - SL_BUFFER  # behind ON LOW (the level that was broken)
        else:
            stop_loss = breakout_candle.high + SL_BUFFER  # behind candle wick
    else:
        # Bull breakout: long on retracement down into candle body
        direction = "long"
        entry_price = breakout_candle.body_bottom + PENETRATION
        if sl_mode == "on_level":
            stop_loss = overnight_high + SL_BUFFER  # behind ON HIGH
        else:
            stop_loss = breakout_candle.low - SL_BUFFER  # behind candle wick

    # Round to MES tick
    entry_price = round(entry_price * 4) / 4

    # Min SL snap
    risk = abs(entry_price - stop_loss)
    if risk < MIN_SL_PTS:
        stop_loss = entry_price - MIN_SL_PTS if direction == "long" else entry_price + MIN_SL_PTS
        risk = MIN_SL_PTS

    # SL validation
    if direction == "long" and stop_loss >= entry_price:
        return None
    if direction == "short" and stop_loss <= entry_price:
        return None

    risk = abs(entry_price - stop_loss)

    # M2 SL cap: 20pts base, 28pts for large breakout candles (> 30pt body)
    max_sl = M2_MAX_SL_BASE
    if breakout_candle.body_size > 30.0:
        max_sl = M2_MAX_SL_EXTENDED
    if risk > max_sl:
        return None
    if risk > ABSOLUTE_MAX_SL:
        return None

    take_profit = entry_price + risk if direction == "long" else entry_price - risk

    # Scan 1m bars for entry fill
    start = retracement_candle.timestamp
    end = start + timedelta(minutes=15)
    lo = mb.bisect_left(ts_1m, start)
    hi = mb.bisect_left(ts_1m, end)
    bars = candles_1m[lo:hi]

    entry_time = None
    for b in bars:
        if direction == "long" and b.low <= entry_price:
            entry_time = b.timestamp
            break
        if direction == "short" and b.high >= entry_price:
            entry_time = b.timestamp
            break
    if entry_time is None:
        return None

    return mb.Trade(
        model=2,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_time=entry_time,
        gap=None,
        notes=f"M2 breakout retracement [{sl_mode}]",
    )


# Legacy wrapper for old code paths that pass a Gap
def m2_breakout_qualifies(gap: mb.Gap, overnight_high: float, overnight_low: float) -> bool:
    return m2_breakout_qualifies_candle(gap.candle, overnight_high, overnight_low) is not None



def are_adjacent_candles(c1: mb.Candle, c2: mb.Candle,
                         ts_15m: list[pd.Timestamp]) -> bool:
    """True if c1 and c2 are consecutive 15m candles."""
    idx1 = mb.bisect_left(ts_15m, c1.timestamp)
    idx2 = mb.bisect_left(ts_15m, c2.timestamp)
    return abs(idx1 - idx2) == 1


def find_backtoback_pairs(gaps: list[mb.Gap],
                          ts_15m: list[pd.Timestamp]) -> list[tuple[mb.Gap, mb.Gap]]:
    """Find pairs of gaps whose candles are consecutive 15m bars."""
    pairs: list[tuple[mb.Gap, mb.Gap]] = []
    for i, g1 in enumerate(gaps):
        for g2 in gaps[i + 1:]:
            if are_adjacent_candles(g1.candle, g2.candle, ts_15m):
                pairs.append((g1, g2))
    return pairs


def compute_backtoback_entry(
    gap_further: mb.Gap,
    gap_closer: mb.Gap,
    candle_15m: mb.Candle,
) -> tuple[float, str]:
    """Compute 51% penetration entry for back-to-back gappers.

    Returns (entry_price, direction).
    The further gap's body is trimmed by the closer gap's body overlap.
    Entry = 51% into that effective body from the side price approaches.
    """
    direction = determine_direction(gap_further, candle_15m)

    fb_bot = gap_further.candle.body_bottom
    fb_top = gap_further.candle.body_top
    cb_bot = gap_closer.candle.body_bottom
    cb_top = gap_closer.candle.body_top

    # Trim further body by closer body overlap.
    # The closer candle sits between price and the further candle.
    # Remove the portion of the further body that the closer body covers.
    if direction == "long":
        # Price falling from above into gap → closer is above further
        # Trim the top of the further body where closer body overlaps
        eff_bot = fb_bot
        eff_top = min(fb_top, cb_bot)  # cut where closer body starts from above
    else:
        # Price rising from below into gap → closer is below further
        # Trim the bottom of the further body where closer body overlaps
        eff_bot = max(fb_bot, cb_top)  # cut where closer body ends from below
        eff_top = fb_top

    body_size = max(eff_top - eff_bot, 0.0)
    if body_size < 0.5:
        # Degenerate case: no effective body after trim
        if direction == "long":
            entry = gap_further.gap_top - PENETRATION
        else:
            entry = gap_further.gap_bottom + PENETRATION
    elif direction == "long":
        entry = eff_top - 0.51 * body_size
    else:
        entry = eff_bot + 0.51 * body_size

    # Round to MES tick (0.25)
    entry = round(entry * 4) / 4
    return entry, direction


def try_gap_entry(
    gap: mb.Gap,
    candle_15m: mb.Candle,
    candles_1m: list[mb.Candle],
    ts_index_1m: list[pd.Timestamp],
    model: int,
    notes: str,
    entry_style: str = "body",  # kept for API compat, ignored
    candles_15m: Optional[list[mb.Candle]] = None,
    ts_15m: Optional[list[pd.Timestamp]] = None,
) -> Optional[mb.Trade]:
    """Entry at edge of remaining unfilled gap minus penetration.
    SL is placed SL_BUFFER pts past the gap candle wick.
    If risk < MIN_SL_PTS, snap SL to exactly MIN_SL_PTS from entry."""
    ct = candle_15m.timestamp.time()
    if ct < TRADE_WINDOW_START or ct >= TRADE_WINDOW_END:
        return None

    direction = determine_direction(gap, candle_15m)

    # Penetration scaling: large gaps need deeper penetration
    pen = PENETRATION
    if gap.size >= 30.0:
        pen = max(PENETRATION, min(7.0, gap.size * 0.20))
    elif gap.size >= 20.0:
        pen = max(PENETRATION, min(5.0, gap.size * 0.15))

    # Entry at the edge of the remaining unfilled gap minus penetration.
    # SL at the gap candle's wick extreme.
    if direction == "long":
        entry_price = gap.gap_top - pen
        raw_sl_base = gap.candle.low
    else:
        entry_price = gap.gap_bottom + pen
        raw_sl_base = gap.candle.high

    # Apply SL buffer (3 pts past wick)
    if direction == "long":
        stop_loss = raw_sl_base - SL_BUFFER
    else:
        stop_loss = raw_sl_base + SL_BUFFER

    # Min SL snap: if risk < 8pts, snap SL to exactly 8pts from entry
    if abs(entry_price - stop_loss) < MIN_SL_PTS:
        stop_loss = entry_price - MIN_SL_PTS if direction == "long" else entry_price + MIN_SL_PTS

    if direction == "long" and stop_loss >= entry_price:
        return None
    if direction == "short" and stop_loss <= entry_price:
        return None

    risk = abs(entry_price - stop_loss)
    if risk < mb.MIN_RISK:
        return None

    # Model-specific SL caps
    if model == 1 and risk > M1_MAX_SL_PTS:
        return None
    if risk > ABSOLUTE_MAX_SL:
        return None

    take_profit = entry_price + risk if direction == "long" else entry_price - risk

    start = candle_15m.timestamp
    end = start + timedelta(minutes=15)
    lo = mb.bisect_left(ts_index_1m, start)
    hi = mb.bisect_left(ts_index_1m, end)
    bars = candles_1m[lo:hi]

    entry_time = None
    for b in bars:
        if direction == "long" and b.low <= entry_price:
            entry_time = b.timestamp
            break
        if direction == "short" and b.high >= entry_price:
            entry_time = b.timestamp
            break
    if entry_time is None:
        return None

    return mb.Trade(
        model=model,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_time=entry_time,
        gap=gap,
        notes=f"{notes} [{entry_style} entry]",
    )


def simulate_trade_with_partials_and_costs(
    trade: mb.Trade,
    candles_1m: list[mb.Candle],
    ts_index_1m: list[pd.Timestamp],
) -> mb.Trade:
    """Simulate exits with large-stop partials + runner + execution costs."""
    risk = abs(trade.entry_price - trade.stop_loss)
    contracts = min(MAX_CONTRACTS, mb.compute_contracts(risk))
    contracts = max(1, contracts)
    trade.num_contracts = contracts

    # Validate entry fill: the entry bar must have traded at the entry price.
    entry_idx = mb.bisect_left(ts_index_1m, trade.entry_time)
    if entry_idx < len(candles_1m):
        entry_bar = candles_1m[entry_idx]
        if trade.direction == "long" and entry_bar.low > trade.entry_price:
            trade.pnl_dollars = 0.0
            trade.pnl = 0.0
            trade.result = "no_fill"
            trade.exit_time = trade.entry_time
            trade.exit_price = trade.entry_price
            return trade
        if trade.direction == "short" and entry_bar.high < trade.entry_price:
            trade.pnl_dollars = 0.0
            trade.pnl = 0.0
            trade.result = "no_fill"
            trade.exit_time = trade.entry_time
            trade.exit_price = trade.entry_price
            return trade

    lo = mb.bisect_right(ts_index_1m, trade.entry_time)
    eod = trade.entry_time.normalize() + timedelta(days=1)
    hi = mb.bisect_left(ts_index_1m, eod)
    future = candles_1m[lo:hi]

    large_stop_mode = risk >= LARGE_STOP and contracts >= 2
    partial_lot = contracts // 2 if large_stop_mode else 0
    runner_lot = contracts - partial_lot if large_stop_mode else 0
    runner_tp = (trade.entry_price + risk * RUNNER_R
                 if trade.direction == "long"
                 else trade.entry_price - risk * RUNNER_R)
    partial_px = (trade.entry_price + PARTIAL_PROFIT
                  if trade.direction == "long"
                  else trade.entry_price - PARTIAL_PROFIT)

    points_x_contracts = 0.0
    open_lot = contracts
    partial_done = False
    runner_stop = trade.stop_loss

    mae, mfe = 0.0, 0.0

    def add_exit(exit_price: float, lot: int):
        nonlocal points_x_contracts, open_lot
        if lot <= 0:
            return
        if trade.direction == "long":
            pts = exit_price - trade.entry_price
        else:
            pts = trade.entry_price - exit_price
        points_x_contracts += pts * lot
        open_lot -= lot

    for c in future:
        if trade.direction == "long":
            adverse = trade.entry_price - c.low
            favorable = c.high - trade.entry_price
        else:
            adverse = c.high - trade.entry_price
            favorable = trade.entry_price - c.low
        mae = max(mae, adverse)
        mfe = max(mfe, favorable)

        if not large_stop_mode:
            if trade.direction == "long":
                if c.low <= trade.stop_loss:
                    add_exit(trade.stop_loss, open_lot)
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = c.timestamp
                    break
                if c.high >= trade.take_profit:
                    add_exit(trade.take_profit, open_lot)
                    trade.exit_price = trade.take_profit
                    trade.exit_time = c.timestamp
                    break
            else:
                if c.high >= trade.stop_loss:
                    add_exit(trade.stop_loss, open_lot)
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = c.timestamp
                    break
                if c.low <= trade.take_profit:
                    add_exit(trade.take_profit, open_lot)
                    trade.exit_price = trade.take_profit
                    trade.exit_time = c.timestamp
                    break
        else:
            if not partial_done:
                if trade.direction == "long":
                    if c.low <= trade.stop_loss:
                        add_exit(trade.stop_loss, open_lot)
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = c.timestamp
                        break
                    if c.high >= partial_px:
                        add_exit(partial_px, partial_lot)
                        partial_done = True
                        runner_stop = trade.entry_price
                        trade.notes += " [Partial +14, runner BE]"
                        if runner_lot > 0 and c.high >= runner_tp:
                            add_exit(runner_tp, runner_lot)
                            trade.exit_price = runner_tp
                            trade.exit_time = c.timestamp
                            break
                else:
                    if c.high >= trade.stop_loss:
                        add_exit(trade.stop_loss, open_lot)
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = c.timestamp
                        break
                    if c.low <= partial_px:
                        add_exit(partial_px, partial_lot)
                        partial_done = True
                        runner_stop = trade.entry_price
                        trade.notes += " [Partial +14, runner BE]"
                        if runner_lot > 0 and c.low <= runner_tp:
                            add_exit(runner_tp, runner_lot)
                            trade.exit_price = runner_tp
                            trade.exit_time = c.timestamp
                            break
            else:
                if trade.direction == "long":
                    if c.low <= runner_stop:
                        add_exit(runner_stop, runner_lot)
                        trade.exit_price = runner_stop
                        trade.exit_time = c.timestamp
                        break
                    if c.high >= runner_tp:
                        add_exit(runner_tp, runner_lot)
                        trade.exit_price = runner_tp
                        trade.exit_time = c.timestamp
                        break
                else:
                    if c.high >= runner_stop:
                        add_exit(runner_stop, runner_lot)
                        trade.exit_price = runner_stop
                        trade.exit_time = c.timestamp
                        break
                    if c.low <= runner_tp:
                        add_exit(runner_tp, runner_lot)
                        trade.exit_price = runner_tp
                        trade.exit_time = c.timestamp
                        break

        if c.timestamp.time() >= time(16, 0) and open_lot > 0:
            add_exit(c.close, open_lot)
            trade.exit_price = c.close
            trade.exit_time = c.timestamp
            trade.notes += " [EOD exit]"
            break

    if open_lot > 0 and future:
        last = future[-1]
        add_exit(last.close, open_lot)
        trade.exit_price = last.close
        trade.exit_time = last.timestamp
        trade.notes += " [data end exit]"

    gross_dollars = points_x_contracts * mb.MES_MULTIPLIER
    slippage_dollars = contracts * (2.0 * SLIPPAGE_PER_SIDE_PTS * mb.MES_MULTIPLIER)
    fee_dollars = contracts * ROUND_TRIP_FEE_PER_CONTRACT
    net_dollars = gross_dollars - slippage_dollars - fee_dollars

    trade.pnl_dollars = round(net_dollars, 2)
    trade.pnl = (net_dollars / (mb.MES_MULTIPLIER * contracts)) if contracts > 0 else 0.0
    trade.result = "win" if trade.pnl_dollars > 0 else "loss"
    trade.max_adverse_excursion = mae
    trade.max_favorable_excursion = mfe
    trade.max_favorable_excursion_r = mfe / risk if risk > 0 else 0.0
    return trade


def bucket_label_30m_pt(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(PT)
    bucket_min = (t.minute // 30) * 30
    return f"{t.hour:02d}:{bucket_min:02d}"


def bucket_stats_by_gap_creation_pt(trades: list[mb.Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        if not t.gap:
            continue
        ts = t.gap.created_at
        pt_time = ts.tz_convert(PT).time()
        if pt_time < time(5, 0) or pt_time >= time(13, 0):
            continue
        rows.append({
            "bucket_pt": bucket_label_30m_pt(ts),
            "pnl_dollars": float(t.pnl_dollars or 0.0),
            "is_win": 1 if t.result == "win" else 0,
        })
    if not rows:
        return pd.DataFrame(columns=["bucket_pt", "trades", "wins", "win_rate", "ev_dollars", "total_pnl"])
    df = pd.DataFrame(rows)
    out = (
        df.groupby("bucket_pt")
        .agg(
            trades=("pnl_dollars", "count"),
            wins=("is_win", "sum"),
            win_rate=("is_win", "mean"),
            ev_dollars=("pnl_dollars", "mean"),
            total_pnl=("pnl_dollars", "sum"),
        )
        .reset_index()
    )
    out["win_rate"] = out["win_rate"] * 100.0
    out = out.sort_values("bucket_pt").reset_index(drop=True)
    return out


def equity_df(trades: list[mb.Trade]) -> pd.DataFrame:
    running = 0.0
    peak = 0.0
    rows = []
    for i, t in enumerate(trades, start=1):
        running += float(t.pnl_dollars or 0.0)
        peak = max(peak, running)
        dd = peak - running
        rows.append({
            "trade_id": i,
            "timestamp": t.exit_time or t.entry_time,
            "pnl_dollars": float(t.pnl_dollars or 0.0),
            "equity_dollars": running,
            "drawdown_dollars": dd,
            "model": t.model,
            "direction": t.direction,
        })
    return pd.DataFrame(rows)


def summarize(trades: list[mb.Trade], eq: pd.DataFrame) -> dict:
    total = len(trades)
    wins = sum(1 for t in trades if t.result == "win")
    wr = (100.0 * wins / total) if total else 0.0
    pnl = float(eq["equity_dollars"].iloc[-1]) if not eq.empty else 0.0
    dd = float(eq["drawdown_dollars"].max()) if not eq.empty else 0.0
    by_model = {}
    for m in (1, 2, 3):
        mt = [t for t in trades if t.model == m]
        if not mt:
            continue
        mw = sum(1 for t in mt if t.result == "win")
        by_model[m] = {
            "trades": len(mt),
            "win_rate": 100.0 * mw / len(mt),
            "pnl_dollars": sum(float(t.pnl_dollars or 0.0) for t in mt),
        }
    return {
        "trades": total,
        "wins": wins,
        "win_rate": wr,
        "total_pnl_dollars": pnl,
        "max_drawdown_dollars": dd,
        "by_model": by_model,
    }


def trade_rows(trades: list[mb.Trade]) -> list[dict]:
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "model": t.model,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "exit_price": t.exit_price,
            "result": t.result,
            "pnl_pts": t.pnl,
            "pnl_dollars": t.pnl_dollars,
            "num_contracts": t.num_contracts,
            "gap_created_at": t.gap.created_at if t.gap else None,
            "gap_size": t.gap.size if t.gap else None,
            "notes": t.notes,
        })
    return rows


def run_variant(
    config: VariantConfig,
    candles_1m: list[mb.Candle],
    candles_15m_all: list[mb.Candle],
    candles_1h_all: list[mb.Candle],
    **kwargs,
) -> VariantResult:
    # Precomputed gap snapshots (optional — falls back to find_unfilled_gaps_at_time)
    _gap_snaps_15m = kwargs.get("gap_snapshots_15m")
    _gap_snaps_15m_bff = kwargs.get("gap_snapshots_15m_bff")
    _gap_snaps_1h = kwargs.get("gap_snapshots_1h")
    _gap_ts_15m = kwargs.get("gap_ts_15m")
    _gap_ts_1h = kwargs.get("gap_ts_1h")

    ts_1m = [c.timestamp for c in candles_1m]
    ts_15m = [c.timestamp for c in candles_15m_all]
    ts_1h = [c.timestamp for c in candles_1h_all]

    trading_days = sorted({
        c.timestamp.normalize() for c in candles_15m_all if c.timestamp.weekday() < 5
    })

    traded_candle_keys: set[str] = set()
    all_trades: list[mb.Trade] = []
    # BTB setups persist across days until triggered or the further gap fills.
    # Maps direction -> (entry_price, sl, tp, gap, notes)
    pending_btb: dict[str, tuple[float, float, float, mb.Gap, str]] = {}

    for trade_date in trading_days:
        day_start = trade_date.replace(hour=TRADE_WINDOW_START.hour, minute=TRADE_WINDOW_START.minute)
        day_end = trade_date.replace(hour=TRADE_WINDOW_END.hour, minute=TRADE_WINDOW_END.minute)
        day_candles = [c for c in candles_15m_all if day_start <= c.timestamp < day_end]
        if not day_candles:
            continue

        overnight_high, overnight_low = mb.get_overnight_high_low(candles_1m, trade_date, ts_index=ts_1m)
        if overnight_high is None or overnight_low is None:
            continue
        m3_mid = overnight_midpoint_prev_close_to_5am_pt(candles_1m, ts_1m, trade_date)
        prev_close_et, _ = overnight_window_for_day(trade_date)

        open_trade: Optional[mb.Trade] = None
        daily_entry_keys: set[str] = set()
        # M2 breakout tracking: (breakout_candle, direction, candles_remaining)
        pending_m2: Optional[tuple[mb.Candle, str, int]] = None

        for candle in day_candles:
            if open_trade and open_trade.exit_time and open_trade.exit_time <= candle.timestamp:
                open_trade = None

            if _gap_snaps_15m is not None:
                gaps_15m = mb.find_gaps_at_time_fast(
                    _gap_snaps_15m, _gap_ts_15m, candle.timestamp,
                    lookback_days=25, immediate_entry=config.immediate_entry,
                )
            else:
                gaps_15m = mb.find_unfilled_gaps_at_time(
                    candles_15m_all,
                    candle.timestamp,
                    lookback_days=25,
                    ts_index_15m=ts_15m,
                    immediate_entry=config.immediate_entry,
                    body_fill_first=config.body_fill_first,
                )
            gaps_15m = sorted(gaps_15m, key=lambda g: -g.size)

            if _gap_snaps_1h is not None:
                gaps_1h = mb.find_gaps_at_time_fast(
                    _gap_snaps_1h, _gap_ts_1h, candle.timestamp,
                    lookback_days=25,
                )
            else:
                gaps_1h = mb.find_unfilled_gaps_at_time(
                    candles_1h_all,
                    candle.timestamp,
                    lookback_days=25,
                    ts_index_15m=ts_1h,
                    body_fill_first=config.body_fill_first,
                )

            # Pre-filter + annotate per-gap context once.
            gap_rows: list[dict] = []
            m1_potential_dirs: set[str] = set()
            m1_setup_available = False
            m1_sides: set[str] = set()
            pre_rth_m1_exists = False  # Change 5: block M2/M3 if pre-RTH M1 gap

            for gap in gaps_15m:
                if gap.size <= MIN_GAP_SIZE:
                    continue
                if gap_age_days(gap, candle.timestamp) >= GAP_MAX_AGE_DAYS:
                    continue
                # Change 4: weekly gap expiry
                if not gap_is_same_trading_week(gap, candle.timestamp):
                    continue
                if gap.candle.timestamp.isoformat() in traded_candle_keys:
                    continue

                direction = determine_direction(gap, candle)
                dist_ext = distance_to_overnight_extrema(gap, overnight_high, overnight_low)
                is_overnight = gap_is_from_current_overnight(gap, trade_date)
                cutoff_ok = not is_overnight
                oversize_ok = True
                if gap.size >= OVERSIZED_GAP:
                    oversize_ok = has_1h_alignment(gap, candle, gaps_1h, direction)

                is_m1_setup = bool(dist_ext < M1_EXTREMA_DISTANCE and cutoff_ok and oversize_ok and not is_overnight)
                if is_m1_setup:
                    m1_setup_available = True
                    m1_sides.add(direction)
                    # Change 5: check if gap created during 08:00-09:30 ET (5-6:30 PT)
                    gap_et = gap.created_at.astimezone(ET).time()
                    if time(8, 0) <= gap_et < time(9, 30):
                        pre_rth_m1_exists = True

                if dist_ext <= M1_POTENTIAL_DISTANCE and cutoff_ok:
                    m1_potential_dirs.add(direction)

                gap_rows.append({
                    "gap": gap,
                    "direction": direction,
                    "m1_setup": is_m1_setup,
                    "oversize_ok": oversize_ok,
                })

            # 1h gap fallback M1
            m1_fallback_rows: list[dict] = []
            for gap_1h in gaps_1h:
                if gap_1h.size <= MIN_GAP_SIZE:
                    continue
                if gap_1h.candle.timestamp.isoformat() in traded_candle_keys:
                    continue
                if gap_is_from_current_overnight(gap_1h, trade_date):
                    continue
                direction = determine_direction(gap_1h, candle)
                if direction in m1_sides:
                    continue
                dist_ext = distance_to_overnight_extrema(gap_1h, overnight_high, overnight_low)
                if dist_ext >= M1_EXTREMA_DISTANCE:
                    continue
                m1_fallback_rows.append({
                    "gap": gap_1h,
                    "direction": direction,
                })

            # ── Build M1 candidates (always, regardless of priority mode) ──
            m1_candidates: list[mb.Trade] = []
            m2_candidates: list[mb.Trade] = []
            m3_candidates: list[mb.Trade] = []

            for row in gap_rows:
                gap = row["gap"]
                if not row["m1_setup"]:
                    continue
                t = try_gap_entry(
                    gap, candle, candles_1m, ts_1m, 1,
                    "M1 near overnight extrema",
                    config.entry_style,
                    candles_15m=candles_15m_all,
                    ts_15m=ts_15m,
                )
                if t:
                    m1_candidates.append(t)

            # 1h fallback M1
            for row in m1_fallback_rows:
                gap_1h = row["gap"]
                t = try_gap_entry(
                    gap_1h, candle, candles_1m, ts_1m, 1,
                    "M1 fallback (1h gap)",
                    config.entry_style,
                    candles_15m=candles_15m_all,
                    ts_15m=ts_15m,
                )
                if t:
                    m1_candidates.append(t)

            # Back-to-back gapper rule
            btb_pairs = find_backtoback_pairs(gaps_15m, ts_15m)
            for g1, g2 in btb_pairs:
                d1 = distance_to_overnight_extrema(g1, overnight_high, overnight_low)
                d2 = distance_to_overnight_extrema(g2, overnight_high, overnight_low)
                if min(d1, d2) >= BTB_EXTREMA_DISTANCE:
                    continue
                if d1 > d2:
                    further, closer = g1, g2
                else:
                    further, closer = g2, g1
                if gap_is_from_current_overnight(further, trade_date):
                    continue
                if gap_is_from_current_overnight(closer, trade_date):
                    continue
                if further.candle.timestamp.isoformat() in traded_candle_keys:
                    continue
                btb_entry, btb_dir = compute_backtoback_entry(further, closer, candle)
                if btb_dir in pending_btb:
                    continue
                earlier = further if further.candle.timestamp < closer.candle.timestamp else closer
                earlier_idx = mb.bisect_left(ts_15m, earlier.candle.timestamp)
                sl_candles = [further.candle, closer.candle]
                if earlier_idx > 0:
                    sl_candles.append(candles_15m_all[earlier_idx - 1])
                if btb_dir == "long":
                    btb_sl = min(c.low for c in sl_candles) - SL_BUFFER
                else:
                    btb_sl = max(c.high for c in sl_candles) + SL_BUFFER
                if abs(btb_entry - btb_sl) < MIN_SL_PTS:
                    btb_sl = btb_entry - MIN_SL_PTS if btb_dir == "long" else btb_entry + MIN_SL_PTS
                btb_risk = abs(btb_entry - btb_sl)
                btb_tp = btb_entry + btb_risk if btb_dir == "long" else btb_entry - btb_risk
                pending_btb[btb_dir] = (btb_entry, btb_sl, btb_tp, further, "M1 Back-to-Back 51% penetration")

            # Trigger pending BTB entries
            current_gap_keys = {g.candle.timestamp.isoformat() for g in gaps_15m}
            for btb_dir in list(pending_btb):
                _, _, _, btb_gap, _ = pending_btb[btb_dir]
                if btb_gap.candle.timestamp.isoformat() not in current_gap_keys:
                    del pending_btb[btb_dir]
            if pending_btb:
                btb_sides = set(pending_btb.keys())
                m1_candidates = [t for t in m1_candidates if t.direction not in btb_sides]
            for btb_dir, (btb_entry, btb_sl, btb_tp, btb_gap, btb_notes) in list(pending_btb.items()):
                start = candle.timestamp
                end = start + timedelta(minutes=15)
                lo_1m = mb.bisect_left(ts_1m, start)
                hi_1m = mb.bisect_left(ts_1m, end)
                bars = candles_1m[lo_1m:hi_1m]
                entry_time = None
                for b in bars:
                    if btb_dir == "long" and b.low <= btb_entry:
                        entry_time = b.timestamp
                        break
                    if btb_dir == "short" and b.high >= btb_entry:
                        entry_time = b.timestamp
                        break
                if entry_time is not None:
                    btb_trade = mb.Trade(
                        model=1, direction=btb_dir,
                        entry_price=btb_entry, stop_loss=btb_sl,
                        take_profit=btb_tp, entry_time=entry_time,
                        gap=btb_gap, notes=btb_notes,
                    )
                    m1_candidates.append(btb_trade)
                    m1_setup_available = True
                    del pending_btb[btb_dir]

            # Covered gapper (merged with M1)
            covered_candidates: list[mb.Trade] = []
            if _gap_snaps_15m_bff is not None:
                covered_gaps_15m = mb.find_gaps_at_time_fast(
                    _gap_snaps_15m_bff, _gap_ts_15m, candle.timestamp,
                    lookback_days=25, immediate_entry=config.immediate_entry,
                )
            else:
                covered_gaps_15m = mb.find_unfilled_gaps_at_time(
                    candles_15m_all, candle.timestamp, lookback_days=25,
                    ts_index_15m=ts_15m, immediate_entry=config.immediate_entry,
                    body_fill_first=True,
                )
            normal_candle_keys = {g.candle.timestamp.isoformat() for g in gaps_15m}
            for cov_gap in covered_gaps_15m:
                if cov_gap.candle.timestamp.isoformat() in normal_candle_keys:
                    continue
                if cov_gap.size <= MIN_GAP_SIZE:
                    continue
                if gap_age_days(cov_gap, candle.timestamp) >= GAP_MAX_AGE_DAYS:
                    continue
                if not gap_is_same_trading_week(cov_gap, candle.timestamp):
                    continue
                is_overnight = gap_is_from_current_overnight(cov_gap, trade_date)
                if is_overnight:
                    continue
                direction = determine_direction(cov_gap, candle)
                dist_ext = distance_to_overnight_extrema(cov_gap, overnight_high, overnight_low)
                if dist_ext >= M1_EXTREMA_DISTANCE:
                    continue
                if not has_1h_alignment(cov_gap, candle, gaps_1h, direction):
                    continue
                t = try_gap_entry(
                    cov_gap, candle, candles_1m, ts_1m, 1,
                    "Covered Gapper (1h aligned)",
                    config.entry_style,
                    candles_15m=candles_15m_all,
                    ts_15m=ts_15m,
                )
                if t:
                    covered_candidates.append(t)

            # ── M2 breakout detection (new: candle-based, no gap required) ──
            if config.enable_m2_rewrite and not pre_rth_m1_exists:
                # Check if current candle is a breakout
                bo_dir = m2_breakout_qualifies_candle(candle, overnight_high, overnight_low)
                if bo_dir is not None:
                    pending_m2 = (candle, bo_dir, M2_RETRACEMENT_WINDOW)
                elif pending_m2 is not None:
                    # Also check candle combination: 09:30 + 09:45
                    et_time = candle.timestamp.astimezone(ET).time()
                    if et_time == time(9, 45) and pending_m2[0].timestamp.astimezone(ET).time() == time(9, 30):
                        combo = m2_combine_candles(pending_m2[0], candle)
                        combo_dir = m2_breakout_qualifies_candle(combo, overnight_high, overnight_low)
                        if combo_dir is not None:
                            pending_m2 = (combo, combo_dir, M2_RETRACEMENT_WINDOW)

                # Try to fill pending M2 on retracement
                if pending_m2 is not None:
                    bo_candle, bo_dir, candles_left = pending_m2
                    if candles_left > 0 and candle.timestamp > bo_candle.timestamp:
                        t = try_m2_breakout_entry(
                            bo_candle, bo_dir, candle, candles_1m, ts_1m,
                            overnight_high, overnight_low, config.m2_sl_mode,
                        )
                        if t:
                            m2_candidates.append(t)
                        pending_m2 = (bo_candle, bo_dir, candles_left - 1)
                    elif candles_left <= 0:
                        pending_m2 = None  # expired

            # ── M3 candidates (only if no M1 and no M2, and no pre-RTH block) ──
            if not m1_setup_available and not m2_candidates and not pre_rth_m1_exists:
                candle_pt = candle.timestamp.astimezone(PT).time()
                if candle_pt >= M3_START_PT and m3_mid is not None:
                    for row in gap_rows:
                        gap = row["gap"]
                        if row["m1_setup"]:
                            continue
                        direction = determine_direction(gap, candle)
                        # M3 opposite direction rule: if M1 potential exists, M3 must be opposite
                        if m1_potential_dirs and direction in m1_potential_dirs:
                            continue
                        t = try_gap_entry(
                            gap, candle, candles_1m, ts_1m, 3,
                            "M3 discount/premium overnight midpoint",
                            config.entry_style,
                            candles_15m=candles_15m_all,
                            ts_15m=ts_15m,
                        )
                        if t:
                            m3_candidates.append(t)

            # ── Selection cascade ──
            selected: list[mb.Trade] = []

            if config.priority_mode == "first_fill":
                # Change 2A: Merge M1 + covered + M2; pick by earliest 1m fill time
                all_candidates = m1_candidates + covered_candidates + m2_candidates
                if all_candidates:
                    all_candidates.sort(key=lambda t: t.entry_time)
                    selected = all_candidates
                elif m3_candidates:
                    selected = m3_candidates

            elif config.priority_mode == "m1_pre_rth":
                # Change 2B: Pre-RTH M1 takes priority; otherwise first-fill M1 vs M2
                if pre_rth_m1_exists and (m1_candidates or covered_candidates):
                    all_m1 = m1_candidates + covered_candidates
                    all_m1.sort(key=lambda t: t.entry_time)
                    selected = all_m1
                else:
                    all_candidates = m1_candidates + covered_candidates + m2_candidates
                    if all_candidates:
                        all_candidates.sort(key=lambda t: t.entry_time)
                        selected = all_candidates
                    elif m3_candidates:
                        selected = m3_candidates

            else:
                # Legacy: strict M1 > M2 > M3
                if m1_candidates or covered_candidates:
                    all_m1 = m1_candidates + covered_candidates
                    all_m1.sort(key=lambda t: distance_to_overnight_extrema(t.gap, overnight_high, overnight_low) if t.gap else 999)
                    selected = all_m1
                elif m2_candidates:
                    selected = m2_candidates
                elif m3_candidates:
                    selected = m3_candidates

            for t in selected:
                if open_trade is not None:
                    break
                ek = f"{t.entry_price:.2f}_{t.stop_loss:.2f}_{t.direction}"
                if ek in daily_entry_keys:
                    continue
                t = simulate_trade_with_partials_and_costs(t, candles_1m, ts_1m)
                all_trades.append(t)
                open_trade = t
                daily_entry_keys.add(ek)
                if t.gap:
                    traded_candle_keys.add(t.gap.candle.timestamp.isoformat())
                break

    eq = equity_df(all_trades)
    buckets = bucket_stats_by_gap_creation_pt(all_trades)
    summary = summarize(all_trades, eq)
    return VariantResult(config=config, trades=all_trades, equity=eq, bucket_stats=buckets, summary=summary)


def plot_equity_comparison(variant_a: VariantResult, variant_b: VariantResult, out_path: str):
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    if not variant_a.equity.empty:
        ax1.plot(pd.to_datetime(variant_a.equity["timestamp"]), variant_a.equity["equity_dollars"],
                 label=variant_a.config.name, linewidth=2.0)
        ax2.plot(pd.to_datetime(variant_a.equity["timestamp"]), -variant_a.equity["drawdown_dollars"],
                 label=f"{variant_a.config.name} DD", linewidth=1.8)
    else:
        ax1.plot([], [], label=f"{variant_a.config.name} (no trades)")
        ax2.plot([], [], label=f"{variant_a.config.name} DD (no trades)")

    if not variant_b.equity.empty:
        ax1.plot(pd.to_datetime(variant_b.equity["timestamp"]), variant_b.equity["equity_dollars"],
                 label=variant_b.config.name, linewidth=2.0)
        ax2.plot(pd.to_datetime(variant_b.equity["timestamp"]), -variant_b.equity["drawdown_dollars"],
                 label=f"{variant_b.config.name} DD", linewidth=1.8)
    else:
        ax1.plot([], [], label=f"{variant_b.config.name} (no trades)")
        ax2.plot([], [], label=f"{variant_b.config.name} DD (no trades)")

    ax1.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"MES Friend Base Strategy: {variant_a.config.name} vs {variant_b.config.name}")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Trade Time (ET)")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bucket_ev_wr(variant: VariantResult, out_path: str):
    plt = _get_plt()
    df = variant.bucket_stats.copy()
    if df.empty:
        return
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    ax1.bar(x - 0.2, df["ev_dollars"], width=0.4, label="EV ($/trade)")
    ax2.plot(x + 0.2, df["win_rate"], marker="o", linewidth=1.8, label="Win rate %")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["bucket_pt"], rotation=45)
    ax1.set_ylabel("EV ($/trade)")
    ax2.set_ylabel("Win rate (%)")
    ax1.set_title(f"{variant.config.name}: Win Rate / EV by Gap-Creation 30m Bucket (PT)")
    ax1.grid(True, axis="y", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "mes_data", "mes_1m.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing data file: {db_path}")

    print("Loading Databento 1m MES...")
    candles_1m = mb.load_databento_data(db_path, start_date=START_DATE, end_date=END_DATE)
    candles_15m_all = mb.build_15m_candles(candles_1m)
    candles_1h_all = build_candles(candles_1m, 60)

    variants = [
        VariantConfig(
            name="V1 Legacy M1>M2>M3",
            m1_gap_cutoff_pt=time(5, 0),
            entry_style="wick",
            immediate_entry=True,
            priority_mode="legacy",
            enable_m2_rewrite=False,
        ),
        VariantConfig(
            name="V2 FirstFill CandleWickSL",
            m1_gap_cutoff_pt=time(5, 0),
            entry_style="wick",
            immediate_entry=True,
            priority_mode="first_fill",
            m2_sl_mode="candle_wick",
            enable_m2_rewrite=True,
        ),
        VariantConfig(
            name="V3 FirstFill ONLevelSL",
            m1_gap_cutoff_pt=time(5, 0),
            entry_style="wick",
            immediate_entry=True,
            priority_mode="first_fill",
            m2_sl_mode="on_level",
            enable_m2_rewrite=True,
        ),
        VariantConfig(
            name="V4 PreRTH CandleWickSL",
            m1_gap_cutoff_pt=time(5, 0),
            entry_style="wick",
            immediate_entry=True,
            priority_mode="m1_pre_rth",
            m2_sl_mode="candle_wick",
            enable_m2_rewrite=True,
        ),
        VariantConfig(
            name="V5 PreRTH ONLevelSL",
            m1_gap_cutoff_pt=time(5, 0),
            entry_style="wick",
            immediate_entry=True,
            priority_mode="m1_pre_rth",
            m2_sl_mode="on_level",
            enable_m2_rewrite=True,
        ),
    ]

    out_dir = os.path.join(base_dir, "outputs", "mes_m1m2m3_rewrite")
    os.makedirs(out_dir, exist_ok=True)

    def safe_name(s: str) -> str:
        out = s.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        return out.replace("/", "_").replace(">", "")

    results: list[VariantResult] = []
    for v in variants:
        print(f"\nRunning {v.name} ...")
        res = run_backtest_silent(run_variant, v, candles_1m, candles_15m_all, candles_1h_all)
        results.append(res)
        s = res.summary
        m2_count = s["by_model"].get(2, {}).get("trades", 0)
        print(
            f"  trades={s['trades']} | WR={s['win_rate']:.2f}% | "
            f"P&L=${s['total_pnl_dollars']:,.2f} | maxDD=${s['max_drawdown_dollars']:,.2f} | "
            f"M2={m2_count}"
        )

    # Save per-variant tables
    summary_rows = []
    for res in results:
        s = res.summary
        row = {
            "variant": res.config.name,
            "trades": s["trades"],
            "wins": s["wins"],
            "win_rate": round(s["win_rate"], 2),
            "total_pnl_dollars": round(s["total_pnl_dollars"], 2),
            "max_drawdown_dollars": round(s["max_drawdown_dollars"], 2),
        }
        for m in (1, 2, 3):
            ms = s["by_model"].get(m, {})
            row[f"M{m}_trades"] = ms.get("trades", 0)
            row[f"M{m}_wr"] = round(ms.get("win_rate", 0), 2)
            row[f"M{m}_pnl"] = round(ms.get("pnl_dollars", 0), 2)
        summary_rows.append(row)

        pd.DataFrame(trade_rows(res.trades)).to_csv(
            os.path.join(out_dir, f"trades_{safe_name(res.config.name)}.csv"),
            index=False,
        )
        res.equity.to_csv(
            os.path.join(out_dir, f"equity_{safe_name(res.config.name)}.csv"),
            index=False,
        )

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Summary text
    summary_txt = os.path.join(out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("M1/M2/M3 Rewrite — 5-Variant Comparison\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n")
        f.write("=" * 80 + "\n\n")
        for res in results:
            s = res.summary
            f.write(f"{res.config.name}\n")
            f.write(f"  trades={s['trades']} wins={s['wins']} WR={s['win_rate']:.2f}%\n")
            f.write(f"  total_pnl=${s['total_pnl_dollars']:,.2f} maxDD=${s['max_drawdown_dollars']:,.2f}\n")
            for m in (1, 2, 3):
                if m in s["by_model"]:
                    ms = s["by_model"][m]
                    f.write(
                        f"  M{m}: trades={ms['trades']} WR={ms['win_rate']:.2f}% "
                        f"P&L=${ms['pnl_dollars']:,.2f}\n"
                    )
            f.write("\n")

    # Equity comparison chart: all variants overlaid
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for res, color in zip(results, colors):
        if not res.equity.empty:
            ax1.plot(pd.to_datetime(res.equity["timestamp"]), res.equity["equity_dollars"],
                     label=res.config.name, linewidth=1.5, color=color)
            ax2.plot(pd.to_datetime(res.equity["timestamp"]), -res.equity["drawdown_dollars"],
                     label=res.config.name, linewidth=1.2, color=color, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("M1/M2/M3 Rewrite — 5-Variant Comparison")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Trade Time (ET)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equity_comparison_5var.png"), dpi=180)
    plt.close(fig)

    print(f"\nAll outputs saved to: {out_dir}")

    print(f"\nSaved outputs:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
