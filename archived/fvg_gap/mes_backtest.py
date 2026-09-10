"""
MES Futures Gap Fill Backtest
==============================
Applies the same gap fill strategy to Micro E-mini S&P 500 (MES) futures
using 1m candle data from TradingView or Databento CSV exports.

MES specifics:
  - $5 per point per contract (tick = 0.25 pts = $1.25)
  - Thresholds scaled 10x from SPY (MES ≈ SPX ≈ 10x SPY)
  - Dynamic sizing: $600 risk budget / (SL distance * $5) = contracts
  - RTH session: 9:30-16:00 ET for gaps & entries
  - Overnight: 16:00-9:30 ET for O/N high/low

Data sources:
  - TradingView: unix timestamps, columns: time,open,high,low,close,volume
  - Databento: ISO timestamps, columns: ts_event,...,open,high,low,close,volume,symbol
    Databento has individual quarterly contracts (MESH2, MESM2, etc.)
    We build a continuous front-month by rolling on daily volume crossover.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo
from bisect import bisect_left, bisect_right

ET = ZoneInfo("US/Eastern")

# ── MES-scaled thresholds (10x SPY) ──
PENETRATION = 3.0           # 10x SPY $0.30
MIN_VALID_GAP_PTS = 4.0     # 10x SPY $0.40
GAP_MAX_SIZE_PTS = 25.0     # 10x SPY $2.50
OVERSIZED_GAP_PTS = 26.0    # 10x SPY $2.60
LARGE_STOP_PTS = 17.0       # 10x SPY $1.70
PARTIAL_PROFIT_PTS = 14.0   # 10x SPY $1.40
BREAKOUT_SIGNIFICANCE_PTS = 3.0   # 10x SPY $0.30
BREAKOUT_CANDLE_MIN_PTS = 14.0    # 10x SPY $1.40
MIN_RISK = 1.0              # minimum risk in MES points
SL_BUFFER = 3.0             # 3 pts past the wick for SL placement
MIN_SL_PTS = 8.0            # minimum 8 pts raw SL before buffer

# ── Time boundaries (ET) ──
TRADE_WINDOW_START = time(9, 30)
TRADE_WINDOW_END = time(15, 0)
OVERNIGHT_START = time(18, 0)   # 15:00 PT = 18:00 ET
OVERNIGHT_END = time(9, 30)    # 06:30 PT = 09:30 ET
GAP_MAX_AGE_DAYS = 21

# ── Position sizing ──
MES_MULTIPLIER = 5.0        # $5 per point per contract
RISK_BUDGET = 600.0         # $600 risk per trade


# ── Data Classes ──

@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_top: float = field(init=False, repr=False)
    body_bottom: float = field(init=False, repr=False)
    body_size: float = field(init=False, repr=False)
    is_bullish: bool = field(init=False, repr=False)

    def __post_init__(self):
        self.body_top = max(self.open, self.close)
        self.body_bottom = min(self.open, self.close)
        self.body_size = self.body_top - self.body_bottom
        self.is_bullish = self.close >= self.open


@dataclass
class Gap:
    gap_top: float
    gap_bottom: float
    candle: Candle
    direction: str        # 'up' or 'down'
    created_at: pd.Timestamp
    filled_at: Optional[pd.Timestamp] = None  # when fully covered (None = still open)

    @property
    def size(self) -> float:
        return self.gap_top - self.gap_bottom

    def is_valid_age(self, current_time: pd.Timestamp) -> bool:
        age = (current_time - self.created_at).total_seconds() / 86400
        return age <= GAP_MAX_AGE_DAYS


@dataclass
class Trade:
    model: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None          # in MES points
    pnl_dollars: Optional[float] = None  # in dollars
    result: Optional[str] = None
    gap: Optional[Gap] = None
    notes: str = ""
    num_contracts: int = 1
    is_reentry: bool = False
    gap_size_at_entry: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_favorable_excursion_r: Optional[float] = None
    mfe_pts: float = 0.0   # Maximum Favorable Excursion in points
    mae_pts: float = 0.0   # Maximum Adverse Excursion in points (positive number)
    bars_to_mfe: int = 0          # bar index when MFE was reached
    bars_to_mae: int = 0          # bar index when MAE was reached
    mfe_before_mae: Optional[bool] = None  # did trade go favorable before adverse?


# ── Data Loading ──

def load_mes_data(*csv_paths) -> list[Candle]:
    """Load MES 1m candles from TradingView CSV exports, merge and dedupe."""
    all_rows = {}  # keyed by timestamp for dedup
    for path in csv_paths:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                ts = int(row[0])
                if ts not in all_rows:
                    all_rows[ts] = row

    candles = []
    for ts in sorted(all_rows.keys()):
        row = all_rows[ts]
        dt = pd.Timestamp(datetime.fromtimestamp(int(row[0]), tz=ET))
        candles.append(Candle(
            timestamp=dt,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=int(row[5]),
        ))
    return candles


def load_databento_data(csv_path: str, start_date: str = None,
                        end_date: str = None) -> list[Candle]:
    """Load MES 1m candles from Databento CSV, building a continuous contract.

    Databento provides individual quarterly contracts (MESH2, MESM2, MESU2, MESZ2, ...).
    We build a continuous front-month by selecting the highest-volume contract per day,
    rolling when the next contract's daily volume exceeds the current.

    Args:
        csv_path: Path to decompressed Databento CSV.
        start_date: Optional 'YYYY-MM-DD' filter (inclusive).
        end_date: Optional 'YYYY-MM-DD' filter (inclusive).
    """
    print("  Loading Databento CSV with pandas (fast)...")

    # Load with pandas for speed — only columns we need
    df = pd.read_csv(csv_path, usecols=["ts_event", "open", "high", "low",
                                         "close", "volume", "symbol"])

    # Filter out spread symbols
    df = df[~df["symbol"].str.contains("-", na=False)].copy()

    # Extract date string for filtering
    df["date"] = df["ts_event"].str[:10]

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    print(f"  Rows after date filter: {len(df):,}")

    # Determine front-month per day via daily volume
    daily_vol = df.groupby(["date", "symbol"])["volume"].sum().reset_index()
    front_month_idx = daily_vol.groupby("date")["volume"].idxmax()
    front_month_map = daily_vol.loc[front_month_idx].set_index("date")["symbol"].to_dict()

    contracts_used = sorted(set(front_month_map.values()))
    print(f"  Contracts used: {', '.join(contracts_used)}")

    # Count roll events
    prev = None
    rolls = 0
    for date in sorted(front_month_map.keys()):
        if prev and front_month_map[date] != prev:
            rolls += 1
        prev = front_month_map[date]
    print(f"  Roll events: {rolls}")

    # Filter to only front-month bars
    df["front"] = df["date"].map(front_month_map)
    df = df[df["symbol"] == df["front"]].copy()

    # Parse timestamps -> ET
    print("  Converting timestamps...")
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(ET)
    df = df.sort_values("ts").reset_index(drop=True)

    # Build candle list — vectorized extraction to avoid slow iterrows()
    print("  Building candle objects...")
    timestamps = df["ts"].tolist()
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    candles = [
        Candle(timestamp=timestamps[i], open=opens[i], high=highs[i],
               low=lows[i], close=closes[i], volume=int(volumes[i]))
        for i in range(len(df))
    ]

    print(f"  Total front-month 1m candles: {len(candles):,}")
    if candles:
        print(f"  Date range: {candles[0].timestamp.strftime('%Y-%m-%d')} to "
              f"{candles[-1].timestamp.strftime('%Y-%m-%d')}")
    return candles


def build_15m_candles(candles_1m: list[Candle]) -> list[Candle]:
    """Aggregate 1m candles into 15m candles."""
    buckets = {}
    for c in candles_1m:
        # Round down to 15m boundary
        minute = (c.timestamp.minute // 15) * 15
        bucket_ts = c.timestamp.replace(minute=minute, second=0, microsecond=0)
        if bucket_ts not in buckets:
            buckets[bucket_ts] = []
        buckets[bucket_ts].append(c)

    candles_15m = []
    for ts in sorted(buckets.keys()):
        bars = buckets[ts]
        candles_15m.append(Candle(
            timestamp=ts,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
        ))
    return candles_15m


def filter_rth(candles: list[Candle]) -> list[Candle]:
    """Filter candles to RTH only (9:30-16:00 ET)."""
    return [c for c in candles if TRADE_WINDOW_START <= c.timestamp.time() < time(16, 1)]


# ── Gap Detection ──

def detect_gaps(candles: list[Candle], current_time: pd.Timestamp = None,
                body_fill_first: bool = False) -> list[Gap]:
    """Detect unfilled candle body gaps.

    A gap is the full candle body — the imbalance created by that candle.
    The prior candle's wick does NOT reduce the gap; only subsequent
    price action (wicks/bodies after the gap candle) erodes it.

    body_fill_first: if True, the first candle after the gapper uses body
    (open/close) instead of wick (high/low) to determine what's filled.
    All subsequent candles still use full wick range.
    """
    if len(candles) < 2:
        return []

    gaps = []
    for i, candle in enumerate(candles):
        if candle.body_size < 0.01:
            continue

        # Full candle body = gap zone
        unfilled_ranges: list[tuple[float, float]] = [(candle.body_bottom, candle.body_top)]

        for j in range(i + 1, len(candles)):
            later = candles[j]
            # Skip 4:00 PM closing auction candle
            lt = later.timestamp
            if lt.hour == 16 and lt.minute == 0:
                continue
            # First candle after gapper: optionally use body instead of wick
            if body_fill_first and j == i + 1:
                cover_low = later.body_bottom
                cover_high = later.body_top
            else:
                cover_low = later.low
                cover_high = later.high

            new_ranges = []
            for (bot, top) in unfilled_ranges:
                if cover_low <= bot and cover_high >= top:
                    continue
                if cover_high <= bot or cover_low >= top:
                    new_ranges.append((bot, top))
                    continue
                if cover_low > bot:
                    new_ranges.append((bot, min(cover_low, top)))
                if cover_high < top:
                    new_ranges.append((max(cover_high, bot), top))

            unfilled_ranges = new_ranges
            if not unfilled_ranges:
                break

        for (bot, top) in unfilled_ranges:
            if top - bot < 0.01:
                continue
            direction = "up" if candle.is_bullish else "down"
            gap = Gap(gap_top=top, gap_bottom=bot, candle=candle,
                      direction=direction, created_at=candle.timestamp)
            if current_time and gap.is_valid_age(current_time):
                gaps.append(gap)
            elif current_time is None:
                gaps.append(gap)

    return gaps


# ── Precomputed Gap Index ──

def precompute_gap_index(candles: list[Candle],
                         body_fill_first: bool = False,
                         ) -> tuple[dict[pd.Timestamp, list[Gap]], list[pd.Timestamp]]:
    """Precompute the set of unfilled gaps at every candle timestamp.

    Forward-scans candles once, maintaining a running set of active gaps and
    eroding them incrementally.  Stores a snapshot at each candle timestamp.

    Returns (snapshots_dict, sorted_timestamps) where:
      snapshots_dict[ts] = list of Gap objects with current unfilled ranges
      sorted_timestamps = sorted list of keys for binary search

    Complexity: O(N * G_avg) — one pass, ~150K ops for 90-day window.
    """
    if len(candles) < 2:
        return {}, []

    # Active gaps: list of [gap_candle, direction, ranges, gap_array_idx]
    # ranges = list of (bot, top) tuples
    # gap_array_idx = index in candles[] where this gap was created
    active: list[list] = []
    snapshots: dict[pd.Timestamp, list[Gap]] = {}

    for i, candle in enumerate(candles):
        ts = candle.timestamp
        is_auction = (ts.hour == 16 and ts.minute == 0)

        # 1. Erode all active gaps by this candle's price range
        if not is_auction:
            new_active = []
            for entry in active:
                gap_candle, direction, ranges, gap_idx = entry

                # body_fill_first: only for the candle immediately after
                # the gap candle (j == i + 1 in original detect_gaps)
                use_body = body_fill_first and (i == gap_idx + 1)
                if use_body:
                    cover_low = candle.body_bottom
                    cover_high = candle.body_top
                else:
                    cover_low = candle.low
                    cover_high = candle.high

                new_ranges = []
                for (bot, top) in ranges:
                    if cover_low <= bot and cover_high >= top:
                        continue  # fully covered
                    if cover_high <= bot or cover_low >= top:
                        new_ranges.append((bot, top))
                        continue
                    if cover_low > bot:
                        new_ranges.append((bot, min(cover_low, top)))
                    if cover_high < top:
                        new_ranges.append((max(cover_high, bot), top))

                if new_ranges:
                    entry[2] = new_ranges
                    new_active.append(entry)

            active = new_active

        # 2. Create new gap from this candle (if body has size)
        if candle.body_size >= 0.01:
            direction = "up" if candle.is_bullish else "down"
            active.append([candle, direction,
                           [(candle.body_bottom, candle.body_top)],
                           i])  # store array index for body_fill_first check

        # 3. Store snapshot: convert active gaps to Gap objects
        gap_list = []
        for entry in active:
            gap_candle, direction, ranges, _ = entry
            for (bot, top) in ranges:
                if top - bot < 0.01:
                    continue
                gap_list.append(Gap(
                    gap_top=top, gap_bottom=bot, candle=gap_candle,
                    direction=direction, created_at=gap_candle.timestamp,
                ))
        snapshots[ts] = gap_list

    sorted_ts = sorted(snapshots.keys())
    return snapshots, sorted_ts


def find_gaps_at_time_fast(
    snapshots: dict[pd.Timestamp, list[Gap]],
    sorted_ts: list[pd.Timestamp],
    at_time: pd.Timestamp,
    lookback_days: float = 25,
    immediate_entry: bool = False,
) -> list[Gap]:
    """Fast gap lookup from precomputed snapshots.

    Uses binary search to find the snapshot at or before at_time,
    then filters by age (lookback_days).
    """
    if not sorted_ts:
        return []

    # immediate_entry=True: exclude the current candle → use snapshot BEFORE at_time
    # immediate_entry=False: include current candle → use snapshot AT or before at_time
    if immediate_entry:
        idx = bisect_left(sorted_ts, at_time) - 1
    else:
        idx = bisect_right(sorted_ts, at_time) - 1

    if idx < 0:
        return []

    snap_ts = sorted_ts[idx]
    gaps = snapshots.get(snap_ts, [])

    # Filter by lookback age + GAP_MAX_AGE_DAYS (matches detect_gaps is_valid_age filter)
    cutoff = at_time - timedelta(days=lookback_days)
    return [g for g in gaps
            if g.created_at >= cutoff
            and g.created_at < at_time
            and g.is_valid_age(at_time)]


def clear_gap_caches():
    """Clear module-level gap caches between search candidates."""
    global _gap_cache, _true_gap_cache
    _gap_cache.clear()
    _true_gap_cache.clear()


# ── True (Session) Gap Detection ──

def detect_true_gaps(candles: list[Candle], current_time: pd.Timestamp = None,
                     min_gap_minutes: int = 60) -> list[Gap]:
    """Detect session-to-session true gaps (price jumps across time breaks).

    A true gap forms when there is a large time break between consecutive
    candles (e.g. Friday close → Sunday open, or daily close → open) and
    price jumps.

    The gap zone stretches from the *close* of the before-candle to the
    *close* of the after-candle (the after-candle's close extends the zone
    when it pushes further in the gap direction — more inefficiency).
    The after-candle's wick then partially fills the zone.  Subsequent
    candles continue eroding whatever remains.

    Returns Gap objects where `candle` is the after-candle (the one that
    created the gap by opening away from prior close).
    """
    if len(candles) < 3:
        return []

    gaps: list[Gap] = []

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        after = candles[i]

        # Check for a time break (session boundary)
        gap_minutes = (after.timestamp - prev.timestamp).total_seconds() / 60.0
        if gap_minutes < min_gap_minutes:
            continue

        # Determine gap direction and zone edges
        # Zone: from prev.close to after.close
        zone_bot = min(prev.close, after.close)
        zone_top = max(prev.close, after.close)

        if zone_top - zone_bot < MIN_VALID_GAP_PTS:
            continue

        # The after-candle's wick partially fills the zone
        cover_low = after.low
        cover_high = after.high

        # Compute unfilled ranges
        unfilled_ranges: list[tuple[float, float]] = []
        if zone_bot < cover_low:
            unfilled_ranges.append((zone_bot, min(cover_low, zone_top)))
        if zone_top > cover_high:
            unfilled_ranges.append((max(cover_high, zone_bot), zone_top))
        # If the wick covers the entire zone from inside (zone within wick)
        if not unfilled_ranges and cover_low <= zone_bot and cover_high >= zone_top:
            continue  # fully covered by the after candle
        if not unfilled_ranges:
            # Wick may sit entirely outside the zone — full zone unfilled
            if cover_high <= zone_bot or cover_low >= zone_top:
                unfilled_ranges.append((zone_bot, zone_top))
            else:
                continue

        # Erode by subsequent candles
        for j in range(i + 1, len(candles)):
            later = candles[j]
            lt = later.timestamp
            if lt.hour == 16 and lt.minute == 0:
                continue

            new_ranges = []
            for (bot, top) in unfilled_ranges:
                if later.low <= bot and later.high >= top:
                    continue
                if later.high <= bot or later.low >= top:
                    new_ranges.append((bot, top))
                    continue
                if later.low > bot:
                    new_ranges.append((bot, min(later.low, top)))
                if later.high < top:
                    new_ranges.append((max(later.high, bot), top))

            unfilled_ranges = new_ranges
            if not unfilled_ranges:
                break

        # Gap direction: "up" if price gapped higher, "down" if lower
        direction = "up" if after.close > prev.close else "down"

        for (bot, top) in unfilled_ranges:
            if top - bot < 0.01:
                continue
            gap = Gap(gap_top=top, gap_bottom=bot, candle=after,
                      direction=direction, created_at=after.timestamp)
            if current_time and gap.is_valid_age(current_time):
                gaps.append(gap)
            elif current_time is None:
                gaps.append(gap)

    return gaps


_true_gap_cache: dict = {}


def find_unfilled_true_gaps_at_time(candles_15m: list[Candle], at_time: pd.Timestamp,
                                     lookback_days: int = 25,
                                     ts_index_15m: list = None) -> list[Gap]:
    """Find unfilled session-to-session true gaps at a given time."""
    cutoff = at_time - timedelta(days=lookback_days)
    if ts_index_15m is not None:
        lo = bisect_left(ts_index_15m, cutoff)
        hi = bisect_left(ts_index_15m, at_time)
        relevant = candles_15m[lo:hi]
    else:
        relevant = [c for c in candles_15m if cutoff <= c.timestamp < at_time]
    if len(relevant) < 3:
        return []

    cache_key = (len(relevant), relevant[-1].timestamp, "true_gap")
    if cache_key in _true_gap_cache:
        return [g for g in _true_gap_cache[cache_key] if g.created_at < at_time]

    gaps = detect_true_gaps(relevant, current_time=at_time)
    _true_gap_cache[cache_key] = gaps
    if len(_true_gap_cache) > 5000:
        _true_gap_cache.clear()

    return [g for g in gaps if g.created_at < at_time]


_gap_cache = {}
_ts_index_15m = None

def find_unfilled_gaps_at_time(candles_15m: list[Candle], at_time: pd.Timestamp,
                                lookback_days: int = 25,
                                ts_index_15m: list = None,
                                immediate_entry: bool = False,
                                body_fill_first: bool = False) -> list[Gap]:
    """Find unfilled gaps at a given time.

    immediate_entry: if True, exclude the current candle from gap detection so
    the candle immediately after the gapper can trigger an entry (the gap is
    computed from candles *before* the current bar only).
    body_fill_first: if True, the first candle after each gapper uses body
    (close) instead of wick to determine fill. Gives shallower entry levels.
    """
    cutoff = at_time - timedelta(days=lookback_days)
    if ts_index_15m is not None:
        lo = bisect_left(ts_index_15m, cutoff)
        hi = bisect_left(ts_index_15m, at_time) if immediate_entry else bisect_right(ts_index_15m, at_time)
        relevant = candles_15m[lo:hi]
    else:
        if immediate_entry:
            relevant = [c for c in candles_15m if cutoff <= c.timestamp < at_time]
        else:
            relevant = [c for c in candles_15m if cutoff <= c.timestamp <= at_time]
    if len(relevant) < 2:
        return []

    cache_key = (len(relevant), relevant[-1].timestamp, immediate_entry, body_fill_first)
    if cache_key in _gap_cache:
        return [g for g in _gap_cache[cache_key] if g.created_at < at_time]

    gaps = detect_gaps(relevant, current_time=at_time, body_fill_first=body_fill_first)
    _gap_cache[cache_key] = gaps
    if len(_gap_cache) > 5000:
        _gap_cache.clear()

    return [g for g in gaps if g.created_at < at_time]


# ── Overnight H/L ──

def _build_timestamp_index(candles_1m: list[Candle]):
    """Pre-build sorted timestamp array for binary search."""
    return [c.timestamp for c in candles_1m]

_ts_index = None

def get_overnight_high_low(candles_1m: list[Candle], trade_date: pd.Timestamp,
                           ts_index: list = None):
    """Get overnight H/L from 1m candles: 18:00 ET prev day to 9:30 ET trade day.
    (15:00 PT to 06:30 PT overnight session.)
    Uses binary search when ts_index is provided for O(log n) lookup."""
    prev_day = trade_date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    on_start = prev_day.replace(hour=OVERNIGHT_START.hour, minute=OVERNIGHT_START.minute,
                                second=0, microsecond=0)
    on_end = trade_date.replace(hour=OVERNIGHT_END.hour, minute=OVERNIGHT_END.minute,
                                second=0, microsecond=0)

    if ts_index is not None:
        lo = bisect_left(ts_index, on_start)
        hi = bisect_left(ts_index, on_end)
        overnight = candles_1m[lo:hi]
    else:
        overnight = [c for c in candles_1m if on_start <= c.timestamp < on_end]

    if not overnight:
        return None, None
    return max(c.high for c in overnight), min(c.low for c in overnight)


# ── Gap Classification ──

def classify_gap(gap: Gap, overnight_high, overnight_low) -> int:
    if overnight_high is None:
        return 1

    candle = gap.candle
    overnight_mid = (overnight_high + overnight_low) / 2.0

    # Model 2: breakout candle
    if candle.high > overnight_high + BREAKOUT_SIGNIFICANCE_PTS:
        effective_size = candle.high - candle.body_bottom
        if effective_size >= BREAKOUT_CANDLE_MIN_PTS:
            return 2
    if candle.low < overnight_low - BREAKOUT_SIGNIFICANCE_PTS:
        effective_size = candle.body_top - candle.low
        if effective_size >= BREAKOUT_CANDLE_MIN_PTS:
            return 2

    # Model 3: discount/premium zone
    overnight_range = overnight_high - overnight_low
    if overnight_range >= MIN_VALID_GAP_PTS:
        in_discount = gap.gap_top <= overnight_mid and gap.gap_bottom >= overnight_low - 1.0
        in_premium = gap.gap_bottom >= overnight_mid and gap.gap_top <= overnight_high + 1.0
        if in_discount or in_premium:
            return 3

    return 1


# ── Entry Logic ──

def try_gap_entry_1m(gap: Gap, candle_15m: Candle, candles_1m: list[Candle],
                     overnight_high, overnight_low,
                     is_reentry: bool = False,
                     ts_index: list = None,
                     candles_15m: list = None,
                     ts_index_15m: list = None) -> Optional[Trade]:
    """Check if 1m candles within this 15m window trigger a gap entry."""
    ct = candle_15m.timestamp.time()
    if ct < TRADE_WINDOW_START or ct >= TRADE_WINDOW_END:
        return None
    if not gap.is_valid_age(candle_15m.timestamp):
        return None

    body_top = gap.candle.body_top
    body_bottom = gap.candle.body_bottom

    if candle_15m.open >= body_top:
        direction = "long"
    elif candle_15m.open <= body_bottom:
        direction = "short"
    else:
        mid = (body_top + body_bottom) / 2
        direction = "long" if candle_15m.open >= mid else "short"

    if is_reentry:
        edge_top = gap.gap_top
        edge_bottom = gap.gap_bottom
    else:
        edge_top = body_top
        edge_bottom = body_bottom

    if direction == "long":
        entry_price = edge_top - PENETRATION
        raw_sl_wick = gap.candle.low
    else:
        entry_price = edge_bottom + PENETRATION
        raw_sl_wick = gap.candle.high

    # Check raw SL distance (before buffer). If < MIN_SL_PTS, walk back
    # to previous candle and use its wick as the base.
    raw_risk = abs(raw_sl_wick - entry_price)
    if raw_risk < MIN_SL_PTS and candles_15m is not None and ts_index_15m is not None:
        # Find the gap candle's position and walk back one
        gap_ts = gap.candle.timestamp
        idx = bisect_left(ts_index_15m, gap_ts)
        if idx > 0:
            prev_candle = candles_15m[idx - 1]
            if direction == "long":
                raw_sl_wick = min(raw_sl_wick, prev_candle.low)
            else:
                raw_sl_wick = max(raw_sl_wick, prev_candle.high)

    # Add 3pt buffer past the wick
    if direction == "long":
        sl = raw_sl_wick - SL_BUFFER
    else:
        sl = raw_sl_wick + SL_BUFFER

    if direction == "short" and sl <= entry_price:
        return None
    if direction == "long" and sl >= entry_price:
        return None

    risk = abs(sl - entry_price)
    if risk < MIN_RISK:
        return None
    tp = entry_price + risk if direction == "long" else entry_price - risk

    # Find 1m bar where penetration occurs
    window_start = candle_15m.timestamp
    window_end = window_start + timedelta(minutes=15)
    if ts_index is not None:
        lo = bisect_left(ts_index, window_start)
        hi = bisect_left(ts_index, window_end)
        bars_1m = candles_1m[lo:hi]
    else:
        bars_1m = [c for c in candles_1m if window_start <= c.timestamp < window_end]

    entry_time = None
    for bar in bars_1m:
        if direction == "long" and bar.low <= entry_price:
            entry_time = bar.timestamp
            break
        elif direction == "short" and bar.high >= entry_price:
            entry_time = bar.timestamp
            break

    if entry_time is None:
        return None

    model = classify_gap(gap, overnight_high, overnight_low)
    entry_side = "from above" if direction == "long" else "from below"
    notes = f"M{model} {direction} ({entry_side})"
    if is_reentry:
        notes += f" [RE-ENTRY frag={gap.size:.1f}]"
    if risk >= LARGE_STOP_PTS:
        notes += f" [Large stop: partial at +{PARTIAL_PROFIT_PTS:.1f}]"

    return Trade(
        model=model, direction=direction,
        entry_price=entry_price, stop_loss=sl, take_profit=tp,
        entry_time=entry_time, gap=gap, notes=notes,
        is_reentry=is_reentry, gap_size_at_entry=gap.size,
    )


# ── Trade Simulation ──

def simulate_trade(trade: Trade, candles_1m: list[Candle],
                   ts_index: list = None) -> Trade:
    """Simulate trade forward through 1m candles to determine outcome."""
    if ts_index is not None:
        lo = bisect_right(ts_index, trade.entry_time)
        # Find end of day (next day midnight or 17:00)
        eod = trade.entry_time.normalize() + timedelta(days=1)
        hi = bisect_left(ts_index, eod)
        future = candles_1m[lo:hi]
    else:
        future = [c for c in candles_1m
                  if c.timestamp > trade.entry_time
                  and c.timestamp.date() == trade.entry_time.date()]

    mae, mfe = 0.0, 0.0
    risk = abs(trade.entry_price - trade.stop_loss)

    for candle in future:
        if trade.direction == "long":
            adverse = trade.entry_price - candle.low
            favorable = candle.high - trade.entry_price
        else:
            adverse = candle.high - trade.entry_price
            favorable = trade.entry_price - candle.low
        mae = max(mae, adverse)
        mfe = max(mfe, favorable)

        if trade.direction == "long":
            if candle.low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_time = candle.timestamp
                trade.pnl = trade.stop_loss - trade.entry_price
                trade.result = "loss"
                break
            if candle.high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = candle.timestamp
                trade.pnl = trade.take_profit - trade.entry_price
                trade.result = "win"
                break
        else:
            if candle.high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_time = candle.timestamp
                trade.pnl = trade.entry_price - trade.stop_loss
                trade.result = "loss"
                break
            if candle.low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = candle.timestamp
                trade.pnl = trade.entry_price - trade.take_profit
                trade.result = "win"
                break

        # EOD exit
        if candle.timestamp.time() >= time(16, 0):
            trade.exit_price = candle.close
            trade.exit_time = candle.timestamp
            trade.pnl = (candle.close - trade.entry_price if trade.direction == "long"
                         else trade.entry_price - candle.close)
            trade.result = "win" if trade.pnl > 0 else "loss"
            trade.notes += " [EOD exit]"
            break
    else:
        if future:
            last = future[-1]
            trade.exit_price = last.close
            trade.exit_time = last.timestamp
            trade.pnl = (last.close - trade.entry_price if trade.direction == "long"
                         else trade.entry_price - last.close)
            trade.result = "win" if trade.pnl > 0 else "loss"
            trade.notes += " [data end exit]"

    trade.max_adverse_excursion = mae
    trade.max_favorable_excursion = mfe
    trade.max_favorable_excursion_r = mfe / risk if risk > 0 else 0.0
    return trade


# ── Position Sizing ──

def compute_contracts(risk_pts: float, risk_budget: float = RISK_BUDGET) -> int:
    """Dynamic sizing: $600 risk / (SL distance × $5/pt) = contracts."""
    risk_dollars = risk_pts * MES_MULTIPLIER
    if risk_dollars <= 0:
        return 1
    return max(1, int(risk_budget / risk_dollars))


# ── Backtest Engine ──

def run_mes_backtest(candles_1m: list[Candle], candles_15m_rth: list[Candle],
                     skip_lunch: bool = True,
                     sl_mult: float = 1.0, tp_r: float = 1.0,
                     adaptive_config: dict = None,
                     immediate_entry: bool = False,
                     body_fill_first: bool = False,
                     max_gap_age_days: float = None,
                     label: str = "") -> list[Trade]:
    """Run the full MES gap fill backtest.

    sl_mult: multiply the raw SL distance (0.85 = tighter stop).
    tp_r: R-multiple for TP (1.25 = 1.25R reward).
    adaptive_config: dict of {(lo,hi): (sl_m, tp_r)} for per-gap-size SL/TP.
    immediate_entry: allow entry on the candle right after the gapper.
    body_fill_first: use body (not wick) of confirmation candle for fill calc.
    max_gap_age_days: if set, skip gaps older than this many days.
    label: display label for this run.
    """
    # Pre-build timestamp indices for binary search
    ts_index = _build_timestamp_index(candles_1m)
    ts_index_15m = [c.timestamp for c in candles_15m_rth]

    # Get unique trading days
    trading_days = sorted(set(
        c.timestamp.normalize() for c in candles_15m_rth
        if c.timestamp.weekday() < 5
    ))

    mode_label = label or "baseline"
    if immediate_entry:
        mode_label += " [IMMEDIATE ENTRY]"
    if body_fill_first:
        mode_label += " [BODY FILL]"
    print(f"\nTrading days: {len(trading_days)}")
    print(f"15m RTH candles: {len(candles_15m_rth)}")
    print(f"1m candles: {len(candles_1m)}")
    print(f"Mode: {mode_label}")
    print(f"{'='*60}\n")

    all_trades = []
    traded_candle_bodies = set()

    def entry_key(t: Trade) -> str:
        return f"{t.entry_price:.2f}_{t.stop_loss:.2f}_{t.direction}"

    for day_idx, trade_date in enumerate(trading_days):
        day_str = trade_date.strftime("%Y-%m-%d")
        if day_idx % 50 == 0:
            print(f"  Processing day {day_idx+1}/{len(trading_days)} ({day_str})...")

        overnight_high, overnight_low = get_overnight_high_low(candles_1m, trade_date,
                                                                ts_index=ts_index)

        window_start = trade_date.replace(hour=TRADE_WINDOW_START.hour,
                                           minute=TRADE_WINDOW_START.minute)
        window_end = trade_date.replace(hour=TRADE_WINDOW_END.hour,
                                         minute=TRADE_WINDOW_END.minute)

        window_candles = [c for c in candles_15m_rth
                          if window_start <= c.timestamp < window_end]
        if not window_candles:
            continue

        day_trades = []
        daily_entry_keys = set()
        open_trade = None  # 1 position at a time (no hedging — prop firm rule)

        for candle in window_candles:
            if open_trade and open_trade.exit_time and open_trade.exit_time <= candle.timestamp:
                open_trade = None

            current_gaps = find_unfilled_gaps_at_time(candles_15m_rth, candle.timestamp,
                                                         ts_index_15m=ts_index_15m,
                                                         immediate_entry=immediate_entry,
                                                         body_fill_first=body_fill_first)
            current_gaps.sort(key=lambda g: -g.size)

            for gap in current_gaps:
                if gap.size <= MIN_VALID_GAP_PTS:
                    continue
                if gap.size > GAP_MAX_SIZE_PTS:
                    continue
                if max_gap_age_days is not None:
                    age = (candle.timestamp - gap.created_at).total_seconds() / 86400
                    if age > max_gap_age_days:
                        continue

                candle_body_id = gap.candle.timestamp.isoformat()
                is_reentry = candle_body_id in traded_candle_bodies
                if is_reentry:
                    continue  # no re-entries

                trade = try_gap_entry_1m(gap, candle, candles_1m,
                                         overnight_high, overnight_low,
                                         is_reentry=False,
                                         ts_index=ts_index,
                                         candles_15m=candles_15m_rth,
                                         ts_index_15m=ts_index_15m)
                if trade:
                    # Apply SL/TP multipliers
                    gap_sz = gap.size
                    if adaptive_config:
                        eff_sl_mult, eff_tp_r = sl_mult, tp_r  # fallback
                        for (lo, hi), (sm, tr) in adaptive_config.items():
                            if lo <= gap_sz < hi:
                                eff_sl_mult, eff_tp_r = sm, tr
                                break
                    else:
                        eff_sl_mult, eff_tp_r = sl_mult, tp_r

                    if eff_sl_mult != 1.0 or eff_tp_r != 1.0:
                        risk = abs(trade.entry_price - trade.stop_loss)
                        new_risk = risk * eff_sl_mult
                        if trade.direction == "long":
                            trade.stop_loss = trade.entry_price - new_risk
                            trade.take_profit = trade.entry_price + new_risk * eff_tp_r
                        else:
                            trade.stop_loss = trade.entry_price + new_risk
                            trade.take_profit = trade.entry_price - new_risk * eff_tp_r

                    # Skip lunch
                    if skip_lunch and trade.entry_time is not None:
                        et = trade.entry_time.time()
                        if time(12, 30) <= et < time(14, 0):
                            continue

                    # Max 1 position at a time (prop firm — no hedging)
                    if open_trade is not None:
                        continue

                    ek = entry_key(trade)
                    if ek in daily_entry_keys:
                        continue

                    # Simulate trade
                    trade = simulate_trade(trade, candles_1m, ts_index=ts_index)

                    # Dynamic sizing
                    risk_pts = abs(trade.entry_price - trade.stop_loss)
                    nc = compute_contracts(risk_pts)
                    trade.num_contracts = nc
                    trade.pnl_dollars = round(trade.pnl * MES_MULTIPLIER * nc, 2) if trade.pnl else 0

                    day_trades.append(trade)
                    traded_candle_bodies.add(candle_body_id)
                    daily_entry_keys.add(ek)

                    open_trade = trade

        for trade in day_trades:
            all_trades.append(trade)
            result_str = f"{'WIN' if trade.result == 'win' else 'LOSS'}"
            pnl_str = f"${trade.pnl_dollars:+.2f}" if trade.pnl_dollars else "N/A"
            print(f"  {day_str} | M{trade.model} {trade.direction:5s} | "
                  f"Entry: {trade.entry_price:.2f} | "
                  f"SL: {trade.stop_loss:.2f} | TP: {trade.take_profit:.2f} | "
                  f"{result_str} {trade.pnl:+.2f} pts ({pnl_str}) | "
                  f"x{trade.num_contracts} | {trade.notes}")

    return all_trades


# ── Reporting ──

def print_report(trades: list[Trade]):
    if not trades:
        print("\nNo trades generated.")
        return

    print(f"\n{'='*60}")
    print("MES FUTURES BACKTEST RESULTS")
    print(f"{'='*60}\n")

    total = len(trades)
    wins = sum(1 for t in trades if t.result == "win")
    losses = total - wins
    wr = wins / total * 100 if total > 0 else 0

    total_pnl_pts = sum(t.pnl for t in trades if t.pnl)
    total_pnl_dollars = sum(t.pnl_dollars for t in trades if t.pnl_dollars)
    avg_pnl_pts = total_pnl_pts / total if total > 0 else 0
    avg_pnl_dollars = total_pnl_dollars / total if total > 0 else 0

    avg_win_pts = np.mean([t.pnl for t in trades if t.result == "win" and t.pnl]) if wins else 0
    avg_loss_pts = np.mean([t.pnl for t in trades if t.result == "loss" and t.pnl]) if losses else 0
    avg_win_dollars = np.mean([t.pnl_dollars for t in trades if t.result == "win" and t.pnl_dollars]) if wins else 0
    avg_loss_dollars = np.mean([t.pnl_dollars for t in trades if t.result == "loss" and t.pnl_dollars]) if losses else 0

    contract_counts = [t.num_contracts for t in trades]

    print(f"Total Trades:     {total}")
    print(f"Wins:             {wins} ({wr:.1f}%)")
    print(f"Losses:           {losses}")
    print()
    print(f"--- Points (MES) ---")
    print(f"Total P&L:        {total_pnl_pts:+.2f} pts")
    print(f"Avg Trade:        {avg_pnl_pts:+.2f} pts")
    print(f"Avg Win:          {avg_win_pts:+.2f} pts")
    print(f"Avg Loss:         {avg_loss_pts:+.2f} pts")
    print()
    print(f"--- Dollars (dynamic sizing: {min(contract_counts)}-{max(contract_counts)}x, "
          f"avg {np.mean(contract_counts):.1f}x, ${RISK_BUDGET:.0f} risk/trade) ---")
    print(f"Total P&L:        ${total_pnl_dollars:+,.2f}")
    print(f"Avg Trade:        ${avg_pnl_dollars:+,.2f}")
    print(f"Avg Win:          ${avg_win_dollars:+,.2f}")
    print(f"Avg Loss:         ${avg_loss_dollars:+,.2f}")

    # Per-model breakdown
    print(f"\n--- Per-Model Breakdown ---")
    for m in [1, 2, 3]:
        m_trades = [t for t in trades if t.model == m]
        if m_trades:
            m_wins = sum(1 for t in m_trades if t.result == "win")
            m_pnl = sum(t.pnl_dollars for t in m_trades if t.pnl_dollars)
            print(f"  Model {m}: {len(m_trades)} trades | "
                  f"Win Rate: {m_wins/len(m_trades)*100:.1f}% | "
                  f"P&L: ${m_pnl:+,.2f}")

    # Direction breakdown
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    print(f"\nLong Trades:      {len(longs)} (W: {sum(1 for t in longs if t.result == 'win')})")
    print(f"Short Trades:     {len(shorts)} (W: {sum(1 for t in shorts if t.result == 'win')})")

    # Daily breakdown
    print(f"\n--- Daily Breakdown ---")
    daily = {}
    for t in trades:
        day = t.entry_time.strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = {"trades": 0, "wins": 0, "pnl": 0.0}
        daily[day]["trades"] += 1
        if t.result == "win":
            daily[day]["wins"] += 1
        daily[day]["pnl"] += t.pnl_dollars or 0

    for day, stats in sorted(daily.items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
        print(f"  {day}: {stats['trades']:2d} trades | WR: {wr:5.1f}% | P&L: ${stats['pnl']:+,.2f}")

    # Drawdown
    cumulative = []
    running = 0.0
    for t in trades:
        running += t.pnl_dollars or 0
        cumulative.append(running)

    if cumulative:
        peak = cumulative[0]
        max_dd = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        print(f"\nMax Drawdown:     ${max_dd:,.2f}")
        print(f"Final Cum. P&L:   ${cumulative[-1]:+,.2f}")

    # Streaks
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for t in trades:
        if t.result == "win":
            current_streak = current_streak + 1 if current_streak > 0 else 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(current_streak))

    print(f"Max Win Streak:   {max_win_streak}")
    print(f"Max Loss Streak:  {max_loss_streak}")
    print(f"\n{'='*60}\n")


# ── Equity Curve ──

def plot_equity_curve(trades: list[Trade], save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trades:
        return

    dates = [trades[0].entry_time - timedelta(hours=1)]
    cum_dollars = [0.0]
    cum_pts = [0.0]
    running_dollars = 0.0
    running_pts = 0.0

    for t in trades:
        running_dollars += t.pnl_dollars or 0
        running_pts += t.pnl or 0
        dates.append(t.entry_time)
        cum_dollars.append(running_dollars)
        cum_pts.append(running_pts)

    total = len(trades)
    wins = sum(1 for t in trades if t.result == "win")
    wr = wins / total * 100 if total else 0
    date_range = (f"{trades[0].entry_time.strftime('%b %d')} - "
                  f"{trades[-1].entry_time.strftime('%b %d, %Y')}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14),
                                         height_ratios=[3, 2, 1],
                                         sharex=True, layout="constrained")

    # Panel 1: Dollar P&L
    ax1.plot(dates, cum_dollars, color="#2196F3", linewidth=2,
             label=f"Cumulative P&L  (${cum_dollars[-1]:+,.2f})")
    ax1.fill_between(dates, 0, cum_dollars, alpha=0.12, color="#2196F3",
                     where=[v >= 0 for v in cum_dollars])
    ax1.fill_between(dates, 0, cum_dollars, alpha=0.12, color="#F44336",
                     where=[v < 0 for v in cum_dollars])
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Win/loss markers
    win_dates = [t.entry_time for t in trades if t.result == "win"]
    loss_dates = [t.entry_time for t in trades if t.result == "loss"]
    r = 0.0
    win_cum, loss_cum = [], []
    for t in trades:
        r += t.pnl_dollars or 0
        if t.result == "win":
            win_cum.append(r)
        else:
            loss_cum.append(r)
    ax1.scatter(win_dates, win_cum, color="#4CAF50", s=15, zorder=5, alpha=0.6)
    ax1.scatter(loss_dates, loss_cum, color="#F44336", s=30, zorder=5,
                marker="x", linewidths=1.5)

    ax1.set_title(f"MES Futures Gap Fill Strategy  |  {date_range}  |  "
                  f"{total} trades, {wr:.1f}% WR  |  ${RISK_BUDGET:.0f} risk/trade",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:+,.0f}'))

    # Panel 2: Points P&L
    ax2.plot(dates, cum_pts, color="#009688", linewidth=1.5,
             label=f"Cumulative P&L  ({cum_pts[-1]:+.1f} pts)")
    ax2.fill_between(dates, 0, cum_pts, alpha=0.12, color="#009688",
                     where=[v >= 0 for v in cum_pts])
    ax2.fill_between(dates, 0, cum_pts, alpha=0.12, color="#F44336",
                     where=[v < 0 for v in cum_pts])
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Cumulative P&L (MES pts)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdown
    equity = [cum_dollars[0]]
    peak = cum_dollars[0]
    dd_pct = [0.0]
    for v in cum_dollars[1:]:
        if v > peak:
            peak = v
        dd = ((v - peak) / max(peak, 1)) * 100 if peak > 0 else 0
        dd_pct.append(dd)
    ax3.fill_between(dates, 0, dd_pct, color="#F44336", alpha=0.3)
    ax3.plot(dates, dd_pct, color="#F44336", linewidth=1)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "mes_equity_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MES equity curve saved to {save_path}")


# ── 6-Way Comparison Chart ──

ADAPTIVE_CONFIG = {
    (3, 6):   (0.70, 1.5),
    (6, 9):   (0.75, 1.5),
    (9, 13):  (0.70, 1.5),
    (13, 17): (0.60, 1.5),
    (17, 25): (0.60, 1.5),
}
UNIFORM_SL, UNIFORM_TP = 0.85, 1.25


def _variant_stats(trades):
    """Compute summary stats for a list of trades."""
    if not trades:
        return {}
    total = len(trades)
    wins = sum(1 for t in trades if t.result == "win")
    wr = wins / total * 100
    total_pnl = sum(t.pnl_dollars or 0 for t in trades)
    avg_pnl = total_pnl / total

    # Max drawdown
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        running += t.pnl_dollars or 0
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_wins = sum(t.pnl_dollars for t in trades if (t.pnl_dollars or 0) > 0)
    gross_losses = abs(sum(t.pnl_dollars for t in trades if (t.pnl_dollars or 0) < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Avg win / avg loss
    avg_win = np.mean([t.pnl_dollars for t in trades if t.result == "win" and t.pnl_dollars]) if wins else 0
    avg_loss = np.mean([t.pnl_dollars for t in trades if t.result == "loss" and t.pnl_dollars]) if (total - wins) else 0

    return dict(
        total=total, wins=wins, wr=wr, total_pnl=total_pnl, avg_pnl=avg_pnl,
        max_dd=max_dd, pf=pf, avg_win=avg_win, avg_loss=avg_loss,
    )


def plot_six_way_comparison(results: dict, save_path: str = None):
    """Plot 6-variant comparison: 2 rows (OLD vs NEW) x 3 cols (baseline/uniform/adaptive).

    results: dict of {label: trades_list}
    Expected 6 keys in order: old_baseline, old_uniform, old_adaptive,
                               new_baseline, new_uniform, new_adaptive
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    labels = list(results.keys())
    all_stats = {k: _variant_stats(v) for k, v in results.items()}

    # Build equity curves
    curves = {}
    for label, trades in results.items():
        dates = []
        cum = [0.0]
        running = 0.0
        if trades:
            dates.append(trades[0].entry_time - timedelta(hours=1))
            for t in trades:
                running += t.pnl_dollars or 0
                dates.append(t.entry_time)
                cum.append(running)
        curves[label] = (dates, cum)

    # Build drawdown curves
    dd_curves = {}
    for label, trades in results.items():
        dates = []
        dd_vals = [0.0]
        running = 0.0
        peak = 0.0
        if trades:
            dates.append(trades[0].entry_time - timedelta(hours=1))
            for t in trades:
                running += t.pnl_dollars or 0
                if running > peak:
                    peak = running
                dd = peak - running
                dates.append(t.entry_time)
                dd_vals.append(-dd)
        dd_curves[label] = (dates, dd_vals)

    # Layout: 4 rows x 3 cols
    # Row 0: OLD equity curves (baseline, uniform, adaptive)
    # Row 1: OLD drawdowns
    # Row 2: NEW equity curves
    # Row 3: NEW drawdowns
    fig = plt.figure(figsize=(22, 20), facecolor="#1a1a2e")
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.92, bottom=0.04, left=0.06, right=0.97)

    old_labels = labels[:3]
    new_labels = labels[3:]
    colors = ["#4FC3F7", "#FFB74D", "#81C784"]  # baseline, uniform, adaptive
    # Derive row titles from the first word of each group's keys
    prefixes = list(dict.fromkeys(k.split("_")[0] for k in labels))
    row_titles = [p.upper() + " fill" for p in prefixes[:2]]

    for group_idx, group_labels in enumerate([old_labels, new_labels]):
        eq_row = group_idx * 2
        dd_row = group_idx * 2 + 1

        for col, label in enumerate(group_labels):
            stats = all_stats[label]
            dates_eq, cum_eq = curves[label]
            dates_dd, dd_vals = dd_curves[label]
            color = colors[col]

            # Equity curve
            ax_eq = fig.add_subplot(gs[eq_row, col])
            ax_eq.set_facecolor("#0d1117")
            if dates_eq:
                ax_eq.plot(dates_eq, cum_eq, color=color, linewidth=1.5)
                ax_eq.fill_between(dates_eq, 0, cum_eq, alpha=0.15, color=color,
                                   where=[v >= 0 for v in cum_eq])
                ax_eq.fill_between(dates_eq, 0, cum_eq, alpha=0.15, color="#F44336",
                                   where=[v < 0 for v in cum_eq])
            ax_eq.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:+,.0f}'))
            ax_eq.tick_params(colors="white", labelsize=8)
            ax_eq.set_ylabel("Cumulative P&L", fontsize=9, color="white")
            for spine in ax_eq.spines.values():
                spine.set_color("#333")
            ax_eq.grid(True, alpha=0.15)

            # Title with stats
            short_name = label.split("_", 1)[1] if "_" in label else label
            title = f"{row_titles[group_idx]} — {short_name.upper()}"
            stat_line = ""
            if stats:
                stat_line = (f"{stats['total']} trades | {stats['wr']:.1f}% WR | "
                             f"${stats['total_pnl']:+,.0f} | PF {stats['pf']:.2f}")
            ax_eq.set_title(f"{title}\n{stat_line}", fontsize=10, color="white",
                            fontweight="bold", pad=8)

            # Drawdown
            ax_dd = fig.add_subplot(gs[dd_row, col])
            ax_dd.set_facecolor("#0d1117")
            if dates_dd:
                ax_dd.fill_between(dates_dd, 0, dd_vals, color="#F44336", alpha=0.35)
                ax_dd.plot(dates_dd, dd_vals, color="#F44336", linewidth=0.8)
            ax_dd.tick_params(colors="white", labelsize=8)
            ax_dd.set_ylabel("Drawdown", fontsize=9, color="white")
            ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            for spine in ax_dd.spines.values():
                spine.set_color("#333")
            ax_dd.grid(True, alpha=0.15)

            dd_label = f"Max DD: ${stats['max_dd']:,.0f}" if stats else ""
            extra = ""
            if stats:
                extra = f"  |  Avg W: ${stats['avg_win']:+,.0f}  Avg L: ${stats['avg_loss']:+,.0f}"
            ax_dd.set_title(f"{dd_label}{extra}", fontsize=9, color="#F44336", pad=4)

    fig.suptitle("MES Gap Fill — 6-Way Comparison: WICK Fill vs BODY Fill\n"
                 "$600 risk/trade | No re-entry | Skip lunch 12:30-14:00",
                 fontsize=14, fontweight="bold", color="white", y=0.97)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "mes_6way_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n6-way comparison chart saved to {save_path}")


# ── Main ──

def main():
    base_dir = os.path.dirname(__file__)
    use_databento = "--databento" in sys.argv or "--db" in sys.argv

    if use_databento:
        db_path = os.path.join(base_dir, "mes_data", "mes_1m.csv")
        if not os.path.exists(db_path):
            print(f"Error: Databento CSV not found at {db_path}")
            print("Extract and decompress the .csv.zst file first.")
            sys.exit(1)

        start_date = "2025-02-15"
        end_date = "2026-02-15"
        for i, arg in enumerate(sys.argv):
            if arg == "--start" and i + 1 < len(sys.argv):
                start_date = sys.argv[i + 1]
            if arg == "--end" and i + 1 < len(sys.argv):
                end_date = sys.argv[i + 1]
            if arg == "--full":
                start_date = None
                end_date = None

        label = f"{start_date or '2022-02-15'} to {end_date or '2026-02-15'}"
        print(f"Loading Databento MES data ({label})...")
        candles_1m = load_databento_data(db_path, start_date=start_date,
                                         end_date=end_date)
    else:
        csv1 = os.path.join(base_dir, "CME_MINI_MES1!, 1.csv")
        csv2 = os.path.join(base_dir, "CME_MINI_MES1!, 1 (1).csv")
        print("Loading MES 1m data (TradingView)...")
        candles_1m = load_mes_data(csv1, csv2)

    print(f"  Total 1m candles: {len(candles_1m):,}")
    if candles_1m:
        print(f"  Date range: {candles_1m[0].timestamp} to {candles_1m[-1].timestamp}")

    print("Building 15m RTH candles...")
    candles_15m_all = build_15m_candles(candles_1m)
    candles_15m_rth = filter_rth(candles_15m_all)
    print(f"  15m candles (all): {len(candles_15m_all):,}")
    print(f"  15m candles (RTH): {len(candles_15m_rth):,}")

    print(f"\n{'='*60}")
    print(f"  MES Corrected Backtest")
    print(f"  Overnight 18:00-9:30 ET | SL 3pts past wick | Min 8pt SL")
    print(f"  Body-fill | 3-day gap age | Single position | Skip lunch")
    print(f"{'='*60}")

    variants = {
        "corrected_3day": dict(sl_mult=1.0, tp_r=1.0, adaptive_config=None,
                               body_fill_first=True, max_gap_age_days=3.0,
                               label="Corrected Baseline (3-day, SL+3 buffer)"),
    }

    results = {}
    for name, kwargs in variants.items():
        _gap_cache.clear()
        print(f"\n{'─'*60}")
        print(f"  Running: {kwargs['label']}")
        print(f"{'─'*60}")
        trades = run_mes_backtest(candles_1m, candles_15m_rth, skip_lunch=True, **kwargs)
        print_report(trades)
        results[name] = trades

    # ── Summary table ──
    print(f"\n{'='*100}")
    print("GAP AGE COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"{'Variant':<40} {'Trades':>6} {'WR%':>7} {'Total P&L':>12} {'Max DD':>10} {'PF':>6} {'Avg W':>9} {'Avg L':>9}")
    print(f"{'-'*100}")
    for name, trades in results.items():
        s = _variant_stats(trades)
        if s:
            print(f"{name:<40} {s['total']:>6} {s['wr']:>6.1f}% ${s['total_pnl']:>+10,.0f} "
                  f"${s['max_dd']:>8,.0f} {s['pf']:>5.2f} ${s['avg_win']:>+7,.0f} ${s['avg_loss']:>+7,.0f}")
    print(f"{'='*100}")

    # ── Plot ──
    plot_six_way_comparison(results)


if __name__ == "__main__":
    main()
