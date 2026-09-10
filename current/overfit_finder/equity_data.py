"""
Equity Data Pipeline
====================
Loads equity 1m bar data (from Polygon-sourced CSVs) into Candle objects
compatible with the strategy DSL evaluator.

Equity CSVs have columns: timestamp, open, high, low, close, volume
(produced by infra/fetch_equity_data.py)

Also provides `get_prev_day_high_low()` — the equity equivalent of
`mb.get_overnight_high_low()`. For equities, "overnight levels" are
the previous RTH day's high and low.
"""
from __future__ import annotations

import bisect
from datetime import time
from pathlib import Path

import pandas as pd

# Reuse Candle dataclass from mes_backtest
import sys

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "archived" / "fvg_gap"))

import mes_backtest as mb

RTH_START = time(9, 30)
RTH_END = time(16, 0)
ET = pd.Timestamp.now(tz="US/Eastern").tzinfo


def load_equity_data(csv_path: str) -> list[mb.Candle]:
    """Load equity 1m CSV into Candle objects.

    Expected CSV columns: timestamp, open, high, low, close, volume
    Timestamps should be US/Eastern timezone-aware ISO strings,
    or UTC timestamps that will be converted.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Equity data not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle different timestamp column names
    ts_col = None
    for col in ["timestamp", "ts", "ts_event", "datetime"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found in {csv_path}. "
            f"Expected one of: timestamp, ts, ts_event, datetime. "
            f"Got: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df[ts_col])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    # Rename columns to standard names if needed
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vol": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Build Candle objects
    candles = []
    for _, row in df.iterrows():
        candles.append(mb.Candle(
            timestamp=row["timestamp"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        ))

    print(f"  Loaded {len(candles):,} equity 1m candles from {path.name}")
    return candles


def get_prev_day_high_low(
    candles_1m: list[mb.Candle],
    day: pd.Timestamp,
    ts_index: list[pd.Timestamp],
) -> tuple[float | None, float | None]:
    """Get previous trading day's RTH high and low.

    For equities, this replaces mb.get_overnight_high_low().
    The "overnight reference levels" are simply yesterday's high/low.

    Returns (prev_high, prev_low) or (None, None) if no previous day data.
    """
    # Find the start of the current day
    day_start = day.replace(hour=RTH_START.hour, minute=RTH_START.minute, second=0)

    # Find previous trading day: go back and find RTH candles before today
    # Look back up to 5 calendar days to handle weekends/holidays
    prev_day_end = day.normalize()  # midnight of current day
    prev_day_start = prev_day_end - pd.Timedelta(days=5)

    # Use bisect for efficient slicing
    lo = bisect.bisect_left(ts_index, prev_day_start)
    hi = bisect.bisect_left(ts_index, prev_day_end)

    if lo >= hi:
        return None, None

    # Filter to RTH candles only, find the most recent trading day
    rth_candles = []
    last_date = None
    for c in reversed(candles_1m[lo:hi]):
        t = c.timestamp.time()
        if t < RTH_START or t >= RTH_END:
            continue
        c_date = c.timestamp.date()
        if last_date is None:
            last_date = c_date
        if c_date != last_date:
            break  # hit a different (older) day, stop
        rth_candles.append(c)

    if not rth_candles:
        return None, None

    prev_high = max(c.high for c in rth_candles)
    prev_low = min(c.low for c in rth_candles)

    return prev_high, prev_low
