"""
GEX (Gamma Exposure) + Market Context Features for OverfitAlpha
================================================================
Computes volume-based GEX, put/call walls, IV regime features,
daily market stats (VWAP, realized vol), and earnings proximity.

Data sources:
  - SPY 0DTE option chains (491 days, 1-min bars) → GEX, walls, IV
  - SPY 1m underlying bars (via fetch_extra.py daily_stats) → VWAP, RV
  - Earnings calendar (via fetch_extra.py) → earnings_nearby flag

GEX regime classification:
  - Positive GEX → dealers long gamma → buy dips, sell rips → mean-reversion
  - Negative GEX → dealers short gamma → hedge in direction → momentum/trending
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add zero_dte to path for bsm imports (lazy — only needed by GexCache)
_SCRIPT_DIR = Path(__file__).parent
_ZERO_DTE_DIR = _SCRIPT_DIR.parent / "zero_dte"
if str(_ZERO_DTE_DIR) not in sys.path:
    sys.path.insert(0, str(_ZERO_DTE_DIR))

# bsm imported lazily inside GexCache.precompute() so DailyCache works
# even when scipy is unavailable
_bsm = None

def _get_bsm():
    global _bsm
    if _bsm is None:
        import bsm as _bsm_mod
        _bsm = _bsm_mod
    return _bsm

# Data directory for chain parquets
_CHAIN_DIR = Path(os.environ.get(
    "ZERO_DTE_DATA_DIR",
    str(_ZERO_DTE_DIR / "data"),
)) / "options"

# Risk-free rate (approximate, not sensitive for gamma calc)
RISK_FREE = 0.05

# SPY → S&P 500 index conversion factor (SPY ≈ 1/10 of S&P 500)
SPY_TO_INDEX = 10.0

# Snapshot times to pre-compute (ET)
DEFAULT_SNAPSHOT_TIMES = [
    time(9, 45),
    time(10, 0),
    time(10, 15),
    time(10, 30),
    time(10, 45),
    time(11, 0),
    time(11, 30),
]

# GEX regime z-score thresholds
GEX_POS_THRESHOLD = 0.5
GEX_NEG_THRESHOLD = -0.5


@dataclass
class DailyContext:
    """Daily market stats computed from SPY 1m bars (via fetch_extra.py)."""
    prev_vwap: float            # previous day's VWAP (volume-weighted avg price)
    realized_vol_5m: float      # prev day's 5-min realized vol (annualized)
    realized_vol_30m: float     # prev day's 30-min realized vol (annualized)
    prev_range_pct: float       # prev day's range as % of price
    earnings_nearby: list[str]  # tickers with earnings within 2 days (e.g. ["AAPL", "MSFT"])


@dataclass
class GexContext:
    """GEX and options flow context for a single timestamp."""
    gex_regime: str             # "positive" | "negative" | "neutral"
    total_volume_gex: float     # raw volume-weighted gamma value
    gex_zscore: float           # normalized vs rolling history
    call_wall_strike: float     # max call volume strike (in index pts)
    put_wall_strike: float      # max put volume strike (in index pts)
    dist_to_call_wall: float    # spot - call_wall (index pts)
    dist_to_put_wall: float     # spot - put_wall (index pts)
    atm_iv: float               # ATM implied volatility
    iv_skew: float              # put IV - call IV (positive = bearish skew)
    pc_volume_ratio: float      # total put volume / total call volume


def _load_chain(d: date) -> pd.DataFrame:
    """Load cached options chain parquet for a day."""
    path = _CHAIN_DIR / f"chain_{d.isoformat()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_chain_meta(d: date) -> dict:
    """Load chain metadata (ATM, strikes)."""
    path = _CHAIN_DIR / f"chain_{d.isoformat()}_meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def compute_gex_snapshot(
    chain: pd.DataFrame,
    meta: dict,
    snapshot_time: time,
) -> Optional[GexContext]:
    """Compute GEX features from chain data at a specific time.

    Args:
        chain: DataFrame with columns [ts, o, h, l, c, vol, strike, cp, ...]
        meta: Dict with 'atm' key for spot price
        snapshot_time: Time (ET) to compute snapshot at

    Returns:
        GexContext or None if insufficient data
    """
    if chain.empty or not meta:
        return None

    spot_spy = meta.get("atm")
    if spot_spy is None:
        return None

    # Filter to bars at or before snapshot_time
    snap_min = _time_to_minutes(snapshot_time)
    chain = chain.copy()
    chain["bar_min"] = chain["ts"].dt.hour * 60 + chain["ts"].dt.minute
    chain = chain[chain["bar_min"] <= snap_min]

    if chain.empty:
        return None

    # Get latest bar per (strike, cp) — cumulative volume up to snapshot
    # Group by strike+cp, aggregate: last close price, sum of volume
    agg = (
        chain.groupby(["strike", "cp"])
        .agg(
            price=("c", "last"),      # latest close as mid proxy
            cum_vol=("vol", "sum"),    # cumulative volume up to snapshot
        )
        .reset_index()
    )

    if len(agg) < 4:
        return None

    # Compute IV and gamma for each contract
    strikes = agg["strike"].values
    prices = agg["price"].values
    cp_str = agg["cp"].values
    volumes = agg["cum_vol"].values
    n_contracts = len(strikes)

    # cp: 1 for calls, -1 for puts
    cp_num = np.where(cp_str == "C", 1.0, -1.0)

    # Time to expiry: 0DTE, from snapshot to 4:00 PM ET
    close_min = 960.0  # 16:00
    remaining_min = max(close_min - snap_min, 1.0)
    T_scalar = remaining_min / (390.0 * 252.0)  # annualized

    # BSM functions expect arrays — broadcast scalars
    S_arr = np.full(n_contracts, spot_spy)
    T_arr = np.full(n_contracts, T_scalar)

    # Compute IV via Newton-Raphson (all args must be arrays for masking)
    bsm = _get_bsm()
    iv = bsm.implied_vol(prices, S_arr, strikes, T_arr, RISK_FREE, cp_num)

    # Replace NaN IVs with a default (high IV, so gamma stays reasonable)
    iv_valid = np.where(np.isfinite(iv), iv, 0.30)

    # Compute gamma per contract
    gamma = bsm.bs_gamma(S_arr, strikes, T_arr, RISK_FREE, iv_valid)

    # Volume GEX per contract: call_vol * gamma (positive) vs put_vol * gamma (negative)
    # Convention: calls contribute positive GEX, puts contribute negative
    contract_gex = np.where(
        cp_str == "C",
        volumes * gamma,   # call: positive contribution
        -volumes * gamma,  # put: negative contribution
    )

    # Total volume GEX (normalized by spot^2 * 100 for dollar gamma)
    total_gex = np.sum(contract_gex) * spot_spy * spot_spy * 100.0

    # --- Put/Call Walls ---
    calls = agg[agg["cp"] == "C"]
    puts = agg[agg["cp"] == "P"]

    if calls.empty or puts.empty:
        return None

    call_wall_idx = calls["cum_vol"].idxmax()
    put_wall_idx = puts["cum_vol"].idxmax()
    call_wall_spy = calls.loc[call_wall_idx, "strike"]
    put_wall_spy = puts.loc[put_wall_idx, "strike"]

    # Convert to index points
    call_wall = call_wall_spy * SPY_TO_INDEX
    put_wall = put_wall_spy * SPY_TO_INDEX
    spot_index = spot_spy * SPY_TO_INDEX

    # --- IV features ---
    # ATM: find the call and put closest to spot
    atm_call = calls.iloc[(calls["strike"] - spot_spy).abs().argsort().iloc[0]]
    atm_put = puts.iloc[(puts["strike"] - spot_spy).abs().argsort().iloc[0]]

    # IV for ATM contracts
    atm_call_strike = atm_call["strike"]
    atm_put_strike = atm_put["strike"]

    # Get IVs for these specific contracts
    call_mask = (agg["strike"] == atm_call_strike) & (agg["cp"] == "C")
    put_mask = (agg["strike"] == atm_put_strike) & (agg["cp"] == "P")

    call_iv_val = iv[call_mask.values]
    put_iv_val = iv[put_mask.values]

    atm_iv_val = 0.20  # default
    iv_skew_val = 0.0
    if len(call_iv_val) > 0 and np.isfinite(call_iv_val[0]):
        atm_iv_val = float(call_iv_val[0])
    if len(put_iv_val) > 0 and np.isfinite(put_iv_val[0]):
        if len(call_iv_val) > 0 and np.isfinite(call_iv_val[0]):
            atm_iv_val = (float(call_iv_val[0]) + float(put_iv_val[0])) / 2
        iv_skew_val = float(put_iv_val[0]) - (float(call_iv_val[0]) if len(call_iv_val) > 0 and np.isfinite(call_iv_val[0]) else float(put_iv_val[0]))

    # --- Put/Call volume ratio ---
    total_call_vol = calls["cum_vol"].sum()
    total_put_vol = puts["cum_vol"].sum()
    pc_ratio = float(total_put_vol / max(total_call_vol, 1))

    return GexContext(
        gex_regime="neutral",  # set after z-score normalization
        total_volume_gex=float(total_gex),
        gex_zscore=0.0,        # set after z-score normalization
        call_wall_strike=float(call_wall),
        put_wall_strike=float(put_wall),
        dist_to_call_wall=float(spot_index - call_wall),
        dist_to_put_wall=float(spot_index - put_wall),
        atm_iv=float(atm_iv_val),
        iv_skew=float(iv_skew_val),
        pc_volume_ratio=float(pc_ratio),
    )


class GexCache:
    """Pre-compute and cache GEX features for all available chain days."""

    def __init__(self, chain_dir: Path = None):
        self._chain_dir = chain_dir or _CHAIN_DIR
        self._cache: dict[tuple[date, time], GexContext] = {}
        self._gex_history: list[float] = []  # for z-score normalization

    def precompute(
        self,
        start_date: str = None,
        end_date: str = None,
        snapshot_times: list[time] = None,
        lookback_days: int = 20,
    ) -> None:
        """Compute GEX for all available days in date range.

        Args:
            start_date: ISO date string, or None for earliest available
            end_date: ISO date string, or None for latest available
            snapshot_times: list of ET times to compute at
            lookback_days: rolling window for z-score normalization
        """
        if snapshot_times is None:
            snapshot_times = DEFAULT_SNAPSHOT_TIMES

        # Find all available chain files
        chain_files = sorted(self._chain_dir.glob("chain_*_meta.json"))
        available_dates = []
        for f in chain_files:
            d_str = f.stem.replace("chain_", "").replace("_meta", "")
            try:
                available_dates.append(date.fromisoformat(d_str))
            except ValueError:
                continue

        if not available_dates:
            print("GexCache: No chain data found")
            return

        # Filter date range
        if start_date:
            sd = date.fromisoformat(start_date)
            available_dates = [d for d in available_dates if d >= sd]
        if end_date:
            ed = date.fromisoformat(end_date)
            available_dates = [d for d in available_dates if d <= ed]

        print(f"GexCache: Pre-computing GEX for {len(available_dates)} days "
              f"× {len(snapshot_times)} snapshots...")

        # First pass: compute raw GEX values
        raw_snapshots: dict[tuple[date, time], GexContext] = {}
        daily_gex: dict[date, float] = {}

        for i, d in enumerate(available_dates):
            chain = _load_chain(d)
            meta = _load_chain_meta(d)
            if chain.empty or not meta:
                continue

            for snap_time in snapshot_times:
                ctx = compute_gex_snapshot(chain, meta, snap_time)
                if ctx is not None:
                    raw_snapshots[(d, snap_time)] = ctx
                    # Use first snapshot's GEX as the day's representative
                    if d not in daily_gex:
                        daily_gex[d] = ctx.total_volume_gex

        # Second pass: normalize with rolling z-score and set regime
        sorted_dates = sorted(daily_gex.keys())
        gex_values = [daily_gex[d] for d in sorted_dates]

        for (d, snap_time), ctx in raw_snapshots.items():
            # Find this date's position in the sorted list
            date_idx = -1
            for idx, sd in enumerate(sorted_dates):
                if sd == d:
                    date_idx = idx
                    break

            if date_idx < 0:
                continue

            # Rolling window for z-score
            window_start = max(0, date_idx - lookback_days)
            window = gex_values[window_start:date_idx + 1]

            if len(window) >= 5:
                mean_gex = np.mean(window)
                std_gex = np.std(window)
                if std_gex > 1e-10:
                    zscore = (ctx.total_volume_gex - mean_gex) / std_gex
                else:
                    zscore = 0.0
            else:
                zscore = 0.0

            # Set regime based on z-score
            if zscore > GEX_POS_THRESHOLD:
                regime = "positive"
            elif zscore < GEX_NEG_THRESHOLD:
                regime = "negative"
            else:
                regime = "neutral"

            # Update context with z-score and regime
            ctx.gex_zscore = float(zscore)
            ctx.gex_regime = regime
            self._cache[(d, snap_time)] = ctx

        self._gex_history = gex_values
        print(f"GexCache: Cached {len(self._cache)} snapshots "
              f"({len(daily_gex)} unique days)")

    def get(self, trade_date: date, current_time: time) -> Optional[GexContext]:
        """Get nearest GEX snapshot for a given date/time.

        Returns the most recent snapshot at or before current_time.
        Returns None if no snapshot exists at or before current_time
        (never falls back to a future snapshot — prevents look-ahead bias).
        """
        current_min = _time_to_minutes(current_time)

        # Find the best matching snapshot (latest one at or before current_time)
        best_key = None
        best_min = -1

        for (d, snap_time) in self._cache:
            if d != trade_date:
                continue
            snap_min = _time_to_minutes(snap_time)
            if snap_min <= current_min and snap_min > best_min:
                best_min = snap_min
                best_key = (d, snap_time)

        if best_key is None:
            return None

        return self._cache[best_key]

    @property
    def stats(self) -> dict:
        """Summary statistics for cached GEX data."""
        if not self._cache:
            return {"n_snapshots": 0}

        dates = set(d for d, _ in self._cache)
        regimes = [ctx.gex_regime for ctx in self._cache.values()]
        return {
            "n_snapshots": len(self._cache),
            "n_days": len(dates),
            "pct_positive": sum(1 for r in regimes if r == "positive") / len(regimes) * 100,
            "pct_negative": sum(1 for r in regimes if r == "negative") / len(regimes) * 100,
            "pct_neutral": sum(1 for r in regimes if r == "neutral") / len(regimes) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DailyCache — VWAP, Realized Vol, Earnings
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_DIR = Path(os.environ.get(
    "ZERO_DTE_DATA_DIR",
    str(_ZERO_DTE_DIR / "data"),
))
_DAILY_STATS_DIR = _DATA_DIR / "daily_stats"
_EARNINGS_DIR = _DATA_DIR / "earnings"


class DailyCache:
    """Load and cache daily stats (VWAP, realized vol) and earnings calendar."""

    def __init__(self):
        self._cache: dict[date, DailyContext] = {}
        self._earnings: list[dict] = []

    def precompute(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> None:
        """Load daily stats and earnings for the date range."""
        # Load all earnings files
        self._earnings = []
        for p in sorted(_EARNINGS_DIR.glob("earnings_*.json")):
            with open(p) as f:
                self._earnings.extend(json.load(f))

        # Load daily stats
        stats_files = sorted(_DAILY_STATS_DIR.glob("daily_*.json"))
        loaded = 0
        for p in stats_files:
            d_str = p.stem.replace("daily_", "")
            try:
                d = date.fromisoformat(d_str)
            except ValueError:
                continue

            if start_date and d < date.fromisoformat(start_date):
                continue
            if end_date and d > date.fromisoformat(end_date):
                continue

            with open(p) as f:
                stats = json.load(f)

            # Find earnings near this date
            # Prefer report_date (announcement) over filing_date (SEC filing)
            nearby_earnings = []
            for e in self._earnings:
                try:
                    edate = date.fromisoformat(
                        e.get("report_date") or e.get("filing_date", ""))
                    if abs((edate - d).days) <= 2:
                        nearby_earnings.append(e["ticker"])
                except (ValueError, KeyError):
                    continue

            self._cache[d] = DailyContext(
                prev_vwap=stats.get("vwap", 0.0),
                realized_vol_5m=stats.get("realized_vol_5m", 0.0),
                realized_vol_30m=stats.get("realized_vol_30m", 0.0),
                prev_range_pct=stats.get("range_pct", 0.0),
                earnings_nearby=nearby_earnings,
            )
            loaded += 1

        print(f"DailyCache: Loaded {loaded} daily stats, "
              f"{len(self._earnings)} earnings events")

    def get(self, trade_date: date) -> Optional[DailyContext]:
        """Get daily context for a date.

        Returns the PREVIOUS day's stats (to avoid look-ahead bias).
        """
        # Find most recent date before trade_date
        prev_dates = [d for d in self._cache if d < trade_date]
        if not prev_dates:
            return None
        prev_d = max(prev_dates)
        # Only use if within 5 calendar days (skip holidays/weekends)
        if (trade_date - prev_d).days > 5:
            return None

        ctx = self._cache[prev_d]
        # Update earnings_nearby to check against trade_date, not prev_d
        # Prefer report_date (announcement) over filing_date (SEC filing)
        nearby = []
        for e in self._earnings:
            try:
                edate = date.fromisoformat(
                    e.get("report_date") or e.get("filing_date", ""))
                if abs((edate - trade_date).days) <= 2:
                    nearby.append(e["ticker"])
            except (ValueError, KeyError):
                continue

        return DailyContext(
            prev_vwap=ctx.prev_vwap,
            realized_vol_5m=ctx.realized_vol_5m,
            realized_vol_30m=ctx.realized_vol_30m,
            prev_range_pct=ctx.prev_range_pct,
            earnings_nearby=nearby,
        )

    @property
    def stats(self) -> dict:
        if not self._cache:
            return {"n_days": 0}
        rvols = [c.realized_vol_5m for c in self._cache.values() if c.realized_vol_5m > 0]
        earnings_days = sum(1 for c in self._cache.values() if c.earnings_nearby)
        return {
            "n_days": len(self._cache),
            "avg_rv_5m": np.mean(rvols) if rvols else 0,
            "n_earnings_days": earnings_days,
            "n_earnings_events": len(self._earnings),
        }
