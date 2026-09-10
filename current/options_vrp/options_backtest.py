"""
VRP Credit Spread Backtester
=============================
Simulates SPY credit spread trades using 5-DTE chain data.

Chain data (parquet): columns ts, o, h, l, c, vol, vwap, trades, strike, cp, option_ticker
  - c = close price (used as mid-price proxy)
  - cp = 'C' (call) or 'P' (put)
Meta data (JSON): trade_date, expiry_date, dte, atm, n_contracts, n_strikes, strikes

External interface:
    evaluate_day(genes: VRPGenes, day: str, data_dir: str) -> FitnessResult
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from options_genes import VRPGenes

# ─── Constants ────────────────────────────────────────────────────────────────

COMMISSION_PER_LEG = 0.65   # $ per contract per leg
SLIPPAGE_PER_LEG = 0.02     # $ per contract per leg (proxy for half-spread)
CONTRACT_MULTIPLIER = 100   # 100 shares per contract

_N_LEGS = {"bull_put": 2, "bear_call": 2, "iron_condor": 4}


def _friction(spread_type: str) -> float:
    """Total round-trip friction (entry + exit) in dollars."""
    n = _N_LEGS[spread_type]
    one_way = (SLIPPAGE_PER_LEG * n) * CONTRACT_MULTIPLIER + COMMISSION_PER_LEG * n
    return one_way * 2  # entry + exit


# ─── Result Type ──────────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    total_pnl: float
    win_rate: float
    num_trades: int
    sharpe: float


_ZERO = FitnessResult(total_pnl=0.0, win_rate=0.0, num_trades=0, sharpe=0.0)


# ─── Data Loading ─────────────────────────────────────────────────────────────

_META_CACHE: dict[tuple, dict] = {}
_CHAIN_CACHE: dict[tuple, pd.DataFrame] = {}


def _load_meta(data_dir: Path, day: str) -> dict:
    key = (str(data_dir), day)
    if key in _META_CACHE:
        return _META_CACHE[key]
    path = data_dir / f"chain_{day}_meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        result = json.load(f)
    _META_CACHE[key] = result
    return result


def _load_chain(data_dir: Path, day: str) -> pd.DataFrame:
    key = (str(data_dir), day)
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    path = data_dir / f"chain_{day}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" in df.columns and not hasattr(df["ts"].dtype, "tz"):
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(
            "US/Eastern", ambiguous="NaT", nonexistent="NaT"
        )
    if "_minute" not in df.columns and "ts" in df.columns:
        df = df.copy()
        df["_minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    _CHAIN_CACHE[key] = df
    return df


# ─── Spread Construction ──────────────────────────────────────────────────────

def _get_atm_idx(strikes: list, atm_price: float) -> int:
    """Return index of strike closest to ATM price."""
    return min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_price))


def _leg_bars(chain: pd.DataFrame, strike: float, cp: str) -> pd.DataFrame:
    """Extract bars for a specific strike/cp from the chain."""
    mask = (chain["strike"].values == strike) & (chain["cp"].values == cp)
    return chain.loc[mask]


def _first_bar_price(bars: pd.DataFrame) -> tuple[float, int] | None:
    """Return (close_price, minute) for the first bar in the set."""
    if bars.empty:
        return None
    row = bars.iloc[0]
    price = float(row["c"])
    minute = int(row["_minute"])
    return price, minute


def _build_spread(
    chain: pd.DataFrame,
    meta: dict,
    genes: VRPGenes,
) -> dict | None:
    """
    Build a credit spread position at the open of the day.

    Returns dict with:
        opening_credit: net credit received per contract (in $, pre-multiplier)
        short_bars, long_bars: DataFrames for put/call legs
        short_bars_2, long_bars_2: second pair for iron condor (or None)
        entry_minute: minute of entry fill
        spread_type: from genes
        n_legs: 2 or 4
    """
    strikes = meta.get("strikes", [])
    atm_price = meta.get("atm", 0)
    if not strikes or atm_price <= 0:
        return None

    atm_idx = _get_atm_idx(strikes, atm_price)

    def _build_one(cp: str, short_idx: int, long_idx: int) -> tuple | None:
        if short_idx < 0 or short_idx >= len(strikes):
            return None
        if long_idx < 0 or long_idx >= len(strikes):
            return None
        s_strike = strikes[short_idx]
        l_strike = strikes[long_idx]
        s_bars = _leg_bars(chain, s_strike, cp)
        l_bars = _leg_bars(chain, l_strike, cp)
        s_first = _first_bar_price(s_bars)
        l_first = _first_bar_price(l_bars)
        if s_first is None or l_first is None:
            return None
        s_price, s_min = s_first
        l_price, l_min = l_first
        credit = s_price - l_price  # net credit (short premium - long premium)
        entry_min = max(s_min, l_min)
        return credit, s_bars, l_bars, entry_min

    stype = genes.spread_type
    ofs = genes.short_strike_offset
    ww = genes.wing_width

    if stype == "bull_put":
        # Short put OTM below ATM, long put further below
        short_idx = atm_idx - ofs
        long_idx = atm_idx - ofs - ww
        result = _build_one("P", short_idx, long_idx)
        if result is None:
            return None
        credit, s_bars, l_bars, entry_min = result
        if credit <= 0:
            return None
        return {
            "opening_credit": credit,
            "short_bars": s_bars, "long_bars": l_bars,
            "short_bars_2": None, "long_bars_2": None,
            "entry_minute": entry_min,
            "spread_type": stype, "n_legs": 2,
        }

    elif stype == "bear_call":
        # Short call OTM above ATM, long call further above
        short_idx = atm_idx + ofs
        long_idx = atm_idx + ofs + ww
        result = _build_one("C", short_idx, long_idx)
        if result is None:
            return None
        credit, s_bars, l_bars, entry_min = result
        if credit <= 0:
            return None
        return {
            "opening_credit": credit,
            "short_bars": s_bars, "long_bars": l_bars,
            "short_bars_2": None, "long_bars_2": None,
            "entry_minute": entry_min,
            "spread_type": stype, "n_legs": 2,
        }

    elif stype == "iron_condor":
        # Bull put + bear call simultaneously
        put_short_idx = atm_idx - ofs
        put_long_idx = atm_idx - ofs - ww
        call_short_idx = atm_idx + ofs
        call_long_idx = atm_idx + ofs + ww
        put_result = _build_one("P", put_short_idx, put_long_idx)
        call_result = _build_one("C", call_short_idx, call_long_idx)
        if put_result is None or call_result is None:
            return None
        p_credit, p_s_bars, p_l_bars, p_min = put_result
        c_credit, c_s_bars, c_l_bars, c_min = call_result
        total_credit = p_credit + c_credit
        if total_credit <= 0:
            return None
        entry_min = max(p_min, c_min)
        return {
            "opening_credit": total_credit,
            "short_bars": p_s_bars, "long_bars": p_l_bars,
            "short_bars_2": c_s_bars, "long_bars_2": c_l_bars,
            "entry_minute": entry_min,
            "spread_type": stype, "n_legs": 4,
        }

    return None


# ─── P&L Simulation ──────────────────────────────────────────────────────────

def _leg_arrays(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted (minutes, closes) numpy arrays for fast simulation."""
    mins = bars["_minute"].values
    closes = bars["c"].values.astype(float)
    order = np.argsort(mins)
    return mins[order], closes[order]


def _spread_value_at_minute(
    s_bars: pd.DataFrame,
    l_bars: pd.DataFrame,
    minute: int,
) -> float | None:
    """
    Compute current spread value (short_close - long_close) at a given minute.
    Uses the most recent bar at or before the given minute for each leg.
    """
    s_mask = s_bars["_minute"].values <= minute
    l_mask = l_bars["_minute"].values <= minute
    if not s_mask.any() or not l_mask.any():
        return None
    s_price = float(s_bars.loc[s_mask].iloc[-1]["c"])
    l_price = float(l_bars.loc[l_mask].iloc[-1]["c"])
    return s_price - l_price


def _simulate(
    spread_info: dict,
    genes: VRPGenes,
) -> tuple[float, str]:
    """
    Walk forward bars to simulate TP/SL/hold-to-end.

    Returns (exit_spread_value, exit_reason).
    exit_spread_value is the spread value at close (cost to buy back).
    pnl_gross = (opening_credit - exit_spread_value) * 100
    """
    opening_credit = spread_info["opening_credit"]
    entry_minute = spread_info["entry_minute"]
    s_bars = spread_info["short_bars"]
    l_bars = spread_info["long_bars"]
    s_bars_2 = spread_info["short_bars_2"]
    l_bars_2 = spread_info["long_bars_2"]

    tp_threshold = genes.tp_pct * opening_credit      # value <= this triggers TP
    sl_threshold = genes.sl_multiple * opening_credit  # value >= this triggers SL

    # Extract numpy arrays for vectorized simulation
    s_mins_arr, s_cls_arr = _leg_arrays(s_bars)
    l_mins_arr, l_cls_arr = _leg_arrays(l_bars)

    # Build post-entry minute axis
    post_s = s_mins_arr[s_mins_arr > entry_minute]
    post_l = l_mins_arr[l_mins_arr > entry_minute]
    post_entry_mins = np.union1d(post_s, post_l)

    if s_bars_2 is not None and l_bars_2 is not None:
        s2_mins_arr, s2_cls_arr = _leg_arrays(s_bars_2)
        l2_mins_arr, l2_cls_arr = _leg_arrays(l_bars_2)
        post_entry_mins = np.union1d(
            post_entry_mins,
            np.union1d(
                s2_mins_arr[s2_mins_arr > entry_minute],
                l2_mins_arr[l2_mins_arr > entry_minute],
            ),
        )
    else:
        s2_mins_arr = s2_cls_arr = l2_mins_arr = l2_cls_arr = None

    def _price_at(mins_arr: np.ndarray, cls_arr: np.ndarray, minute: int) -> float | None:
        idx = np.searchsorted(mins_arr, minute, side="right") - 1
        if idx < 0:
            return None
        return float(cls_arr[idx])

    if len(post_entry_mins) == 0:
        # No post-entry bars
        sp = _price_at(s_mins_arr, s_cls_arr, entry_minute + 9999)
        lp = _price_at(l_mins_arr, l_cls_arr, entry_minute + 9999)
        if sp is None or lp is None:
            val = opening_credit
        else:
            val = sp - lp
            if s2_mins_arr is not None:
                sp2 = _price_at(s2_mins_arr, s2_cls_arr, entry_minute + 9999)
                lp2 = _price_at(l2_mins_arr, l2_cls_arr, entry_minute + 9999)
                if sp2 is not None and lp2 is not None:
                    val += sp2 - lp2
        return val, "no_data"

    last_value = None
    for minute in post_entry_mins:
        sp = _price_at(s_mins_arr, s_cls_arr, minute)
        lp = _price_at(l_mins_arr, l_cls_arr, minute)
        if sp is None or lp is None:
            continue
        current_value = sp - lp
        if s2_mins_arr is not None and l2_mins_arr is not None:
            sp2 = _price_at(s2_mins_arr, s2_cls_arr, minute)
            lp2 = _price_at(l2_mins_arr, l2_cls_arr, minute)
            if sp2 is not None and lp2 is not None:
                current_value += sp2 - lp2
        last_value = current_value

        if current_value <= tp_threshold:
            return current_value, "tp"
        if current_value >= sl_threshold:
            return current_value, "sl"

    # Hold to end (EOD or last bar = proxy for expiry)
    if last_value is None:
        last_value = opening_credit
    return last_value, "hold"


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def evaluate_day(genes: VRPGenes, day: str, data_dir: str) -> FitnessResult:
    """
    Evaluate a VRPGenes strategy on a single trading day.

    Args:
        genes: VRPGenes dataclass
        day:   YYYY-MM-DD trade date
        data_dir: path to options_5dte directory

    Returns:
        FitnessResult with total_pnl, win_rate, num_trades, sharpe.
        Returns zero FitnessResult if the day has no data or the spread
        can't be constructed (insufficient strikes, zero credit, etc.).
    """
    base = Path(data_dir)
    meta = _load_meta(base, day)
    if not meta:
        return _ZERO

    chain = _load_chain(base, day)
    if chain.empty:
        return _ZERO

    spread_info = _build_spread(chain, meta, genes)
    if spread_info is None:
        return _ZERO

    opening_credit = spread_info["opening_credit"]
    exit_value, exit_reason = _simulate(spread_info, genes)

    # Gross P&L: collected credit minus cost to close (per contract, pre-multiplier)
    pnl_gross = (opening_credit - exit_value) * CONTRACT_MULTIPLIER
    # Net P&L after friction
    total_friction = _friction(genes.spread_type)
    pnl_net = pnl_gross - total_friction

    win = 1 if pnl_net > 0 else 0

    return FitnessResult(
        total_pnl=pnl_net,
        win_rate=float(win),
        num_trades=1,
        sharpe=0.0,  # undefined for single-day single-trade
    )
