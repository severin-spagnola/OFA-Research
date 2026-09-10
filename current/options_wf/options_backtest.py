"""
Options Gene Search Backtester
==============================
Simulates SPY options trades using actual 1-minute chain data from Polygon.
Entry signals come from the same gene catalog as MES (price_broke, gex_regime_is,
etc.) but exits are options-native: premium-based TP/SL, time stops.

Key differences from MES backtester:
  - P&L from actual option chain bars, not futures price × multiplier
  - Spread construction: long ATM, short ATM+1 (calls) or ATM-1 (puts)
  - Premium-based exits: TP/SL as % of max gain or entry debit
  - Realistic friction: $7.30 per round trip ($0.03 slippage/leg + $0.65 comm/leg)
  - Bar-by-bar exit scanning on 1m option prices

Data requirements:
  - SPY underlying 1m bars (parquet, one per day)
  - SPY options chain 1m bars (parquet, one per day) with columns:
    ts, o, h, l, c, vol, vwap, trades, strike, cp, option_ticker
  - Chain metadata JSON with: trade_date, expiry_date, dte, atm, strikes
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

RTH_START = time(9, 30)
RTH_END = time(16, 0)

# ─── Audit Config Flags ─────────────────────────────────────────────────────
# These flags control backtest realism assumptions. Defaults are the
# corrected (more realistic) settings. Set to False/0.0 to reproduce
# original pre-audit results for direct comparison.

# FILL_NEXT_BAR: When True, option entry fills use the first available option
# bar AFTER the signal minute (next-bar fill). When False, fills use the bar
# at or before the signal minute (same-bar fill — the original behavior).
# Why: The signal fires on the underlying bar close at time T. The option bar
# close at time T already reflects the underlying move that triggered the
# signal. A realistic fill requires routing the order and filling on the next
# available bar. For live trading, you'd observe bar T's close, decide to
# trade, and get filled at bar T+1's price at earliest.
FILL_NEXT_BAR = True

# FILL_NEXT_BAR_MAX_DELAY: Maximum minutes to look ahead for a fill bar.
# If no option bar exists within this window after the signal, the trade is
# skipped. Prevents filling from stale bars far in the future.
FILL_NEXT_BAR_MAX_DELAY = 5

# REALISTIC_STOP_FILLS: When True, stop/TSL exits fill at the bar's actual
# price extreme (low for longs, high for shorts) rather than the theoretical
# trigger level — unless the trigger level is worse than the bar extreme.
# When False, fills at the theoretical trigger level (original behavior).
# Why: When a stop triggers, the market order fills at whatever price is
# available, not the exact trigger level. The bar's extreme is the best
# proxy for the worst fill within that minute.
REALISTIC_STOP_FILLS = True

# Execution friction (per spread, per side)
SLIPPAGE_PER_LEG = 0.03      # $0.03 premium slippage
COMMISSION_PER_LEG = 0.65    # $0.65 per contract per leg
# Total per spread entry: (0.03 * 2) * 100 + 0.65 * 2 = $7.30
ENTRY_FRICTION = (SLIPPAGE_PER_LEG * 2) * 100 + COMMISSION_PER_LEG * 2
EXIT_FRICTION = COMMISSION_PER_LEG * 2  # $1.30 (no slippage at limit exits)


@dataclass
class CostModel:
    """Tunable cost model for stress testing.

    Slippage model: entry slippage represents the cost of crossing the spread
    to get filled (market order). Exit slippage is the same — you're crossing
    the spread on the way out. The original code had exit_slippage=0, which
    assumed perfect limit fills on exit. The corrected default ($0.03) matches
    entry slippage. Set exit_slippage_per_leg=0.0 to reproduce original results.

    Note on bid/ask: The chain data uses bar close prices, not bid/ask. The
    slippage parameters are a proxy for the half-spread cost. For ITM3 SPY
    options, typical spreads are $0.03-0.10; $0.03 is a conservative floor.
    During momentum moves, spreads can widen to $0.30-0.50.
    """
    slippage_per_leg: float = 0.03
    exit_slippage_per_leg: float = 0.03  # was 0.0 pre-audit
    commission_per_leg: float = 0.65

    @property
    def entry_friction(self) -> float:
        """Total entry cost for a 2-leg spread (slippage * 2 legs * 100 multiplier + commission * 2)."""
        return (self.slippage_per_leg * 2) * 100 + self.commission_per_leg * 2

    @property
    def exit_friction(self) -> float:
        """Total exit cost for a 2-leg spread."""
        return (self.exit_slippage_per_leg * 2) * 100 + self.commission_per_leg * 2

    @property
    def naked_entry_cost(self) -> float:
        """Entry cost for a single-leg naked option."""
        return self.slippage_per_leg * 100 + self.commission_per_leg

    @property
    def naked_exit_cost(self) -> float:
        """Exit cost for a single-leg naked option."""
        return self.exit_slippage_per_leg * 100 + self.commission_per_leg


@dataclass
class OptionsTrade:
    """A completed options trade with full metadata."""
    trade_date: str
    direction: str              # "long" or "short" (of the underlying direction)
    spread_type: str            # "call_debit" or "put_debit"
    long_strike: float
    short_strike: float
    expiry_date: str
    dte: int
    entry_time: str
    entry_debit: float          # per-contract debit paid (premium)
    exit_time: str = ""
    exit_value: float = 0.0     # per-contract spread value at exit
    exit_reason: str = ""
    pnl_per_contract: float = 0.0    # (exit_value - entry_debit) * 100
    pnl_net: float = 0.0             # after friction
    max_gain: float = 0.0       # max spread width (1.00 for $1-wide)
    mfe_pct: float = 0.0        # max favorable excursion as % of max_gain
    mae_pct: float = 0.0        # max adverse excursion as % of entry_debit
    result: str = ""            # "win" / "loss"
    spy_entry_price: float = 0.0
    spy_exit_price: float = 0.0

    @property
    def r_multiple(self) -> float:
        """P&L as multiple of risk (entry debit)."""
        if self.entry_debit <= 0:
            return 0.0
        return self.pnl_net / (self.entry_debit * 100)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_underlying(data_dir: Path, d: date) -> pd.DataFrame:
    """Load SPY underlying 1m bars for a day."""
    path = data_dir / "underlying" / f"SPY_{d.isoformat()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" in df.columns and not hasattr(df["ts"].dtype, "tz"):
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    return df


def load_chain(data_dir: Path, d: date, chain_subdir: str = "options_5dte") -> pd.DataFrame:
    """Load options chain 1m bars for a day."""
    path = data_dir / chain_subdir / f"chain_{d.isoformat()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" in df.columns and not hasattr(df["ts"].dtype, "tz"):
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    return df


def load_chain_meta(data_dir: Path, d: date, chain_subdir: str = "options_5dte") -> dict:
    """Load chain metadata JSON."""
    path = data_dir / chain_subdir / f"chain_{d.isoformat()}_meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_available_days(data_dir: Path, chain_subdir: str = "options_5dte") -> list[date]:
    """Get all days with both underlying AND chain data."""
    chain_dir = data_dir / chain_subdir
    und_dir = data_dir / "underlying"
    if not chain_dir.exists() or not und_dir.exists():
        return []

    chain_dates = set()
    for p in chain_dir.glob("chain_*.parquet"):
        name = p.stem  # chain_2024-03-01
        if "_meta" not in name:
            ds = name.replace("chain_", "")
            try:
                chain_dates.add(date.fromisoformat(ds))
            except ValueError:
                pass

    und_dates = set()
    for p in und_dir.glob("SPY_*.parquet"):
        ds = p.stem.replace("SPY_", "")
        try:
            und_dates.add(date.fromisoformat(ds))
        except ValueError:
            pass

    both = sorted(chain_dates & und_dates)
    return both


# ─── Spread Construction ─────────────────────────────────────────────────────


def _nearest_prior_bar(mins: np.ndarray, target: int) -> int | None:
    """Find index of nearest bar at or before target minute.

    Returns None if no bars exist at or before target.
    Used when FILL_NEXT_BAR=False (original same-bar fill behavior).
    """
    mask = mins <= target
    if not mask.any():
        return None
    candidates = np.where(mask)[0]
    return int(candidates[np.argmin(target - mins[candidates])])


def _nearest_next_bar(mins: np.ndarray, target: int, max_delay: int = 5) -> int | None:
    """Find index of first bar strictly after target minute.

    Returns None if no bars exist within (target, target + max_delay].
    Used when FILL_NEXT_BAR=True (realistic next-bar fill).
    """
    mask = (mins > target) & (mins <= target + max_delay)
    if not mask.any():
        return None
    candidates = np.where(mask)[0]
    return int(candidates[0])  # first bar after target


def _get_fill_bar(mins: np.ndarray, target: int) -> int | None:
    """Get the appropriate fill bar index based on FILL_NEXT_BAR config.

    When FILL_NEXT_BAR=True: returns first bar strictly after target minute.
    When FILL_NEXT_BAR=False: returns nearest bar at or before target minute.
    """
    if FILL_NEXT_BAR:
        return _nearest_next_bar(mins, target, FILL_NEXT_BAR_MAX_DELAY)
    else:
        return _nearest_prior_bar(mins, target)


def construct_trade(
    chain: pd.DataFrame,
    meta: dict,
    direction: str,
    entry_minute: int,
    trade_type: dict | None = None,
) -> dict | None:
    """Construct an options trade from actual chain data at a specific minute.

    Supports debit spreads (various widths) and naked/ITM single-leg options.

    Args:
        chain: 1m chain bars (all strikes, both C/P)
        meta: chain metadata (atm, strikes, dte, etc.)
        direction: "long" -> calls, "short" -> puts
        entry_minute: minute offset from 9:30 (e.g., 30 = 10:00 AM)
        trade_type: {"kind": "debit_spread", "width": 1} or
                    {"kind": "naked", "moneyness": "atm"/"itm1"/"itm2"/"itm3"}

    Returns:
        dict with trade info or None if can't construct.
    """
    if trade_type is None:
        trade_type = {"kind": "debit_spread", "width": 1}

    atm = meta.get("atm", 0)
    if atm <= 0:
        return None

    atm_strike = round(atm)
    kind = trade_type.get("kind", "debit_spread")

    # Target minute (absolute, e.g. 9:30=570, 10:00=600)
    target_min = 570 + entry_minute  # 9*60+30 + offset

    # Use pre-computed _minute column if available
    min_col = "_minute" if "_minute" in chain.columns else None

    if kind == "naked":
        # Single-leg option
        moneyness = trade_type.get("moneyness", "atm")
        cp = "C" if direction == "long" else "P"

        # ITM strikes: for calls, ITM means lower strike; for puts, higher strike
        itm_offset = {"atm": 0, "itm1": 1, "itm2": 2, "itm3": 3}.get(moneyness, 0)
        if cp == "C":
            strike = atm_strike - itm_offset
        else:
            strike = atm_strike + itm_offset

        strike_mask = chain["strike"].values
        cp_mask = chain["cp"].values
        leg_mask = (strike_mask == strike) & (cp_mask == cp)
        if not leg_mask.any():
            return None

        leg_bars = chain.loc[leg_mask]

        if min_col:
            leg_mins = leg_bars[min_col].values
        elif "ts" in leg_bars.columns:
            leg_bars = leg_bars.copy()
            leg_bars["minute"] = leg_bars["ts"].dt.hour * 60 + leg_bars["ts"].dt.minute
            leg_mins = leg_bars["minute"].values
            min_col = "minute"
        else:
            return None

        closest = _get_fill_bar(leg_mins, target_min)
        if closest is None:
            return None
        entry_premium = float(leg_bars.iloc[closest]["c"])
        actual_fill_min = int(leg_mins[closest])

        if entry_premium <= 0.05:  # too cheap, likely bad data
            return None

        entry_ts_str = str(leg_bars.iloc[closest]["ts"])

        if min_col == "_minute" and "minute" not in leg_bars.columns:
            leg_bars = leg_bars.rename(columns={"_minute": "minute"})

        return {
            "trade_kind": "naked",
            "spread_type": f"naked_{cp.lower()}_{moneyness}",
            "long_strike": strike,
            "short_strike": 0,
            "cp": cp,
            "entry_debit": entry_premium,
            "max_gain": entry_premium * 2,  # theoretical; naked has unlimited upside
            "spread_width": 0,
            "entry_time": entry_ts_str,
            "entry_minute": actual_fill_min,
            "long_bars": leg_bars,
            "short_bars": None,
        }

    elif kind == "credit_spread":
        # Credit spread: sell ATM, buy OTM for protection
        # Bull put credit: sell ATM put, buy lower put (direction="long", bullish)
        # Bear call credit: sell ATM call, buy higher call (direction="short", bearish)
        spread_width = trade_type.get("width", 1)

        if direction == "long":
            # Bull put credit spread
            short_strike = atm_strike       # sell ATM put (higher premium)
            long_strike = atm_strike - spread_width  # buy OTM put (protection)
            cp = "P"
        else:
            # Bear call credit spread
            short_strike = atm_strike       # sell ATM call (higher premium)
            long_strike = atm_strike + spread_width  # buy OTM call (protection)
            cp = "C"

        strike_mask = chain["strike"].values
        cp_mask = chain["cp"].values
        short_mask = (strike_mask == short_strike) & (cp_mask == cp)
        long_mask = (strike_mask == long_strike) & (cp_mask == cp)

        if not short_mask.any() or not long_mask.any():
            return None

        short_bars = chain.loc[short_mask]  # the leg we sold (ATM)
        long_bars = chain.loc[long_mask]    # the leg we bought (OTM protection)

        if min_col:
            short_mins = short_bars[min_col].values
            long_mins = long_bars[min_col].values
        elif "ts" in short_bars.columns:
            short_bars = short_bars.copy()
            long_bars = long_bars.copy()
            short_bars["minute"] = short_bars["ts"].dt.hour * 60 + short_bars["ts"].dt.minute
            long_bars["minute"] = long_bars["ts"].dt.hour * 60 + long_bars["ts"].dt.minute
            short_mins = short_bars["minute"].values
            long_mins = long_bars["minute"].values
            min_col = "minute"
        else:
            return None

        s_closest = _get_fill_bar(short_mins, target_min)
        l_closest = _get_fill_bar(long_mins, target_min)
        if s_closest is None or l_closest is None:
            return None

        short_price = float(short_bars.iloc[s_closest]["c"])  # ATM premium (we sell)
        long_price = float(long_bars.iloc[l_closest]["c"])    # OTM premium (we buy)
        credit = short_price - long_price  # net credit received
        actual_fill_min = int(short_mins[s_closest])

        if credit <= 0 or credit >= spread_width:
            return None

        entry_ts_str = str(short_bars.iloc[s_closest]["ts"])

        if min_col == "_minute" and "minute" not in short_bars.columns:
            short_bars = short_bars.rename(columns={"_minute": "minute"})
            long_bars = long_bars.rename(columns={"_minute": "minute"})

        # For scan_exit: we track the "position value" from seller's perspective.
        # entry_debit = credit (what we collected)
        # At any point, spread_value = short_close - long_close
        # Our P&L = (credit - current_spread_value) * 100
        # We profit when spread_value goes DOWN (toward 0).
        #
        # To reuse scan_exit uniformly: store bars so that
        # long_bars = short leg (ATM, higher value) and short_bars = long leg (OTM)
        # Then spread_val = long_close - short_close = ATM - OTM = current spread cost
        # scan_exit tracks this; for credit spreads, we want it to go DOWN.
        return {
            "trade_kind": "credit_spread",
            "spread_type": f"{'put' if direction == 'long' else 'call'}_credit_${spread_width}",
            "long_strike": long_strike,
            "short_strike": short_strike,
            "cp": cp,
            "entry_debit": credit,       # credit collected
            "max_gain": credit,           # max gain = keep the full credit
            "spread_width": spread_width,
            "entry_time": entry_ts_str,
            "entry_minute": actual_fill_min,
            # Store ATM (sold) as "long_bars" and OTM (bought) as "short_bars"
            # so spread_val = ATM_close - OTM_close = current spread cost to close
            "long_bars": short_bars,      # ATM leg (sold)
            "short_bars": long_bars,      # OTM leg (bought)
            # Scan-order strikes: match the swapped bar assignment above.
            # long_bars = ATM (short_strike), short_bars = OTM (long_strike)
            # Overnight chaining MUST use these to fetch correct legs on day 1+.
            "scan_long_strike": short_strike,   # strike for long_bars (ATM sold)
            "scan_short_strike": long_strike,   # strike for short_bars (OTM bought)
        }

    else:
        # Debit spread
        spread_width = trade_type.get("width", 1)

        if direction == "long":
            long_strike = atm_strike
            short_strike = atm_strike + spread_width
            cp = "C"
        else:
            long_strike = atm_strike
            short_strike = atm_strike - spread_width
            cp = "P"

        strike_mask = chain["strike"].values
        cp_mask = chain["cp"].values
        long_mask = (strike_mask == long_strike) & (cp_mask == cp)
        short_mask = (strike_mask == short_strike) & (cp_mask == cp)

        if not long_mask.any() or not short_mask.any():
            return None

        long_bars = chain.loc[long_mask]
        short_bars = chain.loc[short_mask]

        if min_col:
            long_mins = long_bars[min_col].values
            short_mins = short_bars[min_col].values
        elif "ts" in long_bars.columns:
            long_bars = long_bars.copy()
            short_bars = short_bars.copy()
            long_bars["minute"] = long_bars["ts"].dt.hour * 60 + long_bars["ts"].dt.minute
            short_bars["minute"] = short_bars["ts"].dt.hour * 60 + short_bars["ts"].dt.minute
            long_mins = long_bars["minute"].values
            short_mins = short_bars["minute"].values
            min_col = "minute"
        else:
            return None

        l_closest = _get_fill_bar(long_mins, target_min)
        s_closest = _get_fill_bar(short_mins, target_min)
        if l_closest is None or s_closest is None:
            return None

        long_price = float(long_bars.iloc[l_closest]["c"])
        short_price = float(short_bars.iloc[s_closest]["c"])
        entry_debit = long_price - short_price
        actual_fill_min = int(long_mins[l_closest])

        if entry_debit <= 0 or entry_debit >= spread_width:
            return None

        entry_ts_str = str(long_bars.iloc[l_closest]["ts"])

        if min_col == "_minute" and "minute" not in long_bars.columns:
            long_bars = long_bars.rename(columns={"_minute": "minute"})
            short_bars = short_bars.rename(columns={"_minute": "minute"})

        return {
            "trade_kind": "debit_spread",
            "spread_type": f"{'call' if direction == 'long' else 'put'}_debit_${spread_width}",
            "long_strike": long_strike,
            "short_strike": short_strike,
            "cp": cp,
            "entry_debit": entry_debit,
            "max_gain": spread_width - entry_debit,
            "spread_width": spread_width,
            "entry_time": entry_ts_str,
            "entry_minute": actual_fill_min,
            "long_bars": long_bars,
            "short_bars": short_bars,
        }


# Keep backward-compatible alias
construct_spread = construct_trade


# ─── Exit Scanning ───────────────────────────────────────────────────────────

def scan_exit(
    spread_info: dict,
    exit_rules: dict,
    cost: CostModel | None = None,
) -> dict:
    """Scan 1m bars forward from entry to find exit point (vectorized).

    Handles both debit spreads (two legs) and naked options (single leg).

    Exit rules:
        tp_pct: take profit as % of max_gain (spreads) or entry_premium (naked)
        sl_pct: stop loss as % of entry_debit/premium lost
        hold_minutes: max hold time in minutes
        eod_exit: True to force exit at 15:55 ET

    Returns dict with exit info.
    """
    if cost is None:
        cost = CostModel()

    entry_debit = spread_info["entry_debit"]
    spread_width = spread_info["spread_width"]
    max_gain = spread_info["max_gain"]
    entry_min = spread_info["entry_minute"]
    trade_kind = spread_info.get("trade_kind", "debit_spread")

    tp_pct = exit_rules.get("tp_pct", 0.80)
    sl_pct = exit_rules.get("sl_pct", 0.80)
    hold_minutes = exit_rules.get("hold_minutes", 120)
    eod_exit = exit_rules.get("eod_exit", True)

    long_bars = spread_info["long_bars"]

    no_data_ret = {
        "exit_value": entry_debit, "exit_reason": "no_data",
        "exit_time": spread_info["entry_time"],
        "mfe_pct": 0.0, "mae_pct": 0.0, "bars_scanned": 0,
    }

    tsl_pct = exit_rules.get("tsl_pct")

    if trade_kind == "naked":
        # Naked option: value = premium of the single leg
        tp_value = entry_debit * (1 + tp_pct)
        sl_value = entry_debit * (1 - sl_pct)

        long_mins = long_bars["minute"].values
        long_mask = long_mins > entry_min
        if not long_mask.any():
            return no_data_ret

        bar_mins = long_mins[long_mask]
        l_close = long_bars["c"].values
        l_low = long_bars["l"].values if "l" in long_bars.columns else l_close
        l_min_to_idx = {int(m): i for i, m in enumerate(long_mins)}
        l_indices = np.array([l_min_to_idx[int(m)] for m in bar_mins])
        vals = l_close[l_indices]
        bar_lows = l_low[l_indices]

        # Trailing stop for naked options — bar-by-bar high watermark tracking
        if tsl_pct is not None:
            max_minute = entry_min + hold_minutes
            eod_minute = 955
            ts_values = long_bars["ts"].values
            hwm = entry_debit  # high watermark starts at entry
            best_value = entry_debit
            worst_value = entry_debit

            for i in range(len(vals)):
                v = float(vals[i])
                bar_low = float(bar_lows[i])
                if v > hwm:
                    hwm = v
                best_value = max(best_value, v)
                worst_value = min(worst_value, v)

                # Trailing stop: premium drops tsl_pct below high watermark
                tsl_level = hwm * (1 - tsl_pct)
                # Only activate TSL once we're above entry (don't trail from a loss)
                tsl_hit = (hwm > entry_debit) and (v <= tsl_level)

                # Fixed SL still active
                sl_hit = v <= sl_value

                # Time / EOD
                bm = int(bar_mins[i])
                time_hit = bm >= max_minute
                eod_hit = (bm >= eod_minute) if eod_exit else False

                if tsl_hit or sl_hit or time_hit or eod_hit:
                    exit_ts = str(ts_values[l_indices[i]])
                    if tsl_hit:
                        exit_reason = "tsl"
                        if REALISTIC_STOP_FILLS:
                            # We're selling the option (closing a long). The worst
                            # fill within this bar is the bar low. Since the stop
                            # triggered (close <= tsl_level), bar_low <= close,
                            # so bar_low is at or below the trigger. Use bar_low
                            # as the realistic fill — you can't guarantee a fill
                            # at the exact trigger level.
                            exit_val = max(bar_low, 0.0)
                        else:
                            exit_val = tsl_level
                    elif sl_hit:
                        exit_reason = "sl"
                        if REALISTIC_STOP_FILLS:
                            # Same logic: selling at market when SL triggers,
                            # bar low is the worst-case fill.
                            exit_val = max(bar_low, 0.0)
                        else:
                            exit_val = sl_value
                    elif time_hit:
                        exit_reason = "time_stop"
                        exit_val = v
                    else:
                        exit_reason = "eod"
                        exit_val = v

                    mfe_pct = (best_value - entry_debit) / entry_debit if entry_debit > 0 else 0
                    mae_pct = (entry_debit - worst_value) / entry_debit if entry_debit > 0 else 0
                    return {
                        "exit_value": float(exit_val),
                        "exit_reason": exit_reason,
                        "exit_time": exit_ts,
                        "mfe_pct": max(0.0, mfe_pct),
                        "mae_pct": max(0.0, mae_pct),
                        "bars_scanned": len(bar_mins),
                    }

            # No exit triggered — hold to end
            exit_val = float(vals[-1])
            exit_ts = str(ts_values[l_indices[-1]])
            mfe_pct = (best_value - entry_debit) / entry_debit if entry_debit > 0 else 0
            mae_pct = (entry_debit - worst_value) / entry_debit if entry_debit > 0 else 0
            return {
                "exit_value": float(exit_val),
                "exit_reason": "hold",
                "exit_time": exit_ts,
                "mfe_pct": max(0.0, mfe_pct),
                "mae_pct": max(0.0, mae_pct),
                "bars_scanned": len(bar_mins),
            }

    elif trade_kind == "credit_spread":
        # Credit spread: vals = ATM_close - OTM_close = cost to close
        # We profit when vals goes DOWN (spread narrows).
        # TP: vals drops to credit * (1 - tp_pct) → keep tp_pct of credit
        # SL: vals rises to credit + max_loss * sl_pct
        short_bars = spread_info["short_bars"]
        long_mins = long_bars["minute"].values
        short_mins = short_bars["minute"].values
        long_mask = long_mins > entry_min
        short_mask = short_mins > entry_min

        if not long_mask.any() or not short_mask.any():
            return no_data_ret

        l_mins = long_mins[long_mask]
        s_mins = short_mins[short_mask]
        bar_mins = np.intersect1d(l_mins, s_mins)

        if len(bar_mins) == 0:
            return {**no_data_ret, "exit_reason": "no_merge"}

        l_close = long_bars["c"].values
        s_close = short_bars["c"].values
        l_min_to_idx = {int(m): i for i, m in enumerate(long_mins)}
        s_min_to_idx = {int(m): i for i, m in enumerate(short_mins)}

        l_indices = np.array([l_min_to_idx[int(m)] for m in bar_mins])
        s_indices = np.array([s_min_to_idx[int(m)] for m in bar_mins])
        vals = np.clip(l_close[l_indices] - s_close[s_indices], 0.0, spread_width)

        credit = entry_debit
        max_loss = spread_width - credit
        # TP when spread narrows: close cost drops below threshold
        tp_value = credit * (1 - tp_pct)  # e.g., tp_pct=0.50 → close at 50% of credit
        # SL when spread widens: close cost rises above threshold
        sl_value = credit + max_loss * sl_pct  # e.g., sl_pct=0.50 → lose 50% of max loss

        # For credit spreads, TP = vals <= tp_value, SL = vals >= sl_value (inverted)
        max_minute = entry_min + hold_minutes
        eod_minute = 955

        tp_hits = vals <= tp_value
        sl_hits = vals >= sl_value
        time_hits = bar_mins >= max_minute
        eod_hits = bar_mins >= eod_minute if eod_exit else np.zeros(len(bar_mins), dtype=bool)

        any_exit = tp_hits | sl_hits | time_hits | eod_hits
        exit_indices = np.where(any_exit)[0]

        ts_values = long_bars["ts"].values

        if len(exit_indices) > 0:
            ei = exit_indices[0]
            exit_val = float(vals[ei])
            exit_ts = str(ts_values[l_indices[ei]])
            if tp_hits[ei]:
                exit_reason = "tp"
                exit_val = tp_value
            elif sl_hits[ei]:
                exit_reason = "sl"
                if REALISTIC_STOP_FILLS:
                    # Credit spread SL: spread widened against us. Actual bar
                    # value (vals[ei]) is at or above sl_value. Use actual.
                    exit_val = float(vals[ei])
                else:
                    exit_val = sl_value
            elif time_hits[ei]:
                exit_reason = "time_stop"
            else:
                exit_reason = "eod"
            # For credit: best = lowest spread cost, worst = highest
            best_value = float(vals[:ei + 1].min())
            worst_value = float(vals[:ei + 1].max())
        else:
            exit_val = float(vals[-1])
            exit_ts = str(ts_values[l_indices[-1]])
            exit_reason = "hold"
            best_value = float(vals.min())
            worst_value = float(vals.max())

        # MFE: how much the spread narrowed in our favor
        mfe_pct = (credit - best_value) / credit if credit > 0 else 0
        # MAE: how much the spread widened against us
        mae_pct = (worst_value - credit) / max_loss if max_loss > 0 else 0

        return {
            "exit_value": float(exit_val),
            "exit_reason": exit_reason,
            "exit_time": exit_ts,
            "mfe_pct": max(0.0, mfe_pct),
            "mae_pct": max(0.0, mae_pct),
            "bars_scanned": len(bar_mins),
        }

    else:
        # Debit spread: value = long_close - short_close, clipped to [0, width]
        short_bars = spread_info["short_bars"]
        long_mins = long_bars["minute"].values
        short_mins = short_bars["minute"].values
        long_mask = long_mins > entry_min
        short_mask = short_mins > entry_min

        if not long_mask.any() or not short_mask.any():
            return no_data_ret

        l_mins = long_mins[long_mask]
        s_mins = short_mins[short_mask]
        bar_mins = np.intersect1d(l_mins, s_mins)

        if len(bar_mins) == 0:
            return {**no_data_ret, "exit_reason": "no_merge"}

        l_close = long_bars["c"].values
        s_close = short_bars["c"].values
        l_min_to_idx = {int(m): i for i, m in enumerate(long_mins)}
        s_min_to_idx = {int(m): i for i, m in enumerate(short_mins)}

        l_indices = np.array([l_min_to_idx[int(m)] for m in bar_mins])
        s_indices = np.array([s_min_to_idx[int(m)] for m in bar_mins])
        vals = np.clip(l_close[l_indices] - s_close[s_indices], 0.0, spread_width)

        tp_value = entry_debit + max_gain * tp_pct
        sl_value = entry_debit * (1 - sl_pct)

    # Find first exit condition hit (debit spread and naked share this logic)
    max_minute = entry_min + hold_minutes
    eod_minute = 955  # 15:55

    tp_hits = vals >= tp_value
    sl_hits = vals <= sl_value
    time_hits = bar_mins >= max_minute
    eod_hits = bar_mins >= eod_minute if eod_exit else np.zeros(len(bar_mins), dtype=bool)

    any_exit = tp_hits | sl_hits | time_hits | eod_hits
    exit_indices = np.where(any_exit)[0]

    ts_values = long_bars["ts"].values

    if len(exit_indices) > 0:
        ei = exit_indices[0]
        exit_val = float(vals[ei])
        exit_ts = str(ts_values[l_indices[ei]])

        if tp_hits[ei]:
            exit_reason = "tp"
            # TP fill: for debit spreads, using tp_value is fine (limit order fills
            # at your price or better). For naked, same logic applies.
            exit_val = tp_value
        elif sl_hits[ei]:
            exit_reason = "sl"
            if REALISTIC_STOP_FILLS:
                # SL triggered: actual bar value (vals[ei]) is at or below sl_value.
                # Market order fills at the actual price, not the trigger level.
                exit_val = float(vals[ei])
            else:
                exit_val = sl_value
        elif time_hits[ei]:
            exit_reason = "time_stop"
        else:
            exit_reason = "eod"

        best_value = max(entry_debit, float(vals[:ei + 1].max()))
        worst_value = min(entry_debit, float(vals[:ei + 1].min()))
    else:
        exit_val = float(vals[-1])
        exit_ts = str(ts_values[l_indices[-1]])
        exit_reason = "hold"
        best_value = max(entry_debit, float(vals.max()))
        worst_value = min(entry_debit, float(vals.min()))

    if trade_kind == "naked":
        mfe_pct = (best_value - entry_debit) / entry_debit if entry_debit > 0 else 0
    else:
        mfe_pct = (best_value - entry_debit) / max_gain if max_gain > 0 else 0
    mae_pct = (entry_debit - worst_value) / entry_debit if entry_debit > 0 else 0

    return {
        "exit_value": float(exit_val),
        "exit_reason": exit_reason,
        "exit_time": exit_ts,
        "mfe_pct": max(0.0, mfe_pct),
        "mae_pct": max(0.0, mae_pct),
        "bars_scanned": len(bar_mins),
    }


# ─── Multi-Day Exit Scanning ─────────────────────────────────────────────────

# Minutes per trading session (9:30→16:00 = 390 minutes)
_RTH_MINUTES = 390
_RTH_START_MINUTE = 570  # 9*60+30


def _chain_bars_for_contract(
    chain: pd.DataFrame, strike: float, cp: str,
    option_ticker: str | None = None,
) -> pd.DataFrame | None:
    """Extract bars for a specific contract from a chain.

    Args:
        option_ticker: if provided, filter by exact OCC ticker to ensure
            same-contract continuity across days (prevents expiry mismatch).
    """
    if option_ticker and "option_ticker" in chain.columns:
        mask = chain["option_ticker"] == option_ticker
    else:
        mask = (chain["strike"] == strike) & (chain["cp"] == cp)
    if not mask.any():
        return None
    bars = chain.loc[mask].copy()
    if "minute" not in bars.columns:
        if "_minute" in bars.columns:
            bars["minute"] = bars["_minute"]
        elif "ts" in bars.columns:
            bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
        else:
            return None
    return bars


def _concat_multiday_bars(
    day_bars_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate bars across days with session-relative minutes.

    Converts raw minute-of-day (570-960) to session-relative minutes so that
    time stops measure actual trading minutes, not clock minutes with gaps.

    Day 0 bar at 9:30 (min 570) → session minute 0
    Day 0 bar at 16:00 (min 960) → session minute 390
    Day 1 bar at 9:30 (min 570) → session minute 390
    Day 1 bar at 16:00 (min 960) → session minute 780
    etc.
    """
    parts = []
    for day_offset, bars in enumerate(day_bars_list):
        if bars is None or bars.empty:
            continue
        b = bars.copy()
        # Convert to session-relative: (minute - RTH_start) + day_offset * 390
        b["minute"] = (b["minute"] - _RTH_START_MINUTE) + (day_offset * _RTH_MINUTES)
        parts.append(b)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def evaluate_day_overnight(
    und: pd.DataFrame,
    chain: pd.DataFrame,
    meta: dict,
    genes: dict,
    subsequent_chains: list[pd.DataFrame],
    cost: CostModel | None = None,
    _ctx: dict | None = None,
) -> OptionsTrade | None:
    """Evaluate a gene set with multi-day hold support.

    Entry signals are evaluated on the entry day only (same as evaluate_day).
    If eod_exit=False and the trade doesn't hit TP/SL/time_stop intraday,
    it continues scanning on subsequent days' bars for the same contract.

    Args:
        subsequent_chains: list of chain DataFrames for days AFTER the entry day,
            pre-filtered to contain the same expiry. Ordered chronologically.
    """
    if cost is None:
        cost = CostModel()

    if und.empty or chain.empty or not meta:
        return None

    trade_date = meta.get("trade_date", "")
    expiry_date = meta.get("expiry_date", "")
    dte = meta.get("dte", 5)

    tw = genes.get("time_window", {"start": "09:45", "end": "14:00"})
    tw_s = tw.get("start", "09:45").split(":")
    tw_e = tw.get("end", "14:00").split(":")
    tw_start_min = int(tw_s[0]) * 60 + int(tw_s[1])
    tw_end_min = int(tw_e[0]) * 60 + int(tw_e[1])

    ctx = _ctx
    if ctx is None:
        ctx = precompute_day_context(und)
        if ctx is None:
            return None

    direction = genes.get("direction", "long")
    entry_conditions = list(genes.get("entry_conditions", []))
    fc = genes.get("filter_condition")
    if fc is not None:
        entry_conditions.append(fc)
    exit_rules = genes.get("exit_rules", {})

    regime_filters = genes.get("regime_filters", [])
    if regime_filters and not _check_regime_filters(regime_filters, ctx, direction):
        return None

    minutes = ctx["minutes"]
    closes = ctx["closes"]
    eod_exit = exit_rules.get("eod_exit", True)

    for idx in range(ctx["n"]):
        bar_min = int(minutes[idx])

        if bar_min < tw_start_min or bar_min > tw_end_min:
            continue

        if not _check_entry_conditions_fast(entry_conditions, ctx, idx, meta):
            continue

        entry_minute = bar_min - _RTH_START_MINUTE
        trade_type = genes.get("trade_type", {"kind": "debit_spread", "width": 1})
        trade_info = construct_trade(chain, meta, direction, entry_minute, trade_type)
        if trade_info is None:
            continue

        # If eod_exit=True or no subsequent data, use standard same-day scan
        if eod_exit or not subsequent_chains:
            exit_info = scan_exit(trade_info, exit_rules, cost)
        else:
            # Build multi-day bars for the same contract
            # Use scan-order strikes (accounts for credit spread leg swap)
            scan_long_strike = trade_info.get("scan_long_strike", trade_info["long_strike"])
            scan_short_strike = trade_info.get("scan_short_strike", trade_info.get("short_strike", 0))
            tk = trade_info.get("trade_kind", "debit_spread")
            cp = trade_info.get("cp", "C" if direction == "long" else "P")

            # Extract option_tickers from entry-day bars for same-contract continuity
            long_bars_d0 = trade_info["long_bars"]
            short_bars_d0 = trade_info.get("short_bars")
            long_ticker = None
            short_ticker = None
            if "option_ticker" in long_bars_d0.columns and not long_bars_d0.empty:
                long_ticker = str(long_bars_d0["option_ticker"].iloc[0])
            if short_bars_d0 is not None and "option_ticker" in short_bars_d0.columns and not short_bars_d0.empty:
                short_ticker = str(short_bars_d0["option_ticker"].iloc[0])

            # Collect long leg bars across days (using scan-order strike + ticker)
            long_day_bars = [long_bars_d0]
            for next_chain in subsequent_chains:
                next_bars = _chain_bars_for_contract(
                    next_chain, scan_long_strike, cp, option_ticker=long_ticker)
                if next_bars is not None:
                    long_day_bars.append(next_bars)

            # Collect short leg bars across days (for spreads, using scan-order strike + ticker)
            short_day_bars = None
            if tk != "naked" and short_bars_d0 is not None:
                short_day_bars = [short_bars_d0]
                for next_chain in subsequent_chains:
                    next_bars = _chain_bars_for_contract(
                        next_chain, scan_short_strike, cp, option_ticker=short_ticker)
                    if next_bars is not None:
                        short_day_bars.append(next_bars)

            # Concatenate with session-relative minutes
            multi_long = _concat_multiday_bars(long_day_bars)
            if multi_long.empty:
                exit_info = scan_exit(trade_info, exit_rules, cost)
            else:
                multi_trade_info = dict(trade_info)
                multi_trade_info["long_bars"] = multi_long
                # Convert entry_minute to session-relative to match bar minutes
                raw_entry_min = trade_info["entry_minute"]
                multi_trade_info["entry_minute"] = raw_entry_min - _RTH_START_MINUTE
                if short_day_bars:
                    multi_short = _concat_multiday_bars(short_day_bars)
                    multi_trade_info["short_bars"] = multi_short
                exit_info = scan_exit(multi_trade_info, exit_rules, cost)

        entry_debit = trade_info["entry_debit"]
        exit_value = exit_info["exit_value"]
        tk = trade_info.get("trade_kind", "debit_spread")

        if tk == "credit_spread":
            pnl_gross = (entry_debit - exit_value) * 100
        else:
            pnl_gross = (exit_value - entry_debit) * 100

        if tk == "naked":
            entry_cost = cost.naked_entry_cost
            exit_cost = cost.naked_exit_cost
        else:
            entry_cost = cost.entry_friction
            exit_cost = cost.exit_friction
        pnl_net = pnl_gross - entry_cost - exit_cost

        return OptionsTrade(
            trade_date=trade_date,
            direction=direction,
            spread_type=trade_info["spread_type"],
            long_strike=trade_info["long_strike"],
            short_strike=trade_info.get("short_strike", 0),
            expiry_date=expiry_date,
            dte=dte,
            entry_time=trade_info["entry_time"],
            entry_debit=entry_debit,
            exit_time=exit_info["exit_time"],
            exit_value=exit_value,
            exit_reason=exit_info["exit_reason"],
            pnl_per_contract=pnl_gross,
            pnl_net=pnl_net,
            max_gain=trade_info["max_gain"],
            mfe_pct=exit_info["mfe_pct"],
            mae_pct=exit_info["mae_pct"],
            result="win" if pnl_net > 0 else "loss",
            spy_entry_price=float(closes[idx]),
        )

    return None


# ─── Single-Day Evaluation ───────────────────────────────────────────────────

def precompute_day_context(und: pd.DataFrame, chain: pd.DataFrame | None = None,
                           meta: dict | None = None) -> dict | None:
    """Pre-compute numpy arrays from underlying bars for fast condition checking.

    Call once per day, reuse across all candidates.
    Optionally enriches with chain-derived signals (PCR, skew, strike flow).
    """
    if und.empty or "ts" not in und.columns:
        return None

    n = len(und)
    closes = und["c"].values.astype(np.float64)
    opens = und["o"].values.astype(np.float64)
    highs = und["h"].values.astype(np.float64)
    lows = und["l"].values.astype(np.float64)
    vols = und["vol"].values.astype(np.float64)

    # Minutes (absolute: 9:30=570, etc.)
    minutes = (und["ts"].dt.hour * 60 + und["ts"].dt.minute).values.astype(np.int32)

    # Cumulative VWAP
    cum_pv = np.cumsum(closes * vols)
    cum_v = np.cumsum(vols)
    vwap = np.where(cum_v > 0, cum_pv / cum_v, closes)

    # Running high/low
    run_high = np.maximum.accumulate(highs)
    run_low = np.minimum.accumulate(lows)

    day_open = float(opens[0])

    # Opening volume: sum of first 5 bars
    open_vol_5 = float(vols[:min(5, n)].sum())
    # Average 5-bar volume across first hour in 5-bar chunks
    if n >= 10:
        usable = min(60, n) // 5 * 5  # trim to multiple of 5
        avg_5bar_vol = float(vols[:usable].reshape(-1, 5).sum(axis=1).mean())
    else:
        avg_5bar_vol = open_vol_5

    ctx = {
        "closes": closes, "opens": opens, "highs": highs, "lows": lows,
        "vols": vols, "minutes": minutes, "vwap": vwap,
        "run_high": run_high, "run_low": run_low,
        "day_open": day_open, "n": n,
        # Regime-level fields (day-level, computed once)
        "open_vol_5": open_vol_5,
        "avg_5bar_vol": avg_5bar_vol,
    }

    # Previous day range (set by caller via meta)
    if meta:
        prev_close = meta.get("prev_close", 0)
        prev_high = meta.get("prev_high", 0)
        prev_low = meta.get("prev_low", 0)
        if prev_close > 0:
            gap_pct = (day_open - prev_close) / prev_close * 100
            ctx["gap_pct"] = gap_pct
        if prev_high > 0 and prev_low > 0 and prev_close > 0:
            ctx["prev_range_pct"] = (prev_high - prev_low) / prev_close * 100

    # Chain-derived signals (per-minute rolling arrays aligned to underlying bars)
    if chain is not None and not chain.empty and meta:
        _enrich_ctx_with_chain(ctx, chain, meta, minutes)

    return ctx


def _enrich_ctx_with_chain(ctx: dict, chain: pd.DataFrame, meta: dict,
                           und_minutes: np.ndarray) -> None:
    """Add chain-derived per-minute signals to context dict.

    Uses groupby aggregation + forward-fill for O(n) instead of O(n²).
    """
    n = len(und_minutes)
    atm = round(meta.get("atm", 0))
    if atm <= 0:
        return

    min_col = "_minute" if "_minute" in chain.columns else None
    if min_col is None and "ts" in chain.columns:
        chain_mins = (chain["ts"].dt.hour * 60 + chain["ts"].dt.minute).values.astype(np.int32)
    elif min_col:
        chain_mins = chain[min_col].values.astype(np.int32)
    else:
        return

    cp_vals = chain["cp"].values
    strike_vals = chain["strike"].values
    vol_vals = chain["vol"].values.astype(np.float64)
    close_vals = chain["c"].values.astype(np.float64)

    call_mask = cp_vals == "C"
    put_mask = cp_vals == "P"

    # Build per-minute aggregates using numpy bincount-style approach
    # Minute range: typically 570 (9:30) to 960 (16:00)
    min_min = int(und_minutes[0])
    max_min = int(und_minutes[-1])
    n_bins = max_min - min_min + 1

    def _bin_sum(mask, offset=min_min):
        """Sum vol_vals[mask] into per-minute bins, then cumsum."""
        m = chain_mins[mask] - offset
        v = vol_vals[mask]
        valid = (m >= 0) & (m < n_bins)
        bins = np.zeros(n_bins, dtype=np.float64)
        np.add.at(bins, m[valid], v[valid])
        return np.cumsum(bins)

    # Cumulative call/put volumes per minute
    cum_call_vol = _bin_sum(call_mask)
    cum_put_vol = _bin_sum(put_mask)
    cum_otm_call_vol = _bin_sum(call_mask & (strike_vals > atm))
    cum_otm_put_vol = _bin_sum(put_mask & (strike_vals < atm))

    # ATM skew: last close at each minute for ATM call and ATM put
    atm_call_mask = call_mask & (strike_vals == atm)
    atm_put_mask = put_mask & (strike_vals == atm)

    # For skew, grab last close per minute
    atm_c_by_min = np.zeros(n_bins, dtype=np.float64)
    atm_p_by_min = np.zeros(n_bins, dtype=np.float64)

    ac_mins = chain_mins[atm_call_mask] - min_min
    ac_close = close_vals[atm_call_mask]
    for j in range(len(ac_mins)):
        m_idx = ac_mins[j]
        if 0 <= m_idx < n_bins:
            atm_c_by_min[m_idx] = ac_close[j]

    ap_mins = chain_mins[atm_put_mask] - min_min
    ap_close = close_vals[atm_put_mask]
    for j in range(len(ap_mins)):
        m_idx = ap_mins[j]
        if 0 <= m_idx < n_bins:
            atm_p_by_min[m_idx] = ap_close[j]

    # Forward-fill: carry last non-zero close forward
    for j in range(1, n_bins):
        if atm_c_by_min[j] == 0:
            atm_c_by_min[j] = atm_c_by_min[j - 1]
        if atm_p_by_min[j] == 0:
            atm_p_by_min[j] = atm_p_by_min[j - 1]

    # Map to underlying minute indices
    pcr = np.ones(n, dtype=np.float64)
    skew = np.ones(n, dtype=np.float64)
    call_vol_ratio = np.ones(n, dtype=np.float64)
    put_vol_ratio = np.ones(n, dtype=np.float64)

    for i in range(n):
        b = int(und_minutes[i]) - min_min
        if b < 0 or b >= n_bins:
            continue
        cv = cum_call_vol[b]
        pv = cum_put_vol[b]
        if cv > 0:
            pcr[i] = pv / cv
            call_vol_ratio[i] = cum_otm_call_vol[b] / cv
        if pv > 0:
            put_vol_ratio[i] = cum_otm_put_vol[b] / pv
        if atm_c_by_min[b] > 0:
            skew[i] = atm_p_by_min[b] / atm_c_by_min[b]

    ctx["pcr"] = pcr
    ctx["skew"] = skew
    ctx["call_vol_ratio"] = call_vol_ratio
    ctx["put_vol_ratio"] = put_vol_ratio

    # Regime-level chain signals: IV proxy and opening skew
    # IV proxy = ATM call premium at open as % of underlying price
    day_open = ctx.get("day_open", 0)
    if atm_c_by_min[0] > 0 and day_open > 0:
        ctx["iv_proxy_pct"] = atm_c_by_min[0] / day_open * 100
    # Opening skew = ATM put / ATM call at open
    if atm_c_by_min[0] > 0 and atm_p_by_min[0] > 0:
        ctx["open_skew"] = atm_p_by_min[0] / atm_c_by_min[0]


def precompute_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Add _minute column to chain once per day for reuse."""
    if "_minute" not in chain.columns and "ts" in chain.columns:
        chain = chain.copy()
        chain["_minute"] = chain["ts"].dt.hour * 60 + chain["ts"].dt.minute
    return chain


def evaluate_day(
    und: pd.DataFrame,
    chain: pd.DataFrame,
    meta: dict,
    genes: dict,
    cost: CostModel | None = None,
    _ctx: dict | None = None,
) -> OptionsTrade | None:
    """Evaluate a gene set on a single trading day.

    Args:
        und: SPY 1m bars for the day
        chain: options chain 1m bars (with _minute pre-computed)
        meta: chain metadata
        genes: gene dict with entry_conditions, direction, time_window, exit_rules
        cost: optional cost model override
        _ctx: pre-computed day context (from precompute_day_context)

    Returns:
        OptionsTrade if a trade was taken, None otherwise.
    """
    if cost is None:
        cost = CostModel()

    if und.empty or chain.empty or not meta:
        return None

    trade_date = meta.get("trade_date", "")
    expiry_date = meta.get("expiry_date", "")
    dte = meta.get("dte", 5)

    # Parse time window as absolute minutes
    tw = genes.get("time_window", {"start": "09:45", "end": "14:00"})
    tw_s = tw.get("start", "09:45").split(":")
    tw_e = tw.get("end", "14:00").split(":")
    tw_start_min = int(tw_s[0]) * 60 + int(tw_s[1])
    tw_end_min = int(tw_e[0]) * 60 + int(tw_e[1])

    # Use pre-computed context or build it
    ctx = _ctx
    if ctx is None:
        ctx = precompute_day_context(und)
        if ctx is None:
            return None

    direction = genes.get("direction", "long")
    entry_conditions = list(genes.get("entry_conditions", []))
    # Merge filter_condition into entry conditions if present
    fc = genes.get("filter_condition")
    if fc is not None:
        entry_conditions.append(fc)
    exit_rules = genes.get("exit_rules", {})

    # Check regime filters (day-level, before bar scanning)
    regime_filters = genes.get("regime_filters", [])
    if regime_filters and not _check_regime_filters(regime_filters, ctx, direction):
        return None

    minutes = ctx["minutes"]
    closes = ctx["closes"]

    for idx in range(ctx["n"]):
        bar_min = int(minutes[idx])

        if bar_min < tw_start_min or bar_min > tw_end_min:
            continue

        if not _check_entry_conditions_fast(entry_conditions, ctx, idx, meta):
            continue

        # Signal fired — construct trade (same-bar fill; SPY options liquidity supports this)
        entry_minute = bar_min - 570  # offset from 9:30
        trade_type = genes.get("trade_type", {"kind": "debit_spread", "width": 1})
        trade_info = construct_trade(chain, meta, direction, entry_minute, trade_type)
        if trade_info is None:
            continue

        exit_info = scan_exit(trade_info, exit_rules, cost)

        entry_debit = trade_info["entry_debit"]
        exit_value = exit_info["exit_value"]
        tk = trade_info.get("trade_kind", "debit_spread")

        if tk == "credit_spread":
            # Credit spread: P&L = (credit - close_cost) * 100
            pnl_gross = (entry_debit - exit_value) * 100
        else:
            # Debit spread / naked: P&L = (exit - entry) * 100
            pnl_gross = (exit_value - entry_debit) * 100

        # Friction: naked = single leg, spreads = two legs
        if tk == "naked":
            entry_cost = cost.naked_entry_cost
            exit_cost = cost.naked_exit_cost
        else:
            entry_cost = cost.entry_friction
            exit_cost = cost.exit_friction
        pnl_net = pnl_gross - entry_cost - exit_cost

        return OptionsTrade(
            trade_date=trade_date,
            direction=direction,
            spread_type=trade_info["spread_type"],
            long_strike=trade_info["long_strike"],
            short_strike=trade_info["short_strike"],
            expiry_date=expiry_date,
            dte=dte,
            entry_time=trade_info["entry_time"],
            entry_debit=entry_debit,
            exit_time=exit_info["exit_time"],
            exit_value=exit_value,
            exit_reason=exit_info["exit_reason"],
            pnl_per_contract=pnl_gross,
            pnl_net=pnl_net,
            max_gain=trade_info["max_gain"],
            mfe_pct=exit_info["mfe_pct"],
            mae_pct=exit_info["mae_pct"],
            result="win" if pnl_net > 0 else "loss",
            spy_entry_price=float(closes[idx]),
        )

    return None


def evaluate_day_multi(
    und: pd.DataFrame,
    chain: pd.DataFrame,
    meta: dict,
    genes: dict,
    cost: CostModel | None = None,
    _ctx: dict | None = None,
) -> list[OptionsTrade]:
    """Evaluate a gene set on a single day, allowing re-entry after exit.

    Same logic as evaluate_day but after a trade exits (TP, SL, TSL, time_stop),
    continues scanning for new entries from the bar after exit. EOD exits and
    hold-to-end do NOT allow re-entry (day is over).

    Returns list of trades (0 to N). Empty list if no trades.
    """
    if cost is None:
        cost = CostModel()

    if und.empty or chain.empty or not meta:
        return []

    trade_date = meta.get("trade_date", "")
    expiry_date = meta.get("expiry_date", "")
    dte = meta.get("dte", 5)

    tw = genes.get("time_window", {"start": "09:45", "end": "14:00"})
    tw_s = tw.get("start", "09:45").split(":")
    tw_e = tw.get("end", "14:00").split(":")
    tw_start_min = int(tw_s[0]) * 60 + int(tw_s[1])
    tw_end_min = int(tw_e[0]) * 60 + int(tw_e[1])

    ctx = _ctx
    if ctx is None:
        ctx = precompute_day_context(und)
        if ctx is None:
            return []

    direction = genes.get("direction", "long")
    entry_conditions = list(genes.get("entry_conditions", []))
    fc = genes.get("filter_condition")
    if fc is not None:
        entry_conditions.append(fc)
    exit_rules = genes.get("exit_rules", {})

    regime_filters = genes.get("regime_filters", [])
    if regime_filters and not _check_regime_filters(regime_filters, ctx, direction):
        return []

    minutes = ctx["minutes"]
    closes = ctx["closes"]
    hold_minutes = exit_rules.get("hold_minutes", 120)

    trades = []
    # resume_after: absolute minute after which we can scan for new entries
    resume_after = 0

    for idx in range(ctx["n"]):
        bar_min = int(minutes[idx])

        if bar_min < tw_start_min or bar_min > tw_end_min:
            continue

        # Skip bars before previous trade's exit
        if bar_min <= resume_after:
            continue

        if not _check_entry_conditions_fast(entry_conditions, ctx, idx, meta):
            continue

        entry_minute = bar_min - 570
        trade_type = genes.get("trade_type", {"kind": "debit_spread", "width": 1})
        trade_info = construct_trade(chain, meta, direction, entry_minute, trade_type)
        if trade_info is None:
            continue

        exit_info = scan_exit(trade_info, exit_rules, cost)

        entry_debit = trade_info["entry_debit"]
        exit_value = exit_info["exit_value"]
        tk = trade_info.get("trade_kind", "debit_spread")

        if tk == "credit_spread":
            pnl_gross = (entry_debit - exit_value) * 100
        else:
            pnl_gross = (exit_value - entry_debit) * 100

        if tk == "naked":
            entry_cost = cost.naked_entry_cost
            exit_cost = cost.naked_exit_cost
        else:
            entry_cost = cost.entry_friction
            exit_cost = cost.exit_friction
        pnl_net = pnl_gross - entry_cost - exit_cost

        trade = OptionsTrade(
            trade_date=trade_date,
            direction=direction,
            spread_type=trade_info["spread_type"],
            long_strike=trade_info["long_strike"],
            short_strike=trade_info["short_strike"],
            expiry_date=expiry_date,
            dte=dte,
            entry_time=trade_info["entry_time"],
            entry_debit=entry_debit,
            exit_time=exit_info["exit_time"],
            exit_value=exit_value,
            exit_reason=exit_info["exit_reason"],
            pnl_per_contract=pnl_gross,
            pnl_net=pnl_net,
            max_gain=trade_info["max_gain"],
            mfe_pct=exit_info["mfe_pct"],
            mae_pct=exit_info["mae_pct"],
            result="win" if pnl_net > 0 else "loss",
            spy_entry_price=float(closes[idx]),
        )
        trades.append(trade)

        # Determine exit minute to set resume_after
        exit_reason = exit_info["exit_reason"]
        # EOD / hold = day is done, no re-entry possible
        if exit_reason in ("eod", "hold"):
            break

        # For TP/SL/TSL/time_stop: compute exit minute from entry + bars scanned
        # The exit happened within hold_minutes of entry at most
        # Use entry bar_min + hold_minutes as upper bound, but actual exit is earlier
        # Parse exit time to get exact minute
        exit_time_str = exit_info.get("exit_time", "")
        try:
            # exit_time is a timestamp like "2025-03-03 10:45:00-05:00"
            # extract HH:MM
            time_part = exit_time_str.split(" ")[1] if " " in exit_time_str else ""
            hh, mm = time_part.split(":")[:2]
            exit_abs_min = int(hh) * 60 + int(mm)
        except (ValueError, IndexError):
            # Fallback: entry + hold_minutes
            exit_abs_min = bar_min + hold_minutes

        resume_after = exit_abs_min

    return trades


# ─── Regime Filter Evaluation ────────────────────────────────────────────────

def _check_regime_filters(
    filters: list[dict],
    ctx: dict,
    direction: str,
) -> bool:
    """Check day-level regime filters. Returns False to skip the entire day."""
    for f in filters:
        ftype = f.get("type", "")

        if ftype == "iv_proxy_high":
            iv = ctx.get("iv_proxy_pct", 0)
            if iv < f.get("min_pct", 0.8):
                return False

        elif ftype == "iv_proxy_low":
            iv = ctx.get("iv_proxy_pct", 999)
            if iv > f.get("max_pct", 0.6):
                return False

        elif ftype == "prev_range_wide":
            pr = ctx.get("prev_range_pct", 0)
            if pr < f.get("min_pct", 0.8):
                return False

        elif ftype == "prev_range_narrow":
            pr = ctx.get("prev_range_pct", 999)
            if pr > f.get("max_pct", 0.5):
                return False

        elif ftype == "gap_aligns":
            gap = ctx.get("gap_pct", 0)
            min_pct = f.get("min_pct", 0.1)
            # Gap aligns: long wants gap up, short wants gap down
            if direction == "long" and gap < min_pct:
                return False
            if direction == "short" and gap > -min_pct:
                return False

        elif ftype == "gap_opposes":
            gap = ctx.get("gap_pct", 0)
            min_pct = f.get("min_pct", 0.1)
            # Gap opposes: long wants gap down (fade), short wants gap up (fade)
            if direction == "long" and gap > -min_pct:
                return False
            if direction == "short" and gap < min_pct:
                return False

        elif ftype == "no_gap":
            gap = ctx.get("gap_pct", 999)
            if abs(gap) > f.get("max_pct", 0.1):
                return False

        elif ftype == "open_vol_high":
            ov = ctx.get("open_vol_5", 0)
            avg = ctx.get("avg_5bar_vol", 1)
            if avg <= 0 or ov / avg < f.get("multiplier", 1.5):
                return False

        elif ftype == "open_vol_low":
            ov = ctx.get("open_vol_5", 0)
            avg = ctx.get("avg_5bar_vol", 1)
            if avg <= 0 or ov / avg > f.get("max_mult", 0.7):
                return False

        elif ftype == "open_skew_high":
            sk = ctx.get("open_skew", 1.0)
            if sk < f.get("threshold", 1.2):
                return False

        elif ftype == "open_skew_low":
            sk = ctx.get("open_skew", 1.0)
            if sk > f.get("threshold", 0.8):
                return False

        else:
            return False  # unknown filter type → skip day

    return True


# ─── Entry Condition Evaluation ──────────────────────────────────────────────

def _check_entry_conditions_fast(
    conditions: list[dict],
    ctx: dict,
    bar_idx: int,
    meta: dict,
) -> bool:
    """Fast entry condition check using pre-computed numpy arrays."""
    closes = ctx["closes"]
    opens = ctx["opens"]
    highs = ctx["highs"]
    lows = ctx["lows"]
    vols = ctx["vols"]
    vwap = ctx["vwap"]
    run_high = ctx["run_high"]
    run_low = ctx["run_low"]
    day_open = ctx["day_open"]

    price = closes[bar_idx]

    for cond in conditions:
        ctype = cond.get("type", "")

        if ctype == "price_above_vwap":
            if price <= vwap[bar_idx]:
                return False

        elif ctype == "price_below_vwap":
            if price >= vwap[bar_idx]:
                return False

        elif ctype == "price_extended_above_vwap":
            if (price - vwap[bar_idx]) < cond.get("pts", 2.0):
                return False

        elif ctype == "price_extended_below_vwap":
            if (vwap[bar_idx] - price) < cond.get("pts", 2.0):
                return False

        elif ctype == "price_above_open":
            if price <= day_open:
                return False

        elif ctype == "price_below_open":
            if price >= day_open:
                return False

        elif ctype == "price_broke_high":
            lookback = cond.get("lookback_bars", 5)
            if bar_idx < 1:
                return False
            start = max(0, bar_idx - lookback)
            prev_high = highs[start:bar_idx].max()
            if price <= prev_high:
                return False

        elif ctype == "price_broke_low":
            lookback = cond.get("lookback_bars", 5)
            if bar_idx < 1:
                return False
            start = max(0, bar_idx - lookback)
            prev_low = lows[start:bar_idx].min()
            if price >= prev_low:
                return False

        elif ctype == "gap_up":
            prev_close = meta.get("prev_close", 0)
            gap_pct = cond.get("min_pct", 0.2)
            if prev_close <= 0 or (day_open - prev_close) / prev_close * 100 < gap_pct:
                return False

        elif ctype == "gap_down":
            prev_close = meta.get("prev_close", 0)
            gap_pct = cond.get("min_pct", 0.2)
            if prev_close <= 0 or (prev_close - day_open) / prev_close * 100 < gap_pct:
                return False

        elif ctype == "first_n_minutes_up":
            n = cond.get("minutes", 15)
            if bar_idx < n:
                return False
            nth_close = closes[min(n, ctx["n"] - 1)]
            if nth_close <= opens[0]:
                return False

        elif ctype == "first_n_minutes_down":
            n = cond.get("minutes", 15)
            if bar_idx < n:
                return False
            nth_close = closes[min(n, ctx["n"] - 1)]
            if nth_close >= opens[0]:
                return False

        elif ctype == "candle_bullish":
            if closes[bar_idx] <= opens[bar_idx]:
                return False

        elif ctype == "candle_bearish":
            if closes[bar_idx] >= opens[bar_idx]:
                return False

        elif ctype == "volume_spike":
            lookback = cond.get("lookback", 20)
            mult = cond.get("multiplier", 2.0)
            if bar_idx < lookback:
                return False
            avg_vol = vols[bar_idx - lookback:bar_idx].mean()
            if avg_vol <= 0 or vols[bar_idx] < avg_vol * mult:
                return False

        elif ctype == "range_narrow":
            session_range = run_high[bar_idx] - run_low[bar_idx]
            if session_range / price * 100 > cond.get("max_range_pct", 0.5):
                return False

        elif ctype == "range_wide":
            session_range = run_high[bar_idx] - run_low[bar_idx]
            if session_range / price * 100 < cond.get("min_range_pct", 0.8):
                return False

        elif ctype == "bar_index_gte":
            if bar_idx < cond.get("value", 15):
                return False

        elif ctype == "bar_index_lte":
            if bar_idx > cond.get("value", 60):
                return False

        elif ctype == "pcr_high":
            if "pcr" not in ctx:
                return False
            if ctx["pcr"][bar_idx] < cond.get("threshold", 1.5):
                return False

        elif ctype == "pcr_low":
            if "pcr" not in ctx:
                return False
            if ctx["pcr"][bar_idx] > cond.get("threshold", 0.7):
                return False

        elif ctype == "skew_elevated":
            if "skew" not in ctx:
                return False
            if ctx["skew"][bar_idx] < cond.get("threshold", 1.2):
                return False

        elif ctype == "skew_flat":
            if "skew" not in ctx:
                return False
            if ctx["skew"][bar_idx] > cond.get("threshold", 0.9):
                return False

        elif ctype == "call_volume_heavy":
            if "call_vol_ratio" not in ctx:
                return False
            # Ratio = cum_otm_call_vol / cum_call_vol, bounded [0,1]
            if ctx["call_vol_ratio"][bar_idx] < cond.get("threshold", 0.6):
                return False

        elif ctype == "put_volume_heavy":
            if "put_vol_ratio" not in ctx:
                return False
            if ctx["put_vol_ratio"][bar_idx] < cond.get("threshold", 0.6):
                return False

        elif ctype == "always":
            pass

        else:
            return False

    return True


# ─── Batch Evaluation ────────────────────────────────────────────────────────

def evaluate_genes_on_days(
    genes: dict,
    days: list[date],
    data_dir: Path,
    chain_subdir: str = "options_5dte",
    cost: CostModel | None = None,
    preloaded_und: dict | None = None,
    preloaded_chains: dict | None = None,
    preloaded_metas: dict | None = None,
) -> list[OptionsTrade]:
    """Evaluate a gene set across multiple days.

    Returns list of OptionsTrade objects.
    """
    trades = []

    for d in days:
        # Load data (prefer preloaded)
        if preloaded_und and d in preloaded_und:
            und = preloaded_und[d]
        else:
            und = load_underlying(data_dir, d)

        if preloaded_chains and d in preloaded_chains:
            chain = preloaded_chains[d]
        else:
            chain = load_chain(data_dir, d, chain_subdir)

        if preloaded_metas and d in preloaded_metas:
            meta = preloaded_metas[d]
        else:
            meta = load_chain_meta(data_dir, d, chain_subdir)

        trade = evaluate_day(und, chain, meta, genes, cost)
        if trade is not None:
            trades.append(trade)

    return trades


# ─── Fitness Scoring ─────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    """Fitness metrics for a gene set evaluation."""
    fitness: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    avg_r: float = 0.0
    cum_r: float = 0.0
    max_dd_r: float = 0.0
    avg_pnl: float = 0.0
    cum_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0


def compute_fitness(trades: list[OptionsTrade], min_trades: int = 50) -> FitnessResult:
    """Compute fitness from a list of trades."""
    if len(trades) < min_trades:
        return FitnessResult(fitness=-1.0, n_trades=len(trades))

    pnls = [t.pnl_net for t in trades]
    rs = [t.r_multiple for t in trades]

    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n

    cum_pnl = sum(pnls)
    avg_pnl = cum_pnl / n

    cum_r = sum(rs)
    avg_r = cum_r / n

    # Sharpe (annualized using actual trading frequency)
    std_r = np.std(rs) if len(rs) > 1 else 1.0
    # Compute trades per year from actual date span
    dates = sorted(set(t.trade_date for t in trades))
    if len(dates) >= 2:
        d0 = datetime.strptime(dates[0], "%Y-%m-%d")
        d1 = datetime.strptime(dates[-1], "%Y-%m-%d")
        span_years = max((d1 - d0).days / 365.25, 1 / 365.25)
        trades_per_year = n / span_years
    else:
        trades_per_year = 252  # fallback for single-day
    sharpe = (avg_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0

    # Profit factor and per-side averages
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    gross_win = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    pf = gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
    avg_win_pnl = gross_win / len(win_pnls) if win_pnls else 0.0
    avg_loss_pnl = gross_loss / len(loss_pnls) if loss_pnls else 0.0

    # Max drawdown (in R)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    # Max drawdown (in dollars)
    cum_dollar = 0.0
    peak_dollar = 0.0
    max_dd_dollar = 0.0
    for p in pnls:
        cum_dollar += p
        peak_dollar = max(peak_dollar, cum_dollar)
        dd_dollar = peak_dollar - cum_dollar
        max_dd_dollar = max(max_dd_dollar, dd_dollar)

    # Composite fitness: balance Sharpe, PF, trade count, and win rate
    # Heavily penalize low trade counts (< 50 already filtered)
    trade_bonus = min(1.0, n / 100)  # ramp up to 100 trades
    fitness = sharpe * 0.4 + min(pf, 3.0) * 0.3 + wr * 0.2 + trade_bonus * 0.1

    # Penalize if cum_r is negative
    if cum_r < 0:
        fitness = -abs(fitness)

    return FitnessResult(
        fitness=round(fitness, 4),
        sharpe=round(sharpe, 4),
        profit_factor=round(pf, 4),
        win_rate=round(wr, 4),
        n_trades=n,
        avg_r=round(avg_r, 4),
        cum_r=round(cum_r, 4),
        max_dd_r=round(max_dd, 4),
        avg_pnl=round(avg_pnl, 2),
        cum_pnl=round(cum_pnl, 2),
        avg_win=round(avg_win_pnl, 2),
        avg_loss=round(avg_loss_pnl, 2),
        max_drawdown=round(max_dd_dollar, 2),
    )


def compute_extended_analytics(trades: list[OptionsTrade]) -> dict:
    """Compute deep analytics on a trade list for pre-live due diligence.

    Returns a dict with everything you'd want to know before risking real money.
    """
    if not trades:
        return {}

    pnls = [t.pnl_net for t in trades]
    n = len(pnls)

    # ── Streak analysis ──────────────────────────────────────────────────
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)

    # ── Single-trade extremes ────────────────────────────────────────────
    max_single_win = max(pnls) if pnls else 0
    max_single_loss = min(pnls) if pnls else 0
    median_pnl = float(np.median(pnls))

    # ── Exit reason distribution ─────────────────────────────────────────
    exit_reasons: dict[str, int] = {}
    for t in trades:
        r = t.exit_reason or "unknown"
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # ── Time-of-day P&L (entry hour buckets) ─────────────────────────────
    hourly_pnl: dict[str, dict] = {}
    for t in trades:
        try:
            hour = t.entry_time.split(" ")[-1].split(":")[0] if " " in t.entry_time else "?"
            h_key = f"{hour}:00"
        except Exception:
            h_key = "?"
        if h_key not in hourly_pnl:
            hourly_pnl[h_key] = {"n": 0, "pnl": 0.0, "wins": 0}
        hourly_pnl[h_key]["n"] += 1
        hourly_pnl[h_key]["pnl"] = round(hourly_pnl[h_key]["pnl"] + t.pnl_net, 2)
        if t.pnl_net > 0:
            hourly_pnl[h_key]["wins"] += 1

    # ── Day-of-week P&L ──────────────────────────────────────────────────
    dow_pnl: dict[str, dict] = {}
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for t in trades:
        try:
            d = datetime.strptime(t.trade_date, "%Y-%m-%d")
            dow = dow_names[d.weekday()]
        except Exception:
            dow = "?"
        if dow not in dow_pnl:
            dow_pnl[dow] = {"n": 0, "pnl": 0.0, "wins": 0}
        dow_pnl[dow]["n"] += 1
        dow_pnl[dow]["pnl"] = round(dow_pnl[dow]["pnl"] + t.pnl_net, 2)
        if t.pnl_net > 0:
            dow_pnl[dow]["wins"] += 1

    # ── Monthly P&L ──────────────────────────────────────────────────────
    monthly_pnl: dict[str, dict] = {}
    for t in trades:
        try:
            ym = t.trade_date[:7]  # "YYYY-MM"
        except Exception:
            ym = "?"
        if ym not in monthly_pnl:
            monthly_pnl[ym] = {"n": 0, "pnl": 0.0, "wins": 0}
        monthly_pnl[ym]["n"] += 1
        monthly_pnl[ym]["pnl"] = round(monthly_pnl[ym]["pnl"] + t.pnl_net, 2)
        if t.pnl_net > 0:
            monthly_pnl[ym]["wins"] += 1

    # ── Equity curve stats ───────────────────────────────────────────────
    equity = []
    running = 0.0
    for p in pnls:
        running += p
        equity.append(round(running, 2))
    peak_equity = max(equity) if equity else 0
    trough_from_peak = 0.0
    running_peak = 0.0
    for e in equity:
        running_peak = max(running_peak, e)
        trough_from_peak = max(trough_from_peak, running_peak - e)

    # Time underwater: how many trades from DD start to recovery
    underwater_trades = 0
    max_underwater = 0
    cur_underwater = 0
    running_peak = 0.0
    running_eq = 0.0
    for p in pnls:
        running_eq += p
        if running_eq >= running_peak:
            running_peak = running_eq
            max_underwater = max(max_underwater, cur_underwater)
            cur_underwater = 0
        else:
            cur_underwater += 1
    max_underwater = max(max_underwater, cur_underwater)

    # ── Monte Carlo bootstrap (10K resamples) ────────────────────────────
    rng = np.random.default_rng(42)
    pnl_arr = np.array(pnls)
    n_sims = 10_000
    sim_finals = np.empty(n_sims)
    sim_max_dds = np.empty(n_sims)
    for i in range(n_sims):
        shuffled = rng.choice(pnl_arr, size=n, replace=True)
        cum = np.cumsum(shuffled)
        sim_finals[i] = cum[-1]
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        sim_max_dds[i] = dd.max()

    mc = {
        "cum_pnl_p5": round(float(np.percentile(sim_finals, 5)), 2),
        "cum_pnl_p25": round(float(np.percentile(sim_finals, 25)), 2),
        "cum_pnl_p50": round(float(np.percentile(sim_finals, 50)), 2),
        "cum_pnl_p75": round(float(np.percentile(sim_finals, 75)), 2),
        "cum_pnl_p95": round(float(np.percentile(sim_finals, 95)), 2),
        "max_dd_p50": round(float(np.percentile(sim_max_dds, 50)), 2),
        "max_dd_p75": round(float(np.percentile(sim_max_dds, 75)), 2),
        "max_dd_p95": round(float(np.percentile(sim_max_dds, 95)), 2),
        "prob_positive": round(float((sim_finals > 0).mean()), 4),
    }

    # ── Daily P&L for correlation analysis ───────────────────────────────
    daily_pnl: dict[str, float] = {}
    for t in trades:
        d = t.trade_date
        daily_pnl[d] = round(daily_pnl.get(d, 0.0) + t.pnl_net, 2)

    # ── Return everything ────────────────────────────────────────────────
    return {
        "streaks": {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        },
        "extremes": {
            "max_single_win": round(max_single_win, 2),
            "max_single_loss": round(max_single_loss, 2),
            "median_pnl": round(median_pnl, 2),
        },
        "exit_reasons": exit_reasons,
        "hourly_pnl": hourly_pnl,
        "dow_pnl": dow_pnl,
        "monthly_pnl": monthly_pnl,
        "equity_curve": {
            "peak": round(peak_equity, 2),
            "max_dd_from_peak": round(trough_from_peak, 2),
            "max_trades_underwater": max_underwater,
            "final": equity[-1] if equity else 0,
        },
        "monte_carlo": mc,
        "daily_pnl": daily_pnl,
    }
