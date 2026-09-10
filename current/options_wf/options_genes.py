"""
Options Gene Catalog
====================
Defines the combinatorial space for SPY options strategies.

Entry genes use SPY underlying price action (same signals as MES, adapted
for 1m SPY bars instead of 15m MES bars).

Exit genes are options-native: premium-based TP/SL, time stops, EOD exits.

Sampling produces gene dicts ready for options_backtest.evaluate_day().
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY CONDITION CATALOG
# ═════════════════════════════════════════════════════════════════════════════

# Each entry is (condition_dict, implied_direction, archetype)
ENTRY_CONDITIONS = [
    # ── VWAP conditions ──
    ({"type": "price_above_vwap"}, "long", "vwap_trend"),
    ({"type": "price_below_vwap"}, "short", "vwap_trend"),
    ({"type": "price_extended_above_vwap", "pts": 1.0}, "short", "vwap_fade"),
    ({"type": "price_extended_above_vwap", "pts": 2.0}, "short", "vwap_fade"),
    ({"type": "price_extended_above_vwap", "pts": 3.0}, "short", "vwap_fade"),
    ({"type": "price_extended_below_vwap", "pts": 1.0}, "long", "vwap_fade"),
    ({"type": "price_extended_below_vwap", "pts": 2.0}, "long", "vwap_fade"),
    ({"type": "price_extended_below_vwap", "pts": 3.0}, "long", "vwap_fade"),

    # ── Breakout ──
    ({"type": "price_broke_high", "lookback_bars": 5}, "long", "breakout"),
    ({"type": "price_broke_high", "lookback_bars": 15}, "long", "breakout"),
    ({"type": "price_broke_high", "lookback_bars": 30}, "long", "breakout"),
    ({"type": "price_broke_low", "lookback_bars": 5}, "short", "breakout"),
    ({"type": "price_broke_low", "lookback_bars": 15}, "short", "breakout"),
    ({"type": "price_broke_low", "lookback_bars": 30}, "short", "breakout"),

    # ── Day open reference ──
    ({"type": "price_above_open"}, "long", "open_drive"),
    ({"type": "price_below_open"}, "short", "open_drive"),

    # ── Gap conditions ──
    ({"type": "gap_up", "min_pct": 0.2}, "long", "gap_continuation"),
    ({"type": "gap_up", "min_pct": 0.5}, "long", "gap_continuation"),
    ({"type": "gap_up", "min_pct": 0.2}, "short", "gap_fade"),
    ({"type": "gap_down", "min_pct": 0.2}, "short", "gap_continuation"),
    ({"type": "gap_down", "min_pct": 0.5}, "short", "gap_continuation"),
    ({"type": "gap_down", "min_pct": 0.2}, "long", "gap_fade"),

    # ── First N minutes trend ──
    ({"type": "first_n_minutes_up", "minutes": 15}, "long", "opening_trend"),
    ({"type": "first_n_minutes_up", "minutes": 30}, "long", "opening_trend"),
    ({"type": "first_n_minutes_down", "minutes": 15}, "short", "opening_trend"),
    ({"type": "first_n_minutes_down", "minutes": 30}, "short", "opening_trend"),

    # ── Candle patterns ──
    ({"type": "candle_bullish"}, "long", "candle_signal"),
    ({"type": "candle_bearish"}, "short", "candle_signal"),

    # ── Volume ──
    ({"type": "volume_spike", "lookback": 20, "multiplier": 2.0}, "either", "volume"),
    ({"type": "volume_spike", "lookback": 20, "multiplier": 3.0}, "either", "volume"),

    # ── Range context ──
    ({"type": "range_narrow", "max_range_pct": 0.3}, "either", "range_context"),
    ({"type": "range_narrow", "max_range_pct": 0.5}, "either", "range_context"),
    ({"type": "range_wide", "min_range_pct": 0.8}, "either", "range_context"),
    ({"type": "range_wide", "min_range_pct": 1.0}, "either", "range_context"),

    # ── Time filters (as entry conditions) ──
    ({"type": "bar_index_gte", "value": 15}, "either", "time_filter"),   # 15+ min after open
    ({"type": "bar_index_gte", "value": 30}, "either", "time_filter"),   # 30+ min
    ({"type": "bar_index_lte", "value": 60}, "either", "time_filter"),   # first hour only
    ({"type": "bar_index_lte", "value": 120}, "either", "time_filter"),  # first 2 hours

    # ── Options-derived signals (from chain data) ──
    # Put/call volume ratio: high → bearish flow, low → bullish flow
    ({"type": "pcr_high", "threshold": 1.5}, "short", "options_flow"),
    ({"type": "pcr_high", "threshold": 2.0}, "short", "options_flow"),
    ({"type": "pcr_low", "threshold": 0.7}, "long", "options_flow"),
    ({"type": "pcr_low", "threshold": 0.5}, "long", "options_flow"),
    # Skew: ATM put premium vs call premium ratio
    ({"type": "skew_elevated", "threshold": 1.2}, "short", "vol_skew"),
    ({"type": "skew_elevated", "threshold": 1.5}, "short", "vol_skew"),
    ({"type": "skew_flat", "threshold": 0.9}, "long", "vol_skew"),
    # Call volume concentration at ATM+ (bullish positioning)
    # Ratio = cum_otm_call_vol / cum_call_vol, bounded [0,1]
    ({"type": "call_volume_heavy", "threshold": 0.6}, "long", "strike_flow"),
    # Put volume concentration at ATM- (bearish positioning)
    ({"type": "put_volume_heavy", "threshold": 0.6}, "short", "strike_flow"),

    # ── Unconditional (filter-only strategies) ──
    ({"type": "always"}, "either", "unconditional"),
]


# ═════════════════════════════════════════════════════════════════════════════
# REGIME FILTER CATALOG
# ═════════════════════════════════════════════════════════════════════════════
# Day-level filters evaluated once before bar scanning.
# These decide whether to trade AT ALL on a given day.

REGIME_FILTERS = [
    # ── IV proxy from ATM premium (higher IV → more expensive options) ──
    # ATM premium as % of underlying price — proxy for implied vol
    {"type": "iv_proxy_high", "min_pct": 0.8},   # high IV day (>0.8% of SPY)
    {"type": "iv_proxy_high", "min_pct": 1.0},   # very high IV
    {"type": "iv_proxy_low", "max_pct": 0.6},    # low IV day (<0.6%)
    {"type": "iv_proxy_low", "max_pct": 0.8},    # moderate-low IV

    # ── Prior day range (trending vs choppy context) ──
    {"type": "prev_range_wide", "min_pct": 0.8},   # prior day moved >0.8%
    {"type": "prev_range_wide", "min_pct": 1.2},   # prior day moved >1.2%
    {"type": "prev_range_narrow", "max_pct": 0.5},  # prior day was tight (<0.5%)
    {"type": "prev_range_narrow", "max_pct": 0.8},  # prior day was moderate

    # ── Gap alignment with direction ──
    {"type": "gap_aligns", "min_pct": 0.1},   # gap in same direction as trade
    {"type": "gap_aligns", "min_pct": 0.3},   # strong gap alignment
    {"type": "gap_opposes", "min_pct": 0.1},  # gap against trade direction (fade)
    {"type": "no_gap", "max_pct": 0.1},       # flat open, no gap

    # ── Opening volume vs recent average ──
    {"type": "open_vol_high", "multiplier": 1.5},  # first 5min vol > 1.5x avg
    {"type": "open_vol_high", "multiplier": 2.0},  # first 5min vol > 2x avg
    {"type": "open_vol_low", "max_mult": 0.7},     # quiet open (<0.7x avg)

    # ── ATM skew at open (put premium vs call premium) ──
    {"type": "open_skew_high", "threshold": 1.2},  # puts expensive vs calls
    {"type": "open_skew_low", "threshold": 0.8},   # calls expensive vs puts
]


# ═════════════════════════════════════════════════════════════════════════════
# TRADE TYPE SPACE
# ═════════════════════════════════════════════════════════════════════════════

# Trade types: spreads of various widths + naked/ITM options
TRADE_TYPES = [
    # Debit spreads (long ATM, short OTM) — pay debit, profit if spread widens
    {"kind": "debit_spread", "width": 1},
    {"kind": "debit_spread", "width": 2},
    {"kind": "debit_spread", "width": 3},
    # Credit spreads (short ATM, long OTM) — collect credit, profit if spread narrows/expires
    {"kind": "credit_spread", "width": 1},
    {"kind": "credit_spread", "width": 2},
    {"kind": "credit_spread", "width": 3},
    # Naked ATM options (long single leg)
    {"kind": "naked", "moneyness": "atm"},
    # ITM options (1-3 strikes ITM)
    {"kind": "naked", "moneyness": "itm1"},
    {"kind": "naked", "moneyness": "itm2"},
    {"kind": "naked", "moneyness": "itm3"},
]


# ═════════════════════════════════════════════════════════════════════════════
# EXIT GENE SPACE
# ═════════════════════════════════════════════════════════════════════════════

# Premium-based TP as % of max gain (spreads) or entry premium (naked)
TP_PCTS = [0.30, 0.50, 0.60, 0.80, 1.00]

# Premium-based SL as % of entry debit/premium lost
SL_PCTS = [0.30, 0.50, 0.70, 0.80, 1.00]

# Trailing stop as % below premium high watermark (naked options only)
# None = no trailing stop (use fixed TP/SL only)
TSL_PCTS = [None, 0.10, 0.15, 0.20, 0.30, 0.50]

# Max hold time in minutes
HOLD_MINUTES = [30, 60, 90, 120, 180, 240, 390]

# Multi-day hold times (in minutes, across trading sessions)
# 390 = 1 RTH session, 780 = 2 sessions, 1170 = 3, 1560 = 4, 1950 = 5
HOLD_MINUTES_MULTIDAY = [390, 780, 1170, 1560, 1950]

# Time windows
TIME_WINDOWS = [
    {"start": "09:45", "end": "11:00"},   # morning session
    {"start": "09:45", "end": "12:00"},   # extended morning
    {"start": "10:00", "end": "14:00"},   # mid-day
    {"start": "09:45", "end": "14:30"},   # most of day
    {"start": "09:45", "end": "15:30"},   # full day
    {"start": "10:30", "end": "14:00"},   # avoid first hour
    {"start": "11:00", "end": "15:00"},   # late session
]


# ═════════════════════════════════════════════════════════════════════════════
# GENE DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionsGenes:
    """A complete options strategy defined by its genes."""
    direction: str                           # "long" or "short"
    trade_type: dict = field(default_factory=lambda: {"kind": "debit_spread", "width": 1})
    entry_conditions: list[dict] = field(default_factory=list)  # 1-3 conditions
    filter_condition: Optional[dict] = None  # optional filter
    regime_filters: list[dict] = field(default_factory=list)  # day-level filters
    time_window: dict = field(default_factory=lambda: {"start": "09:45", "end": "14:00"})
    exit_rules: dict = field(default_factory=lambda: {
        "tp_pct": 0.50, "sl_pct": 0.50, "hold_minutes": 120, "eod_exit": True
    })
    archetype: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def describe(self) -> str:
        """Human-readable description."""
        tt = self.trade_type
        kind = tt.get("kind", "debit_spread")
        if kind == "debit_spread":
            tt_str = f"${tt.get('width', 1)}debit"
        elif kind == "credit_spread":
            tt_str = f"${tt.get('width', 1)}credit"
        else:
            tt_str = f"naked_{tt.get('moneyness', 'atm')}"
        parts = [f"{self.direction.upper()} {tt_str}"]
        for c in self.entry_conditions:
            parts.append(c.get("type", "?"))
        if self.filter_condition:
            parts.append(f"[{self.filter_condition.get('type', '?')}]")
        if self.regime_filters:
            rf_types = "+".join(r.get("type", "?") for r in self.regime_filters)
            parts.append(f"{{{rf_types}}}")
        exit_r = self.exit_rules
        tsl = exit_r.get("tsl_pct")
        if tsl is not None:
            exit_str = (f"TSL{int(tsl*100)}% "
                        f"SL{int(exit_r.get('sl_pct', 0.5)*100)}% "
                        f"{exit_r.get('hold_minutes', 120)}m")
        else:
            exit_str = (f"TP{int(exit_r.get('tp_pct', 0.5)*100)}% "
                        f"SL{int(exit_r.get('sl_pct', 0.5)*100)}% "
                        f"{exit_r.get('hold_minutes', 120)}m")
        parts.append(exit_str)
        return " | ".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def sample_candidates(
    n: int = 5000,
    seed: int = 42,
    allowed_trade_types: list[dict] | None = None,
    allow_overnight: bool = False,
) -> list[OptionsGenes]:
    """Sample n unique strategy candidates from the combinatorial space.

    Args:
        allowed_trade_types: if set, only sample trade types from this list.
            E.g., [{"kind": "debit_spread", "width": 3}] for spread-only mode.
        allow_overnight: if True, include multi-day hold strategies with eod_exit=False.
    """
    rng = random.Random(seed)
    candidates = []
    seen = set()

    max_attempts = n * 10

    for _ in range(max_attempts):
        if len(candidates) >= n:
            break

        genes = _sample_one(rng, allowed_trade_types=allowed_trade_types,
                            allow_overnight=allow_overnight)
        key = _genes_key(genes)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(genes)

    return candidates


def _sample_one(rng: random.Random, allowed_trade_types: list[dict] | None = None,
                allow_overnight: bool = False) -> OptionsGenes:
    """Sample a single random strategy."""
    # Pick 1-2 entry conditions
    n_conditions = rng.choice([1, 1, 1, 2, 2])

    # Sample primary entry condition
    primary = rng.choice(ENTRY_CONDITIONS)
    primary_cond, primary_dir, archetype = primary

    # Direction: from primary condition (or random if "either")
    if primary_dir == "either":
        direction = rng.choice(["long", "short"])
    else:
        direction = primary_dir

    conditions = [primary_cond]

    # Optionally add a second condition (must be direction-compatible)
    if n_conditions >= 2:
        compatible = [
            (c, d, a) for c, d, a in ENTRY_CONDITIONS
            if (d == "either" or d == direction) and c != primary_cond
        ]
        if compatible:
            secondary_cond, _, _ = rng.choice(compatible)
            conditions.append(secondary_cond)

    # Optional filter condition (30% chance)
    filter_cond = None
    if rng.random() < 0.30:
        filter_options = [
            c for c, d, a in ENTRY_CONDITIONS
            if a in ("range_context", "time_filter", "volume")
            and c not in conditions
        ]
        if filter_options:
            filter_cond = rng.choice(filter_options)

    # Regime filters (0-2 day-level filters, 50% chance of at least one)
    regime_filters = []
    if rng.random() < 0.50:
        n_regime = rng.choice([1, 1, 2])
        regime_pool = list(REGIME_FILTERS)
        rng.shuffle(regime_pool)
        for rf in regime_pool[:n_regime]:
            regime_filters.append(rf)

    # Trade type
    trade_pool = allowed_trade_types if allowed_trade_types else TRADE_TYPES
    trade_type = rng.choice(trade_pool)

    # Time window
    tw = rng.choice(TIME_WINDOWS)

    # Exit rules
    tsl_pct = rng.choice(TSL_PCTS)

    # Multi-day hold: 40% of overnight candidates use multi-day holds
    if allow_overnight and rng.random() < 0.40:
        hold_mins = rng.choice(HOLD_MINUTES_MULTIDAY)
        eod_exit = False
    else:
        hold_mins = rng.choice(HOLD_MINUTES)
        eod_exit = True

    exit_rules = {
        "tp_pct": rng.choice(TP_PCTS),
        "sl_pct": rng.choice(SL_PCTS),
        "hold_minutes": hold_mins,
        "eod_exit": eod_exit,
        "tsl_pct": tsl_pct,
    }
    # TSL only makes sense for naked options — disable for spreads
    if trade_type.get("kind") != "naked":
        exit_rules["tsl_pct"] = None

    return OptionsGenes(
        direction=direction,
        trade_type=trade_type,
        entry_conditions=conditions,
        filter_condition=filter_cond,
        regime_filters=regime_filters,
        time_window=tw,
        exit_rules=exit_rules,
        archetype=archetype,
    )


def _genes_key(genes: OptionsGenes) -> str:
    """Unique key for deduplication."""
    conds = tuple(sorted(json.dumps(c, sort_keys=True) for c in genes.entry_conditions))
    filt = json.dumps(genes.filter_condition, sort_keys=True) if genes.filter_condition else ""
    rf_k = tuple(sorted(json.dumps(r, sort_keys=True) for r in genes.regime_filters)) if genes.regime_filters else ()
    tt_k = json.dumps(genes.trade_type, sort_keys=True)
    exit_k = f"{genes.exit_rules['tp_pct']}_{genes.exit_rules['sl_pct']}_{genes.exit_rules['hold_minutes']}_{genes.exit_rules.get('tsl_pct')}_{genes.exit_rules.get('eod_exit', True)}"
    tw_k = f"{genes.time_window['start']}_{genes.time_window['end']}"
    return f"{genes.direction}|{tt_k}|{conds}|{filt}|{rf_k}|{exit_k}|{tw_k}"


import json


def describe_genes(genes: OptionsGenes) -> str:
    """Human-readable summary of gene set."""
    return genes.describe()
