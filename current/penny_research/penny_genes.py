"""
Penny Gapper Gene Catalog & Combinator
=======================================
Defines the gene space for penny gapper intraday strategies.
Genes are modular building blocks assembled into candidate strategies
that the backtester evaluates across pooled ticker-day data.

Gene categories:
  - Entry type: the primary trigger condition (ramp, VWAP, ORB, pullback, etc.)
  - Entry filters: secondary confirmation layers (volume, momentum, spread, timing)
  - Exit: TP, SL, trailing, time, partial
  - Direction: long (continuation) or short (fade)

The entry type system is extensible — each type has its own parameter
sub-space, and the combinator samples across all types uniformly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY TYPE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
# Each entry type is a dict with:
#   "type": str — unique identifier used in backtester dispatch
#   "params": dict — type-specific parameters (sampled from ranges)
#   "archetype": str — category for balance sampling
#   "description": str — human-readable

ENTRY_TYPE_CATALOG = {

    # ── RAMP: consecutive green bars from open ──
    "ramp": {
        "archetype": "momentum",
        "description": "Opening ramp — consecutive green bars with volume",
        "params": {
            "min_ramp_bars": [1, 2, 3, 4, 5],
            "min_ramp_body_pct": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
            "max_wick_ratio": [0.3, 0.5, 0.7, 1.0, 999],  # 999 = no filter
        },
    },

    # ── ORB: Opening Range Breakout ──
    "orb": {
        "archetype": "breakout",
        "description": "Break above/below the N-minute opening range",
        "params": {
            "orb_minutes": [1, 2, 3, 5, 10, 15],  # opening range window
            "orb_buffer_pct": [0, 0.1, 0.25, 0.5],  # % above/below range to trigger
        },
    },

    # ── VWAP: entries relative to session VWAP ──
    "vwap_cross": {
        "archetype": "trend",
        "description": "Price crosses above/below developing session VWAP",
        "params": {
            "cross_direction": ["above", "below"],  # above = bullish, below = bearish
            "min_distance_pct": [0, 0.5, 1.0, 2.0],  # how far past VWAP before trigger
        },
    },
    "vwap_bounce": {
        "archetype": "mean_reversion",
        "description": "Price pulls back to VWAP and bounces (support/resistance)",
        "params": {
            "bounce_direction": ["long_from_below", "short_from_above"],
            "proximity_pct": [0.25, 0.5, 1.0, 1.5],  # how close to VWAP to consider 'at'
            "min_bounce_bars": [1, 2, 3],  # bars of reversal from VWAP
        },
    },

    # ── EMA: moving average conditions ──
    "ema_cross": {
        "archetype": "trend",
        "description": "Fast EMA crosses slow EMA",
        "params": {
            "fast_period": [3, 5, 8],
            "slow_period": [13, 21, 34],
        },
    },
    "ema_trend": {
        "archetype": "trend",
        "description": "Price above/below EMA — trend filter",
        "params": {
            "ema_period": [5, 8, 13, 21, 34],
            "price_vs_ema": ["above", "below"],
            "min_distance_pct": [0, 0.5, 1.0, 2.0],
        },
    },

    # ── PULLBACK: entry after initial move + retracement ──
    "pullback": {
        "archetype": "mean_reversion",
        "description": "Enter on pullback after initial thrust — fib-style retracement",
        "params": {
            "min_initial_move_pct": [2.0, 3.0, 5.0, 8.0, 10.0],  # min thrust from open
            "pullback_depth_pct": [20, 30, 40, 50, 62],  # % retracement of initial move
            "pullback_tolerance_pct": [5, 10, 15],  # how close to target level
        },
    },

    # ── MACD: momentum divergence ──
    "macd_cross": {
        "archetype": "momentum",
        "description": "MACD line crosses signal line",
        "params": {
            "fast_period": [5, 8, 12],
            "slow_period": [13, 21, 26],
            "signal_period": [5, 9],
            "cross_type": ["bullish", "bearish"],  # MACD crosses above/below signal
        },
    },

    # ── RSI: overbought/oversold ──
    "rsi_extreme": {
        "archetype": "mean_reversion",
        "description": "RSI hits extreme level — potential reversal",
        "params": {
            "rsi_period": [5, 7, 14],
            "oversold_level": [20, 25, 30],
            "overbought_level": [70, 75, 80],
            "entry_on": ["oversold", "overbought"],  # which extreme triggers
        },
    },

    # ── CANDLE PATTERN: specific bar patterns ──
    "candle_pattern": {
        "archetype": "price_action",
        "description": "Specific candlestick pattern recognition",
        "params": {
            "pattern": [
                "engulfing_bull",    # green bar engulfs prior red bar
                "engulfing_bear",    # red bar engulfs prior green bar
                "hammer",            # long lower wick, small body (reversal)
                "shooting_star",     # long upper wick, small body (reversal)
                "doji_reversal",     # doji after trend (indecision → reversal)
                "three_soldiers",    # 3 consecutive green bars (strong bull)
                "three_crows",       # 3 consecutive red bars (strong bear)
                "inside_bar_break",  # inside bar followed by breakout
            ],
            "min_body_pct": [0.3, 0.5, 1.0],  # min body size as % of price
        },
    },

    # ── VOLUME CLIMAX: extreme volume spike ──
    "volume_climax": {
        "archetype": "momentum",
        "description": "Volume spikes to extreme — momentum or exhaustion",
        "params": {
            "vol_multiple": [3.0, 5.0, 8.0, 10.0, 20.0],  # vs trailing avg
            "lookback_bars": [5, 10, 20],
            "interpretation": ["momentum", "exhaustion"],  # trade with or against spike
        },
    },

    # ── RANGE BREAKOUT: break out of consolidation ──
    "range_breakout": {
        "archetype": "breakout",
        "description": "Break above/below N-bar high/low consolidation range",
        "params": {
            "lookback_bars": [5, 10, 15, 20, 30],
            "break_direction": ["high", "low"],
            "min_consolidation_bars": [3, 5, 8],  # min bars of sideways before break
            "max_range_pct": [2.0, 3.0, 5.0, 8.0],  # max range width to qualify as consolidation
        },
    },

    # ── GAP FILL: fade back toward previous close ──
    "gap_fill_entry": {
        "archetype": "mean_reversion",
        "description": "Enter expecting gap to fill back toward prev close",
        "params": {
            "min_gap_pct": [30, 50, 75, 100],  # min gap to trigger fade
            "wait_bars": [1, 3, 5, 10],  # bars to wait after open before fading
            "require_red_bar": [True, False],  # need a red bar to confirm reversal
        },
    },

    # ── HOD BREAK: new high of day breakout ──
    "hod_break": {
        "archetype": "breakout",
        "description": "Price makes new high of day — momentum continuation",
        "params": {
            "min_hod_age_bars": [5, 10, 15, 30],  # HOD must be at least N bars old
            "break_margin_pct": [0, 0.1, 0.25, 0.5],  # must break HOD by this %
        },
    },

    # ── FIRST RED: first red bar after sustained green run ──
    "first_red_fade": {
        "archetype": "mean_reversion",
        "description": "First red bar after N green bars — fade the exhaustion",
        "params": {
            "min_green_run": [3, 4, 5, 7, 10],  # how many green bars before red
            "min_run_pct": [3.0, 5.0, 8.0, 10.0],  # min % move during green run
        },
    },

    # ── PRICE LEVEL: key price level interactions ──
    "price_level": {
        "archetype": "price_action",
        "description": "Interaction with key price levels (whole numbers, prev close, open)",
        "params": {
            "level_type": [
                "whole_dollar",      # price near $1, $2, $3 etc.
                "half_dollar",       # price near $0.50, $1.50 etc.
                "prev_close",        # interaction with previous close
                "day_open",          # interaction with day's opening price
                "premarket_high",    # break above premarket high (approximated)
            ],
            "interaction": ["break_above", "break_below", "bounce_from"],
            "proximity_pct": [0.25, 0.5, 1.0],
        },
    },

    # ── MOMENTUM ACCELERATION: rate of change increasing ──
    "momentum_accel": {
        "archetype": "momentum",
        "description": "Price momentum accelerating — each N-bar return > previous N-bar",
        "params": {
            "window_bars": [3, 5, 8],
            "min_windows": [2, 3],  # how many consecutive windows must show acceleration
            "min_accel_pct": [0.5, 1.0, 2.0],  # min increase in return between windows
        },
    },

    # ── RELATIVE VOLUME: sustained high volume throughout ──
    "sustained_volume": {
        "archetype": "momentum",
        "description": "Volume remains elevated — not just a spike but sustained interest",
        "params": {
            "min_avg_vol_ratio": [2.0, 3.0, 5.0, 10.0],  # vs prev day avg bar vol
            "min_sustained_bars": [5, 10, 15, 20],  # consecutive bars above threshold
        },
    },
}

# Archetypes for balance sampling
ARCHETYPES = ["momentum", "breakout", "mean_reversion", "trend", "price_action"]


# ═════════════════════════════════════════════════════════════════════════════
# GENE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PennyGenes:
    """Complete gene set for a penny gapper strategy candidate."""

    # ── Entry: Primary trigger ──
    entry_type: str = "ramp"         # key from ENTRY_TYPE_CATALOG
    entry_params: dict = field(default_factory=dict)  # type-specific params

    # ── Entry: Filters (applied ON TOP of primary trigger) ──
    # Volume filter
    min_vol_spike: float = 2.0       # min volume spike at entry (vs baseline)
    require_vol_accel: bool = False   # volume must be increasing

    # Timing filter
    entry_delay_bars: int = 0        # bars to wait after trigger
    earliest_entry_min: int = 0      # minutes after 9:30
    latest_entry_min: int = 60       # cutoff

    # Momentum filter
    min_first_bar_ret_pct: float = 0  # min first bar return %
    max_spread_pct: float = 5.0       # max bar range % (spread/liquidity proxy)

    # Confirmation filter (optional second condition)
    confirmation_type: str = "none"  # "none", "volume_above", "ema_aligned", "green_bar"
    confirmation_params: dict = field(default_factory=dict)

    # ── Exit: Take profit ──
    tp_r: float = 2.0

    # ── Exit: Stop loss ──
    sl_type: str = "ramp_low"        # "ramp_low", "fixed_pct", "atr_mult", "swing_low"
    sl_fixed_pct: float = 5.0
    sl_buffer_pct: float = 0.5
    sl_atr_mult: float = 2.0
    sl_atr_period: int = 14

    # ── Exit: Trailing stop ──
    trail_type: str = "none"         # "none", "fixed_pct", "bar_low", "ema_trail"
    trail_activation_r: float = 1.0
    trail_distance_pct: float = 0.05
    trail_ema_period: int = 8

    # ── Exit: Time ──
    max_hold_minutes: int = 120
    eod_exit: bool = True

    # ── Exit: Partial ──
    partial_exit: bool = False
    partial_pct: float = 0.33
    partial_target_r: float = 1.0

    # ── Direction ──
    direction: str = "long"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PennyGenes":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ═════════════════════════════════════════════════════════════════════════════
# FILTER / CONFIRMATION GENE RANGES
# ═════════════════════════════════════════════════════════════════════════════

FILTER_RANGES = {
    # Volume
    "min_vol_spike": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
    "require_vol_accel": [True, False],
    # Timing
    "entry_delay_bars": [0, 1, 2, 3],
    "earliest_entry_min": [0, 5, 10, 15, 30],
    "latest_entry_min": [15, 30, 60, 120, 240, 390],
    # Momentum
    "min_first_bar_ret_pct": [0, 0.5, 1.0, 2.0, 3.0, 5.0],
    "max_spread_pct": [0.5, 1.0, 2.0, 3.0, 5.0, 999],
}

CONFIRMATION_TYPES = {
    "none": {},
    "volume_above": {
        "vol_threshold": [2.0, 3.0, 5.0, 10.0],
    },
    "ema_aligned": {
        "ema_period": [5, 8, 13, 21],
        "alignment": ["price_above", "price_below"],
    },
    "green_bar": {},  # just requires the trigger bar to be green
    "red_bar": {},    # just requires the trigger bar to be red
    "volume_declining": {
        "decline_bars": [3, 5, 8],
    },
    "higher_lows": {
        "lookback_bars": [3, 5, 8],
    },
    "lower_highs": {
        "lookback_bars": [3, 5, 8],
    },
}

EXIT_RANGES = {
    "tp_r": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0],
    "sl_type": ["ramp_low", "fixed_pct", "atr_mult", "swing_low"],
    "sl_fixed_pct": [2.0, 3.0, 5.0, 8.0, 10.0, 15.0],
    "sl_buffer_pct": [0.0, 0.25, 0.5, 1.0, 2.0],
    "sl_atr_mult": [1.0, 1.5, 2.0, 3.0],
    "sl_atr_period": [5, 10, 14],
    "trail_type": ["none", "fixed_pct", "bar_low", "ema_trail"],
    "trail_activation_r": [0.25, 0.5, 1.0, 1.5, 2.0],
    "trail_distance_pct": [0.02, 0.03, 0.05, 0.08, 0.12],
    "trail_ema_period": [5, 8, 13],
    "max_hold_minutes": [15, 30, 60, 120, 240, 390],
    "partial_exit": [True, False],
    "partial_pct": [0.25, 0.33, 0.50],
    "partial_target_r": [0.5, 1.0, 1.5, 2.0],
    "direction": ["long", "short"],
}


# ═════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def _sample_entry_type_params(entry_type: str, rng: random.Random) -> dict:
    """Sample parameters for a specific entry type."""
    if entry_type not in ENTRY_TYPE_CATALOG:
        return {}
    param_ranges = ENTRY_TYPE_CATALOG[entry_type]["params"]
    return {k: rng.choice(v) for k, v in param_ranges.items()}


def _sample_confirmation(rng: random.Random) -> tuple[str, dict]:
    """Sample a confirmation type and its params."""
    # 50% chance of no confirmation
    if rng.random() < 0.50:
        return "none", {}
    conf_type = rng.choice([k for k in CONFIRMATION_TYPES if k != "none"])
    params = {k: rng.choice(v) for k, v in CONFIRMATION_TYPES[conf_type].items()}
    return conf_type, params


def _normalize_direction(entry_type: str, entry_params: dict, direction: str) -> str:
    """Align direction with entry type / params to prevent mismatches."""
    if entry_type == "gap_fill_entry":
        return "short"
    elif entry_type == "first_red_fade":
        return "short"
    elif entry_type == "hod_break":
        return "long"

    if entry_type == "vwap_cross":
        if entry_params.get("cross_direction") == "above":
            return "long"
        elif entry_params.get("cross_direction") == "below":
            return "short"

    if entry_type == "macd_cross":
        if entry_params.get("cross_type") == "bullish":
            return "long"
        elif entry_params.get("cross_type") == "bearish":
            return "short"

    if entry_type == "rsi_extreme":
        if entry_params.get("entry_on") == "oversold":
            return "long"
        elif entry_params.get("entry_on") == "overbought":
            return "short"

    if entry_type == "candle_pattern":
        pattern = entry_params.get("pattern", "")
        if pattern in ("engulfing_bull", "hammer", "three_soldiers"):
            return "long"
        elif pattern in ("engulfing_bear", "shooting_star", "three_crows"):
            return "short"

    if entry_type == "range_breakout":
        if entry_params.get("break_direction") == "high":
            return "long"
        elif entry_params.get("break_direction") == "low":
            return "short"

    if entry_type == "vwap_bounce":
        if entry_params.get("bounce_direction") == "long_from_below":
            return "long"
        elif entry_params.get("bounce_direction") == "short_from_above":
            return "short"

    return direction


def sample_random_genes(rng: random.Random | None = None) -> PennyGenes:
    """Sample a single random gene configuration from the full space."""
    if rng is None:
        rng = random.Random()

    # ── Sample entry type ──
    entry_type = rng.choice(list(ENTRY_TYPE_CATALOG.keys()))
    entry_params = _sample_entry_type_params(entry_type, rng)

    # ── Sample filters ──
    filters = {k: rng.choice(v) for k, v in FILTER_RANGES.items()}

    # ── Sample confirmation ──
    conf_type, conf_params = _sample_confirmation(rng)

    # ── Sample exits ──
    exits = {k: rng.choice(v) for k, v in EXIT_RANGES.items()}

    # ── Apply conditional logic ──

    # Trail: if none, zero out trail params
    if exits["trail_type"] == "none":
        exits["trail_activation_r"] = 0
        exits["trail_distance_pct"] = 0
        exits["trail_ema_period"] = 0

    # Partial: if off, zero out
    if not exits["partial_exit"]:
        exits["partial_pct"] = 0
        exits["partial_target_r"] = 0

    # SL type conditionals
    if exits["sl_type"] == "ramp_low":
        exits["sl_fixed_pct"] = 0
        exits["sl_atr_mult"] = 0
    elif exits["sl_type"] == "fixed_pct":
        exits["sl_atr_mult"] = 0
    elif exits["sl_type"] == "atr_mult":
        exits["sl_fixed_pct"] = 0
    elif exits["sl_type"] == "swing_low":
        exits["sl_fixed_pct"] = 0
        exits["sl_atr_mult"] = 0

    # Ensure partial < TP
    if exits["partial_exit"] and exits["partial_target_r"] >= exits["tp_r"]:
        exits["partial_target_r"] = exits["tp_r"] * 0.5

    # Ensure timing consistency
    if filters["earliest_entry_min"] >= filters["latest_entry_min"]:
        filters["latest_entry_min"] = max(filters["earliest_entry_min"] + 15, 60)

    direction = _normalize_direction(entry_type, entry_params, exits["direction"])

    return PennyGenes(
        entry_type=entry_type,
        entry_params=entry_params,
        min_vol_spike=filters["min_vol_spike"],
        require_vol_accel=filters["require_vol_accel"],
        entry_delay_bars=filters["entry_delay_bars"],
        earliest_entry_min=filters["earliest_entry_min"],
        latest_entry_min=filters["latest_entry_min"],
        min_first_bar_ret_pct=filters["min_first_bar_ret_pct"],
        max_spread_pct=filters["max_spread_pct"],
        confirmation_type=conf_type,
        confirmation_params=conf_params,
        tp_r=exits["tp_r"],
        sl_type=exits["sl_type"],
        sl_fixed_pct=exits["sl_fixed_pct"],
        sl_buffer_pct=exits["sl_buffer_pct"],
        sl_atr_mult=exits["sl_atr_mult"],
        sl_atr_period=exits["sl_atr_period"],
        trail_type=exits["trail_type"],
        trail_activation_r=exits["trail_activation_r"],
        trail_distance_pct=exits["trail_distance_pct"],
        trail_ema_period=exits["trail_ema_period"],
        max_hold_minutes=exits["max_hold_minutes"],
        eod_exit=True,
        partial_exit=exits["partial_exit"],
        partial_pct=exits["partial_pct"],
        partial_target_r=exits["partial_target_r"],
        direction=direction,
    )


def sample_candidates(
    n: int = 5000,
    seed: int = 42,
    coverage_pct: float = 0.30,
    archetype_pct: float = 0.20,
) -> list[PennyGenes]:
    """Sample N candidate gene configurations with 3-pass strategy.

    Pass 1 (coverage_pct): Force coverage of every entry type.
    Pass 2 (1 - coverage_pct - archetype_pct): Pure random sampling.
    Pass 3 (archetype_pct): Balance archetypes and direction.
    """
    rng = random.Random(seed)
    candidates = []
    seen_hashes = set()

    def _gene_hash(g: PennyGenes) -> str:
        d = g.to_dict()
        return str(sorted(d.items()))

    def _add_unique(g: PennyGenes) -> bool:
        h = _gene_hash(g)
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        candidates.append(g)
        return True

    entry_types = list(ENTRY_TYPE_CATALOG.keys())

    # ── Pass 1: Coverage — ensure every entry type is represented ──
    n_coverage = int(n * coverage_pct)
    per_type = max(3, n_coverage // len(entry_types))

    for et in entry_types:
        for _ in range(per_type):
            g = sample_random_genes(rng)
            g.entry_type = et
            g.entry_params = _sample_entry_type_params(et, rng)
            g.direction = _normalize_direction(et, g.entry_params, g.direction)
            _add_unique(g)

    # ── Pass 2: Random ──
    pass2_target = n - int(n * archetype_pct)
    attempts = 0
    max_attempts = (pass2_target - len(candidates)) * 3
    while len(candidates) < pass2_target and attempts < max_attempts:
        g = sample_random_genes(rng)
        _add_unique(g)
        attempts += 1

    # ── Pass 3: Archetype balance ──
    remaining = n - len(candidates)

    # Ensure enough of each archetype
    archetype_counts = {}
    for c in candidates:
        a = ENTRY_TYPE_CATALOG.get(c.entry_type, {}).get("archetype", "unknown")
        archetype_counts[a] = archetype_counts.get(a, 0) + 1

    target_per_arch = n // (len(ARCHETYPES) + 1)
    for arch in ARCHETYPES:
        current = archetype_counts.get(arch, 0)
        if current < target_per_arch and remaining > 0:
            # Find entry types with this archetype
            arch_types = [k for k, v in ENTRY_TYPE_CATALOG.items()
                         if v["archetype"] == arch]
            if not arch_types:
                continue
            deficit = min(target_per_arch - current, remaining)
            for _ in range(deficit):
                g = sample_random_genes(rng)
                g.entry_type = rng.choice(arch_types)
                g.entry_params = _sample_entry_type_params(g.entry_type, rng)
                g.direction = _normalize_direction(g.entry_type, g.entry_params, g.direction)
                if _add_unique(g):
                    remaining -= 1

    # Ensure enough shorts
    n_shorts = sum(1 for c in candidates if c.direction == "short")
    while n_shorts < n * 0.30 and remaining > 0:
        g = sample_random_genes(rng)
        # Prefer short, but let entry type override if it forces a direction
        g.direction = _normalize_direction(g.entry_type, g.entry_params, "short")
        if _add_unique(g):
            n_shorts += 1
            remaining -= 1

    # Fill remaining
    while len(candidates) < n:
        g = sample_random_genes(rng)
        _add_unique(g)

    return candidates[:n]


# ═════════════════════════════════════════════════════════════════════════════
# DESCRIBE / DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def describe_genes(g: PennyGenes) -> str:
    """Human-readable one-liner for a gene config."""
    dir_str = "L" if g.direction == "long" else "S"

    # Entry type summary
    et = g.entry_type
    ep = g.entry_params
    if et == "ramp":
        entry_str = f"ramp({ep.get('min_ramp_bars',2)}bar/{ep.get('min_ramp_body_pct',1)}%)"
    elif et == "orb":
        entry_str = f"ORB({ep.get('orb_minutes',5)}m+{ep.get('orb_buffer_pct',0)}%)"
    elif et == "vwap_cross":
        entry_str = f"VWAP-x({ep.get('cross_direction','above')})"
    elif et == "vwap_bounce":
        entry_str = f"VWAP-bounce({ep.get('bounce_direction','long')})"
    elif et == "ema_cross":
        entry_str = f"EMA-x({ep.get('fast_period',5)}/{ep.get('slow_period',21)})"
    elif et == "ema_trend":
        entry_str = f"EMA-trend({ep.get('ema_period',13)},{ep.get('price_vs_ema','above')})"
    elif et == "pullback":
        entry_str = f"PB({ep.get('pullback_depth_pct',50)}%of{ep.get('min_initial_move_pct',5)}%)"
    elif et == "macd_cross":
        entry_str = f"MACD({ep.get('cross_type','bullish')})"
    elif et == "rsi_extreme":
        entry_str = f"RSI({ep.get('entry_on','oversold')},{ep.get('rsi_period',14)})"
    elif et == "candle_pattern":
        entry_str = f"candle({ep.get('pattern','engulfing')})"
    elif et == "volume_climax":
        entry_str = f"vol-climax({ep.get('vol_multiple',5)}x,{ep.get('interpretation','momentum')})"
    elif et == "range_breakout":
        entry_str = f"range-break({ep.get('lookback_bars',10)}bar,{ep.get('break_direction','high')})"
    elif et == "gap_fill_entry":
        entry_str = f"gap-fill(>{ep.get('min_gap_pct',50)}%)"
    elif et == "hod_break":
        entry_str = f"HOD-break(age>{ep.get('min_hod_age_bars',10)})"
    elif et == "first_red_fade":
        entry_str = f"1st-red(after{ep.get('min_green_run',5)}green)"
    elif et == "price_level":
        entry_str = f"level({ep.get('level_type','whole_dollar')},{ep.get('interaction','break')})"
    elif et == "momentum_accel":
        entry_str = f"mom-accel({ep.get('window_bars',5)}bar)"
    elif et == "sustained_volume":
        entry_str = f"sust-vol({ep.get('min_avg_vol_ratio',3)}x/{ep.get('min_sustained_bars',10)}bar)"
    else:
        entry_str = et

    parts = [
        f"{dir_str} {entry_str}",
        f"TP={g.tp_r}R",
        f"SL={g.sl_type}",
    ]

    if g.confirmation_type != "none":
        parts.append(f"+{g.confirmation_type}")

    if g.trail_type != "none":
        parts.append(f"trail={g.trail_type}@{g.trail_activation_r}R")

    if g.max_hold_minutes < 390:
        parts.append(f"hold<={g.max_hold_minutes}m")

    if g.partial_exit:
        parts.append(f"partial={g.partial_pct*100:.0f}%@{g.partial_target_r}R")

    return " | ".join(parts)


def print_gene_space_summary():
    """Print overview of the full gene space."""
    print(f"\n{'='*70}")
    print(f"PENNY GENE SPACE SUMMARY")
    print(f"{'='*70}")

    print(f"\n  ENTRY TYPES ({len(ENTRY_TYPE_CATALOG)}):")
    for et, info in ENTRY_TYPE_CATALOG.items():
        n_params = 1
        for v in info["params"].values():
            n_params *= len(v)
        print(f"    {et:<22s} [{info['archetype']:<16s}] {n_params:>6} param combos | {info['description']}")

    total_entry_combos = sum(
        max(1, np.prod([len(v) for v in info["params"].values()]))
        for info in ENTRY_TYPE_CATALOG.values()
    )

    n_filter = 1
    for v in FILTER_RANGES.values():
        n_filter *= len(v)

    n_conf = sum(max(1, np.prod([len(v) for v in params.values()]) if params else 1)
                 for params in CONFIRMATION_TYPES.values())

    n_exit = 1
    for v in EXIT_RANGES.values():
        n_exit *= len(v)

    print(f"\n  DIMENSIONS:")
    print(f"    Entry type combos:        {total_entry_combos:>12,}")
    print(f"    Filter combos:            {n_filter:>12,}")
    print(f"    Confirmation combos:      {n_conf:>12,}")
    print(f"    Exit combos:              {n_exit:>12,}")
    total = total_entry_combos * n_filter * n_conf * n_exit
    print(f"    TOTAL (raw):              {total:>12,} ({total/1e9:.1f}B)")

    print(f"\n  ARCHETYPES:")
    for arch in ARCHETYPES:
        types = [k for k, v in ENTRY_TYPE_CATALOG.items() if v["archetype"] == arch]
        print(f"    {arch:<20s}: {', '.join(types)}")

    print()


# Need numpy for print_gene_space_summary
import numpy as np
