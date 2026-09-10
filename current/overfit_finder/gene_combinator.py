"""
Gene Combinator — Systematic Strategy Assembly Engine
=====================================================
Breaks strategies into modular genes (entry, confirmation, filter, exit,
direction, time window) and systematically explores the combinatorial space.

The 22 existing DSL condition types ARE the genes — this module just
assembles them into valid strategy JSON instead of relying on 22 hand-
written seeds. No new condition code, no look-ahead bias risk.

Analysis showed: 64% of DSL condition types (14 of 22) never appeared
in a winning strategy. This module forces exploration of the full space.

Usage:
    from gene_combinator import run_gene_search

    results = run_gene_search(cache, "2024-06-01", "2024-08-01",
                              window_days=45, max_candidates=5000)
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# StrategyDefinition imported at runtime (requires pandas/numpy)


# ═════════════════════════════════════════════════════════════════════════════
# GENE CATALOG — concrete condition dicts, ready for entry.conditions[]
# ═════════════════════════════════════════════════════════════════════════════

# Each entry gene has: condition dict, implied direction, archetype tag, and
# an optional level (for level-based stop losses).

@dataclass
class EntryGene:
    condition: dict
    implied_direction: str   # "long", "short", or "either"
    archetype: str
    level: Optional[str] = None  # for level-based SL


ENTRY_GENES = [
    # ── price_broke variants ──
    EntryGene({"type": "price_broke", "level": "on_high", "direction": "above", "lookback_bars": 3},
              "long", "on_breakout", "on_high"),
    EntryGene({"type": "price_broke", "level": "on_low", "direction": "below", "lookback_bars": 3},
              "short", "on_breakout", "on_low"),
    EntryGene({"type": "price_broke", "level": "prev_close", "direction": "above", "lookback_bars": 3},
              "long", "prev_close", "prev_close"),
    EntryGene({"type": "price_broke", "level": "prev_close", "direction": "below", "lookback_bars": 3},
              "short", "prev_close", "prev_close"),
    EntryGene({"type": "price_broke", "level": "day_open", "direction": "above", "lookback_bars": 3},
              "long", "day_open", "day_open"),
    EntryGene({"type": "price_broke", "level": "day_open", "direction": "below", "lookback_bars": 3},
              "short", "day_open", "day_open"),
    EntryGene({"type": "price_broke", "level": "on_mid", "direction": "above", "lookback_bars": 3},
              "long", "mean_reversion", "on_mid"),
    EntryGene({"type": "price_broke", "level": "on_mid", "direction": "below", "lookback_bars": 3},
              "short", "mean_reversion", "on_mid"),

    # ── pullback_to variants ──
    EntryGene({"type": "pullback_to", "reference": "on_high", "buffer_pts": 3.0},
              "short", "mean_reversion", "on_high"),
    EntryGene({"type": "pullback_to", "reference": "on_low", "buffer_pts": 3.0},
              "long", "mean_reversion", "on_low"),
    EntryGene({"type": "pullback_to", "reference": "prev_close", "buffer_pts": 3.0},
              "either", "prev_close", "prev_close"),
    EntryGene({"type": "pullback_to", "reference": "day_open", "buffer_pts": 3.0},
              "either", "day_open", "day_open"),
    EntryGene({"type": "pullback_to", "reference": "on_mid", "buffer_pts": 3.0},
              "either", "mean_reversion", "on_mid"),

    # ── GEX wall proximity ──
    EntryGene({"type": "price_near_wall", "wall": "put", "within_pts": 8.0},
              "long", "gex_mean_reversion"),
    EntryGene({"type": "price_near_wall", "wall": "call", "within_pts": 8.0},
              "short", "gex_mean_reversion"),

    # ── VWAP fade — price extended from today's rolling VWAP, fade back ──
    EntryGene({"type": "price_extended_above_vwap", "pts": 2.0}, "short", "vwap_fade"),
    EntryGene({"type": "price_extended_above_vwap", "pts": 3.0}, "short", "vwap_fade"),
    EntryGene({"type": "price_extended_above_vwap", "pts": 4.0}, "short", "vwap_fade"),
    EntryGene({"type": "price_extended_above_vwap", "pts": 6.0}, "short", "vwap_fade"),
    EntryGene({"type": "price_extended_above_vwap", "pts": 8.0}, "short", "vwap_fade"),
    EntryGene({"type": "price_extended_below_vwap", "pts": 2.0}, "long", "vwap_fade"),
    EntryGene({"type": "price_extended_below_vwap", "pts": 3.0}, "long", "vwap_fade"),
    EntryGene({"type": "price_extended_below_vwap", "pts": 4.0}, "long", "vwap_fade"),
    EntryGene({"type": "price_extended_below_vwap", "pts": 6.0}, "long", "vwap_fade"),
    EntryGene({"type": "price_extended_below_vwap", "pts": 8.0}, "long", "vwap_fade"),

    # ── Gap ──
    EntryGene({"type": "gap_exists", "location": "any"},
              "either", "gap_fill_fade"),
    EntryGene({"type": "gap_exists", "location": "near_on_high"},
              "short", "gap_fill_fade"),
    EntryGene({"type": "gap_exists", "location": "near_on_low"},
              "long", "gap_fill_fade"),

    # ── GEX Momentum — trade WITH the gamma exposure direction ──
    # Positive GEX = dealer hedging pins price → breakout through walls is momentum
    EntryGene({"type": "gex_regime_is", "regime": "positive"},
              "long", "gex_momentum"),
    EntryGene({"type": "gex_regime_is", "regime": "negative"},
              "short", "gex_momentum"),
    # High/low z-score = extreme positioning → momentum entry
    EntryGene({"type": "gex_zscore_above", "value": 1.5},
              "long", "gex_momentum"),
    EntryGene({"type": "gex_zscore_above", "value": 1.0},
              "long", "gex_momentum"),
    EntryGene({"type": "gex_zscore_below", "value": -1.5},
              "short", "gex_momentum"),
    EntryGene({"type": "gex_zscore_below", "value": -1.0},
              "short", "gex_momentum"),
    # Breakout through GEX wall = strong momentum signal
    EntryGene({"type": "price_near_wall", "wall": "call", "within_pts": 5.0},
              "long", "gex_momentum"),
    EntryGene({"type": "price_near_wall", "wall": "put", "within_pts": 5.0},
              "short", "gex_momentum"),
    # High P/C ratio = heavy put buying → contrarian long or momentum short
    EntryGene({"type": "pc_ratio_above", "value": 1.5},
              "short", "gex_momentum"),
    EntryGene({"type": "pc_ratio_above", "value": 2.0},
              "long", "gex_momentum"),   # extreme P/C = contrarian long

    # ── Earnings Catalyst — trade around earnings-driven volatility ──
    EntryGene({"type": "earnings_nearby", "min_tickers": 1},
              "either", "earnings_catalyst"),
    EntryGene({"type": "earnings_nearby", "min_tickers": 3},
              "either", "earnings_catalyst"),
    EntryGene({"type": "earnings_nearby", "min_tickers": 5},
              "either", "earnings_catalyst"),
    # No earnings = low-vol mean reversion environment
    EntryGene({"type": "no_earnings_nearby"},
              "either", "earnings_catalyst"),

    # ── Vol Regime — trade based on realized volatility state ──
    EntryGene({"type": "realized_vol_above", "value": 0.20},
              "either", "vol_regime"),
    EntryGene({"type": "realized_vol_above", "value": 0.15},
              "either", "vol_regime"),
    EntryGene({"type": "realized_vol_below", "value": 0.10},
              "either", "vol_regime"),
    EntryGene({"type": "realized_vol_below", "value": 0.08},
              "either", "vol_regime"),
    # IV-driven entries
    EntryGene({"type": "iv_above", "value": 0.25},
              "either", "vol_regime"),
    EntryGene({"type": "iv_above", "value": 0.20},
              "either", "vol_regime"),

    # ── Momentum entries — price structure patterns ──
    # Broke ON levels with tighter lookback = fresh momentum
    EntryGene({"type": "price_broke", "level": "on_high", "direction": "above", "lookback_bars": 1},
              "long", "on_breakout", "on_high"),
    EntryGene({"type": "price_broke", "level": "on_low", "direction": "below", "lookback_bars": 1},
              "short", "on_breakout", "on_low"),
    # Wider lookback = confirmed breakout
    EntryGene({"type": "price_broke", "level": "on_high", "direction": "above", "lookback_bars": 5},
              "long", "on_breakout", "on_high"),
    EntryGene({"type": "price_broke", "level": "on_low", "direction": "below", "lookback_bars": 5},
              "short", "on_breakout", "on_low"),
    # Prev close with different lookbacks
    EntryGene({"type": "price_broke", "level": "prev_close", "direction": "above", "lookback_bars": 1},
              "long", "prev_close", "prev_close"),
    EntryGene({"type": "price_broke", "level": "prev_close", "direction": "below", "lookback_bars": 1},
              "short", "prev_close", "prev_close"),

    # ── Pullback variants with tighter/wider buffers ──
    EntryGene({"type": "pullback_to", "reference": "on_high", "buffer_pts": 5.0},
              "short", "mean_reversion", "on_high"),
    EntryGene({"type": "pullback_to", "reference": "on_low", "buffer_pts": 5.0},
              "long", "mean_reversion", "on_low"),
    EntryGene({"type": "pullback_to", "reference": "prev_close", "buffer_pts": 5.0},
              "either", "prev_close", "prev_close"),
    EntryGene({"type": "pullback_to", "reference": "prev_close", "buffer_pts": 1.5},
              "either", "prev_close", "prev_close"),

    # ── ON range as entry (wide range = momentum day, tight = mean revert) ──
    EntryGene({"type": "on_range_between", "min": 15, "max": 50},
              "either", "on_breakout"),
    EntryGene({"type": "on_range_between", "min": 5, "max": 12},
              "either", "mean_reversion"),

    # ═══ OVERNIGHT-SPECIFIC ENTRY GENES ═══

    # ── Session VWAP cross ──
    EntryGene({"type": "session_vwap_cross", "direction": "above"},
              "long", "vwap_fade"),
    EntryGene({"type": "session_vwap_cross", "direction": "below"},
              "short", "vwap_fade"),

    # ── Opening range breakout (first N 15m bars) ──
    EntryGene({"type": "orb_breakout", "n_bars": 1, "direction": "above"},
              "long", "on_breakout"),   # first 15 min
    EntryGene({"type": "orb_breakout", "n_bars": 1, "direction": "below"},
              "short", "on_breakout"),
    EntryGene({"type": "orb_breakout", "n_bars": 2, "direction": "above"},
              "long", "on_breakout"),   # first 30 min
    EntryGene({"type": "orb_breakout", "n_bars": 2, "direction": "below"},
              "short", "on_breakout"),
    EntryGene({"type": "orb_breakout", "n_bars": 4, "direction": "above"},
              "long", "on_breakout"),   # first 60 min
    EntryGene({"type": "orb_breakout", "n_bars": 4, "direction": "below"},
              "short", "on_breakout"),

    # ── Reversion to session open ──
    EntryGene({"type": "reversion_to_session_open", "buffer_pts": 2.0, "min_excursion_pts": 4.0},
              "either", "mean_reversion"),
    EntryGene({"type": "reversion_to_session_open", "buffer_pts": 3.0, "min_excursion_pts": 6.0},
              "either", "mean_reversion"),
    EntryGene({"type": "reversion_to_session_open", "buffer_pts": 1.5, "min_excursion_pts": 3.0},
              "either", "mean_reversion"),

    # ── Consecutive candle momentum (fade exhaustion or ride continuation) ──
    EntryGene({"type": "consecutive_candles", "n": 3, "direction": "bullish"},
              "short", "mean_reversion"),  # 3 green → fade
    EntryGene({"type": "consecutive_candles", "n": 3, "direction": "bearish"},
              "long", "mean_reversion"),   # 3 red → fade
    EntryGene({"type": "consecutive_candles", "n": 4, "direction": "bullish"},
              "short", "mean_reversion"),  # 4 green → fade
    EntryGene({"type": "consecutive_candles", "n": 4, "direction": "bearish"},
              "long", "mean_reversion"),   # 4 red → fade
    EntryGene({"type": "consecutive_candles", "n": 3, "direction": "bullish"},
              "long", "on_breakout"),      # 3 green → continuation
    EntryGene({"type": "consecutive_candles", "n": 3, "direction": "bearish"},
              "short", "on_breakout"),     # 3 red → continuation

    # ── Extended from session extreme (mean-reversion from session high/low) ──
    EntryGene({"type": "extended_from_session_extreme", "extreme": "high", "min_dist_pts": 3.0},
              "long", "mean_reversion"),   # dropped from session high → buy
    EntryGene({"type": "extended_from_session_extreme", "extreme": "high", "min_dist_pts": 5.0},
              "long", "mean_reversion"),
    EntryGene({"type": "extended_from_session_extreme", "extreme": "low", "min_dist_pts": 3.0},
              "short", "mean_reversion"),  # rallied from session low → sell
    EntryGene({"type": "extended_from_session_extreme", "extreme": "low", "min_dist_pts": 5.0},
              "short", "mean_reversion"),

    # ── Bar range compression (squeeze → breakout) ──
    EntryGene({"type": "bar_range_compression", "n": 3, "max_range_pts": 1.5},
              "either", "vol_regime"),
    EntryGene({"type": "bar_range_compression", "n": 4, "max_range_pts": 2.0},
              "either", "vol_regime"),
    EntryGene({"type": "bar_range_compression", "n": 5, "max_range_pts": 2.0},
              "either", "vol_regime"),

    # ── RTH close bias (overnight directional bias from where RTH closed) ──
    EntryGene({"type": "rth_close_bias", "bias": "bullish", "threshold": 0.75},
              "long", "prev_close"),   # RTH closed near high → ON bullish
    EntryGene({"type": "rth_close_bias", "bias": "bearish", "threshold": 0.75},
              "short", "prev_close"),  # RTH closed near low → ON bearish
    EntryGene({"type": "rth_close_bias", "bias": "bullish", "threshold": 0.85},
              "long", "prev_close"),   # strong close near high
    EntryGene({"type": "rth_close_bias", "bias": "bearish", "threshold": 0.85},
              "short", "prev_close"),  # strong close near low
]


# ── Confirmation genes (0-1 per strategy) ──

CONFIRMATION_GENES = [
    {"type": "candle_pattern", "pattern": "is_bullish", "_dir": "long"},
    {"type": "candle_pattern", "pattern": "is_bearish", "_dir": "short"},
    {"type": "candle_pattern", "pattern": "large_wick_up", "size_pts": 5.0, "_dir": "short"},
    {"type": "candle_pattern", "pattern": "large_wick_down", "size_pts": 5.0, "_dir": "long"},
    {"type": "min_candle_size", "size_pts": 5.0, "_dir": "any"},
    # Overnight-specific confirmations
    {"type": "body_range_ratio", "op": "below", "threshold": 0.3, "_dir": "any"},  # doji = indecision
    {"type": "body_range_ratio", "op": "above", "threshold": 0.7, "_dir": "any"},  # marubozu = conviction
    None,  # no confirmation
]


# ── Filter genes (entry-level conditions used as environment checks, 0-2 per strategy) ──

FILTER_ENTRY_GENES = [
    # GEX regime
    {"type": "gex_regime_is", "regime": "positive"},
    {"type": "gex_regime_is", "regime": "negative"},
    # GEX z-score
    {"type": "gex_zscore_above", "value": 1.0},
    {"type": "gex_zscore_below", "value": -1.0},
    # IV
    {"type": "iv_above", "value": 0.20},
    # P/C ratio
    {"type": "pc_ratio_above", "value": 1.5},
    # Realized vol
    {"type": "realized_vol_above", "value": 0.15},
    {"type": "realized_vol_below", "value": 0.10},
    # Earnings
    {"type": "earnings_nearby", "min_tickers": 1},
    {"type": "no_earnings_nearby"},
    # ON range
    {"type": "on_range_between", "min": 8, "max": 30},
    {"type": "on_range_between", "min": 15, "max": 50},
    # Overnight-specific filters
    {"type": "rth_close_bias", "bias": "bullish", "threshold": 0.70},
    {"type": "rth_close_bias", "bias": "bearish", "threshold": 0.70},
    {"type": "consecutive_candles", "n": 3, "direction": "bullish"},
    {"type": "consecutive_candles", "n": 3, "direction": "bearish"},
    None,  # no filter
]


# ── Day-level filters (applied before candle iteration) ──

DAY_FILTER_SETS = [
    [{"type": "min_on_range", "value": 8}, {"type": "max_on_range", "value": 40}],
    [{"type": "min_on_range", "value": 10}, {"type": "max_on_range", "value": 50}],
    [{"type": "min_on_range", "value": 5}],
    [],  # no day filters
]


# ── Exit genes ──

_BASE_EXITS_RTH = [
    {"sl": {"type": "fixed_pts", "value": 8},  "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 10}, "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 10}, "tp": {"type": "risk_multiple", "value": 2.0}},
    {"sl": {"type": "fixed_pts", "value": 12}, "tp": {"type": "risk_multiple", "value": 2.0}},
    {"sl": {"type": "fixed_pts", "value": 12}, "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 15}, "tp": {"type": "risk_multiple", "value": 1.0}},
    {"sl": {"type": "fixed_pts", "value": 10}, "tp": {"type": "risk_multiple", "value": 2.5}},
    {"sl": {"type": "candle_wick", "buffer_pts": 3.0}, "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "candle_wick", "buffer_pts": 3.0}, "tp": {"type": "risk_multiple", "value": 2.0}},
]

_BASE_EXITS_OVERNIGHT = [
    {"sl": {"type": "fixed_pts", "value": 4},  "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 5},  "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 5},  "tp": {"type": "risk_multiple", "value": 2.0}},
    {"sl": {"type": "fixed_pts", "value": 6},  "tp": {"type": "risk_multiple", "value": 2.0}},
    {"sl": {"type": "fixed_pts", "value": 6},  "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "fixed_pts", "value": 8},  "tp": {"type": "risk_multiple", "value": 1.0}},
    {"sl": {"type": "fixed_pts", "value": 5},  "tp": {"type": "risk_multiple", "value": 2.5}},
    {"sl": {"type": "candle_wick", "buffer_pts": 2.0}, "tp": {"type": "risk_multiple", "value": 1.5}},
    {"sl": {"type": "candle_wick", "buffer_pts": 2.0}, "tp": {"type": "risk_multiple", "value": 2.0}},
]

def _build_tsl_be_options(meta=None) -> list[dict]:
    """Generate TSL/BE option variants from metaconfig ranges.

    When meta is provided, samples 5 spread-out combos within the configured
    ranges. When meta is None, uses hardcoded defaults for backward compat.
    """
    if meta is None:
        return [
            {},
            {"be_trigger_pts": 8},
            {"be_trigger_pts": 10},
            {"be_trigger_pts": 8,  "trail_distance_pts": 5},
            {"be_trigger_pts": 10, "trail_distance_pts": 8},
            {"be_trigger_pts": 12, "trail_distance_pts": 10},
        ]
    be_lo = meta.tsl_be_min_trigger_pts
    be_hi = meta.tsl_be_max_trigger_pts
    tr_lo = meta.tsl_min_trail_pts
    tr_hi = meta.tsl_max_trail_pts
    be_mid = round((be_lo + be_hi) / 2, 1)
    tr_mid = round((tr_lo + tr_hi) / 2, 1)
    return [
        {},                                                        # no TSL/BE
        {"be_trigger_pts": round(be_lo, 1)},                       # BE only, tight
        {"be_trigger_pts": round(be_hi, 1)},                       # BE only, wide
        {"be_trigger_pts": round(be_lo, 1), "trail_distance_pts": round(tr_lo, 1)},  # tight BE + tight trail
        {"be_trigger_pts": be_mid,          "trail_distance_pts": tr_mid},             # mid BE + mid trail
        {"be_trigger_pts": round(be_hi, 1), "trail_distance_pts": round(tr_hi, 1)},  # wide BE + wide trail
    ]


def _build_exit_genes(meta=None, time_stop: str = "15:45", overnight: bool = False) -> list[dict]:
    """Build EXIT_GENES from base exits × TSL/BE options."""
    tsl_be_options = _build_tsl_be_options(meta)
    base_exits = _BASE_EXITS_OVERNIGHT if overnight else _BASE_EXITS_RTH
    genes = []
    for base in base_exits:
        for tsl_be in tsl_be_options:
            genes.append({**base, "time_stop": time_stop, **tsl_be})
    return genes


# Default EXIT_GENES (no metaconfig) for backward compat
EXIT_GENES = _build_exit_genes()


def _level_exits(level: str, meta=None, time_stop: str = "15:45", overnight: bool = False) -> list[dict]:
    """Generate level-based SL exits for an entry gene with a known level."""
    if not level:
        return []
    tsl_be_options = _build_tsl_be_options(meta)
    # Use empty + 2 TSL/BE combos (tight + mid)
    tsl_be_subset = [tsl_be_options[0]]  # no TSL/BE
    if len(tsl_be_options) > 3:
        tsl_be_subset.append(tsl_be_options[3])  # tight combo
    if len(tsl_be_options) > 4:
        tsl_be_subset.append(tsl_be_options[4])  # mid combo
    buf_lo = 2.0 if overnight else 3.0
    buf_hi = 3.0 if overnight else 4.0
    exits = []
    for tsl_be in tsl_be_subset:
        exits.append({"sl": {"type": "level", "value": level, "buffer_pts": buf_lo},
                       "tp": {"type": "risk_multiple", "value": 1.5},
                       "time_stop": time_stop, **tsl_be})
        exits.append({"sl": {"type": "level", "value": level, "buffer_pts": buf_hi},
                       "tp": {"type": "risk_multiple", "value": 2.0},
                       "time_stop": time_stop, **tsl_be})
    return exits


def sample_exit_gene(meta, rng: random.Random | None = None, time_stop: str = "15:45", overnight: bool = False) -> dict:
    """Sample a single exit gene from MetaConfig ranges.

    Three exit types weighted by probability (overnight shifts toward simpler exits):
    - fixed_sl_tp: Fixed SL + fixed TP in pts
    - tsl_be: Fixed SL + risk_multiple TP + breakeven/trailing stop
    - sl_only: Fixed SL only, wide TP (trade runs to EOD or SL)
    """
    _rng = rng or random
    if overnight:
        # Simpler exits overnight — fewer degrees of freedom
        weights = [0.50, 0.30, 0.20]
    else:
        weights = [0.35, 0.50, 0.15]
    exit_type = _rng.choices(
        ["fixed_sl_tp", "tsl_be", "sl_only"],
        weights=weights,
    )[0]

    sl_pts = round(_rng.uniform(meta.sl_range_min, meta.sl_range_max), 1)

    if exit_type == "fixed_sl_tp":
        tp_pts = round(_rng.uniform(meta.tp_range_min, meta.tp_range_max), 1)
        return {
            "sl": {"type": "fixed_pts", "value": sl_pts},
            "tp": {"type": "fixed_pts", "value": tp_pts},
            "time_stop": time_stop,
        }
    elif exit_type == "tsl_be":
        # Risk-multiple TP as backstop, TSL/BE does the real exit management
        mult = _rng.choice([1.5, 2.0, 2.5, 3.0])
        be_trigger = round(_rng.uniform(
            meta.tsl_be_min_trigger_pts, meta.tsl_be_max_trigger_pts), 1)
        trail_dist = round(_rng.uniform(
            meta.tsl_min_trail_pts, meta.tsl_max_trail_pts), 1)
        return {
            "sl": {"type": "fixed_pts", "value": sl_pts},
            "tp": {"type": "risk_multiple", "value": mult},
            "time_stop": time_stop,
            "be_trigger_pts": be_trigger,
            "trail_distance_pts": trail_dist,
        }
    else:  # sl_only — wide TP, trade runs to SL or EOD
        return {
            "sl": {"type": "fixed_pts", "value": sl_pts},
            "tp": {"type": "risk_multiple", "value": 10.0},
            "time_stop": time_stop,
        }


# ── Time windows ──

TIME_WINDOWS = [
    {"start": "09:30", "end": "10:30"},  # first hour
    {"start": "09:45", "end": "11:00"},  # morning
    {"start": "10:00", "end": "12:00"},  # mid-morning
    {"start": "09:30", "end": "15:45"},  # all day
    {"start": "12:00", "end": "15:45"},  # afternoon
    {"start": "09:30", "end": "11:30"},  # extended morning
]


# ── Overnight time windows ──
# Asia session: 18:00 ET → 02:00 ET (next day)
# London session: 03:00 ET → 08:00 ET
# Pre-market: 08:00 ET → 09:30 ET
# Full overnight: 18:00 ET → 09:30 ET
OVERNIGHT_TIME_WINDOWS = [
    {"start": "18:00", "end": "02:00"},   # Asia session
    {"start": "18:00", "end": "23:00"},   # early Asia
    {"start": "20:00", "end": "02:00"},   # late Asia
    {"start": "03:00", "end": "08:00"},   # London session
    {"start": "03:00", "end": "06:00"},   # early London
    {"start": "05:00", "end": "08:00"},   # late London
    {"start": "08:00", "end": "09:30"},   # pre-market
    {"start": "18:00", "end": "09:30"},   # full overnight
    {"start": "18:00", "end": "08:00"},   # Asia + London
    {"start": "03:00", "end": "09:30"},   # London + pre-market
]

# Overnight time_stop per session — exit before next session boundary
OVERNIGHT_TIME_STOPS = {
    "02:00": "01:45",
    "23:00": "22:45",
    "06:00": "05:45",
    "08:00": "07:45",
    "09:30": "09:15",
}


# ── Overnight gene filtering ──
# These condition types require GEX/options/earnings data unavailable overnight
_GEX_DEPENDENT_TYPES = {
    "gex_regime_is", "gex_zscore_above", "gex_zscore_below",
    "price_near_wall", "pc_ratio_above", "iv_above",
    "earnings_nearby", "no_earnings_nearby",
    # Daily stats conditions — ctx.daily is None for overnight
    "realized_vol_above", "realized_vol_below",
}


def _is_overnight_compatible(gene) -> bool:
    """Check if an entry gene or filter is compatible with overnight sessions."""
    if gene is None:
        return True
    cond = gene.condition if isinstance(gene, EntryGene) else gene
    return cond.get("type", "") not in _GEX_DEPENDENT_TYPES


# Pre-filtered overnight gene lists
OVERNIGHT_ENTRY_GENES = [g for g in ENTRY_GENES if _is_overnight_compatible(g)]
OVERNIGHT_FILTER_ENTRY_GENES = [f for f in FILTER_ENTRY_GENES if _is_overnight_compatible(f)]


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRAINT VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

# Contradiction pairs: if both appear, the combo is invalid
_CONTRADICTIONS = {
    frozenset(["price_above_vwap", "price_below_vwap"]),
    frozenset(["price_extended_above_vwap", "price_extended_below_vwap"]),
    frozenset(["earnings_nearby", "no_earnings_nearby"]),
    frozenset(["realized_vol_above", "realized_vol_below"]),
}

# Same-type contradictions (e.g. two gex_regime_is with different regimes)
_SAME_TYPE_EXCLUSIVE = {"gex_regime_is", "on_range_between", "gex_zscore_above",
                         "gex_zscore_below", "iv_above", "pc_ratio_above",
                         "earnings_nearby", "realized_vol_above", "realized_vol_below",
                         "price_extended_above_vwap", "price_extended_below_vwap"}


def _cond_type(c: dict) -> str:
    return c.get("type", "")


def is_valid_combo(
    entry: EntryGene,
    confirmation: Optional[dict],
    filter1: Optional[dict],
    filter2: Optional[dict],
    direction: str,
) -> bool:
    """Check if a gene combination is valid (no contradictions/redundancy)."""

    # 1. Direction consistency
    if entry.implied_direction != "either" and entry.implied_direction != direction:
        return False

    if confirmation is not None:
        conf_dir = confirmation.get("_dir", "any")
        if conf_dir != "any":
            if conf_dir != direction:
                return False

    # 2. Collect all condition types
    all_conds = [entry.condition]
    if confirmation:
        all_conds.append(confirmation)
    if filter1:
        all_conds.append(filter1)
    if filter2:
        all_conds.append(filter2)

    types = [_cond_type(c) for c in all_conds]

    # 3. No redundancy: same condition type shouldn't appear twice
    #    (except candle_pattern with different patterns is OK)
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            if types[i] == types[j]:
                # Allow candle_pattern + min_candle_size (different types)
                # Disallow exact same type
                if types[i] in _SAME_TYPE_EXCLUSIVE:
                    return False
                if types[i] == types[j] and types[i] != "candle_pattern":
                    return False
                # Two candle_patterns only OK if different patterns
                if types[i] == "candle_pattern":
                    if all_conds[i].get("pattern") == all_conds[j].get("pattern"):
                        return False

    # 4. No contradictions
    type_set = set(types)
    for contradiction in _CONTRADICTIONS:
        if contradiction.issubset(type_set):
            return False

    # 5. Don't pair gex_regime_is positive + gex_zscore_below (or vice versa)
    has_gex_pos = any(c.get("type") == "gex_regime_is" and c.get("regime") == "positive" for c in all_conds)
    has_gex_neg = any(c.get("type") == "gex_regime_is" and c.get("regime") == "negative" for c in all_conds)
    has_zscore_above = any(c.get("type") == "gex_zscore_above" for c in all_conds)
    has_zscore_below = any(c.get("type") == "gex_zscore_below" for c in all_conds)
    if has_gex_pos and has_zscore_below:
        return False
    if has_gex_neg and has_zscore_above:
        return False

    # 6. filter2 only valid if filter1 exists
    if filter2 is not None and filter1 is None:
        return False

    return True


# ═════════════════════════════════════════════════════════════════════════════
# ARCHETYPE INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def infer_archetype(entry: EntryGene, filters: list[dict]) -> str:
    """Infer archetype from the entry gene and filters."""
    base = entry.archetype

    # If entry gene already specifies a specialized archetype, keep it
    if base in ("gex_momentum", "gex_mean_reversion", "earnings_catalyst", "vol_regime"):
        return base

    # GEX filter upgrades the archetype
    has_gex_filter = any(
        f.get("type", "").startswith("gex_") or f.get("type") == "price_near_wall"
        for f in filters if f
    )
    if has_gex_filter:
        if base in ("on_breakout", "day_open"):
            return "gex_momentum"
        else:
            return "gex_mean_reversion"

    # Earnings filter upgrades to earnings_catalyst
    has_earnings_filter = any(
        f.get("type") in ("earnings_nearby", "no_earnings_nearby")
        for f in filters if f
    )
    if has_earnings_filter:
        return "earnings_catalyst"

    # Vol filter upgrades to vol_regime
    has_vol_filter = any(
        f.get("type") in ("realized_vol_above", "realized_vol_below", "iv_above")
        for f in filters if f
    )
    if has_vol_filter:
        return "vol_regime"

    return base


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

_counter = 0

def _assemble_strategy(
    entry: EntryGene,
    confirmation: Optional[dict],
    filter1: Optional[dict],
    filter2: Optional[dict],
    exit_gene: dict,
    direction: str,
    time_window: dict,
    day_filters: list[dict],
) -> dict:
    """Assemble a complete strategy dict from genes."""
    global _counter
    _counter += 1

    # Build entry conditions
    conditions = [copy.deepcopy(entry.condition)]
    if confirmation:
        cond = {k: v for k, v in confirmation.items() if k != "_dir"}
        conditions.append(cond)
    if filter1:
        conditions.append(copy.deepcopy(filter1))
    if filter2:
        conditions.append(copy.deepcopy(filter2))

    # Infer archetype
    filter_list = [f for f in [filter1, filter2] if f]
    archetype = infer_archetype(entry, filter_list)

    # Build exit
    exit_cfg = {
        "stop_loss": copy.deepcopy(exit_gene["sl"]),
        "take_profit": copy.deepcopy(exit_gene["tp"]),
        "time_stop": exit_gene.get("time_stop", "15:45"),
    }
    if exit_gene.get("be_trigger_pts"):
        exit_cfg["be_trigger_pts"] = exit_gene["be_trigger_pts"]
    if exit_gene.get("trail_distance_pts"):
        exit_cfg["trail_distance_pts"] = exit_gene["trail_distance_pts"]

    # Build description from gene names
    entry_desc = entry.condition.get("type", "unknown")
    entry_level = entry.condition.get("level", entry.condition.get("reference", ""))
    dir_short = "L" if direction == "long" else "S"
    conf_desc = confirmation.get("pattern", "") if confirmation else ""
    filt_desc = "+".join(f.get("type", "")[:8] for f in filter_list) if filter_list else "nofilt"

    name = f"GC-{_counter} {dir_short} {entry_desc}({entry_level}) {conf_desc} [{filt_desc}]"

    return {
        "name": name,
        "description": f"Gene combinator strategy #{_counter}",
        "archetype": archetype,
        "entry": {
            "direction": direction,
            "time_window": copy.deepcopy(time_window),
            "conditions": conditions,
        },
        "exit": exit_cfg,
        "filters": copy.deepcopy(day_filters),
        "position_size": {"risk_dollars": 600},
    }


# ═════════════════════════════════════════════════════════════════════════════
# SMART SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

# Condition types that have never appeared in a winner (from regime_db analysis)
_UNDEREXPLORED_TYPES = {
    "gex_regime_is", "gex_zscore_above", "gex_zscore_below", "price_near_wall",
    "iv_above", "pc_ratio_above", "price_extended_above_vwap", "price_extended_below_vwap",
    "realized_vol_above", "realized_vol_below", "earnings_nearby",
    "no_earnings_nearby", "on_range_between",
}


def _pick_exit_gene(rng: random.Random, exit_genes, entry_level, meta=None, time_stop: str = "15:45", overnight: bool = False):
    """Pick an exit gene: 50% from pre-built list, 50% from sample_exit_gene when meta available."""
    level_exits = _level_exits(entry_level, meta, time_stop=time_stop, overnight=overnight)
    if level_exits and rng.random() < 0.3:
        return rng.choice(level_exits)
    if meta and rng.random() < 0.5:
        return sample_exit_gene(meta, rng, time_stop=time_stop, overnight=overnight)
    gene = copy.deepcopy(rng.choice(exit_genes))
    gene["time_stop"] = time_stop
    return gene


def _random_combo(rng: random.Random, exit_genes=None, meta=None, overnight: bool = False) -> Optional[dict]:
    """Generate one random valid strategy, or None if combo is invalid."""
    _entry_genes = OVERNIGHT_ENTRY_GENES if overnight else ENTRY_GENES
    _filter_genes = OVERNIGHT_FILTER_ENTRY_GENES if overnight else FILTER_ENTRY_GENES
    _time_windows = OVERNIGHT_TIME_WINDOWS if overnight else TIME_WINDOWS

    entry = rng.choice(_entry_genes)

    # Direction: use implied, or pick if "either"
    if entry.implied_direction == "either":
        direction = rng.choice(["long", "short"])
    else:
        direction = entry.implied_direction

    # Confirmation — lower probability overnight to reduce complexity
    _conf_prob = 0.25 if overnight else 0.5
    confirmation = rng.choice(CONFIRMATION_GENES) if rng.random() < _conf_prob else None

    # Filters (1-2, minimum 1 required) — fewer double-filters overnight
    _non_none_filters = [f for f in _filter_genes if f is not None]
    filter1 = rng.choice(_non_none_filters)
    _f2_prob = 0.10 if overnight else 0.3
    filter2 = rng.choice(_filter_genes) if rng.random() < _f2_prob else None

    if not is_valid_combo(entry, confirmation, filter1, filter2, direction):
        return None

    # Time window and matching time_stop
    time_window = rng.choice(_time_windows)
    tw_end = time_window["end"]
    if overnight:
        time_stop = OVERNIGHT_TIME_STOPS.get(tw_end, tw_end)
    else:
        time_stop = "15:45"

    # Exit: pick from pre-built genes or sample fresh from metaconfig ranges
    _exit_genes = exit_genes or EXIT_GENES
    exit_gene = _pick_exit_gene(rng, _exit_genes, entry.level, meta, time_stop=time_stop, overnight=overnight)

    day_filters = rng.choice(DAY_FILTER_SETS)

    return _assemble_strategy(
        entry, confirmation, filter1, filter2,
        exit_gene, direction, time_window, day_filters,
    )


def _coverage_combo(rng: random.Random, target_type: str, exit_genes=None, meta=None, overnight: bool = False) -> Optional[dict]:
    """Generate a strategy that uses a specific condition type."""
    _entry_genes = OVERNIGHT_ENTRY_GENES if overnight else ENTRY_GENES
    _filter_genes = OVERNIGHT_FILTER_ENTRY_GENES if overnight else FILTER_ENTRY_GENES
    _time_windows = OVERNIGHT_TIME_WINDOWS if overnight else TIME_WINDOWS

    # Find genes containing the target type
    matching_entries = [e for e in _entry_genes if e.condition.get("type") == target_type]
    matching_filters = [f for f in _filter_genes if f and f.get("type") == target_type]

    if not matching_entries and not matching_filters:
        return None

    # Prefer using the target as an entry gene if possible
    _non_none_filters = [f for f in _filter_genes if f is not None]
    if matching_entries and (not matching_filters or rng.random() < 0.6):
        entry = rng.choice(matching_entries)
        filter1 = rng.choice(_non_none_filters)  # always at least 1 filter
        filter2 = None
    else:
        # Use target as a filter gene
        entry = rng.choice(_entry_genes)
        filter1 = rng.choice(matching_filters)
        filter2 = rng.choice(_filter_genes) if rng.random() < 0.2 else None

    # Direction
    if entry.implied_direction == "either":
        direction = rng.choice(["long", "short"])
    else:
        direction = entry.implied_direction

    # Confirmation
    confirmation = rng.choice(CONFIRMATION_GENES) if rng.random() < 0.5 else None

    if not is_valid_combo(entry, confirmation, filter1, filter2, direction):
        return None

    # Time window and matching time_stop
    time_window = rng.choice(_time_windows)
    tw_end = time_window["end"]
    if overnight:
        time_stop = OVERNIGHT_TIME_STOPS.get(tw_end, tw_end)
    else:
        time_stop = "15:45"

    _exit_genes = exit_genes or EXIT_GENES
    exit_gene = _pick_exit_gene(rng, _exit_genes, entry.level, meta, time_stop=time_stop, overnight=overnight)

    day_filters = rng.choice(DAY_FILTER_SETS)

    return _assemble_strategy(
        entry, confirmation, filter1, filter2,
        exit_gene, direction, time_window, day_filters,
    )


def sample_candidates(
    max_candidates: int = 5000,
    seed: int = 42,
    meta=None,
    overnight: bool = False,
) -> list[dict]:
    """Sample diverse strategy candidates using three-pass sampling.

    When meta is provided, TSL/BE exit parameters are sampled from metaconfig
    ranges instead of hardcoded defaults.

    When overnight=True, uses overnight-compatible genes and time windows only
    (excludes GEX/earnings/IV dependent genes).

    Pass 1 (30%): Coverage — force exploration of underexplored condition types
    Pass 2 (50%): Random — uniform draws from full combo space
    Pass 3 (20%): Archetype balance — ensure each archetype has minimum representation

    Returns list of complete strategy dicts ready for StrategyDefinition.from_dict().
    """
    global _counter
    _counter = 0
    rng = random.Random(seed)

    _entry_genes = OVERNIGHT_ENTRY_GENES if overnight else ENTRY_GENES
    _filter_genes = OVERNIGHT_FILTER_ENTRY_GENES if overnight else FILTER_ENTRY_GENES
    _time_windows = OVERNIGHT_TIME_WINDOWS if overnight else TIME_WINDOWS

    # Build exit genes from metaconfig if available
    # For overnight, use a representative time_stop (varies per window, handled in _random_combo)
    _exit_genes = _build_exit_genes(meta, overnight=overnight) if meta else EXIT_GENES

    candidates = []
    seen_names = set()  # dedup by gene combination

    def _add(strat: Optional[dict]):
        if strat is None:
            return
        # Dedup by conditions + direction + exit + time window
        cond_sig = str(sorted(
            (c.get("type"), c.get("level", c.get("reference", c.get("pattern", c.get("regime", "")))))
            for c in strat["entry"]["conditions"]
        ))
        exit_sig = (str(strat["exit"]["stop_loss"].get("type", "")) + str(strat["exit"]["stop_loss"].get("value", ""))
                    + str(strat["exit"].get("be_trigger_pts", "")) + str(strat["exit"].get("trail_distance_pts", "")))
        tw_sig = strat["entry"]["time_window"]["start"]
        sig = cond_sig + strat["entry"]["direction"] + exit_sig + tw_sig
        if sig in seen_names:
            return
        seen_names.add(sig)
        candidates.append(strat)

    # ── Pass 1: Coverage (30%) ──
    # For overnight, skip GEX-dependent underexplored types
    _underexplored = {t for t in _UNDEREXPLORED_TYPES
                      if not overnight or t not in _GEX_DEPENDENT_TYPES}
    coverage_budget = int(max_candidates * 0.3)
    per_type = max(10, coverage_budget // max(len(_underexplored), 1))
    attempts_per_type = per_type * 5  # allow for invalid combos

    for target_type in _underexplored:
        count = 0
        for _ in range(attempts_per_type):
            if count >= per_type:
                break
            strat = _coverage_combo(rng, target_type, exit_genes=_exit_genes, meta=meta, overnight=overnight)
            if strat:
                _add(strat)
                count += 1

    print(f"  [genes] Coverage pass: {len(candidates)} candidates "
          f"(targeting {len(_underexplored)} underexplored types)")

    # ── Pass 2: Random (50%) ──
    random_budget = int(max_candidates * 0.5)
    random_attempts = random_budget * 3
    before = len(candidates)

    for _ in range(random_attempts):
        if len(candidates) - before >= random_budget:
            break
        strat = _random_combo(rng, exit_genes=_exit_genes, meta=meta, overnight=overnight)
        _add(strat)

    print(f"  [genes] Random pass: +{len(candidates) - before} candidates")

    # ── Pass 3: Archetype balance (20%) ──
    balance_budget = max_candidates - len(candidates)
    if balance_budget > 0:
        archetype_counts = {}
        for c in candidates:
            a = c.get("archetype", "unknown")
            archetype_counts[a] = archetype_counts.get(a, 0) + 1

        all_archetypes = set(e.archetype for e in _entry_genes)
        min_per_archetype = max(20, balance_budget // len(all_archetypes))

        # Boost underrepresented archetypes that have structural disadvantages
        _BOOST_ARCHETYPES = {"gex_momentum", "gex_mean_reversion", "vol_regime", "vwap_fade"}
        boost_factor = 3  # 3x the minimum for these archetypes

        before = len(candidates)
        for archetype in all_archetypes:
            current = archetype_counts.get(archetype, 0)
            target = min_per_archetype * boost_factor if archetype in _BOOST_ARCHETYPES else min_per_archetype
            needed = target - current
            if needed <= 0:
                continue

            matching_entries = [e for e in _entry_genes if e.archetype == archetype]
            if not matching_entries:
                continue

            count = 0
            for _ in range(needed * 5):
                if count >= needed:
                    break
                entry = rng.choice(matching_entries)
                direction = entry.implied_direction if entry.implied_direction != "either" else rng.choice(["long", "short"])
                confirmation = rng.choice(CONFIRMATION_GENES) if rng.random() < 0.5 else None
                _non_none_filters = [f for f in _filter_genes if f is not None]
                filter1 = rng.choice(_non_none_filters)  # always at least 1 filter
                filter2 = None

                if not is_valid_combo(entry, confirmation, filter1, filter2, direction):
                    continue

                time_window = rng.choice(_time_windows)
                tw_end = time_window["end"]
                _ts = OVERNIGHT_TIME_STOPS.get(tw_end, tw_end) if overnight else "15:45"
                exit_gene = _pick_exit_gene(rng, _exit_genes, entry.level, meta, time_stop=_ts, overnight=overnight)
                day_filters = rng.choice(DAY_FILTER_SETS)

                strat = _assemble_strategy(entry, confirmation, filter1, filter2,
                                          exit_gene, direction, time_window, day_filters)
                _add(strat)
                count += 1

        print(f"  [genes] Balance pass: +{len(candidates) - before} candidates")

    # Cap at max
    if len(candidates) > max_candidates:
        rng.shuffle(candidates)
        candidates = candidates[:max_candidates]

    # Final stats
    archetype_dist = {}
    for c in candidates:
        a = c.get("archetype", "unknown")
        archetype_dist[a] = archetype_dist.get(a, 0) + 1

    print(f"  [genes] Total: {len(candidates)} candidates")
    print(f"  [genes] Archetype distribution: {dict(sorted(archetype_dist.items()))}")

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SEARCH FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_gene_search(
    cache,       # DataCache
    start_date: str,
    end_date: str,
    window_days: int = 30,
    max_candidates: int = 5000,
    top_n: int = 1,
    n_refinements: int = 10,
    prior_results: list[dict] | None = None,
    failed_archetypes: list[str] | None = None,
    meta=None,
    overnight: bool = False,
):
    """Systematic gene-combinatorial search with LLM refinement of top candidates.

    1. Sample max_candidates gene combinations (3-pass: coverage + random + balance)
    2. Train/val split (2/3 train, 1/3 val)
    3. Backtest all on TRAIN — parallel
    4. Filter: fitness > 0 and n_trades >= 3
    5. Validate top 50 on VAL — parallel
    6. Rank by adjusted fitness
    7. LLM-refine top 3 → n_refinements variants each
    8. Return top_n winners

    Returns list[SearchResult] — same type as run_seed_search().
    """
    import time as tm
    import pandas as pd
    import mes_backtest as mb
    from strategy_dsl import StrategyDefinition
    from overfit_search import (
        _parallel_eval_strategies, run_strategy_candidate,
        SearchResult, ParamSet,
    )
    from strategy_generator import refine_seed, build_market_context

    if failed_archetypes is None:
        failed_archetypes = []

    ac = cache.asset_config

    # Train/validation split
    train_days = int(window_days * 2 / 3)
    train_start = start_date
    train_end_ts = pd.Timestamp(start_date) + pd.Timedelta(days=train_days)
    train_end = train_end_ts.strftime("%Y-%m-%d")
    val_start_ts = train_end_ts + pd.Timedelta(days=1)
    val_start = val_start_ts.strftime("%Y-%m-%d")
    val_end = end_date
    val_days = max(1, window_days - train_days - 1)

    _mode_label = "OVERNIGHT" if overnight else "RTH"
    print(f"\n  [GENE COMBINATOR ({_mode_label})] {start_date} → {end_date} ({window_days}d)")
    print(f"  Train/val split: {train_start}→{train_end} | {val_start}→{val_end}")

    # Slice data
    c1m, c15m, c1h = cache.slice_window(start_date, end_date)
    snaps_15m, ts_15m_idx = mb.precompute_gap_index(c15m, body_fill_first=False)
    precomputed = (snaps_15m, ts_15m_idx)

    # 1. Sample candidates
    t0 = tm.time()
    candidate_dicts = sample_candidates(max_candidates, meta=meta, overnight=overnight)
    print(f"  [TIMER] Sampling: {tm.time() - t0:.1f}s")

    # Convert to StrategyDefinition objects
    strategies = []
    valid_indices = []
    for i, cd in enumerate(candidate_dicts):
        try:
            strategies.append(StrategyDefinition.from_dict(cd))
            valid_indices.append(i)
        except Exception:
            continue

    # TSL/BE diagnostic: count at sampling stage
    _n_tsl_sampled = sum(1 for cd in candidate_dicts
                         if cd.get("exit", {}).get("be_trigger_pts") or cd.get("exit", {}).get("trail_distance_pts"))
    print(f"  {len(strategies)} valid strategies (of {len(candidate_dicts)} sampled)")
    print(f"  [TSL/BE] At sampling: {_n_tsl_sampled}/{len(candidate_dicts)} "
          f"({_n_tsl_sampled/max(len(candidate_dicts),1)*100:.0f}%) have TSL/BE")

    # 2. Backtest all on TRAIN — parallel
    t0 = tm.time()
    print(f"  Backtesting {len(strategies)} candidates on train split...")
    train_evals = _parallel_eval_strategies(
        strategies, c1m, c15m, c1h, precomputed, train_days,
        train_start, train_end, asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
        overnight=overnight,
    )
    print(f"  [TIMER] Train backtests: {tm.time() - t0:.1f}s ({len(train_evals)} results)")

    # 3. Build results, filter by fitness > 0 and min trade count from metaconfig
    _min_train_trades = meta.min_trade_count if meta else 3
    results = []
    for idx, fitness, summary in train_evals:
        if fitness.fitness <= 0 or fitness.n_trades < _min_train_trades:
            continue
        orig_idx = valid_indices[idx]
        sr = SearchResult(rank=0, params=ParamSet(), fitness=fitness, summary=summary)
        sr.summary["strategy_def"] = candidate_dicts[orig_idx]
        sr.summary["archetype"] = candidate_dicts[orig_idx].get("archetype", "unknown")
        results.append(sr)

    # TSL/BE diagnostic: count after train filter
    _n_tsl_train = sum(1 for r in results
                       if r.summary.get("strategy_def", {}).get("exit", {}).get("be_trigger_pts")
                       or r.summary.get("strategy_def", {}).get("exit", {}).get("trail_distance_pts"))
    print(f"  {len(results)} passed train filter (fitness > 0, trades >= 3)")
    print(f"  [TSL/BE] After train filter: {_n_tsl_train}/{len(results)} "
          f"({_n_tsl_train/max(len(results),1)*100:.0f}%) have TSL/BE")

    if not results:
        print("  No candidates passed training filter.")
        return []

    # Sort by train fitness
    results.sort(key=lambda r: r.fitness.fitness, reverse=True)

    # TSL/BE diagnostic: count in top 50
    _top50 = results[:min(50, len(results))]
    _n_tsl_top50 = sum(1 for r in _top50
                       if r.summary.get("strategy_def", {}).get("exit", {}).get("be_trigger_pts")
                       or r.summary.get("strategy_def", {}).get("exit", {}).get("trail_distance_pts"))
    print(f"  [TSL/BE] In top 50 by train fitness: {_n_tsl_top50}/{len(_top50)} "
          f"({_n_tsl_top50/max(len(_top50),1)*100:.0f}%)")
    # Show fitness comparison
    _tsl_fitnesses = [r.fitness.fitness for r in _top50
                      if r.summary.get("strategy_def", {}).get("exit", {}).get("be_trigger_pts")
                      or r.summary.get("strategy_def", {}).get("exit", {}).get("trail_distance_pts")]
    _notsl_fitnesses = [r.fitness.fitness for r in _top50
                        if not (r.summary.get("strategy_def", {}).get("exit", {}).get("be_trigger_pts")
                                or r.summary.get("strategy_def", {}).get("exit", {}).get("trail_distance_pts"))]
    if _tsl_fitnesses:
        print(f"  [TSL/BE] Top-50 fitness: TSL/BE med={sorted(_tsl_fitnesses)[len(_tsl_fitnesses)//2]:.3f} "
              f"vs plain med={sorted(_notsl_fitnesses)[len(_notsl_fitnesses)//2]:.3f}" if _notsl_fitnesses else "")

    # 4. Validate top 50 on holdout
    top_for_val = min(50, len(results))
    val_strats = []
    val_map = []
    for i in range(top_for_val):
        try:
            sd = StrategyDefinition.from_dict(results[i].summary["strategy_def"])
            val_strats.append(sd)
            val_map.append(i)
        except Exception:
            continue

    t0 = tm.time()
    print(f"  Validating top {len(val_strats)} on holdout...")
    val_evals = _parallel_eval_strategies(
        val_strats, c1m, c15m, c1h, precomputed, val_days,
        val_start, val_end, asset_config=ac,
        gex_cache=cache.gex_cache,
        daily_cache=cache.daily_cache,
        overnight=overnight,
    )
    print(f"  [TIMER] Validation backtests: {tm.time() - t0:.1f}s")

    val_lookup = {}
    for idx, val_fitness, _ in val_evals:
        val_lookup[val_map[idx]] = val_fitness

    # 5. Score: reject OOS losers, penalize failed archetypes
    _min_val_trades = 4 if overnight else 2
    validated = []
    for i in range(top_for_val):
        val_fitness = val_lookup.get(i)
        if val_fitness is None:
            continue
        r = results[i]
        archetype = r.summary.get("archetype", "unknown")
        strat_name = r.summary.get("strategy_name", "?")

        if val_fitness.total_pnl <= 0:
            print(f"    REJECTED (OOS): {strat_name} [{archetype}] "
                  f"val P&L=${val_fitness.total_pnl:,.0f}")
            continue

        if val_fitness.n_trades < _min_val_trades:
            print(f"    REJECTED (min val trades): {strat_name} [{archetype}] "
                  f"val trades={val_fitness.n_trades} < {_min_val_trades}")
            continue

        # Expectancy gate: avg P&L per trade must show real edge
        val_expectancy = val_fitness.total_pnl / val_fitness.n_trades
        if val_expectancy < 10:
            print(f"    REJECTED (expectancy): {strat_name} [{archetype}] "
                  f"val expectancy=${val_expectancy:,.1f}/trade")
            continue

        # Profit factor gate
        if val_fitness.profit_factor < 1.1:
            print(f"    REJECTED (PF): {strat_name} [{archetype}] "
                  f"val PF={val_fitness.profit_factor:.2f} < 1.1")
            continue

        archetype_penalty = 0.0
        fail_count = failed_archetypes.count(archetype)
        if fail_count > 0:
            archetype_penalty = fail_count * 0.5

        r.summary["val_pnl"] = val_fitness.total_pnl
        r.summary["val_fitness"] = val_fitness.fitness
        r.summary["adjusted_fitness"] = (
            r.fitness.fitness + val_fitness.fitness * 0.3 - archetype_penalty
        )
        validated.append(r)
        print(f"    PASSED: {strat_name} [{archetype}] "
              f"train=${r.fitness.total_pnl:,.0f} val=${val_fitness.total_pnl:,.0f} "
              f"expectancy=${val_expectancy:,.1f}/trade PF={val_fitness.profit_factor:.2f} "
              f"adj={r.summary['adjusted_fitness']:.3f}")

    # TSL/BE diagnostic: count after validation
    _n_tsl_val = sum(1 for r in validated
                     if r.summary.get("strategy_def", {}).get("exit", {}).get("be_trigger_pts")
                     or r.summary.get("strategy_def", {}).get("exit", {}).get("trail_distance_pts"))
    print(f"  {len(validated)} passed validation (OOS profitable)")
    print(f"  [TSL/BE] After validation: {_n_tsl_val}/{len(validated)} "
          f"({_n_tsl_val/max(len(validated),1)*100:.0f}%) have TSL/BE")

    if not validated:
        print("  No candidates passed validation. Skipping regime (no unvalidated fallback).")
        return []

    validated.sort(key=lambda r: r.summary.get("adjusted_fitness", 0), reverse=True)

    # Print top 5
    for i, r in enumerate(validated[:5]):
        print(f"    #{i+1}: {r.summary.get('strategy_name', '?')} [{r.summary['archetype']}] "
              f"train=${r.fitness.total_pnl:,.0f} val=${r.summary['val_pnl']:,.0f} "
              f"adj={r.summary['adjusted_fitness']:.3f}")

    # 6. LLM refinement of top 2 (gene search already has broad coverage)
    #    Parallelized so both vLLM workers get used simultaneously
    if n_refinements > 0:
        t0 = tm.time()
        context = build_market_context(c1m, start_date, end_date, asset_config=ac)

        top_for_refine = min(2, len(validated))
        all_variants = []

        from overfit_search import compute_regime_mfe_mae_context, _UNCENSORED_RECORDS

        # Build regime context for LLM prompt enrichment
        _regime_ctx = {}
        if cache.gex_cache and start_date:
            try:
                from datetime import date as _date, time as _time
                _opt_d = _date.fromisoformat(start_date)
                _snap = cache.gex_cache.get(_opt_d, _time(10, 0))
                if _snap:
                    _regime_ctx["gex_regime"] = _snap.gex_regime
            except Exception:
                pass
        # Top 3 archetypes by adjusted_fitness from current window
        _arch_fitness = {}
        for r in results:
            a = r.summary.get("archetype", "unknown")
            af = r.summary.get("adjusted_fitness", r.fitness.fitness)
            if a not in _arch_fitness or af > _arch_fitness[a]:
                _arch_fitness[a] = af
        _regime_ctx["top_archetypes"] = sorted(
            [{"archetype": a, "fitness": f} for a, f in _arch_fitness.items()],
            key=lambda x: x["fitness"], reverse=True
        )[:3]

        def _refine_one(r):
            """Refine a single candidate — runs in thread."""
            strat_dict = r.summary["strategy_def"]
            archetype = r.summary["archetype"]
            strat = StrategyDefinition.from_dict(strat_dict)
            mfe_mae_ctx = compute_regime_mfe_mae_context(results, archetype,
                                                          uncensored_records=_UNCENSORED_RECORDS)
            try:
                variants = refine_seed(
                    strat, r.fitness.to_dict(), context,
                    archetype=archetype,
                    n_variants=n_refinements,
                    prior_results=prior_results,
                    asset_config=ac,
                    mfe_mae_context=mfe_mae_ctx,
                    regime_context=_regime_ctx,
                )
                return [(var, archetype) for var in variants]
            except Exception as e:
                print(f"    LLM refinement failed for {strat.name}: {e}")
                return []

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=top_for_refine) as pool:
            futures = [pool.submit(_refine_one, r) for r in validated[:top_for_refine]]
            for fut in futures:
                all_variants.extend(fut.result())

        if all_variants:
            print(f"  Backtesting {len(all_variants)} LLM refinements...")
            variant_strats = [v[0] for v in all_variants]
            ref_evals = _parallel_eval_strategies(
                variant_strats, c1m, c15m, c1h, precomputed, train_days,
                train_start, train_end, asset_config=ac,
                gex_cache=cache.gex_cache,
                daily_cache=cache.daily_cache,
                overnight=overnight,
            )

            for idx, fitness, summary in ref_evals:
                if fitness.fitness <= 0:
                    continue
                ref_sr = SearchResult(rank=0, params=ParamSet(), fitness=fitness, summary=summary)
                ref_sr.summary["strategy_def"] = all_variants[idx][0].to_dict()
                ref_sr.summary["archetype"] = all_variants[idx][1]

                # Quick validate on holdout
                try:
                    sd = StrategyDefinition.from_dict(ref_sr.summary["strategy_def"])
                    vf, _vs = run_strategy_candidate(
                        sd, c1m, c15m, c1h, val_days,
                        precomputed=precomputed,
                        start_date=val_start, end_date=val_end,
                        asset_config=ac,
                        gex_cache=cache.gex_cache,
                        daily_cache=cache.daily_cache,
                        overnight=overnight,
                    )
                    if vf.total_pnl > 0:
                        penalty = failed_archetypes.count(all_variants[idx][1]) * 0.5
                        ref_sr.summary["val_pnl"] = vf.total_pnl
                        ref_sr.summary["val_fitness"] = vf.fitness
                        ref_sr.summary["adjusted_fitness"] = (
                            fitness.fitness + vf.fitness * 0.3 - penalty
                        )
                        validated.append(ref_sr)
                except Exception:
                    pass

            print(f"  [TIMER] LLM refinement: {tm.time() - t0:.1f}s")

    # Final ranking
    validated.sort(key=lambda r: r.summary.get("adjusted_fitness", 0), reverse=True)
    for i, r in enumerate(validated):
        r.rank = i + 1

    # Print winner
    if validated:
        w = validated[0]
        print(f"\n  GENE WINNER: {w.summary.get('strategy_name', '?')} [{w.summary['archetype']}]")
        print(f"    Train: ${w.fitness.total_pnl:,.0f} | Val: ${w.summary.get('val_pnl', 0):,.0f} | "
              f"WR: {w.fitness.win_rate:.1f}% | Trades: {w.fitness.n_trades} | "
              f"Adj fitness: {w.summary.get('adjusted_fitness', 0):.3f}")

    # ── Log ALL candidates (winners + losers) for meta-analysis ──
    _log_candidates(results, validated, start_date, end_date)

    # ── Attach top-50 candidate dicts to winner for pipeline transport ──
    validated_set = {id(r) for r in validated}
    top50_dicts = []
    for r in results[:50]:
        top50_dicts.append(_candidate_to_dict(r, passed_val=id(r) in validated_set))
    for r in validated:
        if id(r) not in {id(x) for x in results[:50]}:
            top50_dicts.append(_candidate_to_dict(r, passed_val=True))
    if validated:
        validated[0].summary["_candidates_top50"] = top50_dicts

    return validated[:top_n]


def _log_candidates(
    all_results: list,
    validated: list,
    start_date: str,
    end_date: str,
) -> None:
    """Append candidate records (winners + losers) to candidate_logs.jsonl.

    Each candidate is one JSON line with batch-level metadata (log_timestamp,
    optimize_start, search_mode) plus all per-candidate fields.
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    validated_set = {id(r) for r in validated}

    # Collect top 50 train-ranked + all validated (deduplicated)
    logged = {}  # id(r) -> entry
    for r in all_results[:50]:
        logged[id(r)] = _candidate_to_dict(r, passed_val=id(r) in validated_set)
    for r in validated:
        if id(r) not in logged:
            logged[id(r)] = _candidate_to_dict(r, passed_val=True)

    # Write one JSONL line per candidate
    jsonl_path = Path(__file__).parent / "results" / "candidate_logs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(jsonl_path, "a") as f:
            for entry in logged.values():
                record = {
                    "log_timestamp": ts,
                    "optimize_start": start_date,
                    "search_mode": "gene",
                    **entry,
                }
                f.write(json.dumps(record, default=str) + "\n")
        print(f"  [LOG] Appended {len(logged)} candidates to candidate_logs.jsonl")
    except Exception as e:
        print(f"  [LOG] Failed to append candidates: {e}")


def _trajectory_fields(pnls: list) -> dict:
    """Compute trajectory metrics from a pnls array."""
    if not pnls or len(pnls) < 6:
        return {"early_pnl": None, "late_pnl": None, "trajectory_score": None, "frontloaded": None}
    n = len(pnls)
    n3 = max(n // 3, 1)
    early_pnl = round(sum(pnls[:n3]), 2)
    late_pnl = round(sum(pnls[2 * n3:]), 2)
    activity = abs(early_pnl) + abs(late_pnl) + 1.0
    trajectory = (late_pnl - early_pnl) / activity
    trajectory_score = round(max(trajectory, 0.0), 4)
    frontloaded = bool(early_pnl > late_pnl and early_pnl > 0)
    return {"early_pnl": early_pnl, "late_pnl": late_pnl,
            "trajectory_score": trajectory_score, "frontloaded": frontloaded}


def _candidate_to_dict(r, passed_val: bool) -> dict:
    """Convert a SearchResult to a serializable dict for logging."""
    sd = r.summary.get("strategy_def", {})
    exit_d = sd.get("exit", {})
    pnls = r.summary.get("pnls_array", [])
    traj = _trajectory_fields(pnls)
    rob = r.summary.get("robustness", {}) or {}
    return {
        "rank": r.rank,
        "archetype": r.summary.get("archetype", "unknown"),
        "strategy_name": r.summary.get("strategy_name", "?"),
        "passed_validation": passed_val,
        "train_fitness": r.fitness.fitness,
        "train_sharpe": r.fitness.sharpe,
        "train_pf": r.fitness.profit_factor,
        "train_pnl": r.fitness.total_pnl,
        "train_max_dd": r.fitness.max_dd,
        "train_wr": r.fitness.win_rate,
        "train_n_trades": r.fitness.n_trades,
        "train_avg_pnl": r.fitness.avg_trade_pnl,
        "val_pnl": r.summary.get("val_pnl"),
        "val_fitness": r.summary.get("val_fitness"),
        "adjusted_fitness": r.summary.get("adjusted_fitness"),
        "has_tsl_be": bool(exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts")),
        "exit_type": ("tsl_be" if (exit_d.get("be_trigger_pts") or exit_d.get("trail_distance_pts"))
                      else "fixed_sl_tp"),
        "pnls_array": pnls,
        "mfe_array": r.summary.get("mfe_array", []),
        "mae_array": r.summary.get("mae_array", []),
        "mfe_p50": r.summary.get("mfe_p50"),
        "mfe_p75": r.summary.get("mfe_p75"),
        "mae_p50": r.summary.get("mae_p50"),
        "mae_p75": r.summary.get("mae_p75"),
        "mfe_mae_ratio": r.summary.get("mfe_mae_ratio"),
        "bars_to_mfe_median": r.summary.get("bars_to_mfe_median"),
        "bars_to_mae_median": r.summary.get("bars_to_mae_median"),
        "mfe_before_mae_pct": r.summary.get("mfe_before_mae_pct"),
        "mfe_before_mae_raw": r.summary.get("mfe_before_mae_raw", []),
        **traj,
        # Robustness metrics (previously missing from candidate_logs)
        "win_loss_ratio": rob.get("win_loss_ratio"),
        "expectancy_per_trade": rob.get("expectancy_per_trade"),
        "kelly_fraction": rob.get("kelly_fraction"),
        "trades_per_day": rob.get("trades_per_day"),
        "mc_p_value": rob.get("mc_p_value"),
        "ttest_p_value": rob.get("ttest_p_value"),
        "pf_stability_ratio": rob.get("pf_stability_ratio"),
        "top_trade_pct": rob.get("top_trade_pct"),
        "remove_best_still_positive": rob.get("remove_best_still_positive"),
        "wr_stability_ratio": rob.get("wr_stability_ratio"),
        "payoff_consistency": rob.get("payoff_consistency"),
        "strategy_def": sd,
    }
