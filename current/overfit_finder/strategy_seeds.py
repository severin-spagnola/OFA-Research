"""
Strategy Seeds — Curated Strategy Archetypes for MES Intraday
=============================================================
25 strategy seeds expressed in the existing DSL format. Each is a
complete, runnable strategy with sensible defaults. The LLM tunes
parameters on these structures rather than inventing from scratch.

Seeds 3-5 killed based on 73-regime walk-forward analysis (22% and 14%
forward win rates). 11 active price-action seeds + 6 GEX/flow seeds
+ 5 daily-context seeds (VWAP, realized vol, earnings).

Archetypes: prev_close, smc_sweep, on_breakout, mean_reversion,
day_open, gex_mean_reversion, gex_momentum, vwap_fade,
vol_regime, earnings_catalyst.

GEX seeds use options flow conditions (gex_regime_is, price_near_wall,
iv_above, pc_ratio_above). They require GexContext in CandleContext
and gracefully return no trades on assets without chain data.

Daily-context seeds use VWAP, realized vol, and earnings conditions.
They require DailyContext in CandleContext.
"""
from __future__ import annotations

from strategy_dsl import StrategyDefinition


# ─── Seed 1: ON High Breakout (Long) ────────────────────────────────────────
ON_HIGH_BREAKOUT = {
    "name": "ON High Breakout",
    "description": "Long when price breaks above overnight high with pullback confirmation",
    "archetype": "on_breakout",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "price_broke", "level": "on_high", "lookback_bars": 3, "direction": "above"},
            {"type": "pullback_to", "reference": "on_high", "buffer_pts": 3.0},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_high", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
        {"type": "max_on_range", "value": 50},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 2: ON Low Breakdown (Short) ───────────────────────────────────────
ON_LOW_BREAKDOWN = {
    "name": "ON Low Breakdown",
    "description": "Short when price breaks below overnight low with pullback confirmation",
    "archetype": "on_breakout",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "price_broke", "level": "on_low", "lookback_bars": 3, "direction": "below"},
            {"type": "pullback_to", "reference": "on_low", "buffer_pts": 3.0},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_low", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
        {"type": "max_on_range", "value": 50},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 3: Gap Fill Fade High (Short) ─────────────────────────────────────
GAP_FILL_FADE_HIGH = {
    "name": "Gap Fill Fade High",
    "description": "Short fade when unfilled gap exists near overnight high and price is extended above",
    "archetype": "gap_fill_fade",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "gap_exists", "location": "near_on_high", "max_distance_pts": 12.0, "min_size_pts": 4.0},
            {"type": "price_above", "level": "on_high"},
            {"type": "candle_pattern", "pattern": "large_wick_up", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 12},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 4: Gap Fill Fade Low (Long) ───────────────────────────────────────
GAP_FILL_FADE_LOW = {
    "name": "Gap Fill Fade Low",
    "description": "Long fade when unfilled gap exists near overnight low and price is extended below",
    "archetype": "gap_fill_fade",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "gap_exists", "location": "near_on_low", "max_distance_pts": 12.0, "min_size_pts": 4.0},
            {"type": "price_below", "level": "on_low"},
            {"type": "candle_pattern", "pattern": "large_wick_down", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 12},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 5: First Bar Momentum ─────────────────────────────────────────────
FIRST_BAR_MOMENTUM = {
    "name": "First Bar Momentum",
    "description": "Trade direction of a strong first 15m bar with minimum body size",
    "archetype": "first_bar",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "09:30", "end": "09:45"},
        "conditions": [
            {"type": "bar_index", "value": 0, "op": "eq"},
            {"type": "min_candle_size", "size_pts": 8.0},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 5.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "candle_wick", "buffer_pts": 2.0},
        "take_profit": {"type": "risk_multiple", "value": 1.2},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 8},
        {"type": "max_on_range", "value": 45},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 6: Previous Close Rejection ───────────────────────────────────────
PREV_CLOSE_REJECTION = {
    "name": "Previous Close Rejection",
    "description": "Fade rejection off previous close level with candle confirmation",
    "archetype": "prev_close",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "09:30", "end": "11:00"},
        "conditions": [
            {"type": "pullback_to", "reference": "prev_close", "buffer_pts": 2.5},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 8},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 7: ON Mid Reversion ───────────────────────────────────────────────
ON_MID_REVERSION = {
    "name": "ON Mid Reversion",
    "description": "Mean reversion trade when price returns to overnight midpoint",
    "archetype": "mean_reversion",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "10:00", "end": "14:00"},
        "conditions": [
            {"type": "pullback_to", "reference": "on_mid", "buffer_pts": 2.0},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 8},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 15},
        {"type": "max_on_range", "value": 45},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 8: Afternoon ON Sweep (SMC-style) ─────────────────────────────────
AFTERNOON_ON_SWEEP = {
    "name": "Afternoon ON Sweep",
    "description": "Fade afternoon liquidity sweep above ON high with bearish reversal candle",
    "archetype": "smc_sweep",
    "entry": {
        "direction": "short",
        "time_window": {"start": "13:00", "end": "15:30"},
        "conditions": [
            {"type": "price_broke", "level": "on_high", "lookback_bars": 2, "direction": "above"},
            {"type": "price_below", "level": "on_high"},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 8},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}


# ─── Seed 9: Prev Close Breakout (Long) ──────────────────────────────────────
PREV_CLOSE_BREAKOUT_LONG = {
    "name": "Prev Close Breakout Long",
    "description": "Long when price breaks above prev close and pulls back to retest it",
    "archetype": "prev_close",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "price_broke", "level": "prev_close", "lookback_bars": 3, "direction": "above"},
            {"type": "pullback_to", "reference": "prev_close", "buffer_pts": 2.0},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "prev_close", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 8},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 10: Prev Close Breakout (Short) ─────────────────────────────────────
PREV_CLOSE_BREAKOUT_SHORT = {
    "name": "Prev Close Breakout Short",
    "description": "Short when price breaks below prev close and pulls back to retest it",
    "archetype": "prev_close",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "price_broke", "level": "prev_close", "lookback_bars": 3, "direction": "below"},
            {"type": "pullback_to", "reference": "prev_close", "buffer_pts": 2.0},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "prev_close", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 8},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 11: Afternoon ON Low Sweep (Long) ──────────────────────────────────
AFTERNOON_ON_LOW_SWEEP = {
    "name": "Afternoon ON Low Sweep Long",
    "description": "Fade afternoon liquidity sweep below ON low with bullish reversal candle",
    "archetype": "smc_sweep",
    "entry": {
        "direction": "long",
        "time_window": {"start": "13:00", "end": "15:30"},
        "conditions": [
            {"type": "price_broke", "level": "on_low", "lookback_bars": 2, "direction": "below"},
            {"type": "price_above", "level": "on_low"},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 8},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 12: Day Open Rejection Fade ────────────────────────────────────────
DAY_OPEN_REJECTION = {
    "name": "Day Open Rejection Fade",
    "description": "Fade rejection off the RTH open price with candle confirmation",
    "archetype": "day_open",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "pullback_to", "reference": "day_open", "buffer_pts": 2.0},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 8},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 13: Day Open Breakout (Long) ───────────────────────────────────────
DAY_OPEN_BREAKOUT_LONG = {
    "name": "Day Open Breakout Long",
    "description": "Long when price breaks above day open and retests it with bullish confirmation",
    "archetype": "day_open",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:45", "end": "11:00"},
        "conditions": [
            {"type": "price_broke", "level": "day_open", "lookback_bars": 3, "direction": "above"},
            {"type": "pullback_to", "reference": "day_open", "buffer_pts": 2.5},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "day_open", "buffer_pts": 3.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 8},
        {"type": "max_on_range", "value": 50},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 14: Day Open Breakdown (Short) ─────────────────────────────────────
DAY_OPEN_BREAKDOWN_SHORT = {
    "name": "Day Open Breakdown Short",
    "description": "Short when price breaks below day open and retests it with bearish confirmation",
    "archetype": "day_open",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:45", "end": "11:00"},
        "conditions": [
            {"type": "price_broke", "level": "day_open", "lookback_bars": 3, "direction": "below"},
            {"type": "pullback_to", "reference": "day_open", "buffer_pts": 2.5},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "day_open", "buffer_pts": 3.0},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 8},
        {"type": "max_on_range", "value": 50},
    ],
    "position_size": {"risk_dollars": 600},
}


# ═══════════════════════════════════════════════════════════════════════════════
# GEX / Options Flow Seeds
# ═══════════════════════════════════════════════════════════════════════════════
# These seeds use GEX regime, put/call walls, IV, and P/C ratio conditions.
# They only fire on assets with GexContext (currently MES via SPY chain data).

# ─── Seed 15: GEX Fade to Put Wall (Long) ─────────────────────────────────────
GEX_FADE_PUT_WALL = {
    "name": "GEX Fade to Put Wall",
    "description": "Long near put wall in positive GEX regime — dealers dampen selloff, bounce expected",
    "archetype": "gex_mean_reversion",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "gex_regime_is", "regime": "positive"},
            {"type": "price_near_wall", "wall": "put", "within_pts": 8.0},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 16: GEX Fade to Call Wall (Short) ───────────────────────────────────
GEX_FADE_CALL_WALL = {
    "name": "GEX Fade to Call Wall",
    "description": "Short near call wall in positive GEX regime — dealers dampen rally, fade expected",
    "archetype": "gex_mean_reversion",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "gex_regime_is", "regime": "positive"},
            {"type": "price_near_wall", "wall": "call", "within_pts": 8.0},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 17: High IV Mean Reversion ──────────────────────────────────────────
HIGH_IV_MEAN_REVERSION = {
    "name": "High IV Mean Reversion",
    "description": "Fade extended moves in high IV + positive GEX — vol crush + dealer dampening",
    "archetype": "gex_mean_reversion",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "10:00", "end": "14:00"},
        "conditions": [
            {"type": "gex_regime_is", "regime": "positive"},
            {"type": "iv_above", "value": 0.25},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 5.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 12},
        "take_profit": {"type": "fixed_pts", "value": 10},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 12},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 18: GEX Momentum Breakout Long ──────────────────────────────────────
GEX_MOMENTUM_BREAKOUT_LONG = {
    "name": "GEX Momentum Breakout Long",
    "description": "Long ON high break in negative GEX — dealers amplify the move, ride momentum",
    "archetype": "gex_momentum",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "gex_regime_is", "regime": "negative"},
            {"type": "price_broke", "level": "on_high", "lookback_bars": 3, "direction": "above"},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 5.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_high", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 19: GEX Momentum Breakdown Short ────────────────────────────────────
GEX_MOMENTUM_BREAKDOWN_SHORT = {
    "name": "GEX Momentum Breakdown Short",
    "description": "Short ON low break in negative GEX — dealers amplify the selloff, ride momentum",
    "archetype": "gex_momentum",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "gex_regime_is", "regime": "negative"},
            {"type": "price_broke", "level": "on_low", "lookback_bars": 3, "direction": "below"},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 5.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_low", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 20: Skew Fade (Long) ────────────────────────────────────────────────
SKEW_FADE = {
    "name": "Skew Fade",
    "description": "Long when put skew extreme (high P/C ratio) near put wall — crowd is over-hedged, snap-back likely",
    "archetype": "gex_momentum",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:45", "end": "11:00"},
        "conditions": [
            {"type": "pc_ratio_above", "value": 1.5},
            {"type": "price_near_wall", "wall": "put", "within_pts": 10.0},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "risk_multiple", "value": 1.5},
        "time_stop": "15:45",
    },
    "filters": [],
    "position_size": {"risk_dollars": 600},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Daily Context Seeds — VWAP, Realized Vol, Earnings
# ═══════════════════════════════════════════════════════════════════════════════
# These seeds use previous day's VWAP, realized volatility, and earnings
# proximity. They require DailyContext in CandleContext.

# ─── Seed 21: VWAP Fade Long ─────────────────────────────────────────────────
VWAP_FADE_LONG = {
    "name": "VWAP Fade Long",
    "description": "Long on pullback below prev VWAP with bullish confirmation — institutions buy below VWAP",
    "archetype": "vwap_fade",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "price_below_vwap"},
            {"type": "pullback_to", "reference": "prev_vwap", "buffer_pts": 3.0},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 22: VWAP Fade Short ────────────────────────────────────────────────
VWAP_FADE_SHORT = {
    "name": "VWAP Fade Short",
    "description": "Short on rally above prev VWAP with bearish confirmation — institutions sell above VWAP",
    "archetype": "vwap_fade",
    "entry": {
        "direction": "short",
        "time_window": {"start": "09:45", "end": "11:30"},
        "conditions": [
            {"type": "price_above_vwap"},
            {"type": "pullback_to", "reference": "prev_vwap", "buffer_pts": 3.0},
            {"type": "candle_pattern", "pattern": "is_bearish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 10},
        "take_profit": {"type": "fixed_pts", "value": 12},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 10},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 23: High RV Breakout ───────────────────────────────────────────────
HIGH_RV_BREAKOUT = {
    "name": "High RV Breakout",
    "description": "Breakout above ON high when previous day's realized vol was elevated — momentum continues",
    "archetype": "vol_regime",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "realized_vol_above", "value": 0.15},
            {"type": "price_broke", "level": "on_high", "lookback_bars": 3, "direction": "above"},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 5.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_high", "buffer_pts": 4.0},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 12},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 24: Low RV Mean Reversion ──────────────────────────────────────────
LOW_RV_MEAN_REVERSION = {
    "name": "Low RV Mean Reversion",
    "description": "Fade moves to ON boundaries when prev day's vol was low — range-bound regime favors reversion",
    "archetype": "vol_regime",
    "entry": {
        "direction": "adaptive",
        "time_window": {"start": "10:00", "end": "14:00"},
        "conditions": [
            {"type": "realized_vol_below", "value": 0.10},
            {"type": "pullback_to", "reference": "on_mid", "buffer_pts": 3.0},
            {"type": "candle_pattern", "pattern": "min_body_size", "size_pts": 3.0},
        ],
    },
    "exit": {
        "stop_loss": {"type": "fixed_pts", "value": 8},
        "take_profit": {"type": "fixed_pts", "value": 8},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "max_on_range", "value": 30},
    ],
    "position_size": {"risk_dollars": 600},
}

# ─── Seed 25: Earnings Day Momentum ──────────────────────────────────────────
EARNINGS_DAY_MOMENTUM = {
    "name": "Earnings Day Momentum",
    "description": "Ride ON high breakout when major SPY components report — wider ranges and follow-through",
    "archetype": "earnings_catalyst",
    "entry": {
        "direction": "long",
        "time_window": {"start": "09:30", "end": "10:30"},
        "conditions": [
            {"type": "earnings_nearby", "min_tickers": 1},
            {"type": "price_broke", "level": "on_high", "lookback_bars": 3, "direction": "above"},
            {"type": "candle_pattern", "pattern": "is_bullish"},
        ],
    },
    "exit": {
        "stop_loss": {"type": "level", "value": "on_high", "buffer_pts": 5.0},
        "take_profit": {"type": "risk_multiple", "value": 2.0},
        "time_stop": "15:45",
    },
    "filters": [
        {"type": "min_on_range", "value": 12},
    ],
    "position_size": {"risk_dollars": 600},
}


# ─── Exports ─────────────────────────────────────────────────────────────────

STRATEGY_SEEDS: list[dict] = [
    # Price-action seeds (11 active)
    ON_HIGH_BREAKOUT,
    ON_LOW_BREAKDOWN,
    # GAP_FILL_FADE_HIGH — killed: 22% forward win rate across 17 regimes
    # GAP_FILL_FADE_LOW — killed: 22% forward win rate across 17 regimes
    # FIRST_BAR_MOMENTUM — killed: 14% forward win rate across 7 regimes
    PREV_CLOSE_REJECTION,
    ON_MID_REVERSION,
    AFTERNOON_ON_SWEEP,
    PREV_CLOSE_BREAKOUT_LONG,
    PREV_CLOSE_BREAKOUT_SHORT,
    AFTERNOON_ON_LOW_SWEEP,
    DAY_OPEN_REJECTION,
    DAY_OPEN_BREAKOUT_LONG,
    DAY_OPEN_BREAKDOWN_SHORT,
    # GEX / options flow seeds (6 active — require GexContext)
    GEX_FADE_PUT_WALL,
    GEX_FADE_CALL_WALL,
    HIGH_IV_MEAN_REVERSION,
    GEX_MOMENTUM_BREAKOUT_LONG,
    GEX_MOMENTUM_BREAKDOWN_SHORT,
    SKEW_FADE,
    # Daily context seeds (5 active — require DailyContext)
    VWAP_FADE_LONG,
    VWAP_FADE_SHORT,
    HIGH_RV_BREAKOUT,
    LOW_RV_MEAN_REVERSION,
    EARNINGS_DAY_MOMENTUM,
]


def get_seeds() -> list[StrategyDefinition]:
    """Convert all seed dicts to StrategyDefinition objects."""
    return [StrategyDefinition.from_dict(s) for s in STRATEGY_SEEDS]


def get_seed_archetype(strategy_dict: dict) -> str:
    """Extract the archetype tag from a strategy dict, or 'unknown'."""
    return strategy_dict.get("archetype", "unknown")


def scale_seeds_for_asset(asset_config) -> list[dict]:
    """Scale MES seed parameters to a different asset by price ratio.

    Point-based values (buffer_pts, size_pts, fixed_pts SL/TP, ON range filters)
    are scaled by the ratio of the asset's approximate price to MES (~5800).
    Risk multiples and time windows are not scaled.

    Args:
        asset_config: An AssetConfig instance (from asset_config.py).

    Returns:
        List of scaled seed dicts.
    """
    import copy

    MES_PRICE = 5800.0
    ratio = asset_config.price_approx / MES_PRICE

    if abs(ratio - 1.0) < 0.01:
        # MES or very similar price — no scaling needed
        return [copy.deepcopy(s) for s in STRATEGY_SEEDS]

    scaled = []
    for seed in STRATEGY_SEEDS:
        s = copy.deepcopy(seed)

        # Scale entry conditions
        for cond in s.get("entry", {}).get("conditions", []):
            if "buffer_pts" in cond:
                cond["buffer_pts"] = round(cond["buffer_pts"] * ratio, 4)
            if "size_pts" in cond:
                cond["size_pts"] = round(cond["size_pts"] * ratio, 4)
            if "max_distance_pts" in cond:
                cond["max_distance_pts"] = round(cond["max_distance_pts"] * ratio, 4)
            if "min_size_pts" in cond:
                cond["min_size_pts"] = round(cond["min_size_pts"] * ratio, 4)

        # Scale exit rules
        exit_rules = s.get("exit", {})
        sl = exit_rules.get("stop_loss", {})
        if sl.get("type") == "fixed_pts" and "value" in sl:
            sl["value"] = round(sl["value"] * ratio, 4)
        if "buffer_pts" in sl:
            sl["buffer_pts"] = round(sl["buffer_pts"] * ratio, 4)

        tp = exit_rules.get("take_profit", {})
        if tp.get("type") == "fixed_pts" and "value" in tp:
            tp["value"] = round(tp["value"] * ratio, 4)
        # risk_multiple — no scaling needed

        # Scale filters (ON range thresholds)
        for filt in s.get("filters", []):
            if filt.get("type") == "min_on_range" and "value" in filt:
                filt["value"] = round(filt["value"] * ratio, 4)
            if filt.get("type") == "max_on_range" and "value" in filt:
                filt["value"] = round(filt["value"] * ratio, 4)

        scaled.append(s)

    return scaled
