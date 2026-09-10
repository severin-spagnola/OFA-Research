"""
Strategy Generator — LLM-Powered Strategy Creation
====================================================
Uses Claude Sonnet (or GPT-4o fallback) to generate complete strategy
definitions as JSON. Strategies are expressed in the DSL format defined
in strategy_dsl.py.

The LLM receives:
  1. A system prompt explaining MES futures, available condition types, and the JSON schema
  2. Market context (summary stats for the optimization window)
  3. Optionally, feedback from prior regime results

Usage:
    # Called by overfit_search.py --llm-search
    # Or standalone:
    python strategy_generator.py --context "trending up, avg ON range 18pts, 60 trading days"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_SCRIPT_DIR = Path(__file__).parent

from llm_variant import call_llm, parse_variants
from strategy_dsl import StrategyDefinition

if TYPE_CHECKING:
    from asset_config import AssetConfig


# ─── System Prompt Template ──────────────────────────────────────────────────

def _build_system_prompt(ac: AssetConfig | None = None) -> str:
    """Build the strategy generation system prompt for any asset."""
    if ac is None:
        from asset_config import MES_CONFIG
        ac = MES_CONFIG

    return f"""\
You are a quantitative trading strategy designer for {ac.name} ({ac.ticker}).

## Instrument
- {ac.ticker} = {ac.name}
- {ac.instrument_details}
- Commissions: ~${ac.round_trip_fee:.2f} round-trip per {'contract' if ac.asset_type == 'futures' else 'share'}
- Slippage: ~{ac.slippage_per_side} pts per side
- We trade 1-{ac.max_contracts} {'contracts' if ac.asset_type == 'futures' else 'shares'} per position, ${ac.risk_budget:.0f} risk budget per trade
- RTH session: {ac.rth_start.strftime('%H:%M')}–{ac.rth_end.strftime('%H:%M')} ET
- {ac.overnight_details}

## Available Data
- 1-minute candle bars (open/high/low/close/volume)
- 15-minute candle bars (aggregated from 1m)
- ON_HIGH and ON_LOW — {'overnight session high/low (18:00-09:30 ET)' if ac.asset_type == 'futures' else "previous RTH day's high and low"}
- Day open (first RTH bar's open)
- Previous day's close
- Unfilled gaps (list of price gaps that haven't been fully retraced)
- GEX regime (positive/negative/neutral) — from SPY options volume-weighted gamma
- Call wall / put wall — highest volume strikes acting as support/resistance
- ATM implied volatility and put/call volume ratio
- Previous day's VWAP (volume-weighted average price) — key institutional level
- Previous day's realized volatility (5-min and 30-min, annualized)
- Earnings proximity — which top-20 SPY components report within 2 days

## Strategy JSON Format

Output strategies as a JSON array. Each strategy is:

```json
{{
  "name": "Short descriptive name",
  "description": "One sentence explaining the thesis",
  "entry": {{
    "direction": "long" | "short" | "adaptive",
    "time_window": {{"start": "HH:MM", "end": "HH:MM"}},
    "conditions": [
      {{"type": "condition_type", ...params}}
    ]
  }},
  "exit": {{
    "stop_loss": {{"type": "fixed_pts|candle_wick|level", "value": ...}},
    "take_profit": {{"type": "risk_multiple|fixed_pts|level", "value": ...}},
    "time_stop": "15:45"
  }},
  "filters": [
    {{"type": "filter_type", ...params}}
  ],
  "position_size": {{"risk_dollars": {ac.risk_budget:.0f}}}
}}
```

## Supported Condition Types

### Entry Conditions (all must be true — AND logic)

1. **price_above** / **price_below**
   `{{"type": "price_above", "level": "on_high|on_low|day_open|prev_close|on_mid|prev_vwap|<number>"}}`
   Current 15m candle close vs a reference level.

2. **price_broke**
   `{{"type": "price_broke", "level": "on_high|on_low|...", "lookback_bars": 2, "direction": "above|below"}}`
   Price crossed a level within the last N 15m bars.

3. **pullback_to**
   `{{"type": "pullback_to", "reference": "on_high|on_low|day_open|breakout_candle_body_bottom|...", "buffer_pts": {ac.default_sl/4:.1f}}}`
   Current price is within buffer_pts of a reference level (retracement).

4. **candle_pattern**
   `{{"type": "candle_pattern", "pattern": "is_bullish|is_bearish|min_body_size|max_body_size|large_wick_up|large_wick_down", "size_pts": {ac.default_sl/2:.1f}}}`
   Current 15m candle shape.

5. **min_candle_size**
   `{{"type": "min_candle_size", "size_pts": {ac.default_sl:.1f}}}`
   Current 15m candle body must be at least this many points.

6. **gap_exists**
   `{{"type": "gap_exists", "location": "any|near_on_high|near_on_low", "max_distance_pts": {ac.sl_max/2:.1f}, "min_size_pts": {ac.sl_min:.1f}}}`
   An unfilled gap exists near a reference level.

7. **time_is**
   `{{"type": "time_is", "after": "09:30", "before": "11:00"}}`
   Current time must be within range.

8. **bar_index**
   `{{"type": "bar_index", "value": 0, "op": "eq|gte|lte"}}`
   Position within the RTH session (0 = first 15m bar = 09:30).

9. **on_range_between**
   `{{"type": "on_range_between", "min": {ac.sl_min*2.5:.0f}, "max": {ac.sl_max*1.5:.0f}}}`
   Overnight range size filter as a condition.

10. **gex_regime_is**
    `{{"type": "gex_regime_is", "regime": "positive|negative|neutral"}}`
    GEX (Gamma Exposure) regime classification from options flow data.
    Positive = dealers long gamma, they dampen moves (mean-reversion regime).
    Negative = dealers short gamma, they amplify moves (momentum regime).
    Only available for S&P-linked assets (MES). Returns false on others.

11. **price_near_wall**
    `{{"type": "price_near_wall", "wall": "call|put", "within_pts": 8.0}}`
    Price is within N points of the options volume wall.
    Call wall = highest call volume strike (resistance). Put wall = highest put volume strike (support).

12. **gex_zscore_above** / **gex_zscore_below**
    `{{"type": "gex_zscore_above", "value": 1.0}}`
    GEX z-score vs 20-day rolling history exceeds threshold. Use for extreme GEX readings.

13. **iv_above**
    `{{"type": "iv_above", "value": 0.25}}`
    ATM implied volatility threshold. High IV = fearful market, potential mean-reversion.

14. **pc_ratio_above**
    `{{"type": "pc_ratio_above", "value": 1.5}}`
    Put/call volume ratio threshold. High ratio = crowd over-hedged, contrarian signal.

15. **price_above_vwap** / **price_below_vwap**
    `{{"type": "price_above_vwap"}}`
    Price vs previous day's VWAP. Above = bullish bias, below = bearish bias.
    VWAP is the key institutional benchmark — institutions buy below VWAP and sell above it.

16. **realized_vol_above** / **realized_vol_below**
    `{{"type": "realized_vol_above", "value": 0.15}}`
    Previous day's 5-min realized volatility threshold (annualized).
    High RV = trending/volatile regime, low RV = range-bound/quiet regime.

17. **earnings_nearby** / **no_earnings_nearby**
    `{{"type": "earnings_nearby", "min_tickers": 1}}`
    Whether top-20 SPY components have earnings within 2 days.
    Earnings days have wider ranges and more momentum. Use as a filter.

### Direction
- "long" or "short": fixed direction
- "adaptive": inferred from conditions (price_broke on_high → long, price_broke on_low → short, gap near_on_high → short/fade, etc.)

### Exit Rules

**Stop loss types:**
- `{{"type": "fixed_pts", "value": {ac.default_sl:.1f}}}` — fixed point distance
- `{{"type": "candle_wick", "buffer_pts": {ac.default_sl/4:.1f}}}` — SL at entry candle wick ± buffer
- `{{"type": "level", "value": "on_low", "buffer_pts": {ac.sl_min:.1f}}}` — SL at a reference level

**Take profit types:**
- `{{"type": "risk_multiple", "value": 1.5}}` — TP as multiple of SL distance
- `{{"type": "fixed_pts", "value": {ac.default_sl*1.25:.1f}}}` — fixed point distance
- `{{"type": "level", "value": "on_high"}}` — TP at a reference level

SL distance is clamped to {ac.sl_min}–{ac.sl_max} pts regardless of type.

### Filters (per-day pre-checks)
- `{{"type": "min_on_range", "value": {ac.sl_min*2.5:.0f}}}` — skip days with tiny ON range
- `{{"type": "max_on_range", "value": {ac.sl_max*2.5:.0f}}}` — skip days with extreme ON range
- `{{"type": "day_of_week", "days": [0,1,2,3,4]}}` — 0=Monday, 4=Friday

## Strategy Design Rules

1. **KEEP IT SIMPLE.** Each strategy should be expressible in one sentence.
   Good: "Buy the first pullback to ON high after a breakout above it in the first hour."
   Bad: "Buy when RSI < 30 AND MACD crossover AND Bollinger band squeeze AND..." (we don't have indicators)

2. **USE THE AVAILABLE DATA.** You have: price, ON H/L, day open, prev close, gaps, candle shapes, time,
   GEX regime, put/call walls, ATM IV, put/call ratio.
   You do NOT have: indicators (RSI, MACD, etc.), tick-level order flow.

3. **DIVERSIFY.** Generate strategies with different theses:
   - Breakout strategies (price breaks ON high/low)
   - Mean reversion (fade moves to ON high/low)
   - Gap fill trades (trade unfilled gaps)
   - Opening range plays (first N bars of session)
   - Time-of-day patterns (morning vs afternoon)
   - GEX regime plays (positive GEX = fade, negative GEX = momentum)
   - Options flow plays (put/call wall bounce/rejection, skew fades)
   - VWAP plays (fade above/below prev VWAP, VWAP pullback entries)
   - Volatility regime plays (breakouts in high RV, mean-reversion in low RV)
   - Earnings catalyst plays (momentum on earnings days, avoid chop on quiet days)

4. **REALISTIC STOPS.** SL should be {ac.sl_min*1.5:.1f}–{ac.sl_max*0.6:.1f} pts (realistic for {ac.ticker} intraday). TP should be 0.8–2.5x risk.

5. **TIME WINDOWS.** Most intraday edge is in the first 2 hours (09:30–11:30) or last hour (15:00–15:45). Avoid midday chop (12:00–14:00).

6. **NO LOOK-AHEAD.** Conditions must be evaluable at the current bar — no future data.

7. **AT MOST 4 CONDITIONS per strategy.** More conditions = fewer trades = no statistical significance.

Respond with ONLY a JSON array of strategy objects. No explanation, no markdown, just the JSON.
"""


def _build_seed_refinement_prompt(ac: AssetConfig | None = None) -> str:
    """Build the seed refinement prompt for any asset."""
    if ac is None:
        from asset_config import MES_CONFIG
        ac = MES_CONFIG

    return f"""\
You are a parameter tuner for {ac.name} ({ac.ticker}) intraday strategies.

You will receive a COMPLETE strategy definition (JSON) that already works. Your job
is to produce {{n_variants}} parameter variations that might perform better in the
current market regime.

## CRITICAL RULES — READ CAREFULLY

1. **DO NOT change the strategy structure.** Keep the SAME condition types, the SAME
   number of conditions, and the SAME overall thesis.

2. **ONLY tune these parameters:**
   - `stop_loss.value` — adjust SL distance (stay within {ac.sl_min}–{ac.sl_max} pts)
   - `take_profit.value` — adjust TP distance or risk_multiple ±0.3-0.5
   - `time_window.start` / `time_window.end` — shift ±30 min
   - `buffer_pts` in any condition — adjust ±{ac.default_sl/4:.1f} pts
   - `lookback_bars` in price_broke — adjust ±1
   - `size_pts` in candle_pattern / min_candle_size — adjust ±{ac.default_sl/6:.1f} pts
   - `min_size_pts` / `max_distance_pts` in gap_exists — adjust ±{ac.sl_min:.1f} pts
   - Filter thresholds (min_on_range, max_on_range) — adjust ±{ac.sl_min*1.25:.1f} pts
   - `direction` — can change from "adaptive" to explicit "long" or "short"
   - `regime` in gex_regime_is — can change between "positive"/"negative"/"neutral"
   - `within_pts` in price_near_wall — adjust ±5 pts
   - `value` in gex_zscore_above/below — adjust ±0.5
   - `value` in iv_above — adjust ±0.05
   - `value` in pc_ratio_above — adjust ±0.3
   - `value` in realized_vol_above/below — adjust ±0.03
   - `min_tickers` in earnings_nearby — adjust 1-3

3. **DO NOT:**
   - Add or remove conditions
   - Change condition types (e.g., price_broke → pullback_to)
   - Change reference levels (e.g., on_high → on_low)
   - Invent new filter types
   - Change the strategy name significantly (append " v2", " tight", etc.)

4. **Make each variant MEANINGFULLY different.** Don't just change one number slightly.
   Try: tight stops, wide stops, narrow time window, wide time window, different
   risk:reward ratios. Each variant should represent a distinct parameter regime.

## Instrument Details
- {ac.ticker} = {ac.instrument_details}
- RTH: {ac.rth_start.strftime('%H:%M')}–{ac.rth_end.strftime('%H:%M')} ET
- Risk budget: ${ac.risk_budget:.0f}/trade
- SL must be {ac.sl_min}–{ac.sl_max} pts

Respond with ONLY a JSON array of {{n_variants}} strategy objects. No explanation.
"""


# Keep module-level constants for backwards compat (MES defaults)
STRATEGY_SYSTEM_PROMPT = _build_system_prompt(None)
SEED_REFINEMENT_PROMPT = _build_seed_refinement_prompt(None)


# ─── Market Context ─────────────────────────────────────────────────────────

def build_market_context(
    candles_1m: list,
    start_date: str,
    end_date: str,
    asset_config: AssetConfig | None = None,
) -> str:
    """Compute summary statistics for the optimization window.

    Returns a text block the LLM uses to tailor strategies to current conditions.
    """
    import pandas as pd
    from datetime import time as dtime

    ticker = asset_config.ticker if asset_config else "MES"

    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date)
    if hasattr(candles_1m[0].timestamp, 'tz') and candles_1m[0].timestamp.tz:
        sd = sd.tz_localize(candles_1m[0].timestamp.tz)
        ed = ed.tz_localize(candles_1m[0].timestamp.tz)

    window = [c for c in candles_1m if sd <= c.timestamp <= ed + pd.Timedelta(days=1)]
    if not window:
        return "No data in window."

    # Trading days
    trading_days = sorted(set(c.timestamp.normalize() for c in window
                              if dtime(9, 30) <= c.timestamp.time() < dtime(16, 0)))
    n_days = len(trading_days)
    if n_days < 2:
        return f"Only {n_days} trading days — insufficient for context."

    # Daily ranges
    daily_ranges = []
    daily_opens = []
    daily_closes = []
    for day in trading_days:
        day_bars = [c for c in window
                    if c.timestamp.normalize() == day
                    and dtime(9, 30) <= c.timestamp.time() < dtime(16, 0)]
        if day_bars:
            h = max(c.high for c in day_bars)
            l = min(c.low for c in day_bars)
            daily_ranges.append(h - l)
            daily_opens.append(day_bars[0].open)
            daily_closes.append(day_bars[-1].close)

    avg_range = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0

    # Overnight / previous-day ranges
    import mes_backtest as mb
    on_ranges = []
    ts_index = [c.timestamp for c in candles_1m]
    use_prev_rth = asset_config and asset_config.overnight_session == "prev_rth"

    for day in trading_days:
        if use_prev_rth:
            from equity_data import get_prev_day_high_low
            on_h, on_l = get_prev_day_high_low(candles_1m, day, ts_index)
        else:
            on_h, on_l = mb.get_overnight_high_low(candles_1m, day, ts_index)
        if on_h is not None and on_l is not None:
            on_ranges.append(on_h - on_l)
    avg_on_range = sum(on_ranges) / len(on_ranges) if on_ranges else 0

    on_label = "previous-day" if use_prev_rth else "overnight"

    # Trend
    start_price = daily_opens[0] if daily_opens else 0
    end_price = daily_closes[-1] if daily_closes else 0
    pct_change = ((end_price - start_price) / start_price * 100) if start_price else 0
    if pct_change > 2:
        trend = "trending UP"
    elif pct_change < -2:
        trend = "trending DOWN"
    else:
        trend = "sideways/choppy"

    # Volatility
    price_level = (start_price + end_price) / 2 if start_price else 5000
    vol_pct = avg_range / price_level * 100

    lines = [
        f"## Market Context for {ticker} ({start_date} to {end_date})",
        f"- {n_days} trading days",
        f"- Price level: ~{price_level:,.2f}",
        f"- Direction: {trend} ({pct_change:+.1f}% over period)",
        f"- Avg daily RTH range: {avg_range:.2f} pts",
        f"- Avg {on_label} range: {avg_on_range:.2f} pts",
        f"- Daily volatility: {vol_pct:.2f}% of price",
    ]

    # Gap-up / gap-down frequency (scale threshold by price)
    gap_threshold = 3.0 if not asset_config else asset_config.default_sl / 4
    gap_ups = gap_downs = 0
    for i in range(1, len(daily_opens)):
        diff = daily_opens[i] - daily_closes[i - 1]
        if diff > gap_threshold:
            gap_ups += 1
        elif diff < -gap_threshold:
            gap_downs += 1
    if gap_ups or gap_downs:
        lines.append(f"- Gap-up opens: {gap_ups}/{n_days}, gap-down opens: {gap_downs}/{n_days}")

    return "\n".join(lines)


# ─── Strategy Generation ────────────────────────────────────────────────────

def generate_strategies(
    market_context: str,
    n: int = 10,
    prior_results: list[dict] | None = None,
    asset_config: AssetConfig | None = None,
) -> list[StrategyDefinition]:
    """Generate N candidate strategies using LLM.

    Args:
        market_context: Output of build_market_context()
        n: Number of strategies to generate
        prior_results: Optional list of dicts with {strategy_name, fitness, death_reason}
                       from previous regimes for feedback
        asset_config: Asset configuration (defaults to MES)

    Returns:
        List of validated StrategyDefinition objects
    """
    ticker = asset_config.ticker if asset_config else "MES"

    user_parts = [
        market_context,
        f"\nGenerate exactly {n} diverse {ticker} intraday strategies as a JSON array.",
        "Include a mix of: breakout, mean reversion, gap fill, and opening range strategies.",
        "Vary the time windows, directions, and stop/target sizes across strategies.",
    ]

    if prior_results:
        user_parts.append("\n## Feedback from Previous Regimes")
        user_parts.append("These strategies were tried before. Learn from what worked and what didn't:\n")
        for pr in prior_results[-5:]:  # last 5 regimes
            name = pr.get("strategy_name", "unknown")
            fitness = pr.get("fitness", 0)
            reason = pr.get("death_reason", "unknown")
            pnl = pr.get("forward_pnl", 0)
            wr = pr.get("forward_win_rate", 0)
            user_parts.append(
                f"- \"{name}\": fitness={fitness:.2f}, P&L=${pnl:,.0f}, "
                f"WR={wr:.0f}%, died because: {reason}"
            )
        user_parts.append("\nAvoid repeating strategies similar to ones that failed badly.")
        user_parts.append("Generate variations on themes that showed promise.")

    user_prompt = "\n".join(user_parts)
    sys_prompt = _build_system_prompt(asset_config)

    print(f"  Generating {n} {ticker} strategies via LLM...")
    try:
        response = call_llm(sys_prompt, user_prompt, temperature=0.9)
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return []

    return _parse_strategies(response, asset_config)


def refine_strategy(
    winner: StrategyDefinition,
    fitness: dict,
    market_context: str,
    n_variants: int = 5,
    asset_config: AssetConfig | None = None,
) -> list[StrategyDefinition]:
    """Generate variations of a winning strategy.

    Takes a strategy that performed well and asks the LLM for focused mutations.
    """
    user_parts = [
        market_context,
        "\n## Winning Strategy to Refine",
        f"Name: {winner.name}",
        f"Description: {winner.description}",
        f"Definition: {winner.to_json()}",
        f"\nPerformance:",
        f"- Fitness: {fitness.get('fitness', 0):.3f}",
        f"- Sharpe: {fitness.get('sharpe', 0):.2f}",
        f"- Win Rate: {fitness.get('win_rate', 0):.1f}%",
        f"- P&L: ${fitness.get('total_pnl', 0):,.0f}",
        f"- Trades: {fitness.get('n_trades', 0)}",
        f"\nGenerate exactly {n_variants} variations of this strategy as a JSON array.",
        "Try:",
        "1. Tighter and wider stops",
        "2. Different time windows (shift start/end by 30-60 min)",
        "3. Adding or removing one filter",
        "4. Changing TP from risk-multiple to fixed-pts or vice versa",
        "5. Adjusting buffer/lookback values",
        "\nKeep the core thesis the same but vary the execution parameters.",
    ]

    user_prompt = "\n".join(user_parts)
    sys_prompt = _build_system_prompt(asset_config)

    print(f"  Refining '{winner.name}' → {n_variants} variants...")
    try:
        response = call_llm(sys_prompt, user_prompt, temperature=0.5)
    except Exception as e:
        print(f"  LLM refinement failed: {e}")
        return []

    return _parse_strategies(response, asset_config)


# ─── Seed Refinement ────────────────────────────────────────────────────────

def build_archetype_history(
    archetype: str,
    prior_results: list[dict] | None,
) -> str:
    """Summarize how a specific archetype has performed across prior regimes."""
    if not prior_results:
        return ""

    matches = [pr for pr in prior_results if pr.get("archetype") == archetype]
    if not matches:
        return ""

    lines = [f"\n## History for '{archetype}' archetype ({len(matches)} prior attempts):"]
    for pr in matches[-5:]:
        name = pr.get("strategy_name", "?")
        pnl = pr.get("forward_pnl", 0)
        wr = pr.get("forward_win_rate", 0)
        days = pr.get("forward_days", 0)
        reason = pr.get("death_reason", "?")
        lines.append(f"- \"{name}\": P&L=${pnl:,.0f}, WR={wr:.0f}%, lived {days}d, died: {reason}")

    avg_pnl = sum(pr.get("forward_pnl", 0) for pr in matches) / len(matches)
    avg_days = sum(pr.get("forward_days", 0) for pr in matches) / len(matches)
    lines.append(f"- Average: P&L=${avg_pnl:,.0f}, lifespan={avg_days:.0f} days")

    return "\n".join(lines)


def refine_seed(
    seed: StrategyDefinition,
    fitness_result: dict,
    market_context: str,
    archetype: str,
    n_variants: int = 5,
    prior_results: list[dict] | None = None,
    batch_size: int = 5,
    asset_config: AssetConfig | None = None,
    mfe_mae_context: dict | None = None,
    regime_context: dict | None = None,
) -> list[StrategyDefinition]:
    """Generate parameter-tuned variants of a strategy seed.

    Unlike refine_strategy() which allows structural changes, this function
    constrains the LLM to parameter tuning only — same conditions, same thesis,
    different numbers.

    If n_variants > batch_size, splits into multiple LLM calls of batch_size
    each to avoid output truncation.
    """
    archetype_history = build_archetype_history(archetype, prior_results)

    base_user_parts = [
        market_context,
        f"\n## Strategy Seed to Tune",
        f"Name: {seed.name}",
        f"Description: {seed.description}",
        f"Archetype: {archetype}",
        f"Definition:\n```json\n{seed.to_json()}\n```",
        f"\n## Current Performance on Optimization Window",
        f"- Fitness: {fitness_result.get('fitness', 0):.3f}",
        f"- Sharpe: {fitness_result.get('sharpe', 0):.2f}",
        f"- Win Rate: {fitness_result.get('win_rate', 0):.1f}%",
        f"- P&L: ${fitness_result.get('total_pnl', 0):,.0f}",
        f"- Trades: {fitness_result.get('n_trades', 0)}",
        f"- Max DD: ${fitness_result.get('max_dd', 0):,.0f}",
    ]

    if archetype_history:
        base_user_parts.append(archetype_history)

    # Inject empirical MFE/MAE context if available
    ctx = mfe_mae_context or {}
    if ctx and ctx.get("mfe_p50"):
        source_label = ("UNCENSORED true distributions" if ctx.get("source") == "uncensored"
                        else "in-sample distributions (censored)")
        mae_floor = ctx.get("mae_p75", ctx["mae_p50"])
        trail_dist = round(ctx["mfe_p50"] * 0.35, 1)
        trail_exit = round(ctx["mfe_p50"] * 0.65, 1)
        mfe_mae_block = (
            f"\n## EMPIRICAL EXIT DATA for {archetype} ({source_label}, n={ctx['n_candidates']}):\n"
            f"\nTrue price move distributions for this archetype in current market regime:\n"
            f"- MFE p25: {ctx.get('mfe_p25', 'N/A')}pts | MFE p50: {ctx['mfe_p50']}pts"
            f" | MFE p75: {ctx.get('mfe_p75', 'N/A')}pts\n"
            f"- MAE p50: {ctx['mae_p50']}pts | MAE p75: {ctx.get('mae_p75', 'N/A')}pts\n"
            f"\nEXIT CALIBRATION RULES (derive from above):\n"
            f"1. SL must be >= MAE p75 ({mae_floor}pts) — trades commonly dip this far before recovering\n"
            f"   Setting SL tighter than MAE p75 stops out trades that would have been winners\n"
            f"2. TP should target MFE p50 ({ctx['mfe_p50']}pts) — reliable capture of median profitable move\n"
            f"   TP at MFE p75+ is possible but infrequent, increases time-in-trade risk\n"
            f"3. If using TSL: trigger at MFE p25 ({ctx.get('mfe_p25', 'N/A')}pts) to confirm favorable move first\n"
            f"   Trail distance = MFE p50 x 0.35 = {trail_dist}pts\n"
            f"   Expected exit = MFE p50 - trail = {trail_exit}pts favorable\n"
            f"\nDO NOT use SL tighter than {mae_floor}pts.\n"
            f"DO NOT use TP lower than {ctx['mfe_p50']}pts."
        )
        base_user_parts.append(mfe_mae_block)
        print(f"  [LLM] context: source={ctx.get('source', 'unknown')}, n={ctx['n_candidates']}, "
              f"mfe_p50={ctx['mfe_p50']}pts, mae_p75={ctx.get('mae_p75')}pts")
    else:
        print(f"  [LLM] context: insufficient data for {archetype}, no injection")

    # Inject regime context: GEX regime, top archetypes, entry variation instruction
    rc = regime_context or {}
    if rc:
        parts = ["\n## REGIME CONTEXT"]
        if rc.get("gex_regime"):
            parts.append(f"GEX regime at optimization start: **{rc['gex_regime']}**")
        if rc.get("top_archetypes"):
            top3 = rc["top_archetypes"][:3]
            parts.append("Top winning archetypes this window: " +
                         ", ".join(f"{a['archetype']} (fitness={a['fitness']:.3f})" for a in top3))
        parts.append(
            "\nVARIATION GUIDANCE: Across the variants, vary entry condition PARAMETERS "
            "(buffer_pts, thresholds, time windows, direction, regime filters) — "
            "not just exit SL/TP. Each variant should explore a different entry hypothesis."
        )
        base_user_parts.append("\n".join(parts))
        print(f"  [LLM] regime_context: gex={rc.get('gex_regime')}, "
              f"top_archetypes={len(rc.get('top_archetypes', []))}")

    # Split into batches to avoid output truncation
    all_variants = []
    n_batches = max(1, (n_variants + batch_size - 1) // batch_size)
    remaining = n_variants

    print(f"  Refining seed '{seed.name}' ({archetype}) → {n_variants} variants "
          f"in {n_batches} batch(es)...")

    refinement_prompt = _build_seed_refinement_prompt(asset_config)

    for batch_i in range(n_batches):
        this_batch = min(batch_size, remaining)
        remaining -= this_batch

        user_parts = list(base_user_parts)
        user_parts.append(
            f"\nGenerate exactly {this_batch} parameter-tuned variants as a JSON array."
        )

        sys_prompt = refinement_prompt.replace("{n_variants}", str(this_batch))
        user_prompt = "\n".join(user_parts)

        try:
            response = call_llm(sys_prompt, user_prompt, temperature=0.5)
        except Exception as e:
            print(f"    Batch {batch_i+1}/{n_batches} failed: {e}")
            continue

        batch_variants = _parse_strategies(response, asset_config)
        print(f"    Batch {batch_i+1}/{n_batches}: {len(batch_variants)} variants parsed")
        all_variants.extend(batch_variants)

    return all_variants


# ─── Parsing ─────────────────────────────────────────────────────────────────

def _parse_strategies(
    response_text: str,
    asset_config: AssetConfig | None = None,
) -> list[StrategyDefinition]:
    """Parse LLM response into StrategyDefinition objects."""
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object in the text
        start_arr = text.find("[")
        end_arr = text.rfind("]") + 1
        start_obj = text.find("{")
        end_obj = text.rfind("}") + 1

        if start_arr >= 0 and end_arr > start_arr and start_arr <= start_obj:
            try:
                data = json.loads(text[start_arr:end_arr])
            except json.JSONDecodeError:
                print(f"  Failed to parse LLM response as JSON array")
                return []
        elif start_obj >= 0 and end_obj > start_obj:
            try:
                data = json.loads(text[start_obj:end_obj])
            except json.JSONDecodeError:
                print(f"  Failed to parse LLM response as JSON")
                return []
        else:
            print(f"  No JSON found in LLM response")
            return []

    # Handle different response shapes
    if isinstance(data, dict):
        if "strategies" in data:
            data = data["strategies"]
        elif "variants" in data:
            data = data["variants"]
        else:
            data = [data]

    if not isinstance(data, list):
        print(f"  Unexpected response type: {type(data)}")
        return []

    # Validate and convert each strategy
    strategies = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        try:
            strat = _validate_strategy(item, asset_config)
            if strat:
                strategies.append(strat)
        except Exception as e:
            print(f"  Skipping strategy #{i+1}: {e}")
            continue

    print(f"  Parsed {len(strategies)}/{len(data)} valid strategies")
    return strategies


def _validate_strategy(
    d: dict,
    asset_config: AssetConfig | None = None,
) -> StrategyDefinition | None:
    """Validate a strategy dict and return a StrategyDefinition or None."""
    # Required fields
    if "entry" not in d:
        return None

    entry = d["entry"]
    if not isinstance(entry, dict):
        return None

    # Ensure conditions exist
    conditions = entry.get("conditions", [])
    if not conditions:
        return None

    # Validate condition types (filter, don't mutate during iteration)
    valid_types = {
        "price_above", "price_below", "price_broke", "pullback_to",
        "candle_pattern", "min_candle_size", "gap_exists", "time_is",
        "bar_index", "on_range_between",
        # GEX / options flow conditions
        "gex_regime_is", "price_near_wall", "gex_zscore_above",
        "gex_zscore_below", "iv_above", "pc_ratio_above",
        # Daily stats / earnings conditions
        "price_above_vwap", "price_below_vwap",
        "price_extended_above_vwap", "price_extended_below_vwap",
        "realized_vol_above", "realized_vol_below",
        "earnings_nearby", "no_earnings_nearby",
    }
    cleaned = []
    for cond in conditions:
        if not isinstance(cond, dict):
            return None
        ctype = cond.get("type", "")
        if ctype in valid_types:
            cleaned.append(cond)
        else:
            print(f"    Unknown condition type '{ctype}' — removing")
    conditions[:] = cleaned

    if not conditions:
        return None

    # Ensure exit rules exist
    if "exit" not in d:
        d["exit"] = {
            "stop_loss": {"type": "fixed_pts", "value": 12},
            "take_profit": {"type": "risk_multiple", "value": 1.0},
        }

    exit_rules = d["exit"]
    if "stop_loss" not in exit_rules:
        exit_rules["stop_loss"] = {"type": "fixed_pts", "value": 12}
    if "take_profit" not in exit_rules:
        exit_rules["take_profit"] = {"type": "risk_multiple", "value": 1.0}

    # Validate SL value range (asset-specific bounds)
    sl_min = asset_config.sl_min if asset_config else 4
    sl_max = asset_config.sl_max if asset_config else 32
    sl = exit_rules["stop_loss"]
    if sl.get("type") == "fixed_pts":
        val = sl.get("value", asset_config.default_sl if asset_config else 12)
        sl["value"] = max(sl_min, min(sl_max, val))

    return StrategyDefinition.from_dict(d)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Generator — LLM Strategy Creation")
    parser.add_argument("--context", type=str, default=None,
                        help="Market context string (or auto-compute from data)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of strategies to generate (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    context = args.context or "## Market Context\n- 60 trading days\n- Price level: ~5500\n- Direction: sideways\n- Avg daily range: 45 pts\n- Avg ON range: 20 pts"

    strategies = generate_strategies(context, n=args.n)

    if not strategies:
        print("No strategies generated.")
        return

    print(f"\nGenerated {len(strategies)} strategies:")
    for i, s in enumerate(strategies):
        print(f"\n  {i+1}. {s.name}")
        print(f"     {s.description}")
        direction = s.entry.get("direction", "adaptive")
        tw = s.entry.get("time_window", {})
        n_conds = len(s.entry.get("conditions", []))
        print(f"     dir={direction}, window={tw.get('start','?')}-{tw.get('end','?')}, "
              f"{n_conds} conditions")

    if args.output:
        out = [s.to_dict() for s in strategies]
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
