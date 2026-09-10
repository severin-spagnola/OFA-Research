# Penny Gapper Gene Search — Independent Audit

You are auditing a genetic/combinatorial strategy search system for penny stock gap-up day-trading. Your job is to find **any** source of lookahead bias, data leakage, or methodological error that would produce inflated or overoptimistic backtest results. Be adversarial — assume bugs exist until proven otherwise.

## Files to audit

All files are in this directory:

| File | Purpose |
|------|---------|
| `penny_backtest.py` | Core backtester: indicator computation, 18 entry detectors, stop/trail/exit logic, trade simulation, pooled evaluation |
| `penny_genes.py` | Gene definition: 18 entry types, parameter ranges, sampling, candidate generation |
| `penny_fitness.py` | Fitness scoring: composite score from Sharpe, PF, trade count, return:DD, bonuses, penalties |
| `penny_search.py` | Search orchestrator: profile definitions, temporal train/val split, parallel evaluation, ranking |
| `penny_gapper_survey.py` | Data pipeline: scan gappers, enrich metadata, pull 1m bars, feature analysis (not part of live search — data prep only) |

## Audit checklist

### 1. Lookahead bias in indicators
- Every indicator (`compute_ema`, `compute_rsi`, `compute_macd`, `compute_vwap`, `compute_atr`) must produce `output[i]` using ONLY `bars[0..i]`. No future bar should influence any indicator value.
- Check initial seeding / warm-up: does the first value of each indicator accidentally peek at later data?
- VWAP resets daily — since we only process single days, verify it accumulates from bar 0 forward.

### 2. Lookahead bias in entry detectors
- There are 18 entry detectors (`_detect_ramp`, `_detect_orb`, `_detect_vwap_cross`, `_detect_vwap_bounce`, `_detect_ema_cross`, `_detect_ema_trend`, `_detect_pullback`, `_detect_macd_cross`, `_detect_rsi_extreme`, `_detect_candle_pattern`, `_detect_volume_climax`, `_detect_range_breakout`, `_detect_gap_fill_entry`, `_detect_hod_break`, `_detect_first_red_fade`, `_detect_price_level`, `_detect_momentum_accel`, `_detect_sustained_volume`).
- For EACH detector, verify:
  - The trigger condition at bar `i` uses only data from bars `[0..i]`
  - Entry is on the NEXT bar's open (`i+1`), never on the trigger bar's close
  - Stop price is computed from data available at trigger time, not from future bars
- Pay special attention to detectors that look backward over a window (e.g., ramp detection, range breakout consolidation window) — the window must not extend past the trigger bar.

### 3. Lookahead bias in trade simulation
- `simulate_trade()` must process bars sequentially from entry to exit.
- On each bar, check: is the stop-loss checked BEFORE the take-profit? (Conservative ordering: assume worst case first.)
- Trail stop updates must use the PREVIOUS bar's close/low/high, not the current bar.
- Partial exits: verify partial fill prices are realistic (use the bar's price, not a future price).
- EOD forced exit at 15:55 — verify the exit price uses that bar's close, not a later bar.

### 4. Lookahead bias in stop computation
- `compute_stop()` handles 4 SL types: `ramp_low`, `fixed_pct`, `atr_mult`, `swing_low`.
- `swing_low` must look backward from the trigger bar, never forward.
- `atr_mult` must use ATR computed up to the trigger bar only.

### 5. Confirmation checks
- `check_confirmation()` validates 8 secondary conditions at the trigger bar.
- Verify each condition uses only data available at bar `trigger_idx`.
- `ema_aligned` — which EMA, and is it fully warmed up by trigger time?

### 6. Data leakage in train/val split
- `run_gene_search()` in `penny_search.py` does a temporal train/val split.
- Verify the split is strictly temporal: all train dates < all val dates. No shuffling.
- Verify that NO information from val ticker-days leaks into training (e.g., shared indicator state, shared normalization, shared caching that mixes train/val data).
- Verify the ranking uses train fitness for initial selection and val fitness for final ranking — not the other way around.

### 7. Data leakage in profile/filter definitions
- Profiles in `penny_search.py` define which ticker-days are included (exchange, vol_ratio, gap_size).
- These filters must use ONLY pre-market or prior-day data (gap size, volume ratio vs avg). They must NOT use any intraday data that would only be known after the open.
- Check: does `load_ticker_days()` filter on any intraday feature (e.g., first_5m_return, open_ramp_bars) that constitutes lookahead?

### 8. Survivorship / selection bias in data pipeline
- `penny_gapper_survey.py` scans for gap-up stocks. Does it exclude delisted tickers that gapped up and then got halted/delisted? (If Polygon doesn't return them, that's survivorship bias.)
- Are tickers that get halted intraday handled correctly, or do they silently disappear from the bar data?

### 9. Gene sampling bias
- `sample_random_genes()` and `sample_candidates()` in `penny_genes.py`.
- Does the 3-pass sampling (coverage, random, archetype balance) introduce any systematic bias that would favor certain entry types?
- Are impossible parameter combinations pruned correctly (e.g., short direction with long-only entry types)?

### 10. Fitness function gaming
- `compute_fitness()` in `penny_fitness.py`.
- Can a degenerate strategy (e.g., 1 trade with huge R) game the scoring? Check that penalties properly suppress:
  - Low trade count
  - Single-trade concentration
  - Suspiciously high Sharpe (>5) or PF (>5)
  - Negative cumulative return
- Is the hard rejection threshold (Sharpe > 12, PF > 10) appropriate or too lenient?

### 11. Price assumptions
- Entry at next bar open: is this realistic for penny stocks? Can you actually get filled at the open of a 1-minute bar?
- Stop-loss fills: does the sim assume you get filled at exactly the stop price, or does it account for gaps through the stop?
- Are there any fill assumptions that are unrealistically favorable for penny stocks (wide spreads, low liquidity)?

### 12. Edge cases
- What happens when a ticker-day has < 10 bars of data?
- What happens when indicators haven't warmed up (e.g., RSI needs 14 bars, MACD needs 26)?
- What happens when the entry detector returns a trigger at the last bar of the day?
- Division by zero: are there any unguarded divisions (by volume, by price, by ATR)?

## Output format

For each issue found, report:

```
SEVERITY: [CRITICAL / HIGH / MEDIUM / LOW]
LOCATION: file:line_number(s)
ISSUE: one-line description
DETAIL: explanation of the bias/leak and how it inflates results
FIX: suggested fix
```

If no issues are found in a category, explicitly state "NO ISSUES FOUND" for that category.

At the end, provide a summary verdict:
- **PASS**: No critical or high issues. Results can be trusted with noted caveats.
- **CONDITIONAL PASS**: Medium issues exist that should be fixed but don't fundamentally invalidate results.
- **FAIL**: Critical or high issues exist. Results cannot be trusted until fixed.
