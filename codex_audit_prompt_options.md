# Codex Audit Prompt: Options Walk-Forward Regime Search

You are auditing a walk-forward options strategy search system for **data leakage, lookahead bias, optimistic fill assumptions, and statistical validity**. The system ran 45 walk-forward windows across ~22 months of SPY 1-minute options chain data and produced a 56.6% forward-profitable rate (509/899 regimes). All 45 windows produced at least one profitable strategy.

Your job is to determine whether this result is real or an artifact of subtle biases.

---

## System Architecture

The pipeline has 7 files across 2 directories:

| File | Lines | Role |
|------|-------|------|
| `current/options_wf/wf_runner.py` | 463 | Core walk-forward orchestrator: generates windows, evaluates candidates, writes JSONL |
| `current/options_wf/options_backtest.py` | 1995 | Backtest engine: trade construction, exit scanning, fitness computation, day context |
| `current/options_wf/options_genes.py` | ~400 | Gene catalog: 48 entry conditions, 17 regime filters, 10 trade types |
| `current/options_wf/options_search.py` | ~300 | Parallel search wrapper (not used by wf_runner) |
| `current/options_wf/wf_analyze.py` | 271 | Post-hoc analysis: reads regime_db.jsonl, computes stats |
| `infra/wf_runpod_handler.py` | 340 | RunPod serverless handler: receives window payload, downloads data, calls evaluate_window |
| `infra/wf_orchestrator.py` | ~450 | Local dispatcher: generates windows, submits one RunPod job per window, aggregates results |

Output: `results/options_wf/regime_db.jsonl` — 899 records, one JSON object per line.

---

## AUDIT AREA 1: Walk-Forward Leakage

### 1A. Window generation — are train/val strictly partitioned?

`wf_runner.py:generate_windows()` (lines 62-100) slices from a sorted list of trading days:

```python
train_days = all_days[i : i + train_size]          # line 85
val_days = all_days[i + train_size : i + total_needed]  # line 86
```

- Train = 60 trading days, Val = 20 trading days, Step = 10 trading days.
- Windows overlap in train data (step < train_size). Window N's late train days are Window N+1's early train days.

**Audit question**: The train windows overlap. Does this mean the same gene combination could be "discovered" in multiple windows using shared train data, inflating the count of apparently independent forward-profitable regimes? Since the gene sampler uses `seed + window_id * 137` (line 380), different windows sample different candidates — but the overlapping train data means similar strategies could independently appear in adjacent windows.

### 1B. Data preloading — does precomputation leak val into train?

In `wf_runner.py:evaluate_window()` (lines 123-303), data for ALL window days (train + val) is loaded and precomputed together:

```python
all_window_days = window.train_days + window.val_days   # line 154
# ...
for d in all_window_days:                                # lines 159-176
    preloaded_und[d] = load_underlying(...)
    preloaded_chains[d] = load_chain(...)
    preloaded_metas[d] = load_chain_meta(...)
```

Then `prev_close` is computed for regime filters across ALL days:

```python
sorted_und_days = sorted(preloaded_und.keys())           # line 179
for i, d in enumerate(sorted_und_days):                  # line 180
    if d in preloaded_metas and i > 0:
        prev_d = sorted_und_days[i - 1]
        preloaded_metas[d]["prev_close"] = float(prev_und[prev_d].iloc[-1]["c"])  # line 185
```

**Audit question**: The `prev_close` linkage crosses the train/val boundary. The last train day's close becomes the first val day's `prev_close`. This is correct (it's causal — yesterday's close is known today). But verify: does the presence of val days in `sorted_und_days` ever cause a val day's data to influence a train day's `prev_close`? (It shouldn't, since val days are strictly after train days.)

### 1C. Day context precomputation — does chain enrichment leak?

```python
preloaded_ctxs = {}
for d in all_window_days:                                # line 195
    ctx = precompute_day_context(                        # line 197
        preloaded_und[d], preloaded_chains.get(d), preloaded_metas.get(d))
```

`precompute_day_context()` (line 968) enriches with chain-derived signals via `_enrich_ctx_with_chain()` (line 1036). This uses cumulative call/put volumes, ATM skew, and IV proxy — all computed from that day's own chain data only.

**Audit question**: `_enrich_ctx_with_chain()` uses forward-fill for missing minutes (lines 1107-1111):

```python
for j in range(1, n_bins):
    if atm_c_by_min[j] == 0:
        atm_c_by_min[j] = atm_c_by_min[j - 1]
```

This fills minutes within a single day using prior minutes. Is this intraday forward-fill introducing any lookahead? (It shouldn't — it's filling forward in time within the day, so minute 600's value comes from minute 599 or earlier.)

### 1D. RunPod handler — does it introduce different leakage?

`infra/wf_runpod_handler.py:_handle_wf_window()` (lines 209-301) also pre-loads all window days:

```python
all_window_days = train_days + val_days                   # line 252
for d in all_window_days:                                 # line 257
    preloaded_und[d] = load_underlying(data_dir, d)
```

**Audit question**: The RunPod handler and `wf_runner.py` implement the same preloading logic independently. Are there any divergences between the two that could cause different behavior? The handler passes preloaded data to `evaluate_window()`, which then skips loading (lines 157-176 check `if preloaded_und is None`). Verify both paths produce identical results.

---

## AUDIT AREA 2: Regime Record Integrity

### 2A. Val inclusion criteria — are we counting noise as signal?

```python
val_fr = compute_fitness(val_trades, min_trades=1)        # line 272, wf_runner.py
if val_fr.n_trades == 0:                                  # line 275
    continue
```

The train pass requires `min_trades=30` (default), but val requires only `min_trades=1`. A regime with 1 val trade that happens to be profitable is recorded as `forward_profitable=True`.

**Audit question**: How many of the 509 forward-profitable regimes have `val_n_trades` < 5? A single profitable trade is not statistically meaningful. In the sample records, `val_n_trades` ranges from 9-19. But does the full dataset contain regimes with 1-3 val trades? If so, what fraction of the "56.6% forward profitable" rate comes from these low-trade-count regimes?

### 2B. Forward profitable definition

```python
val_cum_pnl = sum(t.pnl_net for t in val_trades)          # line 278
forward_profitable = val_cum_pnl > 0                       # line 295
```

**Audit question**: `forward_profitable` is a binary flag based on whether cumulative PnL > $0. A regime with $1.50 cumulative PnL on 15 trades is "forward profitable." Is there a more meaningful threshold? The records show val_cum_pnl ranging from $1.50 to $558.80 for profitable regimes. What's the distribution? How many regimes are profitable by less than one round-trip friction cost ($4.30 for naked)?

### 2C. Train fitness computation and val asymmetry

`compute_fitness()` (line 1743) computes a composite fitness score:

```python
fitness = sharpe * 0.4 + min(pf, 3.0) * 0.3 + wr * 0.2 + trade_bonus * 0.1  # line 1806
```

Where `trade_bonus = min(1.0, n / 100)`. With `min_trades=30` on train, most candidates get `trade_bonus ≈ 0.3-0.5`. The top 20 by fitness are selected (line 245).

**Audit question**: The fitness function weights Sharpe at 40%. Sharpe is annualized using actual date span (lines 1764-1772):

```python
span_years = max((d1 - d0).days / 365.25, 1 / 365.25)
trades_per_year = n / span_years
sharpe = (avg_r / std_r) * np.sqrt(trades_per_year)
```

For a 60-day train window with ~50 trades, this is ~50/0.164 ≈ 305 trades/year, annualization factor ≈ sqrt(305) ≈ 17.5. Is this inflating Sharpe? Sharpe values in the records range from 2.0 to 4.8 on train, which seems suspiciously high for a 3-month window. Does the annualization assumption (constant trade rate across the year) hold for strategies that trade only under specific conditions?

---

## AUDIT AREA 3: Backtest Engine Timing and Fill Assumptions

### 3A. Entry fill price — bar close, not mid

`construct_trade()` (line 169) fills entries at the close price of the nearest prior bar:

```python
closest = _nearest_prior_bar(leg_mins, target_min)        # line 237
entry_premium = float(leg_bars.iloc[closest]["c"])         # line 240
```

**Audit question**: Using bar close as fill price is optimistic. The bar close is known only at the end of the minute. In reality, you'd need to route the order and get filled during the next bar. The entry signal fires at the close of the underlying's bar, but the option fill uses the same bar's close. This is same-bar fill — a common source of ~1 minute lookahead.

However, the code finds the bar at or BEFORE the target minute (`_nearest_prior_bar`, line 157):

```python
mask = mins <= target
candidates = np.where(mask)[0]
return int(candidates[np.argmin(target - mins[candidates])])
```

So if the signal fires at minute 600, the option fill is from the bar at minute 600 or earlier. **Is this really no-lookahead?** The signal comes from the underlying's bar at minute 600 (which includes that bar's close), and the option fill comes from the option's bar at the same minute 600 (which also includes that bar's close). Both closes are known simultaneously at the end of minute 600. But you can't observe the underlying close and then fill the option at that same bar's close — you'd need the next bar. This is a **potential 1-minute lookahead bias**.

### 3B. TSL exit fill — theoretical stop level, not bar price

For naked options with trailing stop, `scan_exit()` (lines 496-562) fills at the TSL level, not the actual bar price:

```python
if tsl_hit:
    exit_reason = "tsl"
    exit_val = tsl_level                                  # line 528
```

Where `tsl_level = hwm * (1 - tsl_pct)` (line 512). If the option's premium drops from a high watermark of $5.00 through the TSL level of $4.25 (15% TSL) to a bar close of $3.80, the system fills at $4.25, not $3.80.

**Audit question**: This is optimistic. In reality, when a trailing stop is hit, you get filled at market — which could be anywhere between the TSL level and the bar's low. The bar close of $3.80 means the actual fill would likely be worse than $4.25. The 1-minute bars on SPY options are liquid, but slippage on stop exits is real. Similarly, SL exits fill at `sl_value` (line 531), not bar price.

Contrast: time_stop and eod exits correctly fill at `v` (the actual bar close, lines 534/537). Only TP-like stops (TSL, SL) use the theoretical level.

### 3C. Naked option friction model

```python
if tk == "naked":
    entry_cost = cost.slippage_per_leg * 100 + cost.commission_per_leg  # line 1248
    exit_cost = cost.commission_per_leg                                  # line 1249
```

With defaults: entry = $0.03 * 100 + $0.65 = $3.65, exit = $0.65. Total = $4.30 per round trip.

**Audit question**: $0.03/leg slippage on entry only, zero slippage on exit. Why no exit slippage? Market orders on exit (especially TSL/SL stops) would have at least the same slippage as entry. This understates friction by ~$3.00/trade. With avg_pnl of $20-30, this is a 10-15% drag that's being ignored.

### 3D. One-trade-per-day limitation

`evaluate_day()` (line 1156) returns after the first trade:

```python
for idx in range(ctx["n"]):                               # line 1217
    # ...signal checks...
    return OptionsTrade(...)                               # line 1255
return None                                                # line 1277
```

**Audit question**: The system takes at most one trade per day per gene set. This is realistic (reduces overtrading), but it means the first qualifying signal each day is always taken. There's no "wait for better setup" logic. Is this biasing toward morning entries where signals fire earliest? Check the regime_db records: do most entries cluster in the 09:45-10:30 range?

---

## AUDIT AREA 4: 45/45 Windows Scrutiny

### 4A. All 45 windows produced at least one profitable strategy

`wf_analyze.py` reports: "Windows with at least 1 profitable strategy: 45/45 (100%)."

**Audit question**: With 1000 candidates per window, top 20 selected, and a binary outcome (forward_profitable), what's the probability that at least 1 of 20 strategies randomly produces val_cum_pnl > $0 even with zero skill? If the null hypothesis is 50% chance per strategy, P(at least 1 of 20 profitable) = 1 - 0.5^20 ≈ 99.9999%. So 45/45 is **expected under pure noise**. This metric is meaningless without adjusting for multiple testing. The relevant question is: is the 56.6% aggregate rate significantly above 50%?

### 4B. Naked ITM3 dominance

The analysis showed naked_itm3 at 66.3% forward profitable rate, $78K cumulative PnL. All spread types were 0-25%.

**Audit question**: ITM3 calls/puts have high delta (~0.85-0.95). They behave almost like the underlying with leverage. The trailing stop + high delta combination means you're basically running a momentum strategy on the underlying with natural convexity from the option's gamma. Is the "edge" just momentum on SPY, amplified by high-delta options + trailing stop? If so, it would work in the trending period (2024 was a strong year for SPY) but fail in a mean-reverting regime. This is not necessarily wrong, but the edge is regime-dependent, not structural.

### 4C. Edge concentration

"Windows contributing 80% of positive PnL: 22/45" — about half the windows drive most of the PnL.

**Audit question**: Is this genuinely distributed, or are the 22 "good" windows clustered in time? If the 22 productive windows are all in Q2-Q3 2024 (the strong trend period), the edge is period-specific, not persistent. Cross-reference window_ids with their date ranges to check temporal clustering.

### 4D. Overlapping val windows

With step=10 and val_size=20, consecutive windows' val periods overlap by 10 days. Window 0's val days include days 60-79, Window 1's val days include days 70-89. Days 70-79 appear in both windows' val sets.

**Audit question**: This means the same trades on the same days can count as "forward profitable" in two different windows. If a strategy does well on days 70-79, it inflates the profitable count for both Window 0 and Window 1. The 899 regime records are NOT independent observations. How much does val overlap inflate the apparent success rate?

---

## AUDIT AREA 5: Data Pipeline Assumptions

### 5A. Weekday approximation for window generation

`infra/wf_orchestrator.py` generates windows locally using weekday approximation when it can't get exact trading days. It generates Mon-Fri dates, which includes holidays. The server-side handler (wf_runpod_handler.py) then uses actual trading days.

**Audit question**: The orchestrator generated 522 weekday dates vs 471 actual trading days. This means ~51 holiday dates were included in window day lists. The handler silently skips days without data (lines 258-266 load and check). But windows that fall on holidays have fewer actual train/val days than expected. The handler's guard `if len(valid_train) < 20 or len(valid_val) < 5` (wf_runner.py:213) catches extreme cases, but a window with 48/60 actual train days still proceeds. Does this affect fitness comparability across windows?

### 5B. VWAP computation uses cumulative close * volume

```python
cum_pv = np.cumsum(closes * vols)                         # line 989
cum_v = np.cumsum(vols)
vwap = np.where(cum_v > 0, cum_pv / cum_v, closes)       # line 991
```

**Audit question**: Standard VWAP uses typical price `(H+L+C)/3 * volume`, not `close * volume`. Using close-only VWAP gives slightly different values and may make VWAP-based signals (price_above_vwap, price_below_vwap — the most common entry conditions by record count) fire at different times than they would with standard VWAP. This doesn't invalidate results but means live execution with a standard VWAP would see signal divergence.

### 5C. Chain data quality — $0.05 minimum premium filter

```python
if entry_premium <= 0.05:  # too cheap, likely bad data   # line 243
    return None
```

**Audit question**: The $0.05 floor filters out stale/bad quotes but also filters out cheap OTM options. For ITM3 options on SPY (~$500), premiums are typically $3-8, so this filter is irrelevant for the dominant trade type. But for ATM options, it could matter during low-vol periods. Is this filter affecting the ATM vs ITM comparison?

---

## AUDIT AREA 6: Statistical Validity

### 6A. Multiple testing correction

1000 candidates are tested per window. The top 20 are selected. This is heavy optimization. The val pass is supposed to act as out-of-sample validation, but:

- `min_trades=1` on val means noise can survive
- `forward_profitable = val_cum_pnl > $0` is a weak bar
- 20 strategies are tested on val — no correction for testing 20 hypotheses

**Audit question**: What's the false discovery rate? If the 20 train-selected strategies have no real edge, and val outcomes are roughly symmetric around $0, you'd expect ~10/20 (50%) to be forward profitable by chance. The observed rate is 509/899 ≈ 56.6%. Is 56.6% significantly different from 50%? A binomial test: P(X ≥ 509 | n=899, p=0.5) — compute this. If the p-value is < 0.01, there's likely some real signal. If it's > 0.05, the result is noise.

But wait — the 899 observations are not independent (val overlap, same train data across windows). The effective sample size is smaller than 899. A bootstrap or permutation test that respects the window structure would be more appropriate.

### 6B. VWAP proximity as signal quality

The best-performing archetypes (vwap_trend, breakout, open_drive) all use simple price-relative-to-reference signals. These are essentially momentum signals. VWAP-above = recent uptrend, breakout = new high, above-open = positive day.

**Audit question**: Are these signals adding value beyond "buy SPY when it's going up, sell when it's going down"? A null hypothesis test: run the same walk-forward with `{"type": "always"}` as the only entry condition (unconditional entry at random times within the time window) using naked_itm3. If the unconditional base rate is also ~55%, the entry signals aren't adding edge — it's all coming from the TSL + high-delta combination.

### 6C. Fitness function circularity risk

The fitness function (line 1806) uses Sharpe, profit factor, win rate, and trade count. All of these are computed from the same trade list. There's no regularization or penalty for complexity.

**Audit question**: A strategy with 5 entry conditions that fires 32 trades (just above min_trades=30) is ranked alongside a strategy with 1 condition that fires 55 trades. The simpler strategy should be preferred (less overfit). Does the fitness function account for gene complexity? (It doesn't — the `trade_bonus` ramp only rewards more trades, not simpler genes.)

---

## Summary of Red Flags to Investigate

1. **Same-bar fill**: Entry signal and option fill use the same minute's close — potential 1-minute lookahead
2. **TSL/SL fills at theoretical level**: Stop exits fill at the trigger price, not market price — optimistic by ~$2-5/trade
3. **No exit slippage**: $0.00 slippage on exit vs $3.00 on entry — understates friction by ~$3/trade
4. **min_trades=1 on val**: Noise survives as "signal" with 1-3 val trades
5. **Val window overlap**: 10-day overlap means regime records are not independent observations
6. **45/45 windows is expected under noise**: Multiple testing makes this metric meaningless
7. **No multiple testing correction**: 20 strategies tested on val with no FDR control
8. **Close-only VWAP**: Diverges from standard VWAP calculation
9. **Period-specific edge**: ITM3 + momentum + TSL may only work in trending markets (2024 was strongly bullish)
10. **Fitness function ignores complexity**: No penalty for overfit multi-condition genes

## What Would Increase Confidence

- Raise val `min_trades` to at least 10
- Use `forward_profitable = val_cum_pnl > 2 * round_trip_cost` instead of `> $0`
- Apply Bonferroni or Benjamini-Hochberg correction for the 20 val tests per window
- Run the unconditional null hypothesis test (always-entry + naked_itm3 + TSL)
- Fill TSL/SL exits at bar close (or bar low for sells), not theoretical level
- Add exit slippage matching entry slippage
- Use next-bar fill for entries instead of same-bar fill
- De-overlap val windows (set step_size >= val_size = 20) and re-run
- Check if edge persists in H1 2025 (out of training distribution entirely)
