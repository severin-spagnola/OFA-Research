# Audit Fixes — Options Walk-Forward Regime Search

Summary of all changes made in response to the Codex audit (`codex_audit_prompt_options.md`).

---

## Fix 1: Option Fill Timing (High)

**Problem**: Entry signals fire on the underlying bar close at time T, but option fills
used the option bar at or *before* T (`_nearest_prior_bar`). This means the option fill
price already reflects the underlying move that triggered the signal — a 1-minute
lookahead bias.

**Fix**: Added `_nearest_next_bar()` and `_get_fill_bar()` functions. When `FILL_NEXT_BAR=True`
(default), option fills use the first available option bar *strictly after* the signal
minute, with a maximum 5-minute delay (`FILL_NEXT_BAR_MAX_DELAY`). If no bar exists
within that window, the trade is skipped.

**Files changed**:
- `current/options_wf/options_backtest.py`
  - Lines 34-60: Added config flags `FILL_NEXT_BAR`, `FILL_NEXT_BAR_MAX_DELAY`, `REALISTIC_STOP_FILLS`
  - Lines 190-210: Added `_nearest_next_bar()` and `_get_fill_bar()` functions
  - Lines 270, 341-342, 430-431: Changed `_nearest_prior_bar` → `_get_fill_bar` in construct_trade for naked, credit, and debit fills

**Config flags**:
| Flag | Default | Pre-audit | Effect |
|------|---------|-----------|--------|
| `FILL_NEXT_BAR` | `True` | N/A (same-bar) | Next-bar fill for options |
| `FILL_NEXT_BAR_MAX_DELAY` | `5` | N/A | Max minutes to look ahead |

**S1 Scrutiny — Underlying Signal**: Confirmed that the underlying entry price is
informational only (`spy_entry_price=float(closes[idx])` at line 1274). The actual P&L
comes entirely from option fill prices. Fix 1 correctly addresses the real issue (option
fill timing) without adding artificial delay to the underlying signal evaluation.

---

## Fix 2: prev_close Context Buffer (High)

**Problem**: RunPod runs preloaded only train+val days, so the first train day had no
`prev_close`. Local runs preloaded the full dataset, so context was always present.
This made results non-reproducible across execution paths.

**Fix**: Added `N_CONTEXT_DAYS = 5` constant. Both execution paths now load 5 extra days
before the first train day for prev_close/prev_range enrichment. Context days are never
evaluated for trades or included in fitness metrics.

**Files changed**:
- `current/options_wf/wf_runner.py`
  - Lines 103-115: Added `N_CONTEXT_DAYS`, `VAL_MIN_TRADES`, `VAL_MIN_PNL`, `VAL_MIN_SHARPE` constants
  - Lines 117-230: Updated `evaluate_window()` to accept `context_days` parameter, load context data, assert no overlap
- `infra/wf_runpod_handler.py`
  - Lines 209-301: Updated `_handle_wf_window()` to discover and load context days from available data

**Regime DB marking**: 20 records flagged with `"prev_context_reliable": false` — these
use `gap_aligns`, `gap_opposes`, `no_gap`, `prev_range_wide`, or `prev_range_narrow`
regime filters that depend on prev_close context. They should be excluded from classifier
training until a rerun confirms their results.

---

## Fix 3: Exit Fill Prices and Slippage (High)

### Part A — Realistic Stop Fills

**Problem**: TSL and SL exits filled at the theoretical trigger level (e.g., `tsl_level`,
`sl_value`) rather than the actual bar price. When a stop triggers, the market order fills
at whatever price is available, which is at best the bar's low (for selling a long option).

**Fix**: When `REALISTIC_STOP_FILLS=True` (default):
- **Naked TSL/SL**: Fill at `max(bar_low, 0.0)` — the worst price within the bar
- **Debit/Credit spread SL**: Fill at `vals[ei]` — the actual spread value at the bar

**Files changed**:
- `current/options_wf/options_backtest.py`
  - Lines 570-573: Added `l_low` (bar low) extraction for naked options
  - Lines 606-622: TSL fill uses `bar_low` when `REALISTIC_STOP_FILLS=True`
  - Lines 623-632: SL fill uses `bar_low` when `REALISTIC_STOP_FILLS=True`
  - Lines 720-726: Credit spread SL fill uses actual bar value
  - Lines 803-810: Debit spread SL fill uses actual bar value

**Config flags**:
| Flag | Default | Pre-audit | Effect |
|------|---------|-----------|--------|
| `REALISTIC_STOP_FILLS` | `True` | N/A (theoretical) | Bar-extreme fills on stops |

**Directional audit (post-fix verification)**: Confirmed that `bar_low` is correct for
all naked option stop exits. The backtest is always *long* the option premium — even for
`direction="short"` trades (which buy puts). "Short" refers to the underlying direction
(bearish), not the option position. On stop-out, the system sells an owned option at
market; the worst fill when selling is bar_low. No short-option (written/sold) positions
exist in the naked path, so bar_high is never needed for naked stops.

### Part B — Exit Slippage

**Problem**: Exit friction included commission only ($0.65/leg), zero slippage. Entry
included $0.03/leg slippage. No justification for asymmetric treatment.

**Fix**: Added `exit_slippage_per_leg` to `CostModel` (default $0.03), and `naked_entry_cost`
/ `naked_exit_cost` properties. Updated all three `evaluate_day*` functions to use the
new properties.

**Files changed**:
- `current/options_wf/options_backtest.py`
  - Lines 78-115: Expanded `CostModel` with `exit_slippage_per_leg`, `naked_entry_cost`, `naked_exit_cost`
  - Lines 1297-1300, 1447-1450, 1013-1016: Updated friction computation in `evaluate_day`, `evaluate_day_multi`, `evaluate_day_overnight`

**Cost model before/after**:
| Metric | Pre-audit | Post-audit |
|--------|-----------|------------|
| Naked entry cost | $3.65 | $3.65 |
| Naked exit cost | $0.65 | $3.65 |
| Naked round-trip | $4.30 | $7.30 |
| Spread entry cost | $7.30 | $7.30 |
| Spread exit cost | $1.30 | $7.30 |
| Spread round-trip | $8.60 | $14.60 |

**S3 Scrutiny — Slippage Assumption**: Chain data has columns `ts, o, h, l, c, vol, vwap,
trades, strike, cp, option_ticker` — no bid/ask. The $0.03 default is a conservative
floor based on typical SPY ITM option spreads ($0.03-0.10). During momentum moves,
actual spreads can widen to $0.30-0.50. This is documented in the CostModel docstring.

---

## Fix 4: Non-Overlapping Window Analysis (Medium)

**Problem**: With step=10 and val=20, adjacent windows share 50% of validation days.
The analyzer treated 45/45 windows having a profitable strategy as evidence of consistency,
but these are not 45 independent tests.

### Part A — Analysis Correction

**Fix**: Added `compute_nonoverlapping_stats()` to `wf_analyze.py`. Greedy algorithm
selects windows whose val periods don't overlap. Both the full-dataset rate and the
non-overlapping rate are reported, clearly labeled.

**Files changed**:
- `current/options_wf/wf_analyze.py`
  - Lines 33-85: Added `compute_nonoverlapping_stats()` function
  - Lines 190-260: Added "NON-OVERLAPPING WINDOW ANALYSIS" section
  - Lines 262-295: Updated "CONCENTRATION ANALYSIS" and "VERDICT" sections

**Stats on existing regime_db.jsonl (pre-rerun)**:
- Full dataset: 509/899 = 56.6% forward profitable
- Non-overlapping (23 windows): 270/460 = 58.7% forward profitable
- All 23 non-overlapping windows had at least one profitable strategy

### Part B — Documentation

**Fix**: Added detailed docstring to `generate_windows()` in `wf_runner.py` explaining
the step/val overlap relationship and its implications.

**Files changed**:
- `current/options_wf/wf_runner.py` lines 75-90: Added overlap documentation

---

## Fix 5: Append-Only Deduplication (Medium)

**Problem**: Reruns silently appended duplicate records to regime_db.jsonl.

**Fix**:
- `wf_runner.py`: Added `_load_existing_keys()`, `--overwrite`, `--dry-run` flags.
  Default behavior checks for existing window_ids and skips duplicates with a warning.
- `wf_orchestrator.py`: Added `--overwrite`, `--dry-run` flags and pre-submission
  dedup against existing output file.

**Files changed**:
- `current/options_wf/wf_runner.py` lines 371-410, 490-515, 555-580
- `infra/wf_orchestrator.py` lines 169-172, 262-285

---

## S2: Val Pass Criteria Tightening

**Problem**: Val pass required only `val_n_trades > 0` and `val_cum_pnl > $0`. A strategy
with 1 lucky trade passed. The audit noted this was "too loose."

**Fix**: Added configurable thresholds:
- `VAL_MIN_TRADES = 10` (was: 1 effectively, via `min_trades=1`)
- `VAL_MIN_SHARPE = 0.5` (was: no check) — a strategy with 10 trades and near-zero
  Sharpe is noise, not edge. To reproduce pre-fix results, set back to `0.0`.

**VAL_MIN_PNL removed**: The initial audit added a `val_cum_pnl > 0` filter, but this
made `forward_profitable` (defined as `val_cum_pnl > 0`) tautologically True for all
surviving records. The v2.0 rerun showed 100% forward profitable (257/257) — not because
of edge, but because every record with negative val PnL was filtered out before labeling.
Strategies that pass train but lose on val must be recorded to measure the actual forward
profitable rate. Only `VAL_MIN_TRADES` and `VAL_MIN_SHARPE` are applied as pre-filters.

These are module-level constants in `wf_runner.py` (lines 143-149) and can be adjusted
for sensitivity testing. To reproduce original results, set `VAL_MIN_TRADES=1`,
`VAL_MIN_SHARPE=0.0`.

**Impact**: Cannot be computed from existing records without a rerun. Records with
`val_n_trades < 10` or `val_sharpe < 0.5` will be filtered out in new runs.

---

## What Requires a Full Rerun

The following changes alter the backtest engine itself (how trades execute), not just
post-hoc analysis. The existing regime_db.jsonl stores aggregate metrics (val_cum_pnl,
val_n_trades) but NOT individual trade-level data, so retroactive recomputation is
impossible.

| Change | Why rerun is needed |
|--------|---------------------|
| Fix 1: FILL_NEXT_BAR | Entry fill prices change → different trades taken |
| Fix 3A: REALISTIC_STOP_FILLS | Exit fill prices change → different PnL per trade |
| Fix 3B: Exit slippage | Friction changes → different pnl_net per trade |
| Fix 2: Context buffer | prev_close values change → different regime filter outcomes |
| S2: VAL_MIN_TRADES=10, VAL_MIN_SHARPE=0.5 | Noise strategies filtered; more losers now recorded |

**To run the corrected walk-forward**:
```bash
python infra/wf_orchestrator.py --overwrite --output results/options_wf/regime_db_v2.jsonl
```

Or locally:
```bash
python current/options_wf/wf_runner.py --overwrite --output results/options_wf/regime_db_v2.jsonl
```

To compare against pre-audit results, temporarily set:
```python
# In options_backtest.py:
FILL_NEXT_BAR = False
REALISTIC_STOP_FILLS = False
# In CostModel:
exit_slippage_per_leg = 0.0
# In wf_runner.py:
VAL_MIN_TRADES = 1
VAL_MIN_SHARPE = 0.0
```

---

## The Honest Baseline Number

**Cannot be computed without a full rerun.** Here's why:

The forward profitable rate after all fixes, on non-overlapping windows only, with
realistic stop fills and exit slippage, requires re-executing the entire backtest with
the corrected engine. The existing records used same-bar fills, theoretical stop fills,
and zero exit slippage — the individual trade outcomes change under the new assumptions.

**What we can say from the existing data**:
- Pre-audit non-overlapping rate: 58.7% (270/460 regimes across 23 independent windows)
- Pre-audit full rate: 56.6% (509/899)
- The non-overlapping rate is *higher* than full, suggesting the edge is not an artifact
  of val overlap

**Expected impact of fixes on rerun**:
- Fix 1 (next-bar fill): Will reduce profitable rate — fills are worse by ~1 minute of price movement
- Fix 3A (realistic stops): Will reduce profitable rate — stop exits fill at bar low instead of trigger
- Fix 3B (exit slippage): Adds ~$3.00/trade to naked costs — will flip marginal winners to losers
- Fix 2 (context buffer): Minimal impact — only 20 records affected
- S2 (val_min_trades=10): Will reduce record count by filtering low-trade regimes

**Estimated post-fix non-overlapping rate**: Unknown without rerun. Conservative estimate
based on the $3/trade exit slippage impact alone (avg_pnl ≈ $20-30, so ~10-15% drag):
roughly 45-52% on non-overlapping windows. If below 50%, the edge is indistinguishable
from noise without further analysis.

**Action required**: Run the corrected walk-forward and report the actual number.
