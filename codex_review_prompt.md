# Codex Review: Kill Conditions, Live Score, Contract Scaling & Dynamic DD

Review the interaction between these four systems in `current/options_research/options_walkforward.py` and `infra/options_runpod_handler.py`. Verify correctness and flag any bugs, logical inconsistencies, or edge cases.

## Systems to review

### 1. Contract Scaling (`get_contracts`, `KillConfig.scale_every`)
- `contracts = 1 + int(cum_pnl / scale_every)` where `scale_every=750`
- Only scales up when `scale_contracts=True` and `cum_pnl > 0`
- `contracts_history` list tracks how many contracts each trade was executed with

### 2. Live Performance Score (`live_score_multiplier`)
- Returns a multiplier (1.0–2.0) based on return:DD ratio, avg PnL/trade, days alive
- Only activates when `cum_pnl > 0` and `n_trades >= live_score_min_trades` (default 10)
- Used to widen DD and STL kill thresholds for proven strategies

### 3. Dynamic DD Allowance (in `check_kill`)
- `eff_dd_limit = base_dd * live_score_mult * cur_contracts`
- DD allowance scales with both live performance AND position size
- STL stays per-contract: `eff_stl_limit = base_stl * live_score_mult` (no contract scaling)

### 4. Multi-Entry (`evaluate_day_multi` in `options_backtest.py`)
- After a trade exits via TP/SL/TSL/time_stop, scan for new entries on same day
- EOD/hold exits end the day (no re-entry)
- Each re-entry trade goes through the same contract scaling logic

## Specific things to verify

1. **DD is computed on the scaled equity curve** (trade.pnl_net * contracts for each trade), and the DD _allowance_ also scales with contracts. Confirm there's no double-counting or mismatch — the drawdown measured and the threshold it's compared against should be consistent.

2. **Live score uses the scaled cum_pnl and scaled max_dd** (not per-contract). Confirm the inputs to `live_score_multiplier()` in `check_kill()` are the scaled values.

3. **STL is per-contract** — it checks `trades[-1].pnl_net` (unscaled) against the STL limit. The STL limit gets live_score boost but NOT contract scaling. This is intentional (measures trade quality). Verify this is correct.

4. **`cur_contracts` for DD scaling** uses `contracts_history[-1]` — the contract count of the most recent trade. Is this the right choice, or should it use the current `get_contracts(cum_pnl)` instead? Consider: after a losing streak, cum_pnl drops, so `get_contracts` might return fewer contracts than what was actually traded. But `contracts_history[-1]` reflects what was actually traded. Which is more correct for determining the DD allowance?

5. **Edge case: strategy with 0 trades** — `check_kill` returns early before computing any scaling. Verify this is safe.

6. **Edge case: `contracts_history` is None or empty** — falls back to unscaled PnL and `cur_contracts=1`. Verify the fallback is consistent.

7. **Edge case: negative cum_pnl** — `live_score_multiplier` returns 1.0, `get_contracts` returns 1. DD allowance = base. Verify a losing strategy doesn't accidentally get bonus DD.

8. **Flat regime / avg_pnl_floor checks** — these use `cum_pnl` (scaled). With contract scaling, a strategy trading 2 contracts with $10/trade avg would show $20/trade scaled avg. Is this correct, or should these checks use per-contract PnL? The `flat_regime_pnl` threshold ($150) and `avg_pnl_floor` ($3) were calibrated for 1-contract trading.

9. **Negative trajectory check** — uses scaled `cum_pnl` against `negative_trajectory_pnl=-500`. With 2 contracts, a $250 per-contract loss shows as $500 scaled. Should this threshold also scale with contracts?

10. **Trade execution order in `run_live_sim`** — Step 1 checks kills, Step 2 retrains, Step 3 executes trades. Confirm that `contracts_history` is always in sync with `trades` at kill-check time (i.e., both lists have the same length).

11. **RunPod handler passthrough** — confirm `live_score`, `live_score_min_trades`, `scale_contracts`, `scale_every` are all in the KillConfig passthrough list in `_handle_livesim()`.

12. **Progress logging** — the progress log computes `total_contracts` by calling `get_contracts()` for each active strategy. This is for display only. Verify it doesn't mutate any state.

## Files to read
- `current/options_research/options_walkforward.py` — `KillConfig`, `live_score_multiplier`, `get_contracts`, `check_kill`, `ActiveStrategy`, `run_live_sim`
- `current/options_research/options_backtest.py` — `evaluate_day_multi` (multi-entry logic)
- `infra/options_runpod_handler.py` — `_handle_livesim`, `_handle_diagnostic`
