#!/usr/bin/env python3
"""
Path D Livesim: No-Filter OOS Baseline (Fundamental Criteria Only, No Classifier)
==================================================================================
Tests what the strategy does on OOS windows 420-457 using ONLY fundamental filters
(val_sharpe >= 0.5 AND val_n_trades >= 10) — no ML classifier.

Three kill variants:
  1. NO kills at all (win = fwd_cum_pnl > 0)
  2. Drawdown-only kill at dd_kill=0.20 (≈$500 base, matching existing data)
  3. Drawdown-only kill at dd_kill=0.15 (≈$375, stricter)

LIMITATION: Forward data is kill_gated at source — fwd_cum_pnl reflects PnL at
kill point, NOT at period end. We cannot truly simulate "no kill" because the
forward period was truncated when the original kill fired. Variant 1 ("no kills")
really means "was the PnL positive at the moment the kill stopped tracking?"
"""

import json
import os
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/Users/severinspagnola/Desktop/OFA-Research")
DATA_PATH = BASE_DIR / "results" / "options_wf" / "regime_db_short3mo.jsonl"
OUTPUT_PATH = BASE_DIR / "path_d_results.json"

# The existing drawdown kill base is $500.
# dd_kill=0.20 maps to ~$500 (20% of ~$2500 notional)
# dd_kill=0.15 maps to ~$375 (15% of ~$2500 notional)
DD_THRESHOLD_V2 = 500.0   # matches existing data base
DD_THRESHOLD_V3 = 375.0   # stricter


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_kill_type(exit_reason):
    if not exit_reason:
        return "unknown"
    return exit_reason.split()[0]


def run_variant(records, variant_name, kill_logic):
    """
    Run a kill variant over all records.
    kill_logic(record) -> True if killed, False if survived
    """
    n = len(records)
    wins = 0
    kills_triggered = 0
    total_pnl = 0.0
    kill_reasons = Counter()

    for r in records:
        pnl = r.get("fwd_cum_pnl", 0)
        total_pnl += pnl

        if kill_logic(r):
            kills_triggered += 1
            kill_reasons[parse_kill_type(r.get("forward_exit_reason", ""))] += 1
            # Killed = loss
        else:
            if pnl > 0:
                wins += 1

    wr = wins / n if n > 0 else 0.0
    return {
        "variant": variant_name,
        "n_qualifying": n,
        "n_deployed": n,
        "wins": wins,
        "win_rate": round(wr, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_deploy": round(total_pnl / n, 2) if n > 0 else 0,
        "kills_triggered": kills_triggered,
        "kill_reasons": dict(kill_reasons.most_common()) if kill_reasons else {},
    }


def main():
    print("=" * 60)
    print("PATH D LIVESIM — No-Classifier OOS Baseline")
    print("=" * 60)

    records = load_jsonl(DATA_PATH)
    total = len(records)
    print(f"Loaded {total} records from regime_db_short3mo.jsonl")

    # Fundamental filter
    qualifying = [r for r in records
                  if r.get("val_sharpe", 0) >= 0.5
                  and r.get("val_n_trades", 0) >= 10]
    n_qual = len(qualifying)
    print(f"Fundamental filter (val_sharpe>=0.5, val_n_trades>=10): {n_qual}/{total} pass ({n_qual/total:.1%})")

    # Sort by window_id for deployment order
    qualifying.sort(key=lambda r: r.get("window_id", 0))

    # Show window coverage
    windows = sorted(set(r["window_id"] for r in qualifying))
    print(f"Windows: {min(windows)}-{max(windows)} ({len(windows)} unique windows)")

    # Per-window stats
    from collections import defaultdict
    by_window = defaultdict(list)
    for r in qualifying:
        by_window[r["window_id"]].append(r)
    print(f"Records per window: min={min(len(v) for v in by_window.values())}, "
          f"max={max(len(v) for v in by_window.values())}, "
          f"avg={sum(len(v) for v in by_window.values())/len(by_window):.1f}")

    # Direction breakdown
    directions = Counter(r.get("direction", "unknown") for r in qualifying)
    print(f"Direction breakdown: {dict(directions)}")

    # --- Variant 1: NO kills ---
    def no_kill(r):
        return False  # nothing is killed

    v1 = run_variant(qualifying, "NO_KILLS", no_kill)

    # --- Variant 2: Drawdown-only kill at $500 (dd_kill=0.20) ---
    # Only count as killed if exit_reason is "drawdown"
    # (other kill types would not exist under dd-only regime)
    def dd_only_v2(r):
        exit_reason = r.get("forward_exit_reason", "")
        kill_type = parse_kill_type(exit_reason)
        if kill_type == "drawdown":
            return True  # DD kill fires (base $500 matches dd_kill=0.20)
        # For non-DD killed strategies: under dd-only regime, would DD have fired?
        # Check if fwd_max_dd >= threshold
        if kill_type != "data_end" and r.get("fwd_max_dd", 0) >= DD_THRESHOLD_V2:
            return True  # Would have been DD-killed even though original kill was different
        return False

    v2 = run_variant(qualifying, "DD_ONLY_0.20", dd_only_v2)

    # --- Variant 3: Drawdown-only kill at $375 (dd_kill=0.15, stricter) ---
    def dd_only_v3(r):
        exit_reason = r.get("forward_exit_reason", "")
        kill_type = parse_kill_type(exit_reason)
        if kill_type == "drawdown":
            return True  # Original DD kill definitely fires at stricter threshold
        # For non-DD killed strategies: check if fwd_max_dd >= stricter threshold
        if kill_type != "data_end" and r.get("fwd_max_dd", 0) >= DD_THRESHOLD_V3:
            return True
        return False

    v3 = run_variant(qualifying, "DD_ONLY_0.15", dd_only_v3)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for v in [v1, v2, v3]:
        print(f"\n  {v['variant']}:")
        print(f"    Qualifying/Deployed: {v['n_qualifying']}")
        print(f"    Win Rate: {v['win_rate']:.1%} ({v['wins']}/{v['n_deployed']})")
        print(f"    Total PnL: ${v['total_pnl']:+,.2f}")
        print(f"    Avg PnL/Deploy: ${v['avg_pnl_per_deploy']:+,.2f}")
        print(f"    Kills Triggered: {v['kills_triggered']}")
        if v['kill_reasons']:
            print(f"    Kill Reasons: {v['kill_reasons']}")

    # Build output
    output = {
        "description": "Path D: No-classifier OOS livesim on regime_db_short3mo.jsonl",
        "data_file": str(DATA_PATH),
        "total_records": total,
        "fundamental_filter": "val_sharpe >= 0.5 AND val_n_trades >= 10",
        "n_passing_filter": n_qual,
        "filter_pass_rate": round(n_qual / total, 4),
        "windows_covered": f"{min(windows)}-{max(windows)} ({len(windows)} unique)",
        "direction_breakdown": dict(directions),
        "limitation": (
            "Forward data is kill_gated at source. fwd_cum_pnl reflects PnL at kill "
            "point, not at period end. 'No kills' variant uses PnL at the moment tracking "
            "stopped. DD-only variants re-classify which kills would fire under a DD-only "
            "regime, but cannot extend the forward period past the original kill point. "
            "For strategies killed by non-DD reasons, we check if fwd_max_dd >= threshold "
            "to determine if DD kill would have fired independently."
        ),
        "variants": {
            "NO_KILLS": v1,
            "DD_ONLY_0.20": v2,
            "DD_ONLY_0.15": v3,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
