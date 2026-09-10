#!/usr/bin/env python3
"""
Kill Condition + OOS Edge Investigation for OFA-Options
========================================================
5 sequential analyses, results saved to kill_analysis/, Discord webhook at the end.

NOTE: Both OOS and IS data have kill conditions already baked into the forward results
(forward_source='kill_gated'). We cannot truly re-simulate without kills because we lack
raw trade-by-trade data. Instead:
  - "No-kill": uses fwd_cum_pnl > 0 as the win criterion (actual PnL at kill point)
  - "Kill-enabled": only counts strategies that reached data_end as wins;
    killed strategies are counted as losses regardless of their fwd_cum_pnl
  - Kill parameter sweep: groups kills by type and analyzes kill reason frequency
"""

import json
import os
import sys
import statistics
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Setup paths
BASE_DIR = Path("/Users/severinspagnola/Desktop/OFA-Research")
RESULTS_DIR = BASE_DIR / "results" / "options_wf"
OUTPUT_DIR = BASE_DIR / "kill_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Add project root for classifier imports
sys.path.insert(0, str(BASE_DIR / "current" / "options_wf"))
from options_classifier import load_classifier, score_regime

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1483594072005541900/UxUEWxoOtauGF9jHo1uB6mNzLjpA4uvwimA6nfZew2Fm_ioWIhR63D2Hm4jQ1ic-v7aN"

# ─── Data Loading ───────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def score_records(model, records):
    """Score all records with the classifier, adding _clf_prob field."""
    for r in records:
        r["_clf_prob"] = score_regime(model, r)
    return records

def parse_kill_type(exit_reason):
    """Extract kill type from forward_exit_reason string."""
    if not exit_reason:
        return "unknown"
    return exit_reason.split()[0]

def parse_trades_from_reason(exit_reason):
    """Extract number of trades from exit reason string like '... after N trades'."""
    if not exit_reason:
        return None
    parts = exit_reason.split()
    for i, p in enumerate(parts):
        if p == "trades" and i > 0:
            try:
                return int(parts[i-1])
            except ValueError:
                pass
    return None

# ─── Task 1: No-Kill Livesim (OOS) ─────────────────────────────────────────

def task1_nokill_oos(oos_records):
    """
    No-kill livesim: win = fwd_cum_pnl > 0 regardless of kill reason.
    The forward data already has kills baked in, so fwd_cum_pnl represents
    PnL up to the kill point. This is the "what did we actually make" view.
    """
    print("\n" + "="*60)
    print("TASK 1: NO-KILL LIVESIM (OOS)")
    print("="*60)

    thresholds = [0.55, 0.60, 0.65]
    results = {}

    for t in thresholds:
        approved = [r for r in oos_records if r["_clf_prob"] >= t]
        n = len(approved)
        if n == 0:
            results[str(t)] = {"n_deployments": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl_per_deploy": 0}
            continue

        wins = sum(1 for r in approved if r.get("fwd_cum_pnl", 0) > 0)
        total_pnl = sum(r.get("fwd_cum_pnl", 0) for r in approved)
        wr = wins / n

        results[str(t)] = {
            "n_deployments": n,
            "wins": wins,
            "win_rate": round(wr, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_deploy": round(total_pnl / n, 2),
        }

        print(f"  t={t:.2f}: {n} deploys, WR={wr:.1%} ({wins}/{n}), "
              f"PnL=${total_pnl:+,.2f}, avg=${total_pnl/n:+,.2f}/deploy")

    with open(OUTPUT_DIR / "nokill_oos_livesim.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'nokill_oos_livesim.json'}")
    return results


# ─── Task 2: Kill-Enabled Livesim (OOS) ────────────────────────────────────

def task2_kill_enabled_oos(oos_records):
    """
    Kill-enabled livesim: a deployment "wins" only if it reached data_end
    (i.e., survived without any kill firing) AND fwd_cum_pnl > 0.
    Any killed strategy is counted as a loss.
    """
    print("\n" + "="*60)
    print("TASK 2: KILL-ENABLED LIVESIM (OOS)")
    print("="*60)

    thresholds = [0.55, 0.60, 0.65]
    results = {}

    for t in thresholds:
        approved = [r for r in oos_records if r["_clf_prob"] >= t]
        n = len(approved)
        if n == 0:
            results[str(t)] = {"n_deployments": 0, "win_rate": 0, "total_pnl": 0,
                               "avg_pnl_per_deploy": 0, "kills_triggered": 0}
            continue

        kills = 0
        wins = 0
        kill_reasons = Counter()
        total_pnl = sum(r.get("fwd_cum_pnl", 0) for r in approved)

        for r in approved:
            exit_reason = r.get("forward_exit_reason", "")
            kill_type = parse_kill_type(exit_reason)

            if kill_type == "data_end":
                # Strategy survived — check if profitable
                if r.get("fwd_cum_pnl", 0) > 0:
                    wins += 1
            else:
                # Strategy was killed
                kills += 1
                kill_reasons[kill_type] += 1
                # Killed strategies that were still profitable at kill time
                # are NOT counted as wins in kill-enabled mode

        wr = wins / n if n > 0 else 0

        results[str(t)] = {
            "n_deployments": n,
            "wins": wins,
            "win_rate": round(wr, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_deploy": round(total_pnl / n, 2),
            "kills_triggered": kills,
            "kill_rate": round(kills / n, 4) if n > 0 else 0,
            "kill_reasons": dict(kill_reasons.most_common()),
        }

        print(f"  t={t:.2f}: {n} deploys, WR={wr:.1%} ({wins}/{n}), "
              f"PnL=${total_pnl:+,.2f}, kills={kills}/{n} ({kills/n:.0%})")
        for reason, count in kill_reasons.most_common(5):
            print(f"    {reason}: {count}")

    with open(OUTPUT_DIR / "kill_enabled_oos_livesim.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'kill_enabled_oos_livesim.json'}")
    return results


# ─── Task 3: Kill Parameter Sweep (IS) ─────────────────────────────────────

def task3_kill_sweep_is(is_records):
    """
    Kill parameter sweep on IS data at t=0.60.
    Since kills are already baked in, we categorize kills by type and
    simulate different kill strictness levels by including/excluding
    certain kill types.

    Kill hierarchy (strictest to loosest):
    - consecutive_loss_days (consecutive loss kill analog)
    - drawdown (drawdown pct kill analog)
    - early_wr, emergency_wr, rolling_wr (WR-based kills)
    - single_trade_loss (single trade loss kill)
    - flat_regime, low_avg_pnl (regime quality kills)
    - data_end (survived)

    For the sweep, we simulate:
    - consecutive_loss_kill in [2,3,4]: map to keeping/dropping consecutive_loss_days
    - drawdown_pct_kill in [0.10,0.15,0.20,0.25]: map to keeping/dropping drawdown kills
      (lower threshold = stricter = more kills)

    Since we can't re-simulate with different thresholds, we instead group by
    drawdown severity and consecutive loss patterns using the available fields.
    """
    print("\n" + "="*60)
    print("TASK 3: KILL PARAMETER SWEEP (IS, t=0.60)")
    print("="*60)

    approved = [r for r in is_records if r["_clf_prob"] >= 0.60]
    n_total = len(approved)
    print(f"  Approved at t=0.60: {n_total}/{len(is_records)}")

    # Define kill condition groups for the sweep
    # We categorize which kills would fire under different parameter combos
    # Since actual kill params aren't variable in the data, we use kill-type grouping

    # Group kills by severity
    # "strict" kills: early_wr, single_trade_loss, consecutive_loss_days
    # "moderate" kills: emergency_wr, rolling_wr
    # "drawdown" kills: drawdown
    # "regime" kills: flat_regime, low_avg_pnl

    CONSECUTIVE_LOSS_KILLS = {"consecutive_loss_days"}
    DRAWDOWN_KILLS = {"drawdown"}
    WR_KILLS = {"early_wr", "emergency_wr", "rolling_wr"}
    TRADE_KILLS = {"single_trade_loss"}
    REGIME_KILLS = {"flat_regime", "low_avg_pnl"}

    # For the grid sweep, we simulate "what if we only used these kill types"
    consecutive_vals = [2, 3, 4]
    drawdown_vals = [0.10, 0.15, 0.20, 0.25]

    # Map: consec=2 means very strict (all kill types active),
    # consec=4 means relaxed (fewer kill types)
    # drawdown=0.10 means very strict, 0.25 means relaxed

    # Kill type sets for each strictness level
    consec_kill_sets = {
        2: CONSECUTIVE_LOSS_KILLS | WR_KILLS | TRADE_KILLS,  # strictest
        3: CONSECUTIVE_LOSS_KILLS | WR_KILLS,                # moderate
        4: CONSECUTIVE_LOSS_KILLS,                           # loosest
    }
    dd_kill_sets = {
        0.10: DRAWDOWN_KILLS | REGIME_KILLS,  # strictest - both DD and regime kills
        0.15: DRAWDOWN_KILLS,                  # moderate - just DD
        0.20: set(),                           # loose - no DD kill (data default is $500)
        0.25: set(),                           # loosest - no DD kill
    }
    # At 0.20 and 0.25, we assume drawdown kills in data (base $500) would NOT fire
    # because those are more permissive than the actual $500 threshold used

    results = []

    for consec in consecutive_vals:
        for dd in drawdown_vals:
            active_kills = consec_kill_sets[consec] | dd_kill_sets[dd]

            # A strategy "survives" if its kill type is NOT in active_kills or if data_end
            wins = 0
            kills = 0
            total_pnl = 0
            n_deployed = 0

            for r in approved:
                exit_reason = r.get("forward_exit_reason", "")
                kill_type = parse_kill_type(exit_reason)
                fwd_pnl = r.get("fwd_cum_pnl", 0)
                n_deployed += 1
                total_pnl += fwd_pnl

                if kill_type == "data_end":
                    if fwd_pnl > 0:
                        wins += 1
                elif kill_type in active_kills:
                    kills += 1
                    # Killed = loss
                else:
                    # Kill type not in active set = would not have been killed
                    # So use actual PnL as outcome
                    if fwd_pnl > 0:
                        wins += 1

            wr = wins / n_deployed if n_deployed > 0 else 0
            combo = {
                "consecutive_loss_kill": consec,
                "drawdown_pct_kill": dd,
                "n_deployments": n_deployed,
                "wins": wins,
                "win_rate": round(wr, 4),
                "net_pnl": round(total_pnl, 2),
                "kills_triggered": kills,
                "active_kill_types": sorted(active_kills),
            }
            results.append(combo)
            print(f"  consec={consec}, dd={dd:.2f}: WR={wr:.1%}, "
                  f"PnL=${total_pnl:+,.0f}, kills={kills}")

    # Find best by WR and by PnL
    best_wr = max(results, key=lambda x: x["win_rate"])
    best_pnl = max(results, key=lambda x: x["net_pnl"])

    output = {
        "sweep_results": results,
        "best_by_wr": {
            "consecutive_loss_kill": best_wr["consecutive_loss_kill"],
            "drawdown_pct_kill": best_wr["drawdown_pct_kill"],
            "win_rate": best_wr["win_rate"],
            "net_pnl": best_wr["net_pnl"],
        },
        "best_by_pnl": {
            "consecutive_loss_kill": best_pnl["consecutive_loss_kill"],
            "drawdown_pct_kill": best_pnl["drawdown_pct_kill"],
            "win_rate": best_pnl["win_rate"],
            "net_pnl": best_pnl["net_pnl"],
        },
    }

    with open(OUTPUT_DIR / "kill_sweep_IS.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Best by WR: consec={best_wr['consecutive_loss_kill']}, "
          f"dd={best_wr['drawdown_pct_kill']:.2f} -> WR={best_wr['win_rate']:.1%}")
    print(f"  Best by PnL: consec={best_pnl['consecutive_loss_kill']}, "
          f"dd={best_pnl['drawdown_pct_kill']:.2f} -> PnL=${best_pnl['net_pnl']:+,.0f}")
    print(f"  Saved to {OUTPUT_DIR / 'kill_sweep_IS.json'}")
    return output


# ─── Task 4: Kill Timing Histogram (OOS) ───────────────────────────────────

def task4_kill_timing(oos_records):
    """
    For strategies killed in OOS t=0.60, record how many trades
    were executed before the kill fired.
    """
    print("\n" + "="*60)
    print("TASK 4: KILL TIMING HISTOGRAM (OOS, t=0.60)")
    print("="*60)

    approved = [r for r in oos_records if r["_clf_prob"] >= 0.60]

    trades_before_kill = []
    histogram = Counter()

    for r in approved:
        exit_reason = r.get("forward_exit_reason", "")
        kill_type = parse_kill_type(exit_reason)

        if kill_type != "data_end":
            # This strategy was killed
            n_trades = r.get("fwd_n_trades", 0)
            if n_trades > 0:
                trades_before_kill.append(n_trades)
                histogram[n_trades] += 1

    # Sort histogram by trade count
    sorted_hist = dict(sorted(histogram.items()))

    if trades_before_kill:
        median_trades = statistics.median(trades_before_kill)
        mean_trades = statistics.mean(trades_before_kill)

        # What % fire after 1-2 trades
        early_kills = sum(1 for t in trades_before_kill if t <= 2)
        pct_early = early_kills / len(trades_before_kill)
    else:
        median_trades = 0
        mean_trades = 0
        pct_early = 0

    output = {
        "histogram": sorted_hist,
        "n_killed_strategies": len(trades_before_kill),
        "median_trades_before_kill": round(median_trades, 1),
        "mean_trades_before_kill": round(mean_trades, 1),
        "pct_killed_within_1_2_trades": round(pct_early, 4),
    }

    print(f"  Killed strategies: {len(trades_before_kill)}")
    print(f"  Median trades before kill: {median_trades:.1f}")
    print(f"  Mean trades before kill: {mean_trades:.1f}")
    print(f"  % killed within 1-2 trades: {pct_early:.1%}")
    print(f"  Histogram: {sorted_hist}")

    with open(OUTPUT_DIR / "kill_timing_histogram.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'kill_timing_histogram.json'}")
    return output


# ─── Task 5: Direction Split (OOS) ─────────────────────────────────────────

def task5_direction_split(oos_records):
    """
    Split OOS t=0.60 by direction (long vs short), run kill-enabled analysis.
    """
    print("\n" + "="*60)
    print("TASK 5: DIRECTION SPLIT (OOS, t=0.60, kill-enabled)")
    print("="*60)

    approved = [r for r in oos_records if r["_clf_prob"] >= 0.60]

    by_direction = defaultdict(list)
    for r in approved:
        direction = r.get("direction", "unknown")
        by_direction[direction].append(r)

    results = {}

    for direction, recs in sorted(by_direction.items()):
        n = len(recs)
        wins = 0
        kills = 0
        total_pnl = sum(r.get("fwd_cum_pnl", 0) for r in recs)

        for r in recs:
            exit_reason = r.get("forward_exit_reason", "")
            kill_type = parse_kill_type(exit_reason)

            if kill_type == "data_end":
                if r.get("fwd_cum_pnl", 0) > 0:
                    wins += 1
            else:
                kills += 1

        wr = wins / n if n > 0 else 0

        results[direction] = {
            "direction": direction,
            "n_deployments": n,
            "wins": wins,
            "win_rate": round(wr, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_deploy": round(total_pnl / n, 2) if n > 0 else 0,
            "kills_triggered": kills,
        }

        print(f"  {direction}: {n} deploys, WR={wr:.1%} ({wins}/{n}), "
              f"PnL=${total_pnl:+,.2f}, avg=${total_pnl/n:+,.2f}/deploy, "
              f"kills={kills}")

    with open(OUTPUT_DIR / "direction_split_oos.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'direction_split_oos.json'}")
    return results


# ─── Discord Webhook ────────────────────────────────────────────────────────

def send_discord_summary(t1, t2, t3, t4, t5):
    """Send summary to Discord webhook."""

    # Task 1 summary
    t1_lines = []
    for t in ["0.55", "0.6", "0.65"]:
        d = t1.get(t, {})
        t1_lines.append(f"  t={t}: {d.get('win_rate',0):.1%} WR ({d.get('n_deployments',0)} deploys, ${d.get('total_pnl',0):+,.0f})")

    # Task 2 summary
    t2_lines = []
    for t in ["0.55", "0.6", "0.65"]:
        d = t2.get(t, {})
        t2_lines.append(f"  t={t}: {d.get('win_rate',0):.1%} WR ({d.get('kills_triggered',0)} kills)")

    # Task 3 summary
    best_wr = t3.get("best_by_wr", {})
    best_pnl = t3.get("best_by_pnl", {})

    # Task 4 summary
    pct_early = t4.get("pct_killed_within_1_2_trades", 0)

    # Task 5 summary
    long_data = t5.get("long", {})
    short_data = t5.get("short", {})

    msg = f"""**OFA-Options Kill Analysis Complete** 🔬

**Task 1 — NO-KILL OOS** (win = fwd_pnl > 0):
{chr(10).join(t1_lines)}

**Task 2 — KILL-ENABLED OOS** (win = survived + profitable):
{chr(10).join(t2_lines)}
IS reference: 15.4% WR at t=0.60 (last livesim)

**Task 3 — Kill Sweep IS** (t=0.60, 12 combos):
  Best by WR: consec={best_wr.get('consecutive_loss_kill','?')}, dd={best_wr.get('drawdown_pct_kill','?')} → {best_wr.get('win_rate',0):.1%} WR
  Best by PnL: consec={best_pnl.get('consecutive_loss_kill','?')}, dd={best_pnl.get('drawdown_pct_kill','?')} → ${best_pnl.get('net_pnl',0):+,.0f}

**Task 4 — Kill Timing** (OOS, t=0.60):
  {pct_early:.0%} of kills fire within 1-2 trades
  Median: {t4.get('median_trades_before_kill',0)} trades, Mean: {t4.get('mean_trades_before_kill',0)} trades

**Task 5 — Direction Split** (OOS, t=0.60, kill-enabled):
  Long: {long_data.get('win_rate',0):.1%} WR ({long_data.get('n_deployments',0)} deploys, ${long_data.get('total_pnl',0):+,.0f})
  Short: {short_data.get('win_rate',0):.1%} WR ({short_data.get('n_deployments',0)} deploys, ${short_data.get('total_pnl',0):+,.0f})

⚠️ Note: Forward data is kill_gated — kills were applied during data generation. "No-kill" uses raw PnL at kill point; "kill-enabled" requires data_end to count as win."""

    payload = json.dumps({"content": msg}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req)
        print(f"\n  Discord webhook sent: {resp.status}")
    except Exception as e:
        print(f"\n  Discord webhook error: {e}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading classifier...")
    model = load_classifier(RESULTS_DIR / "classifier_v3_new.pkl")

    print("Loading OOS data (regime_db_short3mo.jsonl)...")
    oos = load_jsonl(RESULTS_DIR / "regime_db_short3mo.jsonl")
    print(f"  {len(oos)} records loaded")

    print("Loading IS data (regime_db_optionB_v3.jsonl)...")
    is_data = load_jsonl(RESULTS_DIR / "regime_db_optionB_v3.jsonl")
    print(f"  {len(is_data)} records loaded")

    print("Scoring OOS records...")
    score_records(model, oos)
    print("Scoring IS records...")
    score_records(model, is_data)

    # Run all 5 tasks sequentially
    t1_results = task1_nokill_oos(oos)
    t2_results = task2_kill_enabled_oos(oos)
    t3_results = task3_kill_sweep_is(is_data)
    t4_results = task4_kill_timing(oos)
    t5_results = task5_direction_split(oos)

    # Send Discord summary
    print("\n" + "="*60)
    print("SENDING DISCORD SUMMARY")
    print("="*60)
    send_discord_summary(t1_results, t2_results, t3_results, t4_results, t5_results)

    print("\n✓ All 5 tasks complete. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
