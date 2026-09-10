"""
Walk-Forward Regime Analyzer
=============================
Reads current/options_vrp/results/options_vrp/regime_db.jsonl and answers the core question:
"Is there consistent edge across windows, or are results concentrated
in a few lucky windows?"

Usage:
    python current/options_vrp/wf_analyze.py
    python current/options_vrp/wf_analyze.py --file current/options_vrp/results/options_vrp/regime_db.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


def load_regime_db(path: Path) -> list[dict]:
    """Load regime records from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_nonoverlapping_stats(records: list[dict]) -> tuple[set[int], dict]:
    """Select non-overlapping windows and compute stats on that independent subset.

    Greedy algorithm: iterate windows by ID, select a window only if its val
    period doesn't overlap with any already-selected window's val period.
    This produces the largest possible independent set when windows are ordered.

    Returns:
        (selected_window_ids, stats_dict)
    """
    by_window = defaultdict(list)
    for r in records:
        by_window[r.get("window_id", 0)].append(r)

    # Build window metadata: val date ranges
    window_meta = {}
    for wid, recs in by_window.items():
        val_starts = [r.get("val_start", "") for r in recs if r.get("val_start")]
        val_ends = [r.get("val_end", "") for r in recs if r.get("val_end")]
        if val_starts and val_ends:
            window_meta[wid] = {
                "val_start": min(val_starts),
                "val_end": max(val_ends),
            }

    # Greedy selection: pick windows whose val doesn't overlap prior selections
    selected = set()
    last_val_end = ""
    for wid in sorted(window_meta.keys()):
        meta = window_meta[wid]
        if meta["val_start"] > last_val_end:
            selected.add(wid)
            last_val_end = meta["val_end"]

    # Compute stats on selected windows only
    selected_records = [r for r in records if r.get("window_id", 0) in selected]
    n_regimes = len(selected_records)
    n_profitable = sum(1 for r in selected_records if r.get("forward_profitable", False))
    rate = n_profitable / n_regimes * 100 if n_regimes else 0

    n_windows_profitable = 0
    for wid in selected:
        recs = by_window[wid]
        if any(r.get("forward_profitable", False) for r in recs):
            n_windows_profitable += 1

    return selected, {
        "n_windows": len(selected),
        "n_windows_profitable": n_windows_profitable,
        "n_regimes": n_regimes,
        "n_profitable": n_profitable,
        "forward_profitable_rate": rate,
    }


def analyze(records: list[dict]) -> None:
    """Print full analysis of regime_db records."""
    if not records:
        print("No records to analyze.")
        return

    n = len(records)
    n_profitable = sum(1 for r in records if r.get("forward_profitable", False))
    rate = n_profitable / n * 100

    print("=" * 72)
    print("WALK-FORWARD REGIME ANALYSIS")
    print("=" * 72)

    # ── Overall stats ────────────────────────────────────────────────────
    print(f"\nTotal regime records:     {n}")
    print(f"Forward profitable:       {n_profitable} ({rate:.1f}%)")
    print(f"Forward unprofitable:     {n - n_profitable} ({100 - rate:.1f}%)")

    val_pnls = [r.get("val_cum_pnl", 0) for r in records]
    print(f"Total val cum PnL:        ${sum(val_pnls):,.2f}")
    print(f"Mean val cum PnL:         ${sum(val_pnls)/n:,.2f}")
    print(f"Median val cum PnL:       ${sorted(val_pnls)[n//2]:,.2f}")

    val_sharpes = [r.get("val_sharpe", 0) for r in records]
    print(f"\nVal Sharpe distribution:")
    print(f"  Min:    {min(val_sharpes):.2f}")
    print(f"  P25:    {sorted(val_sharpes)[n//4]:.2f}")
    print(f"  Median: {sorted(val_sharpes)[n//2]:.2f}")
    print(f"  P75:    {sorted(val_sharpes)[3*n//4]:.2f}")
    print(f"  Max:    {max(val_sharpes):.2f}")

    # ── By spread type ───────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("FORWARD PROFITABLE RATE BY SPREAD TYPE")
    print(f"{'─' * 72}")

    by_spread = defaultdict(list)
    for r in records:
        spread = r.get("spread_type", "unknown")
        by_spread[spread].append(r)

    spread_rows = []
    for spread, recs in sorted(by_spread.items()):
        total = len(recs)
        pos = sum(1 for r in recs if r.get("forward_profitable", False))
        pnl = sum(r.get("val_cum_pnl", 0) for r in recs)
        spread_rows.append((spread, total, pos, pos / total * 100 if total else 0, pnl))

    spread_rows.sort(key=lambda x: -x[3])
    print(f"  {'SpreadType':<25s} {'Count':>6s} {'Pos':>5s} {'Rate':>7s} {'CumPnL':>10s}")
    for spread, total, pos, rate_a, pnl in spread_rows:
        print(f"  {spread:<25s} {total:>6d} {pos:>5d} {rate_a:>6.1f}% ${pnl:>9,.0f}")

    # ── By strike offset ─────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("FORWARD PROFITABLE RATE BY STRIKE OFFSET")
    print(f"{'─' * 72}")

    by_offset = defaultdict(list)
    for r in records:
        offset = r.get("genes", {}).get("short_strike_offset", "?")
        by_offset[offset].append(r)

    for offset, recs in sorted(by_offset.items(), key=lambda x: str(x[0])):
        total = len(recs)
        pos = sum(1 for r in recs if r.get("forward_profitable", False))
        pnl = sum(r.get("val_cum_pnl", 0) for r in recs)
        print(f"  {str(offset):<10s} {total:>5d} regimes | {pos:>4d} profitable "
              f"({pos/total*100:.1f}%) | ${pnl:>+10,.0f} cum PnL")

    # ── By TP% / SL multiple ─────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("FORWARD PROFITABLE RATE BY TP% / SL MULTIPLE")
    print(f"{'─' * 72}")

    by_tpsl = defaultdict(list)
    for r in records:
        genes = r.get("genes", {})
        tp = genes.get("tp_pct")
        sl = genes.get("sl_multiple")
        label = f"tp={tp}/sl={sl}"
        by_tpsl[label].append(r)

    tpsl_rows = []
    for label, recs in sorted(by_tpsl.items()):
        total = len(recs)
        pos = sum(1 for r in recs if r.get("forward_profitable", False))
        pnl = sum(r.get("val_cum_pnl", 0) for r in recs)
        tpsl_rows.append((label, total, pos, pos / total * 100 if total else 0, pnl))

    tpsl_rows.sort(key=lambda x: -x[3])
    print(f"  {'TP%/SL':<22s} {'Count':>6s} {'Pos':>5s} {'Rate':>7s} {'CumPnL':>10s}")
    for label, total, pos, rate_tt, pnl in tpsl_rows:
        print(f"  {label:<22s} {total:>6d} {pos:>5d} {rate_tt:>6.1f}% ${pnl:>9,.0f}")

    # ── Parameter clusters ───────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("PARAMETER CLUSTERS (top-15 by win rate, min 3 records)")
    print("  [patterns worth targeting in next-gen search]")
    print(f"{'─' * 72}")

    by_cluster = defaultdict(list)
    for r in records:
        genes = r.get("genes", {})
        key = (
            r.get("spread_type", "unknown"),
            genes.get("short_strike_offset", "?"),
            genes.get("wing_width", "?"),
            genes.get("tp_pct", "?"),
            genes.get("sl_multiple", "?"),
        )
        by_cluster[key].append(r)

    cluster_rows = []
    for key, recs in by_cluster.items():
        total = len(recs)
        if total < 3:
            continue
        pos = sum(1 for r in recs if r.get("forward_profitable", False))
        rate_c = pos / total * 100
        pnl = sum(r.get("val_cum_pnl", 0) for r in recs)
        cluster_rows.append((key, total, pos, rate_c, pnl))

    cluster_rows.sort(key=lambda x: -x[3])
    print(f"  {'SpreadType':<16s} {'Offset':>7s} {'Width':>6s} {'TP%':>6s} {'SL':>5s} "
          f"{'Count':>6s} {'Pos':>5s} {'Rate':>7s} {'CumPnL':>10s}")
    for key, total, pos, rate_c, pnl in cluster_rows[:15]:
        spread, offset, width, tp, sl = key
        print(f"  {str(spread):<16s} {str(offset):>7s} {str(width):>6s} {str(tp):>6s} {str(sl):>5s} "
              f"{total:>6d} {pos:>5d} {rate_c:>6.1f}% ${pnl:>9,.0f}")

    if not cluster_rows:
        print("  (No clusters with count >= 3)")

    # ── By window recency ────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("FORWARD PROFITABLE RATE BY TRAIN WINDOW RECENCY")
    print(f"{'─' * 72}")

    # Group windows into thirds (early / mid / late)
    window_ids = sorted(set(r.get("window_id", 0) for r in records))
    if len(window_ids) >= 3:
        third = len(window_ids) // 3
        early_ids = set(window_ids[:third])
        mid_ids = set(window_ids[third:2*third])
        late_ids = set(window_ids[2*third:])

        for label, id_set in [("Early (oldest)", early_ids),
                               ("Middle", mid_ids),
                               ("Late (most recent)", late_ids)]:
            recs = [r for r in records if r.get("window_id", 0) in id_set]
            total = len(recs)
            pos = sum(1 for r in recs if r.get("forward_profitable", False))
            pnl = sum(r.get("val_cum_pnl", 0) for r in recs)
            if total:
                print(f"  {label:<25s} {total:>5d} regimes | {pos:>4d} profitable "
                      f"({pos/total*100:.1f}%) | ${pnl:>+10,.0f}")
    else:
        print("  (Not enough distinct windows for recency breakdown)")

    # ── Per-window summary table ─────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("PER-WINDOW SUMMARY (best val strategy per window)")
    print(f"{'─' * 72}")

    by_window = defaultdict(list)
    for r in records:
        by_window[r.get("window_id", 0)].append(r)

    print(f"  {'WinID':>5s}  {'Train':>21s}  {'Val':>21s}  "
          f"{'#Reg':>4s} {'#Pos':>4s}  {'BestSharpe':>10s} {'BestPnL':>9s}  {'BestDesc'}")

    windows_with_any_profit = 0
    windows_total = 0

    for wid in sorted(by_window.keys()):
        recs = by_window[wid]
        windows_total += 1
        n_pos = sum(1 for r in recs if r.get("forward_profitable", False))
        if n_pos > 0:
            windows_with_any_profit += 1

        # Best by val sharpe
        best = max(recs, key=lambda r: r.get("val_sharpe", -999))
        train_range = f"{best.get('train_start', '?')}→{best.get('train_end', '?')[-5:]}"
        val_range = f"{best.get('val_start', '?')}→{best.get('val_end', '?')[-5:]}"
        desc = best.get("genes_desc", "")
        if len(desc) > 50:
            desc = desc[:47] + "..."

        print(f"  {wid:>5d}  {train_range:>21s}  {val_range:>21s}  "
              f"{len(recs):>4d} {n_pos:>4d}  {best.get('val_sharpe', 0):>10.2f} "
              f"${best.get('val_cum_pnl', 0):>8,.0f}  {desc}")

    # ── Non-overlapping window analysis ──────────────────────────────────
    print(f"\n{'─' * 72}")
    print("NON-OVERLAPPING WINDOW ANALYSIS")
    print(f"{'─' * 72}")

    nonoverlap_ids, nonoverlap_stats = compute_nonoverlapping_stats(records)
    n_nonoverlap = nonoverlap_stats["n_windows"]
    n_nonoverlap_profit = nonoverlap_stats["n_windows_profitable"]
    nonoverlap_rate = nonoverlap_stats["forward_profitable_rate"]
    nonoverlap_n = nonoverlap_stats["n_regimes"]
    nonoverlap_n_pos = nonoverlap_stats["n_profitable"]

    print(f"\n  Non-overlapping windows selected: {n_nonoverlap} "
          f"(from {windows_total} total)")
    print(f"  Window IDs: {sorted(nonoverlap_ids)}")
    print(f"  Regimes in non-overlapping set: {nonoverlap_n}")
    print(f"  Forward profitable: {nonoverlap_n_pos}/{nonoverlap_n} "
          f"({nonoverlap_rate:.1f}%)")
    print(f"  Non-overlapping windows with edge: "
          f"{n_nonoverlap_profit}/{n_nonoverlap}")

    # ── Concentration analysis ───────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("EDGE CONCENTRATION ANALYSIS")
    print(f"{'─' * 72}")

    print(f"\n  Windows with at least 1 profitable strategy: "
          f"{windows_with_any_profit}/{windows_total} "
          f"({windows_with_any_profit/windows_total*100:.0f}%)")
    print(f"  NOTE: Adjacent windows share ~50% of val days (step=10, val=20).")
    print(f"        Non-overlapping independent subset: "
          f"{n_nonoverlap_profit}/{n_nonoverlap} windows with edge.")

    # How many windows contribute >80% of total profitable PnL?
    window_pnls = {}
    for wid, recs in by_window.items():
        pos_pnl = sum(r.get("val_cum_pnl", 0) for r in recs if r.get("forward_profitable", False))
        window_pnls[wid] = pos_pnl

    total_pos_pnl = sum(p for p in window_pnls.values() if p > 0)
    if total_pos_pnl > 0:
        sorted_window_pnls = sorted(window_pnls.values(), reverse=True)
        cum = 0
        n_for_80 = 0
        for p in sorted_window_pnls:
            if p <= 0:
                continue
            cum += p
            n_for_80 += 1
            if cum >= total_pos_pnl * 0.80:
                break

        print(f"  Windows contributing 80% of positive PnL: {n_for_80}/{windows_total}")
        if n_for_80 <= 3 and windows_total > 10:
            print(f"  ⚠  CONCENTRATED — edge comes from {n_for_80} windows, "
                  f"not broadly distributed")
        elif n_for_80 >= windows_total * 0.3:
            print(f"  ✓  DISTRIBUTED — edge spread across {n_for_80} windows")
        else:
            print(f"  ~  MODERATE — edge somewhat concentrated")
    else:
        print("  No positive forward PnL across any window.")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    fwd_rate = n_profitable / n * 100 if n else 0

    print(f"\n  ALL WINDOWS (includes overlap):")
    print(f"    Forward profitable rate: {fwd_rate:.1f}% ({n_profitable}/{n})")
    print(f"    Windows with edge: {windows_with_any_profit}/{windows_total}")

    print(f"\n  NON-OVERLAPPING WINDOWS (independent subset):")
    print(f"    Forward profitable rate: {nonoverlap_rate:.1f}% "
          f"({nonoverlap_n_pos}/{nonoverlap_n})")
    print(f"    Windows with edge: {n_nonoverlap_profit}/{n_nonoverlap}")

    # Use non-overlapping rate for the honest verdict
    if nonoverlap_rate >= 40 and n_nonoverlap_profit >= n_nonoverlap * 0.5:
        print(f"\n  → CONSISTENT EDGE (non-overlapping) — "
              f"worth building a classifier on this data")
    elif nonoverlap_rate >= 30:
        print(f"\n  → MARGINAL EDGE (non-overlapping) — classifier might extract "
              f"signal, but base rate is close to noise")
    else:
        print(f"\n  → NO CONSISTENT EDGE (non-overlapping) — "
              f"stop here, do not build classifier")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze walk-forward regime database")
    parser.add_argument("--file", type=str,
                        default="current/options_vrp/results/options_vrp/regime_db.jsonl")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run wf_runner.py or wf_orchestrator.py first to generate regime data.")
        sys.exit(1)

    records = load_regime_db(path)
    analyze(records)


if __name__ == "__main__":
    main()
