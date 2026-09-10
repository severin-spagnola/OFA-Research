"""
Livesim replay driver — scores regime_db records through a trained classifier
and reports portfolio-level metrics as if the classifier gate had been active.

Usage:
    python run_livesim.py \
        --classifier classifier_v4_baseline.pkl \
        --data regime_db_v4_merged.jsonl \
        --threshold 0.50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Add project root so we can import the classifier module
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "current" / "options_wf"))

from options_classifier import load_classifier, score_regime


def main():
    parser = argparse.ArgumentParser(description="Livesim replay with classifier gate")
    parser.add_argument("--classifier", type=str, required=True,
                        help="Path to trained classifier .pkl")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to regime_db JSONL")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Classifier probability threshold (default 0.50)")
    args = parser.parse_args()

    # Load classifier
    model = load_classifier(args.classifier)

    # Load regimes
    regimes = []
    with open(args.data) as f:
        for line in f:
            regimes.append(json.loads(line))

    print(f"Loaded {len(regimes)} regimes from {args.data}")
    print(f"Classifier threshold: {args.threshold:.2f}")
    print()

    # Score each regime through the classifier
    approved = []
    rejected = []
    for r in regimes:
        prob = score_regime(model, r)
        r["_clf_prob"] = prob
        if prob >= args.threshold:
            approved.append(r)
        else:
            rejected.append(r)

    print(f"Classifier gate results:")
    print(f"  Approved: {len(approved)} / {len(regimes)} ({len(approved)/len(regimes):.1%})")
    print(f"  Rejected: {len(rejected)} / {len(regimes)} ({len(rejected)/len(regimes):.1%})")
    print()

    # Compute livesim metrics on APPROVED strategies
    # Each approved regime represents one "deployment" — its forward test is the live result
    cum_pnl = 0.0
    total_fwd_days = 0
    kill_reasons = Counter()
    n_killed = 0
    n_profitable = 0
    all_fwd_pnls = []

    for r in approved:
        fwd_pnl = r.get("fwd_cum_pnl", 0.0)
        fwd_days = r.get("forward_days", 0)
        exit_reason = r.get("forward_exit_reason", "unknown")
        profitable = r.get("forward_profitable", False)

        cum_pnl += fwd_pnl
        total_fwd_days += fwd_days
        all_fwd_pnls.append(fwd_pnl)

        if profitable:
            n_profitable += 1

        # Parse kill reason (extract the kill condition name)
        if exit_reason and exit_reason != "data_end":
            # Kill reasons look like "emergency_wr 16.7% < 20% after 6 trades"
            # or "drawdown $503 >= $500" — extract the first word as the condition
            kill_condition = exit_reason.split()[0] if exit_reason else "unknown"
            kill_reasons[kill_condition] += 1
            n_killed += 1

    # Also compute for ALL (no classifier) for comparison
    cum_pnl_all = sum(r.get("fwd_cum_pnl", 0.0) for r in regimes)
    n_profitable_all = sum(1 for r in regimes if r.get("forward_profitable", False))
    n_killed_all = sum(1 for r in regimes
                       if r.get("forward_exit_reason", "") and
                       r.get("forward_exit_reason", "") != "data_end")
    kill_reasons_all = Counter()
    for r in regimes:
        er = r.get("forward_exit_reason", "")
        if er and er != "data_end":
            kill_reasons_all[er.split()[0]] += 1

    # Trading days: sum of forward_days across approved deployments
    # (each deployment runs for forward_days before being killed)
    trading_days = total_fwd_days

    kill_rate = n_killed / len(approved) * 100 if approved else 0

    print("=" * 60)
    print("LIVESIM RESULTS (classifier-gated, threshold={:.2f})".format(args.threshold))
    print("=" * 60)
    print()
    print(f"  Cumulative PnL:           ${cum_pnl:+,.2f}")
    print(f"  Total forward days:       {trading_days}")
    print(f"  Strategies approved:      {len(approved)}")
    print(f"  Strategies profitable:    {n_profitable} / {len(approved)} "
          f"({n_profitable/len(approved):.1%})" if approved else "")
    print(f"  Avg PnL per deployment:   ${cum_pnl/len(approved):+,.2f}" if approved else "")
    print()
    print(f"  Strategies killed:        {n_killed} / {len(approved)} (kill rate: {kill_rate:.1f}%)")
    print()
    print("  Kill reason breakdown:")
    for reason, count in kill_reasons.most_common():
        print(f"    {reason:30s} {count:4d} ({count/n_killed*100:.1f}%)" if n_killed else "")
    print()

    print("=" * 60)
    print("COMPARISON: No classifier (all {} deployments)".format(len(regimes)))
    print("=" * 60)
    print(f"  Cumulative PnL:           ${cum_pnl_all:+,.2f}")
    print(f"  Strategies profitable:    {n_profitable_all} / {len(regimes)} "
          f"({n_profitable_all/len(regimes):.1%})")
    print(f"  Strategies killed:        {n_killed_all} / {len(regimes)} "
          f"({n_killed_all/len(regimes)*100:.1f}%)")
    print()
    print("  Kill reason breakdown (all):")
    for reason, count in kill_reasons_all.most_common():
        print(f"    {reason:30s} {count:4d} ({count/n_killed_all*100:.1f}%)" if n_killed_all else "")
    print()

    # Classifier improvement
    if approved:
        appr_wr = n_profitable / len(approved)
        all_wr = n_profitable_all / len(regimes)
        print("=" * 60)
        print("CLASSIFIER LIFT")
        print("=" * 60)
        print(f"  Base rate (no classifier):  {all_wr:.1%}")
        print(f"  Approved win rate:          {appr_wr:.1%}")
        print(f"  Lift:                       {appr_wr - all_wr:+.1%}")
        print(f"  Avg PnL (all):              ${cum_pnl_all/len(regimes):+,.2f}")
        print(f"  Avg PnL (approved):         ${cum_pnl/len(approved):+,.2f}")


if __name__ == "__main__":
    main()
