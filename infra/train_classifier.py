"""
Train Regime Classifier — Predict forward profitability from training metrics.
==============================================================================
Uses training-side features + market snapshot to predict whether a regime
will be profitable in forward testing.

Features:
- Training metrics (fitness, sharpe, PF, trade count, robustness)
- Strategy structure (complexity, SL/TP, direction, archetype)
- Market snapshot (volatility, trend, overnight range, gaps)

Model: Gradient Boosted Trees (sklearn) with class weighting for imbalance.
Validation: Temporal split (first 70% train, last 30% test) to prevent
lookahead bias.

Usage:
    python infra/train_classifier.py
    python infra/train_classifier.py --save   # save model to results/
"""
from __future__ import annotations

import json
import glob
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _REPO_ROOT / "results"

# Features available at deployment time (no forward-looking data)
TRAINING_FEATURES = [
    "train_fitness",
    "train_sharpe",
    "train_pf",
    "train_n_trades",
    # Robustness
    "train_mc_p",
    "train_pf_stability",
    "train_wr_stability",
    "train_top_trade_pct",
    "train_remove_best_positive",
    "train_payoff_consistency",
    "train_expectancy",
    "train_kelly",
    "train_win_loss_ratio",
    "train_avg_win",
    "train_avg_loss",
    "train_max_consec_losses",
    "train_trades_per_day",
]

STRATEGY_FEATURES = [
    "n_conditions",
    "complexity",
    "has_tsl",
    "sl_pts",
    "tp_pts",
    "direction",
]

SNAPSHOT_FEATURES = [
    "daily_vol_pct",
    "on_range_pct",
    "avg_on_range_pts",
    "ret_10d_pct",
    "ret_20d_pct",
    "range_ratio_5d_20d",
    "max_range_5d_pct",
    "gap_pct",
    "dir_consistency_pct",
    "streak_days",
]

ALL_FEATURES = TRAINING_FEATURES + STRATEGY_FEATURES + SNAPSHOT_FEATURES


def load_regime_records() -> list[dict]:
    """Load all regime records from overnight results."""
    records = []
    for f in sorted(_RESULTS_DIR.glob("overnight_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get("regimes", []):
            records.append(r)
    return records


def build_feature_matrix(records: list[dict]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build feature matrix and labels from regime records.

    Returns (X, y, dates) where dates is forward_start for temporal sorting.
    """
    rows = []
    labels = []
    dates = []

    for r in records:
        fpnl = r.get("forward_pnl", 0) or 0
        ft = r.get("forward_trades", 0) or 0
        if ft < 1:
            continue

        snap = r.get("market_snapshot") or {}
        if "error" in snap:
            snap = {}

        rob = r.get("train_robustness") or {}
        params = r.get("winner_params") or {}
        entry = params.get("entry", {})
        exit_def = params.get("exit", {})

        # Strategy structure
        n_conditions = len(entry.get("conditions", []))
        has_confirmation = 1 if entry.get("confirmation") else 0
        has_filter = 1 if entry.get("filter") else 0
        has_filter2 = 1 if entry.get("filter2") else 0
        complexity = n_conditions + has_confirmation + has_filter + has_filter2
        has_tsl = 1 if (exit_def.get("be_trigger_pts") or exit_def.get("trail_distance_pts")) else 0
        sl = exit_def.get("stop_loss_pts", 0) or 0
        tp = exit_def.get("take_profit_pts", 0) or 0
        direction = 1 if entry.get("direction") == "long" else 0

        row = {
            # Training metrics
            "train_fitness": r.get("winner_fitness", 0) or 0,
            "train_sharpe": r.get("train_sharpe", 0) or 0,
            "train_pf": r.get("train_pf", 0) or 0,
            "train_n_trades": r.get("train_n_trades", 0) or 0,
            # Robustness
            "train_mc_p": rob.get("mc_p_value", 0.5),
            "train_pf_stability": rob.get("pf_stability_ratio", 0),
            "train_wr_stability": rob.get("wr_stability_ratio", 0),
            "train_top_trade_pct": rob.get("top_trade_pct", 50),
            "train_remove_best_positive": 1 if rob.get("remove_best_still_positive") else 0,
            "train_payoff_consistency": rob.get("payoff_consistency", 1.0),
            "train_expectancy": rob.get("expectancy_per_trade", 0),
            "train_kelly": rob.get("kelly_fraction", 0),
            "train_win_loss_ratio": rob.get("win_loss_ratio", 1.0),
            "train_avg_win": rob.get("avg_win_dollars", 0),
            "train_avg_loss": rob.get("avg_loss_dollars", 0),
            "train_max_consec_losses": rob.get("max_consecutive_losses", 0),
            "train_trades_per_day": rob.get("trades_per_day", 0),
            # Strategy structure
            "n_conditions": n_conditions,
            "complexity": complexity,
            "has_tsl": has_tsl,
            "sl_pts": sl,
            "tp_pts": tp,
            "direction": direction,
            # Snapshot
            "daily_vol_pct": snap.get("daily_vol_pct", 1.0),
            "on_range_pct": snap.get("on_range_pct", 0.1),
            "avg_on_range_pts": snap.get("avg_on_range_pts", 5.0),
            "ret_10d_pct": snap.get("ret_10d_pct", 0.0),
            "ret_20d_pct": snap.get("ret_20d_pct", 0.0),
            "range_ratio_5d_20d": snap.get("range_ratio_5d_20d", 1.0),
            "max_range_5d_pct": snap.get("max_range_5d_pct", 1.0),
            "gap_pct": snap.get("gap_pct", 50.0),
            "dir_consistency_pct": snap.get("dir_consistency_pct", 50.0),
            "streak_days": snap.get("streak_days", 1),
        }

        rows.append(row)
        labels.append(1 if fpnl > 0 else 0)
        dates.append(r.get("forward_start", ""))

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    return X, y, dates


def train_and_evaluate(save: bool = False):
    """Train classifier with temporal split and print evaluation."""
    records = load_regime_records()
    print(f"Loaded {len(records)} regime records")

    X, y, dates = build_feature_matrix(records)
    print(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(Counter(y))}")

    # Temporal split — sort by forward_start, train on first 70%
    sort_idx = np.argsort(dates)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    dates_sorted = [dates[i] for i in sort_idx]

    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"\nTemporal split: train={len(X_train)} ({dates_sorted[0][:10]} to "
          f"{dates_sorted[split-1][:10]}), test={len(X_test)} "
          f"({dates_sorted[split][:10]} to {dates_sorted[-1][:10]})")
    print(f"Train class dist: {dict(Counter(y_train))}")
    print(f"Test class dist:  {dict(Counter(y_test))}")

    # Handle NaN/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train with class weighting
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)
    weights = y_train.map({0: 1.0, 1: scale_pos})

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=weights)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"CLASSIFIER EVALUATION (temporal holdout)")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Loser", "Winner"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        print(f"\nAUC-ROC: {auc:.3f}")
        print(f"Avg Precision: {ap:.3f}")
    else:
        print(f"\nOnly one class in test set — can't compute AUC")

    # Feature importance
    importances = clf.feature_importances_
    feat_imp = sorted(zip(ALL_FEATURES, importances), key=lambda x: -x[1])
    print(f"\nTop 15 feature importances:")
    for feat, imp in feat_imp[:15]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<25s} {imp:.4f} {bar}")

    # P&L simulation: what if we only traded classifier "winners"?
    print(f"\n{'='*60}")
    print(f"P&L SIMULATION (test set)")
    print(f"{'='*60}")

    test_records = [records[sort_idx[i]] for i in range(split, len(sort_idx))]
    all_pnl = sum(r.get("forward_pnl", 0) or 0 for r in test_records)
    pred_win_pnl = sum(
        (r.get("forward_pnl", 0) or 0)
        for r, pred in zip(test_records, y_pred) if pred == 1
    )
    pred_lose_pnl = sum(
        (r.get("forward_pnl", 0) or 0)
        for r, pred in zip(test_records, y_pred) if pred == 0
    )
    n_deployed = sum(y_pred)
    n_skipped = len(y_pred) - n_deployed

    print(f"  Unfiltered:     {len(test_records)} regimes  P&L=${all_pnl:>9,.0f}")
    print(f"  Clf 'deploy':   {n_deployed} regimes  P&L=${pred_win_pnl:>9,.0f}")
    print(f"  Clf 'skip':     {n_skipped} regimes  P&L=${pred_lose_pnl:>9,.0f}")
    print(f"  Clf value:      ${all_pnl - pred_win_pnl:>+9,.0f} avoided")

    # Probability-bucketed analysis
    print(f"\nProbability buckets (test set):")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bucket_pnl = sum(
            (r.get("forward_pnl", 0) or 0)
            for r, m in zip(test_records, mask) if m
        )
        bucket_wr = sum(
            1 for r, m in zip(test_records, mask)
            if m and (r.get("forward_pnl", 0) or 0) > 0
        )
        print(f"  [{lo:.1f}-{hi:.1f}): n={mask.sum():>3}  "
              f"P&L=${bucket_pnl:>8,.0f}  WR={bucket_wr/mask.sum()*100:.0f}%")

    if save:
        # Save model
        model_path = _RESULTS_DIR / "classifier_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": clf,
                "features": ALL_FEATURES,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_date_range": (dates_sorted[0], dates_sorted[split-1]),
                "test_date_range": (dates_sorted[split], dates_sorted[-1]),
            }, f)
        print(f"\nSaved model to {model_path}")

        # Save feature importances
        imp_path = _RESULTS_DIR / "classifier_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(feat_imp, f, indent=2)
        print(f"Saved feature importances to {imp_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    train_and_evaluate(save=args.save)
