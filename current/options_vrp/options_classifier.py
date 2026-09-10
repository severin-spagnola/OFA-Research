"""
Options Walk-Forward Classifier (v2 — 3-way split)
====================================================
LightGBM classifier trained on regime_db.jsonl (3-way split records) to predict
forward_profitable — measured on a genuine forward period, NOT val.

Features:
  - Train fitness metrics (sharpe, PF, WR, n_trades, PnL, DD, etc.)
  - Val fitness metrics (now safe — val is a filter, not the label source)
  - Gene structure (archetype, direction, moneyness, exit params)
  - Robustness metrics from train trades (mc_p_value, remove_best, stability)
  - Derived ratios (val/train sharpe, PnL ratio, overfitting signals)

Usage:
    # Train and save model
    python options_classifier.py --train --data regime_db_v3.jsonl --out classifier.pkl

    # Score a regime dict
    from options_classifier import load_classifier, score_regime
    model = load_classifier("classifier.pkl")
    prob = score_regime(model, regime_dict)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─── Feature Engineering ─────────────────────────────────────────────────────

# Categorical mappings
ARCHETYPE_MAP = {
    "breakout": 0, "candle_signal": 1, "open_drive": 2, "opening_trend": 3,
    "range_context": 4, "strike_flow": 5, "time_filter": 6, "unconditional": 7,
    "vol_skew": 8, "volume": 9, "vwap_fade": 10, "vwap_trend": 11,
}
DIRECTION_MAP = {"long": 0, "short": 1}
MONEYNESS_MAP = {"atm": 0, "itm1": 1, "itm2": 2, "itm3": 3}
KIND_MAP = {"naked": 0, "debit_spread": 1}

FEATURE_NAMES = [
    # Train fitness (13)
    "train_sharpe", "train_pf", "train_wr", "train_n_trades", "train_cum_pnl",
    "train_max_dd", "train_max_dd_r", "train_avg_pnl", "train_avg_win",
    "train_avg_loss", "train_fitness", "train_cum_r", "train_avg_r",
    # Val fitness (7) — safe now that forward_profitable comes from forward period
    "val_sharpe", "val_pf", "val_wr", "val_n_trades", "val_cum_pnl",
    "val_avg_pnl", "val_fitness",
    # Val/train ratios (3) — overfitting signals
    "val_train_sharpe_ratio", "val_train_pnl_ratio", "val_train_wr_ratio",
    # Gene structure (9)
    "archetype_encoded", "direction_encoded", "moneyness_encoded",
    "kind_encoded", "n_entry_conditions", "hold_minutes",
    "sl_pct", "tp_pct", "tsl_pct",
    # Time window (2)
    "tw_start_minutes", "tw_duration_minutes",
    # Train quality signals (4)
    "train_pnl_per_trade", "train_win_loss_ratio",
    "train_expectancy", "train_kelly",
    # Risk ratios (3)
    "train_pnl_dd_ratio", "train_sharpe_per_trade", "tp_sl_ratio",
    # Robustness metrics (8) — from compute_robustness_metrics on train trades
    "mc_p_value", "ttest_p_value", "pf_stability_ratio",
    "remove_best_still_positive", "wr_stability_ratio",
    "top_trade_pct", "payoff_consistency", "kelly_fraction",
    # SPY context (1)
    "spy_bull_pct",
]


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or b is None:
        return default
    return a / b


def _parse_time_minutes(t: str) -> int:
    """Parse 'HH:MM' to minutes since midnight."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def extract_features(regime: dict) -> np.ndarray:
    """Extract feature vector from a regime dict. Returns 1D numpy array."""
    tf = regime.get("train_fitness", {})
    vf = regime.get("val_fitness", {})
    genes = regime.get("genes", {})
    exit_rules = genes.get("exit_rules", {})
    trade_type = genes.get("trade_type", {})
    tw = genes.get("time_window", {"start": "09:45", "end": "14:00"})
    rob = regime.get("robustness", {})

    # Train fitness
    train_sharpe = float(tf.get("sharpe", 0))
    train_pf = float(tf.get("profit_factor", 0))
    train_wr = float(tf.get("win_rate", 0))
    train_n = float(tf.get("n_trades", 0))
    train_pnl = float(tf.get("cum_pnl", 0))
    train_dd = float(tf.get("max_drawdown", 0))
    train_dd_r = float(tf.get("max_dd_r", 0))
    train_avg_pnl = float(tf.get("avg_pnl", 0))
    train_avg_win = float(tf.get("avg_win", 0))
    train_avg_loss = float(tf.get("avg_loss", 0))
    train_fitness = float(tf.get("fitness", 0))
    train_cum_r = float(tf.get("cum_r", 0))
    train_avg_r = float(tf.get("avg_r", 0))

    # Val fitness — safe to use with 3-way split (forward_profitable != val_pnl > 0)
    val_sharpe = float(vf.get("sharpe", 0))
    val_pf = float(vf.get("profit_factor", 0))
    val_wr = float(vf.get("win_rate", 0))
    val_n = float(vf.get("n_trades", 0))
    val_cum_pnl = float(regime.get("val_cum_pnl", 0))
    val_avg_pnl = float(vf.get("avg_pnl", 0))
    val_fitness = float(vf.get("fitness", 0))

    # Val/train ratios — overfitting detection
    val_train_sharpe = _safe_div(val_sharpe, train_sharpe)
    val_train_pnl = _safe_div(val_cum_pnl, train_pnl)
    val_train_wr = _safe_div(val_wr, train_wr)

    # Gene structure
    archetype = ARCHETYPE_MAP.get(regime.get("archetype", ""), -1)
    direction = DIRECTION_MAP.get(regime.get("direction", ""), -1)
    moneyness = MONEYNESS_MAP.get(trade_type.get("moneyness", ""), -1)
    kind = KIND_MAP.get(trade_type.get("kind", ""), -1)
    n_conditions = len(genes.get("entry_conditions", []))
    if genes.get("filter_condition"):
        n_conditions += 1
    hold_minutes = float(exit_rules.get("hold_minutes") or 0)
    sl_pct = float(exit_rules.get("sl_pct") or 0)
    tp_pct = float(exit_rules.get("tp_pct") or 0)
    tsl_pct = float(exit_rules.get("tsl_pct") or 0)

    # Time window
    tw_start = _parse_time_minutes(tw.get("start", "09:45"))
    tw_end = _parse_time_minutes(tw.get("end", "14:00"))
    tw_duration = tw_end - tw_start

    # Train quality signals
    train_pnl_per_trade = _safe_div(train_pnl, train_n)
    train_wl_ratio = _safe_div(train_avg_win, train_avg_loss)
    train_expectancy = train_wr * train_avg_win - (1 - train_wr) * train_avg_loss
    train_kelly = _safe_div(train_wr - (1 - train_wr) / max(train_wl_ratio, 0.001), 1.0)

    # Risk ratios
    train_pnl_dd_ratio = _safe_div(train_pnl, train_dd)
    train_sharpe_per_trade = _safe_div(train_sharpe, train_n)
    tp_sl_ratio = _safe_div(tp_pct, sl_pct)

    # Robustness metrics (from train trades — computed by wf_runner)
    mc_p = float(rob.get("mc_p_value", 0.5))
    ttest_p = float(rob.get("ttest_p_value", 1.0))
    pf_stab = float(rob.get("pf_stability_ratio", 0))
    remove_best = float(rob.get("remove_best_still_positive", 0))
    wr_stab = float(rob.get("wr_stability_ratio", 0))
    top_trade = float(rob.get("top_trade_pct", 100))
    payoff_con = float(rob.get("payoff_consistency", 0))
    kelly_frac = float(rob.get("kelly_fraction", 0))

    feats = [
        # Train (13)
        train_sharpe, train_pf, train_wr, train_n, train_pnl,
        train_dd, train_dd_r, train_avg_pnl, train_avg_win,
        train_avg_loss, train_fitness, train_cum_r, train_avg_r,
        # Val (7)
        val_sharpe, val_pf, val_wr, val_n, val_cum_pnl,
        val_avg_pnl, val_fitness,
        # Val/train ratios (3)
        val_train_sharpe, val_train_pnl, val_train_wr,
        # Genes (9)
        archetype, direction, moneyness, kind, n_conditions,
        hold_minutes, sl_pct, tp_pct, tsl_pct,
        # Time (2)
        tw_start, tw_duration,
        # Train quality (4)
        train_pnl_per_trade, train_wl_ratio, train_expectancy, train_kelly,
        # Risk ratios (3)
        train_pnl_dd_ratio, train_sharpe_per_trade, tp_sl_ratio,
        # Robustness (8)
        mc_p, ttest_p, pf_stab, remove_best, wr_stab,
        top_trade, payoff_con, kelly_frac,
        # SPY context (1)
        float(regime.get("spy_bull_pct", 0.5)),
    ]

    return np.array(feats, dtype=np.float32)


# ─── Training ────────────────────────────────────────────────────────────────

def build_dataset(regimes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build X, y from regime list.

    Only uses records where forward_source == 'forward' (genuine 3-way split).
    Falls back to all records if no 3-way records exist (legacy data).
    """
    # Prefer kill-gated forward labels (or fixed-window forward labels)
    genuine = [r for r in regimes
               if r.get("forward_source") in ("kill_gated", "forward")]
    if genuine:
        print(f"[classifier] Using {len(genuine)}/{len(regimes)} records with "
              f"genuine forward labels")
        regimes = genuine
    else:
        print(f"[classifier] WARNING: No genuine forward records found. "
              f"Using {len(regimes)} legacy records (forward_profitable = val_pnl > 0)")

    X = np.array([extract_features(r) for r in regimes])
    y = np.array([1 if r.get("forward_profitable") else 0 for r in regimes])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def train_classifier(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Any:
    """Train LightGBM classifier with class imbalance handling.

    Uses sample weights for minority class upweighting, plus
    stratified 5-fold CV for evaluation.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        classification_report, roc_auc_score, precision_recall_curve, f1_score
    )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    print(f"\n{'='*60}")
    print(f"Training classifier: {len(y)} samples, {n_pos} positive, {n_neg} negative")
    print(f"Class ratio: {n_pos/len(y):.1%} positive / {n_neg/len(y):.1%} negative")
    print(f"Features: {len(FEATURE_NAMES)}")
    print(f"{'='*60}\n")

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_aucs = []
    cv_f1s = []
    cv_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Sample weights: upweight minority class
        weights = np.ones(len(y_tr))
        neg_mask = y_tr == 0
        if neg_mask.sum() > 0:
            weights[neg_mask] = n_pos / max(n_neg, 1)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=seed + fold,
            verbose=-1,
            is_unbalance=False,
        )
        model.fit(X_tr, y_tr, sample_weight=weights)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc = 0.5  # single class in fold
        f1 = f1_score(y_te, y_pred, zero_division=0)
        cv_aucs.append(auc)
        cv_f1s.append(f1)
        report = classification_report(y_te, y_pred, zero_division=0)
        cv_reports.append(report)

        print(f"Fold {fold+1}: AUC={auc:.3f}  F1={f1:.3f}  "
              f"neg_in_test={int((y_te==0).sum())}/{len(y_te)}")

    print(f"\nCV Results: AUC={np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}  "
          f"F1={np.mean(cv_f1s):.3f} +/- {np.std(cv_f1s):.3f}")
    print(f"\nLast fold report:\n{cv_reports[-1]}")

    # Train final model on all data
    print("Training final model on all data...")
    weights_all = np.ones(len(y))
    neg_mask_all = y == 0
    if neg_mask_all.sum() > 0:
        weights_all[neg_mask_all] = n_pos / max(n_neg, 1)

    final_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    final_model.fit(X, y, sample_weight=weights_all)

    # Feature importance
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importances),
                      key=lambda x: x[1], reverse=True)
    print("\nTop 15 features:")
    for name, imp in feat_imp[:15]:
        print(f"  {name}: {imp}")

    return final_model


# ─── Scoring ─────────────────────────────────────────────────────────────────

def load_classifier(path: str | Path) -> Any:
    """Load a trained classifier from disk."""
    import joblib
    return joblib.load(path)


def score_regime(model: Any, regime: dict) -> float:
    """Score a regime dict. Returns P(forward_profitable)."""
    feats = extract_features(regime)
    feats = np.nan_to_num(feats.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
    return float(model.predict_proba(feats)[0, 1])


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options WF Classifier")
    parser.add_argument("--train", action="store_true", help="Train classifier")
    parser.add_argument("--data", type=str, default="regime_db_v2.jsonl")
    parser.add_argument("--out", type=str, default="options_classifier.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", type=str, help="Score a single regime JSON file")
    args = parser.parse_args()

    if args.train:
        regimes = []
        data_path = Path(args.data)
        if not data_path.exists():
            # Try relative to results
            data_path = Path(__file__).parent.parent.parent / "results" / "options_wf" / args.data
        with open(data_path) as f:
            for line in f:
                regimes.append(json.loads(line))

        X, y = build_dataset(regimes)
        model = train_classifier(X, y, seed=args.seed)

        import joblib
        out_path = Path(args.out)
        joblib.dump(model, out_path)
        print(f"\nModel saved to {out_path}")
        print(f"Features: {len(FEATURE_NAMES)}")
        print(f"Samples: {len(y)} ({int(y.sum())} positive, {int(len(y)-y.sum())} negative)")

    elif args.score:
        with open(args.score) as f:
            regime = json.load(f)
        model = load_classifier(args.out)
        prob = score_regime(model, regime)
        print(f"P(forward_profitable) = {prob:.4f}")


if __name__ == "__main__":
    main()
