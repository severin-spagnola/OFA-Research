"""
Regime Filter — Deterministic Lookup Table for Strategy Quality
================================================================
Rule-based pre-deployment gate that scores strategies using training-side
metrics. A lightweight alternative to the full LightGBM classifier that
works with small sample sizes (< 100 regimes).

OVERNIGHT-CALIBRATED: Traditional robustness metrics (MC p-value, PF
stability, WR stability) are INVERTED or zeroed for overnight because
empirically, "messy" training stats correlate with forward success.
The real separating signals are expectancy/trade and training trade count.

Usage:
    from regime_filter import score_strategy, should_deploy

    score = score_strategy(train_fitness, train_robustness, strategy_def, meta)
    if should_deploy(score):
        # deploy the strategy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterScore:
    """Result of the deterministic filter scoring."""
    total_score: float          # 0-1 composite score
    decision: str               # "deploy", "reject", "borderline"
    reasons: list[str]          # human-readable reasons for the decision
    component_scores: dict      # breakdown of individual scores

    # Hard reject flags
    hard_reject: bool = False
    hard_reject_reason: str = ""


# ── Filter Configuration ────────────────────────────────────────────────────
# Overnight-calibrated from 12 regimes (4 winners, 8 losers).
#
# Key finding: traditional robustness metrics are INVERSELY correlated
# with overnight forward success. Winning strategies look "messy" in
# training because overnight edge is inherently lumpy (few big trades).
#
# Separating signals:
#   - Expectancy/trade: winners avg $340 vs losers $243
#   - Training trade count: winners avg 15 vs losers 11
#   - Traditional stats (MC, PF stability, WR stability): INVERTED

FILTER_CONFIG = {
    # ── Hard reject rules (any one = instant reject) ──
    "hard_reject_mc_p_above": 1.01,         # DISABLED — MC test unreliable overnight
    "hard_reject_top_trade_pct_above": 200,  # only reject extreme single-trade dependency
    "hard_reject_train_trades_below": 6,     # too few trades to trust anything
    "hard_reject_pf_below": 1.0,            # must at least be profitable in training

    # ── Soft scoring weights (sum to 1.0) ──
    # Heavy on expectancy and trade count; traditional metrics zeroed/minimal
    "w_expectancy": 0.40,           # PRIMARY: raw expected value per trade
    "w_train_n_trades": 0.30,       # PRIMARY: more trades = more trustworthy
    "w_train_fitness": 0.15,        # moderate — composite fitness from search
    "w_val_expectancy": 0.15,       # validation expectancy if available

    # Legacy weights — kept at 0 for overnight, can re-enable for RTH
    "w_mc_p_value": 0.0,
    "w_pf_stability": 0.0,
    "w_remove_best_positive": 0.0,
    "w_wr_stability": 0.0,
    "w_payoff_consistency": 0.0,

    # ── Scoring thresholds ──
    "expectancy_excellent": 300.0,   # $/trade (winners avg $340)
    "expectancy_good": 200.0,        # $/trade (losers avg $243, so midpoint ~$270)
    "train_n_excellent": 18,         # trades (winners avg 15, want more)
    "train_n_good": 12,              # trades (losers avg 11)
    "fitness_excellent": 2.0,
    "fitness_good": 1.2,
    "val_expectancy_excellent": 20.0,  # $/trade in validation
    "val_expectancy_good": 5.0,

    # ── Deployment threshold ──
    "deploy_threshold": 0.50,        # score >= this → deploy
    "borderline_low": 0.35,          # between this and deploy = borderline
}


def _score_range(value: float, excellent: float, good: float,
                 lower_is_better: bool = False) -> float:
    """Score a value on 0-1 scale given excellent/good thresholds."""
    if lower_is_better:
        if value <= excellent:
            return 1.0
        elif value <= good:
            return 0.5 + 0.5 * (good - value) / (good - excellent)
        else:
            return max(0.0, 0.5 * (1.0 - (value - good) / good))
    else:
        if value >= excellent:
            return 1.0
        elif value >= good:
            return 0.5 + 0.5 * (value - good) / (excellent - good)
        else:
            return max(0.0, 0.5 * value / good) if good > 0 else 0.0


def score_strategy(
    train_fitness: float,
    train_robustness: Optional[dict],
    train_n_trades: int = 0,
    val_pnl: Optional[float] = None,
    val_expectancy: Optional[float] = None,
    config: Optional[dict] = None,
) -> FilterScore:
    """Score a strategy using deterministic rules on training metrics.

    Overnight-calibrated: prioritizes expectancy and trade count over
    traditional robustness metrics.

    Args:
        train_fitness: Composite fitness score from search
        train_robustness: Dict from RobustnessMetrics.to_dict()
        train_n_trades: Number of trades in training window
        val_pnl: Validation P&L (if available)
        val_expectancy: Validation expectancy per trade (if available)
        config: Override default FILTER_CONFIG

    Returns:
        FilterScore with decision and breakdown
    """
    cfg = config or FILTER_CONFIG
    rob = train_robustness or {}
    reasons = []
    components = {}

    # ── Hard reject checks ──
    mc_p = rob.get("mc_p_value", 0.5)
    top_trade_pct = rob.get("top_trade_pct", 50.0)

    if mc_p > cfg["hard_reject_mc_p_above"]:
        return FilterScore(
            total_score=0.0, decision="reject", reasons=[
                f"MC p-value {mc_p:.3f} > {cfg['hard_reject_mc_p_above']}"
            ],
            component_scores={}, hard_reject=True,
            hard_reject_reason="mc_p_value_too_high",
        )

    if top_trade_pct > cfg["hard_reject_top_trade_pct_above"]:
        return FilterScore(
            total_score=0.0, decision="reject", reasons=[
                f"Top trade = {top_trade_pct:.1f}% of P&L > "
                f"{cfg['hard_reject_top_trade_pct_above']}%"
            ],
            component_scores={}, hard_reject=True,
            hard_reject_reason="top_trade_concentration",
        )

    if train_n_trades > 0 and train_n_trades < cfg["hard_reject_train_trades_below"]:
        return FilterScore(
            total_score=0.0, decision="reject", reasons=[
                f"Only {train_n_trades} training trades < {cfg['hard_reject_train_trades_below']}"
            ],
            component_scores={}, hard_reject=True,
            hard_reject_reason="insufficient_trades",
        )

    # ── Primary scores ──

    # 1. Expectancy per trade (strongest signal)
    expectancy = rob.get("expectancy_per_trade", 0.0)
    exp_score = _score_range(
        expectancy, cfg["expectancy_excellent"], cfg["expectancy_good"]
    )
    components["expectancy"] = round(exp_score, 3)

    # 2. Training trade count
    n_trades = train_n_trades or int(rob.get("trades_per_day", 0) * 30)
    n_score = _score_range(
        float(n_trades), float(cfg["train_n_excellent"]), float(cfg["train_n_good"])
    )
    components["train_n_trades"] = round(n_score, 3)

    # 3. Train fitness
    fit_score = _score_range(
        train_fitness, cfg["fitness_excellent"], cfg["fitness_good"]
    )
    components["train_fitness"] = round(fit_score, 3)

    # 4. Validation expectancy (if available)
    if val_expectancy is not None and val_expectancy > 0:
        val_exp_score = _score_range(
            val_expectancy, cfg["val_expectancy_excellent"],
            cfg["val_expectancy_good"]
        )
    elif val_pnl is not None:
        # Rough proxy: val_pnl > 0 = 0.5, val_pnl > $100 = 1.0
        val_exp_score = min(1.0, max(0.0, val_pnl / 200.0))
    else:
        val_exp_score = 0.5  # neutral if no val data
    components["val_expectancy"] = round(val_exp_score, 3)

    # ── Weighted composite ──
    total = (
        cfg["w_expectancy"] * exp_score
        + cfg["w_train_n_trades"] * n_score
        + cfg["w_train_fitness"] * fit_score
        + cfg["w_val_expectancy"] * val_exp_score
    )
    total = round(total, 4)

    # ── Decision ──
    if total >= cfg["deploy_threshold"]:
        decision = "deploy"
        reasons.append(f"Score {total:.3f} >= {cfg['deploy_threshold']} threshold")
    elif total >= cfg["borderline_low"]:
        decision = "borderline"
        reasons.append(f"Score {total:.3f} in borderline range "
                       f"[{cfg['borderline_low']}, {cfg['deploy_threshold']})")
    else:
        decision = "reject"
        reasons.append(f"Score {total:.3f} < {cfg['borderline_low']} threshold")

    # Add top contributing/detracting factors
    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
    if sorted_components:
        best = sorted_components[0]
        worst = sorted_components[-1]
        reasons.append(f"Best: {best[0]}={best[1]:.2f}")
        reasons.append(f"Worst: {worst[0]}={worst[1]:.2f}")

    return FilterScore(
        total_score=total,
        decision=decision,
        reasons=reasons,
        component_scores=components,
    )


def should_deploy(score: FilterScore) -> bool:
    """Simple boolean: should we deploy this strategy?"""
    return score.decision == "deploy"


def backtest_filter(regime_records: list[dict], config: Optional[dict] = None) -> dict:
    """Backtest the filter against historical regime data.

    Takes a list of regime record dicts (from RegimeRecord.to_dict()),
    scores each one, and computes accuracy metrics.

    Returns dict with:
        - accuracy: % of correct deploy/reject decisions
        - precision: % of deployed strategies that were profitable
        - recall: % of profitable strategies that were deployed
        - regimes: list of {regime, score, actual_profitable, decision}
    """
    cfg = config or FILTER_CONFIG
    results = []

    for r in regime_records:
        train_rob = r.get("train_robustness") or {}
        fitness = r.get("winner_fitness", 0.0)

        fscore = score_strategy(
            train_fitness=fitness,
            train_robustness=train_rob,
            train_n_trades=r.get("train_n_trades", 0),
            val_pnl=r.get("val_pnl"),
            config=cfg,
        )

        actual_profitable = (r.get("forward_pnl", 0) or 0) > 0

        results.append({
            "optimize_start": r.get("optimize_start"),
            "forward_pnl": r.get("forward_pnl"),
            "forward_trades": r.get("forward_trades"),
            "actual_profitable": actual_profitable,
            "filter_score": fscore.total_score,
            "filter_decision": fscore.decision,
            "would_deploy": should_deploy(fscore),
            "correct": (should_deploy(fscore) and actual_profitable) or
                       (not should_deploy(fscore) and not actual_profitable),
            "components": fscore.component_scores,
            "reasons": fscore.reasons,
        })

    # Compute metrics
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "regimes": []}

    correct = sum(1 for r in results if r["correct"])
    deployed = [r for r in results if r["would_deploy"]]
    profitable = [r for r in results if r["actual_profitable"]]

    deployed_profitable = sum(1 for r in deployed if r["actual_profitable"])
    profitable_deployed = sum(1 for r in profitable if r["would_deploy"])

    return {
        "n_regimes": n,
        "accuracy": round(correct / n * 100, 1) if n else 0,
        "precision": round(deployed_profitable / len(deployed) * 100, 1) if deployed else 0,
        "recall": round(profitable_deployed / len(profitable) * 100, 1) if profitable else 0,
        "n_deployed": len(deployed),
        "n_rejected": n - len(deployed),
        "n_profitable": len(profitable),
        "deployed_pnl": sum(r["forward_pnl"] for r in deployed),
        "rejected_pnl": sum(r["forward_pnl"] for r in results if not r["would_deploy"]),
        "regimes": results,
    }


def calibrate_from_data(regime_records: list[dict]) -> dict:
    """Auto-calibrate filter thresholds from regime data.

    Computes optimal thresholds by finding values that maximize
    precision (avoiding bad deploys) while maintaining reasonable recall.

    Requires 20+ regimes for meaningful calibration.
    Returns updated FILTER_CONFIG dict.
    """
    if len(regime_records) < 20:
        print(f"  [regime_filter] Only {len(regime_records)} regimes — "
              f"need 20+ for calibration, using defaults")
        return FILTER_CONFIG.copy()

    winners = [r for r in regime_records if (r.get("forward_pnl", 0) or 0) > 0]
    losers = [r for r in regime_records if (r.get("forward_pnl", 0) or 0) <= 0]

    if not winners or not losers:
        print(f"  [regime_filter] Need both winners and losers for calibration")
        return FILTER_CONFIG.copy()

    def _median(values):
        s = sorted(v for v in values if v is not None)
        if not s:
            return 0
        mid = len(s) // 2
        return s[mid]

    def _rob_field(records, field, default=0):
        return [r.get("train_robustness", {}).get(field, default)
                for r in records if r.get("train_robustness")]

    cfg = FILTER_CONFIG.copy()

    # Calibrate expectancy threshold from winner/loser split
    w_exp = _median(_rob_field(winners, "expectancy_per_trade", 0))
    l_exp = _median(_rob_field(losers, "expectancy_per_trade", 0))
    if w_exp > l_exp:
        cfg["expectancy_good"] = round((w_exp + l_exp) / 2, 2)
        cfg["expectancy_excellent"] = round(w_exp, 2)

    # Calibrate trade count threshold
    w_n = _median([r.get("train_n_trades", 0) for r in winners])
    l_n = _median([r.get("train_n_trades", 0) for r in losers])
    if w_n > l_n:
        cfg["train_n_good"] = round((w_n + l_n) / 2)
        cfg["train_n_excellent"] = round(w_n)

    # Test different deploy thresholds
    best_threshold = cfg["deploy_threshold"]
    best_score = -999
    for thresh in [x / 100 for x in range(20, 70, 5)]:
        test_cfg = cfg.copy()
        test_cfg["deploy_threshold"] = thresh
        test_cfg["borderline_low"] = thresh - 0.10
        bt = backtest_filter(regime_records, test_cfg)
        # Optimize for precision * sqrt(recall) — prioritize not deploying losers
        if bt["precision"] > 0 and bt["recall"] > 0:
            score = bt["precision"] * (bt["recall"] ** 0.5)
            if score > best_score:
                best_score = score
                best_threshold = thresh

    cfg["deploy_threshold"] = best_threshold
    cfg["borderline_low"] = round(best_threshold - 0.10, 2)

    print(f"  [regime_filter] Calibrated from {len(regime_records)} regimes "
          f"({len(winners)} winners, {len(losers)} losers)")
    print(f"  [regime_filter] Deploy threshold: {best_threshold:.2f}")
    print(f"  [regime_filter] Expectancy: good={cfg['expectancy_good']:.0f} "
          f"excellent={cfg['expectancy_excellent']:.0f}")
    print(f"  [regime_filter] Trade count: good={cfg['train_n_good']} "
          f"excellent={cfg['train_n_excellent']}")

    return cfg
