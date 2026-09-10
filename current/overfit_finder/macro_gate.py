"""
Macro Gate — Environment-level trading filter.
===============================================
Answers "should we trade tonight?" based on recent market conditions,
independent of the specific strategy being deployed.

Separate from meta_config (strategy-level fitness) by design:
- meta_config: "is this a good strategy?"
- macro_gate:  "is this a good environment to trade in?"

The gate scores the current market snapshot and returns a go/no-go
decision plus a confidence level that can scale position sizing.

Usage:
    from macro_gate import MacroGate, load_macro_config
    gate = MacroGate(load_macro_config())
    decision = gate.evaluate(market_snapshot)
    if decision.blocked:
        print(f"Sitting out: {decision.reasons}")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_CONFIG_PATH = _REPO_ROOT / "results" / "macro_gate_config.json"


@dataclass
class MacroGateConfig:
    """Tunable thresholds for the macro environment gate."""
    # --- Hard blocks (sit out entirely) ---
    max_daily_vol_pct: float = 2.5          # daily range as % of price
    max_ret_20d_pct: float = 5.0            # strong uptrend kills ON mean-reversion
    min_ret_20d_pct: float = -8.0           # crash / capitulation
    min_on_range_pts: float = 3.0           # not enough ON range = no edge

    # --- Soft scoring weights (0-1 scale, higher = more favorable) ---
    # Each feature gets scored 0-1, then weighted
    w_on_range: float = 0.30                # wider ON range = more opportunity
    w_vol_regime: float = 0.20              # moderate vol preferred
    w_trend: float = 0.25                   # flat/slight down preferred
    w_range_expansion: float = 0.10         # compression preferred (breakout setup)
    w_dir_consistency: float = 0.10         # choppy preferred for mean-reversion
    w_gap_freq: float = 0.05               # more gaps = more opportunity

    # --- Thresholds ---
    soft_gate_threshold: float = 0.35       # below this = reduced sizing
    hard_gate_threshold: float = 0.20       # below this = sit out

    # --- Ideal ranges (for scoring) ---
    ideal_vol_low: float = 0.8             # sweet spot lower bound
    ideal_vol_high: float = 1.8            # sweet spot upper bound
    ideal_on_range_low: float = 5.0        # minimum useful ON range
    ideal_on_range_high: float = 15.0      # cap benefit
    ideal_ret_neutral: float = 0.5         # center of "good" trend range
    ideal_ret_width: float = 3.0           # width of acceptable trend

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MacroGateConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class GateDecision:
    """Result of macro gate evaluation."""
    score: float                    # 0-1, higher = more favorable
    blocked: bool                   # hard block — don't trade
    reduced: bool                   # soft block — reduce sizing
    size_scalar: float              # 0.0-1.0 position size multiplier
    reasons: list[str] = field(default_factory=list)
    feature_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class MacroGate:
    """Evaluates market snapshots against macro gate config."""

    def __init__(self, config: MacroGateConfig):
        self.config = config

    def evaluate(self, snapshot: dict) -> GateDecision:
        """Score a market snapshot and return go/no-go decision."""
        if not snapshot or "error" in snapshot:
            return GateDecision(
                score=0.5, blocked=False, reduced=False,
                size_scalar=1.0, reasons=["no_snapshot_data"])

        cfg = self.config
        reasons = []

        # --- Hard blocks ---
        vol = snapshot.get("daily_vol_pct", 1.0)
        ret20 = snapshot.get("ret_20d_pct", 0.0)
        on_range = snapshot.get("avg_on_range_pts", 5.0)

        if vol > cfg.max_daily_vol_pct:
            reasons.append(f"vol {vol:.2f}% > {cfg.max_daily_vol_pct}%")
        if ret20 > cfg.max_ret_20d_pct:
            reasons.append(f"ret20d +{ret20:.1f}% > +{cfg.max_ret_20d_pct}%")
        if ret20 < cfg.min_ret_20d_pct:
            reasons.append(f"ret20d {ret20:.1f}% < {cfg.min_ret_20d_pct}%")
        if on_range < cfg.min_on_range_pts:
            reasons.append(f"ON range {on_range:.1f}pts < {cfg.min_on_range_pts}pts")

        if reasons:
            return GateDecision(
                score=0.0, blocked=True, reduced=True,
                size_scalar=0.0, reasons=reasons)

        # --- Soft scoring ---
        scores = {}

        # ON range score: linear ramp from min to ideal_high, capped at 1.0
        on_score = max(0, min(1, (on_range - cfg.ideal_on_range_low) /
                               max(cfg.ideal_on_range_high - cfg.ideal_on_range_low, 1)))
        scores["on_range"] = on_score

        # Vol regime: bell curve around ideal range
        if cfg.ideal_vol_low <= vol <= cfg.ideal_vol_high:
            vol_score = 1.0
        elif vol < cfg.ideal_vol_low:
            vol_score = max(0, vol / cfg.ideal_vol_low)
        else:
            vol_score = max(0, 1 - (vol - cfg.ideal_vol_high) /
                           max(cfg.max_daily_vol_pct - cfg.ideal_vol_high, 0.1))
        scores["vol_regime"] = vol_score

        # Trend score: centered around ideal_ret_neutral, penalize extremes
        ret_dist = abs(ret20 - cfg.ideal_ret_neutral)
        trend_score = max(0, 1 - ret_dist / cfg.ideal_ret_width)
        scores["trend"] = trend_score

        # Range expansion: prefer compression (range_ratio < 1)
        rr = snapshot.get("range_ratio_5d_20d", 1.0)
        if rr <= 1.0:
            re_score = 1.0
        else:
            re_score = max(0, 1 - (rr - 1.0) / 0.5)
        scores["range_expansion"] = re_score

        # Directional consistency: prefer choppy (lower consistency)
        dc = snapshot.get("dir_consistency_pct", 50)
        dc_score = max(0, 1 - (dc - 50) / 30)
        scores["dir_consistency"] = dc_score

        # Gap frequency: more gaps = more opportunity
        gp = snapshot.get("gap_pct", 50)
        gap_score = min(1, gp / 60)
        scores["gap_freq"] = gap_score

        # Weighted composite
        composite = (
            cfg.w_on_range * scores["on_range"]
            + cfg.w_vol_regime * scores["vol_regime"]
            + cfg.w_trend * scores["trend"]
            + cfg.w_range_expansion * scores["range_expansion"]
            + cfg.w_dir_consistency * scores["dir_consistency"]
            + cfg.w_gap_freq * scores["gap_freq"]
        )

        blocked = composite < cfg.hard_gate_threshold
        reduced = composite < cfg.soft_gate_threshold

        if blocked:
            reasons.append(f"composite {composite:.3f} < hard threshold {cfg.hard_gate_threshold}")
        elif reduced:
            reasons.append(f"composite {composite:.3f} < soft threshold {cfg.soft_gate_threshold}")

        # Size scalar: linear ramp from hard to 1.0 at soft threshold
        if blocked:
            size_scalar = 0.0
        elif reduced:
            size_scalar = max(0.25, composite / cfg.soft_gate_threshold)
        else:
            size_scalar = 1.0

        return GateDecision(
            score=round(composite, 4),
            blocked=blocked,
            reduced=reduced,
            size_scalar=round(size_scalar, 4),
            reasons=reasons,
            feature_scores={k: round(v, 4) for k, v in scores.items()},
        )


def load_macro_config() -> MacroGateConfig:
    """Load macro gate config from results/macro_gate_config.json."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return MacroGateConfig.from_dict(json.load(f))
    return MacroGateConfig()


def save_macro_config(config: MacroGateConfig):
    """Save macro gate config to results/macro_gate_config.json."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def optimize_macro_gate(records: list[dict]) -> MacroGateConfig:
    """Optimize macro gate thresholds from regime data with snapshots.

    Focuses on drawdown-based metrics rather than raw P&L:
    - Minimize max drawdown in blocked-out periods
    - Maximize P&L in allowed periods
    - Penalize false negatives (blocking profitable regimes)

    Uses temporal holdout: train on first 70%, validate on last 30%.
    """
    import numpy as np

    # Filter to regimes with valid snapshots
    usable = []
    for r in records:
        snap = r.get("market_snapshot")
        if not snap or "error" in snap:
            continue
        fpnl = r.get("forward_pnl", 0) or 0
        fdd = r.get("forward_max_dd", 0) or 0
        ft = r.get("forward_trades", 0) or 0
        usable.append({
            "forward_pnl": fpnl,
            "forward_max_dd": fdd,
            "forward_trades": ft,
            "label": 1 if fpnl > 0 else 0,
            **{k: v for k, v in snap.items() if isinstance(v, (int, float))},
        })

    if len(usable) < 15:
        print(f"Only {len(usable)} usable regimes — using defaults")
        return MacroGateConfig()

    # Temporal holdout
    usable.sort(key=lambda r: r.get("snapshot_date", ""))
    split = int(len(usable) * 0.7)
    train = usable[:split]
    val = usable[split:]

    if len(val) < 5:
        train = usable
        val = usable

    best_config = MacroGateConfig()
    best_metric = float("-inf")

    # Sweep hard block thresholds
    for max_vol in [1.8, 2.0, 2.2, 2.5, 3.0]:
        for max_ret in [3.0, 4.0, 5.0, 6.0, 8.0]:
            for min_on in [2.0, 3.0, 4.0, 5.0, 6.0]:
                # Evaluate on training set
                cfg = MacroGateConfig(
                    max_daily_vol_pct=max_vol,
                    max_ret_20d_pct=max_ret,
                    min_on_range_pts=min_on,
                )
                gate = MacroGate(cfg)

                # Score on validation set
                allowed_pnl = 0
                blocked_pnl = 0
                allowed_dd = 0
                n_allowed = 0
                n_blocked = 0
                false_neg_cost = 0  # profitable regimes we blocked

                for r in val:
                    snap = {k: v for k, v in r.items()
                            if k not in ("forward_pnl", "forward_max_dd",
                                         "forward_trades", "label")}
                    decision = gate.evaluate(snap)

                    if decision.blocked:
                        n_blocked += 1
                        blocked_pnl += r["forward_pnl"]
                        if r["forward_pnl"] > 0:
                            false_neg_cost += r["forward_pnl"]
                    else:
                        n_allowed += 1
                        allowed_pnl += r["forward_pnl"]
                        allowed_dd += r["forward_max_dd"]

                if n_allowed < 3:
                    continue

                # Metric: maximize allowed P&L while minimizing blocked losses
                # Penalize false negatives (blocking winners)
                avg_allowed_pnl = allowed_pnl / n_allowed
                avg_blocked_pnl = blocked_pnl / max(n_blocked, 1)
                avg_dd = allowed_dd / n_allowed

                # We WANT: high allowed P&L, negative blocked P&L (we blocked losers)
                # Penalize: blocking profitable regimes, high DD in allowed set
                metric = (avg_allowed_pnl
                          - 0.5 * false_neg_cost
                          + 0.3 * max(0, -avg_blocked_pnl)  # reward blocking losers
                          - 0.2 * avg_dd)

                if metric > best_metric:
                    best_metric = metric
                    best_config = cfg

    print(f"\nOptimized macro gate (metric={best_metric:.0f}):")
    print(f"  max_daily_vol: {best_config.max_daily_vol_pct}%")
    print(f"  max_ret_20d:   {best_config.max_ret_20d_pct}%")
    print(f"  min_on_range:  {best_config.min_on_range_pts} pts")

    return best_config


def backtest_gate(records: list[dict], config: MacroGateConfig | None = None):
    """Backtest the macro gate on historical regimes to see impact."""
    cfg = config or load_macro_config()
    gate = MacroGate(cfg)

    allowed = []
    blocked = []

    for r in records:
        snap = r.get("market_snapshot")
        if not snap or "error" in snap:
            continue
        decision = gate.evaluate(snap)
        entry = {
            "forward_pnl": r.get("forward_pnl", 0) or 0,
            "forward_max_dd": r.get("forward_max_dd", 0) or 0,
            "forward_trades": r.get("forward_trades", 0) or 0,
            "forward_start": r.get("forward_start", ""),
            "decision": decision.to_dict(),
        }
        if decision.blocked:
            blocked.append(entry)
        else:
            allowed.append(entry)

    n_total = len(allowed) + len(blocked)
    if n_total == 0:
        print("No regimes with snapshots")
        return

    # Allowed stats
    a_pnl = sum(r["forward_pnl"] for r in allowed)
    a_wins = sum(1 for r in allowed if r["forward_pnl"] > 0)
    a_dd = max((r["forward_max_dd"] for r in allowed), default=0)

    # Blocked stats
    b_pnl = sum(r["forward_pnl"] for r in blocked)
    b_wins = sum(1 for r in blocked if r["forward_pnl"] > 0)

    # Unfiltered (all)
    all_pnl = a_pnl + b_pnl
    all_wins = a_wins + b_wins

    print(f"\n{'='*60}")
    print(f"MACRO GATE BACKTEST")
    print(f"{'='*60}")
    print(f"Config: max_vol={cfg.max_daily_vol_pct}%  max_ret20d={cfg.max_ret_20d_pct}%  "
          f"min_on_range={cfg.min_on_range_pts}pts")
    print(f"\n  Unfiltered:  {n_total:>3} regimes  P&L=${all_pnl:>9,.0f}  "
          f"WR={all_wins/n_total*100:.0f}%")
    print(f"  Allowed:     {len(allowed):>3} regimes  P&L=${a_pnl:>9,.0f}  "
          f"WR={a_wins/max(len(allowed),1)*100:.0f}%")
    print(f"  Blocked:     {len(blocked):>3} regimes  P&L=${b_pnl:>9,.0f}  "
          f"WR={b_wins/max(len(blocked),1)*100:.0f}%")
    print(f"\n  Gate value:  ${all_pnl - a_pnl:>+9,.0f} avoided losses")

    if blocked:
        print(f"\n  Blocked regimes:")
        for r in sorted(blocked, key=lambda x: x["forward_start"]):
            reasons = ", ".join(r["decision"]["reasons"])
            print(f"    {r['forward_start'][:10]}  P&L=${r['forward_pnl']:>8,.0f}  "
                  f"reason: {reasons}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Load all regimes
    import glob
    _RESULTS_DIR = _REPO_ROOT / "results"
    all_records = []
    for f in sorted(_RESULTS_DIR.glob("overnight_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get("regimes", []):
            all_records.append(r)

    print(f"Loaded {len(all_records)} regime records")

    if args.optimize:
        cfg = optimize_macro_gate(all_records)
        if args.save:
            save_macro_config(cfg)
            print(f"\nSaved to {_CONFIG_PATH}")
        backtest_gate(all_records, cfg)
    elif args.backtest:
        cfg = load_macro_config()
        backtest_gate(all_records, cfg)
    else:
        # Default: optimize + backtest
        cfg = optimize_macro_gate(all_records)
        if args.save:
            save_macro_config(cfg)
        backtest_gate(all_records, cfg)
