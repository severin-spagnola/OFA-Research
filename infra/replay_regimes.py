"""
Replay Regimes — Rewrite historical regime records under new kill rules.
=========================================================================
Takes all saved overnight results and produces a cleaned regime database
suitable for classifier training.

Since we don't have individual trade PnLs in the saved results, we can't
perfectly replay equity curves. Instead we:
1. Filter out regimes with train_n_trades < min_trade_count
2. For regimes killed by non-DD reasons that were profitable at death,
   keep their P&L (conservative — they were making money when killed)
3. For regimes killed by DD, cap P&L at -max_dd_dollars
4. Re-label death reasons to reflect what WOULD have happened under
   DD-only + circuit breaker rules

Usage:
    python infra/replay_regimes.py
    python infra/replay_regimes.py --min-trades 18 --max-dd 2000
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _REPO_ROOT / "results"


def load_all_regimes() -> list[dict]:
    """Load all regime records from overnight result files."""
    all_regimes = []
    for path in sorted(_RESULTS_DIR.glob("overnight_*.json")):
        with open(path) as f:
            data = json.load(f)
        for r in data.get("regimes", []):
            r["_source_file"] = path.name
            all_regimes.append(r)
    return all_regimes


def replay_regime(regime: dict, max_dd: float = 2000.0,
                  circuit_breaker_wr: float = 0.25,
                  circuit_breaker_min_trades: int = 8) -> dict:
    """Replay a single regime under DD-only + circuit breaker rules.

    Returns a new dict with adjusted fields.
    """
    r = dict(regime)  # shallow copy
    death = r.get("death_reason", "")
    fwd_pnl = r.get("forward_pnl", 0) or 0
    fwd_dd = r.get("forward_max_dd", 0) or 0
    fwd_trades = r.get("forward_trades", 0) or 0
    fwd_wr = r.get("forward_win_rate", 0) or 0

    original_death = death
    r["original_death_reason"] = original_death

    # Classify what would have happened under new rules
    if "drawdown" in death:
        # DD kill — would still happen, but cap at our max_dd
        if fwd_dd > max_dd:
            r["replayed_death"] = f"drawdown ${fwd_dd:,.0f} exceeds ${max_dd:,.0f}"
            # P&L is already recorded at the point of death
            # If the recorded DD > our new limit, the strategy would have
            # been killed earlier (less loss). Approximate by capping.
            if fwd_pnl < -max_dd:
                r["forward_pnl"] = -max_dd
                r["replayed_pnl_adjusted"] = True
            else:
                r["replayed_pnl_adjusted"] = False
        else:
            r["replayed_death"] = death
            r["replayed_pnl_adjusted"] = False

    elif "wr_circuit" in death:
        # Circuit breaker — check if it would trigger under new threshold
        wr_at_death = fwd_wr / 100.0 if fwd_wr > 1 else fwd_wr
        if fwd_trades >= circuit_breaker_min_trades and wr_at_death < circuit_breaker_wr:
            r["replayed_death"] = f"circuit_breaker: {fwd_wr:.0f}% WR < {circuit_breaker_wr*100:.0f}%"
        else:
            # Would NOT have been killed — strategy continues
            r["replayed_death"] = "would_continue"
        r["replayed_pnl_adjusted"] = False

    elif any(k in death for k in ["early kill", "bayesian", "negative trajectory",
                                   "flat regime", "win rate", "late death"]):
        # These kills are DISABLED — strategy would have continued
        r["replayed_death"] = "would_continue"
        r["replayed_pnl_adjusted"] = False

    elif "data_end" in death:
        r["replayed_death"] = "data_end"
        r["replayed_pnl_adjusted"] = False

    else:
        r["replayed_death"] = death
        r["replayed_pnl_adjusted"] = False

    return r


def build_classifier_dataset(
    min_trade_count: int = 18,
    max_dd: float = 2000.0,
    circuit_breaker_wr: float = 0.25,
    circuit_breaker_min_trades: int = 8,
) -> list[dict]:
    """Build a cleaned dataset for classifier training.

    Returns list of feature dicts, one per regime.
    """
    all_regimes = load_all_regimes()
    print(f"Loaded {len(all_regimes)} total regimes")

    # Filter by min_trade_count
    filtered = [r for r in all_regimes if (r.get("train_n_trades", 0) or 0) >= min_trade_count]
    excluded = len(all_regimes) - len(filtered)
    print(f"After min_trade_count >= {min_trade_count}: {len(filtered)} regimes ({excluded} excluded)")

    # Replay under new kill rules
    replayed = [replay_regime(r, max_dd, circuit_breaker_wr, circuit_breaker_min_trades)
                for r in filtered]

    # Build feature records for classifier
    records = []
    for r in replayed:
        train_rob = r.get("train_robustness") or {}
        fwd_rob = r.get("forward_robustness") or {}
        params = r.get("winner_params") or {}
        entry = params.get("entry", {})
        exit_def = params.get("exit", {})

        # Label: profitable forward?
        fwd_pnl = r.get("forward_pnl", 0) or 0
        label = 1 if fwd_pnl > 0 else 0

        # Strategy complexity
        n_conditions = len(entry.get("conditions", []))
        has_confirmation = 1 if entry.get("confirmation") else 0
        has_filter = 1 if entry.get("filter") else 0
        has_filter2 = 1 if entry.get("filter2") else 0
        complexity = n_conditions + has_confirmation + has_filter + has_filter2

        # Exit type
        has_tsl = 1 if (exit_def.get("be_trigger_pts") or exit_def.get("trail_distance_pts")) else 0
        sl = exit_def.get("stop_loss_pts", 0) or 0
        tp = exit_def.get("take_profit_pts", 0) or 0

        # Direction
        direction = 1 if entry.get("direction") == "long" else 0

        record = {
            # ── Label ──
            "label": label,
            "forward_pnl": fwd_pnl,
            "replayed_death": r.get("replayed_death", ""),
            "original_death": r.get("original_death_reason", ""),

            # ── Training features (available at deploy time) ──
            "train_fitness": r.get("winner_fitness", 0),
            "train_sharpe": r.get("train_sharpe", 0),
            "train_pf": r.get("train_pf", 0),
            "train_n_trades": r.get("train_n_trades", 0),
            "archetype": params.get("archetype", "unknown"),

            # Train robustness
            "train_mc_p": train_rob.get("mc_p_value", 0.5),
            "train_pf_stability": train_rob.get("pf_stability_ratio", 0),
            "train_wr_stability": train_rob.get("wr_stability_ratio", 0),
            "train_top_trade_pct": train_rob.get("top_trade_pct", 50),
            "train_remove_best_positive": 1 if train_rob.get("remove_best_still_positive") else 0,
            "train_payoff_consistency": train_rob.get("payoff_consistency", 1.0),
            "train_expectancy": train_rob.get("expectancy_per_trade", 0),
            "train_kelly": train_rob.get("kelly_fraction", 0),
            "train_win_loss_ratio": train_rob.get("win_loss_ratio", 1.0),
            "train_avg_win": train_rob.get("avg_win_dollars", 0),
            "train_avg_loss": train_rob.get("avg_loss_dollars", 0),
            "train_max_consec_losses": train_rob.get("max_consecutive_losses", 0),
            "train_trades_per_day": train_rob.get("trades_per_day", 0),

            # Strategy structure
            "n_conditions": n_conditions,
            "complexity": complexity,
            "has_tsl": has_tsl,
            "sl_pts": sl,
            "tp_pts": tp,
            "direction": direction,

            # Context
            "optimize_start": r.get("optimize_start", ""),
            "forward_start": r.get("forward_start", ""),
            "forward_trades": r.get("forward_trades", 0),
            "forward_days": r.get("forward_days", 0),
            "forward_max_dd": r.get("forward_max_dd", 0),
        }
        records.append(record)

    return records


def print_summary(records: list[dict]):
    """Print summary stats of the classifier dataset."""
    n = len(records)
    if n == 0:
        print("No records")
        return

    winners = [r for r in records if r["label"] == 1]
    losers = [r for r in records if r["label"] == 0]

    print(f"\n{'='*60}")
    print(f"CLASSIFIER DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total regimes: {n}")
    print(f"Winners: {len(winners)} ({len(winners)/n*100:.0f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/n*100:.0f}%)")

    total_pnl = sum(r["forward_pnl"] for r in records)
    print(f"Total P&L: ${total_pnl:,.0f}")
    print(f"Avg P&L: ${total_pnl/n:,.0f}")

    # Replayed death distribution
    from collections import Counter
    deaths = Counter(r["replayed_death"] for r in records)
    print(f"\nReplayed death reasons:")
    for reason, count in deaths.most_common():
        pnls = [r["forward_pnl"] for r in records if r["replayed_death"] == reason]
        print(f"  {reason:40s} n={count:>2} avg=${sum(pnls)/len(pnls):>7,.0f}")

    # Feature means by label
    print(f"\nFeature means (Winner vs Loser):")
    features = ["train_n_trades", "train_expectancy", "train_fitness",
                 "train_sharpe", "train_pf", "complexity", "train_kelly",
                 "train_top_trade_pct", "train_pf_stability"]
    for feat in features:
        w_vals = [r[feat] for r in winners if r[feat] is not None]
        l_vals = [r[feat] for r in losers if r[feat] is not None]
        w_mean = sum(w_vals)/len(w_vals) if w_vals else 0
        l_mean = sum(l_vals)/len(l_vals) if l_vals else 0
        sep = ">>>" if abs(w_mean - l_mean) > 0.3 * max(abs(w_mean), abs(l_mean), 0.01) else "   "
        print(f"  {feat:25s} W={w_mean:>8.2f}  L={l_mean:>8.2f} {sep}")

    # Archetype breakdown
    print(f"\nArchetype breakdown:")
    by_arch = defaultdict(list)
    for r in records:
        by_arch[r["archetype"]].append(r)
    for arch, regs in sorted(by_arch.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for r in regs if r["label"] == 1)
        pnl = sum(r["forward_pnl"] for r in regs)
        print(f"  {arch:25s} n={len(regs):>2} wins={wins} wr={wins/len(regs)*100:>4.0f}% pnl=${pnl:>8,.0f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay regimes under new kill rules")
    parser.add_argument("--min-trades", type=int, default=18)
    parser.add_argument("--max-dd", type=float, default=2000.0)
    parser.add_argument("--cb-wr", type=float, default=0.25, help="Circuit breaker WR threshold")
    parser.add_argument("--cb-trades", type=int, default=8, help="Circuit breaker min trades")
    parser.add_argument("--save", action="store_true", help="Save dataset to JSON")
    args = parser.parse_args()

    records = build_classifier_dataset(
        min_trade_count=args.min_trades,
        max_dd=args.max_dd,
        circuit_breaker_wr=args.cb_wr,
        circuit_breaker_min_trades=args.cb_trades,
    )

    print_summary(records)

    if args.save:
        out_path = _RESULTS_DIR / "classifier_dataset.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nSaved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
