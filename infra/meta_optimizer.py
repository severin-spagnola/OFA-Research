"""
Meta-Optimizer — Online Parameter Update from Regime Data
=========================================================
Ingests walk-forward result JSONs, accumulates regime records, and
re-computes optimal meta-parameters (archetype scores, gate threshold,
death params, fitness weights) using the full accumulated dataset.

Each iteration refines the meta-config. The orchestrator calls
`update_meta_config()` after each batch of walk-forward jobs.

No LLM calls — pure math on regime records.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent


@dataclass
class MetaConfig:
    """Mirror of overfit_search.MetaConfig — kept in sync.

    Duplicated here so the orchestrator/meta_optimizer can run without
    importing numpy/pandas (heavy deps that live on RunPod, not locally).
    """
    archetype_scores: dict = field(default_factory=lambda: {
        "prev_close": 1.10, "smc_sweep": 1.33, "on_breakout": 0.50,
        "day_open": 0.0, "mean_reversion": -0.33,
        "gap_fill_fade": -0.82, "first_bar": -1.29,
        "gex_mean_reversion": 0.0, "gex_momentum": 0.0,
        "vwap_fade": 0.0, "vol_regime": 0.0, "quiet_day_fade": 0.0,
    })
    gate_threshold: float = 0.0
    fitness_bonus_high: float = 3.0
    fitness_bonus_low: float = 0.5
    early_death_wr_floor: float = 35.0
    early_death_min_trades: int = 15
    max_dd_dollars: float = 3500.0
    negative_trajectory_pnl: float = -500.0
    negative_trajectory_trades: int = 20
    flat_regime_pnl: float = 200.0
    flat_regime_trades: int = 25
    sharpe_weight: float = 0.20
    pf_weight: float = 0.20
    trade_count_weight: float = 0.50
    return_dd_weight: float = 0.10
    # Overfit penalty params
    sharpe_penalty_above: float = 5.0
    pf_penalty_above: float = 5.0
    hard_sharpe_ceiling: float = 10.0
    hard_pf_ceiling: float = 8.0
    min_trade_count: int = 8
    # Trajectory scoring
    trajectory_weight: float = 0.15
    # Bonus/penalty params
    tsl_be_bonus: float = 0.15
    frontload_penalty: float = 0.20
    sweet_spot_bonus: float = 0.20
    # Recency bias
    recency_weight: float = 0.40
    # MFE/MAE ratio gate
    mfe_mae_ratio_ceiling: float = 5.0
    mfe_mae_ceiling_penalty: float = 0.20
    # mfe_before_mae entry quality signal
    mfe_before_mae_bonus_high: float = 0.10
    mfe_before_mae_bonus_mid: float = 0.05
    mfe_before_mae_penalty: float = -0.10
    mfe_before_mae_high_threshold: float = 0.70
    mfe_before_mae_mid_threshold: float = 0.50
    mfe_before_mae_low_threshold: float = 0.35
    # SL/TP ranges — sampled from by gene combinator and LLM
    sl_range_min: float = 8.0
    sl_range_max: float = 15.0
    tp_range_min: float = 10.0
    tp_range_max: float = 20.0
    # TSL trail distance ranges
    tsl_min_trail_pts: float = 4.0
    tsl_max_trail_pts: float = 6.0
    tsl_be_min_trigger_pts: float = 10.0
    tsl_be_max_trigger_pts: float = 13.0
    # Bayesian edge scoring
    bayesian_assumed_edge_wr: float = 0.70
    bayesian_prior_edge: float = 0.15
    bayesian_weight: float = 0.20
    bayesian_kill_threshold: float = 0.35
    # Hard WR circuit breaker
    wr_circuit_breaker_threshold: float = 0.40
    wr_circuit_breaker_min_trades: int = 8
    # Direction filter
    allow_long_strategies: bool = True
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MetaConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


DEFAULT_META_CONFIG = MetaConfig()


# ─── Candidate Logs ─────────────────────────────────────────────────────────

_CANDIDATE_LOGS_PATH = _REPO_ROOT / "results" / "candidate_logs.jsonl"


def load_candidate_logs(path: str | Path | None = None) -> list[dict]:
    """Load candidate log records from JSONL file."""
    p = Path(path) if path else _CANDIDATE_LOGS_PATH
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ─── Regime Database ────────────────────────────────────────────────────────

DB_PATH = _REPO_ROOT / "results" / "regime_db.json"  # MES default (backwards compat)


def _db_path(asset: str = "MES") -> Path:
    """Per-asset regime DB path. MES uses legacy path for backwards compat."""
    if asset == "MES":
        return _REPO_ROOT / "results" / "regime_db.json"
    return _REPO_ROOT / "results" / asset.lower() / "regime_db.json"


def _config_path(asset: str = "MES") -> Path:
    """Per-asset meta-config path. MES uses legacy path for backwards compat."""
    if asset == "MES":
        return _REPO_ROOT / "results" / "meta_config.json"
    return _REPO_ROOT / "results" / asset.lower() / "meta_config.json"


def load_regime_db(asset: str = "MES") -> list[dict]:
    """Load accumulated regime records from disk."""
    path = _db_path(asset)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_regime_db(records: list[dict], asset: str = "MES") -> None:
    """Save regime records to disk."""
    path = _db_path(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  [meta] Saved {len(records)} regime records to {path}")


def extract_regimes_from_result(result_json: dict) -> list[dict]:
    """Extract regime records from a walk-forward result JSON.

    Each regime record contains:
    - archetype: str
    - train_fitness: float
    - train_sharpe: float (fitness component)
    - train_pf: float (fitness component)
    - train_n_trades: int (fitness component)
    - forward_pnl: float
    - forward_wr: float
    - forward_trades: int
    - forward_max_dd: float
    - regime_days: int
    - death_reason: str
    - winner: bool (forward_pnl > 0)
    - strategy_name: str
    - optimize_start: str
    - optimize_end: str
    - forward_end: str
    """
    # Handle RunPod wrapper: result may be nested under "output"
    if "output" in result_json and isinstance(result_json["output"], dict):
        result_json = result_json["output"]

    regimes = result_json.get("regimes", [])
    records = []
    for r in regimes:
        # Extract from the regime record structure
        strategy_def = r.get("winner_params", {})
        archetype = strategy_def.get("archetype", "unknown")

        # winner_fitness can be a float or a dict depending on version
        train_fit = r.get("winner_fitness", 0)
        if isinstance(train_fit, dict):
            train_fitness_val = train_fit.get("fitness", 0)
        else:
            train_fitness_val = float(train_fit) if train_fit else 0

        # Forward metrics
        fwd_pnl = r.get("forward_pnl", 0)
        fwd_wr = r.get("forward_win_rate", r.get("forward_wr", 0))
        fwd_trades = r.get("forward_trades", 0)
        fwd_max_dd = r.get("forward_max_dd", 0)

        # Dates
        opt_start = r.get("optimize_start", "")
        opt_end = r.get("optimize_end", "")
        fwd_end = r.get("forward_end", "")

        # Regime lifespan
        regime_days = r.get("forward_days", r.get("regime_lifespan_days", 0))

        # Fitness components (new fields — default to 0 for old records)
        train_sharpe = r.get("train_sharpe", 0.0)
        train_pf = r.get("train_pf", 0.0)
        train_n_trades = r.get("train_n_trades", 0)

        records.append({
            "archetype": archetype,
            "train_fitness": train_fitness_val,
            "train_sharpe": train_sharpe,
            "train_pf": train_pf,
            "train_n_trades": train_n_trades,
            "forward_pnl": fwd_pnl,
            "forward_wr": fwd_wr,
            "forward_trades": fwd_trades,
            "forward_max_dd": fwd_max_dd,
            "regime_days": regime_days,
            "death_reason": r.get("death_reason", ""),
            "winner": fwd_pnl > 0,
            "strategy_name": strategy_def.get("name", ""),
            "optimize_start": opt_start,
            "optimize_end": opt_end,
            "forward_end": fwd_end,
            # Fields previously missing from extraction
            "winner_params": strategy_def,
            "kill_posterior": r.get("kill_posterior"),
            "n_evolutions": r.get("n_evolutions", 0),
            "death_gex_regime": r.get("death_gex_regime"),
            "death_rv": r.get("death_rv"),
            "optimize_gex_regime": r.get("optimize_gex_regime"),
            "optimize_rv": r.get("optimize_rv"),
        })

    return records


def ingest_result_file(path: str | Path, asset: str = "MES") -> int:
    """Ingest a walk-forward result JSON into the regime DB.

    Routes to the correct per-asset DB based on the asset parameter.
    Returns number of new regimes added.
    """
    path = Path(path)
    with open(path) as f:
        result = json.load(f)

    # Auto-detect asset from result if not explicitly specified
    output = result.get("output", result) if isinstance(result.get("output"), dict) else result
    if asset == "MES" and output.get("asset"):
        asset = output["asset"]

    new_records = extract_regimes_from_result(result)
    if not new_records:
        print(f"  [meta] No regimes found in {path.name}")
        return 0

    db = load_regime_db(asset)

    # Deduplicate by (archetype, optimize_start, forward_end)
    existing_keys = {
        (r["archetype"], r["optimize_start"], r["forward_end"])
        for r in db
    }
    added = 0
    for rec in new_records:
        key = (rec["archetype"], rec["optimize_start"], rec["forward_end"])
        if key not in existing_keys:
            db.append(rec)
            existing_keys.add(key)
            added += 1

    if added:
        save_regime_db(db, asset)
    print(f"  [meta] Ingested {added} new {asset} regimes from {path.name} "
          f"(total: {len(db)})")
    return added


# ─── Meta-Parameter Update ──────────────────────────────────────────────────

def compute_archetype_scores(records: list[dict]) -> dict[str, float]:
    """Compute archetype scores from regime data.

    Score = mean forward Return:DD per regime, capped to [-2, 2].
    Uses Return:DD instead of raw P&L because scalability matters more —
    a high-R:DD archetype can be scaled on prop accounts.

    Falls back to DEFAULT_META_CONFIG for archetypes with < 3 regimes.
    """
    by_archetype: dict[str, list[float]] = defaultdict(list)
    for r in records:
        arch = r.get("archetype", "unknown")
        by_archetype[arch].append(_forward_return_dd(r))

    scores = dict(DEFAULT_META_CONFIG.archetype_scores)  # start with defaults

    for arch, rdds in by_archetype.items():
        if len(rdds) < 3:
            continue  # not enough data, keep default

        avg_rdd = sum(rdds) / len(rdds)
        score = max(-2.0, min(2.0, avg_rdd))  # clamp
        scores[arch] = round(score, 2)

    return scores


def compute_optimal_gate_threshold(records: list[dict],
                                    archetype_scores: dict[str, float],
                                    ) -> float:
    """Find gate threshold that maximizes aggregate forward P&L.

    Sweeps thresholds from -1.0 to 1.0 in steps of 0.1.
    For each threshold, sums the forward P&L of regimes that would pass.
    """
    if not records:
        return 0.0

    # Pre-compute gate scores for each regime
    scored = []
    for r in records:
        arch = r.get("archetype", "unknown")
        gate_score = archetype_scores.get(arch, 0.0)
        tf = r.get("train_fitness", 0)
        if tf >= 3.0:
            gate_score += 0.5
        elif tf < 0.5:
            gate_score -= 0.5
        scored.append((gate_score, _forward_return_dd(r)))

    best_threshold = 0.0
    best_rdd = float("-inf")

    for t_x10 in range(-10, 11):  # -1.0 to 1.0
        threshold = t_x10 / 10.0
        passing = [rdd for gs, rdd in scored if gs >= threshold]
        if len(passing) < 3:
            continue
        avg_rdd = sum(passing) / len(passing)
        if avg_rdd > best_rdd:
            best_rdd = avg_rdd
            best_threshold = threshold

    return round(best_threshold, 1)


def compute_optimal_early_death(records: list[dict]) -> tuple[float, int]:
    """Find optimal early death WR floor and min trades.

    Sweeps WR floor [25, 30, 35, 40, 45] and min trades [10, 15, 20].
    Picks combo that maximizes: avg P&L of surviving regimes.

    Returns (wr_floor, min_trades).
    """
    if not records:
        return 35.0, 15

    best_wr = 35.0
    best_min = 15
    best_metric = float("-inf")

    for wr_floor in [25.0, 30.0, 35.0, 40.0, 45.0]:
        for min_trades in [10, 15, 20]:
            # Simulate: regimes that would survive the early death check
            survived_rdd = []
            for r in records:
                fwd_wr = r.get("forward_wr", 50)
                fwd_trades = r.get("forward_trades", 0)

                # Would this regime have been killed early?
                if fwd_trades >= min_trades and fwd_wr < wr_floor:
                    continue  # killed
                survived_rdd.append(_forward_return_dd(r))

            if not survived_rdd:
                continue

            avg_rdd = sum(survived_rdd) / len(survived_rdd)
            # Reward both higher avg Return:DD and keeping enough regimes
            metric = avg_rdd * min(len(survived_rdd) / len(records), 1.0)

            if metric > best_metric:
                best_metric = metric
                best_wr = wr_floor
                best_min = min_trades

    return best_wr, best_min


def _forward_return_dd(r: dict) -> float:
    """Compute forward Return:DD for a regime record.

    Return:DD = forward_pnl / forward_max_dd.
    - If no drawdown, returns pnl/100 (capped reward for zero-DD winners).
    - If pnl <= 0, returns negative ratio (penalty scales with loss severity).
    """
    pnl = r.get("forward_pnl", 0)
    dd = r.get("forward_max_dd", 0)

    if dd <= 0:
        # No drawdown recorded — cap at pnl/100 to reward but not overweight
        return max(pnl / 100.0, -1.0) if pnl != 0 else 0.0

    if pnl <= 0:
        # Losing strategy — return negative ratio (worse loss = worse score)
        return pnl / max(dd, 1.0)

    return pnl / dd


def compute_optimal_fitness_weights(records: list[dict]) -> tuple[float, float, float, float]:
    """Find optimal (sharpe_weight, pf_weight, trade_count_weight, return_dd_weight).

    Sweeps weight combos and picks the combo where high-weight-component
    regimes have the best forward Return:DD. All 4 weights are tuned together.

    The scoring mirrors the new compute_fitness() logic: Sharpe and PF
    contributions are capped, extreme values are penalized, and trade count
    is the dominant signal.

    Returns (sharpe_weight, pf_weight, trade_count_weight, return_dd_weight).
    """
    usable = [r for r in records
              if r.get("train_sharpe", 0) != 0 or r.get("train_pf", 0) != 0]

    if len(usable) < 10:
        return 0.20, 0.20, 0.50, 0.10

    best_weights = (0.20, 0.20, 0.50, 0.10)
    best_metric = float("-inf")

    for sw_x10 in range(1, 5):  # sharpe: 0.1 to 0.4 (capped — no longer dominant)
        for pw_x10 in range(1, 5):  # pf: 0.1 to 0.4
            tw_x10 = 10 - sw_x10 - pw_x10
            if tw_x10 < 3:  # trade_count must be at least 0.3
                continue
            sw = sw_x10 / 10.0
            pw = pw_x10 / 10.0
            tw = tw_x10 / 10.0

            for rdd_x10 in range(0, 4):  # return_dd: 0.0 to 0.3
                rdd = rdd_x10 / 10.0

                scores_fwd = []
                for r in usable:
                    sharpe = r.get("train_sharpe", 0)
                    pf = r.get("train_pf", 0)
                    n_trades = r.get("train_n_trades", 0)
                    train_fitness = r.get("train_fitness", 0)
                    rdd_score = min(train_fitness / 3.0, 1.0) if train_fitness > 0 else 0.0

                    # Mirror compute_fitness: cap sharpe/pf, penalize extremes
                    overfit_pen = max(0, sharpe - 5) * 0.3 + max(0, pf - 5) * 0.2
                    trade_bonus = min(n_trades / 20.0, 1.0)

                    train_score = (min(sharpe, 5.0) * sw
                                   + min(pf, 5.0) * pw
                                   + trade_bonus * tw
                                   + rdd_score * rdd
                                   - overfit_pen)
                    scores_fwd.append((train_score, _forward_return_dd(r)))

                scores_fwd.sort(key=lambda x: x[0], reverse=True)
                mid = len(scores_fwd) // 2
                if mid < 3:
                    continue
                top_avg = sum(p for _, p in scores_fwd[:mid]) / mid
                bot_avg = sum(p for _, p in scores_fwd[mid:]) / (len(scores_fwd) - mid)
                metric = top_avg - bot_avg

                if metric > best_metric:
                    best_metric = metric
                    best_weights = (sw, pw, tw, rdd)

    return best_weights


def compute_optimal_overfit_penalties(records: list[dict]) -> dict:
    """Find optimal overfit penalty thresholds from regime data.

    Sweeps sharpe_penalty_above, pf_penalty_above, and min_trade_count
    to find values that maximize forward Return:DD separation between
    top-half and bottom-half candidates.

    Returns dict with tuned values.
    """
    usable = [r for r in records
              if r.get("train_sharpe", 0) != 0 or r.get("train_pf", 0) != 0]

    if len(usable) < 15:
        return {
            "sharpe_penalty_above": 5.0,
            "pf_penalty_above": 5.0,
            "hard_sharpe_ceiling": 10.0,
            "hard_pf_ceiling": 8.0,
            "min_trade_count": 8,
        }

    # Temporal holdout: train on first 70%, validate on last 30%
    # to avoid p-hacking min_trade_count on the same data it filters
    usable_sorted = sorted(usable, key=lambda r: r.get("optimize_start", "") or r.get("forward_start", ""))
    split_idx = int(len(usable_sorted) * 0.7)
    train_set = usable_sorted[:split_idx]
    val_set = usable_sorted[split_idx:]

    # Fall back to full dataset if validation set is too small
    if len(val_set) < 5 or len(train_set) < 10:
        train_set = usable_sorted
        val_set = usable_sorted

    best = {
        "sharpe_penalty_above": 5.0,
        "pf_penalty_above": 5.0,
        "hard_sharpe_ceiling": 10.0,
        "hard_pf_ceiling": 8.0,
        "min_trade_count": 8,
    }
    best_metric = float("-inf")

    for sp in [3.0, 4.0, 5.0, 6.0, 7.0]:        # sharpe_penalty_above
        for pp in [3.0, 4.0, 5.0, 6.0, 7.0]:      # pf_penalty_above
            for mtc in [6, 8, 10, 12, 15, 18, 20, 25]:  # min_trade_count
                # Train: find parameters that separate good from bad
                surviving = []
                for r in train_set:
                    sharpe = r.get("train_sharpe", 0)
                    pf = r.get("train_pf", 0)
                    n_trades = r.get("train_n_trades", 0)

                    # Hard reject (fixed ceilings for this sweep)
                    if sharpe > 10.0 or pf > 8.0 or n_trades < mtc:
                        continue

                    # Score with penalties
                    overfit_pen = max(0, sharpe - sp) * 0.3 + max(0, pf - pp) * 0.2
                    trade_bonus = min(n_trades / 20.0, 1.0)
                    score = (min(sharpe, sp) * 0.2
                             + min(pf, pp) * 0.2
                             + trade_bonus * 0.5
                             - overfit_pen)
                    surviving.append((score, _forward_return_dd(r)))

                if len(surviving) < 8:
                    continue

                # Validate: check if these params also separate on held-out data
                val_surviving = []
                for r in val_set:
                    sharpe = r.get("train_sharpe", 0)
                    pf = r.get("train_pf", 0)
                    n_trades = r.get("train_n_trades", 0)
                    if sharpe > 10.0 or pf > 8.0 or n_trades < mtc:
                        continue
                    overfit_pen = max(0, sharpe - sp) * 0.3 + max(0, pf - pp) * 0.2
                    trade_bonus = min(n_trades / 20.0, 1.0)
                    score = (min(sharpe, sp) * 0.2
                             + min(pf, pp) * 0.2
                             + trade_bonus * 0.5
                             - overfit_pen)
                    val_surviving.append((score, _forward_return_dd(r)))

                if len(val_surviving) < 3:
                    continue

                # Use validation set metric (not training set) for selection
                val_surviving.sort(key=lambda x: x[0], reverse=True)
                mid = len(val_surviving) // 2
                if mid == 0:
                    continue
                top_avg = sum(p for _, p in val_surviving[:mid]) / mid
                bot_avg = sum(p for _, p in val_surviving[mid:]) / (len(val_surviving) - mid)
                metric = top_avg - bot_avg

                if metric > best_metric:
                    best_metric = metric
                    best = {
                        "sharpe_penalty_above": sp,
                        "pf_penalty_above": pp,
                        "hard_sharpe_ceiling": 10.0,
                        "hard_pf_ceiling": 8.0,
                        "min_trade_count": mtc,
                    }

    return best


def compute_optimal_kill_thresholds(records: list[dict]) -> dict:
    """Find optimal kill thresholds from regime data.

    Tunes: max_dd_dollars, negative_trajectory_pnl, negative_trajectory_trades,
           flat_regime_pnl, flat_regime_trades.

    For each parameter, sweeps a range and picks the value that maximizes
    aggregate forward P&L (balancing early kills that save money vs
    premature kills that miss recoveries).

    Returns dict of optimal values.
    """
    if len(records) < 10:
        return {
            "max_dd_dollars": 3500.0,
            "negative_trajectory_pnl": -500.0,
            "negative_trajectory_trades": 20,
            "flat_regime_pnl": 200.0,
            "flat_regime_trades": 25,
        }

    # ── max_dd_dollars ──
    # What DD limit maximizes average forward Return:DD of surviving regimes?
    # Tighter DD = fewer blowups pass through, better avg Return:DD
    best_dd = 3500.0
    best_dd_metric = float("-inf")
    for dd in [2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        rdds = []
        for r in records:
            fwd_max_dd = r.get("forward_max_dd", 0)
            if fwd_max_dd >= dd:
                # Killed at DD limit — count as capped loss
                fwd_pnl = r.get("forward_pnl", 0)
                capped_pnl = max(fwd_pnl, -dd)
                rdds.append(capped_pnl / dd)
            else:
                rdds.append(_forward_return_dd(r))
        if rdds:
            avg_rdd = sum(rdds) / len(rdds)
            if avg_rdd > best_dd_metric:
                best_dd_metric = avg_rdd
                best_dd = float(dd)

    # ── negative_trajectory ──
    # When to kill a regime that's losing money steadily
    best_neg_pnl = -500.0
    best_neg_trades = 20
    best_neg_metric = float("-inf")
    for neg_pnl in [-300, -500, -750, -1000]:
        for neg_trades in [15, 20, 25, 30]:
            rdds = []
            for r in records:
                fwd_pnl = r.get("forward_pnl", 0)
                fwd_trades = r.get("forward_trades", 0)

                if fwd_trades >= neg_trades and fwd_pnl < neg_pnl:
                    # Killed — cap loss
                    capped = max(fwd_pnl, neg_pnl)
                    dd = r.get("forward_max_dd", abs(capped))
                    rdds.append(capped / max(dd, 1.0))
                else:
                    rdds.append(_forward_return_dd(r))

            if rdds:
                avg_rdd = sum(rdds) / len(rdds)
                if avg_rdd > best_neg_metric:
                    best_neg_metric = avg_rdd
                    best_neg_pnl = float(neg_pnl)
                    best_neg_trades = neg_trades

    # ── flat_regime ──
    # When to kill a regime that's going nowhere
    best_flat_pnl = 200.0
    best_flat_trades = 25
    best_flat_metric = float("-inf")
    for flat_pnl in [100, 200, 300, 500]:
        for flat_trades in [20, 25, 30, 40]:
            rdds = []
            for r in records:
                fwd_pnl = r.get("forward_pnl", 0)
                fwd_trades = r.get("forward_trades", 0)

                if fwd_trades >= flat_trades and 0 < fwd_pnl < flat_pnl:
                    # Killed as flat — take the small profit
                    rdds.append(_forward_return_dd(r))
                else:
                    rdds.append(_forward_return_dd(r))

            if rdds:
                avg_rdd = sum(rdds) / len(rdds)
                if avg_rdd > best_flat_metric:
                    best_flat_metric = avg_rdd
                    best_flat_pnl = float(flat_pnl)
                    best_flat_trades = flat_trades

    return {
        "max_dd_dollars": best_dd,
        "negative_trajectory_pnl": best_neg_pnl,
        "negative_trajectory_trades": best_neg_trades,
        "flat_regime_pnl": best_flat_pnl,
        "flat_regime_trades": best_flat_trades,
    }


def compute_optimal_fitness_bonuses(records: list[dict],
                                     archetype_scores: dict[str, float],
                                     ) -> tuple[float, float]:
    """Find optimal fitness_bonus_high and fitness_bonus_low from regime data.

    fitness_bonus_high: bonus added to gate score when train_fitness >= threshold
    fitness_bonus_low: penalty subtracted from gate score when train_fitness < threshold

    Sweeps thresholds and picks values that maximize pass-through P&L.

    Returns (fitness_bonus_high, fitness_bonus_low).
    """
    if len(records) < 10:
        return 3.0, 0.5

    best_high = 3.0
    best_low = 0.5
    best_metric = float("-inf")

    for high_x10 in range(10, 51, 5):  # 1.0 to 5.0
        for low_x10 in range(0, 21, 5):  # 0.0 to 2.0
            bonus_high = high_x10 / 10.0
            bonus_low = low_x10 / 10.0

            # Simulate gate scoring with these bonuses
            passed_rdds = []
            for r in records:
                arch = r.get("archetype", "unknown")
                gate_score = archetype_scores.get(arch, 0.0)
                tf = r.get("train_fitness", 0)
                if tf >= 3.0:
                    gate_score += bonus_high
                elif tf < 0.5:
                    gate_score -= bonus_low

                if gate_score >= 0.0:  # Would pass gate
                    passed_rdds.append(_forward_return_dd(r))

            # Reward both avg Return:DD and keeping enough regimes active
            if len(passed_rdds) < 5:
                continue
            metric = sum(passed_rdds) / len(passed_rdds)

            if metric > best_metric:
                best_metric = metric
                best_high = bonus_high
                best_low = bonus_low

    return best_high, best_low


def compute_optimal_recency_weight(records: list[dict]) -> float:
    """Find optimal recency_weight from regime data.

    Recency weight controls how much the fitness function favors recent
    performance over the full window. Higher values = more emphasis on
    recent trades performing well.

    Approach: for each candidate weight, simulate a recency bias score
    using regime date ordering, then check if higher-recency-bias regimes
    tend to have better forward P&L.

    Returns optimal recency_weight in [0.0, 0.8].
    """
    if len(records) < 15:
        return 0.4

    # Sort records by optimize_end date to establish temporal ordering
    dated = [r for r in records if r.get("optimize_end")]
    if len(dated) < 10:
        return 0.4

    dated.sort(key=lambda r: r["optimize_end"])

    best_rw = 0.4
    best_metric = float("-inf")

    for rw_x10 in range(0, 9):  # 0.0 to 0.8
        rw = rw_x10 / 10.0

        # For each regime, compute a recency-adjusted score:
        # blend of train_fitness with position in temporal sequence
        # Higher recency_weight → later regimes get boosted more
        scores_fwd = []
        n = len(dated)
        for i, r in enumerate(dated):
            tf = r.get("train_fitness", 0)
            # Temporal position: 0 (oldest) to 1 (newest)
            temporal_pos = i / max(n - 1, 1)
            # Recency-adjusted score
            adjusted = tf * (1 - rw) + tf * temporal_pos * rw
            scores_fwd.append((adjusted, _forward_return_dd(r)))

        scores_fwd.sort(key=lambda x: x[0], reverse=True)
        mid = len(scores_fwd) // 2
        if mid < 3:
            continue
        top_avg = sum(p for _, p in scores_fwd[:mid]) / mid
        bot_avg = sum(p for _, p in scores_fwd[mid:]) / (len(scores_fwd) - mid)
        metric = top_avg - bot_avg

        if metric > best_metric:
            best_metric = metric
            best_rw = rw

    return best_rw


def compute_optimal_trajectory_weight(
    candidates: list[dict],
) -> float:
    """Find optimal trajectory_weight from candidate log data.

    Uses candidates with pnls_array to test which trajectory_weight
    best separates validation winners from losers.

    Returns optimal trajectory_weight in [0.05, 0.30].
    """
    usable = [c for c in candidates
              if c.get("pnls_array") and len(c.get("pnls_array", [])) >= 6
              and c.get("val_pnl") is not None]

    if len(usable) < 20:
        return 0.15

    best_tw = 0.15
    best_metric = float("-inf")

    for tw_x100 in range(5, 31, 5):  # 0.05 to 0.30
        tw = tw_x100 / 100.0
        scores = []
        for c in usable:
            pnls = c["pnls_array"]
            n = len(pnls)
            n3 = max(n // 3, 1)
            early = sum(pnls[:n3])
            late = sum(pnls[2 * n3:])
            activity = abs(early) + abs(late) + 1.0
            traj = max((late - early) / activity, 0.0)
            base_fitness = c.get("train_fitness", 0)
            adjusted = base_fitness + traj * tw
            val_pnl = c.get("val_pnl", 0) or 0
            scores.append((adjusted, val_pnl))

        scores.sort(key=lambda x: x[0], reverse=True)
        mid = len(scores) // 2
        if mid < 5:
            continue
        top_avg = sum(p for _, p in scores[:mid]) / mid
        bot_avg = sum(p for _, p in scores[mid:]) / (len(scores) - mid)
        metric = top_avg - bot_avg

        if metric > best_metric:
            best_metric = metric
            best_tw = tw

    return best_tw


def compute_optimal_bonus_penalties(
    candidates: list[dict],
) -> dict:
    """Find optimal tsl_be_bonus, frontload_penalty, sweet_spot_bonus.

    Sweeps each parameter and picks values that maximize validation
    P&L separation between top-half and bottom-half candidates.

    Returns dict with tuned values.
    """
    usable = [c for c in candidates
              if c.get("val_pnl") is not None
              and c.get("train_fitness") is not None]

    if len(usable) < 20:
        return {
            "tsl_be_bonus": 0.15,
            "frontload_penalty": 0.20,
            "sweet_spot_bonus": 0.20,
        }

    best = {"tsl_be_bonus": 0.15, "frontload_penalty": 0.20, "sweet_spot_bonus": 0.20}
    best_metric = float("-inf")

    for tsl_x100 in range(5, 26, 5):      # 0.05 to 0.25
        for fl_x100 in range(10, 41, 10):  # 0.10 to 0.40
            for ss_x100 in range(10, 31, 10):  # 0.10 to 0.30
                tsl_b = tsl_x100 / 100.0
                fl_p = fl_x100 / 100.0
                ss_b = ss_x100 / 100.0

                scores = []
                for c in usable:
                    base = c.get("train_fitness", 0)
                    adjusted = base

                    # TSL/BE bonus
                    if c.get("has_tsl_be"):
                        adjusted += tsl_b

                    # Sweet spot bonus (approximate — check if train metrics are in range)
                    sharpe = c.get("train_sharpe", 0)
                    pf = c.get("train_pf", 0)
                    n = c.get("train_n_trades", 0)
                    if 3 <= sharpe < 9 and 2 <= pf < 5 and 5 <= n <= 12:
                        adjusted += ss_b

                    # Frontload penalty
                    if c.get("frontloaded"):
                        adjusted -= fl_p

                    val_pnl = c.get("val_pnl", 0) or 0
                    scores.append((adjusted, val_pnl))

                scores.sort(key=lambda x: x[0], reverse=True)
                mid = len(scores) // 2
                if mid < 5:
                    continue
                top_avg = sum(p for _, p in scores[:mid]) / mid
                bot_avg = sum(p for _, p in scores[mid:]) / (len(scores) - mid)
                metric = top_avg - bot_avg

                if metric > best_metric:
                    best_metric = metric
                    best = {
                        "tsl_be_bonus": tsl_b,
                        "frontload_penalty": fl_p,
                        "sweet_spot_bonus": ss_b,
                    }

    return best


def compute_optimal_mfe_mae_gate(
    candidates: list[dict],
) -> dict:
    """Find optimal mfe_mae_ratio_ceiling and penalty.

    Sweeps ceiling (3-8) and penalty (0.10-0.35) to maximize val P&L
    separation between top-half and bottom-half candidates.
    """
    usable = [c for c in candidates
              if c.get("val_pnl") is not None
              and c.get("train_fitness") is not None
              and c.get("mfe_mae_ratio") is not None]

    if len(usable) < 20:
        return {"mfe_mae_ratio_ceiling": 5.0, "mfe_mae_ceiling_penalty": 0.20}

    best = {"mfe_mae_ratio_ceiling": 5.0, "mfe_mae_ceiling_penalty": 0.20}
    best_metric = float("-inf")

    for ceil_x10 in range(30, 81, 5):     # 3.0 to 8.0
        for pen_x100 in range(10, 36, 5):  # 0.10 to 0.35
            ceiling = ceil_x10 / 10.0
            penalty = pen_x100 / 100.0

            scores = []
            for c in usable:
                base = c.get("train_fitness", 0)
                adjusted = base
                if c["mfe_mae_ratio"] > ceiling:
                    adjusted -= penalty
                val_pnl = c.get("val_pnl", 0) or 0
                scores.append((adjusted, val_pnl))

            scores.sort(key=lambda x: x[0], reverse=True)
            mid = len(scores) // 2
            if mid < 5:
                continue
            top_avg = sum(p for _, p in scores[:mid]) / mid
            bot_avg = sum(p for _, p in scores[mid:]) / (len(scores) - mid)
            metric = top_avg - bot_avg

            if metric > best_metric:
                best_metric = metric
                best = {"mfe_mae_ratio_ceiling": ceiling,
                        "mfe_mae_ceiling_penalty": penalty}

    return best


def compute_optimal_tsl_ranges(
    candidates: list[dict],
) -> dict:
    """Find optimal TSL/BE trigger and trail distance ranges.

    Analyzes candidates with TSL/BE to find which be_trigger_pts and
    trail_distance_pts values produce the best val P&L.
    """
    usable = [c for c in candidates
              if c.get("val_pnl") is not None
              and c.get("has_tsl_be")]

    if len(usable) < 20:
        return {
            "tsl_min_trail_pts": 4.0, "tsl_max_trail_pts": 6.0,
            "tsl_be_min_trigger_pts": 10.0, "tsl_be_max_trigger_pts": 13.0,
        }

    # Extract actual be_trigger and trail values from strategy defs
    be_vals, trail_vals = [], []
    good_be, good_trail = [], []  # from positive-val candidates
    for c in usable:
        sdef = c.get("strategy_def", {})
        exit_ = sdef.get("exit", {}) if isinstance(sdef, dict) else {}
        be = exit_.get("be_trigger_pts")
        tr = exit_.get("trail_distance_pts")
        val = c.get("val_pnl", 0) or 0
        if be:
            be_vals.append(be)
            if val > 0:
                good_be.append(be)
        if tr:
            trail_vals.append(tr)
            if val > 0:
                good_trail.append(tr)

    # Use positive-val ranges if enough data, else full range
    if len(good_be) >= 5:
        be_lo = round(min(good_be) * 0.9, 1)
        be_hi = round(max(good_be) * 1.1, 1)
    elif be_vals:
        be_lo = round(min(be_vals), 1)
        be_hi = round(max(be_vals), 1)
    else:
        be_lo, be_hi = 6.0, 12.0

    if len(good_trail) >= 5:
        tr_lo = round(min(good_trail) * 0.9, 1)
        tr_hi = round(max(good_trail) * 1.1, 1)
    elif trail_vals:
        tr_lo = round(min(trail_vals), 1)
        tr_hi = round(max(trail_vals), 1)
    else:
        tr_lo, tr_hi = 7.0, 12.0

    # Clamp to reasonable bounds
    be_lo = max(4.0, min(be_lo, 10.0))
    be_hi = max(8.0, min(be_hi, 16.0))
    tr_lo = max(4.0, min(tr_lo, 10.0))
    tr_hi = max(8.0, min(tr_hi, 20.0))

    return {
        "tsl_min_trail_pts": tr_lo,
        "tsl_max_trail_pts": tr_hi,
        "tsl_be_min_trigger_pts": be_lo,
        "tsl_be_max_trigger_pts": be_hi,
    }


def compute_optimal_sl_tp_ranges(
    candidates: list[dict],
) -> dict:
    """Find optimal SL/TP fixed-pts ranges from candidate data.

    Analyzes candidates' exit SL/TP values to find which ranges
    produce the best validation P&L.
    """
    usable = [c for c in candidates
              if c.get("val_pnl") is not None]

    if len(usable) < 20:
        return {
            "sl_range_min": 8.0, "sl_range_max": 15.0,
            "tp_range_min": 10.0, "tp_range_max": 20.0,
        }

    sl_vals, tp_vals = [], []
    good_sl, good_tp = [], []
    for c in usable:
        sdef = c.get("strategy_def", {})
        exit_ = sdef.get("exit", {}) if isinstance(sdef, dict) else {}
        sl_spec = exit_.get("stop_loss", {})
        tp_spec = exit_.get("take_profit", {})
        val = c.get("val_pnl", 0) or 0

        # Extract SL pts
        if sl_spec.get("type") == "fixed_pts":
            sv = sl_spec.get("value")
            if sv:
                sl_vals.append(sv)
                if val > 0:
                    good_sl.append(sv)

        # Extract TP pts (fixed_pts type)
        if tp_spec.get("type") == "fixed_pts":
            tv = tp_spec.get("value")
            if tv:
                tp_vals.append(tv)
                if val > 0:
                    good_tp.append(tv)

    # Use positive-val ranges if enough data
    if len(good_sl) >= 5:
        sl_lo = round(min(good_sl) * 0.9, 1)
        sl_hi = round(max(good_sl) * 1.1, 1)
    elif sl_vals:
        sl_lo = round(min(sl_vals), 1)
        sl_hi = round(max(sl_vals), 1)
    else:
        sl_lo, sl_hi = 8.0, 15.0

    if len(good_tp) >= 5:
        tp_lo = round(min(good_tp) * 0.9, 1)
        tp_hi = round(max(good_tp) * 1.1, 1)
    elif tp_vals:
        tp_lo = round(min(tp_vals), 1)
        tp_hi = round(max(tp_vals), 1)
    else:
        tp_lo, tp_hi = 10.0, 20.0

    # Clamp to reasonable bounds
    sl_lo = max(6.0, min(sl_lo, 10.0))
    sl_hi = max(12.0, min(sl_hi, 18.0))
    tp_lo = max(8.0, min(tp_lo, 14.0))
    tp_hi = max(15.0, min(tp_hi, 25.0))

    return {
        "sl_range_min": sl_lo,
        "sl_range_max": sl_hi,
        "tp_range_min": tp_lo,
        "tp_range_max": tp_hi,
    }


def compute_optimal_bayesian_params(
    candidates: list[dict],
) -> dict:
    """Find optimal Bayesian edge scoring parameters.

    Sweeps assumed_edge_wr, prior_edge, and bayesian_weight to maximize
    val P&L separation between top-half and bottom-half candidates.

    Uses a simplified Bayesian posterior calculation (no scipy needed —
    binomial PMF approximated inline for small trade counts).
    """
    usable = [c for c in candidates
              if c.get("val_pnl") is not None
              and c.get("train_fitness") is not None
              and c.get("train_n_trades", 0) >= 5
              and c.get("train_wr") is not None]

    if len(usable) < 20:
        return {
            "bayesian_assumed_edge_wr": 0.70,
            "bayesian_prior_edge": 0.15,
            "bayesian_weight": 0.20,
            "bayesian_kill_threshold": 0.25,
        }

    def _binom_pmf(k, n, p):
        """Simple binomial PMF without scipy."""
        from math import comb, log, exp
        if p <= 0 or p >= 1:
            return 0.0
        try:
            log_pmf = log(comb(n, k)) + k * log(p) + (n - k) * log(1 - p)
            return exp(log_pmf)
        except (ValueError, OverflowError):
            return 0.0

    best = {
        "bayesian_assumed_edge_wr": 0.70,
        "bayesian_prior_edge": 0.15,
        "bayesian_weight": 0.20,
        "bayesian_kill_threshold": 0.25,
    }
    best_metric = float("-inf")

    for wr_x100 in range(55, 86, 5):      # assumed edge WR: 0.55 to 0.85
        for prior_x100 in range(5, 31, 5):  # prior: 0.05 to 0.30
            for bw_x100 in range(10, 31, 5):  # weight: 0.10 to 0.30
                assumed_wr = wr_x100 / 100.0
                prior = prior_x100 / 100.0
                bw = bw_x100 / 100.0

                scores = []
                for c in usable:
                    base = c.get("train_fitness", 0)
                    n_trades = c.get("train_n_trades", 0)
                    wr = c.get("train_wr", 50) / 100.0
                    n_wins = round(wr * n_trades)

                    # Compute Bayesian posterior
                    p_edge = _binom_pmf(n_wins, n_trades, assumed_wr)
                    p_noise = _binom_pmf(n_wins, n_trades, 0.50)
                    num = p_edge * prior
                    den = num + p_noise * (1.0 - prior)
                    posterior = num / den if den > 1e-10 else prior

                    adjusted = base + posterior * bw
                    val_pnl = c.get("val_pnl", 0) or 0
                    scores.append((adjusted, val_pnl))

                scores.sort(key=lambda x: x[0], reverse=True)
                mid = len(scores) // 2
                if mid < 5:
                    continue
                top_avg = sum(p for _, p in scores[:mid]) / mid
                bot_avg = sum(p for _, p in scores[mid:]) / (len(scores) - mid)
                metric = top_avg - bot_avg

                if metric > best_metric:
                    best_metric = metric
                    best = {
                        "bayesian_assumed_edge_wr": assumed_wr,
                        "bayesian_prior_edge": prior,
                        "bayesian_weight": bw,
                        "bayesian_kill_threshold": 0.25,  # keep fixed for now
                    }

    return best


def _ema_blend(old: float, new: float, lr: float) -> float:
    """EMA blend helper."""
    return old * (1 - lr) + new * lr


def update_meta_config(
    current: MetaConfig | None = None,
    records: list[dict] | None = None,
    candidates: list[dict] | None = None,
    learning_rate: float = 0.5,
    asset: str = "MES",
) -> MetaConfig:
    """Compute updated MetaConfig from regime data + candidate logs.

    Tunes ALL parameters using data-driven optimization:
    - Archetype scores — from mean forward Return:DD per archetype
    - Gate threshold — sweep for max aggregate Return:DD
    - Early death WR floor + min trades — sweep for best survival Return:DD
    - Fitness weights (sharpe, PF, trade_count, return_dd) — correlation with forward Return:DD
    - Kill thresholds (DD, trajectory, flat) — sweep for optimal cutoffs
    - Gate bonuses (high, low) — sweep for optimal pass-through Return:DD
    - Overfit penalties (sharpe/pf penalty, min_trade_count)
    - Recency weight
    - Trajectory weight — from candidate logs pnls_array
    - TSL/BE bonus, frontload penalty, sweet spot bonus — from candidate logs

    Uses exponential moving average with learning_rate to blend
    new estimates with current config (prevents wild swings).

    Args:
        current: Current MetaConfig (or None for defaults)
        records: Regime records (or None to load from DB)
        candidates: Candidate log records (or None to load from JSONL)
        learning_rate: How fast to move toward new estimates (0=no change, 1=full replace)
        asset: Asset ticker for per-asset DB lookup

    Returns:
        Updated MetaConfig
    """
    if current is None:
        current = DEFAULT_META_CONFIG
    if records is None:
        records = load_regime_db(asset)
    if candidates is None:
        candidates = load_candidate_logs()

    if len(records) < 5:
        print(f"  [meta] Only {len(records)} regimes — keeping defaults")
        return current

    print(f"  [meta] Updating ALL meta-config params from {len(records)} regimes...")
    lr = learning_rate

    # 1. Archetype scores
    new_arch_scores = compute_archetype_scores(records)
    blended_arch = {}
    for arch in set(list(current.archetype_scores.keys()) + list(new_arch_scores.keys())):
        old = current.archetype_scores.get(arch, 0.0)
        new = new_arch_scores.get(arch, old)
        blended_arch[arch] = round(_ema_blend(old, new, lr), 2)

    # 2. Gate threshold
    new_threshold = compute_optimal_gate_threshold(records, blended_arch)
    blended_threshold = round(_ema_blend(current.gate_threshold, new_threshold, lr), 2)

    # 3. Early death params
    new_wr_floor, new_min_trades = compute_optimal_early_death(records)
    blended_wr = round(_ema_blend(current.early_death_wr_floor, new_wr_floor, lr), 1)
    blended_min = round(_ema_blend(current.early_death_min_trades, new_min_trades, lr))

    # 4. Fitness weights (sharpe, PF, trade_count, return_dd)
    new_sw, new_pw, _, new_rddw = compute_optimal_fitness_weights(records)
    blended_sw = round(_ema_blend(current.sharpe_weight, new_sw, lr), 2)
    blended_pw = round(_ema_blend(current.pf_weight, new_pw, lr), 2)
    blended_tw = round(1.0 - blended_sw - blended_pw, 2)  # ensure sum = 1.0
    blended_rddw = round(_ema_blend(current.return_dd_weight, new_rddw, lr), 2)
    # Clamp: trade_count must be at least 0.30, sharpe/pf max 0.40 each
    blended_sw = min(blended_sw, 0.40)
    blended_pw = min(blended_pw, 0.40)
    blended_tw = round(1.0 - blended_sw - blended_pw, 2)
    if blended_tw < 0.30:
        blended_tw = 0.30
        remainder = 1.0 - blended_tw
        ratio = blended_sw / (blended_sw + blended_pw) if (blended_sw + blended_pw) > 0 else 0.5
        blended_sw = round(remainder * ratio, 2)
        blended_pw = round(remainder * (1 - ratio), 2)
    blended_rddw = max(0.0, min(0.3, blended_rddw))

    # 4b. Overfit penalty params
    new_penalties = compute_optimal_overfit_penalties(records)
    blended_sp_above = round(_ema_blend(
        current.sharpe_penalty_above, new_penalties["sharpe_penalty_above"], lr), 1)
    blended_pp_above = round(_ema_blend(
        current.pf_penalty_above, new_penalties["pf_penalty_above"], lr), 1)
    blended_mtc = round(_ema_blend(
        current.min_trade_count, new_penalties["min_trade_count"], lr))
    # Clamp ranges
    blended_sp_above = max(3.0, min(7.0, blended_sp_above))
    blended_pp_above = max(3.0, min(7.0, blended_pp_above))
    blended_mtc = max(6, min(15, blended_mtc))

    # 5. Kill thresholds
    kill_thresholds = compute_optimal_kill_thresholds(records)
    blended_dd = round(_ema_blend(
        current.max_dd_dollars, kill_thresholds["max_dd_dollars"], lr))
    blended_neg_pnl = round(_ema_blend(
        current.negative_trajectory_pnl, kill_thresholds["negative_trajectory_pnl"], lr))
    blended_neg_trades = round(_ema_blend(
        current.negative_trajectory_trades, kill_thresholds["negative_trajectory_trades"], lr))
    blended_flat_pnl = round(_ema_blend(
        current.flat_regime_pnl, kill_thresholds["flat_regime_pnl"], lr))
    blended_flat_trades = round(_ema_blend(
        current.flat_regime_trades, kill_thresholds["flat_regime_trades"], lr))

    # 6. Gate bonuses
    new_bonus_high, new_bonus_low = compute_optimal_fitness_bonuses(
        records, blended_arch)
    blended_bonus_high = round(_ema_blend(
        current.fitness_bonus_high, new_bonus_high, lr), 1)
    blended_bonus_low = round(_ema_blend(
        current.fitness_bonus_low, new_bonus_low, lr), 1)

    # 7. Recency weight
    new_recency = compute_optimal_recency_weight(records)
    blended_recency = round(_ema_blend(current.recency_weight, new_recency, lr), 2)
    blended_recency = max(0.0, min(0.8, blended_recency))

    # 8. Trajectory weight (from candidate logs)
    new_traj_w = compute_optimal_trajectory_weight(candidates)
    blended_traj_w = round(_ema_blend(current.trajectory_weight, new_traj_w, lr), 2)
    blended_traj_w = max(0.05, min(0.30, blended_traj_w))

    # 9. Bonus/penalty params (from candidate logs)
    new_bp = compute_optimal_bonus_penalties(candidates)
    blended_tsl_be = round(_ema_blend(current.tsl_be_bonus, new_bp["tsl_be_bonus"], lr), 2)
    blended_fl_pen = round(_ema_blend(current.frontload_penalty, new_bp["frontload_penalty"], lr), 2)
    blended_ss_bonus = round(_ema_blend(current.sweet_spot_bonus, new_bp["sweet_spot_bonus"], lr), 2)
    blended_tsl_be = max(0.05, min(0.25, blended_tsl_be))
    blended_fl_pen = max(0.10, min(0.40, blended_fl_pen))
    blended_ss_bonus = max(0.10, min(0.30, blended_ss_bonus))

    # 10. MFE/MAE ratio gate (from candidate logs)
    new_mfe = compute_optimal_mfe_mae_gate(candidates)
    blended_mfe_ceil = round(_ema_blend(
        current.mfe_mae_ratio_ceiling, new_mfe["mfe_mae_ratio_ceiling"], lr), 1)
    blended_mfe_pen = round(_ema_blend(
        current.mfe_mae_ceiling_penalty, new_mfe["mfe_mae_ceiling_penalty"], lr), 2)
    blended_mfe_ceil = max(3.0, min(8.0, blended_mfe_ceil))
    blended_mfe_pen = max(0.10, min(0.35, blended_mfe_pen))

    # 11. TSL trail distance ranges (from candidate logs)
    new_tsl = compute_optimal_tsl_ranges(candidates)
    blended_tsl_min_trail = round(_ema_blend(
        current.tsl_min_trail_pts, new_tsl["tsl_min_trail_pts"], lr), 1)
    blended_tsl_max_trail = round(_ema_blend(
        current.tsl_max_trail_pts, new_tsl["tsl_max_trail_pts"], lr), 1)
    blended_tsl_be_min = round(_ema_blend(
        current.tsl_be_min_trigger_pts, new_tsl["tsl_be_min_trigger_pts"], lr), 1)
    blended_tsl_be_max = round(_ema_blend(
        current.tsl_be_max_trigger_pts, new_tsl["tsl_be_max_trigger_pts"], lr), 1)
    blended_tsl_min_trail = max(4.0, min(10.0, blended_tsl_min_trail))
    blended_tsl_max_trail = max(8.0, min(20.0, blended_tsl_max_trail))
    blended_tsl_be_min = max(4.0, min(10.0, blended_tsl_be_min))
    blended_tsl_be_max = max(8.0, min(16.0, blended_tsl_be_max))

    # 11b. SL/TP fixed-pts ranges (from candidate logs)
    new_sl_tp = compute_optimal_sl_tp_ranges(candidates)
    blended_sl_min = round(_ema_blend(
        current.sl_range_min, new_sl_tp["sl_range_min"], lr), 1)
    blended_sl_max = round(_ema_blend(
        current.sl_range_max, new_sl_tp["sl_range_max"], lr), 1)
    blended_tp_min = round(_ema_blend(
        current.tp_range_min, new_sl_tp["tp_range_min"], lr), 1)
    blended_tp_max = round(_ema_blend(
        current.tp_range_max, new_sl_tp["tp_range_max"], lr), 1)
    blended_sl_min = max(6.0, min(10.0, blended_sl_min))
    blended_sl_max = max(12.0, min(18.0, blended_sl_max))
    blended_tp_min = max(8.0, min(14.0, blended_tp_min))
    blended_tp_max = max(15.0, min(25.0, blended_tp_max))

    # 12. Bayesian edge scoring params (from candidate logs)
    new_bayes = compute_optimal_bayesian_params(candidates)
    blended_bayes_wr = round(_ema_blend(
        current.bayesian_assumed_edge_wr, new_bayes["bayesian_assumed_edge_wr"], lr), 2)
    blended_bayes_prior = round(_ema_blend(
        current.bayesian_prior_edge, new_bayes["bayesian_prior_edge"], lr), 2)
    blended_bayes_weight = round(_ema_blend(
        current.bayesian_weight, new_bayes["bayesian_weight"], lr), 2)
    blended_bayes_kill = round(_ema_blend(
        current.bayesian_kill_threshold, new_bayes["bayesian_kill_threshold"], lr), 2)
    # Clamp ranges
    blended_bayes_wr = max(0.55, min(0.85, blended_bayes_wr))
    blended_bayes_prior = max(0.05, min(0.30, blended_bayes_prior))
    blended_bayes_weight = max(0.10, min(0.30, blended_bayes_weight))
    blended_bayes_kill = max(0.10, min(0.40, blended_bayes_kill))

    updated = MetaConfig(
        archetype_scores=blended_arch,
        gate_threshold=blended_threshold,
        early_death_wr_floor=blended_wr,
        early_death_min_trades=blended_min,
        fitness_bonus_high=blended_bonus_high,
        fitness_bonus_low=blended_bonus_low,
        max_dd_dollars=blended_dd,
        negative_trajectory_pnl=blended_neg_pnl,
        negative_trajectory_trades=blended_neg_trades,
        flat_regime_pnl=blended_flat_pnl,
        flat_regime_trades=blended_flat_trades,
        sharpe_weight=blended_sw,
        pf_weight=blended_pw,
        trade_count_weight=blended_tw,
        return_dd_weight=blended_rddw,
        sharpe_penalty_above=blended_sp_above,
        pf_penalty_above=blended_pp_above,
        hard_sharpe_ceiling=10.0,  # fixed — not tuned
        hard_pf_ceiling=8.0,      # fixed — not tuned
        min_trade_count=blended_mtc,
        trajectory_weight=blended_traj_w,
        tsl_be_bonus=blended_tsl_be,
        frontload_penalty=blended_fl_pen,
        sweet_spot_bonus=blended_ss_bonus,
        recency_weight=blended_recency,
        mfe_mae_ratio_ceiling=blended_mfe_ceil,
        mfe_mae_ceiling_penalty=blended_mfe_pen,
        mfe_before_mae_bonus_high=current.mfe_before_mae_bonus_high,
        mfe_before_mae_bonus_mid=current.mfe_before_mae_bonus_mid,
        mfe_before_mae_penalty=current.mfe_before_mae_penalty,
        mfe_before_mae_high_threshold=current.mfe_before_mae_high_threshold,
        mfe_before_mae_mid_threshold=current.mfe_before_mae_mid_threshold,
        mfe_before_mae_low_threshold=current.mfe_before_mae_low_threshold,
        sl_range_min=blended_sl_min,
        sl_range_max=blended_sl_max,
        tp_range_min=blended_tp_min,
        tp_range_max=blended_tp_max,
        tsl_min_trail_pts=blended_tsl_min_trail,
        tsl_max_trail_pts=blended_tsl_max_trail,
        tsl_be_min_trigger_pts=blended_tsl_be_min,
        tsl_be_max_trigger_pts=blended_tsl_be_max,
        bayesian_assumed_edge_wr=blended_bayes_wr,
        bayesian_prior_edge=blended_bayes_prior,
        bayesian_weight=blended_bayes_weight,
        bayesian_kill_threshold=blended_bayes_kill,
        wr_circuit_breaker_threshold=current.wr_circuit_breaker_threshold,
        wr_circuit_breaker_min_trades=current.wr_circuit_breaker_min_trades,
    )

    return updated


def check_convergence(
    old: MetaConfig, new: MetaConfig,
    arch_tol: float = 0.1, scalar_tol: float = 0.05,
) -> bool:
    """Check if meta-config has converged (ALL params stopped moving).

    Returns True if converged.
    """
    # Check archetype scores
    for arch in set(list(old.archetype_scores.keys()) + list(new.archetype_scores.keys())):
        old_val = old.archetype_scores.get(arch, 0.0)
        new_val = new.archetype_scores.get(arch, 0.0)
        if abs(old_val - new_val) > arch_tol:
            return False

    # Gate & early death
    if abs(old.gate_threshold - new.gate_threshold) > scalar_tol:
        return False
    if abs(old.early_death_wr_floor - new.early_death_wr_floor) > 2.0:
        return False
    if abs(old.early_death_min_trades - new.early_death_min_trades) > 2:
        return False

    # Fitness weights
    if abs(old.sharpe_weight - new.sharpe_weight) > scalar_tol:
        return False
    if abs(old.pf_weight - new.pf_weight) > scalar_tol:
        return False
    if abs(old.trade_count_weight - new.trade_count_weight) > scalar_tol:
        return False
    if abs(old.return_dd_weight - new.return_dd_weight) > scalar_tol:
        return False
    if abs(old.recency_weight - new.recency_weight) > scalar_tol:
        return False

    # Kill thresholds
    if abs(old.max_dd_dollars - new.max_dd_dollars) > 250:
        return False
    if abs(old.negative_trajectory_pnl - new.negative_trajectory_pnl) > 100:
        return False
    if abs(old.negative_trajectory_trades - new.negative_trajectory_trades) > 3:
        return False
    if abs(old.flat_regime_pnl - new.flat_regime_pnl) > 50:
        return False
    if abs(old.flat_regime_trades - new.flat_regime_trades) > 3:
        return False

    # Gate bonuses
    if abs(old.fitness_bonus_high - new.fitness_bonus_high) > 0.5:
        return False
    if abs(old.fitness_bonus_low - new.fitness_bonus_low) > 0.2:
        return False

    # Overfit penalties
    if abs(old.sharpe_penalty_above - new.sharpe_penalty_above) > 0.5:
        return False
    if abs(old.pf_penalty_above - new.pf_penalty_above) > 0.5:
        return False
    if abs(old.min_trade_count - new.min_trade_count) > 2:
        return False

    # New params
    if abs(old.trajectory_weight - new.trajectory_weight) > scalar_tol:
        return False
    if abs(old.tsl_be_bonus - new.tsl_be_bonus) > scalar_tol:
        return False
    if abs(old.frontload_penalty - new.frontload_penalty) > scalar_tol:
        return False
    if abs(old.sweet_spot_bonus - new.sweet_spot_bonus) > scalar_tol:
        return False

    # Bayesian params
    if abs(old.bayesian_assumed_edge_wr - new.bayesian_assumed_edge_wr) > scalar_tol:
        return False
    if abs(old.bayesian_weight - new.bayesian_weight) > scalar_tol:
        return False

    return True


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    """CLI: ingest result files and/or update meta-config."""
    import argparse
    parser = argparse.ArgumentParser(description="Meta-Optimizer")
    parser.add_argument("--ingest", nargs="+", help="Result JSON files to ingest")
    parser.add_argument("--update", action="store_true", help="Update meta-config from DB")
    parser.add_argument("--show-db", action="store_true", help="Show regime DB stats")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--asset", type=str, default="MES",
                        help="Asset ticker (MES, F, BAC, SOFI, SNAP). Default: MES")
    args = parser.parse_args()

    asset = args.asset.upper()

    if args.ingest:
        for path in args.ingest:
            ingest_result_file(path, asset=asset)

    if args.show_db:
        db = load_regime_db(asset)
        print(f"\n{asset} Regime DB: {len(db)} records")
        by_arch = defaultdict(list)
        for r in db:
            by_arch[r["archetype"]].append(r["forward_pnl"])
        for arch, pnls in sorted(by_arch.items(), key=lambda x: -sum(x[1])):
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"  {arch}: {len(pnls)} regimes | avg P&L ${avg:,.0f} | WR {wr:.0f}%")

    if args.update:
        config = update_meta_config(learning_rate=args.lr, asset=asset)
        print(f"\nUpdated {asset} config:")
        print(json.dumps(config.to_dict(), indent=2))

        # Save config to disk for orchestrator
        config_path = _config_path(asset)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"Saved to {config_path}")


if __name__ == "__main__":
    main()
