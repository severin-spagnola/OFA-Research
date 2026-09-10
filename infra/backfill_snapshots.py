"""
Backfill Market Snapshots — Add market_snapshot to existing regime records.
==========================================================================
Uses Yahoo Finance daily data (ES=F as MES proxy) to compute market
conditions at each regime's forward_start date, then patches the JSON files.

Usage:
    python infra/backfill_snapshots.py
    python infra/backfill_snapshots.py --dry-run   # preview without saving
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _REPO_ROOT / "results"


def fetch_daily_data(ticker: str = "ES=F", start: str = "2021-12-01") -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance."""
    print(f"Fetching {ticker} daily data from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    print(f"  Got {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def compute_snapshot_from_daily(
    daily: pd.DataFrame,
    snapshot_date: str,
    lookback_days: int = 20,
    gap_threshold: float = 3.0,
) -> dict:
    """Compute market snapshot from daily OHLC data.

    Mirrors the logic in overfit_search.py compute_market_snapshot()
    but uses daily bars instead of 1-minute candles.
    """
    snap_ts = pd.Timestamp(snapshot_date)

    # Get trading days before snapshot
    mask = daily.index < snap_ts
    available = daily[mask]

    if len(available) < 5:
        return {"error": "insufficient_days", "n_days": len(available)}

    # Take last N trading days
    window = available.tail(lookback_days)
    n_days = len(window)

    highs = window["High"].values
    lows = window["Low"].values
    opens = window["Open"].values
    closes = window["Close"].values
    daily_ranges = highs - lows

    price_level = (opens[0] + closes[-1]) / 2

    # --- Realized volatility ---
    avg_range_20d = np.mean(daily_ranges)
    vol_pct = avg_range_20d / price_level * 100

    # --- Overnight range proxy (gap from prev close to open) ---
    on_ranges = []
    for i in range(1, n_days):
        # Overnight range ~ abs(open - prev_close) + some intraday ON activity
        # This is a proxy — real ON range from 1m data would be better
        gap = abs(opens[i] - closes[i - 1])
        on_ranges.append(gap)
    avg_on_range = np.mean(on_ranges) if on_ranges else 0
    on_range_pct = avg_on_range / price_level * 100

    # --- Trend ---
    ret_20d = (closes[-1] - opens[0]) / opens[0] * 100
    mid = max(n_days // 2, 1)
    ret_10d = (closes[-1] - opens[mid]) / opens[mid] * 100

    # --- Range regime ---
    avg_range_5d = np.mean(daily_ranges[-5:])
    range_ratio = avg_range_5d / avg_range_20d if avg_range_20d > 0 else 1.0

    # --- Gap frequency ---
    n_gap_up = 0
    n_gap_down = 0
    for i in range(1, n_days):
        diff = opens[i] - closes[i - 1]
        if diff > gap_threshold:
            n_gap_up += 1
        elif diff < -gap_threshold:
            n_gap_down += 1
    gap_pct = (n_gap_up + n_gap_down) / max(n_days - 1, 1) * 100

    # --- Directional consistency ---
    up_days = sum(1 for i in range(n_days) if closes[i] > opens[i])
    down_days = n_days - up_days
    dir_consistency = max(up_days, down_days) / n_days * 100

    # --- Streak ---
    streak = 1
    last_dir = 1 if closes[-1] > opens[-1] else -1
    for i in range(n_days - 2, -1, -1):
        d_dir = 1 if closes[i] > opens[i] else -1
        if d_dir == last_dir:
            streak += 1
        else:
            break

    # --- Max range 5d ---
    max_range_5d = float(np.max(daily_ranges[-5:]))
    max_range_5d_pct = max_range_5d / price_level * 100

    return {
        "snapshot_date": snapshot_date,
        "n_days": int(n_days),
        "price_level": round(float(price_level), 2),
        "source": "yahoo_daily_backfill",
        # Volatility
        "avg_daily_range_pts": round(float(avg_range_20d), 2),
        "daily_vol_pct": round(float(vol_pct), 4),
        "avg_on_range_pts": round(float(avg_on_range), 2),
        "on_range_pct": round(float(on_range_pct), 4),
        # Trend
        "ret_10d_pct": round(float(ret_10d), 4),
        "ret_20d_pct": round(float(ret_20d), 4),
        # Range regime
        "range_ratio_5d_20d": round(float(range_ratio), 4),
        "max_range_5d_pct": round(float(max_range_5d_pct), 4),
        # Gaps & direction
        "gap_pct": round(float(gap_pct), 2),
        "gap_up_count": int(n_gap_up),
        "gap_down_count": int(n_gap_down),
        "dir_consistency_pct": round(float(dir_consistency), 2),
        "streak_days": int(streak),
        "streak_direction": "up" if last_dir == 1 else "down",
    }


def backfill(dry_run: bool = False):
    """Backfill market_snapshot into all existing regime records."""
    daily = fetch_daily_data()

    files = sorted(_RESULTS_DIR.glob("overnight_*.json"))
    print(f"\nProcessing {len(files)} result files...")

    total_regimes = 0
    backfilled = 0
    skipped = 0
    errors = 0

    for path in files:
        with open(path) as f:
            data = json.load(f)

        regimes = data.get("regimes", [])
        if not regimes:
            continue

        modified = False
        for r in regimes:
            total_regimes += 1
            fs = r.get("forward_start", "")
            if not fs:
                skipped += 1
                continue

            # Skip if already has a snapshot
            if r.get("market_snapshot") and "error" not in r["market_snapshot"]:
                skipped += 1
                continue

            try:
                snap = compute_snapshot_from_daily(daily, fs)
                if "error" in snap:
                    print(f"  {path.name} regime {r.get('regime_id','?')} "
                          f"({fs}): {snap['error']}")
                    errors += 1
                    continue

                r["market_snapshot"] = snap
                modified = True
                backfilled += 1
            except Exception as e:
                print(f"  {path.name} regime {r.get('regime_id','?')} "
                      f"({fs}): ERROR {e}")
                errors += 1

        if modified and not dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {path.name}")

    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Total regimes: {total_regimes}")
    print(f"Backfilled:    {backfilled}")
    print(f"Skipped:       {skipped}")
    print(f"Errors:        {errors}")
    if dry_run:
        print(f"\n  (DRY RUN — no files were modified)")


def correlate():
    """Print correlation between snapshot features and forward P&L."""
    files = sorted(_RESULTS_DIR.glob("overnight_*.json"))
    records = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for r in data.get("regimes", []):
            snap = r.get("market_snapshot")
            if not snap or "error" in snap:
                continue
            fpnl = r.get("forward_pnl", 0) or 0
            ft = r.get("forward_trades", 0) or 0
            if ft < 1:
                continue
            rec = {
                "forward_pnl": fpnl,
                "label": 1 if fpnl > 0 else 0,
                "forward_start": r.get("forward_start", ""),
            }
            for k, v in snap.items():
                if isinstance(v, (int, float)):
                    rec[k] = v
            records.append(rec)

    if not records:
        print("No regimes with snapshots found.")
        return

    df = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print(f"SNAPSHOT FEATURE CORRELATIONS ({len(df)} regimes)")
    print(f"{'='*60}")

    features = [c for c in df.columns
                if c not in ("forward_pnl", "label", "forward_start", "n_days",
                             "price_level", "gap_up_count", "gap_down_count",
                             "snapshot_date")]

    print(f"\n{'Feature':<25s} {'Corr w/ PnL':>12s} {'Win PnL>0':>10s} {'Lose PnL<0':>10s} {'Sep':>5s}")
    print("-" * 65)
    winners = df[df["label"] == 1]
    losers = df[df["label"] == 0]

    correlations = []
    for feat in features:
        if feat not in df.columns:
            continue
        corr = df["forward_pnl"].corr(df[feat])
        w_mean = winners[feat].mean() if len(winners) > 0 else 0
        l_mean = losers[feat].mean() if len(losers) > 0 else 0
        sep = abs(w_mean - l_mean) / max(abs(w_mean), abs(l_mean), 0.01)
        marker = "***" if sep > 0.3 else "  *" if sep > 0.15 else ""
        correlations.append((abs(corr), feat, corr, w_mean, l_mean, marker))

    for _, feat, corr, w_mean, l_mean, marker in sorted(correlations, reverse=True):
        print(f"  {feat:<23s} {corr:>+10.4f}   {w_mean:>10.3f} {l_mean:>10.3f} {marker:>5s}")

    # Year breakdown with snapshot means
    print(f"\nSnapshot means by half-year:")
    df["half"] = df["forward_start"].str[:4] + "-H" + df["forward_start"].str[5:7].astype(int).apply(
        lambda m: "1" if m <= 6 else "2")
    for half, grp in sorted(df.groupby("half")):
        pnl = grp["forward_pnl"].sum()
        wr = (grp["label"].sum() / len(grp) * 100)
        vol = grp["daily_vol_pct"].mean()
        rr = grp["range_ratio_5d_20d"].mean()
        ret = grp["ret_20d_pct"].mean()
        print(f"  {half}: n={len(grp):>2}  P&L=${pnl:>9,.0f}  WR={wr:>4.0f}%  "
              f"vol={vol:.3f}%  range_ratio={rr:.2f}  ret20d={ret:+.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--correlate", action="store_true",
                        help="Print correlations (run after backfill)")
    args = parser.parse_args()

    if args.correlate:
        correlate()
    else:
        backfill(dry_run=args.dry_run)
        correlate()
