"""
Regime Monitor — Kill Switch for Overfit Strategies
====================================================
Monitors live trade results from an NDJSON trade log and detects when the
current strategy regime has died. When triggered, signals that a new
overfit search should be run.

Detection signals:
  1. N consecutive losing days (default: 3)
  2. Drawdown from recent peak exceeds threshold
  3. Win rate over last M trades drops below floor

Usage:
    # Check current status
    python regime_monitor.py --log /path/to/trades.ndjson

    # Watch mode (check every 5 min)
    python regime_monitor.py --log /path/to/trades.ndjson --watch

    # Custom thresholds
    python regime_monitor.py --log trades.ndjson --max-losing-days 2 --dd-pct 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time as tm
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MAX_CONSECUTIVE_LOSING_DAYS = 5
DEFAULT_MAX_DD_DOLLARS = 3500        # max drawdown in absolute dollars
DEFAULT_MIN_WIN_RATE_FLOOR = 50.0    # WR below this over last N trades = alarm
DEFAULT_WIN_RATE_LOOKBACK = 20       # number of recent trades for WR check


# ─── Trade log parsing ────────────────────────────────────────────────────────

def load_trades(log_path: str) -> list[dict]:
    """Load trades from NDJSON file."""
    trades = []
    path = Path(log_path)
    if not path.exists():
        return trades

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Only include actual trades (not events)
                if record.get("strategy") or record.get("pnl_dollars") is not None:
                    trades.append(record)
            except json.JSONDecodeError:
                continue

    return trades


def load_eod_events(log_path: str) -> list[dict]:
    """Load trade_eod events from events NDJSON file."""
    events = []
    path = Path(log_path)
    if not path.exists():
        return events

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("event_type") == "trade_eod":
                    events.append(record)
            except json.JSONDecodeError:
                continue

    return events


# ─── Analysis ─────────────────────────────────────────────────────────────────

def daily_pnl(trades: list[dict]) -> dict[str, float]:
    """Aggregate P&L by date."""
    by_day = defaultdict(float)
    for t in trades:
        d = t.get("date")
        if not d:
            # Try to extract date from timestamp
            ts = t.get("ts", t.get("exit_time", t.get("signal_time", "")))
            if ts:
                d = str(ts)[:10]
        if d:
            pnl = t.get("pnl_dollars", 0) or 0
            by_day[d] += pnl
    return dict(sorted(by_day.items()))


def check_consecutive_losers(daily: dict[str, float],
                              max_days: int) -> tuple[bool, int, list[str]]:
    """Check for N consecutive losing days.

    Returns (triggered, streak_count, losing_dates).
    """
    if not daily:
        return False, 0, []

    dates = list(daily.keys())
    streak = 0
    losing_dates = []

    # Count from most recent backwards
    for d in reversed(dates):
        if daily[d] < 0:
            streak += 1
            losing_dates.insert(0, d)
        else:
            break

    triggered = streak >= max_days
    return triggered, streak, losing_dates


def check_drawdown(daily: dict[str, float],
                    max_dd_dollars: float) -> tuple[bool, float, float]:
    """Check if drawdown from peak exceeds absolute dollar threshold.

    Returns (triggered, current_dd_dollars, peak_equity).
    """
    if not daily:
        return False, 0, 0

    equity = 0
    peak = 0
    max_dd = 0
    for d in sorted(daily.keys()):
        equity += daily[d]
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    return max_dd >= max_dd_dollars, round(max_dd, 2), round(peak, 2)


def check_win_rate(trades: list[dict], lookback: int,
                    floor: float) -> tuple[bool, float, int]:
    """Check if recent win rate is below floor.

    Returns (triggered, recent_wr, n_recent).
    """
    if len(trades) < lookback // 2:  # need at least half the lookback
        return False, 0, 0

    recent = trades[-lookback:]
    wins = sum(1 for t in recent
               if (t.get("pnl_dollars") or 0) > 0
               or t.get("result") == "win"
               or t.get("expired_itm") is True)
    wr = wins / len(recent) * 100

    return wr < floor, round(wr, 2), len(recent)


# ─── Status report ────────────────────────────────────────────────────────────

def regime_status(
    trades: list[dict],
    max_consecutive_losing_days: int = DEFAULT_MAX_CONSECUTIVE_LOSING_DAYS,
    max_dd_dollars: float = DEFAULT_MAX_DD_DOLLARS,
    min_wr_floor: float = DEFAULT_MIN_WIN_RATE_FLOOR,
    wr_lookback: int = DEFAULT_WIN_RATE_LOOKBACK,
    # Legacy alias — ignored, use max_dd_dollars instead
    max_dd_pct: float | None = None,
) -> dict:
    """Full regime health check. Returns status dict."""
    daily = daily_pnl(trades)

    lose_triggered, lose_streak, lose_dates = check_consecutive_losers(
        daily, max_consecutive_losing_days
    )
    dd_triggered, dd_amount, peak = check_drawdown(daily, max_dd_dollars)
    wr_triggered, recent_wr, n_recent = check_win_rate(
        trades, wr_lookback, min_wr_floor
    )

    # Overall verdict
    kill = lose_triggered or dd_triggered or wr_triggered
    reasons = []
    if lose_triggered:
        reasons.append(f"{lose_streak} consecutive losing days")
    if dd_triggered:
        reasons.append(f"drawdown ${dd_amount:.0f} exceeds ${max_dd_dollars:.0f} threshold")
    if wr_triggered:
        reasons.append(f"win rate {recent_wr}% below {min_wr_floor}% floor")

    # Summary stats
    total_pnl = sum(daily.values())
    n_days = len(daily)
    n_winning_days = sum(1 for v in daily.values() if v > 0)
    n_losing_days = sum(1 for v in daily.values() if v < 0)

    return {
        "status": "DEAD" if kill else "ALIVE",
        "kill": kill,
        "reasons": reasons,
        "n_trades": len(trades),
        "n_days": n_days,
        "total_pnl": round(total_pnl, 2),
        "winning_days": n_winning_days,
        "losing_days": n_losing_days,
        "current_losing_streak": lose_streak,
        "dd_from_peak_dollars": dd_amount,
        "peak_equity": peak,
        "recent_win_rate": recent_wr,
        "recent_trades_checked": n_recent,
        "daily_pnl": daily,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def print_status(status: dict):
    """Pretty-print regime status."""
    verdict = status["status"]
    marker = "REGIME DEAD — REGENERATE STRATEGY" if status["kill"] else "REGIME ALIVE"

    print(f"\n{'=' * 60}")
    print(f"  {marker}")
    print(f"{'=' * 60}")

    if status["reasons"]:
        print(f"\n  Kill signals:")
        for r in status["reasons"]:
            print(f"    - {r}")

    print(f"\n  Trades: {status['n_trades']} over {status['n_days']} days")
    print(f"  Total P&L: ${status['total_pnl']:,.2f}")
    print(f"  Days: {status['winning_days']}W / {status['losing_days']}L")
    print(f"  Current losing streak: {status['current_losing_streak']} days")
    dd_key = "dd_from_peak_dollars" if "dd_from_peak_dollars" in status else "dd_from_peak_pct"
    dd_val = status[dd_key]
    print(f"  DD from peak: ${dd_val:,.2f} (peak ${status['peak_equity']:,.2f})")
    print(f"  Recent WR ({status['recent_trades_checked']} trades): {status['recent_win_rate']}%")

    if status["daily_pnl"]:
        print(f"\n  Recent daily P&L:")
        for d, pnl in list(status["daily_pnl"].items())[-10:]:
            marker = "+" if pnl >= 0 else ""
            print(f"    {d}: {marker}${pnl:,.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Regime Monitor — Kill Switch")
    parser.add_argument("--trades", type=str,
                        help="Path to trades.ndjson file")
    parser.add_argument("--events", type=str,
                        help="Path to events.ndjson file (uses trade_eod events)")
    parser.add_argument("--max-losing-days", type=int,
                        default=DEFAULT_MAX_CONSECUTIVE_LOSING_DAYS,
                        help=f"Kill after N consecutive losing days (default: {DEFAULT_MAX_CONSECUTIVE_LOSING_DAYS})")
    parser.add_argument("--dd-dollars", type=float,
                        default=DEFAULT_MAX_DD_DOLLARS,
                        help=f"Kill if DD exceeds this dollar amount (default: {DEFAULT_MAX_DD_DOLLARS})")
    parser.add_argument("--wr-floor", type=float,
                        default=DEFAULT_MIN_WIN_RATE_FLOOR,
                        help=f"Kill if recent WR below this (default: {DEFAULT_MIN_WIN_RATE_FLOOR})")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode — re-check every 5 minutes")
    parser.add_argument("--interval", type=int, default=300,
                        help="Watch interval in seconds (default: 300)")
    args = parser.parse_args()

    if not args.trades and not args.events:
        # Try default paths
        default_trades = Path(__file__).parent.parent / "zero_dte" / "logs" / "trades.ndjson"
        default_events = Path(__file__).parent.parent / "zero_dte" / "logs" / "events.ndjson"
        if default_events.exists():
            args.events = str(default_events)
            print(f"Using default events log: {args.events}")
        elif default_trades.exists():
            args.trades = str(default_trades)
            print(f"Using default trades log: {args.trades}")
        else:
            print("No log file specified and no default found.")
            print("Use --trades or --events to specify a log file.")
            sys.exit(1)

    while True:
        # Load trades
        if args.events:
            trades = load_eod_events(args.events)
        else:
            trades = load_trades(args.trades)

        if not trades:
            print("No trades found in log file.")
            if not args.watch:
                sys.exit(0)
            print(f"Watching... (next check in {args.interval}s)")
            tm.sleep(args.interval)
            continue

        status = regime_status(
            trades,
            max_consecutive_losing_days=args.max_losing_days,
            max_dd_dollars=args.dd_dollars,
            min_wr_floor=args.wr_floor,
        )

        print_status(status)

        if status["kill"]:
            print("ACTION REQUIRED: Run overfit_search.py to generate new strategy")
            if not args.watch:
                sys.exit(1)  # non-zero exit for scripting

        if not args.watch:
            break

        print(f"Watching... (next check in {args.interval}s)")
        tm.sleep(args.interval)


if __name__ == "__main__":
    main()
