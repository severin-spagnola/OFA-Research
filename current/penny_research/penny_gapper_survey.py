"""
Penny Gapper Bird's-Eye Survey
===============================
Multi-phase pipeline to determine whether penny gapper features predict
profitability, move size, or reversion strength.

Phase 1: Scan gappers via Polygon grouped daily (1 call/day, all tickers)
Phase 2: Enrich with ticker details (sector, market cap, float) via Polygon reference
Phase 3: Pull 1m bars for intraday features (continuation, reversion, ramp shape)
Phase 4: Correlate everything and output analysis

Usage:
    python penny_gapper_survey.py                    # full pipeline
    python penny_gapper_survey.py --phase 1          # just scan gappers
    python penny_gapper_survey.py --phase 2          # enrich (needs phase 1 output)
    python penny_gapper_survey.py --phase 3          # 1m bars (needs phase 2 output)
    python penny_gapper_survey.py --phase 4          # analyze (needs phase 3 output)
    python penny_gapper_survey.py --months 18        # scan 18 months of history
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as tm
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Try to load .env from Strat (where the Polygon key lives) and local
for env_path in [
    Path(__file__).parent / ".env",
    Path.home() / "Desktop" / "Strat" / ".env",
    Path.home() / "Desktop" / "OFA-Research" / ".env",
]:
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # Manual .env parse
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

POLYGON_KEY = os.getenv("POLYGON_API_KY") or os.getenv("POLYGON_API_KEY") or ""
POLYGON_BASE = "https://api.polygon.io"

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
BAR_CACHE_DIR = DATA_DIR / "bar_cache"
OUT_DIR = SCRIPT_DIR / "outputs"

_session = requests.Session()


# =====================================================================
# UTILITIES
# =====================================================================

def get_trading_days(start: str, end: str) -> list[str]:
    days = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def is_valid_ticker(ticker: str) -> bool:
    if not ticker or len(ticker) > 6:
        return False
    if not ticker[0].isalpha():
        return False
    if any(c in ticker for c in [".", "/", "$", " "]):
        return False
    return True


def polygon_get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    params = params or {}
    params["apiKey"] = POLYGON_KEY
    for attempt in range(retries):
        try:
            r = _session.get(f"{POLYGON_BASE}{url}", params=params, timeout=30)
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"    Rate limited, waiting {wait}s...")
                tm.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            print(f"    HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"    Request error: {e}")
            tm.sleep(1)
    return {}


# =====================================================================
# PHASE 1: SCAN GAPPERS
# =====================================================================

def phase1_scan_gappers(months: int = 12, min_gap: float = 30.0,
                        max_price: float = 10.0, min_price: float = 0.005,
                        min_volume: int = 50_000) -> pd.DataFrame:
    """Scan Polygon grouped daily for penny gappers."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: SCAN GAPPERS")
    print(f"{'='*70}")

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=months * 30)
    end_str = end_date.strftime("%Y-%m-%d")
    start_str = start_date.strftime("%Y-%m-%d")

    print(f"  Period: {start_str} to {end_str} ({months} months)")
    print(f"  Filters: gap >= {min_gap}%, price ${min_price}-${max_price}, vol >= {min_volume:,}")

    trading_days = get_trading_days(start_str, end_str)
    print(f"  Trading days: {len(trading_days)}")

    prev_day_data: dict[str, dict] = {}  # ticker -> {close, volume}
    all_gappers = []
    days_processed = 0

    for day in trading_days:
        data = polygon_get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{day}",
            {"adjusted": "true", "include_otc": "true"},
        )

        bars = data.get("results", [])
        if not bars:
            days_processed += 1
            continue

        today_data: dict[str, dict] = {}
        day_gappers = []

        for bar in bars:
            ticker = bar.get("T")
            if not is_valid_ticker(ticker):
                continue

            o = bar.get("o")
            h = bar.get("h")
            l = bar.get("l")
            c = bar.get("c")
            v = bar.get("v", 0) or 0
            vw = bar.get("vw")  # volume-weighted avg price
            n = bar.get("n", 0) or 0  # number of transactions

            if o is None or c is None:
                continue

            today_data[ticker] = {"close": c, "volume": v}

            if ticker in prev_day_data:
                prev = prev_day_data[ticker]
                prev_close = prev["close"]
                prev_vol = prev["volume"]

                if prev_close <= 0 or prev_close < min_price or prev_close > max_price:
                    continue
                if o < min_price:
                    continue
                if v < min_volume:
                    continue

                gap_pct = ((o - prev_close) / prev_close) * 100

                if abs(gap_pct) >= min_gap:
                    day_return = ((c - o) / o) * 100 if o > 0 else 0
                    day_range = ((h - l) / o) * 100 if o > 0 and h and l else 0
                    max_runup = ((h - o) / o) * 100 if o > 0 and h else 0
                    max_drawdown = ((l - o) / o) * 100 if o > 0 and l else 0

                    # Gap fill: did price retrace to prev close?
                    if gap_pct > 0:
                        gap_filled = 1 if (l is not None and l <= prev_close) else 0
                    else:
                        gap_filled = 1 if (h is not None and h >= prev_close) else 0

                    day_gappers.append({
                        "date": day,
                        "ticker": ticker,
                        "prev_close": round(prev_close, 4),
                        "open": round(o, 4),
                        "high": round(h, 4) if h else o,
                        "low": round(l, 4) if l else o,
                        "close": round(c, 4),
                        "volume": int(v),
                        "prev_volume": int(prev_vol),
                        "vwap": round(vw, 4) if vw else None,
                        "n_transactions": int(n),
                        "gap_pct": round(gap_pct, 2),
                        "gap_direction": "up" if gap_pct > 0 else "down",
                        "day_return_pct": round(day_return, 2),
                        "day_range_pct": round(day_range, 2),
                        "max_runup_pct": round(max_runup, 2),
                        "max_drawdown_pct": round(max_drawdown, 2),
                        "gap_filled": gap_filled,
                        "vol_ratio": round(v / max(prev_vol, 1), 2),
                        "close_vs_open": "green" if c > o else "red" if c < o else "doji",
                    })

        all_gappers.extend(day_gappers)
        prev_day_data = today_data
        days_processed += 1

        if days_processed % 20 == 0:
            print(f"  {day}: day {days_processed}/{len(trading_days)}, "
                  f"{len(day_gappers)} gappers today, {len(all_gappers)} total")

    if not all_gappers:
        print("  No gappers found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_gappers)
    df = df.sort_values(["date", "gap_pct"], ascending=[True, False]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "phase1_gappers.parquet"
    df.to_parquet(path, index=False)
    print(f"\n  Saved {len(df)} gapper events ({df['ticker'].nunique()} unique tickers) -> {path}")
    return df


# =====================================================================
# PHASE 2: ENRICH WITH TICKER DETAILS
# =====================================================================

def phase2_enrich_tickers(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add sector, market cap, float, type from Polygon reference API."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: ENRICH TICKER DETAILS")
    print(f"{'='*70}")

    if df is None:
        path = DATA_DIR / "phase1_gappers.parquet"
        if not path.exists():
            print("  ERROR: Run phase 1 first")
            return pd.DataFrame()
        df = pd.read_parquet(path)

    tickers = sorted(df["ticker"].unique())
    print(f"  Enriching {len(tickers)} unique tickers...")

    # Check for cached ticker details
    cache_path = DATA_DIR / "ticker_details_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            ticker_cache = json.load(f)
        print(f"  Loaded {len(ticker_cache)} cached ticker details")
    else:
        ticker_cache = {}

    new_lookups = 0
    for i, ticker in enumerate(tickers):
        if ticker in ticker_cache:
            continue

        data = polygon_get(f"/v3/reference/tickers/{ticker}")
        results = data.get("results", {})

        if results:
            ticker_cache[ticker] = {
                "name": results.get("name", ""),
                "market": results.get("market", ""),
                "locale": results.get("locale", ""),
                "type": results.get("type", ""),
                "currency": results.get("currency_name", ""),
                "primary_exchange": results.get("primary_exchange", ""),
                "sic_code": results.get("sic_code", ""),
                "sic_description": results.get("sic_description", ""),
                "market_cap": results.get("market_cap"),
                "share_class_shares_outstanding": results.get("share_class_shares_outstanding"),
                "weighted_shares_outstanding": results.get("weighted_shares_outstanding"),
                "round_lot": results.get("round_lot"),
            }
        else:
            ticker_cache[ticker] = {"name": "", "type": "unknown"}

        new_lookups += 1
        if new_lookups % 50 == 0:
            print(f"    {new_lookups}/{len(tickers) - (len(ticker_cache) - new_lookups)} new lookups...")
            # Save cache periodically
            with open(cache_path, "w") as f:
                json.dump(ticker_cache, f)

    # Final cache save
    if new_lookups > 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(ticker_cache, f)
        print(f"  Completed {new_lookups} new ticker lookups")

    # Merge into dataframe
    detail_rows = []
    for _, row in df.iterrows():
        details = ticker_cache.get(row["ticker"], {})
        detail_rows.append({
            "sector": details.get("sic_description", ""),
            "sic_code": details.get("sic_code", ""),
            "market_cap": details.get("market_cap"),
            "shares_outstanding": details.get("share_class_shares_outstanding")
                or details.get("weighted_shares_outstanding"),
            "primary_exchange": details.get("primary_exchange", ""),
            "ticker_type": details.get("type", ""),
        })

    detail_df = pd.DataFrame(detail_rows)
    df = pd.concat([df.reset_index(drop=True), detail_df], axis=1)

    # Compute float-derived features where possible
    df["float_millions"] = df["shares_outstanding"].apply(
        lambda x: round(x / 1e6, 2) if pd.notna(x) and x > 0 else None
    )
    df["mcap_millions"] = df["market_cap"].apply(
        lambda x: round(x / 1e6, 2) if pd.notna(x) and x > 0 else None
    )
    # Dollar volume
    df["dollar_volume"] = (df["volume"] * df["vwap"]).where(df["vwap"].notna(),
                           df["volume"] * (df["open"] + df["close"]) / 2)
    df["dollar_volume_millions"] = (df["dollar_volume"] / 1e6).round(2)

    # Turnover ratio (volume / float)
    df["turnover_ratio"] = (df["volume"] / df["shares_outstanding"]).where(
        df["shares_outstanding"].notna() & (df["shares_outstanding"] > 0)
    ).round(2)

    path = DATA_DIR / "phase2_enriched.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved enriched data -> {path}")
    return df


# =====================================================================
# PHASE 3: 1-MINUTE INTRADAY FEATURES
# =====================================================================

def fetch_1m_bars(ticker: str, date: str) -> pd.DataFrame:
    """Fetch 1m bars from Polygon, with local cache."""
    BAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = BAR_CACHE_DIR / f"{ticker}_{date}.parquet"

    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    data = polygon_get(
        f"/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )

    results = data.get("results", [])
    if not results:
        pd.DataFrame().to_parquet(cache_file, index=False)
        return pd.DataFrame()

    rows = []
    for bar in results:
        ts = pd.Timestamp(bar["t"], unit="ms", tz="US/Eastern")
        rows.append({
            "timestamp": ts,
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar.get("v", 0),
            "vwap": bar.get("vw"),
            "n_trades": bar.get("n", 0),
        })

    bars_df = pd.DataFrame(rows)
    bars_df.to_parquet(cache_file, index=False)
    return bars_df


def compute_intraday_features(bars: pd.DataFrame, open_price: float,
                               prev_close: float) -> dict:
    """Compute intraday features from 1m bars."""
    if bars.empty or len(bars) < 5:
        return {"intraday_error": "insufficient_bars"}

    # Filter to RTH
    rth_start = pd.Timestamp("09:30").time()
    rth_end = pd.Timestamp("16:00").time()
    rth = bars[
        (bars["timestamp"].dt.time >= rth_start) &
        (bars["timestamp"].dt.time < rth_end)
    ].copy()

    if len(rth) < 5:
        return {"intraday_error": "insufficient_rth_bars"}

    opens = rth["open"].values
    highs = rth["high"].values
    lows = rth["low"].values
    closes = rth["close"].values
    volumes = rth["volume"].values

    # First N minute windows
    n_bars = len(rth)
    first_5m = min(5, n_bars)
    first_15m = min(15, n_bars)
    first_30m = min(30, n_bars)
    first_60m = min(60, n_bars)

    features = {}

    # --- Opening action ---
    features["first_5m_ret"] = round(((closes[first_5m-1] - opens[0]) / opens[0]) * 100, 2)
    features["first_15m_ret"] = round(((closes[first_15m-1] - opens[0]) / opens[0]) * 100, 2)
    features["first_30m_ret"] = round(((closes[first_30m-1] - opens[0]) / opens[0]) * 100, 2)
    features["first_60m_ret"] = round(((closes[first_60m-1] - opens[0]) / opens[0]) * 100, 2)

    # --- Max move from open ---
    cum_high = np.maximum.accumulate(highs)
    cum_low = np.minimum.accumulate(lows)
    features["max_runup_from_open"] = round(((cum_high[-1] - opens[0]) / opens[0]) * 100, 2)
    features["max_dd_from_open"] = round(((cum_low[-1] - opens[0]) / opens[0]) * 100, 2)

    # Time to max high
    max_high_idx = np.argmax(highs)
    features["time_to_hod_bars"] = int(max_high_idx)
    features["time_to_hod_pct"] = round(max_high_idx / max(n_bars, 1) * 100, 1)

    # --- Reversion from high ---
    hod = highs[max_high_idx]
    close_price = closes[-1]
    features["reversion_from_hod_pct"] = round(((close_price - hod) / hod) * 100, 2) if hod > 0 else 0

    # --- Gap fill analysis ---
    gap_size = open_price - prev_close
    if gap_size > 0:  # gap up
        features["gap_fill_pct"] = round(
            min(max((open_price - cum_low[-1]) / gap_size * 100, 0), 100), 1
        ) if gap_size > 0 else 0
    else:  # gap down
        features["gap_fill_pct"] = round(
            min(max((cum_high[-1] - open_price) / abs(gap_size) * 100, 0), 100), 1
        ) if gap_size < 0 else 0

    # --- Volume profile ---
    total_vol = volumes.sum()
    if total_vol > 0:
        first_30m_vol_pct = round(volumes[:first_30m].sum() / total_vol * 100, 1)
        first_60m_vol_pct = round(volumes[:first_60m].sum() / total_vol * 100, 1)
    else:
        first_30m_vol_pct = 0
        first_60m_vol_pct = 0
    features["first_30m_vol_pct"] = first_30m_vol_pct
    features["first_60m_vol_pct"] = first_60m_vol_pct

    # Volume decay rate: ratio of last hour vol to first hour vol
    last_60m = max(n_bars - 60, 0)
    first_hr_vol = volumes[:first_60m].sum()
    last_hr_vol = volumes[last_60m:].sum()
    features["vol_decay_ratio"] = round(last_hr_vol / max(first_hr_vol, 1), 3)

    # --- Price action shape ---
    # How much of the day's range happened in first 30 min
    day_range = cum_high[-1] - cum_low[-1]
    first_30m_range = highs[:first_30m].max() - lows[:first_30m].min()
    features["first_30m_range_pct"] = round(
        first_30m_range / max(day_range, 0.001) * 100, 1
    )

    # Close relative to day range (0 = closed at low, 100 = closed at high)
    features["close_in_range_pct"] = round(
        (close_price - cum_low[-1]) / max(day_range, 0.001) * 100, 1
    )

    # Number of green vs red bars
    green_bars = sum(1 for i in range(n_bars) if closes[i] > opens[i])
    features["green_bar_pct"] = round(green_bars / max(n_bars, 1) * 100, 1)

    # --- Ramp detection (first 3 consecutive green bars after open) ---
    ramp_len = 0
    ramp_body_total = 0
    for i in range(min(10, n_bars)):
        if closes[i] > opens[i]:
            ramp_len += 1
            ramp_body_total += closes[i] - opens[i]
        else:
            break
    features["opening_ramp_bars"] = ramp_len
    features["opening_ramp_body_pct"] = round(
        ramp_body_total / max(opens[0], 0.001) * 100, 2
    ) if ramp_len > 0 else 0

    # Spread proxy: avg high-low per bar in first 30 min
    avg_bar_range = np.mean(highs[:first_30m] - lows[:first_30m])
    features["avg_bar_range_pct"] = round(avg_bar_range / max(opens[0], 0.001) * 100, 3)

    return features


def phase3_intraday_features(df: pd.DataFrame | None = None,
                              max_tickers: int = 0) -> pd.DataFrame:
    """Pull 1m bars and compute intraday features for each gapper event."""
    print(f"\n{'='*70}")
    print(f"PHASE 3: INTRADAY FEATURES (1m bars)")
    print(f"{'='*70}")

    if df is None:
        path = DATA_DIR / "phase2_enriched.parquet"
        if not path.exists():
            print("  ERROR: Run phase 2 first")
            return pd.DataFrame()
        df = pd.read_parquet(path)

    print(f"  Processing {len(df)} gapper events...")

    all_features = []
    fetched = 0
    cached = 0
    errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        ticker = row["ticker"]
        date = row["date"]

        cache_file = BAR_CACHE_DIR / f"{ticker}_{date}.parquet"
        is_cached = cache_file.exists()

        bars = fetch_1m_bars(ticker, date)

        if is_cached:
            cached += 1
        else:
            fetched += 1

        if bars.empty:
            all_features.append({"intraday_error": "no_bars"})
            errors += 1
        else:
            features = compute_intraday_features(
                bars, row["open"], row["prev_close"]
            )
            all_features.append(features)
            if "intraday_error" in features:
                errors += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(df)} done ({fetched} fetched, {cached} cached, {errors} errors)")

        if max_tickers > 0 and fetched >= max_tickers:
            print(f"    Stopping at {max_tickers} new fetches (use --max-tickers 0 for all)")
            # Fill remaining with empty
            for _ in range(len(df) - i - 1):
                all_features.append({"intraday_error": "skipped"})
            break

    features_df = pd.DataFrame(all_features)
    df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    path = DATA_DIR / "phase3_full.parquet"
    df.to_parquet(path, index=False)
    print(f"\n  Saved {len(df)} events with intraday features -> {path}")
    print(f"  Fetched: {fetched}, Cached: {cached}, Errors: {errors}")
    return df


# =====================================================================
# PHASE 4: ANALYSIS & CORRELATIONS
# =====================================================================

def phase4_analyze(df: pd.DataFrame | None = None):
    """Correlate features with profitability/move size/reversion."""
    print(f"\n{'='*70}")
    print(f"PHASE 4: ANALYSIS & CORRELATIONS")
    print(f"{'='*70}")

    if df is None:
        path = DATA_DIR / "phase3_full.parquet"
        if not path.exists():
            path = DATA_DIR / "phase2_enriched.parquet"
        if not path.exists():
            path = DATA_DIR / "phase1_gappers.parquet"
        if not path.exists():
            print("  ERROR: No data found, run earlier phases first")
            return
        df = pd.read_parquet(path)

    # Filter to gap-ups only for cleaner analysis (gap-downs are different dynamics)
    gap_up = df[df["gap_pct"] > 0].copy()
    gap_down = df[df["gap_pct"] < 0].copy()

    print(f"\n  Total events: {len(df)}")
    print(f"  Gap ups: {len(gap_up)}, Gap downs: {len(gap_down)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 1. Gap size vs day return ────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  GAP SIZE BUCKETS (gap-ups only)")
    print(f"  {'─'*60}")
    print(f"  {'Bucket':<15} {'N':>6} {'AvgRet':>8} {'MedRet':>8} {'WR%':>6} "
          f"{'AvgRunup':>9} {'AvgDD':>8} {'GapFill%':>9}")

    buckets = [(30, 50), (50, 75), (75, 100), (100, 150), (150, 200),
               (200, 300), (300, 500), (500, 1000), (1000, 99999)]
    bucket_stats = []
    for lo, hi in buckets:
        subset = gap_up[(gap_up["gap_pct"] >= lo) & (gap_up["gap_pct"] < hi)]
        if len(subset) < 3:
            continue
        label = f"{lo}-{hi}%" if hi < 99999 else f"{lo}%+"
        avg_ret = subset["day_return_pct"].mean()
        med_ret = subset["day_return_pct"].median()
        wr = (subset["day_return_pct"] > 0).mean() * 100
        avg_runup = subset["max_runup_pct"].mean()
        avg_dd = subset["max_drawdown_pct"].mean()
        gap_fill = subset["gap_filled"].mean() * 100

        print(f"  {label:<15} {len(subset):>6} {avg_ret:>+7.1f}% {med_ret:>+7.1f}% "
              f"{wr:>5.0f}% {avg_runup:>+8.1f}% {avg_dd:>+7.1f}% {gap_fill:>8.0f}%")
        bucket_stats.append({
            "bucket": label, "n": len(subset), "avg_ret": avg_ret,
            "med_ret": med_ret, "wr": wr, "avg_runup": avg_runup,
            "avg_dd": avg_dd, "gap_fill_rate": gap_fill,
        })

    # ─── 2. Volume features ──────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  VOLUME RATIO BUCKETS (day vol / prev day vol)")
    print(f"  {'─'*60}")
    if "vol_ratio" in gap_up.columns:
        vol_buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 99999)]
        print(f"  {'VolRatio':<12} {'N':>6} {'AvgRet':>8} {'WR%':>6} {'AvgRunup':>9}")
        for lo, hi in vol_buckets:
            subset = gap_up[(gap_up["vol_ratio"] >= lo) & (gap_up["vol_ratio"] < hi)]
            if len(subset) < 3:
                continue
            label = f"{lo}-{hi}x" if hi < 99999 else f"{lo}x+"
            print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+7.1f}% "
                  f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}% "
                  f"{subset['max_runup_pct'].mean():>+8.1f}%")

    # ─── 3. Market cap buckets ────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  MARKET CAP BUCKETS")
    print(f"  {'─'*60}")
    if "mcap_millions" in gap_up.columns:
        has_mcap = gap_up[gap_up["mcap_millions"].notna()].copy()
        print(f"  ({len(has_mcap)}/{len(gap_up)} events have market cap data)")
        if len(has_mcap) > 10:
            mcap_buckets = [(0, 1), (1, 5), (5, 10), (10, 50), (50, 100),
                            (100, 500), (500, 99999)]
            print(f"  {'MCap($M)':<12} {'N':>6} {'AvgRet':>8} {'WR%':>6} {'AvgGap':>8} {'AvgRunup':>9}")
            for lo, hi in mcap_buckets:
                subset = has_mcap[(has_mcap["mcap_millions"] >= lo) & (has_mcap["mcap_millions"] < hi)]
                if len(subset) < 3:
                    continue
                label = f"${lo}-{hi}M" if hi < 99999 else f"${lo}M+"
                print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+7.1f}% "
                      f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}% "
                      f"{subset['gap_pct'].mean():>+7.1f}% "
                      f"{subset['max_runup_pct'].mean():>+8.1f}%")

    # ─── 4. Sector breakdown ─────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  SECTOR BREAKDOWN (top 15 by count)")
    print(f"  {'─'*60}")
    if "sector" in gap_up.columns:
        has_sector = gap_up[gap_up["sector"].notna() & (gap_up["sector"] != "")].copy()
        if len(has_sector) > 0:
            sector_stats = has_sector.groupby("sector").agg(
                n=("day_return_pct", "count"),
                avg_ret=("day_return_pct", "mean"),
                avg_gap=("gap_pct", "mean"),
                avg_runup=("max_runup_pct", "mean"),
            ).sort_values("n", ascending=False).head(15)
            print(f"  {'Sector':<45} {'N':>5} {'AvgRet':>8} {'AvgGap':>8} {'AvgRunup':>9}")
            for sector, row in sector_stats.iterrows():
                label = str(sector)[:44]
                print(f"  {label:<45} {row['n']:>5.0f} {row['avg_ret']:>+7.1f}% "
                      f"{row['avg_gap']:>+7.1f}% {row['avg_runup']:>+8.1f}%")

    # ─── 5. Turnover ratio ───────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  TURNOVER RATIO (vol / shares outstanding)")
    print(f"  {'─'*60}")
    if "turnover_ratio" in gap_up.columns:
        has_turn = gap_up[gap_up["turnover_ratio"].notna()].copy()
        if len(has_turn) > 10:
            turn_buckets = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0),
                            (2.0, 5.0), (5.0, 99999)]
            print(f"  {'Turnover':<12} {'N':>6} {'AvgRet':>8} {'WR%':>6} {'AvgRunup':>9}")
            for lo, hi in turn_buckets:
                subset = has_turn[(has_turn["turnover_ratio"] >= lo) & (has_turn["turnover_ratio"] < hi)]
                if len(subset) < 3:
                    continue
                label = f"{lo}-{hi}x" if hi < 99999 else f"{lo}x+"
                print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+7.1f}% "
                      f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}% "
                      f"{subset['max_runup_pct'].mean():>+8.1f}%")

    # ─── 6. Intraday features (if available) ─────────────────────
    has_intraday = "first_5m_ret" in df.columns
    if has_intraday:
        valid = gap_up[gap_up["first_5m_ret"].notna()].copy()
        print(f"\n  {'─'*60}")
        print(f"  INTRADAY FEATURES ({len(valid)} events with 1m data)")
        print(f"  {'─'*60}")

        if len(valid) > 20:
            # First 5 min momentum as predictor
            print(f"\n  First 5-min return vs day outcome:")
            f5_buckets = [(-999, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 999)]
            print(f"  {'First5m':<12} {'N':>6} {'AvgDayRet':>10} {'WR%':>6}")
            for lo, hi in f5_buckets:
                subset = valid[(valid["first_5m_ret"] >= lo) & (valid["first_5m_ret"] < hi)]
                if len(subset) < 3:
                    continue
                label = f"{lo:+d} to {hi:+d}%" if hi < 999 else f"{lo:+d}%+"
                if lo <= -999:
                    label = f"<{hi:+d}%"
                print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+9.1f}% "
                      f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}%")

            # Opening ramp strength
            print(f"\n  Opening ramp (consecutive green bars from open):")
            ramp_buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
            print(f"  {'RampBars':<12} {'N':>6} {'AvgDayRet':>10} {'WR%':>6} {'AvgRunup':>9}")
            for lo, hi in ramp_buckets:
                subset = valid[(valid["opening_ramp_bars"] >= lo) & (valid["opening_ramp_bars"] < hi)]
                if len(subset) < 3:
                    continue
                label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
                print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+9.1f}% "
                      f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}% "
                      f"{subset['max_runup_pct'].mean():>+8.1f}%")

            # Time to HOD
            print(f"\n  Time to HOD (% of day):")
            hod_buckets = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
            print(f"  {'HOD%':<12} {'N':>6} {'AvgDayRet':>10} {'AvgReversion':>13}")
            for lo, hi in hod_buckets:
                subset = valid[(valid["time_to_hod_pct"] >= lo) & (valid["time_to_hod_pct"] < hi)]
                if len(subset) < 3:
                    continue
                label = f"{lo}-{hi}%"
                print(f"  {label:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+9.1f}% "
                      f"{subset['reversion_from_hod_pct'].mean():>+12.1f}%")

    # ─── 7. Feature correlations ─────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  FEATURE CORRELATIONS WITH DAY RETURN")
    print(f"  {'─'*60}")

    numeric_cols = gap_up.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = ["day_return_pct", "max_runup_pct", "max_drawdown_pct"]
    feature_cols = [c for c in numeric_cols if c not in target_cols
                    and c not in ["intraday_error"]
                    and gap_up[c].notna().sum() > 20]

    if feature_cols:
        for target in target_cols:
            if target not in gap_up.columns:
                continue
            print(f"\n  Correlations with {target}:")
            corrs = []
            for feat in feature_cols:
                valid = gap_up[[feat, target]].dropna()
                if len(valid) < 20:
                    continue
                corr = valid[feat].corr(valid[target])
                if pd.notna(corr):
                    corrs.append((abs(corr), feat, corr))

            corrs.sort(reverse=True)
            print(f"  {'Feature':<30} {'Corr':>8}")
            for _, feat, corr in corrs[:20]:
                bar = "+" * int(abs(corr) * 40) if corr > 0 else "-" * int(abs(corr) * 40)
                print(f"  {feat:<30} {corr:>+7.4f}  {bar}")

    # ─── 8. Day-of-week & month effects ──────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  DAY-OF-WEEK & MONTH EFFECTS")
    print(f"  {'─'*60}")
    gap_up_copy = gap_up.copy()
    gap_up_copy["dow"] = pd.to_datetime(gap_up_copy["date"]).dt.day_name()
    gap_up_copy["month"] = pd.to_datetime(gap_up_copy["date"]).dt.month

    print(f"  {'Day':<12} {'N':>6} {'AvgRet':>8} {'WR%':>6}")
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        subset = gap_up_copy[gap_up_copy["dow"] == dow]
        if len(subset) > 0:
            print(f"  {dow:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+7.1f}% "
                  f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}%")

    print()
    print(f"  {'Month':<12} {'N':>6} {'AvgRet':>8} {'WR%':>6}")
    for m in range(1, 13):
        subset = gap_up_copy[gap_up_copy["month"] == m]
        if len(subset) > 0:
            print(f"  {m:<12} {len(subset):>6} {subset['day_return_pct'].mean():>+7.1f}% "
                  f"{(subset['day_return_pct'] > 0).mean() * 100:>5.0f}%")

    # ─── 9. Exchange breakdown ────────────────────────────────────
    if "primary_exchange" in gap_up.columns:
        print(f"\n  {'─'*60}")
        print(f"  EXCHANGE BREAKDOWN")
        print(f"  {'─'*60}")
        has_exch = gap_up[gap_up["primary_exchange"].notna() & (gap_up["primary_exchange"] != "")].copy()
        if len(has_exch) > 0:
            exch_stats = has_exch.groupby("primary_exchange").agg(
                n=("day_return_pct", "count"),
                avg_ret=("day_return_pct", "mean"),
                wr=("day_return_pct", lambda x: (x > 0).mean() * 100),
                avg_gap=("gap_pct", "mean"),
            ).sort_values("n", ascending=False).head(10)
            print(f"  {'Exchange':<20} {'N':>6} {'AvgRet':>8} {'WR%':>6} {'AvgGap':>8}")
            for exch, row in exch_stats.iterrows():
                print(f"  {str(exch):<20} {row['n']:>6.0f} {row['avg_ret']:>+7.1f}% "
                      f"{row['wr']:>5.0f}% {row['avg_gap']:>+7.1f}%")

    # ─── 10. Repeat gappers ──────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  REPEAT GAPPERS (same ticker multiple gaps)")
    print(f"  {'─'*60}")
    freq = gap_up["ticker"].value_counts()
    repeaters = freq[freq >= 3]
    if len(repeaters) > 0:
        print(f"  {len(repeaters)} tickers gapped 3+ times")
        repeater_data = gap_up[gap_up["ticker"].isin(repeaters.index)]
        single_data = gap_up[~gap_up["ticker"].isin(repeaters.index)]
        print(f"  Repeaters: n={len(repeater_data)}, avg ret={repeater_data['day_return_pct'].mean():+.1f}%, "
              f"WR={( repeater_data['day_return_pct'] > 0).mean()*100:.0f}%")
        print(f"  One-offs:  n={len(single_data)}, avg ret={single_data['day_return_pct'].mean():+.1f}%, "
              f"WR={(single_data['day_return_pct'] > 0).mean()*100:.0f}%")

    # ─── Save full analysis CSV ──────────────────────────────────
    csv_path = OUT_DIR / "full_survey.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved full survey CSV -> {csv_path}")

    # ─── VERDICT ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"VERDICT: IS THERE SIGNAL?")
    print(f"{'='*70}")

    if feature_cols and "day_return_pct" in gap_up.columns:
        strong_corrs = []
        for feat in feature_cols:
            valid = gap_up[[feat, "day_return_pct"]].dropna()
            if len(valid) < 20:
                continue
            corr = valid[feat].corr(valid["day_return_pct"])
            if pd.notna(corr) and abs(corr) > 0.10:
                strong_corrs.append((feat, corr))

        if strong_corrs:
            print(f"  Features with |corr| > 0.10 with day return:")
            for feat, corr in sorted(strong_corrs, key=lambda x: -abs(x[1])):
                print(f"    {feat}: {corr:+.3f}")
            print(f"\n  TENTATIVE: Some features show weak-to-moderate correlation.")
            print(f"  Worth exploring as OFA regime features.")
        else:
            print(f"  No features with |corr| > 0.10 found.")
            print(f"  Penny gapper day return appears largely random given these features.")

    print()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Penny Gapper Bird's-Eye Survey")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1-4), 0=all")
    parser.add_argument("--months", type=int, default=12, help="Months of history (default: 12)")
    parser.add_argument("--min-gap", type=float, default=30.0, help="Min gap %% (default: 30)")
    parser.add_argument("--max-price", type=float, default=10.0, help="Max prev close (default: 10)")
    parser.add_argument("--min-volume", type=int, default=50000, help="Min day volume (default: 50000)")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Max new 1m bar fetches in phase 3 (0=unlimited)")
    args = parser.parse_args()

    if not POLYGON_KEY:
        print("ERROR: No Polygon API key found. Set POLYGON_API_KY or POLYGON_API_KEY in .env")
        sys.exit(1)

    print(f"Polygon key: ...{POLYGON_KEY[-4:]}")

    df = None
    phases = [args.phase] if args.phase > 0 else [1, 2, 3, 4]

    if 1 in phases:
        df = phase1_scan_gappers(
            months=args.months, min_gap=args.min_gap,
            max_price=args.max_price, min_volume=args.min_volume,
        )

    if 2 in phases:
        df = phase2_enrich_tickers(df)

    if 3 in phases:
        df = phase3_intraday_features(df, max_tickers=args.max_tickers)

    if 4 in phases:
        phase4_analyze(df)


if __name__ == "__main__":
    main()
