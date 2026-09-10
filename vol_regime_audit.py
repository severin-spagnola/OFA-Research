"""
VIX Term Structure Regime Filter — Research Audit
Thesis: VIX > VIX3M (inverted term structure) as a mean-revert regime filter.
In-sample: 2019-2023. OOS: 2024+. DO NOT tune on OOS.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

OUTPUT_FILE = "/Users/severinspagnola/Desktop/OFA-Research/vol_regime_audit.txt"

# ── output router ─────────────────────────────────────────────────────────────
lines_buffer = []

def p(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    lines_buffer.append(msg)

def save_output():
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines_buffer))
    print(f"\n[Saved output to {OUTPUT_FILE}]")

# ── STEP 1: DATA AUDIT ───────────────────────────────────────────────────────
p("=" * 70)
p("VIX TERM STRUCTURE REGIME FILTER — RESEARCH AUDIT")
p(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
p("=" * 70)

p("\n" + "=" * 70)
p("STEP 1 — DATA AUDIT")
p("=" * 70)

DATA_DIR = "/Users/severinspagnola/Desktop/OFA-Research/data/"

# Check for local VIX files
vix_files = (
    glob.glob(os.path.join(DATA_DIR, "*VIX*")) +
    glob.glob(os.path.join(DATA_DIR, "*vix*"))
)
if vix_files:
    p(f"Found local VIX files: {vix_files}")
else:
    p("No local VIX files found in data/. Will use yfinance.")

# Pull via yfinance
try:
    import yfinance as yf
    p("\nPulling ^VIX via yfinance (max period)...")
    vix_raw = yf.download("^VIX", period="max", auto_adjust=True, progress=False)
    p(f"  ^VIX rows: {len(vix_raw)}")
    if len(vix_raw) > 0:
        p(f"  ^VIX date range: {vix_raw.index[0].date()} to {vix_raw.index[-1].date()}")
        vix_gaps = vix_raw['Close'].isna().sum()
        p(f"  ^VIX NaN values: {vix_gaps}")
    else:
        p("  ERROR: ^VIX returned empty DataFrame")

    p("\nPulling ^VIX3M via yfinance (max period)...")
    vix3m_raw = yf.download("^VIX3M", period="max", auto_adjust=True, progress=False)
    p(f"  ^VIX3M rows: {len(vix3m_raw)}")
    if len(vix3m_raw) > 0:
        p(f"  ^VIX3M date range: {vix3m_raw.index[0].date()} to {vix3m_raw.index[-1].date()}")
        vix3m_gaps = vix3m_raw['Close'].isna().sum()
        p(f"  ^VIX3M NaN values: {vix3m_gaps}")
        vix3m_available = True
    else:
        p("  WARNING: ^VIX3M returned empty DataFrame. Trying VIXM ETF...")
        vix3m_available = False

    if not vix3m_available or len(vix3m_raw) == 0:
        p("\nPulling VIXM (ETF proxy for VIX3M)...")
        vixm_raw = yf.download("VIXM", period="max", auto_adjust=True, progress=False)
        p(f"  VIXM rows: {len(vixm_raw)}")
        if len(vixm_raw) > 0:
            p(f"  VIXM date range: {vixm_raw.index[0].date()} to {vixm_raw.index[-1].date()}")
            vix3m_raw = vixm_raw
            vix3m_available = True
        else:
            p("  ERROR: Both ^VIX3M and VIXM unavailable. HALTING.")
            save_output()
            sys.exit(1)

    # Pull SPY
    p("\nPulling SPY via yfinance (max period)...")
    spy_raw = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    p(f"  SPY rows: {len(spy_raw)}")
    if len(spy_raw) > 0:
        p(f"  SPY date range: {spy_raw.index[0].date()} to {spy_raw.index[-1].date()}")

except Exception as e:
    p(f"\nFATAL ERROR pulling data via yfinance: {e}")
    save_output()
    sys.exit(1)

# ── Build aligned DataFrame ───────────────────────────────────────────────────
p("\nBuilding aligned DataFrame (VIX, VIX3M, SPY)...")

# Handle MultiIndex columns from yfinance
def extract_close(df, ticker_hint=""):
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find 'Close' level
        close_cols = [c for c in df.columns if 'Close' in c or 'close' in c]
        if close_cols:
            return df[close_cols[0]].squeeze()
        # Fall back to first column
        return df.iloc[:, 0].squeeze()
    elif 'Close' in df.columns:
        return df['Close'].squeeze()
    else:
        return df.iloc[:, 0].squeeze()

vix_s = extract_close(vix_raw, "VIX").rename("VIX")
vix3m_s = extract_close(vix3m_raw, "VIX3M").rename("VIX3M")
spy_s = extract_close(spy_raw, "SPY").rename("SPY")

# Ensure DatetimeIndex
for s in [vix_s, vix3m_s, spy_s]:
    s.index = pd.to_datetime(s.index)

df = pd.DataFrame({"VIX": vix_s, "VIX3M": vix3m_s, "SPY": spy_s})
df = df.dropna(subset=["VIX", "VIX3M"])
p(f"  Aligned rows (VIX+VIX3M both present): {len(df)}")
p(f"  Aligned date range: {df.index[0].date()} to {df.index[-1].date()}")

# SPY returns
df["SPY_ret"] = df["SPY"].pct_change()
df["SPY_next"] = df["SPY_ret"].shift(-1)   # next-day return

# ── STEP 2: TERM STRUCTURE SLOPE ─────────────────────────────────────────────
p("\n" + "=" * 70)
p("STEP 2 — TERM STRUCTURE SLOPE")
p("=" * 70)

df["slope"] = df["VIX"] / df["VIX3M"]
df["inverted"] = df["slope"] > 1.0

total_days = len(df)
inv_days_all = df["inverted"].sum()
pct_inv_all = 100.0 * inv_days_all / total_days
p(f"\nFull history:")
p(f"  Total trading days with both VIX+VIX3M: {total_days}")
p(f"  Inverted days (VIX > VIX3M):            {inv_days_all}")
p(f"  % inverted:                              {pct_inv_all:.2f}%")

# 2019-2023 subset
mask_is = (df.index >= "2019-01-01") & (df.index <= "2023-12-31")
df_is = df[mask_is].copy()
inv_is = df_is["inverted"].sum()
pct_inv_is = 100.0 * inv_is / len(df_is) if len(df_is) > 0 else 0
p(f"\n2019-2023 (in-sample):")
p(f"  Trading days: {len(df_is)}")
p(f"  Inverted days: {inv_is}")
p(f"  % inverted: {pct_inv_is:.2f}%")

# Monthly breakdown of inversion frequency
p("\nMonthly inversion frequency (2019-2023) — inverted days / total days per month:")
p(f"  {'Year-Month':<12} {'Inv Days':>9} {'Total Days':>11} {'Inv %':>8}")
p(f"  {'-'*12} {'-'*9} {'-'*11} {'-'*8}")
df_is["ym"] = df_is.index.to_period("M")
monthly = df_is.groupby("ym")["inverted"].agg(["sum", "count"])
monthly["pct"] = 100.0 * monthly["sum"] / monthly["count"]
for ym, row in monthly.iterrows():
    p(f"  {str(ym):<12} {int(row['sum']):>9} {int(row['count']):>11} {row['pct']:>7.1f}%")

p("\nYearly inversion frequency (2019-2023):")
df_is["year"] = df_is.index.year
yearly = df_is.groupby("year")["inverted"].agg(["sum", "count"])
yearly["pct"] = 100.0 * yearly["sum"] / yearly["count"]
for yr, row in yearly.iterrows():
    p(f"  {yr}: {int(row['sum'])} inverted / {int(row['count'])} days = {row['pct']:.1f}%")

# ── STEP 3: BASE SIGNAL AUDIT (IN-SAMPLE 2019-2023) ──────────────────────────
p("\n" + "=" * 70)
p("STEP 3 — BASE SIGNAL AUDIT (IN-SAMPLE: 2019-2023 ONLY)")
p("=" * 70)

# Drop rows with no next-day return
df_is_clean = df_is.dropna(subset=["SPY_next"]).copy()
p(f"\nIn-sample rows with valid next-day SPY return: {len(df_is_clean)}")

inv_rows = df_is_clean[df_is_clean["inverted"] == True]
norm_rows = df_is_clean[df_is_clean["inverted"] == False]

n_inv = len(inv_rows)
n_norm = len(norm_rows)
p(f"  Inverted days: {n_inv}")
p(f"  Normal days:   {n_norm}")

mean_ret_inv = inv_rows["SPY_next"].mean()
mean_ret_norm = norm_rows["SPY_next"].mean()

p(f"\na) Mean next-day SPY return:")
p(f"   Inverted: {mean_ret_inv*100:.4f}%")
p(f"   Normal:   {mean_ret_norm*100:.4f}%")
p(f"   Edge (inverted - normal): {(mean_ret_inv - mean_ret_norm)*100:.4f}%")

win_inv = (inv_rows["SPY_next"] > 0).sum()
win_rate_inv = win_inv / n_inv if n_inv > 0 else 0
p(f"\nb) Win rate when inverted (next-day SPY > 0): {win_rate_inv*100:.2f}%  ({win_inv}/{n_inv})")

win_norm = (norm_rows["SPY_next"] > 0).sum()
win_rate_norm = win_norm / n_norm if n_norm > 0 else 0
p(f"   Win rate when normal:                     {win_rate_norm*100:.2f}%  ({win_norm}/{n_norm})")

# Binomial test
from scipy.stats import binomtest
btest = binomtest(int(win_inv), n=n_inv, p=0.50, alternative='greater')
p(f"\nc) Binomial test (H0: win_rate = 0.50, alt: greater):")
p(f"   Inverted win count: {win_inv}  n: {n_inv}  win_rate: {win_rate_inv:.4f}")
p(f"   p-value: {btest.pvalue:.6f}")
p(f"   Statistically significant (p < 0.10): {'YES' if btest.pvalue < 0.10 else 'NO'}")
p(f"   Statistically significant (p < 0.05): {'YES' if btest.pvalue < 0.05 else 'NO'}")

p(f"\nd) Sample size n for inverted events (2019-2023): {n_inv}")

# Annualized signal density
years_is = 5  # 2019-2023
events_per_year = n_inv / years_is
events_per_week = events_per_year / 52
p(f"\ne) Annualized signal density:")
p(f"   Inversion events: {n_inv} over {years_is} years")
p(f"   Events/year: {events_per_year:.1f}")
p(f"   Events/week: {events_per_week:.2f}")
p(f"   >= 1 event/week: {'YES' if events_per_week >= 1.0 else 'NO'}")

# ── STEP 4: TRANSACTION COST SURVIVAL ────────────────────────────────────────
p("\n" + "=" * 70)
p("STEP 4 — TRANSACTION COST SURVIVAL (IN-SAMPLE: 2019-2023)")
p("=" * 70)

RT_COST = 0.0005  # 0.05% round-trip

inv_returns = inv_rows["SPY_next"].copy()
inv_returns_net = inv_returns - RT_COST

mean_gross = inv_returns.mean()
mean_net = inv_returns_net.mean()

trading_days_per_year = 252
ann_gross = mean_gross * trading_days_per_year
ann_net = mean_net * trading_days_per_year

p(f"\nEdge per trade (gross): {mean_gross*100:.4f}%  (annualized: {ann_gross*100:.2f}%)")
p(f"Edge per trade (net after {RT_COST*100:.2f}% RT cost): {mean_net*100:.4f}%  (annualized: {ann_net*100:.2f}%)")
p(f"Net edge survives 0.05% RT cost: {'YES' if mean_net > 0 else 'NO'}")

# Sharpe — inverted-days-only strategy
std_inv = inv_returns_net.std()
sharpe_inv = (mean_net / std_inv) * np.sqrt(trading_days_per_year) if std_inv > 0 else np.nan
p(f"\nSharpe — Inverted-days strategy (net, IS 2019-2023): {sharpe_inv:.4f}")

# Buy-and-hold SPY Sharpe (same IS period)
spy_daily = df_is_clean["SPY_ret"].dropna()
spy_mean = spy_daily.mean()
spy_std = spy_daily.std()
sharpe_bh = (spy_mean / spy_std) * np.sqrt(trading_days_per_year) if spy_std > 0 else np.nan
p(f"Sharpe — Buy-and-hold SPY (IS 2019-2023): {sharpe_bh:.4f}")

p(f"\nNote: Inverted-days Sharpe computed only on days when signal fires ({n_inv} days).")
p(f"Buy-and-hold Sharpe computed on all IS days ({len(df_is_clean)} days).")

# ── STEP 5: OOS SNIFF TEST (2024 ONLY) ───────────────────────────────────────
p("\n" + "=" * 70)
p("STEP 5 — OOS SNIFF TEST (2024 ONLY)")
p("*** DO NOT USE FOR PARAMETER TUNING — OUT-OF-SAMPLE ONLY ***")
p("=" * 70)

mask_oos = (df.index >= "2024-01-01") & (df.index <= "2024-12-31")
df_oos = df[mask_oos].dropna(subset=["SPY_next"]).copy()
p(f"\nOOS rows (2024): {len(df_oos)}")

if len(df_oos) == 0:
    p("  No OOS data available.")
else:
    inv_oos = df_oos[df_oos["inverted"] == True]
    norm_oos = df_oos[df_oos["inverted"] == False]
    n_inv_oos = len(inv_oos)
    n_norm_oos = len(norm_oos)

    pct_inv_oos = 100.0 * n_inv_oos / len(df_oos)
    p(f"  Inverted days (OOS): {n_inv_oos} / {len(df_oos)} = {pct_inv_oos:.1f}%")

    if n_inv_oos > 0:
        mean_inv_oos = inv_oos["SPY_next"].mean()
        mean_norm_oos = norm_oos["SPY_next"].mean() if n_norm_oos > 0 else np.nan
        win_inv_oos = (inv_oos["SPY_next"] > 0).sum()
        win_rate_oos = win_inv_oos / n_inv_oos

        p(f"\n  a) Mean next-day SPY return (OOS):")
        p(f"     Inverted: {mean_inv_oos*100:.4f}%")
        p(f"     Normal:   {mean_norm_oos*100:.4f}%")

        p(f"\n  b) Win rate inverted (OOS): {win_rate_oos*100:.2f}%  ({win_inv_oos}/{n_inv_oos})")

        btest_oos = binomtest(int(win_inv_oos), n=n_inv_oos, p=0.50, alternative='greater')
        p(f"\n  c) OOS Binomial test p-value: {btest_oos.pvalue:.6f}")
        p(f"     p < 0.10: {'YES' if btest_oos.pvalue < 0.10 else 'NO'}")

        p(f"\n  d) OOS sample size: {n_inv_oos}")

        events_per_year_oos = n_inv_oos / 1.0
        events_per_week_oos = events_per_year_oos / 52
        p(f"\n  e) OOS signal density: {n_inv_oos} events in 2024")
        p(f"     Events/week (OOS): {events_per_week_oos:.2f}")

        # TC survival OOS
        inv_ret_oos_net = inv_oos["SPY_next"] - RT_COST
        mean_net_oos = inv_ret_oos_net.mean()
        p(f"\n  f) Net edge after costs (OOS): {mean_net_oos*100:.4f}%/trade")
        p(f"     Net edge > 0: {'YES' if mean_net_oos > 0 else 'NO'}")

        std_oos = inv_ret_oos_net.std()
        sharpe_oos = (mean_net_oos / std_oos) * np.sqrt(trading_days_per_year) if std_oos > 0 else np.nan
        p(f"\n  g) OOS Sharpe (inverted-days, net): {sharpe_oos:.4f}")

        spy_oos_all = df_oos["SPY_ret"].dropna()
        sharpe_bh_oos = (spy_oos_all.mean() / spy_oos_all.std()) * np.sqrt(trading_days_per_year)
        p(f"     OOS Buy-and-hold SPY Sharpe: {sharpe_bh_oos:.4f}")
    else:
        p("  No inverted days in OOS period.")

# ── STEP 6: BRUTAL HONESTY SUMMARY ───────────────────────────────────────────
p("\n" + "=" * 70)
p("STEP 6 — BRUTAL HONESTY SUMMARY")
p("=" * 70)

p_lt_10 = btest.pvalue < 0.10
density_ok = events_per_week >= 1.0
edge_survives = mean_net > 0

p(f"\n  Is p-value < 0.10?                         {'YES' if p_lt_10 else 'NO'}  (p={btest.pvalue:.4f})")
p(f"  Is signal density >= 1 event/week?         {'YES' if density_ok else 'NO'}  ({events_per_week:.2f} events/week)")
p(f"  Does net edge survive 0.05% RT cost?       {'YES' if edge_survives else 'NO'}  (net={mean_net*100:.4f}%/trade)")

checks_passed = sum([p_lt_10, density_ok, edge_survives])

p(f"\n  Checks passed: {checks_passed} / 3")

if checks_passed >= 2 and p_lt_10:
    recommendation = "PROCEED (with caution — verify OOS alignment)"
elif checks_passed >= 2:
    recommendation = "CONDITIONAL — address p-value concern before building"
else:
    recommendation = "DROP — insufficient statistical support"

p(f"\n  RECOMMENDATION: {recommendation}")
p("\n" + "=" * 70)
p("END OF AUDIT")
p("=" * 70)

# Save output
save_output()
