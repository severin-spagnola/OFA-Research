"""
XS-Momentum: Sector-Rotation ETF Momentum Backtest
Strategy: 12-1 month Jegadeesh-Titman signal on 11 SPDR ETFs
Long top 3, short bottom 3, monthly rebalance, dollar-neutral
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
START = "2015-01-01"
END = "2025-12-31"
OUTPUT_DIR = "/Users/severinspagnola/Desktop/OFA-Research/state/XS-Momentum"
N_LONG = 3
N_SHORT = 3


# ── 1. Download data ─────────────────────────────────────────────────────────
print("Downloading ETF prices...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

# Extract adjusted close; handle MultiIndex or single-level
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    prices = raw[["Close"]]

prices.index = pd.to_datetime(prices.index)
prices.sort_index(inplace=True)

# Save daily prices
prices.to_csv(f"{OUTPUT_DIR}/etf_prices.csv")
print(f"  Saved daily prices: {prices.shape[0]} rows x {prices.shape[1]} tickers")
print(f"  Coverage: {prices.index[0].date()} to {prices.index[-1].date()}")
missing = prices.isna().sum()
if missing.any():
    print(f"  Missing data (NaN count per ticker):\n{missing[missing > 0]}")


# ── 2. Resample to month-end ─────────────────────────────────────────────────
monthly = prices.resample("ME").last()
print(f"\nMonthly prices: {monthly.shape[0]} months")

# ── 3. Compute 12-1 momentum signal ─────────────────────────────────────────
# Signal at month t = price[t-2] / price[t-13] - 1  (skip t-1)
# i.e. 11-month return from 13 months ago to 2 months ago
def compute_momentum(monthly_prices):
    """Return DataFrame of 12-1 signals, NaN where insufficient history."""
    # shift(2) = price 2 months ago; shift(13) = price 13 months ago
    ret = monthly_prices.shift(2) / monthly_prices.shift(13) - 1
    return ret

signals = compute_momentum(monthly)

# ── 4. Portfolio construction & returns ─────────────────────────────────────
# Next-month return for each ticker
fwd_returns = monthly.pct_change().shift(-1)  # return earned NEXT month

portfolio_returns = []
long_legs = []
short_legs = []
rebalance_dates = []

# Start after 13 months of history: first rebalance at 2016-02-29 (month-end)
start_date = pd.Timestamp("2016-02-01")
end_date = pd.Timestamp("2025-11-30")  # last rebalance where next-month return exists

rebalance_index = monthly.index[(monthly.index >= start_date) & (monthly.index <= end_date)]

for date in rebalance_index:
    sig = signals.loc[date].dropna()
    if len(sig) < (N_LONG + N_SHORT):
        continue  # not enough tickers

    ranked = sig.sort_values(ascending=False)
    longs = ranked.iloc[:N_LONG].index.tolist()
    shorts = ranked.iloc[-N_SHORT:].index.tolist()

    # Next-month returns
    if date not in fwd_returns.index:
        continue
    fwd = fwd_returns.loc[date]

    long_ret = fwd[longs].mean()
    short_ret = fwd[shorts].mean()

    if pd.isna(long_ret) or pd.isna(short_ret):
        continue

    port_ret = long_ret - short_ret

    portfolio_returns.append(port_ret)
    long_legs.append(set(longs))
    short_legs.append(set(shorts))
    rebalance_dates.append(date)

returns_series = pd.Series(portfolio_returns, index=rebalance_dates, name="portfolio_return")

# ── 5. Compute metrics ───────────────────────────────────────────────────────
def compute_sharpe(ret_series):
    if len(ret_series) < 2:
        return np.nan
    mean = ret_series.mean()
    std = ret_series.std()
    if std == 0:
        return np.nan
    return (mean / std) * np.sqrt(12)

def compute_max_drawdown(ret_series):
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def compute_turnover(long_legs, short_legs):
    """Average fraction of portfolio replaced each month."""
    turnovers = []
    for i in range(1, len(long_legs)):
        prev_all = long_legs[i-1] | short_legs[i-1]
        curr_all = long_legs[i] | short_legs[i]
        added = len(curr_all - prev_all)
        total_slots = N_LONG + N_SHORT
        turnovers.append(added / total_slots)
    return np.mean(turnovers) if turnovers else np.nan

n = len(returns_series)
ann_return = returns_series.mean() * 12
ann_sharpe = compute_sharpe(returns_series)
max_dd = compute_max_drawdown(returns_series)
win_rate = (returns_series > 0).mean()
avg_turnover = compute_turnover(long_legs, short_legs)

# Annual returns
annual_returns = {}
for year in range(2016, 2026):
    yr_ret = returns_series[returns_series.index.year == year]
    if len(yr_ret) > 0:
        annual_returns[str(year)] = round(float((1 + yr_ret).prod() - 1), 6)

# Sub-period Sharpe
ret_2016_2020 = returns_series[(returns_series.index.year >= 2016) & (returns_series.index.year <= 2020)]
ret_2021_2025 = returns_series[(returns_series.index.year >= 2021) & (returns_series.index.year <= 2025)]
sharpe_2016_2020 = compute_sharpe(ret_2016_2020)
sharpe_2021_2025 = compute_sharpe(ret_2021_2025)

# ── 6. Print summary ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  XS-MOMENTUM: SECTOR ETF MOMENTUM BACKTEST RESULTS")
print("="*55)
print(f"  Period          : {returns_series.index[0].date()} to {returns_series.index[-1].date()}")
print(f"  Months          : {n}")
print(f"  Annualized Return: {ann_return:>8.2%}")
print(f"  Annualized Sharpe: {ann_sharpe:>8.3f}  ← DROP THRESHOLD < 0.5")
print(f"  Max Drawdown    : {max_dd:>8.2%}")
print(f"  Win Rate        : {win_rate:>8.1%}")
print(f"  Avg Monthly Turnover: {avg_turnover:>5.1%}")
print("-"*55)
print("  CROWDING CHECK (Sub-period Sharpe)")
print(f"  2016-2020 Sharpe : {sharpe_2016_2020:>8.3f}")
print(f"  2021-2025 Sharpe : {sharpe_2021_2025:>8.3f}")
print("-"*55)
print("  Annual Returns")
for year, ret in annual_returns.items():
    bar = "+" if ret >= 0 else "-"
    print(f"  {year}: {ret:>8.2%}  {'█' * int(abs(ret)*100) if abs(ret)*100 < 40 else '█'*40}")
print("="*55)

# ── 7. Save results ──────────────────────────────────────────────────────────
results = {
    "strategy": "XS-Momentum: Sector ETF 12-1 Momentum",
    "period": f"{returns_series.index[0].date()} to {returns_series.index[-1].date()}",
    "n_months": n,
    "annualized_return": round(float(ann_return), 6),
    "annualized_sharpe": round(float(ann_sharpe), 6),
    "max_drawdown": round(float(max_dd), 6),
    "win_rate": round(float(win_rate), 6),
    "avg_monthly_turnover": round(float(avg_turnover), 6),
    "crowding_check": {
        "sharpe_2016_2020": round(float(sharpe_2016_2020), 6),
        "sharpe_2021_2025": round(float(sharpe_2021_2025), 6),
    },
    "annual_returns": annual_returns,
}

with open(f"{OUTPUT_DIR}/backtest_results.json", "w") as f:
    json.dump(results, f, indent=2)

returns_series.index = returns_series.index.strftime("%Y-%m-%d")
returns_series.to_csv(f"{OUTPUT_DIR}/monthly_returns.csv", header=True)

print(f"\nResults saved to {OUTPUT_DIR}/")
print(f"  backtest_results.json")
print(f"  monthly_returns.csv")
print(f"  etf_prices.csv")

# Drop threshold check
if ann_sharpe < 0.5:
    print(f"\n  *** ALERT: Sharpe {ann_sharpe:.3f} < 0.5 DROP THRESHOLD ***")
else:
    print(f"\n  >>> Sharpe {ann_sharpe:.3f} >= 0.5 threshold — strategy viable")
