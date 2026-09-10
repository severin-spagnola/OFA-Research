# Penny/Small-Cap Gapper Gene Search — Research Summary

**Date**: 2026-03-12
**Status**: Shelved
**Verdict**: No robust tradeable edge under conservative gates

---

## Methodology

Exhaustive gene combinator search over pooled penny and small-cap gapper data. Random gene candidates are evaluated on a 70/30 chronological train/val split, then filtered through three gates:

1. **min_trades = 50** — statistical significance
2. **Positive train cum_r** — must be profitable in-sample
3. **2x cost stress test** — re-simulate val with doubled spread (0.30%), slippage (0.20%), stop slip (0.40%)

## Data

- **Source**: Polygon API via multi-phase survey pipeline
- **Events**: 20,817 gapper events (15%+ gap, $0.005-$10 price, 50K+ prev volume)
- **Bar data**: 1-minute bars for each event day
- **Period**: ~6 months of trading days

## Profiles Tested

| Profile | Description | Gap Threshold |
|---|---|---|
| listed_highvol | NASDAQ/NYSE/AMEX, vol >= 100K | 30% |
| nasdaq_highvol | NASDAQ only, vol >= 200K | 30% |
| otc_highvol | OTC Link, vol >= 50K | 30% |
| smallcap_volatile | Listed $3-10, vol >= 50K | 15% |
| allcap_volatile | Listed under $10, vol >= 50K | 15% |

## Results

### Batch 1 — 100K candidates (30%+ gaps)

| Profile | Viable | Val-pass |
|---|---|---|
| listed_highvol | 24 | 0 |
| nasdaq_highvol | 8 | 0 |
| otc_highvol | 4 | 0 |
| smallcap_volatile | 5 | **1** |
| allcap_volatile | 20 | 0 |

### Batch 2 — 100K candidates (expanded data, 15%+ gaps for smallcap/allcap)

| Profile | Viable | Val-pass |
|---|---|---|
| listed_highvol | 17 | 0 |
| nasdaq_highvol | 18 | **2** |
| otc_highvol | 2 | 0 |
| smallcap_volatile | 13 | 0 |
| allcap_volatile | — | Timed out |

### Total: 3 survivors out of ~200K candidates (0.0015%)

## Surviving Strategies

All three survivors are **short (gap fade)** strategies:

### 1. Short VWAP Cross-Below (nasdaq_highvol)
- **Entry**: Price crosses below VWAP
- **Exit**: TP=2R, SL=fixed 8%, trail=bar_low@2R, hold <= 30min
- **Train**: 320 trades, 46% WR, +7.4R cum, PF 1.29
- **Val**: 125 trades, 49% WR, +8.9R cum, PF ~1.3
- **Val fitness**: 0.2206

### 2. Short 1st-Red After 3 Green (nasdaq_highvol)
- **Entry**: First red bar after 3 consecutive green bars
- **Exit**: TP=3R, SL=fixed, trail=bar_low@1.5R, partial 25%@1R
- **Train**: 113 trades, 47% WR, +4.9R cum
- **Val**: 50 trades, 50% WR, +5.6R cum
- **Val fitness**: 0.0532

### 3. Short Premarket High Break-Below (smallcap_volatile, batch 1 only)
- **Entry**: Price breaks below premarket high
- **Exit**: TP=8R, SL=fixed 15%, hold <= 60min, partial 25%@2R
- **Train**: 190 trades, 55% WR, +15.0R cum, PF 1.29
- **Val**: 54 trades, 61% WR, +9.98R cum, PF 1.72
- **Stress test**: 54 trades, 61% WR, +7.76R cum, PF 1.53
- **Val fitness**: 0.6166

## Key Findings

1. **Edge is exclusively in fading (shorting) gaps** — zero long/continuation strategies survived
2. **Only NASDAQ high-vol gappers** showed repeatable signal
3. **OTC is dead** — too illiquid, too few viable strategies
4. **Cost sensitivity is extreme** — the 2x stress test kills almost everything
5. **0.0015% survival rate** is consistent with random noise, not genuine alpha
6. **All survivors are shorts** — harder to execute on small-caps (locate/borrow risk)

## Why Shelved

- Survival rate indistinguishable from noise
- Short-only edge on small-caps is impractical (borrow costs, locate difficulty)
- Static train/val split (not walk-forward) — survivors may still be overfit
- Cost model is already generous; real execution friction likely higher
- Better research opportunities elsewhere (dark pools, order flow, vol surface)

## Infrastructure (Reusable)

- RunPod serverless gene search pipeline
- Polygon survey pipeline (scan -> enrich -> fetch 1m bars)
- Parallel orchestrator for batch job submission
- Three-gate validation framework
- Profile-based universe filtering
