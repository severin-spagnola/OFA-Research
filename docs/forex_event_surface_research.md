# OFA Forex: Event-Weighted Adaptive Strategy System

## Core Concept

Instead of OFA's current discrete strategy lifecycle (search → deploy → kill → replace), forex strategies **continuously adapt their internal parameters** in response to an event surface. The strategy's event weights, interaction coefficients, and decay functions are themselves the searchable genome — and can be updated online without full strategy replacement.

### Why Forex

- 24/5 market with continuous flow of macro events across multiple economies
- Event impact is measurable (surprise = actual - consensus) and decays over time
- Cross-currency dynamics create a rich interaction surface (hawkish Fed + dovish ECB ≠ either alone)
- Execution is cheap (sub-pip spreads on majors outside of events)
- OFA lifecycle transfers directly; the gene representation is what changes

### Key Difference from Options OFA

| | Options OFA | Forex OFA |
|---|---|---|
| Strategy representation | Discrete genes (entry condition, TP, SL, trade type) | Continuous coefficients (event weights, interaction terms, decay functions) |
| Adaptation | None — strategy is static until killed | Online — coefficients adjust as events resolve |
| Regime detection | Implicit (kill conditions fire when regime shifts) | Explicit macro regime layer (risk-on/off/crisis) + continuous adaptation within regime |
| Search space | ~20-30 gene combinations, brute force random search | 50-200+ parameters, requires sparse search (L1/Bayesian/differential evolution) |
| Kill cycle | Strategy dies, new one replaces it | Coefficients adapt; full replacement only on structural regime change |

---

## Architecture

### Layer 1: Event Surface

The raw input — a time series of macro events with surprise metrics.

```
EventRecord:
  timestamp: datetime (UTC)
  event_type: str          # "US_NFP", "ECB_RATE", "UK_CPI", etc.
  country: str             # "US", "EU", "UK", "JP", "AU", "CA", "CH"
  actual: float            # reported value
  consensus: float         # market expectation
  prior: float             # previous release
  surprise: float          # (actual - consensus) / historical_std
  revision: float          # prior_revised - prior_original
```

~30-50 event types across 6-8 economies. Each event fires 1-12 times per month depending on type.

### Layer 2: Signal Model (the searchable genome)

```python
@dataclass
class ForexGenes:
    # Event weights — how much each event type moves the signal
    event_weights: dict[str, float]     # {"US_NFP": 0.7, "ECB_RATE": 1.2, ...}

    # Interaction terms — cross-event amplifiers
    interactions: list[InteractionTerm]  # [("FED_HAWK", "ECB_DOVE", 1.5), ...]

    # Decay function — how quickly event impact fades
    decay_type: str                     # "exponential" | "linear" | "step"
    decay_halflife_hours: float         # e.g., 48 = event impact halves every 2 days

    # Signal aggregation
    signal_fn: str                      # "weighted_sum" | "polynomial" | "rank"
    entry_threshold: float              # signal > threshold → enter long
    exit_threshold: float               # signal < -threshold → enter short

    # Risk params (same as options OFA)
    tp_pips: float
    sl_pips: float
    max_position_hours: int

    # Regime filter
    regime_filter: str | None           # "risk_on" | "risk_off" | None
```

### Layer 3: Online Adaptation

After each event resolves (price moves in response), update coefficients:

```
For each resolved event:
  1. Measure actual price response (Δprice in window after event)
  2. Compare to model's predicted response (event_weight * surprise)
  3. Update weight: w_new = w_old + lr * (actual_response - predicted_response) * surprise
```

This is lightweight gradient descent on the event weights. No full retrain needed — just nudge coefficients toward reality as new data arrives.

### Layer 4: OFA Lifecycle (modified)

```
Every N days (or on regime change detection):
  1. Evaluate current model's rolling performance
  2. If degraded beyond kill thresholds → full coefficient re-search
  3. Otherwise → continue with online adaptation

Regime change detection:
  - VIX regime shift (calm → volatile → crisis)
  - Yield curve inversion/steepening
  - Correlation breakdown (USD pairs decorrelate)
  - → Triggers full re-search of coefficient space
```

---

## Data Requirements

### Tier 1: Minimum Viable Prototype (~$50-100/month)

| Data | Source | Cost | Historical Depth | Notes |
|---|---|---|---|---|
| FX minute bars (6 major pairs) | OANDA API / Dukascopy | $0-50/mo | 5-20 years | EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF |
| Economic calendar + surprises | Trading Economics API | $50/mo | 10+ years | Timestamp, actual, consensus, prior for all major releases |
| US yields (2Y, 5Y, 10Y) | FRED | $0 | 20+ years | Daily, free |
| VIX / DXY | Yahoo Finance / FRED | $0 | 20+ years | Risk regime detection |

**This is enough to prototype.** EUR/USD + US macro events only. ~15-20 event types (NFP, CPI, PPI, retail sales, GDP, ISM, FOMC, jobless claims, etc.) with surprise metrics.

### Tier 2: Multi-Economy Expansion (~$100-300/month)

| Data | Source | Cost | Notes |
|---|---|---|---|
| ECB/BOE/BOJ/RBA rate decisions + minutes | Central bank websites | $0 | Scraping + LLM scoring for hawk/dove |
| European/UK/JP macro releases | Trading Economics | Included in Tier 1 sub | Extends event surface to 50+ types |
| German/EU yields | ECB Statistical Data Warehouse | $0 | Yield differentials drive EUR/USD |
| Commodity prices (gold, oil, copper) | Yahoo Finance | $0 | Risk sentiment proxies |
| CFTC Commitment of Traders | CFTC website | $0 | Weekly positioning data, 3-day lag |
| Retail sentiment | OANDA Order Book / IG | $0 | Contrarian signal |

### Tier 3: Advanced (future, if edge proven)

| Data | Source | Cost | Notes |
|---|---|---|---|
| Central bank speech scoring | Claude API batch processing | ~$50-100 one-time | Score historical speeches for hawk/dove, build time series |
| Real-time news headlines | Reuters/DJ Newswires | $1,000+/mo | Only needed for live sub-minute event trading |
| Options-implied vol surfaces | Bloomberg / Refinitiv | $500+/mo | FX vol smile dynamics |
| Cross-asset flow data | Prime broker feeds | Not available to retail | Institutional positioning |

---

## Search Space Design

### The Overfitting Problem

Options OFA has ~30 gene combinations → brute force 5,000 random candidates works fine. Forex event surface with 50 event types, interaction terms, and decay params could have 200+ dimensions. Random search won't work.

### Approach: Sparse Coefficient Search

1. **L1-regularized search** — most event weights should be zero. The search finds the 8-12 events that actually drive price for a given pair in a given regime. Everything else is noise.

2. **Hierarchical search**:
   - Stage 1: Search individual event weights (50 params, parallelizable)
   - Stage 2: Search interaction terms between top-10 events (45 pairs)
   - Stage 3: Search decay functions and signal aggregation (5-10 params)

3. **Differential evolution** instead of random search — population-based optimization that works well in continuous spaces. Can run on RunPod same as current options search.

4. **Bayesian optimization** for fine-tuning — after DE finds a good region, use GP-based BO to optimize within it. Especially useful for decay halflife and threshold params.

### Validation

Same train/val/forward-test split as options OFA, but:
- **Walk-forward on event time, not calendar time** — each event is a sample, not each day
- **Out-of-sample events matter more than out-of-sample days** — a model that predicts NFP response well on training NFPs but fails on holdout NFPs is overfit, even if the holdout days are chronologically later
- **Minimum event count per type** — need at least 10-15 instances of each event type in training window for statistical significance. Monthly events need 12-18 months of training data minimum.

---

## Execution Considerations

### Spread Dynamics

Unlike SPY options (fixed, tight spreads), forex spreads are variable:
- EUR/USD normal: 0.1-0.5 pips (~$1-5 per 100K lot)
- EUR/USD during NFP: 3-15 pips (~$30-150 per 100K lot)
- Exotic pairs: 5-30 pips always

**Implication**: Event-driven strategies must account for spread widening. If the model says "go long EUR/USD on positive NFP surprise," the entry cost is 10-30x normal. The edge must exceed the widened spread.

**Approach**: Include spread model in backtester. Use historical tick data to estimate spread at event time. Gate entries on minimum expected edge vs. estimated spread.

### Position Sizing

Multiple concurrent strategies (like uncapped options OFA) means multiple positions. Need:
- Per-pair position limits (don't go 5x long EUR/USD from 5 strategies)
- Net exposure tracking (long EUR/USD + long EUR/GBP = 2x EUR exposure)
- Correlation-aware sizing (USD/JPY and EUR/JPY are ~0.7 correlated)

### Broker Selection

- **OANDA**: Good API, reasonable spreads, easy to start. No ECN.
- **Interactive Brokers**: Best execution, lowest costs, but complex API.
- **For prototype/backtest**: Doesn't matter — just need historical data.

---

## Implementation Roadmap

### Phase 0: Data Pipeline (1-2 weeks)
- [ ] Sign up for Trading Economics API
- [ ] Build event scraper/downloader → parquet files with surprise metrics
- [ ] Download OANDA/Dukascopy minute bars for EUR/USD (5+ years)
- [ ] Download FRED yields, VIX, DXY
- [ ] Build alignment layer (all timestamps UTC, event→price bar matching)
- [ ] Validate: no lookahead bias in surprise calculations

### Phase 1: EUR/USD Prototype (2-3 weeks)
- [ ] Define ForexGenes dataclass (event weights, decay, thresholds)
- [ ] Build backtester: event → signal → entry → TP/SL exit
- [ ] Build spread model from tick data
- [ ] Implement sparse search (DE or L1-regularized random)
- [ ] Run walk-forward on EUR/USD with US macro events only
- [ ] Evaluate: is there signal? What events matter?

### Phase 2: Online Adaptation (1-2 weeks)
- [ ] Implement coefficient update rule (lightweight gradient)
- [ ] Compare: static coefficients vs. online-adapted
- [ ] Test regime detection (VIX-based, yield-curve-based)
- [ ] Evaluate: does adaptation improve or does it overfit to recent events?

### Phase 3: Multi-Pair Multi-Economy (2-3 weeks)
- [ ] Extend to GBP/USD, USD/JPY, AUD/USD
- [ ] Add ECB, BOE, BOJ events to surface
- [ ] Implement cross-pair correlation tracking
- [ ] Portfolio-level risk management (net exposure, correlation-adjusted sizing)
- [ ] Live sim (same architecture as options OFA livesim)

### Phase 4: Central Bank NLP (1-2 weeks)
- [ ] Batch-score historical Fed/ECB/BOE speeches with Claude
- [ ] Add hawk/dove score as event type in surface
- [ ] Test: does adding speech sentiment improve out-of-sample?

---

## Key Risks

1. **Crowded trade**: Everyone trades NFP. The edge may be in *how* you weight the interaction of NFP with other recent events, not in NFP alone.

2. **Regime fragility**: 2022 (aggressive rate hikes) looks nothing like 2019 (easing cycle). Coefficients optimized on one may fail on the other. Online adaptation is the mitigation, but it needs enough events to adapt before losing too much.

3. **Spread costs eat edge**: If the edge is 2-3 pips per event trade and spreads widen by 5+ pips during events, you're underwater. May need to trade *around* events (position before, exit after) rather than *on* events.

4. **Event rarity**: FOMC is 8x/year. That's 8 samples per year for training. Need multi-year windows or cross-economy pooling to get statistical significance.

5. **Execution speed**: We're not competing with HFT. The edge would need to be in *which combination of events matters* and *how they interact over hours/days*, not in being 50ms faster to react to NFP.

---

## Budget Estimate

### Research Phase (3-6 months)
- Trading Economics API: $50/mo × 6 = $300
- RunPod compute (same as options): ~$50-100/mo = $300-600
- Claude API for speech scoring: ~$50-100 one-time
- **Total: ~$650-1,000**

### Live Trading Phase (ongoing)
- Data feeds: $50-100/mo
- Broker: OANDA or IB (no monthly fee, just spreads/commissions)
- RunPod for retraining: ~$20-50/mo
- **Total: ~$70-150/mo**

Easily justifiable once options OFA is generating consistent PnL.
