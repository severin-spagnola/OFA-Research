# OFA-Research

Quantitative strategy research for OverfitAlpha. This repository contains
research code, walk-forward evaluation utilities, result artifacts, and local
dashboard tooling. It does not contain live trading execution or credentials.

> **No brokerage execution lives here.** This repository contains research,
> evaluation utilities, local tooling, and reproducible artifacts. It has no
> credentials and cannot submit orders.

## Research Approach

OFA research follows an adversarial lifecycle:

1. Search a defined strategy space for candidate rules.
2. Validate chronologically with holdouts, walk-forward evaluation, and
   execution-friction assumptions.
3. Stress candidates with drawdown, regime, and robustness analysis.
4. Reject or retire hypotheses when their forward behavior fails stated gates.

The repository retains both successful experiments and falsified ones. A
backtest is research evidence, not a recommendation to trade.

## Repository Layout

```text
current/            Active research modules
  options_research/   SPY 0-5DTE options strategy search
  options_vrp/        Volatility-risk-premium credit-spread research
  options_wf/         Options walk-forward research utilities
  overfit_finder/     MES futures strategy search and robustness analysis
  penny_research/     Gap-up equity tradeability research

infra/              Cloud compute, orchestration, data, and smoke checks
dashboard/          Local read-only demonstration dashboard
results/            Search outputs and validation artifacts
state/              Per-project run bookkeeping and analysis notes
kill_analysis/      Kill-switch threshold and IS/OOS studies
raw_analysis/       Baseline trade-level analysis reports
archived/           Legacy research code
docs/               Research write-ups
scripts/            Standalone smoke runners
```

## Research Modules

### `options_research/`

The original intraday SPY options search. It evaluates debit and credit spreads
with price-action entries and options-native exits, including premium-based
targets, stops, and time exits.

- `options_walkforward.py`: lifecycle simulator with kill gates.
- `options_search.py`: parallel candidate-evaluation engine.
- `options_backtest.py`: spread simulator with execution friction.
- `options_genes.py`: strategy-space definitions.

### `options_vrp/`

Volatility-risk-premium credit-spread research for 5DTE SPY structures. The
pipeline evaluates strategy rules, regime filters, and risk configurations
without reintroducing look-ahead.

- `vrp_wf_v2.py`: three-phase walk-forward evaluation.
- `vrp_live_sim.py`: sequential out-of-sample simulator.
- `vrp_backtest_v2.py`: spread simulator with intrabar TP/SL handling.
- `vrp_genes_v2.py`, `vrp_context.py`, and `vrp_classifier.py`: candidate
  definitions, lookahead-safe context, and profitability gating.

### `options_wf/`

Walk-forward options research with asymmetric payoff modelling and joint
gate/kill-condition optimization.

- `wf_runner.py`: chronological train, validation, and forward runner.
- `joint_gate_kill_optimizer.py`: threshold sweeps against faithful kill logic.
- `oos_validation.py` and `oos_full_validation.py`: held-out validation,
  Monte Carlo, VaR/CVaR, and Calmar analysis.

### `overfit_finder/`

The core MES research engine explores a constrained strategy DSL over recent
data, then relies on forward validation, regime monitoring, and explicit
failure gates rather than treating in-sample selection as evidence.

- `overfit_search.py`: search engine.
- `strategy_dsl.py`: strategy definition and execution DSL.
- `strategy_generator.py` and `strategy_seeds.py`: candidate generation.
- `regime_filter.py`, `regime_monitor.py`, and `macro_gate.py`: deployment
  gating research.
- `robustness_metrics.py`: shared robustness metrics.

### `penny_research/`

Research into whether penny-stock gap-up setups remain tradeable after liquidity
and execution constraints.

- `penny_gapper_survey.py`: scan, enrich, minute-bar, and analysis pipeline.
- `penny_search.py`: pooled ticker-day search by listing and liquidity profile.
- `penny_backtest.py`: cost-aware pooled evaluator.

## Compute And Tooling

Heavy research jobs can use RunPod serverless or Modal; local orchestration and
data utilities live under `infra/`.

- `orchestrator.py`: overnight MES research driver.
- `modal_*.py`: Modal applications for searches and validation.
- `*_runpod_handler.py`: RunPod entrypoints.
- `train_*.py`: versioned classifier-training utilities.
- `meta_optimizer.py`: standard-library online meta-parameter research tool.

## Research Standard

Strategies are evaluated with time-ordered splits, walk-forward testing,
execution-friction assumptions, and explicit drawdown analysis. These are
minimum research requirements, not guarantees of future performance.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run individual research modules from the repository root. External services,
when required by a module, are configured through local environment variables
and are never committed. Inputs vary by module and can include Polygon market
data, yfinance daily data, RunPod, Modal, or local data paths.

## Example Entry Points

```bash
# Overnight MES research orchestration
python infra/orchestrator.py --workers 8

# Options walk-forward research
python current/options_wf/wf_runner.py --train-days 45 --val-days 15 --candidates 2000

# VRP walk-forward research
python current/options_vrp/vrp_wf_v2.py --data-dir /path/to/options_5dte --workers 36

# Local mock dashboard
python dashboard/serve.py
```

## Caveats

- Most candidates researched here should be expected to fail; retained analyses
  make their failure modes inspectable.
- Research artifacts are not investment advice and are not connected to a
  brokerage account.
