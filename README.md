# Agri-Commodity Intelligence Copilot

A decision-support copilot for FPOs, traders, and agri-businesses that answers: **"What will this
commodity's price do, and should I sell now or hold?"** It *forecasts* the price (the number),
*retrieves* the market drivers behind it (the why), and composes a grounded, confidence-tagged
recommendation.

## Coherence thesis

This is one coherent decision, not a forecaster and a chatbot glued together:

- **Time-series forecasting (the quant core)** → the number: where price/arrivals are headed.
- **Market-intelligence RAG (the context)** → the why (monsoon, export policy, festival demand), cited.
- **Data-science analytics (the signal)** → anomalies, market co-movement/clustering, demand–supply read.
- **Agent layer (the composition)** → assembles forecast + drivers + analytics into one grounded answer.

Forecasting gives the number, RAG grounds the why, analytics enrich the signal, the agent composes
the decision.

## Status

Build in progress, following `agri-commodity-copilot-build-guide.md`. Current phase: **forecasting**.

- [x] Phase 1 — repo scaffold
- [x] Phase 2 — price data ingestion (RAG driver corpus still pending)
- [x] Phase 3 — feature engineering
- [ ] Phase 4 — forecasting models
- [ ] Phase 5 — walk-forward backtest harness
- [ ] Phase 6 — analytics layer
- [ ] Phase 7 — RAG
- [ ] Phase 8 — agent fusion
- [ ] Phase 9 — guardrails
- [ ] Phase 10 — dashboard + serving

## Scope

**Commodities: Onion, Potato.** Markets: Agra, Indore, Lasalgaon, Pune (Latur has no coverage in
the current data source and is excluded from the working set, though still in `DEFAULT_MARKETS`
for when a better source is added).

Narrowed from the original 5-commodity plan (Onion, Tomato, Potato, Wheat, Tur/pulse) after
Phase 2's coverage report: the historical source we ended up using only has ~2 full years of
Onion and Potato data; Tomato, Wheat, and Rice are capped at 2-8 months regardless of market,
too short for a seasonal-naive baseline. See `data/processed/coverage_report.csv` for the exact
numbers, and `src/config.py` for the reasoning. Revisit if a source with fuller multi-commodity
history is added.

## Data sources

- **data.gov.in Agmarknet API** — only returns *today's* snapshot (not historical), despite
  being the guide's suggested primary source. Kept in the pipeline for future day-by-day
  accumulation, but useless for backfilling history.
- **CEDA Agri Market Data** — has a real historical API, but it requires emailing
  `ceda@ashoka.edu.in` for a token; not self-serve. Not used yet.
- **Kaggle — "Indian Agricultural Mandi Prices (2023-2025)"** — the actual current source.
  Manually downloaded (Kaggle blocks scripted access), placed in `data/raw/kaggle/`, parsed by
  `load_kaggle_fallback()` in `src/data/fetch.py`. Does not include arrivals volume.

## Setup

```bash
pip install -r requirements.txt
export DATA_GOV_IN_API_KEY=your_key_here   # get one at https://data.gov.in (free registration)
python -m src.data.fetch
```

`fetch.py` merges whatever's available from the live API, `data/raw/ceda/` exports, and
`data/raw/kaggle/` exports, then writes `data/processed/prices.parquet` +
`data/processed/coverage_report.csv`.

## Limitations (honest, up front)

- Mandi prices are noisy and semi-efficient; not every commodity will be forecastable beyond a
  seasonal-naive baseline, and we report that honestly rather than overstate model skill.
- Agmarknet has reporting gaps and lag; freshness limits are surfaced in the UI.
- No arrivals (supply/volume) data yet in any source we've wired up — the "demand-supply read"
  in Phase 6 will be feature-limited to price alone until we get one.
- Current data source (Kaggle) has ~10-40% missing days even within its covered window; Phase 3
  needs a real gap-handling strategy, not just an assumption of clean daily data.
- Retrieved drivers are plausible context, not proven causation.
- This is decision-support, **not** trading infrastructure and **not** financial advice.

## Repo structure

```
├─ CLAUDE.md
├─ README.md
├─ requirements.txt
├─ data/{raw,processed,corpus}/
├─ reports/{forecast_eval,analytics}.md
├─ src/{data,forecast,analytics,rag,agent,api}/
├─ ui/
└─ tests/
```
