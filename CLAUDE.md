# Project: Agri-Commodity Intelligence Copilot

Coherence: forecasting gives the number, RAG the why, analytics the signal, agent the decision.
Build order: data → features → forecasting → backtest harness → analytics → RAG → agent →
guardrails → dashboard.

## Rules
- Backtest with walk-forward/expanding-window ONLY; report MASE vs seasonal-naive; never random splits.
- Decision-support, not advice: always show uncertainty + a disclaimer; drivers are context, not causation.
- Beat the baseline before adding complexity; report honestly when a model doesn't.
- Every phase ships tests + updates its report. Secrets via env vars (DATA_GOV_IN_API_KEY, ANTHROPIC_API_KEY).

## Scope (current slice)
- Commodities: Onion, Potato (narrowed from a 5-commodity plan; see README.md Scope section)
- Markets: Agra, Indore, Lasalgaon, Pune (Latur has no coverage in the current data source)
- Data source: Kaggle "Indian Agricultural Mandi Prices (2023-2025)" export in
  data/raw/kaggle/, no arrivals column. See README.md "Data sources" for why data.gov.in and
  CEDA weren't usable as the primary historical source.

## Stack
pandas, darts / pytorch-forecasting / statsforecast, LightGBM, scikit-learn,
FAISS + BGE/E5 + RAGAS, Anthropic API (agent), FastAPI + Streamlit.

## Phase-gate loop
implement → test → check acceptance → commit → next. See `agri-commodity-copilot-build-guide.md`
for full phase-by-phase goals, acceptance criteria, and prompts.
