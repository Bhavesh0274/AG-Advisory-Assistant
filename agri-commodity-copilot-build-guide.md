# Agri-Commodity Intelligence Copilot
### End-to-end build guide (for Claude Code) — forecasting + RAG + data science, on real Indian mandi data

A decision-support copilot for FPOs, traders, and agri-businesses that answers the question they face constantly: **"What will this commodity's price do, and should I sell now or hold?"** It *forecasts* the price (the number), *retrieves* the market drivers behind it (the why), and composes a grounded, confidence-tagged recommendation.

Every build phase has a **goal**, **acceptance criteria**, and a **ready-to-paste Claude Code prompt**. Work top to bottom and gate each phase.

---

## 0. What you're building — and why every technique earns its place

The value of this project is that it's **one coherent decision**, not a forecaster and a chatbot glued together. Each technique solves a real part of the same problem:

- **Time-series forecasting (the quant core)** → produces the number: where the price/arrivals are headed.
- **Market-intelligence RAG (the context)** → explains *why* (monsoon delay, export-policy change, festival demand), with citations — because a bare forecast is unactionable.
- **Data-science analytics (the signal)** → anomaly detection (price spikes/crashes), market correlation/clustering (which mandis move together → arbitrage), demand–supply reading from arrivals.
- **An agent layer (the composition)** → takes a natural-language question and assembles forecast + drivers + analytics into a single grounded recommendation.

State this coherence explicitly in your README. The failure mode — and the thing that reads as *junior* — is bolting techniques together to check boxes. The strong version: forecasting gives the number, RAG grounds the why, analytics enrich the signal, the agent composes the decision.

**Who uses it:** FPOs deciding when to aggregate and sell; traders timing purchases; agri-businesses planning procurement. **The KPI:** better sell/hold timing → less distress-selling, better realized prices.

---

## 1. The honesty frame (read first — it's the credibility)

Two disciplines make or break this project, and both are *your* strengths:

1. **Rigorous backtesting.** Mandi prices are noisy. Any forecast must be validated with **walk-forward / expanding-window** evaluation (never random splits — that leaks the future), and benchmarked against a **seasonal-naive baseline using MASE**. Be fully prepared to report "the LSTM/TFT barely beats seasonal-naive" if that's the truth — on noisy commodity data it often is. That honesty is the maturity signal, not a weakness. A rigorously-validated modest result beats an impressive-looking overfit one.

2. **Decision-support, not autonomous trading — and not financial advice.** The copilot *informs* a human; it never trades and never claims certainty. Every recommendation carries an uncertainty band and a disclaimer. Linking a retrieved advisory to a price move is *plausible context*, not a proven causal claim — word it that way.

---

## 2. Data — where to get it

### 2a. Quantitative core: mandi prices + arrivals (the forecasting input)

- **Agmarknet daily prices — data.gov.in** — `https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi` — daily wholesale min/max/**modal** price per commodity/variety/market, government-published, with a **Catalog API** and Zip download.
- **Agmarknet portal** — `https://agmarknet.gov.in/` — daily **price *and arrival*** reports. Arrivals = the supply signal (crucial: price without volume is half the story).
- **CEDA Agri Market Data (Ashoka University)** — `https://agmarknet.ceda.ashoka.edu.in/` — a **cleaned, historical** version of Agmarknet prices + arrivals at national/state/district level. Use this to skip a lot of the raw-data mess.

> ⚠️ Raw Agmarknet is messy: missing days, inconsistent commodity/market spellings, unit quirks, outliers (fat-finger prices). Cleaning is real work — the CEDA version helps; still budget time for it.

### 2b. RAG corpus: the market drivers (the "why")

Openly available government sources that *explain* price moves:

- **IMD** (`mausam.imd.gov.in`) — weather forecasts, monsoon updates, and **agromet advisory** bulletins.
- **Crop production estimates** — Ministry of Agriculture Advance Estimates / *Agricultural Statistics at a Glance* (supply outlook).
- **DGFT** (`dgft.gov.in`) — export/import **policy notifications** (e.g., onion export bans/relaxations move prices sharply).
- Commodity advisories / market commentary from government and commodity boards.

### 2c. Scope deliberately

Pick **3–5 commodities** with real price volatility and policy sensitivity (onion, tomato, potato are classics; add a pulse and a cereal) across **a handful of major mandis**. Depth over breadth.

**Acceptance criteria:** a multi-year price+arrivals series for your chosen commodities/markets in `data/processed/`, plus a small RAG corpus of driver documents in `data/corpus/` with a `sources.csv`.

**Claude Code prompt:**
> Build `src/data/fetch.py`: pull Agmarknet daily price+arrivals for my chosen commodities/markets (via the data.gov.in Catalog API, with the CEDA cleaned data as a fallback/backfill), normalize commodity/market names, and store a clean daily series in `data/processed/`. Separately, set up `data/corpus/` for driver docs (IMD advisories, crop estimates, DGFT notifications) with a `sources.csv` template. Print coverage (date range, % missing days per series).

---

## 3. Data pipeline & feature engineering

**Goal:** analysis-ready series + features.

- Resolve to a consistent daily (or weekly) frequency per commodity×market; handle missing days (forward-fill short gaps, flag long ones).
- Outlier detection/treatment on prices.
- Features: calendar (month, festival flags, sowing/harvest season), lags & rolling stats of price and **arrivals**, price spreads across nearby mandis, and exogenous hooks (a weather/rainfall series if you add one).

**Acceptance criteria:** `data/processed/features.parquet` with a documented feature set; a notebook showing seasonality and the price–arrivals relationship.

**Claude Code prompt:**
> Build `src/data/build_features.py`: resample to consistent frequency, impute/flag gaps, treat outliers, and engineer calendar (incl. season/festival flags), price & arrivals lags/rolling stats, and cross-mandi spreads. Save `data/processed/features.parquet` and a short EDA notebook on seasonality and price↔arrivals.

---

## 4. Forecasting core (the quant heart)

**Goal:** forecast modal price (and optionally arrivals) at a useful horizon (e.g., 7/14/30 days).

Build up in order and **always compare against the baseline**:

1. **Baselines:** seasonal-naive (last season's same period) and a naive random-walk — these are what you must beat.
2. **Classical:** ARIMA/SARIMA and/or Prophet (handles seasonality, interpretable).
3. **ML:** LightGBM/XGBoost on lag + calendar + arrivals features (usually the strongest on tabularized series).
4. **DL:** an LSTM and a **Temporal Fusion Transformer** (TFT handles multiple series + known-future covariates like season/festival, and gives interpretable attention).

**Acceptance criteria:** all model families produce horizon forecasts for each commodity×market; a leaderboard exists (built in §5).

**Claude Code prompt:**
> Build `src/forecast/`: implement seasonal-naive and random-walk baselines, ARIMA/Prophet, a LightGBM lag-feature model, and an LSTM + Temporal Fusion Transformer (via a library like darts or pytorch-forecasting), all with a common `fit/predict(horizon)` interface over `features.parquet` for each commodity×market.

---

## 5. Backtesting & evaluation harness ⭐ (the star — your moat)

This is where the project earns credibility. **The harness is the point; the models are what it measures.**

- **Walk-forward / expanding-window** backtesting only — retrain as the window rolls; never a random split.
- Metrics: **MASE** (scaled against seasonal-naive — the honest one), plus MAE/RMSE and MAPE, per horizon and per commodity.
- A results table: every model × commodity × horizon, with the baseline as the reference row.
- **Directional accuracy** (did it call up vs down correctly?) — often what a trader cares about more than exact price.
- Report the honest finding, whatever it is — including "model X only beats seasonal-naive on commodities with strong seasonality; on volatile ones the baseline wins."

**Acceptance criteria:** one command runs the walk-forward backtest and writes `reports/forecast_eval.md` with the leaderboard + a written interpretation; you can quote MASE vs seasonal-naive per commodity.

**Claude Code prompt:**
> Build `src/forecast/backtest.py`: walk-forward/expanding-window evaluation retraining each step, computing MASE (vs seasonal-naive), MAE/RMSE, MAPE, and directional accuracy per model×commodity×horizon. Aggregate into `reports/forecast_eval.md` with a leaderboard and an honest interpretation of where models beat the baseline and where they don't.

---

## 6. Data-science analytics layer

**Goal:** turn the series into decision signals beyond the point forecast.

- **Anomaly detection:** flag abnormal price spikes/crashes (STL residuals, robust z-score, or a model) → alerting.
- **Market correlation & clustering:** which mandis/commodities move together (correlation matrix + clustering) → arbitrage and substitution signals.
- **Demand–supply read:** relate arrivals to price to characterize gluts/shortages.

**Acceptance criteria:** `reports/analytics.md` with detected anomalies, a market-correlation/cluster map, and an arrivals-vs-price read for each commodity.

**Claude Code prompt:**
> Build `src/analytics/`: price anomaly detection (STL/robust z-score), a market correlation matrix + clustering of co-moving mandis/commodities, and an arrivals-vs-price demand–supply analysis. Output `reports/analytics.md` with the findings.

---

## 7. Market-intelligence RAG (the "why")

**Goal:** ground the forecast in real, cited drivers.

- Ingest the §2b driver corpus (IMD advisories, crop estimates, DGFT notifications); chunk, embed (BGE/E5), index (FAISS/Chroma).
- Retrieval answers "what's driving <commodity> prices now?" with **citations**, and surfaces relevant recent events (e.g., an export-policy change) alongside the forecast.
- Keep it honest: retrieved context is *plausible explanation*, not proven causation — phrase accordingly.

**Acceptance criteria:** for a commodity, the system returns 2–4 cited driver snippets relevant to the current outlook; abstains when the corpus has nothing relevant.

**Claude Code prompt:**
> Build `src/rag/`: ingest the driver corpus (IMD/crop-estimates/DGFT), embed + FAISS index, and `explain(commodity)` returning cited driver snippets relevant to the recent window, abstaining when nothing relevant is retrieved. Phrase outputs as plausible context, not causal claims.

---

## 8. The fusion / agent layer (compose the decision)

**Goal:** one natural-language interface — *"Should I hold or sell onions in Nashik over the next two weeks?"* — that composes everything.

The agent, given the question, calls tools: `get_forecast(commodity, market, horizon)`, `get_analytics(commodity, market)`, `explain_drivers(commodity)`, then produces a structured recommendation: **outlook + forecast (with uncertainty band) + the cited drivers + a hold/sell suggestion tagged with confidence + the disclaimer.** It never hides uncertainty and never claims certainty.

**Acceptance criteria:** a NL question yields a composed answer citing the forecast number, its uncertainty, the driver sources, and a clearly-hedged recommendation.

**Claude Code prompt:**
> Build `src/agent/`: a Claude tool-calling loop with tools wrapping the forecaster, analytics, and RAG. Given a hold/sell question, it composes a structured answer — outlook, forecast + uncertainty band, cited drivers, confidence-tagged suggestion, and disclaimer. Log tool calls; never emit a recommendation without the uncertainty band and disclaimer.

---

## 9. Guardrails & honest UX

- **Uncertainty always shown** — prediction intervals, not just a point; degrade gracefully to "too volatile to call" when intervals are wide.
- **Financial-decision disclaimer** on every recommendation — decision-support, informational, not advice; the user decides.
- **Driver claims hedged** — "possible factors," never "prices will rise because."
- **Data-freshness indicator** — show the last date the series covers.

**Acceptance criteria:** no recommendation renders without an uncertainty band, freshness date, and disclaimer.

---

## 10. Dashboard + serving

- **FastAPI** backend wrapping the agent + tools.
- A dashboard: price history + forecast band per commodity/market, anomaly flags, the co-movement map, a driver/news panel, and the chat box for hold/sell questions.
- Deploy API + UI; secrets via env vars; a scheduled job to refresh the Agmarknet pull daily.

**Acceptance criteria:** a deployed URL where you pick a commodity/market, see the forecast + drivers + analytics, and ask a hold/sell question end-to-end.

**Claude Code prompt:**
> Build `src/api/` (FastAPI) + `ui/` (Streamlit or React): forecast-band charts, anomaly flags, co-movement map, cited-driver panel, and a chat box over the agent. Add a daily scheduler to refresh the Agmarknet data. Dockerfile + deploy notes with env-var secrets.

---

## 11. Limitations & productionization (README)

- **Noisy, semi-efficient prices** — cap your claims; report MASE honestly; some commodities won't be forecastable beyond the baseline, and saying so is correct.
- **Data quality & latency** — Agmarknet has reporting gaps/lags; state the freshness limits.
- **Causation vs correlation** — drivers are context, not proof.
- **Not trading infrastructure** — decision-support only; productionizing would add real-time feeds, per-mandi calibration, and monitoring of forecast drift.

---

## 12. Stretch goals

- **Weather/satellite fusion** — add rainfall/NDVI (Sentinel-2) as exogenous features → ties in your remote-sensing/CV strength.
- **Probabilistic forecasting** — quantile/TFT intervals surfaced directly as risk bands.
- **Multilingual + WhatsApp delivery** — push alerts to FPOs in regional languages (real-world distribution).
- **Backtested "trading" strategy** — a simulated hold/sell policy vs sell-at-harvest, with honest PnL — strong quant-interview material.

---

## 13. Repo structure & Claude Code workflow

```
commodity-copilot/
├─ CLAUDE.md
├─ README.md            # coherence thesis + forecast leaderboard + limitations
├─ requirements.txt
├─ data/{raw,processed,corpus}/
├─ reports/{forecast_eval,analytics}.md
├─ src/{data,forecast,analytics,rag,agent,api}/
├─ ui/
└─ tests/
```

**`CLAUDE.md` starter:**
```md
# Project: Agri-Commodity Intelligence Copilot
Coherence: forecasting gives the number, RAG the why, analytics the signal, agent the decision.
Build order: data → features → forecasting → backtest harness → analytics → RAG → agent →
guardrails → dashboard.

## Rules
- Backtest with walk-forward/expanding-window ONLY; report MASE vs seasonal-naive; never random splits.
- Decision-support, not advice: always show uncertainty + a disclaimer; drivers are context, not causation.
- Beat the baseline before adding complexity; report honestly when a model doesn't.
- Every phase ships tests + updates its report. Secrets via env vars.

## Stack
pandas, darts / pytorch-forecasting / statsforecast, LightGBM, scikit-learn,
FAISS + BGE/E5 + RAGAS, Anthropic API (agent), FastAPI + Streamlit.
```

**Phase-gate loop:** implement → test → check acceptance → commit → next. Ship a thin slice first: one commodity → baselines + LightGBM → backtest table. That alone is a legitimate forecasting project; RAG and the agent deepen it.

---

## 14. Dependencies (`requirements.txt` start)

```
pandas  numpy  requests  pyarrow
statsforecast  prophet  darts  pytorch-forecasting  lightgbm  scikit-learn
sentence-transformers  faiss-cpu  chromadb  ragas
anthropic
fastapi  uvicorn  streamlit  apscheduler
pytest
```

Pin versions once it runs; `torch` per your CUDA.

---

## Build-order recap

`data (Agmarknet + drivers) → features → forecasting (baselines → ML → DL) → walk-forward backtest → analytics (anomaly · co-movement · demand-supply) → RAG drivers → agent fusion → guardrails → dashboard.`

Internalize one sentence: **the walk-forward backtest is the spine, and the coherence is the story.** A forecast validated honestly against seasonal-naive, explained by cited real-world drivers, composed into one hold/sell answer — that's a project that sits exactly where your quant interest meets your agriculture domain, and almost nobody else can build it.
