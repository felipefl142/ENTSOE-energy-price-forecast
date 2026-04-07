# Energy Price Forecast

24-hour ahead electricity spot price forecasting for the **DE_LU** (Germany-Luxembourg) market zone.

## Overview

A local-first MLOps project that covers the full data-to-serving pipeline:

| Layer | Technology |
|---|---|
| ETL | DuckDB + Parquet (medallion architecture) |
| Feature store | Feast (local provider, SQLite online store) |
| Pipelines & experiment tracking | ZenML (open source) |
| Model | LightGBM MultiOutputRegressor (24 horizons) |
| Serving | FastAPI |
| Dashboard | Streamlit |

**Data sources:**
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — day-ahead prices + grid load
- [Open-Meteo](https://open-meteo.com/) — hourly weather (Frankfurt proxy, free, no key needed)

---

## Architecture

```
ENTSO-E / Open-Meteo
        │
        ▼
data/raw/  (Hive-partitioned Parquet: year=YYYY/month=MM/)
        │  DuckDB: type cast + dedup
        ▼
data/bronze/  prices | load | weather
        │  DuckDB: lag / rolling / calendar SQL template
        ▼
data/silver/  features.parquet  ← point-in-time correct
        │  DuckDB: self-join for 24 target columns
        ▼
data/gold/  abt_train | abt_test  (temporal split)
        │
        ├──► Feast offline + online store
        │
        ▼  ZenML training_pipeline
models/  lgbm_multioutput.pkl | lgbm_q10.pkl | lgbm_q90.pkl
        │
        ▼  FastAPI + Feast online
POST /predict  →  24-hour price forecast + uncertainty bands
        │
        ▼
Streamlit dashboard (Forecast / Model Comparison / EDA / DuckDB Console)
```

---

## Setup

```bash
# Clone and enter the project
git clone git@github.com:felipefl142/ENTSOE-energy-price-forecast.git
cd ENTSOE-energy-price-forecast

# Create and activate virtual environment
python3.12 -m venv .venv  # ZenML requires Python <3.14
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ENTSOE_API_KEY
```

> Get a free ENTSO-E API key: create an account at [transparency.entsoe.eu](https://transparency.entsoe.eu/), then request a token via My Account Settings.

---

## Running the Pipeline

```bash
# 1. Collect data (adjust date range as needed)
python -m etl.run_pipeline --start 2018-01-01 --end 2024-12-31

# 2. Feature store
cd feature_store && feast apply
cd feature_store && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# 3. ZenML setup + training
zenml init   # run once
zenml up     # dashboard → http://127.0.0.1:8237
python -c "from pipelines.training_pipeline import training_pipeline; training_pipeline()"

# 4. Evaluate
python -m ml.evaluate

# 5. Serve
uvicorn serving.api:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs

# 6. Dashboard
streamlit run app/main.py
```

---

## Project Structure

```
ENTSOE-energy-price-forecast/
├── etl/                    # Data pipeline (collect → bronze → silver → gold)
│   └── sql/               # DuckDB SQL templates
├── feature_store/          # Feast definitions (entity, feature views)
├── pipelines/              # ZenML @pipeline definitions
├── ml/                     # Model training and evaluation
├── models/                 # Serialized model artifacts (gitignored)
├── serving/                # FastAPI endpoint
├── app/                    # Streamlit dashboard (4 tabs)
├── notebooks/              # Step-by-step exploration (01–08)
└── data/                   # All data layers (gitignored)
```

---

## Notebooks

| # | Title | Purpose |
|---|---|---|
| 01 | Data Ingestion | Pull data, verify Parquet structure |
| 02 | EDA — Prices | Seasonality, autocorrelation, negative prices |
| 03 | Feature Engineering | Bronze + silver ETL, leakage validation |
| 04 | Feature Store | Feast apply, materialize, historical/online queries |
| 05 | Model Training | Train MultiOutput LightGBM + quantile models |
| 06 | Model Evaluation | OOT MAE per horizon, error analysis |
| 07 | ZenML Pipelines | Trigger pipelines, inspect artifacts |
| 08 | Serving Demo | FastAPI requests + forecast visualization |

---

## Key Design Decisions

- **Direct multi-step forecasting** — 24 independent LightGBM models, one per horizon. Avoids error compounding vs. recursive strategy.
- **Quantile models** — separate `LGBMRegressor(objective="quantile")` for 10th/90th percentile → prediction intervals in the dashboard.
- **Point-in-time correctness** — silver layer uses `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` exclusively. No look-ahead.
- **Temporal train/test split** — `TRAIN_END = 2023-12-31`. No shuffling. OOT test = 2024 onward.
- **ZenML local** — open source, no external service. `zenml up` launches the dashboard locally.
