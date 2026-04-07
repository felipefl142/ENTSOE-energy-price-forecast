# Energy Price Forecast — CLAUDE.md

## Overview

Local-first MLOps platform for 24-hour ahead electricity spot price forecasting (DE_LU market zone).

- **Data**: ENTSO-E (day-ahead prices + grid load) + Open-Meteo (weather, Frankfurt proxy)
- **ETL**: DuckDB medallion pipeline (raw → bronze → silver → gold)
- **Feature store**: Feast (local provider, SQLite online store, Parquet offline store)
- **Pipelines**: ZenML open source (`zenml up` → http://127.0.0.1:8237)
- **Model**: LightGBM MultiOutputRegressor (24 independent models, one per horizon)
- **Serving**: FastAPI + Feast online lookup
- **Dashboard**: Streamlit with 4 tabs

---

## Architecture

```
ENTSO-E / Open-Meteo
        │
        ▼
data/raw/  (Hive-partitioned Parquet: year=YYYY/month=MM/)
        │
        ▼  DuckDB: type cast + dedup
data/bronze/  prices.parquet | load.parquet | weather.parquet
        │
        ▼  DuckDB: lag/rolling/calendar SQL template
data/silver/  features.parquet  (point-in-time correct)
        │
        ▼  DuckDB: self-join for 24 target columns
data/gold/  abt.parquet | abt_train.parquet | abt_test.parquet
        │
        ├──► data/feast/  (Feast offline + online store)
        │
        ▼  ZenML training_pipeline
models/lgbm_multioutput.pkl
        │
        ▼  FastAPI + Feast online
POST /predict → 24-hour price forecast
        │
        ▼
Streamlit dashboard
```

---

## Commands

```bash
# Full ETL (collect + bronze + silver + gold)
python -m etl.run_pipeline --start 2018-01-01 --end 2024-12-31

# Individual ETL steps
python -m etl.collect --start 2018-01-01 --end 2024-12-31
python -m etl.bronze
python -m etl.silver
python -m etl.gold

# Feast
cd feature_store && feast apply
cd feature_store && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# ZenML
zenml init                   # run once in project root
zenml up                     # start dashboard at http://127.0.0.1:8237
python -c "from pipelines.training_pipeline import training_pipeline; training_pipeline()"
python -c "from pipelines.inference_pipeline import inference_pipeline; inference_pipeline()"

# FastAPI
uvicorn serving.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

# Streamlit
streamlit run app/main.py
```

---

## Medallion Layers

| Layer | Path | Description |
|---|---|---|
| raw | `data/raw/{source}/year=YYYY/month=MM/` | Original API data, hive-partitioned |
| bronze | `data/bronze/{prices,load,weather}.parquet` | Typed, deduplicated |
| silver | `data/silver/features.parquet` | Lag + rolling + calendar + weather features (point-in-time) |
| gold | `data/gold/abt*.parquet` | ABT with 24 target columns, temporal train/test split |
| feast | `data/feast/` | Feast-formatted offline store + SQLite online store |

---

## Key Conventions

- **SQL engine**: DuckDB for all transformations. No pandas for heavy lifting.
- **Point-in-time**: Silver layer uses `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` exclusively.
  Features at row `t` only use data from `t-N` through `t-1`. No look-ahead, ever.
- **Targets**: 24 columns `price_t_plus_1h` … `price_t_plus_24h` in gold ABT.
- **Train/test split**: strictly temporal. `TRAIN_END = "2023-12-31"`. No shuffling.
- **Model**: 24 independent LightGBM models (direct multi-step, avoids error compounding).
- **Quantile models**: separate `LGBMRegressor(objective="quantile")` for 10th/90th percentile
  uncertainty bands. Stored in `models/lgbm_q10.pkl` and `models/lgbm_q90.pkl`.
- **Feature cols**: curated list defined once in `ml/train.py:FEATURE_COLS`, imported everywhere.
- **Feast entity**: `(market_zone, prediction_ts_utc)` string pair as join keys.
- **ZenML artifacts**: stored in `.zen/local_stores/`. View lineage in dashboard.
- **SQL templates**: `{prices_path}`, `{load_path}`, `{weather_path}`, `{silver_path}` placeholders.

---

## Environment Setup

**Always use the project's `.venv` — never install packages into the system/native Python.**

```bash
# Create venv (first time only) — must use python3.12; ZenML requires Python <3.14
python3.12 -m venv .venv

# Activate (do this before running any command)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # fill in ENTSOE_API_KEY
zenml init
```

All commands in this project assume `.venv` is active. When suggesting commands to run,
always prefix with `source .venv/bin/activate` if context is unclear.
