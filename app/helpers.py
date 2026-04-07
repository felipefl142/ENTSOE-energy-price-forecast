"""Shared helpers for the Streamlit dashboard."""

import os
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BRONZE_PRICES = str(DATA_DIR / "bronze" / "prices.parquet")
BRONZE_LOAD = str(DATA_DIR / "bronze" / "load.parquet")
BRONZE_WEATHER = str(DATA_DIR / "bronze" / "weather.parquet")
SILVER_FEATURES = str(DATA_DIR / "silver" / "features.parquet")
GOLD_ABT = str(DATA_DIR / "gold" / "abt.parquet")
GOLD_TRAIN = str(DATA_DIR / "gold" / "abt_train.parquet")
GOLD_TEST = str(DATA_DIR / "gold" / "abt_test.parquet")
FEAST_FEATURES = str(DATA_DIR / "feast" / "features.parquet")
PREDICTIONS_DIR = str(DATA_DIR / "predictions")
MODELS_DIR = str(BASE_DIR / "models")

AVAILABLE_TABLES = {
    "Bronze — Prices":      f"read_parquet('{BRONZE_PRICES}')",
    "Bronze — Load":        f"read_parquet('{BRONZE_LOAD}')",
    "Bronze — Weather":     f"read_parquet('{BRONZE_WEATHER}')",
    "Silver — Features":    f"read_parquet('{SILVER_FEATURES}')",
    "Gold — ABT (full)":    f"read_parquet('{GOLD_ABT}')",
    "Gold — ABT (train)":   f"read_parquet('{GOLD_TRAIN}')",
    "Gold — ABT (test)":    f"read_parquet('{GOLD_TEST}')",
    "Feast — Features":     f"read_parquet('{FEAST_FEATURES}')",
}

EXAMPLE_QUERIES = {
    "Price range by year": f"""
SELECT
    EXTRACT(YEAR FROM timestamp_utc) AS year,
    MIN(price_eur_mwh) AS min_price,
    AVG(price_eur_mwh) AS avg_price,
    MAX(price_eur_mwh) AS max_price,
    COUNT(*) AS n_hours
FROM read_parquet('{BRONZE_PRICES}')
GROUP BY 1
ORDER BY 1
""".strip(),

    "Average price by hour of day": f"""
SELECT
    EXTRACT(HOUR FROM timestamp_utc) AS hour_of_day,
    AVG(price_eur_mwh) AS avg_price,
    STDDEV(price_eur_mwh) AS std_price
FROM read_parquet('{BRONZE_PRICES}')
GROUP BY 1
ORDER BY 1
""".strip(),

    "Price vs load correlation": f"""
SELECT
    CORR(p.price_eur_mwh, l.load_mw) AS price_load_corr
FROM read_parquet('{BRONZE_PRICES}') p
JOIN read_parquet('{BRONZE_LOAD}') l USING (timestamp_utc)
""".strip(),

    "Silver feature schema": f"""
DESCRIBE SELECT * FROM read_parquet('{SILVER_FEATURES}')
""".strip(),

    "ABT row counts": f"""
SELECT 'train' AS split, COUNT(*) AS n_rows FROM read_parquet('{GOLD_TRAIN}')
UNION ALL
SELECT 'test',            COUNT(*)            FROM read_parquet('{GOLD_TEST}')
""".strip(),

    "Negative prices count by year": f"""
SELECT
    EXTRACT(YEAR FROM timestamp_utc) AS year,
    COUNT(*) AS n_negative_hours
FROM read_parquet('{BRONZE_PRICES}')
WHERE price_eur_mwh < 0
GROUP BY 1
ORDER BY 1
""".strip(),
}


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()
