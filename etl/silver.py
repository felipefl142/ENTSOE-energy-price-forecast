"""
Silver layer: point-in-time correct feature computation via DuckDB SQL template.
Reads from bronze layer and writes data/silver/features.parquet.

Features include: price lags, rolling stats, momentum, load lags, weather, calendar.
All window functions use ROWS BETWEEN N PRECEDING AND 1 PRECEDING — no look-ahead.

Usage:
    python -m etl.silver
"""

import os
from pathlib import Path

import duckdb
import holidays
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"
SILVER_DIR = BASE_DIR / "data" / "silver"
SQL_DIR = Path(__file__).resolve().parent / "sql"

PRICES_PATH = str(BRONZE_DIR / "prices.parquet")
LOAD_PATH = str(BRONZE_DIR / "load.parquet")
WEATHER_PATH = str(BRONZE_DIR / "weather.parquet")
SILVER_PATH = str(SILVER_DIR / "features.parquet")


def _add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_holiday column (Germany public holidays) using the holidays package."""
    years = df["timestamp_utc"].dt.year.unique().tolist()
    de_holidays = set()
    for y in years:
        de_holidays.update(holidays.Germany(years=y).keys())

    df["is_holiday"] = df["timestamp_utc"].dt.date.astype(str).map(
        lambda d: 1 if d in {str(h) for h in de_holidays} else 0
    ).astype(int)
    return df


def build_silver():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    sql_template = (SQL_DIR / "silver_features.sql").read_text()
    sql = (
        sql_template
        .replace("{prices_path}", PRICES_PATH)
        .replace("{load_path}", LOAD_PATH)
        .replace("{weather_path}", WEATHER_PATH)
    )

    print("  [Silver] Running feature SQL template...")
    con = duckdb.connect()
    df = con.execute(sql).fetchdf()
    con.close()

    print(f"  [Silver] Adding holiday flag...")
    df = _add_holiday_flag(df)

    df.to_parquet(SILVER_PATH, index=False)
    print(f"  [Silver] Done: {len(df):,} rows, {len(df.columns)} columns → {SILVER_PATH}")


if __name__ == "__main__":
    print("Building silver layer...")
    build_silver()
