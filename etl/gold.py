"""
Gold layer: builds the Analytical Base Table (ABT) and temporal train/test split.
Also prepares the Feast-formatted features.parquet for the feature store.

Outputs:
  data/gold/abt.parquet          — full ABT (features + 24 targets)
  data/gold/abt_train.parquet    — rows <= TRAIN_END
  data/gold/abt_test.parquet     — rows >  TRAIN_END (OOT test)
  data/feast/features.parquet    — Feast-formatted offline store

Usage:
    python -m etl.gold
"""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"
FEAST_DIR = BASE_DIR / "data" / "feast"
SQL_DIR = Path(__file__).resolve().parent / "sql"

SILVER_PATH = str(SILVER_DIR / "features.parquet")
ABT_PATH = str(GOLD_DIR / "abt.parquet")
ABT_TRAIN_PATH = str(GOLD_DIR / "abt_train.parquet")
ABT_TEST_PATH = str(GOLD_DIR / "abt_test.parquet")
FEAST_FEATURES_PATH = str(FEAST_DIR / "features.parquet")

# Temporal split: OOT test = everything after this date
TRAIN_END = "2023-12-31 23:00:00+00"


def _row_count(con: duckdb.DuckDBPyConnection, path: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]


def build_gold():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    FEAST_DIR.mkdir(parents=True, exist_ok=True)

    sql_template = (SQL_DIR / "gold_abt.sql").read_text()
    sql = sql_template.replace("{silver_path}", SILVER_PATH)

    con = duckdb.connect()

    # ------------------------------------------------------------------
    # Full ABT
    # ------------------------------------------------------------------
    print("  [Gold] Building abt.parquet...")
    con.execute(f"COPY ({sql}) TO '{ABT_PATH}' (FORMAT PARQUET)")
    n = _row_count(con, ABT_PATH)
    print(f"    → {n:,} rows")

    # ------------------------------------------------------------------
    # Temporal train/test split
    # ------------------------------------------------------------------
    print(f"  [Gold] Splitting at {TRAIN_END}...")

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{ABT_PATH}')
            WHERE timestamp_utc <= TIMESTAMPTZ '{TRAIN_END}'
            ORDER BY timestamp_utc
        ) TO '{ABT_TRAIN_PATH}' (FORMAT PARQUET)
    """)
    n_train = _row_count(con, ABT_TRAIN_PATH)

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{ABT_PATH}')
            WHERE timestamp_utc > TIMESTAMPTZ '{TRAIN_END}'
            ORDER BY timestamp_utc
        ) TO '{ABT_TEST_PATH}' (FORMAT PARQUET)
    """)
    n_test = _row_count(con, ABT_TEST_PATH)

    print(f"    → train: {n_train:,} rows | test (OOT): {n_test:,} rows")

    # ------------------------------------------------------------------
    # Feast-formatted offline store
    # ------------------------------------------------------------------
    print("  [Gold] Preparing Feast features.parquet...")
    df = con.execute(
        f"SELECT * FROM read_parquet('{SILVER_PATH}') ORDER BY timestamp_utc"
    ).fetchdf()
    con.close()

    # Feast requires: event_timestamp (tz-aware), entity join keys, feature columns
    df["event_timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["created"] = datetime.now(timezone.utc)
    df["market_zone"] = "DE_LU"
    df["prediction_ts_utc"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df.to_parquet(FEAST_FEATURES_PATH, index=False)
    print(f"    → {len(df):,} rows → {FEAST_FEATURES_PATH}")
    print("  [Gold] Done.")


if __name__ == "__main__":
    print("Building gold layer...")
    build_gold()
