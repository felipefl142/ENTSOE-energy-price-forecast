"""
Bronze layer: type casting, deduplication, and consolidation via DuckDB.
Reads hive-partitioned raw Parquet and writes three clean files:
  data/bronze/prices.parquet
  data/bronze/load.parquet
  data/bronze/weather.parquet

Usage:
    python -m etl.bronze
"""

import os
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def _row_count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]


def build_bronze():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    prices_raw = str(RAW_DIR / "prices" / "**" / "*.parquet")
    prices_out = str(BRONZE_DIR / "prices.parquet")

    print("  [Bronze] Building prices.parquet...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(timestamp_utc AS TIMESTAMPTZ) AS timestamp_utc,
                TRY_CAST(price_eur_mwh AS DOUBLE)  AS price_eur_mwh,
                COALESCE(TRY_CAST(market_zone AS VARCHAR), 'DE_LU') AS market_zone
            FROM read_parquet('{prices_raw}', hive_partitioning=true)
            WHERE timestamp_utc IS NOT NULL
              AND price_eur_mwh IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(timestamp_utc AS TIMESTAMPTZ)
                ORDER BY CAST(timestamp_utc AS TIMESTAMPTZ)
            ) = 1
            ORDER BY timestamp_utc
        ) TO '{prices_out}' (FORMAT PARQUET)
    """)
    n = _row_count(con, Path(prices_out))
    print(f"    → {n:,} rows")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    load_raw = str(RAW_DIR / "load" / "**" / "*.parquet")
    load_out = str(BRONZE_DIR / "load.parquet")

    print("  [Bronze] Building load.parquet...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(timestamp_utc AS TIMESTAMPTZ) AS timestamp_utc,
                TRY_CAST(load_mw AS DOUBLE) AS load_mw
            FROM read_parquet('{load_raw}', hive_partitioning=true)
            WHERE timestamp_utc IS NOT NULL
              AND load_mw IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(timestamp_utc AS TIMESTAMPTZ)
                ORDER BY CAST(timestamp_utc AS TIMESTAMPTZ)
            ) = 1
            ORDER BY timestamp_utc
        ) TO '{load_out}' (FORMAT PARQUET)
    """)
    n = _row_count(con, Path(load_out))
    print(f"    → {n:,} rows")

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    weather_raw = str(RAW_DIR / "weather" / "**" / "*.parquet")
    weather_out = str(BRONZE_DIR / "weather.parquet")

    print("  [Bronze] Building weather.parquet...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(timestamp_utc AS TIMESTAMPTZ) AS timestamp_utc,
                TRY_CAST(temperature_2m AS DOUBLE)      AS temperature_2m,
                TRY_CAST(wind_speed_10m AS DOUBLE)      AS wind_speed_10m,
                TRY_CAST(wind_direction_10m AS DOUBLE)  AS wind_direction_10m,
                TRY_CAST(precipitation AS DOUBLE)       AS precipitation,
                TRY_CAST(cloud_cover AS DOUBLE)         AS cloud_cover,
                TRY_CAST(solar_radiation AS DOUBLE)     AS solar_radiation,
                TRY_CAST(surface_pressure AS DOUBLE)    AS surface_pressure
            FROM read_parquet('{weather_raw}', hive_partitioning=true)
            WHERE timestamp_utc IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(timestamp_utc AS TIMESTAMPTZ)
                ORDER BY CAST(timestamp_utc AS TIMESTAMPTZ)
            ) = 1
            ORDER BY timestamp_utc
        ) TO '{weather_out}' (FORMAT PARQUET)
    """)
    n = _row_count(con, Path(weather_out))
    print(f"    → {n:,} rows")

    con.close()
    print("  [Bronze] Done.")


if __name__ == "__main__":
    print("Building bronze layer...")
    build_bronze()
