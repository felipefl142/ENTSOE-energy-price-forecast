"""
Data ingestion: ENTSO-E (day-ahead prices + load) and Open-Meteo (weather).
Outputs hive-partitioned Parquet under data/raw/{source}/year=YYYY/month=MM/.

Usage:
    python -m etl.collect --start 2018-01-01 --end 2024-12-31
    python -m etl.collect --start 2025-01-01 --end 2025-12-31 --force
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from entsoe import EntsoePandasClient
from retry_requests import retry
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

MARKET_ZONE = "DE_LU"
# Frankfurt proxy for weather (central Germany)
WEATHER_LAT = 50.11
WEATHER_LON = 8.68


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_range(start: datetime, end: datetime):
    """Yield (year, month) tuples from start to end inclusive."""
    cur = start.replace(day=1)
    while cur <= end:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)


def _partition_path(source: str, year: int, month: int) -> Path:
    return RAW_DIR / source / f"year={year}" / f"month={month:02d}" / f"{source}.parquet"


def _already_collected(source: str, year: int, month: int, force: bool) -> bool:
    path = _partition_path(source, year, month)
    if path.exists() and not force:
        print(f"  [{source}] {year}-{month:02d} already collected, skipping.")
        return True
    return False


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class CollectPrices:
    def __init__(self, api_key: str):
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_month(self, year: int, month: int) -> pd.DataFrame:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        if month == 12:
            end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        else:
            end = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC")

        series = self.client.query_day_ahead_prices(MARKET_ZONE, start=start, end=end)
        df = series.reset_index()
        df.columns = ["timestamp_utc", "price_eur_mwh"]
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["market_zone"] = MARKET_ZONE
        return df

    def process(self, start: datetime, end: datetime, force: bool = False):
        print("\n[Prices] Starting collection...")
        for year, month in tqdm(list(_month_range(start, end)), desc="Prices"):
            if _already_collected("prices", year, month, force):
                continue
            try:
                df = self.fetch_month(year, month)
                out = _partition_path("prices", year, month)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                print(f"  [Prices] {year}-{month:02d}: {len(df)} rows → {out}")
                time.sleep(1)
            except Exception as e:
                print(f"  [Prices] {year}-{month:02d} ERROR: {e}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class CollectLoad:
    def __init__(self, api_key: str):
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_month(self, year: int, month: int) -> pd.DataFrame:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        if month == 12:
            end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        else:
            end = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC")

        series = self.client.query_load(MARKET_ZONE, start=start, end=end)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        df = series.reset_index()
        df.columns = ["timestamp_utc", "load_mw"]
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df

    def process(self, start: datetime, end: datetime, force: bool = False):
        print("\n[Load] Starting collection...")
        for year, month in tqdm(list(_month_range(start, end)), desc="Load"):
            if _already_collected("load", year, month, force):
                continue
            try:
                df = self.fetch_month(year, month)
                out = _partition_path("load", year, month)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                print(f"  [Load] {year}-{month:02d}: {len(df)} rows → {out}")
                time.sleep(1)
            except Exception as e:
                print(f"  [Load] {year}-{month:02d} ERROR: {e}")


# ---------------------------------------------------------------------------
# Weather (Open-Meteo archive, free, no key needed)
# ---------------------------------------------------------------------------

WEATHER_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "surface_pressure",
]

WEATHER_COL_RENAME = {"shortwave_radiation": "solar_radiation"}


class CollectWeather:
    def __init__(self):
        cache_session = requests_cache.CachedSession(".weather_cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def fetch_month(self, year: int, month: int) -> pd.DataFrame:
        if month == 12:
            end_year, end_month = year + 1, 1
        else:
            end_year, end_month = year, month + 1

        start_date = f"{year}-{month:02d}-01"
        end_date = f"{end_year}-{end_month:02d}-01"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": WEATHER_VARIABLES,
            "timezone": "UTC",
        }
        responses = self.client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        data = {"timestamp_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )}
        for i, var in enumerate(WEATHER_VARIABLES):
            col = WEATHER_COL_RENAME.get(var, var)
            data[col] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data)
        # Keep only rows up to (but not including) the first day of next month
        cutoff = pd.Timestamp(end_year, end_month, 1, tz="UTC")
        df = df[df["timestamp_utc"] < cutoff].copy()
        return df

    def process(self, start: datetime, end: datetime, force: bool = False):
        print("\n[Weather] Starting collection...")
        for year, month in tqdm(list(_month_range(start, end)), desc="Weather"):
            if _already_collected("weather", year, month, force):
                continue
            try:
                df = self.fetch_month(year, month)
                out = _partition_path("weather", year, month)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                print(f"  [Weather] {year}-{month:02d}: {len(df)} rows → {out}")
            except Exception as e:
                print(f"  [Weather] {year}-{month:02d} ERROR: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect energy data from ENTSO-E + Open-Meteo")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--force", "-f", action="store_true", help="Re-download existing partitions")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError("ENTSOE_API_KEY not set. Check your .env file.")

    print("=" * 60)
    print(f"Collecting data: {args.start} → {args.end}")
    print("=" * 60)

    CollectPrices(api_key).process(start, end, force=args.force)
    CollectLoad(api_key).process(start, end, force=args.force)
    CollectWeather().process(start, end, force=args.force)

    print("\nCollection complete.")


if __name__ == "__main__":
    main()
