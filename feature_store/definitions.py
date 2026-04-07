"""
Feast feature store definitions for the energy price forecast project.

Entity:   price_point — (market_zone, prediction_ts_utc) pair
Sources:  data/feast/features.parquet (prepared by etl/gold.py)

Feature views (all online=True, ttl=48h):
  price_lag_features  — price lags + load
  calendar_features   — cyclical time encodings + flags
  weather_features    — meteorological variables

Usage:
    cd feature_store
    feast apply
    feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64, String

# ---------------------------------------------------------------------------
# Paths (relative to feature_store/ directory — where feast apply is run)
# ---------------------------------------------------------------------------
FEATURES_PARQUET = "../data/feast/features.parquet"

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
price_point = Entity(
    name="price_point",
    join_keys=["market_zone", "prediction_ts_utc"],
    description="Hourly electricity price observation point (zone + UTC timestamp string)",
)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
features_source = FileSource(
    name="energy_features_source",
    path=FEATURES_PARQUET,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

price_lag_features = FeatureView(
    name="price_lag_features",
    entities=[price_point],
    ttl=timedelta(hours=48),
    schema=[
        Field(name="price_lag_1h",        dtype=Float64),
        Field(name="price_lag_2h",        dtype=Float64),
        Field(name="price_lag_3h",        dtype=Float64),
        Field(name="price_lag_6h",        dtype=Float64),
        Field(name="price_lag_12h",       dtype=Float64),
        Field(name="price_lag_24h",       dtype=Float64),
        Field(name="price_lag_48h",       dtype=Float64),
        Field(name="price_lag_168h",      dtype=Float64),
        Field(name="price_roll_mean_24h", dtype=Float64),
        Field(name="price_roll_std_24h",  dtype=Float64),
        Field(name="price_roll_min_24h",  dtype=Float64),
        Field(name="price_roll_max_24h",  dtype=Float64),
        Field(name="price_roll_mean_7d",  dtype=Float64),
        Field(name="price_roll_std_7d",   dtype=Float64),
        Field(name="price_delta_24h",     dtype=Float64),
        Field(name="price_delta_168h",    dtype=Float64),
        Field(name="load_mw",             dtype=Float64),
        Field(name="load_lag_24h",        dtype=Float64),
        Field(name="load_lag_168h",       dtype=Float64),
        Field(name="load_roll_mean_24h",  dtype=Float64),
    ],
    source=features_source,
    online=True,
)

calendar_features = FeatureView(
    name="calendar_features",
    entities=[price_point],
    ttl=timedelta(hours=48),
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month",       dtype=Int64),
        Field(name="year",        dtype=Int64),
        Field(name="quarter",     dtype=Int64),
        Field(name="is_weekend",  dtype=Int64),
        Field(name="is_holiday",  dtype=Int64),
        Field(name="hour_sin",    dtype=Float64),
        Field(name="hour_cos",    dtype=Float64),
        Field(name="dow_sin",     dtype=Float64),
        Field(name="dow_cos",     dtype=Float64),
        Field(name="month_sin",   dtype=Float64),
        Field(name="month_cos",   dtype=Float64),
    ],
    source=features_source,
    online=True,
)

weather_features = FeatureView(
    name="weather_features",
    entities=[price_point],
    ttl=timedelta(hours=48),
    schema=[
        Field(name="temperature_2m",     dtype=Float64),
        Field(name="wind_speed_10m",     dtype=Float64),
        Field(name="wind_direction_10m", dtype=Float64),
        Field(name="precipitation",      dtype=Float64),
        Field(name="cloud_cover",        dtype=Float64),
        Field(name="solar_radiation",    dtype=Float64),
        Field(name="surface_pressure",   dtype=Float64),
        Field(name="temp_lag_24h",       dtype=Float64),
        Field(name="solar_lag_24h",      dtype=Float64),
    ],
    source=features_source,
    online=True,
)
