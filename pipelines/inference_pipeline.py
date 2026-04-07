"""
ZenML inference pipeline.
Runs: fetch online features → predict 24h → store predictions as Parquet.

Usage:
    python -c "from pipelines.inference_pipeline import inference_pipeline; inference_pipeline()"
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from zenml import step, pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
FEATURE_STORE_REPO = BASE_DIR / "feature_store"

MARKET_ZONE = "DE_LU"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@step
def fetch_latest_features() -> pd.DataFrame:
    """
    Fetch the latest available features from the Feast online store.
    Uses the most recent materialized timestamp for the DE_LU zone.
    """
    from feast import FeatureStore
    from ml.train import FEATURE_COLS

    store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))

    # Use current UTC time as entity key (represents the prediction moment)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")

    entity_rows = [{
        "market_zone": MARKET_ZONE,
        "prediction_ts_utc": now_str,
    }]

    feature_refs = [
        f"price_lag_features:{col}"
        for col in [
            "price_lag_1h", "price_lag_2h", "price_lag_3h", "price_lag_6h",
            "price_lag_12h", "price_lag_24h", "price_lag_48h", "price_lag_168h",
            "price_roll_mean_24h", "price_roll_std_24h", "price_roll_min_24h",
            "price_roll_max_24h", "price_roll_mean_7d", "price_roll_std_7d",
            "price_delta_24h", "price_delta_168h",
            "load_mw", "load_lag_24h", "load_lag_168h", "load_roll_mean_24h",
        ]
    ] + [
        f"calendar_features:{col}"
        for col in [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "is_weekend", "is_holiday",
        ]
    ] + [
        f"weather_features:{col}"
        for col in [
            "temperature_2m", "wind_speed_10m", "solar_radiation",
            "precipitation", "cloud_cover", "surface_pressure",
            "temp_lag_24h", "solar_lag_24h",
        ]
    ]

    feature_df = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_df()

    # Ensure columns are in the canonical FEATURE_COLS order
    for col in FEATURE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = float("nan")

    return feature_df[FEATURE_COLS]


@step
def run_inference(features: pd.DataFrame) -> pd.DataFrame:
    """Load the trained model and predict 24 price horizons."""
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training_pipeline first.")

    model = joblib.load(model_path)
    preds = model.predict(features)

    horizon_cols = [f"price_t_plus_{i}h" for i in range(1, 25)]
    result = pd.DataFrame(preds, columns=horizon_cols)
    result["prediction_ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    result["market_zone"] = MARKET_ZONE
    return result


@step
def store_predictions(predictions: pd.DataFrame) -> None:
    """Write 24h forecast to data/predictions/ as a timestamped Parquet file."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = PREDICTIONS_DIR / f"{ts}_forecast.parquet"
    predictions.to_parquet(out_path, index=False)
    print(f"Forecast saved to {out_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pipeline
def inference_pipeline():
    features = fetch_latest_features()
    predictions = run_inference(features)
    store_predictions(predictions)


if __name__ == "__main__":
    inference_pipeline()
