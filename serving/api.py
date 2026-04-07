"""
FastAPI serving endpoint for 24-hour ahead electricity price forecasting.

At startup: loads the trained LightGBM model + Feast feature store.
POST /predict: fetches online features from Feast → runs inference → returns 24h forecast.

Usage:
    uvicorn serving.api:app --reload --port 8000
    # Swagger UI: http://localhost:8000/docs
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_REPO = BASE_DIR / "feature_store"

TARGET_COLS = [f"price_t_plus_{i}h" for i in range(1, 25)]

_state: dict = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    market_zone: str = Field(default="DE_LU", description="ENTSO-E market zone code")
    prediction_ts_utc: str = Field(
        ...,
        description="ISO8601 UTC timestamp to forecast FROM, e.g. '2024-01-15T12:00:00Z'",
        examples=["2024-01-15T12:00:00Z"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "market_zone": "DE_LU",
                "prediction_ts_utc": "2024-01-15T12:00:00Z",
            }
        }


class HourlyForecast(BaseModel):
    horizon: str
    price_eur_mwh: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResponse(BaseModel):
    market_zone: str
    prediction_ts_utc: str
    forecasts: list[HourlyForecast]


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _build_feature_vector(market_zone: str, prediction_ts_utc: str) -> pd.DataFrame:
    """Retrieve online features for a given entity from the Feast store."""
    from ml.train import FEATURE_COLS

    store = _state["store"]
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
        entity_rows=[{"market_zone": market_zone, "prediction_ts_utc": prediction_ts_utc}],
    ).to_df()

    for col in FEATURE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = float("nan")

    return feature_df[FEATURE_COLS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load mean model
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run the training pipeline first."
        )
    _state["model"] = joblib.load(str(model_path))

    # Load quantile models (optional — used for uncertainty bands)
    for q_str in ["q10", "q90"]:
        q_path = MODELS_DIR / f"lgbm_{q_str}.pkl"
        if q_path.exists():
            _state[f"model_{q_str}"] = joblib.load(str(q_path))

    # Load Feast store
    from feast import FeatureStore
    _state["store"] = FeatureStore(repo_path=str(FEATURE_STORE_REPO))

    # Load feature column list
    fc_path = MODELS_DIR / "feature_columns.json"
    if fc_path.exists():
        _state["feature_columns"] = json.loads(fc_path.read_text())

    print(f"Model loaded. Quantile models: {'q10' in _state and 'q90' in _state}")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Energy Price Forecast API",
    description="24-hour ahead electricity spot price forecast for DE_LU market zone.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": "LightGBM MultiOutputRegressor",
        "horizons": 24,
        "feature_columns": _state.get("feature_columns", []),
        "quantile_models_loaded": "model_q10" in _state and "model_q90" in _state,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    try:
        features = _build_feature_vector(request.market_zone, request.prediction_ts_utc)

        mean_preds = _state["model"].predict(features)[0]  # shape (24,)

        lower_preds = (
            _state["model_q10"].predict(features)[0]
            if "model_q10" in _state else None
        )
        upper_preds = (
            _state["model_q90"].predict(features)[0]
            if "model_q90" in _state else None
        )

        forecasts = [
            HourlyForecast(
                horizon=f"t+{i + 1}h",
                price_eur_mwh=round(float(mean_preds[i]), 4),
                lower_bound=round(float(lower_preds[i]), 4) if lower_preds is not None else None,
                upper_bound=round(float(upper_preds[i]), 4) if upper_preds is not None else None,
            )
            for i in range(24)
        ]

        return ForecastResponse(
            market_zone=request.market_zone,
            prediction_ts_utc=request.prediction_ts_utc,
            forecasts=forecasts,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
