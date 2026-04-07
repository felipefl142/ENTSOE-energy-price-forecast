"""
Model training: LightGBM MultiOutputRegressor (24 independent models, one per horizon).
Also trains quantile models (10th/90th percentile) for uncertainty bands.

The FEATURE_COLS list is the single source of truth — imported by pipelines and serving.

Usage:
    python -m ml.train
"""

import io
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"

ABT_TRAIN_PATH = str(GOLD_DIR / "abt_train.parquet")

# ---------------------------------------------------------------------------
# Feature and target column definitions
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Price lag features
    "price_lag_1h",
    "price_lag_2h",
    "price_lag_3h",
    "price_lag_6h",
    "price_lag_12h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    # Price rolling stats
    "price_roll_mean_24h",
    "price_roll_std_24h",
    "price_roll_min_24h",
    "price_roll_max_24h",
    "price_roll_mean_7d",
    "price_roll_std_7d",
    # Price momentum
    "price_delta_24h",
    "price_delta_168h",
    # Load
    "load_mw",
    "load_lag_24h",
    "load_lag_168h",
    "load_roll_mean_24h",
    # Weather
    "temperature_2m",
    "wind_speed_10m",
    "solar_radiation",
    "precipitation",
    "cloud_cover",
    "surface_pressure",
    "temp_lag_24h",
    "solar_lag_24h",
    # Calendar (cyclical + flags)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]

TARGET_COLS = [f"price_t_plus_{i}h" for i in range(1, 25)]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _lgbm_base() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _lgbm_quantile(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _build_pipeline(base_model: LGBMRegressor) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", MultiOutputRegressor(base_model, n_jobs=4)),
    ])


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_lgbm_multioutput(train_df: pd.DataFrame) -> Pipeline:
    """Train mean LightGBM MultiOutputRegressor on the training ABT."""
    X = train_df[FEATURE_COLS]
    Y = train_df[TARGET_COLS]
    pipeline = _build_pipeline(_lgbm_base())
    print(f"  Training mean model: {len(X):,} rows × {len(FEATURE_COLS)} features × {len(TARGET_COLS)} targets")
    pipeline.fit(X, Y)
    return pipeline


def train_quantile_models(train_df: pd.DataFrame, quantiles=(0.1, 0.9)) -> dict:
    """Train one MultiOutputRegressor per quantile for prediction intervals."""
    X = train_df[FEATURE_COLS]
    Y = train_df[TARGET_COLS]
    models = {}
    for q in quantiles:
        print(f"  Training quantile model (q={q})...")
        pipe = _build_pipeline(_lgbm_quantile(q))
        pipe.fit(X, Y)
        models[q] = pipe
    return models


# ---------------------------------------------------------------------------
# Direct training entry point
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading training data...")
    con = duckdb.connect()
    train_df = con.execute(f"SELECT * FROM read_parquet('{ABT_TRAIN_PATH}')").fetchdf()
    con.close()
    print(f"  → {len(train_df):,} rows loaded")

    print("\nTraining mean model...")
    model = train_lgbm_multioutput(train_df)
    joblib.dump(model, MODELS_DIR / "lgbm_multioutput.pkl")
    print(f"  → saved to {MODELS_DIR / 'lgbm_multioutput.pkl'}")

    print("\nTraining quantile models (q=0.10, q=0.90)...")
    q_models = train_quantile_models(train_df)
    for q, m in q_models.items():
        q_str = f"q{int(q * 100):02d}"
        joblib.dump(m, MODELS_DIR / f"lgbm_{q_str}.pkl")
        print(f"  → saved to {MODELS_DIR / f'lgbm_{q_str}.pkl'}")

    # Save feature column list for serving
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    print(f"\nDone. Feature list saved to {MODELS_DIR / 'feature_columns.json'}")


if __name__ == "__main__":
    main()
