"""
ZenML training pipeline.
Runs: load data → train model → evaluate → save artifacts.

Usage:
    zenml init  (run once in project root)
    zenml up    (start dashboard at http://127.0.0.1:8237)
    python -c "from pipelines.training_pipeline import training_pipeline; training_pipeline()"
"""

import io
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from zenml import step, pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
ABT_TRAIN_PATH = str(BASE_DIR / "data" / "gold" / "abt_train.parquet")
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@step
def load_train_data() -> pd.DataFrame:
    """Load training ABT from gold layer via DuckDB."""
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_TRAIN_PATH}')").fetchdf()
    con.close()
    print(f"Loaded {len(df):,} training rows, {len(df.columns)} columns")
    return df


@step
def train_model(train_df: pd.DataFrame) -> bytes:
    """Train LightGBM MultiOutputRegressor. Returns joblib-serialized model bytes."""
    from ml.train import train_lgbm_multioutput
    model = train_lgbm_multioutput(train_df)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


@step
def evaluate_model(model_bytes: bytes) -> dict:
    """Evaluate model on OOT test set. Returns metrics dict."""
    from ml.evaluate import evaluate_oot
    model = joblib.load(io.BytesIO(model_bytes))
    metrics = evaluate_oot(model)
    print(f"Mean MAE (all horizons): {metrics['mae_mean_all']:.4f} EUR/MWh")
    return metrics


@step
def save_model(model_bytes: bytes, metrics: dict) -> None:
    """Persist model to models/ and write metrics.json."""
    MODELS_DIR.mkdir(exist_ok=True)
    model = joblib.load(io.BytesIO(model_bytes))
    joblib.dump(model, MODELS_DIR / "lgbm_multioutput.pkl")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODELS_DIR / 'lgbm_multioutput.pkl'}")
    print(f"Metrics saved to {MODELS_DIR / 'metrics.json'}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pipeline
def training_pipeline():
    train_df = load_train_data()
    model_bytes = train_model(train_df)
    metrics = evaluate_model(model_bytes)
    save_model(model_bytes, metrics)


if __name__ == "__main__":
    training_pipeline()
