"""
Model evaluation: MAE per horizon + pinball loss for quantile models.
Uses the OOT test set (data/gold/abt_test.parquet) — strictly temporal.

Usage:
    python -m ml.evaluate
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"

ABT_TEST_PATH = str(GOLD_DIR / "abt_test.parquet")


def mean_absolute_error_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE for each of the 24 horizons. Shape: (24,)."""
    return np.abs(y_true - y_pred).mean(axis=0)


def mean_absolute_percentage_error_per_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """MAPE per horizon. Clips near-zero actuals to avoid inf. Shape: (24,)."""
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return (np.abs(y_true - y_pred) / denom).mean(axis=0) * 100.0


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss averaged over all samples and horizons."""
    diff = y_true - y_pred
    return float(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff).mean())


def evaluate_oot(model, test_path: str = ABT_TEST_PATH) -> dict:
    """
    Load OOT test set, run inference, and return metrics dict.
    Returns MAE per horizon + overall summary.
    """
    from ml.train import FEATURE_COLS, TARGET_COLS

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{test_path}')").fetchdf()
    con.close()

    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS].values
    preds = model.predict(X)

    mae_per_h = mean_absolute_error_per_horizon(Y, preds)
    mape_per_h = mean_absolute_percentage_error_per_horizon(Y, preds)

    metrics = {
        f"mae_t_plus_{i + 1}h": float(mae_per_h[i]) for i in range(24)
    }
    metrics["mae_mean_all"] = float(mae_per_h.mean())
    metrics["mae_t_plus_1h_to_6h"] = float(mae_per_h[:6].mean())
    metrics["mae_t_plus_7h_to_24h"] = float(mae_per_h[6:].mean())
    metrics["mape_mean_all"] = float(mape_per_h.mean())
    metrics["n_test_rows"] = int(len(df))

    return metrics


def evaluate_quantile_models(test_path: str = ABT_TEST_PATH) -> dict:
    """Evaluate 10th and 90th quantile models for coverage and pinball loss."""
    from ml.train import FEATURE_COLS, TARGET_COLS

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{test_path}')").fetchdf()
    con.close()

    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS].values

    results = {}
    for q_str, q in [("q10", 0.10), ("q90", 0.90)]:
        model_path = MODELS_DIR / f"lgbm_{q_str}.pkl"
        if not model_path.exists():
            print(f"  Skipping {q_str}: model not found at {model_path}")
            continue
        model = joblib.load(model_path)
        preds = model.predict(X)
        results[f"pinball_loss_{q_str}"] = pinball_loss(Y, preds, q)

    # Interval coverage: fraction of actuals within [q10, q90]
    q10_path = MODELS_DIR / "lgbm_q10.pkl"
    q90_path = MODELS_DIR / "lgbm_q90.pkl"
    if q10_path.exists() and q90_path.exists():
        lower = joblib.load(q10_path).predict(X)
        upper = joblib.load(q90_path).predict(X)
        coverage = float(((Y >= lower) & (Y <= upper)).mean())
        results["interval_coverage_80pct"] = coverage

    return results


def main():
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    print("Loading model...")
    model = joblib.load(model_path)

    print("Evaluating on OOT test set...")
    metrics = evaluate_oot(model)

    print("\nQuantile model evaluation...")
    q_metrics = evaluate_quantile_models()
    metrics.update(q_metrics)

    print("\n--- OOT Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = MODELS_DIR / "metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
