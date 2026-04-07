"""Forecast tab: 24-hour ahead price forecast with uncertainty bands."""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SILVER_PATH = str(BASE_DIR / "data" / "silver" / "features.parquet")


@st.cache_resource
def _load_models():
    models = {}
    for name in ["lgbm_multioutput", "lgbm_q10", "lgbm_q90"]:
        p = MODELS_DIR / f"{name}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
    return models


def render_forecast():
    st.header("24-Hour Ahead Price Forecast")

    models = _load_models()
    if "lgbm_multioutput" not in models:
        st.warning("No trained model found. Run the training pipeline first.")
        return

    # --- Timestamp selector ---
    con = duckdb.connect()
    try:
        ts_range = con.execute(
            f"SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM read_parquet('{SILVER_PATH}')"
        ).fetchone()
        con.close()
    except Exception:
        con.close()
        st.warning("Silver layer features not found. Run the ETL pipeline first.")
        return

    min_ts = pd.to_datetime(ts_range[0], utc=True)
    max_ts = pd.to_datetime(ts_range[1], utc=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Prediction date",
            value=max_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
        )
    with col2:
        selected_hour = st.slider("Prediction hour (UTC)", 0, 23, int(max_ts.hour))

    pred_ts = f"{selected_date}T{selected_hour:02d}:00:00"

    # --- Load feature vector ---
    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{SILVER_PATH}') "
            f"WHERE timestamp_utc = TIMESTAMPTZ '{pred_ts}+00:00' LIMIT 1"
        ).fetchdf()
        con.close()
    except Exception as e:
        con.close()
        st.error(f"Error loading features: {e}")
        return

    if row.empty:
        st.warning(f"No features found for {pred_ts} UTC. Try a different timestamp.")
        return

    from ml.train import FEATURE_COLS
    X = row[FEATURE_COLS]

    # --- Run inference ---
    mean_preds = models["lgbm_multioutput"].predict(X)[0]
    lower_preds = models["lgbm_q10"].predict(X)[0] if "lgbm_q10" in models else None
    upper_preds = models["lgbm_q90"].predict(X)[0] if "lgbm_q90" in models else None

    horizons = list(range(1, 25))
    forecast_ts = pd.date_range(
        start=pd.Timestamp(pred_ts, tz="UTC") + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
    )

    # --- Plot ---
    fig = go.Figure()

    if lower_preds is not None and upper_preds is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_ts) + list(forecast_ts[::-1]),
            y=list(upper_preds) + list(lower_preds[::-1]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% prediction interval",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=forecast_ts,
        y=mean_preds,
        mode="lines+markers",
        name="Forecast (mean)",
        line=dict(color="#636EFA", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title=f"24h Price Forecast from {pred_ts} UTC — DE_LU",
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Price (EUR/MWh)",
        hovermode="x unified",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Table ---
    st.subheader("Hourly Forecast Table")
    df_out = pd.DataFrame({
        "Horizon": [f"t+{h}h" for h in horizons],
        "Timestamp (UTC)": forecast_ts.strftime("%Y-%m-%d %H:%M"),
        "Forecast (EUR/MWh)": mean_preds.round(2),
    })
    if lower_preds is not None:
        df_out["Lower 10% (EUR/MWh)"] = lower_preds.round(2)
        df_out["Upper 90% (EUR/MWh)"] = upper_preds.round(2)

    st.dataframe(df_out, use_container_width=True, hide_index=True)
