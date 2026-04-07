"""Model comparison tab: MAE per horizon from metrics.json."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def render_model_comparison():
    st.header("Model Evaluation")

    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        st.warning("No metrics.json found. Run the training pipeline + evaluation first.")
        return

    metrics = json.loads(metrics_path.read_text())

    # --- MAE per horizon bar chart ---
    st.subheader("MAE per Forecast Horizon (OOT Test Set)")

    horizons = list(range(1, 25))
    mae_values = [metrics.get(f"mae_t_plus_{h}h", None) for h in horizons]

    if all(v is not None for v in mae_values):
        fig = go.Figure(go.Bar(
            x=[f"t+{h}h" for h in horizons],
            y=mae_values,
            marker_color=mae_values,
            marker_colorscale="Blues",
            text=[f"{v:.2f}" for v in mae_values],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Horizon",
            yaxis_title="MAE (EUR/MWh)",
            title="Higher MAE at longer horizons is expected",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Summary metrics ---
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean MAE (all)", f"{metrics.get('mae_mean_all', 0):.3f} €/MWh")
    col2.metric("MAE (t+1h→t+6h)", f"{metrics.get('mae_t_plus_1h_to_6h', 0):.3f} €/MWh")
    col3.metric("MAE (t+7h→t+24h)", f"{metrics.get('mae_t_plus_7h_to_24h', 0):.3f} €/MWh")
    col4.metric("MAPE (mean)", f"{metrics.get('mape_mean_all', 0):.2f}%")

    if "interval_coverage_80pct" in metrics:
        st.metric(
            "Interval Coverage (10th–90th pct)",
            f"{metrics['interval_coverage_80pct'] * 100:.1f}%",
            help="Fraction of actual prices falling within the predicted 80% interval. Target: ~80%."
        )

    if "n_test_rows" in metrics:
        st.caption(f"Evaluated on {metrics['n_test_rows']:,} OOT test rows.")

    # --- Full metrics expander ---
    with st.expander("All metrics (raw JSON)"):
        st.json(metrics)
