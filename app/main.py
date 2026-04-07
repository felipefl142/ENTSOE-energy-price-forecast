import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Energy Price Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("Energy Price Forecast — DE_LU Market Zone")
st.caption("24-hour ahead electricity spot price forecasting | DuckDB · Feast · ZenML · LightGBM")

tab_forecast, tab_models, tab_eda, tab_sql = st.tabs([
    "Forecast",
    "Model Comparison",
    "EDA",
    "DuckDB Console",
])

with tab_forecast:
    from app.tab_forecast import render_forecast
    render_forecast()

with tab_models:
    from app.tab_model_comparison import render_model_comparison
    render_model_comparison()

with tab_eda:
    from app.tab_eda import render_eda
    render_eda()

with tab_sql:
    from app.tab_duckdb import render_duckdb
    render_duckdb()
