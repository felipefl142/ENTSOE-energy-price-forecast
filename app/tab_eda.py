"""EDA tab: price time series, distributions, and correlations."""

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.helpers import BRONZE_PRICES, BRONZE_LOAD, BRONZE_WEATHER, get_duckdb_connection


def render_eda():
    st.header("Exploratory Data Analysis")

    con = get_duckdb_connection()

    try:
        ts_range = con.execute(
            f"SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM read_parquet('{BRONZE_PRICES}')"
        ).fetchone()
    except Exception:
        st.warning("Bronze layer data not found. Run the ETL pipeline first.")
        con.close()
        return

    min_ts = pd.to_datetime(ts_range[0], utc=True).date()
    max_ts = pd.to_datetime(ts_range[1], utc=True).date()

    # --- Date range selector ---
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=min_ts, min_value=min_ts, max_value=max_ts)
    with col2:
        end_date = st.date_input("To", value=max_ts, min_value=min_ts, max_value=max_ts)

    # --- Price time series ---
    st.subheader("Electricity Spot Price — DE_LU")
    df_price = con.execute(f"""
        SELECT timestamp_utc, price_eur_mwh
        FROM read_parquet('{BRONZE_PRICES}')
        WHERE timestamp_utc BETWEEN TIMESTAMPTZ '{start_date}T00:00:00Z'
                                AND TIMESTAMPTZ '{end_date}T23:59:59Z'
        ORDER BY timestamp_utc
    """).fetchdf()

    if not df_price.empty:
        fig = px.line(df_price, x="timestamp_utc", y="price_eur_mwh",
                      title="Hourly Day-Ahead Price", template="plotly_dark")
        fig.update_layout(xaxis_title="", yaxis_title="EUR/MWh", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(df_price):,} hours | "
            f"min: {df_price.price_eur_mwh.min():.1f} | "
            f"avg: {df_price.price_eur_mwh.mean():.1f} | "
            f"max: {df_price.price_eur_mwh.max():.1f} EUR/MWh | "
            f"negative: {(df_price.price_eur_mwh < 0).sum()} hours"
        )

    # --- Price by hour of day ---
    st.subheader("Price Distribution by Hour of Day")
    df_hour = con.execute(f"""
        SELECT
            EXTRACT(HOUR FROM timestamp_utc) AS hour_of_day,
            price_eur_mwh
        FROM read_parquet('{BRONZE_PRICES}')
        WHERE timestamp_utc BETWEEN TIMESTAMPTZ '{start_date}T00:00:00Z'
                                AND TIMESTAMPTZ '{end_date}T23:59:59Z'
    """).fetchdf()

    if not df_hour.empty:
        fig = px.box(
            df_hour, x="hour_of_day", y="price_eur_mwh",
            title="Price Distribution by Hour (Box Plot)",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_title="Hour (UTC)", yaxis_title="EUR/MWh", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Price vs Load scatter ---
    st.subheader("Price vs Grid Load")
    df_pl = con.execute(f"""
        SELECT p.price_eur_mwh, l.load_mw,
               EXTRACT(HOUR FROM p.timestamp_utc) AS hour
        FROM read_parquet('{BRONZE_PRICES}') p
        JOIN read_parquet('{BRONZE_LOAD}') l USING (timestamp_utc)
        WHERE p.timestamp_utc BETWEEN TIMESTAMPTZ '{start_date}T00:00:00Z'
                                  AND TIMESTAMPTZ '{end_date}T23:59:59Z'
        USING SAMPLE 5000
    """).fetchdf()

    if not df_pl.empty:
        fig = px.scatter(
            df_pl, x="load_mw", y="price_eur_mwh", color="hour",
            title="Price vs Load (sampled)", opacity=0.5,
            template="plotly_dark", color_continuous_scale="Viridis",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Price vs Temperature scatter ---
    st.subheader("Price vs Temperature")
    df_pt = con.execute(f"""
        SELECT p.price_eur_mwh, w.temperature_2m,
               EXTRACT(MONTH FROM p.timestamp_utc) AS month
        FROM read_parquet('{BRONZE_PRICES}') p
        JOIN read_parquet('{BRONZE_WEATHER}') w USING (timestamp_utc)
        WHERE p.timestamp_utc BETWEEN TIMESTAMPTZ '{start_date}T00:00:00Z'
                                  AND TIMESTAMPTZ '{end_date}T23:59:59Z'
        USING SAMPLE 5000
    """).fetchdf()

    if not df_pt.empty:
        fig = px.scatter(
            df_pt, x="temperature_2m", y="price_eur_mwh", color="month",
            title="Price vs Temperature (sampled)", opacity=0.5,
            template="plotly_dark", color_continuous_scale="RdBu_r",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Lag autocorrelations ---
    st.subheader("Price Autocorrelation by Lag")
    df_acf = con.execute(f"""
        SELECT
            lag_n,
            CORR(price_eur_mwh, lagged) AS autocorr
        FROM (
            SELECT
                unnest([1,2,3,6,12,24,48,72,168]) AS lag_n,
                unnest([
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 1)   OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 2)   OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 3)   OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 6)   OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 12)  OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 24)  OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 48)  OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 72)  OVER w),
                    CORR(price_eur_mwh, LAG(price_eur_mwh, 168) OVER w)
                ]) AS autocorr
            FROM (
                SELECT price_eur_mwh,
                    LAG(price_eur_mwh, 1)   OVER w AS l1,
                    LAG(price_eur_mwh, 2)   OVER w AS l2,
                    LAG(price_eur_mwh, 3)   OVER w AS l3,
                    LAG(price_eur_mwh, 6)   OVER w AS l6,
                    LAG(price_eur_mwh, 12)  OVER w AS l12,
                    LAG(price_eur_mwh, 24)  OVER w AS l24,
                    LAG(price_eur_mwh, 48)  OVER w AS l48,
                    LAG(price_eur_mwh, 72)  OVER w AS l72,
                    LAG(price_eur_mwh, 168) OVER w AS l168
                FROM read_parquet('{BRONZE_PRICES}')
                WINDOW w AS (ORDER BY timestamp_utc)
            )
        )
        GROUP BY lag_n ORDER BY lag_n
    """).fetchdf()

    if not df_acf.empty:
        fig = px.bar(
            df_acf, x="lag_n", y="autocorr",
            title="Autocorrelation at Key Lags (24h and 168h seasonality expected)",
            labels={"lag_n": "Lag (hours)", "autocorr": "Pearson Correlation"},
            template="plotly_dark",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    con.close()
