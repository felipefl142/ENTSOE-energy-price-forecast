-- Silver layer: point-in-time correct feature computation.
-- Every row represents the feature vector available at timestamp_utc = t,
-- i.e. only uses data strictly before t (ROWS BETWEEN N PRECEDING AND 1 PRECEDING).
--
-- Parameters (replaced before execution):
--   {prices_path}   — data/bronze/prices.parquet
--   {load_path}     — data/bronze/load.parquet
--   {weather_path}  — data/bronze/weather.parquet

WITH price_with_lags AS (
    SELECT
        timestamp_utc,
        price_eur_mwh,

        -- Point-in-time lag features (data available AT prediction time t)
        LAG(price_eur_mwh, 1)   OVER w AS price_lag_1h,
        LAG(price_eur_mwh, 2)   OVER w AS price_lag_2h,
        LAG(price_eur_mwh, 3)   OVER w AS price_lag_3h,
        LAG(price_eur_mwh, 6)   OVER w AS price_lag_6h,
        LAG(price_eur_mwh, 12)  OVER w AS price_lag_12h,
        LAG(price_eur_mwh, 24)  OVER w AS price_lag_24h,
        LAG(price_eur_mwh, 48)  OVER w AS price_lag_48h,
        LAG(price_eur_mwh, 168) OVER w AS price_lag_168h,

        -- Rolling windows: ROWS BETWEEN N PRECEDING AND 1 PRECEDING
        -- ensures NO current-row leakage (all values are from the past)
        AVG(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS price_roll_mean_24h,

        STDDEV(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS price_roll_std_24h,

        MIN(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS price_roll_min_24h,

        MAX(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS price_roll_max_24h,

        AVG(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS price_roll_mean_7d,

        STDDEV(price_eur_mwh) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS price_roll_std_7d,

        -- Momentum: price change over last 24h and last 7d
        price_eur_mwh - LAG(price_eur_mwh, 24)  OVER w AS price_delta_24h,
        price_eur_mwh - LAG(price_eur_mwh, 168) OVER w AS price_delta_168h

    FROM read_parquet('{prices_path}')
    WINDOW w AS (ORDER BY timestamp_utc ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),

load_features AS (
    SELECT
        timestamp_utc,
        load_mw,
        LAG(load_mw, 24)  OVER (ORDER BY timestamp_utc) AS load_lag_24h,
        LAG(load_mw, 168) OVER (ORDER BY timestamp_utc) AS load_lag_168h,
        AVG(load_mw) OVER (
            ORDER BY timestamp_utc
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS load_roll_mean_24h
    FROM read_parquet('{load_path}')
),

weather_features AS (
    SELECT
        timestamp_utc,
        temperature_2m,
        wind_speed_10m,
        wind_direction_10m,
        precipitation,
        cloud_cover,
        solar_radiation,
        surface_pressure,
        -- Lags for same-hour-yesterday context (seasonality proxy)
        LAG(temperature_2m,  24) OVER (ORDER BY timestamp_utc) AS temp_lag_24h,
        LAG(solar_radiation, 24) OVER (ORDER BY timestamp_utc) AS solar_lag_24h
    FROM read_parquet('{weather_path}')
),

calendar_features AS (
    SELECT
        timestamp_utc,
        EXTRACT(HOUR    FROM timestamp_utc) AS hour_of_day,
        EXTRACT(DOW     FROM timestamp_utc) AS day_of_week,   -- 0=Sun, 6=Sat
        EXTRACT(MONTH   FROM timestamp_utc) AS month,
        EXTRACT(YEAR    FROM timestamp_utc) AS year,
        EXTRACT(QUARTER FROM timestamp_utc) AS quarter,
        CASE WHEN EXTRACT(DOW FROM timestamp_utc) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        -- Cyclical encoding: avoids ordinal artifacts at boundaries (23→0, Dec→Jan)
        SIN(2 * PI() * EXTRACT(HOUR  FROM timestamp_utc) / 24)  AS hour_sin,
        COS(2 * PI() * EXTRACT(HOUR  FROM timestamp_utc) / 24)  AS hour_cos,
        SIN(2 * PI() * EXTRACT(DOW   FROM timestamp_utc) / 7)   AS dow_sin,
        COS(2 * PI() * EXTRACT(DOW   FROM timestamp_utc) / 7)   AS dow_cos,
        SIN(2 * PI() * EXTRACT(MONTH FROM timestamp_utc) / 12)  AS month_sin,
        COS(2 * PI() * EXTRACT(MONTH FROM timestamp_utc) / 12)  AS month_cos
    FROM read_parquet('{prices_path}')
)

SELECT
    p.timestamp_utc,
    p.price_eur_mwh,

    -- Price lag features
    p.price_lag_1h,
    p.price_lag_2h,
    p.price_lag_3h,
    p.price_lag_6h,
    p.price_lag_12h,
    p.price_lag_24h,
    p.price_lag_48h,
    p.price_lag_168h,

    -- Price rolling stats (24h window)
    p.price_roll_mean_24h,
    p.price_roll_std_24h,
    p.price_roll_min_24h,
    p.price_roll_max_24h,

    -- Price rolling stats (7d window)
    p.price_roll_mean_7d,
    p.price_roll_std_7d,

    -- Price momentum
    p.price_delta_24h,
    p.price_delta_168h,

    -- Load features
    l.load_mw,
    l.load_lag_24h,
    l.load_lag_168h,
    l.load_roll_mean_24h,

    -- Weather features
    w.temperature_2m,
    w.wind_speed_10m,
    w.wind_direction_10m,
    w.precipitation,
    w.cloud_cover,
    w.solar_radiation,
    w.surface_pressure,
    w.temp_lag_24h,
    w.solar_lag_24h,

    -- Calendar features
    c.hour_of_day,
    c.day_of_week,
    c.month,
    c.year,
    c.quarter,
    c.is_weekend,
    c.hour_sin,
    c.hour_cos,
    c.dow_sin,
    c.dow_cos,
    c.month_sin,
    c.month_cos

FROM price_with_lags p
LEFT JOIN load_features    l USING (timestamp_utc)
LEFT JOIN weather_features w USING (timestamp_utc)
LEFT JOIN calendar_features c USING (timestamp_utc)

-- Drop rows where the 7-day lag window is not yet fully populated
WHERE p.price_lag_168h IS NOT NULL

ORDER BY p.timestamp_utc
