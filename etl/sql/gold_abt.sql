-- Gold layer: Analytical Base Table (ABT).
-- One row per timestamp_utc (prediction moment t).
-- Features come from the silver layer at time t.
-- 24 target columns are the actual prices at t+1h through t+24h (self-joined).
--
-- Parameters:
--   {silver_path}  — data/silver/features.parquet

WITH spine AS (
    SELECT *
    FROM read_parquet('{silver_path}')
),

-- Pre-compute the origin timestamps for each future hour to enable self-joins.
-- origin_Nh is the timestamp from which price_t_plus_Nh would be forecast.
price_targets AS (
    SELECT
        timestamp_utc AS target_ts,
        price_eur_mwh AS target_price,

        timestamp_utc - INTERVAL '1'  HOUR AS origin_1h,
        timestamp_utc - INTERVAL '2'  HOUR AS origin_2h,
        timestamp_utc - INTERVAL '3'  HOUR AS origin_3h,
        timestamp_utc - INTERVAL '4'  HOUR AS origin_4h,
        timestamp_utc - INTERVAL '5'  HOUR AS origin_5h,
        timestamp_utc - INTERVAL '6'  HOUR AS origin_6h,
        timestamp_utc - INTERVAL '7'  HOUR AS origin_7h,
        timestamp_utc - INTERVAL '8'  HOUR AS origin_8h,
        timestamp_utc - INTERVAL '9'  HOUR AS origin_9h,
        timestamp_utc - INTERVAL '10' HOUR AS origin_10h,
        timestamp_utc - INTERVAL '11' HOUR AS origin_11h,
        timestamp_utc - INTERVAL '12' HOUR AS origin_12h,
        timestamp_utc - INTERVAL '13' HOUR AS origin_13h,
        timestamp_utc - INTERVAL '14' HOUR AS origin_14h,
        timestamp_utc - INTERVAL '15' HOUR AS origin_15h,
        timestamp_utc - INTERVAL '16' HOUR AS origin_16h,
        timestamp_utc - INTERVAL '17' HOUR AS origin_17h,
        timestamp_utc - INTERVAL '18' HOUR AS origin_18h,
        timestamp_utc - INTERVAL '19' HOUR AS origin_19h,
        timestamp_utc - INTERVAL '20' HOUR AS origin_20h,
        timestamp_utc - INTERVAL '21' HOUR AS origin_21h,
        timestamp_utc - INTERVAL '22' HOUR AS origin_22h,
        timestamp_utc - INTERVAL '23' HOUR AS origin_23h,
        timestamp_utc - INTERVAL '24' HOUR AS origin_24h

    FROM read_parquet('{silver_path}')
)

SELECT
    s.*,

    -- Target columns: actual future prices (the labels for supervised learning)
    t1.target_price  AS price_t_plus_1h,
    t2.target_price  AS price_t_plus_2h,
    t3.target_price  AS price_t_plus_3h,
    t4.target_price  AS price_t_plus_4h,
    t5.target_price  AS price_t_plus_5h,
    t6.target_price  AS price_t_plus_6h,
    t7.target_price  AS price_t_plus_7h,
    t8.target_price  AS price_t_plus_8h,
    t9.target_price  AS price_t_plus_9h,
    t10.target_price AS price_t_plus_10h,
    t11.target_price AS price_t_plus_11h,
    t12.target_price AS price_t_plus_12h,
    t13.target_price AS price_t_plus_13h,
    t14.target_price AS price_t_plus_14h,
    t15.target_price AS price_t_plus_15h,
    t16.target_price AS price_t_plus_16h,
    t17.target_price AS price_t_plus_17h,
    t18.target_price AS price_t_plus_18h,
    t19.target_price AS price_t_plus_19h,
    t20.target_price AS price_t_plus_20h,
    t21.target_price AS price_t_plus_21h,
    t22.target_price AS price_t_plus_22h,
    t23.target_price AS price_t_plus_23h,
    t24.target_price AS price_t_plus_24h

FROM spine s
LEFT JOIN price_targets t1  ON t1.origin_1h  = s.timestamp_utc
LEFT JOIN price_targets t2  ON t2.origin_2h  = s.timestamp_utc
LEFT JOIN price_targets t3  ON t3.origin_3h  = s.timestamp_utc
LEFT JOIN price_targets t4  ON t4.origin_4h  = s.timestamp_utc
LEFT JOIN price_targets t5  ON t5.origin_5h  = s.timestamp_utc
LEFT JOIN price_targets t6  ON t6.origin_6h  = s.timestamp_utc
LEFT JOIN price_targets t7  ON t7.origin_7h  = s.timestamp_utc
LEFT JOIN price_targets t8  ON t8.origin_8h  = s.timestamp_utc
LEFT JOIN price_targets t9  ON t9.origin_9h  = s.timestamp_utc
LEFT JOIN price_targets t10 ON t10.origin_10h = s.timestamp_utc
LEFT JOIN price_targets t11 ON t11.origin_11h = s.timestamp_utc
LEFT JOIN price_targets t12 ON t12.origin_12h = s.timestamp_utc
LEFT JOIN price_targets t13 ON t13.origin_13h = s.timestamp_utc
LEFT JOIN price_targets t14 ON t14.origin_14h = s.timestamp_utc
LEFT JOIN price_targets t15 ON t15.origin_15h = s.timestamp_utc
LEFT JOIN price_targets t16 ON t16.origin_16h = s.timestamp_utc
LEFT JOIN price_targets t17 ON t17.origin_17h = s.timestamp_utc
LEFT JOIN price_targets t18 ON t18.origin_18h = s.timestamp_utc
LEFT JOIN price_targets t19 ON t19.origin_19h = s.timestamp_utc
LEFT JOIN price_targets t20 ON t20.origin_20h = s.timestamp_utc
LEFT JOIN price_targets t21 ON t21.origin_21h = s.timestamp_utc
LEFT JOIN price_targets t22 ON t22.origin_22h = s.timestamp_utc
LEFT JOIN price_targets t23 ON t23.origin_23h = s.timestamp_utc
LEFT JOIN price_targets t24 ON t24.origin_24h = s.timestamp_utc

-- Only keep rows where all 24 future targets are known (no partial training rows)
WHERE t24.target_price IS NOT NULL

ORDER BY s.timestamp_utc
