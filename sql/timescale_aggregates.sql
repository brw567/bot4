-- TimescaleDB Continuous Aggregates
-- Task: 8.5.1 - REAL-time OHLCV generation
-- Automated aggregation for multiple timeframes

-- =====================================================
-- CONTINUOUS AGGREGATES FOR OHLCV
-- =====================================================

-- 1-minute OHLCV from trades
CREATE MATERIALIZED VIEW aggregates.ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS time,
    exchange,
    symbol,
    '1m' AS timeframe,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume,
    SUM(CASE WHEN side = 'B' THEN volume ELSE 0 END) AS buy_volume,
    SUM(CASE WHEN side = 'S' THEN volume ELSE 0 END) AS sell_volume,
    COUNT(*) AS trades,
    SUM(price * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM market_data.trades
GROUP BY time_bucket('1 minute', time), exchange, symbol
WITH NO DATA;

-- 5-minute OHLCV from 1-minute
CREATE MATERIALIZED VIEW aggregates.ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS time,
    exchange,
    symbol,
    '5m' AS timeframe,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(buy_volume) AS buy_volume,
    SUM(sell_volume) AS sell_volume,
    SUM(trades) AS trades,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM aggregates.ohlcv_1m
WHERE timeframe = '1m'
GROUP BY time_bucket('5 minutes', time), exchange, symbol
WITH NO DATA;

-- 15-minute OHLCV
CREATE MATERIALIZED VIEW aggregates.ohlcv_15m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', time) AS time,
    exchange,
    symbol,
    '15m' AS timeframe,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(buy_volume) AS buy_volume,
    SUM(sell_volume) AS sell_volume,
    SUM(trades) AS trades,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM aggregates.ohlcv_1m
WHERE timeframe = '1m'
GROUP BY time_bucket('15 minutes', time), exchange, symbol
WITH NO DATA;

-- 1-hour OHLCV
CREATE MATERIALIZED VIEW aggregates.ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    exchange,
    symbol,
    '1h' AS timeframe,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(buy_volume) AS buy_volume,
    SUM(sell_volume) AS sell_volume,
    SUM(trades) AS trades,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM aggregates.ohlcv_5m
WHERE timeframe = '5m'
GROUP BY time_bucket('1 hour', time), exchange, symbol
WITH NO DATA;

-- 4-hour OHLCV
CREATE MATERIALIZED VIEW aggregates.ohlcv_4h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('4 hours', time) AS time,
    exchange,
    symbol,
    '4h' AS timeframe,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(buy_volume) AS buy_volume,
    SUM(sell_volume) AS sell_volume,
    SUM(trades) AS trades,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM aggregates.ohlcv_1h
WHERE timeframe = '1h'
GROUP BY time_bucket('4 hours', time), exchange, symbol
WITH NO DATA;

-- Daily OHLCV
CREATE MATERIALIZED VIEW aggregates.ohlcv_1d
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS time,
    exchange,
    symbol,
    '1d' AS timeframe,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(buy_volume) AS buy_volume,
    SUM(sell_volume) AS sell_volume,
    SUM(trades) AS trades,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM aggregates.ohlcv_1h
WHERE timeframe = '1h'
GROUP BY time_bucket('1 day', time), exchange, symbol
WITH NO DATA;

-- =====================================================
-- SPREAD & LIQUIDITY AGGREGATES
-- =====================================================

-- 1-minute spread statistics
CREATE MATERIALIZED VIEW aggregates.spread_stats_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS time,
    exchange,
    symbol,
    AVG(spread) AS avg_spread,
    MIN(spread) AS min_spread,
    MAX(spread) AS max_spread,
    STDDEV(spread) AS spread_stddev,
    AVG(bid_volume + ask_volume) AS avg_liquidity,
    AVG(imbalance) AS avg_imbalance
FROM market_data.order_books
GROUP BY time_bucket('1 minute', time), exchange, symbol
WITH NO DATA;

-- =====================================================
-- VOLUME PROFILE AGGREGATES
-- =====================================================

-- Hourly volume profile
CREATE MATERIALIZED VIEW aggregates.volume_profile_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    exchange,
    symbol,
    '1h' AS timeframe,
    WIDTH_BUCKET(price, MIN(price) OVER w, MAX(price) OVER w, 100) AS price_bucket,
    AVG(price) AS price_level,
    SUM(volume) AS volume,
    SUM(CASE WHEN side = 'B' THEN volume ELSE 0 END) AS buy_volume,
    SUM(CASE WHEN side = 'S' THEN volume ELSE 0 END) AS sell_volume,
    COUNT(*) AS trades
FROM market_data.trades
WINDOW w AS (PARTITION BY time_bucket('1 hour', time), exchange, symbol)
GROUP BY time_bucket('1 hour', time), exchange, symbol, price_bucket
WITH NO DATA;

-- =====================================================
-- MARKET MICROSTRUCTURE METRICS
-- =====================================================

-- Order flow imbalance
CREATE MATERIALIZED VIEW aggregates.order_flow_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS time,
    exchange,
    symbol,
    SUM(CASE WHEN side = 'B' THEN volume ELSE 0 END) AS buy_volume,
    SUM(CASE WHEN side = 'S' THEN volume ELSE 0 END) AS sell_volume,
    SUM(CASE WHEN side = 'B' THEN volume ELSE -volume END) AS net_flow,
    SUM(CASE WHEN side = 'B' THEN 1 ELSE 0 END) AS buy_trades,
    SUM(CASE WHEN side = 'S' THEN 1 ELSE 0 END) AS sell_trades,
    AVG(CASE WHEN side = 'B' THEN price END) AS avg_buy_price,
    AVG(CASE WHEN side = 'S' THEN price END) AS avg_sell_price
FROM market_data.trades
GROUP BY time_bucket('1 minute', time), exchange, symbol
WITH NO DATA;

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Function to get latest OHLCV for a symbol
CREATE OR REPLACE FUNCTION aggregates.get_latest_ohlcv(
    p_exchange VARCHAR,
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE(
    time TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT o.time, o.open, o.high, o.low, o.close, o.volume
    FROM aggregates.ohlcv o
    WHERE o.exchange = p_exchange
      AND o.symbol = p_symbol
      AND o.timeframe = p_timeframe
    ORDER BY o.time DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate realized volatility
CREATE OR REPLACE FUNCTION aggregates.calculate_volatility(
    p_exchange VARCHAR,
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_period INTEGER DEFAULT 20
)
RETURNS NUMERIC AS $$
DECLARE
    v_volatility NUMERIC;
BEGIN
    WITH returns AS (
        SELECT 
            LN(close / LAG(close) OVER (ORDER BY time)) AS log_return
        FROM aggregates.ohlcv
        WHERE exchange = p_exchange
          AND symbol = p_symbol
          AND timeframe = p_timeframe
        ORDER BY time DESC
        LIMIT p_period + 1
    )
    SELECT STDDEV(log_return) * SQRT(252) INTO v_volatility
    FROM returns;
    
    RETURN v_volatility;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Grant permissions on continuous aggregates
GRANT SELECT ON ALL TABLES IN SCHEMA aggregates TO bot3_app;
GRANT SELECT ON ALL TABLES IN SCHEMA aggregates TO bot3_reader;

-- Log aggregates creation complete
DO $$
BEGIN
    RAISE NOTICE 'Continuous aggregates created at %', NOW();
END
$$;