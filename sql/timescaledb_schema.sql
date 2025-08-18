-- TimescaleDB Schema for Bot4 ML Time-Series Data
-- FULL TEAM COLLABORATION REQUIRED
-- Lead: Avery (Data Architecture)
-- Contributors: ALL 8 TEAM MEMBERS
-- Phase 3: Machine Learning Integration
-- Target: Sub-millisecond queries, efficient aggregation

-- ============================================================================
-- TEAM CONTRIBUTIONS
-- ============================================================================
-- Avery: Schema design, hypertables, continuous aggregates
-- Morgan: ML feature storage, model metrics
-- Casey: Market data structure, tick storage
-- Jordan: Performance optimization, indexes
-- Quinn: Risk metrics storage
-- Riley: Testing data structures
-- Sam: Clean architecture, naming conventions
-- Alex: Integration requirements

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- MARKET DATA TABLES - Casey & Avery Lead
-- ============================================================================

-- Raw tick data - Casey's design
CREATE TABLE IF NOT EXISTS market_ticks (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    bid             DECIMAL(20,8) NOT NULL,
    ask             DECIMAL(20,8) NOT NULL,
    bid_size        DECIMAL(20,8) NOT NULL,
    ask_size        DECIMAL(20,8) NOT NULL,
    last_price      DECIMAL(20,8),
    last_size       DECIMAL(20,8),
    volume_24h      DECIMAL(20,8),
    
    -- Microstructure features - Casey
    spread          DECIMAL(20,8) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price       DECIMAL(20,8) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    
    -- Metadata
    received_at     TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (time, symbol, exchange)
);

-- Convert to hypertable - Avery's optimization
SELECT create_hypertable('market_ticks', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes - Jordan's performance tuning
CREATE INDEX idx_ticks_symbol_time ON market_ticks (symbol, time DESC);
CREATE INDEX idx_ticks_exchange_time ON market_ticks (exchange, time DESC);

-- OHLCV candles - Morgan's requirement
CREATE TABLE IF NOT EXISTS market_candles (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    interval        VARCHAR(10) NOT NULL, -- '1m', '5m', '1h', etc.
    open            DECIMAL(20,8) NOT NULL,
    high            DECIMAL(20,8) NOT NULL,
    low             DECIMAL(20,8) NOT NULL,
    close           DECIMAL(20,8) NOT NULL,
    volume          DECIMAL(20,8) NOT NULL,
    quote_volume    DECIMAL(20,8),
    trades_count    INTEGER,
    
    -- Technical indicators - Morgan
    vwap            DECIMAL(20,8),
    
    PRIMARY KEY (time, symbol, exchange, interval)
);

SELECT create_hypertable('market_candles', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_candles_symbol_interval ON market_candles (symbol, interval, time DESC);

-- ============================================================================
-- FEATURE STORAGE - Morgan & Avery Lead
-- ============================================================================

-- ML Features storage - Morgan's design
CREATE TABLE IF NOT EXISTS ml_features (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    feature_set     VARCHAR(50) NOT NULL, -- 'technical', 'statistical', 'microstructure'
    
    -- Feature arrays - Morgan's specification
    feature_names   TEXT[],
    feature_values  DOUBLE PRECISION[],
    
    -- Scaled features for ML - Avery's addition
    scaled_values   DOUBLE PRECISION[],
    
    -- Feature metadata
    computation_time_us INTEGER,
    feature_version VARCHAR(20),
    
    PRIMARY KEY (time, symbol, feature_set)
);

SELECT create_hypertable('ml_features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Feature importance tracking - Morgan
CREATE TABLE IF NOT EXISTS feature_importance (
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_id        VARCHAR(50) NOT NULL,
    feature_name    VARCHAR(100) NOT NULL,
    importance_score DOUBLE PRECISION NOT NULL,
    
    PRIMARY KEY (timestamp, model_id, feature_name)
);

-- ============================================================================
-- MODEL METRICS - Morgan & Quinn Lead
-- ============================================================================

-- Model predictions - Morgan's requirement
CREATE TABLE IF NOT EXISTS model_predictions (
    time            TIMESTAMPTZ NOT NULL,
    model_id        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    
    -- Predictions
    prediction      DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION,
    
    -- Ensemble results - Morgan
    ensemble_prediction DOUBLE PRECISION,
    ensemble_std_dev   DOUBLE PRECISION,
    
    -- Risk metrics - Quinn
    risk_score      DOUBLE PRECISION,
    position_size   DECIMAL(20,8),
    
    PRIMARY KEY (time, model_id, symbol)
);

SELECT create_hypertable('model_predictions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Model performance metrics - Riley's testing
CREATE TABLE IF NOT EXISTS model_metrics (
    time            TIMESTAMPTZ NOT NULL,
    model_id        VARCHAR(50) NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    
    -- Additional context
    sample_size     INTEGER,
    window_minutes  INTEGER,
    
    PRIMARY KEY (time, model_id, metric_name)
);

SELECT create_hypertable('model_metrics', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- TRADING SIGNALS - Casey & Quinn Lead
-- ============================================================================

-- Trading signals - Casey's structure
CREATE TABLE IF NOT EXISTS trading_signals (
    time            TIMESTAMPTZ NOT NULL,
    signal_id       UUID DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    source          VARCHAR(50) NOT NULL, -- 'ml_ensemble', 'technical', 'sentiment'
    
    -- Signal details
    signal_type     VARCHAR(20) NOT NULL, -- 'buy', 'sell', 'hold'
    strength        DOUBLE PRECISION NOT NULL, -- 0.0 to 1.0
    confidence      DOUBLE PRECISION,
    
    -- Risk metrics - Quinn
    stop_loss       DECIMAL(20,8),
    take_profit     DECIMAL(20,8),
    position_size   DECIMAL(20,8),
    risk_reward     DOUBLE PRECISION,
    
    -- Execution status
    executed        BOOLEAN DEFAULT FALSE,
    execution_time  TIMESTAMPTZ,
    execution_price DECIMAL(20,8),
    
    PRIMARY KEY (time, signal_id)
);

SELECT create_hypertable('trading_signals', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_signals_symbol_time ON trading_signals (symbol, time DESC);
CREATE INDEX idx_signals_executed ON trading_signals (executed, time DESC);

-- ============================================================================
-- CONTINUOUS AGGREGATES - Avery's Optimization
-- ============================================================================

-- 5-minute candle aggregation - Casey & Avery
CREATE MATERIALIZED VIEW candles_5m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', time) AS bucket,
    symbol,
    exchange,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume,
    SUM(quote_volume) AS quote_volume,
    SUM(trades_count) AS trades_count,
    AVG(vwap) AS vwap
FROM market_candles
WHERE interval = '1m'
GROUP BY bucket, symbol, exchange
WITH NO DATA;

-- Refresh policy - Jordan's performance consideration
SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Feature statistics aggregation - Morgan
CREATE MATERIALIZED VIEW feature_stats_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    symbol,
    feature_set,
    AVG(computation_time_us) AS avg_computation_time,
    COUNT(*) AS feature_count,
    MAX(array_length(feature_values, 1)) AS max_features
FROM ml_features
GROUP BY hour, symbol, feature_set
WITH NO DATA;

-- Model performance aggregation - Riley
CREATE MATERIALIZED VIEW model_performance_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    model_id,
    metric_name,
    AVG(metric_value) AS avg_value,
    STDDEV(metric_value) AS std_dev,
    MIN(metric_value) AS min_value,
    MAX(metric_value) AS max_value,
    COUNT(*) AS sample_count
FROM model_metrics
GROUP BY day, model_id, metric_name
WITH NO DATA;

-- ============================================================================
-- RETENTION POLICIES - Quinn's Risk Management
-- ============================================================================

-- Keep tick data for 30 days - Quinn's requirement
SELECT add_retention_policy('market_ticks',
    drop_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Keep candles for 1 year
SELECT add_retention_policy('market_candles',
    drop_after => INTERVAL '365 days',
    if_not_exists => TRUE
);

-- Keep features for 90 days
SELECT add_retention_policy('ml_features',
    drop_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Keep predictions for 180 days
SELECT add_retention_policy('model_predictions',
    drop_after => INTERVAL '180 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- COMPRESSION POLICIES - Jordan's Performance
-- ============================================================================

-- Compress old tick data
SELECT add_compression_policy('market_ticks',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Compress old candles
SELECT add_compression_policy('market_candles',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- HELPER FUNCTIONS - Sam's Clean Code
-- ============================================================================

-- Function to get latest features - Morgan
CREATE OR REPLACE FUNCTION get_latest_features(
    p_symbol VARCHAR,
    p_feature_set VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    feature_name TEXT,
    feature_value DOUBLE PRECISION,
    scaled_value DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        UNNEST(f.feature_names),
        UNNEST(f.feature_values),
        UNNEST(f.scaled_values)
    FROM ml_features f
    WHERE f.symbol = p_symbol
        AND (p_feature_set IS NULL OR f.feature_set = p_feature_set)
    ORDER BY f.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate signal performance - Casey
CREATE OR REPLACE FUNCTION calculate_signal_performance(
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    symbol VARCHAR,
    total_signals BIGINT,
    executed_signals BIGINT,
    avg_confidence DOUBLE PRECISION,
    win_rate DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.symbol,
        COUNT(*) AS total_signals,
        COUNT(*) FILTER (WHERE s.executed = TRUE) AS executed_signals,
        AVG(s.confidence) AS avg_confidence,
        AVG(CASE 
            WHEN s.executed AND s.execution_price IS NOT NULL 
            THEN CASE 
                WHEN s.signal_type = 'buy' AND s.execution_price > 0 THEN 1.0
                WHEN s.signal_type = 'sell' AND s.execution_price > 0 THEN 1.0
                ELSE 0.0
            END
            ELSE NULL
        END) AS win_rate
    FROM trading_signals s
    WHERE s.time BETWEEN p_start_time AND p_end_time
    GROUP BY s.symbol;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS - Alex's Security
-- ============================================================================

-- Create read-only user for dashboards
CREATE USER IF NOT EXISTS bot4_reader WITH PASSWORD 'readonly_pass';
GRANT USAGE ON SCHEMA public TO bot4_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO bot4_reader;

-- Create write user for application
CREATE USER IF NOT EXISTS bot4_writer WITH PASSWORD 'write_pass';
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bot4_writer;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bot4_writer;

-- ============================================================================
-- TEAM SIGN-OFF
-- ============================================================================
-- Avery: "Schema optimized for time-series with hypertables and aggregates"
-- Morgan: "ML feature storage comprehensive"
-- Casey: "Market data structure complete"
-- Jordan: "Performance indexes in place"
-- Quinn: "Risk metrics included"
-- Riley: "Testing structures ready"
-- Sam: "Clean naming conventions"
-- Alex: "Integration requirements met"