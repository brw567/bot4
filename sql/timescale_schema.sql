-- TimescaleDB Schema Definition
-- Task: 8.5.1 - REAL market data schema
-- Optimized for high-frequency trading data

-- =====================================================
-- MARKET DATA SCHEMA
-- =====================================================

-- Trades/Ticks table - Raw trade data from exchanges
CREATE TABLE IF NOT EXISTS market_data.trades (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    side CHAR(1) CHECK (side IN ('B', 'S', 'U')), -- Buy/Sell/Unknown
    trade_id BIGINT,
    order_type VARCHAR(10),
    conditions TEXT[],
    -- Audit fields
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    source VARCHAR(20),
    -- Constraints
    CONSTRAINT trades_positive_price CHECK (price > 0),
    CONSTRAINT trades_positive_volume CHECK (volume > 0)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'market_data.trades',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for common queries
CREATE INDEX idx_trades_exchange_symbol_time 
    ON market_data.trades (exchange, symbol, time DESC);
CREATE INDEX idx_trades_symbol_time 
    ON market_data.trades (symbol, time DESC);

-- Order book snapshots - Full book state at points in time
CREATE TABLE IF NOT EXISTS market_data.order_books (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bids JSONB NOT NULL, -- Array of {price, volume, count}
    asks JSONB NOT NULL, -- Array of {price, volume, count}
    bid_volume NUMERIC(20, 8),
    ask_volume NUMERIC(20, 8),
    spread NUMERIC(20, 8),
    mid_price NUMERIC(20, 8),
    imbalance NUMERIC(5, 4), -- -1 to 1
    sequence_number BIGINT,
    -- Audit fields
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    checksum VARCHAR(64)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_data.order_books',
    'time',
    chunk_time_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

-- Indexes for order books
CREATE INDEX idx_order_books_exchange_symbol_time 
    ON market_data.order_books (exchange, symbol, time DESC);

-- Order book updates - Incremental changes
CREATE TABLE IF NOT EXISTS market_data.book_updates (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side CHAR(1) NOT NULL CHECK (side IN ('B', 'S')),
    action VARCHAR(10) NOT NULL CHECK (action IN ('ADD', 'UPDATE', 'DELETE')),
    price NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8),
    count INTEGER,
    position INTEGER,
    sequence_number BIGINT
);

-- Convert to hypertable with smaller chunks for updates
SELECT create_hypertable(
    'market_data.book_updates',
    'time',
    chunk_time_interval => INTERVAL '2 hours',
    if_not_exists => TRUE
);

-- Best bid/ask tracking
CREATE TABLE IF NOT EXISTS market_data.best_quotes (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bid_price NUMERIC(20, 8),
    bid_volume NUMERIC(20, 8),
    ask_price NUMERIC(20, 8),
    ask_volume NUMERIC(20, 8),
    spread NUMERIC(20, 8),
    mid_price NUMERIC(20, 8)
);

SELECT create_hypertable(
    'market_data.best_quotes',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Exchange status tracking
CREATE TABLE IF NOT EXISTS market_data.exchange_status (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('ONLINE', 'DEGRADED', 'MAINTENANCE', 'OFFLINE')),
    latency_ms INTEGER,
    message_rate INTEGER,
    error_rate NUMERIC(5, 4),
    details JSONB
);

SELECT create_hypertable(
    'market_data.exchange_status',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =====================================================
-- AGGREGATES SCHEMA
-- =====================================================

-- OHLCV data for multiple timeframes
CREATE TABLE IF NOT EXISTS aggregates.ohlcv (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d'
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    buy_volume NUMERIC(20, 8),
    sell_volume NUMERIC(20, 8),
    trades INTEGER,
    vwap NUMERIC(20, 8),
    -- Constraints
    CONSTRAINT ohlcv_price_check CHECK (
        high >= low AND 
        high >= open AND 
        high >= close AND 
        low <= open AND 
        low <= close
    ),
    CONSTRAINT ohlcv_positive_volume CHECK (volume >= 0)
);

SELECT create_hypertable(
    'aggregates.ohlcv',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Composite index for fast queries
CREATE INDEX idx_ohlcv_composite 
    ON aggregates.ohlcv (exchange, symbol, timeframe, time DESC);

-- Volume profile aggregates
CREATE TABLE IF NOT EXISTS aggregates.volume_profile (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    price_level NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    buy_volume NUMERIC(20, 8),
    sell_volume NUMERIC(20, 8),
    trades INTEGER
);

SELECT create_hypertable(
    'aggregates.volume_profile',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Market statistics
CREATE TABLE IF NOT EXISTS aggregates.market_stats (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    volatility NUMERIC(10, 6),
    skew NUMERIC(10, 6),
    kurtosis NUMERIC(10, 6),
    correlation JSONB, -- Correlation with other symbols
    beta NUMERIC(10, 6),
    sharpe_ratio NUMERIC(10, 6)
);

SELECT create_hypertable(
    'aggregates.market_stats',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =====================================================
-- AUDIT SCHEMA
-- =====================================================

-- Data quality tracking
CREATE TABLE IF NOT EXISTS audit.data_quality (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    check_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    details JSONB,
    gaps_detected INTEGER DEFAULT 0,
    duplicates_detected INTEGER DEFAULT 0,
    anomalies_detected INTEGER DEFAULT 0
);

SELECT create_hypertable(
    'audit.data_quality',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Ingestion metrics
CREATE TABLE IF NOT EXISTS audit.ingestion_metrics (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(50) NOT NULL,
    records_received BIGINT,
    records_processed BIGINT,
    records_failed BIGINT,
    processing_time_ms INTEGER,
    errors JSONB
);

SELECT create_hypertable(
    'audit.ingestion_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =====================================================
-- GRANT PERMISSIONS
-- =====================================================

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA market_data TO bot3_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA aggregates TO bot3_app;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO bot3_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA market_data TO bot3_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA aggregates TO bot3_app;

-- Grant read-only permissions
GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO bot3_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA aggregates TO bot3_reader;

-- Log schema creation complete
DO $$
BEGIN
    RAISE NOTICE 'Schema creation complete at %', NOW();
END
$$;