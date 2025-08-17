-- Bot4 Trading Platform - Core Database Schema
-- PostgreSQL with TimescaleDB extensions
-- Designed for <150ms simple trades, <750ms ML-enhanced trades

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;

-- ================================================================
-- ACCOUNTS AND AUTHENTICATION
-- ================================================================

CREATE TABLE trading.exchanges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT true,
    api_endpoint VARCHAR(255) NOT NULL,
    ws_endpoint VARCHAR(255),
    rate_limit_per_second INTEGER DEFAULT 10,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE trading.trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exchange_id UUID REFERENCES trading.exchanges(id),
    account_name VARCHAR(100) NOT NULL,
    api_key_encrypted TEXT, -- Encrypted with ring
    api_secret_encrypted TEXT, -- Encrypted with ring
    is_testnet BOOLEAN DEFAULT true,
    is_active BOOLEAN DEFAULT true,
    max_position_size DECIMAL(20, 8) DEFAULT 0.02, -- 2% max per Quinn's risk rules
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(exchange_id, account_name)
);

-- ================================================================
-- MARKET DATA (TimescaleDB Hypertables)
-- ================================================================

CREATE TABLE trading.candles (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    quote_volume DECIMAL(30, 8),
    trades_count INTEGER,
    taker_buy_volume DECIMAL(30, 8),
    taker_buy_quote_volume DECIMAL(30, 8),
    close_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, open_time)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trading.candles', 'open_time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX idx_candles_symbol_time ON trading.candles (symbol, open_time DESC);
CREATE INDEX idx_candles_timeframe ON trading.candles (timeframe, open_time DESC);

CREATE TABLE trading.orderbook_snapshots (
    id UUID DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    exchange_id UUID REFERENCES trading.exchanges(id),
    timestamp TIMESTAMPTZ NOT NULL,
    bids JSONB NOT NULL, -- [{price, quantity}]
    asks JSONB NOT NULL, -- [{price, quantity}]
    bid_total DECIMAL(30, 8),
    ask_total DECIMAL(30, 8),
    spread DECIMAL(20, 8),
    mid_price DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable('trading.orderbook_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- ================================================================
-- ORDERS AND POSITIONS
-- ================================================================

CREATE TYPE trading.order_status AS ENUM (
    'pending',
    'submitted',
    'partial_filled',
    'filled',
    'cancelled',
    'rejected',
    'expired'
);

CREATE TYPE trading.order_side AS ENUM ('buy', 'sell');
CREATE TYPE trading.order_type AS ENUM ('market', 'limit', 'stop_loss', 'take_profit');

CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading.trading_accounts(id),
    exchange_order_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side trading.order_side NOT NULL,
    order_type trading.order_type NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    status trading.order_status DEFAULT 'pending',
    filled_quantity DECIMAL(30, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),
    
    -- Risk management fields (per Quinn's requirements)
    position_size_pct DECIMAL(5, 4), -- % of portfolio
    risk_amount DECIMAL(20, 8), -- Max loss in base currency
    stop_loss_price DECIMAL(20, 8), -- Mandatory per Quinn
    take_profit_price DECIMAL(20, 8),
    
    -- Strategy metadata
    strategy_id VARCHAR(50),
    signal_strength DECIMAL(5, 4), -- 0-1 confidence
    ml_prediction JSONB, -- ML model outputs
    ta_signals JSONB, -- TA indicator values
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Performance tracking
    latency_ms INTEGER, -- Track our <150ms target
    
    CONSTRAINT check_position_size CHECK (position_size_pct <= 0.02) -- 2% max
);

CREATE INDEX idx_orders_account_symbol ON trading.orders (account_id, symbol, created_at DESC);
CREATE INDEX idx_orders_status ON trading.orders (status) WHERE status IN ('pending', 'submitted');
CREATE INDEX idx_orders_strategy ON trading.orders (strategy_id, created_at DESC);

CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading.trading_accounts(id),
    symbol VARCHAR(20) NOT NULL,
    side trading.order_side NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    
    -- P&L tracking
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 4),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    
    -- Risk management
    stop_loss_price DECIMAL(20, 8) NOT NULL, -- Mandatory
    take_profit_price DECIMAL(20, 8),
    max_drawdown DECIMAL(20, 8),
    
    -- Strategy tracking
    strategy_id VARCHAR(50),
    entry_signals JSONB,
    
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(account_id, symbol) -- One position per symbol per account
);

-- ================================================================
-- STRATEGY AND ML MODELS
-- ================================================================

CREATE TABLE trading.strategies (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    is_active BOOLEAN DEFAULT false,
    
    -- Configuration
    config JSONB NOT NULL,
    risk_params JSONB NOT NULL,
    
    -- Performance metrics (updated daily)
    total_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),
    avg_profit DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    
    -- APY tracking (per Nexus's realistic targets)
    apy_30d DECIMAL(10, 4),
    apy_90d DECIMAL(10, 4),
    apy_365d DECIMAL(10, 4),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE trading.ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'lightgbm', etc.
    version VARCHAR(20) NOT NULL,
    
    -- Model artifacts
    model_path TEXT, -- Path to saved model
    features JSONB NOT NULL, -- Feature list
    hyperparameters JSONB,
    
    -- Performance (from backtesting)
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    
    -- Latency tracking (per Nexus's benchmarks)
    avg_inference_ms INTEGER, -- Should be <200ms per model
    p99_inference_ms INTEGER,
    
    is_active BOOLEAN DEFAULT false,
    trained_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(model_name, version)
);

-- ================================================================
-- RISK MANAGEMENT (Per Quinn's requirements)
-- ================================================================

CREATE TABLE trading.risk_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading.trading_accounts(id),
    
    -- Position limits
    max_position_size DECIMAL(5, 4) DEFAULT 0.02, -- 2% default
    max_positions INTEGER DEFAULT 10,
    max_correlation DECIMAL(3, 2) DEFAULT 0.70, -- Max 0.7 correlation
    
    -- Loss limits
    daily_loss_limit DECIMAL(20, 8),
    weekly_loss_limit DECIMAL(20, 8),
    max_drawdown_pct DECIMAL(5, 4) DEFAULT 0.15, -- 15% max drawdown
    
    -- Mandatory stop-loss
    require_stop_loss BOOLEAN DEFAULT true,
    default_stop_loss_pct DECIMAL(5, 4) DEFAULT 0.02, -- 2% default SL
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(account_id)
);

CREATE TABLE trading.risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading.trading_accounts(id),
    event_type VARCHAR(50) NOT NULL, -- 'drawdown', 'correlation', 'position_size', etc.
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'critical'
    
    description TEXT,
    metrics JSONB,
    
    -- Circuit breaker integration
    circuit_breaker_triggered BOOLEAN DEFAULT false,
    circuit_breaker_state VARCHAR(20),
    
    occurred_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- ================================================================
-- PERFORMANCE TRACKING
-- ================================================================

CREATE TABLE trading.trade_performance (
    id UUID DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES trading.orders(id),
    account_id UUID REFERENCES trading.trading_accounts(id),
    
    symbol VARCHAR(20) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    quantity DECIMAL(30, 8) NOT NULL,
    
    -- P&L
    pnl DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    commission_total DECIMAL(20, 8),
    net_pnl DECIMAL(20, 8),
    
    -- Timing
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    hold_time_minutes INTEGER,
    
    -- Strategy attribution
    strategy_id VARCHAR(50),
    ml_contribution DECIMAL(5, 4), -- 0-1, how much ML influenced
    ta_contribution DECIMAL(5, 4), -- 0-1, how much TA influenced
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Composite primary key including time partition column
    PRIMARY KEY (id, entry_time)
);

SELECT create_hypertable('trading.trade_performance', 'entry_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

-- ================================================================
-- SYSTEM MONITORING
-- ================================================================

CREATE TABLE trading.system_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    tags JSONB,
    PRIMARY KEY (metric_name, timestamp)
);

SELECT create_hypertable('trading.system_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- ================================================================
-- HELPER FUNCTIONS
-- ================================================================

-- Function to calculate portfolio value
CREATE OR REPLACE FUNCTION trading.calculate_portfolio_value(p_account_id UUID)
RETURNS DECIMAL AS $$
DECLARE
    total_value DECIMAL(20, 8) := 0;
BEGIN
    SELECT COALESCE(SUM(quantity * current_price), 0)
    INTO total_value
    FROM trading.positions
    WHERE account_id = p_account_id
    AND closed_at IS NULL;
    
    RETURN total_value;
END;
$$ LANGUAGE plpgsql;

-- Function to check risk limits
CREATE OR REPLACE FUNCTION trading.check_risk_limits()
RETURNS TRIGGER AS $$
DECLARE
    v_max_position_size DECIMAL(5, 4);
    v_portfolio_value DECIMAL(20, 8);
BEGIN
    -- Get risk limits
    SELECT max_position_size INTO v_max_position_size
    FROM trading.risk_limits
    WHERE account_id = NEW.account_id;
    
    -- Check position size
    IF NEW.position_size_pct > COALESCE(v_max_position_size, 0.02) THEN
        RAISE EXCEPTION 'Position size % exceeds limit %', 
            NEW.position_size_pct, v_max_position_size;
    END IF;
    
    -- Check stop loss is set (Quinn's requirement)
    IF NEW.stop_loss_price IS NULL THEN
        RAISE EXCEPTION 'Stop loss is mandatory for all positions';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_order_risk_limits
    BEFORE INSERT OR UPDATE ON trading.orders
    FOR EACH ROW
    EXECUTE FUNCTION trading.check_risk_limits();

-- ================================================================
-- INITIAL DATA
-- ================================================================

-- Insert supported exchanges
INSERT INTO trading.exchanges (name, api_endpoint, ws_endpoint, rate_limit_per_second)
VALUES 
    ('binance', 'https://api.binance.com', 'wss://stream.binance.com:9443', 20),
    ('binance_testnet', 'https://testnet.binance.vision', 'wss://testnet.binance.vision', 20),
    ('kraken', 'https://api.kraken.com', 'wss://ws.kraken.com', 10),
    ('coinbase', 'https://api.exchange.coinbase.com', 'wss://ws-feed.exchange.coinbase.com', 10);

-- Create indexes for performance
CREATE INDEX idx_candles_latest ON trading.candles (symbol, timeframe, open_time DESC);
CREATE INDEX idx_orders_pending ON trading.orders (status, created_at) 
    WHERE status IN ('pending', 'submitted');
CREATE INDEX idx_positions_open ON trading.positions (account_id, closed_at) 
    WHERE closed_at IS NULL;

-- Grant permissions
GRANT ALL ON SCHEMA trading TO bot3user;
GRANT ALL ON ALL TABLES IN SCHEMA trading TO bot3user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA trading TO bot3user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA trading TO bot3user;