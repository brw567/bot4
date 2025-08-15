-- Bot3 Trading Platform Database Schema
-- Version: 2.0
-- Description: Complete schema for trading platform with all required tables

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Drop existing tables if needed (for development)
-- Comment these out in production
-- DROP TABLE IF EXISTS trades CASCADE;
-- DROP TABLE IF EXISTS positions CASCADE;
-- DROP TABLE IF EXISTS price_history CASCADE;
-- DROP TABLE IF EXISTS orders CASCADE;
-- DROP TABLE IF EXISTS strategies CASCADE;
-- DROP TABLE IF EXISTS risk_metrics CASCADE;
-- DROP TABLE IF EXISTS ml_predictions CASCADE;

-- ====================================
-- Core Trading Tables
-- ====================================

-- Price history table (time-series optimized)
CREATE TABLE IF NOT EXISTS price_history (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trades_count INTEGER,
    bid DECIMAL(20, 8),
    ask DECIMAL(20, 8),
    spread DECIMAL(20, 8),
    PRIMARY KEY (time, symbol, exchange)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('price_history', 'time', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_exchange ON price_history(exchange, time DESC);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(100),
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    size DECIMAL(20, 8) NOT NULL,
    value DECIMAL(20, 8),
    pnl DECIMAL(20, 8) DEFAULT 0,
    pnl_percentage DECIMAL(10, 4) DEFAULT 0,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    trailing_stop BOOLEAN DEFAULT FALSE,
    trailing_distance DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'pending', 'error')),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    risk_score DECIMAL(5, 2),
    correlation_group VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_opened ON positions(opened_at DESC);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id),
    order_id VARCHAR(100),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(100),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    trade_type VARCHAR(20) CHECK (trade_type IN ('entry', 'exit', 'partial', 'stop_loss', 'take_profit')),
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8) DEFAULT 0,
    fee_currency VARCHAR(20),
    slippage DECIMAL(10, 6),
    pnl DECIMAL(20, 8),
    pnl_percentage DECIMAL(10, 4),
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_executed ON trades(executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) UNIQUE,
    position_id UUID REFERENCES positions(id),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(100),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit', 'trailing_stop')),
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    size DECIMAL(20, 8) NOT NULL,
    filled_size DECIMAL(20, 8) DEFAULT 0,
    remaining_size DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'open', 'partial', 'filled', 'cancelled', 'rejected', 'expired')),
    time_in_force VARCHAR(10) DEFAULT 'GTC',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_position ON orders(position_id);

-- ====================================
-- Strategy & Risk Tables
-- ====================================

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    version VARCHAR(20),
    enabled BOOLEAN DEFAULT TRUE,
    parameters JSONB,
    performance_metrics JSONB,
    risk_limits JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    total_exposure DECIMAL(20, 8),
    position_count INTEGER,
    leverage DECIMAL(10, 4),
    daily_pnl DECIMAL(20, 8),
    daily_pnl_percentage DECIMAL(10, 4),
    drawdown DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    var_95 DECIMAL(20, 8),
    cvar_95 DECIMAL(20, 8),
    correlation_risk DECIMAL(10, 4),
    liquidity_score DECIMAL(10, 4),
    risk_score DECIMAL(10, 4),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp DESC);

-- ====================================
-- ML & Analytics Tables
-- ====================================

-- ML predictions table
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    symbol VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50),
    prediction_value DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    features JSONB,
    actual_value DECIMAL(20, 8),
    error DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_name, created_at DESC);

-- Performance tracking table
CREATE TABLE IF NOT EXISTS performance_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    total_volume DECIMAL(20, 8) DEFAULT 0,
    win_rate DECIMAL(5, 4),
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_tracking(period_start, period_end);

-- ====================================
-- Audit & System Tables
-- ====================================

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    action_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    user_id VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);

-- System events table
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(100),
    message TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events(severity, timestamp DESC);

-- ====================================
-- Views for Common Queries
-- ====================================

-- Current positions view
CREATE OR REPLACE VIEW current_positions AS
SELECT 
    p.*,
    COALESCE(p.pnl, 0) as realized_pnl,
    CASE 
        WHEN p.side = 'long' THEN (p.current_price - p.entry_price) * p.size
        ELSE (p.entry_price - p.current_price) * p.size
    END as unrealized_pnl
FROM positions p
WHERE p.status = 'open';

-- Daily PnL view
CREATE OR REPLACE VIEW daily_pnl AS
SELECT 
    DATE(executed_at) as trade_date,
    SUM(pnl) as total_pnl,
    COUNT(*) as trade_count,
    AVG(pnl_percentage) as avg_pnl_percentage
FROM trades
WHERE pnl IS NOT NULL
GROUP BY DATE(executed_at)
ORDER BY trade_date DESC;

-- Strategy performance view
CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
    strategy_id,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    AVG(pnl_percentage) as avg_pnl_percentage,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade
FROM trades
WHERE strategy_id IS NOT NULL
GROUP BY strategy_id;

-- ====================================
-- Functions and Triggers
-- ====================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Calculate PnL trigger
CREATE OR REPLACE FUNCTION calculate_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.current_price IS NOT NULL AND NEW.entry_price IS NOT NULL THEN
        IF NEW.side = 'long' THEN
            NEW.pnl := (NEW.current_price - NEW.entry_price) * NEW.size;
        ELSE
            NEW.pnl := (NEW.entry_price - NEW.current_price) * NEW.size;
        END IF;
        
        NEW.pnl_percentage := (NEW.pnl / (NEW.entry_price * NEW.size)) * 100;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_pnl_on_update BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION calculate_position_pnl();

-- ====================================
-- Initial Data
-- ====================================

-- Insert default strategies
INSERT INTO strategies (id, name, strategy_type, parameters, risk_limits) VALUES
    ('grid_trading', 'Grid Trading Strategy', 'grid', '{"grid_levels": 10, "grid_spacing": 0.005}', '{"max_position": 0.02, "stop_loss": 0.02}'),
    ('arbitrage', 'Arbitrage Strategy', 'arbitrage', '{"min_spread": 0.001, "execution_time": 100}', '{"max_position": 0.05, "max_slippage": 0.002}'),
    ('ml_momentum', 'ML Momentum Strategy', 'ml', '{"model": "xgboost", "features": 50}', '{"max_position": 0.01, "confidence_threshold": 0.7}')
ON CONFLICT (id) DO NOTHING;

-- ====================================
-- Permissions
-- ====================================

-- Grant appropriate permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bot3user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bot3user;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO bot3user;