-- Bot3 Trading Platform Database Schema
-- PostgreSQL initialization script

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;

-- Set default schema
SET search_path TO trading, public;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    order_id VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    consensus_score DECIMAL(5, 4),
    agent_votes JSONB,
    pnl DECIMAL(20, 8),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8) NOT NULL,
    take_profit DECIMAL(20, 8),
    strategy VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    close_price DECIMAL(20, 8),
    close_reason VARCHAR(100),
    pnl DECIMAL(20, 8),
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    close_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ML predictions table
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prediction JSONB NOT NULL,
    confidence DECIMAL(5, 4),
    features JSONB,
    correct BOOLEAN,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk events table
CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vetoed signals table (for analysis)
CREATE TABLE IF NOT EXISTS vetoed_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    agent_votes JSONB,
    reason VARCHAR(200),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    path VARCHAR(500) NOT NULL,
    accuracy DECIMAL(5, 4),
    parameters JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    action VARCHAR(200) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp DESC);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX idx_positions_strategy ON positions(strategy);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_ml_predictions_model_timestamp ON ml_predictions(model, timestamp DESC);
CREATE INDEX idx_risk_events_timestamp ON risk_events(timestamp DESC);
CREATE INDEX idx_risk_events_severity ON risk_events(severity);

-- Create hypertable for time-series data (if TimescaleDB is available)
-- SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
-- SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate PnL
CREATE OR REPLACE FUNCTION calculate_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND NEW.close_price IS NOT NULL THEN
        IF NEW.side = 'buy' THEN
            NEW.pnl = (NEW.close_price - NEW.entry_price) * NEW.quantity;
        ELSE
            NEW.pnl = (NEW.entry_price - NEW.close_price) * NEW.quantity;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_position_pnl BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION calculate_pnl();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO bot3user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO bot3user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO bot3user;

-- Insert initial configuration
INSERT INTO performance_metrics (metric_name, value, metadata)
VALUES 
    ('initial_capital', 10000, '{"currency": "USDT"}'),
    ('target_sharpe_ratio', 1.0, '{"description": "Minimum acceptable Sharpe ratio"}'),
    ('max_drawdown_limit', 0.15, '{"description": "Maximum drawdown threshold"}');

-- Success message
DO $$ 
BEGIN 
    RAISE NOTICE 'Bot3 Trading Platform database initialized successfully!';
END $$;