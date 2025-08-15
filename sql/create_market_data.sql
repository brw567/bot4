-- Create Market Data Table for ML Training
-- This table stores OHLCV data for model training

-- Market data table for training
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
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
    vwap DECIMAL(20, 8),
    rsi DECIMAL(10, 4),
    macd DECIMAL(20, 8),
    bollinger_upper DECIMAL(20, 8),
    bollinger_lower DECIMAL(20, 8),
    atr DECIMAL(20, 8),
    obv DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, exchange, timestamp)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_exchange ON market_data(exchange, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC);

-- Model registry table for tracking trained models
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    model_path TEXT,
    training_accuracy DECIMAL(5, 4),
    validation_accuracy DECIMAL(5, 4),
    production_accuracy DECIMAL(5, 4),
    overfitting_gap DECIMAL(5, 4),
    parameters JSONB,
    features JSONB,
    training_samples INTEGER,
    training_duration_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    metadata JSONB,
    UNIQUE(model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_registry_deployed ON model_registry(deployed_at DESC) WHERE deployed_at IS NOT NULL;

-- Training history table
CREATE TABLE IF NOT EXISTS training_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    epoch INTEGER,
    train_loss DECIMAL(10, 6),
    val_loss DECIMAL(10, 6),
    train_accuracy DECIMAL(5, 4),
    val_accuracy DECIMAL(5, 4),
    learning_rate DECIMAL(10, 8),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_training_history ON training_history(model_name, timestamp DESC);

-- Feature importance tracking
CREATE TABLE IF NOT EXISTS feature_importance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score DECIMAL(10, 8),
    rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_importance ON feature_importance(model_name, model_version);

-- Insert some sample data for testing
INSERT INTO market_data (symbol, exchange, timestamp, open, high, low, close, volume)
VALUES 
    ('BTCUSDT', 'binance', NOW() - INTERVAL '1 hour', 50000, 50500, 49800, 50200, 1000),
    ('BTCUSDT', 'binance', NOW() - INTERVAL '55 minutes', 50200, 50300, 50000, 50100, 900),
    ('BTCUSDT', 'binance', NOW() - INTERVAL '50 minutes', 50100, 50400, 50050, 50350, 1100),
    ('ETHUSDT', 'binance', NOW() - INTERVAL '1 hour', 3000, 3050, 2980, 3020, 500),
    ('ETHUSDT', 'binance', NOW() - INTERVAL '55 minutes', 3020, 3030, 3000, 3010, 450)
ON CONFLICT (symbol, exchange, timestamp) DO NOTHING;

-- Verify tables created
SELECT 'Market data infrastructure created successfully' as status;