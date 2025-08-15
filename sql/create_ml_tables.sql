-- Create ML-related tables for model registry and training history

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

-- Add missing columns to market_data
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS quote_volume DECIMAL(20, 8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS trades_count INTEGER;
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bid DECIMAL(20, 8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ask DECIMAL(20, 8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS spread DECIMAL(20, 8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS vwap DECIMAL(20, 8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- Verify tables created
SELECT 'ML infrastructure created successfully' as status;