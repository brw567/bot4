-- Feature Store Tables for Bot3 Trading Platform
-- Morgan's domain: Advanced feature engineering and management

-- Feature sets table
CREATE TABLE IF NOT EXISTS feature_sets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    features JSONB NOT NULL, -- List of feature names
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metrics JSONB DEFAULT '{}', -- Performance metrics
    version INTEGER DEFAULT 1,
    status VARCHAR(50) DEFAULT 'active', -- active, deprecated, testing
    created_by VARCHAR(100) DEFAULT 'morgan'
);

-- Feature definitions table
CREATE TABLE IF NOT EXISTS feature_definitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL, -- numeric, categorical, binary, text
    source VARCHAR(100) NOT NULL, -- market_data, technical, sentiment, etc.
    calculation TEXT NOT NULL, -- Formula or method
    dependencies JSONB DEFAULT '[]', -- List of dependent features
    importance DECIMAL(10, 6) DEFAULT 0.0,
    version INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Feature values table (for caching computed features)
CREATE TABLE IF NOT EXISTS feature_values (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 8),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, feature_name, timestamp)
);

-- Feature importance history
CREATE TABLE IF NOT EXISTS feature_importance_history (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    importance DECIMAL(10, 6) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metrics JSONB DEFAULT '{}'
);

-- Feature usage tracking
CREATE TABLE IF NOT EXISTS feature_usage (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    strategy_name VARCHAR(100),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    performance_impact DECIMAL(10, 6), -- Impact on model performance
    UNIQUE(feature_name, model_name, strategy_name)
);

-- Feature drift monitoring
CREATE TABLE IF NOT EXISTS feature_drift (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    drift_score DECIMAL(10, 6) NOT NULL,
    baseline_mean DECIMAL(20, 8),
    baseline_std DECIMAL(20, 8),
    current_mean DECIMAL(20, 8),
    current_std DECIMAL(20, 8),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity VARCHAR(20) -- low, medium, high, critical
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_sets_name ON feature_sets(name);
CREATE INDEX IF NOT EXISTS idx_feature_sets_status ON feature_sets(status);

CREATE INDEX IF NOT EXISTS idx_feature_definitions_name ON feature_definitions(name);
CREATE INDEX IF NOT EXISTS idx_feature_definitions_source ON feature_definitions(source);

CREATE INDEX IF NOT EXISTS idx_feature_values_lookup ON feature_values(symbol, feature_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp ON feature_values(timestamp);

CREATE INDEX IF NOT EXISTS idx_feature_importance_history ON feature_importance_history(feature_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feature_importance_model ON feature_importance_history(model_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_feature_usage ON feature_usage(feature_name, model_name);
CREATE INDEX IF NOT EXISTS idx_feature_usage_last ON feature_usage(last_used DESC);

CREATE INDEX IF NOT EXISTS idx_feature_drift ON feature_drift(feature_name, symbol, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_drift_severity ON feature_drift(severity, detected_at DESC);

-- Trigger to update 'updated_at' column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_feature_sets_updated_at BEFORE UPDATE
    ON feature_sets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();