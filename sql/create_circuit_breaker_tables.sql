-- Circuit Breaker Persistence Tables
-- Quinn's requirement: Never lose risk state on restart

-- Circuit breaker states
CREATE TABLE IF NOT EXISTS circuit_breaker_states (
    id SERIAL PRIMARY KEY,
    breaker_name VARCHAR(100) NOT NULL UNIQUE,
    breaker_type VARCHAR(50) NOT NULL,  -- risk, trading, exchange, strategy
    is_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    trigger_time TIMESTAMP,
    trigger_reason TEXT,
    trigger_value NUMERIC,  -- The value that triggered it (loss %, count, etc)
    threshold_value NUMERIC,  -- The threshold it exceeded
    cooldown_seconds INTEGER,
    cooldown_expires_at TIMESTAMP,
    risk_level VARCHAR(20),  -- LOW, NORMAL, ELEVATED, HIGH, CRITICAL
    position_limit NUMERIC,  -- Current position size limit (0.0 - 1.0)
    metadata JSONB,  -- Additional breaker-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker trigger history
CREATE TABLE IF NOT EXISTS circuit_breaker_history (
    id SERIAL PRIMARY KEY,
    breaker_name VARCHAR(100) NOT NULL,
    breaker_type VARCHAR(50) NOT NULL,
    trigger_time TIMESTAMP NOT NULL,
    reset_time TIMESTAMP,
    trigger_reason TEXT,
    trigger_value NUMERIC,
    threshold_value NUMERIC,
    risk_level VARCHAR(20),
    market_conditions JSONB,  -- Market state at trigger time
    affected_positions JSONB,  -- Positions affected by the breaker
    actions_taken JSONB,  -- What actions were taken (closed positions, etc)
    manual_override BOOLEAN DEFAULT FALSE,
    override_by VARCHAR(100),
    override_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker configuration
CREATE TABLE IF NOT EXISTS circuit_breaker_config (
    id SERIAL PRIMARY KEY,
    breaker_name VARCHAR(100) NOT NULL UNIQUE,
    breaker_type VARCHAR(50) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    config JSONB NOT NULL,  -- Thresholds, cooldowns, etc per risk level
    escalation_rules JSONB,  -- Rules for escalating alerts
    auto_reset BOOLEAN DEFAULT TRUE,
    reset_conditions JSONB,  -- Conditions for auto-reset
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker patterns - for ML analysis
CREATE TABLE IF NOT EXISTS circuit_breaker_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    breaker_name VARCHAR(100),
    pattern_type VARCHAR(50),  -- frequent_trigger, cascade, market_correlated
    trigger_count INTEGER,
    time_window_hours INTEGER,
    correlation_factors JSONB,  -- What correlates with triggers
    recommendations JSONB,  -- ML-generated recommendations
    confidence_score NUMERIC,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_breaker_states_name ON circuit_breaker_states(breaker_name);
CREATE INDEX idx_breaker_states_triggered ON circuit_breaker_states(is_triggered);
CREATE INDEX idx_breaker_states_expires ON circuit_breaker_states(cooldown_expires_at);
CREATE INDEX idx_breaker_history_name_time ON circuit_breaker_history(breaker_name, trigger_time DESC);
CREATE INDEX idx_breaker_history_type ON circuit_breaker_history(breaker_type);
CREATE INDEX idx_breaker_patterns_name ON circuit_breaker_patterns(breaker_name);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating timestamps
CREATE TRIGGER update_circuit_breaker_states_updated_at BEFORE UPDATE
    ON circuit_breaker_states FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_circuit_breaker_config_updated_at BEFORE UPDATE
    ON circuit_breaker_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initial configuration for core breakers
INSERT INTO circuit_breaker_config (breaker_name, breaker_type, config) VALUES
('risk_engine_main', 'risk', '{
    "LOW": {"max_loss": 0.03, "cooldown": 30, "position_limit": 1.0},
    "NORMAL": {"max_loss": 0.02, "cooldown": 60, "position_limit": 0.8},
    "ELEVATED": {"max_loss": 0.015, "cooldown": 180, "position_limit": 0.6},
    "HIGH": {"max_loss": 0.01, "cooldown": 300, "position_limit": 0.4},
    "CRITICAL": {"max_loss": 0.005, "cooldown": 3600, "position_limit": 0.2}
}'::jsonb),
('trading_integrity_daily', 'trading', '{
    "max_daily_loss_pct": 0.05,
    "max_consecutive_losses": 5,
    "pause_after_circuit_break": 300,
    "auto_reset_after_hours": 24
}'::jsonb),
('exchange_rate_limit', 'exchange', '{
    "max_requests_per_second": 10,
    "cooldown_on_rate_limit": 60,
    "backoff_multiplier": 2
}'::jsonb)
ON CONFLICT (breaker_name) DO NOTHING;