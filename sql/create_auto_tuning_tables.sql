-- ============================================================================
-- AUTO-TUNING AND ADAPTIVE PARAMETERS DATABASE SCHEMA
-- Team: Full team DEEP DIVE implementation
-- Alex: "CRITICAL - All auto-tuning parameters MUST be persisted!"
-- ============================================================================

-- Enable extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- ============================================================================
-- 1. ADAPTIVE PARAMETERS TABLE
-- Stores current adaptive parameters that auto-adjust to market conditions
-- ============================================================================
CREATE TABLE IF NOT EXISTS adaptive_parameters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    current_value DECIMAL(20, 8) NOT NULL,
    min_value DECIMAL(20, 8) NOT NULL,
    max_value DECIMAL(20, 8) NOT NULL,
    default_value DECIMAL(20, 8) NOT NULL,
    
    -- Auto-tuning metadata
    last_adjustment TIMESTAMPTZ DEFAULT NOW(),
    adjustment_count INTEGER DEFAULT 0,
    adjustment_reason TEXT,
    
    -- Performance tracking
    performance_impact DECIMAL(10, 4), -- Percentage impact on returns
    stability_score DECIMAL(5, 4),     -- How stable the parameter is (0-1)
    
    -- Market regime association
    optimal_for_regime VARCHAR(20) CHECK (optimal_for_regime IN ('Bull', 'Bear', 'Sideways', 'Crisis')),
    
    -- Metadata
    description TEXT,
    unit VARCHAR(20), -- 'percentage', 'ratio', 'multiplier', etc.
    category VARCHAR(50), -- 'risk', 'execution', 'ml', 'portfolio'
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert critical adaptive parameters with defaults
INSERT INTO adaptive_parameters (parameter_name, current_value, min_value, max_value, default_value, description, unit, category)
VALUES 
    ('var_limit', 0.02, 0.005, 0.04, 0.02, 'Value at Risk limit - maximum daily loss tolerance', 'percentage', 'risk'),
    ('volatility_target', 0.15, 0.10, 0.30, 0.15, 'Target portfolio volatility', 'percentage', 'risk'),
    ('kelly_fraction', 0.25, 0.10, 0.40, 0.25, 'Fraction of Kelly Criterion to use', 'percentage', 'risk'),
    ('leverage_cap', 2.0, 1.0, 5.0, 2.0, 'Maximum leverage allowed', 'multiplier', 'risk'),
    ('position_size_limit', 0.02, 0.005, 0.05, 0.02, 'Maximum size per position', 'percentage', 'risk'),
    ('correlation_threshold', 0.7, 0.5, 0.9, 0.7, 'Maximum correlation between positions', 'ratio', 'portfolio'),
    ('stop_loss_multiplier', 2.0, 1.5, 3.0, 2.0, 'ATR multiplier for stop loss', 'multiplier', 'execution'),
    ('take_profit_multiplier', 3.0, 2.0, 5.0, 3.0, 'ATR multiplier for take profit', 'multiplier', 'execution'),
    ('min_edge_threshold', 0.01, 0.005, 0.02, 0.01, 'Minimum edge required to trade', 'percentage', 'execution'),
    ('ml_confidence_threshold', 0.6, 0.5, 0.8, 0.6, 'Minimum ML confidence to act on signal', 'ratio', 'ml')
ON CONFLICT (parameter_name) DO NOTHING;

-- ============================================================================
-- 2. Q-LEARNING TABLE
-- Stores Q-values for reinforcement learning state-action pairs
-- ============================================================================
CREATE TABLE IF NOT EXISTS q_learning_table (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- State representation
    state_hash VARCHAR(64) NOT NULL, -- Hash of state features
    market_regime VARCHAR(20) NOT NULL CHECK (market_regime IN ('Bull', 'Bear', 'Sideways', 'Crisis')),
    volatility_bucket INTEGER NOT NULL CHECK (volatility_bucket BETWEEN 0 AND 10),
    drawdown_bucket INTEGER NOT NULL CHECK (drawdown_bucket BETWEEN 0 AND 10),
    momentum_bucket INTEGER NOT NULL CHECK (momentum_bucket BETWEEN -5 AND 5),
    
    -- Action (parameter adjustments)
    action_id INTEGER NOT NULL,
    action_description TEXT,
    
    -- Q-value and learning metrics
    q_value DECIMAL(20, 8) NOT NULL DEFAULT 0,
    visit_count INTEGER DEFAULT 0,
    total_reward DECIMAL(20, 8) DEFAULT 0,
    avg_reward DECIMAL(20, 8) DEFAULT 0,
    
    -- Learning parameters
    learning_rate DECIMAL(10, 8) DEFAULT 0.1,
    discount_factor DECIMAL(10, 8) DEFAULT 0.95,
    exploration_rate DECIMAL(10, 8) DEFAULT 0.1,
    
    -- Timestamps
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint on state-action pair
    UNIQUE(state_hash, action_id)
);

CREATE INDEX IF NOT EXISTS idx_q_learning_state ON q_learning_table(state_hash);
CREATE INDEX IF NOT EXISTS idx_q_learning_regime ON q_learning_table(market_regime);
CREATE INDEX IF NOT EXISTS idx_q_learning_q_value ON q_learning_table(q_value DESC);

-- ============================================================================
-- 3. MARKET REGIME HISTORY
-- Tracks detected market regimes over time
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_regime_history (
    id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    regime VARCHAR(20) NOT NULL CHECK (regime IN ('Bull', 'Bear', 'Sideways', 'Crisis')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    
    -- Regime detection inputs
    trend_slope DECIMAL(20, 8),
    average_return DECIMAL(20, 8),
    volatility DECIMAL(20, 8),
    volume_surge BOOLEAN DEFAULT FALSE,
    correlation_stable BOOLEAN DEFAULT TRUE,
    
    -- Market metrics at detection time
    vix_level DECIMAL(10, 4),
    volume_24h DECIMAL(20, 8),
    bid_ask_spread DECIMAL(20, 8),
    
    -- Transition tracking
    previous_regime VARCHAR(20),
    regime_duration_hours INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_regime_history', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_regime_history_regime ON market_regime_history(regime, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_regime_history_confidence ON market_regime_history(confidence DESC);

-- ============================================================================
-- 4. PERFORMANCE FEEDBACK TABLE
-- Records outcomes for reinforcement learning
-- ============================================================================
CREATE TABLE IF NOT EXISTS performance_feedback (
    id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Context
    market_regime VARCHAR(20) NOT NULL,
    position_id UUID,
    signal_id VARCHAR(100),
    
    -- Action taken
    action_type VARCHAR(50), -- 'adjust_var', 'adjust_kelly', 'adjust_leverage', etc.
    old_value DECIMAL(20, 8),
    new_value DECIMAL(20, 8),
    
    -- Outcome
    pnl DECIMAL(20, 8),
    return_pct DECIMAL(10, 6),
    sharpe_contribution DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    
    -- Reward calculation
    immediate_reward DECIMAL(20, 8),
    delayed_reward DECIMAL(20, 8),
    total_reward DECIMAL(20, 8),
    
    -- Success metrics
    success BOOLEAN,
    success_criteria TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('performance_feedback', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feedback_regime ON performance_feedback(market_regime, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_success ON performance_feedback(success, timestamp DESC);

-- ============================================================================
-- 5. AUTO-TUNING AUDIT LOG
-- Complete audit trail of all parameter adjustments
-- ============================================================================
CREATE TABLE IF NOT EXISTS auto_tuning_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- What changed
    parameter_name VARCHAR(100) NOT NULL,
    old_value DECIMAL(20, 8),
    new_value DECIMAL(20, 8),
    change_percentage DECIMAL(10, 6),
    
    -- Why it changed
    trigger_reason VARCHAR(100), -- 'regime_change', 'performance_decay', 'q_learning', 'manual'
    market_regime VARCHAR(20),
    performance_metrics JSONB,
    
    -- Impact assessment
    expected_impact DECIMAL(10, 6),
    actual_impact DECIMAL(10, 6), -- Updated later
    impact_measured_at TIMESTAMPTZ,
    
    -- Metadata
    adjusted_by VARCHAR(50), -- 'auto_tuner', 'ml_system', 'manual', 'emergency'
    rollback_to UUID, -- Reference to previous value if rolled back
    notes TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_parameter ON auto_tuning_audit(parameter_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trigger ON auto_tuning_audit(trigger_reason, timestamp DESC);

-- ============================================================================
-- 6. STRATEGY PERFORMANCE HISTORY
-- Tracks performance metrics for different strategies/parameters
-- ============================================================================
CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    
    -- Performance metrics
    total_pnl DECIMAL(20, 8),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    calmar_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 6),
    
    -- Risk metrics with current parameters
    var_limit_used DECIMAL(20, 8),
    kelly_fraction_used DECIMAL(20, 8),
    leverage_used DECIMAL(20, 8),
    
    -- Market conditions
    market_regime VARCHAR(20),
    volatility_level DECIMAL(20, 8),
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_strategy_performance ON strategy_performance(strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_sharpe ON strategy_performance(sharpe_ratio DESC);

-- ============================================================================
-- 7. FUNCTIONS FOR AUTO-TUNING
-- ============================================================================

-- Function to update adaptive parameter with audit trail
CREATE OR REPLACE FUNCTION update_adaptive_parameter(
    param_name VARCHAR,
    new_val DECIMAL,
    reason TEXT DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    old_val DECIMAL;
BEGIN
    -- Get current value
    SELECT current_value INTO old_val 
    FROM adaptive_parameters 
    WHERE parameter_name = param_name;
    
    -- Update parameter
    UPDATE adaptive_parameters
    SET current_value = new_val,
        last_adjustment = NOW(),
        adjustment_count = adjustment_count + 1,
        adjustment_reason = reason,
        updated_at = NOW()
    WHERE parameter_name = param_name;
    
    -- Create audit entry
    INSERT INTO auto_tuning_audit (
        parameter_name, old_value, new_value, 
        change_percentage, trigger_reason, adjusted_by
    ) VALUES (
        param_name, old_val, new_val,
        ((new_val - old_val) / old_val) * 100,
        reason, 'auto_tuner'
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get optimal parameters for current regime
CREATE OR REPLACE FUNCTION get_regime_parameters(regime VARCHAR)
RETURNS TABLE(
    parameter_name VARCHAR,
    optimal_value DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT ap.parameter_name, ap.current_value
    FROM adaptive_parameters ap
    WHERE ap.optimal_for_regime = regime
       OR ap.optimal_for_regime IS NULL
    ORDER BY ap.category, ap.parameter_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 8. TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_adaptive_parameters_updated_at
    BEFORE UPDATE ON adaptive_parameters
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 9. INITIAL Q-TABLE POPULATION
-- ============================================================================

-- Populate initial Q-table with exploration values
DO $$
DECLARE
    regime VARCHAR;
    vol_bucket INTEGER;
    dd_bucket INTEGER;
    action INTEGER;
BEGIN
    FOR regime IN SELECT unnest(ARRAY['Bull', 'Bear', 'Sideways', 'Crisis'])
    LOOP
        FOR vol_bucket IN 0..10
        LOOP
            FOR dd_bucket IN 0..10
            LOOP
                FOR action IN 0..9  -- 10 possible actions
                LOOP
                    INSERT INTO q_learning_table (
                        state_hash,
                        market_regime,
                        volatility_bucket,
                        drawdown_bucket,
                        momentum_bucket,
                        action_id,
                        q_value
                    ) VALUES (
                        MD5(regime || vol_bucket || dd_bucket || '0')::VARCHAR(64),
                        regime,
                        vol_bucket,
                        dd_bucket,
                        0,  -- neutral momentum
                        action,
                        0.0  -- Initial Q-value
                    ) ON CONFLICT DO NOTHING;
                END LOOP;
            END LOOP;
        END LOOP;
    END LOOP;
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
SELECT 'Auto-tuning database schema created successfully!' as status;

-- List created tables
SELECT table_name, 
       CASE 
           WHEN table_name LIKE '%history' OR table_name LIKE '%feedback' OR table_name LIKE '%performance' 
           THEN 'Time-series (Hypertable)'
           ELSE 'Regular Table'
       END as table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN (
    'adaptive_parameters',
    'q_learning_table', 
    'market_regime_history',
    'performance_feedback',
    'auto_tuning_audit',
    'strategy_performance'
  )
ORDER BY table_name;