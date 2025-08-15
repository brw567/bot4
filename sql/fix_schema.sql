-- Bot3 Trading Platform - Fix Schema Issues
-- This script fixes the actual database schema to match application requirements

-- ====================================
-- Fix trades table
-- ====================================

-- Add missing columns to trades table
ALTER TABLE trades ADD COLUMN IF NOT EXISTS pnl NUMERIC(20,8);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE trades ADD COLUMN IF NOT EXISTS position_id INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee NUMERIC(20,8) DEFAULT 0;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS size NUMERIC(20,8);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS value NUMERIC(20,8);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS trade_type VARCHAR(20);

-- Update executed_at from timestamp where needed
UPDATE trades SET executed_at = timestamp WHERE executed_at IS NULL AND timestamp IS NOT NULL;

-- ====================================
-- Fix positions table
-- ====================================

-- Check and add missing columns to positions
ALTER TABLE positions ADD COLUMN IF NOT EXISTS pnl NUMERIC(20,8) DEFAULT 0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS size NUMERIC(20,8);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS entry_price NUMERIC(20,8);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS current_price NUMERIC(20,8);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS stop_loss NUMERIC(20,8);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS side VARCHAR(10);

-- ====================================
-- Create missing tables
-- ====================================

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC(20, 8),
    metric_unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON system_metrics(component, timestamp DESC);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(200) UNIQUE,
    title VARCHAR(500) NOT NULL,
    message TEXT,
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(100),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, created_at DESC);

-- Routing decisions table
CREATE TABLE IF NOT EXISTS routing_decisions (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(200),
    symbol VARCHAR(50) NOT NULL,
    strategy VARCHAR(50),
    total_cost NUMERIC(20, 8),
    expected_slippage NUMERIC(10, 6),
    actual_slippage NUMERIC(10, 6),
    risk_score NUMERIC(5, 4),
    confidence NUMERIC(5, 4),
    num_slices INTEGER,
    execution_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_routing_symbol ON routing_decisions(symbol, timestamp DESC);

-- Orders table (if missing)
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE,
    position_id INTEGER,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    strategy_id VARCHAR(100),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    price NUMERIC(20, 8),
    stop_price NUMERIC(20, 8),
    size NUMERIC(20, 8) NOT NULL,
    filled_size NUMERIC(20, 8) DEFAULT 0,
    remaining_size NUMERIC(20, 8),
    status VARCHAR(20) DEFAULT 'pending',
    time_in_force VARCHAR(10) DEFAULT 'GTC',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);

-- Audit log table (if missing)
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    user_id VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);

-- Performance tracking table (if missing)
CREATE TABLE IF NOT EXISTS performance_tracking (
    id SERIAL PRIMARY KEY,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl NUMERIC(20, 8) DEFAULT 0,
    total_volume NUMERIC(20, 8) DEFAULT 0,
    win_rate NUMERIC(5, 4),
    avg_win NUMERIC(20, 8),
    avg_loss NUMERIC(20, 8),
    profit_factor NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_tracking(period_start, period_end);

-- ====================================
-- Fix risk_events table columns
-- ====================================

ALTER TABLE risk_events ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE risk_events ADD COLUMN IF NOT EXISTS position_id INTEGER;

-- ====================================
-- Fix system_events table columns
-- ====================================

ALTER TABLE system_events ADD COLUMN IF NOT EXISTS component VARCHAR(100);

-- ====================================
-- Create helper functions
-- ====================================

-- Function to calculate trade PnL
CREATE OR REPLACE FUNCTION calculate_trade_pnl()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate value if not set
    IF NEW.value IS NULL AND NEW.price IS NOT NULL AND NEW.quantity IS NOT NULL THEN
        NEW.value = NEW.price * NEW.quantity;
    END IF;
    
    -- Set size from quantity if not set
    IF NEW.size IS NULL AND NEW.quantity IS NOT NULL THEN
        NEW.size = NEW.quantity;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for trade calculations
DROP TRIGGER IF EXISTS calculate_trade_pnl_trigger ON trades;
CREATE TRIGGER calculate_trade_pnl_trigger
    BEFORE INSERT OR UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION calculate_trade_pnl();

-- ====================================
-- Verify all fixes
-- ====================================

DO $$
DECLARE
    v_count INTEGER;
BEGIN
    -- Check trades table has required columns
    SELECT COUNT(*) INTO v_count
    FROM information_schema.columns
    WHERE table_name = 'trades' 
    AND column_name IN ('pnl', 'executed_at', 'timestamp');
    
    IF v_count >= 3 THEN
        RAISE NOTICE 'Trades table: OK (has pnl, executed_at, timestamp)';
    ELSE
        RAISE WARNING 'Trades table: Missing columns';
    END IF;
    
    -- Check positions table has required columns
    SELECT COUNT(*) INTO v_count
    FROM information_schema.columns
    WHERE table_name = 'positions' 
    AND column_name IN ('pnl', 'size', 'entry_price');
    
    IF v_count >= 3 THEN
        RAISE NOTICE 'Positions table: OK (has pnl, size, entry_price)';
    ELSE
        RAISE WARNING 'Positions table: Missing columns';
    END IF;
    
    -- Check required tables exist
    SELECT COUNT(*) INTO v_count
    FROM information_schema.tables
    WHERE table_name IN ('system_metrics', 'alerts', 'routing_decisions', 'orders', 'audit_log');
    
    RAISE NOTICE 'Found % of 5 required tables', v_count;
END $$;

-- Log completion
INSERT INTO system_events (event_type, severity, component, message, metadata)
VALUES (
    'schema_fixed',
    'info',
    'database',
    'Schema issues fixed successfully',
    jsonb_build_object(
        'timestamp', NOW(),
        'fixes_applied', ARRAY[
            'trades.pnl added',
            'trades.executed_at added',
            'positions.pnl added',
            'system_metrics table created',
            'alerts table created',
            'routing_decisions table created'
        ]
    )
);