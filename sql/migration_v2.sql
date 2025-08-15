-- Bot3 Trading Platform Database Migration v2.0
-- Purpose: Add missing tables and fix column inconsistencies
-- Date: 2025-01-10
-- Description: This migration reconciles database schema with application requirements

-- ====================================
-- PHASE 1: Add Missing Tables
-- ====================================

-- System metrics table (required by SmartOrderRouter)
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 8),
    metric_unit VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON system_metrics(component, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name, timestamp DESC);

-- Alerts table (required by MonitoringBridge)
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(200) UNIQUE,
    title VARCHAR(500) NOT NULL,
    message TEXT,
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(100),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged, created_at DESC);

-- Routing decisions table (required by SmartOrderRouter)
CREATE TABLE IF NOT EXISTS routing_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(200),
    symbol VARCHAR(50) NOT NULL,
    strategy VARCHAR(50),
    total_cost DECIMAL(20, 8),
    expected_slippage DECIMAL(10, 6),
    actual_slippage DECIMAL(10, 6),
    risk_score DECIMAL(5, 4),
    confidence DECIMAL(5, 4),
    num_slices INTEGER,
    execution_time_ms INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_routing_symbol ON routing_decisions(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_routing_strategy ON routing_decisions(strategy, timestamp DESC);

-- Risk events table (required by StopLossVerifier)
CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    component VARCHAR(100),
    position_id UUID,
    symbol VARCHAR(50),
    risk_metric VARCHAR(100),
    threshold_value DECIMAL(20, 8),
    actual_value DECIMAL(20, 8),
    action_taken VARCHAR(200),
    resolved BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_position ON risk_events(position_id);

-- ====================================
-- PHASE 2: Fix Column Inconsistencies
-- ====================================

-- Check if 'pnl' column exists in positions table
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='positions' 
        AND column_name='pnl'
    ) THEN
        ALTER TABLE positions ADD COLUMN pnl DECIMAL(20, 8) DEFAULT 0;
    END IF;
END $$;

-- Check if 'pnl' column exists in trades table
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='trades' 
        AND column_name='pnl'
    ) THEN
        ALTER TABLE trades ADD COLUMN pnl DECIMAL(20, 8);
    END IF;
END $$;

-- Add 'timestamp' column as alias for 'executed_at' in trades if needed
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='trades' 
        AND column_name='timestamp'
    ) THEN
        ALTER TABLE trades ADD COLUMN timestamp TIMESTAMPTZ;
        -- Copy existing executed_at values to timestamp
        UPDATE trades SET timestamp = executed_at WHERE timestamp IS NULL;
        -- Set default for new records
        ALTER TABLE trades ALTER COLUMN timestamp SET DEFAULT NOW();
    END IF;
END $$;

-- Add 'strategy_id' to trades table if missing
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='trades' 
        AND column_name='strategy_id'
    ) THEN
        ALTER TABLE trades ADD COLUMN strategy_id VARCHAR(100);
    END IF;
END $$;

-- ====================================
-- PHASE 3: Add Monitoring Tables
-- ====================================

-- SLI/SLO tracking table
CREATE TABLE IF NOT EXISTS sli_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 6),
    target_value DECIMAL(10, 6),
    is_meeting_slo BOOLEAN,
    measurement_period VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_sli_metrics_name ON sli_metrics(metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sli_metrics_slo ON sli_metrics(is_meeting_slo, timestamp DESC);

-- Component health table
CREATE TABLE IF NOT EXISTS component_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    latency_ms DECIMAL(10, 2),
    error_rate DECIMAL(5, 4),
    last_check TIMESTAMPTZ DEFAULT NOW(),
    last_healthy TIMESTAMPTZ,
    failure_count INTEGER DEFAULT 0,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_component_health_name ON component_health(component_name, last_check DESC);
CREATE INDEX IF NOT EXISTS idx_component_health_status ON component_health(status);

-- ====================================
-- PHASE 4: Enhanced Trading Tables
-- ====================================

-- Order book snapshots table
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    bid_price DECIMAL(20, 8),
    bid_size DECIMAL(20, 8),
    ask_price DECIMAL(20, 8),
    ask_size DECIMAL(20, 8),
    spread DECIMAL(20, 8),
    mid_price DECIMAL(20, 8),
    imbalance DECIMAL(10, 6),
    depth_10_bps DECIMAL(20, 8),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orderbook_symbol ON order_book_snapshots(symbol, timestamp DESC);

-- Signal history table
CREATE TABLE IF NOT EXISTS signal_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id VARCHAR(100),
    symbol VARCHAR(50) NOT NULL,
    signal_type VARCHAR(50),
    signal_strength DECIMAL(5, 4),
    confidence DECIMAL(5, 4),
    features JSONB,
    action_taken VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signal_history(strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signal_history(symbol, timestamp DESC);

-- ====================================
-- PHASE 5: Create Views for Compatibility
-- ====================================

-- Create view for backward compatibility with 'timestamp' vs 'executed_at'
CREATE OR REPLACE VIEW trades_compat AS
SELECT 
    *,
    COALESCE(timestamp, executed_at) as unified_timestamp
FROM trades;

-- Create performance summary view
CREATE OR REPLACE VIEW performance_summary AS
SELECT 
    DATE_TRUNC('day', executed_at) as trading_day,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(pnl) as daily_pnl,
    AVG(pnl) as avg_pnl_per_trade,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade,
    SUM(fee) as total_fees
FROM trades
WHERE executed_at IS NOT NULL
GROUP BY DATE_TRUNC('day', executed_at)
ORDER BY trading_day DESC;

-- Create active positions view
CREATE OR REPLACE VIEW active_positions_view AS
SELECT 
    p.*,
    (p.current_price - p.entry_price) * p.size * 
        CASE WHEN p.side = 'long' THEN 1 ELSE -1 END as unrealized_pnl,
    ABS(p.current_price - p.entry_price) / p.entry_price as price_change_pct
FROM positions p
WHERE p.status = 'open'
ORDER BY p.opened_at DESC;

-- ====================================
-- PHASE 6: Add Triggers and Functions
-- ====================================

-- Function to update position PnL
CREATE OR REPLACE FUNCTION update_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.current_price IS NOT NULL AND NEW.entry_price IS NOT NULL THEN
        NEW.pnl = (NEW.current_price - NEW.entry_price) * NEW.size * 
                  CASE WHEN NEW.side = 'long' THEN 1 ELSE -1 END;
        NEW.pnl_percentage = (NEW.pnl / (NEW.entry_price * NEW.size)) * 100;
    END IF;
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic PnL calculation
DROP TRIGGER IF EXISTS update_position_pnl_trigger ON positions;
CREATE TRIGGER update_position_pnl_trigger
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_position_pnl();

-- Function to sync timestamp columns
CREATE OR REPLACE FUNCTION sync_trade_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.executed_at IS NOT NULL AND NEW.timestamp IS NULL THEN
        NEW.timestamp = NEW.executed_at;
    ELSIF NEW.timestamp IS NOT NULL AND NEW.executed_at IS NULL THEN
        NEW.executed_at = NEW.timestamp;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to keep timestamps in sync
DROP TRIGGER IF EXISTS sync_trade_timestamps_trigger ON trades;
CREATE TRIGGER sync_trade_timestamps_trigger
    BEFORE INSERT OR UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION sync_trade_timestamps();

-- ====================================
-- PHASE 7: Grant Permissions
-- ====================================

-- Grant permissions to bot3user (adjust username as needed)
DO $$ 
BEGIN
    -- Grant usage on schema
    EXECUTE 'GRANT USAGE ON SCHEMA public TO bot3user';
    
    -- Grant permissions on all tables
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO bot3user';
    
    -- Grant permissions on all sequences
    EXECUTE 'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bot3user';
    
    -- Grant execute on all functions
    EXECUTE 'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO bot3user';
EXCEPTION
    WHEN undefined_object THEN
        -- User doesn't exist, skip grants
        NULL;
END $$;

-- ====================================
-- PHASE 8: Verification Queries
-- ====================================

-- Verify all required tables exist
DO $$
DECLARE
    missing_tables TEXT[] := ARRAY[]::TEXT[];
    required_tables TEXT[] := ARRAY[
        'positions', 'trades', 'orders', 'strategies',
        'risk_metrics', 'system_metrics', 'alerts',
        'routing_decisions', 'risk_events', 'audit_log',
        'system_events', 'ml_predictions', 'performance_tracking'
    ];
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = tbl
        ) THEN
            missing_tables := array_append(missing_tables, tbl);
        END IF;
    END LOOP;
    
    IF array_length(missing_tables, 1) > 0 THEN
        RAISE NOTICE 'Missing tables: %', missing_tables;
    ELSE
        RAISE NOTICE 'All required tables exist ✓';
    END IF;
END $$;

-- Log migration completion
INSERT INTO system_events (event_type, severity, component, message, metadata)
VALUES (
    'migration_completed',
    'info',
    'database',
    'Database migration v2.0 completed successfully',
    jsonb_build_object(
        'version', '2.0',
        'timestamp', NOW(),
        'tables_added', ARRAY['system_metrics', 'alerts', 'routing_decisions', 'risk_events'],
        'columns_fixed', ARRAY['trades.pnl', 'trades.timestamp', 'positions.pnl']
    )
);

-- Display summary
SELECT 
    'Migration v2.0 Complete' as status,
    NOW() as completed_at,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as total_tables,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'public') as total_columns;