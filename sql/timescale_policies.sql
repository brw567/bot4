-- TimescaleDB Automation Policies
-- Task: 8.5.1 - REAL automation for data management
-- Compression, retention, and refresh policies

-- =====================================================
-- CONTINUOUS AGGREGATE REFRESH POLICIES
-- =====================================================

-- Refresh 1-minute OHLCV every minute with 2-minute lag
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_1m',
    start_offset => INTERVAL '3 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Refresh 5-minute OHLCV every 5 minutes
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_5m',
    start_offset => INTERVAL '15 minutes',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Refresh 15-minute OHLCV every 15 minutes
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_15m',
    start_offset => INTERVAL '45 minutes',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- Refresh 1-hour OHLCV every hour
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_1h',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Refresh 4-hour OHLCV every 4 hours
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_4h',
    start_offset => INTERVAL '12 hours',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- Refresh daily OHLCV once per day
SELECT add_continuous_aggregate_policy('aggregates.ohlcv_1d',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Refresh spread statistics every minute
SELECT add_continuous_aggregate_policy('aggregates.spread_stats_1m',
    start_offset => INTERVAL '5 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Refresh order flow every minute
SELECT add_continuous_aggregate_policy('aggregates.order_flow_1m',
    start_offset => INTERVAL '5 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- =====================================================
-- COMPRESSION POLICIES
-- =====================================================

-- Compress trades older than 7 days
SELECT add_compression_policy('market_data.trades',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Compress order books older than 3 days (they're larger)
SELECT add_compression_policy('market_data.order_books',
    compress_after => INTERVAL '3 days',
    if_not_exists => TRUE
);

-- Compress book updates older than 1 day (highest volume)
SELECT add_compression_policy('market_data.book_updates',
    compress_after => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Compress best quotes older than 7 days
SELECT add_compression_policy('market_data.best_quotes',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Compress OHLCV data older than 30 days
SELECT add_compression_policy('aggregates.ohlcv',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Compress audit data older than 30 days
SELECT add_compression_policy('audit.data_quality',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('audit.ingestion_metrics',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- =====================================================
-- RETENTION POLICIES
-- =====================================================

-- Keep raw trades for 1 year
SELECT add_retention_policy('market_data.trades',
    drop_after => INTERVAL '365 days',
    if_not_exists => TRUE
);

-- Keep order book snapshots for 90 days
SELECT add_retention_policy('market_data.order_books',
    drop_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Keep book updates for 30 days (very high volume)
SELECT add_retention_policy('market_data.book_updates',
    drop_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Keep aggregated OHLCV forever (it's compressed)
-- No retention policy for aggregates.ohlcv

-- Keep audit data for 2 years
SELECT add_retention_policy('audit.data_quality',
    drop_after => INTERVAL '730 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('audit.ingestion_metrics',
    drop_after => INTERVAL '730 days',
    if_not_exists => TRUE
);

-- =====================================================
-- CUSTOM JOBS
-- =====================================================

-- Job to update exchange status every minute
CREATE OR REPLACE PROCEDURE market_data.update_exchange_status()
LANGUAGE plpgsql
AS $$
DECLARE
    v_exchange RECORD;
    v_latency INTEGER;
    v_message_rate INTEGER;
    v_error_rate NUMERIC;
BEGIN
    FOR v_exchange IN 
        SELECT DISTINCT exchange 
        FROM market_data.trades 
        WHERE time > NOW() - INTERVAL '1 minute'
    LOOP
        -- Calculate metrics
        SELECT 
            EXTRACT(MILLISECONDS FROM (MAX(ingested_at) - MAX(time)))::INTEGER,
            COUNT(*)::INTEGER,
            0.0 -- Error rate would come from error logs
        INTO v_latency, v_message_rate, v_error_rate
        FROM market_data.trades
        WHERE exchange = v_exchange.exchange
          AND time > NOW() - INTERVAL '1 minute';
        
        -- Insert status
        INSERT INTO market_data.exchange_status (
            time, exchange, status, latency_ms, message_rate, error_rate
        ) VALUES (
            NOW(), 
            v_exchange.exchange, 
            CASE 
                WHEN v_latency < 100 THEN 'ONLINE'
                WHEN v_latency < 1000 THEN 'DEGRADED'
                ELSE 'OFFLINE'
            END,
            v_latency,
            v_message_rate,
            v_error_rate
        );
    END LOOP;
END;
$$;

-- Schedule exchange status update job
SELECT add_job('market_data.update_exchange_status',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Job to check data quality
CREATE OR REPLACE PROCEDURE audit.check_data_quality()
LANGUAGE plpgsql
AS $$
DECLARE
    v_exchange RECORD;
    v_gaps INTEGER;
    v_duplicates INTEGER;
BEGIN
    FOR v_exchange IN 
        SELECT DISTINCT exchange, symbol 
        FROM market_data.trades 
        WHERE time > NOW() - INTERVAL '1 hour'
    LOOP
        -- Check for gaps (simplified)
        WITH time_diffs AS (
            SELECT 
                time - LAG(time) OVER (ORDER BY time) AS diff
            FROM market_data.trades
            WHERE exchange = v_exchange.exchange
              AND symbol = v_exchange.symbol
              AND time > NOW() - INTERVAL '1 hour'
        )
        SELECT COUNT(*) INTO v_gaps
        FROM time_diffs
        WHERE diff > INTERVAL '1 minute';
        
        -- Check for duplicates
        SELECT COUNT(*) - COUNT(DISTINCT (time, trade_id)) INTO v_duplicates
        FROM market_data.trades
        WHERE exchange = v_exchange.exchange
          AND symbol = v_exchange.symbol
          AND time > NOW() - INTERVAL '1 hour';
        
        -- Log quality check
        INSERT INTO audit.data_quality (
            exchange, symbol, check_type, status, gaps_detected, duplicates_detected
        ) VALUES (
            v_exchange.exchange,
            v_exchange.symbol,
            'HOURLY_CHECK',
            CASE WHEN v_gaps = 0 AND v_duplicates = 0 THEN 'OK' ELSE 'WARNING' END,
            v_gaps,
            v_duplicates
        );
    END LOOP;
END;
$$;

-- Schedule data quality check job
SELECT add_job('audit.check_data_quality',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- =====================================================
-- MAINTENANCE JOBS
-- =====================================================

-- Reorder chunks for better compression
SELECT add_job('reorder_chunk',
    schedule_interval => INTERVAL '1 day',
    config => '{"hypertable_name": "trades", "index_name": "trades_exchange_symbol_time_idx"}',
    if_not_exists => TRUE
);

-- Update table statistics for query planner
CREATE OR REPLACE PROCEDURE maintenance.update_statistics()
LANGUAGE plpgsql
AS $$
BEGIN
    ANALYZE market_data.trades;
    ANALYZE market_data.order_books;
    ANALYZE aggregates.ohlcv;
END;
$$;

SELECT add_job('maintenance.update_statistics',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

-- =====================================================
-- MONITORING VIEWS
-- =====================================================

-- View to monitor chunk sizes
CREATE OR REPLACE VIEW monitoring.chunk_sizes AS
SELECT 
    hypertable_name,
    chunk_name,
    pg_size_pretty(total_bytes) AS total_size,
    pg_size_pretty(index_bytes) AS index_size,
    pg_size_pretty(table_bytes) AS table_size,
    range_start,
    range_end
FROM timescaledb_information.chunks
ORDER BY total_bytes DESC;

-- View to monitor compression savings
CREATE OR REPLACE VIEW monitoring.compression_stats AS
SELECT 
    hypertable_name,
    COUNT(*) AS total_chunks,
    COUNT(*) FILTER (WHERE is_compressed) AS compressed_chunks,
    pg_size_pretty(SUM(before_compression_total_bytes)) AS before_compression,
    pg_size_pretty(SUM(after_compression_total_bytes)) AS after_compression,
    ROUND(100 * (1 - SUM(after_compression_total_bytes)::FLOAT / 
                      NULLIF(SUM(before_compression_total_bytes), 0)), 2) AS compression_ratio
FROM timescaledb_information.compression_stats
GROUP BY hypertable_name;

-- View to monitor continuous aggregate refresh status
CREATE OR REPLACE VIEW monitoring.cagg_refresh_status AS
SELECT 
    view_name,
    refresh_interval,
    max_interval_per_job,
    materialization_hypertable_name,
    last_run_started_at,
    last_run_duration,
    last_run_success,
    next_scheduled_run
FROM timescaledb_information.continuous_aggregate_stats;

-- Log policies creation complete
DO $$
BEGIN
    RAISE NOTICE 'Automation policies configured at %', NOW();
    RAISE NOTICE 'Compression will activate after specified intervals';
    RAISE NOTICE 'Retention policies will maintain data lifecycle';
    RAISE NOTICE 'Continuous aggregates will refresh automatically';
END
$$;