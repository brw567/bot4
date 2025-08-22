-- PostgreSQL Multi-Core Performance Tuning for Bot4 Trading System
-- Team: Avery (Data) & Alex (Lead)
-- Date: 2024
-- Purpose: Maximize PostgreSQL parallel query performance

-- ============================================================================
-- PARALLEL QUERY SETTINGS (Use ALL cores for trading analytics)
-- ============================================================================

-- Set maximum parallel workers to match CPU cores (12 cores on our system)
ALTER SYSTEM SET max_parallel_workers = 12;

-- Allow up to 8 workers per query for heavy analytical queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 8;

-- Increase total worker processes to support parallel + background tasks
ALTER SYSTEM SET max_worker_processes = 16;

-- ============================================================================
-- MEMORY SETTINGS (Optimize for parallel execution)
-- ============================================================================

-- Each parallel worker needs its own work_mem
-- Set conservatively: 256MB per worker (8 workers * 256MB = 2GB max)
ALTER SYSTEM SET work_mem = '256MB';

-- Maintenance operations can use more memory
ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- Shared buffers for caching (25% of RAM for dedicated DB server)
ALTER SYSTEM SET shared_buffers = '8GB';

-- Effective cache size (total RAM available for caching)
ALTER SYSTEM SET effective_cache_size = '24GB';

-- ============================================================================
-- PARALLEL QUERY COST SETTINGS (Encourage parallel plans)
-- ============================================================================

-- Lower the cost threshold for parallel queries (default 1000)
ALTER SYSTEM SET parallel_setup_cost = 100;
ALTER SYSTEM SET parallel_tuple_cost = 0.01;

-- Minimum table size for parallel scan (default 8MB, reduce to 2MB)
ALTER SYSTEM SET min_parallel_table_scan_size = '2MB';
ALTER SYSTEM SET min_parallel_index_scan_size = '256kB';

-- ============================================================================
-- TRADING-SPECIFIC OPTIMIZATIONS
-- ============================================================================

-- Force parallel execution for aggregate queries (OHLCV calculations)
ALTER SYSTEM SET force_parallel_mode = 'off'; -- Use 'on' for testing only

-- JIT compilation for complex queries (ML feature extraction)
ALTER SYSTEM SET jit = 'on';
ALTER SYSTEM SET jit_above_cost = 100000;
ALTER SYSTEM SET jit_inline_above_cost = 500000;
ALTER SYSTEM SET jit_optimize_above_cost = 500000;

-- ============================================================================
-- CONNECTION POOLING (For high-frequency trading)
-- ============================================================================

-- Maximum connections (use connection pooler like PgBouncer in production)
ALTER SYSTEM SET max_connections = 200;

-- Statement timeout for runaway queries
ALTER SYSTEM SET statement_timeout = '30s';

-- Lock timeout to prevent deadlocks
ALTER SYSTEM SET lock_timeout = '10s';

-- ============================================================================
-- WRITE PERFORMANCE (For order execution)
-- ============================================================================

-- Increase WAL buffers for write-heavy workloads
ALTER SYSTEM SET wal_buffers = '64MB';

-- Checkpoint settings for consistent write performance
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Commit delay for batching (microseconds)
ALTER SYSTEM SET commit_delay = 100;
ALTER SYSTEM SET commit_siblings = 5;

-- ============================================================================
-- STATISTICS AND MONITORING
-- ============================================================================

-- More accurate statistics for query planning
ALTER SYSTEM SET default_statistics_target = 500;

-- Track query performance
ALTER SYSTEM SET track_activities = 'on';
ALTER SYSTEM SET track_counts = 'on';
ALTER SYSTEM SET track_io_timing = 'on';
ALTER SYSTEM SET track_functions = 'all';

-- Log slow queries for analysis
ALTER SYSTEM SET log_min_duration_statement = '100ms';
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- ============================================================================
-- APPLY SETTINGS (Requires restart for some parameters)
-- ============================================================================

-- Reload configuration for non-restart parameters
SELECT pg_reload_conf();

-- Show current parallel settings
SELECT name, setting, unit, category, short_desc
FROM pg_settings 
WHERE name LIKE '%parallel%'
ORDER BY name;

-- Show worker process settings
SELECT name, setting, unit, category, short_desc
FROM pg_settings 
WHERE name LIKE '%worker%'
ORDER BY name;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check if parallel query is being used
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT 
    symbol,
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(close) as avg_price,
    SUM(volume) as total_volume,
    COUNT(*) as tick_count
FROM market_data
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY symbol, hour
ORDER BY symbol, hour;

-- Monitor parallel worker activity
SELECT 
    pid,
    usename,
    application_name,
    backend_type,
    state,
    query
FROM pg_stat_activity
WHERE backend_type LIKE '%parallel%'
ORDER BY pid;

-- Check parallel query statistics
SELECT 
    query,
    calls,
    mean_exec_time,
    max_exec_time,
    total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%Parallel%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- ============================================================================
-- NOTES FOR BOT4 TRADING SYSTEM
-- ============================================================================

/*
Parallel Query Benefits for Trading:
1. OHLCV aggregations: 4-8x speedup with parallel workers
2. Technical indicator calculations: Parallel scans for large windows
3. Backtesting queries: Massive speedup for historical data analysis
4. Risk calculations: Parallel aggregation of positions
5. Market depth analysis: Parallel processing of order book data

Optimal Settings for 12-core system:
- max_parallel_workers = 12 (use all cores)
- max_parallel_workers_per_gather = 8 (leave 4 cores for other tasks)
- work_mem = 256MB (per worker, total 2GB for parallel query)

Monitor with:
- pg_stat_activity for active parallel workers
- EXPLAIN ANALYZE to verify parallel plans
- pg_stat_statements for query performance

Restart required after changing:
- max_worker_processes
- shared_buffers
- max_connections

No restart required:
- max_parallel_workers_per_gather
- work_mem
- parallel costs
*/