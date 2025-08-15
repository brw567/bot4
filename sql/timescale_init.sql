-- TimescaleDB Initialization Script
-- Task: 8.5.1 - REAL Time-series database setup
-- Creates extensions and base configuration

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create application user with limited privileges
CREATE USER bot3_app WITH PASSWORD 'app_secure_pass_change_me';

-- Create read-only user for analytics
CREATE USER bot3_reader WITH PASSWORD 'reader_secure_pass_change_me';

-- Set default configuration
ALTER DATABASE bot3_market_data SET timescaledb.telemetry_level = 'off';
ALTER DATABASE bot3_market_data SET log_min_duration_statement = 1000; -- Log slow queries > 1s

-- Performance settings for the database
ALTER DATABASE bot3_market_data SET random_page_cost = 1.1;
ALTER DATABASE bot3_market_data SET effective_io_concurrency = 200;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS aggregates;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS backtest;

-- Grant schema usage
GRANT USAGE ON SCHEMA market_data TO bot3_app;
GRANT USAGE ON SCHEMA aggregates TO bot3_app;
GRANT USAGE ON SCHEMA audit TO bot3_app;
GRANT USAGE ON SCHEMA backtest TO bot3_app;

GRANT USAGE ON SCHEMA market_data TO bot3_reader;
GRANT USAGE ON SCHEMA aggregates TO bot3_reader;

-- Set search path
ALTER DATABASE bot3_market_data SET search_path TO market_data, aggregates, public;

-- Create audit function for tracking changes
CREATE OR REPLACE FUNCTION audit.track_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.created_at = NOW();
        NEW.updated_at = NOW();
        NEW.created_by = current_user;
        NEW.updated_by = current_user;
    ELSIF TG_OP = 'UPDATE' THEN
        NEW.updated_at = NOW();
        NEW.updated_by = current_user;
        NEW.created_at = OLD.created_at;
        NEW.created_by = OLD.created_by;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Log initialization complete
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB initialization complete at %', NOW();
END
$$;