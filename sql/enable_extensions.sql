-- Enable required PostgreSQL extensions
-- Run as superuser if needed

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable TimescaleDB for time-series optimization (optional)
-- CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Verify extensions are enabled
SELECT * FROM pg_extension WHERE extname IN ('uuid-ossp', 'timescaledb');