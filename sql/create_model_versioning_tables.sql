-- Model Versioning Tables for Bot3 Trading Platform
-- Morgan's domain: Complete model lifecycle management

-- Model versions table with comprehensive metadata
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,  -- Semantic version: major.minor.patch
    state VARCHAR(20) NOT NULL,    -- development, staging, production, deprecated, failed
    model_path TEXT NOT NULL,       -- Path to model file
    
    -- Performance metrics
    training_accuracy DECIMAL(10, 6) DEFAULT 0.0,
    validation_accuracy DECIMAL(10, 6) DEFAULT 0.0,
    test_accuracy DECIMAL(10, 6) DEFAULT 0.0,
    overfitting_gap DECIMAL(10, 6) DEFAULT 0.0,
    inference_latency_ms DECIMAL(10, 2) DEFAULT 0.0,
    
    -- Training metadata
    training_data_hash VARCHAR(64),
    feature_set JSONB DEFAULT '[]',
    hyperparameters JSONB DEFAULT '{}',
    training_duration_seconds DECIMAL(10, 2) DEFAULT 0.0,
    
    -- Deployment metadata
    promoted_at TIMESTAMP,
    deprecated_at TIMESTAMP,
    rollback_from VARCHAR(20),  -- Version this was rolled back from
    
    -- A/B testing
    traffic_percentage DECIMAL(5, 2) DEFAULT 0.0,
    total_predictions INTEGER DEFAULT 0,
    successful_predictions INTEGER DEFAULT 0,
    
    -- Model optimization
    model_size_mb DECIMAL(10, 2) DEFAULT 0.0,
    is_compressed BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'morgan',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(model_name, version)
);

-- Model deployment history
CREATE TABLE IF NOT EXISTS model_deployments (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    from_state VARCHAR(20),
    to_state VARCHAR(20) NOT NULL,
    deployed_by VARCHAR(100),
    deployment_type VARCHAR(50), -- manual, auto_promote, rollback
    reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    
    -- Foreign key removed due to composite key issue
    -- Will enforce in application layer
);

-- A/B test configurations
CREATE TABLE IF NOT EXISTS model_ab_tests (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    control_version VARCHAR(20) NOT NULL,
    treatment_version VARCHAR(20) NOT NULL,
    traffic_split JSONB NOT NULL, -- {"control": 80, "treatment": 20}
    
    -- Test configuration
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active', -- active, completed, cancelled
    
    -- Results
    control_metrics JSONB DEFAULT '{}',
    treatment_metrics JSONB DEFAULT '{}',
    winner VARCHAR(20), -- control or treatment
    statistical_significance DECIMAL(5, 4),
    
    -- Metadata
    created_by VARCHAR(100) DEFAULT 'morgan',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance history
CREATE TABLE IF NOT EXISTS model_performance_history (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    -- Metrics
    accuracy DECIMAL(10, 6),
    precision_score DECIMAL(10, 6),
    recall DECIMAL(10, 6),
    f1_score DECIMAL(10, 6),
    latency_p50 DECIMAL(10, 2),
    latency_p95 DECIMAL(10, 2),
    latency_p99 DECIMAL(10, 2),
    
    -- Context
    prediction_count INTEGER,
    error_count INTEGER,
    
    -- Time window
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    
    -- Foreign key removed due to composite key issue
    -- Will enforce in application layer
);

-- Model version comparisons
CREATE TABLE IF NOT EXISTS model_comparisons (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version1 VARCHAR(20) NOT NULL,
    version2 VARCHAR(20) NOT NULL,
    
    -- Comparison metrics
    accuracy_diff DECIMAL(10, 6),
    overfitting_diff DECIMAL(10, 6),
    latency_diff DECIMAL(10, 2),
    size_diff DECIMAL(10, 2),
    
    -- Benchmark results
    benchmark_dataset VARCHAR(100),
    benchmark_results JSONB,
    
    -- Recommendation
    recommendation VARCHAR(20), -- version1, version2, further_testing
    reason TEXT,
    
    compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    compared_by VARCHAR(100) DEFAULT 'morgan'
);

-- Shadow mode testing
CREATE TABLE IF NOT EXISTS model_shadow_tests (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    production_version VARCHAR(20) NOT NULL,
    shadow_version VARCHAR(20) NOT NULL,
    
    -- Test results
    agreement_rate DECIMAL(5, 2), -- Percentage of matching predictions
    shadow_accuracy DECIMAL(10, 6),
    shadow_latency_ms DECIMAL(10, 2),
    
    -- Test period
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_predictions INTEGER DEFAULT 0,
    
    status VARCHAR(20) DEFAULT 'active' -- active, completed, failed
);

-- Model rollback history
CREATE TABLE IF NOT EXISTS model_rollbacks (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    from_version VARCHAR(20) NOT NULL,
    to_version VARCHAR(20) NOT NULL,
    
    -- Rollback reason
    reason VARCHAR(100) NOT NULL, -- performance_degradation, errors, manual
    details TEXT,
    
    -- Metrics at rollback time
    from_version_accuracy DECIMAL(10, 6),
    to_version_accuracy DECIMAL(10, 6),
    performance_drop DECIMAL(10, 6),
    
    rollback_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    initiated_by VARCHAR(100) DEFAULT 'system'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_state ON model_versions(state);
CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_deployments ON model_deployments(model_name, version, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_ab_tests_active ON model_ab_tests(model_name, status) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_model_performance ON model_performance_history(model_name, version, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_comparisons ON model_comparisons(model_name, compared_at DESC);

CREATE INDEX IF NOT EXISTS idx_shadow_tests_active ON model_shadow_tests(model_name, status) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_model_rollbacks ON model_rollbacks(model_name, rollback_time DESC);

-- Trigger to update 'updated_at' column
CREATE OR REPLACE FUNCTION update_model_versions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_model_versions_updated_at_trigger
    BEFORE UPDATE ON model_versions
    FOR EACH ROW
    EXECUTE FUNCTION update_model_versions_updated_at();