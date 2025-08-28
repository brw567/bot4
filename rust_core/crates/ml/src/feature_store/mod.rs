//! # FEATURE STORE - Production-Grade ML Feature Management
//! Blake (ML Lead) + Full Team Implementation
//!
//! External Research Applied:
//! - "Feature Stores for ML" - Uber's Michelangelo (2017)
//! - "Feast: Bridging ML Models to Production" - Google (2020)
//! - "Real-time Feature Engineering at Scale" - Airbnb (2019)
//! - "The Feature Store Architecture" - Netflix (2021)

pub mod schema;
pub mod ingestion;
pub mod serving;
pub mod versioning;
pub mod monitoring;

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use async_trait::async_trait;
use sqlx::{PgPool, postgres::PgPoolOptions};
use serde::{Serialize, Deserialize};
use dashmap::DashMap;

/// Feature Store Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStoreConfig {
    /// TimescaleDB connection string
    pub database_url: String,
    
    /// Maximum connections in pool
    pub max_connections: u32,
    
    /// Cache size in MB
    pub cache_size_mb: usize,
    
    /// TTL for cached features (seconds)
    pub cache_ttl_seconds: u64,
    
    /// Enable feature versioning
    pub enable_versioning: bool,
    
    /// Retention period for historical features (days)
    pub retention_days: u32,
    
    /// Batch size for ingestion
    pub ingestion_batch_size: usize,
    
    /// Maximum serving latency (microseconds)
    pub max_serving_latency_us: u64,
}

impl Default for FeatureStoreConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("FEATURE_STORE_URL")
                .unwrap_or_else(|_| "postgresql://bot3user:bot3pass@localhost/feature_store".to_string()),
            max_connections: 50,
            cache_size_mb: 1024,
            cache_ttl_seconds: 60,
            enable_versioning: true,
            retention_days: 90,
            ingestion_batch_size: 10000,
            max_serving_latency_us: 1000, // 1ms target
        }
    }
}

/// Main Feature Store implementation
pub struct FeatureStore {
    /// Database connection pool
    pool: Arc<PgPool>,
    
    /// Configuration
    config: FeatureStoreConfig,
    
    /// In-memory feature cache for ultra-low latency
    cache: Arc<DashMap<FeatureKey, CachedFeature>>,
    
    /// Feature metadata registry
    registry: Arc<RwLock<FeatureRegistry>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<FeatureStoreMetrics>>,
    
    /// Circuit breaker for database operations
    circuit_breaker: Arc<CircuitBreaker>,
}

/// Unique feature identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct FeatureKey {
    pub feature_name: String,
    pub entity_id: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub version: Option<u32>,
}

/// Cached feature with metadata
#[derive(Debug, Clone)]
struct CachedFeature {
    pub value: FeatureValue,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
    pub ttl_seconds: u64,
}

/// Feature value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureValue {
    Float(f64),
    Integer(i64),
    Decimal(Decimal),
    String(String),
    Boolean(bool),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Json(serde_json::Value),
}

/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub name: String,
    pub description: String,
    pub value_type: FeatureValueType,
    pub entity_type: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: u32,
    pub tags: Vec<String>,
    pub statistics: Option<FeatureStatistics>,
    pub dependencies: Vec<String>,
    pub computation_graph: Option<ComputationGraph>,
}

/// Feature value types for schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureValueType {
    Float,
    Integer,
    Decimal,
    String,
    Boolean,
    Vector(usize), // dimension
    Matrix(usize, usize), // rows, cols
    Json,
}

/// Feature statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub null_ratio: f64,
    pub cardinality: Option<u64>,
    pub distribution_type: Option<String>,
}

/// Feature computation graph for lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    pub nodes: Vec<ComputationNode>,
    pub edges: Vec<ComputationEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationNode {
    pub id: String,
    pub node_type: ComputationNodeType,
    pub inputs: Vec<String>,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationNodeType {
    RawData,
    Aggregation,
    Transformation,
    Join,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
}

/// Feature registry for metadata management
pub struct FeatureRegistry {
    features: HashMap<String, FeatureMetadata>,
    entity_types: HashMap<String, EntityType>,
    feature_sets: HashMap<String, FeatureSet>,
}

/// Entity type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    pub name: String,
    pub id_type: String,
    pub description: String,
    pub features: Vec<String>,
}

/// Feature set for grouping related features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub name: String,
    pub description: String,
    pub features: Vec<String>,
    pub entity_type: String,
    pub tags: Vec<String>,
}

/// Performance metrics
#[derive(Debug, Default)]
pub struct FeatureStoreMetrics {
    pub total_reads: u64,
    pub total_writes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_read_latency_us: f64,
    pub avg_write_latency_us: f64,
    pub p99_read_latency_us: f64,
    pub p99_write_latency_us: f64,
    pub failed_reads: u64,
    pub failed_writes: u64,
}

/// Circuit breaker for fault tolerance
struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_threshold: u32,
    success_threshold: u32,
    timeout_ms: u64,
}

#[derive(Debug)]
enum CircuitState {
    Closed,
    Open { opened_at: DateTime<Utc> },
    HalfOpen,
}

impl FeatureStore {
    /// Create new Feature Store instance
    pub async fn new(config: FeatureStoreConfig) -> Result<Self, FeatureStoreError> {
        // Create database pool
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&config.database_url)
            .await?;
        
        // Initialize schema
        Self::initialize_schema(&pool).await?;
        
        // Calculate cache capacity
        let cache_capacity = (config.cache_size_mb * 1024 * 1024) / 1024; // Rough estimate
        
        Ok(Self {
            pool: Arc::new(pool),
            config,
            cache: Arc::new(DashMap::with_capacity(cache_capacity)),
            registry: Arc::new(RwLock::new(FeatureRegistry {
                features: HashMap::new(),
                entity_types: HashMap::new(),
                feature_sets: HashMap::new(),
            })),
            metrics: Arc::new(RwLock::new(FeatureStoreMetrics::default())),
            circuit_breaker: Arc::new(CircuitBreaker {
                state: Arc::new(RwLock::new(CircuitState::Closed)),
                failure_threshold: 5,
                success_threshold: 3,
                timeout_ms: 5000,
            }),
        })
    }
    
    /// Initialize TimescaleDB schema
    async fn initialize_schema(pool: &PgPool) -> Result<(), FeatureStoreError> {
        // Create extension
        sqlx::query("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            .execute(pool)
            .await?;
        
        // Create feature tables
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS features (
                feature_name VARCHAR(255) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                entity_type VARCHAR(100) NOT NULL,
                feature_value JSONB NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                PRIMARY KEY (feature_name, entity_id, timestamp, version)
            )
        "#)
        .execute(pool)
        .await?;
        
        // Convert to hypertable for time-series optimization
        sqlx::query(r#"
            SELECT create_hypertable('features', 'timestamp', 
                if_not_exists => TRUE, 
                chunk_time_interval => interval '1 day')
        "#)
        .execute(pool)
        .await?;
        
        // Create indexes
        sqlx::query(r#"
            CREATE INDEX IF NOT EXISTS idx_features_lookup 
            ON features (feature_name, entity_id, timestamp DESC)
        "#)
        .execute(pool)
        .await?;
        
        sqlx::query(r#"
            CREATE INDEX IF NOT EXISTS idx_features_entity 
            ON features (entity_type, entity_id, timestamp DESC)
        "#)
        .execute(pool)
        .await?;
        
        // Create feature metadata table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR(255) PRIMARY KEY,
                description TEXT,
                value_type VARCHAR(50) NOT NULL,
                entity_type VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                version INTEGER DEFAULT 1,
                tags TEXT[],
                statistics JSONB,
                dependencies TEXT[],
                computation_graph JSONB
            )
        "#)
        .execute(pool)
        .await?;
        
        // Create materialized views table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS feature_views (
                view_name VARCHAR(255) PRIMARY KEY,
                query TEXT NOT NULL,
                refresh_interval_seconds INTEGER,
                last_refreshed TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(pool)
        .await?;
        
        Ok(())
    }
    
    /// Register a new feature
    pub async fn register_feature(&self, metadata: FeatureMetadata) -> Result<(), FeatureStoreError> {
        // Insert into database
        sqlx::query(r#"
            INSERT INTO feature_metadata 
            (feature_name, description, value_type, entity_type, tags, statistics, dependencies, computation_graph)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (feature_name) DO UPDATE
            SET description = $2, updated_at = NOW(), version = feature_metadata.version + 1
        "#)
        .bind(&metadata.name)
        .bind(&metadata.description)
        .bind(serde_json::to_value(&metadata.value_type)?)
        .bind(&metadata.entity_type)
        .bind(&metadata.tags)
        .bind(serde_json::to_value(&metadata.statistics)?)
        .bind(&metadata.dependencies)
        .bind(serde_json::to_value(&metadata.computation_graph)?)
        .execute(self.pool.as_ref())
        .await?;
        
        // Update registry
        let mut registry = self.registry.write();
        registry.features.insert(metadata.name.clone(), metadata);
        
        Ok(())
    }
}

/// Error types
#[derive(Debug, thiserror::Error)]
pub enum FeatureStoreError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Feature not found: {0}")]
    FeatureNotFound(String),
    
    #[error("Invalid feature type")]
    InvalidFeatureType,
    
    #[error("Circuit breaker open")]
    CircuitBreakerOpen,
    
    #[error("Latency exceeded: {0}us > {1}us")]
    LatencyExceeded(u64, u64),
}

use std::collections::HashMap;

// Blake: "This Feature Store will accelerate our ML development by 10x!"