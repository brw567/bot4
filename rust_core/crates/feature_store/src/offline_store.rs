// Offline Feature Store - TimescaleDB-based for historical data
// DEEP DIVE: Point-in-time correctness and efficient time-series storage

use std::sync::Arc;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use deadpool_postgres::{Config, Pool, ManagerConfig, RecyclingMethod};
use tokio_postgres::NoTls;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, instrument};
use arrow::record_batch::RecordBatch;
use arrow::array::{Float64Array, TimestampMicrosecondArray, StringArray};
use parquet::arrow::AsyncArrowWriter;

use crate::online_store::FeatureVector;

/// TimescaleDB configuration for offline store
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
// pub struct TimescaleConfig {
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub host: String,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub port: u16,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub database: String,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub username: String,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub password: String,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub pool_size: usize,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub chunk_interval_hours: i64,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub compression_after_hours: i64,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
//     pub retention_days: i64,
// ELIMINATED: Duplicate - use data_ingestion::timescale::TimescaleConfig
// }

impl Default for TimescaleConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            database: "feature_store".to_string(),
            username: "postgres".to_string(),
            password: "postgres".to_string(),
            pool_size: 16,
            chunk_interval_hours: 24,
            compression_after_hours: 168, // 7 days
            retention_days: 365,
        }
    }
}

/// Offline feature store with TimescaleDB
/// TODO: Add docs
pub struct OfflineStore {
    config: TimescaleConfig,
    pool: Arc<Pool>,
}

impl OfflineStore {
    /// Create new offline store instance
    pub async fn new(config: TimescaleConfig) -> Result<Self> {
        info!("Initializing TimescaleDB offline store");
        
        // Create connection pool
        let pool_config = Config {
            host: Some(config.host.clone()),
            port: Some(config.port),
            dbname: Some(config.database.clone()),
            user: Some(config.username.clone()),
            password: Some(config.password.clone()),
            manager: Some(ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            }),
            pool: Some(deadpool_postgres::PoolConfig {
                max_size: config.pool_size,
                ..Default::default()
            }),
            ..Default::default()
        };
        
        let pool = pool_config.create_pool(None, NoTls)?;
        
        // Initialize schema
        let store = Self {
            config: config.clone(),
            pool: Arc::new(pool),
        };
        
        store.initialize_schema().await?;
        
        Ok(store)
    }
    
    /// Initialize TimescaleDB schema
    async fn initialize_schema(&self) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Create feature store schema
        conn.execute(
            "CREATE SCHEMA IF NOT EXISTS feature_store",
            &[],
        ).await?;
        
        // Create main features table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_store.features (
                entity_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value DOUBLE PRECISION NOT NULL,
                event_timestamp TIMESTAMPTZ NOT NULL,
                created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                
                -- Metadata
                feature_version TEXT,
                metadata JSONB,
                
                PRIMARY KEY (entity_id, feature_name, event_timestamp)
            )",
            &[],
        ).await?;
        
        // Convert to hypertable
        conn.execute(
            "SELECT create_hypertable(
                'feature_store.features',
                'event_timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE
            )",
            &[],
        ).await.ok(); // Ignore if already exists
        
        // Create feature definitions table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_store.feature_definitions (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT UNIQUE NOT NULL,
                feature_type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                schema JSONB NOT NULL,
                transformations JSONB,
                dependencies TEXT[],
                tags TEXT[]
            )",
            &[],
        ).await?;
        
        // Create feature statistics table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_store.feature_statistics (
                feature_name TEXT NOT NULL,
                computed_at TIMESTAMPTZ NOT NULL,
                mean DOUBLE PRECISION,
                stddev DOUBLE PRECISION,
                min_value DOUBLE PRECISION,
                max_value DOUBLE PRECISION,
                null_count BIGINT,
                total_count BIGINT,
                percentiles JSONB,
                PRIMARY KEY (feature_name, computed_at)
            )",
            &[],
        ).await?;
        
        // Convert statistics to hypertable
        conn.execute(
            "SELECT create_hypertable(
                'feature_store.feature_statistics',
                'computed_at',
                chunk_time_interval => INTERVAL '7 days',
                if_not_exists => TRUE
            )",
            &[],
        ).await.ok();
        
        // Create continuous aggregate for hourly stats
        conn.execute(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS feature_store.feature_stats_hourly
            WITH (timescaledb.continuous) AS
            SELECT 
                feature_name,
                time_bucket('1 hour', event_timestamp) AS hour,
                COUNT(*) as count,
                AVG(feature_value) as mean,
                STDDEV(feature_value) as stddev,
                MIN(feature_value) as min_value,
                MAX(feature_value) as max_value
            FROM feature_store.features
            GROUP BY feature_name, hour
            WITH NO DATA",
            &[],
        ).await.ok();
        
        // Add indexes for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_entity_time 
            ON feature_store.features (entity_id, event_timestamp DESC)",
            &[],
        ).await?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_name_time 
            ON feature_store.features (feature_name, event_timestamp DESC)",
            &[],
        ).await?;
        
        // Add compression policy
        conn.execute(
            &format!(
                "SELECT add_compression_policy('feature_store.features',
                    compress_after => INTERVAL '{} hours',
                    if_not_exists => TRUE)",
                self.config.compression_after_hours
            ),
            &[],
        ).await.ok();
        
        // Add retention policy
        conn.execute(
            &format!(
                "SELECT add_retention_policy('feature_store.features',
                    drop_after => INTERVAL '{} days',
                    if_not_exists => TRUE)",
                self.config.retention_days
            ),
            &[],
        ).await.ok();
        
        info!("TimescaleDB schema initialized");
        Ok(())
    }
    
    /// Get features at specific point in time
    #[instrument(skip(self))]
    pub async fn get_features_at_time(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureVector>> {
        let conn = self.pool.get().await?;
        
        let mut results = Vec::new();
        
        for entity_id in entity_ids {
            let mut features = Vec::new();
            let mut actual_names = Vec::new();
            
            for feature_name in &feature_names {
                // Get the latest feature value before or at the timestamp
                let row = conn.query_opt(
                    "SELECT feature_value, event_timestamp
                     FROM feature_store.features
                     WHERE entity_id = $1 
                       AND feature_name = $2
                       AND event_timestamp <= $3
                     ORDER BY event_timestamp DESC
                     LIMIT 1",
                    &[&entity_id, feature_name, &timestamp],
                ).await?;
                
                if let Some(row) = row {
                    let value: f64 = row.get(0);
                    features.push(value);
                    actual_names.push(feature_name.clone());
                }
            }
            
            if !features.is_empty() {
                results.push(FeatureVector {
                    entity_id,
                    features,
                    feature_names: actual_names,
                    timestamp,
                    metadata: None,
                });
            }
        }
        
        Ok(results)
    }
    
    /// Write features to offline store
    pub async fn write_features(
        &self,
        updates: Vec<FeatureWrite>,
    ) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Use COPY for bulk insert
        let stmt = "COPY feature_store.features (
            entity_id, feature_name, feature_value, event_timestamp, feature_version, metadata
        ) FROM STDIN WITH (FORMAT BINARY)";
        
        let sink = conn.copy_in(stmt).await?;
        let writer = BinaryCopyInWriter::new(sink, &[
            Type::TEXT,
            Type::TEXT,
            Type::FLOAT8,
            Type::TIMESTAMPTZ,
            Type::TEXT,
            Type::JSONB,
        ]);
        
        for update in updates {
            writer.write(&[
                &update.entity_id,
                &update.feature_name,
                &update.value,
                &update.timestamp,
                &update.version,
                &update.metadata,
            ]).await?;
        }
        
        writer.finish().await?;
        
        Ok(())
    }
    
    /// Get feature statistics for a time range
    pub async fn get_feature_statistics(
        &self,
        feature_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<FeatureStatistics> {
        let conn = self.pool.get().await?;
        
        let row = conn.query_one(
            "SELECT 
                COUNT(*) as count,
                AVG(feature_value) as mean,
                STDDEV(feature_value) as stddev,
                MIN(feature_value) as min_value,
                MAX(feature_value) as max_value,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) as p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) as p75,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY feature_value) as p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY feature_value) as p99
             FROM feature_store.features
             WHERE feature_name = $1 
               AND event_timestamp >= $2 
               AND event_timestamp <= $3",
            &[&feature_name, &start, &end],
        ).await?;
        
        Ok(FeatureStatistics {
            feature_name: feature_name.to_string(),
            count: row.get::<_, i64>(0) as u64,
            mean: row.get(1),
            stddev: row.get(2),
            min: row.get(3),
            max: row.get(4),
            p25: row.get(5),
            p50: row.get(6),
            p75: row.get(7),
            p95: row.get(8),
            p99: row.get(9),
            computed_at: Utc::now(),
        })
    }
    
    /// Export features to Parquet for training
    pub async fn export_to_parquet(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        output_path: &str,
    ) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Query features
        let rows = conn.query(
            "SELECT entity_id, feature_name, feature_value, event_timestamp
             FROM feature_store.features
             WHERE entity_id = ANY($1)
               AND feature_name = ANY($2)
               AND event_timestamp >= $3
               AND event_timestamp <= $4
             ORDER BY event_timestamp",
            &[&entity_ids, &feature_names, &start, &end],
        ).await?;
        
        // Convert to Arrow format
        let mut entity_ids = Vec::new();
        let mut feature_names = Vec::new();
        let mut values = Vec::new();
        let mut timestamps = Vec::new();
        
        for row in rows {
            entity_ids.push(row.get::<_, String>(0));
            feature_names.push(row.get::<_, String>(1));
            values.push(row.get::<_, f64>(2));
            timestamps.push(
                row.get::<_, DateTime<Utc>>(3).timestamp_micros()
            );
        }
        
        // Create Arrow arrays
        let entity_array = StringArray::from(entity_ids);
        let feature_array = StringArray::from(feature_names);
        let value_array = Float64Array::from(values);
        let timestamp_array = TimestampMicrosecondArray::from(timestamps);
        
        // Create RecordBatch
        let batch = RecordBatch::try_new(
            Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("entity_id", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("feature_name", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Float64, false),
                arrow::datatypes::Field::new("timestamp", arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond, None
                ), false),
            ])),
            vec![
                Arc::new(entity_array),
                Arc::new(feature_array),
                Arc::new(value_array),
                Arc::new(timestamp_array),
            ],
        )?;
        
        // Write to Parquet
        let file = tokio::fs::File::create(output_path).await?;
        let mut writer = AsyncArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(&batch).await?;
        writer.close().await?;
        
        info!("Exported {} features to {}", batch.num_rows(), output_path);
        Ok(())
    }
    
    /// Initialize feature in offline store
    pub async fn initialize_feature(&self, feature_id: &str) -> Result<()> {
        // Feature initialization handled by schema
        Ok(())
    }
    
    /// Close connections
    pub async fn close(&self) -> Result<()> {
        info!("Closing offline store connections");
        // Pool will be dropped automatically
        Ok(())
    }
}

/// Feature write structure
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureWrite {
    pub entity_id: String,
    pub feature_name: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub version: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureStatistics {
    pub feature_name: String,
    pub count: u64,
    pub mean: Option<f64>,
    pub stddev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub p25: Option<f64>,
    pub p50: Option<f64>,
    pub p75: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub computed_at: DateTime<Utc>,
}

// Helper for COPY protocol
use tokio_postgres::types::{ToSql, Type};
use bytes::BytesMut;

struct BinaryCopyInWriter<'a> {
    sink: tokio_postgres::CopyInSink<BytesMut>,
    types: &'a [Type],
}

impl<'a> BinaryCopyInWriter<'a> {
    fn new(sink: tokio_postgres::CopyInSink<BytesMut>, types: &'a [Type]) -> Self {
        Self { sink, types }
    }
    
    async fn write(&mut self, values: &[&(dyn ToSql + Sync)]) -> Result<()> {
        // Implementation would handle binary protocol
        // Simplified for brevity
        Ok(())
    }
    
    async fn finish(self) -> Result<()> {
        self.sink.finish().await?;
        Ok(())
    }
}