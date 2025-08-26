// Hypertable Management for TimescaleDB
// DEEP DIVE: Optimal chunking and partitioning for 1M+ events/sec

use std::sync::Arc;
use anyhow::{Result, Context};
use chrono::Duration;
use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};

/// Hypertable configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub table_name: String,
    pub time_column: String,
    pub chunk_interval: Duration,
    pub space_partition_column: Option<String>,
    pub number_partitions: Option<i32>,
    pub compression_after: Option<Duration>,
    pub retention_period: Option<Duration>,
}

/// Hypertable manager for schema operations
pub struct HypertableManager {
    pool: Arc<Pool>,
}

impl HypertableManager {
    pub async fn new(pool: Arc<Pool>) -> Result<Self> {
        Ok(Self { pool })
    }
    
    /// Create or verify hypertable configuration
    pub async fn ensure_hypertable(&self, config: &ChunkConfig) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Check if already a hypertable
        let exists = conn.query_one(
            "SELECT COUNT(*) FROM timescaledb_information.hypertables 
             WHERE hypertable_name = $1",
            &[&config.table_name],
        ).await?;
        
        let count: i64 = exists.get(0);
        
        if count > 0 {
            debug!("Hypertable {} already exists", config.table_name);
            return Ok(());
        }
        
        // Create hypertable
        let mut query = format!(
            "SELECT create_hypertable('{}', '{}', chunk_time_interval => INTERVAL '{} seconds'",
            config.table_name,
            config.time_column,
            config.chunk_interval.num_seconds()
        );
        
        if let Some(ref col) = config.space_partition_column {
            if let Some(partitions) = config.number_partitions {
                query.push_str(&format!(
                    ", partitioning_column => '{}', number_partitions => {}",
                    col, partitions
                ));
            }
        }
        
        query.push_str(", if_not_exists => TRUE)");
        
        conn.execute(&query, &[]).await?;
        
        info!("Created hypertable: {}", config.table_name);
        
        // Add compression policy if specified
        if let Some(compress_after) = config.compression_after {
            self.add_compression_policy(&config.table_name, compress_after).await?;
        }
        
        // Add retention policy if specified
        if let Some(retention) = config.retention_period {
            self.add_retention_policy(&config.table_name, retention).await?;
        }
        
        Ok(())
    }
    
    /// Add compression policy to hypertable
    async fn add_compression_policy(&self, table: &str, after: Duration) -> Result<()> {
        let conn = self.pool.get().await?;
        
        conn.execute(
            "SELECT add_compression_policy($1, compress_after => $2::INTERVAL, if_not_exists => TRUE)",
            &[&table, &format!("{} seconds", after.num_seconds())],
        ).await?;
        
        info!("Added compression policy to {}: compress after {:?}", table, after);
        Ok(())
    }
    
    /// Add retention policy to hypertable
    async fn add_retention_policy(&self, table: &str, retention: Duration) -> Result<()> {
        let conn = self.pool.get().await?;
        
        conn.execute(
            "SELECT add_retention_policy($1, drop_after => $2::INTERVAL, if_not_exists => TRUE)",
            &[&table, &format!("{} seconds", retention.num_seconds())],
        ).await?;
        
        info!("Added retention policy to {}: drop after {:?}", table, retention);
        Ok(())
    }
    
    /// Get chunk statistics for monitoring
    pub async fn get_chunk_stats(&self, table: &str) -> Result<ChunkStats> {
        let conn = self.pool.get().await?;
        
        let row = conn.query_one(
            "SELECT 
                COUNT(*) as total_chunks,
                COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                SUM(total_bytes) as total_bytes,
                SUM(total_bytes) FILTER (WHERE is_compressed) as compressed_bytes,
                MIN(range_start) as oldest_data,
                MAX(range_end) as newest_data
             FROM timescaledb_information.chunks
             WHERE hypertable_name = $1",
            &[&table],
        ).await?;
        
        Ok(ChunkStats {
            total_chunks: row.get(0),
            compressed_chunks: row.get(1),
            total_bytes: row.get(2),
            compressed_bytes: row.get(3),
            oldest_data: row.get(4),
            newest_data: row.get(5),
        })
    }
    
    /// Reorder chunks for better compression
    pub async fn optimize_chunks(&self, table: &str) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Find uncompressed chunks older than 1 hour
        let chunks = conn.query(
            "SELECT chunk_schema, chunk_name 
             FROM timescaledb_information.chunks
             WHERE hypertable_name = $1 
               AND is_compressed = false
               AND range_end < NOW() - INTERVAL '1 hour'
             LIMIT 10",
            &[&table],
        ).await?;
        
        for chunk in chunks {
            let schema: String = chunk.get(0);
            let name: String = chunk.get(1);
            
            // Reorder chunk for better compression
            conn.execute(
                &format!("SELECT reorder_chunk('{}.{}', '{}')", 
                    schema, name, "time DESC, symbol"),
                &[],
            ).await?;
            
            debug!("Reordered chunk {}.{}", schema, name);
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkStats {
    pub total_chunks: i64,
    pub compressed_chunks: i64,
    pub total_bytes: Option<i64>,
    pub compressed_bytes: Option<i64>,
    pub oldest_data: Option<chrono::DateTime<chrono::Utc>>,
    pub newest_data: Option<chrono::DateTime<chrono::Utc>>,
}