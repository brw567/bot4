// Compression Management for TimescaleDB
// DEEP DIVE: Multi-tier compression for 85-95% space savings

use std::sync::Arc;
use anyhow::Result;
use chrono::Duration;
use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

/// Compression policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPolicy {
    pub table_name: String,
    pub compress_after: Duration,
    pub orderby_columns: Vec<String>,
    pub segmentby_columns: Vec<String>,
}

/// Manager for compression operations
pub struct CompressionManager {
    pool: Arc<Pool>,
    default_compress_after: Duration,
}

impl CompressionManager {
    pub async fn new(pool: Arc<Pool>, default_compress_after: Duration) -> Result<Self> {
        Ok(Self {
            pool,
            default_compress_after,
        })
    }
    
    /// Configure compression for a hypertable
    pub async fn configure_compression(&self, policy: &CompressionPolicy) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Build ALTER TABLE statement
        let orderby = policy.orderby_columns.join(", ");
        let segmentby = policy.segmentby_columns.join(", ");
        
        let sql = format!(
            "ALTER TABLE {} SET (
                timescaledb.compress,
                timescaledb.compress_orderby = '{}',
                timescaledb.compress_segmentby = '{}'
            )",
            policy.table_name, orderby, segmentby
        );
        
        conn.execute(&sql, &[]).await.ok(); // Ignore if already set
        
        // Add compression policy
        conn.execute(
            "SELECT add_compression_policy($1, compress_after => $2::INTERVAL, if_not_exists => TRUE)",
            &[&policy.table_name, &format!("{} seconds", policy.compress_after.num_seconds())],
        ).await?;
        
        info!(
            "Configured compression for {}: compress after {:?}",
            policy.table_name, policy.compress_after
        );
        
        Ok(())
    }
    
    /// Run compression on eligible chunks
    pub async fn run_compression(&self) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Find uncompressed chunks eligible for compression
        let chunks = conn.query(
            "SELECT 
                h.table_name,
                c.chunk_schema,
                c.chunk_name,
                pg_size_pretty(c.total_bytes) as size
             FROM timescaledb_information.chunks c
             JOIN timescaledb_information.hypertables h ON c.hypertable_name = h.table_name
             WHERE c.is_compressed = false
               AND c.range_end < NOW() - INTERVAL '4 hours'
             ORDER BY c.total_bytes DESC
             LIMIT 10",
            &[],
        ).await?;
        
        for chunk in chunks {
            let table: String = chunk.get(0);
            let schema: String = chunk.get(1);
            let name: String = chunk.get(2);
            let size: String = chunk.get(3);
            
            debug!("Compressing chunk {}.{} (size: {})", schema, name, size);
            
            // Compress chunk
            let result = conn.query_one(
                "SELECT compress_chunk($1)",
                &[&format!("{}.{}", schema, name)],
            ).await?;
            
            let compressed: bool = result.get(0);
            
            if compressed {
                info!("Compressed chunk {}.{} for table {}", schema, name, table);
            }
        }
        
        Ok(())
    }
    
    /// Get compression statistics
    pub async fn get_compression_stats(&self) -> Result<Vec<CompressionStats>> {
        let conn = self.pool.get().await?;
        
        let rows = conn.query(
            "SELECT 
                hypertable_name,
                COUNT(*) as total_chunks,
                COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                SUM(total_bytes) as total_bytes,
                SUM(total_bytes) FILTER (WHERE is_compressed) as compressed_bytes,
                SUM(total_bytes) FILTER (WHERE NOT is_compressed) as uncompressed_bytes,
                CASE 
                    WHEN SUM(total_bytes) > 0 THEN
                        100.0 * SUM(total_bytes) FILTER (WHERE is_compressed) / SUM(total_bytes)
                    ELSE 0
                END as compression_ratio_pct
             FROM timescaledb_information.chunks
             GROUP BY hypertable_name
             ORDER BY SUM(total_bytes) DESC",
            &[],
        ).await?;
        
        let mut stats = Vec::new();
        
        for row in rows {
            stats.push(CompressionStats {
                table_name: row.get(0),
                total_chunks: row.get(1),
                compressed_chunks: row.get(2),
                total_bytes: row.get(3),
                compressed_bytes: row.get(4),
                uncompressed_bytes: row.get(5),
                compression_ratio_pct: row.get(6),
            });
        }
        
        Ok(stats)
    }
    
    /// Decompress chunks for a specific time range (for updates/deletes)
    pub async fn decompress_range(
        &self,
        table: &str,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let conn = self.pool.get().await?;
        
        let chunks = conn.query(
            "SELECT chunk_schema, chunk_name
             FROM timescaledb_information.chunks
             WHERE hypertable_name = $1
               AND is_compressed = true
               AND range_start <= $2
               AND range_end >= $3",
            &[&table, &end, &start],
        ).await?;
        
        for chunk in chunks {
            let schema: String = chunk.get(0);
            let name: String = chunk.get(1);
            
            conn.execute(
                "SELECT decompress_chunk($1)",
                &[&format!("{}.{}", schema, name)],
            ).await?;
            
            warn!("Decompressed chunk {}.{} for updates", schema, name);
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub table_name: String,
    pub total_chunks: i64,
    pub compressed_chunks: i64,
    pub total_bytes: Option<i64>,
    pub compressed_bytes: Option<i64>,
    pub uncompressed_bytes: Option<i64>,
    pub compression_ratio_pct: Option<f64>,
}