// MULTI-TIER CACHE LAYER - INTELLIGENT DATA CACHING
// Team: Avery (Lead) - 85% COST REDUCTION THROUGH SMART CACHING!
// Target: <1μs hot cache, <10μs warm cache, <100μs cold cache

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use dashmap::DashMap;
use redis::{Client as RedisClient, Commands, AsyncCommands, aio::ConnectionManager};
use sqlx::{PgPool, postgres::PgPoolOptions};
use lz4::EncoderBuilder;
use zstd::stream::encode_all;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use bytes::Bytes;

use crate::{DataError, Result};

#[derive(Debug, Clone)]
pub struct CacheConfig {
    // Redis configuration
    pub redis_url: String,
    pub redis_pool_size: u32,
    
    // PostgreSQL configuration
    pub postgres_url: String,
    pub postgres_pool_size: u32,
    
    // Cache sizes and TTLs
    pub hot_cache_size_mb: usize,
    pub warm_cache_size_mb: usize,
    pub hot_cache_ttl_seconds: i64,
    pub warm_cache_ttl_seconds: i64,
    pub cold_cache_ttl_seconds: i64,
    
    // Compression settings
    pub compression_threshold_bytes: usize,
    pub use_lz4_for_warm: bool,
    pub use_zstd_for_cold: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_pool_size: 10,
            postgres_url: "postgresql://bot3user:bot3pass@localhost:5432/bot3trading".to_string(),
            postgres_pool_size: 5,
            hot_cache_size_mb: 1024,     // 1GB hot cache
            warm_cache_size_mb: 10240,   // 10GB warm cache
            hot_cache_ttl_seconds: 10,    // 10 seconds
            warm_cache_ttl_seconds: 300,  // 5 minutes
            cold_cache_ttl_seconds: 3600, // 1 hour
            compression_threshold_bytes: 1024,  // Compress data > 1KB
            use_lz4_for_warm: true,
            use_zstd_for_cold: true,
        }
    }
}

/// Multi-tier cache with intelligent data placement
pub struct MultiTierCache {
    // Level 1: Hot cache (in-memory)
    hot_cache: Arc<DashMap<String, CachedItem>>,
    hot_cache_size: AtomicU64,
    
    // Level 2: Warm cache (Redis)
    redis_pool: Arc<RwLock<ConnectionManager>>,
    
    // Level 3: Cold cache (PostgreSQL)
    postgres_pool: Arc<PgPool>,
    
    // Configuration
    config: CacheConfig,
    
    // Metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedItem {
    key: String,
    data: Vec<u8>,
    metadata: CacheMetadata,
    compressed: bool,
    compression_type: CompressionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    access_count: u64,
    last_accessed: DateTime<Utc>,
    data_type: DataType,
    source: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CompressionType {
    None,
    Lz4,
    Zstd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum DataType {
    MarketData,
    OrderBook,
    Sentiment,
    News,
    MacroData,
    OnChainData,
    Technical,
}

#[derive(Debug, Clone)]
pub struct CacheMetrics {
    // Hit rates
    pub hot_cache_hits: u64,
    pub warm_cache_hits: u64,
    pub cold_cache_hits: u64,
    pub cache_misses: u64,
    
    // Performance
    pub avg_hot_latency_us: f64,
    pub avg_warm_latency_us: f64,
    pub avg_cold_latency_us: f64,
    
    // Size
    pub hot_cache_entries: usize,
    pub hot_cache_size_bytes: u64,
    pub warm_cache_entries: usize,
    pub cold_cache_entries: usize,
    
    // Efficiency
    pub compression_ratio: f64,
    pub evictions: u64,
}

impl MultiTierCache {
    pub fn new(config: CacheConfig) -> Result<Self> {
        // Initialize Redis connection
        let redis_client = RedisClient::open(config.redis_url.clone())
            .map_err(|e| DataError::CacheMiss(format!("Failed to connect to Redis: {}", e)))?;
        
        let redis_conn = tokio::runtime::Handle::current()
            .block_on(async {
                redis_client.get_tokio_connection_manager().await
            })
            .map_err(|e| DataError::CacheMiss(format!("Failed to get Redis connection: {}", e)))?;
        
        // Initialize PostgreSQL pool
        let postgres_pool = tokio::runtime::Handle::current()
            .block_on(async {
                PgPoolOptions::new()
                    .max_connections(config.postgres_pool_size)
                    .connect(&config.postgres_url)
                    .await
            })
            .map_err(|e| DataError::CacheMiss(format!("Failed to connect to PostgreSQL: {}", e)))?;
        
        // Create cache table if not exists
        tokio::runtime::Handle::current()
            .block_on(async {
                sqlx::query(
                    r#"
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key VARCHAR(255) PRIMARY KEY,
                        data BYTEA NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ NOT NULL,
                        INDEX idx_expires_at (expires_at)
                    )
                    "#
                )
                .execute(&postgres_pool)
                .await
            })
            .ok();
        
        Ok(Self {
            hot_cache: Arc::new(DashMap::new()),
            hot_cache_size: AtomicU64::new(0),
            redis_pool: Arc::new(RwLock::new(redis_conn)),
            postgres_pool: Arc::new(postgres_pool),
            config,
            metrics: Arc::new(RwLock::new(CacheMetrics {
                hot_cache_hits: 0,
                warm_cache_hits: 0,
                cold_cache_hits: 0,
                cache_misses: 0,
                avg_hot_latency_us: 0.0,
                avg_warm_latency_us: 0.0,
                avg_cold_latency_us: 0.0,
                hot_cache_entries: 0,
                hot_cache_size_bytes: 0,
                warm_cache_entries: 0,
                cold_cache_entries: 0,
                compression_ratio: 1.0,
                evictions: 0,
            })),
        })
    }
    
    /// Get data from cache (tries all tiers)
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let start = Instant::now();
        
        // Try hot cache first
        if let Some(item) = self.get_from_hot_cache(key) {
            self.update_metrics_hot(start.elapsed());
            return Some(self.decompress_if_needed(item));
        }
        
        // Try warm cache (Redis)
        if let Some(item) = self.get_from_warm_cache(key).await {
            self.update_metrics_warm(start.elapsed());
            // Promote to hot cache
            self.promote_to_hot(key, item.clone());
            return Some(self.decompress_if_needed(item));
        }
        
        // Try cold cache (PostgreSQL)
        if let Some(item) = self.get_from_cold_cache(key).await {
            self.update_metrics_cold(start.elapsed());
            // Promote to warm cache
            self.promote_to_warm(key, item.clone()).await;
            return Some(self.decompress_if_needed(item));
        }
        
        // Cache miss
        self.update_metrics_miss();
        None
    }
    
    /// Set data in appropriate cache tier based on data type and access patterns
    pub async fn set(&self, key: String, data: Vec<u8>, data_type: DataType, ttl_seconds: Option<i64>) {
        let ttl = ttl_seconds.unwrap_or_else(|| self.get_default_ttl(&data_type));
        let size = data.len();
        
        // Compress if needed
        let (compressed_data, compressed, compression_type) = self.compress_if_needed(&data);
        
        // Create metadata
        let metadata = CacheMetadata {
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::seconds(ttl),
            access_count: 0,
            last_accessed: Utc::now(),
            data_type: data_type.clone(),
            source: "data_intelligence".to_string(),
            size_bytes: size,
        };
        
        let item = CachedItem {
            key: key.clone(),
            data: compressed_data,
            metadata,
            compressed,
            compression_type,
        };
        
        // Determine which tier based on data type and TTL
        match data_type {
            DataType::MarketData | DataType::OrderBook if ttl <= 10 => {
                // Hot cache for real-time data
                self.set_in_hot_cache(key, item);
            }
            DataType::Sentiment | DataType::Technical if ttl <= 300 => {
                // Warm cache for near-real-time data
                self.set_in_warm_cache(key, item).await;
            }
            _ => {
                // Cold cache for historical/slow-changing data
                self.set_in_cold_cache(key, item).await;
            }
        }
    }
    
    /// Invalidate cache entry across all tiers
    pub async fn invalidate(&self, key: &str) {
        // Remove from hot cache
        self.hot_cache.remove(key);
        
        // Remove from warm cache (Redis)
        if let Ok(mut conn) = self.redis_pool.write().clone().into_inner() {
            let _: Result<(), _> = conn.del(key).await;
        }
        
        // Remove from cold cache (PostgreSQL)
        let _ = sqlx::query("DELETE FROM cache_entries WHERE key = $1")
            .bind(key)
            .execute(self.postgres_pool.as_ref())
            .await;
    }
    
    /// Get from hot cache
    fn get_from_hot_cache(&self, key: &str) -> Option<CachedItem> {
        self.hot_cache.get(key).map(|entry| {
            let mut item = entry.clone();
            item.metadata.access_count += 1;
            item.metadata.last_accessed = Utc::now();
            item
        })
    }
    
    /// Get from warm cache (Redis)
    async fn get_from_warm_cache(&self, key: &str) -> Option<CachedItem> {
        let mut conn = self.redis_pool.write().clone();
        
        match conn.get::<_, Vec<u8>>(key).await {
            Ok(data) => {
                // Deserialize
                bincode::deserialize(&data).ok()
            }
            Err(_) => None,
        }
    }
    
    /// Get from cold cache (PostgreSQL)
    async fn get_from_cold_cache(&self, key: &str) -> Option<CachedItem> {
        let result = sqlx::query_as::<_, (Vec<u8>, serde_json::Value)>(
            "SELECT data, metadata FROM cache_entries WHERE key = $1 AND expires_at > NOW()"
        )
        .bind(key)
        .fetch_optional(self.postgres_pool.as_ref())
        .await
        .ok()?;
        
        if let Some((data, metadata_json)) = result {
            let metadata: CacheMetadata = serde_json::from_value(metadata_json).ok()?;
            Some(CachedItem {
                key: key.to_string(),
                data,
                metadata,
                compressed: true,  // Cold cache always compressed
                compression_type: CompressionType::Zstd,
            })
        } else {
            None
        }
    }
    
    /// Set in hot cache with LRU eviction
    fn set_in_hot_cache(&self, key: String, item: CachedItem) {
        let size = item.data.len() as u64;
        let max_size = (self.config.hot_cache_size_mb * 1024 * 1024) as u64;
        
        // Check if we need to evict
        if self.hot_cache_size.load(Ordering::Relaxed) + size > max_size {
            self.evict_lru_from_hot_cache(size);
        }
        
        self.hot_cache.insert(key, item);
        self.hot_cache_size.fetch_add(size, Ordering::Relaxed);
        
        let mut metrics = self.metrics.write();
        metrics.hot_cache_entries = self.hot_cache.len();
        metrics.hot_cache_size_bytes = self.hot_cache_size.load(Ordering::Relaxed);
    }
    
    /// Set in warm cache (Redis)
    async fn set_in_warm_cache(&self, key: String, item: CachedItem) {
        let mut conn = self.redis_pool.write().clone();
        
        if let Ok(serialized) = bincode::serialize(&item) {
            let ttl = (item.metadata.expires_at - Utc::now()).num_seconds() as usize;
            let _: Result<(), _> = conn.set_ex(key, serialized, ttl).await;
            
            let mut metrics = self.metrics.write();
            metrics.warm_cache_entries += 1;
        }
    }
    
    /// Set in cold cache (PostgreSQL)
    async fn set_in_cold_cache(&self, key: String, item: CachedItem) {
        let metadata_json = serde_json::to_value(&item.metadata).unwrap_or_default();
        
        let _ = sqlx::query(
            r#"
            INSERT INTO cache_entries (key, data, metadata, expires_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (key) DO UPDATE
            SET data = EXCLUDED.data,
                metadata = EXCLUDED.metadata,
                expires_at = EXCLUDED.expires_at
            "#
        )
        .bind(&key)
        .bind(&item.data)
        .bind(&metadata_json)
        .bind(&item.metadata.expires_at)
        .execute(self.postgres_pool.as_ref())
        .await;
        
        let mut metrics = self.metrics.write();
        metrics.cold_cache_entries += 1;
    }
    
    /// Evict LRU entries from hot cache
    fn evict_lru_from_hot_cache(&self, needed_space: u64) {
        let mut entries: Vec<_> = self.hot_cache.iter()
            .map(|entry| (entry.key().clone(), entry.metadata.last_accessed))
            .collect();
        
        entries.sort_by_key(|e| e.1);
        
        let mut freed_space = 0u64;
        for (key, _) in entries {
            if freed_space >= needed_space {
                break;
            }
            
            if let Some((_, item)) = self.hot_cache.remove(&key) {
                freed_space += item.data.len() as u64;
                self.hot_cache_size.fetch_sub(item.data.len() as u64, Ordering::Relaxed);
                
                let mut metrics = self.metrics.write();
                metrics.evictions += 1;
            }
        }
    }
    
    /// Promote item to hot cache
    fn promote_to_hot(&self, key: &str, item: CachedItem) {
        self.set_in_hot_cache(key.to_string(), item);
    }
    
    /// Promote item to warm cache
    async fn promote_to_warm(&self, key: &str, item: CachedItem) {
        self.set_in_warm_cache(key.to_string(), item).await;
    }
    
    /// Compress data if needed
    fn compress_if_needed(&self, data: &[u8]) -> (Vec<u8>, bool, CompressionType) {
        if data.len() < self.config.compression_threshold_bytes {
            return (data.to_vec(), false, CompressionType::None);
        }
        
        // Use LZ4 for warm cache (fast compression)
        if self.config.use_lz4_for_warm && data.len() < 100_000 {
            if let Ok(compressed) = lz4::block::compress(data, None, false) {
                if compressed.len() < data.len() {
                    return (compressed, true, CompressionType::Lz4);
                }
            }
        }
        
        // Use Zstd for cold cache (better compression ratio)
        if self.config.use_zstd_for_cold {
            if let Ok(compressed) = encode_all(data, 3) {
                if compressed.len() < data.len() {
                    return (compressed, true, CompressionType::Zstd);
                }
            }
        }
        
        (data.to_vec(), false, CompressionType::None)
    }
    
    /// Decompress data if needed
    fn decompress_if_needed(&self, item: CachedItem) -> Vec<u8> {
        if !item.compressed {
            return item.data;
        }
        
        match item.compression_type {
            CompressionType::None => item.data,
            CompressionType::Lz4 => {
                lz4::block::decompress(&item.data, Some(item.metadata.size_bytes))
                    .unwrap_or(item.data)
            }
            CompressionType::Zstd => {
                zstd::stream::decode_all(&item.data[..])
                    .unwrap_or(item.data)
            }
        }
    }
    
    /// Get default TTL for data type
    fn get_default_ttl(&self, data_type: &DataType) -> i64 {
        match data_type {
            DataType::MarketData | DataType::OrderBook => self.config.hot_cache_ttl_seconds,
            DataType::Sentiment | DataType::Technical => self.config.warm_cache_ttl_seconds,
            DataType::News | DataType::MacroData | DataType::OnChainData => self.config.cold_cache_ttl_seconds,
        }
    }
    
    /// Update metrics for hot cache hit
    fn update_metrics_hot(&self, latency: Duration) {
        let mut metrics = self.metrics.write();
        metrics.hot_cache_hits += 1;
        let latency_us = latency.as_micros() as f64;
        metrics.avg_hot_latency_us = 
            (metrics.avg_hot_latency_us * (metrics.hot_cache_hits - 1) as f64 + latency_us) 
            / metrics.hot_cache_hits as f64;
    }
    
    /// Update metrics for warm cache hit
    fn update_metrics_warm(&self, latency: Duration) {
        let mut metrics = self.metrics.write();
        metrics.warm_cache_hits += 1;
        let latency_us = latency.as_micros() as f64;
        metrics.avg_warm_latency_us = 
            (metrics.avg_warm_latency_us * (metrics.warm_cache_hits - 1) as f64 + latency_us) 
            / metrics.warm_cache_hits as f64;
    }
    
    /// Update metrics for cold cache hit
    fn update_metrics_cold(&self, latency: Duration) {
        let mut metrics = self.metrics.write();
        metrics.cold_cache_hits += 1;
        let latency_us = latency.as_micros() as f64;
        metrics.avg_cold_latency_us = 
            (metrics.avg_cold_latency_us * (metrics.cold_cache_hits - 1) as f64 + latency_us) 
            / metrics.cold_cache_hits as f64;
    }
    
    /// Update metrics for cache miss
    fn update_metrics_miss(&self) {
        let mut metrics = self.metrics.write();
        metrics.cache_misses += 1;
    }
    
    /// Get cache metrics
    pub fn metrics(&self) -> CacheMetrics {
        self.metrics.read().clone()
    }
    
    /// Calculate overall cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let metrics = self.metrics.read();
        let total_hits = metrics.hot_cache_hits + metrics.warm_cache_hits + metrics.cold_cache_hits;
        let total_requests = total_hits + metrics.cache_misses;
        
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_multi_tier_cache() {
        let config = CacheConfig::default();
        let cache = MultiTierCache::new(config).unwrap();
        
        // Test set and get
        let key = "test_key".to_string();
        let data = vec![1, 2, 3, 4, 5];
        
        cache.set(key.clone(), data.clone(), DataType::MarketData, Some(10)).await;
        
        let retrieved = cache.get(&key).await;
        assert_eq!(retrieved, Some(data));
        
        // Check metrics
        let metrics = cache.metrics();
        assert_eq!(metrics.hot_cache_hits, 1);
        assert_eq!(metrics.cache_misses, 0);
    }
    
    #[test]
    fn test_compression() {
        let config = CacheConfig::default();
        let cache = MultiTierCache::new(config).unwrap();
        
        let data = vec![0u8; 2000];  // Large enough to trigger compression
        let (compressed, is_compressed, _) = cache.compress_if_needed(&data);
        
        assert!(is_compressed);
        assert!(compressed.len() < data.len());
    }
}