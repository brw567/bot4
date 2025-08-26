// Online Feature Store - Redis-based for <1ms latency
// DEEP DIVE: Sub-millisecond serving for HFT requirements

use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{Result, Context};
use async_trait::async_trait;
use redis::{aio::MultiplexedConnection, AsyncCommands, Client, cluster::ClusterClient};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, debug, instrument};
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use arrow::array::{Float64Array, ArrayRef};
use arrow::record_batch::RecordBatch;

/// Redis configuration for online store
#[derive(Debug, Clone, Deserialize)]
pub struct RedisConfig {
    pub urls: Vec<String>,
    pub cluster_mode: bool,
    pub connection_pool_size: usize,
    pub max_connections: usize,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub ttl_seconds: u64, // Feature TTL in Redis
    pub cache_size: usize, // Local LRU cache size
    pub compression: bool,
    pub pipeline_size: usize,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            urls: vec!["redis://127.0.0.1:6379".to_string()],
            cluster_mode: false,
            connection_pool_size: 32,
            max_connections: 128,
            timeout_ms: 100,
            retry_attempts: 3,
            ttl_seconds: 3600, // 1 hour hot data
            cache_size: 10000,
            compression: true,
            pipeline_size: 1000,
        }
    }
}

/// Feature vector for serving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub entity_id: String,
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl FeatureVector {
    /// Convert to Arrow RecordBatch for zero-copy operations
    pub fn to_arrow_batch(&self) -> Result<RecordBatch> {
        let mut arrays: Vec<ArrayRef> = Vec::new();
        
        for (i, value) in self.features.iter().enumerate() {
            let array = Float64Array::from(vec![*value]);
            arrays.push(Arc::new(array));
        }
        
        // Create schema and batch
        let schema = self.create_arrow_schema()?;
        RecordBatch::try_new(Arc::new(schema), arrays)
            .context("Failed to create Arrow batch")
    }
    
    fn create_arrow_schema(&self) -> Result<arrow::datatypes::Schema> {
        use arrow::datatypes::{DataType, Field, Schema};
        
        let fields: Vec<Field> = self.feature_names
            .iter()
            .map(|name| Field::new(name, DataType::Float64, false))
            .collect();
        
        Ok(Schema::new(fields))
    }
}

/// Online feature store with Redis backend
pub struct OnlineStore {
    config: RedisConfig,
    client: Option<Client>,
    cluster_client: Option<ClusterClient>,
    connections: Arc<RwLock<Vec<MultiplexedConnection>>>,
    connection_semaphore: Arc<Semaphore>,
    
    // Local caching for ultra-low latency
    local_cache: Arc<Mutex<LruCache<String, FeatureVector>>>,
    
    // Metrics
    hit_count: Arc<RwLock<u64>>,
    miss_count: Arc<RwLock<u64>>,
    latencies: Arc<RwLock<Vec<Duration>>>,
    
    // Circuit breaker
    circuit_breaker: Arc<infrastructure::CircuitBreaker>,
}

impl OnlineStore {
    /// Create new online store instance
    pub async fn new(config: RedisConfig) -> Result<Self> {
        info!("Initializing Redis online store for <1ms serving");
        
        let (client, cluster_client) = if config.cluster_mode {
            let cluster = ClusterClient::new(config.urls.clone())?;
            (None, Some(cluster))
        } else {
            let client = Client::open(config.urls[0].as_str())?;
            (Some(client), None)
        };
        
        // Create connection pool
        let mut connections = Vec::new();
        for _ in 0..config.connection_pool_size {
            let conn = if let Some(ref client) = client {
                client.get_multiplexed_tokio_connection().await?
            } else if let Some(ref cluster) = cluster_client {
                cluster.get_async_connection().await?
            } else {
                return Err(anyhow::anyhow!("No Redis client configured"));
            };
            connections.push(conn);
        }
        
        // Initialize local cache
        let local_cache = Arc::new(Mutex::new(LruCache::new(
            std::num::NonZeroUsize::new(config.cache_size).unwrap()
        )));
        
        // Circuit breaker for fault tolerance
        let circuit_breaker = Arc::new(
            infrastructure::CircuitBreaker::new(
                "online_store".to_string(),
                3, // failure threshold
                Duration::from_secs(30), // reset timeout
            )
        );
        
        Ok(Self {
            config: config.clone(),
            client,
            cluster_client,
            connections: Arc::new(RwLock::new(connections)),
            connection_semaphore: Arc::new(Semaphore::new(config.max_connections)),
            local_cache,
            hit_count: Arc::new(RwLock::new(0)),
            miss_count: Arc::new(RwLock::new(0)),
            latencies: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            circuit_breaker,
        })
    }
    
    /// Get features with <1ms latency target
    #[instrument(skip(self))]
    pub async fn get_features(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
    ) -> Result<Vec<FeatureVector>> {
        let start = Instant::now();
        
        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            return Err(anyhow::anyhow!("Circuit breaker is open"));
        }
        
        let mut results = Vec::with_capacity(entity_ids.len());
        
        for entity_id in entity_ids {
            // Try local cache first (nanosecond latency)
            if let Some(cached) = self.get_from_cache(&entity_id) {
                *self.hit_count.write().await += 1;
                results.push(cached);
                continue;
            }
            
            *self.miss_count.write().await += 1;
            
            // Fetch from Redis
            let features = self.fetch_from_redis(&entity_id, &feature_names).await?;
            
            // Update cache
            self.update_cache(entity_id.clone(), features.clone());
            
            results.push(features);
        }
        
        // Record latency
        let latency = start.elapsed();
        self.record_latency(latency).await;
        
        if latency.as_millis() > 1 {
            warn!("Online store latency exceeded 1ms: {:?}", latency);
        }
        
        Ok(results)
    }
    
    /// Batch get for efficiency
    pub async fn batch_get_features(
        &self,
        entity_feature_pairs: Vec<(String, Vec<String>)>,
    ) -> Result<Vec<FeatureVector>> {
        let start = Instant::now();
        
        // Use pipelining for batch efficiency
        let _permit = self.connection_semaphore.acquire().await?;
        let connections = self.connections.read().await;
        let mut conn = connections[0].clone();
        
        let mut pipe = redis::pipe();
        
        for (entity_id, feature_names) in &entity_feature_pairs {
            for feature_name in feature_names {
                let key = self.build_key(entity_id, feature_name);
                pipe.get(&key);
            }
        }
        
        let values: Vec<Option<Vec<u8>>> = pipe
            .query_async(&mut conn)
            .await
            .context("Failed to execute Redis pipeline")?;
        
        // Parse results
        let mut results = Vec::new();
        let mut value_idx = 0;
        
        for (entity_id, feature_names) in entity_feature_pairs {
            let mut features = Vec::new();
            
            for _ in &feature_names {
                if let Some(Some(bytes)) = values.get(value_idx) {
                    let value = self.deserialize_value(bytes)?;
                    features.push(value);
                }
                value_idx += 1;
            }
            
            results.push(FeatureVector {
                entity_id,
                features,
                feature_names,
                timestamp: chrono::Utc::now(),
                metadata: None,
            });
        }
        
        let latency = start.elapsed();
        debug!("Batch get {} features in {:?}", results.len(), latency);
        
        Ok(results)
    }
    
    /// Set features in online store
    pub async fn set_features(
        &self,
        updates: Vec<(String, String, f64)>, // (entity_id, feature_name, value)
    ) -> Result<()> {
        let _permit = self.connection_semaphore.acquire().await?;
        let connections = self.connections.read().await;
        let mut conn = connections[0].clone();
        
        let mut pipe = redis::pipe();
        
        for (entity_id, feature_name, value) in updates {
            let key = self.build_key(&entity_id, &feature_name);
            let serialized = self.serialize_value(value)?;
            
            pipe.set_ex(&key, serialized, self.config.ttl_seconds);
            
            // Invalidate cache
            self.invalidate_cache(&entity_id);
        }
        
        pipe.query_async(&mut conn)
            .await
            .context("Failed to set features in Redis")?;
        
        Ok(())
    }
    
    /// Initialize feature space
    pub async fn initialize_feature(&self, feature_id: &str) -> Result<()> {
        // Create Redis hash for feature metadata
        let _permit = self.connection_semaphore.acquire().await?;
        let connections = self.connections.read().await;
        let mut conn = connections[0].clone();
        
        let metadata_key = format!("feature:meta:{}", feature_id);
        conn.hset(&metadata_key, "initialized", chrono::Utc::now().to_rfc3339())
            .await?;
        
        Ok(())
    }
    
    /// Get from local cache
    fn get_from_cache(&self, entity_id: &str) -> Option<FeatureVector> {
        let mut cache = self.local_cache.lock();
        cache.get(entity_id).cloned()
    }
    
    /// Update local cache
    fn update_cache(&self, entity_id: String, features: FeatureVector) {
        let mut cache = self.local_cache.lock();
        cache.put(entity_id, features);
    }
    
    /// Invalidate cache entry
    fn invalidate_cache(&self, entity_id: &str) {
        let mut cache = self.local_cache.lock();
        cache.pop(entity_id);
    }
    
    /// Fetch from Redis
    async fn fetch_from_redis(
        &self,
        entity_id: &str,
        feature_names: &[String],
    ) -> Result<FeatureVector> {
        let _permit = self.connection_semaphore.acquire().await?;
        let connections = self.connections.read().await;
        let mut conn = connections[0].clone();
        
        let mut features = Vec::new();
        
        for feature_name in feature_names {
            let key = self.build_key(entity_id, feature_name);
            let value: Option<Vec<u8>> = conn.get(&key).await?;
            
            if let Some(bytes) = value {
                features.push(self.deserialize_value(&bytes)?);
            } else {
                features.push(0.0); // Default value
            }
        }
        
        Ok(FeatureVector {
            entity_id: entity_id.to_string(),
            features,
            feature_names: feature_names.to_vec(),
            timestamp: chrono::Utc::now(),
            metadata: None,
        })
    }
    
    /// Build Redis key
    fn build_key(&self, entity_id: &str, feature_name: &str) -> String {
        format!("f:{}:{}", entity_id, feature_name)
    }
    
    /// Serialize value with optional compression
    fn serialize_value(&self, value: f64) -> Result<Vec<u8>> {
        let bytes = value.to_le_bytes().to_vec();
        
        if self.config.compression {
            Ok(lz4_flex::compress_prepend_size(&bytes))
        } else {
            Ok(bytes)
        }
    }
    
    /// Deserialize value
    fn deserialize_value(&self, bytes: &[u8]) -> Result<f64> {
        let decompressed = if self.config.compression {
            lz4_flex::decompress_size_prepended(bytes)?
        } else {
            bytes.to_vec()
        };
        
        if decompressed.len() == 8 {
            Ok(f64::from_le_bytes([
                decompressed[0], decompressed[1], decompressed[2], decompressed[3],
                decompressed[4], decompressed[5], decompressed[6], decompressed[7],
            ]))
        } else {
            Err(anyhow::anyhow!("Invalid feature value size"))
        }
    }
    
    /// Record latency for monitoring
    async fn record_latency(&self, latency: Duration) {
        let mut latencies = self.latencies.write().await;
        latencies.push(latency);
        
        if latencies.len() > 10000 {
            latencies.drain(0..5000); // Keep last 5000
        }
    }
    
    /// Get performance statistics
    pub async fn get_stats(&self) -> OnlineStoreStats {
        let hits = *self.hit_count.read().await;
        let misses = *self.miss_count.read().await;
        let latencies = self.latencies.read().await;
        
        let mut sorted_latencies: Vec<Duration> = latencies.clone();
        sorted_latencies.sort();
        
        let p50 = sorted_latencies.get(sorted_latencies.len() / 2)
            .copied()
            .unwrap_or(Duration::ZERO);
        
        let p99 = sorted_latencies.get(sorted_latencies.len() * 99 / 100)
            .copied()
            .unwrap_or(Duration::ZERO);
        
        OnlineStoreStats {
            hit_rate: hits as f64 / (hits + misses) as f64,
            total_requests: hits + misses,
            p50_latency_us: p50.as_micros() as u64,
            p99_latency_us: p99.as_micros() as u64,
        }
    }
    
    /// Close connections gracefully
    pub async fn close(&self) -> Result<()> {
        info!("Closing online store connections");
        // Connections will be dropped automatically
        Ok(())
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineStoreStats {
    pub hit_rate: f64,
    pub total_requests: u64,
    pub p50_latency_us: u64,
    pub p99_latency_us: u64,
}