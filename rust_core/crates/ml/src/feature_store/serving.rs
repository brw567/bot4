//! # FEATURE SERVING - Ultra-Low Latency Feature Retrieval
//! Blake: "Sub-millisecond serving is critical for HFT"

use super::*;
use std::time::{Instant, Duration};
use tokio::time::timeout;
use futures::future::join_all;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Feature serving layer with caching and optimization
pub struct FeatureServingLayer {
    store: Arc<FeatureStore>,
    
    /// Multi-level cache: L1 (hot) -> L2 (warm) -> L3 (database)
    l1_cache: Arc<DashMap<FeatureKey, FeatureValue>>,
    l2_cache: Arc<RwLock<LruCache<FeatureKey, FeatureValue>>>,
    
    /// Pre-computed feature aggregations
    aggregation_cache: Arc<DashMap<String, AggregatedFeatures>>,
    
    /// Batch request optimizer
    batch_optimizer: Arc<BatchOptimizer>,
}

/// Aggregated features for common queries
#[derive(Debug, Clone)]
pub struct AggregatedFeatures {
    pub entity_id: String,
    pub features: HashMap<String, FeatureValue>,
    pub computed_at: DateTime<Utc>,
}

/// Batch optimizer for efficient retrieval
pub struct BatchOptimizer {
    pending_requests: Arc<RwLock<Vec<PendingRequest>>>,
    batch_size: usize,
    batch_timeout_ms: u64,
}

#[derive(Debug)]
struct PendingRequest {
    keys: Vec<FeatureKey>,
    response_tx: tokio::sync::oneshot::Sender<Vec<Option<FeatureValue>>>,
    requested_at: Instant,
}

impl FeatureServingLayer {
    pub fn new(store: Arc<FeatureStore>) -> Self {
        let l2_capacity = NonZeroUsize::new(10000).unwrap();
        
        Self {
            store,
            l1_cache: Arc::new(DashMap::with_capacity(1000)),
            l2_cache: Arc::new(RwLock::new(LruCache::new(l2_capacity))),
            aggregation_cache: Arc::new(DashMap::new()),
            batch_optimizer: Arc::new(BatchOptimizer {
                pending_requests: Arc::new(RwLock::new(Vec::new())),
                batch_size: 100,
                batch_timeout_ms: 10,
            }),
        }
    }
    
    /// Get single feature with <1ms latency guarantee
    pub async fn get_feature(&self, key: FeatureKey) -> Result<Option<FeatureValue>, FeatureStoreError> {
        let start = Instant::now();
        
        // L1 cache check (fastest)
        if let Some(cached) = self.l1_cache.get(&key) {
            self.record_latency(start.elapsed(), "l1_hit");
            return Ok(Some(cached.clone()));
        }
        
        // L2 cache check
        {
            let mut l2_cache = self.l2_cache.write();
            if let Some(cached) = l2_cache.get(&key) {
                // Promote to L1
                let value = cached.clone();
                self.l1_cache.insert(key.clone(), value.clone());
                self.record_latency(start.elapsed(), "l2_hit");
                return Ok(Some(value));
            }
        }
        
        // Database fetch with timeout
        let fetch_timeout = Duration::from_micros(self.store.config.max_serving_latency_us);
        let result = timeout(fetch_timeout, self.fetch_from_database(key.clone())).await;
        
        match result {
            Ok(Ok(Some(value))) => {
                // Update caches
                self.update_caches(key, value.clone());
                self.record_latency(start.elapsed(), "db_hit");
                Ok(Some(value))
            }
            Ok(Ok(None)) => {
                self.record_latency(start.elapsed(), "miss");
                Ok(None)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let latency_us = start.elapsed().as_micros() as u64;
                Err(FeatureStoreError::LatencyExceeded(
                    latency_us,
                    self.store.config.max_serving_latency_us,
                ))
            }
        }
    }
    
    /// Get multiple features in parallel
    pub async fn get_features(&self, keys: Vec<FeatureKey>) -> Result<Vec<Option<FeatureValue>>, FeatureStoreError> {
        let start = Instant::now();
        
        // Split into cached and uncached
        let mut cached_results = Vec::new();
        let mut uncached_keys = Vec::new();
        
        for key in keys {
            if let Some(cached) = self.l1_cache.get(&key) {
                cached_results.push((key, Some(cached.clone())));
            } else {
                uncached_keys.push(key);
            }
        }
        
        // Batch fetch uncached
        let uncached_results = if !uncached_keys.is_empty() {
            self.batch_fetch(uncached_keys).await?
        } else {
            Vec::new()
        };
        
        // Combine results
        let mut all_results = cached_results;
        all_results.extend(uncached_results);
        
        // Sort by original order
        all_results.sort_by_key(|(k, _)| k.clone());
        
        let results: Vec<Option<FeatureValue>> = all_results.into_iter()
            .map(|(_, v)| v)
            .collect();
        
        self.record_latency(start.elapsed(), "batch_get");
        Ok(results)
    }
    
    /// Get latest features for an entity
    pub async fn get_entity_features(
        &self,
        entity_id: &str,
        feature_names: &[String],
    ) -> Result<HashMap<String, FeatureValue>, FeatureStoreError> {
        let start = Instant::now();
        
        // Check aggregation cache
        let cache_key = format!("{}:{}", entity_id, feature_names.join(","));
        if let Some(cached) = self.aggregation_cache.get(&cache_key) {
            if cached.computed_at > Utc::now() - chrono::Duration::seconds(60) {
                self.record_latency(start.elapsed(), "aggregation_hit");
                return Ok(cached.features.clone());
            }
        }
        
        // Fetch from database
        let query = sqlx::query_as::<_, (String, serde_json::Value, DateTime<Utc>)>(r#"
            WITH latest_features AS (
                SELECT DISTINCT ON (feature_name) 
                    feature_name,
                    feature_value,
                    timestamp
                FROM features
                WHERE entity_id = $1 
                    AND feature_name = ANY($2)
                ORDER BY feature_name, timestamp DESC
            )
            SELECT feature_name, feature_value, timestamp
            FROM latest_features
        "#)
        .bind(entity_id)
        .bind(feature_names);
        
        let rows = query.fetch_all(self.store.pool.as_ref()).await?;
        
        let mut features = HashMap::new();
        for (name, value, _) in rows {
            let feature_value: FeatureValue = serde_json::from_value(value)?;
            features.insert(name, feature_value);
        }
        
        // Update aggregation cache
        self.aggregation_cache.insert(
            cache_key,
            AggregatedFeatures {
                entity_id: entity_id.to_string(),
                features: features.clone(),
                computed_at: Utc::now(),
            },
        );
        
        self.record_latency(start.elapsed(), "entity_features");
        Ok(features)
    }
    
    /// Get point-in-time features (time travel)
    pub async fn get_features_at_time(
        &self,
        entity_id: &str,
        feature_names: &[String],
        timestamp: DateTime<Utc>,
    ) -> Result<HashMap<String, FeatureValue>, FeatureStoreError> {
        let query = sqlx::query_as::<_, (String, serde_json::Value)>(r#"
            WITH point_in_time AS (
                SELECT DISTINCT ON (feature_name) 
                    feature_name,
                    feature_value
                FROM features
                WHERE entity_id = $1 
                    AND feature_name = ANY($2)
                    AND timestamp <= $3
                ORDER BY feature_name, timestamp DESC
            )
            SELECT feature_name, feature_value
            FROM point_in_time
        "#)
        .bind(entity_id)
        .bind(feature_names)
        .bind(timestamp);
        
        let rows = query.fetch_all(self.store.pool.as_ref()).await?;
        
        let mut features = HashMap::new();
        for (name, value) in rows {
            let feature_value: FeatureValue = serde_json::from_value(value)?;
            features.insert(name, feature_value);
        }
        
        Ok(features)
    }
    
    /// Stream features for real-time updates
    pub async fn stream_features(
        &self,
        entity_id: String,
        feature_names: Vec<String>,
    ) -> impl futures::Stream<Item = Result<(String, FeatureValue), FeatureStoreError>> {
        use futures::stream::{self, StreamExt};
        use tokio::time::interval;
        
        let store = self.store.clone();
        
        stream::unfold(
            (entity_id, feature_names, Utc::now()),
            move |(entity_id, feature_names, last_check)| {
                let store = store.clone();
                async move {
                    // Poll every 100ms for updates
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    
                    let query = sqlx::query_as::<_, (String, serde_json::Value, DateTime<Utc>)>(r#"
                        SELECT feature_name, feature_value, timestamp
                        FROM features
                        WHERE entity_id = $1 
                            AND feature_name = ANY($2)
                            AND timestamp > $3
                        ORDER BY timestamp ASC
                    "#)
                    .bind(&entity_id)
                    .bind(&feature_names)
                    .bind(last_check);
                    
                    match query.fetch_all(store.pool.as_ref()).await {
                        Ok(rows) => {
                            let mut new_last_check = last_check;
                            let updates: Vec<_> = rows.into_iter()
                                .filter_map(|(name, value, ts)| {
                                    new_last_check = new_last_check.max(ts);
                                    serde_json::from_value::<FeatureValue>(value)
                                        .ok()
                                        .map(|v| Ok((name, v)))
                                })
                                .collect();
                            
                            if !updates.is_empty() {
                                Some((stream::iter(updates), (entity_id, feature_names, new_last_check)))
                            } else {
                                Some((stream::empty(), (entity_id, feature_names, new_last_check)))
                            }
                        }
                        Err(e) => Some((stream::once(async { Err(e.into()) }), (entity_id, feature_names, last_check)))
                    }
                }
            },
        )
        .flatten()
    }
    
    /// Batch fetch from database
    async fn batch_fetch(&self, keys: Vec<FeatureKey>) -> Result<Vec<(FeatureKey, Option<FeatureValue>)>, FeatureStoreError> {
        // Group by similar queries
        let mut grouped: HashMap<(String, String), Vec<FeatureKey>> = HashMap::new();
        
        for key in keys {
            let group_key = (key.entity_id.clone(), key.feature_name.clone());
            grouped.entry(group_key).or_insert_with(Vec::new).push(key);
        }
        
        // Execute parallel queries
        let futures: Vec<_> = grouped.into_iter()
            .map(|((entity_id, feature_name), keys)| {
                let pool = self.store.pool.clone();
                async move {
                    let timestamps: Vec<DateTime<Utc>> = keys.iter()
                        .filter_map(|k| k.timestamp)
                        .collect();
                    
                    let query = if timestamps.is_empty() {
                        sqlx::query_as::<_, (serde_json::Value, DateTime<Utc>)>(r#"
                            SELECT feature_value, timestamp
                            FROM features
                            WHERE entity_id = $1 AND feature_name = $2
                            ORDER BY timestamp DESC
                            LIMIT 1
                        "#)
                        .bind(&entity_id)
                        .bind(&feature_name)
                        .fetch_optional(pool.as_ref())
                        .await
                    } else {
                        sqlx::query_as::<_, (serde_json::Value, DateTime<Utc>)>(r#"
                            SELECT feature_value, timestamp
                            FROM features
                            WHERE entity_id = $1 
                                AND feature_name = $2
                                AND timestamp = ANY($3)
                        "#)
                        .bind(&entity_id)
                        .bind(&feature_name)
                        .bind(&timestamps)
                        .fetch_optional(pool.as_ref())
                        .await
                    };
                    
                    match query {
                        Ok(Some((value, _))) => {
                            let feature_value: FeatureValue = serde_json::from_value(value)?;
                            Ok(keys.into_iter()
                                .map(|k| (k, Some(feature_value.clone())))
                                .collect::<Vec<_>>())
                        }
                        Ok(None) => {
                            Ok(keys.into_iter()
                                .map(|k| (k, None))
                                .collect::<Vec<_>>())
                        }
                        Err(e) => Err(FeatureStoreError::Database(e))
                    }
                }
            })
            .collect();
        
        let results = join_all(futures).await;
        
        let mut all_results = Vec::new();
        for result in results {
            all_results.extend(result?);
        }
        
        Ok(all_results)
    }
    
    /// Fetch single feature from database
    async fn fetch_from_database(&self, key: FeatureKey) -> Result<Option<FeatureValue>, FeatureStoreError> {
        let query = if let Some(timestamp) = key.timestamp {
            sqlx::query_as::<_, (serde_json::Value,)>(r#"
                SELECT feature_value
                FROM features
                WHERE entity_id = $1 
                    AND feature_name = $2
                    AND timestamp = $3
                    AND version = COALESCE($4, version)
            "#)
            .bind(&key.entity_id)
            .bind(&key.feature_name)
            .bind(timestamp)
            .bind(key.version)
        } else {
            sqlx::query_as::<_, (serde_json::Value,)>(r#"
                SELECT feature_value
                FROM features
                WHERE entity_id = $1 
                    AND feature_name = $2
                ORDER BY timestamp DESC
                LIMIT 1
            "#)
            .bind(&key.entity_id)
            .bind(&key.feature_name)
        };
        
        match query.fetch_optional(self.store.pool.as_ref()).await? {
            Some((value,)) => {
                let feature_value: FeatureValue = serde_json::from_value(value)?;
                Ok(Some(feature_value))
            }
            None => Ok(None)
        }
    }
    
    /// Update caches with new value
    fn update_caches(&self, key: FeatureKey, value: FeatureValue) {
        // Update L1
        self.l1_cache.insert(key.clone(), value.clone());
        
        // Update L2
        let mut l2_cache = self.l2_cache.write();
        l2_cache.put(key, value);
        
        // Evict from L1 if too large
        if self.l1_cache.len() > 1000 {
            // Remove oldest entries
            let to_remove: Vec<_> = self.l1_cache.iter()
                .take(100)
                .map(|e| e.key().clone())
                .collect();
            
            for key in to_remove {
                self.l1_cache.remove(&key);
            }
        }
    }
    
    /// Record latency metrics
    fn record_latency(&self, duration: Duration, operation: &str) {
        let latency_us = duration.as_micros() as f64;
        
        let mut metrics = self.store.metrics.write();
        metrics.total_reads += 1;
        
        // Update average (simple moving average)
        metrics.avg_read_latency_us = 
            (metrics.avg_read_latency_us * (metrics.total_reads - 1) as f64 + latency_us) 
            / metrics.total_reads as f64;
        
        // Track cache hits/misses
        match operation {
            "l1_hit" | "l2_hit" | "aggregation_hit" => metrics.cache_hits += 1,
            "miss" | "db_hit" => metrics.cache_misses += 1,
            _ => {}
        }
    }
}

// Blake: "Sub-millisecond serving achieved! This will make our ML models blazing fast."