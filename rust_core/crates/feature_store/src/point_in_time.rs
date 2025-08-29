use domain_types::Event;
use domain_types::ValidationResult;
// Point-in-Time Correctness for Backtesting
// DEEP DIVE: Prevent data leakage and ensure temporal consistency

use std::sync::Arc;
use std::collections::{HashMap, BTreeMap};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use tracing::{info, debug, warn, instrument};
use serde::{Deserialize, Serialize};

use crate::offline_store::{OfflineStore, FeatureWrite};
use crate::online_store::FeatureVector;

/// Point-in-time correctness configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct PointInTimeConfig {
    pub feature_lag_ms: i64, // Minimum lag between event and feature availability
    pub max_lookback_days: i64,
    pub enable_caching: bool,
    pub cache_size: usize,
}

impl Default for PointInTimeConfig {
    fn default() -> Self {
        Self {
            feature_lag_ms: 100, // 100ms minimum lag (realistic for HFT)
            max_lookback_days: 365,
            enable_caching: true,
            cache_size: 10000,
        }
    }
}

/// Point-in-time correctness engine
/// TODO: Add docs
pub struct PointInTimeCorrectness {
    offline_store: Arc<OfflineStore>,
    config: PointInTimeConfig,
    cache: Option<Arc<parking_lot::RwLock<lru::LruCache<String, CachedFeatures>>>>,
}

#[derive(Clone)]
struct CachedFeatures {
    features: BTreeMap<DateTime<Utc>, FeatureVector>,
    last_accessed: std::time::Instant,
}

impl PointInTimeCorrectness {
    pub fn new(offline_store: Arc<OfflineStore>) -> Self {
        Self::with_config(offline_store, PointInTimeConfig::default())
    }
    
    pub fn with_config(
        offline_store: Arc<OfflineStore>,
        config: PointInTimeConfig,
    ) -> Self {
        let cache = if config.enable_caching {
            Some(Arc::new(parking_lot::RwLock::new(
                lru::LruCache::new(
                    std::num::NonZeroUsize::new(config.cache_size).unwrap()
                )
            )))
        } else {
            None
        };
        
        Self {
            offline_store,
            config,
            cache,
        }
    }
    
    /// Get features with point-in-time correctness
    #[instrument(skip(self))]
    pub async fn get_features_at_time(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureVector>> {
        debug!(
            "Getting PIT features for {} entities at {}",
            entity_ids.len(),
            timestamp
        );
        
        // Apply feature lag to prevent leakage
        let adjusted_timestamp = timestamp - Duration::milliseconds(self.config.feature_lag_ms);
        
        let mut results = Vec::new();
        
        for entity_id in entity_ids {
            // Check cache first
            if let Some(cached) = self.get_from_cache(&entity_id, adjusted_timestamp).await {
                results.push(cached);
                continue;
            }
            
            // Fetch from offline store with temporal constraints
            let features = self.fetch_with_constraints(
                &entity_id,
                &feature_names,
                adjusted_timestamp,
            ).await?;
            
            // Update cache
            if let Some(ref features) = features {
                self.update_cache(entity_id.clone(), adjusted_timestamp, features.clone()).await;
                results.push(features.clone());
            }
        }
        
        Ok(results)
    }
    
    /// Perform temporal join with feature lag enforcement
    pub async fn temporal_join(
        &self,
        events: Vec<Event>,
        feature_names: Vec<String>,
    ) -> Result<Vec<EnrichedEvent>> {
        info!("Performing temporal join on {} events", events.len());
        
        let mut enriched = Vec::with_capacity(events.len());
        
        for event in events {
            // Get features as they would have been available at event time
            let features = self.get_features_at_time(
                vec![event.entity_id.clone()],
                feature_names.clone(),
                event.timestamp,
            ).await?;
            
            enriched.push(EnrichedEvent {
                event,
                features: features.into_iter().next(),
            });
        }
        
        Ok(enriched)
    }
    
    /// Validate no data leakage in feature set
    pub async fn validate_no_leakage(
        &self,
        feature_timestamps: HashMap<String, DateTime<Utc>>,
        prediction_time: DateTime<Utc>,
    ) -> Result<ValidationResult> {
        let mut violations = Vec::new();
        
        for (feature_name, feature_time) in feature_timestamps {
            // Check if feature was created after prediction time
            if feature_time > prediction_time {
                violations.push(LeakageViolation {
                    feature_name: feature_name.clone(),
                    feature_time,
                    prediction_time,
                    violation_type: ViolationType::FutureLeak,
                });
            }
            
            // Check if feature violates minimum lag requirement
            let time_diff = prediction_time.signed_duration_since(feature_time);
            if time_diff.num_milliseconds() < self.config.feature_lag_ms {
                violations.push(LeakageViolation {
                    feature_name,
                    feature_time,
                    prediction_time,
                    violation_type: ViolationType::InsufficientLag,
                });
            }
        }
        
        Ok(ValidationResult {
            is_valid: violations.is_empty(),
            violations,
            checked_at: Utc::now(),
        })
    }
    
    /// Create training dataset with PIT correctness
    pub async fn create_training_dataset(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        labels: Vec<Label>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<TrainingDataset> {
        info!(
            "Creating PIT-correct training dataset: {} entities, {} features, {} labels",
            entity_ids.len(),
            feature_names.len(),
            labels.len()
        );
        
        let mut samples = Vec::new();
        
        for label in labels {
            // Ensure we only use features available before the label
            let feature_cutoff = label.timestamp - Duration::milliseconds(self.config.feature_lag_ms);
            
            // Get features at the appropriate time
            let features = self.offline_store.get_features_at_time(
                vec![label.entity_id.clone()],
                feature_names.clone(),
                feature_cutoff,
            ).await?;
            
            if let Some(feature_vec) = features.into_iter().next() {
                samples.push(TrainingSample {
                    features: feature_vec,
                    label: label.value,
                    timestamp: label.timestamp,
                    entity_id: label.entity_id,
                });
            }
        }
        
        // Validate temporal ordering
        self.validate_temporal_ordering(&samples)?;
        
        Ok(TrainingDataset {
            samples,
            feature_names,
            created_at: Utc::now(),
            config: self.config.clone(),
        })
    }
    
    /// Fetch features with temporal constraints
    async fn fetch_with_constraints(
        &self,
        entity_id: &str,
        feature_names: &[String],
        timestamp: DateTime<Utc>,
    ) -> Result<Option<FeatureVector>> {
        // Calculate lookback window
        let lookback_start = timestamp - Duration::days(self.config.max_lookback_days);
        
        // Query features with constraints
        let conn = self.offline_store.pool.get().await?;
        
        let mut features = Vec::new();
        let mut actual_names = Vec::new();
        
        for feature_name in feature_names {
            let row = conn.query_opt(
                "SELECT feature_value, event_timestamp, created_timestamp
                 FROM feature_store.features
                 WHERE entity_id = $1 
                   AND feature_name = $2
                   AND event_timestamp <= $3
                   AND event_timestamp >= $4
                   AND created_timestamp <= $3  -- Ensure feature was created before query time
                 ORDER BY event_timestamp DESC
                 LIMIT 1",
                &[&entity_id, feature_name, &timestamp, &lookback_start],
            ).await?;
            
            if let Some(row) = row {
                let value: f64 = row.get(0);
                let event_time: DateTime<Utc> = row.get(1);
                let created_time: DateTime<Utc> = row.get(2);
                
                // Additional validation
                if created_time <= timestamp {
                    features.push(value);
                    actual_names.push(feature_name.clone());
                } else {
                    warn!(
                        "Feature {} created after query time, skipping to prevent leakage",
                        feature_name
                    );
                }
            }
        }
        
        if features.is_empty() {
            Ok(None)
        } else {
            Ok(Some(FeatureVector {
                entity_id: entity_id.to_string(),
                features,
                feature_names: actual_names,
                timestamp,
                metadata: Some(serde_json::json!({
                    "pit_corrected": true,
                    "lag_ms": self.config.feature_lag_ms
                })),
            }))
        }
    }
    
    /// Get from cache
    async fn get_from_cache(
        &self,
        entity_id: &str,
        timestamp: DateTime<Utc>,
    ) -> Option<FeatureVector> {
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache.write();
            
            if let Some(cached) = cache_guard.get_mut(entity_id) {
                // Find the most recent features before timestamp
                let range = cached.features.range(..=timestamp);
                if let Some((_, features)) = range.last() {
                    cached.last_accessed = std::time::Instant::now();
                    return Some(features.clone());
                }
            }
        }
        None
    }
    
    /// Update cache
    async fn update_cache(
        &self,
        entity_id: String,
        timestamp: DateTime<Utc>,
        features: FeatureVector,
    ) {
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache.write();
            
            let cached = cache_guard.get_mut(&entity_id)
                .map(|c| {
                    c.features.insert(timestamp, features);
                    c.last_accessed = std::time::Instant::now();
                })
                .or_else(|| {
                    let mut feature_map = BTreeMap::new();
                    feature_map.insert(timestamp, features);
                    
                    cache_guard.put(entity_id, CachedFeatures {
                        features: feature_map,
                        last_accessed: std::time::Instant::now(),
                    });
                    Some(())
                });
        }
    }
    
    /// Validate temporal ordering in dataset
    fn validate_temporal_ordering(&self, samples: &[TrainingSample]) -> Result<()> {
        for window in samples.windows(2) {
            if window[0].timestamp > window[1].timestamp {
                return Err(anyhow::anyhow!(
                    "Temporal ordering violation: samples not in chronological order"
                ));
            }
        }
        Ok(())
    }
}

/// Temporal join helper
/// TODO: Add docs
pub struct TemporalJoin {
    pit: PointInTimeCorrectness,
}

impl TemporalJoin {
    pub fn new(offline_store: Arc<OfflineStore>) -> Self {
        Self {
            pit: PointInTimeCorrectness::new(offline_store),
        }
    }
    
    /// Join events with features respecting temporal constraints
    pub async fn join(
        &self,
        events: Vec<Event>,
        feature_names: Vec<String>,
    ) -> Result<Vec<EnrichedEvent>> {
        self.pit.temporal_join(events, feature_names).await
    }
}

/// Event structure for temporal join
#[derive(Debug, Clone, Serialize, Deserialize)]
// ELIMINATED: use domain_types::Event
// pub struct Event {
    pub entity_id: String,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

/// Enriched event with features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct EnrichedEvent {
    pub event: Event,
    pub features: Option<FeatureVector>,
}

/// Label for training
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Label {
    pub entity_id: String,
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

/// Training sample with PIT-correct features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TrainingSample {
    pub features: FeatureVector,
    pub label: f64,
    pub timestamp: DateTime<Utc>,
    pub entity_id: String,
}

/// Training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TrainingDataset {
    pub samples: Vec<TrainingSample>,
    pub feature_names: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub config: PointInTimeConfig,
}

/// Leakage violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum ViolationType {
    FutureLeak,       // Feature from future
    InsufficientLag,  // Feature too recent
}

/// Leakage violation
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct LeakageViolation {
    pub feature_name: String,
    pub feature_time: DateTime<Utc>,
    pub prediction_time: DateTime<Utc>,
    pub violation_type: ViolationType,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
// ELIMINATED: use domain_types::ValidationResult
// pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<LeakageViolation>,
    pub checked_at: DateTime<Utc>,
}