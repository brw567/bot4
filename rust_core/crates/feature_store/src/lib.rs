// Layer 1.5: Feature Store Implementation
// DEEP DIVE - Production-ready centralized feature management
//
// Architecture:
// - Redis cluster for <1ms online serving (hot data)
// - TimescaleDB for <10ms historical serving (warm data)
// - Apache Arrow for zero-copy data exchange
// - Point-in-time correctness for backtesting
// - Feature versioning with lineage tracking
// - A/B testing support with experiment tracking
// - Drift detection using KL divergence and PSI
//
// External Research Applied:
// - Uber's Michelangelo platform architecture
// - Two Sigma's feature infrastructure (380+ PB scale)
// - Airbnb's Zipline feature pipeline
// - Jane Street's <1μs decision systems
// - Tecton's sub-10ms serving patterns
// - Hopsworks' RonDB for sub-millisecond latency

pub mod online_store;
pub mod offline_store;
pub mod feature_registry;
pub mod feature_pipeline;
pub mod point_in_time;
pub mod drift_detection;
pub mod ab_testing;
pub mod monitoring;

use std::sync::Arc;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};

// Re-exports
pub use online_store::{OnlineStore, RedisConfig, FeatureVector};
pub use offline_store::{OfflineStore, TimescaleConfig as FeatureTimescaleConfig};
pub use feature_registry::{FeatureRegistry, FeatureDefinition, FeatureMetadata};
pub use feature_pipeline::{FeaturePipeline, StreamingEngine};
pub use point_in_time::{PointInTimeCorrectness, TemporalJoin};
pub use drift_detection::{DriftDetector, DriftMetrics, DriftAlert};
pub use ab_testing::{ABTestManager, Experiment, ExperimentResults};

/// Main Feature Store configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureStoreConfig {
    pub online_store: online_store::RedisConfig,
    pub offline_store: offline_store::TimescaleConfig,
    pub feature_registry: feature_registry::RegistryConfig,
    pub pipeline: feature_pipeline::PipelineConfig,
    pub drift_detection: drift_detection::DriftConfig,
    pub ab_testing: ab_testing::ABConfig,
    pub monitoring: monitoring::MonitoringConfig,
}

impl Default for FeatureStoreConfig {
    fn default() -> Self {
        Self {
            online_store: Default::default(),
            offline_store: Default::default(),
            feature_registry: Default::default(),
            pipeline: Default::default(),
            drift_detection: Default::default(),
            ab_testing: Default::default(),
            monitoring: Default::default(),
        }
    }
}

/// Main Feature Store interface
pub struct FeatureStore {
    config: FeatureStoreConfig,
    online_store: Arc<OnlineStore>,
    offline_store: Arc<OfflineStore>,
    registry: Arc<FeatureRegistry>,
    pipeline: Arc<FeaturePipeline>,
    drift_detector: Arc<DriftDetector>,
    ab_manager: Arc<ABTestManager>,
    monitor: Arc<monitoring::FeatureMonitor>,
}

impl FeatureStore {
    /// Create new Feature Store instance
    pub async fn new(config: FeatureStoreConfig) -> Result<Self> {
        info!("Initializing Feature Store for HFT with <10ms latency target");
        
        // Initialize components
        let online_store = Arc::new(
            OnlineStore::new(config.online_store.clone()).await?
        );
        
        let offline_store = Arc::new(
            OfflineStore::new(config.offline_store.clone()).await?
        );
        
        let registry = Arc::new(
            FeatureRegistry::new(config.feature_registry.clone()).await?
        );
        
        let pipeline = Arc::new(
            FeaturePipeline::new(
                config.pipeline.clone(),
                online_store.clone(),
                offline_store.clone(),
                registry.clone(),
            ).await?
        );
        
        let drift_detector = Arc::new(
            DriftDetector::new(
                config.drift_detection.clone(),
                registry.clone(),
            ).await?
        );
        
        let ab_manager = Arc::new(
            ABTestManager::new(
                config.ab_testing.clone(),
                registry.clone(),
            ).await?
        );
        
        let monitor = Arc::new(
            monitoring::FeatureMonitor::new(
                config.monitoring.clone(),
                online_store.clone(),
                offline_store.clone(),
            ).await?
        );
        
        // Start background tasks
        pipeline.start_streaming().await?;
        drift_detector.start_monitoring().await?;
        monitor.start_collection().await?;
        
        info!("Feature Store initialized successfully");
        
        Ok(Self {
            config,
            online_store,
            offline_store,
            registry,
            pipeline,
            drift_detector,
            ab_manager,
            monitor,
        })
    }
    
    /// Get features for online serving (<10ms SLA)
    pub async fn get_online_features(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        experiment_id: Option<String>,
    ) -> Result<Vec<FeatureVector>> {
        let start = std::time::Instant::now();
        
        // Check if part of A/B test
        let features_to_fetch = if let Some(exp_id) = experiment_id {
            self.ab_manager.get_treatment_features(&exp_id, &feature_names).await?
        } else {
            feature_names
        };
        
        // Fetch from online store (Redis)
        let result = self.online_store
            .get_features(entity_ids, features_to_fetch)
            .await?;
        
        // Record latency
        let latency = start.elapsed();
        self.monitor.record_serving_latency(latency).await;
        
        if latency.as_millis() > 10 {
            warn!("Online serving exceeded 10ms SLA: {:?}", latency);
        }
        
        Ok(result)
    }
    
    /// Get historical features with point-in-time correctness
    pub async fn get_historical_features(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
        timestamp: DateTime<Utc>,
        point_in_time: bool,
    ) -> Result<Vec<FeatureVector>> {
        if point_in_time {
            // Ensure no data leakage for backtesting
            let pit_corrector = point_in_time::PointInTimeCorrectness::new(
                self.offline_store.clone()
            );
            
            pit_corrector.get_features_at_time(
                entity_ids,
                feature_names,
                timestamp,
            ).await
        } else {
            // Regular historical fetch
            self.offline_store
                .get_features_at_time(entity_ids, feature_names, timestamp)
                .await
        }
    }
    
    /// Register new feature definition
    pub async fn register_feature(
        &self,
        definition: FeatureDefinition,
    ) -> Result<String> {
        // Validate feature definition
        definition.validate()?;
        
        // Register in metadata store
        let feature_id = self.registry.register(definition.clone()).await?;
        
        // Initialize in stores
        self.online_store.initialize_feature(&feature_id).await?;
        self.offline_store.initialize_feature(&feature_id).await?;
        
        // Set up drift monitoring
        self.drift_detector.add_feature(&feature_id).await?;
        
        info!("Registered new feature: {}", feature_id);
        Ok(feature_id)
    }
    
    /// Update features in the store
    pub async fn update_features(
        &self,
        updates: Vec<FeatureUpdate>,
    ) -> Result<()> {
        // Validate updates
        for update in &updates {
            self.registry.validate_update(update).await?;
        }
        
        // Apply updates through pipeline
        self.pipeline.process_updates(updates.clone()).await?;
        
        // Check for drift
        for update in &updates {
            if let Some(alert) = self.drift_detector.check_update(update).await? {
                self.handle_drift_alert(alert).await?;
            }
        }
        
        Ok(())
    }
    
    /// Start A/B test for features
    pub async fn start_ab_test(
        &self,
        experiment: Experiment,
    ) -> Result<String> {
        let exp_id = self.ab_manager.create_experiment(experiment).await?;
        
        info!("Started A/B test: {}", exp_id);
        Ok(exp_id)
    }
    
    /// Get A/B test results
    pub async fn get_ab_results(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentResults> {
        self.ab_manager.get_results(experiment_id).await
    }
    
    /// Get feature lineage
    pub async fn get_lineage(
        &self,
        feature_id: &str,
    ) -> Result<FeatureLineage> {
        self.registry.get_lineage(feature_id).await
    }
    
    /// Monitor feature health
    pub async fn get_health_metrics(&self) -> Result<HealthMetrics> {
        Ok(HealthMetrics {
            online_store_latency_ms: self.monitor.get_online_latency_p99().await?,
            offline_store_latency_ms: self.monitor.get_offline_latency_p99().await?,
            feature_freshness_ms: self.monitor.get_freshness_p99().await?,
            drift_alerts: self.drift_detector.get_active_alerts().await?,
            active_experiments: self.ab_manager.get_active_count().await?,
            total_features: self.registry.get_feature_count().await?,
            storage_usage_gb: self.monitor.get_storage_usage().await?,
        })
    }
    
    /// Handle drift alert
    async fn handle_drift_alert(&self, alert: DriftAlert) -> Result<()> {
        match alert.severity {
            drift_detection::Severity::Critical => {
                error!("Critical feature drift detected: {:?}", alert);
                // Could trigger circuit breaker or pause trading
            }
            drift_detection::Severity::High => {
                warn!("High feature drift detected: {:?}", alert);
                // Could notify monitoring system
            }
            _ => {
                debug!("Feature drift detected: {:?}", alert);
            }
        }
        Ok(())
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Feature Store");
        
        self.pipeline.stop().await?;
        self.drift_detector.stop().await?;
        self.monitor.stop().await?;
        self.online_store.close().await?;
        self.offline_store.close().await?;
        
        Ok(())
    }
}

/// Feature update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureUpdate {
    pub entity_id: String,
    pub feature_id: String,
    pub value: FeatureValue,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Feature value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureValue {
    Float(f64),
    Integer(i64),
    String(String),
    Vector(Vec<f64>),
    Tensor(ndarray::ArrayD<f64>),
}

/// Feature lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureLineage {
    pub feature_id: String,
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub dependencies: Vec<String>,
    pub transformations: Vec<String>,
    pub data_sources: Vec<String>,
}

/// Health metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub online_store_latency_ms: f64,
    pub offline_store_latency_ms: f64,
    pub feature_freshness_ms: f64,
    pub drift_alerts: usize,
    pub active_experiments: usize,
    pub total_features: usize,
    pub storage_usage_gb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_feature_store_initialization() {
        let config = FeatureStoreConfig::default();
        let store = FeatureStore::new(config).await;
        assert!(store.is_ok());
    }
    
    #[tokio::test]
    async fn test_online_serving_latency() {
        // Test that online serving meets <10ms SLA
        let config = FeatureStoreConfig::default();
        let store = FeatureStore::new(config).await.unwrap();
        
        let start = std::time::Instant::now();
        let _ = store.get_online_features(
            vec!["entity1".to_string()],
            vec!["feature1".to_string()],
            None,
        ).await;
        let latency = start.elapsed();
        
        assert!(latency.as_millis() < 10, "Online serving must be <10ms");
    }
}