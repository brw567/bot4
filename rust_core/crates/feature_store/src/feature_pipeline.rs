// Feature Pipeline - Streaming and Batch Processing
// DEEP DIVE: Real-time feature computation and materialization

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, instrument};
use tokio::sync::{RwLock, mpsc};
use crossbeam_channel::{bounded, Sender, Receiver};

use crate::{OnlineStore, OfflineStore, FeatureRegistry, FeatureUpdate, FeatureValue};

/// Pipeline configuration
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
    pub parallelism: usize,
    pub buffer_size: usize,
    pub checkpoint_interval_seconds: u64,
    pub enable_streaming: bool,
    pub enable_batch: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            batch_timeout_ms: 100,
            parallelism: 8,
            buffer_size: 100000,
            checkpoint_interval_seconds: 60,
            enable_streaming: true,
            enable_batch: true,
        }
    }
}

/// Feature pipeline for processing updates
pub struct FeaturePipeline {
    config: PipelineConfig,
    online_store: Arc<OnlineStore>,
    offline_store: Arc<OfflineStore>,
    registry: Arc<FeatureRegistry>,
    
    // Streaming engine
    streaming_engine: Arc<StreamingEngine>,
    
    // Batch processor
    batch_processor: Arc<BatchProcessor>,
    
    // Transform registry
    transformers: Arc<RwLock<HashMap<String, Box<dyn FeatureTransformer + Send + Sync>>>>,
    
    // Pipeline state
    is_running: Arc<RwLock<bool>>,
}

impl FeaturePipeline {
    /// Create new feature pipeline
    pub async fn new(
        config: PipelineConfig,
        online_store: Arc<OnlineStore>,
        offline_store: Arc<OfflineStore>,
        registry: Arc<FeatureRegistry>,
    ) -> Result<Self> {
        info!("Initializing Feature Pipeline");
        
        let streaming_engine = Arc::new(StreamingEngine::new(
            config.clone(),
            online_store.clone(),
            offline_store.clone(),
        ));
        
        let batch_processor = Arc::new(BatchProcessor::new(
            config.clone(),
            offline_store.clone(),
        ));
        
        let pipeline = Self {
            config,
            online_store,
            offline_store,
            registry,
            streaming_engine,
            batch_processor,
            transformers: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        };
        
        // Register default transformers
        pipeline.register_default_transformers().await?;
        
        Ok(pipeline)
    }
    
    /// Start streaming pipeline
    pub async fn start_streaming(&self) -> Result<()> {
        if *self.is_running.read().await {
            return Ok(());
        }
        
        info!("Starting feature streaming pipeline");
        
        *self.is_running.write().await = true;
        
        if self.config.enable_streaming {
            self.streaming_engine.start().await?;
        }
        
        if self.config.enable_batch {
            self.batch_processor.start().await?;
        }
        
        Ok(())
    }
    
    /// Stop pipeline
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping feature pipeline");
        
        *self.is_running.write().await = false;
        
        self.streaming_engine.stop().await?;
        self.batch_processor.stop().await?;
        
        Ok(())
    }
    
    /// Process feature updates
    #[instrument(skip(self, updates))]
    pub async fn process_updates(&self, updates: Vec<FeatureUpdate>) -> Result<()> {
        debug!("Processing {} feature updates", updates.len());
        
        // Apply transformations
        let mut transformed = Vec::new();
        for update in updates {
            let result = self.apply_transformations(update).await?;
            transformed.extend(result);
        }
        
        // Write to online store (hot path)
        let online_updates: Vec<_> = transformed.iter()
            .map(|u| (u.entity_id.clone(), u.feature_id.clone(), 
                     match &u.value {
                         FeatureValue::Float(f) => *f,
                         _ => 0.0,
                     }))
            .collect();
        
        self.online_store.set_features(online_updates).await?;
        
        // Write to offline store (async)
        let offline_updates: Vec<_> = transformed.into_iter()
            .map(|u| crate::offline_store::FeatureWrite {
                entity_id: u.entity_id,
                feature_name: u.feature_id,
                value: match u.value {
                    FeatureValue::Float(f) => f,
                    _ => 0.0,
                },
                timestamp: u.timestamp,
                version: None,
                metadata: u.metadata,
            })
            .collect();
        
        self.offline_store.write_features(offline_updates).await?;
        
        Ok(())
    }
    
    /// Apply transformations to update
    async fn apply_transformations(&self, update: FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        let transformers = self.transformers.read().await;
        
        // Check if feature has transformations
        let definition = self.registry.get_feature_definition(&update.feature_id).await?;
        
        let mut results = vec![update.clone()];
        
        if let Some(transform_spec) = definition.transformations {
            // Parse transformation pipeline
            if let Some(transforms) = transform_spec.as_array() {
                for transform in transforms {
                    if let Some(name) = transform.as_str() {
                        if let Some(transformer) = transformers.get(name) {
                            let new_updates = transformer.transform(&update).await?;
                            results.extend(new_updates);
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Register default transformers
    async fn register_default_transformers(&self) -> Result<()> {
        let mut transformers = self.transformers.write().await;
        
        // Moving average transformer
        transformers.insert(
            "moving_average".to_string(),
            Box::new(MovingAverageTransformer::new(20)),
        );
        
        // Exponential moving average
        transformers.insert(
            "ema".to_string(),
            Box::new(EMATransformer::new(0.1)),
        );
        
        // Z-score normalization
        transformers.insert(
            "zscore".to_string(),
            Box::new(ZScoreTransformer::new()),
        );
        
        // Log transform
        transformers.insert(
            "log".to_string(),
            Box::new(LogTransformer::new()),
        );
        
        // Lag features
        transformers.insert(
            "lag_1".to_string(),
            Box::new(LagTransformer::new(1)),
        );
        
        info!("Registered {} default transformers", transformers.len());
        Ok(())
    }
}

/// Streaming engine for real-time processing
pub struct StreamingEngine {
    config: PipelineConfig,
    online_store: Arc<OnlineStore>,
    offline_store: Arc<OfflineStore>,
    
    // Channels for streaming
    input_tx: Option<Sender<FeatureUpdate>>,
    input_rx: Option<Receiver<FeatureUpdate>>,
    
    // Processing workers
    workers: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl StreamingEngine {
    /// Create new streaming engine
    fn new(
        config: PipelineConfig,
        online_store: Arc<OnlineStore>,
        offline_store: Arc<OfflineStore>,
    ) -> Self {
        let (tx, rx) = bounded(config.buffer_size);
        
        Self {
            config,
            online_store,
            offline_store,
            input_tx: Some(tx),
            input_rx: Some(rx),
            workers: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Start streaming workers
    async fn start(&self) -> Result<()> {
        info!("Starting {} streaming workers", self.config.parallelism);
        
        let mut workers = self.workers.write().await;
        
        for worker_id in 0..self.config.parallelism {
            let rx = self.input_rx.as_ref().unwrap().clone();
            let online_store = self.online_store.clone();
            let batch_size = self.config.batch_size;
            let batch_timeout_ms = self.config.batch_timeout_ms;
            
            let handle = tokio::spawn(async move {
                let mut batch = Vec::new();
                let mut last_flush = std::time::Instant::now();
                
                loop {
                    // Try to receive with timeout
                    match rx.recv_timeout(std::time::Duration::from_millis(batch_timeout_ms)) {
                        Ok(update) => {
                            batch.push(update);
                            
                            // Flush if batch is full
                            if batch.len() >= batch_size {
                                Self::flush_batch(&online_store, &mut batch, worker_id).await;
                                last_flush = std::time::Instant::now();
                            }
                        }
                        Err(_) => {
                            // Timeout - flush if we have data
                            if !batch.is_empty() && 
                               last_flush.elapsed().as_millis() > batch_timeout_ms as u128 {
                                Self::flush_batch(&online_store, &mut batch, worker_id).await;
                                last_flush = std::time::Instant::now();
                            }
                        }
                    }
                }
            });
            
            workers.push(handle);
        }
        
        Ok(())
    }
    
    /// Flush batch to store
    async fn flush_batch(
        online_store: &Arc<OnlineStore>,
        batch: &mut Vec<FeatureUpdate>,
        worker_id: usize,
    ) {
        if batch.is_empty() {
            return;
        }
        
        debug!("Worker {} flushing {} updates", worker_id, batch.len());
        
        let updates: Vec<_> = batch.drain(..)
            .map(|u| (u.entity_id, u.feature_id, 
                     match u.value {
                         FeatureValue::Float(f) => f,
                         _ => 0.0,
                     }))
            .collect();
        
        if let Err(e) = online_store.set_features(updates).await {
            warn!("Worker {} failed to flush batch: {}", worker_id, e);
        }
    }
    
    /// Stop streaming
    async fn stop(&self) -> Result<()> {
        let workers = self.workers.write().await;
        for handle in workers.iter() {
            handle.abort();
        }
        Ok(())
    }
}

/// Batch processor for offline features
struct BatchProcessor {
    config: PipelineConfig,
    offline_store: Arc<OfflineStore>,
    handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl BatchProcessor {
    fn new(config: PipelineConfig, offline_store: Arc<OfflineStore>) -> Self {
        Self {
            config,
            offline_store,
            handle: Arc::new(RwLock::new(None)),
        }
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting batch processor");
        
        let offline_store = self.offline_store.clone();
        let checkpoint_interval = self.config.checkpoint_interval_seconds;
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(checkpoint_interval)
            );
            
            loop {
                interval.tick().await;
                
                // In production, would process batch jobs here
                debug!("Batch processor checkpoint");
            }
        });
        
        *self.handle.write().await = Some(handle);
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.handle.write().await.take() {
            handle.abort();
        }
        Ok(())
    }
}

/// Feature transformer trait
#[async_trait]
trait FeatureTransformer: Send + Sync {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>>;
}

/// Moving average transformer
struct MovingAverageTransformer {
    window: usize,
    history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl MovingAverageTransformer {
    fn new(window: usize) -> Self {
        Self {
            window,
            history: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl FeatureTransformer for MovingAverageTransformer {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        if let FeatureValue::Float(value) = update.value {
            let mut history = self.history.write().await;
            let values = history.entry(update.entity_id.clone()).or_insert_with(Vec::new);
            
            values.push(value);
            if values.len() > self.window {
                values.remove(0);
            }
            
            let ma = values.iter().sum::<f64>() / values.len() as f64;
            
            Ok(vec![FeatureUpdate {
                entity_id: update.entity_id.clone(),
                feature_id: format!("{}_ma_{}", update.feature_id, self.window),
                value: FeatureValue::Float(ma),
                timestamp: update.timestamp,
                metadata: update.metadata.clone(),
            }])
        } else {
            Ok(vec![])
        }
    }
}

/// Exponential moving average transformer
struct EMATransformer {
    alpha: f64,
    state: Arc<RwLock<HashMap<String, f64>>>,
}

impl EMATransformer {
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            state: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl FeatureTransformer for EMATransformer {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        if let FeatureValue::Float(value) = update.value {
            let mut state = self.state.write().await;
            let ema = state.entry(update.entity_id.clone())
                .and_modify(|e| *e = self.alpha * value + (1.0 - self.alpha) * *e)
                .or_insert(value);
            
            Ok(vec![FeatureUpdate {
                entity_id: update.entity_id.clone(),
                feature_id: format!("{}_ema", update.feature_id),
                value: FeatureValue::Float(*ema),
                timestamp: update.timestamp,
                metadata: update.metadata.clone(),
            }])
        } else {
            Ok(vec![])
        }
    }
}

/// Z-score normalization transformer
struct ZScoreTransformer {
    stats: Arc<RwLock<HashMap<String, (f64, f64)>>>, // mean, std
}

impl ZScoreTransformer {
    fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl FeatureTransformer for ZScoreTransformer {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        if let FeatureValue::Float(value) = update.value {
            let stats = self.stats.read().await;
            
            // In production, would get stats from feature registry
            let (mean, std) = stats.get(&update.feature_id)
                .unwrap_or(&(0.0, 1.0));
            
            let zscore = if *std > 0.0 {
                (value - mean) / std
            } else {
                0.0
            };
            
            Ok(vec![FeatureUpdate {
                entity_id: update.entity_id.clone(),
                feature_id: format!("{}_zscore", update.feature_id),
                value: FeatureValue::Float(zscore),
                timestamp: update.timestamp,
                metadata: update.metadata.clone(),
            }])
        } else {
            Ok(vec![])
        }
    }
}

/// Log transformer
struct LogTransformer;

impl LogTransformer {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl FeatureTransformer for LogTransformer {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        if let FeatureValue::Float(value) = update.value {
            let log_value = if value > 0.0 {
                value.ln()
            } else {
                f64::NEG_INFINITY
            };
            
            Ok(vec![FeatureUpdate {
                entity_id: update.entity_id.clone(),
                feature_id: format!("{}_log", update.feature_id),
                value: FeatureValue::Float(log_value),
                timestamp: update.timestamp,
                metadata: update.metadata.clone(),
            }])
        } else {
            Ok(vec![])
        }
    }
}

/// Lag transformer
struct LagTransformer {
    lag: usize,
    history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl LagTransformer {
    fn new(lag: usize) -> Self {
        Self {
            lag,
            history: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl FeatureTransformer for LagTransformer {
    async fn transform(&self, update: &FeatureUpdate) -> Result<Vec<FeatureUpdate>> {
        if let FeatureValue::Float(value) = update.value {
            let mut history = self.history.write().await;
            let values = history.entry(update.entity_id.clone()).or_insert_with(Vec::new);
            
            values.push(value);
            
            if values.len() > self.lag {
                let lagged_value = values[values.len() - self.lag - 1];
                
                Ok(vec![FeatureUpdate {
                    entity_id: update.entity_id.clone(),
                    feature_id: format!("{}_lag_{}", update.feature_id, self.lag),
                    value: FeatureValue::Float(lagged_value),
                    timestamp: update.timestamp,
                    metadata: update.metadata.clone(),
                }])
            } else {
                Ok(vec![])
            }
        } else {
            Ok(vec![])
        }
    }
}

// Extension to FeatureRegistry
impl FeatureRegistry {
    /// Get feature definition (stub for compilation)
    pub async fn get_feature_definition(&self, feature_id: &str) -> Result<crate::feature_registry::FeatureDefinition> {
        // In production, would fetch from registry
        Err(anyhow::anyhow!("Feature not found: {}", feature_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_moving_average_transformer() {
        let transformer = MovingAverageTransformer::new(3);
        
        let update1 = FeatureUpdate {
            entity_id: "entity1".to_string(),
            feature_id: "price".to_string(),
            value: FeatureValue::Float(100.0),
            timestamp: Utc::now(),
            metadata: None,
        };
        
        let result1 = transformer.transform(&update1).await.unwrap();
        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].feature_id, "price_ma_3");
    }
    
    #[tokio::test]
    async fn test_ema_transformer() {
        let transformer = EMATransformer::new(0.2);
        
        let update = FeatureUpdate {
            entity_id: "entity1".to_string(),
            feature_id: "volume".to_string(),
            value: FeatureValue::Float(1000.0),
            timestamp: Utc::now(),
            metadata: None,
        };
        
        let result = transformer.transform(&update).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].feature_id, "volume_ema");
    }
}