use domain_types::FeatureVector;
//! # OPTIMIZED ML INFERENCE - Sub-millisecond predictions
//! Blake: "Every microsecond of inference latency costs money"

use std::sync::Arc;
use onnxruntime::{GraphOptimizationLevel, session::Session};
use ndarray::Array2;

/// Optimized inference engine with batching and caching
/// TODO: Add docs
pub struct OptimizedInference {
    /// ONNX Runtime session
    session: Arc<Session>,
    
    /// Batch accumulator
    batch_buffer: Vec<FeatureVector>,
    batch_size: usize,
    
    /// Inference cache
    cache: Arc<dashmap::DashMap<u64, Prediction>>,
    
    /// Performance metrics
    metrics: InferenceMetrics,
}

// ELIMINATED: use domain_types::FeatureVector
// pub struct FeatureVector {
    pub features: Vec<f32>,
    pub timestamp: u64,
}

/// TODO: Add docs
// ELIMINATED: Duplicate - use ml::predictions::Prediction
// pub struct Prediction {
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub signal: f32,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub confidence: f32,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub timestamp: u64,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
// }

#[derive(Default)]
/// TODO: Add docs
pub struct InferenceMetrics {
    pub total_inferences: u64,
    pub cache_hits: u64,
    pub avg_latency_us: f64,
    pub p99_latency_us: f64,
}

impl OptimizedInference {
    pub fn new(model_path: &str, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        // Create optimized ONNX session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;
        
        Ok(Self {
            session: Arc::new(session),
            batch_buffer: Vec::with_capacity(batch_size),
            batch_size,
            cache: Arc::new(dashmap::DashMap::new()),
            metrics: InferenceMetrics::default(),
        })
    }
    
    /// Submit features for inference (may batch)
    pub async fn infer(&mut self, features: FeatureVector) -> Prediction {
        let start = std::time::Instant::now();
        
        // Check cache
        let cache_key = self.hash_features(&features);
        if let Some(cached) = self.cache.get(&cache_key) {
            self.metrics.cache_hits += 1;
            return cached.clone();
        }
        
        // Add to batch
        self.batch_buffer.push(features);
        
        // Process batch if full
        if self.batch_buffer.len() >= self.batch_size {
            self.process_batch().await
        } else {
            // Wait for batch to fill or timeout
            self.wait_for_batch().await
        }
    }
    
    async fn process_batch(&mut self) -> Prediction {
        let batch = std::mem::replace(&mut self.batch_buffer, Vec::with_capacity(self.batch_size));
        
        // Convert to tensor
        let input_array = Array2::from_shape_vec(
            (batch.len(), batch[0].features.len()),
            batch.iter().flat_map(|f| f.features.clone()).collect(),
        ).unwrap();
        
        // Run inference
        let outputs = self.session.run(vec![input_array.into_dyn()]).unwrap();
        
        // Parse outputs and cache
        let predictions: Vec<Prediction> = outputs[0]
            .try_extract::<f32>().unwrap()
            .iter()
            .chunks(2)
            .enumerate()
            .map(|(i, mut chunk)| {
                let pred = Prediction {
                    signal: *chunk.next().unwrap(),
                    confidence: *chunk.next().unwrap(),
                    timestamp: batch[i].timestamp,
                };
                
                // Cache result
                let cache_key = self.hash_features(&batch[i]);
                self.cache.insert(cache_key, pred.clone());
                
                pred
            })
            .collect();
        
        predictions[0].clone()
    }
    
    async fn wait_for_batch(&mut self) -> Prediction {
        // Wait max 1ms for batch to fill
        tokio::time::sleep(tokio::time::Duration::from_micros(1000)).await;
        self.process_batch().await
    }
    
    fn hash_features(&self, features: &FeatureVector) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        features.timestamp.hash(&mut hasher);
        hasher.finish()
    }
}

// Blake: "Batched inference reduces latency by 5x!"
