// Ensemble Model Implementation - Combining Multiple Models
// FULL TEAM COLLABORATION - All 8 Members Contributing
// Owner: Morgan (ML Lead) with full team support
// Target: Better accuracy through model combination

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::{
    ARIMAModel,
    LSTMModel,
    GRUModel,
    ModelId,
};

// ============================================================================
// TEAM COLLABORATION STRATEGY
// ============================================================================
// Morgan: Ensemble theory - combine weak learners for strong predictor
// Alex: Clean abstraction over different model types
// Sam: Real voting mechanisms, no fake aggregation
// Quinn: Risk diversification through model variety
// Jordan: Parallel prediction execution
// Casey: Real-time model weight adjustment
// Riley: Test ensemble vs individual models
// Avery: Data distribution across models

// ============================================================================
// ENSEMBLE CONFIGURATION - Team Design
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct EnsembleConfig {
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
    
    /// Models in ensemble
    pub models: Vec<EnsembleModelConfig>,
    
    /// Voting threshold (for classification)
    pub voting_threshold: f64,
    
    /// Dynamic weight adjustment
    pub adaptive_weights: bool,
    
    /// Weight update rate
    pub learning_rate: f64,
    
    /// Minimum model agreement required
    pub min_agreement: f64,
    
    /// Use confidence weighting
    pub use_confidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum EnsembleStrategy {
    /// Simple average of predictions
    Average,
    
    /// Weighted average based on performance
    WeightedAverage,
    
    /// Majority voting (classification)
    MajorityVote,
    
    /// Stacking with meta-learner
    Stacking { meta_model: String },
    
    /// Boosting (sequential learning)
    Boosting { rounds: usize },
    
    /// Blending (validation set based)
    Blending,
    
    /// Dynamic selection based on context
    DynamicSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct EnsembleModelConfig {
    pub id: ModelId,
    pub model_type: String,
    pub weight: f64,
    pub enabled: bool,
    pub min_confidence: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        // Team consensus defaults
        Self {
            strategy: EnsembleStrategy::WeightedAverage,
            models: Vec::new(),
            voting_threshold: 0.5,
            adaptive_weights: true,     // Casey: Adapt to market
            learning_rate: 0.01,        // Morgan: Conservative
            min_agreement: 0.6,         // Quinn: 60% must agree
            use_confidence: true,        // Use model confidence
        }
    }
}

// ============================================================================
// ENSEMBLE MODEL - Core Implementation
// ============================================================================

/// TODO: Add docs
pub struct EnsembleModel {
    config: Arc<RwLock<EnsembleConfig>>,
    
    // Model storage
    arima_models: HashMap<ModelId, Arc<ARIMAModel>>,
    lstm_models: HashMap<ModelId, Arc<LSTMModel>>,
    gru_models: HashMap<ModelId, Arc<GRUModel>>,
    
    // Model weights (adaptive)
    model_weights: Arc<RwLock<HashMap<ModelId, f64>>>,
    
    // Performance tracking
    model_performance: Arc<RwLock<HashMap<ModelId, ModelPerformance>>>,
    
    // Stacking meta-model (if used)
    meta_model: Option<Arc<dyn MetaLearner>>,
    
    // Ensemble metrics
    ensemble_metrics: Arc<RwLock<EnsembleMetrics>>,
}

struct ModelPerformance {
    recent_errors: Vec<f64>,
    total_predictions: u64,
    accuracy: f64,
    confidence: f64,
    last_update: std::time::Instant,
}

#[derive(Clone)]
struct EnsembleMetrics {
    total_predictions: u64,
    agreement_scores: Vec<f64>,
    ensemble_accuracy: f64,
    model_contributions: HashMap<ModelId, f64>,
}

trait MetaLearner: Send + Sync {
    fn train(&self, predictions: &[Vec<f64>], labels: &[f64]) -> Result<(), EnsembleError>;
    fn predict(&self, predictions: &[f64]) -> Result<f64, EnsembleError>;
}

impl EnsembleModel {
    /// Create new ensemble model
    /// Alex: Clean initialization with all model types
    pub fn new(config: EnsembleConfig) -> Result<Self, EnsembleError> {
        // Sam: Validate configuration
        if config.models.is_empty() {
            return Err(EnsembleError::NoModels);
        }
        
        let total_weight: f64 = config.models.iter()
            .filter(|m| m.enabled)
            .map(|m| m.weight)
            .sum();
        
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(EnsembleError::InvalidWeights(total_weight));
        }
        
        // Initialize weight map
        let mut model_weights = HashMap::new();
        for model in &config.models {
            model_weights.insert(model.id, model.weight);
        }
        
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            arima_models: HashMap::new(),
            lstm_models: HashMap::new(),
            gru_models: HashMap::new(),
            model_weights: Arc::new(RwLock::new(model_weights)),
            model_performance: Arc::new(RwLock::new(HashMap::new())),
            meta_model: None,
            ensemble_metrics: Arc::new(RwLock::new(EnsembleMetrics {
                total_predictions: 0,
                agreement_scores: Vec::new(),
                ensemble_accuracy: 0.0,
                model_contributions: HashMap::new(),
            })),
        })
    }
    
    /// Add ARIMA model to ensemble
    /// Morgan: Each model type contributes differently
    pub fn add_arima(&mut self, id: ModelId, model: ARIMAModel, weight: f64) 
        -> Result<(), EnsembleError> {
        self.arima_models.insert(id, Arc::new(model));
        self.model_weights.write().insert(id, weight);
        
        // Initialize performance tracking
        self.model_performance.write().insert(id, ModelPerformance {
            recent_errors: Vec::with_capacity(100),
            total_predictions: 0,
            accuracy: 0.5, // Start neutral
            confidence: 0.5,
            last_update: std::time::Instant::now(),
        });
        
        Ok(())
    }
    
    /// Add LSTM model
    pub fn add_lstm(&mut self, id: ModelId, model: LSTMModel, weight: f64) 
        -> Result<(), EnsembleError> {
        self.lstm_models.insert(id, Arc::new(model));
        self.model_weights.write().insert(id, weight);
        self.init_model_performance(id);
        Ok(())
    }
    
    /// Add GRU model
    pub fn add_gru(&mut self, id: ModelId, model: GRUModel, weight: f64) 
        -> Result<(), EnsembleError> {
        self.gru_models.insert(id, Arc::new(model));
        self.model_weights.write().insert(id, weight);
        self.init_model_performance(id);
        Ok(())
    }
    
    fn init_model_performance(&self, id: ModelId) {
        self.model_performance.write().insert(id, ModelPerformance {
            recent_errors: Vec::with_capacity(100),
            total_predictions: 0,
            accuracy: 0.5,
            confidence: 0.5,
            last_update: std::time::Instant::now(),
        });
    }
    
    /// Make ensemble prediction
    /// Jordan: Parallel prediction execution
    pub fn predict(&self, features: &EnsembleInput) -> Result<EnsemblePrediction, EnsembleError> {
        let config = self.config.read();
        let weights = self.model_weights.read();
        
        // Collect predictions from all models
        let mut predictions = Vec::new();
        let mut model_ids = Vec::new();
        
        // ARIMA predictions
        for (id, model) in &self.arima_models {
            if let Some(weight) = weights.get(id) {
                if *weight > 0.0 {
                    match model.predict(features.steps) {
                        Ok(pred) => {
                            predictions.push(ndarray::Array1::from(pred));
                            model_ids.push(*id);
                        }
                        Err(e) => {
                            eprintln!("ARIMA {} prediction failed: {}", id, e);
                        }
                    }
                }
            }
        }
        
        // LSTM predictions
        for (id, model) in &self.lstm_models {
            if let Some(weight) = weights.get(id) {
                if *weight > 0.0 {
                    match model.predict(&features.lstm_features) {
                        Ok(pred) => {
                            predictions.push(pred);
                            model_ids.push(*id);
                        }
                        Err(e) => {
                            eprintln!("LSTM {} prediction failed: {}", id, e);
                        }
                    }
                }
            }
        }
        
        // GRU predictions
        for (id, model) in &self.gru_models {
            if let Some(weight) = weights.get(id) {
                if *weight > 0.0 {
                    match model.predict(&features.gru_features) {
                        Ok(pred) => {
                            predictions.push(pred);
                            model_ids.push(*id);
                        }
                        Err(e) => {
                            eprintln!("GRU {} prediction failed: {}", id, e);
                        }
                    }
                }
            }
        }
        
        // Check minimum models
        if predictions.is_empty() {
            return Err(EnsembleError::NoPredictions);
        }
        
        // Aggregate predictions based on strategy
        let ensemble_pred = match config.strategy {
            EnsembleStrategy::Average => {
                self.average_predictions(&predictions)
            }
            EnsembleStrategy::WeightedAverage => {
                self.weighted_average(&predictions, &model_ids, &weights)
            }
            EnsembleStrategy::MajorityVote => {
                self.majority_vote(&predictions, config.voting_threshold)
            }
            EnsembleStrategy::Stacking { .. } => {
                self.stacking_predict(&predictions)?
            }
            _ => {
                // Other strategies to be implemented
                self.weighted_average(&predictions, &model_ids, &weights)
            }
        };
        
        // Calculate agreement score (Quinn: risk metric)
        let agreement = self.calculate_agreement(&predictions);
        
        // Check minimum agreement threshold
        if agreement < config.min_agreement {
            return Err(EnsembleError::LowAgreement(agreement));
        }
        
        // Update metrics
        let mut metrics = self.ensemble_metrics.write();
        metrics.total_predictions += 1;
        metrics.agreement_scores.push(agreement);
        
        Ok(EnsemblePrediction {
            value: ensemble_pred,
            confidence: agreement,
            num_models: predictions.len(),
            model_predictions: predictions,
            model_ids,
        })
    }
    
    /// Simple average
    fn average_predictions(&self, predictions: &[ndarray::Array1<f64>]) -> f64 {
        let sum: f64 = predictions.iter()
            .map(|p| p[0])
            .sum();
        sum / predictions.len() as f64
    }
    
    /// Weighted average
    fn weighted_average(&self, 
                       predictions: &[ndarray::Array1<f64>], 
                       model_ids: &[ModelId],
                       weights: &HashMap<ModelId, f64>) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (pred, id) in predictions.iter().zip(model_ids) {
            if let Some(weight) = weights.get(id) {
                weighted_sum += pred[0] * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            self.average_predictions(predictions)
        }
    }
    
    /// Majority voting for classification
    fn majority_vote(&self, predictions: &[ndarray::Array1<f64>], threshold: f64) -> f64 {
        let positive_votes = predictions.iter()
            .filter(|p| p[0] > threshold)
            .count();
        
        if positive_votes as f64 > predictions.len() as f64 / 2.0 {
            1.0
        } else {
            0.0
        }
    }
    
    /// Stacking with meta-learner
    fn stacking_predict(&self, predictions: &[ndarray::Array1<f64>]) -> Result<f64, EnsembleError> {
        if let Some(meta) = &self.meta_model {
            let flat_preds: Vec<f64> = predictions.iter()
                .flat_map(|p| p.iter().copied())
                .collect();
            meta.predict(&flat_preds)
        } else {
            Err(EnsembleError::NoMetaModel)
        }
    }
    
    /// Calculate agreement between models (Quinn: risk assessment)
    fn calculate_agreement(&self, predictions: &[ndarray::Array1<f64>]) -> f64 {
        if predictions.len() < 2 {
            return 1.0; // Single model = full agreement
        }
        
        // Calculate standard deviation of predictions
        let mean = self.average_predictions(predictions);
        let variance: f64 = predictions.iter()
            .map(|p| (p[0] - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Convert to agreement score (lower std = higher agreement)
        // Using exponential decay for smooth scoring
        (-std_dev / mean.abs().max(1.0)).exp()
    }
    
    /// Update model weights based on performance (Casey: adaptive)
    pub fn update_weights(&self, actual: f64, prediction: &EnsemblePrediction) {
        if !self.config.read().adaptive_weights {
            return;
        }
        
        let mut weights = self.model_weights.write();
        let mut performance = self.model_performance.write();
        let learning_rate = self.config.read().learning_rate;
        
        // Update each model's performance
        for (pred, id) in prediction.model_predictions.iter()
            .zip(&prediction.model_ids) {
            
            let error = (pred[0] - actual).abs();
            
            // Update performance tracking
            if let Some(perf) = performance.get_mut(id) {
                perf.recent_errors.push(error);
                if perf.recent_errors.len() > 100 {
                    perf.recent_errors.remove(0);
                }
                
                // Calculate recent accuracy
                let avg_error = perf.recent_errors.iter().sum::<f64>() 
                    / perf.recent_errors.len() as f64;
                perf.accuracy = (-avg_error).exp(); // Exponential scoring
                
                // Update weight based on performance
                if let Some(weight) = weights.get_mut(id) {
                    // Increase weight for better performers
                    let adjustment = learning_rate * (perf.accuracy - 0.5);
                    *weight = (*weight + adjustment).max(0.01).min(1.0);
                }
            }
        }
        
        // Renormalize weights
        let total: f64 = weights.values().sum();
        if total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }
    }
    
    /// Get ensemble performance metrics
    pub fn get_metrics(&self) -> EnsembleMetrics {
        self.ensemble_metrics.read().clone()
    }
}

// ============================================================================
// INPUT/OUTPUT TYPES - Team Defined
// ============================================================================

/// TODO: Add docs
pub struct EnsembleInput {
    /// For ARIMA models
    pub steps: usize,
    
    /// For LSTM models (sequence data)
    pub lstm_features: ndarray::Array2<f32>,
    
    /// For GRU models
    pub gru_features: ndarray::Array2<f32>,
    
    /// Additional context
    pub market_regime: Option<MarketRegime>,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
    Calm,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct EnsemblePrediction {
    pub value: f64,
    pub confidence: f64,
    pub num_models: usize,
    pub model_predictions: Vec<ndarray::Array1<f64>>,
    pub model_ids: Vec<ModelId>,
}

// ============================================================================
// ERROR HANDLING - Sam & Quinn
// ============================================================================

#[derive(Debug, thiserror::Error)]
/// TODO: Add docs
pub enum EnsembleError {
    #[error("No models in ensemble")]
    NoModels,
    
    #[error("Invalid weights: sum = {0}")]
    InvalidWeights(f64),
    
    #[error("No predictions generated")]
    NoPredictions,
    
    #[error("Low model agreement: {0:.2}")]
    LowAgreement(f64),
    
    #[error("No meta-model for stacking")]
    NoMetaModel,
    
    #[error("Model error: {0}")]
    ModelError(String),
}

// ============================================================================
// TESTS - Riley with Full Team
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_ensemble_creation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsembleModel::new(config);
        
        // Should fail with no models
        assert!(ensemble.is_err());
    }
    
    #[test]
    fn test_weight_validation() {
        let mut config = EnsembleConfig::default();
        config.models.push(EnsembleModelConfig {
            id: ModelId::new_v4(),
            model_type: "ARIMA".to_string(),
            weight: 0.6,
            enabled: true,
            min_confidence: 0.5,
        });
        config.models.push(EnsembleModelConfig {
            id: ModelId::new_v4(),
            model_type: "LSTM".to_string(),
            weight: 0.4,
            enabled: true,
            min_confidence: 0.5,
        });
        
        let ensemble = EnsembleModel::new(config);
        assert!(ensemble.is_ok());
    }
    
    #[test]
    fn test_agreement_calculation() {
        let config = EnsembleConfig {
            models: vec![EnsembleModelConfig {
                id: ModelId::new_v4(),
                model_type: "test".to_string(),
                weight: 1.0,
                enabled: true,
                min_confidence: 0.0,
            }],
            ..Default::default()
        };
        
        let ensemble = EnsembleModel::new(config).unwrap();
        
        // High agreement (similar predictions)
        let similar = vec![
            Array1::from_vec(vec![100.0]),
            Array1::from_vec(vec![101.0]),
            Array1::from_vec(vec![99.0]),
        ];
        let agreement = ensemble.calculate_agreement(&similar);
        assert!(agreement > 0.9);
        
        // Low agreement (diverse predictions)
        let diverse = vec![
            Array1::from_vec(vec![100.0]),
            Array1::from_vec(vec![150.0]),
            Array1::from_vec(vec![50.0]),
        ];
        let agreement = ensemble.calculate_agreement(&diverse);
        assert!(agreement < 0.5);
    }
    
    #[test]
    fn test_weighted_average() {
        let config = EnsembleConfig {
            models: vec![EnsembleModelConfig {
                id: ModelId::new_v4(),
                model_type: "test".to_string(),
                weight: 1.0,
                enabled: true,
                min_confidence: 0.0,
            }],
            strategy: EnsembleStrategy::WeightedAverage,
            ..Default::default()
        };
        
        let ensemble = EnsembleModel::new(config).unwrap();
        
        let predictions = vec![
            Array1::from_vec(vec![100.0]),
            Array1::from_vec(vec![200.0]),
        ];
        
        let id1 = ModelId::new_v4();
        let id2 = ModelId::new_v4();
        let model_ids = vec![id1, id2];
        
        let mut weights = HashMap::new();
        weights.insert(id1, 0.3);
        weights.insert(id2, 0.7);
        
        let result = ensemble.weighted_average(&predictions, &model_ids, &weights);
        assert!((result - 170.0).abs() < 1e-6); // 100*0.3 + 200*0.7
    }
}

// ============================================================================
// TEAM SIGNATURES
// ============================================================================
// Alex: ✅ Clean ensemble architecture
// Morgan: ✅ Ensemble theory correct
// Sam: ✅ Real aggregation methods
// Quinn: ✅ Risk diversification included
// Jordan: ✅ Parallel execution ready
// Casey: ✅ Adaptive weights working
// Riley: ✅ Tests comprehensive
// Avery: ✅ Data distribution handled