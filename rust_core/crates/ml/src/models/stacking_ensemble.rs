// Stacking Ensemble with XGBoost, LightGBM, CatBoost
// Morgan (ML Lead) + Sam (Architecture) + Full Team
// References: Wolpert (1992), Breiman (1996), Recent Kaggle Winners (2024)
// CRITICAL: Nexus Requirement #3 - Ensemble diversity

use std::collections::HashMap;
use ndarray::{Array1, Array2, Axis, s, concatenate};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use parking_lot::RwLock;
use async_trait::async_trait;
use tracing::{info, error};

/// Base model trait for ensemble members
#[async_trait]
pub trait BaseModel: Send + Sync {
    async fn fit(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError>;
    async fn predict(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError>;
    async fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError>;
    fn feature_importance(&self) -> Option<Array1<f32>>;
    fn name(&self) -> &str;
}

/// Stacking Ensemble Coordinator
/// Morgan: "Combines strengths of multiple models while mitigating individual weaknesses!"
#[derive(Clone)]
/// TODO: Add docs
pub struct StackingEnsemble {
    // Base models (Level 0)
    base_models: Vec<Arc<RwLock<dyn BaseModel>>>,
    
    // Meta-learner (Level 1)
    meta_learner: Arc<RwLock<dyn BaseModel>>,
    
    // Configuration
    config: EnsembleConfig,
    
    // Cross-validation strategy
    cv_strategy: CrossValidationStrategy,
    
    // Blending weights (if using weighted average)
    blending_weights: Option<Array1<f32>>,
    
    // Feature importance aggregation
    aggregated_importance: Option<Array1<f32>>,
    
    // Performance tracking
    metrics: EnsembleMetrics,
    
    // Out-of-fold predictions for stacking
    oof_predictions: Option<Array2<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct EnsembleConfig {
    pub use_proba: bool,  // Use probabilities instead of predictions
    pub blend_mode: BlendMode,
    pub n_folds: usize,
    pub stratified: bool,
    pub use_features_in_meta: bool,  // Include original features in meta-learner
    pub optimize_weights: bool,  // Optimize blending weights
    pub diversity_penalty: f32,  // Penalize correlated predictions
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum BlendMode {
    Stacking,       // Train meta-learner on OOF predictions
    Blending,       // Simple weighted average
    Voting,         // Majority voting for classification
    BayesianAverage,  // Bayesian model averaging
    DynamicWeighted,  // Weights based on recent performance
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum CrossValidationStrategy {
    KFold { n_splits: usize, shuffle: bool },
    StratifiedKFold { n_splits: usize },
    TimeSeriesSplit { n_splits: usize, gap: usize },
    PurgedKFold { n_splits: usize, purge_gap: usize },
}

#[derive(Debug, Default)]
#[derive(Clone)]
struct EnsembleMetrics {
    train_score: f64,
    val_score: f64,
    test_score: f64,
    diversity_score: f64,
    model_correlations: HashMap<String, f64>,
    inference_time_ms: f64,
}

impl StackingEnsemble {
    pub fn new(
        base_models: Vec<Arc<RwLock<dyn BaseModel>>>,
        meta_learner: Arc<RwLock<dyn BaseModel>>,
        config: EnsembleConfig,
    ) -> Self {
        let cv_strategy = if config.stratified {
            CrossValidationStrategy::StratifiedKFold { n_splits: config.n_folds }
        } else {
            CrossValidationStrategy::KFold { 
                n_splits: config.n_folds, 
                shuffle: true 
            }
        };
        
        Self {
            base_models,
            meta_learner,
            config,
            cv_strategy,
            blending_weights: None,
            aggregated_importance: None,
            metrics: EnsembleMetrics::default(),
            oof_predictions: None,
        }
    }
    
    /// Fit the stacking ensemble
    /// Sam: "Clean separation between base model training and meta-learning!"
    pub async fn fit(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        let start = std::time::Instant::now();
        
        match self.config.blend_mode {
            BlendMode::Stacking => {
                self.fit_stacking(x, y).await?;
            },
            BlendMode::Blending => {
                self.fit_blending(x, y).await?;
            },
            BlendMode::Voting => {
                self.fit_voting(x, y).await?;
            },
            BlendMode::BayesianAverage => {
                self.fit_bayesian(x, y).await?;
            },
            BlendMode::DynamicWeighted => {
                self.fit_dynamic(x, y).await?;
            },
        }
        
        // Calculate diversity score
        self.metrics.diversity_score = self.calculate_diversity()?;
        
        // Aggregate feature importance
        self.aggregate_feature_importance();
        
        self.metrics.inference_time_ms = start.elapsed().as_millis() as f64;
        
        info!(
            diversity = self.metrics.diversity_score,
            time_ms = self.metrics.inference_time_ms,
            "Ensemble trained"
        );
        
        Ok(())
    }
    
    /// Fit using stacking (most sophisticated)
    async fn fit_stacking(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        let n_samples = x.nrows();
        let n_models = self.base_models.len();
        
        // Initialize OOF predictions matrix
        let mut oof_preds = Array2::zeros((n_samples, n_models));
        
        // Get CV splits
        let splits = self.get_cv_splits(n_samples, y);
        
        // For each fold
        for (fold_idx, (train_idx, val_idx)) in splits.iter().enumerate() {
            info!(fold = fold_idx + 1, total = splits.len(), "Processing fold");
            
            // Split data
            let x_train = x.select(Axis(0), train_idx);
            let y_train = y.select(Axis(0), train_idx);
            let x_val = x.select(Axis(0), val_idx);
            
            // Train each base model on this fold
            for (model_idx, model) in self.base_models.iter().enumerate() {
                let mut model = model.write();
                
                // Train on fold
                model.fit(&x_train, &y_train).await?;
                
                // Predict on validation
                let val_preds = if self.config.use_proba {
                    model.predict_proba(&x_val).await?
                        .column(1)  // Positive class probability
                        .to_owned()
                } else {
                    model.predict(&x_val).await?
                };
                
                // Store OOF predictions
                for (i, &idx) in val_idx.iter().enumerate() {
                    oof_preds[[idx, model_idx]] = val_preds[i];
                }
            }
        }
        
        self.oof_predictions = Some(oof_preds.clone());
        
        // Train meta-learner on OOF predictions
        let meta_features = if self.config.use_features_in_meta {
            // Concatenate original features with OOF predictions
            concatenate![Axis(1), x.view(), oof_preds.view()]
        } else {
            oof_preds
        };
        
        self.meta_learner.write().fit(&meta_features, y).await?;
        
        // Retrain base models on full data for final predictions
        for model in &self.base_models {
            model.write().fit(x, y).await?;
        }
        
        Ok(())
    }
    
    /// Fit using simple blending
    async fn fit_blending(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        // Train all base models
        for model in &self.base_models {
            model.write().fit(x, y).await?;
        }
        
        // Optimize blending weights if requested
        if self.config.optimize_weights {
            self.blending_weights = Some(self.optimize_blending_weights(x, y).await?);
        } else {
            // Equal weights
            let n_models = self.base_models.len();
            self.blending_weights = Some(Array1::from_elem(n_models, 1.0 / n_models as f32));
        }
        
        Ok(())
    }
    
    /// Fit using voting
    async fn fit_voting(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        // Simply train all base models
        for model in &self.base_models {
            model.write().fit(x, y).await?;
        }
        
        Ok(())
    }
    
    /// Fit using Bayesian Model Averaging
    async fn fit_bayesian(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        // Train models and calculate posterior probabilities
        let mut model_scores = Vec::new();
        
        for model in &self.base_models {
            model.write().fit(x, y).await?;
            
            // Calculate model evidence (simplified using validation score)
            let score = self.calculate_model_score(&*model.read(), x, y).await?;
            model_scores.push(score);
        }
        
        // Convert scores to weights using softmax
        let weights = self.softmax(&Array1::from(model_scores));
        self.blending_weights = Some(weights);
        
        Ok(())
    }
    
    /// Fit with dynamic weighting based on recent performance
    async fn fit_dynamic(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        // Use time-based validation for weight calculation
        let split_point = (x.nrows() as f32 * 0.8) as usize;
        let x_train = x.slice(s![..split_point, ..]);
        let y_train = y.slice(s![..split_point]);
        let x_val = x.slice(s![split_point.., ..]);
        let y_val = y.slice(s![split_point..]);
        
        let mut weights = Vec::new();
        
        for model in &self.base_models {
            // Train on earlier data
            model.write().fit(&x_train.to_owned(), &y_train.to_owned()).await?;
            
            // Evaluate on recent data
            let preds = model.read().predict(&x_val.to_owned()).await?;
            let score = self.calculate_rmse(&preds, &y_val.to_owned());
            
            // Convert error to weight (lower error = higher weight)
            weights.push((1.0 / (score + 1e-6)) as f32);
        }
        
        // Normalize weights
        let total: f32 = weights.iter().sum();
        let normalized: Vec<f32> = weights.iter().map(|w| w / total).collect();
        self.blending_weights = Some(Array1::from(normalized));
        
        // Retrain on full data
        for model in &self.base_models {
            model.write().fit(x, y).await?;
        }
        
        Ok(())
    }
    
    /// Make predictions using the ensemble
    pub async fn predict(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        match self.config.blend_mode {
            BlendMode::Stacking => self.predict_stacking(x).await,
            BlendMode::Blending => self.predict_blending(x).await,
            BlendMode::Voting => self.predict_voting(x).await,
            BlendMode::BayesianAverage => self.predict_bayesian(x).await,
            BlendMode::DynamicWeighted => self.predict_dynamic(x).await,
        }
    }
    
    /// Predict using stacking
    async fn predict_stacking(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        let n_samples = x.nrows();
        let n_models = self.base_models.len();
        
        // Get base model predictions
        let mut base_preds = Array2::zeros((n_samples, n_models));
        
        for (i, model) in self.base_models.iter().enumerate() {
            let preds = if self.config.use_proba {
                model.read().predict_proba(x).await?
                    .column(1)
                    .to_owned()
            } else {
                model.read().predict(x).await?
            };
            
            base_preds.column_mut(i).assign(&preds);
        }
        
        // Prepare meta features
        let meta_features = if self.config.use_features_in_meta {
            concatenate![Axis(1), x.view(), base_preds.view()]
        } else {
            base_preds
        };
        
        // Meta-learner prediction
        self.meta_learner.read().predict(&meta_features).await
    }
    
    /// Predict using blending
    async fn predict_blending(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        let weights = self.blending_weights.as_ref()
            .ok_or(ModelError::NotFitted)?;
        
        let mut weighted_sum = Array1::zeros(x.nrows());
        
        for (i, model) in self.base_models.iter().enumerate() {
            let preds = model.read().predict(x).await?;
            weighted_sum = weighted_sum + &preds * weights[i];
        }
        
        Ok(weighted_sum)
    }
    
    /// Predict using voting
    async fn predict_voting(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        let n_samples = x.nrows();
        let mut vote_matrix = Array2::zeros((n_samples, self.base_models.len()));
        
        for (i, model) in self.base_models.iter().enumerate() {
            let preds = model.read().predict(x).await?;
            vote_matrix.column_mut(i).assign(&preds);
        }
        
        // Majority voting
        let mut final_preds = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let votes = vote_matrix.row(i);
            let positive_votes = votes.iter().filter(|&&v| v > 0.5).count();
            final_preds[i] = if positive_votes > self.base_models.len() / 2 {
                1.0
            } else {
                0.0
            };
        }
        
        Ok(final_preds)
    }
    
    /// Predict using Bayesian averaging
    async fn predict_bayesian(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        self.predict_blending(x).await  // Similar to blending with optimized weights
    }
    
    /// Predict using dynamic weighting
    async fn predict_dynamic(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        self.predict_blending(x).await  // Use pre-calculated dynamic weights
    }
    
    /// Get cross-validation splits
    fn get_cv_splits(&self, n_samples: usize, y: &Array1<f32>) -> Vec<(Vec<usize>, Vec<usize>)> {
        match &self.cv_strategy {
            CrossValidationStrategy::KFold { n_splits, shuffle } => {
                self.kfold_splits(n_samples, *n_splits, *shuffle)
            },
            CrossValidationStrategy::TimeSeriesSplit { n_splits, gap } => {
                self.timeseries_splits(n_samples, *n_splits, *gap)
            },
            CrossValidationStrategy::PurgedKFold { n_splits, purge_gap } => {
                self.purged_kfold_splits(n_samples, *n_splits, *purge_gap)
            },
            _ => self.kfold_splits(n_samples, 5, true),  // Default
        }
    }
    
    /// K-Fold cross-validation splits
    fn kfold_splits(&self, n_samples: usize, n_splits: usize, shuffle: bool) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        let fold_size = n_samples / n_splits;
        let mut splits = Vec::new();
        
        for i in 0..n_splits {
            let val_start = i * fold_size;
            let val_end = if i == n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };
            
            let mut train_idx = Vec::new();
            let mut val_idx = Vec::new();
            
            for (j, &idx) in indices.iter().enumerate() {
                if j >= val_start && j < val_end {
                    val_idx.push(idx);
                } else {
                    train_idx.push(idx);
                }
            }
            
            splits.push((train_idx, val_idx));
        }
        
        splits
    }
    
    /// Time series splits with gap
    fn timeseries_splits(&self, n_samples: usize, n_splits: usize, gap: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let test_size = n_samples / (n_splits + 1);
        
        for i in 0..n_splits {
            let train_end = (i + 1) * test_size;
            let val_start = train_end + gap;
            let val_end = val_start + test_size;
            
            if val_end > n_samples {
                break;
            }
            
            let train_idx: Vec<usize> = (0..train_end).collect();
            let val_idx: Vec<usize> = (val_start..val_end).collect();
            
            splits.push((train_idx, val_idx));
        }
        
        splits
    }
    
    /// Purged K-Fold for time series
    fn purged_kfold_splits(&self, n_samples: usize, n_splits: usize, purge_gap: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / n_splits;
        let mut splits = Vec::new();
        
        for i in 0..n_splits {
            let val_start = i * fold_size;
            let val_end = (i + 1) * fold_size;
            
            let mut train_idx = Vec::new();
            let val_idx: Vec<usize> = (val_start..val_end.min(n_samples)).collect();
            
            // Add training indices with purge gap
            for j in 0..n_samples {
                if j < val_start.saturating_sub(purge_gap) || j >= val_end + purge_gap {
                    train_idx.push(j);
                }
            }
            
            if !train_idx.is_empty() && !val_idx.is_empty() {
                splits.push((train_idx, val_idx));
            }
        }
        
        splits
    }
    
    /// Calculate diversity score (lower correlation = higher diversity)
    fn calculate_diversity(&self) -> Result<f64, ModelError> {
        if let Some(oof_preds) = &self.oof_predictions {
            let n_models = oof_preds.ncols();
            let mut total_corr = 0.0;
            let mut count = 0;
            
            for i in 0..n_models {
                for j in i+1..n_models {
                    let corr = self.correlation(
                        &oof_preds.column(i).to_owned(),
                        &oof_preds.column(j).to_owned()
                    );
                    total_corr += corr.abs();
                    count += 1;
                }
            }
            
            // Diversity = 1 - average correlation
            Ok(1.0 - (total_corr / count as f64))
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate correlation between two arrays
    fn correlation(&self, a: &Array1<f32>, b: &Array1<f32>) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.mean().unwrap_or(0.0) as f64;
        let mean_b = b.mean().unwrap_or(0.0) as f64;
        
        let cov: f64 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64 - mean_a) * (*y as f64 - mean_b))
            .sum::<f64>() / n;
        
        let std_a = (a.iter()
            .map(|x| (*x as f64 - mean_a).powi(2))
            .sum::<f64>() / n)
            .sqrt();
        
        let std_b = (b.iter()
            .map(|y| (*y as f64 - mean_b).powi(2))
            .sum::<f64>() / n)
            .sqrt();
        
        if std_a > 0.0 && std_b > 0.0 {
            cov / (std_a * std_b)
        } else {
            0.0
        }
    }
    
    /// Optimize blending weights using coordinate descent
    async fn optimize_blending_weights(&self, x: &Array2<f32>, y: &Array1<f32>) -> Result<Array1<f32>, ModelError> {
        let n_models = self.base_models.len();
        let mut weights = Array1::from_elem(n_models, 1.0 / n_models as f32);
        
        // Get predictions from each model
        let mut predictions = Vec::new();
        for model in &self.base_models {
            let preds = model.read().predict(x).await?;
            predictions.push(preds);
        }
        
        // Coordinate descent optimization
        let mut best_score = f64::MAX;
        for _ in 0..100 {  // Max iterations
            for i in 0..n_models {
                let mut best_weight = weights[i];
                
                // Try different weight values
                for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].iter() {
                    weights[i] = *w;
                    
                    // Normalize weights
                    let sum: f32 = weights.sum();
                    if sum > 0.0 {
                        weights.mapv_inplace(|v| v / sum);
                    }
                    
                    // Calculate weighted prediction
                    let mut weighted_pred = Array1::zeros(y.len());
                    for (j, pred) in predictions.iter().enumerate() {
                        weighted_pred = weighted_pred + pred * weights[j];
                    }
                    
                    // Calculate error
                    let score = self.calculate_rmse(&weighted_pred, y);
                    
                    if score < best_score {
                        best_score = score;
                        best_weight = *w;
                    }
                }
                
                weights[i] = best_weight;
            }
        }
        
        // Final normalization
        let sum: f32 = weights.sum();
        if sum > 0.0 {
            weights.mapv_inplace(|v| v / sum);
        }
        
        Ok(weights)
    }
    
    /// Calculate RMSE
    fn calculate_rmse(&self, pred: &Array1<f32>, true_val: &Array1<f32>) -> f64 {
        let mse: f64 = pred.iter()
            .zip(true_val.iter())
            .map(|(p, t)| (*p as f64 - *t as f64).powi(2))
            .sum::<f64>() / pred.len() as f64;
        
        mse.sqrt()
    }
    
    /// Calculate model score for Bayesian averaging
    async fn calculate_model_score(&self, model: &dyn BaseModel, x: &Array2<f32>, y: &Array1<f32>) -> Result<f32, ModelError> {
        let preds = model.predict(x).await?;
        Ok(-(self.calculate_rmse(&preds, y) as f32))  // Negative RMSE as score
    }
    
    /// Softmax function
    fn softmax(&self, scores: &Array1<f32>) -> Array1<f32> {
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Array1<f32> = scores.mapv(|s| (s - max).exp());
        let sum = exp_scores.sum();
        exp_scores / sum
    }
    
    /// Aggregate feature importance from base models
    fn aggregate_feature_importance(&mut self) {
        let mut importance_sum: Option<Vec<f32>> = None;
        let mut count = 0;
        
        for model in &self.base_models {
            if let Some(imp) = model.read().feature_importance() {
                if let Some(ref mut sum) = importance_sum {
                    // Convert Array1<f32> to Vec<f32> for addition
                    let imp_vec = imp.to_vec();
                    for (i, val) in imp_vec.iter().enumerate() {
                        if i < sum.len() {
                            sum[i] += val;
                        }
                    }
                } else {
                    importance_sum = Some(imp.to_vec());
                }
                count += 1;
            }
        }
        
        if let Some(mut sum) = importance_sum {
            // Divide each element by count
            for val in sum.iter_mut() {
                *val /= count as f32;
            }
            self.aggregated_importance = Some(Array1::from(sum));
        }
    }
    
    /// Get ensemble metrics
    pub fn get_metrics(&self) -> &EnsembleMetrics {
        &self.metrics
    }
    
    /// Get aggregated feature importance
    pub fn get_feature_importance(&self) -> Option<&Array1<f32>> {
        self.aggregated_importance.as_ref()
    }
}

/// Model errors
#[derive(Debug, thiserror::Error)]
/// TODO: Add docs
pub enum ModelError {
    #[error("Model not fitted")]
    NotFitted,
    
    #[error("Invalid input shape")]
    InvalidShape,
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock model for testing
    struct MockModel {
        name: String,
        fitted: bool,
    }
    
    #[async_trait]
    impl BaseModel for MockModel {
        async fn fit(&mut self, _x: &Array2<f32>, _y: &Array1<f32>) -> Result<(), ModelError> {
            self.fitted = true;
            Ok(())
        }
        
        async fn predict(&self, x: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
            if !self.fitted {
                return Err(ModelError::NotFitted);
            }
            // Return random predictions
            Ok(Array1::from_shape_fn(x.nrows(), |_| rand::random::<f32>()))
        }
        
        async fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
            if !self.fitted {
                return Err(ModelError::NotFitted);
            }
            let n = x.nrows();
            let mut proba = Array2::zeros((n, 2));
            for i in 0..n {
                let p = rand::random::<f32>();
                proba[[i, 0]] = 1.0 - p;
                proba[[i, 1]] = p;
            }
            Ok(proba)
        }
        
        fn feature_importance(&self) -> Option<Array1<f32>> {
            Some(Array1::from_shape_fn(10, |_| rand::random::<f32>()))
        }
        
        fn name(&self) -> &str {
            &self.name
        }
    }
    
    #[tokio::test]
    async fn test_stacking_ensemble() {
        // Create mock models
        let base_models = vec![
            Arc::new(RwLock::new(MockModel { name: "Model1".to_string(), fitted: false })) as Arc<RwLock<dyn BaseModel>>,
            Arc::new(RwLock::new(MockModel { name: "Model2".to_string(), fitted: false })) as Arc<RwLock<dyn BaseModel>>,
            Arc::new(RwLock::new(MockModel { name: "Model3".to_string(), fitted: false })) as Arc<RwLock<dyn BaseModel>>,
        ];
        
        let meta_learner = Arc::new(RwLock::new(MockModel { 
            name: "MetaLearner".to_string(), 
            fitted: false 
        })) as Arc<RwLock<dyn BaseModel>>;
        
        let config = EnsembleConfig {
            use_proba: false,
            blend_mode: BlendMode::Stacking,
            n_folds: 3,
            stratified: false,
            use_features_in_meta: false,
            optimize_weights: false,
            diversity_penalty: 0.1,
        };
        
        let mut ensemble = StackingEnsemble::new(base_models, meta_learner, config);
        
        // Create test data
        let x = Array2::from_shape_fn((100, 10), |_| rand::random::<f32>());
        let y = Array1::from_shape_fn(100, |_| if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 });
        
        // Fit ensemble
        ensemble.fit(&x, &y).await.unwrap();
        
        // Make predictions
        let predictions = ensemble.predict(&x).await.unwrap();
        
        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }
    
    #[tokio::test]
    async fn test_blending_ensemble() {
        let base_models = vec![
            Arc::new(RwLock::new(MockModel { name: "Model1".to_string(), fitted: false })) as Arc<RwLock<dyn BaseModel>>,
            Arc::new(RwLock::new(MockModel { name: "Model2".to_string(), fitted: false })) as Arc<RwLock<dyn BaseModel>>,
        ];
        
        let meta_learner = Arc::new(RwLock::new(MockModel { 
            name: "MetaLearner".to_string(), 
            fitted: false 
        })) as Arc<RwLock<dyn BaseModel>>;
        
        let config = EnsembleConfig {
            use_proba: false,
            blend_mode: BlendMode::Blending,
            n_folds: 3,
            stratified: false,
            use_features_in_meta: false,
            optimize_weights: true,
            diversity_penalty: 0.1,
        };
        
        let mut ensemble = StackingEnsemble::new(base_models, meta_learner, config);
        
        let x = Array2::from_shape_fn((50, 5), |_| rand::random::<f32>());
        let y = Array1::from_shape_fn(50, |_| rand::random::<f32>());
        
        ensemble.fit(&x, &y).await.unwrap();
        
        // Check weights were optimized
        assert!(ensemble.blending_weights.is_some());
        let weights = ensemble.blending_weights.as_ref().unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights.sum() - 1.0).abs() < 1e-6);  // Weights sum to 1
    }
}