use domain_types::FeatureVector;
//! # ML Layer Abstractions (Layer 3)
//!
//! ML abstractions that feature_store can use without depending on ML crate.
//! This inverts the dependency so ML depends on abstractions, not vice versa.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::AbstractionResult;

/// Feature provider trait - ML consumes features from this
#[async_trait]
pub trait FeatureProvider: Send + Sync {
    /// Get feature vector for entity
    async fn get_features(
        &self,
        entity_id: &str,
        feature_names: Vec<String>,
    ) -> AbstractionResult<FeatureVector>;
    
    /// Get batch of features
    async fn get_batch_features(
        &self,
        entity_ids: Vec<String>,
        feature_names: Vec<String>,
    ) -> AbstractionResult<Vec<FeatureVector>>;
    
    /// Get time-series features
    async fn get_timeseries_features(
        &self,
        entity_id: &str,
        feature_names: Vec<String>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> AbstractionResult<TimeSeriesFeatures>;
    
    /// Register feature computation
    async fn register_feature(
        &self,
        name: String,
        computation: FeatureComputation,
    ) -> AbstractionResult<()>;
}

/// Feature vector
#[derive(Debug, Clone, Serialize, Deserialize)]
// ELIMINATED: use domain_types::FeatureVector
// pub struct FeatureVector {
    /// Entity ID
    pub entity_id: String,
    /// Features as key-value pairs
    pub features: HashMap<String, FeatureValue>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Feature value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
/// TODO: Add docs
pub enum FeatureValue {
    /// Numeric feature
    Numeric(f64),
    /// Categorical feature
    Categorical(String),
    /// Vector feature
    Vector(Vec<f64>),
    /// Binary feature
    Binary(bool),
    /// Missing value
    Missing,
}

/// Time series features
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TimeSeriesFeatures {
    /// Entity ID
    pub entity_id: String,
    /// Time series data
    pub series: Vec<TimePoint>,
    /// Feature names
    pub feature_names: Vec<String>,
}

/// Time point in series
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TimePoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Feature values
    pub values: Vec<FeatureValue>,
}

/// Feature computation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureComputation {
    /// Feature name
    pub name: String,
    /// Computation type
    pub computation_type: ComputationType,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Window size (for windowed features)
    pub window_size: Option<u32>,
    /// Aggregation function
    pub aggregation: Option<AggregationType>,
}

/// Computation types
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum ComputationType {
    /// Direct computation
    Direct,
    /// Rolling window
    RollingWindow,
    /// Exponential weighted
    ExponentialWeighted,
    /// Lagged feature
    Lagged,
    /// Derived from other features
    Derived,
}

/// Aggregation types
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AggregationType {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Median
    Median,
    /// Min
    Min,
    /// Max
    Max,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Count
    Count,
}

/// Model prediction abstraction
#[async_trait]
pub trait ModelPredictor: Send + Sync {
    /// Make prediction
    async fn predict(
        &self,
        features: FeatureVector,
    ) -> AbstractionResult<Prediction>;
    
    /// Batch prediction
    async fn predict_batch(
        &self,
        features: Vec<FeatureVector>,
    ) -> AbstractionResult<Vec<Prediction>>;
    
    /// Get model metadata
    async fn get_metadata(&self) -> AbstractionResult<ModelMetadata>;
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
// ELIMINATED: Duplicate - use ml::predictions::Prediction
// pub struct Prediction {
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     /// Model ID
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub model_id: String,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     /// Predicted value
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub value: PredictionValue,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     /// Confidence/probability
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub confidence: f64,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     /// Prediction timestamp
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub timestamp: DateTime<Utc>,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     /// Feature importance
// ELIMINATED: Duplicate - use ml::predictions::Prediction
//     pub feature_importance: Option<HashMap<String, f64>>,
// ELIMINATED: Duplicate - use ml::predictions::Prediction
// }

/// Prediction value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
/// TODO: Add docs
pub enum PredictionValue {
    /// Regression output
    Regression(f64),
    /// Classification output
    Classification(String),
    /// Multi-class probabilities
    Probabilities(HashMap<String, f64>),
    /// Multi-output
    MultiOutput(Vec<f64>),
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
// pub struct ModelMetadata {
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Model ID
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub model_id: String,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Model type
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub model_type: String,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Version
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub version: String,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Training date
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub trained_at: DateTime<Utc>,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Performance metrics
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub metrics: HashMap<String, f64>,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     /// Feature names
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
//     pub feature_names: Vec<String>,
// ELIMINATED: Duplicate - use ml::model_metadata::ModelMetadata
// }

/// Model training abstraction
#[async_trait]
pub trait ModelTrainer: Send + Sync {
    /// Train model
    async fn train(
        &self,
        features: Vec<FeatureVector>,
        labels: Vec<f64>,
        config: TrainingConfig,
    ) -> AbstractionResult<String>; // Returns model_id
    
    /// Evaluate model
    async fn evaluate(
        &self,
        model_id: &str,
        features: Vec<FeatureVector>,
        labels: Vec<f64>,
    ) -> AbstractionResult<EvaluationMetrics>;
    
    /// Retrain model
    async fn retrain(
        &self,
        model_id: &str,
        features: Vec<FeatureVector>,
        labels: Vec<f64>,
    ) -> AbstractionResult<String>; // Returns new model_id
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TrainingConfig {
    /// Model type
    pub model_type: String,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, Value>,
    /// Validation split
    pub validation_split: f64,
    /// Cross validation folds
    pub cv_folds: Option<u32>,
    /// Early stopping
    pub early_stopping: bool,
    /// Max epochs/iterations
    pub max_iterations: u32,
}

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct EvaluationMetrics {
    /// Mean squared error
    pub mse: Option<f64>,
    /// Mean absolute error
    pub mae: Option<f64>,
    /// R-squared
    pub r2: Option<f64>,
    /// Accuracy
    pub accuracy: Option<f64>,
    /// Precision
    pub precision: Option<f64>,
    /// Recall
    pub recall: Option<f64>,
    /// F1 score
    pub f1: Option<f64>,
    /// AUC-ROC
    pub auc_roc: Option<f64>,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
}