// Machine Learning Core Library
// Owner: Morgan | ML Lead | Phase 3
// 360-DEGREE REVIEW: All modules require team consensus

pub mod feature_engine;
pub mod models;

// Re-export main types
pub use feature_engine::{
    indicators::{IndicatorEngine, IndicatorConfig},
    indicators_extended::ExtendedIndicators,
};

pub use models::{
    ARIMAModel, ARIMAConfig, ARIMAError,
    ModelRegistry, ModelMetadata, ModelVersion, ModelType,
    DeploymentStrategy, ModelId,
};