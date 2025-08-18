// ML Models Module
// Owner: Morgan | Phase 3 Week 2
// FULL TEAM COLLABORATION: All models reviewed by all 8 members

pub mod arima;
pub mod lstm;
pub mod gru;
pub mod ensemble;
pub mod registry;

// ARIMA exports
pub use arima::{ARIMAModel, ARIMAConfig, ARIMAError, FitResult};

// LSTM exports
pub use lstm::{LSTMModel, LSTMConfig, LSTMError};

// GRU exports  
pub use gru::{GRUModel, GRUConfig, GRUError};

// Ensemble exports
pub use ensemble::{
    EnsembleModel, EnsembleConfig, EnsembleStrategy, 
    EnsembleInput, EnsemblePrediction, EnsembleError
};

// Registry exports
pub use registry::{
    ModelRegistry, ModelMetadata, ModelVersion, ModelType, 
    ModelStatus, ModelMetrics, DeploymentStrategy, DeploymentResult,
    ABTestConfig, PerformanceSnapshot, ComparisonResult, RegistryError
};

// Common types
pub type ModelId = uuid::Uuid;

// Training result (shared between LSTM and GRU)
pub use lstm::TrainingResult as LSTMTrainingResult;
pub use gru::TrainingResult as GRUTrainingResult;