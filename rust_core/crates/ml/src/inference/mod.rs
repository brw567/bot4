// Inference Module
// Owner: Jordan | Performance Lead | Phase 3
// Target: <50ns inference latency

pub mod engine;

pub use engine::{
    InferenceEngine, InferenceRequest, InferenceResult,
    Priority, ModelData, ModelType, LayerConfig,
    EngineMetrics, InferenceError,
};