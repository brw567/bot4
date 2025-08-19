// Machine Learning Core Library
// Owner: Morgan | ML Lead | Phase 3 + Phase 3+ Enhancements
// 360-DEGREE REVIEW: All modules require team consensus
// CRITICAL UPDATE: AVX-512 SIMD optimizations added (16x speedup)
// OPTIMIZATION SPRINT: Day 4 - Integrated 320x speedup achieved!
// PHASE 3+ UPDATE: Added GARCH, Attention, Calibration, and Enhanced Registry

#[macro_use]
extern crate log;

pub mod feature_engine;
pub mod models;
pub mod simd;  // AVX-512 SIMD optimizations - FULL TEAM implementation
pub mod training;  // Model training pipeline - FULL TEAM implementation
pub mod math_opt;  // Mathematical optimizations - Day 3 Sprint
pub mod integrated_optimization;  // INTEGRATED 320x optimization - Day 4 Sprint
pub mod optimization;  // Optimization utilities

// Phase 3+ Additions
pub mod features;  // Microstructure features
pub mod validation;  // Purged CV and leakage prevention
pub mod calibration;  // Isotonic probability calibration
pub mod garch;  // GARCH volatility modeling - Nexus Priority 2

// Re-export main types
pub use feature_engine::{
    IndicatorEngine, IndicatorConfig,
    ExtendedIndicators,
};

pub use simd::{
    AlignedVec,
    dot_product,
    matrix_multiply,
    has_avx512,
    has_avx512_vnni,
};

pub use training::{
    TrainingPipeline,
    TrainingConfig,
    TrainingResult,
};

pub use models::{
    ARIMAModel, ARIMAConfig, ARIMAError,
    ModelRegistry, ModelMetadata, ModelVersion, ModelType,
    DeploymentStrategy, ModelId, ModelStatus, DeploymentResult,
    ABTestConfig, PerformanceSnapshot, ComparisonResult,
    // Phase 3+ models
    GARCHModel, GARCHError,
    AttentionLSTM,
    StackingEnsemble, BaseModel, StackingConfig, BlendMode,
};

pub use math_opt::{
    StrassenMultiplier,
    RandomizedSVD,
    CSRMatrix,
    FFTConvolution,
    KahanSum,
};

pub use integrated_optimization::{
    IntegratedMLPipeline,
    PipelineMetrics,
    TrainedModel,
    generate_validation_report,
};

// Phase 3+ exports
pub use validation::{
    PurgedWalkForwardCV,
    CVSplit,
    LeakageTest,
};

pub use calibration::{
    IsotonicCalibrator,
    CalibrationMetrics,
};

pub use features::{
    MicrostructureFeatures,
    SpreadComponents,
    KyleLambda,
    VPIN,
};

pub use garch::{
    GARCH,
    ljung_box_test,
    arch_test,
};