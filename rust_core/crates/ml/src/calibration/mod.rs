// Calibration Module
// Phase 3+ Enhancement
// Morgan (ML Lead) + Quinn (Risk) + Full Team

pub mod isotonic;

pub use isotonic::{
    IsotonicCalibrator,
    ReliabilityDiagram,
};

// Type alias for backward compatibility
pub type CalibrationMetrics = ReliabilityDiagram;