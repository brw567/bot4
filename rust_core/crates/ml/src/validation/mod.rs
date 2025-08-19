// Validation Module
// Phase 3+ Enhancement
// Morgan (ML Lead) + Riley (Testing) + Full Team

pub mod purged_cv;

pub use purged_cv::{
    PurgedWalkForwardCV,
    LeakageSentinel,
    LeakageTestResult,
};

// Type aliases for backward compatibility
pub type CVSplit = (Vec<usize>, Vec<usize>);
pub type LeakageTest = LeakageTestResult;