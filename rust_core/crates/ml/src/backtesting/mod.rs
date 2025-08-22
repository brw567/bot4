// Backtesting Module - Morgan's Anti-Overfitting Arsenal
// Team: Morgan (Lead) + Riley (Testing) + Full Team

pub mod walk_forward;

pub use walk_forward::{
    WalkForwardAnalysis,
    WalkForwardConfig,
    WalkForwardResults,
    WindowPerformance,
    TradingModel,
};