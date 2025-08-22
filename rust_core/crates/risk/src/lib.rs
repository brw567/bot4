// Risk Management Module - Phase 3+ Implementation
// Team: Quinn (Lead) + Morgan + Sam + Full Team
// CRITICAL: Sophie's requirements integrated

pub mod clamps;
pub mod kelly_sizing;
pub mod kelly_validation;

pub use clamps::{RiskClampSystem, ClampConfig, ClampMetrics};
pub use kelly_sizing::{
    KellySizer, 
    KellyConfig, 
    KellyRecommendation,
    KellyStatistics,
    TradeOutcome,
    RiskAdjustments,
};

// Re-export for convenience
pub fn create_risk_system() -> (RiskClampSystem, KellySizer) {
    let clamp_config = ClampConfig::default();
    let kelly_config = KellyConfig::default();
    
    (
        RiskClampSystem::new(clamp_config),
        KellySizer::new(kelly_config),
    )
}