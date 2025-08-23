// Risk Management Module - Phase 3+ Implementation
// Team: Quinn (Lead) + Morgan + Sam + Full Team
// CRITICAL: Sophie's requirements integrated

pub mod clamps;
pub mod kelly_sizing;
pub mod kelly_validation;
pub mod garch;
pub mod isotonic;
pub mod auto_tuning;  // Auto-adaptation system
pub mod auto_tuning_persistence; // Database persistence for auto-tuning
pub mod decision_orchestrator; // UNIFIED ML+TA decision maker
pub mod order_book_analytics; // Order book imbalance, VPIN, Kyle's Lambda - CRITICAL!
pub mod unified_types;  // Unified type system - solves API mismatch
pub mod profit_extractor;  // Profit extraction engine - makes MONEY!
pub mod ml_feedback;  // ML feedback loops - CRITICAL for continuous improvement!
pub mod market_analytics;  // REAL market calculations - NO SIMPLIFICATIONS!
pub mod portfolio_manager;  // Portfolio state management - NO HARDCODED VALUES!
pub mod funding_rates;  // Funding rate arbitrage - 15-30% ADDITIONAL PROFIT!
pub mod optimal_execution;  // TWAP/VWAP/POV execution algorithms - MINIMIZE MARKET IMPACT!
pub mod monte_carlo;  // Monte Carlo simulations - VALIDATE STRATEGIES UNDER ALL CONDITIONS!
pub mod feature_importance;  // SHAP values for ML explainability - UNDERSTAND WHAT DRIVES PREDICTIONS!
pub mod hyperparameter_optimization;  // Bayesian hyperparameter optimization - AUTO-TUNING WITH TPE!
pub mod hyperparameter_integration;  // Integration of optimization into ALL components - MAXIMUM EXTRACTION!
pub mod deep_dive_validation_study;  // Academic validation - NO SIMPLIFICATIONS!
pub mod parameter_manager;  // CRITICAL: No hardcoded values - ALL parameters auto-tuned!
pub mod game_theory_advanced;  // DEEP DIVE: Full game theory implementation - Nash, Prisoner's Dilemma, Multi-agent!
pub mod performance_optimizations;  // DEEP DIVE: Zero allocations, lock-free, <1μs latency!

#[cfg(test)]
mod comprehensive_tests;
#[cfg(test)]
mod hyperparameter_optimization_tests;
#[cfg(test)]
mod kyle_lambda_validation;
#[cfg(test)]
mod vpin_validation;
#[cfg(test)]
mod manipulation_detection;
#[cfg(test)]
mod auto_tuning_test;
#[cfg(test)]
mod deep_dive_tests; // DEEP DIVE tests - NO SIMPLIFICATIONS!
#[cfg(test)]
mod ml_integration_tests; // ML INTEGRATION tests - CRITICAL FOR CONTINUOUS IMPROVEMENT!
#[cfg(test)]
mod ta_accuracy_audit; // TA ACCURACY AUDIT - NO SIMPLIFICATIONS!
#[cfg(test)]
mod risk_chain_audit; // COMPLETE RISK CHAIN AUDIT - EVERY LINK VERIFIED!
#[cfg(test)]
mod deep_dive_integration_tests; // DEEP DIVE integration tests - ALL enhancements validated!
mod ta_improvements; // ADVANCED TA INDICATORS - PROPER IMPLEMENTATIONS!

pub use clamps::{RiskClampSystem, ClampConfig, ClampMetrics};
pub use kelly_sizing::{
    KellySizer, 
    KellyConfig, 
    KellyRecommendation,
    KellyStatistics,
    TradeOutcome,
    RiskAdjustments,
};
pub use garch::GARCHModel;
pub use isotonic::{IsotonicCalibrator, MarketRegime};
pub use ml_feedback::{MLFeedbackSystem, MLMetrics};
pub use profit_extractor::{ProfitExtractor, PerformanceStats, ExtendedMarketData};

// Re-export for convenience
pub fn create_risk_system() -> (RiskClampSystem, KellySizer) {
    let clamp_config = ClampConfig::default();
    let kelly_config = KellyConfig::default();
    
    (
        RiskClampSystem::new(clamp_config),
        KellySizer::new(kelly_config),
    )
}