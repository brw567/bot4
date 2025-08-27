//! # Mathematical Operations - Single Source of Truth
//! 
//! Consolidates 30+ duplicate mathematical function implementations into
//! one optimized, thoroughly tested, SIMD-accelerated library.
//!
//! ## Design Principles (DEEP DIVE)
//! - DRY: One implementation per mathematical concept
//! - Performance: SIMD optimization with runtime detection
//! - Accuracy: High-precision calculations for financial data
//! - Testability: Property-based testing for correctness
//! - Maintainability: Clear, documented algorithms
//!
//! ## Architecture Compliance
//! Layer: 1.6.2 - Mathematical Functions Consolidation
//! Owner: Full 8-member team collaboration
//! Hours: 40 (Week 2 of deduplication sprint)
//!
//! ## External Research Applied
//! - "Numerical Recipes in C" (Press et al.)
//! - "Statistics and Data Analysis for Financial Engineering" (Ruppert)
//! - "Machine Learning for Asset Managers" (López de Prado)
//! - Intel SIMD Programming Guide
//! - NVIDIA CUDA Math Libraries (algorithms only)

#![warn(missing_docs)]
#![deny(unsafe_code)]  // We'll selectively allow for SIMD
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Core mathematical modules
pub mod correlation;
pub mod variance;
pub mod kelly;
pub mod volatility;
pub mod returns;

// Technical indicators
pub mod indicators;

// Statistical functions
pub mod statistics;

// Risk metrics
pub mod risk_metrics;

// Matrix operations
pub mod matrix_ops;

// SIMD utilities
#[cfg(feature = "simd")]
pub mod simd;

// Utilities
pub mod utils;

// Re-export primary functions
pub use correlation::{calculate_correlation, CorrelationMethod};
pub use variance::{calculate_var, calculate_cvar, VarMethod};
pub use kelly::{calculate_kelly, calculate_fractional_kelly, KellyConfig};
pub use volatility::{calculate_volatility, VolatilityModel};
pub use returns::{calculate_returns, calculate_log_returns, ReturnsType};

// Re-export indicators
pub use indicators::{
    calculate_ema, calculate_sma, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_stochastic,
    IndicatorConfig,
};

// Re-export statistics
pub use statistics::{
    mean, median, mode, standard_deviation, skewness, kurtosis,
    percentile, quantile,
};

// Re-export risk metrics
pub use risk_metrics::{
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    value_at_risk, conditional_value_at_risk,
};

/// Version of the mathematical operations library
pub const MATHEMATICAL_OPS_VERSION: &str = "1.0.0";

/// Prelude for common imports
pub mod prelude {
    pub use crate::{
        calculate_correlation, calculate_var, calculate_kelly,
        calculate_volatility, calculate_returns,
        calculate_ema, calculate_sma, calculate_rsi,
        sharpe_ratio, sortino_ratio, max_drawdown,
        CorrelationMethod, VarMethod, KellyConfig, VolatilityModel,
    };
}