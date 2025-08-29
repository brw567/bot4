//! Canonical RiskLimits - Single Source of Truth
//! Team: Full 8-member collaboration
//! Lead: Cameron (RiskQuant) + Avery (Architect)

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

// Unified RiskLimits supporting all use cases
// Consolidates 7 duplicate definitions into one
// ELIMINATED: pub struct RiskLimits {
// ELIMINATED:     // Position limits
// ELIMINATED:     pub max_position_pct: Decimal,        // % of portfolio
// ELIMINATED:     pub max_position_value: Decimal,      // Absolute value
// ELIMINATED:     pub max_positions_per_symbol: u32,    // Concentration limit
// ELIMINATED:     pub max_total_positions: u32,         // Portfolio limit
// ELIMINATED:     
// ELIMINATED:     // Loss limits
// ELIMINATED:     pub max_loss_per_trade: Decimal,      // Single trade
// ELIMINATED:     pub max_daily_loss: Decimal,          // Daily limit
// ELIMINATED:     pub max_weekly_loss: Decimal,         // Weekly limit
// ELIMINATED:     pub max_drawdown: Decimal,            // Maximum drawdown
// ELIMINATED:     
// ELIMINATED:     // Exposure limits
// ELIMINATED:     pub max_leverage: Decimal,            // Leverage limit
// ELIMINATED:     pub max_gross_exposure: Decimal,      // Gross exposure
// ELIMINATED:     pub max_net_exposure: Decimal,        // Net exposure
// ELIMINATED:     pub max_sector_exposure: Decimal,     // Per sector
// ELIMINATED:     
// ELIMINATED:     // Correlation & diversification
// ELIMINATED:     pub max_correlation: Decimal,         // Between positions
// ELIMINATED:     pub min_diversification: Decimal,     // Minimum required
// ELIMINATED:     pub max_concentration: Decimal,       // Single asset concentration
// ELIMINATED:     
// ELIMINATED:     // Risk metrics thresholds
// ELIMINATED:     pub max_var_95: Decimal,              // 95% VaR limit
// ELIMINATED:     pub max_var_99: Decimal,              // 99% VaR limit
// ELIMINATED:     pub max_expected_shortfall: Decimal,  // CVaR limit
// ELIMINATED:     pub min_sharpe_ratio: Decimal,        // Performance threshold
// ELIMINATED:     
// ELIMINATED:     // Kelly criterion limits
// ELIMINATED:     pub max_kelly_fraction: Decimal,      // Maximum Kelly %
// ELIMINATED:     pub kelly_safety_factor: Decimal,     // Safety multiplier
// ELIMINATED:     
// ELIMINATED:     // Circuit breaker thresholds
// ELIMINATED:     pub circuit_breaker_threshold: Decimal,
// ELIMINATED:     pub emergency_stop_loss: Decimal,
// ELIMINATED:     
// ELIMINATED:     // Operational limits
// ELIMINATED:     pub require_stop_loss: bool,
// ELIMINATED:     pub require_take_profit: bool,
// ELIMINATED:     pub allow_overnight: bool,
// ELIMINATED:     pub allow_weekend: bool,
// ELIMINATED: }

// ELIMINATED: impl Default for RiskLimits {
//     fn default() -> Self {
//         Self {
//             // Conservative defaults for safety
//             max_position_pct: Decimal::from_str("0.02").unwrap(),      // 2%
//             max_position_value: Decimal::from_str("10000").unwrap(),   
//             max_positions_per_symbol: 3,
//             max_total_positions: 20,
//             
//             max_loss_per_trade: Decimal::from_str("100").unwrap(),
//             max_daily_loss: Decimal::from_str("1000").unwrap(),
//             max_weekly_loss: Decimal::from_str("5000").unwrap(),
//             max_drawdown: Decimal::from_str("0.15").unwrap(),         // 15%
//             
//             max_leverage: Decimal::from_str("3.0").unwrap(),
//             max_gross_exposure: Decimal::from_str("1.5").unwrap(),
//             max_net_exposure: Decimal::from_str("1.0").unwrap(),
//             max_sector_exposure: Decimal::from_str("0.3").unwrap(),
//             
//             max_correlation: Decimal::from_str("0.7").unwrap(),
//             min_diversification: Decimal::from_str("0.3").unwrap(),
//             max_concentration: Decimal::from_str("0.25").unwrap(),
//             
//             max_var_95: Decimal::from_str("0.02").unwrap(),
//             max_var_99: Decimal::from_str("0.05").unwrap(),
//             max_expected_shortfall: Decimal::from_str("0.07").unwrap(),
//             min_sharpe_ratio: Decimal::from_str("1.5").unwrap(),
//             
//             max_kelly_fraction: Decimal::from_str("0.25").unwrap(),
//             kelly_safety_factor: Decimal::from_str("0.5").unwrap(),
//             
//             circuit_breaker_threshold: Decimal::from_str("0.05").unwrap(),
//             emergency_stop_loss: Decimal::from_str("0.10").unwrap(),
//             
//             require_stop_loss: true,
//             require_take_profit: false,
//             allow_overnight: true,
//             allow_weekend: false,
//         }
//     }
// }

// ELIMINATED: impl RiskLimits {
//     /// Create production limits (aggressive but safe)
//     pub fn production() -> Self {
//         let mut limits = Self::default();
//         limits.max_kelly_fraction = Decimal::from_str("0.20").unwrap();
//         limits.max_leverage = Decimal::from_str("2.0").unwrap();
//         limits
//     }
//     
//     /// Create conservative limits for volatile markets
//     pub fn conservative() -> Self {
//         let mut limits = Self::default();
//         limits.max_position_pct = Decimal::from_str("0.01").unwrap();
//         limits.max_kelly_fraction = Decimal::from_str("0.10").unwrap();
//         limits.max_leverage = Decimal::ONE;
//         limits
//     }
//     
//     /// Validate if a proposed position meets limits
//     pub fn validate_position(&self, position_size: Decimal, portfolio_value: Decimal) -> bool {
//         let position_pct = position_size / portfolio_value;
//         position_pct <= self.max_position_pct && position_size <= self.max_position_value
//     }
// }
