// Risk Limits Configuration
// Quinn's immutable rules - these are NOT suggestions

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Master risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub position_limits: PositionLimits,
    pub loss_limits: LossLimits,
    pub exposure_limits: ExposureLimits,
    pub correlation_limits: CorrelationLimits,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            position_limits: PositionLimits::default(),
            loss_limits: LossLimits::default(),
            exposure_limits: ExposureLimits::default(),
            correlation_limits: CorrelationLimits::default(),
        }
    }
}

/// Position-specific limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimits {
    /// Maximum position size as % of portfolio (Quinn's 2% rule)
    pub max_position_size_pct: Decimal,
    
    /// Maximum number of open positions
    pub max_open_positions: usize,
    
    /// Maximum positions per symbol
    pub max_positions_per_symbol: usize,
    
    /// Require stop loss on all positions
    pub require_stop_loss: bool,
    
    /// Default stop loss percentage
    pub default_stop_loss_pct: Decimal,
    
    /// Maximum leverage allowed
    pub max_leverage: Decimal,
    
    /// Maximum total exposure
    pub max_total_exposure: Decimal,
}

impl Default for PositionLimits {
    fn default() -> Self {
        Self {
            max_position_size_pct: dec!(0.02),      // 2% max
            max_open_positions: 20,
            max_positions_per_symbol: 3,
            require_stop_loss: true,                // Quinn insists
            default_stop_loss_pct: dec!(0.02),      // 2% stop loss
            max_leverage: dec!(3),                  // 3x max leverage
            max_total_exposure: dec!(100000),       // $100k max exposure
        }
    }
}

/// Loss limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossLimits {
    /// Maximum daily loss
    pub daily_loss_limit: Option<Decimal>,
    
    /// Maximum weekly loss
    pub weekly_loss_limit: Option<Decimal>,
    
    /// Maximum monthly loss
    pub monthly_loss_limit: Option<Decimal>,
    
    /// Maximum drawdown percentage (15% per requirements)
    pub max_drawdown_pct: Decimal,
    
    /// Consecutive loss trades before pause
    pub max_consecutive_losses: usize,
    
    /// Loss amount that triggers emergency stop
    pub emergency_stop_loss: Decimal,
}

impl Default for LossLimits {
    fn default() -> Self {
        Self {
            daily_loss_limit: Some(dec!(2000)),     // $2k daily loss limit
            weekly_loss_limit: Some(dec!(5000)),    // $5k weekly loss limit
            monthly_loss_limit: Some(dec!(10000)),  // $10k monthly loss limit
            max_drawdown_pct: dec!(0.15),          // 15% max drawdown
            max_consecutive_losses: 5,
            emergency_stop_loss: dec!(5000),        // Emergency stop at $5k loss
        }
    }
}

/// Exposure limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureLimits {
    /// Maximum exposure per asset class
    pub max_crypto_exposure: Decimal,
    
    /// Maximum exposure per exchange
    pub max_exchange_exposure: Decimal,
    
    /// Maximum exposure in single direction (long/short)
    pub max_directional_exposure: Decimal,
    
    /// Maximum notional value
    pub max_notional_value: Decimal,
}

impl Default for ExposureLimits {
    fn default() -> Self {
        Self {
            max_crypto_exposure: dec!(50000),       // $50k max in crypto
            max_exchange_exposure: dec!(30000),     // $30k max per exchange
            max_directional_exposure: dec!(40000),  // $40k max long or short
            max_notional_value: dec!(100000),       // $100k max notional
        }
    }
}

/// Correlation limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationLimits {
    /// Maximum correlation between positions (0.7 per requirements)
    pub max_position_correlation: f64,
    
    /// Maximum number of correlated positions
    pub max_correlated_positions: usize,
    
    /// Correlation calculation window (days)
    pub correlation_window_days: usize,
    
    /// Minimum data points for correlation calculation
    pub min_correlation_samples: usize,
}

impl Default for CorrelationLimits {
    fn default() -> Self {
        Self {
            max_position_correlation: 0.7,          // 70% max correlation
            max_correlated_positions: 3,
            correlation_window_days: 30,
            min_correlation_samples: 20,
        }
    }
}

impl RiskLimits {
    /// Create conservative limits (Quinn's preference)
    pub fn conservative() -> Self {
        Self {
            position_limits: PositionLimits {
                max_position_size_pct: dec!(0.01),  // 1% only
                max_open_positions: 10,
                max_positions_per_symbol: 1,
                require_stop_loss: true,
                default_stop_loss_pct: dec!(0.01),  // 1% stop
                max_leverage: dec!(1),              // No leverage
                max_total_exposure: dec!(50000),
            },
            loss_limits: LossLimits {
                daily_loss_limit: Some(dec!(1000)),
                weekly_loss_limit: Some(dec!(2500)),
                monthly_loss_limit: Some(dec!(5000)),
                max_drawdown_pct: dec!(0.10),       // 10% max drawdown
                max_consecutive_losses: 3,
                emergency_stop_loss: dec!(2500),
            },
            exposure_limits: ExposureLimits {
                max_crypto_exposure: dec!(25000),
                max_exchange_exposure: dec!(15000),
                max_directional_exposure: dec!(20000),
                max_notional_value: dec!(50000),
            },
            correlation_limits: CorrelationLimits {
                max_position_correlation: 0.5,       // 50% max correlation
                max_correlated_positions: 2,
                correlation_window_days: 30,
                min_correlation_samples: 30,
            },
        }
    }
    
    /// Create aggressive limits (not recommended by Quinn)
    pub fn aggressive() -> Self {
        Self {
            position_limits: PositionLimits {
                max_position_size_pct: dec!(0.05),  // 5% (risky!)
                max_open_positions: 50,
                max_positions_per_symbol: 5,
                require_stop_loss: true,            // Still mandatory
                default_stop_loss_pct: dec!(0.05),
                max_leverage: dec!(10),             // 10x (very risky!)
                max_total_exposure: dec!(500000),
            },
            loss_limits: LossLimits {
                daily_loss_limit: Some(dec!(10000)),
                weekly_loss_limit: Some(dec!(25000)),
                monthly_loss_limit: Some(dec!(50000)),
                max_drawdown_pct: dec!(0.30),       // 30% drawdown (ouch!)
                max_consecutive_losses: 10,
                emergency_stop_loss: dec!(25000),
            },
            exposure_limits: ExposureLimits {
                max_crypto_exposure: dec!(250000),
                max_exchange_exposure: dec!(150000),
                max_directional_exposure: dec!(200000),
                max_notional_value: dec!(500000),
            },
            correlation_limits: CorrelationLimits {
                max_position_correlation: 0.9,       // High correlation allowed
                max_correlated_positions: 10,
                correlation_window_days: 7,
                min_correlation_samples: 10,
            },
        }
    }
    
    /// Validate limits are sensible
    pub fn validate(&self) -> Result<(), String> {
        // Position size can't exceed 100%
        if self.position_limits.max_position_size_pct > dec!(1) {
            return Err("Position size cannot exceed 100%".to_string());
        }
        
        // Drawdown can't exceed 100%
        if self.loss_limits.max_drawdown_pct > dec!(1) {
            return Err("Drawdown cannot exceed 100%".to_string());
        }
        
        // Correlation must be between -1 and 1
        if self.correlation_limits.max_position_correlation > 1.0 
            || self.correlation_limits.max_position_correlation < -1.0 {
            return Err("Correlation must be between -1 and 1".to_string());
        }
        
        // Stop loss should be reasonable
        if self.position_limits.default_stop_loss_pct > dec!(0.5) {
            return Err("Default stop loss >50% is unreasonable".to_string());
        }
        
        Ok(())
    }
    
    /// Get limit by name (for dynamic configuration)
    pub fn get_limit(&self, name: &str) -> Option<Decimal> {
        match name {
            "max_position_size" => Some(self.position_limits.max_position_size_pct),
            "max_leverage" => Some(self.position_limits.max_leverage),
            "max_drawdown" => Some(self.loss_limits.max_drawdown_pct),
            "max_exposure" => Some(self.position_limits.max_total_exposure),
            "daily_loss_limit" => self.loss_limits.daily_loss_limit,
            "emergency_stop" => Some(self.loss_limits.emergency_stop_loss),
            _ => None,
        }
    }
}

/// Risk limit adjustments (requires Quinn's approval)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitAdjustment {
    pub limit_name: String,
    pub old_value: Decimal,
    pub new_value: Decimal,
    pub reason: String,
    pub approved_by: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl LimitAdjustment {
    pub fn new(limit_name: String, old_value: Decimal, new_value: Decimal, reason: String) -> Self {
        Self {
            limit_name,
            old_value,
            new_value,
            reason,
            approved_by: "Pending".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Quinn approves the adjustment
    pub fn approve(&mut self, approver: &str) {
        self.approved_by = approver.to_string();
    }
    
    /// Check if adjustment is approved
    pub fn is_approved(&self) -> bool {
        self.approved_by != "Pending"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_limits() {
        let limits = RiskLimits::default();
        assert_eq!(limits.position_limits.max_position_size_pct, dec!(0.02));
        assert!(limits.position_limits.require_stop_loss);
        assert_eq!(limits.loss_limits.max_drawdown_pct, dec!(0.15));
    }
    
    #[test]
    fn test_conservative_limits() {
        let limits = RiskLimits::conservative();
        assert_eq!(limits.position_limits.max_position_size_pct, dec!(0.01));
        assert_eq!(limits.position_limits.max_leverage, dec!(1));
    }
    
    #[test]
    fn test_limit_validation() {
        let mut limits = RiskLimits::default();
        assert!(limits.validate().is_ok());
        
        // Invalid position size
        limits.position_limits.max_position_size_pct = dec!(1.5);
        assert!(limits.validate().is_err());
        
        // Invalid drawdown
        limits = RiskLimits::default();
        limits.loss_limits.max_drawdown_pct = dec!(1.5);
        assert!(limits.validate().is_err());
    }
}