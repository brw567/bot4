// Value Object: Fee
// Immutable fee representation with maker/taker rates
// Addresses Sophia's #3 critical feedback
// Owner: Casey | Reviewer: Quinn

use anyhow::{Result, bail};
use std::fmt;

/// Fee amount with currency
#[derive(Debug, Clone, PartialEq)]
pub struct Fee {
    /// Fee amount (can be negative for rebates)
    amount: f64,
    /// Currency of the fee
    currency: String,
    /// Whether this is a rebate (negative fee)
    is_rebate: bool,
}

impl Fee {
    /// Create a new fee
    pub fn new(amount: f64, currency: String) -> Result<Self> {
        if !amount.is_finite() {
            bail!("Fee amount must be finite");
        }
        
        Ok(Fee {
            amount: amount.abs(),
            currency,
            is_rebate: amount < 0.0,
        })
    }
    
    /// Create a zero fee
    pub fn zero(currency: String) -> Self {
        Fee {
            amount: 0.0,
            currency,
            is_rebate: false,
        }
    }
    
    /// Get the fee amount (negative if rebate)
    pub fn amount(&self) -> f64 {
        if self.is_rebate {
            -self.amount
        } else {
            self.amount
        }
    }
    
    /// Get the absolute fee amount
    pub fn abs_amount(&self) -> f64 {
        self.amount
    }
    
    /// Get the currency
    pub fn currency(&self) -> &str {
        &self.currency
    }
    
    /// Check if this is a rebate
    pub fn is_rebate(&self) -> bool {
        self.is_rebate
    }
    
    /// Add two fees (must be same currency)
    pub fn add(&self, other: &Fee) -> Result<Fee> {
        if self.currency != other.currency {
            bail!("Cannot add fees with different currencies: {} vs {}", 
                  self.currency, other.currency);
        }
        
        let total = self.amount() + other.amount();
        Fee::new(total, self.currency.clone())
    }
}

impl fmt::Display for Fee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_rebate {
            write!(f, "-{:.8} {}", self.amount, self.currency)
        } else {
            write!(f, "{:.8} {}", self.amount, self.currency)
        }
    }
}

/// Fee tier for volume-based fee schedules
#[derive(Debug, Clone)]
pub struct FeeTier {
    /// Minimum volume for this tier (in quote currency)
    pub min_volume: f64,
    /// Maker fee in basis points (can be negative for rebates)
    pub maker_fee_bps: i32,
    /// Taker fee in basis points
    pub taker_fee_bps: i32,
}

/// Fee model for calculating trading fees
#[derive(Debug, Clone)]
pub struct FeeModel {
    /// Base maker fee in basis points (can be negative for rebates)
    maker_fee_bps: i32,
    /// Base taker fee in basis points
    taker_fee_bps: i32,
    /// Minimum fee amount (in quote currency)
    min_fee: Option<f64>,
    /// Maximum fee amount (in quote currency)
    max_fee: Option<f64>,
    /// Volume-based fee tiers (optional)
    tiers: Vec<FeeTier>,
}

impl FeeModel {
    /// Create a new fee model
    pub fn new(maker_fee_bps: i32, taker_fee_bps: i32) -> Self {
        FeeModel {
            maker_fee_bps,
            taker_fee_bps,
            min_fee: None,
            max_fee: None,
            tiers: Vec::new(),
        }
    }
    
    /// Create a standard exchange fee model (Binance-like)
    pub fn standard() -> Self {
        FeeModel {
            maker_fee_bps: 10,  // 0.10% maker fee
            taker_fee_bps: 10,  // 0.10% taker fee
            min_fee: None,
            max_fee: None,
            tiers: vec![
                FeeTier {
                    min_volume: 0.0,
                    maker_fee_bps: 10,
                    taker_fee_bps: 10,
                },
                FeeTier {
                    min_volume: 50_000.0,
                    maker_fee_bps: 9,
                    taker_fee_bps: 10,
                },
                FeeTier {
                    min_volume: 100_000.0,
                    maker_fee_bps: 8,
                    taker_fee_bps: 10,
                },
                FeeTier {
                    min_volume: 500_000.0,
                    maker_fee_bps: 2,
                    taker_fee_bps: 6,
                },
                FeeTier {
                    min_volume: 1_000_000.0,
                    maker_fee_bps: 0,
                    taker_fee_bps: 4,
                },
                FeeTier {
                    min_volume: 5_000_000.0,
                    maker_fee_bps: -2,  // Rebate for high-volume makers
                    taker_fee_bps: 3,
                },
            ],
        }
    }
    
    /// Create a zero-fee model (for testing)
    pub fn zero() -> Self {
        FeeModel {
            maker_fee_bps: 0,
            taker_fee_bps: 0,
            min_fee: None,
            max_fee: None,
            tiers: Vec::new(),
        }
    }
    
    /// Set minimum fee
    pub fn with_min_fee(mut self, min_fee: f64) -> Self {
        self.min_fee = Some(min_fee);
        self
    }
    
    /// Set maximum fee
    pub fn with_max_fee(mut self, max_fee: f64) -> Self {
        self.max_fee = Some(max_fee);
        self
    }
    
    /// Add fee tiers
    pub fn with_tiers(mut self, tiers: Vec<FeeTier>) -> Self {
        self.tiers = tiers;
        self
    }
    
    /// Get the applicable fee tier for a given volume
    fn get_tier(&self, volume_30d: f64) -> (i32, i32) {
        if self.tiers.is_empty() {
            return (self.maker_fee_bps, self.taker_fee_bps);
        }
        
        // Find the highest tier that applies
        let mut maker = self.maker_fee_bps;
        let mut taker = self.taker_fee_bps;
        
        for tier in &self.tiers {
            if volume_30d >= tier.min_volume {
                maker = tier.maker_fee_bps;
                taker = tier.taker_fee_bps;
            } else {
                break;
            }
        }
        
        (maker, taker)
    }
    
    /// Calculate fee for a fill
    pub fn calculate_fee(
        &self,
        quantity: f64,
        price: f64,
        is_maker: bool,
        volume_30d: f64,
        quote_currency: String,
    ) -> Fee {
        // Get applicable tier
        let (maker_bps, taker_bps) = self.get_tier(volume_30d);
        
        // Select fee rate
        let fee_bps = if is_maker { maker_bps } else { taker_bps };
        
        // Calculate raw fee
        let notional = quantity * price;
        let raw_fee = notional * (fee_bps as f64 / 10000.0);
        
        // Apply min/max limits
        let mut final_fee = raw_fee;
        
        if let Some(min) = self.min_fee {
            if raw_fee.abs() < min && raw_fee != 0.0 {
                final_fee = if raw_fee < 0.0 { -min } else { min };
            }
        }
        
        if let Some(max) = self.max_fee {
            if raw_fee.abs() > max {
                final_fee = if raw_fee < 0.0 { -max } else { max };
            }
        }
        
        Fee::new(final_fee, quote_currency).expect("SAFETY: Add proper error handling")
    }
}

/// Fill with fee information
#[derive(Debug, Clone)]
pub struct FillWithFee {
    /// Fill quantity
    pub quantity: f64,
    /// Fill price
    pub price: f64,
    /// Whether this was a maker fill
    pub is_maker: bool,
    /// Fee charged/rebated
    pub fee: Fee,
    /// Timestamp
    pub timestamp: i64,
}

impl FillWithFee {
    /// Calculate net proceeds after fee
    pub fn net_proceeds(&self, is_buy: bool) -> f64 {
        let gross = self.quantity * self.price;
        
        if is_buy {
            // For buys, we pay the fee on top
            gross + self.fee.amount()  // fee.amount() is negative for rebates
        } else {
            // For sells, we receive less
            gross - self.fee.amount()
        }
    }
    
    /// Calculate effective price including fees
    pub fn effective_price(&self, is_buy: bool) -> f64 {
        if is_buy {
            // For buys, effective price is higher due to fees
            self.price * (1.0 + self.fee.amount() / (self.quantity * self.price))
        } else {
            // For sells, effective price is lower due to fees
            self.price * (1.0 - self.fee.amount() / (self.quantity * self.price))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fee_creation() {
        let fee = Fee::new(10.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        assert_eq!(fee.amount(), 10.0);
        assert_eq!(fee.currency(), "USDT");
        assert!(!fee.is_rebate());
        
        let rebate = Fee::new(-5.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        assert_eq!(rebate.amount(), -5.0);
        assert_eq!(rebate.abs_amount(), 5.0);
        assert!(rebate.is_rebate());
    }
    
    #[test]
    fn test_fee_addition() {
        let fee1 = Fee::new(10.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        let fee2 = Fee::new(5.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        let total = fee1.add(&fee2).expect("SAFETY: Add proper error handling");
        assert_eq!(total.amount(), 15.0);
        
        let fee3 = Fee::new(-3.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        let net = total.add(&fee3).expect("SAFETY: Add proper error handling");
        assert_eq!(net.amount(), 12.0);
    }
    
    #[test]
    fn test_fee_model_basic() {
        let model = FeeModel::new(10, 20); // 0.1% maker, 0.2% taker
        
        // Taker fee
        let fee = model.calculate_fee(
            1.0,          // quantity
            50000.0,      // price
            false,        // is_maker
            0.0,          // volume_30d
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), 10.0); // 50000 * 0.002 = 100
        
        // Maker fee
        let fee = model.calculate_fee(
            1.0,
            50000.0,
            true,         // is_maker
            0.0,
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), 5.0); // 50000 * 0.001 = 50
    }
    
    #[test]
    fn test_fee_model_with_tiers() {
        let model = FeeModel::standard();
        
        // Low volume - standard fees
        let fee = model.calculate_fee(
            1.0,
            50000.0,
            true,
            0.0,  // No volume
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), 5.0); // 0.1% of 50000
        
        // High volume - rebate
        let fee = model.calculate_fee(
            1.0,
            50000.0,
            true,
            10_000_000.0,  // High volume
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), -1.0); // -0.02% rebate
        assert!(fee.is_rebate());
    }
    
    #[test]
    fn test_fee_model_min_max() {
        let model = FeeModel::new(1, 1)  // 0.01% fees
            .with_min_fee(1.0)
            .with_max_fee(100.0);
        
        // Below minimum
        let fee = model.calculate_fee(
            0.001,
            100.0,
            false,
            0.0,
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), 1.0); // Minimum fee applied
        
        // Above maximum
        let fee = model.calculate_fee(
            100.0,
            100000.0,
            false,
            0.0,
            "USDT".to_string(),
        );
        assert_eq!(fee.amount(), 100.0); // Maximum fee applied
    }
    
    #[test]
    fn test_fill_with_fee() {
        let fee = Fee::new(10.0, "USDT".to_string()).expect("SAFETY: Add proper error handling");
        let fill = FillWithFee {
            quantity: 1.0,
            price: 50000.0,
            is_maker: false,
            fee,
            timestamp: 1234567890,
        };
        
        // Buy: pay 50000 + 10 fee = 50010
        assert_eq!(fill.net_proceeds(true), 50010.0);
        assert_eq!(fill.effective_price(true), 50010.0);
        
        // Sell: receive 50000 - 10 fee = 49990
        assert_eq!(fill.net_proceeds(false), 49990.0);
        assert_eq!(fill.effective_price(false), 49990.0);
    }
}