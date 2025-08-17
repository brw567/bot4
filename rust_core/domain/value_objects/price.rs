// Domain Value Object: Price
// Immutable, no identity, represents a price in the trading domain
// Owner: Sam | Reviewer: Morgan

use std::fmt;
use anyhow::{Result, bail};

/// Price value object - immutable representation of a price
/// 
/// # Invariants
/// - Price must be positive (> 0)
/// - Price must be finite
/// - Price has 8 decimal places precision (satoshi level)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Price(f64);

impl Price {
    /// Create a new Price with validation
    /// 
    /// # Arguments
    /// * `value` - The price value in base currency
    /// 
    /// # Returns
    /// * `Ok(Price)` if valid
    /// * `Err` if price is invalid (<=0, infinite, or NaN)
    /// 
    /// # Example
    /// ```
    /// let price = Price::new(50000.0)?; // BTC at $50k
    /// ```
    pub fn new(value: f64) -> Result<Self> {
        if value <= 0.0 {
            bail!("Price must be positive, got: {}", value);
        }
        if !value.is_finite() {
            bail!("Price must be finite, got: {}", value);
        }
        Ok(Price(value))
    }
    
    /// Get the underlying value
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.0
    }
    
    /// Apply slippage to the price
    /// 
    /// # Arguments
    /// * `slippage_bps` - Slippage in basis points (100 bps = 1%)
    /// 
    /// # Returns
    /// A new Price with slippage applied
    #[inline]
    pub fn apply_slippage(&self, slippage_bps: i32) -> Result<Price> {
        let multiplier = 1.0 + (slippage_bps as f64 / 10000.0);
        Price::new(self.0 * multiplier)
    }
    
    /// Calculate the percentage difference to another price
    /// 
    /// # Returns
    /// Percentage difference (can be negative)
    #[inline]
    pub fn percentage_diff(&self, other: &Price) -> f64 {
        ((other.0 - self.0) / self.0) * 100.0
    }
    
    /// Round to a specific number of decimal places
    #[inline]
    pub fn round_to(&self, decimals: u32) -> Price {
        let multiplier = 10_f64.powi(decimals as i32);
        Price((self.0 * multiplier).round() / multiplier)
    }
    
    /// Check if price is within a range
    #[inline]
    pub fn is_within_range(&self, min: &Price, max: &Price) -> bool {
        self.0 >= min.0 && self.0 <= max.0
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.8}", self.0)
    }
}

// Arithmetic operations return Result to maintain invariants
impl std::ops::Add for Price {
    type Output = Result<Price>;
    
    fn add(self, other: Price) -> Self::Output {
        Price::new(self.0 + other.0)
    }
}

impl std::ops::Sub for Price {
    type Output = Result<Price>;
    
    fn sub(self, other: Price) -> Self::Output {
        Price::new(self.0 - other.0)
    }
}

impl std::ops::Mul<f64> for Price {
    type Output = Result<Price>;
    
    fn mul(self, scalar: f64) -> Self::Output {
        Price::new(self.0 * scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn should_create_valid_price() {
        let price = Price::new(100.0).unwrap();
        assert_eq!(price.value(), 100.0);
    }
    
    #[test]
    fn should_reject_negative_price() {
        let result = Price::new(-100.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_reject_zero_price() {
        let result = Price::new(0.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_reject_infinite_price() {
        let result = Price::new(f64::INFINITY);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_apply_positive_slippage() {
        let price = Price::new(100.0).unwrap();
        let slipped = price.apply_slippage(10).unwrap(); // 0.1% slippage
        assert_eq!(slipped.value(), 100.1);
    }
    
    #[test]
    fn should_apply_negative_slippage() {
        let price = Price::new(100.0).unwrap();
        let slipped = price.apply_slippage(-10).unwrap(); // -0.1% slippage
        assert_eq!(slipped.value(), 99.9);
    }
    
    #[test]
    fn should_calculate_percentage_diff() {
        let price1 = Price::new(100.0).unwrap();
        let price2 = Price::new(110.0).unwrap();
        assert_eq!(price1.percentage_diff(&price2), 10.0);
    }
    
    #[test]
    fn should_round_to_decimals() {
        let price = Price::new(100.123456789).unwrap();
        let rounded = price.round_to(2);
        assert_eq!(rounded.value(), 100.12);
    }
    
    #[test]
    fn should_check_range() {
        let price = Price::new(100.0).unwrap();
        let min = Price::new(90.0).unwrap();
        let max = Price::new(110.0).unwrap();
        assert!(price.is_within_range(&min, &max));
    }
    
    #[test]
    fn should_add_prices() {
        let p1 = Price::new(100.0).unwrap();
        let p2 = Price::new(50.0).unwrap();
        let sum = (p1 + p2).unwrap();
        assert_eq!(sum.value(), 150.0);
    }
}