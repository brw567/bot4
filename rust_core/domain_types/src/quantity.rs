//! # Canonical Quantity Type
//! 
//! Consolidates multiple Quantity/Volume/Size implementations into one canonical type.
//! Ensures quantities are always positive and within valid trading ranges.
//!
//! ## Design Decisions
//! - Immutable value object
//! - Decimal precision for accurate lot sizes
//! - Support for fractional quantities (crypto)
//! - Min/max lot size validation

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Errors that can occur with Quantity operations
#[derive(Debug, Error, Clone, PartialEq)]
/// TODO: Add docs
pub enum QuantityError {
    /// Quantity cannot be negative
    #[error("Quantity cannot be negative: {0}")]
    NegativeQuantity(Decimal),
    
    /// Quantity cannot be zero
    #[error("Quantity cannot be zero")]
    ZeroQuantity,
    
    /// Quantity below minimum lot size
    #[error("Quantity {0} below minimum lot size {1}")]
    BelowMinimum(Decimal, Decimal),
    
    /// Quantity exceeds maximum allowed
    #[error("Quantity {0} exceeds maximum {1}")]
    ExceedsMaximum(Decimal, Decimal),
    
    /// Invalid quantity format
    #[error("Invalid quantity format: {0}")]
    InvalidFormat(String),
    
    /// Overflow in calculation
    #[error("Quantity calculation overflow")]
    Overflow,
    
    /// Division by zero
    #[error("Division by zero")]
    DivisionByZero,
    
    /// Not a valid lot size multiple
    #[error("Quantity {0} is not a multiple of lot size {1}")]
    InvalidLotSize(Decimal, Decimal),
}

/// Canonical Quantity type for all volume/size/amount values
///
/// # Invariants
/// - Always non-negative
/// - Can be zero for special cases (e.g., fully filled orders)
/// - Respects minimum and maximum lot sizes
///
/// # Examples
/// ```
/// use domain_types::Quantity;
/// 
/// let qty = Quantity::new(0.001)?; // 0.001 BTC
/// let doubled = qty.multiply(2.0)?;
/// assert_eq!(doubled.as_decimal().to_string(), "0.002");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Quantity {
    value: Decimal,
    precision: u32,
}

impl Quantity {
    /// Maximum allowed quantity (1 billion units)
    pub const MAX: Decimal = Decimal::from_parts(1_000_000_000, 0, 0, false, 0);
    
    /// Minimum tradeable quantity (satoshi level for BTC)
    pub const MIN: Decimal = Decimal::from_parts(1, 0, 0, false, 8); // 0.00000001
    
    /// Creates a new Quantity
    ///
    /// # Errors
    /// Returns error if quantity is negative or exceeds maximum
    pub fn new(value: impl Into<Decimal>) -> Result<Self, QuantityError> {
        let decimal_value = value.into();
        
        if decimal_value < Decimal::ZERO {
            return Err(QuantityError::NegativeQuantity(decimal_value));
        }
        
        if decimal_value > Self::MAX {
            return Err(QuantityError::ExceedsMaximum(decimal_value, Self::MAX));
        }
        
        Ok(Self {
            value: decimal_value,
            precision: Self::detect_precision(decimal_value),
        })
    }
    
    /// Creates a Quantity that must be non-zero
    pub fn new_non_zero(value: impl Into<Decimal>) -> Result<Self, QuantityError> {
        let decimal_value = value.into();
        
        if decimal_value <= Decimal::ZERO {
            if decimal_value.is_zero() {
                return Err(QuantityError::ZeroQuantity);
            }
            return Err(QuantityError::NegativeQuantity(decimal_value));
        }
        
        if decimal_value > Self::MAX {
            return Err(QuantityError::ExceedsMaximum(decimal_value, Self::MAX));
        }
        
        Ok(Self {
            value: decimal_value,
            precision: Self::detect_precision(decimal_value),
        })
    }
    
    /// Creates a zero quantity
    pub const fn zero() -> Self {
        Self {
            value: Decimal::ZERO,
            precision: 8,
        }
    }
    
    /// Checks if quantity is zero
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
    
    /// Gets the underlying decimal value
    pub const fn as_decimal(&self) -> Decimal {
        self.value
    }
    
    /// Gets the value as f64 (use with caution)
    pub fn as_f64(&self) -> f64 {
        self.value.to_f64().unwrap_or(0.0)
    }
    
    /// Gets the precision
    pub const fn precision(&self) -> u32 {
        self.precision
    }
    
    /// Sets precision for display
    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision;
        self
    }
    
    /// Rounds to specified decimal places
    pub fn round(&self, decimal_places: u32) -> Self {
        Self {
            value: self.value.round_dp(decimal_places),
            precision: decimal_places,
        }
    }
    
    /// Validates quantity meets lot size requirements
    pub fn validate_lot_size(&self, lot_size: Quantity) -> Result<(), QuantityError> {
        if lot_size.is_zero() {
            return Ok(()); // No lot size restriction
        }
        
        let remainder = self.value % lot_size.value;
        if !remainder.is_zero() {
            return Err(QuantityError::InvalidLotSize(self.value, lot_size.value));
        }
        
        Ok(())
    }
    
    /// Rounds to nearest valid lot size
    pub fn round_to_lot_size(&self, lot_size: Quantity) -> Result<Self, QuantityError> {
        if lot_size.is_zero() {
            return Ok(*self);
        }
        
        let lots = (self.value / lot_size.value).round();
        let rounded_value = lots * lot_size.value;
        Self::new(rounded_value)
    }
    
    /// Validates quantity is within min/max range
    pub fn validate_range(&self, min: Option<Quantity>, max: Option<Quantity>) -> Result<(), QuantityError> {
        if let Some(min_qty) = min {
            if self.value < min_qty.value {
                return Err(QuantityError::BelowMinimum(self.value, min_qty.value));
            }
        }
        
        if let Some(max_qty) = max {
            if self.value > max_qty.value {
                return Err(QuantityError::ExceedsMaximum(self.value, max_qty.value));
            }
        }
        
        Ok(())
    }
    
    /// Adds another quantity
    pub fn add(&self, other: Quantity) -> Result<Self, QuantityError> {
        let result = self.value.checked_add(other.value)
            .ok_or(QuantityError::Overflow)?;
        Self::new(result)
    }
    
    /// Subtracts another quantity
    pub fn subtract(&self, other: Quantity) -> Result<Self, QuantityError> {
        let result = self.value.checked_sub(other.value)
            .ok_or(QuantityError::Overflow)?;
        
        if result < Decimal::ZERO {
            return Err(QuantityError::NegativeQuantity(result));
        }
        
        Self::new(result)
    }
    
    /// Multiplies by a scalar
    pub fn multiply(&self, scalar: impl Into<Decimal>) -> Result<Self, QuantityError> {
        let multiplier = scalar.into();
        let result = self.value.checked_mul(multiplier)
            .ok_or(QuantityError::Overflow)?;
        Self::new(result)
    }
    
    /// Divides by a scalar
    pub fn divide(&self, divisor: impl Into<Decimal>) -> Result<Self, QuantityError> {
        let div_value = divisor.into();
        
        if div_value.is_zero() {
            return Err(QuantityError::DivisionByZero);
        }
        
        let result = self.value.checked_div(div_value)
            .ok_or(QuantityError::Overflow)?;
        Self::new(result)
    }
    
    /// Calculates percentage of another quantity
    pub fn percentage_of(&self, total: Quantity) -> Result<Decimal, QuantityError> {
        if total.is_zero() {
            return Err(QuantityError::DivisionByZero);
        }
        
        Ok(self.value / total.value * Decimal::from(100))
    }
    
    /// Splits quantity into n equal parts
    pub fn split(&self, parts: u32) -> Result<Vec<Quantity>, QuantityError> {
        if parts == 0 {
            return Err(QuantityError::DivisionByZero);
        }
        
        let part_size = self.value / Decimal::from(parts);
        let mut result = Vec::with_capacity(parts as usize);
        
        for i in 0..parts {
            if i == parts - 1 {
                // Last part gets any remainder due to rounding
                let sum: Decimal = result.iter().map(|q: &Quantity| q.value).sum();
                let remainder = self.value - sum;
                result.push(Self::new(remainder)?);
            } else {
                result.push(Self::new(part_size)?);
            }
        }
        
        Ok(result)
    }
    
    /// Detects precision from decimal value
    fn detect_precision(value: Decimal) -> u32 {
        value.scale()
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.prec$}", self.value, prec = self.precision as usize)
    }
}

impl FromStr for Quantity {
    type Err = QuantityError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = Decimal::from_str(s)
            .map_err(|_| QuantityError::InvalidFormat(s.to_string()))?;
        Self::new(value)
    }
}

impl TryFrom<f64> for Quantity {
    type Error = QuantityError;
    
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(QuantityError::InvalidFormat("Non-finite float".to_string()));
        }
        
        let decimal = Decimal::try_from(value)
            .map_err(|_| QuantityError::InvalidFormat(format!("Invalid float: {}", value)))?;
        Self::new(decimal)
    }
}

impl From<Quantity> for Decimal {
    fn from(quantity: Quantity) -> Self {
        quantity.value
    }
}

impl Default for Quantity {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_quantity_creation() {
        let qty = Quantity::new(dec!(0.001)).unwrap();
        assert_eq!(qty.as_decimal(), dec!(0.001));
        assert_eq!(qty.precision(), 3);
    }
    
    #[test]
    fn test_negative_quantity_rejected() {
        let result = Quantity::new(dec!(-1.0));
        assert!(matches!(result, Err(QuantityError::NegativeQuantity(_))));
    }
    
    #[test]
    fn test_quantity_arithmetic() {
        let q1 = Quantity::new(dec!(10)).unwrap();
        let q2 = Quantity::new(dec!(5)).unwrap();
        
        assert_eq!(q1.add(q2).unwrap().as_decimal(), dec!(15));
        assert_eq!(q1.subtract(q2).unwrap().as_decimal(), dec!(5));
        assert_eq!(q1.multiply(dec!(2)).unwrap().as_decimal(), dec!(20));
        assert_eq!(q1.divide(dec!(2)).unwrap().as_decimal(), dec!(5));
    }
    
    #[test]
    fn test_lot_size_validation() {
        let qty = Quantity::new(dec!(0.123)).unwrap();
        let lot_size = Quantity::new(dec!(0.01)).unwrap();
        
        // 0.123 is not a multiple of 0.01
        assert!(qty.validate_lot_size(lot_size).is_err());
        
        let rounded = qty.round_to_lot_size(lot_size).unwrap();
        assert_eq!(rounded.as_decimal(), dec!(0.12));
        assert!(rounded.validate_lot_size(lot_size).is_ok());
    }
    
    #[test]
    fn test_quantity_split() {
        let qty = Quantity::new(dec!(10)).unwrap();
        let parts = qty.split(3).unwrap();
        
        assert_eq!(parts.len(), 3);
        let sum: Decimal = parts.iter().map(|q| q.as_decimal()).sum();
        assert_eq!(sum, dec!(10));
    }
    
    #[test]
    fn test_range_validation() {
        let qty = Quantity::new(dec!(5)).unwrap();
        let min = Some(Quantity::new(dec!(1)).unwrap());
        let max = Some(Quantity::new(dec!(10)).unwrap());
        
        assert!(qty.validate_range(min, max).is_ok());
        
        let too_small = Quantity::new(dec!(0.5)).unwrap();
        assert!(too_small.validate_range(min, max).is_err());
        
        let too_large = Quantity::new(dec!(15)).unwrap();
        assert!(too_large.validate_range(min, max).is_err());
    }
}