// Domain Value Object: Quantity
// Immutable, no identity, represents a trading quantity
// Owner: Sam | Reviewer: Morgan

use std::fmt;
use anyhow::{Result, bail};

/// Quantity value object - immutable representation of an amount
/// 
/// # Invariants
/// - Quantity must be non-negative (>= 0)
/// - Quantity must be finite
/// - Zero is a valid quantity (for comparisons)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Quantity(f64);

impl Quantity {
    /// Create a new Quantity with validation
    /// 
    /// # Arguments
    /// * `value` - The quantity value
    /// 
    /// # Returns
    /// * `Ok(Quantity)` if valid
    /// * `Err` if quantity is invalid (<0, infinite, or NaN)
    pub fn new(value: f64) -> Result<Self> {
        if value < 0.0 {
            bail!("Quantity cannot be negative, got: {}", value);
        }
        if !value.is_finite() {
            bail!("Quantity must be finite, got: {}", value);
        }
        Ok(Quantity(value))
    }
    
    /// Create a zero quantity
    pub fn zero() -> Self {
        Quantity(0.0)
    }
    
    /// Get the underlying value
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.0
    }
    
    /// Check if quantity is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
    
    /// Check if quantity is positive (> 0)
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.0 > 0.0
    }
    
    /// Add two quantities
    pub fn add(&self, other: &Quantity) -> Result<Quantity> {
        Quantity::new(self.0 + other.0)
    }
    
    /// Subtract a quantity (result must be non-negative)
    pub fn subtract(&self, other: &Quantity) -> Result<Quantity> {
        let result = self.0 - other.0;
        if result < 0.0 {
            bail!("Subtraction would result in negative quantity: {} - {} = {}", 
                  self.0, other.0, result);
        }
        Quantity::new(result)
    }
    
    /// Multiply by a scalar
    pub fn multiply(&self, scalar: f64) -> Result<Quantity> {
        if scalar < 0.0 {
            bail!("Cannot multiply quantity by negative scalar: {}", scalar);
        }
        Quantity::new(self.0 * scalar)
    }
    
    /// Divide by a scalar
    pub fn divide(&self, scalar: f64) -> Result<Quantity> {
        if scalar <= 0.0 {
            bail!("Cannot divide quantity by non-positive scalar: {}", scalar);
        }
        Quantity::new(self.0 / scalar)
    }
    
    /// Split into n equal parts
    pub fn split(&self, parts: usize) -> Result<Vec<Quantity>> {
        if parts == 0 {
            bail!("Cannot split into 0 parts");
        }
        
        let part_size = self.0 / parts as f64;
        let mut result = Vec::with_capacity(parts);
        
        for _ in 0..parts {
            result.push(Quantity::new(part_size)?);
        }
        
        Ok(result)
    }
    
    /// Round to a specific number of decimal places
    #[inline]
    pub fn round_to(&self, decimals: u32) -> Quantity {
        let multiplier = 10_f64.powi(decimals as i32);
        Quantity((self.0 * multiplier).round() / multiplier)
    }
    
    /// Get the minimum of two quantities
    #[inline]
    pub fn min(&self, other: &Quantity) -> Quantity {
        if self.0 <= other.0 {
            *self
        } else {
            *other
        }
    }
    
    /// Get the maximum of two quantities
    #[inline]
    pub fn max(&self, other: &Quantity) -> Quantity {
        if self.0 >= other.0 {
            *self
        } else {
            *other
        }
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.0)
    }
}

// Implement PartialEq with f64 for convenience
impl PartialEq<f64> for Quantity {
    fn eq(&self, other: &f64) -> bool {
        (self.0 - other).abs() < f64::EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn should_create_valid_quantity() {
        let qty = Quantity::new(10.0).unwrap();
        assert_eq!(qty.value(), 10.0);
    }
    
    #[test]
    fn should_create_zero_quantity() {
        let qty = Quantity::zero();
        assert_eq!(qty.value(), 0.0);
        assert!(qty.is_zero());
    }
    
    #[test]
    fn should_reject_negative_quantity() {
        let result = Quantity::new(-1.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_reject_infinite_quantity() {
        let result = Quantity::new(f64::INFINITY);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_add_quantities() {
        let q1 = Quantity::new(10.0).unwrap();
        let q2 = Quantity::new(5.0).unwrap();
        let sum = q1.add(&q2).unwrap();
        assert_eq!(sum.value(), 15.0);
    }
    
    #[test]
    fn should_subtract_quantities() {
        let q1 = Quantity::new(10.0).unwrap();
        let q2 = Quantity::new(3.0).unwrap();
        let diff = q1.subtract(&q2).unwrap();
        assert_eq!(diff.value(), 7.0);
    }
    
    #[test]
    fn should_reject_negative_subtraction() {
        let q1 = Quantity::new(5.0).unwrap();
        let q2 = Quantity::new(10.0).unwrap();
        let result = q1.subtract(&q2);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_multiply_by_scalar() {
        let qty = Quantity::new(10.0).unwrap();
        let result = qty.multiply(2.5).unwrap();
        assert_eq!(result.value(), 25.0);
    }
    
    #[test]
    fn should_reject_negative_multiplication() {
        let qty = Quantity::new(10.0).unwrap();
        let result = qty.multiply(-2.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_divide_by_scalar() {
        let qty = Quantity::new(10.0).unwrap();
        let result = qty.divide(2.0).unwrap();
        assert_eq!(result.value(), 5.0);
    }
    
    #[test]
    fn should_reject_division_by_zero() {
        let qty = Quantity::new(10.0).unwrap();
        let result = qty.divide(0.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn should_split_into_parts() {
        let qty = Quantity::new(10.0).unwrap();
        let parts = qty.split(4).unwrap();
        assert_eq!(parts.len(), 4);
        for part in parts {
            assert_eq!(part.value(), 2.5);
        }
    }
    
    #[test]
    fn should_round_to_decimals() {
        let qty = Quantity::new(10.123456789).unwrap();
        let rounded = qty.round_to(2);
        assert_eq!(rounded.value(), 10.12);
    }
    
    #[test]
    fn should_get_min_quantity() {
        let q1 = Quantity::new(10.0).unwrap();
        let q2 = Quantity::new(5.0).unwrap();
        let min = q1.min(&q2);
        assert_eq!(min.value(), 5.0);
    }
    
    #[test]
    fn should_get_max_quantity() {
        let q1 = Quantity::new(10.0).unwrap();
        let q2 = Quantity::new(5.0).unwrap();
        let max = q1.max(&q2);
        assert_eq!(max.value(), 10.0);
    }
}