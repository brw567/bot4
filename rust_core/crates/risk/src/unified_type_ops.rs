// UNIFIED TYPE OPERATIONS - Complete arithmetic implementations
// Team: Sam (Architecture) + Jordan (Performance) + Full Team
// DEEP DIVE: Type-safe operations prevent mixing incompatible values
// References:
// - "Type Systems for Systems Software" - Crary (2003)
// - "Practical Type Theory for Rust" - Jung et al. (2018)

use crate::unified_types::{Price, Quantity, Percentage};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::ops::{Add, Sub, Mul, Div};

// ============================================================================
// PRICE OPERATIONS - Complete implementation
// ============================================================================

impl Add for Price {
    type Output = Price;
    
    #[inline(always)]
    fn add(self, other: Price) -> Price {
        Price::new(self.inner() + other.inner())
    }
}

impl Sub for Price {
    type Output = Price;
    
    #[inline(always)]
    fn sub(self, other: Price) -> Price {
        Price::new(self.inner() - other.inner())
    }
}

impl Mul<Decimal> for Price {
    type Output = Price;
    
    #[inline(always)]
    fn mul(self, scalar: Decimal) -> Price {
        Price::new(self.inner() * scalar)
    }
}

impl Mul<Quantity> for Price {
    type Output = Decimal;  // Price * Quantity = Value (in currency)
    
    #[inline(always)]
    fn mul(self, quantity: Quantity) -> Decimal {
        self.inner() * quantity.inner()
    }
}

impl Div<Decimal> for Price {
    type Output = Price;
    
    #[inline(always)]
    fn div(self, scalar: Decimal) -> Price {
        if scalar == Decimal::ZERO {
            Price::ZERO
        } else {
            Price::new(self.inner() / scalar)
        }
    }
}

impl Div for Price {
    type Output = Decimal;  // Price / Price = Ratio
    
    #[inline(always)]
    fn div(self, other: Price) -> Decimal {
        if other.inner() == Decimal::ZERO {
            Decimal::ZERO
        } else {
            self.inner() / other.inner()
        }
    }
}

// Comparison operators
impl PartialEq<Decimal> for Price {
    fn eq(&self, other: &Decimal) -> bool {
        self.inner() == *other
    }
}

impl PartialOrd<Decimal> for Price {
    fn partial_cmp(&self, other: &Decimal) -> Option<std::cmp::Ordering> {
        self.inner().partial_cmp(other)
    }
}

// ============================================================================
// QUANTITY OPERATIONS - Complete implementation
// ============================================================================

impl Add for Quantity {
    type Output = Quantity;
    
    #[inline(always)]
    fn add(self, other: Quantity) -> Quantity {
        Quantity::new(self.inner() + other.inner())
    }
}

impl Sub for Quantity {
    type Output = Quantity;
    
    #[inline(always)]
    fn sub(self, other: Quantity) -> Quantity {
        // Quantities are always positive, so max with zero
        let result = self.inner() - other.inner();
        Quantity::new(result.max(Decimal::ZERO))
    }
}

impl Mul<Decimal> for Quantity {
    type Output = Quantity;
    
    #[inline(always)]
    fn mul(self, scalar: Decimal) -> Quantity {
        Quantity::new(self.inner() * scalar)
    }
}

impl Div<Decimal> for Quantity {
    type Output = Quantity;
    
    #[inline(always)]
    fn div(self, scalar: Decimal) -> Quantity {
        if scalar == Decimal::ZERO {
            Quantity::ZERO
        } else {
            Quantity::new(self.inner() / scalar)
        }
    }
}

impl Div for Quantity {
    type Output = Decimal;  // Quantity / Quantity = Ratio
    
    #[inline(always)]
    fn div(self, other: Quantity) -> Decimal {
        if other.inner() == Decimal::ZERO {
            Decimal::ZERO
        } else {
            self.inner() / other.inner()
        }
    }
}

// Comparison with Decimal
impl PartialEq<Decimal> for Quantity {
    fn eq(&self, other: &Decimal) -> bool {
        self.inner() == *other
    }
}

impl PartialOrd<Decimal> for Quantity {
    fn partial_cmp(&self, other: &Decimal) -> Option<std::cmp::Ordering> {
        self.inner().partial_cmp(other)
    }
}

// ============================================================================
// PERCENTAGE OPERATIONS - Complete implementation
// ============================================================================

impl Add for Percentage {
    type Output = Percentage;
    
    #[inline(always)]
    fn add(self, other: Percentage) -> Percentage {
        Percentage::new(self.value() + other.value())
    }
}

impl Sub for Percentage {
    type Output = Percentage;
    
    #[inline(always)]
    fn sub(self, other: Percentage) -> Percentage {
        Percentage::new(self.value() - other.value())
    }
}

impl Mul<f64> for Percentage {
    type Output = Percentage;
    
    #[inline(always)]
    fn mul(self, scalar: f64) -> Percentage {
        Percentage::new(self.value() * scalar)
    }
}

impl Div<f64> for Percentage {
    type Output = Percentage;
    
    #[inline(always)]
    fn div(self, scalar: f64) -> Percentage {
        if scalar == 0.0 {
            Percentage::ZERO
        } else {
            Percentage::new(self.value() / scalar)
        }
    }
}

// Apply percentage to Price
impl Mul<Percentage> for Price {
    type Output = Price;
    
    #[inline(always)]
    fn mul(self, pct: Percentage) -> Price {
        Price::new(self.inner() * Decimal::from_f64(pct.value()).unwrap_or(Decimal::ZERO))
    }
}

// Apply percentage to Quantity
impl Mul<Percentage> for Quantity {
    type Output = Quantity;
    
    #[inline(always)]
    fn mul(self, pct: Percentage) -> Quantity {
        Quantity::new(self.inner() * Decimal::from_f64(pct.value()).unwrap_or(Decimal::ZERO))
    }
}

// ============================================================================
// CONVERSION HELPERS - For interoperability
// ============================================================================

/// Convert Decimal to Price safely
pub trait ToPriceExt {
    fn to_price(self) -> Price;
}

impl ToPriceExt for Decimal {
    #[inline(always)]
    fn to_price(self) -> Price {
        Price::new(self)
    }
}

impl ToPriceExt for f64 {
    #[inline(always)]
    fn to_price(self) -> Price {
        Price::from_f64(self)
    }
}

/// Convert Decimal to Quantity safely
pub trait ToQuantityExt {
    fn to_quantity(self) -> Quantity;
}

impl ToQuantityExt for Decimal {
    #[inline(always)]
    fn to_quantity(self) -> Quantity {
        Quantity::new(self)
    }
}

impl ToQuantityExt for f64 {
    #[inline(always)]
    fn to_quantity(self) -> Quantity {
        Quantity::from_f64(self)
    }
}

/// Convert to Percentage safely
pub trait ToPercentageExt {
    fn to_percentage(self) -> Percentage;
}

impl ToPercentageExt for f64 {
    #[inline(always)]
    fn to_percentage(self) -> Percentage {
        Percentage::new(self)
    }
}

impl ToPercentageExt for Decimal {
    #[inline(always)]
    fn to_percentage(self) -> Percentage {
        Percentage::new(self.to_f64().unwrap_or(0.0))
    }
}

// ============================================================================
// DISPLAY IMPLEMENTATIONS - Human readable output
// ============================================================================

use std::fmt;

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.inner())
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner())
    }
}

impl fmt::Display for Percentage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.value() * 100.0)
    }
}

// ============================================================================
// DEFAULT IMPLEMENTATIONS
// ============================================================================

impl Default for Price {
    fn default() -> Self {
        Price::ZERO
    }
}

impl Default for Quantity {
    fn default() -> Self {
        Quantity::ZERO
    }
}

impl Default for Percentage {
    fn default() -> Self {
        Percentage::ZERO
    }
}

// ============================================================================
// TESTS - DEEP DIVE validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_price_operations() {
        let p1 = Price::from_f64(100.0);
        let p2 = Price::from_f64(50.0);
        
        // Addition
        let sum = p1 + p2;
        assert_eq!(sum.to_f64(), 150.0);
        
        // Subtraction
        let diff = p1 - p2;
        assert_eq!(diff.to_f64(), 50.0);
        
        // Multiplication by scalar
        let scaled = p1 * dec!(2);
        assert_eq!(scaled.to_f64(), 200.0);
        
        // Division
        let ratio = p1 / p2;
        assert_eq!(ratio, dec!(2));
    }
    
    #[test]
    fn test_quantity_operations() {
        let q1 = Quantity::from_f64(10.0);
        let q2 = Quantity::from_f64(5.0);
        
        // Addition
        let sum = q1 + q2;
        assert_eq!(sum.to_f64(), 15.0);
        
        // Subtraction (never negative)
        let diff = q2 - q1;
        assert_eq!(diff.to_f64(), 0.0);  // Clamped to zero
        
        // Multiplication
        let scaled = q1 * dec!(3);
        assert_eq!(scaled.to_f64(), 30.0);
    }
    
    #[test]
    fn test_price_quantity_multiplication() {
        let price = Price::from_f64(50000.0);  // $50k BTC
        let quantity = Quantity::from_f64(0.5);  // 0.5 BTC
        
        let value = price * quantity;
        assert_eq!(value, dec!(25000));  // $25k total value
    }
    
    #[test]
    fn test_percentage_application() {
        let price = Price::from_f64(100.0);
        let pct = Percentage::new(0.1);  // 10%
        
        let adjusted = price * pct;
        assert_eq!(adjusted.to_f64(), 10.0);
        
        // Test basis points
        let bps = Percentage::from_basis_points(50.0);  // 50 bps = 0.5%
        assert_eq!(bps.value(), 0.005);
    }
    
    #[test]
    fn test_display_formatting() {
        let price = Price::from_f64(12345.67);
        assert_eq!(format!("{}", price), "$12345.67");
        
        let qty = Quantity::from_f64(10.5);
        assert_eq!(format!("{}", qty), "10.5");
        
        let pct = Percentage::new(0.1525);
        assert_eq!(format!("{}", pct), "15.25%");
    }
}