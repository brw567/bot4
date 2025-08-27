//! # Price Type Comprehensive Tests
//! 
//! 100% coverage for Price value object including edge cases,
//! arithmetic operations, conversions, and validation.

use crate::{Price, PriceError};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use quickcheck_macros::quickcheck;
use proptest::prelude::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod unit_tests {
    use super::*;
    
    #[test]
    fn test_price_creation_valid() {
        let price = Price::new(dec!(100.50)).unwrap();
        assert_eq!(price.as_decimal(), dec!(100.50));
        assert_eq!(price.precision(), 2);
    }
    
    #[test]
    fn test_price_creation_zero() {
        let result = Price::new(dec!(0));
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), PriceError::InvalidValue(_)));
    }
    
    #[test]
    fn test_price_creation_negative() {
        let result = Price::new(dec!(-100));
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), PriceError::InvalidValue(_)));
    }
    
    #[test]
    fn test_price_from_f64_valid() {
        let price = Price::try_from(100.50).unwrap();
        assert_relative_eq!(price.as_f64(), 100.50, epsilon = 0.0001);
    }
    
    #[test]
    fn test_price_from_f64_nan() {
        let result = Price::try_from(f64::NAN);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_price_from_f64_infinity() {
        let result = Price::try_from(f64::INFINITY);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_price_arithmetic_add() {
        let p1 = Price::new(dec!(100)).unwrap();
        let p2 = Price::new(dec!(50)).unwrap();
        let result = p1.add(&p2).unwrap();
        assert_eq!(result.as_decimal(), dec!(150));
    }
    
    #[test]
    fn test_price_arithmetic_subtract() {
        let p1 = Price::new(dec!(100)).unwrap();
        let p2 = Price::new(dec!(50)).unwrap();
        let result = p1.subtract(&p2).unwrap();
        assert_eq!(result.as_decimal(), dec!(50));
    }
    
    #[test]
    fn test_price_arithmetic_subtract_underflow() {
        let p1 = Price::new(dec!(50)).unwrap();
        let p2 = Price::new(dec!(100)).unwrap();
        let result = p1.subtract(&p2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_price_arithmetic_multiply() {
        let price = Price::new(dec!(100)).unwrap();
        let result = price.multiply(dec!(1.5)).unwrap();
        assert_eq!(result.as_decimal(), dec!(150));
    }
    
    #[test]
    fn test_price_arithmetic_divide() {
        let price = Price::new(dec!(100)).unwrap();
        let result = price.divide(dec!(2)).unwrap();
        assert_eq!(result.as_decimal(), dec!(50));
    }
    
    #[test]
    fn test_price_arithmetic_divide_by_zero() {
        let price = Price::new(dec!(100)).unwrap();
        let result = price.divide(dec!(0));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_price_comparisons() {
        let p1 = Price::new(dec!(100)).unwrap();
        let p2 = Price::new(dec!(100)).unwrap();
        let p3 = Price::new(dec!(200)).unwrap();
        
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
        assert!(p1 < p3);
        assert!(p3 > p1);
        assert!(p1 <= p2);
        assert!(p1 >= p2);
    }
    
    #[test]
    fn test_price_display() {
        let price = Price::new(dec!(100.123456)).unwrap();
        assert_eq!(format!("{}", price), "100.123456");
    }
    
    #[test]
    fn test_price_debug() {
        let price = Price::new(dec!(100.50)).unwrap();
        let debug_str = format!("{:?}", price);
        assert!(debug_str.contains("Price"));
        assert!(debug_str.contains("100.50"));
    }
    
    #[test]
    fn test_price_serialization() {
        let price = Price::new(dec!(100.50)).unwrap();
        let json = serde_json::to_string(&price).unwrap();
        let deserialized: Price = serde_json::from_str(&json).unwrap();
        assert_eq!(price, deserialized);
    }
    
    #[test]
    fn test_price_clone() {
        let p1 = Price::new(dec!(100)).unwrap();
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::tests::test_utils::boundary_prices;
    
    #[quickcheck]
    fn prop_price_creation_positive(value: u64) -> bool {
        if value == 0 { return true; }
        let decimal = Decimal::from(value);
        Price::new(decimal).is_ok()
    }
    
    #[quickcheck]
    fn prop_price_add_commutative(a: u64, b: u64) -> bool {
        if a == 0 || b == 0 { return true; }
        let p1 = Price::new(Decimal::from(a)).unwrap();
        let p2 = Price::new(Decimal::from(b)).unwrap();
        p1.add(&p2).unwrap() == p2.add(&p1).unwrap()
    }
    
    #[quickcheck]
    fn prop_price_add_associative(a: u64, b: u64, c: u64) -> bool {
        if a == 0 || b == 0 || c == 0 { return true; }
        if a > 1_000_000 || b > 1_000_000 || c > 1_000_000 { return true; }
        
        let p1 = Price::new(Decimal::from(a)).unwrap();
        let p2 = Price::new(Decimal::from(b)).unwrap();
        let p3 = Price::new(Decimal::from(c)).unwrap();
        
        let left = p1.add(&p2).unwrap().add(&p3).unwrap();
        let right = p1.add(&p2.add(&p3).unwrap()).unwrap();
        left == right
    }
    
    #[quickcheck]
    fn prop_price_multiply_identity(value: u64) -> bool {
        if value == 0 { return true; }
        let price = Price::new(Decimal::from(value)).unwrap();
        let result = price.multiply(dec!(1)).unwrap();
        result == price
    }
    
    #[quickcheck]
    fn prop_price_divide_identity(value: u64) -> bool {
        if value == 0 { return true; }
        let price = Price::new(Decimal::from(value)).unwrap();
        let result = price.divide(dec!(1)).unwrap();
        result == price
    }
    
    #[quickcheck]
    fn prop_price_round_trip_conversion(value: u64) -> bool {
        if value == 0 || value > 1_000_000_000 { return true; }
        let decimal = Decimal::from(value) / dec!(100);
        let price = Price::new(decimal).unwrap();
        let converted = Price::new(price.as_decimal()).unwrap();
        price == converted
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::tests::test_utils::boundary_prices;
    
    #[test]
    fn test_price_boundary_values() {
        for value in boundary_prices() {
            let result = Price::new(value);
            assert!(result.is_ok(), "Failed for boundary value: {}", value);
            
            let price = result.unwrap();
            assert_eq!(price.as_decimal(), value);
            
            // Test arithmetic at boundaries
            let doubled = price.multiply(dec!(2)).unwrap();
            assert_eq!(doubled.as_decimal(), value * dec!(2));
            
            let halved = price.divide(dec!(2)).unwrap();
            assert_eq!(halved.as_decimal(), value / dec!(2));
        }
    }
    
    #[test]
    fn test_price_precision_limits() {
        // Test various precision levels
        let precisions = vec![
            dec!(0.00000001),  // 8 decimals (satoshi)
            dec!(0.0001),      // 4 decimals
            dec!(0.01),        // 2 decimals (cents)
            dec!(1),           // No decimals
        ];
        
        for precision in precisions {
            let base = dec!(12345.67890123);
            let price = Price::new(base).unwrap();
            let rounded = price.round_to_precision(precision.scale());
            assert!(rounded.as_decimal().scale() <= precision.scale());
        }
    }
    
    #[test]
    fn test_price_overflow_protection() {
        let max = Price::new(dec!(999_999_999_999)).unwrap();
        let result = max.multiply(dec!(1000));
        assert!(result.is_err()); // Should fail on overflow
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_price_in_order_context() {
        use crate::{Order, OrderSide, Quantity, TimeInForce};
        
        let price = Price::new(dec!(50000)).unwrap();
        let quantity = Quantity::new(dec!(0.1)).unwrap();
        let order = Order::limit(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            price.clone(),
            quantity,
            TimeInForce::GTC
        );
        
        assert_eq!(order.price, Some(price));
    }
    
    #[test]
    fn test_price_in_market_data_context() {
        use crate::{BookLevel, Quantity};
        
        let price = Price::new(dec!(50000)).unwrap();
        let quantity = Quantity::new(dec!(1)).unwrap();
        let level = BookLevel::new(price.clone(), quantity);
        
        assert_eq!(level.price, price);
    }
}