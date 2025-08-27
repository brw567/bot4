//! # Parallel Validation System Tests
//! 
//! Tests the GitHub Scientist-inspired parallel validation system
//! that ensures safe migration from legacy to canonical types.

use crate::{
    Order, Price, Quantity, OrderSide, TimeInForce,
};
#[cfg(feature = "parallel_validation")]
use crate::parallel_validation::{
    ParallelValidator, ValidationResult, Discrepancy, DiscrepancySeverity,
    ParallelValidatable,
};
use rust_decimal_macros::dec;
use std::time::Duration;

#[cfg(all(test, feature = "parallel_validation"))]
mod validator_tests {
    use super::*;
    
    #[test]
    fn test_validator_default_configuration() {
        let validator = ParallelValidator::default();
        assert_eq!(validator.fail_on_discrepancy, false);
        assert_eq!(validator.log_performance, true);
        assert_eq!(validator.sampling_rate, 1.0);
    }
    
    #[test]
    fn test_validator_custom_configuration() {
        let validator = ParallelValidator::new(true, false, 0.5);
        assert_eq!(validator.fail_on_discrepancy, true);
        assert_eq!(validator.log_performance, false);
        assert_eq!(validator.sampling_rate, 0.5);
    }
    
    #[test]
    fn test_validator_sampling_rate_clamping() {
        let validator_high = ParallelValidator::new(false, false, 1.5);
        assert_eq!(validator_high.sampling_rate, 1.0);
        
        let validator_low = ParallelValidator::new(false, false, -0.5);
        assert_eq!(validator_low.sampling_rate, 0.0);
    }
    
    #[test]
    fn test_validation_matching_results() {
        let validator = ParallelValidator::default();
        
        let canonical_fn = || Ok(Price::new(dec!(100)).unwrap());
        let legacy_fn = || Ok(Price::new(dec!(100)).unwrap());
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "price_creation"
        ).unwrap();
        
        assert!(result.results_match);
        assert!(result.discrepancies.is_empty());
        assert_eq!(result.canonical_result.as_decimal(), dec!(100));
        assert_eq!(result.legacy_result.unwrap().as_decimal(), dec!(100));
    }
    
    #[test]
    fn test_validation_with_discrepancy() {
        let validator = ParallelValidator::default();
        
        let canonical_fn = || Ok(Price::new(dec!(100.00)).unwrap());
        let legacy_fn = || Ok(Price::new(dec!(100.01)).unwrap());
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "price_with_difference"
        ).unwrap();
        
        assert!(!result.results_match);
        assert!(!result.discrepancies.is_empty());
        assert_eq!(result.discrepancies[0].field, "value");
        assert_eq!(result.discrepancies[0].severity, DiscrepancySeverity::Warning);
    }
    
    #[test]
    fn test_validation_legacy_failure() {
        let validator = ParallelValidator::default();
        
        let canonical_fn = || Ok(Price::new(dec!(100)).unwrap());
        let legacy_fn = || Err::<Price, String>("Legacy system error".to_string());
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "legacy_failure_test"
        ).unwrap();
        
        assert!(!result.results_match);
        assert!(result.legacy_result.is_none());
        assert_eq!(result.discrepancies.len(), 1);
        assert_eq!(result.discrepancies[0].field, "execution");
        assert_eq!(result.discrepancies[0].severity, DiscrepancySeverity::Warning);
    }
    
    #[test]
    fn test_validation_performance_tracking() {
        let validator = ParallelValidator::new(false, true, 1.0);
        
        let canonical_fn = || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(Price::new(dec!(100)).unwrap())
        };
        
        let legacy_fn = || {
            std::thread::sleep(Duration::from_millis(20));
            Ok(Price::new(dec!(100)).unwrap())
        };
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "performance_test"
        ).unwrap();
        
        assert!(result.performance.canonical_duration < result.performance.legacy_duration.unwrap());
        assert!(result.performance.improvement_percent > 0.0);
    }
    
    #[test]
    fn test_validation_skip_on_zero_sampling() {
        let validator = ParallelValidator::new(false, false, 0.0);
        
        let canonical_fn = || Ok(Price::new(dec!(100)).unwrap());
        let legacy_fn = || panic!("Should not be called with 0% sampling");
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "zero_sampling_test"
        ).unwrap();
        
        assert!(result.results_match);
        assert!(result.legacy_result.is_none());
        assert!(result.discrepancies.is_empty());
    }
}

#[cfg(all(test, feature = "parallel_validation"))]
mod parallel_validatable_tests {
    use super::*;
    
    #[test]
    fn test_order_comparison_identical() {
        let order1 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        let order2 = order1.clone();
        
        let discrepancies = order1.compare_with(&order2);
        assert!(discrepancies.is_empty());
    }
    
    #[test]
    fn test_order_comparison_different_symbol() {
        let order1 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        let order2 = Order::market("ETH/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        
        let discrepancies = order1.compare_with(&order2);
        assert_eq!(discrepancies.len(), 1);
        assert_eq!(discrepancies[0].field, "symbol");
        assert_eq!(discrepancies[0].severity, DiscrepancySeverity::Critical);
    }
    
    #[test]
    fn test_order_comparison_different_side() {
        let order1 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        let order2 = Order::market("BTC/USDT".to_string(), OrderSide::Sell, Quantity::new(dec!(1)).unwrap());
        
        let discrepancies = order1.compare_with(&order2);
        assert!(discrepancies.iter().any(|d| d.field == "side" && d.severity == DiscrepancySeverity::Critical));
    }
    
    #[test]
    fn test_order_comparison_different_quantity() {
        let order1 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        let order2 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(2)).unwrap());
        
        let discrepancies = order1.compare_with(&order2);
        assert!(discrepancies.iter().any(|d| d.field == "quantity" && d.severity == DiscrepancySeverity::Critical));
    }
    
    #[test]
    fn test_order_comparison_different_metadata() {
        let mut order1 = Order::market("BTC/USDT".to_string(), OrderSide::Buy, Quantity::new(dec!(1)).unwrap());
        let mut order2 = order1.clone();
        
        order1.strategy_id = Some("strategy_v1".to_string());
        order2.strategy_id = Some("strategy_v2".to_string());
        
        let discrepancies = order1.compare_with(&order2);
        assert_eq!(discrepancies.len(), 1);
        assert_eq!(discrepancies[0].field, "strategy_id");
        assert_eq!(discrepancies[0].severity, DiscrepancySeverity::Info);
    }
    
    #[test]
    fn test_price_comparison_identical() {
        let price1 = Price::new(dec!(100.50)).unwrap();
        let price2 = Price::new(dec!(100.50)).unwrap();
        
        let discrepancies = price1.compare_with(&price2);
        assert!(discrepancies.is_empty());
    }
    
    #[test]
    fn test_price_comparison_minor_difference() {
        let price1 = Price::new(dec!(100.00)).unwrap();
        let price2 = Price::new(dec!(100.001)).unwrap(); // 0.001% difference
        
        let discrepancies = price1.compare_with(&price2);
        assert_eq!(discrepancies.len(), 1);
        assert_eq!(discrepancies[0].field, "value");
        assert_eq!(discrepancies[0].severity, DiscrepancySeverity::Warning);
    }
    
    #[test]
    fn test_price_comparison_major_difference() {
        let price1 = Price::new(dec!(100)).unwrap();
        let price2 = Price::new(dec!(101)).unwrap(); // 1% difference
        
        let discrepancies = price1.compare_with(&price2);
        assert_eq!(discrepancies.len(), 1);
        assert_eq!(discrepancies[0].field, "value");
        assert_eq!(discrepancies[0].severity, DiscrepancySeverity::Critical);
    }
    
    #[test]
    fn test_quantity_comparison() {
        let qty1 = Quantity::new(dec!(10)).unwrap();
        let qty2 = Quantity::new(dec!(10)).unwrap();
        let qty3 = Quantity::new(dec!(11)).unwrap();
        
        let discrepancies1 = qty1.compare_with(&qty2);
        assert!(discrepancies1.is_empty());
        
        let discrepancies2 = qty1.compare_with(&qty3);
        assert_eq!(discrepancies2.len(), 1);
        assert_eq!(discrepancies2[0].field, "value");
        assert_eq!(discrepancies2[0].severity, DiscrepancySeverity::Critical);
    }
}

#[cfg(all(test, feature = "parallel_validation"))]
mod integration_validation_tests {
    use super::*;
    
    #[test]
    fn test_full_order_validation_workflow() {
        let validator = ParallelValidator::new(false, true, 1.0);
        
        // Simulate canonical implementation
        let create_order_canonical = || {
            Ok(Order::limit(
                "BTC/USDT".to_string(),
                OrderSide::Buy,
                Price::new(dec!(50000)).unwrap(),
                Quantity::new(dec!(0.1)).unwrap(),
                TimeInForce::GTC
            ))
        };
        
        // Simulate legacy implementation with slight differences
        let create_order_legacy = || {
            let mut order = Order::limit(
                "BTC/USDT".to_string(),
                OrderSide::Buy,
                Price::new(dec!(50000)).unwrap(),
                Quantity::new(dec!(0.1)).unwrap(),
                TimeInForce::IOC  // Different time in force
            );
            order.strategy_id = Some("legacy_strategy".to_string());
            Ok(order)
        };
        
        let result = validator.validate(
            create_order_canonical,
            create_order_legacy,
            "order_creation_validation"
        ).unwrap();
        
        assert!(!result.results_match);
        
        // Should have discrepancies for time_in_force and strategy_id
        assert!(result.discrepancies.iter().any(|d| d.field == "time_in_force"));
        assert!(result.discrepancies.iter().any(|d| d.field == "strategy_id"));
        
        // time_in_force is Warning level, strategy_id is Info level
        let tif_discrepancy = result.discrepancies.iter()
            .find(|d| d.field == "time_in_force").unwrap();
        assert_eq!(tif_discrepancy.severity, DiscrepancySeverity::Warning);
        
        let strategy_discrepancy = result.discrepancies.iter()
            .find(|d| d.field == "strategy_id").unwrap();
        assert_eq!(strategy_discrepancy.severity, DiscrepancySeverity::Info);
    }
    
    #[test]
    fn test_validation_with_fail_on_discrepancy() {
        let validator = ParallelValidator::new(true, false, 1.0);
        
        let canonical_fn = || Ok(Price::new(dec!(100)).unwrap());
        let legacy_fn = || Ok(Price::new(dec!(110)).unwrap()); // 10% difference - critical
        
        let result = validator.validate(
            canonical_fn,
            legacy_fn,
            "fail_on_discrepancy_test"
        );
        
        // Should still return Ok with canonical result, but log the failure
        assert!(result.is_ok());
        let validation = result.unwrap();
        assert!(!validation.results_match);
        assert_eq!(validation.canonical_result.as_decimal(), dec!(100));
    }
}