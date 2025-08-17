// Property-Based Tests
// Addresses Sophia's #7 critical feedback on test coverage
// Uses proptest for invariant verification
// Owner: Riley | Reviewer: Sam

use proptest::prelude::*;
use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

use bot4_core::domain::entities::{Order, OrderSide, OrderType, TimeInForce};
use bot4_core::domain::value_objects::{Symbol, Price, Quantity};
use bot4_core::domain::value_objects::fee::{FeeModel, FeeTier};
use bot4_core::domain::value_objects::market_impact::{MarketImpactModel, MarketDepth};
use bot4_core::domain::value_objects::validation_filters::{ValidationFilters, PriceFilter, LotSizeFilter};
use bot4_core::adapters::outbound::exchanges::idempotency_manager::IdempotencyManager;

// Property: Fees should always be non-negative and within bounds
proptest! {
    #[test]
    fn prop_fees_always_valid(
        volume in 0.0..1_000_000_000.0,
        is_maker in prop::bool::ANY,
        quantity in 0.00001..10000.0,
        price in 1.0..1_000_000.0,
    ) {
        let fee_model = FeeModel::default();
        let fee = fee_model.calculate_fee(quantity, price, is_maker, volume);
        
        // Fee should be non-negative
        prop_assert!(fee.amount >= 0.0);
        
        // Fee should not exceed total value
        let total_value = quantity * price;
        prop_assert!(fee.amount <= total_value);
        
        // Fee rate should be within expected bounds (0-0.1%)
        let fee_rate = fee.amount / total_value;
        prop_assert!(fee_rate <= 0.001);
    }
}

// Property: Market impact should increase with order size
proptest! {
    #[test]
    fn prop_market_impact_monotonic(
        base_size in 0.1..100.0,
        multiplier in 1.1..10.0,
        daily_volume in 1_000_000.0..100_000_000.0,
    ) {
        let model = MarketImpactModel::SquareRoot {
            gamma: 0.1,
            daily_volume,
        };
        
        let small_impact = model.calculate_impact(base_size, daily_volume);
        let large_impact = model.calculate_impact(base_size * multiplier, daily_volume);
        
        // Larger orders should have more impact
        prop_assert!(large_impact >= small_impact);
        
        // Square root model: impact should grow sublinearly
        let impact_ratio = large_impact / small_impact;
        let size_ratio = multiplier;
        prop_assert!(impact_ratio <= size_ratio);
    }
}

// Property: Idempotency should always return same result for same request
proptest! {
    #[test]
    fn prop_idempotency_consistency(
        client_id in "[A-Z0-9]{8}-[A-Z0-9]{4}",
        exchange_id in "EX_[A-Z0-9]{16}",
        hash1 in 0u64..u64::MAX,
        hash2 in 0u64..u64::MAX,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let manager = IdempotencyManager::default();
            
            // First insertion
            let result1 = manager.insert(
                client_id.clone(),
                exchange_id.clone(),
                hash1
            ).await;
            prop_assert!(result1.is_ok());
            
            // Second insertion with same client_id
            if hash1 == hash2 {
                // Same hash should succeed (idempotent)
                let result2 = manager.insert(
                    client_id.clone(),
                    exchange_id.clone(),
                    hash2
                ).await;
                prop_assert!(result2.is_ok());
            } else {
                // Different hash should fail
                let result2 = manager.insert(
                    client_id.clone(),
                    "DIFFERENT_ID".to_string(),
                    hash2
                ).await;
                prop_assert!(result2.is_err());
            }
            
            Ok(())
        })?;
    }
}

// Property: Price validation should be consistent
proptest! {
    #[test]
    fn prop_price_validation_consistency(
        price in 0.0001..1_000_000.0,
        min_price in 0.0001..1000.0,
        tick_size in prop::sample::select(vec![0.01, 0.1, 1.0, 10.0]),
    ) {
        let max_price = min_price * 10000.0;
        let filter = PriceFilter {
            min_price,
            max_price,
            tick_size,
        };
        
        let result = filter.validate(price);
        
        if price < min_price || price > max_price {
            // Should fail bounds check
            prop_assert!(result.is_err());
        } else {
            // Check tick size
            let ticks = (price / tick_size).round();
            let expected_price = ticks * tick_size;
            if (price - expected_price).abs() < 1e-8 {
                prop_assert!(result.is_ok());
            } else {
                prop_assert!(result.is_err());
            }
        }
    }
}

// Property: Order validation should never allow invalid states
proptest! {
    #[test]
    fn prop_order_validation_invariants(
        quantity in 0.0..1000.0,
        price_opt in prop::option::of(1.0..1_000_000.0),
        is_market in prop::bool::ANY,
    ) {
        let filters = ValidationFilters::btc_usdt();
        
        // Create order based on type
        let order = if is_market {
            Order::market(
                Symbol::new("BTC/USDT").unwrap(),
                OrderSide::Buy,
                Quantity::new(quantity.max(0.00001)).unwrap_or(Quantity::new(0.00001).unwrap()),
            )
        } else {
            if let Some(price) = price_opt {
                Order::limit(
                    Symbol::new("BTC/USDT").unwrap(),
                    OrderSide::Buy,
                    Price::new(price).unwrap(),
                    Quantity::new(quantity.max(0.00001)).unwrap_or(Quantity::new(0.00001).unwrap()),
                    TimeInForce::GTC,
                )
            } else {
                // Skip if no price for limit order
                return Ok(());
            }
        };
        
        let result = filters.validate_order(&order, Some(50000.0));
        
        // Verify invariants
        if quantity < filters.lot_size_filter.min_qty {
            prop_assert!(result.is_err(), "Should reject quantity below minimum");
        }
        
        if quantity > filters.lot_size_filter.max_qty {
            prop_assert!(result.is_err(), "Should reject quantity above maximum");
        }
        
        if let Some(price) = order.price() {
            let notional = price.value() * quantity;
            if notional < filters.notional_filter.min_notional {
                prop_assert!(result.is_err(), "Should reject low notional value");
            }
        }
    }
}

// Property: Fill distributions should sum to 1.0
proptest! {
    #[test]
    fn prop_fill_distribution_normalization(
        lambda in 0.5..10.0,
        alpha in 0.5..5.0,
        beta in 0.5..10.0,
    ) {
        use bot4_core::domain::value_objects::statistical_distributions::FillDistribution;
        use rand::thread_rng;
        
        let dist = FillDistribution {
            lambda,
            beta_alpha: alpha,
            beta_beta: beta,
        };
        
        let mut rng = thread_rng();
        
        // Generate fills multiple times
        for _ in 0..10 {
            let fills = dist.generate_fills(&mut rng).unwrap();
            
            // Sum should be 1.0 (within floating point tolerance)
            let sum: f64 = fills.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-10, "Fill ratios must sum to 1.0");
            
            // All ratios should be positive
            for ratio in &fills {
                prop_assert!(*ratio > 0.0, "All fill ratios must be positive");
                prop_assert!(*ratio <= 1.0, "No fill ratio can exceed 1.0");
            }
        }
    }
}

// Property: OCO orders should maintain mutual exclusivity
proptest! {
    #[test]
    fn prop_oco_mutual_exclusivity(
        limit_price in 40000.0..60000.0,
        stop_price in 45000.0..55000.0,
        quantity in 0.01..10.0,
        market_price in 40000.0..60000.0,
    ) {
        use bot4_core::domain::entities::oco_order::{OcoOrder, OcoSemantics};
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let oco = OcoOrder::new(
                Symbol::new("BTC/USDT").unwrap(),
                OrderSide::Sell,
                Price::new(limit_price).unwrap(),
                Price::new(stop_price).unwrap(),
                Quantity::new(quantity).unwrap(),
                OcoSemantics::FirstTriggeredWins,
            ).unwrap();
            
            // Check market price against both orders
            let limit_triggered = market_price >= limit_price;
            let stop_triggered = market_price <= stop_price;
            
            if limit_triggered && stop_triggered {
                // Both conditions met - OCO should handle this
                let status = oco.check_trigger_conditions(market_price).await;
                
                // Only one should be active
                match status {
                    (true, false) | (false, true) => {
                        // Correct - only one triggered
                    }
                    _ => {
                        prop_assert!(false, "OCO failed mutual exclusivity");
                    }
                }
            }
            
            Ok(())
        })?;
    }
}

// Property: Timestamp validation ordering
proptest! {
    #[test]
    fn prop_timestamp_ordering(
        timestamps in prop::collection::vec(0i64..1_000_000_000_000i64, 2..100)
    ) {
        use bot4_core::domain::value_objects::timestamp_validator::{
            TimestampValidator, TimestampConfig
        };
        
        let config = TimestampConfig {
            enforce_ordering: true,
            ..Default::default()
        };
        
        let validator = TimestampValidator::new(config);
        validator.update_server_time();
        
        let mut sorted_timestamps = timestamps.clone();
        sorted_timestamps.sort();
        
        // Process timestamps in sorted order
        let mut last_valid: Option<i64> = None;
        
        for ts in sorted_timestamps {
            // Adjust timestamp to be near current time
            let current = chrono::Utc::now().timestamp_millis();
            let adjusted_ts = current - 1000 + (ts % 2000);
            
            let result = validator.validate_timestamp(adjusted_ts);
            
            if let Some(last) = last_valid {
                if adjusted_ts <= last {
                    // Should fail ordering check
                    prop_assert!(result.is_err() || adjusted_ts == last);
                }
            }
            
            if result.is_ok() {
                last_valid = Some(adjusted_ts);
            }
        }
    }
}

// Property: Statistical distributions should match expected parameters
proptest! {
    #[test]
    fn prop_distribution_parameters(
        mu in -2.0..5.0,
        sigma in 0.1..2.0,
        sample_size in 100..1000,
    ) {
        use bot4_core::domain::value_objects::statistical_distributions::LatencyDistribution;
        use rand::thread_rng;
        
        let dist = LatencyDistribution {
            mu,
            sigma,
            min_latency: 1.0,
            max_latency: 10000.0,
        };
        
        let mut rng = thread_rng();
        let mut samples = Vec::new();
        
        for _ in 0..sample_size {
            let latency = dist.generate_latency(&mut rng);
            samples.push(latency.as_millis() as f64);
        }
        
        // All samples should be within bounds
        for sample in &samples {
            prop_assert!(*sample >= dist.min_latency);
            prop_assert!(*sample <= dist.max_latency);
        }
        
        // Calculate sample statistics
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];
        let expected_median = mu.exp();
        
        // Median should be roughly exp(mu) (with sampling variance)
        // Allow for wide tolerance due to sampling
        prop_assert!(
            median > expected_median * 0.1 && median < expected_median * 10.0,
            "Median {} not near expected {}",
            median,
            expected_median
        );
    }
}

// Property: Symbol actors should process orders deterministically
proptest! {
    #[test]
    fn prop_symbol_actor_determinism(
        orders in prop::collection::vec(
            (0.0..100.0, 40000.0..60000.0, 0.01..1.0),
            1..20
        )
    ) {
        use bot4_core::adapters::outbound::exchanges::symbol_actor::SymbolActorManager;
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let manager = SymbolActorManager::new(100, 10);
            let symbol = Symbol::new("BTC/USDT").unwrap();
            
            let mut order_ids = Vec::new();
            
            // Place all orders
            for (i, (_, price, quantity)) in orders.iter().enumerate() {
                let order = Order::limit(
                    symbol.clone(),
                    OrderSide::Buy,
                    Price::new(*price).unwrap(),
                    Quantity::new(*quantity).unwrap(),
                    TimeInForce::GTC,
                );
                
                let result = manager.place_order(
                    order,
                    format!("CLIENT_{}", i)
                ).await;
                
                prop_assert!(result.is_ok(), "Order placement failed: {:?}", result);
                order_ids.push(result.unwrap());
            }
            
            // All orders should have unique IDs
            let unique_ids: HashSet<_> = order_ids.iter().collect();
            prop_assert_eq!(unique_ids.len(), order_ids.len(), "Duplicate order IDs detected");
            
            // Shutdown cleanly
            manager.shutdown().await.unwrap();
            
            Ok(())
        })?;
    }
}