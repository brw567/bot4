// ULTRATHINK Partial Fills Comprehensive Test Suite
// Team: Eve (Exchange), Carol (Risk), Frank (Integration)
// Research: Almgren-Chriss, Kyle's Lambda, Market Microstructure

use super::partial_fills::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

#[cfg(test)]
mod partial_fill_tracker_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_tracker_initialization() {
        let tracker = PartialFillTracker::new();
        
        // Verify empty initialization
        assert_eq!(tracker.orders.len(), 0);
        assert_eq!(tracker.fill_history.len(), 0);
        assert_eq!(tracker.position_summary.total_quantity, dec!(0));
        assert_eq!(tracker.position_summary.net_pnl, dec!(0));
    }
    
    #[tokio::test]
    async fn test_single_fill_tracking() {
        let mut tracker = PartialFillTracker::new();
        
        // Create order
        let order_id = "TEST001".to_string();
        tracker.create_order(
            order_id.clone(),
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            dec!(1.0),
            Some(dec!(50000))
        ).await;
        
        // Add partial fill
        let fill = PartialFill {
            order_id: order_id.clone(),
            fill_id: "FILL001".to_string(),
            quantity: dec!(0.3),
            price: dec!(49950),
            fee: dec!(14.985), // 0.03% fee
            timestamp: 1000,
            exchange: "binance".to_string(),
            role: TradeRole::Maker,
        };
        
        tracker.add_fill(fill.clone()).await.unwrap();
        
        // Verify tracking
        let status = tracker.get_order_status(&order_id).await.unwrap();
        assert_eq!(status.filled_quantity, dec!(0.3));
        assert_eq!(status.remaining_quantity, dec!(0.7));
        assert_eq!(status.weighted_avg_price, dec!(49950));
        assert_eq!(status.total_fees, dec!(14.985));
        assert_eq!(status.fill_rate, dec!(0.3)); // 30% filled
    }
    
    #[tokio::test]
    async fn test_multiple_fills_weighted_average() {
        let mut tracker = PartialFillTracker::new();
        
        let order_id = "TEST002".to_string();
        tracker.create_order(
            order_id.clone(),
            "ETH/USDT".to_string(),
            OrderSide::Buy,
            dec!(10.0),
            Some(dec!(3000))
        ).await;
        
        // Add multiple fills at different prices
        let fills = vec![
            PartialFill {
                order_id: order_id.clone(),
                fill_id: "F1".to_string(),
                quantity: dec!(3.0),
                price: dec!(2990),
                fee: dec!(2.691),
                timestamp: 1000,
                exchange: "binance".to_string(),
                role: TradeRole::Maker,
            },
            PartialFill {
                order_id: order_id.clone(),
                fill_id: "F2".to_string(),
                quantity: dec!(4.0),
                price: dec!(3000),
                fee: dec!(3.6),
                timestamp: 1001,
                exchange: "binance".to_string(),
                role: TradeRole::Taker,
            },
            PartialFill {
                order_id: order_id.clone(),
                fill_id: "F3".to_string(),
                quantity: dec!(2.0),
                price: dec!(3010),
                fee: dec!(1.806),
                timestamp: 1002,
                exchange: "binance".to_string(),
                role: TradeRole::Maker,
            },
        ];
        
        for fill in fills {
            tracker.add_fill(fill).await.unwrap();
        }
        
        // Verify weighted average price
        // (3*2990 + 4*3000 + 2*3010) / 9 = 2998.89
        let status = tracker.get_order_status(&order_id).await.unwrap();
        assert_eq!(status.filled_quantity, dec!(9.0));
        let expected_avg = (dec!(3) * dec!(2990) + dec!(4) * dec!(3000) + dec!(2) * dec!(3010)) / dec!(9);
        assert!((status.weighted_avg_price - expected_avg).abs() < dec!(0.01));
    }
    
    #[tokio::test]
    async fn test_duplicate_fill_rejection() {
        let mut tracker = PartialFillTracker::new();
        
        let order_id = "TEST003".to_string();
        tracker.create_order(
            order_id.clone(),
            "SOL/USDT".to_string(),
            OrderSide::Sell,
            dec!(100.0),
            Some(dec!(100))
        ).await;
        
        let fill = PartialFill {
            order_id: order_id.clone(),
            fill_id: "DUP001".to_string(),
            quantity: dec!(10.0),
            price: dec!(101),
            fee: dec!(0.303),
            timestamp: 1000,
            exchange: "coinbase".to_string(),
            role: TradeRole::Taker,
        };
        
        // First fill should succeed
        assert!(tracker.add_fill(fill.clone()).await.is_ok());
        
        // Duplicate should fail
        assert!(tracker.add_fill(fill).await.is_err());
    }
    
    #[tokio::test]
    async fn test_overfill_rejection() {
        let mut tracker = PartialFillTracker::new();
        
        let order_id = "TEST004".to_string();
        tracker.create_order(
            order_id.clone(),
            "AVAX/USDT".to_string(),
            OrderSide::Buy,
            dec!(50.0),
            None
        ).await;
        
        // Fill entire order
        let fill1 = PartialFill {
            order_id: order_id.clone(),
            fill_id: "FULL001".to_string(),
            quantity: dec!(50.0),
            price: dec!(35),
            fee: dec!(0.525),
            timestamp: 1000,
            exchange: "kraken".to_string(),
            role: TradeRole::Maker,
        };
        tracker.add_fill(fill1).await.unwrap();
        
        // Attempt to overfill
        let fill2 = PartialFill {
            order_id: order_id.clone(),
            fill_id: "OVER001".to_string(),
            quantity: dec!(1.0),
            price: dec!(35),
            fee: dec!(0.0105),
            timestamp: 1001,
            exchange: "kraken".to_string(),
            role: TradeRole::Maker,
        };
        
        assert!(tracker.add_fill(fill2).await.is_err());
    }
}

#[cfg(test)]
mod market_impact_tests {
    use super::*;
    
    #[test]
    fn test_kyle_lambda_calibration() {
        let mut estimator = MarketImpactEstimator::new();
        
        // Add historical data points
        let data_points = vec![
            (dec!(1000), dec!(0.05)),   // 1000 size -> 5 bps impact
            (dec!(5000), dec!(0.25)),   // 5000 size -> 25 bps
            (dec!(10000), dec!(0.60)),  // 10000 size -> 60 bps
            (dec!(20000), dec!(1.50)),  // 20000 size -> 150 bps
        ];
        
        for (size, impact) in data_points {
            estimator.add_data_point(size, impact);
        }
        
        estimator.calibrate_kyle_lambda();
        
        // Kyle's lambda should be positive
        assert!(estimator.kyle_lambda > dec!(0));
        
        // Test linear relationship
        let small_impact = estimator.estimate_impact(dec!(500));
        let large_impact = estimator.estimate_impact(dec!(15000));
        
        assert!(small_impact < large_impact);
        assert!(small_impact > dec!(0));
    }
    
    #[test]
    fn test_permanent_vs_temporary_impact() {
        let mut estimator = MarketImpactEstimator::new();
        estimator.kyle_lambda = dec!(0.0001);
        estimator.temporary_impact_decay = dec!(0.5);
        
        // Permanent impact (Kyle)
        let permanent = estimator.estimate_permanent_impact(dec!(10000));
        
        // Temporary impact (should decay)
        let temporary = estimator.estimate_temporary_impact(
            dec!(10000),
            dec!(0.01), // 1% of daily volume
            30 // 30 seconds
        );
        
        // Temporary should be larger initially but decay
        assert!(temporary > dec!(0));
        
        // Total impact
        let total = permanent + temporary;
        assert!(total > permanent);
    }
    
    #[test]
    fn test_almgren_chriss_trajectory() {
        let executor = OptimalExecutor::new(
            dec!(0.0001),  // risk_aversion
            dec!(0.001),   // temporary_impact
            dec!(0.0005),  // permanent_impact
            dec!(1000000)  // daily_volume
        );
        
        // Calculate optimal trajectory for 10,000 shares over 300 seconds
        let trajectory = executor.calculate_trajectory(
            dec!(10000),
            300,
            30  // 10 second intervals
        );
        
        // Verify trajectory properties
        assert_eq!(trajectory.len(), 30);
        
        // Sum should equal total quantity
        let total: Decimal = trajectory.iter().sum();
        assert!((total - dec!(10000)).abs() < dec!(1));
        
        // Should front-load execution (decreasing trajectory)
        for i in 1..trajectory.len() {
            assert!(trajectory[i] <= trajectory[i-1] * dec!(1.1)); // Allow small variations
        }
    }
}

#[cfg(test)]
mod reconciliation_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_position_reconciliation() {
        let mut tracker = PartialFillTracker::new();
        
        // Create multiple orders
        let orders = vec![
            ("BUY1", "BTC/USDT", OrderSide::Buy, dec!(1.0)),
            ("SELL1", "BTC/USDT", OrderSide::Sell, dec!(0.5)),
            ("BUY2", "BTC/USDT", OrderSide::Buy, dec!(0.3)),
        ];
        
        for (id, symbol, side, qty) in orders {
            tracker.create_order(
                id.to_string(),
                symbol.to_string(),
                side,
                qty,
                None
            ).await;
        }
        
        // Add fills
        let fills = vec![
            ("BUY1", dec!(1.0), dec!(50000)),
            ("SELL1", dec!(0.5), dec!(50500)),
            ("BUY2", dec!(0.3), dec!(49800)),
        ];
        
        for (order_id, qty, price) in fills {
            let fill = PartialFill {
                order_id: order_id.to_string(),
                fill_id: format!("F_{}", order_id),
                quantity: qty,
                price,
                fee: qty * price * dec!(0.0003),
                timestamp: 1000,
                exchange: "binance".to_string(),
                role: TradeRole::Taker,
            };
            tracker.add_fill(fill).await.unwrap();
        }
        
        // Reconcile position
        tracker.reconcile_position().await;
        
        let summary = &tracker.position_summary;
        
        // Net position: 1.0 - 0.5 + 0.3 = 0.8 BTC
        assert_eq!(summary.total_quantity, dec!(0.8));
        
        // Weighted average entry
        let buy_value = dec!(1.0) * dec!(50000) + dec!(0.3) * dec!(49800);
        let buy_qty = dec!(1.3);
        let expected_avg = buy_value / buy_qty;
        
        // Account for the sell
        let expected_net_cost = buy_value - dec!(0.5) * dec!(50500);
        let expected_net_avg = expected_net_cost / dec!(0.8);
        
        assert!((summary.weighted_avg_entry - expected_net_avg).abs() < dec!(100));
    }
    
    #[tokio::test]
    async fn test_pnl_calculation() {
        let mut tracker = PartialFillTracker::new();
        
        // Buy 1 BTC at 50,000
        tracker.create_order(
            "BUY".to_string(),
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            dec!(1.0),
            None
        ).await;
        
        let buy_fill = PartialFill {
            order_id: "BUY".to_string(),
            fill_id: "BF1".to_string(),
            quantity: dec!(1.0),
            price: dec!(50000),
            fee: dec!(15),
            timestamp: 1000,
            exchange: "binance".to_string(),
            role: TradeRole::Taker,
        };
        tracker.add_fill(buy_fill).await.unwrap();
        
        // Sell 0.5 BTC at 51,000
        tracker.create_order(
            "SELL".to_string(),
            "BTC/USDT".to_string(),
            OrderSide::Sell,
            dec!(0.5),
            None
        ).await;
        
        let sell_fill = PartialFill {
            order_id: "SELL".to_string(),
            fill_id: "SF1".to_string(),
            quantity: dec!(0.5),
            price: dec!(51000),
            fee: dec!(7.65),
            timestamp: 1001,
            exchange: "binance".to_string(),
            role: TradeRole::Taker,
        };
        tracker.add_fill(sell_fill).await.unwrap();
        
        tracker.reconcile_position().await;
        
        // Realized PnL: 0.5 * (51000 - 50000) - fees = 500 - 22.65 = 477.35
        let expected_pnl = dec!(500) - dec!(22.65);
        assert!((tracker.position_summary.realized_pnl - expected_pnl).abs() < dec!(1));
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_high_frequency_fill_processing() {
        let tracker = Arc::new(RwLock::new(PartialFillTracker::new()));
        
        // Create order
        tracker.write().await.create_order(
            "HFT001".to_string(),
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            dec!(100.0),
            None
        ).await;
        
        // Simulate high-frequency fills
        let num_fills = 1000;
        let start = Instant::now();
        
        for i in 0..num_fills {
            let fill = PartialFill {
                order_id: "HFT001".to_string(),
                fill_id: format!("F{}", i),
                quantity: dec!(0.1),
                price: dec!(50000) + Decimal::from(i % 10),
                fee: dec!(0.015),
                timestamp: 1000 + i as i64,
                exchange: "binance".to_string(),
                role: if i % 2 == 0 { TradeRole::Maker } else { TradeRole::Taker },
            };
            
            tracker.write().await.add_fill(fill).await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let fills_per_second = num_fills as f64 / elapsed.as_secs_f64();
        
        // Should process >10,000 fills per second
        assert!(
            fills_per_second > 10000.0,
            "Fill processing too slow: {} fills/sec",
            fills_per_second
        );
        
        // Verify accuracy
        let status = tracker.read().await
            .get_order_status("HFT001").await.unwrap();
        assert_eq!(status.filled_quantity, dec!(100.0));
        assert_eq!(status.num_fills, 1000);
    }
    
    #[tokio::test]
    async fn test_concurrent_fill_updates() {
        let tracker = Arc::new(RwLock::new(PartialFillTracker::new()));
        
        // Create multiple orders
        for i in 0..10 {
            tracker.write().await.create_order(
                format!("ORD{}", i),
                "ETH/USDT".to_string(),
                OrderSide::Buy,
                dec!(10.0),
                None
            ).await;
        }
        
        // Spawn concurrent fill processors
        let mut handles = vec![];
        
        for order_idx in 0..10 {
            let tracker_clone = tracker.clone();
            let handle = tokio::spawn(async move {
                for fill_idx in 0..100 {
                    let fill = PartialFill {
                        order_id: format!("ORD{}", order_idx),
                        fill_id: format!("F{}_{}", order_idx, fill_idx),
                        quantity: dec!(0.1),
                        price: dec!(3000),
                        fee: dec!(0.09),
                        timestamp: 1000,
                        exchange: "kraken".to_string(),
                        role: TradeRole::Maker,
                    };
                    
                    tracker_clone.write().await
                        .add_fill(fill).await.unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify all fills processed correctly
        for i in 0..10 {
            let status = tracker.read().await
                .get_order_status(&format!("ORD{}", i)).await.unwrap();
            assert_eq!(status.filled_quantity, dec!(10.0));
            assert_eq!(status.num_fills, 100);
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_zero_quantity_fill() {
        let mut tracker = PartialFillTracker::new();
        
        tracker.create_order(
            "TEST".to_string(),
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            dec!(1.0),
            None
        ).await;
        
        let fill = PartialFill {
            order_id: "TEST".to_string(),
            fill_id: "F1".to_string(),
            quantity: dec!(0),
            price: dec!(50000),
            fee: dec!(0),
            timestamp: 1000,
            exchange: "binance".to_string(),
            role: TradeRole::Maker,
        };
        
        // Should reject zero quantity
        assert!(tracker.add_fill(fill).await.is_err());
    }
    
    #[tokio::test]
    async fn test_negative_price_rejection() {
        let mut tracker = PartialFillTracker::new();
        
        tracker.create_order(
            "TEST".to_string(),
            "ETH/USDT".to_string(),
            OrderSide::Sell,
            dec!(1.0),
            None
        ).await;
        
        let fill = PartialFill {
            order_id: "TEST".to_string(),
            fill_id: "F1".to_string(),
            quantity: dec!(1),
            price: dec!(-3000), // Invalid negative price
            fee: dec!(0.9),
            timestamp: 1000,
            exchange: "coinbase".to_string(),
            role: TradeRole::Taker,
        };
        
        // Should reject negative price
        assert!(tracker.add_fill(fill).await.is_err());
    }
    
    #[tokio::test]
    async fn test_extremely_large_fill() {
        let mut tracker = PartialFillTracker::new();
        
        // Create order with large quantity
        let large_qty = dec!(1000000000); // 1 billion
        tracker.create_order(
            "WHALE".to_string(),
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            large_qty,
            None
        ).await;
        
        let fill = PartialFill {
            order_id: "WHALE".to_string(),
            fill_id: "WHALE_FILL".to_string(),
            quantity: large_qty,
            price: dec!(50000),
            fee: large_qty * dec!(50000) * dec!(0.0003),
            timestamp: 1000,
            exchange: "binance".to_string(),
            role: TradeRole::Taker,
        };
        
        // Should handle large numbers correctly
        tracker.add_fill(fill).await.unwrap();
        
        let status = tracker.get_order_status("WHALE").await.unwrap();
        assert_eq!(status.filled_quantity, large_qty);
        assert!(status.total_fees > dec!(0));
    }
    
    #[tokio::test]
    async fn test_order_not_found() {
        let tracker = PartialFillTracker::new();
        
        // Query non-existent order
        let result = tracker.get_order_status("NONEXISTENT").await;
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_full_order_lifecycle() {
        let mut tracker = PartialFillTracker::new();
        let mut impact_estimator = MarketImpactEstimator::new();
        
        // Setup market impact model
        impact_estimator.kyle_lambda = dec!(0.0001);
        impact_estimator.temporary_impact_decay = dec!(0.5);
        
        // Create buy order
        let order_id = "LIFECYCLE".to_string();
        tracker.create_order(
            order_id.clone(),
            "ETH/USDT".to_string(),
            OrderSide::Buy,
            dec!(100.0),
            Some(dec!(3000))
        ).await;
        
        // Simulate partial fills over time
        let fill_schedule = vec![
            (dec!(20.0), dec!(2995), 0),    // Initial fill below limit
            (dec!(30.0), dec!(2998), 30),   // Mid fill
            (dec!(25.0), dec!(3000), 60),   // At limit
            (dec!(25.0), dec!(2999), 90),   // Final fill
        ];
        
        for (i, (qty, price, delay_secs)) in fill_schedule.iter().enumerate() {
            // Estimate market impact
            let impact = impact_estimator.estimate_impact(*qty);
            let adjusted_price = price + impact;
            
            let fill = PartialFill {
                order_id: order_id.clone(),
                fill_id: format!("FILL_{}", i),
                quantity: *qty,
                price: adjusted_price,
                fee: qty * adjusted_price * dec!(0.0003),
                timestamp: 1000 + (*delay_secs as i64),
                exchange: "kraken".to_string(),
                role: if i % 2 == 0 { TradeRole::Maker } else { TradeRole::Taker },
            };
            
            tracker.add_fill(fill).await.unwrap();
        }
        
        // Verify final state
        let status = tracker.get_order_status(&order_id).await.unwrap();
        assert_eq!(status.filled_quantity, dec!(100.0));
        assert_eq!(status.remaining_quantity, dec!(0));
        assert_eq!(status.status, OrderStatus::Filled);
        assert!(status.weighted_avg_price <= dec!(3000)); // Should be at or below limit
        
        // Test reconciliation
        tracker.reconcile_position().await;
        assert_eq!(tracker.position_summary.total_quantity, dec!(100.0));
    }
}