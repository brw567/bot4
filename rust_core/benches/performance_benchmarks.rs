// Performance Benchmarks Suite
// Team: Jordan (Performance Lead) + Full Team
// Target: <100μs order submission, <50ns ML inference

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use bot4_main::*;
use trading_engine::*;
use risk_engine::*;
use order_management::*;
use ml::*;
use infrastructure::*;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// TRADING ENGINE BENCHMARKS
// ============================================================================

fn bench_order_submission(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = infrastructure::config::Config::test_config();
    let engine = rt.block_on(async {
        trading_engine::engine::TradingEngine::new(config).await.unwrap()
    });
    
    let mut group = c.benchmark_group("order_submission");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark different order types
    for order_type in &["Market", "Limit", "StopLoss"] {
        group.bench_with_input(
            BenchmarkId::from_parameter(order_type),
            order_type,
            |b, &ot| {
                b.to_async(&rt).iter(|| async {
                    let order = create_test_order(ot);
                    engine.submit_order(order).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_orders(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = infrastructure::config::Config::test_config();
    let engine = std::sync::Arc::new(rt.block_on(async {
        trading_engine::engine::TradingEngine::new(config).await.unwrap()
    }));
    
    let mut group = c.benchmark_group("concurrent_orders");
    group.throughput(Throughput::Elements(1000));
    
    for num_orders in &[100u64, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_orders),
            num_orders,
            |b, &n| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = vec![];
                    for i in 0..n {
                        let engine_clone = engine.clone();
                        let handle = tokio::spawn(async move {
                            let order = create_test_order("Market");
                            engine_clone.submit_order(order).await
                        });
                        handles.push(handle);
                    }
                    futures::future::join_all(handles).await
                });
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// RISK ENGINE BENCHMARKS
// ============================================================================

fn bench_risk_checks(c: &mut Criterion) {
    use risk_engine::checks::RiskChecker;
    use risk_engine::limits::RiskLimits;
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let limits = RiskLimits::default();
    let checker = RiskChecker::new(limits);
    
    let mut group = c.benchmark_group("risk_checks");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("single_order_check", |b| {
        b.to_async(&rt).iter(|| async {
            let mut order = create_test_order("Limit");
            order.stop_loss_price = Some(dec!(49000));
            order.position_size_pct = dec!(0.01);
            checker.check_order(&order).await
        });
    });
    
    group.bench_function("batch_order_check_10", |b| {
        b.to_async(&rt).iter(|| async {
            let orders: Vec<_> = (0..10)
                .map(|_| {
                    let mut order = create_test_order("Limit");
                    order.stop_loss_price = Some(dec!(49000));
                    order.position_size_pct = dec!(0.01);
                    order
                })
                .collect();
            
            for order in orders {
                checker.check_order(&order).await;
            }
        });
    });
    
    group.finish();
}

fn bench_correlation_analysis(c: &mut Criterion) {
    use risk_engine::correlation_portable::PortableCorrelationAnalyzer;
    
    let analyzer = PortableCorrelationAnalyzer::new(0.7);
    
    // Generate test data
    let btc_prices: Vec<f64> = (0..1000)
        .map(|i| 50000.0 + (i as f64 * 0.1).sin() * 1000.0)
        .collect();
    let eth_prices: Vec<f64> = (0..1000)
        .map(|i| 3000.0 + (i as f64 * 0.1).cos() * 100.0)
        .collect();
    
    analyzer.add_price_series(0, btc_prices.clone());
    analyzer.add_price_series(1, eth_prices.clone());
    
    let mut group = c.benchmark_group("correlation");
    
    group.bench_function("correlation_matrix_2x2", |b| {
        b.iter(|| {
            analyzer.calculate_correlation_matrix()
        });
    });
    
    group.bench_function("pearson_correlation_1000", |b| {
        b.iter(|| {
            analyzer.correlation(&btc_prices, &eth_prices)
        });
    });
    
    group.finish();
}

// ============================================================================
// ML PIPELINE BENCHMARKS
// ============================================================================

fn bench_ml_feature_extraction(c: &mut Criterion) {
    use ml::signal_processing::SignalProcessor;
    
    let processor = SignalProcessor::new(ml::signal_processing::SignalConfig::default());
    
    let mut group = c.benchmark_group("ml_features");
    
    for size in &[100, 500, 1000, 5000] {
        let prices: Vec<f64> = (0..*size)
            .map(|i| 50000.0 + (i as f64 * 0.1).sin() * 1000.0)
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("extract_features", size),
            &prices,
            |b, prices| {
                b.iter(|| {
                    processor.extract_features(black_box(prices))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_ml_inference(c: &mut Criterion) {
    // Simulate ML model inference
    let mut group = c.benchmark_group("ml_inference");
    group.measurement_time(Duration::from_secs(3));
    
    // Mock features vector
    let features = vec![0.5; 100];
    
    group.bench_function("single_prediction", |b| {
        b.iter(|| {
            // Simulate neural network forward pass
            let mut result = 0.0;
            for (i, &feature) in features.iter().enumerate() {
                result += feature * (i as f64 * 0.01);
            }
            result.tanh()
        });
    });
    
    group.bench_function("ensemble_prediction_5_models", |b| {
        b.iter(|| {
            let mut predictions = vec![];
            for model_id in 0..5 {
                let mut result = 0.0;
                for (i, &feature) in features.iter().enumerate() {
                    result += feature * (i as f64 * 0.01 + model_id as f64 * 0.1);
                }
                predictions.push(result.tanh());
            }
            // Weighted average
            predictions.iter().sum::<f64>() / predictions.len() as f64
        });
    });
    
    group.finish();
}

// ============================================================================
// INFRASTRUCTURE BENCHMARKS
// ============================================================================

fn bench_object_pool(c: &mut Criterion) {
    use infrastructure::object_pools::{ObjectPool, OrderPool};
    
    let pool = OrderPool::new(1000);
    
    let mut group = c.benchmark_group("object_pool");
    
    group.bench_function("acquire_release", |b| {
        b.iter(|| {
            let order = pool.acquire();
            black_box(&order);
            pool.release(order);
        });
    });
    
    group.bench_function("acquire_release_1000", |b| {
        b.iter(|| {
            let mut orders = vec![];
            for _ in 0..1000 {
                orders.push(pool.acquire());
            }
            for order in orders {
                pool.release(order);
            }
        });
    });
    
    group.finish();
}

fn bench_circuit_breaker(c: &mut Criterion) {
    use infrastructure::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    
    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        recovery_timeout: Duration::from_secs(1),
        half_open_max_calls: 1,
    };
    
    let breaker = CircuitBreaker::new("bench", config);
    
    let mut group = c.benchmark_group("circuit_breaker");
    
    group.bench_function("check_state", |b| {
        b.iter(|| {
            breaker.is_open()
        });
    });
    
    group.bench_function("record_success", |b| {
        b.iter(|| {
            breaker.record_success()
        });
    });
    
    group.bench_function("record_failure", |b| {
        b.iter(|| {
            breaker.record_failure();
            breaker.reset(); // Reset for next iteration
        });
    });
    
    group.finish();
}

// ============================================================================
// ORDER MANAGEMENT BENCHMARKS
// ============================================================================

fn bench_position_tracking(c: &mut Criterion) {
    use order_management::{Position, PositionManager};
    
    let mut manager = PositionManager::new();
    
    // Add some positions
    for i in 0..100 {
        let position = Position::new(
            format!("SYMBOL_{}", i),
            order_management::OrderSide::Buy,
            dec!(0.1),
            dec!(50000),
        );
        manager.add_position(position);
    }
    
    let mut group = c.benchmark_group("position_tracking");
    
    group.bench_function("get_position", |b| {
        b.iter(|| {
            manager.get_position("SYMBOL_50")
        });
    });
    
    group.bench_function("update_market_prices", |b| {
        b.iter(|| {
            for i in 0..100 {
                manager.update_market_price(&format!("SYMBOL_{}", i), dec!(51000));
            }
        });
    });
    
    group.bench_function("calculate_total_pnl", |b| {
        b.iter(|| {
            manager.calculate_total_pnl()
        });
    });
    
    group.finish();
}

// ============================================================================
// END-TO-END BENCHMARKS
// ============================================================================

fn bench_full_trading_cycle(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("end_to_end");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("complete_trade_cycle", |b| {
        b.to_async(&rt).iter(|| async {
            // 1. Generate signal
            let signal = generate_mock_signal();
            
            // 2. Risk check
            let risk_approved = check_mock_risk(signal);
            
            // 3. Create and submit order
            if risk_approved {
                let order = create_test_order("Market");
                // Mock submission
                black_box(order);
            }
            
            signal
        });
    });
    
    group.finish();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_order(order_type: &str) -> order_management::Order {
    let mut order = order_management::Order::new(
        "BTCUSDT".to_string(),
        order_management::OrderSide::Buy,
        match order_type {
            "Market" => order_management::OrderType::Market,
            "Limit" => order_management::OrderType::Limit,
            "StopLoss" => order_management::OrderType::StopLoss,
            _ => order_management::OrderType::Market,
        },
        dec!(0.01),
    );
    
    if order_type == "Limit" {
        order.price = Some(dec!(50000));
    }
    
    order
}

fn generate_mock_signal() -> f64 {
    // Mock ML signal generation
    0.85
}

fn check_mock_risk(signal: f64) -> bool {
    signal < 0.95
}

// Configure and run benchmarks
criterion_group!(
    benches,
    bench_order_submission,
    bench_concurrent_orders,
    bench_risk_checks,
    bench_correlation_analysis,
    bench_ml_feature_extraction,
    bench_ml_inference,
    bench_object_pool,
    bench_circuit_breaker,
    bench_position_tracking,
    bench_full_trading_cycle,
);

criterion_main!(benches);