// Risk Engine Performance Benchmarks
// Sophia Issue #7: Proving <10μs pre-trade checks with statistical confidence
// Nexus requested: Raw perf stat output for independent verification

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rust_decimal_macros::dec;
use std::sync::Arc;
use std::time::Duration;

// Import the risk engine components
use risk_engine::{
    checks::{RiskEngine, RiskConfig, PositionLimit},
    correlation::CorrelationAnalyzer,
    emergency::{KillSwitch, EmergencyStop},
};
use order_management::order::{Order, OrderSide, OrderType, OrderState};

fn setup_risk_engine() -> Arc<RiskEngine> {
    let config = RiskConfig {
        max_position_pct: dec!(0.02),  // 2% max position
        require_stop_loss: true,
        max_daily_loss: dec!(1000),
        max_leverage: dec!(3),
        max_drawdown: dec!(0.15),  // 15% max drawdown
        min_order_size: dec!(10),
        max_order_size: dec!(10000),
        max_exposure: dec!(100000),
        check_correlation: true,
        max_correlation: 0.7,
    };
    
    Arc::new(RiskEngine::new(config))
}

fn create_test_order(size: rust_decimal::Decimal) -> Order {
    Order {
        id: uuid::Uuid::new_v4(),
        symbol: "BTC-USD".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: size,
        price: Some(dec!(50000)),
        stop_loss: Some(dec!(49000)),
        take_profit: None,
        client_order_id: None,
        state: OrderState::Created,
        created_at: std::time::SystemTime::now(),
        updated_at: std::time::SystemTime::now(),
    }
}

/// Benchmark pre-trade risk checks - must prove <10μs claim
fn bench_pre_trade_checks(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    let mut group = c.benchmark_group("pre_trade_checks");
    group.measurement_time(Duration::from_secs(30));  // Long measurement for accuracy
    group.sample_size(10000);  // Large sample for statistical confidence
    group.warm_up_time(Duration::from_secs(5));
    
    // Test with different order sizes to check scaling
    for size in &[dec!(10), dec!(100), dec!(1000)] {
        let order = create_test_order(*size);
        
        group.bench_with_input(
            BenchmarkId::new("order_size", size),
            &order,
            |b, order| {
                b.iter(|| {
                    let result = risk_engine.check_order_sync(black_box(order));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark position limit checks
fn bench_position_limits(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    c.bench_function("position_limit_check", |b| {
        let order = create_test_order(dec!(100));
        let portfolio_value = dec!(100000);
        
        b.iter(|| {
            let result = risk_engine.check_position_limit(
                black_box(&order),
                black_box(portfolio_value),
            );
            black_box(result);
        });
    });
}

/// Benchmark stop loss validation
fn bench_stop_loss_check(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    c.bench_function("stop_loss_validation", |b| {
        let order = create_test_order(dec!(100));
        
        b.iter(|| {
            let result = risk_engine.validate_stop_loss(black_box(&order));
            black_box(result);
        });
    });
}

/// Benchmark correlation analysis
fn bench_correlation_check(c: &mut Criterion) {
    let analyzer = CorrelationAnalyzer::new(0.7);
    
    // Pre-populate with some positions
    for i in 0..10 {
        analyzer.add_position(format!("ASSET_{}", i), dec!(1000));
    }
    
    c.bench_function("correlation_analysis", |b| {
        b.iter(|| {
            let correlation = analyzer.calculate_correlation(
                black_box("BTC-USD"),
                black_box("ETH-USD"),
            );
            black_box(correlation);
        });
    });
}

/// Benchmark emergency kill switch activation
fn bench_kill_switch(c: &mut Criterion) {
    c.bench_function("kill_switch_activation", |b| {
        let kill_switch = KillSwitch::new(None);
        
        b.iter(|| {
            let is_active = kill_switch.is_active();
            black_box(is_active);
        });
    });
}

/// Benchmark full risk check pipeline (all checks combined)
fn bench_full_pipeline(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    let mut group = c.benchmark_group("full_risk_pipeline");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10000);
    
    // Test with increasing complexity
    for num_checks in &[1, 3, 5, 7] {
        let order = create_test_order(dec!(100));
        
        group.bench_with_input(
            BenchmarkId::new("parallel_checks", num_checks),
            num_checks,
            |b, &num_checks| {
                b.iter(|| {
                    // Simulate running multiple checks in parallel
                    let mut results = Vec::with_capacity(num_checks);
                    
                    for _ in 0..num_checks {
                        let result = risk_engine.check_order_sync(black_box(&order));
                        results.push(result);
                    }
                    
                    black_box(results);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark throughput - how many risk checks per second
fn bench_throughput(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    let mut group = c.benchmark_group("risk_check_throughput");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("checks_per_second", |b| {
        let order = create_test_order(dec!(100));
        
        b.iter(|| {
            let result = risk_engine.check_order_sync(black_box(&order));
            black_box(result);
        });
    });
    
    group.finish();
}

/// Latency percentiles benchmark - critical for proving <10μs claim
fn bench_latency_percentiles(c: &mut Criterion) {
    let risk_engine = setup_risk_engine();
    
    let mut group = c.benchmark_group("latency_percentiles");
    group.measurement_time(Duration::from_secs(60));  // Long run for accurate percentiles
    group.sample_size(100000);  // Very large sample for p99.9 accuracy
    group.significance_level(0.01);  // High confidence level
    
    group.bench_function("pre_trade_latency_distribution", |b| {
        let order = create_test_order(dec!(100));
        
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = risk_engine.check_order_sync(black_box(&order));
            let elapsed = start.elapsed();
            
            // Track if we meet the <10μs requirement
            assert!(
                elapsed.as_micros() < 10,
                "Risk check took {}μs, exceeding 10μs limit",
                elapsed.as_micros()
            );
            
            black_box(result);
        });
    });
    
    group.finish();
}

// Criterion benchmark groups
criterion_group!(
    name = risk_benchmarks;
    config = Criterion::default()
        .with_profiler(criterion::profiler::perf::PerfProfiler)
        .plotting_backend(criterion::PlottingBackend::Gnuplot);
    targets = 
        bench_pre_trade_checks,
        bench_position_limits,
        bench_stop_loss_check,
        bench_correlation_check,
        bench_kill_switch,
        bench_full_pipeline,
        bench_throughput,
        bench_latency_percentiles
);

criterion_main!(risk_benchmarks);