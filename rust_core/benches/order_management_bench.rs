// Order Management Performance Benchmarks
// Sophia Issue #8: Order processing benchmarks with CI artifacts
// Proving <100μs internal processing claim

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rust_decimal_macros::dec;
use std::sync::Arc;
use std::time::Duration;

use order_management::{
    order::{Order, OrderId, OrderSide, OrderType, OrderState},
    state_machine::{OrderStateMachine, StateTransition},
    manager::OrderManager,
    router::{OrderRouter, RoutingStrategy, ExchangeRoute},
};

fn create_test_order() -> Order {
    Order {
        id: OrderId::new(),
        symbol: "BTC-USD".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        quantity: dec!(1.5),
        price: Some(dec!(50000)),
        stop_loss: Some(dec!(49000)),
        take_profit: Some(dec!(51000)),
        client_order_id: Some("test_order_001".to_string()),
        state: OrderState::Created,
        created_at: std::time::SystemTime::now(),
        updated_at: std::time::SystemTime::now(),
    }
}

/// Benchmark atomic state transitions
fn bench_state_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transitions");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10000);
    
    // Test each valid state transition
    let transitions = vec![
        (OrderState::Created, OrderState::Validated),
        (OrderState::Validated, OrderState::Submitted),
        (OrderState::Submitted, OrderState::PartiallyFilled),
        (OrderState::PartiallyFilled, OrderState::Filled),
    ];
    
    for (from, to) in transitions {
        let state_machine = Arc::new(OrderStateMachine::new(from));
        
        group.bench_with_input(
            BenchmarkId::new("transition", format!("{:?}_to_{:?}", from, to)),
            &to,
            |b, &to_state| {
                b.iter(|| {
                    let result = state_machine.transition_to(
                        black_box(to_state),
                        black_box(None),
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark order validation
fn bench_order_validation(c: &mut Criterion) {
    let manager = OrderManager::new();
    
    c.bench_function("order_validation", |b| {
        let order = create_test_order();
        
        b.iter(|| {
            let result = manager.validate_order(black_box(&order));
            black_box(result);
        });
    });
}

/// Benchmark smart order routing
fn bench_order_routing(c: &mut Criterion) {
    let router = OrderRouter::new();
    
    // Add some test routes
    router.add_route(ExchangeRoute {
        exchange: "Binance".to_string(),
        fee_rate: dec!(0.001),
        liquidity_score: 0.95,
        avg_latency_ms: 25,
    });
    
    router.add_route(ExchangeRoute {
        exchange: "Kraken".to_string(),
        fee_rate: dec!(0.0015),
        liquidity_score: 0.85,
        avg_latency_ms: 30,
    });
    
    router.add_route(ExchangeRoute {
        exchange: "Coinbase".to_string(),
        fee_rate: dec!(0.002),
        liquidity_score: 0.90,
        avg_latency_ms: 20,
    });
    
    let mut group = c.benchmark_group("order_routing");
    
    for strategy in &[
        RoutingStrategy::BestPrice,
        RoutingStrategy::LowestFee,
        RoutingStrategy::SmartRoute,
    ] {
        let order = create_test_order();
        
        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{:?}", strategy)),
            strategy,
            |b, strategy| {
                b.iter(|| {
                    let route = router.select_route(
                        black_box(&order),
                        black_box(*strategy),
                    );
                    black_box(route);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark position tracking
fn bench_position_tracking(c: &mut Criterion) {
    let manager = OrderManager::new();
    
    // Pre-populate with some positions
    for i in 0..10 {
        let mut order = create_test_order();
        order.symbol = format!("ASSET_{}", i);
        manager.add_order(order);
    }
    
    c.bench_function("position_update", |b| {
        let order_id = OrderId::new();
        let new_quantity = dec!(2.5);
        
        b.iter(|| {
            let result = manager.update_position(
                black_box(&order_id),
                black_box(new_quantity),
            );
            black_box(result);
        });
    });
}

/// Benchmark P&L calculation
fn bench_pnl_calculation(c: &mut Criterion) {
    let manager = OrderManager::new();
    
    // Create position with entry and current prices
    let entry_price = dec!(50000);
    let current_price = dec!(51000);
    let quantity = dec!(1.5);
    
    c.bench_function("pnl_calculation", |b| {
        b.iter(|| {
            let pnl = manager.calculate_pnl(
                black_box(entry_price),
                black_box(current_price),
                black_box(quantity),
            );
            black_box(pnl);
        });
    });
}

/// Benchmark full order processing pipeline
fn bench_full_order_pipeline(c: &mut Criterion) {
    let manager = Arc::new(OrderManager::new());
    let router = Arc::new(OrderRouter::new());
    
    // Setup exchange routes
    router.add_route(ExchangeRoute {
        exchange: "Binance".to_string(),
        fee_rate: dec!(0.001),
        liquidity_score: 0.95,
        avg_latency_ms: 25,
    });
    
    let mut group = c.benchmark_group("full_order_pipeline");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10000);
    
    group.bench_function("complete_order_flow", |b| {
        b.iter(|| {
            let order = create_test_order();
            
            // 1. Validate order
            let validation = manager.validate_order(&order);
            
            // 2. State transition to validated
            if validation.is_ok() {
                let state_machine = OrderStateMachine::new(order.state);
                let _ = state_machine.transition_to(OrderState::Validated, None);
                
                // 3. Route order
                let route = router.select_route(&order, RoutingStrategy::SmartRoute);
                
                // 4. Submit order (state transition)
                let _ = state_machine.transition_to(OrderState::Submitted, None);
                
                // 5. Track position
                manager.add_order(order.clone());
                
                black_box(route);
            }
            
            black_box(validation);
        });
    });
    
    group.finish();
}

/// Benchmark order throughput
fn bench_order_throughput(c: &mut Criterion) {
    let manager = OrderManager::new();
    
    let mut group = c.benchmark_group("order_throughput");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("orders_per_second", |b| {
        b.iter(|| {
            let order = create_test_order();
            manager.add_order(black_box(order));
        });
    });
    
    group.finish();
}

/// Critical latency benchmark - must prove <100μs claim
fn bench_critical_latency(c: &mut Criterion) {
    let manager = Arc::new(OrderManager::new());
    
    let mut group = c.benchmark_group("critical_latency");
    group.measurement_time(Duration::from_secs(60));  // Long measurement
    group.sample_size(100000);  // Large sample for p99.9
    group.significance_level(0.01);
    
    group.bench_function("order_processing_latency", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            
            let order = create_test_order();
            
            // Complete internal processing
            let _ = manager.validate_order(&order);
            let state_machine = OrderStateMachine::new(order.state);
            let _ = state_machine.transition_to(OrderState::Validated, None);
            let _ = state_machine.transition_to(OrderState::Submitted, None);
            
            let elapsed = start.elapsed();
            
            // Assert we meet the <100μs requirement
            assert!(
                elapsed.as_micros() < 100,
                "Order processing took {}μs, exceeding 100μs limit",
                elapsed.as_micros()
            );
            
            black_box(order);
        });
    });
    
    group.finish();
}

/// Benchmark concurrent order processing
fn bench_concurrent_orders(c: &mut Criterion) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let manager = Arc::new(OrderManager::new());
    let processed = Arc::new(AtomicUsize::new(0));
    
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(20));
    
    for num_threads in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let mut handles = vec![];
                    
                    for _ in 0..num_threads {
                        let mgr = manager.clone();
                        let counter = processed.clone();
                        
                        let handle = std::thread::spawn(move || {
                            let order = create_test_order();
                            mgr.add_order(order);
                            counter.fetch_add(1, Ordering::Relaxed);
                        });
                        
                        handles.push(handle);
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

// Benchmark groups with perf profiler for detailed analysis
criterion_group!(
    name = order_benchmarks;
    config = Criterion::default()
        .with_profiler(criterion::profiler::perf::PerfProfiler)
        .plotting_backend(criterion::PlottingBackend::Gnuplot);
    targets = 
        bench_state_transitions,
        bench_order_validation,
        bench_order_routing,
        bench_position_tracking,
        bench_pnl_calculation,
        bench_full_order_pipeline,
        bench_order_throughput,
        bench_critical_latency,
        bench_concurrent_orders
);

criterion_main!(order_benchmarks);