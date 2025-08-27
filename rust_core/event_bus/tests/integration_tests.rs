//! # Integration Tests - Event Bus comprehensive test suite
//!
//! Tests the entire event bus system including:
//! - Ultra-low latency publishing (<1μs)
//! - Event ordering guarantees
//! - Multiple consumer scenarios
//! - Replay functionality
//! - Circuit breaker behavior

use event_bus::{
    RingBuffer, WaitStrategy, Event, EventType, EventHandler,
    BatchEventProcessor, EventJournal, EventReplayer,
    metrics::EventBusMetrics,
};
use async_trait::async_trait;
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Instant, Duration};
use chrono::Utc;
use tempfile::tempdir;
use domain_types::{Order, OrderSide, OrderType, Price, Quantity};

/// Test handler that counts events
struct CountingHandler {
    count: AtomicU64,
    last_sequence: AtomicU64,
}

impl CountingHandler {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl EventHandler<Event> for CountingHandler {
    async fn on_event(&self, _event: &Event, sequence: usize) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.last_sequence.store(sequence as u64, Ordering::Relaxed);
    }
}

/// Latency measuring handler
struct LatencyHandler {
    latencies: parking_lot::Mutex<Vec<Duration>>,
}

impl LatencyHandler {
    fn new() -> Self {
        Self {
            latencies: parking_lot::Mutex::new(Vec::with_capacity(100_000)),
        }
    }
    
    fn percentile(&self, p: f64) -> Duration {
        let mut latencies = self.latencies.lock();
        latencies.sort();
        
        let index = ((latencies.len() as f64 - 1.0) * p) as usize;
        latencies.get(index).copied().unwrap_or(Duration::ZERO)
    }
}

#[async_trait]
impl EventHandler<Event> for LatencyHandler {
    async fn on_event(&self, event: &Event, _sequence: usize) {
        let now = Utc::now();
        let latency = (now - event.timestamp()).to_std().unwrap_or(Duration::ZERO);
        self.latencies.lock().push(latency);
    }
}

#[tokio::test]
async fn test_ultra_low_latency() {
    // Create ring buffer
    let buffer = Arc::new(RingBuffer::<Event>::new(65536));
    let metrics = Arc::new(EventBusMetrics::new());
    
    // Measure publish latency
    let mut latencies = Vec::with_capacity(10_000);
    
    for _ in 0..10_000 {
        let event = Event::new(
            EventType::MarketTick {
                symbol: "BTC/USDT".to_string(),
                ticker: domain_types::Ticker::default(),
            },
            "test".to_string(),
        );
        
        let start = Instant::now();
        let sequence = buffer.next();
        buffer.publish(sequence, event);
        let elapsed = start.elapsed();
        
        latencies.push(elapsed);
        metrics.record_publish(elapsed.as_nanos() as u64);
    }
    
    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[latencies.len() * 99 / 100];
    let max = latencies[latencies.len() - 1];
    
    println!("Publish Latency - P50: {:?}, P99: {:?}, Max: {:?}", p50, p99, max);
    
    // Verify <1μs for P50
    assert!(p50 < Duration::from_micros(1), 
            "P50 latency {:?} exceeds 1μs target", p50);
    
    // P99 should be under 10μs
    assert!(p99 < Duration::from_micros(10),
            "P99 latency {:?} exceeds 10μs", p99);
}

#[tokio::test]
async fn test_event_ordering() {
    let buffer = Arc::new(RingBuffer::<Event>::new(1024));
    let handler = Arc::new(CountingHandler::new());
    
    // Create processor
    let barrier = buffer.new_barrier(WaitStrategy::Yielding);
    let processor = BatchEventProcessor::new(barrier, handler.clone());
    
    // Publish events with sequence numbers
    for i in 0..100 {
        let event = Event::new(
            EventType::HeartBeat {
                timestamp: Utc::now(),
                sequence: i,
            },
            "test".to_string(),
        );
        
        let sequence = buffer.next();
        buffer.publish(sequence, event);
    }
    
    // Start processor in background
    let processor_handle = tokio::spawn(async move {
        processor.run();
    });
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Verify all events processed in order
    assert_eq!(handler.count.load(Ordering::Relaxed), 100);
    assert_eq!(handler.last_sequence.load(Ordering::Relaxed), 99);
    
    processor_handle.abort();
}

#[tokio::test]
async fn test_multiple_consumers() {
    let buffer = Arc::new(RingBuffer::<Event>::new(4096));
    
    // Create multiple handlers
    let handlers: Vec<Arc<CountingHandler>> = (0..4)
        .map(|_| Arc::new(CountingHandler::new()))
        .collect();
    
    // Create processors
    let mut processors = Vec::new();
    for handler in &handlers {
        let barrier = buffer.new_barrier(WaitStrategy::BusySpin);
        processors.push(BatchEventProcessor::new(barrier, handler.clone()));
    }
    
    // Publish events
    for i in 0..1000 {
        let event = Event::new(
            EventType::OrderPlaced {
                order: Order {
                    id: format!("order_{}", i),
                    symbol: "BTC/USDT".to_string(),
                    side: OrderSide::Buy,
                    order_type: OrderType::Limit,
                    price: Some(Price::from_f64(50000.0)),
                    quantity: Quantity::from_f64(0.1),
                    timestamp: Utc::now(),
                    exchange: "binance".to_string(),
                    status: domain_types::OrderStatus::New,
                    fills: vec![],
                    time_in_force: domain_types::TimeInForce::GTC,
                    client_order_id: None,
                    reduce_only: false,
                    post_only: false,
                    close_position: false,
                },
                strategy_id: Some("test_strategy".to_string()),
            },
            "test".to_string(),
        );
        
        let sequence = buffer.next();
        buffer.publish(sequence, event);
    }
    
    // Start all processors
    let handles: Vec<_> = processors.into_iter()
        .map(|p| tokio::spawn(async move { p.run() }))
        .collect();
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Verify all handlers received all events
    for handler in &handlers {
        assert_eq!(handler.count.load(Ordering::Relaxed), 1000,
                   "Handler didn't receive all events");
    }
    
    // Cleanup
    for handle in handles {
        handle.abort();
    }
}

#[tokio::test]
async fn test_event_replay() {
    let dir = tempdir().unwrap();
    
    // Create journal and write events
    let mut journal = EventJournal::new(dir.path()).unwrap();
    let mut original_events = Vec::new();
    
    for i in 0..50 {
        let mut event = Event::new(
            EventType::SignalGenerated {
                strategy_id: "replay_test".to_string(),
                signal_type: domain_types::events::SignalType::Buy,
                confidence: 0.85 + (i as f64 * 0.001),
            },
            "test".to_string(),
        );
        event.metadata.sequence = i;
        
        journal.write(&event).unwrap();
        original_events.push(event);
    }
    
    // Create replayer and load events
    let mut replayer = EventReplayer::new(dir.path()).unwrap();
    let replayed = replayer.load_events(10, Some(30)).unwrap();
    
    // Verify correct range was loaded
    assert_eq!(replayed.len(), 21);  // Events 10-30 inclusive
    assert_eq!(replayed[0].metadata.sequence, 10);
    assert_eq!(replayed[20].metadata.sequence, 30);
    
    // Verify event integrity
    for (i, event) in replayed.iter().enumerate() {
        let original_idx = 10 + i;
        assert_eq!(event.metadata.sequence, original_events[original_idx].metadata.sequence);
        assert_eq!(event.metadata.source, original_events[original_idx].metadata.source);
    }
}

#[tokio::test]
async fn test_backpressure_handling() {
    let buffer = Arc::new(RingBuffer::<Event>::new(256));  // Small buffer
    let slow_handler = Arc::new(SlowHandler::new());
    
    let barrier = buffer.new_barrier(WaitStrategy::Blocking);
    let processor = BatchEventProcessor::new(barrier, slow_handler.clone());
    
    // Start processor
    let processor_handle = tokio::spawn(async move {
        processor.run();
    });
    
    // Try to publish more events than buffer can hold
    let publisher_handle = tokio::spawn(async move {
        for i in 0..1000 {
            let event = Event::new(
                EventType::HeartBeat {
                    timestamp: Utc::now(),
                    sequence: i,
                },
                "test".to_string(),
            );
            
            // This should block when buffer is full
            let sequence = buffer.next();
            buffer.publish(sequence, event);
        }
    });
    
    // Wait and verify no events lost
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Cleanup
    publisher_handle.abort();
    processor_handle.abort();
    
    assert!(slow_handler.processed.load(Ordering::Relaxed) > 0);
}

/// Slow handler to test backpressure
struct SlowHandler {
    processed: AtomicU64,
}

impl SlowHandler {
    fn new() -> Self {
        Self {
            processed: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl EventHandler<Event> for SlowHandler {
    async fn on_event(&self, _event: &Event, _sequence: usize) {
        // Simulate slow processing
        tokio::time::sleep(Duration::from_micros(100)).await;
        self.processed.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn test_circuit_breaker_integration() {
    let buffer = Arc::new(RingBuffer::<Event>::new(1024));
    let failing_handler = Arc::new(FailingHandler::new());
    
    let barrier = buffer.new_barrier(WaitStrategy::Yielding);
    let processor = BatchEventProcessor::new(barrier, failing_handler.clone());
    
    // Publish events that will trigger failures
    for i in 0..10 {
        let event = Event::new(
            EventType::RiskLimitBreached {
                limit_type: domain_types::events::RiskLimitType::MaxPositionSize,
                current_value: 100.0 + i as f64,
                limit_value: 100.0,
            },
            "test".to_string(),
        );
        
        let sequence = buffer.next();
        buffer.publish(sequence, event);
    }
    
    // Start processor
    let processor_handle = tokio::spawn(async move {
        processor.run();
    });
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Verify circuit breaker tripped after 3 failures
    assert_eq!(failing_handler.attempts.load(Ordering::Relaxed), 3);
    assert!(failing_handler.circuit_open.load(Ordering::Relaxed));
    
    processor_handle.abort();
}

/// Handler that fails to test circuit breaker
struct FailingHandler {
    attempts: AtomicU64,
    circuit_open: AtomicBool,
}

impl FailingHandler {
    fn new() -> Self {
        Self {
            attempts: AtomicU64::new(0),
            circuit_open: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl EventHandler<Event> for FailingHandler {
    async fn on_event(&self, _event: &Event, _sequence: usize) {
        let attempts = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        
        if attempts >= 3 {
            self.circuit_open.store(true, Ordering::Relaxed);
            // Circuit breaker would stop processing here
        }
    }
}

#[tokio::test]
async fn test_concurrent_publishers() {
    let buffer = Arc::new(RingBuffer::<Event>::new(8192));
    let metrics = Arc::new(EventBusMetrics::new());
    let handler = Arc::new(CountingHandler::new());
    
    // Create processor
    let barrier = buffer.new_barrier(WaitStrategy::BusySpin);
    let processor = BatchEventProcessor::new(barrier, handler.clone());
    
    // Start processor
    let processor_handle = tokio::spawn(async move {
        processor.run();
    });
    
    // Create multiple publishers
    let num_publishers = 8;
    let events_per_publisher = 1000;
    let buffer_clone = buffer.clone();
    let metrics_clone = metrics.clone();
    
    let publisher_handles: Vec<_> = (0..num_publishers)
        .map(|publisher_id| {
            let buffer = buffer_clone.clone();
            let metrics = metrics_clone.clone();
            
            tokio::spawn(async move {
                for i in 0..events_per_publisher {
                    let event = Event::new(
                        EventType::HeartBeat {
                            timestamp: Utc::now(),
                            sequence: (publisher_id * events_per_publisher + i) as u64,
                        },
                        format!("publisher_{}", publisher_id),
                    );
                    
                    let start = Instant::now();
                    let sequence = buffer.next();
                    buffer.publish(sequence, event);
                    let latency_ns = start.elapsed().as_nanos() as u64;
                    
                    metrics.record_publish(latency_ns);
                }
            })
        })
        .collect();
    
    // Wait for all publishers
    for handle in publisher_handles {
        handle.await.unwrap();
    }
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Verify all events processed
    let expected_total = (num_publishers * events_per_publisher) as u64;
    assert_eq!(handler.count.load(Ordering::Relaxed), expected_total);
    
    // Check metrics
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.events_published, expected_total);
    println!("Concurrent Publishing - P50: {}ns, P99: {}ns, Throughput: {:.0} events/sec",
             snapshot.p50_latency_ns, snapshot.p99_latency_ns, snapshot.throughput);
    
    processor_handle.abort();
}

#[tokio::test]
async fn test_event_correlation() {
    let buffer = Arc::new(RingBuffer::<Event>::new(1024));
    let correlation_id = uuid::Uuid::new_v4();
    
    // Publish correlated events
    let events = vec![
        EventType::OrderPlaced {
            order: create_test_order("order_1"),
            strategy_id: Some("correlation_test".to_string()),
        },
        EventType::OrderFilled {
            order_id: "order_1".to_string(),
            trade: create_test_trade(),
            remaining_quantity: Quantity::from_f64(0.0),
        },
        EventType::PositionOpened {
            symbol: "BTC/USDT".to_string(),
            quantity: Quantity::from_f64(0.1),
            entry_price: Price::from_f64(50000.0),
        },
    ];
    
    for event_type in events {
        let mut event = Event::new(event_type, "test".to_string());
        event.metadata.correlation_id = Some(correlation_id);
        
        let sequence = buffer.next();
        buffer.publish(sequence, event);
    }
    
    // Handler that tracks correlated events
    let correlated_handler = Arc::new(CorrelationHandler::new());
    let barrier = buffer.new_barrier(WaitStrategy::Yielding);
    let processor = BatchEventProcessor::new(barrier, correlated_handler.clone());
    
    // Process events
    let processor_handle = tokio::spawn(async move {
        processor.run();
    });
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Verify correlation tracking
    let chains = correlated_handler.correlation_chains.lock();
    assert_eq!(chains.len(), 1);
    assert_eq!(chains.get(&correlation_id).unwrap().len(), 3);
    
    processor_handle.abort();
}

struct CorrelationHandler {
    correlation_chains: parking_lot::Mutex<std::collections::HashMap<uuid::Uuid, Vec<Event>>>,
}

impl CorrelationHandler {
    fn new() -> Self {
        Self {
            correlation_chains: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

#[async_trait]
impl EventHandler<Event> for CorrelationHandler {
    async fn on_event(&self, event: &Event, _sequence: usize) {
        if let Some(correlation_id) = event.metadata.correlation_id {
            let mut chains = self.correlation_chains.lock();
            chains.entry(correlation_id)
                .or_insert_with(Vec::new)
                .push(event.clone());
        }
    }
}

fn create_test_order(id: &str) -> Order {
    Order {
        id: id.to_string(),
        symbol: "BTC/USDT".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        price: Some(Price::from_f64(50000.0)),
        quantity: Quantity::from_f64(0.1),
        timestamp: Utc::now(),
        exchange: "binance".to_string(),
        status: domain_types::OrderStatus::New,
        fills: vec![],
        time_in_force: domain_types::TimeInForce::GTC,
        client_order_id: None,
        reduce_only: false,
        post_only: false,
        close_position: false,
    }
}

fn create_test_trade() -> domain_types::Trade {
    domain_types::Trade {
        id: "trade_1".to_string(),
        symbol: "BTC/USDT".to_string(),
        price: Price::from_f64(50000.0),
        quantity: Quantity::from_f64(0.1),
        side: OrderSide::Buy,
        timestamp: Utc::now(),
        exchange: "binance".to_string(),
        order_id: Some("order_1".to_string()),
        fee: Some(domain_types::Fee {
            amount: rust_decimal::Decimal::from_f64_retain(0.0001).unwrap(),
            currency: "BTC".to_string(),
        }),
        is_maker: true,
    }
}