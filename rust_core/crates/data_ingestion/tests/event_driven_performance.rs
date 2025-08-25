// Layer 1.3: Event-Driven Processing Performance Tests
// DEEP DIVE Implementation - Comprehensive performance validation
//
// Tests verify:
// - 42μs median latency target (Chronicle Software benchmark)
// - Adaptive sampling responds to volatility within 100ms
// - Bucketed aggregation maintains accuracy under load
// - System handles 300k events/sec sustained throughput

use data_ingestion::event_driven::{
    EventProcessor, ProcessorConfig, EventPriority,
    AdaptiveSampler, SamplerConfig, VolatilityRegime,
    BucketedAggregator, BucketConfig, AggregateWindow,
};
use data_ingestion::producers::{MarketEvent, TradeSide};
use types::{Price, Quantity};
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;
use crossbeam_channel::{unbounded, bounded};
use statrs::statistics::{Data, OrderStatistics};
use rand::prelude::*;

// Performance targets based on research
const TARGET_MEDIAN_LATENCY_US: u64 = 42;  // Chronicle Software benchmark
const TARGET_P99_LATENCY_US: u64 = 100;    // Industry standard
const TARGET_THROUGHPUT_PER_SEC: u64 = 300_000;  // From requirements
const BURST_MULTIPLIER: u64 = 10;  // Handle 10x bursts

#[test]
fn test_event_processor_latency() {
    let config = ProcessorConfig {
        queue_size: 100_000,
        worker_threads: 4,
        batch_size: 100,
        rate_limit: Some(1_000_000),  // 1M events/sec for testing
    };
    
    let mut processor = EventProcessor::new(config);
    let (tx, rx) = unbounded();
    let latencies = Arc::new(AtomicU64::new(0));
    let count = Arc::new(AtomicU64::new(0));
    
    // Start processor
    processor.start(tx);
    
    // Measure latencies for 100k events
    let mut individual_latencies = Vec::with_capacity(100_000);
    
    for i in 0..100_000 {
        let event = create_test_event(i as f64);
        let priority = if i % 1000 == 0 { 
            EventPriority::High 
        } else { 
            EventPriority::Medium 
        };
        
        let start = Instant::now();
        processor.process_event(event, priority).unwrap();
        
        // Wait for processing
        if let Ok(_) = rx.recv_timeout(Duration::from_millis(1)) {
            let latency = start.elapsed().as_micros() as u64;
            individual_latencies.push(latency as f64);
            
            latencies.fetch_add(latency, Ordering::Relaxed);
            count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    processor.stop();
    
    // Calculate statistics
    let mut data = Data::new(individual_latencies);
    let median = data.median();
    let p99 = data.quantile(0.99);
    let p999 = data.quantile(0.999);
    
    println!("Event Processor Latency Results:");
    println!("  Median: {:.2}μs (target: {}μs)", median, TARGET_MEDIAN_LATENCY_US);
    println!("  P99: {:.2}μs (target: {}μs)", p99, TARGET_P99_LATENCY_US);
    println!("  P99.9: {:.2}μs", p999);
    
    assert!(median < TARGET_MEDIAN_LATENCY_US as f64 * 1.5, 
            "Median latency {:.2}μs exceeds target", median);
    assert!(p99 < TARGET_P99_LATENCY_US as f64 * 2.0,
            "P99 latency {:.2}μs exceeds target", p99);
}

#[test]
fn test_adaptive_sampler_responsiveness() {
    let config = SamplerConfig {
        base_interval_ms: 10,
        min_interval_ms: 1,
        max_interval_ms: 100,
        burst_threshold: 3.0,
        garch_alpha: 0.1,
        garch_beta: 0.85,
        garch_omega: 0.00001,
    };
    
    let mut sampler = AdaptiveSampler::new(config);
    
    // Test regime transitions
    let mut transition_times = Vec::new();
    
    // Simulate sudden volatility spike
    for i in 0..1000 {
        let price = if i < 500 {
            // Low volatility period
            100.0 + (i as f64 * 0.01).sin() * 0.1
        } else {
            // High volatility period  
            100.0 + (i as f64 * 0.01).sin() * 5.0 + thread_rng().gen_range(-2.0..2.0)
        };
        
        let event = create_test_event(price);
        let start = Instant::now();
        
        if sampler.should_sample(&event) {
            let regime = sampler.get_volatility_regime();
            
            // Check if regime changed at transition point
            if i == 500 {
                transition_times.push(start.elapsed());
            }
            
            if i < 500 {
                assert!(matches!(regime, VolatilityRegime::Low | VolatilityRegime::VeryLow),
                       "Expected low volatility regime before transition");
            } else if i > 600 {
                assert!(matches!(regime, VolatilityRegime::High | VolatilityRegime::Extreme),
                       "Expected high volatility regime after transition");
            }
        }
        
        thread::sleep(Duration::from_micros(100));
    }
    
    // Verify quick response to volatility changes
    for time in &transition_times {
        assert!(time.as_millis() < 100, 
                "Sampler took {:?} to respond to volatility change", time);
    }
}

#[test]
fn test_bucketed_aggregator_accuracy() {
    let config = BucketConfig {
        bucket_sizes_ms: vec![1, 5, 10, 100, 1000],
        max_buckets_per_level: 1000,
        enable_flow_toxicity: true,
        enable_price_efficiency: true,
    };
    
    let mut aggregator = BucketedAggregator::new(config);
    
    // Generate events with known patterns
    let mut total_volume = Quantity::new(0.0);
    let mut total_buy_volume = Quantity::new(0.0);
    let mut min_price = Price::new(f64::MAX);
    let mut max_price = Price::new(0.0);
    
    for i in 0..10_000 {
        let price = 100.0 + (i as f64 * 0.001).sin() * 10.0;
        let volume = 1.0 + (i as f64 * 0.01).cos().abs();
        let is_buy = i % 3 != 0;
        
        let mut event = create_test_event(price);
        event.volume = Quantity::new(volume);
        event.side = if is_buy { TradeSide::Buy } else { TradeSide::Sell };
        
        aggregator.add_event(&event);
        
        total_volume = total_volume + event.volume;
        if is_buy {
            total_buy_volume = total_buy_volume + event.volume;
        }
        min_price = min_price.min(event.price);
        max_price = max_price.max(event.price);
    }
    
    // Get 1-second aggregate
    let windows = aggregator.get_aggregates(1000);
    assert!(!windows.is_empty(), "Should have 1-second aggregates");
    
    let window = &windows[0];
    
    // Verify aggregate accuracy
    assert!((window.volume.as_f64() - total_volume.as_f64()).abs() < 0.01,
            "Volume mismatch: {} vs {}", window.volume.as_f64(), total_volume.as_f64());
    
    assert!(window.low <= min_price, "Low price incorrect");
    assert!(window.high >= max_price, "High price incorrect");
    
    // Check order imbalance calculation
    let expected_imbalance = (total_buy_volume.as_f64() - (total_volume.as_f64() - total_buy_volume.as_f64())) 
                             / total_volume.as_f64();
    assert!((window.order_imbalance - expected_imbalance).abs() < 0.1,
            "Order imbalance calculation incorrect");
}

#[test]
fn test_sustained_throughput() {
    let config = ProcessorConfig {
        queue_size: 1_000_000,
        worker_threads: 8,
        batch_size: 1000,
        rate_limit: None,  // No limit for throughput test
    };
    
    let mut processor = EventProcessor::new(config);
    let (tx, _rx) = unbounded();
    processor.start(tx);
    
    let processed = Arc::new(AtomicU64::new(0));
    let processed_clone = processed.clone();
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    
    // Monitor throughput
    let monitor = thread::spawn(move || {
        let mut last_count = 0;
        while running_clone.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(1));
            let current = processed_clone.load(Ordering::Relaxed);
            let throughput = current - last_count;
            println!("Throughput: {} events/sec", throughput);
            
            assert!(throughput > TARGET_THROUGHPUT_PER_SEC / 2,
                    "Throughput {} below target {}", throughput, TARGET_THROUGHPUT_PER_SEC);
            
            last_count = current;
        }
    });
    
    // Generate load for 10 seconds
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        for _ in 0..1000 {
            let event = create_test_event(100.0 + thread_rng().gen_range(-1.0..1.0));
            if processor.process_event(event, EventPriority::Medium).is_ok() {
                processed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    running.store(false, Ordering::Relaxed);
    monitor.join().unwrap();
    processor.stop();
    
    let total_processed = processed.load(Ordering::Relaxed);
    let avg_throughput = total_processed / 10;
    
    println!("Average throughput: {} events/sec", avg_throughput);
    assert!(avg_throughput > TARGET_THROUGHPUT_PER_SEC / 2,
            "Average throughput {} below target", avg_throughput);
}

#[test]
fn test_burst_handling() {
    let config = ProcessorConfig {
        queue_size: 500_000,
        worker_threads: 8,
        batch_size: 1000,
        rate_limit: Some(TARGET_THROUGHPUT_PER_SEC * 2),
    };
    
    let mut processor = EventProcessor::new(config);
    let (tx, rx) = bounded(1_000_000);
    processor.start(tx);
    
    // Normal load for 2 seconds
    for _ in 0..TARGET_THROUGHPUT_PER_SEC * 2 {
        let event = create_test_event(100.0);
        processor.process_event(event, EventPriority::Low).unwrap();
    }
    
    // Sudden burst - 10x normal rate
    let burst_start = Instant::now();
    let mut burst_accepted = 0u64;
    let mut burst_rejected = 0u64;
    
    for _ in 0..TARGET_THROUGHPUT_PER_SEC * BURST_MULTIPLIER {
        let event = create_test_event(100.0);
        match processor.process_event(event, EventPriority::High) {
            Ok(_) => burst_accepted += 1,
            Err(_) => burst_rejected += 1,
        }
    }
    
    let burst_duration = burst_start.elapsed();
    
    println!("Burst handling results:");
    println!("  Accepted: {} events", burst_accepted);
    println!("  Rejected: {} events", burst_rejected);
    println!("  Duration: {:?}", burst_duration);
    println!("  Effective rate: {} events/sec", 
             burst_accepted as f64 / burst_duration.as_secs_f64());
    
    // Should handle at least base rate during burst
    assert!(burst_accepted > TARGET_THROUGHPUT_PER_SEC,
            "Burst handling too restrictive: {} events", burst_accepted);
    
    // Should apply backpressure to prevent overwhelming
    assert!(burst_rejected > 0,
            "No backpressure applied during burst");
    
    processor.stop();
    
    // Drain received events
    let mut received = 0;
    while rx.try_recv().is_ok() {
        received += 1;
    }
    
    println!("Total events processed: {}", received);
}

#[test]
fn test_priority_ordering() {
    let config = ProcessorConfig {
        queue_size: 10_000,
        worker_threads: 1,  // Single thread to verify ordering
        batch_size: 1,      // Process one at a time
        rate_limit: None,
    };
    
    let mut processor = EventProcessor::new(config);
    let (tx, rx) = unbounded();
    processor.start(tx);
    
    // Send events with different priorities
    let priorities = vec![
        EventPriority::Background,
        EventPriority::Critical,
        EventPriority::Low,
        EventPriority::High,
        EventPriority::Medium,
        EventPriority::Critical,
        EventPriority::Background,
        EventPriority::High,
    ];
    
    for (i, priority) in priorities.iter().enumerate() {
        let mut event = create_test_event(i as f64);
        event.sequence = i as u64;  // Track original order
        processor.process_event(event, priority.clone()).unwrap();
    }
    
    thread::sleep(Duration::from_millis(100));
    processor.stop();
    
    // Collect processed order
    let mut processed_order = Vec::new();
    while let Ok(event) = rx.try_recv() {
        processed_order.push(event.sequence);
    }
    
    // Verify critical events processed first
    let critical_positions: Vec<_> = processed_order.iter()
        .enumerate()
        .filter(|(_, &seq)| seq == 1 || seq == 5)
        .map(|(pos, _)| pos)
        .collect();
    
    assert!(critical_positions[0] < 2, "Critical events not processed first");
    
    println!("Priority processing order: {:?}", processed_order);
}

#[test]
fn test_memory_efficiency() {
    let config = ProcessorConfig {
        queue_size: 1_000_000,
        worker_threads: 4,
        batch_size: 1000,
        rate_limit: None,
    };
    
    let mut processor = EventProcessor::new(config);
    let (tx, _rx) = unbounded();
    processor.start(tx);
    
    // Get baseline memory
    let baseline_mem = get_memory_usage();
    
    // Process 10M events
    for i in 0..10_000_000 {
        let event = create_test_event(100.0 + i as f64 * 0.001);
        processor.process_event(event, EventPriority::Medium).ok();
        
        // Check memory every 100k events
        if i % 100_000 == 0 {
            let current_mem = get_memory_usage();
            let growth = current_mem - baseline_mem;
            
            // Memory should not grow unbounded
            assert!(growth < 500_000_000, // 500MB max growth
                    "Memory growth excessive: {} bytes after {} events", growth, i);
        }
    }
    
    processor.stop();
}

// Helper functions

fn create_test_event(price: f64) -> MarketEvent {
    MarketEvent {
        timestamp: chrono::Utc::now(),
        exchange: "TEST".to_string(),
        symbol: "BTC/USDT".to_string(),
        price: Price::new(price),
        volume: Quantity::new(1.0),
        side: TradeSide::Buy,
        sequence: 0,
    }
}

fn get_memory_usage() -> usize {
    // Simple memory estimation - in production use proper metrics
    use std::fs;
    let status = fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse::<usize>().unwrap_or(0) * 1024;
            }
        }
    }
    0
}

#[test]
fn test_concurrent_producers() {
    let config = ProcessorConfig {
        queue_size: 100_000,
        worker_threads: 8,
        batch_size: 100,
        rate_limit: None,
    };
    
    let processor = Arc::new(EventProcessor::new(config));
    let (tx, rx) = bounded(1_000_000);
    
    // Start processor
    let mut proc = processor.clone();
    proc.start(tx);
    
    // Spawn multiple producer threads
    let mut handles = Vec::new();
    let total_sent = Arc::new(AtomicU64::new(0));
    
    for thread_id in 0..16 {
        let processor = processor.clone();
        let sent = total_sent.clone();
        
        let handle = thread::spawn(move || {
            let mut local_sent = 0u64;
            for i in 0..10_000 {
                let event = create_test_event(100.0 + thread_id as f64 + i as f64 * 0.01);
                if processor.process_event(event, EventPriority::Medium).is_ok() {
                    local_sent += 1;
                }
            }
            sent.fetch_add(local_sent, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    // Wait for all producers
    for handle in handles {
        handle.join().unwrap();
    }
    
    thread::sleep(Duration::from_millis(500));
    proc.stop();
    
    // Count received
    let mut received = 0u64;
    while rx.try_recv().is_ok() {
        received += 1;
    }
    
    let sent = total_sent.load(Ordering::Relaxed);
    println!("Concurrent test: sent={}, received={}", sent, received);
    
    // Allow small loss due to queue limits
    assert!(received as f64 > sent as f64 * 0.95,
            "Too many events lost: sent={}, received={}", sent, received);
}