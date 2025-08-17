// Contention Benchmarks for Circuit Breaker
// Sophia Test Requirement #3: Performance under high contention (64-256 threads)
// Validates p99/p99.9 latencies remain within targets under load

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::sync::atomic::{AtomicU64, Ordering};

use infrastructure::circuit_breaker::{
    GlobalCircuitBreaker, CircuitConfig, SystemClock, CircuitState
};

/// Benchmark circuit breaker under various thread counts
fn bench_circuit_breaker_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker_contention");
    group.measurement_time(Duration::from_secs(30));
    
    // Test with different thread counts
    for thread_count in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        group.throughput(Throughput::Elements(thread_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("acquire_release", thread_count),
            &thread_count,
            |b, &threads| {
                let clock = Arc::new(SystemClock);
                let config = Arc::new(CircuitConfig::default());
                let breaker = Arc::new(GlobalCircuitBreaker::new(
                    clock, 
                    config.clone(), 
                    Some(1000)  // Bounded event channel
                ));
                
                b.iter_custom(|iters| {
                    let mut handles = Vec::new();
                    let start = Instant::now();
                    let iterations_per_thread = iters / threads as u64;
                    
                    for thread_id in 0..threads {
                        let breaker = breaker.clone();
                        handles.push(thread::spawn(move || {
                            for i in 0..iterations_per_thread {
                                let component = format!("comp-{}", thread_id % 10);
                                if let Ok(guard) = breaker.acquire(&component) {
                                    // Simulate some work
                                    black_box(i);
                                    // Record outcome
                                    if i % 10 == 0 {
                                        guard.record(infrastructure::circuit_breaker::Outcome::Failure);
                                    } else {
                                        guard.record(infrastructure::circuit_breaker::Outcome::Success);
                                    }
                                }
                            }
                        }));
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    start.elapsed()
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark latency distribution under contention
fn bench_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_distribution");
    
    for thread_count in [64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("p99_p999", thread_count),
            &thread_count,
            |b, &threads| {
                let clock = Arc::new(SystemClock);
                let config = Arc::new(CircuitConfig::default());
                let breaker = Arc::new(GlobalCircuitBreaker::new(
                    clock,
                    config.clone(),
                    Some(1000)
                ));
                
                // Shared latency collection
                let latencies = Arc::new(parking_lot::Mutex::new(Vec::new()));
                
                b.iter_custom(|_iters| {
                    let mut handles = Vec::new();
                    let breaker = breaker.clone();
                    let latencies = latencies.clone();
                    
                    // Run for fixed duration
                    let run_duration = Duration::from_secs(5);
                    let start_time = Instant::now();
                    
                    for thread_id in 0..threads {
                        let breaker = breaker.clone();
                        let latencies = latencies.clone();
                        
                        handles.push(thread::spawn(move || {
                            let mut local_latencies = Vec::new();
                            
                            while start_time.elapsed() < run_duration {
                                let op_start = Instant::now();
                                
                                let component = format!("comp-{}", thread_id % 10);
                                if let Ok(guard) = breaker.acquire(&component) {
                                    guard.record(infrastructure::circuit_breaker::Outcome::Success);
                                }
                                
                                let latency = op_start.elapsed();
                                local_latencies.push(latency.as_nanos() as u64);
                            }
                            
                            // Add to global collection
                            latencies.lock().extend(local_latencies);
                        }));
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    // Calculate percentiles
                    let mut all_latencies = latencies.lock().clone();
                    all_latencies.sort_unstable();
                    
                    if !all_latencies.is_empty() {
                        let p50_idx = all_latencies.len() / 2;
                        let p99_idx = all_latencies.len() * 99 / 100;
                        let p999_idx = all_latencies.len() * 999 / 1000;
                        
                        let p50 = all_latencies[p50_idx];
                        let p99 = all_latencies[p99_idx.min(all_latencies.len() - 1)];
                        let p999 = all_latencies[p999_idx.min(all_latencies.len() - 1)];
                        
                        println!("\n{} threads latency distribution:", threads);
                        println!("  p50:  {} ns", p50);
                        println!("  p99:  {} ns", p99);
                        println!("  p99.9: {} ns", p999);
                        
                        // Verify targets
                        assert!(p99 < 1_000_000, "p99 latency {} exceeds 1ms", p99);
                        assert!(p999 < 10_000_000, "p99.9 latency {} exceeds 10ms", p999);
                    }
                    
                    start_time.elapsed()
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark hot path with maximum contention
fn bench_hot_path_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_contention");
    
    group.bench_function("256_threads_single_component", |b| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig::default());
        let breaker = Arc::new(GlobalCircuitBreaker::new(
            clock,
            config.clone(),
            None  // No event channel for max performance
        ));
        
        // Pre-warm the component
        let component = breaker.component("hot");
        
        b.iter_custom(|iters| {
            let mut handles = Vec::new();
            let start = Instant::now();
            let iterations_per_thread = iters / 256;
            
            for _ in 0..256 {
                let component = component.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..iterations_per_thread {
                        // Direct component access (hottest path)
                        if let Ok(guard) = component.try_acquire("hot") {
                            black_box(guard);
                        }
                    }
                }));
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

/// Benchmark state transition storms
fn bench_state_transition_storm(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transition_storm");
    
    group.bench_function("rapid_open_close_cycles", |b| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig {
            consecutive_failures_threshold: 2,
            half_open_required_successes: 2,
            min_calls: 5,
            ..Default::default()
        });
        
        let breaker = Arc::new(GlobalCircuitBreaker::new(
            clock,
            config.clone(),
            Some(10000)  // Large event buffer
        ));
        
        b.iter_custom(|_iters| {
            let mut handles = Vec::new();
            let start = Instant::now();
            
            // Create threads that cause rapid state changes
            for thread_id in 0..64 {
                let breaker = breaker.clone();
                handles.push(thread::spawn(move || {
                    let component = format!("storm-{}", thread_id % 4);
                    
                    for cycle in 0..10 {
                        // Cause failures to trip breaker
                        for _ in 0..3 {
                            if let Ok(guard) = breaker.acquire(&component) {
                                guard.record(infrastructure::circuit_breaker::Outcome::Failure);
                            }
                        }
                        
                        // Wait briefly (simulate cooldown)
                        thread::sleep(Duration::from_micros(100));
                        
                        // Cause successes to close breaker
                        for _ in 0..3 {
                            if let Ok(guard) = breaker.acquire(&component) {
                                guard.record(infrastructure::circuit_breaker::Outcome::Success);
                            }
                        }
                    }
                }));
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

/// Benchmark memory usage under load
fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    
    group.bench_function("many_components_high_load", |b| {
        b.iter_custom(|_iters| {
            let clock = Arc::new(SystemClock);
            let config = Arc::new(CircuitConfig::default());
            let breaker = Arc::new(GlobalCircuitBreaker::new(
                clock,
                config.clone(),
                Some(1000)
            ));
            
            let mut handles = Vec::new();
            let start = Instant::now();
            
            // Create many components accessed by many threads
            for thread_id in 0..128 {
                let breaker = breaker.clone();
                handles.push(thread::spawn(move || {
                    for component_id in 0..100 {
                        let component = format!("comp-{}-{}", thread_id, component_id);
                        
                        for _ in 0..10 {
                            if let Ok(guard) = breaker.acquire(&component) {
                                guard.record(infrastructure::circuit_breaker::Outcome::Success);
                            }
                        }
                    }
                }));
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let elapsed = start.elapsed();
            
            // Verify we created many components
            // In production would check actual memory usage
            
            elapsed
        });
    });
    
    group.finish();
}

/// Validate no starvation under extreme contention
fn bench_fairness(c: &mut Criterion) {
    let mut group = c.benchmark_group("fairness");
    
    group.bench_function("no_starvation_256_threads", |b| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig::default());
        let breaker = Arc::new(GlobalCircuitBreaker::new(
            clock,
            config.clone(),
            None
        ));
        
        // Track acquisitions per thread
        let acquisitions: Vec<Arc<AtomicU64>> = (0..256)
            .map(|_| Arc::new(AtomicU64::new(0)))
            .collect();
        
        b.iter_custom(|_iters| {
            let mut handles = Vec::new();
            let start = Instant::now();
            let run_duration = Duration::from_secs(2);
            
            for thread_id in 0..256 {
                let breaker = breaker.clone();
                let counter = acquisitions[thread_id].clone();
                
                handles.push(thread::spawn(move || {
                    let start = Instant::now();
                    
                    while start.elapsed() < run_duration {
                        if let Ok(guard) = breaker.acquire("shared") {
                            counter.fetch_add(1, Ordering::Relaxed);
                            guard.record(infrastructure::circuit_breaker::Outcome::Success);
                        }
                    }
                }));
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            // Check fairness - no thread should be starved
            let counts: Vec<u64> = acquisitions.iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            
            let min = *counts.iter().min().unwrap();
            let max = *counts.iter().max().unwrap();
            let avg = counts.iter().sum::<u64>() / counts.len() as u64;
            
            println!("\nFairness stats (256 threads):");
            println!("  Min acquisitions: {}", min);
            println!("  Max acquisitions: {}", max);
            println!("  Avg acquisitions: {}", avg);
            println!("  Max/Min ratio: {:.2}", max as f64 / min.max(1) as f64);
            
            // No thread should be completely starved
            assert!(min > 0, "Thread starvation detected");
            
            // Ratio shouldn't be too extreme
            assert!(
                max as f64 / min.max(1) as f64 < 100.0,
                "Unfair acquisition distribution"
            );
            
            start.elapsed()
        });
    });
    
    group.finish();
}

criterion_group!(
    name = contention_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(30));
    targets = 
        bench_circuit_breaker_contention,
        bench_latency_distribution,
        bench_hot_path_contention,
        bench_state_transition_storm,
        bench_memory_pressure,
        bench_fairness
);

criterion_main!(contention_benches);