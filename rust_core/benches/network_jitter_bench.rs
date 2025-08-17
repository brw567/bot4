// Network Jitter Simulation Benchmarks
// Nexus Minor Improvement #1: Realistic network conditions
// Simulates real exchange network latency and jitter

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use std::sync::Arc;
use rand::{Rng, thread_rng};
use rand_distr::{Normal, Distribution};

// Simulate different network conditions
#[derive(Debug, Clone)]
pub struct NetworkCondition {
    pub name: String,
    pub base_latency_ms: f64,
    pub jitter_ms: f64,
    pub packet_loss_rate: f64,
    pub burst_delay_ms: Option<f64>,
}

impl NetworkCondition {
    pub fn perfect() -> Self {
        Self {
            name: "Perfect".to_string(),
            base_latency_ms: 1.0,
            jitter_ms: 0.1,
            packet_loss_rate: 0.0,
            burst_delay_ms: None,
        }
    }
    
    pub fn good() -> Self {
        Self {
            name: "Good".to_string(),
            base_latency_ms: 10.0,
            jitter_ms: 2.0,
            packet_loss_rate: 0.001,
            burst_delay_ms: None,
        }
    }
    
    pub fn average() -> Self {
        Self {
            name: "Average".to_string(),
            base_latency_ms: 50.0,
            jitter_ms: 10.0,
            packet_loss_rate: 0.01,
            burst_delay_ms: Some(100.0),
        }
    }
    
    pub fn poor() -> Self {
        Self {
            name: "Poor".to_string(),
            base_latency_ms: 150.0,
            jitter_ms: 50.0,
            packet_loss_rate: 0.05,
            burst_delay_ms: Some(500.0),
        }
    }
    
    pub fn exchange_outage() -> Self {
        Self {
            name: "Exchange Outage".to_string(),
            base_latency_ms: 1000.0,
            jitter_ms: 500.0,
            packet_loss_rate: 0.3,
            burst_delay_ms: Some(5000.0),
        }
    }
    
    /// Simulate network delay with jitter
    pub fn simulate_delay(&self) -> Duration {
        let mut rng = thread_rng();
        
        // Check for packet loss
        if rng.gen::<f64>() < self.packet_loss_rate {
            // Simulate timeout/retry
            return Duration::from_millis(5000);
        }
        
        // Calculate jittered delay
        let normal = Normal::new(self.base_latency_ms, self.jitter_ms).unwrap();
        let delay_ms = normal.sample(&mut rng).max(0.0);
        
        // Occasionally add burst delay (network congestion)
        let final_delay = if let Some(burst) = self.burst_delay_ms {
            if rng.gen::<f64>() < 0.05 {  // 5% chance of burst
                delay_ms + burst
            } else {
                delay_ms
            }
        } else {
            delay_ms
        };
        
        Duration::from_secs_f64(final_delay / 1000.0)
    }
}

/// Simulate order submission with network conditions
fn simulate_order_submission(condition: &NetworkCondition) -> Result<Duration, &'static str> {
    let start = Instant::now();
    
    // Simulate pre-trade checks (local)
    std::thread::sleep(Duration::from_micros(10));  // Our 10μs risk check
    
    // Simulate network round-trip to exchange
    let network_delay = condition.simulate_delay();
    std::thread::sleep(network_delay);
    
    // Simulate exchange processing
    std::thread::sleep(Duration::from_millis(5));
    
    // Simulate response network delay
    let return_delay = condition.simulate_delay();
    std::thread::sleep(return_delay);
    
    Ok(start.elapsed())
}

/// Benchmark order submission under different network conditions
fn bench_order_submission_with_jitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_submission_network");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);  // Reduced due to sleep operations
    
    let conditions = vec![
        NetworkCondition::perfect(),
        NetworkCondition::good(),
        NetworkCondition::average(),
        NetworkCondition::poor(),
    ];
    
    for condition in conditions {
        group.bench_with_input(
            BenchmarkId::new("network", &condition.name),
            &condition,
            |b, cond| {
                b.iter(|| {
                    let result = simulate_order_submission(black_box(cond));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark WebSocket message processing with jitter
fn bench_websocket_with_jitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("websocket_network");
    
    let conditions = vec![
        NetworkCondition::perfect(),
        NetworkCondition::good(),
        NetworkCondition::average(),
    ];
    
    for condition in conditions {
        group.bench_with_input(
            BenchmarkId::new("throughput", &condition.name),
            &condition,
            |b, cond| {
                b.iter(|| {
                    // Simulate processing 100 messages
                    let mut total_latency = Duration::ZERO;
                    for _ in 0..100 {
                        // Local processing (our <1ms target)
                        std::thread::sleep(Duration::from_micros(500));
                        
                        // Add network jitter
                        total_latency += cond.simulate_delay();
                    }
                    black_box(total_latency);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark recovery from exchange outage
fn bench_outage_recovery(c: &mut Criterion) {
    c.bench_function("exchange_outage_recovery", |b| {
        let outage = NetworkCondition::exchange_outage();
        let good = NetworkCondition::good();
        
        b.iter(|| {
            // Simulate outage detection
            let start = Instant::now();
            
            // Try 3 times with outage conditions
            for _ in 0..3 {
                let _ = simulate_order_submission(&outage);
                if start.elapsed() > Duration::from_secs(5) {
                    break;  // Circuit breaker trips
                }
            }
            
            // Recovery with exponential backoff
            std::thread::sleep(Duration::from_secs(1));
            
            // Test recovery with good conditions
            let recovery = simulate_order_submission(&good);
            
            black_box(recovery);
        });
    });
}

/// Benchmark latency percentiles under realistic conditions
fn bench_realistic_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_latency");
    group.measurement_time(Duration::from_secs(60));
    
    group.bench_function("production_conditions", |b| {
        let condition = NetworkCondition::average();
        let mut latencies = Vec::new();
        
        b.iter(|| {
            let latency = simulate_order_submission(&condition).unwrap();
            latencies.push(latency.as_millis());
            black_box(latency);
        });
        
        // Calculate percentiles
        if !latencies.is_empty() {
            latencies.sort_unstable();
            let p50 = latencies[latencies.len() / 2];
            let p95 = latencies[latencies.len() * 95 / 100];
            let p99 = latencies[latencies.len() * 99 / 100];
            
            println!("Latency Distribution:");
            println!("  p50: {}ms", p50);
            println!("  p95: {}ms", p95);
            println!("  p99: {}ms", p99);
        }
    });
    
    group.finish();
}

criterion_group!(
    name = network_benches;
    config = Criterion::default();
    targets = 
        bench_order_submission_with_jitter,
        bench_websocket_with_jitter,
        bench_outage_recovery,
        bench_realistic_latency_distribution
);

criterion_main!(network_benches);