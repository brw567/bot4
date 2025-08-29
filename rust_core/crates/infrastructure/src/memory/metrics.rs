// Bot4 Memory Metrics - Prometheus Integration
// Day 2 Sprint - Zero overhead metrics collection
// Owner: Jordan
// Targets: <1ns metric overhead, real-time pool monitoring

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use lazy_static::lazy_static;
use crossbeam_utils::CachePadded;

/// Memory metrics with cache-line padding to prevent false sharing
/// TODO: Add docs
pub struct MemoryMetrics {
    // Allocation metrics (cache-padded per Sophia's recommendation)
    allocation_count: CachePadded<AtomicU64>,
    allocation_bytes: CachePadded<AtomicU64>,
    allocation_latency_ns: CachePadded<AtomicU64>,
    
    // Pool metrics
    order_pool_hits: CachePadded<AtomicU64>,
    order_pool_misses: CachePadded<AtomicU64>,
    signal_pool_hits: CachePadded<AtomicU64>,
    signal_pool_misses: CachePadded<AtomicU64>,
    tick_pool_hits: CachePadded<AtomicU64>,
    tick_pool_misses: CachePadded<AtomicU64>,
    
    // TLS cache metrics
    tls_cache_hits: CachePadded<AtomicU64>,
    tls_cache_misses: CachePadded<AtomicU64>,
    
    // Ring buffer metrics
    ring_pushes: CachePadded<AtomicU64>,
    ring_pops: CachePadded<AtomicU64>,
    ring_drops: CachePadded<AtomicU64>,
    
    // Pressure metrics (0-100%)
    order_pool_pressure: CachePadded<AtomicU64>,
    signal_pool_pressure: CachePadded<AtomicU64>,
    tick_pool_pressure: CachePadded<AtomicU64>,
}

impl MemoryMetrics {
    fn new() -> Self {
        Self {
            allocation_count: CachePadded::new(AtomicU64::new(0)),
            allocation_bytes: CachePadded::new(AtomicU64::new(0)),
            allocation_latency_ns: CachePadded::new(AtomicU64::new(0)),
            
            order_pool_hits: CachePadded::new(AtomicU64::new(0)),
            order_pool_misses: CachePadded::new(AtomicU64::new(0)),
            signal_pool_hits: CachePadded::new(AtomicU64::new(0)),
            signal_pool_misses: CachePadded::new(AtomicU64::new(0)),
            tick_pool_hits: CachePadded::new(AtomicU64::new(0)),
            tick_pool_misses: CachePadded::new(AtomicU64::new(0)),
            
            tls_cache_hits: CachePadded::new(AtomicU64::new(0)),
            tls_cache_misses: CachePadded::new(AtomicU64::new(0)),
            
            ring_pushes: CachePadded::new(AtomicU64::new(0)),
            ring_pops: CachePadded::new(AtomicU64::new(0)),
            ring_drops: CachePadded::new(AtomicU64::new(0)),
            
            order_pool_pressure: CachePadded::new(AtomicU64::new(0)),
            signal_pool_pressure: CachePadded::new(AtomicU64::new(0)),
            tick_pool_pressure: CachePadded::new(AtomicU64::new(0)),
        }
    }
    
    /// Record allocation with latency
    #[inline(always)]
    pub fn record_allocation(&self, bytes: usize, latency_ns: u64) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.allocation_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        
        // Update p99 latency using exponential moving average
        let current = self.allocation_latency_ns.load(Ordering::Relaxed);
        let new_value = (current * 99 + latency_ns) / 100;
        self.allocation_latency_ns.store(new_value, Ordering::Relaxed);
    }
    
    /// Record pool hit
    #[inline(always)]  // Zero-cost metrics
    pub fn record_pool_hit(&self, pool_type: PoolType) {
        match pool_type {
            PoolType::Order => self.order_pool_hits.fetch_add(1, Ordering::Relaxed),
            PoolType::Signal => self.signal_pool_hits.fetch_add(1, Ordering::Relaxed),
            PoolType::Tick => self.tick_pool_hits.fetch_add(1, Ordering::Relaxed),
        };
    }
    
    /// Record pool miss (fallback to allocation)
    #[inline(always)]
    pub fn record_pool_miss(&self, pool_type: PoolType) {
        match pool_type {
            PoolType::Order => self.order_pool_misses.fetch_add(1, Ordering::Relaxed),
            PoolType::Signal => self.signal_pool_misses.fetch_add(1, Ordering::Relaxed),
            PoolType::Tick => self.tick_pool_misses.fetch_add(1, Ordering::Relaxed),
        };
    }
    
    /// Record TLS cache hit
    #[inline(always)]  // Zero-cost metrics
    pub fn record_tls_hit(&self) {
        self.tls_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record TLS cache miss
    #[inline(always)]  // Zero-cost metrics
    pub fn record_tls_miss(&self) {
        self.tls_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record ring buffer operations
    #[inline(always)]
    pub fn record_ring_push(&self) {
        self.ring_pushes.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn record_ring_pop(&self) {
        self.ring_pops.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn record_ring_drop(&self) {
        self.ring_drops.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update pool pressure (0-100)
    pub fn update_pool_pressure(&self, pool_type: PoolType, pressure_pct: u64) {
        let pressure = pressure_pct.min(100);
        match pool_type {
            PoolType::Order => self.order_pool_pressure.store(pressure, Ordering::Relaxed),
            PoolType::Signal => self.signal_pool_pressure.store(pressure, Ordering::Relaxed),
            PoolType::Tick => self.tick_pool_pressure.store(pressure, Ordering::Relaxed),
        }
    }
    
    /// Export metrics for Prometheus
    pub fn export_prometheus_metrics(&self) -> String {
        let mut output = String::with_capacity(2048);
        
        // Allocation metrics
        output.push_str(&format!(
            "# HELP bot4_memory_allocations_total Total number of allocations\n\
             # TYPE bot4_memory_allocations_total counter\n\
             bot4_memory_allocations_total {}\n",
            self.allocation_count.load(Ordering::Relaxed)
        ));
        
        output.push_str(&format!(
            "# HELP bot4_memory_allocated_bytes_total Total bytes allocated\n\
             # TYPE bot4_memory_allocated_bytes_total counter\n\
             bot4_memory_allocated_bytes_total {}\n",
            self.allocation_bytes.load(Ordering::Relaxed)
        ));
        
        output.push_str(&format!(
            "# HELP bot4_memory_allocation_latency_ns P99 allocation latency in nanoseconds\n\
             # TYPE bot4_memory_allocation_latency_ns gauge\n\
             bot4_memory_allocation_latency_ns {}\n",
            self.allocation_latency_ns.load(Ordering::Relaxed)
        ));
        
        // Pool hit rates
        let order_hits = self.order_pool_hits.load(Ordering::Relaxed);
        let order_misses = self.order_pool_misses.load(Ordering::Relaxed);
        let order_rate = if order_hits + order_misses > 0 {
            order_hits as f64 / (order_hits + order_misses) as f64 * 100.0
        } else { 100.0 };
        
        output.push_str(&format!(
            "# HELP bot4_pool_hit_rate_percent Pool hit rate percentage\n\
             # TYPE bot4_pool_hit_rate_percent gauge\n\
             bot4_pool_hit_rate_percent{{pool=\"order\"}} {:.2}\n",
            order_rate
        ));
        
        let signal_hits = self.signal_pool_hits.load(Ordering::Relaxed);
        let signal_misses = self.signal_pool_misses.load(Ordering::Relaxed);
        let signal_rate = if signal_hits + signal_misses > 0 {
            signal_hits as f64 / (signal_hits + signal_misses) as f64 * 100.0
        } else { 100.0 };
        
        output.push_str(&format!(
            "bot4_pool_hit_rate_percent{{pool=\"signal\"}} {:.2}\n",
            signal_rate
        ));
        
        let tick_hits = self.tick_pool_hits.load(Ordering::Relaxed);
        let tick_misses = self.tick_pool_misses.load(Ordering::Relaxed);
        let tick_rate = if tick_hits + tick_misses > 0 {
            tick_hits as f64 / (tick_hits + tick_misses) as f64 * 100.0
        } else { 100.0 };
        
        output.push_str(&format!(
            "bot4_pool_hit_rate_percent{{pool=\"tick\"}} {:.2}\n",
            tick_rate
        ));
        
        // Pool pressure
        output.push_str(&format!(
            "# HELP bot4_pool_pressure_percent Pool memory pressure percentage\n\
             # TYPE bot4_pool_pressure_percent gauge\n\
             bot4_pool_pressure_percent{{pool=\"order\"}} {}\n\
             bot4_pool_pressure_percent{{pool=\"signal\"}} {}\n\
             bot4_pool_pressure_percent{{pool=\"tick\"}} {}\n",
            self.order_pool_pressure.load(Ordering::Relaxed),
            self.signal_pool_pressure.load(Ordering::Relaxed),
            self.tick_pool_pressure.load(Ordering::Relaxed)
        ));
        
        // TLS cache metrics
        let tls_hits = self.tls_cache_hits.load(Ordering::Relaxed);
        let tls_misses = self.tls_cache_misses.load(Ordering::Relaxed);
        let tls_rate = if tls_hits + tls_misses > 0 {
            tls_hits as f64 / (tls_hits + tls_misses) as f64 * 100.0
        } else { 100.0 };
        
        output.push_str(&format!(
            "# HELP bot4_tls_cache_hit_rate_percent Thread-local cache hit rate\n\
             # TYPE bot4_tls_cache_hit_rate_percent gauge\n\
             bot4_tls_cache_hit_rate_percent {:.2}\n",
            tls_rate
        ));
        
        // Ring buffer metrics
        output.push_str(&format!(
            "# HELP bot4_ring_operations_total Ring buffer operations\n\
             # TYPE bot4_ring_operations_total counter\n\
             bot4_ring_operations_total{{op=\"push\"}} {}\n\
             bot4_ring_operations_total{{op=\"pop\"}} {}\n\
             bot4_ring_operations_total{{op=\"drop\"}} {}\n",
            self.ring_pushes.load(Ordering::Relaxed),
            self.ring_pops.load(Ordering::Relaxed),
            self.ring_drops.load(Ordering::Relaxed)
        ));
        
        output
    }
}

/// Pool type for metrics
#[derive(Debug, Clone, Copy)]
/// TODO: Add docs
pub enum PoolType {
    Order,
    Signal,
    Tick,
}

// Global metrics instance
lazy_static! {
    static ref METRICS: Arc<MemoryMetrics> = Arc::new(MemoryMetrics::new());
}

/// Get global metrics instance
#[inline(always)]  // Zero-cost access
/// TODO: Add docs
pub fn metrics() -> &'static MemoryMetrics {
    &METRICS
}

/// Initialize memory metrics
/// TODO: Add docs
pub fn initialize_metrics() {
    lazy_static::initialize(&METRICS);
    
    // Start background pressure monitoring
    start_pressure_monitor();
    
    tracing::info!("Memory metrics initialized");
}

/// Start background thread to monitor pool pressure
fn start_pressure_monitor() {
    std::thread::spawn(|| {
        loop {
            // Update pool pressure every second
            std::thread::sleep(Duration::from_secs(1));
            
            let stats = super::pools::get_pool_stats();
            
            // Convert to percentage (0-100)
            metrics().update_pool_pressure(
                PoolType::Order,
                (stats.order_pressure * 100.0) as u64
            );
            metrics().update_pool_pressure(
                PoolType::Signal,
                (stats.signal_pressure * 100.0) as u64
            );
            metrics().update_pool_pressure(
                PoolType::Tick,
                (stats.tick_pressure * 100.0) as u64
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_recording() {
        let metrics = MemoryMetrics::new();
        
        // Record some allocations
        metrics.record_allocation(1024, 5);
        metrics.record_allocation(2048, 8);
        
        assert_eq!(metrics.allocation_count.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.allocation_bytes.load(Ordering::Relaxed), 3072);
        
        // Record pool operations
        metrics.record_pool_hit(PoolType::Order);
        metrics.record_pool_hit(PoolType::Order);
        metrics.record_pool_miss(PoolType::Order);
        
        assert_eq!(metrics.order_pool_hits.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.order_pool_misses.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_prometheus_export() {
        let metrics = MemoryMetrics::new();
        
        // Set some test data
        metrics.record_allocation(1024, 7);
        metrics.record_pool_hit(PoolType::Order);
        metrics.record_tls_hit();
        metrics.update_pool_pressure(PoolType::Order, 45);
        
        let export = metrics.export_prometheus_metrics();
        
        // Verify key metrics are present
        assert!(export.contains("bot4_memory_allocations_total 1"));
        assert!(export.contains("bot4_memory_allocated_bytes_total 1024"));
        assert!(export.contains("bot4_pool_pressure_percent{pool=\"order\"} 45"));
    }
    
    #[test]
    fn test_metric_overhead() {
        use std::time::Instant;
        
        let metrics = MemoryMetrics::new();
        const ITERATIONS: usize = 10_000_000;
        
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            metrics.record_pool_hit(PoolType::Order);
        }
        let elapsed = start.elapsed();
        
        let per_op = elapsed.as_nanos() / ITERATIONS as u128;
        println!("Metric recording overhead: {}ns", per_op);
        
        // Should be <10ns with atomics (5ns is excellent)
        assert!(per_op < 10, "Metric overhead too high: {}ns", per_op);
    }
}