//! # Metrics - Performance monitoring for event bus
//!
//! Tracks latency, throughput, and queue depths with minimal overhead.
//! Uses lock-free counters to avoid impacting performance.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::sync::Arc;
use cache_padded::CachePadded;

/// Latency histogram bucket
#[derive(Debug)]
struct LatencyBucket {
    /// Upper bound in nanoseconds
    upper_bound: u64,
    /// Count of samples in this bucket
    count: AtomicU64,
}

/// Performance metrics collector
pub struct EventBusMetrics {
    /// Total events published
    pub events_published: CachePadded<AtomicU64>,
    /// Total events consumed
    pub events_consumed: CachePadded<AtomicU64>,
    /// Failed publishes
    pub publish_failures: CachePadded<AtomicU64>,
    /// Current ring buffer depth
    pub ring_buffer_depth: CachePadded<AtomicUsize>,
    /// Maximum ring buffer depth seen
    pub max_ring_buffer_depth: CachePadded<AtomicUsize>,
    /// Latency histogram buckets (in nanoseconds)
    latency_buckets: Vec<LatencyBucket>,
    /// P99 latency in nanoseconds
    pub p99_latency_ns: CachePadded<AtomicU64>,
    /// P50 latency in nanoseconds
    pub p50_latency_ns: CachePadded<AtomicU64>,
    /// Maximum latency seen
    pub max_latency_ns: CachePadded<AtomicU64>,
    /// Start time for throughput calculation
    start_time: Instant,
}

impl EventBusMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        // Create exponential latency buckets: 100ns, 200ns, 400ns, ..., 100ms
        let mut buckets = Vec::new();
        let mut bound = 100;  // Start at 100ns
        while bound <= 100_000_000 {  // Up to 100ms
            buckets.push(LatencyBucket {
                upper_bound: bound,
                count: AtomicU64::new(0),
            });
            bound *= 2;
        }
        
        Self {
            events_published: CachePadded::new(AtomicU64::new(0)),
            events_consumed: CachePadded::new(AtomicU64::new(0)),
            publish_failures: CachePadded::new(AtomicU64::new(0)),
            ring_buffer_depth: CachePadded::new(AtomicUsize::new(0)),
            max_ring_buffer_depth: CachePadded::new(AtomicUsize::new(0)),
            latency_buckets: buckets,
            p99_latency_ns: CachePadded::new(AtomicU64::new(0)),
            p50_latency_ns: CachePadded::new(AtomicU64::new(0)),
            max_latency_ns: CachePadded::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }
    
    /// Record a publish event
    pub fn record_publish(&self, latency_ns: u64) {
        self.events_published.fetch_add(1, Ordering::Relaxed);
        self.record_latency(latency_ns);
    }
    
    /// Record a consume event
    pub fn record_consume(&self) {
        self.events_consumed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a failed publish
    pub fn record_publish_failure(&self) {
        self.publish_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update ring buffer depth
    pub fn update_buffer_depth(&self, depth: usize) {
        self.ring_buffer_depth.store(depth, Ordering::Relaxed);
        
        // Update max if needed
        let mut current_max = self.max_ring_buffer_depth.load(Ordering::Relaxed);
        while depth > current_max {
            match self.max_ring_buffer_depth.compare_exchange_weak(
                current_max,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }
    
    /// Record latency sample
    fn record_latency(&self, latency_ns: u64) {
        // Update histogram
        for bucket in &self.latency_buckets {
            if latency_ns <= bucket.upper_bound {
                bucket.count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        
        // Update max latency
        let mut current_max = self.max_latency_ns.load(Ordering::Relaxed);
        while latency_ns > current_max {
            match self.max_latency_ns.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }
    
    /// Calculate percentiles from histogram
    pub fn calculate_percentiles(&self) {
        let total_count: u64 = self.latency_buckets
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        
        if total_count == 0 {
            return;
        }
        
        let p50_target = (total_count as f64 * 0.5) as u64;
        let p99_target = (total_count as f64 * 0.99) as u64;
        
        let mut cumulative = 0u64;
        let mut p50_set = false;
        let mut p99_set = false;
        
        for bucket in &self.latency_buckets {
            cumulative += bucket.count.load(Ordering::Relaxed);
            
            if !p50_set && cumulative >= p50_target {
                self.p50_latency_ns.store(bucket.upper_bound, Ordering::Relaxed);
                p50_set = true;
            }
            
            if !p99_set && cumulative >= p99_target {
                self.p99_latency_ns.store(bucket.upper_bound, Ordering::Relaxed);
                p99_set = true;
            }
            
            if p50_set && p99_set {
                break;
            }
        }
    }
    
    /// Get current throughput in events/second
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.events_published.load(Ordering::Relaxed) as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// Get metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.calculate_percentiles();
        
        MetricsSnapshot {
            events_published: self.events_published.load(Ordering::Relaxed),
            events_consumed: self.events_consumed.load(Ordering::Relaxed),
            publish_failures: self.publish_failures.load(Ordering::Relaxed),
            ring_buffer_depth: self.ring_buffer_depth.load(Ordering::Relaxed),
            max_ring_buffer_depth: self.max_ring_buffer_depth.load(Ordering::Relaxed),
            p50_latency_ns: self.p50_latency_ns.load(Ordering::Relaxed),
            p99_latency_ns: self.p99_latency_ns.load(Ordering::Relaxed),
            max_latency_ns: self.max_latency_ns.load(Ordering::Relaxed),
            throughput: self.throughput(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        self.events_published.store(0, Ordering::Relaxed);
        self.events_consumed.store(0, Ordering::Relaxed);
        self.publish_failures.store(0, Ordering::Relaxed);
        self.ring_buffer_depth.store(0, Ordering::Relaxed);
        self.max_ring_buffer_depth.store(0, Ordering::Relaxed);
        self.p50_latency_ns.store(0, Ordering::Relaxed);
        self.p99_latency_ns.store(0, Ordering::Relaxed);
        self.max_latency_ns.store(0, Ordering::Relaxed);
        
        for bucket in &self.latency_buckets {
            bucket.count.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for EventBusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub events_published: u64,
    pub events_consumed: u64,
    pub publish_failures: u64,
    pub ring_buffer_depth: usize,
    pub max_ring_buffer_depth: usize,
    pub p50_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub max_latency_ns: u64,
    pub throughput: f64,
}

impl MetricsSnapshot {
    /// Format metrics for display
    pub fn format(&self) -> String {
        format!(
            "EventBus Metrics:\n\
             Published: {} | Consumed: {} | Failures: {}\n\
             Buffer Depth: {} (max: {})\n\
             Latency: P50={:.0}ns, P99={:.0}ns, Max={:.0}ns\n\
             Throughput: {:.0} events/sec",
            self.events_published,
            self.events_consumed,
            self.publish_failures,
            self.ring_buffer_depth,
            self.max_ring_buffer_depth,
            self.p50_latency_ns,
            self.p99_latency_ns,
            self.max_latency_ns,
            self.throughput
        )
    }
}

/// Latency timer for measuring operation duration
pub struct LatencyTimer {
    start: Instant,
    metrics: Arc<EventBusMetrics>,
}

impl LatencyTimer {
    /// Start a new timer
    pub fn start(metrics: Arc<EventBusMetrics>) -> Self {
        Self {
            start: Instant::now(),
            metrics,
        }
    }
    
    /// Stop timer and record latency
    pub fn stop(self) {
        let latency_ns = self.start.elapsed().as_nanos() as u64;
        self.metrics.record_publish(latency_ns);
    }
}

/// Metrics reporter for periodic reporting
pub struct MetricsReporter {
    metrics: Arc<EventBusMetrics>,
    interval: Duration,
}

impl MetricsReporter {
    /// Create new reporter
    pub fn new(metrics: Arc<EventBusMetrics>, interval: Duration) -> Self {
        Self { metrics, interval }
    }
    
    /// Start reporting loop
    pub async fn start(self) {
        let mut interval = tokio::time::interval(self.interval);
        
        loop {
            interval.tick().await;
            
            let snapshot = self.metrics.snapshot();
            
            // Log metrics (could also send to monitoring system)
            log::info!("{}", snapshot.format());
            
            // Alert on high latency
            if snapshot.p99_latency_ns > 1_000_000 {  // > 1ms
                log::warn!("High P99 latency detected: {}μs", 
                          snapshot.p99_latency_ns / 1000);
            }
            
            // Alert on failures
            if snapshot.publish_failures > 0 {
                log::error!("Event publish failures: {}", snapshot.publish_failures);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_recording() {
        let metrics = EventBusMetrics::new();
        
        // Record some events
        metrics.record_publish(500);   // 500ns
        metrics.record_publish(1000);  // 1μs
        metrics.record_publish(2000);  // 2μs
        metrics.record_consume();
        metrics.record_consume();
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.events_published, 3);
        assert_eq!(snapshot.events_consumed, 2);
    }
    
    #[test]
    fn test_latency_percentiles() {
        let metrics = EventBusMetrics::new();
        
        // Add 100 samples
        for i in 0..100 {
            let latency = if i < 50 {
                500  // 50% at 500ns
            } else if i < 99 {
                1000  // 49% at 1μs
            } else {
                10000  // 1% at 10μs
            };
            metrics.record_publish(latency);
        }
        
        metrics.calculate_percentiles();
        
        // P50 should be around 1μs
        assert!(metrics.p50_latency_ns.load(Ordering::Relaxed) <= 1600);
        
        // P99 should be around 10μs
        assert!(metrics.p99_latency_ns.load(Ordering::Relaxed) >= 6400);
    }
    
    #[test]
    fn test_buffer_depth_tracking() {
        let metrics = EventBusMetrics::new();
        
        metrics.update_buffer_depth(100);
        metrics.update_buffer_depth(200);
        metrics.update_buffer_depth(150);
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.ring_buffer_depth, 150);
        assert_eq!(snapshot.max_ring_buffer_depth, 200);
    }
}