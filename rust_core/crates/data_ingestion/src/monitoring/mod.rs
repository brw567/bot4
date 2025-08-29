use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

/// Metrics collector trait for unified metrics interface
pub trait MetricsCollector: Send + Sync {
    fn record(&self, value: f64);
    fn increment(&self);
    fn get_value(&self) -> f64;
}

/// Simple counter implementation
/// TODO: Add docs
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
}

impl MetricsCollector for Counter {
    fn record(&self, value: f64) {
        self.value.store(value as u64, Ordering::Relaxed);
    }
    
    fn increment(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
    
    fn get_value(&self) -> f64 {
        self.value.load(Ordering::Relaxed) as f64
    }
}

/// Histogram implementation for distribution tracking
/// TODO: Add docs
pub struct Histogram {
    values: Arc<RwLock<Vec<f64>>>,
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            values: Arc::new(RwLock::new(Vec::with_capacity(1000))),
        }
    }
    
    pub fn percentile(&self, p: f64) -> f64 {
        let mut values = self.values.read().clone();
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((values.len() - 1) as f64 * p) as usize;
        values[idx]
    }
}

impl MetricsCollector for Histogram {
    fn record(&self, value: f64) {
        let mut values = self.values.write();
        values.push(value);
        // Keep last 10k samples for percentile calculations
        if values.len() > 10000 {
            values.drain(0..5000);
        }
    }
    
    fn increment(&self) {
        self.record(1.0);
    }
    
    fn get_value(&self) -> f64 {
        let values = self.values.read();
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
}

/// Register a histogram metric
/// TODO: Add docs
pub fn register_histogram(_name: &str) -> Arc<dyn MetricsCollector> {
    Arc::new(Histogram::new())
}

/// Register a counter metric
/// TODO: Add docs
pub fn register_counter(_name: &str) -> Arc<dyn MetricsCollector> {
    Arc::new(Counter::new())
}

/// TODO: Add docs
// ELIMINATED: pub struct ProducerMetrics {
// ELIMINATED:     queuing_latency_us: Arc<AtomicU64>,
// ELIMINATED:     throughput_events: Arc<AtomicU64>,
// ELIMINATED:     throughput_bytes: Arc<AtomicU64>,
// ELIMINATED: }

impl ProducerMetrics {
    pub fn new() -> Self {
        Self {
            queuing_latency_us: Arc::new(AtomicU64::new(0)),
            throughput_events: Arc::new(AtomicU64::new(0)),
            throughput_bytes: Arc::new(AtomicU64::new(0)),
        }
    }
    
    pub fn record_queuing_latency(&self, latency: Duration) {
        self.queuing_latency_us.store(latency.as_micros() as u64, Ordering::Relaxed);
    }
    
    pub fn update_throughput(&self, events_per_sec: u64, bytes_per_sec: u64) {
        self.throughput_events.store(events_per_sec, Ordering::Relaxed);
        self.throughput_bytes.store(bytes_per_sec, Ordering::Relaxed);
    }
}

/// TODO: Add docs
pub struct ConsumerMetrics {
    events_processed: Arc<AtomicU64>,
    processing_latency_us: Arc<AtomicU64>,
}

impl ConsumerMetrics {
    pub fn new() -> Self {
        Self {
            events_processed: Arc::new(AtomicU64::new(0)),
            processing_latency_us: Arc::new(AtomicU64::new(0)),
        }
    }
}

// ClickHouse metrics moved here to avoid circular dependency
/// TODO: Add docs
pub struct ClickHouseMetrics {
    pub writes_total: Arc<dyn MetricsCollector>,
    pub write_latency_ms: Arc<dyn MetricsCollector>,
    pub errors_total: Arc<dyn MetricsCollector>,
    pub bytes_written: Arc<dyn MetricsCollector>,
    pub buffer_size: Arc<AtomicU64>,
}

impl Default for ClickHouseMetrics {
    fn default() -> Self {
        Self {
            writes_total: register_counter("clickhouse_writes_total"),
            write_latency_ms: register_histogram("clickhouse_write_latency_ms"),
            errors_total: register_counter("clickhouse_errors_total"),
            bytes_written: register_counter("clickhouse_bytes_written"),
            buffer_size: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl ClickHouseMetrics {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_write(&self, latency_ms: f64, bytes: usize) {
        self.writes_total.increment();
        self.write_latency_ms.record(latency_ms);
        self.bytes_written.record(bytes as f64);
    }
    
    pub fn record_error(&self) {
        self.errors_total.increment();
    }
    
    pub fn update_buffer_size(&self, size: usize) {
        self.buffer_size.store(size as u64, Ordering::Relaxed);
    }
}