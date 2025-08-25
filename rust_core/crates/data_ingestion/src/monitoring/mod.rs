use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct ProducerMetrics {
    queuing_latency_us: Arc<AtomicU64>,
    throughput_events: Arc<AtomicU64>,
    throughput_bytes: Arc<AtomicU64>,
}

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

pub use crate::sinks::clickhouse_sink::ClickHouseMetrics;