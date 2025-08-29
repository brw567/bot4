// Bot4 Observability - Metrics Module
// Day 1 Sprint - Wire metrics to Prometheus
// Owner: Avery
// Performance Target: <1µs metric recording

use prometheus::{
    Gauge, GaugeVec, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};
use lazy_static::lazy_static;
use std::sync::Once;
use std::time::Instant;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    // Decision Latency Metrics - CRITICAL PATH
    pub static ref DECISION_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "decision_latency_microseconds",
            "Decision making latency in microseconds"
        ).buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),  // µs buckets
        &["component", "decision_type"]
    ).expect("metric creation failed");
    
    // Risk Check Latency - CRITICAL PATH
    pub static ref RISK_CHECK_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "risk_check_latency_microseconds", 
            "Risk check latency in microseconds"
        ).buckets(vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),  // µs buckets
        &["check_type", "symbol"]
    ).expect("metric creation failed");
    
    // Order Pipeline Latency - CRITICAL PATH
    pub static ref ORDER_INTERNAL_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "order_internal_latency_microseconds",
            "Order internal processing latency in microseconds"
        ).buckets(vec![10.0, 20.0, 50.0, 100.0, 200.0, 500.0]),  // µs buckets
        &["order_type", "exchange"]
    ).expect("metric creation failed");
    
    // Circuit Breaker Metrics
    pub static ref CB_STATE: IntGaugeVec = IntGaugeVec::new(
        Opts::new("cb_state", "Circuit breaker state (0=closed, 1=half-open, 2=open)"),
        &["breaker_name"]
    ).expect("metric creation failed");
    
    pub static ref CB_TRIP_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("cb_trip_total", "Total circuit breaker trips"),
        &["breaker_name", "reason"]
    ).expect("metric creation failed");
    
    pub static ref CB_FAILURE_RATE: GaugeVec = GaugeVec::new(
        Opts::new("cb_failure_rate", "Current failure rate (0.0-1.0)"),
        &["breaker_name"]
    ).expect("metric creation failed");
    
    pub static ref CB_CHECK_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "cb_check_latency_nanoseconds",
            "Circuit breaker check latency in nanoseconds"
        ).buckets(vec![10.0, 20.0, 50.0, 100.0, 200.0]),  // ns buckets
        &["breaker_name"]
    ).expect("metric creation failed");
    
    // Throughput Metrics
    pub static ref OPERATIONS_TOTAL: IntCounter = IntCounter::new(
        "operations_total", "Total operations processed"
    ).expect("metric creation failed");
    
    pub static ref ORDERS_PROCESSED: IntCounterVec = IntCounterVec::new(
        Opts::new("orders_processed_total", "Total orders processed"),
        &["status", "exchange"]
    ).expect("metric creation failed");
    
    pub static ref ORDERS_RECEIVED: IntCounter = IntCounter::new(
        "orders_received_total", "Total orders received"
    ).expect("metric creation failed");
    
    // Memory Pool Metrics
    pub static ref MEMORY_POOL_AVAILABLE: IntGaugeVec = IntGaugeVec::new(
        Opts::new("memory_pool_available", "Available objects in pool"),
        &["pool_name"]
    ).expect("metric creation failed");
    
    pub static ref MEMORY_POOL_TOTAL: IntGaugeVec = IntGaugeVec::new(
        Opts::new("memory_pool_total", "Total capacity of pool"),
        &["pool_name"]
    ).expect("metric creation failed");
    
    pub static ref ALLOCATION_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "allocation_latency_nanoseconds",
            "Memory allocation latency in nanoseconds"
        ).buckets(vec![5.0, 10.0, 20.0, 50.0, 100.0]),  // ns buckets
        &["allocator", "size_class"]
    ).expect("metric creation failed");
    
    // Risk Metrics
    pub static ref MAX_DRAWDOWN_CURRENT: Gauge = Gauge::new(
        "max_drawdown_current", "Current drawdown percentage (0.0-1.0)"
    ).expect("metric creation failed");
    
    pub static ref MAX_POSITION_SIZE_RATIO: Gauge = Gauge::new(
        "max_position_size_ratio", "Maximum position size as ratio of portfolio"
    ).expect("metric creation failed");
    
    pub static ref PORTFOLIO_CORRELATION: GaugeVec = GaugeVec::new(
        Opts::new("portfolio_correlation_matrix", "Correlation between positions"),
        &["pair"]
    ).expect("metric creation failed");
    
    pub static ref RISK_CHECK_PASS: IntCounterVec = IntCounterVec::new(
        Opts::new("risk_check_pass_total", "Risk checks passed"),
        &["check_type"]
    ).expect("metric creation failed");
    
    pub static ref RISK_CHECK_FAIL: IntCounterVec = IntCounterVec::new(
        Opts::new("risk_check_fail_total", "Risk checks failed"),
        &["check_type", "reason"]
    ).expect("metric creation failed");
    
    // Order Queue Metrics
    pub static ref ORDER_QUEUE_DEPTH: IntGauge = IntGauge::new(
        "order_queue_depth", "Current order queue depth"
    ).expect("metric creation failed");
    
    pub static ref ORDERS_BY_TYPE: IntCounterVec = IntCounterVec::new(
        Opts::new("orders_by_type_total", "Orders by type"),
        &["type"]
    ).expect("metric creation failed");
    
    pub static ref ORDERS_SUCCESS: IntCounter = IntCounter::new(
        "orders_success_total", "Successful orders"
    ).expect("metric creation failed");
    
    pub static ref ORDERS_FAILED: IntCounter = IntCounter::new(
        "orders_failed_total", "Failed orders"
    ).expect("metric creation failed");
    
    // Exchange API Metrics
    pub static ref EXCHANGE_API_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "exchange_api_latency_milliseconds",
            "Exchange API latency in milliseconds"
        ).buckets(vec![10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]),
        &["exchange", "endpoint"]
    ).expect("metric creation failed");
    
    // Market Data Metrics
    pub static ref MARKET_DATA_LAST_UPDATE: GaugeVec = GaugeVec::new(
        Opts::new("market_data_last_update_timestamp", "Last market data update timestamp"),
        &["symbol", "exchange"]
    ).expect("metric creation failed");
    
    pub static ref ORDER_BOOK_DEPTH: IntGaugeVec = IntGaugeVec::new(
        Opts::new("order_book_depth_total", "Total order book depth"),
        &["symbol", "side"]
    ).expect("metric creation failed");
}

static INIT: Once = Once::new();

/// TODO: Add docs
pub fn init_metrics() {
    INIT.call_once(|| {
        // Register all metrics with the registry
        let _ = REGISTRY.register(Box::new(DECISION_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(RISK_CHECK_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(ORDER_INTERNAL_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(CB_STATE.clone()));
        let _ = REGISTRY.register(Box::new(CB_TRIP_TOTAL.clone()));
        let _ = REGISTRY.register(Box::new(CB_FAILURE_RATE.clone()));
        let _ = REGISTRY.register(Box::new(CB_CHECK_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(OPERATIONS_TOTAL.clone()));
        let _ = REGISTRY.register(Box::new(ORDERS_PROCESSED.clone()));
        let _ = REGISTRY.register(Box::new(ORDERS_RECEIVED.clone()));
        let _ = REGISTRY.register(Box::new(MEMORY_POOL_AVAILABLE.clone()));
        let _ = REGISTRY.register(Box::new(MEMORY_POOL_TOTAL.clone()));
        let _ = REGISTRY.register(Box::new(ALLOCATION_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(MAX_DRAWDOWN_CURRENT.clone()));
        let _ = REGISTRY.register(Box::new(MAX_POSITION_SIZE_RATIO.clone()));
        let _ = REGISTRY.register(Box::new(PORTFOLIO_CORRELATION.clone()));
        let _ = REGISTRY.register(Box::new(RISK_CHECK_PASS.clone()));
        let _ = REGISTRY.register(Box::new(RISK_CHECK_FAIL.clone()));
        let _ = REGISTRY.register(Box::new(ORDER_QUEUE_DEPTH.clone()));
        let _ = REGISTRY.register(Box::new(ORDERS_BY_TYPE.clone()));
        let _ = REGISTRY.register(Box::new(ORDERS_SUCCESS.clone()));
        let _ = REGISTRY.register(Box::new(ORDERS_FAILED.clone()));
        let _ = REGISTRY.register(Box::new(EXCHANGE_API_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(MARKET_DATA_LAST_UPDATE.clone()));
        let _ = REGISTRY.register(Box::new(ORDER_BOOK_DEPTH.clone()));
    });
}

// Timing helper for measuring latencies
/// TODO: Add docs
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Timer {
            start: Instant::now(),
        }
    }
    
    pub fn elapsed_micros(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64 / 1000.0
    }
    
    pub fn elapsed_nanos(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64
    }
    
    pub fn elapsed_millis(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

// Macros for easy metric recording
#[macro_export]
macro_rules! record_decision_latency {
    ($component:expr, $decision_type:expr, $timer:expr) => {
        $crate::observability::metrics::DECISION_LATENCY
            .with_label_values(&[$component, $decision_type])
            .observe($timer.elapsed_micros());
    };
}

#[macro_export]
macro_rules! record_risk_check {
    ($check_type:expr, $symbol:expr, $timer:expr) => {
        $crate::observability::metrics::RISK_CHECK_LATENCY
            .with_label_values(&[$check_type, $symbol])
            .observe($timer.elapsed_micros());
    };
}

#[macro_export]
macro_rules! record_order_latency {
    ($order_type:expr, $exchange:expr, $timer:expr) => {
        $crate::observability::metrics::ORDER_INTERNAL_LATENCY
            .with_label_values(&[$order_type, $exchange])
            .observe($timer.elapsed_micros());
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_initialization() {
        init_metrics();
        
        // Test recording a decision latency
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_micros(1));
        DECISION_LATENCY
            .with_label_values(&["test_component", "test_decision"])
            .observe(timer.elapsed_micros());
        
        // Verify metric was recorded
        let metric_families = REGISTRY.gather();
        assert!(!metric_families.is_empty());
    }
    
    #[test]
    fn test_timer_precision() {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_nanos(100));
        
        let nanos = timer.elapsed_nanos();
        assert!(nanos >= 100.0);
        
        let micros = timer.elapsed_micros();
        assert!(micros >= 0.1);
    }
}