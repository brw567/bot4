// Enhanced Rayon Parallelization - Nexus Priority 1 Optimization  
// Team: Jordan (Performance) + Sam (Architecture) + Full Team
// Implements comprehensive Rayon parallelization for all hot paths
// Target: 500k+ ops/sec with perfect load balancing

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam::utils::CachePadded;
use dashmap::DashMap;
use anyhow::{Result, Context};
use std::time::{Duration, Instant};
use rust_decimal::prelude::*;

// Import our object pools
use crate::object_pools::{
    acquire_order, acquire_signal, acquire_market_data,
    Order, Signal, MarketData, SignalType,
};

// ============================================================================
// PARALLEL TRADING ENGINE - Jordan's Design
// ============================================================================

/// Main parallel trading engine with work-stealing
pub struct ParallelTradingEngine {
    /// Dedicated thread pool for trading operations
    trading_pool: Arc<ThreadPool>,
    /// Dedicated thread pool for ML inference
    ml_pool: Arc<ThreadPool>,
    /// Dedicated thread pool for risk checks
    risk_pool: Arc<ThreadPool>,
    /// Performance metrics
    metrics: Arc<EngineMetrics>,
    /// Instrument sharding
    sharding: Arc<InstrumentSharding>,
}

impl ParallelTradingEngine {
    /// Create new parallel trading engine with optimal configuration
    pub fn new() -> Result<Self> {
        let num_cores = num_cpus::get();
        
        // Split cores optimally across pools
        // Trading: 50% of cores (6 on 12-core)
        // ML: 25% of cores (3 on 12-core)  
        // Risk: 25% of cores (3 on 12-core)
        let trading_threads = (num_cores / 2).max(1);
        let ml_threads = (num_cores / 4).max(1);
        let risk_threads = (num_cores / 4).max(1);
        
        println!("Initializing Parallel Trading Engine:");
        println!("  Trading threads: {}", trading_threads);
        println!("  ML threads: {}", ml_threads);
        println!("  Risk threads: {}", risk_threads);
        
        // Build dedicated thread pools with CPU affinity
        let trading_pool = ThreadPoolBuilder::new()
            .num_threads(trading_threads)
            .thread_name(|i| format!("trading-{}", i))
            .start_handler(move |i| {
                #[cfg(target_os = "linux")]
                pin_to_core(i); // Trading on cores 0-5
            })
            .build()
            .context("Failed to build trading thread pool")?;
            
        let ml_pool = ThreadPoolBuilder::new()
            .num_threads(ml_threads)
            .thread_name(|i| format!("ml-{}", i))
            .start_handler(move |i| {
                #[cfg(target_os = "linux")]
                pin_to_core(trading_threads + i); // ML on cores 6-8
            })
            .build()
            .context("Failed to build ML thread pool")?;
            
        let risk_pool = ThreadPoolBuilder::new()
            .num_threads(risk_threads)
            .thread_name(|i| format!("risk-{}", i))
            .start_handler(move |i| {
                #[cfg(target_os = "linux")]
                pin_to_core(trading_threads + ml_threads + i); // Risk on cores 9-11
            })
            .build()
            .context("Failed to build risk thread pool")?;
        
        Ok(Self {
            trading_pool: Arc::new(trading_pool),
            ml_pool: Arc::new(ml_pool),
            risk_pool: Arc::new(risk_pool),
            metrics: Arc::new(EngineMetrics::new()),
            sharding: Arc::new(InstrumentSharding::new(trading_threads)),
        })
    }
    
    /// Process market data in parallel with perfect load balancing
    pub fn process_market_data(&self, data: Vec<MarketData>) -> Vec<Signal> {
        let start = Instant::now();
        
        // Group data by instrument for cache locality
        let by_instrument: DashMap<String, Vec<MarketData>> = DashMap::new();
        for item in data {
            by_instrument.entry(item.symbol.clone())
                .or_default()
                .push(item);
        }
        
        // Process each instrument group in parallel
        let signals: Vec<Signal> = self.trading_pool.install(|| {
            // Convert DashMap to Vec first
            let instrument_groups: Vec<(String, Vec<MarketData>)> = 
                by_instrument.into_iter().collect();
            
            instrument_groups.into_par_iter()
                .flat_map(|(symbol, data_points)| {
                    // Process all data points for this instrument
                    data_points.par_iter()
                        .filter_map(|data| {
                            self.generate_signal(data)
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        
        // Update metrics
        let elapsed = start.elapsed();
        self.metrics.record_batch_processing(signals.len(), elapsed);
        
        signals
    }
    
    /// Generate trading signal from market data
    fn generate_signal(&self, data: &MarketData) -> Option<Signal> {
        // Use object pool for zero allocation
        let mut signal = acquire_signal();
        
        // Simple momentum signal (example)
        let spread = (data.ask - data.bid).to_f64().unwrap_or(0.0);
        let mid = ((data.ask + data.bid) / Decimal::from(2)).to_f64().unwrap_or(0.0);
        
        // Signal strength based on spread tightness
        let strength = 1.0 / (1.0 + spread / mid);
        
        if strength > 0.7 {
            signal.symbol = data.symbol.clone();
            signal.strength = strength;
            signal.signal_type = if data.last > data.ask {
                SignalType::Long
            } else if data.last < data.bid {
                SignalType::Short
            } else {
                SignalType::Neutral
            };
            signal.confidence = strength * 0.9;
            signal.timestamp = data.timestamp;
            
            // Convert PoolGuard to owned Signal
            Some((*signal).clone())
        } else {
            None
        }
    }
    
    /// Parallel risk checks with dedicated thread pool
    pub fn check_risks_parallel(&self, orders: &[Order]) -> Vec<bool> {
        self.risk_pool.install(|| {
            orders.par_iter()
                .map(|order| self.check_single_risk(order))
                .collect()
        })
    }
    
    /// Check risk for single order
    fn check_single_risk(&self, order: &Order) -> bool {
        // Simulate risk checks
        let position_check = order.quantity.to_f64().unwrap_or(0.0) < 10000.0;
        let price_check = order.price.to_f64().unwrap_or(0.0) > 0.0;
        let symbol_check = !order.symbol.is_empty();
        
        position_check && price_check && symbol_check
    }
    
    /// Parallel ML inference with dedicated thread pool
    pub fn run_ml_inference(&self, features: Vec<Vec<f64>>) -> Vec<f64> {
        let start = Instant::now();
        
        let predictions: Vec<f64> = self.ml_pool.install(|| {
            features.par_iter()
                .map(|feature_vec| {
                    // Simulate ML inference
                    self.simulate_inference(feature_vec)
                })
                .collect()
        });
        
        let elapsed = start.elapsed();
        self.metrics.record_ml_inference(predictions.len(), elapsed);
        
        predictions
    }
    
    /// Simulate ML inference (placeholder for real model)
    fn simulate_inference(&self, features: &[f64]) -> f64 {
        // Simple weighted sum as placeholder
        features.iter().enumerate()
            .map(|(i, &f)| f * (1.0 / (i + 1) as f64))
            .sum()
    }
    
    /// Process orders in parallel batches
    pub fn process_order_batch(&self, orders: Vec<Order>) -> Vec<Result<String>> {
        // Split into optimal batch sizes for cache efficiency
        const BATCH_SIZE: usize = 64; // Cache line aligned
        
        let results: Vec<Result<String>> = orders
            .par_chunks(BATCH_SIZE)
            .flat_map(|batch| {
                batch.par_iter()
                    .map(|order| self.process_single_order(order))
                    .collect::<Vec<_>>()
            })
            .collect();
            
        self.metrics.record_orders_processed(results.len());
        results
    }
    
    /// Process single order
    fn process_single_order(&self, order: &Order) -> Result<String> {
        // Simulate order processing
        if self.check_single_risk(order) {
            Ok(format!("ORDER_{}_ACCEPTED", order.id))
        } else {
            Err(anyhow::anyhow!("Risk check failed"))
        }
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> EngineMetrics {
        (*self.metrics).clone()
    }
}

/// Performance metrics for the parallel engine
#[derive(Clone)]
pub struct EngineMetrics {
    /// Total operations processed
    pub total_ops: Arc<CachePadded<AtomicU64>>,
    /// Total processing time in microseconds
    pub total_time_us: Arc<CachePadded<AtomicU64>>,
    /// Peak throughput ops/sec
    pub peak_throughput: Arc<CachePadded<AtomicU64>>,
    /// ML inference count
    pub ml_inference_count: Arc<CachePadded<AtomicU64>>,
    /// ML total time microseconds
    pub ml_total_time_us: Arc<CachePadded<AtomicU64>>,
    /// Orders processed
    pub orders_processed: Arc<CachePadded<AtomicU64>>,
}

impl EngineMetrics {
    fn new() -> Self {
        Self {
            total_ops: Arc::new(CachePadded::new(AtomicU64::new(0))),
            total_time_us: Arc::new(CachePadded::new(AtomicU64::new(0))),
            peak_throughput: Arc::new(CachePadded::new(AtomicU64::new(0))),
            ml_inference_count: Arc::new(CachePadded::new(AtomicU64::new(0))),
            ml_total_time_us: Arc::new(CachePadded::new(AtomicU64::new(0))),
            orders_processed: Arc::new(CachePadded::new(AtomicU64::new(0))),
        }
    }
    
    fn record_batch_processing(&self, count: usize, duration: Duration) {
        self.total_ops.fetch_add(count as u64, Ordering::Relaxed);
        self.total_time_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        
        // Calculate and update peak throughput
        let throughput = (count as f64 / duration.as_secs_f64()) as u64;
        self.update_peak_throughput(throughput);
    }
    
    fn record_ml_inference(&self, count: usize, duration: Duration) {
        self.ml_inference_count.fetch_add(count as u64, Ordering::Relaxed);
        self.ml_total_time_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }
    
    fn record_orders_processed(&self, count: usize) {
        self.orders_processed.fetch_add(count as u64, Ordering::Relaxed);
    }
    
    fn update_peak_throughput(&self, current: u64) {
        let mut peak = self.peak_throughput.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_throughput.compare_exchange_weak(
                peak,
                current,
                Ordering::Release,
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }
    
    /// Get average latency in microseconds
    pub fn avg_latency_us(&self) -> f64 {
        let ops = self.total_ops.load(Ordering::Relaxed);
        if ops == 0 {
            return 0.0;
        }
        let total = self.total_time_us.load(Ordering::Relaxed);
        total as f64 / ops as f64
    }
    
    /// Get ML average latency in microseconds
    pub fn ml_avg_latency_us(&self) -> f64 {
        let count = self.ml_inference_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total = self.ml_total_time_us.load(Ordering::Relaxed);
        total as f64 / count as f64
    }
}

/// Instrument sharding for cache-efficient parallel processing
pub struct InstrumentSharding {
    num_shards: usize,
    instrument_map: DashMap<String, usize>,
    next_shard: CachePadded<AtomicUsize>,
}

impl InstrumentSharding {
    pub fn new(num_shards: usize) -> Self {
        Self {
            num_shards,
            instrument_map: DashMap::new(),
            next_shard: CachePadded::new(AtomicUsize::new(0)),
        }
    }
    
    pub fn get_shard(&self, instrument: &str) -> usize {
        *self.instrument_map.entry(instrument.to_string())
            .or_insert_with(|| {
                self.next_shard.fetch_add(1, Ordering::Relaxed) % self.num_shards
            })
    }
}

/// Pin thread to specific CPU core (Linux only)
#[cfg(target_os = "linux")]
fn pin_to_core(core_id: usize) {
    use libc::{cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};
    use std::mem;
    
    unsafe {
        let mut set: cpu_set_t = mem::zeroed();
        CPU_ZERO(&mut set);
        CPU_SET(core_id, &mut set);
        
        sched_setaffinity(
            0, // Current thread
            mem::size_of::<cpu_set_t>(),
            &set
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_core(_core_id: usize) {
    // No-op on non-Linux platforms
}

// ============================================================================
// ADVANCED RAYON PATTERNS - Sam's Contribution
// ============================================================================

/// Parallel pipeline for stream processing
pub struct ParallelPipeline<T: Send + Sync> {
    stages: Vec<Box<dyn PipelineStage<T>>>,
}

pub trait PipelineStage<T: Send + Sync>: Send + Sync {
    fn process(&self, input: T) -> Option<T>;
}

impl<T: Send + Sync + 'static> ParallelPipeline<T> {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
        }
    }
    
    pub fn add_stage<S: PipelineStage<T> + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }
    
    pub fn process_batch(&self, items: Vec<T>) -> Vec<T> {
        items.into_par_iter()
            .filter_map(|item| {
                let mut current = Some(item);
                for stage in &self.stages {
                    if let Some(val) = current {
                        current = stage.process(val);
                    } else {
                        break;
                    }
                }
                current
            })
            .collect()
    }
}

// ============================================================================
// BENCHMARKS
// ============================================================================

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, Criterion};
    
    fn bench_parallel_processing(c: &mut Criterion) {
        let engine = ParallelTradingEngine::new().unwrap();
        
        // Generate test data
        let mut market_data = Vec::new();
        for i in 0..10000 {
            let mut data = acquire_market_data();
            data.symbol = format!("BTC/USDT_{}", i % 100);
            data.bid = rust_decimal::Decimal::from(50000 + i);
            data.ask = rust_decimal::Decimal::from(50010 + i);
            data.last = rust_decimal::Decimal::from(50005 + i);
            market_data.push((*data).clone());
        }
        
        c.bench_function("parallel_market_processing", |b| {
            b.iter(|| {
                let signals = engine.process_market_data(black_box(market_data.clone()));
                black_box(signals);
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_engine_creation() {
        let engine = ParallelTradingEngine::new().unwrap();
        let metrics = engine.metrics();
        assert_eq!(metrics.total_ops.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_parallel_market_processing() {
        let engine = ParallelTradingEngine::new().unwrap();
        
        // Create test data
        let mut market_data = Vec::new();
        for i in 0..100 {
            let mut data = acquire_market_data();
            data.symbol = format!("TEST_{}", i % 10);
            data.bid = rust_decimal::Decimal::from(100 + i);
            data.ask = rust_decimal::Decimal::from(101 + i);
            data.last = rust_decimal::Decimal::from(100 + i);
            data.timestamp = i as i64;
            market_data.push((*data).clone());
        }
        
        let signals = engine.process_market_data(market_data);
        assert!(!signals.is_empty());
        
        let metrics = engine.metrics();
        assert!(metrics.total_ops.load(Ordering::Relaxed) > 0);
        assert!(metrics.avg_latency_us() > 0.0);
    }
    
    #[test]
    fn test_parallel_risk_checks() {
        let engine = ParallelTradingEngine::new().unwrap();
        
        let mut orders = Vec::new();
        for i in 0..50 {
            let mut order = acquire_order();
            order.id = i;
            order.symbol = format!("BTC/USDT");
            order.quantity = rust_decimal::Decimal::from(i * 100);
            order.price = rust_decimal::Decimal::from(50000);
            orders.push((*order).clone());
        }
        
        let results = engine.check_risks_parallel(&orders);
        assert_eq!(results.len(), orders.len());
    }
    
    #[test]
    fn test_throughput_target() {
        let engine = ParallelTradingEngine::new().unwrap();
        
        // Generate large batch
        let mut market_data = Vec::new();
        for i in 0..50000 {
            let mut data = acquire_market_data();
            data.symbol = format!("SYM_{}", i % 1000);
            data.bid = rust_decimal::Decimal::from(1000 + (i % 100));
            data.ask = rust_decimal::Decimal::from(1001 + (i % 100));
            data.last = rust_decimal::Decimal::from(1000 + (i % 100));
            market_data.push((*data).clone());
        }
        
        let start = Instant::now();
        let _signals = engine.process_market_data(market_data);
        let elapsed = start.elapsed();
        
        let throughput = 50000.0 / elapsed.as_secs_f64();
        println!("Achieved throughput: {:.0} ops/sec", throughput);
        
        // Should achieve at least 100k ops/sec (conservative target)
        assert!(throughput > 100_000.0, 
            "Throughput {:.0} below 100k ops/sec target", throughput);
    }
}