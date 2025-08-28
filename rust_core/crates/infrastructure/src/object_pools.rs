use domain_types::market_data::MarketData;
use domain_types::order::OrderError;
pub use domain_types::position_canonical::{Position, PositionId, PositionSide, PositionStatus};

//! Module uses canonical Order type from domain_types
//! Avery: "Single source of truth for Order struct"

pub use domain_types::order::{
    Order, OrderId, OrderSide, OrderType, OrderStatus, TimeInForce,
    OrderError, Fill, FillId
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type OrderResult<T> = Result<T, OrderError>;

// Comprehensive Object Pools - Nexus Priority 1 Optimization
// Team: Jordan (Performance Lead) + Sam (Architecture) + Full Team
// Implements 1M+ pre-allocated objects for zero-allocation hot paths
// Target: <100ns pool operations, 100% cache hit rate

use crate::zero_copy::{ObjectPool, PoolGuard};
use std::sync::Arc;
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use rust_decimal::Decimal;
use std::collections::HashMap;

// ============================================================================
// TRADING OBJECT DEFINITIONS
// ============================================================================

    pub id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub order_type: OrderType,
    pub timestamp: i64,
    pub exchange: String,
    pub status: OrderStatus,
}


pub enum OrderSide {
    #[default]
    Buy,
    Sell,
}


pub enum OrderType {
    #[default]
    Market,
    Limit,
    StopLoss,
    TakeProfit,
    PostOnly,
}


pub enum OrderStatus {
    #[default]
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}


// REMOVED: use domain_types::Signal
// pub struct Signal {
    pub symbol: String,
    pub strength: f64,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub timestamp: i64,
    pub features: Vec<f64>,
    pub metadata: HashMap<String, f64>,
}


pub enum SignalType {
    #[default]
    Long,
    Short,
    Neutral,
    ClosePosition,
}


// REMOVED: Using canonical domain_types::market_data::MarketData
// pub struct MarketData {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub last: Decimal,
    pub volume: Decimal,
    pub timestamp: i64,
    pub exchange: String,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
}




pub struct RiskCheck {
    pub order_id: u64,
    pub passed: bool,
    pub checks_performed: Vec<String>,
    pub violations: Vec<String>,
    pub risk_score: f64,
    pub timestamp: i64,
}


pub struct ExecutionReport {
    pub order_id: u64,
    pub exec_id: String,
    pub filled_quantity: Decimal,
    pub filled_price: Decimal,
    pub fees: Decimal,
    pub timestamp: i64,
    pub exchange: String,
}


pub struct Feature {
    pub name: String,
    pub values: Vec<f64>,
    pub computed_at: i64,
    pub window_size: usize,
}


pub struct MLInference {
    pub model_id: String,
    pub predictions: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub features_used: Vec<String>,
    pub latency_us: u64,
    pub timestamp: i64,
}

// ============================================================================
// POOL SIZES - Based on Nexus's Recommendations
// ============================================================================

const ORDER_POOL_SIZE: usize = 100_000;        // 100K orders
const SIGNAL_POOL_SIZE: usize = 200_000;       // 200K signals  
const MARKET_DATA_POOL_SIZE: usize = 500_000;  // 500K market updates
const POSITION_POOL_SIZE: usize = 10_000;      // 10K positions
const RISK_CHECK_POOL_SIZE: usize = 100_000;   // 100K risk checks
const EXECUTION_POOL_SIZE: usize = 50_000;     // 50K executions
const FEATURE_POOL_SIZE: usize = 100_000;      // 100K features
const ML_INFERENCE_POOL_SIZE: usize = 50_000;  // 50K inferences

// Total: 1,110,000 pre-allocated objects (exceeds 1M requirement)

// ============================================================================
// GLOBAL POOL REGISTRY - Singleton Pattern
// ============================================================================

lazy_static! {
    /// Global registry of all object pools - Jordan's design
    pub static ref POOL_REGISTRY: PoolRegistry = PoolRegistry::new();
}

pub struct PoolRegistry {
    pub orders: Arc<ObjectPool<Order>>,
    pub signals: Arc<ObjectPool<Signal>>,
    pub market_data: Arc<ObjectPool<MarketData>>,
    pub positions: Arc<ObjectPool<Position>>,
    pub risk_checks: Arc<ObjectPool<RiskCheck>>,
    pub executions: Arc<ObjectPool<ExecutionReport>>,
    pub features: Arc<ObjectPool<Feature>>,
    pub ml_inferences: Arc<ObjectPool<MLInference>>,
}

impl PoolRegistry {
    fn new() -> Self {
        println!("Initializing global object pools (1.1M objects)...");
        let start = std::time::Instant::now();
        
        let registry = Self {
            orders: Arc::new(ObjectPool::new(ORDER_POOL_SIZE)),
            signals: Arc::new(ObjectPool::new(SIGNAL_POOL_SIZE)),
            market_data: Arc::new(ObjectPool::new(MARKET_DATA_POOL_SIZE)),
            positions: Arc::new(ObjectPool::new(POSITION_POOL_SIZE)),
            risk_checks: Arc::new(ObjectPool::new(RISK_CHECK_POOL_SIZE)),
            executions: Arc::new(ObjectPool::new(EXECUTION_POOL_SIZE)),
            features: Arc::new(ObjectPool::new(FEATURE_POOL_SIZE)),
            ml_inferences: Arc::new(ObjectPool::new(ML_INFERENCE_POOL_SIZE)),
        };
        
        let elapsed = start.elapsed();
        println!("Object pools initialized in {:?}", elapsed);
        println!("Total objects pre-allocated: 1,110,000");
        println!("Memory overhead: ~{}MB", 
            (ORDER_POOL_SIZE * std::mem::size_of::<Order>() +
             SIGNAL_POOL_SIZE * std::mem::size_of::<Signal>() +
             MARKET_DATA_POOL_SIZE * std::mem::size_of::<MarketData>() +
             POSITION_POOL_SIZE * std::mem::size_of::<Position>() +
             RISK_CHECK_POOL_SIZE * std::mem::size_of::<RiskCheck>() +
             EXECUTION_POOL_SIZE * std::mem::size_of::<ExecutionReport>() +
             FEATURE_POOL_SIZE * std::mem::size_of::<Feature>() +
             ML_INFERENCE_POOL_SIZE * std::mem::size_of::<MLInference>()) / 1_048_576
        );
        
        registry
    }
    
    /// Get statistics for all pools
    pub fn global_stats(&self) -> GlobalPoolStats {
        GlobalPoolStats {
            orders: self.orders.stats(),
            signals: self.signals.stats(),
            market_data: self.market_data.stats(),
            positions: self.positions.stats(),
            risk_checks: self.risk_checks.stats(),
            executions: self.executions.stats(),
            features: self.features.stats(),
            ml_inferences: self.ml_inferences.stats(),
            total_objects: ORDER_POOL_SIZE + SIGNAL_POOL_SIZE + MARKET_DATA_POOL_SIZE +
                          POSITION_POOL_SIZE + RISK_CHECK_POOL_SIZE + EXECUTION_POOL_SIZE +
                          FEATURE_POOL_SIZE + ML_INFERENCE_POOL_SIZE,
        }
    }
}


pub struct GlobalPoolStats {
    pub orders: crate::zero_copy::PoolStats,
    pub signals: crate::zero_copy::PoolStats,
    pub market_data: crate::zero_copy::PoolStats,
    pub positions: crate::zero_copy::PoolStats,
    pub risk_checks: crate::zero_copy::PoolStats,
    pub executions: crate::zero_copy::PoolStats,
    pub features: crate::zero_copy::PoolStats,
    pub ml_inferences: crate::zero_copy::PoolStats,
    pub total_objects: usize,
}

// ============================================================================
// CONVENIENCE FUNCTIONS - Zero-allocation accessors
// ============================================================================

/// Acquire an Order from the pool
#[inline(always)]
pub fn acquire_order() -> PoolGuard<Order> {
    POOL_REGISTRY.orders.acquire()
}

/// Acquire a Signal from the pool
#[inline(always)]
pub fn acquire_signal() -> PoolGuard<Signal> {
    POOL_REGISTRY.signals.acquire()
}

/// Acquire MarketData from the pool
#[inline(always)]
pub fn acquire_market_data() -> PoolGuard<MarketData> {
    POOL_REGISTRY.market_data.acquire()
}

/// Acquire a Position from the pool
#[inline(always)]
pub fn acquire_position() -> PoolGuard<Position> {
    POOL_REGISTRY.positions.acquire()
}

/// Acquire a RiskCheck from the pool
#[inline(always)]
pub fn acquire_risk_check() -> PoolGuard<RiskCheck> {
    POOL_REGISTRY.risk_checks.acquire()
}

/// Acquire an ExecutionReport from the pool
#[inline(always)]
pub fn acquire_execution() -> PoolGuard<ExecutionReport> {
    POOL_REGISTRY.executions.acquire()
}

/// Acquire a Feature from the pool
#[inline(always)]
pub fn acquire_feature() -> PoolGuard<Feature> {
    POOL_REGISTRY.features.acquire()
}

/// Acquire an MLInference from the pool
#[inline(always)]
pub fn acquire_ml_inference() -> PoolGuard<MLInference> {
    POOL_REGISTRY.ml_inferences.acquire()
}

// ============================================================================
// BENCHMARKS - Verify <100ns operations
// ============================================================================

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_order_pool(c: &mut Criterion) {
        c.bench_function("order_pool_acquire_release", |b| {
            b.iter(|| {
                let mut order = acquire_order();
                order.id = black_box(12345);
                order.quantity = black_box(Decimal::from(100));
                // Guard automatically returns to pool on drop
            });
        });
    }
    
    fn bench_signal_pool(c: &mut Criterion) {
        c.bench_function("signal_pool_acquire_release", |b| {
            b.iter(|| {
                let mut signal = acquire_signal();
                signal.strength = black_box(0.85);
                signal.confidence = black_box(0.92);
            });
        });
    }
    
    fn bench_market_data_pool(c: &mut Criterion) {
        c.bench_function("market_data_pool_acquire_release", |b| {
            b.iter(|| {
                let mut data = acquire_market_data();
                data.bid = black_box(Decimal::from(50000));
                data.ask = black_box(Decimal::from(50010));
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_initialization() {
        // Force initialization
        let _ = &*POOL_REGISTRY;
        
        let stats = POOL_REGISTRY.global_stats();
        assert_eq!(stats.total_objects, 1_110_000);
        
        // All pools should be at capacity
        assert_eq!(stats.orders.capacity, ORDER_POOL_SIZE);
        assert_eq!(stats.signals.capacity, SIGNAL_POOL_SIZE);
        assert_eq!(stats.market_data.capacity, MARKET_DATA_POOL_SIZE);
    }
    
    #[test]
    fn test_zero_allocation_hot_path() {
        // Warm up pools
        let _ = &*POOL_REGISTRY;
        
        // Get initial stats
        let initial_stats = POOL_REGISTRY.orders.stats();
        
        // Acquire and release 1000 times
        for i in 0..1000 {
            let mut order = acquire_order();
            order.id = i;
            order.quantity = Decimal::from(100);
            // Auto-return on drop
        }
        
        // Check stats
        let final_stats = POOL_REGISTRY.orders.stats();
        
        // Should have 1000 new hits, 0 misses (no new allocations)
        let hit_diff = final_stats.hits - initial_stats.hits;
        let miss_diff = final_stats.misses - initial_stats.misses;
        
        // BUGFIX: In parallel test runs, other tests might use the pool
        // We expect AT LEAST 1000 hits (our operations) but could be more
        assert!(hit_diff >= 1000, "Expected at least 1000 hits, got {}", hit_diff);
        assert_eq!(miss_diff, 0, "Expected 0 misses, got {}", miss_diff);
        // BUGFIX: allocated should not increase from initial value
        // The pool pre-allocates objects, so we check no NEW allocations
        assert_eq!(final_stats.allocated, initial_stats.allocated, 
                   "Pool allocated {} new objects (expected 0)", 
                   final_stats.allocated - initial_stats.allocated);
    }
    
    #[test]
    fn test_concurrent_pool_access() {
        use std::thread;
        use std::sync::atomic::{AtomicU64, Ordering};
        
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = vec![];
        
        // Spawn 10 threads, each acquiring 1000 orders
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    let mut order = acquire_order();
                    order.id = counter_clone.fetch_add(1, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(counter.load(Ordering::SeqCst), 10000);
        
        // Check pool integrity
        let stats = POOL_REGISTRY.orders.stats();
        assert!(stats.hit_rate > 0.99); // Should be near 100% hit rate
    }
}
