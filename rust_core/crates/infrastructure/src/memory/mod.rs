// Bot4 Memory Management Module
// Day 2 Sprint - Critical Path
// Owner: Jordan
// Target: <10ns allocation, zero-alloc hot paths

// NOTE: Global MiMalloc allocator is now defined in crate::allocator
// This ensures <10ns allocation performance globally

pub mod pools;
pub mod rings;
pub mod metrics;
pub mod safe_pools;  // NEW: Memory-safe pools with cleanup
pub mod pools_upgraded;  // Existing upgraded pools

#[cfg(test)]
mod zero_alloc_tests;

// Re-export key types
pub use pools::{OrderPool, SignalPool, TickPool, PoolStats};
pub use rings::{SpscRing, MpmcRing};
pub use metrics::MemoryMetrics;

// NEW: Export safe pools
pub use safe_pools::{
    SafeObjectPool, Order as SafeOrder, Signal as SafeSignal, Tick as SafeTick,
    ORDER_POOL, SIGNAL_POOL, TICK_POOL,
    create_order_pool, create_signal_pool, create_tick_pool,
};

/// Initialize all memory pools at startup
/// Must be called before any trading operations
pub fn initialize_memory_system() {
    // Pre-warm allocator
    warm_allocator();
    
    // Initialize pools
    pools::initialize_all_pools();
    
    // Verify allocation performance
    verify_allocation_latency();
    
    // Initialize metrics
    metrics::initialize_metrics();
    
    tracing::info!("Memory system initialized with MiMalloc");
}

/// Pre-warm the allocator to avoid cold start penalties
fn warm_allocator() {
    const WARM_SIZE: usize = 1024 * 1024; // 1MB
    const ITERATIONS: usize = 100;
    
    for _ in 0..ITERATIONS {
        let v: Vec<u8> = Vec::with_capacity(WARM_SIZE);
        drop(v);
    }
}

/// Verify allocation meets <10ns target
fn verify_allocation_latency() {
    use std::time::Instant;
    
    const TEST_ITERATIONS: usize = 100_000;
    const SIZES: &[usize] = &[64, 256, 1024, 4096];
    
    for &size in SIZES {
        let start = Instant::now();
        
        for _ in 0..TEST_ITERATIONS {
            let v = Vec::<u8>::with_capacity(size);
            std::hint::black_box(v);
        }
        
        let elapsed = start.elapsed();
        let per_alloc_ns = elapsed.as_nanos() / TEST_ITERATIONS as u128;
        
        if per_alloc_ns > 10 {
            tracing::warn!(
                "Allocation latency {}ns for size {} exceeds 10ns target",
                per_alloc_ns, size
            );
        } else {
            tracing::debug!(
                "Allocation latency {}ns for size {} ✓",
                per_alloc_ns, size
            );
        }
    }
}

/// Memory statistics for monitoring
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub reserved_bytes: usize,
    pub pool_stats: PoolStats,
}

impl MemoryStats {
    pub fn current() -> Self {
        // MiMalloc stats would go here
        // For now, use pool stats
        Self {
            allocated_bytes: 0,
            reserved_bytes: 0,
            pool_stats: pools::get_pool_stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    
    #[test]
    fn test_allocation_performance() {
        const ITERATIONS: usize = 1_000_000;
        
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let v = Vec::<u8>::with_capacity(256);
            std::hint::black_box(v);
        }
        let elapsed = start.elapsed();
        
        let per_alloc = elapsed.as_nanos() / ITERATIONS as u128;
        println!("Allocation latency: {}ns", per_alloc);
        
        // With MiMalloc active, we MUST achieve <50ns
        assert!(per_alloc < 50, "Allocation too slow: {}ns (target <50ns with MiMalloc)", per_alloc);
    }
}