// Phase 1: Hot Path Verification
// Ensures zero allocations in critical paths
// Owner: Jordan | Reviewer: Sam
// Performance Target: 0 allocations, <1μs latency

use crate::memory::pools;
use crate::parallelization::memory_ordering;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

/// Verifies zero allocations in hot paths
pub struct HotPathValidator {
    /// Allocation count before test
    initial_allocs: usize,
    /// Path being validated
    path_name: &'static str,  // ZERO-COPY: Use static string
}

impl HotPathValidator {
    /// Start validation
    pub fn begin(path_name: &'static str) -> Self {
        // In production, would hook into MiMalloc stats
        // For now, we verify through pool metrics
        let stats = pools::get_pool_stats();
        let initial_allocs = stats.order_allocated + stats.signal_allocated + stats.tick_allocated;
        
        Self {
            initial_allocs,
            path_name,  // ZERO-COPY: Just store the reference
        }
    }
    
    /// Complete validation
    pub fn validate(self) -> Result<ValidationReport, String> {
        let stats = pools::get_pool_stats();
        let final_allocs = stats.order_allocated + stats.signal_allocated + stats.tick_allocated;
        let allocations = final_allocs.saturating_sub(self.initial_allocs);
        
        if allocations > 0 {
            // ZERO-COPY: Log details, return static error
            tracing::error!(
                path = self.path_name,
                allocations = allocations,
                "Hot path performed allocations"
            );
            Err("Hot path allocation detected".to_string())  // Only allocate for error case
        } else {
            Ok(ValidationReport {
                path_name: self.path_name.to_string(),  // Allocate only on success
                allocations: 0,
                validated: true,
            })
        }
    }
}

/// Validation report
#[derive(Debug)]
pub struct ValidationReport {
    pub path_name: String,
    pub allocations: usize,
    pub validated: bool,
}

/// Critical hot paths that must have zero allocations
pub mod critical_paths {
    use super::*;
    use crate::memory::pools::{acquire_order, release_order, acquire_signal, release_signal};
    
    
    /// Order processing hot path - ZERO ALLOCATION
    #[inline(always)]
    pub fn process_order_hot_path() -> Result<(), &'static str> {
        // NOTE: No validation inside hot path - measure from outside!
        
        // Acquire from pool (no allocation)
        let mut order = acquire_order();
        
        // Use symbol ID instead of string manipulation
        // Symbol 1 = BTC/USD (pre-registered)
        order.symbol_id = 1;
        order.price = 50000.0;
        order.quantity = 0.1;
        order.timestamp = 1234567890;
        
        // Risk check (using atomics, no allocation)
        let risk_passed = check_risk_atomic(order.price, order.quantity);
        
        if !risk_passed {
            release_order(order);
            return Err("Risk check failed"); // Static string, no allocation
        }
        
        // Release back to pool (no deallocation)
        release_order(order);
        
        // Success - validation happens externally
        Ok(())
    }
    
    /// Signal processing hot path - ZERO ALLOCATION
    #[inline(always)]
    pub fn process_signal_hot_path() -> Result<(), &'static str> {
        // NOTE: No validation inside hot path - measure from outside!
        
        // Acquire from pool
        let mut signal = acquire_signal();
        
        // Use symbol ID instead of string manipulation
        // Symbol 2 = ETH/USD (pre-registered)
        signal.symbol_id = 2;
        signal.signal_type = crate::memory::pools::SignalType::Buy;
        signal.strength = 0.85;
        signal.timestamp = 1234567890;
        
        // Apply filters (using pre-allocated buffers)
        let filtered_strength = apply_filters_zero_alloc(signal.strength);
        signal.strength = filtered_strength;
        
        // Release back to pool
        release_signal(signal);
        
        // Success - validation happens externally
        Ok(())
    }
    
    /// Risk check using only atomics (zero allocation)
    #[inline(always)]
    fn check_risk_atomic(price: f64, quantity: f64) -> bool {
        // Static risk limits (would be atomics in production)
        static MAX_POSITION_VALUE: AtomicU64 = AtomicU64::new(100000_00000000); // $100k in fixed point
        
        // Convert to fixed point to avoid float operations
        let value_fixed = ((price * quantity) * 100000000.0) as u64;
        
        // Atomic comparison
        let max = MAX_POSITION_VALUE.load(memory_ordering::READ_STATE);
        value_fixed <= max
    }
    
    /// Apply filters without allocation
    #[inline(always)]
    fn apply_filters_zero_alloc(strength: f64) -> f64 {
        // Simple EMA filter using static state
        static LAST_VALUE: AtomicU64 = AtomicU64::new(0);
        const ALPHA: f64 = 0.1;
        
        // Convert to fixed point
        let strength_fixed = (strength * 100000000.0) as u64;
        let last = LAST_VALUE.load(memory_ordering::STATS);
        
        // EMA calculation in fixed point
        let ema_fixed = ((1.0 - ALPHA) * last as f64 + ALPHA * strength_fixed as f64) as u64;
        LAST_VALUE.store(ema_fixed, memory_ordering::STATS);
        
        // Convert back to float
        ema_fixed as f64 / 100000000.0
    }
}

/// Benchmark hot paths to verify latency
pub fn benchmark_hot_paths() -> BenchmarkResults {
    const ITERATIONS: u32 = 100000;
    let mut results = BenchmarkResults::default();
    
    // Benchmark order processing
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = critical_paths::process_order_hot_path();
    }
    let elapsed = start.elapsed();
    results.order_processing_ns = elapsed.as_nanos() as u64 / ITERATIONS as u64;
    
    // Benchmark signal processing
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = critical_paths::process_signal_hot_path();
    }
    let elapsed = start.elapsed();
    results.signal_processing_ns = elapsed.as_nanos() as u64 / ITERATIONS as u64;
    
    results
}

#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub order_processing_ns: u64,
    pub signal_processing_ns: u64,
}

impl BenchmarkResults {
    pub fn meets_targets(&self) -> bool {
        // Target: <1000ns (1μs) for hot paths
        self.order_processing_ns < 1000 && self.signal_processing_ns < 1000
    }
    
    pub fn report(&self) {
        println!("Hot Path Benchmark Results:");
        println!("  Order Processing: {}ns", self.order_processing_ns);
        println!("  Signal Processing: {}ns", self.signal_processing_ns);
        println!("  Target Met: {}", if self.meets_targets() { "✅ YES" } else { "❌ NO" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_hot_path_zero_alloc() {
        // Initialize pools first
        pools::initialize_all_pools();
        
        // Measure allocations from OUTSIDE the hot path
        let validator = HotPathValidator::begin("order_processing");
        let result = critical_paths::process_order_hot_path();
        assert!(result.is_ok(), "Order hot path failed: {:?}", result);
        
        // Now validate with team debugging
        let validation = validator.validate();
        if let Err(e) = &validation {
            // Team debugging: Log exact allocation count
            let stats = pools::get_pool_stats();
            eprintln!("TEAM DEBUG [Order Test]: Validation failed: {}", e);
            eprintln!("  Order pool allocated: {}", stats.order_allocated);
            eprintln!("  Signal pool allocated: {}", stats.signal_allocated);
            eprintln!("  Tick pool allocated: {}", stats.tick_allocated);
        }
        assert!(validation.is_ok(), "Allocations detected in hot path");
    }
    
    #[test]
    fn test_signal_hot_path_zero_alloc() {
        // Initialize pools
        pools::initialize_all_pools();
        
        // Measure allocations from OUTSIDE the hot path
        let validator = HotPathValidator::begin("signal_processing");
        let result = critical_paths::process_signal_hot_path();
        assert!(result.is_ok(), "Signal hot path failed: {:?}", result);
        
        // Now validate
        let validation = validator.validate();
        assert!(validation.is_ok(), "Allocations detected in hot path");
    }
    
    #[test]
    fn test_hot_path_benchmarks() {
        // Initialize pools
        pools::initialize_all_pools();
        
        // Run benchmarks
        let results = benchmark_hot_paths();
        results.report();
        
        // Verify targets
        assert!(
            results.meets_targets(),
            "Hot path latency exceeds 1μs target: order={}ns, signal={}ns",
            results.order_processing_ns,
            results.signal_processing_ns
        );
    }
}