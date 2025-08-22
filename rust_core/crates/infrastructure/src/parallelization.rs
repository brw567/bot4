// Phase 1: Parallelization Infrastructure
// CRITICAL: Required by Nexus for performance targets
// Owner: Sam | Reviewer: Jordan
// Performance Target: 500k+ ops/sec sustained

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam::utils::CachePadded;
use dashmap::DashMap;
use rayon::prelude::*;
use anyhow::{Result, Context};

/// Global configuration for parallelization
/// Uses Rayon for data parallelism with CPU affinity
pub struct ParallelizationConfig {
    /// Number of worker threads (11 for 12-core system)
    pub worker_threads: usize,
    /// Main thread core (usually 0)
    pub main_core: usize,
    /// Worker cores (1-11)
    pub worker_cores: Vec<usize>,
    /// Enable CPU pinning
    pub enable_pinning: bool,
}

impl Default for ParallelizationConfig {
    fn default() -> Self {
        let num_cores = num_cpus::get();
        let worker_threads = num_cores.saturating_sub(1).max(1);
        
        Self {
            worker_threads,
            main_core: 0,
            worker_cores: (1..=worker_threads).collect(),
            enable_pinning: true,
        }
    }
}

/// Per-core sharding for instruments
/// Each core handles a subset of instruments to minimize contention
pub struct InstrumentSharding {
    /// Number of shards (one per worker core)
    num_shards: usize,
    /// Map of instrument to shard index
    instrument_map: DashMap<String, usize>,
    /// Round-robin counter for new instruments
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
    
    /// Get shard index for an instrument
    pub fn get_shard(&self, instrument: &str) -> usize {
        *self.instrument_map.entry(instrument.to_string())
            .or_insert_with(|| {
                self.next_shard.fetch_add(1, Ordering::Relaxed) % self.num_shards
            })
    }
    
    /// Process instruments in parallel by shard
    pub fn process_parallel<F, T>(&self, instruments: &[String], f: F) -> Vec<T>
    where
        F: Fn(&str) -> T + Sync + Send,
        T: Send,
    {
        // Group instruments by shard
        let mut shards: Vec<Vec<String>> = vec![vec![]; self.num_shards];
        for instrument in instruments {
            let shard_idx = self.get_shard(instrument);
            shards[shard_idx].push(instrument.clone());
        }
        
        // Process each shard in parallel
        shards.par_iter()
            .flat_map(|shard_instruments| {
                shard_instruments.iter()
                    .map(|inst| f(inst))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Lock-free statistics with CachePadded atomics
/// Prevents false sharing between cores
pub struct LockFreeStats {
    /// Total operations processed
    pub ops_count: CachePadded<AtomicUsize>,
    /// Total latency in nanoseconds
    pub total_latency_ns: CachePadded<AtomicUsize>,
    /// Number of errors
    pub error_count: CachePadded<AtomicUsize>,
    /// Peak throughput ops/sec
    pub peak_throughput: CachePadded<AtomicUsize>,
}

impl Default for LockFreeStats {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeStats {
    pub fn new() -> Self {
        Self {
            ops_count: CachePadded::new(AtomicUsize::new(0)),
            total_latency_ns: CachePadded::new(AtomicUsize::new(0)),
            error_count: CachePadded::new(AtomicUsize::new(0)),
            peak_throughput: CachePadded::new(AtomicUsize::new(0)),
        }
    }
    
    /// Record an operation with latency
    pub fn record_op(&self, latency_ns: u64) {
        self.ops_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns as usize, Ordering::Relaxed);
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update peak throughput if current is higher
    pub fn update_peak_throughput(&self, current: usize) {
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
    
    /// Get average latency in nanoseconds
    pub fn avg_latency_ns(&self) -> u64 {
        let ops = self.ops_count.load(Ordering::Relaxed);
        if ops == 0 {
            return 0;
        }
        let total = self.total_latency_ns.load(Ordering::Relaxed);
        (total / ops) as u64
    }
}

/// CPU affinity manager for pinning threads to cores
pub struct CpuAffinityManager {
    config: ParallelizationConfig,
}

impl CpuAffinityManager {
    pub fn new(config: ParallelizationConfig) -> Self {
        Self { config }
    }
    
    /// Initialize Rayon thread pool with CPU affinity
    pub fn initialize_rayon(&self) -> Result<()> {
        let enable_pinning = self.config.enable_pinning;
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.worker_threads)
            .thread_name(|i| format!("rayon-worker-{}", i))
            .start_handler(move |i| {
                if enable_pinning {
                    // Workers on cores 1-11
                    #[cfg(target_os = "linux")]
                    Self::pin_thread_to_core(i + 1);
                }
            })
            .build_global()
            .context("Failed to build Rayon thread pool")?;
            
        Ok(())
    }
    
    /// Pin current thread to specific core (static version for Rayon)
    #[cfg(target_os = "linux")]
    fn pin_thread_to_core(core_id: usize) {
        use libc::{cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};
        use std::mem;
        
        unsafe {
            let mut set: cpu_set_t = mem::zeroed();
            CPU_ZERO(&mut set);
            CPU_SET(core_id, &mut set);
            
            let result = sched_setaffinity(
                0, // Current thread
                mem::size_of::<cpu_set_t>(),
                &set
            );
            
            if result != 0 {
                log::warn!("Failed to pin thread to core {}: {}", core_id, result);
            } else {
                log::info!("Thread pinned to core {}", core_id);
            }
        }
    }
    
    /// Pin current thread to specific core (instance method)
    #[cfg(target_os = "linux")]
    fn pin_to_core(&self, core_id: usize) {
        Self::pin_thread_to_core(core_id);
    }
    
    #[cfg(not(target_os = "linux"))]
    fn pin_to_core(&self, core_id: usize) {
        log::warn!("CPU pinning not supported on this platform (core {})", core_id);
    }
    
    /// Pin main thread to core 0
    pub fn pin_main_thread(&self) -> Result<()> {
        if self.config.enable_pinning {
            self.pin_to_core(self.config.main_core);
        }
        Ok(())
    }
}

/// Parallel data processor using Rayon
/// Processes market data in parallel while maintaining order
pub struct ParallelProcessor<T: Send + Sync> {
    /// Sharding strategy
    sharding: Arc<InstrumentSharding>,
    /// Statistics
    stats: Arc<LockFreeStats>,
    /// Phantom data for type
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + Sync> ParallelProcessor<T> {
    pub fn new(num_shards: usize) -> Self {
        Self {
            sharding: Arc::new(InstrumentSharding::new(num_shards)),
            stats: Arc::new(LockFreeStats::new()),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Process items in parallel maintaining per-instrument order
    pub fn process_batch<F, R>(&self, items: Vec<(String, T)>, processor: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        R: Send,
    {
        let start = std::time::Instant::now();
        
        // Group by instrument for ordering
        let by_instrument: DashMap<String, Vec<T>> = DashMap::new();
        for (instrument, item) in items {
            by_instrument.entry(instrument).or_default().push(item);
        }
        
        // Process each instrument's items in parallel
        // Convert DashMap to Vec first for parallel iteration
        let instrument_items: Vec<(String, Vec<T>)> = by_instrument
            .into_iter()
            .collect();
            
        let results: Vec<R> = instrument_items
            .into_par_iter()
            .flat_map(|(_instrument, items)| {
                items.into_iter()
                    .map(&processor)
                    .collect::<Vec<_>>()
            })
            .collect();
        
        // Record stats
        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats.record_op(elapsed);
        
        results
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> &Arc<LockFreeStats> {
        &self.stats
    }
}

/// Memory ordering guidelines for different scenarios
pub mod memory_ordering {
    use std::sync::atomic::Ordering;
    
    /// For counters and statistics (no ordering required)
    pub const STATS: Ordering = Ordering::Relaxed;
    
    /// For flags and state updates (visibility required)
    pub const STATE: Ordering = Ordering::Release;
    
    /// For reading flags and states
    pub const READ_STATE: Ordering = Ordering::Acquire;
    
    /// For compare-and-swap operations
    pub const CAS_SUCCESS: Ordering = Ordering::Release;
    pub const CAS_FAILURE: Ordering = Ordering::Relaxed;
    
    /// For critical sections and barriers
    pub const CRITICAL: Ordering = Ordering::SeqCst;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_instrument_sharding() {
        let sharding = InstrumentSharding::new(4);
        
        // Same instrument should always map to same shard
        let shard1 = sharding.get_shard("BTC/USD");
        let shard2 = sharding.get_shard("BTC/USD");
        assert_eq!(shard1, shard2);
        
        // Different instruments should distribute
        let instruments = vec![
            "BTC/USD".to_string(),
            "ETH/USD".to_string(),
            "XRP/USD".to_string(),
            "ADA/USD".to_string(),
        ];
        
        let shards: Vec<usize> = instruments.iter()
            .map(|i| sharding.get_shard(i))
            .collect();
        
        // Should use multiple shards
        let unique_shards: std::collections::HashSet<_> = shards.iter().collect();
        assert!(unique_shards.len() > 1);
    }
    
    #[test]
    fn test_lock_free_stats() {
        let stats = LockFreeStats::new();
        
        // Record some operations
        stats.record_op(1000); // 1 microsecond
        stats.record_op(2000); // 2 microseconds
        stats.record_op(1500); // 1.5 microseconds
        
        assert_eq!(stats.ops_count.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_latency_ns(), 1500);
        
        // Test peak throughput update
        stats.update_peak_throughput(100_000);
        assert_eq!(stats.peak_throughput.load(Ordering::Relaxed), 100_000);
        
        stats.update_peak_throughput(50_000); // Should not update
        assert_eq!(stats.peak_throughput.load(Ordering::Relaxed), 100_000);
        
        stats.update_peak_throughput(200_000); // Should update
        assert_eq!(stats.peak_throughput.load(Ordering::Relaxed), 200_000);
    }
    
    #[test]
    fn test_parallel_processor() {
        let processor = ParallelProcessor::<i32>::new(4);
        
        let items = vec![
            ("BTC/USD".to_string(), 100),
            ("ETH/USD".to_string(), 200),
            ("BTC/USD".to_string(), 150),
            ("XRP/USD".to_string(), 300),
        ];
        
        let results = processor.process_batch(items, |x| x * 2);
        
        assert_eq!(results.len(), 4);
        assert!(results.contains(&200)); // 100 * 2
        assert!(results.contains(&400)); // 200 * 2
        assert!(results.contains(&300)); // 150 * 2
        assert!(results.contains(&600)); // 300 * 2
        
        // Check stats
        assert_eq!(processor.stats().ops_count.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_cpu_affinity_config() {
        let config = ParallelizationConfig::default();
        assert!(config.worker_threads > 0);
        assert_eq!(config.main_core, 0);
        assert!(!config.worker_cores.is_empty());
    }
}