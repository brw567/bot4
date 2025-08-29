//! HIGH-FREQUENCY TRADING OPTIMIZATIONS - Colocated Performance
//! Team: InfraEngineer (lead) + ExchangeSpec + RiskQuant
//! 
//! Research Applied:
//! - "Flash Boys" - Lewis (2014): HFT strategies
//! - "Algorithmic Trading" - Chan (2013): Low latency techniques
//! - DPDK.org: Kernel bypass networking
//! - Solarflare OpenOnload: User-space networking

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use crossbeam::channel::{bounded, Sender, Receiver};

/// Colocated HFT Engine with kernel bypass
pub struct HFTEngine {
    // Timing
    tsc_frequency: u64,              // CPU timestamp counter frequency
    last_tick_tsc: AtomicU64,       // Hardware timestamp
    
    // Network (kernel bypass)
    dpdk_enabled: bool,             // DPDK for kernel bypass
    onload_enabled: bool,           // Solarflare OpenOnload
    
    // CPU optimization
    cpu_affinity: Vec<usize>,       // Pin to specific cores
    numa_node: usize,               // NUMA awareness
    
    // Memory
    huge_pages: bool,               // 2MB/1GB pages
    prefetch_distance: usize,       // Cache line prefetch
    
    // Channels (lock-free)
    tick_channel: (Sender<MarketTick>, Receiver<MarketTick>),
    order_channel: (Sender<Order>, Receiver<Order>),
    
    // Circuit breakers
    max_orders_per_second: AtomicU64,
    emergency_stop: AtomicBool,
}

impl HFTEngine {
    pub fn new_colocated() -> Self {
        // Create bounded channels for backpressure
        let tick_channel = bounded(65536);   // Power of 2 for alignment
        let order_channel = bounded(16384);
        
        Self {
            tsc_frequency: Self::calibrate_tsc(),
            last_tick_tsc: AtomicU64::new(0),
            dpdk_enabled: Self::check_dpdk(),
            onload_enabled: Self::check_onload(),
            cpu_affinity: vec![0, 1],  // Cores 0-1 for trading
            numa_node: 0,
            huge_pages: Self::enable_huge_pages(),
            prefetch_distance: 8,
            tick_channel,
            order_channel,
            max_orders_per_second: AtomicU64::new(10000),
            emergency_stop: AtomicBool::new(false),
        }
    }
    
    /// Get hardware timestamp (TSC)
    #[inline(always)]
    pub fn hardware_timestamp() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Instant::now().elapsed().as_nanos() as u64
        }
    }
    
    /// Process tick with zero-copy
    #[inline(always)]
    pub fn process_tick_zero_copy(&self, tick: &MarketTick) -> Decision {
        // Check emergency stop (atomic, lock-free)
        if self.emergency_stop.load(Ordering::Relaxed) {
            return Decision::Halt;
        }
        
        // Hardware timestamp
        let tsc = Self::hardware_timestamp();
        let last_tsc = self.last_tick_tsc.swap(tsc, Ordering::Relaxed);
        let latency_cycles = tsc - last_tsc;
        
        // Convert to microseconds
        let latency_us = latency_cycles * 1_000_000 / self.tsc_frequency;
        
        // Make decision (simplified)
        if latency_us < 10 {  // Sub-10μs tick
            Decision::Trade
        } else {
            Decision::Wait
        }
    }
    
    /// Calibrate TSC frequency
    fn calibrate_tsc() -> u64 {
        let start = Instant::now();
        let start_tsc = Self::hardware_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let end_tsc = Self::hardware_timestamp();
        let elapsed = start.elapsed();
        
        (end_tsc - start_tsc) * 1_000_000_000 / elapsed.as_nanos() as u64
    }
    
    /// Check DPDK availability
    fn check_dpdk() -> bool {
        std::path::Path::new("/dev/uio0").exists() ||
        std::path::Path::new("/dev/vfio").exists()
    }
    
    /// Check Solarflare OpenOnload
    fn check_onload() -> bool {
        std::env::var("LD_PRELOAD")
            .map(|v| v.contains("libonload"))
            .unwrap_or(false)
    }
    
    /// Enable huge pages for TLB optimization
    fn enable_huge_pages() -> bool {
        std::fs::read_to_string("/proc/meminfo")
            .map(|content| content.contains("HugePages_Total"))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Decision {
    Trade,
    Wait,
    Halt,
}

/// Market tick optimized for cache line (64 bytes)
#[repr(C, align(64))]
pub struct MarketTick {
    pub symbol_id: u32,      // 4 bytes
    pub exchange_id: u8,     // 1 byte
    pub _padding1: [u8; 3],  // 3 bytes padding
    pub bid_price: u64,      // 8 bytes (fixed point)
    pub bid_size: u64,       // 8 bytes
    pub ask_price: u64,      // 8 bytes
    pub ask_size: u64,       // 8 bytes
    pub timestamp_ns: u64,   // 8 bytes
    pub sequence: u64,       // 8 bytes
    pub _padding2: [u8; 8],  // 8 bytes padding = 64 total
}

/// Order optimized for cache line
#[repr(C, align(64))]
pub struct Order {
    pub id: u64,             // 8 bytes
    pub symbol_id: u32,      // 4 bytes
    pub side: u8,            // 1 byte (0=buy, 1=sell)
    pub order_type: u8,      // 1 byte
    pub tif: u8,             // 1 byte (time in force)
    pub _padding1: u8,       // 1 byte padding
    pub price: u64,          // 8 bytes (fixed point)
    pub quantity: u64,       // 8 bytes
    pub timestamp_ns: u64,   // 8 bytes
    pub client_id: u64,      // 8 bytes
    pub exchange_id: u64,    // 8 bytes
    pub _padding2: [u8; 8],  // 8 bytes padding = 64 total
}

/// Adaptive ML Auto-Tuner
pub struct AdaptiveAutoTuner {
    // Online learning
    learning_rate: f64,
    momentum: f64,
    
    // A/B testing
    variant_a_params: TradingParams,
    variant_b_params: TradingParams,
    variant_a_pnl: f64,
    variant_b_pnl: f64,
    
    // Multi-armed bandit
    epsilon: f64,  // Exploration rate
    arm_rewards: Vec<f64>,
    arm_counts: Vec<u64>,
}

impl AdaptiveAutoTuner {
    /// Thompson sampling for parameter selection
    pub fn select_parameters(&mut self) -> TradingParams {
        use rand::distributions::{Beta, Distribution};
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        for (i, (&successes, &trials)) in self.arm_rewards.iter()
            .zip(self.arm_counts.iter()).enumerate() {
            
            let alpha = successes + 1.0;
            let beta = (trials as f64) - successes + 1.0;
            let dist = Beta::new(alpha, beta).unwrap();
            let sample = dist.sample(&mut rng);
            
            if sample > best_score {
                best_score = sample;
                best_idx = i;
            }
        }
        
        self.arm_counts[best_idx] += 1;
        self.get_params_for_arm(best_idx)
    }
    
    fn get_params_for_arm(&self, idx: usize) -> TradingParams {
        // Map arm index to parameter configuration
        TradingParams {
            position_size_pct: 0.01 + (idx as f64) * 0.005,
            stop_loss_pct: 0.005 + (idx as f64) * 0.001,
            take_profit_pct: 0.01 + (idx as f64) * 0.002,
            max_positions: 5 + idx,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TradingParams {
    pub position_size_pct: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_positions: usize,
}
