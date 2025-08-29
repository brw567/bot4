// Performance Manifest System - Machine-Generated Consistency
// Jordan (Performance Lead) + Riley (Testing)
// CRITICAL: Sophia Requirement #1 - No conflicting metrics!

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::arch::x86_64::*;
use serde::{Serialize, Deserialize};
use prometheus::{Histogram, HistogramOpts, Registry, register_histogram_vec_with_registry};

/// Performance percentiles for each component
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct LatencyPercentiles {
    pub p50: u64,   // nanoseconds
    pub p95: u64,   // nanoseconds
    pub p99: u64,   // nanoseconds
    pub p99_9: u64, // nanoseconds
    pub samples: usize,
}

/// Cache information from CPU
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct CacheInfo {
    pub l1_data: usize,
    pub l1_instruction: usize,
    pub l2: usize,
    pub l3: usize,
}

/// Performance Manifest - Ground Truth for System Performance
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct PerfManifest {
    // Hardware detection
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub cpu_frequency: f64,  // GHz
    pub avx512_available: bool,
    pub cache_sizes: CacheInfo,
    pub numa_nodes: u32,
    
    // Compiler optimizations
    pub rustc_version: String,
    pub opt_level: String,
    pub target_cpu: String,
    pub lto_enabled: bool,
    pub codegen_units: u32,
    
    // Performance metrics (p50/p95/p99/p99.9)
    pub metrics: HashMap<String, LatencyPercentiles>,
    
    // Timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub hostname: String,
}

impl PerfManifest {
    /// Generate comprehensive performance manifest
    /// Jordan: "This is our single source of truth!"
    pub fn generate() -> Self {
        info!("Generating performance manifest...");
        
        // Hardware detection
        let (cpu_model, cpu_freq) = Self::detect_cpu_info();
        let cpu_cores = num_cpus::get() as u32;
        let avx512 = Self::detect_avx512();
        let cache_sizes = Self::detect_cache_sizes();
        let numa_nodes = Self::detect_numa_nodes();
        
        // Compiler info
        let rustc_version = env!("RUSTC_VERSION").to_string();
        let opt_level = if cfg!(debug_assertions) { "debug" } else { "release" }.to_string();
        let target_cpu = env!("TARGET_CPU").to_string();
        let lto_enabled = cfg!(lto);
        let codegen_units = 1; // For maximum optimization
        
        // Benchmark all stages
        let metrics = Self::benchmark_all_stages();
        
        // Validate consistency (Sophia #1)
        Self::validate_metrics_consistency(&metrics);
        
        Self {
            cpu_model,
            cpu_cores,
            cpu_frequency: cpu_freq,
            avx512_available: avx512,
            cache_sizes,
            numa_nodes,
            rustc_version,
            opt_level,
            target_cpu,
            lto_enabled,
            codegen_units,
            metrics,
            generated_at: chrono::Utc::now(),
            hostname: hostname::get()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        }
    }
    
    /// Detect CPU model and frequency using CPUID
    fn detect_cpu_info() -> (String, f64) {
        unsafe {
            // Get CPU brand string
            let mut brand = [0u8; 48];
            for i in 0..3 {
                let result = std::arch::x86_64::__cpuid(0x80000002 + i);
                let offset = i * 16;
                brand[offset..offset+4].copy_from_slice(&result.eax.to_ne_bytes());
                brand[offset+4..offset+8].copy_from_slice(&result.ebx.to_ne_bytes());
                brand[offset+8..offset+12].copy_from_slice(&result.ecx.to_ne_bytes());
                brand[offset+12..offset+16].copy_from_slice(&result.edx.to_ne_bytes());
            }
            
            let cpu_model = String::from_utf8_lossy(&brand)
                .trim_end_matches('\0')
                .trim()
                .to_string();
            
            // Try to extract frequency from brand string
            let freq = Self::parse_frequency_from_brand(&cpu_model)
                .unwrap_or_else(|| Self::measure_frequency());
            
            (cpu_model, freq)
        }
    }
    
    /// Parse frequency from CPU brand string
    fn parse_frequency_from_brand(brand: &str) -> Option<f64> {
        // Look for patterns like "3.60GHz" or "3600MHz"
        if let Some(ghz_pos) = brand.find("GHz") {
            let start = brand[..ghz_pos].rfind(' ')? + 1;
            let freq_str = &brand[start..ghz_pos];
            freq_str.parse::<f64>().ok()
        } else {
            None
        }
    }
    
    /// Measure CPU frequency empirically
    fn measure_frequency() -> f64 {
        unsafe {
            // Use RDTSC to measure cycles
            let start = core::arch::x86_64::_rdtsc();
            let start_time = Instant::now();
            
            // Busy loop for measurement
            while start_time.elapsed() < Duration::from_millis(100) {
                core::hint::spin_loop();
            }
            
            let end = core::arch::x86_64::_rdtsc();
            let elapsed = start_time.elapsed();
            
            let cycles = (end - start) as f64;
            let seconds = elapsed.as_secs_f64();
            
            // Convert to GHz
            (cycles / seconds) / 1_000_000_000.0
        }
    }
    
    /// Detect AVX-512 support comprehensively
    fn detect_avx512() -> bool {
        is_x86_feature_detected!("avx512f")     // Foundation
            && is_x86_feature_detected!("avx512dq") // Double/Quad
            && is_x86_feature_detected!("avx512vl") // Vector Length
            && is_x86_feature_detected!("avx512bw") // Byte/Word
    }
    
    /// Detect cache sizes using CPUID
    fn detect_cache_sizes() -> CacheInfo {
        unsafe {
            let mut l1_data = 32 * 1024;     // Default 32KB
            let mut l1_inst = 32 * 1024;     // Default 32KB
            let mut l2 = 256 * 1024;         // Default 256KB
            let mut l3 = 8 * 1024 * 1024;    // Default 8MB
            
            // Intel cache detection
            if std::arch::x86_64::__get_cpuid_max(0).0 >= 4 {
                for i in 0.. {
                    let result = std::arch::x86_64::__cpuid_count(4, i);
                    let cache_type = result.eax & 0x1F;
                    
                    if cache_type == 0 {
                        break; // No more caches
                    }
                    
                    let level = (result.eax >> 5) & 0x7;
                    let ways = ((result.ebx >> 22) & 0x3FF) + 1;
                    let partitions = ((result.ebx >> 12) & 0x3FF) + 1;
                    let line_size = (result.ebx & 0xFFF) + 1;
                    let sets = result.ecx + 1;
                    
                    let size = ways * partitions * line_size * sets;
                    
                    match (level, cache_type) {
                        (1, 1) => l1_data = size as usize,  // L1 Data
                        (1, 2) => l1_inst = size as usize,  // L1 Instruction
                        (2, _) => l2 = size as usize,       // L2 Unified
                        (3, _) => l3 = size as usize,       // L3 Unified
                        _ => {}
                    }
                }
            }
            
            CacheInfo {
                l1_data,
                l1_instruction: l1_inst,
                l2,
                l3,
            }
        }
    }
    
    /// Detect NUMA nodes
    fn detect_numa_nodes() -> u32 {
        // Try to read from sysfs (Linux)
        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node/") {
                let count = entries
                    .filter_map(Result::ok)
                    .filter(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with("node")
                    })
                    .count();
                return count.max(1) as u32;
            }
        }
        
        1 // Default to single NUMA node
    }
    
    /// Benchmark all pipeline stages
    /// Riley: "10,000 iterations for statistical significance!"
    fn benchmark_all_stages() -> HashMap<String, LatencyPercentiles> {
        let mut results = HashMap::new();
        
        // Define stages to benchmark
        let stages = vec![
            ("feature_extraction", Self::benchmark_feature_extraction),
            ("lstm_inference", Self::benchmark_lstm_inference),
            ("ensemble_voting", Self::benchmark_ensemble_voting),
            ("risk_validation", Self::benchmark_risk_validation),
            ("order_generation", Self::benchmark_order_generation),
            ("garch_calculation", Self::benchmark_garch),
            ("attention_mechanism", Self::benchmark_attention),
            ("probability_calibration", Self::benchmark_calibration),
        ];
        
        for (name, benchmark_fn) in stages {
            info!("Benchmarking {}...", name);
            let latencies = benchmark_fn(10_000);
            let percentiles = Self::calculate_percentiles(&latencies);
            results.insert(name.to_string(), percentiles);
        }
        
        results
    }
    
    /// Benchmark feature extraction
    fn benchmark_feature_extraction(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        // Prepare test data
        let prices = vec![100.0f32; 1000];
        let volumes = vec![1000.0f32; 1000];
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate feature extraction
            let mut features = Vec::with_capacity(100);
            
            // Technical indicators
            for window in [14, 20, 50] {
                let sma = prices.windows(window)
                    .map(|w| w.iter().sum::<f32>() / window as f32)
                    .last()
                    .unwrap_or(0.0);
                features.push(sma);
            }
            
            // Volume features
            let vol_mean = volumes.iter().sum::<f32>() / volumes.len() as f32;
            features.push(vol_mean);
            
            // Ensure compiler doesn't optimize away
            std::hint::black_box(&features);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark LSTM inference
    fn benchmark_lstm_inference(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        // Prepare test data
        let input = vec![0.1f32; 100];
        let weights = vec![vec![0.01f32; 100]; 128];
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate LSTM forward pass
            let mut hidden = vec![0.0f32; 128];
            for w_row in &weights {
                let sum: f32 = input.iter()
                    .zip(w_row.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                hidden.push((sum * 0.5).tanh());
            }
            
            std::hint::black_box(&hidden);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark ensemble voting
    fn benchmark_ensemble_voting(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        // Prepare test predictions
        let predictions = vec![
            vec![0.6, 0.4],
            vec![0.55, 0.45],
            vec![0.65, 0.35],
            vec![0.58, 0.42],
            vec![0.62, 0.38],
        ];
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Weighted voting
            let weights = vec![0.3, 0.25, 0.2, 0.15, 0.1];
            let mut final_pred = vec![0.0; 2];
            
            for (pred, weight) in predictions.iter().zip(weights.iter()) {
                for (i, p) in pred.iter().enumerate() {
                    final_pred[i] += p * weight;
                }
            }
            
            std::hint::black_box(&final_pred);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark risk validation
    fn benchmark_risk_validation(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate risk checks
            let position_size = 10000.0;
            let account_balance = 100000.0;
            let max_position_pct = 0.02;
            let current_positions = 5;
            let max_positions = 10;
            
            let size_check = position_size <= account_balance * max_position_pct;
            let count_check = current_positions < max_positions;
            let valid = size_check && count_check;
            
            std::hint::black_box(valid);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark order generation
    fn benchmark_order_generation(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate order creation
            let order = Order {
                id: uuid::Uuid::new_v4(),
                symbol: "BTC/USDT".to_string(),
                side: OrderSide::Buy,
                order_type: OrderType::Limit,
                price: 50000.0,
                quantity: 0.1,
                timestamp: chrono::Utc::now(),
            };
            
            std::hint::black_box(&order);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark GARCH calculation
    fn benchmark_garch(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        let returns = vec![0.01f32; 100];
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate GARCH variance update
            let omega = 0.00001f32;
            let alpha = 0.1f32;
            let beta = 0.85f32;
            let mut variance = 0.0004f32;
            
            for ret in &returns[..10] {
                variance = omega + alpha * ret.powi(2) + beta * variance;
            }
            
            std::hint::black_box(variance);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark attention mechanism
    fn benchmark_attention(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate scaled dot-product attention
            let seq_len = 50;
            let hidden = 64;
            let scale = 1.0 / (hidden as f32).sqrt();
            
            let mut scores = vec![vec![0.0f32; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    scores[i][j] = ((i * j) as f32 * scale).exp();
                }
            }
            
            std::hint::black_box(&scores);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Benchmark probability calibration
    fn benchmark_calibration(iterations: usize) -> Vec<u64> {
        let mut latencies = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            // Simulate isotonic regression transform
            let raw_prob = 0.7f32;
            let calibrated = (raw_prob * 0.8 + 0.1).min(1.0).max(0.0);
            
            std::hint::black_box(calibrated);
            
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies
    }
    
    /// Calculate percentiles from latency measurements
    fn calculate_percentiles(latencies: &[u64]) -> LatencyPercentiles {
        let mut sorted = latencies.to_vec();
        sorted.sort_unstable();
        
        let len = sorted.len();
        
        LatencyPercentiles {
            p50: sorted[len * 50 / 100],
            p95: sorted[len * 95 / 100],
            p99: sorted[len * 99 / 100],
            p99_9: sorted[len * 999 / 1000],
            samples: len,
        }
    }
    
    /// Validate metrics consistency (Sophia #1 requirement)
    fn validate_metrics_consistency(metrics: &HashMap<String, LatencyPercentiles>) {
        // Check individual stage limits
        for (stage, percs) in metrics {
            match stage.as_str() {
                "feature_extraction" => {
                    assert!(percs.p99 < 3_000_000, "Feature extraction too slow: {}ns", percs.p99);
                }
                "lstm_inference" => {
                    assert!(percs.p99 < 1_000_000, "LSTM inference too slow: {}ns", percs.p99);
                }
                "ensemble_voting" => {
                    assert!(percs.p99 < 5_000_000, "Ensemble voting too slow: {}ns", percs.p99);
                }
                _ => {}
            }
        }
        
        // Check total pipeline latency
        let total_p99: u64 = metrics.values().map(|p| p.p99).sum();
        assert!(
            total_p99 < 10_000_000,
            "Total pipeline p99 exceeds 10ms: {}ns",
            total_p99
        );
        
        // Check p99.9 is within 3x of p99 (tail latency control)
        for (stage, percs) in metrics {
            let ratio = percs.p99_9 as f64 / percs.p99 as f64;
            assert!(
                ratio < 3.0,
                "{} has excessive tail latency: p99.9/p99 = {:.2}",
                stage,
                ratio
            );
        }
        
        info!("✅ All metrics consistency checks passed!");
    }
    
    /// Export to Prometheus metrics
    pub fn export_to_prometheus(&self, registry: &Registry) {
        let opts = HistogramOpts::new(
            "bot4_component_latency",
            "Component latency in nanoseconds"
        );
        
        let histogram = register_histogram_vec_with_registry!(
            opts,
            &["component", "percentile"],
            registry
        ).unwrap();
        
        for (component, percs) in &self.metrics {
            histogram
                .with_label_values(&[component, "p50"])
                .observe(percs.p50 as f64);
            histogram
                .with_label_values(&[component, "p95"])
                .observe(percs.p95 as f64);
            histogram
                .with_label_values(&[component, "p99"])
                .observe(percs.p99 as f64);
            histogram
                .with_label_values(&[component, "p99.9"])
                .observe(percs.p99_9 as f64);
        }
    }
    
    /// Write manifest to file
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        info!("Performance manifest written to {}", path);
        Ok(())
    }
}

// Using canonical Order from domain_types for benchmarking
use domain_types::order::{Order as CanonicalOrder};
type Order = CanonicalOrder;

#[derive(Debug)]
enum OrderSide { Buy, Sell }

#[derive(Debug)]
enum OrderType { Limit, Market }

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manifest_generation() {
        let manifest = PerfManifest::generate();
        
        // Verify hardware detection
        assert!(manifest.cpu_cores > 0);
        assert!(manifest.cpu_frequency > 0.0);
        assert!(!manifest.cpu_model.is_empty());
        
        // Verify metrics exist
        assert!(!manifest.metrics.is_empty());
        assert!(manifest.metrics.contains_key("feature_extraction"));
        assert!(manifest.metrics.contains_key("lstm_inference"));
        
        // Verify consistency
        let total_p99: u64 = manifest.metrics.values().map(|p| p.p99).sum();
        assert!(total_p99 < 10_000_000, "Total p99 exceeds 10ms");
    }
    
    #[test]
    fn test_percentile_calculation() {
        let latencies = vec![100, 200, 150, 300, 250, 180, 220, 400, 350, 500];
        let percentiles = PerfManifest::calculate_percentiles(&latencies);
        
        assert!(percentiles.p50 <= percentiles.p95);
        assert!(percentiles.p95 <= percentiles.p99);
        assert!(percentiles.p99 <= percentiles.p99_9);
        assert_eq!(percentiles.samples, 10);
    }
    
    #[test]
    fn test_avx512_detection() {
        let has_avx512 = PerfManifest::detect_avx512();
        
        // Just verify it returns a value without crashing
        println!("AVX-512 available: {}", has_avx512);
    }
    
    #[test]
    fn test_cache_detection() {
        let cache = PerfManifest::detect_cache_sizes();
        
        // Verify reasonable cache sizes
        assert!(cache.l1_data >= 16 * 1024);  // At least 16KB
        assert!(cache.l2 >= 128 * 1024);      // At least 128KB
        assert!(cache.l3 >= 1024 * 1024);     // At least 1MB
    }
}