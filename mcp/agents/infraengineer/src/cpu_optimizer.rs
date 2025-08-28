//! CPU Optimization Module for Bot4 InfraEngineer
//! CPU-only performance optimization for crypto trading

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use num_cpus;
use cpu_time::{ProcessTime, ThreadTime};
use std::collections::HashMap;
use tracing::{info, warn, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimizationParams {
    pub thread_pool_size: usize,
    pub batch_size: usize,
    pub prefetch_distance: usize,
    pub cache_line_size: usize,
    pub numa_aware: bool,
    pub simd_enabled: bool,
    pub avx2_enabled: bool,
    pub avx512_enabled: bool,
}

impl Default for CpuOptimizationParams {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            thread_pool_size: cpu_count.saturating_sub(2).max(1), // Leave 2 cores for system
            batch_size: 1024,
            prefetch_distance: 64,
            cache_line_size: 64,
            numa_aware: false,
            simd_enabled: true,
            avx2_enabled: Self::detect_avx2(),
            avx512_enabled: Self::detect_avx512(),
        }
    }
}

impl CpuOptimizationParams {
    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }
}

pub struct CpuOptimizer {
    params: Arc<RwLock<CpuOptimizationParams>>,
    performance_history: Arc<RwLock<Vec<PerformanceMetric>>>,
    auto_tune_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub operation: String,
    pub duration_us: u64,
    pub throughput: f64,
    pub cpu_usage: f64,
    pub cache_misses: u64,
    pub thread_pool_size: usize,
    pub batch_size: usize,
}

impl CpuOptimizer {
    pub fn new(auto_tune: bool) -> Self {
        Self {
            params: Arc::new(RwLock::new(CpuOptimizationParams::default())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            auto_tune_enabled: auto_tune,
        }
    }

    /// Auto-tune CPU parameters based on workload
    pub fn auto_tune(&self) -> Result<()> {
        if !self.auto_tune_enabled {
            return Ok(());
        }

        let history = self.performance_history.read();
        if history.len() < 100 {
            return Ok(()); // Not enough data
        }

        // Analyze recent performance
        let recent: Vec<_> = history.iter().rev().take(100).collect();
        
        // Calculate optimal thread pool size
        let avg_cpu_usage: f64 = recent.iter().map(|m| m.cpu_usage).sum::<f64>() / recent.len() as f64;
        let avg_throughput: f64 = recent.iter().map(|m| m.throughput).sum::<f64>() / recent.len() as f64;
        
        let mut params = self.params.write();
        
        // Adjust thread pool size
        if avg_cpu_usage < 50.0 && params.thread_pool_size < num_cpus::get() {
            params.thread_pool_size += 1;
            info!("Auto-tuning: Increasing thread pool to {}", params.thread_pool_size);
        } else if avg_cpu_usage > 90.0 && params.thread_pool_size > 2 {
            params.thread_pool_size -= 1;
            info!("Auto-tuning: Decreasing thread pool to {}", params.thread_pool_size);
        }
        
        // Adjust batch size based on cache misses
        let avg_cache_misses: f64 = recent.iter().map(|m| m.cache_misses as f64).sum::<f64>() / recent.len() as f64;
        if avg_cache_misses > 1000.0 && params.batch_size > 256 {
            params.batch_size /= 2;
            info!("Auto-tuning: Decreasing batch size to {}", params.batch_size);
        } else if avg_cache_misses < 100.0 && params.batch_size < 4096 {
            params.batch_size *= 2;
            info!("Auto-tuning: Increasing batch size to {}", params.batch_size);
        }
        
        Ok(())
    }

    /// Optimize array operations for cache efficiency
    pub fn optimize_array_operation<T, F>(&self, data: &mut [T], operation: F) -> Result<()>
    where
        T: Send + Sync,
        F: Fn(&mut T) + Send + Sync + Copy,
    {
        let params = self.params.read();
        let batch_size = params.batch_size;
        
        // Use parallel processing with optimized batch size
        data.par_chunks_mut(batch_size)
            .for_each(|chunk| {
                // Prefetch data for better cache utilization
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    let ptr = chunk.as_ptr() as *const i8;
                    _mm_prefetch(ptr, 1); // Prefetch to L2 cache
                }
                
                for item in chunk {
                    operation(item);
                }
            });
        
        Ok(())
    }

    /// Profile CPU performance for a given operation
    pub fn profile_operation<F, R>(&self, name: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> R,
    {
        let start = ProcessTime::now();
        let start_time = std::time::Instant::now();
        
        let result = operation();
        
        let cpu_time = start.elapsed();
        let wall_time = start_time.elapsed();
        
        let metric = PerformanceMetric {
            timestamp: chrono::Utc::now(),
            operation: name.to_string(),
            duration_us: wall_time.as_micros() as u64,
            throughput: 1_000_000.0 / wall_time.as_micros() as f64,
            cpu_usage: (cpu_time.as_micros() as f64 / wall_time.as_micros() as f64) * 100.0,
            cache_misses: 0, // Would need perf counters for real measurement
            thread_pool_size: self.params.read().thread_pool_size,
            batch_size: self.params.read().batch_size,
        };
        
        self.performance_history.write().push(metric.clone());
        
        debug!("Operation '{}' completed in {}μs (CPU usage: {:.1}%)", 
               name, metric.duration_us, metric.cpu_usage);
        
        // Auto-tune if enabled
        if self.auto_tune_enabled && self.performance_history.read().len() % 10 == 0 {
            self.auto_tune()?;
        }
        
        Ok(result)
    }

    /// Configure Rayon thread pool for optimal performance
    pub fn configure_thread_pool(&self) -> Result<()> {
        let params = self.params.read();
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(params.thread_pool_size)
            .thread_name(|i| format!("bot4-worker-{}", i))
            .build_global()
            .map_err(|e| anyhow!("Failed to configure thread pool: {}", e))?;
        
        info!("Configured thread pool with {} threads", params.thread_pool_size);
        Ok(())
    }

    /// Get current optimization parameters
    pub fn get_params(&self) -> CpuOptimizationParams {
        self.params.read().clone()
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        let history = self.performance_history.read();
        
        if history.is_empty() {
            return HashMap::new();
        }
        
        let mut stats = HashMap::new();
        
        // Calculate averages
        let avg_duration: f64 = history.iter().map(|m| m.duration_us as f64).sum::<f64>() / history.len() as f64;
        let avg_throughput: f64 = history.iter().map(|m| m.throughput).sum::<f64>() / history.len() as f64;
        let avg_cpu: f64 = history.iter().map(|m| m.cpu_usage).sum::<f64>() / history.len() as f64;
        
        stats.insert("avg_duration_us".to_string(), avg_duration);
        stats.insert("avg_throughput_ops_sec".to_string(), avg_throughput);
        stats.insert("avg_cpu_usage_percent".to_string(), avg_cpu);
        stats.insert("total_operations".to_string(), history.len() as f64);
        
        // Calculate percentiles
        let mut durations: Vec<_> = history.iter().map(|m| m.duration_us).collect();
        durations.sort_unstable();
        
        if let Some(&p50) = durations.get(durations.len() / 2) {
            stats.insert("p50_duration_us".to_string(), p50 as f64);
        }
        if let Some(&p99) = durations.get(durations.len() * 99 / 100) {
            stats.insert("p99_duration_us".to_string(), p99 as f64);
        }
        
        stats
    }

    /// Optimize for specific workload type
    pub fn optimize_for_workload(&self, workload: WorkloadType) -> Result<()> {
        let mut params = self.params.write();
        
        match workload {
            WorkloadType::HighFrequencyTrading => {
                // Optimize for low latency
                params.thread_pool_size = num_cpus::get().saturating_sub(1).max(1);
                params.batch_size = 256;
                params.prefetch_distance = 128;
                info!("Optimized for high-frequency trading");
            }
            WorkloadType::MachineLearning => {
                // Optimize for throughput
                params.thread_pool_size = num_cpus::get();
                params.batch_size = 2048;
                params.prefetch_distance = 256;
                info!("Optimized for machine learning workloads");
            }
            WorkloadType::DataIngestion => {
                // Balance between latency and throughput
                params.thread_pool_size = (num_cpus::get() * 3 / 4).max(1);
                params.batch_size = 1024;
                params.prefetch_distance = 192;
                info!("Optimized for data ingestion");
            }
            WorkloadType::RiskCalculation => {
                // CPU-intensive calculations
                params.thread_pool_size = num_cpus::get();
                params.batch_size = 512;
                params.prefetch_distance = 64;
                info!("Optimized for risk calculations");
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    HighFrequencyTrading,
    MachineLearning,
    DataIngestion,
    RiskCalculation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let params = CpuOptimizationParams::default();
        assert!(params.thread_pool_size > 0);
        println!("CPU cores: {}", num_cpus::get());
        println!("Thread pool size: {}", params.thread_pool_size);
        println!("AVX2 support: {}", params.avx2_enabled);
        println!("AVX512 support: {}", params.avx512_enabled);
    }

    #[test]
    fn test_auto_tuning() {
        let optimizer = CpuOptimizer::new(true);
        
        // Simulate some operations
        for i in 0..150 {
            let _ = optimizer.profile_operation(
                &format!("test_op_{}", i),
                || {
                    // Simulate work
                    std::thread::sleep(std::time::Duration::from_micros(10));
                    42
                }
            );
        }
        
        let stats = optimizer.get_performance_stats();
        assert!(stats.contains_key("avg_duration_us"));
        assert!(stats.contains_key("p50_duration_us"));
    }

    #[test]
    fn test_workload_optimization() {
        let optimizer = CpuOptimizer::new(false);
        
        optimizer.optimize_for_workload(WorkloadType::HighFrequencyTrading).unwrap();
        let params = optimizer.get_params();
        assert_eq!(params.batch_size, 256);
        
        optimizer.optimize_for_workload(WorkloadType::MachineLearning).unwrap();
        let params = optimizer.get_params();
        assert_eq!(params.batch_size, 2048);
    }
}