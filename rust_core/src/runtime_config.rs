// Runtime Configuration
// Nexus Optimizations: CPU pinning and Tokio tuning
// Expected gains: 10-15% variance reduction, 10-15% less task stealing

use core_affinity::{self, CoreId};
use tokio::runtime::{Builder, Runtime};
use std::thread;
use std::sync::Arc;

/// CPU pinning configuration for reduced variance
pub struct CpuAffinity {
    cores: Vec<CoreId>,
    main_thread_core: Option<CoreId>,
    worker_cores: Vec<CoreId>,
}

impl CpuAffinity {
    /// Detect available cores and create optimal pinning
    pub fn auto_detect() -> Self {
        let cores = core_affinity::get_core_ids().unwrap_or_default();
        let num_cores = cores.len();
        
        if num_cores == 0 {
            tracing::warn!("No CPU cores detected for pinning");
            return CpuAffinity {
                cores: vec![],
                main_thread_core: None,
                worker_cores: vec![],
            };
        }
        
        // Reserve first core for main thread
        let main_thread_core = cores.first().cloned();
        
        // Use remaining cores for workers
        let worker_cores = if num_cores > 1 {
            cores[1..].to_vec()
        } else {
            vec![]
        };
        
        tracing::info!(
            "CPU affinity configured: {} total cores, 1 main, {} workers",
            num_cores,
            worker_cores.len()
        );
        
        CpuAffinity {
            cores,
            main_thread_core,
            worker_cores,
        }
    }
    
    /// Pin current thread to specific core
    pub fn pin_to_core(&self, core_id: CoreId) -> bool {
        core_affinity::set_for_current(core_id)
    }
    
    /// Pin main thread
    pub fn pin_main_thread(&self) -> bool {
        if let Some(core) = self.main_thread_core {
            let result = self.pin_to_core(core);
            if result {
                tracing::info!("Main thread pinned to core {:?}", core);
            }
            result
        } else {
            false
        }
    }
}

/// Optimized Tokio runtime configuration (Nexus recommendations)
pub struct OptimizedRuntime {
    runtime: Runtime,
    cpu_affinity: Arc<CpuAffinity>,
}

impl OptimizedRuntime {
    /// Create runtime with Nexus-recommended settings
    pub fn new() -> std::io::Result<Self> {
        let cpu_affinity = Arc::new(CpuAffinity::auto_detect());
        let num_cores = cpu_affinity.cores.len();
        
        // Nexus recommendation: workers = cores - 1 (reserve 1 for main)
        let worker_threads = if num_cores > 1 {
            num_cores - 1
        } else {
            1
        };
        
        // Nexus recommendation: blocking pool = 512 threads
        let blocking_threads = 512;
        
        tracing::info!(
            "Configuring Tokio runtime: {} workers, {} blocking threads",
            worker_threads,
            blocking_threads
        );
        
        let affinity_clone = cpu_affinity.clone();
        
        let runtime = Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .max_blocking_threads(blocking_threads)
            .thread_name("bot4-worker")
            .thread_stack_size(2 * 1024 * 1024)  // 2MB stack
            .enable_all()
            .on_thread_start(move || {
                // Pin each worker to a specific core
                let thread_id = thread::current().id();
                let affinity = affinity_clone.clone();
                
                // Try to pin to available worker core
                if let Some(cores) = affinity.worker_cores.get(0) {
                    if core_affinity::set_for_current(*cores) {
                        tracing::debug!("Worker thread {:?} pinned to core {:?}", thread_id, cores);
                    }
                }
            })
            .build()?;
        
        Ok(OptimizedRuntime {
            runtime,
            cpu_affinity,
        })
    }
    
    /// Get the runtime handle
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }
    
    /// Block on a future
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: std::future::Future,
    {
        self.runtime.block_on(future)
    }
    
    /// Spawn a task
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }
}

/// System-wide performance optimizations
pub struct SystemOptimizations;

impl SystemOptimizations {
    /// Apply all recommended optimizations
    pub fn apply_all() {
        Self::set_thread_priorities();
        Self::configure_memory_allocator();
        Self::set_cpu_governor();
    }
    
    /// Set thread priorities for reduced latency
    fn set_thread_priorities() {
        #[cfg(target_os = "linux")]
        {
            use libc::{pthread_self, sched_param, sched_setscheduler, SCHED_FIFO};
            
            unsafe {
                let param = sched_param {
                    sched_priority: 50,  // Medium-high priority
                };
                
                let result = sched_setscheduler(
                    0,  // Current process
                    SCHED_FIFO,
                    &param as *const _,
                );
                
                if result == 0 {
                    tracing::info!("Thread scheduling set to SCHED_FIFO");
                } else {
                    tracing::warn!("Failed to set thread scheduling (may need root)");
                }
            }
        }
    }
    
    /// Configure memory allocator for performance
    fn configure_memory_allocator() {
        // Using system allocator as per Nexus (jemalloc only 5% improvement)
        tracing::info!("Using system allocator (jemalloc not worth 5% gain)");
    }
    
    /// Set CPU governor to performance mode
    fn set_cpu_governor() {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";
            
            match fs::read_to_string(governor_path) {
                Ok(current) => {
                    if !current.contains("performance") {
                        tracing::warn!(
                            "CPU governor is '{}', recommend 'performance' mode",
                            current.trim()
                        );
                        
                        // Try to set (requires root)
                        if let Err(e) = fs::write(governor_path, "performance") {
                            tracing::info!(
                                "Cannot set CPU governor (requires root): {}",
                                e
                            );
                        } else {
                            tracing::info!("CPU governor set to performance mode");
                        }
                    } else {
                        tracing::info!("CPU governor already in performance mode");
                    }
                }
                Err(_) => {
                    tracing::debug!("Cannot read CPU governor (may not be available)");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_affinity_detection() {
        let affinity = CpuAffinity::auto_detect();
        
        // Should detect at least 1 core
        assert!(!affinity.cores.is_empty());
        
        // Should have main thread core if any cores available
        if !affinity.cores.is_empty() {
            assert!(affinity.main_thread_core.is_some());
        }
        
        println!("Detected {} CPU cores", affinity.cores.len());
        println!("Worker cores: {}", affinity.worker_cores.len());
    }
    
    #[tokio::test]
    async fn test_optimized_runtime() {
        let runtime = OptimizedRuntime::new().unwrap();
        
        // Test spawning tasks
        let handle = runtime.spawn(async {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            42
        });
        
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }
}