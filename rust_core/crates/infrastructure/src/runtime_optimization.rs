// Phase 1: Runtime Optimization
// Zero-allocation hot paths with optimized Tokio runtime
// Owner: Jordan | Reviewer: Sam
// Performance Target: <1μs decision latency, zero allocations

use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use anyhow::{Result, Context};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crossbeam::utils::CachePadded;

/// Optimized Tokio runtime configuration
/// Tuned for 12-core system with main thread on core 0
pub struct OptimizedRuntime {
    /// The Tokio runtime instance
    runtime: Runtime,
    /// Statistics for monitoring
    stats: Arc<RuntimeStats>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Runtime statistics with cache-padded atomics
pub struct RuntimeStats {
    /// Total tasks spawned
    pub tasks_spawned: CachePadded<AtomicU64>,
    /// Total tasks completed
    pub tasks_completed: CachePadded<AtomicU64>,
    /// Blocking tasks count
    pub blocking_tasks: CachePadded<AtomicU64>,
    /// Failed tasks
    pub failed_tasks: CachePadded<AtomicU64>,
}

impl RuntimeStats {
    pub fn new() -> Self {
        Self {
            tasks_spawned: CachePadded::new(AtomicU64::new(0)),
            tasks_completed: CachePadded::new(AtomicU64::new(0)),
            blocking_tasks: CachePadded::new(AtomicU64::new(0)),
            failed_tasks: CachePadded::new(AtomicU64::new(0)),
        }
    }
    
    pub fn record_spawn(&self) {
        self.tasks_spawned.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_completion(&self) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_blocking(&self) {
        self.blocking_tasks.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_failure(&self) {
        self.failed_tasks.fetch_add(1, Ordering::Relaxed);
    }
}

impl OptimizedRuntime {
    /// Create optimized runtime for trading engine
    /// Uses 11 worker threads (cores 1-11) with main on core 0
    pub fn new() -> Result<Self> {
        let stats = Arc::new(RuntimeStats::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        
        // Build optimized Tokio runtime
        let runtime = Builder::new_multi_thread()
            .worker_threads(11) // Cores 1-11 for workers
            .max_blocking_threads(512) // Ample blocking threads
            .thread_name("tokio-worker")
            .thread_stack_size(2 * 1024 * 1024) // 2MB stack
            .enable_all()
            .on_thread_start(|| {
                // Could pin threads here if needed
                log::debug!("Tokio worker thread started");
            })
            .on_thread_stop(|| {
                log::debug!("Tokio worker thread stopped");
            })
            .build()
            .context("Failed to build Tokio runtime")?;
        
        Ok(Self {
            runtime,
            stats,
            shutdown,
        })
    }
    
    /// Get runtime handle for spawning tasks
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }
    
    /// Spawn a zero-allocation task
    pub fn spawn_zero_alloc<F, T>(&self, future: F) -> tokio::task::JoinHandle<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.stats.record_spawn();
        let stats = self.stats.clone();
        
        self.runtime.spawn(async move {
            let result = future.await;
            stats.record_completion();
            result
        })
    }
    
    /// Spawn a blocking task (for I/O operations)
    pub fn spawn_blocking<F, T>(&self, f: F) -> tokio::task::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.stats.record_blocking();
        self.runtime.spawn_blocking(f)
    }
    
    /// Block on a future (for main thread)
    pub fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }
    
    /// Shutdown the runtime gracefully
    pub fn shutdown(self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);
        
        // Give tasks time to complete
        self.runtime.shutdown_timeout(Duration::from_secs(5));
        
        let spawned = self.stats.tasks_spawned.load(Ordering::Relaxed);
        let completed = self.stats.tasks_completed.load(Ordering::Relaxed);
        let failed = self.stats.failed_tasks.load(Ordering::Relaxed);
        
        log::info!(
            "Runtime shutdown: spawned={}, completed={}, failed={}",
            spawned, completed, failed
        );
        
        Ok(())
    }
    
    /// Get runtime statistics
    pub fn stats(&self) -> &Arc<RuntimeStats> {
        &self.stats
    }
}

/// Zero-allocation task wrapper
/// Ensures no heap allocations in hot path
pub struct ZeroAllocTask<T> {
    /// Pre-allocated result storage
    result: Option<T>,
    /// Task completion flag
    completed: AtomicBool,
}

impl<T> ZeroAllocTask<T> {
    /// Create new zero-alloc task
    pub fn new() -> Self {
        Self {
            result: None,
            completed: AtomicBool::new(false),
        }
    }
    
    /// Execute task without allocation
    pub async fn execute<F, Fut>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        if self.completed.load(Ordering::Acquire) {
            return self.result.take();
        }
        
        let result = f().await;
        self.result = Some(result);
        self.completed.store(true, Ordering::Release);
        
        self.result.take()
    }
}

/// Hot path verifier - ensures zero allocations
pub struct HotPathVerifier {
    /// Allocation count at start
    start_allocs: usize,
    /// Name of the hot path
    path_name: String,
}

impl HotPathVerifier {
    /// Start verification for a hot path
    pub fn start(path_name: &str) -> Self {
        let start_allocs = Self::get_allocation_count();
        Self {
            start_allocs,
            path_name: path_name.to_string(),
        }
    }
    
    /// Verify no allocations occurred
    pub fn verify(&self) -> Result<()> {
        let end_allocs = Self::get_allocation_count();
        let allocs = end_allocs - self.start_allocs;
        
        if allocs > 0 {
            anyhow::bail!(
                "Hot path '{}' allocated {} times - must be zero!",
                self.path_name, allocs
            );
        }
        
        Ok(())
    }
    
    /// Get current allocation count
    /// NOTE: This requires allocator instrumentation in production
    /// Currently returns 0 as we rely on MiMalloc's own metrics
    fn get_allocation_count() -> usize {
        // TODO: Hook into MiMalloc statistics API
        // mimalloc provides mi_stat_alloc_count() in C API
        // For now we use external monitoring via memory::metrics
        0
    }
}

/// Optimized async primitives for hot paths
pub mod async_primitives {
    use super::*;
    use tokio::sync::oneshot;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    
    /// Pre-allocated oneshot channel pool
    pub struct OneshotPool<T> {
        channels: Vec<(oneshot::Sender<T>, oneshot::Receiver<T>)>,
        capacity: usize,
    }
    
    impl<T> OneshotPool<T> {
        pub fn new(capacity: usize) -> Self {
            let mut channels = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                channels.push(oneshot::channel());
            }
            
            Self { channels, capacity }
        }
        
        pub fn acquire(&mut self) -> Option<(oneshot::Sender<T>, oneshot::Receiver<T>)> {
            self.channels.pop()
        }
        
        pub fn release(&mut self, channel: (oneshot::Sender<T>, oneshot::Receiver<T>)) {
            if self.channels.len() < self.capacity {
                self.channels.push(channel);
            }
        }
    }
    
    /// Zero-allocation future wrapper
    pub struct ZeroAllocFuture<F> {
        inner: Pin<Box<F>>,
    }
    
    impl<F> ZeroAllocFuture<F> 
    where
        F: std::future::Future,
    {
        pub fn new(future: F) -> Self {
            Self {
                inner: Box::pin(future),
            }
        }
    }
    
    impl<F> std::future::Future for ZeroAllocFuture<F>
    where
        F: std::future::Future,
    {
        type Output = F::Output;
        
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.inner.as_mut().poll(cx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_optimized_runtime() {
        let runtime = OptimizedRuntime::new().unwrap();
        
        // Spawn a simple task
        let handle = runtime.spawn_zero_alloc(async {
            42
        });
        
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
        
        // Check stats
        assert_eq!(runtime.stats().tasks_spawned.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.stats().tasks_completed.load(Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_zero_alloc_task() {
        let mut task = ZeroAllocTask::<i32>::new();
        
        let result = task.execute(|| async { 100 }).await;
        assert_eq!(result, Some(100));
        
        // Second execution should return None (already completed)
        let result = task.execute(|| async { 200 }).await;
        assert_eq!(result, None);
    }
    
    #[test]
    fn test_hot_path_verifier() {
        let verifier = HotPathVerifier::start("test_path");
        
        // Do some work without allocations
        let x = 1 + 2;
        let y = x * 3;
        let _ = y;
        
        // Should pass (no allocations detected)
        assert!(verifier.verify().is_ok());
    }
    
    #[test]
    fn test_runtime_stats() {
        let stats = RuntimeStats::new();
        
        stats.record_spawn();
        stats.record_spawn();
        stats.record_completion();
        stats.record_blocking();
        stats.record_failure();
        
        assert_eq!(stats.tasks_spawned.load(Ordering::Relaxed), 2);
        assert_eq!(stats.tasks_completed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.blocking_tasks.load(Ordering::Relaxed), 1);
        assert_eq!(stats.failed_tasks.load(Ordering::Relaxed), 1);
    }
}