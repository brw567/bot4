// Bot4 Ring Buffers - Lock-free SPSC/MPMC
// Day 2 Sprint - Critical for hot paths
// Owner: Jordan
// Target: <15ns enqueue/dequeue

use super::metrics::metrics;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam::queue::{ArrayQueue, SegQueue};

/// Single Producer Single Consumer ring buffer
/// Optimal for market data feed -> strategy pipeline
/// TODO: Add docs
pub struct SpscRing<T> {
    buffer: Arc<ArrayQueue<T>>,
    cached_size: usize,
}

impl<T> SpscRing<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ArrayQueue::new(capacity)),
            cached_size: capacity,
        }
    }
    
    /// Push item to ring - returns false if full
    #[inline(always)]
    pub fn push(&self, item: T) -> bool {
        let result = self.buffer.push(item).is_ok();
        if result {
            metrics().record_ring_push();
        }
        result
    }
    
    /// Pop item from ring - returns None if empty
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let result = self.buffer.pop();
        if result.is_some() {
            metrics().record_ring_pop();
        }
        result
    }
    
    /// Current number of items
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Check if full
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.cached_size
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.cached_size
    }
}

/// Multi Producer Multi Consumer ring buffer
/// For control plane and non-critical paths
/// TODO: Add docs
pub struct MpmcRing<T> {
    buffer: Arc<ArrayQueue<T>>,
    cached_size: usize,
}

impl<T> MpmcRing<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ArrayQueue::new(capacity)),
            cached_size: capacity,
        }
    }
    
    /// Push item to ring - returns false if full
    #[inline(always)]
    pub fn push(&self, item: T) -> bool {
        let result = self.buffer.push(item).is_ok();
        if result {
            metrics().record_ring_push();
        }
        result
    }
    
    /// Pop item from ring - returns None if empty
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let result = self.buffer.pop();
        if result.is_some() {
            metrics().record_ring_pop();
        }
        result
    }
    
    /// Current number of items
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.cached_size
    }
}

/// Unbounded MPMC queue for non-critical paths
/// TODO: Add docs
pub struct UnboundedQueue<T> {
    queue: Arc<SegQueue<T>>,
}

impl<T> Default for UnboundedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> UnboundedQueue<T> {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(SegQueue::new()),
        }
    }
    
    /// Push item - always succeeds
    #[inline(always)]
    pub fn push(&self, item: T) {
        self.queue.push(item);
    }
    
    /// Pop item - returns None if empty
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }
    
    /// Approximate length (may be stale)
    pub fn len_approx(&self) -> usize {
        self.queue.len()
    }
    
    /// Check if likely empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Specialized tick data ring for market data
/// TODO: Add docs
pub struct TickRing {
    ring: SpscRing<super::pools::Tick>,
    dropped: AtomicUsize,
}

impl Default for TickRing {
    fn default() -> Self {
        Self::new()
    }
}

impl TickRing {
    const CAPACITY: usize = 100_000; // 100k ticks buffer
    
    pub fn new() -> Self {
        Self {
            ring: SpscRing::new(Self::CAPACITY),
            dropped: AtomicUsize::new(0),
        }
    }
    
    /// Push tick - drops oldest if full
    pub fn push_tick(&self, tick: super::pools::Tick) -> bool {
        if !self.ring.push(tick.clone()) {
            // Ring full, drop oldest
            self.dropped.fetch_add(1, Ordering::Relaxed);
            metrics().record_ring_drop();
            self.ring.pop(); // Drop oldest
            self.ring.push(tick) // Try again
        } else {
            true
        }
    }
    
    /// Pop tick for processing
    pub fn pop_tick(&self) -> Option<super::pools::Tick> {
        self.ring.pop()
    }
    
    /// Get dropped tick count
    pub fn dropped_count(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
    
    /// Current depth
    pub fn depth(&self) -> usize {
        self.ring.len()
    }
}

/// Order queue for order management
/// TODO: Add docs
pub struct OrderQueue {
    ring: MpmcRing<super::pools::Order>,
    rejected: AtomicUsize,
}

impl Default for OrderQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderQueue {
    const CAPACITY: usize = 10_000; // 10k orders buffer
    
    pub fn new() -> Self {
        Self {
            ring: MpmcRing::new(Self::CAPACITY),
            rejected: AtomicUsize::new(0),
        }
    }
    
    /// Submit order - returns false if queue full
    pub fn submit(&self, order: super::pools::Order) -> bool {
        if !self.ring.push(order) {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            false
        } else {
            true
        }
    }
    
    /// Take order for processing
    pub fn take(&self) -> Option<super::pools::Order> {
        self.ring.pop()
    }
    
    /// Get rejected count
    pub fn rejected_count(&self) -> usize {
        self.rejected.load(Ordering::Relaxed)
    }
    
    /// Current queue depth
    pub fn depth(&self) -> usize {
        self.ring.len()
    }
    
    /// Queue pressure (0.0 - 1.0)
    pub fn pressure(&self) -> f64 {
        self.ring.len() as f64 / Self::CAPACITY as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::pools;
    use std::time::Instant;
    use std::thread;
    
    #[test]
    fn test_spsc_performance() {
        let ring = SpscRing::new(1000);
        const ITERATIONS: usize = 1_000_000;
        
        let start = Instant::now();
        for i in 0..ITERATIONS {
            ring.push(i);
            ring.pop();
        }
        let elapsed = start.elapsed();
        
        let per_op = elapsed.as_nanos() / (ITERATIONS * 2) as u128;
        println!("SPSC push/pop: {}ns per operation", per_op);
        
        assert!(per_op < 50, "SPSC too slow: {}ns", per_op);
    }
    
    #[test]
    fn test_mpmc_concurrent() {
        let ring = Arc::new(MpmcRing::new(10000));
        let mut handles = vec![];
        
        // Spawn producers
        for i in 0..4 {
            let r = ring.clone();
            handles.push(thread::spawn(move || {
                for j in 0..1000 {
                    while !r.push(i * 1000 + j) {
                        thread::yield_now();
                    }
                }
            }));
        }
        
        // Spawn consumers
        for _ in 0..4 {
            let r = ring.clone();
            handles.push(thread::spawn(move || {
                let mut count = 0;
                while count < 1000 {
                    if r.pop().is_some() {
                        count += 1;
                    }
                }
            }));
        }
        
        // Wait for completion
        for h in handles {
            h.join().unwrap();
        }
        
        // Ring should be empty
        assert!(ring.is_empty());
    }
    
    #[test]
    fn test_tick_ring_overflow() {
        let ring = TickRing::new();
        
        // Fill ring beyond capacity
        for i in 0..150_000 {
            let tick = pools::Tick {
                symbol: format!("TEST{}", i % 10),
                bid: i as f64,
                ask: i as f64 + 0.1,
                bid_volume: 100.0,
                ask_volume: 100.0,
                timestamp: i as u64,
            };
            ring.push_tick(tick);
        }
        
        // Should have dropped ~50k ticks
        assert!(ring.dropped_count() > 40_000);
        assert!(ring.depth() <= 100_000);
    }
}