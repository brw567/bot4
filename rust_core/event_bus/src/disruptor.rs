//! # Disruptor Implementation - Lock-Free Ring Buffer
//! 
//! Core implementation of the LMAX Disruptor pattern for ultra-low latency.
//! Uses mechanical sympathy principles for optimal CPU cache utilization.
//!
//! ## Design Principles
//! - Single producer, multiple consumers (SPMC)
//! - Lock-free using atomic operations
//! - Cache-line padding to prevent false sharing
//! - Pre-allocated memory to avoid GC pressure
//! - Batch processing for throughput
//!
//! ## Performance Targets
//! - <1μs publish latency
//! - 10M+ events/second throughput
//! - Zero allocation after initialization

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use cache_padded::CachePadded;
use crossbeam::utils::CachePadded as CrossbeamPadded;
use parking_lot::{Mutex, Condvar};
use tracing::{trace, debug, warn};

use crate::events::Event;
use crate::sequencer::{Sequence, SequenceBarrier};
use crate::handlers::EventHandler;

/// Configuration for the Disruptor
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct DisruptorConfig {
    /// Size of the ring buffer (must be power of 2)
    pub ring_size: usize,
    /// Wait strategy for consumers
    pub wait_strategy: WaitStrategy,
    /// Producer type
    pub producer_type: ProducerType,
    /// Enable event batching
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for DisruptorConfig {
    fn default() -> Self {
        Self {
            ring_size: crate::DEFAULT_RING_SIZE,
            wait_strategy: WaitStrategy::BusySpin,
            producer_type: ProducerType::Single,
            enable_batching: true,
            max_batch_size: crate::MAX_BATCH_SIZE,
            enable_metrics: true,
        }
    }
}

/// Wait strategies for consumers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum WaitStrategy {
    /// Busy spin (lowest latency, highest CPU usage)
    BusySpin,
    /// Yield thread (balanced)
    Yield,
    /// Park thread (lowest CPU, higher latency)
    Park,
    /// Hybrid (spin then yield then park)
    Hybrid { spin_tries: u32, yield_tries: u32 },
}

/// Producer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum ProducerType {
    /// Single producer (fastest)
    Single,
    /// Multiple producers (requires CAS operations)
    Multi,
}

/// Lock-free ring buffer implementation
/// TODO: Add docs
pub struct RingBuffer<T> {
    /// Buffer storage (pre-allocated)
    buffer: Vec<UnsafeCell<Option<T>>>,
    /// Size of the buffer (power of 2)
    size: usize,
    /// Mask for fast modulo (size - 1)
    mask: usize,
    /// Cursor for the next write position
    cursor: CachePadded<AtomicUsize>,
    /// Gating sequences for readers
    gating_sequences: Vec<Arc<Sequence>>,
    /// Cache of minimum gating sequence
    cached_gate: CachePadded<AtomicUsize>,
    /// Shutdown flag
    shutdown: AtomicBool,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Ring buffer size must be power of 2");
        assert!(size >= 2, "Ring buffer size must be at least 2");
        
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(UnsafeCell::new(None));
        }
        
        Self {
            buffer,
            size,
            mask: size - 1,
            cursor: CachePadded::new(AtomicUsize::new(0)),
            gating_sequences: Vec::new(),
            cached_gate: CachePadded::new(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
        }
    }
    
    /// Publish an event to the ring buffer
    pub fn publish(&self, sequence: usize, event: T) {
        let index = sequence & self.mask;
        unsafe {
            let cell = &self.buffer[index];
            let slot = &mut *cell.get();
            *slot = Some(event);
        }
        
        // Update cursor to indicate this slot is written
        self.cursor.store(sequence, Ordering::Release);
    }
    
    /// Get the next available sequence for writing
    pub fn next(&self) -> Option<usize> {
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }
        
        let current = self.cursor.load(Ordering::Acquire);
        let next = current + 1;
        
        // Check if we would wrap around and overwrite unread data
        loop {
            let min_gate = self.get_min_gating_sequence();
            let wrap_point = next - self.size;
            
            if wrap_point <= min_gate {
                // Safe to proceed
                break;
            }
            
            // Need to wait for readers to catch up
            std::hint::spin_loop();
        }
        
        Some(next)
    }
    
    /// Claim a sequence for writing (single producer)
    pub fn claim(&self) -> Option<usize> {
        self.next().map(|seq| {
            self.cursor.store(seq, Ordering::Release);
            seq
        })
    }
    
    /// Claim a sequence for writing (multi-producer)
    pub fn claim_multi(&self) -> Option<usize> {
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }
        
        loop {
            let current = self.cursor.load(Ordering::Acquire);
            let next = current + 1;
            
            // CAS operation for multi-producer
            match self.cursor.compare_exchange_weak(
                current,
                next,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(next),
                Err(_) => std::hint::spin_loop(),
            }
        }
    }
    
    /// Get an event at the given sequence (for reading)
    pub unsafe fn get(&self, sequence: usize) -> Option<&T> {
        let index = sequence & self.mask;
        let cell = &self.buffer[index];
        (*cell.get()).as_ref()
    }
    
    /// Get minimum gating sequence (slowest reader)
    fn get_min_gating_sequence(&self) -> usize {
        if self.gating_sequences.is_empty() {
            return usize::MAX;
        }
        
        // Check cached value first
        let cached = self.cached_gate.load(Ordering::Acquire);
        
        // Periodically update cache
        let mut min = usize::MAX;
        for seq in &self.gating_sequences {
            let val = seq.get();
            if val < min {
                min = val;
            }
        }
        
        // Update cache if changed
        if min != cached {
            self.cached_gate.store(min, Ordering::Release);
        }
        
        min
    }
    
    /// Add a gating sequence (reader)
    pub fn add_gating_sequence(&mut self, sequence: Arc<Sequence>) {
        self.gating_sequences.push(sequence);
    }
    
    /// Shutdown the ring buffer
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

/// Main Disruptor coordinator
/// TODO: Add docs
pub struct Disruptor<T> {
    /// Configuration
    config: DisruptorConfig,
    /// Ring buffer
    ring_buffer: Arc<RingBuffer<T>>,
    /// Event handlers
    handlers: Vec<Box<dyn EventHandler<T>>>,
    /// Sequence barriers for coordination
    barriers: Vec<SequenceBarrier>,
    /// Producer sequence
    producer_sequence: Arc<Sequence>,
    /// Running flag
    running: AtomicBool,
}

impl<T: Send + Sync + 'static> Disruptor<T> {
    /// Create a new Disruptor
    pub fn new(config: DisruptorConfig) -> Self {
        let ring_buffer = Arc::new(RingBuffer::new(config.ring_size));
        let producer_sequence = Arc::new(Sequence::new(0));
        
        Self {
            config,
            ring_buffer,
            handlers: Vec::new(),
            barriers: Vec::new(),
            producer_sequence,
            running: AtomicBool::new(false),
        }
    }
    
    /// Add an event handler
    pub fn add_handler(&mut self, handler: Box<dyn EventHandler<T>>) {
        self.handlers.push(handler);
    }
    
    /// Start the disruptor
    pub fn start(&self) {
        if self.running.swap(true, Ordering::AcqRel) {
            warn!("Disruptor already running");
            return;
        }
        
        debug!("Starting Disruptor with {} handlers", self.handlers.len());
        
        // Start event processors for each handler
        for (i, handler) in self.handlers.iter().enumerate() {
            let barrier = SequenceBarrier::new(
                self.producer_sequence.clone(),
                self.config.wait_strategy,
            );
            
            // Spawn processor thread
            self.spawn_processor(handler, barrier, i);
        }
    }
    
    /// Spawn a processor thread for a handler
    fn spawn_processor(
        &self,
        handler: &Box<dyn EventHandler<T>>,
        barrier: SequenceBarrier,
        id: usize,
    ) {
        let ring_buffer = self.ring_buffer.clone();
        let wait_strategy = self.config.wait_strategy;
        
        std::thread::spawn(move || {
            debug!("Event processor {} started", id);
            
            let mut sequence = 0;
            let mut batch = Vec::with_capacity(crate::MAX_BATCH_SIZE);
            
            loop {
                // Wait for next available sequence
                match barrier.wait_for(sequence + 1, wait_strategy) {
                    Some(available) => {
                        // Process batch of events
                        batch.clear();
                        
                        while sequence < available && batch.len() < crate::MAX_BATCH_SIZE {
                            sequence += 1;
                            
                            // Get event from ring buffer
                            if let Some(event) = unsafe { ring_buffer.get(sequence) } {
                                batch.push(event);
                            }
                        }
                        
                        // Process batch
                        if !batch.is_empty() {
                            trace!("Processor {} handling batch of {} events", id, batch.len());
                            // handler.on_batch(&batch);
                        }
                    }
                    None => {
                        debug!("Event processor {} shutting down", id);
                        break;
                    }
                }
            }
        });
    }
    
    /// Publish an event (single producer)
    pub fn publish(&self, event: T) -> Result<usize, PublishError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(PublishError::NotRunning);
        }
        
        // Claim next sequence
        let sequence = self.ring_buffer
            .claim()
            .ok_or(PublishError::Shutdown)?;
        
        // Write event to ring buffer
        unsafe {
            self.ring_buffer.publish(sequence, event);
        }
        
        // Update producer sequence
        self.producer_sequence.set(sequence);
        
        Ok(sequence)
    }
    
    /// Publish multiple events (batch)
    pub fn publish_batch(&self, events: Vec<T>) -> Result<Vec<usize>, PublishError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(PublishError::NotRunning);
        }
        
        let mut sequences = Vec::with_capacity(events.len());
        
        for event in events {
            let seq = self.publish(event)?;
            sequences.push(seq);
        }
        
        Ok(sequences)
    }
    
    /// Shutdown the disruptor
    pub fn shutdown(&self) {
        if !self.running.swap(false, Ordering::AcqRel) {
            return;
        }
        
        debug!("Shutting down Disruptor");
        self.ring_buffer.shutdown();
    }
}

/// Errors that can occur during publishing
#[derive(Debug, thiserror::Error)]
/// TODO: Add docs
pub enum PublishError {
    #[error("Disruptor is not running")]
    NotRunning,
    
    #[error("Disruptor is shutting down")]
    Shutdown,
    
    #[error("Ring buffer is full")]
    BufferFull,
}

/// Sequence barrier wait implementation
impl SequenceBarrier {
    /// Wait for a sequence to become available
    pub fn wait_for(&self, sequence: usize, strategy: WaitStrategy) -> Option<usize> {
        let mut spin_count = 0;
        let mut yield_count = 0;
        
        loop {
            let available = self.cursor.get();
            
            if available >= sequence {
                return Some(available);
            }
            
            if self.is_alerted() {
                return None;
            }
            
            // Apply wait strategy
            match strategy {
                WaitStrategy::BusySpin => {
                    std::hint::spin_loop();
                }
                WaitStrategy::Yield => {
                    std::thread::yield_now();
                }
                WaitStrategy::Park => {
                    std::thread::park_timeout(std::time::Duration::from_micros(1));
                }
                WaitStrategy::Hybrid { spin_tries, yield_tries } => {
                    if spin_count < spin_tries {
                        std::hint::spin_loop();
                        spin_count += 1;
                    } else if yield_count < yield_tries {
                        std::thread::yield_now();
                        yield_count += 1;
                    } else {
                        std::thread::park_timeout(std::time::Duration::from_micros(10));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_power_of_two() {
        let rb = RingBuffer::<u64>::new(64);
        assert_eq!(rb.size, 64);
        assert_eq!(rb.mask, 63);
    }
    
    #[test]
    #[should_panic(expected = "Ring buffer size must be power of 2")]
    fn test_ring_buffer_not_power_of_two() {
        let _ = RingBuffer::<u64>::new(63);
    }
    
    #[test]
    fn test_single_producer_claim() {
        let rb = RingBuffer::<u64>::new(8);
        
        let seq1 = rb.claim();
        assert_eq!(seq1, Some(1));
        
        let seq2 = rb.claim();
        assert_eq!(seq2, Some(2));
    }
}