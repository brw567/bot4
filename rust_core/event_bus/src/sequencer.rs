//! # Sequencer - Sequence management for event ordering

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use cache_padded::CachePadded;

/// Sequence counter with cache padding
/// TODO: Add docs
pub struct Sequence {
    /// Current sequence value
    value: CachePadded<AtomicUsize>,
}

impl Sequence {
    /// Create new sequence
    pub fn new(initial: usize) -> Self {
        Self {
            value: CachePadded::new(AtomicUsize::new(initial)),
        }
    }
    
    /// Get current value
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Acquire)
    }
    
    /// Set value
    pub fn set(&self, value: usize) {
        self.value.store(value, Ordering::Release);
    }
    
    /// Increment and get
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::AcqRel) + 1
    }
}

/// Sequence barrier for coordination
/// TODO: Add docs
pub struct SequenceBarrier {
    /// Cursor sequence to track
    pub cursor: Arc<Sequence>,
    /// Alert flag for shutdown
    alerted: AtomicBool,
    /// Dependent sequences
    dependents: Vec<Arc<Sequence>>,
}

impl SequenceBarrier {
    /// Create new barrier
    pub fn new(cursor: Arc<Sequence>, wait_strategy: crate::disruptor::WaitStrategy) -> Self {
        Self {
            cursor,
            alerted: AtomicBool::new(false),
            dependents: Vec::new(),
        }
    }
    
    /// Check if alerted (shutdown)
    pub fn is_alerted(&self) -> bool {
        self.alerted.load(Ordering::Acquire)
    }
    
    /// Alert the barrier
    pub fn alert(&self) {
        self.alerted.store(true, Ordering::Release);
    }
}