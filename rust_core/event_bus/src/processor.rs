//! # Event Processors - Batch and work processors

use crate::handlers::EventHandler;
use crate::sequencer::{Sequence, SequenceBarrier};
use std::sync::Arc;

/// Event processor trait
pub trait EventProcessor: Send + Sync {
    /// Run the processor
    fn run(&self);
    
    /// Get current sequence
    fn get_sequence(&self) -> usize;
    
    /// Halt the processor
    fn halt(&self);
}

/// Batch event processor
/// TODO: Add docs
pub struct BatchEventProcessor<T> {
    sequence: Arc<Sequence>,
    barrier: SequenceBarrier,
    handler: Arc<dyn EventHandler<T>>,
}

impl<T: Send + Sync> BatchEventProcessor<T> {
    /// Create new batch processor
    pub fn new(
        barrier: SequenceBarrier,
        handler: Arc<dyn EventHandler<T>>,
    ) -> Self {
        Self {
            sequence: Arc::new(Sequence::new(0)),
            barrier,
            handler,
        }
    }
}

impl<T: Send + Sync + 'static> EventProcessor for BatchEventProcessor<T> {
    fn run(&self) {
        // Implementation would process batches
    }
    
    fn get_sequence(&self) -> usize {
        self.sequence.get()
    }
    
    fn halt(&self) {
        self.barrier.alert();
    }
}

/// Work processor for parallel processing
/// TODO: Add docs
pub struct WorkProcessor<T> {
    sequence: Arc<Sequence>,
    barrier: SequenceBarrier,
    handler: Arc<dyn EventHandler<T>>,
}

impl<T: Send + Sync> WorkProcessor<T> {
    /// Create new work processor
    pub fn new(
        barrier: SequenceBarrier,
        handler: Arc<dyn EventHandler<T>>,
    ) -> Self {
        Self {
            sequence: Arc::new(Sequence::new(0)),
            barrier,
            handler,
        }
    }
}