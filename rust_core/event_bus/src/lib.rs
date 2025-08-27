//! # Event Bus - Ultra-Low Latency Event Processing System
//! 
//! Implements the LMAX Disruptor pattern for lock-free, high-performance event processing.
//! Consolidates 6+ event processing implementations and 30+ trading operations.
//!
//! ## Architecture (LMAX Disruptor Pattern)
//! - Single writer, multiple readers
//! - Lock-free ring buffer with cache-line padding
//! - Mechanical sympathy for CPU caches
//! - Event sourcing for replay capability
//! - <1μs publish latency target
//!
//! ## External Research Applied
//! - "The LMAX Architecture" (Fowler & Thompson, 2011)
//! - "Mechanical Sympathy" (Gil Tene)
//! - "Disruptor: High Performance Alternative to Bounded Queues" (LMAX)
//! - "Event Sourcing" (Greg Young)
//! - "Lock-Free Programming" (Herb Sutter)
//! - Aeron messaging patterns (Real Logic)

#![warn(missing_docs)]
#![allow(unsafe_code)]  // Required for lock-free ring buffer
#![allow(clippy::module_name_repetitions)]

// Core modules
pub mod disruptor;
pub mod events;
pub mod handlers;
pub mod sequencer;
pub mod processor;
pub mod replay;

// Trading operations
pub mod trading_ops;

// Utilities
pub mod metrics;

// Re-exports
pub use disruptor::{RingBuffer, WaitStrategy};
pub use events::{Event, EventType, EventPriority, EventMetadata};
pub use handlers::{EventHandler, HandlerChain, HandlerPriority};
pub use sequencer::{Sequence, SequenceBarrier};
pub use processor::{EventProcessor, BatchEventProcessor, WorkProcessor};
pub use replay::{EventStore, EventReplayer, EventJournal, ReplayCoordinator};

// Trading operations exports
pub use trading_ops::{
    TradingOperation, OrderOperation, PositionOperation,
    place_order, cancel_order, update_position, get_balance,
    validate_order,
};

// Constants
/// Default ring buffer size (power of 2 for fast modulo)
pub const DEFAULT_RING_SIZE: usize = 65_536;

/// Cache line size for padding (Intel/AMD x86_64)
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum batch size for processing
pub const MAX_BATCH_SIZE: usize = 256;

/// Version of the event bus
pub const EVENT_BUS_VERSION: &str = "1.0.0";

/// Prelude for common imports
pub mod prelude {
    pub use crate::{
        RingBuffer, WaitStrategy, Event, EventType,
        EventHandler, EventProcessor,
        place_order, cancel_order, update_position,
        DEFAULT_RING_SIZE,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_RING_SIZE, 65_536);
        assert!(DEFAULT_RING_SIZE.is_power_of_two());
        assert_eq!(CACHE_LINE_SIZE, 64);
    }
}