//! # Event Handlers - Event processing implementations

use async_trait::async_trait;
use std::sync::Arc;
use crate::events::{Event, EventType, EventPriority};

/// Event handler trait
#[async_trait]
pub trait EventHandler<T>: Send + Sync 
where
    T: Send + Sync,
{
    /// Handle a single event
    async fn on_event(&self, event: &T, sequence: usize);
    
    /// Handle a batch of events
    async fn on_batch(&self, events: &[&T]) {
        for (i, event) in events.iter().enumerate() {
            self.on_event(event, i).await;
        }
    }
    
    /// Called on startup
    async fn on_start(&self) {}
    
    /// Called on shutdown
    async fn on_shutdown(&self) {}
}

/// Handler priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandlerPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

/// Chain of handlers
pub struct HandlerChain {
    handlers: Vec<(HandlerPriority, Box<dyn EventHandler<Event>>)>,
}

impl HandlerChain {
    /// Create new chain
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }
    
    /// Add handler to chain
    pub fn add(&mut self, priority: HandlerPriority, handler: Box<dyn EventHandler<Event>>) {
        self.handlers.push((priority, handler));
        self.handlers.sort_by_key(|(p, _)| *p);
    }
}

impl Default for HandlerChain {
    fn default() -> Self {
        Self::new()
    }
}