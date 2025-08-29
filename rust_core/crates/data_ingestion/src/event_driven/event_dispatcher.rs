// Event Dispatcher - Route events to handlers
// Placeholder for implementation

use serde::{Deserialize, Serialize};

/// TODO: Add docs
pub struct EventDispatcher;
/// TODO: Add docs
pub enum DispatchStrategy {
    RoundRobin,
    Priority,
    LoadBalanced,
}
/// TODO: Add docs
pub struct EventRoute;
/// TODO: Add docs
pub struct DispatchMetrics;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum Priority {
    Critical = 0,
    High = 1,
    Medium = 2,
    Low = 3,
}