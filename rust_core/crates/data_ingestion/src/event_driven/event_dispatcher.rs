// Event Dispatcher - Route events to handlers
// Placeholder for implementation

use serde::{Deserialize, Serialize};

pub struct EventDispatcher;
pub enum DispatchStrategy {
    RoundRobin,
    Priority,
    LoadBalanced,
}
pub struct EventRoute;
pub struct DispatchMetrics;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Medium = 2,
    Low = 3,
}