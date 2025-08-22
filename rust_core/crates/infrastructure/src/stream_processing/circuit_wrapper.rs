// Circuit Breaker Wrapper for Stream Processing
// Team Lead: Quinn (Safety Systems)
// Full Implementation - NO SIMPLIFICATIONS

use crate::circuit_breaker::{ComponentBreaker, CircuitConfig, GlobalTripConditions, SystemClock, Outcome};
use std::sync::Arc;
use std::time::Duration;

/// Wrapper for ComponentBreaker to provide simpler API
pub struct StreamCircuitBreaker {
    inner: ComponentBreaker,
    name: String,
}

impl StreamCircuitBreaker {
    /// Create new circuit breaker with proper configuration
    pub fn new(name: &str, max_failures: u32, reset_timeout: Duration, error_threshold: f64) -> Self {
        let clock = Arc::new(SystemClock {});
        let config = Arc::new(CircuitConfig {
            // Window configuration
            rolling_window: reset_timeout,
            min_calls: 10,
            error_rate_threshold: error_threshold as f32,
            
            // Alternative trigger
            consecutive_failures_threshold: max_failures,
            
            // Recovery configuration
            open_cooldown: reset_timeout,
            half_open_max_concurrent: 5,
            half_open_required_successes: 3,
            half_open_allowed_failures: 2,
            
            // Global trip conditions
            global_trip_conditions: GlobalTripConditions {
                component_open_ratio: 0.5,  // Trip if 50% of components are open
                min_components: 3,          // Need at least 3 components before ratio applies
            },
        });
        
        Self {
            inner: ComponentBreaker::new(clock, config),
            name: name.to_string(),
        }
    }
    
    /// Check if circuit is open
    pub fn is_open(&self) -> bool {
        use crate::circuit_breaker::CircuitState;
        matches!(self.inner.current_state(), CircuitState::Open)
    }
    
    /// Record successful operation
    pub fn record_success(&self) {
        self.inner.record_outcome(Outcome::Success);
    }
    
    /// Record failed operation
    pub fn record_error(&self) {
        self.inner.record_outcome(Outcome::Failure);
    }
    
    /// Try to acquire permission
    pub fn try_acquire(&self) -> bool {
        self.inner.try_acquire().is_ok()
    }
}