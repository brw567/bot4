// Circuit Breaker Implementation - Sophia-Approved Design
// Addresses all 7 critical issues from architecture review

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use arc_swap::ArcSwap;
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Circuit breaker states encoded as u8 for atomic operations
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CircuitState {
    Closed = 0,     // Normal operation
    Open = 1,       // Circuit tripped, rejecting calls
    HalfOpen = 2,   // Testing if service recovered
}

impl From<u8> for CircuitState {
    fn from(value: u8) -> Self {
        match value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

/// Configuration for circuit breaker behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConfig {
    // Window configuration
    pub rolling_window: Duration,
    pub min_calls: u32,
    pub error_rate_threshold: f32,
    
    // Alternative trigger
    pub consecutive_failures_threshold: u32,
    
    // Recovery configuration
    pub open_cooldown: Duration,
    pub half_open_max_concurrent: u32,
    pub half_open_required_successes: u32,
    pub half_open_allowed_failures: u32,
    
    // Global trip conditions
    pub global_trip_conditions: GlobalTripConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalTripConditions {
    pub component_open_ratio: f32,  // e.g., 0.5 = trip if 50% components open
    pub min_components: u32,        // minimum components before ratio applies
}

impl Default for CircuitConfig {
    fn default() -> Self {
        CircuitConfig {
            rolling_window: Duration::from_secs(10),
            min_calls: 10,
            error_rate_threshold: 0.5,
            consecutive_failures_threshold: 5,
            open_cooldown: Duration::from_secs(30),
            half_open_max_concurrent: 3,
            half_open_required_successes: 3,
            half_open_allowed_failures: 1,
            global_trip_conditions: GlobalTripConditions {
                component_open_ratio: 0.5,
                min_components: 3,
            },
        }
    }
}

/// Clock abstraction for testability (addresses issue #5)
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
    fn elapsed(&self, since: Instant) -> Duration {
        self.now().duration_since(since)
    }
}

/// System clock for production
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Test clock for deterministic testing
#[cfg(test)]
pub struct FakeClock {
    current: std::sync::Mutex<Instant>,
}

#[cfg(test)]
impl FakeClock {
    pub fn new() -> Self {
        FakeClock {
            current: std::sync::Mutex::new(Instant::now()),
        }
    }
    
    pub fn advance(&self, duration: Duration) {
        let mut current = self.current.lock().unwrap();
        *current += duration;
    }
}

#[cfg(test)]
impl Clock for FakeClock {
    fn now(&self) -> Instant {
        *self.current.lock().unwrap()
    }
}

/// Errors that can occur in circuit breaker operations
#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("Circuit is open")]
    Open,
    
    #[error("Half-open capacity exhausted")]
    HalfOpenExhausted,
    
    #[error("Global circuit breaker tripped")]
    GlobalOpen,
    
    #[error("Component not found: {0}")]
    ComponentMissing(String),
    
    #[error("Configuration invalid: {0}")]
    ConfigInvalid(String),
}

/// Call outcome for recording results
#[derive(Debug, Copy, Clone)]
pub enum Outcome {
    Success,
    Failure,
}

/// RAII guard for calls (addresses issue #4)
pub struct CallGuard {
    breaker: Arc<ComponentBreaker>,
    component: String,
    start: Instant,
    completed: bool,
}

impl CallGuard {
    fn new(breaker: Arc<ComponentBreaker>, component: String, start: Instant) -> Self {
        CallGuard {
            breaker,
            component,
            start,
            completed: false,
        }
    }
    
    /// Record the outcome of the call
    pub fn record(mut self, outcome: Outcome) {
        self.completed = true;
        self.breaker.record_outcome(outcome);
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        if !self.completed {
            // Auto-record as failure if guard dropped without explicit record
            self.breaker.record_outcome(Outcome::Failure);
        }
    }
}

/// Permission to make a call
pub enum Permit {
    Allowed(CallGuard),
    Rejected(CircuitError),
}

/// Component-level circuit breaker
pub struct ComponentBreaker {
    // State - using atomic for lock-free operation (addresses issue #3)
    state: AtomicU8,
    
    // Metrics - all atomic for lock-free updates
    total_calls: AtomicU64,
    error_calls: AtomicU64,
    consecutive_failures: AtomicU32,
    
    // Half-open state
    half_open_tokens: AtomicU32,
    half_open_successes: AtomicU32,
    half_open_failures: AtomicU32,
    
    // Timing
    last_transition: AtomicU64,  // nanos since epoch
    
    // Shared resources
    clock: Arc<dyn Clock>,
    config: Arc<CircuitConfig>,
}

impl ComponentBreaker {
    pub fn new(clock: Arc<dyn Clock>, config: Arc<CircuitConfig>) -> Self {
        ComponentBreaker {
            state: AtomicU8::new(CircuitState::Closed as u8),
            total_calls: AtomicU64::new(0),
            error_calls: AtomicU64::new(0),
            consecutive_failures: AtomicU32::new(0),
            half_open_tokens: AtomicU32::new(0),
            half_open_successes: AtomicU32::new(0),
            half_open_failures: AtomicU32::new(0),
            last_transition: AtomicU64::new(0),
            clock,
            config,
        }
    }
    
    pub fn current_state(&self) -> CircuitState {
        self.state.load(Ordering::Acquire).into()
    }
    
    pub fn should_trip(&self) -> bool {
        let total = self.total_calls.load(Ordering::Relaxed);
        let errors = self.error_calls.load(Ordering::Relaxed);
        let consecutive = self.consecutive_failures.load(Ordering::Relaxed);
        
        // Check minimum calls threshold
        if total < self.config.min_calls as u64 {
            return false;
        }
        
        // Check error rate
        let error_rate = errors as f32 / total as f32;
        if error_rate > self.config.error_rate_threshold {
            return true;
        }
        
        // Check consecutive failures
        consecutive >= self.config.consecutive_failures_threshold
    }
    
    pub fn try_acquire(&self) -> Result<(), CircuitError> {
        let state = self.current_state();
        
        match state {
            CircuitState::Closed => Ok(()),
            
            CircuitState::Open => {
                // Check if cooldown expired
                let last = self.last_transition.load(Ordering::Relaxed);
                let elapsed = self.clock.now().duration_since(Instant::now() - Duration::from_nanos(last));
                
                if elapsed >= self.config.open_cooldown {
                    // Transition to half-open
                    self.transition_to_half_open();
                    self.try_acquire_half_open()
                } else {
                    Err(CircuitError::Open)
                }
            }
            
            CircuitState::HalfOpen => self.try_acquire_half_open(),
        }
    }
    
    fn try_acquire_half_open(&self) -> Result<(), CircuitError> {
        let current = self.half_open_tokens.fetch_add(1, Ordering::AcqRel);
        
        if current < self.config.half_open_max_concurrent {
            Ok(())
        } else {
            self.half_open_tokens.fetch_sub(1, Ordering::AcqRel);
            Err(CircuitError::HalfOpenExhausted)
        }
    }
    
    fn transition_to_half_open(&self) {
        self.state.store(CircuitState::HalfOpen as u8, Ordering::Release);
        self.half_open_tokens.store(0, Ordering::Release);
        self.half_open_successes.store(0, Ordering::Release);
        self.half_open_failures.store(0, Ordering::Release);
        self.record_transition();
    }
    
    fn transition_to_open(&self) {
        self.state.store(CircuitState::Open as u8, Ordering::Release);
        self.record_transition();
    }
    
    fn transition_to_closed(&self) {
        self.state.store(CircuitState::Closed as u8, Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        self.record_transition();
    }
    
    fn record_transition(&self) {
        let nanos = self.clock.now().elapsed().as_nanos() as u64;
        self.last_transition.store(nanos, Ordering::Release);
    }
    
    pub fn record_outcome(&self, outcome: Outcome) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        
        match outcome {
            Outcome::Success => {
                self.consecutive_failures.store(0, Ordering::Relaxed);
                
                if self.current_state() == CircuitState::HalfOpen {
                    let successes = self.half_open_successes.fetch_add(1, Ordering::AcqRel);
                    
                    if successes + 1 >= self.config.half_open_required_successes {
                        self.transition_to_closed();
                    }
                }
            }
            
            Outcome::Failure => {
                self.error_calls.fetch_add(1, Ordering::Relaxed);
                self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
                
                match self.current_state() {
                    CircuitState::Closed => {
                        if self.should_trip() {
                            self.transition_to_open();
                        }
                    }
                    
                    CircuitState::HalfOpen => {
                        let failures = self.half_open_failures.fetch_add(1, Ordering::AcqRel);
                        
                        if failures + 1 > self.config.half_open_allowed_failures {
                            self.transition_to_open();
                        }
                    }
                    
                    _ => {}
                }
            }
        }
    }
}

/// Global circuit breaker managing all components (addresses issue #1)
pub struct GlobalCircuitBreaker {
    breakers: Arc<DashMap<String, Arc<ComponentBreaker>>>,
    config: ArcSwap<CircuitConfig>,  // Hot-reloadable config (addresses issue #6)
    clock: Arc<dyn Clock>,
    
    // Event callback for telemetry (addresses issue #7)
    on_event: Option<Arc<dyn Fn(CircuitEvent) + Send + Sync>>,
}

/// Events emitted by circuit breaker
#[derive(Debug, Clone)]
pub enum CircuitEvent {
    StateChange { component: String, from: CircuitState, to: CircuitState },
    ConfigReload { old: CircuitConfig, new: CircuitConfig },
    CallRejected { component: String, reason: String },
}

impl GlobalCircuitBreaker {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        GlobalCircuitBreaker {
            breakers: Arc::new(DashMap::new()),
            config: ArcSwap::from_pointee(CircuitConfig::default()),
            clock,
            on_event: None,
        }
    }
    
    pub fn with_event_handler<F>(mut self, handler: F) -> Self 
    where
        F: Fn(CircuitEvent) + Send + Sync + 'static
    {
        self.on_event = Some(Arc::new(handler));
        self
    }
    
    /// Reload configuration with validation
    pub fn reload_config(&self, new_config: CircuitConfig) -> Result<(), CircuitError> {
        // Validate config
        if new_config.error_rate_threshold > 1.0 || new_config.error_rate_threshold < 0.0 {
            return Err(CircuitError::ConfigInvalid(
                "error_rate_threshold must be between 0 and 1".into()
            ));
        }
        
        if new_config.min_calls == 0 {
            return Err(CircuitError::ConfigInvalid(
                "min_calls must be greater than 0".into()
            ));
        }
        
        let old_config = self.config.load().as_ref().clone();
        self.config.store(Arc::new(new_config.clone()));
        
        // Emit event
        if let Some(handler) = &self.on_event {
            handler(CircuitEvent::ConfigReload {
                old: old_config,
                new: new_config,
            });
        }
        
        Ok(())
    }
    
    /// Acquire permission to make a call
    pub fn acquire(&self, component: &str) -> Permit {
        // Check global state first (derived from components)
        if self.is_globally_tripped() {
            return Permit::Rejected(CircuitError::GlobalOpen);
        }
        
        // Get or create component breaker
        let breaker = self.breakers.entry(component.to_string())
            .or_insert_with(|| {
                Arc::new(ComponentBreaker::new(
                    self.clock.clone(),
                    self.config.load().clone(),
                ))
            })
            .clone();
        
        // Try to acquire from component
        match breaker.try_acquire() {
            Ok(()) => {
                let guard = CallGuard::new(
                    breaker,
                    component.to_string(),
                    self.clock.now(),
                );
                Permit::Allowed(guard)
            }
            
            Err(e) => {
                if let Some(handler) = &self.on_event {
                    handler(CircuitEvent::CallRejected {
                        component: component.to_string(),
                        reason: e.to_string(),
                    });
                }
                Permit::Rejected(e)
            }
        }
    }
    
    /// Check if globally tripped based on component states
    fn is_globally_tripped(&self) -> bool {
        let config = self.config.load();
        
        if self.breakers.len() < config.global_trip_conditions.min_components as usize {
            return false;
        }
        
        let open_count = self.breakers.iter()
            .filter(|entry| entry.value().current_state() == CircuitState::Open)
            .count();
        
        let ratio = open_count as f32 / self.breakers.len() as f32;
        ratio >= config.global_trip_conditions.component_open_ratio
    }
    
    /// Get current state of a component
    pub fn component_state(&self, component: &str) -> Option<CircuitState> {
        self.breakers.get(component)
            .map(|entry| entry.value().current_state())
    }
    
    /// Get metrics for monitoring
    pub fn metrics(&self) -> CircuitMetrics {
        let mut metrics = CircuitMetrics::default();
        
        for entry in self.breakers.iter() {
            let state = entry.value().current_state();
            match state {
                CircuitState::Closed => metrics.closed_count += 1,
                CircuitState::Open => metrics.open_count += 1,
                CircuitState::HalfOpen => metrics.half_open_count += 1,
            }
            
            metrics.total_calls += entry.value().total_calls.load(Ordering::Relaxed);
            metrics.error_calls += entry.value().error_calls.load(Ordering::Relaxed);
        }
        
        metrics
    }
}

#[derive(Debug, Default)]
pub struct CircuitMetrics {
    pub closed_count: usize,
    pub open_count: usize,
    pub half_open_count: usize,
    pub total_calls: u64,
    pub error_calls: u64,
}

// Helper function for async operations
pub async fn run_protected<F, T, E>(
    breaker: &GlobalCircuitBreaker,
    component: &str,
    f: F,
) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
    E: From<CircuitError>,
{
    match breaker.acquire(component) {
        Permit::Allowed(guard) => {
            match f.await {
                Ok(result) => {
                    guard.record(Outcome::Success);
                    Ok(result)
                }
                Err(e) => {
                    guard.record(Outcome::Failure);
                    Err(e)
                }
            }
        }
        Permit::Rejected(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_transitions() {
        let clock = Arc::new(FakeClock::new());
        let config = Arc::new(CircuitConfig {
            consecutive_failures_threshold: 3,
            ..Default::default()
        });
        
        let breaker = ComponentBreaker::new(clock.clone(), config);
        
        // Initial state should be closed
        assert_eq!(breaker.current_state(), CircuitState::Closed);
        
        // Record failures to trip the breaker
        for _ in 0..3 {
            breaker.record_outcome(Outcome::Failure);
        }
        
        assert_eq!(breaker.current_state(), CircuitState::Open);
    }
    
    // More tests to be added...
}