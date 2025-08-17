// Circuit Breaker Implementation - Sophia-Approved Design
// Addresses all 7 critical issues from architecture review

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use arc_swap::ArcSwap;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use crossbeam_utils::CachePadded;  // Sophia's recommendation for cache line isolation

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
/// Explicit Send + Sync bounds for thread safety (Sophia Issue #2)
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
    fn elapsed(&self, since: Instant) -> Duration {
        self.now().duration_since(since)
    }
    
    /// Get monotonic nanos since some epoch (for atomic storage)
    fn monotonic_nanos(&self) -> u64 {
        // Using elapsed from a fixed point to get monotonic time
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
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
    
    #[error("Minimum calls not met for statistical confidence")]
    MinCallsNotMet,  // Sophia's minor nit #3
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
    is_half_open: bool,  // Track if this was a half-open call (Sophia Issue #4)
}

impl CallGuard {
    fn new(breaker: Arc<ComponentBreaker>, component: String, start: Instant) -> Self {
        let is_half_open = breaker.current_state() == CircuitState::HalfOpen;
        CallGuard {
            breaker,
            component,
            start,
            completed: false,
            is_half_open,
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
        
        // Release half-open token if this was a half-open call (Sophia Issue #4)
        if self.is_half_open {
            self.breaker.release_half_open_token();
        }
    }
}

/// Permission to make a call
pub enum Permit {
    Allowed(CallGuard),
    Rejected(CircuitError),
}

/// Component-level circuit breaker
/// Sophia Fix #1: Using CachePadded to prevent false sharing on hot atomics
pub struct ComponentBreaker {
    // State - using atomic for lock-free operation (addresses issue #3)
    state: CachePadded<AtomicU8>,
    
    // Hot path metrics - CachePadded to prevent false sharing
    total_calls: CachePadded<AtomicU64>,
    error_calls: CachePadded<AtomicU64>,
    consecutive_failures: CachePadded<AtomicU32>,
    
    // Half-open state - separate cache line
    half_open_tokens: CachePadded<AtomicU32>,
    half_open_successes: CachePadded<AtomicU32>,
    half_open_failures: CachePadded<AtomicU32>,
    
    // Timing - using AtomicU64 for lock-free operation (Sophia Issue #1 fix)
    last_failure_time: CachePadded<AtomicU64>,  // monotonic nanos for lock-free access
    last_transition: CachePadded<AtomicU64>,  // monotonic nanos since start
    
    // Shared resources (cold path)
    clock: Arc<dyn Clock>,
    config: Arc<CircuitConfig>,
}

impl ComponentBreaker {
    pub fn new(clock: Arc<dyn Clock>, config: Arc<CircuitConfig>) -> Self {
        ComponentBreaker {
            state: CachePadded::new(AtomicU8::new(CircuitState::Closed as u8)),
            total_calls: CachePadded::new(AtomicU64::new(0)),
            error_calls: CachePadded::new(AtomicU64::new(0)),
            consecutive_failures: CachePadded::new(AtomicU32::new(0)),
            half_open_tokens: CachePadded::new(AtomicU32::new(0)),
            half_open_successes: CachePadded::new(AtomicU32::new(0)),
            half_open_failures: CachePadded::new(AtomicU32::new(0)),
            last_failure_time: CachePadded::new(AtomicU64::new(0)),
            last_transition: CachePadded::new(AtomicU64::new(0)),
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
        // Sophia Issue #4: Proper token limiting with CAS to prevent races
        loop {
            let current = self.half_open_tokens.load(Ordering::Acquire);
            
            if current >= self.config.half_open_max_concurrent {
                return Err(CircuitError::HalfOpenExhausted);
            }
            
            // Try to acquire token atomically
            match self.half_open_tokens.compare_exchange(
                current,
                current + 1,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(_) => continue,  // Retry on race
            }
        }
    }
    
    fn release_half_open_token(&self) {
        // Safely decrement token count, ensuring we don't underflow
        loop {
            let current = self.half_open_tokens.load(Ordering::Acquire);
            if current == 0 {
                break;  // Already at zero, nothing to release
            }
            
            match self.half_open_tokens.compare_exchange(
                current,
                current - 1,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,  // Retry on race
            }
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
        // Store monotonic nanos for lock-free access
        let nanos = self.clock.now().elapsed().as_nanos() as u64;
        self.last_transition.store(nanos, Ordering::Release);
    }
    
    fn record_failure(&self) {
        // Store failure time as monotonic nanos (Sophia Issue #1 fix)
        let nanos = self.clock.now().elapsed().as_nanos() as u64;
        self.last_failure_time.store(nanos, Ordering::Release);
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
                self.record_failure();  // Record failure time atomically
                
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
    
    // Global state derived from components (Sophia Issue #3)
    global_state: AtomicU8,  // CircuitState encoded as u8
    global_trip_time: AtomicU64,  // Monotonic nanos when globally tripped
    
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
            global_state: AtomicU8::new(CircuitState::Closed as u8),
            global_trip_time: AtomicU64::new(0),
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
        // Update global state based on components (Sophia Issue #3)
        self.update_global_state();
        
        // Check global state first
        let global_state = self.global_state.load(Ordering::Acquire);
        if global_state == CircuitState::Open as u8 {
            // Check if global cooldown expired
            let config = self.config.load();
            let trip_time = self.global_trip_time.load(Ordering::Acquire);
            
            if trip_time > 0 {
                let now_nanos = self.clock.monotonic_nanos();
                let elapsed_nanos = now_nanos.saturating_sub(trip_time);
                let cooldown_nanos = config.open_cooldown.as_nanos() as u64;
                
                if elapsed_nanos < cooldown_nanos {
                    return Permit::Rejected(CircuitError::GlobalOpen);
                } else {
                    // Transition to half-open for recovery attempt
                    self.global_state.store(CircuitState::HalfOpen as u8, Ordering::Release);
                }
            }
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
    
    /// Update global state based on component states (Sophia Issue #3)
    fn update_global_state(&self) {
        let config = self.config.load();
        
        // Skip if not enough components
        if self.breakers.len() < config.global_trip_conditions.min_components as usize {
            return;
        }
        
        // Count component states
        let mut open_count = 0;
        let mut half_open_count = 0;
        let total_count = self.breakers.len();
        
        for entry in self.breakers.iter() {
            match entry.value().current_state() {
                CircuitState::Open => open_count += 1,
                CircuitState::HalfOpen => half_open_count += 1,
                _ => {}
            }
        }
        
        let open_ratio = open_count as f32 / total_count as f32;
        let current_global = self.global_state.load(Ordering::Acquire);
        
        // Determine new global state
        let new_global_state = if open_ratio >= config.global_trip_conditions.component_open_ratio {
            // Trip globally if too many components are open
            if current_global != CircuitState::Open as u8 {
                // Record trip time
                let nanos = self.clock.monotonic_nanos();
                self.global_trip_time.store(nanos, Ordering::Release);
                
                // Emit event
                if let Some(handler) = &self.on_event {
                    handler(CircuitEvent::StateChange {
                        component: "GLOBAL".to_string(),
                        from: CircuitState::from(current_global),
                        to: CircuitState::Open,
                    });
                }
            }
            CircuitState::Open as u8
        } else if open_count == 0 && half_open_count == 0 {
            // All components healthy
            if current_global != CircuitState::Closed as u8 {
                // Emit recovery event
                if let Some(handler) = &self.on_event {
                    handler(CircuitEvent::StateChange {
                        component: "GLOBAL".to_string(),
                        from: CircuitState::from(current_global),
                        to: CircuitState::Closed,
                    });
                }
            }
            CircuitState::Closed as u8
        } else if half_open_count > 0 && open_count == 0 {
            // Some components recovering
            CircuitState::HalfOpen as u8
        } else {
            // Keep current state
            current_global
        };
        
        // Update global state atomically
        self.global_state.store(new_global_state, Ordering::Release);
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