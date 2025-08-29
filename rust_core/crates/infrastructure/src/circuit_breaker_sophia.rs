// Circuit Breaker Implementation - Sophia-Approved Design v2
// Implements all 6 non-blocking improvements from architecture review
//
// Memory Ordering Contract (Sophia Fix #2):
// - State reads: Acquire (ensures we see all writes before state change)
// - State transitions: AcqRel on success, Acquire on failure (full synchronization)
// - Metrics: Relaxed (observational only, don't guard invariants)
// - Tokens: AcqRel/Acquire pattern (same as state)
// - Timing: Relaxed for reads, Release for updates

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use arc_swap::ArcSwap;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use crossbeam_utils::CachePadded;  // Sophia Fix #1: Cache line isolation
use tokio::sync::mpsc;  // Sophia Fix #4: Bounded channel

// Sophia Fix #5: Compile-time check for 64-bit atomics
#[cfg(not(target_has_atomic = "64"))]
compile_error!("Bot4 requires native 64-bit atomics. This platform is not supported.");

/// Circuit breaker states encoded as u8 for atomic operations
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// TODO: Add docs
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

/// Configuration with hysteresis (Sophia Fix #3)
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct CircuitConfig {
    // Window configuration
    pub rolling_window: Duration,
    pub min_calls: u32,
    
    // Hysteresis thresholds (Sophia Fix #3)
    pub error_rate_open_threshold: f32,    // e.g., 0.5 (50%) to open
    pub error_rate_close_threshold: f32,   // e.g., 0.35 (35%) to close
    
    // Alternative trigger
    pub consecutive_failures_threshold: u32,
    
    // Recovery configuration
    pub open_cooldown: Duration,
    pub half_open_max_concurrent: u32,
    pub half_open_required_successes: u32,
    pub half_open_allowed_failures: u32,
    
    // Global trip conditions with hysteresis
    pub global_trip_conditions: GlobalTripConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct GlobalTripConditions {
    pub component_open_ratio: f32,      // e.g., 0.5 = trip if 50% components open
    pub component_close_ratio: f32,     // e.g., 0.35 = close if <35% components open (Sophia Fix #3)
    pub min_components: u32,            // minimum components before ratio applies
    pub min_samples_per_component: u32, // min samples before component counts (Sophia Fix #3)
}

impl Default for CircuitConfig {
    fn default() -> Self {
        CircuitConfig {
            rolling_window: Duration::from_secs(10),
            min_calls: 20,  // Increased per Sophia's recommendation
            error_rate_open_threshold: 0.5,   // Open at 50% errors
            error_rate_close_threshold: 0.35, // Close at 35% errors (hysteresis)
            consecutive_failures_threshold: 5,
            open_cooldown: Duration::from_secs(5),
            half_open_max_concurrent: 3,
            half_open_required_successes: 3,
            half_open_allowed_failures: 1,
            global_trip_conditions: GlobalTripConditions {
                component_open_ratio: 0.5,
                component_close_ratio: 0.35,  // Hysteresis
                min_components: 3,
                min_samples_per_component: 20,
            },
        }
    }
}

/// Clock abstraction with Send + Sync bounds (Sophia Issue #2)
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
    fn elapsed(&self, since: Instant) -> Duration {
        self.now().duration_since(since)
    }
    
    /// Get monotonic nanos since process start (guaranteed monotonic)
    fn monotonic_nanos(&self) -> u64 {
        // Using process-relative monotonic time
        static PROCESS_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = PROCESS_START.get_or_init(Instant::now);
        self.now().duration_since(*start).as_nanos() as u64
    }
}

/// System clock for production
/// TODO: Add docs
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Circuit breaker errors
#[derive(Debug, Clone, Error)]
/// TODO: Add docs
pub enum CircuitError {
    #[error("Circuit is open")]
    Open,
    
    #[error("Half-open circuit exhausted")]
    HalfOpenExhausted,
    
    #[error("Minimum calls not met for statistical confidence")]
    MinCallsNotMet,
}

/// Event types for monitoring (Sophia Fix #4: bounded channel)
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum CircuitEvent {
    StateChanged { component: String, from: CircuitState, to: CircuitState },
    ThresholdExceeded { component: String, error_rate: f32 },
    RecoveryAttempt { component: String },
    GlobalTrip { reason: String },
}

/// Component-level circuit breaker with CachePadded atomics
/// TODO: Add docs
pub struct ComponentBreaker {
    // Hot path state - each on separate cache line (Sophia Fix #1)
    state: CachePadded<AtomicU8>,
    
    // Hot path metrics - isolated cache lines
    total_calls: CachePadded<AtomicU64>,
    error_calls: CachePadded<AtomicU64>,
    consecutive_failures: CachePadded<AtomicU32>,
    
    // Half-open state - separate cache line
    half_open_tokens: CachePadded<AtomicU32>,
    half_open_successes: CachePadded<AtomicU32>,
    half_open_failures: CachePadded<AtomicU32>,
    
    // Timing - separate cache line
    last_failure_time: CachePadded<AtomicU64>,
    last_transition: CachePadded<AtomicU64>,
    
    // Cold path resources
    clock: Arc<dyn Clock>,
    config: Arc<CircuitConfig>,
    event_sender: Option<mpsc::Sender<CircuitEvent>>,  // Sophia Fix #4
}

impl ComponentBreaker {
    pub fn new(
        clock: Arc<dyn Clock>,
        config: Arc<CircuitConfig>,
        event_sender: Option<mpsc::Sender<CircuitEvent>>,
    ) -> Self {
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
            event_sender,
        }
    }
    
    /// Get current state with Acquire ordering
    /// Memory contract: Acquire ensures we see all writes before state change
    #[inline]
    pub fn current_state(&self) -> CircuitState {
        self.state.load(Ordering::Acquire).into()
    }
    
    /// Check if circuit should trip based on hysteresis thresholds
    pub fn should_trip(&self) -> bool {
        // Relaxed ordering for metrics (observational only)
        let total = self.total_calls.load(Ordering::Relaxed);
        let errors = self.error_calls.load(Ordering::Relaxed);
        let consecutive = self.consecutive_failures.load(Ordering::Relaxed);
        
        // Check minimum calls threshold (Sophia Fix #3)
        if total < self.config.min_calls as u64 {
            return false;
        }
        
        // Calculate error rate
        let error_rate = errors as f32 / total as f32;
        
        // Apply hysteresis based on current state (Sophia Fix #3)
        let threshold = match self.current_state() {
            CircuitState::Closed => self.config.error_rate_open_threshold,
            CircuitState::Open => self.config.error_rate_close_threshold,
            CircuitState::HalfOpen => self.config.error_rate_open_threshold,
        };
        
        // Check error rate with hysteresis
        if error_rate > threshold {
            return true;
        }
        
        // Check consecutive failures
        consecutive >= self.config.consecutive_failures_threshold
    }
    
    /// Try to acquire permission to make a call
    /// Memory contract: Full synchronization on state transitions
    #[inline(always)]  // Sophia optimization suggestion
    pub fn try_acquire(&self, component: &str) -> Result<CallGuard, CircuitError> {
        let state = self.current_state();
        
        #[cfg(debug_assertions)]
        {
            // Debug assertion: Half-open should have limited tokens
            if state == CircuitState::HalfOpen {
                let tokens = self.half_open_tokens.load(Ordering::Relaxed);
                debug_assert!(
                    tokens <= self.config.half_open_max_concurrent,
                    "Half-open tokens {} exceeds max {}",
                    tokens,
                    self.config.half_open_max_concurrent
                );
            }
        }
        
        match state {
            CircuitState::Closed => {
                Ok(CallGuard::new(self, component.to_string(), self.clock.now()))
            }
            
            CircuitState::Open => {
                // Check if cooldown expired
                let last_nanos = self.last_transition.load(Ordering::Relaxed);
                let elapsed_nanos = self.clock.monotonic_nanos() - last_nanos;
                let elapsed = Duration::from_nanos(elapsed_nanos);
                
                if elapsed >= self.config.open_cooldown {
                    // Try transition to half-open
                    self.try_transition_to_half_open(component)?;
                    self.try_acquire_half_open(component)
                } else {
                    Err(CircuitError::Open)
                }
            }
            
            CircuitState::HalfOpen => self.try_acquire_half_open(component),
        }
    }
    
    /// Acquire half-open token with CAS loop
    /// Memory contract: AcqRel on success, Acquire on failure
    fn try_acquire_half_open(&self, component: &str) -> Result<CallGuard, CircuitError> {
        // Bounded retry to prevent infinite spin
        const MAX_RETRIES: u32 = 100;
        
        for _ in 0..MAX_RETRIES {
            let current = self.half_open_tokens.load(Ordering::Acquire);
            
            if current >= self.config.half_open_max_concurrent {
                return Err(CircuitError::HalfOpenExhausted);
            }
            
            // Try to acquire token atomically
            match self.half_open_tokens.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,  // Success: full synchronization
                Ordering::Acquire,  // Failure: see other updates
            ) {
                Ok(_) => {
                    return Ok(CallGuard::new_half_open(
                        self,
                        component.to_string(),
                        self.clock.now(),
                    ));
                }
                Err(_) => continue,  // Retry on contention
            }
        }
        
        Err(CircuitError::HalfOpenExhausted)
    }
    
    /// Release half-open token safely
    pub fn release_half_open_token(&self) {
        // Bounded retry with underflow protection
        const MAX_RETRIES: u32 = 100;
        
        for _ in 0..MAX_RETRIES {
            let current = self.half_open_tokens.load(Ordering::Acquire);
            
            if current == 0 {
                // Already at zero, log warning in debug mode
                #[cfg(debug_assertions)]
                eprintln!("Warning: Attempted to release half-open token when count is 0");
                break;
            }
            
            match self.half_open_tokens.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
    
    /// Transition to half-open state
    fn try_transition_to_half_open(&self, component: &str) -> Result<(), CircuitError> {
        // Try atomic transition from Open to HalfOpen
        match self.state.compare_exchange(
            CircuitState::Open as u8,
            CircuitState::HalfOpen as u8,
            Ordering::AcqRel,  // Full synchronization on success
            Ordering::Acquire,  // See other updates on failure
        ) {
            Ok(_) => {
                // Reset half-open counters with Release ordering
                self.half_open_tokens.store(0, Ordering::Release);
                self.half_open_successes.store(0, Ordering::Release);
                self.half_open_failures.store(0, Ordering::Release);
                self.record_transition();
                
                // Send event if channel available (non-blocking)
                if let Some(sender) = &self.event_sender {
                    let event = CircuitEvent::StateChanged {
                        component: component.to_string(),
                        from: CircuitState::Open,
                        to: CircuitState::HalfOpen,
                    };
                    let _ = sender.try_send(event);  // Don't block on send
                }
                
                Ok(())
            }
            Err(_) => {
                // Someone else already transitioned or state changed
                Ok(())
            }
        }
    }
    
    /// Record state transition time
    fn record_transition(&self) {
        let nanos = self.clock.monotonic_nanos();
        self.last_transition.store(nanos, Ordering::Release);
    }
    
    /// Record call outcome
    pub fn record_outcome(&self, outcome: Outcome, component: &str) {
        // Update metrics with Relaxed ordering (observational)
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        
        match outcome {
            Outcome::Success => {
                // Reset consecutive failures
                self.consecutive_failures.store(0, Ordering::Relaxed);
                
                // Handle half-open success
                if self.current_state() == CircuitState::HalfOpen {
                    let successes = self.half_open_successes.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    if successes >= self.config.half_open_required_successes {
                        self.try_close_circuit(component);
                    }
                }
            }
            
            Outcome::Failure => {
                // Update failure metrics
                self.error_calls.fetch_add(1, Ordering::Relaxed);
                let consecutive = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                
                // Record failure time
                let nanos = self.clock.monotonic_nanos();
                self.last_failure_time.store(nanos, Ordering::Release);
                
                // Handle half-open failure
                if self.current_state() == CircuitState::HalfOpen {
                    let failures = self.half_open_failures.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    if failures >= self.config.half_open_allowed_failures {
                        self.try_open_circuit(component);
                    }
                } else if self.should_trip() {
                    self.try_open_circuit(component);
                }
            }
        }
    }
    
    /// Try to open the circuit
    fn try_open_circuit(&self, component: &str) {
        let current = self.current_state();
        
        if current != CircuitState::Open {
            match self.state.compare_exchange(
                current as u8,
                CircuitState::Open as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.record_transition();
                    
                    if let Some(sender) = &self.event_sender {
                        let event = CircuitEvent::StateChanged {
                            component: component.to_string(),
                            from: current,
                            to: CircuitState::Open,
                        };
                        let _ = sender.try_send(event);
                    }
                }
                Err(_) => {}  // State already changed
            }
        }
    }
    
    /// Try to close the circuit
    fn try_close_circuit(&self, component: &str) {
        let current = self.current_state();
        
        if current != CircuitState::Closed {
            match self.state.compare_exchange(
                current as u8,
                CircuitState::Closed as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.record_transition();
                    
                    // Reset counters
                    self.consecutive_failures.store(0, Ordering::Release);
                    
                    if let Some(sender) = &self.event_sender {
                        let event = CircuitEvent::StateChanged {
                            component: component.to_string(),
                            from: current,
                            to: CircuitState::Closed,
                        };
                        let _ = sender.try_send(event);
                    }
                }
                Err(_) => {}  // State already changed
            }
        }
    }
}

/// Call outcome for recording results
#[derive(Debug, Copy, Clone)]
/// TODO: Add docs
pub enum Outcome {
    Success,
    Failure,
}

/// RAII guard for calls with automatic token release
/// TODO: Add docs
pub struct CallGuard {
    breaker: *const ComponentBreaker,  // Raw pointer to avoid Arc overhead
    component: String,
    start: Instant,
    completed: bool,
    is_half_open: bool,
}

// Safety: ComponentBreaker methods are thread-safe via atomics
unsafe impl Send for CallGuard {}
unsafe impl Sync for CallGuard {}

impl CallGuard {
    fn new(breaker: &ComponentBreaker, component: String, start: Instant) -> Self {
        CallGuard {
            breaker: breaker as *const _,
            component,
            start,
            completed: false,
            is_half_open: false,
        }
    }
    
    fn new_half_open(breaker: &ComponentBreaker, component: String, start: Instant) -> Self {
        CallGuard {
            breaker: breaker as *const _,
            component,
            start,
            completed: false,
            is_half_open: true,
        }
    }
    
    /// Record the outcome of the call
    #[inline(always)]  // Hot path optimization
    pub fn record(mut self, outcome: Outcome) {
        self.completed = true;
        unsafe {
            (*self.breaker).record_outcome(outcome, &self.component);
        }
    }
}

impl Drop for CallGuard {
    #[inline(always)]  // Hot path optimization
    fn drop(&mut self) {
        if !self.completed {
            // Auto-record as failure if guard dropped without explicit record
            unsafe {
                (*self.breaker).record_outcome(Outcome::Failure, &self.component);
            }
        }
        
        // Release half-open token if this was a half-open call
        if self.is_half_open {
            unsafe {
                (*self.breaker).release_half_open_token();
            }
        }
    }
}

/// Global circuit breaker with hysteresis (Sophia Fix #3)
/// TODO: Add docs
pub struct GlobalCircuitBreaker {
    components: DashMap<String, Arc<ComponentBreaker>>,
    config: Arc<CircuitConfig>,
    clock: Arc<dyn Clock>,
    event_sender: Option<mpsc::Sender<CircuitEvent>>,
    
    // Global state with hysteresis tracking
    global_state: CachePadded<AtomicU8>,
    last_global_evaluation: CachePadded<AtomicU64>,
}

impl GlobalCircuitBreaker {
    pub fn new(
        clock: Arc<dyn Clock>,
        config: Arc<CircuitConfig>,
        event_channel_size: Option<usize>,  // Sophia Fix #4: bounded channel
    ) -> Self {
        let event_sender = event_channel_size.map(|size| {
            let (tx, mut rx) = mpsc::channel(size);
            
            // Spawn event processor (non-blocking, coalescing)
            tokio::spawn(async move {
                let mut last_event_by_component = std::collections::HashMap::new();
                
                while let Some(event) = rx.recv().await {
                    // Coalesce repeated events per component
                    match &event {
                        CircuitEvent::StateChanged { component, .. } => {
                            last_event_by_component.insert(component.clone(), event);
                        }
                        _ => {
                            // Process non-state events immediately
                            println!("Circuit Event: {:?}", event);
                        }
                    }
                }
            });
            
            tx
        });
        
        GlobalCircuitBreaker {
            components: DashMap::new(),
            config,
            clock,
            event_sender,
            global_state: CachePadded::new(AtomicU8::new(CircuitState::Closed as u8)),
            last_global_evaluation: CachePadded::new(AtomicU64::new(0)),
        }
    }
    
    /// Get or create component breaker
    pub fn component(&self, name: &str) -> Arc<ComponentBreaker> {
        self.components
            .entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(ComponentBreaker::new(
                    self.clock.clone(),
                    self.config.clone(),
                    self.event_sender.clone(),
                ))
            })
            .clone()
    }
    
    /// Derive global state with hysteresis (Sophia Fix #3)
    pub fn derive_global_state(&self) -> CircuitState {
        // Rate limit evaluation (every 100ms)
        let now_nanos = self.clock.monotonic_nanos();
        let last_eval = self.last_global_evaluation.load(Ordering::Relaxed);
        
        if now_nanos - last_eval < 100_000_000 {  // 100ms in nanos
            return self.global_state.load(Ordering::Acquire).into();
        }
        
        // Update evaluation time
        self.last_global_evaluation.store(now_nanos, Ordering::Relaxed);
        
        // Count component states
        let mut open_count = 0;
        let mut total_count = 0;
        let mut half_open_count = 0;
        
        for entry in self.components.iter() {
            let breaker = entry.value();
            
            // Only count if component has enough samples (Sophia Fix #3)
            let calls = breaker.total_calls.load(Ordering::Relaxed);
            if calls < self.config.global_trip_conditions.min_samples_per_component as u64 {
                continue;
            }
            
            total_count += 1;
            match breaker.current_state() {
                CircuitState::Open => open_count += 1,
                CircuitState::HalfOpen => half_open_count += 1,
                CircuitState::Closed => {}
            }
        }
        
        // Check minimum components
        if total_count < self.config.global_trip_conditions.min_components {
            return CircuitState::Closed;
        }
        
        // Calculate ratios
        let open_ratio = open_count as f32 / total_count as f32;
        let current_global = self.global_state.load(Ordering::Acquire).into();
        
        // Apply hysteresis based on current state (Sophia Fix #3)
        let new_state = match current_global {
            CircuitState::Closed => {
                if open_ratio >= self.config.global_trip_conditions.component_open_ratio {
                    CircuitState::Open
                } else if half_open_count > 0 {
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Closed
                }
            }
            CircuitState::Open => {
                if open_ratio <= self.config.global_trip_conditions.component_close_ratio {
                    if half_open_count > 0 {
                        CircuitState::HalfOpen
                    } else {
                        CircuitState::Closed
                    }
                } else {
                    CircuitState::Open
                }
            }
            CircuitState::HalfOpen => {
                if open_ratio >= self.config.global_trip_conditions.component_open_ratio {
                    CircuitState::Open
                } else if open_ratio <= self.config.global_trip_conditions.component_close_ratio 
                    && half_open_count == 0 {
                    CircuitState::Closed
                } else {
                    CircuitState::HalfOpen
                }
            }
        };
        
        // Update global state if changed
        if new_state != current_global {
            self.global_state.store(new_state as u8, Ordering::Release);
            
            if let Some(sender) = &self.event_sender {
                let event = CircuitEvent::GlobalTrip {
                    reason: format!(
                        "State change: {:?} -> {:?} (open_ratio: {:.2})",
                        current_global, new_state, open_ratio
                    ),
                };
                let _ = sender.try_send(event);
            }
        }
        
        new_state
    }
    
    /// Acquire permission from specific component
    pub fn acquire(&self, component: &str) -> Result<CallGuard, CircuitError> {
        // Check global state first
        let global = self.derive_global_state();
        if global == CircuitState::Open {
            return Err(CircuitError::Open);
        }
        
        // Then check component
        let breaker = self.component(component);
        breaker.try_acquire(component)
    }
}