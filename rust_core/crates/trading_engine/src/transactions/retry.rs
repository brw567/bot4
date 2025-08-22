// Retry Mechanism with Exponential Backoff and Circuit Breaker
// Task 2.5: FULL retry implementation with jitter and circuit breaking
// Team: Jordan (Performance) + Riley (Testing) + Sam (Architecture)
// References:
// - AWS Architecture Blog: "Exponential Backoff And Jitter"
// - Netflix Hystrix Circuit Breaker Pattern
// - Google SRE Book: "Handling Overload"

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use anyhow::{Result, Context};
use tokio::time::sleep;
use uuid::Uuid;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial retry delay (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum retry delay (milliseconds)
    pub max_delay_ms: u64,
    /// Backoff multiplier (e.g., 2.0 for exponential)
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Retry on these error types
    pub retryable_errors: Vec<String>,
    /// Don't retry on these error types (takes precedence)
    pub non_retryable_errors: Vec<String>,
    /// Use circuit breaker
    pub use_circuit_breaker: bool,
    /// Circuit breaker threshold
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker timeout (milliseconds)
    pub circuit_breaker_timeout_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 30000, // 30 seconds max
            backoff_multiplier: 2.0,
            jitter_factor: 0.3,
            retryable_errors: vec![
                "ConnectionError".to_string(),
                "TimeoutError".to_string(),
                "RateLimitError".to_string(),
                "ServiceUnavailable".to_string(),
            ],
            non_retryable_errors: vec![
                "InvalidCredentials".to_string(),
                "InsufficientFunds".to_string(),
                "OrderNotFound".to_string(),
                "PermissionDenied".to_string(),
            ],
            use_circuit_breaker: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 60000, // 1 minute
        }
    }
}

impl RetryPolicy {
    /// Calculate delay for attempt with jitter
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        // Exponential backoff: initial_delay * multiplier^attempt
        let base_delay = self.initial_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32);
        
        // Cap at max delay
        let capped_delay = base_delay.min(self.max_delay_ms as f64);
        
        // Add jitter to prevent thundering herd
        let jitter_range = capped_delay * self.jitter_factor;
        let jitter = rand::thread_rng().gen_range(-jitter_range..=jitter_range);
        
        let final_delay = (capped_delay + jitter).max(0.0) as u64;
        
        Duration::from_millis(final_delay)
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self, error: &str) -> bool {
        // Check non-retryable first (takes precedence)
        if self.non_retryable_errors.iter().any(|e| error.contains(e)) {
            return false;
        }
        
        // Check retryable errors
        if self.retryable_errors.is_empty() {
            // If no specific errors configured, retry all except non-retryable
            true
        } else {
            self.retryable_errors.iter().any(|e| error.contains(e))
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, limited requests for testing
    HalfOpen,
}

/// Circuit breaker for preventing cascading failures
pub struct CircuitBreaker {
    /// Current state
    state: Arc<RwLock<CircuitState>>,
    /// Failure count
    failure_count: AtomicU32,
    /// Success count (for half-open state)
    success_count: AtomicU32,
    /// Last state change timestamp
    last_state_change: AtomicU64,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Metrics
    metrics: Arc<CircuitBreakerMetrics>,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    /// Success threshold to close circuit from half-open
    pub success_threshold: u32,
    /// Timeout before trying half-open (milliseconds)
    pub timeout_ms: u64,
    /// Percentage of requests to allow in half-open (0-100)
    pub half_open_percentage: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 60000, // 1 minute
            half_open_percentage: 10, // 10% of requests
        }
    }
}

#[derive(Debug, Default)]
pub struct CircuitBreakerMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    rejected_requests: AtomicU64,
    state_changes: AtomicU64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_state_change: AtomicU64::new(
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
            ),
            config,
            metrics: Arc::new(CircuitBreakerMetrics::default()),
        }
    }
    
    /// Check if request should be allowed
    pub fn should_allow(&self) -> bool {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        
        let current_state = *self.state.read();
        
        match current_state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout expired
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
                let last_change = self.last_state_change.load(Ordering::Acquire);
                
                if now - last_change > self.config.timeout_ms {
                    // Transition to half-open
                    self.transition_to(CircuitState::HalfOpen);
                    true
                } else {
                    self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow percentage of requests through
                let random = rand::thread_rng().gen_range(0..100);
                if random < self.config.half_open_percentage {
                    true
                } else {
                    self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
        }
    }
    
    /// Record successful request
    pub fn record_success(&self) {
        self.metrics.successful_requests.fetch_add(1, Ordering::Relaxed);
        
        let current_state = *self.state.read();
        
        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                
                if success_count >= self.config.success_threshold {
                    // Enough successes, close circuit
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
            }
        }
    }
    
    /// Record failed request
    pub fn record_failure(&self) {
        self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
        
        let current_state = *self.state.read();
        
        match current_state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
                
                if failure_count >= self.config.failure_threshold {
                    // Too many failures, open circuit
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                // Single failure in half-open, immediately open
                self.transition_to(CircuitState::Open);
            }
            CircuitState::Open => {
                // Already open, no action
            }
        }
    }
    
    /// Transition to new state
    fn transition_to(&self, new_state: CircuitState) {
        let mut state = self.state.write();
        let old_state = *state;
        
        if old_state != new_state {
            *state = new_state;
            
            // Reset counters
            self.failure_count.store(0, Ordering::Release);
            self.success_count.store(0, Ordering::Release);
            
            // Update timestamp
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            self.last_state_change.store(now, Ordering::Release);
            
            // Update metrics
            self.metrics.state_changes.fetch_add(1, Ordering::Relaxed);
            
            log::info!("Circuit breaker state changed: {:?} -> {:?}", old_state, new_state);
        }
    }
    
    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.read()
    }
    
    /// Get metrics
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            total_requests: AtomicU64::new(self.metrics.total_requests.load(Ordering::Relaxed)),
            successful_requests: AtomicU64::new(self.metrics.successful_requests.load(Ordering::Relaxed)),
            failed_requests: AtomicU64::new(self.metrics.failed_requests.load(Ordering::Relaxed)),
            rejected_requests: AtomicU64::new(self.metrics.rejected_requests.load(Ordering::Relaxed)),
            state_changes: AtomicU64::new(self.metrics.state_changes.load(Ordering::Relaxed)),
        }
    }
}

/// Retry manager with circuit breaker integration
pub struct RetryManager {
    /// Retry policies by operation type
    policies: Arc<RwLock<HashMap<String, RetryPolicy>>>,
    /// Circuit breakers by service
    circuit_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    /// Retry queue for scheduled retries
    retry_queue: Arc<RwLock<VecDeque<ScheduledRetry>>>,
    /// Metrics
    metrics: Arc<RetryMetrics>,
}

#[derive(Debug, Clone)]
struct ScheduledRetry {
    id: Uuid,
    operation: String,
    attempt: u32,
    scheduled_at: SystemTime,
}

#[derive(Debug, Default)]
struct RetryMetrics {
    total_retries: AtomicU64,
    successful_retries: AtomicU64,
    failed_retries: AtomicU64,
    circuit_breaker_rejections: AtomicU64,
}

impl RetryManager {
    pub fn new() -> Self {
        let manager = Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            retry_queue: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RetryMetrics::default()),
        };
        
        // Add default policies
        manager.add_policy("default", RetryPolicy::default());
        manager.add_policy("aggressive", RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 50,
            max_delay_ms: 10000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.5,
            ..Default::default()
        });
        manager.add_policy("conservative", RetryPolicy {
            max_attempts: 2,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_multiplier: 3.0,
            jitter_factor: 0.2,
            ..Default::default()
        });
        
        manager
    }
    
    /// Add retry policy
    pub fn add_policy(&self, name: &str, policy: RetryPolicy) {
        self.policies.write().insert(name.to_string(), policy);
    }
    
    /// Get or create circuit breaker for service
    fn get_circuit_breaker(&self, service: &str) -> Arc<CircuitBreaker> {
        let mut breakers = self.circuit_breakers.write();
        
        breakers.entry(service.to_string())
            .or_insert_with(|| Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default())))
            .clone()
    }
    
    /// Execute operation with retry
    pub async fn execute_with_retry<F, T>(
        &self,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Result<T> + Send + Sync,
        T: Send,
    {
        self.execute_with_retry_and_policy(operation, "default", "default").await
    }
    
    /// Execute operation with specific retry policy and service
    pub async fn execute_with_retry_and_policy<F, T>(
        &self,
        operation: F,
        policy_name: &str,
        service_name: &str,
    ) -> Result<T>
    where
        F: Fn() -> Result<T> + Send + Sync,
        T: Send,
    {
        let policy = self.policies.read()
            .get(policy_name)
            .cloned()
            .unwrap_or_default();
        
        let circuit_breaker = if policy.use_circuit_breaker {
            Some(self.get_circuit_breaker(service_name))
        } else {
            None
        };
        
        let mut attempt = 0;
        let mut last_error = None;
        
        while attempt <= policy.max_attempts {
            // Check circuit breaker
            if let Some(ref breaker) = circuit_breaker {
                if !breaker.should_allow() {
                    self.metrics.circuit_breaker_rejections.fetch_add(1, Ordering::Relaxed);
                    return Err(anyhow::anyhow!("Circuit breaker open for service {}", service_name));
                }
            }
            
            // Execute operation
            match operation() {
                Ok(result) => {
                    // Success
                    if attempt > 0 {
                        self.metrics.successful_retries.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    if let Some(ref breaker) = circuit_breaker {
                        breaker.record_success();
                    }
                    
                    return Ok(result);
                }
                Err(error) => {
                    let error_str = error.to_string();
                    
                    // Check if error is retryable
                    if !policy.is_retryable(&error_str) {
                        if let Some(ref breaker) = circuit_breaker {
                            breaker.record_failure();
                        }
                        return Err(error);
                    }
                    
                    // Record failure
                    if let Some(ref breaker) = circuit_breaker {
                        breaker.record_failure();
                    }
                    
                    last_error = Some(error);
                    
                    if attempt < policy.max_attempts {
                        // Calculate delay
                        let delay = policy.calculate_delay(attempt);
                        
                        log::warn!(
                            "Operation failed (attempt {}/{}), retrying after {:?}: {}",
                            attempt + 1, policy.max_attempts, delay, error_str
                        );
                        
                        self.metrics.total_retries.fetch_add(1, Ordering::Relaxed);
                        
                        // Wait before retry
                        sleep(delay).await;
                    }
                    
                    attempt += 1;
                }
            }
        }
        
        // All retries exhausted
        self.metrics.failed_retries.fetch_add(1, Ordering::Relaxed);
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
    }
    
    /// Schedule a retry for later
    pub fn schedule_retry(&self, id: Uuid) -> Result<()> {
        let retry = ScheduledRetry {
            id,
            operation: "transaction".to_string(),
            attempt: 0,
            scheduled_at: SystemTime::now() + Duration::from_secs(60),
        };
        
        self.retry_queue.write().push_back(retry);
        Ok(())
    }
    
    /// Get metrics
    pub fn metrics(&self) -> RetryMetrics {
        RetryMetrics {
            total_retries: AtomicU64::new(self.metrics.total_retries.load(Ordering::Relaxed)),
            successful_retries: AtomicU64::new(self.metrics.successful_retries.load(Ordering::Relaxed)),
            failed_retries: AtomicU64::new(self.metrics.failed_retries.load(Ordering::Relaxed)),
            circuit_breaker_rejections: AtomicU64::new(self.metrics.circuit_breaker_rejections.load(Ordering::Relaxed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    
    #[test]
    fn test_retry_policy_delay_calculation() {
        let policy = RetryPolicy {
            initial_delay_ms: 100,
            max_delay_ms: 10000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0, // No jitter for deterministic test
            ..Default::default()
        };
        
        // Test exponential backoff
        assert_eq!(policy.calculate_delay(0).as_millis(), 100);
        assert_eq!(policy.calculate_delay(1).as_millis(), 200);
        assert_eq!(policy.calculate_delay(2).as_millis(), 400);
        assert_eq!(policy.calculate_delay(3).as_millis(), 800);
        
        // Test max delay cap
        assert_eq!(policy.calculate_delay(10).as_millis(), 10000);
    }
    
    #[test]
    fn test_circuit_breaker_state_transitions() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_ms: 100,
            half_open_percentage: 100,
        };
        
        let breaker = CircuitBreaker::new(config);
        
        // Initial state should be closed
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.should_allow());
        
        // Record failures to open circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        
        breaker.record_failure(); // Third failure
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.should_allow());
        
        // Wait for timeout and transition to half-open
        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.should_allow()); // This transitions to half-open
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        // Record successes to close circuit
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        breaker.record_success(); // Second success
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
    
    #[tokio::test]
    async fn test_retry_manager() {
        let manager = RetryManager::new();
        
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        // Operation that fails twice then succeeds
        let operation = move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(anyhow::anyhow!("ConnectionError: Temporary failure"))
            } else {
                Ok(42)
            }
        };
        
        let result = manager.execute_with_retry(operation).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Called 3 times
        
        // Check metrics
        let metrics = manager.metrics();
        assert_eq!(metrics.successful_retries.load(Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_non_retryable_error() {
        let manager = RetryManager::new();
        
        let operation = || {
            Err(anyhow::anyhow!("InsufficientFunds: Not enough balance"))
        };
        
        let result = manager.execute_with_retry(operation).await;
        assert!(result.is_err());
        
        // Should not retry on non-retryable error
        let metrics = manager.metrics();
        assert_eq!(metrics.total_retries.load(Ordering::Relaxed), 0);
    }
}