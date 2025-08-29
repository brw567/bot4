use domain_types::CircuitBreaker;
// Retry Logic with Exponential Backoff and Jitter
// Team: Casey (Exchange) + Jordan (Performance) + Sam (Code Quality)
// CRITICAL: Handle transient failures gracefully
// References:
// - "Exponential Backoff And Jitter" - AWS Architecture Blog
// - "Circuit Breaker Pattern" - Martin Fowler
// - "Retry Storm Prevention" - Google SRE Book

use std::time::Duration;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use async_trait::async_trait;
use tokio::time::sleep;
use tracing::{error, warn, info, debug};
use rand::Rng;
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Retry errors
#[derive(Debug, Error)]
/// TODO: Add docs
pub enum RetryError<E> {
    #[error("Maximum retries ({max}) exceeded")]
    MaxRetriesExceeded { max: u32, last_error: E },
    
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: Duration },
    
    #[error("Non-retryable error: {0}")]
    NonRetryable(E),
}

/// Determines if an error is retryable
#[async_trait]
pub trait RetryableError: std::error::Error {
    /// Check if this error should trigger a retry
    fn is_retryable(&self) -> bool;
    
    /// Get the suggested wait time before retry (if any)
    fn suggested_wait(&self) -> Option<Duration> {
        None
    }
    
    /// Check if this is a rate limit error
    fn is_rate_limit(&self) -> bool {
        false
    }
}

/// Retry policy configuration
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    
    /// Initial backoff duration
    pub initial_backoff: Duration,
    
    /// Maximum backoff duration
    pub max_backoff: Duration,
    
    /// Exponential base (typically 2.0)
    pub exponential_base: f64,
    
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
    
    /// Total timeout for all retries
    pub total_timeout: Option<Duration>,
    
    /// Enable circuit breaker
    pub circuit_breaker_enabled: bool,
    
    /// Circuit breaker error threshold
    pub circuit_breaker_threshold: u32,
    
    /// Circuit breaker recovery time
    pub circuit_breaker_recovery: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            exponential_base: 2.0,
            jitter_factor: 0.3,
            total_timeout: Some(Duration::from_secs(60)),
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_recovery: Duration::from_secs(60),
        }
    }
}

impl RetryPolicy {
    /// Create a policy for exchange API calls
    pub fn for_exchange_api() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(60),
            exponential_base: 2.0,
            jitter_factor: 0.5, // More jitter to avoid thundering herd
            total_timeout: Some(Duration::from_secs(120)),
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 10,
            circuit_breaker_recovery: Duration::from_secs(300),
        }
    }
    
    /// Create a policy for critical operations (like stop loss)
    pub fn for_critical_operations() -> Self {
        Self {
            max_retries: 10, // More aggressive retries
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_secs(5), // Shorter max to retry faster
            exponential_base: 1.5, // Less aggressive backoff
            jitter_factor: 0.1, // Less jitter for consistency
            total_timeout: Some(Duration::from_secs(30)),
            circuit_breaker_enabled: false, // Never give up on critical ops
            circuit_breaker_threshold: 0,
            circuit_breaker_recovery: Duration::from_secs(0),
        }
    }
    
    /// Calculate backoff duration for attempt number
    fn calculate_backoff(&self, attempt: u32) -> Duration {
        let base_backoff = self.initial_backoff.as_millis() as f64 * 
                          self.exponential_base.powi(attempt as i32);
        
        let capped_backoff = base_backoff.min(self.max_backoff.as_millis() as f64);
        
        // Add jitter
        let mut rng = rand::thread_rng();
        let jitter_range = capped_backoff * self.jitter_factor;
        let jitter = rng.gen_range(-jitter_range..=jitter_range);
        
        let final_backoff = (capped_backoff + jitter).max(0.0) as u64;
        
        Duration::from_millis(final_backoff)
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject requests
    HalfOpen,   // Testing if recovered
}

/// Circuit breaker for preventing cascading failures
// ELIMINATED: use domain_types::CircuitBreaker
// pub struct CircuitBreaker {
    state: Arc<parking_lot::RwLock<CircuitState>>,
    failurecount: Arc<AtomicU32>,
    successcount: Arc<AtomicU32>,
    last_failure_time: Arc<parking_lot::RwLock<Option<DateTime<Utc>>>>,
    threshold: u32,
    recovery_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            state: Arc::new(parking_lot::RwLock::new(CircuitState::Closed)),
            failurecount: Arc::new(AtomicU32::new(0)),
            successcount: Arc::new(AtomicU32::new(0)),
            last_failure_time: Arc::new(parking_lot::RwLock::new(None)),
            threshold,
            recovery_timeout,
        }
    }
    
    /// Check if circuit breaker allows request
    pub fn can_proceed(&self) -> bool {
        let mut state = self.state.write();
        
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    if Utc::now() - last_failure > chrono::Duration::from_std(self.recovery_timeout).unwrap() {
                        *state = CircuitState::HalfOpen;
                        info!("Circuit breaker entering half-open state");
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
    
    /// Record successful operation
    pub fn record_success(&self) {
        let mut state = self.state.write();
        
        match *state {
            CircuitState::HalfOpen => {
                let successcount = self.successcount.fetch_add(1, Ordering::SeqCst) + 1;
                
                // Need multiple successes to close circuit
                if successcount >= 3 {
                    *state = CircuitState::Closed;
                    self.failurecount.store(0, Ordering::SeqCst);
                    self.successcount.store(0, Ordering::SeqCst);
                    info!("Circuit breaker closed after successful recovery");
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failurecount.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }
    
    /// Record failed operation
    pub fn record_failure(&self) {
        let mut state = self.state.write();
        
        let failurecount = self.failurecount.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure_time.write() = Some(Utc::now());
        
        match *state {
            CircuitState::Closed => {
                if failurecount >= self.threshold {
                    *state = CircuitState::Open;
                    error!("Circuit breaker opened after {} failures", failurecount);
                }
            }
            CircuitState::HalfOpen => {
                // Single failure in half-open reopens circuit
                *state = CircuitState::Open;
                self.successcount.store(0, Ordering::SeqCst);
                warn!("Circuit breaker reopened after failure in half-open state");
            }
            _ => {}
        }
    }
}

/// Retry executor with backoff and circuit breaker
/// TODO: Add docs
pub struct RetryExecutor {
    policy: RetryPolicy,
    circuit_breaker: Option<CircuitBreaker>,
    total_attempts: Arc<AtomicU64>,
    total_failures: Arc<AtomicU64>,
}

impl RetryExecutor {
    pub fn new(policy: RetryPolicy) -> Self {
        let circuit_breaker = if policy.circuit_breaker_enabled {
            Some(CircuitBreaker::new(
                policy.circuit_breaker_threshold,
                policy.circuit_breaker_recovery,
            ))
        } else {
            None
        };
        
        Self {
            policy,
            circuit_breaker,
            total_attempts: Arc::new(AtomicU64::new(0)),
            total_failures: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Execute operation with retry logic
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, RetryError<E>>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: RetryableError,
    {
        // Check circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            if !cb.can_proceed() {
                return Err(RetryError::CircuitBreakerOpen);
            }
        }
        
        let start_time = tokio::time::Instant::now();
        let mut last_error = None;
        
        for attempt in 0..=self.policy.max_retries {
            self.total_attempts.fetch_add(1, Ordering::Relaxed);
            
            // Check total timeout
            if let Some(timeout) = self.policy.total_timeout {
                if start_time.elapsed() > timeout {
                    return Err(RetryError::Timeout { duration: timeout });
                }
            }
            
            // Execute operation
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_success();
                    }
                    
                    return Ok(result);
                }
                Err(error) => {
                    // Check if error is retryable
                    if !error.is_retryable() {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure();
                        }
                        return Err(RetryError::NonRetryable(error));
                    }
                    
                    last_error = Some(error);
                    
                    // Don't sleep after last attempt
                    if attempt < self.policy.max_retries {
                        let backoff = if let Some(ref err) = last_error {
                            // Use suggested wait time for rate limits
                            if err.is_rate_limit() {
                                err.suggested_wait().unwrap_or_else(|| 
                                    self.policy.calculate_backoff(attempt))
                            } else {
                                self.policy.calculate_backoff(attempt)
                            }
                        } else {
                            self.policy.calculate_backoff(attempt)
                        };
                        
                        debug!(
                            "Retry attempt {} failed, backing off for {:?}",
                            attempt + 1, backoff
                        );
                        
                        sleep(backoff).await;
                    }
                }
            }
        }
        
        // All retries exhausted
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        
        if let Some(ref cb) = self.circuit_breaker {
            cb.record_failure();
        }
        
        Err(RetryError::MaxRetriesExceeded {
            max: self.policy.max_retries,
            last_error: last_error.unwrap(),
        })
    }
    
    /// Get retry statistics
    pub fn get_stats(&self) -> RetryStats {
        RetryStats {
            total_attempts: self.total_attempts.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            circuitstate: self.circuit_breaker.as_ref().map(|cb| {
                *cb.state.read()
            }),
        }
    }
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RetryStats {
    pub total_attempts: u64,
    pub total_failures: u64,
    pub circuitstate: Option<CircuitState>,
}

/// Retry guard for automatic retry on drop (for cleanup operations)
/// TODO: Add docs
pub struct RetryGuard<F, Fut, E> 
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), E>> + Send,
    E: RetryableError + Send,
{
    cleanup: Option<F>,
    executor: Arc<RetryExecutor>,
    _phantom: std::marker::PhantomData<(Fut, E)>,
}

impl<F, Fut, E> RetryGuard<F, Fut, E>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), E>> + Send,
    E: RetryableError + Send,
{
    pub fn new(cleanup: F, executor: Arc<RetryExecutor>) -> Self {
        Self {
            cleanup: Some(cleanup),
            executor,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn disarm(mut self) {
        self.cleanup = None;
    }
}

impl<F, Fut, E> Drop for RetryGuard<F, Fut, E>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), E>> + Send,
    E: RetryableError + Send,
{
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            // Schedule cleanup with retries
            let executor = self.executor.clone();
            tokio::spawn(async move {
                if let Err(e) = executor.execute(&cleanup).await {
                    error!("Cleanup operation failed after retries: {:?}", e);
                }
            });
        }
    }
}

// ============================================================================
// TESTS - Casey & Jordan validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    
    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("Transient error")]
        Transient,
        
        #[error("Permanent error")]
        Permanent,
        
        #[error("Rate limit")]
        RateLimit,
    }
    
    impl RetryableError for TestError {
        fn is_retryable(&self) -> bool {
            !matches!(_self, TestError::Permanent)
        }
        
        fn is_rate_limit(&self) -> bool {
            matches!(_self, TestError::RateLimit)
        }
        
        fn suggested_wait(&self) -> Option<Duration> {
            match self {
                TestError::RateLimit => Some(Duration::from_secs(1)),
                _ => None,
            }
        }
    }
    
    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let policy = RetryPolicy {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            ..Default::default()
        };
        
        let executor = RetryExecutor::new(policy);
        
        let result = executor.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let count = counter.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(TestError::Transient)
                } else {
                    Ok("success")
                }
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Failed twice, succeeded on third
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let policy = RetryPolicy {
            max_retries: 1,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 2,
            circuit_breaker_recovery: Duration::from_millis(100),
            ..Default::default()
        };
        
        let executor = RetryExecutor::new(policy);
        
        // Fail twice to open circuit
        for _ in 0..2 {
            let _ = executor.execute(|| async { 
                Err::<(), TestError>(TestError::Transient) 
            }).await;
        }
        
        // Next attempt should fail immediately
        let result = executor.execute(|| async { 
            Ok::<_, TestError>("should not execute") 
        }).await;
        
        assert!(matches!(_result, Err(RetryError::CircuitBreakerOpen)));
        
        // Wait for recovery
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should work again
        let result = executor.execute(|| async { 
            Ok::<_, TestError>("success") 
        }).await;
        
        assert!(result.is_ok());
    }
}