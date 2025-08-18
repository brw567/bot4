// Backpressure Policies for Queue Management
// Owner: Riley | Reviewer: Jordan (Performance), Casey (Integration)
// Pre-Production Requirement #7 from Sophia
// Target: Prevent system overload with proper flow control

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, SemaphorePermit};
use async_trait::async_trait;

/// Backpressure policy types
#[derive(Debug, Clone)]
pub enum BackpressurePolicy {
    /// Drop new messages when queue is full
    DropNewest,
    
    /// Drop oldest messages when queue is full
    DropOldest,
    
    /// Block producer until space available
    BlockProducer {
        timeout: Option<Duration>,
    },
    
    /// Reject with error immediately
    RejectWithError,
    
    /// Adaptive rate limiting based on queue depth
    AdaptiveRateLimit {
        min_rate: u64,
        max_rate: u64,
        target_fill_ratio: f64,
    },
    
    /// Circuit breaker pattern
    CircuitBreaker {
        threshold: f64,
        cooldown: Duration,
    },
    
    /// Token bucket rate limiting
    TokenBucket {
        capacity: u64,
        refill_rate: u64,
        refill_interval: Duration,
    },
}

/// Backpressure manager for controlling flow
pub struct BackpressureManager {
    policy: BackpressurePolicy,
    
    // Queue metrics
    queue_capacity: usize,
    current_depth: Arc<AtomicU64>,
    
    // Rate limiting state
    tokens: Arc<AtomicU64>,
    last_refill: Arc<parking_lot::Mutex<Instant>>,
    
    // Circuit breaker state
    circuit_open: Arc<AtomicBool>,
    circuit_opened_at: Arc<parking_lot::Mutex<Option<Instant>>>,
    
    // Semaphore for blocking policies
    semaphore: Arc<Semaphore>,
    
    // Metrics
    dropped_count: Arc<AtomicU64>,
    rejected_count: Arc<AtomicU64>,
    throttled_count: Arc<AtomicU64>,
}

impl BackpressureManager {
    pub fn new(policy: BackpressurePolicy, queue_capacity: usize) -> Self {
        let tokens = match &policy {
            BackpressurePolicy::TokenBucket { capacity, .. } => *capacity,
            _ => queue_capacity as u64,
        };
        
        Self {
            policy,
            queue_capacity,
            current_depth: Arc::new(AtomicU64::new(0)),
            tokens: Arc::new(AtomicU64::new(tokens)),
            last_refill: Arc::new(parking_lot::Mutex::new(Instant::now())),
            circuit_open: Arc::new(AtomicBool::new(false)),
            circuit_opened_at: Arc::new(parking_lot::Mutex::new(None)),
            semaphore: Arc::new(Semaphore::new(queue_capacity)),
            dropped_count: Arc::new(AtomicU64::new(0)),
            rejected_count: Arc::new(AtomicU64::new(0)),
            throttled_count: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Check if we should accept a new message
    pub async fn should_accept(&self) -> BackpressureResult {
        match &self.policy {
            BackpressurePolicy::DropNewest => self.check_drop_newest(),
            BackpressurePolicy::DropOldest => self.check_drop_oldest(),
            BackpressurePolicy::BlockProducer { timeout } => {
                self.check_block_producer(*timeout).await
            }
            BackpressurePolicy::RejectWithError => self.check_reject(),
            BackpressurePolicy::AdaptiveRateLimit { .. } => self.check_adaptive_rate(),
            BackpressurePolicy::CircuitBreaker { threshold, cooldown } => {
                self.check_circuit_breaker(*threshold, *cooldown)
            }
            BackpressurePolicy::TokenBucket { .. } => self.check_token_bucket(),
        }
    }
    
    /// Update queue depth
    pub fn update_depth(&self, new_depth: u64) {
        self.current_depth.store(new_depth, Ordering::Release);
    }
    
    /// Get current fill ratio
    fn fill_ratio(&self) -> f64 {
        let depth = self.current_depth.load(Ordering::Acquire) as f64;
        depth / self.queue_capacity as f64
    }
    
    fn check_drop_newest(&self) -> BackpressureResult {
        if self.fill_ratio() >= 1.0 {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            BackpressureResult::Drop
        } else {
            BackpressureResult::Accept
        }
    }
    
    fn check_drop_oldest(&self) -> BackpressureResult {
        if self.fill_ratio() >= 1.0 {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            BackpressureResult::DropOldest
        } else {
            BackpressureResult::Accept
        }
    }
    
    async fn check_block_producer(&self, timeout: Option<Duration>) -> BackpressureResult {
        match timeout {
            Some(duration) => {
                match tokio::time::timeout(duration, self.semaphore.acquire()).await {
                    Ok(Ok(permit)) => BackpressureResult::AcceptWithPermit(permit),
                    _ => {
                        self.rejected_count.fetch_add(1, Ordering::Relaxed);
                        BackpressureResult::Reject("Timeout waiting for queue space".to_string())
                    }
                }
            }
            None => {
                match self.semaphore.acquire().await {
                    Ok(permit) => BackpressureResult::AcceptWithPermit(permit),
                    Err(_) => BackpressureResult::Reject("Semaphore closed".to_string()),
                }
            }
        }
    }
    
    fn check_reject(&self) -> BackpressureResult {
        if self.fill_ratio() >= 0.9 {  // Reject at 90% capacity
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            BackpressureResult::Reject("Queue near capacity".to_string())
        } else {
            BackpressureResult::Accept
        }
    }
    
    fn check_adaptive_rate(&self) -> BackpressureResult {
        if let BackpressurePolicy::AdaptiveRateLimit { min_rate, max_rate, target_fill_ratio } = &self.policy {
            let fill = self.fill_ratio();
            
            // Calculate adaptive rate based on queue fill
            let rate = if fill < *target_fill_ratio {
                *max_rate
            } else {
                let scale = (1.0 - fill) / (1.0 - target_fill_ratio);
                (*min_rate as f64 + (*max_rate - *min_rate) as f64 * scale) as u64
            };
            
            // Apply rate limit
            if self.try_acquire_token(rate) {
                BackpressureResult::Accept
            } else {
                self.throttled_count.fetch_add(1, Ordering::Relaxed);
                BackpressureResult::Throttle(Duration::from_millis(1000 / rate))
            }
        } else {
            BackpressureResult::Accept
        }
    }
    
    fn check_circuit_breaker(&self, threshold: f64, cooldown: Duration) -> BackpressureResult {
        // Check if circuit is open
        if self.circuit_open.load(Ordering::Acquire) {
            let opened_at = self.circuit_opened_at.lock();
            if let Some(time) = *opened_at {
                if time.elapsed() < cooldown {
                    return BackpressureResult::Reject("Circuit breaker open".to_string());
                } else {
                    // Try to close circuit
                    self.circuit_open.store(false, Ordering::Release);
                }
            }
        }
        
        // Check if we should open circuit
        if self.fill_ratio() > threshold {
            self.circuit_open.store(true, Ordering::Release);
            *self.circuit_opened_at.lock() = Some(Instant::now());
            BackpressureResult::Reject("Circuit breaker triggered".to_string())
        } else {
            BackpressureResult::Accept
        }
    }
    
    fn check_token_bucket(&self) -> BackpressureResult {
        if let BackpressurePolicy::TokenBucket { capacity, refill_rate, refill_interval } = &self.policy {
            // Refill tokens if needed
            let mut last_refill = self.last_refill.lock();
            let now = Instant::now();
            
            if now.duration_since(*last_refill) >= *refill_interval {
                let tokens_to_add = refill_rate;
                let current = self.tokens.load(Ordering::Acquire);
                let new_tokens = (current + tokens_to_add).min(*capacity);
                self.tokens.store(new_tokens, Ordering::Release);
                *last_refill = now;
            }
            
            // Try to acquire token
            loop {
                let current = self.tokens.load(Ordering::Acquire);
                if current == 0 {
                    self.throttled_count.fetch_add(1, Ordering::Relaxed);
                    return BackpressureResult::Throttle(*refill_interval);
                }
                
                if self.tokens.compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::Release,
                    Ordering::Acquire,
                ).is_ok() {
                    return BackpressureResult::Accept;
                }
            }
        } else {
            BackpressureResult::Accept
        }
    }
    
    fn try_acquire_token(&self, rate: u64) -> bool {
        // Simple rate limiting check
        let tokens = self.tokens.load(Ordering::Acquire);
        if tokens > 0 {
            self.tokens.fetch_sub(1, Ordering::Release);
            true
        } else {
            false
        }
    }
    
    /// Get backpressure metrics
    pub fn metrics(&self) -> BackpressureMetrics {
        BackpressureMetrics {
            queue_depth: self.current_depth.load(Ordering::Relaxed),
            queue_capacity: self.queue_capacity,
            fill_ratio: self.fill_ratio(),
            dropped_count: self.dropped_count.load(Ordering::Relaxed),
            rejected_count: self.rejected_count.load(Ordering::Relaxed),
            throttled_count: self.throttled_count.load(Ordering::Relaxed),
            circuit_open: self.circuit_open.load(Ordering::Relaxed),
        }
    }
}

/// Result of backpressure check
pub enum BackpressureResult {
    /// Accept the message
    Accept,
    
    /// Accept with semaphore permit
    AcceptWithPermit(SemaphorePermit<'static>),
    
    /// Drop the new message
    Drop,
    
    /// Drop oldest message to make room
    DropOldest,
    
    /// Reject with error message
    Reject(String),
    
    /// Throttle for specified duration
    Throttle(Duration),
}

#[derive(Debug, Clone)]
pub struct BackpressureMetrics {
    pub queue_depth: u64,
    pub queue_capacity: usize,
    pub fill_ratio: f64,
    pub dropped_count: u64,
    pub rejected_count: u64,
    pub throttled_count: u64,
    pub circuit_open: bool,
}

/// Adaptive backpressure controller
pub struct AdaptiveBackpressure {
    managers: Vec<Arc<BackpressureManager>>,
    
    // Global coordination
    global_pressure: Arc<AtomicU64>,
    
    // Metrics window
    metrics_window: Duration,
    last_adjustment: Arc<parking_lot::Mutex<Instant>>,
}

impl AdaptiveBackpressure {
    pub fn new(policies: Vec<(BackpressurePolicy, usize)>) -> Self {
        let managers = policies
            .into_iter()
            .map(|(policy, capacity)| Arc::new(BackpressureManager::new(policy, capacity)))
            .collect();
        
        Self {
            managers,
            global_pressure: Arc::new(AtomicU64::new(0)),
            metrics_window: Duration::from_secs(10),
            last_adjustment: Arc::new(parking_lot::Mutex::new(Instant::now())),
        }
    }
    
    /// Update global pressure based on all queues
    pub fn update_global_pressure(&self) {
        let total_pressure: f64 = self.managers
            .iter()
            .map(|m| m.fill_ratio())
            .sum();
        
        let avg_pressure = total_pressure / self.managers.len() as f64;
        self.global_pressure.store((avg_pressure * 100.0) as u64, Ordering::Release);
    }
    
    /// Get recommended action based on global state
    pub fn get_global_action(&self) -> GlobalAction {
        let pressure = self.global_pressure.load(Ordering::Acquire);
        
        match pressure {
            0..=30 => GlobalAction::Normal,
            31..=60 => GlobalAction::SlowDown(0.8),
            61..=80 => GlobalAction::SlowDown(0.5),
            81..=95 => GlobalAction::SlowDown(0.2),
            _ => GlobalAction::EmergencyStop,
        }
    }
}

#[derive(Debug)]
pub enum GlobalAction {
    Normal,
    SlowDown(f64),  // Fraction of normal rate
    EmergencyStop,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_drop_newest_policy() {
        let manager = BackpressureManager::new(
            BackpressurePolicy::DropNewest,
            100
        );
        
        manager.update_depth(50);
        assert!(matches!(manager.should_accept().await, BackpressureResult::Accept));
        
        manager.update_depth(100);
        assert!(matches!(manager.should_accept().await, BackpressureResult::Drop));
    }
    
    #[tokio::test]
    async fn test_token_bucket() {
        let manager = BackpressureManager::new(
            BackpressurePolicy::TokenBucket {
                capacity: 10,
                refill_rate: 5,
                refill_interval: Duration::from_millis(100),
            },
            100
        );
        
        // Should accept up to capacity
        for _ in 0..10 {
            assert!(matches!(manager.should_accept().await, BackpressureResult::Accept));
        }
        
        // Should throttle when empty
        assert!(matches!(manager.should_accept().await, BackpressureResult::Throttle(_)));
        
        // Wait for refill
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Should accept again after refill
        for _ in 0..5 {
            assert!(matches!(manager.should_accept().await, BackpressureResult::Accept));
        }
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let manager = BackpressureManager::new(
            BackpressurePolicy::CircuitBreaker {
                threshold: 0.8,
                cooldown: Duration::from_millis(100),
            },
            100
        );
        
        manager.update_depth(70);
        assert!(matches!(manager.should_accept().await, BackpressureResult::Accept));
        
        manager.update_depth(85);
        assert!(matches!(manager.should_accept().await, BackpressureResult::Reject(_)));
        
        // Circuit should be open
        manager.update_depth(50);
        assert!(matches!(manager.should_accept().await, BackpressureResult::Reject(_)));
        
        // Wait for cooldown
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Circuit should close
        assert!(matches!(manager.should_accept().await, BackpressureResult::Accept));
    }
}

// Backpressure characteristics:
// - Multiple policy types for different scenarios
// - Adaptive rate limiting based on queue depth
// - Circuit breaker for overload protection
// - Token bucket for smooth rate limiting
// - Global coordination across queues