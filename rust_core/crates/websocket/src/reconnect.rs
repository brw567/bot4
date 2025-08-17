// Reconnection Strategies for WebSocket
// Implements exponential backoff with jitter for resilient reconnection

use async_trait::async_trait;
use std::time::Duration;
use rand::Rng;

#[async_trait]
pub trait ReconnectStrategy: Send + Sync {
    /// Calculate next reconnection delay
    async fn next_delay(&mut self, attempt: u32) -> Option<Duration>;
    
    /// Reset the strategy (called on successful connection)
    fn reset(&mut self);
    
    /// Check if we should continue trying
    fn should_retry(&self, attempt: u32) -> bool;
}

/// Exponential backoff with jitter
pub struct ExponentialBackoff {
    base_delay: Duration,
    max_delay: Duration,
    max_attempts: u32,
    multiplier: f64,
    jitter_factor: f64,
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            max_attempts: 10,
            multiplier: 2.0,
            jitter_factor: 0.3,
        }
    }
}

impl ExponentialBackoff {
    pub fn new(
        base_delay: Duration,
        max_delay: Duration,
        max_attempts: u32,
    ) -> Self {
        Self {
            base_delay,
            max_delay,
            max_attempts,
            multiplier: 2.0,
            jitter_factor: 0.3,
        }
    }
    
    fn calculate_delay(&self, attempt: u32) -> Duration {
        // Calculate exponential delay
        let exponential_delay = self.base_delay.as_secs_f64() * self.multiplier.powi(attempt as i32);
        
        // Cap at max delay
        let capped_delay = exponential_delay.min(self.max_delay.as_secs_f64());
        
        // Add jitter
        let mut rng = rand::thread_rng();
        let jitter_range = capped_delay * self.jitter_factor;
        let jitter = rng.gen_range(-jitter_range..=jitter_range);
        let final_delay = (capped_delay + jitter).max(0.0);
        
        Duration::from_secs_f64(final_delay)
    }
}

#[async_trait]
impl ReconnectStrategy for ExponentialBackoff {
    async fn next_delay(&mut self, attempt: u32) -> Option<Duration> {
        if self.should_retry(attempt) {
            Some(self.calculate_delay(attempt))
        } else {
            None
        }
    }
    
    fn reset(&mut self) {
        // No state to reset for exponential backoff
    }
    
    fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Fixed interval reconnection
pub struct FixedInterval {
    interval: Duration,
    max_attempts: u32,
}

impl FixedInterval {
    pub fn new(interval: Duration, max_attempts: u32) -> Self {
        Self { interval, max_attempts }
    }
}

#[async_trait]
impl ReconnectStrategy for FixedInterval {
    async fn next_delay(&mut self, attempt: u32) -> Option<Duration> {
        if self.should_retry(attempt) {
            Some(self.interval)
        } else {
            None
        }
    }
    
    fn reset(&mut self) {}
    
    fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Linear backoff strategy
pub struct LinearBackoff {
    base_delay: Duration,
    increment: Duration,
    max_delay: Duration,
    max_attempts: u32,
}

impl LinearBackoff {
    pub fn new(
        base_delay: Duration,
        increment: Duration,
        max_delay: Duration,
        max_attempts: u32,
    ) -> Self {
        Self {
            base_delay,
            increment,
            max_delay,
            max_attempts,
        }
    }
}

#[async_trait]
impl ReconnectStrategy for LinearBackoff {
    async fn next_delay(&mut self, attempt: u32) -> Option<Duration> {
        if self.should_retry(attempt) {
            let delay = self.base_delay + self.increment * attempt;
            Some(delay.min(self.max_delay))
        } else {
            None
        }
    }
    
    fn reset(&mut self) {}
    
    fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_exponential_backoff() {
        let mut strategy = ExponentialBackoff::default();
        
        // First attempt
        let delay1 = strategy.next_delay(0).await.unwrap();
        assert!(delay1 >= Duration::from_millis(700)); // With jitter
        assert!(delay1 <= Duration::from_millis(1300));
        
        // Second attempt
        let delay2 = strategy.next_delay(1).await.unwrap();
        assert!(delay2 >= Duration::from_millis(1400));
        assert!(delay2 <= Duration::from_millis(2600));
        
        // Should stop after max attempts
        assert!(strategy.next_delay(10).await.is_none());
    }
    
    #[tokio::test]
    async fn test_fixed_interval() {
        let mut strategy = FixedInterval::new(Duration::from_secs(5), 3);
        
        assert_eq!(strategy.next_delay(0).await.unwrap(), Duration::from_secs(5));
        assert_eq!(strategy.next_delay(1).await.unwrap(), Duration::from_secs(5));
        assert_eq!(strategy.next_delay(2).await.unwrap(), Duration::from_secs(5));
        assert!(strategy.next_delay(3).await.is_none());
    }
    
    #[tokio::test]
    async fn test_linear_backoff() {
        let mut strategy = LinearBackoff::new(
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(10),
            5,
        );
        
        assert_eq!(strategy.next_delay(0).await.unwrap(), Duration::from_secs(1));
        assert_eq!(strategy.next_delay(1).await.unwrap(), Duration::from_secs(3));
        assert_eq!(strategy.next_delay(2).await.unwrap(), Duration::from_secs(5));
        assert_eq!(strategy.next_delay(3).await.unwrap(), Duration::from_secs(7));
        assert_eq!(strategy.next_delay(4).await.unwrap(), Duration::from_secs(9));
        assert!(strategy.next_delay(5).await.is_none());
    }
}