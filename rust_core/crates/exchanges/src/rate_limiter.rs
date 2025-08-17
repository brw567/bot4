// Exchange API Rate Limiter
// Nexus Requirement: Realistic API rate limit simulation
// Actual exchange limits: Binance 20-50/sec, Kraken 15-30/sec, Coinbase 10-25/sec

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use tokio::sync::Semaphore;
use thiserror::Error;
use crossbeam_utils::CachePadded;

#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded: {current}/{limit} per {window:?}")]
    Exceeded {
        current: u32,
        limit: u32,
        window: Duration,
    },
    
    #[error("Burst limit exceeded: {burst_count} requests in {duration:?}")]
    BurstExceeded {
        burst_count: u32,
        duration: Duration,
    },
    
    #[error("API temporarily banned: {remaining:?} remaining")]
    TemporaryBan {
        remaining: Duration,
    },
}

/// Exchange-specific rate limits (per Nexus reality check)
#[derive(Debug, Clone)]
pub struct ExchangeLimits {
    pub spot_orders_per_sec: u32,
    pub futures_orders_per_sec: u32,
    pub websocket_messages_per_sec: u32,
    pub rest_requests_per_min: u32,
    pub weight_limit_per_min: u32,  // Binance uses weighted limits
    pub burst_multiplier: f32,      // Allow short bursts
}

impl ExchangeLimits {
    pub fn binance() -> Self {
        ExchangeLimits {
            spot_orders_per_sec: 20,
            futures_orders_per_sec: 50,
            websocket_messages_per_sec: 100,
            rest_requests_per_min: 1200,
            weight_limit_per_min: 6000,
            burst_multiplier: 1.5,
        }
    }
    
    pub fn kraken() -> Self {
        ExchangeLimits {
            spot_orders_per_sec: 15,
            futures_orders_per_sec: 30,
            websocket_messages_per_sec: 75,
            rest_requests_per_min: 900,
            weight_limit_per_min: 4500,
            burst_multiplier: 1.3,
        }
    }
    
    pub fn coinbase() -> Self {
        ExchangeLimits {
            spot_orders_per_sec: 10,
            futures_orders_per_sec: 25,
            websocket_messages_per_sec: 50,
            rest_requests_per_min: 600,
            weight_limit_per_min: 3000,
            burst_multiplier: 1.2,
        }
    }
}

/// Token bucket rate limiter with burst support
pub struct RateLimiter {
    // Token bucket state - CachePadded for high contention
    tokens: CachePadded<AtomicU32>,
    last_refill: CachePadded<AtomicU64>,  // nanos since epoch
    
    // Configuration
    max_tokens: u32,
    refill_rate: u32,  // tokens per second
    refill_interval: Duration,
    
    // Burst handling
    burst_tokens: CachePadded<AtomicU32>,
    burst_window_start: CachePadded<AtomicU64>,
    burst_limit: u32,
    
    // Statistics
    total_requests: CachePadded<AtomicU64>,
    rejected_requests: CachePadded<AtomicU64>,
    
    // Semaphore for async coordination
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    pub fn new(requests_per_sec: u32, burst_multiplier: f32) -> Self {
        let max_tokens = requests_per_sec;
        let burst_limit = (requests_per_sec as f32 * burst_multiplier) as u32;
        
        RateLimiter {
            tokens: CachePadded::new(AtomicU32::new(max_tokens)),
            last_refill: CachePadded::new(AtomicU64::new(0)),
            max_tokens,
            refill_rate: requests_per_sec,
            refill_interval: Duration::from_secs(1),
            burst_tokens: CachePadded::new(AtomicU32::new(burst_limit)),
            burst_window_start: CachePadded::new(AtomicU64::new(0)),
            burst_limit,
            total_requests: CachePadded::new(AtomicU64::new(0)),
            rejected_requests: CachePadded::new(AtomicU64::new(0)),
            semaphore: Arc::new(Semaphore::new(max_tokens as usize)),
        }
    }
    
    /// Try to acquire a token (non-blocking)
    pub fn try_acquire(&self) -> Result<(), RateLimitError> {
        self.refill_tokens();
        
        // Try to get a regular token
        let mut current_tokens = self.tokens.load(Ordering::Acquire);
        
        loop {
            if current_tokens > 0 {
                match self.tokens.compare_exchange_weak(
                    current_tokens,
                    current_tokens - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.total_requests.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    Err(actual) => current_tokens = actual,
                }
            } else {
                // No regular tokens, try burst
                return self.try_acquire_burst();
            }
        }
    }
    
    /// Try to acquire from burst bucket
    fn try_acquire_burst(&self) -> Result<(), RateLimitError> {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let window_start = self.burst_window_start.load(Ordering::Acquire);
        
        // Reset burst window if expired (1 second window)
        if now - window_start > 1_000_000_000 {
            self.burst_window_start.store(now, Ordering::Release);
            self.burst_tokens.store(self.burst_limit, Ordering::Release);
        }
        
        // Try to get burst token
        let mut burst = self.burst_tokens.load(Ordering::Acquire);
        
        loop {
            if burst > 0 {
                match self.burst_tokens.compare_exchange_weak(
                    burst,
                    burst - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.total_requests.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    Err(actual) => burst = actual,
                }
            } else {
                self.rejected_requests.fetch_add(1, Ordering::Relaxed);
                return Err(RateLimitError::Exceeded {
                    current: self.refill_rate,
                    limit: self.refill_rate,
                    window: Duration::from_secs(1),
                });
            }
        }
    }
    
    /// Refill tokens based on elapsed time
    fn refill_tokens(&self) {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let last = self.last_refill.load(Ordering::Acquire);
        let elapsed_nanos = now - last;
        
        // Refill every 100ms for smoother rate limiting
        if elapsed_nanos > 100_000_000 {  // 100ms in nanos
            let elapsed_secs = elapsed_nanos as f64 / 1_000_000_000.0;
            let tokens_to_add = (self.refill_rate as f64 * elapsed_secs) as u32;
            
            if tokens_to_add > 0 {
                // Update last refill time
                self.last_refill.store(now, Ordering::Release);
                
                // Add tokens up to max
                let mut current = self.tokens.load(Ordering::Acquire);
                loop {
                    let new_tokens = (current + tokens_to_add).min(self.max_tokens);
                    match self.tokens.compare_exchange_weak(
                        current,
                        new_tokens,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => break,
                        Err(actual) => current = actual,
                    }
                }
            }
        }
    }
    
    /// Async acquire with waiting
    pub async fn acquire(&self) -> Result<(), RateLimitError> {
        // Try non-blocking first
        if self.try_acquire().is_ok() {
            return Ok(());
        }
        
        // Wait for token
        let permit = self.semaphore.acquire().await
            .map_err(|_| RateLimitError::Exceeded {
                current: self.refill_rate,
                limit: self.refill_rate,
                window: Duration::from_secs(1),
            })?;
        
        // Permit acquired, consume token
        self.try_acquire()?;
        permit.forget();  // Don't return permit
        
        Ok(())
    }
    
    /// Get current statistics
    pub fn stats(&self) -> RateLimiterStats {
        RateLimiterStats {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            current_tokens: self.tokens.load(Ordering::Relaxed),
            burst_tokens: self.burst_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimiterStats {
    pub total_requests: u64,
    pub rejected_requests: u64,
    pub current_tokens: u32,
    pub burst_tokens: u32,
}

/// Multi-exchange rate limiter manager
pub struct ExchangeRateLimitManager {
    binance_spot: Arc<RateLimiter>,
    binance_futures: Arc<RateLimiter>,
    kraken_spot: Arc<RateLimiter>,
    kraken_futures: Arc<RateLimiter>,
    coinbase_spot: Arc<RateLimiter>,
    coinbase_futures: Arc<RateLimiter>,
}

impl ExchangeRateLimitManager {
    pub fn new() -> Self {
        let binance = ExchangeLimits::binance();
        let kraken = ExchangeLimits::kraken();
        let coinbase = ExchangeLimits::coinbase();
        
        ExchangeRateLimitManager {
            binance_spot: Arc::new(RateLimiter::new(
                binance.spot_orders_per_sec,
                binance.burst_multiplier,
            )),
            binance_futures: Arc::new(RateLimiter::new(
                binance.futures_orders_per_sec,
                binance.burst_multiplier,
            )),
            kraken_spot: Arc::new(RateLimiter::new(
                kraken.spot_orders_per_sec,
                kraken.burst_multiplier,
            )),
            kraken_futures: Arc::new(RateLimiter::new(
                kraken.futures_orders_per_sec,
                kraken.burst_multiplier,
            )),
            coinbase_spot: Arc::new(RateLimiter::new(
                coinbase.spot_orders_per_sec,
                coinbase.burst_multiplier,
            )),
            coinbase_futures: Arc::new(RateLimiter::new(
                coinbase.futures_orders_per_sec,
                coinbase.burst_multiplier,
            )),
        }
    }
    
    pub fn get_limiter(&self, exchange: &str, market_type: &str) -> Option<Arc<RateLimiter>> {
        match (exchange.to_lowercase().as_str(), market_type.to_lowercase().as_str()) {
            ("binance", "spot") => Some(self.binance_spot.clone()),
            ("binance", "futures") => Some(self.binance_futures.clone()),
            ("kraken", "spot") => Some(self.kraken_spot.clone()),
            ("kraken", "futures") => Some(self.kraken_futures.clone()),
            ("coinbase", "spot") => Some(self.coinbase_spot.clone()),
            ("coinbase", "futures") => Some(self.coinbase_futures.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_rate_limiter_enforcement() {
        let limiter = RateLimiter::new(10, 1.5);  // 10/sec with 1.5x burst
        
        // Should allow 10 immediate requests
        for i in 0..10 {
            assert!(
                limiter.try_acquire().is_ok(),
                "Request {} should succeed",
                i
            );
        }
        
        // 11th should fail
        assert!(limiter.try_acquire().is_err());
        
        // Wait for refill
        sleep(Duration::from_millis(200)).await;
        
        // Should have ~2 tokens refilled
        assert!(limiter.try_acquire().is_ok());
        assert!(limiter.try_acquire().is_ok());
    }
    
    #[tokio::test]
    async fn test_burst_handling() {
        let limiter = RateLimiter::new(10, 1.5);  // 15 burst capacity
        
        // Exhaust regular tokens
        for _ in 0..10 {
            limiter.try_acquire().unwrap();
        }
        
        // Should still allow 5 burst requests
        for i in 0..5 {
            assert!(
                limiter.try_acquire().is_ok(),
                "Burst request {} should succeed",
                i
            );
        }
        
        // 16th should fail
        assert!(limiter.try_acquire().is_err());
    }
    
    #[test]
    fn test_exchange_limits_realistic() {
        let binance = ExchangeLimits::binance();
        assert_eq!(binance.spot_orders_per_sec, 20);
        assert_eq!(binance.futures_orders_per_sec, 50);
        
        let kraken = ExchangeLimits::kraken();
        assert_eq!(kraken.spot_orders_per_sec, 15);
        
        let coinbase = ExchangeLimits::coinbase();
        assert_eq!(coinbase.spot_orders_per_sec, 10);
    }
}