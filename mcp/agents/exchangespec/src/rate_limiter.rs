//! Exchange-specific rate limiting

use anyhow::{Result, bail};
use governor::{Quota, RateLimiter as Gov, Jitter};
use governor::clock::DefaultClock;
use governor::state::{direct::DirectStateStore, InMemoryState};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub struct ExchangeRateLimiter {
    exchange: String,
    order_limiter: Arc<Gov<String, DirectStateStore<String>, DefaultClock>>,
    data_limiter: Arc<Gov<String, DirectStateStore<String>, DefaultClock>>,
    weight_limiter: Option<Arc<Gov<String, DirectStateStore<String>, DefaultClock>>>,
    current_weight: Arc<RwLock<u32>>,
    max_weight: u32,
}

impl ExchangeRateLimiter {
    pub fn new(exchange: &str) -> Result<Self> {
        let (order_quota, data_quota, weight_quota, max_weight) = match exchange {
            "binance" => {
                // Binance limits: 1200 requests/minute, 10 orders/second
                let order = Quota::per_second(10.try_into().unwrap());
                let data = Quota::per_minute(1200.try_into().unwrap());
                let weight = Some(Quota::per_minute(6000.try_into().unwrap()));
                (order, data, weight, 6000)
            }
            "kraken" => {
                // Kraken limits: 60 calls/minute for most endpoints
                let order = Quota::per_second(1.try_into().unwrap());
                let data = Quota::per_minute(60.try_into().unwrap());
                (order, data, None, 0)
            }
            "coinbase" => {
                // Coinbase limits: 10 requests/second
                let order = Quota::per_second(3.try_into().unwrap());
                let data = Quota::per_second(10.try_into().unwrap());
                (order, data, None, 0)
            }
            _ => {
                // Default conservative limits
                let order = Quota::per_second(1.try_into().unwrap());
                let data = Quota::per_second(5.try_into().unwrap());
                (order, data, None, 0)
            }
        };
        
        let order_limiter = Arc::new(Gov::direct(order));
        let data_limiter = Arc::new(Gov::direct(data));
        let weight_limiter = weight_quota.map(|q| Arc::new(Gov::direct(q)));
        
        Ok(Self {
            exchange: exchange.to_string(),
            order_limiter,
            data_limiter,
            weight_limiter,
            current_weight: Arc::new(RwLock::new(0)),
            max_weight,
        })
    }
    
    pub async fn check_limit(&self) -> Result<bool> {
        // Check if we can make a request
        match self.order_limiter.check() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    pub async fn wait_for_capacity(&self) -> Result<()> {
        // Wait until we have capacity
        self.order_limiter.until_ready_with_jitter(Jitter::up_to(Duration::from_millis(100))).await;
        Ok(())
    }
    
    pub async fn consume_order_request(&self) -> Result<()> {
        self.order_limiter.check()
            .map_err(|_| anyhow::anyhow!("Rate limit exceeded for orders"))?;
        Ok(())
    }
    
    pub async fn consume_data_request(&self) -> Result<()> {
        self.data_limiter.check()
            .map_err(|_| anyhow::anyhow!("Rate limit exceeded for data requests"))?;
        Ok(())
    }
    
    pub async fn consume_weight(&self, weight: u32) -> Result<()> {
        if let Some(ref limiter) = self.weight_limiter {
            // For weight-based systems like Binance
            let mut current = self.current_weight.write().await;
            
            if *current + weight > self.max_weight {
                bail!("Weight limit would be exceeded: {} + {} > {}", 
                      current, weight, self.max_weight);
            }
            
            *current += weight;
            
            // Schedule weight reset
            let weight_arc = self.current_weight.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let mut w = weight_arc.write().await;
                *w = (*w).saturating_sub(weight);
            });
        }
        
        Ok(())
    }
    
    pub async fn get_status(&self) -> Result<String> {
        let order_available = self.order_limiter.check().is_ok();
        let data_available = self.data_limiter.check().is_ok();
        
        let weight_status = if self.weight_limiter.is_some() {
            let current = *self.current_weight.read().await;
            format!(", weight: {}/{}", current, self.max_weight)
        } else {
            String::new()
        };
        
        Ok(format!(
            "Orders: {}, Data: {}{}",
            if order_available { "available" } else { "limited" },
            if data_available { "available" } else { "limited" },
            weight_status
        ))
    }
    
    pub async fn reset(&self) {
        // Reset weight counter
        *self.current_weight.write().await = 0;
    }
    
    pub fn get_wait_time(&self) -> Duration {
        // Estimate wait time until next available slot
        // This is approximate since governor doesn't expose exact timing
        match self.exchange.as_str() {
            "binance" => Duration::from_millis(100),  // 10 req/sec = 100ms between
            "kraken" => Duration::from_secs(1),       // 1 req/sec
            "coinbase" => Duration::from_millis(333), // 3 req/sec
            _ => Duration::from_secs(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = ExchangeRateLimiter::new("binance").unwrap();
        
        // First request should succeed
        assert!(limiter.check_limit().await.unwrap());
        
        // Consume an order request
        limiter.consume_order_request().await.unwrap();
        
        // Status should show availability
        let status = limiter.get_status().await.unwrap();
        assert!(status.contains("available"));
    }
    
    #[tokio::test]
    async fn test_weight_limiting() {
        let limiter = ExchangeRateLimiter::new("binance").unwrap();
        
        // Consume some weight
        limiter.consume_weight(100).await.unwrap();
        
        let status = limiter.get_status().await.unwrap();
        assert!(status.contains("100/6000"));
        
        // Reset
        limiter.reset().await;
        
        let status = limiter.get_status().await.unwrap();
        assert!(status.contains("0/6000"));
    }
}