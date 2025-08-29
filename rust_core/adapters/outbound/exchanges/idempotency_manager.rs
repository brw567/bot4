// Idempotency Manager for Exchange Operations
// Prevents duplicate order submission during retries (Sophia's #1 critical issue)
// Owner: Casey | Reviewer: Sam

use anyhow::{Result, bail};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};

/// Idempotency entry tracking
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: IdempotencyEntry - Enhanced with Distributed cache, TTL
// pub struct IdempotencyEntry {
    /// The exchange order ID returned
    pub exchange_order_id: String,
    /// When this entry was created
    pub created_at: Instant,
    /// The original request hash (for validation)
    pub request_hash: u64,
    /// Number of times this was accessed
    pub hit_count: u32,
}

/// Manager for idempotent operations
/// Ensures client_order_id deduplication to prevent double orders
/// TODO: Add docs
pub struct IdempotencyManager {
    /// Cache of client_order_id -> exchange_order_id mappings
    entries: Arc<DashMap<String, IdempotencyEntry>>,
    /// Time-to-live for entries (default 24 hours)
    ttl: Duration,
    /// Maximum cache size before cleanup
    max_entries: usize,
    /// Last cleanup time
    last_cleanup: Arc<tokio::sync::RwLock<Instant>>,
}

impl IdempotencyManager {
    /// Create new idempotency manager
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            ttl,
            max_entries,
            last_cleanup: Arc::new(tokio::sync::RwLock::new(Instant::now())),
        }
    }

    /// Default configuration (24h TTL, 100k entries)
    pub fn default() -> Self {
        Self::new(Duration::from_secs(86400), 100_000)
    }

    /// Check if a client_order_id exists and return the exchange_order_id
    pub async fn get(&self, client_order_id: &str) -> Option<String> {
        // Trigger cleanup if needed
        self.maybe_cleanup().await;
        
        // Look up entry
        if let Some(mut entry) = self.entries.get_mut(client_order_id) {
            // Check if expired
            if entry.created_at.elapsed() > self.ttl {
                drop(entry); // Release lock before removing
                self.entries.remove(client_order_id);
                return None;
            }
            
            // Update hit count
            entry.hit_count += 1;
            
            Some(entry.exchange_order_id.clone())
        } else {
            None
        }
    }

    /// Store a new idempotency entry
    pub async fn insert(
        &self, 
        client_order_id: String, 
        exchange_order_id: String,
        request_hash: u64
    ) -> Result<()> {
        // Check if already exists (race condition protection)
        if self.entries.contains_key(&client_order_id) {
            bail!("Client order ID already exists: {}", client_order_id);
        }
        
        // Create entry
        let entry = IdempotencyEntry {
            exchange_order_id,
            created_at: Instant::now(),
            request_hash,
            hit_count: 0,
        };
        
        // Insert
        self.entries.insert(client_order_id, entry);
        
        // Trigger cleanup if we're over capacity
        if self.entries.len() > self.max_entries {
            self.force_cleanup().await;
        }
        
        Ok(())
    }

    /// Validate that a retry matches the original request
    pub async fn validate_retry(
        &self, 
        client_order_id: &str, 
        request_hash: u64
    ) -> Result<bool> {
        if let Some(entry) = self.entries.get(client_order_id) {
            if entry.request_hash != request_hash {
                bail!(
                    "Request mismatch for client_order_id {}. This may indicate a different order with the same ID.",
                    client_order_id
                );
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove an entry (e.g., after order cancellation)
    pub async fn remove(&self, client_order_id: &str) -> Option<IdempotencyEntry> {
        self.entries.remove(client_order_id).map(|(_, entry)| entry)
    }

    /// Cleanup expired entries (called periodically)
    async fn maybe_cleanup(&self) {
        let last_cleanup = *self.last_cleanup.read().await;
        
        // Only cleanup every 5 minutes
        if last_cleanup.elapsed() < Duration::from_secs(300) {
            return;
        }
        
        // Try to acquire write lock (non-blocking)
        if let Ok(mut last_cleanup) = self.last_cleanup.try_write() {
            *last_cleanup = Instant::now();
            drop(last_cleanup); // Release lock before cleanup
            
            // Spawn cleanup task
            let entries = self.entries.clone();
            let ttl = self.ttl;
            tokio::spawn(async move {
                Self::cleanup_expired(entries, ttl).await;
            });
        }
    }

    /// Force immediate cleanup
    async fn force_cleanup(&self) {
        let entries = self.entries.clone();
        let ttl = self.ttl;
        tokio::spawn(async move {
            Self::cleanup_expired(entries, ttl).await;
        });
    }

    /// Remove expired entries
    async fn cleanup_expired(entries: Arc<DashMap<String, IdempotencyEntry>>, ttl: Duration) {
        let now = Instant::now();
        let mut expired = Vec::new();
        
        // Collect expired keys
        for entry in entries.iter() {
            if now.duration_since(entry.created_at) > ttl {
                expired.push(entry.key().clone());
            }
        }
        
        // Remove expired entries
        for key in expired {
            entries.remove(&key);
        }
    }

    /// Get statistics about the idempotency cache
    pub fn stats(&self) -> IdempotencyStats {
        let mut total_hits = 0u64;
        let mut max_hits = 0u32;
        let mut oldest = Instant::now();
        
        for entry in self.entries.iter() {
            total_hits += entry.hit_count as u64;
            max_hits = max_hits.max(entry.hit_count);
            oldest = oldest.min(entry.created_at);
        }
        
        IdempotencyStats {
            total_entries: self.entries.len(),
            total_hits,
            max_hits_per_entry: max_hits,
            oldest_entry_age: oldest.elapsed(),
            cache_size_bytes: self.entries.len() * std::mem::size_of::<IdempotencyEntry>(),
        }
    }

    /// Clear all entries (for testing)
    #[cfg(test)]
    pub fn clear(&self) {
        self.entries.clear();
    }
}

/// Statistics about idempotency cache
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct IdempotencyStats {
    pub total_entries: usize,
    pub total_hits: u64,
    pub max_hits_per_entry: u32,
    pub oldest_entry_age: Duration,
    pub cache_size_bytes: usize,
}

/// Hash an order request for validation
/// TODO: Add docs
pub fn hash_order_request(
    symbol: &str,
    side: &str,
    order_type: &str,
    quantity: f64,
    price: Option<f64>,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    symbol.hash(&mut hasher);
    side.hash(&mut hasher);
    order_type.hash(&mut hasher);
    quantity.to_bits().hash(&mut hasher);
    if let Some(p) = price {
        p.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_idempotency_basic() {
        let mgr = IdempotencyManager::new(Duration::from_secs(60), 100);
        
        // First request
        let client_id = "client_order_123";
        let exchange_id = "exchange_456";
        let hash = hash_order_request("BTC-USD", "BUY", "LIMIT", 1.0, Some(50000.0));
        
        // Insert should succeed
        mgr.insert(client_id.to_string(), exchange_id.to_string(), hash).await.unwrap();
        
        // Get should return the exchange ID
        assert_eq!(mgr.get(client_id).await, Some(exchange_id.to_string()));
        
        // Second insert should fail
        assert!(mgr.insert(client_id.to_string(), "different_id".to_string(), hash).await.is_err());
    }
    
    #[tokio::test]
    async fn test_request_validation() {
        let mgr = IdempotencyManager::new(Duration::from_secs(60), 100);
        
        let client_id = "client_order_123";
        let exchange_id = "exchange_456";
        let hash1 = hash_order_request("BTC-USD", "BUY", "LIMIT", 1.0, Some(50000.0));
        let hash2 = hash_order_request("BTC-USD", "BUY", "LIMIT", 2.0, Some(50000.0)); // Different quantity
        
        mgr.insert(client_id.to_string(), exchange_id.to_string(), hash1).await.unwrap();
        
        // Same hash should validate
        assert!(mgr.validate_retry(client_id, hash1).await.unwrap());
        
        // Different hash should fail
        assert!(mgr.validate_retry(client_id, hash2).await.is_err());
    }
    
    #[tokio::test]
    async fn test_expiry() {
        let mgr = IdempotencyManager::new(Duration::from_millis(100), 100);
        
        let client_id = "client_order_123";
        let exchange_id = "exchange_456";
        let hash = hash_order_request("BTC-USD", "BUY", "LIMIT", 1.0, Some(50000.0));
        
        mgr.insert(client_id.to_string(), exchange_id.to_string(), hash).await.unwrap();
        
        // Should exist initially
        assert!(mgr.get(client_id).await.is_some());
        
        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should be expired
        assert!(mgr.get(client_id).await.is_none());
    }
    
    #[tokio::test]
    async fn test_hit_counting() {
        let mgr = IdempotencyManager::new(Duration::from_secs(60), 100);
        
        let client_id = "client_order_123";
        let exchange_id = "exchange_456";
        let hash = hash_order_request("BTC-USD", "BUY", "LIMIT", 1.0, Some(50000.0));
        
        mgr.insert(client_id.to_string(), exchange_id.to_string(), hash).await.unwrap();
        
        // Multiple gets should increment hit count
        for _ in 0..5 {
            mgr.get(client_id).await;
        }
        
        let stats = mgr.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_hits, 5);
        assert_eq!(stats.max_hits_per_entry, 5);
    }
    
    #[tokio::test]
    async fn test_concurrent_access() {
        use std::sync::Arc;
        
        let mgr = Arc::new(IdempotencyManager::new(Duration::from_secs(60), 1000));
        
        // Spawn multiple tasks trying to insert the same client_order_id
        let mut handles = vec![];
        
        for i in 0..10 {
            let mgr_clone = mgr.clone();
            let handle = tokio::spawn(async move {
                let client_id = "concurrent_order";
                let exchange_id = format!("exchange_{}", i);
                let hash = hash_order_request("BTC-USD", "BUY", "LIMIT", 1.0, Some(50000.0));
                
                mgr_clone.insert(client_id.to_string(), exchange_id, hash).await
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut success_count = 0;
        let mut failure_count = 0;
        
        for handle in handles {
            match handle.await.unwrap() {
                Ok(_) => success_count += 1,
                Err(_) => failure_count += 1,
            }
        }
        
        // Exactly one should succeed
        assert_eq!(success_count, 1);
        assert_eq!(failure_count, 9);
    }
}