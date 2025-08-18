// Bounded Idempotency Manager with LRU Eviction
// Owner: Casey | Reviewer: Sam (Code Quality), Quinn (Risk)
// Pre-Production Requirement #1 from Sophia
// Target: Zero duplicate orders with bounded memory

use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use sha2::{Sha256, Digest};

/// Bounded idempotency manager with LRU eviction and time-wheel cleanup
/// Sophia's requirement: Prevent memory unbounded growth
pub struct BoundedIdempotencyManager {
    // Primary cache with DashMap for lock-free reads
    cache: Arc<DashMap<String, IdempotencyEntry>>,
    
    // LRU eviction queue
    lru_queue: Arc<Mutex<LruCache<String, Instant>>>,
    
    // Time-wheel for TTL cleanup
    time_wheel: Arc<TimeWheel>,
    
    // Configuration
    max_entries: usize,
    ttl: Duration,
    
    // Metrics
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
}

#[derive(Clone, Debug)]
pub struct IdempotencyEntry {
    pub order_id: String,
    pub request_hash: String,
    pub created_at: Instant,
    pub response: OrderResponse,
    pub access_count: AtomicU32,
}

/// Time-wheel for efficient TTL-based cleanup
pub struct TimeWheel {
    buckets: Vec<Mutex<Vec<(String, Instant)>>>,
    bucket_duration: Duration,
    current_bucket: AtomicUsize,
    last_rotation: Mutex<Instant>,
}

impl BoundedIdempotencyManager {
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            lru_queue: Arc::new(Mutex::new(LruCache::new(max_entries))),
            time_wheel: Arc::new(TimeWheel::new(ttl, 60)), // 60 buckets
            max_entries,
            ttl,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Check if request is duplicate and return cached response
    pub fn check_duplicate(&self, client_order_id: &str, request: &OrderRequest) -> Option<OrderResponse> {
        // Generate request hash for parameter validation
        let request_hash = self.hash_request(request);
        
        // Check cache
        if let Some(entry) = self.cache.get(client_order_id) {
            // Validate request hasn't changed
            if entry.request_hash == request_hash {
                // Check TTL
                if entry.created_at.elapsed() < self.ttl {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    entry.access_count.fetch_add(1, Ordering::Relaxed);
                    
                    // Update LRU position
                    self.update_lru(client_order_id);
                    
                    return Some(entry.response.clone());
                } else {
                    // Expired, remove it
                    self.cache.remove(client_order_id);
                }
            } else {
                // Request parameters changed - potential attack or error
                warn!("Request parameters changed for order {}", client_order_id);
                return Some(OrderResponse::error("Request parameters mismatch"));
            }
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Store order response with automatic eviction if needed
    pub fn store(&self, client_order_id: String, request: &OrderRequest, response: OrderResponse) {
        let request_hash = self.hash_request(request);
        
        // Check if we need to evict
        if self.cache.len() >= self.max_entries {
            self.evict_lru();
        }
        
        let entry = IdempotencyEntry {
            order_id: client_order_id.clone(),
            request_hash,
            created_at: Instant::now(),
            response,
            access_count: AtomicU32::new(0),
        };
        
        // Store in cache
        self.cache.insert(client_order_id.clone(), entry.clone());
        
        // Update LRU
        {
            let mut lru = self.lru_queue.lock();
            lru.put(client_order_id.clone(), Instant::now());
        }
        
        // Add to time-wheel for TTL cleanup
        self.time_wheel.add(client_order_id, entry.created_at + self.ttl);
    }
    
    /// Evict least recently used entry
    fn evict_lru(&self) {
        let mut lru = self.lru_queue.lock();
        
        // Find LRU entry
        if let Some((key, _)) = lru.pop_lru() {
            self.cache.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            
            debug!("Evicted LRU entry: {}", key);
        }
    }
    
    /// Update LRU position for accessed entry
    fn update_lru(&self, key: &str) {
        let mut lru = self.lru_queue.lock();
        lru.get(key); // This moves it to MRU position
    }
    
    /// Generate hash of request parameters
    fn hash_request(&self, request: &OrderRequest) -> String {
        let mut hasher = Sha256::new();
        
        // Hash all relevant fields
        hasher.update(request.symbol.as_bytes());
        hasher.update(request.side.to_string().as_bytes());
        hasher.update(request.order_type.to_string().as_bytes());
        hasher.update(request.quantity.to_le_bytes());
        hasher.update(request.price.map(|p| p.to_le_bytes()).unwrap_or([0u8; 8]));
        hasher.update(request.time_in_force.to_string().as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Periodic cleanup of expired entries
    pub async fn cleanup_expired(&self) {
        let now = Instant::now();
        let expired_keys: Vec<String> = self.cache
            .iter()
            .filter(|entry| now.duration_since(entry.created_at) > self.ttl)
            .map(|entry| entry.order_id.clone())
            .collect();
        
        for key in expired_keys {
            self.cache.remove(&key);
            
            // Remove from LRU
            let mut lru = self.lru_queue.lock();
            lru.pop(&key);
        }
    }
    
    /// Get metrics for monitoring
    pub fn metrics(&self) -> IdempotencyMetrics {
        IdempotencyMetrics {
            total_entries: self.cache.len(),
            cache_hits: self.hits.load(Ordering::Relaxed),
            cache_misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: self.calculate_hit_rate(),
        }
    }
    
    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        
        if hits + misses == 0.0 {
            return 0.0;
        }
        
        hits / (hits + misses)
    }
}

impl TimeWheel {
    pub fn new(total_duration: Duration, num_buckets: usize) -> Self {
        let bucket_duration = total_duration / num_buckets as u32;
        let buckets = (0..num_buckets)
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        
        Self {
            buckets,
            bucket_duration,
            current_bucket: AtomicUsize::new(0),
            last_rotation: Mutex::new(Instant::now()),
        }
    }
    
    pub fn add(&self, key: String, expiry: Instant) {
        let bucket_index = self.calculate_bucket(expiry);
        let mut bucket = self.buckets[bucket_index].lock();
        bucket.push((key, expiry));
    }
    
    pub fn tick(&self) -> Vec<String> {
        let mut expired = Vec::new();
        let now = Instant::now();
        
        // Rotate wheel if needed
        let mut last_rotation = self.last_rotation.lock();
        if now.duration_since(*last_rotation) > self.bucket_duration {
            *last_rotation = now;
            
            let current = self.current_bucket.fetch_add(1, Ordering::SeqCst) % self.buckets.len();
            let mut bucket = self.buckets[current].lock();
            
            // Check all entries in current bucket
            bucket.retain(|(key, expiry)| {
                if now >= *expiry {
                    expired.push(key.clone());
                    false
                } else {
                    true
                }
            });
        }
        
        expired
    }
    
    fn calculate_bucket(&self, expiry: Instant) -> usize {
        let now = Instant::now();
        let duration_until = expiry.duration_since(now);
        let bucket_offset = duration_until.as_secs() / self.bucket_duration.as_secs();
        
        (self.current_bucket.load(Ordering::Relaxed) + bucket_offset as usize) % self.buckets.len()
    }
}

// ============================================================================
// SELF-TRADE PREVENTION (STP) POLICIES
// ============================================================================

/// Self-Trade Prevention policies (Sophia's requirement #2)
#[derive(Clone, Debug)]
pub enum StpPolicy {
    /// Cancel the newer order
    CancelNew,
    
    /// Cancel the resting order
    CancelResting,
    
    /// Cancel both orders
    CancelBoth,
    
    /// Decrement and cancel
    DecrementBoth,
}

pub struct StpManager {
    // Track orders by account
    orders_by_account: Arc<DashMap<AccountId, Vec<OrderId>>>,
    
    // STP policy per account pair
    policies: Arc<DashMap<(AccountId, AccountId), StpPolicy>>,
    
    // Default policy
    default_policy: StpPolicy,
}

impl StpManager {
    pub fn new(default_policy: StpPolicy) -> Self {
        Self {
            orders_by_account: Arc::new(DashMap::new()),
            policies: Arc::new(DashMap::new()),
            default_policy,
        }
    }
    
    /// Check if order would self-trade and apply STP policy
    pub fn check_self_trade(&self, new_order: &Order, resting_orders: &[Order]) -> StpResult {
        for resting_order in resting_orders {
            if self.would_self_trade(new_order, resting_order) {
                let policy = self.get_policy(new_order.account_id, resting_order.account_id);
                
                return match policy {
                    StpPolicy::CancelNew => StpResult::CancelNew(new_order.id.clone()),
                    StpPolicy::CancelResting => StpResult::CancelResting(resting_order.id.clone()),
                    StpPolicy::CancelBoth => StpResult::CancelBoth {
                        new: new_order.id.clone(),
                        resting: resting_order.id.clone(),
                    },
                    StpPolicy::DecrementBoth => {
                        let min_qty = new_order.quantity.min(resting_order.quantity);
                        StpResult::DecrementBoth {
                            new_remaining: new_order.quantity - min_qty,
                            resting_remaining: resting_order.quantity - min_qty,
                        }
                    }
                };
            }
        }
        
        StpResult::NoSelfTrade
    }
    
    fn would_self_trade(&self, new_order: &Order, resting_order: &Order) -> bool {
        // Same account or linked accounts
        new_order.account_id == resting_order.account_id ||
        self.are_accounts_linked(new_order.account_id, resting_order.account_id)
    }
    
    fn are_accounts_linked(&self, account1: AccountId, account2: AccountId) -> bool {
        // Check if accounts are linked (same entity)
        // This would be configured based on exchange rules
        self.policies.contains_key(&(account1, account2))
    }
    
    fn get_policy(&self, account1: AccountId, account2: AccountId) -> StpPolicy {
        self.policies
            .get(&(account1, account2))
            .map(|p| p.clone())
            .unwrap_or(self.default_policy.clone())
    }
}

#[derive(Debug)]
pub enum StpResult {
    NoSelfTrade,
    CancelNew(OrderId),
    CancelResting(OrderId),
    CancelBoth { new: OrderId, resting: OrderId },
    DecrementBoth { new_remaining: f64, resting_remaining: f64 },
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bounded_idempotency() {
        let manager = BoundedIdempotencyManager::new(1000, Duration::from_secs(3600));
        
        let request = OrderRequest {
            client_order_id: "TEST001".to_string(),
            symbol: "BTC-USD".to_string(),
            side: OrderSide::Buy,
            quantity: 1.0,
            price: Some(50000.0),
            // ... other fields
        };
        
        let response = OrderResponse::success("ORDER123");
        
        // Store order
        manager.store("TEST001".to_string(), &request, response.clone());
        
        // Check duplicate - should return cached
        let cached = manager.check_duplicate("TEST001", &request);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().order_id, "ORDER123");
        
        // Check metrics
        let metrics = manager.metrics();
        assert_eq!(metrics.cache_hits, 1);
        assert_eq!(metrics.total_entries, 1);
    }
    
    #[test]
    fn test_lru_eviction() {
        let manager = BoundedIdempotencyManager::new(2, Duration::from_secs(3600));
        
        // Fill cache to capacity
        for i in 0..3 {
            let request = OrderRequest {
                client_order_id: format!("TEST{:03}", i),
                // ... other fields
            };
            manager.store(format!("TEST{:03}", i), &request, OrderResponse::success(&format!("ORDER{}", i)));
        }
        
        // First entry should be evicted
        assert!(manager.cache.get("TEST000").is_none());
        assert!(manager.cache.get("TEST001").is_some());
        assert!(manager.cache.get("TEST002").is_some());
        
        let metrics = manager.metrics();
        assert_eq!(metrics.evictions, 1);
    }
    
    #[test]
    fn test_stp_policies() {
        let stp = StpManager::new(StpPolicy::CancelNew);
        
        let new_order = Order {
            id: "NEW001".to_string(),
            account_id: "ACCOUNT1",
            quantity: 10.0,
            // ... other fields
        };
        
        let resting_order = Order {
            id: "REST001".to_string(),
            account_id: "ACCOUNT1",  // Same account
            quantity: 5.0,
            // ... other fields
        };
        
        let result = stp.check_self_trade(&new_order, &[resting_order]);
        
        match result {
            StpResult::CancelNew(id) => assert_eq!(id, "NEW001"),
            _ => panic!("Expected CancelNew"),
        }
    }
}

// Performance characteristics:
// - Idempotency lookup: O(1) with DashMap
// - LRU eviction: O(1) amortized
// - Time-wheel cleanup: O(1) per tick
// - Memory bounded: max_entries enforced
// - STP check: O(n) where n = resting orders