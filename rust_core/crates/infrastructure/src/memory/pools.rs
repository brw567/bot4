// Bot4 Object Pools - Zero Allocation Hot Paths
// Day 2 Sprint - Critical Component
// Owner: Jordan
// Targets: Orders: 10k, Signals: 100k, Ticks: 1M capacity

use super::metrics::{metrics, PoolType};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam::queue::ArrayQueue;
use thread_local::ThreadLocal;
use std::cell::RefCell;

/// Thread-local cache size for each pool
const TLS_CACHE_SIZE: usize = 128;

/// Order object pool - 10,000 capacity
pub struct OrderPool {
    global: Arc<ArrayQueue<Box<Order>>>,
    local: ThreadLocal<RefCell<Vec<Box<Order>>>>,
    allocated: AtomicUsize,
    returned: AtomicUsize,
}

/// Signal object pool - 100,000 capacity  
pub struct SignalPool {
    global: Arc<ArrayQueue<Box<Signal>>>,
    local: ThreadLocal<RefCell<Vec<Box<Signal>>>>,
    allocated: AtomicUsize,
    returned: AtomicUsize,
}

/// Tick object pool - 1,000,000 capacity
pub struct TickPool {
    global: Arc<ArrayQueue<Box<Tick>>>,
    local: ThreadLocal<RefCell<Vec<Box<Tick>>>>,
    allocated: AtomicUsize,
    returned: AtomicUsize,
}

// Domain objects
#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct Signal {
    pub id: u64,
    pub symbol: String,
    pub signal_type: SignalType,
    pub strength: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone)]
pub struct Tick {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub timestamp: u64,
}

// Pool implementations
impl OrderPool {
    const CAPACITY: usize = 10_000;
    
    pub fn new() -> Self {
        let global = Arc::new(ArrayQueue::new(Self::CAPACITY));
        
        // Pre-allocate orders
        let pool = Self {
            global: global.clone(),
            local: ThreadLocal::new(),
            allocated: AtomicUsize::new(0),
            returned: AtomicUsize::new(0),
        };
        
        // Pre-fill pool
        for _ in 0..Self::CAPACITY {
            let order = Box::new(Order {
                id: 0,
                symbol: String::with_capacity(16),
                side: OrderSide::Buy,
                quantity: 0.0,
                price: 0.0,
                timestamp: 0,
            });
            let _ = global.push(order);
        }
        
        pool
    }
    
    pub fn acquire(&self) -> Box<Order> {
        // Try thread-local cache first
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        
        if let Some(order) = local.borrow_mut().pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_tls_hit();
            metrics().record_pool_hit(PoolType::Order);
            return order;
        }
        
        metrics().record_tls_miss();
        
        // Try global pool
        if let Some(order) = self.global.pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_pool_hit(PoolType::Order);
            return order;
        }
        
        // Fallback to allocation (should be rare)
        tracing::warn!("OrderPool exhausted, allocating new");
        self.allocated.fetch_add(1, Ordering::Relaxed);
        metrics().record_pool_miss(PoolType::Order);
        Box::new(Order {
            id: 0,
            symbol: String::with_capacity(16),
            side: OrderSide::Buy,
            quantity: 0.0,
            price: 0.0,
            timestamp: 0,
        })
    }
    
    pub fn release(&self, mut order: Box<Order>) {
        // Reset order
        order.id = 0;
        order.symbol.clear();
        order.quantity = 0.0;
        order.price = 0.0;
        order.timestamp = 0;
        
        self.returned.fetch_add(1, Ordering::Relaxed);
        
        // Try to return to thread-local cache
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        let mut cache = local.borrow_mut();
        
        if cache.len() < TLS_CACHE_SIZE {
            cache.push(order);
            return;
        }
        
        // Return to global pool
        if self.global.push(order).is_err() {
            // Pool is full, drop the order
            tracing::debug!("OrderPool full, dropping order");
        }
    }
}

impl SignalPool {
    const CAPACITY: usize = 100_000;
    
    pub fn new() -> Self {
        let global = Arc::new(ArrayQueue::new(Self::CAPACITY));
        
        let pool = Self {
            global: global.clone(),
            local: ThreadLocal::new(),
            allocated: AtomicUsize::new(0),
            returned: AtomicUsize::new(0),
        };
        
        // Pre-fill pool
        for _ in 0..Self::CAPACITY {
            let signal = Box::new(Signal {
                id: 0,
                symbol: String::with_capacity(16),
                signal_type: SignalType::Hold,
                strength: 0.0,
                timestamp: 0,
            });
            let _ = global.push(signal);
        }
        
        pool
    }
    
    pub fn acquire(&self) -> Box<Signal> {
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        
        if let Some(signal) = local.borrow_mut().pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_tls_hit();
            metrics().record_pool_hit(PoolType::Signal);
            return signal;
        }
        
        metrics().record_tls_miss();
        
        if let Some(signal) = self.global.pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_pool_hit(PoolType::Signal);
            return signal;
        }
        
        tracing::warn!("SignalPool exhausted, allocating new");
        self.allocated.fetch_add(1, Ordering::Relaxed);
        metrics().record_pool_miss(PoolType::Signal);
        Box::new(Signal {
            id: 0,
            symbol: String::with_capacity(16),
            signal_type: SignalType::Hold,
            strength: 0.0,
            timestamp: 0,
        })
    }
    
    pub fn release(&self, mut signal: Box<Signal>) {
        signal.id = 0;
        signal.symbol.clear();
        signal.strength = 0.0;
        signal.timestamp = 0;
        
        self.returned.fetch_add(1, Ordering::Relaxed);
        
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        let mut cache = local.borrow_mut();
        
        if cache.len() < TLS_CACHE_SIZE {
            cache.push(signal);
            return;
        }
        
        if self.global.push(signal).is_err() {
            tracing::debug!("SignalPool full, dropping signal");
        }
    }
}

impl TickPool {
    const CAPACITY: usize = 1_000_000;
    
    pub fn new() -> Self {
        let global = Arc::new(ArrayQueue::new(Self::CAPACITY));
        
        let pool = Self {
            global: global.clone(),
            local: ThreadLocal::new(),
            allocated: AtomicUsize::new(0),
            returned: AtomicUsize::new(0),
        };
        
        // Pre-fill pool (partial due to size)
        for _ in 0..10_000 {
            let tick = Box::new(Tick {
                symbol: String::with_capacity(16),
                bid: 0.0,
                ask: 0.0,
                bid_volume: 0.0,
                ask_volume: 0.0,
                timestamp: 0,
            });
            let _ = global.push(tick);
        }
        
        pool
    }
    
    pub fn acquire(&self) -> Box<Tick> {
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        
        if let Some(tick) = local.borrow_mut().pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_tls_hit();
            metrics().record_pool_hit(PoolType::Tick);
            return tick;
        }
        
        metrics().record_tls_miss();
        
        if let Some(tick) = self.global.pop() {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            metrics().record_pool_hit(PoolType::Tick);
            return tick;
        }
        
        self.allocated.fetch_add(1, Ordering::Relaxed);
        metrics().record_pool_miss(PoolType::Tick);
        Box::new(Tick {
            symbol: String::with_capacity(16),
            bid: 0.0,
            ask: 0.0,
            bid_volume: 0.0,
            ask_volume: 0.0,
            timestamp: 0,
        })
    }
    
    pub fn release(&self, mut tick: Box<Tick>) {
        tick.symbol.clear();
        tick.bid = 0.0;
        tick.ask = 0.0;
        tick.bid_volume = 0.0;
        tick.ask_volume = 0.0;
        tick.timestamp = 0;
        
        self.returned.fetch_add(1, Ordering::Relaxed);
        
        let local = self.local.get_or(|| RefCell::new(Vec::with_capacity(TLS_CACHE_SIZE)));
        let mut cache = local.borrow_mut();
        
        if cache.len() < TLS_CACHE_SIZE {
            cache.push(tick);
            return;
        }
        
        if self.global.push(tick).is_err() {
            tracing::debug!("TickPool full, dropping tick");
        }
    }
}

// Global pool instances
lazy_static::lazy_static! {
    static ref ORDER_POOL: OrderPool = OrderPool::new();
    static ref SIGNAL_POOL: SignalPool = SignalPool::new();
    static ref TICK_POOL: TickPool = TickPool::new();
}

/// Initialize all pools
pub fn initialize_all_pools() {
    lazy_static::initialize(&ORDER_POOL);
    lazy_static::initialize(&SIGNAL_POOL);
    lazy_static::initialize(&TICK_POOL);
    
    tracing::info!("Object pools initialized: Orders=10k, Signals=100k, Ticks=1M");
}

/// Get pool statistics for monitoring
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub order_allocated: usize,
    pub order_returned: usize,
    pub order_pressure: f64,
    pub signal_allocated: usize,
    pub signal_returned: usize,
    pub signal_pressure: f64,
    pub tick_allocated: usize,
    pub tick_returned: usize,
    pub tick_pressure: f64,
}

pub fn get_pool_stats() -> PoolStats {
    let order_alloc = ORDER_POOL.allocated.load(Ordering::Relaxed);
    let order_ret = ORDER_POOL.returned.load(Ordering::Relaxed);
    let order_active = order_alloc.saturating_sub(order_ret);
    
    let signal_alloc = SIGNAL_POOL.allocated.load(Ordering::Relaxed);
    let signal_ret = SIGNAL_POOL.returned.load(Ordering::Relaxed);
    let signal_active = signal_alloc.saturating_sub(signal_ret);
    
    let tick_alloc = TICK_POOL.allocated.load(Ordering::Relaxed);
    let tick_ret = TICK_POOL.returned.load(Ordering::Relaxed);
    let tick_active = tick_alloc.saturating_sub(tick_ret);
    
    PoolStats {
        order_allocated: order_alloc,
        order_returned: order_ret,
        order_pressure: order_active as f64 / OrderPool::CAPACITY as f64,
        signal_allocated: signal_alloc,
        signal_returned: signal_ret,
        signal_pressure: signal_active as f64 / SignalPool::CAPACITY as f64,
        tick_allocated: tick_alloc,
        tick_returned: tick_ret,
        tick_pressure: tick_active as f64 / TickPool::CAPACITY as f64,
    }
}

/// Acquire an order from the pool
pub fn acquire_order() -> Box<Order> {
    ORDER_POOL.acquire()
}

/// Release an order back to the pool
pub fn release_order(order: Box<Order>) {
    ORDER_POOL.release(order);
}

/// Acquire a signal from the pool
pub fn acquire_signal() -> Box<Signal> {
    SIGNAL_POOL.acquire()
}

/// Release a signal back to the pool
pub fn release_signal(signal: Box<Signal>) {
    SIGNAL_POOL.release(signal);
}

/// Acquire a tick from the pool
pub fn acquire_tick() -> Box<Tick> {
    TICK_POOL.acquire()
}

/// Release a tick back to the pool
pub fn release_tick(tick: Box<Tick>) {
    TICK_POOL.release(tick);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_order_pool_performance() {
        initialize_all_pools();
        
        const ITERATIONS: usize = 100_000;
        
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let order = acquire_order();
            release_order(order);
        }
        let elapsed = start.elapsed();
        
        let per_op = elapsed.as_nanos() / (ITERATIONS * 2) as u128;
        println!("Order pool acquire/release: {}ns per operation", per_op);
        
        assert!(per_op < 100, "Pool operations too slow: {}ns", per_op);
    }
    
    #[test]
    fn test_pool_pressure() {
        initialize_all_pools();
        
        // Acquire many orders
        let mut orders = Vec::new();
        for _ in 0..1000 {
            orders.push(acquire_order());
        }
        
        let stats = get_pool_stats();
        assert!(stats.order_pressure > 0.0);
        assert!(stats.order_pressure < 0.2); // Should be <20% with 1k/10k
        
        // Release orders
        for order in orders {
            release_order(order);
        }
        
        let stats = get_pool_stats();
        assert!(stats.order_pressure < 0.01); // Should be near 0
    }
}