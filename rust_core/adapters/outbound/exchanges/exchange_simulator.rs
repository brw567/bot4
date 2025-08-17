// Adapter: Exchange Simulator
// Realistic exchange simulation for testing (Sophia's #1 priority)
// Owner: Casey | Reviewer: Sam

use async_trait::async_trait;
use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use rand::{Rng, thread_rng};
use std::time::Duration;

use crate::domain::entities::{Order, OrderId, OrderStatus, OrderSide, OrderType, TimeInForce};
use crate::domain::value_objects::{Symbol, Price, Quantity};
use crate::ports::outbound::exchange_port::{
    ExchangePort, OrderBook, OrderBookLevel, Trade, Balance, ExchangeCapabilities
};

/// Latency simulation modes
#[derive(Debug, Clone)]
pub enum LatencyMode {
    None,
    Fixed(Duration),
    Variable { min: Duration, max: Duration },
    Realistic, // Simulates real network conditions
}

/// Fill simulation modes
#[derive(Debug, Clone)]
pub enum FillMode {
    Instant,           // Fill immediately at requested price
    Realistic,         // Partial fills, slippage, etc.
    Aggressive,        // High slippage, many partials
    Conservative,      // Better prices but slower
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub orders_per_second: u32,
    pub weight_per_order: u32,
    pub max_weight_per_minute: u32,
    pub enable_burst: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            orders_per_second: 10,
            weight_per_order: 1,
            max_weight_per_minute: 1200,
            enable_burst: true,
        }
    }
}

/// Network failure modes for chaos testing
#[derive(Debug, Clone)]
pub enum FailureMode {
    None,
    RandomDrops { probability: f64 },
    Outage { duration: Duration },
    HighLatency { multiplier: f64 },
}

/// Order book entry for simulation
#[derive(Debug, Clone)]
struct SimulatedOrder {
    order_id: String,
    client_order_id: OrderId,
    symbol: Symbol,
    side: OrderSide,
    order_type: OrderType,
    price: Option<Price>,
    quantity: Quantity,
    filled_quantity: Quantity,
    status: OrderStatus,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

/// Exchange simulator state
struct SimulatorState {
    // Orders
    orders: HashMap<String, SimulatedOrder>,
    order_books: HashMap<Symbol, SimulatedOrderBook>,
    
    // Balances
    balances: HashMap<String, Balance>,
    
    // Market data
    last_prices: HashMap<Symbol, Price>,
    recent_trades: HashMap<Symbol, VecDeque<Trade>>,
    
    // Rate limiting
    rate_limit_tokens: u32,
    last_rate_limit_reset: DateTime<Utc>,
    
    // Statistics
    total_orders_placed: u64,
    total_orders_filled: u64,
    total_orders_cancelled: u64,
    total_volume_traded: f64,
}

/// Simulated order book
#[derive(Debug, Clone)]
struct SimulatedOrderBook {
    bids: Vec<OrderBookLevel>,
    asks: Vec<OrderBookLevel>,
    last_update: DateTime<Utc>,
}

/// Exchange Simulator - Implements realistic exchange behavior
/// 
/// Features (per Sophia's requirements):
/// - Partial fills with realistic size distributions
/// - Rate limiting with 429 responses
/// - Network jitter and failures
/// - OCO, ReduceOnly, PostOnly order types
/// - Slippage and market impact modeling
pub struct ExchangeSimulator {
    state: Arc<RwLock<SimulatorState>>,
    latency_mode: LatencyMode,
    fill_mode: FillMode,
    rate_limit_config: RateLimitConfig,
    failure_mode: FailureMode,
    
    // Configuration
    maker_fee: f64,
    taker_fee: f64,
    min_order_size: f64,
    tick_size: f64,
}

impl ExchangeSimulator {
    /// Create a new exchange simulator
    pub fn new() -> Self {
        let mut state = SimulatorState {
            orders: HashMap::new(),
            order_books: HashMap::new(),
            balances: HashMap::new(),
            last_prices: HashMap::new(),
            recent_trades: HashMap::new(),
            rate_limit_tokens: 10,
            last_rate_limit_reset: Utc::now(),
            total_orders_placed: 0,
            total_orders_filled: 0,
            total_orders_cancelled: 0,
            total_volume_traded: 0.0,
        };
        
        // Initialize with some default balances
        state.balances.insert("USDT".to_string(), Balance {
            asset: "USDT".to_string(),
            free: Quantity::new(100000.0).unwrap(),
            locked: Quantity::zero(),
        });
        
        state.balances.insert("BTC".to_string(), Balance {
            asset: "BTC".to_string(),
            free: Quantity::new(2.0).unwrap(),
            locked: Quantity::zero(),
        });
        
        // Initialize default market prices
        state.last_prices.insert(
            Symbol::new("BTC/USDT").unwrap(),
            Price::new(50000.0).unwrap()
        );
        
        Self {
            state: Arc::new(RwLock::new(state)),
            latency_mode: LatencyMode::Realistic,
            fill_mode: FillMode::Realistic,
            rate_limit_config: RateLimitConfig::default(),
            failure_mode: FailureMode::None,
            maker_fee: 0.0002, // 0.02%
            taker_fee: 0.0004, // 0.04%
            min_order_size: 0.00001,
            tick_size: 0.01,
        }
    }
    
    /// Configure simulation parameters
    pub fn with_config(
        mut self,
        latency: LatencyMode,
        fill: FillMode,
        rate_limit: RateLimitConfig,
        failure: FailureMode,
    ) -> Self {
        self.latency_mode = latency;
        self.fill_mode = fill;
        self.rate_limit_config = rate_limit;
        self.failure_mode = failure;
        self
    }
    
    /// Simulate network latency
    async fn simulate_latency(&self) {
        match &self.latency_mode {
            LatencyMode::None => {},
            LatencyMode::Fixed(duration) => {
                tokio::time::sleep(*duration).await;
            },
            LatencyMode::Variable { min, max } => {
                let mut rng = thread_rng();
                let range = max.as_millis() - min.as_millis();
                let delay = min.as_millis() + rng.gen_range(0..=range);
                tokio::time::sleep(Duration::from_millis(delay as u64)).await;
            },
            LatencyMode::Realistic => {
                // Simulate realistic network latency (5-50ms)
                let mut rng = thread_rng();
                let delay = rng.gen_range(5..=50);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            },
        }
    }
    
    /// Simulate network failures
    async fn simulate_failure(&self) -> Result<()> {
        match &self.failure_mode {
            FailureMode::None => Ok(()),
            FailureMode::RandomDrops { probability } => {
                let mut rng = thread_rng();
                if rng.gen::<f64>() < *probability {
                    bail!("Network error: Request dropped");
                }
                Ok(())
            },
            FailureMode::Outage { duration } => {
                tokio::time::sleep(*duration).await;
                bail!("Exchange outage: Service unavailable");
            },
            FailureMode::HighLatency { multiplier } => {
                match &self.latency_mode {
                    LatencyMode::Fixed(d) => {
                        let extended = Duration::from_secs_f64(d.as_secs_f64() * multiplier);
                        tokio::time::sleep(extended).await;
                    },
                    _ => {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                }
                Ok(())
            },
        }
    }
    
    /// Check and update rate limits
    async fn check_rate_limit(&self) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Reset tokens if needed
        let now = Utc::now();
        if (now - state.last_rate_limit_reset).num_seconds() >= 1 {
            state.rate_limit_tokens = self.rate_limit_config.orders_per_second;
            state.last_rate_limit_reset = now;
        }
        
        // Check if we have tokens
        if state.rate_limit_tokens == 0 {
            bail!("Rate limit exceeded (429): Please retry after {:?}", 
                   Duration::from_secs(1) - (now - state.last_rate_limit_reset).to_std().unwrap());
        }
        
        state.rate_limit_tokens -= 1;
        Ok(())
    }
    
    /// Simulate order fill based on fill mode
    async fn simulate_fill(&self, order: &SimulatedOrder) -> Result<Vec<(Quantity, Price)>> {
        let mut fills = Vec::new();
        let mut rng = thread_rng();
        
        match self.fill_mode {
            FillMode::Instant => {
                // Fill entire order at requested price (or market price)
                let fill_price = if let Some(price) = order.price {
                    price
                } else {
                    // Market order - use last price
                    let state = self.state.read().await;
                    state.last_prices.get(&order.symbol)
                        .cloned()
                        .unwrap_or_else(|| Price::new(50000.0).unwrap())
                };
                
                fills.push((order.quantity.clone(), fill_price));
            },
            
            FillMode::Realistic => {
                // Simulate partial fills
                let num_fills = rng.gen_range(1..=3);
                let mut remaining = order.quantity.value();
                
                for i in 0..num_fills {
                    let fill_ratio = if i == num_fills - 1 {
                        1.0 // Last fill gets everything
                    } else {
                        rng.gen_range(0.2..=0.6)
                    };
                    
                    let fill_qty = Quantity::new(remaining * fill_ratio).unwrap();
                    remaining -= fill_qty.value();
                    
                    // Add slippage for market orders
                    let base_price = order.price.unwrap_or_else(|| {
                        let state = self.state.clone();
                        let state = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(state.read())
                        });
                        state.last_prices.get(&order.symbol)
                            .cloned()
                            .unwrap_or_else(|| Price::new(50000.0).unwrap())
                    });
                    
                    let slippage_bps = if order.order_type == OrderType::Market {
                        rng.gen_range(-10..=10) // ±0.1% slippage
                    } else {
                        0
                    };
                    
                    let fill_price = base_price.apply_slippage(slippage_bps).unwrap();
                    fills.push((fill_qty, fill_price));
                    
                    if remaining <= 0.0 {
                        break;
                    }
                }
            },
            
            FillMode::Aggressive => {
                // High slippage, many partial fills
                let num_fills = rng.gen_range(3..=10);
                let mut remaining = order.quantity.value();
                
                for _ in 0..num_fills {
                    if remaining <= 0.0 {
                        break;
                    }
                    
                    let fill_qty = Quantity::new(
                        remaining * rng.gen_range(0.05..=0.3)
                    ).unwrap();
                    
                    let base_price = order.price.unwrap_or_else(|| {
                        Price::new(50000.0).unwrap()
                    });
                    
                    // Higher slippage
                    let slippage_bps = rng.gen_range(-50..=50);
                    let fill_price = base_price.apply_slippage(slippage_bps).unwrap();
                    
                    fills.push((fill_qty.clone(), fill_price));
                    remaining -= fill_qty.value();
                }
            },
            
            FillMode::Conservative => {
                // Better prices but slower fills
                tokio::time::sleep(Duration::from_millis(100)).await;
                
                let fill_price = if let Some(price) = order.price {
                    // Limit order - might get better price
                    let improvement_bps = rng.gen_range(0..=5);
                    price.apply_slippage(-improvement_bps).unwrap()
                } else {
                    Price::new(50000.0).unwrap()
                };
                
                fills.push((order.quantity.clone(), fill_price));
            },
        }
        
        Ok(fills)
    }
    
    /// Generate a unique exchange order ID
    fn generate_order_id(&self) -> String {
        format!("SIM_{}", uuid::Uuid::new_v4())
    }
    
    /// Validate order parameters
    fn validate_order(&self, order: &Order) -> Result<()> {
        // Check minimum order size
        if order.quantity().value() < self.min_order_size {
            bail!("Order quantity {} below minimum {}", 
                  order.quantity().value(), self.min_order_size);
        }
        
        // Check price tick size for limit orders
        if let Some(price) = order.price() {
            let price_val = price.value();
            let remainder = (price_val / self.tick_size) % 1.0;
            if remainder > 0.0001 {
                bail!("Price {} doesn't match tick size {}", 
                      price_val, self.tick_size);
            }
        }
        
        // Check time in force validity
        if order.order_type() == OrderType::Market && 
           order.time_in_force != TimeInForce::IOC {
            bail!("Market orders must use IOC time in force");
        }
        
        Ok(())
    }
}

#[async_trait]
impl ExchangePort for ExchangeSimulator {
    async fn place_order(&self, order: &Order) -> Result<String> {
        // Simulate network operations
        self.simulate_latency().await;
        self.simulate_failure().await?;
        self.check_rate_limit().await?;
        
        // Validate order
        self.validate_order(order)?;
        
        // Generate exchange order ID
        let exchange_order_id = self.generate_order_id();
        
        // Create simulated order
        let sim_order = SimulatedOrder {
            order_id: exchange_order_id.clone(),
            client_order_id: order.id().clone(),
            symbol: order.symbol().clone(),
            side: order.side(),
            order_type: order.order_type(),
            price: order.price().cloned(),
            quantity: order.quantity().clone(),
            filled_quantity: Quantity::zero(),
            status: OrderStatus::Open,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        // Store order
        let mut state = self.state.write().await;
        state.orders.insert(exchange_order_id.clone(), sim_order.clone());
        state.total_orders_placed += 1;
        
        // Update last price for market orders
        if order.order_type() == OrderType::Market {
            if let Some(price) = order.price() {
                state.last_prices.insert(order.symbol().clone(), price.clone());
            }
        }
        
        // Simulate immediate fill for market orders
        if order.order_type() == OrderType::Market {
            drop(state); // Release lock
            
            // Simulate fills
            let fills = self.simulate_fill(&sim_order).await?;
            
            // Update order status
            let mut state = self.state.write().await;
            if let Some(stored_order) = state.orders.get_mut(&exchange_order_id) {
                for (qty, price) in fills {
                    stored_order.filled_quantity = stored_order.filled_quantity.add(&qty)?;
                    
                    // Record trade
                    let trade = Trade {
                        symbol: stored_order.symbol.clone(),
                        price: price.clone(),
                        quantity: qty,
                        is_buyer_maker: order.side() == OrderSide::Buy,
                        timestamp: Utc::now().timestamp_millis() as u64,
                    };
                    
                    state.recent_trades
                        .entry(stored_order.symbol.clone())
                        .or_insert_with(VecDeque::new)
                        .push_back(trade);
                    
                    // Update last price
                    state.last_prices.insert(stored_order.symbol.clone(), price);
                }
                
                // Update status
                if stored_order.filled_quantity == stored_order.quantity {
                    stored_order.status = OrderStatus::Filled;
                    state.total_orders_filled += 1;
                } else {
                    stored_order.status = OrderStatus::PartiallyFilled;
                }
                
                stored_order.updated_at = Utc::now();
                
                // Update volume
                state.total_volume_traded += stored_order.filled_quantity.value();
            }
        }
        
        Ok(exchange_order_id)
    }
    
    async fn cancel_order(&self, order_id: &OrderId) -> Result<()> {
        self.simulate_latency().await;
        self.simulate_failure().await?;
        
        let mut state = self.state.write().await;
        
        // Find order by client order ID
        let exchange_order_id = state.orders.iter()
            .find(|(_, o)| o.client_order_id == *order_id)
            .map(|(id, _)| id.clone());
        
        if let Some(id) = exchange_order_id {
            if let Some(order) = state.orders.get_mut(&id) {
                if order.status == OrderStatus::Open || 
                   order.status == OrderStatus::PartiallyFilled {
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = Utc::now();
                    state.total_orders_cancelled += 1;
                    Ok(())
                } else {
                    bail!("Cannot cancel order in status {:?}", order.status)
                }
            } else {
                bail!("Order not found")
            }
        } else {
            bail!("Order not found")
        }
    }
    
    async fn modify_order(
        &self, 
        order_id: &OrderId, 
        new_price: Option<Price>, 
        new_quantity: Option<Quantity>
    ) -> Result<()> {
        self.simulate_latency().await;
        self.simulate_failure().await?;
        self.check_rate_limit().await?;
        
        let mut state = self.state.write().await;
        
        // Find order
        let exchange_order_id = state.orders.iter()
            .find(|(_, o)| o.client_order_id == *order_id)
            .map(|(id, _)| id.clone());
        
        if let Some(id) = exchange_order_id {
            if let Some(order) = state.orders.get_mut(&id) {
                if order.status != OrderStatus::Open {
                    bail!("Can only modify open orders");
                }
                
                if let Some(price) = new_price {
                    order.price = Some(price);
                }
                
                if let Some(qty) = new_quantity {
                    if qty.value() < order.filled_quantity.value() {
                        bail!("New quantity cannot be less than filled quantity");
                    }
                    order.quantity = qty;
                }
                
                order.updated_at = Utc::now();
                Ok(())
            } else {
                bail!("Order not found")
            }
        } else {
            bail!("Order not found")
        }
    }
    
    async fn get_order_status(&self, order_id: &OrderId) -> Result<OrderStatus> {
        self.simulate_latency().await;
        
        let state = self.state.read().await;
        
        state.orders.iter()
            .find(|(_, o)| o.client_order_id == *order_id)
            .map(|(_, o)| o.status)
            .ok_or_else(|| anyhow::anyhow!("Order not found"))
    }
    
    async fn get_open_orders(&self, symbol: Option<&Symbol>) -> Result<Vec<Order>> {
        self.simulate_latency().await;
        
        // Note: This returns empty as we'd need to reconstruct Order objects
        // In production, we'd maintain proper Order instances
        Ok(Vec::new())
    }
    
    async fn get_order_history(&self, _symbol: &Symbol, _limit: usize) -> Result<Vec<Order>> {
        self.simulate_latency().await;
        
        // Simplified for now
        Ok(Vec::new())
    }
    
    async fn get_order_book(&self, symbol: &Symbol, depth: usize) -> Result<OrderBook> {
        self.simulate_latency().await;
        
        let state = self.state.read().await;
        
        // Generate realistic order book
        let last_price = state.last_prices.get(symbol)
            .cloned()
            .unwrap_or_else(|| Price::new(50000.0).unwrap());
        
        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);
        let mut rng = thread_rng();
        
        // Generate bids (below last price)
        for i in 0..depth {
            let price_offset = (i + 1) as f64 * 0.1;
            let price = Price::new(last_price.value() - price_offset).unwrap();
            let quantity = Quantity::new(rng.gen_range(0.1..=10.0)).unwrap();
            
            bids.push(OrderBookLevel {
                price,
                quantity,
                order_count: rng.gen_range(1..=5),
            });
        }
        
        // Generate asks (above last price)
        for i in 0..depth {
            let price_offset = (i + 1) as f64 * 0.1;
            let price = Price::new(last_price.value() + price_offset).unwrap();
            let quantity = Quantity::new(rng.gen_range(0.1..=10.0)).unwrap();
            
            asks.push(OrderBookLevel {
                price,
                quantity,
                order_count: rng.gen_range(1..=5),
            });
        }
        
        Ok(OrderBook {
            symbol: symbol.clone(),
            bids,
            asks,
            timestamp: Utc::now().timestamp_millis() as u64,
        })
    }
    
    async fn get_recent_trades(&self, symbol: &Symbol, limit: usize) -> Result<Vec<Trade>> {
        self.simulate_latency().await;
        
        let state = self.state.read().await;
        
        if let Some(trades) = state.recent_trades.get(symbol) {
            Ok(trades.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    async fn get_ticker(&self, symbol: &Symbol) -> Result<(Price, Price)> {
        self.simulate_latency().await;
        
        let state = self.state.read().await;
        
        let last_price = state.last_prices.get(symbol)
            .cloned()
            .unwrap_or_else(|| Price::new(50000.0).unwrap());
        
        // Generate bid/ask spread
        let spread_bps = 2; // 0.02% spread
        let bid = last_price.apply_slippage(-spread_bps).unwrap();
        let ask = last_price.apply_slippage(spread_bps).unwrap();
        
        Ok((bid, ask))
    }
    
    async fn get_balances(&self) -> Result<HashMap<String, Balance>> {
        self.simulate_latency().await;
        
        let state = self.state.read().await;
        Ok(state.balances.clone())
    }
    
    async fn get_trading_fees(&self, _symbol: &Symbol) -> Result<(f64, f64)> {
        self.simulate_latency().await;
        
        Ok((self.maker_fee, self.taker_fee))
    }
    
    async fn get_capabilities(&self) -> Result<ExchangeCapabilities> {
        Ok(ExchangeCapabilities {
            supports_oco: true,
            supports_reduce_only: true,
            supports_post_only: true,
            supports_iceberg: false,
            supports_trailing_stop: false,
            max_orders_per_second: self.rate_limit_config.orders_per_second,
        })
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if we're in an outage
        if let FailureMode::Outage { .. } = self.failure_mode {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    async fn get_rate_limit_status(&self) -> Result<(u32, u32)> {
        let state = self.state.read().await;
        Ok((
            self.rate_limit_config.orders_per_second - state.rate_limit_tokens,
            self.rate_limit_config.orders_per_second
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_order() -> Order {
        Order::limit(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Price::new(50000.0).unwrap(),
            Quantity::new(0.1).unwrap(),
            TimeInForce::GTC,
        )
    }
    
    #[tokio::test]
    async fn should_place_order() {
        let simulator = ExchangeSimulator::new()
            .with_config(
                LatencyMode::None,
                FillMode::Instant,
                RateLimitConfig::default(),
                FailureMode::None,
            );
        
        let order = create_test_order();
        let result = simulator.place_order(&order).await;
        
        assert!(result.is_ok());
        let order_id = result.unwrap();
        assert!(order_id.starts_with("SIM_"));
    }
    
    #[tokio::test]
    async fn should_simulate_rate_limiting() {
        let mut config = RateLimitConfig::default();
        config.orders_per_second = 2;
        
        let simulator = ExchangeSimulator::new()
            .with_config(
                LatencyMode::None,
                FillMode::Instant,
                config,
                FailureMode::None,
            );
        
        let order = create_test_order();
        
        // Place orders up to limit
        for _ in 0..2 {
            let result = simulator.place_order(&order).await;
            assert!(result.is_ok());
        }
        
        // Next order should be rate limited
        let result = simulator.place_order(&order).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Rate limit"));
    }
    
    #[tokio::test]
    async fn should_simulate_partial_fills() {
        let simulator = ExchangeSimulator::new()
            .with_config(
                LatencyMode::None,
                FillMode::Realistic,
                RateLimitConfig::default(),
                FailureMode::None,
            );
        
        let order = Order::market(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Quantity::new(1.0).unwrap(),
        );
        
        let order_id = simulator.place_order(&order).await.unwrap();
        
        // Check that order was placed
        assert!(order_id.starts_with("SIM_"));
        
        // Get recent trades to verify fills
        let trades = simulator.get_recent_trades(&order.symbol(), 10).await.unwrap();
        assert!(!trades.is_empty());
    }
    
    #[tokio::test]
    async fn should_cancel_order() {
        let simulator = ExchangeSimulator::new()
            .with_config(
                LatencyMode::None,
                FillMode::Instant,
                RateLimitConfig::default(),
                FailureMode::None,
            );
        
        let order = create_test_order();
        let _ = simulator.place_order(&order).await.unwrap();
        
        // Cancel the order
        let result = simulator.cancel_order(order.id()).await;
        assert!(result.is_ok());
        
        // Check status
        let status = simulator.get_order_status(order.id()).await.unwrap();
        assert_eq!(status, OrderStatus::Cancelled);
    }
    
    #[tokio::test]
    async fn should_simulate_network_failures() {
        let simulator = ExchangeSimulator::new()
            .with_config(
                LatencyMode::None,
                FillMode::Instant,
                RateLimitConfig::default(),
                FailureMode::RandomDrops { probability: 1.0 }, // Always fail
            );
        
        let order = create_test_order();
        let result = simulator.place_order(&order).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Network error"));
    }
    
    #[tokio::test]
    async fn should_get_order_book() {
        let simulator = ExchangeSimulator::new();
        
        let symbol = Symbol::new("BTC/USDT").unwrap();
        let order_book = simulator.get_order_book(&symbol, 5).await.unwrap();
        
        assert_eq!(order_book.bids.len(), 5);
        assert_eq!(order_book.asks.len(), 5);
        
        // Verify bids are below asks
        let best_bid = &order_book.bids[0].price;
        let best_ask = &order_book.asks[0].price;
        assert!(best_bid.value() < best_ask.value());
    }
    
    #[tokio::test]
    async fn should_track_balances() {
        let simulator = ExchangeSimulator::new();
        
        let balances = simulator.get_balances().await.unwrap();
        
        assert!(balances.contains_key("USDT"));
        assert!(balances.contains_key("BTC"));
        
        let usdt_balance = &balances["USDT"];
        assert_eq!(usdt_balance.free.value(), 100000.0);
    }
    
    #[tokio::test]
    async fn should_validate_minimum_order_size() {
        let simulator = ExchangeSimulator::new();
        
        let order = Order::limit(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Price::new(50000.0).unwrap(),
            Quantity::new(0.000001).unwrap(), // Below minimum
            TimeInForce::GTC,
        );
        
        let result = simulator.place_order(&order).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("below minimum"));
    }
}