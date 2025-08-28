pub use domain_types::market_data::{OrderBook, OrderBookLevel, OrderBookUpdate};

// LOB Simulator - Core Order Book Reconstruction and Simulation
// DEEP DIVE: Full order book dynamics with L3 data support
//
// References:
// - "Trades, Quotes and Prices" - Bouchaud, Bonart, Donier, Gould (2018)
// - "The Microstructure of the Euro Money Market" - ECB (2024)
// - NASDAQ TotalView-ITCH 5.0 specification
// - CME MDP 3.0 (Market Data Protocol)

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{debug, warn, error, instrument};
use dashmap::DashMap;
use ahash::AHashMap;

use types::{Price, Quantity, Symbol, Exchange};
// TODO: use infrastructure::metrics::{MetricsCollector, register_histogram, register_counter};

/// Order book level with complete information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: u32,  // Number of orders at this level
    pub exchange_timestamp: DateTime<Utc>,
    pub local_timestamp: DateTime<Utc>,
    pub implied_quantity: Option<Quantity>,  // For futures with implied orders
}

/// Side of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

/// Update type for order book changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    /// New order added
    Add {
        order_id: u64,
        side: Side,
        price: Price,
        quantity: Quantity,
    },
    /// Order modified
    Modify {
        order_id: u64,
        new_quantity: Quantity,
    },
    /// Order cancelled
    Cancel {
        order_id: u64,
    },
    /// Trade executed
    Trade {
        order_id: u64,
        traded_quantity: Quantity,
        aggressor_side: Side,
    },
    /// Full snapshot refresh
    Snapshot {
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
    },
    /// Clear entire book (trading halt)
    Clear,
}

/// Complete order book state at a point in time
#[derive(Debug, Clone)]

/// Individual order information for L3 reconstruction
#[derive(Debug, Clone)]
pub struct OrderInfo {
    pub order_id: u64,
    pub side: Side,
    pub price: Price,
    pub original_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub timestamp: DateTime<Utc>,
    pub priority_timestamp: DateTime<Utc>,  // For time priority
    pub hidden: bool,  // Iceberg/hidden orders
}

/// Order book snapshot for efficient storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: Symbol,
    pub exchange: Exchange,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
    pub bid_levels: Vec<OrderBookLevel>,
    pub ask_levels: Vec<OrderBookLevel>,
    pub checksum: u32,  // For integrity validation
}

/// Order book update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    pub symbol: Symbol,
    pub exchange: Exchange,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
    pub update_type: UpdateType,
    pub latency_ns: u64,  // Network latency in nanoseconds
}

/// Configuration for the LOB simulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Maximum number of price levels to maintain
    pub max_depth_levels: usize,
    
    /// Enable L3 order tracking
    pub enable_order_tracking: bool,
    
    /// Track hidden/iceberg orders
    pub track_hidden_liquidity: bool,
    
    /// Snapshot interval in milliseconds
    pub snapshot_interval_ms: u64,
    
    /// Enable crossed book detection
    pub detect_crossed_books: bool,
    
    /// Queue position modeling
    pub model_queue_position: bool,
    
    /// Latency simulation parameters
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub latency_std_dev_us: f64,
    
    /// Memory pool size for zero-copy
    pub memory_pool_size: usize,
    
    /// Checkpointing for recovery
    pub checkpoint_interval_sec: u64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            max_depth_levels: 100,
            enable_order_tracking: true,
            track_hidden_liquidity: true,
            snapshot_interval_ms: 100,
            detect_crossed_books: true,
            model_queue_position: true,
            min_latency_us: 50,
            max_latency_us: 1000,
            latency_std_dev_us: 100.0,
            memory_pool_size: 1_000_000,
            checkpoint_interval_sec: 60,
        }
    }
}

/// Metrics for monitoring simulation performance
pub struct SimulatorMetrics {
    pub updates_processed: Arc<dyn MetricsCollector>,
    pub snapshots_generated: Arc<dyn MetricsCollector>,
    pub crossed_books_detected: Arc<dyn MetricsCollector>,
    pub latency_distribution: Arc<dyn MetricsCollector>,
    pub depth_imbalance: Arc<dyn MetricsCollector>,
    pub spread_distribution: Arc<dyn MetricsCollector>,
}

/// Main LOB Simulator implementation
pub struct LOBSimulator {
    config: Arc<SimulatorConfig>,
    
    // Order books per symbol
    books: Arc<DashMap<Symbol, Arc<RwLock<OrderBook>>>>,
    
    // Historical snapshots for replay
    snapshots: Arc<DashMap<Symbol, VecDeque<OrderBookSnapshot>>>,
    
    // Update queue for replay
    update_queue: Arc<Mutex<VecDeque<OrderBookUpdate>>>,
    
    // Metrics
    metrics: Arc<SimulatorMetrics>,
    
    // Sequence number tracking
    sequence_tracker: Arc<DashMap<Symbol, u64>>,
    
    // Checkpointing
    last_checkpoint: Arc<RwLock<DateTime<Utc>>>,
}

impl LOBSimulator {
    pub fn new(config: SimulatorConfig) -> Result<Self> {
        let metrics = Arc::new(SimulatorMetrics {
            updates_processed: register_counter("lob_updates_processed"),
            snapshots_generated: register_counter("lob_snapshots_generated"),
            crossed_books_detected: register_counter("lob_crossed_books"),
            latency_distribution: register_histogram("lob_latency_us"),
            depth_imbalance: register_histogram("lob_depth_imbalance"),
            spread_distribution: register_histogram("lob_spread_bps"),
        });
        
        Ok(Self {
            config: Arc::new(config),
            books: Arc::new(DashMap::new()),
            snapshots: Arc::new(DashMap::new()),
            update_queue: Arc::new(Mutex::new(VecDeque::with_capacity(100_000))),
            metrics,
            sequence_tracker: Arc::new(DashMap::new()),
            last_checkpoint: Arc::new(RwLock::new(Utc::now())),
        })
    }
    
    /// Initialize order book for a symbol
    #[instrument(skip(self))]
    pub fn initialize_book(&self, symbol: Symbol, exchange: Exchange) -> Result<()> {
        let book = OrderBook {
            symbol: symbol.clone(),
            exchange,
            timestamp: Utc::now(),
            sequence_number: 0,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            is_crossed: false,
            is_locked: false,
            last_trade_price: None,
            last_trade_quantity: None,
            total_bid_depth: Quantity(Decimal::ZERO),
            total_ask_depth: Quantity(Decimal::ZERO),
            spread: None,
            mid_price: None,
            weighted_mid_price: None,
            micro_price: None,
        };
        
        self.books.insert(symbol.clone(), Arc::new(RwLock::new(book)));
        self.sequence_tracker.insert(symbol, 0);
        
        Ok(())
    }
    
    /// Process an order book update
    #[instrument(skip(self, update), fields(symbol = %update.symbol.0, seq = update.sequence_number))]
    pub async fn process_update(&self, update: OrderBookUpdate) -> Result<()> {
        // Validate sequence number
        self.validate_sequence(&update)?;
        
        // Get or create order book
        let book_arc = self.books
            .entry(update.symbol.clone())
            .or_insert_with(|| {
                let mut book = OrderBook {
                    symbol: update.symbol.clone(),
                    exchange: update.exchange.clone(),
                    timestamp: update.timestamp,
                    sequence_number: update.sequence_number,
                    bids: BTreeMap::new(),
                    asks: BTreeMap::new(),
                    orders: HashMap::new(),
                    is_crossed: false,
                    is_locked: false,
                    last_trade_price: None,
                    last_trade_quantity: None,
                    total_bid_depth: Quantity(Decimal::ZERO),
                    total_ask_depth: Quantity(Decimal::ZERO),
                    spread: None,
                    mid_price: None,
                    weighted_mid_price: None,
                    micro_price: None,
                };
                Arc::new(RwLock::new(book))
            });
        
        let mut book = book_arc.write();
        
        // Apply update based on type
        match &update.update_type {
            UpdateType::Add { order_id, side, price, quantity } => {
                self.apply_add_order(&mut book, *order_id, *side, price.clone(), quantity.clone())?;
            }
            UpdateType::Modify { order_id, new_quantity } => {
                self.apply_modify_order(&mut book, *order_id, new_quantity.clone())?;
            }
            UpdateType::Cancel { order_id } => {
                self.apply_cancel_order(&mut book, *order_id)?;
            }
            UpdateType::Trade { order_id, traded_quantity, aggressor_side } => {
                self.apply_trade(&mut book, *order_id, traded_quantity.clone(), *aggressor_side)?;
            }
            UpdateType::Snapshot { bids, asks } => {
                self.apply_snapshot(&mut book, bids.clone(), asks.clone())?;
            }
            UpdateType::Clear => {
                self.clear_book(&mut book)?;
            }
        }
        
        // Update book statistics
        self.update_book_stats(&mut book)?;
        
        // Check for crossed book
        if self.config.detect_crossed_books {
            self.check_crossed_book(&book)?;
        }
        
        // Update metrics
// self.metrics.updates_processed.increment(1);
        
        // Record latency
// self.metrics.latency_distribution.record(update.latency_ns as f64 / 1000.0);
        
        // Generate snapshot if needed
        if self.should_snapshot(&book) {
            self.generate_snapshot(&book)?;
        }
        
        Ok(())
    }
    
    /// Apply add order update
    fn apply_add_order(
        &self,
        book: &mut OrderBook,
        order_id: u64,
        side: Side,
        price: Price,
        quantity: Quantity,
    ) -> Result<()> {
        // Track individual order if L3 enabled
        if self.config.enable_order_tracking {
            let order_info = OrderInfo {
                order_id,
                side,
                price: price.clone(),
                original_quantity: quantity.clone(),
                remaining_quantity: quantity.clone(),
                timestamp: book.timestamp,
                priority_timestamp: book.timestamp,
                hidden: false,
            };
            book.orders.insert(order_id, order_info);
        }
        
        // Update price level
        let price_decimal = price.0;
        let level_map = match side {
            Side::Bid => &mut book.bids,
            Side::Ask => &mut book.asks,
        };
        
        level_map.entry(price_decimal)
            .and_modify(|level| {
                level.quantity = Quantity(level.quantity.0 + quantity.0);
                level.order_count += 1;
            })
            .or_insert_with(|| OrderBookLevel {
                price: price.clone(),
                quantity: quantity.clone(),
                order_count: 1,
                exchange_timestamp: book.timestamp,
                local_timestamp: Utc::now(),
                implied_quantity: None,
            });
        
        // Update total depth
        match side {
            Side::Bid => book.total_bid_depth = Quantity(book.total_bid_depth.0 + quantity.0),
            Side::Ask => book.total_ask_depth = Quantity(book.total_ask_depth.0 + quantity.0),
        }
        
        Ok(())
    }
    
    /// Apply modify order update
    fn apply_modify_order(
        &self,
        book: &mut OrderBook,
        order_id: u64,
        new_quantity: Quantity,
    ) -> Result<()> {
        if !self.config.enable_order_tracking {
            return Ok(());
        }
        
        if let Some(order_info) = book.orders.get_mut(&order_id) {
            let old_quantity = order_info.remaining_quantity.clone();
            let quantity_diff = new_quantity.0 - old_quantity.0;
            
            order_info.remaining_quantity = new_quantity.clone();
            
            // Update price level
            let price_decimal = order_info.price.0;
            let level_map = match order_info.side {
                Side::Bid => &mut book.bids,
                Side::Ask => &mut book.asks,
            };
            
            if let Some(level) = level_map.get_mut(&price_decimal) {
                level.quantity = Quantity(level.quantity.0 + quantity_diff);
            }
            
            // Update total depth
            match order_info.side {
                Side::Bid => book.total_bid_depth = Quantity(book.total_bid_depth.0 + quantity_diff),
                Side::Ask => book.total_ask_depth = Quantity(book.total_ask_depth.0 + quantity_diff),
            }
        }
        
        Ok(())
    }
    
    /// Apply cancel order update
    fn apply_cancel_order(
        &self,
        book: &mut OrderBook,
        order_id: u64,
    ) -> Result<()> {
        if !self.config.enable_order_tracking {
            return Ok(());
        }
        
        if let Some(order_info) = book.orders.remove(&order_id) {
            let price_decimal = order_info.price.0;
            let level_map = match order_info.side {
                Side::Bid => &mut book.bids,
                Side::Ask => &mut book.asks,
            };
            
            if let Some(level) = level_map.get_mut(&price_decimal) {
                level.quantity = Quantity(level.quantity.0 - order_info.remaining_quantity.0);
                level.order_count = level.order_count.saturating_sub(1);
                
                // Remove level if empty
                if level.order_count == 0 || level.quantity.0 <= Decimal::ZERO {
                    level_map.remove(&price_decimal);
                }
            }
            
            // Update total depth
            match order_info.side {
                Side::Bid => {
                    book.total_bid_depth = Quantity(
                        (book.total_bid_depth.0 - order_info.remaining_quantity.0).max(Decimal::ZERO)
                    )
                },
                Side::Ask => {
                    book.total_ask_depth = Quantity(
                        (book.total_ask_depth.0 - order_info.remaining_quantity.0).max(Decimal::ZERO)
                    )
                },
            }
        }
        
        Ok(())
    }
    
    /// Apply trade update
    fn apply_trade(
        &self,
        book: &mut OrderBook,
        order_id: u64,
        traded_quantity: Quantity,
        aggressor_side: Side,
    ) -> Result<()> {
        // Update order if tracked
        if self.config.enable_order_tracking {
            if let Some(order_info) = book.orders.get_mut(&order_id) {
                order_info.remaining_quantity = Quantity(
                    (order_info.remaining_quantity.0 - traded_quantity.0).max(Decimal::ZERO)
                );
                
                // Update price level
                let price_decimal = order_info.price.0;
                let level_map = match order_info.side {
                    Side::Bid => &mut book.bids,
                    Side::Ask => &mut book.asks,
                };
                
                if let Some(level) = level_map.get_mut(&price_decimal) {
                    level.quantity = Quantity(
                        (level.quantity.0 - traded_quantity.0).max(Decimal::ZERO)
                    );
                    
                    // Set last trade info
                    book.last_trade_price = Some(order_info.price.clone());
                    book.last_trade_quantity = Some(traded_quantity.clone());
                }
                
                // Remove order if fully filled
                if order_info.remaining_quantity.0 <= Decimal::ZERO {
                    book.orders.remove(&order_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply snapshot update
    fn apply_snapshot(
        &self,
        book: &mut OrderBook,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
    ) -> Result<()> {
        // Clear existing book
        book.bids.clear();
        book.asks.clear();
        book.orders.clear();
        
        // Rebuild bid side
        book.total_bid_depth = Quantity(Decimal::ZERO);
        for level in bids {
            book.total_bid_depth = Quantity(book.total_bid_depth.0 + level.quantity.0);
            book.bids.insert(level.price.0, level);
        }
        
        // Rebuild ask side
        book.total_ask_depth = Quantity(Decimal::ZERO);
        for level in asks {
            book.total_ask_depth = Quantity(book.total_ask_depth.0 + level.quantity.0);
            book.asks.insert(level.price.0, level);
        }
        
        Ok(())
    }
    
    /// Clear entire order book
    fn clear_book(&self, book: &mut OrderBook) -> Result<()> {
        book.bids.clear();
        book.asks.clear();
        book.orders.clear();
        book.total_bid_depth = Quantity(Decimal::ZERO);
        book.total_ask_depth = Quantity(Decimal::ZERO);
        book.spread = None;
        book.mid_price = None;
        book.weighted_mid_price = None;
        book.micro_price = None;
        
        Ok(())
    }
    
    /// Update book statistics
    fn update_book_stats(&self, book: &mut OrderBook) -> Result<()> {
        // Get best bid and ask
        let best_bid = book.bids.iter().next_back().map(|(_, l)| l);
        let best_ask = book.asks.iter().next().map(|(_, l)| l);
        
        // Calculate spread
        if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
            book.spread = Some(ask.price.0 - bid.price.0);
            book.mid_price = Some(Price((bid.price.0 + ask.price.0) / Decimal::from(2)));
            
            // Weighted mid price
            let total_qty = bid.quantity.0 + ask.quantity.0;
            if total_qty > Decimal::ZERO {
                let weighted = (bid.price.0 * ask.quantity.0 + ask.price.0 * bid.quantity.0) / total_qty;
                book.weighted_mid_price = Some(Price(weighted));
            }
            
            // Micro price (considers multiple levels)
            if let Some(micro) = self.calculate_micro_price(book) {
                book.micro_price = Some(micro);
            }
            
            // Check if crossed or locked
            book.is_crossed = bid.price.0 > ask.price.0;
            book.is_locked = bid.price.0 == ask.price.0;
            
            // Record spread in basis points
            if let Some(mid) = &book.mid_price {
                let spread_bps = (book.spread.unwrap() / mid.0 * Decimal::from(10000))
                    .to_f64()
                    .unwrap_or(0.0);
// self.metrics.spread_distribution.record(spread_bps);
            }
        }
        
        // Record depth imbalance
        let total_depth = book.total_bid_depth.0 + book.total_ask_depth.0;
        if total_depth > Decimal::ZERO {
            let imbalance = ((book.total_bid_depth.0 - book.total_ask_depth.0) / total_depth)
                .to_f64()
                .unwrap_or(0.0);
// self.metrics.depth_imbalance.record(imbalance);
        }
        
        Ok(())
    }
    
    /// Calculate micro price using multiple levels
    fn calculate_micro_price(&self, book: &OrderBook) -> Option<Price> {
        let levels = 5; // Use top 5 levels
        
        let mut bid_value = Decimal::ZERO;
        let mut bid_size = Decimal::ZERO;
        for (_, level) in book.bids.iter().rev().take(levels) {
            bid_value += level.price.0 * level.quantity.0;
            bid_size += level.quantity.0;
        }
        
        let mut ask_value = Decimal::ZERO;
        let mut ask_size = Decimal::ZERO;
        for (_, level) in book.asks.iter().take(levels) {
            ask_value += level.price.0 * level.quantity.0;
            ask_size += level.quantity.0;
        }
        
        let total_size = bid_size + ask_size;
        if total_size > Decimal::ZERO {
            Some(Price((bid_value + ask_value) / total_size))
        } else {
            None
        }
    }
    
    /// Validate sequence number
    fn validate_sequence(&self, update: &OrderBookUpdate) -> Result<()> {
        let mut tracker = self.sequence_tracker.entry(update.symbol.clone()).or_insert(0);
        
        if update.sequence_number != *tracker + 1 {
            warn!(
                "Sequence gap detected for {}: expected {}, got {}",
                update.symbol.0,
                *tracker + 1,
                update.sequence_number
            );
        }
        
        *tracker = update.sequence_number;
        Ok(())
    }
    
    /// Check for crossed book condition
    fn check_crossed_book(&self, book: &OrderBook) -> Result<()> {
        if book.is_crossed {
// self.metrics.crossed_books_detected.increment(1);
            warn!("Crossed book detected for {}: bid > ask", book.symbol.0);
        }
        Ok(())
    }
    
    /// Determine if we should generate a snapshot
    fn should_snapshot(&self, book: &OrderBook) -> bool {
        // Simple time-based snapshotting for now
        let snapshot_interval = Duration::milliseconds(self.config.snapshot_interval_ms as i64);
        book.timestamp - self.last_checkpoint.read().clone() > snapshot_interval
    }
    
    /// Generate order book snapshot
    fn generate_snapshot(&self, book: &OrderBook) -> Result<()> {
        let snapshot = OrderBookSnapshot {
            symbol: book.symbol.clone(),
            exchange: book.exchange.clone(),
            timestamp: book.timestamp,
            sequence_number: book.sequence_number,
            bid_levels: book.bids.values().cloned().collect(),
            ask_levels: book.asks.values().cloned().collect(),
            checksum: self.calculate_checksum(book),
        };
        
        // Store snapshot
        self.snapshots
            .entry(book.symbol.clone())
            .or_insert_with(|| VecDeque::with_capacity(1000))
            .push_back(snapshot);
        
// self.metrics.snapshots_generated.increment(1);
        
        Ok(())
    }
    
    /// Calculate checksum for integrity validation
    fn calculate_checksum(&self, book: &OrderBook) -> u32 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash top 10 levels each side
        for (price, level) in book.bids.iter().rev().take(10) {
            price.hash(&mut hasher);
            level.quantity.0.hash(&mut hasher);
        }
        
        for (price, level) in book.asks.iter().take(10) {
            price.hash(&mut hasher);
            level.quantity.0.hash(&mut hasher);
        }
        
        hasher.finish() as u32
    }
    
    /// Get current order book state
    pub fn get_book(&self, symbol: &Symbol) -> Option<OrderBook> {
        self.books.get(symbol).map(|book_arc| book_arc.read().clone())
    }
    
    /// Get book depth to specified level
    pub fn get_book_depth(&self, symbol: &Symbol, levels: usize) -> Option<(Vec<OrderBookLevel>, Vec<OrderBookLevel>)> {
        self.books.get(symbol).map(|book_arc| {
            let book = book_arc.read();
            let bids: Vec<OrderBookLevel> = book.bids.values().rev().take(levels).cloned().collect();
            let asks: Vec<OrderBookLevel> = book.asks.values().take(levels).cloned().collect();
            (bids, asks)
        })
    }
    
    /// Queue position estimation for an order
    pub fn estimate_queue_position(&self, symbol: &Symbol, side: Side, price: Price, quantity: Quantity) -> Option<f64> {
        if !self.config.model_queue_position {
            return None;
        }
        
        self.books.get(symbol).map(|book_arc| {
            let book = book_arc.read();
            
            let level = match side {
                Side::Bid => book.bids.get(&price.0),
                Side::Ask => book.asks.get(&price.0),
            };
            
            if let Some(level) = level {
                // Simple pro-rata model
                // More sophisticated models would consider time priority
                let position_ratio = quantity.0 / level.quantity.0;
                position_ratio.to_f64().unwrap_or(0.5)
            } else {
                // New price level, would be first in queue
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_lob_simulator_initialization() {
        let config = SimulatorConfig::default();
        let simulator = LOBSimulator::new(config).unwrap();
        
        let symbol = Symbol("BTC-USDT".to_string());
        let exchange = Exchange("Binance".to_string());
        
        simulator.initialize_book(symbol.clone(), exchange).unwrap();
        
        let book = simulator.get_book(&symbol).unwrap();
        assert_eq!(book.symbol.0, "BTC-USDT");
        assert_eq!(book.bids.len(), 0);
        assert_eq!(book.asks.len(), 0);
    }
    
    #[tokio::test]
    async fn test_order_book_updates() {
        let config = SimulatorConfig::default();
        let simulator = LOBSimulator::new(config).unwrap();
        
        let symbol = Symbol("ETH-USDT".to_string());
        let exchange = Exchange("Binance".to_string());
        
        // Add buy order
        let update = OrderBookUpdate {
            symbol: symbol.clone(),
            exchange: exchange.clone(),
            timestamp: Utc::now(),
            sequence_number: 1,
            update_type: UpdateType::Add {
                order_id: 1001,
                side: Side::Bid,
                price: Price(Decimal::from_str("2000.50").unwrap()),
                quantity: Quantity(Decimal::from_str("10.5").unwrap()),
            },
            latency_ns: 100_000,
        };
        
        simulator.process_update(update).await.unwrap();
        
        let book = simulator.get_book(&symbol).unwrap();
        assert_eq!(book.bids.len(), 1);
        assert_eq!(book.total_bid_depth.0, Decimal::from_str("10.5").unwrap());
    }
    
    #[tokio::test]
    async fn test_crossed_book_detection() {
        let config = SimulatorConfig {
            detect_crossed_books: true,
            ..Default::default()
        };
        let simulator = LOBSimulator::new(config).unwrap();
        
        let symbol = Symbol("SOL-USDT".to_string());
        let exchange = Exchange("FTX".to_string());
        
        // Add bid higher than ask (crossed book)
        let bid_update = OrderBookUpdate {
            symbol: symbol.clone(),
            exchange: exchange.clone(),
            timestamp: Utc::now(),
            sequence_number: 1,
            update_type: UpdateType::Add {
                order_id: 2001,
                side: Side::Bid,
                price: Price(Decimal::from_str("100.50").unwrap()),
                quantity: Quantity(Decimal::from_str("100").unwrap()),
            },
            latency_ns: 50_000,
        };
        
        let ask_update = OrderBookUpdate {
            symbol: symbol.clone(),
            exchange: exchange.clone(),
            timestamp: Utc::now(),
            sequence_number: 2,
            update_type: UpdateType::Add {
                order_id: 2002,
                side: Side::Ask,
                price: Price(Decimal::from_str("100.00").unwrap()),
                quantity: Quantity(Decimal::from_str("50").unwrap()),
            },
            latency_ns: 50_000,
        };
        
        simulator.process_update(bid_update).await.unwrap();
        simulator.process_update(ask_update).await.unwrap();
        
        let book = simulator.get_book(&symbol).unwrap();
        assert!(book.is_crossed);
    }
}