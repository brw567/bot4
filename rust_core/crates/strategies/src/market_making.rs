use domain_types::MarketState;
//! # MARKET MAKING ENGINE - Liquidity Provision Strategy
//! Drew (Strategy Lead) + Full Team
//! 
//! Research Applied:
//! - "High-Frequency Trading" - Aldridge (2013)
//! - "Market Microstructure Theory" - O'Hara (1995)
//! - "Optimal Market Making" - Avellaneda & Stoikov (2008)
//! - "The Economics of Market Making" - Grossman & Miller (1988)

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc, Duration};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Market Making configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct MarketMakingConfig {
    /// Spread percentage from mid price
    pub base_spread_bps: u32,
    
    /// Minimum spread in basis points
    pub min_spread_bps: u32,
    
    /// Maximum spread in basis points
    pub max_spread_bps: u32,
    
    /// Order size per level
    pub order_size: Decimal,
    
    /// Number of price levels on each side
    pub num_levels: usize,
    
    /// Inventory target (neutral = 0)
    pub target_inventory: Decimal,
    
    /// Maximum inventory position
    pub max_inventory: Decimal,
    
    /// Inventory risk aversion parameter
    pub inventory_risk_aversion: f64,
    
    /// Skew orders based on inventory
    pub enable_inventory_skew: bool,
    
    /// Enable volatility-based spread adjustment
    pub enable_volatility_adjustment: bool,
    
    /// Cancel and replace threshold (price movement)
    pub update_threshold_bps: u32,
    
    /// Minimum time between updates (milliseconds)
    pub min_update_interval_ms: u64,
    
    /// Enable adverse selection protection
    pub enable_adverse_selection_protection: bool,
}

impl Default for MarketMakingConfig {
    fn default() -> Self {
        Self {
            base_spread_bps: 20, // 0.2%
            min_spread_bps: 10,
            max_spread_bps: 100,
            order_size: Decimal::from_str("0.1").unwrap(),
            num_levels: 5,
            target_inventory: Decimal::ZERO,
            max_inventory: Decimal::from(10),
            inventory_risk_aversion: 0.1,
            enable_inventory_skew: true,
            enable_volatility_adjustment: true,
            update_threshold_bps: 5,
            min_update_interval_ms: 100,
            enable_adverse_selection_protection: true,
        }
    }
}

/// Market Making Engine
/// TODO: Add docs
pub struct MarketMakingEngine {
    /// Configuration
    config: MarketMakingConfig,
    
    /// Current market state
    market_state: Arc<RwLock<MarketState>>,
    
    /// Inventory manager
    inventory_manager: Arc<InventoryManager>,
    
    /// Spread calculator
    spread_calculator: Arc<SpreadCalculator>,
    
    /// Order manager
    order_manager: Arc<OrderManager>,
    
    /// Risk manager
    risk_manager: Arc<RiskManager>,
    
    /// Performance metrics
    metrics: Arc<RwLock<MarketMakingMetrics>>,
}

/// Market state
#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::MarketState
// pub struct MarketState {
    pub bid: Decimal,
    pub ask: Decimal,
    pub mid: Decimal,
    pub last_trade: Decimal,
    pub volume_24h: Decimal,
    pub volatility: f64,
    pub order_book_imbalance: f64,
    pub trade_flow: TradeFlow,
    pub last_update: DateTime<Utc>,
}

/// Trade flow analysis
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TradeFlow {
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub buy_count: u32,
    pub sell_count: u32,
    pub vwap: Decimal,
    pub momentum: f64,
}

/// Inventory Manager
/// TODO: Add docs
pub struct InventoryManager {
    current_inventory: Arc<RwLock<Decimal>>,
    target_inventory: Decimal,
    max_inventory: Decimal,
    inventory_history: Arc<RwLock<Vec<InventorySnapshot>>>,
    risk_aversion: f64,
}

#[derive(Debug, Clone)]
struct InventorySnapshot {
    timestamp: DateTime<Utc>,
    inventory: Decimal,
    mid_price: Decimal,
    pnl: Decimal,
}

impl InventoryManager {
    /// Calculate inventory skew for quotes
    pub fn calculate_skew(&self, current_inventory: Decimal) -> (f64, f64) {
        let inventory_ratio = (current_inventory / self.max_inventory).to_f64().unwrap();
        
        // Avellaneda-Stoikov inventory adjustment
        let skew_factor = self.risk_aversion * inventory_ratio;
        
        // Asymmetric skew: reduce bid spread when long, reduce ask spread when short
        let bid_skew = if inventory_ratio > 0.0 {
            1.0 + skew_factor.abs() // Wider bid when long
        } else {
            1.0 - skew_factor.abs() * 0.5 // Tighter bid when short
        };
        
        let ask_skew = if inventory_ratio < 0.0 {
            1.0 + skew_factor.abs() // Wider ask when short
        } else {
            1.0 - skew_factor.abs() * 0.5 // Tighter ask when long
        };
        
        (bid_skew, ask_skew)
    }
    
    /// Check if inventory is within risk limits
    pub fn is_within_limits(&self, current_inventory: Decimal) -> bool {
        current_inventory.abs() <= self.max_inventory
    }
    
    /// Calculate inventory cost
    pub fn calculate_inventory_cost(&self, inventory: Decimal, mid_price: Decimal) -> Decimal {
        // Quadratic inventory penalty (Avellaneda-Stoikov)
        let penalty = Decimal::from_f64(self.risk_aversion).unwrap() 
            * inventory.abs() 
            * inventory.abs() 
            / self.max_inventory;
        
        penalty * mid_price
    }
}

/// Spread Calculator with advanced models
/// TODO: Add docs
pub struct SpreadCalculator {
    base_spread_bps: u32,
    min_spread_bps: u32,
    max_spread_bps: u32,
    volatility_multiplier: f64,
}

impl SpreadCalculator {
    /// Calculate optimal spread using Avellaneda-Stoikov model
    pub fn calculate_optimal_spread(
        &self,
        market_state: &MarketState,
        inventory: Decimal,
        time_horizon: f64,
    ) -> (Decimal, Decimal) {
        // Base spread
        let base_spread = market_state.mid * Decimal::from(self.base_spread_bps) / Decimal::from(10000);
        
        // Volatility adjustment
        let vol_adjustment = if market_state.volatility > 0.0 {
            1.0 + (market_state.volatility - 0.02) * self.volatility_multiplier
        } else {
            1.0
        };
        
        // Time decay factor (urgency increases as time horizon decreases)
        let time_factor = (1.0 / time_horizon).sqrt().min(2.0);
        
        // Order book imbalance adjustment
        let imbalance_adjustment = 1.0 + market_state.order_book_imbalance.abs() * 0.5;
        
        // Calculate final spreads
        let adjusted_spread = base_spread * Decimal::from_f64(vol_adjustment * time_factor * imbalance_adjustment).unwrap();
        
        // Apply min/max constraints
        let min_spread = market_state.mid * Decimal::from(self.min_spread_bps) / Decimal::from(10000);
        let max_spread = market_state.mid * Decimal::from(self.max_spread_bps) / Decimal::from(10000);
        
        let final_spread = adjusted_spread.max(min_spread).min(max_spread);
        
        (final_spread / Decimal::from(2), final_spread / Decimal::from(2))
    }
    
    /// Calculate spread using Garman's inventory model
    pub fn calculate_garman_spread(
        &self,
        volatility: f64,
        order_arrival_rate: f64,
        inventory: Decimal,
        max_inventory: Decimal,
    ) -> Decimal {
        // Garman (1976) optimal spread formula
        // s* = sqrt(2 * volatility^2 * ln(1 + gamma/lambda))
        // where gamma = risk aversion, lambda = order arrival rate
        
        let gamma = 0.1; // Risk aversion parameter
        let optimal_spread = (2.0 * volatility * volatility * (1.0 + gamma / order_arrival_rate).ln()).sqrt();
        
        // Adjust for inventory
        let inventory_ratio = (inventory / max_inventory).to_f64().unwrap();
        let inventory_adjustment = 1.0 + inventory_ratio.abs() * 0.2;
        
        Decimal::from_f64(optimal_spread * inventory_adjustment * 10000.0).unwrap() // Convert to bps
    }
}

/// Order Manager for quote management
/// TODO: Add docs
// ELIMINATED: pub struct OrderManager {
// ELIMINATED:     active_orders: Arc<RwLock<Vec<MakerOrder>>>,
// ELIMINATED:     order_history: Arc<RwLock<Vec<OrderEvent>>>,
// ELIMINATED:     last_update: Arc<RwLock<DateTime<Utc>>>,
// ELIMINATED: }

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MakerOrder {
    pub id: String,
    pub side: OrderSide,
    pub price: Decimal,
    pub size: Decimal,
    pub level: usize,
    pub created_at: DateTime<Utc>,
    pub status: OrderStatus,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderEvent {
    Placed { order: MakerOrder, timestamp: DateTime<Utc> },
    Filled { order_id: String, fill_price: Decimal, fill_size: Decimal, timestamp: DateTime<Utc> },
    Cancelled { order_id: String, reason: String, timestamp: DateTime<Utc> },
    Updated { old_order: MakerOrder, new_order: MakerOrder, timestamp: DateTime<Utc> },
}

impl OrderManager {
    /// Generate quote ladder
    pub fn generate_quotes(
        &self,
        mid_price: Decimal,
        bid_spread: Decimal,
        ask_spread: Decimal,
        num_levels: usize,
        size_per_level: Decimal,
    ) -> Vec<Quote> {
        let mut quotes = Vec::new();
        
        // Generate bid levels
        for i in 0..num_levels {
            let level_adjustment = Decimal::from(i) * mid_price * Decimal::from_str("0.0001").unwrap(); // 1 bps per level
            let price = mid_price - bid_spread - level_adjustment;
            
            quotes.push(Quote {
                side: OrderSide::Buy,
                price,
                size: size_per_level,
                level: i,
            });
        }
        
        // Generate ask levels
        for i in 0..num_levels {
            let level_adjustment = Decimal::from(i) * mid_price * Decimal::from_str("0.0001").unwrap();
            let price = mid_price + ask_spread + level_adjustment;
            
            quotes.push(Quote {
                side: OrderSide::Sell,
                price,
                size: size_per_level,
                level: i,
            });
        }
        
        quotes
    }
    
    /// Check if quotes need update
    pub fn needs_update(&self, market_state: &MarketState, threshold_bps: u32) -> bool {
        let orders = self.active_orders.read();
        
        if orders.is_empty() {
            return true;
        }
        
        // Check if mid price moved beyond threshold
        for order in orders.iter() {
            let price_diff = ((order.price - market_state.mid) / market_state.mid).abs();
            if price_diff > Decimal::from(threshold_bps) / Decimal::from(10000) {
                return true;
            }
        }
        
        // Check if enough time passed
        let last_update = *self.last_update.read();
        if Utc::now() - last_update > Duration::seconds(1) {
            return true;
        }
        
        false
    }
}

/// Risk Manager for market making
/// TODO: Add docs
pub struct RiskManager {
    max_inventory: Decimal,
    max_order_size: Decimal,
    min_spread_bps: u32,
    adverse_selection_threshold: f64,
}

impl RiskManager {
    /// Check if order passes risk checks
    pub fn validate_order(&self, quote: &Quote, current_inventory: Decimal) -> Result<(), RiskViolation> {
        // Check inventory limits
        let projected_inventory = match quote.side {
            OrderSide::Buy => current_inventory + quote.size,
            OrderSide::Sell => current_inventory - quote.size,
        };
        
        if projected_inventory.abs() > self.max_inventory {
            return Err(RiskViolation::InventoryLimitExceeded);
        }
        
        // Check order size
        if quote.size > self.max_order_size {
            return Err(RiskViolation::OrderSizeTooLarge);
        }
        
        Ok(())
    }
    
    /// Detect adverse selection
    pub fn detect_adverse_selection(&self, fills: &[FillEvent]) -> bool {
        if fills.len() < 10 {
            return false;
        }
        
        // Calculate realized spread (execution price vs mid price after delay)
        let mut negative_spreads = 0;
        for fill in fills.iter() {
            if fill.realized_spread < Decimal::ZERO {
                negative_spreads += 1;
            }
        }
        
        let adverse_rate = negative_spreads as f64 / fills.len() as f64;
        adverse_rate > self.adverse_selection_threshold
    }
}

/// Quote representation
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct Quote {
    pub side: OrderSide,
    pub price: Decimal,
    pub size: Decimal,
    pub level: usize,
}

/// Fill event
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct FillEvent {
    pub order_id: String,
    pub price: Decimal,
    pub size: Decimal,
    pub side: OrderSide,
    pub timestamp: DateTime<Utc>,
    pub mid_at_fill: Decimal,
    pub mid_after_delay: Decimal,
    pub realized_spread: Decimal,
}

/// Performance metrics
#[derive(Debug, Default)]
/// TODO: Add docs
pub struct MarketMakingMetrics {
    pub total_volume: Decimal,
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub num_fills: u64,
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub spread_captured: Decimal,
    pub avg_spread_bps: f64,
    pub fill_rate: f64,
    pub adverse_selection_rate: f64,
    pub inventory_turnover: f64,
    pub sharpe_ratio: f64,
}

impl MarketMakingEngine {
    pub fn new(config: MarketMakingConfig) -> Self {
        let inventory_manager = Arc::new(InventoryManager {
            current_inventory: Arc::new(RwLock::new(Decimal::ZERO)),
            target_inventory: config.target_inventory,
            max_inventory: config.max_inventory,
            inventory_history: Arc::new(RwLock::new(Vec::new())),
            risk_aversion: config.inventory_risk_aversion,
        });
        
        let spread_calculator = Arc::new(SpreadCalculator {
            base_spread_bps: config.base_spread_bps,
            min_spread_bps: config.min_spread_bps,
            max_spread_bps: config.max_spread_bps,
            volatility_multiplier: 10.0,
        });
        
        Self {
            config,
            market_state: Arc::new(RwLock::new(MarketState {
                bid: Decimal::ZERO,
                ask: Decimal::ZERO,
                mid: Decimal::ZERO,
                last_trade: Decimal::ZERO,
                volume_24h: Decimal::ZERO,
                volatility: 0.02,
                order_book_imbalance: 0.0,
                trade_flow: TradeFlow {
                    buy_volume: Decimal::ZERO,
                    sell_volume: Decimal::ZERO,
                    buy_count: 0,
                    sell_count: 0,
                    vwap: Decimal::ZERO,
                    momentum: 0.0,
                },
                last_update: Utc::now(),
            })),
            inventory_manager,
            spread_calculator,
            order_manager: Arc::new(OrderManager {
                active_orders: Arc::new(RwLock::new(Vec::new())),
                order_history: Arc::new(RwLock::new(Vec::new())),
                last_update: Arc::new(RwLock::new(Utc::now())),
            }),
            risk_manager: Arc::new(RiskManager {
                max_inventory: config.max_inventory,
                max_order_size: config.order_size * Decimal::from(5),
                min_spread_bps: config.min_spread_bps,
                adverse_selection_threshold: 0.6,
            }),
            metrics: Arc::new(RwLock::new(MarketMakingMetrics::default())),
        }
    }
    
    /// Main market making loop
    pub async fn run(&self) -> Result<(), MarketMakingError> {
        loop {
            // Update market state
            self.update_market_state().await?;
            
            // Check if quotes need update
            let market_state = self.market_state.read();
            if self.order_manager.needs_update(&market_state, self.config.update_threshold_bps) {
                // Cancel existing orders
                self.cancel_all_orders().await?;
                
                // Calculate new quotes
                let quotes = self.calculate_quotes(&market_state)?;
                
                // Validate and place orders
                for quote in quotes {
                    let current_inventory = *self.inventory_manager.current_inventory.read();
                    
                    if let Err(e) = self.risk_manager.validate_order(&quote, current_inventory) {
                        eprintln!("Risk check failed for quote: {:?}", e);
                        continue;
                    }
                    
                    self.place_order(quote).await?;
                }
                
                // Update metrics
                self.update_metrics();
            }
            
            // Sleep for minimum interval
            tokio::time::sleep(tokio::time::Duration::from_millis(self.config.min_update_interval_ms)).await;
        }
    }
    
    /// Calculate optimal quotes
    fn calculate_quotes(&self, market_state: &MarketState) -> Result<Vec<Quote>, MarketMakingError> {
        let current_inventory = *self.inventory_manager.current_inventory.read();
        
        // Calculate base spreads
        let (mut bid_spread, mut ask_spread) = self.spread_calculator.calculate_optimal_spread(
            market_state,
            current_inventory,
            300.0, // 5 minute horizon
        );
        
        // Apply inventory skew
        if self.config.enable_inventory_skew {
            let (bid_skew, ask_skew) = self.inventory_manager.calculate_skew(current_inventory);
            bid_spread = bid_spread * Decimal::from_f64(bid_skew).unwrap();
            ask_spread = ask_spread * Decimal::from_f64(ask_skew).unwrap();
        }
        
        // Generate quote ladder
        let quotes = self.order_manager.generate_quotes(
            market_state.mid,
            bid_spread,
            ask_spread,
            self.config.num_levels,
            self.config.order_size,
        );
        
        Ok(quotes)
    }
    
    async fn update_market_state(&self) -> Result<(), MarketMakingError> {
        // Placeholder - would fetch real market data
        Ok(())
    }
    
    async fn cancel_all_orders(&self) -> Result<(), MarketMakingError> {
        // Placeholder - would cancel orders via exchange API
        Ok(())
    }
    
    async fn place_order(&self, quote: Quote) -> Result<(), MarketMakingError> {
        // Placeholder - would place order via exchange API
        Ok(())
    }
    
    fn update_metrics(&self) {
        // Update performance metrics
        let mut metrics = self.metrics.write();
        // Calculate and update metrics
    }
}

/// Market making errors
#[derive(Debug, thiserror::Error)]
/// TODO: Add docs
pub enum MarketMakingError {
    #[error("Exchange error: {0}")]
    ExchangeError(String),
    
    #[error("Risk violation: {0:?}")]
    RiskViolation(RiskViolation),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

#[derive(Debug)]
/// TODO: Add docs
pub enum RiskViolation {
    InventoryLimitExceeded,
    OrderSizeTooLarge,
    SpreadTooTight,
    AdverseSelectionDetected,
}

#[derive(Debug, Clone, PartialEq)]
/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderStatus {
    New,
    Filled,
    PartiallyFilled,
    Cancelled,
}

use rust_decimal_macros::dec;
use std::str::FromStr;

// Drew: "Market making provides liquidity and captures the bid-ask spread!"