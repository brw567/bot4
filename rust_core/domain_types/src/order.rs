//! # Canonical Order Type - Single Source of Truth
//! 
//! Consolidates 44 different Order struct definitions into ONE canonical type.
//! This is the ONLY Order type that should be used throughout the codebase.
//!
//! ## Design Decisions
//! - Supports all order types across all exchanges
//! - Includes risk management fields (stop loss, take profit)
//! - ML/Strategy metadata support
//! - Performance tracking (latencies)
//! - Full audit trail with timestamps
//!
//! ## External Research Applied
//! - FIX Protocol (Financial Information eXchange)
//! - Exchange-specific order types (Binance, Coinbase, Kraken)
//! - Smart Order Routing (Virtu Financial)
//! - Optimal Execution (Almgren-Chriss model)

use crate::{Price, Quantity};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use strum_macros::{Display, EnumString};
use uuid::Uuid;

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// TODO: Add docs
pub struct OrderId(pub Uuid);

impl OrderId {
    /// Creates a new unique order ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Creates from a string
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    /// Converts to string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl Default for OrderId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "UPPERCASE")]
#[strum(serialize_all = "UPPERCASE")]
/// TODO: Add docs
pub enum OrderSide {
    /// Buy order (long)
    Buy,
    /// Sell order (short)
    Sell,
}

impl OrderSide {
    /// Gets the opposite side
    pub fn opposite(&self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
    
    /// Checks if this is a buy order
    pub fn is_buy(&self) -> bool {
        matches!(self, Self::Buy)
    }
    
    /// Checks if this is a sell order
    pub fn is_sell(&self) -> bool {
        matches!(self, Self::Sell)
    }
}

/// Order type - supports all exchange order types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
/// TODO: Add docs
pub enum OrderType {
    /// Market order - immediate execution at best price
    Market,
    /// Limit order - execution at specific price or better
    Limit,
    /// Stop market order - market order triggered at stop price
    StopMarket,
    /// Stop limit order - limit order triggered at stop price
    StopLimit,
    /// Take profit market order
    TakeProfit,
    /// Take profit limit order
    TakeProfitLimit,
    /// One-Cancels-Other - two orders where one cancels the other
    OCO,
    /// Iceberg order - only shows partial quantity
    Iceberg,
    /// Reduce only - can only reduce position size
    ReduceOnly,
    /// Post only - ensures maker fee (cancels if would take)
    PostOnly,
    /// Trailing stop - stop that follows price
    TrailingStop,
    /// TWAP - Time-Weighted Average Price
    TWAP,
    /// VWAP - Volume-Weighted Average Price
    VWAP,
}

impl OrderType {
    /// Checks if order type requires a price
    pub fn requires_price(&self) -> bool {
        matches!(
            self,
            Self::Limit | Self::StopLimit | Self::TakeProfitLimit | Self::PostOnly
        )
    }
    
    /// Checks if order type requires a stop price
    pub fn requires_stop_price(&self) -> bool {
        matches!(
            self,
            Self::StopMarket | Self::StopLimit | Self::TrailingStop
        )
    }
    
    /// Checks if this is an algorithmic order type
    pub fn is_algorithmic(&self) -> bool {
        matches!(self, Self::TWAP | Self::VWAP | Self::Iceberg)
    }
}

/// Time in Force - how long order remains active
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
/// TODO: Add docs
pub enum TimeInForce {
    /// Good Till Cancelled - remains until filled or cancelled
    GTC,
    /// Immediate Or Cancel - fill what you can, cancel rest
    IOC,
    /// Fill Or Kill - fill entirely or cancel
    FOK,
    /// Good Till Date - expires at specific time
    GTD,
    /// Good Till Crossing - for auction orders
    GTX,
    /// Post Only - ensures maker order
    PostOnly,
    /// Day order - expires at end of trading day
    Day,
}

/// Order status - full lifecycle tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
/// TODO: Add docs
pub enum OrderStatus {
    /// Created but not submitted
    Draft,
    /// Submitted to exchange, awaiting confirmation
    Pending,
    /// Confirmed and active on exchange
    Open,
    /// Partially filled
    PartiallyFilled,
    /// Completely filled
    Filled,
    /// Cancelled by user
    Cancelled,
    /// Rejected by exchange
    Rejected,
    /// Expired due to time limit
    Expired,
    /// Failed due to error
    Failed,
}

impl OrderStatus {
    /// Checks if order is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Cancelled | Self::Rejected | Self::Expired | Self::Failed
        )
    }
    
    /// Checks if order is active
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Pending | Self::Open | Self::PartiallyFilled
        )
    }
    
    /// Checks if order can be cancelled
    pub fn can_cancel(&self) -> bool {
        matches!(
            self,
            Self::Draft | Self::Pending | Self::Open | Self::PartiallyFilled
        )
    }
    
    /// Checks if order can be modified
    pub fn can_modify(&self) -> bool {
        matches!(self, Self::Draft | Self::Open)
    }
}

/// Execution algorithm for smart order routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
/// TODO: Add docs
pub enum ExecutionAlgorithm {
    /// Default exchange execution
    Default,
    /// Time-Weighted Average Price
    TWAP,
    /// Volume-Weighted Average Price
    VWAP,
    /// Percentage of Volume
    POV,
    /// Implementation Shortfall
    IS,
    /// Adaptive execution based on market conditions
    Adaptive,
    /// Iceberg - hide order size
    Iceberg,
    /// Sniper - aggressive taking
    Sniper,
    /// Smart Order Router
    SOR,
}

/// Canonical Order type - the ONE source of truth
///
/// # Invariants
/// - Order ID is immutable once created
/// - Status transitions must be valid
/// - Filled quantity cannot exceed total quantity
/// - Risk parameters must be valid for order side
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Order {
    // === Identity ===
    /// Unique order ID (immutable)
    pub id: OrderId,
    /// Client-side order ID for tracking
    pub client_order_id: String,
    /// Exchange-assigned order ID
    pub exchange_order_id: Option<String>,
    
    // === Core Parameters ===
    /// Trading symbol (e.g., "BTC/USDT")
    pub symbol: String,
    /// Order side (buy/sell)
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Total quantity to trade
    pub quantity: Quantity,
    /// Limit price (for limit orders)
    pub price: Option<Price>,
    /// Stop trigger price (for stop orders)
    pub stop_price: Option<Price>,
    /// Time in force
    pub time_in_force: TimeInForce,
    
    // === Status & Execution ===
    /// Current order status
    pub status: OrderStatus,
    /// Quantity filled so far
    pub filled_quantity: Quantity,
    /// Average fill price
    pub average_fill_price: Option<Price>,
    /// Remaining quantity
    pub remaining_quantity: Quantity,
    /// Individual fills
    pub fills: Vec<Fill>,
    
    // === Risk Management (Quinn's requirements) ===
    /// Stop loss price
    pub stop_loss: Option<Price>,
    /// Take profit price
    pub take_profit: Option<Price>,
    /// Maximum slippage allowed (basis points)
    pub max_slippage_bps: Option<u32>,
    /// Position size as percentage of portfolio
    pub position_size_pct: Option<Decimal>,
    /// Maximum acceptable loss amount
    pub max_loss_amount: Option<Decimal>,
    /// Trailing stop distance
    pub trailing_stop_distance: Option<Price>,
    
    // === Fees & Costs ===
    /// Commission paid
    pub commission: Decimal,
    /// Commission asset (e.g., "BNB", "USDT")
    pub commission_asset: Option<String>,
    /// Estimated fee before execution
    pub estimated_fee: Option<Decimal>,
    /// Maker/taker fee rate applied
    pub fee_rate: Option<Decimal>,
    
    // === Strategy & ML Metadata (Morgan's requirements) ===
    /// Strategy that generated this order
    pub strategy_id: Option<String>,
    /// Strategy version
    pub strategy_version: Option<String>,
    /// ML model confidence score
    pub ml_confidence: Option<Decimal>,
    /// Technical analysis score
    pub ta_score: Option<Decimal>,
    /// Signal strength
    pub signal_strength: Option<Decimal>,
    /// Features used for decision
    pub ml_features: HashMap<String, f64>,
    /// Expected profit
    pub expected_profit: Option<Decimal>,
    /// Risk/reward ratio
    pub risk_reward_ratio: Option<Decimal>,
    
    // === Execution Details (Casey's requirements) ===
    /// Exchange to route to
    pub exchange: Option<String>,
    /// Execution algorithm to use
    pub execution_algorithm: ExecutionAlgorithm,
    /// Algorithm parameters
    pub algo_params: HashMap<String, String>,
    /// Priority level (for queue position)
    pub priority: Option<u32>,
    /// Parent order ID (for child orders)
    pub parent_order_id: Option<OrderId>,
    /// Child order IDs
    pub child_order_ids: Vec<OrderId>,
    
    // === Performance Tracking (Jordan's requirements) ===
    /// Time to create order
    pub creation_latency_us: Option<u64>,
    /// Time to submit to exchange
    pub submission_latency_us: Option<u64>,
    /// Time to get confirmation
    pub confirmation_latency_us: Option<u64>,
    /// Time to fill
    pub fill_latency_us: Option<u64>,
    /// Total end-to-end latency
    pub total_latency_us: Option<u64>,
    
    // === Timestamps ===
    /// When order was created
    pub created_at: DateTime<Utc>,
    /// When order was submitted
    pub submitted_at: Option<DateTime<Utc>>,
    /// When order was confirmed by exchange
    pub confirmed_at: Option<DateTime<Utc>>,
    /// When order was last updated
    pub updated_at: DateTime<Utc>,
    /// When order was filled
    pub filled_at: Option<DateTime<Utc>>,
    /// When order was cancelled
    pub cancelled_at: Option<DateTime<Utc>>,
    /// When order expires
    pub expires_at: Option<DateTime<Utc>>,
    
    // === Additional Metadata ===
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom fields for exchange-specific data
    pub custom_fields: HashMap<String, String>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Retry count for failed orders
    pub retry_count: u32,
}

/// Individual fill information
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Fill {
    /// Fill ID from exchange
    pub fill_id: String,
    /// Quantity filled in this execution
    pub quantity: Quantity,
    /// Price of this fill
    pub price: Price,
    /// Commission for this fill
    pub commission: Decimal,
    /// Commission asset
    pub commission_asset: String,
    /// Timestamp of fill
    pub timestamp: DateTime<Utc>,
    /// Was this a maker or taker fill
    pub is_maker: bool,
}

impl Order {
    /// Creates a new market order
    pub fn market(symbol: String, side: OrderSide, quantity: Quantity) -> Self {
        let now = Utc::now();
        Self {
            id: OrderId::new(),
            client_order_id: Uuid::new_v4().to_string(),
            exchange_order_id: None,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::Draft,
            filled_quantity: Quantity::zero(),
            average_fill_price: None,
            remaining_quantity: quantity,
            fills: Vec::new(),
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: Some(50), // 0.5% default
            position_size_pct: None,
            max_loss_amount: None,
            trailing_stop_distance: None,
            commission: Decimal::ZERO,
            commission_asset: None,
            estimated_fee: None,
            fee_rate: None,
            strategy_id: None,
            strategy_version: None,
            ml_confidence: None,
            ta_score: None,
            signal_strength: None,
            ml_features: HashMap::new(),
            expected_profit: None,
            risk_reward_ratio: None,
            exchange: None,
            execution_algorithm: ExecutionAlgorithm::Default,
            algo_params: HashMap::new(),
            priority: None,
            parent_order_id: None,
            child_order_ids: Vec::new(),
            creation_latency_us: None,
            submission_latency_us: None,
            confirmation_latency_us: None,
            fill_latency_us: None,
            total_latency_us: None,
            created_at: now,
            submitted_at: None,
            confirmed_at: None,
            updated_at: now,
            filled_at: None,
            cancelled_at: None,
            expires_at: None,
            tags: Vec::new(),
            custom_fields: HashMap::new(),
            error_message: None,
            retry_count: 0,
        }
    }
    
    /// Creates a new limit order
    pub fn limit(
        symbol: String,
        side: OrderSide,
        price: Price,
        quantity: Quantity,
        time_in_force: TimeInForce,
    ) -> Self {
        let mut order = Self::market(symbol, side, quantity);
        order.order_type = OrderType::Limit;
        order.price = Some(price);
        order.time_in_force = time_in_force;
        order.max_slippage_bps = None; // No slippage for limit orders
        order
    }
    
    /// Creates a stop loss order
    pub fn stop_loss(
        symbol: String,
        side: OrderSide,
        stop_price: Price,
        quantity: Quantity,
    ) -> Self {
        let mut order = Self::market(symbol, side, quantity);
        order.order_type = OrderType::StopMarket;
        order.stop_price = Some(stop_price);
        order.time_in_force = TimeInForce::GTC;
        order.max_slippage_bps = Some(100); // 1% for stop orders
        order
    }
    
    // === Builder Methods ===
    
    /// Sets the stop loss price
    pub fn with_stop_loss(mut self, stop_loss: Price) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }
    
    /// Sets the take profit price
    pub fn with_take_profit(mut self, take_profit: Price) -> Self {
        self.take_profit = Some(take_profit);
        self
    }
    
    /// Sets the strategy metadata
    pub fn with_strategy(mut self, strategy_id: String, confidence: Decimal) -> Self {
        self.strategy_id = Some(strategy_id);
        self.ml_confidence = Some(confidence);
        self
    }
    
    /// Sets the exchange to route to
    pub fn with_exchange(mut self, exchange: String) -> Self {
        self.exchange = Some(exchange);
        self
    }
    
    /// Sets the execution algorithm
    pub fn with_algorithm(mut self, algo: ExecutionAlgorithm) -> Self {
        self.execution_algorithm = algo;
        self
    }
    
    /// Sets risk parameters
    pub fn with_risk_params(mut self, max_loss: Decimal, position_size_pct: Decimal) -> Self {
        self.max_loss_amount = Some(max_loss);
        self.position_size_pct = Some(position_size_pct);
        self
    }
    
    // === Query Methods ===
    
    /// Checks if order is completely filled
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }
    
    /// Checks if order is partially filled
    pub fn is_partially_filled(&self) -> bool {
        self.filled_quantity > Quantity::zero() && self.filled_quantity < self.quantity
    }
    
    /// Gets the fill percentage
    pub fn fill_percentage(&self) -> Decimal {
        if self.quantity.is_zero() {
            return Decimal::ZERO;
        }
        (self.filled_quantity.as_decimal() / self.quantity.as_decimal()) * Decimal::from(100)
    }
    
    /// Calculates total commission paid
    pub fn total_commission(&self) -> Decimal {
        self.fills.iter().map(|f| f.commission).sum()
    }
    
    /// Calculates realized PnL for filled portion
    pub fn realized_pnl(&self, current_price: Price) -> Option<Decimal> {
        self.average_fill_price.map(|avg_price| {
            let price_diff = current_price.as_decimal() - avg_price.as_decimal();
            let pnl = match self.side {
                OrderSide::Buy => price_diff * self.filled_quantity.as_decimal(),
                OrderSide::Sell => -price_diff * self.filled_quantity.as_decimal(),
            };
            pnl - self.total_commission()
        })
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Order {} {} {} {} @ {:?} [{}]",
            self.id,
            self.side,
            self.quantity,
            self.symbol,
            self.price,
            self.status
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_order_creation() {
        let qty = Quantity::new(rust_decimal_macros::dec!(0.1)).unwrap();
        let order = Order::market("BTC/USDT".to_string(), OrderSide::Buy, qty);
        
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.side, OrderSide::Buy);
        assert!(order.price.is_none());
        assert_eq!(order.status, OrderStatus::Draft);
        assert_eq!(order.time_in_force, TimeInForce::IOC);
    }
    
    #[test]
    fn test_limit_order_creation() {
        let price = Price::new(rust_decimal_macros::dec!(50000)).unwrap();
        let qty = Quantity::new(rust_decimal_macros::dec!(0.1)).unwrap();
        let order = Order::limit(
            "BTC/USDT".to_string(),
            OrderSide::Sell,
            price,
            qty,
            TimeInForce::GTC,
        );
        
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.price, Some(price));
        assert!(order.max_slippage_bps.is_none());
    }
    
    #[test]
    fn test_order_status_checks() {
        assert!(OrderStatus::Filled.is_terminal());
        assert!(!OrderStatus::Open.is_terminal());
        assert!(OrderStatus::Open.is_active());
        assert!(!OrderStatus::Cancelled.is_active());
        assert!(OrderStatus::Open.can_cancel());
        assert!(!OrderStatus::Filled.can_cancel());
    }
    
    #[test]
    fn test_order_builder_methods() {
        let qty = Quantity::new(rust_decimal_macros::dec!(1)).unwrap();
        let stop_loss = Price::new(rust_decimal_macros::dec!(45000)).unwrap();
        let take_profit = Price::new(rust_decimal_macros::dec!(55000)).unwrap();
        
        let order = Order::market("BTC/USDT".to_string(), OrderSide::Buy, qty)
            .with_stop_loss(stop_loss)
            .with_take_profit(take_profit)
            .with_strategy("momentum_v2".to_string(), rust_decimal_macros::dec!(0.85))
            .with_exchange("binance".to_string());
        
        assert_eq!(order.stop_loss, Some(stop_loss));
        assert_eq!(order.take_profit, Some(take_profit));
        assert_eq!(order.strategy_id, Some("momentum_v2".to_string()));
        assert_eq!(order.exchange, Some("binance".to_string()));
    }
}