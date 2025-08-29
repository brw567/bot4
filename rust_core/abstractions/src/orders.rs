use domain_types::ValidationResult;
//! # Order Abstractions (Layer 2)
//!
//! Order-related abstractions that lower layers can use without
//! depending on the order_management crate (Layer 5).

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use domain_types::{Price, Quantity};
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::AbstractionResult;

/// Abstract order representation for lower layers
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AbstractOrder {
    /// Order ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Side (buy/sell)
    pub side: OrderSide,
    /// Quantity
    pub quantity: Quantity,
    /// Price (if limit order)
    pub price: Option<Price>,
    /// Order type
    pub order_type: OrderType,
    /// Status
    pub status: OrderStatus,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Order side abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

/// Order type abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderType {
    /// Market order
    Market,
    /// Limit order
    Limit,
    /// Stop order
    Stop,
    /// Stop limit order
    StopLimit,
}

/// Order status abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderStatus {
    /// New order
    New,
    /// Partially filled
    PartiallyFilled,
    /// Fully filled
    Filled,
    /// Cancelled
    Cancelled,
    /// Rejected
    Rejected,
}

/// Abstract position representation
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AbstractPosition {
    /// Position ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Quantity (positive for long, negative for short)
    pub quantity: Decimal,
    /// Average entry price
    pub avg_entry_price: Price,
    /// Current value
    pub current_value: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Realized P&L
    pub realized_pnl: Decimal,
    /// Last update time
    pub last_update: DateTime<Utc>,
}

/// Abstract fill representation
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AbstractFill {
    /// Fill ID
    pub id: String,
    /// Order ID
    pub order_id: String,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Fill price
    pub price: Price,
    /// Fill quantity
    pub quantity: Quantity,
    /// Fee
    pub fee: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Order validation trait for risk checks
#[async_trait]
pub trait OrderValidator: Send + Sync {
    /// Validate order before submission
    async fn validate(
        &self,
        order: &AbstractOrder,
    ) -> AbstractionResult<ValidationResult>;
    
    /// Pre-trade risk check
    async fn pre_trade_check(
        &self,
        order: &AbstractOrder,
    ) -> AbstractionResult<bool>;
    
    /// Post-trade risk check
    async fn post_trade_check(
        &self,
        fill: &AbstractFill,
    ) -> AbstractionResult<()>;
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
// ELIMINATED: use domain_types::ValidationResult
// pub struct ValidationResult {
    /// Is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Risk score
    pub risk_score: f64,
}

/// Position tracker abstraction
#[async_trait]
pub trait PositionTracker: Send + Sync {
    /// Get position for symbol
    async fn get_position(
        &self,
        symbol: &str,
    ) -> AbstractionResult<Option<AbstractPosition>>;
    
    /// Get all positions
    async fn get_all_positions(&self) -> AbstractionResult<Vec<AbstractPosition>>;
    
    /// Update position from fill
    async fn update_from_fill(
        &self,
        fill: &AbstractFill,
    ) -> AbstractionResult<AbstractPosition>;
    
    /// Get total exposure
    async fn get_total_exposure(&self) -> AbstractionResult<Decimal>;
    
    /// Get P&L
    async fn get_pnl(&self) -> AbstractionResult<(Decimal, Decimal)>;
}

/// Order events for event-driven architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderEvent {
    /// Order placed
    OrderPlaced(AbstractOrder),
    /// Order filled
    OrderFilled(AbstractFill),
    /// Order cancelled
    OrderCancelled(String),
    /// Order rejected
    OrderRejected(String, String), // (order_id, reason)
    /// Position updated
    PositionUpdated(AbstractPosition),
}

/// Convert from domain types
impl From<domain_types::OrderSide> for OrderSide {
    fn from(side: domain_types::OrderSide) -> Self {
        match side {
            domain_types::OrderSide::Buy => OrderSide::Buy,
            domain_types::OrderSide::Sell => OrderSide::Sell,
        }
    }
}

impl From<domain_types::OrderType> for OrderType {
    fn from(order_type: domain_types::OrderType) -> Self {
        match order_type {
            domain_types::OrderType::Market => OrderType::Market,
            domain_types::OrderType::Limit => OrderType::Limit,
            domain_types::OrderType::StopMarket | domain_types::OrderType::StopLimit => OrderType::Stop,
            domain_types::OrderType::TakeProfit => OrderType::Limit,
            _ => OrderType::Limit, // Default to limit for other types
        }
    }
}

impl From<domain_types::OrderStatus> for OrderStatus {
    fn from(status: domain_types::OrderStatus) -> Self {
        match status {
            domain_types::OrderStatus::Draft | domain_types::OrderStatus::Pending => OrderStatus::New,
            domain_types::OrderStatus::Open => OrderStatus::New,
            domain_types::OrderStatus::PartiallyFilled => OrderStatus::PartiallyFilled,
            domain_types::OrderStatus::Filled => OrderStatus::Filled,
            domain_types::OrderStatus::Cancelled => OrderStatus::Cancelled,
            domain_types::OrderStatus::Rejected | domain_types::OrderStatus::Failed => OrderStatus::Rejected,
            domain_types::OrderStatus::Expired => OrderStatus::Cancelled,
        }
    }
}