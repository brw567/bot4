// Order types and structures
// Designed for exchange-agnostic order handling with precise decimal math

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;
use strum_macros::{Display, EnumString};

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub Uuid);

impl Default for OrderId {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn opposite(&self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
    TakeProfit,
    TakeProfitLimit,
}

impl OrderType {
    pub fn requires_price(&self) -> bool {
        matches!(
            self,
            OrderType::Limit | OrderType::StopLimit | OrderType::TakeProfitLimit
        )
    }
    
    pub fn requires_stop_price(&self) -> bool {
        matches!(
            self,
            OrderType::StopLoss | OrderType::StopLimit | OrderType::TakeProfit | OrderType::TakeProfitLimit
        )
    }
}

/// Time in force for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
pub enum TimeInForce {
    /// Good Till Cancelled
    GTC,
    /// Immediate Or Cancel
    IOC,
    /// Fill Or Kill
    FOK,
    /// Good Till Date
    GTD,
    /// Post Only (maker only)
    PostOnly,
}

/// Complete order structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    // Identity
    pub id: OrderId,
    pub client_order_id: String,
    pub exchange_order_id: Option<String>,
    
    // Core parameters
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub time_in_force: TimeInForce,
    
    // Risk management (Quinn's requirements)
    pub stop_loss_price: Option<Decimal>,
    pub take_profit_price: Option<Decimal>,
    pub position_size_pct: Decimal,  // % of portfolio
    pub max_loss_amount: Decimal,    // Max acceptable loss
    
    // Execution details
    pub filled_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub commission: Decimal,
    pub commission_asset: Option<String>,
    
    // Strategy metadata
    pub strategy_id: Option<String>,
    pub signal_strength: Option<Decimal>,
    pub ml_confidence: Option<Decimal>,
    pub ta_score: Option<Decimal>,
    
    // Timestamps
    pub created_at: DateTime<Utc>,
    pub submitted_at: Option<DateTime<Utc>>,
    pub filled_at: Option<DateTime<Utc>>,
    pub cancelled_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    
    // Performance tracking
    pub submission_latency_us: Option<u64>,
    pub fill_latency_us: Option<u64>,
}

impl Order {
    pub fn new(
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: OrderId::new(),
            client_order_id: Uuid::new_v4().to_string(),
            exchange_order_id: None,
            symbol,
            side,
            order_type,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            stop_loss_price: None,
            take_profit_price: None,
            position_size_pct: Decimal::ZERO,
            max_loss_amount: Decimal::ZERO,
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: Decimal::ZERO,
            commission_asset: None,
            strategy_id: None,
            signal_strength: None,
            ml_confidence: None,
            ta_score: None,
            created_at: now,
            submitted_at: None,
            filled_at: None,
            cancelled_at: None,
            updated_at: now,
            submission_latency_us: None,
            fill_latency_us: None,
        }
    }
    
    pub fn with_price(mut self, price: Decimal) -> Self {
        self.price = Some(price);
        self
    }
    
    pub fn with_stop_price(mut self, stop_price: Decimal) -> Self {
        self.stop_price = Some(stop_price);
        self
    }
    
    pub fn with_stop_loss(mut self, stop_loss_price: Decimal) -> Self {
        self.stop_loss_price = Some(stop_loss_price);
        self
    }
    
    pub fn with_take_profit(mut self, take_profit_price: Decimal) -> Self {
        self.take_profit_price = Some(take_profit_price);
        self
    }
    
    pub fn with_strategy(mut self, strategy_id: String) -> Self {
        self.strategy_id = Some(strategy_id);
        self
    }
    
    pub fn with_risk_params(mut self, position_size_pct: Decimal, max_loss: Decimal) -> Self {
        self.position_size_pct = position_size_pct;
        self.max_loss_amount = max_loss;
        self
    }
    
    /// Check if order is completely filled
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }
    
    /// Check if order is partially filled
    pub fn is_partially_filled(&self) -> bool {
        self.filled_quantity > Decimal::ZERO && self.filled_quantity < self.quantity
    }
    
    /// Get remaining quantity to fill
    pub fn remaining_quantity(&self) -> Decimal {
        self.quantity - self.filled_quantity
    }
    
    /// Calculate fill percentage
    pub fn fill_percentage(&self) -> Decimal {
        if self.quantity.is_zero() {
            return Decimal::ZERO;
        }
        (self.filled_quantity / self.quantity) * Decimal::from(100)
    }
    
    /// Validate order parameters
    pub fn validate(&self) -> Result<(), OrderValidationError> {
        // Check quantity
        if self.quantity <= Decimal::ZERO {
            return Err(OrderValidationError::InvalidQuantity(
                "Quantity must be positive".to_string()
            ));
        }
        
        // Check price for limit orders
        if self.order_type.requires_price() && self.price.is_none() {
            return Err(OrderValidationError::MissingPrice);
        }
        
        if let Some(price) = self.price {
            if price <= Decimal::ZERO {
                return Err(OrderValidationError::InvalidPrice(
                    "Price must be positive".to_string()
                ));
            }
        }
        
        // Check stop price for stop orders
        if self.order_type.requires_stop_price() && self.stop_price.is_none() {
            return Err(OrderValidationError::MissingStopPrice);
        }
        
        // Check position size (Quinn's 2% rule)
        if self.position_size_pct > Decimal::from_str_exact("0.02").unwrap() {
            return Err(OrderValidationError::PositionSizeTooLarge(
                format!("Position size {}% exceeds 2% limit", self.position_size_pct * Decimal::from(100))
            ));
        }
        
        // Check stop loss is set for non-market orders (Quinn's requirement)
        if self.order_type != OrderType::Market && self.stop_loss_price.is_none() {
            return Err(OrderValidationError::MissingStopLoss);
        }
        
        Ok(())
    }
}


#[derive(Debug, thiserror::Error)]
pub enum OrderValidationError {
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(String),
    
    #[error("Invalid price: {0}")]
    InvalidPrice(String),
    
    #[error("Missing price for limit order")]
    MissingPrice,
    
    #[error("Missing stop price for stop order")]
    MissingStopPrice,
    
    #[error("Missing stop loss (required by risk management)")]
    MissingStopLoss,
    
    #[error("Position size too large: {0}")]
    PositionSizeTooLarge(String),
}

/// Order update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub order_id: OrderId,
    pub exchange_order_id: Option<String>,
    pub filled_quantity: Decimal,
    pub fill_price: Decimal,
    pub commission: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Order fill information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFill {
    pub order_id: OrderId,
    pub fill_id: String,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    pub commission_asset: String,
    pub timestamp: DateTime<Utc>,
    pub is_maker: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.01),
        )
        .with_price(dec!(50000))
        .with_stop_loss(dec!(49000));
        
        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, dec!(0.01));
        assert_eq!(order.price, Some(dec!(50000)));
        assert_eq!(order.stop_loss_price, Some(dec!(49000)));
    }
    
    #[test]
    fn test_order_validation() {
        // Valid order
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.01),
        )
        .with_price(dec!(50000))
        .with_stop_loss(dec!(49000))
        .with_risk_params(dec!(0.01), dec!(100));
        
        assert!(order.validate().is_ok());
        
        // Invalid: no price for limit order
        let mut bad_order = order.clone();
        bad_order.price = None;
        assert!(bad_order.validate().is_err());
        
        // Invalid: position size too large
        let mut bad_order = order.clone();
        bad_order.position_size_pct = dec!(0.03);
        assert!(bad_order.validate().is_err());
        
        // Invalid: no stop loss
        let mut bad_order = order.clone();
        bad_order.stop_loss_price = None;
        assert!(bad_order.validate().is_err());
    }
    
    #[test]
    fn test_fill_calculations() {
        let mut order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            dec!(1.0),
        );
        
        assert!(!order.is_filled());
        assert!(!order.is_partially_filled());
        assert_eq!(order.remaining_quantity(), dec!(1.0));
        assert_eq!(order.fill_percentage(), dec!(0));
        
        order.filled_quantity = dec!(0.3);
        assert!(!order.is_filled());
        assert!(order.is_partially_filled());
        assert_eq!(order.remaining_quantity(), dec!(0.7));
        assert_eq!(order.fill_percentage(), dec!(30));
        
        order.filled_quantity = dec!(1.0);
        assert!(order.is_filled());
        assert!(!order.is_partially_filled());
        assert_eq!(order.remaining_quantity(), dec!(0));
        assert_eq!(order.fill_percentage(), dec!(100));
    }
}