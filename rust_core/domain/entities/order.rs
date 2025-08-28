use domain_types::order::OrderError;
//! Module uses canonical Order type from domain_types
//! Avery: "Single source of truth for Order struct"

pub use domain_types::order::{
    Order, OrderId, OrderSide, OrderType, OrderStatus, TimeInForce,
    OrderError, Fill, FillId
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type OrderResult<T> = Result<T, OrderError>;

// Domain Entity: Order
// Mutable with identity, represents a trading order
// Owner: Sam | Reviewer: Quinn

use chrono::{DateTime, Utc};
use uuid::Uuid;
use anyhow::{Result, bail};
use std::fmt;

use crate::domain::value_objects::{Price, Quantity, Symbol};

/// Unique identifier for an Order

impl OrderId {
    pub fn new() -> Self {
        OrderId(Uuid::new_v4())
    }
    
    pub fn from_string(s: &str) -> Result<Self> {
        Ok(OrderId(Uuid::parse_str(s)?))
    }
    
    pub fn as_string(&self) -> String {
        self.0.to_string()
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Order type enumeration

pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    OCO,           // One-Cancels-Other
    ReduceOnly,    // Can only reduce position
    PostOnly,      // Maker only
}

/// Order side (direction)

pub enum OrderSide {
    Buy,
    Sell,
}

/// Time in Force for orders

pub enum TimeInForce {
    GTC,  // Good Till Canceled
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
    GTX,  // Good Till Crossing
}

/// Order status lifecycle

pub enum OrderStatus {
    Draft,           // Not yet submitted
    Pending,         // Submitted, awaiting confirmation
    Open,            // Active on exchange
    PartiallyFilled, // Partially executed
    Filled,          // Fully executed
    Cancelled,       // Cancelled by user
    Rejected,        // Rejected by exchange
    Expired,         // Time limit expired
}

/// Order entity - represents a trading order with full lifecycle
/// 
/// # Invariants
/// - Order must have valid price (for limit orders)
/// - Order must have positive quantity
/// - Status transitions must be valid
/// - Cannot modify filled or cancelled orders

impl Order {
    /// Create a new market order
    pub fn market(symbol: Symbol, side: OrderSide, quantity: Quantity) -> Self {
        let now = Utc::now();
        let id = OrderId::new();
        
        Order {
            id,
            symbol,
            order_type: OrderType::Market,
            side,
            price: None,
            stop_price: None,
            quantity,
            time_in_force: TimeInForce::IOC, // Market orders are IOC by default
            status: OrderStatus::Draft,
            filled_quantity: Quantity::zero(),
            average_fill_price: None,
            created_at: now,
            updated_at: now,
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: Some(50), // 0.5% default slippage
            exchange_order_id: None,
            client_order_id: format!("bot4_{}", id),
        }
    }
    
    /// Create a new limit order
    pub fn limit(
        symbol: Symbol,
        side: OrderSide,
        price: Price,
        quantity: Quantity,
        time_in_force: TimeInForce,
    ) -> Self {
        let now = Utc::now();
        let id = OrderId::new();
        
        Order {
            id,
            symbol,
            order_type: OrderType::Limit,
            side,
            price: Some(price),
            stop_price: None,
            quantity,
            time_in_force,
            status: OrderStatus::Draft,
            filled_quantity: Quantity::zero(),
            average_fill_price: None,
            created_at: now,
            updated_at: now,
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None, // No slippage for limit orders
            exchange_order_id: None,
            client_order_id: format!("bot4_{}", id),
        }
    }
    
    /// Create a new stop market order
    pub fn stop_market(
        symbol: Symbol,
        side: OrderSide,
        stop_price: Price,
        quantity: Quantity,
    ) -> Self {
        let now = Utc::now();
        let id = OrderId::new();
        
        Order {
            id,
            symbol,
            order_type: OrderType::StopMarket,
            side,
            price: None,
            stop_price: Some(stop_price),
            quantity,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Draft,
            filled_quantity: Quantity::zero(),
            average_fill_price: None,
            created_at: now,
            updated_at: now,
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: Some(100), // 1% default for stop orders
            exchange_order_id: None,
            client_order_id: format!("bot4_{}", id),
        }
    }
    
    /// Create a new stop limit order
    pub fn stop_limit(
        symbol: Symbol,
        side: OrderSide,
        stop_price: Price,
        limit_price: Price,
        quantity: Quantity,
        time_in_force: TimeInForce,
    ) -> Self {
        let now = Utc::now();
        let id = OrderId::new();
        
        Order {
            id,
            symbol,
            order_type: OrderType::StopLimit,
            side,
            price: Some(limit_price),
            stop_price: Some(stop_price),
            quantity,
            time_in_force,
            status: OrderStatus::Draft,
            filled_quantity: Quantity::zero(),
            average_fill_price: None,
            created_at: now,
            updated_at: now,
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None,
            exchange_order_id: None,
            client_order_id: format!("bot4_{}", id),
        }
    }
    
    // Builder methods for modifying orders
    
    /// Set the price (for amending orders)
    pub fn with_price(mut self, price: Price) -> Result<Self> {
        if !self.can_modify() {
            bail!("Cannot modify order in status: {:?}", self.status);
        }
        self.price = Some(price);
        self.updated_at = Utc::now();
        Ok(self)
    }
    
    /// Set the stop price (for amending stop orders)
    pub fn with_stop_price(mut self, stop_price: Price) -> Result<Self> {
        if !self.can_modify() {
            bail!("Cannot modify order in status: {:?}", self.status);
        }
        self.stop_price = Some(stop_price);
        self.updated_at = Utc::now();
        Ok(self)
    }
    
    /// Set the quantity (for amending orders)
    pub fn with_quantity(mut self, quantity: Quantity) -> Result<Self> {
        if !self.can_modify() {
            bail!("Cannot modify order in status: {:?}", self.status);
        }
        self.quantity = quantity;
        self.updated_at = Utc::now();
        Ok(self)
    }
    
    // Getters (immutable access)
    pub fn id(&self) -> &OrderId { &self.id }
    pub fn symbol(&self) -> &Symbol { &self.symbol }
    pub fn order_type(&self) -> OrderType { self.order_type }
    pub fn side(&self) -> OrderSide { self.side }
    pub fn price(&self) -> Option<&Price> { self.price.as_ref() }
    pub fn stop_price(&self) -> Option<&Price> { self.stop_price.as_ref() }
    pub fn quantity(&self) -> &Quantity { &self.quantity }
    pub fn status(&self) -> OrderStatus { self.status }
    pub fn filled_quantity(&self) -> &Quantity { &self.filled_quantity }
    pub fn remaining_quantity(&self) -> Result<Quantity> {
        self.quantity.subtract(&self.filled_quantity)
    }
    
    // Business logic methods
    
    /// Check if order can be cancelled
    pub fn can_cancel(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Draft | OrderStatus::Pending | OrderStatus::Open | OrderStatus::PartiallyFilled
        )
    }
    
    /// Check if order can be modified
    pub fn can_modify(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Draft | OrderStatus::Pending | OrderStatus::Open
        )
    }
    
    /// Check if order is terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected | OrderStatus::Expired
        )
    }
    
    /// Check if order is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::Open | OrderStatus::PartiallyFilled
        )
    }
    
    // State transitions (domain events)
    
    /// Submit order for execution
    pub fn submit(mut self) -> Result<(Order, OrderEvent)> {
        if self.status != OrderStatus::Draft {
            bail!("Can only submit draft orders, current status: {:?}", self.status);
        }
        
        self.status = OrderStatus::Pending;
        self.updated_at = Utc::now();
        
        let event = OrderEvent::Submitted {
            order_id: self.id,
            symbol: self.symbol.clone(),
            side: self.side,
            quantity: self.quantity.clone(),
            price: self.price.clone(),
            timestamp: self.updated_at,
        };
        
        Ok((self, event))
    }
    
    /// Mark order as confirmed by exchange
    pub fn confirm(mut self, exchange_order_id: String) -> Result<(Order, OrderEvent)> {
        if self.status != OrderStatus::Pending {
            bail!("Can only confirm pending orders, current status: {:?}", self.status);
        }
        
        self.status = OrderStatus::Open;
        self.exchange_order_id = Some(exchange_order_id.clone());
        self.updated_at = Utc::now();
        
        let event = OrderEvent::Confirmed {
            order_id: self.id,
            exchange_order_id,
            timestamp: self.updated_at,
        };
        
        Ok((self, event))
    }
    
    /// Record a partial or full fill
    pub fn fill(
        mut self,
        filled_quantity: Quantity,
        fill_price: Price,
    ) -> Result<(Order, OrderEvent)> {
        if !self.is_active() {
            bail!("Cannot fill inactive order, status: {:?}", self.status);
        }
        
        // Update filled quantity
        let new_filled = self.filled_quantity.add(&filled_quantity)?;
        
        // Check for overfill
        if new_filled.value() > self.quantity.value() {
            bail!(
                "Fill would exceed order quantity: {} + {} > {}",
                self.filled_quantity,
                filled_quantity,
                self.quantity
            );
        }
        
        // Update average fill price
        let prev_value = self.filled_quantity.value() * 
            self.average_fill_price.map(|p| p.value()).unwrap_or(0.0);
        let new_value = filled_quantity.value() * fill_price.value();
        let total_value = prev_value + new_value;
        let avg_price = Price::new(total_value / new_filled.value())?;
        
        self.filled_quantity = new_filled;
        self.average_fill_price = Some(avg_price);
        
        // Update status
        if self.filled_quantity == self.quantity {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
        
        self.updated_at = Utc::now();
        
        let event = OrderEvent::Filled {
            order_id: self.id,
            filled_quantity,
            fill_price,
            remaining_quantity: self.remaining_quantity()?,
            is_complete: self.status == OrderStatus::Filled,
            timestamp: self.updated_at,
        };
        
        Ok((self, event))
    }
    
    /// Cancel the order
    pub fn cancel(mut self, reason: String) -> Result<(Order, OrderEvent)> {
        if !self.can_cancel() {
            bail!("Cannot cancel order in status: {:?}", self.status);
        }
        
        self.status = OrderStatus::Cancelled;
        self.updated_at = Utc::now();
        
        let event = OrderEvent::Cancelled {
            order_id: self.id,
            reason,
            remaining_quantity: self.remaining_quantity()?,
            timestamp: self.updated_at,
        };
        
        Ok((self, event))
    }
    
    /// Reject the order
    pub fn reject(mut self, reason: String) -> Result<(Order, OrderEvent)> {
        if self.is_terminal() {
            bail!("Cannot reject order in terminal status: {:?}", self.status);
        }
        
        self.status = OrderStatus::Rejected;
        self.updated_at = Utc::now();
        
        let event = OrderEvent::Rejected {
            order_id: self.id,
            reason,
            timestamp: self.updated_at,
        };
        
        Ok((self, event))
    }
    
    /// Set risk parameters
    pub fn with_risk_params(
        mut self,
        stop_loss: Option<Price>,
        take_profit: Option<Price>,
    ) -> Result<Self> {
        if self.status != OrderStatus::Draft {
            bail!("Can only set risk params on draft orders");
        }
        
        // Validate stop loss
        if let Some(sl) = &stop_loss {
            match self.side {
                OrderSide::Buy => {
                    if let Some(price) = &self.price {
                        if sl.value() >= price.value() {
                            bail!("Buy stop loss must be below entry price");
                        }
                    }
                }
                OrderSide::Sell => {
                    if let Some(price) = &self.price {
                        if sl.value() <= price.value() {
                            bail!("Sell stop loss must be above entry price");
                        }
                    }
                }
            }
        }
        
        // Validate take profit
        if let Some(tp) = &take_profit {
            match self.side {
                OrderSide::Buy => {
                    if let Some(price) = &self.price {
                        if tp.value() <= price.value() {
                            bail!("Buy take profit must be above entry price");
                        }
                    }
                }
                OrderSide::Sell => {
                    if let Some(price) = &self.price {
                        if tp.value() >= price.value() {
                            bail!("Sell take profit must be below entry price");
                        }
                    }
                }
            }
        }
        
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self.updated_at = Utc::now();
        
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_symbol() -> Symbol {
        Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling")
    }
    
    fn create_test_price() -> Price {
        Price::new(50000.0).expect("SAFETY: Add proper error handling")
    }
    
    fn create_test_quantity() -> Quantity {
        Quantity::new(0.1).expect("SAFETY: Add proper error handling")
    }
    
    #[test]
    fn should_create_market_order() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            create_test_quantity(),
        );
        
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.side, OrderSide::Buy);
        assert!(order.price.is_none());
        assert_eq!(order.status, OrderStatus::Draft);
    }
    
    #[test]
    fn should_create_limit_order() {
        let order = Order::limit(
            create_test_symbol(),
            OrderSide::Sell,
            create_test_price(),
            create_test_quantity(),
            TimeInForce::GTC,
        );
        
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.side, OrderSide::Sell);
        assert!(order.price.is_some());
        assert_eq!(order.status, OrderStatus::Draft);
    }
    
    #[test]
    fn should_submit_draft_order() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            create_test_quantity(),
        );
        
        let (order, event) = order.submit().expect("SAFETY: Add proper error handling");
        assert_eq!(order.status, OrderStatus::Pending);
        assert!(matches!(event, OrderEvent::Submitted { .. }));
    }
    
    #[test]
    fn should_not_submit_non_draft_order() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            create_test_quantity(),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let result = order.submit();
        assert!(result.is_err());
    }
    
    #[test]
    fn should_confirm_pending_order() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            create_test_quantity(),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let (order, event) = order.confirm("EX123".to_string()).expect("SAFETY: Add proper error handling");
        
        assert_eq!(order.status, OrderStatus::Open);
        assert_eq!(order.exchange_order_id, Some("EX123".to_string()));
        assert!(matches!(event, OrderEvent::Confirmed { .. }));
    }
    
    #[test]
    fn should_fill_order_partially() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let (order, _) = order.confirm("EX123".to_string()).expect("SAFETY: Add proper error handling");
        
        let fill_qty = Quantity::new(0.3).expect("SAFETY: Add proper error handling");
        let fill_price = Price::new(50000.0).expect("SAFETY: Add proper error handling");
        
        let (order, event) = order.fill(fill_qty, fill_price).expect("SAFETY: Add proper error handling");
        
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.filled_quantity.value(), 0.3);
        assert_eq!(order.average_fill_price.expect("SAFETY: Add proper error handling").value(), 50000.0);
        
        if let OrderEvent::Filled { is_complete, .. } = event {
            assert!(!is_complete);
        } else {
            panic!("Expected Filled event");
        }
    }
    
    #[test]
    fn should_fill_order_completely() {
        let qty = Quantity::new(1.0).expect("SAFETY: Add proper error handling");
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            qty.clone(),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let (order, _) = order.confirm("EX123".to_string()).expect("SAFETY: Add proper error handling");
        
        let fill_price = Price::new(50000.0).expect("SAFETY: Add proper error handling");
        
        let (order, event) = order.fill(qty, fill_price).expect("SAFETY: Add proper error handling");
        
        assert_eq!(order.status, OrderStatus::Filled);
        assert_eq!(order.filled_quantity.value(), 1.0);
        
        if let OrderEvent::Filled { is_complete, .. } = event {
            assert!(is_complete);
        } else {
            panic!("Expected Filled event");
        }
    }
    
    #[test]
    fn should_cancel_open_order() {
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            create_test_quantity(),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let (order, _) = order.confirm("EX123".to_string()).expect("SAFETY: Add proper error handling");
        
        assert!(order.can_cancel());
        
        let (order, event) = order.cancel("User requested".to_string()).expect("SAFETY: Add proper error handling");
        
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert!(matches!(event, OrderEvent::Cancelled { .. }));
    }
    
    #[test]
    fn should_not_cancel_filled_order() {
        let qty = Quantity::new(1.0).expect("SAFETY: Add proper error handling");
        let order = Order::market(
            create_test_symbol(),
            OrderSide::Buy,
            qty.clone(),
        );
        
        let (order, _) = order.submit().expect("SAFETY: Add proper error handling");
        let (order, _) = order.confirm("EX123".to_string()).expect("SAFETY: Add proper error handling");
        let (order, _) = order.fill(qty, create_test_price()).expect("SAFETY: Add proper error handling");
        
        assert!(!order.can_cancel());
        
        let result = order.cancel("Test".to_string());
        assert!(result.is_err());
    }
    
    #[test]
    fn should_set_risk_params_on_buy_order() {
        let entry_price = Price::new(50000.0).expect("SAFETY: Add proper error handling");
        let stop_loss = Price::new(49000.0).expect("SAFETY: Add proper error handling");
        let take_profit = Price::new(51000.0).expect("SAFETY: Add proper error handling");
        
        let order = Order::limit(
            create_test_symbol(),
            OrderSide::Buy,
            entry_price,
            create_test_quantity(),
            TimeInForce::GTC,
        );
        
        let order = order.with_risk_params(Some(stop_loss), Some(take_profit)).expect("SAFETY: Add proper error handling");
        
        assert_eq!(order.stop_loss, Some(stop_loss));
        assert_eq!(order.take_profit, Some(take_profit));
    }
    
    #[test]
    fn should_reject_invalid_buy_stop_loss() {
        let entry_price = Price::new(50000.0).expect("SAFETY: Add proper error handling");
        let invalid_stop = Price::new(51000.0).expect("SAFETY: Add proper error handling"); // Above entry
        
        let order = Order::limit(
            create_test_symbol(),
            OrderSide::Buy,
            entry_price,
            create_test_quantity(),
            TimeInForce::GTC,
        );
        
        let result = order.with_risk_params(Some(invalid_stop), None);
        assert!(result.is_err());
    }
}
