// OCO Order Entity - One-Cancels-Other Implementation
// Addresses Sophia's #2 critical feedback on OCO semantics
// Owner: Casey | Reviewer: Quinn

use anyhow::{Result, bail};
use crate::domain::entities::{Order, OrderId, OrderStatus, OrderType};
use crate::domain::value_objects::{Price, Quantity, Symbol};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

/// OCO Priority determines which order executes if both trigger simultaneously
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum OcoPriority {
    /// Limit order takes precedence
    LimitFirst,
    /// Stop order takes precedence
    StopFirst,
    /// First triggered based on timestamp wins
    Timestamp,
}

/// OCO cancellation semantics
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct OcoSemantics {
    /// Cancel sibling when one leg triggers
    pub trigger_cancels_sibling: bool,
    /// Cancel sibling when one leg partially fills
    pub partial_fill_cancels_sibling: bool,
    /// Priority when both legs trigger simultaneously
    pub priority: OcoPriority,
    /// Allow amending one leg without canceling the other
    pub allow_independent_amend: bool,
    /// Automatically cancel both if one leg fails validation
    pub validation_cancels_both: bool,
}

impl Default for OcoSemantics {
    fn default() -> Self {
        Self {
            trigger_cancels_sibling: true,           // Standard behavior
            partial_fill_cancels_sibling: false,     // Wait for full trigger
            priority: OcoPriority::LimitFirst,       // Prefer limit order
            allow_independent_amend: false,          // Safer to prevent
            validation_cancels_both: true,           // Fail safe
        }
    }
}

/// OCO Order State
#[derive(Debug, Clone, PartialEq, Eq)]
/// TODO: Add docs
pub enum OcoState {
    /// Both legs are pending
    Pending,
    /// Limit leg is active, stop is waiting
    LimitActive,
    /// Stop leg is active, limit is waiting
    StopActive,
    /// One leg triggered, other cancelled
    Triggered { winning_leg: OcoLeg },
    /// Both legs cancelled by user
    Cancelled { reason: String },
    /// Error state (validation failure, etc.)
    Failed { error: String },
}

/// Which leg of the OCO
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum OcoLeg {
    Limit,
    Stop,
}

/// OCO Order - One-Cancels-Other order pair
/// TODO: Add docs
pub struct OcoOrder {
    /// Unique OCO ID
    id: String,
    /// The limit order leg
    limit_order: Order,
    /// The stop order leg  
    stop_order: Order,
    /// OCO execution semantics
    semantics: OcoSemantics,
    /// Current state
    state: Arc<RwLock<OcoState>>,
    /// Creation timestamp
    created_at: DateTime<Utc>,
    /// Last update timestamp
    updated_at: Arc<RwLock<DateTime<Utc>>>,
    /// Which leg was triggered (if any)
    triggered_leg: Arc<RwLock<Option<OcoLeg>>>,
}

impl OcoOrder {
    /// Create a new OCO order
    pub fn new(
        limit_order: Order,
        stop_order: Order,
        semantics: OcoSemantics,
    ) -> Result<Self> {
        // Validate orders
        Self::validate_oco_pair(&limit_order, &stop_order)?;
        
        Ok(Self {
            id: format!("OCO_{}", uuid::Uuid::new_v4()),
            limit_order,
            stop_order,
            semantics,
            state: Arc::new(RwLock::new(OcoState::Pending)),
            created_at: Utc::now(),
            updated_at: Arc::new(RwLock::new(Utc::now())),
            triggered_leg: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Validate OCO order pair
    fn validate_oco_pair(limit: &Order, stop: &Order) -> Result<()> {
        // Must be same symbol
        if limit.symbol() != stop.symbol() {
            bail!("OCO orders must have the same symbol");
        }
        
        // Must be same side
        if limit.side() != stop.side() {
            bail!("OCO orders must have the same side");
        }
        
        // Must be same quantity
        if limit.quantity() != stop.quantity() {
            bail!("OCO orders must have the same quantity");
        }
        
        // Limit must be limit order
        if limit.order_type() != OrderType::Limit {
            bail!("First leg must be a limit order");
        }
        
        // Stop must be stop or stop-limit
        if stop.order_type() != OrderType::StopMarket && 
           stop.order_type() != OrderType::StopLimit {
            bail!("Second leg must be a stop order");
        }
        
        // Price validation
        let limit_price = limit.price()
            .ok_or_else(|| anyhow::anyhow!("Limit order must have a price"))?;
        
        // For buy orders: limit price < stop trigger price
        // For sell orders: limit price > stop trigger price
        if let Some(stop_price) = stop.stop_price() {
            use crate::domain::entities::OrderSide;
            match limit.side() {
                OrderSide::Buy => {
                    if limit_price.value() >= stop_price.value() {
                        bail!("For buy OCO: limit price must be below stop price");
                    }
                }
                OrderSide::Sell => {
                    if limit_price.value() <= stop_price.value() {
                        bail!("For sell OCO: limit price must be above stop price");
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle trigger of one leg
    pub async fn handle_trigger(&self, triggered_leg: OcoLeg) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Check current state
        match *state {
            OcoState::Pending | OcoState::LimitActive | OcoState::StopActive => {
                // Valid states for triggering
            }
            _ => {
                bail!("Cannot trigger OCO in state: {:?}", *state);
            }
        }
        
        // Update state
        *state = OcoState::Triggered { winning_leg: triggered_leg };
        
        // Update triggered leg
        let mut triggered = self.triggered_leg.write().await;
        *triggered = Some(triggered_leg);
        
        // Update timestamp
        let mut updated = self.updated_at.write().await;
        *updated = Utc::now();
        
        Ok(())
    }
    
    /// Handle partial fill of one leg
    pub async fn handle_partial_fill(
        &self, 
        leg: OcoLeg, 
        filled_qty: Quantity
    ) -> Result<bool> {
        // Check semantics
        if !self.semantics.partial_fill_cancels_sibling {
            return Ok(false); // Don't cancel sibling
        }
        
        // Trigger the leg
        self.handle_trigger(leg).await?;
        
        Ok(true) // Sibling should be cancelled
    }
    
    /// Handle simultaneous trigger (both legs trigger in same tick)
    pub async fn handle_simultaneous_trigger(
        &self,
        limit_triggered_at: DateTime<Utc>,
        stop_triggered_at: DateTime<Utc>,
    ) -> Result<OcoLeg> {
        let winning_leg = match self.semantics.priority {
            OcoPriority::LimitFirst => OcoLeg::Limit,
            OcoPriority::StopFirst => OcoLeg::Stop,
            OcoPriority::Timestamp => {
                if limit_triggered_at <= stop_triggered_at {
                    OcoLeg::Limit
                } else {
                    OcoLeg::Stop
                }
            }
        };
        
        self.handle_trigger(winning_leg).await?;
        Ok(winning_leg)
    }
    
    /// Cancel the OCO order
    pub async fn cancel(&self, reason: String) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Can only cancel if not already triggered or cancelled
        match *state {
            OcoState::Pending | OcoState::LimitActive | OcoState::StopActive => {
                *state = OcoState::Cancelled { reason };
                
                // Update timestamp
                let mut updated = self.updated_at.write().await;
                *updated = Utc::now();
                
                Ok(())
            }
            _ => {
                bail!("Cannot cancel OCO in state: {:?}", *state);
            }
        }
    }
    
    /// Amend one leg of the OCO
    pub async fn amend_leg(
        &mut self,
        leg: OcoLeg,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
    ) -> Result<()> {
        // Check if independent amend is allowed
        if !self.semantics.allow_independent_amend {
            bail!("Independent amend not allowed for this OCO");
        }
        
        // Check state
        let state = self.state.read().await;
        match *state {
            OcoState::Pending | OcoState::LimitActive | OcoState::StopActive => {
                // Valid states for amending
            }
            _ => {
                bail!("Cannot amend OCO in state: {:?}", *state);
            }
        }
        drop(state);
        
        // Amend the appropriate leg
        match leg {
            OcoLeg::Limit => {
                if let Some(price) = new_price {
                    self.limit_order = self.limit_order.clone().with_price(price)?;
                }
                if let Some(qty) = new_quantity {
                    self.limit_order = self.limit_order.clone().with_quantity(qty)?;
                }
            }
            OcoLeg::Stop => {
                if let Some(price) = new_price {
                    self.stop_order = self.stop_order.clone().with_stop_price(price)?;
                }
                if let Some(qty) = new_quantity {
                    self.stop_order = self.stop_order.clone().with_quantity(qty)?;
                }
            }
        }
        
        // Re-validate the pair
        Self::validate_oco_pair(&self.limit_order, &self.stop_order)?;
        
        // Update timestamp
        let mut updated = self.updated_at.write().await;
        *updated = Utc::now();
        
        Ok(())
    }
    
    /// Get the winning order (if triggered)
    pub async fn get_winning_order(&self) -> Option<Order> {
        let triggered = self.triggered_leg.read().await;
        match *triggered {
            Some(OcoLeg::Limit) => Some(self.limit_order.clone()),
            Some(OcoLeg::Stop) => Some(self.stop_order.clone()),
            None => None,
        }
    }
    
    /// Get the cancelled order (if triggered)
    pub async fn get_cancelled_order(&self) -> Option<Order> {
        let triggered = self.triggered_leg.read().await;
        match *triggered {
            Some(OcoLeg::Limit) => Some(self.stop_order.clone()),
            Some(OcoLeg::Stop) => Some(self.limit_order.clone()),
            None => None,
        }
    }
    
    // Getters
    pub fn id(&self) -> &str { &self.id }
    pub fn limit_order(&self) -> &Order { &self.limit_order }
    pub fn stop_order(&self) -> &Order { &self.stop_order }
    pub fn semantics(&self) -> &OcoSemantics { &self.semantics }
    pub async fn state(&self) -> OcoState { self.state.read().await.clone() }
    pub fn created_at(&self) -> DateTime<Utc> { self.created_at }
    pub async fn updated_at(&self) -> DateTime<Utc> { *self.updated_at.read().await }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::{OrderSide, TimeInForce};
    
    fn create_test_oco() -> OcoOrder {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        
        // Buy OCO: Limit buy at 49000, stop buy at 51000
        let limit_order = Order::limit(
            symbol.clone(),
            OrderSide::Buy,
            Price::new(49000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
            TimeInForce::GTC,
        );
        
        let stop_order = Order::stop_market(
            symbol,
            OrderSide::Buy,
            Price::new(51000.0).expect("SAFETY: Add proper error handling"), // Stop trigger price
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
        );
        
        OcoOrder::new(limit_order, stop_order, OcoSemantics::default()).expect("SAFETY: Add proper error handling")
    }
    
    #[tokio::test]
    async fn test_oco_creation() {
        let oco = create_test_oco();
        assert_eq!(oco.state().await, OcoState::Pending);
        assert!(oco.id().starts_with("OCO_"));
    }
    
    #[tokio::test]
    async fn test_oco_trigger_limit() {
        let oco = create_test_oco();
        
        // Trigger limit leg
        oco.handle_trigger(OcoLeg::Limit).await.expect("SAFETY: Add proper error handling");
        
        assert_eq!(oco.state().await, OcoState::Triggered { winning_leg: OcoLeg::Limit });
        assert_eq!(oco.get_winning_order().await.expect("SAFETY: Add proper error handling").id(), oco.limit_order().id());
        assert_eq!(oco.get_cancelled_order().await.expect("SAFETY: Add proper error handling").id(), oco.stop_order().id());
    }
    
    #[tokio::test]
    async fn test_oco_partial_fill_semantics() {
        let mut semantics = OcoSemantics::default();
        semantics.partial_fill_cancels_sibling = true;
        
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        let limit_order = Order::limit(
            symbol.clone(),
            OrderSide::Buy,
            Price::new(49000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
            TimeInForce::GTC,
        );
        
        let stop_order = Order::stop_market(
            symbol,
            OrderSide::Buy,
            Price::new(51000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
        );
        
        let oco = OcoOrder::new(limit_order, stop_order, semantics).expect("SAFETY: Add proper error handling");
        
        // Partial fill should trigger cancellation
        let should_cancel = oco.handle_partial_fill(
            OcoLeg::Limit, 
            Quantity::new(0.5).expect("SAFETY: Add proper error handling")
        ).await.expect("SAFETY: Add proper error handling");
        
        assert!(should_cancel);
        assert_eq!(oco.state().await, OcoState::Triggered { winning_leg: OcoLeg::Limit });
    }
    
    #[tokio::test]
    async fn test_oco_simultaneous_trigger() {
        let mut semantics = OcoSemantics::default();
        semantics.priority = OcoPriority::Timestamp;
        
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        let limit_order = Order::limit(
            symbol.clone(),
            OrderSide::Buy,
            Price::new(49000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
            TimeInForce::GTC,
        );
        
        let stop_order = Order::stop_market(
            symbol,
            OrderSide::Buy,
            Price::new(51000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
        );
        
        let oco = OcoOrder::new(limit_order, stop_order, semantics).expect("SAFETY: Add proper error handling");
        
        let now = Utc::now();
        let limit_time = now;
        let stop_time = now + chrono::Duration::milliseconds(1);
        
        // Limit triggered first by timestamp
        let winner = oco.handle_simultaneous_trigger(limit_time, stop_time).await.expect("SAFETY: Add proper error handling");
        assert_eq!(winner, OcoLeg::Limit);
    }
    
    #[tokio::test]
    async fn test_oco_validation() {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        
        // Invalid: limit price above stop price for buy order
        let limit_order = Order::limit(
            symbol.clone(),
            OrderSide::Buy,
            Price::new(52000.0).expect("SAFETY: Add proper error handling"), // Above stop price
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
            TimeInForce::GTC,
        );
        
        let stop_order = Order::stop_market(
            symbol,
            OrderSide::Buy,
            Price::new(51000.0).expect("SAFETY: Add proper error handling"),
            Quantity::new(1.0).expect("SAFETY: Add proper error handling"),
        );
        
        let result = OcoOrder::new(limit_order, stop_order, OcoSemantics::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("limit price must be below stop price"));
    }
}