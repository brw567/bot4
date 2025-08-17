// Order Manager - Central orchestration for order lifecycle
// Phase: 1.4 - Order Management
// Performance: <100μs for order operations

use std::sync::Arc;
use std::time::Instant;
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, postgres::PgRow, Row};
use thiserror::Error;
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::order::{Order, OrderId, OrderUpdate, OrderFill, OrderValidationError};
use crate::state_machine::{OrderStateMachine, OrderState, OrderEvent, StateTransitionError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerConfig {
    pub max_open_orders: usize,
    pub max_orders_per_symbol: usize,
    pub enable_risk_checks: bool,
    pub enable_duplicate_checks: bool,
    pub order_timeout_seconds: u64,
}

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            max_open_orders: 100,
            max_orders_per_symbol: 10,
            enable_risk_checks: true,
            enable_duplicate_checks: true,
            order_timeout_seconds: 300, // 5 minutes
        }
    }
}

pub struct OrderManager {
    config: Arc<OrderManagerConfig>,
    orders: Arc<DashMap<OrderId, Arc<RwLock<Order>>>>,
    state_machines: Arc<DashMap<OrderId, Arc<OrderStateMachine>>>,
    symbol_orders: Arc<DashMap<String, Vec<OrderId>>>,
    db_pool: Option<PgPool>,
    metrics: Arc<OrderMetrics>,
}

impl OrderManager {
    pub fn new(config: OrderManagerConfig, db_pool: Option<PgPool>) -> Self {
        Self {
            config: Arc::new(config),
            orders: Arc::new(DashMap::new()),
            state_machines: Arc::new(DashMap::new()),
            symbol_orders: Arc::new(DashMap::new()),
            db_pool,
            metrics: Arc::new(OrderMetrics::new()),
        }
    }
    
    /// Create and validate a new order
    pub async fn create_order(&self, mut order: Order) -> Result<OrderId, OrderManagerError> {
        let start = Instant::now();
        
        // Validate order parameters
        order.validate()
            .map_err(OrderManagerError::ValidationError)?;
        
        // Check duplicate if enabled
        if self.config.enable_duplicate_checks {
            if self.check_duplicate(&order).await? {
                return Err(OrderManagerError::DuplicateOrder(order.client_order_id));
            }
        }
        
        // Check order limits
        self.check_order_limits(&order)?;
        
        // Risk checks if enabled
        if self.config.enable_risk_checks {
            self.perform_risk_checks(&order)?;
        }
        
        let order_id = order.id;
        
        // Create state machine
        let state_machine = Arc::new(OrderStateMachine::new(order_id));
        
        // Store order and state machine
        self.orders.insert(order_id, Arc::new(RwLock::new(order.clone())));
        self.state_machines.insert(order_id, state_machine.clone());
        
        // Track by symbol
        self.symbol_orders
            .entry(order.symbol.clone())
            .or_insert_with(Vec::new)
            .push(order_id);
        
        // Transition to validated state
        state_machine.process_event(OrderEvent::Validate)?;
        
        // Persist to database if available
        if let Some(pool) = &self.db_pool {
            self.persist_order(pool, &order).await?;
        }
        
        // Update metrics
        self.metrics.orders_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.creation_latency_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed
        );
        
        info!(
            "Order created: {} for {} {} {} @ {:?}",
            order_id, order.side, order.quantity, order.symbol, order.price
        );
        
        Ok(order_id)
    }
    
    /// Submit order to exchange
    pub async fn submit_order(&self, order_id: OrderId) -> Result<(), OrderManagerError> {
        let state_machine = self.get_state_machine(order_id)?;
        
        // Transition to submitted state
        state_machine.process_event(OrderEvent::Submit)?;
        
        // Update order
        if let Some(order_ref) = self.orders.get(&order_id) {
            let mut order = order_ref.write();
            order.submitted_at = Some(chrono::Utc::now());
            order.updated_at = chrono::Utc::now();
        }
        
        self.metrics.orders_submitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        info!("Order submitted: {}", order_id);
        Ok(())
    }
    
    /// Process order fill
    pub async fn process_fill(
        &self,
        order_id: OrderId,
        fill: OrderFill,
    ) -> Result<(), OrderManagerError> {
        let state_machine = self.get_state_machine(order_id)?;
        
        // Update order with fill details
        let is_complete = if let Some(order_ref) = self.orders.get(&order_id) {
            let mut order = order_ref.write();
            
            order.filled_quantity += fill.quantity;
            order.commission += fill.commission;
            
            // Update average fill price
            if let Some(avg_price) = order.average_fill_price {
                let total_value = avg_price * (order.filled_quantity - fill.quantity) 
                    + fill.price * fill.quantity;
                order.average_fill_price = Some(total_value / order.filled_quantity);
            } else {
                order.average_fill_price = Some(fill.price);
            }
            
            order.updated_at = chrono::Utc::now();
            
            order.is_filled()
        } else {
            return Err(OrderManagerError::OrderNotFound(order_id));
        };
        
        // Update state machine
        if is_complete {
            state_machine.process_event(OrderEvent::Fill {
                quantity: fill.quantity,
                price: fill.price,
            })?;
            self.metrics.orders_filled.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            state_machine.process_event(OrderEvent::PartialFill {
                quantity: fill.quantity,
                price: fill.price,
            })?;
            self.metrics.orders_partially_filled.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
        info!(
            "Order {} fill processed: {} @ {} (complete: {})",
            order_id, fill.quantity, fill.price, is_complete
        );
        
        Ok(())
    }
    
    /// Cancel order
    pub async fn cancel_order(&self, order_id: OrderId) -> Result<(), OrderManagerError> {
        let state_machine = self.get_state_machine(order_id)?;
        
        // Check if order can be cancelled
        if state_machine.is_terminal() {
            return Err(OrderManagerError::InvalidState(
                format!("Order {} is already in terminal state", order_id)
            ));
        }
        
        // Transition to cancelled
        state_machine.process_event(OrderEvent::Cancel)?;
        
        // Update order
        if let Some(order_ref) = self.orders.get(&order_id) {
            let mut order = order_ref.write();
            order.cancelled_at = Some(chrono::Utc::now());
            order.updated_at = chrono::Utc::now();
        }
        
        self.metrics.orders_cancelled.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        info!("Order cancelled: {}", order_id);
        Ok(())
    }
    
    /// Get order by ID
    pub fn get_order(&self, order_id: OrderId) -> Option<Order> {
        self.orders.get(&order_id).map(|o| o.read().clone())
    }
    
    /// Get all orders for a symbol
    pub fn get_symbol_orders(&self, symbol: &str) -> Vec<Order> {
        self.symbol_orders
            .get(symbol)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_order(*id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<Order> {
        self.orders
            .iter()
            .filter_map(|entry| {
                let order_id = *entry.key();
                if let Some(sm) = self.state_machines.get(&order_id) {
                    if sm.is_active() {
                        Some(entry.value().read().clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get order state
    pub fn get_order_state(&self, order_id: OrderId) -> Result<OrderState, OrderManagerError> {
        let state_machine = self.get_state_machine(order_id)?;
        Ok(state_machine.current_state())
    }
    
    // Internal helper methods
    
    fn get_state_machine(&self, order_id: OrderId) -> Result<Arc<OrderStateMachine>, OrderManagerError> {
        self.state_machines
            .get(&order_id)
            .map(|sm| sm.clone())
            .ok_or(OrderManagerError::OrderNotFound(order_id))
    }
    
    async fn check_duplicate(&self, order: &Order) -> Result<bool, OrderManagerError> {
        // Check by client order ID
        for entry in self.orders.iter() {
            if entry.value().read().client_order_id == order.client_order_id {
                return Ok(true);
            }
        }
        Ok(false)
    }
    
    fn check_order_limits(&self, order: &Order) -> Result<(), OrderManagerError> {
        // Check total open orders
        let active_count = self.get_active_orders().len();
        if active_count >= self.config.max_open_orders {
            return Err(OrderManagerError::TooManyOrders(
                format!("Maximum {} open orders reached", self.config.max_open_orders)
            ));
        }
        
        // Check orders per symbol
        let symbol_orders = self.get_symbol_orders(&order.symbol);
        let active_symbol_orders = symbol_orders
            .iter()
            .filter(|o| {
                self.get_order_state(o.id)
                    .map(|s| s.is_active())
                    .unwrap_or(false)
            })
            .count();
            
        if active_symbol_orders >= self.config.max_orders_per_symbol {
            return Err(OrderManagerError::TooManyOrders(
                format!("Maximum {} orders per symbol reached for {}", 
                    self.config.max_orders_per_symbol, order.symbol)
            ));
        }
        
        Ok(())
    }
    
    fn perform_risk_checks(&self, order: &Order) -> Result<(), OrderManagerError> {
        // Check position size (Quinn's 2% rule)
        if order.position_size_pct > Decimal::from_str_exact("0.02").unwrap() {
            return Err(OrderManagerError::RiskCheckFailed(
                format!("Position size {}% exceeds 2% limit", order.position_size_pct * Decimal::from(100))
            ));
        }
        
        // Check stop loss is set
        if order.stop_loss_price.is_none() {
            return Err(OrderManagerError::RiskCheckFailed(
                "Stop loss is required for all orders".to_string()
            ));
        }
        
        Ok(())
    }
    
    async fn persist_order(&self, _pool: &PgPool, _order: &Order) -> Result<(), OrderManagerError> {
        // TODO: Implement database persistence with proper SQLx offline mode
        // For now, skip database operations to allow compilation
        Ok(())
    }
}

use rust_decimal::prelude::FromStr;
use std::sync::atomic::AtomicU64;

/// Order management metrics
struct OrderMetrics {
    orders_created: AtomicU64,
    orders_submitted: AtomicU64,
    orders_filled: AtomicU64,
    orders_partially_filled: AtomicU64,
    orders_cancelled: AtomicU64,
    orders_rejected: AtomicU64,
    creation_latency_us: AtomicU64,
}

impl OrderMetrics {
    fn new() -> Self {
        Self {
            orders_created: AtomicU64::new(0),
            orders_submitted: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_partially_filled: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            creation_latency_us: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Error)]
pub enum OrderManagerError {
    #[error("Order not found: {0}")]
    OrderNotFound(OrderId),
    
    #[error("Validation error: {0}")]
    ValidationError(#[from] OrderValidationError),
    
    #[error("State transition error: {0}")]
    StateTransitionError(#[from] StateTransitionError),
    
    #[error("Duplicate order: {0}")]
    DuplicateOrder(String),
    
    #[error("Too many orders: {0}")]
    TooManyOrders(String),
    
    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use crate::order::{Order, OrderSide, OrderType};
    
    #[tokio::test]
    async fn test_order_creation() {
        let manager = OrderManager::new(OrderManagerConfig::default(), None);
        
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.01),
        )
        .with_price(dec!(50000))
        .with_stop_loss(dec!(49000))
        .with_risk_params(dec!(0.01), dec!(100));
        
        let order_id = manager.create_order(order).await.unwrap();
        
        let retrieved = manager.get_order(order_id).unwrap();
        assert_eq!(retrieved.id, order_id);
        assert_eq!(retrieved.symbol, "BTCUSDT");
        
        let state = manager.get_order_state(order_id).unwrap();
        assert_eq!(state, OrderState::Validated);
    }
    
    #[tokio::test]
    async fn test_order_limits() {
        let mut config = OrderManagerConfig::default();
        config.max_orders_per_symbol = 2;
        
        let manager = OrderManager::new(config, None);
        
        // Create first order
        let order1 = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            dec!(0.01),
        )
        .with_stop_loss(dec!(49000))
        .with_risk_params(dec!(0.01), dec!(100));
        
        let id1 = manager.create_order(order1).await.unwrap();
        manager.submit_order(id1).await.unwrap();
        
        // Create second order
        let order2 = order1.clone();
        let id2 = manager.create_order(order2).await.unwrap();
        manager.submit_order(id2).await.unwrap();
        
        // Third order should fail
        let order3 = order1.clone();
        let result = manager.create_order(order3).await;
        assert!(result.is_err());
    }
}