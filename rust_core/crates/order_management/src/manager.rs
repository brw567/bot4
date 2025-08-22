// Order Manager - Central orchestration for order lifecycle
// Phase: 1.4 - Order Management
// Performance: <100μs for order operations

use std::sync::Arc;
use std::time::Instant;
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use thiserror::Error;
use tracing::{info, error, debug, warn};

use crate::order::{Order, OrderId, OrderFill, OrderValidationError};
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
    pub async fn create_order(&self, order: Order) -> Result<OrderId, OrderManagerError> {
        let start = Instant::now();
        
        // COMPREHENSIVE LOGGING: Order creation entry
        debug!("Creating new order: symbol={}, side={:?}, type={:?}, quantity={}",
               order.symbol, order.side, order.order_type, order.quantity);
        
        // Validate order parameters
        order.validate()
            .map_err(|e| {
                warn!("Order validation failed: {}", e);
                OrderManagerError::ValidationError(e)
            })?;
        
        debug!("Order validation passed for {}", order.client_order_id);
        
        // Check duplicate if enabled
        if self.config.enable_duplicate_checks {
            debug!("Checking for duplicate order: {}", order.client_order_id);
            if self.check_duplicate(&order).await? {
                warn!("Duplicate order detected: {}", order.client_order_id);
                return Err(OrderManagerError::DuplicateOrder(order.client_order_id));
            }
        }
        
        // Check order limits
        debug!("Checking order limits for {}", order.symbol);
        self.check_order_limits(&order)?;
        debug!("Order limits check passed");
        
        // Risk checks if enabled
        if self.config.enable_risk_checks {
            debug!("Performing risk checks for order {}", order.id);
            self.perform_risk_checks(&order)?;
            debug!("Risk checks passed for order {}", order.id);
        }
        
        let order_id = order.id;
        
        // Create state machine
        let state_machine = Arc::new(OrderStateMachine::new(order_id));
        
        // ZERO-COPY: Extract values needed for logging before moving order
        let symbol = order.symbol.clone(); // Single clone for symbol tracking
        let side = order.side;
        let quantity = order.quantity;
        let price = order.price;
        
        // Persist to database if available (before moving order)
        if let Some(pool) = &self.db_pool {
            self.persist_order(pool, &order).await?;
        }
        
        // Store order and state machine - MOVE order, no clone!
        self.orders.insert(order_id, Arc::new(RwLock::new(order)));
        self.state_machines.insert(order_id, state_machine.clone());
        
        // Track by symbol (using pre-cloned symbol)
        self.symbol_orders
            .entry(symbol.clone())
            .or_default()
            .push(order_id);
        
        // Transition to validated state
        state_machine.process_event(OrderEvent::Validate)
            .map_err(|e| {
                error!("Failed to transition order {} to validated state: {}", order_id, e);
                e
            })?;
        debug!("Order {} transitioned to validated state", order_id);
        
        // Update metrics
        self.metrics.orders_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.creation_latency_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed
        );
        
        info!(
            "Order created: {} for {} {} {} @ {:?} (latency: {}μs)",
            order_id, side, quantity, symbol, price,
            start.elapsed().as_micros()
        );
        
        // COMPREHENSIVE LOGGING: Performance tracking
        if start.elapsed().as_micros() > 100 {
            warn!("Slow order creation: {}μs for order {}", 
                  start.elapsed().as_micros(), order_id);
        }
        
        Ok(order_id)
    }
    
    /// Submit order to exchange
    pub async fn submit_order(&self, order_id: OrderId) -> Result<(), OrderManagerError> {
        debug!("Submitting order {} to exchange", order_id);
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
        
        info!("Order submitted: {} (active orders: {})", 
              order_id, self.get_active_orders().len());
        Ok(())
    }
    
    /// Process order fill
    pub async fn process_fill(
        &self,
        order_id: OrderId,
        fill: OrderFill,
    ) -> Result<(), OrderManagerError> {
        debug!("Processing fill for order {}: quantity={}, price={}",
               order_id, fill.quantity, fill.price);
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
            "Order {} fill processed: {} @ {} (complete: {}, total_filled: {})",
            order_id, fill.quantity, fill.price, is_complete,
            if let Some(order_ref) = self.orders.get(&order_id) {
                order_ref.read().filled_quantity
            } else {
                Decimal::ZERO
            }
        );
        
        // COMPREHENSIVE LOGGING: Fill metrics
        if is_complete {
            debug!("Order {} fully filled, removing from active orders", order_id);
        } else {
            debug!("Order {} partially filled, remaining in active orders", order_id);
        }
        
        Ok(())
    }
    
    /// Cancel order
    pub async fn cancel_order(&self, order_id: OrderId) -> Result<(), OrderManagerError> {
        debug!("Attempting to cancel order {}", order_id);
        let state_machine = self.get_state_machine(order_id)?;
        
        // Check if order can be cancelled
        if state_machine.is_terminal() {
            warn!("Cannot cancel order {} - already in terminal state", order_id);
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
        
        info!("Order cancelled: {} (reason: user_requested)", order_id);
        
        // COMPREHENSIVE LOGGING: Cancellation metrics
        debug!("Total cancelled orders: {}", 
               self.metrics.orders_cancelled.load(std::sync::atomic::Ordering::Relaxed));
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
        debug!("Active orders: {}/{}", active_count, self.config.max_open_orders);
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
            warn!("Symbol {} has reached max orders limit: {}/{}",
                  order.symbol, active_symbol_orders, self.config.max_orders_per_symbol);
            return Err(OrderManagerError::TooManyOrders(
                format!("Maximum {} orders per symbol reached for {}", 
                    self.config.max_orders_per_symbol, order.symbol)
            ));
        }
        
        debug!("Symbol {} has {}/{} active orders",
               order.symbol, active_symbol_orders, self.config.max_orders_per_symbol);
        
        Ok(())
    }
    
    fn perform_risk_checks(&self, order: &Order) -> Result<(), OrderManagerError> {
        debug!("Performing risk checks: position_size={}%, stop_loss={:?}",
               order.position_size_pct * Decimal::from(100), order.stop_loss_price);
        
        // Check position size (Quinn's 2% rule)
        if order.position_size_pct > Decimal::from_str_exact("0.02").unwrap() {
            return Err(OrderManagerError::RiskCheckFailed(
                format!("Position size {}% exceeds 2% limit", order.position_size_pct * Decimal::from(100))
            ));
        }
        
        // Check stop loss is set
        if order.stop_loss_price.is_none() {
            error!("Risk check failed: Stop loss not set for order {}", order.id);
            return Err(OrderManagerError::RiskCheckFailed(
                "Stop loss is required for all orders".to_string()
            ));
        }
        
        debug!("Risk checks passed: position_size={}%, stop_loss={:?}",
               order.position_size_pct * Decimal::from(100), order.stop_loss_price);
        
        Ok(())
    }
    
    async fn persist_order(&self, _pool: &PgPool, _order: &Order) -> Result<(), OrderManagerError> {
        // TODO: Implement database persistence with proper SQLx offline mode
        // For now, skip database operations to allow compilation
        Ok(())
    }
}

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
        
        let order2 = order1.clone();
        let order3 = order1.clone();
        
        let id1 = manager.create_order(order1).await.unwrap();
        manager.submit_order(id1).await.unwrap();
        
        // Create second order
        let id2 = manager.create_order(order2).await.unwrap();
        manager.submit_order(id2).await.unwrap();
        
        // Third order should fail
        let result = manager.create_order(order3).await;
        assert!(result.is_err());
    }
}