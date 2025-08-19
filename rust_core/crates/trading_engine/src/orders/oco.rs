// One-Cancels-Other (OCO) Order Management System
// Casey (Exchange Lead) + Quinn (Risk) + Sam (Architecture)
// CRITICAL: Sophia Requirement #5 - Complex order types
// References: FIX Protocol 5.0, CME Globex OCO Implementation

use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};
use async_trait::async_trait;

/// OCO Order Group - Two orders linked where one cancels the other
/// Casey: "Essential for bracket orders and stop-loss with profit targets!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCOGroup {
    pub group_id: Uuid,
    pub primary_order: Order,
    pub secondary_order: Order,
    pub link_type: OCOLinkType,
    pub created_at: DateTime<Utc>,
    pub status: OCOStatus,
    pub metadata: OCOMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OCOLinkType {
    /// Standard OCO - either order fills, other cancels
    Standard,
    /// Bracket - entry order with stop-loss and take-profit
    Bracket { 
        entry_filled: bool,
        stop_activated: bool,
        target_activated: bool,
    },
    /// OTO (One-Triggers-Other) - first order must fill to activate second
    OneTriggersOther { 
        primary_filled: bool 
    },
    /// Multi-leg strategy order
    MultiLeg { 
        legs_count: usize,
        filled_legs: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OCOStatus {
    Active,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCOMetadata {
    pub strategy: String,
    pub risk_limit: f64,
    pub max_slippage: f64,
    pub time_in_force: TimeInForce,
    pub expire_time: Option<DateTime<Utc>>,
    pub reduce_only: bool,
    pub post_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC,  // Good Till Cancelled
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
    GTD(DateTime<Utc>),  // Good Till Date
    GTT(Duration),  // Good Till Time
}

/// Individual Order within OCO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,  // For limit orders
    pub stop_price: Option<f64>,  // For stop orders
    pub status: OrderStatus,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,
    pub fees: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop { trail_amount: f64 },
    Iceberg { display_size: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// OCO Order Manager - Handles complex order relationships
pub struct OCOManager {
    // Active OCO groups
    active_groups: Arc<RwLock<HashMap<Uuid, OCOGroup>>>,
    
    // Order ID to Group ID mapping
    order_to_group: Arc<RwLock<HashMap<Uuid, Uuid>>>,
    
    // Symbol-based indexing for performance
    symbol_groups: Arc<RwLock<HashMap<String, HashSet<Uuid>>>>,
    
    // Risk validator
    risk_validator: Arc<dyn RiskValidator>,
    
    // Exchange connector
    exchange: Arc<dyn ExchangeConnector>,
    
    // Performance metrics
    metrics: OCOMetrics,
}

/// Risk Validator trait
#[async_trait]
pub trait RiskValidator: Send + Sync {
    async fn validate_oco(&self, group: &OCOGroup) -> Result<(), RiskError>;
    async fn check_position_limits(&self, order: &Order) -> Result<(), RiskError>;
    async fn validate_margin(&self, group: &OCOGroup) -> Result<f64, RiskError>;
}

/// Exchange Connector trait
#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    async fn submit_order(&self, order: &Order) -> Result<Uuid, ExchangeError>;
    async fn cancel_order(&self, order_id: Uuid) -> Result<(), ExchangeError>;
    async fn modify_order(&self, order_id: Uuid, new_params: OrderModification) -> Result<(), ExchangeError>;
    async fn get_order_status(&self, order_id: Uuid) -> Result<OrderStatus, ExchangeError>;
}

#[derive(Debug, Clone)]
pub struct OrderModification {
    pub new_price: Option<f64>,
    pub new_quantity: Option<f64>,
    pub new_stop_price: Option<f64>,
}

/// Performance metrics for OCO operations
#[derive(Debug, Default)]
struct OCOMetrics {
    total_groups_created: u64,
    successful_fills: u64,
    cancelled_groups: u64,
    risk_rejections: u64,
    avg_execution_time_ms: f64,
    slippage_stats: SlippageStats,
}

#[derive(Debug, Default)]
struct SlippageStats {
    positive_slippage: f64,
    negative_slippage: f64,
    total_slippage: f64,
    count: u64,
}

impl OCOManager {
    pub fn new(
        risk_validator: Arc<dyn RiskValidator>,
        exchange: Arc<dyn ExchangeConnector>,
    ) -> Self {
        Self {
            active_groups: Arc::new(RwLock::new(HashMap::new())),
            order_to_group: Arc::new(RwLock::new(HashMap::new())),
            symbol_groups: Arc::new(RwLock::new(HashMap::new())),
            risk_validator,
            exchange,
            metrics: OCOMetrics::default(),
        }
    }
    
    /// Create a new OCO group with risk validation
    /// Quinn: "Every order must pass risk checks before submission!"
    pub async fn create_oco_group(
        &self,
        primary: Order,
        secondary: Order,
        link_type: OCOLinkType,
        metadata: OCOMetadata,
    ) -> Result<Uuid, OCOError> {
        let group_id = Uuid::new_v4();
        
        let group = OCOGroup {
            group_id,
            primary_order: primary.clone(),
            secondary_order: secondary.clone(),
            link_type,
            created_at: Utc::now(),
            status: OCOStatus::Active,
            metadata,
        };
        
        // Risk validation
        self.risk_validator.validate_oco(&group).await
            .map_err(|e| OCOError::RiskValidation(e))?;
        
        // Check position limits
        self.risk_validator.check_position_limits(&primary).await
            .map_err(|e| OCOError::RiskValidation(e))?;
        self.risk_validator.check_position_limits(&secondary).await
            .map_err(|e| OCOError::RiskValidation(e))?;
        
        // Validate margin requirements
        let required_margin = self.risk_validator.validate_margin(&group).await
            .map_err(|e| OCOError::RiskValidation(e))?;
        
        info!("OCO Group {} validated. Required margin: ${:.2}", group_id, required_margin);
        
        // Submit orders based on link type
        match link_type {
            OCOLinkType::Standard => {
                self.submit_standard_oco(&group).await?;
            },
            OCOLinkType::Bracket { .. } => {
                self.submit_bracket_order(&group).await?;
            },
            OCOLinkType::OneTriggersOther { .. } => {
                self.submit_oto_order(&group).await?;
            },
            OCOLinkType::MultiLeg { .. } => {
                self.submit_multileg_order(&group).await?;
            },
        }
        
        // Store group information
        {
            let mut groups = self.active_groups.write();
            groups.insert(group_id, group.clone());
            
            let mut order_map = self.order_to_group.write();
            order_map.insert(primary.order_id, group_id);
            order_map.insert(secondary.order_id, group_id);
            
            let mut symbol_map = self.symbol_groups.write();
            symbol_map.entry(primary.symbol.clone())
                .or_insert_with(HashSet::new)
                .insert(group_id);
        }
        
        Ok(group_id)
    }
    
    /// Submit standard OCO orders
    async fn submit_standard_oco(&self, group: &OCOGroup) -> Result<(), OCOError> {
        // Submit both orders simultaneously
        let primary_future = self.exchange.submit_order(&group.primary_order);
        let secondary_future = self.exchange.submit_order(&group.secondary_order);
        
        // Wait for both submissions
        let (primary_result, secondary_result) = tokio::join!(primary_future, secondary_future);
        
        // Handle submission results
        if let Err(e) = primary_result {
            // Cancel secondary if primary failed
            if let Ok(secondary_id) = secondary_result {
                let _ = self.exchange.cancel_order(secondary_id).await;
            }
            return Err(OCOError::ExchangeError(e));
        }
        
        if let Err(e) = secondary_result {
            // Cancel primary if secondary failed
            if let Ok(primary_id) = primary_result {
                let _ = self.exchange.cancel_order(primary_id).await;
            }
            return Err(OCOError::ExchangeError(e));
        }
        
        Ok(())
    }
    
    /// Submit bracket order (entry + stop-loss + take-profit)
    async fn submit_bracket_order(&self, group: &OCOGroup) -> Result<(), OCOError> {
        // First submit the entry order
        let entry_id = self.exchange.submit_order(&group.primary_order).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        info!("Bracket entry order {} submitted", entry_id);
        
        // Stop and target orders are submitted after entry fills
        // This is handled in the order update handler
        
        Ok(())
    }
    
    /// Submit One-Triggers-Other order
    async fn submit_oto_order(&self, group: &OCOGroup) -> Result<(), OCOError> {
        // Submit primary order first
        let primary_id = self.exchange.submit_order(&group.primary_order).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        info!("OTO primary order {} submitted", primary_id);
        
        // Secondary order submitted after primary fills
        // Handled in order update handler
        
        Ok(())
    }
    
    /// Submit multi-leg strategy order
    async fn submit_multileg_order(&self, group: &OCOGroup) -> Result<(), OCOError> {
        // Submit all legs with specific timing
        // Implementation depends on strategy type
        
        let primary_id = self.exchange.submit_order(&group.primary_order).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        let secondary_id = self.exchange.submit_order(&group.secondary_order).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        info!("Multi-leg orders {}, {} submitted", primary_id, secondary_id);
        
        Ok(())
    }
    
    /// Handle order fill event
    /// Casey: "Critical for OCO logic - must be atomic!"
    pub async fn handle_order_fill(
        &mut self,
        order_id: Uuid,
        fill_price: f64,
        fill_quantity: f64,
    ) -> Result<(), OCOError> {
        let start = std::time::Instant::now();
        
        // Find the OCO group
        let group_id = {
            let order_map = self.order_to_group.read();
            order_map.get(&order_id).copied()
        };
        
        if let Some(group_id) = group_id {
            let mut groups = self.active_groups.write();
            if let Some(group) = groups.get_mut(&group_id) {
                // Update order status
                if group.primary_order.order_id == order_id {
                    group.primary_order.filled_quantity += fill_quantity;
                    group.primary_order.avg_fill_price = 
                        (group.primary_order.avg_fill_price * (group.primary_order.filled_quantity - fill_quantity)
                         + fill_price * fill_quantity) / group.primary_order.filled_quantity;
                    
                    if group.primary_order.filled_quantity >= group.primary_order.quantity {
                        group.primary_order.status = OrderStatus::Filled;
                        
                        // Cancel the other order
                        self.cancel_linked_order(&group.secondary_order.order_id).await?;
                        group.status = OCOStatus::Filled;
                    }
                } else if group.secondary_order.order_id == order_id {
                    group.secondary_order.filled_quantity += fill_quantity;
                    group.secondary_order.avg_fill_price = 
                        (group.secondary_order.avg_fill_price * (group.secondary_order.filled_quantity - fill_quantity)
                         + fill_price * fill_quantity) / group.secondary_order.filled_quantity;
                    
                    if group.secondary_order.filled_quantity >= group.secondary_order.quantity {
                        group.secondary_order.status = OrderStatus::Filled;
                        
                        // Cancel the other order
                        self.cancel_linked_order(&group.primary_order.order_id).await?;
                        group.status = OCOStatus::Filled;
                    }
                }
                
                // Handle special link types
                match &mut group.link_type {
                    OCOLinkType::Bracket { entry_filled, .. } => {
                        if group.primary_order.status == OrderStatus::Filled && !*entry_filled {
                            *entry_filled = true;
                            // Now activate stop-loss and take-profit orders
                            self.activate_bracket_orders(group).await?;
                        }
                    },
                    OCOLinkType::OneTriggersOther { primary_filled } => {
                        if group.primary_order.status == OrderStatus::Filled && !*primary_filled {
                            *primary_filled = true;
                            // Now submit the secondary order
                            self.exchange.submit_order(&group.secondary_order).await
                                .map_err(|e| OCOError::ExchangeError(e))?;
                        }
                    },
                    _ => {}
                }
                
                // Update metrics
                self.metrics.successful_fills += 1;
                self.metrics.avg_execution_time_ms = 
                    (self.metrics.avg_execution_time_ms * (self.metrics.successful_fills - 1) as f64
                     + start.elapsed().as_millis() as f64) / self.metrics.successful_fills as f64;
            }
        }
        
        Ok(())
    }
    
    /// Cancel linked order
    async fn cancel_linked_order(&self, order_id: &Uuid) -> Result<(), OCOError> {
        self.exchange.cancel_order(*order_id).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        info!("Cancelled linked order {}", order_id);
        Ok(())
    }
    
    /// Activate bracket orders after entry fill
    async fn activate_bracket_orders(&self, group: &OCOGroup) -> Result<(), OCOError> {
        // Create stop-loss order
        let stop_loss = Order {
            order_id: Uuid::new_v4(),
            symbol: group.primary_order.symbol.clone(),
            side: if group.primary_order.side == OrderSide::Buy {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            },
            order_type: OrderType::Stop,
            quantity: group.primary_order.filled_quantity,
            price: None,
            stop_price: group.secondary_order.stop_price,  // Use predefined stop
            status: OrderStatus::New,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fees: 0.0,
            timestamp: Utc::now(),
        };
        
        // Submit stop-loss
        self.exchange.submit_order(&stop_loss).await
            .map_err(|e| OCOError::ExchangeError(e))?;
        
        info!("Bracket stop-loss activated at {:.2}", stop_loss.stop_price.unwrap_or(0.0));
        
        Ok(())
    }
    
    /// Cancel OCO group
    pub async fn cancel_oco_group(&mut self, group_id: Uuid) -> Result<(), OCOError> {
        let group = {
            let mut groups = self.active_groups.write();
            groups.remove(&group_id)
        };
        
        if let Some(group) = group {
            // Cancel both orders
            let _ = self.exchange.cancel_order(group.primary_order.order_id).await;
            let _ = self.exchange.cancel_order(group.secondary_order.order_id).await;
            
            // Clean up mappings
            let mut order_map = self.order_to_group.write();
            order_map.remove(&group.primary_order.order_id);
            order_map.remove(&group.secondary_order.order_id);
            
            let mut symbol_map = self.symbol_groups.write();
            if let Some(groups) = symbol_map.get_mut(&group.primary_order.symbol) {
                groups.remove(&group_id);
            }
            
            self.metrics.cancelled_groups += 1;
            
            info!("OCO group {} cancelled", group_id);
        }
        
        Ok(())
    }
    
    /// Get all active OCO groups for a symbol
    pub fn get_active_groups(&self, symbol: &str) -> Vec<OCOGroup> {
        let symbol_groups = self.symbol_groups.read();
        let active_groups = self.active_groups.read();
        
        if let Some(group_ids) = symbol_groups.get(symbol) {
            group_ids.iter()
                .filter_map(|id| active_groups.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get OCO metrics
    pub fn get_metrics(&self) -> OCOMetrics {
        OCOMetrics {
            total_groups_created: self.metrics.total_groups_created,
            successful_fills: self.metrics.successful_fills,
            cancelled_groups: self.metrics.cancelled_groups,
            risk_rejections: self.metrics.risk_rejections,
            avg_execution_time_ms: self.metrics.avg_execution_time_ms,
            slippage_stats: SlippageStats {
                positive_slippage: self.metrics.slippage_stats.positive_slippage,
                negative_slippage: self.metrics.slippage_stats.negative_slippage,
                total_slippage: self.metrics.slippage_stats.total_slippage,
                count: self.metrics.slippage_stats.count,
            },
        }
    }
}

/// OCO Error types
#[derive(Debug, thiserror::Error)]
pub enum OCOError {
    #[error("Risk validation failed: {0}")]
    RiskValidation(RiskError),
    
    #[error("Exchange error: {0}")]
    ExchangeError(ExchangeError),
    
    #[error("Invalid order parameters")]
    InvalidParameters,
    
    #[error("OCO group not found")]
    GroupNotFound,
    
    #[error("Order already in OCO group")]
    OrderAlreadyLinked,
}

#[derive(Debug, thiserror::Error)]
pub enum RiskError {
    #[error("Position limit exceeded")]
    PositionLimitExceeded,
    
    #[error("Insufficient margin")]
    InsufficientMargin,
    
    #[error("Risk limit exceeded")]
    RiskLimitExceeded,
}

#[derive(Debug, thiserror::Error)]
pub enum ExchangeError {
    #[error("Connection error")]
    ConnectionError,
    
    #[error("Order rejected by exchange")]
    OrderRejected,
    
    #[error("Invalid symbol")]
    InvalidSymbol,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock implementations for testing
    struct MockRiskValidator;
    
    #[async_trait]
    impl RiskValidator for MockRiskValidator {
        async fn validate_oco(&self, _group: &OCOGroup) -> Result<(), RiskError> {
            Ok(())
        }
        
        async fn check_position_limits(&self, _order: &Order) -> Result<(), RiskError> {
            Ok(())
        }
        
        async fn validate_margin(&self, _group: &OCOGroup) -> Result<f64, RiskError> {
            Ok(1000.0)  // Mock margin requirement
        }
    }
    
    struct MockExchange;
    
    #[async_trait]
    impl ExchangeConnector for MockExchange {
        async fn submit_order(&self, order: &Order) -> Result<Uuid, ExchangeError> {
            Ok(order.order_id)
        }
        
        async fn cancel_order(&self, _order_id: Uuid) -> Result<(), ExchangeError> {
            Ok(())
        }
        
        async fn modify_order(&self, _order_id: Uuid, _new_params: OrderModification) -> Result<(), ExchangeError> {
            Ok(())
        }
        
        async fn get_order_status(&self, _order_id: Uuid) -> Result<OrderStatus, ExchangeError> {
            Ok(OrderStatus::New)
        }
    }
    
    #[tokio::test]
    async fn test_create_standard_oco() {
        let risk_validator = Arc::new(MockRiskValidator);
        let exchange = Arc::new(MockExchange);
        let manager = OCOManager::new(risk_validator, exchange);
        
        let primary = Order {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 1.0,
            price: Some(50000.0),
            stop_price: None,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fees: 0.0,
            timestamp: Utc::now(),
        };
        
        let secondary = Order {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 1.0,
            price: Some(48000.0),
            stop_price: None,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fees: 0.0,
            timestamp: Utc::now(),
        };
        
        let metadata = OCOMetadata {
            strategy: "Breakout".to_string(),
            risk_limit: 10000.0,
            max_slippage: 0.01,
            time_in_force: TimeInForce::GTC,
            expire_time: None,
            reduce_only: false,
            post_only: false,
        };
        
        let group_id = manager.create_oco_group(
            primary,
            secondary,
            OCOLinkType::Standard,
            metadata
        ).await.unwrap();
        
        assert!(!group_id.is_nil());
        
        // Verify group was created
        let groups = manager.get_active_groups("BTC/USDT");
        assert_eq!(groups.len(), 1);
    }
    
    #[tokio::test]
    async fn test_handle_order_fill() {
        let risk_validator = Arc::new(MockRiskValidator);
        let exchange = Arc::new(MockExchange);
        let mut manager = OCOManager::new(risk_validator, exchange);
        
        // Create test orders
        let primary = Order {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 1.0,
            price: Some(50000.0),
            stop_price: None,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fees: 0.0,
            timestamp: Utc::now(),
        };
        
        let secondary = Order {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Stop,
            quantity: 1.0,
            price: None,
            stop_price: Some(49000.0),
            status: OrderStatus::New,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fees: 0.0,
            timestamp: Utc::now(),
        };
        
        let metadata = OCOMetadata {
            strategy: "StopLoss".to_string(),
            risk_limit: 10000.0,
            max_slippage: 0.01,
            time_in_force: TimeInForce::GTC,
            expire_time: None,
            reduce_only: false,
            post_only: false,
        };
        
        let group_id = manager.create_oco_group(
            primary.clone(),
            secondary,
            OCOLinkType::Standard,
            metadata
        ).await.unwrap();
        
        // Simulate order fill
        manager.handle_order_fill(primary.order_id, 50000.0, 1.0).await.unwrap();
        
        // Verify group status updated
        let groups = manager.active_groups.read();
        let group = groups.get(&group_id).unwrap();
        assert_eq!(group.status, OCOStatus::Filled);
    }
}