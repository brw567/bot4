//! Order management and tracking

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub fees: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderResult {
    pub order_id: String,
    pub status: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CancelResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderStatusResult {
    pub status: String,
    pub filled_quantity: Decimal,
    pub remaining_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub fees: Decimal,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeStats {
    pub total_orders: u64,
    pub successful_orders: u64,
    pub failed_orders: u64,
    pub average_latency_ms: f64,
}

pub struct OrderManager {
    orders: Arc<DashMap<String, Order>>,
    exchange_orders: Arc<DashMap<String, Vec<String>>>,
    stats: Arc<RwLock<DashMap<String, ExchangeStats>>>,
}

impl OrderManager {
    pub fn new() -> Self {
        Self {
            orders: Arc::new(DashMap::new()),
            exchange_orders: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(DashMap::new())),
        }
    }
    
    pub async fn submit_order(&self, mut order: Order) -> Result<OrderResult> {
        let start = std::time::Instant::now();
        
        // Validate order
        self.validate_order(&order)?;
        
        // Generate unique order ID
        if order.id.is_empty() {
            order.id = Uuid::new_v4().to_string();
        }
        
        // Update status to submitted
        order.status = OrderStatus::Submitted;
        order.updated_at = Utc::now();
        
        // Store order
        self.orders.insert(order.id.clone(), order.clone());
        
        // Track by exchange
        self.exchange_orders
            .entry(order.exchange.clone())
            .or_insert_with(Vec::new)
            .push(order.id.clone());
        
        // Simulate exchange submission (in production, would call actual API)
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        // Update stats
        self.update_stats(&order.exchange, true, start.elapsed().as_millis() as f64).await;
        
        Ok(OrderResult {
            order_id: order.id,
            status: "submitted".to_string(),
            message: format!("Order submitted to {}", order.exchange),
            timestamp: Utc::now(),
        })
    }
    
    pub async fn cancel_order(&self, exchange: &str, order_id: &str) -> Result<CancelResult> {
        // Find order
        let mut order = self.orders.get_mut(order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        
        // Check if order can be cancelled
        if !matches!(order.status, OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::PartiallyFilled) {
            return Ok(CancelResult {
                success: false,
                message: format!("Order cannot be cancelled in status: {:?}", order.status),
            });
        }
        
        // Update status
        order.status = OrderStatus::Cancelled;
        order.updated_at = Utc::now();
        
        Ok(CancelResult {
            success: true,
            message: "Order successfully cancelled".to_string(),
        })
    }
    
    pub async fn get_order_status(&self, exchange: &str, order_id: &str) -> Result<OrderStatusResult> {
        let order = self.orders.get(order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        
        Ok(OrderStatusResult {
            status: format!("{:?}", order.status),
            filled_quantity: order.filled_quantity,
            remaining_quantity: order.quantity - order.filled_quantity,
            average_fill_price: order.average_fill_price,
            fees: order.fees,
            last_update: order.updated_at,
        })
    }
    
    pub async fn update_order_fill(&self, order_id: &str, filled_qty: Decimal, 
                                   fill_price: Decimal, fees: Decimal) -> Result<()> {
        let mut order = self.orders.get_mut(order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        
        // Update filled quantity
        order.filled_quantity += filled_qty;
        
        // Update average fill price
        if let Some(avg_price) = order.average_fill_price {
            let total_value = avg_price * (order.filled_quantity - filled_qty) + fill_price * filled_qty;
            order.average_fill_price = Some(total_value / order.filled_quantity);
        } else {
            order.average_fill_price = Some(fill_price);
        }
        
        // Update fees
        order.fees += fees;
        
        // Update status
        if order.filled_quantity >= order.quantity {
            order.status = OrderStatus::Filled;
        } else {
            order.status = OrderStatus::PartiallyFilled;
        }
        
        order.updated_at = Utc::now();
        
        Ok(())
    }
    
    pub async fn get_exchange_stats(&self, exchange: &str) -> Result<ExchangeStats> {
        let stats = self.stats.read().await;
        
        if let Some(exchange_stats) = stats.get(exchange) {
            Ok(exchange_stats.clone())
        } else {
            Ok(ExchangeStats {
                total_orders: 0,
                successful_orders: 0,
                failed_orders: 0,
                average_latency_ms: 0.0,
            })
        }
    }
    
    pub async fn get_open_orders(&self, exchange: Option<&str>) -> Vec<Order> {
        let mut open_orders = Vec::new();
        
        for entry in self.orders.iter() {
            let order = entry.value();
            
            // Filter by exchange if specified
            if let Some(ex) = exchange {
                if order.exchange != ex {
                    continue;
                }
            }
            
            // Check if order is open
            if matches!(order.status, 
                       OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::PartiallyFilled) {
                open_orders.push(order.clone());
            }
        }
        
        open_orders
    }
    
    pub async fn get_order_history(&self, exchange: Option<&str>, limit: usize) -> Vec<Order> {
        let mut orders: Vec<Order> = self.orders.iter()
            .filter(|entry| {
                if let Some(ex) = exchange {
                    entry.value().exchange == ex
                } else {
                    true
                }
            })
            .map(|entry| entry.value().clone())
            .collect();
        
        // Sort by creation time (newest first)
        orders.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        // Apply limit
        orders.truncate(limit);
        
        orders
    }
    
    fn validate_order(&self, order: &Order) -> Result<()> {
        // Validate quantity
        if order.quantity <= dec!(0) {
            bail!("Order quantity must be positive");
        }
        
        // Validate price for limit orders
        if order.order_type == "limit" {
            if order.price.is_none() || order.price.unwrap() <= dec!(0) {
                bail!("Limit orders must have a positive price");
            }
        }
        
        // Validate side
        if !["buy", "sell"].contains(&order.side.as_str()) {
            bail!("Invalid order side: {}", order.side);
        }
        
        // Validate order type
        if !["market", "limit", "stop_limit"].contains(&order.order_type.as_str()) {
            bail!("Invalid order type: {}", order.order_type);
        }
        
        Ok(())
    }
    
    async fn update_stats(&self, exchange: &str, success: bool, latency_ms: f64) {
        let mut stats = self.stats.write().await;
        
        let mut exchange_stats = stats.entry(exchange.to_string())
            .or_insert(ExchangeStats {
                total_orders: 0,
                successful_orders: 0,
                failed_orders: 0,
                average_latency_ms: 0.0,
            });
        
        exchange_stats.total_orders += 1;
        
        if success {
            exchange_stats.successful_orders += 1;
        } else {
            exchange_stats.failed_orders += 1;
        }
        
        // Update average latency (exponential moving average)
        let alpha = 0.1;
        exchange_stats.average_latency_ms = 
            alpha * latency_ms + (1.0 - alpha) * exchange_stats.average_latency_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_submit_order() {
        let manager = OrderManager::new();
        
        let order = Order {
            id: String::new(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "limit".to_string(),
            quantity: dec!(0.1),
            price: Some(dec!(50000)),
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: dec!(0),
            average_fill_price: None,
            fees: dec!(0),
        };
        
        let result = manager.submit_order(order).await.unwrap();
        assert!(!result.order_id.is_empty());
        assert_eq!(result.status, "submitted");
    }
    
    #[tokio::test]
    async fn test_cancel_order() {
        let manager = OrderManager::new();
        
        let order = Order {
            id: "test-order".to_string(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "limit".to_string(),
            quantity: dec!(0.1),
            price: Some(dec!(50000)),
            status: OrderStatus::Submitted,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: dec!(0),
            average_fill_price: None,
            fees: dec!(0),
        };
        
        manager.orders.insert(order.id.clone(), order);
        
        let result = manager.cancel_order("binance", "test-order").await.unwrap();
        assert!(result.success);
    }
}