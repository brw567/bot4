// DTO: Order Response Data Transfer Objects
// For API output - separate from domain models
// Owner: Sam | Reviewer: Alex

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::domain::entities::{Order, OrderId, OrderStatus, OrderSide, OrderType};

/// Order response DTO for API output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct OrderResponse {
    pub order_id: String,
    pub symbol: String,
    
    #[serde(rename = "type")]
    pub order_type: String,
    
    pub side: String,
    pub status: String,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub price: Option<f64>,
    pub average_fill_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub exchange_order_id: Option<String>,
}

impl OrderResponse {
    /// Create from domain Order
    pub fn from_domain(order: &Order) -> Self {
        OrderResponse {
            order_id: order.id().to_string(),
            symbol: order.symbol().to_string(),
            order_type: format!("{:?}", order.order_type()).to_uppercase(),
            side: format!("{:?}", order.side()).to_uppercase(),
            status: format!("{:?}", order.status()).to_uppercase(),
            quantity: order.quantity().value(),
            filled_quantity: order.filled_quantity().value(),
            remaining_quantity: order.remaining_quantity()
                .map(|q| q.value())
                .unwrap_or(0.0),
            price: order.price().map(|p| p.value()),
            average_fill_price: order.average_fill_price.map(|p| p.value()),
            stop_loss: order.stop_loss.map(|p| p.value()),
            take_profit: order.take_profit.map(|p| p.value()),
            created_at: order.created_at,
            updated_at: order.updated_at,
            exchange_order_id: order.exchange_order_id.clone(),
        }
    }
}

/// Simplified order summary for lists
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct OrderSummary {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

impl OrderSummary {
    pub fn from_domain(order: &Order) -> Self {
        OrderSummary {
            order_id: order.id().to_string(),
            symbol: order.symbol().to_string(),
            side: format!("{:?}", order.side()).to_uppercase(),
            quantity: order.quantity().value(),
            price: order.price().map(|p| p.value()),
            status: format!("{:?}", order.status()).to_uppercase(),
            created_at: order.created_at,
        }
    }
}

/// Order execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
// ELIMINATED: Duplicate ExecutionReport - use execution::reports::ExecutionReport

/// Order placement response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct PlaceOrderResponse {
    pub success: bool,
    pub order_id: String,
    pub exchange_order_id: Option<String>,
    pub message: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl PlaceOrderResponse {
    pub fn success(order_id: String, exchange_order_id: Option<String>) -> Self {
        PlaceOrderResponse {
            success: true,
            order_id,
            exchange_order_id,
            message: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn failure(message: String) -> Self {
        PlaceOrderResponse {
            success: false,
            order_id: String::new(),
            exchange_order_id: None,
            message: Some(message),
            timestamp: Utc::now(),
        }
    }
}

/// Cancel order response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct CancelOrderResponse {
    pub success: bool,
    pub order_id: String,
    pub message: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Batch order response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct BatchOrderResponse {
    pub total_orders: usize,
    pub successful_orders: usize,
    pub failed_orders: usize,
    pub results: Vec<PlaceOrderResponse>,
    pub timestamp: DateTime<Utc>,
}

/// Order statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct OrderStatisticsResponse {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
    pub active_orders: usize,
    pub total_volume: f64,
    pub win_rate: f64,
    pub average_profit: f64,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

/// Error response DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct ErrorResponse {
    pub error_code: String,
    pub message: String,
    pub details: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl ErrorResponse {
    pub fn new(code: String, message: String) -> Self {
        ErrorResponse {
            error_code: code,
            message,
            details: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::value_objects::{Symbol, Price, Quantity};
    
    #[test]
    fn should_convert_order_to_response() {
        let order = Order::limit(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Price::new(50000.0).unwrap(),
            Quantity::new(0.1).unwrap(),
            crate::domain::entities::TimeInForce::GTC,
        );
        
        let response = OrderResponse::from_domain(&order);
        
        assert_eq!(response.symbol, "BTC/USDT");
        assert_eq!(response.side, "BUY");
        assert_eq!(response.quantity, 0.1);
        assert_eq!(response.price, Some(50000.0));
        assert_eq!(response.status, "DRAFT");
    }
    
    #[test]
    fn should_create_success_response() {
        let response = PlaceOrderResponse::success(
            "ORDER123".to_string(),
            Some("EX456".to_string()),
        );
        
        assert!(response.success);
        assert_eq!(response.order_id, "ORDER123");
        assert_eq!(response.exchange_order_id, Some("EX456".to_string()));
        assert!(response.message.is_none());
    }
    
    #[test]
    fn should_create_failure_response() {
        let response = PlaceOrderResponse::failure(
            "Insufficient balance".to_string(),
        );
        
        assert!(!response.success);
        assert_eq!(response.message, Some("Insufficient balance".to_string()));
    }
    
    #[test]
    fn should_create_error_response_with_details() {
        let error = ErrorResponse::new(
            "INVALID_QUANTITY".to_string(),
            "Quantity below minimum".to_string(),
        ).with_details("Minimum quantity is 0.001 BTC".to_string());
        
        assert_eq!(error.error_code, "INVALID_QUANTITY");
        assert_eq!(error.message, "Quantity below minimum");
        assert_eq!(error.details, Some("Minimum quantity is 0.001 BTC".to_string()));
    }
}