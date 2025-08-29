// DTO: Order Request Data Transfer Objects
// For API input - separate from domain models
// Owner: Sam | Reviewer: Alex

use serde::{Deserialize, Serialize};
use validator::Validate;
use anyhow::{Result, bail};

use crate::domain::entities::{Order, OrderSide, OrderType, TimeInForce};
use crate::domain::value_objects::{Symbol, Price, Quantity};

/// Place order request DTO
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct PlaceOrderRequest {
    #[validate(length(min = 1, max = 20))]
    pub symbol: String,
    
    #[serde(rename = "type")]
    pub order_type: String,
    
    pub side: String,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub quantity: f64,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub price: Option<f64>,
    
    pub time_in_force: Option<String>,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub stop_loss: Option<f64>,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub take_profit: Option<f64>,
    
    #[validate(range(min = 0, max = 10000))]
    pub max_slippage_bps: Option<u32>,
}

impl PlaceOrderRequest {
    /// Convert DTO to domain Order
    pub fn to_domain(&self) -> Result<Order> {
        // Validate the DTO first
        self.validate()
            .map_err(|e| anyhow::anyhow!("Validation failed: {}", e))?;
        
        // Parse symbol
        let symbol = Symbol::new(&self.symbol)?;
        
        // Parse side
        let side = match self.side.to_uppercase().as_str() {
            "BUY" => OrderSide::Buy,
            "SELL" => OrderSide::Sell,
            _ => bail!("Invalid order side: {}", self.side),
        };
        
        // Parse order type
        let order_type = match self.order_type.to_uppercase().as_str() {
            "MARKET" => OrderType::Market,
            "LIMIT" => OrderType::Limit,
            "STOP_MARKET" => OrderType::StopMarket,
            "STOP_LIMIT" => OrderType::StopLimit,
            _ => bail!("Invalid order type: {}", self.order_type),
        };
        
        // Parse quantity
        let quantity = Quantity::new(self.quantity)?;
        
        // Create order based on type
        let mut order = match order_type {
            OrderType::Market => {
                Order::market(symbol, side, quantity)
            },
            OrderType::Limit => {
                let price = self.price
                    .ok_or_else(|| anyhow::anyhow!("Price required for limit orders"))?;
                let price = Price::new(price)?;
                
                let time_in_force = self.parse_time_in_force()?;
                
                Order::limit(symbol, side, price, quantity, time_in_force)
            },
            _ => bail!("Order type {} not yet implemented", self.order_type),
        };
        
        // Add risk parameters if provided
        if self.stop_loss.is_some() || self.take_profit.is_some() {
            let stop_loss = self.stop_loss
                .map(Price::new)
                .transpose()?;
            
            let take_profit = self.take_profit
                .map(Price::new)
                .transpose()?;
            
            order = order.with_risk_params(stop_loss, take_profit)?;
        }
        
        Ok(order)
    }
    
    fn parse_time_in_force(&self) -> Result<TimeInForce> {
        match self.time_in_force.as_deref().unwrap_or("GTC").to_uppercase().as_str() {
            "GTC" => Ok(TimeInForce::GTC),
            "IOC" => Ok(TimeInForce::IOC),
            "FOK" => Ok(TimeInForce::FOK),
            "GTX" => Ok(TimeInForce::GTX),
            _ => bail!("Invalid time in force: {:?}", self.time_in_force),
        }
    }
}

/// Cancel order request DTO
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct CancelOrderRequest {
    #[validate(length(min = 1, max = 100))]
    pub order_id: String,
    
    #[validate(length(max = 500))]
    pub reason: Option<String>,
}

/// Modify order request DTO
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct ModifyOrderRequest {
    #[validate(length(min = 1, max = 100))]
    pub order_id: String,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub new_price: Option<f64>,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub new_quantity: Option<f64>,
}

/// Batch order request for multiple orders
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct BatchOrderRequest {
    #[validate(length(min = 1, max = 100))]
    pub orders: Vec<PlaceOrderRequest>,
    
    pub stop_on_error: bool,
}

/// OCO (One-Cancels-Other) order request
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(rename_all = "camelCase")]
/// TODO: Add docs
pub struct OcoOrderRequest {
    #[validate(length(min = 1, max = 20))]
    pub symbol: String,
    
    pub side: String,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub quantity: f64,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub limit_price: f64,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub stop_price: f64,
    
    #[validate(range(min = 0.0, exclusive_min = true))]
    pub stop_limit_price: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn should_validate_place_order_request() {
        let request = PlaceOrderRequest {
            symbol: "BTC/USDT".to_string(),
            order_type: "LIMIT".to_string(),
            side: "BUY".to_string(),
            quantity: 0.1,
            price: Some(50000.0),
            time_in_force: Some("GTC".to_string()),
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None,
        };
        
        assert!(request.validate().is_ok());
    }
    
    #[test]
    fn should_reject_invalid_quantity() {
        let request = PlaceOrderRequest {
            symbol: "BTC/USDT".to_string(),
            order_type: "MARKET".to_string(),
            side: "BUY".to_string(),
            quantity: -0.1, // Invalid
            price: None,
            time_in_force: None,
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None,
        };
        
        assert!(request.validate().is_err());
    }
    
    #[test]
    fn should_convert_to_domain_order() {
        let request = PlaceOrderRequest {
            symbol: "BTC/USDT".to_string(),
            order_type: "LIMIT".to_string(),
            side: "BUY".to_string(),
            quantity: 0.1,
            price: Some(50000.0),
            time_in_force: Some("GTC".to_string()),
            stop_loss: Some(49000.0),
            take_profit: Some(51000.0),
            max_slippage_bps: None,
        };
        
        let order = request.to_domain().unwrap();
        
        assert_eq!(order.symbol().as_str(), "BTC/USDT");
        assert_eq!(order.side(), OrderSide::Buy);
        assert_eq!(order.quantity().value(), 0.1);
        assert_eq!(order.price().unwrap().value(), 50000.0);
    }
    
    #[test]
    fn should_require_price_for_limit_orders() {
        let request = PlaceOrderRequest {
            symbol: "BTC/USDT".to_string(),
            order_type: "LIMIT".to_string(),
            side: "BUY".to_string(),
            quantity: 0.1,
            price: None, // Missing price
            time_in_force: Some("GTC".to_string()),
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None,
        };
        
        let result = request.to_domain();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Price required"));
    }
    
    #[test]
    fn should_parse_time_in_force() {
        let request = PlaceOrderRequest {
            symbol: "BTC/USDT".to_string(),
            order_type: "LIMIT".to_string(),
            side: "SELL".to_string(),
            quantity: 0.1,
            price: Some(50000.0),
            time_in_force: Some("IOC".to_string()),
            stop_loss: None,
            take_profit: None,
            max_slippage_bps: None,
        };
        
        let order = request.to_domain().unwrap();
        assert_eq!(order.time_in_force, TimeInForce::IOC);
    }
}