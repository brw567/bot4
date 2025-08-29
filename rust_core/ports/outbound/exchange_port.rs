pub use domain_types::market_data::{OrderBook, OrderBookLevel, OrderBookUpdate};

pub use domain_types::trade::{Trade, TradeId, TradeError};

// Port: Exchange Interface
// Defines the contract for exchange interactions
// Owner: Casey | Reviewer: Sam

use async_trait::async_trait;
use anyhow::Result;
use std::collections::HashMap;

use crate::domain::entities::{Order, OrderId, OrderStatus};
use crate::domain::value_objects::{Symbol, Price, Quantity};

/// Market depth level
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct OrderBookLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: usize,
}

/// Order book snapshot
#[derive(Debug, Clone)]

/// Trade/tick data
#[derive(Debug, Clone)]

/// Balance information
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct Balance {
    pub asset: String,
    pub free: Quantity,
    pub locked: Quantity,
}

/// Exchange capabilities
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ExchangeCapabilities {
    pub supports_oco: bool,
    pub supports_reduce_only: bool,
    pub supports_post_only: bool,
    pub supports_iceberg: bool,
    pub supports_trailing_stop: bool,
    pub max_orders_per_second: u32,
}

/// Port interface for exchange operations
/// This is the contract that all exchange adapters must implement
#[async_trait]
pub trait ExchangePort: Send + Sync {
    // Order Management
    
    /// Place an order on the exchange
    async fn place_order(&self, order: &Order) -> Result<String>; // Returns exchange_order_id
    
    /// Cancel an order
    async fn cancel_order(&self, order_id: &OrderId) -> Result<()>;
    
    /// Modify an existing order
    async fn modify_order(&self, order_id: &OrderId, new_price: Option<Price>, new_quantity: Option<Quantity>) -> Result<()>;
    
    /// Get order status
    async fn get_order_status(&self, order_id: &OrderId) -> Result<OrderStatus>;
    
    /// Get all open orders
    async fn get_open_orders(&self, symbol: Option<&Symbol>) -> Result<Vec<Order>>;
    
    /// Get order history
    async fn get_order_history(&self, symbol: &Symbol, limit: usize) -> Result<Vec<Order>>;
    
    // Market Data
    
    /// Get current order book
    async fn get_order_book(&self, symbol: &Symbol, depth: usize) -> Result<OrderBook>;
    
    /// Get recent trades
    async fn get_recent_trades(&self, symbol: &Symbol, limit: usize) -> Result<Vec<Trade>>;
    
    /// Get current ticker price
    async fn get_ticker(&self, symbol: &Symbol) -> Result<(Price, Price)>; // (bid, ask)
    
    // Account Management
    
    /// Get account balances
    async fn get_balances(&self) -> Result<HashMap<String, Balance>>;
    
    /// Get trading fees for a symbol
    async fn get_trading_fees(&self, symbol: &Symbol) -> Result<(f64, f64)>; // (maker_fee, taker_fee)
    
    // Exchange Information
    
    /// Get exchange capabilities
    async fn get_capabilities(&self) -> Result<ExchangeCapabilities>;
    
    /// Check if exchange is connected and operational
    async fn health_check(&self) -> Result<bool>;
    
    /// Get rate limit status
    async fn get_rate_limit_status(&self) -> Result<(u32, u32)>; // (used, limit)
}

/// Extended exchange features (optional)
#[async_trait]
pub trait ExtendedExchangePort: ExchangePort {
    /// Place OCO (One-Cancels-Other) order
    async fn place_oco_order(
        &self,
        symbol: &Symbol,
        side: crate::domain::entities::OrderSide,
        quantity: Quantity,
        price: Price,
        stop_price: Price,
        stop_limit_price: Option<Price>,
    ) -> Result<(String, String)>; // Returns (order_id, stop_order_id)
    
    /// Place reduce-only order
    async fn place_reduce_only_order(&self, order: &Order) -> Result<String>;
    
    /// Place post-only order
    async fn place_post_only_order(&self, order: &Order) -> Result<String>;
    
    /// Place iceberg order
    async fn place_iceberg_order(&self, order: &Order, visible_quantity: Quantity) -> Result<String>;
}