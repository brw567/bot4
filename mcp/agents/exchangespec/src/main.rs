//! Bot4 ExchangeSpec Agent
//! Exchange connectivity, order management, and market data streaming

use anyhow::Result;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use rmcp::{
    server::{Server, ServerBuilder, ToolHandler},
    transport::DockerTransport,
    types::{Tool, ToolCall, ToolResult, Resource, Prompt},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use governor::{Quota, RateLimiter};

mod order_manager;
mod websocket_manager;
mod rate_limiter;
mod smart_router;

use order_manager::{OrderManager, Order, OrderStatus};
use websocket_manager::WebSocketManager;
use rate_limiter::ExchangeRateLimiter;
use smart_router::SmartRouter;

/// ExchangeSpec agent implementation
struct ExchangeSpecAgent {
    redis: ConnectionManager,
    order_manager: Arc<OrderManager>,
    websocket_manager: Arc<WebSocketManager>,
    rate_limiters: Arc<DashMap<String, Arc<ExchangeRateLimiter>>>,
    smart_router: Arc<SmartRouter>,
    supported_exchanges: Vec<String>,
}

impl ExchangeSpecAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Initialize supported exchanges
        let supported_exchanges = vec![
            "binance".to_string(),
            "kraken".to_string(),
            "coinbase".to_string(),
        ];
        
        // Create rate limiters for each exchange
        let rate_limiters = Arc::new(DashMap::new());
        for exchange in &supported_exchanges {
            rate_limiters.insert(
                exchange.clone(),
                Arc::new(ExchangeRateLimiter::new(exchange)?),
            );
        }
        
        Ok(Self {
            redis,
            order_manager: Arc::new(OrderManager::new()),
            websocket_manager: Arc::new(WebSocketManager::new()),
            rate_limiters,
            smart_router: Arc::new(SmartRouter::new()),
            supported_exchanges,
        })
    }
    
    /// Place order on exchange
    async fn place_order(&self, exchange: String, symbol: String, side: String, 
                        order_type: String, quantity: f64, price: Option<f64>) -> Result<ToolResult> {
        info!("Placing order on {}: {} {} {} @ {:?}", exchange, side, quantity, symbol, price);
        
        // Check rate limits
        if let Some(limiter) = self.rate_limiters.get(&exchange) {
            if !limiter.check_limit().await? {
                return Ok(ToolResult::Error("Rate limit exceeded, please wait".to_string()));
            }
        }
        
        // Create order
        let order = Order {
            id: uuid::Uuid::new_v4().to_string(),
            exchange: exchange.clone(),
            symbol: symbol.clone(),
            side: side.clone(),
            order_type: order_type.clone(),
            quantity: Decimal::from_f64_retain(quantity).unwrap_or(dec!(0)),
            price: price.map(|p| Decimal::from_f64_retain(p).unwrap_or(dec!(0))),
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: dec!(0),
            average_fill_price: None,
            fees: dec!(0),
        };
        
        // Submit order through order manager
        let result = self.order_manager.submit_order(order.clone()).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "order_id": result.order_id,
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "status": result.status,
            "message": result.message,
            "timestamp": result.timestamp,
        })))
    }
    
    /// Cancel order
    async fn cancel_order(&self, exchange: String, order_id: String) -> Result<ToolResult> {
        info!("Cancelling order {} on {}", order_id, exchange);
        
        let result = self.order_manager.cancel_order(&exchange, &order_id).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "order_id": order_id,
            "exchange": exchange,
            "cancelled": result.success,
            "message": result.message,
            "timestamp": Utc::now(),
        })))
    }
    
    /// Get order status
    async fn get_order_status(&self, exchange: String, order_id: String) -> Result<ToolResult> {
        info!("Getting status for order {} on {}", order_id, exchange);
        
        let status = self.order_manager.get_order_status(&exchange, &order_id).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "order_id": order_id,
            "exchange": exchange,
            "status": status.status,
            "filled_quantity": status.filled_quantity,
            "remaining_quantity": status.remaining_quantity,
            "average_fill_price": status.average_fill_price,
            "fees": status.fees,
            "last_update": status.last_update,
        })))
    }
    
    /// Subscribe to market data stream
    async fn subscribe_market_data(&self, exchange: String, symbols: Vec<String>, 
                                  data_types: Vec<String>) -> Result<ToolResult> {
        info!("Subscribing to {} market data for {:?} on {}", 
              data_types.join(","), symbols, exchange);
        
        let subscription_id = self.websocket_manager.subscribe(
            &exchange,
            symbols.clone(),
            data_types.clone()
        ).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "subscription_id": subscription_id,
            "exchange": exchange,
            "symbols": symbols,
            "data_types": data_types,
            "status": "connected",
            "message": "Successfully subscribed to market data",
        })))
    }
    
    /// Get orderbook snapshot
    async fn get_orderbook(&self, exchange: String, symbol: String, depth: u32) -> Result<ToolResult> {
        info!("Getting orderbook for {} on {} (depth: {})", symbol, exchange, depth);
        
        // Simulated orderbook data
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        let mid_price = 50000.0; // BTC example
        
        for i in 0..depth {
            let spread = 0.0001 * (i + 1) as f64;
            bids.push(vec![
                mid_price * (1.0 - spread),
                rand::random::<f64>() * 10.0,
            ]);
            asks.push(vec![
                mid_price * (1.0 + spread),
                rand::random::<f64>() * 10.0,
            ]);
        }
        
        Ok(ToolResult::Success(serde_json::json!({
            "exchange": exchange,
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "spread": asks[0][0] - bids[0][0],
            "mid_price": (asks[0][0] + bids[0][0]) / 2.0,
            "timestamp": Utc::now(),
        })))
    }
    
    /// Smart order routing
    async fn smart_route_order(&self, symbol: String, side: String, quantity: f64, 
                              slippage_tolerance: f64) -> Result<ToolResult> {
        info!("Smart routing order: {} {} {} (slippage: {}%)", 
              side, quantity, symbol, slippage_tolerance * 100.0);
        
        let routing_result = self.smart_router.route_order(
            &symbol,
            &side,
            Decimal::from_f64_retain(quantity).unwrap_or(dec!(0)),
            Decimal::from_f64_retain(slippage_tolerance).unwrap_or(dec!(0.001))
        ).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "symbol": symbol,
            "side": side,
            "total_quantity": quantity,
            "routing": routing_result.splits,
            "estimated_price": routing_result.estimated_average_price,
            "estimated_slippage": routing_result.estimated_slippage,
            "best_exchange": routing_result.best_exchange,
            "execution_plan": routing_result.execution_plan,
        })))
    }
    
    /// Get exchange status
    async fn get_exchange_status(&self, exchange: String) -> Result<ToolResult> {
        info!("Getting status for exchange: {}", exchange);
        
        // Check WebSocket connection
        let ws_connected = self.websocket_manager.is_connected(&exchange).await;
        
        // Check rate limit status
        let rate_limit_status = if let Some(limiter) = self.rate_limiters.get(&exchange) {
            limiter.get_status().await?
        } else {
            "unknown".to_string()
        };
        
        // Get recent order statistics
        let order_stats = self.order_manager.get_exchange_stats(&exchange).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "exchange": exchange,
            "websocket_connected": ws_connected,
            "rate_limit_status": rate_limit_status,
            "api_status": "operational",
            "order_stats": {
                "total_orders": order_stats.total_orders,
                "successful_orders": order_stats.successful_orders,
                "failed_orders": order_stats.failed_orders,
                "average_latency_ms": order_stats.average_latency_ms,
            },
            "supported_pairs": self.get_supported_pairs(&exchange),
            "timestamp": Utc::now(),
        })))
    }
    
    /// Calculate trading fees
    async fn calculate_fees(&self, exchange: String, symbol: String, 
                           volume: f64, is_maker: bool) -> Result<ToolResult> {
        info!("Calculating fees for {} on {} (volume: {}, maker: {})", 
              symbol, exchange, volume, is_maker);
        
        let fee_rate = match exchange.as_str() {
            "binance" => if is_maker { 0.001 } else { 0.001 },
            "kraken" => if is_maker { 0.0016 } else { 0.0026 },
            "coinbase" => if is_maker { 0.004 } else { 0.006 },
            _ => 0.002,
        };
        
        let fee_amount = volume * fee_rate;
        
        Ok(ToolResult::Success(serde_json::json!({
            "exchange": exchange,
            "symbol": symbol,
            "volume": volume,
            "is_maker": is_maker,
            "fee_rate": fee_rate,
            "fee_amount": fee_amount,
            "fee_currency": "USD",
            "tier": "standard",
            "discount_available": false,
        })))
    }
    
    fn get_supported_pairs(&self, exchange: &str) -> Vec<String> {
        match exchange {
            "binance" => vec![
                "BTC/USDT".to_string(),
                "ETH/USDT".to_string(),
                "SOL/USDT".to_string(),
                "BNB/USDT".to_string(),
            ],
            "kraken" => vec![
                "XBT/USD".to_string(),
                "ETH/USD".to_string(),
                "SOL/USD".to_string(),
                "ADA/USD".to_string(),
            ],
            "coinbase" => vec![
                "BTC-USD".to_string(),
                "ETH-USD".to_string(),
                "SOL-USD".to_string(),
                "MATIC-USD".to_string(),
            ],
            _ => vec![],
        }
    }
}

#[async_trait]
impl ToolHandler for ExchangeSpecAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "place_order" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("binance").to_string();
                let symbol = tool_call.arguments["symbol"].as_str().unwrap_or("").to_string();
                let side = tool_call.arguments["side"].as_str().unwrap_or("buy").to_string();
                let order_type = tool_call.arguments["order_type"].as_str().unwrap_or("limit").to_string();
                let quantity = tool_call.arguments["quantity"].as_f64().unwrap_or(0.0);
                let price = tool_call.arguments["price"].as_f64();
                self.place_order(exchange, symbol, side, order_type, quantity, price).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to place order: {}", e))
                })
            }
            "cancel_order" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("").to_string();
                let order_id = tool_call.arguments["order_id"].as_str().unwrap_or("").to_string();
                self.cancel_order(exchange, order_id).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to cancel order: {}", e))
                })
            }
            "get_order_status" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("").to_string();
                let order_id = tool_call.arguments["order_id"].as_str().unwrap_or("").to_string();
                self.get_order_status(exchange, order_id).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to get order status: {}", e))
                })
            }
            "subscribe_market_data" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("").to_string();
                let symbols = tool_call.arguments["symbols"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let data_types = tool_call.arguments["data_types"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_else(|| vec!["trades".to_string(), "orderbook".to_string()]);
                self.subscribe_market_data(exchange, symbols, data_types).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to subscribe: {}", e))
                })
            }
            "get_orderbook" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("").to_string();
                let symbol = tool_call.arguments["symbol"].as_str().unwrap_or("").to_string();
                let depth = tool_call.arguments["depth"].as_u64().unwrap_or(10) as u32;
                self.get_orderbook(exchange, symbol, depth).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to get orderbook: {}", e))
                })
            }
            "smart_route_order" => {
                let symbol = tool_call.arguments["symbol"].as_str().unwrap_or("").to_string();
                let side = tool_call.arguments["side"].as_str().unwrap_or("buy").to_string();
                let quantity = tool_call.arguments["quantity"].as_f64().unwrap_or(0.0);
                let slippage = tool_call.arguments["slippage_tolerance"].as_f64().unwrap_or(0.001);
                self.smart_route_order(symbol, side, quantity, slippage).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to route order: {}", e))
                })
            }
            "get_exchange_status" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("binance").to_string();
                self.get_exchange_status(exchange).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to get exchange status: {}", e))
                })
            }
            "calculate_fees" => {
                let exchange = tool_call.arguments["exchange"].as_str().unwrap_or("").to_string();
                let symbol = tool_call.arguments["symbol"].as_str().unwrap_or("").to_string();
                let volume = tool_call.arguments["volume"].as_f64().unwrap_or(0.0);
                let is_maker = tool_call.arguments["is_maker"].as_bool().unwrap_or(false);
                self.calculate_fees(exchange, symbol, volume, is_maker).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to calculate fees: {}", e))
                })
            }
            _ => ToolResult::Error(format!("Unknown tool: {}", tool_call.name))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .init();
    
    info!("Starting Bot4 ExchangeSpec Agent v1.0");
    
    // Create agent
    let agent = ExchangeSpecAgent::new().await?;
    
    // Define tools
    let tools = vec![
        Tool {
            name: "place_order".to_string(),
            description: "Place an order on a cryptocurrency exchange".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "enum": ["binance", "kraken", "coinbase"]},
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "order_type": {"type": "string", "enum": ["market", "limit", "stop_limit"]},
                    "quantity": {"type": "number"},
                    "price": {"type": "number"}
                },
                "required": ["exchange", "symbol", "side", "order_type", "quantity"]
            }),
        },
        Tool {
            name: "cancel_order".to_string(),
            description: "Cancel an existing order".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"},
                    "order_id": {"type": "string"}
                },
                "required": ["exchange", "order_id"]
            }),
        },
        Tool {
            name: "get_order_status".to_string(),
            description: "Get the current status of an order".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"},
                    "order_id": {"type": "string"}
                },
                "required": ["exchange", "order_id"]
            }),
        },
        Tool {
            name: "subscribe_market_data".to_string(),
            description: "Subscribe to real-time market data streams".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "data_types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exchange", "symbols"]
            }),
        },
        Tool {
            name: "get_orderbook".to_string(),
            description: "Get current orderbook snapshot".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"},
                    "symbol": {"type": "string"},
                    "depth": {"type": "integer", "default": 10}
                },
                "required": ["exchange", "symbol"]
            }),
        },
        Tool {
            name: "smart_route_order".to_string(),
            description: "Smart route order across multiple exchanges".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "quantity": {"type": "number"},
                    "slippage_tolerance": {"type": "number", "default": 0.001}
                },
                "required": ["symbol", "side", "quantity"]
            }),
        },
        Tool {
            name: "get_exchange_status".to_string(),
            description: "Get current exchange connectivity and health status".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"}
                },
                "required": ["exchange"]
            }),
        },
        Tool {
            name: "calculate_fees".to_string(),
            description: "Calculate trading fees for an order".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "exchange": {"type": "string"},
                    "symbol": {"type": "string"},
                    "volume": {"type": "number"},
                    "is_maker": {"type": "boolean", "default": false}
                },
                "required": ["exchange", "symbol", "volume"]
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("exchangespec-agent", "1.0.0")
        .with_tools(tools)
        .with_tool_handler(agent)
        .build()?;
    
    // Use Docker transport
    let transport = DockerTransport::new()?;
    server.run(transport).await?;
    
    Ok(())
}