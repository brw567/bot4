//! WebSocket connection management for real-time market data

use anyhow::{Result, bail};
use dashmap::DashMap;
use futures::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Subscription {
    pub id: String,
    pub exchange: String,
    pub symbols: Vec<String>,
    pub data_types: Vec<String>,
    pub active: bool,
}

pub struct WebSocketManager {
    connections: Arc<DashMap<String, ConnectionState>>,
    subscriptions: Arc<DashMap<String, Subscription>>,
}

struct ConnectionState {
    url: String,
    connected: bool,
    reconnect_attempts: u32,
    sender: mpsc::Sender<Message>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
            subscriptions: Arc::new(DashMap::new()),
        }
    }
    
    pub async fn subscribe(&self, exchange: &str, symbols: Vec<String>, 
                           data_types: Vec<String>) -> Result<String> {
        // Create subscription ID
        let subscription_id = Uuid::new_v4().to_string();
        
        // Create subscription
        let subscription = Subscription {
            id: subscription_id.clone(),
            exchange: exchange.to_string(),
            symbols: symbols.clone(),
            data_types: data_types.clone(),
            active: true,
        };
        
        // Store subscription
        self.subscriptions.insert(subscription_id.clone(), subscription);
        
        // Ensure connection exists
        if !self.connections.contains_key(exchange) {
            self.connect_exchange(exchange).await?;
        }
        
        // Send subscription message
        self.send_subscription_message(exchange, &symbols, &data_types).await?;
        
        info!("Created subscription {} for {} on {}", subscription_id, symbols.join(","), exchange);
        
        Ok(subscription_id)
    }
    
    pub async fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        if let Some((_, subscription)) = self.subscriptions.remove(subscription_id) {
            // Send unsubscribe message
            self.send_unsubscription_message(&subscription.exchange, &subscription.symbols).await?;
            info!("Removed subscription {}", subscription_id);
        }
        
        Ok(())
    }
    
    pub async fn is_connected(&self, exchange: &str) -> bool {
        self.connections.get(exchange)
            .map(|conn| conn.connected)
            .unwrap_or(false)
    }
    
    pub async fn reconnect(&self, exchange: &str) -> Result<()> {
        info!("Reconnecting to {}", exchange);
        
        // Remove old connection
        self.connections.remove(exchange);
        
        // Create new connection
        self.connect_exchange(exchange).await?;
        
        // Resubscribe all active subscriptions for this exchange
        let subs_to_restore: Vec<Subscription> = self.subscriptions.iter()
            .filter(|s| s.exchange == exchange && s.active)
            .map(|s| s.value().clone())
            .collect();
        
        for sub in subs_to_restore {
            self.send_subscription_message(exchange, &sub.symbols, &sub.data_types).await?;
        }
        
        Ok(())
    }
    
    async fn connect_exchange(&self, exchange: &str) -> Result<()> {
        let url = self.get_websocket_url(exchange)?;
        
        // Create message channel
        let (tx, mut rx) = mpsc::channel(100);
        
        // Store connection state
        self.connections.insert(exchange.to_string(), ConnectionState {
            url: url.clone(),
            connected: false,
            reconnect_attempts: 0,
            sender: tx.clone(),
        });
        
        // Spawn connection handler
        let exchange_name = exchange.to_string();
        let connections = self.connections.clone();
        
        tokio::spawn(async move {
            if let Err(e) = Self::handle_connection(&exchange_name, &url, tx, rx, connections).await {
                error!("WebSocket connection error for {}: {}", exchange_name, e);
            }
        });
        
        // Wait for connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        Ok(())
    }
    
    async fn handle_connection(exchange: &str, url: &str, 
                               sender: mpsc::Sender<Message>,
                               mut receiver: mpsc::Receiver<Message>,
                               connections: Arc<DashMap<String, ConnectionState>>) -> Result<()> {
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();
        
        info!("Connected to {} WebSocket", exchange);
        
        // Update connection state
        if let Some(mut conn) = connections.get_mut(exchange) {
            conn.connected = true;
            conn.reconnect_attempts = 0;
        }
        
        // Handle messages
        loop {
            tokio::select! {
                // Receive from WebSocket
                Some(msg) = read.next() => {
                    match msg {
                        Ok(Message::Text(text)) => {
                            // Process market data
                            if let Err(e) = Self::process_market_data(exchange, &text).await {
                                warn!("Error processing market data: {}", e);
                            }
                        }
                        Ok(Message::Close(_)) => {
                            info!("WebSocket closed for {}", exchange);
                            break;
                        }
                        Ok(Message::Ping(data)) => {
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                error!("Failed to send pong: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("WebSocket error for {}: {}", exchange, e);
                            break;
                        }
                        _ => {}
                    }
                }
                
                // Send to WebSocket
                Some(msg) = receiver.recv() => {
                    if let Err(e) = write.send(msg).await {
                        error!("Failed to send message: {}", e);
                        break;
                    }
                }
                
                else => break,
            }
        }
        
        // Update connection state
        if let Some(mut conn) = connections.get_mut(exchange) {
            conn.connected = false;
            conn.reconnect_attempts += 1;
        }
        
        Ok(())
    }
    
    async fn process_market_data(exchange: &str, data: &str) -> Result<()> {
        // Parse and process market data
        // In production, would parse exchange-specific formats
        
        // Example: Parse JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(event_type) = json["type"].as_str() {
                match event_type {
                    "trade" => {
                        // Process trade data
                        info!("Trade update from {}: {}", exchange, data);
                    }
                    "orderbook" => {
                        // Process orderbook update
                        info!("Orderbook update from {}: {}", exchange, data);
                    }
                    "ticker" => {
                        // Process ticker update
                        info!("Ticker update from {}: {}", exchange, data);
                    }
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    async fn send_subscription_message(&self, exchange: &str, symbols: &[String], 
                                      data_types: &[String]) -> Result<()> {
        if let Some(conn) = self.connections.get(exchange) {
            let msg = self.build_subscription_message(exchange, symbols, data_types)?;
            conn.sender.send(Message::Text(msg)).await?;
        } else {
            bail!("No connection to {}", exchange);
        }
        
        Ok(())
    }
    
    async fn send_unsubscription_message(&self, exchange: &str, symbols: &[String]) -> Result<()> {
        if let Some(conn) = self.connections.get(exchange) {
            let msg = self.build_unsubscription_message(exchange, symbols)?;
            conn.sender.send(Message::Text(msg)).await?;
        }
        
        Ok(())
    }
    
    fn get_websocket_url(&self, exchange: &str) -> Result<String> {
        let url = match exchange {
            "binance" => "wss://stream.binance.com:9443/ws",
            "kraken" => "wss://ws.kraken.com",
            "coinbase" => "wss://ws-feed.exchange.coinbase.com",
            _ => bail!("Unsupported exchange: {}", exchange),
        };
        
        Ok(url.to_string())
    }
    
    fn build_subscription_message(&self, exchange: &str, symbols: &[String], 
                                 data_types: &[String]) -> Result<String> {
        let msg = match exchange {
            "binance" => {
                // Binance subscription format
                let streams: Vec<String> = symbols.iter()
                    .flat_map(|symbol| {
                        data_types.iter().map(|dtype| {
                            format!("{}@{}", symbol.to_lowercase().replace("/", ""), 
                                   self.map_data_type_binance(dtype))
                        })
                    })
                    .collect();
                
                serde_json::json!({
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1
                }).to_string()
            }
            "kraken" => {
                // Kraken subscription format
                serde_json::json!({
                    "event": "subscribe",
                    "pair": symbols,
                    "subscription": {
                        "name": data_types.get(0).unwrap_or(&"ticker".to_string())
                    }
                }).to_string()
            }
            "coinbase" => {
                // Coinbase subscription format
                serde_json::json!({
                    "type": "subscribe",
                    "product_ids": symbols,
                    "channels": data_types
                }).to_string()
            }
            _ => bail!("Unsupported exchange: {}", exchange),
        };
        
        Ok(msg)
    }
    
    fn build_unsubscription_message(&self, exchange: &str, symbols: &[String]) -> Result<String> {
        let msg = match exchange {
            "binance" => {
                serde_json::json!({
                    "method": "UNSUBSCRIBE",
                    "params": symbols,
                    "id": 1
                }).to_string()
            }
            "kraken" => {
                serde_json::json!({
                    "event": "unsubscribe",
                    "pair": symbols
                }).to_string()
            }
            "coinbase" => {
                serde_json::json!({
                    "type": "unsubscribe",
                    "product_ids": symbols
                }).to_string()
            }
            _ => bail!("Unsupported exchange: {}", exchange),
        };
        
        Ok(msg)
    }
    
    fn map_data_type_binance(&self, data_type: &str) -> &str {
        match data_type {
            "trades" => "trade",
            "orderbook" => "depth",
            "ticker" => "ticker",
            _ => "trade",
        }
    }
    
    pub async fn get_active_subscriptions(&self) -> Vec<Subscription> {
        self.subscriptions.iter()
            .filter(|s| s.active)
            .map(|s| s.value().clone())
            .collect()
    }
    
    pub async fn pause_subscription(&self, subscription_id: &str) -> Result<()> {
        if let Some(mut sub) = self.subscriptions.get_mut(subscription_id) {
            sub.active = false;
            Ok(())
        } else {
            bail!("Subscription not found")
        }
    }
    
    pub async fn resume_subscription(&self, subscription_id: &str) -> Result<()> {
        if let Some(mut sub) = self.subscriptions.get_mut(subscription_id) {
            sub.active = true;
            
            // Resend subscription message
            self.send_subscription_message(&sub.exchange, &sub.symbols, &sub.data_types).await?;
            
            Ok(())
        } else {
            bail!("Subscription not found")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_subscription() {
        let manager = WebSocketManager::new();
        
        // Note: This test won't actually connect to exchanges
        // In a real test, you'd use a mock WebSocket server
        
        let result = manager.subscribe(
            "binance",
            vec!["BTC/USDT".to_string()],
            vec!["trades".to_string()]
        ).await;
        
        // Will fail without actual connection, but tests the structure
        assert!(result.is_err());
    }
    
    #[test]
    fn test_url_generation() {
        let manager = WebSocketManager::new();
        
        let binance_url = manager.get_websocket_url("binance").unwrap();
        assert!(binance_url.contains("binance"));
        
        let kraken_url = manager.get_websocket_url("kraken").unwrap();
        assert!(kraken_url.contains("kraken"));
    }
}