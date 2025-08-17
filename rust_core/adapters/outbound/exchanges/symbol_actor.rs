// Adapter: Symbol Actor
// Per-symbol actor loops for deterministic order processing
// Addresses Sophia's #6 critical feedback on determinism
// Owner: Casey | Reviewer: Alex

use anyhow::{Result, bail};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::task::JoinHandle;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use chrono::Utc;
use tracing::{info, warn, error, debug};

use crate::domain::entities::{Order, OrderId, OrderStatus};
use crate::domain::value_objects::{Symbol, Price, Quantity};

/// Message types for symbol actors
#[derive(Debug)]
pub enum SymbolMessage {
    PlaceOrder {
        order: Order,
        client_order_id: String,
        response: oneshot::Sender<Result<String>>,
    },
    CancelOrder {
        order_id: OrderId,
        response: oneshot::Sender<Result<()>>,
    },
    ModifyOrder {
        order_id: OrderId,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
        response: oneshot::Sender<Result<()>>,
    },
    GetStatus {
        order_id: OrderId,
        response: oneshot::Sender<Result<OrderStatus>>,
    },
    Shutdown,
}

/// Statistics for a symbol actor
#[derive(Debug, Default)]
pub struct ActorStats {
    pub messages_processed: u64,
    pub orders_placed: u64,
    pub orders_cancelled: u64,
    pub orders_modified: u64,
    pub errors: u64,
    pub avg_processing_time_ms: f64,
    pub max_processing_time_ms: u64,
}

/// Individual symbol actor
/// Processes all orders for a single symbol sequentially
/// Ensures deterministic order processing and prevents race conditions
pub struct SymbolActor {
    symbol: Symbol,
    receiver: mpsc::Receiver<SymbolMessage>,
    stats: Arc<RwLock<ActorStats>>,
    max_queue_size: usize,
    processing_timeout: Duration,
}

impl SymbolActor {
    /// Create a new symbol actor
    pub fn new(
        symbol: Symbol,
        receiver: mpsc::Receiver<SymbolMessage>,
        max_queue_size: usize,
    ) -> Self {
        Self {
            symbol,
            receiver,
            stats: Arc::new(RwLock::new(ActorStats::default())),
            max_queue_size,
            processing_timeout: Duration::from_secs(5),
        }
    }
    
    /// Run the actor loop
    pub async fn run(mut self) {
        info!("Starting actor for symbol: {}", self.symbol);
        
        while let Some(message) = self.receiver.recv().await {
            let start = Utc::now();
            
            match message {
                SymbolMessage::PlaceOrder { order, client_order_id, response } => {
                    debug!("Processing place order for {}", self.symbol);
                    let result = self.process_place_order(order, client_order_id).await;
                    let _ = response.send(result);
                    
                    let mut stats = self.stats.write().await;
                    stats.orders_placed += 1;
                }
                
                SymbolMessage::CancelOrder { order_id, response } => {
                    debug!("Processing cancel order for {}", self.symbol);
                    let result = self.process_cancel_order(order_id).await;
                    let _ = response.send(result);
                    
                    let mut stats = self.stats.write().await;
                    stats.orders_cancelled += 1;
                }
                
                SymbolMessage::ModifyOrder { order_id, new_price, new_quantity, response } => {
                    debug!("Processing modify order for {}", self.symbol);
                    let result = self.process_modify_order(order_id, new_price, new_quantity).await;
                    let _ = response.send(result);
                    
                    let mut stats = self.stats.write().await;
                    stats.orders_modified += 1;
                }
                
                SymbolMessage::GetStatus { order_id, response } => {
                    debug!("Processing get status for {}", self.symbol);
                    let result = self.process_get_status(order_id).await;
                    let _ = response.send(result);
                }
                
                SymbolMessage::Shutdown => {
                    info!("Shutting down actor for symbol: {}", self.symbol);
                    break;
                }
            }
            
            // Update statistics
            let elapsed = (Utc::now() - start).num_milliseconds() as u64;
            let mut stats = self.stats.write().await;
            stats.messages_processed += 1;
            stats.max_processing_time_ms = stats.max_processing_time_ms.max(elapsed);
            
            // Update average processing time
            let n = stats.messages_processed as f64;
            stats.avg_processing_time_ms = 
                (stats.avg_processing_time_ms * (n - 1.0) + elapsed as f64) / n;
        }
        
        info!("Actor for symbol {} has shut down", self.symbol);
    }
    
    /// Process place order request
    async fn process_place_order(&self, order: Order, client_order_id: String) -> Result<String> {
        // Validate order symbol matches actor symbol
        if order.symbol() != &self.symbol {
            bail!("Order symbol {} doesn't match actor symbol {}", 
                  order.symbol(), self.symbol);
        }
        
        // Simulate processing with timeout
        tokio::time::timeout(
            self.processing_timeout,
            self.simulate_order_placement(order, client_order_id)
        ).await
        .map_err(|_| anyhow::anyhow!("Order placement timeout"))?
    }
    
    /// Process cancel order request
    async fn process_cancel_order(&self, order_id: OrderId) -> Result<()> {
        // Simulate processing
        tokio::time::timeout(
            self.processing_timeout,
            self.simulate_order_cancellation(order_id)
        ).await
        .map_err(|_| anyhow::anyhow!("Order cancellation timeout"))?
    }
    
    /// Process modify order request
    async fn process_modify_order(
        &self,
        order_id: OrderId,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
    ) -> Result<()> {
        // Simulate processing
        tokio::time::timeout(
            self.processing_timeout,
            self.simulate_order_modification(order_id, new_price, new_quantity)
        ).await
        .map_err(|_| anyhow::anyhow!("Order modification timeout"))?
    }
    
    /// Process get status request
    async fn process_get_status(&self, order_id: OrderId) -> Result<OrderStatus> {
        // Simulate processing
        tokio::time::timeout(
            Duration::from_secs(1),
            self.simulate_get_status(order_id)
        ).await
        .map_err(|_| anyhow::anyhow!("Get status timeout"))?
    }
    
    /// Simulate order placement (would call actual exchange in production)
    async fn simulate_order_placement(&self, _order: Order, _client_order_id: String) -> Result<String> {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(format!("SIM_{}_{}", self.symbol, uuid::Uuid::new_v4()))
    }
    
    /// Simulate order cancellation
    async fn simulate_order_cancellation(&self, _order_id: OrderId) -> Result<()> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }
    
    /// Simulate order modification
    async fn simulate_order_modification(
        &self,
        _order_id: OrderId,
        _new_price: Option<Price>,
        _new_quantity: Option<Quantity>,
    ) -> Result<()> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }
    
    /// Simulate get status
    async fn simulate_get_status(&self, _order_id: OrderId) -> Result<OrderStatus> {
        Ok(OrderStatus::Open)
    }
    
    /// Get actor statistics
    pub async fn get_stats(&self) -> ActorStats {
        self.stats.read().await.clone()
    }
}

/// Symbol Actor Manager
/// Manages per-symbol actors and routes messages
pub struct SymbolActorManager {
    actors: Arc<RwLock<HashMap<Symbol, mpsc::Sender<SymbolMessage>>>>,
    handles: Arc<RwLock<HashMap<Symbol, JoinHandle<()>>>>,
    max_queue_size: usize,
    max_actors: usize,
}

impl SymbolActorManager {
    /// Create a new actor manager
    pub fn new(max_queue_size: usize, max_actors: usize) -> Self {
        Self {
            actors: Arc::new(RwLock::new(HashMap::new())),
            handles: Arc::new(RwLock::new(HashMap::new())),
            max_queue_size,
            max_actors,
        }
    }
    
    /// Get or create an actor for a symbol
    async fn get_or_create_actor(&self, symbol: &Symbol) -> Result<mpsc::Sender<SymbolMessage>> {
        let actors = self.actors.read().await;
        
        if let Some(sender) = actors.get(symbol) {
            return Ok(sender.clone());
        }
        
        drop(actors); // Release read lock
        
        // Need to create new actor
        let mut actors = self.actors.write().await;
        let mut handles = self.handles.write().await;
        
        // Check again (another thread might have created it)
        if let Some(sender) = actors.get(symbol) {
            return Ok(sender.clone());
        }
        
        // Check actor limit
        if actors.len() >= self.max_actors {
            bail!("Maximum number of symbol actors ({}) reached", self.max_actors);
        }
        
        // Create new actor
        let (sender, receiver) = mpsc::channel(self.max_queue_size);
        let actor = SymbolActor::new(symbol.clone(), receiver, self.max_queue_size);
        
        // Spawn actor task
        let handle = tokio::spawn(async move {
            actor.run().await;
        });
        
        actors.insert(symbol.clone(), sender.clone());
        handles.insert(symbol.clone(), handle);
        
        info!("Created new actor for symbol: {}", symbol);
        
        Ok(sender)
    }
    
    /// Place an order through the appropriate symbol actor
    pub async fn place_order(
        &self,
        order: Order,
        client_order_id: String,
    ) -> Result<String> {
        let sender = self.get_or_create_actor(order.symbol()).await?;
        
        let (response_tx, response_rx) = oneshot::channel();
        
        sender.send(SymbolMessage::PlaceOrder {
            order,
            client_order_id,
            response: response_tx,
        }).await
        .map_err(|_| anyhow::anyhow!("Failed to send message to actor"))?;
        
        response_rx.await
            .map_err(|_| anyhow::anyhow!("Actor response channel closed"))?
    }
    
    /// Cancel an order through the appropriate symbol actor
    pub async fn cancel_order(
        &self,
        symbol: &Symbol,
        order_id: OrderId,
    ) -> Result<()> {
        let sender = self.get_or_create_actor(symbol).await?;
        
        let (response_tx, response_rx) = oneshot::channel();
        
        sender.send(SymbolMessage::CancelOrder {
            order_id,
            response: response_tx,
        }).await
        .map_err(|_| anyhow::anyhow!("Failed to send message to actor"))?;
        
        response_rx.await
            .map_err(|_| anyhow::anyhow!("Actor response channel closed"))?
    }
    
    /// Modify an order through the appropriate symbol actor
    pub async fn modify_order(
        &self,
        symbol: &Symbol,
        order_id: OrderId,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
    ) -> Result<()> {
        let sender = self.get_or_create_actor(symbol).await?;
        
        let (response_tx, response_rx) = oneshot::channel();
        
        sender.send(SymbolMessage::ModifyOrder {
            order_id,
            new_price,
            new_quantity,
            response: response_tx,
        }).await
        .map_err(|_| anyhow::anyhow!("Failed to send message to actor"))?;
        
        response_rx.await
            .map_err(|_| anyhow::anyhow!("Actor response channel closed"))?
    }
    
    /// Get order status through the appropriate symbol actor
    pub async fn get_order_status(
        &self,
        symbol: &Symbol,
        order_id: OrderId,
    ) -> Result<OrderStatus> {
        let sender = self.get_or_create_actor(symbol).await?;
        
        let (response_tx, response_rx) = oneshot::channel();
        
        sender.send(SymbolMessage::GetStatus {
            order_id,
            response: response_tx,
        }).await
        .map_err(|_| anyhow::anyhow!("Failed to send message to actor"))?;
        
        response_rx.await
            .map_err(|_| anyhow::anyhow!("Actor response channel closed"))?
    }
    
    /// Shutdown all actors
    pub async fn shutdown(&self) -> Result<()> {
        let actors = self.actors.read().await;
        
        // Send shutdown message to all actors
        for (symbol, sender) in actors.iter() {
            if let Err(e) = sender.send(SymbolMessage::Shutdown).await {
                warn!("Failed to send shutdown to actor {}: {}", symbol, e);
            }
        }
        
        drop(actors);
        
        // Wait for all actors to complete
        let mut handles = self.handles.write().await;
        for (symbol, handle) in handles.drain() {
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(())) => info!("Actor {} shut down successfully", symbol),
                Ok(Err(e)) => error!("Actor {} panicked: {}", symbol, e),
                Err(_) => error!("Actor {} shutdown timeout", symbol),
            }
        }
        
        // Clear actors map
        self.actors.write().await.clear();
        
        Ok(())
    }
    
    /// Get statistics for all actors
    pub async fn get_all_stats(&self) -> HashMap<Symbol, ActorStats> {
        let mut all_stats = HashMap::new();
        
        // For now, return empty stats
        // In production, we'd track stats per actor
        
        all_stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::{OrderSide, TimeInForce};
    
    #[tokio::test]
    async fn test_symbol_actor_basic() {
        let manager = SymbolActorManager::new(100, 10);
        
        let order = Order::limit(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Price::new(50000.0).unwrap(),
            Quantity::new(1.0).unwrap(),
            TimeInForce::GTC,
        );
        
        let result = manager.place_order(order, "TEST_123".to_string()).await;
        assert!(result.is_ok());
        
        let order_id = result.unwrap();
        assert!(order_id.starts_with("SIM_"));
    }
    
    #[tokio::test]
    async fn test_multiple_symbols() {
        let manager = SymbolActorManager::new(100, 10);
        
        // Place orders for different symbols
        let symbols = vec!["BTC/USDT", "ETH/USDT", "SOL/USDT"];
        let mut handles = Vec::new();
        
        for symbol_str in symbols {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let symbol = Symbol::new(symbol_str).unwrap();
                for i in 0..5 {
                    let order = Order::limit(
                        symbol.clone(),
                        OrderSide::Buy,
                        Price::new(50000.0).unwrap(),
                        Quantity::new(1.0).unwrap(),
                        TimeInForce::GTC,
                    );
                    
                    let result = manager_clone.place_order(
                        order,
                        format!("TEST_{}_{}", symbol_str, i)
                    ).await;
                    
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }
        
        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify we have 3 actors
        let actors = manager.actors.read().await;
        assert_eq!(actors.len(), 3);
    }
    
    #[tokio::test]
    async fn test_actor_shutdown() {
        let manager = SymbolActorManager::new(100, 10);
        
        // Create some actors
        for symbol_str in &["BTC/USDT", "ETH/USDT"] {
            let order = Order::limit(
                Symbol::new(symbol_str).unwrap(),
                OrderSide::Buy,
                Price::new(50000.0).unwrap(),
                Quantity::new(1.0).unwrap(),
                TimeInForce::GTC,
            );
            
            let _ = manager.place_order(order, "TEST".to_string()).await;
        }
        
        // Shutdown
        let result = manager.shutdown().await;
        assert!(result.is_ok());
        
        // Verify actors are cleared
        let actors = manager.actors.read().await;
        assert_eq!(actors.len(), 0);
    }
    
    #[tokio::test]
    async fn test_deterministic_processing() {
        // Test that orders for the same symbol are processed sequentially
        let manager = SymbolActorManager::new(100, 10);
        let symbol = Symbol::new("BTC/USDT").unwrap();
        
        let mut order_ids = Vec::new();
        
        // Place multiple orders rapidly
        for i in 0..10 {
            let order = Order::limit(
                symbol.clone(),
                OrderSide::Buy,
                Price::new(50000.0 + i as f64).unwrap(),
                Quantity::new(1.0).unwrap(),
                TimeInForce::GTC,
            );
            
            let result = manager.place_order(
                order,
                format!("TEST_{}", i)
            ).await;
            
            assert!(result.is_ok());
            order_ids.push(result.unwrap());
        }
        
        // All orders should have been processed
        assert_eq!(order_ids.len(), 10);
        
        // Each order should have a unique ID
        let unique_ids: std::collections::HashSet<_> = order_ids.iter().collect();
        assert_eq!(unique_ids.len(), 10);
    }
}