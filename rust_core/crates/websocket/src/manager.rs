// WebSocket Manager - Manages multiple WebSocket connections
// Handles connection pooling, load balancing, and failover

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::{mpsc, broadcast};
use tokio::time::interval;
use tracing::{error, info, warn};
use crate::client::{WebSocketClient, WebSocketConfig, WebSocketStats};
use crate::message::{Message, MessageType};

/// Manages multiple WebSocket connections with load balancing
pub struct WebSocketManager {
    connections: Arc<DashMap<String, Arc<WebSocketClient>>>,
    message_router: Arc<MessageRouter>,
    stats_collector: Arc<StatsCollector>,
    is_running: Arc<AtomicBool>,
}

use std::sync::atomic::AtomicBool;

impl Default for WebSocketManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
            message_router: Arc::new(MessageRouter::new()),
            stats_collector: Arc::new(StatsCollector::new()),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Add a new WebSocket connection
    pub async fn add_connection(
        &self,
        name: String,
        config: WebSocketConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = Arc::new(WebSocketClient::new(config));
        
        // Connect the client
        client.connect().await?;
        
        // Store the connection
        self.connections.insert(name.clone(), client.clone());
        
        info!("Added WebSocket connection: {}", name);
        Ok(())
    }
    
    /// Remove a connection
    pub async fn remove_connection(&self, name: &str) {
        if let Some((_, client)) = self.connections.remove(name) {
            client.disconnect().await;
            info!("Removed WebSocket connection: {}", name);
        }
    }
    
    /// Get a specific connection
    pub fn get_connection(&self, name: &str) -> Option<Arc<WebSocketClient>> {
        self.connections.get(name).map(|entry| entry.clone())
    }
    
    /// Send message to a specific connection
    pub async fn send_to(
        &self,
        connection_name: &str,
        message: Message,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(client) = self.get_connection(connection_name) {
            client.send(message).await?;
            Ok(())
        } else {
            Err(format!("Connection not found: {}", connection_name).into())
        }
    }
    
    /// Broadcast message to all connections
    pub async fn broadcast(&self, message: Message) -> Vec<(String, Result<(), String>)> {
        let mut results = Vec::new();
        
        for entry in self.connections.iter() {
            let name = entry.key().clone();
            let client = entry.value().clone();
            
            let result = match client.send(message.clone()).await {
                Ok(_) => Ok(()),
                Err(e) => Err(e.to_string()),
            };
            
            results.push((name, result));
        }
        
        results
    }
    
    /// Get statistics for all connections
    pub fn get_all_stats(&self) -> HashMap<String, WebSocketStats> {
        let mut stats = HashMap::new();
        
        for entry in self.connections.iter() {
            stats.insert(entry.key().clone(), entry.value().stats());
        }
        
        stats
    }
    
    /// Start health monitoring
    pub async fn start_monitoring(&self, check_interval: Duration) {
        let connections = self.connections.clone();
        let is_running = self.is_running.clone();
        
        is_running.store(true, Ordering::SeqCst);
        
        tokio::spawn(async move {
            let mut interval = interval(check_interval);
            
            while is_running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                // Check each connection
                for entry in connections.iter() {
                    let name = entry.key();
                    let client = entry.value();
                    
                    if !client.is_connected() {
                        warn!("Connection {} is disconnected, attempting reconnect", name);
                        
                        // Try to reconnect
                        if let Err(e) = client.connect().await {
                            error!("Failed to reconnect {}: {}", name, e);
                        }
                    }
                }
            }
        });
    }
    
    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }
}

/// Routes messages to appropriate handlers
struct MessageRouter {
    handlers: Arc<DashMap<MessageType, Vec<mpsc::Sender<Message>>>>,
    broadcast_channel: broadcast::Sender<Message>,
}

impl MessageRouter {
    fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(10000);
        
        Self {
            handlers: Arc::new(DashMap::new()),
            broadcast_channel: broadcast_tx,
        }
    }
    
    /// Register a handler for specific message type
    pub fn register_handler(
        &self,
        msg_type: MessageType,
        handler: mpsc::Sender<Message>,
    ) {
        self.handlers
            .entry(msg_type)
            .or_default()
            .push(handler);
    }
    
    /// Route a message to appropriate handlers
    pub async fn route(&self, message: Message) {
        let msg_type = message.message_type();
        
        // Send to specific handlers
        if let Some(handlers) = self.handlers.get(&msg_type) {
            for handler in handlers.iter() {
                let _ = handler.send(message.clone()).await;
            }
        }
        
        // Broadcast to all listeners
        let _ = self.broadcast_channel.send(message);
    }
    
    /// Subscribe to all messages
    pub fn subscribe(&self) -> broadcast::Receiver<Message> {
        self.broadcast_channel.subscribe()
    }
}

/// Collects statistics from all connections
struct StatsCollector {
    total_messages_sent: Arc<AtomicU64>,
    total_messages_received: Arc<AtomicU64>,
    connection_count: Arc<AtomicUsize>,
    last_update: Arc<RwLock<Instant>>,
}

impl StatsCollector {
    fn new() -> Self {
        Self {
            total_messages_sent: Arc::new(AtomicU64::new(0)),
            total_messages_received: Arc::new(AtomicU64::new(0)),
            connection_count: Arc::new(AtomicUsize::new(0)),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    fn update(&self, stats: &HashMap<String, WebSocketStats>) {
        let mut total_sent = 0;
        let mut total_received = 0;
        let mut connected = 0;
        
        for stat in stats.values() {
            total_sent += stat.messages_sent;
            total_received += stat.messages_received;
            if stat.is_connected {
                connected += 1;
            }
        }
        
        self.total_messages_sent.store(total_sent, Ordering::Relaxed);
        self.total_messages_received.store(total_received, Ordering::Relaxed);
        self.connection_count.store(connected, Ordering::Relaxed);
        *self.last_update.write() = Instant::now();
    }
    
    fn get_summary(&self) -> StatsSummary {
        StatsSummary {
            total_messages_sent: self.total_messages_sent.load(Ordering::Relaxed),
            total_messages_received: self.total_messages_received.load(Ordering::Relaxed),
            active_connections: self.connection_count.load(Ordering::Relaxed),
            last_update: *self.last_update.read(),
        }
    }
}

#[derive(Debug, Clone)]
struct StatsSummary {
    total_messages_sent: u64,
    total_messages_received: u64,
    active_connections: usize,
    last_update: Instant,
}

/// Connection pool for load balancing
pub struct ConnectionPool {
    connections: Vec<Arc<WebSocketClient>>,
    current_index: Arc<AtomicUsize>,
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    Random,
}

impl ConnectionPool {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            connections: Vec::new(),
            current_index: Arc::new(AtomicUsize::new(0)),
            strategy,
        }
    }
    
    pub fn add(&mut self, client: Arc<WebSocketClient>) {
        self.connections.push(client);
    }
    
    pub fn get_next(&self) -> Option<Arc<WebSocketClient>> {
        if self.connections.is_empty() {
            return None;
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index = self.current_index.fetch_add(1, Ordering::Relaxed) % self.connections.len();
                Some(self.connections[index].clone())
            }
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                let index = rand::thread_rng().gen_range(0..self.connections.len());
                Some(self.connections[index].clone())
            }
            LoadBalancingStrategy::LeastConnections => {
                // Find connection with least messages
                self.connections
                    .iter()
                    .filter(|c| c.is_connected())
                    .min_by_key(|c| c.stats().messages_sent + c.stats().messages_received)
                    .cloned()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_manager_creation() {
        let manager = WebSocketManager::new();
        let stats = manager.get_all_stats();
        assert!(stats.is_empty());
    }
    
    #[test]
    fn test_connection_pool_round_robin() {
        let pool = ConnectionPool::new(LoadBalancingStrategy::RoundRobin);
        assert!(pool.get_next().is_none());
    }
}