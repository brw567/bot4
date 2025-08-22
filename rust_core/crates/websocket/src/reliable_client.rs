// Reliable WebSocket Client with Automatic Reconnection
// Team: Casey (Exchange Integration) & Jordan (Performance) & Sam (Code Quality)
// CRITICAL: Must maintain connection for 24/7 trading
// References:
// - "WebSocket Programming" - Andrew Lombardi
// - "Building Reliable Trading Systems" - Kaufman
// - "Network Programming with Rust" - Blandy

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, broadcast, oneshot};
use tokio::time::{interval, timeout, sleep};
use tokio_tungstenite::{
    connect_async,
    tungstenite::protocol::Message as WsMessage,
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};
use url::Url;
use chrono::{DateTime, Utc};

use crate::message::Message;
use infrastructure::{Clock, RetryExecutor, RetryPolicy};

#[derive(Debug, Error)]
pub enum ReliableWebSocketError {
    #[error("Connection failed after {attempts} attempts: {reason}")]
    ConnectionExhausted { attempts: u32, reason: String },
    
    #[error("Send failed: {0}")]
    SendFailed(String),
    
    #[error("Receive failed: {0}")]
    ReceiveFailed(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Circuit breaker open")]
    CircuitBreakerOpen,
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("Channel closed")]
    ChannelClosed,
    
    #[error("Shutdown requested")]
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct ReliableWebSocketConfig {
    pub url: String,
    pub name: String, // For logging
    
    // Reconnection settings
    pub auto_reconnect: bool,
    pub initial_reconnect_delay: Duration,
    pub max_reconnect_delay: Duration,
    pub max_reconnect_attempts: u32,
    pub reconnect_jitter: f64,
    
    // Connection health
    pub ping_interval: Duration,
    pub pong_timeout: Duration,
    pub idle_timeout: Duration,
    
    // Message handling
    pub message_buffer_size: usize,
    pub max_pending_messages: usize,
    pub max_frame_size: usize,
    
    // Circuit breaker
    pub circuit_breaker_enabled: bool,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_recovery: Duration,
}

impl Default for ReliableWebSocketConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            name: "WebSocket".to_string(),
            
            auto_reconnect: true,
            initial_reconnect_delay: Duration::from_secs(1),
            max_reconnect_delay: Duration::from_secs(60),
            max_reconnect_attempts: u32::MAX, // Infinite retries
            reconnect_jitter: 0.3,
            
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(90),
            
            message_buffer_size: 10000,
            max_pending_messages: 1000,
            max_frame_size: 64 * 1024 * 1024,
            
            circuit_breaker_enabled: false, // Don't give up on market data
            circuit_breaker_threshold: 10,
            circuit_breaker_recovery: Duration::from_secs(300),
        }
    }
}

impl ReliableWebSocketConfig {
    /// Create config for critical market data streams
    pub fn for_market_data(url: String, exchange: &str) -> Self {
        Self {
            url,
            name: format!("{}_market_data", exchange),
            auto_reconnect: true,
            max_reconnect_attempts: u32::MAX, // Never give up
            ping_interval: Duration::from_secs(20),
            pong_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(60),
            circuit_breaker_enabled: false, // Must stay connected
            ..Default::default()
        }
    }
    
    /// Create config for order management streams
    pub fn for_orders(url: String, exchange: &str) -> Self {
        Self {
            url,
            name: format!("{}_orders", exchange),
            auto_reconnect: true,
            max_reconnect_attempts: 100, // Limited retries
            initial_reconnect_delay: Duration::from_millis(500),
            max_reconnect_delay: Duration::from_secs(30),
            ping_interval: Duration::from_secs(15),
            pong_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(45),
            circuit_breaker_enabled: true, // Protect from bad state
            circuit_breaker_threshold: 5,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting { attempt: u32, next_retry: DateTime<Utc> },
    Failed { reason: String },
}

pub struct ReliableWebSocketClient {
    config: Arc<ReliableWebSocketConfig>,
    state: Arc<RwLock<ConnectionState>>,
    
    // Connection management
    is_running: Arc<AtomicBool>,
    reconnect_attempts: Arc<AtomicU32>,
    consecutive_failures: Arc<AtomicU32>,
    
    // Statistics
    total_connects: Arc<AtomicU64>,
    total_disconnects: Arc<AtomicU64>,
    messages_sent: Arc<AtomicU64>,
    messages_received: Arc<AtomicU64>,
    last_message_time: Arc<RwLock<Instant>>,
    
    // Channels
    outbound_tx: mpsc::Sender<WsMessage>,
    outbound_rx: Arc<RwLock<Option<mpsc::Receiver<WsMessage>>>>,
    inbound_tx: broadcast::Sender<Message>,
    
    // Pending messages during reconnection
    pending_messages: Arc<RwLock<VecDeque<WsMessage>>>,
    
    // Retry executor for resilient operations
    retry_executor: Arc<RetryExecutor>,
    
    // Shutdown signal
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,
}

impl ReliableWebSocketClient {
    pub fn new(config: ReliableWebSocketConfig) -> Self {
        let (outbound_tx, outbound_rx) = mpsc::channel(config.message_buffer_size);
        let (inbound_tx, _) = broadcast::channel(config.message_buffer_size);
        
        let retry_policy = RetryPolicy {
            max_retries: config.max_reconnect_attempts,
            initial_backoff: config.initial_reconnect_delay,
            max_backoff: config.max_reconnect_delay,
            exponential_base: 2.0,
            jitter_factor: config.reconnect_jitter,
            total_timeout: None, // No total timeout for persistent connections
            circuit_breaker_enabled: config.circuit_breaker_enabled,
            circuit_breaker_threshold: config.circuit_breaker_threshold,
            circuit_breaker_recovery: config.circuit_breaker_recovery,
        };
        
        Self {
            config: Arc::new(config),
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            
            is_running: Arc::new(AtomicBool::new(false)),
            reconnect_attempts: Arc::new(AtomicU32::new(0)),
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            
            total_connects: Arc::new(AtomicU64::new(0)),
            total_disconnects: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            last_message_time: Arc::new(RwLock::new(Instant::now())),
            
            outbound_tx,
            outbound_rx: Arc::new(RwLock::new(Some(outbound_rx))),
            inbound_tx,
            
            pending_messages: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            
            retry_executor: Arc::new(RetryExecutor::new(retry_policy)),
            
            shutdown_tx: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Start the WebSocket client with automatic reconnection
    pub async fn start(&self) -> Result<(), ReliableWebSocketError> {
        if self.is_running.load(Ordering::SeqCst) {
            return Ok(()); // Already running
        }
        
        self.is_running.store(true, Ordering::SeqCst);
        info!("[{}] Starting reliable WebSocket client", self.config.name);
        
        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        *self.shutdown_tx.write() = Some(shutdown_tx);
        
        // Clone for async task
        let self_clone = Arc::new(self.clone_inner());
        
        // Spawn main connection loop
        tokio::spawn(async move {
            loop {
                // Check if shutdown requested
                if !self_clone.is_running.load(Ordering::SeqCst) {
                    break;
                }
                
                // Try to connect
                match self_clone.connect_with_retry().await {
                    Ok(wsstream) => {
                        info!("[{}] WebSocket connected successfully", self_clone.config.name);
                        self_clone.total_connects.fetch_add(1, Ordering::Relaxed);
                        self_clone.consecutive_failures.store(0, Ordering::SeqCst);
                        self_clone.reconnect_attempts.store(0, Ordering::SeqCst);
                        *self_clone.state.write() = ConnectionState::Connected;
                        
                        // Send any pending messages
                        self_clone.flush_pending_messages().await;
                        
                        // Handle connection until it fails
                        if let Err(e) = self_clone.handle_connection(wsstream).await {
                            error!("[{}] Connection error: {}", self_clone.config.name, e);
                            self_clone.total_disconnects.fetch_add(1, Ordering::Relaxed);
                        }
                        
                        *self_clone.state.write() = ConnectionState::Disconnected;
                    }
                    Err(e) => {
                        error!("[{}] Failed to connect: {}", self_clone.config.name, e);
                        self_clone.consecutive_failures.fetch_add(1, Ordering::SeqCst);
                        
                        if !self_clone.config.auto_reconnect {
                            *self_clone.state.write() = ConnectionState::Failed {
                                reason: e.to_string(),
                            };
                            break;
                        }
                        
                        // Check if we've exhausted retries
                        let attempts = self_clone.reconnect_attempts.load(Ordering::SeqCst);
                        if attempts >= self_clone.config.max_reconnect_attempts {
                            error!("[{}] Maximum reconnection attempts reached", self_clone.config.name);
                            *self_clone.state.write() = ConnectionState::Failed {
                                reason: format!("Exhausted {} reconnection attempts", attempts),
                            };
                            break;
                        }
                    }
                }
                
                // Wait before next reconnection attempt
                if self_clone.config.auto_reconnect && self_clone.is_running.load(Ordering::SeqCst) {
                    let delay = self_clone.calculate_reconnect_delay();
                    let next_retry = Utc::now() + chrono::Duration::from_std(delay).unwrap();
                    
                    *self_clone.state.write() = ConnectionState::Reconnecting {
                        attempt: self_clone.reconnect_attempts.load(Ordering::SeqCst),
                        next_retry,
                    };
                    
                    info!("[{}] Reconnecting in {:?}", self_clone.config.name, delay);
                    sleep(delay).await;
                }
            }
            
            info!("[{}] WebSocket client stopped", self_clone.config.name);
        });
        
        Ok(())
    }
    
    /// Connect with retry logic
    async fn connect_with_retry(&self) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>, ReliableWebSocketError> {
        let url = Url::parse(&self.config.url)
            .map_err(|e| ReliableWebSocketError::ConnectionExhausted {
                attempts: 0,
                reason: format!("Invalid URL: {}", e),
            })?;
        
        let connect_timeout = Duration::from_secs(10);
        
        match timeout(connect_timeout, connect_async(&url)).await {
            Ok(Ok((wsstream, _))) => Ok(wsstream),
            Ok(Err(e)) => Err(ReliableWebSocketError::ConnectionExhausted {
                attempts: self.reconnect_attempts.fetch_add(1, Ordering::SeqCst),
                reason: e.to_string(),
            }),
            Err(_) => Err(ReliableWebSocketError::Timeout("Connection timeout".to_string())),
        }
    }
    
    /// Handle active WebSocket connection
    async fn handle_connection(
        &self,
        wsstream: WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) -> Result<(), ReliableWebSocketError> {
        let (mut write, mut read) = wsstream.split();
        
        // Take the outbound receiver
        let mut outbound_rx = self.outbound_rx.write().take()
            .ok_or(ReliableWebSocketError::ChannelClosed)?;
        
        // Clone for async tasks
        let inbound_tx = self.inbound_tx.clone();
        let messages_received = self.messages_received.clone();
        let last_message_time = self.last_message_time.clone();
        let is_running = self.is_running.clone();
        let name = self.config.name.clone();
        
        // Spawn read task
        let read_handle = tokio::spawn(async move {
            while let Some(msg_result) = read.next().await {
                match msg_result {
                    Ok(WsMessage::Text(text)) => {
                        messages_received.fetch_add(1, Ordering::Relaxed);
                        *last_message_time.write() = Instant::now();
                        
                        // Parse and broadcast message
                        match serde_json::from_str::<Message>(&text) {
                            Ok(msg) => {
                                if let Err(e) = inbound_tx.send(msg) {
                                    debug!("[{}] No receivers for message: {}", name, e);
                                }
                            }
                            Err(e) => {
                                warn!("[{}] Failed to parse message: {}", name, e);
                            }
                        }
                    }
                    Ok(WsMessage::Binary(data)) => {
                        messages_received.fetch_add(1, Ordering::Relaxed);
                        *last_message_time.write() = Instant::now();
                        debug!("[{}] Received binary message: {} bytes", name, data.len());
                    }
                    Ok(WsMessage::Pong(_)) => {
                        debug!("[{}] Received pong", name);
                        *last_message_time.write() = Instant::now();
                    }
                    Ok(WsMessage::Close(_)) => {
                        info!("[{}] WebSocket closed by server", name);
                        break;
                    }
                    Ok(WsMessage::Ping(data)) => {
                        debug!("[{}] Received ping, sending pong", name);
                        // Pong is handled automatically by tungstenite
                    }
                    Err(e) => {
                        error!("[{}] WebSocket read error: {}", name, e);
                        break;
                    }
                    _ => {}
                }
            }
            
            info!("[{}] Read task ended", name);
        });
        
        // Clone for write task
        let messages_sent = self.messages_sent.clone();
        let name_write = self.config.name.clone();
        let pending_messages = self.pending_messages.clone();
        
        // Spawn write task
        let write_handle = tokio::spawn(async move {
            // Send any pending messages first
            let pending: Vec<WsMessage> = pending_messages.write().drain(..).collect();
            for msg in pending {
                if let Err(e) = write.send(msg).await {
                    error!("[{}] Failed to send pending message: {}", name_write, e);
                    break;
                }
                messages_sent.fetch_add(1, Ordering::Relaxed);
            }
            
            // Process outbound messages
            while let Some(msg) = outbound_rx.recv().await {
                if let Err(e) = write.send(msg).await {
                    error!("[{}] Failed to send message: {}", name_write, e);
                    break;
                }
                messages_sent.fetch_add(1, Ordering::Relaxed);
            }
            
            info!("[{}] Write task ended", name_write);
        });
        
        // Spawn ping task
        let ping_interval_duration = self.config.ping_interval;
        let outbound_tx = self.outbound_tx.clone();
        let is_running_ping = self.is_running.clone();
        let name_ping = self.config.name.clone();
        
        let ping_handle = tokio::spawn(async move {
            let mut ping_interval = interval(ping_interval_duration);
            
            while is_running_ping.load(Ordering::SeqCst) {
                ping_interval.tick().await;
                
                if let Err(e) = outbound_tx.send(WsMessage::Ping(vec![])).await {
                    warn!("[{}] Failed to send ping: {}", name_ping, e);
                    break;
                }
                
                debug!("[{}] Sent ping", name_ping);
            }
        });
        
        // Spawn idle detection task
        let idle_timeout = self.config.idle_timeout;
        let last_message_time_idle = self.last_message_time.clone();
        let is_running_idle = self.is_running.clone();
        let name_idle = self.config.name.clone();
        
        let idle_handle = tokio::spawn(async move {
            let mut check_interval = interval(Duration::from_secs(10));
            
            while is_running_idle.load(Ordering::SeqCst) {
                check_interval.tick().await;
                
                let last_msg = *last_message_time_idle.read();
                if Instant::now().duration_since(last_msg) > idle_timeout {
                    warn!("[{}] Connection idle for too long, reconnecting", name_idle);
                    break;
                }
            }
        });
        
        // Wait for any task to finish
        tokio::select! {
            _ = read_handle => {
                info!("[{}] Read task finished", self.config.name);
            }
            _ = write_handle => {
                info!("[{}] Write task finished", self.config.name);
            }
            _ = ping_handle => {
                info!("[{}] Ping task finished", self.config.name);
            }
            _ = idle_handle => {
                info!("[{}] Idle detection triggered", self.config.name);
            }
        }
        
        // Note: outbound_rx is consumed by the write task and cannot be put back
        // A new receiver will be created on reconnection if needed
        
        Ok(())
    }
    
    /// Calculate reconnect delay with exponential backoff
    fn calculate_reconnect_delay(&self) -> Duration {
        let attempt = self.reconnect_attempts.load(Ordering::SeqCst);
        let base = self.config.initial_reconnect_delay.as_secs_f64();
        let max = self.config.max_reconnect_delay.as_secs_f64();
        
        let exponential = base * 2_f64.powi(attempt as i32);
        let capped = exponential.min(max);
        
        // Add jitter
        let jitter_range = capped * self.config.reconnect_jitter;
        let jitter = rand::random::<f64>() * jitter_range * 2.0 - jitter_range;
        let final_delay = (capped + jitter).max(0.1);
        
        Duration::from_secs_f64(final_delay)
    }
    
    /// Flush pending messages after reconnection
    async fn flush_pending_messages(&self) {
        let pendingcount = self.pending_messages.read().len();
        if pendingcount > 0 {
            info!("[{}] Flushing {} pending messages", self.config.name, pendingcount);
            // Messages will be sent by the write task
        }
    }
    
    /// Send a message
    pub async fn send(&self, message: Message) -> Result<(), ReliableWebSocketError> {
        let json = serde_json::to_string(&message)
            .map_err(|e| ReliableWebSocketError::ParseError(e.to_string()))?;
        
        let wsmessage = WsMessage::Text(json);
        
        // Check connection state
        match &*self.state.read() {
            ConnectionState::Connected => {
                // Send directly
                self.outbound_tx.send(wsmessage).await
                    .map_err(|e| ReliableWebSocketError::SendFailed(e.to_string()))
            }
            ConnectionState::Reconnecting { .. } | ConnectionState::Connecting => {
                // Queue for later
                let mut pending = self.pending_messages.write();
                if pending.len() >= self.config.max_pending_messages {
                    pending.pop_front(); // Drop oldest
                }
                pending.push_back(wsmessage);
                Ok(())
            }
            _ => Err(ReliableWebSocketError::SendFailed("Not connected".to_string())),
        }
    }
    
    /// Subscribe to incoming messages
    pub fn subscribe(&self) -> broadcast::Receiver<Message> {
        self.inbound_tx.subscribe()
    }
    
    /// Stop the WebSocket client
    pub async fn stop(&self) {
        info!("[{}] Stopping WebSocket client", self.config.name);
        self.is_running.store(false, Ordering::SeqCst);
        
        if let Some(shutdown_tx) = self.shutdown_tx.write().take() {
            let _ = shutdown_tx.send(());
        }
    }
    
    /// Get current connection state
    pub fn state(&self) -> ConnectionState {
        self.state.read().clone()
    }
    
    /// Get statistics
    pub fn stats(&self) -> WebSocketStats {
        WebSocketStats {
            total_connects: self.total_connects.load(Ordering::Relaxed),
            total_disconnects: self.total_disconnects.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            reconnect_attempts: self.reconnect_attempts.load(Ordering::SeqCst),
            consecutive_failures: self.consecutive_failures.load(Ordering::SeqCst),
            pending_messages: self.pending_messages.read().len(),
            last_message_time: *self.last_message_time.read(),
            state: self.state.read().clone(),
        }
    }
    
    /// Clone inner references for spawning
    fn clone_inner(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            is_running: self.is_running.clone(),
            reconnect_attempts: self.reconnect_attempts.clone(),
            consecutive_failures: self.consecutive_failures.clone(),
            total_connects: self.total_connects.clone(),
            total_disconnects: self.total_disconnects.clone(),
            messages_sent: self.messages_sent.clone(),
            messages_received: self.messages_received.clone(),
            last_message_time: self.last_message_time.clone(),
            outbound_tx: self.outbound_tx.clone(),
            outbound_rx: self.outbound_rx.clone(),
            inbound_tx: self.inbound_tx.clone(),
            pending_messages: self.pending_messages.clone(),
            retry_executor: self.retry_executor.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WebSocketStats {
    pub total_connects: u64,
    pub total_disconnects: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub reconnect_attempts: u32,
    pub consecutive_failures: u32,
    pub pending_messages: usize,
    pub last_message_time: Instant,
    pub state: ConnectionState,
}

// ============================================================================
// TESTS - Casey & Jordan validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_config_for_market_data() {
        let config = ReliableWebSocketConfig::for_market_data(
            "wss://stream.binance.com:9443/ws".to_string(),
            "binance"
        );
        
        assert_eq!(config.name, "binance_market_data");
        assert!(config.auto_reconnect);
        assert_eq!(config.max_reconnect_attempts, u32::MAX);
        assert!(!config.circuit_breaker_enabled);
    }
    
    #[tokio::test]
    async fn test_config_for_orders() {
        let config = ReliableWebSocketConfig::for_orders(
            "wss://stream.binance.com:9443/ws".to_string(),
            "binance"
        );
        
        assert_eq!(config.name, "binance_orders");
        assert!(config.auto_reconnect);
        assert_eq!(config.max_reconnect_attempts, 100);
        assert!(config.circuit_breaker_enabled);
    }
    
    #[tokio::test]
    async fn test_reconnect_delay_calculation() {
        let config = ReliableWebSocketConfig {
            initial_reconnect_delay: Duration::from_secs(1),
            max_reconnect_delay: Duration::from_secs(60),
            reconnect_jitter: 0.0, // No jitter for deterministic test
            ..Default::default()
        };
        
        let client = ReliableWebSocketClient::new(config);
        
        // First attempt
        client.reconnect_attempts.store(0, Ordering::SeqCst);
        let delay = client.calculate_reconnect_delay();
        assert_eq!(_delay, Duration::from_secs(1));
        
        // Second attempt (2x)
        client.reconnect_attempts.store(1, Ordering::SeqCst);
        let delay = client.calculate_reconnect_delay();
        assert_eq!(_delay, Duration::from_secs(2));
        
        // Third attempt (4x)
        client.reconnect_attempts.store(2, Ordering::SeqCst);
        let delay = client.calculate_reconnect_delay();
        assert_eq!(_delay, Duration::from_secs(4));
        
        // Should cap at max_reconnect_delay
        client.reconnect_attempts.store(10, Ordering::SeqCst);
        let delay = client.calculate_reconnect_delay();
        assert_eq!(_delay, Duration::from_secs(60));
    }
    
    #[tokio::test]
    async fn test_pending_messages() {
        let config = ReliableWebSocketConfig {
            max_pending_messages: 3,
            ..Default::default()
        };
        
        let client = ReliableWebSocketClient::new(config);
        
        // Simulate disconnected state
        *client.state.write() = ConnectionState::Reconnecting {
            attempt: 1,
            next_retry: Utc::now(),
        };
        
        // Add messages to pending queue
        for i in 0..5 {
            let msg = Message::Error(crate::message::ErrorMessage {
                code: i,
                message: format!("test_{}", i),
                timestamp: Utc::now(),
            });
            let _ = client.send(msg).await;
        }
        
        // Should only keep max_pending_messages
        assert_eq!(client.pending_messages.read().len(), 3);
    }
}