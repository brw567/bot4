// WebSocket Infrastructure for Bot4 Trading Platform
// High-performance, reconnecting WebSocket client with backpressure handling
// Performance target: Handle 10,000+ messages/second with <1ms latency

pub mod client;
pub mod manager;
pub mod message;
pub mod reconnect;

pub use client::{WebSocketClient, WebSocketConfig};
pub use manager::{WebSocketManager, ConnectionPool};
pub use message::{Message, MessageType, MarketData};
pub use reconnect::{ReconnectStrategy, ExponentialBackoff};
