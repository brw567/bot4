//! # Data Layer Abstractions (Layer 1)
//!
//! Data layer abstractions for ingestion, storage, and retrieval.
//! Higher layers use these abstractions without depending on implementation details.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use domain_types::{Candle, Ticker, OrderBook, Trade};
use crate::AbstractionResult;

/// Market data provider abstraction
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Get latest ticker
    async fn get_ticker(
        &self,
        symbol: &str,
    ) -> AbstractionResult<Ticker>;
    
    /// Get order book
    async fn get_order_book(
        &self,
        symbol: &str,
        depth: usize,
    ) -> AbstractionResult<OrderBook>;
    
    /// Get historical candles
    async fn get_candles(
        &self,
        symbol: &str,
        interval: CandleInterval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> AbstractionResult<Vec<Candle>>;
    
    /// Get recent trades
    async fn get_trades(
        &self,
        symbol: &str,
        limit: usize,
    ) -> AbstractionResult<Vec<Trade>>;
    
    /// Subscribe to real-time updates
    async fn subscribe(
        &self,
        symbol: &str,
        data_type: DataType,
    ) -> AbstractionResult<DataStream>;
}

/// Candle intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandleInterval {
    /// 1 minute
    OneMinute,
    /// 5 minutes
    FiveMinutes,
    /// 15 minutes
    FifteenMinutes,
    /// 1 hour
    OneHour,
    /// 4 hours
    FourHours,
    /// 1 day
    OneDay,
}

/// Market data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// Ticker updates
    Ticker,
    /// Order book updates
    OrderBook,
    /// Trade updates
    Trades,
    /// Candle updates
    Candles,
}

/// Data stream for real-time updates
pub struct DataStream {
    /// Stream ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Data type
    pub data_type: DataType,
    /// Receiver channel
    pub receiver: tokio::sync::mpsc::Receiver<DataUpdate>,
}

/// Data update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataUpdate {
    /// Ticker update
    Ticker(Ticker),
    /// Order book update
    OrderBook(OrderBook),
    /// Trade update
    Trade(Trade),
    /// Candle update
    Candle(Candle),
}

/// Time series database abstraction
#[async_trait]
pub trait TimeSeriesDB: Send + Sync {
    /// Write time series data
    async fn write(
        &self,
        measurement: &str,
        tags: Vec<(&str, &str)>,
        fields: Vec<(&str, f64)>,
        timestamp: DateTime<Utc>,
    ) -> AbstractionResult<()>;
    
    /// Query time series data
    async fn query(
        &self,
        query: TimeSeriesQuery,
    ) -> AbstractionResult<TimeSeriesResult>;
    
    /// Batch write
    async fn batch_write(
        &self,
        points: Vec<DataPoint>,
    ) -> AbstractionResult<()>;
    
    /// Create continuous query
    async fn create_continuous_query(
        &self,
        name: &str,
        query: &str,
        interval: &str,
    ) -> AbstractionResult<()>;
}

/// Time series query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesQuery {
    /// Measurement name
    pub measurement: String,
    /// Time range
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    /// Tags to filter
    pub tags: Vec<(String, String)>,
    /// Fields to select
    pub fields: Vec<String>,
    /// Aggregation
    pub aggregation: Option<TimeSeriesAggregation>,
    /// Group by
    pub group_by: Vec<String>,
    /// Limit
    pub limit: Option<usize>,
}

/// Time series aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAggregation {
    /// Function (mean, sum, min, max, etc.)
    pub function: String,
    /// Time window
    pub window: String,
}

/// Time series result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesResult {
    /// Column names
    pub columns: Vec<String>,
    /// Data rows
    pub values: Vec<Vec<serde_json::Value>>,
}

/// Data point for batch writing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Measurement
    pub measurement: String,
    /// Tags
    pub tags: Vec<(String, String)>,
    /// Fields
    pub fields: Vec<(String, f64)>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Cache abstraction
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get value
    async fn get<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> AbstractionResult<Option<T>>;
    
    /// Set value with TTL
    async fn set<T: Serialize>(
        &self,
        key: &str,
        value: T,
        ttl_seconds: Option<u64>,
    ) -> AbstractionResult<()>;
    
    /// Delete value
    async fn delete(&self, key: &str) -> AbstractionResult<()>;
    
    /// Check if key exists
    async fn exists(&self, key: &str) -> AbstractionResult<bool>;
    
    /// Get multiple values
    async fn mget<T: serde::de::DeserializeOwned>(
        &self,
        keys: Vec<String>,
    ) -> AbstractionResult<Vec<Option<T>>>;
    
    /// Set multiple values
    async fn mset<T: Serialize>(
        &self,
        items: Vec<(&str, T)>,
        ttl_seconds: Option<u64>,
    ) -> AbstractionResult<()>;
}

/// Message queue abstraction
#[async_trait]
pub trait MessageQueue: Send + Sync {
    /// Publish message
    async fn publish(
        &self,
        topic: &str,
        message: Vec<u8>,
    ) -> AbstractionResult<()>;
    
    /// Subscribe to topic
    async fn subscribe(
        &self,
        topic: &str,
    ) -> AbstractionResult<MessageStream>;
    
    /// Acknowledge message
    async fn ack(&self, message_id: &str) -> AbstractionResult<()>;
    
    /// Negative acknowledge (requeue)
    async fn nack(&self, message_id: &str) -> AbstractionResult<()>;
}

/// Message stream
pub struct MessageStream {
    /// Stream ID
    pub id: String,
    /// Topic
    pub topic: String,
    /// Receiver
    pub receiver: tokio::sync::mpsc::Receiver<Message>,
}

/// Message from queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message ID
    pub id: String,
    /// Topic
    pub topic: String,
    /// Payload
    pub payload: Vec<u8>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Redelivery count
    pub redelivery_count: u32,
}