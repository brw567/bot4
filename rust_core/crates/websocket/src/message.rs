use domain_types::market_data::MarketData;
// Message Types for WebSocket Communication
// Supports multiple exchange formats with zero-copy deserialization where possible

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    Subscribe(SubscribeMessage),
    Unsubscribe(UnsubscribeMessage),
    MarketData(MarketData),
    OrderUpdate(OrderUpdate),
    AccountUpdate(AccountUpdate),
    Error(ErrorMessage),
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeMessage {
    pub channels: Vec<String>,
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeMessage {
    pub channels: Vec<String>,
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
// REMOVED: Using canonical domain_types::market_data::MarketData
// pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub data_type: MarketDataType,
    pub exchange: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "data_type", rename_all = "snake_case")]
pub enum MarketDataType {
    Trade(TradeData),
    OrderBook(OrderBookData),
    Ticker(TickerData),
    Candle(CandleData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub trade_id: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookData {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub open_24h: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timeframe: String,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub status: OrderStatus,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountUpdate {
    pub event_type: AccountEventType,
    pub balances: Vec<Balance>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountEventType {
    BalanceUpdate,
    PositionUpdate,
    MarginCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub free: f64,
    pub locked: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub code: i32,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::Subscribe(msg) => write!(f, "Subscribe: {:?}", msg.symbols),
            Message::Unsubscribe(msg) => write!(f, "Unsubscribe: {:?}", msg.symbols),
            Message::MarketData(data) => write!(f, "MarketData: {}", data.symbol),
            Message::OrderUpdate(update) => write!(f, "OrderUpdate: {}", update.order_id),
            Message::AccountUpdate(_) => write!(f, "AccountUpdate"),
            Message::Error(err) => write!(f, "Error: {}", err.message),
            Message::Ping => write!(f, "Ping"),
            Message::Pong => write!(f, "Pong"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    Subscribe,
    Unsubscribe,
    MarketData,
    OrderUpdate,
    AccountUpdate,
    Error,
    Ping,
    Pong,
}

impl Message {
    pub fn message_type(&self) -> MessageType {
        match self {
            Message::Subscribe(_) => MessageType::Subscribe,
            Message::Unsubscribe(_) => MessageType::Unsubscribe,
            Message::MarketData(_) => MessageType::MarketData,
            Message::OrderUpdate(_) => MessageType::OrderUpdate,
            Message::AccountUpdate(_) => MessageType::AccountUpdate,
            Message::Error(_) => MessageType::Error,
            Message::Ping => MessageType::Ping,
            Message::Pong => MessageType::Pong,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_serialization() {
        let msg = Message::Subscribe(SubscribeMessage {
            channels: vec!["trades".to_string()],
            symbols: vec!["BTCUSDT".to_string()],
        });
        
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.message_type(), MessageType::Subscribe);
    }
    
    #[test]
    fn test_market_data_types() {
        let trade = MarketDataType::Trade(TradeData {
            price: 50000.0,
            quantity: 0.1,
            side: TradeSide::Buy,
            trade_id: "123".to_string(),
            timestamp: Utc::now(),
        });
        
        let json = serde_json::to_string(&trade).unwrap();
        assert!(json.contains("\"data_type\":\"trade\""));
    }
}