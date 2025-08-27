//! # Event Types - Core event definitions
//! 
//! Defines all event types that flow through the system.
//! Consolidates multiple event definitions into a unified taxonomy.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::any::Any;

use domain_types::{Order, Trade, Price, Quantity, OrderBook, Ticker};

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    /// Critical events (circuit breakers, kill switch)
    Critical = 0,
    /// High priority (risk events, order fills)
    High = 1,
    /// Normal priority (market data, signals)
    Normal = 2,
    /// Low priority (metrics, logging)
    Low = 3,
}

/// Core event type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    // === Market Data Events ===
    MarketTick {
        symbol: String,
        ticker: Ticker,
    },
    OrderBookUpdate {
        symbol: String,
        order_book: OrderBook,
    },
    TradeEvent {
        symbol: String,
        trade: Trade,
    },
    
    // === Order Management Events ===
    OrderPlaced {
        order: Order,
        strategy_id: Option<String>,
    },
    OrderCancelled {
        order_id: String,
        reason: String,
    },
    OrderFilled {
        order_id: String,
        trade: Trade,
        remaining_quantity: Quantity,
    },
    OrderRejected {
        order: Order,
        reason: String,
    },
    OrderUpdated {
        order_id: String,
        updates: OrderUpdate,
    },
    
    // === Position Events ===
    PositionOpened {
        symbol: String,
        quantity: Quantity,
        entry_price: Price,
    },
    PositionClosed {
        symbol: String,
        exit_price: Price,
        pnl: f64,
    },
    PositionUpdated {
        symbol: String,
        new_quantity: Quantity,
        average_price: Price,
    },
    
    // === Risk Events ===
    RiskLimitBreached {
        limit_type: RiskLimitType,
        current_value: f64,
        limit_value: f64,
    },
    CircuitBreakerTriggered {
        breaker_id: String,
        reason: String,
        cooldown_ms: u64,
    },
    MarginCall {
        required_margin: f64,
        current_margin: f64,
    },
    
    // === Strategy Events ===
    SignalGenerated {
        strategy_id: String,
        signal_type: SignalType,
        confidence: f64,
    },
    StrategyStateChange {
        strategy_id: String,
        old_state: String,
        new_state: String,
    },
    
    // === System Events ===
    SystemStartup {
        version: String,
        config: String,
    },
    SystemShutdown {
        reason: String,
    },
    HeartBeat {
        timestamp: DateTime<Utc>,
        sequence: u64,
    },
    ConfigUpdate {
        component: String,
        old_value: String,
        new_value: String,
    },
    
    // === ML Events ===
    ModelPrediction {
        model_id: String,
        prediction: f64,
        features: Vec<f64>,
    },
    ModelRetrained {
        model_id: String,
        metrics: ModelMetrics,
    },
    
    // === Exchange Events ===
    ExchangeConnected {
        exchange: String,
    },
    ExchangeDisconnected {
        exchange: String,
        reason: String,
    },
    ExchangeRateLimit {
        exchange: String,
        wait_ms: u64,
    },
}

/// Order update fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub price: Option<Price>,
    pub quantity: Option<Quantity>,
    pub stop_loss: Option<Price>,
    pub take_profit: Option<Price>,
}

/// Risk limit types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLimitType {
    MaxPositionSize,
    MaxDrawdown,
    MaxLeverage,
    DailyLossLimit,
    ConcentrationLimit,
}

/// Signal types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    ClosePosition,
    IncreasePosition,
    DecreasePosition,
}

/// ML model metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
}

/// Event metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Unique event ID
    pub id: Uuid,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Sequence number in stream
    pub sequence: u64,
    /// Event priority
    pub priority: EventPriority,
    /// Source component
    pub source: String,
    /// Correlation ID for tracing
    pub correlation_id: Option<Uuid>,
    /// Event version for schema evolution
    pub version: u32,
}

impl EventMetadata {
    /// Create new metadata
    pub fn new(source: String, priority: EventPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            sequence: 0,
            priority,
            source,
            correlation_id: None,
            version: 1,
        }
    }
    
    /// Create with correlation ID
    pub fn with_correlation(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
}

/// Main event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event metadata
    pub metadata: EventMetadata,
    /// Event payload
    pub payload: EventType,
}

impl Event {
    /// Create a new event
    pub fn new(event_type: EventType, source: String) -> Self {
        let priority = Self::infer_priority(&event_type);
        Self {
            metadata: EventMetadata::new(source, priority),
            payload: event_type,
        }
    }
    
    /// Create with specific priority
    pub fn with_priority(event_type: EventType, source: String, priority: EventPriority) -> Self {
        Self {
            metadata: EventMetadata::new(source, priority),
            payload: event_type,
        }
    }
    
    /// Infer priority from event type
    fn infer_priority(event_type: &EventType) -> EventPriority {
        match event_type {
            EventType::CircuitBreakerTriggered { .. } |
            EventType::RiskLimitBreached { .. } |
            EventType::MarginCall { .. } |
            EventType::SystemShutdown { .. } => EventPriority::Critical,
            
            EventType::OrderFilled { .. } |
            EventType::OrderRejected { .. } |
            EventType::PositionOpened { .. } |
            EventType::PositionClosed { .. } |
            EventType::SignalGenerated { .. } => EventPriority::High,
            
            EventType::MarketTick { .. } |
            EventType::OrderBookUpdate { .. } |
            EventType::TradeEvent { .. } |
            EventType::OrderPlaced { .. } => EventPriority::Normal,
            
            EventType::HeartBeat { .. } |
            EventType::ConfigUpdate { .. } |
            EventType::ModelRetrained { .. } => EventPriority::Low,
            
            _ => EventPriority::Normal,
        }
    }
    
    /// Check if event is critical
    pub fn is_critical(&self) -> bool {
        self.metadata.priority == EventPriority::Critical
    }
    
    /// Get event ID
    pub fn id(&self) -> Uuid {
        self.metadata.id
    }
    
    /// Get event timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.metadata.timestamp
    }
}

/// Event builder for fluent API
pub struct EventBuilder {
    event_type: Option<EventType>,
    source: Option<String>,
    priority: Option<EventPriority>,
    correlation_id: Option<Uuid>,
}

impl EventBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            event_type: None,
            source: None,
            priority: None,
            correlation_id: None,
        }
    }
    
    /// Set event type
    pub fn event_type(mut self, event_type: EventType) -> Self {
        self.event_type = Some(event_type);
        self
    }
    
    /// Set source
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    
    /// Set priority
    pub fn priority(mut self, priority: EventPriority) -> Self {
        self.priority = Some(priority);
        self
    }
    
    /// Set correlation ID
    pub fn correlation_id(mut self, id: Uuid) -> Self {
        self.correlation_id = Some(id);
        self
    }
    
    /// Build the event
    pub fn build(self) -> Result<Event, &'static str> {
        let event_type = self.event_type.ok_or("Event type is required")?;
        let source = self.source.ok_or("Source is required")?;
        
        let mut event = if let Some(priority) = self.priority {
            Event::with_priority(event_type, source, priority)
        } else {
            Event::new(event_type, source)
        };
        
        if let Some(correlation_id) = self.correlation_id {
            event.metadata.correlation_id = Some(correlation_id);
        }
        
        Ok(event)
    }
}

impl Default for EventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_priority_inference() {
        let critical = Event::new(
            EventType::CircuitBreakerTriggered {
                breaker_id: "test".to_string(),
                reason: "test".to_string(),
                cooldown_ms: 1000,
            },
            "test".to_string(),
        );
        assert_eq!(critical.metadata.priority, EventPriority::Critical);
        assert!(critical.is_critical());
        
        let normal = Event::new(
            EventType::MarketTick {
                symbol: "BTC/USDT".to_string(),
                ticker: Ticker::default(),
            },
            "test".to_string(),
        );
        assert_eq!(normal.metadata.priority, EventPriority::Normal);
        assert!(!normal.is_critical());
    }
    
    #[test]
    fn test_event_builder() {
        let event = EventBuilder::new()
            .event_type(EventType::HeartBeat {
                timestamp: Utc::now(),
                sequence: 1,
            })
            .source("test")
            .priority(EventPriority::Low)
            .correlation_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        assert_eq!(event.metadata.source, "test");
        assert_eq!(event.metadata.priority, EventPriority::Low);
        assert!(event.metadata.correlation_id.is_some());
    }
}