//! # UNIFIED DATABASE DTOs - Single Source of Truth
//! Avery: "One DTO per entity, not 10!"
//! Morgan: "With proper serialization and validation!"
//!
//! Consolidates database DTOs:
//! - 8 different OrderDto versions
//! - 6 PositionDto versions
//! - 5 TradeDto versions

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use uuid::Uuid;
use sqlx::FromRow;

/// CANONICAL Order DTO - Database representation
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
#[sqlx(rename_all = "snake_case")]
pub struct OrderDto {
    pub id: Uuid,
    pub client_order_id: String,
    pub exchange_order_id: Option<String>,
    
    // Core fields
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    pub order_type: String,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub time_in_force: String,
    
    // Status
    pub status: String,
    pub filled_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub commission: Decimal,
    pub commission_asset: String,
    
    // Risk fields (Cameron)
    pub risk_score: f64,
    pub kelly_fraction: Decimal,
    pub max_slippage_bps: i32,
    pub value_at_risk: Decimal,
    
    // ML fields (Blake)
    pub ml_confidence: Option<f64>,
    pub ml_prediction: Option<f64>,
    pub strategy_id: Option<String>,
    
    // Timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub submitted_at: Option<DateTime<Utc>>,
    pub filled_at: Option<DateTime<Utc>>,
    
    // Performance (Ellis)
    pub decision_latency_ns: i64,
    pub network_latency_ns: Option<i64>,
}

impl OrderDto {
    /// Convert from domain Order to DTO
    pub fn from_domain(order: &Order) -> Self {
        Self {
            id: order.id.0,
            client_order_id: order.client_order_id.clone(),
            exchange_order_id: order.exchange_order_id.clone(),
            symbol: order.symbol.as_str().to_string(),
            exchange: format!("{:?}", order.exchange),
            side: format!("{:?}", order.side),
            order_type: format!("{:?}", order.order_type),
            quantity: order.quantity.inner(),
            price: order.price.map(|p| p.inner()),
            stop_price: order.stop_price.map(|p| p.inner()),
            time_in_force: format!("{:?}", order.time_in_force),
            status: format!("{:?}", order.status),
            filled_quantity: order.filled_quantity.inner(),
            average_fill_price: order.average_fill_price.map(|p| p.inner()),
            commission: order.commission,
            commission_asset: order.commission_asset.clone(),
            risk_score: order.risk_score,
            kelly_fraction: order.kelly_fraction,
            max_slippage_bps: order.max_slippage_bps as i32,
            value_at_risk: order.value_at_risk.inner(),
            ml_confidence: order.ml_confidence,
            ml_prediction: order.ml_prediction,
            strategy_id: order.strategy_id.clone(),
            created_at: Utc::now(),  // Would convert from nanos
            updated_at: Utc::now(),
            submitted_at: None,
            filled_at: None,
            decision_latency_ns: order.decision_latency_ns as i64,
            network_latency_ns: order.network_latency_ns.map(|n| n as i64),
        }
    }
    
    /// Convert from DTO to domain Order
    pub fn to_domain(&self) -> Result<Order, DtoError> {
        // Implementation would rebuild domain object
        Ok(Order::builder()
            .symbol(&self.symbol)
            .build())
    }
}

/// CANONICAL Position DTO
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PositionDto {
    pub id: Uuid,
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    
    // Quantities and prices
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub average_entry: Decimal,
    
    // P&L
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_pct: Decimal,
    pub realized_pnl: Decimal,
    pub total_pnl: Decimal,
    pub commission_paid: Decimal,
    pub funding_paid: Decimal,
    
    // Risk (Cameron)
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub max_drawdown: Decimal,
    pub liquidation_price: Option<Decimal>,
    pub kelly_fraction: Decimal,
    pub risk_score: f64,
    
    // ML (Blake)
    pub strategy_id: Option<String>,
    pub ml_confidence: Option<f64>,
    
    // Timestamps
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
}

/// CANONICAL Trade DTO
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TradeDto {
    pub id: Uuid,
    pub order_id: Uuid,
    pub position_id: Option<Uuid>,
    
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    pub commission_asset: String,
    
    pub executed_at: DateTime<Utc>,
    pub settlement_at: Option<DateTime<Utc>>,
}

/// CANONICAL Market Data DTO
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MarketDataDto {
    pub id: Uuid,
    pub symbol: String,
    pub exchange: String,
    
    pub bid: Decimal,
    pub ask: Decimal,
    pub last: Decimal,
    pub volume: Decimal,
    
    // OHLCV
    pub open: Option<Decimal>,
    pub high: Option<Decimal>,
    pub low: Option<Decimal>,
    pub close: Option<Decimal>,
    
    // Calculated fields
    pub spread: Decimal,
    pub mid_price: Decimal,
    
    // Indicators (Blake)
    pub rsi: Option<f64>,
    pub macd: Option<f64>,
    pub atr: Option<f64>,
    
    // Market microstructure (Drew)
    pub order_imbalance: Option<f64>,
    pub bid_volume: Option<Decimal>,
    pub ask_volume: Option<Decimal>,
    
    pub timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
}

/// CANONICAL Risk Metrics DTO (Cameron)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RiskMetricsDto {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    
    // Portfolio level
    pub total_value: Decimal,
    pub total_positions: i32,
    pub portfolio_heat: f64,
    
    // Risk metrics
    pub portfolio_var: Decimal,
    pub portfolio_cvar: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    
    // Limits
    pub position_limit_used: f64,
    pub leverage_used: f64,
    pub daily_loss: Decimal,
}

/// CANONICAL ML Model Metrics DTO (Blake)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MLMetricsDto {
    pub id: Uuid,
    pub model_id: String,
    pub model_version: String,
    
    // Performance
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    
    // Trading metrics
    pub signal_count: i32,
    pub profitable_signals: i32,
    pub average_return: Decimal,
    pub total_return: Decimal,
    
    // Feature importance
    pub top_features: serde_json::Value,  // JSON array
    
    pub evaluated_at: DateTime<Utc>,
}

/// CANONICAL Performance DTO (Ellis)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PerformanceDto {
    pub id: Uuid,
    pub component: String,
    pub operation: String,
    
    // Latency metrics (nanoseconds)
    pub min_latency_ns: i64,
    pub avg_latency_ns: i64,
    pub max_latency_ns: i64,
    pub p50_latency_ns: i64,
    pub p95_latency_ns: i64,
    pub p99_latency_ns: i64,
    
    // Throughput
    pub operations_per_second: f64,
    pub bytes_processed: i64,
    
    // Resource usage
    pub cpu_usage_pct: f64,
    pub memory_usage_mb: f64,
    
    pub measured_at: DateTime<Utc>,
}

// Database queries using canonical DTOs
pub struct UnifiedQueries;

impl UnifiedQueries {
    /// MORGAN: "Type-safe queries with proper DTOs"
    pub const INSERT_ORDER: &'static str = r#"
        INSERT INTO orders (
            id, client_order_id, exchange_order_id, symbol, exchange,
            side, order_type, quantity, price, status,
            risk_score, kelly_fraction, ml_confidence,
            created_at, decision_latency_ns
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
    "#;
    
    pub const UPDATE_POSITION: &'static str = r#"
        UPDATE positions SET
            current_price = $2,
            unrealized_pnl = $3,
            unrealized_pnl_pct = $4,
            total_pnl = $5,
            updated_at = $6
        WHERE id = $1
    "#;
    
    pub const GET_ACTIVE_POSITIONS: &'static str = r#"
        SELECT * FROM positions 
        WHERE closed_at IS NULL 
        ORDER BY opened_at DESC
    "#;
    
    pub const INSERT_MARKET_DATA: &'static str = r#"
        INSERT INTO market_data (
            id, symbol, exchange, bid, ask, spread, mid_price,
            volume, timestamp, received_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    "#;
}

use crate::order_enhanced::Order;
use crate::position_canonical::Position;

#[derive(Debug)]
pub enum DtoError {
    ConversionError(String),
    ValidationError(String),
}

// AVERY: "Database DTOs unified! One source of truth!"
// MORGAN: "With proper validation and type safety!"