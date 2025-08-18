// DTO: Database Order Data Transfer Object
// Represents the database schema for orders
// Owner: Avery | Reviewer: Sam
// Follows Clean Architecture - complete separation from domain

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Row};
use anyhow::Result;

/// Database representation of an Order
/// This DTO maps directly to the database schema
/// Complete separation from domain model per hexagonal architecture
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct OrderDto {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub status: String,
    pub client_order_id: Option<String>,
    pub exchange_order_id: Option<String>,
    pub filled_quantity: f64,
    pub average_fill_price: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl OrderDto {
    /// Create from a database row
    pub fn from_row<R: Row>(row: R) -> Result<Self> 
    where
        for<'r> String: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
        for<'r> f64: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
        for<'r> Option<f64>: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
        for<'r> Option<String>: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
        for<'r> DateTime<Utc>: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
        for<'r> Option<serde_json::Value>: sqlx::Decode<'r, <R as Row>::Database> + sqlx::Type<<R as Row>::Database>,
    {
        Ok(Self {
            id: row.try_get("id")?,
            symbol: row.try_get("symbol")?,
            side: row.try_get("side")?,
            order_type: row.try_get("order_type")?,
            quantity: row.try_get("quantity")?,
            price: row.try_get("price")?,
            status: row.try_get("status")?,
            client_order_id: row.try_get("client_order_id")?,
            exchange_order_id: row.try_get("exchange_order_id")?,
            filled_quantity: row.try_get("filled_quantity")?,
            average_fill_price: row.try_get("average_fill_price")?,
            created_at: row.try_get("created_at")?,
            updated_at: row.try_get("updated_at")?,
            metadata: row.try_get("metadata")?,
        })
    }
}

/// Database representation of Order Fill
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct FillDto {
    pub id: String,
    pub order_id: String,
    pub exchange_fill_id: String,
    pub price: f64,
    pub quantity: f64,
    pub fee: f64,
    pub fee_asset: String,
    pub timestamp: DateTime<Utc>,
    pub is_maker: bool,
}

/// Database representation of Position
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PositionDto {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Database representation of Trade
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TradeDto {
    pub id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub quantity: f64,
    pub fee: f64,
    pub fee_asset: String,
    pub pnl: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Database representation of Balance
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BalanceDto {
    pub asset: String,
    pub free: f64,
    pub locked: f64,
    pub total: f64,
    pub updated_at: DateTime<Utc>,
}

/// Database representation of Account
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AccountDto {
    pub id: String,
    pub exchange: String,
    pub account_type: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Database representation of Risk Metrics
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RiskMetricsDto {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub total_exposure: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub correlation_matrix: serde_json::Value,
    pub portfolio_heat: f64,
}

/// Database representation of ML Model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ModelDto {
    pub id: String,
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub parameters: serde_json::Value,
    pub metrics: serde_json::Value,
    pub training_data_hash: String,
    pub created_at: DateTime<Utc>,
    pub is_active: bool,
}

/// Database representation of Signal
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SignalDto {
    pub id: String,
    pub source: String,
    pub symbol: String,
    pub signal_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub metadata: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Database representation of Audit Log
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditLogDto {
    pub id: String,
    pub entity_type: String,
    pub entity_id: String,
    pub action: String,
    pub actor: String,
    pub changes: serde_json::Value,
    pub metadata: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_dto_creation() {
        let dto = OrderDto {
            id: "order123".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "limit".to_string(),
            quantity: 0.1,
            price: Some(50000.0),
            status: "open".to_string(),
            client_order_id: Some("client123".to_string()),
            exchange_order_id: None,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: None,
        };
        
        assert_eq!(dto.id, "order123");
        assert_eq!(dto.symbol, "BTC/USDT");
        assert_eq!(dto.quantity, 0.1);
    }
}