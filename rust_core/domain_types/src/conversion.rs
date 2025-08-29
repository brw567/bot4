//! # Conversion Traits for Legacy Type Compatibility
//! 
//! Enables gradual migration from 158 duplicate types to canonical types
//! using the Strangler Fig pattern - wrapping and replacing incrementally.
//!
//! ## Design Principles
//! - Zero-cost abstractions where possible
//! - Fallible conversions for safety
//! - Bidirectional conversion support
//! - Feature flag controlled
//!
//! ## Migration Strategy
//! 1. Implement conversions for all legacy types
//! 2. Update code to use conversions transparently
//! 3. Gradually replace legacy types with canonical
//! 4. Remove conversions once migration complete

use crate::{OrderSide, OrderType, OrderStatus};
use crate::CandleInterval;


/// Trait for converting from legacy types to canonical
pub trait ToCanonical<T> {
    /// Converts to canonical type
    fn to_canonical(self) -> Result<T, ConversionError>;
}

/// Trait for converting from canonical types to legacy
pub trait FromCanonical<T> {
    /// Converts from canonical type
    fn from_canonical(canonical: T) -> Result<Self, ConversionError>
    where
        Self: Sized;
}

/// Re-export conversion trait for legacy compatibility
pub trait FromLegacy<T> {
    /// Converts from legacy type
    fn from_legacy(legacy: T) -> Result<Self, ConversionError>
    where
        Self: Sized;
}

/// Errors that can occur during conversion
#[derive(Debug, thiserror::Error)]
/// TODO: Add docs
pub enum ConversionError {
    #[error("Invalid price value: {0}")]
    InvalidPrice(String),
    
    #[error("Invalid quantity value: {0}")]
    InvalidQuantity(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    
    #[error("Unsupported conversion: {0}")]
    UnsupportedConversion(String),
    
    #[error("Value out of range: {0}")]
    OutOfRange(String),
}

// ===== Price/Quantity Conversions =====
// Note: Basic TryFrom implementations are in price.rs and quantity.rs
// Here we only add conversion helpers for legacy compatibility

// ===== Legacy Order Structure Examples =====

/// Example legacy order from domain module
#[cfg(feature = "legacy_compat")]
pub mod legacy_domain {
    use rust_decimal::Decimal;
    use chrono::{DateTime, Utc};
    
    pub struct LegacyDomainOrder {
        pub id: String,
        pub symbol: String,
        pub side: String, // "BUY" or "SELL"
        pub price: Option<Decimal>,
        pub quantity: Decimal,
        pub order_type: String,
        pub status: String,
        pub created_at: DateTime<Utc>,
    }
}

/// Example legacy order from exchange module
#[cfg(feature = "legacy_compat")]
pub mod legacy_exchange {
    pub struct LegacyExchangeOrder {
        pub order_id: u64,
        pub pair: String,
        pub is_buy: bool,
        pub limit_price: Option<f64>,
        pub size: f64,
        pub filled: f64,
        pub timestamp: i64,
    }
}

// ===== Conversion Implementations for Legacy Orders =====

#[cfg(feature = "legacy_compat")]
impl ToCanonical<Order> for legacy_domain::LegacyDomainOrder {
    fn to_canonical(self) -> Result<Order, ConversionError> {
        let side = match self.side.as_str() {
            "BUY" | "Buy" | "buy" => OrderSide::Buy,
            "SELL" | "Sell" | "sell" => OrderSide::Sell,
            _ => return Err(ConversionError::TypeMismatch {
                expected: "BUY or SELL".to_string(),
                actual: self.side,
            }),
        };
        
        let order_type = match self.order_type.as_str() {
            "MARKET" | "Market" => OrderType::Market,
            "LIMIT" | "Limit" => OrderType::Limit,
            "STOP" | "StopMarket" => OrderType::StopMarket,
            _ => OrderType::Market, // Default fallback
        };
        
        let quantity = Quantity::new(self.quantity)
            .map_err(|e| ConversionError::InvalidQuantity(e.to_string()))?;
        
        let mut order = if let Some(price_decimal) = self.price {
            let price = Price::new(price_decimal)
                .map_err(|e| ConversionError::InvalidPrice(e.to_string()))?;
            Order::limit(self.symbol, side, price, quantity, TimeInForce::GTC)
        } else {
            Order::market(self.symbol, side, quantity)
        };
        
        // Try to parse the ID
        if let Ok(uuid) = self.id.parse::<uuid::Uuid>() {
            order.id = OrderId(uuid);
        }
        
        order.created_at = self.created_at;
        
        Ok(order)
    }
}

#[cfg(feature = "legacy_compat")]
impl ToCanonical<Order> for legacy_exchange::LegacyExchangeOrder {
    fn to_canonical(self) -> Result<Order, ConversionError> {
        let side = if self.is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        
        let quantity = Quantity::try_from(self.size)?;
        let filled_quantity = Quantity::try_from(self.filled)?;
        
        let mut order = if let Some(price_f64) = self.limit_price {
            let price = Price::try_from(price_f64)?;
            Order::limit(self.pair, side, price, quantity, TimeInForce::GTC)
        } else {
            Order::market(self.pair, side, quantity)
        };
        
        order.filled_quantity = filled_quantity;
        order.created_at = DateTime::from_timestamp(self.timestamp, 0)
            .unwrap_or_else(Utc::now);
        
        Ok(order)
    }
}

// ===== Generic Conversion Helpers =====

/// Converts a string side to OrderSide
/// TODO: Add docs
pub fn parse_order_side(side: &str) -> Result<OrderSide, ConversionError> {
    match side.to_uppercase().as_str() {
        "BUY" | "BID" | "LONG" => Ok(OrderSide::Buy),
        "SELL" | "ASK" | "SHORT" => Ok(OrderSide::Sell),
        _ => Err(ConversionError::TypeMismatch {
            expected: "BUY or SELL".to_string(),
            actual: side.to_string(),
        }),
    }
}

/// Converts a string type to OrderType
/// TODO: Add docs
pub fn parse_order_type(order_type: &str) -> Result<OrderType, ConversionError> {
    match order_type.to_uppercase().as_str() {
        "MARKET" => Ok(OrderType::Market),
        "LIMIT" => Ok(OrderType::Limit),
        "STOP" | "STOP_MARKET" | "STOP_LOSS" => Ok(OrderType::StopMarket),
        "STOP_LIMIT" => Ok(OrderType::StopLimit),
        "TAKE_PROFIT" => Ok(OrderType::TakeProfit),
        "OCO" => Ok(OrderType::OCO),
        "ICEBERG" => Ok(OrderType::Iceberg),
        "POST_ONLY" | "POSTONLY" => Ok(OrderType::PostOnly),
        _ => Err(ConversionError::UnsupportedConversion(
            format!("Unknown order type: {}", order_type)
        )),
    }
}

/// Converts a string status to OrderStatus  
/// TODO: Add docs
pub fn parse_order_status(status: &str) -> Result<OrderStatus, ConversionError> {
    match status.to_uppercase().as_str() {
        "DRAFT" | "NEW" | "PENDING_NEW" => Ok(OrderStatus::Draft),
        "PENDING" | "SUBMITTED" => Ok(OrderStatus::Pending),
        "OPEN" | "ACTIVE" | "ACCEPTED" => Ok(OrderStatus::Open),
        "PARTIALLY_FILLED" | "PARTIAL" => Ok(OrderStatus::PartiallyFilled),
        "FILLED" | "COMPLETED" | "EXECUTED" => Ok(OrderStatus::Filled),
        "CANCELLED" | "CANCELED" => Ok(OrderStatus::Cancelled),
        "REJECTED" | "FAILED" => Ok(OrderStatus::Rejected),
        "EXPIRED" => Ok(OrderStatus::Expired),
        _ => Err(ConversionError::UnsupportedConversion(
            format!("Unknown order status: {}", status)
        )),
    }
}

/// Converts string interval to CandleInterval
/// TODO: Add docs
pub fn parse_candle_interval(interval: &str) -> Result<CandleInterval, ConversionError> {
    match interval {
        "1s" => Ok(CandleInterval::Second1),
        "5s" => Ok(CandleInterval::Second5),
        "15s" => Ok(CandleInterval::Second15),
        "30s" => Ok(CandleInterval::Second30),
        "1m" | "1min" => Ok(CandleInterval::Minute1),
        "3m" | "3min" => Ok(CandleInterval::Minute3),
        "5m" | "5min" => Ok(CandleInterval::Minute5),
        "15m" | "15min" => Ok(CandleInterval::Minute15),
        "30m" | "30min" => Ok(CandleInterval::Minute30),
        "1h" | "60m" | "1hr" => Ok(CandleInterval::Hour1),
        "2h" | "2hr" => Ok(CandleInterval::Hour2),
        "4h" | "4hr" => Ok(CandleInterval::Hour4),
        "6h" | "6hr" => Ok(CandleInterval::Hour6),
        "8h" | "8hr" => Ok(CandleInterval::Hour8),
        "12h" | "12hr" => Ok(CandleInterval::Hour12),
        "1d" | "24h" | "1day" => Ok(CandleInterval::Day1),
        "3d" | "3day" => Ok(CandleInterval::Day3),
        "1w" | "7d" | "1week" => Ok(CandleInterval::Week1),
        "1M" | "30d" | "1month" => Ok(CandleInterval::Month1),
        _ => Err(ConversionError::UnsupportedConversion(
            format!("Unknown candle interval: {}", interval)
        )),
    }
}

// ===== Batch Conversion Utilities =====

/// Converts a vector of legacy types to canonical
/// TODO: Add docs
pub fn batch_convert<L, C, F>(legacy_items: Vec<L>, converter: F) -> Result<Vec<C>, Vec<ConversionError>>
where
    F: Fn(L) -> Result<C, ConversionError>,
{
    let mut results = Vec::with_capacity(legacy_items.len());
    let mut errors = Vec::new();
    
    for item in legacy_items {
        match converter(item) {
            Ok(canonical) => results.push(canonical),
            Err(e) => errors.push(e),
        }
    }
    
    if errors.is_empty() {
        Ok(results)
    } else {
        Err(errors)
    }
}

/// Parallel conversion for large datasets
#[cfg(feature = "parallel_validation")]
/// TODO: Add docs
pub fn parallel_batch_convert<L, C, F>(legacy_items: Vec<L>, converter: F) -> Result<Vec<C>, Vec<ConversionError>>
where
    L: Send,
    C: Send,
    F: Fn(L) -> Result<C, ConversionError> + Send + Sync + Clone,
{
    use rayon::prelude::*;
    
    let results: Vec<_> = legacy_items
        .into_par_iter()
        .map(converter)
        .collect();
    
    let mut successes = Vec::new();
    let mut errors = Vec::new();
    
    for result in results {
        match result {
            Ok(canonical) => successes.push(canonical),
            Err(e) => errors.push(e),
        }
    }
    
    if errors.is_empty() {
        Ok(successes)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_price_conversion_from_decimal() {
        let decimal = dec!(100.50);
        let price: Price = decimal.try_into().unwrap();
        assert_eq!(price.as_decimal(), decimal);
    }
    
    #[test]
    fn test_quantity_conversion_from_f64() {
        let float = 0.123;
        let quantity: Quantity = float.try_into().unwrap();
        assert!((quantity.as_f64() - float).abs() < 0.000001);
    }
    
    #[test]
    fn test_parse_order_side() {
        assert_eq!(parse_order_side("BUY").unwrap(), OrderSide::Buy);
        assert_eq!(parse_order_side("sell").unwrap(), OrderSide::Sell);
        assert_eq!(parse_order_side("LONG").unwrap(), OrderSide::Buy);
        assert_eq!(parse_order_side("SHORT").unwrap(), OrderSide::Sell);
    }
    
    #[test]
    fn test_parse_order_type() {
        assert_eq!(parse_order_type("MARKET").unwrap(), OrderType::Market);
        assert_eq!(parse_order_type("limit").unwrap(), OrderType::Limit);
        assert_eq!(parse_order_type("STOP_LOSS").unwrap(), OrderType::StopMarket);
    }
    
    #[test]
    fn test_parse_candle_interval() {
        assert_eq!(parse_candle_interval("1m").unwrap(), CandleInterval::Minute1);
        assert_eq!(parse_candle_interval("1h").unwrap(), CandleInterval::Hour1);
        assert_eq!(parse_candle_interval("1d").unwrap(), CandleInterval::Day1);
    }
    
    #[test]
    fn test_batch_conversion() {
        let decimals = vec![dec!(100), dec!(200), dec!(300)];
        let converter = |d: Decimal| Price::new(d).map_err(|e| ConversionError::InvalidPrice(e.to_string()));
        
        let prices = batch_convert(decimals, converter).unwrap();
        assert_eq!(prices.len(), 3);
        assert_eq!(prices[0].as_decimal(), dec!(100));
        assert_eq!(prices[1].as_decimal(), dec!(200));
        assert_eq!(prices[2].as_decimal(), dec!(300));
    }
}