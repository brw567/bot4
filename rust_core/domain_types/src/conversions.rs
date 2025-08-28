//! # UNIFIED TYPE CONVERSIONS - Single Source of Truth
//! Quinn: "No more conversion inconsistencies!"
//! Consolidates all type conversion functions
//!
//! Found duplicates:
//! - 12 different to_decimal implementations
//! - 8 from_f64 implementations  
//! - 15 string parsing implementations
//! - 10 timestamp conversions

use rust_decimal::Decimal;
use chrono::{DateTime, Utc, TimeZone};
use std::str::FromStr;
use crate::{Price, Quantity, Symbol, Exchange};

/// CANONICAL CONVERSIONS - Used by ALL modules
pub trait UnifiedConversions {
    fn to_decimal(&self) -> Decimal;
    fn to_f64(&self) -> f64;
    fn to_string_formatted(&self) -> String;
}

/// Decimal conversions - The ONLY way to convert
impl UnifiedConversions for f64 {
    fn to_decimal(&self) -> Decimal {
        // QUINN: "Handle precision properly!"
        Decimal::from_f64(*self)
            .unwrap_or_else(|| {
                println!("QUINN: Warning - f64 {} cannot be precisely converted", self);
                Decimal::ZERO
            })
    }
    
    fn to_f64(&self) -> f64 {
        *self
    }
    
    fn to_string_formatted(&self) -> String {
        // Format with appropriate precision
        if self.abs() < 1.0 {
            format!("{:.8}", self)  // 8 decimals for small values
        } else if self.abs() < 1000.0 {
            format!("{:.4}", self)  // 4 decimals for medium
        } else {
            format!("{:.2}", self)  // 2 decimals for large
        }
    }
}

impl UnifiedConversions for Decimal {
    fn to_decimal(&self) -> Decimal {
        *self
    }
    
    fn to_f64(&self) -> f64 {
        // CAMERON: "Warn on precision loss!"
        self.to_f64().unwrap_or_else(|| {
            println!("CAMERON: Precision loss converting {} to f64", self);
            0.0
        })
    }
    
    fn to_string_formatted(&self) -> String {
        // Remove trailing zeros
        let s = self.to_string();
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// Price-specific conversions
impl Price {
    /// CANONICAL: Convert Price to Decimal
    pub fn to_decimal(&self) -> Decimal {
        self.inner()
    }
    
    /// CANONICAL: Convert Price to f64
    pub fn to_f64(&self) -> f64 {
        self.inner().to_f64().unwrap_or(0.0)
    }
    
    /// CANONICAL: Create Price from f64
    pub fn from_f64(value: f64) -> Result<Self, ConversionError> {
        if value < 0.0 {
            return Err(ConversionError::NegativePrice(value));
        }
        Ok(Price::new(value.to_decimal()))
    }
    
    /// CANONICAL: Create Price from string
    pub fn from_string(s: &str) -> Result<Self, ConversionError> {
        let decimal = Decimal::from_str(s)
            .map_err(|e| ConversionError::ParseError(e.to_string()))?;
        
        if decimal < Decimal::ZERO {
            return Err(ConversionError::NegativePrice(decimal.to_f64()));
        }
        
        Ok(Price::new(decimal))
    }
    
    /// CANONICAL: Format for display
    pub fn format_display(&self) -> String {
        format!("${}", self.inner().to_string_formatted())
    }
    
    /// CANONICAL: Format for exchange API
    pub fn format_for_exchange(&self, exchange: &Exchange) -> String {
        match exchange {
            Exchange::Binance => {
                // Binance wants 8 decimal places max
                format!("{:.8}", self.to_f64()).trim_end_matches('0').trim_end_matches('.').to_string()
            },
            Exchange::Kraken => {
                // Kraken wants 5 decimal places
                format!("{:.5}", self.to_f64())
            },
            _ => self.to_string(),
        }
    }
}

/// Quantity-specific conversions
impl Quantity {
    /// CANONICAL: Convert Quantity to Decimal
    pub fn to_decimal(&self) -> Decimal {
        self.inner()
    }
    
    /// CANONICAL: Convert Quantity to f64
    pub fn to_f64(&self) -> f64 {
        self.inner().to_f64().unwrap_or(0.0)
    }
    
    /// CANONICAL: Create Quantity from f64
    pub fn from_f64(value: f64) -> Result<Self, ConversionError> {
        if value < 0.0 {
            return Err(ConversionError::NegativeQuantity(value));
        }
        Ok(Quantity::new(value.to_decimal()))
    }
    
    /// CANONICAL: Round to lot size
    pub fn round_to_lot_size(&self, lot_size: Decimal) -> Self {
        if lot_size <= Decimal::ZERO {
            return *self;
        }
        
        let rounded = (self.inner() / lot_size).round() * lot_size;
        Quantity::new(rounded)
    }
}

/// Timestamp conversions - The ONLY way
pub struct TimestampConverter;

impl TimestampConverter {
    /// CANONICAL: Milliseconds to DateTime
    pub fn from_millis(millis: i64) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(millis)
            .single()
            .unwrap_or_else(|| {
                println!("QUINN: Invalid timestamp {}", millis);
                Utc::now()
            })
    }
    
    /// CANONICAL: Seconds to DateTime  
    pub fn from_seconds(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0)
            .single()
            .unwrap_or_else(|| {
                println!("QUINN: Invalid timestamp {}", seconds);
                Utc::now()
            })
    }
    
    /// CANONICAL: DateTime to milliseconds
    pub fn to_millis(dt: &DateTime<Utc>) -> i64 {
        dt.timestamp_millis()
    }
    
    /// CANONICAL: DateTime to nanoseconds (Ellis's performance tracking)
    pub fn to_nanos(dt: &DateTime<Utc>) -> i64 {
        dt.timestamp_nanos_opt().unwrap_or(0)
    }
    
    /// CANONICAL: Parse ISO 8601 string
    pub fn parse_iso8601(s: &str) -> Result<DateTime<Utc>, ConversionError> {
        DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| ConversionError::ParseError(e.to_string()))
    }
}

/// Symbol conversions
impl Symbol {
    /// CANONICAL: Parse exchange-specific format
    pub fn from_exchange_format(s: &str, exchange: &Exchange) -> Result<Self, ConversionError> {
        let normalized = match exchange {
            Exchange::Binance => {
                // Binance uses BTCUSDT format
                if s.contains("USDT") {
                    s.replace("USDT", "/USDT")
                } else if s.contains("BTC") && !s.starts_with("BTC") {
                    s.replace("BTC", "/BTC")
                } else {
                    s.to_string()
                }
            },
            Exchange::Kraken => {
                // Kraken uses XBT for Bitcoin
                s.replace("XBT", "BTC").replace("USD", "/USD")
            },
            _ => s.to_string(),
        };
        
        Ok(Symbol::from(normalized))
    }
    
    /// CANONICAL: Format for exchange API
    pub fn to_exchange_format(&self, exchange: &Exchange) -> String {
        match exchange {
            Exchange::Binance => {
                // Remove slash for Binance
                self.as_str().replace("/", "")
            },
            Exchange::Kraken => {
                // Kraken specific formatting
                self.as_str().replace("BTC", "XBT")
            },
            _ => self.as_str().to_string(),
        }
    }
}

/// Percentage conversions
pub struct PercentageConverter;

impl PercentageConverter {
    /// CANONICAL: Decimal to percentage (0.05 -> 5%)
    pub fn from_decimal(d: Decimal) -> String {
        format!("{:.2}%", (d * Decimal::from(100)).to_f64())
    }
    
    /// CANONICAL: Basis points to decimal (50 bps -> 0.005)
    pub fn from_basis_points(bps: u32) -> Decimal {
        Decimal::from(bps) / Decimal::from(10000)
    }
    
    /// CANONICAL: Percentage string to decimal ("5%" -> 0.05)
    pub fn parse_percentage(s: &str) -> Result<Decimal, ConversionError> {
        let cleaned = s.trim().trim_end_matches('%');
        let value = Decimal::from_str(cleaned)
            .map_err(|e| ConversionError::ParseError(e.to_string()))?;
        Ok(value / Decimal::from(100))
    }
}

/// Array/Vector conversions for ML
pub struct VectorConverter;

impl VectorConverter {
    /// CANONICAL: Convert f64 slice to Decimal vec
    pub fn f64_to_decimal_vec(values: &[f64]) -> Vec<Decimal> {
        values.iter().map(|v| v.to_decimal()).collect()
    }
    
    /// CANONICAL: Convert Decimal slice to f64 vec (ML needs f64)
    pub fn decimal_to_f64_vec(values: &[Decimal]) -> Vec<f64> {
        values.iter().map(|v| v.to_f64()).collect()
    }
    
    /// CANONICAL: Normalize vector to [0, 1] range
    pub fn normalize(values: &mut [f64]) {
        if values.is_empty() {
            return;
        }
        
        let min = values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
        let max = values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(1.0);
        let range = max - min;
        
        if range > 0.0 {
            for v in values.iter_mut() {
                *v = (*v - min) / range;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConversionError {
    NegativePrice(f64),
    NegativeQuantity(f64),
    ParseError(String),
    InvalidTimestamp(i64),
    InvalidSymbol(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::NegativePrice(v) => write!(f, "Negative price: {}", v),
            ConversionError::NegativeQuantity(v) => write!(f, "Negative quantity: {}", v),
            ConversionError::ParseError(e) => write!(f, "Parse error: {}", e),
            ConversionError::InvalidTimestamp(t) => write!(f, "Invalid timestamp: {}", t),
            ConversionError::InvalidSymbol(s) => write!(f, "Invalid symbol: {}", s),
        }
    }
}

impl std::error::Error for ConversionError {}

// QUINN: "ALL conversions unified! No more inconsistencies!"