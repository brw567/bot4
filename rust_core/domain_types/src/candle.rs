//! # Canonical Candle Type (OHLCV)
//! 
//! Consolidates 6 different Candle struct definitions into one canonical type.
//! Represents price action over a time period with volume.
//!
//! ## Design Decisions
//! - Immutable OHLCV data
//! - Multiple timeframe support
//! - Volume profile information
//! - Trade count and other microstructure data
//!
//! ## External Research Applied
//! - Japanese Candlestick Charting (Nison)
//! - Volume Profile Analysis
//! - Market Profile Theory (Steidlmayer)

use crate::{Price, Quantity};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::fmt;
use strum_macros::{Display, EnumString};

/// Candle time intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
pub enum CandleInterval {
    /// 1 second
    #[strum(serialize = "1s")]
    Second1,
    /// 5 seconds
    #[strum(serialize = "5s")]
    Second5,
    /// 15 seconds
    #[strum(serialize = "15s")]
    Second15,
    /// 30 seconds
    #[strum(serialize = "30s")]
    Second30,
    /// 1 minute
    #[strum(serialize = "1m")]
    Minute1,
    /// 3 minutes
    #[strum(serialize = "3m")]
    Minute3,
    /// 5 minutes
    #[strum(serialize = "5m")]
    Minute5,
    /// 15 minutes
    #[strum(serialize = "15m")]
    Minute15,
    /// 30 minutes
    #[strum(serialize = "30m")]
    Minute30,
    /// 1 hour
    #[strum(serialize = "1h")]
    Hour1,
    /// 2 hours
    #[strum(serialize = "2h")]
    Hour2,
    /// 4 hours
    #[strum(serialize = "4h")]
    Hour4,
    /// 6 hours
    #[strum(serialize = "6h")]
    Hour6,
    /// 8 hours
    #[strum(serialize = "8h")]
    Hour8,
    /// 12 hours
    #[strum(serialize = "12h")]
    Hour12,
    /// 1 day
    #[strum(serialize = "1d")]
    Day1,
    /// 3 days
    #[strum(serialize = "3d")]
    Day3,
    /// 1 week
    #[strum(serialize = "1w")]
    Week1,
    /// 1 month
    #[strum(serialize = "1M")]
    Month1,
}

impl CandleInterval {
    /// Gets the duration of this interval
    pub fn duration(&self) -> Duration {
        match self {
            Self::Second1 => Duration::seconds(1),
            Self::Second5 => Duration::seconds(5),
            Self::Second15 => Duration::seconds(15),
            Self::Second30 => Duration::seconds(30),
            Self::Minute1 => Duration::minutes(1),
            Self::Minute3 => Duration::minutes(3),
            Self::Minute5 => Duration::minutes(5),
            Self::Minute15 => Duration::minutes(15),
            Self::Minute30 => Duration::minutes(30),
            Self::Hour1 => Duration::hours(1),
            Self::Hour2 => Duration::hours(2),
            Self::Hour4 => Duration::hours(4),
            Self::Hour6 => Duration::hours(6),
            Self::Hour8 => Duration::hours(8),
            Self::Hour12 => Duration::hours(12),
            Self::Day1 => Duration::days(1),
            Self::Day3 => Duration::days(3),
            Self::Week1 => Duration::weeks(1),
            Self::Month1 => Duration::days(30), // Approximate
        }
    }
    
    /// Gets the number of seconds in this interval
    pub fn seconds(&self) -> i64 {
        self.duration().num_seconds()
    }
    
    /// Checks if this is an intraday interval
    pub fn is_intraday(&self) -> bool {
        matches!(
            self,
            Self::Second1 | Self::Second5 | Self::Second15 | Self::Second30 |
            Self::Minute1 | Self::Minute3 | Self::Minute5 | Self::Minute15 | Self::Minute30 |
            Self::Hour1 | Self::Hour2 | Self::Hour4 | Self::Hour6 | Self::Hour8 | Self::Hour12
        )
    }
}

/// Canonical Candle type (OHLCV data)
///
/// # Invariants
/// - High >= Max(Open, Close)
/// - Low <= Min(Open, Close)
/// - High >= Low
/// - Volume >= 0
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    // === Core OHLCV ===
    /// Opening price
    pub open: Price,
    /// Highest price
    pub high: Price,
    /// Lowest price
    pub low: Price,
    /// Closing price
    pub close: Price,
    /// Volume traded
    pub volume: Quantity,
    
    // === Time Information ===
    /// Start time of candle
    pub open_time: DateTime<Utc>,
    /// End time of candle
    pub close_time: DateTime<Utc>,
    /// Time interval
    pub interval: CandleInterval,
    
    // === Additional Metrics ===
    /// Number of trades
    pub trade_count: u64,
    /// Quote asset volume (e.g., USDT volume)
    pub quote_volume: Decimal,
    /// Taker buy base volume
    pub taker_buy_volume: Quantity,
    /// Taker buy quote volume
    pub taker_buy_quote_volume: Decimal,
    
    // === Derived Metrics ===
    /// Whether candle is complete
    pub is_complete: bool,
    /// Whether this is a green candle (close > open)
    pub is_bullish: bool,
    /// Body size (abs(close - open))
    pub body_size: Price,
    /// Upper shadow size
    pub upper_shadow: Price,
    /// Lower shadow size
    pub lower_shadow: Price,
    /// Range (high - low)
    pub range: Price,
}

impl Candle {
    /// Creates a new candle
    pub fn new(
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: Quantity,
        open_time: DateTime<Utc>,
        interval: CandleInterval,
    ) -> Result<Self, String> {
        // Validate invariants
        if high < open || high < close {
            return Err(format!("High {} must be >= max(open {}, close {})", high, open, close));
        }
        
        if low > open || low > close {
            return Err(format!("Low {} must be <= min(open {}, close {})", low, open, close));
        }
        
        if high < low {
            return Err(format!("High {} must be >= low {}", high, low));
        }
        
        let close_time = open_time + interval.duration();
        let is_bullish = close > open;
        
        // Calculate derived metrics
        let body_size = if close >= open {
            close.subtract(open).unwrap_or(Price::zero())
        } else {
            open.subtract(close).unwrap_or(Price::zero())
        };
        
        let candle_high = if close >= open { close } else { open };
        let candle_low = if close >= open { open } else { close };
        
        let upper_shadow = high.subtract(candle_high).unwrap_or(Price::zero());
        let lower_shadow = candle_low.subtract(low).unwrap_or(Price::zero());
        let range = high.subtract(low).unwrap_or(Price::zero());
        
        Ok(Self {
            open,
            high,
            low,
            close,
            volume,
            open_time,
            close_time,
            interval,
            trade_count: 0,
            quote_volume: Decimal::ZERO,
            taker_buy_volume: Quantity::zero(),
            taker_buy_quote_volume: Decimal::ZERO,
            is_complete: false,
            is_bullish,
            body_size,
            upper_shadow,
            lower_shadow,
            range,
        })
    }
    
    /// Creates an empty candle for initialization
    pub fn empty(open_time: DateTime<Utc>, interval: CandleInterval) -> Self {
        let price = Price::zero();
        Self {
            open: price,
            high: price,
            low: price,
            close: price,
            volume: Quantity::zero(),
            open_time,
            close_time: open_time + interval.duration(),
            interval,
            trade_count: 0,
            quote_volume: Decimal::ZERO,
            taker_buy_volume: Quantity::zero(),
            taker_buy_quote_volume: Decimal::ZERO,
            is_complete: false,
            is_bullish: false,
            body_size: price,
            upper_shadow: price,
            lower_shadow: price,
            range: price,
        }
    }
    
    /// Updates candle with a new price and volume
    pub fn update(&mut self, price: Price, volume: Quantity) {
        // Update OHLC
        if self.volume.is_zero() {
            // First update
            self.open = price;
            self.high = price;
            self.low = price;
        } else {
            if price > self.high {
                self.high = price;
            }
            if price < self.low {
                self.low = price;
            }
        }
        self.close = price;
        
        // Update volume
        self.volume = self.volume.add(volume).unwrap_or(self.volume);
        self.trade_count += 1;
        
        // Recalculate derived metrics
        self.is_bullish = self.close > self.open;
        self.body_size = if self.close >= self.open {
            self.close.subtract(self.open).unwrap_or(Price::zero())
        } else {
            self.open.subtract(self.close).unwrap_or(Price::zero())
        };
        
        let candle_high = if self.close >= self.open { self.close } else { self.open };
        let candle_low = if self.close >= self.open { self.open } else { self.close };
        
        self.upper_shadow = self.high.subtract(candle_high).unwrap_or(Price::zero());
        self.lower_shadow = candle_low.subtract(self.low).unwrap_or(Price::zero());
        self.range = self.high.subtract(self.low).unwrap_or(Price::zero());
    }
    
    /// Marks candle as complete
    pub fn complete(&mut self) {
        self.is_complete = true;
    }
    
    /// Gets the typical price (HLC/3)
    pub fn typical_price(&self) -> Price {
        let sum = self.high.as_decimal() + self.low.as_decimal() + self.close.as_decimal();
        Price::new(sum / Decimal::from(3)).unwrap_or(self.close)
    }
    
    /// Gets the weighted close (HLC + Close)/4
    pub fn weighted_close(&self) -> Price {
        let sum = self.high.as_decimal() + self.low.as_decimal() + 
                  self.close.as_decimal() * Decimal::from(2);
        Price::new(sum / Decimal::from(4)).unwrap_or(self.close)
    }
    
    /// Gets the midpoint (H + L)/2
    pub fn midpoint(&self) -> Price {
        self.high.midpoint(self.low).unwrap_or(self.close)
    }
    
    /// Checks if this is a doji candle (small body)
    pub fn is_doji(&self, threshold_percent: Decimal) -> bool {
        if self.range.is_zero() {
            return true;
        }
        
        let body_percent = (self.body_size.as_decimal() / self.range.as_decimal()) 
            * Decimal::from(100);
        body_percent < threshold_percent
    }
    
    /// Checks if this is a hammer pattern
    pub fn is_hammer(&self) -> bool {
        // Hammer: Small body at top, long lower shadow
        let body_to_range = self.body_size.as_decimal() / self.range.as_decimal();
        let lower_to_range = self.lower_shadow.as_decimal() / self.range.as_decimal();
        
        body_to_range < Decimal::from_str("0.3").unwrap() &&
        lower_to_range > Decimal::from_str("0.6").unwrap()
    }
    
    /// Checks if this is a shooting star pattern
    pub fn is_shooting_star(&self) -> bool {
        // Shooting star: Small body at bottom, long upper shadow
        let body_to_range = self.body_size.as_decimal() / self.range.as_decimal();
        let upper_to_range = self.upper_shadow.as_decimal() / self.range.as_decimal();
        
        body_to_range < Decimal::from_str("0.3").unwrap() &&
        upper_to_range > Decimal::from_str("0.6").unwrap()
    }
    
    /// Gets percentage change from open to close
    pub fn change_percent(&self) -> Decimal {
        if self.open.is_zero() {
            return Decimal::ZERO;
        }
        
        ((self.close.as_decimal() - self.open.as_decimal()) / self.open.as_decimal()) 
            * Decimal::from(100)
    }
}

impl fmt::Display for Candle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Candle {} O:{} H:{} L:{} C:{} V:{} {}",
            self.interval,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            if self.is_bullish { "↑" } else { "↓" }
        )
    }
}

/// Series of candles for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleSeries {
    /// Symbol for this series
    pub symbol: String,
    /// Interval of candles
    pub interval: CandleInterval,
    /// The candles in chronological order
    pub candles: Vec<Candle>,
}

impl CandleSeries {
    /// Creates a new candle series
    pub fn new(symbol: String, interval: CandleInterval) -> Self {
        Self {
            symbol,
            interval,
            candles: Vec::new(),
        }
    }
    
    /// Adds a candle to the series
    pub fn add(&mut self, candle: Candle) {
        self.candles.push(candle);
    }
    
    /// Gets the most recent candle
    pub fn latest(&self) -> Option<&Candle> {
        self.candles.last()
    }
    
    /// Gets candles for a time range
    pub fn range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&Candle> {
        self.candles
            .iter()
            .filter(|c| c.open_time >= start && c.close_time <= end)
            .collect()
    }
    
    /// Calculates simple moving average of close prices
    pub fn sma(&self, period: usize) -> Option<Decimal> {
        if self.candles.len() < period {
            return None;
        }
        
        let sum: Decimal = self.candles
            .iter()
            .rev()
            .take(period)
            .map(|c| c.close.as_decimal())
            .sum();
            
        Some(sum / Decimal::from(period))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_candle_creation() {
        let open = Price::new(dec!(100)).unwrap();
        let high = Price::new(dec!(110)).unwrap();
        let low = Price::new(dec!(95)).unwrap();
        let close = Price::new(dec!(105)).unwrap();
        let volume = Quantity::new(dec!(1000)).unwrap();
        let open_time = Utc::now();
        
        let candle = Candle::new(
            open,
            high,
            low,
            close,
            volume,
            open_time,
            CandleInterval::Minute1,
        ).unwrap();
        
        assert!(candle.is_bullish);
        assert_eq!(candle.body_size.as_decimal(), dec!(5));
        assert_eq!(candle.range.as_decimal(), dec!(15));
    }
    
    #[test]
    fn test_candle_validation() {
        let price = Price::new(dec!(100)).unwrap();
        let high = Price::new(dec!(90)).unwrap(); // Invalid: high < open
        let volume = Quantity::new(dec!(1000)).unwrap();
        
        let result = Candle::new(
            price,
            high,
            price,
            price,
            volume,
            Utc::now(),
            CandleInterval::Minute1,
        );
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_candle_patterns() {
        // Test doji detection
        let open = Price::new(dec!(100)).unwrap();
        let high = Price::new(dec!(101)).unwrap();
        let low = Price::new(dec!(99)).unwrap();
        let close = Price::new(dec!(100.1)).unwrap();
        let volume = Quantity::new(dec!(100)).unwrap();
        
        let candle = Candle::new(
            open,
            high,
            low,
            close,
            volume,
            Utc::now(),
            CandleInterval::Minute1,
        ).unwrap();
        
        assert!(candle.is_doji(dec!(10))); // Body is < 10% of range
    }
}