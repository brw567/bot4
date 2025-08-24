// Unified Type System - Solving API Mismatch Problem
// Team: Sam (Architecture) + Jordan (Performance) + Full Team
// CRITICAL: All components MUST use these types for communication
// References: 
// - IEEE 754 for numerical precision
// - FIX protocol for financial data types
// - Quantlib for risk calculations

use rust_decimal::Decimal;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use std::ops::{Add, Sub, Mul, Div};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Unified Price type - ALWAYS use this for prices
/// Sam: "No more f32 vs Decimal confusion!"
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Price(Decimal);

impl Price {
    pub const ZERO: Price = Price(Decimal::ZERO);
    pub const ONE: Price = Price(Decimal::ONE);
    
    #[inline(always)]
    pub fn new(value: Decimal) -> Self {
        Price(value)
    }
    
    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        Price(Decimal::from_f64(value).unwrap_or(Decimal::ZERO))
    }
    
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Price(Decimal::from_f32(value).unwrap_or(Decimal::ZERO))
    }
    
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap_or(0.0)
    }
    
    #[inline(always)]
    pub fn to_f32(&self) -> f32 {
        self.0.to_f32().unwrap_or(0.0)
    }
    
    #[inline(always)]
    pub fn inner(&self) -> Decimal {
        self.0
    }
}

/// Unified Quantity type - for position sizes, volumes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Quantity(Decimal);

impl Quantity {
    pub const ZERO: Quantity = Quantity(Decimal::ZERO);
    
    #[inline(always)]
    pub fn new(value: Decimal) -> Self {
        Quantity(value.abs()) // Quantities are always positive
    }
    
    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        Quantity::new(Decimal::from_f64(value.abs()).unwrap_or(Decimal::ZERO))
    }
    
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap_or(0.0)
    }
    
    #[inline(always)]
    pub fn inner(&self) -> Decimal {
        self.0
    }
}

/// Unified Percentage type - for returns, volatility, etc.
/// Jordan: "Percentages should be 0.01 for 1%, not 1.0!"
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Percentage(f64);

impl Percentage {
    pub const ZERO: Percentage = Percentage(0.0);
    pub const ONE_PERCENT: Percentage = Percentage(0.01);
    pub const ONE_HUNDRED_PERCENT: Percentage = Percentage(1.0);
    
    #[inline(always)]
    pub fn new(value: f64) -> Self {
        Percentage(value)
    }
    
    #[inline(always)]
    pub fn from_basis_points(bps: f64) -> Self {
        Percentage(bps / 10000.0)
    }
    
    #[inline(always)]
    pub fn to_basis_points(&self) -> f64 {
        self.0 * 10000.0
    }
    
    #[inline(always)]
    pub fn as_decimal(&self) -> Decimal {
        Decimal::from_f64(self.0).unwrap_or(Decimal::ZERO)
    }
    
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.0
    }
    
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        self.0
    }
}

/// Risk Metrics - unified structure for all risk calculations
/// Quinn: "Everything in one place, properly typed!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub position_size: Quantity,
    pub confidence: Percentage,
    pub expected_return: Percentage,
    pub volatility: Percentage,
    pub var_limit: Percentage,
    pub sharpe_ratio: f64,
    pub kelly_fraction: Percentage,
    pub max_drawdown: Percentage,
    pub current_heat: Percentage,
    pub leverage: f64,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            position_size: Quantity::ZERO,
            confidence: Percentage::new(0.5),
            expected_return: Percentage::ZERO,
            volatility: Percentage::new(0.15),
            var_limit: Percentage::new(0.02),
            sharpe_ratio: 0.0,
            kelly_fraction: Percentage::new(0.25),
            max_drawdown: Percentage::ZERO,
            current_heat: Percentage::ZERO,
            leverage: 1.0,
        }
    }
}

/// Market Data - unified structure for price/volume data
/// Casey: "Consistent data format for all exchanges!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: u64,
    pub bid: Price,
    pub ask: Price,
    pub last: Price,
    pub volume: Quantity,
    pub bid_size: Quantity,
    pub ask_size: Quantity,
    pub spread: Price,
    pub mid: Price,
}

impl MarketData {
    pub fn spread_percentage(&self) -> Percentage {
        let spread_val = (self.ask.0 - self.bid.0) / self.mid.0;
        Percentage::new(spread_val.to_f64().unwrap_or(0.0))
    }
    
    pub fn is_liquid(&self, min_volume: Quantity) -> bool {
        self.volume >= min_volume && self.spread_percentage() < Percentage::new(0.002)
    }
}

/// Position - unified position representation
/// Morgan: "Track everything needed for ML features!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: Side,
    pub quantity: Quantity,
    pub entry_price: Price,
    pub current_price: Price,
    pub unrealized_pnl: Price,
    pub realized_pnl: Price,
    pub holding_period: u64, // seconds
    pub max_profit: Price,
    pub max_loss: Price,
}

impl Position {
    pub fn pnl_percentage(&self) -> Percentage {
        let pnl_ratio = self.unrealized_pnl.0 / (self.entry_price.0 * self.quantity.0);
        Percentage::new(pnl_ratio.to_f64().unwrap_or(0.0))
    }
    
    pub fn is_profitable(&self) -> bool {
        self.unrealized_pnl > Price::ZERO
    }
    
    pub fn time_weighted_return(&self) -> Percentage {
        let annual_seconds = 365.25 * 24.0 * 3600.0;
        let time_factor = self.holding_period as f64 / annual_seconds;
        let return_pct = self.pnl_percentage().value();
        
        if time_factor > 0.0 {
            Percentage::new(return_pct / time_factor)
        } else {
            Percentage::ZERO
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

/// Trading Signal - unified signal format
/// Alex: "All strategies output this format!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub timestamp: u64,
    pub symbol: String,
    pub action: SignalAction,
    pub confidence: Percentage,
    pub size: Quantity,
    pub reason: String,
    pub risk_metrics: RiskMetrics,
    pub ml_features: Vec<f64>,
    pub ta_indicators: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalAction {
    Buy,
    Sell,
    Hold,
    ClosePosition,
    ReducePosition,
    IncreasePosition,
}

/// Order - unified order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedOrder {
    pub id: uuid::Uuid,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: Quantity,
    pub price: Option<Price>,
    pub stop_price: Option<Price>,
    pub take_profit: Option<Price>,
    pub time_in_force: TimeInForce,
    pub reduce_only: bool,
    pub post_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TakeProfit,
    TakeProfitLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC,  // Good Till Cancel
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
    GTX,  // Good Till Crossing
}

// Arithmetic implementations moved to unified_type_ops.rs

/// Type conversion utilities
/// Jordan: "Zero-cost abstractions with inline!"
pub mod conversions {
    use super::*;
    
    #[inline(always)]
    pub fn decimal_to_f64(d: Decimal) -> f64 {
        d.to_f64().unwrap_or(0.0)
    }
    
    #[inline(always)]
    pub fn f64_to_decimal(f: f64) -> Decimal {
        Decimal::from_f64(f).unwrap_or(Decimal::ZERO)
    }
    
    #[inline(always)]
    pub fn percentage_to_decimal(p: Percentage) -> Decimal {
        Decimal::from_f64(p.0).unwrap_or(Decimal::ZERO)
    }
    
    #[inline(always)]
    pub fn decimal_to_percentage(d: Decimal) -> Percentage {
        Percentage::new(d.to_f64().unwrap_or(0.0))
    }
}

// External dependencies
use uuid;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_price_arithmetic() {
        let p1 = Price::from_f64(100.0);
        let p2 = Price::from_f64(50.0);
        let q = Quantity::from_f64(2.0);
        
        assert_eq!((p1 + p2).to_f64(), 150.0);
        assert_eq!((p1 - p2).to_f64(), 50.0);
        assert_eq!((p1 * q).to_f64(), 200.0);
    }
    
    #[test]
    fn test_percentage_conversions() {
        let pct = Percentage::from_basis_points(100.0);
        assert_eq!(pct.value(), 0.01);
        assert_eq!(pct.to_basis_points(), 100.0);
        
        let pct2 = Percentage::new(0.15);
        assert_eq!(format!("{}", pct2), "15.00%");
    }
    
    #[test]
    fn test_position_pnl() {
        let pos = Position {
            symbol: "BTC/USDT".to_string(),
            side: Side::Long,
            quantity: Quantity::from_f64(1.0),
            entry_price: Price::from_f64(50000.0),
            current_price: Price::from_f64(55000.0),
            unrealized_pnl: Price::from_f64(5000.0),
            realized_pnl: Price::ZERO,
            holding_period: 86400, // 1 day
            max_profit: Price::from_f64(6000.0),
            max_loss: Price::from_f64(-1000.0),
        };
        
        assert!(pos.is_profitable());
        assert_eq!(pos.pnl_percentage().value(), 0.10); // 10% profit
    }
}

// Alex: "THIS is how you solve API mismatches - unified types!"
// Sam: "Type safety with zero-cost abstractions!"
// Jordan: "All inline for maximum performance!"
// Quinn: "Risk metrics properly structured!"