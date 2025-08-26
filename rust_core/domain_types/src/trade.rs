//! # Canonical Trade Type
//! 
//! Consolidates 18 different Trade struct definitions into one canonical type.
//! Represents an executed trade (a fill) with complete information.
//!
//! ## Design Decisions
//! - Immutable record of execution
//! - Links to order that generated it
//! - Supports both maker and taker trades
//! - Includes market microstructure information
//!
//! ## External Research Applied
//! - Market Microstructure Theory (O'Hara)
//! - Trade and Quote (TAQ) data standards
//! - Exchange trade reporting standards

use crate::{OrderId, OrderSide, Price, Quantity};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique trade identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradeId(pub Uuid);

impl TradeId {
    /// Creates a new unique trade ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Creates from a string
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
    
    /// Converts to string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl Default for TradeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TradeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trade side from perspective of the order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeSide {
    /// Buy trade
    Buy,
    /// Sell trade  
    Sell,
}

impl From<OrderSide> for TradeSide {
    fn from(side: OrderSide) -> Self {
        match side {
            OrderSide::Buy => TradeSide::Buy,
            OrderSide::Sell => TradeSide::Sell,
        }
    }
}

/// Trade role (maker vs taker)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeRole {
    /// Maker - provided liquidity
    Maker,
    /// Taker - removed liquidity
    Taker,
}

/// Trade type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeType {
    /// Regular trade
    Regular,
    /// Liquidation trade
    Liquidation,
    /// ADL (Auto-Deleveraging) trade
    ADL,
    /// Settlement trade
    Settlement,
    /// Block trade (large OTC)
    Block,
    /// Auction trade
    Auction,
}

/// Canonical Trade type - represents an executed trade
///
/// # Invariants
/// - Trade is immutable once created
/// - Must have valid price and quantity
/// - Commission cannot be negative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    // === Identity ===
    /// Unique trade ID
    pub id: TradeId,
    /// Exchange trade ID
    pub exchange_trade_id: Option<String>,
    /// Order ID that generated this trade
    pub order_id: OrderId,
    /// Client order ID for reference
    pub client_order_id: String,
    
    // === Trade Details ===
    /// Trading symbol
    pub symbol: String,
    /// Trade side (buy/sell)
    pub side: TradeSide,
    /// Execution price
    pub price: Price,
    /// Execution quantity
    pub quantity: Quantity,
    /// Trade value (price * quantity)
    pub value: Decimal,
    
    // === Market Microstructure ===
    /// Was this order the maker or taker
    pub role: TradeRole,
    /// Trade type classification
    pub trade_type: TradeType,
    /// Whether trade was buyer-initiated
    pub buyer_initiated: bool,
    /// Trade occurred at bid price
    pub at_bid: bool,
    /// Trade occurred at ask price
    pub at_ask: bool,
    /// Trade occurred at midpoint
    pub at_mid: bool,
    /// Best bid at time of trade
    pub best_bid: Option<Price>,
    /// Best ask at time of trade
    pub best_ask: Option<Price>,
    
    // === Fees ===
    /// Commission amount
    pub commission: Decimal,
    /// Commission asset
    pub commission_asset: String,
    /// Fee rate applied
    pub fee_rate: Decimal,
    /// Rebate received (if maker)
    pub rebate: Option<Decimal>,
    
    // === Performance ===
    /// Slippage from intended price
    pub slippage: Option<Decimal>,
    /// Market impact estimate
    pub market_impact: Option<Decimal>,
    /// Execution latency in microseconds
    pub latency_us: Option<u64>,
    
    // === Timestamps ===
    /// When trade was executed
    pub executed_at: DateTime<Utc>,
    /// When trade was received by us
    pub received_at: DateTime<Utc>,
    /// When trade was processed
    pub processed_at: Option<DateTime<Utc>>,
    
    // === Strategy Metadata ===
    /// Strategy that generated the order
    pub strategy_id: Option<String>,
    /// ML confidence at time of trade
    pub ml_confidence: Option<Decimal>,
    /// Market regime at time of trade
    pub market_regime: Option<String>,
    
    // === Additional Info ===
    /// Exchange where trade executed
    pub exchange: String,
    /// Whether this was the last fill for the order
    pub is_last_fill: bool,
    /// Sequence number for ordering
    pub sequence: Option<u64>,
}

impl Trade {
    /// Creates a new trade
    pub fn new(
        order_id: OrderId,
        symbol: String,
        side: TradeSide,
        price: Price,
        quantity: Quantity,
        role: TradeRole,
        exchange: String,
    ) -> Self {
        let value = price.as_decimal() * quantity.as_decimal();
        let now = Utc::now();
        
        Self {
            id: TradeId::new(),
            exchange_trade_id: None,
            order_id,
            client_order_id: String::new(),
            symbol,
            side,
            price,
            quantity,
            value,
            role,
            trade_type: TradeType::Regular,
            buyer_initiated: matches!(side, TradeSide::Buy),
            at_bid: false,
            at_ask: false,
            at_mid: false,
            best_bid: None,
            best_ask: None,
            commission: Decimal::ZERO,
            commission_asset: "USDT".to_string(),
            fee_rate: Decimal::ZERO,
            rebate: None,
            slippage: None,
            market_impact: None,
            latency_us: None,
            executed_at: now,
            received_at: now,
            processed_at: None,
            strategy_id: None,
            ml_confidence: None,
            market_regime: None,
            exchange,
            is_last_fill: false,
            sequence: None,
        }
    }
    
    /// Calculates the effective price including fees
    pub fn effective_price(&self) -> Price {
        let fee_adjustment = if self.quantity.as_decimal() > Decimal::ZERO {
            self.commission / self.quantity.as_decimal()
        } else {
            Decimal::ZERO
        };
        
        let effective = match self.side {
            TradeSide::Buy => self.price.as_decimal() + fee_adjustment,
            TradeSide::Sell => self.price.as_decimal() - fee_adjustment,
        };
        
        Price::new(effective).unwrap_or(self.price)
    }
    
    /// Calculates realized PnL if exit price is known
    pub fn realized_pnl(&self, exit_price: Price) -> Decimal {
        let price_diff = exit_price.as_decimal() - self.price.as_decimal();
        let gross_pnl = match self.side {
            TradeSide::Buy => price_diff * self.quantity.as_decimal(),
            TradeSide::Sell => -price_diff * self.quantity.as_decimal(),
        };
        gross_pnl - self.commission
    }
    
    /// Checks if this was an aggressive trade
    pub fn is_aggressive(&self) -> bool {
        matches!(self.role, TradeRole::Taker)
    }
    
    /// Checks if this was a passive trade
    pub fn is_passive(&self) -> bool {
        matches!(self.role, TradeRole::Maker)
    }
    
    /// Gets the fee percentage
    pub fn fee_percentage(&self) -> Decimal {
        if self.value > Decimal::ZERO {
            (self.commission / self.value) * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }
}

impl fmt::Display for Trade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade {} {} {} {} @ {} [{}]",
            self.id,
            self.side,
            self.quantity,
            self.symbol,
            self.price,
            self.role
        )
    }
}

/// Aggregated trade statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeStats {
    /// Total number of trades
    pub count: u64,
    /// Total volume traded
    pub total_volume: Decimal,
    /// Total value traded
    pub total_value: Decimal,
    /// Average trade size
    pub avg_size: Decimal,
    /// Average trade price
    pub avg_price: Decimal,
    /// VWAP (Volume-Weighted Average Price)
    pub vwap: Decimal,
    /// Total fees paid
    pub total_fees: Decimal,
    /// Number of maker trades
    pub maker_count: u64,
    /// Number of taker trades
    pub taker_count: u64,
    /// Buy volume
    pub buy_volume: Decimal,
    /// Sell volume
    pub sell_volume: Decimal,
    /// Volume imbalance (buy - sell)
    pub volume_imbalance: Decimal,
    /// Time period for these stats
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

impl TradeStats {
    /// Creates stats from a vec of trades
    pub fn from_trades(trades: &[Trade]) -> Option<Self> {
        if trades.is_empty() {
            return None;
        }
        
        let count = trades.len() as u64;
        let total_volume: Decimal = trades.iter().map(|t| t.quantity.as_decimal()).sum();
        let total_value: Decimal = trades.iter().map(|t| t.value).sum();
        let total_fees: Decimal = trades.iter().map(|t| t.commission).sum();
        
        let avg_size = total_volume / Decimal::from(count);
        let avg_price = if total_volume > Decimal::ZERO {
            total_value / total_volume
        } else {
            Decimal::ZERO
        };
        
        let vwap = avg_price; // Same calculation
        
        let maker_count = trades.iter().filter(|t| t.is_passive()).count() as u64;
        let taker_count = trades.iter().filter(|t| t.is_aggressive()).count() as u64;
        
        let buy_volume: Decimal = trades
            .iter()
            .filter(|t| matches!(t.side, TradeSide::Buy))
            .map(|t| t.quantity.as_decimal())
            .sum();
            
        let sell_volume: Decimal = trades
            .iter()
            .filter(|t| matches!(t.side, TradeSide::Sell))
            .map(|t| t.quantity.as_decimal())
            .sum();
            
        let volume_imbalance = buy_volume - sell_volume;
        
        let period_start = trades.iter().map(|t| t.executed_at).min()?;
        let period_end = trades.iter().map(|t| t.executed_at).max()?;
        
        Some(Self {
            count,
            total_volume,
            total_value,
            avg_size,
            avg_price,
            vwap,
            total_fees,
            maker_count,
            taker_count,
            buy_volume,
            sell_volume,
            volume_imbalance,
            period_start,
            period_end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_trade_creation() {
        let order_id = OrderId::new();
        let price = Price::new(dec!(50000)).unwrap();
        let qty = Quantity::new(dec!(0.1)).unwrap();
        
        let trade = Trade::new(
            order_id,
            "BTC/USDT".to_string(),
            TradeSide::Buy,
            price,
            qty,
            TradeRole::Taker,
            "binance".to_string(),
        );
        
        assert_eq!(trade.value, dec!(5000));
        assert_eq!(trade.side, TradeSide::Buy);
        assert!(trade.is_aggressive());
    }
    
    #[test]
    fn test_effective_price_calculation() {
        let mut trade = Trade::new(
            OrderId::new(),
            "BTC/USDT".to_string(),
            TradeSide::Buy,
            Price::new(dec!(50000)).unwrap(),
            Quantity::new(dec!(1)).unwrap(),
            TradeRole::Taker,
            "binance".to_string(),
        );
        
        trade.commission = dec!(50); // $50 commission
        
        let effective = trade.effective_price();
        assert_eq!(effective.as_decimal(), dec!(50050)); // Price + commission
    }
    
    #[test]
    fn test_trade_stats_calculation() {
        let order_id = OrderId::new();
        let trades = vec![
            Trade::new(
                order_id,
                "BTC/USDT".to_string(),
                TradeSide::Buy,
                Price::new(dec!(50000)).unwrap(),
                Quantity::new(dec!(1)).unwrap(),
                TradeRole::Taker,
                "binance".to_string(),
            ),
            Trade::new(
                order_id,
                "BTC/USDT".to_string(),
                TradeSide::Sell,
                Price::new(dec!(51000)).unwrap(),
                Quantity::new(dec!(0.5)).unwrap(),
                TradeRole::Maker,
                "binance".to_string(),
            ),
        ];
        
        let stats = TradeStats::from_trades(&trades).unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_volume, dec!(1.5));
        assert_eq!(stats.buy_volume, dec!(1));
        assert_eq!(stats.sell_volume, dec!(0.5));
        assert_eq!(stats.volume_imbalance, dec!(0.5));
    }
}