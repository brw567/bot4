//! # UNIFIED TYPE SYSTEM - Zero Duplicates
//! Karl: "One source of truth for every type"

// Re-export all canonical types
pub use crate::price::Price;
pub use crate::quantity::Quantity;
pub use crate::order::{Order, OrderId, OrderStatus, OrderType, OrderSide};
pub use crate::position_canonical::{Position, PositionId, PositionStatus};
pub use crate::trade::{Trade, TradeId, TradeStatus};
pub use crate::candle::Candle;
pub use crate::market_data::{OrderBook, OrderBookLevel, Tick};

// Type aliases for legacy compatibility
pub type Money = crate::money::Money;
pub type Currency = crate::currency::Currency;
pub type TradingPair = crate::trading_pair::TradingPair;

// Ensure all modules use these canonical types
pub mod prelude {
    pub use super::{
        Price, Quantity, Order, OrderId, OrderStatus, OrderType, OrderSide,
        Position, PositionId, PositionStatus,
        Trade, TradeId, TradeStatus,
        Candle, OrderBook, OrderBookLevel, Tick,
        Money, Currency, TradingPair,
    };
}

// Karl: "This is the way. No duplicates, only unity."
