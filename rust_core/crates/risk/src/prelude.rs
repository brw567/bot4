// RISK CRATE PRELUDE - All types in one place
// Team: Sam (Architecture) + Full Team
// Purpose: Single import for all risk types

// Re-export all unified types
pub use crate::unified_types::*;

// Re-export complete trading types
pub use crate::trading_types_complete::{
    EnhancedTradingSignal,
    EnhancedOrderBook,
    CompleteMarketData,
    ExecutionAlgorithm,
    SentimentData,
    AssetClass,
    OptimizationStrategy,
    OptimizationDirection,
    SignalAction,
    TimeInForce,
    MarketRegime,
    OrderLevel,
    Trade,
    TradeSide,
    OrderFlow,
    MarketConditions,
    BacktestMetrics,
    RiskParameters,
};

// Type aliases for compatibility
pub type OrderBook = EnhancedOrderBook;
pub type MarketData = CompleteMarketData;
pub type TradingSignal = EnhancedTradingSignal;

// Re-export compatibility functions
pub use crate::type_compatibility::{
    enhance_signal,
    create_default_sentiment,
    create_enhanced_order_book,
};

// Missing value that some modules expect
pub const tail_risk: f64 = 0.05;  // 5% tail risk threshold

// Chrono re-export for Utc
pub use chrono::{DateTime, Utc};