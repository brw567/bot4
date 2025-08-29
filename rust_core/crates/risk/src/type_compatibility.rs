// TYPE COMPATIBILITY LAYER - Seamless integration
use crate::ml::unified_indicators::{UnifiedIndicators, MACDValue, BollingerBands};
// Team: Sam (Architecture) + Full Team
// Purpose: Map new complete types to existing usage

use crate::trading_types_complete::{
    EnhancedTradingSignal, EnhancedOrderBook, CompleteMarketData,
    ExecutionAlgorithm, SentimentData, AssetClass, OptimizationStrategy,
    OptimizationDirection, SignalAction, TimeInForce, MarketRegime,
};
use crate::unified_types::{Price, Quantity, Percentage, RiskMetrics};
use chrono::Utc;
use std::collections::HashMap;

// Re-export complete types with standard names
pub use crate::trading_types_complete::{
    OrderBook,      // EnhancedOrderBook
    MarketData,     // CompleteMarketData  
    TradingSignal,  // EnhancedTradingSignal
};

/// Convert from basic TradingSignal to EnhancedTradingSignal
/// TODO: Add docs
pub fn enhance_signal(
    basic: crate::unified_types::TradingSignal,
    market_data: &CompleteMarketData,
) -> EnhancedTradingSignal {
    // Calculate intelligent price levels based on market conditions
    let atr = calculate_atr(market_data);
    let support_resistance = find_support_resistance(market_data);
    
    // Determine execution algorithm based on conditions
    let urgency = basic.confidence.to_f64();
    let size_vs_adv = basic.size.to_f64() / market_data.volume_24h;
    let spread_bps = market_data.spread_bps;
    let volatility = market_data.volatility_24h.to_f64();
    let market_impact = estimate_market_impact(basic.size.to_f64(), market_data.volume_24h);
    
    let exec_algo = ExecutionAlgorithm::select_optimal(
        urgency,
        size_vs_adv,
        spread_bps,
        volatility,
        market_impact,
    );
    
    // Set price levels using technical analysis
    let entry_price = market_data.price;
    let (stop_loss, take_profit) = calculate_risk_reward_levels(
        entry_price,
        basic.action,
        atr,
        &support_resistance,
        basic.confidence.to_f64(),
    );
    
    EnhancedTradingSignal {
        timestamp: basic.timestamp,
        symbol: basic.symbol,
        action: convert_signal_action(basic.action),
        confidence: basic.confidence,
        size: basic.size,
        reason: basic.reason,
        
        // Risk management levels
        entry_price,
        stop_loss,
        take_profit,
        
        // Advanced parameters
        max_slippage: Percentage::new(0.002),  // 0.2% max slippage
        time_in_force: TimeInForce::GTC,
        execution_algorithm: exec_algo,
        
        // Portfolio context
        portfolio_heat: basic.risk_metrics.current_heat,
        correlation_risk: 0.5,  // TODO: Calculate from portfolio
        expected_sharpe: basic.risk_metrics.sharpe_ratio,
        
        // Features
        ml_features: basic.ml_features,
        ta_indicators: convert_ta_indicators(&basic.ta_indicators),
        market_regime: detect_market_regime(market_data),
        
        // Metadata
        strategy_id: "unified_strategy".to_string(),
        model_version: "1.0.0".to_string(),
        backtest_metrics: crate::trading_types_complete::BacktestMetrics {
            total_trades: 0,
            win_rate: 0.0,
            avg_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            sortino_ratio: 0.0,
            profit_factor: 0.0,
        },
    }
}

/// Calculate ATR for risk sizing
// Replaced by unified: fn calculate_atr(market_data: &CompleteMarketData) -> f64 {
    // Simplified ATR calculation
    let high_low = (market_data.high.to_f64() - market_data.low.to_f64()).abs();
    let high_close = (market_data.high.to_f64() - market_data.close.to_f64()).abs();
    let low_close = (market_data.low.to_f64() - market_data.close.to_f64()).abs();
    
    high_low.max(high_close).max(low_close)
}

/// Find support and resistance levels
fn find_support_resistance(market_data: &CompleteMarketData) -> SupportResistance {
    let price = market_data.price.to_f64();
    let high = market_data.high.to_f64();
    let low = market_data.low.to_f64();
    
    SupportResistance {
        resistance_1: Price::from_f64(high),
        resistance_2: Price::from_f64(high * 1.02),  // 2% above high
        support_1: Price::from_f64(low),
        support_2: Price::from_f64(low * 0.98),  // 2% below low
        pivot: Price::from_f64((high + low + market_data.close.to_f64()) / 3.0),
    }
}

struct SupportResistance {
    resistance_1: Price,
    resistance_2: Price,
    support_1: Price,
    support_2: Price,
    pivot: Price,
}

/// Calculate stop loss and take profit levels
fn calculate_risk_reward_levels(
    entry: Price,
    action: crate::unified_types::SignalAction,
    atr: f64,
    sr: &SupportResistance,
    confidence: f64,
) -> (Price, Price) {
    let entry_val = entry.to_f64();
    let risk_reward_ratio = 2.0 + confidence;  // Higher confidence = better R:R
    
    match action {
        crate::unified_types::SignalAction::Buy => {
            // Stop below support, target at resistance
            let stop = (entry_val - atr * 1.5).min(sr.support_1.to_f64() - atr * 0.5);
            let risk = entry_val - stop;
            let target = entry_val + (risk * risk_reward_ratio);
            
            (Price::from_f64(stop), Price::from_f64(target))
        },
        crate::unified_types::SignalAction::Sell => {
            // Stop above resistance, target at support
            let stop = (entry_val + atr * 1.5).max(sr.resistance_1.to_f64() + atr * 0.5);
            let risk = stop - entry_val;
            let target = entry_val - (risk * risk_reward_ratio);
            
            (Price::from_f64(stop), Price::from_f64(target))
        },
        _ => {
            // Default: 2% stop, 4% target
            let stop = entry_val * 0.98;
            let target = entry_val * 1.04;
            (Price::from_f64(stop), Price::from_f64(target))
        }
    }
}

/// Estimate market impact
fn estimate_market_impact(size: f64, daily_volume: f64) -> f64 {
    // Square-root model: Impact = γ * sqrt(V/ADV)
    let gamma = 0.1;  // Impact coefficient
    let participation = size / daily_volume;
    gamma * participation.sqrt()
}

/// Convert signal action
fn convert_signal_action(basic: crate::unified_types::SignalAction) -> SignalAction {
    match basic {
        crate::unified_types::SignalAction::Buy => SignalAction::Buy,
        crate::unified_types::SignalAction::Sell => SignalAction::Sell,
        crate::unified_types::SignalAction::Hold => SignalAction::Hold,
        crate::unified_types::SignalAction::ClosePosition => SignalAction::ClosePosition,
        crate::unified_types::SignalAction::ReducePosition => SignalAction::ReducePosition,
        crate::unified_types::SignalAction::IncreasePosition => SignalAction::IncreasePosition,
    }
}

/// Convert TA indicators to HashMap
fn convert_ta_indicators(indicators: &[f64]) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    
    // Map array indices to named indicators
    let names = ["rsi", "macd", "bb_upper", "bb_lower", "ema_9", "ema_21", "volume_ratio"];
    
    for (i, &value) in indicators.iter().enumerate() {
        if i < names.len() {
            map.insert(names[i].to_string(), value);
        } else {
            map.insert(format!("indicator_{}", i), value);
        }
    }
    
    map
}

/// Detect market regime
fn detect_market_regime(market_data: &CompleteMarketData) -> MarketRegime {
    let volatility = market_data.volatility_24h.to_f64();
    let returns = market_data.returns_24h.to_f64();
    
    if volatility > 0.5 {
        MarketRegime::Volatile
    } else if volatility < 0.1 {
        MarketRegime::Quiet
    } else if returns > 0.05 {
        MarketRegime::Trending
    } else if returns < -0.05 {
        MarketRegime::Breakdown
    } else {
        MarketRegime::RangeB


    }
}

/// Create default sentiment data
/// TODO: Add docs
pub fn create_default_sentiment() -> SentimentData {
    SentimentData {
        timestamp: Utc::now(),
        twitter_sentiment: 0.0,
        reddit_sentiment: 0.0,
        news_sentiment: 0.0,
        fear_greed_index: 50.0,
        put_call_ratio: 1.0,
        vix: 20.0,
        long_short_ratio: 1.0,
        funding_rate: 0.0,
        open_interest: 0.0,
        overall_sentiment: 0.0,
        sentiment_momentum: 0.0,
        sentiment_divergence: 0.0,
    }
}

/// Create enhanced order book from basic data
/// TODO: Add docs
pub fn create_enhanced_order_book(
    bids: Vec<(Price, Quantity)>,
    asks: Vec<(Price, Quantity)>,
    timestamp: u64,
) -> EnhancedOrderBook {
    use crate::trading_types_complete::{OrderLevel, Trade, OrderFlow, TradeSide};
    
    let bid_levels: Vec<OrderLevel> = bids.into_iter()
        .map(|(price, qty)| OrderLevel {
            price,
            quantity: qty,
            order_count: 1,  // Default
        })
        .collect();
        
    let ask_levels: Vec<OrderLevel> = asks.into_iter()
        .map(|(price, qty)| OrderLevel {
            price,
            quantity: qty,
            order_count: 1,  // Default
        })
        .collect();
    
    // Calculate imbalance
    let bid_vol: f64 = bid_levels.iter().map(|l| l.quantity.to_f64()).sum();
    let ask_vol: f64 = ask_levels.iter().map(|l| l.quantity.to_f64()).sum();
    let imbalance = if bid_vol + ask_vol > 0.0 {
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    } else {
        0.0
    };
    
    // Calculate microprice
    let micro_price = if !bid_levels.is_empty() && !ask_levels.is_empty() {
        let best_bid = bid_levels[0].price.to_f64();
        let best_ask = ask_levels[0].price.to_f64();
        let bid_size = bid_levels[0].quantity.to_f64();
        let ask_size = ask_levels[0].quantity.to_f64();
        
        let weighted = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size);
        Price::from_f64(weighted)
    } else {
        Price::ZERO
    };
    
    EnhancedOrderBook {
        bids: bid_levels,
        asks: ask_levels,
        timestamp,
        exchange: "unified".to_string(),
        trade_flow: Vec::new(),
        order_flow: OrderFlow {
            buy_volume: bid_vol,
            sell_volume: ask_vol,
            buy_trades: 0,
            sell_trades: 0,
            large_buy_volume: 0.0,
            large_sell_volume: 0.0,
            toxicity: 0.0,
        },
        book_imbalance: imbalance,
        micro_price,
    }
}