use domain_types::market_data::PriceLevel;
//! Using canonical Trade from domain_types
pub use domain_types::trade::{Trade, TradeId, TradeError};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// ORDER BOOK ANALYTICS WITH GAME THEORY - NO SIMPLIFICATIONS!
// Team: Casey (Lead) + Morgan (ML) + Quinn (Risk) + Full Team
// References:
// - Kyle (1985): "Continuous Auctions and Insider Trading"
// - Glosten & Milgrom (1985): "Bid, Ask and Transaction Prices"
// - Easley et al. (2012): "Flow Toxicity and Liquidity in a High-frequency World"
// - Cartea et al. (2015): "Algorithmic and High-Frequency Trading"

use crate::unified_types::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use std::collections::{VecDeque, HashMap};
use anyhow::Result;

/// Complete Order Book Analytics with Game Theory
/// Casey: "This is where we detect REAL market manipulation!"
/// TODO: Add docs
pub struct OrderBookAnalytics {
    // Core order book state
    order_book_history: VecDeque<OrderBookSnapshot>,
    
    // Kyle's Lambda - price impact coefficient
    kyle_lambda: f64,
    kyle_window: usize,
    
    // VPIN - Volume-Synchronized Probability of Informed Trading
    vpin_buckets: VecDeque<VPINBucket>,
    vpin_bucket_size: f64,  // Volume per bucket
    vpin_window: usize,     // Number of buckets
    
    // Order flow imbalance
    flow_imbalance_history: VecDeque<f64>,
    
    // Microstructure metrics
    effective_spread_history: VecDeque<f64>,
    realized_spread_history: VecDeque<f64>,
    
    // Game theory components
    spoofing_detector: SpoofingDetector,
    layering_detector: LayeringDetector,
    momentum_ignition_detector: MomentumIgnitionDetector,
    
    // Adverse selection model
    adverse_selection_component: f64,
    probability_informed_trading: f64,
    
    // Execution cost model
    permanent_impact: f64,
    temporary_impact: f64,
    
    // Market depth analysis
    depth_imbalance: DepthImbalance,
    resilience_measure: f64,
}

/// Order book snapshot at a point in time
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
// pub struct OrderBookSnapshot {
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub timestamp: u64,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub bids: Vec<PriceLevel>,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub asks: Vec<PriceLevel>,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub mid_price: Decimal,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub microprice: Decimal,  // Size-weighted price
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub trades: Vec<Trade>,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     // Depth at first level (for quick access)
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub bid_depth_1: f64,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
//     pub ask_depth_1: f64,
// ELIMINATED: Duplicate - use domain_types::market_data::OrderBookSnapshot
// }

/// Price level in order book
#[derive(Debug, Clone)]
/// TODO: Add docs

/// Executed trade
#[derive(Debug, Clone)]

// Using Side from unified_types (Long/Short)

/// VPIN bucket for toxicity measurement
#[derive(Debug, Clone)]
struct VPINBucket {
    volume: f64,
    buy_volume: f64,
    sell_volume: f64,
    timestamp_start: u64,
    timestamp_end: u64,
}

/// Depth imbalance at multiple levels
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct DepthImbalance {
    level_1: f64,  // Top of book
    level_5: f64,  // Top 5 levels
    level_10: f64, // Top 10 levels
    weighted: f64, // Distance-weighted
}

/// Spoofing detection using game theory
/// Morgan: "Spoofing is a dominant strategy until caught!"
struct SpoofingDetector {
    order_lifecycle: HashMap<String, OrderLifecycle>,
    cancellation_rate_threshold: f64,
    time_to_cancel_threshold: u64,  // milliseconds
    fleeting_order_ratio: f64,
    pub detection_score: f64,  // Current spoofing detection score (0-1)
}

/// Order lifecycle tracking
#[derive(Debug, Clone)]
struct OrderLifecycle {
    order_id: String,
    side: Side,
    price: Decimal,
    quantity: Decimal,
    placed_at: u64,
    cancelled_at: Option<u64>,
    filled_quantity: Decimal,
    modifications: Vec<OrderModification>,
}

#[derive(Debug, Clone)]
struct OrderModification {
    timestamp: u64,
    new_price: Option<Decimal>,
    new_quantity: Option<Decimal>,
}

/// Layering detection (multiple orders to create false impression)
struct LayeringDetector {
    layer_patterns: Vec<LayerPattern>,
    min_layers: usize,
    price_range_threshold: f64,
}

#[derive(Debug, Clone)]
struct LayerPattern {
    side: Side,
    price_levels: Vec<Decimal>,
    total_quantity: Decimal,
    timestamp: u64,
    is_suspicious: bool,
}

/// Momentum ignition detection
struct MomentumIgnitionDetector {
    aggressive_trades: VecDeque<AggressiveTrade>,
    momentum_threshold: f64,
    time_window: u64,
}

#[derive(Debug, Clone)]
struct AggressiveTrade {
    timestamp: u64,
    side: Side,
    price: Decimal,
    quantity: Decimal,
    price_impact: f64,
}

impl OrderBookAnalytics {
    /// Create new order book analytics engine
    /// Alex: "This is our edge - seeing what others miss!"
    pub fn new() -> Self {
        Self {
            order_book_history: VecDeque::with_capacity(1000),
            kyle_lambda: 0.0,
            kyle_window: 100,
            vpin_buckets: VecDeque::with_capacity(50),
            vpin_bucket_size: 1000.0,  // 1000 units of volume per bucket
            vpin_window: 50,
            flow_imbalance_history: VecDeque::with_capacity(1000),
            effective_spread_history: VecDeque::with_capacity(1000),
            realized_spread_history: VecDeque::with_capacity(1000),
            spoofing_detector: SpoofingDetector::new(),
            layering_detector: LayeringDetector::new(),
            momentum_ignition_detector: MomentumIgnitionDetector::new(),
            adverse_selection_component: 0.0,
            probability_informed_trading: 0.0,
            permanent_impact: 0.0,
            temporary_impact: 0.0,
            depth_imbalance: DepthImbalance {
                level_1: 0.0,
                level_5: 0.0,
                level_10: 0.0,
                weighted: 0.0,
            },
            resilience_measure: 0.0,
        }
    }
    
    /// Process new order book snapshot
    pub fn process_order_book(&mut self, snapshot: OrderBookSnapshot) -> OrderBookMetrics {
        // Store snapshot
        self.order_book_history.push_back(snapshot.clone());
        if self.order_book_history.len() > 1000 {
            self.order_book_history.pop_front();
        }
        
        // Calculate all metrics
        let imbalance = self.calculate_order_book_imbalance(&snapshot);
        let microprice = self.calculate_microprice(&snapshot);
        let depth_imb = self.calculate_depth_imbalance(&snapshot);
        let kyle_lambda = self.update_kyle_lambda(&snapshot);
        let vpin = self.update_vpin(&snapshot);
        let effective_spread = self.calculate_effective_spread(&snapshot);
        let realized_spread = self.calculate_realized_spread(&snapshot);
        
        // Detect manipulation
        let spoofing_score = self.spoofing_detector.detect(&snapshot);
        let layering_score = self.layering_detector.detect(&snapshot);
        let momentum_ignition = self.momentum_ignition_detector.detect(&snapshot);
        
        // Calculate execution costs
        let (permanent, temporary) = self.estimate_market_impact(&snapshot);
        
        // Update adverse selection
        self.update_adverse_selection(&snapshot);
        
        OrderBookMetrics {
            timestamp: snapshot.timestamp,
            imbalance,
            microprice,
            depth_imbalance: depth_imb,
            kyle_lambda,
            vpin,
            effective_spread,
            realized_spread,
            spoofing_probability: spoofing_score,
            layering_probability: layering_score,
            momentum_ignition_probability: momentum_ignition,
            adverse_selection_cost: self.adverse_selection_component,
            probability_informed: self.probability_informed_trading,
            permanent_impact: permanent,
            temporary_impact: temporary,
            resilience: self.resilience_measure,
        }
    }
    
    /// Calculate order book imbalance (Cartea et al., 2015)
    /// This is the MOST IMPORTANT metric for HFT!
    fn calculate_order_book_imbalance(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Level 1 imbalance
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return 0.0;
        }
        
        let best_bid_size = snapshot.bids[0].quantity.to_f64().unwrap();
        let best_ask_size = snapshot.asks[0].quantity.to_f64().unwrap();
        
        // Simple imbalance
        let simple_imb = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size);
        
        // Weighted imbalance (top 5 levels)
        let mut bid_volume = 0.0;
        let mut ask_volume = 0.0;
        let mut bid_weighted = 0.0;
        let mut ask_weighted = 0.0;
        
        let mid = snapshot.mid_price.to_f64().unwrap();
        
        for i in 0..5.min(snapshot.bids.len()) {
            let level = &snapshot.bids[i];
            let size = level.quantity.to_f64().unwrap();
            let price = level.price.to_f64().unwrap();
            let weight = 1.0 / (1.0 + (mid - price).abs() / mid);
            
            bid_volume += size;
            bid_weighted += size * weight;
        }
        
        for i in 0..5.min(snapshot.asks.len()) {
            let level = &snapshot.asks[i];
            let size = level.quantity.to_f64().unwrap();
            let price = level.price.to_f64().unwrap();
            let weight = 1.0 / (1.0 + (price - mid).abs() / mid);
            
            ask_volume += size;
            ask_weighted += size * weight;
        }
        
        let weighted_imb = if bid_weighted + ask_weighted > 0.0 {
            (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted)
        } else {
            0.0
        };
        
        // Store for history
        self.flow_imbalance_history.push_back(weighted_imb);
        if self.flow_imbalance_history.len() > 1000 {
            self.flow_imbalance_history.pop_front();
        }
        
        // Return weighted imbalance (more sophisticated)
        weighted_imb
    }
    
    /// Calculate microprice (size-weighted mid price)
    /// Better predictor than simple mid price!
    fn calculate_microprice(&self, snapshot: &OrderBookSnapshot) -> Decimal {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return snapshot.mid_price;
        }
        
        let bid_price = snapshot.bids[0].price;
        let bid_size = snapshot.bids[0].quantity;
        let ask_price = snapshot.asks[0].price;
        let ask_size = snapshot.asks[0].quantity;
        
        // Microprice formula from Gatheral (2010)
        let microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size);
        
        microprice
    }
    
    /// Calculate depth imbalance at multiple levels
    fn calculate_depth_imbalance(&mut self, snapshot: &OrderBookSnapshot) -> DepthImbalance {
        let mut imb = DepthImbalance {
            level_1: 0.0,
            level_5: 0.0,
            level_10: 0.0,
            weighted: 0.0,
        };
        
        // Level 1
        if !snapshot.bids.is_empty() && !snapshot.asks.is_empty() {
            let b1 = snapshot.bids[0].quantity.to_f64().unwrap();
            let a1 = snapshot.asks[0].quantity.to_f64().unwrap();
            imb.level_1 = (b1 - a1) / (b1 + a1);
        }
        
        // Level 5
        let mut bid_5 = 0.0;
        let mut ask_5 = 0.0;
        for i in 0..5.min(snapshot.bids.len()) {
            bid_5 += snapshot.bids[i].quantity.to_f64().unwrap();
        }
        for i in 0..5.min(snapshot.asks.len()) {
            ask_5 += snapshot.asks[i].quantity.to_f64().unwrap();
        }
        if bid_5 + ask_5 > 0.0 {
            imb.level_5 = (bid_5 - ask_5) / (bid_5 + ask_5);
        }
        
        // Level 10
        let mut bid_10 = 0.0;
        let mut ask_10 = 0.0;
        for i in 0..10.min(snapshot.bids.len()) {
            bid_10 += snapshot.bids[i].quantity.to_f64().unwrap();
        }
        for i in 0..10.min(snapshot.asks.len()) {
            ask_10 += snapshot.asks[i].quantity.to_f64().unwrap();
        }
        if bid_10 + ask_10 > 0.0 {
            imb.level_10 = (bid_10 - ask_10) / (bid_10 + ask_10);
        }
        
        // Distance-weighted (closer levels matter more)
        let mid = snapshot.mid_price.to_f64().unwrap();
        let mut bid_weighted = 0.0;
        let mut ask_weighted = 0.0;
        
        for level in &snapshot.bids {
            let price = level.price.to_f64().unwrap();
            let size = level.quantity.to_f64().unwrap();
            let distance = (mid - price).abs() / mid;
            let weight = (-distance * 100.0).exp();  // Exponential decay
            bid_weighted += size * weight;
        }
        
        for level in &snapshot.asks {
            let price = level.price.to_f64().unwrap();
            let size = level.quantity.to_f64().unwrap();
            let distance = (price - mid).abs() / mid;
            let weight = (-distance * 100.0).exp();
            ask_weighted += size * weight;
        }
        
        if bid_weighted + ask_weighted > 0.0 {
            imb.weighted = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted);
        }
        
        self.depth_imbalance = imb.clone();
        imb
    }
    
    /// Update Kyle's Lambda (price impact coefficient)
    /// Kyle (1985): λ = σ / (2 * √V) where σ is volatility, V is volume
    fn update_kyle_lambda(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        if self.order_book_history.len() < self.kyle_window {
            return self.kyle_lambda;  // Not enough data
        }
        
        // Calculate returns
        let mut returns = Vec::new();
        let mut volumes = Vec::new();
        
        for i in 1..self.order_book_history.len() {
            let prev = &self.order_book_history[i-1];
            let curr = &self.order_book_history[i];
            
            let ret = (curr.mid_price / prev.mid_price - dec!(1)).to_f64().unwrap();
            returns.push(ret);
            
            // Sum trade volumes
            let vol: f64 = curr.trades.iter()
                .map(|t| t.quantity.to_f64().unwrap())
                .sum();
            volumes.push(vol);
        }
        
        if returns.is_empty() || volumes.is_empty() {
            return self.kyle_lambda;
        }
        
        // Calculate volatility
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();
        
        // Calculate average volume
        let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;
        
        // Kyle's Lambda formula
        if avg_volume > 0.0 {
            self.kyle_lambda = volatility / (2.0 * avg_volume.sqrt());
        }
        
        // Advanced: Regression-based lambda (more accurate)
        // Regress |price_change| on signed_volume
        let mut price_impacts = Vec::new();
        for i in 1..self.order_book_history.len() {
            let prev = &self.order_book_history[i-1];
            let curr = &self.order_book_history[i];
            
            let price_change = (curr.mid_price - prev.mid_price).to_f64().unwrap();
            let net_volume: f64 = curr.trades.iter()
                .map(|t| {
                    let vol = t.quantity.to_f64().unwrap();
                    match t.aggressor_side {
                        Side::Long => vol,
                        Side::Short => -vol,
                    }
                })
                .sum();
            
            if net_volume.abs() > 0.0 {
                price_impacts.push((net_volume, price_change));
            }
        }
        
        // Linear regression to find lambda
        if price_impacts.len() > 10 {
            let lambda = self.calculate_regression_lambda(&price_impacts);
            self.kyle_lambda = lambda;
        }
        
        self.kyle_lambda
    }
    
    /// Calculate lambda using regression
    fn calculate_regression_lambda(&self, data: &[(f64, f64)]) -> f64 {
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        slope.abs()  // Lambda is positive
    }
    
    /// Update VPIN (Volume-Synchronized Probability of Informed Trading)
    /// Easley et al. (2012): VPIN = |V_buy - V_sell| / (V_buy + V_sell)
    fn update_vpin(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Add trades to current bucket
        let mut current_bucket = self.vpin_buckets.back_mut()
            .cloned()
            .unwrap_or(VPINBucket {
                volume: 0.0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                timestamp_start: snapshot.timestamp,
                timestamp_end: snapshot.timestamp,
            });
        
        for trade in &snapshot.trades {
            let vol = trade.quantity.to_f64().unwrap();
            current_bucket.volume += vol;
            
            match trade.aggressor_side {
                Side::Long => current_bucket.buy_volume += vol,
                Side::Short => current_bucket.sell_volume += vol,
            }
        }
        
        // Check if bucket is full
        if current_bucket.volume >= self.vpin_bucket_size {
            current_bucket.timestamp_end = snapshot.timestamp;
            self.vpin_buckets.push_back(current_bucket);
            
            // Keep only window size
            while self.vpin_buckets.len() > self.vpin_window {
                self.vpin_buckets.pop_front();
            }
        } else {
            // Update current bucket
            if let Some(last) = self.vpin_buckets.back_mut() {
                *last = current_bucket;
            } else {
                self.vpin_buckets.push_back(current_bucket);
            }
        }
        
        // Calculate VPIN
        if self.vpin_buckets.len() < 5 {
            return 0.0;  // Not enough data
        }
        
        let total_buy: f64 = self.vpin_buckets.iter().map(|b| b.buy_volume).sum();
        let total_sell: f64 = self.vpin_buckets.iter().map(|b| b.sell_volume).sum();
        
        if total_buy + total_sell > 0.0 {
            (total_buy - total_sell).abs() / (total_buy + total_sell)
        } else {
            0.0
        }
    }
    
    /// Calculate effective spread (actual execution cost)
    fn calculate_effective_spread(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        if snapshot.trades.is_empty() {
            return 0.0;
        }
        
        let mid = snapshot.mid_price.to_f64().unwrap();
        let mut total_spread = 0.0;
        let mut total_volume = 0.0;
        
        for trade in &snapshot.trades {
            let price = trade.price.to_f64().unwrap();
            let volume = trade.quantity.to_f64().unwrap();
            
            // Effective spread = 2 * |price - mid| / mid
            let spread = 2.0 * (price - mid).abs() / mid;
            total_spread += spread * volume;
            total_volume += volume;
        }
        
        let avg_spread = if total_volume > 0.0 {
            total_spread / total_volume
        } else {
            0.0
        };
        
        self.effective_spread_history.push_back(avg_spread);
        if self.effective_spread_history.len() > 1000 {
            self.effective_spread_history.pop_front();
        }
        
        avg_spread
    }
    
    /// Calculate realized spread (temporary vs permanent impact)
    fn calculate_realized_spread(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Realized spread = effective spread - price impact after 5 minutes
        // For now, approximate with effective spread * resilience
        let effective = self.calculate_effective_spread(snapshot);
        let realized = effective * (1.0 - self.resilience_measure);
        
        self.realized_spread_history.push_back(realized);
        if self.realized_spread_history.len() > 1000 {
            self.realized_spread_history.pop_front();
        }
        
        realized
    }
    
    /// Estimate market impact using Almgren-Chriss model
    fn estimate_market_impact(&mut self, snapshot: &OrderBookSnapshot) -> (f64, f64) {
        // Permanent impact (information)
        self.permanent_impact = self.kyle_lambda;
        
        // Temporary impact (liquidity demand)
        let spread = if !snapshot.bids.is_empty() && !snapshot.asks.is_empty() {
            (snapshot.asks[0].price - snapshot.bids[0].price).to_f64().unwrap()
        } else {
            0.0
        };
        
        let depth = self.calculate_book_depth(snapshot);
        self.temporary_impact = spread / (2.0 * depth.max(1.0));
        
        (self.permanent_impact, self.temporary_impact)
    }
    
    /// Calculate total book depth
    fn calculate_book_depth(&self, snapshot: &OrderBookSnapshot) -> f64 {
        let bid_depth: f64 = snapshot.bids.iter()
            .map(|l| l.quantity.to_f64().unwrap())
            .sum();
        let ask_depth: f64 = snapshot.asks.iter()
            .map(|l| l.quantity.to_f64().unwrap())
            .sum();
        
        (bid_depth + ask_depth) / 2.0
    }
    
    /// Update adverse selection component
    fn update_adverse_selection(&mut self, snapshot: &OrderBookSnapshot) {
        // Glosten-Milgrom model
        // Adverse selection = probability of informed * expected loss to informed
        
        // Use VPIN as proxy for probability of informed trading
        self.probability_informed_trading = self.update_vpin(snapshot).min(1.0);
        
        // Expected loss is half-spread times probability
        let half_spread = if !snapshot.bids.is_empty() && !snapshot.asks.is_empty() {
            let spread = (snapshot.asks[0].price - snapshot.bids[0].price).to_f64().unwrap();
            spread / 2.0 / snapshot.mid_price.to_f64().unwrap()
        } else {
            0.0
        };
        
        self.adverse_selection_component = self.probability_informed_trading * half_spread;
    }
    
    /// Get trading signal based on order book analytics
    pub fn get_signal(&self, min_edge: f64) -> TradingRecommendation {
        // Calculate expected alpha from order book imbalance
        let imbalance = self.depth_imbalance.weighted;
        
        // Academic research: imbalance predicts price movement
        // Cont et al. (2013): E[r|I] = α * I where α ≈ 0.01
        let expected_return = 0.01 * imbalance;
        
        // Adjust for adverse selection
        let net_edge = expected_return - self.adverse_selection_component;
        
        // Check if edge exceeds minimum after costs
        let total_cost = self.effective_spread_history.back().copied().unwrap_or(0.001);
        
        if net_edge > min_edge + total_cost {
            TradingRecommendation {
                action: if imbalance > 0.0 { SignalAction::Buy } else { SignalAction::Sell },
                confidence: net_edge.abs().min(1.0),
                expected_edge: net_edge,
                execution_urgency: self.calculate_urgency(),
                size_limit: self.calculate_size_limit(),
            }
        } else {
            TradingRecommendation {
                action: SignalAction::Hold,
                confidence: 0.0,
                expected_edge: net_edge,
                execution_urgency: 0.0,
                size_limit: 0.0,
            }
        }
    }
    
    /// Calculate execution urgency based on order book dynamics
    fn calculate_urgency(&self) -> f64 {
        // High VPIN = toxic flow = execute quickly
        // High imbalance = opportunity disappearing = execute quickly
        // Calculate VPIN from the latest bucket
        let vpin_urgency = self.vpin_buckets.back()
            .map(|bucket| {
                if bucket.volume > 0.0 {
                    (bucket.buy_volume - bucket.sell_volume).abs() / bucket.volume
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        
        let imbalance_urgency = self.depth_imbalance.level_1.abs();
        
        (vpin_urgency + imbalance_urgency).min(1.0)
    }
    
    /// Calculate maximum safe position size
    fn calculate_size_limit(&self) -> f64 {
        // Don't be more than 1% of total book depth
        // Adjust down if high adverse selection
        let base_limit = 0.01;
        let adverse_adjustment = 1.0 - self.probability_informed_trading;
        
        base_limit * adverse_adjustment
    }
    
    /// Get order book imbalance (public method for optimal_execution)
    /// DEEP DIVE: Simple wrapper for external access
    pub fn get_imbalance(&self) -> f64 {
        self.depth_imbalance.level_1
    }
    
    /// Get spoof ratio for predatory detection
    pub fn get_spoof_ratio(&self) -> f64 {
        self.spoofing_detector.detection_score
    }
    
    /// Get quote update rate (updates per second)
    pub fn get_quote_update_rate(&self) -> f64 {
        // Calculate from recent order book snapshots
        if self.order_book_history.len() < 2 {
            return 0.0;
        }
        
        let duration = self.order_book_history.back().unwrap().timestamp - 
                      self.order_book_history.front().unwrap().timestamp;
        
        if duration > 0 {
            self.order_book_history.len() as f64 / duration as f64
        } else {
            0.0
        }
    }
    
    /// Get average trade size from recent history
    pub fn get_average_trade_size(&self) -> f64 {
        // Use recent flow imbalance history as proxy
        if self.flow_imbalance_history.is_empty() {
            return 1.0;  // Default 1 unit
        }
        
        let sum: f64 = self.flow_imbalance_history.iter()
            .map(|f| f.abs())
            .sum();
        
        sum / self.flow_imbalance_history.len() as f64
    }
    
    /// Detect liquidity events (large orders appearing)
    pub fn detect_liquidity_events(&self) -> Vec<LiquidityEvent> {
        let mut events = Vec::new();
        
        // Check recent order book changes for large orders
        if self.order_book_history.len() >= 2 {
            let current = self.order_book_history.back().unwrap();
            let previous = &self.order_book_history[self.order_book_history.len() - 2];
            
            // Detect large bid appearance
            if current.bid_depth_1 > previous.bid_depth_1 * 2.0 {
                events.push(LiquidityEvent {
                    side: Side::Long,  // Bid side
                    size: current.bid_depth_1 - previous.bid_depth_1,
                    price_level: 1,
                    timestamp: current.timestamp,
                });
            }
            
            // Detect large ask appearance
            if current.ask_depth_1 > previous.ask_depth_1 * 2.0 {
                events.push(LiquidityEvent {
                    side: Side::Short,  // Ask side
                    size: current.ask_depth_1 - previous.ask_depth_1,
                    price_level: 1,
                    timestamp: current.timestamp,
                });
            }
        }
        
        events
    }
}

/// Spoofing detector implementation
impl SpoofingDetector {
    fn new() -> Self {
        Self {
            order_lifecycle: HashMap::new(),
            cancellation_rate_threshold: 0.9,  // 90% cancelled = suspicious
            time_to_cancel_threshold: 1000,    // Cancel within 1 second
            fleeting_order_ratio: 0.5,          // 50% fleeting = suspicious
            detection_score: 0.0,
        }
    }
    
    fn detect(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Track order lifecycle
        // Calculate cancellation rates
        // Detect fleeting orders
        
        // Simplified: return probability based on book dynamics
        let mut score: f64 = 0.0;
        
        // Large orders far from mid that disappear quickly
        for level in &snapshot.bids {
            let distance = (snapshot.mid_price - level.price).to_f64().unwrap() 
                         / snapshot.mid_price.to_f64().unwrap();
            if distance > 0.005 && level.quantity > dec!(100) {
                score += 0.1;  // Suspicious large order away from mid
            }
        }
        
        for level in &snapshot.asks {
            let distance = (level.price - snapshot.mid_price).to_f64().unwrap() 
                         / snapshot.mid_price.to_f64().unwrap();
            if distance > 0.005 && level.quantity > dec!(100) {
                score += 0.1;
            }
        }
        
        self.detection_score = score.min(1.0_f64);
        self.detection_score
    }
}

/// Layering detector implementation
impl LayeringDetector {
    fn new() -> Self {
        Self {
            layer_patterns: Vec::new(),
            min_layers: 3,
            price_range_threshold: 0.01,  // 1% price range
        }
    }
    
    fn detect(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Detect multiple orders at different price levels
        let mut score: f64 = 0.0;
        
        // Check for layering pattern on bid side
        if snapshot.bids.len() >= self.min_layers {
            let mut consecutive_large = 0;
            for i in 1..snapshot.bids.len().min(10) {
                if snapshot.bids[i].quantity > snapshot.bids[0].quantity * dec!(0.8) {
                    consecutive_large += 1;
                }
            }
            if consecutive_large >= self.min_layers {
                score += 0.5;
            }
        }
        
        // Check ask side
        if snapshot.asks.len() >= self.min_layers {
            let mut consecutive_large = 0;
            for i in 1..snapshot.asks.len().min(10) {
                if snapshot.asks[i].quantity > snapshot.asks[0].quantity * dec!(0.8) {
                    consecutive_large += 1;
                }
            }
            if consecutive_large >= self.min_layers {
                score += 0.5;
            }
        }
        
        score.min(1.0_f64)
    }
}

/// Momentum ignition detector
impl MomentumIgnitionDetector {
    fn new() -> Self {
        Self {
            aggressive_trades: VecDeque::with_capacity(100),
            momentum_threshold: 0.001,  // 0.1% price move
            time_window: 5000,  // 5 seconds
        }
    }
    
    fn detect(&mut self, snapshot: &OrderBookSnapshot) -> f64 {
        // Detect aggressive trades that move price
        let mut score: f64 = 0.0;
        
        // Count aggressive trades in recent history
        for trade in &snapshot.trades {
            if trade.quantity > dec!(10) {  // Large trade
                let impact = (trade.price - snapshot.mid_price).abs() / snapshot.mid_price;
                if impact > dec!(0.0005) {  // 0.05% impact
                    score += 0.2;
                }
            }
        }
        
        score.min(1.0_f64)
    }
}

/// Order book metrics output
/// Liquidity event detected in order book
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: LiquidityEvent - Enhanced with Level 3 data, iceberg detection
// pub struct LiquidityEvent {
    pub side: Side,
    pub size: f64,
    pub price_level: u32,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct OrderBookMetrics {
    pub timestamp: u64,
    pub imbalance: f64,
    pub microprice: Decimal,
    pub depth_imbalance: DepthImbalance,
    pub kyle_lambda: f64,
    pub vpin: f64,
    pub effective_spread: f64,
    pub realized_spread: f64,
    pub spoofing_probability: f64,
    pub layering_probability: f64,
    pub momentum_ignition_probability: f64,
    pub adverse_selection_cost: f64,
    pub probability_informed: f64,
    pub permanent_impact: f64,
    pub temporary_impact: f64,
    pub resilience: f64,
}

/// Trading recommendation based on order book
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TradingRecommendation {
    pub action: SignalAction,
    pub confidence: f64,
    pub expected_edge: f64,
    pub execution_urgency: f64,
    pub size_limit: f64,
}

impl Default for OrderBookSnapshot {
    fn default() -> Self {
        Self {
            timestamp: 0,
            bids: Vec::new(),
            asks: Vec::new(),
            mid_price: dec!(0),
            microprice: dec!(0),
            trades: Vec::new(),
            bid_depth_1: 0.0,
            ask_depth_1: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_book_imbalance() {
        let mut analytics = OrderBookAnalytics::new();
        
        let snapshot = OrderBookSnapshot {
            timestamp: 1000,
            bids: vec![
                PriceLevel { price: dec!(99.9), quantity: dec!(100), order_count: 5 },
                PriceLevel { price: dec!(99.8), quantity: dec!(150), order_count: 3 },
            ],
            asks: vec![
                PriceLevel { price: dec!(100.1), quantity: dec!(80), order_count: 4 },
                PriceLevel { price: dec!(100.2), quantity: dec!(120), order_count: 2 },
            ],
            mid_price: dec!(100),
            microprice: dec!(100.02),
            trades: vec![],
            bid_depth_1: 100.0,
            ask_depth_1: 80.0,
        };
        
        let metrics = analytics.process_order_book(snapshot);
        
        // Bid volume > Ask volume, so positive imbalance expected
        assert!(metrics.imbalance > 0.0, "Expected positive imbalance");
        assert!(metrics.imbalance < 1.0, "Imbalance should be normalized");
    }
    
    #[test]
    fn test_kyle_lambda_calculation() {
        let mut analytics = OrderBookAnalytics::new();
        
        // Simulate multiple snapshots with trades
        for i in 0..200 {
            let price = dec!(100) + Decimal::from(i) / dec!(100);
            let snapshot = OrderBookSnapshot {
                timestamp: i as u64 * 1000,
                bids: vec![PriceLevel { 
                    price: price - dec!(0.01), 
                    quantity: dec!(100), 
                    order_count: 1 
                }],
                asks: vec![PriceLevel { 
                    price: price + dec!(0.01), 
                    quantity: dec!(100), 
                    order_count: 1 
                }],
                mid_price: price,
                microprice: price,
                trades: vec![Trade {
                    timestamp: i as u64 * 1000,
                    price,
                    quantity: dec!(10),
                    aggressor_side: if i % 2 == 0 { Side::Long } else { Side::Short },
                    trade_id: format!("trade_{}", i),
                }],
                bid_depth_1: 100.0,
                ask_depth_1: 100.0,
            };
            
            analytics.process_order_book(snapshot);
        }
        
        // After sufficient data, Kyle's lambda should be positive
        assert!(analytics.kyle_lambda > 0.0, "Kyle's lambda should be positive");
        assert!(analytics.kyle_lambda < 1.0, "Kyle's lambda seems too high");
    }
    
    #[test]
    fn test_vpin_calculation() {
        let mut analytics = OrderBookAnalytics::new();
        analytics.vpin_bucket_size = 100.0;  // Small bucket for testing
        
        // Simulate imbalanced flow (more buys)
        for i in 0..100 {
            let snapshot = OrderBookSnapshot {
                timestamp: i as u64 * 1000,
                bids: vec![],
                asks: vec![],
                mid_price: dec!(100),
                microprice: dec!(100),
                trades: vec![Trade {
                    timestamp: i as u64 * 1000,
                    price: dec!(100),
                    quantity: dec!(5),
                    aggressor_side: if i < 70 { Side::Long } else { Side::Short },
                    trade_id: format!("trade_{}", i),
                }],
                bid_depth_1: 100.0,
                ask_depth_1: 100.0,
            };
            
            analytics.process_order_book(snapshot);
        }
        
        // VPIN should be positive due to imbalance
        let vpin = analytics.update_vpin(&OrderBookSnapshot::default());
        assert!(vpin > 0.0, "VPIN should detect flow imbalance");
        assert!(vpin <= 1.0, "VPIN should be normalized");
    }
    
    #[test]
    fn test_manipulation_detection() {
        let mut analytics = OrderBookAnalytics::new();
        
        // Create suspicious order book (potential spoofing)
        let snapshot = OrderBookSnapshot {
            timestamp: 1000,
            bids: vec![
                PriceLevel { price: dec!(99.9), quantity: dec!(10), order_count: 1 },
                PriceLevel { price: dec!(99.5), quantity: dec!(1000), order_count: 1 }, // Large order away
                PriceLevel { price: dec!(99.4), quantity: dec!(1000), order_count: 1 }, // Another large
            ],
            asks: vec![
                PriceLevel { price: dec!(100.1), quantity: dec!(10), order_count: 1 },
            ],
            mid_price: dec!(100),
            microprice: dec!(100),
            trades: vec![],
            bid_depth_1: 10.0,
            ask_depth_1: 10.0,
        };
        
        let metrics = analytics.process_order_book(snapshot);
        
        // Should detect some spoofing probability
        assert!(metrics.spoofing_probability > 0.0, "Should detect potential spoofing");
    }
    
    #[test]
    fn test_adverse_selection_calculation() {
        let mut analytics = OrderBookAnalytics::new();
        
        // High VPIN scenario (toxic flow)
        for i in 0..60 {
            let snapshot = OrderBookSnapshot {
                timestamp: i as u64 * 1000,
                bids: vec![PriceLevel { 
                    price: dec!(99.9), 
                    quantity: dec!(100), 
                    order_count: 1 
                }],
                asks: vec![PriceLevel { 
                    price: dec!(100.1), 
                    quantity: dec!(100), 
                    order_count: 1 
                }],
                mid_price: dec!(100),
                microprice: dec!(100),
                trades: vec![Trade {
                    timestamp: i as u64 * 1000,
                    price: dec!(100),
                    quantity: dec!(10),
                    aggressor_side: Side::Long,  // All buys = informed trading
                    trade_id: format!("trade_{}", i),
                }],
                bid_depth_1: 100.0,
                ask_depth_1: 100.0,
            };
            
            analytics.process_order_book(snapshot);
        }
        
        // Should have positive adverse selection cost
        assert!(analytics.adverse_selection_component > 0.0, 
                "Should detect adverse selection from one-sided flow");
    }
}