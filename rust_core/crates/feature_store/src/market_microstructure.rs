//! Using canonical Trade from domain_types
pub use domain_types::trade::{Trade, TradeId, TradeError};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Market Microstructure Features - Advanced Order Flow Analysis
// DEEP DIVE: Ultra-low latency microstructure signals
//
// External Research Applied:
// - "Trades, Quotes and Prices" - Bouchaud, Bonart, Donier, Gould (2018)
// - "Empirical Market Microstructure" - Hasbrouck (2007)
// - "High-Frequency Trading: A Practical Guide" - Aldridge (2013)
// - PIN model (Easley, Kiefer, O'Hara, Paperman)
// - VPIN (Volume-Synchronized PIN) - Easley, Lopez de Prado, O'Hara
// - Amihud Illiquidity measure
// - Roll's effective spread estimator
// - Corwin-Schultz high-low spread estimator

use std::sync::Arc;
use std::collections::VecDeque;
use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};
use parking_lot::RwLock;
use statrs::distribution::{Normal, ContinuousCDF, Poisson, Discrete};
use nalgebra::{DVector, DMatrix};

use crate::{FeatureUpdate, FeatureValue};

/// Microstructure configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct MicrostructureConfig {
    pub enable_pin: bool,
    pub enable_vpin: bool,
    pub enable_amihud: bool,
    pub enable_roll: bool,
    pub enable_corwin_schultz: bool,
    pub enable_hasbrouck: bool,
    pub enable_order_flow: bool,
    pub enable_tick_rule: bool,
    pub bucket_size_ms: i64,
    pub volume_buckets: usize,
    pub min_trades: usize,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            enable_pin: true,
            enable_vpin: true,
            enable_amihud: true,
            enable_roll: true,
            enable_corwin_schultz: true,
            enable_hasbrouck: true,
            enable_order_flow: true,
            enable_tick_rule: true,
            bucket_size_ms: 100, // 100ms buckets for HFT
            volume_buckets: 50,
            min_trades: 10,
        }
    }
}

/// Market microstructure calculator
/// TODO: Add docs
pub struct MicrostructureCalculator {
    config: MicrostructureConfig,
    
    // Data buffers
    trades: Arc<RwLock<VecDeque<Trade>>>,
    quotes: Arc<RwLock<VecDeque<Quote>>>,
    order_flow: Arc<RwLock<VecDeque<OrderFlow>>>,
    
    // Computed metrics
    pin_estimate: Arc<RwLock<f64>>,
    vpin_estimate: Arc<RwLock<f64>>,
    effective_spread: Arc<RwLock<f64>>,
    realized_spread: Arc<RwLock<f64>>,
    price_impact: Arc<RwLock<f64>>,
}

impl MicrostructureCalculator {
    pub fn new(config: MicrostructureConfig) -> Self {
        Self {
            config,
            trades: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            quotes: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            order_flow: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            pin_estimate: Arc::new(RwLock::new(0.0)),
            vpin_estimate: Arc::new(RwLock::new(0.0)),
            effective_spread: Arc::new(RwLock::new(0.0)),
            realized_spread: Arc::new(RwLock::new(0.0)),
            price_impact: Arc::new(RwLock::new(0.0)),
        }
    }
    
    /// Calculate all microstructure features
    #[instrument(skip(self))]
    pub async fn calculate_features(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        
        // PIN (Probability of Informed Trading)
        if self.config.enable_pin {
            features.push(self.calculate_pin(symbol, timestamp).await?);
        }
        
        // VPIN (Volume-Synchronized PIN)
        if self.config.enable_vpin {
            features.push(self.calculate_vpin(symbol, timestamp).await?);
        }
        
        // Amihud Illiquidity
        if self.config.enable_amihud {
            features.push(self.calculate_amihud(symbol, timestamp).await?);
        }
        
        // Roll's effective spread
        if self.config.enable_roll {
            features.push(self.calculate_roll_spread(symbol, timestamp).await?);
        }
        
        // Corwin-Schultz spread estimator
        if self.config.enable_corwin_schultz {
            features.extend(self.calculate_corwin_schultz(symbol, timestamp).await?);
        }
        
        // Hasbrouck's information share
        if self.config.enable_hasbrouck {
            features.push(self.calculate_hasbrouck_info(symbol, timestamp).await?);
        }
        
        // Order flow imbalance
        if self.config.enable_order_flow {
            features.extend(self.calculate_order_flow(symbol, timestamp).await?);
        }
        
        // Tick rule classification
        if self.config.enable_tick_rule {
            features.extend(self.calculate_tick_rule(symbol, timestamp).await?);
        }
        
        Ok(features)
    }
    
    /// Calculate PIN using maximum likelihood estimation
    async fn calculate_pin(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        let trades = self.trades.read();
        
        // Get daily trade counts
        let mut buy_counts = Vec::new();
        let mut sell_counts = Vec::new();
        
        let mut current_day_buys = 0;
        let mut current_day_sells = 0;
        let mut current_day = None;
        
        for trade in trades.iter() {
            let trade_day = trade.timestamp.date_naive();
            
            if current_day.is_none() {
                current_day = Some(trade_day);
            }
            
            if Some(trade_day) != current_day {
                buy_counts.push(current_day_buys);
                sell_counts.push(current_day_sells);
                current_day_buys = 0;
                current_day_sells = 0;
                current_day = Some(trade_day);
            }
            
            if trade.is_buy {
                current_day_buys += 1;
            } else {
                current_day_sells += 1;
            }
        }
        
        if current_day_buys > 0 || current_day_sells > 0 {
            buy_counts.push(current_day_buys);
            sell_counts.push(current_day_sells);
        }
        
        // PIN model parameters (simplified MLE)
        let pin = if buy_counts.len() >= 5 {
            let total_days = buy_counts.len() as f64;
            
            // Calculate order imbalance days
            let imbalanced_days = buy_counts.iter().zip(sell_counts.iter())
                .filter(|(b, s)| {
                    let total = (*b + *s) as f64;
                    if total > 0.0 {
                        let imbalance = ((*b as f64) - (*s as f64)).abs() / total;
                        imbalance > 0.2 // 20% imbalance threshold
                    } else {
                        false
                    }
                })
                .count() as f64;
            
            // Probability of information event
            let alpha = imbalanced_days / total_days;
            
            // Probability of bad news given information
            let delta = 0.5; // Simplified assumption
            
            // Arrival rates
            let epsilon_b = buy_counts.iter().sum::<i32>() as f64 / total_days;
            let epsilon_s = sell_counts.iter().sum::<i32>() as f64 / total_days;
            let mu = (epsilon_b - epsilon_s).abs();
            
            // PIN = α * μ / (α * μ + ε_b + ε_s)
            if epsilon_b + epsilon_s > 0.0 {
                (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        *self.pin_estimate.write() = pin;
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "pin_informed_trading".to_string(),
            value: FeatureValue::Float(pin),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate VPIN (Volume-Synchronized PIN)
    async fn calculate_vpin(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        let trades = self.trades.read();
        
        if trades.len() < self.config.min_trades {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "vpin_toxicity".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Create volume buckets
        let total_volume: f64 = trades.iter().map(|t| t.volume).sum();
        let bucket_volume = total_volume / self.config.volume_buckets as f64;
        
        let mut volume_buckets = Vec::new();
        let mut current_bucket = VolumeBucket::new();
        
        for trade in trades.iter() {
            current_bucket.add_trade(trade);
            
            if current_bucket.volume >= bucket_volume {
                volume_buckets.push(current_bucket.clone());
                current_bucket = VolumeBucket::new();
            }
        }
        
        if current_bucket.volume > 0.0 {
            volume_buckets.push(current_bucket);
        }
        
        // Calculate VPIN
        let vpin = if volume_buckets.len() >= 2 {
            let order_imbalances: Vec<f64> = volume_buckets.iter()
                .map(|b| (b.buy_volume - b.sell_volume).abs())
                .collect();
            
            let total_imbalance: f64 = order_imbalances.iter().sum();
            let n = volume_buckets.len() as f64;
            
            total_imbalance / (n * bucket_volume)
        } else {
            0.0
        };
        
        *self.vpin_estimate.write() = vpin;
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "vpin_toxicity".to_string(),
            value: FeatureValue::Float(vpin),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate Amihud illiquidity measure
    async fn calculate_amihud(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        let trades = self.trades.read();
        
        // Amihud = (1/N) * Σ |return| / volume
        let mut illiquidity_values = Vec::new();
        
        for window in trades.windows(2) {
            let return_pct = (window[1].price / window[0].price - 1.0).abs();
            let volume_dollars = window[1].price * window[1].volume;
            
            if volume_dollars > 0.0 {
                illiquidity_values.push(return_pct / volume_dollars);
            }
        }
        
        let amihud = if !illiquidity_values.is_empty() {
            illiquidity_values.iter().sum::<f64>() / illiquidity_values.len() as f64
        } else {
            0.0
        };
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "amihud_illiquidity".to_string(),
            value: FeatureValue::Float(amihud * 1e6), // Scale for readability
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate Roll's effective spread
    async fn calculate_roll_spread(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        let trades = self.trades.read();
        
        if trades.len() < 2 {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "roll_spread".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Roll's measure: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))
        let price_changes: Vec<f64> = trades.windows(2)
            .map(|w| (w[1].price / w[0].price).ln())
            .collect();
        
        if price_changes.len() < 2 {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "roll_spread".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Calculate autocovariance
        let mean_change = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        
        let autocovariance: f64 = price_changes.windows(2)
            .map(|w| (w[0] - mean_change) * (w[1] - mean_change))
            .sum::<f64>() / (price_changes.len() - 1) as f64;
        
        let roll_spread = if autocovariance < 0.0 {
            2.0 * (-autocovariance).sqrt()
        } else {
            0.0 // Positive autocovariance suggests no bid-ask bounce
        };
        
        *self.effective_spread.write() = roll_spread;
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "roll_spread".to_string(),
            value: FeatureValue::Float(roll_spread),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate Corwin-Schultz bid-ask spread estimator
    async fn calculate_corwin_schultz(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        let trades = self.trades.read();
        
        // Group trades into bars (e.g., 1-minute bars)
        let mut bars = Vec::new();
        let mut current_bar = PriceBar::new(timestamp);
        
        for trade in trades.iter() {
            if trade.timestamp.signed_duration_since(current_bar.start).num_seconds() >= 60 {
                if current_bar.trade_count > 0 {
                    bars.push(current_bar);
                }
                current_bar = PriceBar::new(trade.timestamp);
            }
            current_bar.update(trade.price);
        }
        
        if current_bar.trade_count > 0 {
            bars.push(current_bar);
        }
        
        if bars.len() < 2 {
            return Ok(features);
        }
        
        // Calculate spread using high-low estimator
        let mut spreads = Vec::new();
        
        for window in bars.windows(2) {
            let beta = ((window[0].high / window[0].low).ln().powi(2) +
                       (window[1].high / window[1].low).ln().powi(2)).sqrt();
            
            let gamma = ((window[0].high.max(window[1].high) /
                         window[0].low.min(window[1].low)).ln()).powi(2);
            
            let alpha = (2.0_f64.sqrt() - 1.0) * beta.sqrt() / 3.0 -
                       (gamma / (3.0 * 2.0_f64.sqrt())).sqrt();
            
            let spread = 2.0 * (alpha.exp() - 1.0) / (alpha.exp() + 1.0);
            
            if spread > 0.0 && spread < 1.0 {
                spreads.push(spread);
            }
        }
        
        let avg_spread = if !spreads.is_empty() {
            spreads.iter().sum::<f64>() / spreads.len() as f64
        } else {
            0.0
        };
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "corwin_schultz_spread".to_string(),
            value: FeatureValue::Float(avg_spread),
            timestamp,
            metadata: None,
        });
        
        // Also calculate spread volatility
        if spreads.len() > 1 {
            let spread_mean = spreads.iter().sum::<f64>() / spreads.len() as f64;
            let spread_var = spreads.iter()
                .map(|s| (s - spread_mean).powi(2))
                .sum::<f64>() / spreads.len() as f64;
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "spread_volatility".to_string(),
                value: FeatureValue::Float(spread_var.sqrt()),
                timestamp,
                metadata: None,
            });
        }
        
        Ok(features)
    }
    
    /// Calculate Hasbrouck's information share
    async fn calculate_hasbrouck_info(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        // Simplified Hasbrouck information share
        // Measures how much price discovery happens at this venue
        
        let trades = self.trades.read();
        let quotes = self.quotes.read();
        
        if trades.is_empty() || quotes.is_empty() {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "hasbrouck_info_share".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Calculate price innovations
        let mut price_innovations = Vec::new();
        
        for window in trades.windows(2) {
            let innovation = (window[1].price / window[0].price - 1.0).abs();
            price_innovations.push(innovation);
        }
        
        // Calculate quote midpoint innovations
        let mut quote_innovations = Vec::new();
        
        for window in quotes.windows(2) {
            let mid1 = (window[0].bid + window[0].ask) / 2.0;
            let mid2 = (window[1].bid + window[1].ask) / 2.0;
            let innovation = (mid2 / mid1 - 1.0).abs();
            quote_innovations.push(innovation);
        }
        
        // Information share = variance of trade innovations / total variance
        let trade_var = if !price_innovations.is_empty() {
            let mean = price_innovations.iter().sum::<f64>() / price_innovations.len() as f64;
            price_innovations.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / price_innovations.len() as f64
        } else {
            0.0
        };
        
        let quote_var = if !quote_innovations.is_empty() {
            let mean = quote_innovations.iter().sum::<f64>() / quote_innovations.len() as f64;
            quote_innovations.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / quote_innovations.len() as f64
        } else {
            0.0
        };
        
        let info_share = if trade_var + quote_var > 0.0 {
            trade_var / (trade_var + quote_var)
        } else {
            0.5
        };
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "hasbrouck_info_share".to_string(),
            value: FeatureValue::Float(info_share),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate order flow imbalance metrics
    async fn calculate_order_flow(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        let order_flow = self.order_flow.read();
        
        if order_flow.is_empty() {
            return Ok(features);
        }
        
        // Recent order flow (last N milliseconds)
        let cutoff = timestamp - Duration::milliseconds(self.config.bucket_size_ms);
        let recent_flow: Vec<_> = order_flow.iter()
            .filter(|f| f.timestamp >= cutoff)
            .collect();
        
        if recent_flow.is_empty() {
            return Ok(features);
        }
        
        // Calculate various order flow metrics
        let total_buy_volume: f64 = recent_flow.iter()
            .map(|f| f.buy_volume)
            .sum();
        
        let total_sell_volume: f64 = recent_flow.iter()
            .map(|f| f.sell_volume)
            .sum();
        
        let total_volume = total_buy_volume + total_sell_volume;
        
        // Order Flow Imbalance (OFI)
        let ofi = if total_volume > 0.0 {
            (total_buy_volume - total_sell_volume) / total_volume
        } else {
            0.0
        };
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "order_flow_imbalance".to_string(),
            value: FeatureValue::Float(ofi),
            timestamp,
            metadata: None,
        });
        
        // Cumulative Volume Delta (CVD)
        let cvd = total_buy_volume - total_sell_volume;
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "cumulative_volume_delta".to_string(),
            value: FeatureValue::Float(cvd),
            timestamp,
            metadata: None,
        });
        
        // Volume-Weighted Average Trade Size
        let avg_buy_size = if recent_flow.iter().map(|f| f.buy_count).sum::<u32>() > 0 {
            total_buy_volume / recent_flow.iter().map(|f| f.buy_count).sum::<u32>() as f64
        } else {
            0.0
        };
        
        let avg_sell_size = if recent_flow.iter().map(|f| f.sell_count).sum::<u32>() > 0 {
            total_sell_volume / recent_flow.iter().map(|f| f.sell_count).sum::<u32>() as f64
        } else {
            0.0
        };
        
        let trade_size_imbalance = if avg_buy_size + avg_sell_size > 0.0 {
            (avg_buy_size - avg_sell_size) / (avg_buy_size + avg_sell_size)
        } else {
            0.0
        };
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "trade_size_imbalance".to_string(),
            value: FeatureValue::Float(trade_size_imbalance),
            timestamp,
            metadata: None,
        });
        
        Ok(features)
    }
    
    /// Calculate tick rule classification
    async fn calculate_tick_rule(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        let trades = self.trades.read();
        
        if trades.len() < 2 {
            return Ok(features);
        }
        
        // Lee-Ready tick rule classification
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        let mut neutral_volume = 0.0;
        
        for window in trades.windows(2) {
            let price_change = window[1].price - window[0].price;
            
            if price_change > 0.0 {
                // Uptick - classify as buy
                buy_volume += window[1].volume;
            } else if price_change < 0.0 {
                // Downtick - classify as sell
                sell_volume += window[1].volume;
            } else {
                // Zero tick - look at previous direction
                neutral_volume += window[1].volume;
            }
        }
        
        let total_volume = buy_volume + sell_volume + neutral_volume;
        
        if total_volume > 0.0 {
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "tick_rule_buy_ratio".to_string(),
                value: FeatureValue::Float(buy_volume / total_volume),
                timestamp,
                metadata: None,
            });
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "tick_rule_sell_ratio".to_string(),
                value: FeatureValue::Float(sell_volume / total_volume),
                timestamp,
                metadata: None,
            });
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "tick_rule_neutral_ratio".to_string(),
                value: FeatureValue::Float(neutral_volume / total_volume),
                timestamp,
                metadata: None,
            });
        }
        
        Ok(features)
    }
    
    /// Update with new trade
    pub fn add_trade(&self, trade: Trade) {
        let mut trades = self.trades.write();
        trades.push_back(trade);
        
        // Keep only recent trades
        let cutoff = Utc::now() - Duration::seconds(300); // 5 minutes
        while let Some(front) = trades.front() {
            if front.timestamp < cutoff {
                trades.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Update with new quote
    pub fn add_quote(&self, quote: Quote) {
        let mut quotes = self.quotes.write();
        quotes.push_back(quote);
        
        // Keep only recent quotes
        let cutoff = Utc::now() - Duration::seconds(300);
        while let Some(front) = quotes.front() {
            if front.timestamp < cutoff {
                quotes.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Update order flow
    pub fn add_order_flow(&self, flow: OrderFlow) {
        let mut order_flow = self.order_flow.write();
        order_flow.push_back(flow);
        
        // Keep only recent flow
        let cutoff = Utc::now() - Duration::seconds(60); // 1 minute
        while let Some(front) = order_flow.front() {
            if front.timestamp < cutoff {
                order_flow.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Trade data
#[derive(Debug, Clone)]

/// Quote data
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: pub struct Quote {
// ELIMINATED:     pub timestamp: DateTime<Utc>,
// ELIMINATED:     pub bid: f64,
// ELIMINATED:     pub ask: f64,
// ELIMINATED:     pub bid_size: f64,
// ELIMINATED:     pub ask_size: f64,
// ELIMINATED: }

/// Order flow data
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: pub struct OrderFlow {
// ELIMINATED:     pub timestamp: DateTime<Utc>,
// ELIMINATED:     pub buy_volume: f64,
// ELIMINATED:     pub sell_volume: f64,
// ELIMINATED:     pub buy_count: u32,
// ELIMINATED:     pub sell_count: u32,
// ELIMINATED: }

/// Volume bucket for VPIN
#[derive(Debug, Clone)]
struct VolumeBucket {
    volume: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: u32,
}

impl VolumeBucket {
    fn new() -> Self {
        Self {
            volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            trade_count: 0,
        }
    }
    
    fn add_trade(&mut self, trade: &Trade) {
        self.volume += trade.volume;
        if trade.is_buy {
            self.buy_volume += trade.volume;
        } else {
            self.sell_volume += trade.volume;
        }
        self.trade_count += 1;
    }
}

/// Price bar for Corwin-Schultz estimator
#[derive(Debug, Clone)]
struct PriceBar {
    start: DateTime<Utc>,
    high: f64,
    low: f64,
    trade_count: u32,
}

impl PriceBar {
    fn new(start: DateTime<Utc>) -> Self {
        Self {
            start,
            high: 0.0,
            low: f64::MAX,
            trade_count: 0,
        }
    }
    
    fn update(&mut self, price: f64) {
        if self.trade_count == 0 {
            self.high = price;
            self.low = price;
        } else {
            self.high = self.high.max(price);
            self.low = self.low.min(price);
        }
        self.trade_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pin_calculation() {
        let config = MicrostructureConfig::default();
        let calculator = MicrostructureCalculator::new(config);
        
        // Add some trades
        for i in 0..100 {
            calculator.add_trade(Trade {
                timestamp: Utc::now() - Duration::seconds(i),
                price: 100.0 + (i as f64 * 0.01),
                volume: 100.0,
                is_buy: i % 3 != 0,
            });
        }
        
        let feature = calculator.calculate_pin("BTC", Utc::now()).await.unwrap();
        
        if let FeatureValue::Float(pin) = feature.value {
            assert!(pin >= 0.0 && pin <= 1.0);
        }
    }
    
    #[tokio::test]
    async fn test_roll_spread() {
        let config = MicrostructureConfig::default();
        let calculator = MicrostructureCalculator::new(config);
        
        // Add trades with bid-ask bounce pattern
        for i in 0..50 {
            calculator.add_trade(Trade {
                timestamp: Utc::now() - Duration::milliseconds(i * 100),
                price: if i % 2 == 0 { 100.01 } else { 99.99 },
                volume: 100.0,
                is_buy: i % 2 == 0,
            });
        }
        
        let feature = calculator.calculate_roll_spread("BTC", Utc::now()).await.unwrap();
        
        if let FeatureValue::Float(spread) = feature.value {
            assert!(spread > 0.0); // Should detect spread from bid-ask bounce
        }
    }
}