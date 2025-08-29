pub use domain_types::trade::{Trade, TradeId, TradeError};

// ORDER BOOK EXTENSIONS - Complete Missing Methods
// Team: Full collaboration to complete all data pipes

use crate::decision_orchestrator::OrderBook;
use crate::unified_types::{Price, Quantity};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

/// Extension trait for OrderBook to add missing methods
pub trait OrderBookExtensions {
    fn total_bid_volume(&self) -> f64;
    fn total_ask_volume(&self) -> f64;
    fn volume_imbalance(&self) -> f64;
    fn bid_ask_spread(&self) -> f64;
    fn mid_price(&self) -> f64;
    fn order_flow_imbalance(&self) -> f64;
    fn depth_imbalance(&self) -> f64;
    fn spread_bps(&self) -> f64;
    fn recent_trades(&self) -> Vec<Trade>;
}

#[derive(Debug, Clone)]

impl OrderBookExtensions for OrderBook {
    /// Calculate total bid volume
    fn total_bid_volume(&self) -> f64 {
        self.bids.iter()
            .map(|order| order.quantity.to_f64())
            .sum()
    }
    
    /// Calculate total ask volume
    fn total_ask_volume(&self) -> f64 {
        self.asks.iter()
            .map(|order| order.quantity.to_f64())
            .sum()
    }
    
    /// Calculate volume imbalance
    fn volume_imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        
        if bid_vol + ask_vol > 0.0 {
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
        } else {
            0.0
        }
    }
    
    /// Calculate bid-ask spread
    fn bid_ask_spread(&self) -> f64 {
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let best_ask = self.asks.first().unwrap().price.to_f64();
            let best_bid = self.bids.first().unwrap().price.to_f64();
            best_ask - best_bid
        } else {
            0.0
        }
    }
    
    /// Calculate mid price
    fn mid_price(&self) -> f64 {
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let best_ask = self.asks.first().unwrap().price.to_f64();
            let best_bid = self.bids.first().unwrap().price.to_f64();
            (best_ask + best_bid) / 2.0
        } else {
            0.0
        }
    }
    
    /// Calculate order flow imbalance
    fn order_flow_imbalance(&self) -> f64 {
        // Simplified: use volume-weighted price pressure
        let bid_pressure = self.bids.iter()
            .take(5)  // Top 5 levels
            .enumerate()
            .map(|(i, order)| {
                let weight = 1.0 / (1.0 + i as f64);  // Decay by level
                order.quantity.to_f64() * weight
            })
            .sum::<f64>();
            
        let ask_pressure = self.asks.iter()
            .take(5)  // Top 5 levels
            .enumerate()
            .map(|(i, order)| {
                let weight = 1.0 / (1.0 + i as f64);  // Decay by level
                order.quantity.to_f64() * weight
            })
            .sum::<f64>();
        
        if bid_pressure + ask_pressure > 0.0 {
            (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
        } else {
            0.0
        }
    }
    
    /// Calculate depth imbalance
    fn depth_imbalance(&self) -> f64 {
        // Measure depth at multiple levels
        let mut bid_depth = 0.0;
        let mut ask_depth = 0.0;
        
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let mid = self.mid_price();
            let threshold = mid * 0.002;  // 0.2% from mid
            
            // Sum volume within threshold
            for bid in &self.bids {
                if (mid - bid.price.to_f64()) <= threshold {
                    bid_depth += bid.quantity.to_f64();
                } else {
                    break;
                }
            }
            
            for ask in &self.asks {
                if (ask.price.to_f64() - mid) <= threshold {
                    ask_depth += ask.quantity.to_f64();
                } else {
                    break;
                }
            }
        }
        
        if bid_depth + ask_depth > 0.0 {
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
        } else {
            0.0
        }
    }
    
    /// Calculate spread in basis points
    fn spread_bps(&self) -> f64 {
        let spread = self.bid_ask_spread();
        let mid = self.mid_price();
        
        if mid > 0.0 {
            (spread / mid) * 10000.0  // Convert to bps
        } else {
            0.0
        }
    }
    
    /// Get recent trades (simulated for now)
    fn recent_trades(&self) -> Vec<Trade> {
        // In production, this would come from WebSocket feed
        // For now, simulate from order book changes
        vec![
            Trade {
                price: self.mid_price(),
                volume: 1.0,
                is_buy: self.volume_imbalance() > 0.0,
                timestamp: self.timestamp,
            }
        ]
    }
}

/// Complete ML prediction structure
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MLPrediction {
    pub signal: f64,  // -1 to 1 (sell to buy)
    pub confidence: f64,  // 0 to 1
    pub features_used: Vec<String>,
    pub model_version: String,
}

impl MLPrediction {
    pub fn new(signal: f64, confidence: f64) -> Self {
        Self {
            signal,
            confidence,
            features_used: Vec::new(),
            model_version: "v1.0".to_string(),
        }
    }
}

/// Extension for MLFeedbackSystem to return proper predictions
pub trait MLFeedbackExtensions {
    fn predict(&self, features: &[f64]) -> MLPrediction;
    fn calibrate_probability(&self, raw_prob: f64) -> f64;
    fn update_prediction_history(&mut self, prediction: MLPrediction);
    fn should_retrain(&self) -> bool;
    fn online_learning_update(&mut self);
    fn add_training_example(&mut self, features: Vec<f64>, label: f64, weight: f64);
}

// Note: Implementation would go in ml_feedback.rs