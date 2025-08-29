// ORDER BOOK ANALYTICS EXTENSIONS
// Team: Casey (Exchange Expert) + Full Team
// Purpose: Add missing analytics methods to basic OrderBook

use crate::decision_orchestrator::OrderBook;
use crate::unified_types::{Price, Quantity};

/// Extension trait for OrderBook analytics
pub trait OrderBookAnalytics {
    fn total_bid_volume(&self) -> f64;
    fn total_ask_volume(&self) -> f64;
    fn volume_imbalance(&self) -> f64;
    fn bid_ask_spread(&self) -> f64;
    fn spread_bps(&self) -> f64;
    fn mid_price(&self) -> f64;
    fn order_flow_imbalance(&self) -> f64;
    fn depth_imbalance(&self, levels: usize) -> f64;
    fn weighted_mid_price(&self) -> f64;
    fn micro_price(&self) -> f64;
    fn book_pressure(&self) -> f64;
}

impl OrderBookAnalytics for OrderBook {
    /// Calculate total bid volume across all levels
    fn total_bid_volume(&self) -> f64 {
        self.bids.iter()
            .map(|order| order.quantity.to_f64())
            .sum()
    }
    
    /// Calculate total ask volume across all levels
    fn total_ask_volume(&self) -> f64 {
        self.asks.iter()
            .map(|order| order.quantity.to_f64())
            .sum()
    }
    
    /// Calculate volume imbalance (bid vs ask)
    /// Positive = more bid volume (bullish), Negative = more ask volume (bearish)
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
            let best_ask = self.asks[0].price.to_f64();
            let best_bid = self.bids[0].price.to_f64();
            best_ask - best_bid
        } else {
            0.0
        }
    }
    
    /// Calculate spread in basis points
    fn spread_bps(&self) -> f64 {
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let best_ask = self.asks[0].price.to_f64();
            let best_bid = self.bids[0].price.to_f64();
            let mid_price = (best_ask + best_bid) / 2.0;
            if mid_price > 0.0 {
                ((best_ask - best_bid) / mid_price) * 10000.0  // Convert to bps
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Calculate mid price (average of best bid and ask)
    fn mid_price(&self) -> f64 {
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let best_ask = self.asks[0].price.to_f64();
            let best_bid = self.bids[0].price.to_f64();
            (best_ask + best_bid) / 2.0
        } else {
            0.0
        }
    }
    
    /// Calculate order flow imbalance at best bid/ask
    /// This measures immediate buying vs selling pressure
    fn order_flow_imbalance(&self) -> f64 {
        if !self.bids.is_empty() && !self.asks.is_empty() {
            let best_bid_vol = self.bids[0].quantity.to_f64();
            let best_ask_vol = self.asks[0].quantity.to_f64();
            
            if best_bid_vol + best_ask_vol > 0.0 {
                (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Calculate depth imbalance at specified number of levels
    /// Measures cumulative volume imbalance across price levels
    fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_depth: f64 = self.bids.iter()
            .take(levels)
            .map(|order| order.quantity.to_f64())
            .sum();
            
        let ask_depth: f64 = self.asks.iter()
            .take(levels)
            .map(|order| order.quantity.to_f64())
            .sum();
            
        if bid_depth + ask_depth > 0.0 {
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
        } else {
            0.0
        }
    }
    
    /// Calculate volume-weighted mid price
    /// Weights price by volume at each level
    fn weighted_mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return self.mid_price();
        }
        
        let best_bid = self.bids[0].price.to_f64();
        let best_ask = self.asks[0].price.to_f64();
        let bid_vol = self.bids[0].quantity.to_f64();
        let ask_vol = self.asks[0].quantity.to_f64();
        
        if bid_vol + ask_vol > 0.0 {
            (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        } else {
            self.mid_price()
        }
    }
    
    /// Calculate micro price (probability-weighted price)
    /// Based on Kyle's Lambda and probability of execution
    fn micro_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return self.mid_price();
        }
        
        let best_bid = self.bids[0].price.to_f64();
        let best_ask = self.asks[0].price.to_f64();
        let bid_vol = self.bids[0].quantity.to_f64();
        let ask_vol = self.asks[0].quantity.to_f64();
        
        // Probability of bid execution (simplified)
        let prob_bid = ask_vol / (bid_vol + ask_vol).max(1.0);
        let prob_ask = bid_vol / (bid_vol + ask_vol).max(1.0);
        
        best_bid * prob_bid + best_ask * prob_ask
    }
    
    /// Calculate book pressure (buying vs selling pressure)
    /// Combines multiple metrics for overall pressure assessment
    fn book_pressure(&self) -> f64 {
        let volume_imb = self.volume_imbalance();
        let flow_imb = self.order_flow_imbalance();
        let depth_imb = self.depth_imbalance(5);
        
        // Weighted average of different imbalance measures
        // Flow imbalance is most important (50%), then volume (30%), then depth (20%)
        0.5 * flow_imb + 0.3 * volume_imb + 0.2 * depth_imb
    }
}

// Advanced market microstructure calculations
/// TODO: Add docs
pub struct MicrostructureMetrics;

impl MicrostructureMetrics {
    /// Calculate Kyle's Lambda (price impact coefficient)
    /// Lambda = (price change) / (volume traded)
    pub fn kyles_lambda(price_changes: &[f64], volumes: &[f64]) -> f64 {
        if price_changes.len() != volumes.len() || price_changes.is_empty() {
            return 0.0;
        }
        
        let total_impact: f64 = price_changes.iter()
            .zip(volumes.iter())
            .map(|(dp, v)| dp.abs() / v.max(&1.0))
            .sum();
            
        total_impact / price_changes.len() as f64
    }
    
    /// Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
    /// Measures flow toxicity in the order flow
    pub fn calculate_vpin(buy_volumes: &[f64], sell_volumes: &[f64], bucket_size: f64) -> f64 {
        if buy_volumes.len() != sell_volumes.len() || buy_volumes.is_empty() {
            return 0.0;
        }
        
        let mut vpin_values = Vec::new();
        let mut cumulative_volume = 0.0;
        let mut bucket_buy = 0.0;
        let mut bucket_sell = 0.0;
        
        for (buy, sell) in buy_volumes.iter().zip(sell_volumes.iter()) {
            bucket_buy += buy;
            bucket_sell += sell;
            cumulative_volume += buy + sell;
            
            if cumulative_volume >= bucket_size {
                let imbalance = (bucket_buy - bucket_sell).abs();
                let total = bucket_buy + bucket_sell;
                if total > 0.0 {
                    vpin_values.push(imbalance / total);
                }
                
                // Reset for next bucket
                cumulative_volume = 0.0;
                bucket_buy = 0.0;
                bucket_sell = 0.0;
            }
        }
        
        if !vpin_values.is_empty() {
            vpin_values.iter().sum::<f64>() / vpin_values.len() as f64
        } else {
            0.0
        }
    }
    
    /// Calculate Amihud Illiquidity Ratio
    /// Measures price impact per unit of trading volume
    pub fn amihud_illiquidity(returns: &[f64], volumes: &[f64]) -> f64 {
        if returns.len() != volumes.len() || returns.is_empty() {
            return 0.0;
        }
        
        let ratios: f64 = returns.iter()
            .zip(volumes.iter())
            .map(|(r, v)| r.abs() / v.max(&1.0))
            .sum();
            
        ratios / returns.len() as f64
    }
    
    /// Calculate Roll's Effective Spread
    /// Estimates effective spread from price changes
    pub fn roll_spread(price_changes: &[f64]) -> f64 {
        if price_changes.len() < 2 {
            return 0.0;
        }
        
        // Calculate autocovariance of price changes
        let mean_change = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        
        let mut autocovariance = 0.0;
        for i in 1..price_changes.len() {
            autocovariance += (price_changes[i] - mean_change) * (price_changes[i-1] - mean_change);
        }
        autocovariance /= (price_changes.len() - 1) as f64;
        
        // Roll's spread estimate: 2 * sqrt(-autocovariance) if negative
        if autocovariance < 0.0 {
            2.0 * (-autocovariance).sqrt()
        } else {
            0.0 // No spread estimate if autocovariance is positive
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_orchestrator::Order;
    
    #[test]
    fn test_order_book_analytics() {
        let order_book = OrderBook {
            bids: vec![
                Order { price: Price::from_f64(100.0), quantity: Quantity::from_f64(10.0) },
                Order { price: Price::from_f64(99.9), quantity: Quantity::from_f64(20.0) },
                Order { price: Price::from_f64(99.8), quantity: Quantity::from_f64(15.0) },
            ],
            asks: vec![
                Order { price: Price::from_f64(100.1), quantity: Quantity::from_f64(8.0) },
                Order { price: Price::from_f64(100.2), quantity: Quantity::from_f64(25.0) },
                Order { price: Price::from_f64(100.3), quantity: Quantity::from_f64(12.0) },
            ],
            timestamp: 1234567890,
        };
        
        assert_eq!(order_book.total_bid_volume(), 45.0);
        assert_eq!(order_book.total_ask_volume(), 45.0);
        assert_eq!(order_book.volume_imbalance(), 0.0);
        assert!((order_book.bid_ask_spread() - 0.1).abs() < 0.001);
        assert!((order_book.mid_price() - 100.05).abs() < 0.001);
    }
    
    #[test]
    fn test_microstructure_metrics() {
        let price_changes = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let volumes = vec![100.0, 150.0, 200.0, 120.0, 180.0];
        
        let lambda = MicrostructureMetrics::kyles_lambda(&price_changes, &volumes);
        assert!(lambda > 0.0);
        
        let buy_vols = vec![60.0, 80.0, 120.0, 70.0, 100.0];
        let sell_vols = vec![40.0, 70.0, 80.0, 50.0, 80.0];
        let vpin = MicrostructureMetrics::calculate_vpin(&buy_vols, &sell_vols, 200.0);
        assert!(vpin >= 0.0 && vpin <= 1.0);
    }
}