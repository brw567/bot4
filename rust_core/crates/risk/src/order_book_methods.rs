// ORDER BOOK METHODS - Complete implementation for EnhancedOrderBook
// Team: Casey (Exchange Integration) + Full Team
// References:
// - "Market Microstructure Theory" - O'Hara (1995)
// - "The Microstructure Approach to Exchange Rates" - Lyons (2001)
// - "High-Frequency Trading" - Aldridge (2013)

use crate::trading_types_complete::{EnhancedOrderBook, OrderLevel};
use crate::unified_types::{Price, Quantity};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

impl EnhancedOrderBook {
    /// Calculate total bid volume across all levels
    /// DEEP DIVE: Essential for liquidity assessment
    pub fn total_bid_volume(&self) -> Decimal {
        self.bids
            .iter()
            .map(|level| level.quantity.inner())  // Extract Decimal from Quantity
            .fold(Decimal::ZERO, |acc, size| acc + size)
    }
    
    /// Calculate total ask volume across all levels
    /// DEEP DIVE: Reveals supply pressure
    pub fn total_ask_volume(&self) -> Decimal {
        self.asks
            .iter()
            .map(|level| level.quantity.inner())  // Extract Decimal from Quantity
            .fold(Decimal::ZERO, |acc, size| acc + size)
    }
    
    /// Calculate weighted mid price
    /// Reference: "Optimal Execution" - Gatheral (2010)
    pub fn weighted_mid_price(&self) -> Price {
        if self.bids.is_empty() || self.asks.is_empty() {
            return Price::ZERO;
        }
        
        let best_bid = self.bids[0].price;
        let best_ask = self.asks[0].price;
        let bid_quantity = self.bids[0].quantity;
        let ask_quantity = self.asks[0].quantity;
        
        let total_size = bid_quantity + ask_quantity;
        if total_size.inner() == Decimal::ZERO {
            Price::new((best_bid.inner() + best_ask.inner()) / Decimal::from(2))
        } else {
            // Weighted mid: (bid_price * ask_size + ask_price * bid_size) / total_size
            let weighted_sum = best_bid.inner() * ask_quantity.inner() + 
                               best_ask.inner() * bid_quantity.inner();
            Price::new(weighted_sum / total_size.inner())
        }
    }
    
    /// Calculate Volume-Weighted Average Price (VWAP) for N levels
    /// DEEP DIVE: Better execution price estimation than mid-price
    pub fn calculate_vwap(&self, levels: usize) -> Price {
        let mut total_value = Decimal::ZERO;
        let mut total_volume = Decimal::ZERO;
        
        // Process bid side
        for level in self.bids.iter().take(levels) {
            total_value += level.price.inner() * level.quantity.inner();
            total_volume += level.quantity.inner();
        }
        
        // Process ask side
        for level in self.asks.iter().take(levels) {
            total_value += level.price.inner() * level.quantity.inner();
            total_volume += level.quantity.inner();
        }
        
        if total_volume == Decimal::ZERO {
            self.weighted_mid_price()
        } else {
            Price::new(total_value / total_volume)
        }
    }
    
    /// Calculate order book imbalance
    /// Reference: "Order Flow and Liquidity" - Cont et al. (2014)
    pub fn calculate_imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;
        
        if total == Decimal::ZERO {
            return 0.0;
        }
        
        ((bid_vol - ask_vol) / total).to_f64().unwrap_or(0.0)
    }
    
    /// Calculate Kyle's Lambda (price impact coefficient)
    /// Reference: "Continuous Auctions and Insider Trading" - Kyle (1985)
    /// Lambda = ΔP / ΔQ (price change per unit volume)
    pub fn calculate_kyle_lambda(&self, depth_levels: usize) -> f64 {
        if self.bids.len() < depth_levels || self.asks.len() < depth_levels {
            return 0.0001; // Default low impact
        }
        
        let mut price_impacts = Vec::new();
        
        for i in 0..depth_levels.min(self.bids.len()).min(self.asks.len()) {
            let bid_price = self.bids[i].price.to_f64();
            let ask_price = self.asks[i].price.to_f64();
            let mid_price = (bid_price + ask_price) / 2.0;
            
            let bid_quantity = self.bids[i].quantity.to_f64();
            let ask_quantity = self.asks[i].quantity.to_f64();
            let total_volume = bid_quantity + ask_quantity;
            
            // Price impact per unit volume
            let spread = ask_price - bid_price;
            let impact = spread / (mid_price * total_volume.max(1.0));
            
            price_impacts.push(impact);
        }
        
        // Average impact across levels
        if price_impacts.is_empty() {
            0.0001
        } else {
            price_impacts.iter().sum::<f64>() / price_impacts.len() as f64
        }
    }
    
    /// Calculate microprice (probability-weighted price)
    /// Reference: "The Microstructure of the Flash Crash" - Kirilenko et al. (2017)
    /// Microprice = (Bid * P(up) + Ask * P(down))
    pub fn calculate_microprice(&self) -> Price {
        if self.bids.is_empty() || self.asks.is_empty() {
            return Price::ZERO;
        }
        
        let best_bid = &self.bids[0];
        let best_ask = &self.asks[0];
        
        // Probability of up move based on relative sizes
        let total_size = best_bid.quantity + best_ask.quantity;
        
        if total_size.inner() == Decimal::ZERO {
            Price::new((best_bid.price.inner() + best_ask.price.inner()) / Decimal::from(2))
        } else {
            // More ask size = higher probability of down move
            let prob_up = best_ask.quantity.inner() / total_size.inner();
            let prob_down = best_bid.quantity.inner() / total_size.inner();
            
            Price::new(best_bid.price.inner() * prob_up + best_ask.price.inner() * prob_down)
        }
    }
    
    /// Calculate effective spread (actual trading cost)
    /// Reference: "Market Microstructure in Practice" - Lehalle & Laruelle (2018)
    pub fn calculate_effective_spread(&self, trade_price: Price, is_buy: bool) -> Price {
        let mid_price = self.weighted_mid_price();
        
        if is_buy {
            Price::new((trade_price.inner() - mid_price.inner()) * Decimal::from(2))
        } else {
            Price::new((mid_price.inner() - trade_price.inner()) * Decimal::from(2))
        }
    }
    
    /// Calculate realized spread (market maker profit)
    /// Reference: "Empirical Market Microstructure" - Hasbrouck (2007)
    pub fn calculate_realized_spread(&self, trade_price: Price, future_mid: Price, is_buy: bool) -> Price {
        if is_buy {
            Price::new((trade_price.inner() - future_mid.inner()) * Decimal::from(2))
        } else {
            Price::new((future_mid.inner() - trade_price.inner()) * Decimal::from(2))
        }
    }
    
    /// Get depth at price level (cumulative volume to price)
    pub fn get_depth_at_price(&self, target_price: Price, is_bid: bool) -> Decimal {
        let levels = if is_bid { &self.bids } else { &self.asks };
        
        let mut cumulative_volume = Decimal::ZERO;
        
        for level in levels {
            if (is_bid && level.price.inner() >= target_price.inner()) || 
               (!is_bid && level.price.inner() <= target_price.inner()) {
                cumulative_volume += level.quantity.inner();
            } else {
                break;
            }
        }
        
        cumulative_volume
    }
    
    /// Calculate book pressure (order flow toxicity indicator)
    /// Reference: "Flow Toxicity and Liquidity in a High Frequency World" - Easley et al. (2012)
    pub fn calculate_book_pressure(&self, depth: usize) -> f64 {
        let bid_pressure = self.bids
            .iter()
            .take(depth)
            .enumerate()
            .map(|(i, level)| {
                let weight = 1.0 / (i as f64 + 1.0); // Decay by level
                level.quantity.to_f64() * weight
            })
            .sum::<f64>();
            
        let ask_pressure = self.asks
            .iter()
            .take(depth)
            .enumerate()
            .map(|(i, level)| {
                let weight = 1.0 / (i as f64 + 1.0);
                level.quantity.to_f64() * weight
            })
            .sum::<f64>();
            
        let total_pressure = bid_pressure + ask_pressure;
        
        if total_pressure == 0.0 {
            0.0
        } else {
            (bid_pressure - ask_pressure) / total_pressure
        }
    }
    
    /// Check if book is crossed (arbitrage opportunity)
    pub fn is_crossed(&self) -> bool {
        if self.bids.is_empty() || self.asks.is_empty() {
            return false;
        }
        
        self.bids[0].price >= self.asks[0].price
    }
    
    /// Check if book is locked (best bid = best ask)
    pub fn is_locked(&self) -> bool {
        if self.bids.is_empty() || self.asks.is_empty() {
            return false;
        }
        
        self.bids[0].price == self.asks[0].price
    }
    
    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        
        let best_bid = self.bids[0].price.to_f64();
        let best_ask = self.asks[0].price.to_f64();
        let mid = (best_bid + best_ask) / 2.0;
        
        if mid == 0.0 {
            0.0
        } else {
            ((best_ask - best_bid) / mid) * 10000.0
        }
    }
    
    // Alias methods for compatibility with other modules
    
    /// Alias for calculate_imbalance
    pub fn volume_imbalance(&self) -> f64 {
        self.calculate_imbalance()
    }
    
    /// Calculate bid-ask spread in price units
    pub fn bid_ask_spread(&self) -> Price {
        if self.bids.is_empty() || self.asks.is_empty() {
            return Price::ZERO;
        }
        self.asks[0].price - self.bids[0].price
    }
    
    /// Get mid price (alias for weighted_mid_price)
    pub fn mid_price(&self) -> Price {
        self.weighted_mid_price()
    }
    
    /// Calculate order flow imbalance (alias for calculate_imbalance)
    pub fn order_flow_imbalance(&self) -> f64 {
        self.calculate_imbalance()
    }
    
    /// Calculate depth imbalance at multiple levels
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_depth: Decimal = self.bids.iter()
            .take(levels)
            .map(|l| l.quantity.inner())
            .sum();
            
        let ask_depth: Decimal = self.asks.iter()
            .take(levels)
            .map(|l| l.quantity.inner())
            .sum();
            
        let total = bid_depth + ask_depth;
        if total == Decimal::ZERO {
            0.0
        } else {
            ((bid_depth - ask_depth) / total).to_f64().unwrap_or(0.0)
        }
    }
}

// Extension trait for Price type conversion - made public for external use
pub trait PriceExt {
    fn to_f64(&self) -> f64;
}

impl PriceExt for Price {
    fn to_f64(&self) -> f64 {
        // Convert Decimal Price to f64 using inner value
        self.inner().to_f64().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_order_book_imbalance() {
        let mut book = EnhancedOrderBook::default();
        
        // Add bids (more volume)
        book.bids.push(OrderLevel {
            price: Price::new(dec!(50000)),
            quantity: Quantity::new(dec!(10)),
        });
        book.bids.push(OrderLevel {
            price: Price::new(dec!(49990)),
            quantity: Quantity::new(dec!(5)),
        });
        
        // Add asks (less volume)
        book.asks.push(OrderLevel {
            price: Price::new(dec!(50010)),
            quantity: Quantity::new(dec!(5)),
        });
        book.asks.push(OrderLevel {
            price: Price::new(dec!(50020)),
            quantity: Quantity::new(dec!(3)),
        });
        
        let imbalance = book.calculate_imbalance();
        assert!(imbalance > 0.0); // More bids than asks
        
        let bid_vol = book.total_bid_volume();
        assert_eq!(bid_vol, dec!(15));
        
        let ask_vol = book.total_ask_volume();
        assert_eq!(ask_vol, dec!(8));
    }
    
    #[test]
    fn test_kyle_lambda() {
        let mut book = EnhancedOrderBook::default();
        
        // Add 5 levels of depth
        for i in 0..5 {
            book.bids.push(OrderLevel {
                price: Price::new(dec!(50000) - dec!(10) * Decimal::from(i)),
                quantity: Quantity::new(dec!(10) + Decimal::from(i)),
            });
            book.asks.push(OrderLevel {
                price: Price::new(dec!(50010) + dec!(10) * Decimal::from(i)),
                quantity: Quantity::new(dec!(10) + Decimal::from(i)),
            });
        }
        
        let lambda = book.calculate_kyle_lambda(5);
        assert!(lambda > 0.0);
        assert!(lambda < 0.01); // Reasonable range for liquid market
    }
    
    #[test]
    fn test_microprice() {
        let mut book = EnhancedOrderBook::default();
        
        // Asymmetric book - more ask size
        book.bids.push(OrderLevel {
            price: Price::new(dec!(50000)),
            quantity: Quantity::new(dec!(5)),
        });
        book.asks.push(OrderLevel {
            price: Price::new(dec!(50010)),
            quantity: Quantity::new(dec!(15)), // 3x bid size
        });
        
        let microprice = book.calculate_microprice();
        let mid = Price::new((dec!(50000) + dec!(50010)) / dec!(2));
        
        // Microprice should be below mid (more ask pressure)
        assert!(microprice < mid);
    }
    
    #[test]
    fn test_book_pressure() {
        let mut book = EnhancedOrderBook::default();
        
        // Strong bid pressure
        for i in 0..3 {
            book.bids.push(OrderLevel {
                price: Price::new(dec!(50000) - dec!(10) * Decimal::from(i)),
                quantity: Quantity::new(dec!(20) - Decimal::from(i * 2)), // Decreasing size
            });
        }
        
        // Weak ask pressure
        for i in 0..3 {
            book.asks.push(OrderLevel {
                price: Price::new(dec!(50010) + dec!(10) * Decimal::from(i)),
                quantity: Quantity::new(dec!(5) + Decimal::from(i)), // Smaller sizes
            });
        }
        
        let pressure = book.calculate_book_pressure(3);
        assert!(pressure > 0.0); // Positive = bid pressure
    }
}