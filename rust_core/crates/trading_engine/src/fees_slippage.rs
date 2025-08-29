use domain_types::order::OrderError;
//! Module uses canonical Order type from domain_types
//! Avery: "Single source of truth for Order struct"

pub use domain_types::order::{
    Order, OrderId, OrderSide, OrderType, OrderStatus, TimeInForce,
    OrderError, Fill, FillId
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type OrderResult<T> = Result<T, OrderError>;

// Fee & Slippage Modeling - Realistic Market Microstructure
// Owner: Casey | Reviewer: Quinn (Risk), Sophia (Trading)
// Reference: "Optimal Trading Strategies" - Kissell & Glantz

use rust_decimal::Decimal;
use std::collections::HashMap;

/// Comprehensive fee structure per exchange

/// TODO: Add docs
pub struct FeeStructure {
    pub exchange: String,
    pub maker_fee_bps: Decimal,     // Negative = rebate
    pub taker_fee_bps: Decimal,
    pub tier: TradingTier,
}


/// TODO: Add docs
pub enum TradingTier {
    Tier1 { volume_30d: Decimal },  // >$100M
    Tier2 { volume_30d: Decimal },  // $50-100M
    Tier3 { volume_30d: Decimal },  // $10-50M
    Tier4 { volume_30d: Decimal },  // $1-10M
    Tier5 { volume_30d: Decimal },  // <$1M
}

/// Market impact model - Square root law

/// TODO: Add docs
// ELIMINATED: MarketImpactModel - Enhanced with Almgren-Chriss, Kyle lambda
// pub struct MarketImpactModel {
    /// γ coefficient (typically 0.1-0.3 for crypto)
    gamma: f64,
    
    /// Daily volume participation rate
    participation_rate: f64,
    
    /// Temporary vs permanent impact ratio
    temp_impact_ratio: f64,
}

impl MarketImpactModel {
    /// Calculate expected slippage using Almgren-Chriss model
    /// Impact = γ * σ * sqrt(V/ADV)
    /// Where:
    ///   γ = market impact coefficient
    ///   σ = volatility
    ///   V = trade volume
    ///   ADV = average daily volume
    pub fn calculate_impact(&self, 
        volume: f64, 
        volatility: f64, 
        adv: f64
    ) -> f64 {
        let participation = volume / adv;
        
        // Square-root impact model
        let permanent_impact = self.gamma * volatility * participation.sqrt();
        
        // Temporary impact (reversion)
        let temp_impact = permanent_impact * self.temp_impact_ratio;
        
        permanent_impact + temp_impact
    }
    
    /// Adverse selection cost model
    pub fn adverse_selection_cost(&self, 
        spread: f64, 
        order_size: f64,
        book_depth: f64
    ) -> f64 {
        // Probability of adverse price movement
        let adverse_prob = 0.5 + 0.3 * (order_size / book_depth).min(1.0);
        
        // Expected adverse move = half spread * probability
        spread * 0.5 * adverse_prob
    }
}

/// Queue position model for limit orders

/// TODO: Add docs
pub struct QueueModel {
    /// Average queue size at best bid/ask
    avg_queue_size: f64,
    
    /// Order arrival rate (orders per second)
    arrival_rate: f64,
    
    /// Cancellation rate
    cancel_rate: f64,
}

impl QueueModel {
    /// Estimate fill probability for limit order
    pub fn fill_probability(&self, 
        queue_position: usize,
        time_horizon: f64
    ) -> f64 {
        // M/M/1 queue model
        let service_rate = self.arrival_rate - self.cancel_rate;
        
        if service_rate <= 0.0 {
            return 0.0;
        }
        
        // Probability of reaching front of queue
        let p_front = (-queue_position as f64 / self.avg_queue_size).exp();
        
        // Probability of fill within time horizon
        let p_fill = 1.0 - (-service_rate * time_horizon).exp();
        
        p_front * p_fill
    }
    
    /// Expected time to fill
    pub fn expected_fill_time(&self, queue_position: usize) -> f64 {
        let service_rate = self.arrival_rate - self.cancel_rate;
        
        if service_rate <= 0.0 {
            return f64::INFINITY;
        }
        
        queue_position as f64 / service_rate
    }
}

/// LOB (Limit Order Book) simulator for backtesting
/// TODO: Add docs
pub struct LOBSimulator {
    /// Historical L2 data
    snapshots: Vec<OrderBookSnapshot>,
    
    /// Fee structures per venue
    fees: HashMap<String, FeeStructure>,
    
    /// Market impact model
    impact_model: MarketImpactModel,
    
    /// Queue model
    queue_model: QueueModel,
}

    pub timestamp: i64,
    pub bids: Vec<(f64, f64)>,  // (price, size)
    pub asks: Vec<(f64, f64)>,
    pub last_trade: Option<(f64, f64)>,
}

impl LOBSimulator {
    /// Simulate order execution with realistic fills
    pub fn simulate_order(&self, 
        order: &Order,
        snapshot: &OrderBookSnapshot
    ) -> ExecutionResult {
        match order.order_type {
            OrderType::Market => self.simulate_market_order(order, snapshot),
            OrderType::Limit => self.simulate_limit_order(order, snapshot),
            OrderType::PostOnly => self.simulate_post_only(order, snapshot),
        }
    }
    
    fn simulate_market_order(&self, 
        order: &Order,
        snapshot: &OrderBookSnapshot
    ) -> ExecutionResult {
        let (book, spread) = if order.side == Side::Buy {
            (&snapshot.asks, self.calculate_spread(snapshot))
        } else {
            (&snapshot.bids, self.calculate_spread(snapshot))
        };
        
        let mut remaining = order.quantity;
        let mut total_cost = 0.0;
        let mut fills = vec![];
        
        // Walk the book
        for &(price, size) in book {
            let fill_size = remaining.min(size);
            total_cost += price * fill_size;
            fills.push((price, fill_size));
            remaining -= fill_size;
            
            if remaining <= 0.0 {
                break;
            }
        }
        
        // Calculate fees
        let fee_structure = self.fees.get(&order.exchange).unwrap();
        let fee_bps = fee_structure.taker_fee_bps;
        let fee_amount = total_cost * fee_bps.to_f64().unwrap() / 10000.0;
        
        // Calculate slippage
        let mid_price = self.calculate_midpoint(snapshot);
        let avg_price = if order.quantity > 0.0 {
            total_cost / order.quantity
        } else {
            mid_price
        };
        let slippage = (avg_price - mid_price).abs() / mid_price;
        
        // Market impact
        let impact = self.impact_model.calculate_impact(
            order.quantity,
            0.02,  // 2% daily volatility assumption
            10000.0  // ADV assumption
        );
        
        // Adverse selection
        let adverse = self.impact_model.adverse_selection_cost(
            spread,
            order.quantity,
            self.total_book_depth(snapshot)
        );
        
        ExecutionResult {
            filled_quantity: order.quantity - remaining,
            average_price: avg_price,
            fees: fee_amount,
            slippage_bps: slippage * 10000.0,
            market_impact_bps: impact * 10000.0,
            adverse_selection_bps: adverse * 10000.0,
            net_edge: self.calculate_net_edge(order, avg_price, fee_amount, impact, adverse),
            fills,
        }
    }
    
    fn simulate_limit_order(&self, 
        order: &Order,
        snapshot: &OrderBookSnapshot
    ) -> ExecutionResult {
        let spread = self.calculate_spread(snapshot);
        let mid = self.calculate_midpoint(snapshot);
        
        // Estimate queue position
        let queue_pos = self.estimate_queue_position(order, snapshot);
        
        // Calculate fill probability
        let fill_prob = self.queue_model.fill_probability(
            queue_pos,
            60.0  // 60 second horizon
        );
        
        // Simulate fill based on probability
        let filled = if rand::random::<f64>() < fill_prob {
            order.quantity
        } else {
            0.0
        };
        
        // Calculate fees (maker fee/rebate)
        let fee_structure = self.fees.get(&order.exchange).unwrap();
        let fee_bps = fee_structure.maker_fee_bps;
        let fee_amount = filled * order.price * fee_bps.to_f64().unwrap() / 10000.0;
        
        // No slippage for limit orders at specified price
        let slippage = 0.0;
        
        // But there's opportunity cost and adverse selection
        let adverse = if filled > 0.0 {
            self.impact_model.adverse_selection_cost(spread, filled, self.total_book_depth(snapshot))
        } else {
            0.0
        };
        
        ExecutionResult {
            filled_quantity: filled,
            average_price: order.price,
            fees: fee_amount,
            slippage_bps: slippage,
            market_impact_bps: 0.0,
            adverse_selection_bps: adverse * 10000.0,
            net_edge: self.calculate_net_edge(order, order.price, fee_amount, 0.0, adverse),
            fills: if filled > 0.0 { vec![(order.price, filled)] } else { vec![] },
        }
    }
    
    fn simulate_post_only(&self, 
        order: &Order,
        snapshot: &OrderBookSnapshot
    ) -> ExecutionResult {
        // Post-only orders are rejected if they would cross
        let (best_bid, best_ask) = self.get_best_prices(snapshot);
        
        let would_cross = match order.side {
            Side::Buy => order.price >= best_ask,
            Side::Sell => order.price <= best_bid,
        };
        
        if would_cross {
            // Order rejected
            return ExecutionResult {
                filled_quantity: 0.0,
                average_price: 0.0,
                fees: 0.0,
                slippage_bps: 0.0,
                market_impact_bps: 0.0,
                adverse_selection_bps: 0.0,
                net_edge: 0.0,
                fills: vec![],
            };
        }
        
        // Otherwise, treat as limit order with maker rebate
        self.simulate_limit_order(order, snapshot)
    }
    
    fn calculate_spread(&self, snapshot: &OrderBookSnapshot) -> f64 {
        let (bid, ask) = self.get_best_prices(snapshot);
        ask - bid
    }
    
    fn calculate_midpoint(&self, snapshot: &OrderBookSnapshot) -> f64 {
        let (bid, ask) = self.get_best_prices(snapshot);
        (bid + ask) / 2.0
    }
    
    fn get_best_prices(&self, snapshot: &OrderBookSnapshot) -> (f64, f64) {
        let best_bid = snapshot.bids.first().map(|&(p, _)| p).unwrap_or(0.0);
        let best_ask = snapshot.asks.first().map(|&(p, _)| p).unwrap_or(f64::MAX);
        (best_bid, best_ask)
    }
    
    fn total_book_depth(&self, snapshot: &OrderBookSnapshot) -> f64 {
        let bid_depth: f64 = snapshot.bids.iter().map(|&(_, s)| s).sum();
        let ask_depth: f64 = snapshot.asks.iter().map(|&(_, s)| s).sum();
        bid_depth + ask_depth
    }
    
    fn estimate_queue_position(&self, order: &Order, snapshot: &OrderBookSnapshot) -> usize {
        // Estimate based on order price relative to best price
        let (best_bid, best_ask) = self.get_best_prices(snapshot);
        
        match order.side {
            Side::Buy => {
                if order.price < best_bid {
                    // Behind the queue
                    (self.queue_model.avg_queue_size * 1.5) as usize
                } else if order.price == best_bid {
                    // Join the queue
                    (self.queue_model.avg_queue_size * 0.5) as usize
                } else {
                    // Better price, front of queue
                    1
                }
            }
            Side::Sell => {
                if order.price > best_ask {
                    (self.queue_model.avg_queue_size * 1.5) as usize
                } else if order.price == best_ask {
                    (self.queue_model.avg_queue_size * 0.5) as usize
                } else {
                    1
                }
            }
        }
    }
    
    fn calculate_net_edge(&self, 
        order: &Order,
        fill_price: f64,
        fees: f64,
        impact: f64,
        adverse: f64
    ) -> f64 {
        let mid = order.expected_mid.unwrap_or(fill_price);
        let gross_edge = match order.side {
            Side::Buy => mid - fill_price,
            Side::Sell => fill_price - mid,
        };
        
        // Net edge = gross - fees - impact - adverse selection
        let cost_per_unit = (fees / order.quantity.max(1.0)) + impact + adverse;
        gross_edge - cost_per_unit
    }
}

    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub order_type: OrderType,
    pub exchange: String,
    pub expected_mid: Option<f64>,
}


/// TODO: Add docs
pub enum Side {
    Buy,
    Sell,
}


/// TODO: Add docs
pub enum OrderType {
    Market,
    Limit,
    PostOnly,
}


/// TODO: Add docs
pub struct ExecutionResult {
    pub filled_quantity: f64,
    pub average_price: f64,
    pub fees: f64,
    pub slippage_bps: f64,
    pub market_impact_bps: f64,
    pub adverse_selection_bps: f64,
    pub net_edge: f64,
    pub fills: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_impact() {
        let model = MarketImpactModel {
            gamma: 0.2,
            participation_rate: 0.1,
            temp_impact_ratio: 0.5,
        };
        
        let impact = model.calculate_impact(1000.0, 0.02, 100000.0);
        assert!(impact > 0.0 && impact < 0.01);  // Less than 1%
    }
    
    #[test]
    fn test_queue_fill_probability() {
        let queue = QueueModel {
            avg_queue_size: 100.0,
            arrival_rate: 10.0,
            cancel_rate: 5.0,
        };
        
        let prob = queue.fill_probability(50, 60.0);
        assert!(prob > 0.0 && prob < 1.0);
    }
    
    #[test]
    fn test_net_edge_positive() {
        // Test that we reject trades with negative net edge
        let sim = create_test_simulator();
        let order = Order {
            side: Side::Buy,
            quantity: 100.0,
            price: 50000.0,
            order_type: OrderType::Market,
            exchange: "binance".to_string(),
            expected_mid: Some(50000.0),
        };
        
        let snapshot = create_test_snapshot();
        let result = sim.simulate_order(&order, &snapshot);
        
        // Should have positive net edge for profitable trades
        if result.filled_quantity > 0.0 {
            assert!(result.net_edge > 0.0, "Should not execute negative edge trades");
        }
    }
}
