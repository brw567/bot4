// Value Object: Market Impact Models
// Implements realistic market impact for large orders
// Addresses Nexus's critical feedback on linear model limitations
// Owner: Morgan | Reviewer: Nexus

use anyhow::{Result, bail};
use std::f64::consts::E;

/// Market impact model types
#[derive(Debug, Clone)]
pub enum MarketImpactModel {
    /// Linear impact: impact = coefficient * volume
    /// Simple but underestimates large orders by 20-30%
    Linear { 
        coefficient: f64 
    },
    
    /// Square-root impact: impact = γ * √(Volume/ADV)
    /// Industry standard, recommended by Nexus
    SquareRoot { 
        gamma: f64,              // Impact coefficient (typically 0.01-0.1)
        daily_volume: f64,       // Average daily volume
    },
    
    /// Almgren-Chriss model for optimal execution
    /// Sophisticated model with permanent and temporary components
    AlmgrenChriss {
        permanent_impact: f64,   // α: Permanent price impact
        temporary_impact: f64,   // β: Temporary price impact
        decay_rate: f64,        // κ: Impact decay rate
    },
    
    /// Order book driven impact
    /// Walks the book to calculate actual impact
    OrderBookDriven {
        walk_depth: bool,       // Whether to walk full depth
        max_levels: usize,      // Maximum levels to consider
    },
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        // Square-root model with typical crypto parameters
        MarketImpactModel::SquareRoot {
            gamma: 0.05,              // 5% impact for √1 volume ratio
            daily_volume: 1_000_000.0, // $1M daily volume
        }
    }
}

impl MarketImpactModel {
    /// Create a conservative model (lower impact)
    pub fn conservative() -> Self {
        MarketImpactModel::SquareRoot {
            gamma: 0.02,
            daily_volume: 10_000_000.0,
        }
    }
    
    /// Create an aggressive model (higher impact)
    pub fn aggressive() -> Self {
        MarketImpactModel::SquareRoot {
            gamma: 0.10,
            daily_volume: 100_000.0,
        }
    }
    
    /// Create Almgren-Chriss model with standard parameters
    pub fn almgren_chriss_standard() -> Self {
        MarketImpactModel::AlmgrenChriss {
            permanent_impact: 0.01,   // 1% permanent impact per unit
            temporary_impact: 0.05,   // 5% temporary impact
            decay_rate: 0.1,         // Decay over ~10 time units
        }
    }
}

/// Market impact calculator
pub struct MarketImpact {
    model: MarketImpactModel,
}

impl MarketImpact {
    /// Create new market impact calculator
    pub fn new(model: MarketImpactModel) -> Self {
        MarketImpact { model }
    }
    
    /// Calculate price impact in basis points
    pub fn calculate_impact_bps(
        &self,
        order_size: f64,
        market_depth: Option<f64>,
        execution_time: Option<f64>,
    ) -> Result<f64> {
        if order_size <= 0.0 {
            bail!("Order size must be positive");
        }
        
        let impact = match &self.model {
            MarketImpactModel::Linear { coefficient } => {
                coefficient * order_size
            }
            
            MarketImpactModel::SquareRoot { gamma, daily_volume } => {
                // Impact = γ * √(Volume/ADV)
                // Nexus's recommended formula
                let volume_ratio = order_size / daily_volume;
                gamma * volume_ratio.sqrt() * 10000.0 // Convert to bps
            }
            
            MarketImpactModel::AlmgrenChriss { 
                permanent_impact, 
                temporary_impact, 
                decay_rate 
            } => {
                let time = execution_time.unwrap_or(1.0);
                let depth = market_depth.unwrap_or(order_size * 10.0);
                
                // Permanent impact: linear in size
                let permanent = permanent_impact * order_size;
                
                // Temporary impact: square-root of size/depth ratio, with decay
                let size_depth_ratio = order_size / depth;
                let temporary = temporary_impact * size_depth_ratio.sqrt() 
                    * E.powf(-decay_rate * time);
                
                (permanent + temporary) * 10000.0 // Convert to bps
            }
            
            MarketImpactModel::OrderBookDriven { walk_depth, max_levels } => {
                // Simplified order book impact
                let depth = market_depth.unwrap_or(order_size * 5.0);
                
                if *walk_depth {
                    // Walk the book: impact increases with square of penetration
                    let penetration = (order_size / depth).min(1.0);
                    let levels_crossed = (*max_levels as f64 * penetration) as usize;
                    
                    // Each level adds increasing impact
                    let mut total_impact = 0.0;
                    for level in 0..levels_crossed {
                        total_impact += (level + 1) as f64 * 2.0; // 2 bps per level
                    }
                    
                    total_impact
                } else {
                    // Simple approximation
                    let penetration = order_size / depth;
                    penetration * penetration * 1000.0 // Quadratic in penetration
                }
            }
        };
        
        Ok(impact)
    }
    
    /// Calculate slippage for a given order
    pub fn calculate_slippage(
        &self,
        base_price: f64,
        order_size: f64,
        is_buy: bool,
        market_depth: Option<f64>,
    ) -> Result<f64> {
        let impact_bps = self.calculate_impact_bps(order_size, market_depth, None)?;
        
        // Buy orders pay more (positive slippage)
        // Sell orders receive less (negative from seller's perspective)
        let slippage_multiplier = if is_buy { 1.0 } else { -1.0 };
        
        let slippage = base_price * (impact_bps / 10000.0) * slippage_multiplier;
        
        Ok(slippage)
    }
    
    /// Calculate effective execution price
    pub fn calculate_execution_price(
        &self,
        base_price: f64,
        order_size: f64,
        is_buy: bool,
        market_depth: Option<f64>,
    ) -> Result<f64> {
        let slippage = self.calculate_slippage(base_price, order_size, is_buy, market_depth)?;
        Ok(base_price + slippage)
    }
    
    /// Estimate optimal execution schedule (Almgren-Chriss)
    pub fn optimal_execution_schedule(
        &self,
        total_size: f64,
        time_horizon: f64,
        risk_aversion: f64,
    ) -> Result<Vec<(f64, f64)>> {
        match &self.model {
            MarketImpactModel::AlmgrenChriss { 
                permanent_impact, 
                temporary_impact, 
                decay_rate 
            } => {
                // Solve Euler-Lagrange equation for optimal trajectory
                // x(t) = X * sinh(κ(T-t)) / sinh(κT)
                
                let kappa = (risk_aversion * permanent_impact / temporary_impact).sqrt();
                let num_slices = 10;
                let dt = time_horizon / num_slices as f64;
                
                let mut schedule = Vec::new();
                
                for i in 0..num_slices {
                    let t = i as f64 * dt;
                    let remaining_time = time_horizon - t;
                    
                    // Optimal trading rate
                    let denominator = (kappa * time_horizon).sinh();
                    let numerator = (kappa * remaining_time).sinh();
                    
                    let fraction = if denominator != 0.0 {
                        numerator / denominator
                    } else {
                        1.0 / num_slices as f64
                    };
                    
                    let slice_size = total_size * fraction / num_slices as f64;
                    schedule.push((t, slice_size));
                }
                
                Ok(schedule)
            }
            _ => {
                // For non-Almgren-Chriss models, use simple TWAP
                let num_slices = 10;
                let slice_size = total_size / num_slices as f64;
                let dt = time_horizon / num_slices as f64;
                
                let schedule: Vec<(f64, f64)> = (0..num_slices)
                    .map(|i| (i as f64 * dt, slice_size))
                    .collect();
                
                Ok(schedule)
            }
        }
    }
}

/// Market depth snapshot
#[derive(Debug, Clone)]
pub struct MarketDepth {
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>, // (price, quantity)
    pub timestamp: i64,
}

impl MarketDepth {
    /// Calculate total liquidity up to a price level
    pub fn liquidity_to_price(&self, target_price: f64, is_buy: bool) -> f64 {
        if is_buy {
            // Sum ask liquidity up to target price
            self.asks.iter()
                .filter(|(price, _)| *price <= target_price)
                .map(|(_, qty)| qty)
                .sum()
        } else {
            // Sum bid liquidity down to target price
            self.bids.iter()
                .filter(|(price, _)| *price >= target_price)
                .map(|(_, qty)| qty)
                .sum()
        }
    }
    
    /// Calculate average daily volume (simplified)
    pub fn estimate_adv(&self) -> f64 {
        // Rough estimate: 100x the visible liquidity
        let total_liquidity: f64 = self.bids.iter().chain(self.asks.iter())
            .map(|(_, qty)| qty)
            .sum();
        
        total_liquidity * 100.0
    }
    
    /// Calculate market impact using actual order book
    pub fn calculate_walk_impact(&self, order_size: f64, is_buy: bool) -> f64 {
        let levels = if is_buy { &self.asks } else { &self.bids };
        
        let mut remaining = order_size;
        let mut total_cost = 0.0;
        let mut total_filled = 0.0;
        
        for (price, qty) in levels {
            if remaining <= 0.0 {
                break;
            }
            
            let fill = remaining.min(*qty);
            total_cost += fill * price;
            total_filled += fill;
            remaining -= fill;
        }
        
        if total_filled > 0.0 {
            let avg_price = total_cost / total_filled;
            let best_price = if is_buy { 
                levels.first().map(|(p, _)| *p).unwrap_or(0.0)
            } else {
                levels.first().map(|(p, _)| *p).unwrap_or(0.0)
            };
            
            // Impact in bps
            ((avg_price / best_price - 1.0).abs() * 10000.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_square_root_impact() {
        let model = MarketImpactModel::SquareRoot {
            gamma: 0.05,
            daily_volume: 1_000_000.0,
        };
        
        let impact = MarketImpact::new(model);
        
        // Small order: 0.1% of ADV
        let small_impact = impact.calculate_impact_bps(1000.0, None, None).expect("SAFETY: Add proper error handling");
        assert!(small_impact < 10.0); // Less than 10 bps
        
        // Large order: 10% of ADV
        let large_impact = impact.calculate_impact_bps(100_000.0, None, None).expect("SAFETY: Add proper error handling");
        assert!(large_impact > 100.0); // More than 100 bps
        assert!(large_impact < 200.0); // But less than 200 bps
        
        // Verify square-root relationship
        let ratio = large_impact / small_impact;
        let expected_ratio = (100.0_f64).sqrt(); // √(100k/1k) = √100 = 10
        assert!((ratio - expected_ratio).abs() < 0.1);
    }
    
    #[test]
    fn test_almgren_chriss_model() {
        let model = MarketImpactModel::almgren_chriss_standard();
        let impact = MarketImpact::new(model);
        
        // Test with decay over time
        let impact_t0 = impact.calculate_impact_bps(
            1000.0, 
            Some(10_000.0), 
            Some(0.0)
        ).expect("SAFETY: Add proper error handling");
        
        let impact_t10 = impact.calculate_impact_bps(
            1000.0, 
            Some(10_000.0), 
            Some(10.0)
        ).expect("SAFETY: Add proper error handling");
        
        // Impact should decay over time
        assert!(impact_t10 < impact_t0);
    }
    
    #[test]
    fn test_optimal_execution_schedule() {
        let model = MarketImpactModel::almgren_chriss_standard();
        let impact = MarketImpact::new(model);
        
        let schedule = impact.optimal_execution_schedule(
            10_000.0,  // Total size
            60.0,      // Time horizon (60 minutes)
            0.01,      // Risk aversion
        ).expect("SAFETY: Add proper error handling");
        
        // Should have 10 slices
        assert_eq!(schedule.len(), 10);
        
        // Total should sum to order size (approximately)
        let total: f64 = schedule.iter().map(|(_, size)| size).sum();
        assert!((total - 10_000.0).abs() < 100.0);
        
        // Early slices should be larger (front-loaded for Almgren-Chriss)
        // This depends on parameters, but generally true
        let first_half: f64 = schedule[..5].iter().map(|(_, s)| s).sum();
        let second_half: f64 = schedule[5..].iter().map(|(_, s)| s).sum();
        
        // With positive risk aversion, should front-load execution
        assert!(first_half >= second_half * 0.9); // Allow some tolerance
    }
    
    #[test]
    fn test_market_depth_walk() {
        let depth = MarketDepth {
            bids: vec![
                (49990.0, 1.0),
                (49980.0, 2.0),
                (49970.0, 3.0),
            ],
            asks: vec![
                (50010.0, 1.0),
                (50020.0, 2.0),
                (50030.0, 3.0),
            ],
            timestamp: 1234567890,
        };
        
        // Buy 4 units (cross 2 levels)
        let impact = depth.calculate_walk_impact(4.0, true);
        
        // Average price: (1*50010 + 2*50020 + 1*50030) / 4 = 50020
        // Best price: 50010
        // Impact: (50020/50010 - 1) * 10000 ≈ 20 bps
        assert!((impact - 20.0).abs() < 1.0);
    }
    
    #[test]
    fn test_slippage_calculation() {
        let model = MarketImpactModel::SquareRoot {
            gamma: 0.05,
            daily_volume: 1_000_000.0,
        };
        
        let impact = MarketImpact::new(model);
        
        // Buy order should have positive slippage (pay more)
        let buy_price = impact.calculate_execution_price(
            50000.0,
            10000.0,
            true,
            None,
        ).expect("SAFETY: Add proper error handling");
        
        assert!(buy_price > 50000.0);
        
        // Sell order should have negative slippage (receive less)
        let sell_price = impact.calculate_execution_price(
            50000.0,
            10000.0,
            false,
            None,
        ).expect("SAFETY: Add proper error handling");
        
        assert!(sell_price < 50000.0);
    }
}