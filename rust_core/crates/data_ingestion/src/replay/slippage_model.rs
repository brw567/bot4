// Slippage Modeling and Market Impact
// DEEP DIVE: Realistic execution cost modeling
//
// References:
// - "Optimal Trading with Stochastic Liquidity and Volatility" - Almgren (2012)
// - "Optimal Execution of Portfolio Transactions" - Almgren & Chriss (2001)
// - "The Cost of Algorithmic Trading" - Kissell (2013)
// - "Market Impact and Trading Profile of Hidden Orders" - Hautsch & Huang (2012)
// - JP Morgan's Implementation Shortfall analysis
// - Goldman Sachs' TCA framework

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Timelike};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use statrs::distribution::{Normal, ContinuousCDF};
use statrs::statistics::Statistics;

use types::{Price, Quantity, Symbol};
use crate::replay::lob_simulator::OrderBook;
// TODO: use infrastructure::metrics::{MetricsCollector, register_histogram};

/// Slippage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct SlippageConfig {
    /// Linear impact coefficient (basis points per unit size)
    pub linear_impact_bps: f64,
    
    /// Square-root impact coefficient (Almgren-Chriss)
    pub sqrt_impact_bps: f64,
    
    /// Temporary impact decay rate
    pub temp_impact_decay_rate: f64,
    
    /// Permanent impact ratio
    pub permanent_impact_ratio: f64,
    
    /// Spread cost multiplier
    pub spread_cost_multiplier: f64,
    
    /// Volatility adjustment factor
    pub volatility_factor: f64,
    
    /// Time of day adjustment
    pub enable_intraday_patterns: bool,
    
    /// Participation rate limit
    pub max_participation_rate: f64,
    
    /// Enable adverse selection modeling
    pub model_adverse_selection: bool,
    
    /// Opportunity cost rate (bps per minute)
    pub opportunity_cost_bps: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            linear_impact_bps: 0.1,
            sqrt_impact_bps: 0.5,
            temp_impact_decay_rate: 0.95,
            permanent_impact_ratio: 0.3,
            spread_cost_multiplier: 0.5,
            volatility_factor: 1.5,
            enable_intraday_patterns: true,
            max_participation_rate: 0.1,
            model_adverse_selection: true,
            opportunity_cost_bps: 0.05,
        }
    }
}

/// Execution cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ExecutionCost {
    pub symbol: Symbol,
    pub side: TradeSide,
    pub quantity: Quantity,
    pub arrival_price: Price,
    pub execution_price: Price,
    
    // Cost components (all in basis points)
    pub spread_cost_bps: f64,
    pub temporary_impact_bps: f64,
    pub permanent_impact_bps: f64,
    pub timing_cost_bps: f64,
    pub opportunity_cost_bps: f64,
    pub total_cost_bps: f64,
    
    // Additional metrics
    pub participation_rate: f64,
    pub execution_time_ms: u64,
    pub market_volatility: f64,
    pub adverse_selection_cost_bps: Option<f64>,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum TradeSide {
    Buy,
    Sell,
}

/// Market impact model types
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum MarketImpactModel {
    /// Linear impact: I = β * Q
    Linear { beta: f64 },
    
    /// Square-root impact: I = β * √Q (Almgren-Chriss)
    SquareRoot { beta: f64 },
    
    /// Power law: I = β * Q^α
    PowerLaw { beta: f64, alpha: f64 },
    
    /// Logarithmic: I = β * log(1 + Q)
    Logarithmic { beta: f64 },
    
    /// Hybrid model combining multiple effects
    Hybrid {
        linear_coef: f64,
        sqrt_coef: f64,
        log_coef: f64,
    },
}

/// Temporary impact component
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TemporaryImpact {
    pub initial_impact_bps: f64,
    pub decay_rate: f64,
    pub half_life_ms: u64,
}

/// Permanent impact component
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PermanentImpact {
    pub impact_bps: f64,
    pub information_ratio: f64,
}

/// Intraday patterns for liquidity
struct IntradayPattern {
    hour_multipliers: [f64; 24],
}

impl IntradayPattern {
    fn new() -> Self {
        // U-shaped intraday pattern (high at open/close, low midday)
        // Based on empirical studies of crypto markets
        let mut multipliers = [1.0; 24];
        
        // UTC hours (adjust for exchange timezone)
        multipliers[0..6] = [1.2, 1.1, 1.0, 0.9, 0.85, 0.8];    // Asia morning
        multipliers[6..12] = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05];  // Europe morning
        multipliers[12..18] = [1.1, 1.15, 1.2, 1.15, 1.1, 1.05]; // US morning
        multipliers[18..24] = [1.0, 0.95, 0.9, 0.95, 1.0, 1.1];  // US afternoon
        
        Self {
            hour_multipliers: multipliers,
        }
    }
    
    fn get_multiplier(&self, time: DateTime<Utc>) -> f64 {
        self.hour_multipliers[time.hour() as usize]
    }
}

/// Main slippage model implementation
/// TODO: Add docs
// ELIMINATED: pub struct SlippageModel {
// ELIMINATED:     config: Arc<SlippageConfig>,
// ELIMINATED:     
// ELIMINATED:     // Historical volatility tracker
// ELIMINATED:     volatility_tracker: Arc<RwLock<VolatilityTracker>>,
// ELIMINATED:     
// ELIMINATED:     // Intraday patterns
// ELIMINATED:     intraday_pattern: Arc<IntradayPattern>,
// ELIMINATED:     
// ELIMINATED:     // Metrics
// ELIMINATED:     slippage_histogram: Arc<dyn MetricsCollector>,
// ELIMINATED:     impact_histogram: Arc<dyn MetricsCollector>,
// ELIMINATED:     execution_time_histogram: Arc<dyn MetricsCollector>,
// ELIMINATED: }

/// Volatility tracker for dynamic adjustment
struct VolatilityTracker {
    returns: Vec<f64>,
    window_size: usize,
    ewma_alpha: f64,
    current_volatility: f64,
    realized_volatility: f64,
    garch_volatility: Option<f64>,
}

impl VolatilityTracker {
    fn new(window_size: usize) -> Self {
        Self {
            returns: Vec::with_capacity(window_size),
            window_size,
            ewma_alpha: 0.94, // RiskMetrics standard
            current_volatility: 0.02, // 2% default
            realized_volatility: 0.02,
            garch_volatility: None,
        }
    }
    
    fn update(&mut self, price: f64, prev_price: f64) {
        let return_val = (price / prev_price).ln();
        
        if self.returns.len() >= self.window_size {
            self.returns.remove(0);
        }
        self.returns.push(return_val);
        
        // Calculate realized volatility
        if self.returns.len() > 1 {
            let returns_vec: Vec<f64> = self.returns.clone();
            self.realized_volatility = returns_vec.std_dev();
            
            // EWMA volatility
            self.current_volatility = (self.ewma_alpha * self.current_volatility.powi(2) + 
                                      (1.0 - self.ewma_alpha) * return_val.powi(2)).sqrt();
        }
    }
    
    fn get_volatility(&self) -> f64 {
        // Use GARCH if available, otherwise EWMA
        self.garch_volatility.unwrap_or(self.current_volatility)
    }
}

impl SlippageModel {
    pub fn new(config: SlippageConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config),
            volatility_tracker: Arc::new(RwLock::new(VolatilityTracker::new(100))),
            intraday_pattern: Arc::new(IntradayPattern::new()),
            slippage_histogram: register_histogram("slippage_bps"),
            impact_histogram: register_histogram("market_impact_bps"),
            execution_time_histogram: register_histogram("execution_time_ms"),
        })
    }
    
    /// Calculate expected slippage for an order
    pub fn calculate_slippage(
        &self,
        symbol: &Symbol,
        side: TradeSide,
        quantity: Quantity,
        order_book: &OrderBook,
        execution_time_ms: u64,
    ) -> Result<ExecutionCost> {
        let arrival_price = self.get_arrival_price(order_book, side)?;
        
        // Calculate various cost components
        let spread_cost = self.calculate_spread_cost(order_book)?;
        let market_impact = self.calculate_market_impact(quantity.clone(), order_book)?;
        let (temp_impact, perm_impact) = self.split_impact(market_impact);
        let timing_cost = self.calculate_timing_cost(execution_time_ms);
        let opportunity_cost = self.calculate_opportunity_cost(quantity.clone(), execution_time_ms);
        
        // Adverse selection if enabled
        let adverse_selection = if self.config.model_adverse_selection {
            Some(self.calculate_adverse_selection(side, order_book)?)
        } else {
            None
        };
        
        // Intraday adjustment
        let intraday_mult = if self.config.enable_intraday_patterns {
            self.intraday_pattern.get_multiplier(order_book.timestamp)
        } else {
            1.0
        };
        
        // Volatility adjustment
        let vol_mult = 1.0 + self.volatility_tracker.read().get_volatility() * self.config.volatility_factor;
        
        // Calculate total cost
        let total_cost_bps = (spread_cost + temp_impact + perm_impact + timing_cost + opportunity_cost) 
                            * intraday_mult * vol_mult
                            + adverse_selection.unwrap_or(0.0);
        
        // Calculate execution price
        let price_impact = arrival_price.0 * Decimal::from_f64_retain(total_cost_bps / 10000.0)
            .unwrap_or(Decimal::ZERO);
        
        let execution_price = match side {
            TradeSide::Buy => Price(arrival_price.0 + price_impact),
            TradeSide::Sell => Price(arrival_price.0 - price_impact),
        };
        
        // Calculate participation rate
        let market_volume = self.estimate_market_volume(order_book);
        let participation_rate = quantity.0.to_f64().unwrap_or(0.0) / market_volume;
        
        // Record metrics
        self.slippage_histogram.record(total_cost_bps);
        self.impact_histogram.record(temp_impact + perm_impact);
        self.execution_time_histogram.record(execution_time_ms as f64);
        
        Ok(ExecutionCost {
            symbol: symbol.clone(),
            side,
            quantity,
            arrival_price,
            execution_price,
            spread_cost_bps: spread_cost,
            temporary_impact_bps: temp_impact,
            permanent_impact_bps: perm_impact,
            timing_cost_bps: timing_cost,
            opportunity_cost_bps: opportunity_cost,
            total_cost_bps,
            participation_rate,
            execution_time_ms,
            market_volatility: self.volatility_tracker.read().get_volatility(),
            adverse_selection_cost_bps: adverse_selection,
        })
    }
    
    /// Get arrival price (mid-market or touch price)
    fn get_arrival_price(&self, order_book: &OrderBook, side: TradeSide) -> Result<Price> {
        // Use weighted mid price as arrival
        order_book.weighted_mid_price
            .or(order_book.mid_price)
            .ok_or_else(|| anyhow::anyhow!("No mid price available"))
    }
    
    /// Calculate spread cost component
    fn calculate_spread_cost(&self, order_book: &OrderBook) -> Result<f64> {
        if let Some(spread) = order_book.spread {
            let mid_price = order_book.mid_price
                .ok_or_else(|| anyhow::anyhow!("No mid price"))?;
            
            // Half-spread cost in basis points
            let spread_bps = (spread / mid_price.0 * Decimal::from(10000) / Decimal::from(2))
                .to_f64()
                .unwrap_or(0.0);
            
            Ok(spread_bps * self.config.spread_cost_multiplier)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate market impact using configured model
    fn calculate_market_impact(&self, quantity: Quantity, order_book: &OrderBook) -> Result<f64> {
        let order_size = quantity.0.to_f64().unwrap_or(0.0);
        let market_depth = (order_book.total_bid_depth.0 + order_book.total_ask_depth.0)
            .to_f64()
            .unwrap_or(1.0);
        
        let normalized_size = order_size / market_depth;
        
        // Almgren-Chriss square-root model
        let sqrt_impact = self.config.sqrt_impact_bps * normalized_size.sqrt();
        
        // Linear impact for large orders
        let linear_impact = self.config.linear_impact_bps * normalized_size;
        
        // Kyle lambda (price impact per unit volume)
        let kyle_lambda = self.calculate_kyle_lambda(order_book);
        let kyle_impact = kyle_lambda * normalized_size * 10000.0; // Convert to bps
        
        // Take maximum of different models (conservative approach)
        Ok(sqrt_impact.max(linear_impact).max(kyle_impact))
    }
    
    /// Calculate Kyle's lambda from order book
    fn calculate_kyle_lambda(&self, order_book: &OrderBook) -> f64 {
        // Simplified Kyle lambda calculation
        // In production, would use order flow regression
        if let (Some(spread), Some(depth)) = (order_book.spread, order_book.total_bid_depth.0.to_f64()) {
            let spread_f64 = spread.to_f64().unwrap_or(0.0001);
            spread_f64 / (2.0 * depth)
        } else {
            0.0001 // Default conservative estimate
        }
    }
    
    /// Split impact into temporary and permanent components
    fn split_impact(&self, total_impact: f64) -> (f64, f64) {
        let permanent = total_impact * self.config.permanent_impact_ratio;
        let temporary = total_impact * (1.0 - self.config.permanent_impact_ratio);
        (temporary, permanent)
    }
    
    /// Calculate timing cost (delay cost)
    fn calculate_timing_cost(&self, execution_time_ms: u64) -> f64 {
        // Cost of delay based on typical price movement
        let minutes = execution_time_ms as f64 / 60000.0;
        let volatility = self.volatility_tracker.read().get_volatility();
        
        // Brownian motion expected move
        volatility * minutes.sqrt() * 10000.0 / 16.0 // Convert to bps, daily vol to minute vol
    }
    
    /// Calculate opportunity cost
    fn calculate_opportunity_cost(&self, quantity: Quantity, execution_time_ms: u64) -> f64 {
        let minutes = execution_time_ms as f64 / 60000.0;
        self.config.opportunity_cost_bps * minutes
    }
    
    /// Calculate adverse selection cost
    fn calculate_adverse_selection(&self, side: TradeSide, order_book: &OrderBook) -> Result<f64> {
        // Glosten-Milgrom model simplified
        // Adverse selection higher when:
        // 1. Order book is thin
        // 2. Spread is wide
        // 3. Recent volatility is high
        
        let spread_bps = if let Some(spread) = order_book.spread {
            let mid = order_book.mid_price.ok_or_else(|| anyhow::anyhow!("No mid price"))?;
            (spread / mid.0 * Decimal::from(10000)).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };
        
        let depth_imbalance = if order_book.total_bid_depth.0 + order_book.total_ask_depth.0 > Decimal::ZERO {
            ((order_book.total_bid_depth.0 - order_book.total_ask_depth.0).abs() /
             (order_book.total_bid_depth.0 + order_book.total_ask_depth.0))
                .to_f64().unwrap_or(0.0)
        } else {
            0.0
        };
        
        let volatility = self.volatility_tracker.read().get_volatility();
        
        // Adverse selection increases with spread, imbalance, and volatility
        let adverse_selection_bps = spread_bps * 0.1 + 
                                   depth_imbalance * 5.0 + 
                                   volatility * 100.0;
        
        Ok(adverse_selection_bps)
    }
    
    /// Estimate market volume from order book
    fn estimate_market_volume(&self, order_book: &OrderBook) -> f64 {
        // Estimate based on book depth and typical turnover
        let total_depth = (order_book.total_bid_depth.0 + order_book.total_ask_depth.0)
            .to_f64()
            .unwrap_or(1.0);
        
        // Assume 10x turnover of visible liquidity per period
        total_depth * 10.0
    }
    
    /// Update volatility tracker with new price
    pub fn update_volatility(&self, price: Price, prev_price: Price) {
        let mut tracker = self.volatility_tracker.write();
        tracker.update(
            price.0.to_f64().unwrap_or(0.0),
            prev_price.0.to_f64().unwrap_or(1.0),
        );
    }
    
    /// Simulate execution with path-dependent impact
    pub fn simulate_execution_path(
        &self,
        symbol: &Symbol,
        side: TradeSide,
        total_quantity: Quantity,
        num_slices: usize,
        order_book: &OrderBook,
    ) -> Result<Vec<ExecutionCost>> {
        let slice_size = Quantity(total_quantity.0 / Decimal::from(num_slices));
        let mut execution_path = Vec::new();
        let mut cumulative_impact = 0.0;
        
        for i in 0..num_slices {
            let execution_time = (i as u64 + 1) * 100; // 100ms per slice
            
            // Apply cumulative permanent impact
            let mut cost = self.calculate_slippage(
                symbol,
                side,
                slice_size.clone(),
                order_book,
                execution_time,
            )?;
            
            // Add cumulative permanent impact from previous slices
            cost.permanent_impact_bps += cumulative_impact;
            cost.total_cost_bps += cumulative_impact;
            
            // Decay temporary impact
            cumulative_impact += cost.permanent_impact_bps;
            cumulative_impact *= self.config.temp_impact_decay_rate;
            
            execution_path.push(cost);
        }
        
        Ok(execution_path)
    }
    
    /// Calculate implementation shortfall
    pub fn calculate_implementation_shortfall(
        &self,
        execution_costs: &[ExecutionCost],
        benchmark_price: Price,
    ) -> f64 {
        let total_quantity: Decimal = execution_costs.iter()
            .map(|c| c.quantity.0)
            .fold(Decimal::ZERO, |acc, q| acc + q);
        
        let total_cost: Decimal = execution_costs.iter()
            .map(|c| c.quantity.0 * c.execution_price.0)
            .fold(Decimal::ZERO, |acc, cost| acc + cost);
        
        if total_quantity > Decimal::ZERO {
            let avg_execution_price = total_cost / total_quantity;
            let shortfall = (avg_execution_price - benchmark_price.0) / benchmark_price.0 * Decimal::from(10000);
            shortfall.to_f64().unwrap_or(0.0).abs()
        } else {
            0.0
        }
    }
    
    /// Optimal execution schedule (Almgren-Chriss)
    pub fn calculate_optimal_schedule(
        &self,
        total_quantity: Quantity,
        time_horizon_ms: u64,
        risk_aversion: f64,
    ) -> Vec<(u64, Quantity)> {
        // Simplified Almgren-Chriss optimal execution
        // In production, would solve the full optimization problem
        
        let num_intervals = (time_horizon_ms / 100).max(1) as usize; // 100ms intervals
        let mut schedule = Vec::new();
        
        // Risk-averse: front-load execution
        // Risk-neutral: uniform execution
        // Risk-seeking: back-load execution
        
        for i in 0..num_intervals {
            let time = (i as u64 + 1) * 100;
            
            let weight = if risk_aversion > 1.0 {
                // Front-loaded (decreasing)
                2.0 * (num_intervals - i) as f64 / (num_intervals * (num_intervals + 1)) as f64
            } else if risk_aversion < -1.0 {
                // Back-loaded (increasing)
                2.0 * (i + 1) as f64 / (num_intervals * (num_intervals + 1)) as f64
            } else {
                // Uniform
                1.0 / num_intervals as f64
            };
            
            let quantity = Quantity(total_quantity.0 * Decimal::from_f64_retain(weight).unwrap_or(Decimal::ZERO));
            schedule.push((time, quantity));
        }
        
        schedule
    }
}

/// Advanced slippage models
/// TODO: Add docs
pub struct AdvancedSlippageModels;

impl AdvancedSlippageModels {
    /// Obizhaev-Wang model for large metaorders
    pub fn obizhaev_wang_impact(
        order_size: f64,
        daily_volume: f64,
        volatility: f64,
        num_days: f64,
    ) -> f64 {
        // I = 0.314 * σ * (Q/V)^0.6 * T^0.4
        let participation = order_size / (daily_volume * num_days);
        0.314 * volatility * participation.powf(0.6) * num_days.powf(0.4) * 10000.0
    }
    
    /// Hasbrouck model using trade indicator regression
    pub fn hasbrouck_impact(
        trade_sign: f64,  // +1 buy, -1 sell
        order_size: f64,
        avg_trade_size: f64,
    ) -> f64 {
        // Simplified Hasbrouck model
        let normalized_size = order_size / avg_trade_size;
        trade_sign * 0.1 * normalized_size.ln() * 10000.0
    }
    
    /// Gatheral no-dynamic-arbitrage model
    pub fn gatheral_impact(
        order_size: f64,
        market_depth: f64,
        decay_rate: f64,
        time: f64,
    ) -> f64 {
        let instantaneous = order_size / market_depth;
        let decay_factor = (-decay_rate * time).exp();
        instantaneous * decay_factor * 10000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::Exchange;
    use std::collections::BTreeMap;
    
    fn create_test_order_book() -> OrderBook {
        let mut book = OrderBook {
            symbol: Symbol("BTC-USDT".to_string()),
            exchange: Exchange("TestExchange".to_string()),
            timestamp: Utc::now(),
            sequence_number: 1,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: std::collections::HashMap::new(),
            is_crossed: false,
            is_locked: false,
            last_trade_price: None,
            last_trade_quantity: None,
            total_bid_depth: Quantity(Decimal::from(1000)),
            total_ask_depth: Quantity(Decimal::from(1000)),
            spread: Some(Decimal::from_str("10.0").unwrap()),
            mid_price: Some(Price(Decimal::from(50000))),
            weighted_mid_price: Some(Price(Decimal::from(50000))),
            micro_price: Some(Price(Decimal::from(50000))),
        };
        book
    }
    
    #[test]
    fn test_slippage_calculation() {
        let config = SlippageConfig::default();
        let model = SlippageModel::new(config).unwrap();
        
        let symbol = Symbol("BTC-USDT".to_string());
        let book = create_test_order_book();
        
        let cost = model.calculate_slippage(
            &symbol,
            TradeSide::Buy,
            Quantity(Decimal::from(10)),
            &book,
            100,
        ).unwrap();
        
        assert!(cost.total_cost_bps > 0.0);
        assert!(cost.spread_cost_bps >= 0.0);
        assert!(cost.temporary_impact_bps >= 0.0);
        assert!(cost.permanent_impact_bps >= 0.0);
    }
    
    #[test]
    fn test_execution_path_simulation() {
        let config = SlippageConfig::default();
        let model = SlippageModel::new(config).unwrap();
        
        let symbol = Symbol("ETH-USDT".to_string());
        let book = create_test_order_book();
        
        let path = model.simulate_execution_path(
            &symbol,
            TradeSide::Sell,
            Quantity(Decimal::from(100)),
            10,
            &book,
        ).unwrap();
        
        assert_eq!(path.len(), 10);
        
        // Check that permanent impact accumulates
        let first_permanent = path[0].permanent_impact_bps;
        let last_permanent = path[9].permanent_impact_bps;
        assert!(last_permanent >= first_permanent);
    }
    
    #[test]
    fn test_optimal_schedule() {
        let config = SlippageConfig::default();
        let model = SlippageModel::new(config).unwrap();
        
        // Risk-averse schedule (front-loaded)
        let schedule = model.calculate_optimal_schedule(
            Quantity(Decimal::from(1000)),
            1000,
            2.0,
        );
        
        assert!(!schedule.is_empty());
        
        // Check front-loading
        let first_qty = schedule[0].1.0;
        let last_qty = schedule[schedule.len() - 1].1.0;
        assert!(first_qty > last_qty);
    }
    
    #[test]
    fn test_advanced_models() {
        let impact = AdvancedSlippageModels::obizhaev_wang_impact(
            10000.0,  // Order size
            1000000.0, // Daily volume
            0.02,     // 2% volatility
            1.0,      // 1 day
        );
        
        assert!(impact > 0.0);
        assert!(impact < 100.0); // Reasonable bound
    }
}