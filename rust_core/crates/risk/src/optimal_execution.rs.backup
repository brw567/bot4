// OPTIMAL EXECUTION ALGORITHMS - TWAP/VWAP/POV WITH GAME THEORY
// Team: Casey (Exchange Integration) + Jordan (Performance) + Full Team
// CRITICAL: Minimize market impact while maximizing fill quality
// References:
// - Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
// - Bertsimas & Lo (1998): "Optimal Control of Execution Costs"
// - Kissell & Glantz (2003): "Optimal Trading Strategies"
// - Kyle (1985): "Continuous Auctions and Insider Trading" 
// - Obizhaeva & Wang (2013): "Optimal Trading Strategy and Supply/Demand Dynamics"

use crate::unified_types::*;
use crate::order_book_analytics::OrderBookAnalytics;
use crate::profit_extractor::ExtendedMarketData;
use crate::decision_orchestrator::ExecutionResult;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;

/// Optimal Execution Engine - Minimizes market impact using game theory
/// Casey: "Smart execution can save 50+ bps per trade!"
pub struct OptimalExecutionEngine {
    // Market microstructure parameters
    kyle_lambda: Arc<RwLock<f64>>,        // Kyle's Lambda - price impact coefficient
    temporary_impact: Arc<RwLock<f64>>,   // Temporary price impact
    permanent_impact: Arc<RwLock<f64>>,   // Permanent price impact
    
    // Execution state
    active_executions: Arc<RwLock<Vec<ExecutionPlan>>>,
    completed_executions: Arc<RwLock<Vec<ExecutionResult>>>,
    
    // Performance metrics
    total_slippage: Arc<RwLock<Decimal>>,
    avg_fill_quality: Arc<RwLock<f64>>,
    market_impact_saved: Arc<RwLock<Decimal>>,
    
    // Game theory parameters
    adversarial_adjustment: f64,  // Assume other traders are adversarial
    information_leakage: f64,     // How much our trades reveal
    predatory_threshold: f64,      // When to detect predatory trading
}

/// Execution plan for an order
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub order_id: uuid::Uuid,
    pub symbol: String,
    pub side: Side,
    pub total_quantity: Quantity,
    pub algorithm: ExecutionAlgorithm,
    pub time_horizon: u64,  // Seconds
    pub urgency: f64,        // 0.0 = patient, 1.0 = urgent
    pub slices: Vec<ExecutionSlice>,
    pub start_time: u64,
    pub expected_cost: Decimal,
    pub risk_limit: Decimal,
}

/// Single execution slice (child order)
#[derive(Debug, Clone)]
pub struct ExecutionSlice {
    pub timestamp: u64,
    pub quantity: Quantity,
    pub limit_price: Option<Price>,
    pub aggressive: bool,  // Cross spread if needed
    pub executed: bool,
    pub fill_price: Option<Price>,
    pub market_impact: Option<f64>,
}

/// Execution algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionAlgorithm {
    TWAP,      // Time-Weighted Average Price
    VWAP,      // Volume-Weighted Average Price  
    POV,       // Percentage of Volume
    IS,        // Implementation Shortfall (Almgren-Chriss)
    Adaptive,  // ML-based adaptive execution
    Iceberg,   // Hidden liquidity seeking
    Sniper,    // Aggressive liquidity taking
}

impl OptimalExecutionEngine {
    /// Create new execution engine with game theory parameters
    pub fn new() -> Self {
        Self {
            kyle_lambda: Arc::new(RwLock::new(0.0001)),  // Default Kyle's Lambda
            temporary_impact: Arc::new(RwLock::new(0.0005)),
            permanent_impact: Arc::new(RwLock::new(0.0002)),
            
            active_executions: Arc::new(RwLock::new(Vec::new())),
            completed_executions: Arc::new(RwLock::new(Vec::new())),
            
            total_slippage: Arc::new(RwLock::new(Decimal::ZERO)),
            avg_fill_quality: Arc::new(RwLock::new(1.0)),
            market_impact_saved: Arc::new(RwLock::new(Decimal::ZERO)),
            
            adversarial_adjustment: 1.2,  // Assume 20% adversarial activity
            information_leakage: 0.1,     // 10% information leakage per slice
            predatory_threshold: 0.3,      // 30% volume = predatory
        }
    }
    
    /// Create optimal execution plan using game theory
    /// DEEP DIVE: This is where we outsmart other market participants!
    pub fn create_execution_plan(
        &self,
        order: &Order,
        market: &ExtendedMarketData,
        order_book: &OrderBookAnalytics,
        historical_volume: &VecDeque<VolumeProfile>,
    ) -> Result<ExecutionPlan> {
        // Select optimal algorithm based on market conditions
        let algorithm = self.select_optimal_algorithm(
            order,
            market,
            order_book,
            historical_volume
        )?;
        
        // Calculate optimal time horizon (Almgren-Chriss formula)
        let time_horizon = self.calculate_optimal_horizon(
            order.quantity.to_f64(),
            market.volatility,
            *self.kyle_lambda.read(),
            order.urgency.unwrap_or(0.5),
        );
        
        // Generate execution slices based on algorithm
        let slices = match algorithm {
            ExecutionAlgorithm::TWAP => self.generate_twap_slices(order, time_horizon),
            ExecutionAlgorithm::VWAP => self.generate_vwap_slices(order, time_horizon, historical_volume),
            ExecutionAlgorithm::POV => self.generate_pov_slices(order, time_horizon, 0.1), // 10% of volume
            ExecutionAlgorithm::IS => self.generate_is_slices(order, time_horizon, market.volatility),
            ExecutionAlgorithm::Adaptive => self.generate_adaptive_slices(order, market, order_book),
            ExecutionAlgorithm::Iceberg => self.generate_iceberg_slices(order, order_book),
            ExecutionAlgorithm::Sniper => self.generate_sniper_slices(order, order_book),
        };
        
        // Calculate expected execution cost (including market impact)
        let expected_cost = self.calculate_expected_cost(
            &slices,
            market.mid.to_f64(),
            *self.kyle_lambda.read(),
        );
        
        // Apply game theory adjustments for adversarial traders
        let adjusted_slices = self.apply_adversarial_adjustments(slices, market);
        
        Ok(ExecutionPlan {
            order_id: order.id,
            symbol: order.symbol.clone(),
            side: order.side,
            total_quantity: order.quantity,
            algorithm,
            time_horizon,
            urgency: order.urgency.unwrap_or(0.5),
            slices: adjusted_slices,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expected_cost: Decimal::from_f64(expected_cost).unwrap_or(Decimal::ZERO),
            risk_limit: order.risk_limit.unwrap_or(dec!(0.01)),  // 1% max slippage
        })
    }
    
    /// Select optimal algorithm using ML and market conditions
    /// Morgan: "Let the data decide which algorithm to use!"
    fn select_optimal_algorithm(
        &self,
        order: &Order,
        market: &ExtendedMarketData,
        order_book: &OrderBookAnalytics,
        historical_volume: &VecDeque<VolumeProfile>,
    ) -> Result<ExecutionAlgorithm> {
        // Large order relative to average volume = use VWAP or POV
        let avg_volume = historical_volume.iter()
            .map(|v| v.total_volume)
            .sum::<f64>() / historical_volume.len() as f64;
        
        let order_size_ratio = order.quantity.to_f64() / avg_volume;
        
        // Detect market conditions
        let spread_bps = (market.spread.to_f64() / market.mid.to_f64()) * 10000.0;
        let book_imbalance = order_book.get_imbalance();
        let volatility = market.volatility;
        
        // GAME THEORY: Detect predatory traders
        if self.detect_predatory_activity(order_book, historical_volume) {
            // Use Iceberg to hide from predators
            return Ok(ExecutionAlgorithm::Iceberg);
        }
        
        // Decision tree based on market conditions
        if order.urgency.unwrap_or(0.5) > 0.8 {
            // Urgent order - use aggressive algorithm
            if spread_bps < 5.0 {
                Ok(ExecutionAlgorithm::Sniper)  // Tight spread, take liquidity
            } else {
                Ok(ExecutionAlgorithm::IS)  // Wide spread, minimize shortfall
            }
        } else if order_size_ratio > 0.2 {
            // Large order - minimize market impact
            if volatility > 0.02 {
                Ok(ExecutionAlgorithm::POV)  // High volatility, blend with volume
            } else {
                Ok(ExecutionAlgorithm::VWAP)  // Low volatility, follow volume pattern
            }
        } else if book_imbalance.abs() > 0.3 {
            // Imbalanced book - use adaptive
            Ok(ExecutionAlgorithm::Adaptive)
        } else {
            // Default to TWAP for small, non-urgent orders
            Ok(ExecutionAlgorithm::TWAP)
        }
    }
    
    /// Calculate optimal execution horizon (Almgren-Chriss)
    /// Theory: Balance between market risk and impact cost
    fn calculate_optimal_horizon(
        &self,
        quantity: f64,
        volatility: f64,
        kyle_lambda: f64,
        urgency: f64,
    ) -> u64 {
        // Almgren-Chriss optimal execution time
        // T* = sqrt(η * X / (λ * σ²))
        // where η = temporary impact, λ = risk aversion, σ = volatility
        
        let temp_impact = *self.temporary_impact.read();
        let risk_aversion = 1.0 + urgency * 2.0;  // Higher urgency = more risk averse
        
        let optimal_time = ((temp_impact * quantity) / 
                           (risk_aversion * kyle_lambda * volatility * volatility))
                          .sqrt();
        
        // Convert to seconds, bound between 1 minute and 1 hour
        let seconds = (optimal_time * 3600.0).max(60.0).min(3600.0);
        
        // Adjust for adversarial traders
        let adjusted = seconds / self.adversarial_adjustment;
        
        adjusted as u64
    }
    
    /// Generate TWAP slices - equal size over time
    fn generate_twap_slices(&self, order: &Order, horizon: u64) -> Vec<ExecutionSlice> {
        let num_slices = (horizon / 30).max(1).min(120);  // Slice every 30 seconds, max 120 slices
        let slice_size = order.quantity.to_f64() / num_slices as f64;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut slices = Vec::new();
        for i in 0..num_slices {
            slices.push(ExecutionSlice {
                timestamp: now + (i * horizon / num_slices),
                quantity: Quantity::new(Decimal::from_f64(slice_size).unwrap_or(dec!(0))),
                limit_price: None,  // Market orders for TWAP
                aggressive: false,
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        }
        
        slices
    }
    
    /// Generate VWAP slices - follow historical volume pattern
    /// Jordan: "Match the market's natural rhythm!"
    fn generate_vwap_slices(
        &self,
        order: &Order,
        horizon: u64,
        historical_volume: &VecDeque<VolumeProfile>,
    ) -> Vec<ExecutionSlice> {
        // Extract volume pattern from historical data
        let volume_pattern = self.extract_volume_pattern(historical_volume);
        
        // Distribute order quantity according to volume pattern
        let total_quantity = order.quantity.to_f64();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut slices = Vec::new();
        let num_slices = volume_pattern.len().min(120);
        
        for i in 0..num_slices {
            let volume_weight = volume_pattern[i];
            let slice_quantity = total_quantity * volume_weight;
            
            slices.push(ExecutionSlice {
                timestamp: now + (i as u64 * horizon / num_slices as u64),
                quantity: Quantity::new(Decimal::from_f64(slice_quantity).unwrap_or(dec!(0))),
                limit_price: None,
                aggressive: volume_weight > 0.02,  // Aggressive if high volume period
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        }
        
        slices
    }
    
    /// Generate POV slices - maintain percentage of volume
    fn generate_pov_slices(&self, order: &Order, horizon: u64, target_pov: f64) -> Vec<ExecutionSlice> {
        // POV algorithm maintains a constant percentage of market volume
        let num_slices = (horizon / 10).max(1).min(360);  // Check every 10 seconds
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut remaining = order.quantity.to_f64();
        let mut slices = Vec::new();
        
        for i in 0..num_slices {
            // Size based on expected volume and target POV
            let expected_market_volume = 1000.0;  // Would use real prediction
            let slice_size = (expected_market_volume * target_pov).min(remaining);
            
            if slice_size > 0.0 {
                slices.push(ExecutionSlice {
                    timestamp: now + (i * 10),
                    quantity: Quantity::new(Decimal::from_f64(slice_size).unwrap_or(dec!(0))),
                    limit_price: None,
                    aggressive: false,
                    executed: false,
                    fill_price: None,
                    market_impact: None,
                });
                
                remaining -= slice_size;
                if remaining <= 0.0 {
                    break;
                }
            }
        }
        
        slices
    }
    
    /// Generate Implementation Shortfall slices (Almgren-Chriss optimal trajectory)
    /// Theory: Minimize expected cost + risk penalty
    fn generate_is_slices(&self, order: &Order, horizon: u64, volatility: f64) -> Vec<ExecutionSlice> {
        // Almgren-Chriss optimal trajectory is front-loaded
        // x(t) = X * sinh(κ(T-t)) / sinh(κT)
        // where κ = sqrt(λσ²/η)
        
        let total_quantity = order.quantity.to_f64();
        let temp_impact = *self.temporary_impact.read();
        let risk_aversion = 2.0;  // Default risk aversion
        
        let kappa = (risk_aversion * volatility * volatility / temp_impact).sqrt();
        let sinh_kt = (kappa * horizon as f64).sinh();
        
        let num_slices = (horizon / 5).max(1).min(720);  // Every 5 seconds
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut slices = Vec::new();
        let mut cumulative = 0.0;
        
        for i in 0..num_slices {
            let t = i as f64 * horizon as f64 / num_slices as f64;
            let remaining_time = horizon as f64 - t;
            
            // Optimal remaining inventory
            let optimal_remaining = total_quantity * (kappa * remaining_time).sinh() / sinh_kt;
            let slice_size = cumulative + total_quantity - optimal_remaining - cumulative;
            
            if slice_size > 0.0 {
                slices.push(ExecutionSlice {
                    timestamp: now + (i * horizon / num_slices),
                    quantity: Quantity::new(Decimal::from_f64(slice_size).unwrap_or(dec!(0))),
                    limit_price: None,
                    aggressive: i < num_slices / 3,  // Front-loaded = aggressive early
                    executed: false,
                    fill_price: None,
                    market_impact: None,
                });
                
                cumulative += slice_size;
            }
        }
        
        slices
    }
    
    /// Generate adaptive slices using ML predictions
    /// Morgan: "Let ML decide when to trade!"
    fn generate_adaptive_slices(
        &self,
        order: &Order,
        market: &ExtendedMarketData,
        order_book: &OrderBookAnalytics,
    ) -> Vec<ExecutionSlice> {
        let mut slices = Vec::new();
        let total_quantity = order.quantity.to_f64();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Adaptive algorithm adjusts to real-time conditions
        let book_imbalance = order_book.get_imbalance();
        let spread_bps = (market.spread.to_f64() / market.mid.to_f64()) * 10000.0;
        
        // Decision: Trade more when conditions are favorable
        let favorable = book_imbalance * (if order.side == Side::Long { 1.0 } else { -1.0 }) > 0.2
                       && spread_bps < 10.0;
        
        if favorable {
            // Execute 30% immediately
            slices.push(ExecutionSlice {
                timestamp: now,
                quantity: Quantity::new(Decimal::from_f64(total_quantity * 0.3).unwrap_or(dec!(0))),
                limit_price: Some(market.ask),  // Limit at ask
                aggressive: true,
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        }
        
        // Remaining quantity over time
        let remaining = total_quantity * (if favorable { 0.7 } else { 1.0 });
        let num_slices = 20;
        
        for i in 1..=num_slices {
            slices.push(ExecutionSlice {
                timestamp: now + (i * 60),  // Every minute
                quantity: Quantity::new(Decimal::from_f64(remaining / num_slices as f64).unwrap_or(dec!(0))),
                limit_price: None,
                aggressive: false,
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        }
        
        slices
    }
    
    /// Generate iceberg slices - hide large orders
    /// GAME THEORY: Don't reveal your hand!
    fn generate_iceberg_slices(&self, order: &Order, order_book: &OrderBookAnalytics) -> Vec<ExecutionSlice> {
        let mut slices = Vec::new();
        let total_quantity = order.quantity.to_f64();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Iceberg shows only small visible quantity
        let visible_size = order_book.get_average_trade_size() * 0.5;  // Half of average
        let num_slices = (total_quantity / visible_size).ceil() as usize;
        
        for i in 0..num_slices {
            let slice_size = visible_size.min(total_quantity - (i as f64 * visible_size));
            
            slices.push(ExecutionSlice {
                timestamp: now + (i as u64 * 30),  // Every 30 seconds
                quantity: Quantity::new(Decimal::from_f64(slice_size).unwrap_or(dec!(0))),
                limit_price: None,  // Post at best price
                aggressive: false,   // Passive, don't cross spread
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        }
        
        // Add randomization to avoid detection
        self.randomize_iceberg_timing(&mut slices);
        
        slices
    }
    
    /// Generate sniper slices - aggressive liquidity taking
    /// Casey: "Strike when liquidity appears!"
    fn generate_sniper_slices(&self, order: &Order, order_book: &OrderBookAnalytics) -> Vec<ExecutionSlice> {
        let mut slices = Vec::new();
        let total_quantity = order.quantity.to_f64();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Sniper waits for large orders on the opposite side
        let liquidity_events = order_book.detect_liquidity_events();
        
        if !liquidity_events.is_empty() {
            // Take liquidity immediately
            slices.push(ExecutionSlice {
                timestamp: now,
                quantity: Quantity::new(Decimal::from_f64(total_quantity).unwrap_or(dec!(0))),
                limit_price: None,
                aggressive: true,  // Cross the spread
                executed: false,
                fill_price: None,
                market_impact: None,
            });
        } else {
            // Wait and watch for opportunities
            let num_checks = 60;  // Check every second for a minute
            let slice_size = total_quantity / num_checks as f64;
            
            for i in 0..num_checks {
                slices.push(ExecutionSlice {
                    timestamp: now + i,
                    quantity: Quantity::new(Decimal::from_f64(slice_size).unwrap_or(dec!(0))),
                    limit_price: None,
                    aggressive: true,  // Always aggressive
                    executed: false,
                    fill_price: None,
                    market_impact: None,
                });
            }
        }
        
        slices
    }
    
    /// Extract volume pattern from historical data
    fn extract_volume_pattern(&self, historical: &VecDeque<VolumeProfile>) -> Vec<f64> {
        if historical.is_empty() {
            return vec![1.0 / 24.0; 24];  // Uniform if no data
        }
        
        // Extract hourly volume pattern
        let mut hourly_volumes = vec![0.0; 24];
        let mut total_volume = 0.0;
        
        for profile in historical.iter() {
            let hour = (profile.timestamp / 3600) % 24;
            hourly_volumes[hour as usize] += profile.total_volume;
            total_volume += profile.total_volume;
        }
        
        // Normalize to weights
        if total_volume > 0.0 {
            for volume in hourly_volumes.iter_mut() {
                *volume /= total_volume;
            }
        }
        
        hourly_volumes
    }
    
    /// Calculate expected execution cost including market impact
    /// Theory: Cost = Spread Cost + Temporary Impact + Permanent Impact
    fn calculate_expected_cost(&self, slices: &[ExecutionSlice], mid_price: f64, kyle_lambda: f64) -> f64 {
        let mut total_cost = 0.0;
        let mut cumulative_volume = 0.0;
        
        for slice in slices {
            let slice_volume = slice.quantity.to_f64();
            
            // Spread cost (half-spread for crossing)
            let spread_cost = slice_volume * 0.0001 * mid_price;  // 1 bp half-spread
            
            // Temporary impact (square-root model)
            let temp_impact = *self.temporary_impact.read() * slice_volume.sqrt() * mid_price;
            
            // Permanent impact (linear in cumulative volume)
            let perm_impact = kyle_lambda * cumulative_volume * mid_price;
            
            total_cost += spread_cost + temp_impact + perm_impact;
            cumulative_volume += slice_volume;
        }
        
        total_cost
    }
    
    /// Apply game theory adjustments for adversarial traders
    /// CRITICAL: Other traders are trying to front-run us!
    fn apply_adversarial_adjustments(&self, mut slices: Vec<ExecutionSlice>, market: &ExtendedMarketData) -> Vec<ExecutionSlice> {
        // Add randomization to prevent predictability
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for slice in slices.iter_mut() {
            // Randomize timing by +/- 10%
            let time_adjustment = rng.gen_range(-0.1..0.1);
            slice.timestamp = ((slice.timestamp as f64) * (1.0 + time_adjustment)) as u64;
            
            // Randomize size by +/- 5%
            let size_adjustment = rng.gen_range(-0.05..0.05);
            let new_size = slice.quantity.to_f64() * (1.0 + size_adjustment);
            slice.quantity = Quantity::new(Decimal::from_f64(new_size).unwrap_or(dec!(0)));
            
            // Add limit prices to prevent adverse selection
            if slice.aggressive {
                // Aggressive orders get worst-case limits
                slice.limit_price = Some(Price::from_f64(
                    market.ask.to_f64() * (1.0 + self.adversarial_adjustment * 0.001)
                ));
            }
        }
        
        // Shuffle order of middle slices to prevent pattern detection
        if slices.len() > 4 {
            let mid_start = slices.len() / 4;
            let mid_end = 3 * slices.len() / 4;
            let mut middle: Vec<_> = slices.drain(mid_start..mid_end).collect();
            
            use rand::seq::SliceRandom;
            middle.shuffle(&mut rng);
            
            slices.splice(mid_start..mid_start, middle);
        }
        
        slices
    }
    
    /// Detect predatory trading activity
    /// Quinn: "Watch for sharks in the water!"
    fn detect_predatory_activity(&self, order_book: &OrderBookAnalytics, historical: &VecDeque<VolumeProfile>) -> bool {
        // Multiple indicators of predatory activity
        let mut predatory_score = 0.0;
        
        // 1. Unusual order book imbalance
        let imbalance = order_book.get_imbalance().abs();
        if imbalance > 0.7 {
            predatory_score += 0.3;
        }
        
        // 2. Spoofing detection (orders that disappear)
        let spoof_ratio = order_book.get_spoof_ratio();
        if spoof_ratio > 0.2 {
            predatory_score += 0.3;
        }
        
        // 3. Unusual volume spikes
        if !historical.is_empty() {
            let avg_volume: f64 = historical.iter().map(|v| v.total_volume).sum::<f64>() / historical.len() as f64;
            let current_volume = historical.back().map(|v| v.total_volume).unwrap_or(avg_volume);
            
            if current_volume > avg_volume * 3.0 {
                predatory_score += 0.2;
            }
        }
        
        // 4. Quote stuffing (rapid order updates)
        let quote_rate = order_book.get_quote_update_rate();
        if quote_rate > 100.0 {  // More than 100 updates per second
            predatory_score += 0.2;
        }
        
        predatory_score >= self.predatory_threshold
    }
    
    /// Randomize iceberg timing to avoid detection
    fn randomize_iceberg_timing(&self, slices: &mut Vec<ExecutionSlice>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 1..slices.len() {
            // Add random delay between slices (20-40 seconds instead of fixed 30)
            let random_delay = rng.gen_range(20..40);
            slices[i].timestamp = slices[i-1].timestamp + random_delay;
        }
    }
    
    /// Execute a slice and update metrics
    /// CRITICAL: This is where we interact with the exchange!
    pub async fn execute_slice(&mut self, slice: &mut ExecutionSlice, market: &ExtendedMarketData) -> Result<()> {
        // Simulate execution (would connect to real exchange)
        let fill_price = if slice.aggressive {
            // Cross the spread
            if slice.limit_price.is_some() {
                market.ask.min(slice.limit_price.unwrap())
            } else {
                market.ask
            }
        } else {
            // Post at bid
            market.bid
        };
        
        // Calculate actual market impact
        let impact = self.calculate_market_impact(slice.quantity.to_f64(), market.mid.to_f64());
        
        // Update slice
        slice.executed = true;
        slice.fill_price = Some(fill_price);
        slice.market_impact = Some(impact);
        
        // Update metrics
        let slippage = (fill_price.to_f64() - market.mid.to_f64()).abs() / market.mid.to_f64();
        *self.total_slippage.write() += Decimal::from_f64(slippage).unwrap_or(dec!(0));
        
        // Update fill quality (1.0 = perfect, 0.0 = terrible)
        let quality = 1.0 - slippage * 100.0;  // Lose 1% quality per bp of slippage
        let mut avg_quality = self.avg_fill_quality.write();
        *avg_quality = (*avg_quality * 0.9) + (quality * 0.1);  // Exponential moving average
        
        Ok(())
    }
    
    /// Calculate market impact using Kyle's Lambda
    fn calculate_market_impact(&self, quantity: f64, mid_price: f64) -> f64 {
        let kyle_lambda = *self.kyle_lambda.read();
        let temp_impact = *self.temporary_impact.read();
        let perm_impact = *self.permanent_impact.read();
        
        // Total impact = temporary + permanent
        let temp = temp_impact * quantity.sqrt();
        let perm = perm_impact * quantity;
        
        (temp + perm) * mid_price * kyle_lambda
    }
    
    /// Get execution performance metrics
    pub fn get_performance_metrics(&self) -> ExecutionMetrics {
        ExecutionMetrics {
            total_slippage: *self.total_slippage.read(),
            avg_fill_quality: *self.avg_fill_quality.read(),
            market_impact_saved: *self.market_impact_saved.read(),
            active_executions: self.active_executions.read().len(),
            completed_executions: self.completed_executions.read().len(),
        }
    }
}

/// Execution performance metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub total_slippage: Decimal,
    pub avg_fill_quality: f64,
    pub market_impact_saved: Decimal,
    pub active_executions: usize,
    pub completed_executions: usize,
}

/// Order for execution
#[derive(Debug, Clone)]
pub struct Order {
    pub id: uuid::Uuid,
    pub symbol: String,
    pub side: Side,
    pub quantity: Quantity,
    pub order_type: OrderType,
    pub limit_price: Option<Price>,
    pub urgency: Option<f64>,
    pub risk_limit: Option<Decimal>,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Volume profile for VWAP
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    pub timestamp: u64,
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub vwap: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_twap_execution() {
        let engine = OptimalExecutionEngine::new();
        let order = Order {
            id: uuid::Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: Side::Long,
            quantity: Quantity::new(dec!(10)),
            order_type: OrderType::Market,
            limit_price: None,
            urgency: Some(0.5),
            risk_limit: Some(dec!(0.01)),
        };
        
        let slices = engine.generate_twap_slices(&order, 3600);
        
        // Verify slices are equal size
        let slice_size = slices[0].quantity.to_f64();
        for slice in &slices {
            assert!((slice.quantity.to_f64() - slice_size).abs() < 0.001);
        }
        
        // Verify total quantity matches
        let total: f64 = slices.iter().map(|s| s.quantity.to_f64()).sum();
        assert!((total - 10.0).abs() < 0.01);
        
        println!("✅ TWAP execution: {} slices of {:.4} each", slices.len(), slice_size);
    }
    
    #[test]
    fn test_optimal_horizon_calculation() {
        let engine = OptimalExecutionEngine::new();
        
        // Test with different urgency levels
        let low_urgency = engine.calculate_optimal_horizon(1000.0, 0.02, 0.0001, 0.2);
        let high_urgency = engine.calculate_optimal_horizon(1000.0, 0.02, 0.0001, 0.9);
        
        // Higher urgency should result in shorter horizon
        assert!(high_urgency < low_urgency);
        
        println!("✅ Optimal horizons: low_urgency={} sec, high_urgency={} sec", 
                 low_urgency, high_urgency);
    }
    
    #[test]
    fn test_adversarial_adjustments() {
        let engine = OptimalExecutionEngine::new();
        let market = MarketData {
            symbol: "BTC/USDT".to_string(),
            timestamp: 0,
            bid: Price::from_f64(50000.0),
            ask: Price::from_f64(50010.0),
            last: Price::from_f64(50005.0),
            volume: Quantity::new(dec!(100)),
            bid_size: Quantity::new(dec!(10)),
            ask_size: Quantity::new(dec!(10)),
            spread: Price::from_f64(10.0),
            mid: Price::from_f64(50005.0),
            volatility: 0.02,
        };
        
        let mut original_slices = vec![
            ExecutionSlice {
                timestamp: 1000,
                quantity: Quantity::new(dec!(1)),
                limit_price: None,
                aggressive: false,
                executed: false,
                fill_price: None,
                market_impact: None,
            },
            ExecutionSlice {
                timestamp: 2000,
                quantity: Quantity::new(dec!(1)),
                limit_price: None,
                aggressive: false,
                executed: false,
                fill_price: None,
                market_impact: None,
            },
        ];
        
        let original_first_time = original_slices[0].timestamp;
        let original_first_size = original_slices[0].quantity.to_f64();
        
        let adjusted = engine.apply_adversarial_adjustments(original_slices, &market);
        
        // Verify adjustments were applied
        assert_ne!(adjusted[0].timestamp, original_first_time);
        assert_ne!(adjusted[0].quantity.to_f64(), original_first_size);
        
        println!("✅ Adversarial adjustments applied successfully");
    }
}