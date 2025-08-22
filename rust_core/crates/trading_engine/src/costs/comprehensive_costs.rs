// Comprehensive Trading Cost Model - Sophia's Critical Requirement
// Team: Casey (Lead) + Sam + Quinn + Full Team
// References: 
// - "Trading and Exchanges" by Larry Harris
// - "Market Microstructure Theory" by O'Hara
// - Binance/Coinbase fee schedules

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use anyhow::{Result, Context};

// ============================================================================
// COMPREHENSIVE COST MODEL - NO SHORTCUTS!
// ============================================================================

/// Complete trading cost model accounting for ALL costs
/// Sophia identified we were missing $1,800/month in costs!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveCostModel {
    /// Exchange-specific fee structures
    pub exchange_fees: HashMap<String, ExchangeFeeStructure>,
    
    /// Funding rate calculator for perpetuals
    pub funding_calculator: FundingRateCalculator,
    
    /// Market impact and slippage model
    pub slippage_model: SlippageModel,
    
    /// Spread cost estimator
    pub spread_cost_estimator: SpreadCostEstimator,
    
    /// Historical cost tracking
    pub cost_history: CostHistory,
}

/// Exchange fee structure with volume-based tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFeeStructure {
    pub exchange: String,
    pub spot_fees: TieredFeeSchedule,
    pub futures_fees: TieredFeeSchedule,
    pub withdrawal_fees: HashMap<String, Decimal>,
    pub min_order_sizes: HashMap<String, Decimal>,
}

/// Tiered fee schedule based on 30-day volume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredFeeSchedule {
    pub tiers: Vec<FeeTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTier {
    pub min_volume: Decimal,
    pub max_volume: Option<Decimal>,
    pub maker_fee: Decimal,  // Can be negative (rebate)
    pub taker_fee: Decimal,
}

impl TieredFeeSchedule {
    /// Get fees for current volume level
    pub fn get_fees(&self, volume_30d: Decimal) -> (Decimal, Decimal) {
        for tier in &self.tiers {
            let in_tier = volume_30d >= tier.min_volume && 
                         tier.max_volume.map_or(true, |max| volume_30d < max);
            if in_tier {
                return (tier.maker_fee, tier.taker_fee);
            }
        }
        // Default to highest tier if not found
        self.tiers.last()
            .map(|t| (t.maker_fee, t.taker_fee))
            .unwrap_or((dec!(0.001), dec!(0.001))) // 10bps default
    }
}

/// Funding rate calculator for perpetual futures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRateCalculator {
    /// Current funding rates by symbol
    pub current_rates: HashMap<String, FundingRate>,
    
    /// Historical funding rates for analysis
    pub historical_rates: Vec<HistoricalFunding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub rate: Decimal,  // Per 8 hours typically
    pub next_funding_time: i64,
    pub interval_hours: u32,  // Usually 8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalFunding {
    pub symbol: String,
    pub timestamp: i64,
    pub rate: Decimal,
    pub mark_price: Decimal,
}

impl FundingRateCalculator {
    /// Calculate funding cost for a position
    pub fn calculate_funding_cost(
        &self,
        symbol: &str,
        position_value: Decimal,
        holding_hours: f64,
    ) -> Result<Decimal> {
        let rate = self.current_rates
            .get(symbol)
            .context("Symbol not found in funding rates")?;
        
        // Funding payments occur every interval (usually 8 hours)
        let num_payments = (holding_hours / rate.interval_hours as f64).floor() as i32;
        
        // Total funding = Position Value * Rate * Number of Payments
        // Positive rate means longs pay shorts
        // Negative rate means shorts pay longs
        Ok(position_value * rate.rate * Decimal::from(num_payments))
    }
    
    /// Estimate average funding rate from historical data
    pub fn estimate_average_funding(&self, symbol: &str, lookback_days: u32) -> Decimal {
        let cutoff_time = chrono::Utc::now().timestamp() - (lookback_days as i64 * 86400);
        
        let rates: Vec<Decimal> = self.historical_rates
            .iter()
            .filter(|h| h.symbol == symbol && h.timestamp > cutoff_time)
            .map(|h| h.rate)
            .collect();
        
        if rates.is_empty() {
            return Decimal::ZERO;
        }
        
        let sum: Decimal = rates.iter().sum();
        sum / Decimal::from(rates.len())
    }
}

/// Slippage and market impact model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Linear impact coefficient (temporary impact)
    pub linear_impact: Decimal,
    
    /// Square-root impact coefficient (permanent impact)
    pub sqrt_impact: Decimal,
    
    /// Participation rate threshold
    pub max_participation_rate: Decimal,
}

impl SlippageModel {
    /// Calculate expected slippage using Almgren-Chriss model
    /// References: "Optimal Trading with Stochastic Liquidity and Volatility"
    pub fn calculate_slippage(
        &self,
        order_size: Decimal,
        avg_daily_volume: Decimal,
        volatility: Decimal,
        urgency: Decimal,  // 0 = patient, 1 = aggressive
    ) -> Decimal {
        // Participation rate = Order Size / ADV
        let participation = order_size / avg_daily_volume;
        
        // Warn if participation too high
        if participation > self.max_participation_rate {
            log::warn!(
                "High participation rate: {:.2}% of ADV",
                participation * dec!(100)
            );
        }
        
        // Temporary impact (linear in urgency)
        let temp_impact = self.linear_impact * volatility * participation * urgency;
        
        // Permanent impact (square-root of participation)
        let perm_impact = self.sqrt_impact * volatility * participation.sqrt();
        
        // Total expected slippage
        temp_impact + perm_impact
    }
    
    /// Estimate slippage for different order types
    pub fn estimate_by_order_type(
        &self,
        order_type: &str,
        order_size: Decimal,
        spread: Decimal,
        book_depth: Decimal,
    ) -> Decimal {
        match order_type {
            "market" => {
                // Market orders cross the spread + potential book walking
                let spread_cost = spread / dec!(2);
                let depth_ratio = order_size / book_depth;
                let walk_cost = if depth_ratio > dec!(1) {
                    // Walking the book adds significant cost
                    spread * depth_ratio.sqrt()
                } else {
                    Decimal::ZERO
                };
                spread_cost + walk_cost
            },
            "limit" => {
                // Limit orders may get adverse selection
                // Estimate 25% chance of adverse price movement
                spread * dec!(0.25) * dec!(0.5)
            },
            "post_only" => {
                // Post-only avoids crossing spread but has opportunity cost
                Decimal::ZERO  // No direct slippage, but may not fill
            },
            _ => spread / dec!(2),  // Default to half spread
        }
    }
}

/// Spread cost estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadCostEstimator {
    /// Current spreads by symbol
    pub current_spreads: HashMap<String, SpreadMetrics>,
    
    /// Historical spread statistics
    pub spread_stats: HashMap<String, SpreadStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadMetrics {
    pub bid: Decimal,
    pub ask: Decimal,
    pub mid: Decimal,
    pub spread_bps: Decimal,  // Spread in basis points
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadStatistics {
    pub symbol: String,
    pub avg_spread_bps: Decimal,
    pub median_spread_bps: Decimal,
    pub p95_spread_bps: Decimal,
    pub volatility: Decimal,
}

impl SpreadCostEstimator {
    /// Calculate spread cost for an order
    pub fn calculate_spread_cost(
        &self,
        symbol: &str,
        order_size: Decimal,
        is_aggressive: bool,
    ) -> Result<Decimal> {
        let metrics = self.current_spreads
            .get(symbol)
            .context("Symbol not found in spread data")?;
        
        let spread_cost = if is_aggressive {
            // Aggressive orders cross the spread
            (metrics.ask - metrics.bid) / dec!(2)
        } else {
            // Passive orders may get mid price
            Decimal::ZERO
        };
        
        Ok(spread_cost * order_size)
    }
    
    /// Check if spread is abnormal (potential issue)
    pub fn is_spread_abnormal(&self, symbol: &str) -> bool {
        if let (Some(current), Some(stats)) = 
            (self.current_spreads.get(symbol), self.spread_stats.get(symbol)) {
            // Alert if spread > 95th percentile
            current.spread_bps > stats.p95_spread_bps
        } else {
            false
        }
    }
}

/// Historical cost tracking for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostHistory {
    pub trades: Vec<TradeCost>,
    pub daily_summaries: Vec<DailyCostSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeCost {
    pub trade_id: String,
    pub timestamp: i64,
    pub symbol: String,
    pub side: String,
    pub quantity: Decimal,
    pub price: Decimal,
    pub exchange_fee: Decimal,
    pub slippage: Decimal,
    pub spread_cost: Decimal,
    pub funding_cost: Decimal,
    pub total_cost: Decimal,
    pub cost_bps: Decimal,  // Total cost in basis points
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyCostSummary {
    pub date: String,
    pub total_volume: Decimal,
    pub total_fees: Decimal,
    pub total_slippage: Decimal,
    pub total_spread_cost: Decimal,
    pub total_funding: Decimal,
    pub avg_cost_bps: Decimal,
    pub tradecount: u32,
}

impl ComprehensiveCostModel {
    /// Create default model with standard exchange fees
    pub fn new() -> Self {
        let mut model = Self {
            exchange_fees: HashMap::new(),
            funding_calculator: FundingRateCalculator {
                current_rates: HashMap::new(),
                historical_rates: Vec::new(),
            },
            slippage_model: SlippageModel {
                linear_impact: dec!(0.1),    // 10% of volatility per 100% ADV
                sqrt_impact: dec!(0.5),      // 50% of volatility per sqrt(100% ADV)
                max_participation_rate: dec!(0.1),  // Max 10% of ADV
            },
            spread_cost_estimator: SpreadCostEstimator {
                current_spreads: HashMap::new(),
                spread_stats: HashMap::new(),
            },
            cost_history: CostHistory {
                trades: Vec::new(),
                daily_summaries: Vec::new(),
            },
        };
        
        // Add Binance fee structure
        model.add_binance_fees();
        // Add Coinbase fee structure
        model.add_coinbase_fees();
        
        model
    }
    
    /// Add Binance fee tiers
    fn add_binance_fees(&mut self) {
        let binance_spot = TieredFeeSchedule {
            tiers: vec![
                FeeTier {
                    min_volume: dec!(0),
                    max_volume: Some(dec!(1_000_000)),
                    maker_fee: dec!(0.001),   // 0.10%
                    taker_fee: dec!(0.001),   // 0.10%
                },
                FeeTier {
                    min_volume: dec!(1_000_000),
                    max_volume: Some(dec!(5_000_000)),
                    maker_fee: dec!(0.0009),  // 0.09%
                    taker_fee: dec!(0.001),   // 0.10%
                },
                FeeTier {
                    min_volume: dec!(5_000_000),
                    max_volume: Some(dec!(10_000_000)),
                    maker_fee: dec!(0.0008),  // 0.08%
                    taker_fee: dec!(0.001),   // 0.10%
                },
                // Add more tiers as needed
            ],
        };
        
        let binance = ExchangeFeeStructure {
            exchange: "binance".to_string(),
            spot_fees: binance_spot.clone(),
            futures_fees: binance_spot,  // Similar structure for futures
            withdrawal_fees: HashMap::new(),
            min_order_sizes: HashMap::new(),
        };
        
        self.exchange_fees.insert("binance".to_string(), binance);
    }
    
    /// Add Coinbase fee tiers
    fn add_coinbase_fees(&mut self) {
        let coinbase_spot = TieredFeeSchedule {
            tiers: vec![
                FeeTier {
                    min_volume: dec!(0),
                    max_volume: Some(dec!(10_000_000)),
                    maker_fee: dec!(0.004),   // 0.40%
                    taker_fee: dec!(0.006),   // 0.60%
                },
                FeeTier {
                    min_volume: dec!(10_000_000),
                    max_volume: Some(dec!(50_000_000)),
                    maker_fee: dec!(0.0035),  // 0.35%
                    taker_fee: dec!(0.0045),  // 0.45%
                },
                // Add more tiers
            ],
        };
        
        let coinbase = ExchangeFeeStructure {
            exchange: "coinbase".to_string(),
            spot_fees: coinbase_spot.clone(),
            futures_fees: coinbase_spot,
            withdrawal_fees: HashMap::new(),
            min_order_sizes: HashMap::new(),
        };
        
        self.exchange_fees.insert("coinbase".to_string(), coinbase);
    }
    
    /// Calculate total cost for a trade (Casey's comprehensive calculation)
    pub fn calculate_total_cost(
        &self,
        exchange: &str,
        symbol: &str,
        side: &str,
        quantity: Decimal,
        price: Decimal,
        order_type: &str,
        volume_30d: Decimal,
        avg_daily_volume: Decimal,
        volatility: Decimal,
        holding_hours: Option<f64>,
    ) -> Result<TradeCost> {
        // 1. Exchange fees
        let exchange_fee = self.calculate_exchange_fee(
            exchange,
            quantity * price,
            order_type,
            volume_30d,
        )?;
        
        // 2. Slippage
        let urgency = if order_type == "market" { dec!(1) } else { dec!(0.3) };
        let slippage = self.slippage_model.calculate_slippage(
            quantity * price,
            avg_daily_volume,
            volatility,
            urgency,
        );
        
        // 3. Spread cost
        let is_aggressive = order_type == "market";
        let spread_cost = self.spread_cost_estimator
            .calculate_spread_cost(_symbol, quantity, is_aggressive)?;
        
        // 4. Funding cost (for perpetuals)
        let funding_cost = if let Some(hours) = holding_hours {
            self.funding_calculator
                .calculate_funding_cost(_symbol, quantity * price, hours)
                .unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };
        
        // Total cost
        let total_cost = exchange_fee + slippage + spread_cost + funding_cost.abs();
        let cost_bps = (total_cost / (quantity * price)) * dec!(10000);
        
        Ok(TradeCost {
            trade_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity,
            price,
            exchange_fee,
            slippage,
            spread_cost,
            funding_cost,
            total_cost,
            cost_bps,
        })
    }
    
    /// Calculate exchange fee based on order type and volume tier
    fn calculate_exchange_fee(
        &self,
        exchange: &str,
        notional: Decimal,
        order_type: &str,
        volume_30d: Decimal,
    ) -> Result<Decimal> {
        let fee_structure = self.exchange_fees
            .get(exchange)
            .context("Exchange not found")?;
        
        let (_maker_fee, taker_fee) = fee_structure.spot_fees.get_fees(volume_30d);
        
        let fee_rate = match order_type {
            "market" => taker_fee,
            "limit" | "post_only" => maker_fee,
            _ => taker_fee,  // Default to taker
        };
        
        Ok(notional * fee_rate)
    }
    
    /// Generate monthly cost report (Sophia's requirement)
    pub fn generate_monthly_report(&self, month: &str) -> MonthlyCostReport {
        let trades_in_month: Vec<&TradeCost> = self.cost_history.trades
            .iter()
            .filter(|t| {
                let trade_date = chrono::NaiveDateTime::from_timestamp_opt(t.timestamp, 0)
                    .map(|dt| dt.format("%Y-%m").to_string())
                    .unwrap_or_default();
                trade_date == month
            })
            .collect();
        
        let total_volume: Decimal = trades_in_month.iter()
            .map(|t| t.quantity * t.price)
            .sum();
        
        let total_fees: Decimal = trades_in_month.iter()
            .map(|t| t.exchange_fee)
            .sum();
        
        let total_slippage: Decimal = trades_in_month.iter()
            .map(|t| t.slippage)
            .sum();
        
        let total_spread: Decimal = trades_in_month.iter()
            .map(|t| t.spread_cost)
            .sum();
        
        let total_funding: Decimal = trades_in_month.iter()
            .map(|t| t.funding_cost)
            .sum();
        
        let total_cost = total_fees + total_slippage + total_spread + total_funding.abs();
        
        MonthlyCostReport {
            month: month.to_string(),
            tradecount: trades_in_month.len() as u32,
            total_volume,
            total_fees,
            total_slippage,
            total_spread_cost: total_spread,
            total_funding,
            total_cost,
            avg_cost_bps: if total_volume > Decimal::ZERO {
                (total_cost / total_volume) * dec!(10000)
            } else {
                Decimal::ZERO
            },
            cost_breakdown: CostBreakdown {
                fees_pct: (total_fees / total_cost * dec!(100)),
                slippage_pct: (total_slippage / total_cost * dec!(100)),
                spread_pct: (total_spread / total_cost * dec!(100)),
                funding_pct: (total_funding.abs() / total_cost * dec!(100)),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyCostReport {
    pub month: String,
    pub tradecount: u32,
    pub total_volume: Decimal,
    pub total_fees: Decimal,
    pub total_slippage: Decimal,
    pub total_spread_cost: Decimal,
    pub total_funding: Decimal,
    pub total_cost: Decimal,
    pub avg_cost_bps: Decimal,
    pub cost_breakdown: CostBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub fees_pct: Decimal,
    pub slippage_pct: Decimal,
    pub spread_pct: Decimal,
    pub funding_pct: Decimal,
}

// ============================================================================
// TESTS - Riley's comprehensive validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binance_fee_tiers() {
        let model = ComprehensiveCostModel::new();
        let binance = model.exchange_fees.get("binance").unwrap();
        
        // Test tier 1 (< $1M volume)
        let (_maker, taker) = binance.spot_fees.get_fees(dec!(500_000));
        assert_eq!(_maker, dec!(0.001));  // 10bps
        assert_eq!(_taker, dec!(0.001));  // 10bps
        
        // Test tier 2 ($1M - $5M)
        let (_maker, taker) = binance.spot_fees.get_fees(dec!(2_000_000));
        assert_eq!(_maker, dec!(0.0009)); // 9bps
        assert_eq!(_taker, dec!(0.001));  // 10bps
    }
    
    #[test]
    fn test_slippage_calculation() {
        let model = SlippageModel {
            linear_impact: dec!(0.1),
            sqrt_impact: dec!(0.5),
            max_participation_rate: dec!(0.1),
        };
        
        // Test small order (1% of ADV)
        let slippage = model.calculate_slippage(
            dec!(10_000),    // $10k order
            dec!(1_000_000), // $1M ADV
            dec!(0.02),      // 2% volatility
            dec!(0.5),       // Medium urgency
        );
        
        // Should be small for 1% participation
        assert!(slippage < dec!(0.001)); // Less than 10bps
        
        // Test large order (10% of ADV)
        let slippage = model.calculate_slippage(
            dec!(100_000),   // $100k order
            dec!(1_000_000), // $1M ADV
            dec!(0.02),      // 2% volatility
            dec!(1.0),       // High urgency
        );
        
        // Should be significant for 10% participation
        assert!(slippage > dec!(0.002)); // More than 20bps
    }
    
    #[test]
    fn test_total_cost_calculation() {
        let model = ComprehensiveCostModel::new();
        
        let cost = model.calculate_total_cost(
            "binance",
            "BTC/USDT",
            "buy",
            dec!(1),          // 1 BTC
            dec!(50_000),     // $50k per BTC
            "market",
            dec!(500_000),    // $500k 30-day volume
            dec!(10_000_000), // $10M ADV
            dec!(0.02),       // 2% volatility
            Some(24.0),       // Hold for 24 hours
        ).unwrap();
        
        // Check all components are calculated
        assert!(cost.exchange_fee > Decimal::ZERO);
        assert!(cost.slippage >= Decimal::ZERO);
        assert!(cost.total_cost > cost.exchange_fee);
        
        // Total cost should be reasonable (< 1% for liquid market)
        assert!(cost.cost_bps < dec!(100)); // Less than 100bps
    }
    
    #[test]
    fn test_monthly_cost_tracking() {
        let mut model = ComprehensiveCostModel::new();
        
        // Add some sample trades
        for i in 0..100 {
            let trade = TradeCost {
                trade_id: format!("trade_{}", i),
                timestamp: chrono::Utc::now().timestamp(),
                symbol: "BTC/USDT".to_string(),
                side: if i % 2 == 0 { "buy" } else { "sell" }.to_string(),
                quantity: dec!(0.1),
                price: dec!(50_000),
                exchange_fee: dec!(5),      // $5 fee
                slippage: dec!(10),         // $10 slippage
                spread_cost: dec!(2.5),     // $2.50 spread
                funding_cost: dec!(-1),     // $1 funding income
                total_cost: dec!(16.5),
                cost_bps: dec!(33),         // 33bps
            };
            model.cost_history.trades.push(trade);
        }
        
        let report = model.generate_monthly_report(
            &chrono::Utc::now().format("%Y-%m").to_string()
        );
        
        assert_eq!(report.tradecount, 100);
        assert_eq!(report.total_volume, dec!(500_000)); // 100 * 0.1 * 50000
        assert_eq!(report.total_fees, dec!(500));       // 100 * $5
        assert_eq!(report.total_slippage, dec!(1000));  // 100 * $10
        
        // Average cost should match
        let expected_avg = (dec!(16.5) * dec!(100)) / dec!(500_000) * dec!(10000);
        assert_eq!(report.avg_cost_bps, expected_avg);
    }
}