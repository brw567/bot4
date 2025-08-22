// Fee Optimization Engine
use rust_decimal::prelude::ToPrimitive;
// Team: Avery (Data) + Casey (Exchange) + Alex (Strategy)
// CRITICAL: Minimize trading costs through intelligent fee management
// References:
// - "Optimal Execution with Limit and Market Orders" - Obizhaeva & Wang (2013)
// - "Trading Fees and Efficiency in Limit Order Markets" - Colliard & Foucault (2012)
// - Exchange fee schedules: Binance, Coinbase, Kraken, FTX (historical)

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Fee structure for an exchange
#[derive(Debug, Clone)]
pub struct ExchangeFeeStructure {
    pub exchange_name: String,
    pub base_maker_fee_bps: i32,        // Base maker fee in basis points
    pub base_taker_fee_bps: i32,        // Base taker fee in basis points
    pub volume_tiers: Vec<VolumeTier>,  // Volume-based discounts
    pub vip_tiers: Vec<VipTier>,        // VIP program tiers
    pub rebate_available: bool,         // Maker rebates available
    pub rebate_bps: i32,               // Rebate amount if available
    pub special_programs: Vec<SpecialProgram>,
    pub native_token_discount: Option<TokenDiscount>,
    pub referral_discount_pct: f64,     // Referral program discount
    pub updated_at: DateTime<Utc>,
}

/// Volume-based fee tier
#[derive(Debug, Clone)]
pub struct VolumeTier {
    pub min_volume_usd: Decimal,
    pub maker_fee_bps: i32,
    pub taker_fee_bps: i32,
}

/// VIP tier with special benefits
#[derive(Debug, Clone)]
pub struct VipTier {
    pub level: u8,
    pub min_volume_30d: Decimal,
    pub min_balance: Decimal,
    pub maker_fee_bps: i32,
    pub taker_fee_bps: i32,
    pub benefits: Vec<String>,
}

/// Special fee programs
#[derive(Debug, Clone)]
pub enum SpecialProgram {
    MarketMakerProgram {
        requirements: MarketMakerRequirements,
        maker_rebate_bps: i32,
        taker_fee_bps: i32,
    },
    LiquidityProvider {
        min_depth_bps: i32,
        min_uptime_pct: f64,
        fee_reduction_pct: f64,
    },
    StablecoinPairs {
        fee_reduction_pct: f64,
        eligible_pairs: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct MarketMakerRequirements {
    pub min_volume_30d: Decimal,
    pub min_orders_per_day: u64,
    pub max_spread_bps: i32,
    pub min_uptime_pct: f64,
}

/// Native token discount program
#[derive(Debug, Clone)]
pub struct TokenDiscount {
    pub token_symbol: String,
    pub discount_pct: f64,
    pub min_holding: Decimal,
    pub payment_in_token: bool,
}

/// Trading statistics for fee calculation
#[derive(Debug, Clone, Default)]
pub struct TradingStatistics {
    pub volume_30d: Decimal,
    pub volume_mtd: Decimal,           // Month to date
    pub maker_volume_30d: Decimal,
    pub taker_volume_30d: Decimal,
    pub orders_per_day_avg: f64,
    pub native_token_balance: Decimal,
    pub vip_level: Option<u8>,
    pub is_market_maker: bool,
}

/// Fee calculation result
#[derive(Debug, Clone)]
pub struct FeeCalculation {
    pub exchange: String,
    pub order_type: OrderType,
    pub base_fee_bps: i32,
    pub effective_fee_bps: i32,
    pub fee_amount: Decimal,
    pub rebate_amount: Decimal,
    pub net_fee: Decimal,
    pub savings_from_optimization: Decimal,
    pub applied_discounts: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Maker,
    Taker,
}

/// Fee optimization strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    MinimizeFees,           // Absolute lowest fees
    MaximizeRebates,        // Focus on rebates
    BalanceSpeedAndCost,    // Balance execution speed vs cost
    QualifyForNextTier,     // Trade to reach better tier
    MaintainMarketMaker,    // Maintain MM status
}

/// Fee Optimization Engine
/// Avery: "Every basis point counts when you're trading millions"
pub struct FeeOptimizationEngine {
    /// Exchange fee structures
    exchange_fees: Arc<RwLock<HashMap<String, ExchangeFeeStructure>>>,
    
    /// Our trading statistics per exchange
    our_stats: Arc<RwLock<HashMap<String, TradingStatistics>>>,
    
    /// Current optimization strategy
    strategy: Arc<RwLock<OptimizationStrategy>>,
    
    /// Fee history for analysis
    fee_history: Arc<RwLock<Vec<FeeCalculation>>>,
    
    /// Cached optimal routes
    optimal_routes_cache: Arc<RwLock<HashMap<String, OptimalRoute>>>,
    
    /// Configuration
    config: FeeOptimizerConfig,
}

/// Optimal routing decision
#[derive(Debug, Clone)]
struct OptimalRoute {
    pub symbol: String,
    pub exchanges_ranked: Vec<ExchangeRanking>,
    pub recommended_type: OrderType,
    pub estimated_savings_bps: i32,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ExchangeRanking {
    pub exchange: String,
    pub score: f64,
    pub effective_fee_bps: i32,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
}

/// Configuration for fee optimizer
#[derive(Debug, Clone)]
pub struct FeeOptimizerConfig {
    pub max_acceptable_fee_bps: i32,          // Maximum fee we'll pay
    pub min_rebate_to_provide_liquidity: i32, // Min rebate to act as maker
    pub tier_qualification_window: i32,        // Days to qualify for tier
    pub cache_duration_seconds: i64,          // How long to cache routes
    pub consider_slippage: bool,              // Include slippage in calculations
    pub slippage_estimate_bps: i32,          // Expected slippage
}

impl Default for FeeOptimizerConfig {
    fn default() -> Self {
        Self {
            max_acceptable_fee_bps: 30,        // 0.3% max
            min_rebate_to_provide_liquidity: 2, // 0.02% min rebate
            tier_qualification_window: 30,      // 30 days
            cache_duration_seconds: 60,         // 1 minute cache
            consider_slippage: true,
            slippage_estimate_bps: 5,          // 0.05% slippage
        }
    }
}

impl FeeOptimizationEngine {
    pub fn new(config: FeeOptimizerConfig) -> Self {
        Self {
            exchange_fees: Arc::new(RwLock::new(HashMap::new())),
            our_stats: Arc::new(RwLock::new(HashMap::new())),
            strategy: Arc::new(RwLock::new(OptimizationStrategy::MinimizeFees)),
            fee_history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            optimal_routes_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Initialize exchange fee structures
    pub fn initialize_exchanges(&self) {
        let mut fees = self.exchange_fees.write();
        
        // Binance fee structure
        fees.insert("binance".to_string(), ExchangeFeeStructure {
            exchange_name: "binance".to_string(),
            base_maker_fee_bps: 10,  // 0.10%
            base_taker_fee_bps: 10,  // 0.10%
            volume_tiers: vec![
                VolumeTier { min_volume_usd: dec!(0), maker_fee_bps: 10, taker_fee_bps: 10 },
                VolumeTier { min_volume_usd: dec!(1000000), maker_fee_bps: 9, taker_fee_bps: 10 },
                VolumeTier { min_volume_usd: dec!(5000000), maker_fee_bps: 8, taker_fee_bps: 10 },
                VolumeTier { min_volume_usd: dec!(10000000), maker_fee_bps: 7, taker_fee_bps: 9 },
                VolumeTier { min_volume_usd: dec!(50000000), maker_fee_bps: 4, taker_fee_bps: 7 },
            ],
            vip_tiers: vec![
                VipTier {
                    level: 1,
                    min_volume_30d: dec!(1000000),
                    min_balance: dec!(25),  // BNB
                    maker_fee_bps: 9,
                    taker_fee_bps: 9,
                    benefits: vec!["Priority support".to_string()],
                },
            ],
            rebate_available: false,
            rebate_bps: 0,
            special_programs: vec![],
            native_token_discount: Some(TokenDiscount {
                token_symbol: "BNB".to_string(),
                discount_pct: 25.0,
                min_holding: dec!(1),
                payment_in_token: true,
            }),
            referral_discount_pct: 20.0,
            updated_at: Utc::now(),
        });
        
        // Coinbase fee structure
        fees.insert("coinbase".to_string(), ExchangeFeeStructure {
            exchange_name: "coinbase".to_string(),
            base_maker_fee_bps: 50,  // 0.50%
            base_taker_fee_bps: 50,  // 0.50%
            volume_tiers: vec![
                VolumeTier { min_volume_usd: dec!(0), maker_fee_bps: 50, taker_fee_bps: 50 },
                VolumeTier { min_volume_usd: dec!(10000), maker_fee_bps: 35, taker_fee_bps: 45 },
                VolumeTier { min_volume_usd: dec!(100000), maker_fee_bps: 15, taker_fee_bps: 25 },
                VolumeTier { min_volume_usd: dec!(1000000), maker_fee_bps: 10, taker_fee_bps: 20 },
                VolumeTier { min_volume_usd: dec!(100000000), maker_fee_bps: 0, taker_fee_bps: 10 },
            ],
            vip_tiers: vec![],
            rebate_available: true,
            rebate_bps: 1,  // 0.01% maker rebate at top tier
            special_programs: vec![
                SpecialProgram::StablecoinPairs {
                    fee_reduction_pct: 100.0,  // No fees on stablecoin pairs
                    eligible_pairs: vec!["USDC-USD".to_string(), "DAI-USDC".to_string()],
                },
            ],
            native_token_discount: None,
            referral_discount_pct: 0.0,
            updated_at: Utc::now(),
        });
        
        // Kraken fee structure
        fees.insert("kraken".to_string(), ExchangeFeeStructure {
            exchange_name: "kraken".to_string(),
            base_maker_fee_bps: 16,  // 0.16%
            base_taker_fee_bps: 26,  // 0.26%
            volume_tiers: vec![
                VolumeTier { min_volume_usd: dec!(0), maker_fee_bps: 16, taker_fee_bps: 26 },
                VolumeTier { min_volume_usd: dec!(50000), maker_fee_bps: 14, taker_fee_bps: 24 },
                VolumeTier { min_volume_usd: dec!(100000), maker_fee_bps: 12, taker_fee_bps: 22 },
                VolumeTier { min_volume_usd: dec!(250000), maker_fee_bps: 10, taker_fee_bps: 20 },
                VolumeTier { min_volume_usd: dec!(500000), maker_fee_bps: 8, taker_fee_bps: 18 },
            ],
            vip_tiers: vec![],
            rebate_available: false,
            rebate_bps: 0,
            special_programs: vec![],
            native_token_discount: None,
            referral_discount_pct: 20.0,
            updated_at: Utc::now(),
        });
    }
    
    /// Update trading statistics for an exchange
    pub fn update_statistics(&self, exchange: &str, stats: TradingStatistics) {
        self.our_stats.write().insert(exchange.to_string(), stats);
        
        // Invalidate cache when stats change
        self.optimal_routes_cache.write().clear();
    }
    
    /// Calculate fee for an order
    pub fn calculate_fee(
        &self,
        exchange: &str,
        order_type: OrderType,
        order_value: Decimal,
    ) -> Result<FeeCalculation, String> {
        let fees = self.exchange_fees.read();
        let fee_structure = fees.get(exchange)
            .ok_or_else(|| format!("Unknown exchange: {}", exchange))?;
        
        let stats = self.our_stats.read()
            .get(exchange)
            .cloned()
            .unwrap_or_default();
        
        // Determine base fee
        let base_fee_bps = match order_type {
            OrderType::Maker => fee_structure.base_maker_fee_bps,
            OrderType::Taker => fee_structure.base_taker_fee_bps,
        };
        
        // Apply volume tier discount
        let tier_fee_bps = self.get_tier_fee(fee_structure, &stats, order_type);
        
        // Apply VIP discount if applicable
        let vip_fee_bps = if let Some(vip_level) = stats.vip_level {
            self.get_vip_fee(fee_structure, vip_level, order_type).unwrap_or(tier_fee_bps)
        } else {
            tier_fee_bps
        };
        
        // Apply native token discount
        let mut effective_fee_bps = vip_fee_bps;
        let mut applied_discounts = Vec::new();
        
        if let Some(token_discount) = &fee_structure.native_token_discount {
            if stats.native_token_balance >= token_discount.min_holding {
                let discount = (vip_fee_bps as f64 * token_discount.discount_pct / 100.0) as i32;
                effective_fee_bps -= discount;
                applied_discounts.push(format!(
                    "{} discount: -{}%",
                    token_discount.token_symbol,
                    token_discount.discount_pct
                ));
            }
        }
        
        // Apply referral discount
        if fee_structure.referral_discount_pct > 0.0 {
            let discount = (effective_fee_bps as f64 * 
                          fee_structure.referral_discount_pct / 100.0) as i32;
            effective_fee_bps -= discount;
            applied_discounts.push(format!(
                "Referral discount: -{}%",
                fee_structure.referral_discount_pct
            ));
        }
        
        // Check for special programs
        for program in &fee_structure.special_programs {
            match program {
                SpecialProgram::MarketMakerProgram {  maker_rebate_bps, .. } => {
                    if stats.is_market_maker && order_type == OrderType::Maker {
                        effective_fee_bps = -*maker_rebate_bps; // Negative = rebate
                        applied_discounts.push("Market maker rebate".to_string());
                    }
                }
                SpecialProgram::StablecoinPairs {  .. } => {
                    // Would need to check if this is a stablecoin pair
                    // Simplified for this implementation
                }
                _ => {}
            }
        }
        
        // Calculate actual amounts
        let fee_amount = if effective_fee_bps > 0 {
            order_value * Decimal::from(effective_fee_bps) / dec!(10000)
        } else {
            Decimal::ZERO
        };
        
        let rebate_amount = if effective_fee_bps < 0 {
            order_value * Decimal::from(-effective_fee_bps) / dec!(10000)
        } else {
            Decimal::ZERO
        };
        
        let net_fee = fee_amount - rebate_amount;
        
        // Calculate savings from optimization
        let base_fee_amount = order_value * Decimal::from(base_fee_bps) / dec!(10000);
        let savings = base_fee_amount - net_fee;
        
        let calculation = FeeCalculation {
            exchange: exchange.to_string(),
            order_type,
            base_fee_bps,
            effective_fee_bps,
            fee_amount,
            rebate_amount,
            net_fee,
            savings_from_optimization: savings,
            applied_discounts,
        };
        
        // Store in history
        self.fee_history.write().push(calculation.clone());
        
        Ok(calculation)
    }
    
    /// Get fee based on volume tier
    fn get_tier_fee(
        &self,
        structure: &ExchangeFeeStructure,
        stats: &TradingStatistics,
        order_type: OrderType,
    ) -> i32 {
        for tier in structure.volume_tiers.iter().rev() {
            if stats.volume_30d >= tier.min_volume_usd {
                return match order_type {
                    OrderType::Maker => tier.maker_fee_bps,
                    OrderType::Taker => tier.taker_fee_bps,
                };
            }
        }
        
        // Default to base fee
        match order_type {
            OrderType::Maker => structure.base_maker_fee_bps,
            OrderType::Taker => structure.base_taker_fee_bps,
        }
    }
    
    /// Get VIP tier fee
    fn get_vip_fee(
        &self,
        structure: &ExchangeFeeStructure,
        vip_level: u8,
        order_type: OrderType,
    ) -> Option<i32> {
        structure.vip_tiers.iter()
            .find(|t| t.level == vip_level)
            .map(|t| match order_type {
                OrderType::Maker => t.maker_fee_bps,
                OrderType::Taker => t.taker_fee_bps,
            })
    }
    
    /// Find optimal exchange for an order
    pub fn find_optimal_exchange(
        &self,
        symbol: &str,
        order_value: Decimal,
        urgency: OrderUrgency,
    ) -> Result<OptimalRoute, String> {
        // Check cache first
        if let Some(cached) = self.get_cached_route(symbol) {
            return Ok(cached);
        }
        
        let mut rankings = Vec::new();
        let strategy = *self.strategy.read();
        
        for (exchange_name, fee_structure) in self.exchange_fees.read().iter() {
            // Calculate fees for both maker and taker
            let maker_calc = self.calculate_fee(exchange_name, OrderType::Maker, order_value)?;
            let taker_calc = self.calculate_fee(exchange_name, OrderType::Taker, order_value)?;
            
            // Score based on strategy
            let (score, recommended_type, pros, cons) = match strategy {
                OptimizationStrategy::MinimizeFees => {
                    if maker_calc.net_fee <= taker_calc.net_fee {
                        (
                            100.0 - maker_calc.effective_fee_bps as f64,
                            OrderType::Maker,
                            vec![format!("Lowest fee: {} bps", maker_calc.effective_fee_bps)],
                            vec![],
                        )
                    } else {
                        (
                            100.0 - taker_calc.effective_fee_bps as f64,
                            OrderType::Taker,
                            vec![format!("Lower taker fee: {} bps", taker_calc.effective_fee_bps)],
                            vec!["Requires immediate execution".to_string()],
                        )
                    }
                }
                OptimizationStrategy::MaximizeRebates => {
                    if maker_calc.rebate_amount > Decimal::ZERO {
                        (
                            100.0 + maker_calc.rebate_amount.to_f64().unwrap_or(0.0),
                            OrderType::Maker,
                            vec![format!("Rebate available: ${}", maker_calc.rebate_amount)],
                            vec!["Requires providing liquidity".to_string()],
                        )
                    } else {
                        (
                            50.0 - maker_calc.effective_fee_bps as f64,
                            OrderType::Maker,
                            vec!["No rebate but lower maker fee".to_string()],
                            vec!["No rebate available".to_string()],
                        )
                    }
                }
                OptimizationStrategy::BalanceSpeedAndCost => {
                    let speed_weight = match urgency {
                        OrderUrgency::Immediate => 0.8,
                        OrderUrgency::Normal => 0.5,
                        OrderUrgency::Patient => 0.2,
                    };
                    
                    let cost_weight = 1.0 - speed_weight;
                    let taker_score = 100.0 * speed_weight - taker_calc.effective_fee_bps as f64 * cost_weight;
                    let maker_score = 50.0 * speed_weight - maker_calc.effective_fee_bps as f64 * cost_weight;
                    
                    if taker_score > maker_score {
                        (
                            taker_score,
                            OrderType::Taker,
                            vec!["Balanced for speed".to_string()],
                            vec![format!("Higher fee: {} bps", taker_calc.effective_fee_bps)],
                        )
                    } else {
                        (
                            maker_score,
                            OrderType::Maker,
                            vec!["Balanced for cost".to_string()],
                            vec!["Slower execution".to_string()],
                        )
                    }
                }
                _ => {
                    // Default to minimize fees
                    (
                        100.0 - maker_calc.effective_fee_bps as f64,
                        OrderType::Maker,
                        vec!["Default strategy".to_string()],
                        vec![],
                    )
                }
            };
            
            rankings.push(ExchangeRanking {
                exchange: exchange_name.clone(),
                score,
                effective_fee_bps: if recommended_type == OrderType::Maker {
                    maker_calc.effective_fee_bps
                } else {
                    taker_calc.effective_fee_bps
                },
                pros,
                cons,
            });
        }
        
        // Sort by score (highest first)
        rankings.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        if rankings.is_empty() {
            return Err("No exchanges available".to_string());
        }
        
        let best = &rankings[0];
        let worst = rankings.last()
            .ok_or_else(|| "No exchanges available for comparison".to_string())?;
        let estimated_savings_bps = worst.effective_fee_bps - best.effective_fee_bps;
        
        let route = OptimalRoute {
            symbol: symbol.to_string(),
            exchanges_ranked: rankings,
            recommended_type: OrderType::Maker, // Simplified
            estimated_savings_bps,
            calculated_at: Utc::now(),
        };
        
        // Cache the route
        self.optimal_routes_cache.write().insert(symbol.to_string(), route.clone());
        
        Ok(route)
    }
    
    /// Get cached route if still valid
    fn get_cached_route(&self, symbol: &str) -> Option<OptimalRoute> {
        let cache = self.optimal_routes_cache.read();
        cache.get(symbol).and_then(|route| {
            let age = (Utc::now() - route.calculated_at).num_seconds();
            if age < self.config.cache_duration_seconds {
                Some(route.clone())
            } else {
                None
            }
        })
    }
    
    /// Get recommendations for tier qualification
    pub fn get_tier_recommendations(&self) -> Vec<TierRecommendation> {
        let mut recommendations = Vec::new();
        
        for (exchange, stats) in self.our_stats.read().iter() {
            if let Some(fee_structure) = self.exchange_fees.read().get(exchange) {
                // Find next tier
                for tier in &fee_structure.volume_tiers {
                    if stats.volume_30d < tier.min_volume_usd {
                        let volume_needed = tier.min_volume_usd - stats.volume_30d;
                        let current_fee = self.get_tier_fee(fee_structure, stats, OrderType::Maker);
                        let savings_bps = current_fee - tier.maker_fee_bps;
                        
                        if savings_bps > 0 {
                            recommendations.push(TierRecommendation {
                                exchange: exchange.clone(),
                                current_volume: stats.volume_30d,
                                target_volume: tier.min_volume_usd,
                                volume_needed,
                                days_remaining: self.config.tier_qualification_window,
                                daily_volume_required: volume_needed / 
                                    Decimal::from(self.config.tier_qualification_window),
                                savings_per_trade_bps: savings_bps,
                                estimated_monthly_savings: self.estimate_monthly_savings(
                                    stats.volume_mtd,
                                    savings_bps
                                ),
                            });
                        }
                        break; // Only show next tier
                    }
                }
            }
        }
        
        recommendations.sort_by_key(|r| -(r.estimated_monthly_savings.to_f64().unwrap_or(0.0) * 100.0) as i64);
        recommendations
    }
    
    /// Estimate monthly savings from fee reduction
    fn estimate_monthly_savings(&self, monthly_volume: Decimal, savings_bps: i32) -> Decimal {
        monthly_volume * Decimal::from(savings_bps) / dec!(10000)
    }
    
    /// Get fee statistics
    pub fn get_statistics(&self) -> FeeStatistics {
        let history = self.fee_history.read();
        
        let total_fees: Decimal = history.iter().map(|f| f.net_fee).sum();
        let total_rebates: Decimal = history.iter().map(|f| f.rebate_amount).sum();
        let total_savings: Decimal = history.iter().map(|f| f.savings_from_optimization).sum();
        
        let avg_fee_bps = if !history.is_empty() {
            history.iter().map(|f| f.effective_fee_bps).sum::<i32>() / history.len() as i32
        } else {
            0
        };
        
        FeeStatistics {
            total_fees_paid: total_fees,
            total_rebates_earned: total_rebates,
            total_savings,
            avg_effective_fee_bps: avg_fee_bps,
            trades_analyzed: history.len(),
            best_exchange: self.find_best_exchange(),
            worst_exchange: self.find_worst_exchange(),
        }
    }
    
    /// Find best exchange by average fees
    fn find_best_exchange(&self) -> Option<String> {
        let history = self.fee_history.read();
        let mut exchange_fees: HashMap<String, (Decimal, usize)> = HashMap::new();
        
        for calc in history.iter() {
            let entry = exchange_fees.entry(calc.exchange.clone()).or_insert((Decimal::ZERO, 0));
            entry.0 += calc.net_fee;
            entry.1 += 1;
        }
        
        exchange_fees.iter()
            .map(|(ex, (total, count))| (ex.clone(), *total / Decimal::from(*count)))
            .min_by_key(|(_, avg)| (avg.to_f64().unwrap_or(0.0) * 10000.0) as i64)
            .map(|(ex, _)| ex)
    }
    
    /// Find worst exchange by average fees
    fn find_worst_exchange(&self) -> Option<String> {
        let history = self.fee_history.read();
        let mut exchange_fees: HashMap<String, (Decimal, usize)> = HashMap::new();
        
        for calc in history.iter() {
            let entry = exchange_fees.entry(calc.exchange.clone()).or_insert((Decimal::ZERO, 0));
            entry.0 += calc.net_fee;
            entry.1 += 1;
        }
        
        exchange_fees.iter()
            .map(|(ex, (total, count))| (ex.clone(), *total / Decimal::from(*count)))
            .max_by_key(|(_, avg)| (avg.to_f64().unwrap_or(0.0) * 10000.0) as i64)
            .map(|(ex, _)| ex)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderUrgency {
    Immediate,  // Must execute now
    Normal,     // Standard execution
    Patient,    // Can wait for better price
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierRecommendation {
    pub exchange: String,
    pub current_volume: Decimal,
    pub target_volume: Decimal,
    pub volume_needed: Decimal,
    pub days_remaining: i32,
    pub daily_volume_required: Decimal,
    pub savings_per_trade_bps: i32,
    pub estimated_monthly_savings: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStatistics {
    pub total_fees_paid: Decimal,
    pub total_rebates_earned: Decimal,
    pub total_savings: Decimal,
    pub avg_effective_fee_bps: i32,
    pub trades_analyzed: usize,
    pub best_exchange: Option<String>,
    pub worst_exchange: Option<String>,
}

// ============================================================================
// TESTS - Avery & Casey validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fee_calculation() {
        let engine = FeeOptimizationEngine::new(FeeOptimizerConfig::default());
        engine.initialize_exchanges();
        
        // Test basic fee calculation
        let fee = engine.calculate_fee("binance", OrderType::Maker, dec!(10000)).unwrap();
        assert_eq!(fee.base_fee_bps, 10);
        assert_eq!(fee.exchange, "binance");
        
        // Test with volume tier
        let mut stats = TradingStatistics::default();
        stats.volume_30d = dec!(5000000); // $5M volume
        engine.update_statistics("binance", stats);
        
        let fee = engine.calculate_fee("binance", OrderType::Maker, dec!(10000)).unwrap();
        assert!(fee.effective_fee_bps < 10); // Should have discount
    }
    
    #[test]
    fn test_optimal_exchange_selection() {
        let engine = FeeOptimizationEngine::new(FeeOptimizerConfig::default());
        engine.initialize_exchanges();
        
        // Find optimal exchange
        let route = engine.find_optimal_exchange(
            "BTCUSDT",
            dec!(10000),
            OrderUrgency::Normal
        ).unwrap();
        
        assert!(!route.exchanges_ranked.is_empty());
        assert!(route.estimated_savings_bps >= 0);
    }
    
    #[test]
    fn test_tier_recommendations() {
        let engine = FeeOptimizationEngine::new(FeeOptimizerConfig::default());
        engine.initialize_exchanges();
        
        // Add some volume stats
        let mut stats = TradingStatistics::default();
        stats.volume_30d = dec!(900000); // Just under $1M
        stats.volume_mtd = dec!(300000);
        engine.update_statistics("binance", stats);
        
        let recommendations = engine.get_tier_recommendations();
        assert!(!recommendations.is_empty());
        
        // Should recommend reaching $1M tier
        let binance_rec = recommendations.iter()
            .find(|r| r.exchange == "binance")
            .unwrap();
        assert_eq!(binance_rec.target_volume, dec!(1000000));
    }
}