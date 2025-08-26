// Fee Calculator with Exchange-Specific Tiers
// DEEP DIVE: Accurate fee modeling including VIP tiers and rebates
//
// References:
// - Binance VIP Fee Structure 2024
// - Coinbase Advanced Trade Fee Schedule
// - Kraken Fee Schedule
// - FTX (historical) Maker-Taker Model
// - NYSE/NASDAQ Fee and Rebate Schedules

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

use types::{Price, Quantity, Symbol, Exchange};

/// Fee structure for an exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    pub exchange: Exchange,
    pub base_maker_fee_bps: f64,
    pub base_taker_fee_bps: f64,
    pub volume_tiers: Vec<VolumeTier>,
    pub maker_rebate_available: bool,
    pub special_programs: Vec<SpecialProgram>,
    pub payment_discount: Option<PaymentDiscount>,
}

/// Volume-based fee tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeTier {
    pub tier_name: String,
    pub min_volume_usd: Decimal,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub maker_rebate_bps: Option<f64>,
}

/// Special fee programs (e.g., market maker programs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecialProgram {
    MarketMaker {
        name: String,
        maker_rebate_bps: f64,
        min_quote_time_pct: f64,
        min_spread_bps: f64,
    },
    VIP {
        level: u8,
        maker_discount_pct: f64,
        taker_discount_pct: f64,
        min_balance_usd: Decimal,
    },
    Referral {
        discount_pct: f64,
        kickback_pct: f64,
    },
}

/// Payment method discount (e.g., BNB on Binance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentDiscount {
    pub token_symbol: String,
    pub discount_pct: f64,
    pub requires_holding: bool,
    pub min_holding_amount: Option<Decimal>,
}

/// Maker/Taker fee structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MakerTakerFees {
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub maker_rebate_bps: Option<f64>,
    pub effective_date: DateTime<Utc>,
}

/// Volume discount calculation
#[derive(Debug, Clone)]
pub struct VolumeDiscount {
    pub thirty_day_volume_usd: Decimal,
    pub current_tier: TierLevel,
    pub next_tier_volume: Option<Decimal>,
    pub days_in_period: u32,
}

/// Tier level information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierLevel {
    pub level: u8,
    pub name: String,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub benefits: Vec<String>,
}

/// Exchange-specific fees database
#[derive(Debug, Clone)]
pub struct ExchangeFees {
    pub binance: FeeStructure,
    pub coinbase: FeeStructure,
    pub kraken: FeeStructure,
    pub okx: FeeStructure,
    pub bybit: FeeStructure,
}

impl ExchangeFees {
    pub fn new() -> Self {
        Self {
            binance: Self::create_binance_fees(),
            coinbase: Self::create_coinbase_fees(),
            kraken: Self::create_kraken_fees(),
            okx: Self::create_okx_fees(),
            bybit: Self::create_bybit_fees(),
        }
    }
    
    fn create_binance_fees() -> FeeStructure {
        FeeStructure {
            exchange: Exchange("Binance".to_string()),
            base_maker_fee_bps: 10.0,  // 0.10%
            base_taker_fee_bps: 10.0,  // 0.10%
            volume_tiers: vec![
                VolumeTier {
                    tier_name: "VIP 0".to_string(),
                    min_volume_usd: Decimal::ZERO,
                    maker_fee_bps: 10.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 1".to_string(),
                    min_volume_usd: Decimal::from(50_000_000),
                    maker_fee_bps: 9.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 2".to_string(),
                    min_volume_usd: Decimal::from(100_000_000),
                    maker_fee_bps: 8.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 3".to_string(),
                    min_volume_usd: Decimal::from(250_000_000),
                    maker_fee_bps: 7.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 4".to_string(),
                    min_volume_usd: Decimal::from(500_000_000),
                    maker_fee_bps: 6.5,
                    taker_fee_bps: 9.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 5".to_string(),
                    min_volume_usd: Decimal::from(1_000_000_000),
                    maker_fee_bps: 6.0,
                    taker_fee_bps: 8.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 6".to_string(),
                    min_volume_usd: Decimal::from(2_000_000_000),
                    maker_fee_bps: 5.5,
                    taker_fee_bps: 7.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 7".to_string(),
                    min_volume_usd: Decimal::from(4_000_000_000),
                    maker_fee_bps: 5.0,
                    taker_fee_bps: 6.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 8".to_string(),
                    min_volume_usd: Decimal::from(8_000_000_000),
                    maker_fee_bps: 4.0,
                    taker_fee_bps: 5.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 9".to_string(),
                    min_volume_usd: Decimal::from(15_000_000_000),
                    maker_fee_bps: 2.0,
                    taker_fee_bps: 4.0,
                    maker_rebate_bps: Some(0.5),  // Maker rebate at highest tier
                },
            ],
            maker_rebate_available: true,
            special_programs: vec![
                SpecialProgram::MarketMaker {
                    name: "Binance MM Program".to_string(),
                    maker_rebate_bps: 1.0,
                    min_quote_time_pct: 90.0,
                    min_spread_bps: 5.0,
                },
            ],
            payment_discount: Some(PaymentDiscount {
                token_symbol: "BNB".to_string(),
                discount_pct: 25.0,
                requires_holding: true,
                min_holding_amount: Some(Decimal::from(1)),
            }),
        }
    }
    
    fn create_coinbase_fees() -> FeeStructure {
        FeeStructure {
            exchange: Exchange("Coinbase".to_string()),
            base_maker_fee_bps: 60.0,  // 0.60%
            base_taker_fee_bps: 60.0,  // 0.60%
            volume_tiers: vec![
                VolumeTier {
                    tier_name: "Tier 1".to_string(),
                    min_volume_usd: Decimal::ZERO,
                    maker_fee_bps: 60.0,
                    taker_fee_bps: 60.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 2".to_string(),
                    min_volume_usd: Decimal::from(10_000),
                    maker_fee_bps: 40.0,
                    taker_fee_bps: 50.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 3".to_string(),
                    min_volume_usd: Decimal::from(50_000),
                    maker_fee_bps: 25.0,
                    taker_fee_bps: 35.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 4".to_string(),
                    min_volume_usd: Decimal::from(100_000),
                    maker_fee_bps: 15.0,
                    taker_fee_bps: 25.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 5".to_string(),
                    min_volume_usd: Decimal::from(1_000_000),
                    maker_fee_bps: 10.0,
                    taker_fee_bps: 20.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 6".to_string(),
                    min_volume_usd: Decimal::from(15_000_000),
                    maker_fee_bps: 5.0,
                    taker_fee_bps: 15.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 7".to_string(),
                    min_volume_usd: Decimal::from(75_000_000),
                    maker_fee_bps: 0.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Tier 8".to_string(),
                    min_volume_usd: Decimal::from(250_000_000),
                    maker_fee_bps: 0.0,
                    taker_fee_bps: 8.0,
                    maker_rebate_bps: None,
                },
            ],
            maker_rebate_available: false,
            special_programs: vec![],
            payment_discount: None,
        }
    }
    
    fn create_kraken_fees() -> FeeStructure {
        FeeStructure {
            exchange: Exchange("Kraken".to_string()),
            base_maker_fee_bps: 16.0,  // 0.16%
            base_taker_fee_bps: 26.0,  // 0.26%
            volume_tiers: vec![
                VolumeTier {
                    tier_name: "Starter".to_string(),
                    min_volume_usd: Decimal::ZERO,
                    maker_fee_bps: 16.0,
                    taker_fee_bps: 26.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Intermediate".to_string(),
                    min_volume_usd: Decimal::from(50_000),
                    maker_fee_bps: 14.0,
                    taker_fee_bps: 24.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro".to_string(),
                    min_volume_usd: Decimal::from(100_000),
                    maker_fee_bps: 12.0,
                    taker_fee_bps: 22.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 2".to_string(),
                    min_volume_usd: Decimal::from(250_000),
                    maker_fee_bps: 10.0,
                    taker_fee_bps: 20.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 3".to_string(),
                    min_volume_usd: Decimal::from(500_000),
                    maker_fee_bps: 8.0,
                    taker_fee_bps: 18.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 4".to_string(),
                    min_volume_usd: Decimal::from(1_000_000),
                    maker_fee_bps: 6.0,
                    taker_fee_bps: 16.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 5".to_string(),
                    min_volume_usd: Decimal::from(2_500_000),
                    maker_fee_bps: 4.0,
                    taker_fee_bps: 14.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 6".to_string(),
                    min_volume_usd: Decimal::from(5_000_000),
                    maker_fee_bps: 2.0,
                    taker_fee_bps: 12.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Pro 7".to_string(),
                    min_volume_usd: Decimal::from(10_000_000),
                    maker_fee_bps: 0.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
            ],
            maker_rebate_available: false,
            special_programs: vec![],
            payment_discount: None,
        }
    }
    
    fn create_okx_fees() -> FeeStructure {
        FeeStructure {
            exchange: Exchange("OKX".to_string()),
            base_maker_fee_bps: 8.0,   // 0.08%
            base_taker_fee_bps: 10.0,  // 0.10%
            volume_tiers: vec![
                VolumeTier {
                    tier_name: "Lv1".to_string(),
                    min_volume_usd: Decimal::ZERO,
                    maker_fee_bps: 8.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Lv2".to_string(),
                    min_volume_usd: Decimal::from(10_000_000),
                    maker_fee_bps: 6.5,
                    taker_fee_bps: 8.5,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Lv3".to_string(),
                    min_volume_usd: Decimal::from(20_000_000),
                    maker_fee_bps: 6.0,
                    taker_fee_bps: 8.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Lv4".to_string(),
                    min_volume_usd: Decimal::from(100_000_000),
                    maker_fee_bps: 5.0,
                    taker_fee_bps: 7.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "Lv5".to_string(),
                    min_volume_usd: Decimal::from(200_000_000),
                    maker_fee_bps: 2.0,
                    taker_fee_bps: 5.0,
                    maker_rebate_bps: Some(1.0),
                },
            ],
            maker_rebate_available: true,
            special_programs: vec![
                SpecialProgram::VIP {
                    level: 5,
                    maker_discount_pct: 20.0,
                    taker_discount_pct: 20.0,
                    min_balance_usd: Decimal::from(5_000_000),
                },
            ],
            payment_discount: Some(PaymentDiscount {
                token_symbol: "OKB".to_string(),
                discount_pct: 25.0,
                requires_holding: true,
                min_holding_amount: Some(Decimal::from(500)),
            }),
        }
    }
    
    fn create_bybit_fees() -> FeeStructure {
        FeeStructure {
            exchange: Exchange("Bybit".to_string()),
            base_maker_fee_bps: 10.0,  // 0.10%
            base_taker_fee_bps: 10.0,  // 0.10%
            volume_tiers: vec![
                VolumeTier {
                    tier_name: "None".to_string(),
                    min_volume_usd: Decimal::ZERO,
                    maker_fee_bps: 10.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 1".to_string(),
                    min_volume_usd: Decimal::from(100_000),
                    maker_fee_bps: 6.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 2".to_string(),
                    min_volume_usd: Decimal::from(250_000),
                    maker_fee_bps: 5.0,
                    taker_fee_bps: 9.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 3".to_string(),
                    min_volume_usd: Decimal::from(1_000_000),
                    maker_fee_bps: 4.0,
                    taker_fee_bps: 8.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 4".to_string(),
                    min_volume_usd: Decimal::from(10_000_000),
                    maker_fee_bps: 2.5,
                    taker_fee_bps: 7.0,
                    maker_rebate_bps: None,
                },
                VolumeTier {
                    tier_name: "VIP 5".to_string(),
                    min_volume_usd: Decimal::from(50_000_000),
                    maker_fee_bps: 0.0,
                    taker_fee_bps: 5.5,
                    maker_rebate_bps: Some(1.0),
                },
            ],
            maker_rebate_available: true,
            special_programs: vec![],
            payment_discount: None,
        }
    }
}

/// Fee calculator implementation
pub struct FeeCalculator {
    exchange_fees: Arc<ExchangeFees>,
    volume_tracker: Arc<RwLock<HashMap<(Exchange, String), VolumeTracker>>>,
    custom_overrides: Arc<RwLock<HashMap<Exchange, FeeStructure>>>,
}

/// Volume tracking for tier calculation
#[derive(Debug, Clone)]
struct VolumeTracker {
    user_id: String,
    exchange: Exchange,
    thirty_day_volume: Decimal,
    current_tier: TierLevel,
    last_update: DateTime<Utc>,
    daily_volumes: VecDeque<(DateTime<Utc>, Decimal)>,
}

impl VolumeTracker {
    fn new(user_id: String, exchange: Exchange) -> Self {
        Self {
            user_id,
            exchange,
            thirty_day_volume: Decimal::ZERO,
            current_tier: TierLevel {
                level: 0,
                name: "Base".to_string(),
                maker_fee_bps: 10.0,
                taker_fee_bps: 10.0,
                benefits: vec![],
            },
            last_update: Utc::now(),
            daily_volumes: VecDeque::with_capacity(30),
        }
    }
    
    fn update_volume(&mut self, volume: Decimal, timestamp: DateTime<Utc>) {
        // Add to daily volumes
        if let Some(last) = self.daily_volumes.back_mut() {
            if last.0.date() == timestamp.date() {
                last.1 += volume;
            } else {
                self.daily_volumes.push_back((timestamp, volume));
            }
        } else {
            self.daily_volumes.push_back((timestamp, volume));
        }
        
        // Remove volumes older than 30 days
        let cutoff = timestamp - Duration::days(30);
        while let Some(front) = self.daily_volumes.front() {
            if front.0 < cutoff {
                self.daily_volumes.pop_front();
            } else {
                break;
            }
        }
        
        // Recalculate 30-day volume
        self.thirty_day_volume = self.daily_volumes.iter()
            .map(|(_, v)| *v)
            .fold(Decimal::ZERO, |acc, v| acc + v);
        
        self.last_update = timestamp;
    }
}

use std::collections::VecDeque;

impl FeeCalculator {
    pub fn new() -> Self {
        Self {
            exchange_fees: Arc::new(ExchangeFees::new()),
            volume_tracker: Arc::new(RwLock::new(HashMap::new())),
            custom_overrides: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Calculate fees for a trade
    pub fn calculate_fee(
        &self,
        exchange: &Exchange,
        symbol: &Symbol,
        quantity: Quantity,
        price: Price,
        is_maker: bool,
        user_id: Option<String>,
    ) -> Result<(Decimal, MakerTakerFees)> {
        let fee_structure = self.get_fee_structure(exchange)?;
        let user_id = user_id.unwrap_or_else(|| "default".to_string());
        
        // Get user's volume tier
        let tier = self.get_user_tier(exchange, &user_id)?;
        
        // Get base fee
        let base_fee_bps = if is_maker {
            tier.maker_fee_bps
        } else {
            tier.taker_fee_bps
        };
        
        // Apply discounts
        let mut final_fee_bps = base_fee_bps;
        
        // Payment token discount
        if let Some(discount) = &fee_structure.payment_discount {
            // Assume user has the token for simulation
            final_fee_bps *= (100.0 - discount.discount_pct) / 100.0;
        }
        
        // Special program discounts
        for program in &fee_structure.special_programs {
            match program {
                SpecialProgram::MarketMaker { maker_rebate_bps, .. } if is_maker => {
                    // Market maker rebate
                    final_fee_bps = -*maker_rebate_bps;
                }
                SpecialProgram::VIP { maker_discount_pct, taker_discount_pct, .. } => {
                    let discount = if is_maker { maker_discount_pct } else { taker_discount_pct };
                    final_fee_bps *= (100.0 - discount) / 100.0;
                }
                SpecialProgram::Referral { discount_pct, .. } => {
                    final_fee_bps *= (100.0 - discount_pct) / 100.0;
                }
                _ => {}
            }
        }
        
        // Calculate fee amount
        let trade_value = quantity.0 * price.0;
        let fee_amount = trade_value * Decimal::from_f64_retain(final_fee_bps / 10000.0)
            .unwrap_or(Decimal::ZERO);
        
        let fees = MakerTakerFees {
            maker_fee_bps: tier.maker_fee_bps,
            taker_fee_bps: tier.taker_fee_bps,
            maker_rebate_bps: if final_fee_bps < 0.0 { Some(-final_fee_bps) } else { None },
            effective_date: Utc::now(),
        };
        
        Ok((fee_amount, fees))
    }
    
    /// Get fee structure for an exchange
    fn get_fee_structure(&self, exchange: &Exchange) -> Result<FeeStructure> {
        // Check for custom overrides first
        if let Some(custom) = self.custom_overrides.read().get(exchange) {
            return Ok(custom.clone());
        }
        
        // Get standard fee structure
        let fee_structure = match exchange.0.as_str() {
            "Binance" => self.exchange_fees.binance.clone(),
            "Coinbase" => self.exchange_fees.coinbase.clone(),
            "Kraken" => self.exchange_fees.kraken.clone(),
            "OKX" => self.exchange_fees.okx.clone(),
            "Bybit" => self.exchange_fees.bybit.clone(),
            _ => {
                // Default fee structure for unknown exchanges
                FeeStructure {
                    exchange: exchange.clone(),
                    base_maker_fee_bps: 10.0,
                    base_taker_fee_bps: 10.0,
                    volume_tiers: vec![],
                    maker_rebate_available: false,
                    special_programs: vec![],
                    payment_discount: None,
                }
            }
        };
        
        Ok(fee_structure)
    }
    
    /// Get user's current tier
    fn get_user_tier(&self, exchange: &Exchange, user_id: &str) -> Result<TierLevel> {
        let key = (exchange.clone(), user_id.to_string());
        let trackers = self.volume_tracker.read();
        
        if let Some(tracker) = trackers.get(&key) {
            Ok(tracker.current_tier.clone())
        } else {
            // Return base tier for new users
            Ok(TierLevel {
                level: 0,
                name: "Base".to_string(),
                maker_fee_bps: 10.0,
                taker_fee_bps: 10.0,
                benefits: vec![],
            })
        }
    }
    
    /// Update user's trading volume
    pub fn update_user_volume(
        &self,
        exchange: &Exchange,
        user_id: &str,
        volume_usd: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        let key = (exchange.clone(), user_id.to_string());
        let mut trackers = self.volume_tracker.write();
        
        let tracker = trackers.entry(key)
            .or_insert_with(|| VolumeTracker::new(user_id.to_string(), exchange.clone()));
        
        tracker.update_volume(volume_usd, timestamp);
        
        // Update tier based on new volume
        let fee_structure = self.get_fee_structure(exchange)?;
        for tier in fee_structure.volume_tiers.iter().rev() {
            if tracker.thirty_day_volume >= tier.min_volume_usd {
                tracker.current_tier = TierLevel {
                    level: 0,  // Would need proper level tracking
                    name: tier.tier_name.clone(),
                    maker_fee_bps: tier.maker_fee_bps,
                    taker_fee_bps: tier.taker_fee_bps,
                    benefits: vec![],
                };
                break;
            }
        }
        
        Ok(())
    }
    
    /// Calculate total fees for multiple trades
    pub fn calculate_batch_fees(
        &self,
        trades: &[(Exchange, Symbol, Quantity, Price, bool)],
        user_id: Option<String>,
    ) -> Result<Decimal> {
        let mut total_fees = Decimal::ZERO;
        
        for (exchange, symbol, quantity, price, is_maker) in trades {
            let (fee, _) = self.calculate_fee(
                exchange,
                symbol,
                quantity.clone(),
                price.clone(),
                *is_maker,
                user_id.clone(),
            )?;
            total_fees += fee;
        }
        
        Ok(total_fees)
    }
    
    /// Estimate fees for different order types
    pub fn estimate_order_fee(
        &self,
        exchange: &Exchange,
        symbol: &Symbol,
        quantity: Quantity,
        price: Price,
        order_type: OrderType,
        user_id: Option<String>,
    ) -> Result<(Decimal, Decimal)> {
        let (maker_fee, _) = self.calculate_fee(
            exchange,
            symbol,
            quantity.clone(),
            price.clone(),
            true,
            user_id.clone(),
        )?;
        
        let (taker_fee, _) = self.calculate_fee(
            exchange,
            symbol,
            quantity.clone(),
            price.clone(),
            false,
            user_id,
        )?;
        
        match order_type {
            OrderType::Market => Ok((taker_fee.clone(), taker_fee)),
            OrderType::Limit => Ok((maker_fee.clone(), maker_fee)),
            OrderType::LimitIOC => Ok((taker_fee.clone(), taker_fee)),
            OrderType::PostOnly => Ok((maker_fee.clone(), maker_fee)),
            OrderType::Mixed => {
                // Assume 50/50 maker/taker
                let avg_fee = (maker_fee + taker_fee) / Decimal::from(2);
                Ok((maker_fee, avg_fee))
            }
        }
    }
    
    /// Set custom fee structure for testing
    pub fn set_custom_fees(&self, exchange: Exchange, fee_structure: FeeStructure) {
        self.custom_overrides.write().insert(exchange, fee_structure);
    }
}

/// Order types for fee estimation
#[derive(Debug, Clone, Copy)]
pub enum OrderType {
    Market,
    Limit,
    LimitIOC,
    PostOnly,
    Mixed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_fee_calculation() {
        let calculator = FeeCalculator::new();
        
        let exchange = Exchange("Binance".to_string());
        let symbol = Symbol("BTC-USDT".to_string());
        let quantity = Quantity(dec!(1.0));
        let price = Price(dec!(50000));
        
        // Test taker fee
        let (fee, fees) = calculator.calculate_fee(
            &exchange,
            &symbol,
            quantity.clone(),
            price.clone(),
            false,  // taker
            None,
        ).unwrap();
        
        // Base tier: 0.10% = 10 bps
        let expected = dec!(50000) * dec!(0.001);  // 0.1%
        assert_eq!(fee, expected);
        assert_eq!(fees.taker_fee_bps, 10.0);
        
        // Test maker fee
        let (fee, fees) = calculator.calculate_fee(
            &exchange,
            &symbol,
            quantity,
            price,
            true,  // maker
            None,
        ).unwrap();
        
        assert_eq!(fee, expected);
        assert_eq!(fees.maker_fee_bps, 10.0);
    }
    
    #[test]
    fn test_volume_tier_upgrade() {
        let calculator = FeeCalculator::new();
        let exchange = Exchange("Binance".to_string());
        let user_id = "test_user";
        
        // Update volume to VIP 1 level
        calculator.update_user_volume(
            &exchange,
            user_id,
            dec!(50_000_000),
            Utc::now(),
        ).unwrap();
        
        // Check that tier was updated
        let tier = calculator.get_user_tier(&exchange, user_id).unwrap();
        assert_eq!(tier.name, "VIP 1");
        assert_eq!(tier.maker_fee_bps, 9.0);
    }
    
    #[test]
    fn test_payment_discount() {
        let calculator = FeeCalculator::new();
        
        // Binance has BNB discount of 25%
        let exchange = Exchange("Binance".to_string());
        let symbol = Symbol("ETH-USDT".to_string());
        let quantity = Quantity(dec!(10));
        let price = Price(dec!(2000));
        
        let (fee_with_discount, _) = calculator.calculate_fee(
            &exchange,
            &symbol,
            quantity,
            price,
            false,
            None,
        ).unwrap();
        
        // Expected: 10 bps * 0.75 (25% discount) = 7.5 bps
        let expected = dec!(20000) * dec!(0.00075);
        assert_eq!(fee_with_discount, expected);
    }
    
    #[test]
    fn test_batch_fee_calculation() {
        let calculator = FeeCalculator::new();
        
        let trades = vec![
            (Exchange("Binance".to_string()), Symbol("BTC-USDT".to_string()), 
             Quantity(dec!(1)), Price(dec!(50000)), false),
            (Exchange("Coinbase".to_string()), Symbol("ETH-USDT".to_string()), 
             Quantity(dec!(10)), Price(dec!(2000)), true),
            (Exchange("Kraken".to_string()), Symbol("SOL-USDT".to_string()), 
             Quantity(dec!(100)), Price(dec!(100)), false),
        ];
        
        let total_fees = calculator.calculate_batch_fees(&trades, None).unwrap();
        assert!(total_fees > Decimal::ZERO);
    }
}