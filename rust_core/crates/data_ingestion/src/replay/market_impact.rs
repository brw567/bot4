// Market Impact Calculator
// DEEP DIVE: Advanced models for price impact estimation
//
// References:
// - "Empirical Properties of Asset Returns" - Cont (2001)
// - "Price Impact and Portfolio Execution" - Almgren, Thum, Hauptmann, Li (2005)
// - "The Square-Root Law of Price Impact" - Gabaix et al. (2006)
// - "Optimal Trading with Linear Costs" - Obizhaev & Wang (2013)

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use statrs::distribution::{Normal, ContinuousCDF};

use crate::types::{Price, Quantity};
use crate::replay::lob_simulator::OrderBook;

/// Kyle's lambda - permanent price impact coefficient
#[derive(Debug, Clone)]
pub struct KyleLambda {
    /// Lambda coefficient (price impact per unit volume)
    pub lambda: f64,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    
    /// R-squared of regression
    pub r_squared: f64,
    
    /// Sample size used for estimation
    pub sample_size: usize,
}

/// Almgren-Chriss model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlmgrenChriss {
    /// Permanent impact coefficient
    pub gamma: f64,
    
    /// Temporary impact coefficient  
    pub eta: f64,
    
    /// Risk aversion parameter
    pub lambda: f64,
    
    /// Daily volatility
    pub sigma: f64,
    
    /// Daily volume
    pub daily_volume: f64,
}

/// Obizhaev-Wang model parameters
#[derive(Debug, Clone)]
pub struct ObizhaevWang {
    /// Market depth parameter
    pub kappa: f64,
    
    /// Resilience speed
    pub rho: f64,
    
    /// Permanent impact ratio
    pub alpha: f64,
}

/// Impact calculation parameters
#[derive(Debug, Clone)]
pub struct ImpactParameters {
    /// Order size
    pub order_size: Quantity,
    
    /// Average daily volume
    pub adv: Quantity,
    
    /// Volatility (annualized)
    pub volatility: f64,
    
    /// Bid-ask spread
    pub spread: Decimal,
    
    /// Execution duration (seconds)
    pub duration_sec: u64,
    
    /// Participation rate
    pub participation_rate: f64,
}

/// Market impact calculator
pub struct MarketImpactCalculator {
    /// Historical order flow data for calibration
    order_flow_history: Arc<RwLock<Vec<OrderFlowSample>>>,
    
    /// Calibrated Kyle lambda
    kyle_lambda: Arc<RwLock<Option<KyleLambda>>>,
    
    /// Calibrated Almgren-Chriss parameters
    ac_params: Arc<RwLock<Option<AlmgrenChriss>>>,
    
    /// Calibrated Obizhaev-Wang parameters
    ow_params: Arc<RwLock<Option<ObizhaevWang>>>,
}

/// Order flow sample for calibration
#[derive(Debug, Clone)]
struct OrderFlowSample {
    timestamp: DateTime<Utc>,
    signed_volume: f64,  // Positive for buys, negative for sells
    price_change: f64,
    spread: f64,
    market_volume: f64,
}

impl MarketImpactCalculator {
    pub fn new() -> Self {
        Self {
            order_flow_history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            kyle_lambda: Arc::new(RwLock::new(None)),
            ac_params: Arc::new(RwLock::new(None)),
            ow_params: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Add order flow sample for calibration
    pub fn add_order_flow_sample(
        &self,
        timestamp: DateTime<Utc>,
        signed_volume: f64,
        price_change: f64,
        spread: f64,
        market_volume: f64,
    ) {
        let mut history = self.order_flow_history.write();
        
        // Keep last 10000 samples
        if history.len() >= 10000 {
            history.remove(0);
        }
        
        history.push(OrderFlowSample {
            timestamp,
            signed_volume,
            price_change,
            spread,
            market_volume,
        });
    }
    
    /// Calibrate Kyle's lambda from order flow
    pub fn calibrate_kyle_lambda(&self) -> Result<KyleLambda> {
        let history = self.order_flow_history.read();
        
        if history.len() < 100 {
            anyhow::bail!("Insufficient data for Kyle lambda calibration");
        }
        
        // Linear regression: ΔP = λ * Q + ε
        let n = history.len() as f64;
        let mut sum_q = 0.0;
        let mut sum_p = 0.0;
        let mut sum_qq = 0.0;
        let mut sum_qp = 0.0;
        
        for sample in history.iter() {
            let q = sample.signed_volume / sample.market_volume;  // Normalized volume
            let p = sample.price_change;
            
            sum_q += q;
            sum_p += p;
            sum_qq += q * q;
            sum_qp += q * p;
        }
        
        // Calculate lambda using OLS
        let lambda = (n * sum_qp - sum_q * sum_p) / (n * sum_qq - sum_q * sum_q);
        
        // Calculate R-squared
        let mean_p = sum_p / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        
        for sample in history.iter() {
            let q = sample.signed_volume / sample.market_volume;
            let p = sample.price_change;
            let p_pred = lambda * q;
            
            ss_tot += (p - mean_p).powi(2);
            ss_res += (p - p_pred).powi(2);
        }
        
        let r_squared = 1.0 - (ss_res / ss_tot);
        
        // Calculate confidence interval (95%)
        let se = (ss_res / (n - 2.0)).sqrt();
        let t_stat = 1.96;  // Approximate for large samples
        let se_lambda = se / (sum_qq - sum_q * sum_q / n).sqrt();
        
        let kyle = KyleLambda {
            lambda,
            confidence_interval: (
                lambda - t_stat * se_lambda,
                lambda + t_stat * se_lambda,
            ),
            r_squared,
            sample_size: history.len(),
        };
        
        *self.kyle_lambda.write() = Some(kyle.clone());
        
        Ok(kyle)
    }
    
    /// Calibrate Almgren-Chriss model
    pub fn calibrate_almgren_chriss(
        &self,
        daily_volume: f64,
        volatility: f64,
    ) -> Result<AlmgrenChriss> {
        let history = self.order_flow_history.read();
        
        if history.len() < 100 {
            anyhow::bail!("Insufficient data for Almgren-Chriss calibration");
        }
        
        // Estimate permanent impact (gamma)
        // γ = λ * σ / V^(1/2) where λ is Kyle's lambda
        let kyle = self.kyle_lambda.read()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                // Use default if not calibrated
                KyleLambda {
                    lambda: 0.1,
                    confidence_interval: (0.05, 0.15),
                    r_squared: 0.0,
                    sample_size: 0,
                }
            });
        
        let gamma = kyle.lambda * volatility / daily_volume.sqrt();
        
        // Estimate temporary impact (eta)
        // η ≈ spread / (2 * average_trade_size)
        let avg_spread = history.iter()
            .map(|s| s.spread)
            .sum::<f64>() / history.len() as f64;
        
        let avg_trade_size = history.iter()
            .map(|s| s.signed_volume.abs())
            .sum::<f64>() / history.len() as f64;
        
        let eta = avg_spread / (2.0 * avg_trade_size);
        
        // Risk aversion (default moderate)
        let lambda = 1e-6;
        
        let ac = AlmgrenChriss {
            gamma,
            eta,
            lambda,
            sigma: volatility,
            daily_volume,
        };
        
        *self.ac_params.write() = Some(ac.clone());
        
        Ok(ac)
    }
    
    /// Calibrate Obizhaev-Wang model
    pub fn calibrate_obizhaev_wang(&self) -> Result<ObizhaevWang> {
        let history = self.order_flow_history.read();
        
        if history.len() < 100 {
            anyhow::bail!("Insufficient data for Obizhaev-Wang calibration");
        }
        
        // Estimate market depth (kappa)
        // κ = average_volume_at_best / average_spread
        let avg_spread = history.iter()
            .map(|s| s.spread)
            .sum::<f64>() / history.len() as f64;
        
        let avg_volume = history.iter()
            .map(|s| s.market_volume)
            .sum::<f64>() / history.len() as f64;
        
        let kappa = avg_volume / avg_spread;
        
        // Estimate resilience speed (rho)
        // Measure how quickly price reverts after trades
        // Simplified: use autocorrelation of price changes
        let mut autocorr = 0.0;
        let mut count = 0;
        
        for i in 1..history.len() {
            autocorr += history[i].price_change * history[i-1].price_change;
            count += 1;
        }
        
        if count > 0 {
            autocorr /= count as f64;
        }
        
        // Negative autocorrelation suggests price reversion
        let rho = (-autocorr).max(0.01);  // Ensure positive
        
        // Permanent impact ratio (typically 0.2-0.5)
        let alpha = 0.3;
        
        let ow = ObizhaevWang {
            kappa,
            rho,
            alpha,
        };
        
        *self.ow_params.write() = Some(ow.clone());
        
        Ok(ow)
    }
    
    /// Calculate impact using Kyle's model
    pub fn calculate_kyle_impact(&self, params: &ImpactParameters) -> f64 {
        let kyle = self.kyle_lambda.read();
        
        if let Some(ref kyle) = *kyle {
            // I = λ * Q
            let normalized_size = params.order_size.0.to_f64().unwrap_or(0.0) / 
                                params.adv.0.to_f64().unwrap_or(1.0);
            
            kyle.lambda * normalized_size * 10000.0  // Convert to basis points
        } else {
            // Fallback to simple linear model
            let participation = params.participation_rate;
            participation * 10.0  // 10 bps per 100% participation
        }
    }
    
    /// Calculate impact using Almgren-Chriss model
    pub fn calculate_almgren_chriss_impact(&self, params: &ImpactParameters) -> (f64, f64) {
        let ac = self.ac_params.read();
        
        if let Some(ref ac) = *ac {
            let x = params.order_size.0.to_f64().unwrap_or(0.0);
            let t = params.duration_sec as f64;
            let v = params.adv.0.to_f64().unwrap_or(1.0);
            
            // Permanent impact: γ * x
            let permanent = ac.gamma * x / v * 10000.0;
            
            // Temporary impact: η * (x/t)
            let temporary = ac.eta * (x / t) / v * 10000.0;
            
            (permanent, temporary)
        } else {
            // Fallback
            let impact = params.participation_rate * 10.0;
            (impact * 0.3, impact * 0.7)  // 30% permanent, 70% temporary
        }
    }
    
    /// Calculate impact using Obizhaev-Wang model
    pub fn calculate_obizhaev_wang_impact(&self, params: &ImpactParameters) -> f64 {
        let ow = self.ow_params.read();
        
        if let Some(ref ow) = *ow {
            let q = params.order_size.0.to_f64().unwrap_or(0.0);
            let t = params.duration_sec as f64;
            
            // Impact = (q / κ) * (α + (1-α) * exp(-ρ*t))
            let base_impact = q / ow.kappa;
            let time_factor = ow.alpha + (1.0 - ow.alpha) * (-ow.rho * t).exp();
            
            base_impact * time_factor * 10000.0  // Convert to bps
        } else {
            // Fallback
            params.participation_rate * 10.0
        }
    }
    
    /// Calculate square-root impact (empirical law)
    pub fn calculate_sqrt_impact(&self, params: &ImpactParameters) -> f64 {
        // Square-root law: I = Y * σ * √(Q/V)
        // Y ≈ 1 for most markets (empirical constant)
        
        let y = 1.0;
        let sigma = params.volatility;
        let q = params.order_size.0.to_f64().unwrap_or(0.0);
        let v = params.adv.0.to_f64().unwrap_or(1.0);
        
        y * sigma * (q / v).sqrt() * 10000.0  // Convert to bps
    }
    
    /// Calculate logarithmic impact (for large orders)
    pub fn calculate_log_impact(&self, params: &ImpactParameters) -> f64 {
        // Log impact: I = β * σ * log(1 + Q/V)
        let beta = 0.5;  // Empirical constant
        let sigma = params.volatility;
        let q = params.order_size.0.to_f64().unwrap_or(0.0);
        let v = params.adv.0.to_f64().unwrap_or(1.0);
        
        beta * sigma * (1.0 + q / v).ln() * 10000.0  // Convert to bps
    }
    
    /// Calculate power-law impact
    pub fn calculate_power_impact(&self, params: &ImpactParameters, alpha: f64) -> f64 {
        // Power law: I = β * (Q/V)^α
        let beta = 10.0;  // Calibrated constant (bps)
        let q = params.order_size.0.to_f64().unwrap_or(0.0);
        let v = params.adv.0.to_f64().unwrap_or(1.0);
        
        beta * (q / v).powf(alpha)
    }
    
    /// Get consensus impact estimate using multiple models
    pub fn calculate_consensus_impact(&self, params: &ImpactParameters) -> f64 {
        let kyle = self.calculate_kyle_impact(params);
        let (ac_perm, ac_temp) = self.calculate_almgren_chriss_impact(params);
        let ow = self.calculate_obizhaev_wang_impact(params);
        let sqrt = self.calculate_sqrt_impact(params);
        let log = self.calculate_log_impact(params);
        
        // Weighted average of models
        let weights = vec![0.2, 0.25, 0.2, 0.25, 0.1];  // Kyle, AC, OW, Sqrt, Log
        let impacts = vec![kyle, ac_perm + ac_temp, ow, sqrt, log];
        
        let weighted_sum: f64 = impacts.iter()
            .zip(weights.iter())
            .map(|(i, w)| i * w)
            .sum();
        
        weighted_sum
    }
    
    /// Estimate pre-trade impact (information leakage)
    pub fn estimate_pretrade_impact(&self, order_size: Quantity, market_cap: f64) -> f64 {
        // Pre-trade impact from information leakage
        // Larger orders relative to market cap have more leakage
        
        let size = order_size.0.to_f64().unwrap_or(0.0);
        let relative_size = size / market_cap;
        
        // Exponential model for information leakage
        let leakage_factor = 1.0 - (-relative_size * 1000.0).exp();
        
        leakage_factor * 5.0  // Up to 5 bps pre-trade impact
    }
    
    /// Estimate post-trade impact (price discovery)
    pub fn estimate_posttrade_impact(&self, execution_quality: f64) -> f64 {
        // Post-trade impact depends on execution quality
        // Poor execution leads to more adverse price discovery
        
        // execution_quality: 0 (worst) to 1 (best)
        let adverse_factor = 1.0 - execution_quality;
        
        adverse_factor * 3.0  // Up to 3 bps post-trade impact
    }
    
    /// Calculate total implementation cost
    pub fn calculate_total_cost(
        &self,
        params: &ImpactParameters,
        include_pretrade: bool,
        include_posttrade: bool,
        market_cap: Option<f64>,
    ) -> f64 {
        let mut total = 0.0;
        
        // Main market impact
        total += self.calculate_consensus_impact(params);
        
        // Spread cost
        let spread_cost = params.spread.to_f64().unwrap_or(0.0) * 10000.0 / 2.0;
        total += spread_cost;
        
        // Pre-trade impact
        if include_pretrade {
            if let Some(cap) = market_cap {
                total += self.estimate_pretrade_impact(params.order_size.clone(), cap);
            }
        }
        
        // Post-trade impact
        if include_posttrade {
            // Assume average execution quality
            total += self.estimate_posttrade_impact(0.7);
        }
        
        total
    }
}

/// Impact surface for visualization
pub struct ImpactSurface {
    /// Participation rate grid
    pub participation_rates: Vec<f64>,
    
    /// Time horizon grid
    pub time_horizons: Vec<f64>,
    
    /// Impact values (2D grid)
    pub impacts: Vec<Vec<f64>>,
}

impl ImpactSurface {
    pub fn generate(
        calculator: &MarketImpactCalculator,
        base_params: ImpactParameters,
        participation_range: (f64, f64),
        time_range: (u64, u64),
        grid_size: usize,
    ) -> Self {
        let mut participation_rates = Vec::with_capacity(grid_size);
        let mut time_horizons = Vec::with_capacity(grid_size);
        let mut impacts = Vec::with_capacity(grid_size);
        
        // Create grids
        for i in 0..grid_size {
            let p = participation_range.0 + 
                   (participation_range.1 - participation_range.0) * i as f64 / (grid_size - 1) as f64;
            participation_rates.push(p);
            
            let t = time_range.0 + 
                   (time_range.1 - time_range.0) * i as u64 / (grid_size - 1) as u64;
            time_horizons.push(t as f64);
        }
        
        // Calculate impacts
        for p in &participation_rates {
            let mut row = Vec::with_capacity(grid_size);
            
            for t in &time_horizons {
                let mut params = base_params.clone();
                params.participation_rate = *p;
                params.duration_sec = *t as u64;
                
                let impact = calculator.calculate_consensus_impact(&params);
                row.push(impact);
            }
            
            impacts.push(row);
        }
        
        ImpactSurface {
            participation_rates,
            time_horizons,
            impacts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kyle_lambda_calibration() {
        let calculator = MarketImpactCalculator::new();
        
        // Add synthetic order flow data
        for i in 0..200 {
            let signed_volume = if i % 2 == 0 { 100.0 } else { -100.0 };
            let price_change = signed_volume * 0.0001;  // Simple linear relationship
            
            calculator.add_order_flow_sample(
                Utc::now(),
                signed_volume,
                price_change,
                0.01,
                10000.0,
            );
        }
        
        let kyle = calculator.calibrate_kyle_lambda().unwrap();
        
        // Should recover the linear coefficient
        assert!((kyle.lambda - 0.01).abs() < 0.001);
        assert!(kyle.r_squared > 0.9);
    }
    
    #[test]
    fn test_sqrt_impact() {
        let calculator = MarketImpactCalculator::new();
        
        let params = ImpactParameters {
            order_size: Quantity(Decimal::from(1000)),
            adv: Quantity(Decimal::from(100000)),
            volatility: 0.3,  // 30% annual vol
            spread: Decimal::from_str("0.001").unwrap(),
            duration_sec: 300,
            participation_rate: 0.1,
        };
        
        let impact = calculator.calculate_sqrt_impact(&params);
        
        // Square root of 1% of ADV with 30% vol should be ~3 bps
        assert!(impact > 0.0 && impact < 10.0);
    }
    
    #[test]
    fn test_consensus_impact() {
        let calculator = MarketImpactCalculator::new();
        
        // Add calibration data
        for i in 0..100 {
            calculator.add_order_flow_sample(
                Utc::now(),
                100.0,
                0.01,
                0.01,
                10000.0,
            );
        }
        
        calculator.calibrate_kyle_lambda().unwrap();
        calculator.calibrate_almgren_chriss(1000000.0, 0.2).unwrap();
        calculator.calibrate_obizhaev_wang().unwrap();
        
        let params = ImpactParameters {
            order_size: Quantity(Decimal::from(5000)),
            adv: Quantity(Decimal::from(100000)),
            volatility: 0.2,
            spread: Decimal::from_str("0.001").unwrap(),
            duration_sec: 600,
            participation_rate: 0.05,
        };
        
        let impact = calculator.calculate_consensus_impact(&params);
        
        // 5% of ADV should have measurable impact
        assert!(impact > 1.0 && impact < 20.0);
    }
    
    #[test]
    fn test_impact_surface() {
        let calculator = MarketImpactCalculator::new();
        
        let base_params = ImpactParameters {
            order_size: Quantity(Decimal::from(1000)),
            adv: Quantity(Decimal::from(100000)),
            volatility: 0.25,
            spread: Decimal::from_str("0.001").unwrap(),
            duration_sec: 300,
            participation_rate: 0.05,
        };
        
        let surface = ImpactSurface::generate(
            &calculator,
            base_params,
            (0.01, 0.2),   // 1% to 20% participation
            (60, 3600),    // 1 minute to 1 hour
            10,
        );
        
        assert_eq!(surface.participation_rates.len(), 10);
        assert_eq!(surface.time_horizons.len(), 10);
        assert_eq!(surface.impacts.len(), 10);
        assert_eq!(surface.impacts[0].len(), 10);
        
        // Impact should increase with participation rate
        let first_row = &surface.impacts[0];
        let last_row = &surface.impacts[9];
        assert!(last_row[0] > first_row[0]);
    }
}