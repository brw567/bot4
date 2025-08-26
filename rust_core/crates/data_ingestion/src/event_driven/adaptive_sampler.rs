// Adaptive Sampler - Volatility-based sampling rate adjustment
// DEEP DIVE: Dynamic sampling based on market conditions
//
// References:
// - "Realized Volatility" - Andersen et al. (2003)
// - "DeepVol: Volatility Forecasting" - Journal of Financial Econometrics (2024)
// - "TimeMixer for Volatility" - ArXiv (2024)
// - "Optimal Sampling Frequency" - Zhang, Mykland, Aït-Sahalia (2005)

use std::sync::Arc;
use std::collections::VecDeque;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{debug, info, warn, instrument};
use statrs::statistics::Statistics;

use types::{Price, Quantity, Symbol};
// TODO: use infrastructure::metrics::{MetricsCollector, register_histogram, register_counter};

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    VeryLow,    // < 10% annualized
    Low,        // 10-20% annualized
    Normal,     // 20-40% annualized
    High,       // 40-80% annualized
    Extreme,    // > 80% annualized
    Crisis,     // Market stress/crash conditions
}

/// Sampling strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Fixed rate sampling
    Fixed { interval_ms: u64 },
    
    /// Adaptive based on volatility
    Adaptive {
        min_interval_ms: u64,
        max_interval_ms: u64,
    },
    
    /// Event-driven with rate limits
    EventDriven {
        min_gap_ms: u64,
        burst_size: u32,
    },
    
    /// Hybrid approach
    Hybrid {
        base_interval_ms: u64,
        volatility_multiplier: f64,
    },
}

/// Sampling rate configuration
#[derive(Debug, Clone)]
pub struct SamplingRate {
    pub interval_ms: u64,
    pub events_per_second: f64,
    pub regime: VolatilityRegime,
    pub confidence: f64,
}

/// Sampler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Base sampling interval (milliseconds)
    pub base_interval_ms: u64,
    
    /// Minimum sampling interval (milliseconds)
    pub min_interval_ms: u64,
    
    /// Maximum sampling interval (milliseconds)
    pub max_interval_ms: u64,
    
    /// Volatility lookback window (seconds)
    pub volatility_window_sec: u64,
    
    /// Regime change threshold (%)
    pub regime_change_threshold: f64,
    
    /// Enable GARCH volatility model
    pub use_garch: bool,
    
    /// Enable realized volatility
    pub use_realized_vol: bool,
    
    /// Enable implied volatility (if available)
    pub use_implied_vol: bool,
    
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    
    /// Burst detection threshold
    pub burst_threshold: f64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            base_interval_ms: 5,  // 5ms base (200 Hz)
            min_interval_ms: 1,   // 1ms minimum (1000 Hz)
            max_interval_ms: 100, // 100ms maximum (10 Hz)
            volatility_window_sec: 300, // 5 minutes
            regime_change_threshold: 0.2, // 20% change
            use_garch: true,
            use_realized_vol: true,
            use_implied_vol: false,
            strategy: SamplingStrategy::Adaptive {
                min_interval_ms: 1,
                max_interval_ms: 100,
            },
            burst_threshold: 3.0, // 3 sigma events
        }
    }
}

/// Price sample for volatility calculation
#[derive(Debug, Clone)]
struct PriceSample {
    timestamp: DateTime<Utc>,
    price: Price,
    volume: Quantity,
    spread: Option<Decimal>,
}

/// GARCH(1,1) model for volatility
struct GarchModel {
    omega: f64,    // Constant
    alpha: f64,    // ARCH coefficient
    beta: f64,     // GARCH coefficient
    current_vol: f64,
    last_return: f64,
}

impl GarchModel {
    fn new() -> Self {
        // Standard GARCH(1,1) parameters
        Self {
            omega: 0.000001,
            alpha: 0.1,      // Weight on squared return
            beta: 0.85,      // Weight on previous volatility
            current_vol: 0.02, // 2% initial volatility
            last_return: 0.0,
        }
    }
    
    fn update(&mut self, return_val: f64) {
        // GARCH(1,1): σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
        let variance = self.omega + 
                      self.alpha * self.last_return.powi(2) + 
                      self.beta * self.current_vol.powi(2);
        
        self.current_vol = variance.sqrt();
        self.last_return = return_val;
    }
    
    fn forecast(&self, horizon: usize) -> f64 {
        // Multi-step ahead forecast
        let long_run_variance = self.omega / (1.0 - self.alpha - self.beta);
        let persistence = self.alpha + self.beta;
        
        let current_variance = self.current_vol.powi(2);
        let forecast_variance = long_run_variance + 
            (current_variance - long_run_variance) * persistence.powi(horizon as i32);
        
        forecast_variance.sqrt()
    }
}

/// Main adaptive sampler implementation
pub struct AdaptiveSampler {
    config: Arc<SamplerConfig>,
    
    // Price history for volatility calculation
    price_history: Arc<RwLock<VecDeque<PriceSample>>>,
    
    // Current volatility metrics
    current_volatility: Arc<RwLock<f64>>,
    current_regime: Arc<RwLock<VolatilityRegime>>,
    
    // GARCH model
    garch_model: Arc<RwLock<GarchModel>>,
    
    // Sampling rate
    current_rate: Arc<RwLock<SamplingRate>>,
    
    // Metrics
    sampling_rate_changes: Arc<dyn MetricsCollector>,
    regime_changes: Arc<dyn MetricsCollector>,
    volatility_histogram: Arc<dyn MetricsCollector>,
}

impl AdaptiveSampler {
    pub fn new(config: SamplerConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config),
            price_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            current_volatility: Arc::new(RwLock::new(0.02)), // 2% default
            current_regime: Arc::new(RwLock::new(VolatilityRegime::Normal)),
            garch_model: Arc::new(RwLock::new(GarchModel::new())),
            current_rate: Arc::new(RwLock::new(SamplingRate {
                interval_ms: 5,
                events_per_second: 200.0,
                regime: VolatilityRegime::Normal,
                confidence: 0.95,
            })),
            sampling_rate_changes: register_counter("adaptive_sampler_rate_changes"),
            regime_changes: register_counter("adaptive_sampler_regime_changes"),
            volatility_histogram: register_histogram("adaptive_sampler_volatility"),
        })
    }
    
    /// Add price sample and update volatility
    #[instrument(skip(self), fields(symbol = %symbol.0))]
    pub fn add_price_sample(
        &self,
        symbol: &Symbol,
        price: Price,
        volume: Quantity,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        // Add to history
        let mut history = self.price_history.write();
        
        // Remove old samples
        let cutoff = timestamp - Duration::seconds(self.config.volatility_window_sec as i64);
        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
        
        // Add new sample
        history.push_back(PriceSample {
            timestamp,
            price: price.clone(),
            volume,
            spread: None,
        });
        
        // Update volatility if we have enough samples
        if history.len() >= 20 {
            self.update_volatility(&history)?;
        }
        
        Ok(())
    }
    
    /// Update volatility calculations
    fn update_volatility(&self, history: &VecDeque<PriceSample>) -> Result<()> {
        // Calculate returns
        let mut returns = Vec::with_capacity(history.len() - 1);
        
        for i in 1..history.len() {
            let prev_price = history[i-1].price.0.to_f64().unwrap_or(0.0);
            let curr_price = history[i].price.0.to_f64().unwrap_or(0.0);
            
            if prev_price > 0.0 && curr_price > 0.0 {
                let return_val = (curr_price / prev_price).ln();
                returns.push(return_val);
            }
        }
        
        if returns.is_empty() {
            return Ok(());
        }
        
        // Calculate realized volatility
        let realized_vol = if self.config.use_realized_vol {
            self.calculate_realized_volatility(&returns)?
        } else {
            0.0
        };
        
        // Update GARCH model
        let garch_vol = if self.config.use_garch {
            let mut garch = self.garch_model.write();
            for ret in &returns {
                garch.update(*ret);
            }
            garch.current_vol
        } else {
            0.0
        };
        
        // Combine volatility estimates
        let combined_vol = if self.config.use_garch && self.config.use_realized_vol {
            // Weighted average
            0.6 * garch_vol + 0.4 * realized_vol
        } else if self.config.use_garch {
            garch_vol
        } else {
            realized_vol
        };
        
        // Annualize (assuming 365 days)
        let annualized_vol = combined_vol * (365.0_f64).sqrt();
        
        // Update current volatility
        *self.current_volatility.write() = annualized_vol;
        self.volatility_histogram.record(annualized_vol * 100.0); // Record as percentage
        
        // Update regime
        self.update_regime(annualized_vol)?;
        
        // Update sampling rate
        self.update_sampling_rate(annualized_vol)?;
        
        Ok(())
    }
    
    /// Calculate realized volatility
    fn calculate_realized_volatility(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        // Standard deviation of returns
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Update volatility regime
    fn update_regime(&self, annualized_vol: f64) -> Result<()> {
        let vol_pct = annualized_vol * 100.0;
        
        let new_regime = if vol_pct < 10.0 {
            VolatilityRegime::VeryLow
        } else if vol_pct < 20.0 {
            VolatilityRegime::Low
        } else if vol_pct < 40.0 {
            VolatilityRegime::Normal
        } else if vol_pct < 80.0 {
            VolatilityRegime::High
        } else if vol_pct < 150.0 {
            VolatilityRegime::Extreme
        } else {
            VolatilityRegime::Crisis
        };
        
        let mut current = self.current_regime.write();
        if *current != new_regime {
            info!("Volatility regime change: {:?} -> {:?} ({}%)", 
                  *current, new_regime, vol_pct);
            self.regime_changes.increment(1);
            *current = new_regime;
        }
        
        Ok(())
    }
    
    /// Update sampling rate based on volatility
    fn update_sampling_rate(&self, annualized_vol: f64) -> Result<()> {
        let interval_ms = match self.config.strategy {
            SamplingStrategy::Fixed { interval_ms } => interval_ms,
            
            SamplingStrategy::Adaptive { min_interval_ms, max_interval_ms } => {
                // Higher volatility = higher sampling rate (lower interval)
                // Map volatility [0%, 100%] to interval [max, min]
                let vol_pct = (annualized_vol * 100.0).min(100.0).max(0.0);
                let ratio = 1.0 - (vol_pct / 100.0); // Invert so high vol = low ratio
                
                let range = max_interval_ms - min_interval_ms;
                min_interval_ms + (range as f64 * ratio) as u64
            }
            
            SamplingStrategy::EventDriven { min_gap_ms, .. } => {
                // Event-driven doesn't use fixed intervals
                min_gap_ms
            }
            
            SamplingStrategy::Hybrid { base_interval_ms, volatility_multiplier } => {
                // Adjust base interval by volatility
                let adjustment = 1.0 / (1.0 + annualized_vol * volatility_multiplier);
                (base_interval_ms as f64 * adjustment) as u64
            }
        };
        
        // Clamp to configured limits
        let interval_ms = interval_ms
            .max(self.config.min_interval_ms)
            .min(self.config.max_interval_ms);
        
        let events_per_second = 1000.0 / interval_ms as f64;
        
        let mut current_rate = self.current_rate.write();
        if current_rate.interval_ms != interval_ms {
            debug!("Sampling rate changed: {}ms -> {}ms ({:.1} Hz)", 
                   current_rate.interval_ms, interval_ms, events_per_second);
            self.sampling_rate_changes.increment(1);
            
            *current_rate = SamplingRate {
                interval_ms,
                events_per_second,
                regime: *self.current_regime.read(),
                confidence: self.calculate_confidence(annualized_vol),
            };
        }
        
        Ok(())
    }
    
    /// Calculate confidence in volatility estimate
    fn calculate_confidence(&self, volatility: f64) -> f64 {
        let history = self.price_history.read();
        let sample_size = history.len();
        
        // More samples = higher confidence
        let size_factor = (sample_size as f64 / 100.0).min(1.0);
        
        // Stable volatility = higher confidence
        let stability_factor = 1.0 / (1.0 + volatility * 10.0);
        
        size_factor * 0.5 + stability_factor * 0.5
    }
    
    /// Check if we should sample now
    pub fn should_sample(&self, timestamp: DateTime<Utc>) -> bool {
        let rate = self.current_rate.read();
        
        match self.config.strategy {
            SamplingStrategy::EventDriven { burst_size, .. } => {
                // Always sample for event-driven (with burst control elsewhere)
                true
            }
            _ => {
                // Time-based sampling
                // This would typically check against last sample time
                true // Simplified for now
            }
        }
    }
    
    /// Detect market microbursts
    pub fn detect_burst(&self, recent_events: &[DateTime<Utc>]) -> bool {
        if recent_events.len() < 10 {
            return false;
        }
        
        // Calculate event rate
        let duration = recent_events.last().unwrap()
            .signed_duration_since(*recent_events.first().unwrap());
        
        if duration.num_milliseconds() == 0 {
            return true; // Multiple events at same timestamp
        }
        
        let events_per_ms = recent_events.len() as f64 / duration.num_milliseconds() as f64;
        let events_per_sec = events_per_ms * 1000.0;
        
        // Compare to expected rate
        let expected_rate = self.current_rate.read().events_per_second;
        let ratio = events_per_sec / expected_rate;
        
        ratio > self.config.burst_threshold
    }
    
    /// Get current sampling rate
    pub fn get_current_rate(&self) -> SamplingRate {
        self.current_rate.read().clone()
    }
    
    /// Get current volatility
    pub fn get_volatility(&self) -> f64 {
        *self.current_volatility.read()
    }
    
    /// Get current regime
    pub fn get_regime(&self) -> VolatilityRegime {
        *self.current_regime.read()
    }
    
    /// Forecast future volatility
    pub fn forecast_volatility(&self, horizon_hours: usize) -> f64 {
        let garch = self.garch_model.read();
        
        // Convert hours to periods (assuming 5-minute bars)
        let periods = horizon_hours * 12;
        garch.forecast(periods)
    }
    
    /// Get optimal sampling rate for given conditions
    pub fn get_optimal_rate(
        &self,
        volatility: f64,
        liquidity: f64,
        spread: f64,
    ) -> u64 {
        // Zhang-Mykland-Aït-Sahalia optimal sampling
        // Balances microstructure noise vs information loss
        
        // Higher volatility -> higher frequency
        let vol_factor = (volatility * 100.0).sqrt();
        
        // Lower liquidity -> lower frequency (avoid noise)
        let liquidity_factor = liquidity.sqrt();
        
        // Wider spread -> lower frequency (higher transaction costs)
        let spread_factor = 1.0 / (1.0 + spread * 10000.0);
        
        let optimal_hz = 10.0 * vol_factor * liquidity_factor * spread_factor;
        let optimal_ms = (1000.0 / optimal_hz) as u64;
        
        optimal_ms.max(self.config.min_interval_ms)
                  .min(self.config.max_interval_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_adaptive_sampler_creation() {
        let config = SamplerConfig::default();
        let sampler = AdaptiveSampler::new(config).unwrap();
        
        assert_eq!(sampler.get_regime(), VolatilityRegime::Normal);
        assert_eq!(sampler.get_current_rate().interval_ms, 5);
    }
    
    #[test]
    fn test_volatility_regime_detection() {
        let config = SamplerConfig::default();
        let sampler = AdaptiveSampler::new(config).unwrap();
        
        let symbol = Symbol("BTC-USDT".to_string());
        let mut timestamp = Utc::now();
        
        // Add samples with increasing volatility
        let base_price = 50000.0;
        for i in 0..100 {
            let volatility = 0.001 * (i as f64 / 10.0); // Increasing volatility
            let price_change = base_price * volatility * (if i % 2 == 0 { 1.0 } else { -1.0 });
            let price = Price(dec!(50000) + Decimal::from_f64_retain(price_change).unwrap());
            
            sampler.add_price_sample(
                &symbol,
                price,
                Quantity(dec!(1)),
                timestamp,
            ).unwrap();
            
            timestamp = timestamp + Duration::seconds(1);
        }
        
        // Check that volatility was detected
        let vol = sampler.get_volatility();
        assert!(vol > 0.0);
    }
    
    #[test]
    fn test_sampling_rate_adjustment() {
        let config = SamplerConfig {
            strategy: SamplingStrategy::Adaptive {
                min_interval_ms: 1,
                max_interval_ms: 100,
            },
            ..Default::default()
        };
        
        let sampler = AdaptiveSampler::new(config).unwrap();
        
        // Low volatility should give high interval (low frequency)
        sampler.update_sampling_rate(0.1).unwrap(); // 10% annual vol
        let rate1 = sampler.get_current_rate();
        
        // High volatility should give low interval (high frequency)
        sampler.update_sampling_rate(0.8).unwrap(); // 80% annual vol
        let rate2 = sampler.get_current_rate();
        
        assert!(rate2.interval_ms < rate1.interval_ms);
        assert!(rate2.events_per_second > rate1.events_per_second);
    }
    
    #[test]
    fn test_burst_detection() {
        let config = SamplerConfig::default();
        let sampler = AdaptiveSampler::new(config).unwrap();
        
        let now = Utc::now();
        
        // Normal event spacing (5ms apart)
        let normal_events: Vec<DateTime<Utc>> = (0..10)
            .map(|i| now + Duration::milliseconds(i * 5))
            .collect();
        
        assert!(!sampler.detect_burst(&normal_events));
        
        // Burst events (1ms apart)
        let burst_events: Vec<DateTime<Utc>> = (0..10)
            .map(|i| now + Duration::milliseconds(i))
            .collect();
        
        assert!(sampler.detect_burst(&burst_events));
    }
    
    #[test]
    fn test_garch_model() {
        let mut garch = GarchModel::new();
        
        // Simulate returns
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.03, -0.025];
        
        for ret in returns {
            garch.update(ret);
        }
        
        // Forecast should be positive
        let forecast = garch.forecast(10);
        assert!(forecast > 0.0);
        assert!(forecast < 1.0); // Reasonable volatility
    }
}