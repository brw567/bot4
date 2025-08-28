//! Auto-Tuning Module for Bot4 Trading System
//! Automatically adjusts system parameters based on market conditions

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use tracing::{info, warn, debug};
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCondition {
    pub timestamp: DateTime<Utc>,
    pub volatility: f64,
    pub volume: f64,
    pub spread: f64,
    pub trend: TrendDirection,
    pub regime: MarketRegime,
    pub liquidity_score: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    StrongUp,
    Up,
    Neutral,
    Down,
    StrongDown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
    Calm,
    Crisis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingParameters {
    // Risk parameters
    pub max_position_size: f64,
    pub stop_loss_percent: f64,
    pub take_profit_percent: f64,
    pub max_correlation: f64,
    pub kelly_fraction: f64,
    
    // Execution parameters
    pub order_timeout_ms: u64,
    pub max_slippage_bps: f64,
    pub min_order_size: f64,
    pub max_orders_per_second: f64,
    
    // ML parameters
    pub model_confidence_threshold: f64,
    pub feature_window_size: usize,
    pub prediction_horizon: usize,
    pub ensemble_weight_distribution: Vec<f64>,
    
    // System parameters
    pub data_batch_size: usize,
    pub websocket_reconnect_delay_ms: u64,
    pub cache_ttl_seconds: u64,
    pub parallel_workers: usize,
}

impl Default for TradingParameters {
    fn default() -> Self {
        Self {
            // Conservative defaults
            max_position_size: 0.02,
            stop_loss_percent: 2.0,
            take_profit_percent: 3.0,
            max_correlation: 0.7,
            kelly_fraction: 0.15,
            
            order_timeout_ms: 5000,
            max_slippage_bps: 10.0,
            min_order_size: 10.0,
            max_orders_per_second: 10.0,
            
            model_confidence_threshold: 0.65,
            feature_window_size: 100,
            prediction_horizon: 10,
            ensemble_weight_distribution: vec![0.4, 0.3, 0.3],
            
            data_batch_size: 1024,
            websocket_reconnect_delay_ms: 1000,
            cache_ttl_seconds: 60,
            parallel_workers: 4,
        }
    }
}

pub struct AutoTuner {
    current_params: Arc<RwLock<TradingParameters>>,
    market_history: Arc<RwLock<VecDeque<MarketCondition>>>,
    performance_history: Arc<RwLock<VecDeque<PerformanceRecord>>>,
    tuning_enabled: bool,
    parameter_bounds: ParameterBounds,
    last_tuning: Arc<RwLock<DateTime<Utc>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub timestamp: DateTime<Utc>,
    pub pnl: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub trades_executed: u32,
    pub avg_latency_ms: f64,
}

struct ParameterBounds {
    max_position_size: (f64, f64),
    stop_loss_percent: (f64, f64),
    take_profit_percent: (f64, f64),
    kelly_fraction: (f64, f64),
    order_timeout_ms: (u64, u64),
    model_confidence_threshold: (f64, f64),
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            max_position_size: (0.001, 0.05),
            stop_loss_percent: (0.5, 5.0),
            take_profit_percent: (1.0, 10.0),
            kelly_fraction: (0.05, 0.25),
            order_timeout_ms: (1000, 30000),
            model_confidence_threshold: (0.5, 0.9),
        }
    }
}

impl AutoTuner {
    pub fn new(enable_tuning: bool) -> Self {
        Self {
            current_params: Arc::new(RwLock::new(TradingParameters::default())),
            market_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            tuning_enabled: enable_tuning,
            parameter_bounds: ParameterBounds::default(),
            last_tuning: Arc::new(RwLock::new(Utc::now())),
        }
    }

    /// Update market conditions
    pub fn update_market_conditions(&self, condition: MarketCondition) -> Result<()> {
        let mut history = self.market_history.write();
        history.push_back(condition.clone());
        
        // Keep only last 1000 records
        if history.len() > 1000 {
            history.pop_front();
        }
        
        // Check if tuning needed
        if self.should_tune()? {
            self.tune_parameters()?;
        }
        
        Ok(())
    }

    /// Record performance metrics
    pub fn record_performance(&self, record: PerformanceRecord) -> Result<()> {
        let mut history = self.performance_history.write();
        history.push_back(record);
        
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }

    /// Check if parameters should be tuned
    fn should_tune(&self) -> Result<bool> {
        if !self.tuning_enabled {
            return Ok(false);
        }
        
        let last_tuning = *self.last_tuning.read();
        let time_since_tuning = Utc::now() - last_tuning;
        
        // Tune at most every 5 minutes
        if time_since_tuning < Duration::minutes(5) {
            return Ok(false);
        }
        
        let market_history = self.market_history.read();
        if market_history.len() < 50 {
            return Ok(false); // Not enough data
        }
        
        // Check for regime change
        let recent_regime = market_history.back().map(|m| m.regime);
        let prev_regime = market_history.iter().rev().nth(10).map(|m| m.regime);
        
        if recent_regime != prev_regime {
            info!("Market regime changed from {:?} to {:?}", prev_regime, recent_regime);
            return Ok(true);
        }
        
        // Check for significant volatility change
        let recent_vol: f64 = market_history.iter().rev().take(10)
            .map(|m| m.volatility).sum::<f64>() / 10.0;
        let prev_vol: f64 = market_history.iter().rev().skip(50).take(10)
            .map(|m| m.volatility).sum::<f64>() / 10.0;
        
        if (recent_vol - prev_vol).abs() / prev_vol > 0.3 {
            info!("Significant volatility change: {:.2}% -> {:.2}%", prev_vol * 100.0, recent_vol * 100.0);
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Tune parameters based on current conditions
    fn tune_parameters(&self) -> Result<()> {
        info!("Auto-tuning parameters based on market conditions");
        
        let market_history = self.market_history.read();
        let perf_history = self.performance_history.read();
        
        if market_history.is_empty() {
            return Ok(());
        }
        
        let current_condition = market_history.back().unwrap();
        let mut params = self.current_params.write();
        
        // Adjust based on market regime
        match current_condition.regime {
            MarketRegime::Trending => {
                // Increase position size in trending markets
                params.max_position_size = self.clamp_value(
                    params.max_position_size * 1.2,
                    self.parameter_bounds.max_position_size
                );
                params.take_profit_percent *= 1.5;
                params.model_confidence_threshold *= 0.95; // Be slightly less strict
                info!("Tuned for trending market");
            }
            MarketRegime::Volatile => {
                // Reduce risk in volatile markets
                params.max_position_size = self.clamp_value(
                    params.max_position_size * 0.7,
                    self.parameter_bounds.max_position_size
                );
                params.stop_loss_percent = self.clamp_value(
                    params.stop_loss_percent * 0.8,
                    self.parameter_bounds.stop_loss_percent
                );
                params.kelly_fraction = self.clamp_value(
                    params.kelly_fraction * 0.8,
                    self.parameter_bounds.kelly_fraction
                );
                params.model_confidence_threshold = self.clamp_value(
                    params.model_confidence_threshold * 1.1,
                    self.parameter_bounds.model_confidence_threshold
                );
                info!("Tuned for volatile market");
            }
            MarketRegime::Ranging => {
                // Optimize for mean reversion
                params.take_profit_percent = self.clamp_value(
                    params.take_profit_percent * 0.7,
                    self.parameter_bounds.take_profit_percent
                );
                params.stop_loss_percent = self.clamp_value(
                    params.stop_loss_percent * 1.2,
                    self.parameter_bounds.stop_loss_percent
                );
                info!("Tuned for ranging market");
            }
            MarketRegime::Crisis => {
                // Maximum risk reduction
                params.max_position_size = self.clamp_value(
                    params.max_position_size * 0.3,
                    self.parameter_bounds.max_position_size
                );
                params.kelly_fraction = self.clamp_value(
                    params.kelly_fraction * 0.5,
                    self.parameter_bounds.kelly_fraction
                );
                params.order_timeout_ms = self.clamp_value(
                    params.order_timeout_ms / 2,
                    self.parameter_bounds.order_timeout_ms
                );
                params.model_confidence_threshold = self.clamp_value(
                    0.8,
                    self.parameter_bounds.model_confidence_threshold
                );
                warn!("Crisis mode activated - maximum risk reduction");
            }
            MarketRegime::Calm => {
                // Standard parameters
                *params = TradingParameters::default();
                info!("Reset to default parameters for calm market");
            }
        }
        
        // Adjust based on recent performance
        if perf_history.len() >= 20 {
            let recent_sharpe: f64 = perf_history.iter().rev().take(20)
                .map(|p| p.sharpe_ratio).sum::<f64>() / 20.0;
            
            if recent_sharpe < 0.5 {
                // Poor performance - reduce risk
                params.max_position_size *= 0.9;
                params.kelly_fraction *= 0.9;
                warn!("Reducing risk due to poor Sharpe ratio: {:.2}", recent_sharpe);
            } else if recent_sharpe > 2.0 {
                // Good performance - can increase risk slightly
                params.max_position_size = self.clamp_value(
                    params.max_position_size * 1.05,
                    self.parameter_bounds.max_position_size
                );
                info!("Increasing position size due to good Sharpe ratio: {:.2}", recent_sharpe);
            }
        }
        
        // Update last tuning time
        *self.last_tuning.write() = Utc::now();
        
        info!("Parameters tuned: max_position={:.4}, stop_loss={:.2}%, confidence={:.2}",
              params.max_position_size, params.stop_loss_percent, params.model_confidence_threshold);
        
        Ok(())
    }

    fn clamp_value<T>(&self, value: T, bounds: (T, T)) -> T
    where
        T: PartialOrd,
    {
        if value < bounds.0 {
            bounds.0
        } else if value > bounds.1 {
            bounds.1
        } else {
            value
        }
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> TradingParameters {
        self.current_params.read().clone()
    }

    /// Get tuning statistics
    pub fn get_tuning_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let market_history = self.market_history.read();
        let perf_history = self.performance_history.read();
        let params = self.current_params.read();
        
        // Market stats
        if !market_history.is_empty() {
            let recent_market = market_history.back().unwrap();
            stats.insert("current_regime".to_string(), serde_json::json!(recent_market.regime));
            stats.insert("current_volatility".to_string(), serde_json::json!(recent_market.volatility));
            stats.insert("current_trend".to_string(), serde_json::json!(recent_market.trend));
        }
        
        // Performance stats
        if !perf_history.is_empty() {
            let recent_perf: Vec<_> = perf_history.iter().rev().take(20).collect();
            let avg_sharpe = recent_perf.iter().map(|p| p.sharpe_ratio).sum::<f64>() / recent_perf.len() as f64;
            let avg_win_rate = recent_perf.iter().map(|p| p.win_rate).sum::<f64>() / recent_perf.len() as f64;
            
            stats.insert("avg_sharpe_ratio".to_string(), serde_json::json!(avg_sharpe));
            stats.insert("avg_win_rate".to_string(), serde_json::json!(avg_win_rate));
        }
        
        // Current parameters
        stats.insert("max_position_size".to_string(), serde_json::json!(params.max_position_size));
        stats.insert("kelly_fraction".to_string(), serde_json::json!(params.kelly_fraction));
        stats.insert("confidence_threshold".to_string(), serde_json::json!(params.model_confidence_threshold));
        stats.insert("last_tuning".to_string(), serde_json::json!(*self.last_tuning.read()));
        
        stats
    }

    /// Force parameter tuning
    pub fn force_tune(&self) -> Result<()> {
        if !self.tuning_enabled {
            return Err(anyhow!("Auto-tuning is disabled"));
        }
        
        self.tune_parameters()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tuning_regime_change() {
        let tuner = AutoTuner::new(true);
        
        // Simulate calm market
        let calm_condition = MarketCondition {
            timestamp: Utc::now(),
            volatility: 0.01,
            volume: 1000000.0,
            spread: 0.001,
            trend: TrendDirection::Neutral,
            regime: MarketRegime::Calm,
            liquidity_score: 0.9,
        };
        
        tuner.update_market_conditions(calm_condition).unwrap();
        let params1 = tuner.get_parameters();
        
        // Simulate volatile market
        let volatile_condition = MarketCondition {
            timestamp: Utc::now(),
            volatility: 0.05,
            volume: 5000000.0,
            spread: 0.005,
            trend: TrendDirection::Down,
            regime: MarketRegime::Volatile,
            liquidity_score: 0.6,
        };
        
        // Add enough history to trigger tuning
        for _ in 0..60 {
            tuner.update_market_conditions(volatile_condition.clone()).unwrap();
        }
        
        // Force immediate tuning for test
        tuner.force_tune().unwrap();
        let params2 = tuner.get_parameters();
        
        // Parameters should have changed
        assert!(params2.max_position_size < params1.max_position_size);
        assert!(params2.kelly_fraction < params1.kelly_fraction);
    }

    #[test]
    fn test_performance_based_tuning() {
        let tuner = AutoTuner::new(true);
        
        // Record poor performance
        for i in 0..25 {
            tuner.record_performance(PerformanceRecord {
                timestamp: Utc::now(),
                pnl: -100.0,
                sharpe_ratio: 0.3,
                win_rate: 0.35,
                max_drawdown: 0.1,
                trades_executed: 10,
                avg_latency_ms: 50.0,
            }).unwrap();
        }
        
        // Market condition to trigger tuning
        let condition = MarketCondition {
            timestamp: Utc::now(),
            volatility: 0.02,
            volume: 2000000.0,
            spread: 0.002,
            trend: TrendDirection::Neutral,
            regime: MarketRegime::Ranging,
            liquidity_score: 0.8,
        };
        
        for _ in 0..60 {
            tuner.update_market_conditions(condition.clone()).unwrap();
        }
        
        tuner.force_tune().unwrap();
        let params = tuner.get_parameters();
        
        // Should have reduced risk due to poor performance
        assert!(params.max_position_size < TradingParameters::default().max_position_size);
    }
}