// Comprehensive 8-Layer Risk Clamp System
// Quinn (Risk Lead) + Sam (Implementation)
// CRITICAL: Sophia Requirement #4 - Multiple safety layers
// References: Kelly Criterion, Markowitz, Taleb's "Antifragile"

use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use log::info;

const MIN_TRADE_SIZE: f32 = 0.001;  // Minimum BTC trade size
const CRISIS_REDUCTION: f32 = 0.3;  // Reduce to 30% in crisis
const MAX_CORRELATION: f32 = 0.7;   // Correlation threshold

/// Comprehensive Risk Clamp System
/// 8 sequential layers of risk control to prevent catastrophic losses
/// Quinn: "Each layer is independent - if ANY triggers, position is reduced!"
#[derive(Debug, Clone)]
pub struct RiskClampSystem {
    // Risk parameters
    config: ClampConfig,
    
    // State tracking
    current_var: f32,
    current_es: f32,
    portfolio_positions: Vec<Position>,
    
    // Crisis detection
    crisis_indicators: CrisisIndicators,
    
    // Metrics
    clamp_triggers: Arc<RwLock<ClampMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClampConfig {
    /// Target volatility (e.g., 20% annualized)
    pub vol_target: f32,
    /// Value at Risk limit (e.g., 2% daily)
    pub var_limit: f32,
    /// Expected Shortfall limit (CVaR)
    pub es_limit: f32,
    /// Portfolio heat capacity (e.g., 0.8)
    pub heat_cap: f32,
    /// Maximum leverage (e.g., 3x)
    pub leverage_cap: f32,
    /// Correlation penalty threshold
    pub correlation_threshold: f32,
}

impl Default for ClampConfig {
    fn default() -> Self {
        Self {
            vol_target: 0.20,              // 20% annualized
            var_limit: 0.02,               // 2% daily VaR
            es_limit: 0.03,                // 3% daily ES
            heat_cap: 0.8,                 // 80% heat capacity
            leverage_cap: 3.0,             // 3x max leverage
            correlation_threshold: 0.7,    // 70% correlation threshold
        }
    }
}

#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    size: f32,
    entry_price: f32,
    current_price: f32,
    pnl: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CrisisIndicators {
    vix_spike: bool,
    volume_surge: bool,
    correlation_breakdown: bool,
    bid_ask_spread_widening: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClampMetrics {
    pub vol_clamps: u64,
    pub var_clamps: u64,
    pub es_clamps: u64,
    pub heat_clamps: u64,
    pub correlation_clamps: u64,
    pub leverage_clamps: u64,
    pub crisis_clamps: u64,
    pub min_size_filters: u64,
}

impl RiskClampSystem {
    pub fn new(config: ClampConfig) -> Self {
        Self {
            config,
            current_var: 0.0,
            current_es: 0.0,
            portfolio_positions: Vec::new(),
            crisis_indicators: CrisisIndicators::default(),
            clamp_triggers: Arc::new(RwLock::new(ClampMetrics::default())),
        }
    }
    
    /// Calculate position size with 8 layers of risk control
    /// CRITICAL: This prevents oversizing that leads to catastrophic losses!
    pub fn calculate_position_size(
        &mut self,
        ml_confidence: f32,
        current_volatility: f32,
        portfolio_heat: f32,
        correlation: f32,
        account_equity: f32,
    ) -> f32 {
        info!("=== Risk Clamp System: Starting 8-Layer Analysis ===");
        
        // Layer 0: Basic calibration (simplified without isotonic)
        let calibrated = self.simple_calibration(ml_confidence);
        
        info!("Layer 0 - Calibration: raw={:.3} -> calibrated={:.3}", 
              ml_confidence, calibrated);
        
        // Convert to directional signal [-1, 1]
        let base_signal = (2.0 * calibrated - 1.0).clamp(-1.0, 1.0);
        
        // Layer 1: Volatility targeting (simplified without GARCH)
        let vol_ratio = (self.config.vol_target / current_volatility).min(1.5);
        let vol_adjusted = base_signal * vol_ratio;
        
        if vol_ratio < 1.0 {
            self.clamp_triggers.write().vol_clamps += 1;
            info!("Layer 1 - Vol Target TRIGGERED: ratio={:.3}", vol_ratio);
        }
        
        // Layer 2: Value at Risk (VaR) constraint
        self.update_var_es();
        let var_ratio = (1.0 - (self.current_var / self.config.var_limit)).max(0.0);
        let var_adjusted = vol_adjusted * var_ratio;
        
        if var_ratio < 1.0 {
            self.clamp_triggers.write().var_clamps += 1;
            info!("Layer 2 - VaR Limit TRIGGERED: ratio={:.3}", var_ratio);
        }
        
        // Layer 3: Expected Shortfall (CVaR) constraint
        let es_ratio = (1.0 - (self.current_es / self.config.es_limit)).max(0.0);
        let es_adjusted = var_adjusted * es_ratio;
        
        if es_ratio < 1.0 {
            self.clamp_triggers.write().es_clamps += 1;
            info!("Layer 3 - ES Limit TRIGGERED: ratio={:.3}", es_ratio);
        }
        
        // Layer 4: Portfolio heat constraint
        let heat_ratio = (1.0 - (portfolio_heat / self.config.heat_cap)).max(0.0);
        let heat_adjusted = es_adjusted * heat_ratio;
        
        if heat_ratio < 1.0 {
            self.clamp_triggers.write().heat_clamps += 1;
            info!("Layer 4 - Heat Cap TRIGGERED: ratio={:.3}", heat_ratio);
        }
        
        // Layer 5: Correlation penalty (diversification)
        let corr_adjusted = if correlation > self.config.correlation_threshold {
            self.clamp_triggers.write().correlation_clamps += 1;
            let penalty = 1.0 - (correlation - self.config.correlation_threshold) 
                              / (1.0 - self.config.correlation_threshold);
            info!("Layer 5 - Correlation TRIGGERED: penalty={:.3}", penalty);
            heat_adjusted * penalty
        } else {
            heat_adjusted
        };
        
        // Layer 6: Leverage constraint
        let current_leverage = self.calculate_leverage();
        let leverage_adjusted = if current_leverage > self.config.leverage_cap * 0.8 {
            self.clamp_triggers.write().leverage_clamps += 1;
            let reduction = (self.config.leverage_cap - current_leverage) / self.config.leverage_cap;
            info!("Layer 6 - Leverage TRIGGERED: reduction={:.3}", reduction);
            corr_adjusted * reduction.max(0.0)
        } else {
            corr_adjusted
        };
        
        // Layer 7: Crisis mode detection
        let crisis_adjusted = if self.detect_crisis() {
            self.clamp_triggers.write().crisis_clamps += 1;
            info!("Layer 7 - CRISIS MODE ACTIVATED: reducing to {:.0}%", CRISIS_REDUCTION * 100.0);
            leverage_adjusted * CRISIS_REDUCTION
        } else {
            leverage_adjusted
        };
        
        // Layer 8: Minimum trade size filter
        let position_value = crisis_adjusted.abs() * account_equity;
        let final_size = if position_value < MIN_TRADE_SIZE * 50000.0 {  // Assume BTC ~$50k
            self.clamp_triggers.write().min_size_filters += 1;
            info!("Layer 8 - MIN SIZE: Position too small, zeroing");
            0.0
        } else {
            crisis_adjusted
        };
        
        info!("=== Final Position Size: {:.3} ({:.1}% of max) ===", 
              final_size, final_size.abs() * 100.0);
        
        final_size
    }
    
    /// Simple calibration without isotonic regression
    fn simple_calibration(&self, raw_confidence: f32) -> f32 {
        // Simple logistic calibration
        let x = (raw_confidence - 0.5) * 4.0;  // Scale to [-2, 2]
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Detect market regime (simplified)
    fn detect_regime(&self) -> MarketRegime {
        if self.crisis_indicators.vix_spike {
            MarketRegime::Crisis
        } else if self.crisis_indicators.volume_surge {
            MarketRegime::Volatile
        } else {
            MarketRegime::Normal
        }
    }
    
    /// Update VaR and ES estimates
    fn update_var_es(&mut self) {
        // Simplified calculation using portfolio positions
        if self.portfolio_positions.is_empty() {
            self.current_var = 0.0;
            self.current_es = 0.0;
            return;
        }
        
        let total_pnl: f32 = self.portfolio_positions.iter()
            .map(|p| p.pnl)
            .sum();
        
        let portfolio_value: f32 = self.portfolio_positions.iter()
            .map(|p| p.size * p.current_price)
            .sum();
        
        if portfolio_value > 0.0 {
            // Simplified VaR (95% confidence)
            self.current_var = (total_pnl / portfolio_value).abs() * 1.65;
            // Simplified ES (conditional VaR)
            self.current_es = self.current_var * 1.2;
        }
    }
    
    /// Calculate current leverage
    fn calculate_leverage(&self) -> f32 {
        if self.portfolio_positions.is_empty() {
            return 0.0;
        }
        
        let notional: f32 = self.portfolio_positions.iter()
            .map(|p| (p.size * p.current_price).abs())
            .sum();
        
        let equity = 100000.0;  // Placeholder - should come from account
        notional / equity
    }
    
    /// Detect crisis conditions
    fn detect_crisis(&self) -> bool {
        self.crisis_indicators.vix_spike ||
        self.crisis_indicators.correlation_breakdown ||
        self.crisis_indicators.bid_ask_spread_widening > 0.005
    }
    
    /// Update crisis indicators
    pub fn update_crisis_indicators(
        &mut self,
        vix: f32,
        volume_ratio: f32,
        correlation_matrix_det: f32,
        avg_spread: f32,
    ) {
        self.crisis_indicators.vix_spike = vix > 30.0;
        self.crisis_indicators.volume_surge = volume_ratio > 2.0;
        self.crisis_indicators.correlation_breakdown = correlation_matrix_det < 0.1;
        self.crisis_indicators.bid_ask_spread_widening = avg_spread;
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> ClampMetrics {
        self.clamp_triggers.read().clone()
    }
    
    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        *self.clamp_triggers.write() = ClampMetrics::default();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MarketRegime {
    Normal,
    Volatile,
    Crisis,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clamp_system_creation() {
        let config = ClampConfig::default();
        let mut system = RiskClampSystem::new(config);
        
        // Test with normal conditions
        let size = system.calculate_position_size(
            0.7,      // 70% ML confidence
            0.15,     // 15% volatility
            0.5,      // 50% portfolio heat
            0.5,      // 50% correlation
            100000.0, // $100k account
        );
        
        assert!(size > 0.0);
        assert!(size <= 1.0);
    }
    
    #[test]
    fn test_crisis_mode() {
        let config = ClampConfig::default();
        let mut system = RiskClampSystem::new(config);
        
        // Trigger crisis mode
        system.update_crisis_indicators(
            35.0,  // High VIX
            3.0,   // Volume surge
            0.05,  // Correlation breakdown
            0.01,  // Wide spreads
        );
        
        let size = system.calculate_position_size(
            0.9,      // High confidence
            0.15,     // Normal volatility
            0.5,      // Normal heat
            0.5,      // Normal correlation
            100000.0,
        );
        
        // Should be reduced due to crisis
        assert!(size <= CRISIS_REDUCTION);
    }
    
    #[test]
    fn test_min_size_filter() {
        let config = ClampConfig::default();
        let mut system = RiskClampSystem::new(config);
        
        // Very small signal
        let size = system.calculate_position_size(
            0.51,     // Near neutral confidence
            0.15,     // Normal volatility
            0.5,      // Normal heat
            0.5,      // Normal correlation
            1000.0,   // Small account
        );
        
        // Should be filtered to zero
        assert_eq!(size, 0.0);
        assert!(system.get_metrics().min_size_filters > 0);
    }
}