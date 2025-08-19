// Comprehensive 8-Layer Risk Clamp System
// Quinn (Risk Lead) + Sam (Implementation)
// CRITICAL: Sophia Requirement #4 - Multiple safety layers
// References: Kelly Criterion, Markowitz, Taleb's "Antifragile"

use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use crate::garch::GARCHModel;
use crate::calibration::isotonic::{IsotonicCalibrator, MarketRegime};

const MIN_TRADE_SIZE: f32 = 0.001;  // Minimum BTC trade size
const CRISIS_REDUCTION: f32 = 0.3;  // Reduce to 30% in crisis
const MAX_CORRELATION: f32 = 0.7;   // Correlation threshold

/// Comprehensive Risk Clamp System
/// 8 sequential layers of risk control to prevent catastrophic losses
/// Quinn: "Each layer is independent - if ANY triggers, position is reduced!"
#[derive(Debug, Clone)]
pub struct RiskClampSystem {
    // Risk parameters
    vol_target: f32,              // Target volatility (e.g., 20% annualized)
    var_limit: f32,               // Value at Risk limit (e.g., 2% daily)
    es_limit: f32,                // Expected Shortfall limit (e.g., 3% daily)
    heat_cap: f32,                // Portfolio heat capacity (e.g., 0.8)
    leverage_cap: f32,            // Maximum leverage (e.g., 3x)
    correlation_threshold: f32,   // Correlation penalty threshold
    
    // Models
    calibrator: Arc<RwLock<IsotonicCalibrator>>,
    garch: Arc<RwLock<GARCHModel>>,
    
    // State tracking
    current_var: f32,
    current_es: f32,
    portfolio_positions: Vec<Position>,
    
    // Crisis detection
    crisis_indicators: CrisisIndicators,
    
    // Metrics
    clamp_triggers: ClampMetrics,
}

#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    size: f32,
    entry_price: f32,
    current_price: f32,
    correlation_to_portfolio: f32,
}

#[derive(Debug, Clone, Default)]
struct CrisisIndicators {
    vix_level: f32,
    correlation_breakdown: bool,
    volume_spike_ratio: f32,
    drawdown_severity: f32,
    bid_ask_spread_widening: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ClampMetrics {
    vol_clamps: u64,
    var_clamps: u64,
    es_clamps: u64,
    heat_clamps: u64,
    correlation_clamps: u64,
    leverage_clamps: u64,
    crisis_clamps: u64,
    min_size_filters: u64,
}

impl RiskClampSystem {
    pub fn new(
        calibrator: Arc<RwLock<IsotonicCalibrator>>,
        garch: Arc<RwLock<GARCHModel>>,
    ) -> Self {
        Self {
            vol_target: 0.20,  // 20% annualized
            var_limit: 0.02,   // 2% daily VaR
            es_limit: 0.03,    // 3% daily ES
            heat_cap: 0.8,     // 80% portfolio heat
            leverage_cap: 3.0,  // 3x max leverage
            correlation_threshold: MAX_CORRELATION,
            calibrator,
            garch,
            current_var: 0.0,
            current_es: 0.0,
            portfolio_positions: Vec::new(),
            crisis_indicators: CrisisIndicators::default(),
            clamp_triggers: ClampMetrics::default(),
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
        
        // Layer 0: Calibrated probability (prevents overconfidence)
        let regime = self.detect_regime();
        let calibrated = self.calibrator.read()
            .transform(ml_confidence, regime);
        
        info!("Layer 0 - Calibration: raw={:.3} -> calibrated={:.3}", 
              ml_confidence, calibrated);
        
        // Convert to directional signal [-1, 1]
        let base_signal = (2.0 * calibrated - 1.0).clamp(-1.0, 1.0);
        
        // Layer 1: Volatility targeting with GARCH forecast
        let garch_vol = self.garch.read().forecast(1)[0];
        let vol_ratio = (self.vol_target / garch_vol).min(1.5);
        let vol_adjusted = base_signal * vol_ratio;
        
        if vol_ratio < 1.0 {
            self.clamp_triggers.vol_clamps += 1;
            info!("Layer 1 - Vol Target TRIGGERED: ratio={:.3}", vol_ratio);
        }
        
        // Layer 2: Value at Risk (VaR) constraint
        self.update_var_es();
        let var_ratio = (1.0 - (self.current_var / self.var_limit)).max(0.0);
        let var_adjusted = vol_adjusted * var_ratio;
        
        if var_ratio < 1.0 {
            self.clamp_triggers.var_clamps += 1;
            info!("Layer 2 - VaR Limit TRIGGERED: ratio={:.3}", var_ratio);
        }
        
        // Layer 3: Expected Shortfall (CVaR) constraint
        let es_ratio = (1.0 - (self.current_es / self.es_limit)).max(0.0);
        let es_adjusted = var_adjusted * es_ratio;
        
        if es_ratio < 1.0 {
            self.clamp_triggers.es_clamps += 1;
            info!("Layer 3 - ES Limit TRIGGERED: ratio={:.3}", es_ratio);
        }
        
        // Layer 4: Portfolio heat constraint
        let heat_ratio = (1.0 - (portfolio_heat / self.heat_cap)).max(0.0);
        let heat_adjusted = es_adjusted * heat_ratio;
        
        if heat_ratio < 1.0 {
            self.clamp_triggers.heat_clamps += 1;
            info!("Layer 4 - Heat Cap TRIGGERED: ratio={:.3}", heat_ratio);
        }
        
        // Layer 5: Correlation penalty (diversification)
        let corr_adjusted = if correlation > self.correlation_threshold {
            let penalty = 1.0 - (correlation - self.correlation_threshold);
            self.clamp_triggers.correlation_clamps += 1;
            info!("Layer 5 - Correlation TRIGGERED: penalty={:.3}", penalty);
            heat_adjusted * penalty
        } else {
            heat_adjusted
        };
        
        // Layer 6: Leverage cap
        let max_position = self.leverage_cap * account_equity;
        let leverage_adjusted = corr_adjusted.abs().min(max_position) * corr_adjusted.signum();
        
        if leverage_adjusted.abs() < corr_adjusted.abs() {
            self.clamp_triggers.leverage_clamps += 1;
            info!("Layer 6 - Leverage Cap TRIGGERED");
        }
        
        // Layer 7: Crisis override (nuclear option)
        let final_size = if self.detect_crisis() {
            self.clamp_triggers.crisis_clamps += 1;
            warn!("Layer 7 - CRISIS MODE ACTIVATED! Reducing to 30%");
            leverage_adjusted * CRISIS_REDUCTION
        } else {
            leverage_adjusted
        };
        
        // Layer 8: Minimum size filter
        if final_size.abs() < MIN_TRADE_SIZE {
            self.clamp_triggers.min_size_filters += 1;
            info!("Layer 8 - Below minimum size, zeroing position");
            0.0
        } else {
            info!("=== Final Position Size: {:.4} ===", final_size);
            final_size
        }
    }
    
    /// Update VaR and ES calculations
    fn update_var_es(&mut self) {
        if self.portfolio_positions.is_empty() {
            self.current_var = 0.0;
            self.current_es = 0.0;
            return;
        }
        
        // Calculate portfolio VaR using historical simulation
        let portfolio_value: f32 = self.portfolio_positions.iter()
            .map(|p| p.size * p.current_price)
            .sum();
        
        // Get GARCH volatility forecast
        let vol_forecast = self.garch.read().forecast(1)[0];
        
        // Calculate VaR (95% confidence)
        self.current_var = portfolio_value * vol_forecast * 1.645;  // Normal approximation
        
        // Calculate Expected Shortfall (CVaR)
        self.current_es = self.current_var * 1.2;  // ES typically 20% higher than VaR
    }
    
    /// Detect market regime for calibration
    fn detect_regime(&self) -> MarketRegime {
        let vol = self.garch.read().forecast(1)[0];
        let trend_strength = self.calculate_trend_strength();
        
        if vol > 0.05 {
            MarketRegime::Crisis
        } else if trend_strength > 0.7 {
            MarketRegime::Trending
        } else if trend_strength > 0.4 {
            MarketRegime::Breakout
        } else {
            MarketRegime::RangeBound
        }
    }
    
    /// Detect crisis conditions
    /// Quinn: "Multiple indicators must align for crisis detection"
    fn detect_crisis(&self) -> bool {
        let mut crisis_score = 0.0;
        
        // VIX spike (fear gauge)
        if self.crisis_indicators.vix_level > 30.0 {
            crisis_score += 0.3;
        }
        
        // Correlation breakdown (everything moving together)
        if self.crisis_indicators.correlation_breakdown {
            crisis_score += 0.3;
        }
        
        // Volume spike (panic trading)
        if self.crisis_indicators.volume_spike_ratio > 3.0 {
            crisis_score += 0.2;
        }
        
        // Severe drawdown
        if self.crisis_indicators.drawdown_severity > 0.15 {
            crisis_score += 0.2;
        }
        
        // Bid-ask spread widening (liquidity crisis)
        if self.crisis_indicators.bid_ask_spread_widening > 2.0 {
            crisis_score += 0.2;
        }
        
        crisis_score >= 0.5  // Need multiple indicators
    }
    
    /// Calculate trend strength for regime detection
    fn calculate_trend_strength(&self) -> f32 {
        // Simplified trend strength (would use actual price data)
        0.5
    }
    
    /// Update crisis indicators from market data
    pub fn update_crisis_indicators(
        &mut self,
        vix: f32,
        avg_correlation: f32,
        volume_ratio: f32,
        drawdown: f32,
        spread_ratio: f32,
    ) {
        self.crisis_indicators.vix_level = vix;
        self.crisis_indicators.correlation_breakdown = avg_correlation > 0.8;
        self.crisis_indicators.volume_spike_ratio = volume_ratio;
        self.crisis_indicators.drawdown_severity = drawdown;
        self.crisis_indicators.bid_ask_spread_widening = spread_ratio;
    }
    
    /// Add position to portfolio tracking
    pub fn add_position(&mut self, position: Position) {
        self.portfolio_positions.push(position);
        self.update_var_es();
    }
    
    /// Remove position from portfolio
    pub fn remove_position(&mut self, symbol: &str) {
        self.portfolio_positions.retain(|p| p.symbol != symbol);
        self.update_var_es();
    }
    
    /// Get current risk metrics
    pub fn get_risk_metrics(&self) -> RiskMetrics {
        RiskMetrics {
            current_var: self.current_var,
            current_es: self.current_es,
            var_utilization: self.current_var / self.var_limit,
            es_utilization: self.current_es / self.es_limit,
            portfolio_heat: self.calculate_portfolio_heat(),
            is_crisis: self.detect_crisis(),
            clamp_metrics: self.clamp_triggers.clone(),
        }
    }
    
    /// Calculate portfolio heat (concentration risk)
    fn calculate_portfolio_heat(&self) -> f32 {
        if self.portfolio_positions.is_empty() {
            return 0.0;
        }
        
        let total_value: f32 = self.portfolio_positions.iter()
            .map(|p| (p.size * p.current_price).abs())
            .sum();
        
        // Herfindahl index for concentration
        let herfindahl: f32 = self.portfolio_positions.iter()
            .map(|p| {
                let weight = (p.size * p.current_price).abs() / total_value;
                weight * weight
            })
            .sum();
        
        herfindahl.sqrt()  // Normalized concentration
    }
}

/// Risk metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub current_var: f32,
    pub current_es: f32,
    pub var_utilization: f32,
    pub es_utilization: f32,
    pub portfolio_heat: f32,
    pub is_crisis: bool,
    pub clamp_metrics: ClampMetrics,
}

/// Kelly Criterion calculator with safety factor
/// CRITICAL: Prevents overbetting based on edge estimation
pub struct KellyCriterion {
    safety_factor: f32,  // Typically 0.25 (quarter Kelly)
    max_fraction: f32,   // Maximum Kelly fraction allowed
}

impl KellyCriterion {
    pub fn new() -> Self {
        Self {
            safety_factor: 0.25,  // Conservative quarter Kelly
            max_fraction: 0.5,    // Never bet more than 50%
        }
    }
    
    /// Calculate Kelly fraction with safety adjustments
    /// f* = (p*b - q) / b
    /// where p = win probability, q = loss probability, b = win/loss ratio
    pub fn calculate(
        &self,
        win_probability: f32,
        risk_reward_ratio: f32,
    ) -> f32 {
        if win_probability <= 0.0 || win_probability >= 1.0 {
            return 0.0;
        }
        
        if risk_reward_ratio <= 0.0 {
            return 0.0;
        }
        
        let loss_probability = 1.0 - win_probability;
        
        // Kelly formula
        let kelly_fraction = (win_probability * risk_reward_ratio - loss_probability) 
                           / risk_reward_ratio;
        
        // Apply safety factor (fractional Kelly)
        let safe_fraction = kelly_fraction * self.safety_factor;
        
        // Cap at maximum
        safe_fraction.min(self.max_fraction).max(0.0)
    }
    
    /// Calculate with estimation uncertainty
    /// Adjusts for parameter uncertainty using Bayesian approach
    pub fn calculate_with_uncertainty(
        &self,
        win_probability: f32,
        win_probability_std: f32,
        risk_reward_ratio: f32,
        risk_reward_std: f32,
    ) -> f32 {
        // Adjust for uncertainty (conservative approach)
        let adjusted_win_prob = win_probability - win_probability_std;  // 1 std dev penalty
        let adjusted_rr = risk_reward_ratio - risk_reward_std;
        
        self.calculate(adjusted_win_prob.max(0.0), adjusted_rr.max(0.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_system() -> RiskClampSystem {
        let calibrator = Arc::new(RwLock::new(IsotonicCalibrator::new()));
        let garch = Arc::new(RwLock::new(GARCHModel::new()));
        RiskClampSystem::new(calibrator, garch)
    }
    
    #[test]
    fn test_all_clamps_trigger() {
        let mut system = create_test_system();
        
        // Set extreme conditions to trigger all clamps
        system.current_var = 0.019;  // Near VaR limit
        system.current_es = 0.029;   // Near ES limit
        
        let size = system.calculate_position_size(
            0.9,    // High confidence
            0.1,    // High volatility
            0.79,   // High heat
            0.8,    // High correlation
            10000.0 // Account equity
        );
        
        // Should be heavily reduced
        assert!(size < 1000.0, "Position should be heavily clamped");
        
        // Check that clamps were triggered
        assert!(system.clamp_triggers.var_clamps > 0);
        assert!(system.clamp_triggers.es_clamps > 0);
        assert!(system.clamp_triggers.heat_clamps > 0);
        assert!(system.clamp_triggers.correlation_clamps > 0);
    }
    
    #[test]
    fn test_crisis_detection() {
        let mut system = create_test_system();
        
        // Normal conditions
        system.update_crisis_indicators(15.0, 0.3, 1.5, 0.05, 1.2);
        assert!(!system.detect_crisis());
        
        // Crisis conditions
        system.update_crisis_indicators(35.0, 0.85, 4.0, 0.20, 2.5);
        assert!(system.detect_crisis());
    }
    
    #[test]
    fn test_minimum_size_filter() {
        let mut system = create_test_system();
        
        let size = system.calculate_position_size(
            0.51,   // Barely positive
            0.02,   // Low volatility
            0.1,    // Low heat
            0.1,    // Low correlation
            100.0   // Small account
        );
        
        // Should be filtered to zero if below minimum
        if size != 0.0 {
            assert!(size.abs() >= MIN_TRADE_SIZE);
        }
    }
    
    #[test]
    fn test_kelly_criterion() {
        let kelly = KellyCriterion::new();
        
        // Favorable bet: 60% win, 2:1 reward
        let fraction = kelly.calculate(0.6, 2.0);
        assert!(fraction > 0.0 && fraction < 0.5);
        
        // Unfavorable bet: 40% win, 1:1 reward
        let fraction = kelly.calculate(0.4, 1.0);
        assert_eq!(fraction, 0.0);
        
        // With uncertainty
        let fraction = kelly.calculate_with_uncertainty(0.6, 0.1, 2.0, 0.5);
        assert!(fraction < kelly.calculate(0.6, 2.0));  // More conservative
    }
    
    #[test]
    fn test_portfolio_heat_calculation() {
        let mut system = create_test_system();
        
        // Concentrated portfolio
        system.add_position(Position {
            symbol: "BTC".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            current_price: 50000.0,
            correlation_to_portfolio: 1.0,
        });
        
        let heat = system.calculate_portfolio_heat();
        assert_eq!(heat, 1.0, "Single position should have heat of 1.0");
        
        // Diversified portfolio
        system.add_position(Position {
            symbol: "ETH".to_string(),
            size: 10.0,
            entry_price: 3000.0,
            current_price: 3000.0,
            correlation_to_portfolio: 0.5,
        });
        
        let heat = system.calculate_portfolio_heat();
        assert!(heat < 1.0, "Diversified portfolio should have lower heat");
    }
}