//! # ENHANCED RISK ENGINE - Production-Grade Implementation
//! Cameron (Risk Lead) + Full Team Collaboration
//! 
//! External Research Applied:
//! - "Risk Management in Electronic Trading" - Aldridge (2013)
//! - "The Kelly Capital Growth Investment Criterion" - MacLean et al. (2011)
//! - "Extreme Value Theory for Risk Managers" - Embrechts (2000)
//! - "Copula Methods in Finance" - Cherubini et al. (2004)
//! - "Dynamic Hedging" - Taleb (1997)

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use statrs::distribution::{Normal, ContinuousCDF};

/// Enhanced Risk Engine with Advanced Metrics
pub struct EnhancedRiskEngine {
    /// Portfolio state
    portfolio: Arc<RwLock<Portfolio>>,
    
    /// Risk limits (configurable, not hardcoded)
    config: RiskConfig,
    
    /// Real-time risk metrics
    metrics: Arc<RwLock<RiskMetrics>>,
    
    /// Correlation matrix (dynamic)
    correlations: Arc<RwLock<CorrelationMatrix>>,
    
    /// Stress test scenarios
    stress_scenarios: Vec<StressScenario>,
    
    /// Circuit breakers
    circuit_breakers: CircuitBreakerSet,
}

#[derive(Clone, Debug)]
pub struct RiskConfig {
    /// Maximum position size as % of portfolio
    pub max_position_pct: f64,  // Default: 0.02 (2%)
    
    /// Maximum daily loss
    pub max_daily_loss_pct: f64,  // Default: 0.02 (2%)
    
    /// Maximum leverage
    pub max_leverage: f64,  // Default: 3.0
    
    /// VaR confidence level
    pub var_confidence: f64,  // Default: 0.95
    
    /// Kelly fraction safety factor
    pub kelly_safety_factor: f64,  // Default: 0.25
    
    /// Correlation breakdown threshold
    pub correlation_breakdown_threshold: f64,  // Default: 0.8
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_pct: 0.02,
            max_daily_loss_pct: 0.02,
            max_leverage: 3.0,
            var_confidence: 0.95,
            kelly_safety_factor: 0.25,
            correlation_breakdown_threshold: 0.8,
        }
    }
}

// REMOVED: Duplicate
// pub struct RiskMetrics {
    /// Portfolio-level metrics
    pub portfolio_var: f64,
    pub portfolio_cvar: f64,
    pub portfolio_heat: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    
    /// Position-level metrics
    pub position_vars: HashMap<String, f64>,
    pub position_correlations: HashMap<(String, String), f64>,
    
    /// Dynamic metrics
    pub rolling_volatility: f64,
    pub regime_probability: MarketRegime,
    pub tail_risk_score: f64,
}

#[derive(Debug, Clone)]
pub enum MarketRegime {
    Normal { confidence: f64 },
    Trending { direction: f64, strength: f64 },
    Volatile { intensity: f64 },
    Crisis { severity: f64 },
}

impl EnhancedRiskEngine {
    /// Calculate position size using fractional Kelly
    pub fn calculate_kelly_position(
        &self,
        signal: &TradingSignal,
        current_portfolio_value: f64,
    ) -> Result<f64, RiskError> {
        // Kelly formula: f* = (p(b+1) - 1) / b
        // where p = probability of win, b = odds
        
        let p = signal.win_probability;
        let b = signal.expected_return / signal.expected_loss;
        
        // Full Kelly
        let full_kelly = (p * (b + 1.0) - 1.0) / b;
        
        // Apply safety factor (Cameron: "Never full Kelly in production")
        let safe_kelly = full_kelly * self.config.kelly_safety_factor;
        
        // Apply maximum position limit
        let max_position = current_portfolio_value * self.config.max_position_pct;
        let kelly_position = current_portfolio_value * safe_kelly;
        
        Ok(kelly_position.min(max_position))
    }
    
    /// Calculate portfolio VaR using Cornish-Fisher expansion
    pub fn calculate_portfolio_var_cornish_fisher(
        &self,
        returns: &[f64],
        confidence: f64,
    ) -> f64 {
        // Calculate moments
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        // Skewness
        let skewness = returns.iter()
            .map(|r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>() / n;
        
        // Excess kurtosis
        let kurtosis = returns.iter()
            .map(|r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0;
        
        // Standard normal quantile
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - confidence);
        
        // Cornish-Fisher expansion
        let cf_quantile = z 
            + (z.powi(2) - 1.0) * skewness / 6.0
            + (z.powi(3) - 3.0 * z) * kurtosis / 24.0
            - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;
        
        // VaR = mean - cf_quantile * std_dev
        -(mean + cf_quantile * std_dev)
    }
    
    /// Detect correlation breakdown using eigenvalue analysis
    pub fn detect_correlation_breakdown(&self) -> bool {
        let corr_matrix = self.correlations.read();
        
        // Calculate condition number (ratio of largest to smallest eigenvalue)
        // High condition number indicates correlation breakdown
        let eigenvalues = self.calculate_eigenvalues(&corr_matrix);
        
        if let (Some(max_eigen), Some(min_eigen)) = 
            (eigenvalues.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
             eigenvalues.iter().filter(|&&e| e > 1e-10).min_by(|a, b| a.partial_cmp(b).unwrap())) {
            
            let condition_number = max_eigen / min_eigen;
            condition_number > self.config.correlation_breakdown_threshold
        } else {
            false
        }
    }
    
    /// Run stress test scenarios
    pub fn run_stress_tests(&self) -> Vec<StressTestResult> {
        let mut results = Vec::new();
        
        for scenario in &self.stress_scenarios {
            let result = self.apply_stress_scenario(scenario);
            results.push(result);
        }
        
        results
    }
    
    /// Calculate tail risk using Extreme Value Theory
    pub fn calculate_tail_risk(&self, returns: &[f64], threshold_percentile: f64) -> TailRisk {
        // Implement POT (Peaks Over Threshold) method
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let threshold_idx = (returns.len() as f64 * threshold_percentile) as usize;
        let threshold = sorted_returns[threshold_idx];
        
        // Extract exceedances
        let exceedances: Vec<f64> = sorted_returns.iter()
            .filter(|&&r| r < threshold)
            .map(|&r| threshold - r)
            .collect();
        
        // Fit GPD (Generalized Pareto Distribution) parameters
        let (xi, beta) = self.fit_gpd(&exceedances);
        
        TailRisk {
            threshold,
            shape_parameter: xi,
            scale_parameter: beta,
            expected_shortfall: self.calculate_expected_shortfall(xi, beta, threshold),
            tail_index: 1.0 / xi,  // Indicates heaviness of tails
        }
    }
    
    /// Dynamic hedge ratio calculation using cointegration
    pub fn calculate_dynamic_hedge_ratio(
        &self,
        asset_returns: &[f64],
        hedge_returns: &[f64],
    ) -> f64 {
        // Use OLS regression for hedge ratio
        let n = asset_returns.len() as f64;
        
        let mean_asset = asset_returns.iter().sum::<f64>() / n;
        let mean_hedge = hedge_returns.iter().sum::<f64>() / n;
        
        let covariance: f64 = asset_returns.iter()
            .zip(hedge_returns.iter())
            .map(|(a, h)| (a - mean_asset) * (h - mean_hedge))
            .sum::<f64>() / (n - 1.0);
        
        let hedge_variance: f64 = hedge_returns.iter()
            .map(|h| (h - mean_hedge).powi(2))
            .sum::<f64>() / (n - 1.0);
        
        // Hedge ratio = Cov(Asset, Hedge) / Var(Hedge)
        covariance / hedge_variance
    }
    
    /// Real-time portfolio heat calculation
    pub fn calculate_portfolio_heat(&self) -> f64 {
        let portfolio = self.portfolio.read();
        
        // Heat = sum of (position_size / avg_volume)^2
        let mut heat = 0.0;
        
        for position in &portfolio.positions {
            let market_data = self.get_market_data(&position.symbol);
            let avg_volume = market_data.avg_daily_volume;
            
            let position_heat = (position.size / avg_volume).powi(2);
            heat += position_heat;
        }
        
        heat.sqrt()  // Return square root for interpretability
    }
    
    /// Check all risk limits
    pub fn validate_order(&self, order: &Order) -> Result<(), RiskViolation> {
        // Check position limit
        if order.value > self.portfolio.read().total_value * self.config.max_position_pct {
            return Err(RiskViolation::PositionLimitExceeded);
        }
        
        // Check leverage
        let current_leverage = self.calculate_leverage();
        if current_leverage > self.config.max_leverage {
            return Err(RiskViolation::LeverageExceeded);
        }
        
        // Check daily loss
        if self.metrics.read().daily_pnl < -self.config.max_daily_loss_pct {
            return Err(RiskViolation::DailyLossLimitExceeded);
        }
        
        // Check correlation breakdown
        if self.detect_correlation_breakdown() {
            return Err(RiskViolation::CorrelationBreakdown);
        }
        
        // Check portfolio heat
        if self.calculate_portfolio_heat() > 0.5 {
            return Err(RiskViolation::PortfolioOverheated);
        }
        
        Ok(())
    }
}

// Supporting structures
pub struct Portfolio {
    pub positions: Vec<Position>,
    pub total_value: f64,
    pub cash: f64,
    pub margin_used: f64,
}

// Using canonical Position from domain_types
use domain_types::position_canonical::Position;

pub struct TradingSignal {
    pub symbol: String,
    pub win_probability: f64,
    pub expected_return: f64,
    pub expected_loss: f64,
    pub confidence: f64,
}

pub struct TailRisk {
    pub threshold: f64,
    pub shape_parameter: f64,
    pub scale_parameter: f64,
    pub expected_shortfall: f64,
    pub tail_index: f64,
}

#[derive(Debug)]
pub enum RiskViolation {
    PositionLimitExceeded,
    LeverageExceeded,
    DailyLossLimitExceeded,
    CorrelationBreakdown,
    PortfolioOverheated,
}

// CAMERON: "This is production-grade risk management. No shortcuts!"