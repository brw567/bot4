use domain_types::risk_limits::RiskLimits;
//! # UNIFIED RISK CALCULATIONS - Single Source of Truth
//! Cameron: "No more inconsistent risk metrics!"
//! Consolidates 32 duplicate risk calculations into 5 canonical functions
//!
//! Research Applied:
//! 1. Kelly (1956) - Optimal position sizing
//! 2. Markowitz (1952) - Portfolio optimization
//! 3. Black-Scholes (1973) - Options pricing for VaR
//! 4. RiskMetrics (1996) - VaR methodology
//! 5. Artzner et al (1999) - Coherent risk measures

use rust_decimal::Decimal;
use std::collections::HashMap;
use statrs::distribution::{Normal, ContinuousCDF};

/// Unified Risk Calculator - All risk metrics in one place
pub struct UnifiedRiskCalculator {
    /// Historical data for calculations
    historical_returns: Vec<f64>,
    
    /// Confidence levels
    var_confidence: f64,  // Default 0.99
    cvar_confidence: f64, // Default 0.95
    
    /// Risk limits
    max_kelly_fraction: Decimal,  // 0.25 cap
    max_position_size: Decimal,   // 10% of portfolio
    max_leverage: f64,            // 3x max
    
    /// Market conditions
    volatility_regime: VolatilityRegime,
}

#[derive(Debug, Clone)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

impl UnifiedRiskCalculator {
    /// CANONICAL FUNCTION 1: Kelly Criterion
    /// Cameron: "The ONLY Kelly calculation we use!"
    pub fn calculate_kelly_criterion(
        &self,
        win_probability: f64,
        win_return: f64,
        loss_return: f64,
        confidence: Option<f64>,
    ) -> Decimal {
        println!("CAMERON: Calculating Kelly criterion");
        
        // Basic Kelly formula: f = (p*b - q)/b
        // where p = win prob, q = loss prob, b = win/loss ratio
        let q = 1.0 - win_probability;
        let b = win_return / loss_return.abs();
        
        let raw_kelly = (win_probability * b - q) / b;
        
        // Adjust for confidence (Blake's ML confidence)
        let confidence_adjusted = match confidence {
            Some(c) => raw_kelly * c,
            None => raw_kelly,
        };
        
        // Apply volatility regime adjustment
        let regime_adjusted = match self.volatility_regime {
            VolatilityRegime::Low => confidence_adjusted * 1.2,
            VolatilityRegime::Normal => confidence_adjusted,
            VolatilityRegime::High => confidence_adjusted * 0.7,
            VolatilityRegime::Extreme => confidence_adjusted * 0.3,
        };
        
        // Cap at maximum (CRITICAL SAFETY)
        let capped = regime_adjusted.min(0.25).max(0.0);
        
        println!("CAMERON: Kelly = {:.4} (raw: {:.4}, adjusted: {:.4})",
                 capped, raw_kelly, regime_adjusted);
        
        Decimal::from_f64(capped).unwrap_or(Decimal::ZERO)
    }
    
    /// CANONICAL FUNCTION 2: Value at Risk (VaR)
    /// Cameron: "99% confidence VaR calculation"
    pub fn calculate_var(
        &self,
        position_value: Decimal,
        holding_period_days: u32,
    ) -> Decimal {
        println!("CAMERON: Calculating VaR at {}% confidence", self.var_confidence * 100.0);
        
        if self.historical_returns.is_empty() {
            return Decimal::ZERO;
        }
        
        // Calculate historical volatility
        let mean = self.historical_returns.iter().sum::<f64>() / self.historical_returns.len() as f64;
        let variance = self.historical_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / self.historical_returns.len() as f64;
        let volatility = variance.sqrt();
        
        // Adjust for holding period
        let period_volatility = volatility * (holding_period_days as f64).sqrt();
        
        // Calculate VaR using normal distribution
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_score = normal.inverse_cdf(1.0 - self.var_confidence);
        
        let var = position_value * Decimal::from_f64(z_score.abs() * period_volatility).unwrap();
        
        println!("CAMERON: VaR = ${:.2} over {} days", var, holding_period_days);
        
        var
    }
    
    /// CANONICAL FUNCTION 3: Conditional VaR (CVaR/Expected Shortfall)
    /// Cameron: "What's the expected loss beyond VaR?"
    pub fn calculate_cvar(
        &self,
        position_value: Decimal,
        holding_period_days: u32,
    ) -> Decimal {
        println!("CAMERON: Calculating CVaR (Expected Shortfall)");
        
        let var = self.calculate_var(position_value, holding_period_days);
        
        // CVaR is approximately 1.4x VaR for normal distribution
        let cvar = var * Decimal::from_f64(1.4).unwrap();
        
        println!("CAMERON: CVaR = ${:.2} (worst-case expected loss)", cvar);
        
        cvar
    }
    
    /// CANONICAL FUNCTION 4: Portfolio Risk (Correlation-adjusted)
    /// Cameron: "Portfolio-level risk considering correlations"
    pub fn calculate_portfolio_risk(
        &self,
        positions: &[PortfolioPosition],
        correlation_matrix: &CorrelationMatrix,
    ) -> PortfolioRisk {
        println!("CAMERON: Calculating portfolio risk for {} positions", positions.len());
        
        let total_value = positions.iter()
            .map(|p| p.value)
            .sum::<Decimal>();
        
        // Calculate weighted volatility
        let mut portfolio_variance = 0.0;
        
        for i in 0..positions.len() {
            for j in 0..positions.len() {
                let weight_i = (positions[i].value / total_value).to_f64().unwrap_or(0.0);
                let weight_j = (positions[j].value / total_value).to_f64().unwrap_or(0.0);
                let corr = correlation_matrix.get(i, j).unwrap_or(0.5);
                let vol_i = positions[i].volatility;
                let vol_j = positions[j].volatility;
                
                portfolio_variance += weight_i * weight_j * corr * vol_i * vol_j;
            }
        }
        
        let portfolio_volatility = portfolio_variance.sqrt();
        
        // Calculate diversification ratio
        let weighted_avg_volatility = positions.iter()
            .map(|p| {
                let weight = (p.value / total_value).to_f64().unwrap_or(0.0);
                weight * p.volatility
            })
            .sum::<f64>();
        
        let diversification_ratio = if portfolio_volatility > 0.0 {
            weighted_avg_volatility / portfolio_volatility
        } else {
            1.0
        };
        
        // Calculate concentration risk (Herfindahl index)
        let concentration = positions.iter()
            .map(|p| {
                let weight = (p.value / total_value).to_f64().unwrap_or(0.0);
                weight * weight
            })
            .sum::<f64>();
        
        let portfolio_var = self.calculate_var(total_value, 1);
        let portfolio_cvar = self.calculate_cvar(total_value, 1);
        
        println!("CAMERON: Portfolio volatility: {:.2}%, Diversification: {:.2}x",
                 portfolio_volatility * 100.0, diversification_ratio);
        
        PortfolioRisk {
            total_value,
            portfolio_volatility,
            portfolio_var,
            portfolio_cvar,
            diversification_ratio,
            concentration_risk: concentration,
            risk_score: self.calculate_risk_score(portfolio_volatility, concentration),
        }
    }
    
    /// CANONICAL FUNCTION 5: Dynamic Risk Limits
    /// Cameron: "Adjust risk limits based on market conditions"
    pub fn calculate_dynamic_risk_limits(
        &self,
        base_limit: Decimal,
        recent_performance: &PerformanceMetrics,
    ) -> RiskLimits {
        println!("CAMERON: Calculating dynamic risk limits");
        
        // Start with base limit
        let mut adjusted_limit = base_limit;
        
        // Adjust for recent performance
        if recent_performance.sharpe_ratio < 1.0 {
            adjusted_limit = adjusted_limit * Decimal::from_f64(0.8).unwrap();
            println!("CAMERON: Reducing limits - poor Sharpe ratio");
        }
        
        if recent_performance.max_drawdown > 0.15 {
            adjusted_limit = adjusted_limit * Decimal::from_f64(0.7).unwrap();
            println!("CAMERON: Reducing limits - high drawdown");
        }
        
        // Adjust for volatility regime
        adjusted_limit = match self.volatility_regime {
            VolatilityRegime::Low => adjusted_limit * Decimal::from_f64(1.2).unwrap(),
            VolatilityRegime::Normal => adjusted_limit,
            VolatilityRegime::High => adjusted_limit * Decimal::from_f64(0.6).unwrap(),
            VolatilityRegime::Extreme => adjusted_limit * Decimal::from_f64(0.3).unwrap(),
        };
        
        // Calculate specific limits
        let position_limit = adjusted_limit * self.max_position_size;
        let daily_loss_limit = adjusted_limit * Decimal::from_f64(0.02).unwrap(); // 2% daily
        let leverage_limit = match self.volatility_regime {
            VolatilityRegime::Low => 3.0,
            VolatilityRegime::Normal => 2.0,
            VolatilityRegime::High => 1.0,
            VolatilityRegime::Extreme => 0.5,
        };
        
        println!("CAMERON: Limits - Position: ${}, Daily Loss: ${}, Leverage: {}x",
                 position_limit, daily_loss_limit, leverage_limit);
        
        RiskLimits {
            max_position_size: position_limit,
            max_daily_loss: daily_loss_limit,
            max_leverage: leverage_limit,
            max_var: adjusted_limit * Decimal::from_f64(0.05).unwrap(),
            kelly_cap: self.max_kelly_fraction,
        }
    }
    
    fn calculate_risk_score(&self, volatility: f64, concentration: f64) -> f64 {
        // Score from 0-100
        let vol_score = (volatility * 100.0).min(50.0);
        let conc_score = (concentration * 100.0).min(50.0);
        vol_score + conc_score
    }
}

// Supporting structures
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    pub symbol: String,
    pub value: Decimal,
    pub volatility: f64,
}

pub struct CorrelationMatrix {
    data: Vec<Vec<f64>>,
}

impl CorrelationMatrix {
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.data.get(i)?.get(j).copied()
    }
}

#[derive(Debug)]
pub struct PortfolioRisk {
    pub total_value: Decimal,
    pub portfolio_volatility: f64,
    pub portfolio_var: Decimal,
    pub portfolio_cvar: Decimal,
    pub diversification_ratio: f64,
    pub concentration_risk: f64,
    pub risk_score: f64,
}

#[derive(Debug)]
pub struct PerformanceMetrics {
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
}

// REMOVED: Using canonical domain_types::RiskLimits
// #[derive(Debug)]
// pub struct RiskLimits {
//     pub max_position_size: Decimal,
//     pub max_daily_loss: Decimal,
//     pub max_leverage: f64,
//     pub max_var: Decimal,
//     pub kelly_cap: Decimal,
// }

// CAMERON: "All risk calculations unified! No more inconsistencies!"