//! # Risk Metrics - Portfolio and Strategy Risk Analysis

use crate::statistics::{mean, standard_deviation};
use crate::variance::{calculate_var, calculate_cvar, VarMethod};

/// Calculate Sharpe ratio
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mean_return = mean(returns);
    let std_dev = standard_deviation(returns);
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    // Annualize (assuming daily returns)
    let annual_return = mean_return * 252.0;
    let annual_std = std_dev * (252.0_f64).sqrt();
    
    (annual_return - risk_free_rate) / annual_std
}

/// Calculate Sortino ratio (downside deviation)
pub fn sortino_ratio(returns: &[f64], target_return: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mean_return = mean(returns);
    
    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .collect();
    
    if downside_returns.is_empty() {
        return f64::INFINITY;
    }
    
    let downside_dev = (downside_returns.iter().sum::<f64>() 
        / returns.len() as f64).sqrt();
    
    if downside_dev == 0.0 {
        return f64::INFINITY;
    }
    
    // Annualize
    let annual_return = mean_return * 252.0;
    let annual_downside = downside_dev * (252.0_f64).sqrt();
    
    (annual_return - target_return * 252.0) / annual_downside
}

/// Calculate Calmar ratio
pub fn calmar_ratio(returns: &[f64], period_years: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
    let annual_return = (1.0 + total_return).powf(1.0 / period_years) - 1.0;
    
    let max_dd = max_drawdown(returns);
    
    if max_dd == 0.0 {
        return f64::INFINITY;
    }
    
    annual_return / max_dd.abs()
}

/// Calculate maximum drawdown
pub fn max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    
    for &ret in returns {
        cumulative *= 1.0 + ret;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (cumulative - peak) / peak;
        if drawdown < max_dd {
            max_dd = drawdown;
        }
    }
    
    max_dd
}

/// Calculate Value at Risk (wrapper for variance module)
pub fn value_at_risk(returns: &[f64], confidence: f64) -> f64 {
    calculate_var(returns, confidence, 1, VarMethod::Historical)
        .map(|result| result.var)
        .unwrap_or(0.0)
}

/// Calculate Conditional Value at Risk (CVaR)
pub fn conditional_value_at_risk(returns: &[f64], confidence: f64) -> f64 {
    calculate_cvar(returns, confidence).unwrap_or(0.0)
}