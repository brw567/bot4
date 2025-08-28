//! # UNIFIED RISK METRICS - Single Source of Truth
//! Cameron: "One VaR calculation, consistent everywhere"
//! Quinn: "Safety through consistency"

use crate::variance::calculate_variance;
use statrs::distribution::{Normal, ContinuousCDF};

/// CANONICAL VaR Calculation - Historical Simulation
pub fn calculate_var(returns: &[f64], confidence_level: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
    -sorted[index.min(sorted.len() - 1)]
}

/// CANONICAL CVaR (Expected Shortfall)
pub fn calculate_cvar(returns: &[f64], confidence_level: f64) -> f64 {
    let var = calculate_var(returns, confidence_level);
    
    let tail_losses: Vec<f64> = returns.iter()
        .filter(|&&r| r <= -var)
        .copied()
        .collect();
    
    if tail_losses.is_empty() {
        return var;
    }
    
    -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
}

/// CANONICAL Sharpe Ratio
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = calculate_variance(returns, mean_return);
    let std_dev = variance.sqrt();
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    (mean_return - risk_free_rate) / std_dev * (252.0_f64).sqrt() // Annualized
}

/// CANONICAL Sortino Ratio (downside deviation)
pub fn calculate_sortino(returns: &[f64], target_return: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    
    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns.iter()
        .map(|&r| (target_return - r).max(0.0))
        .collect();
    
    let downside_variance = downside_returns.iter()
        .map(|&d| d * d)
        .sum::<f64>() / downside_returns.len() as f64;
    
    let downside_dev = downside_variance.sqrt();
    
    if downside_dev == 0.0 {
        return 0.0;
    }
    
    (mean_return - target_return) / downside_dev * (252.0_f64).sqrt()
}

/// CANONICAL Maximum Drawdown
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    
    let mut max_drawdown = 0.0;
    let mut peak = equity_curve[0];
    
    for &value in equity_curve.iter() {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    max_drawdown
}

// TEAM: "Risk calculations unified!"
