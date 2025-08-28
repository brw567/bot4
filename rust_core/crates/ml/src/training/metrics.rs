// Metrics Module - Performance Evaluation Metrics
// Team Lead: Riley (Testing & Metrics)
// Contributors: Morgan (ML Metrics), Quinn (Risk Metrics)
// Date: January 18, 2025
// NO SIMPLIFICATIONS - FULL IMPLEMENTATION

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// REGRESSION METRICS - Riley's Implementation
// ============================================================================

/// Mean Absolute Error
pub fn mae(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(f64::abs).mean().unwrap_or(0.0)
}

/// Mean Squared Error
pub fn mse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x * x).mean().unwrap_or(0.0)
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// Mean Absolute Percentage Error
pub fn mape(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    
    for i in 0..y_true.shape()[0] {
        for j in 0..y_true.shape()[1] {
            let true_val = y_true[[i, j]];
            if true_val.abs() > 1e-10 {
                let pred_val = y_pred[[i, j]];
                sum += ((true_val - pred_val) / true_val).abs();
                count += 1;
            }
        }
    }
    
    if count > 0 {
        sum / count as f64 * 100.0
    } else {
        0.0
    }
}

/// R-squared score
pub fn r2_score(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let ss_res = ((y_true - y_pred).mapv(|x| x * x)).sum();
    let y_mean = y_true.mean().unwrap_or(0.0);
    let ss_tot = y_true.mapv(|x| (x - y_mean).powi(2)).sum();
    
    if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - (ss_res / ss_tot)
    }
}

/// Adjusted R-squared
pub fn adjusted_r2(y_true: &Array2<f64>, y_pred: &Array2<f64>, n_features: usize) -> f64 {
    let n_samples = y_true.shape()[0];
    let r2 = r2_score(y_true, y_pred);
    
    if n_samples <= n_features + 1 {
        0.0
    } else {
        1.0 - (1.0 - r2) * (n_samples - 1) as f64 / (n_samples - n_features - 1) as f64
    }
}

/// Huber loss (robust to outliers)
pub fn huber_loss(y_true: &Array2<f64>, y_pred: &Array2<f64>, delta: f64) -> f64 {
    let diff = y_true - y_pred;
    let mut loss = 0.0;
    let n = diff.len() as f64;
    
    for &d in diff.iter() {
        if d.abs() <= delta {
            loss += 0.5 * d * d;
        } else {
            loss += delta * (d.abs() - 0.5 * delta);
        }
    }
    
    loss / n
}

/// Quantile loss for quantile regression
pub fn quantile_loss(y_true: &Array2<f64>, y_pred: &Array2<f64>, quantile: f64) -> f64 {
    let diff = y_true - y_pred;
    let mut loss = 0.0;
    
    for &d in diff.iter() {
        if d >= 0.0 {
            loss += quantile * d;
        } else {
            loss += (quantile - 1.0) * d;
        }
    }
    
    loss / diff.len() as f64
}

// ============================================================================
// TRADING METRICS - Quinn's Risk-Aware Metrics
// ============================================================================

/// Sharpe ratio calculation
pub fn sharpe_ratio(returns: &Array1<f64>, risk_free_rate: f64) -> f64 {
    let mean_return = returns.mean().unwrap_or(0.0);
    let std_return = returns.std(1.0);
    
    if std_return == 0.0 {
        0.0
    } else {
        (mean_return - risk_free_rate) / std_return * (252.0_f64).sqrt() // Annualized
    }
}

/// Sortino ratio (downside risk)
pub fn sortino_ratio(returns: &Array1<f64>, target_return: f64) -> f64 {
    let mean_return = returns.mean().unwrap_or(0.0);
    
    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .collect();
    
    if downside_returns.is_empty() {
        return 0.0;
    }
    
    let downside_dev = (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt();
    
    if downside_dev == 0.0 {
        0.0
    } else {
        (mean_return - target_return) / downside_dev * (252.0_f64).sqrt()
    }
}

/// Maximum drawdown
pub fn max_drawdown(cumulative_returns: &Array1<f64>) -> f64 {
    let mut max_value = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    
    for &value in cumulative_returns.iter() {
        if value > max_value {
            max_value = value;
        }
        let drawdown = (value - max_value) / max_value;
        if drawdown < max_dd {
            max_dd = drawdown;
        }
    }
    
    max_dd.abs()
}

/// Calmar ratio (return / max drawdown)
pub fn calmar_ratio(returns: &Array1<f64>) -> f64 {
    let annual_return = returns.mean().unwrap_or(0.0) * 252.0;
    let cumulative = returns
        .iter()
        .scan(1.0, |acc, &r| {
            *acc *= 1.0 + r;
            Some(*acc)
        })
        .collect::<Array1<f64>>();
    
    let mdd = max_drawdown(&cumulative);
    
    if mdd == 0.0 {
        0.0
    } else {
        annual_return / mdd
    }
}

/// Information ratio
pub fn information_ratio(returns: &Array1<f64>, benchmark_returns: &Array1<f64>) -> f64 {
    let active_returns = returns - benchmark_returns;
    let tracking_error = active_returns.std(1.0);
    
    if tracking_error == 0.0 {
        0.0
    } else {
        active_returns.mean().unwrap_or(0.0) / tracking_error * (252.0_f64).sqrt()
    }
}

// ============================================================================
// DIRECTIONAL METRICS - Morgan's Trading Performance
// ============================================================================

/// Directional accuracy (correct direction predictions)
pub fn directional_accuracy(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let mut correct = 0;
    let mut total = 0;
    
    for i in 1..y_true.shape()[0] {
        let true_direction = y_true[[i, 0]] - y_true[[i-1, 0]];
        let pred_direction = y_pred[[i, 0]] - y_pred[[i-1, 0]];
        
        if true_direction * pred_direction > 0.0 {
            correct += 1;
        }
        total += 1;
    }
    
    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    }
}

/// Hit rate (profitable trades percentage)
pub fn hit_rate(returns: &Array1<f64>) -> f64 {
    let profitable = returns.iter().filter(|&&r| r > 0.0).count();
    let total = returns.len();
    
    if total > 0 {
        profitable as f64 / total as f64
    } else {
        0.0
    }
}

/// Profit factor (gross profit / gross loss)
pub fn profit_factor(returns: &Array1<f64>) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    
    if gross_loss == 0.0 {
        if gross_profit > 0.0 { f64::INFINITY } else { 0.0 }
    } else {
        gross_profit / gross_loss
    }
}

// ============================================================================
// COMPREHENSIVE METRIC CALCULATOR - Riley's Main Implementation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub regression_metrics: HashMap<String, f64>,
    pub trading_metrics: HashMap<String, f64>,
    pub risk_metrics: HashMap<String, f64>,
    pub timestamp: u64,
}

pub struct MetricsCalculator {
    include_trading: bool,
    include_risk: bool,
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCalculator {
    pub fn new() -> Self {
        Self {
            include_trading: true,
            include_risk: true,
        }
    }
    
    pub fn regression_only(mut self) -> Self {
        self.include_trading = false;
        self.include_risk = false;
        self
    }
    
    /// Calculate comprehensive metrics
    pub fn calculate_all(
        &self,
        y_true: &Array2<f64>,
        y_pred: &Array2<f64>,
        returns: Option<&Array1<f64>>,
    ) -> MetricsReport {
        let mut regression_metrics = HashMap::new();
        let mut trading_metrics = HashMap::new();
        let mut risk_metrics = HashMap::new();
        
        // Regression metrics
        regression_metrics.insert("mae".to_string(), mae(y_true, y_pred));
        regression_metrics.insert("mse".to_string(), mse(y_true, y_pred));
        regression_metrics.insert("rmse".to_string(), rmse(y_true, y_pred));
        regression_metrics.insert("mape".to_string(), mape(y_true, y_pred));
        regression_metrics.insert("r2".to_string(), r2_score(y_true, y_pred));
        regression_metrics.insert("huber".to_string(), huber_loss(y_true, y_pred, 1.0));
        regression_metrics.insert("directional_accuracy".to_string(), directional_accuracy(y_true, y_pred));
        
        // Trading metrics
        if self.include_trading && returns.is_some() {
            let ret = returns.unwrap();
            trading_metrics.insert("sharpe_ratio".to_string(), sharpe_ratio(ret, 0.0));
            trading_metrics.insert("sortino_ratio".to_string(), sortino_ratio(ret, 0.0));
            trading_metrics.insert("calmar_ratio".to_string(), calmar_ratio(ret));
            trading_metrics.insert("hit_rate".to_string(), hit_rate(ret));
            trading_metrics.insert("profit_factor".to_string(), profit_factor(ret));
        }
        
        // Risk metrics
        if self.include_risk && returns.is_some() {
            let ret = returns.unwrap();
            let cumulative = ret
                .iter()
                .scan(1.0, |acc, &r| {
                    *acc *= 1.0 + r;
                    Some(*acc)
                })
                .collect::<Array1<f64>>();
            
            risk_metrics.insert("max_drawdown".to_string(), max_drawdown(&cumulative));
            risk_metrics.insert("var_95".to_string(), self.calculate_var(ret, 0.95));
            risk_metrics.insert("cvar_95".to_string(), self.calculate_cvar(ret, 0.95));
        }
        
        MetricsReport {
            regression_metrics,
            trading_metrics,
            risk_metrics,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Calculate Value at Risk
    use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var(&self, returns: &Array1<f64>, confidence: f64) -> f64 {
        let mut sorted_returns: Vec<f64> = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        sorted_returns[index.min(sorted_returns.len() - 1)]
    }
    
    /// Calculate Conditional Value at Risk (Expected Shortfall)
    fn calculate_cvar(&self, returns: &Array1<f64>, confidence: f64) -> f64 {
        let var = self.calculate_var(returns, confidence);
        
        let tail_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r <= var)
            .copied()
            .collect();
        
        if tail_returns.is_empty() {
            var
        } else {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        }
    }
}

// ============================================================================
// TESTS - Riley's Validation Suite
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regression_metrics() {
        let y_true = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y_pred = Array2::from_shape_vec((5, 1), vec![1.1, 2.1, 2.9, 3.9, 5.1]).unwrap();
        
        assert!((mae(&y_true, &y_pred) - 0.1).abs() < 1e-6);
        assert!((mse(&y_true, &y_pred) - 0.01).abs() < 1e-6);
        assert!((rmse(&y_true, &y_pred) - 0.1).abs() < 1e-6);
        
        let r2 = r2_score(&y_true, &y_pred);
        assert!(r2 > 0.95); // Should be very high for near-perfect predictions
    }
    
    #[test]
    fn test_trading_metrics() {
        let returns = Array1::from_vec(vec![0.01, -0.005, 0.02, -0.01, 0.015]);
        
        let sharpe = sharpe_ratio(&returns, 0.0);
        assert!(sharpe > 0.0); // Positive Sharpe for positive mean return
        
        let hit = hit_rate(&returns);
        assert!((hit - 0.6).abs() < 1e-6); // 3 out of 5 positive
        
        let pf = profit_factor(&returns);
        assert!(pf > 1.0); // More profit than loss
    }
    
    #[test]
    fn test_risk_metrics() {
        let returns = Array1::from_vec(vec![0.1, -0.05, 0.03, -0.02, 0.01]);
        let cumulative = returns
            .iter()
            .scan(1.0, |acc, &r| {
                *acc *= 1.0 + r;
                Some(*acc)
            })
            .collect::<Array1<f64>>();
        
        let mdd = max_drawdown(&cumulative);
        assert!(mdd > 0.0 && mdd < 1.0);
    }
    
    #[test]
    fn test_metrics_calculator() {
        let calc = MetricsCalculator::new();
        let y_true = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y_pred = Array2::from_shape_vec((5, 1), vec![1.1, 1.9, 3.1, 3.9, 5.2]).unwrap();
        let returns = Array1::from_vec(vec![0.01, -0.005, 0.02, -0.01, 0.015]);
        
        let report = calc.calculate_all(&y_true, &y_pred, Some(&returns));
        
        assert!(report.regression_metrics.contains_key("mae"));
        assert!(report.trading_metrics.contains_key("sharpe_ratio"));
        assert!(report.risk_metrics.contains_key("max_drawdown"));
    }
}

// ============================================================================
// TEAM SIGN-OFF
// ============================================================================
// Riley: "Comprehensive metrics suite for evaluation"
// Morgan: "Trading performance metrics implemented"
// Quinn: "Risk metrics with VaR/CVaR complete"