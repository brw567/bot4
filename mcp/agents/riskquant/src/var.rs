//! Value at Risk (VaR) and Expected Shortfall calculations

use anyhow::{Result, bail};
use statrs::distribution::{Normal, ContinuousCDF};

pub struct VaRResult {
    pub var_amount: f64,
    pub expected_shortfall: f64,
    pub max_loss: f64,
}

pub struct VaRCalculator;

impl VaRCalculator {
    pub fn new() -> Self {
        Self
    }
    
    /// Calculate Value at Risk using historical simulation
    pub fn calculate(&self, returns: &[f64], confidence: f64, horizon: u32) -> Result<VaRResult> {
        if returns.is_empty() {
            bail!("No returns provided");
        }
        
        if confidence <= 0.0 || confidence >= 1.0 {
            bail!("Confidence level must be between 0 and 1");
        }
        
        // Sort returns in ascending order
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR at confidence level
        let var_index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let var_amount = -sorted_returns[var_index.min(sorted_returns.len() - 1)];
        
        // Calculate Expected Shortfall (CVaR)
        let expected_shortfall = if var_index > 0 {
            let sum: f64 = sorted_returns[..var_index].iter().sum();
            -(sum / var_index as f64)
        } else {
            var_amount
        };
        
        // Scale by time horizon
        let scaled_var = var_amount * (horizon as f64).sqrt();
        let scaled_es = expected_shortfall * (horizon as f64).sqrt();
        
        // Max observed loss
        let max_loss = -sorted_returns[0];
        
        Ok(VaRResult {
            var_amount: scaled_var,
            expected_shortfall: scaled_es,
            max_loss,
        })
    }
    
    /// Calculate parametric VaR assuming normal distribution
    pub fn calculate_parametric(&self, mean: f64, std_dev: f64, confidence: f64, horizon: u32) -> Result<VaRResult> {
        if std_dev <= 0.0 {
            bail!("Standard deviation must be positive");
        }
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_score = normal.inverse_cdf(1.0 - confidence);
        
        // Scale by time horizon
        let scaled_mean = mean * horizon as f64;
        let scaled_std = std_dev * (horizon as f64).sqrt();
        
        let var_amount = -(scaled_mean + z_score * scaled_std);
        
        // For normal distribution, ES = μ - σ * φ(z) / Φ(z)
        // where φ is PDF and Φ is CDF
        let pdf_z = (-z_score.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected_shortfall = -(scaled_mean - scaled_std * pdf_z / (1.0 - confidence));
        
        Ok(VaRResult {
            var_amount,
            expected_shortfall,
            max_loss: var_amount * 3.0, // Rough estimate
        })
    }
    
    /// Calculate percentile of a sorted array
    pub fn percentile(&self, sorted_data: &[f64], percentile: f64) -> Result<f64> {
        if sorted_data.is_empty() {
            bail!("No data provided");
        }
        
        if percentile < 0.0 || percentile > 1.0 {
            bail!("Percentile must be between 0 and 1");
        }
        
        let mut sorted = sorted_data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile * (sorted.len() - 1) as f64) as usize;
        Ok(sorted[index])
    }
    
    /// Backtest VaR model
    pub fn backtest(&self, returns: &[f64], predicted_var: f64, confidence: f64) -> Result<BacktestResult> {
        let violations = returns.iter()
            .filter(|&&r| -r > predicted_var)
            .count();
        
        let expected_violations = ((1.0 - confidence) * returns.len() as f64) as usize;
        let violation_rate = violations as f64 / returns.len() as f64;
        
        // Kupiec test statistic
        let kupiec = self.kupiec_test(violations, returns.len(), confidence)?;
        
        Ok(BacktestResult {
            violations,
            expected_violations,
            violation_rate,
            kupiec_statistic: kupiec,
            model_valid: (kupiec < 3.84), // Chi-squared critical value at 95%
        })
    }
    
    fn kupiec_test(&self, violations: usize, total: usize, confidence: f64) -> Result<f64> {
        let p = 1.0 - confidence;
        let v = violations as f64;
        let t = total as f64;
        
        if violations == 0 {
            return Ok(0.0);
        }
        
        let likelihood_ratio = 2.0 * (
            v * (v / t).ln() + 
            (t - v) * ((t - v) / t).ln() -
            v * p.ln() -
            (t - v) * (1.0 - p).ln()
        );
        
        Ok(likelihood_ratio)
    }
}

pub struct BacktestResult {
    pub violations: usize,
    pub expected_violations: usize,
    pub violation_rate: f64,
    pub kupiec_statistic: f64,
    pub model_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_var_calculation() {
        let calc = VaRCalculator::new();
        
        let returns = vec![0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.05, -0.03, 0.02, -0.01];
        let result = calc.calculate(&returns, 0.95, 1).unwrap();
        
        assert!(result.var_amount > 0.0);
        assert!(result.expected_shortfall >= result.var_amount);
        assert!(result.max_loss >= result.expected_shortfall);
    }
    
    #[test]
    fn test_parametric_var() {
        let calc = VaRCalculator::new();
        
        let result = calc.calculate_parametric(0.001, 0.02, 0.95, 1).unwrap();
        assert!(result.var_amount > 0.0);
    }
    
    #[test]
    fn test_backtest() {
        let calc = VaRCalculator::new();
        
        let returns = vec![0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.05, -0.03, 0.02, -0.01];
        let result = calc.backtest(&returns, 0.03, 0.95).unwrap();
        
        assert_eq!(result.violations, returns.iter().filter(|&&r| -r > 0.03).count());
    }
}