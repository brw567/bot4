//! Portfolio optimization using Markowitz mean-variance optimization

use anyhow::{Result, bail};
use ndarray::{Array1, Array2};
use ndarray_stats::CorrelationExt;

pub struct PortfolioOptimizer;

impl PortfolioOptimizer {
    pub fn new() -> Self {
        Self
    }
    
    /// Optimize portfolio weights to minimize risk for target return
    pub fn optimize(&self, assets: &[String], returns: &[Vec<f64>], target_return: f64) -> Result<Vec<f64>> {
        if assets.is_empty() {
            bail!("No assets provided");
        }
        
        if returns.is_empty() || returns[0].is_empty() {
            bail!("No return data provided");
        }
        
        let n_assets = assets.len();
        if returns.len() != n_assets {
            bail!("Mismatch between assets and returns");
        }
        
        // Calculate expected returns and covariance matrix
        let expected_returns = self.calculate_expected_returns(returns)?;
        let cov_matrix = self.calculate_covariance_matrix(returns)?;
        
        // For now, use equal weight portfolio as baseline
        // TODO: Implement quadratic programming solver for true optimization
        let mut weights = vec![1.0 / n_assets as f64; n_assets];
        
        // Adjust weights based on Sharpe ratio
        for i in 0..n_assets {
            let sharpe = expected_returns[i] / (cov_matrix[(i, i)].sqrt() + 1e-10);
            weights[i] *= (1.0 + sharpe.max(-0.5).min(0.5));
        }
        
        // Normalize weights to sum to 1
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        
        Ok(weights)
    }
    
    /// Calculate correlation matrix
    pub fn calculate_correlations(&self, returns: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n_assets = returns.len();
        let n_periods = returns[0].len();
        
        // Convert to ndarray format
        let mut data = Array2::zeros((n_periods, n_assets));
        for (i, asset_returns) in returns.iter().enumerate() {
            for (j, &ret) in asset_returns.iter().enumerate() {
                data[(j, i)] = ret;
            }
        }
        
        // Calculate correlation matrix
        let corr = data.pearson_correlation()?;
        Ok(corr)
    }
    
    /// Calculate Sharpe ratio for given weights
    pub fn calculate_sharpe(&self, returns: &[Vec<f64>], weights: &[f64]) -> Result<f64> {
        if returns.len() != weights.len() {
            bail!("Mismatch between returns and weights");
        }
        
        // Calculate portfolio returns
        let n_periods = returns[0].len();
        let mut portfolio_returns = vec![0.0; n_periods];
        
        for period in 0..n_periods {
            for (asset_idx, weight) in weights.iter().enumerate() {
                portfolio_returns[period] += weight * returns[asset_idx][period];
            }
        }
        
        // Calculate mean and std
        let mean: f64 = portfolio_returns.iter().sum::<f64>() / n_periods as f64;
        let variance: f64 = portfolio_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n_periods as f64;
        let std = variance.sqrt();
        
        // Sharpe ratio (assuming risk-free rate = 0)
        Ok(mean / (std + 1e-10))
    }
    
    fn calculate_expected_returns(&self, returns: &[Vec<f64>]) -> Result<Vec<f64>> {
        let mut expected = Vec::new();
        
        for asset_returns in returns {
            if asset_returns.is_empty() {
                bail!("Empty returns for asset");
            }
            let mean: f64 = asset_returns.iter().sum::<f64>() / asset_returns.len() as f64;
            expected.push(mean);
        }
        
        Ok(expected)
    }
    
    fn calculate_covariance_matrix(&self, returns: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n_assets = returns.len();
        let expected = self.calculate_expected_returns(returns)?;
        
        let mut cov = Array2::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in 0..n_assets {
                let cov_ij = self.calculate_covariance(&returns[i], &returns[j], expected[i], expected[j])?;
                cov[(i, j)] = cov_ij;
            }
        }
        
        Ok(cov)
    }
    
    fn calculate_covariance(&self, returns1: &[f64], returns2: &[f64], mean1: f64, mean2: f64) -> Result<f64> {
        if returns1.len() != returns2.len() {
            bail!("Return series must have same length");
        }
        
        let n = returns1.len() as f64;
        let cov: f64 = returns1.iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| (r1 - mean1) * (r2 - mean2))
            .sum::<f64>() / (n - 1.0);
        
        Ok(cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_portfolio_optimization() {
        let optimizer = PortfolioOptimizer::new();
        
        let assets = vec!["BTC".to_string(), "ETH".to_string()];
        let returns = vec![
            vec![0.05, 0.02, -0.03, 0.04, 0.01],
            vec![0.03, 0.04, -0.02, 0.05, 0.02],
        ];
        
        let weights = optimizer.optimize(&assets, &returns, 0.03).unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_sharpe_calculation() {
        let optimizer = PortfolioOptimizer::new();
        
        let returns = vec![
            vec![0.05, 0.02, -0.03, 0.04, 0.01],
            vec![0.03, 0.04, -0.02, 0.05, 0.02],
        ];
        let weights = vec![0.6, 0.4];
        
        let sharpe = optimizer.calculate_sharpe(&returns, &weights).unwrap();
        assert!(sharpe > -10.0 && sharpe < 10.0);
    }
}