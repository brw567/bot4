//! Comprehensive risk metrics calculation

use anyhow::{Result, bail};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    pub total_value: Decimal,
    pub max_position: Decimal,
    pub portfolio_risk: Decimal,
    pub max_correlation: Decimal,
    pub leverage: Decimal,
    pub beta: Decimal,
    pub var_95: Decimal,
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub max_drawdown: Decimal,
}

impl PortfolioMetrics {
    pub fn calculate_risk_score(&self) -> Decimal {
        // Weighted risk score (0-100, higher is riskier)
        let mut score = dec!(0);
        
        // Position concentration risk (0-25 points)
        score += self.max_position * dec!(125); // 0.02 max = 2.5 points, 0.2 = 25 points
        
        // Portfolio volatility risk (0-25 points)
        score += self.portfolio_risk * dec!(166.67); // 0.15 max = 25 points
        
        // Correlation risk (0-20 points)
        score += self.max_correlation * dec!(28.57); // 0.7 max = 20 points
        
        // Leverage risk (0-20 points)
        score += (self.leverage - dec!(1)).max(dec!(0)) * dec!(6.67); // 3x leverage = 20 points
        
        // Drawdown risk (0-10 points)
        score += self.max_drawdown * dec!(66.67); // 0.15 max = 10 points
        
        score.min(dec!(100))
    }
}

pub struct RiskMetrics;

impl RiskMetrics {
    pub fn new() -> Self {
        Self
    }
    
    pub fn calculate(&self, positions: &[Value]) -> Result<PortfolioMetrics> {
        if positions.is_empty() {
            return Ok(PortfolioMetrics {
                total_value: dec!(0),
                max_position: dec!(0),
                portfolio_risk: dec!(0),
                max_correlation: dec!(0),
                leverage: dec!(1),
                beta: dec!(0),
                var_95: dec!(0),
                sharpe_ratio: dec!(0),
                sortino_ratio: dec!(0),
                max_drawdown: dec!(0),
            });
        }
        
        let mut total_value = dec!(0);
        let mut max_position_size = dec!(0);
        let mut total_leverage = dec!(0);
        let mut returns_history = Vec::new();
        
        for position in positions {
            let value = Decimal::from_f64_retain(
                position["value"].as_f64().unwrap_or(0.0)
            ).unwrap_or(dec!(0));
            
            let size = Decimal::from_f64_retain(
                position["size"].as_f64().unwrap_or(0.0)
            ).unwrap_or(dec!(0));
            
            let leverage = Decimal::from_f64_retain(
                position["leverage"].as_f64().unwrap_or(1.0)
            ).unwrap_or(dec!(1));
            
            total_value += value;
            max_position_size = max_position_size.max(size);
            total_leverage += leverage * size;
            
            // Collect returns if available
            if let Some(returns) = position["returns"].as_array() {
                for ret in returns {
                    if let Some(r) = ret.as_f64() {
                        returns_history.push(r);
                    }
                }
            }
        }
        
        // Calculate portfolio-level metrics
        let portfolio_risk = self.calculate_portfolio_risk(&returns_history)?;
        let max_correlation = self.estimate_max_correlation(positions)?;
        let leverage = if total_value > dec!(0) {
            total_leverage / total_value
        } else {
            dec!(1)
        };
        
        // Calculate VaR
        let var_95 = self.calculate_simple_var(&returns_history, 0.95)?;
        
        // Calculate performance ratios
        let (sharpe, sortino) = self.calculate_ratios(&returns_history)?;
        
        // Calculate max drawdown
        let max_drawdown = self.calculate_max_drawdown(&returns_history)?;
        
        // Beta estimation (simplified - would need market returns in practice)
        let beta = dec!(1); // Placeholder
        
        Ok(PortfolioMetrics {
            total_value,
            max_position: max_position_size,
            portfolio_risk,
            max_correlation,
            leverage,
            beta,
            var_95,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown,
        })
    }
    
    fn calculate_portfolio_risk(&self, returns: &[f64]) -> Result<Decimal> {
        if returns.is_empty() {
            return Ok(dec!(0));
        }
        
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(Decimal::from_f64_retain(variance.sqrt()).unwrap_or(dec!(0)))
    }
    
    fn estimate_max_correlation(&self, positions: &[Value]) -> Result<Decimal> {
        // Simplified correlation estimation based on asset types
        let mut asset_types = std::collections::HashSet::new();
        
        for position in positions {
            if let Some(asset_type) = position["asset_type"].as_str() {
                asset_types.insert(asset_type);
            }
        }
        
        // If all positions are the same asset type, high correlation
        if asset_types.len() == 1 {
            Ok(dec!(0.9))
        } else if asset_types.len() == 2 {
            Ok(dec!(0.6))
        } else {
            Ok(dec!(0.3))
        }
    }
    
    fn calculate_simple_var(&self, returns: &[f64], confidence: f64) -> Result<Decimal> {
        if returns.is_empty() {
            return Ok(dec!(0));
        }
        
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted.len() as f64) as usize;
        let var = -sorted[index.min(sorted.len() - 1)];
        
        Ok(Decimal::from_f64_retain(var).unwrap_or(dec!(0)))
    }
    
    fn calculate_ratios(&self, returns: &[f64]) -> Result<(Decimal, Decimal)> {
        if returns.is_empty() || returns.len() < 2 {
            return Ok((dec!(0), dec!(0)));
        }
        
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev: f64 = (returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64)
            .sqrt();
        
        // Sharpe ratio (assuming risk-free rate = 0)
        let sharpe = if std_dev > 0.0 {
            mean / std_dev
        } else {
            0.0
        };
        
        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        let sortino = if !downside_returns.is_empty() {
            let downside_dev: f64 = (downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64)
                .sqrt();
            
            if downside_dev > 0.0 {
                mean / downside_dev
            } else {
                0.0
            }
        } else {
            sharpe * 1.5 // If no downside, Sortino > Sharpe
        };
        
        Ok((
            Decimal::from_f64_retain(sharpe).unwrap_or(dec!(0)),
            Decimal::from_f64_retain(sortino).unwrap_or(dec!(0))
        ))
    }
    
    fn calculate_max_drawdown(&self, returns: &[f64]) -> Result<Decimal> {
        if returns.is_empty() {
            return Ok(dec!(0));
        }
        
        let mut cumulative_value = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        
        for &ret in returns {
            cumulative_value *= 1.0 + ret;
            if cumulative_value > peak {
                peak = cumulative_value;
            }
            let drawdown = (peak - cumulative_value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        Ok(Decimal::from_f64_retain(max_drawdown).unwrap_or(dec!(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_risk_metrics() {
        let metrics = RiskMetrics::new();
        
        let positions = vec![
            serde_json::json!({
                "value": 10000.0,
                "size": 0.01,
                "leverage": 1.0,
                "asset_type": "BTC",
                "returns": [0.02, -0.01, 0.03, -0.02]
            }),
            serde_json::json!({
                "value": 5000.0,
                "size": 0.005,
                "leverage": 2.0,
                "asset_type": "ETH",
                "returns": [0.01, -0.02, 0.04, -0.01]
            }),
        ];
        
        let result = metrics.calculate(&positions).unwrap();
        assert!(result.total_value > dec!(0));
        assert!(result.max_position > dec!(0));
        assert!(result.portfolio_risk >= dec!(0));
    }
    
    #[test]
    fn test_risk_score() {
        let portfolio = PortfolioMetrics {
            total_value: dec!(100000),
            max_position: dec!(0.02),
            portfolio_risk: dec!(0.05),
            max_correlation: dec!(0.3),
            leverage: dec!(1.5),
            beta: dec!(1.2),
            var_95: dec!(0.03),
            sharpe_ratio: dec!(1.5),
            sortino_ratio: dec!(2.0),
            max_drawdown: dec!(0.08),
        };
        
        let score = portfolio.calculate_risk_score();
        assert!(score >= dec!(0) && score <= dec!(100));
    }
}