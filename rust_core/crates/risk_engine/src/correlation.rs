use mathematical_ops::correlation::calculate_correlation;
// Correlation Analysis for Risk Management
// Prevents overexposure to correlated assets (Quinn's 0.7 limit)

use ndarray::Array2;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::sync::Arc;

/// Correlation matrix for asset pairs
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    symbols: Vec<String>,
    matrix: Arc<RwLock<Array2<f64>>>,
    last_update: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
}

impl CorrelationMatrix {
    pub fn new(symbols: Vec<String>) -> Self {
        let n = symbols.len();
        let matrix = Array2::eye(n); // Start with identity matrix
        
        Self {
            symbols,
            matrix: Arc::new(RwLock::new(matrix)),
            last_update: Arc::new(RwLock::new(chrono::Utc::now())),
        }
    }
    
    /// Get correlation between two symbols
    pub fn get_correlation(&self, symbol1: &str, symbol2: &str) -> Option<f64> {
        let idx1 = self.symbols.iter().position(|s| s == symbol1)?;
        let idx2 = self.symbols.iter().position(|s| s == symbol2)?;
        
        let matrix = self.matrix.read();
        Some(matrix[[idx1, idx2]])
    }
    
    /// Update correlation value
    pub fn update_correlation(&self, symbol1: &str, symbol2: &str, correlation: f64) {
        if let (Some(idx1), Some(idx2)) = (
            self.symbols.iter().position(|s| s == symbol1),
            self.symbols.iter().position(|s| s == symbol2),
        ) {
            let mut matrix = self.matrix.write();
            matrix[[idx1, idx2]] = correlation;
            matrix[[idx2, idx1]] = correlation; // Symmetric
            *self.last_update.write() = chrono::Utc::now();
        }
    }
    
    /// Get all correlations for a symbol
    pub fn get_symbol_correlations(&self, symbol: &str) -> Vec<(String, f64)> {
        if let Some(idx) = self.symbols.iter().position(|s| s == symbol) {
            let matrix = self.matrix.read();
            self.symbols
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(i, s)| (s.clone(), matrix[[idx, i]]))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Find highly correlated pairs (above threshold)
    pub fn find_correlated_pairs(&self, threshold: f64) -> Vec<(String, String, f64)> {
        let mut pairs = Vec::new();
        let matrix = self.matrix.read();
        
        for i in 0..self.symbols.len() {
            for j in (i + 1)..self.symbols.len() {
                let corr = matrix[[i, j]].abs();
                if corr > threshold {
                    pairs.push((
                        self.symbols[i].clone(),
                        self.symbols[j].clone(),
                        matrix[[i, j]],
                    ));
                }
            }
        }
        
        pairs
    }
}

/// Correlation analyzer for position risk
pub struct CorrelationAnalyzer {
    price_history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    correlation_matrix: Arc<CorrelationMatrix>,
    window_size: usize,
    min_samples: usize,
}

impl CorrelationAnalyzer {
    pub fn new(symbols: Vec<String>, window_size: usize) -> Self {
        Self {
            price_history: Arc::new(RwLock::new(HashMap::new())),
            correlation_matrix: Arc::new(CorrelationMatrix::new(symbols)),
            window_size,
            min_samples: 20, // Minimum samples for correlation
        }
    }
    
    /// Add price data point
    pub fn add_price(&self, symbol: String, price: f64) {
        let mut history = self.price_history.write();
        let prices = history.entry(symbol).or_default();
        prices.push(price);
        
        // Keep only window_size samples
        if prices.len() > self.window_size {
            prices.remove(0);
        }
    }
    
    /// Calculate and update correlations
    pub fn update_correlations(&self) {
        let history = self.price_history.read();
        let symbols: Vec<_> = history.keys().cloned().collect();
        
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                if let (Some(prices1), Some(prices2)) = (
                    history.get(&symbols[i]),
                    history.get(&symbols[j]),
                ) {
                    if prices1.len() >= self.min_samples && prices2.len() >= self.min_samples {
                        if let Some(correlation) = self.calculate_correlation(prices1, prices2) {
                            self.correlation_matrix.update_correlation(
                                &symbols[i],
                                &symbols[j],
                                correlation,
                            );
                        }
                    }
                }
            }
        }
    }
    
    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, prices1: &[f64], prices2: &[f64]) -> Option<f64> {
        if prices1.len() != prices2.len() || prices1.len() < 2 {
            return None;
        }
        
        // Calculate returns
        let returns1 = self.calculate_returns(prices1);
        let returns2 = self.calculate_returns(prices2);
        
        if returns1.is_empty() || returns2.is_empty() {
            return None;
        }
        
        // Calculate correlation
        let mean1 = returns1.clone().mean();
        let mean2 = returns2.clone().mean();
        let std1 = returns1.clone().std_dev();
        let std2 = returns2.clone().std_dev();
        
        if std1 == 0.0 || std2 == 0.0 {
            return None;
        }
        
        let covariance: f64 = returns1
            .iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| (r1 - mean1) * (r2 - mean2))
            .sum::<f64>() / (returns1.len() - 1) as f64;
        
        Some(covariance / (std1 * std2))
    }
    
    /// Calculate returns from prices
    fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
    
    /// Check if adding a position would violate correlation limits
    pub fn check_correlation_limit(
        &self,
        symbol: &str,
        existing_positions: &[String],
        max_correlation: f64,
    ) -> Result<(), String> {
        for existing in existing_positions {
            if let Some(correlation) = self.correlation_matrix.get_correlation(symbol, existing) {
                if correlation.abs() > max_correlation {
                    return Err(format!(
                        "Correlation between {} and {} is {:.2} (max {:.2})",
                        symbol, existing, correlation, max_correlation
                    ));
                }
            }
        }
        Ok(())
    }
    
    /// Get correlation risk score (0-1, higher is riskier)
    pub fn correlation_risk_score(&self, positions: &[String]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }
        
        let mut total_correlation = 0.0;
        let mut count = 0;
        
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if let Some(corr) = self.correlation_matrix.get_correlation(&positions[i], &positions[j]) {
                    total_correlation += corr.abs();
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            total_correlation / count as f64
        } else {
            0.0
        }
    }
}

/// Portfolio correlation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMetrics {
    pub average_correlation: f64,
    pub max_correlation: f64,
    pub correlation_pairs: Vec<(String, String, f64)>,
    pub risk_score: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CorrelationMetrics {
    pub fn calculate(analyzer: &CorrelationAnalyzer, positions: &[String]) -> Self {
        let mut correlations = Vec::new();
        
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if let Some(corr) = analyzer.correlation_matrix.get_correlation(&positions[i], &positions[j]) {
                    correlations.push((positions[i].clone(), positions[j].clone(), corr));
                }
            }
        }
        
        let max_correlation = correlations
            .iter()
            .map(|(_, _, c)| c.abs())
            .fold(0.0, f64::max);
        
        let average_correlation = if !correlations.is_empty() {
            correlations.iter().map(|(_, _, c)| c.abs()).sum::<f64>() / correlations.len() as f64
        } else {
            0.0
        };
        
        Self {
            average_correlation,
            max_correlation,
            correlation_pairs: correlations,
            risk_score: analyzer.correlation_risk_score(positions),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Check if correlations are within limits
    pub fn is_within_limits(&self, max_correlation: f64) -> bool {
        self.max_correlation <= max_correlation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_correlation_matrix() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let matrix = CorrelationMatrix::new(symbols);
        
        // Test diagonal is 1 (self-correlation)
        assert_eq!(matrix.get_correlation("BTC", "BTC"), Some(1.0));
        
        // Test update
        matrix.update_correlation("BTC", "ETH", 0.8);
        assert_eq!(matrix.get_correlation("BTC", "ETH"), Some(0.8));
        assert_eq!(matrix.get_correlation("ETH", "BTC"), Some(0.8)); // Symmetric
    }
    
    #[test]
    fn test_correlation_calculation() {
        let analyzer = CorrelationAnalyzer::new(
            vec!["BTC".to_string(), "ETH".to_string()],
            30,
        );
        
        // Add perfectly correlated prices
        for i in 0..30 {
            let price = 50000.0 + (i as f64 * 100.0);
            analyzer.add_price("BTC".to_string(), price);
            analyzer.add_price("ETH".to_string(), price * 0.1); // Perfectly correlated
        }
        
        analyzer.update_correlations();
        
        let correlation = analyzer.correlation_matrix.get_correlation("BTC", "ETH");
        assert!(correlation.is_some());
        // Should be close to 1.0 (perfect correlation)
        assert!(correlation.unwrap() > 0.99);
    }
    
    #[test]
    fn test_correlation_limit_check() {
        let analyzer = CorrelationAnalyzer::new(
            vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
            30,
        );
        
        analyzer.correlation_matrix.update_correlation("BTC", "ETH", 0.8);
        analyzer.correlation_matrix.update_correlation("BTC", "SOL", 0.3);
        
        // Should fail - correlation too high
        let result = analyzer.check_correlation_limit(
            "ETH",
            &["BTC".to_string()],
            0.7, // Max 0.7
        );
        assert!(result.is_err());
        
        // Should pass - correlation within limit
        let result = analyzer.check_correlation_limit(
            "SOL",
            &["BTC".to_string()],
            0.7,
        );
        assert!(result.is_ok());
    }
}