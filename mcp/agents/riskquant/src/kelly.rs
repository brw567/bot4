//! Kelly Criterion calculator for optimal position sizing

use anyhow::{Result, bail};

pub struct KellyCalculator;

impl KellyCalculator {
    pub fn new() -> Self {
        Self
    }
    
    /// Calculate Kelly fraction for binary outcome
    /// f* = (p*b - q) / b
    /// where:
    /// - p = probability of winning
    /// - q = probability of losing (1-p)
    /// - b = ratio of win amount to loss amount
    pub fn calculate(&self, win_prob: f64, win_return: f64, loss_return: f64) -> Result<f64> {
        // Validate inputs
        if win_prob < 0.0 || win_prob > 1.0 {
            bail!("Win probability must be between 0 and 1");
        }
        
        if win_return <= 0.0 {
            bail!("Win return must be positive");
        }
        
        if loss_return >= 0.0 {
            bail!("Loss return must be negative");
        }
        
        let loss_prob = 1.0 - win_prob;
        let b = win_return / loss_return.abs();
        
        // Kelly formula
        let kelly = (win_prob * b - loss_prob) / b;
        
        // Kelly can be negative (indicating don't bet) or very large (indicating edge is huge)
        // Cap at 0.5 (50% of capital) for safety
        Ok(kelly.max(0.0).min(0.5))
    }
    
    /// Calculate Kelly for multiple outcomes
    pub fn calculate_multi(&self, outcomes: &[(f64, f64)]) -> Result<f64> {
        if outcomes.is_empty() {
            bail!("No outcomes provided");
        }
        
        // Verify probabilities sum to 1
        let prob_sum: f64 = outcomes.iter().map(|(p, _)| p).sum();
        if (prob_sum - 1.0).abs() > 1e-6 {
            bail!("Probabilities must sum to 1, got {}", prob_sum);
        }
        
        // Numerical optimization to find Kelly fraction
        let mut best_kelly = 0.0;
        let mut best_growth = f64::NEG_INFINITY;
        
        for f in (0..=100).map(|i| i as f64 / 200.0) {
            let growth = self.expected_log_return(f, outcomes);
            if growth > best_growth {
                best_growth = growth;
                best_kelly = f;
            }
        }
        
        Ok(best_kelly)
    }
    
    fn expected_log_return(&self, fraction: f64, outcomes: &[(f64, f64)]) -> f64 {
        outcomes.iter()
            .map(|(prob, return_rate)| {
                let final_wealth = 1.0 + fraction * return_rate;
                if final_wealth > 0.0 {
                    prob * final_wealth.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kelly_basic() {
        let calc = KellyCalculator::new();
        
        // 60% win prob, 1:1 payoff
        let kelly = calc.calculate(0.6, 1.0, -1.0).unwrap();
        assert!((kelly - 0.2).abs() < 0.01); // Should be 20%
        
        // 55% win prob, 1:1 payoff  
        let kelly = calc.calculate(0.55, 1.0, -1.0).unwrap();
        assert!((kelly - 0.1).abs() < 0.01); // Should be 10%
        
        // 50% win prob, 1:1 payoff (no edge)
        let kelly = calc.calculate(0.5, 1.0, -1.0).unwrap();
        assert!(kelly.abs() < 0.01); // Should be 0%
    }
    
    #[test]
    fn test_kelly_multi() {
        let calc = KellyCalculator::new();
        
        // Multiple outcomes
        let outcomes = vec![
            (0.3, 2.0),   // 30% chance of 200% return
            (0.5, 0.5),   // 50% chance of 50% return  
            (0.2, -0.8),  // 20% chance of 80% loss
        ];
        
        let kelly = calc.calculate_multi(&outcomes).unwrap();
        assert!(kelly > 0.0 && kelly < 0.5);
    }
}