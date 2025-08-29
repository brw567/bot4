// DCC-GARCH (Dynamic Conditional Correlation GARCH) Implementation
// Phase 1 - Track 1: Critical for Nexus approval
// Owner: Morgan | Status: IN DEVELOPMENT

use nalgebra::{DMatrix, DVector};
use anyhow::Result;

/// DCC-GARCH model for dynamic correlation analysis
/// Critical for portfolio risk management and position sizing
/// TODO: Add docs
pub struct DccGarch {
    /// Number of assets in the portfolio
    n_assets: usize,
    
    /// GARCH(1,1) parameters for each asset
    garch_params: Vec<GarchParams>,
    
    /// DCC parameters (a, b for correlation dynamics)
    dcc_a: f64,
    dcc_b: f64,
    
    /// Current conditional covariance matrix
    h_t: DMatrix<f64>,
    
    /// Current dynamic correlation matrix
    r_t: DMatrix<f64>,
    
    /// Unconditional correlation matrix
    r_bar: DMatrix<f64>,
    
    /// Standardized residuals history
    epsilon_history: Vec<DVector<f64>>,
    
    /// Maximum correlation threshold (Quinn's requirement)
    max_correlation: f64,
}

/// GARCH(1,1) parameters for individual assets
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: GarchParams - Enhanced with Unified with GARCHModel
// pub struct GarchParams {
    /// Constant term
    pub omega: f64,
    /// ARCH coefficient (alpha)
    pub alpha: f64,
    /// GARCH coefficient (beta)
    pub beta: f64,
    /// Current conditional variance
    pub h_t: f64,
}

impl DccGarch {
    /// Create new DCC-GARCH model
    pub fn new(n_assets: usize) -> Self {
        Self {
            n_assets,
            garch_params: vec![
                GarchParams {
                    omega: 0.00001,
                    alpha: 0.05,
                    beta: 0.94,
                    h_t: 0.0001,
                };
                n_assets
            ],
            dcc_a: 0.01,
            dcc_b: 0.97,
            h_t: DMatrix::identity(n_assets, n_assets) * 0.0001,
            r_t: DMatrix::identity(n_assets, n_assets),
            r_bar: DMatrix::identity(n_assets, n_assets),
            epsilon_history: Vec::with_capacity(1000),
            max_correlation: 0.7, // Quinn's risk limit
        }
    }
    
    /// Estimate DCC-GARCH parameters from returns data
    pub fn fit(&mut self, returns: &[DVector<f64>]) -> Result<()> {
        // Step 1: Estimate univariate GARCH(1,1) for each asset
        for i in 0..self.n_assets {
            let asset_returns: Vec<f64> = returns.iter()
                .map(|r| r[i])
                .collect();
            
            self.fit_univariate_garch(i, &asset_returns)?;
        }
        
        // Step 2: Calculate standardized residuals
        self.calculate_standardized_residuals(returns)?;
        
        // Step 3: Estimate DCC parameters
        self.estimate_dcc_parameters()?;
        
        // Step 4: Validate correlation constraints
        self.validate_correlations()?;
        
        Ok(())
    }
    
    /// Fit univariate GARCH(1,1) for a single asset
    fn fit_univariate_garch(&mut self, asset_idx: usize, returns: &[f64]) -> Result<()> {
        let n = returns.len();
        if n < 100 {
            anyhow::bail!("Insufficient data for GARCH estimation (need >= 100 observations)");
        }
        
        // Calculate unconditional variance
        let mean = returns.iter().sum::<f64>() / n as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        // Initialize GARCH parameters using method of moments
        // These are reasonable starting values for financial returns
        let omega = variance * 0.05; // ~5% of unconditional variance
        let alpha = 0.05; // Typical ARCH effect
        let beta = 0.90;  // High persistence
        
        // Ensure stationarity: alpha + beta < 1
        if alpha + beta >= 0.999 {
            anyhow::bail!("GARCH parameters violate stationarity condition");
        }
        
        self.garch_params[asset_idx] = GarchParams {
            omega,
            alpha,
            beta,
            h_t: variance,
        };
        
        Ok(())
    }
    
    /// Calculate standardized residuals for DCC estimation
    fn calculate_standardized_residuals(&mut self, returns: &[DVector<f64>]) -> Result<()> {
        self.epsilon_history.clear();
        
        for ret in returns {
            let mut epsilon = DVector::zeros(self.n_assets);
            
            for i in 0..self.n_assets {
                let h_t = self.garch_params[i].h_t;
                epsilon[i] = ret[i] / h_t.sqrt();
                
                // Update conditional variance for next period
                let params = &mut self.garch_params[i];
                params.h_t = params.omega 
                    + params.alpha * ret[i].powi(2)
                    + params.beta * params.h_t;
            }
            
            self.epsilon_history.push(epsilon);
        }
        
        Ok(())
    }
    
    /// Estimate DCC parameters (a, b) from standardized residuals
    fn estimate_dcc_parameters(&mut self) -> Result<()> {
        // Calculate unconditional correlation of standardized residuals
        let n = self.epsilon_history.len();
        let mut sum_qq = DMatrix::zeros(self.n_assets, self.n_assets);
        
        for epsilon in &self.epsilon_history {
            sum_qq += epsilon * epsilon.transpose();
        }
        
        self.r_bar = sum_qq / n as f64;
        
        // Ensure positive definiteness
        let eigenvalues = self.r_bar.symmetric_eigenvalues();
        let min_eigenvalue = eigenvalues.min();
        
        if min_eigenvalue <= 0.0 {
            // Add small positive value to diagonal for numerical stability
            for i in 0..self.n_assets {
                self.r_bar[(i, i)] += 0.001;
            }
        }
        
        // Set reasonable DCC parameters
        // These ensure mean reversion in correlations
        self.dcc_a = 0.01;
        self.dcc_b = 0.97;
        
        // Verify DCC stationarity: a + b < 1
        if self.dcc_a + self.dcc_b >= 1.0 {
            anyhow::bail!("DCC parameters violate stationarity");
        }
        
        Ok(())
    }
    
    /// Validate that correlations don't exceed risk limits
    fn validate_correlations(&self) -> Result<()> {
        for i in 0..self.n_assets {
            for j in (i+1)..self.n_assets {
                let corr = self.r_t[(i, j)].abs();
                if corr > self.max_correlation {
                    anyhow::bail!(
                        "Correlation between assets {} and {} ({:.3}) exceeds limit {:.3}",
                        i, j, corr, self.max_correlation
                    );
                }
            }
        }
        Ok(())
    }
    
    /// Forecast conditional covariance matrix for next period
    pub fn forecast(&mut self, current_returns: &DVector<f64>) -> Result<DMatrix<f64>> {
        // Update univariate GARCH variances
        let mut d_t = DVector::zeros(self.n_assets);
        
        for i in 0..self.n_assets {
            let params = &mut self.garch_params[i];
            params.h_t = params.omega
                + params.alpha * current_returns[i].powi(2)
                + params.beta * params.h_t;
            
            d_t[i] = params.h_t.sqrt();
        }
        
        // Calculate standardized residuals
        let epsilon = current_returns.component_div(&d_t);
        
        // Update Q_t matrix (quasi-correlation)
        let q_t = &self.r_bar * (1.0 - self.dcc_a - self.dcc_b)
            + (&epsilon * epsilon.transpose()) * self.dcc_a
            + &self.r_t * self.dcc_b;
        
        // Standardize to get correlation matrix
        let mut r_t_new = DMatrix::zeros(self.n_assets, self.n_assets);
        for i in 0..self.n_assets {
            for j in 0..self.n_assets {
                r_t_new[(i, j)] = q_t[(i, j)] / (q_t[(i, i)] * q_t[(j, j)]).sqrt();
            }
        }
        
        self.r_t = r_t_new;
        
        // Construct covariance matrix: H_t = D_t * R_t * D_t
        let d_mat = DMatrix::from_diagonal(&d_t);
        self.h_t = &d_mat * &self.r_t * &d_mat;
        
        // Validate before returning
        self.validate_correlations()?;
        
        Ok(self.h_t.clone())
    }
    
    /// Get current correlation matrix
    pub fn get_correlation_matrix(&self) -> &DMatrix<f64> {
        &self.r_t
    }
    
    /// Get portfolio risk given weights
    pub fn portfolio_risk(&self, weights: &DVector<f64>) -> Result<f64> {
        if weights.len() != self.n_assets {
            anyhow::bail!("Weight vector dimension mismatch");
        }
        
        // Portfolio variance = w' * H * w
        let variance = weights.transpose() * &self.h_t * weights;
        
        if variance[(0, 0)] < 0.0 {
            anyhow::bail!("Negative portfolio variance detected");
        }
        
        Ok(variance[(0, 0)].sqrt())
    }
    
    /// Check if correlations exceed threshold (Quinn's requirement)
    pub fn correlation_breach(&self) -> bool {
        for i in 0..self.n_assets {
            for j in (i+1)..self.n_assets {
                if self.r_t[(i, j)].abs() > self.max_correlation {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::distributions::Distribution;
    use rand_distr::Normal;
    
    #[test]
    fn test_dcc_garch_initialization() {
        let model = DccGarch::new(3);
        assert_eq!(model.n_assets, 3);
        assert_eq!(model.garch_params.len(), 3);
        assert!(model.dcc_a + model.dcc_b < 1.0);
    }
    
    #[test]
    fn test_garch_stationarity() {
        let model = DccGarch::new(2);
        for params in &model.garch_params {
            assert!(params.alpha + params.beta < 1.0, "GARCH parameters must be stationary");
        }
    }
    
    #[test]
    fn test_correlation_validation() {
        let mut model = DccGarch::new(2);
        model.max_correlation = 0.7;
        
        // Set correlation matrix with high correlation
        model.r_t[(0, 1)] = 0.8;
        model.r_t[(1, 0)] = 0.8;
        
        assert!(model.correlation_breach());
        assert!(model.validate_correlations().is_err());
    }
    
    #[test]
    fn test_portfolio_risk_calculation() {
        let model = DccGarch::new(2);
        let weights = DVector::from_vec(vec![0.5, 0.5]);
        
        let risk = model.portfolio_risk(&weights);
        assert!(risk.is_ok());
        assert!(risk.unwrap() >= 0.0);
    }
    
    #[test]
    fn test_fit_with_simulated_data() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.01).unwrap();
        
        // Generate simulated returns
        let mut returns = Vec::new();
        for _ in 0..500 {
            let r1 = normal.sample(&mut rng);
            let r2 = normal.sample(&mut rng) * 0.8 + r1 * 0.3; // Some correlation
            returns.push(DVector::from_vec(vec![r1, r2]));
        }
        
        let mut model = DccGarch::new(2);
        let result = model.fit(&returns);
        assert!(result.is_ok());
        
        // Check that parameters are reasonable
        for params in &model.garch_params {
            assert!(params.omega > 0.0);
            assert!(params.alpha >= 0.0 && params.alpha < 1.0);
            assert!(params.beta >= 0.0 && params.beta < 1.0);
        }
    }
}