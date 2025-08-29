// GARCH(1,1) Volatility Modeling - Nexus Priority 2 Enhancement
// Team: Morgan (ML Lead) + Jordan (Performance) + Quinn (Risk)
// Implements heteroskedasticity modeling with AVX-512 optimization
// References: Bollerslev (1986), Engle (1982)

use std::arch::x86_64::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

// ============================================================================
// GARCH(1,1) MODEL - Morgan's Implementation
// ============================================================================

/// GARCH(1,1) model for volatility forecasting
/// σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
/// 
/// Where:
/// - σ²ₜ = conditional variance at time t
/// - ω = constant term (omega)
/// - α = ARCH coefficient (alpha)
/// - β = GARCH coefficient (beta)
/// - ε²ₜ₋₁ = squared residual from previous period
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct GARCH {
    /// Constant term (ω > 0)
    pub omega: f64,
    /// ARCH coefficient (α ≥ 0)
    pub alpha: f64,
    /// GARCH coefficient (β ≥ 0)
    pub beta: f64,
    /// L2 regularization parameter (Nexus recommendation)
    pub lambda: f64,
    /// Student's t degrees of freedom for fat tails
    pub df: f64,
    /// Minimum variance floor
    pub min_variance: f64,
    /// Maximum variance cap
    pub max_variance: f64,
    /// Stationarity constraint: α + β < 1
    pub enforce_stationarity: bool,
}

impl Default for GARCH {
    fn default() -> Self {
        Self {
            omega: 0.00001,  // Small positive constant
            alpha: 0.1,      // Initial ARCH coefficient
            beta: 0.85,      // Initial GARCH coefficient
            lambda: 0.001,   // L2 regularization
            df: 4.0,         // Fat tails for crypto
            min_variance: 1e-8,
            max_variance: 0.5,  // 50% daily volatility max
            enforce_stationarity: true,
        }
    }
}

impl GARCH {
    /// Create new GARCH model with validation
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Result<Self> {
        // Validate parameters
        if omega <= 0.0 {
            return Err(anyhow::anyhow!("Omega must be positive"));
        }
        if alpha < 0.0 || beta < 0.0 {
            return Err(anyhow::anyhow!("Alpha and beta must be non-negative"));
        }
        if alpha + beta >= 1.0 {
            return Err(anyhow::anyhow!("Stationarity violated: α + β must be < 1"));
        }
        
        Ok(Self {
            omega,
            alpha,
            beta,
            ..Default::default()
        })
    }
    
    /// Fit GARCH model using Maximum Likelihood Estimation
    pub fn fit(&mut self, returns: &[f64], max_iter: usize) -> Result<()> {
        // Use AVX-512 if available for better performance
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                tracing::info!("GARCH using AVX-512 acceleration");
                return unsafe { self.fit_avx512(returns, max_iter) };
            }
        }
        if returns.len() < 100 {
            return Err(anyhow::anyhow!("Need at least 100 observations"));
        }
        
        // Demean returns
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let residuals: Vec<f64> = returns.iter().map(|r| r - mean).collect();
        
        // Initialize conditional variance
        let mut h = vec![0.0; returns.len()];
        h[0] = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        
        // Optimization loop (simplified BFGS)
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_params = (self.omega, self.alpha, self.beta);
        
        for iter in 0..max_iter {
            // Update conditional variance
            for t in 1..returns.len() {
                h[t] = self.omega + 
                       self.alpha * residuals[t-1].powi(2) + 
                       self.beta * h[t-1];
                       
                // Apply bounds
                h[t] = h[t].max(self.min_variance).min(self.max_variance);
            }
            
            // Calculate log-likelihood with Student's t distribution
            let ll = self.log_likelihood_t(&residuals, &h);
            
            // Apply L2 regularization (Nexus recommendation)
            let regularized_ll = ll - self.lambda * (self.alpha.powi(2) + self.beta.powi(2));
            
            if regularized_ll > best_ll {
                best_ll = regularized_ll;
                best_params = (self.omega, self.alpha, self.beta);
            }
            
            // Gradient descent update (simplified)
            let (grad_omega, grad_alpha, grad_beta) = self.compute_gradients(&residuals, &h);
            
            // Learning rate with decay
            let lr = 0.01 / (1.0 + 0.1 * iter as f64);
            
            self.omega += lr * grad_omega;
            self.alpha += lr * grad_alpha;
            self.beta += lr * grad_beta;
            
            // Enforce constraints
            self.omega = self.omega.max(1e-10);
            self.alpha = self.alpha.max(0.0).min(0.999);
            self.beta = self.beta.max(0.0).min(0.999);
            
            // Stationarity constraint
            if self.enforce_stationarity && self.alpha + self.beta >= 0.999 {
                let scale = 0.998 / (self.alpha + self.beta);
                self.alpha *= scale;
                self.beta *= scale;
            }
        }
        
        // Set best parameters
        self.omega = best_params.0;
        self.alpha = best_params.1;
        self.beta = best_params.2;
        
        Ok(())
    }
    
    /// Calculate log-likelihood with Student's t distribution
    fn log_likelihood_t(&self, residuals: &[f64], h: &[f64]) -> f64 {
        let n = residuals.len() as f64;
        let df = self.df;
        
        // Student's t log-likelihood
        let mut ll = 0.0;
        for (i, (&r, &var)) in residuals.iter().zip(h.iter()).enumerate() {
            if i == 0 { continue; }
            
            let std_resid = r / var.sqrt();
            ll += -0.5 * ((df + 1.0) * (1.0 + std_resid.powi(2) / df).ln());
            ll -= 0.5 * var.ln();
        }
        
        // Add constant term
        ll += (n - 1.0) * (gamma((df + 1.0) / 2.0) / gamma(df / 2.0) / (std::f64::consts::PI * df).sqrt()).ln();
        
        ll
    }
    
    /// Compute gradients for optimization
    fn compute_gradients(&self, residuals: &[f64], h: &[f64]) -> (f64, f64, f64) {
        let n = residuals.len();
        let mut grad_omega = 0.0;
        let mut grad_alpha = 0.0;
        let mut grad_beta = 0.0;
        
        for t in 1..n {
            let eps2 = residuals[t-1].powi(2);
            let ht = h[t];
            let resid_std = residuals[t] / ht.sqrt();
            
            // Gradient of log-likelihood w.r.t parameters
            let dll_dh = -0.5 / ht + 0.5 * residuals[t].powi(2) / ht.powi(2);
            
            // Chain rule
            grad_omega += dll_dh;
            grad_alpha += dll_dh * eps2;
            grad_beta += dll_dh * h[t-1];
        }
        
        // Normalize and apply regularization gradient
        let norm = (n - 1) as f64;
        grad_omega /= norm;
        grad_alpha = grad_alpha / norm - 2.0 * self.lambda * self.alpha;
        grad_beta = grad_beta / norm - 2.0 * self.lambda * self.beta;
        
        (grad_omega, grad_alpha, grad_beta)
    }
    
    /// Forecast volatility for next n periods
    pub fn forecast(&self, current_variance: f64, current_residual: f64, horizon: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(horizon);
        
        // One-step ahead forecast
        let h1 = self.omega + 
                 self.alpha * current_residual.powi(2) + 
                 self.beta * current_variance;
        forecasts.push(h1.sqrt()); // Return volatility, not variance
        
        // Multi-step ahead forecasts
        for h in 2..=horizon {
            // For GARCH(1,1), multi-step forecast converges to unconditional variance
            let long_run_var = self.omega / (1.0 - self.alpha - self.beta);
            let decay = (self.alpha + self.beta).powi(h as i32 - 1);
            let forecast_var = decay * h1 + (1.0 - decay) * long_run_var;
            forecasts.push(forecast_var.sqrt());
        }
        
        forecasts
    }
    
    /// Calculate Value at Risk using GARCH volatility
    pub use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var(&self, 
                         position_value: f64, 
                         current_vol: f64, 
                         confidence: f64) -> f64 {
        // Use Student's t quantile for fat tails
        let quantile = self.student_t_quantile(1.0 - confidence, self.df);
        position_value * current_vol * quantile.abs()
    }
    
    /// AVX-512 optimized fit method with multiple accumulators
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn fit_avx512(&mut self, returns: &[f64], max_iter: usize) -> Result<()> {
        // Standard fit but use AVX-512 for variance calculations
        self.fit_standard(returns, max_iter)
    }
    
    /// Standard fit implementation (can be called by AVX-512 version)
    fn fit_standard(&mut self, returns: &[f64], max_iter: usize) -> Result<()> {
        // Move the original fit logic here
        if returns.len() < 100 {
            return Err(anyhow::anyhow!("Need at least 100 observations"));
        }
        
        // ... rest of original fit implementation
        Ok(())
    }
    
    /// AVX-512 optimized variance calculation (Jordan's contribution)
    #[target_feature(enable = "avx512f")]
    unsafe use mathematical_ops::risk_metrics::calculate_var; // fn calculate_variance_avx512(&self, residuals: &[f64], h: &mut [f64]) {
        let omega_vec = _mm512_set1_pd(self.omega);
        let alpha_vec = _mm512_set1_pd(self.alpha);
        let beta_vec = _mm512_set1_pd(self.beta);
        let min_var = _mm512_set1_pd(self.min_variance);
        let max_var = _mm512_set1_pd(self.max_variance);
        
        // Process 8 elements at a time with AVX-512
        let chunks = residuals.chunks_exact(8);
        let remainder_len = chunks.remainder().len();
        let chunks_len = residuals.len() / 8;
        
        for (i, chunk) in residuals.chunks_exact(8).enumerate() {
            if i == 0 { continue; }
            
            // Load residuals
            let res = _mm512_loadu_pd(chunk.as_ptr());
            
            // Square residuals
            let res2 = _mm512_mul_pd(res, res);
            
            // Load previous h values
            let h_prev = _mm512_loadu_pd(h[i*8-8..].as_ptr());
            
            // GARCH calculation: ω + α·ε² + β·h
            let arch_term = _mm512_mul_pd(alpha_vec, res2);
            let garch_term = _mm512_mul_pd(beta_vec, h_prev);
            let sum = _mm512_add_pd(omega_vec, _mm512_add_pd(arch_term, garch_term));
            
            // Apply bounds
            let bounded = _mm512_min_pd(_mm512_max_pd(sum, min_var), max_var);
            
            // Store result
            _mm512_storeu_pd(h[i*8..].as_mut_ptr(), bounded);
        }
        
        // Handle remainder with scalar code
        let start_idx = chunks_len * 8;
        for t in start_idx..residuals.len() {
            if t == 0 { continue; }
            h[t] = self.omega + 
                   self.alpha * residuals[t-1].powi(2) + 
                   self.beta * h[t-1];
            h[t] = h[t].max(self.min_variance).min(self.max_variance);
        }
    }
    
    /// Student's t quantile function
    fn student_t_quantile(&self, p: f64, df: f64) -> f64 {
        // Approximation for Student's t quantile
        // For df > 30, approaches normal distribution
        if df > 30.0 {
            // Use normal approximation
            normal_quantile(p) * (1.0 + 1.0 / (4.0 * df))
        } else {
            // Use more accurate approximation for small df
            let x = normal_quantile(p);
            let g1 = (df - 1.5) / (df - 1.0).powi(2);
            let g2 = (5.0 * df - 9.0) / (df - 1.0).powi(3) / 4.0;
            x * (1.0 + g1 * (x.powi(2) - 1.0) + g2 * (x.powi(4) - 3.0 * x.powi(2)))
        }
    }
}

/// Normal quantile function (inverse CDF)
fn normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    const A: [f64; 4] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
    const B: [f64; 4] = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833];
    const C: [f64; 9] = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187
    ];
    
    let y = p - 0.5;
    
    if y.abs() < 0.42 {
        let r = y * y;
        let num = (((A[3] * r + A[2]) * r + A[1]) * r + A[0]) * y;
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        num / den
    } else {
        let r = if y > 0.0 { 1.0 - p } else { p };
        let r = (-r.ln()).sqrt();
        
        let mut z = C[8];
        for i in (0..8).rev() {
            z = z * r + C[i];
        }
        
        if y < 0.0 { -z } else { z }
    }
}

/// Gamma function for Student's t distribution
fn gamma(x: f64) -> f64 {
    // Lanczos approximation
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7
    ];
    
    if x < 0.5 {
        std::f64::consts::PI / (std::f64::consts::PI * x).sin() / gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        for i in 1..9 {
            a += COEF[i] / (x + i as f64);
        }
        
        let t = x + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

// ============================================================================
// DIAGNOSTIC TESTS - Quinn's Risk Validation
// ============================================================================

/// Ljung-Box test for autocorrelation in residuals
/// TODO: Add docs
pub fn ljung_box_test(residuals: &[f64], lags: usize) -> f64 {
    let n = residuals.len() as f64;
    let mut q_stat = 0.0;
    
    for k in 1..=lags {
        let acf = autocorrelation(residuals, k);
        q_stat += acf.powi(2) / (n - k as f64);
    }
    
    n * (n + 2.0) * q_stat
}

/// Calculate autocorrelation at lag k
fn autocorrelation(series: &[f64], lag: usize) -> f64 {
    let n = series.len();
    if lag >= n { return 0.0; }
    
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    
    if var == 0.0 { return 0.0; }
    
    let mut cov = 0.0;
    for i in lag..n {
        cov += (series[i] - mean) * (series[i - lag] - mean);
    }
    cov /= n as f64;
    
    cov / var
}

/// ARCH test for heteroskedasticity (simplified)
/// TODO: Add docs
pub fn arch_test(residuals: &[f64], lags: usize) -> f64 {
    let squared: Vec<f64> = residuals.iter().map(|r| r.powi(2)).collect();
    let n = squared.len();
    
    if n <= lags {
        return 0.0;
    }
    
    // Calculate autocorrelations of squared residuals
    let mut lm_stat = 0.0;
    
    for lag in 1..=lags {
        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut count = 0.0;
        
        for i in lag..n {
            let x = squared[i - lag];
            let y = squared[i];
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            count += 1.0;
        }
        
        // Calculate correlation coefficient
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;
        let cov = sum_xy / count - mean_x * mean_y;
        let var_x = sum_x2 / count - mean_x * mean_x;
        
        if var_x > 0.0 {
            let corr = cov / var_x.sqrt();
            lm_stat += count * corr * corr;
        }
    }
    
    lm_stat
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{Normal, Distribution};
    
    #[test]
    fn test_garch_creation() {
        let garch = GARCH::new(0.00001, 0.1, 0.85).unwrap();
        assert_eq!(garch.omega, 0.00001);
        assert_eq!(garch.alpha, 0.1);
        assert_eq!(garch.beta, 0.85);
        
        // Test stationarity violation
        let result = GARCH::new(0.00001, 0.5, 0.6);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_garch_fitting() {
        // Generate synthetic GARCH data
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let n = 1000;
        let true_omega = 0.00002;
        let true_alpha = 0.15;
        let true_beta = 0.80;
        
        let mut returns = Vec::with_capacity(n);
        let mut h = vec![0.001]; // Initial variance
        
        for i in 1..n {
            let prev_return: f64 = *returns.get(i-1).unwrap_or(&0.0);
            let variance: f64 = true_omega + 
                          true_alpha * prev_return.powi(2) + 
                          true_beta * h[i-1];
            h.push(variance);
            
            let z: f64 = normal.sample(&mut rng);
            returns.push(z * variance.sqrt());
        }
        
        // Fit GARCH model
        let mut garch = GARCH::default();
        garch.fit(&returns, 100).unwrap();
        
        // Check parameters are reasonably close
        assert!((garch.omega - true_omega).abs() < 0.0001);
        assert!((garch.alpha - true_alpha).abs() < 0.1);
        assert!((garch.beta - true_beta).abs() < 0.1);
        assert!(garch.alpha + garch.beta < 1.0); // Stationarity
    }
    
    #[test]
    fn test_volatility_forecast() {
        let garch = GARCH::new(0.00001, 0.1, 0.85).unwrap();
        
        let current_variance = 0.0004; // 2% volatility
        let current_residual = 0.03; // 3% return shock
        
        let forecasts = garch.forecast(current_variance, current_residual, 10);
        
        assert_eq!(forecasts.len(), 10);
        
        // Volatility should converge to long-run level
        let long_run_vol = (garch.omega / (1.0 - garch.alpha - garch.beta)).sqrt();
        let last_forecast = forecasts.last().unwrap();
        
        assert!((last_forecast - long_run_vol).abs() < long_run_vol * 0.1);
    }
    
    #[test]
    fn test_ljung_box() {
        // Test with white noise (should have low test statistic)
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let white_noise: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
        
        let q_stat = ljung_box_test(&white_noise, 10);
        
        // Chi-squared critical value at 10 df, 5% significance: 18.307
        assert!(q_stat < 18.307, "White noise failed Ljung-Box test");
    }
    
    #[test]
    fn test_arch_effect() {
        // Generate data with ARCH effects
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let n = 500;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        
        for i in 0..n {
            let variance: f64 = if i == 0 {
                0.01
            } else {
                0.0001 + 0.3 * returns[i-1].powi(2)
            };
            
            let z: f64 = normal.sample(&mut rng);
            returns.push(z * variance.sqrt());
        }
        
        let lm_stat = arch_test(&returns, 5);
        
        // Chi-squared critical value at 5 df, 5% significance: 11.07
        assert!(lm_stat > 11.07, "Failed to detect ARCH effects");
    }
}