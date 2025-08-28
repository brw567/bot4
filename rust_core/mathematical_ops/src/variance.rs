//! # Variance and Value at Risk (VaR) Calculations - Consolidated Implementation
//! 
//! Replaces 8+ duplicate VaR implementations with one optimized, comprehensive version.
//! Supports multiple VaR calculation methods with proper risk management integration.
//!
//! ## Methods Supported
//! - Historical VaR
//! - Parametric VaR (Normal and Student-t)
//! - Monte Carlo VaR
//! - GARCH-based VaR
//! - Cornish-Fisher VaR (adjusts for skewness/kurtosis)
//! - Conditional VaR (CVaR/Expected Shortfall)
//!
//! ## External Research Applied
//! - "Value at Risk" (Jorion, 2007)
//! - "Risk Management and Financial Institutions" (Hull)
//! - Basel III regulatory framework
//! - "Beyond Value at Risk" (Dowd) - CVaR techniques
//! - JP Morgan RiskMetrics methodology

use ndarray::{Array1, Array2};
use statrs::distribution::{Normal, StudentsT, ContinuousCDF};
use std::collections::BTreeMap;
use thiserror::Error;
use tracing::{debug, trace, warn};

#[cfg(feature = "simd")]
use crate::simd::simd_variance;

/// Variance/VaR calculation errors
#[derive(Debug, Error)]
pub enum VarError {
    #[error("Insufficient data: need at least {min} points, got {actual}")]
    InsufficientData { min: usize, actual: usize },
    
    #[error("Invalid confidence level: {level} (must be between 0 and 1)")]
    InvalidConfidence { level: f64 },
    
    #[error("Invalid time horizon: {horizon} days")]
    InvalidHorizon { horizon: usize },
    
    #[error("Numerical error in calculation: {details}")]
    NumericalError { details: String },
    
    #[error("Distribution fitting failed: {details}")]
    DistributionError { details: String },
}

/// VaR calculation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarMethod {
    /// Historical simulation
    Historical,
    /// Parametric with normal distribution
    ParametricNormal,
    /// Parametric with Student's t distribution
    ParametricStudentT { df: f64 },
    /// Monte Carlo simulation
    MonteCarlo { simulations: usize },
    /// GARCH-based volatility modeling
    Garch,
    /// Cornish-Fisher expansion (accounts for higher moments)
    CornishFisher,
    /// Filtered Historical Simulation
    FilteredHistorical,
}

/// Configuration for VaR calculations
#[derive(Debug, Clone)]
pub struct VarConfig {
    /// Confidence level (e.g., 0.95 for 95% VaR)
    pub confidence_level: f64,
    /// Time horizon in days
    pub horizon: usize,
    /// Method to use
    pub method: VarMethod,
    /// Number of historical periods to use
    pub lookback_periods: usize,
    /// Apply volatility scaling
    pub volatility_scaling: bool,
    /// Use SIMD if available
    pub use_simd: bool,
}

impl Default for VarConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.99,  // 99% confidence (Basel III standard)
            horizon: 10,  // 10-day horizon (regulatory standard)
            method: VarMethod::Historical,
            lookback_periods: 252,  // 1 year of daily data
            volatility_scaling: true,
            use_simd: cfg!(feature = "simd"),
        }
    }
}

/// VaR calculation result with additional metrics
#[derive(Debug, Clone)]
pub struct VarResult {
    /// Value at Risk (as positive number representing potential loss)
    pub var: f64,
    /// Conditional VaR (Expected Shortfall)
    pub cvar: Option<f64>,
    /// Confidence level used
    pub confidence_level: f64,
    /// Time horizon in days
    pub horizon: usize,
    /// Method used
    pub method: VarMethod,
    /// Additional diagnostics
    pub diagnostics: VarDiagnostics,
}

/// Diagnostic information for VaR calculation
#[derive(Debug, Clone)]
pub struct VarDiagnostics {
    /// Sample mean return
    pub mean_return: f64,
    /// Sample volatility
    pub volatility: f64,
    /// Skewness of returns
    pub skewness: f64,
    /// Excess kurtosis of returns
    pub kurtosis: f64,
    /// Number of observations used
    pub observations: usize,
    /// Number of breaches in backtest (if performed)
    pub backtest_breaches: Option<usize>,
}

/// Calculate Value at Risk for a returns series
///
/// This is the main entry point that replaces all 8+ duplicate implementations.
/// Returns are expected as decimal returns (e.g., 0.01 for 1% gain).
///
/// # Example
/// ```rust
/// use mathematical_ops::variance::{calculate_var, VarMethod};
/// 
/// let returns = vec![-0.02, 0.01, -0.03, 0.02, -0.01, 0.03, -0.015, 0.025];
/// let var = calculate_var(&returns, 0.95, 1, VarMethod::Historical).unwrap();
/// println!("95% 1-day VaR: {:.2}%", var * 100.0);
/// ```
pub use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var(
    returns: &[f64],
    confidence_level: f64,
    horizon: usize,
    method: VarMethod,
) -> Result<VarResult, VarError> {
    calculate_var_with_config(returns, &VarConfig {
        confidence_level,
        horizon,
        method,
        ..Default::default()
    })
}

/// Calculate VaR with custom configuration
pub use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var_with_config(
    returns: &[f64],
    config: &VarConfig,
) -> Result<VarResult, VarError> {
    // Validation
    if returns.len() < config.lookback_periods.min(30) {
        return Err(VarError::InsufficientData {
            min: config.lookback_periods.min(30),
            actual: returns.len(),
        });
    }
    
    if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
        return Err(VarError::InvalidConfidence {
            level: config.confidence_level,
        });
    }
    
    if config.horizon == 0 {
        return Err(VarError::InvalidHorizon {
            horizon: config.horizon,
        });
    }
    
    // Calculate basic statistics
    let diagnostics = calculate_diagnostics(returns);
    
    // Calculate VaR based on method
    let (var, cvar) = match config.method {
        VarMethod::Historical => {
            calculate_historical_var(returns, config)?
        }
        VarMethod::ParametricNormal => {
            calculate_parametric_normal_var(returns, config, &diagnostics)?
        }
        VarMethod::ParametricStudentT { df } => {
            calculate_parametric_studentt_var(returns, config, &diagnostics, df)?
        }
        VarMethod::MonteCarlo { simulations } => {
            calculate_monte_carlo_var(returns, config, &diagnostics, simulations)?
        }
        VarMethod::Garch => {
            calculate_garch_var(returns, config)?
        }
        VarMethod::CornishFisher => {
            calculate_cornish_fisher_var(returns, config, &diagnostics)?
        }
        VarMethod::FilteredHistorical => {
            calculate_filtered_historical_var(returns, config)?
        }
    };
    
    Ok(VarResult {
        var,
        cvar,
        confidence_level: config.confidence_level,
        horizon: config.horizon,
        method: config.method,
        diagnostics,
    })
}

/// Historical VaR calculation
fn calculate_historical_var(
    returns: &[f64],
    config: &VarConfig,
) -> Result<(f64, Option<f64>), VarError> {
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Find the quantile position
    let alpha = 1.0 - config.confidence_level;
    let position = (alpha * returns.len() as f64) as usize;
    let position = position.max(0).min(returns.len() - 1);
    
    // Get VaR (negative of quantile for loss representation)
    let daily_var = -sorted_returns[position];
    
    // Calculate CVaR (average of returns worse than VaR)
    let cvar = if position > 0 {
        let worst_returns = &sorted_returns[..=position];
        Some(-worst_returns.iter().sum::<f64>() / worst_returns.len() as f64)
    } else {
        None
    };
    
    // Scale to horizon using square root rule (assumes IID returns)
    let horizon_adjustment = if config.volatility_scaling {
        (config.horizon as f64).sqrt()
    } else {
        config.horizon as f64
    };
    
    Ok((
        daily_var * horizon_adjustment,
        cvar.map(|cv| cv * horizon_adjustment)
    ))
}

/// Parametric VaR with normal distribution
fn calculate_parametric_normal_var(
    returns: &[f64],
    config: &VarConfig,
    diagnostics: &VarDiagnostics,
) -> Result<(f64, Option<f64>), VarError> {
    let normal = Normal::new(0.0, 1.0).map_err(|e| VarError::DistributionError {
        details: e.to_string()
    })?;
    
    // Get z-score for confidence level
    let z_score = normal.inverse_cdf(1.0 - config.confidence_level);
    
    // Calculate daily VaR
    let daily_var = -diagnostics.mean_return + diagnostics.volatility * (-z_score);
    
    // Calculate CVaR using formula for normal distribution
    let phi_z = normal.pdf(z_score);
    let cvar = diagnostics.volatility * phi_z / (1.0 - config.confidence_level)
        - diagnostics.mean_return;
    
    // Scale to horizon
    let horizon_adjustment = if config.volatility_scaling {
        (config.horizon as f64).sqrt()
    } else {
        config.horizon as f64
    };
    
    Ok((
        daily_var * horizon_adjustment,
        Some(cvar * horizon_adjustment)
    ))
}

/// Parametric VaR with Student's t distribution (fat tails)
fn calculate_parametric_studentt_var(
    returns: &[f64],
    config: &VarConfig,
    diagnostics: &VarDiagnostics,
    df: f64,
) -> Result<(f64, Option<f64>), VarError> {
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| VarError::DistributionError {
        details: e.to_string()
    })?;
    
    // Get t-score for confidence level
    let t_score = t_dist.inverse_cdf(1.0 - config.confidence_level);
    
    // Adjust volatility for t-distribution
    let adjusted_vol = diagnostics.volatility * ((df - 2.0) / df).sqrt();
    
    // Calculate daily VaR
    let daily_var = -diagnostics.mean_return + adjusted_vol * (-t_score);
    
    // Calculate CVaR for t-distribution
    let alpha = 1.0 - config.confidence_level;
    let pdf_t = t_dist.pdf(t_score);
    let cvar_multiplier = pdf_t / alpha * (df + t_score.powi(2)) / (df - 1.0);
    let cvar = adjusted_vol * cvar_multiplier - diagnostics.mean_return;
    
    // Scale to horizon
    let horizon_adjustment = if config.volatility_scaling {
        (config.horizon as f64).sqrt()
    } else {
        config.horizon as f64
    };
    
    Ok((
        daily_var * horizon_adjustment,
        Some(cvar * horizon_adjustment)
    ))
}

/// Monte Carlo VaR calculation
fn calculate_monte_carlo_var(
    returns: &[f64],
    config: &VarConfig,
    diagnostics: &VarDiagnostics,
    simulations: usize,
) -> Result<(f64, Option<f64>), VarError> {
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Normal};
    
    let mut rng = thread_rng();
    let normal = Normal::new(diagnostics.mean_return, diagnostics.volatility)
        .map_err(|e| VarError::DistributionError {
            details: e.to_string()
        })?;
    
    // Generate simulated portfolio returns
    let mut simulated_returns = Vec::with_capacity(simulations);
    
    for _ in 0..simulations {
        let mut path_return = 0.0;
        
        // Simulate path over horizon
        for _ in 0..config.horizon {
            path_return += normal.sample(&mut rng);
        }
        
        simulated_returns.push(path_return);
    }
    
    // Sort and find VaR
    simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let position = ((1.0 - config.confidence_level) * simulations as f64) as usize;
    let var = -simulated_returns[position];
    
    // Calculate CVaR
    let worst_returns = &simulated_returns[..=position];
    let cvar = -worst_returns.iter().sum::<f64>() / worst_returns.len() as f64;
    
    Ok((var, Some(cvar)))
}

/// GARCH-based VaR calculation
fn calculate_garch_var(
    returns: &[f64],
    config: &VarConfig,
) -> Result<(f64, Option<f64>), VarError> {
    // Simplified GARCH(1,1) implementation
    // Full implementation would use the GARCH module
    
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    
    // Initial variance (sample variance)
    let mut variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (n - 1) as f64;
    
    // GARCH(1,1) parameters (typical values)
    let omega = 0.000001;
    let alpha = 0.09;  // Weight on squared return
    let beta = 0.89;   // Weight on previous variance
    
    // Update variance using GARCH
    for i in 1..n {
        let squared_return = (returns[i] - mean).powi(2);
        variance = omega + alpha * squared_return + beta * variance;
    }
    
    let volatility = variance.sqrt();
    
    // Use normal distribution for VaR
    let normal = Normal::new(0.0, 1.0).map_err(|e| VarError::DistributionError {
        details: e.to_string()
    })?;
    
    let z_score = normal.inverse_cdf(1.0 - config.confidence_level);
    
    // Calculate VaR with GARCH volatility
    let daily_var = -mean + volatility * (-z_score);
    
    // Multi-period forecast
    let horizon_variance = (0..config.horizon).fold(variance, |acc, _| {
        omega + (alpha + beta) * acc
    });
    
    let horizon_var = -mean * config.horizon as f64 
        + horizon_variance.sqrt() * (-z_score);
    
    Ok((horizon_var, None))
}

/// Cornish-Fisher VaR (adjusts for skewness and kurtosis)
fn calculate_cornish_fisher_var(
    returns: &[f64],
    config: &VarConfig,
    diagnostics: &VarDiagnostics,
) -> Result<(f64, Option<f64>), VarError> {
    let normal = Normal::new(0.0, 1.0).map_err(|e| VarError::DistributionError {
        details: e.to_string()
    })?;
    
    let z = normal.inverse_cdf(1.0 - config.confidence_level);
    
    // Cornish-Fisher expansion
    let z_cf = z 
        + (z.powi(2) - 1.0) * diagnostics.skewness / 6.0
        + (z.powi(3) - 3.0 * z) * (diagnostics.kurtosis - 3.0) / 24.0
        - (2.0 * z.powi(3) - 5.0 * z) * diagnostics.skewness.powi(2) / 36.0;
    
    // Calculate VaR
    let daily_var = -diagnostics.mean_return + diagnostics.volatility * (-z_cf);
    
    // Scale to horizon
    let horizon_adjustment = if config.volatility_scaling {
        (config.horizon as f64).sqrt()
    } else {
        config.horizon as f64
    };
    
    Ok((daily_var * horizon_adjustment, None))
}

/// Filtered Historical Simulation VaR
fn calculate_filtered_historical_var(
    returns: &[f64],
    config: &VarConfig,
) -> Result<(f64, Option<f64>), VarError> {
    // Use GARCH to filter returns
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    
    // Estimate conditional volatilities
    let mut volatilities = vec![0.0; n];
    volatilities[0] = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (n - 1) as f64;
    volatilities[0] = volatilities[0].sqrt();
    
    for i in 1..n {
        let omega = 0.000001;
        let alpha = 0.09;
        let beta = 0.89;
        
        volatilities[i] = (omega 
            + alpha * (returns[i-1] - mean).powi(2)
            + beta * volatilities[i-1].powi(2)).sqrt();
    }
    
    // Standardize returns
    let standardized: Vec<f64> = returns.iter()
        .zip(volatilities.iter())
        .map(|(r, v)| (r - mean) / v)
        .collect();
    
    // Apply historical simulation to standardized returns
    let (std_var, std_cvar) = calculate_historical_var(&standardized, config)?;
    
    // Rescale using current volatility
    let current_vol = volatilities.last().unwrap();
    
    Ok((
        std_var * current_vol + mean * config.horizon as f64,
        std_cvar.map(|cv| cv * current_vol + mean * config.horizon as f64)
    ))
}

/// Calculate Conditional VaR (Expected Shortfall)
pub fn calculate_cvar(
    returns: &[f64],
    confidence_level: f64,
) -> Result<f64, VarError> {
    if returns.is_empty() {
        return Err(VarError::InsufficientData {
            min: 1,
            actual: 0,
        });
    }
    
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let alpha = 1.0 - confidence_level;
    let position = (alpha * returns.len() as f64) as usize;
    let position = position.max(0).min(returns.len() - 1);
    
    // CVaR is the average of returns worse than VaR
    let worst_returns = &sorted_returns[..=position];
    Ok(-worst_returns.iter().sum::<f64>() / worst_returns.len() as f64)
}

/// Calculate sample variance (using Welford's algorithm for numerical stability)
pub use mathematical_ops::risk_metrics::calculate_var; // fn calculate_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut count = 0.0;
    
    for &value in data {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }
    
    m2 / (count - 1.0)
}

/// Calculate diagnostic statistics
fn calculate_diagnostics(returns: &[f64]) -> VarDiagnostics {
    let n = returns.len() as f64;
    
    // Mean
    let mean = returns.iter().sum::<f64>() / n;
    
    // Central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    
    for &r in returns {
        let diff = r - mean;
        m2 += diff.powi(2);
        m3 += diff.powi(3);
        m4 += diff.powi(4);
    }
    
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    let volatility = m2.sqrt();
    let skewness = if volatility > 0.0 {
        m3 / volatility.powi(3)
    } else {
        0.0
    };
    
    let kurtosis = if volatility > 0.0 {
        m4 / volatility.powi(4) - 3.0  // Excess kurtosis
    } else {
        0.0
    };
    
    VarDiagnostics {
        mean_return: mean,
        volatility,
        skewness,
        kurtosis,
        observations: returns.len(),
        backtest_breaches: None,
    }
}

/// Backtest VaR model
pub fn backtest_var(
    returns: &[f64],
    var_model: &VarConfig,
    window_size: usize,
) -> Result<BacktestResult, VarError> {
    if returns.len() < window_size + 1 {
        return Err(VarError::InsufficientData {
            min: window_size + 1,
            actual: returns.len(),
        });
    }
    
    let mut breaches = 0;
    let mut var_predictions = Vec::new();
    
    for i in window_size..returns.len() {
        let historical_window = &returns[i-window_size..i];
        let var_result = calculate_var_with_config(historical_window, var_model)?;
        
        // Check if actual return breached VaR
        if returns[i] < -var_result.var {
            breaches += 1;
        }
        
        var_predictions.push(var_result.var);
    }
    
    let expected_breaches = ((returns.len() - window_size) as f64 
        * (1.0 - var_model.confidence_level)) as usize;
    
    Ok(BacktestResult {
        actual_breaches: breaches,
        expected_breaches,
        breach_rate: breaches as f64 / (returns.len() - window_size) as f64,
        var_predictions,
    })
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Actual number of VaR breaches
    pub actual_breaches: usize,
    /// Expected number of breaches
    pub expected_breaches: usize,
    /// Breach rate (actual/total)
    pub breach_rate: f64,
    /// VaR predictions for each period
    pub var_predictions: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_historical_var() {
        let returns = vec![
            -0.03, 0.02, -0.01, 0.01, -0.02, 0.03, -0.015, 0.025,
            -0.005, 0.015, -0.025, 0.02, -0.01, 0.005, -0.02
        ];
        
        let result = calculate_var(&returns, 0.95, 1, VarMethod::Historical).unwrap();
        assert!(result.var > 0.0);
        assert!(result.var < 0.05);
    }
    
    #[test]
    fn test_parametric_normal_var() {
        let mut returns = Vec::new();
        use rand::{thread_rng, Rng};
        use rand_distr::{Normal, Distribution};
        
        let normal = Normal::new(0.0, 0.01).unwrap();
        let mut rng = thread_rng();
        
        for _ in 0..1000 {
            returns.push(normal.sample(&mut rng));
        }
        
        let result = calculate_var(&returns, 0.99, 1, 
            VarMethod::ParametricNormal).unwrap();
        
        // For normal(0, 0.01), 99% VaR should be around 2.33 * 0.01 = 0.0233
        assert_relative_eq!(result.var, 0.0233, epsilon = 0.005);
    }
    
    #[test]
    fn test_variance_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = calculate_variance(&data);
        assert_relative_eq!(variance, 2.5, epsilon = 1e-10);
    }
}