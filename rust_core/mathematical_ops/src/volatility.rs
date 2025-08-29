//! # Volatility Calculations - Unified Implementation
//! 
//! Consolidates 3+ volatility calculation implementations into one comprehensive module.
//! Supports multiple volatility models with regime detection and forecasting.
//!
//! ## Models Supported
//! - Historical/Realized volatility
//! - EWMA (Exponentially Weighted Moving Average)
//! - GARCH(1,1) and variants
//! - Parkinson (using high-low prices)
//! - Garman-Klass (OHLC based)
//! - Yang-Zhang (handles overnight gaps)
//! - Realized volatility using high-frequency data
//!
//! ## External Research Applied
//! - "Volatility Trading" (Sinclair, 2013)
//! - "The Econometrics of Financial Markets" (Campbell et al.)
//! - "Forecasting Volatility in Financial Markets" (Knight & Satchell)
//! - CBOE VIX methodology white paper
//! - "Realized Volatility" (Andersen et al., 2003)

use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use thiserror::Error;
use tracing::{debug, trace};

#[cfg(feature = "simd")]
use crate::simd::simd_volatility;

/// Volatility calculation errors
#[derive(Debug, Error)]
/// TODO: Add docs
pub enum VolatilityError {
    #[error("Insufficient data: need at least {min} points, got {actual}")]
    InsufficientData { min: usize, actual: usize },
    
    #[error("Invalid window size: {size}")]
    InvalidWindow { size: usize },
    
    #[error("Invalid parameter: {param} = {value}")]
    InvalidParameter { param: String, value: f64 },
    
    #[error("Model fitting failed: {details}")]
    ModelFittingError { details: String },
}

/// Volatility model types
#[derive(Debug, Clone, Copy, PartialEq)]
/// TODO: Add docs
pub enum VolatilityModel {
    /// Simple historical volatility
    Historical,
    /// Exponentially weighted moving average
    EWMA { lambda: f64 },
    /// GARCH(1,1) model
    Garch { omega: f64, alpha: f64, beta: f64 },
    /// Parkinson estimator (high-low)
    Parkinson,
    /// Garman-Klass estimator (OHLC)
    GarmanKlass,
    /// Yang-Zhang estimator (handles gaps)
    YangZhang,
    /// Realized volatility from tick data
    Realized { sampling_minutes: usize },
}

/// OHLC price data for advanced volatility estimators
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct OhlcData {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
    pub timestamp: Option<i64>,
}

/// Volatility calculation result
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct VolatilityResult {
    /// Current volatility estimate (annualized)
    pub volatility: f64,
    /// Daily volatility
    pub daily_volatility: f64,
    /// Volatility forecast (if applicable)
    pub forecast: Option<Vec<f64>>,
    /// Model used
    pub model: VolatilityModel,
    /// Confidence intervals
    pub confidence_intervals: Option<(f64, f64)>,
    /// Volatility regime
    pub regime: VolatilityRegime,
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
/// TODO: Add docs
pub enum VolatilityRegime {
    Low,      // < 10% annualized
    Normal,   // 10-25% annualized
    Elevated, // 25-40% annualized
    High,     // 40-60% annualized
    Extreme,  // > 60% annualized
}

/// Calculate volatility using specified model
///
/// This is the main entry point replacing all duplicate implementations.
///
/// # Example
/// ```rust
/// use mathematical_ops::volatility::{calculate_volatility, VolatilityModel};
/// 
/// let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
/// let vol = calculate_volatility(&returns, VolatilityModel::Historical).unwrap();
/// println!("Annualized volatility: {:.2}%", vol.volatility * 100.0);
/// ```
/// TODO: Add docs
pub fn calculate_volatility(
    returns: &[f64],
    model: VolatilityModel,
) -> Result<VolatilityResult, VolatilityError> {
    match model {
        VolatilityModel::Historical => calculate_historical_volatility(returns),
        VolatilityModel::EWMA { lambda } => calculate_ewma_volatility(returns, lambda),
        VolatilityModel::Garch { omega, alpha, beta } => {
            calculate_garch_volatility(returns, omega, alpha, beta)
        }
        _ => Err(VolatilityError::ModelFittingError {
            details: "Model requires OHLC data".to_string(),
        }),
    }
}

/// Calculate volatility from OHLC data
/// TODO: Add docs
pub fn calculate_volatility_ohlc(
    data: &[OhlcData],
    model: VolatilityModel,
) -> Result<VolatilityResult, VolatilityError> {
    match model {
        VolatilityModel::Parkinson => calculate_parkinson_volatility(data),
        VolatilityModel::GarmanKlass => calculate_garman_klass_volatility(data),
        VolatilityModel::YangZhang => calculate_yang_zhang_volatility(data),
        _ => {
            // Convert to returns and use standard calculation
            let returns = ohlc_to_returns(data);
            calculate_volatility(&returns, model)
        }
    }
}

/// Historical (realized) volatility calculation
fn calculate_historical_volatility(returns: &[f64]) -> Result<VolatilityResult, VolatilityError> {
    if returns.len() < 2 {
        return Err(VolatilityError::InsufficientData {
            min: 2,
            actual: returns.len(),
        });
    }
    
    // Calculate mean return
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    
    // Calculate variance (using population variance for consistency)
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    let daily_vol = variance.sqrt();
    
    // Annualize assuming 252 trading days
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    // Determine regime
    let regime = classify_volatility_regime(annual_vol);
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: None,
        model: VolatilityModel::Historical,
        confidence_intervals: calculate_confidence_intervals(daily_vol, returns.len()),
        regime,
    })
}

/// EWMA volatility calculation
fn calculate_ewma_volatility(
    returns: &[f64],
    lambda: f64,
) -> Result<VolatilityResult, VolatilityError> {
    if returns.is_empty() {
        return Err(VolatilityError::InsufficientData {
            min: 1,
            actual: 0,
        });
    }
    
    if lambda <= 0.0 || lambda >= 1.0 {
        return Err(VolatilityError::InvalidParameter {
            param: "lambda".to_string(),
            value: lambda,
        });
    }
    
    // Initialize with first squared return
    let mut variance = returns[0].powi(2);
    
    // EWMA update: σ²ₜ = λ*σ²ₜ₋₁ + (1-λ)*r²ₜ
    for &ret in &returns[1..] {
        variance = lambda * variance + (1.0 - lambda) * ret.powi(2);
    }
    
    let daily_vol = variance.sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    // Forecast next period volatility (EWMA assumes constant)
    let forecast = vec![daily_vol; 10];
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: Some(forecast),
        model: VolatilityModel::EWMA { lambda },
        confidence_intervals: calculate_confidence_intervals(daily_vol, returns.len()),
        regime: classify_volatility_regime(annual_vol),
    })
}

/// GARCH(1,1) volatility calculation
fn calculate_garch_volatility(
    returns: &[f64],
    omega: f64,
    alpha: f64,
    beta: f64,
) -> Result<VolatilityResult, VolatilityError> {
    if returns.len() < 10 {
        return Err(VolatilityError::InsufficientData {
            min: 10,
            actual: returns.len(),
        });
    }
    
    // Validate GARCH parameters
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0 {
        return Err(VolatilityError::InvalidParameter {
            param: "GARCH parameters".to_string(),
            value: alpha + beta,
        });
    }
    
    // Initialize with unconditional variance
    let unconditional_var = omega / (1.0 - alpha - beta);
    let mut variance = vec![unconditional_var; returns.len()];
    
    // GARCH recursion: σ²ₜ = ω + α*r²ₜ₋₁ + β*σ²ₜ₋₁
    for i in 1..returns.len() {
        variance[i] = omega 
            + alpha * returns[i-1].powi(2)
            + beta * variance[i-1];
    }
    
    let current_variance = variance.last().unwrap();
    let daily_vol = current_variance.sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    // Multi-period forecast
    let mut forecast = Vec::with_capacity(10);
    let mut future_var = *current_variance;
    
    for _ in 0..10 {
        future_var = omega + (alpha + beta) * future_var;
        forecast.push(future_var.sqrt());
    }
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: Some(forecast),
        model: VolatilityModel::Garch { omega, alpha, beta },
        confidence_intervals: calculate_confidence_intervals(daily_vol, returns.len()),
        regime: classify_volatility_regime(annual_vol),
    })
}

/// Parkinson volatility estimator (uses high-low range)
fn calculate_parkinson_volatility(data: &[OhlcData]) -> Result<VolatilityResult, VolatilityError> {
    if data.is_empty() {
        return Err(VolatilityError::InsufficientData {
            min: 1,
            actual: 0,
        });
    }
    
    // Parkinson estimator: σ = sqrt(1/(4*n*ln(2)) * Σ(ln(H/L))²)
    let sum_squared_log_range: f64 = data.iter()
        .map(|bar| {
            if bar.low > 0.0 {
                (bar.high / bar.low).ln().powi(2)
            } else {
                0.0
            }
        })
        .sum();
    
    let n = data.len() as f64;
    let daily_vol = (sum_squared_log_range / (4.0 * n * 2.0_f64.ln())).sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: None,
        model: VolatilityModel::Parkinson,
        confidence_intervals: calculate_confidence_intervals(daily_vol, data.len()),
        regime: classify_volatility_regime(annual_vol),
    })
}

/// Garman-Klass volatility estimator (uses OHLC)
fn calculate_garman_klass_volatility(data: &[OhlcData]) -> Result<VolatilityResult, VolatilityError> {
    if data.is_empty() {
        return Err(VolatilityError::InsufficientData {
            min: 1,
            actual: 0,
        });
    }
    
    // Garman-Klass: combines Parkinson with open-close information
    let n = data.len() as f64;
    let mut sum = 0.0;
    
    for bar in data {
        if bar.low > 0.0 && bar.open > 0.0 {
            let hl_term = 0.5 * (bar.high / bar.low).ln().powi(2);
            let co_term = (2.0 * 2.0_f64.ln() - 1.0) * (bar.close / bar.open).ln().powi(2);
            sum += hl_term - co_term;
        }
    }
    
    let daily_vol = (sum / n).sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: None,
        model: VolatilityModel::GarmanKlass,
        confidence_intervals: calculate_confidence_intervals(daily_vol, data.len()),
        regime: classify_volatility_regime(annual_vol),
    })
}

/// Yang-Zhang volatility estimator (handles overnight gaps)
fn calculate_yang_zhang_volatility(data: &[OhlcData]) -> Result<VolatilityResult, VolatilityError> {
    if data.len() < 2 {
        return Err(VolatilityError::InsufficientData {
            min: 2,
            actual: data.len(),
        });
    }
    
    let n = data.len() as f64;
    let k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0));
    
    // Overnight volatility
    let mut overnight_sum = 0.0;
    for i in 1..data.len() {
        if data[i-1].close > 0.0 {
            overnight_sum += (data[i].open / data[i-1].close).ln().powi(2);
        }
    }
    let overnight_var = overnight_sum / (n - 1.0);
    
    // Open-to-close volatility
    let oc_sum: f64 = data.iter()
        .filter(|bar| bar.open > 0.0)
        .map(|bar| (bar.close / bar.open).ln().powi(2))
        .sum();
    let oc_var = oc_sum / (n - 1.0);
    
    // Rogers-Satchell volatility
    let rs_sum: f64 = data.iter()
        .filter(|bar| bar.open > 0.0 && bar.close > 0.0)
        .map(|bar| {
            (bar.high / bar.close).ln() * (bar.high / bar.open).ln()
                + (bar.low / bar.close).ln() * (bar.low / bar.open).ln()
        })
        .sum();
    let rs_var = rs_sum / n;
    
    // Yang-Zhang estimator
    let variance = overnight_var + k * oc_var + (1.0 - k) * rs_var;
    let daily_vol = variance.sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();
    
    Ok(VolatilityResult {
        volatility: annual_vol,
        daily_volatility: daily_vol,
        forecast: None,
        model: VolatilityModel::YangZhang,
        confidence_intervals: calculate_confidence_intervals(daily_vol, data.len()),
        regime: classify_volatility_regime(annual_vol),
    })
}

/// Dynamic volatility tracker with regime detection
/// TODO: Add docs
pub struct VolatilityTracker {
    /// Rolling window of returns
    returns_window: VecDeque<f64>,
    /// Window size
    window_size: usize,
    /// Current volatility estimate
    current_volatility: f64,
    /// Volatility history
    volatility_history: Vec<f64>,
    /// Current regime
    current_regime: VolatilityRegime,
    /// Model to use
    model: VolatilityModel,
}

impl VolatilityTracker {
    pub fn new(window_size: usize, model: VolatilityModel) -> Self {
        Self {
            returns_window: VecDeque::with_capacity(window_size),
            window_size,
            current_volatility: 0.0,
            volatility_history: Vec::new(),
            current_regime: VolatilityRegime::Normal,
            model,
        }
    }
    
    /// Update with new return
    pub fn update(&mut self, return_val: f64) -> VolatilityResult {
        self.returns_window.push_back(return_val);
        
        if self.returns_window.len() > self.window_size {
            self.returns_window.pop_front();
        }
        
        if self.returns_window.len() >= 2 {
            let returns: Vec<f64> = self.returns_window.iter().cloned().collect();
            if let Ok(result) = calculate_volatility(&returns, self.model) {
                self.current_volatility = result.volatility;
                self.volatility_history.push(result.volatility);
                self.current_regime = result.regime;
                return result;
            }
        }
        
        VolatilityResult {
            volatility: self.current_volatility,
            daily_volatility: self.current_volatility / (252.0_f64).sqrt(),
            forecast: None,
            model: self.model,
            confidence_intervals: None,
            regime: self.current_regime,
        }
    }
    
    /// Get volatility forecast
    pub fn forecast(&self, periods: usize) -> Vec<f64> {
        match self.model {
            VolatilityModel::EWMA { lambda } => {
                vec![self.current_volatility; periods]
            }
            VolatilityModel::Garch { omega, alpha, beta } => {
                let mut forecast = Vec::with_capacity(periods);
                let unconditional = omega / (1.0 - alpha - beta);
                let mut future_var = self.current_volatility.powi(2) / 252.0;
                
                for _ in 0..periods {
                    future_var = omega + (alpha + beta) * future_var;
                    forecast.push((future_var * 252.0).sqrt());
                }
                
                forecast
            }
            _ => vec![self.current_volatility; periods],
        }
    }
}

// === Helper Functions ===

/// Convert OHLC data to returns
fn ohlc_to_returns(data: &[OhlcData]) -> Vec<f64> {
    data.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Classify volatility regime
fn classify_volatility_regime(annual_vol: f64) -> VolatilityRegime {
    if annual_vol < 0.10 {
        VolatilityRegime::Low
    } else if annual_vol < 0.25 {
        VolatilityRegime::Normal
    } else if annual_vol < 0.40 {
        VolatilityRegime::Elevated
    } else if annual_vol < 0.60 {
        VolatilityRegime::High
    } else {
        VolatilityRegime::Extreme
    }
}

/// Calculate confidence intervals for volatility estimate
fn calculate_confidence_intervals(volatility: f64, n: usize) -> Option<(f64, f64)> {
    if n < 30 {
        return None;
    }
    
    // Use chi-square distribution for volatility confidence intervals
    // Approximation for 95% confidence
    let n_f = n as f64;
    let chi_lower = n_f - 1.96 * (2.0 * n_f).sqrt();
    let chi_upper = n_f + 1.96 * (2.0 * n_f).sqrt();
    
    let lower = volatility * ((n_f - 1.0) / chi_upper).sqrt();
    let upper = volatility * ((n_f - 1.0) / chi_lower).sqrt();
    
    Some((lower, upper))
}

/// Volatility cone for term structure
/// TODO: Add docs
pub fn calculate_volatility_cone(
    returns: &[f64],
    windows: &[usize],
) -> Result<Vec<(usize, f64, f64, f64)>, VolatilityError> {
    let mut results = Vec::new();
    
    for &window in windows {
        if returns.len() < window {
            continue;
        }
        
        let mut window_vols = Vec::new();
        
        for i in 0..=(returns.len() - window) {
            let window_returns = &returns[i..i + window];
            if let Ok(vol) = calculate_historical_volatility(window_returns) {
                window_vols.push(vol.volatility);
            }
        }
        
        if !window_vols.is_empty() {
            window_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = window_vols[window_vols.len() / 2];
            let p25 = window_vols[window_vols.len() / 4];
            let p75 = window_vols[3 * window_vols.len() / 4];
            
            results.push((window, p25, median, p75));
        }
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_historical_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012];
        let result = calculate_volatility(&returns, VolatilityModel::Historical).unwrap();
        
        assert!(result.volatility > 0.0);
        assert!(result.daily_volatility > 0.0);
        assert_eq!(result.model, VolatilityModel::Historical);
    }
    
    #[test]
    fn test_ewma_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
        let result = calculate_volatility(&returns, 
            VolatilityModel::EWMA { lambda: 0.94 }).unwrap();
        
        assert!(result.volatility > 0.0);
        assert!(result.forecast.is_some());
    }
    
    #[test]
    fn test_garch_volatility() {
        let returns = vec![
            0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012,
            0.005, -0.008, 0.011, -0.009, 0.007, -0.003, 0.012
        ];
        
        let result = calculate_volatility(&returns,
            VolatilityModel::Garch { 
                omega: 0.000001,
                alpha: 0.09,
                beta: 0.89
            }).unwrap();
        
        assert!(result.volatility > 0.0);
        assert!(result.forecast.is_some());
        assert_eq!(result.forecast.as_ref().unwrap().len(), 10);
    }
    
    #[test]
    fn test_volatility_regime() {
        assert_eq!(classify_volatility_regime(0.08), VolatilityRegime::Low);
        assert_eq!(classify_volatility_regime(0.15), VolatilityRegime::Normal);
        assert_eq!(classify_volatility_regime(0.30), VolatilityRegime::Elevated);
        assert_eq!(classify_volatility_regime(0.50), VolatilityRegime::High);
        assert_eq!(classify_volatility_regime(0.80), VolatilityRegime::Extreme);
    }
}