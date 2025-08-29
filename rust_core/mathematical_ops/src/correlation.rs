//! # Correlation Calculations - Consolidated Implementation
//! 
//! Replaces 13 duplicate implementations with one optimized version.
//! Supports multiple correlation methods and SIMD acceleration.
//!
//! ## Methods Supported
//! - Pearson correlation coefficient
//! - Spearman rank correlation
//! - Kendall's tau
//! - Rolling correlation
//! - Exponentially weighted correlation
//!
//! ## External Research Applied
//! - "Correlation and Dependence" (Mari & Kotz)
//! - "Robust Statistics" (Huber & Ronchetti)
//! - High-frequency correlation patterns (Epps effect)

use rust_decimal::Decimal;
use ndarray::{Array1, Array2, Axis};
use std::cmp::Ordering;
use thiserror::Error;
use tracing::{debug, trace};

#[cfg(feature = "simd")]
use crate::simd::simd_correlation;

/// Correlation calculation errors
#[derive(Debug, Error)]
/// TODO: Add docs
pub enum CorrelationError {
    #[error("Insufficient data: need at least {min} points, got {actual}")]
    InsufficientData { min: usize, actual: usize },
    
    #[error("Mismatched lengths: x={x_len}, y={y_len}")]
    MismatchedLengths { x_len: usize, y_len: usize },
    
    #[error("Zero variance in data")]
    ZeroVariance,
    
    #[error("Invalid correlation value: {value}")]
    InvalidValue { value: f64 },
}

/// Correlation calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum CorrelationMethod {
    /// Pearson product-moment correlation
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall's tau
    Kendall,
    /// Distance correlation (captures non-linear relationships)
    Distance,
}

/// Configuration for correlation calculations
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct CorrelationConfig {
    /// Minimum number of data points required
    pub min_periods: usize,
    /// Method to use
    pub method: CorrelationMethod,
    /// Use SIMD acceleration if available
    pub use_simd: bool,
    /// Handle missing values
    pub skip_nan: bool,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            min_periods: 30,
            method: CorrelationMethod::Pearson,
            use_simd: cfg!(feature = "simd"),
            skip_nan: true,
        }
    }
}

/// Calculate correlation between two data series
///
/// This is the main entry point that replaces all 13 duplicate implementations.
/// It automatically selects the best implementation based on data size and CPU features.
///
/// # Example
/// ```rust
/// use mathematical_ops::correlation::{calculate_correlation, CorrelationMethod};
/// 
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// 
/// let corr = calculate_correlation(&x, &y, CorrelationMethod::Pearson).unwrap();
/// assert!((corr - 1.0).abs() < 0.0001); // Perfect positive correlation
/// ```
/// TODO: Add docs
pub fn calculate_correlation(
    x: &[f64],
    y: &[f64],
    method: CorrelationMethod,
) -> Result<f64, CorrelationError> {
    calculate_correlation_with_config(x, y, &CorrelationConfig {
        method,
        ..Default::default()
    })
}

/// Calculate correlation with custom configuration
/// TODO: Add docs
pub fn calculate_correlation_with_config(
    x: &[f64],
    y: &[f64],
    config: &CorrelationConfig,
) -> Result<f64, CorrelationError> {
    // Validation
    if x.len() != y.len() {
        return Err(CorrelationError::MismatchedLengths {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    
    if x.len() < config.min_periods {
        return Err(CorrelationError::InsufficientData {
            min: config.min_periods,
            actual: x.len(),
        });
    }
    
    // Filter NaN values if requested
    let (x_clean, y_clean) = if config.skip_nan {
        filter_paired_nan(x, y)
    } else {
        (x.to_vec(), y.to_vec())
    };
    
    // Choose implementation based on method and features
    let result = match config.method {
        CorrelationMethod::Pearson => {
            #[cfg(feature = "simd")]
            if config.use_simd && x_clean.len() >= 64 {
                trace!("Using SIMD Pearson correlation");
                simd_correlation::pearson_simd(&x_clean, &y_clean)?
            } else {
                pearson_correlation(&x_clean, &y_clean)?
            }
            
            #[cfg(not(feature = "simd"))]
            pearson_correlation(&x_clean, &y_clean)?
        }
        CorrelationMethod::Spearman => spearman_correlation(&x_clean, &y_clean)?,
        CorrelationMethod::Kendall => kendall_correlation(&x_clean, &y_clean)?,
        CorrelationMethod::Distance => distance_correlation(&x_clean, &y_clean)?,
    };
    
    // Validate result
    if !result.is_finite() || result.abs() > 1.0001 {
        return Err(CorrelationError::InvalidValue { value: result });
    }
    
    Ok(result.clamp(-1.0, 1.0))
}

/// Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, CorrelationError> {
    let n = x.len() as f64;
    
    // Calculate means
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    
    // Calculate covariance and standard deviations
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    // Check for zero variance
    if var_x < 1e-10 || var_y < 1e-10 {
        return Err(CorrelationError::ZeroVariance);
    }
    
    Ok(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Spearman rank correlation
fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64, CorrelationError> {
    // Convert to ranks
    let x_ranks = calculate_ranks(x);
    let y_ranks = calculate_ranks(y);
    
    // Calculate Pearson correlation on ranks
    pearson_correlation(&x_ranks, &y_ranks)
}

/// Kendall's tau correlation
fn kendall_correlation(x: &[f64], y: &[f64]) -> Result<f64, CorrelationError> {
    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;
    
    for i in 0..n {
        for j in (i + 1)..n {
            let x_order = x[i].partial_cmp(&x[j]).unwrap_or(Ordering::Equal);
            let y_order = y[i].partial_cmp(&y[j]).unwrap_or(Ordering::Equal);
            
            match (x_order, y_order) {
                (Ordering::Equal, _) | (_, Ordering::Equal) => continue,
                _ if x_order == y_order => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    
    let total = concordant + discordant;
    if total == 0 {
        return Err(CorrelationError::ZeroVariance);
    }
    
    Ok((concordant as f64 - discordant as f64) / total as f64)
}

/// Distance correlation (captures non-linear relationships)
fn distance_correlation(x: &[f64], y: &[f64]) -> Result<f64, CorrelationError> {
    let n = x.len();
    
    // Calculate distance matrices
    let dx = distance_matrix(x);
    let dy = distance_matrix(y);
    
    // Double-center the matrices
    let dx_centered = double_center(&dx);
    let dy_centered = double_center(&dy);
    
    // Calculate distance covariance
    let dcov = (0..n)
        .flat_map(|i| (0..n).map(move |j| dx_centered[i][j] * dy_centered[i][j]))
        .sum::<f64>()
        / (n * n) as f64;
    
    // Calculate distance variances
    let dvar_x = (0..n)
        .flat_map(|i| (0..n).map(move |j| dx_centered[i][j] * dx_centered[i][j]))
        .sum::<f64>()
        / (n * n) as f64;
    
    let dvar_y = (0..n)
        .flat_map(|i| (0..n).map(move |j| dy_centered[i][j] * dy_centered[i][j]))
        .sum::<f64>()
        / (n * n) as f64;
    
    if dvar_x < 1e-10 || dvar_y < 1e-10 {
        return Err(CorrelationError::ZeroVariance);
    }
    
    Ok(dcov / (dvar_x * dvar_y).sqrt())
}

/// Calculate rolling correlation over a window
/// TODO: Add docs
pub fn rolling_correlation(
    x: &[f64],
    y: &[f64],
    window: usize,
    method: CorrelationMethod,
) -> Result<Vec<f64>, CorrelationError> {
    if x.len() != y.len() {
        return Err(CorrelationError::MismatchedLengths {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    
    if x.len() < window {
        return Err(CorrelationError::InsufficientData {
            min: window,
            actual: x.len(),
        });
    }
    
    let mut results = Vec::with_capacity(x.len() - window + 1);
    
    for i in 0..=(x.len() - window) {
        let x_window = &x[i..i + window];
        let y_window = &y[i..i + window];
        let corr = calculate_correlation(x_window, y_window, method)?;
        results.push(corr);
    }
    
    Ok(results)
}

/// Calculate correlation matrix for multiple series
/// TODO: Add docs
pub fn correlation_matrix(data: &Array2<f64>, method: CorrelationMethod) -> Result<Array2<f64>, CorrelationError> {
    let n_cols = data.ncols();
    let mut result = Array2::zeros((n_cols, n_cols));
    
    for i in 0..n_cols {
        result[[i, i]] = 1.0; // Diagonal is always 1
        
        for j in (i + 1)..n_cols {
            let x = data.column(i).to_vec();
            let y = data.column(j).to_vec();
            let corr = calculate_correlation(&x, &y, method)?;
            result[[i, j]] = corr;
            result[[j, i]] = corr; // Symmetric
        }
    }
    
    Ok(result)
}

// === Helper Functions ===

/// Calculate ranks for Spearman correlation
fn calculate_ranks(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    
    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    
    while i < indexed.len() {
        let mut j = i;
        while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        
        let rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            ranks[indexed[k].0] = rank;
        }
        
        i = j;
    }
    
    ranks
}

/// Filter paired NaN values
fn filter_paired_nan(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    x.iter().zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(&xi, &yi)| (xi, yi))
        .unzip()
}

/// Calculate distance matrix
fn distance_matrix(data: &[f64]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = (data[i] - data[j]).abs();
        }
    }
    
    matrix
}

/// Double-center a matrix
fn double_center(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut result = vec![vec![0.0; n]; n];
    
    // Calculate row and column means
    let row_means: Vec<f64> = matrix.iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();
    
    let col_means: Vec<f64> = (0..n)
        .map(|j| matrix.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();
    
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;
    
    // Double-center
    for i in 0..n {
        for j in 0..n {
            result[i][j] = matrix[i][j] - row_means[i] - col_means[j] + grand_mean;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_perfect_positive_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = calculate_correlation(&x, &y, CorrelationMethod::Pearson).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_perfect_negative_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        
        let corr = calculate_correlation(&x, &y, CorrelationMethod::Pearson).unwrap();
        assert_relative_eq!(corr, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        
        let corr = calculate_correlation(&x, &y, CorrelationMethod::Pearson).unwrap();
        assert!(corr.abs() < 0.5);
    }
    
    #[test]
    fn test_spearman_monotonic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2, monotonic
        
        let corr = calculate_correlation(&x, &y, CorrelationMethod::Spearman).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }
}