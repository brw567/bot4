//! # Matrix Operations - Linear algebra for portfolio optimization

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array1};

/// Calculate covariance matrix
/// TODO: Add docs
pub fn covariance_matrix(returns: &Array2<f64>) -> Array2<f64> {
    let n = returns.nrows();
    let k = returns.ncols();
    
    let mut cov = Array2::zeros((k, k));
    
    // Calculate means for each asset
    let means: Vec<f64> = (0..k).map(|j| {
        returns.column(j).sum() / n as f64
    }).collect();
    
    // Calculate covariance
    for i in 0..k {
        for j in i..k {
            let mut sum = 0.0;
            for row in 0..n {
                sum += (returns[[row, i]] - means[i]) * (returns[[row, j]] - means[j]);
            }
            let cov_val = sum / (n - 1) as f64;
            cov[[i, j]] = cov_val;
            cov[[j, i]] = cov_val; // Symmetric
        }
    }
    
    cov
}

/// Calculate correlation from covariance
/// TODO: Add docs
pub fn correlation_from_covariance(cov_matrix: &Array2<f64>) -> Array2<f64> {
    let n = cov_matrix.nrows();
    let mut corr = Array2::zeros((n, n));
    
    for i in 0..n {
        for j in 0..n {
            let std_i = cov_matrix[[i, i]].sqrt();
            let std_j = cov_matrix[[j, j]].sqrt();
            
            if std_i > 0.0 && std_j > 0.0 {
                corr[[i, j]] = cov_matrix[[i, j]] / (std_i * std_j);
            } else {
                corr[[i, j]] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    
    corr
}