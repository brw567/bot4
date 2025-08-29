//! # Statistical Functions - Core implementations

use std::collections::HashMap;

/// Calculate mean of a dataset
/// TODO: Add docs
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate median of a dataset
/// TODO: Add docs
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let len = sorted.len();
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Calculate mode of a dataset
/// TODO: Add docs
pub fn mode(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    
    let mut frequency = HashMap::new();
    for &value in data {
        *frequency.entry(value.to_bits()).or_insert(0) += 1;
    }
    
    frequency.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(bits, _)| f64::from_bits(bits))
}

/// Calculate standard deviation
/// TODO: Add docs
pub fn standard_deviation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let variance = data.iter()
        .map(|x| (x - mean_val).powi(2))
        .sum::<f64>() / (data.len() - 1) as f64;
    
    variance.sqrt()
}

/// Calculate skewness (third moment)
/// TODO: Add docs
pub fn skewness(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    
    let n = data.len() as f64;
    let mean_val = mean(data);
    let std_dev = standard_deviation(data);
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    let sum_cubed = data.iter()
        .map(|x| ((x - mean_val) / std_dev).powi(3))
        .sum::<f64>();
    
    (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
}

/// Calculate excess kurtosis (fourth moment)
/// TODO: Add docs
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    
    let n = data.len() as f64;
    let mean_val = mean(data);
    let std_dev = standard_deviation(data);
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    let sum_fourth = data.iter()
        .map(|x| ((x - mean_val) / std_dev).powi(4))
        .sum::<f64>();
    
    let kurt = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sum_fourth
        - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));
    
    kurt
}

/// Calculate percentile
/// TODO: Add docs
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() || p < 0.0 || p > 100.0 {
        return 0.0;
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Calculate quantile (0-1 scale)
/// TODO: Add docs
pub fn quantile(data: &[f64], q: f64) -> f64 {
    percentile(data, q * 100.0)
}