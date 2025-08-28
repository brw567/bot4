//! # Technical Indicators - Consolidated Implementation
use crate::ml::unified_indicators::{UnifiedIndicators, MACDValue, BollingerBands};
//! 
//! Replaces 11+ duplicate indicator implementations with optimized versions.
//! Includes EMA (4→1), RSI (4→1), SMA (3→1), and additional indicators.
//!
//! ## External Research Applied
//! - "Technical Analysis of the Financial Markets" (Murphy)
//! - "New Concepts in Technical Trading Systems" (Wilder)
//! - TradingView Pine Script indicators analysis

use std::collections::VecDeque;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndicatorError {
    #[error("Insufficient data: need {min}, got {actual}")]
    InsufficientData { min: usize, actual: usize },
    #[error("Invalid period: {period}")]
    InvalidPeriod { period: usize },
}

#[derive(Debug, Clone)]
pub struct IndicatorConfig {
    pub period: usize,
    pub smoothing: f64,
}

/// Calculate Simple Moving Average (consolidates 3 implementations)
pub fn calculate_sma(data: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if data.len() < period {
        return Err(IndicatorError::InsufficientData {
            min: period,
            actual: data.len(),
        });
    }
    
    let mut sma = Vec::with_capacity(data.len() - period + 1);
    
    for window in data.windows(period) {
        let sum: f64 = window.iter().sum();
        sma.push(sum / period as f64);
    }
    
    Ok(sma)
}

/// Calculate Exponential Moving Average (consolidates 4 implementations)
pub fn calculate_ema(data: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if data.is_empty() {
        return Err(IndicatorError::InsufficientData {
            min: 1,
            actual: 0,
        });
    }
    
    if period == 0 {
        return Err(IndicatorError::InvalidPeriod { period });
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema = Vec::with_capacity(data.len());
    
    // Initialize with first value or SMA
    if data.len() >= period {
        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        ema.push(initial_sma);
        
        for &value in &data[period..] {
            let new_ema = alpha * value + (1.0 - alpha) * ema.last().unwrap();
            ema.push(new_ema);
        }
    } else {
        ema.push(data[0]);
        for &value in &data[1..] {
            let new_ema = alpha * value + (1.0 - alpha) * ema.last().unwrap();
            ema.push(new_ema);
        }
    }
    
    Ok(ema)
}

/// Calculate Relative Strength Index (consolidates 4 implementations)
pub fn calculate_rsi(data: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if data.len() < period + 1 {
        return Err(IndicatorError::InsufficientData {
            min: period + 1,
            actual: data.len(),
        });
    }
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    // Calculate price changes
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // Calculate initial averages
    let initial_avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
    let initial_avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
    
    let mut avg_gain = initial_avg_gain;
    let mut avg_loss = initial_avg_loss;
    let mut rsi_values = Vec::new();
    
    // Calculate RSI values
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        
        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi_values.push(rsi);
    }
    
    Ok(rsi_values)
}

/// Calculate MACD (Moving Average Convergence Divergence)
pub fn calculate_macd(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), IndicatorError> {
    if data.len() < slow_period {
        return Err(IndicatorError::InsufficientData {
            min: slow_period,
            actual: data.len(),
        });
    }
    
    let fast_ema = calculate_ema(data, fast_period)?;
    let slow_ema = calculate_ema(data, slow_period)?;
    
    // MACD line = fast EMA - slow EMA
    let macd_line: Vec<f64> = fast_ema.iter()
        .skip(slow_period - fast_period)
        .zip(slow_ema.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();
    
    // Signal line = EMA of MACD line
    let signal_line = calculate_ema(&macd_line, signal_period)?;
    
    // MACD histogram = MACD line - Signal line
    let histogram: Vec<f64> = macd_line.iter()
        .skip(signal_period - 1)
        .zip(signal_line.iter())
        .map(|(macd, signal)| macd - signal)
        .collect();
    
    Ok((macd_line, signal_line, histogram))
}

/// Calculate Bollinger Bands
pub fn calculate_bollinger_bands(
    data: &[f64],
    period: usize,
    std_dev: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), IndicatorError> {
    let sma = calculate_sma(data, period)?;
    let mut upper_band = Vec::new();
    let mut lower_band = Vec::new();
    
    for (i, &middle) in sma.iter().enumerate() {
        let window = &data[i..i + period];
        let variance = window.iter()
            .map(|x| (x - middle).powi(2))
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        
        upper_band.push(middle + std_dev * std);
        lower_band.push(middle - std_dev * std);
    }
    
    Ok((sma, upper_band, lower_band))
}

/// Calculate Average True Range (ATR)
pub fn calculate_atr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, IndicatorError> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(IndicatorError::InvalidPeriod { period: 0 });
    }
    
    if high.len() < period + 1 {
        return Err(IndicatorError::InsufficientData {
            min: period + 1,
            actual: high.len(),
        });
    }
    
    let mut true_ranges = Vec::new();
    
    for i in 1..high.len() {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        true_ranges.push(hl.max(hc).max(lc));
    }
    
    calculate_ema(&true_ranges, period)
}

/// Calculate Stochastic Oscillator
pub fn calculate_stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    smooth_k: usize,
    smooth_d: usize,
) -> Result<(Vec<f64>, Vec<f64>), IndicatorError> {
    if high.len() < period {
        return Err(IndicatorError::InsufficientData {
            min: period,
            actual: high.len(),
        });
    }
    
    let mut k_values = Vec::new();
    
    for i in period - 1..high.len() {
        let highest = high[i - period + 1..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = low[i - period + 1..=i]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        
        let k = if highest != lowest {
            100.0 * (close[i] - lowest) / (highest - lowest)
        } else {
            50.0
        };
        
        k_values.push(k);
    }
    
    // Smooth %K to get slow %K
    let slow_k = calculate_sma(&k_values, smooth_k)?;
    
    // %D is SMA of slow %K
    let d_values = calculate_sma(&slow_k, smooth_d)?;
    
    Ok((slow_k, d_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = calculate_sma(&data, 3).unwrap();
        assert_eq!(sma.len(), 3);
        assert_relative_eq!(sma[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sma[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(sma[2], 4.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = calculate_ema(&data, 3).unwrap();
        assert!(!ema.is_empty());
        assert!(ema.last().unwrap() > &3.0);
    }
    
    #[test]
    fn test_rsi() {
        let data = vec![
            44.0, 44.25, 44.5, 43.75, 44.65, 45.12, 45.84, 46.08, 45.89,
            46.03, 45.61, 46.28, 46.28, 46.0, 46.03, 46.41, 46.22, 45.64
        ];
        let rsi = calculate_rsi(&data, 14).unwrap();
        assert!(!rsi.is_empty());
        assert!(rsi[0] >= 0.0 && rsi[0] <= 100.0);
    }
}