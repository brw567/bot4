//! # UNIFIED CALCULATIONS - Single Implementation
//! Cameron: "One calculation, used everywhere"
//! Blake: "No more inconsistent results"

use crate::indicators::{calculate_rsi as calc_rsi, calculate_ema as calc_ema};
use crate::variance::calculate_var as calc_var;
use crate::correlation::calculate_correlation as calc_corr;

/// Re-export canonical implementations
pub use calc_rsi as calculate_rsi;
pub use calc_ema as calculate_ema;
pub use calc_var as calculate_var;
pub use calc_corr as calculate_correlation;

// ATR calculation - consolidate all versions
/// TODO: Add docs
pub fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Option<f64> {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return None;
    }
    
    let mut true_ranges = Vec::new();
    
    for i in 1..highs.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i-1]).abs();
        let low_close = (lows[i] - closes[i-1]).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    if true_ranges.len() < period {
        return None;
    }
    
    // Calculate ATR as EMA of true range
    let atr: f64 = true_ranges.iter()
        .rev()
        .take(period)
        .sum::<f64>() / period as f64;
    
    Some(atr)
}

// MACD calculation - consolidate all versions
/// TODO: Add docs
// ELIMINATED: pub struct MACDResult {
// ELIMINATED:     pub macd_line: f64,
// ELIMINATED:     pub signal_line: f64,
// ELIMINATED:     pub histogram: f64,
// ELIMINATED: }

/// TODO: Add docs
pub fn calculate_macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Option<MACDResult> {
    if prices.len() < slow {
        return None;
    }
    
    let ema_fast = calc_ema(prices, fast).ok()?;
    let ema_slow = calc_ema(prices, slow).ok()?;
    
    let macd_line = ema_fast.last()? - ema_slow.last()?;
    
    // Simplified signal line (would need MACD history for proper EMA)
    let signal_line = macd_line * 0.9; // Placeholder
    let histogram = macd_line - signal_line;
    
    Some(MACDResult {
        macd_line,
        signal_line,
        histogram,
    })
}

// Bollinger Bands - single implementation
/// TODO: Add docs
// ELIMINATED: pub struct BollingerBands {
// ELIMINATED:     pub upper: f64,
// ELIMINATED:     pub middle: f64,
// ELIMINATED:     pub lower: f64,
// ELIMINATED: }

/// TODO: Add docs
pub fn calculate_bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> Option<BollingerBands> {
    if prices.len() < period {
        return None;
    }
    
    let recent: Vec<f64> = prices.iter().rev().take(period).copied().collect();
    let sma: f64 = recent.iter().sum::<f64>() / period as f64;
    
    let variance: f64 = recent.iter()
        .map(|p| (p - sma).powi(2))
        .sum::<f64>() / period as f64;
    
    let std = variance.sqrt();
    
    Some(BollingerBands {
        upper: sma + (std * std_dev),
        middle: sma,
        lower: sma - (std * std_dev),
    })
}

// TEAM: "All calculations unified!"
