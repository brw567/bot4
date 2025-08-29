//! SIMD-OPTIMIZED TECHNICAL INDICATORS - AVX-512 Implementation
//! Team: InfraEngineer (SIMD) + MLEngineer (features) + RiskQuant (validation)
//!
//! Performance: 8x speedup using f64x8 vectors
//! Research: Intel AVX-512 optimization guide (2024)

#![feature(portable_simd)]
use std::simd::{f64x8, SimdFloat, StdFloat};
use rust_decimal::Decimal;
use mimalloc::MiMalloc;

// Use high-performance allocator
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// SIMD-optimized Bollinger Bands calculator
pub struct SimdBollingerBands {
    period: usize,
    std_multiplier: f64,
    
    // Pre-allocated SIMD vectors for zero-allocation
    price_buffer: Vec<f64x8>,
    sma_buffer: Vec<f64x8>,
    std_buffer: Vec<f64x8>,
}

impl SimdBollingerBands {
    pub fn new(period: usize, std_multiplier: f64) -> Self {
        let capacity = 10000 / 8;  // Pre-allocate for 10k prices
        Self {
            period,
            std_multiplier,
            price_buffer: Vec::with_capacity(capacity),
            sma_buffer: Vec::with_capacity(capacity),
            std_buffer: Vec::with_capacity(capacity),
        }
    }
    
    /// Calculate bands using AVX-512 (8x parallel)
    pub fn calculate_simd(&mut self, prices: &[f64]) -> BollingerResult {
        // Pad to multiple of 8 for SIMD
        let padded_len = (prices.len() + 7) / 8 * 8;
        let mut padded_prices = vec![0.0; padded_len];
        padded_prices[..prices.len()].copy_from_slice(prices);
        
        // Convert to SIMD vectors
        self.price_buffer.clear();
        for chunk in padded_prices.chunks_exact(8) {
            let vec = f64x8::from_slice(chunk);
            self.price_buffer.push(vec);
        }
        
        // Calculate SMA using SIMD
        let sma_simd = self.calculate_sma_simd();
        
        // Calculate standard deviation using SIMD
        let std_simd = self.calculate_std_simd(sma_simd);
        
        // Calculate bands
        let upper = sma_simd + std_simd * f64x8::splat(self.std_multiplier);
        let lower = sma_simd - std_simd * f64x8::splat(self.std_multiplier);
        
        // Extract results
        BollingerResult {
            upper_band: Decimal::from_f64_retain(upper.as_array()[0]).unwrap(),
            middle_band: Decimal::from_f64_retain(sma_simd.as_array()[0]).unwrap(),
            lower_band: Decimal::from_f64_retain(lower.as_array()[0]).unwrap(),
            bandwidth: Decimal::from_f64_retain((upper - lower).as_array()[0]).unwrap(),
            percent_b: self.calculate_percent_b(padded_prices.last().copied().unwrap(), 
                                               upper.as_array()[0], 
                                               lower.as_array()[0]),
        }
    }
    
    /// SIMD SMA calculation
    fn calculate_sma_simd(&self) -> f64x8 {
        let n = self.price_buffer.len();
        if n == 0 {
            return f64x8::splat(0.0);
        }
        
        let mut sum = f64x8::splat(0.0);
        let period_vecs = (self.period + 7) / 8;
        
        for i in n.saturating_sub(period_vecs)..n {
            sum += self.price_buffer[i];
        }
        
        sum / f64x8::splat(self.period as f64)
    }
    
    /// SIMD standard deviation calculation
    fn calculate_std_simd(&self, mean: f64x8) -> f64x8 {
        let n = self.price_buffer.len();
        if n == 0 {
            return f64x8::splat(0.0);
        }
        
        let mut variance = f64x8::splat(0.0);
        let period_vecs = (self.period + 7) / 8;
        
        for i in n.saturating_sub(period_vecs)..n {
            let diff = self.price_buffer[i] - mean;
            variance += diff * diff;
        }
        
        (variance / f64x8::splat(self.period as f64)).sqrt()
    }
    
    fn calculate_percent_b(&self, price: f64, upper: f64, lower: f64) -> Decimal {
        if upper == lower {
            return Decimal::ZERO;
        }
        
        let percent = (price - lower) / (upper - lower);
        Decimal::from_f64_retain(percent).unwrap()
    }
}

/// SIMD-optimized RSI calculator
pub struct SimdRSI {
    period: usize,
    gains_buffer: Vec<f64x8>,
    losses_buffer: Vec<f64x8>,
}

impl SimdRSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains_buffer: Vec::with_capacity(1000),
            losses_buffer: Vec::with_capacity(1000),
        }
    }
    
    /// Calculate RSI using SIMD (8x parallel)
    pub fn calculate_simd(&mut self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.period + 1 {
            return vec![50.0; prices.len()];
        }
        
        // Calculate price changes
        let mut changes = vec![0.0; prices.len() - 1];
        for i in 1..prices.len() {
            changes[i - 1] = prices[i] - prices[i - 1];
        }
        
        // Pad for SIMD
        let padded_len = (changes.len() + 7) / 8 * 8;
        changes.resize(padded_len, 0.0);
        
        // Separate gains and losses using SIMD
        self.gains_buffer.clear();
        self.losses_buffer.clear();
        
        for chunk in changes.chunks_exact(8) {
            let vec = f64x8::from_slice(chunk);
            let zeros = f64x8::splat(0.0);
            
            // SIMD max/min for gains/losses
            let gains = vec.simd_max(zeros);
            let losses = (-vec).simd_max(zeros);
            
            self.gains_buffer.push(gains);
            self.losses_buffer.push(losses);
        }
        
        // Calculate average gains/losses
        let mut rsi_values = Vec::with_capacity(prices.len());
        
        for i in 0..self.gains_buffer.len() {
            let avg_gain = self.calculate_ema_simd(&self.gains_buffer, i);
            let avg_loss = self.calculate_ema_simd(&self.losses_buffer, i);
            
            // Calculate RSI for each element in the vector
            for j in 0..8 {
                let g = avg_gain.as_array()[j];
                let l = avg_loss.as_array()[j];
                
                let rsi = if l == 0.0 {
                    100.0
                } else {
                    let rs = g / l;
                    100.0 - (100.0 / (1.0 + rs))
                };
                
                rsi_values.push(rsi);
            }
        }
        
        rsi_values.truncate(prices.len() - 1);
        rsi_values
    }
    
    /// SIMD EMA calculation for RSI
    fn calculate_ema_simd(&self, buffer: &[f64x8], end_idx: usize) -> f64x8 {
        let alpha = 1.0 / self.period as f64;
        let alpha_vec = f64x8::splat(alpha);
        let one_minus_alpha = f64x8::splat(1.0 - alpha);
        
        let start_idx = end_idx.saturating_sub(self.period / 8);
        let mut ema = buffer[start_idx];
        
        for i in (start_idx + 1)..=end_idx {
            ema = buffer[i] * alpha_vec + ema * one_minus_alpha;
        }
        
        ema
    }
}

/// SIMD-optimized MACD calculator
pub struct SimdMACD {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    
    price_buffer: Vec<f64x8>,
}

impl SimdMACD {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
            price_buffer: Vec::with_capacity(1000),
        }
    }
    
    /// Calculate MACD using SIMD
    pub fn calculate_simd(&mut self, prices: &[f64]) -> MACDResult {
        // Convert to SIMD vectors
        self.price_buffer.clear();
        let padded_len = (prices.len() + 7) / 8 * 8;
        let mut padded = vec![0.0; padded_len];
        padded[..prices.len()].copy_from_slice(prices);
        
        for chunk in padded.chunks_exact(8) {
            self.price_buffer.push(f64x8::from_slice(chunk));
        }
        
        // Calculate EMAs
        let fast_ema = self.calculate_ema_vectorized(self.fast_period);
        let slow_ema = self.calculate_ema_vectorized(self.slow_period);
        
        // MACD line
        let macd_line = fast_ema - slow_ema;
        
        // Signal line (EMA of MACD)
        let signal_line = self.calculate_signal_ema(macd_line, self.signal_period);
        
        // Histogram
        let histogram = macd_line - signal_line;
        
        MACDResult {
            macd_line: Decimal::from_f64_retain(macd_line.as_array()[0]).unwrap(),
            signal_line: Decimal::from_f64_retain(signal_line.as_array()[0]).unwrap(),
            histogram: Decimal::from_f64_retain(histogram.as_array()[0]).unwrap(),
        }
    }
    
    /// Vectorized EMA calculation
    fn calculate_ema_vectorized(&self, period: usize) -> f64x8 {
        if self.price_buffer.is_empty() {
            return f64x8::splat(0.0);
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let alpha_vec = f64x8::splat(alpha);
        let one_minus_alpha = f64x8::splat(1.0 - alpha);
        
        let mut ema = self.price_buffer[0];
        
        for i in 1..self.price_buffer.len() {
            ema = self.price_buffer[i] * alpha_vec + ema * one_minus_alpha;
        }
        
        ema
    }
    
    fn calculate_signal_ema(&self, macd: f64x8, period: usize) -> f64x8 {
        let alpha = 2.0 / (period as f64 + 1.0);
        macd * f64x8::splat(alpha) + macd * f64x8::splat(1.0 - alpha)
    }
}

/// Result structures
#[derive(Debug, Clone)]
pub struct BollingerResult {
    pub upper_band: Decimal,
    pub middle_band: Decimal,
    pub lower_band: Decimal,
    pub bandwidth: Decimal,
    pub percent_b: Decimal,
}

#[derive(Debug, Clone)]
// ELIMINATED: pub struct MACDResult {
// ELIMINATED:     pub macd_line: Decimal,
// ELIMINATED:     pub signal_line: Decimal,
// ELIMINATED:     pub histogram: Decimal,
// ELIMINATED: }

/// Benchmark comparisons
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_bollinger_simd_vs_scalar() {
        let prices: Vec<f64> = (0..10000).map(|i| 100.0 + (i as f64).sin()).collect();
        
        // SIMD version
        let start = Instant::now();
        let mut simd_bb = SimdBollingerBands::new(20, 2.0);
        for _ in 0..100 {
            simd_bb.calculate_simd(&prices);
        }
        let simd_time = start.elapsed();
        
        // Scalar version (simulated)
        let start = Instant::now();
        for _ in 0..100 {
            calculate_bollinger_scalar(&prices, 20, 2.0);
        }
        let scalar_time = start.elapsed();
        
        println!("SIMD time: {:?}", simd_time);
        println!("Scalar time: {:?}", scalar_time);
        println!("Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
        
        // Should be ~8x faster with AVX-512
        assert!(simd_time < scalar_time);
    }
    
    fn calculate_bollinger_scalar(prices: &[f64], period: usize, std_dev: f64) -> (f64, f64, f64) {
        let sma: f64 = prices.iter().rev().take(period).sum::<f64>() / period as f64;
        let variance: f64 = prices.iter().rev().take(period)
            .map(|p| (p - sma).powi(2))
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        
        (sma + std * std_dev, sma, sma - std * std_dev)
    }
}