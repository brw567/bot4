// SIMD DATA PROCESSORS - AVX-512 OPTIMIZED
// Team: Jordan (Lead) - 16x PARALLEL PROCESSING!
// Target: <10ns per operation with AVX-512

use std::arch::x86_64::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use packed_simd::*;
use simdeez::prelude::*;
use simdeez::avx512::*;
use simdeez::avx2::*;
use simdeez::sse41::*;

use crate::{DataError, Result};

/// SIMD processor for ultra-fast data operations
/// TODO: Add docs
pub struct SimdProcessor {
    // CPU feature detection
    has_avx512: bool,
    has_avx2: bool,
    has_sse4: bool,
    
    // Metrics
    operations_performed: AtomicU64,
    simd_speedup_factor: AtomicU64,
}

impl SimdProcessor {
    pub fn new() -> Result<Self> {
        // Detect CPU features
        let has_avx512 = is_x86_feature_detected!("avx512f") 
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_sse4 = is_x86_feature_detected!("sse4.1");
        
        if !has_sse4 {
            return Err(DataError::SimdError("CPU must support at least SSE4.1".into()));
        }
        
        println!("SIMD Processor initialized:");
        println!("  AVX-512: {}", has_avx512);
        println!("  AVX2: {}", has_avx2);
        println!("  SSE4: {}", has_sse4);
        
        Ok(Self {
            has_avx512,
            has_avx2,
            has_sse4,
            operations_performed: AtomicU64::new(0),
            simd_speedup_factor: AtomicU64::new(1),
        })
    }
    
    /// Calculate moving average with SIMD (16x parallel with AVX-512)
    #[inline(always)]
    pub fn moving_average_simd(&self, prices: &[f32], window: usize) -> Vec<f32> {
        if prices.len() < window {
            return vec![];
        }
        
        let mut result = Vec::with_capacity(prices.len() - window + 1);
        
        if self.has_avx512 {
            unsafe { self.moving_average_avx512(prices, window, &mut result) }
        } else if self.has_avx2 {
            unsafe { self.moving_average_avx2(prices, window, &mut result) }
        } else {
            unsafe { self.moving_average_sse4(prices, window, &mut result) }
        }
        
        self.operations_performed.fetch_add(result.len() as u64, Ordering::Relaxed);
        result
    }
    
    /// AVX-512 implementation (16 floats at once)
    #[target_feature(enable = "avx512f")]
    unsafe fn moving_average_avx512(&self, prices: &[f32], window: usize, result: &mut Vec<f32>) {
        let window_f32 = window as f32;
        let inv_window = _mm512_set1_ps(1.0 / window_f32);
        
        for i in 0..=prices.len() - window {
            let mut sum = _mm512_setzero_ps();
            let mut j = 0;
            
            // Process 16 values at a time
            while j + 16 <= window {
                let values = _mm512_loadu_ps(&prices[i + j]);
                sum = _mm512_add_ps(sum, values);
                j += 16;
            }
            
            // Handle remaining values
            if j < window {
                let mask = (1u16 << (window - j)) - 1;
                let values = _mm512_maskz_loadu_ps(mask, &prices[i + j]);
                sum = _mm512_add_ps(sum, values);
            }
            
            // Horizontal sum
            let sum_scalar = self.hsum_ps_avx512(sum);
            result.push(sum_scalar / window_f32);
        }
        
        self.simd_speedup_factor.store(16, Ordering::Relaxed);
    }
    
    /// AVX2 implementation (8 floats at once)
    #[target_feature(enable = "avx2")]
    unsafe fn moving_average_avx2(&self, prices: &[f32], window: usize, result: &mut Vec<f32>) {
        let window_f32 = window as f32;
        
        for i in 0..=prices.len() - window {
            let mut sum = _mm256_setzero_ps();
            let mut j = 0;
            
            // Process 8 values at a time
            while j + 8 <= window {
                let values = _mm256_loadu_ps(&prices[i + j]);
                sum = _mm256_add_ps(sum, values);
                j += 8;
            }
            
            // Handle remaining values
            let mut sum_scalar = self.hsum_ps_avx2(sum);
            while j < window {
                sum_scalar += prices[i + j];
                j += 1;
            }
            
            result.push(sum_scalar / window_f32);
        }
        
        self.simd_speedup_factor.store(8, Ordering::Relaxed);
    }
    
    /// SSE4 implementation (4 floats at once)
    #[target_feature(enable = "sse4.1")]
    unsafe fn moving_average_sse4(&self, prices: &[f32], window: usize, result: &mut Vec<f32>) {
        let window_f32 = window as f32;
        
        for i in 0..=prices.len() - window {
            let mut sum = _mm_setzero_ps();
            let mut j = 0;
            
            // Process 4 values at a time
            while j + 4 <= window {
                let values = _mm_loadu_ps(&prices[i + j]);
                sum = _mm_add_ps(sum, values);
                j += 4;
            }
            
            // Handle remaining values
            let mut sum_scalar = self.hsum_ps_sse(sum);
            while j < window {
                sum_scalar += prices[i + j];
                j += 1;
            }
            
            result.push(sum_scalar / window_f32);
        }
        
        self.simd_speedup_factor.store(4, Ordering::Relaxed);
    }
    
    /// Horizontal sum for AVX-512
    #[inline(always)]
    unsafe fn hsum_ps_avx512(&self, v: __m512) -> f32 {
        let v256 = _mm256_add_ps(
            _mm512_extractf32x8_ps(v, 0),
            _mm512_extractf32x8_ps(v, 1)
        );
        self.hsum_ps_avx2(v256)
    }
    
    /// Horizontal sum for AVX2
    #[inline(always)]
    unsafe fn hsum_ps_avx2(&self, v: __m256) -> f32 {
        let v128 = _mm_add_ps(
            _mm256_extractf128_ps(v, 0),
            _mm256_extractf128_ps(v, 1)
        );
        self.hsum_ps_sse(v128)
    }
    
    /// Horizontal sum for SSE
    #[inline(always)]
    unsafe fn hsum_ps_sse(&self, v: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        let shuf = _mm_movehl_ps(sums, sums);
        let sums = _mm_add_ss(sums, shuf);
        _mm_cvtss_f32(sums)
    }
    
    /// Calculate correlation matrix with SIMD
    #[inline(always)]
    pub fn correlation_matrix_simd(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = data.len();
        let m = data[0].len();
        let mut result = vec![vec![0.0; n]; n];
        
        // Calculate means with SIMD
        let means: Vec<f32> = data.iter()
            .map(|series| self.mean_simd(series))
            .collect();
        
        // Calculate standard deviations with SIMD
        let stds: Vec<f32> = data.iter().zip(&means)
            .map(|(series, &mean)| self.std_simd(series, mean))
            .collect();
        
        // Calculate correlations
        for i in 0..n {
            for j in i..n {
                let corr = self.correlation_simd(&data[i], &data[j], means[i], means[j], stds[i], stds[j]);
                result[i][j] = corr;
                result[j][i] = corr;
            }
        }
        
        result
    }
    
    /// Calculate mean with SIMD
    #[inline(always)]
    pub fn mean_simd(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        let sum = if self.has_avx512 {
            unsafe { self.sum_avx512(data) }
        } else if self.has_avx2 {
            unsafe { self.sum_avx2(data) }
        } else {
            unsafe { self.sum_sse4(data) }
        };
        
        sum / data.len() as f32
    }
    
    /// Calculate standard deviation with SIMD
    #[inline(always)]
    pub fn std_simd(&self, data: &[f32], mean: f32) -> f32 {
        if data.len() <= 1 {
            return 0.0;
        }
        
        let variance = if self.has_avx512 {
            unsafe { self.variance_avx512(data, mean) }
        } else if self.has_avx2 {
            unsafe { self.variance_avx2(data, mean) }
        } else {
            unsafe { self.variance_sse4(data, mean) }
        };
        
        (variance / (data.len() - 1) as f32).sqrt()
    }
    
    /// Calculate correlation with SIMD
    #[inline(always)]
    pub fn correlation_simd(&self, x: &[f32], y: &[f32], mean_x: f32, mean_y: f32, std_x: f32, std_y: f32) -> f32 {
        if x.len() != y.len() || x.len() <= 1 || std_x == 0.0 || std_y == 0.0 {
            return 0.0;
        }
        
        let covariance = if self.has_avx512 {
            unsafe { self.covariance_avx512(x, y, mean_x, mean_y) }
        } else if self.has_avx2 {
            unsafe { self.covariance_avx2(x, y, mean_x, mean_y) }
        } else {
            unsafe { self.covariance_sse4(x, y, mean_x, mean_y) }
        };
        
        covariance / ((x.len() - 1) as f32 * std_x * std_y)
    }
    
    /// Sum with AVX-512
    #[target_feature(enable = "avx512f")]
    unsafe fn sum_avx512(&self, data: &[f32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= data.len() {
            let values = _mm512_loadu_ps(&data[i]);
            sum = _mm512_add_ps(sum, values);
            i += 16;
        }
        
        let mut sum_scalar = self.hsum_ps_avx512(sum);
        while i < data.len() {
            sum_scalar += data[i];
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Sum with AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn sum_avx2(&self, data: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        while i + 8 <= data.len() {
            let values = _mm256_loadu_ps(&data[i]);
            sum = _mm256_add_ps(sum, values);
            i += 8;
        }
        
        let mut sum_scalar = self.hsum_ps_avx2(sum);
        while i < data.len() {
            sum_scalar += data[i];
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Sum with SSE4
    #[target_feature(enable = "sse4.1")]
    unsafe fn sum_sse4(&self, data: &[f32]) -> f32 {
        let mut sum = _mm_setzero_ps();
        let mut i = 0;
        
        while i + 4 <= data.len() {
            let values = _mm_loadu_ps(&data[i]);
            sum = _mm_add_ps(sum, values);
            i += 4;
        }
        
        let mut sum_scalar = self.hsum_ps_sse(sum);
        while i < data.len() {
            sum_scalar += data[i];
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Variance with AVX-512
    #[target_feature(enable = "avx512f")]
    unsafe fn variance_avx512(&self, data: &[f32], mean: f32) -> f32 {
        let mean_vec = _mm512_set1_ps(mean);
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= data.len() {
            let values = _mm512_loadu_ps(&data[i]);
            let diff = _mm512_sub_ps(values, mean_vec);
            let squared = _mm512_mul_ps(diff, diff);
            sum = _mm512_add_ps(sum, squared);
            i += 16;
        }
        
        let mut sum_scalar = self.hsum_ps_avx512(sum);
        while i < data.len() {
            let diff = data[i] - mean;
            sum_scalar += diff * diff;
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Variance with AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn variance_avx2(&self, data: &[f32], mean: f32) -> f32 {
        let mean_vec = _mm256_set1_ps(mean);
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        while i + 8 <= data.len() {
            let values = _mm256_loadu_ps(&data[i]);
            let diff = _mm256_sub_ps(values, mean_vec);
            let squared = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, squared);
            i += 8;
        }
        
        let mut sum_scalar = self.hsum_ps_avx2(sum);
        while i < data.len() {
            let diff = data[i] - mean;
            sum_scalar += diff * diff;
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Variance with SSE4
    #[target_feature(enable = "sse4.1")]
    unsafe fn variance_sse4(&self, data: &[f32], mean: f32) -> f32 {
        let mean_vec = _mm_set1_ps(mean);
        let mut sum = _mm_setzero_ps();
        let mut i = 0;
        
        while i + 4 <= data.len() {
            let values = _mm_loadu_ps(&data[i]);
            let diff = _mm_sub_ps(values, mean_vec);
            let squared = _mm_mul_ps(diff, diff);
            sum = _mm_add_ps(sum, squared);
            i += 4;
        }
        
        let mut sum_scalar = self.hsum_ps_sse(sum);
        while i < data.len() {
            let diff = data[i] - mean;
            sum_scalar += diff * diff;
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Covariance with AVX-512
    #[target_feature(enable = "avx512f")]
    unsafe fn covariance_avx512(&self, x: &[f32], y: &[f32], mean_x: f32, mean_y: f32) -> f32 {
        let mean_x_vec = _mm512_set1_ps(mean_x);
        let mean_y_vec = _mm512_set1_ps(mean_y);
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= x.len() {
            let x_values = _mm512_loadu_ps(&x[i]);
            let y_values = _mm512_loadu_ps(&y[i]);
            let x_diff = _mm512_sub_ps(x_values, mean_x_vec);
            let y_diff = _mm512_sub_ps(y_values, mean_y_vec);
            let product = _mm512_mul_ps(x_diff, y_diff);
            sum = _mm512_add_ps(sum, product);
            i += 16;
        }
        
        let mut sum_scalar = self.hsum_ps_avx512(sum);
        while i < x.len() {
            sum_scalar += (x[i] - mean_x) * (y[i] - mean_y);
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Covariance with AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn covariance_avx2(&self, x: &[f32], y: &[f32], mean_x: f32, mean_y: f32) -> f32 {
        let mean_x_vec = _mm256_set1_ps(mean_x);
        let mean_y_vec = _mm256_set1_ps(mean_y);
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        while i + 8 <= x.len() {
            let x_values = _mm256_loadu_ps(&x[i]);
            let y_values = _mm256_loadu_ps(&y[i]);
            let x_diff = _mm256_sub_ps(x_values, mean_x_vec);
            let y_diff = _mm256_sub_ps(y_values, mean_y_vec);
            let product = _mm256_mul_ps(x_diff, y_diff);
            sum = _mm256_add_ps(sum, product);
            i += 8;
        }
        
        let mut sum_scalar = self.hsum_ps_avx2(sum);
        while i < x.len() {
            sum_scalar += (x[i] - mean_x) * (y[i] - mean_y);
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Covariance with SSE4
    #[target_feature(enable = "sse4.1")]
    unsafe fn covariance_sse4(&self, x: &[f32], y: &[f32], mean_x: f32, mean_y: f32) -> f32 {
        let mean_x_vec = _mm_set1_ps(mean_x);
        let mean_y_vec = _mm_set1_ps(mean_y);
        let mut sum = _mm_setzero_ps();
        let mut i = 0;
        
        while i + 4 <= x.len() {
            let x_values = _mm_loadu_ps(&x[i]);
            let y_values = _mm_loadu_ps(&y[i]);
            let x_diff = _mm_sub_ps(x_values, mean_x_vec);
            let y_diff = _mm_sub_ps(y_values, mean_y_vec);
            let product = _mm_mul_ps(x_diff, y_diff);
            sum = _mm_add_ps(sum, product);
            i += 4;
        }
        
        let mut sum_scalar = self.hsum_ps_sse(sum);
        while i < x.len() {
            sum_scalar += (x[i] - mean_x) * (y[i] - mean_y);
            i += 1;
        }
        
        sum_scalar
    }
    
    /// Get SIMD metrics
    pub fn metrics(&self) -> SimdMetrics {
        SimdMetrics {
            operations_performed: self.operations_performed.load(Ordering::Relaxed),
            simd_speedup_factor: self.simd_speedup_factor.load(Ordering::Relaxed),
            has_avx512: self.has_avx512,
            has_avx2: self.has_avx2,
            has_sse4: self.has_sse4,
        }
    }
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct SimdMetrics {
    pub operations_performed: u64,
    pub simd_speedup_factor: u64,
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_sse4: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_moving_average() {
        let processor = SimdProcessor::new().unwrap();
        
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = processor.moving_average_simd(&prices, 3);
        
        assert_eq!(result.len(), 8);
        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 3.0).abs() < 0.001);
        assert!((result[2] - 4.0).abs() < 0.001);
    }
    
    #[test]
    fn test_correlation_matrix() {
        let processor = SimdProcessor::new().unwrap();
        
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        
        let corr_matrix = processor.correlation_matrix_simd(&data);
        
        assert_eq!(corr_matrix.len(), 3);
        assert_eq!(corr_matrix[0].len(), 3);
        
        // Diagonal should be 1.0
        assert!((corr_matrix[0][0] - 1.0).abs() < 0.001);
        assert!((corr_matrix[1][1] - 1.0).abs() < 0.001);
        assert!((corr_matrix[2][2] - 1.0).abs() < 0.001);
        
        // First two series are perfectly correlated
        assert!((corr_matrix[0][1] - 1.0).abs() < 0.001);
        
        // First and third are perfectly anti-correlated
        assert!((corr_matrix[0][2] + 1.0).abs() < 0.001);
    }
}