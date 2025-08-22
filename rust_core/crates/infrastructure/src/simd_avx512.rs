// AVX-512 SIMD Optimizations for Hot Paths
// Team: Jordan (Performance) & Morgan (ML)
// Target: Intel Xeon Gold 6242 with AVX-512 support
// Performance: 16 floats or 8 doubles per instruction

use std::arch::x86_64::*;

/// AVX-512 optimized price calculations
/// Processes 16 prices in parallel
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn calculate_sma_avx512(prices: &[f32]) -> f32 {
    let len = prices.len();
    if len == 0 {
        return 0.0;
    }
    
    // Process 16 floats at a time with AVX-512
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    
    // Main AVX-512 loop - 16 elements per iteration
    while i + 16 <= len {
        let chunk = _mm512_loadu_ps(prices.as_ptr().add(i));
        sum = _mm512_add_ps(sum, chunk);
        i += 16;
    }
    
    // Horizontal sum of AVX-512 register
    let sum_scalar = _mm512_reduce_add_ps(sum);
    
    // Handle remaining elements
    let mut remainder_sum = 0.0f32;
    while i < len {
        remainder_sum += prices[i];
        i += 1;
    }
    
    (sum_scalar + remainder_sum) / len as f32
}

/// AVX-512 optimized EMA calculation
/// Alpha blending with 16-wide SIMD
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn calculate_ema_avx512(
    prices: &[f32],
    prev_ema: &[f32],
    alpha: f32,
    output: &mut [f32]
) {
    let len = prices.len().min(prev_ema.len()).min(output.len());
    
    // Broadcast alpha and (1-alpha) to all lanes
    let alpha_vec = _mm512_set1_ps(alpha);
    let one_minus_alpha = _mm512_set1_ps(1.0 - alpha);
    
    let mut i = 0;
    
    // Process 16 elements at a time
    while i + 16 <= len {
        let price_vec = _mm512_loadu_ps(prices.as_ptr().add(i));
        let prev_vec = _mm512_loadu_ps(prev_ema.as_ptr().add(i));
        
        // EMA = alpha * price + (1-alpha) * prev_ema
        let new_term = _mm512_mul_ps(alpha_vec, price_vec);
        let old_term = _mm512_mul_ps(one_minus_alpha, prev_vec);
        let ema = _mm512_add_ps(new_term, old_term);
        
        _mm512_storeu_ps(output.as_mut_ptr().add(i), ema);
        i += 16;
    }
    
    // Handle remainder with scalar code
    while i < len {
        output[i] = alpha * prices[i] + (1.0 - alpha) * prev_ema[i];
        i += 1;
    }
}

/// AVX-512 optimized RSI calculation
/// Processes gain/loss arrays in parallel
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn calculate_rsi_avx512(gains: &[f32], losses: &[f32]) -> Vec<f32> {
    let len = gains.len().min(losses.len());
    let mut rsi = vec![0.0f32; len];
    
    let hundred = _mm512_set1_ps(100.0);
    // Optimized formula doesn't need 'one' constant
    
    let mut i = 0;
    
    // Process 16 elements at a time
    while i + 16 <= len {
        let gain_vec = _mm512_loadu_ps(gains.as_ptr().add(i));
        let loss_vec = _mm512_loadu_ps(losses.as_ptr().add(i));
        
        // RS = avg_gain / avg_loss
        // RSI = 100 - (100 / (1 + RS))
        // Optimized: RSI = 100 * avg_gain / (avg_gain + avg_loss)
        
        let sum = _mm512_add_ps(gain_vec, loss_vec);
        
        // Mask for zero division protection
        let mask = _mm512_cmp_ps_mask(sum, _mm512_setzero_ps(), _CMP_GT_OQ);
        
        // Safe division with mask
        let ratio = _mm512_mask_div_ps(
            _mm512_setzero_ps(),
            mask,
            gain_vec,
            sum
        );
        
        let rsi_vec = _mm512_mul_ps(hundred, ratio);
        
        _mm512_storeu_ps(rsi.as_mut_ptr().add(i), rsi_vec);
        i += 16;
    }
    
    // Handle remainder
    while i < len {
        let sum = gains[i] + losses[i];
        rsi[i] = if sum > 0.0 {
            100.0 * gains[i] / sum
        } else {
            50.0  // Neutral RSI when no movement
        };
        i += 1;
    }
    
    rsi
}

/// AVX-512 optimized Bollinger Bands calculation
/// Calculates upper, middle, lower bands in parallel
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn calculate_bollinger_avx512(
    prices: &[f32],
    sma: f32,
    num_std: f32
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = prices.len();
    let mut upper = vec![0.0f32; len];
    let mut middle = vec![sma; len];
    let mut lower = vec![0.0f32; len];
    
    // Calculate standard deviation with AVX-512
    let sma_vec = _mm512_set1_ps(sma);
    let mut variance = _mm512_setzero_ps();
    
    let mut i = 0;
    
    // Variance calculation - 16 elements at a time
    while i + 16 <= len {
        let price_vec = _mm512_loadu_ps(prices.as_ptr().add(i));
        let diff = _mm512_sub_ps(price_vec, sma_vec);
        let squared = _mm512_mul_ps(diff, diff);
        variance = _mm512_add_ps(variance, squared);
        i += 16;
    }
    
    // Horizontal sum for variance
    let mut var_sum = _mm512_reduce_add_ps(variance);
    
    // Handle remainder
    while i < len {
        let diff = prices[i] - sma;
        var_sum += diff * diff;
        i += 1;
    }
    
    let std_dev = (var_sum / len as f32).sqrt();
    let band_width = num_std * std_dev;
    
    let upper_bound = _mm512_set1_ps(sma + band_width);
    let lower_bound = _mm512_set1_ps(sma - band_width);
    
    // Fill band arrays
    i = 0;
    while i + 16 <= len {
        _mm512_storeu_ps(upper.as_mut_ptr().add(i), upper_bound);
        _mm512_storeu_ps(lower.as_mut_ptr().add(i), lower_bound);
        i += 16;
    }
    
    // Handle remainder
    while i < len {
        upper[i] = sma + band_width;
        lower[i] = sma - band_width;
        i += 1;
    }
    
    (upper, middle, lower)
}

/// AVX-512 optimized MACD calculation
/// Processes multiple EMAs simultaneously
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn calculate_macd_avx512(
    prices: &[f32],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = prices.len();
    let mut macd_line = vec![0.0f32; len];
    let mut signal_line = vec![0.0f32; len];
    let mut histogram = vec![0.0f32; len];
    
    // Calculate EMAs with different periods
    let alpha_fast = 2.0 / (fast_period as f32 + 1.0);
    let alpha_slow = 2.0 / (slow_period as f32 + 1.0);
    let alpha_signal = 2.0 / (signal_period as f32 + 1.0);
    
    let ema_fast_init = vec![prices[0]; len];
    let ema_slow_init = vec![prices[0]; len];
    let mut ema_fast = vec![0.0f32; len];
    let mut ema_slow = vec![0.0f32; len];
    
    // Calculate fast and slow EMAs
    calculate_ema_avx512(prices, &ema_fast_init, alpha_fast, &mut ema_fast);
    calculate_ema_avx512(prices, &ema_slow_init, alpha_slow, &mut ema_slow);
    
    // Calculate MACD line (fast - slow)
    let mut i = 0;
    while i + 16 <= len {
        let fast_vec = _mm512_loadu_ps(ema_fast.as_ptr().add(i));
        let slow_vec = _mm512_loadu_ps(ema_slow.as_ptr().add(i));
        let macd = _mm512_sub_ps(fast_vec, slow_vec);
        _mm512_storeu_ps(macd_line.as_mut_ptr().add(i), macd);
        i += 16;
    }
    
    while i < len {
        macd_line[i] = ema_fast[i] - ema_slow[i];
        i += 1;
    }
    
    // Calculate signal line (EMA of MACD)
    let signal_init = vec![macd_line[0]; len];
    calculate_ema_avx512(&macd_line, &signal_init, alpha_signal, &mut signal_line);
    
    // Calculate histogram (MACD - signal)
    i = 0;
    while i + 16 <= len {
        let macd_vec = _mm512_loadu_ps(macd_line.as_ptr().add(i));
        let signal_vec = _mm512_loadu_ps(signal_line.as_ptr().add(i));
        let hist = _mm512_sub_ps(macd_vec, signal_vec);
        _mm512_storeu_ps(histogram.as_mut_ptr().add(i), hist);
        i += 16;
    }
    
    while i < len {
        histogram[i] = macd_line[i] - signal_line[i];
        i += 1;
    }
    
    (macd_line, signal_line, histogram)
}

/// AVX-512 vector dot product
/// Multiplies and sums 16 pairs in parallel
#[target_feature(enable = "avx512f")]
#[inline]  // Cannot use inline(always) with target_feature
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = _mm512_setzero_ps();
    
    let mut i = 0;
    
    // Process 16 pairs at a time
    while i + 16 <= len {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let prod = _mm512_mul_ps(a_vec, b_vec);
        sum = _mm512_add_ps(sum, prod);
        i += 16;
    }
    
    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);
    
    // Handle remainder
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Check if AVX-512 is available at runtime
#[inline(always)]
pub fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Benchmark structure for SIMD operations
pub struct SimdBenchmark {
    pub name: &'static str,
    pub scalar_ns: u64,
    pub avx512_ns: u64,
    pub speedup: f32,
}

impl SimdBenchmark {
    pub fn print(&self) {
        println!("SIMD Benchmark: {}", self.name);
        println!("  Scalar: {}ns", self.scalar_ns);
        println!("  AVX-512: {}ns", self.avx512_ns);
        println!("  Speedup: {:.1}x", self.speedup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_avx512_available() {
        // Should be true on Intel Xeon Gold 6242
        assert!(is_avx512_available());
    }
    
    #[test]
    fn test_sma_avx512() {
        if !is_avx512_available() {
            return;
        }
        
        let prices: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        
        unsafe {
            let result = calculate_sma_avx512(&prices);
            let expected = 511.5;  // Average of 0..1024
            assert!((result - expected).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_ema_avx512() {
        if !is_avx512_available() {
            return;
        }
        
        let prices = vec![100.0; 64];
        let prev_ema = vec![95.0; 64];
        let mut output = vec![0.0; 64];
        let alpha = 0.1;
        
        unsafe {
            calculate_ema_avx512(&prices, &prev_ema, alpha, &mut output);
        }
        
        // EMA = 0.1 * 100 + 0.9 * 95 = 10 + 85.5 = 95.5
        for val in output {
            assert!((val - 95.5).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_performance_comparison() {
        if !is_avx512_available() {
            return;
        }
        
        let prices: Vec<f32> = (0..10000).map(|i| (i as f32).sin() * 100.0 + 1000.0).collect();
        
        // Scalar version
        let start = Instant::now();
        let mut scalar_sum = 0.0f32;
        for &p in &prices {
            scalar_sum += p;
        }
        let scalar_avg = scalar_sum / prices.len() as f32;
        let scalar_time = start.elapsed();
        
        // AVX-512 version
        let start = Instant::now();
        let avx512_avg = unsafe { calculate_sma_avx512(&prices) };
        let avx512_time = start.elapsed();
        
        // Results should match
        assert!((scalar_avg - avx512_avg).abs() < 0.1);
        
        // AVX-512 should be faster
        let speedup = scalar_time.as_nanos() as f32 / avx512_time.as_nanos() as f32;
        println!("AVX-512 speedup: {:.1}x", speedup);
        assert!(speedup > 1.0);  // Should be faster
    }
    
    #[test]
    fn test_dot_product_avx512() {
        if !is_avx512_available() {
            return;
        }
        
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();
        
        unsafe {
            let result = dot_product_avx512(&a, &b);
            
            // Calculate expected result
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            
            assert!((result - expected).abs() < 1.0);
        }
    }
}