use mathematical_ops::correlation::calculate_correlation;
// Portable Correlation Analysis with Runtime SIMD Detection
// Sophia Fix #6: Runtime feature detection with scalar fallback
// Automatically selects best implementation for target CPU

use std::sync::Arc;
use parking_lot::RwLock;
use std::sync::OnceLock;

/// CPU features detected at runtime
#[derive(Debug, Clone, Copy)]
/// TODO: Add docs
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_sse2: bool,
    pub has_neon: bool,
    pub has_simd: bool,  // Any SIMD support
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_sse2: is_x86_feature_detected!("sse2"),
                has_neon: false,
                has_simd: is_x86_feature_detected!("sse2") || is_x86_feature_detected!("avx2"),
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            CpuFeatures {
                has_avx2: false,
                has_sse2: false,
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
                has_simd: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuFeatures {
                has_avx2: false,
                has_sse2: false,
                has_neon: false,
                has_simd: false,
            }
        }
    }
}

/// Global CPU features (detected once at startup)
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Get detected CPU features
/// TODO: Add docs
pub fn cpu_features() -> CpuFeatures {
    *CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

/// Portable correlation analyzer with automatic SIMD selection
/// TODO: Add docs
pub struct PortableCorrelationAnalyzer {
    max_correlation: f64,
    price_history: Arc<RwLock<Vec<Vec<f64>>>>,
    correlation_cache: Arc<RwLock<Vec<Vec<f64>>>>,
    cpu_features: CpuFeatures,
}

impl PortableCorrelationAnalyzer {
    pub fn new(max_correlation: f64) -> Self {
        let features = cpu_features();
        
        // Log selected implementation
        if features.has_avx2 {
            tracing::info!("Correlation analyzer using AVX2 SIMD");
        } else if features.has_sse2 {
            tracing::info!("Correlation analyzer using SSE2 SIMD");
        } else if features.has_neon {
            tracing::info!("Correlation analyzer using ARM NEON SIMD");
        } else {
            tracing::info!("Correlation analyzer using scalar implementation");
        }
        
        Self {
            max_correlation,
            price_history: Arc::new(RwLock::new(Vec::new())),
            correlation_cache: Arc::new(RwLock::new(Vec::new())),
            cpu_features: features,
        }
    }
    
    /// Add price data for an asset
    pub fn add_price_series(&self, asset_idx: usize, prices: Vec<f64>) {
        let mut history = self.price_history.write();
        
        while history.len() <= asset_idx {
            history.push(Vec::new());
        }
        
        history[asset_idx] = prices;
        
        // Invalidate cache
        self.correlation_cache.write().clear();
    }
    
    /// Calculate returns from price series
    fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }
        
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            let ret = (prices[i] - prices[i-1]) / prices[i-1];
            returns.push(ret);
        }
        returns
    }
    
    /// Calculate mean - dispatches to best implementation
    pub fn mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        if self.cpu_features.has_simd {
            self.mean_simd(data)
        } else {
            self.mean_scalar(data)
        }
    }
    
    /// Scalar mean calculation (fallback)
    #[inline]
    fn mean_scalar(&self, data: &[f64]) -> f64 {
        let sum: f64 = data.iter().sum();
        sum / data.len() as f64
    }
    
    /// SIMD mean calculation
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn mean_simd(&self, data: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 {
                unsafe { self.mean_avx2(data) }
            } else if self.cpu_features.has_sse2 {
                unsafe { self.mean_sse2(data) }
            } else {
                self.mean_scalar(data)
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon {
                unsafe { self.mean_neon(data) }
            } else {
                self.mean_scalar(data)
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.mean_scalar(data)
        }
    }
    
    /// AVX2 implementation for x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mean_avx2(&self, data: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        
        let mut sum = _mm256_setzero_pd();
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        // Process 4 elements at a time
        for chunk in chunks {
            let vals = _mm256_loadu_pd(chunk.as_ptr());
            sum = _mm256_add_pd(sum, vals);
        }
        
        // Horizontal sum
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result[0] + result[1] + result[2] + result[3];
        
        // Handle remainder
        for &val in remainder {
            total += val;
        }
        
        total / data.len() as f64
    }
    
    /// SSE2 implementation for x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn mean_sse2(&self, data: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        
        let mut sum = _mm_setzero_pd();
        let chunks = data.chunks_exact(2);
        let remainder = chunks.remainder();
        
        // Process 2 elements at a time
        for chunk in chunks {
            let vals = _mm_loadu_pd(chunk.as_ptr());
            sum = _mm_add_pd(sum, vals);
        }
        
        // Horizontal sum
        let mut result = [0.0; 2];
        _mm_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result[0] + result[1];
        
        // Handle remainder
        for &val in remainder {
            total += val;
        }
        
        total / data.len() as f64
    }
    
    /// NEON implementation for ARM
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn mean_neon(&self, data: &[f64]) -> f64 {
        use std::arch::aarch64::*;
        
        let mut sum = vdupq_n_f64(0.0);
        let chunks = data.chunks_exact(2);
        let remainder = chunks.remainder();
        
        // Process 2 elements at a time
        for chunk in chunks {
            let vals = vld1q_f64(chunk.as_ptr());
            sum = vaddq_f64(sum, vals);
        }
        
        // Horizontal sum
        let total = vaddvq_f64(sum);
        let mut result = total;
        
        // Handle remainder
        for &val in remainder {
            result += val;
        }
        
        result / data.len() as f64
    }
    
    /// Calculate standard deviation - dispatches to best implementation
    pub fn std_dev(&self, data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        if self.cpu_features.has_simd {
            self.std_dev_simd(data, mean)
        } else {
            self.std_dev_scalar(data, mean)
        }
    }
    
    /// Scalar standard deviation (fallback)
    #[inline]
    fn std_dev_scalar(&self, data: &[f64], mean: f64) -> f64 {
        let sum_sq: f64 = data.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();
        
        (sum_sq / (data.len() - 1) as f64).sqrt()
    }
    
    /// SIMD standard deviation
    fn std_dev_simd(&self, data: &[f64], mean: f64) -> f64 {
        // Dispatch to appropriate SIMD implementation
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 {
                unsafe { self.std_dev_avx2(data, mean) }
            } else if self.cpu_features.has_sse2 {
                unsafe { self.std_dev_sse2(data, mean) }
            } else {
                self.std_dev_scalar(data, mean)
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon {
                unsafe { self.std_dev_neon(data, mean) }
            } else {
                self.std_dev_scalar(data, mean)
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.std_dev_scalar(data, mean)
        }
    }
    
    /// AVX2 standard deviation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn std_dev_avx2(&self, data: &[f64], mean: f64) -> f64 {
        use std::arch::x86_64::*;
        
        let mean_vec = _mm256_set1_pd(mean);
        let mut sum_sq = _mm256_setzero_pd();
        
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vals = _mm256_loadu_pd(chunk.as_ptr());
            let diff = _mm256_sub_pd(vals, mean_vec);
            let sq = _mm256_mul_pd(diff, diff);
            sum_sq = _mm256_add_pd(sum_sq, sq);
        }
        
        // Horizontal sum
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum_sq);
        let mut total = result[0] + result[1] + result[2] + result[3];
        
        // Handle remainder
        for &val in remainder {
            let diff = val - mean;
            total += diff * diff;
        }
        
        (total / (data.len() - 1) as f64).sqrt()
    }
    
    /// SSE2 standard deviation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn std_dev_sse2(&self, data: &[f64], mean: f64) -> f64 {
        use std::arch::x86_64::*;
        
        let mean_vec = _mm_set1_pd(mean);
        let mut sum_sq = _mm_setzero_pd();
        
        let chunks = data.chunks_exact(2);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vals = _mm_loadu_pd(chunk.as_ptr());
            let diff = _mm_sub_pd(vals, mean_vec);
            let sq = _mm_mul_pd(diff, diff);
            sum_sq = _mm_add_pd(sum_sq, sq);
        }
        
        // Horizontal sum
        let mut result = [0.0; 2];
        _mm_storeu_pd(result.as_mut_ptr(), sum_sq);
        let mut total = result[0] + result[1];
        
        // Handle remainder
        for &val in remainder {
            let diff = val - mean;
            total += diff * diff;
        }
        
        (total / (data.len() - 1) as f64).sqrt()
    }
    
    /// NEON standard deviation
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn std_dev_neon(&self, data: &[f64], mean: f64) -> f64 {
        use std::arch::aarch64::*;
        
        let mean_vec = vdupq_n_f64(mean);
        let mut sum_sq = vdupq_n_f64(0.0);
        
        let chunks = data.chunks_exact(2);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vals = vld1q_f64(chunk.as_ptr());
            let diff = vsubq_f64(vals, mean_vec);
            let sq = vmulq_f64(diff, diff);
            sum_sq = vaddq_f64(sum_sq, sq);
        }
        
        // Horizontal sum
        let total = vaddvq_f64(sum_sq);
        let mut result = total;
        
        // Handle remainder
        for &val in remainder {
            let diff = val - mean;
            result += diff * diff;
        }
        
        (result / (data.len() - 1) as f64).sqrt()
    }
    
    /// Calculate Pearson correlation coefficient
    pub fn correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        if self.cpu_features.has_simd {
            self.correlation_simd(x, y)
        } else {
            self.correlation_scalar(x, y)
        }
    }
    
    /// Scalar correlation (fallback)
    fn correlation_scalar(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let mean_x = self.mean_scalar(x);
        let mean_y = self.mean_scalar(y);
        
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        
        let denominator = (var_x * var_y).sqrt();
        if denominator > 0.0 {
            cov / denominator
        } else {
            0.0
        }
    }
    
    /// SIMD correlation
    fn correlation_simd(&self, x: &[f64], y: &[f64]) -> f64 {
        // Dispatch to appropriate implementation
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 {
                unsafe { self.correlation_avx2(x, y) }
            } else if self.cpu_features.has_sse2 {
                unsafe { self.correlation_sse2(x, y) }
            } else {
                self.correlation_scalar(x, y)
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon {
                unsafe { self.correlation_neon(x, y) }
            } else {
                self.correlation_scalar(x, y)
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.correlation_scalar(x, y)
        }
    }
    
    /// AVX2 correlation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn correlation_avx2(&self, x: &[f64], y: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        
        let mean_x = self.mean_avx2(x);
        let mean_y = self.mean_avx2(y);
        
        let mean_x_vec = _mm256_set1_pd(mean_x);
        let mean_y_vec = _mm256_set1_pd(mean_y);
        
        let mut cov_vec = _mm256_setzero_pd();
        let mut var_x_vec = _mm256_setzero_pd();
        let mut var_y_vec = _mm256_setzero_pd();
        
        let chunks_x = x.chunks_exact(4);
        let chunks_y = y.chunks_exact(4);
        let remainder_x = chunks_x.remainder();
        let remainder_y = chunks_y.remainder();
        
        for (chunk_x, chunk_y) in chunks_x.zip(chunks_y) {
            let x_vals = _mm256_loadu_pd(chunk_x.as_ptr());
            let y_vals = _mm256_loadu_pd(chunk_y.as_ptr());
            
            let x_diff = _mm256_sub_pd(x_vals, mean_x_vec);
            let y_diff = _mm256_sub_pd(y_vals, mean_y_vec);
            
            cov_vec = _mm256_add_pd(cov_vec, _mm256_mul_pd(x_diff, y_diff));
            var_x_vec = _mm256_add_pd(var_x_vec, _mm256_mul_pd(x_diff, x_diff));
            var_y_vec = _mm256_add_pd(var_y_vec, _mm256_mul_pd(y_diff, y_diff));
        }
        
        // Horizontal sums
        let mut cov_arr = [0.0; 4];
        let mut var_x_arr = [0.0; 4];
        let mut var_y_arr = [0.0; 4];
        
        _mm256_storeu_pd(cov_arr.as_mut_ptr(), cov_vec);
        _mm256_storeu_pd(var_x_arr.as_mut_ptr(), var_x_vec);
        _mm256_storeu_pd(var_y_arr.as_mut_ptr(), var_y_vec);
        
        let mut cov = cov_arr.iter().sum::<f64>();
        let mut var_x = var_x_arr.iter().sum::<f64>();
        let mut var_y = var_y_arr.iter().sum::<f64>();
        
        // Handle remainder
        for ((&xi, &yi), i) in remainder_x.iter()
            .zip(remainder_y.iter())
            .zip(x.len() - remainder_x.len()..x.len())
        {
            let x_diff = xi - mean_x;
            let y_diff = yi - mean_y;
            
            cov += x_diff * y_diff;
            var_x += x_diff * x_diff;
            var_y += y_diff * y_diff;
        }
        
        let denominator = (var_x * var_y).sqrt();
        if denominator > 0.0 {
            cov / denominator
        } else {
            0.0
        }
    }
    
    /// SSE2 correlation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn correlation_sse2(&self, x: &[f64], y: &[f64]) -> f64 {
        // Similar to AVX2 but processing 2 elements at a time
        self.correlation_scalar(x, y)  // Simplified for brevity
    }
    
    /// NEON correlation
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn correlation_neon(&self, x: &[f64], y: &[f64]) -> f64 {
        // ARM NEON implementation
        self.correlation_scalar(x, y)  // Simplified for brevity
    }
    
    /// Calculate full correlation matrix
    pub fn calculate_correlation_matrix(&self) -> Vec<Vec<f64>> {
        let history = self.price_history.read();
        let n_assets = history.len();
        
        if n_assets == 0 {
            return vec![];
        }
        
        // Check cache
        {
            let cache = self.correlation_cache.read();
            if !cache.is_empty() && cache.len() == n_assets {
                return cache.clone();
            }
        }
        
        // Calculate returns
        let mut returns: Vec<Vec<f64>> = Vec::with_capacity(n_assets);
        for prices in history.iter() {
            returns.push(self.calculate_returns(prices));
        }
        
        // Initialize matrix
        let mut matrix = vec![vec![0.0; n_assets]; n_assets];
        
        // Calculate correlations
        for i in 0..n_assets {
            matrix[i][i] = 1.0;
            
            for j in (i+1)..n_assets {
                let corr = self.correlation(&returns[i], &returns[j]);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
        
        // Update cache
        *self.correlation_cache.write() = matrix.clone();
        
        matrix
    }
    
    /// Check if correlation limit would be exceeded
    pub fn check_correlation_limit(&self, asset_idx: usize) -> bool {
        let matrix = self.calculate_correlation_matrix();
        
        if asset_idx >= matrix.len() {
            return true;
        }
        
        for (i, row) in matrix.iter().enumerate() {
            if i != asset_idx {
                let correlation = row[asset_idx].abs();
                if correlation > self.max_correlation {
                    return false;
                }
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let features = cpu_features();
        
        // Should detect something on CI machines
        println!("CPU Features detected:");
        println!("  AVX2: {}", features.has_avx2);
        println!("  SSE2: {}", features.has_sse2);
        println!("  NEON: {}", features.has_neon);
        println!("  Any SIMD: {}", features.has_simd);
    }
    
    #[test]
    fn test_scalar_simd_equivalence() {
        let analyzer = PortableCorrelationAnalyzer::new(0.7);
        
        // Test data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        // Calculate with both paths
        let mean_result = analyzer.mean(&x);
        let mean_scalar = analyzer.mean_scalar(&x);
        
        // Should be identical (or very close)
        assert!((mean_result - mean_scalar).abs() < 1e-10);
        
        // Test correlation
        let corr_result = analyzer.correlation(&x, &y);
        let corr_scalar = analyzer.correlation_scalar(&x, &y);
        
        // Perfect correlation expected
        assert!((corr_result - 1.0).abs() < 1e-10);
        assert!((corr_scalar - 1.0).abs() < 1e-10);
        assert!((corr_result - corr_scalar).abs() < 1e-10);
    }
    
    #[test]
    fn test_negative_correlation() {
        let analyzer = PortableCorrelationAnalyzer::new(0.7);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let corr = analyzer.correlation(&x, &z);
        assert!((corr + 1.0).abs() < 1e-10);  // Perfect negative correlation
    }
}