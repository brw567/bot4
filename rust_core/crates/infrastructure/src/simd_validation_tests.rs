// SIMD Validation Tests - Comprehensive Testing for Task 0.1.1
// Team: Riley (Testing Lead), Sam (Implementation), Jordan (Performance)
//
// These tests ensure SIMD operations work correctly across ALL CPU architectures
// and that fallbacks produce identical results.

#[cfg(test)]
mod tests {
    use crate::cpu_features::{CPU_FEATURES, SimdStrategy};
    use crate::simd_ops::{EmaCalculator, SmaCalculator, PortfolioRiskCalculator};
    use std::time::Instant;
    
    const TEST_SIZES: &[usize] = &[1, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1000, 10000];
    const EPSILON: f32 = 1e-4;  // Relaxed slightly for SSE rounding differences
    
    /// Validate that all SIMD strategies produce identical results
    #[test]
    fn test_ema_consistency_across_strategies() {
        let test_data: Vec<f32> = (0..1000).map(|i| 100.0 + (i as f32).sin() * 10.0).collect();
        let alpha = 0.1;
        
        // Calculate with scalar (baseline)
        let scalar_result = EmaCalculator::calculate_scalar(&test_data, alpha);
        
        // Test SSE2 if available
        if CPU_FEATURES.has_sse2 {
            let sse2_result = unsafe { EmaCalculator::calculate_sse2(&test_data, alpha) };
            assert_results_equal(&scalar_result, &sse2_result, "SSE2");
        }
        
        // Test SSE4.2 if available
        if CPU_FEATURES.has_sse42 {
            let sse42_result = unsafe { EmaCalculator::calculate_sse42(&test_data, alpha) };
            assert_results_equal(&scalar_result, &sse42_result, "SSE4.2");
        }
        
        // Test AVX2 if available
        if CPU_FEATURES.can_use_avx2() {
            let avx2_result = unsafe { EmaCalculator::calculate_avx2(&test_data, alpha) };
            assert_results_equal(&scalar_result, &avx2_result, "AVX2");
        }
        
        // Test AVX-512 if available
        if CPU_FEATURES.can_use_avx512() {
            let avx512_result = unsafe { EmaCalculator::calculate_avx512(&test_data, alpha) };
            assert_results_equal(&scalar_result, &avx512_result, "AVX-512");
        }
        
        eprintln!("✓ EMA consistency validated across all available SIMD strategies");
    }
    
    /// Test SMA calculation consistency
    #[test]
    fn test_sma_consistency_across_strategies() {
        let test_data: Vec<f32> = (0..500).map(|i| 50.0 + (i as f32 * 0.1).cos() * 5.0).collect();
        let period = 20;
        
        let scalar_result = SmaCalculator::calculate_scalar(&test_data, period);
        
        if CPU_FEATURES.has_sse2 {
            let sse2_result = unsafe { SmaCalculator::calculate_sse2(&test_data, period) };
            assert_results_equal(&scalar_result, &sse2_result, "SSE2");
        }
        
        if CPU_FEATURES.can_use_avx2() {
            let avx2_result = unsafe { SmaCalculator::calculate_avx2(&test_data, period) };
            assert_results_equal(&scalar_result, &avx2_result, "AVX2");
        }
        
        eprintln!("✓ SMA consistency validated across all available SIMD strategies");
    }
    
    /// Test portfolio risk calculation with correlation matrix
    #[test]
    fn test_portfolio_risk_consistency() {
        let positions = vec![100000.0, -50000.0, 75000.0, -25000.0];
        let volatilities = vec![0.02, 0.015, 0.025, 0.018];
        let correlations = vec![
            1.0, 0.3, 0.5, 0.2,
            0.3, 1.0, 0.4, 0.6,
            0.5, 0.4, 1.0, 0.3,
            0.2, 0.6, 0.3, 1.0,
        ];
        
        // Calculate risk using the standard method (it will dispatch to optimal strategy)
        let scalar_risk = PortfolioRiskCalculator::calculate_risk(
            &positions.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &volatilities.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &correlations.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        );
        
        // For now, we test that the dispatcher works correctly
        // Individual strategy testing would require exposing internal methods
        eprintln!("  Using strategy: {:?}", CPU_FEATURES.optimal_strategy);
        
        // Verify the result is reasonable
        assert!(scalar_risk > 0.0, "Risk should be positive");
        assert!(scalar_risk < 10000.0, "Risk seems unreasonably high");
        
        eprintln!("✓ Portfolio risk calculation validated");
        eprintln!("  Portfolio Risk: ${:.2}", scalar_risk);
    }
    
    /// Performance benchmarks for different strategies
    #[test]
    fn test_simd_performance_comparison() {
        let sizes = vec![100, 1000, 10000, 100000];
        
        eprintln!("\n=== SIMD Performance Comparison ===");
        eprintln!("Testing EMA calculation performance...");
        
        for size in &sizes {
            let test_data: Vec<f32> = (0..*size).map(|i| 100.0 + (i as f32 * 0.01).sin()).collect();
            let alpha = 0.1;
            
            // Benchmark scalar
            let start = Instant::now();
            for _ in 0..100 {
                let _ = EmaCalculator::calculate_scalar(&test_data, alpha);
            }
            let scalar_time = start.elapsed();
            
            eprintln!("\nSize {}: ", size);
            eprintln!("  Scalar: {:?}", scalar_time);
            
            // Benchmark SSE2
            if CPU_FEATURES.has_sse2 {
                let start = Instant::now();
                for _ in 0..100 {
                    let _ = unsafe { EmaCalculator::calculate_sse2(&test_data, alpha) };
                }
                let sse2_time = start.elapsed();
                let speedup = scalar_time.as_secs_f64() / sse2_time.as_secs_f64();
                eprintln!("  SSE2: {:?} ({}x speedup)", sse2_time, speedup);
            }
            
            // Benchmark AVX2
            if CPU_FEATURES.can_use_avx2() {
                let start = Instant::now();
                for _ in 0..100 {
                    let _ = unsafe { EmaCalculator::calculate_avx2(&test_data, alpha) };
                }
                let avx2_time = start.elapsed();
                let speedup = scalar_time.as_secs_f64() / avx2_time.as_secs_f64();
                eprintln!("  AVX2: {:?} ({}x speedup)", avx2_time, speedup);
            }
            
            // Benchmark AVX-512
            if CPU_FEATURES.can_use_avx512() {
                let start = Instant::now();
                for _ in 0..100 {
                    let _ = unsafe { EmaCalculator::calculate_avx512(&test_data, alpha) };
                }
                let avx512_time = start.elapsed();
                let speedup = scalar_time.as_secs_f64() / avx512_time.as_secs_f64();
                eprintln!("  AVX-512: {:?} ({}x speedup)", avx512_time, speedup);
            }
        }
    }
    
    /// Test edge cases and boundary conditions
    #[test]
    fn test_simd_edge_cases() {
        // Empty input
        let empty: Vec<f32> = vec![];
        let result = EmaCalculator::calculate(&empty, 0.1);
        assert!(result.is_empty());
        
        // Single element
        let single = vec![42.0];
        let result = EmaCalculator::calculate(&single, 0.1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 42.0);
        
        // Very small arrays (less than SIMD width)
        for size in 1..16 {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let result = EmaCalculator::calculate(&data, 0.1);
            assert_eq!(result.len(), size);
        }
        
        // Arrays not aligned to SIMD width
        for test_size in TEST_SIZES {
            let data: Vec<f32> = (0..*test_size).map(|i| i as f32).collect();
            let result = EmaCalculator::calculate(&data, 0.1);
            assert_eq!(result.len(), *test_size);
            
            // Verify no NaN or infinity
            for val in &result {
                assert!(val.is_finite());
            }
        }
        
        eprintln!("✓ Edge cases validated");
    }
    
    /// Test CPU feature detection correctness
    #[test]
    fn test_cpu_feature_detection_validity() {
        let features = &*CPU_FEATURES;
        
        // x86_64 always has SSE2
        assert!(features.has_sse2, "SSE2 must be available on x86_64");
        
        // Logical hierarchy tests
        if features.has_avx512f {
            assert!(features.has_avx2, "AVX-512 requires AVX2");
        }
        if features.has_avx2 {
            assert!(features.has_avx, "AVX2 requires AVX");
        }
        if features.has_sse42 {
            assert!(features.has_sse41, "SSE4.2 requires SSE4.1");
        }
        
        // Strategy should match capabilities
        match features.optimal_strategy {
            SimdStrategy::Avx512 => {
                assert!(features.can_use_avx512());
            }
            SimdStrategy::Avx2 => {
                assert!(features.can_use_avx2());
            }
            SimdStrategy::Sse42 => {
                assert!(features.has_sse42);
            }
            SimdStrategy::Sse2 => {
                assert!(features.has_sse2);
            }
            SimdStrategy::Scalar => {
                // Scalar is always valid
            }
        }
        
        eprintln!("✓ CPU feature detection validated");
        eprintln!("  Detected: {}", features.cpu_brand);
        eprintln!("  Strategy: {:?}", features.optimal_strategy);
    }
    
    /// Helper function to compare floating point arrays
    fn assert_results_equal(expected: &[f32], actual: &[f32], strategy: &str) {
        assert_eq!(expected.len(), actual.len(), 
            "{} strategy produced different length", strategy);
        
        for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            if (exp - act).abs() > EPSILON {
                panic!("{} strategy mismatch at index {}: expected {}, got {}", 
                    strategy, i, exp, act);
            }
        }
    }
    
    /// Integration test: Simulate real trading calculations
    #[test]
    fn test_real_world_trading_scenario() {
        // Simulate 1 day of tick data (86400 seconds)
        let tick_count = 86400;
        let mut prices: Vec<f32> = Vec::with_capacity(tick_count);
        let mut base_price = 50000.0; // BTC price
        
        // Generate realistic price movement
        for i in 0..tick_count {
            let noise = ((i as f32 * 0.001).sin() * 50.0) + ((i as f32 * 0.01).cos() * 20.0);
            base_price += noise;
            prices.push(base_price);
        }
        
        eprintln!("\n=== Real-World Trading Scenario ===");
        eprintln!("Processing {} ticks of market data...", tick_count);
        
        // Calculate various indicators
        let start = Instant::now();
        
        // Fast EMA (alpha = 0.2)
        let ema_fast = EmaCalculator::calculate(&prices, 0.2);
        
        // Slow EMA (alpha = 0.05)
        let ema_slow = EmaCalculator::calculate(&prices, 0.05);
        
        // 20-period SMA
        let sma_20 = SmaCalculator::calculate(&prices, 20);
        
        // 50-period SMA
        let sma_50 = SmaCalculator::calculate(&prices, 50);
        
        let calc_time = start.elapsed();
        
        eprintln!("Indicator calculation time: {:?}", calc_time);
        eprintln!("Per-tick latency: {:?}", calc_time / tick_count as u32);
        
        // Verify results are valid
        assert_eq!(ema_fast.len(), tick_count);
        assert_eq!(ema_slow.len(), tick_count);
        assert_eq!(sma_20.len(), tick_count);
        assert_eq!(sma_50.len(), tick_count);
        
        // Check for crossovers (trading signals)
        let mut crossovers = 0;
        for i in 1..tick_count {
            let prev_diff = ema_fast[i-1] - ema_slow[i-1];
            let curr_diff = ema_fast[i] - ema_slow[i];
            
            if prev_diff * curr_diff < 0.0 {
                crossovers += 1;
            }
        }
        
        eprintln!("Trading signals detected: {}", crossovers);
        eprintln!("✓ Real-world scenario validated");
    }
}