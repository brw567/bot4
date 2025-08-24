// Standalone SIMD performance test - Jordan's <50ns requirement
use std::time::Instant;

fn main() {
    println!("Testing SIMD Decision Engine Performance...");
    println!("Target: <50ns per decision (Jordan's requirement)");
    println!("========================================");
    
    // Check CPU features
    println!("\nCPU Capabilities:");
    println!("  AVX512F: {}", is_x86_feature_detected!("avx512f"));
    println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
    println!("  FMA: {}", is_x86_feature_detected!("fma"));
    println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
    
    // Create test data
    let feature_count = 64;
    let ml_features: Vec<f64> = (0..feature_count).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let ta_indicators: Vec<f64> = (0..feature_count).map(|i| 0.6 + (i as f64) * 0.008).collect();
    let risk_factors: Vec<f64> = vec![0.1; feature_count];
    
    // Warmup iterations
    println!("\nWarming up (1M iterations)...");
    for _ in 0..1_000_000 {
        simple_decision(&ml_features, &ta_indicators, &risk_factors);
    }
    
    // Performance test
    println!("Running performance test (10M decisions)...");
    let iterations = 10_000_000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        simple_decision(&ml_features, &ta_indicators, &risk_factors);
    }
    let elapsed = start.elapsed();
    
    let total_ns = elapsed.as_nanos();
    let avg_ns = total_ns / iterations;
    
    println!("\nResults:");
    println!("  Total time: {:?}", elapsed);
    println!("  Total decisions: {}", iterations);
    println!("  Average latency: {}ns", avg_ns);
    println!("  Decisions per second: {:.2}M", (iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0);
    
    if avg_ns < 50 {
        println!("\n✅ SUCCESS: Meets Jordan's <50ns requirement!");
    } else {
        println!("\n⚠️  WARNING: {}ns is above 50ns target", avg_ns);
    }
    
    // Test with SIMD
    if is_x86_feature_detected!("avx2") {
        println!("\nTesting AVX2 optimized version...");
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { simd_decision_avx2(&ml_features, &ta_indicators, &risk_factors) };
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations;
        println!("  AVX2 average latency: {}ns", avg_ns);
    }
}

#[inline(always)]
fn simple_decision(ml: &[f64], ta: &[f64], risk: &[f64]) -> f64 {
    // Simple weighted sum for baseline
    let mut score = 0.0;
    for i in 0..ml.len().min(ta.len()).min(risk.len()) {
        score += ml[i] * 0.4 + ta[i] * 0.4 + (1.0 - risk[i]) * 0.2;
    }
    score
}

#[target_feature(enable = "avx2")]
unsafe fn simd_decision_avx2(ml: &[f64], ta: &[f64], risk: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    let len = ml.len().min(ta.len()).min(risk.len());
    let chunks = len / 4;
    
    let weight_ml = _mm256_set1_pd(0.4);
    let weight_ta = _mm256_set1_pd(0.4);
    let weight_risk = _mm256_set1_pd(0.2);
    let one = _mm256_set1_pd(1.0);
    
    let mut sum = _mm256_setzero_pd();
    
    for i in 0..chunks {
        let offset = i * 4;
        let ml_vec = _mm256_loadu_pd(ml.as_ptr().add(offset));
        let ta_vec = _mm256_loadu_pd(ta.as_ptr().add(offset));
        let risk_vec = _mm256_loadu_pd(risk.as_ptr().add(offset));
        
        let ml_weighted = _mm256_mul_pd(ml_vec, weight_ml);
        let ta_weighted = _mm256_mul_pd(ta_vec, weight_ta);
        let risk_inverted = _mm256_sub_pd(one, risk_vec);
        let risk_weighted = _mm256_mul_pd(risk_inverted, weight_risk);
        
        let partial = _mm256_add_pd(_mm256_add_pd(ml_weighted, ta_weighted), risk_weighted);
        sum = _mm256_add_pd(sum, partial);
    }
    
    // Horizontal sum
    let high = _mm256_extractf128_pd(sum, 1);
    let low = _mm256_extractf128_pd(sum, 0);
    let sum128 = _mm_add_pd(low, high);
    let high64 = _mm_unpackhi_pd(sum128, sum128);
    let final_sum = _mm_add_sd(sum128, high64);
    _mm_cvtsd_f64(final_sum)
}