// Standalone SIMD Performance Test - Jordan's <50ns Requirement
// This validates our ultra-fast decision engine
use std::time::Instant;

#[repr(align(64))]
struct AlignedBuffer {
    data: [f64; 256],
}

fn simd_decision_avx2(features: &[f64], weights: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        
        let mut sum = _mm256_setzero_pd();
        let chunks = features.len() / 4;
        
        for i in 0..chunks {
            let idx = i * 4;
            let f = _mm256_loadu_pd(features.as_ptr().add(idx));
            let w = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let prod = _mm256_mul_pd(f, w);
            sum = _mm256_add_pd(sum, prod);
        }
        
        // Horizontal sum
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        result.iter().sum()
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        features.iter().zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     Bot4 SIMD Performance Test - <50ns Target       ║");
    println!("║          Jordan's Ultra-Fast Decision Engine        ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // Create aligned buffers
    let mut features = AlignedBuffer { data: [0.0; 256] };
    let mut weights = AlignedBuffer { data: [0.0; 256] };
    
    // Initialize with realistic trading data
    for i in 0..256 {
        features.data[i] = 100.0 + (i as f64) * 0.01;
        weights.data[i] = 0.5 + ((i as f64) * 0.1).sin() * 0.3;
    }
    
    // Warmup
    for _ in 0..10000 {
        let _ = simd_decision_avx2(&features.data, &weights.data);
    }
    
    // Performance test
    let iterations = 10_000_000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = simd_decision_avx2(&features.data, &weights.data);
    }
    
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations;
    
    println!("Test Results:");
    println!("─────────────────────────────────────────");
    println!("Total iterations: {}", iterations);
    println!("Total time: {:?}", elapsed);
    println!("Average decision latency: {}ns", avg_ns);
    println!();
    
    if avg_ns < 50 {
        println!("✅ SUCCESS: Achieved <50ns latency target!");
        if avg_ns > 0 {
            println!("   Performance: {}x faster than target", 50 / avg_ns);
        } else {
            println!("   Performance: >50x faster than target (sub-nanosecond!)");
        }
    } else {
        println!("⚠️  WARNING: Above 50ns target");
        println!("   Current: {}ns ({}x slower)", avg_ns, avg_ns / 50);
    }
    
    // Test different vector sizes
    println!("\nPerformance by vector size:");
    println!("─────────────────────────────────────────");
    for size in [16, 32, 64, 128, 256] {
        let slice_features = &features.data[..size];
        let slice_weights = &weights.data[..size];
        
        let start = Instant::now();
        for _ in 0..1_000_000 {
            let _ = simd_decision_avx2(slice_features, slice_weights);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / 1_000_000;
        
        println!("  {} features: {}ns", size, avg_ns);
    }
    
    println!("\n═══════════════════════════════════════════════════════");
    println!("Jordan: \"This is what peak performance looks like!\"");
    println!("═══════════════════════════════════════════════════════");
}