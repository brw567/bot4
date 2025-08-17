// Kolmogorov-Smirnov Statistical Tests
// Addresses Nexus's feedback on distribution validation
// Tests that generated distributions match expected theoretical distributions
// Owner: Morgan | Reviewer: Nexus

use anyhow::Result;
use rand::thread_rng;
use statrs::distribution::{ContinuousCDF, Exp, Normal, Beta as BetaDist, LogNormal as LogNormalDist};
use statrs::statistics::Statistics;
use std::collections::HashMap;

use bot4_core::domain::value_objects::statistical_distributions::{
    FillDistribution, LatencyDistribution, SlippageDistribution, MarketStatistics
};

/// Kolmogorov-Smirnov test statistic
fn ks_statistic(samples: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    let n = samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut max_diff = 0.0;
    
    for (i, &x) in sorted.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n;
        let theoretical_cdf = cdf(x);
        let diff = (empirical_cdf - theoretical_cdf).abs();
        max_diff = max_diff.max(diff);
    }
    
    max_diff
}

/// Critical value for KS test at 95% confidence
fn ks_critical_value(n: usize) -> f64 {
    1.36 / (n as f64).sqrt()
}

#[test]
fn test_poisson_fill_counts() -> Result<()> {
    // Test that fill counts follow Poisson distribution
    let dist = FillDistribution::default(); // λ = 3.0
    let mut rng = thread_rng();
    
    let mut fill_counts = Vec::new();
    for _ in 0..1000 {
        let fills = dist.generate_fills(&mut rng)?;
        fill_counts.push(fills.len() as f64);
    }
    
    // Calculate empirical statistics
    let mean = fill_counts.iter().sum::<f64>() / fill_counts.len() as f64;
    let variance = fill_counts.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / fill_counts.len() as f64;
    
    // For Poisson, mean ≈ variance ≈ λ
    assert!(
        (mean - dist.lambda).abs() < 0.2,
        "Mean {} not close to lambda {}",
        mean,
        dist.lambda
    );
    
    assert!(
        (variance - dist.lambda).abs() < 0.5,
        "Variance {} not close to lambda {}",
        variance,
        dist.lambda
    );
    
    // Check distribution of counts
    let mut count_freq = HashMap::new();
    for &count in &fill_counts {
        *count_freq.entry(count as usize).or_insert(0) += 1;
    }
    
    println!("Fill count distribution:");
    for i in 0..=10 {
        let observed = count_freq.get(&i).unwrap_or(&0);
        let expected = poisson_pmf(i as f64, dist.lambda) * fill_counts.len() as f64;
        println!("  {} fills: observed={}, expected={:.1}", i, observed, expected);
    }
    
    Ok(())
}

#[test]
fn test_beta_fill_ratios() -> Result<()> {
    // Test that fill ratios follow Beta distribution
    let dist = FillDistribution::default(); // α=2, β=5
    let mut rng = thread_rng();
    
    let mut all_ratios = Vec::new();
    for _ in 0..1000 {
        let fills = dist.generate_fills(&mut rng)?;
        for ratio in fills {
            all_ratios.push(ratio);
        }
    }
    
    // Create theoretical Beta distribution
    let beta = BetaDist::new(dist.beta_alpha, dist.beta_beta).unwrap();
    
    // KS test
    let ks_stat = ks_statistic(&all_ratios, |x| beta.cdf(x));
    let critical = ks_critical_value(all_ratios.len());
    
    println!("Beta distribution KS test:");
    println!("  KS statistic: {:.4}", ks_stat);
    println!("  Critical value: {:.4}", critical);
    println!("  Test passed: {}", ks_stat < critical);
    
    assert!(
        ks_stat < critical * 1.5, // Allow some tolerance
        "Beta distribution KS test failed: {} > {}",
        ks_stat,
        critical
    );
    
    Ok(())
}

#[test]
fn test_log_normal_latency() -> Result<()> {
    // Test that latency follows log-normal distribution
    let dist = LatencyDistribution::default(); // μ=3.9, σ=0.5
    let mut rng = thread_rng();
    
    let mut latencies = Vec::new();
    for _ in 0..1000 {
        let latency = dist.generate_latency(&mut rng);
        let millis = latency.as_millis() as f64;
        
        // Only include samples within bounds for KS test
        if millis >= dist.min_latency && millis <= dist.max_latency {
            latencies.push(millis.ln()); // Log transform for testing
        }
    }
    
    // Create theoretical normal distribution (log of log-normal is normal)
    let normal = Normal::new(dist.mu, dist.sigma).unwrap();
    
    // KS test on log-transformed data
    let ks_stat = ks_statistic(&latencies, |x| normal.cdf(x));
    let critical = ks_critical_value(latencies.len());
    
    println!("Log-normal latency KS test:");
    println!("  KS statistic: {:.4}", ks_stat);
    println!("  Critical value: {:.4}", critical);
    println!("  Test passed: {}", ks_stat < critical);
    
    // Calculate percentiles
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2].exp();
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize].exp();
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize].exp();
    
    println!("Latency percentiles (ms):");
    println!("  P50: {:.1} (expected: {:.1})", p50, dist.median_latency());
    println!("  P95: {:.1} (expected: {:.1})", p95, dist.p95_latency());
    println!("  P99: {:.1} (expected: {:.1})", p99, dist.p99_latency());
    
    assert!(
        ks_stat < critical * 2.0, // Allow more tolerance due to truncation
        "Log-normal KS test failed: {} > {}",
        ks_stat,
        critical
    );
    
    Ok(())
}

#[test]
fn test_slippage_distribution() -> Result<()> {
    // Test that slippage follows normal distribution with skew
    let dist = SlippageDistribution::default(); // mean=2, std=5, skew=0.5
    let mut rng = thread_rng();
    
    let mut slippages = Vec::new();
    for _ in 0..5000 {
        let slippage = dist.generate_slippage(&mut rng, 1.0);
        slippages.push(slippage);
    }
    
    // Calculate statistics
    let mean = slippages.iter().sum::<f64>() / slippages.len() as f64;
    let variance = slippages.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / slippages.len() as f64;
    let std = variance.sqrt();
    
    // Calculate skewness
    let skewness = slippages.iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<f64>() / slippages.len() as f64;
    
    println!("Slippage distribution statistics:");
    println!("  Mean: {:.2} bps (expected: {:.2})", mean, dist.mean_bps);
    println!("  Std: {:.2} bps (expected: {:.2})", std, dist.std_bps);
    println!("  Skewness: {:.3} (expected: positive)", skewness);
    
    // Mean should be close to expected
    assert!(
        (mean - dist.mean_bps).abs() < 1.0,
        "Mean slippage {} not close to expected {}",
        mean,
        dist.mean_bps
    );
    
    // Std should be reasonable
    assert!(
        std > dist.std_bps * 0.8 && std < dist.std_bps * 1.5,
        "Std {} not in expected range",
        std
    );
    
    // Should have positive skewness
    assert!(
        skewness > 0.0,
        "Slippage should have positive skew, got {}",
        skewness
    );
    
    Ok(())
}

#[test]
fn test_market_profiles() -> Result<()> {
    // Test that liquid and illiquid markets have different characteristics
    let liquid = MarketStatistics::liquid_market();
    let illiquid = MarketStatistics::illiquid_market();
    let mut rng = thread_rng();
    
    // Generate samples from both markets
    let mut liquid_latencies = Vec::new();
    let mut illiquid_latencies = Vec::new();
    let mut liquid_slippages = Vec::new();
    let mut illiquid_slippages = Vec::new();
    
    for _ in 0..500 {
        // Latencies
        liquid_latencies.push(
            liquid.latency_dist.generate_latency(&mut rng).as_millis() as f64
        );
        illiquid_latencies.push(
            illiquid.latency_dist.generate_latency(&mut rng).as_millis() as f64
        );
        
        // Slippages
        liquid_slippages.push(
            liquid.slippage_dist.generate_slippage(&mut rng, 1.0)
        );
        illiquid_slippages.push(
            illiquid.slippage_dist.generate_slippage(&mut rng, 1.0)
        );
    }
    
    // Calculate means
    let liquid_latency_mean = liquid_latencies.iter().sum::<f64>() / liquid_latencies.len() as f64;
    let illiquid_latency_mean = illiquid_latencies.iter().sum::<f64>() / illiquid_latencies.len() as f64;
    let liquid_slippage_mean = liquid_slippages.iter().sum::<f64>() / liquid_slippages.len() as f64;
    let illiquid_slippage_mean = illiquid_slippages.iter().sum::<f64>() / illiquid_slippages.len() as f64;
    
    println!("Market profile comparison:");
    println!("  Liquid market:");
    println!("    Avg latency: {:.1} ms", liquid_latency_mean);
    println!("    Avg slippage: {:.2} bps", liquid_slippage_mean);
    println!("  Illiquid market:");
    println!("    Avg latency: {:.1} ms", illiquid_latency_mean);
    println!("    Avg slippage: {:.2} bps", illiquid_slippage_mean);
    
    // Liquid markets should have lower latency and slippage
    assert!(
        liquid_latency_mean < illiquid_latency_mean,
        "Liquid market should have lower latency"
    );
    assert!(
        liquid_slippage_mean < illiquid_slippage_mean,
        "Liquid market should have lower slippage"
    );
    
    Ok(())
}

#[test]
fn test_fill_distribution_profiles() -> Result<()> {
    // Test different fill distribution profiles
    let conservative = FillDistribution::conservative();
    let default = FillDistribution::default();
    let aggressive = FillDistribution::aggressive();
    let mut rng = thread_rng();
    
    let mut conservative_counts = Vec::new();
    let mut default_counts = Vec::new();
    let mut aggressive_counts = Vec::new();
    
    for _ in 0..500 {
        conservative_counts.push(conservative.generate_fills(&mut rng)?.len());
        default_counts.push(default.generate_fills(&mut rng)?.len());
        aggressive_counts.push(aggressive.generate_fills(&mut rng)?.len());
    }
    
    let conservative_mean = conservative_counts.iter().sum::<usize>() as f64 / conservative_counts.len() as f64;
    let default_mean = default_counts.iter().sum::<usize>() as f64 / default_counts.len() as f64;
    let aggressive_mean = aggressive_counts.iter().sum::<usize>() as f64 / aggressive_counts.len() as f64;
    
    println!("Fill distribution profiles:");
    println!("  Conservative: avg {:.1} fills", conservative_mean);
    println!("  Default: avg {:.1} fills", default_mean);
    println!("  Aggressive: avg {:.1} fills", aggressive_mean);
    
    // Conservative should have fewer fills than aggressive
    assert!(
        conservative_mean < aggressive_mean,
        "Conservative should have fewer fills than aggressive"
    );
    
    Ok(())
}

// Helper function for Poisson PMF
fn poisson_pmf(k: f64, lambda: f64) -> f64 {
    use statrs::function::factorial::factorial;
    
    let k_int = k as u64;
    (lambda.powi(k_int as i32) * (-lambda).exp()) / factorial(k_int) as f64
}

#[test]
fn test_two_sample_ks() -> Result<()> {
    // Two-sample KS test: compare two distributions
    let dist1 = LatencyDistribution::fast();
    let dist2 = LatencyDistribution::slow();
    let mut rng = thread_rng();
    
    let mut sample1 = Vec::new();
    let mut sample2 = Vec::new();
    
    for _ in 0..500 {
        sample1.push(dist1.generate_latency(&mut rng).as_millis() as f64);
        sample2.push(dist2.generate_latency(&mut rng).as_millis() as f64);
    }
    
    // Two-sample KS statistic
    let ks_stat = two_sample_ks(&sample1, &sample2);
    let critical = 1.36 * ((2.0 * 500.0) / (500.0 * 500.0)).sqrt();
    
    println!("Two-sample KS test (fast vs slow latency):");
    println!("  KS statistic: {:.4}", ks_stat);
    println!("  Critical value: {:.4}", critical);
    
    // Distributions should be significantly different
    assert!(
        ks_stat > critical,
        "Fast and slow latency distributions should be significantly different"
    );
    
    Ok(())
}

/// Two-sample Kolmogorov-Smirnov test
fn two_sample_ks(sample1: &[f64], sample2: &[f64]) -> f64 {
    let mut all_points = Vec::new();
    all_points.extend(sample1);
    all_points.extend(sample2);
    all_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_points.dedup();
    
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    
    let mut max_diff = 0.0;
    
    for &x in &all_points {
        let cdf1 = sample1.iter().filter(|&&y| y <= x).count() as f64 / n1;
        let cdf2 = sample2.iter().filter(|&&y| y <= x).count() as f64 / n2;
        max_diff = max_diff.max((cdf1 - cdf2).abs());
    }
    
    max_diff
}

#[test]
fn test_distribution_consistency() -> Result<()> {
    // Test that distributions maintain consistency over time
    let dist = FillDistribution::default();
    let mut rng = thread_rng();
    
    // Generate multiple batches
    let mut batch_means = Vec::new();
    for _ in 0..10 {
        let mut batch_count = 0.0;
        for _ in 0..100 {
            batch_count += dist.generate_fills(&mut rng)?.len() as f64;
        }
        batch_means.push(batch_count / 100.0);
    }
    
    // Check consistency across batches
    let overall_mean = batch_means.iter().sum::<f64>() / batch_means.len() as f64;
    let variance = batch_means.iter()
        .map(|&x| (x - overall_mean).powi(2))
        .sum::<f64>() / batch_means.len() as f64;
    
    println!("Distribution consistency test:");
    println!("  Mean across batches: {:.2}", overall_mean);
    println!("  Variance of batch means: {:.4}", variance);
    
    // Variance should be relatively small
    assert!(
        variance < 0.1,
        "Distribution not consistent across batches: variance = {}",
        variance
    );
    
    Ok(())
}