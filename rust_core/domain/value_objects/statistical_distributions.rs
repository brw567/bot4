// Value Object: Statistical Distributions
// Realistic probability distributions for market simulation
// Addresses Nexus's feedback on uniform distribution limitations
// Owner: Morgan | Reviewer: Nexus

use anyhow::{Result, bail};
use rand::Rng;
use rand_distr::{Distribution, Poisson, Beta, LogNormal, Normal, Exp};
use std::time::Duration;

/// Fill distribution parameters for realistic order execution
#[derive(Debug, Clone)]
pub struct FillDistribution {
    /// Average number of fills per order (Poisson λ)
    pub lambda: f64,
    /// Beta distribution alpha (shape parameter)
    pub beta_alpha: f64,
    /// Beta distribution beta (shape parameter)
    pub beta_beta: f64,
}

impl Default for FillDistribution {
    fn default() -> Self {
        Self {
            lambda: 3.0,      // Average 3 fills per order
            beta_alpha: 2.0,  // Skewed towards smaller fills
            beta_beta: 5.0,   // Most fills are small, few are large
        }
    }
}

impl FillDistribution {
    /// Conservative distribution (fewer, more even fills)
    pub fn conservative() -> Self {
        Self {
            lambda: 2.0,
            beta_alpha: 3.0,
            beta_beta: 3.0,  // More symmetric
        }
    }
    
    /// Aggressive distribution (many small fills)
    pub fn aggressive() -> Self {
        Self {
            lambda: 5.0,
            beta_alpha: 1.0,
            beta_beta: 8.0,  // Heavily skewed to small fills
        }
    }
    
    /// Generate realistic fill counts and ratios
    pub fn generate_fills<R: Rng>(&self, rng: &mut R) -> Result<Vec<f64>> {
        // Number of fills follows Poisson distribution
        let poisson = Poisson::new(self.lambda)
            .map_err(|e| anyhow::anyhow!("Invalid Poisson lambda: {}", e))?;
        
        let num_fills = poisson.sample(rng).max(1) as usize;
        
        // Fill ratios follow Beta distribution
        let beta = Beta::new(self.beta_alpha, self.beta_beta)
            .map_err(|e| anyhow::anyhow!("Invalid Beta parameters: {}", e))?;
        
        // Generate raw ratios
        let mut ratios: Vec<f64> = (0..num_fills)
            .map(|_| beta.sample(rng))
            .collect();
        
        // Normalize to sum to 1.0
        let sum: f64 = ratios.iter().sum();
        if sum > 0.0 {
            ratios.iter_mut().for_each(|r| *r /= sum);
        } else {
            // Fallback to equal distribution
            let equal = 1.0 / num_fills as f64;
            ratios.iter_mut().for_each(|r| *r = equal);
        }
        
        Ok(ratios)
    }
    
    /// Calculate expected number of fills
    pub fn expected_fills(&self) -> f64 {
        self.lambda
    }
    
    /// Calculate expected smallest fill ratio
    pub fn expected_min_ratio(&self) -> f64 {
        // For Beta distribution, mode = (α-1)/(α+β-2)
        if self.beta_alpha > 1.0 && self.beta_beta > 1.0 {
            (self.beta_alpha - 1.0) / (self.beta_alpha + self.beta_beta - 2.0)
        } else {
            0.1 // Fallback
        }
    }
}

/// Latency distribution for network delays
#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    /// Log-normal μ parameter (log of median)
    pub mu: f64,
    /// Log-normal σ parameter (shape)
    pub sigma: f64,
    /// Minimum latency floor (milliseconds)
    pub min_latency: f64,
    /// Maximum latency cap (milliseconds)
    pub max_latency: f64,
}

impl Default for LatencyDistribution {
    fn default() -> Self {
        Self {
            mu: 3.9,           // ln(50) ≈ 3.9 (50ms median)
            sigma: 0.5,        // Moderate variability
            min_latency: 5.0,  // 5ms minimum
            max_latency: 500.0, // 500ms maximum
        }
    }
}

impl LatencyDistribution {
    /// Fast exchange configuration (e.g., colocation)
    pub fn fast() -> Self {
        Self {
            mu: 2.3,           // ln(10) ≈ 2.3 (10ms median)
            sigma: 0.3,        // Low variability
            min_latency: 1.0,
            max_latency: 100.0,
        }
    }
    
    /// Slow/congested network configuration
    pub fn slow() -> Self {
        Self {
            mu: 4.6,           // ln(100) ≈ 4.6 (100ms median)
            sigma: 0.7,        // High variability
            min_latency: 20.0,
            max_latency: 2000.0,
        }
    }
    
    /// Generate realistic latency in milliseconds
    pub fn generate_latency<R: Rng>(&self, rng: &mut R) -> Duration {
        let log_normal = LogNormal::new(self.mu, self.sigma)
            .unwrap_or_else(|_| LogNormal::new(3.9, 0.5).unwrap());
        
        let mut latency = log_normal.sample(rng);
        
        // Apply floor and cap
        latency = latency.max(self.min_latency).min(self.max_latency);
        
        Duration::from_millis(latency as u64)
    }
    
    /// Get median latency
    pub fn median_latency(&self) -> f64 {
        self.mu.exp()
    }
    
    /// Get 95th percentile latency
    pub fn p95_latency(&self) -> f64 {
        // For log-normal: P95 ≈ exp(μ + 1.645σ)
        (self.mu + 1.645 * self.sigma).exp()
    }
    
    /// Get 99th percentile latency
    pub fn p99_latency(&self) -> f64 {
        // For log-normal: P99 ≈ exp(μ + 2.326σ)
        (self.mu + 2.326 * self.sigma).exp()
    }
}

/// Slippage distribution for market impact
#[derive(Debug, Clone)]
pub struct SlippageDistribution {
    /// Mean slippage in basis points
    pub mean_bps: f64,
    /// Standard deviation in basis points
    pub std_bps: f64,
    /// Skewness factor (positive = worse fills more likely)
    pub skew: f64,
}

impl Default for SlippageDistribution {
    fn default() -> Self {
        Self {
            mean_bps: 2.0,  // 2 bps average slippage
            std_bps: 5.0,   // 5 bps standard deviation
            skew: 0.5,      // Slight positive skew
        }
    }
}

impl SlippageDistribution {
    /// Generate slippage in basis points
    pub fn generate_slippage<R: Rng>(&self, rng: &mut R, order_size: f64) -> f64 {
        // Base slippage from normal distribution
        let normal = Normal::new(self.mean_bps, self.std_bps)
            .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
        
        let mut slippage = normal.sample(rng);
        
        // Apply skewness (simple approach)
        if self.skew != 0.0 && rng.gen::<f64>() < self.skew.abs() {
            if self.skew > 0.0 {
                // Positive skew: occasionally worse fills
                slippage += self.std_bps * rng.gen::<f64>();
            } else {
                // Negative skew: occasionally better fills
                slippage -= self.std_bps * rng.gen::<f64>();
            }
        }
        
        // Scale by order size (larger orders have more slippage)
        let size_factor = (1.0 + order_size).ln() / 10.0;
        slippage * (1.0 + size_factor)
    }
}

/// Order arrival rate distribution (for market simulation)
#[derive(Debug, Clone)]
pub struct ArrivalRateDistribution {
    /// Base arrival rate (orders per second)
    pub base_rate: f64,
    /// Peak rate multiplier
    pub peak_multiplier: f64,
    /// Off-peak rate multiplier
    pub off_peak_multiplier: f64,
}

impl Default for ArrivalRateDistribution {
    fn default() -> Self {
        Self {
            base_rate: 10.0,           // 10 orders/sec baseline
            peak_multiplier: 3.0,      // 30 orders/sec during peaks
            off_peak_multiplier: 0.2,  // 2 orders/sec off-peak
        }
    }
}

impl ArrivalRateDistribution {
    /// Generate inter-arrival time (exponential distribution)
    pub fn generate_inter_arrival<R: Rng>(&self, rng: &mut R, is_peak: bool) -> Duration {
        let rate = if is_peak {
            self.base_rate * self.peak_multiplier
        } else {
            self.base_rate * self.off_peak_multiplier
        };
        
        let exp = Exp::new(rate).unwrap_or_else(|_| Exp::new(1.0).unwrap());
        let seconds = exp.sample(rng);
        
        Duration::from_secs_f64(seconds)
    }
}

/// Comprehensive market statistics generator
pub struct MarketStatistics {
    pub fill_dist: FillDistribution,
    pub latency_dist: LatencyDistribution,
    pub slippage_dist: SlippageDistribution,
    pub arrival_dist: ArrivalRateDistribution,
}

impl Default for MarketStatistics {
    fn default() -> Self {
        Self {
            fill_dist: FillDistribution::default(),
            latency_dist: LatencyDistribution::default(),
            slippage_dist: SlippageDistribution::default(),
            arrival_dist: ArrivalRateDistribution::default(),
        }
    }
}

impl MarketStatistics {
    /// Create statistics for a fast, liquid market
    pub fn liquid_market() -> Self {
        Self {
            fill_dist: FillDistribution::conservative(),
            latency_dist: LatencyDistribution::fast(),
            slippage_dist: SlippageDistribution {
                mean_bps: 1.0,
                std_bps: 2.0,
                skew: 0.1,
            },
            arrival_dist: ArrivalRateDistribution {
                base_rate: 50.0,
                peak_multiplier: 5.0,
                off_peak_multiplier: 0.5,
            },
        }
    }
    
    /// Create statistics for a slow, illiquid market
    pub fn illiquid_market() -> Self {
        Self {
            fill_dist: FillDistribution::aggressive(),
            latency_dist: LatencyDistribution::slow(),
            slippage_dist: SlippageDistribution {
                mean_bps: 5.0,
                std_bps: 10.0,
                skew: 1.0,
            },
            arrival_dist: ArrivalRateDistribution {
                base_rate: 2.0,
                peak_multiplier: 2.0,
                off_peak_multiplier: 0.1,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_fill_distribution() {
        let dist = FillDistribution::default();
        let mut rng = thread_rng();
        
        // Generate fills multiple times
        for _ in 0..10 {
            let fills = dist.generate_fills(&mut rng).unwrap();
            
            // Should have at least 1 fill
            assert!(!fills.is_empty());
            
            // Ratios should sum to 1.0 (within floating point tolerance)
            let sum: f64 = fills.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
            
            // All ratios should be positive
            assert!(fills.iter().all(|&r| r > 0.0));
        }
    }
    
    #[test]
    fn test_latency_distribution() {
        let dist = LatencyDistribution::default();
        let mut rng = thread_rng();
        
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let latency = dist.generate_latency(&mut rng);
            latencies.push(latency.as_millis() as f64);
        }
        
        // Check bounds
        assert!(latencies.iter().all(|&l| l >= dist.min_latency));
        assert!(latencies.iter().all(|&l| l <= dist.max_latency));
        
        // Median should be roughly around exp(mu)
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = latencies[50];
        let expected_median = dist.median_latency();
        
        // Allow for sampling variance
        assert!(median > expected_median * 0.5);
        assert!(median < expected_median * 2.0);
    }
    
    #[test]
    fn test_slippage_distribution() {
        let dist = SlippageDistribution::default();
        let mut rng = thread_rng();
        
        let mut slippages = Vec::new();
        for _ in 0..100 {
            let slippage = dist.generate_slippage(&mut rng, 1.0);
            slippages.push(slippage);
        }
        
        // Calculate mean
        let mean: f64 = slippages.iter().sum::<f64>() / slippages.len() as f64;
        
        // Mean should be roughly around configured mean (with variance)
        assert!(mean > dist.mean_bps - dist.std_bps);
        assert!(mean < dist.mean_bps + dist.std_bps);
    }
    
    #[test]
    fn test_arrival_rate() {
        let dist = ArrivalRateDistribution::default();
        let mut rng = thread_rng();
        
        // Peak time arrivals
        let peak_interval = dist.generate_inter_arrival(&mut rng, true);
        assert!(peak_interval.as_secs_f64() < 1.0); // Should be fast
        
        // Off-peak arrivals
        let off_peak_interval = dist.generate_inter_arrival(&mut rng, false);
        assert!(off_peak_interval.as_secs_f64() > 0.1); // Should be slower
    }
    
    #[test]
    fn test_market_statistics_profiles() {
        // Liquid market should have lower slippage
        let liquid = MarketStatistics::liquid_market();
        assert!(liquid.slippage_dist.mean_bps < 2.0);
        
        // Illiquid market should have higher slippage
        let illiquid = MarketStatistics::illiquid_market();
        assert!(illiquid.slippage_dist.mean_bps > 3.0);
        
        // Liquid market should have faster latency
        assert!(liquid.latency_dist.median_latency() < illiquid.latency_dist.median_latency());
    }
}