// Purged Walk-Forward Cross-Validation (López de Prado)
// Morgan (ML Lead) + Riley (Testing)
// CRITICAL: Prevents temporal leakage (Sophia #2)
// Reference: "Advances in Financial Machine Learning" - López de Prado

use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashSet;

/// Purged Walk-Forward Cross-Validation
/// Prevents temporal leakage in time series data
/// 
/// # Key Concepts:
/// 1. **Purge**: Remove training samples too close to test set
/// 2. **Embargo**: Remove test samples that follow training
/// 3. **Walk-Forward**: Sequential splits respecting time order
#[derive(Debug, Clone)]
pub struct PurgedWalkForwardCV {
    n_splits: usize,        // Number of CV folds
    purge_gap: usize,       // Bars to purge between train/test
    embargo_pct: f32,       // Percentage of training data to embargo after test
    min_train_size: usize,  // Minimum training samples
    min_test_size: usize,   // Minimum test samples
}

impl PurgedWalkForwardCV {
    pub fn new(n_splits: usize, purge_gap: usize, embargo_pct: f32) -> Self {
        assert!(n_splits > 1, "Need at least 2 splits");
        assert!(embargo_pct >= 0.0 && embargo_pct < 1.0, "Embargo must be in [0, 1)");
        
        Self {
            n_splits,
            purge_gap,
            embargo_pct,
            min_train_size: 500,  // Minimum for reliable ML
            min_test_size: 100,   // Minimum for evaluation
        }
    }
    
    /// Generate train/test splits with purging and embargo
    /// Riley: "This is critical for preventing look-ahead bias!"
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples < self.min_train_size + self.min_test_size + self.purge_gap {
            panic!("Insufficient samples for purged CV: need at least {}", 
                   self.min_train_size + self.min_test_size + self.purge_gap);
        }
        
        let mut splits = Vec::new();
        let test_size = n_samples / (self.n_splits + 1);
        
        for i in 0..self.n_splits {
            // Calculate split boundaries
            let train_end = (i + 1) * test_size;
            let test_start = train_end + self.purge_gap;
            let test_end = (test_start + test_size).min(n_samples);
            
            // Calculate embargo size
            let embargo_size = (train_end as f32 * self.embargo_pct) as usize;
            
            // Check if we have enough data
            if test_end + embargo_size > n_samples {
                break;  // Not enough data for this split
            }
            
            // Create purged training indices
            let train_indices: Vec<usize> = (0..train_end)
                .filter(|&idx| {
                    // Remove indices too close to test set (purge)
                    idx < train_end - self.purge_gap
                })
                .collect();
            
            // Create test indices (embargo applied after test)
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            // Validate split sizes
            if train_indices.len() >= self.min_train_size && 
               test_indices.len() >= self.min_test_size {
                splits.push((train_indices, test_indices));
                
                info!(
                    "Split {}: train={} samples, test={} samples, purge={}, embargo={}",
                    i + 1,
                    train_indices.len(),
                    test_indices.len(),
                    self.purge_gap,
                    embargo_size
                );
            }
        }
        
        if splits.is_empty() {
            panic!("No valid splits could be generated with current parameters");
        }
        
        splits
    }
    
    /// Combinatorial Purged CV (for feature importance)
    /// Generates multiple random splits while respecting temporal constraints
    pub fn combinatorial_split(
        &self,
        n_samples: usize,
        n_combinations: usize
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let mut rng = thread_rng();
        
        for _ in 0..n_combinations {
            // Random test period
            let test_size = n_samples / (self.n_splits + 1);
            let max_start = n_samples - test_size - self.purge_gap;
            let test_start = (rand::random::<usize>() % max_start) + self.purge_gap;
            let test_end = (test_start + test_size).min(n_samples);
            
            // Training indices (before and after test, with purge/embargo)
            let mut train_indices = Vec::new();
            
            // Before test period (with purge)
            for i in 0..test_start - self.purge_gap {
                train_indices.push(i);
            }
            
            // After test period (with embargo)
            let embargo_start = test_end + (test_size as f32 * self.embargo_pct) as usize;
            if embargo_start < n_samples {
                for i in embargo_start..n_samples {
                    train_indices.push(i);
                }
            }
            
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            if train_indices.len() >= self.min_train_size && 
               test_indices.len() >= self.min_test_size {
                splits.push((train_indices, test_indices));
            }
        }
        
        splits
    }
}

/// Leakage Detection Tests
/// CRITICAL: Ensures no temporal leakage in features or labels
pub struct LeakageSentinel {
    significance_level: f64,
    min_samples: usize,
}

impl LeakageSentinel {
    pub fn new() -> Self {
        Self {
            significance_level: 0.05,  // 5% significance
            min_samples: 1000,
        }
    }
    
    /// Test for temporal leakage by training on shuffled labels
    /// If model achieves significant performance, there's leakage!
    pub fn test_for_leakage(
        &self,
        features: &Array2<f32>,
        labels: &Array1<f32>,
        model: &mut dyn MLModel,
    ) -> LeakageTestResult {
        let n_samples = features.nrows();
        assert!(n_samples >= self.min_samples, "Need at least {} samples", self.min_samples);
        
        // Shuffle labels to break any real relationship
        let mut shuffled_labels = labels.to_vec();
        shuffled_labels.shuffle(&mut thread_rng());
        let shuffled_labels = Array1::from(shuffled_labels);
        
        // Train model on shuffled data
        model.fit(features, &shuffled_labels).unwrap();
        
        // Make predictions
        let predictions = model.predict(features).unwrap();
        
        // Calculate Sharpe ratio (should be near 0 if no leakage)
        let sharpe = self.calculate_sharpe(&predictions, &shuffled_labels);
        
        // Calculate other metrics
        let correlation = self.calculate_correlation(&predictions, &shuffled_labels);
        let accuracy = self.calculate_accuracy(&predictions, &shuffled_labels);
        
        // Statistical test for significance
        let is_significant = self.test_significance(sharpe, n_samples);
        
        let has_leakage = sharpe.abs() > 0.1 || correlation.abs() > 0.1 || accuracy > 0.55;
        
        if has_leakage {
            error!(
                "⚠️ LEAKAGE DETECTED! Sharpe={:.3}, Correlation={:.3}, Accuracy={:.3}",
                sharpe, correlation, accuracy
            );
        } else {
            info!(
                "✅ No leakage detected. Sharpe={:.3}, Correlation={:.3}, Accuracy={:.3}",
                sharpe, correlation, accuracy
            );
        }
        
        LeakageTestResult {
            has_leakage,
            sharpe_ratio: sharpe,
            correlation,
            accuracy,
            p_value: is_significant.1,
            samples_tested: n_samples,
        }
    }
    
    /// Test for specific feature leakage
    pub fn test_feature_leakage(
        &self,
        feature: &Array1<f32>,
        future_returns: &Array1<f32>,
        lag: usize,
    ) -> bool {
        // Check if feature has information about future returns
        let n = feature.len().min(future_returns.len());
        
        if n < lag {
            return false;
        }
        
        // Calculate correlation between feature and future returns
        let mut correlations = Vec::new();
        
        for offset in 1..=lag {
            if offset >= n {
                break;
            }
            
            let feature_slice = feature.slice(s![..n-offset]);
            let returns_slice = future_returns.slice(s![offset..]);
            
            let corr = self.calculate_correlation_arrays(&feature_slice, &returns_slice);
            correlations.push(corr.abs());
        }
        
        // If any future correlation is high, there's leakage
        let max_corr = correlations.iter().fold(0.0f64, |a, &b| a.max(b));
        
        if max_corr > 0.1 {
            warn!(
                "Feature shows correlation {:.3} with future returns at lag {}",
                max_corr, lag
            );
            true
        } else {
            false
        }
    }
    
    /// Calculate Sharpe ratio
    fn calculate_sharpe(&self, predictions: &Array1<f32>, labels: &Array1<f32>) -> f64 {
        let returns: Vec<f64> = predictions.iter()
            .zip(labels.iter())
            .map(|(p, l)| (*p * *l) as f64)
            .collect();
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = (returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64)
            .sqrt();
        
        if std > 0.0 {
            mean / std * (252.0_f64).sqrt()  // Annualized
        } else {
            0.0
        }
    }
    
    /// Calculate correlation
    fn calculate_correlation(&self, predictions: &Array1<f32>, labels: &Array1<f32>) -> f64 {
        self.calculate_correlation_arrays(predictions, labels)
    }
    
    fn calculate_correlation_arrays(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.mean().unwrap_or(0.0) as f64;
        let mean_b = b.mean().unwrap_or(0.0) as f64;
        
        let cov: f64 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64 - mean_a) * (*y as f64 - mean_b))
            .sum::<f64>() / n;
        
        let std_a = (a.iter()
            .map(|x| (*x as f64 - mean_a).powi(2))
            .sum::<f64>() / n)
            .sqrt();
        
        let std_b = (b.iter()
            .map(|y| (*y as f64 - mean_b).powi(2))
            .sum::<f64>() / n)
            .sqrt();
        
        if std_a > 0.0 && std_b > 0.0 {
            cov / (std_a * std_b)
        } else {
            0.0
        }
    }
    
    /// Calculate accuracy for classification
    fn calculate_accuracy(&self, predictions: &Array1<f32>, labels: &Array1<f32>) -> f64 {
        let correct = predictions.iter()
            .zip(labels.iter())
            .filter(|(p, l)| (p.signum() - l.signum()).abs() < 0.5)
            .count();
        
        correct as f64 / predictions.len() as f64
    }
    
    /// Statistical significance test
    fn test_significance(&self, sharpe: f64, n_samples: usize) -> (bool, f64) {
        // Under null hypothesis (no predictive power), Sharpe ~ N(0, 1/sqrt(n))
        let std_error = 1.0 / (n_samples as f64).sqrt();
        let z_score = sharpe / std_error;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));
        
        (p_value < self.significance_level, p_value)
    }
}

/// Result of leakage test
#[derive(Debug, Clone)]
pub struct LeakageTestResult {
    pub has_leakage: bool,
    pub sharpe_ratio: f64,
    pub correlation: f64,
    pub accuracy: f64,
    pub p_value: f64,
    pub samples_tested: usize,
}

/// Trait for ML models (simplified for testing)
pub trait MLModel: Send + Sync {
    fn fit(&mut self, features: &Array2<f32>, labels: &Array1<f32>) -> Result<(), String>;
    fn predict(&self, features: &Array2<f32>) -> Result<Array1<f32>, String>;
}

use ndarray::{s, ArrayView1};

/// Sample Weights for Time Decay
/// More recent samples get higher weight
pub struct TimeDecayWeights {
    decay_factor: f64,
}

impl TimeDecayWeights {
    pub fn new(decay_factor: f64) -> Self {
        assert!(decay_factor > 0.0 && decay_factor <= 1.0);
        Self { decay_factor }
    }
    
    pub fn calculate_weights(&self, n_samples: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let age = (n_samples - i - 1) as f64;
            let weight = self.decay_factor.powf(age);
            weights.push(weight);
        }
        
        // Normalize weights
        let sum: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sum);
        
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    // Mock model for testing
    struct MockModel {
        coefficients: Option<Vec<f32>>,
    }
    
    impl MLModel for MockModel {
        fn fit(&mut self, features: &Array2<f32>, _labels: &Array1<f32>) -> Result<(), String> {
            self.coefficients = Some(vec![0.1; features.ncols()]);
            Ok(())
        }
        
        fn predict(&self, features: &Array2<f32>) -> Result<Array1<f32>, String> {
            let coef = self.coefficients.as_ref().unwrap();
            let predictions = features.dot(&Array1::from(coef.clone()));
            Ok(predictions)
        }
    }
    
    #[test]
    fn test_purged_cv_splits() {
        let cv = PurgedWalkForwardCV::new(5, 100, 0.01);
        let splits = cv.split(2000);
        
        assert_eq!(splits.len(), 5);
        
        for (i, (train, test)) in splits.iter().enumerate() {
            // Check no overlap
            let train_set: HashSet<_> = train.iter().collect();
            let test_set: HashSet<_> = test.iter().collect();
            assert!(train_set.is_disjoint(&test_set));
            
            // Check purge gap
            let max_train = *train.iter().max().unwrap();
            let min_test = *test.iter().min().unwrap();
            assert!(min_test >= max_train + 100, "Purge gap not respected in split {}", i);
            
            // Check time ordering
            assert!(train.windows(2).all(|w| w[0] < w[1]));
            assert!(test.windows(2).all(|w| w[0] < w[1]));
        }
    }
    
    #[test]
    fn test_leakage_detection() {
        let sentinel = LeakageSentinel::new();
        
        // Create random features and labels (no relationship)
        let features = Array2::from_shape_fn((1000, 10), |_| rand::random::<f32>());
        let labels = Array1::from_shape_fn(1000, |_| rand::random::<f32>() - 0.5);
        
        let mut model = MockModel { coefficients: None };
        let result = sentinel.test_for_leakage(&features, &labels, &mut model);
        
        // Should detect no leakage with random data
        assert!(!result.has_leakage);
        assert!(result.sharpe_ratio.abs() < 0.2);
        assert!(result.correlation.abs() < 0.2);
    }
    
    #[test]
    fn test_feature_leakage() {
        let sentinel = LeakageSentinel::new();
        
        // Create a feature that "knows" future returns (leakage)
        let future_returns = Array1::from_shape_fn(1000, |i| (i as f32 * 0.01).sin());
        let leaky_feature = future_returns.clone();  // Perfect foresight!
        
        let has_leakage = sentinel.test_feature_leakage(&leaky_feature, &future_returns, 1);
        assert!(has_leakage, "Should detect leakage when feature equals future returns");
        
        // Test with random feature (no leakage)
        let random_feature = Array1::from_shape_fn(1000, |_| rand::random::<f32>());
        let no_leakage = sentinel.test_feature_leakage(&random_feature, &future_returns, 1);
        assert!(!no_leakage, "Should not detect leakage with random feature");
    }
    
    #[test]
    fn test_time_decay_weights() {
        let decay = TimeDecayWeights::new(0.9);
        let weights = decay.calculate_weights(10);
        
        // Check weights sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Check monotonic decrease (older samples have lower weight)
        assert!(weights.windows(2).all(|w| w[0] < w[1]));
    }
    
    #[test]
    fn test_combinatorial_splits() {
        let cv = PurgedWalkForwardCV::new(5, 50, 0.01);
        let splits = cv.combinatorial_split(2000, 10);
        
        assert_eq!(splits.len(), 10);
        
        for (train, test) in splits {
            // Check minimum sizes
            assert!(train.len() >= 500);
            assert!(test.len() >= 100);
            
            // Check no overlap
            let train_set: HashSet<_> = train.iter().collect();
            let test_set: HashSet<_> = test.iter().collect();
            assert!(train_set.is_disjoint(&test_set));
        }
    }
}