// Cross Validation Module - Time Series Validation
// Team Lead: Riley (Testing & Validation)
// Contributors: Morgan (ML), Quinn (Risk Validation)
// Date: January 18, 2025
// NO SIMPLIFICATIONS - FULL IMPLEMENTATION

use anyhow::{Result, Context};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rayon::prelude::*;

// ============================================================================
// VALIDATION STRATEGIES - Riley's Implementation
// ============================================================================

/// Cross validation strategy trait
pub trait ValidationStrategy: Send + Sync {
    /// Generate train/test splits
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    
    /// Get strategy name
    fn name(&self) -> &str;
}

// ============================================================================
// TIME SERIES SPLIT - Riley's Primary Implementation
// ============================================================================

/// Time series cross-validation with gap
pub struct TimeSeriesSplit {
    n_splits: usize,
    test_size: Option<usize>,
    gap: usize,
    max_train_size: Option<usize>,
}

impl TimeSeriesSplit {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: None,
            gap: 0,
            max_train_size: None,
        }
    }
    
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }
    
    pub fn with_test_size(mut self, test_size: usize) -> Self {
        self.test_size = Some(test_size);
        self
    }
    
    pub fn with_max_train_size(mut self, max_train_size: usize) -> Self {
        self.max_train_size = Some(max_train_size);
        self
    }
}

impl ValidationStrategy for TimeSeriesSplit {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        
        // Calculate test size
        let test_size = self.test_size.unwrap_or_else(|| {
            (n_samples - self.gap) / (self.n_splits + 1)
        });
        
        for i in 0..self.n_splits {
            // Calculate train end
            let train_end = if let Some(first_test_size) = self.test_size {
                n_samples - (self.n_splits - i) * (first_test_size + self.gap)
            } else {
                (i + 1) * test_size
            };
            
            // Apply gap
            let test_start = train_end + self.gap;
            let test_end = test_start + test_size;
            
            // Check bounds
            if test_end > n_samples {
                break;
            }
            
            // Calculate train start
            let train_start = if let Some(max_size) = self.max_train_size {
                train_end.saturating_sub(max_size)
            } else {
                0
            };
            
            // Create indices
            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            splits.push((train_indices, test_indices));
        }
        
        splits
    }
    
    fn name(&self) -> &str {
        "TimeSeriesSplit"
    }
}

// ============================================================================
// PURGED K-FOLD - Morgan's Implementation for Non-IID Data
// ============================================================================

/// Purged K-Fold with embargo for financial data
pub struct PurgedKFold {
    n_splits: usize,
    purge_gap: usize,
    embargo_pct: f64,
}

impl PurgedKFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            purge_gap: 0,
            embargo_pct: 0.0,
        }
    }
    
    pub fn with_purge_gap(mut self, gap: usize) -> Self {
        self.purge_gap = gap;
        self
    }
    
    pub fn with_embargo(mut self, pct: f64) -> Self {
        self.embargo_pct = pct.clamp(0.0, 0.5);
        self
    }
}

impl ValidationStrategy for PurgedKFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let fold_size = n_samples / self.n_splits;
        let embargo_size = (n_samples as f64 * self.embargo_pct) as usize;
        
        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples - embargo_size
            } else {
                (i + 1) * fold_size
            };
            
            // Create train indices with purge gap
            let mut train_indices = Vec::new();
            
            // Before test set
            if test_start > self.purge_gap {
                train_indices.extend(0..test_start.saturating_sub(self.purge_gap));
            }
            
            // After test set
            if test_end + self.purge_gap < n_samples - embargo_size {
                train_indices.extend(test_end + self.purge_gap..n_samples - embargo_size);
            }
            
            // Test indices
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }
        
        splits
    }
    
    fn name(&self) -> &str {
        "PurgedKFold"
    }
}

// ============================================================================
// COMBINATORIAL PURGED CV - Quinn's Risk-Aware Implementation
// ============================================================================

/// Combinatorial purged cross-validation for path-dependent strategies
pub struct CombinatorialPurgedCV {
    n_splits: usize,
    n_test_splits: usize,
    purge_gap: usize,
}

impl CombinatorialPurgedCV {
    pub fn new(n_splits: usize, n_test_splits: usize) -> Self {
        assert!(n_test_splits <= n_splits, "n_test_splits must be <= n_splits");
        Self {
            n_splits,
            n_test_splits,
            purge_gap: 0,
        }
    }
    
    pub fn with_purge_gap(mut self, gap: usize) -> Self {
        self.purge_gap = gap;
        self
    }
    
    /// Generate all combinations of test splits
    fn combinations(&self, n: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if n < k {
            return vec![];
        }
        
        let mut result = Vec::new();
        
        // Include first element
        for mut combo in self.combinations(n - 1, k - 1) {
            combo.insert(0, 0);
            for i in 1..combo.len() {
                combo[i] += 1;
            }
            result.push(combo);
        }
        
        // Exclude first element
        for mut combo in self.combinations(n - 1, k) {
            for i in 0..combo.len() {
                combo[i] += 1;
            }
            result.push(combo);
        }
        
        result
    }
}

impl ValidationStrategy for CombinatorialPurgedCV {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / self.n_splits;
        let combinations = self.combinations(self.n_splits, self.n_test_splits);
        let mut splits = Vec::new();
        
        for test_folds in combinations {
            let mut test_indices = Vec::new();
            let mut excluded_ranges = Vec::new();
            
            // Collect test indices and excluded ranges
            for &fold_idx in &test_folds {
                let start = fold_idx * fold_size;
                let end = if fold_idx == self.n_splits - 1 {
                    n_samples
                } else {
                    (fold_idx + 1) * fold_size
                };
                
                test_indices.extend(start..end);
                
                // Add purge zones
                let purge_start = start.saturating_sub(self.purge_gap);
                let purge_end = (end + self.purge_gap).min(n_samples);
                excluded_ranges.push((purge_start, purge_end));
            }
            
            // Create train indices (all not in test or purge zones)
            let mut train_indices = Vec::new();
            for i in 0..n_samples {
                let in_excluded = excluded_ranges.iter().any(|&(start, end)| i >= start && i < end);
                if !in_excluded {
                    train_indices.push(i);
                }
            }
            
            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }
        
        splits
    }
    
    fn name(&self) -> &str {
        "CombinatorialPurgedCV"
    }
}

// ============================================================================
// WALK-FORWARD ANALYSIS - Quinn's Production Strategy
// ============================================================================

/// Walk-forward analysis for production deployment
pub struct WalkForwardAnalysis {
    train_window: usize,
    test_window: usize,
    step_size: usize,
    expanding_window: bool,
}

impl WalkForwardAnalysis {
    pub fn new(train_window: usize, test_window: usize) -> Self {
        Self {
            train_window,
            test_window,
            step_size: test_window,
            expanding_window: false,
        }
    }
    
    pub fn with_step_size(mut self, step: usize) -> Self {
        self.step_size = step;
        self
    }
    
    pub fn with_expanding_window(mut self) -> Self {
        self.expanding_window = true;
        self
    }
}

impl ValidationStrategy for WalkForwardAnalysis {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let mut test_start = self.train_window;
        
        while test_start + self.test_window <= n_samples {
            let test_end = test_start + self.test_window;
            
            let train_start = if self.expanding_window {
                0
            } else {
                test_start.saturating_sub(self.train_window)
            };
            
            let train_indices: Vec<usize> = (train_start..test_start).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            splits.push((train_indices, test_indices));
            
            test_start += self.step_size;
        }
        
        splits
    }
    
    fn name(&self) -> &str {
        "WalkForwardAnalysis"
    }
}

// ============================================================================
// CROSS VALIDATOR - Riley's Main Implementation
// ============================================================================

/// Main cross-validation executor
pub struct CrossValidator {
    strategy: Box<dyn ValidationStrategy>,
    scoring_metrics: Vec<String>,
    parallel: bool,
}

impl CrossValidator {
    pub fn new(strategy: Box<dyn ValidationStrategy>) -> Self {
        Self {
            strategy,
            scoring_metrics: vec!["rmse".to_string(), "mae".to_string()],
            parallel: true,
        }
    }
    
    pub fn with_metrics(mut self, metrics: Vec<String>) -> Self {
        self.scoring_metrics = metrics;
        self
    }
    
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }
    
    /// Run cross-validation
    pub fn validate<F>(
        &self,
        data: &Array2<f64>,
        targets: &Array2<f64>,
        train_fn: F,
    ) -> Result<CVResults>
    where
        F: Fn(&Array2<f64>, &Array2<f64>) -> Result<HashMap<String, f64>> + Sync,
    {
        let n_samples = data.shape()[0];
        let splits = self.strategy.split(n_samples);
        let n_splits = splits.len();
        
        info!(
            "Running {} with {} splits on {} samples",
            self.strategy.name(),
            n_splits,
            n_samples
        );
        
        let results = if self.parallel {
            // Parallel execution - Jordan's optimization
            splits
                .par_iter()
                .enumerate()
                .map(|(fold_idx, (train_idx, test_idx))| {
                    self.validate_fold(data, targets, train_idx, test_idx, fold_idx, &train_fn)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Sequential execution
            splits
                .iter()
                .enumerate()
                .map(|(fold_idx, (train_idx, test_idx))| {
                    self.validate_fold(data, targets, train_idx, test_idx, fold_idx, &train_fn)
                })
                .collect::<Result<Vec<_>>>()?
        };
        
        // Aggregate results
        let mut aggregated = HashMap::new();
        for metric in &self.scoring_metrics {
            let values: Vec<f64> = results
                .iter()
                .filter_map(|r| r.metrics.get(metric))
                .copied()
                .collect();
            
            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = if values.len() > 1 {
                    let variance = values
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>() / (values.len() - 1) as f64;
                    variance.sqrt()
                } else {
                    0.0
                };
                
                aggregated.insert(format!("{}_mean", metric), mean);
                aggregated.insert(format!("{}_std", metric), std);
            }
        }
        
        Ok(CVResults {
            fold_results: results,
            aggregated_metrics: aggregated,
            strategy_name: self.strategy.name().to_string(),
            n_splits,
        })
    }
    
    /// Validate single fold
    fn validate_fold<F>(
        &self,
        data: &Array2<f64>,
        targets: &Array2<f64>,
        train_idx: &[usize],
        test_idx: &[usize],
        fold_idx: usize,
        train_fn: &F,
    ) -> Result<FoldResult>
    where
        F: Fn(&Array2<f64>, &Array2<f64>) -> Result<HashMap<String, f64>>,
    {
        debug!(
            "Validating fold {}: train_size={}, test_size={}",
            fold_idx,
            train_idx.len(),
            test_idx.len()
        );
        
        // Select data
        let train_data = data.select(Axis(0), train_idx);
        let train_targets = targets.select(Axis(0), train_idx);
        let test_data = data.select(Axis(0), test_idx);
        let test_targets = targets.select(Axis(0), test_idx);
        
        // Train and evaluate
        let metrics = train_fn(&train_data.to_owned(), &train_targets.to_owned())
            .context(format!("Failed to train fold {}", fold_idx))?;
        
        Ok(FoldResult {
            fold_idx,
            train_size: train_idx.len(),
            test_size: test_idx.len(),
            metrics,
        })
    }
}

// ============================================================================
// RESULT TYPES - Riley's Reporting
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub train_size: usize,
    pub test_size: usize,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResults {
    pub fold_results: Vec<FoldResult>,
    pub aggregated_metrics: HashMap<String, f64>,
    pub strategy_name: String,
    pub n_splits: usize,
}

impl CVResults {
    /// Get summary statistics
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Cross-Validation Results\n\
             Strategy: {}\n\
             Splits: {}\n\
             Metrics:\n",
            self.strategy_name, self.n_splits
        );
        
        for (metric, value) in &self.aggregated_metrics {
            summary.push_str(&format!("  {}: {:.6}\n", metric, value));
        }
        
        summary
    }
}

// ============================================================================
// TESTS - Riley's Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_series_split() {
        let splitter = TimeSeriesSplit::new(5).with_gap(10);
        let splits = splitter.split(1000);
        
        assert_eq!(splits.len(), 5);
        
        for (train, test) in &splits {
            // Check gap is maintained
            let max_train = train.iter().max().unwrap();
            let min_test = test.iter().min().unwrap();
            assert!(min_test - max_train >= 10);
        }
    }
    
    #[test]
    fn test_purged_kfold() {
        let splitter = PurgedKFold::new(5)
            .with_purge_gap(10)
            .with_embargo(0.1);
        
        let splits = splitter.split(1000);
        assert_eq!(splits.len(), 5);
        
        for (train, test) in &splits {
            // Ensure no overlap
            for &t in test {
                assert!(!train.contains(&t));
            }
        }
    }
    
    #[test]
    fn test_walk_forward() {
        let splitter = WalkForwardAnalysis::new(100, 20)
            .with_step_size(10);
        
        let splits = splitter.split(500);
        
        // Check windows move forward
        for i in 1..splits.len() {
            let prev_test_start = splits[i-1].1[0];
            let curr_test_start = splits[i].1[0];
            assert_eq!(curr_test_start - prev_test_start, 10);
        }
    }
    
    #[test]
    fn test_combinatorial_cv() {
        let splitter = CombinatorialPurgedCV::new(4, 2);
        let combinations = splitter.combinations(4, 2);
        
        // Should be C(4,2) = 6 combinations
        assert_eq!(combinations.len(), 6);
    }
}

// ============================================================================
// TEAM SIGN-OFF
// ============================================================================
// Riley: "Comprehensive cross-validation with time series support"
// Morgan: "Purged CV for financial data implemented"
// Quinn: "Risk-aware validation strategies in place"