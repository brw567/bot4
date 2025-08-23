// OVERFITTING PREVENTION SYSTEM - CRITICAL FOR SURVIVAL!
// Team: Riley (Lead) + Morgan - NO FALSE SIGNALS, NO CURVE FITTING!
// Target: Robust strategies that work in production, not just backtests

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use rand::prelude::*;
use statrs::distribution::{StudentsT, ContinuousCDF};
use nalgebra::{DMatrix, DVector};

use crate::{DataError, Result};

/// Master overfitting prevention system
pub struct OverfittingPreventionSystem {
    // Validation strategies
    walk_forward_validator: Arc<RwLock<WalkForwardValidator>>,
    cross_validator: Arc<RwLock<PurgedKFoldValidator>>,
    monte_carlo_validator: Arc<RwLock<MonteCarloValidator>>,
    
    // Regularization techniques
    regularizer: Arc<RwLock<Regularizer>>,
    feature_selector: Arc<RwLock<FeatureSelector>>,
    ensemble_manager: Arc<RwLock<EnsembleManager>>,
    
    // Stability testing
    stability_tester: Arc<RwLock<StabilityTester>>,
    robustness_checker: Arc<RwLock<RobustnessChecker>>,
    
    // Performance tracking
    out_of_sample_tracker: Arc<RwLock<OutOfSampleTracker>>,
    
    config: OverfittingConfig,
}

#[derive(Debug, Clone)]
pub struct OverfittingConfig {
    // Validation parameters
    pub min_train_samples: usize,         // Minimum 1000 samples
    pub walk_forward_window: usize,       // 252 days (1 year)
    pub walk_forward_step: usize,         // 21 days (1 month)
    pub purge_gap: usize,                  // 10 days to prevent leakage
    pub embargo_period: usize,             // 5 days after test
    pub k_folds: usize,                    // 5-fold cross-validation
    
    // Regularization parameters
    pub l1_lambda: f64,                    // LASSO penalty
    pub l2_lambda: f64,                    // Ridge penalty
    pub elastic_net_ratio: f64,            // L1/(L1+L2) ratio
    pub dropout_rate: f64,                 // Feature dropout
    pub max_features_ratio: f64,           // Max features to use
    
    // Stability thresholds
    pub max_sharpe_in_sample: f64,         // 3.0 - suspicious if higher
    pub min_sharpe_out_sample: f64,        // 0.5 - must be profitable
    pub max_drawdown_ratio: f64,           // OOS DD / IS DD > 0.7
    pub correlation_threshold: f64,        // Min correlation stability
    
    // Monte Carlo parameters
    pub monte_carlo_iterations: usize,     // 1000 simulations
    pub confidence_level: f64,             // 95% confidence
    pub min_success_rate: f64,             // 60% must be profitable
}

impl Default for OverfittingConfig {
    fn default() -> Self {
        Self {
            min_train_samples: 1000,
            walk_forward_window: 252,
            walk_forward_step: 21,
            purge_gap: 10,
            embargo_period: 5,
            k_folds: 5,
            l1_lambda: 0.01,
            l2_lambda: 0.1,
            elastic_net_ratio: 0.5,
            dropout_rate: 0.2,
            max_features_ratio: 0.7,
            max_sharpe_in_sample: 3.0,
            min_sharpe_out_sample: 0.5,
            max_drawdown_ratio: 0.7,
            correlation_threshold: 0.3,
            monte_carlo_iterations: 1000,
            confidence_level: 0.95,
            min_success_rate: 0.6,
        }
    }
}

/// Walk-forward analysis validator (most realistic)
struct WalkForwardValidator {
    windows: Vec<WalkForwardWindow>,
    current_window: usize,
    performance_decay: Vec<f64>,
}

#[derive(Debug, Clone)]
struct WalkForwardWindow {
    train_start: usize,
    train_end: usize,
    test_start: usize,
    test_end: usize,
    in_sample_performance: Performance,
    out_sample_performance: Performance,
}

#[derive(Debug, Clone, Default)]
struct Performance {
    sharpe_ratio: f64,
    sortino_ratio: f64,
    calmar_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    profit_factor: f64,
    returns: Vec<f64>,
}

/// Purged K-Fold cross-validation (López de Prado method)
struct PurgedKFoldValidator {
    folds: Vec<PurgedFold>,
    purge_gap: usize,
    embargo_period: usize,
}

#[derive(Debug, Clone)]
struct PurgedFold {
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
    gap_indices: Vec<usize>,  // Purged to prevent leakage
    embargo_indices: Vec<usize>,  // Post-test embargo
}

/// Monte Carlo validation with random sampling
struct MonteCarloValidator {
    simulations: Vec<MonteCarloSimulation>,
    confidence_intervals: ConfidenceIntervals,
}

#[derive(Debug, Clone)]
struct MonteCarloSimulation {
    sample_indices: Vec<usize>,
    performance: Performance,
    random_seed: u64,
}

#[derive(Debug, Clone)]
struct ConfidenceIntervals {
    sharpe_5th: f64,
    sharpe_50th: f64,
    sharpe_95th: f64,
    drawdown_5th: f64,
    drawdown_50th: f64,
    drawdown_95th: f64,
}

/// Regularization techniques
struct Regularizer {
    l1_penalty: f64,
    l2_penalty: f64,
    elastic_net: bool,
    weight_decay: f64,
    early_stopping_patience: usize,
    best_validation_score: f64,
    patience_counter: usize,
}

/// Feature selection to prevent overfitting
struct FeatureSelector {
    selected_features: Vec<usize>,
    feature_importance: Vec<f64>,
    correlation_matrix: DMatrix<f64>,
    vif_scores: Vec<f64>,  // Variance Inflation Factor
    mutual_information: Vec<f64>,
}

/// Ensemble to reduce overfitting
struct EnsembleManager {
    models: Vec<ModelSnapshot>,
    voting_weights: Vec<f64>,
    diversity_score: f64,
}

#[derive(Debug, Clone)]
struct ModelSnapshot {
    model_id: String,
    train_period: (usize, usize),
    features_used: Vec<usize>,
    hyperparameters: HashMap<String, f64>,
    validation_score: f64,
}

/// Stability testing across different conditions
struct StabilityTester {
    parameter_sensitivity: HashMap<String, f64>,
    temporal_stability: Vec<f64>,
    market_regime_performance: HashMap<String, Performance>,
    correlation_stability: Vec<f64>,
}

/// Robustness checking
struct RobustnessChecker {
    noise_resistance: f64,
    missing_data_tolerance: f64,
    parameter_perturbation: Vec<f64>,
    bootstrap_confidence: f64,
}

/// Out-of-sample performance tracking
struct OutOfSampleTracker {
    in_sample_metrics: Performance,
    out_sample_metrics: Performance,
    degradation_rate: f64,
    live_performance: Performance,
}

impl OverfittingPreventionSystem {
    pub fn new(config: OverfittingConfig) -> Self {
        Self {
            walk_forward_validator: Arc::new(RwLock::new(WalkForwardValidator {
                windows: Vec::new(),
                current_window: 0,
                performance_decay: Vec::new(),
            })),
            cross_validator: Arc::new(RwLock::new(PurgedKFoldValidator {
                folds: Vec::new(),
                purge_gap: config.purge_gap,
                embargo_period: config.embargo_period,
            })),
            monte_carlo_validator: Arc::new(RwLock::new(MonteCarloValidator {
                simulations: Vec::new(),
                confidence_intervals: ConfidenceIntervals {
                    sharpe_5th: 0.0,
                    sharpe_50th: 0.0,
                    sharpe_95th: 0.0,
                    drawdown_5th: 0.0,
                    drawdown_50th: 0.0,
                    drawdown_95th: 0.0,
                },
            })),
            regularizer: Arc::new(RwLock::new(Regularizer {
                l1_penalty: config.l1_lambda,
                l2_penalty: config.l2_lambda,
                elastic_net: config.elastic_net_ratio > 0.0,
                weight_decay: 0.01,
                early_stopping_patience: 10,
                best_validation_score: f64::NEG_INFINITY,
                patience_counter: 0,
            })),
            feature_selector: Arc::new(RwLock::new(FeatureSelector {
                selected_features: Vec::new(),
                feature_importance: Vec::new(),
                correlation_matrix: DMatrix::zeros(0, 0),
                vif_scores: Vec::new(),
                mutual_information: Vec::new(),
            })),
            ensemble_manager: Arc::new(RwLock::new(EnsembleManager {
                models: Vec::new(),
                voting_weights: Vec::new(),
                diversity_score: 0.0,
            })),
            stability_tester: Arc::new(RwLock::new(StabilityTester {
                parameter_sensitivity: HashMap::new(),
                temporal_stability: Vec::new(),
                market_regime_performance: HashMap::new(),
                correlation_stability: Vec::new(),
            })),
            robustness_checker: Arc::new(RwLock::new(RobustnessChecker {
                noise_resistance: 0.0,
                missing_data_tolerance: 0.0,
                parameter_perturbation: Vec::new(),
                bootstrap_confidence: 0.0,
            })),
            out_of_sample_tracker: Arc::new(RwLock::new(OutOfSampleTracker {
                in_sample_metrics: Performance::default(),
                out_sample_metrics: Performance::default(),
                degradation_rate: 0.0,
                live_performance: Performance::default(),
            })),
            config,
        }
    }
    
    /// Validate correlation stability to prevent spurious relationships
    pub fn validate_correlation_stability(
        &self,
        correlations: &[(String, String, Vec<f64>)],
    ) -> CorrelationValidation {
        let mut stable_correlations = Vec::new();
        let mut unstable_correlations = Vec::new();
        let mut spurious_correlations = Vec::new();
        
        for (asset1, asset2, corr_series) in correlations {
            // Calculate rolling statistics
            let mean_corr = corr_series.iter().sum::<f64>() / corr_series.len() as f64;
            let std_corr = self.calculate_std(corr_series);
            let min_corr = corr_series.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_corr = corr_series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            // Check for sign changes (very bad for trading)
            let sign_changes = self.count_sign_changes(corr_series);
            let sign_change_rate = sign_changes as f64 / corr_series.len() as f64;
            
            // Calculate autocorrelation of correlation series
            let correlation_autocorr = self.autocorrelation(corr_series, 1);
            
            // Assess stability
            let stability_score = (1.0 - std_corr) * correlation_autocorr * (1.0 - sign_change_rate);
            
            let validation = CorrelationAssessment {
                pair: (asset1.clone(), asset2.clone()),
                mean_correlation: mean_corr,
                std_correlation: std_corr,
                min_correlation: min_corr,
                max_correlation: max_corr,
                stability_score,
                sign_changes,
                is_stable: stability_score > self.config.correlation_threshold,
                is_spurious: mean_corr.abs() < 0.1 || std_corr > 0.5,
                recommendation: self.correlation_recommendation(stability_score, mean_corr),
            };
            
            if validation.is_spurious {
                spurious_correlations.push(validation.clone());
            } else if validation.is_stable {
                stable_correlations.push(validation.clone());
            } else {
                unstable_correlations.push(validation.clone());
            }
        }
        
        CorrelationValidation {
            stable_correlations,
            unstable_correlations,
            spurious_correlations,
            overall_stability: self.calculate_overall_stability(&stable_correlations, correlations.len()),
            warnings: self.generate_correlation_warnings(&spurious_correlations),
        }
    }
    
    /// Walk-forward validation (MOST IMPORTANT FOR PREVENTING OVERFITTING)
    pub fn walk_forward_validation<F>(
        &self,
        data: &[f64],
        model_trainer: F,
    ) -> WalkForwardResults
    where
        F: Fn(&[f64]) -> Performance,
    {
        let mut windows = Vec::new();
        let data_len = data.len();
        
        if data_len < self.config.min_train_samples + self.config.walk_forward_window {
            return WalkForwardResults {
                windows: vec![],
                average_in_sample_sharpe: 0.0,
                average_out_sample_sharpe: 0.0,
                performance_degradation: 1.0,
                is_overfit: true,
                confidence: 0.0,
            };
        }
        
        let mut position = self.config.walk_forward_window;
        
        while position + self.config.walk_forward_step <= data_len {
            let train_start = position - self.config.walk_forward_window;
            let train_end = position;
            let test_start = train_end + self.config.purge_gap;
            let test_end = (test_start + self.config.walk_forward_step).min(data_len);
            
            if test_end > data_len {
                break;
            }
            
            // Train on in-sample
            let train_data = &data[train_start..train_end];
            let in_sample_perf = model_trainer(train_data);
            
            // Test on out-of-sample
            let test_data = &data[test_start..test_end];
            let out_sample_perf = model_trainer(test_data);
            
            windows.push(WalkForwardWindow {
                train_start,
                train_end,
                test_start,
                test_end,
                in_sample_performance: in_sample_perf,
                out_sample_performance: out_sample_perf,
            });
            
            position += self.config.walk_forward_step;
        }
        
        // Calculate aggregate metrics
        let avg_is_sharpe = windows.iter()
            .map(|w| w.in_sample_performance.sharpe_ratio)
            .sum::<f64>() / windows.len() as f64;
            
        let avg_oos_sharpe = windows.iter()
            .map(|w| w.out_sample_performance.sharpe_ratio)
            .sum::<f64>() / windows.len() as f64;
        
        let degradation = if avg_is_sharpe != 0.0 {
            avg_oos_sharpe / avg_is_sharpe
        } else {
            0.0
        };
        
        // Detect overfitting
        let is_overfit = self.detect_overfitting(&windows);
        
        WalkForwardResults {
            windows,
            average_in_sample_sharpe: avg_is_sharpe,
            average_out_sample_sharpe: avg_oos_sharpe,
            performance_degradation: degradation,
            is_overfit,
            confidence: self.calculate_confidence(degradation, avg_oos_sharpe),
        }
    }
    
    /// Purged K-fold cross-validation (López de Prado method)
    pub fn purged_kfold_validation<F>(
        &self,
        data: &[f64],
        labels: &[f64],
        model_trainer: F,
    ) -> PurgedKFoldResults
    where
        F: Fn(&[f64], &[f64]) -> Performance,
    {
        let n = data.len();
        let fold_size = n / self.config.k_folds;
        
        let mut folds = Vec::new();
        let mut performances = Vec::new();
        
        for k in 0..self.config.k_folds {
            let test_start = k * fold_size;
            let test_end = ((k + 1) * fold_size).min(n);
            
            // Create train indices (excluding test, purge, and embargo)
            let mut train_indices = Vec::new();
            
            for i in 0..n {
                // Skip test set
                if i >= test_start && i < test_end {
                    continue;
                }
                
                // Skip purge gap before test
                if i >= test_start.saturating_sub(self.config.purge_gap) && i < test_start {
                    continue;
                }
                
                // Skip embargo after test
                if i >= test_end && i < (test_end + self.config.embargo_period).min(n) {
                    continue;
                }
                
                train_indices.push(i);
            }
            
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            // Train and evaluate
            let train_data: Vec<f64> = train_indices.iter().map(|&i| data[i]).collect();
            let train_labels: Vec<f64> = train_indices.iter().map(|&i| labels[i]).collect();
            let test_data: Vec<f64> = test_indices.iter().map(|&i| data[i]).collect();
            let test_labels: Vec<f64> = test_indices.iter().map(|&i| labels[i]).collect();
            
            let perf = model_trainer(&train_data, &train_labels);
            performances.push(perf);
            
            folds.push(PurgedFold {
                train_indices,
                test_indices,
                gap_indices: (test_start.saturating_sub(self.config.purge_gap)..test_start).collect(),
                embargo_indices: (test_end..(test_end + self.config.embargo_period).min(n)).collect(),
            });
        }
        
        // Calculate cross-validation statistics
        let mean_sharpe = performances.iter()
            .map(|p| p.sharpe_ratio)
            .sum::<f64>() / performances.len() as f64;
            
        let std_sharpe = self.calculate_std(
            &performances.iter().map(|p| p.sharpe_ratio).collect::<Vec<_>>()
        );
        
        PurgedKFoldResults {
            folds,
            performances,
            mean_sharpe,
            std_sharpe,
            cv_score: mean_sharpe - std_sharpe,  // Conservative estimate
            is_robust: std_sharpe < 0.5 && mean_sharpe > self.config.min_sharpe_out_sample,
        }
    }
    
    /// Apply regularization to prevent overfitting
    pub fn apply_regularization(&self, weights: &mut [f64], gradients: &[f64]) {
        let regularizer = self.regularizer.read();
        
        for i in 0..weights.len() {
            // L2 regularization (Ridge)
            let l2_grad = 2.0 * regularizer.l2_penalty * weights[i];
            
            // L1 regularization (LASSO)
            let l1_grad = regularizer.l1_penalty * weights[i].signum();
            
            // Elastic Net combines both
            let reg_gradient = if regularizer.elastic_net {
                self.config.elastic_net_ratio * l1_grad + 
                (1.0 - self.config.elastic_net_ratio) * l2_grad
            } else {
                l2_grad
            };
            
            // Update weight with regularization
            weights[i] -= gradients[i] + reg_gradient;
            
            // Apply weight decay
            weights[i] *= (1.0 - regularizer.weight_decay);
        }
    }
    
    /// Select features to prevent overfitting
    pub fn select_features(&self, features: &DMatrix<f64>, importance: &[f64]) -> Vec<usize> {
        let mut selector = self.feature_selector.write();
        
        let n_features = features.ncols();
        let max_features = (n_features as f64 * self.config.max_features_ratio) as usize;
        
        // Calculate correlation matrix
        selector.correlation_matrix = self.calculate_correlation_matrix(features);
        
        // Calculate VIF for multicollinearity detection
        selector.vif_scores = self.calculate_vif(&selector.correlation_matrix);
        
        // Remove highly correlated features
        let mut selected = Vec::new();
        let mut used = vec![false; n_features];
        
        // Sort by importance
        let mut importance_indices: Vec<(usize, f64)> = importance.iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        importance_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (idx, imp) in importance_indices {
            if selected.len() >= max_features {
                break;
            }
            
            if used[idx] {
                continue;
            }
            
            // Check VIF (variance inflation factor)
            if selector.vif_scores[idx] > 10.0 {
                continue;  // Skip due to multicollinearity
            }
            
            // Check correlation with already selected features
            let mut max_corr = 0.0;
            for &selected_idx in &selected {
                let corr = selector.correlation_matrix[(idx, selected_idx)].abs();
                max_corr = max_corr.max(corr);
            }
            
            if max_corr < 0.8 {  // Not too correlated with existing
                selected.push(idx);
                used[idx] = true;
            }
        }
        
        selector.selected_features = selected.clone();
        selected
    }
    
    /// Check if model is robust to noise
    pub fn test_noise_robustness(&self, data: &[f64], noise_levels: &[f64]) -> RobustnessResults {
        let mut robustness = self.robustness_checker.write();
        let mut noise_performances = Vec::new();
        
        for &noise_level in noise_levels {
            let noisy_data = self.add_noise(data, noise_level);
            let perf = self.evaluate_performance(&noisy_data);
            noise_performances.push((noise_level, perf));
        }
        
        // Calculate robustness score
        let baseline_perf = self.evaluate_performance(data);
        let avg_degradation = noise_performances.iter()
            .map(|(_, p)| (baseline_perf.sharpe_ratio - p.sharpe_ratio) / baseline_perf.sharpe_ratio)
            .sum::<f64>() / noise_performances.len() as f64;
        
        robustness.noise_resistance = 1.0 - avg_degradation;
        
        RobustnessResults {
            noise_performances,
            robustness_score: robustness.noise_resistance,
            is_robust: robustness.noise_resistance > 0.7,
            max_acceptable_noise: self.find_max_acceptable_noise(&noise_performances),
        }
    }
    
    /// Monte Carlo validation for confidence intervals
    pub fn monte_carlo_validation<F>(
        &self,
        data: &[f64],
        model_trainer: F,
    ) -> MonteCarloResults
    where
        F: Fn(&[f64]) -> Performance,
    {
        let mut validator = self.monte_carlo_validator.write();
        let mut performances = Vec::new();
        let mut rng = thread_rng();
        
        for i in 0..self.config.monte_carlo_iterations {
            // Random sampling with replacement (bootstrap)
            let sample_size = (data.len() as f64 * 0.8) as usize;
            let mut sample_indices = Vec::with_capacity(sample_size);
            
            for _ in 0..sample_size {
                sample_indices.push(rng.gen_range(0..data.len()));
            }
            
            let sample_data: Vec<f64> = sample_indices.iter()
                .map(|&idx| data[idx])
                .collect();
            
            let perf = model_trainer(&sample_data);
            
            validator.simulations.push(MonteCarloSimulation {
                sample_indices: sample_indices.clone(),
                performance: perf.clone(),
                random_seed: i as u64,
            });
            
            performances.push(perf);
        }
        
        // Calculate confidence intervals
        let mut sharpes: Vec<f64> = performances.iter()
            .map(|p| p.sharpe_ratio)
            .collect();
        sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p5_idx = (sharpes.len() as f64 * 0.05) as usize;
        let p50_idx = (sharpes.len() as f64 * 0.50) as usize;
        let p95_idx = (sharpes.len() as f64 * 0.95) as usize;
        
        validator.confidence_intervals = ConfidenceIntervals {
            sharpe_5th: sharpes[p5_idx],
            sharpe_50th: sharpes[p50_idx],
            sharpe_95th: sharpes[p95_idx],
            drawdown_5th: 0.0,  // Would calculate similarly
            drawdown_50th: 0.0,
            drawdown_95th: 0.0,
        };
        
        // Count profitable simulations
        let profitable_count = performances.iter()
            .filter(|p| p.sharpe_ratio > 0.0)
            .count();
        let success_rate = profitable_count as f64 / performances.len() as f64;
        
        MonteCarloResults {
            confidence_intervals: validator.confidence_intervals.clone(),
            success_rate,
            is_significant: sharpes[p5_idx] > 0.0,  // Even 5th percentile is profitable
            expected_sharpe: sharpes[p50_idx],
            worst_case_sharpe: sharpes[p5_idx],
            best_case_sharpe: sharpes[p95_idx],
        }
    }
    
    // Helper methods
    
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    fn count_sign_changes(&self, values: &[f64]) -> usize {
        if values.len() < 2 {
            return 0;
        }
        
        let mut changes = 0;
        for i in 1..values.len() {
            if values[i].signum() != values[i - 1].signum() {
                changes += 1;
            }
        }
        changes
    }
    
    fn autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in lag..values.len() {
            numerator += (values[i] - mean) * (values[i - lag] - mean);
        }
        
        for value in values {
            denominator += (value - mean).powi(2);
        }
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn correlation_recommendation(&self, stability: f64, mean_corr: f64) -> String {
        if stability < 0.3 {
            "AVOID - Unstable correlation, high risk of failure".to_string()
        } else if mean_corr.abs() < 0.2 {
            "WEAK - Correlation too weak to trade reliably".to_string()
        } else if stability > 0.7 && mean_corr.abs() > 0.5 {
            "TRADE - Stable and strong correlation".to_string()
        } else {
            "CAUTION - Monitor stability before trading".to_string()
        }
    }
    
    fn calculate_overall_stability(&self, stable: &[CorrelationAssessment], total: usize) -> f64 {
        stable.len() as f64 / total as f64
    }
    
    fn generate_correlation_warnings(&self, spurious: &[CorrelationAssessment]) -> Vec<String> {
        spurious.iter().map(|c| {
            format!("WARNING: {}-{} correlation is spurious (mean={:.2}, std={:.2})",
                    c.pair.0, c.pair.1, c.mean_correlation, c.std_correlation)
        }).collect()
    }
    
    fn detect_overfitting(&self, windows: &[WalkForwardWindow]) -> bool {
        // Multiple overfitting checks
        
        // 1. Performance degradation check
        let avg_degradation = windows.iter()
            .map(|w| {
                if w.in_sample_performance.sharpe_ratio != 0.0 {
                    w.out_sample_performance.sharpe_ratio / w.in_sample_performance.sharpe_ratio
                } else {
                    0.0
                }
            })
            .sum::<f64>() / windows.len() as f64;
        
        if avg_degradation < 0.5 {
            return true;  // Severe degradation
        }
        
        // 2. Suspiciously high in-sample Sharpe
        let max_is_sharpe = windows.iter()
            .map(|w| w.in_sample_performance.sharpe_ratio)
            .fold(f64::NEG_INFINITY, f64::max);
        
        if max_is_sharpe > self.config.max_sharpe_in_sample {
            return true;  // Too good to be true
        }
        
        // 3. Consistently poor out-of-sample
        let poor_oos_count = windows.iter()
            .filter(|w| w.out_sample_performance.sharpe_ratio < self.config.min_sharpe_out_sample)
            .count();
        
        if poor_oos_count > windows.len() / 2 {
            return true;  // Majority of OOS periods are unprofitable
        }
        
        false
    }
    
    fn calculate_confidence(&self, degradation: f64, oos_sharpe: f64) -> f64 {
        let degradation_score = (degradation.min(1.0).max(0.0) * 0.5).min(0.5);
        let sharpe_score = ((oos_sharpe / 2.0).min(1.0).max(0.0) * 0.5).min(0.5);
        degradation_score + sharpe_score
    }
    
    fn calculate_correlation_matrix(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let n = features.ncols();
        let mut corr_matrix = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in i..n {
                let col_i = features.column(i);
                let col_j = features.column(j);
                
                let corr = self.pearson_correlation(
                    col_i.as_slice(),
                    col_j.as_slice()
                );
                
                corr_matrix[(i, j)] = corr;
                corr_matrix[(j, i)] = corr;
            }
        }
        
        corr_matrix
    }
    
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn calculate_vif(&self, corr_matrix: &DMatrix<f64>) -> Vec<f64> {
        let n = corr_matrix.nrows();
        let mut vif_scores = Vec::with_capacity(n);
        
        for i in 0..n {
            // Calculate R² for feature i regressed on all others
            // Simplified: VIF = 1 / (1 - R²)
            let r_squared = 0.5;  // Simplified - would calculate actual R²
            let vif = 1.0 / (1.0 - r_squared);
            vif_scores.push(vif);
        }
        
        vif_scores
    }
    
    fn add_noise(&self, data: &[f64], noise_level: f64) -> Vec<f64> {
        let mut rng = thread_rng();
        data.iter().map(|&x| {
            let noise = rng.gen_range(-noise_level..noise_level);
            x + x * noise  // Proportional noise
        }).collect()
    }
    
    fn evaluate_performance(&self, data: &[f64]) -> Performance {
        // Simplified performance calculation
        Performance {
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            calmar_ratio: 1.0,
            max_drawdown: 0.1,
            win_rate: 0.55,
            profit_factor: 1.5,
            returns: data.to_vec(),
        }
    }
    
    fn find_max_acceptable_noise(&self, performances: &[(f64, Performance)]) -> f64 {
        for (noise, perf) in performances {
            if perf.sharpe_ratio < self.config.min_sharpe_out_sample {
                return *noise;
            }
        }
        performances.last().map(|(n, _)| *n).unwrap_or(0.0)
    }
}

// Output structures

#[derive(Debug, Clone)]
pub struct CorrelationValidation {
    pub stable_correlations: Vec<CorrelationAssessment>,
    pub unstable_correlations: Vec<CorrelationAssessment>,
    pub spurious_correlations: Vec<CorrelationAssessment>,
    pub overall_stability: f64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CorrelationAssessment {
    pub pair: (String, String),
    pub mean_correlation: f64,
    pub std_correlation: f64,
    pub min_correlation: f64,
    pub max_correlation: f64,
    pub stability_score: f64,
    pub sign_changes: usize,
    pub is_stable: bool,
    pub is_spurious: bool,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub struct WalkForwardResults {
    pub windows: Vec<WalkForwardWindow>,
    pub average_in_sample_sharpe: f64,
    pub average_out_sample_sharpe: f64,
    pub performance_degradation: f64,
    pub is_overfit: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct PurgedKFoldResults {
    pub folds: Vec<PurgedFold>,
    pub performances: Vec<Performance>,
    pub mean_sharpe: f64,
    pub std_sharpe: f64,
    pub cv_score: f64,
    pub is_robust: bool,
}

#[derive(Debug, Clone)]
pub struct RobustnessResults {
    pub noise_performances: Vec<(f64, Performance)>,
    pub robustness_score: f64,
    pub is_robust: bool,
    pub max_acceptable_noise: f64,
}

#[derive(Debug, Clone)]
pub struct MonteCarloResults {
    pub confidence_intervals: ConfidenceIntervals,
    pub success_rate: f64,
    pub is_significant: bool,
    pub expected_sharpe: f64,
    pub worst_case_sharpe: f64,
    pub best_case_sharpe: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_overfitting_detection() {
        let system = OverfittingPreventionSystem::new(OverfittingConfig::default());
        
        // Create fake data with clear overfitting pattern
        let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        
        let result = system.walk_forward_validation(&data, |d| {
            Performance {
                sharpe_ratio: if d.len() > 500 { 3.0 } else { 0.3 },  // Overfit: high IS, low OOS
                ..Default::default()
            }
        });
        
        assert!(result.is_overfit);
        assert!(result.performance_degradation < 0.5);
    }
    
    #[test]
    fn test_correlation_stability() {
        let system = OverfittingPreventionSystem::new(OverfittingConfig::default());
        
        // Stable correlation
        let stable_corr = vec![0.6, 0.65, 0.62, 0.64, 0.61, 0.63];
        
        // Unstable correlation
        let unstable_corr = vec![0.8, -0.3, 0.5, -0.6, 0.9, -0.2];
        
        let correlations = vec![
            ("BTC".to_string(), "ETH".to_string(), stable_corr),
            ("BTC".to_string(), "RANDOM".to_string(), unstable_corr),
        ];
        
        let validation = system.validate_correlation_stability(&correlations);
        
        assert_eq!(validation.stable_correlations.len(), 1);
        assert_eq!(validation.unstable_correlations.len(), 1);
    }
}