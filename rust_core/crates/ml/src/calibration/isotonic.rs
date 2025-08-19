// Isotonic Probability Calibration - Prevents Overconfident Predictions
// Morgan (ML Lead) + Quinn (Risk)
// CRITICAL: Sophia Requirement #3 - Calibrates raw ML outputs
// References: Zadrozny & Elkan (2002), Platt (1999), Niculescu-Mizil & Caruana (2005)

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Market regimes for regime-specific calibration
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,      // Strong directional movement
    RangeBound,    // Sideways consolidation
    Crisis,        // High volatility panic
    Breakout,      // Price discovery phase
}

/// Isotonic Calibration - Monotonic probability mapping
/// CRITICAL: Prevents overconfident predictions that lead to oversizing!
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    /// Separate calibrators per regime (prevents regime mixing)
    calibrators: HashMap<MarketRegime, IsotonicRegression>,
    
    /// Minimum samples for reliable calibration
    min_samples: usize,
    
    /// Cross-validation folds for calibration fitting
    cv_folds: usize,
    
    /// Regularization strength (prevents overfitting to calibration data)
    regularization: f32,
    
    /// Performance metrics
    brier_scores: HashMap<MarketRegime, f32>,
    calibration_errors: HashMap<MarketRegime, f32>,
}

impl IsotonicCalibrator {
    pub fn new() -> Self {
        Self {
            calibrators: HashMap::new(),
            min_samples: 1000,  // Need sufficient data for reliable calibration
            cv_folds: 5,        // Cross-validate calibration
            regularization: 0.01,
            brier_scores: HashMap::new(),
            calibration_errors: HashMap::new(),
        }
    }
    
    /// Calibrate model probabilities using isotonic regression
    /// Morgan: "This ensures our probabilities match empirical frequencies!"
    pub fn calibrate(
        &mut self,
        raw_probs: &Array1<f32>,
        outcomes: &Array1<f32>,
        regime: MarketRegime,
    ) -> Result<(), CalibrationError> {
        let n = raw_probs.len();
        
        if n != outcomes.len() {
            return Err(CalibrationError::DimensionMismatch);
        }
        
        if n < self.min_samples {
            return Err(CalibrationError::InsufficientData {
                required: self.min_samples,
                provided: n,
            });
        }
        
        // Cross-validate to prevent overfitting calibration
        let best_calibrator = self.cross_validate_calibration(raw_probs, outcomes)?;
        
        // Calculate calibration metrics
        let brier = self.calculate_brier_score(raw_probs, outcomes);
        let cal_error = self.calculate_calibration_error(raw_probs, outcomes);
        
        info!(
            "Regime {:?} calibration: Brier={:.4}, CalError={:.4}",
            regime, brier, cal_error
        );
        
        // Store calibrator and metrics
        self.calibrators.insert(regime, best_calibrator);
        self.brier_scores.insert(regime, brier);
        self.calibration_errors.insert(regime, cal_error);
        
        // Validate calibration quality
        if brier > 0.25 {
            warn!("High Brier score {:.4} for regime {:?}", brier, regime);
        }
        
        Ok(())
    }
    
    /// Cross-validate calibration to prevent overfitting
    fn cross_validate_calibration(
        &self,
        raw_probs: &Array1<f32>,
        outcomes: &Array1<f32>,
    ) -> Result<IsotonicRegression, CalibrationError> {
        let n = raw_probs.len();
        let fold_size = n / self.cv_folds;
        
        let mut best_score = f32::MAX;
        let mut best_calibrator = None;
        
        for fold in 0..self.cv_folds {
            let val_start = fold * fold_size;
            let val_end = ((fold + 1) * fold_size).min(n);
            
            // Split data
            let mut train_probs = Vec::new();
            let mut train_outcomes = Vec::new();
            let mut val_probs = Vec::new();
            let mut val_outcomes = Vec::new();
            
            for i in 0..n {
                if i >= val_start && i < val_end {
                    val_probs.push(raw_probs[i]);
                    val_outcomes.push(outcomes[i]);
                } else {
                    train_probs.push(raw_probs[i]);
                    train_outcomes.push(outcomes[i]);
                }
            }
            
            // Fit calibrator on training fold
            let mut calibrator = IsotonicRegression::new();
            calibrator.fit_with_regularization(
                &Array1::from(train_probs),
                &Array1::from(train_outcomes),
                self.regularization,
            )?;
            
            // Evaluate on validation fold
            let val_probs_arr = Array1::from(val_probs);
            let val_outcomes_arr = Array1::from(val_outcomes);
            let calibrated = calibrator.transform(&val_probs_arr)?;
            
            let score = self.calculate_brier_score(&calibrated, &val_outcomes_arr);
            
            if score < best_score {
                best_score = score;
                best_calibrator = Some(calibrator);
            }
        }
        
        best_calibrator.ok_or(CalibrationError::CalibrationFailed)
    }
    
    /// Transform raw probabilities to calibrated probabilities
    pub fn transform(&self, raw_prob: f32, regime: MarketRegime) -> f32 {
        self.calibrators
            .get(&regime)
            .map(|cal| cal.transform_single(raw_prob))
            .unwrap_or(raw_prob)
    }
    
    /// Batch transform probabilities
    pub fn transform_batch(
        &self,
        raw_probs: &Array1<f32>,
        regime: MarketRegime,
    ) -> Result<Array1<f32>, CalibrationError> {
        match self.calibrators.get(&regime) {
            Some(cal) => cal.transform(raw_probs),
            None => Ok(raw_probs.clone()),
        }
    }
    
    /// Calculate Brier score (lower is better, 0 is perfect)
    pub fn calculate_brier_score(&self, probs: &Array1<f32>, outcomes: &Array1<f32>) -> f32 {
        let n = probs.len() as f32;
        probs.iter()
            .zip(outcomes.iter())
            .map(|(p, o)| (p - o).powi(2))
            .sum::<f32>() / n
    }
    
    /// Calculate Expected Calibration Error (ECE)
    fn calculate_calibration_error(&self, probs: &Array1<f32>, outcomes: &Array1<f32>) -> f32 {
        const N_BINS: usize = 10;
        let n = probs.len();
        
        let mut bins = vec![Vec::new(); N_BINS];
        
        // Assign to bins
        for i in 0..n {
            let bin_idx = ((probs[i] * N_BINS as f32) as usize).min(N_BINS - 1);
            bins[bin_idx].push((probs[i], outcomes[i]));
        }
        
        // Calculate ECE
        let mut ece = 0.0;
        for bin in &bins {
            if bin.is_empty() {
                continue;
            }
            
            let bin_size = bin.len() as f32;
            let avg_prob: f32 = bin.iter().map(|(p, _)| p).sum::<f32>() / bin_size;
            let avg_outcome: f32 = bin.iter().map(|(_, o)| o).sum::<f32>() / bin_size;
            
            ece += (bin_size / n as f32) * (avg_prob - avg_outcome).abs();
        }
        
        ece
    }
    
    /// Detect current market regime
    pub fn detect_regime(&self, volatility: f32, trend_strength: f32) -> MarketRegime {
        if volatility > 0.05 {
            MarketRegime::Crisis
        } else if trend_strength > 0.7 {
            MarketRegime::Trending
        } else if trend_strength > 0.4 {
            MarketRegime::Breakout
        } else {
            MarketRegime::RangeBound
        }
    }
}

/// Isotonic Regression Implementation
/// Monotonic function fitting using Pool Adjacent Violators Algorithm (PAVA)
#[derive(Debug, Clone)]
struct IsotonicRegression {
    x_points: Vec<f32>,
    y_points: Vec<f32>,
    fitted: bool,
}

impl IsotonicRegression {
    fn new() -> Self {
        Self {
            x_points: Vec::new(),
            y_points: Vec::new(),
            fitted: false,
        }
    }
    
    /// Fit isotonic regression with regularization
    fn fit_with_regularization(
        &mut self,
        x: &Array1<f32>,
        y: &Array1<f32>,
        lambda: f32,
    ) -> Result<(), CalibrationError> {
        let n = x.len();
        
        // Sort by x values
        let mut pairs: Vec<(f32, f32)> = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi, yi))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Apply PAVA algorithm with regularization
        let mut isotonic_y = pairs.iter().map(|(_, y)| *y).collect::<Vec<f32>>();
        
        // Pool Adjacent Violators Algorithm
        let mut i = 0;
        while i < n - 1 {
            if isotonic_y[i] > isotonic_y[i + 1] {
                // Violates monotonicity, pool values
                let mut pool_start = i;
                let mut pool_end = i + 1;
                let mut pool_sum = isotonic_y[i] + isotonic_y[i + 1];
                
                // Extend pool while violations exist
                while pool_end < n - 1 && isotonic_y[pool_start] > isotonic_y[pool_end + 1] {
                    pool_end += 1;
                    pool_sum += isotonic_y[pool_end];
                }
                
                // Apply regularization (shrink towards prior)
                let pool_size = (pool_end - pool_start + 1) as f32;
                let pool_mean = pool_sum / pool_size;
                let regularized_mean = pool_mean * (1.0 - lambda) + 0.5 * lambda;
                
                // Set pooled values
                for j in pool_start..=pool_end {
                    isotonic_y[j] = regularized_mean;
                }
                
                // Restart from beginning of pool
                i = pool_start.saturating_sub(1);
            } else {
                i += 1;
            }
        }
        
        // Store fitted values
        self.x_points = pairs.iter().map(|(x, _)| *x).collect();
        self.y_points = isotonic_y;
        self.fitted = true;
        
        Ok(())
    }
    
    /// Transform single probability
    fn transform_single(&self, x: f32) -> f32 {
        if !self.fitted || self.x_points.is_empty() {
            return x;
        }
        
        // Binary search for interpolation point
        let n = self.x_points.len();
        
        if x <= self.x_points[0] {
            return self.y_points[0];
        }
        if x >= self.x_points[n - 1] {
            return self.y_points[n - 1];
        }
        
        // Find interpolation interval
        let mut left = 0;
        let mut right = n - 1;
        
        while left < right - 1 {
            let mid = (left + right) / 2;
            if self.x_points[mid] <= x {
                left = mid;
            } else {
                right = mid;
            }
        }
        
        // Linear interpolation
        let x0 = self.x_points[left];
        let x1 = self.x_points[right];
        let y0 = self.y_points[left];
        let y1 = self.y_points[right];
        
        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
    
    /// Transform batch of probabilities
    fn transform(&self, x: &Array1<f32>) -> Result<Array1<f32>, CalibrationError> {
        Ok(Array1::from_vec(
            x.iter().map(|&xi| self.transform_single(xi)).collect()
        ))
    }
}

/// Calibration Reliability Diagram
/// Visual tool to assess calibration quality
pub struct ReliabilityDiagram {
    n_bins: usize,
    bins: Vec<CalibrationBin>,
}

#[derive(Debug, Clone)]
struct CalibrationBin {
    min_prob: f32,
    max_prob: f32,
    avg_predicted: f32,
    avg_actual: f32,
    count: usize,
}

impl ReliabilityDiagram {
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            bins: Vec::new(),
        }
    }
    
    pub fn compute(&mut self, predicted: &Array1<f32>, actual: &Array1<f32>) {
        self.bins.clear();
        
        for i in 0..self.n_bins {
            let min_prob = i as f32 / self.n_bins as f32;
            let max_prob = (i + 1) as f32 / self.n_bins as f32;
            
            let mut sum_pred = 0.0;
            let mut sum_actual = 0.0;
            let mut count = 0;
            
            for j in 0..predicted.len() {
                if predicted[j] >= min_prob && predicted[j] < max_prob {
                    sum_pred += predicted[j];
                    sum_actual += actual[j];
                    count += 1;
                }
            }
            
            if count > 0 {
                self.bins.push(CalibrationBin {
                    min_prob,
                    max_prob,
                    avg_predicted: sum_pred / count as f32,
                    avg_actual: sum_actual / count as f32,
                    count,
                });
            }
        }
    }
    
    pub fn print_summary(&self) {
        println!("Reliability Diagram:");
        println!("Bin Range    | Avg Predicted | Avg Actual | Count | Gap");
        println!("-------------|---------------|------------|-------|------");
        
        for bin in &self.bins {
            let gap = (bin.avg_predicted - bin.avg_actual).abs();
            println!(
                "[{:.2}-{:.2}] | {:.3}         | {:.3}      | {:5} | {:.3}",
                bin.min_prob, bin.max_prob,
                bin.avg_predicted, bin.avg_actual,
                bin.count, gap
            );
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("Dimension mismatch between probabilities and outcomes")]
    DimensionMismatch,
    
    #[error("Insufficient data: required {required}, provided {provided}")]
    InsufficientData { required: usize, provided: usize },
    
    #[error("Calibration failed")]
    CalibrationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    
    #[test]
    fn test_isotonic_regression() {
        // Create non-monotonic data
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let y = Array1::from_vec(vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6, 0.8, 0.7, 0.9]);
        
        let mut iso = IsotonicRegression::new();
        iso.fit_with_regularization(&x, &y, 0.01).unwrap();
        
        // Check monotonicity
        for i in 1..iso.y_points.len() {
            assert!(
                iso.y_points[i] >= iso.y_points[i-1],
                "Isotonic regression should be monotonic"
            );
        }
    }
    
    #[test]
    fn test_calibration_prevents_overconfidence() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Create overconfident predictions
        let n = 1000;
        let mut rng = thread_rng();
        let mut raw_probs = Vec::new();
        let mut outcomes = Vec::new();
        
        for _ in 0..n {
            let p = rng.gen_range(0.0..1.0);
            raw_probs.push(p * 0.5 + 0.5);  // Skew towards high confidence
            outcomes.push(if rng.gen::<f32>() < p { 1.0 } else { 0.0 });
        }
        
        let raw_probs = Array1::from(raw_probs);
        let outcomes = Array1::from(outcomes);
        
        // Calibrate
        calibrator.calibrate(&raw_probs, &outcomes, MarketRegime::Trending).unwrap();
        
        // Transform
        let calibrated = calibrator.transform_batch(&raw_probs, MarketRegime::Trending).unwrap();
        
        // Check Brier score improved
        let brier_before = calibrator.calculate_brier_score(&raw_probs, &outcomes);
        let brier_after = calibrator.calculate_brier_score(&calibrated, &outcomes);
        
        assert!(
            brier_after <= brier_before,
            "Calibration should improve or maintain Brier score"
        );
    }
    
    #[test]
    fn test_regime_specific_calibration() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Different regimes should have different calibrations
        let probs = Array1::from_vec(vec![0.3; 1000]);
        let trending_outcomes = Array1::from_vec(vec![0.4; 1000]);  // Trending underestimates
        let crisis_outcomes = Array1::from_vec(vec![0.1; 1000]);    // Crisis overestimates
        
        calibrator.calibrate(&probs, &trending_outcomes, MarketRegime::Trending).unwrap();
        calibrator.calibrate(&probs, &crisis_outcomes, MarketRegime::Crisis).unwrap();
        
        let trending_cal = calibrator.transform(0.3, MarketRegime::Trending);
        let crisis_cal = calibrator.transform(0.3, MarketRegime::Crisis);
        
        assert!(trending_cal > 0.3, "Trending should increase low probabilities");
        assert!(crisis_cal < 0.3, "Crisis should decrease probabilities");
    }
    
    #[test]
    fn test_reliability_diagram() {
        let predicted = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
        let actual = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        
        let mut diagram = ReliabilityDiagram::new(5);
        diagram.compute(&predicted, &actual);
        
        assert!(!diagram.bins.is_empty());
        diagram.print_summary();
    }
    
    #[test]
    fn test_brier_score_calculation() {
        let calibrator = IsotonicCalibrator::new();
        
        // Perfect predictions
        let perfect_probs = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let outcomes = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let perfect_brier = calibrator.calculate_brier_score(&perfect_probs, &outcomes);
        assert_eq!(perfect_brier, 0.0);
        
        // Random predictions
        let random_probs = Array1::from_vec(vec![0.5; 4]);
        let random_brier = calibrator.calculate_brier_score(&random_probs, &outcomes);
        assert_eq!(random_brier, 0.25);
    }
}