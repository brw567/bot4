// Isotonic Regression for Probability Calibration - Full Implementation
// Team: Morgan (ML) + Quinn (Risk) + Sam (Architecture)
// References:
// - Zadrozny & Elkan (2002) "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
// - Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities with Supervised Learning"
// - NO SIMPLIFICATIONS - CRITICAL FOR PREVENTING OVERCONFIDENCE

use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Isotonic Regression Calibrator
/// Maps predicted probabilities to calibrated probabilities
/// Ensures monotonic relationship while correcting systematic biases
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct IsotonicCalibrator {
    // Calibration mapping points
    thresholds: Vec<f64>,     // Input probability thresholds
    calibrated: Vec<f64>,     // Corresponding calibrated probabilities
    
    // Statistics for validation
    samples_used: usize,
    brier_score_before: f64,  // Brier score before calibration
    brier_score_after: f64,   // Brier score after calibration
    ece_before: f64,          // Expected Calibration Error before
    ece_after: f64,           // Expected Calibration Error after
    
    // Regime-specific calibrators
    regime_calibrators: Vec<(MarketRegime, Vec<f64>, Vec<f64>)>,
    current_regime: MarketRegime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// TODO: Add docs
pub enum MarketRegime {
    Normal,
    Volatile,
    Crisis,
    Trending,
    RangeBound,
}

impl IsotonicCalibrator {
    /// Create new calibrator
    pub fn new() -> Self {
        Self {
            thresholds: vec![0.0, 1.0],
            calibrated: vec![0.0, 1.0],
            samples_used: 0,
            brier_score_before: 0.0,
            brier_score_after: 0.0,
            ece_before: 0.0,
            ece_after: 0.0,
            regime_calibrators: Vec::new(),
            current_regime: MarketRegime::Normal,
        }
    }
    
    /// Fit isotonic regression on calibration data
    /// predictions: ML model probability predictions
    /// actuals: Binary outcomes (0 or 1)
    pub fn fit(&mut self, predictions: &[f64], actuals: &[bool]) -> Result<()> {
        if predictions.len() != actuals.len() {
            return Err(anyhow::anyhow!("Predictions and actuals must have same length"));
        }
        
        if predictions.len() < 10 {
            return Err(anyhow::anyhow!("Need at least 10 samples for calibration"));
        }
        
        self.samples_used = predictions.len();
        
        // Calculate initial Brier score
        self.brier_score_before = self.calculate_brier_score(predictions, actuals);
        self.ece_before = self.calculate_ece(predictions, actuals);
        
        // Prepare data for isotonic regression
        let mut data: Vec<(f64, f64, f64)> = predictions.iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| (pred, if actual { 1.0 } else { 0.0 }, 1.0))
            .collect();
        
        // Sort by prediction value
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        
        // Apply Pool Adjacent Violators Algorithm (PAVA)
        let (thresholds, calibrated) = self.pava_algorithm(&data)?;
        
        self.thresholds = thresholds;
        self.calibrated = calibrated;
        
        // Calculate calibrated scores for validation
        let calibrated_preds: Vec<f64> = predictions.iter()
            .map(|&p| self.transform_single(p))
            .collect();
        
        self.brier_score_after = self.calculate_brier_score(&calibrated_preds, actuals);
        self.ece_after = self.calculate_ece(&calibrated_preds, actuals);
        
        log::info!(
            "Isotonic calibration fitted: {} samples, Brier: {:.4} -> {:.4}, ECE: {:.4} -> {:.4}",
            self.samples_used, 
            self.brier_score_before, 
            self.brier_score_after,
            self.ece_before,
            self.ece_after
        );
        
        Ok(())
    }
    
    /// Pool Adjacent Violators Algorithm (PAVA)
    /// Core algorithm for isotonic regression
    fn pava_algorithm(&self, data: &[(f64, f64, f64)]) -> Result<(Vec<f64>, Vec<f64>)> {
        if data.is_empty() {
            return Ok((vec![0.0, 1.0], vec![0.0, 1.0]));
        }
        
        // Initialize blocks
        let mut blocks: Vec<Block> = data.iter()
            .map(|&(x, y, w)| Block {
                x_min: x,
                x_max: x,
                y_sum: y * w,
                w_sum: w,
                y_mean: y,
            })
            .collect();
        
        // Pool adjacent violators
        let mut merged = true;
        while merged {
            merged = false;
            let mut i = 0;
            
            while i < blocks.len() - 1 {
                // Check for violation of monotonicity
                if blocks[i].y_mean > blocks[i + 1].y_mean {
                    // Merge blocks
                    let merged_block = Block {
                        x_min: blocks[i].x_min,
                        x_max: blocks[i + 1].x_max,
                        y_sum: blocks[i].y_sum + blocks[i + 1].y_sum,
                        w_sum: blocks[i].w_sum + blocks[i + 1].w_sum,
                        y_mean: (blocks[i].y_sum + blocks[i + 1].y_sum) / 
                                (blocks[i].w_sum + blocks[i + 1].w_sum),
                    };
                    
                    blocks[i] = merged_block;
                    blocks.remove(i + 1);
                    merged = true;
                } else {
                    i += 1;
                }
            }
        }
        
        // Extract thresholds and calibrated values
        let mut thresholds = Vec::new();
        let mut calibrated = Vec::new();
        
        // Add starting point
        if blocks[0].x_min > 0.0 {
            thresholds.push(0.0);
            calibrated.push(blocks[0].y_mean.min(blocks[0].x_min));
        }
        
        for block in &blocks {
            thresholds.push(block.x_min);
            calibrated.push(block.y_mean);
            
            if block.x_min != block.x_max {
                thresholds.push(block.x_max);
                calibrated.push(block.y_mean);
            }
        }
        
        // Add ending point
        if thresholds.last().unwrap() < &1.0 {
            thresholds.push(1.0);
            let last_cal = *calibrated.last().unwrap();
            let last_thresh = *thresholds.last().unwrap();
            calibrated.push(last_cal.max(last_thresh));
        }
        
        Ok((thresholds, calibrated))
    }
    
    /// Transform a single probability
    pub fn transform(&self, probability: f64, regime: MarketRegime) -> f64 {
        // Use regime-specific calibrator if available
        for (r, thresh, calib) in &self.regime_calibrators {
            if *r == regime {
                return self.interpolate(probability, thresh, calib);
            }
        }
        
        // Fall back to global calibrator
        self.transform_single(probability)
    }
    
    /// Transform using global calibrator
    fn transform_single(&self, probability: f64) -> f64 {
        self.interpolate(probability, &self.thresholds, &self.calibrated)
    }
    
    /// Linear interpolation between calibration points
    fn interpolate(&self, x: f64, thresholds: &[f64], calibrated: &[f64]) -> f64 {
        // Clamp to [0, 1]
        let x = x.max(0.0).min(1.0);
        
        // Find the appropriate interval
        for i in 0..thresholds.len() - 1 {
            if x >= thresholds[i] && x <= thresholds[i + 1] {
                // Linear interpolation
                let t = if thresholds[i + 1] - thresholds[i] > 1e-10 {
                    (x - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                } else {
                    0.5
                };
                
                return calibrated[i] * (1.0 - t) + calibrated[i + 1] * t;
            }
        }
        
        // Shouldn't reach here, but handle edge cases
        if x <= thresholds[0] {
            calibrated[0]
        } else {
            *calibrated.last().unwrap()
        }
    }
    
    /// Fit regime-specific calibrators
    pub fn fit_regime(&mut self, 
                      regime: MarketRegime,
                      predictions: &[f64], 
                      actuals: &[bool]) -> Result<()> {
        
        if predictions.len() < 10 {
            return Err(anyhow::anyhow!("Need at least 10 samples per regime"));
        }
        
        // Prepare data
        let mut data: Vec<(f64, f64, f64)> = predictions.iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| (pred, if actual { 1.0 } else { 0.0 }, 1.0))
            .collect();
        
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        
        // Apply PAVA
        let (thresholds, calibrated) = self.pava_algorithm(&data)?;
        
        // Store or update regime calibrator
        let mut found = false;
        for (r, thresh, calib) in &mut self.regime_calibrators {
            if *r == regime {
                *thresh = thresholds.clone();
                *calib = calibrated.clone();
                found = true;
                break;
            }
        }
        
        if !found {
            self.regime_calibrators.push((regime, thresholds, calibrated));
        }
        
        log::info!("Fitted isotonic calibration for {:?} regime", regime);
        
        Ok(())
    }
    
    /// Calculate Brier score (lower is better)
    fn calculate_brier_score(&self, predictions: &[f64], actuals: &[bool]) -> f64 {
        let n = predictions.len() as f64;
        
        predictions.iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| {
                let y = if actual { 1.0 } else { 0.0 };
                (pred - y).powi(2)
            })
            .sum::<f64>() / n
    }
    
    /// Calculate Expected Calibration Error (ECE)
    fn calculate_ece(&self, predictions: &[f64], actuals: &[bool]) -> f64 {
        const N_BINS: usize = 10;
        let n = predictions.len() as f64;
        
        let mut bins = vec![ECEBin::default(); N_BINS];
        
        // Assign samples to bins
        for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
            let bin_idx = ((pred * N_BINS as f64).floor() as usize).min(N_BINS - 1);
            bins[bin_idx].count += 1;
            bins[bin_idx].sum_confidence += pred;
            if actual {
                bins[bin_idx].sum_positive += 1.0;
            }
        }
        
        // Calculate ECE
        let mut ece = 0.0;
        for bin in &bins {
            if bin.count > 0 {
                let avg_confidence = bin.sum_confidence / bin.count as f64;
                let accuracy = bin.sum_positive / bin.count as f64;
                let weight = bin.count as f64 / n;
                ece += weight * (avg_confidence - accuracy).abs();
            }
        }
        
        ece
    }
    
    /// Get calibration curve for plotting
    pub fn get_calibration_curve(&self, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x_points = Vec::with_capacity(n_points);
        let mut y_points = Vec::with_capacity(n_points);
        
        for i in 0..n_points {
            let x = i as f64 / (n_points - 1) as f64;
            let y = self.transform_single(x);
            x_points.push(x);
            y_points.push(y);
        }
        
        (x_points, y_points)
    }
    
    /// Reliability diagram data
    pub fn reliability_diagram(&self, predictions: &[f64], actuals: &[bool], n_bins: usize) 
        -> Vec<ReliabilityBin> {
        
        let mut bins = vec![ReliabilityBin::default(); n_bins];
        
        // Assign to bins
        for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
            let bin_idx = ((pred * n_bins as f64).floor() as usize).min(n_bins - 1);
            bins[bin_idx].predictions.push(pred);
            bins[bin_idx].actuals.push(actual);
        }
        
        // Calculate statistics
        for bin in &mut bins {
            if !bin.predictions.is_empty() {
                bin.mean_predicted = bin.predictions.iter().sum::<f64>() / 
                                    bin.predictions.len() as f64;
                bin.fraction_positive = bin.actuals.iter().filter(|&&a| a).count() as f64 / 
                                       bin.actuals.len() as f64;
                bin.count = bin.predictions.len();
            }
        }
        
        bins
    }
    
    /// Check if calibrator is fitted
    pub fn is_fitted(&self) -> bool {
        self.samples_used > 0 && self.thresholds.len() > 2
    }
    
    /// Get calibration metrics
    pub fn get_metrics(&self) -> CalibrationMetrics {
        CalibrationMetrics {
            samples_used: self.samples_used,
            brier_improvement: self.brier_score_before - self.brier_score_after,
            ece_improvement: self.ece_before - self.ece_after,
            n_segments: self.thresholds.len() - 1,
            regime_calibrators: self.regime_calibrators.len(),
        }
    }
    
    /// Set current market regime
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.current_regime = regime;
    }
}

/// Block for PAVA algorithm
#[derive(Debug, Clone)]
struct Block {
    x_min: f64,
    x_max: f64,
    y_sum: f64,
    w_sum: f64,
    y_mean: f64,
}

/// ECE calculation bin
#[derive(Debug, Clone, Default)]
struct ECEBin {
    count: usize,
    sum_confidence: f64,
    sum_positive: f64,
}

/// Reliability diagram bin
#[derive(Debug, Clone, Default)]
/// TODO: Add docs
pub struct ReliabilityBin {
    pub predictions: Vec<f64>,
    pub actuals: Vec<bool>,
    pub mean_predicted: f64,
    pub fraction_positive: f64,
    pub count: usize,
}

/// Calibration metrics
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct CalibrationMetrics {
    pub samples_used: usize,
    pub brier_improvement: f64,
    pub ece_improvement: f64,
    pub n_segments: usize,
    pub regime_calibrators: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_isotonic_basic() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Create synthetic miscalibrated data - need at least 10 samples
        let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let actuals = vec![false, false, false, true, false, true, true, true, true, true];
        
        calibrator.fit(&predictions, &actuals).unwrap();
        
        // Check monotonicity
        for i in 1..calibrator.calibrated.len() {
            assert!(calibrator.calibrated[i] >= calibrator.calibrated[i-1],
                   "Calibration must be monotonic");
        }
    }
    
    #[test]
    fn test_perfect_calibration() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Perfectly calibrated data
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        
        for i in 0..100 {
            let p = i as f64 / 100.0;
            predictions.push(p);
            // Actual matches probability
            actuals.push(rand::random::<f64>() < p);
        }
        
        calibrator.fit(&predictions, &actuals).unwrap();
        
        // Should be close to identity (with higher tolerance for random data)
        for p in [0.2, 0.5, 0.8] {
            let calibrated = calibrator.transform_single(p);
            // Use 0.3 tolerance since we're using random data
            assert!((calibrated - p).abs() < 0.3, 
                   "Well-calibrated should be near identity: p={}, cal={}", p, calibrated);
        }
    }
    
    #[test]
    fn test_overconfident_model() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Overconfident model (predicts high but accuracy is lower)
        let predictions = vec![0.9; 100];
        let mut actuals = vec![false; 100];
        // Only 60% are actually positive
        for i in 0..60 {
            actuals[i] = true;
        }
        
        calibrator.fit(&predictions, &actuals).unwrap();
        
        // Should reduce confidence
        let calibrated = calibrator.transform_single(0.9);
        assert!(calibrated < 0.9, "Should reduce overconfident predictions");
        assert!((calibrated - 0.6).abs() < 0.1, "Should be close to actual rate");
    }
    
    #[test]
    fn test_regime_specific() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Fit normal regime
        let normal_preds = vec![0.5; 50];
        let normal_actuals = vec![true; 25].into_iter()
            .chain(vec![false; 25])
            .collect::<Vec<_>>();
        
        calibrator.fit_regime(MarketRegime::Normal, &normal_preds, &normal_actuals).unwrap();
        
        // Fit crisis regime (more conservative)
        let crisis_preds = vec![0.5; 50];
        let crisis_actuals = vec![true; 10].into_iter()
            .chain(vec![false; 40])
            .collect::<Vec<_>>();
        
        calibrator.fit_regime(MarketRegime::Crisis, &crisis_preds, &crisis_actuals).unwrap();
        
        // Normal regime should maintain 0.5
        let normal_cal = calibrator.transform(0.5, MarketRegime::Normal);
        assert!((normal_cal - 0.5).abs() < 0.1);
        
        // Crisis regime should reduce to ~0.2
        let crisis_cal = calibrator.transform(0.5, MarketRegime::Crisis);
        assert!(crisis_cal < 0.3);
    }
    
    #[test]
    fn test_brier_score_improvement() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Create miscalibrated predictions
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        
        for i in 0..100 {
            // Systematic overconfidence
            predictions.push(0.8 + (i as f64 / 500.0));
            actuals.push(i < 50); // Only 50% positive
        }
        
        calibrator.fit(&predictions, &actuals).unwrap();
        
        // Brier score should improve
        assert!(calibrator.brier_score_after < calibrator.brier_score_before,
               "Calibration should improve Brier score");
        
        // ECE should improve
        assert!(calibrator.ece_after < calibrator.ece_before,
               "Calibration should improve ECE");
    }
}