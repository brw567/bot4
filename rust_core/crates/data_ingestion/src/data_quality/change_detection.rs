// Change Point Detection Algorithms
// Based on Killick et al. (2012) PELT, Page (1954) CUSUM
//
// Detects regime changes, structural breaks, and anomalies in time series

use std::collections::VecDeque;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use super::DataBatch;

#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct ChangeDetectionConfig {
    pub algorithm: DetectionAlgorithm,
    pub window_size: usize,
    pub threshold: f64,
    pub min_segment_length: usize,
}

impl Default for ChangeDetectionConfig {
    fn default() -> Self {
        Self {
            algorithm: DetectionAlgorithm::PELT,
            window_size: 100,
            threshold: 0.05,
            min_segment_length: 10,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
/// TODO: Add docs
pub enum DetectionAlgorithm {
    CUSUM,  // Cumulative sum
    PELT,   // Pruned Exact Linear Time
    BayesianOnline,
}

/// TODO: Add docs
pub struct ChangeDetector {
    config: ChangeDetectionConfig,
    buffer: VecDeque<f64>,
}

impl ChangeDetector {
    pub fn new(config: ChangeDetectionConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::with_capacity(1000),
        }
    }

    pub async fn detect(&self, data: &DataBatch) -> Result<Option<ChangePoint>> {
        if data.values.is_empty() {
            return Ok(None);
        }

        match self.config.algorithm {
            DetectionAlgorithm::CUSUM => self.detect_cusum(&data.values),
            DetectionAlgorithm::PELT => self.detect_pelt(&data.values),
            DetectionAlgorithm::BayesianOnline => self.detect_bayesian(&data.values),
        }
    }

    fn detect_cusum(&self, values: &[f64]) -> Result<Option<ChangePoint>> {
        if values.len() < self.config.window_size {
            return Ok(None);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut cusum_pos = 0.0;
        let mut cusum_neg = 0.0;
        
        for (i, &value) in values.iter().enumerate() {
            cusum_pos = (cusum_pos + value - mean - self.config.threshold).max(0.0);
            cusum_neg = (cusum_neg - value + mean - self.config.threshold).max(0.0);
            
            if cusum_pos > self.config.threshold * 10.0 || cusum_neg > self.config.threshold * 10.0 {
                return Ok(Some(ChangePoint {
                    timestamp: Utc::now(),
                    index: i,
                    confidence: 0.95,
                    change_type: ChangeType::LevelShift,
                    magnitude: (cusum_pos.max(cusum_neg) / mean).abs(),
                }));
            }
        }
        
        Ok(None)
    }

    fn detect_pelt(&self, values: &[f64]) -> Result<Option<ChangePoint>> {
        // Simplified PELT implementation
        if values.len() < 2 * self.config.min_segment_length {
            return Ok(None);
        }

        let n = values.len();
        let mut f = vec![0.0; n + 1];
        let mut changepoints = vec![0; n + 1];
        
        for i in self.config.min_segment_length..=n {
            let mut min_cost = f64::MAX;
            let mut best_change = 0;
            
            for j in 0..=(i - self.config.min_segment_length) {
                let cost = self.segment_cost(&values[j..i]) + f[j];
                if cost < min_cost {
                    min_cost = cost;
                    best_change = j;
                }
            }
            
            f[i] = min_cost;
            changepoints[i] = best_change;
        }
        
        // Find most recent change point
        if changepoints[n] > 0 && changepoints[n] < n {
            return Ok(Some(ChangePoint {
                timestamp: Utc::now(),
                index: changepoints[n],
                confidence: 0.9,
                change_type: ChangeType::VarianceChange,
                magnitude: 1.0,
            }));
        }
        
        Ok(None)
    }

    fn detect_bayesian(&self, values: &[f64]) -> Result<Option<ChangePoint>> {
        // Simplified Bayesian changepoint detection
        if values.len() < self.config.window_size {
            return Ok(None);
        }

        let mut run_length_probs = vec![1.0];
        let hazard_rate = 1.0 / self.config.window_size as f64;
        
        for (i, &value) in values.iter().enumerate() {
            let predictive = self.student_t_pdf(value, i as f64, 1.0, 1.0);
            let growth_probs: Vec<f64> = run_length_probs.iter()
                .map(|&p| p * predictive * (1.0 - hazard_rate))
                .collect();
            
            let cp_prob = run_length_probs.iter().sum::<f64>() * predictive * hazard_rate;
            
            if cp_prob > self.config.threshold {
                return Ok(Some(ChangePoint {
                    timestamp: Utc::now(),
                    index: i,
                    confidence: cp_prob.min(1.0),
                    change_type: ChangeType::DistributionChange,
                    magnitude: 1.0,
                }));
            }
            
            run_length_probs = growth_probs;
            run_length_probs.insert(0, cp_prob);
        }
        
        Ok(None)
    }

    fn segment_cost(&self, segment: &[f64]) -> f64 {
        if segment.is_empty() {
            return 0.0;
        }
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
    }

    fn student_t_pdf(&self, x: f64, df: f64, loc: f64, scale: f64) -> f64 {
        let z = (x - loc) / scale;
        let norm = ((df + 1.0) / 2.0).ln() - (df / 2.0).ln() - 0.5 * std::f64::consts::PI.ln() - df.sqrt().ln();
        norm.exp() * (1.0 + z * z / df).powf(-(df + 1.0) / 2.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ChangePoint {
    pub timestamp: DateTime<Utc>,
    pub index: usize,
    pub confidence: f64,
    pub change_type: ChangeType,
    pub magnitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum ChangeType {
    LevelShift,
    VarianceChange,
    DistributionChange,
    TrendChange,
}