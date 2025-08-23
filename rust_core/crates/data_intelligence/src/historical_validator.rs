// HISTORICAL VALIDATOR - DEEP DIVE IMPLEMENTATION
// Team: FULL TEAM - NO SIMPLIFICATIONS!
// Alex: "Validate EVERY data point against historical patterns!"
// Avery: "Statistical anomaly detection with multiple validation layers"

use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Data anomaly detected: {0}")]
    AnomalyDetected(String),
    
    #[error("Historical data missing: {0}")]
    HistoricalDataMissing(String),
}

pub type Result<T> = std::result::Result<T, ValidationError>;

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub z_score_threshold: f64,
    pub min_historical_points: usize,
    pub enable_outlier_detection: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            z_score_threshold: 3.0,
            min_historical_points: 100,
            enable_outlier_detection: true,
        }
    }
}

/// Historical Validator - validates incoming data against historical patterns
pub struct HistoricalValidator {
    config: ValidationConfig,
    historical_data: VecDeque<DataPoint>,
}

#[derive(Debug, Clone)]
struct DataPoint {
    timestamp: DateTime<Utc>,
    value: Decimal,
}

impl HistoricalValidator {
    pub fn new(config: ValidationConfig) -> Result<Self> {
        Ok(Self {
            config,
            historical_data: VecDeque::with_capacity(10000),
        })
    }
    
    /// Validate a new data point
    pub fn validate(&mut self, value: Decimal, timestamp: DateTime<Utc>) -> Result<bool> {
        // Z-score validation
        if self.historical_data.len() >= self.config.min_historical_points {
            let z_score = self.calculate_z_score(value);
            if z_score.abs() > self.config.z_score_threshold {
                return Err(ValidationError::AnomalyDetected(
                    format!("Z-score {} exceeds threshold", z_score)
                ));
            }
        }
        
        // Add to historical data
        self.historical_data.push_back(DataPoint { timestamp, value });
        if self.historical_data.len() > 10000 {
            self.historical_data.pop_front();
        }
        
        Ok(true)
    }
    
    fn calculate_z_score(&self, value: Decimal) -> f64 {
        let values: Vec<f64> = self.historical_data.iter()
            .map(|p| p.value.to_f64().unwrap_or(0.0))
            .collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            (value.to_f64().unwrap_or(0.0) - mean) / std_dev
        } else {
            0.0
        }
    }
}