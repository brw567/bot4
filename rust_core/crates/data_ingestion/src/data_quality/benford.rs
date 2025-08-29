// Benford's Law Validation for Anomaly Detection
// Based on Benford (1938) and Nigrini (2012) - forensic accounting fraud detection
// 
// Theory: In naturally occurring datasets, the leading digit follows:
// P(d) = log10(1 + 1/d) where d is the digit [1-9]
// This gives: 1=30.1%, 2=17.6%, 3=12.5%, 4=9.7%, 5=7.9%, 6=6.7%, 7=5.8%, 8=5.1%, 9=4.6%
//
// Applications in trading:
// - Detect manipulated price/volume data
// - Identify wash trading patterns
// - Spot artificial market making
// - Find data feed corruption

use std::collections::HashMap;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, warn};

use super::DataBatch;

/// Benford's Law configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct BenfordConfig {
    pub min_sample_size: usize,      // Minimum data points for validity
    pub chi_squared_threshold: f64,  // Critical value for goodness-of-fit
    pub deviation_threshold: f64,    // Max allowed deviation per digit
    pub enable_second_digit: bool,   // Also check second digit distribution
    pub enable_mantissa: bool,       // Check mantissa distribution for floats
}

impl Default for BenfordConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 100,
            chi_squared_threshold: 15.507,  // 95% confidence for 8 DOF
            deviation_threshold: 0.05,      // 5% deviation allowed
            enable_second_digit: true,
            enable_mantissa: true,
        }
    }
}

/// Benford's Law validator
/// TODO: Add docs
pub struct BenfordValidator {
    config: BenfordConfig,
    
    // Theoretical Benford distributions
    first_digit_dist: HashMap<u8, f64>,
    second_digit_dist: HashMap<u8, f64>,
    first_two_digits_dist: HashMap<u8, f64>,
    
    // Historical deviations for adaptive thresholds
    historical_deviations: Vec<f64>,
}

impl BenfordValidator {
    /// Create new Benford validator
    pub fn new(config: BenfordConfig) -> Self {
        // Calculate theoretical distributions
        let mut first_digit_dist = HashMap::new();
        let mut second_digit_dist = HashMap::new();
        let mut first_two_digits_dist = HashMap::new();
        
        // First digit: P(d) = log10(1 + 1/d)
        for d in 1..=9 {
            let prob = ((1.0 + 1.0 / d as f64).log10());
            first_digit_dist.insert(d as u8, prob);
        }
        
        // Second digit: P(d) = sum(log10(1 + 1/(10*k + d))) for k=1..9
        for d in 0..=9 {
            let mut prob = 0.0;
            for k in 1..=9 {
                prob += ((1.0 + 1.0 / (10 * k + d) as f64).log10());
            }
            second_digit_dist.insert(d as u8, prob);
        }
        
        // First two digits: P(d1d2) = log10(1 + 1/d1d2)
        for d in 10..=99 {
            let prob = ((1.0 + 1.0 / d as f64).log10());
            first_two_digits_dist.insert(d as u8, prob);
        }
        
        Self {
            config,
            first_digit_dist,
            second_digit_dist,
            first_two_digits_dist,
            historical_deviations: Vec::with_capacity(1000),
        }
    }
    
    /// Validate data batch against Benford's Law
    pub async fn validate(&self, data: &DataBatch) -> Result<Option<BenfordAnomaly>> {
        if data.values.len() < self.config.min_sample_size {
            debug!("Insufficient data for Benford validation: {} < {}", 
                   data.values.len(), self.config.min_sample_size);
            return Ok(None);
        }
        
        // Extract digits from data
        let digits = self.extract_digits(&data.values)?;
        
        // Test first digit distribution
        let first_digit_test = self.test_first_digit(&digits)?;
        if first_digit_test.chi_squared > self.config.chi_squared_threshold {
            return Ok(Some(BenfordAnomaly {
                timestamp: Utc::now(),
                symbol: data.symbol.clone(),
                anomaly_type: AnomalyType::FirstDigit,
                chi_squared: first_digit_test.chi_squared,
                deviation: first_digit_test.max_deviation,
                suspicious_digits: first_digit_test.suspicious_digits,
                confidence: self.calculate_confidence(first_digit_test.chi_squared),
            }));
        }
        
        // Test second digit distribution if enabled
        if self.config.enable_second_digit {
            let second_digit_test = self.test_second_digit(&digits)?;
            if second_digit_test.chi_squared > self.config.chi_squared_threshold {
                return Ok(Some(BenfordAnomaly {
                    timestamp: Utc::now(),
                    symbol: data.symbol.clone(),
                    anomaly_type: AnomalyType::SecondDigit,
                    chi_squared: second_digit_test.chi_squared,
                    deviation: second_digit_test.max_deviation,
                    suspicious_digits: second_digit_test.suspicious_digits,
                    confidence: self.calculate_confidence(second_digit_test.chi_squared),
                }));
            }
        }
        
        // Test mantissa distribution for floating point data
        if self.config.enable_mantissa && data.values.iter().any(|v| v.fract() != 0.0) {
            let mantissa_test = self.test_mantissa(&data.values)?;
            if mantissa_test > self.config.deviation_threshold {
                return Ok(Some(BenfordAnomaly {
                    timestamp: Utc::now(),
                    symbol: data.symbol.clone(),
                    anomaly_type: AnomalyType::Mantissa,
                    chi_squared: 0.0,  // Not applicable for mantissa
                    deviation: mantissa_test,
                    suspicious_digits: Vec::new(),
                    confidence: 0.95,  // High confidence for mantissa anomalies
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Extract digits from numerical values
    fn extract_digits(&self, values: &[f64]) -> Result<DigitData> {
        let mut first_digits = Vec::new();
        let mut second_digits = Vec::new();
        let mut first_two_digits = Vec::new();
        
        for &value in values {
            let abs_value = value.abs();
            if abs_value == 0.0 {
                continue;
            }
            
            // Normalize to get leading digits
            let mut normalized = abs_value;
            while normalized >= 10.0 {
                normalized /= 10.0;
            }
            while normalized < 1.0 {
                normalized *= 10.0;
            }
            
            let first_digit = normalized.floor() as u8;
            first_digits.push(first_digit);
            
            let second_digit = ((normalized * 10.0) % 10.0).floor() as u8;
            second_digits.push(second_digit);
            
            let first_two = (normalized * 10.0).floor() as u8;
            first_two_digits.push(first_two);
        }
        
        Ok(DigitData {
            first_digits,
            second_digits,
            first_two_digits,
        })
    }
    
    /// Test first digit distribution
    fn test_first_digit(&self, digits: &DigitData) -> Result<TestResult> {
        let mut observed_freq = HashMap::new();
        let total = digits.first_digits.len() as f64;
        
        // Count occurrences
        for &digit in &digits.first_digits {
            *observed_freq.entry(digit).or_insert(0.0) += 1.0;
        }
        
        // Convert to proportions
        for freq in observed_freq.values_mut() {
            *freq /= total;
        }
        
        // Calculate chi-squared statistic
        let mut chi_squared = 0.0;
        let mut max_deviation = 0.0;
        let mut suspicious_digits = Vec::new();
        
        for digit in 1..=9 {
            let expected = self.first_digit_dist[&digit];
            let observed = observed_freq.get(&digit).copied().unwrap_or(0.0);
            
            let deviation = (observed - expected).abs();
            if deviation > self.config.deviation_threshold {
                suspicious_digits.push(digit);
            }
            
            if deviation > max_deviation {
                max_deviation = deviation;
            }
            
            if expected > 0.0 {
                chi_squared += ((observed - expected).powi(2)) / expected * total;
            }
        }
        
        Ok(TestResult {
            chi_squared,
            max_deviation,
            suspicious_digits,
        })
    }
    
    /// Test second digit distribution
    fn test_second_digit(&self, digits: &DigitData) -> Result<TestResult> {
        let mut observed_freq = HashMap::new();
        let total = digits.second_digits.len() as f64;
        
        // Count occurrences
        for &digit in &digits.second_digits {
            *observed_freq.entry(digit).or_insert(0.0) += 1.0;
        }
        
        // Convert to proportions
        for freq in observed_freq.values_mut() {
            *freq /= total;
        }
        
        // Calculate chi-squared statistic
        let mut chi_squared = 0.0;
        let mut max_deviation = 0.0;
        let mut suspicious_digits = Vec::new();
        
        for digit in 0..=9 {
            let expected = self.second_digit_dist[&digit];
            let observed = observed_freq.get(&digit).copied().unwrap_or(0.0);
            
            let deviation = (observed - expected).abs();
            if deviation > self.config.deviation_threshold {
                suspicious_digits.push(digit);
            }
            
            if deviation > max_deviation {
                max_deviation = deviation;
            }
            
            if expected > 0.0 {
                chi_squared += ((observed - expected).powi(2)) / expected * total;
            }
        }
        
        Ok(TestResult {
            chi_squared,
            max_deviation,
            suspicious_digits,
        })
    }
    
    /// Test mantissa distribution (should be uniform for natural data)
    fn test_mantissa(&self, values: &[f64]) -> Result<f64> {
        let mantissas: Vec<f64> = values.iter()
            .filter(|&&v| v != 0.0)
            .map(|&v| {
                let log_val = v.abs().log10();
                log_val - log_val.floor()
            })
            .collect();
        
        if mantissas.is_empty() {
            return Ok(0.0);
        }
        
        // Kolmogorov-Smirnov test for uniformity
        let mut sorted = mantissas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len() as f64;
        let mut max_distance = 0.0;
        
        for (i, &value) in sorted.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = value;  // Uniform [0,1]
            let distance = (empirical_cdf - theoretical_cdf).abs();
            
            if distance > max_distance {
                max_distance = distance;
            }
        }
        
        // Critical value for KS test at 95% confidence
        let critical_value = 1.36 / n.sqrt();
        
        if max_distance > critical_value {
            warn!("Mantissa distribution anomaly detected: KS statistic = {:.4}", max_distance);
        }
        
        Ok(max_distance)
    }
    
    /// Calculate confidence level from chi-squared value
    fn calculate_confidence(&self, chi_squared: f64) -> f64 {
        // Using chi-squared distribution with 8 DOF (9 digits - 1)
        // Approximation for confidence level
        if chi_squared < 2.733 { 0.50 }
        else if chi_squared < 7.779 { 0.75 }
        else if chi_squared < 13.362 { 0.90 }
        else if chi_squared < 15.507 { 0.95 }
        else if chi_squared < 20.090 { 0.99 }
        else { 0.999 }
    }
}

/// Digit extraction results
struct DigitData {
    first_digits: Vec<u8>,
    second_digits: Vec<u8>,
    first_two_digits: Vec<u8>,
}

/// Test results
struct TestResult {
    chi_squared: f64,
    max_deviation: f64,
    suspicious_digits: Vec<u8>,
}

/// Benford's Law anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct BenfordAnomaly {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub anomaly_type: AnomalyType,
    pub chi_squared: f64,
    pub deviation: f64,
    pub suspicious_digits: Vec<u8>,
    pub confidence: f64,
}

/// Types of Benford anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AnomalyType {
    FirstDigit,
    SecondDigit,
    FirstTwoDigits,
    Mantissa,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_benford_validation() {
        let config = BenfordConfig::default();
        let validator = BenfordValidator::new(config);
        
        // Generate natural data following Benford's Law
        let mut natural_data = Vec::new();
        for i in 1..=1000 {
            natural_data.push((i as f64).powf(1.5));
        }
        
        let batch = DataBatch {
            symbol: "TEST".to_string(),
            data_type: super::super::DataType::Price,
            timestamp: Utc::now(),
            values: natural_data,
            source: "test".to_string(),
            metadata: None,
        };
        
        let result = validator.validate(&batch).await.unwrap();
        assert!(result.is_none(), "Natural data should not trigger anomaly");
    }
    
    #[tokio::test]
    async fn test_benford_anomaly_detection() {
        let config = BenfordConfig::default();
        let validator = BenfordValidator::new(config);
        
        // Generate manipulated data (all start with 5)
        let mut manipulated_data = Vec::new();
        for i in 0..200 {
            manipulated_data.push(5000.0 + i as f64);
        }
        
        let batch = DataBatch {
            symbol: "MANIP".to_string(),
            data_type: super::super::DataType::Volume,
            timestamp: Utc::now(),
            values: manipulated_data,
            source: "test".to_string(),
            metadata: None,
        };
        
        let result = validator.validate(&batch).await.unwrap();
        assert!(result.is_some(), "Manipulated data should trigger anomaly");
        
        if let Some(anomaly) = result {
            assert!(anomaly.suspicious_digits.contains(&5));
            assert!(anomaly.confidence > 0.9);
        }
    }
}