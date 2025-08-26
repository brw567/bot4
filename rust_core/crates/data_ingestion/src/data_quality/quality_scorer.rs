// Data Quality Scoring System
// Based on DAMA-DMBOK and ISO/IEC 25012 data quality standards
//
// Multi-dimensional quality assessment across 5 key dimensions

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    pub weights: QualityWeights,
    pub decay_rate: f64,  // Time-based quality decay
    pub min_acceptable_score: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weights: QualityWeights::default(),
            decay_rate: 0.99,  // 1% decay per hour
            min_acceptable_score: 0.8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QualityWeights {
    pub completeness: f64,
    pub accuracy: f64,
    pub consistency: f64,
    pub timeliness: f64,
    pub validity: f64,
}

impl Default for QualityWeights {
    fn default() -> Self {
        // Equal weights summing to 1.0
        Self {
            completeness: 0.2,
            accuracy: 0.25,  // Slightly higher for trading
            consistency: 0.2,
            timeliness: 0.25,  // Critical for HFT
            validity: 0.1,
        }
    }
}

pub struct QualityScorer {
    config: ScoringConfig,
}

impl QualityScorer {
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    pub async fn calculate_score(&self, metrics: QualityMetrics) -> Result<QualityScore> {
        let weights = &self.config.weights;
        
        // Calculate weighted score
        let raw_score = 
            metrics.completeness * weights.completeness +
            metrics.accuracy * weights.accuracy +
            metrics.consistency * weights.consistency +
            metrics.timeliness * weights.timeliness +
            metrics.validity * weights.validity;
        
        // Apply time decay if needed
        let final_score = raw_score.min(1.0).max(0.0);
        
        // Determine quality level
        let level = match final_score {
            s if s >= 0.95 => QualityLevel::Excellent,
            s if s >= 0.85 => QualityLevel::Good,
            s if s >= 0.70 => QualityLevel::Acceptable,
            s if s >= 0.50 => QualityLevel::Poor,
            _ => QualityLevel::Unacceptable,
        };
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&metrics);
        
        Ok(QualityScore {
            overall_score: final_score,
            metrics,
            level,
            timestamp: Utc::now(),
            recommendations,
        })
    }

    fn generate_recommendations(&self, metrics: &QualityMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if metrics.completeness < 0.9 {
            recommendations.push("Increase data collection frequency or add more sources".to_string());
        }
        if metrics.accuracy < 0.95 {
            recommendations.push("Implement additional validation rules".to_string());
        }
        if metrics.consistency < 0.9 {
            recommendations.push("Review cross-source reconciliation thresholds".to_string());
        }
        if metrics.timeliness < 0.95 {
            recommendations.push("Optimize data pipeline for lower latency".to_string());
        }
        if metrics.validity < 0.8 {
            recommendations.push("Update validation schemas and business rules".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub completeness: f64,  // Percentage of expected data present
    pub accuracy: f64,       // Correctness of values
    pub consistency: f64,    // Agreement across sources
    pub timeliness: f64,     // Freshness of data
    pub validity: f64,       // Conformance to rules
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    pub overall_score: f64,
    pub metrics: QualityMetrics,
    pub level: QualityLevel,
    pub timestamp: DateTime<Utc>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityLevel {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Unacceptable,
}