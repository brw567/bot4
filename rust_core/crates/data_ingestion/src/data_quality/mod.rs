// Layer 1.6: Data Quality & Validation Implementation
// DEEP DIVE - Production-ready data integrity assurance
//
// Architecture:
// - Benford's Law for digit distribution anomaly detection
// - Kalman filtering for gap detection and prediction
// - Priority-based automatic backfill system
// - Cross-source reconciliation with consensus
// - Change point detection (CUSUM, PELT algorithms)
// - Multi-dimensional quality scoring
// - Real-time alerting with severity levels
//
// External Research Applied:
// - Amiram et al. (2015) - Benford's Law in financial fraud detection
// - Harvey (1989) - Kalman filter applications in finance
// - Page (1954) - CUSUM change detection
// - Killick et al. (2012) - PELT algorithm
// - NYSE TAQ data quality standards
// - CME Group data integrity protocols
// - Two Sigma's data validation framework

pub mod benford;
pub mod kalman_filter;
pub mod backfill;
pub mod reconciliation;
pub mod change_detection;
pub mod quality_scorer;
pub mod monitoring;

use std::sync::Arc;
use std::collections::{HashMap, VecDeque, BTreeMap};
use async_trait::async_trait;
use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, error, debug};

// Re-exports
pub use benford::{BenfordValidator, BenfordAnomaly};
pub use kalman_filter::{KalmanGapDetector, GapEvent};
pub use backfill::{BackfillSystem, BackfillPriority, BackfillRequest};
pub use reconciliation::{CrossSourceReconciler, ReconciliationResult};
pub use change_detection::{ChangeDetector, ChangePoint, DetectionAlgorithm};
pub use quality_scorer::{QualityScorer, QualityMetrics, QualityScore};
pub use monitoring::{QualityMonitor, AlertSeverity, QualityAlert};

/// Configuration for data quality system
#[derive(Debug, Clone, Deserialize)]
pub struct DataQualityConfig {
    pub benford_config: benford::BenfordConfig,
    pub kalman_config: kalman_filter::KalmanConfig,
    pub backfill_config: backfill::BackfillConfig,
    pub reconciliation_config: reconciliation::ReconciliationConfig,
    pub change_detection_config: change_detection::ChangeDetectionConfig,
    pub scoring_config: quality_scorer::ScoringConfig,
    pub monitoring_config: monitoring::MonitoringConfig,
    
    // Global settings
    pub enable_auto_correction: bool,
    pub max_correction_attempts: usize,
    pub quality_threshold: f64,  // Minimum acceptable quality score
    pub alert_cooldown_ms: u64,
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            benford_config: Default::default(),
            kalman_config: Default::default(),
            backfill_config: Default::default(),
            reconciliation_config: Default::default(),
            change_detection_config: Default::default(),
            scoring_config: Default::default(),
            monitoring_config: Default::default(),
            enable_auto_correction: true,
            max_correction_attempts: 3,
            quality_threshold: 0.95,  // 95% quality required
            alert_cooldown_ms: 5000,
        }
    }
}

/// Main Data Quality Manager - orchestrates all validation
pub struct DataQualityManager {
    config: DataQualityConfig,
    
    // Validation components
    benford_validator: Arc<BenfordValidator>,
    gap_detector: Arc<KalmanGapDetector>,
    backfill_system: Arc<BackfillSystem>,
    reconciler: Arc<CrossSourceReconciler>,
    change_detector: Arc<ChangeDetector>,
    quality_scorer: Arc<QualityScorer>,
    monitor: Arc<QualityMonitor>,
    
    // State tracking
    validation_history: Arc<RwLock<VecDeque<ValidationResult>>>,
    quality_scores: Arc<RwLock<HashMap<String, QualityScore>>>,
    active_alerts: Arc<RwLock<Vec<QualityAlert>>>,
    
    // Metrics
    total_validations: Arc<RwLock<u64>>,
    failed_validations: Arc<RwLock<u64>>,
    auto_corrections: Arc<RwLock<u64>>,
}

impl DataQualityManager {
    /// Create new Data Quality Manager
    pub async fn new(config: DataQualityConfig) -> Result<Self> {
        info!("Initializing Data Quality Manager with multi-layer validation");
        
        // Initialize components
        let benford_validator = Arc::new(
            BenfordValidator::new(config.benford_config.clone())
        );
        
        let gap_detector = Arc::new(
            KalmanGapDetector::new(config.kalman_config.clone()).await?
        );
        
        let backfill_system = Arc::new(
            BackfillSystem::new(config.backfill_config.clone()).await?
        );
        
        let reconciler = Arc::new(
            CrossSourceReconciler::new(config.reconciliation_config.clone()).await?
        );
        
        let change_detector = Arc::new(
            ChangeDetector::new(config.change_detection_config.clone())
        );
        
        let quality_scorer = Arc::new(
            QualityScorer::new(config.scoring_config.clone())
        );
        
        let monitor = Arc::new(
            QualityMonitor::new(config.monitoring_config.clone()).await?
        );
        
        // Start background monitoring
        monitor.start_monitoring().await?;
        
        info!("Data Quality Manager initialized with 7 validation layers");
        
        Ok(Self {
            config,
            benford_validator,
            gap_detector,
            backfill_system,
            reconciler,
            change_detector,
            quality_scorer,
            monitor,
            validation_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            quality_scores: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            total_validations: Arc::new(RwLock::new(0)),
            failed_validations: Arc::new(RwLock::new(0)),
            auto_corrections: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Validate incoming data with full pipeline
    pub async fn validate_data(
        &self,
        data: DataBatch,
    ) -> Result<ValidationResult> {
        let start = std::time::Instant::now();
        
        // Increment validation counter
        *self.total_validations.write().await += 1;
        
        let mut validation_result = ValidationResult {
            timestamp: Utc::now(),
            symbol: data.symbol.clone(),
            data_type: data.data_type.clone(),
            is_valid: true,
            quality_score: 1.0,
            issues: Vec::new(),
            corrections_applied: Vec::new(),
        };
        
        // Layer 1: Benford's Law validation
        if let Some(anomaly) = self.benford_validator.validate(&data).await? {
            validation_result.issues.push(ValidationIssue {
                severity: IssueSeverity::from_benford(&anomaly),
                category: IssueCategory::StatisticalAnomaly,
                description: format!("Benford's Law violation: {:?}", anomaly),
                timestamp: Utc::now(),
            });
        }
        
        // Layer 2: Gap detection with Kalman filter
        if let Some(gap) = self.gap_detector.detect_gaps(&data).await? {
            validation_result.issues.push(ValidationIssue {
                severity: IssueSeverity::from_gap(&gap),
                category: IssueCategory::DataGap,
                description: format!("Gap detected: {:?}", gap),
                timestamp: Utc::now(),
            });
            
            // Trigger automatic backfill if enabled
            if self.config.enable_auto_correction {
                self.trigger_backfill(gap).await?;
                *self.auto_corrections.write().await += 1;
            }
        }
        
        // Layer 3: Cross-source reconciliation
        let reconciliation = self.reconciler.reconcile(&data).await?;
        if !reconciliation.is_consistent {
            validation_result.issues.push(ValidationIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::InconsistentData,
                description: format!("Cross-source mismatch: {:?}", reconciliation),
                timestamp: Utc::now(),
            });
        }
        
        // Layer 4: Change point detection
        if let Some(change_point) = self.change_detector.detect(&data).await? {
            validation_result.issues.push(ValidationIssue {
                severity: IssueSeverity::from_change_point(&change_point),
                category: IssueCategory::RegimeChange,
                description: format!("Change point detected: {:?}", change_point),
                timestamp: Utc::now(),
            });
        }
        
        // Layer 5: Calculate quality score
        let quality_metrics = QualityMetrics {
            completeness: reconciliation.completeness_score,
            accuracy: 1.0 - (validation_result.issues.len() as f64 / 10.0).min(1.0),
            consistency: reconciliation.consistency_score,
            timeliness: self.calculate_timeliness(&data).await?,
            validity: if validation_result.issues.is_empty() { 1.0 } else { 0.8 },
        };
        
        let quality_score = self.quality_scorer.calculate_score(quality_metrics).await?;
        validation_result.quality_score = quality_score.overall_score;
        
        // Update quality score history
        self.quality_scores.write().await.insert(
            data.symbol.clone(),
            quality_score.clone(),
        );
        
        // Determine if data is valid based on quality threshold
        validation_result.is_valid = quality_score.overall_score >= self.config.quality_threshold;
        
        if !validation_result.is_valid {
            *self.failed_validations.write().await += 1;
        }
        
        // Layer 6: Generate alerts if needed
        if !validation_result.issues.is_empty() {
            self.generate_alerts(&validation_result).await?;
        }
        
        // Layer 7: Monitor and record
        self.monitor.record_validation(&validation_result).await?;
        
        // Store validation history
        let mut history = self.validation_history.write().await;
        if history.len() >= 10000 {
            history.pop_front();
        }
        history.push_back(validation_result.clone());
        
        // Log performance
        let latency = start.elapsed();
        if latency.as_millis() > 10 {
            warn!("Data validation exceeded 10ms target: {:?}", latency);
        } else {
            debug!("Data validation completed in {:?}", latency);
        }
        
        Ok(validation_result)
    }
    
    /// Validate batch of historical data
    pub async fn validate_historical(
        &self,
        data: Vec<DataBatch>,
        parallel: bool,
    ) -> Result<BatchValidationResult> {
        info!("Validating historical batch of {} items", data.len());
        
        let mut results = Vec::new();
        
        if parallel {
            // Parallel validation for large batches
            use futures::future::join_all;
            let futures = data.into_iter()
                .map(|batch| self.validate_data(batch))
                .collect::<Vec<_>>();
            
            results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
        } else {
            // Sequential validation for ordered data
            for batch in data {
                results.push(self.validate_data(batch).await?);
            }
        }
        
        // Calculate aggregate metrics
        let total = results.len();
        let valid = results.iter().filter(|r| r.is_valid).count();
        let avg_quality = results.iter()
            .map(|r| r.quality_score)
            .sum::<f64>() / total as f64;
        
        Ok(BatchValidationResult {
            total_items: total,
            valid_items: valid,
            invalid_items: total - valid,
            average_quality: avg_quality,
            individual_results: results,
        })
    }
    
    /// Trigger automatic backfill for detected gaps
    async fn trigger_backfill(&self, gap: GapEvent) -> Result<()> {
        let request = BackfillRequest {
            symbol: gap.symbol,
            start_time: gap.gap_start,
            end_time: gap.gap_end,
            priority: BackfillPriority::High,
            source: "auto_detection".to_string(),
            max_retries: 3,
        };
        
        self.backfill_system.request_backfill(request).await?;
        info!("Automatic backfill triggered for gap: {:?}", gap);
        
        Ok(())
    }
    
    /// Calculate timeliness score based on data freshness
    async fn calculate_timeliness(&self, data: &DataBatch) -> Result<f64> {
        let now = Utc::now();
        let age = now - data.timestamp;
        
        // Score based on age (exponential decay)
        // Perfect score for <1s old, 0.5 at 30s, near 0 at 5min
        let age_seconds = age.num_seconds() as f64;
        let score = (-age_seconds / 30.0).exp();
        
        Ok(score)
    }
    
    /// Generate alerts based on validation issues
    async fn generate_alerts(&self, result: &ValidationResult) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;
        
        for issue in &result.issues {
            // Check cooldown to avoid alert spam
            let should_alert = !alerts.iter().any(|a| {
                a.symbol == result.symbol && 
                a.category == issue.category &&
                (Utc::now() - a.timestamp).num_milliseconds() < self.config.alert_cooldown_ms as i64
            });
            
            if should_alert {
                let alert = QualityAlert {
                    timestamp: Utc::now(),
                    symbol: result.symbol.clone(),
                    severity: AlertSeverity::from_issue(&issue.severity),
                    category: issue.category.clone(),
                    message: issue.description.clone(),
                    quality_score: result.quality_score,
                };
                
                // Send to monitoring system
                self.monitor.send_alert(alert.clone()).await?;
                
                // Store active alert
                alerts.push(alert);
            }
        }
        
        // Clean up old alerts
        let cutoff = Utc::now() - Duration::minutes(5);
        alerts.retain(|a| a.timestamp > cutoff);
        
        Ok(())
    }
    
    /// Get current quality status for a symbol
    pub async fn get_quality_status(&self, symbol: &str) -> Result<QualityStatus> {
        let scores = self.quality_scores.read().await;
        let score = scores.get(symbol).cloned();
        
        let history = self.validation_history.read().await;
        let recent_validations: Vec<_> = history.iter()
            .filter(|v| v.symbol == symbol)
            .take(100)
            .cloned()
            .collect();
        
        let alerts = self.active_alerts.read().await;
        let active_alerts: Vec<_> = alerts.iter()
            .filter(|a| a.symbol == symbol)
            .cloned()
            .collect();
        
        Ok(QualityStatus {
            symbol: symbol.to_string(),
            current_score: score,
            recent_validations,
            active_alerts,
            last_updated: Utc::now(),
        })
    }
    
    /// Force reconciliation across all sources
    pub async fn force_reconciliation(&self, symbol: &str) -> Result<ReconciliationResult> {
        info!("Forcing reconciliation for symbol: {}", symbol);
        self.reconciler.force_reconcile(symbol).await
    }
    
    /// Get validation statistics
    pub async fn get_statistics(&self) -> Result<ValidationStatistics> {
        let total = *self.total_validations.read().await;
        let failed = *self.failed_validations.read().await;
        let corrections = *self.auto_corrections.read().await;
        
        let success_rate = if total > 0 {
            (total - failed) as f64 / total as f64
        } else {
            1.0
        };
        
        Ok(ValidationStatistics {
            total_validations: total,
            failed_validations: failed,
            success_rate,
            auto_corrections: corrections,
            active_alerts: self.active_alerts.read().await.len(),
        })
    }
    
    /// Shutdown the quality manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Data Quality Manager");
        
        self.monitor.stop().await?;
        self.backfill_system.shutdown().await?;
        
        Ok(())
    }
}

/// Data batch for validation
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub symbol: String,
    pub data_type: DataType,
    pub timestamp: DateTime<Utc>,
    pub values: Vec<f64>,
    pub source: String,
    pub metadata: Option<serde_json::Value>,
}

/// Types of data being validated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Price,
    Volume,
    OrderBook,
    Trade,
    Quote,
    AggregatedBar,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub data_type: DataType,
    pub is_valid: bool,
    pub quality_score: f64,
    pub issues: Vec<ValidationIssue>,
    pub corrections_applied: Vec<Correction>,
}

/// Validation issue details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    pub timestamp: DateTime<Utc>,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl IssueSeverity {
    fn from_benford(anomaly: &BenfordAnomaly) -> Self {
        match anomaly.deviation {
            d if d < 0.05 => Self::Low,
            d if d < 0.10 => Self::Medium,
            d if d < 0.20 => Self::High,
            _ => Self::Critical,
        }
    }
    
    fn from_gap(gap: &GapEvent) -> Self {
        let gap_duration = (gap.gap_end - gap.gap_start).num_seconds();
        match gap_duration {
            d if d < 60 => Self::Low,
            d if d < 300 => Self::Medium,
            d if d < 3600 => Self::High,
            _ => Self::Critical,
        }
    }
    
    fn from_change_point(cp: &ChangePoint) -> Self {
        match cp.confidence {
            c if c < 0.8 => Self::Low,
            c if c < 0.9 => Self::Medium,
            c if c < 0.95 => Self::High,
            _ => Self::Critical,
        }
    }
}

/// Issue categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueCategory {
    StatisticalAnomaly,
    DataGap,
    InconsistentData,
    RegimeChange,
    StaleData,
    InvalidValue,
}

/// Applied corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    pub correction_type: CorrectionType,
    pub description: String,
    pub timestamp: DateTime<Utc>,
}

/// Types of corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionType {
    Backfill,
    Interpolation,
    SourceSwitch,
    ValueAdjustment,
}

/// Batch validation result
#[derive(Debug, Clone)]
pub struct BatchValidationResult {
    pub total_items: usize,
    pub valid_items: usize,
    pub invalid_items: usize,
    pub average_quality: f64,
    pub individual_results: Vec<ValidationResult>,
}

/// Current quality status for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStatus {
    pub symbol: String,
    pub current_score: Option<QualityScore>,
    pub recent_validations: Vec<ValidationResult>,
    pub active_alerts: Vec<QualityAlert>,
    pub last_updated: DateTime<Utc>,
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub total_validations: u64,
    pub failed_validations: u64,
    pub success_rate: f64,
    pub auto_corrections: u64,
    pub active_alerts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_quality_manager() {
        let config = DataQualityConfig::default();
        let manager = DataQualityManager::new(config).await.unwrap();
        
        // Test data validation
        let data = DataBatch {
            symbol: "BTC-USDT".to_string(),
            data_type: DataType::Price,
            timestamp: Utc::now(),
            values: vec![50000.0, 50100.0, 50050.0],
            source: "binance".to_string(),
            metadata: None,
        };
        
        let result = manager.validate_data(data).await.unwrap();
        assert!(result.quality_score > 0.0);
    }
    
    #[tokio::test]
    async fn test_batch_validation() {
        let config = DataQualityConfig::default();
        let manager = DataQualityManager::new(config).await.unwrap();
        
        // Create batch of test data
        let mut batch = Vec::new();
        for i in 0..10 {
            batch.push(DataBatch {
                symbol: "ETH-USDT".to_string(),
                data_type: DataType::Trade,
                timestamp: Utc::now() - Duration::seconds(i),
                values: vec![3000.0 + i as f64],
                source: "kraken".to_string(),
                metadata: None,
            });
        }
        
        let result = manager.validate_historical(batch, true).await.unwrap();
        assert_eq!(result.total_items, 10);
        assert!(result.average_quality > 0.0);
    }
}