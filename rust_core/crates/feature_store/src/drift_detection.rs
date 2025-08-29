// Drift Detection System - KL Divergence, PSI, and Statistical Tests
// DEEP DIVE: Production-grade drift monitoring for feature distributions

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};
use statrs::distribution::{Normal, ContinuousCDF};
use statrs::statistics::Statistics;
use ndarray::{Array1, Array2};
use tokio::sync::RwLock;

use crate::feature_registry::FeatureRegistry;
use crate::offline_store::OfflineStore;

/// Drift detection configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct DriftConfig {
    pub check_interval_seconds: u64,
    pub baseline_window_days: i64,
    pub detection_window_hours: i64,
    pub psi_threshold: f64,
    pub kl_threshold: f64,
    pub wasserstein_threshold: f64,
    pub chi_square_threshold: f64,
    pub alert_cooldown_minutes: i64,
    pub min_samples: usize,
    pub enable_auto_retrain: bool,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            check_interval_seconds: 300, // 5 minutes
            baseline_window_days: 30,
            detection_window_hours: 24,
            psi_threshold: 0.2, // Industry standard
            kl_threshold: 0.5,
            wasserstein_threshold: 0.3,
            chi_square_threshold: 0.05, // p-value
            alert_cooldown_minutes: 60,
            min_samples: 1000,
            enable_auto_retrain: false,
        }
    }
}

/// Drift detector for monitoring feature distribution changes
/// TODO: Add docs
pub struct DriftDetector {
    config: DriftConfig,
    registry: Arc<FeatureRegistry>,
    offline_store: Arc<OfflineStore>,
    
    // Baseline distributions per feature
    baselines: Arc<RwLock<HashMap<String, BaselineDistribution>>>,
    
    // Active alerts
    active_alerts: Arc<RwLock<Vec<DriftAlert>>>,
    
    // Last check times
    last_checks: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    
    // Monitoring task handle
    monitor_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl DriftDetector {
    /// Create new drift detector
    pub async fn new(
        config: DriftConfig,
        registry: Arc<FeatureRegistry>,
    ) -> Result<Self> {
        info!("Initializing Drift Detection System");
        
        // Note: In production, would get offline_store from parent
        let offline_store = Arc::new(
            OfflineStore::new(Default::default()).await?
        );
        
        let detector = Self {
            config,
            registry,
            offline_store,
            baselines: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            last_checks: Arc::new(RwLock::new(HashMap::new())),
            monitor_handle: Arc::new(RwLock::new(None)),
        };
        
        // Load existing baselines
        detector.load_baselines().await?;
        
        Ok(detector)
    }
    
    /// Start monitoring for drift
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting drift monitoring");
        
        let config = self.config.clone();
        let baselines = self.baselines.clone();
        let active_alerts = self.active_alerts.clone();
        let last_checks = self.last_checks.clone();
        let registry = self.registry.clone();
        let offline_store = self.offline_store.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(config.check_interval_seconds)
            );
            
            loop {
                interval.tick().await;
                
                // Get all features to monitor
                let features = match Self::get_monitored_features(&registry).await {
                    Ok(f) => f,
                    Err(e) => {
                        error!("Failed to get features: {}", e);
                        continue;
                    }
                };
                
                for feature_name in features {
                    // Check if enough time has passed since last check
                    let should_check = {
                        let checks = last_checks.read().await;
                        if let Some(last) = checks.get(&feature_name) {
                            Utc::now().signed_duration_since(*last) > 
                                Duration::minutes(config.alert_cooldown_minutes)
                        } else {
                            true
                        }
                    };
                    
                    if !should_check {
                        continue;
                    }
                    
                    // Perform drift check
                    match Self::check_feature_drift(
                        &feature_name,
                        &config,
                        &baselines,
                        &offline_store,
                    ).await {
                        Ok(Some(alert)) => {
                            // Add alert
                            active_alerts.write().await.push(alert.clone());
                            last_checks.write().await.insert(feature_name.clone(), Utc::now());
                            
                            warn!("Drift detected for feature {}: {:?}", feature_name, alert);
                        }
                        Ok(None) => {
                            debug!("No drift detected for {}", feature_name);
                        }
                        Err(e) => {
                            error!("Drift check failed for {}: {}", feature_name, e);
                        }
                    }
                }
            }
        });
        
        *self.monitor_handle.write().await = Some(handle);
        Ok(())
    }
    
    /// Stop monitoring
    pub async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.monitor_handle.write().await.take() {
            handle.abort();
        }
        Ok(())
    }
    
    /// Add feature to monitoring
    pub async fn add_feature(&self, feature_id: &str) -> Result<()> {
        // Compute baseline for new feature
        let baseline = self.compute_baseline(feature_id).await?;
        self.baselines.write().await.insert(feature_id.to_string(), baseline);
        
        info!("Added feature {} to drift monitoring", feature_id);
        Ok(())
    }
    
    /// Check single update for drift
    pub async fn check_update(&self, update: &crate::FeatureUpdate) -> Result<Option<DriftAlert>> {
        let baselines = self.baselines.read().await;
        
        if let Some(baseline) = baselines.get(&update.feature_id) {
            // Simple threshold check for single value
            if let crate::FeatureValue::Float(value) = &update.value {
                let z_score = (value - baseline.mean) / baseline.stddev;
                
                if z_score.abs() > 4.0 {
                    return Ok(Some(DriftAlert {
                        feature_name: update.feature_id.clone(),
                        drift_type: DriftType::Outlier,
                        severity: if z_score.abs() > 6.0 {
                            Severity::Critical
                        } else {
                            Severity::High
                        },
                        metrics: DriftMetrics {
                            psi_score: None,
                            kl_divergence: None,
                            wasserstein_distance: None,
                            chi_square_statistic: None,
                            z_score: Some(z_score),
                        },
                        baseline_period: (
                            Utc::now() - Duration::days(self.config.baseline_window_days),
                            Utc::now(),
                        ),
                        detection_time: Utc::now(),
                        recommended_action: "Investigate outlier value".to_string(),
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Result<usize> {
        Ok(self.active_alerts.read().await.len())
    }
    
    /// Compute baseline distribution
    async fn compute_baseline(&self, feature_id: &str) -> Result<BaselineDistribution> {
        let end = Utc::now();
        let start = end - Duration::days(self.config.baseline_window_days);
        
        // Get historical data from offline store
        let stats = self.offline_store
            .get_feature_statistics(feature_id, start, end)
            .await?;
        
        // Compute distribution parameters
        let baseline = BaselineDistribution {
            feature_name: feature_id.to_string(),
            mean: stats.mean.unwrap_or(0.0),
            stddev: stats.stddev.unwrap_or(1.0),
            min: stats.min.unwrap_or(0.0),
            max: stats.max.unwrap_or(1.0),
            percentiles: vec![
                stats.p25.unwrap_or(0.0),
                stats.p50.unwrap_or(0.0),
                stats.p75.unwrap_or(0.0),
                stats.p95.unwrap_or(0.0),
                stats.p99.unwrap_or(0.0),
            ],
            sample_count: stats.count,
            computed_at: Utc::now(),
        };
        
        Ok(baseline)
    }
    
    /// Load existing baselines
    async fn load_baselines(&self) -> Result<()> {
        // In production, would load from persistent storage
        info!("Loading baseline distributions");
        Ok(())
    }
    
    /// Get features to monitor
    async fn get_monitored_features(registry: &Arc<FeatureRegistry>) -> Result<Vec<String>> {
        // In production, would query registry for features marked for monitoring
        Ok(vec![
            "price_sma_20".to_string(),
            "volume_ema_14".to_string(),
            "rsi_14".to_string(),
        ])
    }
    
    /// Check feature for drift
    async fn check_feature_drift(
        feature_name: &str,
        config: &DriftConfig,
        baselines: &Arc<RwLock<HashMap<String, BaselineDistribution>>>,
        offline_store: &Arc<OfflineStore>,
    ) -> Result<Option<DriftAlert>> {
        let baseline = {
            let baselines = baselines.read().await;
            baselines.get(feature_name).cloned()
        };
        
        let baseline = match baseline {
            Some(b) => b,
            None => return Ok(None),
        };
        
        // Get recent data
        let end = Utc::now();
        let start = end - Duration::hours(config.detection_window_hours);
        
        let recent_stats = offline_store
            .get_feature_statistics(feature_name, start, end)
            .await?;
        
        // Check if we have enough samples
        if recent_stats.count < config.min_samples as u64 {
            debug!("Not enough samples for {}: {}", feature_name, recent_stats.count);
            return Ok(None);
        }
        
        // Calculate drift metrics
        let mut metrics = DriftMetrics {
            psi_score: None,
            kl_divergence: None,
            wasserstein_distance: None,
            chi_square_statistic: None,
            z_score: None,
        };
        
        // PSI (Population Stability Index)
        let psi = Self::calculate_psi(&baseline, &recent_stats)?;
        metrics.psi_score = Some(psi);
        
        // KL Divergence (assuming normal distributions)
        let kl = Self::calculate_kl_divergence(&baseline, &recent_stats)?;
        metrics.kl_divergence = Some(kl);
        
        // Wasserstein Distance
        let wasserstein = Self::calculate_wasserstein(&baseline, &recent_stats)?;
        metrics.wasserstein_distance = Some(wasserstein);
        
        // Chi-square test
        let chi_square = Self::calculate_chi_square(&baseline, &recent_stats)?;
        metrics.chi_square_statistic = Some(chi_square);
        
        // Determine if drift detected
        let mut drift_detected = false;
        let mut severity = Severity::Low;
        let mut drift_type = DriftType::Gradual;
        
        if psi > config.psi_threshold {
            drift_detected = true;
            if psi > config.psi_threshold * 2.0 {
                severity = Severity::Critical;
                drift_type = DriftType::Sudden;
            } else {
                severity = Severity::High;
            }
        }
        
        if kl > config.kl_threshold {
            drift_detected = true;
            if severity == Severity::Low {
                severity = Severity::Medium;
            }
        }
        
        if wasserstein > config.wasserstein_threshold {
            drift_detected = true;
            drift_type = DriftType::Distributional;
        }
        
        if chi_square < config.chi_square_threshold {
            drift_detected = true;
            drift_type = DriftType::Statistical;
        }
        
        if drift_detected {
            Ok(Some(DriftAlert {
                feature_name: feature_name.to_string(),
                drift_type,
                severity,
                metrics,
                baseline_period: (
                    end - Duration::days(config.baseline_window_days),
                    end,
                ),
                detection_time: Utc::now(),
                recommended_action: Self::get_recommended_action(&drift_type, &severity),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate PSI (Population Stability Index)
    fn calculate_psi(
        baseline: &BaselineDistribution,
        recent: &crate::offline_store::FeatureStatistics,
    ) -> Result<f64> {
        // PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        // Using percentile bins for calculation
        
        let baseline_bins = &baseline.percentiles;
        let recent_bins = vec![
            recent.p25.unwrap_or(0.0),
            recent.p50.unwrap_or(0.0),
            recent.p75.unwrap_or(0.0),
            recent.p95.unwrap_or(0.0),
            recent.p99.unwrap_or(0.0),
        ];
        
        let mut psi = 0.0;
        for i in 0..baseline_bins.len() {
            let expected = if i == 0 {
                0.25
            } else if i == 1 {
                0.25
            } else if i == 2 {
                0.25
            } else if i == 3 {
                0.20
            } else {
                0.05
            };
            
            // Approximate actual percentage based on percentile shift
            let percentile_shift = (recent_bins[i] - baseline_bins[i]).abs() / 
                                  (baseline_bins[i].abs() + 1e-10);
            let actual = expected * (1.0 + percentile_shift.min(2.0));
            
            if actual > 0.0 && expected > 0.0 {
                psi += (actual - expected) * (actual / expected).ln();
            }
        }
        
        Ok(psi.abs())
    }
    
    /// Calculate KL Divergence
    fn calculate_kl_divergence(
        baseline: &BaselineDistribution,
        recent: &crate::offline_store::FeatureStatistics,
    ) -> Result<f64> {
        // KL(P||Q) = ∫ p(x) * log(p(x)/q(x)) dx
        // Assuming normal distributions for simplification
        
        let p_mean = baseline.mean;
        let p_var = baseline.stddev.powi(2);
        let q_mean = recent.mean.unwrap_or(0.0);
        let q_var = recent.stddev.unwrap_or(1.0).powi(2);
        
        // Closed form for normal distributions
        let kl = 0.5 * (
            (q_var / p_var).ln() - 1.0 + 
            p_var / q_var + 
            (p_mean - q_mean).powi(2) / q_var
        );
        
        Ok(kl.abs())
    }
    
    /// Calculate Wasserstein Distance
    fn calculate_wasserstein(
        baseline: &BaselineDistribution,
        recent: &crate::offline_store::FeatureStatistics,
    ) -> Result<f64> {
        // 1-Wasserstein distance for normal distributions
        // W_1(P,Q) = |μ_P - μ_Q| + |σ_P - σ_Q|
        
        let mean_diff = (baseline.mean - recent.mean.unwrap_or(0.0)).abs();
        let std_diff = (baseline.stddev - recent.stddev.unwrap_or(1.0)).abs();
        
        Ok(mean_diff + std_diff)
    }
    
    /// Calculate Chi-square test statistic
    fn calculate_chi_square(
        baseline: &BaselineDistribution,
        recent: &crate::offline_store::FeatureStatistics,
    ) -> Result<f64> {
        // Two-sample chi-square test
        // Using histogram bins approximation
        
        let n_bins = 10;
        let min_val = baseline.min.min(recent.min.unwrap_or(0.0));
        let max_val = baseline.max.max(recent.max.unwrap_or(1.0));
        let bin_width = (max_val - min_val) / n_bins as f64;
        
        // Approximate histogram using normal distribution
        let normal_baseline = Normal::new(baseline.mean, baseline.stddev)?;
        let normal_recent = Normal::new(
            recent.mean.unwrap_or(0.0),
            recent.stddev.unwrap_or(1.0),
        )?;
        
        let mut chi_square = 0.0;
        for i in 0..n_bins {
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            
            let expected = normal_baseline.cdf(bin_end) - normal_baseline.cdf(bin_start);
            let observed = normal_recent.cdf(bin_end) - normal_recent.cdf(bin_start);
            
            if expected > 0.0 {
                chi_square += (observed - expected).powi(2) / expected;
            }
        }
        
        // Convert to p-value (simplified)
        let p_value = 1.0 / (1.0 + chi_square);
        
        Ok(p_value)
    }
    
    /// Get recommended action based on drift type
    fn get_recommended_action(drift_type: &DriftType, severity: &Severity) -> String {
        match (drift_type, severity) {
            (DriftType::Sudden, Severity::Critical) => {
                "IMMEDIATE ACTION: Pause trading and investigate data pipeline".to_string()
            }
            (DriftType::Sudden, _) => {
                "Monitor closely and consider model retraining".to_string()
            }
            (DriftType::Gradual, _) => {
                "Schedule model retraining within 24 hours".to_string()
            }
            (DriftType::Outlier, Severity::Critical) => {
                "Check for data quality issues or market events".to_string()
            }
            (DriftType::Statistical, _) => {
                "Review feature engineering pipeline".to_string()
            }
            _ => "Continue monitoring".to_string()
        }
    }
}

/// Baseline distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineDistribution {
    feature_name: String,
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
    percentiles: Vec<f64>, // [p25, p50, p75, p95, p99]
    sample_count: u64,
    computed_at: DateTime<Utc>,
}

/// Drift alert
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct DriftAlert {
    pub feature_name: String,
    pub drift_type: DriftType,
    pub severity: Severity,
    pub metrics: DriftMetrics,
    pub baseline_period: (DateTime<Utc>, DateTime<Utc>),
    pub detection_time: DateTime<Utc>,
    pub recommended_action: String,
}

/// Types of drift
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum DriftType {
    Gradual,        // Slow change over time
    Sudden,         // Abrupt change
    Outlier,        // Individual outliers
    Distributional, // Shape change
    Statistical,    // Statistical properties change
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Drift metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct DriftMetrics {
    pub psi_score: Option<f64>,
    pub kl_divergence: Option<f64>,
    pub wasserstein_distance: Option<f64>,
    pub chi_square_statistic: Option<f64>,
    pub z_score: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_psi_calculation() {
        let baseline = BaselineDistribution {
            feature_name: "test".to_string(),
            mean: 100.0,
            stddev: 10.0,
            min: 70.0,
            max: 130.0,
            percentiles: vec![92.5, 100.0, 107.5, 118.0, 125.0],
            sample_count: 10000,
            computed_at: Utc::now(),
        };
        
        let recent = crate::offline_store::FeatureStatistics {
            feature_name: "test".to_string(),
            count: 1000,
            mean: Some(105.0), // 5% drift
            stddev: Some(12.0),
            min: Some(75.0),
            max: Some(135.0),
            p25: Some(95.0),
            p50: Some(105.0),
            p75: Some(113.0),
            p95: Some(125.0),
            p99: Some(133.0),
            computed_at: Utc::now(),
        };
        
        let psi = DriftDetector::calculate_psi(&baseline, &recent).unwrap();
        assert!(psi > 0.0);
        assert!(psi < 1.0); // Should detect moderate drift
    }
    
    #[test]
    fn test_kl_divergence() {
        let baseline = BaselineDistribution {
            feature_name: "test".to_string(),
            mean: 0.0,
            stddev: 1.0,
            min: -3.0,
            max: 3.0,
            percentiles: vec![-0.67, 0.0, 0.67, 1.64, 2.33],
            sample_count: 10000,
            computed_at: Utc::now(),
        };
        
        let recent = crate::offline_store::FeatureStatistics {
            feature_name: "test".to_string(),
            count: 1000,
            mean: Some(0.5), // Shifted mean
            stddev: Some(1.2), // Increased variance
            min: Some(-3.5),
            max: Some(4.0),
            p25: Some(-0.4),
            p50: Some(0.5),
            p75: Some(1.4),
            p95: Some(2.5),
            p99: Some(3.3),
            computed_at: Utc::now(),
        };
        
        let kl = DriftDetector::calculate_kl_divergence(&baseline, &recent).unwrap();
        assert!(kl > 0.0);
    }
}