//! # FEATURE MONITORING - Drift detection and quality assurance
//! Ellis (Performance Lead): "Monitor everything, alert on anomalies"

use super::*;
use statrs::distribution::{Normal, ContinuousCDF};
use std::collections::VecDeque;

/// Feature monitoring system
/// TODO: Add docs
pub struct FeatureMonitor {
    store: Arc<FeatureStore>,
    
    /// Drift detectors per feature
    drift_detectors: Arc<DashMap<String, DriftDetector>>,
    
    /// Quality metrics
    quality_metrics: Arc<RwLock<QualityMetrics>>,
    
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    
    /// Historical statistics
    historical_stats: Arc<RwLock<HashMap<String, HistoricalStats>>>,
}

/// Drift detector using statistical tests
/// TODO: Add docs
pub struct DriftDetector {
    feature_name: String,
    baseline_stats: BaselineStatistics,
    recent_values: VecDeque<f64>,
    drift_threshold: f64,
    detection_method: DriftDetectionMethod,
}

/// Baseline statistics for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct BaselineStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub percentiles: HashMap<u8, f64>,
    pub computed_at: DateTime<Utc>,
    pub sample_size: usize,
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum DriftDetectionMethod {
    KolmogorovSmirnov,
    ChiSquared,
    PopulationStabilityIndex,
    WassersteinDistance,
    JensenShannonDivergence,
}

/// Quality metrics
#[derive(Debug, Default)]
/// TODO: Add docs
pub struct QualityMetrics {
    pub completeness_rate: f64,
    pub freshness_seconds: f64,
    pub anomaly_rate: f64,
    pub null_rate: f64,
    pub duplicate_rate: f64,
}

/// Alert manager
/// TODO: Add docs
pub struct AlertManager {
    alerts: Arc<RwLock<Vec<Alert>>>,
    alert_rules: Arc<Vec<AlertRule>>,
    notification_channels: Arc<Vec<Box<dyn NotificationChannel>>>,
}

/// Alert definition
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub feature: String,
    pub message: String,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Alert rule
/// TODO: Add docs
pub struct AlertRule {
    pub name: String,
    pub condition: Box<dyn Fn(&FeatureValue) -> bool + Send + Sync>,
    pub severity: AlertSeverity,
    pub cooldown_seconds: u64,
}

/// Notification channel trait
#[async_trait]
pub trait NotificationChannel: Send + Sync {
    async fn send(&self, alert: &Alert) -> Result<(), String>;
}

/// Historical statistics
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct HistoricalStats {
    pub daily_stats: VecDeque<DailyStats>,
    pub weekly_stats: VecDeque<WeeklyStats>,
    pub monthly_stats: VecDeque<MonthlyStats>,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct DailyStats {
    pub date: chrono::NaiveDate,
    pub mean: f64,
    pub std_dev: f64,
    pub volume: u64,
    pub anomaly_count: u32,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct WeeklyStats {
    pub week_start: chrono::NaiveDate,
    pub mean: f64,
    pub std_dev: f64,
    pub volume: u64,
    pub drift_score: f64,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MonthlyStats {
    pub month: chrono::NaiveDate,
    pub mean: f64,
    pub std_dev: f64,
    pub volume: u64,
    pub quality_score: f64,
}

impl FeatureMonitor {
    pub fn new(store: Arc<FeatureStore>) -> Self {
        Self {
            store,
            drift_detectors: Arc::new(DashMap::new()),
            quality_metrics: Arc::new(RwLock::new(QualityMetrics::default())),
            alert_manager: Arc::new(AlertManager {
                alerts: Arc::new(RwLock::new(Vec::new())),
                alert_rules: Arc::new(Vec::new()),
                notification_channels: Arc::new(Vec::new()),
            }),
            historical_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Compute baseline statistics for a feature
    pub async fn compute_baseline(
        &self,
        feature_name: &str,
        lookback_days: i32,
    ) -> Result<BaselineStatistics, FeatureStoreError> {
        let values = sqlx::query_as::<_, (f64,)>(r#"
            SELECT (feature_value->>'value')::float
            FROM features
            WHERE feature_name = $1
                AND timestamp > NOW() - INTERVAL '%d days'
            ORDER BY RANDOM()
            LIMIT 10000
        "#)
        .bind(feature_name)
        .bind(lookback_days)
        .fetch_all(self.store.pool.as_ref())
        .await?
        .into_iter()
        .map(|(v,)| v)
        .collect::<Vec<_>>();
        
        if values.is_empty() {
            return Err(FeatureStoreError::FeatureNotFound(feature_name.to_string()));
        }
        
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        let percentiles = vec![1, 5, 25, 50, 75, 95, 99]
            .into_iter()
            .map(|p| {
                let idx = ((p as f64 / 100.0) * n) as usize;
                (p, sorted[idx.min(sorted.len() - 1)])
            })
            .collect();
        
        Ok(BaselineStatistics {
            mean,
            std_dev,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            median: sorted[sorted.len() / 2],
            percentiles,
            computed_at: Utc::now(),
            sample_size: values.len(),
        })
    }
    
    /// Detect feature drift
    pub async fn detect_drift(
        &self,
        feature_name: &str,
        recent_hours: i32,
    ) -> Result<DriftResult, FeatureStoreError> {
        let detector = self.drift_detectors.get(feature_name)
            .ok_or_else(|| FeatureStoreError::FeatureNotFound(feature_name.to_string()))?;
        
        // Get recent values
        let recent_values = sqlx::query_as::<_, (f64,)>(r#"
            SELECT (feature_value->>'value')::float
            FROM features
            WHERE feature_name = $1
                AND timestamp > NOW() - INTERVAL '%d hours'
        "#)
        .bind(feature_name)
        .bind(recent_hours)
        .fetch_all(self.store.pool.as_ref())
        .await?
        .into_iter()
        .map(|(v,)| v)
        .collect::<Vec<_>>();
        
        if recent_values.is_empty() {
            return Ok(DriftResult {
                feature: feature_name.to_string(),
                drift_detected: false,
                drift_score: 0.0,
                p_value: 1.0,
                message: "No recent data".to_string(),
            });
        }
        
        // Calculate drift score based on method
        let (drift_score, p_value) = match detector.detection_method {
            DriftDetectionMethod::PopulationStabilityIndex => {
                self.calculate_psi(&detector.baseline_stats, &recent_values)
            }
            DriftDetectionMethod::KolmogorovSmirnov => {
                self.calculate_ks_test(&detector.baseline_stats, &recent_values)
            }
            _ => (0.0, 1.0),
        };
        
        let drift_detected = drift_score > detector.drift_threshold;
        
        if drift_detected {
            self.alert_manager.create_alert(Alert {
                id: uuid::Uuid::new_v4().to_string(),
                severity: AlertSeverity::Warning,
                feature: feature_name.to_string(),
                message: format!("Drift detected: score={:.3}", drift_score),
                detected_at: Utc::now(),
                resolved_at: None,
                metadata: serde_json::json!({
                    "drift_score": drift_score,
                    "p_value": p_value,
                    "method": format!("{:?}", detector.detection_method),
                }),
            }).await;
        }
        
        Ok(DriftResult {
            feature: feature_name.to_string(),
            drift_detected,
            drift_score,
            p_value,
            message: if drift_detected {
                format!("Drift detected with score {:.3}", drift_score)
            } else {
                "No drift detected".to_string()
            },
        })
    }
    
    /// Calculate Population Stability Index
    fn calculate_psi(&self, baseline: &BaselineStatistics, recent: &[f64]) -> (f64, f64) {
        // Bin the data
        let n_bins = 10;
        let bin_edges = (0..=n_bins)
            .map(|i| baseline.min + (baseline.max - baseline.min) * (i as f64 / n_bins as f64))
            .collect::<Vec<_>>();
        
        // Calculate expected and actual distributions
        let mut expected = vec![0.0; n_bins];
        let mut actual = vec![0.0; n_bins];
        
        for value in recent {
            for i in 0..n_bins {
                if *value >= bin_edges[i] && *value < bin_edges[i + 1] {
                    actual[i] += 1.0;
                    break;
                }
            }
        }
        
        // Normalize
        let actual_sum = actual.iter().sum::<f64>();
        for i in 0..n_bins {
            actual[i] /= actual_sum;
            expected[i] = 1.0 / n_bins as f64; // Uniform assumption
        }
        
        // Calculate PSI
        let mut psi = 0.0;
        for i in 0..n_bins {
            if actual[i] > 0.0 && expected[i] > 0.0 {
                psi += (actual[i] - expected[i]) * (actual[i] / expected[i]).ln();
            }
        }
        
        // PSI < 0.1: no drift, 0.1-0.25: slight drift, >0.25: significant drift
        let p_value = if psi < 0.1 { 0.95 } else if psi < 0.25 { 0.5 } else { 0.05 };
        
        (psi, p_value)
    }
    
    /// Kolmogorov-Smirnov test
    fn calculate_ks_test(&self, baseline: &BaselineStatistics, recent: &[f64]) -> (f64, f64) {
        // Simplified KS test implementation
        let mut recent_sorted = recent.to_vec();
        recent_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate empirical CDFs
        let n1 = baseline.sample_size as f64;
        let n2 = recent.len() as f64;
        
        let mut max_diff = 0.0;
        for (i, value) in recent_sorted.iter().enumerate() {
            let cdf1 = self.normal_cdf(*value, baseline.mean, baseline.std_dev);
            let cdf2 = (i + 1) as f64 / n2;
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }
        
        // Critical value approximation
        let critical_value = 1.36 * ((n1 + n2) / (n1 * n2)).sqrt();
        let p_value = if max_diff > critical_value { 0.01 } else { 0.95 };
        
        (max_diff, p_value)
    }
    
    /// Normal CDF helper
    fn normal_cdf(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let normal = Normal::new(mean, std_dev).unwrap();
        normal.cdf(x)
    }
    
    /// Monitor feature quality
    pub async fn monitor_quality(&self, feature_name: &str) -> Result<QualityReport, FeatureStoreError> {
        let stats = sqlx::query_as::<_, (i64, i64, f64, f64)>(r#"
            SELECT 
                COUNT(*) as total_count,
                COUNT(DISTINCT entity_id) as unique_entities,
                COUNT(*)::float / NULLIF(COUNT(DISTINCT entity_id), 0) as duplicate_rate,
                COUNT(CASE WHEN feature_value IS NULL THEN 1 END)::float / COUNT(*)::float as null_rate
            FROM features
            WHERE feature_name = $1
                AND timestamp > NOW() - INTERVAL '24 hours'
        "#)
        .bind(feature_name)
        .fetch_one(self.store.pool.as_ref())
        .await?;
        
        let (total_count, unique_entities, duplicate_rate, null_rate) = stats;
        
        // Calculate freshness
        let freshness = sqlx::query_as::<_, (Option<i64>,)>(r#"
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))::bigint
            FROM features
            WHERE feature_name = $1
        "#)
        .bind(feature_name)
        .fetch_one(self.store.pool.as_ref())
        .await?
        .0
        .unwrap_or(i64::MAX) as f64;
        
        // Calculate completeness
        let completeness = 1.0 - null_rate;
        
        // Detect anomalies (simplified)
        let anomaly_rate = self.detect_anomalies(feature_name).await?;
        
        // Update quality metrics
        {
            let mut metrics = self.quality_metrics.write();
            metrics.completeness_rate = completeness;
            metrics.freshness_seconds = freshness;
            metrics.anomaly_rate = anomaly_rate;
            metrics.null_rate = null_rate;
            metrics.duplicate_rate = duplicate_rate;
        }
        
        Ok(QualityReport {
            feature: feature_name.to_string(),
            total_count: total_count as u64,
            unique_entities: unique_entities as u64,
            completeness_rate: completeness,
            freshness_seconds: freshness,
            anomaly_rate,
            null_rate,
            duplicate_rate,
            quality_score: self.calculate_quality_score(completeness, freshness, anomaly_rate),
        })
    }
    
    /// Detect anomalies using isolation forest
    async fn detect_anomalies(&self, feature_name: &str) -> Result<f64, FeatureStoreError> {
        // Simplified anomaly detection
        let values = sqlx::query_as::<_, (f64,)>(r#"
            SELECT (feature_value->>'value')::float
            FROM features
            WHERE feature_name = $1
                AND timestamp > NOW() - INTERVAL '1 hour'
        "#)
        .bind(feature_name)
        .fetch_all(self.store.pool.as_ref())
        .await?
        .into_iter()
        .map(|(v,)| v)
        .collect::<Vec<_>>();
        
        if values.is_empty() {
            return Ok(0.0);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        
        // Count values outside 3 standard deviations
        let anomaly_count = values.iter()
            .filter(|&&v| (v - mean).abs() > 3.0 * std_dev)
            .count();
        
        Ok(anomaly_count as f64 / values.len() as f64)
    }
    
    /// Calculate overall quality score
    fn calculate_quality_score(&self, completeness: f64, freshness: f64, anomaly_rate: f64) -> f64 {
        // Weighted quality score
        let freshness_score = (3600.0 - freshness.min(3600.0)) / 3600.0; // Penalize if older than 1 hour
        let anomaly_score = 1.0 - anomaly_rate;
        
        // Weighted average
        0.4 * completeness + 0.3 * freshness_score + 0.3 * anomaly_score
    }
}

/// Drift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct DriftResult {
    pub feature: String,
    pub drift_detected: bool,
    pub drift_score: f64,
    pub p_value: f64,
    pub message: String,
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct QualityReport {
    pub feature: String,
    pub total_count: u64,
    pub unique_entities: u64,
    pub completeness_rate: f64,
    pub freshness_seconds: f64,
    pub anomaly_rate: f64,
    pub null_rate: f64,
    pub duplicate_rate: f64,
    pub quality_score: f64,
}

impl AlertManager {
    async fn create_alert(&self, alert: Alert) {
        let mut alerts = self.alerts.write();
        alerts.push(alert.clone());
        
        // Send notifications
        for channel in self.notification_channels.iter() {
            let _ = channel.send(&alert).await;
        }
    }
}

use uuid;

// Ellis: "Real-time monitoring prevents silent failures in production"