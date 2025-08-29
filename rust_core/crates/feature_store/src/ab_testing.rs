// A/B Testing Manager - Feature Experimentation and Rollout
// DEEP DIVE: Safe feature deployment with statistical validation

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, instrument};
use statrs::distribution::{Normal, ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::feature_registry::FeatureRegistry;

/// A/B testing configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct ABConfig {
    pub min_sample_size: usize,
    pub confidence_level: f64, // e.g., 0.95 for 95%
    pub min_detectable_effect: f64, // MDE as percentage
    pub max_experiments: usize,
    pub auto_stop_enabled: bool,
    pub auto_stop_threshold: f64, // p-value for early stopping
    pub allocation_seed: u64, // For reproducible allocation
}

impl Default for ABConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 1000,
            confidence_level: 0.95,
            min_detectable_effect: 0.02, // 2% MDE
            max_experiments: 10,
            auto_stop_enabled: true,
            auto_stop_threshold: 0.001, // Very confident for early stop
            allocation_seed: 42,
        }
    }
}

/// A/B Test Manager
/// TODO: Add docs
pub struct ABTestManager {
    config: ABConfig,
    registry: Arc<FeatureRegistry>,
    experiments: Arc<RwLock<HashMap<String, Experiment>>>,
    allocations: Arc<RwLock<HashMap<String, ExperimentAllocation>>>,
    results: Arc<RwLock<HashMap<String, ExperimentResults>>>,
    rng: Arc<RwLock<ChaCha8Rng>>,
}

impl ABTestManager {
    /// Create new A/B test manager
    pub async fn new(
        config: ABConfig,
        registry: Arc<FeatureRegistry>,
    ) -> Result<Self> {
        info!("Initializing A/B Test Manager");
        
        let manager = Self {
            config: config.clone(),
            registry,
            experiments: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            rng: Arc::new(RwLock::new(ChaCha8Rng::seed_from_u64(config.allocation_seed))),
        };
        
        // Load existing experiments
        manager.load_experiments().await?;
        
        Ok(manager)
    }
    
    /// Create new experiment
    #[instrument(skip(self, experiment))]
    pub async fn create_experiment(&self, mut experiment: Experiment) -> Result<String> {
        // Validate experiment
        experiment.validate()?;
        
        // Check experiment limit
        if self.experiments.read().await.len() >= self.config.max_experiments {
            return Err(anyhow::anyhow!("Maximum experiments limit reached"));
        }
        
        // Generate experiment ID
        experiment.experiment_id = Uuid::new_v4().to_string();
        
        // Calculate required sample size
        let sample_size = self.calculate_sample_size(
            experiment.expected_improvement,
            self.config.confidence_level,
            0.8, // 80% power
        )?;
        
        experiment.required_sample_size = sample_size;
        
        // Create allocation strategy
        let allocation = ExperimentAllocation {
            experiment_id: experiment.experiment_id.clone(),
            control_allocation: experiment.traffic_allocation * (1.0 - experiment.treatment_split),
            treatment_allocation: experiment.traffic_allocation * experiment.treatment_split,
            hash_seed: self.rng.write().await.gen(),
            entity_assignments: HashMap::new(),
        };
        
        // Store experiment
        self.experiments.write().await.insert(
            experiment.experiment_id.clone(),
            experiment.clone(),
        );
        
        self.allocations.write().await.insert(
            experiment.experiment_id.clone(),
            allocation,
        );
        
        // Initialize results tracking
        self.results.write().await.insert(
            experiment.experiment_id.clone(),
            ExperimentResults::new(experiment.experiment_id.clone()),
        );
        
        info!("Created experiment: {} - {}", experiment.experiment_id, experiment.name);
        Ok(experiment.experiment_id)
    }
    
    /// Get treatment features for entity
    pub async fn get_treatment_features(
        &self,
        experiment_id: &str,
        feature_names: &[String],
    ) -> Result<Vec<String>> {
        let experiments = self.experiments.read().await;
        let experiment = experiments.get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found"))?;
        
        // Check if experiment is active
        if experiment.status != ExperimentStatus::Running {
            return Ok(feature_names.to_vec()); // Return control features
        }
        
        // Determine assignment (simplified - in production would use entity ID)
        let is_treatment = self.rng.write().await.gen_bool(experiment.treatment_split);
        
        if is_treatment {
            // Return treatment features
            let mut features = feature_names.to_vec();
            for (control, treatment) in &experiment.feature_mappings {
                if let Some(idx) = features.iter().position(|f| f == control) {
                    features[idx] = treatment.clone();
                }
            }
            Ok(features)
        } else {
            Ok(feature_names.to_vec())
        }
    }
    
    /// Record experiment observation
    pub async fn record_observation(
        &self,
        experiment_id: &str,
        entity_id: &str,
        is_treatment: bool,
        metric_value: f64,
    ) -> Result<()> {
        let mut results = self.results.write().await;
        
        let result = results.get_mut(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Experiment results not found"))?;
        
        if is_treatment {
            result.treatment_group.observations.push(metric_value);
            result.treatment_group.count += 1;
        } else {
            result.control_group.observations.push(metric_value);
            result.control_group.count += 1;
        }
        
        result.last_updated = Utc::now();
        
        // Check for auto-stop conditions
        if self.config.auto_stop_enabled {
            self.check_auto_stop(experiment_id, result).await?;
        }
        
        Ok(())
    }
    
    /// Get experiment results
    pub async fn get_results(&self, experiment_id: &str) -> Result<ExperimentResults> {
        let mut results = self.results.write().await;
        
        let result = results.get_mut(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Experiment results not found"))?;
        
        // Calculate statistics
        self.calculate_statistics(result)?;
        
        Ok(result.clone())
    }
    
    /// Get active experiment count
    pub async fn get_active_count(&self) -> Result<usize> {
        let experiments = self.experiments.read().await;
        let count = experiments.values()
            .filter(|e| e.status == ExperimentStatus::Running)
            .count();
        Ok(count)
    }
    
    /// Stop experiment
    pub async fn stop_experiment(
        &self,
        experiment_id: &str,
        reason: StopReason,
    ) -> Result<()> {
        let mut experiments = self.experiments.write().await;
        
        if let Some(experiment) = experiments.get_mut(experiment_id) {
            experiment.status = ExperimentStatus::Stopped;
            experiment.stopped_at = Some(Utc::now());
            experiment.stop_reason = Some(reason);
            
            info!("Stopped experiment {}: {:?}", experiment_id, reason);
        }
        
        Ok(())
    }
    
    /// Calculate required sample size
    fn calculate_sample_size(
        &self,
        effect_size: f64,
        confidence: f64,
        power: f64,
    ) -> Result<usize> {
        // Using formula for two-sample t-test
        // n = 2 * (Z_α/2 + Z_β)² * σ² / δ²
        
        let normal = Normal::new(0.0, 1.0)?;
        let z_alpha = normal.inverse_cdf((1.0 + confidence) / 2.0);
        let z_beta = normal.inverse_cdf(power);
        
        // Assume σ = 1 for standardized effect
        let variance = 1.0;
        
        let n = 2.0 * (z_alpha + z_beta).powi(2) * variance / effect_size.powi(2);
        
        Ok(n.ceil() as usize)
    }
    
    /// Calculate statistics for results
    fn calculate_statistics(&self, results: &mut ExperimentResults) -> Result<()> {
        // Control group stats
        if !results.control_group.observations.is_empty() {
            let control_data = &results.control_group.observations;
            results.control_group.mean = Some(control_data.mean());
            results.control_group.variance = Some(control_data.variance());
            results.control_group.confidence_interval = Some(
                self.calculate_confidence_interval(control_data, self.config.confidence_level)?
            );
        }
        
        // Treatment group stats
        if !results.treatment_group.observations.is_empty() {
            let treatment_data = &results.treatment_group.observations;
            results.treatment_group.mean = Some(treatment_data.mean());
            results.treatment_group.variance = Some(treatment_data.variance());
            results.treatment_group.confidence_interval = Some(
                self.calculate_confidence_interval(treatment_data, self.config.confidence_level)?
            );
        }
        
        // Statistical significance test
        if let (Some(control_mean), Some(treatment_mean)) = 
            (results.control_group.mean, results.treatment_group.mean) {
            
            // Two-sample t-test
            let t_stat = self.calculate_t_statistic(
                &results.control_group,
                &results.treatment_group,
            )?;
            
            let df = results.control_group.count + results.treatment_group.count - 2;
            let t_dist = StudentsT::new(0.0, 1.0, df as f64)?;
            
            // Two-tailed test
            results.p_value = Some(2.0 * (1.0 - t_dist.cdf(t_stat.abs())));
            
            // Effect size (Cohen's d)
            let pooled_std = ((results.control_group.variance.unwrap_or(1.0) * 
                             (results.control_group.count - 1) as f64 +
                             results.treatment_group.variance.unwrap_or(1.0) * 
                             (results.treatment_group.count - 1) as f64) /
                             (df as f64)).sqrt();
            
            results.effect_size = Some((treatment_mean - control_mean) / pooled_std);
            
            // Lift
            if control_mean != 0.0 {
                results.lift = Some((treatment_mean - control_mean) / control_mean);
            }
            
            // Statistical significance
            results.is_significant = results.p_value.unwrap_or(1.0) < (1.0 - self.config.confidence_level);
        }
        
        Ok(())
    }
    
    /// Calculate confidence interval
    fn calculate_confidence_interval(
        &self,
        data: &[f64],
        confidence: f64,
    ) -> Result<(f64, f64)> {
        let mean = data.mean();
        let std_err = (data.variance() / data.len() as f64).sqrt();
        
        let t_dist = StudentsT::new(0.0, 1.0, (data.len() - 1) as f64)?;
        let t_value = t_dist.inverse_cdf((1.0 + confidence) / 2.0);
        
        let margin = t_value * std_err;
        
        Ok((mean - margin, mean + margin))
    }
    
    /// Calculate t-statistic
    fn calculate_t_statistic(
        &self,
        control: &GroupStats,
        treatment: &GroupStats,
    ) -> Result<f64> {
        let control_mean = control.mean.unwrap_or(0.0);
        let treatment_mean = treatment.mean.unwrap_or(0.0);
        let control_var = control.variance.unwrap_or(1.0);
        let treatment_var = treatment.variance.unwrap_or(1.0);
        
        let pooled_se = ((control_var / control.count as f64) + 
                        (treatment_var / treatment.count as f64)).sqrt();
        
        if pooled_se == 0.0 {
            return Ok(0.0);
        }
        
        Ok((treatment_mean - control_mean) / pooled_se)
    }
    
    /// Check auto-stop conditions
    async fn check_auto_stop(
        &self,
        experiment_id: &str,
        results: &ExperimentResults,
    ) -> Result<()> {
        // Check if we have enough samples
        if results.control_group.count < self.config.min_sample_size ||
           results.treatment_group.count < self.config.min_sample_size {
            return Ok(());
        }
        
        // Check for significant positive or negative results
        if let Some(p_value) = results.p_value {
            if p_value < self.config.auto_stop_threshold {
                // Very significant result - stop early
                let reason = if results.lift.unwrap_or(0.0) > 0.0 {
                    StopReason::SignificantPositive
                } else {
                    StopReason::SignificantNegative
                };
                
                self.stop_experiment(experiment_id, reason).await?;
                
                warn!("Auto-stopped experiment {} due to {:?}", experiment_id, reason);
            }
        }
        
        Ok(())
    }
    
    /// Load existing experiments
    async fn load_experiments(&self) -> Result<()> {
        // In production, would load from persistent storage
        info!("Loading existing experiments");
        Ok(())
    }
}

/// Experiment definition
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Experiment {
    pub experiment_id: String,
    pub name: String,
    pub description: String,
    pub hypothesis: String,
    
    // Feature configuration
    pub feature_mappings: HashMap<String, String>, // control -> treatment feature names
    
    // Experiment parameters
    pub traffic_allocation: f64, // Percentage of traffic in experiment
    pub treatment_split: f64, // Percentage of experiment traffic in treatment
    pub expected_improvement: f64, // Expected effect size
    
    // Success metrics
    pub primary_metric: String,
    pub secondary_metrics: Vec<String>,
    pub guardrail_metrics: Vec<String>, // Metrics that shouldn't degrade
    
    // Duration
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    
    // Status
    pub status: ExperimentStatus,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub stop_reason: Option<StopReason>,
    
    // Sample size
    pub required_sample_size: usize,
}

impl Experiment {
    /// Validate experiment configuration
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow::anyhow!("Experiment name cannot be empty"));
        }
        
        if self.traffic_allocation <= 0.0 || self.traffic_allocation > 1.0 {
            return Err(anyhow::anyhow!("Traffic allocation must be between 0 and 1"));
        }
        
        if self.treatment_split <= 0.0 || self.treatment_split >= 1.0 {
            return Err(anyhow::anyhow!("Treatment split must be between 0 and 1"));
        }
        
        if self.end_date <= self.start_date {
            return Err(anyhow::anyhow!("End date must be after start date"));
        }
        
        if self.feature_mappings.is_empty() {
            return Err(anyhow::anyhow!("Must specify at least one feature mapping"));
        }
        
        Ok(())
    }
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum ExperimentStatus {
    Draft,
    Running,
    Paused,
    Stopped,
    Completed,
}

/// Stop reason
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum StopReason {
    Manual,
    Expired,
    SignificantPositive,
    SignificantNegative,
    GuardrailViolation,
    Error,
}

/// Experiment allocation
#[derive(Debug, Clone)]
struct ExperimentAllocation {
    experiment_id: String,
    control_allocation: f64,
    treatment_allocation: f64,
    hash_seed: u64,
    entity_assignments: HashMap<String, bool>, // entity_id -> is_treatment
}

/// Experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ExperimentResults {
    pub experiment_id: String,
    pub control_group: GroupStats,
    pub treatment_group: GroupStats,
    
    // Statistical results
    pub p_value: Option<f64>,
    pub effect_size: Option<f64>, // Cohen's d
    pub lift: Option<f64>, // Percentage improvement
    pub is_significant: bool,
    
    // Timing
    pub started_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl ExperimentResults {
    fn new(experiment_id: String) -> Self {
        Self {
            experiment_id,
            control_group: GroupStats::new(),
            treatment_group: GroupStats::new(),
            p_value: None,
            effect_size: None,
            lift: None,
            is_significant: false,
            started_at: Utc::now(),
            last_updated: Utc::now(),
        }
    }
}

/// Group statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct GroupStats {
    pub count: usize,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub confidence_interval: Option<(f64, f64)>,
    pub observations: Vec<f64>,
}

impl GroupStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: None,
            variance: None,
            confidence_interval: None,
            observations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_experiment_creation() {
        let config = ABConfig::default();
        let registry = Arc::new(FeatureRegistry::new(Default::default()).await.unwrap());
        let manager = ABTestManager::new(config, registry).await.unwrap();
        
        let experiment = Experiment {
            experiment_id: String::new(),
            name: "New Feature Test".to_string(),
            description: "Testing new price feature".to_string(),
            hypothesis: "New feature improves prediction by 5%".to_string(),
            feature_mappings: vec![
                ("price_sma_20".to_string(), "price_sma_20_v2".to_string())
            ].into_iter().collect(),
            traffic_allocation: 0.1, // 10% of traffic
            treatment_split: 0.5, // 50/50 split
            expected_improvement: 0.05,
            primary_metric: "sharpe_ratio".to_string(),
            secondary_metrics: vec!["profit".to_string()],
            guardrail_metrics: vec!["max_drawdown".to_string()],
            start_date: Utc::now(),
            end_date: Utc::now() + Duration::days(7),
            status: ExperimentStatus::Running,
            created_by: "quant_team".to_string(),
            created_at: Utc::now(),
            stopped_at: None,
            stop_reason: None,
            required_sample_size: 0,
        };
        
        let exp_id = manager.create_experiment(experiment).await.unwrap();
        assert!(!exp_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_statistical_significance() {
        let config = ABConfig::default();
        let registry = Arc::new(FeatureRegistry::new(Default::default()).await.unwrap());
        let manager = ABTestManager::new(config, registry).await.unwrap();
        
        // Create sample results
        let mut results = ExperimentResults::new("test".to_string());
        
        // Control group: mean=100, std=10
        for _ in 0..100 {
            results.control_group.observations.push(100.0 + rand::random::<f64>() * 20.0 - 10.0);
            results.control_group.count += 1;
        }
        
        // Treatment group: mean=105, std=10 (5% improvement)
        for _ in 0..100 {
            results.treatment_group.observations.push(105.0 + rand::random::<f64>() * 20.0 - 10.0);
            results.treatment_group.count += 1;
        }
        
        manager.calculate_statistics(&mut results).unwrap();
        
        assert!(results.p_value.is_some());
        assert!(results.effect_size.is_some());
        assert!(results.lift.is_some());
    }
}