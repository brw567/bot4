// Model Registry with Versioning System
// Owner: Morgan | ML Lead | Phase 3 Week 2
// 360-DEGREE REVIEW REQUIRED: All team members must review
// Target: Zero-downtime model updates, A/B testing support

use std::sync::Arc;
use std::collections::{HashMap, BTreeMap};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// ============================================================================
// 360-DEGREE REVIEW CHECKPOINT #1: Registry Architecture
// Reviewers: Alex (Design), Sam (Code Quality), Jordan (Performance)
// ============================================================================

/// Model metadata for registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: Uuid,
    pub name: String,
    pub version: ModelVersion,
    pub model_type: ModelType,
    pub created_at: DateTime<Utc>,
    pub deployed_at: Option<DateTime<Utc>>,
    pub status: ModelStatus,
    pub metrics: ModelMetrics,
    pub config: serde_json::Value,
    pub tags: Vec<String>,
    pub shadow_mode: bool,
    pub traffic_percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub build: Option<String>,
}

impl ModelVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            build: None,
        }
    }
    
    pub fn to_string(&self) -> String {
        match &self.build {
            Some(build) => format!("{}.{}.{}+{}", self.major, self.minor, self.patch, build),
            None => format!("{}.{}.{}", self.major, self.minor, self.patch),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    ARIMA,
    LSTM,
    GRU,
    Transformer,
    Ensemble,
    RandomForest,
    XGBoost,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Training,
    Validating,
    Staging,
    Production,
    Shadow,      // Running in parallel for comparison
    Deprecated,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub mse: f64,
    pub mae: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub custom: HashMap<String, f64>,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mse: f64::INFINITY,
            mae: f64::INFINITY,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            custom: HashMap::new(),
        }
    }
}

// ============================================================================
// 360-DEGREE REVIEW CHECKPOINT #2: Registry Implementation
// Reviewers: Casey (Integration), Quinn (Risk), Avery (Data)
// ============================================================================

pub struct ModelRegistry {
    // All registered models
    models: Arc<RwLock<HashMap<Uuid, Arc<ModelMetadata>>>>,
    
    // Version index for quick lookup
    version_index: Arc<RwLock<BTreeMap<(String, ModelVersion), Uuid>>>,
    
    // Active models by purpose
    active_models: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
    
    // Model performance history
    performance_history: Arc<RwLock<HashMap<Uuid, Vec<PerformanceSnapshot>>>>,
    
    // A/B test configurations
    ab_tests: Arc<RwLock<HashMap<String, ABTestConfig>>>,
    
    // Deployment strategy
    deployment_strategy: DeploymentStrategy,
}

impl ModelRegistry {
    pub fn new(deployment_strategy: DeploymentStrategy) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            version_index: Arc::new(RwLock::new(BTreeMap::new())),
            active_models: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            ab_tests: Arc::new(RwLock::new(HashMap::new())),
            deployment_strategy,
        }
    }
    
    /// Register a new model
    /// Sam: Validate all metadata fields
    pub fn register_model(&self, metadata: ModelMetadata) -> Result<Uuid, RegistryError> {
        // Validation per Quinn's requirements
        if metadata.traffic_percentage < 0.0 || metadata.traffic_percentage > 1.0 {
            return Err(RegistryError::InvalidTrafficPercentage);
        }
        
        let id = metadata.id;
        let version_key = (metadata.name.clone(), metadata.version.clone());
        
        // Check for duplicate version
        if self.version_index.read().contains_key(&version_key) {
            return Err(RegistryError::DuplicateVersion {
                name: metadata.name.clone(),
                version: metadata.version.to_string(),
            });
        }
        
        // Store model
        self.models.write().insert(id, Arc::new(metadata.clone()));
        self.version_index.write().insert(version_key, id);
        
        // Initialize performance history
        self.performance_history.write().insert(id, Vec::new());
        
        Ok(id)
    }
    
    /// Deploy model to production
    /// Quinn: Ensure gradual rollout with monitoring
    pub fn deploy_model(&self, id: Uuid, purpose: String) -> Result<DeploymentResult, RegistryError> {
        let models = self.models.read();
        let model = models.get(&id)
            .ok_or(RegistryError::ModelNotFound(id))?;
        
        match self.deployment_strategy {
            DeploymentStrategy::Immediate => {
                self.deploy_immediate(id, purpose, model.clone())
            }
            DeploymentStrategy::Canary { initial_percentage, ramp_duration } => {
                self.deploy_canary(id, purpose, model.clone(), initial_percentage, ramp_duration)
            }
            DeploymentStrategy::BlueGreen => {
                self.deploy_blue_green(id, purpose, model.clone())
            }
            DeploymentStrategy::Shadow => {
                self.deploy_shadow(id, purpose, model.clone())
            }
        }
    }
    
    fn deploy_immediate(
        &self,
        id: Uuid,
        purpose: String,
        model: Arc<ModelMetadata>,
    ) -> Result<DeploymentResult, RegistryError> {
        let mut active = self.active_models.write();
        active.insert(purpose.clone(), vec![id]);
        
        // Update model status
        if let Some(model_mut) = self.models.write().get_mut(&id) {
            let mut metadata = (**model_mut).clone();
            metadata.status = ModelStatus::Production;
            metadata.deployed_at = Some(Utc::now());
            metadata.traffic_percentage = 1.0;
            *model_mut = Arc::new(metadata);
        }
        
        Ok(DeploymentResult {
            model_id: id,
            deployment_type: "immediate".to_string(),
            traffic_percentage: 1.0,
            shadow_mode: false,
        })
    }
    
    fn deploy_canary(
        &self,
        id: Uuid,
        purpose: String,
        model: Arc<ModelMetadata>,
        initial_percentage: f32,
        ramp_duration: std::time::Duration,
    ) -> Result<DeploymentResult, RegistryError> {
        // Start with small percentage of traffic
        let mut active = self.active_models.write();
        let current_models = active.entry(purpose.clone()).or_insert_with(Vec::new);
        
        // Keep existing model(s) active
        if !current_models.contains(&id) {
            current_models.push(id);
        }
        
        // Update model with canary percentage
        if let Some(model_mut) = self.models.write().get_mut(&id) {
            let mut metadata = (**model_mut).clone();
            metadata.status = ModelStatus::Production;
            metadata.deployed_at = Some(Utc::now());
            metadata.traffic_percentage = initial_percentage;
            *model_mut = Arc::new(metadata);
        }
        
        Ok(DeploymentResult {
            model_id: id,
            deployment_type: format!("canary_{:.1}%", initial_percentage * 100.0),
            traffic_percentage: initial_percentage,
            shadow_mode: false,
        })
    }
    
    fn deploy_blue_green(
        &self,
        id: Uuid,
        purpose: String,
        model: Arc<ModelMetadata>,
    ) -> Result<DeploymentResult, RegistryError> {
        // Prepare new model in staging
        if let Some(model_mut) = self.models.write().get_mut(&id) {
            let mut metadata = (**model_mut).clone();
            metadata.status = ModelStatus::Staging;
            *model_mut = Arc::new(metadata);
        }
        
        Ok(DeploymentResult {
            model_id: id,
            deployment_type: "blue_green_staging".to_string(),
            traffic_percentage: 0.0,
            shadow_mode: false,
        })
    }
    
    fn deploy_shadow(
        &self,
        id: Uuid,
        purpose: String,
        model: Arc<ModelMetadata>,
    ) -> Result<DeploymentResult, RegistryError> {
        // Run in parallel without serving traffic
        if let Some(model_mut) = self.models.write().get_mut(&id) {
            let mut metadata = (**model_mut).clone();
            metadata.status = ModelStatus::Shadow;
            metadata.shadow_mode = true;
            metadata.deployed_at = Some(Utc::now());
            *model_mut = Arc::new(metadata);
        }
        
        let mut active = self.active_models.write();
        let current_models = active.entry(format!("{}_shadow", purpose)).or_insert_with(Vec::new);
        current_models.push(id);
        
        Ok(DeploymentResult {
            model_id: id,
            deployment_type: "shadow".to_string(),
            traffic_percentage: 0.0,
            shadow_mode: true,
        })
    }
    
    /// Get model for inference based on routing rules
    /// Jordan: Optimize for <10ns routing decision
    #[inline(always)]
    pub fn get_model_for_inference(&self, purpose: &str) -> Option<Uuid> {
        let active = self.active_models.read();
        
        if let Some(models) = active.get(purpose) {
            if models.is_empty() {
                return None;
            }
            
            // Check for A/B test
            if let Some(ab_test) = self.ab_tests.read().get(purpose) {
                return Some(self.select_ab_model(&ab_test, models));
            }
            
            // Get model with highest traffic percentage
            let models_read = self.models.read();
            let mut best_model = models[0];
            let mut best_percentage = 0.0;
            
            for &model_id in models {
                if let Some(metadata) = models_read.get(&model_id) {
                    if metadata.traffic_percentage > best_percentage {
                        best_model = model_id;
                        best_percentage = metadata.traffic_percentage;
                    }
                }
            }
            
            Some(best_model)
        } else {
            None
        }
    }
    
    fn select_ab_model(&self, ab_test: &ABTestConfig, models: &[Uuid]) -> Uuid {
        // Simple random selection based on traffic split
        let random = rand::random::<f32>();
        let mut cumulative = 0.0;
        
        for (model_id, percentage) in &ab_test.model_splits {
            cumulative += percentage;
            if random <= cumulative && models.contains(model_id) {
                return *model_id;
            }
        }
        
        models[0] // Fallback
    }
    
    /// Record model performance
    pub fn record_performance(&self, id: Uuid, snapshot: PerformanceSnapshot) {
        if let Some(history) = self.performance_history.write().get_mut(&id) {
            history.push(snapshot);
            
            // Keep only last 1000 snapshots
            if history.len() > 1000 {
                history.drain(0..history.len() - 1000);
            }
        }
    }
    
    /// Compare model performance
    pub fn compare_models(&self, id1: Uuid, id2: Uuid) -> Result<ComparisonResult, RegistryError> {
        let history = self.performance_history.read();
        
        let perf1 = history.get(&id1)
            .ok_or(RegistryError::ModelNotFound(id1))?;
        let perf2 = history.get(&id2)
            .ok_or(RegistryError::ModelNotFound(id2))?;
        
        if perf1.is_empty() || perf2.is_empty() {
            return Err(RegistryError::InsufficientData);
        }
        
        // Calculate average metrics
        let avg1 = Self::calculate_average_performance(perf1);
        let avg2 = Self::calculate_average_performance(perf2);
        
        Ok(ComparisonResult {
            model1_id: id1,
            model2_id: id2,
            model1_metrics: avg1,
            model2_metrics: avg2,
            winner: if avg1.profit_factor > avg2.profit_factor { id1 } else { id2 },
            confidence: 0.95, // Would use statistical test in production
        })
    }
    
    fn calculate_average_performance(snapshots: &[PerformanceSnapshot]) -> ModelMetrics {
        let mut metrics = ModelMetrics::default();
        let n = snapshots.len() as f64;
        
        for snapshot in snapshots {
            metrics.accuracy += snapshot.accuracy / n;
            metrics.precision += snapshot.precision / n;
            metrics.sharpe_ratio += snapshot.sharpe_ratio / n;
            metrics.profit_factor += snapshot.profit_factor / n;
        }
        
        metrics
    }
}

// ============================================================================
// 360-DEGREE REVIEW CHECKPOINT #3: Deployment Strategies
// Reviewers: Alex (Architecture), Quinn (Risk Management)
// ============================================================================

#[derive(Debug, Clone)]
pub enum DeploymentStrategy {
    Immediate,
    Canary {
        initial_percentage: f32,
        ramp_duration: std::time::Duration,
    },
    BlueGreen,
    Shadow,
}

#[derive(Debug, Clone)]
pub struct DeploymentResult {
    pub model_id: Uuid,
    pub deployment_type: String,
    pub traffic_percentage: f32,
    pub shadow_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ABTestConfig {
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub model_splits: HashMap<Uuid, f32>,
    pub success_metric: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub accuracy: f64,
    pub precision: f64,
    pub latency_ms: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
}

#[derive(Debug)]
pub struct ComparisonResult {
    pub model1_id: Uuid,
    pub model2_id: Uuid,
    pub model1_metrics: ModelMetrics,
    pub model2_metrics: ModelMetrics,
    pub winner: Uuid,
    pub confidence: f64,
}

// ============================================================================
// 360-DEGREE REVIEW CHECKPOINT #4: Error Handling
// Reviewers: Casey (Integration), Sam (Code Quality)
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(Uuid),
    
    #[error("Duplicate model version: {name} v{version}")]
    DuplicateVersion { name: String, version: String },
    
    #[error("Invalid traffic percentage (must be 0.0-1.0)")]
    InvalidTrafficPercentage,
    
    #[error("Insufficient performance data")]
    InsufficientData,
    
    #[error("Deployment failed: {0}")]
    DeploymentFailed(String),
}

// ============================================================================
// TESTS - Riley's 100% Coverage Requirement
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_registration() {
        let registry = ModelRegistry::new(DeploymentStrategy::Immediate);
        
        let metadata = ModelMetadata {
            id: Uuid::new_v4(),
            name: "test_model".to_string(),
            version: ModelVersion::new(1, 0, 0),
            model_type: ModelType::ARIMA,
            created_at: Utc::now(),
            deployed_at: None,
            status: ModelStatus::Training,
            metrics: ModelMetrics::default(),
            config: serde_json::json!({}),
            tags: vec!["test".to_string()],
            shadow_mode: false,
            traffic_percentage: 0.0,
        };
        
        let id = registry.register_model(metadata.clone()).unwrap();
        assert_eq!(id, metadata.id);
    }
    
    #[test]
    fn test_duplicate_version_rejection() {
        let registry = ModelRegistry::new(DeploymentStrategy::Immediate);
        
        let metadata = ModelMetadata {
            id: Uuid::new_v4(),
            name: "test_model".to_string(),
            version: ModelVersion::new(1, 0, 0),
            model_type: ModelType::ARIMA,
            created_at: Utc::now(),
            deployed_at: None,
            status: ModelStatus::Training,
            metrics: ModelMetrics::default(),
            config: serde_json::json!({}),
            tags: vec![],
            shadow_mode: false,
            traffic_percentage: 0.0,
        };
        
        registry.register_model(metadata.clone()).unwrap();
        
        let mut duplicate = metadata;
        duplicate.id = Uuid::new_v4();
        let result = registry.register_model(duplicate);
        assert!(matches!(result, Err(RegistryError::DuplicateVersion { .. })));
    }
    
    #[test]
    fn test_immediate_deployment() {
        let registry = ModelRegistry::new(DeploymentStrategy::Immediate);
        
        let metadata = ModelMetadata {
            id: Uuid::new_v4(),
            name: "prod_model".to_string(),
            version: ModelVersion::new(1, 0, 0),
            model_type: ModelType::LSTM,
            created_at: Utc::now(),
            deployed_at: None,
            status: ModelStatus::Staging,
            metrics: ModelMetrics::default(),
            config: serde_json::json!({}),
            tags: vec![],
            shadow_mode: false,
            traffic_percentage: 0.0,
        };
        
        let id = registry.register_model(metadata).unwrap();
        let result = registry.deploy_model(id, "price_prediction".to_string()).unwrap();
        
        assert_eq!(result.traffic_percentage, 1.0);
        assert!(!result.shadow_mode);
    }
    
    #[test]
    fn test_canary_deployment() {
        let registry = ModelRegistry::new(DeploymentStrategy::Canary {
            initial_percentage: 0.1,
            ramp_duration: std::time::Duration::from_secs(3600),
        });
        
        let metadata = ModelMetadata {
            id: Uuid::new_v4(),
            name: "canary_model".to_string(),
            version: ModelVersion::new(2, 0, 0),
            model_type: ModelType::Transformer,
            created_at: Utc::now(),
            deployed_at: None,
            status: ModelStatus::Staging,
            metrics: ModelMetrics::default(),
            config: serde_json::json!({}),
            tags: vec![],
            shadow_mode: false,
            traffic_percentage: 0.0,
        };
        
        let id = registry.register_model(metadata).unwrap();
        let result = registry.deploy_model(id, "signal_generation".to_string()).unwrap();
        
        assert_eq!(result.traffic_percentage, 0.1);
        assert!(result.deployment_type.contains("canary"));
    }
    
    #[test]
    fn test_model_inference_routing() {
        let registry = ModelRegistry::new(DeploymentStrategy::Immediate);
        
        let metadata = ModelMetadata {
            id: Uuid::new_v4(),
            name: "inference_model".to_string(),
            version: ModelVersion::new(1, 0, 0),
            model_type: ModelType::XGBoost,
            created_at: Utc::now(),
            deployed_at: None,
            status: ModelStatus::Production,
            metrics: ModelMetrics::default(),
            config: serde_json::json!({}),
            tags: vec![],
            shadow_mode: false,
            traffic_percentage: 1.0,
        };
        
        let id = registry.register_model(metadata).unwrap();
        registry.deploy_model(id, "routing_test".to_string()).unwrap();
        
        let selected = registry.get_model_for_inference("routing_test");
        assert_eq!(selected, Some(id));
    }
}

// Performance characteristics:
// - Registration: O(1) with hash map
// - Deployment: O(1) for immediate, O(n) for traffic split
// - Inference routing: O(1) average case
// - Memory: O(models * history_size)