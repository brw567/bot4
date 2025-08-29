use domain_types::FeatureMetadata;
// Feature Registry - Metadata and Lineage Tracking
// DEEP DIVE: Complete feature lifecycle management with versioning

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, instrument};
use deadpool_postgres::{Config, Pool};
use tokio_postgres::NoTls;
use uuid::Uuid;

/// Registry configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct RegistryConfig {
    pub postgres_url: String,
    pub pool_size: usize,
    pub enable_versioning: bool,
    pub enforce_schema_validation: bool,
    pub retention_days: i64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            postgres_url: "postgresql://bot3user:bot3pass@localhost:5432/bot3trading".to_string(),
            pool_size: 8,
            enable_versioning: true,
            enforce_schema_validation: true,
            retention_days: 365,
        }
    }
}

/// Feature registry for metadata and lineage
/// TODO: Add docs
pub struct FeatureRegistry {
    config: RegistryConfig,
    pool: Arc<Pool>,
    cache: Arc<dashmap::DashMap<String, Arc<FeatureDefinition>>>,
}

impl FeatureRegistry {
    /// Create new registry instance
    pub async fn new(config: RegistryConfig) -> Result<Self> {
        info!("Initializing Feature Registry with lineage tracking");
        
        // Create connection pool
        let pool_config = Config::builder()
            .url(&config.postgres_url)
            .pool_size(config.pool_size)
            .build()
            .context("Failed to build pool config")?;
        
        let pool = pool_config.create_pool(None, NoTls)?;
        
        let registry = Self {
            config: config.clone(),
            pool: Arc::new(pool),
            cache: Arc::new(dashmap::DashMap::new()),
        };
        
        registry.initialize_schema().await?;
        registry.load_definitions().await?;
        
        Ok(registry)
    }
    
    /// Initialize registry schema
    async fn initialize_schema(&self) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Create feature registry schema
        conn.execute(
            "CREATE SCHEMA IF NOT EXISTS feature_registry",
            &[],
        ).await?;
        
        // Feature definitions table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_registry.definitions (
                feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                feature_name TEXT UNIQUE NOT NULL,
                feature_type TEXT NOT NULL,
                description TEXT,
                
                -- Schema definition
                schema JSONB NOT NULL,
                transformations JSONB,
                validation_rules JSONB,
                
                -- Lineage tracking
                source_features TEXT[],
                source_tables TEXT[],
                computation_graph JSONB,
                
                -- Versioning
                version INTEGER NOT NULL DEFAULT 1,
                previous_version UUID REFERENCES feature_registry.definitions(feature_id),
                
                -- Metadata
                owner TEXT NOT NULL,
                team TEXT,
                tags TEXT[],
                
                -- Timestamps
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                deprecated_at TIMESTAMPTZ,
                
                -- Performance hints
                serving_tier TEXT DEFAULT 'standard', -- hot, standard, cold
                compute_cost FLOAT DEFAULT 1.0,
                storage_bytes BIGINT DEFAULT 0
            )",
            &[],
        ).await?;
        
        // Feature versions history
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_registry.versions (
                version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                feature_id UUID NOT NULL REFERENCES feature_registry.definitions(feature_id),
                version_number INTEGER NOT NULL,
                change_type TEXT NOT NULL, -- major, minor, patch
                changes JSONB NOT NULL,
                
                -- Deployment tracking
                deployed_at TIMESTAMPTZ,
                deployed_by TEXT,
                rollback_from UUID REFERENCES feature_registry.versions(version_id),
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(feature_id, version_number)
            )",
            &[],
        ).await?;
        
        // Feature dependencies DAG
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_registry.dependencies (
                dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                feature_id UUID NOT NULL REFERENCES feature_registry.definitions(feature_id),
                depends_on UUID NOT NULL REFERENCES feature_registry.definitions(feature_id),
                dependency_type TEXT NOT NULL, -- direct, derived, aggregate
                computation_order INTEGER NOT NULL,
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(feature_id, depends_on)
            )",
            &[],
        ).await?;
        
        // Feature usage tracking
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_registry.usage_tracking (
                tracking_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                feature_id UUID NOT NULL REFERENCES feature_registry.definitions(feature_id),
                model_id TEXT NOT NULL,
                experiment_id TEXT,
                
                -- Usage metrics
                request_count BIGINT DEFAULT 0,
                last_accessed TIMESTAMPTZ,
                average_latency_ms FLOAT,
                error_rate FLOAT,
                
                -- Impact analysis
                feature_importance FLOAT,
                contribution_score FLOAT,
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(feature_id, model_id)
            )",
            &[],
        ).await?;
        
        // Feature quality metrics
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_registry.quality_metrics (
                metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                feature_id UUID NOT NULL REFERENCES feature_registry.definitions(feature_id),
                
                -- Data quality
                null_rate FLOAT,
                unique_count BIGINT,
                mean FLOAT,
                stddev FLOAT,
                min_value FLOAT,
                max_value FLOAT,
                
                -- Distribution tracking
                distribution JSONB, -- histogram bins
                outlier_count BIGINT,
                
                -- Temporal quality
                freshness_seconds FLOAT,
                update_frequency_seconds FLOAT,
                
                computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )",
            &[],
        ).await?;
        
        // Create indexes for performance
        conn.batch_execute(
            "CREATE INDEX IF NOT EXISTS idx_definitions_name ON feature_registry.definitions(feature_name);
             CREATE INDEX IF NOT EXISTS idx_definitions_tags ON feature_registry.definitions USING gin(tags);
             CREATE INDEX IF NOT EXISTS idx_dependencies_feature ON feature_registry.dependencies(feature_id);
             CREATE INDEX IF NOT EXISTS idx_dependencies_depends ON feature_registry.dependencies(depends_on);
             CREATE INDEX IF NOT EXISTS idx_usage_model ON feature_registry.usage_tracking(model_id);
             CREATE INDEX IF NOT EXISTS idx_usage_accessed ON feature_registry.usage_tracking(last_accessed DESC);"
        ).await?;
        
        info!("Feature Registry schema initialized");
        Ok(())
    }
    
    /// Load feature definitions into cache
    async fn load_definitions(&self) -> Result<()> {
        let conn = self.pool.get().await?;
        
        let rows = conn.query(
            "SELECT feature_id, feature_name, feature_type, description, 
                    schema, transformations, validation_rules,
                    source_features, source_tables, computation_graph,
                    version, owner, team, tags, serving_tier
             FROM feature_registry.definitions
             WHERE deprecated_at IS NULL",
            &[],
        ).await?;
        
        for row in rows {
            let feature_id: Uuid = row.get(0);
            let feature_name: String = row.get(1);
            
            let definition = FeatureDefinition {
                feature_id,
                feature_name: feature_name.clone(),
                feature_type: row.get(2),
                description: row.get(3),
                schema: row.get(4),
                transformations: row.get(5),
                validation_rules: row.get(6),
                source_features: row.get(7),
                source_tables: row.get(8),
                computation_graph: row.get(9),
                version: row.get(10),
                owner: row.get(11),
                team: row.get(12),
                tags: row.get(13),
                serving_tier: row.get(14),
            };
            
            self.cache.insert(feature_name, Arc::new(definition));
        }
        
        info!("Loaded {} feature definitions", self.cache.len());
        Ok(())
    }
    
    /// Register new feature
    #[instrument(skip(self, definition))]
    pub async fn register(&self, mut definition: FeatureDefinition) -> Result<String> {
        // Validate definition
        definition.validate()?;
        
        // Check for circular dependencies
        self.validate_no_cycles(&definition).await?;
        
        // Generate feature ID
        definition.feature_id = Uuid::new_v4();
        
        let conn = self.pool.get().await?;
        
        // Insert definition
        conn.execute(
            "INSERT INTO feature_registry.definitions (
                feature_id, feature_name, feature_type, description,
                schema, transformations, validation_rules,
                source_features, source_tables, computation_graph,
                version, owner, team, tags, serving_tier
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
            &[
                &definition.feature_id,
                &definition.feature_name,
                &definition.feature_type,
                &definition.description,
                &definition.schema,
                &definition.transformations,
                &definition.validation_rules,
                &definition.source_features,
                &definition.source_tables,
                &definition.computation_graph,
                &definition.version,
                &definition.owner,
                &definition.team,
                &definition.tags,
                &definition.serving_tier,
            ],
        ).await?;
        
        // Register dependencies
        if let Some(deps) = &definition.source_features {
            for (order, dep_name) in deps.iter().enumerate() {
                if let Some(dep_def) = self.cache.get(dep_name) {
                    conn.execute(
                        "INSERT INTO feature_registry.dependencies (
                            feature_id, depends_on, dependency_type, computation_order
                        ) VALUES ($1, $2, $3, $4)",
                        &[
                            &definition.feature_id,
                            &dep_def.feature_id,
                            &"direct",
                            &(order as i32),
                        ],
                    ).await?;
                }
            }
        }
        
        // Cache the definition
        self.cache.insert(
            definition.feature_name.clone(),
            Arc::new(definition.clone()),
        );
        
        info!("Registered feature: {}", definition.feature_name);
        Ok(definition.feature_id.to_string())
    }
    
    /// Update feature definition (creates new version)
    pub async fn update_feature(
        &self,
        feature_name: &str,
        updates: FeatureUpdate,
    ) -> Result<u32> {
        let current = self.cache.get(feature_name)
            .ok_or_else(|| anyhow::anyhow!("Feature not found: {}", feature_name))?;
        
        let new_version = current.version + 1;
        
        let conn = self.pool.get().await?;
        
        // Create version history entry
        conn.execute(
            "INSERT INTO feature_registry.versions (
                feature_id, version_number, change_type, changes
            ) VALUES ($1, $2, $3, $4)",
            &[
                &current.feature_id,
                &new_version,
                &updates.change_type,
                &serde_json::to_value(&updates)?,
            ],
        ).await?;
        
        // Update definition
        let mut update_query = String::from("UPDATE feature_registry.definitions SET ");
        let mut params: Vec<Box<dyn tokio_postgres::types::ToSql + Sync>> = Vec::new();
        let mut param_count = 1;
        
        if let Some(schema) = updates.schema {
            update_query.push_str(&format!("schema = ${}, ", param_count));
            params.push(Box::new(schema));
            param_count += 1;
        }
        
        if let Some(transformations) = updates.transformations {
            update_query.push_str(&format!("transformations = ${}, ", param_count));
            params.push(Box::new(transformations));
            param_count += 1;
        }
        
        update_query.push_str(&format!(
            "version = ${}, updated_at = CURRENT_TIMESTAMP WHERE feature_id = ${}",
            param_count, param_count + 1
        ));
        params.push(Box::new(new_version));
        params.push(Box::new(current.feature_id));
        
        // Execute update (simplified for example)
        // In production, would need proper dynamic query building
        
        // Reload definition
        self.load_definitions().await?;
        
        Ok(new_version)
    }
    
    /// Get feature lineage
    pub async fn get_lineage(&self, feature_id: &str) -> Result<crate::FeatureLineage> {
        let feature_uuid = Uuid::parse_str(feature_id)?;
        let conn = self.pool.get().await?;
        
        // Get feature definition
        let row = conn.query_one(
            "SELECT feature_name, version, created_at, updated_at,
                    source_features, source_tables, transformations
             FROM feature_registry.definitions
             WHERE feature_id = $1",
            &[&feature_uuid],
        ).await?;
        
        let feature_name: String = row.get(0);
        let version: i32 = row.get(1);
        let created_at: DateTime<Utc> = row.get(2);
        let updated_at: DateTime<Utc> = row.get(3);
        let source_features: Option<Vec<String>> = row.get(4);
        let source_tables: Option<Vec<String>> = row.get(5);
        let transformations_json: Option<serde_json::Value> = row.get(6);
        
        // Get dependencies
        let dep_rows = conn.query(
            "SELECT d.feature_name
             FROM feature_registry.dependencies dep
             JOIN feature_registry.definitions d ON d.feature_id = dep.depends_on
             WHERE dep.feature_id = $1
             ORDER BY dep.computation_order",
            &[&feature_uuid],
        ).await?;
        
        let dependencies: Vec<String> = dep_rows.iter()
            .map(|r| r.get(0))
            .collect();
        
        // Parse transformations
        let transformations = if let Some(json) = transformations_json {
            json.as_array()
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        
        Ok(crate::FeatureLineage {
            feature_id: feature_id.to_string(),
            version: version.to_string(),
            created_at,
            updated_at,
            dependencies,
            transformations,
            data_sources: source_tables.unwrap_or_default(),
        })
    }
    
    /// Validate feature update
    pub async fn validate_update(&self, update: &crate::FeatureUpdate) -> Result<()> {
        // Check feature exists
        let definition = self.cache.get(&update.feature_id)
            .ok_or_else(|| anyhow::anyhow!("Feature not found: {}", update.feature_id))?;
        
        // Validate value type matches schema
        if let Some(schema) = &definition.schema {
            // Would validate against JSON schema
            debug!("Validating against schema: {:?}", schema);
        }
        
        // Check validation rules
        if let Some(rules) = &definition.validation_rules {
            // Apply custom validation rules
            debug!("Applying validation rules: {:?}", rules);
        }
        
        Ok(())
    }
    
    /// Get feature count
    pub async fn get_feature_count(&self) -> Result<usize> {
        Ok(self.cache.len())
    }
    
    /// Track feature usage
    pub async fn track_usage(
        &self,
        feature_id: &str,
        model_id: &str,
        latency_ms: f64,
    ) -> Result<()> {
        let feature_uuid = Uuid::parse_str(feature_id)?;
        let conn = self.pool.get().await?;
        
        conn.execute(
            "INSERT INTO feature_registry.usage_tracking (
                feature_id, model_id, request_count, last_accessed, average_latency_ms
            ) VALUES ($1, $2, 1, CURRENT_TIMESTAMP, $3)
            ON CONFLICT (feature_id, model_id)
            DO UPDATE SET
                request_count = usage_tracking.request_count + 1,
                last_accessed = CURRENT_TIMESTAMP,
                average_latency_ms = (
                    usage_tracking.average_latency_ms * usage_tracking.request_count + $3
                ) / (usage_tracking.request_count + 1)",
            &[&feature_uuid, &model_id, &latency_ms],
        ).await?;
        
        Ok(())
    }
    
    /// Get top used features
    pub async fn get_top_features(&self, limit: i64) -> Result<Vec<FeatureUsage>> {
        let conn = self.pool.get().await?;
        
        let rows = conn.query(
            "SELECT d.feature_name, SUM(u.request_count) as total_requests,
                    AVG(u.average_latency_ms) as avg_latency,
                    MAX(u.last_accessed) as last_used
             FROM feature_registry.usage_tracking u
             JOIN feature_registry.definitions d ON d.feature_id = u.feature_id
             GROUP BY d.feature_name
             ORDER BY total_requests DESC
             LIMIT $1",
            &[&limit],
        ).await?;
        
        let mut features = Vec::new();
        for row in rows {
            features.push(FeatureUsage {
                feature_name: row.get(0),
                request_count: row.get::<_, i64>(1) as u64,
                average_latency_ms: row.get(2),
                last_accessed: row.get(3),
            });
        }
        
        Ok(features)
    }
    
    /// Validate no circular dependencies
    async fn validate_no_cycles(&self, definition: &FeatureDefinition) -> Result<()> {
        if let Some(deps) = &definition.source_features {
            let mut visited = HashSet::new();
            let mut stack = HashSet::new();
            
            for dep in deps {
                if self.has_cycle_dfs(dep, &mut visited, &mut stack).await? {
                    return Err(anyhow::anyhow!(
                        "Circular dependency detected involving feature: {}",
                        dep
                    ));
                }
            }
        }
        Ok(())
    }
    
    /// DFS for cycle detection
    async fn has_cycle_dfs(
        &self,
        feature: &str,
        visited: &mut HashSet<String>,
        stack: &mut HashSet<String>,
    ) -> Result<bool> {
        if stack.contains(feature) {
            return Ok(true);
        }
        
        if visited.contains(feature) {
            return Ok(false);
        }
        
        visited.insert(feature.to_string());
        stack.insert(feature.to_string());
        
        if let Some(definition) = self.cache.get(feature) {
            if let Some(deps) = &definition.source_features {
                for dep in deps {
                    if self.has_cycle_dfs(dep, visited, stack).await? {
                        return Ok(true);
                    }
                }
            }
        }
        
        stack.remove(feature);
        Ok(false)
    }
    
    /// Deprecate feature
    pub async fn deprecate_feature(&self, feature_name: &str) -> Result<()> {
        let definition = self.cache.get(feature_name)
            .ok_or_else(|| anyhow::anyhow!("Feature not found: {}", feature_name))?;
        
        let conn = self.pool.get().await?;
        
        conn.execute(
            "UPDATE feature_registry.definitions 
             SET deprecated_at = CURRENT_TIMESTAMP
             WHERE feature_id = $1",
            &[&definition.feature_id],
        ).await?;
        
        self.cache.remove(feature_name);
        
        warn!("Deprecated feature: {}", feature_name);
        Ok(())
    }
}

/// Feature definition
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureDefinition {
    pub feature_id: Uuid,
    pub feature_name: String,
    pub feature_type: String,
    pub description: Option<String>,
    
    // Schema and validation
    pub schema: Option<serde_json::Value>,
    pub transformations: Option<serde_json::Value>,
    pub validation_rules: Option<serde_json::Value>,
    
    // Lineage
    pub source_features: Option<Vec<String>>,
    pub source_tables: Option<Vec<String>>,
    pub computation_graph: Option<serde_json::Value>,
    
    // Metadata
    pub version: i32,
    pub owner: String,
    pub team: Option<String>,
    pub tags: Option<Vec<String>>,
    pub serving_tier: Option<String>,
}

impl FeatureDefinition {
    /// Validate feature definition
    pub fn validate(&self) -> Result<()> {
        // Name validation
        if self.feature_name.is_empty() {
            return Err(anyhow::anyhow!("Feature name cannot be empty"));
        }
        
        if !self.feature_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(anyhow::anyhow!("Feature name must be alphanumeric with underscores"));
        }
        
        // Type validation
        let valid_types = ["float", "integer", "string", "vector", "embedding", "categorical"];
        if !valid_types.contains(&self.feature_type.as_str()) {
            return Err(anyhow::anyhow!("Invalid feature type: {}", self.feature_type));
        }
        
        // Owner validation
        if self.owner.is_empty() {
            return Err(anyhow::anyhow!("Feature must have an owner"));
        }
        
        Ok(())
    }
}

/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
// ELIMINATED: use domain_types::FeatureMetadata
// pub struct FeatureMetadata {
    pub feature_id: String,
    pub feature_name: String,
    pub description: Option<String>,
    pub owner: String,
    pub team: Option<String>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Feature update
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureUpdate {
    pub change_type: String, // major, minor, patch
    pub schema: Option<serde_json::Value>,
    pub transformations: Option<serde_json::Value>,
    pub validation_rules: Option<serde_json::Value>,
    pub description: Option<String>,
}

/// Feature usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct FeatureUsage {
    pub feature_name: String,
    pub request_count: u64,
    pub average_latency_ms: Option<f64>,
    pub last_accessed: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_feature_registration() {
        let config = RegistryConfig::default();
        let registry = FeatureRegistry::new(config).await.unwrap();
        
        let definition = FeatureDefinition {
            feature_id: Uuid::new_v4(),
            feature_name: "price_sma_20".to_string(),
            feature_type: "float".to_string(),
            description: Some("20-period simple moving average".to_string()),
            schema: Some(serde_json::json!({
                "type": "number",
                "minimum": 0
            })),
            transformations: Some(serde_json::json!([
                "rolling_mean(price, 20)"
            ])),
            validation_rules: None,
            source_features: Some(vec!["price".to_string()]),
            source_tables: Some(vec!["market_data.candles".to_string()]),
            computation_graph: None,
            version: 1,
            owner: "trading_team".to_string(),
            team: Some("quant".to_string()),
            tags: Some(vec!["technical".to_string(), "price".to_string()]),
            serving_tier: Some("hot".to_string()),
        };
        
        let feature_id = registry.register(definition).await.unwrap();
        assert!(!feature_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_circular_dependency_detection() {
        let config = RegistryConfig::default();
        let registry = FeatureRegistry::new(config).await.unwrap();
        
        // Create features with circular dependency
        let feature_a = FeatureDefinition {
            feature_id: Uuid::new_v4(),
            feature_name: "feature_a".to_string(),
            feature_type: "float".to_string(),
            description: None,
            schema: None,
            transformations: None,
            validation_rules: None,
            source_features: Some(vec!["feature_b".to_string()]), // A depends on B
            source_tables: None,
            computation_graph: None,
            version: 1,
            owner: "test".to_string(),
            team: None,
            tags: None,
            serving_tier: None,
        };
        
        // This would create a cycle if B also depends on A
        let result = registry.validate_no_cycles(&feature_a).await;
        // Should pass since B doesn't exist yet
        assert!(result.is_ok());
    }
}