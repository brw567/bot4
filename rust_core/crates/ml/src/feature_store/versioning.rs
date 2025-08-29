//! # FEATURE VERSIONING - Time-travel and reproducibility
//! Cameron (Risk Lead): "Versioning enables perfect backtesting"

use super::*;
use std::collections::BTreeMap;

/// Feature version manager
/// TODO: Add docs
pub struct FeatureVersionManager {
    store: Arc<FeatureStore>,
    versions: Arc<RwLock<BTreeMap<u32, VersionMetadata>>>,
    current_version: Arc<AtomicU32>,
}

/// Version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct VersionMetadata {
    pub version: u32,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub description: String,
    pub schema_changes: Vec<SchemaChange>,
    pub feature_changes: Vec<FeatureChange>,
    pub is_production: bool,
    pub parent_version: Option<u32>,
}

/// Schema change tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum SchemaChange {
    AddFeature { name: String, schema: FeatureSchema },
    RemoveFeature { name: String },
    ModifyFeature { name: String, old: FeatureSchema, new: FeatureSchema },
    AddEntity { entity_type: String },
    RemoveEntity { entity_type: String },
}

/// Feature change tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum FeatureChange {
    ComputationUpdated { feature: String, old_formula: String, new_formula: String },
    SourceChanged { feature: String, old_source: String, new_source: String },
    DependencyAdded { feature: String, dependency: String },
    DependencyRemoved { feature: String, dependency: String },
}

impl FeatureVersionManager {
    pub fn new(store: Arc<FeatureStore>) -> Self {
        Self {
            store,
            versions: Arc::new(RwLock::new(BTreeMap::new())),
            current_version: Arc::new(AtomicU32::new(1)),
        }
    }
    
    /// Create new version
    pub async fn create_version(
        &self,
        description: String,
        created_by: String,
    ) -> Result<u32, FeatureStoreError> {
        let current = self.current_version.load(std::sync::atomic::Ordering::Acquire);
        let new_version = current + 1;
        
        // Create version metadata
        let metadata = VersionMetadata {
            version: new_version,
            created_at: Utc::now(),
            created_by,
            description,
            schema_changes: Vec::new(),
            feature_changes: Vec::new(),
            is_production: false,
            parent_version: Some(current),
        };
        
        // Store in database
        sqlx::query(r#"
            INSERT INTO feature_versions (version, metadata, created_at)
            VALUES ($1, $2, $3)
        "#)
        .bind(new_version as i32)
        .bind(serde_json::to_value(&metadata)?)
        .bind(Utc::now())
        .execute(self.store.pool.as_ref())
        .await?;
        
        // Update in-memory
        self.versions.write().insert(new_version, metadata);
        self.current_version.store(new_version, std::sync::atomic::Ordering::Release);
        
        Ok(new_version)
    }
    
    /// Get features at specific version
    pub async fn get_features_at_version(
        &self,
        entity_id: &str,
        feature_names: &[String],
        version: u32,
    ) -> Result<HashMap<String, FeatureValue>, FeatureStoreError> {
        let query = sqlx::query_as::<_, (String, serde_json::Value)>(r#"
            SELECT feature_name, feature_value
            FROM features
            WHERE entity_id = $1 
                AND feature_name = ANY($2)
                AND version = $3
        "#)
        .bind(entity_id)
        .bind(feature_names)
        .bind(version as i32);
        
        let rows = query.fetch_all(self.store.pool.as_ref()).await?;
        
        let mut features = HashMap::new();
        for (name, value) in rows {
            let feature_value: FeatureValue = serde_json::from_value(value)?;
            features.insert(name, feature_value);
        }
        
        Ok(features)
    }
    
    /// Promote version to production
    pub async fn promote_to_production(&self, version: u32) -> Result<(), FeatureStoreError> {
        // Demote current production
        sqlx::query(r#"
            UPDATE feature_versions 
            SET metadata = jsonb_set(metadata, '{is_production}', 'false')
            WHERE (metadata->>'is_production')::boolean = true
        "#)
        .execute(self.store.pool.as_ref())
        .await?;
        
        // Promote new version
        sqlx::query(r#"
            UPDATE feature_versions 
            SET metadata = jsonb_set(metadata, '{is_production}', 'true')
            WHERE version = $1
        "#)
        .bind(version as i32)
        .execute(self.store.pool.as_ref())
        .await?;
        
        // Update in-memory
        let mut versions = self.versions.write();
        for (_, metadata) in versions.iter_mut() {
            metadata.is_production = false;
        }
        if let Some(metadata) = versions.get_mut(&version) {
            metadata.is_production = true;
        }
        
        Ok(())
    }
    
    /// Create snapshot for reproducibility
    pub async fn create_snapshot(
        &self,
        name: &str,
        entity_ids: &[String],
        feature_names: &[String],
    ) -> Result<String, FeatureStoreError> {
        let snapshot_id = format!("snapshot_{}_{}", name, Utc::now().timestamp());
        
        // Copy features to snapshot table
        sqlx::query(r#"
            INSERT INTO feature_snapshots (snapshot_id, entity_id, feature_name, feature_value, timestamp)
            SELECT $1, entity_id, feature_name, feature_value, timestamp
            FROM features
            WHERE entity_id = ANY($2) AND feature_name = ANY($3)
        "#)
        .bind(&snapshot_id)
        .bind(entity_ids)
        .bind(feature_names)
        .execute(self.store.pool.as_ref())
        .await?;
        
        Ok(snapshot_id)
    }
    
    /// Restore from snapshot
    pub async fn restore_snapshot(
        &self,
        snapshot_id: &str,
    ) -> Result<u64, FeatureStoreError> {
        let result = sqlx::query(r#"
            INSERT INTO features (entity_id, feature_name, entity_type, feature_value, timestamp)
            SELECT entity_id, feature_name, 'restored', feature_value, timestamp
            FROM feature_snapshots
            WHERE snapshot_id = $1
            ON CONFLICT (feature_name, entity_id, timestamp, version) DO NOTHING
        "#)
        .bind(snapshot_id)
        .execute(self.store.pool.as_ref())
        .await?;
        
        Ok(result.rows_affected())
    }
    
    /// Compare versions
    pub async fn compare_versions(
        &self,
        version1: u32,
        version2: u32,
    ) -> Result<VersionComparison, FeatureStoreError> {
        let v1_features = self.get_version_features(version1).await?;
        let v2_features = self.get_version_features(version2).await?;
        
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        
        // Find additions and modifications
        for (name, v2_schema) in &v2_features {
            match v1_features.get(name) {
                None => added.push(name.clone()),
                Some(v1_schema) if v1_schema != v2_schema => {
                    modified.push(name.clone());
                }
                _ => {}
            }
        }
        
        // Find removals
        for name in v1_features.keys() {
            if !v2_features.contains_key(name) {
                removed.push(name.clone());
            }
        }
        
        Ok(VersionComparison {
            version1,
            version2,
            added_features: added,
            removed_features: removed,
            modified_features: modified,
        })
    }
    
    /// Get all features in a version
    async fn get_version_features(&self, version: u32) -> Result<HashMap<String, FeatureSchema>, FeatureStoreError> {
        let result = sqlx::query_as::<_, (String, serde_json::Value)>(r#"
            SELECT feature_name, metadata
            FROM feature_metadata
            WHERE version = $1
        "#)
        .bind(version as i32)
        .fetch_all(self.store.pool.as_ref())
        .await?;
        
        let mut features = HashMap::new();
        for (name, metadata) in result {
            let schema: FeatureSchema = serde_json::from_value(metadata)?;
            features.insert(name, schema);
        }
        
        Ok(features)
    }
}

/// Version comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct VersionComparison {
    pub version1: u32,
    pub version2: u32,
    pub added_features: Vec<String>,
    pub removed_features: Vec<String>,
    pub modified_features: Vec<String>,
}

use std::sync::atomic::{AtomicU32, Ordering};

// Cameron: "Perfect versioning enables perfect backtesting"