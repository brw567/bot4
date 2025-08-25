use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use tokio::sync::{RwLock, Mutex, Semaphore};
use tokio::time::{interval, Duration, Instant};
use dashmap::DashMap;
use reqwest::{Client, StatusCode};
use apache_avro::{Schema as AvroSchema, types::Value as AvroValue};
use schema_registry_converter::{
    async_impl::schema_registry::{SrSettings, get_schema_by_id},
    schema_registry_common::{SubjectNameStrategy, RegisteredSchema},
};
use tracing::{info, warn, error, debug, trace};
use bytes::Bytes;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Schema Registry configuration
#[derive(Debug, Clone)]
pub struct SchemaRegistryConfig {
    /// Schema Registry URL (e.g., http://localhost:8081)
    pub url: String,
    
    /// Authentication credentials (optional)
    pub auth: Option<SchemaAuth>,
    
    /// Connection timeout
    pub connect_timeout: Duration,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Maximum retries for failed requests
    pub max_retries: u32,
    
    /// Retry backoff base duration
    pub retry_backoff: Duration,
    
    /// Cache size for schemas
    pub cache_size: usize,
    
    /// Cache TTL
    pub cache_ttl: Duration,
    
    /// Enable schema validation
    pub enable_validation: bool,
    
    /// Default compatibility level
    pub compatibility: CompatibilityLevel,
    
    /// Subject naming strategy
    pub subject_strategy: SubjectStrategy,
    
    /// Enable auto-registration of schemas
    pub auto_register: bool,
    
    /// Enable schema evolution tracking
    pub track_evolution: bool,
    
    /// Maximum schema versions to keep
    pub max_versions: u32,
}

impl Default for SchemaRegistryConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8081".to_string(),
            auth: None,
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(10),
            max_retries: 3,
            retry_backoff: Duration::from_millis(100),
            cache_size: 1000,
            cache_ttl: Duration::from_secs(300),  // 5 minutes
            enable_validation: true,
            compatibility: CompatibilityLevel::Backward,
            subject_strategy: SubjectStrategy::TopicName,
            auto_register: false,  // Disabled in production
            track_evolution: true,
            max_versions: 100,
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub enum SchemaAuth {
    Basic { username: String, password: String },
    Bearer { token: String },
    ApiKey { key: String, secret: String },
}

/// Schema compatibility levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityLevel {
    None,
    Backward,
    BackwardTransitive,
    Forward,
    ForwardTransitive,
    Full,
    FullTransitive,
}

impl CompatibilityLevel {
    pub fn to_string(&self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Backward => "BACKWARD",
            Self::BackwardTransitive => "BACKWARD_TRANSITIVE",
            Self::Forward => "FORWARD",
            Self::ForwardTransitive => "FORWARD_TRANSITIVE",
            Self::Full => "FULL",
            Self::FullTransitive => "FULL_TRANSITIVE",
        }
    }
}

/// Subject naming strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectStrategy {
    TopicName,
    RecordName,
    TopicRecordName,
}

/// Schema types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaType {
    Avro,
    Json,
    Protobuf,
}

/// Registered schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    pub id: i32,
    pub version: i32,
    pub subject: String,
    pub schema: String,
    pub schema_type: SchemaType,
    pub references: Vec<SchemaReference>,
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
}

/// Schema reference for nested schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaReference {
    pub name: String,
    pub subject: String,
    pub version: i32,
}

/// Schema evolution tracking
#[derive(Debug, Clone)]
pub struct SchemaEvolution {
    pub from_version: i32,
    pub to_version: i32,
    pub changes: Vec<SchemaChange>,
    pub compatibility: CompatibilityLevel,
    pub migration_rules: Vec<MigrationRule>,
}

/// Types of schema changes
#[derive(Debug, Clone)]
pub enum SchemaChange {
    FieldAdded { name: String, default: Option<Value> },
    FieldRemoved { name: String },
    FieldRenamed { old: String, new: String },
    TypeChanged { field: String, old_type: String, new_type: String },
    DefaultChanged { field: String, old: Option<Value>, new: Option<Value> },
}

/// Migration rules for complex evolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRule {
    pub rule_type: String,
    pub field: String,
    pub transformation: String,
}

/// Schema cache entry
struct CacheEntry {
    schema: Arc<SchemaInfo>,
    parsed: Option<Arc<AvroSchema>>,
    created_at: Instant,
    access_count: AtomicU64,
}

/// Metrics for monitoring
pub struct RegistryMetrics {
    pub schemas_registered: AtomicU64,
    pub schemas_fetched: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub validation_success: AtomicU64,
    pub validation_failures: AtomicU64,
    pub compatibility_checks: AtomicU64,
    pub api_latency_us: AtomicU64,
}

impl RegistryMetrics {
    fn new() -> Self {
        Self {
            schemas_registered: AtomicU64::new(0),
            schemas_fetched: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            validation_success: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
            compatibility_checks: AtomicU64::new(0),
            api_latency_us: AtomicU64::new(0),
        }
    }
}

/// Data contract for schema governance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataContract {
    pub name: String,
    pub version: String,
    pub owner: String,
    pub description: String,
    pub schemas: Vec<ContractSchema>,
    pub sla: ServiceLevelAgreement,
    pub quality_rules: Vec<QualityRule>,
    pub metadata: HashMap<String, String>,
}

/// Schema within a data contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSchema {
    pub subject: String,
    pub version: i32,
    pub format: SchemaType,
    pub required: bool,
}

/// Service Level Agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLevelAgreement {
    pub availability: f64,  // e.g., 99.99
    pub latency_p99_ms: u64,
    pub throughput_eps: u64,  // events per second
    pub freshness_ms: u64,  // data freshness requirement
}

/// Data quality rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRule {
    pub name: String,
    pub field: String,
    pub rule_type: String,  // e.g., "not_null", "range", "regex"
    pub parameters: HashMap<String, Value>,
}

/// Main Schema Registry implementation
pub struct SchemaRegistry {
    config: Arc<SchemaRegistryConfig>,
    client: Arc<Client>,
    cache: Arc<RwLock<LruCache<String, Arc<CacheEntry>>>>,
    id_cache: Arc<DashMap<i32, Arc<SchemaInfo>>>,
    evolution_history: Arc<RwLock<HashMap<String, Vec<SchemaEvolution>>>>,
    contracts: Arc<RwLock<HashMap<String, DataContract>>>,
    metrics: Arc<RegistryMetrics>,
    request_semaphore: Arc<Semaphore>,
    shutdown: Arc<AtomicBool>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl SchemaRegistry {
    /// Create a new Schema Registry client
    pub async fn new(config: SchemaRegistryConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .build()
            .context("Failed to create HTTP client")?;
        
        let cache_size = NonZeroUsize::new(config.cache_size)
            .ok_or_else(|| anyhow!("Cache size must be greater than 0"))?;
        
        let registry = Arc::new(Self {
            config: Arc::new(config.clone()),
            client: Arc::new(client),
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            id_cache: Arc::new(DashMap::new()),
            evolution_history: Arc::new(RwLock::new(HashMap::new())),
            contracts: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RegistryMetrics::new()),
            request_semaphore: Arc::new(Semaphore::new(10)),  // Max 10 concurrent requests
            shutdown: Arc::new(AtomicBool::new(false)),
            cleanup_handle: None,
        });
        
        // Test connectivity
        registry.test_connectivity().await?;
        
        // Start cleanup task
        let cleanup_handle = {
            let reg = registry.clone();
            tokio::spawn(async move {
                reg.cleanup_task().await;
            })
        };
        
        // Return with handle
        let mut_registry = Arc::try_unwrap(registry)
            .unwrap_or_else(|arc| (*arc).clone());
        
        Ok(Self {
            cleanup_handle: Some(cleanup_handle),
            ..mut_registry
        })
    }
    
    /// Test connectivity to Schema Registry
    async fn test_connectivity(&self) -> Result<()> {
        let url = format!("{}/subjects", self.config.url);
        let response = self.make_request(reqwest::Method::GET, &url, None).await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Schema Registry connectivity test failed: {}", response.status()));
        }
        
        info!("Successfully connected to Schema Registry at {}", self.config.url);
        Ok(())
    }
    
    /// Register a new schema
    pub async fn register_schema(
        &self,
        subject: &str,
        schema: &str,
        schema_type: SchemaType,
        references: Vec<SchemaReference>,
    ) -> Result<SchemaInfo> {
        let start = Instant::now();
        
        // Validate schema if enabled
        if self.config.enable_validation {
            self.validate_schema(schema, schema_type)?;
        }
        
        // Check compatibility if not first version
        if let Ok(latest) = self.get_latest_schema(subject).await {
            self.check_compatibility(subject, schema, &latest.schema).await?;
        }
        
        // Prepare request body
        let body = json!({
            "schema": schema,
            "schemaType": match schema_type {
                SchemaType::Avro => "AVRO",
                SchemaType::Json => "JSON",
                SchemaType::Protobuf => "PROTOBUF",
            },
            "references": references,
        });
        
        // Register schema
        let url = format!("{}/subjects/{}/versions", self.config.url, subject);
        let response = self.make_request(
            reqwest::Method::POST,
            &url,
            Some(body),
        ).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to register schema: {}", error_text));
        }
        
        let result: Value = response.json().await?;
        let id = result["id"].as_i64().unwrap_or(0) as i32;
        
        // Fetch full schema info
        let schema_info = self.get_schema_by_id(id).await?;
        
        // Cache the schema
        self.cache_schema(subject.to_string(), schema_info.clone()).await;
        
        // Track evolution if enabled
        if self.config.track_evolution {
            self.track_schema_evolution(subject, &schema_info).await?;
        }
        
        // Update metrics
        self.metrics.schemas_registered.fetch_add(1, Ordering::Relaxed);
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.api_latency_us.store(latency, Ordering::Relaxed);
        
        info!("Registered schema for subject '{}' with ID {}", subject, id);
        Ok(schema_info)
    }
    
    /// Get latest schema for a subject
    pub async fn get_latest_schema(&self, subject: &str) -> Result<SchemaInfo> {
        // Check cache first
        let cache_key = format!("{}:latest", subject);
        if let Some(cached) = self.get_from_cache(&cache_key).await {
            return Ok((*cached.schema).clone());
        }
        
        let start = Instant::now();
        
        // Fetch from registry
        let url = format!("{}/subjects/{}/versions/latest", self.config.url, subject);
        let response = self.make_request(reqwest::Method::GET, &url, None).await?;
        
        if response.status() == StatusCode::NOT_FOUND {
            return Err(anyhow!("Subject '{}' not found", subject));
        }
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to get schema: {}", error_text));
        }
        
        let result: Value = response.json().await?;
        let schema_info = self.parse_schema_response(subject, result)?;
        
        // Cache the schema
        self.cache_schema(cache_key, schema_info.clone()).await;
        
        // Update metrics
        self.metrics.schemas_fetched.fetch_add(1, Ordering::Relaxed);
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.api_latency_us.store(latency, Ordering::Relaxed);
        
        Ok(schema_info)
    }
    
    /// Get schema by ID
    pub async fn get_schema_by_id(&self, id: i32) -> Result<SchemaInfo> {
        // Check ID cache first
        if let Some(cached) = self.id_cache.get(&id) {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((*cached).clone());
        }
        
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        let start = Instant::now();
        
        // Fetch from registry
        let url = format!("{}/schemas/ids/{}", self.config.url, id);
        let response = self.make_request(reqwest::Method::GET, &url, None).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to get schema by ID {}: {}", id, error_text));
        }
        
        let result: Value = response.json().await?;
        let schema_info = SchemaInfo {
            id,
            version: 0,  // Version not available from ID endpoint
            subject: String::new(),  // Subject not available from ID endpoint
            schema: result["schema"].as_str().unwrap_or("").to_string(),
            schema_type: match result["schemaType"].as_str() {
                Some("JSON") => SchemaType::Json,
                Some("PROTOBUF") => SchemaType::Protobuf,
                _ => SchemaType::Avro,
            },
            references: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now().timestamp_millis(),
        };
        
        // Cache by ID
        self.id_cache.insert(id, Arc::new(schema_info.clone()));
        
        // Update metrics
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.api_latency_us.store(latency, Ordering::Relaxed);
        
        Ok(schema_info)
    }
    
    /// Get specific version of a schema
    pub async fn get_schema_version(&self, subject: &str, version: i32) -> Result<SchemaInfo> {
        let cache_key = format!("{}:{}", subject, version);
        if let Some(cached) = self.get_from_cache(&cache_key).await {
            return Ok((*cached.schema).clone());
        }
        
        let url = format!("{}/subjects/{}/versions/{}", self.config.url, subject, version);
        let response = self.make_request(reqwest::Method::GET, &url, None).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to get schema version: {}", error_text));
        }
        
        let result: Value = response.json().await?;
        let schema_info = self.parse_schema_response(subject, result)?;
        
        self.cache_schema(cache_key, schema_info.clone()).await;
        
        Ok(schema_info)
    }
    
    /// Delete a subject (all versions)
    pub async fn delete_subject(&self, subject: &str, permanent: bool) -> Result<Vec<i32>> {
        let mut url = format!("{}/subjects/{}", self.config.url, subject);
        if permanent {
            url.push_str("?permanent=true");
        }
        
        let response = self.make_request(reqwest::Method::DELETE, &url, None).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to delete subject: {}", error_text));
        }
        
        let versions: Vec<i32> = response.json().await?;
        
        // Clear from cache
        self.clear_subject_cache(subject).await;
        
        info!("Deleted subject '{}' with {} versions", subject, versions.len());
        Ok(versions)
    }
    
    /// Set compatibility level for a subject
    pub async fn set_compatibility(&self, subject: &str, level: CompatibilityLevel) -> Result<()> {
        let url = format!("{}/config/{}", self.config.url, subject);
        let body = json!({
            "compatibility": level.to_string(),
        });
        
        let response = self.make_request(reqwest::Method::PUT, &url, Some(body)).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to set compatibility: {}", error_text));
        }
        
        info!("Set compatibility level for '{}' to {:?}", subject, level);
        Ok(())
    }
    
    /// Check compatibility between schemas
    pub async fn check_compatibility(
        &self,
        subject: &str,
        new_schema: &str,
        old_schema: &str,
    ) -> Result<bool> {
        self.metrics.compatibility_checks.fetch_add(1, Ordering::Relaxed);
        
        let url = format!(
            "{}/compatibility/subjects/{}/versions/latest",
            self.config.url, subject
        );
        
        let body = json!({
            "schema": new_schema,
        });
        
        let response = self.make_request(reqwest::Method::POST, &url, Some(body)).await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            warn!("Compatibility check failed: {}", error_text);
            return Ok(false);
        }
        
        let result: Value = response.json().await?;
        let is_compatible = result["is_compatible"].as_bool().unwrap_or(false);
        
        if !is_compatible && self.config.track_evolution {
            // Analyze what changed
            let changes = self.analyze_schema_changes(old_schema, new_schema)?;
            warn!("Schema incompatible. Changes: {:?}", changes);
        }
        
        Ok(is_compatible)
    }
    
    /// Validate a schema
    fn validate_schema(&self, schema: &str, schema_type: SchemaType) -> Result<()> {
        match schema_type {
            SchemaType::Avro => {
                AvroSchema::parse_str(schema)
                    .map_err(|e| anyhow!("Invalid Avro schema: {}", e))?;
            },
            SchemaType::Json => {
                serde_json::from_str::<Value>(schema)
                    .map_err(|e| anyhow!("Invalid JSON schema: {}", e))?;
            },
            SchemaType::Protobuf => {
                // Basic validation for protobuf (would need protoc for full validation)
                if !schema.contains("syntax") || !schema.contains("message") {
                    return Err(anyhow!("Invalid Protobuf schema structure"));
                }
            },
        }
        
        self.metrics.validation_success.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Analyze changes between two schemas
    fn analyze_schema_changes(&self, old_schema: &str, new_schema: &str) -> Result<Vec<SchemaChange>> {
        let mut changes = Vec::new();
        
        // Parse schemas as JSON for comparison (simplified)
        let old: Value = serde_json::from_str(old_schema).unwrap_or(json!({}));
        let new: Value = serde_json::from_str(new_schema).unwrap_or(json!({}));
        
        // Compare fields (simplified - real implementation would be more complex)
        if let (Some(old_fields), Some(new_fields)) = (old["fields"].as_array(), new["fields"].as_array()) {
            let old_field_names: HashMap<_, _> = old_fields.iter()
                .filter_map(|f| f["name"].as_str().map(|n| (n.to_string(), f)))
                .collect();
            
            let new_field_names: HashMap<_, _> = new_fields.iter()
                .filter_map(|f| f["name"].as_str().map(|n| (n.to_string(), f)))
                .collect();
            
            // Find added fields
            for (name, field) in &new_field_names {
                if !old_field_names.contains_key(name) {
                    changes.push(SchemaChange::FieldAdded {
                        name: name.clone(),
                        default: field["default"].clone().into(),
                    });
                }
            }
            
            // Find removed fields
            for name in old_field_names.keys() {
                if !new_field_names.contains_key(name) {
                    changes.push(SchemaChange::FieldRemoved {
                        name: name.clone(),
                    });
                }
            }
            
            // Find type changes
            for (name, new_field) in &new_field_names {
                if let Some(old_field) = old_field_names.get(name) {
                    if old_field["type"] != new_field["type"] {
                        changes.push(SchemaChange::TypeChanged {
                            field: name.clone(),
                            old_type: old_field["type"].to_string(),
                            new_type: new_field["type"].to_string(),
                        });
                    }
                }
            }
        }
        
        Ok(changes)
    }
    
    /// Track schema evolution
    async fn track_schema_evolution(&self, subject: &str, schema_info: &SchemaInfo) -> Result<()> {
        let mut history = self.evolution_history.write().await;
        
        let evolution_entry = history.entry(subject.to_string()).or_insert_with(Vec::new);
        
        // Only track if there's a previous version
        if schema_info.version > 1 {
            let prev_version = self.get_schema_version(subject, schema_info.version - 1).await?;
            let changes = self.analyze_schema_changes(&prev_version.schema, &schema_info.schema)?;
            
            evolution_entry.push(SchemaEvolution {
                from_version: schema_info.version - 1,
                to_version: schema_info.version,
                changes,
                compatibility: self.config.compatibility,
                migration_rules: vec![],
            });
        }
        
        Ok(())
    }
    
    /// Create a data contract
    pub async fn create_contract(&self, contract: DataContract) -> Result<()> {
        // Validate all referenced schemas exist
        for schema in &contract.schemas {
            self.get_latest_schema(&schema.subject).await
                .map_err(|e| anyhow!("Schema '{}' in contract not found: {}", schema.subject, e))?;
        }
        
        let mut contracts = self.contracts.write().await;
        contracts.insert(contract.name.clone(), contract.clone());
        
        info!("Created data contract '{}'", contract.name);
        Ok(())
    }
    
    /// Get a data contract
    pub async fn get_contract(&self, name: &str) -> Result<DataContract> {
        let contracts = self.contracts.read().await;
        contracts.get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Contract '{}' not found", name))
    }
    
    /// Validate data against contract
    pub async fn validate_contract(&self, contract_name: &str, data: &[u8]) -> Result<()> {
        let contract = self.get_contract(contract_name).await?;
        
        // Validate against each schema in the contract
        for schema_ref in &contract.schemas {
            if schema_ref.required {
                let schema_info = self.get_latest_schema(&schema_ref.subject).await?;
                // Here you would validate the data against the schema
                // This is simplified - real implementation would deserialize and validate
            }
        }
        
        // Check quality rules
        for rule in &contract.quality_rules {
            // Apply quality rule validation
            // This would involve parsing the data and checking the rules
        }
        
        Ok(())
    }
    
    /// Cache a schema
    async fn cache_schema(&self, key: String, schema_info: SchemaInfo) {
        let parsed = if schema_info.schema_type == SchemaType::Avro {
            AvroSchema::parse_str(&schema_info.schema).ok().map(Arc::new)
        } else {
            None
        };
        
        let entry = Arc::new(CacheEntry {
            schema: Arc::new(schema_info.clone()),
            parsed,
            created_at: Instant::now(),
            access_count: AtomicU64::new(0),
        });
        
        let mut cache = self.cache.write().await;
        cache.put(key, entry);
        
        // Also cache by ID
        self.id_cache.insert(schema_info.id, Arc::new(schema_info));
    }
    
    /// Get from cache
    async fn get_from_cache(&self, key: &str) -> Option<Arc<CacheEntry>> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get(key) {
            // Check TTL
            if entry.created_at.elapsed() < self.config.cache_ttl {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.clone());
            } else {
                // Remove expired entry
                cache.pop(key);
            }
        }
        
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Clear cache for a subject
    async fn clear_subject_cache(&self, subject: &str) {
        let mut cache = self.cache.write().await;
        let keys_to_remove: Vec<String> = cache.iter()
            .filter_map(|(k, _)| {
                if k.starts_with(subject) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for key in keys_to_remove {
            cache.pop(&key);
        }
    }
    
    /// Make HTTP request with retries
    async fn make_request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response> {
        let _permit = self.request_semaphore.acquire().await?;
        
        let mut retries = 0;
        let mut backoff = self.config.retry_backoff;
        
        loop {
            let mut request = self.client.request(method.clone(), url);
            
            // Add authentication if configured
            if let Some(auth) = &self.config.auth {
                request = match auth {
                    SchemaAuth::Basic { username, password } => {
                        request.basic_auth(username, Some(password))
                    },
                    SchemaAuth::Bearer { token } => {
                        request.bearer_auth(token)
                    },
                    SchemaAuth::ApiKey { key, secret } => {
                        request
                            .header("X-API-Key", key)
                            .header("X-API-Secret", secret)
                    },
                };
            }
            
            // Add body if provided
            if let Some(ref body) = body {
                request = request
                    .header("Content-Type", "application/vnd.schemaregistry.v1+json")
                    .json(body);
            }
            
            match request.send().await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if retries >= self.config.max_retries {
                        return Err(anyhow!("Request failed after {} retries: {}", retries, e));
                    }
                    
                    warn!("Request failed (retry {}/{}): {}", retries + 1, self.config.max_retries, e);
                    tokio::time::sleep(backoff).await;
                    
                    retries += 1;
                    backoff *= 2;  // Exponential backoff
                }
            }
        }
    }
    
    /// Parse schema response from registry
    fn parse_schema_response(&self, subject: &str, value: Value) -> Result<SchemaInfo> {
        Ok(SchemaInfo {
            id: value["id"].as_i64().unwrap_or(0) as i32,
            version: value["version"].as_i64().unwrap_or(0) as i32,
            subject: subject.to_string(),
            schema: value["schema"].as_str().unwrap_or("").to_string(),
            schema_type: match value["schemaType"].as_str() {
                Some("JSON") => SchemaType::Json,
                Some("PROTOBUF") => SchemaType::Protobuf,
                _ => SchemaType::Avro,
            },
            references: self.parse_references(&value["references"]),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now().timestamp_millis(),
        })
    }
    
    /// Parse schema references
    fn parse_references(&self, value: &Value) -> Vec<SchemaReference> {
        if let Some(refs) = value.as_array() {
            refs.iter()
                .filter_map(|r| {
                    Some(SchemaReference {
                        name: r["name"].as_str()?.to_string(),
                        subject: r["subject"].as_str()?.to_string(),
                        version: r["version"].as_i64()? as i32,
                    })
                })
                .collect()
        } else {
            vec![]
        }
    }
    
    /// Background cleanup task
    async fn cleanup_task(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(60));
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            // Clean up expired cache entries
            let mut cache = self.cache.write().await;
            let now = Instant::now();
            let ttl = self.config.cache_ttl;
            
            // LRU cache handles eviction, but we can log stats
            let cache_size = cache.len();
            if cache_size > 0 {
                trace!("Schema cache size: {} entries", cache_size);
            }
        }
    }
    
    /// Get metrics
    pub fn metrics(&self) -> RegistryMetrics {
        RegistryMetrics {
            schemas_registered: AtomicU64::new(self.metrics.schemas_registered.load(Ordering::Relaxed)),
            schemas_fetched: AtomicU64::new(self.metrics.schemas_fetched.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.metrics.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.metrics.cache_misses.load(Ordering::Relaxed)),
            validation_success: AtomicU64::new(self.metrics.validation_success.load(Ordering::Relaxed)),
            validation_failures: AtomicU64::new(self.metrics.validation_failures.load(Ordering::Relaxed)),
            compatibility_checks: AtomicU64::new(self.metrics.compatibility_checks.load(Ordering::Relaxed)),
            api_latency_us: AtomicU64::new(self.metrics.api_latency_us.load(Ordering::Relaxed)),
        }
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Schema Registry client...");
        self.shutdown.store(true, Ordering::Relaxed);
        Ok(())
    }
}

// Re-export for convenience
pub use SchemaRegistry as Registry;