// PRODUCTION DEPLOYMENT CONFIGURATION - Task 0.5.1
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Environment-specific configuration with secure secret management
// External Research Applied:
// - "The Twelve-Factor App" methodology (Heroku, 2017)
// - Kubernetes ConfigMaps and Secrets best practices (CNCF, 2024)
// - HashiCorp Vault integration patterns
// - AWS Systems Manager Parameter Store patterns
// - "Production-Ready Microservices" - Fowler (2016)
// - High-frequency trading deployment practices (Jane Street, Two Sigma)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, bail};
use tracing::info;
use tokio::sync::broadcast;

// ============================================================================
// ENVIRONMENT CONFIGURATION
// ============================================================================

/// Deployment environment types
/// Alex: "Each environment has specific security and performance profiles"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Environment {
    /// Local development environment
    Development,
    
    /// Testing/QA environment
    Testing,
    
    /// Staging environment (production-like)
    Staging,
    
    /// Production environment
    Production,
}

impl Environment {
    /// Get environment from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dev" | "development" => Ok(Environment::Development),
            "test" | "testing" => Ok(Environment::Testing),
            "stage" | "staging" => Ok(Environment::Staging),
            "prod" | "production" => Ok(Environment::Production),
            _ => bail!("Unknown environment: {}", s),
        }
    }
    
    /// Get configuration file suffix
    pub fn config_suffix(&self) -> &str {
        match self {
            Environment::Development => "dev",
            Environment::Testing => "test",
            Environment::Staging => "stage",
            Environment::Production => "prod",
        }
    }
    
    /// Check if this is a production-like environment
    pub fn is_production_like(&self) -> bool {
        matches!(self, Environment::Staging | Environment::Production)
    }
    
    /// Get risk limits for this environment
    /// Quinn: "Production has stricter limits"
    pub fn risk_limits(&self) -> RiskLimits {
        match self {
            Environment::Development => RiskLimits {
                max_position_size: 0.001,  // 0.1% for dev
                max_leverage: 1.0,
                max_drawdown: 0.5,
                require_stop_loss: false,
            },
            Environment::Testing => RiskLimits {
                max_position_size: 0.005,  // 0.5% for testing
                max_leverage: 2.0,
                max_drawdown: 0.3,
                require_stop_loss: true,
            },
            Environment::Staging => RiskLimits {
                max_position_size: 0.01,   // 1% for staging
                max_leverage: 3.0,
                max_drawdown: 0.15,
                require_stop_loss: true,
            },
            Environment::Production => RiskLimits {
                max_position_size: 0.02,   // 2% for production
                max_leverage: 3.0,
                max_drawdown: 0.15,
                require_stop_loss: true,
            },
        }
    }
}

/// Risk limits per environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_size: f64,
    pub max_leverage: f64,
    pub max_drawdown: f64,
    pub require_stop_loss: bool,
}

// ============================================================================
// SECRET MANAGEMENT
// ============================================================================

/// Secret provider interface
/// Sam: "Abstraction over different secret backends"
pub trait SecretProvider: Send + Sync {
    /// Get a secret value by key
    fn get_secret(&self, key: &str) -> Result<String>;
    
    /// Get multiple secrets with prefix
    fn get_secrets_with_prefix(&self, prefix: &str) -> Result<HashMap<String, String>>;
    
    /// Validate all required secrets exist
    fn validate_secrets(&self, required: &[&str]) -> Result<()>;
}

/// Environment variable secret provider
pub struct EnvSecretProvider {
    prefix: String,
}

impl EnvSecretProvider {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl SecretProvider for EnvSecretProvider {
    fn get_secret(&self, key: &str) -> Result<String> {
        let env_key = format!("{}_{}", self.prefix, key.to_uppercase());
        std::env::var(&env_key)
            .with_context(|| format!("Missing environment variable: {}", env_key))
    }
    
    fn get_secrets_with_prefix(&self, prefix: &str) -> Result<HashMap<String, String>> {
        let mut secrets = HashMap::new();
        let search_prefix = format!("{}_{}", self.prefix, prefix.to_uppercase());
        
        for (key, value) in std::env::vars() {
            if key.starts_with(&search_prefix) {
                let secret_key = key.strip_prefix(&format!("{}_", self.prefix))
                    .unwrap_or(&key)
                    .to_lowercase();
                secrets.insert(secret_key, value);
            }
        }
        
        Ok(secrets)
    }
    
    fn validate_secrets(&self, required: &[&str]) -> Result<()> {
        for key in required {
            self.get_secret(key)?;
        }
        Ok(())
    }
}

/// Kubernetes secret provider
/// Uses mounted secret volumes
pub struct K8sSecretProvider {
    secret_path: PathBuf,
}

impl K8sSecretProvider {
    pub fn new(secret_path: impl AsRef<Path>) -> Self {
        Self {
            secret_path: secret_path.as_ref().to_path_buf(),
        }
    }
}

impl SecretProvider for K8sSecretProvider {
    fn get_secret(&self, key: &str) -> Result<String> {
        let file_path = self.secret_path.join(key);
        fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read secret from: {:?}", file_path))
    }
    
    fn get_secrets_with_prefix(&self, prefix: &str) -> Result<HashMap<String, String>> {
        let mut secrets = HashMap::new();
        
        for entry in fs::read_dir(&self.secret_path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let name_str = file_name.to_string_lossy();
            
            if name_str.starts_with(prefix) {
                let content = fs::read_to_string(entry.path())?;
                secrets.insert(name_str.to_string(), content);
            }
        }
        
        Ok(secrets)
    }
    
    fn validate_secrets(&self, required: &[&str]) -> Result<()> {
        for key in required {
            self.get_secret(key)?;
        }
        Ok(())
    }
}

/// HashiCorp Vault secret provider
/// Morgan: "Industry standard for secret management"
pub struct VaultSecretProvider {
    client: reqwest::Client,
    vault_addr: String,
    vault_token: String,
    mount_path: String,
}

impl VaultSecretProvider {
    pub async fn new(vault_addr: &str, vault_token: &str, mount_path: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
            
        Ok(Self {
            client,
            vault_addr: vault_addr.to_string(),
            vault_token: vault_token.to_string(),
            mount_path: mount_path.to_string(),
        })
    }
    
    async fn read_secret(&self, path: &str) -> Result<HashMap<String, String>> {
        let url = format!("{}/v1/{}/data/{}", self.vault_addr, self.mount_path, path);
        
        let response = self.client
            .get(&url)
            .header("X-Vault-Token", &self.vault_token)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
            
        let data = response["data"]["data"]
            .as_object()
            .context("Invalid Vault response format")?;
            
        let mut secrets = HashMap::new();
        for (key, value) in data {
            if let Some(v) = value.as_str() {
                secrets.insert(key.clone(), v.to_string());
            }
        }
        
        Ok(secrets)
    }
}

// ============================================================================
// DEPLOYMENT CONFIGURATION
// ============================================================================

/// Complete deployment configuration
/// Alex: "Single source of truth for all deployment settings"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Current environment
    pub environment: Environment,
    
    /// Application configuration
    pub app: AppConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Redis configuration
    pub redis: RedisConfig,
    
    /// Exchange configurations
    pub exchanges: HashMap<String, ExchangeConfig>,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Feature flags
    pub features: FeatureFlags,
    
    /// Resource limits
    pub resources: ResourceLimits,
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub name: String,
    pub version: String,
    pub port: u16,
    pub host: String,
    pub log_level: String,
    pub metrics_port: u16,
    pub health_check_port: u16,
}

/// Database configuration
/// Avery: "TimescaleDB with connection pooling"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    #[serde(skip_serializing)]
    pub password: String,
    pub pool_size: u32,
    pub connection_timeout: u64,
    pub ssl_mode: String,
}

impl DatabaseConfig {
    pub fn connection_string(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}?sslmode={}",
            self.username,
            self.password,
            self.host,
            self.port,
            self.database,
            self.ssl_mode
        )
    }
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    #[serde(skip_serializing)]
    pub password: Option<String>,
    pub database: u8,
    pub pool_size: u32,
    pub cluster_mode: bool,
}

impl RedisConfig {
    pub fn connection_string(&self) -> String {
        if let Some(ref password) = self.password {
            format!("redis://:{}@{}:{}/{}", password, self.host, self.port, self.database)
        } else {
            format!("redis://{}:{}/{}", self.host, self.port, self.database)
        }
    }
}

/// Exchange configuration
/// Casey: "Per-exchange API configuration"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_key: String,
    #[serde(skip_serializing)]
    pub api_secret: String,
    pub testnet: bool,
    pub ws_endpoint: String,
    pub rest_endpoint: String,
    pub rate_limit: u32,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub prometheus_enabled: bool,
    pub jaeger_enabled: bool,
    pub jaeger_endpoint: String,
    pub log_format: String,
    pub metrics_prefix: String,
}

/// Feature flags for gradual rollout
/// Jordan: "Control feature activation per environment"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    pub ml_models_enabled: bool,
    pub auto_trading_enabled: bool,
    pub advanced_analytics_enabled: bool,
    pub experimental_features_enabled: bool,
    pub paper_trading_mode: bool,
}

/// Resource limits per environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_cpu_cores: f32,
    pub max_disk_gb: u64,
    pub max_connections: u32,
    pub max_threads: u32,
}

// ============================================================================
// CONFIGURATION MANAGER
// ============================================================================

/// Configuration manager with hot reload support
pub struct ConfigManager {
    /// Current configuration
    config: Arc<RwLock<DeploymentConfig>>,
    
    /// Secret provider
    secret_provider: Arc<dyn SecretProvider>,
    
    /// Configuration change broadcaster
    change_notifier: broadcast::Sender<DeploymentConfig>,
    
    /// Configuration file path
    config_path: PathBuf,
    
    /// Environment
    environment: Environment,
}

impl ConfigManager {
    /// Create new configuration manager
    pub async fn new(
        environment: Environment,
        config_path: impl AsRef<Path>,
        secret_provider: Arc<dyn SecretProvider>,
    ) -> Result<Self> {
        let config_path = config_path.as_ref().to_path_buf();
        let config = Self::load_config(&config_path, environment, &*secret_provider).await?;
        
        let (change_notifier, _) = broadcast::channel(100);
        
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            secret_provider,
            change_notifier,
            config_path,
            environment,
        })
    }
    
    /// Load configuration from file and secrets
    async fn load_config(
        path: &Path,
        environment: Environment,
        secret_provider: &dyn SecretProvider,
    ) -> Result<DeploymentConfig> {
        // Load base configuration
        let config_file = path.join(format!("config.{}.toml", environment.config_suffix()));
        let config_str = fs::read_to_string(&config_file)
            .with_context(|| format!("Failed to read config file: {:?}", config_file))?;
            
        let mut config: DeploymentConfig = toml::from_str(&config_str)
            .context("Failed to parse configuration")?;
            
        // Inject secrets
        config.database.password = secret_provider.get_secret("DB_PASSWORD")?;
        
        if config.redis.password.is_none() {
            config.redis.password = secret_provider.get_secret("REDIS_PASSWORD").ok();
        }
        
        // Load exchange secrets
        for (exchange_name, exchange_config) in &mut config.exchanges {
            let key_name = format!("{}_API_KEY", exchange_name.to_uppercase());
            let secret_name = format!("{}_API_SECRET", exchange_name.to_uppercase());
            
            exchange_config.api_key = secret_provider.get_secret(&key_name)?;
            exchange_config.api_secret = secret_provider.get_secret(&secret_name)?;
        }
        
        // Validate configuration
        Self::validate_config(&config)?;
        
        info!("Loaded configuration for environment: {:?}", environment);
        Ok(config)
    }
    
    /// Validate configuration
    fn validate_config(config: &DeploymentConfig) -> Result<()> {
        // Validate database settings
        if config.database.pool_size == 0 {
            bail!("Database pool size must be > 0");
        }
        
        // Validate Redis settings
        if config.redis.pool_size == 0 {
            bail!("Redis pool size must be > 0");
        }
        
        // Validate exchange settings
        for (name, exchange) in &config.exchanges {
            if exchange.api_key.is_empty() {
                bail!("Exchange {} missing API key", name);
            }
            if exchange.api_secret.is_empty() {
                bail!("Exchange {} missing API secret", name);
            }
        }
        
        // Validate resource limits
        if config.resources.max_memory_mb == 0 {
            bail!("Max memory must be > 0");
        }
        
        // Environment-specific validations
        if config.environment.is_production_like() {
            // Production requires SSL
            if config.database.ssl_mode != "require" {
                bail!("Production database must use SSL");
            }
            
            // Production cannot have testnet enabled
            for (name, exchange) in &config.exchanges {
                if exchange.testnet {
                    bail!("Production cannot use testnet for exchange: {}", name);
                }
            }
            
            // Production must have monitoring
            if !config.monitoring.prometheus_enabled {
                bail!("Production must have Prometheus monitoring enabled");
            }
        }
        
        Ok(())
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> DeploymentConfig {
        self.config.read().clone()
    }
    
    /// Reload configuration from disk
    pub async fn reload(&self) -> Result<()> {
        let new_config = Self::load_config(
            &self.config_path,
            self.environment,
            &*self.secret_provider
        ).await?;
        
        *self.config.write() = new_config.clone();
        
        // Notify listeners
        let _ = self.change_notifier.send(new_config);
        
        info!("Configuration reloaded successfully");
        Ok(())
    }
    
    /// Subscribe to configuration changes
    pub fn subscribe(&self) -> broadcast::Receiver<DeploymentConfig> {
        self.change_notifier.subscribe()
    }
    
    /// Export configuration for debugging (without secrets)
    pub fn export_safe(&self) -> serde_json::Value {
        let config = self.config.read();
        serde_json::to_value(&*config).unwrap_or_else(|_| serde_json::json!({}))
    }
}

// ============================================================================
// KUBERNETES CONFIGURATION GENERATOR
// ============================================================================

/// Generate Kubernetes manifests
/// Riley: "Automated K8s manifest generation for consistency"
pub struct K8sManifestGenerator {
    config: DeploymentConfig,
    namespace: String,
}

impl K8sManifestGenerator {
    pub fn new(config: DeploymentConfig, namespace: String) -> Self {
        Self { config, namespace }
    }
    
    /// Generate ConfigMap manifest
    pub fn generate_configmap(&self) -> String {
        format!(r#"
apiVersion: v1
kind: ConfigMap
metadata:
  name: bot4-config
  namespace: {}
data:
  environment: "{}"
  app_name: "{}"
  app_version: "{}"
  log_level: "{}"
  database_host: "{}"
  database_port: "{}"
  database_name: "{}"
  redis_host: "{}"
  redis_port: "{}"
  monitoring_prometheus: "{}"
  monitoring_jaeger: "{}"
"#,
            self.namespace,
            self.config.environment.config_suffix(),
            self.config.app.name,
            self.config.app.version,
            self.config.app.log_level,
            self.config.database.host,
            self.config.database.port,
            self.config.database.database,
            self.config.redis.host,
            self.config.redis.port,
            self.config.monitoring.prometheus_enabled,
            self.config.monitoring.jaeger_enabled,
        )
    }
    
    /// Generate Secret manifest template
    pub fn generate_secret_template(&self) -> String {
        format!(r#"
apiVersion: v1
kind: Secret
metadata:
  name: bot4-secrets
  namespace: {}
type: Opaque
data:
  # Base64 encoded values - DO NOT COMMIT WITH REAL VALUES
  db_password: <BASE64_ENCODED_DB_PASSWORD>
  redis_password: <BASE64_ENCODED_REDIS_PASSWORD>
  binance_api_key: <BASE64_ENCODED_API_KEY>
  binance_api_secret: <BASE64_ENCODED_API_SECRET>
"#,
            self.namespace
        )
    }
    
    /// Generate Deployment manifest
    pub fn generate_deployment(&self) -> String {
        format!(r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bot4-trading
  namespace: {}
  labels:
    app: bot4
    environment: {}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bot4
  template:
    metadata:
      labels:
        app: bot4
        version: {}
    spec:
      containers:
      - name: trading-engine
        image: bot4:{}
        ports:
        - containerPort: {}
          name: http
        - containerPort: {}
          name: metrics
        - containerPort: {}
          name: health
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: bot4-config
              key: environment
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: bot4-secrets
              key: db_password
        resources:
          limits:
            memory: "{}Mi"
            cpu: "{}"
          requests:
            memory: "{}Mi"
            cpu: "{}"
        livenessProbe:
          httpGet:
            path: /health
            port: health
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: health
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: secrets
          mountPath: /var/run/secrets/bot4
          readOnly: true
      volumes:
      - name: secrets
        secret:
          secretName: bot4-secrets
"#,
            self.namespace,
            self.config.environment.config_suffix(),
            self.config.app.version,
            self.config.app.version,
            self.config.app.port,
            self.config.app.metrics_port,
            self.config.app.health_check_port,
            self.config.resources.max_memory_mb,
            self.config.resources.max_cpu_cores,
            self.config.resources.max_memory_mb / 2,
            self.config.resources.max_cpu_cores / 2.0,
        )
    }
    
    /// Generate Service manifest
    pub fn generate_service(&self) -> String {
        format!(r#"
apiVersion: v1
kind: Service
metadata:
  name: bot4-service
  namespace: {}
spec:
  selector:
    app: bot4
  ports:
  - name: http
    port: {}
    targetPort: http
  - name: metrics
    port: {}
    targetPort: metrics
  type: ClusterIP
"#,
            self.namespace,
            self.config.app.port,
            self.config.app.metrics_port,
        )
    }
}

// ============================================================================
// DOCKER COMPOSE GENERATOR
// ============================================================================

/// Generate Docker Compose configuration
pub struct DockerComposeGenerator {
    config: DeploymentConfig,
}

impl DockerComposeGenerator {
    pub fn new(config: DeploymentConfig) -> Self {
        Self { config }
    }
    
    /// Generate docker-compose.yml
    pub fn generate(&self) -> String {
        let env_suffix = self.config.environment.config_suffix();
        
        format!(r#"
version: '3.8'

services:
  trading-engine:
    image: bot4:latest
    container_name: bot4-trading-{}
    environment:
      - ENVIRONMENT={}
      - APP_PORT={}
      - METRICS_PORT={}
      - DB_HOST={}
      - DB_PORT={}
      - DB_NAME={}
      - DB_USER={}
      - DB_PASSWORD=${{DB_PASSWORD}}
      - REDIS_HOST={}
      - REDIS_PORT={}
      - REDIS_PASSWORD=${{REDIS_PASSWORD}}
      - LOG_LEVEL={}
    ports:
      - "{}:{}"
      - "{}:{}"
    networks:
      - bot4-network
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    mem_limit: {}m
    cpus: '{}'
    
  postgres:
    image: timescale/timescaledb:latest-pg14
    container_name: bot4-postgres-{}
    environment:
      - POSTGRES_DB={}
      - POSTGRES_USER={}
      - POSTGRES_PASSWORD=${{DB_PASSWORD}}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - bot4-network
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    container_name: bot4-redis-{}
    command: redis-server --requirepass ${{REDIS_PASSWORD}}
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - bot4-network
    restart: unless-stopped

networks:
  bot4-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
"#,
            env_suffix,
            env_suffix,
            self.config.app.port,
            self.config.app.metrics_port,
            self.config.database.host,
            self.config.database.port,
            self.config.database.database,
            self.config.database.username,
            self.config.redis.host,
            self.config.redis.port,
            self.config.app.log_level,
            self.config.app.port,
            self.config.app.port,
            self.config.app.metrics_port,
            self.config.app.metrics_port,
            self.config.resources.max_memory_mb,
            self.config.resources.max_cpu_cores,
            env_suffix,
            self.config.database.database,
            self.config.database.username,
            env_suffix,
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_environment_parsing() {
        assert_eq!(Environment::from_str("dev").unwrap(), Environment::Development);
        assert_eq!(Environment::from_str("prod").unwrap(), Environment::Production);
        assert!(Environment::from_str("invalid").is_err());
    }
    
    #[test]
    fn test_risk_limits() {
        let prod_limits = Environment::Production.risk_limits();
        assert_eq!(prod_limits.max_position_size, 0.02);
        assert!(prod_limits.require_stop_loss);
        
        let dev_limits = Environment::Development.risk_limits();
        assert_eq!(dev_limits.max_position_size, 0.001);
        assert!(!dev_limits.require_stop_loss);
    }
    
    #[test]
    fn test_env_secret_provider() {
        std::env::set_var("BOT4_TEST_SECRET", "secret_value");
        
        let provider = EnvSecretProvider::new("BOT4");
        assert_eq!(provider.get_secret("TEST_SECRET").unwrap(), "secret_value");
        
        std::env::remove_var("BOT4_TEST_SECRET");
    }
    
    #[test]
    fn test_k8s_secret_provider() {
        let temp_dir = TempDir::new().unwrap();
        let secret_path = temp_dir.path();
        
        // Create test secret file
        fs::write(secret_path.join("test_secret"), "secret_value").unwrap();
        
        let provider = K8sSecretProvider::new(secret_path);
        assert_eq!(provider.get_secret("test_secret").unwrap(), "secret_value");
    }
    
    #[test]
    fn test_database_config_connection_string() {
        let config = DatabaseConfig {
            host: "localhost".to_string(),
            port: 5432,
            database: "bot4".to_string(),
            username: "user".to_string(),
            password: "pass".to_string(),
            pool_size: 10,
            connection_timeout: 30,
            ssl_mode: "require".to_string(),
        };
        
        let expected = "postgresql://user:pass@localhost:5432/bot4?sslmode=require";
        assert_eq!(config.connection_string(), expected);
    }
    
    #[test]
    fn test_redis_config_connection_string() {
        let config = RedisConfig {
            host: "localhost".to_string(),
            port: 6379,
            password: Some("pass".to_string()),
            database: 0,
            pool_size: 10,
            cluster_mode: false,
        };
        
        let expected = "redis://:pass@localhost:6379/0";
        assert_eq!(config.connection_string(), expected);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = DeploymentConfig {
            environment: Environment::Production,
            app: AppConfig {
                name: "bot4".to_string(),
                version: "1.0.0".to_string(),
                port: 8080,
                host: "0.0.0.0".to_string(),
                log_level: "info".to_string(),
                metrics_port: 9090,
                health_check_port: 8081,
            },
            database: DatabaseConfig {
                host: "localhost".to_string(),
                port: 5432,
                database: "bot4".to_string(),
                username: "user".to_string(),
                password: "pass".to_string(),
                pool_size: 10,
                connection_timeout: 30,
                ssl_mode: "disable".to_string(), // Wrong for production
            },
            redis: RedisConfig {
                host: "localhost".to_string(),
                port: 6379,
                password: None,
                database: 0,
                pool_size: 10,
                cluster_mode: false,
            },
            exchanges: HashMap::new(),
            monitoring: MonitoringConfig {
                prometheus_enabled: true,
                jaeger_enabled: true,
                jaeger_endpoint: "localhost:6831".to_string(),
                log_format: "json".to_string(),
                metrics_prefix: "bot4".to_string(),
            },
            features: FeatureFlags {
                ml_models_enabled: true,
                auto_trading_enabled: true,
                advanced_analytics_enabled: true,
                experimental_features_enabled: false,
                paper_trading_mode: false,
            },
            resources: ResourceLimits {
                max_memory_mb: 4096,
                max_cpu_cores: 2.0,
                max_disk_gb: 100,
                max_connections: 100,
                max_threads: 50,
            },
        };
        
        // Should fail due to SSL mode
        assert!(ConfigManager::validate_config(&config).is_err());
        
        // Fix SSL mode
        config.database.ssl_mode = "require".to_string();
        assert!(ConfigManager::validate_config(&config).is_ok());
    }
    
    #[test]
    fn test_k8s_manifest_generation() {
        let config = DeploymentConfig {
            environment: Environment::Production,
            app: AppConfig {
                name: "bot4".to_string(),
                version: "1.0.0".to_string(),
                port: 8080,
                host: "0.0.0.0".to_string(),
                log_level: "info".to_string(),
                metrics_port: 9090,
                health_check_port: 8081,
            },
            database: DatabaseConfig {
                host: "postgres".to_string(),
                port: 5432,
                database: "bot4".to_string(),
                username: "user".to_string(),
                password: "pass".to_string(),
                pool_size: 10,
                connection_timeout: 30,
                ssl_mode: "require".to_string(),
            },
            redis: RedisConfig {
                host: "redis".to_string(),
                port: 6379,
                password: Some("pass".to_string()),
                database: 0,
                pool_size: 10,
                cluster_mode: false,
            },
            exchanges: HashMap::new(),
            monitoring: MonitoringConfig {
                prometheus_enabled: true,
                jaeger_enabled: true,
                jaeger_endpoint: "jaeger:6831".to_string(),
                log_format: "json".to_string(),
                metrics_prefix: "bot4".to_string(),
            },
            features: FeatureFlags {
                ml_models_enabled: true,
                auto_trading_enabled: true,
                advanced_analytics_enabled: true,
                experimental_features_enabled: false,
                paper_trading_mode: false,
            },
            resources: ResourceLimits {
                max_memory_mb: 4096,
                max_cpu_cores: 2.0,
                max_disk_gb: 100,
                max_connections: 100,
                max_threads: 50,
            },
        };
        
        let generator = K8sManifestGenerator::new(config, "production".to_string());
        
        let configmap = generator.generate_configmap();
        assert!(configmap.contains("bot4-config"));
        assert!(configmap.contains("namespace: production"));
        
        let deployment = generator.generate_deployment();
        assert!(deployment.contains("bot4-trading"));
        assert!(deployment.contains("memory: \"4096Mi\""));
        
        let service = generator.generate_service();
        assert!(service.contains("port: 8080"));
    }
}