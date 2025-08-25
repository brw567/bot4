// MODE PERSISTENCE AND RECOVERY - Task 0.5.2
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Save and recover control mode state across restarts
// External Research Applied:
// - "Designing Data-Intensive Applications" - Kleppmann (2017)
// - PostgreSQL Write-Ahead Logging (WAL) patterns
// - Event Sourcing and CQRS patterns
// - "Crash-Only Software" - Candea & Fox (2003)
// - Financial trading system recovery patterns (NYSE, NASDAQ)

use std::sync::Arc;
use std::time::Duration;
use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error};
use tokio::sync::{RwLock, broadcast};
use sqlx::{PgPool, postgres::PgPoolOptions, Row};
use chrono::{DateTime, Utc};

use crate::software_control_modes::ControlMode;

// ============================================================================
// MODE TRANSITION RECORD
// ============================================================================

/// Record of mode transitions for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeTransition {
    pub from: ControlMode,
    pub to: ControlMode,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub authorized_by: String,
}

// ============================================================================
// MODE STATE PERSISTENCE
// ============================================================================

/// Persisted mode state with recovery information
/// Alex: "Must survive any crash scenario"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModeState {
    /// Current control mode
    pub mode: ControlMode,
    
    /// Timestamp of last mode change
    pub last_changed: DateTime<Utc>,
    
    /// Reason for current mode
    pub reason: String,
    
    /// Authorized by (user or system)
    pub authorized_by: String,
    
    /// Previous mode before current
    pub previous_mode: Option<ControlMode>,
    
    /// Recovery metadata
    pub recovery_info: RecoveryInfo,
    
    /// Mode-specific state data
    pub state_data: serde_json::Value,
    
    /// Checksum for integrity
    pub checksum: String,
}

/// Recovery information for crash scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryInfo {
    /// System version at time of save
    pub system_version: String,
    
    /// Was this a clean shutdown?
    pub clean_shutdown: bool,
    
    /// Number of crashes in last 24h
    pub crash_count_24h: u32,
    
    /// Last crash timestamp if any
    pub last_crash: Option<DateTime<Utc>>,
    
    /// Active positions at time of save
    pub active_positions: u32,
    
    /// Total exposure at time of save
    pub total_exposure: f64,
    
    /// Circuit breakers tripped
    pub breakers_tripped: Vec<String>,
}

/// Mode recovery policy
/// Quinn: "Defines how we recover from different crash scenarios"
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    /// Maximum crashes before forcing Emergency mode
    pub max_crashes_before_emergency: u32,
    
    /// Time window for crash counting
    pub crash_window: Duration,
    
    /// Force Emergency if unclean shutdown with positions
    pub emergency_on_unclean_with_positions: bool,
    
    /// Downgrade mode on recovery (e.g., FullAuto -> SemiAuto)
    pub downgrade_on_recovery: bool,
    
    /// Require manual approval after Emergency
    pub require_manual_after_emergency: bool,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            max_crashes_before_emergency: 3,
            crash_window: Duration::from_secs(86400), // 24 hours
            emergency_on_unclean_with_positions: true,
            downgrade_on_recovery: true,
            require_manual_after_emergency: true,
        }
    }
}

// ============================================================================
// DATABASE SCHEMA
// ============================================================================

const CREATE_MODE_STATE_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS control_mode_state (
    id SERIAL PRIMARY KEY,
    mode VARCHAR(20) NOT NULL,
    last_changed TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL,
    authorized_by VARCHAR(100) NOT NULL,
    previous_mode VARCHAR(20),
    recovery_info JSONB NOT NULL,
    state_data JSONB NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_mode CHECK (mode IN ('Manual', 'SemiAuto', 'FullAuto', 'Emergency'))
);

CREATE INDEX IF NOT EXISTS idx_mode_state_created ON control_mode_state(created_at DESC);
"#;

const CREATE_MODE_HISTORY_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS control_mode_history (
    id SERIAL PRIMARY KEY,
    from_mode VARCHAR(20) NOT NULL,
    to_mode VARCHAR(20) NOT NULL,
    transition_time TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL,
    authorized_by VARCHAR(100) NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mode_history_time ON control_mode_history(transition_time DESC);
CREATE INDEX IF NOT EXISTS idx_mode_history_success ON control_mode_history(success);
"#;

const CREATE_CRASH_RECOVERY_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS crash_recovery_log (
    id SERIAL PRIMARY KEY,
    crash_time TIMESTAMPTZ NOT NULL,
    recovery_time TIMESTAMPTZ NOT NULL,
    previous_mode VARCHAR(20) NOT NULL,
    recovered_mode VARCHAR(20) NOT NULL,
    recovery_reason TEXT NOT NULL,
    positions_at_crash INTEGER,
    exposure_at_crash DECIMAL(20, 8),
    recovery_action VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crash_recovery_time ON crash_recovery_log(crash_time DESC);
"#;

// ============================================================================
// MODE PERSISTENCE MANAGER
// ============================================================================

/// Manages mode state persistence and recovery
/// Sam: "Critical for safe restarts after crashes"
pub struct ModePersistenceManager {
    /// Database connection pool
    db_pool: PgPool,
    
    /// Current persisted state
    current_state: Arc<RwLock<Option<PersistedModeState>>>,
    
    /// Recovery policy
    recovery_policy: RecoveryPolicy,
    
    /// State change notifier
    state_notifier: broadcast::Sender<PersistedModeState>,
    
    /// System version
    system_version: String,
}

impl ModePersistenceManager {
    /// Create new persistence manager
    pub async fn new(
        database_url: &str,
        recovery_policy: RecoveryPolicy,
        system_version: String,
    ) -> Result<Self> {
        // Create connection pool with retry logic
        let db_pool = PgPoolOptions::new()
            .max_connections(5)
            .min_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .idle_timeout(Some(Duration::from_secs(300)))
            .connect(database_url)
            .await
            .context("Failed to connect to database")?;
            
        // Initialize schema
        Self::initialize_schema(&db_pool).await?;
        
        let (state_notifier, _) = broadcast::channel(100);
        
        Ok(Self {
            db_pool,
            current_state: Arc::new(RwLock::new(None)),
            recovery_policy,
            state_notifier,
            system_version,
        })
    }
    
    /// Initialize database schema
    async fn initialize_schema(pool: &PgPool) -> Result<()> {
        sqlx::query(CREATE_MODE_STATE_TABLE)
            .execute(pool)
            .await
            .context("Failed to create mode_state table")?;
            
        sqlx::query(CREATE_MODE_HISTORY_TABLE)
            .execute(pool)
            .await
            .context("Failed to create mode_history table")?;
            
        sqlx::query(CREATE_CRASH_RECOVERY_TABLE)
            .execute(pool)
            .await
            .context("Failed to create crash_recovery table")?;
            
        info!("Mode persistence schema initialized");
        Ok(())
    }
    
    /// Save current mode state to database
    /// Morgan: "Transactional save with integrity check"
    pub async fn save_mode_state(
        &self,
        mode: ControlMode,
        reason: String,
        authorized_by: String,
        state_data: serde_json::Value,
    ) -> Result<()> {
        let mut tx = self.db_pool.begin().await?;
        
        // Get recovery info
        let recovery_info = self.build_recovery_info().await?;
        
        // Get previous mode if exists
        let previous_mode = self.current_state.read().await
            .as_ref()
            .map(|s| s.mode);
            
        // Create persisted state
        let state = PersistedModeState {
            mode,
            last_changed: Utc::now(),
            reason: reason.clone(),
            authorized_by: authorized_by.clone(),
            previous_mode,
            recovery_info: recovery_info.clone(),
            state_data: state_data.clone(),
            checksum: self.calculate_checksum(&mode, &state_data),
        };
        
        // Insert into database
        sqlx::query(
            r#"
            INSERT INTO control_mode_state 
            (mode, last_changed, reason, authorized_by, previous_mode, 
             recovery_info, state_data, checksum)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#
        )
        .bind(format!("{:?}", mode))
        .bind(state.last_changed)
        .bind(&state.reason)
        .bind(&state.authorized_by)
        .bind(previous_mode.map(|m| format!("{:?}", m)))
        .bind(serde_json::to_value(&recovery_info)?)
        .bind(&state_data)
        .bind(&state.checksum)
        .execute(&mut *tx)
        .await?;
        
        // Record in history
        if let Some(prev) = previous_mode {
            sqlx::query(
                r#"
                INSERT INTO control_mode_history
                (from_mode, to_mode, transition_time, reason, authorized_by, success)
                VALUES ($1, $2, $3, $4, $5, $6)
                "#
            )
            .bind(format!("{:?}", prev))
            .bind(format!("{:?}", mode))
            .bind(Utc::now())
            .bind(&reason)
            .bind(&authorized_by)
            .bind(true)
            .execute(&mut *tx)
            .await?;
        }
        
        // Commit transaction
        tx.commit().await?;
        
        // Update current state
        *self.current_state.write().await = Some(state.clone());
        
        // Notify listeners
        let _ = self.state_notifier.send(state.clone());
        
        info!("Mode state persisted: {:?}", mode);
        Ok(())
    }
    
    /// Recover mode state after restart
    /// Alex: "Critical path - must handle all crash scenarios"
    pub async fn recover_mode_state(&self) -> Result<ControlMode> {
        // Load last saved state
        let last_state = self.load_last_state().await?;
        
        if let Some(state) = last_state {
            // Verify checksum
            if !self.verify_checksum(&state) {
                error!("Mode state checksum verification failed!");
                return Ok(ControlMode::Emergency);
            }
            
            // Check if this was a crash
            let was_crash = !state.recovery_info.clean_shutdown;
            
            if was_crash {
                // Log crash recovery
                self.log_crash_recovery(&state).await?;
                
                // Check crash count
                let crash_count = self.get_crash_count_24h().await?;
                
                if crash_count >= self.recovery_policy.max_crashes_before_emergency {
                    warn!("Too many crashes ({}) - forcing Emergency mode", crash_count);
                    self.save_mode_state(
                        ControlMode::Emergency,
                        format!("Excessive crashes: {} in 24h", crash_count),
                        "System".to_string(),
                        serde_json::json!({"crash_count": crash_count}),
                    ).await?;
                    return Ok(ControlMode::Emergency);
                }
                
                // Check if we had positions during crash
                if self.recovery_policy.emergency_on_unclean_with_positions 
                    && state.recovery_info.active_positions > 0 {
                    warn!("Unclean shutdown with {} positions - forcing Emergency mode", 
                          state.recovery_info.active_positions);
                    self.save_mode_state(
                        ControlMode::Emergency,
                        format!("Unclean shutdown with {} active positions", 
                               state.recovery_info.active_positions),
                        "System".to_string(),
                        serde_json::json!({
                            "positions": state.recovery_info.active_positions,
                            "exposure": state.recovery_info.total_exposure
                        }),
                    ).await?;
                    return Ok(ControlMode::Emergency);
                }
                
                // Apply recovery downgrade if configured
                if self.recovery_policy.downgrade_on_recovery {
                    let recovered_mode = self.downgrade_mode(state.mode);
                    if recovered_mode != state.mode {
                        info!("Downgrading mode from {:?} to {:?} after crash", 
                              state.mode, recovered_mode);
                        self.save_mode_state(
                            recovered_mode,
                            "Mode downgraded after crash recovery".to_string(),
                            "System".to_string(),
                            serde_json::json!({"previous_mode": format!("{:?}", state.mode)}),
                        ).await?;
                        return Ok(recovered_mode);
                    }
                }
            }
            
            // Check if we're recovering from Emergency
            if state.mode == ControlMode::Emergency 
                && self.recovery_policy.require_manual_after_emergency {
                info!("Recovering from Emergency - forcing Manual mode");
                self.save_mode_state(
                    ControlMode::Manual,
                    "Manual mode required after Emergency".to_string(),
                    "System".to_string(),
                    serde_json::json!({}),
                ).await?;
                return Ok(ControlMode::Manual);
            }
            
            // Restore previous state
            *self.current_state.write().await = Some(state.clone());
            info!("Recovered mode state: {:?}", state.mode);
            Ok(state.mode)
            
        } else {
            // No previous state - start in Manual mode
            info!("No previous mode state found - starting in Manual mode");
            self.save_mode_state(
                ControlMode::Manual,
                "Initial system startup".to_string(),
                "System".to_string(),
                serde_json::json!({}),
            ).await?;
            Ok(ControlMode::Manual)
        }
    }
    
    /// Mark current state for clean shutdown
    pub async fn mark_clean_shutdown(&self) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE control_mode_state 
            SET recovery_info = jsonb_set(recovery_info, '{clean_shutdown}', 'true')
            WHERE id = (SELECT id FROM control_mode_state ORDER BY created_at DESC LIMIT 1)
            "#
        )
        .execute(&self.db_pool)
        .await?;
        
        info!("Marked clean shutdown in persistence");
        Ok(())
    }
    
    /// Load last saved state from database
    async fn load_last_state(&self) -> Result<Option<PersistedModeState>> {
        let row = sqlx::query(
            r#"
            SELECT mode, last_changed, reason, authorized_by, previous_mode,
                   recovery_info, state_data, checksum
            FROM control_mode_state
            ORDER BY created_at DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.db_pool)
        .await?;
        
        if let Some(row) = row {
            let mode_str: String = row.get("mode");
            let mode = self.parse_mode(&mode_str)?;
            
            let previous_mode_str: Option<String> = row.get("previous_mode");
            let previous_mode = previous_mode_str
                .map(|s| self.parse_mode(&s))
                .transpose()?;
                
            Ok(Some(PersistedModeState {
                mode,
                last_changed: row.get("last_changed"),
                reason: row.get("reason"),
                authorized_by: row.get("authorized_by"),
                previous_mode,
                recovery_info: serde_json::from_value(row.get("recovery_info"))?,
                state_data: row.get("state_data"),
                checksum: row.get("checksum"),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Build recovery info for current state
    async fn build_recovery_info(&self) -> Result<RecoveryInfo> {
        let crash_count = self.get_crash_count_24h().await?;
        let last_crash = self.get_last_crash_time().await?;
        
        // TODO: Get actual position and exposure data from trading engine
        let active_positions = 0;
        let total_exposure = 0.0;
        
        Ok(RecoveryInfo {
            system_version: self.system_version.clone(),
            clean_shutdown: false, // Will be updated on clean shutdown
            crash_count_24h: crash_count,
            last_crash,
            active_positions,
            total_exposure,
            breakers_tripped: vec![], // TODO: Get from circuit breaker hub
        })
    }
    
    /// Get crash count in last 24 hours
    async fn get_crash_count_24h(&self) -> Result<u32> {
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) 
            FROM crash_recovery_log
            WHERE crash_time > NOW() - INTERVAL '24 hours'
            "#
        )
        .fetch_one(&self.db_pool)
        .await?;
        
        Ok(count as u32)
    }
    
    /// Get last crash timestamp
    async fn get_last_crash_time(&self) -> Result<Option<DateTime<Utc>>> {
        let time: Option<DateTime<Utc>> = sqlx::query_scalar(
            r#"
            SELECT crash_time
            FROM crash_recovery_log
            ORDER BY crash_time DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.db_pool)
        .await?;
        
        Ok(time)
    }
    
    /// Log crash recovery event
    async fn log_crash_recovery(&self, state: &PersistedModeState) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO crash_recovery_log
            (crash_time, recovery_time, previous_mode, recovered_mode, 
             recovery_reason, positions_at_crash, exposure_at_crash, recovery_action)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#
        )
        .bind(state.last_changed)
        .bind(Utc::now())
        .bind(format!("{:?}", state.mode))
        .bind(format!("{:?}", state.mode)) // Will be updated if downgraded
        .bind("Crash recovery")
        .bind(state.recovery_info.active_positions as i32)
        .bind(state.recovery_info.total_exposure)
        .bind("recover")
        .execute(&self.db_pool)
        .await?;
        
        Ok(())
    }
    
    /// Downgrade mode for recovery
    fn downgrade_mode(&self, mode: ControlMode) -> ControlMode {
        match mode {
            ControlMode::FullAuto => ControlMode::SemiAuto,
            ControlMode::SemiAuto => ControlMode::Manual,
            other => other,
        }
    }
    
    /// Parse mode from string
    fn parse_mode(&self, s: &str) -> Result<ControlMode> {
        match s {
            "Manual" => Ok(ControlMode::Manual),
            "SemiAuto" => Ok(ControlMode::SemiAuto),
            "FullAuto" => Ok(ControlMode::FullAuto),
            "Emergency" => Ok(ControlMode::Emergency),
            _ => bail!("Unknown mode: {}", s),
        }
    }
    
    /// Calculate checksum for integrity
    fn calculate_checksum(&self, mode: &ControlMode, data: &serde_json::Value) -> String {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", mode).as_bytes());
        hasher.update(data.to_string().as_bytes());
        hasher.update(self.system_version.as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Verify checksum integrity
    fn verify_checksum(&self, state: &PersistedModeState) -> bool {
        let calculated = self.calculate_checksum(&state.mode, &state.state_data);
        calculated == state.checksum
    }
    
    /// Get mode transition history
    pub async fn get_mode_history(&self, limit: i64) -> Result<Vec<ModeTransition>> {
        let rows = sqlx::query(
            r#"
            SELECT from_mode, to_mode, transition_time, reason, authorized_by
            FROM control_mode_history
            WHERE success = true
            ORDER BY transition_time DESC
            LIMIT $1
            "#
        )
        .bind(limit)
        .fetch_all(&self.db_pool)
        .await?;
        
        let mut transitions = Vec::new();
        for row in rows {
            let from_str: String = row.get("from_mode");
            let to_str: String = row.get("to_mode");
            
            // Convert to ModeTransition format
            transitions.push(ModeTransition {
                from: self.parse_mode(&from_str)?,
                to: self.parse_mode(&to_str)?,
                timestamp: row.get("transition_time"),
                reason: row.get("reason"),
                authorized_by: row.get("authorized_by"),
            });
        }
        
        Ok(transitions)
    }
    
    /// Subscribe to state changes
    pub fn subscribe(&self) -> broadcast::Receiver<PersistedModeState> {
        self.state_notifier.subscribe()
    }
}

// ============================================================================
// PARTIAL STATE RECOVERY
// ============================================================================

/// Handles partial state recovery for specific subsystems
/// Riley: "Granular recovery without full system restart"
pub struct PartialStateRecovery {
    persistence_manager: Arc<ModePersistenceManager>,
}

impl PartialStateRecovery {
    pub fn new(persistence_manager: Arc<ModePersistenceManager>) -> Self {
        Self { persistence_manager }
    }
    
    /// Recover specific subsystem state
    pub async fn recover_subsystem(
        &self,
        subsystem: &str,
        fallback_mode: ControlMode,
    ) -> Result<serde_json::Value> {
        // Load current state
        let state = self.persistence_manager.current_state.read().await;
        
        if let Some(ref persisted) = *state {
            // Extract subsystem-specific data
            if let Some(subsystem_data) = persisted.state_data.get(subsystem) {
                info!("Recovered {} subsystem state", subsystem);
                return Ok(subsystem_data.clone());
            }
        }
        
        // No state found - return default for fallback mode
        warn!("No state found for {} subsystem - using defaults", subsystem);
        Ok(self.get_subsystem_defaults(subsystem, fallback_mode))
    }
    
    /// Get default state for subsystem
    fn get_subsystem_defaults(&self, subsystem: &str, mode: ControlMode) -> serde_json::Value {
        match subsystem {
            "risk" => serde_json::json!({
                "max_position_size": mode.risk_multiplier() * 0.02,
                "max_leverage": 3.0 * mode.risk_multiplier(),
                "max_drawdown": 0.15,
            }),
            "ml" => serde_json::json!({
                "models_enabled": mode.allows_ml(),
                "confidence_threshold": 0.7,
                "ensemble_size": if mode.allows_ml() { 5 } else { 0 },
            }),
            "exchange" => serde_json::json!({
                "trading_enabled": mode.allows_trading(),
                "rate_limit_multiplier": mode.risk_multiplier(),
                "order_types": if mode == ControlMode::Manual {
                    vec!["limit"]
                } else {
                    vec!["limit", "market", "stop"]
                },
            }),
            _ => serde_json::json!({}),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn setup_test_manager() -> Result<ModePersistenceManager> {
        // Use in-memory SQLite for tests
        let manager = ModePersistenceManager::new(
            "sqlite::memory:",
            RecoveryPolicy::default(),
            "test-1.0.0".to_string(),
        ).await?;
        
        Ok(manager)
    }
    
    #[tokio::test]
    async fn test_save_and_recover_mode() {
        let manager = setup_test_manager().await.unwrap();
        
        // Save mode state
        manager.save_mode_state(
            ControlMode::FullAuto,
            "Test save".to_string(),
            "test_user".to_string(),
            serde_json::json!({"test": true}),
        ).await.unwrap();
        
        // Recover mode
        let recovered = manager.recover_mode_state().await.unwrap();
        assert_eq!(recovered, ControlMode::FullAuto);
    }
    
    #[tokio::test]
    async fn test_crash_recovery_downgrade() {
        let mut policy = RecoveryPolicy::default();
        policy.downgrade_on_recovery = true;
        
        let manager = ModePersistenceManager::new(
            "sqlite::memory:",
            policy,
            "test-1.0.0".to_string(),
        ).await.unwrap();
        
        // Save FullAuto mode
        manager.save_mode_state(
            ControlMode::FullAuto,
            "Normal operation".to_string(),
            "system".to_string(),
            serde_json::json!({}),
        ).await.unwrap();
        
        // Simulate crash (no clean shutdown marked)
        // On recovery, should downgrade to SemiAuto
        let recovered = manager.recover_mode_state().await.unwrap();
        assert_eq!(recovered, ControlMode::SemiAuto);
    }
    
    #[tokio::test]
    async fn test_emergency_after_crashes() {
        let mut policy = RecoveryPolicy::default();
        policy.max_crashes_before_emergency = 2;
        
        let manager = ModePersistenceManager::new(
            "sqlite::memory:",
            policy,
            "test-1.0.0".to_string(),
        ).await.unwrap();
        
        // Simulate multiple crashes
        for i in 0..3 {
            manager.save_mode_state(
                ControlMode::FullAuto,
                format!("Crash test {}", i),
                "system".to_string(),
                serde_json::json!({}),
            ).await.unwrap();
            
            // Log crash
            manager.log_crash_recovery(&PersistedModeState {
                mode: ControlMode::FullAuto,
                last_changed: Utc::now(),
                reason: "test".to_string(),
                authorized_by: "system".to_string(),
                previous_mode: None,
                recovery_info: RecoveryInfo {
                    system_version: "test".to_string(),
                    clean_shutdown: false,
                    crash_count_24h: i,
                    last_crash: Some(Utc::now()),
                    active_positions: 0,
                    total_exposure: 0.0,
                    breakers_tripped: vec![],
                },
                state_data: serde_json::json!({}),
                checksum: "test".to_string(),
            }).await.unwrap();
        }
        
        // Should force Emergency mode after crashes
        let recovered = manager.recover_mode_state().await.unwrap();
        assert_eq!(recovered, ControlMode::Emergency);
    }
    
    #[tokio::test]
    async fn test_clean_shutdown() {
        let manager = setup_test_manager().await.unwrap();
        
        // Save mode and mark clean shutdown
        manager.save_mode_state(
            ControlMode::FullAuto,
            "Normal".to_string(),
            "system".to_string(),
            serde_json::json!({}),
        ).await.unwrap();
        
        manager.mark_clean_shutdown().await.unwrap();
        
        // Should recover to same mode after clean shutdown
        let recovered = manager.recover_mode_state().await.unwrap();
        assert_eq!(recovered, ControlMode::FullAuto);
    }
    
    #[tokio::test]
    async fn test_checksum_verification() {
        let manager = setup_test_manager().await.unwrap();
        
        let state = PersistedModeState {
            mode: ControlMode::Manual,
            last_changed: Utc::now(),
            reason: "test".to_string(),
            authorized_by: "test".to_string(),
            previous_mode: None,
            recovery_info: RecoveryInfo {
                system_version: "test-1.0.0".to_string(),
                clean_shutdown: true,
                crash_count_24h: 0,
                last_crash: None,
                active_positions: 0,
                total_exposure: 0.0,
                breakers_tripped: vec![],
            },
            state_data: serde_json::json!({"test": true}),
            checksum: "invalid".to_string(),
        };
        
        // Should fail checksum verification
        assert!(!manager.verify_checksum(&state));
        
        // Calculate correct checksum
        let correct_checksum = manager.calculate_checksum(&state.mode, &state.state_data);
        let mut correct_state = state.clone();
        correct_state.checksum = correct_checksum;
        
        // Should pass with correct checksum
        assert!(manager.verify_checksum(&correct_state));
    }
    
    #[tokio::test]
    async fn test_partial_state_recovery() {
        let manager = Arc::new(setup_test_manager().await.unwrap());
        
        // Save state with subsystem data
        manager.save_mode_state(
            ControlMode::FullAuto,
            "test".to_string(),
            "test".to_string(),
            serde_json::json!({
                "risk": {
                    "custom_limit": 0.01
                },
                "ml": {
                    "model_version": "v2.0"
                }
            }),
        ).await.unwrap();
        
        let recovery = PartialStateRecovery::new(manager.clone());
        
        // Recover risk subsystem
        let risk_state = recovery.recover_subsystem("risk", ControlMode::Manual)
            .await.unwrap();
        assert_eq!(risk_state["custom_limit"], 0.01);
        
        // Try to recover non-existent subsystem
        let exchange_state = recovery.recover_subsystem("exchange", ControlMode::Manual)
            .await.unwrap();
        assert!(exchange_state.get("trading_enabled").is_some());
    }
}