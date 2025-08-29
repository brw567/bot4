// TAMPER-PROOF AUDIT SYSTEM - Task 0.7
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Immutable audit trail for compliance and forensics
// External Research Applied:
// - "Mastering Bitcoin" - Antonopoulos (2017) - Merkle trees
// - "Applied Cryptography" - Schneier (1996) - Hash chains
// - MiFID II audit trail requirements
// - SEC Rule 613 (CAT) compliance standards
// - "Building Secure and Reliable Systems" - Google (2020)
// - Certificate Transparency (RFC 6962) - Append-only logs
// - "The Byzantine Generals Problem" - Lamport (1982)
// - Event Sourcing patterns - Martin Fowler

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error};
use anyhow::Result;
use sha2::{Sha256, Digest};
use chrono::Utc;

// ============================================================================
// AUDIT EVENT TYPES - Comprehensive event taxonomy
// ============================================================================

/// Categories of auditable events
/// Alex: "Every action that affects money or risk must be audited"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AuditEventType {
    // Trading Events
    OrderPlaced,
    OrderModified,
    OrderCancelled,
    OrderExecuted,
    OrderRejected,
    
    // Position Events
    PositionOpened,
    PositionClosed,
    PositionModified,
    StopLossTriggered,
    TakeProfitTriggered,
    Liquidation,
    
    // Risk Events
    RiskLimitBreached,
    MarginCall,
    DrawdownAlert,
    EmergencyStop,
    CircuitBreakerTrip,
    
    // System Events
    SystemStart,
    SystemStop,
    ConfigChange,
    StrategySwitch,
    ManualIntervention,
    
    // Market Events
    DataFeedLoss,
    DataFeedRestore,
    ExchangeDisconnect,
    ExchangeReconnect,
    AnomalyDetected,
    
    // Compliance Events
    ComplianceViolation,
    AuditQuery,
    ReportGenerated,
    DataExport,
    
    // Security Events
    AuthenticationFailure,
    UnauthorizedAccess,
    SuspiciousActivity,
    IntegrityViolation,
}

impl AuditEventType {
    /// Determine if event requires immediate notification
    pub fn requires_immediate_notification(&self) -> bool {
        matches!(
            self,
            AuditEventType::EmergencyStop |
            AuditEventType::Liquidation |
            AuditEventType::ComplianceViolation |
            AuditEventType::IntegrityViolation |
            AuditEventType::UnauthorizedAccess |
            AuditEventType::ManualIntervention
        )
    }
    
    /// Get severity level for prioritization
    pub fn severity(&self) -> AuditSeverity {
        match self {
            AuditEventType::IntegrityViolation |
            AuditEventType::UnauthorizedAccess |
            AuditEventType::EmergencyStop => AuditSeverity::Critical,
            
            AuditEventType::ComplianceViolation |
            AuditEventType::Liquidation |
            AuditEventType::ManualIntervention => AuditSeverity::High,
            
            AuditEventType::RiskLimitBreached |
            AuditEventType::MarginCall |
            AuditEventType::CircuitBreakerTrip => AuditSeverity::Medium,
            
            _ => AuditSeverity::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AuditSeverity {
    Info = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

// ============================================================================
// AUDIT EVENT STRUCTURE - Immutable event record
// ============================================================================

/// Immutable audit event with cryptographic proof
/// Sam: "Each event must be independently verifiable"
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AuditEvent {
    /// Unique event ID (UUID v4)
    pub id: String,
    
    /// Event timestamp (microsecond precision)
    pub timestamp: u64,
    
    /// Sequence number for ordering
    pub sequence: u64,
    
    /// Event type
    pub event_type: AuditEventType,
    
    /// Component that generated the event
    pub source: String,
    
    /// User or system that triggered the event
    pub actor: String,
    
    /// Affected entity (order ID, position ID, etc.)
    pub entity_id: Option<String>,
    
    /// Event description
    pub description: String,
    
    /// Structured event data
    pub data: HashMap<String, serde_json::Value>,
    
    /// Previous event hash (for chain integrity)
    pub previous_hash: String,
    
    /// This event's hash
    pub event_hash: String,
    
    /// Digital signature (Ed25519)
    pub signature: Option<Vec<u8>>,
    
    /// Merkle tree root at time of event
    pub merkle_root: Option<String>,
}

impl AuditEvent {
    /// Create new audit event
    pub fn new(
        event_type: AuditEventType,
        source: String,
        actor: String,
        description: String,
        data: HashMap<String, serde_json::Value>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        Self {
            id,
            timestamp,
            sequence: 0, // Will be set by audit log
            event_type,
            source,
            actor,
            entity_id: None,
            description,
            data,
            previous_hash: String::new(),
            event_hash: String::new(),
            signature: None,
            merkle_root: None,
        }
    }
    
    /// Calculate event hash
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        
        // Include all immutable fields
        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.sequence.to_le_bytes());
        hasher.update(format!("{:?}", self.event_type).as_bytes());
        hasher.update(self.source.as_bytes());
        hasher.update(self.actor.as_bytes());
        
        if let Some(ref entity) = self.entity_id {
            hasher.update(entity.as_bytes());
        }
        
        hasher.update(self.description.as_bytes());
        
        // Serialize data deterministically
        if let Ok(data_json) = serde_json::to_string(&self.data) {
            hasher.update(data_json.as_bytes());
        }
        
        hasher.update(self.previous_hash.as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Verify event integrity
    pub fn verify_hash(&self) -> bool {
        self.event_hash == self.calculate_hash()
    }
}

// ============================================================================
// MERKLE TREE IMPLEMENTATION - Cryptographic proof structure
// ============================================================================

/// Merkle tree node for audit proof
/// Based on "Certificate Transparency" RFC 6962
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MerkleNode {
    hash: String,
    left: Option<Box<MerkleNode>>,
    right: Option<Box<MerkleNode>>,
}

/// Merkle tree for batch verification
/// TODO: Add docs
pub struct MerkleTree {
    root: Option<MerkleNode>,
    leaves: Vec<String>,
}

impl MerkleTree {
    /// Build merkle tree from event hashes
    pub fn from_hashes(hashes: Vec<String>) -> Self {
        if hashes.is_empty() {
            return Self {
                root: None,
                leaves: vec![],
            };
        }
        
        let leaves = hashes.clone();
        let root = Self::build_tree(hashes);
        
        Self {
            root: Some(root),
            leaves,
        }
    }
    
    /// Recursively build merkle tree
    fn build_tree(mut hashes: Vec<String>) -> MerkleNode {
        if hashes.len() == 1 {
            return MerkleNode {
                hash: hashes[0].clone(),
                left: None,
                right: None,
            };
        }
        
        // Ensure even number of nodes
        if hashes.len() % 2 != 0 {
            hashes.push(hashes.last().unwrap().clone());
        }
        
        let mut next_level = Vec::new();
        
        for chunk in hashes.chunks(2) {
            let combined = Self::hash_pair(&chunk[0], &chunk[1]);
            next_level.push(combined);
        }
        
        Self::build_tree(next_level)
    }
    
    /// Hash two nodes together
    fn hash_pair(left: &str, right: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(left.as_bytes());
        hasher.update(right.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Get merkle root hash
    pub fn root_hash(&self) -> Option<String> {
        self.root.as_ref().map(|node| node.hash.clone())
    }
    
    /// Generate merkle proof for a specific leaf
    pub fn generate_proof(&self, leaf_index: usize) -> Vec<String> {
        if leaf_index >= self.leaves.len() {
            return vec![];
        }
        
        let mut proof = Vec::new();
        let mut index = leaf_index;
        let mut level_size = self.leaves.len();
        
        while level_size > 1 {
            let sibling_index = if index % 2 == 0 {
                index + 1
            } else {
                index - 1
            };
            
            if sibling_index < level_size {
                proof.push(self.leaves[sibling_index].clone());
            }
            
            index /= 2;
            level_size = (level_size + 1) / 2;
        }
        
        proof
    }
}

// ============================================================================
// APPEND-ONLY LOG - Immutable event storage
// ============================================================================

/// Append-only audit log with hash chain
/// Quinn: "Log must be tamper-evident and crash-resistant"
/// TODO: Add docs
pub struct AuditLog {
    /// Log file path
    log_path: PathBuf,
    
    /// Current file writer
    writer: Arc<RwLock<BufWriter<File>>>,
    
    /// In-memory cache of recent events
    cache: Arc<RwLock<VecDeque<AuditEvent>>>,
    
    /// Current sequence number
    sequence: Arc<AtomicU64>,
    
    /// Last event hash (for chaining)
    last_hash: Arc<RwLock<String>>,
    
    /// Merkle tree of current batch
    current_batch: Arc<RwLock<Vec<String>>>,
    
    /// Batch size for merkle tree
    batch_size: usize,
    
    /// Statistics
    total_events: Arc<AtomicU64>,
    events_per_type: Arc<RwLock<HashMap<AuditEventType, u64>>>,
}

impl AuditLog {
    /// Create new audit log
    pub fn new(log_dir: &Path, batch_size: usize) -> Result<Self> {
        // Ensure log directory exists
        std::fs::create_dir_all(log_dir)?;
        
        // Generate log file name with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let log_path = log_dir.join(format!("audit_{}.log", timestamp));
        
        // Open file in append-only mode
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        
        let writer = BufWriter::new(file);
        
        // Initialize genesis block
        let genesis_hash = Self::genesis_hash();
        
        Ok(Self {
            log_path,
            writer: Arc::new(RwLock::new(writer)),
            cache: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            sequence: Arc::new(AtomicU64::new(0)),
            last_hash: Arc::new(RwLock::new(genesis_hash)),
            current_batch: Arc::new(RwLock::new(Vec::new())),
            batch_size,
            total_events: Arc::new(AtomicU64::new(0)),
            events_per_type: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Generate genesis block hash
    fn genesis_hash() -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"BOT4_AUDIT_GENESIS_BLOCK_2025");
        format!("{:x}", hasher.finalize())
    }
    
    /// Append event to log
    pub fn append(&self, mut event: AuditEvent) -> Result<()> {
        // Set sequence number
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        event.sequence = sequence;
        
        // Chain to previous event
        {
            let last_hash = self.last_hash.read();
            event.previous_hash = last_hash.clone();
        }
        
        // Calculate event hash
        event.event_hash = event.calculate_hash();
        
        // Write to file (fsync for durability)
        {
            let mut writer = self.writer.write();
            let json = serde_json::to_string(&event)?;
            writeln!(writer, "{}", json)?;
            writer.flush()?;
            writer.get_ref().sync_all()?; // Force to disk
        }
        
        // Update last hash
        {
            let mut last_hash = self.last_hash.write();
            *last_hash = event.event_hash.clone();
        }
        
        // Add to cache
        {
            let mut cache = self.cache.write();
            cache.push_back(event.clone());
            if cache.len() > 10000 {
                cache.pop_front();
            }
        }
        
        // Add to current batch for merkle tree
        {
            let mut batch = self.current_batch.write();
            batch.push(event.event_hash.clone());
            
            if batch.len() >= self.batch_size {
                // Create merkle tree and clear batch
                let tree = MerkleTree::from_hashes(batch.clone());
                if let Some(root) = tree.root_hash() {
                    info!("Merkle root for batch {}: {}", sequence / self.batch_size as u64, root);
                }
                batch.clear();
            }
        }
        
        // Update statistics
        self.total_events.fetch_add(1, Ordering::Relaxed);
        {
            let mut stats = self.events_per_type.write();
            *stats.entry(event.event_type).or_insert(0) += 1;
        }
        
        // Send immediate notification if required
        if event.event_type.requires_immediate_notification() {
            warn!("CRITICAL AUDIT EVENT: {:?} - {}", event.event_type, event.description);
        }
        
        Ok(())
    }
    
    /// Verify log integrity from a specific sequence
    pub fn verify_integrity(&self, from_sequence: u64) -> Result<bool> {
        let cache = self.cache.read();
        let mut previous_hash = Self::genesis_hash();
        
        for event in cache.iter() {
            if event.sequence < from_sequence {
                continue;
            }
            
            // Verify hash chain
            if event.previous_hash != previous_hash {
                error!("Hash chain broken at sequence {}", event.sequence);
                return Ok(false);
            }
            
            // Verify event hash
            if !event.verify_hash() {
                error!("Event hash invalid at sequence {}", event.sequence);
                return Ok(false);
            }
            
            previous_hash = event.event_hash.clone();
        }
        
        Ok(true)
    }
    
    /// Get events by type
    pub fn get_events_by_type(&self, event_type: AuditEventType, limit: usize) -> Vec<AuditEvent> {
        let cache = self.cache.read();
        cache.iter()
            .filter(|e| e.event_type == event_type)
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get events in time range
    pub fn get_events_in_range(&self, start: u64, end: u64) -> Vec<AuditEvent> {
        let cache = self.cache.read();
        cache.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect()
    }
}

// ============================================================================
// COMPLIANCE REPORT GENERATOR - Regulatory reporting
// ============================================================================

/// Compliance report types
/// Morgan: "Reports must meet MiFID II and SEC requirements"
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum ReportType {
    DailyTrading,
    RiskExposure,
    OrderAudit,
    BestExecution,
    MarketAbuse,
    SystemIncident,
    ComplianceSummary,
    Custom(String),
}

/// Compliance report generator
/// TODO: Add docs
pub struct ComplianceReporter {
    audit_log: Arc<AuditLog>,
    report_dir: PathBuf,
}

impl ComplianceReporter {
    pub fn new(audit_log: Arc<AuditLog>, report_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&report_dir).ok();
        Self {
            audit_log,
            report_dir,
        }
    }
    
    /// Generate compliance report
    pub fn generate_report(&self, report_type: ReportType, start: u64, end: u64) -> Result<PathBuf> {
        let events = self.audit_log.get_events_in_range(start, end);
        
        let report = match report_type {
            ReportType::DailyTrading => self.generate_trading_report(&events)?,
            ReportType::RiskExposure => self.generate_risk_report(&events)?,
            ReportType::OrderAudit => self.generate_order_audit(&events)?,
            ReportType::BestExecution => self.generate_best_execution_report(&events)?,
            ReportType::MarketAbuse => self.generate_market_abuse_report(&events)?,
            ReportType::SystemIncident => self.generate_incident_report(&events)?,
            ReportType::ComplianceSummary => self.generate_compliance_summary(&events)?,
            ReportType::Custom(ref name) => self.generate_custom_report(name, &events)?,
        };
        
        // Save report to file
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("{:?}_{}.json", report_type, timestamp);
        let path = self.report_dir.join(filename);
        
        let file = File::create(&path)?;
        serde_json::to_writer_pretty(file, &report)?;
        
        Ok(path)
    }
    
    /// Generate daily trading report
    fn generate_trading_report(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let mut orders = 0;
        let mut executions = 0;
        let mut cancellations = 0;
        let mut rejections = 0;
        
        for event in events {
            match event.event_type {
                AuditEventType::OrderPlaced => orders += 1,
                AuditEventType::OrderExecuted => executions += 1,
                AuditEventType::OrderCancelled => cancellations += 1,
                AuditEventType::OrderRejected => rejections += 1,
                _ => {}
            }
        }
        
        Ok(serde_json::json!({
            "report_type": "DailyTrading",
            "total_orders": orders,
            "executions": executions,
            "cancellations": cancellations,
            "rejections": rejections,
            "execution_rate": if orders > 0 { executions as f64 / orders as f64 } else { 0.0 },
            "events": events.len(),
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate risk exposure report
    fn generate_risk_report(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let mut risk_breaches = 0;
        let mut margin_calls = 0;
        let mut liquidations = 0;
        let mut circuit_trips = 0;
        
        for event in events {
            match event.event_type {
                AuditEventType::RiskLimitBreached => risk_breaches += 1,
                AuditEventType::MarginCall => margin_calls += 1,
                AuditEventType::Liquidation => liquidations += 1,
                AuditEventType::CircuitBreakerTrip => circuit_trips += 1,
                _ => {}
            }
        }
        
        Ok(serde_json::json!({
            "report_type": "RiskExposure",
            "risk_limit_breaches": risk_breaches,
            "margin_calls": margin_calls,
            "liquidations": liquidations,
            "circuit_breaker_trips": circuit_trips,
            "total_risk_events": risk_breaches + margin_calls + liquidations + circuit_trips,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate order audit trail
    fn generate_order_audit(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let order_events: Vec<&AuditEvent> = events.iter()
            .filter(|e| matches!(
                e.event_type,
                AuditEventType::OrderPlaced |
                AuditEventType::OrderModified |
                AuditEventType::OrderCancelled |
                AuditEventType::OrderExecuted |
                AuditEventType::OrderRejected
            ))
            .collect();
        
        Ok(serde_json::json!({
            "report_type": "OrderAudit",
            "total_order_events": order_events.len(),
            "order_trail": order_events,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate best execution report (MiFID II requirement)
    fn generate_best_execution_report(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        // Analyze execution quality
        let executions: Vec<&AuditEvent> = events.iter()
            .filter(|e| e.event_type == AuditEventType::OrderExecuted)
            .collect();
        
        Ok(serde_json::json!({
            "report_type": "BestExecution",
            "total_executions": executions.len(),
            "execution_details": executions,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate market abuse detection report
    fn generate_market_abuse_report(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let suspicious: Vec<&AuditEvent> = events.iter()
            .filter(|e| matches!(
                e.event_type,
                AuditEventType::AnomalyDetected |
                AuditEventType::SuspiciousActivity |
                AuditEventType::ComplianceViolation
            ))
            .collect();
        
        Ok(serde_json::json!({
            "report_type": "MarketAbuse",
            "suspicious_events": suspicious.len(),
            "details": suspicious,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate system incident report
    fn generate_incident_report(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let incidents: Vec<&AuditEvent> = events.iter()
            .filter(|e| matches!(
                e.event_type,
                AuditEventType::EmergencyStop |
                AuditEventType::DataFeedLoss |
                AuditEventType::ExchangeDisconnect |
                AuditEventType::IntegrityViolation
            ))
            .collect();
        
        Ok(serde_json::json!({
            "report_type": "SystemIncident",
            "total_incidents": incidents.len(),
            "incident_details": incidents,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate compliance summary
    fn generate_compliance_summary(&self, events: &[AuditEvent]) -> Result<serde_json::Value> {
        let mut summary = HashMap::new();
        
        for event in events {
            *summary.entry(event.event_type).or_insert(0u64) += 1;
        }
        
        Ok(serde_json::json!({
            "report_type": "ComplianceSummary",
            "total_events": events.len(),
            "event_breakdown": summary,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
    
    /// Generate custom report
    fn generate_custom_report(&self, name: &str, events: &[AuditEvent]) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "report_type": format!("Custom({})", name),
            "total_events": events.len(),
            "events": events,
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }
}

// ============================================================================
// INTERVENTION DETECTOR - Real-time manual intervention detection
// ============================================================================

/// Manual intervention detector
/// Casey: "Must detect any human override of automated systems"
/// TODO: Add docs
pub struct InterventionDetector {
    audit_log: Arc<AuditLog>,
    detection_window: Duration,
    alert_threshold: u32,
}

impl InterventionDetector {
    pub fn new(audit_log: Arc<AuditLog>, detection_window: Duration, alert_threshold: u32) -> Self {
        Self {
            audit_log,
            detection_window,
            alert_threshold,
        }
    }
    
    /// Check for manual interventions
    pub fn check_interventions(&self) -> Vec<InterventionAlert> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let start = now - (self.detection_window.as_micros() as u64);
        let events = self.audit_log.get_events_in_range(start, now);
        
        let mut alerts = Vec::new();
        let mut intervention_count = 0;
        
        for event in &events {
            if event.event_type == AuditEventType::ManualIntervention {
                intervention_count += 1;
                
                alerts.push(InterventionAlert {
                    timestamp: event.timestamp,
                    actor: event.actor.clone(),
                    action: event.description.clone(),
                    severity: InterventionSeverity::from_count(intervention_count),
                    event_id: event.id.clone(),
                });
            }
        }
        
        // Check for pattern of interventions
        if intervention_count >= self.alert_threshold {
            error!(
                "CRITICAL: {} manual interventions detected in {:?}",
                intervention_count, self.detection_window
            );
        }
        
        alerts
    }
}

/// Intervention alert
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct InterventionAlert {
    pub timestamp: u64,
    pub actor: String,
    pub action: String,
    pub severity: InterventionSeverity,
    pub event_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// TODO: Add docs
pub enum InterventionSeverity {
    Low,      // Single intervention
    Medium,   // 2-3 interventions
    High,     // 4-5 interventions
    Critical, // >5 interventions
}

impl InterventionSeverity {
    fn from_count(count: u32) -> Self {
        match count {
            1 => Self::Low,
            2..=3 => Self::Medium,
            4..=5 => Self::High,
            _ => Self::Critical,
        }
    }
}

// ============================================================================
// FORENSIC ANALYZER - Post-incident analysis tools
// ============================================================================

/// Forensic analysis tools
/// Riley: "Must reconstruct exact sequence of events for any incident"
/// TODO: Add docs
pub struct ForensicAnalyzer {
    audit_log: Arc<AuditLog>,
}

impl ForensicAnalyzer {
    pub fn new(audit_log: Arc<AuditLog>) -> Self {
        Self { audit_log }
    }
    
    /// Analyze incident timeline
    pub fn analyze_incident(&self, incident_id: &str) -> Result<IncidentAnalysis> {
        // Find the incident event
        let cache = self.audit_log.cache.read();
        let incident = cache.iter()
            .find(|e| e.id == incident_id || e.entity_id.as_deref() == Some(incident_id))
            .ok_or_else(|| anyhow::anyhow!("Incident not found"))?;
        
        // Get events around the incident (±5 minutes)
        let window = 5 * 60 * 1_000_000; // 5 minutes in microseconds
        let start = incident.timestamp.saturating_sub(window);
        let end = incident.timestamp.saturating_add(window);
        
        let related_events = self.audit_log.get_events_in_range(start, end);
        
        // Analyze causal chain
        let mut preceding_events = Vec::new();
        let mut following_events = Vec::new();
        
        for event in related_events {
            if event.timestamp < incident.timestamp {
                preceding_events.push(event);
            } else if event.timestamp > incident.timestamp {
                following_events.push(event);
            }
        }
        
        // Sort by timestamp
        preceding_events.sort_by_key(|e| e.timestamp);
        following_events.sort_by_key(|e| e.timestamp);
        
        // Identify potential causes
        let potential_causes: Vec<String> = preceding_events.iter()
            .filter(|e| e.event_type.severity() >= AuditSeverity::Medium)
            .map(|e| format!("{:?}: {}", e.event_type, e.description))
            .collect();
        
        // Identify consequences
        let consequences: Vec<String> = following_events.iter()
            .filter(|e| e.event_type.severity() >= AuditSeverity::Medium)
            .map(|e| format!("{:?}: {}", e.event_type, e.description))
            .collect();
        
        Ok(IncidentAnalysis {
            incident_id: incident_id.to_string(),
            incident_type: incident.event_type,
            timestamp: incident.timestamp,
            description: incident.description.clone(),
            preceding_events: preceding_events.len(),
            following_events: following_events.len(),
            potential_causes,
            consequences,
            timeline: self.build_timeline(&preceding_events, incident, &following_events),
        })
    }
    
    /// Build incident timeline
    fn build_timeline(
        &self,
        preceding: &[AuditEvent],
        incident: &AuditEvent,
        following: &[AuditEvent],
    ) -> Vec<TimelineEntry> {
        let mut timeline = Vec::new();
        
        // Add preceding events
        for event in preceding.iter().rev().take(10) {
            timeline.push(TimelineEntry {
                timestamp: event.timestamp,
                event_type: event.event_type,
                description: event.description.clone(),
                is_incident: false,
            });
        }
        
        // Add incident
        timeline.push(TimelineEntry {
            timestamp: incident.timestamp,
            event_type: incident.event_type,
            description: incident.description.clone(),
            is_incident: true,
        });
        
        // Add following events
        for event in following.iter().take(10) {
            timeline.push(TimelineEntry {
                timestamp: event.timestamp,
                event_type: event.event_type,
                description: event.description.clone(),
                is_incident: false,
            });
        }
        
        timeline
    }
    
    /// Verify system state at specific time
    pub fn verify_state_at(&self, timestamp: u64) -> Result<SystemState> {
        let events = self.audit_log.get_events_in_range(0, timestamp);
        
        let mut positions = 0;
        let mut orders = 0;
        let mut risk_breaches = 0;
        let mut manual_interventions = 0;
        
        for event in &events {
            match event.event_type {
                AuditEventType::PositionOpened => positions += 1,
                AuditEventType::PositionClosed => positions -= 1,
                AuditEventType::OrderPlaced => orders += 1,
                AuditEventType::OrderExecuted | AuditEventType::OrderCancelled => orders -= 1,
                AuditEventType::RiskLimitBreached => risk_breaches += 1,
                AuditEventType::ManualIntervention => manual_interventions += 1,
                _ => {}
            }
        }
        
        Ok(SystemState {
            timestamp,
            open_positions: positions.max(0) as u32,
            pending_orders: orders.max(0) as u32,
            risk_breaches: risk_breaches as u32,
            manual_interventions: manual_interventions as u32,
            total_events: events.len() as u64,
        })
    }
}

/// Incident analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct IncidentAnalysis {
    pub incident_id: String,
    pub incident_type: AuditEventType,
    pub timestamp: u64,
    pub description: String,
    pub preceding_events: usize,
    pub following_events: usize,
    pub potential_causes: Vec<String>,
    pub consequences: Vec<String>,
    pub timeline: Vec<TimelineEntry>,
}

/// Timeline entry
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct TimelineEntry {
    pub timestamp: u64,
    pub event_type: AuditEventType,
    pub description: String,
    pub is_incident: bool,
}

/// System state at point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct SystemState {
    pub timestamp: u64,
    pub open_positions: u32,
    pub pending_orders: u32,
    pub risk_breaches: u32,
    pub manual_interventions: u32,
    pub total_events: u64,
}

// ============================================================================
// AUDIT MANAGER - Central coordination
// ============================================================================

/// Central audit system manager
/// Avery: "Single point of coordination for all audit functions"
/// TODO: Add docs
pub struct AuditManager {
    audit_log: Arc<AuditLog>,
    compliance_reporter: Arc<ComplianceReporter>,
    intervention_detector: Arc<InterventionDetector>,
    forensic_analyzer: Arc<ForensicAnalyzer>,
    
    // Configuration
    config: AuditConfig,
    
    // Status
    is_running: Arc<AtomicBool>,
}

impl AuditManager {
    pub fn new(config: AuditConfig) -> Result<Self> {
        let audit_log = Arc::new(AuditLog::new(&config.log_dir, config.batch_size)?);
        
        let compliance_reporter = Arc::new(ComplianceReporter::new(
            audit_log.clone(),
            config.report_dir.clone(),
        ));
        
        let intervention_detector = Arc::new(InterventionDetector::new(
            audit_log.clone(),
            config.intervention_window,
            config.intervention_threshold,
        ));
        
        let forensic_analyzer = Arc::new(ForensicAnalyzer::new(audit_log.clone()));
        
        Ok(Self {
            audit_log,
            compliance_reporter,
            intervention_detector,
            forensic_analyzer,
            config,
            is_running: Arc::new(AtomicBool::new(true)),
        })
    }
    
    /// Log an audit event
    pub fn log_event(
        &self,
        event_type: AuditEventType,
        source: String,
        actor: String,
        description: String,
        data: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let event = AuditEvent::new(event_type, source, actor, description, data);
        self.audit_log.append(event)?;
        Ok(())
    }
    
    /// Generate compliance report
    pub fn generate_report(
        &self,
        report_type: ReportType,
        start: u64,
        end: u64,
    ) -> Result<PathBuf> {
        self.compliance_reporter.generate_report(report_type, start, end)
    }
    
    /// Check for manual interventions
    pub fn check_interventions(&self) -> Vec<InterventionAlert> {
        self.intervention_detector.check_interventions()
    }
    
    /// Analyze incident
    pub fn analyze_incident(&self, incident_id: &str) -> Result<IncidentAnalysis> {
        self.forensic_analyzer.analyze_incident(incident_id)
    }
    
    /// Verify system state at timestamp
    pub fn verify_state_at(&self, timestamp: u64) -> Result<SystemState> {
        self.forensic_analyzer.verify_state_at(timestamp)
    }
    
    /// Verify log integrity
    pub fn verify_integrity(&self) -> Result<bool> {
        self.audit_log.verify_integrity(0)
    }
    
    /// Get audit statistics
    pub fn get_statistics(&self) -> AuditStatistics {
        let stats = self.audit_log.events_per_type.read();
        
        AuditStatistics {
            total_events: self.audit_log.total_events.load(Ordering::Relaxed),
            events_by_type: stats.clone(),
            log_file: self.audit_log.log_path.clone(),
            is_running: self.is_running.load(Ordering::Relaxed),
        }
    }
    
    /// Shutdown audit system
    pub fn shutdown(&self) -> Result<()> {
        self.is_running.store(false, Ordering::SeqCst);
        
        // Flush any pending writes
        let mut writer = self.audit_log.writer.write();
        writer.flush()?;
        writer.get_ref().sync_all()?;
        
        info!("Audit system shutdown complete");
        Ok(())
    }
}

/// Audit system configuration
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct AuditConfig {
    pub log_dir: PathBuf,
    pub report_dir: PathBuf,
    pub batch_size: usize,
    pub intervention_window: Duration,
    pub intervention_threshold: u32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("/var/log/bot4/audit"),
            report_dir: PathBuf::from("/var/log/bot4/reports"),
            batch_size: 100,
            intervention_window: Duration::from_secs(300), // 5 minutes
            intervention_threshold: 5,
        }
    }
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AuditStatistics {
    pub total_events: u64,
    pub events_by_type: HashMap<AuditEventType, u64>,
    pub log_file: PathBuf,
    pub is_running: bool,
}

// ============================================================================
// TESTS - Comprehensive validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_audit_event_creation() {
        let mut data = HashMap::new();
        data.insert("order_id".to_string(), serde_json::json!("12345"));
        
        let event = AuditEvent::new(
            AuditEventType::OrderPlaced,
            "trading_engine".to_string(),
            "system".to_string(),
            "Order placed for BTC/USD".to_string(),
            data,
        );
        
        assert_eq!(event.event_type, AuditEventType::OrderPlaced);
        assert!(!event.id.is_empty());
    }
    
    #[test]
    fn test_event_hash_verification() {
        let event = AuditEvent::new(
            AuditEventType::SystemStart,
            "main".to_string(),
            "system".to_string(),
            "System started".to_string(),
            HashMap::new(),
        );
        
        let mut event_with_hash = event.clone();
        event_with_hash.event_hash = event_with_hash.calculate_hash();
        
        assert!(event_with_hash.verify_hash());
    }
    
    #[test]
    fn test_merkle_tree() {
        let hashes = vec![
            "hash1".to_string(),
            "hash2".to_string(),
            "hash3".to_string(),
            "hash4".to_string(),
        ];
        
        let tree = MerkleTree::from_hashes(hashes);
        assert!(tree.root_hash().is_some());
    }
    
    #[test]
    fn test_audit_log_append() {
        let temp_dir = TempDir::new().unwrap();
        let log = AuditLog::new(temp_dir.path(), 10).unwrap();
        
        let event = AuditEvent::new(
            AuditEventType::OrderPlaced,
            "test".to_string(),
            "user".to_string(),
            "Test order".to_string(),
            HashMap::new(),
        );
        
        log.append(event).unwrap();
        assert_eq!(log.total_events.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_log_integrity_verification() {
        let temp_dir = TempDir::new().unwrap();
        let log = AuditLog::new(temp_dir.path(), 10).unwrap();
        
        // Add multiple events
        for i in 0..5 {
            let event = AuditEvent::new(
                AuditEventType::OrderPlaced,
                "test".to_string(),
                "user".to_string(),
                format!("Order {}", i),
                HashMap::new(),
            );
            log.append(event).unwrap();
        }
        
        // Verify integrity
        assert!(log.verify_integrity(0).unwrap());
    }
    
    #[test]
    fn test_compliance_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let log = Arc::new(AuditLog::new(temp_dir.path(), 10).unwrap());
        let reporter = ComplianceReporter::new(log.clone(), temp_dir.path().to_path_buf());
        
        // Add some events
        for _ in 0..3 {
            let event = AuditEvent::new(
                AuditEventType::OrderExecuted,
                "test".to_string(),
                "system".to_string(),
                "Order executed".to_string(),
                HashMap::new(),
            );
            log.append(event).unwrap();
        }
        
        let report_path = reporter.generate_report(
            ReportType::DailyTrading,
            0,
            u64::MAX,
        ).unwrap();
        
        assert!(report_path.exists());
    }
    
    #[test]
    fn test_intervention_detection() {
        let temp_dir = TempDir::new().unwrap();
        let log = Arc::new(AuditLog::new(temp_dir.path(), 10).unwrap());
        let detector = InterventionDetector::new(
            log.clone(),
            Duration::from_secs(300),
            3,
        );
        
        // Add manual intervention events
        for i in 0..4 {
            let event = AuditEvent::new(
                AuditEventType::ManualIntervention,
                "ui".to_string(),
                "operator".to_string(),
                format!("Manual override {}", i),
                HashMap::new(),
            );
            log.append(event).unwrap();
        }
        
        let alerts = detector.check_interventions();
        assert_eq!(alerts.len(), 4);
        assert!(matches!(alerts.last().unwrap().severity, InterventionSeverity::High));
    }
    
    #[test]
    fn test_forensic_analysis() {
        let temp_dir = TempDir::new().unwrap();
        let log = Arc::new(AuditLog::new(temp_dir.path(), 10).unwrap());
        let analyzer = ForensicAnalyzer::new(log.clone());
        
        // Create incident scenario
        let mut incident_id = String::new();
        
        // Preceding events
        for i in 0..3 {
            let event = AuditEvent::new(
                AuditEventType::RiskLimitBreached,
                "risk".to_string(),
                "system".to_string(),
                format!("Risk breach {}", i),
                HashMap::new(),
            );
            log.append(event).unwrap();
        }
        
        // Incident
        let incident = AuditEvent::new(
            AuditEventType::EmergencyStop,
            "safety".to_string(),
            "system".to_string(),
            "Emergency stop triggered".to_string(),
            HashMap::new(),
        );
        incident_id = incident.id.clone();
        log.append(incident).unwrap();
        
        // Following events
        for i in 0..2 {
            let event = AuditEvent::new(
                AuditEventType::PositionClosed,
                "trading".to_string(),
                "system".to_string(),
                format!("Position closed {}", i),
                HashMap::new(),
            );
            log.append(event).unwrap();
        }
        
        // Analyze incident
        let analysis = analyzer.analyze_incident(&incident_id).unwrap();
        assert_eq!(analysis.incident_type, AuditEventType::EmergencyStop);
        assert!(!analysis.potential_causes.is_empty());
    }
}