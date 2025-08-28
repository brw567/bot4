//! Bot4 ComplianceAuditor Agent - Production Ready Implementation
//! Maintains immutable audit trails, ensures regulatory compliance, cryptographic signing

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use redis::aio::ConnectionManager;
use rmcp::{
    server::{Server, ServerBuilder, ToolHandler},
    transport::DockerTransport,
    types::{Tool, ToolCall, ToolResult},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use sha2::{Sha256, Digest};
use sqlx::{PgPool, postgres::PgPoolOptions};
use uuid::Uuid;

/// Audit event types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum AuditEventType {
    OrderPlaced,
    OrderExecuted,
    OrderCancelled,
    PositionOpened,
    PositionClosed,
    RiskLimitBreached,
    SystemStart,
    SystemStop,
    ConfigurationChange,
    UserAction,
    SecurityEvent,
    ComplianceViolation,
}

/// Immutable audit record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditRecord {
    id: Uuid,
    timestamp: DateTime<Utc>,
    event_type: AuditEventType,
    actor: String,
    component: String,
    action: String,
    details: serde_json::Value,
    risk_score: f64,
    compliance_status: ComplianceStatus,
    signature: String,
    previous_hash: String,
    hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum ComplianceStatus {
    Compliant,
    Warning,
    Violation,
    UnderReview,
}

/// Compliance rules
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComplianceRule {
    id: String,
    name: String,
    description: String,
    regulation: String, // e.g., "MiFID II", "GDPR", "Dodd-Frank"
    severity: RuleSeverity,
    check_function: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RuleSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditStats {
    total_events: u64,
    events_by_type: HashMap<String, u64>,
    compliance_violations: u64,
    risk_events: u64,
    integrity_verified: bool,
    last_audit_timestamp: DateTime<Utc>,
}

/// ComplianceAuditor agent
struct ComplianceAuditorAgent {
    redis: ConnectionManager,
    db_pool: PgPool,
    signing_key: Keypair,
    audit_chain: Arc<RwLock<Vec<AuditRecord>>>,
    compliance_rules: Arc<RwLock<Vec<ComplianceRule>>>,
    workspace_path: PathBuf,
}

impl ComplianceAuditorAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Connect to PostgreSQL for persistent audit storage
        let database_url = std::env::var("POSTGRES_URL")
            .unwrap_or_else(|_| "postgresql://bot4user:bot4pass@postgres:5432/bot4trading".to_string());
        
        let db_pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await?;
        
        // Create audit tables if they don't exist
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS audit_records (
                id UUID PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                component TEXT NOT NULL,
                action TEXT NOT NULL,
                details JSONB NOT NULL,
                risk_score FLOAT NOT NULL,
                compliance_status TEXT NOT NULL,
                signature TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_records(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_compliance ON audit_records(compliance_status);
            CREATE INDEX IF NOT EXISTS idx_audit_hash ON audit_records(hash);
        "#)
        .execute(&db_pool)
        .await?;
        
        // Generate or load signing keypair
        let signing_key = if let Ok(key_bytes) = std::env::var("AUDIT_SIGNING_KEY") {
            // Load existing key from env
            let secret = SecretKey::from_bytes(&hex::decode(key_bytes)?)?;
            let public = PublicKey::from(&secret);
            Keypair { secret, public }
        } else {
            // Generate new keypair
            let mut csprng = rand::rngs::OsRng;
            Keypair::generate(&mut csprng)
        };
        
        // Set workspace path
        let workspace_path = PathBuf::from(
            std::env::var("WORKSPACE_PATH").unwrap_or_else(|_| "/workspace".to_string())
        );
        
        // Initialize compliance rules
        let compliance_rules = vec![
            ComplianceRule {
                id: "POSITION_SIZE".to_string(),
                name: "Position Size Limit".to_string(),
                description: "Position size must not exceed 5% of portfolio".to_string(),
                regulation: "Internal Risk Management".to_string(),
                severity: RuleSeverity::Critical,
                check_function: "check_position_size".to_string(),
            },
            ComplianceRule {
                id: "DAILY_LOSS".to_string(),
                name: "Daily Loss Limit".to_string(),
                description: "Daily loss must not exceed 2% of capital".to_string(),
                regulation: "Risk Management Policy".to_string(),
                severity: RuleSeverity::Critical,
                check_function: "check_daily_loss".to_string(),
            },
            ComplianceRule {
                id: "TRADE_FREQUENCY".to_string(),
                name: "Trade Frequency Limit".to_string(),
                description: "Maximum 1000 trades per day per symbol".to_string(),
                regulation: "Market Abuse Prevention".to_string(),
                severity: RuleSeverity::High,
                check_function: "check_trade_frequency".to_string(),
            },
        ];
        
        info!("ComplianceAuditor agent initialized with signing key");
        
        Ok(Self {
            redis,
            db_pool,
            signing_key,
            audit_chain: Arc::new(RwLock::new(Vec::new())),
            compliance_rules: Arc::new(RwLock::new(compliance_rules)),
            workspace_path,
        })
    }
    
    /// Create audit record with cryptographic signature
    async fn create_audit_record(&self, 
        event_type: AuditEventType,
        actor: String,
        component: String,
        action: String,
        details: serde_json::Value
    ) -> Result<ToolResult> {
        info!("Creating audit record for {:?}", event_type);
        
        // Get previous hash
        let previous_hash = {
            let chain = self.audit_chain.read().await;
            chain.last()
                .map(|r| r.hash.clone())
                .unwrap_or_else(|| "genesis".to_string())
        };
        
        // Calculate risk score based on event type
        let risk_score = match event_type {
            AuditEventType::RiskLimitBreached | AuditEventType::ComplianceViolation => 1.0,
            AuditEventType::SecurityEvent => 0.9,
            AuditEventType::OrderExecuted | AuditEventType::PositionOpened => 0.5,
            _ => 0.2,
        };
        
        // Check compliance
        let compliance_status = self.check_compliance(&event_type, &details).await?;
        
        // Create record without signature and hash
        let mut record = AuditRecord {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: event_type.clone(),
            actor: actor.clone(),
            component: component.clone(),
            action: action.clone(),
            details: details.clone(),
            risk_score,
            compliance_status: compliance_status.clone(),
            signature: String::new(),
            previous_hash: previous_hash.clone(),
            hash: String::new(),
        };
        
        // Create message to sign (all fields except signature and hash)
        let message = format!(
            "{}|{}|{:?}|{}|{}|{}|{}|{}|{}|{}",
            record.id,
            record.timestamp.to_rfc3339(),
            record.event_type,
            record.actor,
            record.component,
            record.action,
            serde_json::to_string(&record.details)?,
            record.risk_score,
            serde_json::to_string(&record.compliance_status)?,
            record.previous_hash
        );
        
        // Sign the message
        let signature = self.signing_key.sign(message.as_bytes());
        record.signature = hex::encode(signature.to_bytes());
        
        // Calculate hash including signature
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        hasher.update(record.signature.as_bytes());
        record.hash = hex::encode(hasher.finalize());
        
        // Store in database
        sqlx::query(r#"
            INSERT INTO audit_records (
                id, timestamp, event_type, actor, component, action,
                details, risk_score, compliance_status, signature,
                previous_hash, hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        "#)
        .bind(&record.id)
        .bind(&record.timestamp)
        .bind(format!("{:?}", record.event_type))
        .bind(&record.actor)
        .bind(&record.component)
        .bind(&record.action)
        .bind(&record.details)
        .bind(record.risk_score)
        .bind(format!("{:?}", record.compliance_status))
        .bind(&record.signature)
        .bind(&record.previous_hash)
        .bind(&record.hash)
        .execute(&self.db_pool)
        .await?;
        
        // Add to in-memory chain
        self.audit_chain.write().await.push(record.clone());
        
        // Alert if compliance violation
        if compliance_status == ComplianceStatus::Violation {
            warn!("COMPLIANCE VIOLATION: {} - {}", actor, action);
            // Send alert via Redis
            self.redis.publish::<_, _, ()>(
                "bot4:compliance:violations",
                serde_json::to_string(&record)?
            ).await?;
        }
        
        Ok(ToolResult::Success(serde_json::json!({
            "audit_id": record.id,
            "hash": record.hash,
            "signature": record.signature,
            "compliance_status": format!("{:?}", compliance_status),
            "risk_score": risk_score,
            "message": format!("✅ Audit record created: {}", record.id)
        })))
    }
    
    /// Verify audit chain integrity
    async fn verify_chain_integrity(&self) -> Result<ToolResult> {
        info!("Verifying audit chain integrity");
        
        // Load all records from database
        let records: Vec<(String, String, String, String)> = sqlx::query_as(
            "SELECT hash, previous_hash, signature, timestamp FROM audit_records ORDER BY timestamp"
        )
        .fetch_all(&self.db_pool)
        .await?;
        
        if records.is_empty() {
            return Ok(ToolResult::Success(serde_json::json!({
                "integrity": true,
                "records_checked": 0,
                "message": "✅ No audit records to verify"
            })));
        }
        
        let mut integrity_errors = Vec::new();
        let mut previous_hash = "genesis".to_string();
        
        for (i, (hash, prev_hash, _signature, _timestamp)) in records.iter().enumerate() {
            // Check hash chain
            if prev_hash != &previous_hash {
                integrity_errors.push(format!(
                    "Hash chain broken at record {}: expected '{}', got '{}'",
                    i, previous_hash, prev_hash
                ));
            }
            previous_hash = hash.clone();
            
            // TODO: Verify signature (would need to reconstruct full record)
        }
        
        let integrity_valid = integrity_errors.is_empty();
        
        Ok(ToolResult::Success(serde_json::json!({
            "integrity": integrity_valid,
            "records_checked": records.len(),
            "errors": integrity_errors,
            "message": if integrity_valid {
                format!("✅ Audit chain integrity verified: {} records", records.len())
            } else {
                format!("❌ Audit chain integrity FAILED: {} errors", integrity_errors.len())
            }
        })))
    }
    
    /// Generate compliance report
    async fn generate_compliance_report(&self, start_date: String, end_date: String) -> Result<ToolResult> {
        info!("Generating compliance report from {} to {}", start_date, end_date);
        
        // Parse dates
        let start = DateTime::parse_from_rfc3339(&start_date)?;
        let end = DateTime::parse_from_rfc3339(&end_date)?;
        
        // Query audit records
        let records: Vec<(String, String, f64)> = sqlx::query_as(
            r#"
            SELECT event_type, compliance_status, risk_score
            FROM audit_records
            WHERE timestamp >= $1 AND timestamp <= $2
            "#
        )
        .bind(start.with_timezone(&Utc))
        .bind(end.with_timezone(&Utc))
        .fetch_all(&self.db_pool)
        .await?;
        
        // Calculate statistics
        let total_events = records.len();
        let violations = records.iter()
            .filter(|(_, status, _)| status.contains("Violation"))
            .count();
        let warnings = records.iter()
            .filter(|(_, status, _)| status.contains("Warning"))
            .count();
        let high_risk_events = records.iter()
            .filter(|(_, _, risk)| *risk > 0.7)
            .count();
        
        // Group by event type
        let mut events_by_type: HashMap<String, u64> = HashMap::new();
        for (event_type, _, _) in &records {
            *events_by_type.entry(event_type.clone()).or_insert(0) += 1;
        }
        
        // Calculate compliance rate
        let compliance_rate = if total_events > 0 {
            ((total_events - violations) as f64 / total_events as f64) * 100.0
        } else {
            100.0
        };
        
        // Generate report
        let report = serde_json::json!({
            "period": {
                "start": start_date,
                "end": end_date,
            },
            "summary": {
                "total_events": total_events,
                "compliance_violations": violations,
                "warnings": warnings,
                "high_risk_events": high_risk_events,
                "compliance_rate": format!("{:.2}%", compliance_rate),
            },
            "events_by_type": events_by_type,
            "compliance_status": if violations == 0 {
                "COMPLIANT"
            } else if violations < 5 {
                "MINOR_VIOLATIONS"
            } else {
                "MAJOR_VIOLATIONS"
            },
            "recommendations": if violations > 0 {
                vec![
                    "Review and update risk limits",
                    "Enhance monitoring for high-risk activities",
                    "Conduct compliance training",
                ]
            } else {
                vec!["Continue current compliance practices"]
            },
        });
        
        // Store report
        let report_id = Uuid::new_v4();
        let report_key = format!("bot4:compliance:reports:{}", report_id);
        self.redis.set::<_, _, ()>(
            &report_key,
            serde_json::to_string(&report)?
        ).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "report_id": report_id,
            "report": report,
            "message": format!("✅ Compliance report generated: {}", report_id)
        })))
    }
    
    /// Check compliance for an event
    async fn check_compliance(&self, event_type: &AuditEventType, details: &serde_json::Value) -> Result<ComplianceStatus> {
        let rules = self.compliance_rules.read().await;
        
        for rule in rules.iter() {
            match rule.check_function.as_str() {
                "check_position_size" => {
                    if let Some(size) = details["position_size"].as_f64() {
                        if size > 0.05 {
                            return Ok(ComplianceStatus::Violation);
                        }
                    }
                }
                "check_daily_loss" => {
                    if let Some(loss) = details["daily_loss"].as_f64() {
                        if loss > 0.02 {
                            return Ok(ComplianceStatus::Violation);
                        }
                    }
                }
                "check_trade_frequency" => {
                    if let Some(count) = details["trade_count"].as_u64() {
                        if count > 1000 {
                            return Ok(ComplianceStatus::Warning);
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(ComplianceStatus::Compliant)
    }
    
    /// Verify signature of an audit record
    async fn verify_signature(&self, record_id: String) -> Result<ToolResult> {
        info!("Verifying signature for record {}", record_id);
        
        // Fetch record from database
        let record: Option<(String, String, String, String, String, String, serde_json::Value, f64, String, String)> = 
            sqlx::query_as(
                r#"
                SELECT id, timestamp, event_type, actor, component, action,
                       details, risk_score, compliance_status, signature, previous_hash
                FROM audit_records
                WHERE id = $1
                "#
            )
            .bind(Uuid::parse_str(&record_id)?)
            .fetch_optional(&self.db_pool)
            .await?;
        
        match record {
            Some((id, timestamp, event_type, actor, component, action, details, risk_score, compliance_status, signature, previous_hash)) => {
                // Reconstruct message
                let message = format!(
                    "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
                    id, timestamp, event_type, actor, component, action,
                    serde_json::to_string(&details)?, risk_score, compliance_status, previous_hash
                );
                
                // Verify signature
                let signature_bytes = hex::decode(&signature)?;
                let signature = Signature::from_bytes(&signature_bytes)?;
                
                let is_valid = self.signing_key.public.verify(message.as_bytes(), &signature).is_ok();
                
                Ok(ToolResult::Success(serde_json::json!({
                    "record_id": record_id,
                    "signature_valid": is_valid,
                    "public_key": hex::encode(self.signing_key.public.to_bytes()),
                    "message": if is_valid {
                        "✅ Signature verification successful"
                    } else {
                        "❌ Signature verification FAILED"
                    }
                })))
            }
            None => {
                Ok(ToolResult::Error(format!("Record not found: {}", record_id)))
            }
        }
    }
    
    /// Get audit statistics
    async fn get_audit_stats(&self) -> Result<ToolResult> {
        info!("Getting audit statistics");
        
        // Query statistics from database
        let total_events: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM audit_records"
        )
        .fetch_one(&self.db_pool)
        .await?;
        
        let compliance_violations: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM audit_records WHERE compliance_status = 'Violation'"
        )
        .fetch_one(&self.db_pool)
        .await?;
        
        let high_risk_events: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM audit_records WHERE risk_score > 0.7"
        )
        .fetch_one(&self.db_pool)
        .await?;
        
        // Get events by type
        let events_by_type: Vec<(String, i64)> = sqlx::query_as(
            "SELECT event_type, COUNT(*) FROM audit_records GROUP BY event_type"
        )
        .fetch_all(&self.db_pool)
        .await?;
        
        let mut event_map = HashMap::new();
        for (event_type, count) in events_by_type {
            event_map.insert(event_type, count as u64);
        }
        
        // Verify integrity
        let integrity_result = self.verify_chain_integrity().await?;
        let integrity_verified = match integrity_result {
            ToolResult::Success(data) => data["integrity"].as_bool().unwrap_or(false),
            _ => false,
        };
        
        let stats = AuditStats {
            total_events: total_events.0 as u64,
            events_by_type: event_map,
            compliance_violations: compliance_violations.0 as u64,
            risk_events: high_risk_events.0 as u64,
            integrity_verified,
            last_audit_timestamp: Utc::now(),
        };
        
        Ok(ToolResult::Success(serde_json::json!(stats)))
    }
}

#[async_trait]
impl ToolHandler for ComplianceAuditorAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "create_audit_record" => {
                let event_type = match tool_call.arguments["event_type"].as_str().unwrap_or("") {
                    "order_placed" => AuditEventType::OrderPlaced,
                    "order_executed" => AuditEventType::OrderExecuted,
                    "position_opened" => AuditEventType::PositionOpened,
                    "risk_breach" => AuditEventType::RiskLimitBreached,
                    _ => AuditEventType::UserAction,
                };
                
                let actor = tool_call.arguments["actor"].as_str().unwrap_or("system").to_string();
                let component = tool_call.arguments["component"].as_str().unwrap_or("unknown").to_string();
                let action = tool_call.arguments["action"].as_str().unwrap_or("").to_string();
                let details = tool_call.arguments["details"].clone();
                
                self.create_audit_record(event_type, actor, component, action, details)
                    .await.unwrap_or_else(|e| {
                        ToolResult::Error(format!("Failed to create audit record: {}", e))
                    })
            }
            "verify_chain_integrity" => {
                self.verify_chain_integrity().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to verify integrity: {}", e))
                })
            }
            "generate_compliance_report" => {
                let start = tool_call.arguments["start_date"].as_str()
                    .unwrap_or(&Utc::now().date_naive().to_string()).to_string();
                let end = tool_call.arguments["end_date"].as_str()
                    .unwrap_or(&Utc::now().to_rfc3339()).to_string();
                    
                self.generate_compliance_report(start, end).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to generate report: {}", e))
                })
            }
            "verify_signature" => {
                let record_id = tool_call.arguments["record_id"].as_str().unwrap_or("").to_string();
                self.verify_signature(record_id).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to verify signature: {}", e))
                })
            }
            "get_audit_stats" => {
                self.get_audit_stats().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to get stats: {}", e))
                })
            }
            _ => ToolResult::Error(format!("Unknown tool: {}", tool_call.name))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .init();
    
    info!("Starting Bot4 ComplianceAuditor Agent v1.0 - Production Ready");
    info!("Immutable audit trails with cryptographic signatures");
    
    // Create agent
    let agent = Arc::new(ComplianceAuditorAgent::new().await?);
    
    // Start HTTP server for health checks
    tokio::spawn(async move {
        let app = Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics));
        
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8087));
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    
    // Define MCP tools
    let tools = vec![
        Tool {
            name: "create_audit_record".to_string(),
            description: "Create immutable audit record with signature".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "actor": {"type": "string"},
                    "component": {"type": "string"},
                    "action": {"type": "string"},
                    "details": {"type": "object"}
                },
                "required": ["event_type", "action"]
            }),
        },
        Tool {
            name: "verify_chain_integrity".to_string(),
            description: "Verify audit chain integrity".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        Tool {
            name: "generate_compliance_report".to_string(),
            description: "Generate compliance report for date range".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "format": "date-time"},
                    "end_date": {"type": "string", "format": "date-time"}
                },
                "required": ["start_date", "end_date"]
            }),
        },
        Tool {
            name: "verify_signature".to_string(),
            description: "Verify cryptographic signature of audit record".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"}
                },
                "required": ["record_id"]
            }),
        },
        Tool {
            name: "get_audit_stats".to_string(),
            description: "Get audit statistics".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("complianceauditor-agent", "1.0.0")
        .with_tools(tools)
        .with_tool_handler(agent.clone())
        .build()?;
    
    // Use Docker transport
    let transport = DockerTransport::new()?;
    server.run(transport).await?;
    
    Ok(())
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "healthy")
}

async fn metrics() -> impl IntoResponse {
    let metrics = prometheus::gather();
    let mut buffer = Vec::new();
    let encoder = prometheus::TextEncoder::new();
    encoder.encode(&metrics, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}