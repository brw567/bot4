//! Bot4 QualityGate Agent - Production Ready Implementation
//! Enforces 100% test coverage, prevents fake implementations, ensures code quality

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
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use syn::{parse_file, Item, ItemFn, ItemStruct, ItemImpl};
use walkdir::WalkDir;
use regex::Regex;
use tokio::process::Command;

mod coverage_analyzer;
mod fake_detector;
mod duplication_checker;
mod security_scanner;

use coverage_analyzer::CoverageAnalyzer;
use fake_detector::FakeDetector;
use duplication_checker::DuplicationChecker;
use security_scanner::SecurityScanner;

/// Quality thresholds for production
const MIN_TEST_COVERAGE: f64 = 100.0;
const MAX_DUPLICATION_PERCENT: f64 = 0.0;
const MAX_CYCLOMATIC_COMPLEXITY: u32 = 10;
const MIN_DOCUMENTATION_COVERAGE: f64 = 100.0;

/// QualityGate agent with strict enforcement
struct QualityGateAgent {
    redis: ConnectionManager,
    coverage_analyzer: Arc<CoverageAnalyzer>,
    fake_detector: Arc<FakeDetector>,
    duplication_checker: Arc<DuplicationChecker>,
    security_scanner: Arc<SecurityScanner>,
    workspace_path: PathBuf,
    enforcement_mode: EnforcementMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EnforcementMode {
    Strict,     // Block all violations
    Warning,    // Warn but allow
    Development,// Relaxed for dev
}

impl QualityGateAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Set workspace path
        let workspace_path = PathBuf::from(
            std::env::var("WORKSPACE_PATH").unwrap_or_else(|_| "/workspace".to_string())
        );
        
        // Determine enforcement mode
        let enforcement_mode = match std::env::var("ENFORCEMENT_MODE").as_deref() {
            Ok("warning") => EnforcementMode::Warning,
            Ok("development") => EnforcementMode::Development,
            _ => EnforcementMode::Strict, // Default to strict
        };
        
        info!("QualityGate agent initialized in {:?} mode", enforcement_mode);
        
        Ok(Self {
            redis,
            coverage_analyzer: Arc::new(CoverageAnalyzer::new()),
            fake_detector: Arc::new(FakeDetector::new()),
            duplication_checker: Arc::new(DuplicationChecker::new()),
            security_scanner: Arc::new(SecurityScanner::new()),
            workspace_path,
            enforcement_mode,
        })
    }
    
    /// Check test coverage for the entire project
    async fn check_test_coverage(&self) -> Result<ToolResult> {
        info!("Checking test coverage for Bot4 project");
        
        // Run cargo tarpaulin
        let output = Command::new("cargo")
            .args(&["tarpaulin", "--out", "Json", "--all-features", "--workspace"])
            .current_dir(&self.workspace_path)
            .output()
            .await?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(ToolResult::Error(format!("Failed to run coverage: {}", stderr)));
        }
        
        // Parse coverage results
        let coverage_data = self.coverage_analyzer.parse_tarpaulin_output(&output.stdout)?;
        
        let total_coverage = coverage_data.line_coverage_percent;
        let uncovered_files = coverage_data.files.iter()
            .filter(|f| f.coverage < MIN_TEST_COVERAGE)
            .collect::<Vec<_>>();
        
        // Check against threshold
        let passes = total_coverage >= MIN_TEST_COVERAGE;
        
        let result = if passes {
            format!("✅ Test coverage: {:.1}% (meets {}% requirement)", 
                    total_coverage, MIN_TEST_COVERAGE)
        } else {
            let message = format!(
                "❌ COVERAGE VIOLATION: {:.1}% < {}% required\n\nUncovered files:\n{}",
                total_coverage,
                MIN_TEST_COVERAGE,
                uncovered_files.iter()
                    .map(|f| format!("  - {} ({:.1}%)", f.path, f.coverage))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            
            if self.enforcement_mode == EnforcementMode::Strict {
                return Ok(ToolResult::Error(message));
            }
            message
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "total_coverage": total_coverage,
            "required_coverage": MIN_TEST_COVERAGE,
            "passes": passes,
            "uncovered_files": uncovered_files.len(),
            "enforcement_mode": format!("{:?}", self.enforcement_mode),
            "message": result,
        })))
    }
    
    /// Detect fake implementations (todo!, unimplemented!, panic!, hardcoded values)
    async fn detect_fake_implementations(&self) -> Result<ToolResult> {
        info!("Detecting fake implementations");
        
        let mut fake_count = 0;
        let mut violations = Vec::new();
        
        // Patterns that indicate fake implementations
        let fake_patterns = vec![
            (r"todo!\s*\(", "todo! macro"),
            (r"unimplemented!\s*\(", "unimplemented! macro"),
            (r"panic!\s*\([^)]*(not.implemented|todo|fixme)", "panic with TODO message"),
            (r"return\s+Ok\(\(\)\)", "empty Ok return"),
            (r"return\s+(0|1|true|false|\"\")\s*;?\s*//.*TODO", "hardcoded return with TODO"),
            (r"sleep\s*\([^)]*\)\s*//.*fake", "fake delay"),
        ];
        
        for entry in WalkDir::new(&self.workspace_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
            .filter(|e| !e.path().to_string_lossy().contains("/tests/"))
            .filter(|e| !e.path().to_string_lossy().contains("/target/"))
        {
            let path = entry.path();
            if let Ok(content) = tokio::fs::read_to_string(path).await {
                for (pattern, description) in &fake_patterns {
                    let re = Regex::new(pattern)?;
                    for mat in re.find_iter(&content) {
                        let line_num = content[..mat.start()].lines().count() + 1;
                        violations.push(format!("{}:{} - {}", 
                            path.display(), line_num, description));
                        fake_count += 1;
                    }
                }
                
                // Parse AST for more sophisticated detection
                if let Ok(syntax_tree) = parse_file(&content) {
                    let ast_violations = self.fake_detector.analyze_ast(&syntax_tree)?;
                    for violation in ast_violations {
                        violations.push(format!("{}: {}", path.display(), violation));
                        fake_count += 1;
                    }
                }
            }
        }
        
        let passes = fake_count == 0;
        
        let result = if passes {
            "✅ No fake implementations detected".to_string()
        } else {
            let message = format!(
                "❌ FAKE IMPLEMENTATIONS FOUND: {} violations\n\n{}",
                fake_count,
                violations.join("\n")
            );
            
            if self.enforcement_mode == EnforcementMode::Strict {
                return Ok(ToolResult::Error(message));
            }
            message
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "fake_implementations": fake_count,
            "violations": violations,
            "passes": passes,
            "enforcement_mode": format!("{:?}", self.enforcement_mode),
            "message": result,
        })))
    }
    
    /// Check for code duplication
    async fn check_duplication(&self) -> Result<ToolResult> {
        info!("Checking for code duplication");
        
        let duplicates = self.duplication_checker.find_duplicates(&self.workspace_path).await?;
        
        let total_files = duplicates.total_files;
        let duplication_percent = (duplicates.duplicate_blocks as f64 / total_files as f64) * 100.0;
        
        let passes = duplication_percent <= MAX_DUPLICATION_PERCENT;
        
        let result = if passes {
            format!("✅ Duplication: {:.1}% (within {:.0}% limit)", 
                    duplication_percent, MAX_DUPLICATION_PERCENT)
        } else {
            let message = format!(
                "❌ DUPLICATION VIOLATION: {:.1}% > {:.0}% allowed\n\nDuplicate blocks:\n{}",
                duplication_percent,
                MAX_DUPLICATION_PERCENT,
                duplicates.locations.iter()
                    .map(|loc| format!("  - {}", loc))
                    .take(10)
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            
            if self.enforcement_mode == EnforcementMode::Strict {
                return Ok(ToolResult::Error(message));
            }
            message
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "duplication_percent": duplication_percent,
            "max_allowed": MAX_DUPLICATION_PERCENT,
            "duplicate_blocks": duplicates.duplicate_blocks,
            "passes": passes,
            "message": result,
        })))
    }
    
    /// Run security scan
    async fn security_scan(&self) -> Result<ToolResult> {
        info!("Running security scan");
        
        let mut vulnerabilities = Vec::new();
        
        // Run cargo audit
        let audit_output = Command::new("cargo")
            .args(&["audit", "--json"])
            .current_dir(&self.workspace_path)
            .output()
            .await?;
        
        if let Ok(audit_data) = serde_json::from_slice::<serde_json::Value>(&audit_output.stdout) {
            if let Some(vulns) = audit_data["vulnerabilities"]["list"].as_array() {
                for vuln in vulns {
                    vulnerabilities.push(format!(
                        "{}: {} (severity: {})",
                        vuln["package"]["name"].as_str().unwrap_or("unknown"),
                        vuln["advisory"]["title"].as_str().unwrap_or("unknown"),
                        vuln["advisory"]["severity"].as_str().unwrap_or("unknown")
                    ));
                }
            }
        }
        
        // Check for hardcoded secrets
        let secret_patterns = vec![
            (r"(api[_\-]?key|apikey)\s*=\s*[\"'][a-zA-Z0-9]{20,}[\"']", "API key"),
            (r"(secret|password|passwd|pwd)\s*=\s*[\"'][^\"']{8,}[\"']", "Password"),
            (r"(token|auth)\s*=\s*[\"'][a-zA-Z0-9]{20,}[\"']", "Token"),
            (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "Private key"),
        ];
        
        for entry in WalkDir::new(&self.workspace_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()).map(|s| 
                s == "rs" || s == "toml" || s == "yaml" || s == "json").unwrap_or(false))
        {
            if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                for (pattern, description) in &secret_patterns {
                    let re = Regex::new(pattern)?;
                    if re.is_match(&content) {
                        vulnerabilities.push(format!(
                            "{}: Potential {} exposed",
                            entry.path().display(),
                            description
                        ));
                    }
                }
            }
        }
        
        let passes = vulnerabilities.is_empty();
        
        let result = if passes {
            "✅ No security vulnerabilities detected".to_string()
        } else {
            let message = format!(
                "❌ SECURITY VIOLATIONS: {} issues found\n\n{}",
                vulnerabilities.len(),
                vulnerabilities.join("\n")
            );
            
            if self.enforcement_mode == EnforcementMode::Strict {
                return Ok(ToolResult::Error(message));
            }
            message
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "vulnerabilities": vulnerabilities.len(),
            "issues": vulnerabilities,
            "passes": passes,
            "message": result,
        })))
    }
    
    /// Run all quality checks
    async fn run_quality_gate(&self) -> Result<ToolResult> {
        info!("Running comprehensive quality gate checks");
        
        let mut all_pass = true;
        let mut results = HashMap::new();
        
        // Test coverage
        match self.check_test_coverage().await {
            Ok(ToolResult::Success(data)) => {
                results.insert("test_coverage", data);
            }
            Ok(ToolResult::Error(msg)) => {
                all_pass = false;
                results.insert("test_coverage", serde_json::json!({"error": msg}));
            }
            Err(e) => {
                all_pass = false;
                results.insert("test_coverage", serde_json::json!({"error": e.to_string()}));
            }
        }
        
        // Fake detection
        match self.detect_fake_implementations().await {
            Ok(ToolResult::Success(data)) => {
                if !data["passes"].as_bool().unwrap_or(false) {
                    all_pass = false;
                }
                results.insert("fake_detection", data);
            }
            Ok(ToolResult::Error(msg)) => {
                all_pass = false;
                results.insert("fake_detection", serde_json::json!({"error": msg}));
            }
            Err(e) => {
                all_pass = false;
                results.insert("fake_detection", serde_json::json!({"error": e.to_string()}));
            }
        }
        
        // Duplication check
        match self.check_duplication().await {
            Ok(ToolResult::Success(data)) => {
                if !data["passes"].as_bool().unwrap_or(false) {
                    all_pass = false;
                }
                results.insert("duplication", data);
            }
            Ok(ToolResult::Error(msg)) => {
                all_pass = false;
                results.insert("duplication", serde_json::json!({"error": msg}));
            }
            Err(e) => {
                all_pass = false;
                results.insert("duplication", serde_json::json!({"error": e.to_string()}));
            }
        }
        
        // Security scan
        match self.security_scan().await {
            Ok(ToolResult::Success(data)) => {
                if !data["passes"].as_bool().unwrap_or(false) {
                    all_pass = false;
                }
                results.insert("security", data);
            }
            Ok(ToolResult::Error(msg)) => {
                all_pass = false;
                results.insert("security", serde_json::json!({"error": msg}));
            }
            Err(e) => {
                all_pass = false;
                results.insert("security", serde_json::json!({"error": e.to_string()}));
            }
        }
        
        let overall_message = if all_pass {
            "✅ All quality gates PASSED - Code is production ready!"
        } else if self.enforcement_mode == EnforcementMode::Strict {
            "❌ Quality gates FAILED - Blocking deployment"
        } else {
            "⚠️ Quality gates have warnings - Review before deployment"
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "all_pass": all_pass,
            "enforcement_mode": format!("{:?}", self.enforcement_mode),
            "results": results,
            "message": overall_message,
        })))
    }
}

#[async_trait]
impl ToolHandler for QualityGateAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "check_test_coverage" => {
                self.check_test_coverage().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check coverage: {}", e))
                })
            }
            "detect_fake_implementations" => {
                self.detect_fake_implementations().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to detect fakes: {}", e))
                })
            }
            "check_duplication" => {
                self.check_duplication().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check duplication: {}", e))
                })
            }
            "security_scan" => {
                self.security_scan().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to run security scan: {}", e))
                })
            }
            "run_quality_gate" => {
                self.run_quality_gate().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to run quality gate: {}", e))
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
    
    info!("Starting Bot4 QualityGate Agent v1.0 - Production Ready");
    info!("Enforcement: 100% coverage, 0% duplication, no fakes allowed");
    
    // Create agent
    let agent = Arc::new(QualityGateAgent::new().await?);
    
    // Start HTTP server for health checks
    let agent_clone = agent.clone();
    tokio::spawn(async move {
        let app = Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics));
        
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8085));
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    
    // Define MCP tools
    let tools = vec![
        Tool {
            name: "check_test_coverage".to_string(),
            description: "Check test coverage (must be 100%)".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        Tool {
            name: "detect_fake_implementations".to_string(),
            description: "Detect fake implementations and placeholders".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        Tool {
            name: "check_duplication".to_string(),
            description: "Check for code duplication".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        Tool {
            name: "security_scan".to_string(),
            description: "Run security vulnerability scan".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        Tool {
            name: "run_quality_gate".to_string(),
            description: "Run all quality gate checks".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("qualitygate-agent", "1.0.0")
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