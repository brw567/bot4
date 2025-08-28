//! Layer architecture auditor for Bot4
//! Ensures strict 7-layer architecture compliance

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tokio::fs;
use toml::Value;
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerAuditResult {
    pub violations: Vec<LayerViolation>,
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub layer_completion: HashMap<usize, f64>,
    pub critical_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerViolation {
    pub source_component: String,
    pub source_layer: usize,
    pub target_component: String,
    pub target_layer: usize,
    pub violation_type: String,
    pub severity: String,
}

pub struct LayerAuditor {
    workspace_path: PathBuf,
    layer_mapping: HashMap<String, usize>,
}

impl LayerAuditor {
    pub fn new(workspace_path: PathBuf) -> Self {
        let mut layer_mapping = HashMap::new();
        
        // Bot4's strict 7-layer architecture
        // Layer 0: Safety Systems (ABSOLUTE BLOCKER)
        layer_mapping.insert("safety".to_string(), 0);
        layer_mapping.insert("circuit_breaker".to_string(), 0);
        layer_mapping.insert("kill_switch".to_string(), 0);
        layer_mapping.insert("audit".to_string(), 0);
        layer_mapping.insert("control_modes".to_string(), 0);
        
        // Layer 1: Data Foundation
        layer_mapping.insert("data_ingestion".to_string(), 1);
        layer_mapping.insert("market_data".to_string(), 1);
        layer_mapping.insert("feature_store".to_string(), 1);
        layer_mapping.insert("timescaledb".to_string(), 1);
        layer_mapping.insert("data_pipeline".to_string(), 1);
        
        // Layer 2: Risk Management
        layer_mapping.insert("risk_engine".to_string(), 2);
        layer_mapping.insert("risk".to_string(), 2);
        layer_mapping.insert("position_manager".to_string(), 2);
        layer_mapping.insert("portfolio".to_string(), 2);
        layer_mapping.insert("kelly_criterion".to_string(), 2);
        
        // Layer 3: ML Pipeline
        layer_mapping.insert("ml_pipeline".to_string(), 3);
        layer_mapping.insert("ml".to_string(), 3);
        layer_mapping.insert("prediction".to_string(), 3);
        layer_mapping.insert("training".to_string(), 3);
        layer_mapping.insert("reinforcement_learning".to_string(), 3);
        layer_mapping.insert("gnn".to_string(), 3);
        
        // Layer 4: Trading Strategies
        layer_mapping.insert("strategies".to_string(), 4);
        layer_mapping.insert("signals".to_string(), 4);
        layer_mapping.insert("indicators".to_string(), 4);
        layer_mapping.insert("market_making".to_string(), 4);
        layer_mapping.insert("arbitrage".to_string(), 4);
        
        // Layer 5: Execution Engine
        layer_mapping.insert("trading_engine".to_string(), 5);
        layer_mapping.insert("execution".to_string(), 5);
        layer_mapping.insert("order_manager".to_string(), 5);
        layer_mapping.insert("smart_router".to_string(), 5);
        layer_mapping.insert("partial_fill".to_string(), 5);
        
        // Layer 6: Infrastructure
        layer_mapping.insert("infrastructure".to_string(), 6);
        layer_mapping.insert("monitoring".to_string(), 6);
        layer_mapping.insert("exchanges".to_string(), 6);
        layer_mapping.insert("websocket".to_string(), 6);
        layer_mapping.insert("redis_bridge".to_string(), 6);
        
        // Layer 7: Integration & Testing
        layer_mapping.insert("integration".to_string(), 7);
        layer_mapping.insert("testing".to_string(), 7);
        layer_mapping.insert("paper_trading".to_string(), 7);
        layer_mapping.insert("backtesting".to_string(), 7);
        
        // Shared/Common (allowed at any layer)
        layer_mapping.insert("domain_types".to_string(), 0);
        layer_mapping.insert("mathematical_ops".to_string(), 0);
        layer_mapping.insert("abstractions".to_string(), 0);
        layer_mapping.insert("common".to_string(), 0);
        
        Self {
            workspace_path,
            layer_mapping,
        }
    }
    
    pub async fn audit_all_layers(&self) -> Result<LayerAuditResult> {
        info!("Starting comprehensive layer architecture audit");
        
        let mut result = LayerAuditResult {
            violations: Vec::new(),
            dependency_graph: HashMap::new(),
            layer_completion: HashMap::new(),
            critical_issues: Vec::new(),
            recommendations: Vec::new(),
        };
        
        // Build dependency graph
        self.build_dependency_graph(&mut result).await?;
        
        // Check for layer violations
        self.check_layer_violations(&mut result)?;
        
        // Calculate layer completion percentages
        self.calculate_layer_completion(&mut result).await?;
        
        // Identify critical issues
        self.identify_critical_issues(&mut result)?;
        
        // Generate recommendations
        self.generate_recommendations(&mut result);
        
        Ok(result)
    }
    
    async fn build_dependency_graph(&self, result: &mut LayerAuditResult) -> Result<()> {
        let crates_paths = [
            self.workspace_path.join("rust_core/crates"),
            self.workspace_path.join("crates"),
        ];
        
        for crates_path in &crates_paths {
            if !crates_path.exists() {
                continue;
            }
            
            for entry in fs::read_dir(crates_path).await? {
                let entry = entry?;
                if entry.file_type().await?.is_dir() {
                    let component = entry.file_name().to_str().unwrap_or("").to_string();
                    let cargo_path = entry.path().join("Cargo.toml");
                    
                    if cargo_path.exists() {
                        let deps = self.parse_dependencies(&cargo_path).await?;
                        result.dependency_graph.insert(component.clone(), deps);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn parse_dependencies(&self, cargo_path: &PathBuf) -> Result<Vec<String>> {
        let content = fs::read_to_string(cargo_path).await?;
        let cargo_toml: Value = toml::from_str(&content)?;
        
        let mut dependencies = Vec::new();
        
        if let Some(deps) = cargo_toml.get("dependencies") {
            if let Some(deps_table) = deps.as_table() {
                for (name, _) in deps_table {
                    // Only track internal dependencies
                    if self.layer_mapping.contains_key(name) || 
                       name.starts_with("bot4") ||
                       result.dependency_graph.contains_key(name) {
                        dependencies.push(name.clone());
                    }
                }
            }
        }
        
        Ok(dependencies)
    }
    
    fn check_layer_violations(&self, result: &mut LayerAuditResult) -> Result<()> {
        for (source, deps) in &result.dependency_graph {
            let source_layer = self.get_component_layer(source);
            
            for dep in deps {
                let target_layer = self.get_component_layer(dep);
                
                // Check layer rules:
                // 1. Layer N can only depend on layers 0 to N-1
                // 2. Layer 0 (Safety) should have minimal dependencies
                // 3. Shared/common layers (0) can be used by anyone
                
                if target_layer > source_layer {
                    // VIOLATION: Higher layer dependency
                    result.violations.push(LayerViolation {
                        source_component: source.clone(),
                        source_layer,
                        target_component: dep.clone(),
                        target_layer,
                        violation_type: "UPWARD_DEPENDENCY".to_string(),
                        severity: "CRITICAL".to_string(),
                    });
                    
                    error!("CRITICAL LAYER VIOLATION: {} (L{}) -> {} (L{})",
                          source, source_layer, dep, target_layer);
                }
                
                // Special rule: Safety layer should be independent
                if source_layer == 0 && !dep.contains("common") && 
                   !dep.contains("domain_types") && target_layer > 0 {
                    result.violations.push(LayerViolation {
                        source_component: source.clone(),
                        source_layer,
                        target_component: dep.clone(),
                        target_layer,
                        violation_type: "SAFETY_LAYER_DEPENDENCY".to_string(),
                        severity: "HIGH".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    async fn calculate_layer_completion(&self, result: &mut LayerAuditResult) -> Result<()> {
        // Define expected components per layer
        let layer_requirements = HashMap::from([
            (0, vec!["kill_switch", "circuit_breaker", "control_modes", "audit"]),
            (1, vec!["data_ingestion", "market_data", "feature_store", "timescaledb"]),
            (2, vec!["risk_engine", "position_manager", "portfolio", "kelly_criterion"]),
            (3, vec!["ml_pipeline", "reinforcement_learning", "gnn", "automl"]),
            (4, vec!["market_making", "arbitrage", "signals", "indicators"]),
            (5, vec!["trading_engine", "order_manager", "smart_router", "partial_fill"]),
            (6, vec!["infrastructure", "monitoring", "exchanges", "websocket"]),
            (7, vec!["integration", "paper_trading", "backtesting", "testing"]),
        ]);
        
        for (layer, required_components) in layer_requirements {
            let mut completed = 0;
            let total = required_components.len();
            
            for component in required_components {
                // Check if component exists
                let paths = [
                    self.workspace_path.join("rust_core/crates").join(component),
                    self.workspace_path.join("crates").join(component),
                ];
                
                for path in &paths {
                    if path.exists() {
                        // Check if it has actual implementation (not just directory)
                        let src_path = path.join("src/lib.rs");
                        let main_path = path.join("src/main.rs");
                        
                        if src_path.exists() || main_path.exists() {
                            if let Ok(content) = fs::read_to_string(
                                if src_path.exists() { &src_path } else { &main_path }
                            ).await {
                                // Check it's not empty or placeholder
                                if content.len() > 100 && !content.contains("todo!()") {
                                    completed += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            
            let completion_percentage = (completed as f64 / total as f64) * 100.0;
            result.layer_completion.insert(layer, completion_percentage);
            
            info!("Layer {} completion: {:.1}% ({}/{})", 
                  layer, completion_percentage, completed, total);
        }
        
        Ok(())
    }
    
    fn identify_critical_issues(&self, result: &mut LayerAuditResult) -> Result<()> {
        // Check Layer 0 (Safety) completion - ABSOLUTE BLOCKER
        if let Some(&layer0_completion) = result.layer_completion.get(&0) {
            if layer0_completion < 100.0 {
                result.critical_issues.push(format!(
                    "🚨 CRITICAL: Layer 0 (Safety Systems) only {:.0}% complete. BLOCKS ALL WORK!",
                    layer0_completion
                ));
            }
        }
        
        // Check for upward dependencies
        let upward_violations = result.violations.iter()
            .filter(|v| v.violation_type == "UPWARD_DEPENDENCY")
            .count();
        
        if upward_violations > 0 {
            result.critical_issues.push(format!(
                "🚨 CRITICAL: {} layer architecture violations found. Must refactor!",
                upward_violations
            ));
        }
        
        // Check for missing critical components
        let critical_components = [
            ("kill_switch", "Hardware Kill Switch"),
            ("circuit_breaker", "Circuit Breaker System"),
            ("risk_engine", "Risk Management Engine"),
            ("feature_store", "Feature Store"),
        ];
        
        for (component, name) in &critical_components {
            if !result.dependency_graph.contains_key(*component) {
                result.critical_issues.push(format!(
                    "⚠️ MISSING: {} ({}) not found in codebase",
                    name, component
                ));
            }
        }
        
        // Check layer dependencies are sequential
        for layer in 1..=7 {
            if let Some(&current_completion) = result.layer_completion.get(&layer) {
                if let Some(&prev_completion) = result.layer_completion.get(&(layer - 1)) {
                    if current_completion > prev_completion && prev_completion < 100.0 {
                        result.critical_issues.push(format!(
                            "⚠️ Layer {} ({:.0}%) more complete than Layer {} ({:.0}%) - violates sequential development",
                            layer, current_completion, layer - 1, prev_completion
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn generate_recommendations(&self, result: &mut LayerAuditResult) {
        // Priority recommendations based on audit results
        
        // Layer 0 is top priority
        if let Some(&layer0) = result.layer_completion.get(&0) {
            if layer0 < 100.0 {
                result.recommendations.push(
                    "1. IMMEDIATE: Complete Layer 0 (Safety Systems) - NOTHING can proceed without this!".to_string()
                );
            }
        }
        
        // Fix violations
        if !result.violations.is_empty() {
            result.recommendations.push(format!(
                "2. Fix {} layer violations using dependency inversion or abstraction layers",
                result.violations.len()
            ));
        }
        
        // Complete layers in order
        for layer in 0..=7 {
            if let Some(&completion) = result.layer_completion.get(&layer) {
                if completion < 100.0 {
                    result.recommendations.push(format!(
                        "{}. Complete Layer {} ({:.0}% done) before starting Layer {}",
                        result.recommendations.len() + 1,
                        layer,
                        completion,
                        layer + 1
                    ));
                    break; // Only recommend the next incomplete layer
                }
            }
        }
        
        // Team focus recommendation
        result.recommendations.push(
            "Remember: ENTIRE TEAM must work on ONE TASK at a time with 360° analysis".to_string()
        );
    }
    
    pub async fn check_recent_changes(&self) -> Result<Vec<String>> {
        // Check git diff for layer violations in recent changes
        let output = std::process::Command::new("git")
            .args(&["diff", "--name-only", "HEAD~1"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        let mut violations = Vec::new();
        
        if output.status.success() {
            let files = String::from_utf8_lossy(&output.stdout);
            
            for file in files.lines() {
                if file.ends_with(".rs") {
                    // Extract component from path
                    if let Some(component) = self.extract_component_from_path(file) {
                        // Check if changes violate layer rules
                        // This is simplified - real implementation would parse the actual changes
                        violations.push(format!("Check {} for layer compliance", component));
                    }
                }
            }
        }
        
        Ok(violations)
    }
    
    pub async fn is_layer_complete(&self, layer: usize) -> Result<bool> {
        let mut audit_result = LayerAuditResult {
            violations: Vec::new(),
            dependency_graph: HashMap::new(),
            layer_completion: HashMap::new(),
            critical_issues: Vec::new(),
            recommendations: Vec::new(),
        };
        
        self.calculate_layer_completion(&mut audit_result).await?;
        
        Ok(audit_result.layer_completion.get(&layer)
            .map(|&completion| completion >= 100.0)
            .unwrap_or(false))
    }
    
    fn get_component_layer(&self, component: &str) -> usize {
        // Check exact match first
        if let Some(&layer) = self.layer_mapping.get(component) {
            return layer;
        }
        
        // Check if component name contains layer keyword
        for (keyword, &layer) in &self.layer_mapping {
            if component.contains(keyword) {
                return layer;
            }
        }
        
        // Default to infrastructure layer if unknown
        6
    }
    
    fn extract_component_from_path(&self, path: &str) -> Option<String> {
        // Extract component name from file path
        // e.g., "rust_core/crates/trading_engine/src/main.rs" -> "trading_engine"
        let parts: Vec<&str> = path.split('/').collect();
        
        for (i, part) in parts.iter().enumerate() {
            if *part == "crates" && i + 1 < parts.len() {
                return Some(parts[i + 1].to_string());
            }
        }
        
        None
    }
}