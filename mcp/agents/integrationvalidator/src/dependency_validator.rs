//! Dependency validation module for Bot4
//! Ensures proper layer dependencies and no circular dependencies

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use tokio::fs;
use toml::Value;
use tracing::{info, debug, warn, error};

#[derive(Debug, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub path: Option<String>,
    pub features: Vec<String>,
}

#[derive(Debug)]
pub struct DependencyGraph {
    edges: HashMap<String, HashSet<String>>,
    layers: HashMap<String, usize>,
}

pub struct DependencyValidator {
    workspace_path: PathBuf,
    layer_mapping: HashMap<String, usize>,
}

impl DependencyValidator {
    pub fn new(workspace_path: PathBuf) -> Self {
        let mut layer_mapping = HashMap::new();
        
        // Bot4 7-layer architecture mapping
        // Layer 0: Safety Systems
        layer_mapping.insert("safety".to_string(), 0);
        layer_mapping.insert("circuit_breaker".to_string(), 0);
        layer_mapping.insert("kill_switch".to_string(), 0);
        layer_mapping.insert("audit".to_string(), 0);
        
        // Layer 1: Data Foundation
        layer_mapping.insert("data_ingestion".to_string(), 1);
        layer_mapping.insert("market_data".to_string(), 1);
        layer_mapping.insert("feature_store".to_string(), 1);
        layer_mapping.insert("timescaledb".to_string(), 1);
        
        // Layer 2: Risk Management
        layer_mapping.insert("risk_engine".to_string(), 2);
        layer_mapping.insert("risk".to_string(), 2);
        layer_mapping.insert("position_manager".to_string(), 2);
        layer_mapping.insert("portfolio".to_string(), 2);
        
        // Layer 3: ML Pipeline
        layer_mapping.insert("ml_pipeline".to_string(), 3);
        layer_mapping.insert("ml".to_string(), 3);
        layer_mapping.insert("prediction".to_string(), 3);
        layer_mapping.insert("training".to_string(), 3);
        
        // Layer 4: Trading Strategies
        layer_mapping.insert("strategies".to_string(), 4);
        layer_mapping.insert("signals".to_string(), 4);
        layer_mapping.insert("indicators".to_string(), 4);
        layer_mapping.insert("market_making".to_string(), 4);
        
        // Layer 5: Execution Engine
        layer_mapping.insert("trading_engine".to_string(), 5);
        layer_mapping.insert("execution".to_string(), 5);
        layer_mapping.insert("order_manager".to_string(), 5);
        layer_mapping.insert("smart_router".to_string(), 5);
        
        // Layer 6: Infrastructure
        layer_mapping.insert("infrastructure".to_string(), 6);
        layer_mapping.insert("monitoring".to_string(), 6);
        layer_mapping.insert("exchanges".to_string(), 6);
        layer_mapping.insert("websocket".to_string(), 6);
        
        // Layer 7: Integration & Testing
        layer_mapping.insert("integration".to_string(), 7);
        layer_mapping.insert("testing".to_string(), 7);
        layer_mapping.insert("paper_trading".to_string(), 7);
        
        // Common/Shared (allowed at any layer)
        layer_mapping.insert("domain_types".to_string(), 0);
        layer_mapping.insert("mathematical_ops".to_string(), 0);
        layer_mapping.insert("abstractions".to_string(), 0);
        
        Self {
            workspace_path,
            layer_mapping,
        }
    }
    
    pub async fn get_component_dependencies(&self, component: &str) -> Result<Vec<String>> {
        let cargo_path = self.workspace_path
            .join("crates")
            .join(component)
            .join("Cargo.toml");
        
        if !cargo_path.exists() {
            // Try rust_core path
            let alt_path = self.workspace_path
                .join("rust_core/crates")
                .join(component)
                .join("Cargo.toml");
            
            if alt_path.exists() {
                return self.parse_cargo_dependencies(&alt_path).await;
            }
            
            return Ok(Vec::new());
        }
        
        self.parse_cargo_dependencies(&cargo_path).await
    }
    
    async fn parse_cargo_dependencies(&self, cargo_path: &PathBuf) -> Result<Vec<String>> {
        let content = fs::read_to_string(cargo_path).await?;
        let cargo_toml: Value = toml::from_str(&content)?;
        
        let mut dependencies = Vec::new();
        
        // Parse regular dependencies
        if let Some(deps) = cargo_toml.get("dependencies") {
            if let Some(deps_table) = deps.as_table() {
                for (name, _) in deps_table {
                    dependencies.push(name.clone());
                }
            }
        }
        
        // Parse dev dependencies
        if let Some(deps) = cargo_toml.get("dev-dependencies") {
            if let Some(deps_table) = deps.as_table() {
                for (name, _) in deps_table {
                    if name.starts_with("bot4") || self.layer_mapping.contains_key(name) {
                        dependencies.push(name.clone());
                    }
                }
            }
        }
        
        Ok(dependencies)
    }
    
    pub async fn has_circular_dependency(&self, component: &str, dependency: &str) -> Result<bool> {
        // Build dependency graph
        let graph = self.build_dependency_graph().await?;
        
        // Check if dependency also depends on component (circular)
        if let Some(dep_deps) = graph.edges.get(dependency) {
            if dep_deps.contains(component) {
                error!("Circular dependency detected: {} <-> {}", component, dependency);
                return Ok(true);
            }
            
            // Check transitive circular dependencies
            let mut visited = HashSet::new();
            if self.has_path_to(&graph, dependency, component, &mut visited) {
                error!("Transitive circular dependency: {} -> ... -> {}", component, dependency);
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    pub async fn check_version_conflicts(&self, component: &str) -> Result<Vec<String>> {
        let mut conflicts = Vec::new();
        let mut version_map: HashMap<String, HashSet<String>> = HashMap::new();
        
        // Get all dependencies and their versions
        let cargo_path = self.workspace_path
            .join("crates")
            .join(component)
            .join("Cargo.toml");
        
        if cargo_path.exists() {
            let content = fs::read_to_string(&cargo_path).await?;
            let cargo_toml: Value = toml::from_str(&content)?;
            
            if let Some(deps) = cargo_toml.get("dependencies") {
                if let Some(deps_table) = deps.as_table() {
                    for (name, spec) in deps_table {
                        let version = self.extract_version(spec);
                        version_map.entry(name.clone())
                            .or_insert_with(HashSet::new)
                            .insert(version);
                    }
                }
            }
        }
        
        // Check workspace-wide for conflicts
        let workspace_cargo = self.workspace_path.join("Cargo.toml");
        if workspace_cargo.exists() {
            let content = fs::read_to_string(&workspace_cargo).await?;
            let cargo_toml: Value = toml::from_str(&content)?;
            
            if let Some(workspace) = cargo_toml.get("workspace") {
                if let Some(deps) = workspace.get("dependencies") {
                    if let Some(deps_table) = deps.as_table() {
                        for (name, spec) in deps_table {
                            let workspace_version = self.extract_version(spec);
                            
                            if let Some(component_versions) = version_map.get(name) {
                                if !component_versions.contains(&workspace_version) {
                                    conflicts.push(format!(
                                        "{}: component wants {:?}, workspace has {}",
                                        name, component_versions, workspace_version
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(conflicts)
    }
    
    pub async fn check_missing_dependencies(&self, component: &str) -> Result<Vec<String>> {
        let mut missing = Vec::new();
        let source_path = self.workspace_path
            .join("crates")
            .join(component)
            .join("src");
        
        if !source_path.exists() {
            return Ok(missing);
        }
        
        // Parse source files for use statements
        let used_crates = self.extract_used_crates(&source_path).await?;
        
        // Get declared dependencies
        let declared = self.get_component_dependencies(component).await?;
        let declared_set: HashSet<_> = declared.iter().cloned().collect();
        
        // Find used but not declared
        for used_crate in used_crates {
            if !declared_set.contains(&used_crate) && 
               !self.is_std_crate(&used_crate) &&
               used_crate != component {
                missing.push(used_crate);
            }
        }
        
        Ok(missing)
    }
    
    pub async fn validate_layer_dependency(&self, source: &str, target: &str) -> Result<bool> {
        self.is_valid_layer_dependency(source, target).await
    }
    
    pub async fn is_valid_layer_dependency(&self, source: &str, target: &str) -> Result<bool> {
        let source_layer = self.get_component_layer(source);
        let target_layer = self.get_component_layer(target);
        
        // A component can only depend on lower layers (or same layer for some cases)
        // Layer N can depend on layers 0 to N-1
        if source_layer < target_layer {
            warn!("Layer violation: {} (layer {}) cannot depend on {} (layer {})",
                  source, source_layer, target, target_layer);
            return Ok(false);
        }
        
        // Special rules for Bot4
        // Layer 0 (Safety) should have minimal dependencies
        if source_layer == 0 && target_layer > 0 {
            warn!("Safety layer should not depend on higher layers");
            return Ok(false);
        }
        
        // Layer 7 (Testing) can depend on anything
        if source_layer == 7 {
            return Ok(true);
        }
        
        Ok(true)
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
    
    async fn build_dependency_graph(&self) -> Result<DependencyGraph> {
        let mut graph = DependencyGraph {
            edges: HashMap::new(),
            layers: HashMap::new(),
        };
        
        // Walk through all crates
        let crates_path = self.workspace_path.join("crates");
        if crates_path.exists() {
            for entry in fs::read_dir(&crates_path).await? {
                let entry = entry?;
                if entry.file_type().await?.is_dir() {
                    let component = entry.file_name().to_str().unwrap_or("").to_string();
                    let deps = self.get_component_dependencies(&component).await?;
                    
                    graph.edges.insert(component.clone(), deps.into_iter().collect());
                    graph.layers.insert(component.clone(), self.get_component_layer(&component));
                }
            }
        }
        
        Ok(graph)
    }
    
    fn has_path_to(
        &self,
        graph: &DependencyGraph,
        from: &str,
        to: &str,
        visited: &mut HashSet<String>
    ) -> bool {
        if from == to {
            return true;
        }
        
        if visited.contains(from) {
            return false;
        }
        
        visited.insert(from.to_string());
        
        if let Some(neighbors) = graph.edges.get(from) {
            for neighbor in neighbors {
                if self.has_path_to(graph, neighbor, to, visited) {
                    return true;
                }
            }
        }
        
        false
    }
    
    fn extract_version(&self, spec: &Value) -> String {
        if let Some(version_str) = spec.as_str() {
            version_str.to_string()
        } else if let Some(table) = spec.as_table() {
            table.get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("*")
                .to_string()
        } else {
            "*".to_string()
        }
    }
    
    async fn extract_used_crates(&self, source_path: &PathBuf) -> Result<HashSet<String>> {
        let mut used_crates = HashSet::new();
        
        for entry in walkdir::WalkDir::new(source_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            // Extract crate names from use statements
            for line in content.lines() {
                if line.trim_start().starts_with("use ") {
                    if let Some(crate_name) = self.extract_crate_from_use(line) {
                        if !self.is_std_crate(&crate_name) {
                            used_crates.insert(crate_name);
                        }
                    }
                }
                
                // Also check extern crate statements
                if line.trim_start().starts_with("extern crate ") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let crate_name = parts[2].trim_end_matches(';');
                        used_crates.insert(crate_name.to_string());
                    }
                }
            }
        }
        
        Ok(used_crates)
    }
    
    fn extract_crate_from_use(&self, line: &str) -> Option<String> {
        // Extract crate name from use statement
        let line = line.trim_start_matches("use ").trim_end_matches(';');
        
        // Handle various use patterns
        if let Some(pos) = line.find("::") {
            let crate_name = &line[..pos];
            
            // Skip self, super, crate
            if !["self", "super", "crate"].contains(&crate_name) {
                return Some(crate_name.to_string());
            }
        }
        
        None
    }
    
    fn is_std_crate(&self, name: &str) -> bool {
        // List of standard library crates
        const STD_CRATES: &[&str] = &[
            "std", "core", "alloc", "collections", "fmt", "io", "fs", "path",
            "vec", "string", "str", "slice", "option", "result", "iter",
            "ops", "cmp", "mem", "ptr", "sync", "thread", "time", "process",
            "env", "net", "os", "ffi", "panic", "error", "convert", "borrow",
            "clone", "hash", "default", "marker", "any", "cell", "rc", "pin",
            "task", "future", "stream", "sink", "async", "await", "tokio",
            "futures", "async_trait", "serde", "serde_json", "anyhow", "thiserror",
            "tracing", "log", "env_logger"
        ];
        
        STD_CRATES.contains(&name)
    }
}