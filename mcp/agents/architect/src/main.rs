//! Bot4 Architect Agent
//! System design, deduplication prevention, and layer enforcement

use anyhow::Result;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use rmcp::{
    server::{Server, ServerBuilder, ToolHandler},
    transport::DockerTransport,
    types::{Tool, ToolCall, ToolResult, Resource, Prompt},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use syn::{parse_file, Item, ItemStruct, ItemFn};
use tokio::fs;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use walkdir::WalkDir;
use regex::Regex;

mod duplication;
mod layers;
mod analysis;

use duplication::DuplicationDetector;
use layers::LayerEnforcer;
use analysis::CodeAnalyzer;

/// Architect agent implementation
struct ArchitectAgent {
    redis: ConnectionManager,
    duplication_detector: DuplicationDetector,
    layer_enforcer: LayerEnforcer,
    code_analyzer: CodeAnalyzer,
    workspace_path: PathBuf,
}

impl ArchitectAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Set workspace path
        let workspace_path = PathBuf::from(
            std::env::var("WORKSPACE_PATH").unwrap_or_else(|_| "/workspace/rust_core".to_string())
        );
        
        Ok(Self {
            redis,
            duplication_detector: DuplicationDetector::new(),
            layer_enforcer: LayerEnforcer::new(),
            code_analyzer: CodeAnalyzer::new(),
            workspace_path,
        })
    }
    
    /// Check for duplicate implementations
    async fn check_duplicates(&self, component: String, component_type: String) -> Result<ToolResult> {
        info!("Checking duplicates for {} (type: {})", component, component_type);
        
        let mut duplicates = Vec::new();
        let mut files_checked = 0;
        
        // Walk through Rust files
        for entry in WalkDir::new(&self.workspace_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            files_checked += 1;
            let path = entry.path();
            
            // Read and parse file
            if let Ok(content) = fs::read_to_string(path).await {
                if let Ok(syntax_tree) = parse_file(&content) {
                    // Check for duplicate structs
                    if component_type == "struct" || component_type == "any" {
                        for item in syntax_tree.items.iter() {
                            if let Item::Struct(item_struct) = item {
                                if item_struct.ident.to_string() == component {
                                    duplicates.push(format!("{}:{}", path.display(), item_struct.ident.span().start().line));
                                }
                            }
                        }
                    }
                    
                    // Check for duplicate functions
                    if component_type == "function" || component_type == "any" {
                        for item in syntax_tree.items.iter() {
                            if let Item::Fn(item_fn) = item {
                                if item_fn.sig.ident.to_string() == component {
                                    duplicates.push(format!("{}:{}", path.display(), item_fn.sig.ident.span().start().line));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        debug!("Checked {} files", files_checked);
        
        // Update duplication registry
        self.duplication_detector.update_registry(&component, &duplicates).await?;
        
        // Generate result
        let result = if duplicates.len() > 1 {
            format!(
                "❌ DUPLICATES FOUND: {} instances of '{}'\n\nLocations:\n{}\n\nAction: Refactor to use existing implementation from domain_types or mathematical_ops crates.",
                duplicates.len(),
                component,
                duplicates.join("\n")
            )
        } else {
            format!("✅ No duplicates found for '{}'. Safe to proceed.", component)
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "component": component,
            "duplicates_found": duplicates.len() > 1,
            "count": duplicates.len(),
            "locations": duplicates,
            "message": result
        })))
    }
    
    /// Check for layer violations
    async fn check_layer_violation(&self, source_layer: String, target_layer: String) -> Result<ToolResult> {
        info!("Checking layer violation: {} -> {}", source_layer, target_layer);
        
        let violation = self.layer_enforcer.check_violation(&source_layer, &target_layer)?;
        
        let result = if violation {
            format!(
                "❌ LAYER VIOLATION: {} cannot import from {}\n\nRule: Layer N can only import from layers 0 to N-1\nSolution: Use dependency inversion or abstractions crate",
                source_layer, target_layer
            )
        } else {
            format!("✅ Valid dependency: {} -> {}", source_layer, target_layer)
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "source_layer": source_layer,
            "target_layer": target_layer,
            "violation": violation,
            "message": result
        })))
    }
    
    /// Decompose task into subtasks
    async fn decompose_task(&self, task_id: String, description: String) -> Result<ToolResult> {
        info!("Decomposing task {}: {}", task_id, description);
        
        let mut subtasks = Vec::new();
        
        // Analyze task type and generate appropriate subtasks
        if description.contains("consolidat") || description.contains("deduplic") {
            subtasks.extend_from_slice(&[
                format!("{}.1: Identify all duplicate implementations", task_id),
                format!("{}.2: Create canonical implementation", task_id),
                format!("{}.3: Update all imports to use canonical version", task_id),
                format!("{}.4: Remove duplicate implementations", task_id),
                format!("{}.5: Verify no compilation errors", task_id),
                format!("{}.6: Update documentation", task_id),
            ]);
        } else if description.contains("layer") || description.contains("architect") {
            subtasks.extend_from_slice(&[
                format!("{}.1: Identify layer violations", task_id),
                format!("{}.2: Create abstraction interfaces", task_id),
                format!("{}.3: Implement dependency inversion", task_id),
                format!("{}.4: Update imports", task_id),
                format!("{}.5: Verify layer integrity", task_id),
            ]);
        } else {
            // Generic decomposition
            subtasks.extend_from_slice(&[
                format!("{}.1: Analyze requirements", task_id),
                format!("{}.2: Design solution", task_id),
                format!("{}.3: Implement core functionality", task_id),
                format!("{}.4: Add tests", task_id),
                format!("{}.5: Document changes", task_id),
            ]);
        }
        
        Ok(ToolResult::Success(serde_json::json!({
            "task_id": task_id,
            "description": description,
            "subtasks": subtasks,
            "count": subtasks.len()
        })))
    }
    
    /// Analyze code for patterns and issues
    async fn analyze_code(&self, file_path: String) -> Result<ToolResult> {
        info!("Analyzing code: {}", file_path);
        
        let path = self.workspace_path.join(&file_path);
        let content = fs::read_to_string(&path).await?;
        
        let analysis = self.code_analyzer.analyze(&content)?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "file": file_path,
            "analysis": analysis
        })))
    }
}

#[async_trait]
impl ToolHandler for ArchitectAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "check_duplicates" => {
                let component = tool_call.arguments["component"].as_str().unwrap_or("").to_string();
                let component_type = tool_call.arguments["type"].as_str().unwrap_or("any").to_string();
                self.check_duplicates(component, component_type).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check duplicates: {}", e))
                })
            }
            "check_layer_violation" => {
                let source = tool_call.arguments["source_layer"].as_str().unwrap_or("").to_string();
                let target = tool_call.arguments["target_layer"].as_str().unwrap_or("").to_string();
                self.check_layer_violation(source, target).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check layer violation: {}", e))
                })
            }
            "decompose_task" => {
                let task_id = tool_call.arguments["task_id"].as_str().unwrap_or("").to_string();
                let description = tool_call.arguments["description"].as_str().unwrap_or("").to_string();
                self.decompose_task(task_id, description).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to decompose task: {}", e))
                })
            }
            "analyze_code" => {
                let file_path = tool_call.arguments["file_path"].as_str().unwrap_or("").to_string();
                self.analyze_code(file_path).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to analyze code: {}", e))
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
    
    info!("Starting Bot4 Architect Agent v1.0");
    
    // Create agent
    let agent = ArchitectAgent::new().await?;
    
    // Define tools
    let tools = vec![
        Tool {
            name: "check_duplicates".to_string(),
            description: "Check for duplicate implementations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "type": {"type": "string", "enum": ["struct", "function", "trait", "any"]}
                },
                "required": ["component"]
            }),
        },
        Tool {
            name: "check_layer_violation".to_string(),
            description: "Check for layer architecture violations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_layer": {"type": "string"},
                    "target_layer": {"type": "string"}
                },
                "required": ["source_layer", "target_layer"]
            }),
        },
        Tool {
            name: "decompose_task".to_string(),
            description: "Decompose a task into subtasks".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["task_id", "description"]
            }),
        },
        Tool {
            name: "analyze_code".to_string(),
            description: "Analyze code for patterns and issues".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("architect-agent", "1.0.0")
        .with_tools(tools)
        .with_tool_handler(agent)
        .build()?;
    
    // Use Docker transport
    let transport = DockerTransport::new()?;
    server.run(transport).await?;
    
    Ok(())
}