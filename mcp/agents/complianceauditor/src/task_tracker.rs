//! Task tracking module for Bot4
//! Tracks task completion according to PROJECT_MANAGEMENT_MASTER.md

use anyhow::Result;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use chrono::{DateTime, Utc};
use tracing::{info, warn};

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskStatus {
    pub task_id: String,
    pub description: String,
    pub layer: usize,
    pub hours_allocated: f64,
    pub hours_spent: f64,
    pub status: String,
    pub assignees: Vec<String>,
    pub blockers: Vec<String>,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationStatus {
    pub code_complete: bool,
    pub tests_pass: bool,
    pub documentation_updated: bool,
    pub integration_verified: bool,
    pub reviewed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectProgress {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub in_progress_tasks: usize,
    pub blocked_tasks: usize,
    pub total_hours: f64,
    pub hours_completed: f64,
    pub overall_percentage: f64,
    pub layers: HashMap<usize, LayerProgress>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerProgress {
    pub layer: usize,
    pub name: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub percentage: f64,
    pub blockers: Vec<String>,
}

pub struct TaskTracker {
    workspace_path: PathBuf,
    redis: ConnectionManager,
}

impl TaskTracker {
    pub fn new(workspace_path: PathBuf, redis: ConnectionManager) -> Self {
        Self {
            workspace_path,
            redis,
        }
    }
    
    pub async fn get_task_status(&self, task_id: &str) -> Result<serde_json::Value> {
        info!("Getting status for task: {}", task_id);
        
        // First check Redis cache
        let cache_key = format!("task:status:{}", task_id);
        if let Ok(cached) = self.redis.clone().get::<_, String>(&cache_key).await {
            if let Ok(status) = serde_json::from_str(&cached) {
                return Ok(status);
            }
        }
        
        // Parse from PROJECT_MANAGEMENT_MASTER.md
        let task = self.parse_task_from_master(task_id).await?;
        
        // Cache the result
        let _: Result<(), redis::RedisError> = self.redis.clone()
            .set_ex(&cache_key, serde_json::to_string(&task)?, 3600).await;
        
        Ok(serde_json::to_value(task)?)
    }
    
    async fn parse_task_from_master(&self, task_id: &str) -> Result<TaskStatus> {
        let master_path = self.workspace_path.join("PROJECT_MANAGEMENT_MASTER.md");
        
        if !master_path.exists() {
            return Err(anyhow::anyhow!("PROJECT_MANAGEMENT_MASTER.md not found"));
        }
        
        let content = fs::read_to_string(&master_path).await?;
        
        // Parse task information
        let mut task = TaskStatus {
            task_id: task_id.to_string(),
            description: String::new(),
            layer: self.extract_layer_from_task_id(task_id),
            hours_allocated: 0.0,
            hours_spent: 0.0,
            status: "unknown".to_string(),
            assignees: Vec::new(),
            blockers: Vec::new(),
            validation_status: ValidationStatus {
                code_complete: false,
                tests_pass: false,
                documentation_updated: false,
                integration_verified: false,
                reviewed: false,
            },
        };
        
        // Find task in document
        for line in content.lines() {
            if line.contains(task_id) {
                // Extract task details from line
                task.description = self.extract_description(line);
                task.hours_allocated = self.extract_hours(line);
                task.status = self.extract_status(line);
                task.assignees = self.extract_assignees(line);
                
                // Check completion markers
                if line.contains("✅") || line.contains("COMPLETE") {
                    task.status = "completed".to_string();
                    task.validation_status.code_complete = true;
                } else if line.contains("🔄") || line.contains("IN PROGRESS") {
                    task.status = "in_progress".to_string();
                } else if line.contains("❌") || line.contains("BLOCKED") {
                    task.status = "blocked".to_string();
                }
                
                break;
            }
        }
        
        // Validate task completion
        task.validation_status = self.validate_task_completion_status(task_id).await?;
        
        Ok(task)
    }
    
    pub async fn validate_task_completion(&self, task_id: &str) -> Result<serde_json::Value> {
        info!("Validating completion for task: {}", task_id);
        
        let mut validation = ValidationStatus {
            code_complete: false,
            tests_pass: false,
            documentation_updated: false,
            integration_verified: false,
            reviewed: false,
        };
        
        // Check code implementation
        validation.code_complete = self.check_code_implementation(task_id).await?;
        
        // Check tests
        validation.tests_pass = self.check_tests_pass(task_id).await?;
        
        // Check documentation
        validation.documentation_updated = self.check_documentation(task_id).await?;
        
        // Check integration
        validation.integration_verified = self.check_integration(task_id).await?;
        
        // Check review status
        validation.reviewed = self.check_review_status(task_id).await?;
        
        // Calculate overall validity
        let all_valid = validation.code_complete &&
                       validation.tests_pass &&
                       validation.documentation_updated &&
                       validation.integration_verified &&
                       validation.reviewed;
        
        Ok(serde_json::json!({
            "task_id": task_id,
            "validation": validation,
            "can_mark_complete": all_valid,
            "missing": self.get_missing_items(&validation),
            "recommendation": if all_valid {
                "✅ Task fully validated and can be marked complete"
            } else {
                "❌ Task incomplete - see missing items"
            }
        }))
    }
    
    pub async fn get_overall_progress(&self) -> Result<serde_json::Value> {
        info!("Calculating overall project progress");
        
        let mut progress = ProjectProgress {
            total_tasks: 0,
            completed_tasks: 0,
            in_progress_tasks: 0,
            blocked_tasks: 0,
            total_hours: 3532.0, // From PROJECT_MANAGEMENT_MASTER.md
            hours_completed: 0.0,
            overall_percentage: 0.0,
            layers: HashMap::new(),
        };
        
        // Initialize layer progress
        for layer in 0..=7 {
            progress.layers.insert(layer, LayerProgress {
                layer,
                name: self.get_layer_name(layer),
                total_tasks: 0,
                completed_tasks: 0,
                percentage: 0.0,
                blockers: Vec::new(),
            });
        }
        
        // Parse all tasks from master document
        let tasks = self.parse_all_tasks().await?;
        
        for task in tasks {
            progress.total_tasks += 1;
            
            let layer = task.layer;
            if let Some(layer_progress) = progress.layers.get_mut(&layer) {
                layer_progress.total_tasks += 1;
            }
            
            match task.status.as_str() {
                "completed" => {
                    progress.completed_tasks += 1;
                    progress.hours_completed += task.hours_allocated;
                    if let Some(layer_progress) = progress.layers.get_mut(&layer) {
                        layer_progress.completed_tasks += 1;
                    }
                }
                "in_progress" => {
                    progress.in_progress_tasks += 1;
                    progress.hours_completed += task.hours_allocated * 0.5; // Assume 50% done
                }
                "blocked" => {
                    progress.blocked_tasks += 1;
                    if let Some(layer_progress) = progress.layers.get_mut(&layer) {
                        layer_progress.blockers.extend(task.blockers);
                    }
                }
                _ => {}
            }
        }
        
        // Calculate percentages
        progress.overall_percentage = (progress.hours_completed / progress.total_hours) * 100.0;
        
        for (_, layer_progress) in progress.layers.iter_mut() {
            if layer_progress.total_tasks > 0 {
                layer_progress.percentage = 
                    (layer_progress.completed_tasks as f64 / layer_progress.total_tasks as f64) * 100.0;
            }
        }
        
        // Add critical status
        let layer0_complete = progress.layers.get(&0)
            .map(|l| l.percentage >= 100.0)
            .unwrap_or(false);
        
        Ok(serde_json::json!({
            "progress": progress,
            "critical_status": if !layer0_complete {
                "🚨 BLOCKED: Layer 0 (Safety) incomplete - NO WORK can proceed!"
            } else if progress.blocked_tasks > 0 {
                format!("⚠️ {} tasks blocked", progress.blocked_tasks)
            } else {
                "✅ No critical blockers"
            },
            "next_priority": self.get_next_priority_task(&progress),
            "estimated_completion": self.estimate_completion(&progress),
        }))
    }
    
    pub async fn get_incomplete_tasks(&self) -> Result<Vec<String>> {
        let tasks = self.parse_all_tasks().await?;
        
        Ok(tasks.into_iter()
            .filter(|t| t.status != "completed")
            .map(|t| format!("{}: {}", t.task_id, t.description))
            .collect())
    }
    
    async fn parse_all_tasks(&self) -> Result<Vec<TaskStatus>> {
        let mut tasks = Vec::new();
        let master_path = self.workspace_path.join("PROJECT_MANAGEMENT_MASTER.md");
        
        if !master_path.exists() {
            return Ok(tasks);
        }
        
        let content = fs::read_to_string(&master_path).await?;
        
        // Parse tasks using regex or line parsing
        for line in content.lines() {
            if let Some(task_id) = self.extract_task_id(line) {
                if let Ok(task) = self.parse_task_from_master(&task_id).await {
                    tasks.push(task);
                }
            }
        }
        
        Ok(tasks)
    }
    
    async fn validate_task_completion_status(&self, task_id: &str) -> Result<ValidationStatus> {
        let mut status = ValidationStatus {
            code_complete: false,
            tests_pass: false,
            documentation_updated: false,
            integration_verified: false,
            reviewed: false,
        };
        
        // Check various aspects of task completion
        status.code_complete = self.check_code_implementation(task_id).await?;
        status.tests_pass = self.check_tests_pass(task_id).await?;
        status.documentation_updated = self.check_documentation(task_id).await?;
        status.integration_verified = self.check_integration(task_id).await?;
        status.reviewed = self.check_review_status(task_id).await?;
        
        Ok(status)
    }
    
    async fn check_code_implementation(&self, task_id: &str) -> Result<bool> {
        // Check if code for this task exists
        // This is simplified - real implementation would check specific components
        
        // Map task ID to expected component
        let component = self.task_id_to_component(task_id);
        let component_path = self.workspace_path.join("rust_core/crates").join(&component);
        
        if component_path.exists() {
            // Check for actual implementation (not empty)
            let src_path = component_path.join("src");
            if src_path.exists() {
                // Count .rs files with actual code
                let mut has_implementation = false;
                
                for entry in walkdir::WalkDir::new(&src_path)
                    .follow_links(true)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
                {
                    if let Ok(content) = fs::read_to_string(entry.path()).await {
                        // Check it's not placeholder
                        if content.len() > 100 && 
                           !content.contains("todo!()") &&
                           !content.contains("unimplemented!()") {
                            has_implementation = true;
                            break;
                        }
                    }
                }
                
                return Ok(has_implementation);
            }
        }
        
        Ok(false)
    }
    
    async fn check_tests_pass(&self, _task_id: &str) -> Result<bool> {
        // Run tests for the component
        let output = std::process::Command::new("cargo")
            .args(&["test", "--all"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        Ok(output.status.success())
    }
    
    async fn check_documentation(&self, _task_id: &str) -> Result<bool> {
        // Check if documentation has been updated recently
        let docs = [
            "PROJECT_MANAGEMENT_MASTER.md",
            "LLM_OPTIMIZED_ARCHITECTURE.md",
        ];
        
        for doc in &docs {
            let path = self.workspace_path.join(doc);
            if let Ok(metadata) = fs::metadata(&path).await {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(elapsed) = modified.elapsed() {
                        // If modified within last day, consider documented
                        if elapsed.as_secs() < 86400 {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        
        Ok(false)
    }
    
    async fn check_integration(&self, task_id: &str) -> Result<bool> {
        // Check integration status from verification script
        let output = std::process::Command::new("./scripts/verify_integration.sh")
            .arg("--check")
            .arg(task_id)
            .current_dir(&self.workspace_path)
            .output();
        
        match output {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false), // Script doesn't exist or failed
        }
    }
    
    async fn check_review_status(&self, task_id: &str) -> Result<bool> {
        // Check Redis for review status
        let review_key = format!("review:approved:{}", task_id);
        let approved: bool = self.redis.clone()
            .get(&review_key)
            .await
            .unwrap_or(false);
        
        Ok(approved)
    }
    
    fn extract_layer_from_task_id(&self, task_id: &str) -> usize {
        // Task IDs are formatted as "Layer.Task" (e.g., "0.1", "1.2")
        task_id.split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
    
    fn extract_task_id(&self, line: &str) -> Option<String> {
        // Look for patterns like "0.1", "1.2", etc.
        let regex = regex::Regex::new(r"\b(\d+\.\d+)\b").ok()?;
        regex.captures(line)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
    }
    
    fn extract_description(&self, line: &str) -> String {
        // Extract description after task ID and dash
        if let Some(pos) = line.find(" - ") {
            let desc = &line[pos + 3..];
            // Remove status markers
            desc.replace("✅", "")
                .replace("🔄", "")
                .replace("❌", "")
                .trim()
                .to_string()
        } else {
            line.to_string()
        }
    }
    
    fn extract_hours(&self, line: &str) -> f64 {
        // Look for patterns like "(40h)" or "40 hours"
        let regex = regex::Regex::new(r"(\d+)h").unwrap();
        regex.captures(line)
            .and_then(|cap| cap.get(1))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0.0)
    }
    
    fn extract_status(&self, line: &str) -> String {
        if line.contains("✅") || line.contains("COMPLETE") {
            "completed".to_string()
        } else if line.contains("🔄") || line.contains("IN PROGRESS") {
            "in_progress".to_string()
        } else if line.contains("❌") || line.contains("BLOCKED") {
            "blocked".to_string()
        } else {
            "pending".to_string()
        }
    }
    
    fn extract_assignees(&self, line: &str) -> Vec<String> {
        // Look for patterns like "(Alex)" or "Team: Alex, Morgan"
        let mut assignees = Vec::new();
        
        // Bot4 team members
        let team_members = ["Alex", "Morgan", "Sam", "Quinn", "Jordan", "Casey", "Riley", "Avery"];
        
        for member in &team_members {
            if line.contains(member) {
                assignees.push(member.to_string());
            }
        }
        
        // If no specific assignee, it's full team
        if assignees.is_empty() && (line.contains("TEAM") || line.contains("ALL")) {
            assignees = team_members.iter().map(|s| s.to_string()).collect();
        }
        
        assignees
    }
    
    fn get_missing_items(&self, validation: &ValidationStatus) -> Vec<String> {
        let mut missing = Vec::new();
        
        if !validation.code_complete {
            missing.push("Code implementation".to_string());
        }
        if !validation.tests_pass {
            missing.push("Passing tests".to_string());
        }
        if !validation.documentation_updated {
            missing.push("Documentation update".to_string());
        }
        if !validation.integration_verified {
            missing.push("Integration verification".to_string());
        }
        if !validation.reviewed {
            missing.push("Code review".to_string());
        }
        
        missing
    }
    
    fn get_layer_name(&self, layer: usize) -> String {
        match layer {
            0 => "Safety Systems".to_string(),
            1 => "Data Foundation".to_string(),
            2 => "Risk Management".to_string(),
            3 => "ML Pipeline".to_string(),
            4 => "Trading Strategies".to_string(),
            5 => "Execution Engine".to_string(),
            6 => "Infrastructure".to_string(),
            7 => "Integration & Testing".to_string(),
            _ => format!("Layer {}", layer),
        }
    }
    
    fn task_id_to_component(&self, task_id: &str) -> String {
        // Map task IDs to expected components
        // This is simplified - real implementation would use detailed mapping
        match task_id {
            "0.1" => "kill_switch",
            "0.2" => "circuit_breaker",
            "0.3" => "control_modes",
            "1.1" => "data_ingestion",
            "1.2" => "market_data",
            "2.1" => "risk_engine",
            _ => "unknown",
        }.to_string()
    }
    
    fn get_next_priority_task(&self, progress: &ProjectProgress) -> String {
        // Find next task based on layer priorities
        for layer in 0..=7 {
            if let Some(layer_progress) = progress.layers.get(&layer) {
                if layer_progress.percentage < 100.0 {
                    return format!("Complete Layer {} ({}: {:.0}% done)",
                                 layer, layer_progress.name, layer_progress.percentage);
                }
            }
        }
        
        "All layers complete!".to_string()
    }
    
    fn estimate_completion(&self, progress: &ProjectProgress) -> String {
        let remaining_hours = progress.total_hours - progress.hours_completed;
        let hours_per_week = 40.0 * 8.0; // 8 team members * 40 hours
        let weeks_remaining = remaining_hours / hours_per_week;
        
        format!("{:.1} weeks ({:.0} hours remaining)", weeks_remaining, remaining_hours)
    }
}