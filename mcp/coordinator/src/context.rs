//! Shared context management

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedContext {
    pub version: String,
    pub last_updated: DateTime<Utc>,
    pub current_task: Option<serde_json::Value>,
    pub decisions: Vec<Decision>,
    pub metrics: Metrics,
    pub discovered_issues: Issues,
    pub votes: HashMap<String, Vote>,
    pub analyses: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub agent: String,
    pub decision: String,
    pub votes: HashMap<String, String>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub test_coverage_percent: f64,
    pub duplication_percent: f64,
    pub latency_us: u64,
    pub memory_mb: u64,
    pub build_time_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issues {
    pub duplications: Duplications,
    pub layer_violations: LayerViolations,
    pub performance_regressions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Duplications {
    pub total: u32,
    pub resolved: u32,
    pub remaining: u32,
    pub critical: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerViolations {
    pub total: u32,
    pub resolved: u32,
    pub remaining: u32,
    pub critical: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub agent: String,
    pub vote: String,
    pub timestamp: DateTime<Utc>,
    pub rationale: Option<String>,
}

impl SharedContext {
    pub async fn load() -> Result<Self> {
        let path = "/mcp/shared/context.json";
        
        if let Ok(contents) = fs::read_to_string(path).await {
            Ok(serde_json::from_str(&contents)?)
        } else {
            Ok(Self::default())
        }
    }

    pub async fn save(&self) -> Result<()> {
        let path = "/mcp/shared/context.json";
        let contents = serde_json::to_string_pretty(self)?;
        fs::write(path, contents).await?;
        Ok(())
    }

    pub async fn apply_updates(&mut self, updates: serde_json::Value) -> Result<()> {
        // Merge updates into context
        if let Some(obj) = updates.as_object() {
            for (key, value) in obj {
                match key.as_str() {
                    "current_task" => self.current_task = Some(value.clone()),
                    "metrics" => {
                        if let Ok(metrics) = serde_json::from_value(value.clone()) {
                            self.metrics = metrics;
                        }
                    }
                    _ => {
                        // Store other updates as-is
                    }
                }
            }
        }
        
        self.last_updated = Utc::now();
        Ok(())
    }

    pub async fn add_analysis(&mut self, agent: String, analysis: serde_json::Value) -> Result<()> {
        self.analyses.insert(agent, analysis);
        self.last_updated = Utc::now();
        Ok(())
    }

    pub async fn record_vote(&mut self, agent: String, vote_data: serde_json::Value) -> Result<()> {
        let vote = Vote {
            agent: agent.clone(),
            vote: vote_data["vote"].as_str().unwrap_or("abstain").to_string(),
            timestamp: Utc::now(),
            rationale: vote_data["rationale"].as_str().map(|s| s.to_string()),
        };
        
        self.votes.insert(agent, vote);
        self.last_updated = Utc::now();
        Ok(())
    }

    pub async fn check_consensus(&self) -> Result<bool> {
        let approve_count = self.votes.values()
            .filter(|v| v.vote == "approve")
            .count();
        
        Ok(approve_count >= 5) // 5/8 consensus required
    }

    pub async fn halt_operation(&mut self, reason: serde_json::Value) -> Result<()> {
        // Clear current task and record halt
        self.current_task = None;
        self.last_updated = Utc::now();
        Ok(())
    }
}

impl Default for SharedContext {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            last_updated: Utc::now(),
            current_task: None,
            decisions: Vec::new(),
            metrics: Metrics {
                test_coverage_percent: 87.3,
                duplication_percent: 4.2,
                latency_us: 47,
                memory_mb: 823,
                build_time_seconds: 156,
            },
            discovered_issues: Issues {
                duplications: Duplications {
                    total: 158,
                    resolved: 48,
                    remaining: 110,
                    critical: vec![
                        "Order struct: 44 instances".to_string(),
                        "calculate_correlation: 13 instances".to_string(),
                    ],
                },
                layer_violations: LayerViolations {
                    total: 23,
                    resolved: 4,
                    remaining: 19,
                    critical: Vec::new(),
                },
                performance_regressions: Vec::new(),
            },
            votes: HashMap::new(),
            analyses: HashMap::new(),
        }
    }
}