//! Agent registry and management

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    pub capabilities: Vec<String>,
    pub last_heartbeat: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentType {
    Architect,
    RiskQuant,
    MLEngineer,
    ExchangeSpec,
    InfraEngineer,
    QualityGate,
    IntegrationValidator,
    ComplianceAuditor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Idle,
    Busy,
    Offline,
    Error,
}

impl Agent {
    pub fn is_healthy(&self) -> bool {
        let now = Utc::now();
        let time_since_heartbeat = now - self.last_heartbeat;
        time_since_heartbeat.num_seconds() < 120 && self.status != AgentStatus::Error
    }
}

pub struct AgentRegistry {
    agents: Arc<DashMap<String, Agent>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
        }
    }

    pub async fn register(&self, mut agent: Agent) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        agent.id = id.clone();
        agent.last_heartbeat = Utc::now();
        self.agents.insert(id.clone(), agent);
        Ok(id)
    }

    pub async fn unregister(&self, id: &str) -> Result<()> {
        self.agents.remove(id);
        Ok(())
    }

    pub async fn update_status(&self, id: String, status: serde_json::Value) -> Result<()> {
        if let Some(mut agent) = self.agents.get_mut(&id) {
            if let Some(status_str) = status.get("status").and_then(|s| s.as_str()) {
                agent.status = match status_str {
                    "active" => AgentStatus::Active,
                    "idle" => AgentStatus::Idle,
                    "busy" => AgentStatus::Busy,
                    "offline" => AgentStatus::Offline,
                    "error" => AgentStatus::Error,
                    _ => agent.status.clone(),
                };
            }
            agent.last_heartbeat = Utc::now();
        }
        Ok(())
    }

    pub async fn list_all(&self) -> Vec<Agent> {
        self.agents.iter().map(|entry| entry.value().clone()).collect()
    }

    pub async fn get(&self, id: &str) -> Option<Agent> {
        self.agents.get(id).map(|entry| entry.value().clone())
    }

    pub async fn list_by_type(&self, agent_type: AgentType) -> Vec<Agent> {
        self.agents
            .iter()
            .filter(|entry| entry.value().agent_type == agent_type)
            .map(|entry| entry.value().clone())
            .collect()
    }
}