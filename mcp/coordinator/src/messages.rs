//! Message types and handling

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub from_agent: String,
    pub to_agents: Vec<String>,
    #[serde(rename = "type")]
    pub msg_type: MessageType,
    pub content: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    TaskAnnouncement,
    AnalysisResult,
    DesignProposal,
    ReviewComment,
    ConsensusVote,
    Veto,
    StatusUpdate,
    ContextUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnnouncement {
    pub task_id: String,
    pub description: String,
    pub estimated_hours: f64,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub findings: Vec<String>,
    pub risks: Vec<String>,
    pub recommendations: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignProposal {
    pub approach: String,
    pub rationale: String,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub file: String,
    pub line: u32,
    pub severity: Severity,
    pub issue: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Critical,
    Major,
    Minor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    pub proposal_id: String,
    pub vote: VoteType,
    pub rationale: String,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VoteType {
    Approve,
    Reject,
    Abstain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Veto {
    pub target: String,
    pub reason: String,
    pub domain: String,
    pub requirements_to_proceed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub task_id: String,
    pub phase: Phase,
    pub progress_percent: u8,
    pub blockers: Vec<String>,
    pub eta_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Phase {
    Analysis,
    Design,
    Implementation,
    Validation,
}

impl Message {
    pub fn new(
        from_agent: String,
        to_agents: Vec<String>,
        msg_type: MessageType,
        content: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_agent,
            to_agents,
            msg_type,
            content,
            timestamp: Utc::now(),
        }
    }
    
    pub fn broadcast(from_agent: String, msg_type: MessageType, content: serde_json::Value) -> Self {
        Self::new(from_agent, vec!["all".to_string()], msg_type, content)
    }
}