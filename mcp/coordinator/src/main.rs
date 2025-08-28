//! Bot4 MCP Coordinator
//! Central hub for multi-agent communication and coordination

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use dashmap::DashMap;
use prometheus::{Encoder, TextEncoder, Counter, Histogram, register_counter, register_histogram};
use redis::{aio::ConnectionManager, AsyncCommands};
use rmcp::{
    server::{Server, ServerBuilder},
    transport::StdioTransport,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

mod agents;
mod context;
mod messages;

use agents::{Agent, AgentRegistry};
use context::SharedContext;
use messages::{Message, MessageType};

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    agents: Arc<AgentRegistry>,
    context: Arc<RwLock<SharedContext>>,
    redis: ConnectionManager,
    metrics: Arc<Metrics>,
}

/// Prometheus metrics
struct Metrics {
    messages_total: Counter,
    message_latency: Histogram,
    consensus_reached: Counter,
    vetos_issued: Counter,
}

impl Metrics {
    fn new() -> Result<Self> {
        Ok(Self {
            messages_total: register_counter!("mcp_messages_total", "Total MCP messages")?,
            message_latency: register_histogram!("mcp_message_latency_seconds", "Message processing latency")?,
            consensus_reached: register_counter!("mcp_consensus_reached_total", "Consensus decisions reached")?,
            vetos_issued: register_counter!("mcp_vetos_total", "Total vetos issued")?,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    info!("Starting Bot4 MCP Coordinator v1.0");

    // Connect to PostgreSQL
    let database_url = std::env::var("POSTGRES_URL")
        .unwrap_or_else(|_| "postgresql://bot4user:bot4pass@postgres:5432/bot4trading".to_string());
    
    let pg_pool = PgPoolOptions::new()
        .max_connections(10)
        .connect_timeout(Duration::from_secs(10))
        .connect(&database_url)
        .await?;
    
    info!("Connected to PostgreSQL");

    // Connect to Redis
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://redis:6379".to_string());
    
    let redis_client = redis::Client::open(redis_url)?;
    let redis_conn = ConnectionManager::new(redis_client).await?;
    
    info!("Connected to Redis");

    // Initialize metrics
    let metrics = Arc::new(Metrics::new()?);

    // Initialize shared state
    let agents = Arc::new(AgentRegistry::new());
    let context = Arc::new(RwLock::new(SharedContext::load().await?));
    
    let state = AppState {
        agents: agents.clone(),
        context: context.clone(),
        redis: redis_conn.clone(),
        metrics: metrics.clone(),
    };

    // Spawn background tasks
    tokio::spawn(context_sync_task(state.clone()));
    tokio::spawn(agent_health_check_task(state.clone()));
    
    // Build HTTP router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        .route("/api/agents", get(list_agents))
        .route("/api/agents/:id/register", post(register_agent))
        .route("/api/messages", post(handle_message))
        .route("/api/context", get(get_context))
        .route("/api/context", post(update_context))
        .route("/api/consensus", post(request_consensus))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    info!("MCP Coordinator listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Prometheus metrics endpoint
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// List all registered agents
async fn list_agents(State(state): State<AppState>) -> impl IntoResponse {
    let agents = state.agents.list_all().await;
    Json(agents)
}

/// Register a new agent
async fn register_agent(
    State(state): State<AppState>,
    Json(agent): Json<Agent>,
) -> impl IntoResponse {
    match state.agents.register(agent).await {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))),
        Err(e) => {
            error!("Failed to register agent: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()})))
        }
    }
}

/// Handle incoming message from agent
async fn handle_message(
    State(state): State<AppState>,
    Json(message): Json<Message>,
) -> impl IntoResponse {
    let timer = state.metrics.message_latency.start_timer();
    state.metrics.messages_total.inc();
    
    // Process message based on type
    let result = match message.msg_type {
        MessageType::TaskAnnouncement => handle_task_announcement(state.clone(), message).await,
        MessageType::AnalysisResult => handle_analysis_result(state.clone(), message).await,
        MessageType::DesignProposal => handle_design_proposal(state.clone(), message).await,
        MessageType::ReviewComment => handle_review_comment(state.clone(), message).await,
        MessageType::ConsensusVote => handle_consensus_vote(state.clone(), message).await,
        MessageType::Veto => {
            state.metrics.vetos_issued.inc();
            handle_veto(state.clone(), message).await
        },
        MessageType::StatusUpdate => handle_status_update(state.clone(), message).await,
        MessageType::ContextUpdate => handle_context_update(state.clone(), message).await,
    };
    
    timer.observe_duration();
    
    match result {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => {
            error!("Message handling failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()})))
        }
    }
}

/// Get current shared context
async fn get_context(State(state): State<AppState>) -> impl IntoResponse {
    let context = state.context.read().await;
    Json(context.clone())
}

/// Update shared context
async fn update_context(
    State(state): State<AppState>,
    Json(updates): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut context = state.context.write().await;
    
    if let Err(e) = context.apply_updates(updates).await {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()})));
    }
    
    // Broadcast update to all agents
    let _ = state.redis.publish::<_, _, ()>(
        "bot4:context:updates",
        serde_json::to_string(&context.clone()).unwrap(),
    ).await;
    
    (StatusCode::OK, Json(serde_json::json!({"status": "updated"})))
}

/// Request consensus from agents
async fn request_consensus(
    State(state): State<AppState>,
    Json(proposal): Json<serde_json::Value>,
) -> impl IntoResponse {
    debug!("Requesting consensus on proposal");
    
    // Broadcast proposal to all agents
    let proposal_id = Uuid::new_v4();
    let message = Message {
        id: Uuid::new_v4(),
        from_agent: "coordinator".to_string(),
        to_agents: vec!["all".to_string()],
        msg_type: MessageType::DesignProposal,
        content: proposal,
        timestamp: chrono::Utc::now(),
    };
    
    // Publish to Redis for all agents
    let _ = state.redis.publish::<_, _, ()>(
        "bot4:proposals",
        serde_json::to_string(&message).unwrap(),
    ).await;
    
    // Wait for votes (simplified - in production would track votes)
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    state.metrics.consensus_reached.inc();
    
    (StatusCode::OK, Json(serde_json::json!({
        "proposal_id": proposal_id,
        "status": "voting",
        "timeout_seconds": 30
    })))
}

// Message handlers
async fn handle_task_announcement(state: AppState, message: Message) -> Result<serde_json::Value> {
    info!("Task announcement from {}: {:?}", message.from_agent, message.content);
    
    // Update context with new task
    let mut context = state.context.write().await;
    context.current_task = Some(message.content.clone());
    
    // Broadcast to all agents
    let _ = state.redis.publish::<_, _, ()>(
        "bot4:tasks",
        serde_json::to_string(&message).unwrap(),
    ).await;
    
    Ok(serde_json::json!({"status": "announced"}))
}

async fn handle_analysis_result(state: AppState, message: Message) -> Result<serde_json::Value> {
    debug!("Analysis result from {}", message.from_agent);
    
    // Store analysis in context
    let mut context = state.context.write().await;
    context.add_analysis(message.from_agent.clone(), message.content.clone()).await?;
    
    Ok(serde_json::json!({"status": "recorded"}))
}

async fn handle_design_proposal(state: AppState, message: Message) -> Result<serde_json::Value> {
    info!("Design proposal from {}", message.from_agent);
    
    // Trigger voting process
    request_consensus(State(state.clone()), Json(message.content)).await;
    
    Ok(serde_json::json!({"status": "voting_initiated"}))
}

async fn handle_review_comment(state: AppState, message: Message) -> Result<serde_json::Value> {
    debug!("Review comment from {}", message.from_agent);
    
    // Broadcast to relevant agents
    let _ = state.redis.publish::<_, _, ()>(
        "bot4:reviews",
        serde_json::to_string(&message).unwrap(),
    ).await;
    
    Ok(serde_json::json!({"status": "broadcast"}))
}

async fn handle_consensus_vote(state: AppState, message: Message) -> Result<serde_json::Value> {
    debug!("Vote from {}: {:?}", message.from_agent, message.content);
    
    // Record vote in context
    let mut context = state.context.write().await;
    context.record_vote(message.from_agent.clone(), message.content.clone()).await?;
    
    // Check if consensus reached (5/8)
    if context.check_consensus().await? {
        state.metrics.consensus_reached.inc();
        info!("Consensus reached!");
    }
    
    Ok(serde_json::json!({"status": "vote_recorded"}))
}

async fn handle_veto(state: AppState, message: Message) -> Result<serde_json::Value> {
    warn!("VETO from {}: {:?}", message.from_agent, message.content);
    
    // Halt current operation
    let mut context = state.context.write().await;
    context.halt_operation(message.content.clone()).await?;
    
    // Broadcast veto to all agents
    let _ = state.redis.publish::<_, _, ()>(
        "bot4:vetos",
        serde_json::to_string(&message).unwrap(),
    ).await;
    
    Ok(serde_json::json!({"status": "operation_halted"}))
}

async fn handle_status_update(state: AppState, message: Message) -> Result<serde_json::Value> {
    debug!("Status update from {}", message.from_agent);
    
    // Update agent status
    state.agents.update_status(message.from_agent.clone(), message.content.clone()).await?;
    
    Ok(serde_json::json!({"status": "updated"}))
}

async fn handle_context_update(state: AppState, message: Message) -> Result<serde_json::Value> {
    debug!("Context update from {}", message.from_agent);
    
    // Apply context updates
    let mut context = state.context.write().await;
    context.apply_updates(message.content).await?;
    
    Ok(serde_json::json!({"status": "context_updated"}))
}

/// Background task to sync context to disk
async fn context_sync_task(state: AppState) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        let context = state.context.read().await;
        if let Err(e) = context.save().await {
            error!("Failed to save context: {}", e);
        } else {
            debug!("Context synced to disk");
        }
    }
}

/// Background task to check agent health
async fn agent_health_check_task(state: AppState) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    
    loop {
        interval.tick().await;
        
        let agents = state.agents.list_all().await;
        for agent in agents {
            if !agent.is_healthy() {
                warn!("Agent {} is unhealthy", agent.id);
                // In production, would trigger recovery actions
            }
        }
    }
}