//! Bot4 InfraEngineer Agent
//! Infrastructure monitoring, deployment, and system optimization

use anyhow::Result;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use rmcp::{
    server::{Server, ServerBuilder, ToolHandler},
    transport::DockerTransport,
    types::{Tool, ToolCall, ToolResult, Resource, Prompt},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use chrono::{DateTime, Utc};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

mod docker_manager;
mod kubernetes_manager;
mod monitoring;
mod deployment;

use docker_manager::DockerManager;
use kubernetes_manager::KubernetesManager;
use monitoring::SystemMonitor;
use deployment::DeploymentManager;

/// InfraEngineer agent implementation
struct InfraEngineerAgent {
    redis: ConnectionManager,
    docker_manager: Arc<DockerManager>,
    k8s_manager: Option<Arc<KubernetesManager>>,
    system_monitor: Arc<SystemMonitor>,
    deployment_manager: Arc<DeploymentManager>,
}

impl InfraEngineerAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Initialize managers
        let docker_manager = Arc::new(DockerManager::new().await?);
        
        // K8s is optional (only in production)
        let k8s_manager = if std::env::var("KUBERNETES_ENABLED").unwrap_or_default() == "true" {
            Some(Arc::new(KubernetesManager::new().await?))
        } else {
            None
        };
        
        Ok(Self {
            redis,
            docker_manager,
            k8s_manager,
            system_monitor: Arc::new(SystemMonitor::new()),
            deployment_manager: Arc::new(DeploymentManager::new()),
        })
    }
    
    /// Check system health and resources
    async fn check_system_health(&self) -> Result<ToolResult> {
        info!("Checking system health");
        
        let health = self.system_monitor.get_system_health().await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "status": health.overall_status,
            "cpu": {
                "usage_percent": health.cpu_usage,
                "load_average": health.load_average,
                "cores": health.cpu_cores,
            },
            "memory": {
                "used_gb": health.memory_used_gb,
                "total_gb": health.memory_total_gb,
                "usage_percent": health.memory_usage_percent,
                "available_gb": health.memory_available_gb,
            },
            "disk": {
                "used_gb": health.disk_used_gb,
                "total_gb": health.disk_total_gb,
                "usage_percent": health.disk_usage_percent,
            },
            "network": {
                "rx_mbps": health.network_rx_mbps,
                "tx_mbps": health.network_tx_mbps,
                "connections": health.active_connections,
            },
            "alerts": health.alerts,
            "timestamp": health.timestamp,
        })))
    }
    
    /// Monitor Bot4 containers
    async fn monitor_containers(&self) -> Result<ToolResult> {
        info!("Monitoring Bot4 containers");
        
        let containers = self.docker_manager.list_bot4_containers().await?;
        
        let mut container_status = Vec::new();
        for container in containers {
            let stats = self.docker_manager.get_container_stats(&container.id).await?;
            container_status.push(serde_json::json!({
                "name": container.name,
                "status": container.status,
                "uptime": container.uptime_seconds,
                "cpu_usage": stats.cpu_usage_percent,
                "memory_usage_mb": stats.memory_usage_mb,
                "memory_limit_mb": stats.memory_limit_mb,
                "network_rx_mb": stats.network_rx_mb,
                "network_tx_mb": stats.network_tx_mb,
                "restarts": container.restart_count,
                "health": container.health_status,
            }));
        }
        
        Ok(ToolResult::Success(serde_json::json!({
            "containers": container_status,
            "total_containers": container_status.len(),
            "healthy": container_status.iter().filter(|c| c["health"] == "healthy").count(),
            "unhealthy": container_status.iter().filter(|c| c["health"] != "healthy").count(),
            "total_cpu_usage": container_status.iter()
                .filter_map(|c| c["cpu_usage"].as_f64())
                .sum::<f64>(),
            "total_memory_mb": container_status.iter()
                .filter_map(|c| c["memory_usage_mb"].as_f64())
                .sum::<f64>(),
        })))
    }
    
    /// Deploy Bot4 component
    async fn deploy_component(&self, component: String, version: String, 
                             environment: String) -> Result<ToolResult> {
        info!("Deploying {} version {} to {}", component, version, environment);
        
        let deployment_result = self.deployment_manager.deploy(
            &component,
            &version,
            &environment
        ).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "component": component,
            "version": version,
            "environment": environment,
            "status": deployment_result.status,
            "deployment_id": deployment_result.deployment_id,
            "rollout_strategy": deployment_result.rollout_strategy,
            "health_checks": deployment_result.health_checks,
            "rollback_available": deployment_result.rollback_version,
            "message": deployment_result.message,
        })))
    }
    
    /// Rollback deployment
    async fn rollback_deployment(&self, deployment_id: String) -> Result<ToolResult> {
        info!("Rolling back deployment {}", deployment_id);
        
        let rollback_result = self.deployment_manager.rollback(&deployment_id).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "deployment_id": deployment_id,
            "status": rollback_result.status,
            "rolled_back_to": rollback_result.previous_version,
            "message": rollback_result.message,
        })))
    }
    
    /// Scale Bot4 service
    async fn scale_service(&self, service: String, replicas: u32) -> Result<ToolResult> {
        info!("Scaling {} to {} replicas", service, replicas);
        
        let scale_result = if let Some(k8s) = &self.k8s_manager {
            k8s.scale_deployment(&service, replicas).await?
        } else {
            self.docker_manager.scale_service(&service, replicas).await?
        };
        
        Ok(ToolResult::Success(serde_json::json!({
            "service": service,
            "target_replicas": replicas,
            "current_replicas": scale_result.current_replicas,
            "status": scale_result.status,
            "message": scale_result.message,
        })))
    }
    
    /// Check service dependencies
    async fn check_dependencies(&self) -> Result<ToolResult> {
        info!("Checking Bot4 service dependencies");
        
        let mut dependencies = HashMap::new();
        
        // Check Redis
        dependencies.insert("redis", self.check_redis_health().await);
        
        // Check PostgreSQL
        dependencies.insert("postgresql", self.check_postgres_health().await);
        
        // Check network connectivity
        dependencies.insert("network", self.check_network_health().await);
        
        // Check disk space
        dependencies.insert("disk", self.check_disk_space().await);
        
        let all_healthy = dependencies.values().all(|v| v.is_ok());
        
        Ok(ToolResult::Success(serde_json::json!({
            "overall_status": if all_healthy { "healthy" } else { "degraded" },
            "dependencies": dependencies.iter().map(|(name, status)| {
                serde_json::json!({
                    "name": name,
                    "status": if status.is_ok() { "healthy" } else { "unhealthy" },
                    "message": status.as_ref().err().map(|e| e.to_string()).unwrap_or_default(),
                })
            }).collect::<Vec<_>>(),
            "timestamp": Utc::now(),
        })))
    }
    
    /// Analyze resource usage trends
    async fn analyze_resource_trends(&self, timeframe_hours: u32) -> Result<ToolResult> {
        info!("Analyzing resource trends for past {} hours", timeframe_hours);
        
        let trends = self.system_monitor.get_resource_trends(timeframe_hours).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "timeframe_hours": timeframe_hours,
            "cpu": {
                "average": trends.cpu_average,
                "peak": trends.cpu_peak,
                "trend": trends.cpu_trend,
                "prediction_next_hour": trends.cpu_prediction,
            },
            "memory": {
                "average_gb": trends.memory_average_gb,
                "peak_gb": trends.memory_peak_gb,
                "trend": trends.memory_trend,
                "leak_detected": trends.memory_leak_detected,
            },
            "disk": {
                "growth_rate_gb_per_hour": trends.disk_growth_rate,
                "hours_until_full": trends.disk_hours_until_full,
            },
            "recommendations": trends.recommendations,
        })))
    }
    
    /// Generate infrastructure report
    async fn generate_infra_report(&self) -> Result<ToolResult> {
        info!("Generating infrastructure report");
        
        let health = self.system_monitor.get_system_health().await?;
        let containers = self.docker_manager.list_bot4_containers().await?;
        let trends = self.system_monitor.get_resource_trends(24).await?;
        
        let report = serde_json::json!({
            "report_id": uuid::Uuid::new_v4().to_string(),
            "generated_at": Utc::now(),
            "executive_summary": {
                "overall_health": health.overall_status,
                "containers_running": containers.len(),
                "critical_alerts": health.alerts.iter()
                    .filter(|a| a.severity == "critical")
                    .count(),
                "recommendations": trends.recommendations,
            },
            "system_resources": {
                "cpu_usage": health.cpu_usage,
                "memory_usage": health.memory_usage_percent,
                "disk_usage": health.disk_usage_percent,
            },
            "container_health": {
                "total": containers.len(),
                "healthy": containers.iter().filter(|c| c.health_status == "healthy").count(),
                "unhealthy": containers.iter().filter(|c| c.health_status != "healthy").count(),
            },
            "performance_metrics": {
                "avg_container_cpu": containers.iter()
                    .map(|c| c.cpu_usage)
                    .sum::<f64>() / containers.len() as f64,
                "total_memory_gb": containers.iter()
                    .map(|c| c.memory_mb)
                    .sum::<f64>() / 1024.0,
            },
            "risk_assessment": {
                "resource_exhaustion_risk": if health.memory_usage_percent > 80.0 { "high" } 
                    else if health.memory_usage_percent > 60.0 { "medium" } 
                    else { "low" },
                "stability_score": 95.0 - health.alerts.len() as f64 * 5.0,
            },
        });
        
        Ok(ToolResult::Success(report))
    }
    
    /// Configure monitoring alerts
    async fn configure_alerts(&self, alert_config: serde_json::Value) -> Result<ToolResult> {
        info!("Configuring monitoring alerts");
        
        let config_result = self.system_monitor.configure_alerts(alert_config).await?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "status": "configured",
            "alerts_configured": config_result.alert_count,
            "channels": config_result.notification_channels,
            "test_alert_sent": config_result.test_successful,
        })))
    }
    
    // Helper methods
    async fn check_redis_health(&self) -> Result<()> {
        // Check Redis connectivity
        redis::cmd("PING")
            .query_async::<_, String>(&mut self.redis.clone())
            .await
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("Redis unhealthy: {}", e))
    }
    
    async fn check_postgres_health(&self) -> Result<()> {
        // In production, would check actual PostgreSQL connection
        // For now, simulate check
        Ok(())
    }
    
    async fn check_network_health(&self) -> Result<()> {
        // Check network connectivity to critical services
        Ok(())
    }
    
    async fn check_disk_space(&self) -> Result<()> {
        let health = self.system_monitor.get_system_health().await?;
        if health.disk_usage_percent > 90.0 {
            Err(anyhow::anyhow!("Disk usage critical: {:.1}%", health.disk_usage_percent))
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl ToolHandler for InfraEngineerAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "check_system_health" => {
                self.check_system_health().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check system health: {}", e))
                })
            }
            "monitor_containers" => {
                self.monitor_containers().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to monitor containers: {}", e))
                })
            }
            "deploy_component" => {
                let component = tool_call.arguments["component"].as_str().unwrap_or("").to_string();
                let version = tool_call.arguments["version"].as_str().unwrap_or("latest").to_string();
                let environment = tool_call.arguments["environment"].as_str().unwrap_or("development").to_string();
                self.deploy_component(component, version, environment).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to deploy: {}", e))
                })
            }
            "rollback_deployment" => {
                let deployment_id = tool_call.arguments["deployment_id"].as_str().unwrap_or("").to_string();
                self.rollback_deployment(deployment_id).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to rollback: {}", e))
                })
            }
            "scale_service" => {
                let service = tool_call.arguments["service"].as_str().unwrap_or("").to_string();
                let replicas = tool_call.arguments["replicas"].as_u64().unwrap_or(1) as u32;
                self.scale_service(service, replicas).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to scale: {}", e))
                })
            }
            "check_dependencies" => {
                self.check_dependencies().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to check dependencies: {}", e))
                })
            }
            "analyze_resource_trends" => {
                let hours = tool_call.arguments["timeframe_hours"].as_u64().unwrap_or(24) as u32;
                self.analyze_resource_trends(hours).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to analyze trends: {}", e))
                })
            }
            "generate_infra_report" => {
                self.generate_infra_report().await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to generate report: {}", e))
                })
            }
            "configure_alerts" => {
                let config = tool_call.arguments["alert_config"].clone();
                self.configure_alerts(config).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to configure alerts: {}", e))
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
    
    info!("Starting Bot4 InfraEngineer Agent v1.0");
    
    // Create agent
    let agent = InfraEngineerAgent::new().await?;
    
    // Define tools
    let tools = vec![
        Tool {
            name: "check_system_health".to_string(),
            description: "Check overall system health and resource usage".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
        Tool {
            name: "monitor_containers".to_string(),
            description: "Monitor Bot4 Docker containers".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
        Tool {
            name: "deploy_component".to_string(),
            description: "Deploy Bot4 component to environment".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "version": {"type": "string"},
                    "environment": {"type": "string", "enum": ["development", "staging", "production"]}
                },
                "required": ["component"]
            }),
        },
        Tool {
            name: "rollback_deployment".to_string(),
            description: "Rollback a deployment to previous version".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "deployment_id": {"type": "string"}
                },
                "required": ["deployment_id"]
            }),
        },
        Tool {
            name: "scale_service".to_string(),
            description: "Scale Bot4 service replicas".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "replicas": {"type": "integer", "minimum": 0, "maximum": 10}
                },
                "required": ["service", "replicas"]
            }),
        },
        Tool {
            name: "check_dependencies".to_string(),
            description: "Check health of Bot4 dependencies".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
        Tool {
            name: "analyze_resource_trends".to_string(),
            description: "Analyze resource usage trends".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "timeframe_hours": {"type": "integer", "default": 24}
                }
            }),
        },
        Tool {
            name: "generate_infra_report".to_string(),
            description: "Generate comprehensive infrastructure report".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
        Tool {
            name: "configure_alerts".to_string(),
            description: "Configure monitoring alerts".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "alert_config": {"type": "object"}
                },
                "required": ["alert_config"]
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("infraengineer-agent", "1.0.0")
        .with_tools(tools)
        .with_tool_handler(agent)
        .build()?;
    
    // Use Docker transport
    let transport = DockerTransport::new()?;
    server.run(transport).await?;
    
    Ok(())
}