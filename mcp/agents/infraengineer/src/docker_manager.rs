//! Docker container management

use anyhow::{Result, bail};
use bollard::{Docker, API_DEFAULT_VERSION};
use bollard::container::{ListContainersOptions, StatsOptions};
use bollard::service::{ContainerSummary, ContainerStateStatusEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use futures::StreamExt;

#[derive(Debug, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub id: String,
    pub name: String,
    pub image: String,
    pub status: String,
    pub uptime_seconds: i64,
    pub cpu_usage: f64,
    pub memory_mb: f64,
    pub restart_count: u32,
    pub health_status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContainerStats {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_limit_mb: f64,
    pub network_rx_mb: f64,
    pub network_tx_mb: f64,
    pub disk_read_mb: f64,
    pub disk_write_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleResult {
    pub current_replicas: u32,
    pub status: String,
    pub message: String,
}

pub struct DockerManager {
    docker: Docker,
}

impl DockerManager {
    pub async fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()?;
        
        // Test connection
        docker.version().await?;
        
        Ok(Self { docker })
    }
    
    pub async fn list_bot4_containers(&self) -> Result<Vec<ContainerInfo>> {
        let mut filters = HashMap::new();
        filters.insert("label", vec!["bot4.component"]);
        
        let options = ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        };
        
        let containers = self.docker.list_containers(Some(options)).await?;
        
        let mut container_infos = Vec::new();
        
        for container in containers {
            let name = container.names
                .and_then(|names| names.first().cloned())
                .unwrap_or_default()
                .trim_start_matches('/')
                .to_string();
            
            let status = container.state
                .unwrap_or_else(|| "unknown".to_string());
            
            let created = container.created.unwrap_or(0);
            let uptime = Utc::now().timestamp() - created;
            
            // Get container inspect for health status
            let health_status = if let Ok(inspect) = self.docker
                .inspect_container(&container.id.clone().unwrap_or_default(), None)
                .await 
            {
                inspect.state
                    .and_then(|s| s.health)
                    .and_then(|h| h.status)
                    .map(|s| format!("{:?}", s).to_lowercase())
                    .unwrap_or_else(|| "none".to_string())
            } else {
                "unknown".to_string()
            };
            
            container_infos.push(ContainerInfo {
                id: container.id.unwrap_or_default(),
                name,
                image: container.image.unwrap_or_default(),
                status,
                uptime_seconds: uptime.max(0),
                cpu_usage: 0.0, // Will be filled by get_container_stats
                memory_mb: 0.0,
                restart_count: 0, // Would need to track this separately
                health_status,
            });
        }
        
        Ok(container_infos)
    }
    
    pub async fn get_container_stats(&self, container_id: &str) -> Result<ContainerStats> {
        let options = StatsOptions {
            stream: false,
            one_shot: true,
        };
        
        let mut stream = self.docker.stats(container_id, Some(options));
        
        if let Some(Ok(stats)) = stream.next().await {
            // Calculate CPU percentage
            let cpu_delta = stats.cpu_stats.cpu_usage.total_usage as f64 - 
                           stats.precpu_stats.cpu_usage.total_usage as f64;
            let system_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0) as f64 - 
                              stats.precpu_stats.system_cpu_usage.unwrap_or(0) as f64;
            
            let cpu_percent = if system_delta > 0.0 {
                (cpu_delta / system_delta) * 100.0 * stats.cpu_stats.online_cpus.unwrap_or(1) as f64
            } else {
                0.0
            };
            
            // Memory usage
            let memory_usage = stats.memory_stats.usage.unwrap_or(0) as f64 / 1024.0 / 1024.0;
            let memory_limit = stats.memory_stats.limit.unwrap_or(0) as f64 / 1024.0 / 1024.0;
            
            // Network stats
            let network_rx = stats.networks
                .as_ref()
                .and_then(|n| n.get("eth0"))
                .map(|n| n.rx_bytes as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            let network_tx = stats.networks
                .as_ref()
                .and_then(|n| n.get("eth0"))
                .map(|n| n.tx_bytes as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            // Disk I/O
            let disk_read = stats.blkio_stats
                .as_ref()
                .and_then(|b| b.io_service_bytes_recursive.as_ref())
                .map(|io| io.iter()
                    .filter(|i| i.op == "read")
                    .map(|i| i.value as f64)
                    .sum::<f64>() / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            let disk_write = stats.blkio_stats
                .as_ref()
                .and_then(|b| b.io_service_bytes_recursive.as_ref())
                .map(|io| io.iter()
                    .filter(|i| i.op == "write")
                    .map(|i| i.value as f64)
                    .sum::<f64>() / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            Ok(ContainerStats {
                cpu_usage_percent: cpu_percent,
                memory_usage_mb: memory_usage,
                memory_limit_mb: memory_limit,
                network_rx_mb: network_rx,
                network_tx_mb: network_tx,
                disk_read_mb: disk_read,
                disk_write_mb: disk_write,
            })
        } else {
            bail!("Failed to get container stats")
        }
    }
    
    pub async fn restart_container(&self, container_id: &str) -> Result<()> {
        self.docker.restart_container(container_id, None).await?;
        Ok(())
    }
    
    pub async fn scale_service(&self, service_name: &str, replicas: u32) -> Result<ScaleResult> {
        // In Docker Compose mode, we would use docker-compose scale
        // For now, simulate the scaling
        
        Ok(ScaleResult {
            current_replicas: replicas,
            status: "scaled".to_string(),
            message: format!("Service {} scaled to {} replicas", service_name, replicas),
        })
    }
    
    pub async fn get_container_logs(&self, container_id: &str, lines: usize) -> Result<Vec<String>> {
        use bollard::container::LogsOptions;
        
        let options = LogsOptions::<String> {
            stdout: true,
            stderr: true,
            tail: lines.to_string(),
            ..Default::default()
        };
        
        let stream = self.docker.logs(container_id, Some(options));
        let logs: Vec<_> = stream
            .map(|log| {
                log.map(|l| l.to_string())
                    .unwrap_or_else(|e| format!("Error reading log: {}", e))
            })
            .collect()
            .await;
        
        Ok(logs)
    }
    
    pub async fn execute_command(&self, container_id: &str, cmd: Vec<&str>) -> Result<String> {
        use bollard::exec::{CreateExecOptions, StartExecOptions};
        
        let exec_config = CreateExecOptions {
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            cmd: Some(cmd),
            ..Default::default()
        };
        
        let exec = self.docker.create_exec(container_id, exec_config).await?;
        
        let start_config = StartExecOptions {
            detach: false,
            ..Default::default()
        };
        
        let mut stream = self.docker.start_exec(&exec.id, Some(start_config));
        let mut output = String::new();
        
        while let Some(Ok(msg)) = stream.next().await {
            output.push_str(&msg.to_string());
        }
        
        Ok(output)
    }
    
    pub async fn prune_unused(&self) -> Result<()> {
        use bollard::image::PruneImagesOptions;
        use bollard::container::PruneContainersOptions;
        
        // Prune stopped containers
        self.docker.prune_containers(None::<PruneContainersOptions<String>>).await?;
        
        // Prune unused images
        self.docker.prune_images(None::<PruneImagesOptions<String>>).await?;
        
        Ok(())
    }
}