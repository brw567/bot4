//! System monitoring and alerting

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt, DiskExt, NetworkExt};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemHealth {
    pub overall_status: String,
    pub cpu_usage: f64,
    pub load_average: [f64; 3],
    pub cpu_cores: usize,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub memory_usage_percent: f64,
    pub memory_available_gb: f64,
    pub disk_used_gb: f64,
    pub disk_total_gb: f64,
    pub disk_usage_percent: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub active_connections: usize,
    pub alerts: Vec<Alert>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Alert {
    pub severity: String,
    pub component: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceTrends {
    pub cpu_average: f64,
    pub cpu_peak: f64,
    pub cpu_trend: String,
    pub cpu_prediction: f64,
    pub memory_average_gb: f64,
    pub memory_peak_gb: f64,
    pub memory_trend: String,
    pub memory_leak_detected: bool,
    pub disk_growth_rate: f64,
    pub disk_hours_until_full: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertConfig {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub disk_threshold: f64,
    pub enable_predictions: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertConfigResult {
    pub alert_count: usize,
    pub notification_channels: Vec<String>,
    pub test_successful: bool,
}

pub struct SystemMonitor {
    history: Arc<RwLock<VecDeque<SystemHealth>>>,
    alert_config: Arc<RwLock<AlertConfig>>,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self {
            history: Arc::new(RwLock::new(VecDeque::with_capacity(1440))), // 24 hours at 1min intervals
            alert_config: Arc::new(RwLock::new(AlertConfig {
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                disk_threshold: 90.0,
                enable_predictions: true,
            })),
        }
    }
    
    pub async fn get_system_health(&self) -> Result<SystemHealth> {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        // CPU metrics
        let cpu_usage = sys.global_cpu_info().cpu_usage() as f64;
        let load_avg = sys.load_average();
        let load_average = [load_avg.one, load_avg.five, load_avg.fifteen];
        let cpu_cores = sys.cpus().len();
        
        // Memory metrics
        let memory_total = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let memory_used = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let memory_available = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let memory_usage_percent = (memory_used / memory_total) * 100.0;
        
        // Disk metrics
        let mut disk_total = 0u64;
        let mut disk_used = 0u64;
        for disk in sys.disks() {
            disk_total += disk.total_space();
            disk_used += disk.total_space() - disk.available_space();
        }
        let disk_total_gb = disk_total as f64 / 1024.0 / 1024.0 / 1024.0;
        let disk_used_gb = disk_used as f64 / 1024.0 / 1024.0 / 1024.0;
        let disk_usage_percent = (disk_used_gb / disk_total_gb) * 100.0;
        
        // Network metrics
        let mut rx_bytes = 0u64;
        let mut tx_bytes = 0u64;
        for (_name, network) in sys.networks() {
            rx_bytes += network.received();
            tx_bytes += network.transmitted();
        }
        let network_rx_mbps = rx_bytes as f64 / 1024.0 / 1024.0 * 8.0;
        let network_tx_mbps = tx_bytes as f64 / 1024.0 / 1024.0 * 8.0;
        
        // Count active network connections (simplified)
        let active_connections = sys.processes().len();
        
        // Generate alerts
        let mut alerts = Vec::new();
        let config = self.alert_config.read().await;
        
        if cpu_usage > config.cpu_threshold {
            alerts.push(Alert {
                severity: if cpu_usage > 90.0 { "critical" } else { "warning" }.to_string(),
                component: "cpu".to_string(),
                message: format!("CPU usage high: {:.1}%", cpu_usage),
                timestamp: Utc::now(),
            });
        }
        
        if memory_usage_percent > config.memory_threshold {
            alerts.push(Alert {
                severity: if memory_usage_percent > 95.0 { "critical" } else { "warning" }.to_string(),
                component: "memory".to_string(),
                message: format!("Memory usage high: {:.1}%", memory_usage_percent),
                timestamp: Utc::now(),
            });
        }
        
        if disk_usage_percent > config.disk_threshold {
            alerts.push(Alert {
                severity: "critical".to_string(),
                component: "disk".to_string(),
                message: format!("Disk usage critical: {:.1}%", disk_usage_percent),
                timestamp: Utc::now(),
            });
        }
        
        // Determine overall status
        let overall_status = if alerts.iter().any(|a| a.severity == "critical") {
            "critical"
        } else if alerts.iter().any(|a| a.severity == "warning") {
            "warning"
        } else {
            "healthy"
        }.to_string();
        
        let health = SystemHealth {
            overall_status,
            cpu_usage,
            load_average,
            cpu_cores,
            memory_used_gb: memory_used,
            memory_total_gb: memory_total,
            memory_usage_percent,
            memory_available_gb: memory_available,
            disk_used_gb,
            disk_total_gb,
            disk_usage_percent,
            network_rx_mbps,
            network_tx_mbps,
            active_connections,
            alerts,
            timestamp: Utc::now(),
        };
        
        // Store in history
        let mut history = self.history.write().await;
        if history.len() >= 1440 {
            history.pop_front();
        }
        history.push_back(health.clone());
        
        Ok(health)
    }
    
    pub async fn get_resource_trends(&self, hours: u32) -> Result<ResourceTrends> {
        let history = self.history.read().await;
        
        // Calculate how many samples to look at (assuming 1-minute intervals)
        let samples_needed = (hours * 60).min(history.len() as u32) as usize;
        
        if samples_needed == 0 {
            // No history available
            return Ok(ResourceTrends {
                cpu_average: 0.0,
                cpu_peak: 0.0,
                cpu_trend: "stable".to_string(),
                cpu_prediction: 0.0,
                memory_average_gb: 0.0,
                memory_peak_gb: 0.0,
                memory_trend: "stable".to_string(),
                memory_leak_detected: false,
                disk_growth_rate: 0.0,
                disk_hours_until_full: f64::INFINITY,
                recommendations: vec!["Insufficient data for analysis".to_string()],
            });
        }
        
        // Get recent samples
        let recent_samples: Vec<SystemHealth> = history.iter()
            .rev()
            .take(samples_needed)
            .cloned()
            .collect();
        
        // CPU analysis
        let cpu_values: Vec<f64> = recent_samples.iter().map(|s| s.cpu_usage).collect();
        let cpu_average = cpu_values.iter().sum::<f64>() / cpu_values.len() as f64;
        let cpu_peak = cpu_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let cpu_trend = self.calculate_trend(&cpu_values);
        let cpu_prediction = self.predict_next_value(&cpu_values);
        
        // Memory analysis
        let memory_values: Vec<f64> = recent_samples.iter().map(|s| s.memory_used_gb).collect();
        let memory_average_gb = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
        let memory_peak_gb = memory_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let memory_trend = self.calculate_trend(&memory_values);
        let memory_leak_detected = self.detect_memory_leak(&memory_values);
        
        // Disk analysis
        let disk_values: Vec<f64> = recent_samples.iter().map(|s| s.disk_used_gb).collect();
        let disk_growth_rate = self.calculate_growth_rate(&disk_values);
        let disk_space_left = recent_samples.last()
            .map(|s| s.disk_total_gb - s.disk_used_gb)
            .unwrap_or(0.0);
        let disk_hours_until_full = if disk_growth_rate > 0.0 {
            disk_space_left / disk_growth_rate
        } else {
            f64::INFINITY
        };
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if cpu_average > 70.0 {
            recommendations.push("Consider scaling up CPU resources".to_string());
        }
        
        if memory_leak_detected {
            recommendations.push("Memory leak detected - investigate container restarts".to_string());
        }
        
        if disk_hours_until_full < 24.0 {
            recommendations.push(format!("Disk will be full in {:.1} hours - clean up logs/data", 
                                       disk_hours_until_full));
        }
        
        if cpu_trend == "increasing" && cpu_prediction > 90.0 {
            recommendations.push("CPU usage trending up - prepare to scale".to_string());
        }
        
        Ok(ResourceTrends {
            cpu_average,
            cpu_peak,
            cpu_trend,
            cpu_prediction,
            memory_average_gb,
            memory_peak_gb,
            memory_trend,
            memory_leak_detected,
            disk_growth_rate,
            disk_hours_until_full,
            recommendations,
        })
    }
    
    pub async fn configure_alerts(&self, config: serde_json::Value) -> Result<AlertConfigResult> {
        let new_config: AlertConfig = serde_json::from_value(config)?;
        
        let mut alert_config = self.alert_config.write().await;
        *alert_config = new_config;
        
        // In production, would configure actual alerting channels
        Ok(AlertConfigResult {
            alert_count: 4, // CPU, Memory, Disk, Network
            notification_channels: vec![
                "slack".to_string(),
                "email".to_string(),
                "pagerduty".to_string(),
            ],
            test_successful: true,
        })
    }
    
    fn calculate_trend(&self, values: &[f64]) -> String {
        if values.len() < 3 {
            return "stable".to_string();
        }
        
        // Simple linear regression
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        
        let slope = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        
        if slope > 0.5 {
            "increasing".to_string()
        } else if slope < -0.5 {
            "decreasing".to_string()
        } else {
            "stable".to_string()
        }
    }
    
    fn predict_next_value(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        // Simple moving average prediction
        let window = values.len().min(5);
        let recent_avg = values.iter().rev().take(window).sum::<f64>() / window as f64;
        
        // Add slight trend adjustment
        let trend = self.calculate_trend(values);
        match trend.as_str() {
            "increasing" => recent_avg * 1.05,
            "decreasing" => recent_avg * 0.95,
            _ => recent_avg,
        }
    }
    
    fn detect_memory_leak(&self, memory_values: &[f64]) -> bool {
        if memory_values.len() < 10 {
            return false;
        }
        
        // Check if memory consistently increases without significant drops
        let mut increasing_count = 0;
        for i in 1..memory_values.len() {
            if memory_values[i] > memory_values[i - 1] {
                increasing_count += 1;
            }
        }
        
        // If memory increases in >80% of samples, likely a leak
        increasing_count as f64 / memory_values.len() as f64 > 0.8
    }
    
    fn calculate_growth_rate(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let first = values.first().unwrap();
        let last = values.last().unwrap();
        let hours = values.len() as f64 / 60.0; // Assuming 1-minute samples
        
        (last - first) / hours
    }
}