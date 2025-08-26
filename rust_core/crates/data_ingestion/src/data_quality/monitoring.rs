// Continuous Data Quality Monitoring and Alerting
// Real-time monitoring with multi-channel alert distribution

use std::collections::VecDeque;
use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{info, warn, error};

use super::{ValidationResult, IssueCategory};

#[derive(Debug, Clone, Deserialize)]
pub struct MonitoringConfig {
    pub check_interval_ms: u64,
    pub alert_channels: Vec<AlertChannel>,
    pub retention_hours: i64,
    pub alert_rate_limit: usize,  // Max alerts per minute
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 1000,
            alert_channels: vec![AlertChannel::Log],
            retention_hours: 24,
            alert_rate_limit: 10,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum AlertChannel {
    Log,
    Webhook(String),
    Email(String),
    Slack(String),
}

#[async_trait]
pub trait AlertSender: Send + Sync {
    async fn send_alert(&self, alert: &QualityAlert) -> Result<()>;
}

pub struct QualityMonitor {
    config: MonitoringConfig,
    metrics_buffer: Arc<RwLock<VecDeque<MonitoringMetrics>>>,
    alert_queue: mpsc::Sender<QualityAlert>,
    alert_receiver: Arc<RwLock<mpsc::Receiver<QualityAlert>>>,
    shutdown: Arc<RwLock<bool>>,
}

impl QualityMonitor {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let (alert_queue, alert_receiver) = mpsc::channel(1000);
        
        Ok(Self {
            config,
            metrics_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            alert_queue,
            alert_receiver: Arc::new(RwLock::new(alert_receiver)),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start_monitoring(&self) -> Result<()> {
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitoring_loop().await;
        });
        
        let alert_processor = self.clone();
        tokio::spawn(async move {
            alert_processor.alert_processing_loop().await;
        });
        
        info!("Quality monitoring started");
        Ok(())
    }

    async fn monitoring_loop(&self) {
        let mut check_interval = interval(std::time::Duration::from_millis(self.config.check_interval_ms));
        
        loop {
            if *self.shutdown.read().await {
                break;
            }
            
            check_interval.tick().await;
            
            // Analyze recent metrics
            if let Err(e) = self.analyze_metrics().await {
                error!("Error analyzing metrics: {}", e);
            }
            
            // Clean old metrics
            self.cleanup_old_metrics().await;
        }
    }

    async fn alert_processing_loop(&self) {
        let receiver = self.alert_receiver.clone();
        let mut rate_limiter = RateLimiter::new(self.config.alert_rate_limit);
        
        loop {
            if *self.shutdown.read().await {
                break;
            }
            
            let alert = {
                let mut recv = receiver.write().await;
                recv.recv().await
            };
            
            if let Some(alert) = alert {
                if rate_limiter.should_allow() {
                    self.distribute_alert(&alert).await;
                } else {
                    warn!("Alert rate limited: {:?}", alert);
                }
            }
        }
    }

    async fn analyze_metrics(&self) -> Result<()> {
        let buffer = self.metrics_buffer.read().await;
        
        if buffer.len() < 10 {
            return Ok(());
        }
        
        // Check for degradation trends
        let recent: Vec<_> = buffer.iter().rev().take(10).collect();
        let avg_quality: f64 = recent.iter().map(|m| m.quality_score).sum::<f64>() / recent.len() as f64;
        
        if avg_quality < 0.8 {
            self.alert_queue.send(QualityAlert {
                timestamp: Utc::now(),
                symbol: "SYSTEM".to_string(),
                severity: AlertSeverity::High,
                category: IssueCategory::DataGap,
                message: format!("Data quality degraded to {:.2}%", avg_quality * 100.0),
                quality_score: avg_quality,
            }).await?;
        }
        
        Ok(())
    }

    async fn cleanup_old_metrics(&self) {
        let mut buffer = self.metrics_buffer.write().await;
        let cutoff = Utc::now() - chrono::Duration::hours(self.config.retention_hours);
        
        while let Some(front) = buffer.front() {
            if front.timestamp < cutoff {
                buffer.pop_front();
            } else {
                break;
            }
        }
    }

    async fn distribute_alert(&self, alert: &QualityAlert) {
        for channel in &self.config.alert_channels {
            match channel {
                AlertChannel::Log => {
                    match alert.severity {
                        AlertSeverity::Critical => error!("CRITICAL: {}", alert.message),
                        AlertSeverity::High => warn!("HIGH: {}", alert.message),
                        _ => info!("{}", alert.message),
                    }
                }
                AlertChannel::Webhook(url) => {
                    // Would implement webhook posting
                }
                AlertChannel::Email(address) => {
                    // Would implement email sending
                }
                AlertChannel::Slack(webhook) => {
                    // Would implement Slack notification
                }
            }
        }
    }

    pub async fn record_validation(&self, result: &ValidationResult) -> Result<()> {
        let metrics = MonitoringMetrics {
            timestamp: result.timestamp,
            symbol: result.symbol.clone(),
            quality_score: result.quality_score,
            issue_count: result.issues.len(),
            is_valid: result.is_valid,
        };
        
        self.metrics_buffer.write().await.push_back(metrics);
        Ok(())
    }

    pub async fn send_alert(&self, alert: QualityAlert) -> Result<()> {
        self.alert_queue.send(alert).await?;
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        *self.shutdown.write().await = true;
        Ok(())
    }
}

impl Clone for QualityMonitor {
    fn clone(&self) -> Self {
        let (alert_queue, alert_receiver) = mpsc::channel(1000);
        
        Self {
            config: self.config.clone(),
            metrics_buffer: self.metrics_buffer.clone(),
            alert_queue,
            alert_receiver: Arc::new(RwLock::new(alert_receiver)),
            shutdown: self.shutdown.clone(),
        }
    }
}

struct RateLimiter {
    max_per_minute: usize,
    recent_times: VecDeque<std::time::Instant>,
}

impl RateLimiter {
    fn new(max_per_minute: usize) -> Self {
        Self {
            max_per_minute,
            recent_times: VecDeque::new(),
        }
    }

    fn should_allow(&mut self) -> bool {
        let now = std::time::Instant::now();
        let one_minute_ago = now - std::time::Duration::from_secs(60);
        
        // Remove old entries
        while let Some(&front) = self.recent_times.front() {
            if front < one_minute_ago {
                self.recent_times.pop_front();
            } else {
                break;
            }
        }
        
        if self.recent_times.len() < self.max_per_minute {
            self.recent_times.push_back(now);
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringMetrics {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub quality_score: f64,
    pub issue_count: usize,
    pub is_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub severity: AlertSeverity,
    pub category: IssueCategory,
    pub message: String,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AlertSeverity {
    pub fn from_issue(issue_severity: &super::IssueSeverity) -> Self {
        match issue_severity {
            super::IssueSeverity::Low => Self::Low,
            super::IssueSeverity::Medium => Self::Medium,
            super::IssueSeverity::High => Self::High,
            super::IssueSeverity::Critical => Self::Critical,
        }
    }
}