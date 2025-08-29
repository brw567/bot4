// Feature Store Monitoring - Latency, Usage, and Health Tracking
// DEEP DIVE: Complete observability for feature serving

use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, instrument};
use prometheus::{Counter, Histogram, Gauge, HistogramOpts, register_histogram, register_counter, register_gauge};
use tokio::sync::RwLock;

use crate::{OnlineStore, OfflineStore};

/// Monitoring configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
// ELIMINATED: Duplicate MonitoringConfig - use infrastructure::monitoring::MonitoringConfig

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_port: 9090,
            collection_interval_seconds: 10,
            latency_buckets: vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            enable_prometheus: true,
            enable_logging: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct AlertThresholds {
    pub online_latency_p99_ms: f64,
    pub offline_latency_p99_ms: f64,
    pub error_rate_percent: f64,
    pub cache_miss_rate_percent: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            online_latency_p99_ms: 10.0,
            offline_latency_p99_ms: 100.0,
            error_rate_percent: 1.0,
            cache_miss_rate_percent: 50.0,
        }
    }
}

/// Feature monitoring system
/// TODO: Add docs
pub struct FeatureMonitor {
    config: MonitoringConfig,
    online_store: Arc<OnlineStore>,
    offline_store: Arc<OfflineStore>,
    
    // Prometheus metrics
    online_latency: Option<Histogram>,
    offline_latency: Option<Histogram>,
    feature_requests: Option<Counter>,
    feature_errors: Option<Counter>,
    cache_hits: Option<Counter>,
    cache_misses: Option<Counter>,
    active_features: Option<Gauge>,
    storage_bytes: Option<Gauge>,
    
    // Internal metrics
    metrics: Arc<RwLock<Metrics>>,
    
    // Collection task
    collector_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl FeatureMonitor {
    /// Create new monitor
    pub async fn new(
        config: MonitoringConfig,
        online_store: Arc<OnlineStore>,
        offline_store: Arc<OfflineStore>,
    ) -> Result<Self> {
        info!("Initializing Feature Store Monitor");
        
        let (online_latency, offline_latency, feature_requests, 
             feature_errors, cache_hits, cache_misses, active_features, storage_bytes) = 
            if config.enable_prometheus {
                (
                    Some(register_histogram!(
                        HistogramOpts::new("feature_online_latency_ms", "Online serving latency")
                            .buckets(config.latency_buckets.clone())
                    )?),
                    Some(register_histogram!(
                        HistogramOpts::new("feature_offline_latency_ms", "Offline serving latency")
                            .buckets(config.latency_buckets.clone())
                    )?),
                    Some(register_counter!("feature_requests_total", "Total feature requests")?),
                    Some(register_counter!("feature_errors_total", "Total feature errors")?),
                    Some(register_counter!("feature_cache_hits_total", "Cache hits")?),
                    Some(register_counter!("feature_cache_misses_total", "Cache misses")?),
                    Some(register_gauge!("active_features", "Number of active features")?),
                    Some(register_gauge!("feature_storage_bytes", "Storage used in bytes")?),
                )
            } else {
                (None, None, None, None, None, None, None, None)
            };
        
        let monitor = Self {
            config,
            online_store,
            offline_store,
            online_latency,
            offline_latency,
            feature_requests,
            feature_errors,
            cache_hits,
            cache_misses,
            active_features,
            storage_bytes,
            metrics: Arc::new(RwLock::new(Metrics::default())),
            collector_handle: Arc::new(RwLock::new(None)),
        };
        
        Ok(monitor)
    }
    
    /// Start metrics collection
    pub async fn start_collection(&self) -> Result<()> {
        info!("Starting metrics collection");
        
        if self.config.enable_prometheus {
            // Start Prometheus HTTP server
            self.start_prometheus_server().await?;
        }
        
        // Start periodic collection
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let online_store = self.online_store.clone();
        let active_features_gauge = self.active_features.clone();
        let storage_bytes_gauge = self.storage_bytes.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_secs(config.collection_interval_seconds)
            );
            
            loop {
                interval.tick().await;
                
                // Collect online store stats
                let online_stats = online_store.get_stats().await;
                
                let mut metrics_guard = metrics.write().await;
                metrics_guard.online_hit_rate = online_stats.hit_rate;
                metrics_guard.online_requests = online_stats.total_requests;
                metrics_guard.online_p50_latency_us = online_stats.p50_latency_us;
                metrics_guard.online_p99_latency_us = online_stats.p99_latency_us;
                
                // Update Prometheus gauges
                if let Some(gauge) = &active_features_gauge {
                    gauge.set(metrics_guard.active_features as f64);
                }
                
                if let Some(gauge) = &storage_bytes_gauge {
                    gauge.set(metrics_guard.storage_bytes as f64);
                }
                
                // Check alert conditions
                Self::check_alerts(&config.alert_thresholds, &metrics_guard);
            }
        });
        
        *self.collector_handle.write().await = Some(handle);
        Ok(())
    }
    
    /// Stop collection
    pub async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.collector_handle.write().await.take() {
            handle.abort();
        }
        Ok(())
    }
    
    /// Record serving latency
    pub async fn record_serving_latency(&self, latency: Duration) {
        if let Some(hist) = &self.online_latency {
            hist.observe(latency.as_secs_f64() * 1000.0);
        }
        
        let mut metrics = self.metrics.write().await;
        metrics.online_latencies.push(latency);
        if metrics.online_latencies.len() > 10000 {
            metrics.online_latencies.drain(0..5000);
        }
    }
    
    /// Record offline query latency
    pub async fn record_offline_latency(&self, latency: Duration) {
        if let Some(hist) = &self.offline_latency {
            hist.observe(latency.as_secs_f64() * 1000.0);
        }
        
        let mut metrics = self.metrics.write().await;
        metrics.offline_latencies.push(latency);
        if metrics.offline_latencies.len() > 10000 {
            metrics.offline_latencies.drain(0..5000);
        }
    }
    
    /// Record feature request
    pub async fn record_request(&self, success: bool) {
        if let Some(counter) = &self.feature_requests {
            counter.inc();
        }
        
        if !success {
            if let Some(counter) = &self.feature_errors {
                counter.inc();
            }
        }
        
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        if !success {
            metrics.error_count += 1;
        }
    }
    
    /// Record cache activity
    pub async fn record_cache(&self, hit: bool) {
        if hit {
            if let Some(counter) = &self.cache_hits {
                counter.inc();
            }
        } else {
            if let Some(counter) = &self.cache_misses {
                counter.inc();
            }
        }
        
        let mut metrics = self.metrics.write().await;
        if hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
    }
    
    /// Get online latency P99
    pub async fn get_online_latency_p99(&self) -> Result<f64> {
        let metrics = self.metrics.read().await;
        Ok(metrics.online_p99_latency_us as f64 / 1000.0) // Convert to ms
    }
    
    /// Get offline latency P99
    pub async fn get_offline_latency_p99(&self) -> Result<f64> {
        let metrics = self.metrics.read().await;
        let mut latencies: Vec<Duration> = metrics.offline_latencies.clone();
        if latencies.is_empty() {
            return Ok(0.0);
        }
        
        latencies.sort();
        let idx = (latencies.len() as f64 * 0.99) as usize;
        Ok(latencies[idx.min(latencies.len() - 1)].as_secs_f64() * 1000.0)
    }
    
    /// Get feature freshness P99
    pub async fn get_freshness_p99(&self) -> Result<f64> {
        // In production, would calculate from actual feature update times
        Ok(100.0) // milliseconds
    }
    
    /// Get storage usage
    pub async fn get_storage_usage(&self) -> Result<f64> {
        let metrics = self.metrics.read().await;
        Ok(metrics.storage_bytes as f64 / 1_073_741_824.0) // Convert to GB
    }
    
    /// Start Prometheus HTTP server
    async fn start_prometheus_server(&self) -> Result<()> {
        let port = self.config.metrics_port;
        
        tokio::spawn(async move {
            use warp::Filter;
            
            let metrics_route = warp::path("metrics")
                .map(move || {
                    use prometheus::Encoder;
                    let encoder = prometheus::TextEncoder::new();
                    let metric_families = prometheus::gather();
                    let mut buffer = Vec::new();
                    encoder.encode(&metric_families, &mut buffer).unwrap();
                    String::from_utf8(buffer).unwrap()
                });
            
            info!("Prometheus metrics available at http://0.0.0.0:{}/metrics", port);
            warp::serve(metrics_route)
                .run(([0, 0, 0, 0], port))
                .await;
        });
        
        Ok(())
    }
    
    /// Check alert conditions
    fn check_alerts(thresholds: &AlertThresholds, metrics: &Metrics) {
        // Check online latency
        let online_p99_ms = metrics.online_p99_latency_us as f64 / 1000.0;
        if online_p99_ms > thresholds.online_latency_p99_ms {
            warn!(
                "Online latency P99 ({:.2}ms) exceeds threshold ({:.2}ms)",
                online_p99_ms, thresholds.online_latency_p99_ms
            );
        }
        
        // Check error rate
        if metrics.total_requests > 0 {
            let error_rate = (metrics.error_count as f64 / metrics.total_requests as f64) * 100.0;
            if error_rate > thresholds.error_rate_percent {
                warn!(
                    "Error rate ({:.2}%) exceeds threshold ({:.2}%)",
                    error_rate, thresholds.error_rate_percent
                );
            }
        }
        
        // Check cache miss rate
        let total_cache = metrics.cache_hits + metrics.cache_misses;
        if total_cache > 0 {
            let miss_rate = (metrics.cache_misses as f64 / total_cache as f64) * 100.0;
            if miss_rate > thresholds.cache_miss_rate_percent {
                warn!(
                    "Cache miss rate ({:.2}%) exceeds threshold ({:.2}%)",
                    miss_rate, thresholds.cache_miss_rate_percent
                );
            }
        }
    }
}

/// Internal metrics storage
#[derive(Debug, Clone, Default)]
struct Metrics {
    // Online store
    online_hit_rate: f64,
    online_requests: u64,
    online_p50_latency_us: u64,
    online_p99_latency_us: u64,
    online_latencies: Vec<Duration>,
    
    // Offline store
    offline_latencies: Vec<Duration>,
    
    // Requests
    total_requests: u64,
    error_count: u64,
    
    // Cache
    cache_hits: u64,
    cache_misses: u64,
    
    // Features
    active_features: usize,
    storage_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monitoring_initialization() {
        let config = MonitoringConfig {
            enable_prometheus: false, // Disable for testing
            ..Default::default()
        };
        
        let online_store = Arc::new(OnlineStore::new(Default::default()).await.unwrap());
        let offline_store = Arc::new(OfflineStore::new(Default::default()).await.unwrap());
        
        let monitor = FeatureMonitor::new(config, online_store, offline_store).await;
        assert!(monitor.is_ok());
    }
    
    #[tokio::test]
    async fn test_latency_recording() {
        let config = MonitoringConfig {
            enable_prometheus: false,
            ..Default::default()
        };
        
        let online_store = Arc::new(OnlineStore::new(Default::default()).await.unwrap());
        let offline_store = Arc::new(OfflineStore::new(Default::default()).await.unwrap());
        
        let monitor = FeatureMonitor::new(config, online_store, offline_store).await.unwrap();
        
        // Record some latencies
        monitor.record_serving_latency(Duration::from_millis(5)).await;
        monitor.record_serving_latency(Duration::from_millis(10)).await;
        monitor.record_serving_latency(Duration::from_millis(15)).await;
        
        let metrics = monitor.metrics.read().await;
        assert_eq!(metrics.online_latencies.len(), 3);
    }
}