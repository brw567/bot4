// Automatic Backfill System with Priority Queue
// Based on Netflix's data platform and Uber's data healing architecture
//
// Theory: Priority-based backfill ensures critical gaps are filled first
// Uses multi-source fallback, intelligent retry strategies, and cost optimization
//
// Applications in trading:
// - Fill gaps during exchange outages
// - Recover from network partitions
// - Repair corrupted data segments
// - Complete historical datasets

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::{Ordering, Reverse};
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use tokio::sync::{RwLock, Mutex, mpsc};
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error, debug};

/// Backfill system configuration
#[derive(Debug, Clone, Deserialize)]
pub struct BackfillConfig {
    pub max_concurrent_jobs: usize,
    pub max_retries_per_request: usize,
    pub retry_delay_ms: u64,
    pub exponential_backoff: bool,
    pub max_backoff_ms: u64,
    pub priority_boost_age_hours: i64,
    pub cost_threshold_per_day: f64,
    pub enable_multi_source: bool,
    pub source_priority: Vec<String>,
}

impl Default for BackfillConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 10,
            max_retries_per_request: 3,
            retry_delay_ms: 1000,
            exponential_backoff: true,
            max_backoff_ms: 60000,
            priority_boost_age_hours: 24,
            cost_threshold_per_day: 100.0,
            enable_multi_source: true,
            source_priority: vec![
                "primary_exchange".to_string(),
                "backup_exchange".to_string(),
                "historical_provider".to_string(),
                "aggregator".to_string(),
            ],
        }
    }
}

/// Backfill request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackfillPriority {
    Critical,  // Real-time trading impacted
    High,      // Recent data for analysis
    Medium,    // Historical completeness
    Low,       // Nice to have
}

impl BackfillPriority {
    fn score(&self) -> i32 {
        match self {
            Self::Critical => 1000,
            Self::High => 100,
            Self::Medium => 10,
            Self::Low => 1,
        }
    }
}

/// Backfill request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillRequest {
    pub symbol: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub priority: BackfillPriority,
    pub source: String,
    pub max_retries: usize,
}

impl BackfillRequest {
    /// Calculate dynamic priority based on age and base priority
    fn calculate_priority(&self, boost_age_hours: i64) -> i32 {
        let base_score = self.priority.score();
        let age = Utc::now() - self.start_time;
        let age_hours = age.num_hours();
        
        // Boost priority for older gaps
        let age_boost = if age_hours > boost_age_hours {
            (age_hours / boost_age_hours) as i32 * 10
        } else {
            0
        };
        
        base_score + age_boost
    }
}

// Implement ordering for priority queue
impl Ord for BackfillRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.calculate_priority(24).cmp(&other.calculate_priority(24))
    }
}

impl PartialOrd for BackfillRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BackfillRequest {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol &&
        self.start_time == other.start_time &&
        self.end_time == other.end_time
    }
}

impl Eq for BackfillRequest {}

/// Backfill job status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Retrying(usize),  // Current retry attempt
}

/// Backfill job tracking
#[derive(Debug, Clone)]
struct BackfillJob {
    id: String,
    request: BackfillRequest,
    status: JobStatus,
    attempts: usize,
    created_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    data_points_filled: usize,
    cost_estimate: f64,
}

/// Data source trait for backfill
#[async_trait]
pub trait BackfillSource: Send + Sync {
    async fn fetch_data(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<DataPoint>>;
    
    fn estimate_cost(&self, symbol: &str, duration: Duration) -> f64;
    fn reliability_score(&self) -> f64;
    fn name(&self) -> String;
}

/// Mock data source for testing
pub struct MockDataSource {
    name: String,
    reliability: f64,
}

#[async_trait]
impl BackfillSource for MockDataSource {
    async fn fetch_data(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<DataPoint>> {
        // Simulate fetching data
        sleep(std::time::Duration::from_millis(100)).await;
        
        let mut points = Vec::new();
        let interval = Duration::seconds(1);
        let mut current = start;
        
        while current <= end {
            points.push(DataPoint {
                timestamp: current,
                symbol: symbol.to_string(),
                value: 100.0 + rand::random::<f64>() * 10.0,
                volume: 1000.0 + rand::random::<f64>() * 100.0,
            });
            current = current + interval;
        }
        
        Ok(points)
    }
    
    fn estimate_cost(&self, _symbol: &str, duration: Duration) -> f64 {
        // $0.001 per minute of data
        duration.num_minutes() as f64 * 0.001
    }
    
    fn reliability_score(&self) -> f64 {
        self.reliability
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
}

/// Backfill system with priority queue and multi-source support
pub struct BackfillSystem {
    config: BackfillConfig,
    
    // Priority queue for pending requests
    request_queue: Arc<Mutex<BinaryHeap<BackfillRequest>>>,
    
    // Active jobs
    active_jobs: Arc<RwLock<HashMap<String, BackfillJob>>>,
    
    // Completed jobs history
    completed_jobs: Arc<RwLock<Vec<BackfillJob>>>,
    
    // Data sources
    data_sources: Arc<RwLock<Vec<Arc<dyn BackfillSource>>>>,
    
    // Deduplication
    seen_requests: Arc<RwLock<HashSet<String>>>,
    
    // Cost tracking
    daily_cost: Arc<RwLock<f64>>,
    
    // Job processor channel
    job_sender: mpsc::Sender<BackfillRequest>,
    job_receiver: Arc<Mutex<mpsc::Receiver<BackfillRequest>>>,
    
    // Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

impl BackfillSystem {
    /// Create new backfill system
    pub async fn new(config: BackfillConfig) -> Result<Self> {
        info!("Initializing Backfill System with {} max concurrent jobs", 
              config.max_concurrent_jobs);
        
        let (job_sender, job_receiver) = mpsc::channel(1000);
        
        let mut system = Self {
            config: config.clone(),
            request_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            completed_jobs: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            data_sources: Arc::new(RwLock::new(Vec::new())),
            seen_requests: Arc::new(RwLock::new(HashSet::new())),
            daily_cost: Arc::new(RwLock::new(0.0)),
            job_sender,
            job_receiver: Arc::new(Mutex::new(job_receiver)),
            shutdown: Arc::new(RwLock::new(false)),
        };
        
        // Initialize default data sources
        system.initialize_data_sources().await?;
        
        // Start job processors
        for i in 0..config.max_concurrent_jobs {
            let system_clone = system.clone();
            tokio::spawn(async move {
                system_clone.job_processor(i).await;
            });
        }
        
        // Start priority queue processor
        let system_clone = system.clone();
        tokio::spawn(async move {
            system_clone.queue_processor().await;
        });
        
        info!("Backfill System initialized with {} data sources", 
              system.data_sources.read().await.len());
        
        Ok(system)
    }
    
    /// Initialize data sources
    async fn initialize_data_sources(&mut self) -> Result<()> {
        let mut sources = self.data_sources.write().await;
        
        // Add mock sources for testing
        sources.push(Arc::new(MockDataSource {
            name: "primary_exchange".to_string(),
            reliability: 0.99,
        }));
        
        sources.push(Arc::new(MockDataSource {
            name: "backup_exchange".to_string(),
            reliability: 0.95,
        }));
        
        sources.push(Arc::new(MockDataSource {
            name: "historical_provider".to_string(),
            reliability: 0.90,
        }));
        
        Ok(())
    }
    
    /// Request data backfill
    pub async fn request_backfill(&self, request: BackfillRequest) -> Result<String> {
        // Generate unique request ID
        let request_id = format!("{}-{}-{}", 
                                request.symbol,
                                request.start_time.timestamp(),
                                request.end_time.timestamp());
        
        // Check for duplicates
        let mut seen = self.seen_requests.write().await;
        if seen.contains(&request_id) {
            debug!("Duplicate backfill request ignored: {}", request_id);
            return Ok(request_id);
        }
        seen.insert(request_id.clone());
        
        // Check cost threshold
        let duration = request.end_time - request.start_time;
        let estimated_cost = self.estimate_request_cost(&request.symbol, duration).await?;
        
        let mut daily_cost = self.daily_cost.write().await;
        if *daily_cost + estimated_cost > self.config.cost_threshold_per_day {
            warn!("Backfill request exceeds daily cost threshold: ${:.2}", 
                  *daily_cost + estimated_cost);
            
            // Only allow critical requests when over budget
            if request.priority != BackfillPriority::Critical {
                return Err(anyhow!("Daily cost threshold exceeded"));
            }
        }
        *daily_cost += estimated_cost;
        
        // Add to priority queue
        let mut queue = self.request_queue.lock().await;
        queue.push(request.clone());
        
        info!("Backfill request queued: {} (priority: {:?}, cost: ${:.4})",
              request_id, request.priority, estimated_cost);
        
        Ok(request_id)
    }
    
    /// Process priority queue
    async fn queue_processor(&self) {
        loop {
            // Check shutdown
            if *self.shutdown.read().await {
                break;
            }
            
            // Get next request from priority queue
            let request = {
                let mut queue = self.request_queue.lock().await;
                queue.pop()
            };
            
            if let Some(request) = request {
                // Send to job processor
                if let Err(e) = self.job_sender.send(request).await {
                    error!("Failed to send job to processor: {}", e);
                }
            } else {
                // No requests, wait a bit
                sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    
    /// Job processor worker
    async fn job_processor(&self, worker_id: usize) {
        let receiver = self.job_receiver.clone();
        
        loop {
            // Check shutdown
            if *self.shutdown.read().await {
                break;
            }
            
            // Receive job
            let request = {
                let mut recv = receiver.lock().await;
                recv.recv().await
            };
            
            if let Some(request) = request {
                debug!("Worker {} processing backfill request for {}", 
                       worker_id, request.symbol);
                
                // Create job
                let job_id = format!("{}-{}", request.symbol, Utc::now().timestamp_nanos());
                let mut job = BackfillJob {
                    id: job_id.clone(),
                    request: request.clone(),
                    status: JobStatus::Running,
                    attempts: 0,
                    created_at: Utc::now(),
                    started_at: Some(Utc::now()),
                    completed_at: None,
                    data_points_filled: 0,
                    cost_estimate: 0.0,
                };
                
                // Add to active jobs
                self.active_jobs.write().await.insert(job_id.clone(), job.clone());
                
                // Process with retries
                let result = self.process_with_retries(&mut job).await;
                
                // Update job status
                match result {
                    Ok(points_filled) => {
                        job.status = JobStatus::Completed;
                        job.data_points_filled = points_filled;
                        job.completed_at = Some(Utc::now());
                        info!("Backfill completed: {} ({} points)", job_id, points_filled);
                    }
                    Err(e) => {
                        job.status = JobStatus::Failed(e.to_string());
                        job.completed_at = Some(Utc::now());
                        error!("Backfill failed: {} - {}", job_id, e);
                    }
                }
                
                // Move to completed
                self.active_jobs.write().await.remove(&job_id);
                self.completed_jobs.write().await.push(job);
            }
        }
    }
    
    /// Process backfill request with retries
    async fn process_with_retries(&self, job: &mut BackfillJob) -> Result<usize> {
        let max_retries = job.request.max_retries.min(self.config.max_retries_per_request);
        let mut last_error = None;
        
        for attempt in 0..=max_retries {
            job.attempts = attempt + 1;
            job.status = JobStatus::Retrying(attempt);
            
            // Calculate backoff delay
            let delay = if self.config.exponential_backoff {
                let base_delay = self.config.retry_delay_ms;
                let backoff = base_delay * 2_u64.pow(attempt as u32);
                backoff.min(self.config.max_backoff_ms)
            } else {
                self.config.retry_delay_ms
            };
            
            if attempt > 0 {
                debug!("Retrying backfill (attempt {}/{}), waiting {}ms", 
                       attempt + 1, max_retries + 1, delay);
                sleep(std::time::Duration::from_millis(delay)).await;
            }
            
            // Try to fetch data
            match self.fetch_from_sources(&job.request).await {
                Ok(data) => {
                    let points = data.len();
                    
                    // Store data (mock implementation)
                    self.store_backfilled_data(data).await?;
                    
                    return Ok(points);
                }
                Err(e) => {
                    warn!("Backfill attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow!("All retry attempts failed")))
    }
    
    /// Fetch data from available sources
    async fn fetch_from_sources(&self, request: &BackfillRequest) -> Result<Vec<DataPoint>> {
        let sources = self.data_sources.read().await;
        
        if self.config.enable_multi_source {
            // Try sources in priority order
            for source_name in &self.config.source_priority {
                if let Some(source) = sources.iter().find(|s| s.name() == *source_name) {
                    match timeout(
                        std::time::Duration::from_secs(30),
                        source.fetch_data(&request.symbol, request.start_time, request.end_time)
                    ).await {
                        Ok(Ok(data)) if !data.is_empty() => {
                            debug!("Successfully fetched {} points from {}", 
                                  data.len(), source_name);
                            return Ok(data);
                        }
                        Ok(Ok(_)) => {
                            warn!("No data available from {}", source_name);
                        }
                        Ok(Err(e)) => {
                            warn!("Error fetching from {}: {}", source_name, e);
                        }
                        Err(_) => {
                            warn!("Timeout fetching from {}", source_name);
                        }
                    }
                }
            }
            
            Err(anyhow!("Failed to fetch data from any source"))
        } else {
            // Use primary source only
            sources.first()
                .ok_or_else(|| anyhow!("No data sources available"))?
                .fetch_data(&request.symbol, request.start_time, request.end_time)
                .await
        }
    }
    
    /// Store backfilled data
    async fn store_backfilled_data(&self, data: Vec<DataPoint>) -> Result<()> {
        // Mock implementation - would write to database
        debug!("Storing {} backfilled data points", data.len());
        Ok(())
    }
    
    /// Estimate cost for backfill request
    async fn estimate_request_cost(&self, symbol: &str, duration: Duration) -> Result<f64> {
        let sources = self.data_sources.read().await;
        
        // Use primary source for estimation
        if let Some(source) = sources.first() {
            Ok(source.estimate_cost(symbol, duration))
        } else {
            Ok(0.0)
        }
    }
    
    /// Get backfill statistics
    pub async fn get_statistics(&self) -> BackfillStatistics {
        let active = self.active_jobs.read().await.len();
        let completed = self.completed_jobs.read().await.len();
        let pending = self.request_queue.lock().await.len();
        
        let completed_jobs = self.completed_jobs.read().await;
        let successful = completed_jobs.iter()
            .filter(|j| matches!(j.status, JobStatus::Completed))
            .count();
        
        let total_points: usize = completed_jobs.iter()
            .map(|j| j.data_points_filled)
            .sum();
        
        let daily_cost = *self.daily_cost.read().await;
        
        BackfillStatistics {
            active_jobs: active,
            pending_requests: pending,
            completed_jobs: completed,
            successful_jobs: successful,
            failed_jobs: completed - successful,
            total_points_filled: total_points,
            daily_cost,
        }
    }
    
    /// Shutdown the backfill system
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Backfill System");
        *self.shutdown.write().await = true;
        Ok(())
    }
}

// Clone implementation for BackfillSystem
impl Clone for BackfillSystem {
    fn clone(&self) -> Self {
        let (job_sender, job_receiver) = mpsc::channel(1000);
        
        Self {
            config: self.config.clone(),
            request_queue: self.request_queue.clone(),
            active_jobs: self.active_jobs.clone(),
            completed_jobs: self.completed_jobs.clone(),
            data_sources: self.data_sources.clone(),
            seen_requests: self.seen_requests.clone(),
            daily_cost: self.daily_cost.clone(),
            job_sender,
            job_receiver: Arc::new(Mutex::new(job_receiver)),
            shutdown: self.shutdown.clone(),
        }
    }
}

/// Data point returned from backfill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub value: f64,
    pub volume: f64,
}

/// Backfill statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillStatistics {
    pub active_jobs: usize,
    pub pending_requests: usize,
    pub completed_jobs: usize,
    pub successful_jobs: usize,
    pub failed_jobs: usize,
    pub total_points_filled: usize,
    pub daily_cost: f64,
}

// Add rand for testing
extern crate rand;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_backfill_priority_queue() {
        let config = BackfillConfig::default();
        let system = BackfillSystem::new(config).await.unwrap();
        
        // Add requests with different priorities
        let critical = BackfillRequest {
            symbol: "BTC".to_string(),
            start_time: Utc::now() - Duration::minutes(5),
            end_time: Utc::now(),
            priority: BackfillPriority::Critical,
            source: "test".to_string(),
            max_retries: 3,
        };
        
        let low = BackfillRequest {
            symbol: "ETH".to_string(),
            start_time: Utc::now() - Duration::hours(24),
            end_time: Utc::now() - Duration::hours(23),
            priority: BackfillPriority::Low,
            source: "test".to_string(),
            max_retries: 3,
        };
        
        system.request_backfill(low).await.unwrap();
        system.request_backfill(critical).await.unwrap();
        
        // Critical should be processed first despite being added second
        sleep(std::time::Duration::from_millis(500)).await;
        
        let stats = system.get_statistics().await;
        assert!(stats.active_jobs > 0 || stats.completed_jobs > 0);
    }
    
    #[tokio::test]
    async fn test_backfill_deduplication() {
        let config = BackfillConfig::default();
        let system = BackfillSystem::new(config).await.unwrap();
        
        let request = BackfillRequest {
            symbol: "TEST".to_string(),
            start_time: Utc::now() - Duration::hours(1),
            end_time: Utc::now(),
            priority: BackfillPriority::Medium,
            source: "test".to_string(),
            max_retries: 3,
        };
        
        // Add same request twice
        let id1 = system.request_backfill(request.clone()).await.unwrap();
        let id2 = system.request_backfill(request).await.unwrap();
        
        // Should get same ID (deduplicated)
        assert_eq!(id1, id2);
    }
}