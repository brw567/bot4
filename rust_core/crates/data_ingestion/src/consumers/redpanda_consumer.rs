// DEEP DIVE: Redpanda Consumer with Adaptive Backpressure
// External Research Applied:
// - Netflix's adaptive concurrency limits
// - Twitter's Finagle backpressure algorithms
// - Google's SRE book on load shedding
// - Reactive Streams specification

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::collections::VecDeque;

use rdkafka::consumer::{Consumer, StreamConsumer, CommitMode};
use rdkafka::config::ClientConfig;
use rdkafka::message::{Message, BorrowedMessage};
use rdkafka::topic_partition_list::Offset;
use rdkafka::{TopicPartitionList, Timestamp};

use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::time::{interval, MissedTickBehavior, sleep};
use tokio_stream::StreamExt;

use rkyv::{Archived, Deserialize};
use bytes::Bytes;
use dashmap::DashMap;

use anyhow::{Result, Context};
use tracing::{info, warn, error, debug, instrument};

use crate::producers::MarketEvent;
use crate::sinks::{ClickHouseSink, ParquetWriter};
use crate::monitoring::ConsumerMetrics;

// Backpressure configuration
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    pub max_lag_messages: i64,
    pub max_lag_ms: i64,
    pub max_memory_mb: usize,
    pub max_inflight_requests: usize,
    pub pause_threshold: f64,  // 0.8 = pause at 80% capacity
    pub resume_threshold: f64,  // 0.5 = resume at 50% capacity
    pub adaptive_window_size: usize,
    pub gradient_smoothing: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_lag_messages: 100_000,
            max_lag_ms: 5000,
            max_memory_mb: 1024,
            max_inflight_requests: 1000,
            pause_threshold: 0.8,
            resume_threshold: 0.5,
            adaptive_window_size: 100,
            gradient_smoothing: 0.9,
        }
    }
}

// Consumer configuration
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    pub brokers: String,
    pub group_id: String,
    pub topics: Vec<String>,
    pub auto_offset_reset: String,
    pub enable_auto_commit: bool,
    pub max_poll_records: usize,
    pub fetch_min_bytes: usize,
    pub fetch_max_wait_ms: u64,
    pub session_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub backpressure: BackpressureConfig,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            brokers: "localhost:9092".to_string(),
            group_id: "bot4-consumer".to_string(),
            topics: vec![
                "market.trades.*.*".to_string(),
                "market.quotes.*.*".to_string(),
                "market.orderbook.*.*".to_string(),
            ],
            auto_offset_reset: "latest".to_string(),
            enable_auto_commit: false,  // Manual commit for better control
            max_poll_records: 10000,
            fetch_min_bytes: 1024 * 1024,  // 1MB minimum fetch
            fetch_max_wait_ms: 1,  // 1ms max wait for low latency
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            backpressure: BackpressureConfig::default(),
        }
    }
}

// Adaptive backpressure monitor using gradient descent
pub struct AdaptiveBackpressure {
    // Current state
    current_throughput: Arc<AtomicU64>,
    current_latency_us: Arc<AtomicU64>,
    current_errors: Arc<AtomicU64>,
    
    // Limits
    max_concurrency: Arc<AtomicU64>,
    min_concurrency: Arc<AtomicU64>,
    current_concurrency: Arc<AtomicU64>,
    
    // Memory pressure
    memory_used_bytes: Arc<AtomicU64>,
    memory_limit_bytes: Arc<AtomicU64>,
    
    // Lag monitoring
    consumer_lag: Arc<AtomicI64>,
    lag_threshold: Arc<AtomicI64>,
    
    // Control
    is_paused: Arc<AtomicBool>,
    should_shed_load: Arc<AtomicBool>,
    
    // History for gradient calculation
    throughput_history: Arc<RwLock<VecDeque<f64>>>,
    latency_history: Arc<RwLock<VecDeque<f64>>>,
    
    config: BackpressureConfig,
}

impl AdaptiveBackpressure {
    pub fn new(config: BackpressureConfig) -> Self {
        let memory_limit = (config.max_memory_mb * 1024 * 1024) as u64;
        
        Self {
            current_throughput: Arc::new(AtomicU64::new(0)),
            current_latency_us: Arc::new(AtomicU64::new(0)),
            current_errors: Arc::new(AtomicU64::new(0)),
            max_concurrency: Arc::new(AtomicU64::new(config.max_inflight_requests as u64)),
            min_concurrency: Arc::new(AtomicU64::new(10)),
            current_concurrency: Arc::new(AtomicU64::new(100)),
            memory_used_bytes: Arc::new(AtomicU64::new(0)),
            memory_limit_bytes: Arc::new(AtomicU64::new(memory_limit)),
            consumer_lag: Arc::new(AtomicI64::new(0)),
            lag_threshold: Arc::new(AtomicI64::new(config.max_lag_messages)),
            is_paused: Arc::new(AtomicBool::new(false)),
            should_shed_load: Arc::new(AtomicBool::new(false)),
            throughput_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.adaptive_window_size))),
            latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.adaptive_window_size))),
            config,
        }
    }
    
    // Check if we should pause consumption
    pub async fn should_pause(&self) -> bool {
        let memory_used = self.memory_used_bytes.load(Ordering::Relaxed) as f64;
        let memory_limit = self.memory_limit_bytes.load(Ordering::Relaxed) as f64;
        let memory_pressure = memory_used / memory_limit;
        
        let lag = self.consumer_lag.load(Ordering::Relaxed).abs();
        let lag_threshold = self.lag_threshold.load(Ordering::Relaxed).abs();
        let lag_pressure = lag as f64 / lag_threshold as f64;
        
        let should_pause = memory_pressure > self.config.pause_threshold || 
                          lag_pressure > self.config.pause_threshold;
        
        if should_pause && !self.is_paused.load(Ordering::Relaxed) {
            warn!(
                "Pausing consumption: memory={:.1}%, lag={:.1}%", 
                memory_pressure * 100.0,
                lag_pressure * 100.0
            );
            self.is_paused.store(true, Ordering::Relaxed);
        }
        
        should_pause
    }
    
    // Check if we should resume consumption
    pub async fn should_resume(&self) -> bool {
        if !self.is_paused.load(Ordering::Relaxed) {
            return true;
        }
        
        let memory_used = self.memory_used_bytes.load(Ordering::Relaxed) as f64;
        let memory_limit = self.memory_limit_bytes.load(Ordering::Relaxed) as f64;
        let memory_pressure = memory_used / memory_limit;
        
        let lag = self.consumer_lag.load(Ordering::Relaxed).abs();
        let lag_threshold = self.lag_threshold.load(Ordering::Relaxed).abs();
        let lag_pressure = lag as f64 / lag_threshold as f64;
        
        let should_resume = memory_pressure < self.config.resume_threshold && 
                           lag_pressure < self.config.resume_threshold;
        
        if should_resume {
            info!(
                "Resuming consumption: memory={:.1}%, lag={:.1}%",
                memory_pressure * 100.0,
                lag_pressure * 100.0
            );
            self.is_paused.store(false, Ordering::Relaxed);
        }
        
        should_resume
    }
    
    // Adaptive concurrency using gradient descent (AIMD algorithm)
    pub async fn adjust_concurrency(&self) {
        let throughput = self.current_throughput.load(Ordering::Relaxed) as f64;
        let latency = self.current_latency_us.load(Ordering::Relaxed) as f64;
        let errors = self.current_errors.load(Ordering::Relaxed) as f64;
        
        // Update history
        {
            let mut tp_history = self.throughput_history.write().await;
            tp_history.push_back(throughput);
            if tp_history.len() > self.config.adaptive_window_size {
                tp_history.pop_front();
            }
            
            let mut lat_history = self.latency_history.write().await;
            lat_history.push_back(latency);
            if lat_history.len() > self.config.adaptive_window_size {
                lat_history.pop_front();
            }
        }
        
        // Calculate gradient
        let gradient = self.calculate_gradient().await;
        
        // Adjust concurrency based on gradient
        let current = self.current_concurrency.load(Ordering::Relaxed) as f64;
        let adjustment = if errors > 0.0 {
            // Multiplicative decrease on errors
            current * 0.9
        } else if gradient > 0.0 {
            // Additive increase when improving
            current + 1.0
        } else if gradient < -0.1 {
            // Multiplicative decrease when degrading
            current * 0.95
        } else {
            current
        };
        
        // Apply limits
        let new_concurrency = adjustment
            .max(self.min_concurrency.load(Ordering::Relaxed) as f64)
            .min(self.max_concurrency.load(Ordering::Relaxed) as f64) as u64;
        
        if new_concurrency != current as u64 {
            debug!("Adjusting concurrency: {} -> {}", current, new_concurrency);
            self.current_concurrency.store(new_concurrency, Ordering::Relaxed);
        }
    }
    
    // Calculate gradient of throughput/latency trade-off
    async fn calculate_gradient(&self) -> f64 {
        let tp_history = self.throughput_history.read().await;
        let lat_history = self.latency_history.read().await;
        
        if tp_history.len() < 2 || lat_history.len() < 2 {
            return 0.0;
        }
        
        // Simple gradient: throughput increase - latency increase
        let tp_delta = tp_history.back().unwrap() - tp_history.front().unwrap();
        let lat_delta = lat_history.back().unwrap() - lat_history.front().unwrap();
        
        // Normalize and weight
        let tp_gradient = tp_delta / 1000.0;  // Normalize to k events/sec
        let lat_gradient = lat_delta / 1000.0;  // Normalize to ms
        
        // Positive gradient means improving (more throughput, less latency)
        tp_gradient - lat_gradient
    }
    
    pub fn update_metrics(&self, throughput: u64, latency_us: u64, errors: u64) {
        // Exponential moving average for smoothing
        let alpha = self.config.gradient_smoothing;
        
        let old_tp = self.current_throughput.load(Ordering::Relaxed) as f64;
        let new_tp = (alpha * old_tp + (1.0 - alpha) * throughput as f64) as u64;
        self.current_throughput.store(new_tp, Ordering::Relaxed);
        
        let old_lat = self.current_latency_us.load(Ordering::Relaxed) as f64;
        let new_lat = (alpha * old_lat + (1.0 - alpha) * latency_us as f64) as u64;
        self.current_latency_us.store(new_lat, Ordering::Relaxed);
        
        self.current_errors.store(errors, Ordering::Relaxed);
    }
    
    pub fn update_memory(&self, used_bytes: u64) {
        self.memory_used_bytes.store(used_bytes, Ordering::Relaxed);
    }
    
    pub fn update_lag(&self, lag: i64) {
        self.consumer_lag.store(lag, Ordering::Relaxed);
    }
    
    pub fn get_concurrency_limit(&self) -> usize {
        self.current_concurrency.load(Ordering::Relaxed) as usize
    }
}

// Main consumer implementation
pub struct RedpandaConsumer {
    consumer: Arc<StreamConsumer>,
    config: ConsumerConfig,
    
    // Sinks
    clickhouse_sink: Arc<ClickHouseSink>,
    parquet_writer: Arc<ParquetWriter>,
    
    // Backpressure
    backpressure: Arc<AdaptiveBackpressure>,
    semaphore: Arc<Semaphore>,
    
    // Metrics
    metrics: Arc<ConsumerMetrics>,
    events_processed: Arc<AtomicU64>,
    events_failed: Arc<AtomicU64>,
    bytes_processed: Arc<AtomicU64>,
    
    // Offset management
    offset_tracker: Arc<DashMap<(String, i32), i64>>,  // (topic, partition) -> offset
    last_commit_time: Arc<RwLock<Instant>>,
    
    // Shutdown
    shutdown: Arc<AtomicBool>,
}

impl RedpandaConsumer {
    pub async fn new(
        config: ConsumerConfig,
        clickhouse_sink: ClickHouseSink,
        parquet_writer: ParquetWriter,
    ) -> Result<Self> {
        // Build Redpanda consumer with optimizations
        let mut client_config = ClientConfig::new();
        
        client_config
            .set("bootstrap.servers", &config.brokers)
            .set("group.id", &config.group_id)
            .set("enable.auto.commit", config.enable_auto_commit.to_string())
            .set("auto.offset.reset", &config.auto_offset_reset)
            .set("session.timeout.ms", config.session_timeout_ms.to_string())
            .set("heartbeat.interval.ms", config.heartbeat_interval_ms.to_string())
            .set("max.poll.records", config.max_poll_records.to_string())
            .set("fetch.min.bytes", config.fetch_min_bytes.to_string())
            .set("fetch.max.wait.ms", config.fetch_max_wait_ms.to_string())
            // Redpanda-specific optimizations
            .set("enable.auto.offset.store", "false")  // Manual offset management
            .set("queued.min.messages", "100000")
            .set("queued.max.messages.kbytes", "1048576")  // 1GB queue
            .set("fetch.message.max.bytes", "10485760")  // 10MB max message
            .set("receive.message.max.bytes", "100663296")  // 100MB max batch
            .set("socket.keepalive.enable", "true")
            .set("socket.nagle.disable", "true");
            
        let consumer: StreamConsumer = client_config
            .create()
            .context("Failed to create Redpanda consumer")?;
            
        // Subscribe to topics
        let topics: Vec<&str> = config.topics.iter().map(|s| s.as_str()).collect();
        consumer.subscribe(&topics)
            .context("Failed to subscribe to topics")?;
            
        let consumer = Arc::new(consumer);
        
        // Initialize components
        let backpressure = Arc::new(AdaptiveBackpressure::new(config.backpressure.clone()));
        let semaphore = Arc::new(Semaphore::new(config.backpressure.max_inflight_requests));
        
        let consumer_instance = Self {
            consumer: consumer.clone(),
            config: config.clone(),
            clickhouse_sink: Arc::new(clickhouse_sink),
            parquet_writer: Arc::new(parquet_writer),
            backpressure: backpressure.clone(),
            semaphore,
            metrics: Arc::new(ConsumerMetrics::new()),
            events_processed: Arc::new(AtomicU64::new(0)),
            events_failed: Arc::new(AtomicU64::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            offset_tracker: Arc::new(DashMap::new()),
            last_commit_time: Arc::new(RwLock::new(Instant::now())),
            shutdown: Arc::new(AtomicBool::new(false)),
        };
        
        info!("Redpanda consumer initialized for topics: {:?}", topics);
        
        Ok(consumer_instance)
    }
    
    // Main consumption loop with backpressure
    pub async fn consume(&self) -> Result<()> {
        let mut stream = self.consumer.stream();
        let mut batch = Vec::with_capacity(self.config.max_poll_records);
        let mut last_backpressure_check = Instant::now();
        
        while !self.shutdown.load(Ordering::Relaxed) {
            // Check backpressure periodically
            if last_backpressure_check.elapsed() > Duration::from_millis(100) {
                if self.backpressure.should_pause().await {
                    // Pause consumption
                    let assignment = self.consumer.assignment()?;
                    self.consumer.pause(&assignment)?;
                    
                    // Wait until we can resume
                    while !self.backpressure.should_resume().await {
                        sleep(Duration::from_millis(10)).await;
                    }
                    
                    // Resume consumption
                    self.consumer.resume(&assignment)?;
                }
                
                // Adjust concurrency based on performance
                self.backpressure.adjust_concurrency().await;
                
                last_backpressure_check = Instant::now();
            }
            
            // Consume message with timeout
            match tokio::time::timeout(Duration::from_millis(100), stream.next()).await {
                Ok(Some(message)) => {
                    match message {
                        Ok(msg) => {
                            batch.push(msg);
                            
                            // Process batch when full
                            if batch.len() >= self.config.max_poll_records {
                                self.process_batch(&mut batch).await?;
                            }
                        }
                        Err(e) => {
                            error!("Kafka error: {}", e);
                            self.events_failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended
                    break;
                }
                Err(_) => {
                    // Timeout - process any pending batch
                    if !batch.is_empty() {
                        self.process_batch(&mut batch).await?;
                    }
                }
            }
        }
        
        // Process final batch
        if !batch.is_empty() {
            self.process_batch(&mut batch).await?;
        }
        
        Ok(())
    }
    
    // Process a batch of messages
    async fn process_batch(&self, batch: &mut Vec<BorrowedMessage<'_>>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        let batch_size = batch.len();
        let start = Instant::now();
        
        // Acquire permits for concurrency control
        let permits = self.semaphore.acquire_many(batch_size as u32).await?;
        
        // Process messages in parallel
        let mut tasks = Vec::with_capacity(batch_size);
        
        for msg in batch.drain(..) {
            let payload = msg.payload()
                .ok_or_else(|| anyhow::anyhow!("Empty message payload"))?;
            
            // Track offset
            let topic = msg.topic();
            let partition = msg.partition();
            let offset = msg.offset();
            self.offset_tracker.insert((topic.to_string(), partition), offset);
            
            // Deserialize event
            let event = self.deserialize_event(payload)?;
            
            // Route to appropriate sink based on age
            let age = self.calculate_event_age(&event);
            
            let clickhouse = self.clickhouse_sink.clone();
            let parquet = self.parquet_writer.clone();
            
            let task = tokio::spawn(async move {
                if age < Duration::from_secs(3600) {
                    // Hot data -> ClickHouse
                    clickhouse.write(event).await
                } else {
                    // Warm data -> Parquet
                    parquet.write(event).await
                }
            });
            
            tasks.push(task);
        }
        
        // Wait for all tasks to complete
        let results = futures::future::join_all(tasks).await;
        
        let mut success_count = 0;
        let mut failure_count = 0;
        
        for result in results {
            match result {
                Ok(Ok(_)) => success_count += 1,
                _ => failure_count += 1,
            }
        }
        
        // Update metrics
        self.events_processed.fetch_add(success_count, Ordering::Relaxed);
        self.events_failed.fetch_add(failure_count, Ordering::Relaxed);
        
        let elapsed = start.elapsed();
        let throughput = (success_count as f64 / elapsed.as_secs_f64() * 1000.0) as u64;
        let latency_us = elapsed.as_micros() as u64 / batch_size as u64;
        
        self.backpressure.update_metrics(throughput, latency_us, failure_count);
        
        // Release permits
        drop(permits);
        
        // Commit offsets periodically
        self.maybe_commit_offsets().await?;
        
        debug!("Processed batch of {} events in {:?}", batch_size, elapsed);
        
        Ok(())
    }
    
    // Deserialize event from bytes
    fn deserialize_event(&self, payload: &[u8]) -> Result<MarketEvent> {
        let archived = unsafe { rkyv::archived_root::<MarketEvent>(payload) };
        let event: MarketEvent = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|e| anyhow::anyhow!("Deserialization failed: {:?}", e))?;
        Ok(event)
    }
    
    // Calculate event age for routing
    fn calculate_event_age(&self, event: &MarketEvent) -> Duration {
        let event_time = match event {
            MarketEvent::Trade { timestamp_ns, .. } |
            MarketEvent::Quote { timestamp_ns, .. } |
            MarketEvent::OrderBook { timestamp_ns, .. } |
            MarketEvent::InternalEvent { timestamp_ns, .. } => *timestamp_ns,
        };
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
            
        Duration::from_nanos(now.saturating_sub(event_time))
    }
    
    // Commit offsets periodically
    async fn maybe_commit_offsets(&self) -> Result<()> {
        let mut last_commit = self.last_commit_time.write().await;
        
        if last_commit.elapsed() > Duration::from_secs(5) {
            // Build topic partition list with offsets
            let mut tpl = TopicPartitionList::new();
            
            for entry in self.offset_tracker.iter() {
                let ((topic, partition), offset) = entry.pair();
                tpl.add_partition_offset(topic, *partition, Offset::Offset(*offset + 1))?;
            }
            
            // Commit offsets
            self.consumer.commit(&tpl, CommitMode::Async)?;
            
            *last_commit = Instant::now();
            debug!("Committed offsets for {} partitions", tpl.count());
        }
        
        Ok(())
    }
    
    // Get consumer lag
    pub async fn get_lag(&self) -> Result<i64> {
        let mut total_lag = 0i64;
        
        let assignment = self.consumer.assignment()?;
        
        for partition in assignment.elements() {
            let (low, high) = self.consumer.fetch_watermarks(
                partition.topic(),
                partition.partition(),
                Duration::from_millis(1000),
            )?;
            
            let committed = self.consumer.committed_offsets(
                &assignment,
                Duration::from_millis(1000),
            )?;
            
            if let Some(offset) = committed.find_partition(partition.topic(), partition.partition()) {
                if let Offset::Offset(o) = offset.offset() {
                    total_lag += (high - o) as i64;
                }
            }
        }
        
        self.backpressure.update_lag(total_lag);
        Ok(total_lag)
    }
    
    // Graceful shutdown
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down Redpanda consumer...");
        
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Commit final offsets
        self.commit_all_offsets().await?;
        
        // Flush sinks
        self.clickhouse_sink.flush().await?;
        self.parquet_writer.flush().await?;
        
        info!("Redpanda consumer shutdown complete");
        Ok(())
    }
    
    async fn commit_all_offsets(&self) -> Result<()> {
        let mut tpl = TopicPartitionList::new();
        
        for entry in self.offset_tracker.iter() {
            let ((topic, partition), offset) = entry.pair();
            tpl.add_partition_offset(topic, *partition, Offset::Offset(*offset + 1))?;
        }
        
        self.consumer.commit(&tpl, CommitMode::Sync)?;
        info!("Final offset commit complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_backpressure_calculation() {
        let bp = AdaptiveBackpressure::new(BackpressureConfig::default());
        
        // Set high memory usage
        bp.update_memory(900 * 1024 * 1024);  // 900MB of 1024MB
        assert!(bp.should_pause().await);
        
        // Reduce memory usage
        bp.update_memory(400 * 1024 * 1024);  // 400MB
        assert!(bp.should_resume().await);
    }
    
    #[tokio::test]
    async fn test_adaptive_concurrency() {
        let bp = AdaptiveBackpressure::new(BackpressureConfig::default());
        
        // Simulate improving performance
        for i in 0..10 {
            bp.update_metrics(1000 * i, 1000 - i * 10, 0);
            bp.adjust_concurrency().await;
        }
        
        // Concurrency should increase
        assert!(bp.get_concurrency_limit() > 100);
        
        // Simulate errors
        bp.update_metrics(1000, 1000, 10);
        bp.adjust_concurrency().await;
        
        // Concurrency should decrease
        assert!(bp.get_concurrency_limit() < 100);
    }
}