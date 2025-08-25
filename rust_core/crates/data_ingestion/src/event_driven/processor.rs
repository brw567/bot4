// Event Processor - Core event-driven processing engine
// DEEP DIVE: Replaces fixed 10ms cadence with adaptive event processing
//
// References:
// - "The Art of Multiprocessor Programming" - Herlihy & Shavit (2020)
// - "Designing Data-Intensive Applications" - Kleppmann (2017)
// - LMAX Disruptor whitepaper
// - Chronicle Software's event sourcing patterns

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use tokio::sync::mpsc;
use tokio::time::{interval, sleep};
use crossbeam_channel::{bounded, unbounded, Sender, Receiver, TryRecvError};
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{debug, info, warn, error, instrument};

use crate::types::{Price, Quantity, Symbol};
use infrastructure::metrics::{MetricsCollector, register_counter, register_histogram};

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    Critical = 0,  // Market halt, circuit breaker
    High = 1,      // Large trades, microbursts
    Medium = 2,    // Normal trades
    Low = 3,       // Book updates
    Background = 4, // Statistics, monitoring
}

/// Processing result
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub event_id: u64,
    pub processing_time_ns: u64,
    pub queue_time_ns: u64,
    pub priority: EventPriority,
    pub success: bool,
    pub error: Option<String>,
}

/// Event to be processed
#[derive(Debug, Clone)]
pub struct Event {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub priority: EventPriority,
    pub payload: EventPayload,
    pub arrival_time: Instant,
}

/// Event payload types
#[derive(Debug, Clone)]
pub enum EventPayload {
    Trade {
        price: Price,
        quantity: Quantity,
        aggressor_side: TradeSide,
    },
    Quote {
        bid: Price,
        ask: Price,
        bid_size: Quantity,
        ask_size: Quantity,
    },
    BookUpdate {
        levels_changed: u32,
        total_depth: Quantity,
    },
    Microburst {
        severity: f64,
        duration_ms: u64,
    },
    MarketStatus {
        is_halted: bool,
        reason: Option<String>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Maximum events per second
    pub max_throughput: u64,
    
    /// Queue sizes per priority
    pub queue_sizes: [usize; 5],
    
    /// Processing threads
    pub worker_threads: usize,
    
    /// Enable batching
    pub enable_batching: bool,
    
    /// Batch size limits
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    
    /// Maximum queue latency before alert (microseconds)
    pub max_queue_latency_us: u64,
    
    /// Enable back-pressure
    pub enable_backpressure: bool,
    
    /// Monitoring interval
    pub monitoring_interval_ms: u64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            max_throughput: 1_000_000,  // 1M events/sec
            queue_sizes: [10_000, 50_000, 100_000, 100_000, 50_000],
            worker_threads: 8,
            enable_batching: true,
            min_batch_size: 10,
            max_batch_size: 1000,
            max_queue_latency_us: 100,  // 100 microseconds
            enable_backpressure: true,
            monitoring_interval_ms: 100,
        }
    }
}

/// Processor metrics
pub struct ProcessorMetrics {
    pub events_processed: Arc<dyn MetricsCollector>,
    pub events_dropped: Arc<dyn MetricsCollector>,
    pub processing_latency: Arc<dyn MetricsCollector>,
    pub queue_latency: Arc<dyn MetricsCollector>,
    pub batch_size: Arc<dyn MetricsCollector>,
    pub throughput: Arc<dyn MetricsCollector>,
}

/// Queue statistics
#[derive(Debug, Clone)]
struct QueueStats {
    pub depth: usize,
    pub max_latency_ns: u64,
    pub avg_latency_ns: u64,
    pub events_processed: u64,
    pub events_dropped: u64,
}

/// Worker thread handle
struct Worker {
    id: usize,
    handle: Option<tokio::task::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<RwLock<WorkerStats>>,
}

/// Worker statistics
#[derive(Debug, Default)]
struct WorkerStats {
    events_processed: u64,
    total_processing_ns: u64,
    last_event_time: Option<Instant>,
}

/// Main event processor implementation
pub struct EventProcessor {
    config: Arc<ProcessorConfig>,
    
    // Priority queues (crossbeam for performance)
    priority_queues: Arc<[Receiver<Event>; 5]>,
    priority_senders: Arc<[Sender<Event>; 5]>,
    
    // Event ID generator
    event_counter: Arc<AtomicU64>,
    
    // Workers
    workers: Arc<RwLock<Vec<Worker>>>,
    
    // Metrics
    metrics: Arc<ProcessorMetrics>,
    
    // Queue statistics
    queue_stats: Arc<DashMap<EventPriority, QueueStats>>,
    
    // Shutdown flag
    shutdown: Arc<AtomicBool>,
    
    // Rate limiter
    rate_limiter: Arc<RwLock<RateLimiter>>,
    
    // Event handlers
    handlers: Arc<DashMap<EventPriority, Vec<Arc<dyn EventHandler>>>>,
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    fn handle(&self, event: &Event) -> Result<()>;
    fn can_handle(&self, event: &Event) -> bool;
}

/// Rate limiter using token bucket
struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(max_throughput: u64) -> Self {
        let max = max_throughput as f64;
        Self {
            tokens: max,
            max_tokens: max,
            refill_rate: max,
            last_refill: Instant::now(),
        }
    }
    
    fn try_acquire(&mut self, count: usize) -> bool {
        // Refill tokens
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
        
        // Try to acquire
        let needed = count as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            true
        } else {
            false
        }
    }
}

impl EventProcessor {
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        // Create priority queues
        let mut receivers = Vec::with_capacity(5);
        let mut senders = Vec::with_capacity(5);
        
        for i in 0..5 {
            let (tx, rx) = bounded(config.queue_sizes[i]);
            receivers.push(rx);
            senders.push(tx);
        }
        
        let priority_queues = Arc::new(receivers.try_into().unwrap());
        let priority_senders = Arc::new(senders.try_into().unwrap());
        
        // Create metrics
        let metrics = Arc::new(ProcessorMetrics {
            events_processed: register_counter("event_processor_processed"),
            events_dropped: register_counter("event_processor_dropped"),
            processing_latency: register_histogram("event_processor_latency_ns"),
            queue_latency: register_histogram("event_processor_queue_latency_ns"),
            batch_size: register_histogram("event_processor_batch_size"),
            throughput: register_histogram("event_processor_throughput"),
        });
        
        Ok(Self {
            config: Arc::new(config.clone()),
            priority_queues,
            priority_senders,
            event_counter: Arc::new(AtomicU64::new(0)),
            workers: Arc::new(RwLock::new(Vec::new())),
            metrics,
            queue_stats: Arc::new(DashMap::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(config.max_throughput))),
            handlers: Arc::new(DashMap::new()),
        })
    }
    
    /// Submit event for processing
    #[instrument(skip(self, payload), fields(symbol = %symbol.0, priority = ?priority))]
    pub fn submit_event(
        &self,
        symbol: Symbol,
        priority: EventPriority,
        payload: EventPayload,
    ) -> Result<u64> {
        // Check shutdown
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Processor is shutting down"));
        }
        
        // Rate limiting
        if !self.rate_limiter.write().try_acquire(1) {
            self.metrics.events_dropped.increment(1);
            return Err(anyhow::anyhow!("Rate limit exceeded"));
        }
        
        // Generate event ID
        let id = self.event_counter.fetch_add(1, Ordering::Relaxed);
        
        // Create event
        let event = Event {
            id,
            timestamp: Utc::now(),
            symbol,
            priority,
            payload,
            arrival_time: Instant::now(),
        };
        
        // Submit to appropriate queue
        let queue_index = priority as usize;
        match self.priority_senders[queue_index].try_send(event) {
            Ok(_) => {
                debug!("Event {} submitted to priority {:?} queue", id, priority);
                Ok(id)
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.metrics.events_dropped.increment(1);
                
                // Apply back-pressure if enabled
                if self.config.enable_backpressure {
                    warn!("Queue full for priority {:?}, applying back-pressure", priority);
                    std::thread::sleep(Duration::from_micros(10));
                }
                
                Err(anyhow::anyhow!("Queue full"))
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(anyhow::anyhow!("Queue disconnected"))
            }
        }
    }
    
    /// Start processing with worker threads
    pub async fn start(&self) -> Result<()> {
        info!("Starting event processor with {} workers", self.config.worker_threads);
        
        let mut workers = self.workers.write();
        
        for i in 0..self.config.worker_threads {
            let worker = self.spawn_worker(i).await?;
            workers.push(worker);
        }
        
        // Start monitoring task
        self.spawn_monitor().await?;
        
        Ok(())
    }
    
    /// Spawn worker thread
    async fn spawn_worker(&self, id: usize) -> Result<Worker> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(RwLock::new(WorkerStats::default()));
        
        let shutdown_clone = shutdown.clone();
        let stats_clone = stats.clone();
        let queues = self.priority_queues.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let handlers = self.handlers.clone();
        
        let handle = tokio::task::spawn(async move {
            Self::worker_loop(
                id,
                queues,
                config,
                metrics,
                handlers,
                stats_clone,
                shutdown_clone,
            ).await;
        });
        
        Ok(Worker {
            id,
            handle: Some(handle),
            shutdown,
            stats,
        })
    }
    
    /// Worker processing loop
    async fn worker_loop(
        id: usize,
        queues: Arc<[Receiver<Event>; 5]>,
        config: Arc<ProcessorConfig>,
        metrics: Arc<ProcessorMetrics>,
        handlers: Arc<DashMap<EventPriority, Vec<Arc<dyn EventHandler>>>>,
        stats: Arc<RwLock<WorkerStats>>,
        shutdown: Arc<AtomicBool>,
    ) {
        info!("Worker {} started", id);
        let mut batch = Vec::with_capacity(config.max_batch_size);
        
        while !shutdown.load(Ordering::Acquire) {
            // Try to receive events from highest priority first
            let mut received = false;
            
            for (priority_level, queue) in queues.iter().enumerate() {
                // Batch receive for efficiency
                batch.clear();
                
                // Try to fill batch
                while batch.len() < config.max_batch_size {
                    match queue.try_recv() {
                        Ok(event) => {
                            batch.push(event);
                            received = true;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            error!("Queue disconnected");
                            return;
                        }
                    }
                }
                
                // Process batch if we have enough or if it's high priority
                if !batch.is_empty() && 
                   (batch.len() >= config.min_batch_size || priority_level <= 1) {
                    Self::process_batch(
                        &batch,
                        &handlers,
                        &metrics,
                        &stats,
                    ).await;
                    break; // Start from highest priority again
                }
            }
            
            // If no events, yield to avoid busy waiting
            if !received {
                tokio::time::sleep(Duration::from_micros(1)).await;
            }
        }
        
        info!("Worker {} stopped", id);
    }
    
    /// Process a batch of events
    async fn process_batch(
        batch: &[Event],
        handlers: &Arc<DashMap<EventPriority, Vec<Arc<dyn EventHandler>>>>,
        metrics: &Arc<ProcessorMetrics>,
        stats: &Arc<RwLock<WorkerStats>>,
    ) {
        let start = Instant::now();
        
        for event in batch {
            let process_start = Instant::now();
            
            // Calculate queue latency
            let queue_latency = process_start.duration_since(event.arrival_time).as_nanos() as u64;
            metrics.queue_latency.record(queue_latency as f64);
            
            // Get handlers for this priority
            if let Some(priority_handlers) = handlers.get(&event.priority) {
                for handler in priority_handlers.iter() {
                    if handler.can_handle(event) {
                        if let Err(e) = handler.handle(event) {
                            error!("Handler error for event {}: {}", event.id, e);
                        }
                    }
                }
            }
            
            // Record processing time
            let processing_time = process_start.elapsed().as_nanos() as u64;
            metrics.processing_latency.record(processing_time as f64);
            
            // Update stats
            let mut worker_stats = stats.write();
            worker_stats.events_processed += 1;
            worker_stats.total_processing_ns += processing_time;
            worker_stats.last_event_time = Some(Instant::now());
        }
        
        // Record batch metrics
        metrics.batch_size.record(batch.len() as f64);
        metrics.events_processed.increment(batch.len() as u64);
        
        let batch_time = start.elapsed();
        let throughput = batch.len() as f64 / batch_time.as_secs_f64();
        metrics.throughput.record(throughput);
    }
    
    /// Spawn monitoring task
    async fn spawn_monitor(&self) -> Result<()> {
        let interval_ms = self.config.monitoring_interval_ms;
        let queue_stats = self.queue_stats.clone();
        let workers = self.workers.clone();
        let queues = self.priority_queues.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(interval_ms));
            
            while !shutdown.load(Ordering::Acquire) {
                interval.tick().await;
                
                // Update queue statistics
                for (i, queue) in queues.iter().enumerate() {
                    let priority = match i {
                        0 => EventPriority::Critical,
                        1 => EventPriority::High,
                        2 => EventPriority::Medium,
                        3 => EventPriority::Low,
                        4 => EventPriority::Background,
                        _ => continue,
                    };
                    
                    let stats = QueueStats {
                        depth: queue.len(),
                        max_latency_ns: 0, // Would calculate from events
                        avg_latency_ns: 0,
                        events_processed: 0,
                        events_dropped: 0,
                    };
                    
                    queue_stats.insert(priority, stats);
                }
                
                // Log worker statistics
                let workers = workers.read();
                for worker in workers.iter() {
                    let stats = worker.stats.read();
                    if stats.events_processed > 0 {
                        let avg_processing = stats.total_processing_ns / stats.events_processed;
                        debug!(
                            "Worker {}: {} events, avg {}ns",
                            worker.id, stats.events_processed, avg_processing
                        );
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Register event handler
    pub fn register_handler(
        &self,
        priority: EventPriority,
        handler: Arc<dyn EventHandler>,
    ) {
        self.handlers
            .entry(priority)
            .or_insert_with(Vec::new)
            .push(handler);
    }
    
    /// Get current queue depth
    pub fn queue_depth(&self, priority: EventPriority) -> usize {
        let index = priority as usize;
        self.priority_queues[index].len()
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        let mut total_processed = 0u64;
        let mut total_processing_ns = 0u64;
        
        let workers = self.workers.read();
        for worker in workers.iter() {
            let stats = worker.stats.read();
            total_processed += stats.events_processed;
            total_processing_ns += stats.total_processing_ns;
        }
        
        let avg_processing_ns = if total_processed > 0 {
            total_processing_ns / total_processed
        } else {
            0
        };
        
        ProcessingStats {
            total_events: self.event_counter.load(Ordering::Relaxed),
            events_processed: total_processed,
            avg_processing_ns,
            queue_depths: [
                self.queue_depth(EventPriority::Critical),
                self.queue_depth(EventPriority::High),
                self.queue_depth(EventPriority::Medium),
                self.queue_depth(EventPriority::Low),
                self.queue_depth(EventPriority::Background),
            ],
        }
    }
    
    /// Shutdown processor
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down event processor");
        
        // Set shutdown flag
        self.shutdown.store(true, Ordering::Release);
        
        // Stop workers
        let mut workers = self.workers.write();
        for worker in workers.iter() {
            worker.shutdown.store(true, Ordering::Release);
        }
        
        // Wait for workers to finish
        for worker in workers.drain(..) {
            if let Some(handle) = worker.handle {
                handle.await?;
            }
        }
        
        info!("Event processor shutdown complete");
        Ok(())
    }
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_events: u64,
    pub events_processed: u64,
    pub avg_processing_ns: u64,
    pub queue_depths: [usize; 5],
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_event_processor_creation() {
        let config = ProcessorConfig::default();
        let processor = EventProcessor::new(config).unwrap();
        
        assert_eq!(processor.event_counter.load(Ordering::Relaxed), 0);
    }
    
    #[tokio::test]
    async fn test_event_submission() {
        let config = ProcessorConfig::default();
        let processor = EventProcessor::new(config).unwrap();
        
        let event_id = processor.submit_event(
            Symbol("BTC-USDT".to_string()),
            EventPriority::High,
            EventPayload::Trade {
                price: Price(Decimal::from(50000)),
                quantity: Quantity(Decimal::from(1)),
                aggressor_side: TradeSide::Buy,
            },
        ).unwrap();
        
        assert_eq!(event_id, 0);
        assert_eq!(processor.queue_depth(EventPriority::High), 1);
    }
    
    #[tokio::test]
    async fn test_priority_ordering() {
        let config = ProcessorConfig::default();
        let processor = EventProcessor::new(config).unwrap();
        
        // Submit events with different priorities
        processor.submit_event(
            Symbol("ETH-USDT".to_string()),
            EventPriority::Low,
            EventPayload::Quote {
                bid: Price(Decimal::from(2000)),
                ask: Price(Decimal::from(2001)),
                bid_size: Quantity(Decimal::from(10)),
                ask_size: Quantity(Decimal::from(10)),
            },
        ).unwrap();
        
        processor.submit_event(
            Symbol("BTC-USDT".to_string()),
            EventPriority::Critical,
            EventPayload::MarketStatus {
                is_halted: true,
                reason: Some("Circuit breaker triggered".to_string()),
            },
        ).unwrap();
        
        // Critical should be in different queue
        assert_eq!(processor.queue_depth(EventPriority::Critical), 1);
        assert_eq!(processor.queue_depth(EventPriority::Low), 1);
    }
}