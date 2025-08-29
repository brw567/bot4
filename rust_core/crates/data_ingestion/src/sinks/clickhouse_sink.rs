// DEEP DIVE: ClickHouse Sink for Hot Data Storage
// External Research Applied:
// - Uber's ClickHouse deployment (petabyte scale)
// - Cloudflare's analytics architecture (100M req/sec)
// - Yandex Metrica patterns (20B events/day)
// - ByteDance's real-time OLAP system

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::{HashMap, VecDeque};

use clickhouse::{Client, Row, Compression};
use clickhouse::sql::Identifier;
use serde::{Serialize, Deserialize};

use tokio::sync::{Mutex, RwLock, Semaphore, mpsc};
use tokio::time::{interval, MissedTickBehavior};

use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use bytes::Bytes;
use ahash::AHashMap;
use parking_lot::RwLock as SyncRwLock;

use anyhow::{Result, Context};
use tracing::{info, warn, error, debug, instrument};

use crate::producers::{MarketEvent, TradeSide};
use crate::monitoring::ClickHouseMetrics;

// ClickHouse schema for market events
#[derive(Debug, Clone, Row, Serialize, Deserialize)]
/// TODO: Add docs
pub struct MarketEventRow {
    // Timestamp - ClickHouse will handle as DateTime64
    pub timestamp: DateTime<Utc>,
    
    // Core fields
    pub exchange: String,
    pub symbol: String,
    pub event_type: String,
    
    // Trade fields
    pub price: Option<f64>,
    pub quantity: Option<f64>,
    pub side: Option<String>,
    pub trade_id: Option<u64>,
    
    // Quote fields
    pub bid_price: Option<f64>,
    pub bid_quantity: Option<f64>,
    pub ask_price: Option<f64>,
    pub ask_quantity: Option<f64>,
    
    // Orderbook fields (stored as arrays)
    pub bid_prices: Vec<f64>,
    pub bid_quantities: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_quantities: Vec<f64>,
    pub sequence: Option<u64>,
    
    // Metadata
    pub received_at: DateTime<Utc>,
    pub processed_at: DateTime<Utc>,
    pub latency_us: u64,
}

// Configuration for ClickHouse sink
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ClickHouseConfig {
    pub url: String,
    pub database: String,
    pub table: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
    pub max_retries: usize,
    pub compression: CompressionType,
    pub pool_size: usize,
    pub ttl_hours: u32,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum CompressionType {
    None,
    Lz4,
    Lz4hc,
    Zstd,
}

impl Default for ClickHouseConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8123".to_string(),
            database: "bot4_trading".to_string(),
            table: "market_events".to_string(),
            username: None,
            password: None,
            batch_size: 10000,
            batch_timeout_ms: 100,  // 100ms for hot data
            max_retries: 3,
            compression: CompressionType::Lz4,
            pool_size: 10,
            ttl_hours: 1,  // Hot data retained for 1 hour
        }
    }
}

// Buffer for batching writes
struct WriteBuffer {
    events: Vec<MarketEventRow>,
    first_event_time: Option<Instant>,
    total_bytes: usize,
}

impl WriteBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            first_event_time: None,
            total_bytes: 0,
        }
    }
    
    fn add(&mut self, event: MarketEventRow) -> usize {
        if self.first_event_time.is_none() {
            self.first_event_time = Some(Instant::now());
        }
        
        // Estimate size (rough approximation)
        let size = std::mem::size_of::<MarketEventRow>() + 
                   event.exchange.len() + 
                   event.symbol.len() +
                   event.bid_prices.len() * 8 * 4;  // Arrays take more space
        
        self.events.push(event);
        self.total_bytes += size;
        size
    }
    
    fn should_flush(&self, max_size: usize, max_age: Duration) -> bool {
        if self.events.len() >= max_size {
            return true;
        }
        
        if let Some(first_time) = self.first_event_time {
            if first_time.elapsed() > max_age {
                return true;
            }
        }
        
        false
    }
    
    fn take(&mut self) -> Vec<MarketEventRow> {
        self.first_event_time = None;
        self.total_bytes = 0;
        std::mem::take(&mut self.events)
    }
}

// Main ClickHouse sink implementation
/// TODO: Add docs
pub struct ClickHouseSink {
    clients: Vec<Arc<Client>>,
    config: ClickHouseConfig,
    
    // Buffering
    write_buffer: Arc<Mutex<WriteBuffer>>,
    buffer_semaphore: Arc<Semaphore>,
    
    // Round-robin client selection
    client_index: Arc<AtomicU64>,
    
    // Metrics
    metrics: Arc<ClickHouseMetrics>,
    events_written: Arc<AtomicU64>,
    events_failed: Arc<AtomicU64>,
    bytes_written: Arc<AtomicU64>,
    write_latency_us: Arc<AtomicU64>,
    
    // Circuit breaker
    consecutive_failures: Arc<AtomicU64>,
    circuit_open: Arc<AtomicBool>,
    circuit_open_until: Arc<Mutex<Option<Instant>>>,
    
    // Shutdown
    shutdown: Arc<AtomicBool>,
    flush_tx: mpsc::Sender<()>,
    flush_rx: Arc<Mutex<mpsc::Receiver<()>>>,
}

impl ClickHouseSink {
    pub async fn new(config: ClickHouseConfig) -> Result<Self> {
        // Create connection pool
        let mut clients = Vec::with_capacity(config.pool_size);
        
        for _ in 0..config.pool_size {
            let mut client = Client::default()
                .with_url(&config.url)
                .with_database(&config.database);
                
            if let Some(ref username) = config.username {
                client = client.with_user(username);
            }
            
            if let Some(ref password) = config.password {
                client = client.with_password(password);
            }
            
            client = match config.compression {
                CompressionType::None => client.with_compression(Compression::None),
                CompressionType::Lz4 => client.with_compression(Compression::Lz4),
                CompressionType::Lz4hc => client.with_compression(Compression::Lz4Hc(9)),
                CompressionType::Zstd => client.with_compression(Compression::Zstd(3)),
            };
            
            clients.push(Arc::new(client));
        }
        
        // Create table if not exists
        Self::create_table(&clients[0], &config).await?;
        
        let (flush_tx, flush_rx) = mpsc::channel(10);
        
        let sink = Self {
            clients: clients.clone(),
            config: config.clone(),
            write_buffer: Arc::new(Mutex::new(WriteBuffer::new(config.batch_size * 2))),
            buffer_semaphore: Arc::new(Semaphore::new(config.batch_size * 10)),
            client_index: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(ClickHouseMetrics::new()),
            events_written: Arc::new(AtomicU64::new(0)),
            events_failed: Arc::new(AtomicU64::new(0)),
            bytes_written: Arc::new(AtomicU64::new(0)),
            write_latency_us: Arc::new(AtomicU64::new(0)),
            consecutive_failures: Arc::new(AtomicU64::new(0)),
            circuit_open: Arc::new(AtomicBool::new(false)),
            circuit_open_until: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            flush_tx,
            flush_rx: Arc::new(Mutex::new(flush_rx)),
        };
        
        // Start background flusher
        sink.start_background_flusher();
        
        // Start metrics reporter
        sink.start_metrics_reporter();
        
        info!("ClickHouse sink initialized with {} connections", config.pool_size);
        
        Ok(sink)
    }
    
    // Create table with optimal schema for hot data
    async fn create_table(client: &Client, config: &ClickHouseConfig) -> Result<()> {
        let query = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                timestamp DateTime64(9, 'UTC') CODEC(DoubleDelta, LZ4),
                exchange LowCardinality(String),
                symbol LowCardinality(String),
                event_type Enum8('trade' = 1, 'quote' = 2, 'orderbook' = 3, 'internal' = 4),
                
                -- Trade fields
                price Nullable(Float64) CODEC(Gorilla, LZ4),
                quantity Nullable(Float64) CODEC(Gorilla, LZ4),
                side Nullable(Enum8('buy' = 1, 'sell' = 2)),
                trade_id Nullable(UInt64),
                
                -- Quote fields
                bid_price Nullable(Float64) CODEC(Gorilla, LZ4),
                bid_quantity Nullable(Float64) CODEC(Gorilla, LZ4),
                ask_price Nullable(Float64) CODEC(Gorilla, LZ4),
                ask_quantity Nullable(Float64) CODEC(Gorilla, LZ4),
                
                -- Orderbook arrays (top 25 levels)
                bid_prices Array(Float64) CODEC(Gorilla, LZ4),
                bid_quantities Array(Float64) CODEC(Gorilla, LZ4),
                ask_prices Array(Float64) CODEC(Gorilla, LZ4),
                ask_quantities Array(Float64) CODEC(Gorilla, LZ4),
                sequence Nullable(UInt64),
                
                -- Metadata
                received_at DateTime64(9, 'UTC'),
                processed_at DateTime64(9, 'UTC'),
                latency_us UInt64,
                
                -- Indexes for common queries
                INDEX idx_symbol symbol TYPE bloom_filter(0.01) GRANULARITY 4,
                INDEX idx_event_type event_type TYPE minmax GRANULARITY 4
            )
            ENGINE = MergeTree()
            PARTITION BY toStartOfHour(timestamp)
            ORDER BY (exchange, symbol, timestamp)
            PRIMARY KEY (exchange, symbol, timestamp)
            TTL timestamp + INTERVAL {} HOUR TO VOLUME 'warm_storage'
            SETTINGS 
                index_granularity = 8192,
                merge_with_ttl_timeout = 3600,
                parts_to_throw_insert = 3000,
                parts_to_delay_insert = 1500,
                max_parts_in_total = 10000
            "#,
            config.table,
            config.ttl_hours
        );
        
        client.query(&query).execute().await
            .context("Failed to create ClickHouse table")?;
            
        // Create materialized views for real-time aggregates
        Self::create_materialized_views(client, config).await?;
        
        info!("ClickHouse table {} created/verified", config.table);
        Ok(())
    }
    
    // Create materialized views for efficient aggregation
    async fn create_materialized_views(client: &Client, config: &ClickHouseConfig) -> Result<()> {
        // 1-second aggregates for ultra-fast queries
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS {}_1s
            ENGINE = AggregatingMergeTree()
            PARTITION BY toDate(timestamp)
            ORDER BY (exchange, symbol, toStartOfSecond(timestamp))
            AS SELECT
                exchange,
                symbol,
                toStartOfSecond(timestamp) as second,
                
                -- OHLCV
                argMinState(price, timestamp) as open,
                maxState(price) as high,
                minState(price) as low,
                argMaxState(price, timestamp) as close,
                sumState(quantity) as volume,
                
                -- Statistics
                countState() as trade_count,
                avgState(price) as avg_price,
                stddevPopState(price) as price_std,
                
                -- Spread
                avgState(ask_price - bid_price) as avg_spread,
                maxState(ask_price - bid_price) as max_spread
                
            FROM {}
            WHERE event_type = 'trade'
            GROUP BY exchange, symbol, second
            "#,
            config.table,
            config.table
        );
        
        client.query(&query).execute().await?;
        
        // Trade flow imbalance for toxicity detection
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS {}_flow_imbalance
            ENGINE = SummingMergeTree()
            PARTITION BY toDate(timestamp)
            ORDER BY (exchange, symbol, toStartOfMinute(timestamp))
            AS SELECT
                exchange,
                symbol,
                toStartOfMinute(timestamp) as minute,
                
                sumIf(quantity, side = 'buy') as buy_volume,
                sumIf(quantity, side = 'sell') as sell_volume,
                sumIf(quantity * price, side = 'buy') as buy_value,
                sumIf(quantity * price, side = 'sell') as sell_value,
                
                -- Order Flow Imbalance (OFI)
                sumIf(quantity, side = 'buy') - sumIf(quantity, side = 'sell') as ofi_volume,
                
                -- VWAP by side
                sumIf(quantity * price, side = 'buy') / sumIf(quantity, side = 'buy') as buy_vwap,
                sumIf(quantity * price, side = 'sell') / sumIf(quantity, side = 'sell') as sell_vwap
                
            FROM {}
            WHERE event_type = 'trade'
            GROUP BY exchange, symbol, minute
            "#,
            config.table,
            config.table
        );
        
        client.query(&query).execute().await?;
        
        info!("ClickHouse materialized views created");
        Ok(())
    }
    
    // Write a market event to ClickHouse
    pub async fn write(&self, event: MarketEvent) -> Result<()> {
        // Check circuit breaker
        if self.circuit_open.load(Ordering::Relaxed) {
            if !self.check_circuit_breaker().await {
                return Err(anyhow::anyhow!("Circuit breaker is open"));
            }
        }
        
        // Convert to ClickHouse row
        let row = self.convert_to_row(event)?;
        
        // Add to buffer
        let mut buffer = self.write_buffer.lock().await;
        let size = buffer.add(row);
        
        // Check if we should flush
        if buffer.should_flush(
            self.config.batch_size,
            Duration::from_millis(self.config.batch_timeout_ms)
        ) {
            let batch = buffer.take();
            drop(buffer);  // Release lock before flushing
            
            // Flush in background
            self.flush_batch(batch).await?;
        }
        
        self.bytes_written.fetch_add(size as u64, Ordering::Relaxed);
        
        Ok(())
    }
    
    // Convert MarketEvent to ClickHouse row
    fn convert_to_row(&self, event: MarketEvent) -> Result<MarketEventRow> {
        let received_at = Utc::now();
        let processed_at = Utc::now();
        
        let (timestamp_ns, exchange, symbol, event_type) = match &event {
            MarketEvent::Trade { timestamp_ns, exchange, symbol, .. } => {
                (*timestamp_ns, exchange.clone(), symbol.clone(), "trade".to_string())
            }
            MarketEvent::Quote { timestamp_ns, exchange, symbol, .. } => {
                (*timestamp_ns, exchange.clone(), symbol.clone(), "quote".to_string())
            }
            MarketEvent::OrderBook { timestamp_ns, exchange, symbol, .. } => {
                (*timestamp_ns, exchange.clone(), symbol.clone(), "orderbook".to_string())
            }
            MarketEvent::InternalEvent { timestamp_ns, .. } => {
                (*timestamp_ns, "internal".to_string(), "".to_string(), "internal".to_string())
            }
        };
        
        let timestamp = DateTime::<Utc>::from_timestamp_nanos(timestamp_ns as i64);
        let latency_us = (processed_at.timestamp_nanos_opt().unwrap() - timestamp_ns as i64) / 1000;
        
        let row = match event {
            MarketEvent::Trade { price, quantity, side, trade_id, .. } => {
                MarketEventRow {
                    timestamp,
                    exchange,
                    symbol,
                    event_type,
                    price: Some(price),
                    quantity: Some(quantity),
                    side: Some(match side {
                        TradeSide::Buy => "buy".to_string(),
                        TradeSide::Sell => "sell".to_string(),
                    }),
                    trade_id: Some(trade_id),
                    bid_price: None,
                    bid_quantity: None,
                    ask_price: None,
                    ask_quantity: None,
                    bid_prices: vec![],
                    bid_quantities: vec![],
                    ask_prices: vec![],
                    ask_quantities: vec![],
                    sequence: None,
                    received_at,
                    processed_at,
                    latency_us: latency_us as u64,
                }
            }
            MarketEvent::Quote { bid_price, bid_quantity, ask_price, ask_quantity, .. } => {
                MarketEventRow {
                    timestamp,
                    exchange,
                    symbol,
                    event_type,
                    price: None,
                    quantity: None,
                    side: None,
                    trade_id: None,
                    bid_price: Some(bid_price),
                    bid_quantity: Some(bid_quantity),
                    ask_price: Some(ask_price),
                    ask_quantity: Some(ask_quantity),
                    bid_prices: vec![],
                    bid_quantities: vec![],
                    ask_prices: vec![],
                    ask_quantities: vec![],
                    sequence: None,
                    received_at,
                    processed_at,
                    latency_us: latency_us as u64,
                }
            }
            MarketEvent::OrderBook { bids, asks, sequence, .. } => {
                // Separate prices and quantities for columnar storage
                let (bid_prices, bid_quantities): (Vec<f64>, Vec<f64>) = 
                    bids.into_iter().take(25).unzip();  // Top 25 levels
                let (ask_prices, ask_quantities): (Vec<f64>, Vec<f64>) = 
                    asks.into_iter().take(25).unzip();
                    
                MarketEventRow {
                    timestamp,
                    exchange,
                    symbol,
                    event_type,
                    price: None,
                    quantity: None,
                    side: None,
                    trade_id: None,
                    bid_price: bid_prices.first().copied(),
                    bid_quantity: bid_quantities.first().copied(),
                    ask_price: ask_prices.first().copied(),
                    ask_quantity: ask_quantities.first().copied(),
                    bid_prices,
                    bid_quantities,
                    ask_prices,
                    ask_quantities,
                    sequence: Some(sequence),
                    received_at,
                    processed_at,
                    latency_us: latency_us as u64,
                }
            }
            MarketEvent::InternalEvent { .. } => {
                // Internal events stored minimally
                MarketEventRow {
                    timestamp,
                    exchange,
                    symbol,
                    event_type,
                    price: None,
                    quantity: None,
                    side: None,
                    trade_id: None,
                    bid_price: None,
                    bid_quantity: None,
                    ask_price: None,
                    ask_quantity: None,
                    bid_prices: vec![],
                    bid_quantities: vec![],
                    ask_prices: vec![],
                    ask_quantities: vec![],
                    sequence: None,
                    received_at,
                    processed_at,
                    latency_us: latency_us as u64,
                }
            }
        };
        
        Ok(row)
    }
    
    // Flush a batch to ClickHouse
    async fn flush_batch(&self, batch: Vec<MarketEventRow>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        let batch_size = batch.len();
        let start = Instant::now();
        
        // Select client round-robin
        let client_idx = self.client_index.fetch_add(1, Ordering::Relaxed) as usize;
        let client = &self.clients[client_idx % self.clients.len()];
        
        // Prepare insert query
        let mut insert = client.insert(&self.config.table)?;
        
        for row in batch {
            insert.write(&row).await?;
        }
        
        // Execute with retries
        let mut retries = 0;
        loop {
            match insert.end().await {
                Ok(_) => {
                    // Success
                    self.events_written.fetch_add(batch_size as u64, Ordering::Relaxed);
                    self.consecutive_failures.store(0, Ordering::Relaxed);
                    
                    let elapsed = start.elapsed();
                    self.write_latency_us.store(elapsed.as_micros() as u64, Ordering::Relaxed);
                    
                    debug!("Flushed {} events to ClickHouse in {:?}", batch_size, elapsed);
                    
                    return Ok(());
                }
                Err(e) if retries < self.config.max_retries => {
                    warn!("ClickHouse write failed (retry {}): {}", retries + 1, e);
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(100 * (1 << retries))).await;
                    
                    // Recreate insert for retry
                    insert = client.insert(&self.config.table)?;
                }
                Err(e) => {
                    // Final failure
                    error!("ClickHouse write failed after {} retries: {}", retries, e);
                    self.events_failed.fetch_add(batch_size as u64, Ordering::Relaxed);
                    
                    let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
                    if failures > 10 {
                        self.open_circuit_breaker().await;
                    }
                    
                    return Err(e.into());
                }
            }
        }
    }
    
    // Background flusher
    fn start_background_flusher(&self) {
        let buffer = self.write_buffer.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();
        let mut flush_rx = self.flush_rx.clone();
        
        let clients = self.clients.clone();
        let client_index = self.client_index.clone();
        let events_written = self.events_written.clone();
        let events_failed = self.events_failed.clone();
        let write_latency_us = self.write_latency_us.clone();
        let consecutive_failures = self.consecutive_failures.clone();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(config.batch_timeout_ms));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            
            while !shutdown.load(Ordering::Relaxed) {
                tokio::select! {
                    _ = ticker.tick() => {
                        // Check if buffer needs flushing
                        let batch = {
                            let mut buf = buffer.lock().await;
                            if buf.should_flush(
                                config.batch_size,
                                Duration::from_millis(config.batch_timeout_ms)
                            ) {
                                buf.take()
                            } else {
                                vec![]
                            }
                        };
                        
                        if !batch.is_empty() {
                            Self::flush_batch_static(
                                batch,
                                &clients,
                                &client_index,
                                &config,
                                &events_written,
                                &events_failed,
                                &write_latency_us,
                                &consecutive_failures,
                            ).await;
                        }
                    }
                    _ = flush_rx.lock().await.recv() => {
                        // Forced flush
                        let batch = buffer.lock().await.take();
                        if !batch.is_empty() {
                            Self::flush_batch_static(
                                batch,
                                &clients,
                                &client_index,
                                &config,
                                &events_written,
                                &events_failed,
                                &write_latency_us,
                                &consecutive_failures,
                            ).await;
                        }
                    }
                }
            }
        });
    }
    
    // Static flush method for background task
    async fn flush_batch_static(
        batch: Vec<MarketEventRow>,
        clients: &[Arc<Client>],
        client_index: &AtomicU64,
        config: &ClickHouseConfig,
        events_written: &AtomicU64,
        events_failed: &AtomicU64,
        write_latency_us: &AtomicU64,
        consecutive_failures: &AtomicU64,
    ) {
        let batch_size = batch.len();
        let start = Instant::now();
        
        let client_idx = client_index.fetch_add(1, Ordering::Relaxed) as usize;
        let client = &clients[client_idx % clients.len()];
        
        match client.insert(&config.table) {
            Ok(mut insert) => {
                for row in batch {
                    if let Err(e) = insert.write(&row).await {
                        error!("Failed to write row: {}", e);
                    }
                }
                
                match insert.end().await {
                    Ok(_) => {
                        events_written.fetch_add(batch_size as u64, Ordering::Relaxed);
                        consecutive_failures.store(0, Ordering::Relaxed);
                        
                        let elapsed = start.elapsed();
                        write_latency_us.store(elapsed.as_micros() as u64, Ordering::Relaxed);
                    }
                    Err(e) => {
                        error!("ClickHouse batch write failed: {}", e);
                        events_failed.fetch_add(batch_size as u64, Ordering::Relaxed);
                        consecutive_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(e) => {
                error!("Failed to create insert: {}", e);
                events_failed.fetch_add(batch_size as u64, Ordering::Relaxed);
            }
        }
    }
    
    // Circuit breaker
    async fn check_circuit_breaker(&self) -> bool {
        let mut open_until = self.circuit_open_until.lock().await;
        
        if let Some(until) = *open_until {
            if Instant::now() >= until {
                // Try to close
                self.circuit_open.store(false, Ordering::Relaxed);
                *open_until = None;
                info!("ClickHouse circuit breaker closed");
                return true;
            }
            return false;
        }
        
        true
    }
    
    async fn open_circuit_breaker(&self) {
        self.circuit_open.store(true, Ordering::Relaxed);
        let mut open_until = self.circuit_open_until.lock().await;
        *open_until = Some(Instant::now() + Duration::from_secs(60));
        warn!("ClickHouse circuit breaker opened for 60 seconds");
    }
    
    // Flush any pending data
    pub async fn flush(&self) -> Result<()> {
        self.flush_tx.send(()).await?;
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }
    
    // Metrics reporter
    fn start_metrics_reporter(&self) {
        let metrics = self.metrics.clone();
        let events_written = self.events_written.clone();
        let events_failed = self.events_failed.clone();
        let bytes_written = self.bytes_written.clone();
        let write_latency_us = self.write_latency_us.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(10));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            
            let mut last_events = 0u64;
            let mut last_bytes = 0u64;
            
            while !shutdown.load(Ordering::Relaxed) {
                ticker.tick().await;
                
                let current_events = events_written.load(Ordering::Relaxed);
                let current_bytes = bytes_written.load(Ordering::Relaxed);
                let failed = events_failed.load(Ordering::Relaxed);
                let latency = write_latency_us.load(Ordering::Relaxed);
                
                let events_per_sec = (current_events - last_events) / 10;
                let bytes_per_sec = (current_bytes - last_bytes) / 10;
                
                metrics.update_throughput(events_per_sec, bytes_per_sec);
                metrics.update_latency(latency);
                
                info!(
                    "ClickHouse: {} events/sec, {} MB/sec, {}μs latency, {} total, {} failed",
                    events_per_sec,
                    bytes_per_sec / 1_000_000,
                    latency,
                    current_events,
                    failed
                );
                
                last_events = current_events;
                last_bytes = current_bytes;
            }
        });
    }
    
    // Query methods for hot data access
    pub async fn query_recent_trades(
        &self,
        exchange: &str,
        symbol: &str,
        seconds: u64,
    ) -> Result<Vec<MarketEventRow>> {
        let client = &self.clients[0];
        
        let query = format!(
            r#"
            SELECT *
            FROM {}
            WHERE exchange = ? AND symbol = ? 
                AND timestamp > now() - INTERVAL ? SECOND
                AND event_type = 'trade'
            ORDER BY timestamp DESC
            LIMIT 10000
            "#,
            self.config.table
        );
        
        let rows = client
            .query(&query)
            .bind(exchange)
            .bind(symbol)
            .bind(seconds)
            .fetch_all::<MarketEventRow>()
            .await?;
            
        Ok(rows)
    }
    
    // Get current orderbook snapshot
    pub async fn get_orderbook_snapshot(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<Option<MarketEventRow>> {
        let client = &self.clients[0];
        
        let query = format!(
            r#"
            SELECT *
            FROM {}
            WHERE exchange = ? AND symbol = ? 
                AND event_type = 'orderbook'
            ORDER BY timestamp DESC
            LIMIT 1
            "#,
            self.config.table
        );
        
        let rows = client
            .query(&query)
            .bind(exchange)
            .bind(symbol)
            .fetch_all::<MarketEventRow>()
            .await?;
            
        Ok(rows.into_iter().next())
    }
    
    // Calculate real-time VWAP
    pub async fn calculate_vwap(
        &self,
        exchange: &str,
        symbol: &str,
        seconds: u64,
    ) -> Result<f64> {
        let client = &self.clients[0];
        
        let query = format!(
            r#"
            SELECT 
                sum(price * quantity) / sum(quantity) as vwap
            FROM {}
            WHERE exchange = ? AND symbol = ? 
                AND timestamp > now() - INTERVAL ? SECOND
                AND event_type = 'trade'
            "#,
            self.config.table
        );
        
        let vwap: Option<f64> = client
            .query(&query)
            .bind(exchange)
            .bind(symbol)
            .bind(seconds)
            .fetch_one()
            .await?;
            
        Ok(vwap.unwrap_or(0.0))
    }
    
    // Graceful shutdown
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down ClickHouse sink...");
        
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Final flush
        self.flush().await?;
        
        // Wait a bit for background tasks
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        info!("ClickHouse sink shutdown complete");
        Ok(())
    }
    
    // Performance monitoring
    pub fn update_latency(&self, latency_us: u64) {
        self.write_latency_us.store(latency_us, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_clickhouse_sink_creation() {
        let config = ClickHouseConfig::default();
        let sink = ClickHouseSink::new(config).await;
        assert!(sink.is_ok());
    }
    
    #[tokio::test]
    async fn test_event_conversion() {
        let event = MarketEvent::Trade {
            exchange: "binance".to_string(),
            symbol: "BTC-USDT".to_string(),
            price: 50000.0,
            quantity: 0.1,
            side: TradeSide::Buy,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            trade_id: 12345,
        };
        
        let config = ClickHouseConfig::default();
        let sink = ClickHouseSink::new(config).await.unwrap();
        let row = sink.convert_to_row(event).unwrap();
        
        assert_eq!(row.exchange, "binance");
        assert_eq!(row.symbol, "BTC-USDT");
        assert_eq!(row.event_type, "trade");
        assert_eq!(row.price, Some(50000.0));
    }
}