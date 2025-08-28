use anyhow::{Result, Context};
use sqlx::{PgPool, PgConnection, Pool, Postgres, Row};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use tokio::time::{interval, Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use chrono::{DateTime, Utc, Datelike, Timelike, NaiveDateTime};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use dashmap::DashMap;
use tracing::{info, warn, error, debug, trace};
use crate::producers::{MarketEvent, TradeSide};
use futures::stream::{StreamExt, TryStreamExt};

/// Configuration for TimescaleDB aggregator
#[derive(Debug, Clone)]
pub struct TimescaleConfig {
    /// Database connection URL
    pub database_url: String,
    
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    
    /// Minimum number of connections to maintain
    pub min_connections: u32,
    
    /// Connection timeout in seconds
    pub connect_timeout: Duration,
    
    /// Idle timeout for connections
    pub idle_timeout: Duration,
    
    /// Max lifetime for a connection
    pub max_lifetime: Duration,
    
    /// Enable continuous aggregates
    pub enable_continuous_aggregates: bool,
    
    /// Enable compression policies
    pub enable_compression: bool,
    
    /// Retention policy in days (0 = no retention policy)
    pub retention_days: u32,
    
    /// Compression policy in days (compress data older than this)
    pub compress_after_days: u32,
    
    /// Batch size for bulk inserts
    pub batch_size: usize,
    
    /// Flush interval for batched writes
    pub flush_interval: Duration,
    
    /// Enable real-time aggregation
    pub enable_realtime_aggregation: bool,
    
    /// Candle intervals to generate
    pub candle_intervals: Vec<CandleInterval>,
    
    /// Enable volume profile calculation
    pub enable_volume_profile: bool,
    
    /// Enable order flow imbalance calculation
    pub enable_order_flow: bool,
    
    /// Enable VWAP calculation
    pub enable_vwap: bool,
}

impl Default for TimescaleConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://bot3user:bot3pass@localhost:5432/bot3trading".to_string(),
            max_connections: 16,
            min_connections: 2,
            connect_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(1800),
            enable_continuous_aggregates: true,
            enable_compression: true,
            retention_days: 90,  // 3 months retention
            compress_after_days: 7,  // Compress after 1 week
            batch_size: 1000,
            flush_interval: Duration::from_secs(1),
            enable_realtime_aggregation: true,
            candle_intervals: vec![
                CandleInterval::OneSecond,
                CandleInterval::FiveSeconds,
                CandleInterval::OneMinute,
                CandleInterval::FiveMinutes,
                CandleInterval::FifteenMinutes,
                CandleInterval::OneHour,
                CandleInterval::FourHours,
                CandleInterval::OneDay,
            ],
            enable_volume_profile: true,
            enable_order_flow: true,
            enable_vwap: true,
        }
    }
}

/// Candle intervals for aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleInterval {
    OneSecond,
    FiveSeconds,
    TenSeconds,
    ThirtySeconds,
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    FourHours,
    OneDay,
    OneWeek,
}

impl CandleInterval {
    /// Convert to PostgreSQL interval string
    pub fn to_interval_string(&self) -> &'static str {
        match self {
            Self::OneSecond => "1 second",
            Self::FiveSeconds => "5 seconds",
            Self::TenSeconds => "10 seconds",
            Self::ThirtySeconds => "30 seconds",
            Self::OneMinute => "1 minute",
            Self::FiveMinutes => "5 minutes",
            Self::FifteenMinutes => "15 minutes",
            Self::ThirtyMinutes => "30 minutes",
            Self::OneHour => "1 hour",
            Self::FourHours => "4 hours",
            Self::OneDay => "1 day",
            Self::OneWeek => "1 week",
        }
    }
    
    /// Convert to seconds
    pub fn to_seconds(&self) -> i64 {
        match self {
            Self::OneSecond => 1,
            Self::FiveSeconds => 5,
            Self::TenSeconds => 10,
            Self::ThirtySeconds => 30,
            Self::OneMinute => 60,
            Self::FiveMinutes => 300,
            Self::FifteenMinutes => 900,
            Self::ThirtyMinutes => 1800,
            Self::OneHour => 3600,
            Self::FourHours => 14400,
            Self::OneDay => 86400,
            Self::OneWeek => 604800,
        }
    }
    
    /// Get table suffix for this interval
    pub fn table_suffix(&self) -> &'static str {
        match self {
            Self::OneSecond => "1s",
            Self::FiveSeconds => "5s",
            Self::TenSeconds => "10s",
            Self::ThirtySeconds => "30s",
            Self::OneMinute => "1m",
            Self::FiveMinutes => "5m",
            Self::FifteenMinutes => "15m",
            Self::ThirtyMinutes => "30m",
            Self::OneHour => "1h",
            Self::FourHours => "4h",
            Self::OneDay => "1d",
            Self::OneWeek => "1w",
        }
    }
}

/// Real-time candle data
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub exchange: String,
    pub symbol: String,
    pub interval: CandleInterval,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub quote_volume: Decimal,  // Volume in quote currency (e.g., USDT)
    pub trades_count: i64,
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub vwap: Option<Decimal>,
    pub twap: Option<Decimal>,  // Time-weighted average price
}

/// Volume profile level
#[derive(Debug, Clone)]
pub struct VolumeLevel {
    pub price: Decimal,
    pub volume: Decimal,
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub trades: i64,
}

/// Order flow imbalance metrics
#[derive(Debug, Clone)]
pub struct OrderFlowMetrics {
    pub timestamp: DateTime<Utc>,
    pub exchange: String,
    pub symbol: String,
    pub interval: CandleInterval,
    pub delta: Decimal,  // Buy volume - Sell volume
    pub cumulative_delta: Decimal,
    pub imbalance_ratio: Decimal,  // Buy / (Buy + Sell)
    pub aggressor_ratio: Decimal,  // Aggressive orders ratio
    pub absorption_ratio: Decimal,  // Passive absorption ratio
}

/// Event batch for bulk insertion
struct EventBatch {
    events: Vec<MarketEvent>,
    created_at: Instant,
}

impl EventBatch {
    fn new() -> Self {
        Self {
            events: Vec::with_capacity(1000),
            created_at: Instant::now(),
        }
    }
    
    fn add(&mut self, event: MarketEvent) {
        self.events.push(event);
    }
    
    fn should_flush(&self, config: &TimescaleConfig) -> bool {
        self.events.len() >= config.batch_size ||
        self.created_at.elapsed() >= config.flush_interval
    }
    
    fn clear(&mut self) {
        self.events.clear();
        self.created_at = Instant::now();
    }
}

/// Metrics for monitoring
pub struct AggregatorMetrics {
    pub events_processed: AtomicU64,
    pub candles_generated: AtomicU64,
    pub insert_latency_us: AtomicU64,
    pub aggregation_latency_us: AtomicU64,
    pub batch_flushes: AtomicU64,
    pub compression_runs: AtomicU64,
    pub query_cache_hits: AtomicU64,
    pub query_cache_misses: AtomicU64,
}

impl AggregatorMetrics {
    fn new() -> Self {
        Self {
            events_processed: AtomicU64::new(0),
            candles_generated: AtomicU64::new(0),
            insert_latency_us: AtomicU64::new(0),
            aggregation_latency_us: AtomicU64::new(0),
            batch_flushes: AtomicU64::new(0),
            compression_runs: AtomicU64::new(0),
            query_cache_hits: AtomicU64::new(0),
            query_cache_misses: AtomicU64::new(0),
        }
    }
}

/// Cache for recent candles
struct CandleCache {
    candles: Arc<DashMap<String, Arc<RwLock<VecDeque<Candle>>>>>,
    max_size_per_key: usize,
}

impl CandleCache {
    fn new(max_size_per_key: usize) -> Self {
        Self {
            candles: Arc::new(DashMap::new()),
            max_size_per_key,
        }
    }
    
    async fn get(&self, key: &str, count: usize) -> Option<Vec<Candle>> {
        if let Some(entry) = self.candles.get(key) {
            let candles = entry.read().await;
            let result: Vec<Candle> = candles.iter()
                .take(count.min(candles.len()))
                .cloned()
                .collect();
            if !result.is_empty() {
                return Some(result);
            }
        }
        None
    }
    
    async fn insert(&self, key: String, candle: Candle) {
        let entry = self.candles.entry(key).or_insert_with(|| {
            Arc::new(RwLock::new(VecDeque::with_capacity(self.max_size_per_key)))
        });
        
        let mut candles = entry.write().await;
        candles.push_front(candle);
        
        // Maintain max size
        while candles.len() > self.max_size_per_key {
            candles.pop_back();
        }
    }
}

/// Main TimescaleDB aggregator implementation
pub struct TimescaleAggregator {
    config: Arc<TimescaleConfig>,
    pool: Arc<PgPool>,
    batch: Arc<Mutex<EventBatch>>,
    metrics: Arc<AggregatorMetrics>,
    cache: Arc<CandleCache>,
    shutdown: Arc<AtomicBool>,
    flush_semaphore: Arc<Semaphore>,
    flush_handle: Option<tokio::task::JoinHandle<()>>,
    aggregation_handle: Option<tokio::task::JoinHandle<()>>,
    compression_handle: Option<tokio::task::JoinHandle<()>>,
}

impl TimescaleAggregator {
    /// Create a new TimescaleDB aggregator
    pub async fn new(config: TimescaleConfig) -> Result<Self> {
        // Create database pool
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.connect_timeout)
            .idle_timeout(Some(config.idle_timeout))
            .max_lifetime(Some(config.max_lifetime))
            .connect(&config.database_url)
            .await
            .context("Failed to create database pool")?;
        
        let aggregator = Arc::new(Self {
            config: Arc::new(config.clone()),
            pool: Arc::new(pool),
            batch: Arc::new(Mutex::new(EventBatch::new())),
            metrics: Arc::new(AggregatorMetrics::new()),
            cache: Arc::new(CandleCache::new(100)),  // Cache last 100 candles per key
            shutdown: Arc::new(AtomicBool::new(false)),
            flush_semaphore: Arc::new(Semaphore::new(1)),
            flush_handle: None,
            aggregation_handle: None,
            compression_handle: None,
        });
        
        // Initialize database schema
        aggregator.initialize_schema().await?;
        
        // Start background tasks
        let flush_handle = {
            let agg = aggregator.clone();
            tokio::spawn(async move {
                agg.flush_task().await;
            })
        };
        
        let aggregation_handle = if config.enable_realtime_aggregation {
            let agg = aggregator.clone();
            Some(tokio::spawn(async move {
                agg.aggregation_task().await;
            }))
        } else {
            None
        };
        
        let compression_handle = if config.enable_compression {
            let agg = aggregator.clone();
            Some(tokio::spawn(async move {
                agg.compression_task().await;
            }))
        } else {
            None
        };
        
        // Return with handles
        let mut_aggregator = Arc::try_unwrap(aggregator)
            .unwrap_or_else(|arc| (*arc).clone());
        
        Ok(Self {
            flush_handle: Some(flush_handle),
            aggregation_handle,
            compression_handle,
            ..mut_aggregator
        })
    }
    
    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        
        // Create TimescaleDB extension
        sqlx::query(
            "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"
        )
        .execute(&mut *tx)
        .await?;
        
        // Create main events table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS market_events (
                timestamp TIMESTAMPTZ NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                event_type VARCHAR(20) NOT NULL,
                price DECIMAL(20, 8),
                quantity DECIMAL(20, 8),
                side VARCHAR(10),
                trade_id VARCHAR(100),
                is_maker BOOLEAN,
                bid_price DECIMAL(20, 8),
                bid_quantity DECIMAL(20, 8),
                ask_price DECIMAL(20, 8),
                ask_quantity DECIMAL(20, 8),
                sequence_number BIGINT,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            "#
        )
        .execute(&mut *tx)
        .await?;
        
        // Convert to hypertable
        sqlx::query(
            "SELECT create_hypertable('market_events', 'timestamp', 
             chunk_time_interval => INTERVAL '1 hour',
             if_not_exists => TRUE)"
        )
        .execute(&mut *tx)
        .await?;
        
        // Create indexes
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_market_events_exchange_symbol_time 
             ON market_events (exchange, symbol, timestamp DESC)"
        )
        .execute(&mut *tx)
        .await?;
        
        // Create candle tables for each interval
        for interval in &self.config.candle_intervals {
            self.create_candle_table(&mut tx, *interval).await?;
        }
        
        // Create volume profile table
        if self.config.enable_volume_profile {
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS volume_profile (
                    timestamp TIMESTAMPTZ NOT NULL,
                    exchange VARCHAR(50) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    price_level DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) NOT NULL,
                    buy_volume DECIMAL(20, 8) NOT NULL,
                    sell_volume DECIMAL(20, 8) NOT NULL,
                    trades_count BIGINT NOT NULL,
                    PRIMARY KEY (timestamp, exchange, symbol, interval, price_level)
                )
                "#
            )
            .execute(&mut *tx)
            .await?;
            
            // Convert to hypertable
            sqlx::query(
                "SELECT create_hypertable('volume_profile', 'timestamp',
                 chunk_time_interval => INTERVAL '1 day',
                 if_not_exists => TRUE)"
            )
            .execute(&mut *tx)
            .await?;
        }
        
        // Create order flow table
        if self.config.enable_order_flow {
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS order_flow_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    exchange VARCHAR(50) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    delta DECIMAL(20, 8) NOT NULL,
                    cumulative_delta DECIMAL(20, 8) NOT NULL,
                    imbalance_ratio DECIMAL(10, 6) NOT NULL,
                    aggressor_ratio DECIMAL(10, 6) NOT NULL,
                    absorption_ratio DECIMAL(10, 6) NOT NULL,
                    PRIMARY KEY (timestamp, exchange, symbol, interval)
                )
                "#
            )
            .execute(&mut *tx)
            .await?;
            
            sqlx::query(
                "SELECT create_hypertable('order_flow_metrics', 'timestamp',
                 chunk_time_interval => INTERVAL '1 day',
                 if_not_exists => TRUE)"
            )
            .execute(&mut *tx)
            .await?;
        }
        
        // Setup compression policies if enabled
        if self.config.enable_compression {
            self.setup_compression_policies(&mut tx).await?;
        }
        
        // Setup retention policies if configured
        if self.config.retention_days > 0 {
            self.setup_retention_policies(&mut tx).await?;
        }
        
        tx.commit().await?;
        
        info!("TimescaleDB schema initialized successfully");
        Ok(())
    }
    
    /// Create candle table for a specific interval
    async fn create_candle_table(
        &self,
        tx: &mut sqlx::Transaction<'_, Postgres>,
        interval: CandleInterval,
    ) -> Result<()> {
        let table_name = format!("candles_{}", interval.table_suffix());
        
        // Create table
        let create_query = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                timestamp TIMESTAMPTZ NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                open DECIMAL(20, 8) NOT NULL,
                high DECIMAL(20, 8) NOT NULL,
                low DECIMAL(20, 8) NOT NULL,
                close DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(20, 8) NOT NULL,
                quote_volume DECIMAL(20, 8) NOT NULL,
                trades_count BIGINT NOT NULL,
                buy_volume DECIMAL(20, 8) NOT NULL,
                sell_volume DECIMAL(20, 8) NOT NULL,
                vwap DECIMAL(20, 8),
                twap DECIMAL(20, 8),
                PRIMARY KEY (timestamp, exchange, symbol)
            )
            "#,
            table_name
        );
        sqlx::query(&create_query).execute(&mut **tx).await?;
        
        // Convert to hypertable
        let chunk_interval = match interval {
            CandleInterval::OneSecond | CandleInterval::FiveSeconds => "1 hour",
            CandleInterval::OneMinute | CandleInterval::FiveMinutes => "1 day",
            CandleInterval::FifteenMinutes | CandleInterval::ThirtyMinutes => "1 week",
            _ => "1 month",
        };
        
        let hypertable_query = format!(
            "SELECT create_hypertable('{}', 'timestamp',
             chunk_time_interval => INTERVAL '{}',
             if_not_exists => TRUE)",
            table_name, chunk_interval
        );
        sqlx::query(&hypertable_query).execute(&mut **tx).await?;
        
        // Create index
        let index_query = format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_exchange_symbol_time
             ON {} (exchange, symbol, timestamp DESC)",
            table_name, table_name
        );
        sqlx::query(&index_query).execute(&mut **tx).await?;
        
        // Create continuous aggregate if enabled
        if self.config.enable_continuous_aggregates {
            self.create_continuous_aggregate(tx, interval).await?;
        }
        
        Ok(())
    }
    
    /// Create continuous aggregate for real-time materialization
    async fn create_continuous_aggregate(
        &self,
        tx: &mut sqlx::Transaction<'_, Postgres>,
        interval: CandleInterval,
    ) -> Result<()> {
        // Skip for sub-minute intervals (too granular for continuous aggregates)
        if matches!(interval, CandleInterval::OneSecond | CandleInterval::FiveSeconds | CandleInterval::TenSeconds | CandleInterval::ThirtySeconds) {
            return Ok(());
        }
        
        let view_name = format!("candles_{}_cagg", interval.table_suffix());
        let interval_str = interval.to_interval_string();
        
        let create_cagg = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS {}
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('{}', timestamp) AS bucket,
                exchange,
                symbol,
                first(price, timestamp) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price, timestamp) AS close,
                sum(quantity) AS volume,
                sum(price * quantity) AS quote_volume,
                count(*) AS trades_count,
                sum(CASE WHEN side = 'buy' THEN quantity ELSE 0 END) AS buy_volume,
                sum(CASE WHEN side = 'sell' THEN quantity ELSE 0 END) AS sell_volume
            FROM market_events
            WHERE event_type = 'trade'
            GROUP BY bucket, exchange, symbol
            "#,
            view_name, interval_str
        );
        
        sqlx::query(&create_cagg).execute(&mut **tx).await?;
        
        // Add refresh policy
        let refresh_policy = format!(
            "SELECT add_continuous_aggregate_policy('{}',
             start_offset => INTERVAL '2 hours',
             end_offset => INTERVAL '1 minute',
             schedule_interval => INTERVAL '1 minute',
             if_not_exists => TRUE)",
            view_name
        );
        sqlx::query(&refresh_policy).execute(&mut **tx).await?;
        
        Ok(())
    }
    
    /// Setup compression policies
    async fn setup_compression_policies(
        &self,
        tx: &mut sqlx::Transaction<'_, Postgres>,
    ) -> Result<()> {
        let compress_after = format!("{} days", self.config.compress_after_days);
        
        // Compression for main events table
        sqlx::query(
            &format!(
                "SELECT add_compression_policy('market_events',
                 INTERVAL '{}',
                 if_not_exists => TRUE)",
                compress_after
            )
        )
        .execute(&mut **tx)
        .await?;
        
        // Compression for candle tables
        for interval in &self.config.candle_intervals {
            let table_name = format!("candles_{}", interval.table_suffix());
            sqlx::query(
                &format!(
                    "SELECT add_compression_policy('{}',
                     INTERVAL '{}',
                     if_not_exists => TRUE)",
                    table_name, compress_after
                )
            )
            .execute(&mut **tx)
            .await?;
        }
        
        Ok(())
    }
    
    /// Setup retention policies
    async fn setup_retention_policies(
        &self,
        tx: &mut sqlx::Transaction<'_, Postgres>,
    ) -> Result<()> {
        let retention = format!("{} days", self.config.retention_days);
        
        // Retention for main events table
        sqlx::query(
            &format!(
                "SELECT add_retention_policy('market_events',
                 INTERVAL '{}',
                 if_not_exists => TRUE)",
                retention
            )
        )
        .execute(&mut **tx)
        .await?;
        
        // Retention for candle tables
        for interval in &self.config.candle_intervals {
            let table_name = format!("candles_{}", interval.table_suffix());
            sqlx::query(
                &format!(
                    "SELECT add_retention_policy('{}',
                     INTERVAL '{}',
                     if_not_exists => TRUE)",
                    table_name, retention
                )
            )
            .execute(&mut **tx)
            .await?;
        }
        
        Ok(())
    }
    
    /// Process a market event
    pub async fn process_event(&self, event: MarketEvent) -> Result<()> {
        let start = Instant::now();
        
        // Add to batch
        {
            let mut batch = self.batch.lock().await;
            batch.add(event);
            
            if batch.should_flush(&self.config) {
                drop(batch);  // Release lock before flushing
                self.flush_batch().await?;
            }
        }
        
        self.metrics.events_processed.fetch_add(1, Ordering::Relaxed);
        
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.insert_latency_us.store(latency, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Flush current batch to database
    async fn flush_batch(&self) -> Result<()> {
        let _permit = self.flush_semaphore.acquire().await?;
        
        let events = {
            let mut batch = self.batch.lock().await;
            if batch.events.is_empty() {
                return Ok(());
            }
            let events = batch.events.clone();
            batch.clear();
            events
        };
        
        // Bulk insert events
        let mut tx = self.pool.begin().await?;
        
        for chunk in events.chunks(100) {
            let mut query_builder = sqlx::QueryBuilder::new(
                "INSERT INTO market_events (
                    timestamp, exchange, symbol, event_type, price, quantity,
                    side, trade_id, is_maker, bid_price, bid_quantity,
                    ask_price, ask_quantity, sequence_number
                ) "
            );
            
            query_builder.push_values(chunk, |mut b, event| {
                let ts = DateTime::<Utc>::from_timestamp_nanos(event.timestamp as i64);
                b.push_bind(ts)
                    .push_bind(&event.exchange)
                    .push_bind(&event.symbol)
                    .push_bind(&event.event_type)
                    .push_bind(event.price.map(|p| Decimal::from_f64(p).unwrap()))
                    .push_bind(event.quantity.map(|q| Decimal::from_f64(q).unwrap()))
                    .push_bind(event.side.as_ref().map(|s| match s {
                        TradeSide::Buy => "buy",
                        TradeSide::Sell => "sell",
                    }))
                    .push_bind(event.trade_id.as_ref())
                    .push_bind(event.is_maker)
                    .push_bind(event.bid_price.map(|p| Decimal::from_f64(p).unwrap()))
                    .push_bind(event.bid_quantity.map(|q| Decimal::from_f64(q).unwrap()))
                    .push_bind(event.ask_price.map(|p| Decimal::from_f64(p).unwrap()))
                    .push_bind(event.ask_quantity.map(|q| Decimal::from_f64(q).unwrap()))
                    .push_bind(event.sequence_number as i64);
            });
            
            query_builder.build().execute(&mut *tx).await?;
        }
        
        tx.commit().await?;
        
        self.metrics.batch_flushes.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Query recent candles
    pub async fn get_candles(
        &self,
        exchange: &str,
        symbol: &str,
        interval: CandleInterval,
        limit: i64,
    ) -> Result<Vec<Candle>> {
        // Check cache first
        let cache_key = format!("{}_{}_{}", exchange, symbol, interval.table_suffix());
        if let Some(cached) = self.cache.get(&cache_key, limit as usize).await {
            self.metrics.query_cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached);
        }
        
        self.metrics.query_cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Query database
        let table_name = format!("candles_{}", interval.table_suffix());
        let query = format!(
            "SELECT timestamp, open, high, low, close, volume, quote_volume,
                    trades_count, buy_volume, sell_volume, vwap, twap
             FROM {}
             WHERE exchange = $1 AND symbol = $2
             ORDER BY timestamp DESC
             LIMIT $3",
            table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(exchange)
            .bind(symbol)
            .bind(limit)
            .fetch_all(&*self.pool)
            .await?;
        
        let mut candles = Vec::with_capacity(rows.len());
        for row in rows {
            let candle = Candle {
                timestamp: row.get(0),
                exchange: exchange.to_string(),
                symbol: symbol.to_string(),
                interval,
                open: row.get(1),
                high: row.get(2),
                low: row.get(3),
                close: row.get(4),
                volume: row.get(5),
                quote_volume: row.get(6),
                trades_count: row.get(7),
                buy_volume: row.get(8),
                sell_volume: row.get(9),
                vwap: row.get(10),
                twap: row.get(11),
            };
            
            // Update cache
            self.cache.insert(cache_key.clone(), candle.clone()).await;
            candles.push(candle);
        }
        
        Ok(candles)
    }
    
    /// Get volume profile for a time range
    pub async fn get_volume_profile(
        &self,
        exchange: &str,
        symbol: &str,
        interval: CandleInterval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        price_levels: i32,
    ) -> Result<Vec<VolumeLevel>> {
        let query = r#"
            WITH price_range AS (
                SELECT 
                    MIN(price) as min_price,
                    MAX(price) as max_price
                FROM market_events
                WHERE exchange = $1 
                    AND symbol = $2
                    AND timestamp >= $3 
                    AND timestamp <= $4
                    AND event_type = 'trade'
            ),
            price_buckets AS (
                SELECT generate_series(
                    min_price,
                    max_price,
                    (max_price - min_price) / $5
                ) as price_level
                FROM price_range
            )
            SELECT 
                pb.price_level,
                COALESCE(SUM(me.quantity), 0) as volume,
                COALESCE(SUM(CASE WHEN me.side = 'buy' THEN me.quantity ELSE 0 END), 0) as buy_volume,
                COALESCE(SUM(CASE WHEN me.side = 'sell' THEN me.quantity ELSE 0 END), 0) as sell_volume,
                COALESCE(COUNT(me.*), 0) as trades
            FROM price_buckets pb
            LEFT JOIN market_events me ON 
                me.exchange = $1 
                AND me.symbol = $2
                AND me.timestamp >= $3 
                AND me.timestamp <= $4
                AND me.event_type = 'trade'
                AND me.price >= pb.price_level 
                AND me.price < pb.price_level + (
                    SELECT (max_price - min_price) / $5 FROM price_range
                )
            GROUP BY pb.price_level
            ORDER BY pb.price_level
        "#;
        
        let rows = sqlx::query(query)
            .bind(exchange)
            .bind(symbol)
            .bind(start_time)
            .bind(end_time)
            .bind(price_levels)
            .fetch_all(&*self.pool)
            .await?;
        
        let mut levels = Vec::with_capacity(rows.len());
        for row in rows {
            levels.push(VolumeLevel {
                price: row.get(0),
                volume: row.get(1),
                buy_volume: row.get(2),
                sell_volume: row.get(3),
                trades: row.get(4),
            });
        }
        
        Ok(levels)
    }
    
    /// Get order flow metrics
    pub async fn get_order_flow(
        &self,
        exchange: &str,
        symbol: &str,
        interval: CandleInterval,
        limit: i64,
    ) -> Result<Vec<OrderFlowMetrics>> {
        let query = r#"
            SELECT 
                timestamp,
                delta,
                cumulative_delta,
                imbalance_ratio,
                aggressor_ratio,
                absorption_ratio
            FROM order_flow_metrics
            WHERE exchange = $1 
                AND symbol = $2 
                AND interval = $3
            ORDER BY timestamp DESC
            LIMIT $4
        "#;
        
        let rows = sqlx::query(query)
            .bind(exchange)
            .bind(symbol)
            .bind(interval.table_suffix())
            .bind(limit)
            .fetch_all(&*self.pool)
            .await?;
        
        let mut metrics = Vec::with_capacity(rows.len());
        for row in rows {
            metrics.push(OrderFlowMetrics {
                timestamp: row.get(0),
                exchange: exchange.to_string(),
                symbol: symbol.to_string(),
                interval,
                delta: row.get(1),
                cumulative_delta: row.get(2),
                imbalance_ratio: row.get(3),
                aggressor_ratio: row.get(4),
                absorption_ratio: row.get(5),
            });
        }
        
        Ok(metrics)
    }
    
    /// Background task to periodically flush batches
    async fn flush_task(self: Arc<Self>) {
        let mut ticker = interval(self.config.flush_interval);
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            if let Err(e) = self.flush_batch().await {
                error!("Error during periodic flush: {}", e);
            }
        }
    }
    
    /// Background task for real-time aggregation
    async fn aggregation_task(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(1));
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            for interval in &self.config.candle_intervals {
                if let Err(e) = self.aggregate_candles(*interval).await {
                    error!("Error aggregating candles for {:?}: {}", interval, e);
                }
            }
        }
    }
    
    /// Aggregate candles for a specific interval
    async fn aggregate_candles(&self, interval: CandleInterval) -> Result<()> {
        let start = Instant::now();
        let table_name = format!("candles_{}", interval.table_suffix());
        let interval_str = interval.to_interval_string();
        
        // Use INSERT ... ON CONFLICT for upsert
        let query = format!(
            r#"
            INSERT INTO {} (
                timestamp, exchange, symbol, open, high, low, close,
                volume, quote_volume, trades_count, buy_volume, sell_volume, vwap, twap
            )
            SELECT 
                time_bucket('{}', timestamp) as bucket,
                exchange,
                symbol,
                first(price, timestamp) as open,
                max(price) as high,
                min(price) as low,
                last(price, timestamp) as close,
                sum(quantity) as volume,
                sum(price * quantity) as quote_volume,
                count(*) as trades_count,
                sum(CASE WHEN side = 'buy' THEN quantity ELSE 0 END) as buy_volume,
                sum(CASE WHEN side = 'sell' THEN quantity ELSE 0 END) as sell_volume,
                sum(price * quantity) / NULLIF(sum(quantity), 0) as vwap,
                avg(price) as twap
            FROM market_events
            WHERE event_type = 'trade'
                AND timestamp >= NOW() - INTERVAL '2 {}'
                AND timestamp < NOW()
            GROUP BY bucket, exchange, symbol
            ON CONFLICT (timestamp, exchange, symbol) 
            DO UPDATE SET
                high = GREATEST({}.high, EXCLUDED.high),
                low = LEAST({}.low, EXCLUDED.low),
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume,
                trades_count = EXCLUDED.trades_count,
                buy_volume = EXCLUDED.buy_volume,
                sell_volume = EXCLUDED.sell_volume,
                vwap = EXCLUDED.vwap,
                twap = EXCLUDED.twap
            "#,
            table_name, interval_str, interval_str, table_name, table_name
        );
        
        sqlx::query(&query).execute(&*self.pool).await?;
        
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.aggregation_latency_us.store(latency, Ordering::Relaxed);
        self.metrics.candles_generated.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Background task for compression
    async fn compression_task(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(3600));  // Run hourly
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            if let Err(e) = self.run_compression().await {
                error!("Error during compression: {}", e);
            }
        }
    }
    
    /// Run compression job
    async fn run_compression(&self) -> Result<()> {
        // TimescaleDB handles compression automatically via policies
        // This is just to track metrics
        self.metrics.compression_runs.fetch_add(1, Ordering::Relaxed);
        
        // Optionally, we can manually trigger compression for specific chunks
        sqlx::query(
            "SELECT compress_chunk(c.schema_name||'.'||c.table_name)
             FROM _timescaledb_catalog.chunk c
             INNER JOIN _timescaledb_catalog.hypertable h ON c.hypertable_id = h.id
             WHERE h.table_name = 'market_events'
                AND c.dropped = false
                AND c.compressed_chunk_id IS NULL
                AND c.range_end < NOW() - INTERVAL '1 day'
             LIMIT 10"
        )
        .execute(&*self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Get current metrics
    pub fn metrics(&self) -> AggregatorMetrics {
        AggregatorMetrics {
            events_processed: AtomicU64::new(self.metrics.events_processed.load(Ordering::Relaxed)),
            candles_generated: AtomicU64::new(self.metrics.candles_generated.load(Ordering::Relaxed)),
            insert_latency_us: AtomicU64::new(self.metrics.insert_latency_us.load(Ordering::Relaxed)),
            aggregation_latency_us: AtomicU64::new(self.metrics.aggregation_latency_us.load(Ordering::Relaxed)),
            batch_flushes: AtomicU64::new(self.metrics.batch_flushes.load(Ordering::Relaxed)),
            compression_runs: AtomicU64::new(self.metrics.compression_runs.load(Ordering::Relaxed)),
            query_cache_hits: AtomicU64::new(self.metrics.query_cache_hits.load(Ordering::Relaxed)),
            query_cache_misses: AtomicU64::new(self.metrics.query_cache_misses.load(Ordering::Relaxed)),
        }
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down TimescaleDB aggregator...");
        
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Final flush
        self.flush_batch().await?;
        
        info!("TimescaleDB aggregator shutdown complete");
        Ok(())
    }
}