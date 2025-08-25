use anyhow::{Result, Context};
use arrow::array::{
    ArrayRef, Float64Array, Int64Array, StringArray, TimestampNanosecondArray,
    BooleanArray, StructArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::{ArrowWriter, AsyncArrowWriter};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{
    WriterProperties, WriterPropertiesBuilder, EnabledStatistics,
};
use parquet::schema::types::ColumnPath;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::primitives::ByteStream;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{interval, Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use chrono::{DateTime, Utc, Datelike, Timelike};
use dashmap::DashMap;
use bytes::Bytes;
use tracing::{info, warn, error, debug};
use crate::producers::{MarketEvent, TradeSide};

/// Configuration for Parquet writer
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Base directory for Parquet files
    pub base_path: PathBuf,
    
    /// S3 bucket for long-term storage (optional)
    pub s3_bucket: Option<String>,
    
    /// S3 prefix for files
    pub s3_prefix: String,
    
    /// Target file size in bytes (default: 128MB)
    pub target_file_size: usize,
    
    /// Number of rows per row group (default: 100k)
    pub row_group_size: usize,
    
    /// Flush interval for pending writes
    pub flush_interval: Duration,
    
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    
    /// Enable dictionary encoding
    pub enable_dictionary: bool,
    
    /// Enable statistics
    pub enable_statistics: bool,
    
    /// Enable bloom filters for selective columns
    pub enable_bloom_filter: bool,
    
    /// Retention policy in hours (files older than this are archived to S3)
    pub local_retention_hours: u32,
    
    /// Enable V-Order optimization (for Direct Lake queries)
    pub enable_v_order: bool,
    
    /// Partition strategy
    pub partition_strategy: PartitionStrategy,
    
    /// Maximum concurrent writers
    pub max_writers: usize,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/data/parquet"),
            s3_bucket: None,
            s3_prefix: "market-data".to_string(),
            target_file_size: 128 * 1024 * 1024,  // 128MB
            row_group_size: 100_000,  // 100k rows per group
            flush_interval: Duration::from_secs(60),  // 1 minute
            compression: CompressionAlgorithm::Zstd(3),
            enable_dictionary: true,
            enable_statistics: true,
            enable_bloom_filter: true,
            local_retention_hours: 24,  // 1 day local retention
            enable_v_order: true,
            partition_strategy: PartitionStrategy::HourlyWithExchange,
            max_writers: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    Snappy,
    Gzip,
    Lz4,
    Zstd(i32),  // Compression level 1-22
    Brotli(u32),  // Compression level 0-11
}

#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    Daily,
    Hourly,
    HourlyWithExchange,
    DailyWithSymbol,
}

/// Buffer for accumulating events before writing
struct EventBuffer {
    events: Vec<MarketEvent>,
    first_timestamp: Option<DateTime<Utc>>,
    last_timestamp: Option<DateTime<Utc>>,
    estimated_size: AtomicUsize,
}

impl EventBuffer {
    fn new() -> Self {
        Self {
            events: Vec::with_capacity(10_000),
            first_timestamp: None,
            last_timestamp: None,
            estimated_size: AtomicUsize::new(0),
        }
    }
    
    fn add(&mut self, event: MarketEvent) {
        let ts = DateTime::<Utc>::from_timestamp_nanos(event.timestamp as i64);
        
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(ts);
        }
        self.last_timestamp = Some(ts);
        
        // Estimate event size (roughly 200 bytes per event)
        self.estimated_size.fetch_add(200, Ordering::Relaxed);
        self.events.push(event);
    }
    
    fn should_flush(&self, config: &ParquetConfig) -> bool {
        self.events.len() >= config.row_group_size ||
        self.estimated_size.load(Ordering::Relaxed) >= config.target_file_size
    }
    
    fn clear(&mut self) {
        self.events.clear();
        self.first_timestamp = None;
        self.last_timestamp = None;
        self.estimated_size.store(0, Ordering::Relaxed);
    }
}

/// Active writer for a specific partition
struct PartitionWriter {
    path: PathBuf,
    writer: AsyncArrowWriter<File>,
    row_count: AtomicU64,
    byte_count: AtomicUsize,
    created_at: Instant,
}

/// Metrics for monitoring
pub struct ParquetMetrics {
    pub files_written: AtomicU64,
    pub total_rows: AtomicU64,
    pub total_bytes: AtomicU64,
    pub write_latency_us: AtomicU64,
    pub compression_ratio: AtomicU64,  // Stored as percentage * 100
    pub s3_uploads: AtomicU64,
    pub s3_upload_bytes: AtomicU64,
    pub buffer_overflow_count: AtomicU64,
}

impl ParquetMetrics {
    fn new() -> Self {
        Self {
            files_written: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            write_latency_us: AtomicU64::new(0),
            compression_ratio: AtomicU64::new(0),
            s3_uploads: AtomicU64::new(0),
            s3_upload_bytes: AtomicU64::new(0),
            buffer_overflow_count: AtomicU64::new(0),
        }
    }
}

/// Main Parquet writer implementation
pub struct ParquetWriter {
    config: Arc<ParquetConfig>,
    schema: Arc<Schema>,
    buffers: Arc<DashMap<String, Arc<Mutex<EventBuffer>>>>,
    active_writers: Arc<DashMap<String, Arc<Mutex<PartitionWriter>>>>,
    s3_client: Option<Arc<S3Client>>,
    metrics: Arc<ParquetMetrics>,
    shutdown: Arc<AtomicBool>,
    flush_handle: Option<tokio::task::JoinHandle<()>>,
    archive_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ParquetWriter {
    /// Create a new Parquet writer with configuration
    pub async fn new(config: ParquetConfig) -> Result<Self> {
        // Create base directory if it doesn't exist
        fs::create_dir_all(&config.base_path)
            .await
            .context("Failed to create base directory")?;
        
        // Initialize S3 client if bucket is configured
        let s3_client = if config.s3_bucket.is_some() {
            let aws_config = aws_config::load_from_env().await;
            Some(Arc::new(S3Client::new(&aws_config)))
        } else {
            None
        };
        
        // Define Arrow schema for market events
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())), false),
            Field::new("exchange", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("event_type", DataType::Utf8, false),
            
            // Trade fields (nullable)
            Field::new("trade_id", DataType::Utf8, true),
            Field::new("price", DataType::Float64, true),
            Field::new("quantity", DataType::Float64, true),
            Field::new("side", DataType::Utf8, true),
            Field::new("is_maker", DataType::Boolean, true),
            
            // Order book fields (nullable)
            Field::new("bid_price", DataType::Float64, true),
            Field::new("bid_quantity", DataType::Float64, true),
            Field::new("ask_price", DataType::Float64, true),
            Field::new("ask_quantity", DataType::Float64, true),
            Field::new("spread", DataType::Float64, true),
            Field::new("mid_price", DataType::Float64, true),
            
            // Liquidation fields (nullable)
            Field::new("liquidation_side", DataType::Utf8, true),
            Field::new("liquidation_price", DataType::Float64, true),
            Field::new("liquidation_quantity", DataType::Float64, true),
            
            // Metadata
            Field::new("sequence_number", DataType::UInt64, false),
            Field::new("received_at", DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())), false),
            Field::new("latency_us", DataType::UInt64, false),
        ]));
        
        let writer = Arc::new(Self {
            config: Arc::new(config.clone()),
            schema,
            buffers: Arc::new(DashMap::new()),
            active_writers: Arc::new(DashMap::new()),
            s3_client,
            metrics: Arc::new(ParquetMetrics::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
            flush_handle: None,
            archive_handle: None,
        });
        
        // Start background tasks
        let flush_handle = {
            let writer = writer.clone();
            tokio::spawn(async move {
                writer.flush_task().await;
            })
        };
        
        let archive_handle = if config.s3_bucket.is_some() {
            let writer = writer.clone();
            tokio::spawn(async move {
                writer.archive_task().await;
            })
        } else {
            None
        };
        
        // Mutable copy to set handles
        let mut_writer = Arc::try_unwrap(writer)
            .unwrap_or_else(|arc| (*arc).clone());
        
        Ok(Self {
            flush_handle: Some(flush_handle),
            archive_handle,
            ..mut_writer
        })
    }
    
    /// Write a market event to the appropriate partition
    pub async fn write(&self, event: MarketEvent) -> Result<()> {
        let start = Instant::now();
        
        // Determine partition key based on strategy
        let partition_key = self.get_partition_key(&event);
        
        // Get or create buffer for this partition
        let buffer = self.buffers
            .entry(partition_key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(EventBuffer::new())));
        
        // Add event to buffer
        let should_flush = {
            let mut buf = buffer.lock().await;
            buf.add(event);
            buf.should_flush(&self.config)
        };
        
        // Flush if buffer is full
        if should_flush {
            self.flush_partition(&partition_key).await?;
        }
        
        // Update metrics
        let latency = start.elapsed().as_micros() as u64;
        self.metrics.write_latency_us.store(latency, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Flush all pending writes
    pub async fn flush(&self) -> Result<()> {
        let partitions: Vec<String> = self.buffers
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        
        for partition in partitions {
            self.flush_partition(&partition).await?;
        }
        
        Ok(())
    }
    
    /// Flush a specific partition
    async fn flush_partition(&self, partition_key: &str) -> Result<()> {
        let buffer = match self.buffers.get(partition_key) {
            Some(buf) => buf.clone(),
            None => return Ok(()),
        };
        
        let events = {
            let mut buf = buffer.lock().await;
            if buf.events.is_empty() {
                return Ok(());
            }
            let events = buf.events.clone();
            buf.clear();
            events
        };
        
        // Convert events to Arrow RecordBatch
        let batch = self.events_to_record_batch(events)?;
        
        // Get or create writer for this partition
        let writer = self.get_or_create_writer(partition_key).await?;
        
        // Write batch
        {
            let mut w = writer.lock().await;
            w.writer.write(&batch).await?;
            w.row_count.fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
            
            // Check if we should rotate the file
            if w.row_count.load(Ordering::Relaxed) >= self.config.row_group_size as u64 * 10 {
                // Close current writer and create new one
                w.writer.close().await?;
                self.active_writers.remove(partition_key);
                
                // Update metrics
                self.metrics.files_written.fetch_add(1, Ordering::Relaxed);
                self.metrics.total_rows.fetch_add(w.row_count.load(Ordering::Relaxed), Ordering::Relaxed);
                
                info!("Rotated Parquet file for partition: {}", partition_key);
            }
        }
        
        Ok(())
    }
    
    /// Convert events to Arrow RecordBatch
    fn events_to_record_batch(&self, events: Vec<MarketEvent>) -> Result<RecordBatch> {
        let num_rows = events.len();
        
        // Prepare arrays for each column
        let mut timestamps = Vec::with_capacity(num_rows);
        let mut exchanges = Vec::with_capacity(num_rows);
        let mut symbols = Vec::with_capacity(num_rows);
        let mut event_types = Vec::with_capacity(num_rows);
        let mut trade_ids = Vec::with_capacity(num_rows);
        let mut prices = Vec::with_capacity(num_rows);
        let mut quantities = Vec::with_capacity(num_rows);
        let mut sides = Vec::with_capacity(num_rows);
        let mut is_makers = Vec::with_capacity(num_rows);
        let mut bid_prices = Vec::with_capacity(num_rows);
        let mut bid_quantities = Vec::with_capacity(num_rows);
        let mut ask_prices = Vec::with_capacity(num_rows);
        let mut ask_quantities = Vec::with_capacity(num_rows);
        let mut spreads = Vec::with_capacity(num_rows);
        let mut mid_prices = Vec::with_capacity(num_rows);
        let mut liquidation_sides = Vec::with_capacity(num_rows);
        let mut liquidation_prices = Vec::with_capacity(num_rows);
        let mut liquidation_quantities = Vec::with_capacity(num_rows);
        let mut sequence_numbers = Vec::with_capacity(num_rows);
        let mut received_ats = Vec::with_capacity(num_rows);
        let mut latencies = Vec::with_capacity(num_rows);
        
        // Process each event
        for event in events {
            timestamps.push(event.timestamp as i64);
            exchanges.push(event.exchange.clone());
            symbols.push(event.symbol.clone());
            event_types.push(event.event_type.clone());
            
            // Trade data
            trade_ids.push(event.trade_id.clone());
            prices.push(event.price);
            quantities.push(event.quantity);
            sides.push(event.side.as_ref().map(|s| match s {
                TradeSide::Buy => "buy",
                TradeSide::Sell => "sell",
            }.to_string()));
            is_makers.push(event.is_maker);
            
            // Order book data
            bid_prices.push(event.bid_price);
            bid_quantities.push(event.bid_quantity);
            ask_prices.push(event.ask_price);
            ask_quantities.push(event.ask_quantity);
            spreads.push(event.spread);
            mid_prices.push(event.mid_price);
            
            // Liquidation data
            liquidation_sides.push(event.liquidation_side.as_ref().map(|s| match s {
                TradeSide::Buy => "buy",
                TradeSide::Sell => "sell",
            }.to_string()));
            liquidation_prices.push(event.liquidation_price);
            liquidation_quantities.push(event.liquidation_quantity);
            
            // Metadata
            sequence_numbers.push(event.sequence_number);
            received_ats.push(event.received_at as i64);
            latencies.push(event.latency_us);
        }
        
        // Create Arrow arrays
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(TimestampNanosecondArray::from(timestamps).with_timezone("UTC")),
            Arc::new(StringArray::from(exchanges)),
            Arc::new(StringArray::from(symbols)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(StringArray::from(trade_ids)),
            Arc::new(Float64Array::from(prices)),
            Arc::new(Float64Array::from(quantities)),
            Arc::new(StringArray::from(sides)),
            Arc::new(BooleanArray::from(is_makers)),
            Arc::new(Float64Array::from(bid_prices)),
            Arc::new(Float64Array::from(bid_quantities)),
            Arc::new(Float64Array::from(ask_prices)),
            Arc::new(Float64Array::from(ask_quantities)),
            Arc::new(Float64Array::from(spreads)),
            Arc::new(Float64Array::from(mid_prices)),
            Arc::new(StringArray::from(liquidation_sides)),
            Arc::new(Float64Array::from(liquidation_prices)),
            Arc::new(Float64Array::from(liquidation_quantities)),
            Arc::new(UInt64Array::from(sequence_numbers)),
            Arc::new(TimestampNanosecondArray::from(received_ats).with_timezone("UTC")),
            Arc::new(UInt64Array::from(latencies)),
        ];
        
        RecordBatch::try_new(self.schema.clone(), arrays)
            .context("Failed to create RecordBatch")
    }
    
    /// Get or create a writer for a partition
    async fn get_or_create_writer(&self, partition_key: &str) -> Result<Arc<Mutex<PartitionWriter>>> {
        // Check if writer already exists
        if let Some(writer) = self.active_writers.get(partition_key) {
            return Ok(writer.clone());
        }
        
        // Create new writer
        let file_path = self.get_file_path(partition_key);
        
        // Ensure directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Open file for writing
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await?;
        
        // Configure writer properties
        let props = self.build_writer_properties();
        
        // Create Arrow writer
        let writer = AsyncArrowWriter::try_new(
            file,
            self.schema.clone(),
            Some(props),
        )?;
        
        let partition_writer = Arc::new(Mutex::new(PartitionWriter {
            path: file_path,
            writer,
            row_count: AtomicU64::new(0),
            byte_count: AtomicUsize::new(0),
            created_at: Instant::now(),
        }));
        
        self.active_writers.insert(partition_key.to_string(), partition_writer.clone());
        
        Ok(partition_writer)
    }
    
    /// Build Parquet writer properties
    fn build_writer_properties(&self) -> WriterProperties {
        let mut builder = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_created_by("Bot4 Trading System".to_string())
            .set_data_page_size_limit(1024 * 1024)  // 1MB pages
            .set_write_batch_size(self.config.row_group_size)
            .set_max_row_group_size(self.config.row_group_size);
        
        // Set compression
        builder = match &self.config.compression {
            CompressionAlgorithm::Snappy => builder.set_compression(Compression::SNAPPY),
            CompressionAlgorithm::Gzip => builder.set_compression(Compression::GZIP),
            CompressionAlgorithm::Lz4 => builder.set_compression(Compression::LZ4),
            CompressionAlgorithm::Zstd(level) => {
                builder.set_compression(Compression::ZSTD(ZstdLevel::try_new(*level).unwrap()))
            },
            CompressionAlgorithm::Brotli(level) => {
                builder.set_compression(Compression::BROTLI(parquet::basic::BrotliLevel::try_new(*level).unwrap()))
            },
        };
        
        // Enable statistics if configured
        if self.config.enable_statistics {
            builder = builder.set_statistics_enabled(EnabledStatistics::Page);
        }
        
        // Enable dictionary encoding for string columns
        if self.config.enable_dictionary {
            builder = builder
                .set_column_dictionary_enabled(ColumnPath::from("exchange"), true)
                .set_column_dictionary_enabled(ColumnPath::from("symbol"), true)
                .set_column_dictionary_enabled(ColumnPath::from("event_type"), true)
                .set_column_dictionary_enabled(ColumnPath::from("side"), true);
        }
        
        // Enable bloom filters for selective columns
        if self.config.enable_bloom_filter {
            builder = builder
                .set_column_bloom_filter_enabled(ColumnPath::from("symbol"), true)
                .set_column_bloom_filter_enabled(ColumnPath::from("exchange"), true)
                .set_column_bloom_filter_enabled(ColumnPath::from("trade_id"), true);
        }
        
        // Set encoding for specific columns
        builder = builder
            .set_column_encoding(ColumnPath::from("timestamp"), Encoding::DELTA_BINARY_PACKED)
            .set_column_encoding(ColumnPath::from("price"), Encoding::BYTE_STREAM_SPLIT)
            .set_column_encoding(ColumnPath::from("quantity"), Encoding::BYTE_STREAM_SPLIT)
            .set_column_encoding(ColumnPath::from("sequence_number"), Encoding::DELTA_BINARY_PACKED);
        
        builder.build()
    }
    
    /// Get partition key for an event
    fn get_partition_key(&self, event: &MarketEvent) -> String {
        let ts = DateTime::<Utc>::from_timestamp_nanos(event.timestamp as i64);
        
        match &self.config.partition_strategy {
            PartitionStrategy::Daily => {
                format!("{}/{:04}/{:02}/{:02}",
                    event.exchange,
                    ts.year(),
                    ts.month(),
                    ts.day()
                )
            },
            PartitionStrategy::Hourly => {
                format!("{}/{:04}/{:02}/{:02}/{:02}",
                    event.exchange,
                    ts.year(),
                    ts.month(),
                    ts.day(),
                    ts.hour()
                )
            },
            PartitionStrategy::HourlyWithExchange => {
                format!("exchange={}/year={:04}/month={:02}/day={:02}/hour={:02}",
                    event.exchange,
                    ts.year(),
                    ts.month(),
                    ts.day(),
                    ts.hour()
                )
            },
            PartitionStrategy::DailyWithSymbol => {
                format!("{}/{}/{:04}/{:02}/{:02}",
                    event.exchange,
                    event.symbol,
                    ts.year(),
                    ts.month(),
                    ts.day()
                )
            },
        }
    }
    
    /// Get file path for a partition
    fn get_file_path(&self, partition_key: &str) -> PathBuf {
        let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let filename = format!("market_data_{}.parquet", timestamp);
        self.config.base_path.join(partition_key).join(filename)
    }
    
    /// Background task to periodically flush buffers
    async fn flush_task(self: Arc<Self>) {
        let mut ticker = interval(self.config.flush_interval);
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            if let Err(e) = self.flush().await {
                error!("Error during periodic flush: {}", e);
            }
        }
    }
    
    /// Background task to archive old files to S3
    async fn archive_task(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(3600));  // Check every hour
        
        while !self.shutdown.load(Ordering::Relaxed) {
            ticker.tick().await;
            
            if let Err(e) = self.archive_old_files().await {
                error!("Error during archive task: {}", e);
            }
        }
    }
    
    /// Archive files older than retention period to S3
    async fn archive_old_files(&self) -> Result<()> {
        let s3_client = match &self.s3_client {
            Some(client) => client,
            None => return Ok(()),
        };
        
        let s3_bucket = match &self.config.s3_bucket {
            Some(bucket) => bucket,
            None => return Ok(()),
        };
        
        let cutoff = Utc::now() - chrono::Duration::hours(self.config.local_retention_hours as i64);
        
        // Walk through all Parquet files
        let mut entries = fs::read_dir(&self.config.base_path).await?;
        let mut archived_count = 0;
        let mut archived_bytes = 0u64;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.extension() == Some(std::ffi::OsStr::new("parquet")) {
                let metadata = entry.metadata().await?;
                
                if let Ok(modified) = metadata.modified() {
                    let modified_time = DateTime::<Utc>::from(modified);
                    
                    if modified_time < cutoff {
                        // Upload to S3
                        let file_name = path.file_name()
                            .and_then(|n| n.to_str())
                            .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;
                        
                        let s3_key = format!("{}/{}", self.config.s3_prefix, file_name);
                        
                        let body = fs::read(&path).await?;
                        let file_size = body.len() as u64;
                        
                        s3_client
                            .put_object()
                            .bucket(s3_bucket)
                            .key(&s3_key)
                            .body(ByteStream::from(body))
                            .storage_class(aws_sdk_s3::types::StorageClass::IntelligentTiering)
                            .send()
                            .await?;
                        
                        // Delete local file after successful upload
                        fs::remove_file(&path).await?;
                        
                        archived_count += 1;
                        archived_bytes += file_size;
                        
                        info!("Archived {} to S3: {} ({} bytes)", file_name, s3_key, file_size);
                    }
                }
            }
        }
        
        if archived_count > 0 {
            self.metrics.s3_uploads.fetch_add(archived_count, Ordering::Relaxed);
            self.metrics.s3_upload_bytes.fetch_add(archived_bytes, Ordering::Relaxed);
            info!("Archived {} files ({} bytes) to S3", archived_count, archived_bytes);
        }
        
        Ok(())
    }
    
    /// Query recent data (for backtesting or analysis)
    pub async fn query_recent(
        &self,
        exchange: &str,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<MarketEvent>> {
        // This would typically query both local Parquet files and S3
        // For now, return a placeholder
        warn!("Query functionality not yet implemented");
        Ok(Vec::new())
    }
    
    /// Get current metrics
    pub fn metrics(&self) -> ParquetMetrics {
        ParquetMetrics {
            files_written: AtomicU64::new(self.metrics.files_written.load(Ordering::Relaxed)),
            total_rows: AtomicU64::new(self.metrics.total_rows.load(Ordering::Relaxed)),
            total_bytes: AtomicU64::new(self.metrics.total_bytes.load(Ordering::Relaxed)),
            write_latency_us: AtomicU64::new(self.metrics.write_latency_us.load(Ordering::Relaxed)),
            compression_ratio: AtomicU64::new(self.metrics.compression_ratio.load(Ordering::Relaxed)),
            s3_uploads: AtomicU64::new(self.metrics.s3_uploads.load(Ordering::Relaxed)),
            s3_upload_bytes: AtomicU64::new(self.metrics.s3_upload_bytes.load(Ordering::Relaxed)),
            buffer_overflow_count: AtomicU64::new(self.metrics.buffer_overflow_count.load(Ordering::Relaxed)),
        }
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Parquet writer...");
        
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Flush all pending data
        self.flush().await?;
        
        // Close all active writers
        for entry in self.active_writers.iter() {
            let mut writer = entry.value().lock().await;
            writer.writer.close().await?;
        }
        
        info!("Parquet writer shutdown complete");
        Ok(())
    }
}