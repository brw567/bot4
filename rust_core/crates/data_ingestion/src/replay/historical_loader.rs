// Historical Data Loader for Multiple Sources
// DEEP DIVE: Support for LOBSTER, Tardis, Arctic, and custom formats
//
// References:
// - LOBSTER academic data format specification
// - Tardis.dev API documentation
// - Arctic/ArcticDB schema
// - KDB+/q tick data format
// - Databento normalized schemas

use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, NaiveDateTime};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context as AnyhowContext};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::fs::File;
use csv_async::{AsyncReaderBuilder, AsyncDeserializer};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};

use crate::types::{Price, Quantity, Symbol, Exchange};
use crate::replay::lob_simulator::{OrderBookUpdate, UpdateType, Side};

/// Supported data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// LOBSTER format (academic standard)
    LOBSTER {
        orderbook_file: PathBuf,
        message_file: PathBuf,
        levels: u32,
    },
    /// Tardis.dev format
    Tardis {
        file_path: PathBuf,
        data_type: TardisDataType,
    },
    /// Arctic/ArcticDB format
    Arctic {
        library_path: PathBuf,
        symbol: String,
        date_range: (DateTime<Utc>, DateTime<Utc>),
    },
    /// Databento format
    Databento {
        file_path: PathBuf,
        schema: DatabentSchema,
    },
    /// Custom CSV format
    CustomCSV {
        file_path: PathBuf,
        delimiter: char,
        has_header: bool,
        timestamp_col: usize,
        price_col: usize,
        quantity_col: usize,
        side_col: usize,
    },
    /// Custom Parquet format
    CustomParquet {
        file_path: PathBuf,
        timestamp_col: String,
        price_col: String,
        quantity_col: String,
        side_col: String,
    },
    /// Binary format (e.g., FIX/FAST)
    Binary {
        file_path: PathBuf,
        format: BinaryFormat,
    },
}

/// Tardis data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TardisDataType {
    BookSnapshot,
    BookUpdate,
    Trade,
    Quote,
    BookChange,
}

/// Databento schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabentSchema {
    MBO,  // Market by Order
    MBP1, // Market by Price (top of book)
    MBP10, // Market by Price (10 levels)
    TBBO,  // Top of book
    Trades,
}

/// Binary formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryFormat {
    FIX,
    FAST,
    SBE,  // Simple Binary Encoding
    Protobuf,
    MsgPack,
}

/// Data format for normalized output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    OrderBookUpdate,
    Trade,
    Quote,
    BookSnapshot,
}

/// Tick data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub exchange: Exchange,
    pub sequence: u64,
    pub tick_type: TickType,
}

/// Tick types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TickType {
    Trade(TradeData),
    Quote(QuoteData),
    BookUpdate(OrderBookUpdate),
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub price: Price,
    pub quantity: Quantity,
    pub side: Side,
    pub trade_id: u64,
    pub is_implied: bool,
}

/// Quote data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteData {
    pub bid_price: Price,
    pub bid_quantity: Quantity,
    pub ask_price: Price,
    pub ask_quantity: Quantity,
    pub bid_count: u32,
    pub ask_count: u32,
}

/// LOBSTER message types
#[derive(Debug)]
enum LOBSTERMessageType {
    NewLimitOrder = 1,
    CancellationPartial = 2,
    CancellationTotal = 3,
    ExecutionVisible = 4,
    ExecutionHidden = 5,
    CrossTrade = 6,
    TradingHalt = 7,
}

/// LOBSTER message structure
#[derive(Debug, Deserialize)]
struct LOBSTERMessage {
    timestamp: f64,
    message_type: u8,
    order_id: u64,
    size: u64,
    price: u64,
    direction: i8,
}

/// Tardis book snapshot structure
#[derive(Debug, Deserialize)]
struct TardisBookSnapshot {
    timestamp: String,
    symbol: String,
    exchange: String,
    bids: Vec<TardisLevel>,
    asks: Vec<TardisLevel>,
}

/// Tardis price level
#[derive(Debug, Deserialize)]
struct TardisLevel {
    price: String,
    amount: String,
}

/// Historical data loader trait
#[async_trait]
pub trait DataLoader: Send + Sync {
    /// Load data from source
    async fn load(&mut self) -> Result<Vec<TickData>>;
    
    /// Stream data incrementally
    async fn stream(&mut self) -> Result<Box<dyn AsyncIterator<Item = Result<TickData>>>>;
    
    /// Get metadata about the data
    fn metadata(&self) -> DataMetadata;
}

/// Data metadata
#[derive(Debug, Clone)]
pub struct DataMetadata {
    pub symbol: Symbol,
    pub exchange: Exchange,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub tick_count: u64,
    pub data_format: DataFormat,
    pub has_l3_data: bool,
}

/// Main historical data loader
pub struct HistoricalDataLoader {
    source: DataSource,
    buffer_size: usize,
    cache: Arc<RwLock<Vec<TickData>>>,
    metadata: Option<DataMetadata>,
}

impl HistoricalDataLoader {
    pub fn new(source: DataSource) -> Self {
        Self {
            source,
            buffer_size: 100_000,
            cache: Arc::new(RwLock::new(Vec::new())),
            metadata: None,
        }
    }
    
    /// Load LOBSTER format data
    async fn load_lobster(
        &mut self,
        orderbook_file: &Path,
        message_file: &Path,
        levels: u32,
    ) -> Result<Vec<TickData>> {
        let mut ticks = Vec::new();
        
        // Read message file
        let message_file = File::open(message_file).await?;
        let message_reader = BufReader::new(message_file);
        let mut message_lines = message_reader.lines();
        
        // Read orderbook file
        let orderbook_file = File::open(orderbook_file).await?;
        let orderbook_reader = BufReader::new(orderbook_file);
        let mut orderbook_lines = orderbook_reader.lines();
        
        let mut sequence = 0u64;
        
        while let (Some(msg_line), Some(book_line)) = (
            message_lines.next_line().await?,
            orderbook_lines.next_line().await?
        ) {
            // Parse message
            let msg_parts: Vec<&str> = msg_line.split(',').collect();
            if msg_parts.len() < 6 {
                continue;
            }
            
            let timestamp_ns = msg_parts[0].parse::<i64>()
                .context("Failed to parse LOBSTER timestamp")?;
            let message_type = msg_parts[1].parse::<u8>()?;
            let order_id = msg_parts[2].parse::<u64>()?;
            let size = msg_parts[3].parse::<u64>()?;
            let price = msg_parts[4].parse::<u64>()?;
            let direction = msg_parts[5].parse::<i8>()?;
            
            // Convert nanoseconds to DateTime
            let timestamp = DateTime::from_utc(
                NaiveDateTime::from_timestamp_opt(
                    timestamp_ns / 1_000_000_000,
                    (timestamp_ns % 1_000_000_000) as u32
                ).unwrap(),
                Utc
            );
            
            // Parse orderbook state
            let book_parts: Vec<&str> = book_line.split(',').collect();
            let mut bids = Vec::new();
            let mut asks = Vec::new();
            
            for i in 0..levels as usize {
                let ask_price_idx = i * 4;
                let ask_size_idx = ask_price_idx + 1;
                let bid_price_idx = ask_price_idx + 2;
                let bid_size_idx = ask_price_idx + 3;
                
                if bid_price_idx < book_parts.len() {
                    let bid_price = book_parts[bid_price_idx].parse::<f64>()?;
                    let bid_size = book_parts[bid_size_idx].parse::<f64>()?;
                    if bid_price > 0.0 && bid_size > 0.0 {
                        bids.push((Price(Decimal::from_f64_retain(bid_price / 10000.0).unwrap()),
                                  Quantity(Decimal::from_f64_retain(bid_size).unwrap())));
                    }
                }
                
                if ask_size_idx < book_parts.len() {
                    let ask_price = book_parts[ask_price_idx].parse::<f64>()?;
                    let ask_size = book_parts[ask_size_idx].parse::<f64>()?;
                    if ask_price > 0.0 && ask_size > 0.0 {
                        asks.push((Price(Decimal::from_f64_retain(ask_price / 10000.0).unwrap()),
                                  Quantity(Decimal::from_f64_retain(ask_size).unwrap())));
                    }
                }
            }
            
            // Convert to OrderBookUpdate
            let update_type = match message_type {
                1 => {
                    // New limit order
                    let side = if direction > 0 { Side::Bid } else { Side::Ask };
                    UpdateType::Add {
                        order_id,
                        side,
                        price: Price(Decimal::from(price) / Decimal::from(10000)),
                        quantity: Quantity(Decimal::from(size)),
                    }
                }
                2 | 3 => {
                    // Cancellation
                    UpdateType::Cancel { order_id }
                }
                4 | 5 => {
                    // Execution
                    let side = if direction > 0 { Side::Bid } else { Side::Ask };
                    UpdateType::Trade {
                        order_id,
                        traded_quantity: Quantity(Decimal::from(size)),
                        aggressor_side: side,
                    }
                }
                7 => {
                    // Trading halt
                    UpdateType::Clear
                }
                _ => continue,
            };
            
            sequence += 1;
            
            let update = OrderBookUpdate {
                symbol: Symbol("LOBSTER".to_string()),
                exchange: Exchange("LOBSTER".to_string()),
                timestamp,
                sequence_number: sequence,
                update_type,
                latency_ns: 0,
            };
            
            ticks.push(TickData {
                timestamp,
                symbol: Symbol("LOBSTER".to_string()),
                exchange: Exchange("LOBSTER".to_string()),
                sequence,
                tick_type: TickType::BookUpdate(update),
            });
        }
        
        Ok(ticks)
    }
    
    /// Load Tardis format data
    async fn load_tardis(
        &mut self,
        file_path: &Path,
        data_type: &TardisDataType,
    ) -> Result<Vec<TickData>> {
        let mut ticks = Vec::new();
        let file = File::open(file_path).await?;
        let mut reader = AsyncReaderBuilder::new()
            .has_headers(true)
            .create_deserializer(file);
        
        let mut sequence = 0u64;
        
        match data_type {
            TardisDataType::BookSnapshot => {
                let mut records: AsyncDeserializer<File, TardisBookSnapshot> = reader;
                while let Some(result) = records.next().await {
                    let snapshot = result?;
                    sequence += 1;
                    
                    let timestamp = DateTime::parse_from_rfc3339(&snapshot.timestamp)?
                        .with_timezone(&Utc);
                    
                    let mut bid_levels = Vec::new();
                    let mut ask_levels = Vec::new();
                    
                    for bid in snapshot.bids {
                        bid_levels.push(crate::replay::lob_simulator::OrderBookLevel {
                            price: Price(Decimal::from_str(&bid.price)?),
                            quantity: Quantity(Decimal::from_str(&bid.amount)?),
                            order_count: 1,
                            exchange_timestamp: timestamp,
                            local_timestamp: Utc::now(),
                            implied_quantity: None,
                        });
                    }
                    
                    for ask in snapshot.asks {
                        ask_levels.push(crate::replay::lob_simulator::OrderBookLevel {
                            price: Price(Decimal::from_str(&ask.price)?),
                            quantity: Quantity(Decimal::from_str(&ask.amount)?),
                            order_count: 1,
                            exchange_timestamp: timestamp,
                            local_timestamp: Utc::now(),
                            implied_quantity: None,
                        });
                    }
                    
                    let update = OrderBookUpdate {
                        symbol: Symbol(snapshot.symbol),
                        exchange: Exchange(snapshot.exchange),
                        timestamp,
                        sequence_number: sequence,
                        update_type: UpdateType::Snapshot {
                            bids: bid_levels,
                            asks: ask_levels,
                        },
                        latency_ns: 0,
                    };
                    
                    ticks.push(TickData {
                        timestamp,
                        symbol: Symbol(snapshot.symbol),
                        exchange: Exchange(snapshot.exchange),
                        sequence,
                        tick_type: TickType::BookUpdate(update),
                    });
                }
            }
            _ => {
                // Other Tardis data types would be implemented similarly
                anyhow::bail!("Tardis data type {:?} not yet implemented", data_type);
            }
        }
        
        Ok(ticks)
    }
    
    /// Load custom CSV format
    async fn load_custom_csv(
        &mut self,
        file_path: &Path,
        delimiter: char,
        has_header: bool,
        timestamp_col: usize,
        price_col: usize,
        quantity_col: usize,
        side_col: usize,
    ) -> Result<Vec<TickData>> {
        let mut ticks = Vec::new();
        let file = File::open(file_path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header if present
        if has_header {
            lines.next_line().await?;
        }
        
        let mut sequence = 0u64;
        
        while let Some(line) = lines.next_line().await? {
            let parts: Vec<&str> = line.split(delimiter).collect();
            
            if parts.len() <= timestamp_col.max(price_col).max(quantity_col).max(side_col) {
                continue;
            }
            
            sequence += 1;
            
            // Parse fields
            let timestamp = DateTime::parse_from_rfc3339(parts[timestamp_col])?
                .with_timezone(&Utc);
            let price = Price(Decimal::from_str(parts[price_col])?);
            let quantity = Quantity(Decimal::from_str(parts[quantity_col])?);
            let side = match parts[side_col].to_lowercase().as_str() {
                "buy" | "bid" | "b" => Side::Bid,
                "sell" | "ask" | "s" => Side::Ask,
                _ => continue,
            };
            
            // Create trade tick
            ticks.push(TickData {
                timestamp,
                symbol: Symbol("CUSTOM".to_string()),
                exchange: Exchange("CUSTOM".to_string()),
                sequence,
                tick_type: TickType::Trade(TradeData {
                    price,
                    quantity,
                    side,
                    trade_id: sequence,
                    is_implied: false,
                }),
            });
        }
        
        Ok(ticks)
    }
    
    /// Load custom Parquet format
    async fn load_custom_parquet(
        &mut self,
        file_path: &Path,
        timestamp_col: &str,
        price_col: &str,
        quantity_col: &str,
        side_col: &str,
    ) -> Result<Vec<TickData>> {
        let mut ticks = Vec::new();
        
        // Open Parquet file
        let file = std::fs::File::open(file_path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut iter = reader.get_row_iter(None)?;
        
        let mut sequence = 0u64;
        
        while let Some(row_result) = iter.next() {
            let row = row_result?;
            sequence += 1;
            
            // Extract fields from row
            let timestamp_ms = row.get_long_by_name(timestamp_col)
                .ok_or_else(|| anyhow::anyhow!("Missing timestamp column"))?;
            let timestamp = DateTime::from_utc(
                NaiveDateTime::from_timestamp_opt(
                    timestamp_ms / 1000,
                    ((timestamp_ms % 1000) * 1_000_000) as u32
                ).unwrap(),
                Utc
            );
            
            let price = row.get_double_by_name(price_col)
                .ok_or_else(|| anyhow::anyhow!("Missing price column"))?;
            let quantity = row.get_double_by_name(quantity_col)
                .ok_or_else(|| anyhow::anyhow!("Missing quantity column"))?;
            let side_str = row.get_string_by_name(side_col)
                .ok_or_else(|| anyhow::anyhow!("Missing side column"))?;
            
            let side = match side_str.to_lowercase().as_str() {
                "buy" | "bid" => Side::Bid,
                "sell" | "ask" => Side::Ask,
                _ => continue,
            };
            
            ticks.push(TickData {
                timestamp,
                symbol: Symbol("PARQUET".to_string()),
                exchange: Exchange("PARQUET".to_string()),
                sequence,
                tick_type: TickType::Trade(TradeData {
                    price: Price(Decimal::from_f64_retain(price).unwrap()),
                    quantity: Quantity(Decimal::from_f64_retain(quantity).unwrap()),
                    side,
                    trade_id: sequence,
                    is_implied: false,
                }),
            });
        }
        
        Ok(ticks)
    }
    
    /// Normalize data to common format
    fn normalize_data(&self, raw_data: Vec<TickData>) -> Vec<TickData> {
        // Data is already in normalized TickData format
        raw_data
    }
    
    /// Filter data by time range
    pub fn filter_by_time(
        &self,
        data: Vec<TickData>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<TickData> {
        data.into_iter()
            .filter(|tick| tick.timestamp >= start && tick.timestamp <= end)
            .collect()
    }
    
    /// Resample data to fixed intervals
    pub fn resample(
        &self,
        data: Vec<TickData>,
        interval_ms: u64,
    ) -> Vec<TickData> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut resampled = Vec::new();
        let mut current_bucket = Vec::new();
        let mut bucket_start = data[0].timestamp;
        let interval = chrono::Duration::milliseconds(interval_ms as i64);
        
        for tick in data {
            if tick.timestamp >= bucket_start + interval {
                // Process current bucket
                if !current_bucket.is_empty() {
                    // Take last tick in bucket (could also aggregate)
                    resampled.push(current_bucket.pop().unwrap());
                }
                
                // Start new bucket
                bucket_start = bucket_start + interval;
                current_bucket.clear();
            }
            
            current_bucket.push(tick);
        }
        
        // Process final bucket
        if !current_bucket.is_empty() {
            resampled.push(current_bucket.pop().unwrap());
        }
        
        resampled
    }
}

#[async_trait]
impl DataLoader for HistoricalDataLoader {
    async fn load(&mut self) -> Result<Vec<TickData>> {
        let data = match &self.source {
            DataSource::LOBSTER { orderbook_file, message_file, levels } => {
                self.load_lobster(orderbook_file, message_file, *levels).await?
            }
            DataSource::Tardis { file_path, data_type } => {
                self.load_tardis(file_path, data_type).await?
            }
            DataSource::CustomCSV {
                file_path,
                delimiter,
                has_header,
                timestamp_col,
                price_col,
                quantity_col,
                side_col,
            } => {
                self.load_custom_csv(
                    file_path,
                    *delimiter,
                    *has_header,
                    *timestamp_col,
                    *price_col,
                    *quantity_col,
                    *side_col,
                ).await?
            }
            DataSource::CustomParquet {
                file_path,
                timestamp_col,
                price_col,
                quantity_col,
                side_col,
            } => {
                self.load_custom_parquet(
                    file_path,
                    timestamp_col,
                    price_col,
                    quantity_col,
                    side_col,
                ).await?
            }
            _ => {
                anyhow::bail!("Data source not yet implemented");
            }
        };
        
        // Normalize and cache
        let normalized = self.normalize_data(data);
        *self.cache.write() = normalized.clone();
        
        Ok(normalized)
    }
    
    async fn stream(&mut self) -> Result<Box<dyn AsyncIterator<Item = Result<TickData>>>> {
        // For now, just return cached data as stream
        // In production, would implement true streaming
        let cache = self.cache.read().clone();
        Ok(Box::new(VecIterator::new(cache)))
    }
    
    fn metadata(&self) -> DataMetadata {
        if let Some(ref metadata) = self.metadata {
            return metadata.clone();
        }
        
        // Calculate metadata from cache
        let cache = self.cache.read();
        if cache.is_empty() {
            return DataMetadata {
                symbol: Symbol("UNKNOWN".to_string()),
                exchange: Exchange("UNKNOWN".to_string()),
                start_time: Utc::now(),
                end_time: Utc::now(),
                tick_count: 0,
                data_format: DataFormat::OrderBookUpdate,
                has_l3_data: false,
            };
        }
        
        DataMetadata {
            symbol: cache[0].symbol.clone(),
            exchange: cache[0].exchange.clone(),
            start_time: cache.first().unwrap().timestamp,
            end_time: cache.last().unwrap().timestamp,
            tick_count: cache.len() as u64,
            data_format: DataFormat::OrderBookUpdate,
            has_l3_data: true,
        }
    }
}

/// Simple async iterator for vec
struct VecIterator {
    data: Vec<TickData>,
    index: usize,
}

impl VecIterator {
    fn new(data: Vec<TickData>) -> Self {
        Self { data, index: 0 }
    }
}

#[async_trait]
trait AsyncIterator: Send {
    type Item;
    async fn next(&mut self) -> Option<Self::Item>;
}

#[async_trait]
impl AsyncIterator for VecIterator {
    type Item = Result<TickData>;
    
    async fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let tick = self.data[self.index].clone();
            self.index += 1;
            Some(Ok(tick))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;
    
    #[tokio::test]
    async fn test_custom_csv_loader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        
        // Create test CSV file
        let mut file = tokio::fs::File::create(&file_path).await.unwrap();
        file.write_all(b"timestamp,price,quantity,side\n").await.unwrap();
        file.write_all(b"2024-01-01T00:00:00Z,50000.0,1.5,buy\n").await.unwrap();
        file.write_all(b"2024-01-01T00:00:01Z,50001.0,2.0,sell\n").await.unwrap();
        file.sync_all().await.unwrap();
        
        let source = DataSource::CustomCSV {
            file_path: file_path.clone(),
            delimiter: ',',
            has_header: true,
            timestamp_col: 0,
            price_col: 1,
            quantity_col: 2,
            side_col: 3,
        };
        
        let mut loader = HistoricalDataLoader::new(source);
        let data = loader.load().await.unwrap();
        
        assert_eq!(data.len(), 2);
        
        // Check first tick
        if let TickType::Trade(trade) = &data[0].tick_type {
            assert_eq!(trade.price.0, Decimal::from(50000));
            assert_eq!(trade.quantity.0, Decimal::from_str("1.5").unwrap());
            assert_eq!(trade.side, Side::Bid);
        } else {
            panic!("Expected trade tick");
        }
    }
    
    #[test]
    fn test_data_resampling() {
        let loader = HistoricalDataLoader::new(DataSource::CustomCSV {
            file_path: PathBuf::from("dummy"),
            delimiter: ',',
            has_header: false,
            timestamp_col: 0,
            price_col: 1,
            quantity_col: 2,
            side_col: 3,
        });
        
        // Create test data
        let base_time = Utc::now();
        let mut data = Vec::new();
        
        for i in 0..100 {
            data.push(TickData {
                timestamp: base_time + chrono::Duration::milliseconds(i * 10),
                symbol: Symbol("TEST".to_string()),
                exchange: Exchange("TEST".to_string()),
                sequence: i as u64,
                tick_type: TickType::Trade(TradeData {
                    price: Price(Decimal::from(50000 + i)),
                    quantity: Quantity(Decimal::from(1)),
                    side: Side::Bid,
                    trade_id: i as u64,
                    is_implied: false,
                }),
            });
        }
        
        // Resample to 100ms intervals
        let resampled = loader.resample(data, 100);
        
        // Should have ~10 samples (100 ticks at 10ms each = 1000ms, resampled to 100ms = 10)
        assert!(resampled.len() >= 9 && resampled.len() <= 11);
    }
}