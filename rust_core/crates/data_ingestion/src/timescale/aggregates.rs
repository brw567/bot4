// Continuous Aggregates Management
// DEEP DIVE: Hierarchical aggregation for maximum efficiency

use std::sync::Arc;
use anyhow::{Result, Context};
use deadpool_postgres::Pool;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, instrument};

/// Aggregate timeframe levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AggregateLevel {
    OneSecond,
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    OneHour,
    FourHours,
    OneDay,
}

impl AggregateLevel {
    pub fn view_name(&self) -> &str {
        match self {
            Self::OneSecond => "aggregates.ohlcv_1s",
            Self::OneMinute => "aggregates.ohlcv_1m",
            Self::FiveMinutes => "aggregates.ohlcv_5m",
            Self::FifteenMinutes => "aggregates.ohlcv_15m",
            Self::OneHour => "aggregates.ohlcv_1h",
            Self::FourHours => "aggregates.ohlcv_4h",
            Self::OneDay => "aggregates.ohlcv_1d",
        }
    }
    
    pub fn interval(&self) -> &str {
        match self {
            Self::OneSecond => "1 second",
            Self::OneMinute => "1 minute",
            Self::FiveMinutes => "5 minutes",
            Self::FifteenMinutes => "15 minutes",
            Self::OneHour => "1 hour",
            Self::FourHours => "4 hours",
            Self::OneDay => "1 day",
        }
    }
    
    pub fn source_view(&self) -> Option<&str> {
        match self {
            Self::OneSecond => None, // Direct from ticks
            Self::OneMinute => Some("aggregates.ohlcv_1s"),
            Self::FiveMinutes => Some("aggregates.ohlcv_1m"),
            Self::FifteenMinutes => Some("aggregates.ohlcv_5m"),
            Self::OneHour => Some("aggregates.ohlcv_15m"),
            Self::FourHours => Some("aggregates.ohlcv_1h"),
            Self::OneDay => Some("aggregates.ohlcv_4h"),
        }
    }
}

/// Manager for continuous aggregate operations
/// TODO: Add docs
pub struct AggregateManager {
    pool: Arc<Pool>,
}

impl AggregateManager {
    pub async fn new(pool: Arc<Pool>) -> Result<Self> {
        let manager = Self { pool };
        
        // Verify aggregates exist
        manager.verify_aggregates().await?;
        
        Ok(manager)
    }
    
    /// Verify all continuous aggregates are configured
    async fn verify_aggregates(&self) -> Result<()> {
        let conn = self.pool.get().await?;
        
        let rows = conn.query(
            "SELECT view_name, materialization_hypertable_name, watermark
             FROM timescaledb_information.continuous_aggregates
             ORDER BY view_name",
            &[],
        ).await?;
        
        for row in rows {
            let view: String = row.get(0);
            let table: String = row.get(1);
            let watermark: Option<DateTime<Utc>> = row.get(2);
            
            debug!(
                "Aggregate {} -> {} (watermark: {:?})",
                view, table, watermark
            );
        }
        
        Ok(())
    }
    
    /// Create or update continuous aggregate
    pub async fn ensure_aggregate(&self, level: AggregateLevel) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Check if aggregate exists
        let exists = conn.query_one(
            "SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates
             WHERE view_name = $1",
            &[&level.view_name()],
        ).await?;
        
        let count: i64 = exists.get(0);
        
        if count > 0 {
            debug!("Aggregate {} already exists", level.view_name());
            return Ok(());
        }
        
        // Create aggregate based on level
        let create_sql = self.generate_aggregate_sql(level);
        conn.execute(&create_sql, &[]).await?;
        
        info!("Created continuous aggregate: {}", level.view_name());
        
        // Add refresh policy
        self.add_refresh_policy(level).await?;
        
        Ok(())
    }
    
    /// Generate SQL for creating continuous aggregate
    fn generate_aggregate_sql(&self, level: AggregateLevel) -> String {
        match level {
            AggregateLevel::OneSecond => {
                // Direct from tick data
                format!(
                    "CREATE MATERIALIZED VIEW {} 
                    WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
                    SELECT 
                        time_bucket('{}', time) AS time,
                        exchange,
                        symbol,
                        FIRST(price, time) AS open,
                        MAX(price) AS high,
                        MIN(price) AS low,
                        LAST(price, time) AS close,
                        SUM(volume) AS volume,
                        COUNT(*) AS trades,
                        SUM(CASE WHEN side = 'B' THEN volume ELSE 0 END) AS buy_volume,
                        SUM(CASE WHEN side = 'S' THEN volume ELSE 0 END) AS sell_volume,
                        STDDEV(price) AS volatility,
                        SUM(price * volume) / NULLIF(SUM(volume), 0) AS vwap
                    FROM market_data.ticks
                    GROUP BY time_bucket('{}', time), exchange, symbol
                    WITH NO DATA",
                    level.view_name(),
                    level.interval(),
                    level.interval()
                )
            }
            _ => {
                // Hierarchical from previous level
                let source = level.source_view().unwrap();
                format!(
                    "CREATE MATERIALIZED VIEW {}
                    WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
                    SELECT 
                        time_bucket('{}', time) AS time,
                        exchange,
                        symbol,
                        FIRST(open, time) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        LAST(close, time) AS close,
                        SUM(volume) AS volume,
                        SUM(trades) AS trades,
                        SUM(buy_volume) AS buy_volume,
                        SUM(sell_volume) AS sell_volume,
                        AVG(volatility) AS volatility,
                        SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap
                    FROM {}
                    GROUP BY time_bucket('{}', time), exchange, symbol
                    WITH NO DATA",
                    level.view_name(),
                    level.interval(),
                    source,
                    level.interval()
                )
            }
        }
    }
    
    /// Add automatic refresh policy for aggregate
    async fn add_refresh_policy(&self, level: AggregateLevel) -> Result<()> {
        let conn = self.pool.get().await?;
        
        let (start_offset, end_offset, schedule_interval) = match level {
            AggregateLevel::OneSecond => ("10 seconds", "1 second", "1 second"),
            AggregateLevel::OneMinute => ("5 minutes", "10 seconds", "10 seconds"),
            AggregateLevel::FiveMinutes => ("15 minutes", "1 minute", "1 minute"),
            AggregateLevel::FifteenMinutes => ("1 hour", "5 minutes", "5 minutes"),
            AggregateLevel::OneHour => ("4 hours", "15 minutes", "15 minutes"),
            AggregateLevel::FourHours => ("1 day", "1 hour", "1 hour"),
            AggregateLevel::OneDay => ("7 days", "4 hours", "4 hours"),
        };
        
        conn.execute(
            "SELECT add_continuous_aggregate_policy($1,
                start_offset => $2::INTERVAL,
                end_offset => $3::INTERVAL,
                schedule_interval => $4::INTERVAL,
                if_not_exists => TRUE)",
            &[&level.view_name(), &start_offset, &end_offset, &schedule_interval],
        ).await?;
        
        debug!(
            "Added refresh policy for {}: every {}",
            level.view_name(),
            schedule_interval
        );
        
        Ok(())
    }
    
    /// Manually refresh aggregate for specific time range
    #[instrument(skip(self))]
    pub async fn refresh_aggregate(&self, level: AggregateLevel) -> Result<()> {
        let conn = self.pool.get().await?;
        
        conn.execute(
            "CALL refresh_continuous_aggregate($1, NULL, NULL)",
            &[&level.view_name()],
        ).await?;
        
        debug!("Refreshed aggregate: {}", level.view_name());
        Ok(())
    }
    
    /// Refresh aggregate for specific time window
    pub async fn refresh_aggregate_window(
        &self,
        level: AggregateLevel,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<()> {
        let conn = self.pool.get().await?;
        
        conn.execute(
            "CALL refresh_continuous_aggregate($1, $2, $3)",
            &[&level.view_name(), &start, &end],
        ).await?;
        
        debug!(
            "Refreshed {} from {} to {}",
            level.view_name(),
            start,
            end
        );
        
        Ok(())
    }
    
    /// Get aggregate freshness (how far behind real-time)
    pub async fn get_aggregate_freshness(&self) -> Result<Vec<AggregateFreshness>> {
        let conn = self.pool.get().await?;
        
        let rows = conn.query(
            "SELECT 
                view_name,
                watermark,
                NOW() - watermark as lag,
                EXTRACT(EPOCH FROM (NOW() - watermark)) as lag_seconds
             FROM timescaledb_information.continuous_aggregates
             ORDER BY view_name",
            &[],
        ).await?;
        
        let mut results = Vec::new();
        
        for row in rows {
            results.push(AggregateFreshness {
                view_name: row.get(0),
                watermark: row.get(1),
                lag_seconds: row.get::<_, Option<f64>>(3).unwrap_or(0.0),
            });
        }
        
        Ok(results)
    }
    
    /// Optimize aggregate compression
    pub async fn optimize_aggregate_compression(&self, level: AggregateLevel) -> Result<()> {
        let conn = self.pool.get().await?;
        
        // Get materialization hypertable name
        let row = conn.query_one(
            "SELECT materialization_hypertable_name 
             FROM timescaledb_information.continuous_aggregates
             WHERE view_name = $1",
            &[&level.view_name()],
        ).await?;
        
        let hypertable: String = row.get(0);
        
        // Add compression policy if not exists
        conn.execute(
            "ALTER TABLE {} SET (
                timescaledb.compress,
                timescaledb.compress_orderby = 'time DESC',
                timescaledb.compress_segmentby = 'exchange, symbol'
            )",
            &[&hypertable],
        ).await.ok(); // Ignore if already set
        
        // Add compression policy
        let compress_after = match level {
            AggregateLevel::OneSecond => "1 hour",
            AggregateLevel::OneMinute => "6 hours",
            AggregateLevel::FiveMinutes => "1 day",
            _ => "7 days",
        };
        
        conn.execute(
            "SELECT add_compression_policy($1, compress_after => $2::INTERVAL, if_not_exists => TRUE)",
            &[&hypertable, &compress_after],
        ).await?;
        
        info!("Optimized compression for {}", level.view_name());
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AggregateFreshness {
    pub view_name: String,
    pub watermark: Option<DateTime<Utc>>,
    pub lag_seconds: f64,
}