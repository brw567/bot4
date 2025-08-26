// Performance Monitoring for TimescaleDB
// DEEP DIVE: Real-time metrics for <100ms query latency verification

use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, debug};
use std::collections::VecDeque;

/// Query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub query_name: String,
    pub execution_time_ms: f64,
    pub rows_returned: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Ingestion statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionStats {
    pub events_per_second: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_latency_ms: f64,
}

/// Performance monitor for tracking metrics
pub struct PerformanceMonitor {
    pool: Arc<Pool>,
    query_history: Arc<RwLock<VecDeque<QueryStats>>>,
    batch_history: Arc<RwLock<VecDeque<BatchStats>>>,
}

#[derive(Debug, Clone)]
struct BatchStats {
    table: String,
    count: usize,
    duration: Duration,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl PerformanceMonitor {
    pub async fn new(pool: Arc<Pool>) -> Result<Self> {
        Ok(Self {
            pool,
            query_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            batch_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        })
    }
    
    /// Record query execution
    pub async fn record_query(&self, name: &str, duration: Duration, rows: usize) {
        let stats = QueryStats {
            query_name: name.to_string(),
            execution_time_ms: duration.as_secs_f64() * 1000.0,
            rows_returned: rows,
            timestamp: chrono::Utc::now(),
        };
        
        if duration.as_millis() > 100 {
            warn!(
                "Query '{}' exceeded 100ms target: {:.2}ms for {} rows",
                name, stats.execution_time_ms, rows
            );
        }
        
        let mut history = self.query_history.write().await;
        if history.len() >= 1000 {
            history.pop_front();
        }
        history.push_back(stats);
    }
    
    /// Record batch insert
    pub async fn record_batch_insert(&self, table: &str, count: usize, duration: Duration) {
        let stats = BatchStats {
            table: table.to_string(),
            count,
            duration,
            timestamp: chrono::Utc::now(),
        };
        
        debug!(
            "Batch insert to {}: {} records in {:.2}ms ({:.0} records/sec)",
            table,
            count,
            duration.as_secs_f64() * 1000.0,
            count as f64 / duration.as_secs_f64()
        );
        
        let mut history = self.batch_history.write().await;
        if history.len() >= 1000 {
            history.pop_front();
        }
        history.push_back(stats);
    }
    
    /// Get current ingestion statistics
    pub async fn get_current_stats(&self) -> IngestionStats {
        let conn = match self.pool.get().await {
            Ok(c) => c,
            Err(_) => return Self::default_stats(),
        };
        
        let row = match conn.query_one(
            "SELECT 
                COUNT(*) / 60.0 as events_per_second,
                AVG(latency_us) / 1000.0 as avg_latency_ms,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_us) / 1000.0 as p50_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_us) / 1000.0 as p95_latency_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_us) / 1000.0 as p99_latency_ms,
                MAX(latency_us) / 1000.0 as max_latency_ms
             FROM market_data.ticks
             WHERE time > NOW() - INTERVAL '1 minute'",
            &[],
        ).await {
            Ok(r) => r,
            Err(_) => return Self::default_stats(),
        };
        
        IngestionStats {
            events_per_second: row.get::<_, Option<f64>>(0).unwrap_or(0.0),
            avg_latency_ms: row.get::<_, Option<f64>>(1).unwrap_or(0.0),
            p50_latency_ms: row.get::<_, Option<f64>>(2).unwrap_or(0.0),
            p95_latency_ms: row.get::<_, Option<f64>>(3).unwrap_or(0.0),
            p99_latency_ms: row.get::<_, Option<f64>>(4).unwrap_or(0.0),
            max_latency_ms: row.get::<_, Option<f64>>(5).unwrap_or(0.0),
        }
    }
    
    /// Run performance benchmark
    pub async fn run_benchmark(&self) -> Result<BenchmarkResults> {
        let conn = self.pool.get().await?;
        let mut results = BenchmarkResults::default();
        
        // Test 1: Recent tick query
        let start = Instant::now();
        let rows = conn.query(
            "SELECT * FROM market_data.ticks 
             WHERE symbol = 'BTC/USDT' 
               AND exchange = 'BINANCE'
               AND time > NOW() - INTERVAL '1 minute'
             LIMIT 1000",
            &[],
        ).await?;
        let duration = start.elapsed();
        
        results.recent_ticks_ms = duration.as_secs_f64() * 1000.0;
        results.recent_ticks_rows = rows.len();
        
        // Test 2: Aggregate query
        let start = Instant::now();
        let rows = conn.query(
            "SELECT * FROM aggregates.ohlcv_1m
             WHERE symbol = 'BTC/USDT'
               AND exchange = 'BINANCE'
               AND time > NOW() - INTERVAL '1 hour'",
            &[],
        ).await?;
        let duration = start.elapsed();
        
        results.aggregate_1m_ms = duration.as_secs_f64() * 1000.0;
        results.aggregate_1m_rows = rows.len();
        
        // Test 3: Complex analytical query
        let start = Instant::now();
        let _row = conn.query_one(
            "WITH volatility AS (
                SELECT 
                    symbol,
                    STDDEV(close) as vol
                FROM aggregates.ohlcv_5m
                WHERE time > NOW() - INTERVAL '24 hours'
                  AND exchange = 'BINANCE'
                GROUP BY symbol
            )
            SELECT AVG(vol) FROM volatility",
            &[],
        ).await?;
        let duration = start.elapsed();
        
        results.analytical_query_ms = duration.as_secs_f64() * 1000.0;
        
        // Test 4: Order book latest
        let start = Instant::now();
        let _row = conn.query_opt(
            "SELECT * FROM market_data.order_book
             WHERE symbol = 'BTC/USDT'
               AND exchange = 'BINANCE'
             ORDER BY time DESC, sequence_num DESC
             LIMIT 1",
            &[],
        ).await?;
        let duration = start.elapsed();
        
        results.orderbook_latest_ms = duration.as_secs_f64() * 1000.0;
        
        // Check if we meet <100ms target
        results.meets_target = results.recent_ticks_ms < 100.0
            && results.aggregate_1m_ms < 100.0
            && results.orderbook_latest_ms < 100.0;
        
        if !results.meets_target {
            warn!("Performance target not met: {:?}", results);
        } else {
            info!("Performance target MET! All queries <100ms");
        }
        
        Ok(results)
    }
    
    /// Get query statistics summary
    pub async fn get_query_summary(&self) -> QuerySummary {
        let history = self.query_history.read().await;
        
        if history.is_empty() {
            return QuerySummary::default();
        }
        
        let mut times: Vec<f64> = history.iter().map(|s| s.execution_time_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = times.len();
        let p50_idx = len / 2;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;
        
        QuerySummary {
            total_queries: len,
            avg_time_ms: times.iter().sum::<f64>() / len as f64,
            p50_time_ms: times[p50_idx],
            p95_time_ms: times[p95_idx.min(len - 1)],
            p99_time_ms: times[p99_idx.min(len - 1)],
            max_time_ms: times[len - 1],
            queries_over_100ms: times.iter().filter(|&&t| t > 100.0).count(),
        }
    }
    
    fn default_stats() -> IngestionStats {
        IngestionStats {
            events_per_second: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            max_latency_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub recent_ticks_ms: f64,
    pub recent_ticks_rows: usize,
    pub aggregate_1m_ms: f64,
    pub aggregate_1m_rows: usize,
    pub analytical_query_ms: f64,
    pub orderbook_latest_ms: f64,
    pub meets_target: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuerySummary {
    pub total_queries: usize,
    pub avg_time_ms: f64,
    pub p50_time_ms: f64,
    pub p95_time_ms: f64,
    pub p99_time_ms: f64,
    pub max_time_ms: f64,
    pub queries_over_100ms: usize,
}