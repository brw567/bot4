// Cross-Source Data Reconciliation
// Based on Google's Mesa architecture and Facebook's data infrastructure
//
// Theory: Multiple data sources provide redundancy but require reconciliation
// Uses consensus algorithms, fuzzy matching, and statistical validation
//
// Applications in trading:
// - Verify prices across exchanges
// - Detect arbitrage opportunities
// - Identify feed manipulation
// - Ensure data consistency

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use statistical::{median, standard_deviation};

use super::DataBatch;

/// Reconciliation configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ReconciliationConfig {
    pub min_sources_for_consensus: usize,
    pub price_tolerance_percent: f64,
    pub volume_tolerance_percent: f64,
    pub timestamp_tolerance_ms: i64,
    pub consensus_threshold: f64,  // Percentage of sources that must agree
    pub enable_outlier_detection: bool,
    pub outlier_std_multiplier: f64,
    pub enable_fuzzy_matching: bool,
    pub fuzzy_threshold: f64,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            min_sources_for_consensus: 2,
            price_tolerance_percent: 0.1,  // 0.1% price difference allowed
            volume_tolerance_percent: 1.0,  // 1% volume difference allowed
            timestamp_tolerance_ms: 100,    // 100ms timestamp difference
            consensus_threshold: 0.66,      // 2/3 must agree
            enable_outlier_detection: true,
            outlier_std_multiplier: 3.0,
            enable_fuzzy_matching: true,
            fuzzy_threshold: 0.95,
        }
    }
}

/// Data source for reconciliation
#[async_trait]
pub trait DataSource: Send + Sync {
    async fn get_latest_data(&self, symbol: &str) -> Result<SourceData>;
    fn name(&self) -> String;
    fn reliability_score(&self) -> f64;
    fn latency_ms(&self) -> u64;
}

/// Mock data source for testing
pub struct MockSource {
    name: String,
    reliability: f64,
    latency: u64,
}

#[async_trait]
impl DataSource for MockSource {
    async fn get_latest_data(&self, symbol: &str) -> Result<SourceData> {
        Ok(SourceData {
            source: self.name.clone(),
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            price: 50000.0 + rand::random::<f64>() * 100.0,
            volume: 1000.0 + rand::random::<f64>() * 100.0,
            bid: 49950.0,
            ask: 50050.0,
            metadata: HashMap::new(),
        })
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
    
    fn reliability_score(&self) -> f64 {
        self.reliability
    }
    
    fn latency_ms(&self) -> u64 {
        self.latency
    }
}

/// Cross-source reconciliation engine
pub struct CrossSourceReconciler {
    config: ReconciliationConfig,
    
    // Registered data sources
    sources: Arc<RwLock<Vec<Arc<dyn DataSource>>>>,
    
    // Reconciliation history
    history: Arc<RwLock<Vec<ReconciliationResult>>>,
    
    // Source performance tracking
    source_metrics: Arc<RwLock<HashMap<String, SourceMetrics>>>,
    
    // Consensus cache
    consensus_cache: Arc<RwLock<HashMap<String, ConsensusData>>>,
}

impl CrossSourceReconciler {
    /// Create new reconciliation engine
    pub async fn new(config: ReconciliationConfig) -> Result<Self> {
        info!("Initializing Cross-Source Reconciler with {} minimum sources", 
              config.min_sources_for_consensus);
        
        let mut reconciler = Self {
            config,
            sources: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            source_metrics: Arc::new(RwLock::new(HashMap::new())),
            consensus_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Initialize default sources
        reconciler.initialize_sources().await?;
        
        Ok(reconciler)
    }
    
    /// Initialize data sources
    async fn initialize_sources(&mut self) -> Result<()> {
        let mut sources = self.sources.write().await;
        
        // Add mock sources for testing
        sources.push(Arc::new(MockSource {
            name: "binance".to_string(),
            reliability: 0.99,
            latency: 10,
        }));
        
        sources.push(Arc::new(MockSource {
            name: "kraken".to_string(),
            reliability: 0.98,
            latency: 15,
        }));
        
        sources.push(Arc::new(MockSource {
            name: "coinbase".to_string(),
            reliability: 0.97,
            latency: 20,
        }));
        
        // Initialize metrics for each source
        let mut metrics = self.source_metrics.write().await;
        for source in sources.iter() {
            metrics.insert(source.name(), SourceMetrics::default());
        }
        
        info!("Initialized {} data sources for reconciliation", sources.len());
        Ok(())
    }
    
    /// Reconcile data from multiple sources
    pub async fn reconcile(&self, data: &DataBatch) -> Result<ReconciliationResult> {
        let symbol = &data.symbol;
        let timestamp = data.timestamp;
        
        // Collect data from all sources
        let source_data = self.collect_source_data(symbol).await?;
        
        if source_data.len() < self.config.min_sources_for_consensus {
            return Ok(ReconciliationResult {
                symbol: symbol.clone(),
                timestamp,
                is_consistent: false,
                consensus_value: None,
                source_count: source_data.len(),
                agreeing_sources: Vec::new(),
                disagreeing_sources: self.sources.read().await.iter()
                    .map(|s| s.name())
                    .collect(),
                completeness_score: 0.0,
                consistency_score: 0.0,
                confidence: 0.0,
            });
        }
        
        // Check temporal alignment
        let temporal_groups = self.group_by_timestamp(&source_data)?;
        
        // Find consensus for each temporal group
        let mut best_consensus = None;
        let mut best_confidence = 0.0;
        
        for group in temporal_groups.values() {
            let consensus = self.find_consensus(group).await?;
            if consensus.confidence > best_confidence {
                best_confidence = consensus.confidence;
                best_consensus = Some(consensus);
            }
        }
        
        // Detect outliers if enabled
        let outliers = if self.config.enable_outlier_detection {
            self.detect_outliers(&source_data)?
        } else {
            Vec::new()
        };
        
        // Update source metrics
        self.update_source_metrics(&source_data, &outliers).await?;
        
        // Build reconciliation result
        let result = if let Some(consensus) = best_consensus {
            ReconciliationResult {
                symbol: symbol.clone(),
                timestamp,
                is_consistent: consensus.confidence >= self.config.consensus_threshold,
                consensus_value: Some(consensus.value),
                source_count: source_data.len(),
                agreeing_sources: consensus.agreeing_sources,
                disagreeing_sources: consensus.disagreeing_sources,
                completeness_score: self.calculate_completeness(&source_data),
                consistency_score: consensus.confidence,
                confidence: best_confidence,
            }
        } else {
            ReconciliationResult {
                symbol: symbol.clone(),
                timestamp,
                is_consistent: false,
                consensus_value: None,
                source_count: source_data.len(),
                agreeing_sources: Vec::new(),
                disagreeing_sources: self.sources.read().await.iter()
                    .map(|s| s.name())
                    .collect(),
                completeness_score: self.calculate_completeness(&source_data),
                consistency_score: 0.0,
                confidence: 0.0,
            }
        };
        
        // Store in history
        let mut history = self.history.write().await;
        if history.len() >= 10000 {
            history.remove(0);
        }
        history.push(result.clone());
        
        Ok(result)
    }
    
    /// Force reconciliation for specific symbol
    pub async fn force_reconcile(&self, symbol: &str) -> Result<ReconciliationResult> {
        info!("Forcing reconciliation for symbol: {}", symbol);
        
        // Create synthetic data batch for reconciliation
        let data = DataBatch {
            symbol: symbol.to_string(),
            data_type: super::super::DataType::Price,
            timestamp: Utc::now(),
            values: Vec::new(),
            source: "reconciler".to_string(),
            metadata: None,
        };
        
        self.reconcile(&data).await
    }
    
    /// Collect data from all sources
    async fn collect_source_data(&self, symbol: &str) -> Result<Vec<SourceData>> {
        let sources = self.sources.read().await;
        let mut source_data = Vec::new();
        
        // Parallel data collection with timeout
        use futures::future::join_all;
        let futures: Vec<_> = sources.iter()
            .map(|source| {
                let symbol = symbol.to_string();
                let source = source.clone();
                async move {
                    tokio::time::timeout(
                        std::time::Duration::from_millis(100),
                        source.get_latest_data(&symbol)
                    ).await
                }
            })
            .collect();
        
        let results = join_all(futures).await;
        
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(Ok(data)) => {
                    source_data.push(data.clone());
                }
                Ok(Err(e)) => {
                    warn!("Source {} failed: {}", sources[i].name(), e);
                }
                Err(_) => {
                    warn!("Source {} timed out", sources[i].name());
                }
            }
        }
        
        Ok(source_data)
    }
    
    /// Group source data by timestamp proximity
    fn group_by_timestamp(&self, data: &[SourceData]) -> Result<HashMap<i64, Vec<SourceData>>> {
        let mut groups: HashMap<i64, Vec<SourceData>> = HashMap::new();
        let tolerance = self.config.timestamp_tolerance_ms;
        
        for item in data {
            let timestamp_ms = item.timestamp.timestamp_millis();
            let bucket = (timestamp_ms / tolerance) * tolerance;
            groups.entry(bucket).or_insert_with(Vec::new).push(item.clone());
        }
        
        Ok(groups)
    }
    
    /// Find consensus among data points
    async fn find_consensus(&self, data: &[SourceData]) -> Result<ConsensusData> {
        if data.is_empty() {
            return Err(anyhow!("No data for consensus"));
        }
        
        let prices: Vec<f64> = data.iter().map(|d| d.price).collect();
        let median_price = median(&prices);
        
        // Determine which sources agree within tolerance
        let mut agreeing = Vec::new();
        let mut disagreeing = Vec::new();
        
        for item in data {
            let deviation = ((item.price - median_price) / median_price).abs();
            if deviation <= self.config.price_tolerance_percent / 100.0 {
                agreeing.push(item.source.clone());
            } else {
                disagreeing.push(item.source.clone());
            }
        }
        
        let confidence = agreeing.len() as f64 / data.len() as f64;
        
        Ok(ConsensusData {
            value: median_price,
            confidence,
            agreeing_sources: agreeing,
            disagreeing_sources: disagreeing,
            timestamp: data[0].timestamp,
        })
    }
    
    /// Detect outliers in source data
    fn detect_outliers(&self, data: &[SourceData]) -> Result<Vec<String>> {
        if data.len() < 3 {
            return Ok(Vec::new());
        }
        
        let prices: Vec<f64> = data.iter().map(|d| d.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let std_dev = standard_deviation(&prices, Some(mean));
        
        let mut outliers = Vec::new();
        
        for item in data {
            let z_score = (item.price - mean).abs() / std_dev;
            if z_score > self.config.outlier_std_multiplier {
                outliers.push(item.source.clone());
                warn!("Outlier detected from {}: price={:.2}, z-score={:.2}", 
                      item.source, item.price, z_score);
            }
        }
        
        Ok(outliers)
    }
    
    /// Update source performance metrics
    async fn update_source_metrics(
        &self,
        data: &[SourceData],
        outliers: &[String],
    ) -> Result<()> {
        let mut metrics = self.source_metrics.write().await;
        
        for item in data {
            if let Some(metric) = metrics.get_mut(&item.source) {
                metric.total_reports += 1;
                
                if outliers.contains(&item.source) {
                    metric.outlier_count += 1;
                }
                
                metric.last_report = item.timestamp;
                metric.reliability = 1.0 - (metric.outlier_count as f64 / metric.total_reports as f64);
            }
        }
        
        Ok(())
    }
    
    /// Calculate data completeness score
    fn calculate_completeness(&self, data: &[SourceData]) -> f64 {
        let total_sources = self.sources.try_read()
            .map(|s| s.len())
            .unwrap_or(1);
        
        data.len() as f64 / total_sources as f64
    }
    
    /// Get reconciliation statistics
    pub async fn get_statistics(&self) -> ReconciliationStatistics {
        let history = self.history.read().await;
        let metrics = self.source_metrics.read().await;
        
        let total = history.len();
        let consistent = history.iter().filter(|r| r.is_consistent).count();
        
        let avg_confidence = if total > 0 {
            history.iter().map(|r| r.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };
        
        let source_stats: HashMap<String, SourceStatistics> = metrics.iter()
            .map(|(name, metric)| {
                (name.clone(), SourceStatistics {
                    total_reports: metric.total_reports,
                    outlier_count: metric.outlier_count,
                    reliability: metric.reliability,
                    last_report: metric.last_report,
                })
            })
            .collect();
        
        ReconciliationStatistics {
            total_reconciliations: total,
            consistent_count: consistent,
            inconsistent_count: total - consistent,
            average_confidence: avg_confidence,
            source_statistics: source_stats,
        }
    }
}

/// Source data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceData {
    pub source: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Consensus data
#[derive(Debug, Clone)]
struct ConsensusData {
    value: f64,
    confidence: f64,
    agreeing_sources: Vec<String>,
    disagreeing_sources: Vec<String>,
    timestamp: DateTime<Utc>,
}

/// Source performance metrics
#[derive(Debug, Clone, Default)]
struct SourceMetrics {
    total_reports: usize,
    outlier_count: usize,
    reliability: f64,
    last_report: DateTime<Utc>,
}

/// Reconciliation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationResult {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub is_consistent: bool,
    pub consensus_value: Option<f64>,
    pub source_count: usize,
    pub agreeing_sources: Vec<String>,
    pub disagreeing_sources: Vec<String>,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub confidence: f64,
}

/// Reconciliation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationStatistics {
    pub total_reconciliations: usize,
    pub consistent_count: usize,
    pub inconsistent_count: usize,
    pub average_confidence: f64,
    pub source_statistics: HashMap<String, SourceStatistics>,
}

/// Source statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStatistics {
    pub total_reports: usize,
    pub outlier_count: usize,
    pub reliability: f64,
    pub last_report: DateTime<Utc>,
}

// Statistical helper module
mod statistical {
    pub fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
    
    pub fn standard_deviation(values: &[f64], mean: Option<f64>) -> f64 {
        let m = mean.unwrap_or_else(|| values.iter().sum::<f64>() / values.len() as f64);
        let variance = values.iter()
            .map(|v| (v - m).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

extern crate rand;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_reconciliation() {
        let config = ReconciliationConfig::default();
        let reconciler = CrossSourceReconciler::new(config).await.unwrap();
        
        let data = DataBatch {
            symbol: "BTC-USDT".to_string(),
            data_type: super::super::DataType::Price,
            timestamp: Utc::now(),
            values: vec![50000.0],
            source: "test".to_string(),
            metadata: None,
        };
        
        let result = reconciler.reconcile(&data).await.unwrap();
        assert!(result.source_count > 0);
    }
    
    #[tokio::test]
    async fn test_outlier_detection() {
        let config = ReconciliationConfig {
            enable_outlier_detection: true,
            outlier_std_multiplier: 2.0,
            ..Default::default()
        };
        
        let reconciler = CrossSourceReconciler::new(config).await.unwrap();
        
        // Test data with outlier
        let data = vec![
            SourceData {
                source: "source1".to_string(),
                symbol: "TEST".to_string(),
                timestamp: Utc::now(),
                price: 100.0,
                volume: 1000.0,
                bid: 99.0,
                ask: 101.0,
                metadata: HashMap::new(),
            },
            SourceData {
                source: "source2".to_string(),
                symbol: "TEST".to_string(),
                timestamp: Utc::now(),
                price: 101.0,
                volume: 1000.0,
                bid: 100.0,
                ask: 102.0,
                metadata: HashMap::new(),
            },
            SourceData {
                source: "outlier".to_string(),
                symbol: "TEST".to_string(),
                timestamp: Utc::now(),
                price: 200.0,  // Outlier
                volume: 1000.0,
                bid: 199.0,
                ask: 201.0,
                metadata: HashMap::new(),
            },
        ];
        
        let outliers = reconciler.detect_outliers(&data).unwrap();
        assert!(outliers.contains(&"outlier".to_string()));
    }
}