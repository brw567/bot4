// Bucketed Aggregator - 1-5ms window aggregation
// DEEP DIVE: High-frequency bucketing for microstructure analysis
//
// References:
// - "Econophysics of Order-driven Markets" - Abergel et al. (2011)
// - "High-Frequency Trading and Price Discovery" - Brogaard et al. (2014)
// - "Intraday Periodicity and Volatility" - Andersen & Bollerslev (1997)

use std::sync::Arc;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{debug, info, warn, instrument};

use types::{Price, Quantity, Symbol};

// Temporary metrics collector trait until infrastructure is fixed
trait MetricsCollector: Send + Sync {
    fn record(&self, value: f64);
}

struct NoopMetricsCollector;

impl MetricsCollector for NoopMetricsCollector {
    fn record(&self, _value: f64) {}
}

/// Window type for aggregation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    /// Fixed time windows (tumbling)
    Tumbling,
    
    /// Overlapping windows (sliding)
    Sliding,
    
    /// Count-based windows
    Count,
    
    /// Volume-based windows
    Volume,
    
    /// Tick-based windows
    Tick,
}

/// Aggregate window
#[derive(Debug, Clone)]
pub struct AggregateWindow {
    pub window_id: u64,
    pub symbol: Symbol,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_ms: u64,
    
    // OHLCV data
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
    
    // Additional statistics
    pub trade_count: u32,
    pub vwap: Price,  // Volume-weighted average price
    pub twap: Price,  // Time-weighted average price
    pub spread_avg: Decimal,
    pub spread_max: Decimal,
    
    // Microstructure metrics
    pub buy_volume: Quantity,
    pub sell_volume: Quantity,
    pub order_imbalance: f64,
    pub tick_direction: i8,  // -1, 0, 1
    
    // Volatility within window
    pub realized_volatility: f64,
    pub high_low_range: Decimal,
}

/// Bucket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConfig {
    /// Window duration in milliseconds
    pub window_duration_ms: u64,
    
    /// Window type
    pub window_type: WindowType,
    
    /// Maximum windows to keep in memory
    pub max_windows: usize,
    
    /// Enable microstructure metrics
    pub enable_microstructure: bool,
    
    /// Enable volatility calculation
    pub enable_volatility: bool,
    
    /// Aggregation levels (1ms, 5ms, 10ms, 100ms, 1s)
    pub aggregation_levels: Vec<u64>,
    
    /// Sliding window overlap (percentage)
    pub sliding_overlap: f64,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            window_duration_ms: 5,  // 5ms default
            window_type: WindowType::Tumbling,
            max_windows: 10000,
            enable_microstructure: true,
            enable_volatility: true,
            aggregation_levels: vec![1, 5, 10, 100, 1000],  // 1ms to 1s
            sliding_overlap: 0.5,  // 50% overlap for sliding windows
        }
    }
}

/// Bucket statistics
#[derive(Debug, Clone)]
pub struct BucketStats {
    pub total_windows: u64,
    pub active_windows: usize,
    pub events_processed: u64,
    pub avg_events_per_window: f64,
    pub compression_ratio: f64,
}

/// Trade event for aggregation
#[derive(Debug, Clone)]
struct TradeEvent {
    timestamp: DateTime<Utc>,
    price: Price,
    quantity: Quantity,
    is_buy: bool,
}

/// Multi-level aggregator
struct MultiLevelAggregator {
    levels: BTreeMap<u64, LevelAggregator>,
}

impl MultiLevelAggregator {
    fn new(levels: &[u64]) -> Self {
        let mut level_map = BTreeMap::new();
        for &duration_ms in levels {
            level_map.insert(duration_ms, LevelAggregator::new(duration_ms));
        }
        Self { levels: level_map }
    }
    
    fn add_event(&mut self, event: &TradeEvent) {
        for (_, aggregator) in self.levels.iter_mut() {
            aggregator.add_event(event);
        }
    }
    
    fn get_windows(&self, level_ms: u64) -> Option<Vec<AggregateWindow>> {
        self.levels.get(&level_ms).map(|agg| agg.get_completed_windows())
    }
}

/// Single level aggregator
struct LevelAggregator {
    duration_ms: u64,
    current_window: Option<WindowBuilder>,
    completed_windows: VecDeque<AggregateWindow>,
}

impl LevelAggregator {
    fn new(duration_ms: u64) -> Self {
        Self {
            duration_ms,
            current_window: None,
            completed_windows: VecDeque::with_capacity(1000),
        }
    }
    
    fn add_event(&mut self, event: &TradeEvent) {
        // Check if we need a new window
        let window_start = self.get_window_start(event.timestamp);
        
        if let Some(ref mut window) = self.current_window {
            if window.start_time != window_start {
                // Complete current window
                if let Some(completed) = window.complete() {
                    self.completed_windows.push_back(completed);
                    if self.completed_windows.len() > 1000 {
                        self.completed_windows.pop_front();
                    }
                }
                // Start new window
                self.current_window = Some(WindowBuilder::new(
                    Symbol("".to_string()), // Would be set properly
                    window_start,
                    self.duration_ms,
                ));
            }
        } else {
            // Create first window
            self.current_window = Some(WindowBuilder::new(
                Symbol("".to_string()),
                window_start,
                self.duration_ms,
            ));
        }
        
        // Add event to current window
        if let Some(ref mut window) = self.current_window {
            window.add_trade(event);
        }
    }
    
    fn get_window_start(&self, timestamp: DateTime<Utc>) -> DateTime<Utc> {
        let millis = timestamp.timestamp_millis();
        let window_millis = (millis / self.duration_ms as i64) * self.duration_ms as i64;
        DateTime::from_timestamp_millis(window_millis).unwrap()
    }
    
    fn get_completed_windows(&self) -> Vec<AggregateWindow> {
        self.completed_windows.iter().cloned().collect()
    }
}

/// Window builder for incremental aggregation
struct WindowBuilder {
    symbol: Symbol,
    start_time: DateTime<Utc>,
    duration_ms: u64,
    window_id: u64,
    
    // Aggregated data
    first_trade: Option<TradeEvent>,
    last_trade: Option<TradeEvent>,
    high_price: Option<Price>,
    low_price: Option<Price>,
    total_volume: Decimal,
    trade_count: u32,
    
    // For VWAP calculation
    volume_price_sum: Decimal,
    
    // For TWAP calculation
    price_time_sum: Decimal,
    time_weight_sum: i64,
    
    // Microstructure
    buy_volume: Decimal,
    sell_volume: Decimal,
    
    // Spread tracking
    spread_sum: Decimal,
    spread_max: Decimal,
    spread_count: u32,
    
    // For volatility
    price_squares_sum: Decimal,
    price_sum: Decimal,
}

impl WindowBuilder {
    fn new(symbol: Symbol, start_time: DateTime<Utc>, duration_ms: u64) -> Self {
        static WINDOW_COUNTER: AtomicU64 = AtomicU64::new(0);
        
        Self {
            symbol,
            start_time,
            duration_ms,
            window_id: WINDOW_COUNTER.fetch_add(1, Ordering::Relaxed),
            first_trade: None,
            last_trade: None,
            high_price: None,
            low_price: None,
            total_volume: Decimal::ZERO,
            trade_count: 0,
            volume_price_sum: Decimal::ZERO,
            price_time_sum: Decimal::ZERO,
            time_weight_sum: 0,
            buy_volume: Decimal::ZERO,
            sell_volume: Decimal::ZERO,
            spread_sum: Decimal::ZERO,
            spread_max: Decimal::ZERO,
            spread_count: 0,
            price_squares_sum: Decimal::ZERO,
            price_sum: Decimal::ZERO,
        }
    }
    
    fn add_trade(&mut self, event: &TradeEvent) {
        // Update first/last
        if self.first_trade.is_none() {
            self.first_trade = Some(event.clone());
        }
        self.last_trade = Some(event.clone());
        
        // Update high/low
        if let Some(ref mut high) = self.high_price {
            if event.price.0 > high.0 {
                *high = event.price.clone();
            }
        } else {
            self.high_price = Some(event.price.clone());
        }
        
        if let Some(ref mut low) = self.low_price {
            if event.price.0 < low.0 {
                *low = event.price.clone();
            }
        } else {
            self.low_price = Some(event.price.clone());
        }
        
        // Update volume
        self.total_volume += event.quantity.0;
        self.trade_count += 1;
        
        // VWAP calculation
        self.volume_price_sum += event.quantity.0 * event.price.0;
        
        // TWAP calculation
        let time_weight = event.timestamp.timestamp_millis() - self.start_time.timestamp_millis();
        self.price_time_sum += event.price.0 * Decimal::from(time_weight);
        self.time_weight_sum += time_weight;
        
        // Microstructure
        if event.is_buy {
            self.buy_volume += event.quantity.0;
        } else {
            self.sell_volume += event.quantity.0;
        }
        
        // Volatility calculation
        self.price_sum += event.price.0;
        self.price_squares_sum += event.price.0 * event.price.0;
    }
    
    fn add_spread(&mut self, spread: Decimal) {
        self.spread_sum += spread;
        self.spread_count += 1;
        if spread > self.spread_max {
            self.spread_max = spread;
        }
    }
    
    fn complete(self) -> Option<AggregateWindow> {
        let first = self.first_trade?;
        let last = self.last_trade?;
        
        // Calculate VWAP
        let vwap = if self.total_volume > Decimal::ZERO {
            Price(self.volume_price_sum / self.total_volume)
        } else {
            first.price.clone()
        };
        
        // Calculate TWAP
        let twap = if self.time_weight_sum > 0 {
            Price(self.price_time_sum / Decimal::from(self.time_weight_sum))
        } else {
            first.price.clone()
        };
        
        // Calculate spread average
        let spread_avg = if self.spread_count > 0 {
            self.spread_sum / Decimal::from(self.spread_count)
        } else {
            Decimal::ZERO
        };
        
        // Calculate order imbalance
        let total_directional = self.buy_volume + self.sell_volume;
        let order_imbalance = if total_directional > Decimal::ZERO {
            ((self.buy_volume - self.sell_volume) / total_directional)
                .to_f64()
                .unwrap_or(0.0)
        } else {
            0.0
        };
        
        // Calculate tick direction
        let tick_direction = if last.price.0 > first.price.0 {
            1
        } else if last.price.0 < first.price.0 {
            -1
        } else {
            0
        };
        
        // Calculate realized volatility
        let realized_volatility = if self.trade_count > 1 {
            let mean_price = self.price_sum / Decimal::from(self.trade_count);
            let variance = (self.price_squares_sum / Decimal::from(self.trade_count)) 
                         - (mean_price * mean_price);
            variance.to_f64().unwrap_or(0.0).sqrt()
        } else {
            0.0
        };
        
        // High-low range
        let high = self.high_price.unwrap_or(first.price.clone());
        let low = self.low_price.unwrap_or(first.price.clone());
        let high_low_range = high.0 - low.0;
        
        Some(AggregateWindow {
            window_id: self.window_id,
            symbol: self.symbol,
            start_time: self.start_time,
            end_time: self.start_time + Duration::milliseconds(self.duration_ms as i64),
            duration_ms: self.duration_ms,
            open: first.price,
            high,
            low,
            close: last.price,
            volume: Quantity(self.total_volume),
            trade_count: self.trade_count,
            vwap,
            twap,
            spread_avg,
            spread_max: self.spread_max,
            buy_volume: Quantity(self.buy_volume),
            sell_volume: Quantity(self.sell_volume),
            order_imbalance,
            tick_direction,
            realized_volatility,
            high_low_range,
        })
    }
}

/// Main bucketed aggregator implementation
pub struct BucketedAggregator {
    config: Arc<BucketConfig>,
    
    // Multi-level aggregation
    aggregators: Arc<RwLock<BTreeMap<Symbol, MultiLevelAggregator>>>,
    
    // Completed windows
    completed_windows: Arc<RwLock<VecDeque<AggregateWindow>>>,
    
    // Statistics
    windows_created: Arc<AtomicU64>,
    events_processed: Arc<AtomicU64>,
    
    // Metrics
    window_duration_histogram: Arc<dyn MetricsCollector>,
    events_per_window: Arc<dyn MetricsCollector>,
    compression_ratio: Arc<dyn MetricsCollector>,
}

impl BucketedAggregator {
    pub fn new(config: BucketConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config),
            aggregators: Arc::new(RwLock::new(BTreeMap::new())),
            completed_windows: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            windows_created: Arc::new(AtomicU64::new(0)),
            events_processed: Arc::new(AtomicU64::new(0)),
            window_duration_histogram: Arc::new(NoopMetricsCollector),
            events_per_window: Arc::new(NoopMetricsCollector),
            compression_ratio: Arc::new(NoopMetricsCollector),
        })
    }
    
    /// Process trade event
    #[instrument(skip(self), fields(symbol = %symbol.0))]
    pub fn process_trade(
        &self,
        symbol: Symbol,
        price: Price,
        quantity: Quantity,
        is_buy: bool,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        let event = TradeEvent {
            timestamp,
            price,
            quantity,
            is_buy,
        };
        
        // Get or create aggregator for symbol
        let mut aggregators = self.aggregators.write();
        let aggregator = aggregators
            .entry(symbol)
            .or_insert_with(|| MultiLevelAggregator::new(&self.config.aggregation_levels));
        
        // Add event to all levels
        aggregator.add_event(&event);
        
        self.events_processed.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get completed windows for a symbol and level
    pub fn get_windows(
        &self,
        symbol: &Symbol,
        level_ms: u64,
        count: usize,
    ) -> Vec<AggregateWindow> {
        let aggregators = self.aggregators.read();
        
        if let Some(multi_agg) = aggregators.get(symbol) {
            if let Some(windows) = multi_agg.get_windows(level_ms) {
                return windows.into_iter()
                    .rev()
                    .take(count)
                    .collect();
            }
        }
        
        Vec::new()
    }
    
    /// Get latest window for symbol
    pub fn get_latest_window(&self, symbol: &Symbol, level_ms: u64) -> Option<AggregateWindow> {
        self.get_windows(symbol, level_ms, 1).into_iter().next()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> BucketStats {
        let windows = self.windows_created.load(Ordering::Relaxed);
        let events = self.events_processed.load(Ordering::Relaxed);
        
        BucketStats {
            total_windows: windows,
            active_windows: self.completed_windows.read().len(),
            events_processed: events,
            avg_events_per_window: if windows > 0 {
                events as f64 / windows as f64
            } else {
                0.0
            },
            compression_ratio: if events > 0 {
                events as f64 / windows.max(1) as f64
            } else {
                0.0
            },
        }
    }
    
    /// Calculate microstructure features from windows
    pub fn calculate_features(
        &self,
        windows: &[AggregateWindow],
    ) -> MicrostructureFeatures {
        if windows.is_empty() {
            return MicrostructureFeatures::default();
        }
        
        // Order flow toxicity (VPIN-like)
        let total_volume: Decimal = windows.iter()
            .map(|w| w.volume.0)
            .fold(Decimal::ZERO, |acc, v| acc + v);
        
        let buy_volume: Decimal = windows.iter()
            .map(|w| w.buy_volume.0)
            .fold(Decimal::ZERO, |acc, v| acc + v);
        
        let sell_volume: Decimal = windows.iter()
            .map(|w| w.sell_volume.0)
            .fold(Decimal::ZERO, |acc, v| acc + v);
        
        let flow_toxicity = if total_volume > Decimal::ZERO {
            ((buy_volume - sell_volume).abs() / total_volume)
                .to_f64()
                .unwrap_or(0.0)
        } else {
            0.0
        };
        
        // Average order imbalance
        let avg_imbalance = windows.iter()
            .map(|w| w.order_imbalance)
            .sum::<f64>() / windows.len() as f64;
        
        // Volatility clustering (GARCH effect)
        let volatilities: Vec<f64> = windows.iter()
            .map(|w| w.realized_volatility)
            .collect();
        
        let vol_autocorrelation = if volatilities.len() > 1 {
            calculate_autocorrelation(&volatilities, 1)
        } else {
            0.0
        };
        
        // Price efficiency (variance ratio test)
        let price_efficiency = calculate_price_efficiency(windows);
        
        MicrostructureFeatures {
            flow_toxicity,
            avg_order_imbalance: avg_imbalance,
            volatility_clustering: vol_autocorrelation,
            price_efficiency,
            avg_spread: windows.iter()
                .map(|w| w.spread_avg)
                .fold(Decimal::ZERO, |acc, s| acc + s) / Decimal::from(windows.len()),
        }
    }
}

/// Microstructure features
#[derive(Debug, Clone, Default)]
pub struct MicrostructureFeatures {
    pub flow_toxicity: f64,
    pub avg_order_imbalance: f64,
    pub volatility_clustering: f64,
    pub price_efficiency: f64,
    pub avg_spread: Decimal,
}

/// Calculate autocorrelation
fn calculate_autocorrelation(series: &[f64], lag: usize) -> f64 {
    if series.len() <= lag {
        return 0.0;
    }
    
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in lag..series.len() {
        numerator += (series[i] - mean) * (series[i - lag] - mean);
    }
    
    for value in series {
        denominator += (value - mean).powi(2);
    }
    
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate price efficiency using variance ratio
fn calculate_price_efficiency(windows: &[AggregateWindow]) -> f64 {
    if windows.len() < 10 {
        return 1.0; // Assume efficient if not enough data
    }
    
    // Calculate returns
    let mut returns_1 = Vec::new();
    let mut returns_5 = Vec::new();
    
    for i in 1..windows.len() {
        let ret = (windows[i].close.0 / windows[i-1].close.0 - Decimal::ONE)
            .to_f64()
            .unwrap_or(0.0);
        returns_1.push(ret);
        
        if i >= 5 {
            let ret_5 = (windows[i].close.0 / windows[i-5].close.0 - Decimal::ONE)
                .to_f64()
                .unwrap_or(0.0);
            returns_5.push(ret_5);
        }
    }
    
    // Calculate variances
    let var_1 = variance(&returns_1);
    let var_5 = variance(&returns_5);
    
    // Variance ratio (should be ~5 for efficient markets)
    if var_1 > 0.0 {
        (var_5 / (5.0 * var_1)).min(2.0).max(0.0)
    } else {
        1.0
    }
}

/// Calculate variance
fn variance(series: &[f64]) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    series.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / series.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_bucketed_aggregator_creation() {
        let config = BucketConfig::default();
        let aggregator = BucketedAggregator::new(config).unwrap();
        
        let stats = aggregator.get_stats();
        assert_eq!(stats.events_processed, 0);
        assert_eq!(stats.total_windows, 0);
    }
    
    #[test]
    fn test_window_aggregation() {
        let config = BucketConfig {
            aggregation_levels: vec![1, 5, 10],
            ..Default::default()
        };
        
        let aggregator = BucketedAggregator::new(config).unwrap();
        let symbol = Symbol("BTC-USDT".to_string());
        
        // Add some trades
        let base_time = Utc::now();
        for i in 0..20 {
            aggregator.process_trade(
                symbol.clone(),
                Price(dec!(50000) + Decimal::from(i)),
                Quantity(dec!(1)),
                i % 2 == 0,
                base_time + Duration::milliseconds(i),
            ).unwrap();
        }
        
        // Check we have windows at different levels
        let windows_1ms = aggregator.get_windows(&symbol, 1, 10);
        let windows_5ms = aggregator.get_windows(&symbol, 5, 10);
        
        // Should have more 1ms windows than 5ms windows
        assert!(windows_1ms.len() >= windows_5ms.len());
    }
    
    #[test]
    fn test_microstructure_calculation() {
        let mut windows = Vec::new();
        
        for i in 0..10 {
            windows.push(AggregateWindow {
                window_id: i,
                symbol: Symbol("TEST".to_string()),
                start_time: Utc::now(),
                end_time: Utc::now() + Duration::milliseconds(5),
                duration_ms: 5,
                open: Price(dec!(100)),
                high: Price(dec!(101)),
                low: Price(dec!(99)),
                close: Price(dec!(100) + Decimal::from(i)),
                volume: Quantity(dec!(100)),
                trade_count: 10,
                vwap: Price(dec!(100)),
                twap: Price(dec!(100)),
                spread_avg: dec!(0.01),
                spread_max: dec!(0.02),
                buy_volume: Quantity(dec!(60)),
                sell_volume: Quantity(dec!(40)),
                order_imbalance: 0.2,
                tick_direction: 1,
                realized_volatility: 0.01 * (i as f64 + 1.0),
                high_low_range: dec!(2),
            });
        }
        
        let aggregator = BucketedAggregator::new(BucketConfig::default()).unwrap();
        let features = aggregator.calculate_features(&windows);
        
        assert!(features.flow_toxicity > 0.0);
        assert!(features.avg_order_imbalance.abs() < 1.0);
        assert_eq!(features.avg_spread, dec!(0.01));
    }
}