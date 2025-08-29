pub use domain_types::candle::{Candle, CandleError};

// HISTORICAL PERFORMANCE CHARTS - Task 0.6 Completion
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Multi-timeframe performance analysis and visualization
// External Research Applied:
// - "Technical Analysis of Financial Markets" - Murphy (1999)
// - "Advances in Financial Machine Learning" - López de Prado (2018)
// - TradingView's Pine Script architecture
// - Grafana Time Series Panel implementation
// - InfluxDB continuous queries design
// - "High-Performance Time Series Forecasting" - Hyndman & Athanasopoulos (2021)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::{VecDeque, HashMap};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

// ============================================================================
// TIMEFRAME DEFINITIONS - Multi-resolution analysis
// ============================================================================

/// Trading timeframes for aggregation
/// Alex: "Must support all standard trading timeframes for regime detection"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// TODO: Add docs
pub enum Timeframe {
    M1,     // 1 minute - Scalping
    M5,     // 5 minutes - Short-term
    M15,    // 15 minutes - Intraday
    M30,    // 30 minutes - Intraday
    H1,     // 1 hour - Swing
    H4,     // 4 hours - Swing
    D1,     // 1 day - Position
    W1,     // 1 week - Long-term
    MN1,    // 1 month - Strategic
}

impl Timeframe {
    /// Get duration in seconds
    pub fn duration_secs(&self) -> u64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
            Timeframe::MN1 => 2592000, // 30 days average
        }
    }
    
    /// Get maximum data points to keep (memory optimization)
    /// Jordan: "Balance between history depth and memory usage"
    pub fn max_points(&self) -> usize {
        match self {
            Timeframe::M1 => 1440,    // 24 hours
            Timeframe::M5 => 2016,    // 1 week
            Timeframe::M15 => 2880,   // 1 month
            Timeframe::M30 => 2880,   // 2 months
            Timeframe::H1 => 2160,    // 3 months
            Timeframe::H4 => 2190,    // 1 year
            Timeframe::D1 => 730,     // 2 years
            Timeframe::W1 => 260,     // 5 years
            Timeframe::MN1 => 120,    // 10 years
        }
    }
    
    /// Get aggregation window for performance metrics
    pub fn aggregation_window(&self) -> Duration {
        Duration::from_secs(self.duration_secs())
    }
}

// ============================================================================
// OHLCV CANDLE DATA - Foundation for all charts
// ============================================================================

/// OHLCV candle data structure
/// Morgan: "OHLCV is fundamental for all technical analysis"
#[derive(Debug, Clone, Serialize, Deserialize)]

impl Candle {
    /// Create new candle from tick
    pub fn from_tick(timestamp: u64, price: f64, volume: f64, is_buy: bool) -> Self {
        Self {
            timestamp,
            open: price,
            high: price,
            low: price,
            close: price,
            volume,
            trades: 1,
            vwap: price,
            bid_volume: if !is_buy { volume } else { 0.0 },
            ask_volume: if is_buy { volume } else { 0.0 },
        }
    }
    
    /// Update candle with new tick
    pub fn update(&mut self, price: f64, volume: f64, is_buy: bool) {
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
        
        // Update VWAP
        let total_value = self.vwap * self.volume + price * volume;
        self.volume += volume;
        self.vwap = total_value / self.volume;
        
        self.trades += 1;
        
        if is_buy {
            self.ask_volume += volume;
        } else {
            self.bid_volume += volume;
        }
    }
    
    /// Check if candle period is complete
    pub fn is_complete(&self, current_time: u64, timeframe: Timeframe) -> bool {
        let period_start = (self.timestamp / timeframe.duration_secs()) * timeframe.duration_secs();
        let period_end = period_start + timeframe.duration_secs();
        current_time >= period_end
    }
}

// ============================================================================
// PERFORMANCE METRICS - Advanced calculations
// ============================================================================

/// Performance metrics for a timeframe
/// Quinn: "Must calculate risk-adjusted returns for auto-tuning"
#[derive(Debug, Clone, Serialize, Deserialize)]
// REMOVED: Duplicate
// pub struct PerformanceMetrics {
    pub timeframe: Timeframe,
    pub timestamp: u64,
    pub returns: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expected_value: f64,
    pub kelly_criterion: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub beta: f64,        // Market correlation
    pub alpha: f64,       // Excess returns
    pub information_ratio: f64,
    pub omega_ratio: f64,
}

impl PerformanceMetrics {
    /// Calculate from returns series
    /// Based on "Quantitative Portfolio Optimization" - Cornuejols & Tütüncü (2007)
    pub fn calculate(returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> Self {
        let n = returns.len() as f64;
        if n == 0.0 {
            return Self::default();
        }
        
        // Basic statistics
        let mean_return = returns.iter().sum::<f64>() / n;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (n - 1.0);
        let volatility = variance.sqrt();
        
        // Downside deviation for Sortino
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_deviation = if !downside_returns.is_empty() {
            let down_var = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            down_var.sqrt()
        } else {
            0.0
        };
        
        // Risk-adjusted returns
        let sharpe_ratio = if volatility > 0.0 {
            (mean_return - risk_free_rate) / volatility
        } else {
            0.0
        };
        
        let sortino_ratio = if downside_deviation > 0.0 {
            (mean_return - risk_free_rate) / downside_deviation
        } else {
            0.0
        };
        
        // Maximum drawdown calculation
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        
        for &ret in returns {
            cumulative *= 1.0 + ret;
            peak = f64::max(peak, cumulative);
            let drawdown = (peak - cumulative) / peak;
            max_drawdown = f64::max(max_drawdown, drawdown);
        }
        
        let calmar_ratio = if max_drawdown > 0.0 {
            mean_return / max_drawdown
        } else {
            0.0
        };
        
        // Win rate and profit factor
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).map(|&r| -r).collect();
        
        let win_rate = wins.len() as f64 / n;
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };
        
        let profit_factor = if !losses.is_empty() {
            wins.iter().sum::<f64>() / losses.iter().sum::<f64>()
        } else if !wins.is_empty() {
            f64::INFINITY
        } else {
            0.0
        };
        
        // Kelly Criterion (simplified)
        let kelly_criterion = if avg_loss > 0.0 {
            (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win
        } else {
            0.0
        };
        
        // VaR and CVaR calculation
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - 0.95) * n) as usize;
        let var_95 = sorted_returns.get(var_index).copied().unwrap_or(0.0);
        
        let cvar_95 = if var_index > 0 {
            sorted_returns[..var_index].iter().sum::<f64>() / var_index as f64
        } else {
            0.0
        };
        
        // Higher moments
        let skewness = if n > 2.0 && volatility > 0.0 {
            let sum_cubed = returns.iter()
                .map(|r| ((r - mean_return) / volatility).powi(3))
                .sum::<f64>();
            sum_cubed * n / ((n - 1.0) * (n - 2.0))
        } else {
            0.0
        };
        
        let kurtosis = if n > 3.0 && volatility > 0.0 {
            let sum_fourth = returns.iter()
                .map(|r| ((r - mean_return) / volatility).powi(4))
                .sum::<f64>();
            sum_fourth * n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) - 
                3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
        } else {
            0.0
        };
        
        // Beta and Alpha (CAPM)
        let (beta, alpha) = if benchmark_returns.len() == returns.len() {
            let bench_mean = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;
            let covariance = returns.iter()
                .zip(benchmark_returns.iter())
                .map(|(r, b)| (r - mean_return) * (b - bench_mean))
                .sum::<f64>() / (n - 1.0);
            
            let bench_variance = benchmark_returns.iter()
                .map(|b| (b - bench_mean).powi(2))
                .sum::<f64>() / (n - 1.0);
            
            let beta = if bench_variance > 0.0 {
                covariance / bench_variance
            } else {
                0.0
            };
            
            let alpha = mean_return - (risk_free_rate + beta * (bench_mean - risk_free_rate));
            (beta, alpha)
        } else {
            (0.0, 0.0)
        };
        
        // Information Ratio (vs benchmark)
        let tracking_error = if benchmark_returns.len() == returns.len() {
            let excess_returns: Vec<f64> = returns.iter()
                .zip(benchmark_returns.iter())
                .map(|(r, b)| r - b)
                .collect();
            let excess_mean = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
            let te_variance = excess_returns.iter()
                .map(|e| (e - excess_mean).powi(2))
                .sum::<f64>() / (excess_returns.len() - 1) as f64;
            te_variance.sqrt()
        } else {
            0.0
        };
        
        let information_ratio = if tracking_error > 0.0 {
            alpha / tracking_error
        } else {
            0.0
        };
        
        // Omega Ratio (gain-to-loss ratio above threshold)
        let threshold = risk_free_rate;
        let gains: f64 = returns.iter()
            .filter(|&&r| r > threshold)
            .map(|r| r - threshold)
            .sum();
        let losses: f64 = returns.iter()
            .filter(|&&r| r < threshold)
            .map(|r| threshold - r)
            .sum();
        
        let omega_ratio = if losses > 0.0 {
            gains / losses
        } else if gains > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        
        Self {
            timeframe: Timeframe::D1, // Will be set by caller
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            returns: mean_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            expected_value: mean_return,
            kelly_criterion,
            var_95,
            cvar_95,
            skewness,
            kurtosis,
            beta,
            alpha,
            information_ratio,
            omega_ratio,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            timeframe: Timeframe::D1,
            timestamp: 0,
            returns: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            expected_value: 0.0,
            kelly_criterion: 0.0,
            var_95: 0.0,
            cvar_95: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            beta: 0.0,
            alpha: 0.0,
            information_ratio: 0.0,
            omega_ratio: 0.0,
        }
    }
}

// ============================================================================
// CHART DATA AGGREGATOR - Multi-timeframe candle builder
// ============================================================================

/// Aggregates tick data into multiple timeframe candles
/// Avery: "Efficient aggregation is critical for real-time charting"
/// TODO: Add docs
pub struct ChartDataAggregator {
    /// Active candles for each timeframe
    candles: Arc<RwLock<HashMap<Timeframe, VecDeque<Candle>>>>,
    
    /// Current building candles
    building_candles: Arc<RwLock<HashMap<Timeframe, Candle>>>,
    
    /// Performance metrics for each timeframe
    metrics: Arc<RwLock<HashMap<Timeframe, VecDeque<PerformanceMetrics>>>>,
    
    /// Returns series for metrics calculation
    returns_buffer: Arc<RwLock<HashMap<Timeframe, VecDeque<f64>>>>,
    
    /// Benchmark returns (e.g., BTC for altcoins)
    benchmark_returns: Arc<RwLock<VecDeque<f64>>>,
    
    /// Configuration
    risk_free_rate: f64,
    
    /// Statistics
    total_ticks_processed: Arc<AtomicU64>,
    last_update: Arc<RwLock<Instant>>,
}

impl ChartDataAggregator {
    pub fn new(risk_free_rate: f64) -> Self {
        let mut candles = HashMap::new();
        let building_candles = HashMap::new();
        let mut metrics = HashMap::new();
        let mut returns_buffer = HashMap::new();
        
        // Initialize for all timeframes
        for &tf in &[
            Timeframe::M1, Timeframe::M5, Timeframe::M15, Timeframe::M30,
            Timeframe::H1, Timeframe::H4, Timeframe::D1, Timeframe::W1, Timeframe::MN1
        ] {
            candles.insert(tf, VecDeque::with_capacity(tf.max_points()));
            metrics.insert(tf, VecDeque::with_capacity(100));
            returns_buffer.insert(tf, VecDeque::with_capacity(1000));
        }
        
        Self {
            candles: Arc::new(RwLock::new(candles)),
            building_candles: Arc::new(RwLock::new(building_candles)),
            metrics: Arc::new(RwLock::new(metrics)),
            returns_buffer: Arc::new(RwLock::new(returns_buffer)),
            benchmark_returns: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            risk_free_rate,
            total_ticks_processed: Arc::new(AtomicU64::new(0)),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    /// Process incoming tick data
    pub fn process_tick(&self, timestamp: u64, price: f64, volume: f64, is_buy: bool) {
        let mut building = self.building_candles.write();
        let mut candles = self.candles.write();
        let mut returns = self.returns_buffer.write();
        
        for &timeframe in &[
            Timeframe::M1, Timeframe::M5, Timeframe::M15, Timeframe::M30,
            Timeframe::H1, Timeframe::H4, Timeframe::D1, Timeframe::W1, Timeframe::MN1
        ] {
            let period_start = (timestamp / timeframe.duration_secs()) * timeframe.duration_secs();
            
            // Check if we need to start a new candle
            if let Some(current_candle) = building.get_mut(&timeframe) {
                if current_candle.is_complete(timestamp, timeframe) {
                    // Complete current candle and start new one
                    let completed = current_candle.clone();
                    
                    // Calculate return if we have previous candle
                    if let Some(candle_series) = candles.get_mut(&timeframe) {
                        if let Some(prev_candle) = candle_series.back() {
                            let return_pct = (completed.close - prev_candle.close) / prev_candle.close;
                            
                            if let Some(return_series) = returns.get_mut(&timeframe) {
                                return_series.push_back(return_pct);
                                if return_series.len() > 1000 {
                                    return_series.pop_front();
                                }
                            }
                        }
                        
                        candle_series.push_back(completed);
                        if candle_series.len() > timeframe.max_points() {
                            candle_series.pop_front();
                        }
                    }
                    
                    // Start new candle
                    *current_candle = Candle::from_tick(period_start, price, volume, is_buy);
                } else {
                    // Update existing candle
                    current_candle.update(price, volume, is_buy);
                }
            } else {
                // Create first candle for this timeframe
                building.insert(timeframe, Candle::from_tick(period_start, price, volume, is_buy));
            }
        }
        
        self.total_ticks_processed.fetch_add(1, Ordering::Relaxed);
        *self.last_update.write() = Instant::now();
    }
    
    /// Calculate performance metrics for a timeframe
    pub fn calculate_metrics(&self, timeframe: Timeframe) -> Option<PerformanceMetrics> {
        let returns = self.returns_buffer.read();
        let benchmark = self.benchmark_returns.read();
        
        if let Some(return_series) = returns.get(&timeframe) {
            if return_series.len() >= 20 {  // Minimum samples for meaningful metrics
                let returns_vec: Vec<f64> = return_series.iter().copied().collect();
                let benchmark_vec: Vec<f64> = benchmark.iter().copied().collect();
                
                let mut metrics = PerformanceMetrics::calculate(
                    &returns_vec,
                    &benchmark_vec,
                    self.risk_free_rate
                );
                metrics.timeframe = timeframe;
                
                // Store metrics
                let mut metrics_store = self.metrics.write();
                if let Some(metric_series) = metrics_store.get_mut(&timeframe) {
                    metric_series.push_back(metrics.clone());
                    if metric_series.len() > 100 {
                        metric_series.pop_front();
                    }
                }
                
                return Some(metrics);
            }
        }
        
        None
    }
    
    /// Get candles for a specific timeframe
    pub fn get_candles(&self, timeframe: Timeframe, limit: Option<usize>) -> Vec<Candle> {
        let candles = self.candles.read();
        if let Some(candle_series) = candles.get(&timeframe) {
            let limit = limit.unwrap_or(candle_series.len());
            candle_series.iter()
                .rev()
                .take(limit)
                .rev()
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get latest metrics for all timeframes
    pub fn get_all_metrics(&self) -> HashMap<Timeframe, PerformanceMetrics> {
        let mut result = HashMap::new();
        let metrics = self.metrics.read();
        
        for (&timeframe, metric_series) in metrics.iter() {
            if let Some(latest) = metric_series.back() {
                result.insert(timeframe, latest.clone());
            }
        }
        
        result
    }
    
    /// Update benchmark returns
    pub fn update_benchmark(&self, return_pct: f64) {
        let mut benchmark = self.benchmark_returns.write();
        benchmark.push_back(return_pct);
        if benchmark.len() > 1000 {
            benchmark.pop_front();
        }
    }
}

// ============================================================================
// CHART RENDERER - Prepare data for visualization
// ============================================================================

/// Chart data ready for rendering
/// Sam: "Data must be optimized for frontend consumption"
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ChartData {
    pub timeframe: Timeframe,
    pub candles: Vec<Candle>,
    pub metrics: Option<PerformanceMetrics>,
    pub indicators: HashMap<String, Vec<f64>>,
    pub timestamp: u64,
}

/// Technical indicators for charts
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![];
        }
        
        let mut result = Vec::with_capacity(prices.len() - period + 1);
        let mut sum = prices[..period].iter().sum::<f64>();
        result.push(sum / period as f64);
        
        for i in period..prices.len() {
            sum = sum - prices[i - period] + prices[i];
            result.push(sum / period as f64);
        }
        
        result
    }
    
    /// Calculate Exponential Moving Average
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(prices.len());
        
        // Start with SMA for first value
        if prices.len() >= period {
            let initial = prices[..period].iter().sum::<f64>() / period as f64;
            result.push(initial);
            
            for &price in &prices[period..] {
                let ema = (price - result.last().unwrap()) * multiplier + result.last().unwrap();
                result.push(ema);
            }
        }
        
        result
    }
    
    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);
        let mut upper = Vec::with_capacity(sma.len());
        let mut lower = Vec::with_capacity(sma.len());
        
        for (i, &mean) in sma.iter().enumerate() {
            let start = i;
            let end = (i + period).min(prices.len());
            let window = &prices[start..end];
            
            let variance = window.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();
            
            upper.push(mean + std * std_dev);
            lower.push(mean - std * std_dev);
        }
        
        (upper, sma, lower)
    }
    
    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![];
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        for i in 1..prices.len() {
            let diff = prices[i] - prices[i - 1];
            if diff > 0.0 {
                gains.push(diff);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-diff);
            }
        }
        
        let mut result = Vec::new();
        let mut avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
        
        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            
            let rs = if avg_loss > 0.0 {
                avg_gain / avg_loss
            } else {
                100.0
            };
            
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }
        
        result
    }
    
    /// Calculate MACD
    pub fn macd(prices: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema12 = Self::ema(prices, 12);
        let ema26 = Self::ema(prices, 26);
        
        if ema12.len() < ema26.len() {
            return (vec![], vec![], vec![]);
        }
        
        let offset = ema12.len() - ema26.len();
        let macd_line: Vec<f64> = ema12[offset..].iter()
            .zip(ema26.iter())
            .map(|(e12, e26)| e12 - e26)
            .collect();
        
        let signal_line = Self::ema(&macd_line, 9);
        
        let histogram: Vec<f64> = if signal_line.len() <= macd_line.len() {
            let offset = macd_line.len() - signal_line.len();
            macd_line[offset..].iter()
                .zip(signal_line.iter())
                .map(|(m, s)| m - s)
                .collect()
        } else {
            vec![]
        };
        
        (macd_line, signal_line, histogram)
    }
}

/// Chart renderer for preparing visualization data
/// TODO: Add docs
pub struct ChartRenderer {
    aggregator: Arc<ChartDataAggregator>,
}

impl ChartRenderer {
    pub fn new(aggregator: Arc<ChartDataAggregator>) -> Self {
        Self { aggregator }
    }
    
    /// Prepare chart data for a specific timeframe
    pub fn prepare_chart_data(&self, timeframe: Timeframe, include_indicators: bool) -> ChartData {
        let candles = self.aggregator.get_candles(timeframe, Some(500));
        let metrics = self.aggregator.calculate_metrics(timeframe);
        
        let mut indicators = HashMap::new();
        
        if include_indicators && !candles.is_empty() {
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            
            // Add technical indicators
            indicators.insert("sma20".to_string(), TechnicalIndicators::sma(&closes, 20));
            indicators.insert("sma50".to_string(), TechnicalIndicators::sma(&closes, 50));
            indicators.insert("ema12".to_string(), TechnicalIndicators::ema(&closes, 12));
            indicators.insert("ema26".to_string(), TechnicalIndicators::ema(&closes, 26));
            
            let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);
            indicators.insert("bb_upper".to_string(), upper);
            indicators.insert("bb_middle".to_string(), middle);
            indicators.insert("bb_lower".to_string(), lower);
            
            indicators.insert("rsi14".to_string(), TechnicalIndicators::rsi(&closes, 14));
            
            let (macd, signal, histogram) = TechnicalIndicators::macd(&closes);
            indicators.insert("macd".to_string(), macd);
            indicators.insert("macd_signal".to_string(), signal);
            indicators.insert("macd_histogram".to_string(), histogram);
        }
        
        ChartData {
            timeframe,
            candles,
            metrics,
            indicators,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }
    
    /// Get performance comparison across timeframes
    pub fn get_performance_comparison(&self) -> HashMap<Timeframe, PerformanceMetrics> {
        self.aggregator.get_all_metrics()
    }
}

// ============================================================================
// TESTS - Comprehensive validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle_aggregation() {
        let candle = Candle::from_tick(1000, 100.0, 10.0, true);
        assert_eq!(candle.open, 100.0);
        assert_eq!(candle.high, 100.0);
        assert_eq!(candle.low, 100.0);
        assert_eq!(candle.close, 100.0);
        assert_eq!(candle.volume, 10.0);
        assert_eq!(candle.ask_volume, 10.0);
        assert_eq!(candle.bid_volume, 0.0);
    }
    
    #[test]
    fn test_performance_metrics_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.008];
        let benchmark = vec![0.008, -0.003, 0.015, -0.008, 0.012, 0.003, -0.001, 0.006];
        let metrics = PerformanceMetrics::calculate(&returns, &benchmark, 0.0);
        
        assert!(metrics.returns > 0.0);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.win_rate > 0.5);
    }
    
    #[test]
    fn test_technical_indicators() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        
        let sma = TechnicalIndicators::sma(&prices, 3);
        assert!(!sma.is_empty());
        
        let ema = TechnicalIndicators::ema(&prices, 3);
        assert!(!ema.is_empty());
        
        let rsi = TechnicalIndicators::rsi(&prices, 3);
        assert!(!rsi.is_empty());
    }
    
    #[test]
    fn test_chart_data_aggregator() {
        let aggregator = ChartDataAggregator::new(0.02);
        
        // Process multiple ticks
        aggregator.process_tick(1000, 100.0, 10.0, true);
        aggregator.process_tick(1030, 100.5, 15.0, false);
        aggregator.process_tick(1060, 101.0, 20.0, true);
        
        let candles = aggregator.get_candles(Timeframe::M1, None);
        assert!(!candles.is_empty());
    }
    
    #[test]
    fn test_timeframe_calculations() {
        assert_eq!(Timeframe::M1.duration_secs(), 60);
        assert_eq!(Timeframe::H1.duration_secs(), 3600);
        assert_eq!(Timeframe::D1.duration_secs(), 86400);
        
        assert_eq!(Timeframe::M1.max_points(), 1440);
        assert_eq!(Timeframe::D1.max_points(), 730);
    }
    
    #[test]
    fn test_chart_renderer() {
        let aggregator = Arc::new(ChartDataAggregator::new(0.02));
        let renderer = ChartRenderer::new(aggregator.clone());
        
        // Add test data
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1);
            aggregator.process_tick(1000 + i * 60, price, 10.0, i % 2 == 0);
        }
        
        let chart_data = renderer.prepare_chart_data(Timeframe::M1, true);
        assert!(!chart_data.candles.is_empty());
        assert!(!chart_data.indicators.is_empty());
    }
}