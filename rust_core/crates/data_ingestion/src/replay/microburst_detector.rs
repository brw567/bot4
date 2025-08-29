use domain_types::MarketImpact;
// Microburst Detection and Simulation
// DEEP DIVE: Detect and simulate rapid market movements
//
// References:
// - "Flash Crash: A Trading Savant, a Global Manhunt" - Vaughan (2020)
// - "The Flash Crash: A Review" - SEC/CFTC Report (2010)
// - "Microstructure Noise, Realized Variance, and Optimal Sampling" - Zhang et al (2005)
// - High-frequency data analysis from Nanex and ThemisST

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{debug, warn, info, instrument};
use statrs::statistics::Statistics;
use statrs::distribution::{Normal, ContinuousCDF};

use types::{Price, Quantity, Symbol};
// TODO: use infrastructure::metrics::{MetricsCollector, register_counter, register_histogram};

/// Types of microburst events
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum MicroburstType {
    /// Sudden volume spike
    VolumeSpike {
        normal_volume: Quantity,
        spike_volume: Quantity,
        duration_ms: u64,
        sigma_deviation: f64,
    },
    /// Rapid price movement
    PriceJump {
        start_price: Price,
        end_price: Price,
        duration_ms: u64,
        tick_velocity: f64,  // Ticks per millisecond
    },
    /// Network/exchange latency spike
    LatencySpike {
        normal_latency_us: u64,
        spike_latency_us: u64,
        affected_symbols: Vec<Symbol>,
    },
    /// Order book evaporation
    LiquidityEvaporation {
        pre_depth: Quantity,
        post_depth: Quantity,
        levels_affected: u32,
    },
    /// Quote stuffing/spoofing
    QuoteStuffing {
        messages_per_second: u64,
        normal_rate: u64,
        pattern: StuffingPattern,
    },
    /// Cascading liquidations
    LiquidationCascade {
        initial_trigger_price: Price,
        final_price: Price,
        liquidated_volume: Quantity,
        cascade_levels: u32,
    },
}

/// Pattern of quote stuffing
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum StuffingPattern {
    Sawtooth,      // Rapid add/cancel at same price
    Layering,      // Multiple orders away from market
    Momentum,      // Create false momentum signals
    Exploratory,   // Test market response
}

/// Volume spike detection
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct VolumeSpike {
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub baseline_volume: Quantity,
    pub spike_volume: Quantity,
    pub z_score: f64,
    pub duration_ms: u64,
    pub orders_in_spike: u32,
}

/// Price jump event
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PriceJump {
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub pre_price: Price,
    pub post_price: Price,
    pub price_change_pct: f64,
    pub ticks_moved: i32,
    pub duration_ms: u64,
    pub volume_during_jump: Quantity,
}

/// Latency spike event
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct LatencySpike {
    pub timestamp: DateTime<Utc>,
    pub normal_latency_us: u64,
    pub spike_latency_us: u64,
    pub duration_ms: u64,
    pub messages_delayed: u32,
    pub potential_arbitrage: bool,
}

/// Microburst event container
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MicroburstEvent {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub event_type: MicroburstType,
    pub severity: EventSeverity,
    pub market_impact: MarketImpact,
    pub detected_by: Vec<DetectionMethod>,
}

/// Event severity classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum EventSeverity {
    Low,      // Normal market noise
    Medium,   // Unusual but not dangerous
    High,     // Significant market event
    Critical, // System-threatening event
}

/// Market impact assessment
#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::MarketImpact
// pub struct MarketImpact {
    pub spread_widening_bps: f64,
    pub depth_reduction_pct: f64,
    pub volatility_increase: f64,
    pub correlation_breakdown: bool,
    pub estimated_slippage_bps: f64,
}

/// Detection method used
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum DetectionMethod {
    ZScore,
    MAD,  // Median Absolute Deviation
    EWMA, // Exponentially Weighted Moving Average
    JarqueBera,
    KolmogorovSmirnov,
    GrangerCausality,
    MachineLearning,
}

/// Configuration for microburst detection
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct DetectorConfig {
    /// Z-score threshold for volume spikes
    pub volume_z_threshold: f64,
    
    /// Price movement threshold (basis points)
    pub price_jump_threshold_bps: f64,
    
    /// Latency spike threshold (microseconds)
    pub latency_spike_threshold_us: u64,
    
    /// Lookback window for baseline calculation
    pub lookback_window_ms: u64,
    
    /// Minimum events for statistical significance
    pub min_sample_size: usize,
    
    /// Enable ML-based detection
    pub enable_ml_detection: bool,
    
    /// Quote stuffing detection threshold
    pub quote_rate_multiplier: f64,
    
    /// Liquidation cascade detection
    pub cascade_detection_enabled: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            volume_z_threshold: 3.0,
            price_jump_threshold_bps: 50.0,
            latency_spike_threshold_us: 10_000,
            lookback_window_ms: 60_000,
            min_sample_size: 100,
            enable_ml_detection: false,
            quote_rate_multiplier: 10.0,
            cascade_detection_enabled: true,
        }
    }
}

/// Statistics tracker for baseline calculation
struct StatisticsTracker {
    volume_window: VecDeque<f64>,
    price_window: VecDeque<f64>,
    latency_window: VecDeque<f64>,
    message_rate_window: VecDeque<f64>,
    window_size: usize,
}

impl StatisticsTracker {
    fn new(window_size: usize) -> Self {
        Self {
            volume_window: VecDeque::with_capacity(window_size),
            price_window: VecDeque::with_capacity(window_size),
            latency_window: VecDeque::with_capacity(window_size),
            message_rate_window: VecDeque::with_capacity(window_size),
            window_size,
        }
    }
    
    fn add_volume(&mut self, volume: f64) {
        if self.volume_window.len() >= self.window_size {
            self.volume_window.pop_front();
        }
        self.volume_window.push_back(volume);
    }
    
    fn add_price(&mut self, price: f64) {
        if self.price_window.len() >= self.window_size {
            self.price_window.pop_front();
        }
        self.price_window.push_back(price);
    }
    
    fn add_latency(&mut self, latency: f64) {
        if self.latency_window.len() >= self.window_size {
            self.latency_window.pop_front();
        }
        self.latency_window.push_back(latency);
    }
    
    fn add_message_rate(&mut self, rate: f64) {
        if self.message_rate_window.len() >= self.window_size {
            self.message_rate_window.pop_front();
        }
        self.message_rate_window.push_back(rate);
    }
    
    fn calculate_z_score(&self, window: &VecDeque<f64>, value: f64) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }
        
        let data: Vec<f64> = window.iter().cloned().collect();
        let mean = data.clone().mean();
        let std_dev = data.std_dev();
        
        if std_dev > 0.0 {
            (value - mean) / std_dev
        } else {
            0.0
        }
    }
    
    fn calculate_mad(&self, window: &VecDeque<f64>, value: f64) -> f64 {
        if window.is_empty() {
            return 0.0;
        }
        
        let mut sorted: Vec<f64> = window.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        let deviations: Vec<f64> = sorted.iter()
            .map(|x| (x - median).abs())
            .collect();
        
        let mut dev_sorted = deviations;
        dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mad = if dev_sorted.len() % 2 == 0 {
            (dev_sorted[dev_sorted.len() / 2 - 1] + dev_sorted[dev_sorted.len() / 2]) / 2.0
        } else {
            dev_sorted[dev_sorted.len() / 2]
        };
        
        if mad > 0.0 {
            (value - median) / (1.4826 * mad)  // 1.4826 is consistency factor for normal distribution
        } else {
            0.0
        }
    }
}

/// Main microburst detector implementation
/// TODO: Add docs
pub struct MicroburstDetector {
    config: Arc<DetectorConfig>,
    
    // Statistics tracking per symbol
    stats: Arc<RwLock<ahash::AHashMap<Symbol, StatisticsTracker>>>,
    
    // Event history
    events: Arc<RwLock<VecDeque<MicroburstEvent>>>,
    
    // Event counter for IDs
    event_counter: Arc<RwLock<u64>>,
    
    // Metrics
    volume_spikes_detected: Arc<dyn MetricsCollector>,
    price_jumps_detected: Arc<dyn MetricsCollector>,
    latency_spikes_detected: Arc<dyn MetricsCollector>,
    quote_stuffing_detected: Arc<dyn MetricsCollector>,
    cascade_events_detected: Arc<dyn MetricsCollector>,
}

impl MicroburstDetector {
    pub fn new(config: DetectorConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config),
            stats: Arc::new(RwLock::new(ahash::AHashMap::new())),
            events: Arc::new(RwLock::new(VecDeque::with_capacity(10_000))),
            event_counter: Arc::new(RwLock::new(0)),
            volume_spikes_detected: register_counter("microburst_volume_spikes"),
            price_jumps_detected: register_counter("microburst_price_jumps"),
            latency_spikes_detected: register_counter("microburst_latency_spikes"),
            quote_stuffing_detected: register_counter("microburst_quote_stuffing"),
            cascade_events_detected: register_counter("microburst_cascades"),
        })
    }
    
    /// Process market data for microburst detection
    #[instrument(skip(self))]
    pub fn process_market_data(
        &self,
        symbol: Symbol,
        price: Price,
        volume: Quantity,
        latency_us: u64,
        message_rate: u64,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        // Get or create statistics tracker
        let mut stats_map = self.stats.write();
        let tracker = stats_map.entry(symbol.clone())
            .or_insert_with(|| StatisticsTracker::new(1000));
        
        // Convert to f64 for statistics
        let price_f64 = price.0.to_f64().unwrap_or(0.0);
        let volume_f64 = volume.0.to_f64().unwrap_or(0.0);
        let latency_f64 = latency_us as f64;
        let message_rate_f64 = message_rate as f64;
        
        // Check for volume spike
        if let Some(event) = self.detect_volume_spike(
            &symbol,
            volume_f64,
            tracker,
            timestamp,
        )? {
            return Ok(Some(event));
        }
        
        // Check for price jump
        if let Some(event) = self.detect_price_jump(
            &symbol,
            price_f64,
            volume_f64,
            tracker,
            timestamp,
        )? {
            return Ok(Some(event));
        }
        
        // Check for latency spike
        if let Some(event) = self.detect_latency_spike(
            &symbol,
            latency_f64,
            tracker,
            timestamp,
        )? {
            return Ok(Some(event));
        }
        
        // Check for quote stuffing
        if let Some(event) = self.detect_quote_stuffing(
            &symbol,
            message_rate_f64,
            tracker,
            timestamp,
        )? {
            return Ok(Some(event));
        }
        
        // Update statistics
        tracker.add_volume(volume_f64);
        tracker.add_price(price_f64);
        tracker.add_latency(latency_f64);
        tracker.add_message_rate(message_rate_f64);
        
        Ok(None)
    }
    
    /// Detect volume spike
    fn detect_volume_spike(
        &self,
        symbol: &Symbol,
        volume: f64,
        tracker: &StatisticsTracker,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        if tracker.volume_window.len() < self.config.min_sample_size {
            return Ok(None);
        }
        
        let z_score = tracker.calculate_z_score(&tracker.volume_window, volume);
        let mad_score = tracker.calculate_mad(&tracker.volume_window, volume);
        
        // Use both Z-score and MAD for robustness
        if z_score > self.config.volume_z_threshold || mad_score > self.config.volume_z_threshold * 1.5 {
            let baseline: Vec<f64> = tracker.volume_window.iter().cloned().collect();
            let baseline_mean = baseline.mean();
            
            let event = self.create_microburst_event(
                symbol.clone(),
                timestamp,
                MicroburstType::VolumeSpike {
                    normal_volume: Quantity(Decimal::from_f64_retain(baseline_mean).unwrap_or(Decimal::ZERO)),
                    spike_volume: Quantity(Decimal::from_f64_retain(volume).unwrap_or(Decimal::ZERO)),
                    duration_ms: 100,  // Would be calculated from actual data
                    sigma_deviation: z_score,
                },
                self.calculate_severity(z_score),
                vec![DetectionMethod::ZScore, DetectionMethod::MAD],
            );
            
            self.volume_spikes_detected.increment(1);
            self.store_event(event.clone());
            
            return Ok(Some(event));
        }
        
        Ok(None)
    }
    
    /// Detect price jump
    fn detect_price_jump(
        &self,
        symbol: &Symbol,
        price: f64,
        volume: f64,
        tracker: &StatisticsTracker,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        if tracker.price_window.is_empty() {
            return Ok(None);
        }
        
        let last_price = *tracker.price_window.back().unwrap();
        let price_change_pct = ((price - last_price) / last_price * 10000.0).abs(); // basis points
        
        if price_change_pct > self.config.price_jump_threshold_bps {
            let event = self.create_microburst_event(
                symbol.clone(),
                timestamp,
                MicroburstType::PriceJump {
                    start_price: Price(Decimal::from_f64_retain(last_price).unwrap_or(Decimal::ZERO)),
                    end_price: Price(Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO)),
                    duration_ms: 50,
                    tick_velocity: price_change_pct / 50.0,
                },
                self.calculate_severity_from_price_move(price_change_pct),
                vec![DetectionMethod::ZScore],
            );
            
            self.price_jumps_detected.increment(1);
            self.store_event(event.clone());
            
            return Ok(Some(event));
        }
        
        Ok(None)
    }
    
    /// Detect latency spike
    fn detect_latency_spike(
        &self,
        symbol: &Symbol,
        latency: f64,
        tracker: &StatisticsTracker,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        if tracker.latency_window.len() < self.config.min_sample_size {
            return Ok(None);
        }
        
        let baseline: Vec<f64> = tracker.latency_window.iter().cloned().collect();
        let baseline_mean = baseline.mean();
        
        if latency > self.config.latency_spike_threshold_us as f64 && 
           latency > baseline_mean * 10.0 {
            let event = self.create_microburst_event(
                symbol.clone(),
                timestamp,
                MicroburstType::LatencySpike {
                    normal_latency_us: baseline_mean as u64,
                    spike_latency_us: latency as u64,
                    affected_symbols: vec![symbol.clone()],
                },
                EventSeverity::High,
                vec![DetectionMethod::ZScore],
            );
            
            self.latency_spikes_detected.increment(1);
            self.store_event(event.clone());
            
            return Ok(Some(event));
        }
        
        Ok(None)
    }
    
    /// Detect quote stuffing
    fn detect_quote_stuffing(
        &self,
        symbol: &Symbol,
        message_rate: f64,
        tracker: &StatisticsTracker,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        if tracker.message_rate_window.len() < self.config.min_sample_size {
            return Ok(None);
        }
        
        let baseline: Vec<f64> = tracker.message_rate_window.iter().cloned().collect();
        let baseline_mean = baseline.mean();
        
        if message_rate > baseline_mean * self.config.quote_rate_multiplier {
            // Detect pattern
            let pattern = self.detect_stuffing_pattern(&tracker.message_rate_window);
            
            let event = self.create_microburst_event(
                symbol.clone(),
                timestamp,
                MicroburstType::QuoteStuffing {
                    messages_per_second: message_rate as u64,
                    normal_rate: baseline_mean as u64,
                    pattern,
                },
                EventSeverity::Medium,
                vec![DetectionMethod::EWMA],
            );
            
            self.quote_stuffing_detected.increment(1);
            self.store_event(event.clone());
            
            return Ok(Some(event));
        }
        
        Ok(None)
    }
    
    /// Detect liquidation cascade
    pub fn detect_liquidation_cascade(
        &self,
        symbol: Symbol,
        price_levels: Vec<Price>,
        volumes: Vec<Quantity>,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<MicroburstEvent>> {
        if !self.config.cascade_detection_enabled || price_levels.len() < 3 {
            return Ok(None);
        }
        
        // Check for accelerating price movement with increasing volume
        let mut price_velocities = Vec::new();
        let mut volume_acceleration = Vec::new();
        
        for i in 1..price_levels.len() {
            let price_change = (price_levels[i].0 - price_levels[i-1].0).abs();
            price_velocities.push(price_change.to_f64().unwrap_or(0.0));
            
            if i > 1 {
                let vol_change = volumes[i].0 - volumes[i-1].0;
                volume_acceleration.push(vol_change.to_f64().unwrap_or(0.0));
            }
        }
        
        // Detect cascade: accelerating price with accelerating volume
        let velocity_increasing = price_velocities.windows(2)
            .all(|w| w[1] > w[0] * 1.2);
        
        let volume_increasing = volume_acceleration.iter()
            .all(|&v| v > 0.0);
        
        if velocity_increasing && volume_increasing {
            let total_volume = volumes.iter()
                .fold(Decimal::ZERO, |acc, v| acc + v.0);
            
            let event = self.create_microburst_event(
                symbol,
                timestamp,
                MicroburstType::LiquidationCascade {
                    initial_trigger_price: price_levels[0].clone(),
                    final_price: price_levels.last().unwrap().clone(),
                    liquidated_volume: Quantity(total_volume),
                    cascade_levels: price_levels.len() as u32,
                },
                EventSeverity::Critical,
                vec![DetectionMethod::GrangerCausality],
            );
            
            self.cascade_events_detected.increment(1);
            self.store_event(event.clone());
            
            return Ok(Some(event));
        }
        
        Ok(None)
    }
    
    /// Detect stuffing pattern type
    fn detect_stuffing_pattern(&self, window: &VecDeque<f64>) -> StuffingPattern {
        // Simple pattern detection - could be enhanced with FFT or wavelets
        let data: Vec<f64> = window.iter().cloned().collect();
        
        // Check for sawtooth (rapid oscillation)
        let mut direction_changes = 0;
        for i in 1..data.len() {
            if i > 0 && ((data[i] > data[i-1]) != (data[i-1] > data[i-2])) {
                direction_changes += 1;
            }
        }
        
        if direction_changes > data.len() / 2 {
            return StuffingPattern::Sawtooth;
        }
        
        // Check for momentum (steady increase)
        let increasing = data.windows(2).filter(|w| w[1] > w[0]).count();
        if increasing > data.len() * 3 / 4 {
            return StuffingPattern::Momentum;
        }
        
        // Default to layering
        StuffingPattern::Layering
    }
    
    /// Calculate event severity from Z-score
    fn calculate_severity(&self, z_score: f64) -> EventSeverity {
        if z_score > 6.0 {
            EventSeverity::Critical
        } else if z_score > 4.0 {
            EventSeverity::High
        } else if z_score > 3.0 {
            EventSeverity::Medium
        } else {
            EventSeverity::Low
        }
    }
    
    /// Calculate severity from price movement
    fn calculate_severity_from_price_move(&self, bps: f64) -> EventSeverity {
        if bps > 500.0 {
            EventSeverity::Critical
        } else if bps > 200.0 {
            EventSeverity::High
        } else if bps > 100.0 {
            EventSeverity::Medium
        } else {
            EventSeverity::Low
        }
    }
    
    /// Create microburst event
    fn create_microburst_event(
        &self,
        symbol: Symbol,
        timestamp: DateTime<Utc>,
        event_type: MicroburstType,
        severity: EventSeverity,
        detected_by: Vec<DetectionMethod>,
    ) -> MicroburstEvent {
        let mut counter = self.event_counter.write();
        *counter += 1;
        let event_id = *counter;
        
        // Calculate market impact
        let market_impact = self.calculate_market_impact(&event_type, severity);
        
        MicroburstEvent {
            id: event_id,
            timestamp,
            symbol,
            event_type,
            severity,
            market_impact,
            detected_by,
        }
    }
    
    /// Calculate market impact
    fn calculate_market_impact(&self, event_type: &MicroburstType, severity: EventSeverity) -> MarketImpact {
        // Simplified impact calculation - would use more sophisticated models in production
        let base_impact = match severity {
            EventSeverity::Low => 1.0,
            EventSeverity::Medium => 2.5,
            EventSeverity::High => 5.0,
            EventSeverity::Critical => 10.0,
        };
        
        let (spread_impact, depth_impact, vol_impact) = match event_type {
            MicroburstType::VolumeSpike { .. } => (base_impact * 2.0, base_impact * 1.5, base_impact * 3.0),
            MicroburstType::PriceJump { .. } => (base_impact * 3.0, base_impact * 2.0, base_impact * 4.0),
            MicroburstType::LatencySpike { .. } => (base_impact * 1.5, base_impact * 1.0, base_impact * 1.5),
            MicroburstType::LiquidityEvaporation { .. } => (base_impact * 4.0, base_impact * 5.0, base_impact * 2.0),
            MicroburstType::QuoteStuffing { .. } => (base_impact * 1.0, base_impact * 0.5, base_impact * 1.0),
            MicroburstType::LiquidationCascade { .. } => (base_impact * 5.0, base_impact * 4.0, base_impact * 6.0),
        };
        
        MarketImpact {
            spread_widening_bps: spread_impact,
            depth_reduction_pct: depth_impact * 10.0,
            volatility_increase: vol_impact / 100.0,
            correlation_breakdown: severity >= EventSeverity::High,
            estimated_slippage_bps: (spread_impact + depth_impact) / 2.0,
        }
    }
    
    /// Store event in history
    fn store_event(&self, event: MicroburstEvent) {
        let mut events = self.events.write();
        if events.len() >= 10_000 {
            events.pop_front();
        }
        events.push_back(event);
    }
    
    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<MicroburstEvent> {
        let events = self.events.read();
        events.iter().rev().take(count).cloned().collect()
    }
    
    /// Simulate microburst for testing
    pub fn simulate_microburst(
        &self,
        symbol: Symbol,
        event_type: MicroburstType,
        start_time: DateTime<Utc>,
        duration_ms: u64,
    ) -> Vec<OrderBookUpdate> {
        use crate::replay::lob_simulator::{OrderBookUpdate, UpdateType, Side};
        
        let mut updates = Vec::new();
        let intervals = 10; // Generate 10 updates during the burst
        let interval_ms = duration_ms / intervals;
        
        match event_type {
            MicroburstType::VolumeSpike { spike_volume, .. } => {
                // Generate rapid trades
                for i in 0..intervals {
                    let timestamp = start_time + Duration::milliseconds((i * interval_ms) as i64);
                    let volume_per_update = Quantity(spike_volume.0 / Decimal::from(intervals));
                    
                    updates.push(OrderBookUpdate {
                        symbol: symbol.clone(),
                        exchange: types::Exchange::new("Simulated"),
                        timestamp,
                        sequence_number: i + 1,
                        update_type: UpdateType::Trade {
                            order_id: 1000000 + i,
                            traded_quantity: volume_per_update,
                            aggressor_side: if i % 2 == 0 { Side::Bid } else { Side::Ask },
                        },
                        latency_ns: 100_000,
                    });
                }
            }
            MicroburstType::PriceJump { start_price, end_price, .. } => {
                // Generate rapid price movements
                let price_step = (end_price.0 - start_price.0) / Decimal::from(intervals);
                
                for i in 0..intervals {
                    let timestamp = start_time + Duration::milliseconds((i * interval_ms) as i64);
                    let current_price = Price(start_price.0 + price_step * Decimal::from(i));
                    
                    // Clear and rebuild at new price
                    updates.push(OrderBookUpdate {
                        symbol: symbol.clone(),
                        exchange: types::Exchange::new("Simulated"),
                        timestamp,
                        sequence_number: i + 1,
                        update_type: UpdateType::Clear,
                        latency_ns: 50_000,
                    });
                    
                    // Add new orders at jumped price
                    updates.push(OrderBookUpdate {
                        symbol: symbol.clone(),
                        exchange: types::Exchange::new("Simulated"),
                        timestamp,
                        sequence_number: i + 2,
                        update_type: UpdateType::Add {
                            order_id: 2000000 + i,
                            side: Side::Bid,
                            price: Price(current_price.0 - Decimal::from_str("0.01").unwrap()),
                            quantity: Quantity(Decimal::from(100)),
                        },
                        latency_ns: 50_000,
                    });
                }
            }
            MicroburstType::LiquidityEvaporation { levels_affected, .. } => {
                // Cancel orders at multiple levels
                for level in 0..levels_affected {
                    let timestamp = start_time + Duration::milliseconds((level as u64 * interval_ms) as i64);
                    
                    updates.push(OrderBookUpdate {
                        symbol: symbol.clone(),
                        exchange: types::Exchange::new("Simulated"),
                        timestamp,
                        sequence_number: level as u64 + 1,
                        update_type: UpdateType::Cancel {
                            order_id: 3000000 + level as u64,
                        },
                        latency_ns: 25_000,
                    });
                }
            }
            _ => {
                // Other types would have their specific simulations
            }
        }
        
        updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::Exchange;
    use crate::replay::lob_simulator::OrderBookUpdate;
    
    #[test]
    fn test_microburst_detector_initialization() {
        let config = DetectorConfig::default();
        let detector = MicroburstDetector::new(config).unwrap();
        
        let events = detector.get_recent_events(10);
        assert_eq!(events.len(), 0);
    }
    
    #[test]
    fn test_volume_spike_detection() {
        let config = DetectorConfig {
            volume_z_threshold: 3.0,
            min_sample_size: 10,
            ..Default::default()
        };
        let detector = MicroburstDetector::new(config).unwrap();
        
        let symbol = Symbol("BTC-USDT".to_string());
        
        // Build baseline
        for i in 0..20 {
            let _ = detector.process_market_data(
                symbol.clone(),
                Price(Decimal::from(50000)),
                Quantity(Decimal::from(100)), // Normal volume
                1000,
                100,
                Utc::now(),
            );
        }
        
        // Trigger spike
        let result = detector.process_market_data(
            symbol.clone(),
            Price(Decimal::from(50000)),
            Quantity(Decimal::from(10000)), // 100x normal volume
            1000,
            100,
            Utc::now(),
        ).unwrap();
        
        assert!(result.is_some());
        if let Some(event) = result {
            match event.event_type {
                MicroburstType::VolumeSpike { .. } => (),
                _ => panic!("Expected volume spike event"),
            }
        }
    }
    
    #[test]
    fn test_price_jump_detection() {
        let config = DetectorConfig {
            price_jump_threshold_bps: 100.0, // 1% threshold
            ..Default::default()
        };
        let detector = MicroburstDetector::new(config).unwrap();
        
        let symbol = Symbol("ETH-USDT".to_string());
        
        // Establish baseline price
        let _ = detector.process_market_data(
            symbol.clone(),
            Price(Decimal::from(2000)),
            Quantity(Decimal::from(100)),
            1000,
            100,
            Utc::now(),
        );
        
        // Jump price by 2%
        let result = detector.process_market_data(
            symbol.clone(),
            Price(Decimal::from(2040)),
            Quantity(Decimal::from(100)),
            1000,
            100,
            Utc::now(),
        ).unwrap();
        
        assert!(result.is_some());
    }
    
    #[test]
    fn test_microburst_simulation() {
        let config = DetectorConfig::default();
        let detector = MicroburstDetector::new(config).unwrap();
        
        let symbol = Symbol("SOL-USDT".to_string());
        let event_type = MicroburstType::VolumeSpike {
            normal_volume: Quantity(Decimal::from(100)),
            spike_volume: Quantity(Decimal::from(5000)),
            duration_ms: 100,
            sigma_deviation: 5.0,
        };
        
        let updates = detector.simulate_microburst(
            symbol,
            event_type,
            Utc::now(),
            100,
        );
        
        assert_eq!(updates.len(), 10);
    }
}