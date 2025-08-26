// Playback Engine - Orchestrates LOB replay and backtesting
// DEEP DIVE: Complete backtesting framework with realistic market simulation
//
// References:
// - "Backtesting" - Pardo (2008)
// - "Quantitative Trading" - Chan (2009)
// - "Inside the Black Box" - Narang (2013)

use std::sync::Arc;
use std::collections::VecDeque;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tokio::sync::mpsc;
use tokio::time::{sleep, Instant};
use async_trait::async_trait;
use tracing::{info, debug, warn, instrument};

use types::{Price, Quantity, Symbol, Exchange};
use crate::replay::{
    lob_simulator::{LOBSimulator, OrderBookUpdate, SimulatorConfig},
    microburst_detector::{MicroburstDetector, DetectorConfig, MicroburstEvent},
    slippage_model::{SlippageModel, SlippageConfig, ExecutionCost, TradeSide},
    fee_calculator::{FeeCalculator, OrderType},
    historical_loader::{HistoricalDataLoader, DataSource, TickData, TickType},
    market_impact::{MarketImpactCalculator, ImpactParameters},
};

/// Playback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackConfig {
    /// Playback speed multiplier (1.0 = realtime, 10.0 = 10x speed)
    pub speed_multiplier: f64,
    
    /// Maximum events per second (rate limiting)
    pub max_events_per_sec: u64,
    
    /// Enable slippage modeling
    pub enable_slippage: bool,
    
    /// Enable fee calculation
    pub enable_fees: bool,
    
    /// Enable microburst detection
    pub enable_microburst_detection: bool,
    
    /// Enable market impact modeling
    pub enable_market_impact: bool,
    
    /// Warmup period (seconds) before trading starts
    pub warmup_period_sec: u64,
    
    /// Buffer size for event queue
    pub event_buffer_size: usize,
    
    /// Enable latency simulation
    pub simulate_latency: bool,
    
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            speed_multiplier: 1.0,
            max_events_per_sec: 100_000,
            enable_slippage: true,
            enable_fees: true,
            enable_microburst_detection: true,
            enable_market_impact: true,
            warmup_period_sec: 60,
            event_buffer_size: 100_000,
            simulate_latency: true,
            random_seed: None,
        }
    }
}

/// Playback speed control
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaybackSpeed {
    /// As fast as possible
    Maximum,
    /// Real-time speed
    Realtime,
    /// Custom multiplier
    Custom(f64),
    /// Step through events manually
    Manual,
}

/// Simulation event types
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    /// Order book update
    BookUpdate(OrderBookUpdate),
    
    /// Trade execution
    Trade {
        symbol: Symbol,
        price: Price,
        quantity: Quantity,
        side: TradeSide,
        timestamp: DateTime<Utc>,
    },
    
    /// Microburst detected
    Microburst(MicroburstEvent),
    
    /// Simulation statistics
    Statistics(SimulationStats),
    
    /// Checkpoint for recovery
    Checkpoint {
        timestamp: DateTime<Utc>,
        sequence: u64,
    },
    
    /// End of data
    EndOfData,
}

/// Event sequence for replay
#[derive(Debug, Clone)]
pub struct EventSequence {
    pub events: VecDeque<SimulationEvent>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_events: u64,
}

/// Replay result
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub events_processed: u64,
    pub events_skipped: u64,
    pub microbursts_detected: u32,
    pub total_slippage_bps: f64,
    pub total_fees_paid: Decimal,
    pub simulation_time_ms: u64,
    pub effective_speed: f64,
}

/// Simulation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStats {
    pub timestamp: DateTime<Utc>,
    pub events_processed: u64,
    pub events_per_second: f64,
    pub current_spread_bps: f64,
    pub book_depth_imbalance: f64,
    pub latency_p50_us: u64,
    pub latency_p99_us: u64,
    pub memory_usage_mb: f64,
}

/// Strategy interface for backtesting
#[async_trait]
pub trait TradingStrategy: Send + Sync {
    /// Called on each order book update
    async fn on_book_update(&mut self, update: &OrderBookUpdate) -> Option<StrategySignal>;
    
    /// Called on each trade
    async fn on_trade(&mut self, symbol: &Symbol, price: Price, quantity: Quantity, side: TradeSide);
    
    /// Called on microburst detection
    async fn on_microburst(&mut self, event: &MicroburstEvent);
    
    /// Get current positions
    fn get_positions(&self) -> Vec<Position>;
    
    /// Calculate PnL
    fn calculate_pnl(&self, current_prices: &[(Symbol, Price)]) -> Decimal;
}

/// Trading signal from strategy
#[derive(Debug, Clone)]
pub struct StrategySignal {
    pub symbol: Symbol,
    pub side: TradeSide,
    pub quantity: Quantity,
    pub order_type: OrderType,
    pub limit_price: Option<Price>,
    pub urgency: SignalUrgency,
}

/// Signal urgency levels
#[derive(Debug, Clone, Copy)]
pub enum SignalUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Position tracking
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: Symbol,
    pub quantity: Decimal,
    pub average_price: Price,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
}

/// Main playback engine
pub struct PlaybackEngine {
    config: Arc<PlaybackConfig>,
    lob_simulator: Arc<LOBSimulator>,
    microburst_detector: Arc<MicroburstDetector>,
    slippage_model: Arc<SlippageModel>,
    fee_calculator: Arc<FeeCalculator>,
    impact_calculator: Arc<MarketImpactCalculator>,
    
    // Event management
    event_queue: Arc<RwLock<VecDeque<SimulationEvent>>>,
    event_sender: mpsc::Sender<SimulationEvent>,
    event_receiver: Arc<RwLock<mpsc::Receiver<SimulationEvent>>>,
    
    // Playback state
    current_time: Arc<RwLock<DateTime<Utc>>>,
    events_processed: Arc<RwLock<u64>>,
    playback_speed: Arc<RwLock<PlaybackSpeed>>,
    is_running: Arc<RwLock<bool>>,
    
    // Statistics
    stats_collector: Arc<StatsCollector>,
}

/// Statistics collector
struct StatsCollector {
    events_processed: RwLock<u64>,
    events_skipped: RwLock<u64>,
    microbursts_detected: RwLock<u32>,
    total_slippage: RwLock<f64>,
    total_fees: RwLock<Decimal>,
    latency_samples: RwLock<Vec<u64>>,
    start_time: RwLock<Option<Instant>>,
}

impl StatsCollector {
    fn new() -> Self {
        Self {
            events_processed: RwLock::new(0),
            events_skipped: RwLock::new(0),
            microbursts_detected: RwLock::new(0),
            total_slippage: RwLock::new(0.0),
            total_fees: RwLock::new(Decimal::ZERO),
            latency_samples: RwLock::new(Vec::with_capacity(10000)),
            start_time: RwLock::new(None),
        }
    }
    
    fn record_event(&self) {
        *self.events_processed.write() += 1;
    }
    
    fn record_microburst(&self) {
        *self.microbursts_detected.write() += 1;
    }
    
    fn record_slippage(&self, slippage_bps: f64) {
        *self.total_slippage.write() += slippage_bps;
    }
    
    fn record_fee(&self, fee: Decimal) {
        *self.total_fees.write() += fee;
    }
    
    fn record_latency(&self, latency_us: u64) {
        let mut samples = self.latency_samples.write();
        if samples.len() >= 10000 {
            samples.remove(0);
        }
        samples.push(latency_us);
    }
    
    fn get_stats(&self) -> SimulationStats {
        let samples = self.latency_samples.read();
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_unstable();
        
        let p50 = if !sorted_samples.is_empty() {
            sorted_samples[sorted_samples.len() / 2]
        } else {
            0
        };
        
        let p99 = if !sorted_samples.is_empty() {
            sorted_samples[sorted_samples.len() * 99 / 100]
        } else {
            0
        };
        
        let events = *self.events_processed.read();
        let elapsed = self.start_time.read()
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(1.0);
        
        SimulationStats {
            timestamp: Utc::now(),
            events_processed: events,
            events_per_second: events as f64 / elapsed,
            current_spread_bps: 0.0,  // Would get from LOB
            book_depth_imbalance: 0.0,  // Would calculate from LOB
            latency_p50_us: p50,
            latency_p99_us: p99,
            memory_usage_mb: 0.0,  // Would get from system
        }
    }
}

impl PlaybackEngine {
    pub fn new(config: PlaybackConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel(config.event_buffer_size);
        
        Ok(Self {
            config: Arc::new(config.clone()),
            lob_simulator: Arc::new(LOBSimulator::new(SimulatorConfig::default())?),
            microburst_detector: Arc::new(MicroburstDetector::new(DetectorConfig::default())?),
            slippage_model: Arc::new(SlippageModel::new(SlippageConfig::default())?),
            fee_calculator: Arc::new(FeeCalculator::new()),
            impact_calculator: Arc::new(MarketImpactCalculator::new()),
            event_queue: Arc::new(RwLock::new(VecDeque::with_capacity(config.event_buffer_size))),
            event_sender: tx,
            event_receiver: Arc::new(RwLock::new(rx)),
            current_time: Arc::new(RwLock::new(Utc::now())),
            events_processed: Arc::new(RwLock::new(0)),
            playback_speed: Arc::new(RwLock::new(PlaybackSpeed::Realtime)),
            is_running: Arc::new(RwLock::new(false)),
            stats_collector: Arc::new(StatsCollector::new()),
        })
    }
    
    /// Load historical data
    #[instrument(skip(self))]
    pub async fn load_data(&self, data_source: DataSource) -> Result<EventSequence> {
        info!("Loading historical data...");
        
        let mut loader = HistoricalDataLoader::new(data_source);
        let ticks = loader.load().await?;
        
        if ticks.is_empty() {
            anyhow::bail!("No data loaded");
        }
        
        let start_time = ticks.first().unwrap().timestamp;
        let end_time = ticks.last().unwrap().timestamp;
        let total_events = ticks.len() as u64;
        
        // Convert ticks to simulation events
        let mut events = VecDeque::with_capacity(ticks.len());
        
        for tick in ticks {
            match tick.tick_type {
                TickType::BookUpdate(update) => {
                    events.push_back(SimulationEvent::BookUpdate(update));
                }
                TickType::Trade(trade) => {
                    events.push_back(SimulationEvent::Trade {
                        symbol: tick.symbol,
                        price: trade.price,
                        quantity: trade.quantity,
                        side: trade.side,
                        timestamp: tick.timestamp,
                    });
                }
                _ => {}
            }
        }
        
        info!("Loaded {} events from {} to {}", total_events, start_time, end_time);
        
        Ok(EventSequence {
            events,
            start_time,
            end_time,
            total_events,
        })
    }
    
    /// Start playback
    #[instrument(skip(self, event_sequence, strategy))]
    pub async fn start_playback(
        &self,
        event_sequence: EventSequence,
        strategy: Option<Box<dyn TradingStrategy>>,
    ) -> Result<ReplayResult> {
        info!("Starting playback with {} events", event_sequence.total_events);
        
        *self.is_running.write() = true;
        *self.current_time.write() = event_sequence.start_time;
        *self.stats_collector.start_time.write() = Some(Instant::now());
        
        // Warmup period
        if self.config.warmup_period_sec > 0 {
            info!("Warming up for {} seconds...", self.config.warmup_period_sec);
            let warmup_end = event_sequence.start_time + Duration::seconds(self.config.warmup_period_sec as i64);
            
            for event in &event_sequence.events {
                if let SimulationEvent::BookUpdate(update) = event {
                    if update.timestamp > warmup_end {
                        break;
                    }
                    self.lob_simulator.process_update(update.clone()).await?;
                }
            }
        }
        
        // Main playback loop
        let mut strategy = strategy;
        let mut last_event_time = event_sequence.start_time;
        let mut events_in_second = 0u64;
        let mut second_start = Instant::now();
        
        for event in event_sequence.events {
            if !*self.is_running.read() {
                break;
            }
            
            // Rate limiting
            events_in_second += 1;
            if events_in_second >= self.config.max_events_per_sec {
                let elapsed = second_start.elapsed();
                if elapsed < std::time::Duration::from_secs(1) {
                    sleep(std::time::Duration::from_secs(1) - elapsed).await;
                }
                events_in_second = 0;
                second_start = Instant::now();
            }
            
            // Playback speed control
            match *self.playback_speed.read() {
                PlaybackSpeed::Realtime => {
                    if let SimulationEvent::BookUpdate(ref update) = event {
                        let time_diff = update.timestamp - last_event_time;
                        if time_diff > Duration::zero() {
                            let sleep_duration = time_diff.to_std()
                                .unwrap_or(std::time::Duration::from_millis(1));
                            sleep(sleep_duration).await;
                        }
                        last_event_time = update.timestamp;
                    }
                }
                PlaybackSpeed::Custom(multiplier) => {
                    if let SimulationEvent::BookUpdate(ref update) = event {
                        let time_diff = update.timestamp - last_event_time;
                        if time_diff > Duration::zero() {
                            let sleep_ms = (time_diff.num_milliseconds() as f64 / multiplier) as u64;
                            if sleep_ms > 0 {
                                sleep(std::time::Duration::from_millis(sleep_ms)).await;
                            }
                        }
                        last_event_time = update.timestamp;
                    }
                }
                PlaybackSpeed::Manual => {
                    // Wait for manual trigger (not implemented here)
                }
                PlaybackSpeed::Maximum => {
                    // No delay
                }
            }
            
            // Process event
            self.process_event(event, &mut strategy).await?;
            
            // Emit statistics periodically
            if *self.events_processed.read() % 10000 == 0 {
                let stats = self.stats_collector.get_stats();
                self.event_sender.send(SimulationEvent::Statistics(stats)).await?;
            }
        }
        
        // Calculate final results
        let elapsed = self.stats_collector.start_time.read()
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(1);
        
        let result = ReplayResult {
            events_processed: *self.stats_collector.events_processed.read(),
            events_skipped: *self.stats_collector.events_skipped.read(),
            microbursts_detected: *self.stats_collector.microbursts_detected.read(),
            total_slippage_bps: *self.stats_collector.total_slippage.read(),
            total_fees_paid: *self.stats_collector.total_fees.read(),
            simulation_time_ms: elapsed,
            effective_speed: event_sequence.total_events as f64 / (elapsed as f64 / 1000.0),
        };
        
        info!("Playback complete: {:?}", result);
        *self.is_running.write() = false;
        
        Ok(result)
    }
    
    /// Process a single event
    async fn process_event(
        &self,
        event: SimulationEvent,
        strategy: &mut Option<Box<dyn TradingStrategy>>,
    ) -> Result<()> {
        match event {
            SimulationEvent::BookUpdate(update) => {
                // Update LOB
                self.lob_simulator.process_update(update.clone()).await?;
                
                // Check for microbursts
                if self.config.enable_microburst_detection {
                    // Would extract relevant data from update
                    // self.microburst_detector.process_market_data(...)?;
                }
                
                // Strategy callback
                if let Some(ref mut strat) = strategy {
                    if let Some(signal) = strat.on_book_update(&update).await {
                        self.process_strategy_signal(signal).await?;
                    }
                }
                
                self.stats_collector.record_event();
            }
            SimulationEvent::Trade { symbol, price, quantity, side, .. } => {
                // Strategy callback
                if let Some(ref mut strat) = strategy {
                    strat.on_trade(&symbol, price, quantity, side).await;
                }
                
                self.stats_collector.record_event();
            }
            SimulationEvent::Microburst(event) => {
                // Strategy callback
                if let Some(ref mut strat) = strategy {
                    strat.on_microburst(&event).await;
                }
                
                self.stats_collector.record_microburst();
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Process strategy signal
    async fn process_strategy_signal(&self, signal: StrategySignal) -> Result<()> {
        // Get current order book
        let book = self.lob_simulator.get_book(&signal.symbol);
        
        if let Some(book) = book {
            // Calculate slippage
            if self.config.enable_slippage {
                let cost = self.slippage_model.calculate_slippage(
                    &signal.symbol,
                    signal.side,
                    signal.quantity.clone(),
                    &book,
                    100,  // Execution time
                )?;
                
                self.stats_collector.record_slippage(cost.total_cost_bps);
            }
            
            // Calculate fees
            if self.config.enable_fees {
                let is_maker = matches!(signal.order_type, OrderType::Limit | OrderType::PostOnly);
                let price = signal.limit_price.unwrap_or_else(|| book.mid_price.unwrap());
                
                let (fee, _) = self.fee_calculator.calculate_fee(
                    &book.exchange,
                    &signal.symbol,
                    signal.quantity,
                    price,
                    is_maker,
                    None,
                )?;
                
                self.stats_collector.record_fee(fee);
            }
        }
        
        Ok(())
    }
    
    /// Set playback speed
    pub fn set_speed(&self, speed: PlaybackSpeed) {
        *self.playback_speed.write() = speed;
    }
    
    /// Pause playback
    pub fn pause(&self) {
        *self.is_running.write() = false;
    }
    
    /// Resume playback
    pub fn resume(&self) {
        *self.is_running.write() = true;
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> SimulationStats {
        self.stats_collector.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;
    
    struct DummyStrategy;
    
    #[async_trait]
    impl TradingStrategy for DummyStrategy {
        async fn on_book_update(&mut self, _update: &OrderBookUpdate) -> Option<StrategySignal> {
            None
        }
        
        async fn on_trade(&mut self, _symbol: &Symbol, _price: Price, _quantity: Quantity, _side: TradeSide) {}
        
        async fn on_microburst(&mut self, _event: &MicroburstEvent) {}
        
        fn get_positions(&self) -> Vec<Position> {
            Vec::new()
        }
        
        fn calculate_pnl(&self, _current_prices: &[(Symbol, Price)]) -> Decimal {
            Decimal::ZERO
        }
    }
    
    #[tokio::test]
    async fn test_playback_engine_creation() {
        let config = PlaybackConfig::default();
        let engine = PlaybackEngine::new(config).unwrap();
        
        assert!(!*engine.is_running.read());
        assert_eq!(*engine.events_processed.read(), 0);
    }
    
    #[tokio::test]
    async fn test_data_loading_and_playback() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        
        // Create test data
        let mut file = tokio::fs::File::create(&file_path).await.unwrap();
        file.write_all(b"timestamp,price,quantity,side\n").await.unwrap();
        file.write_all(b"2024-01-01T00:00:00Z,50000.0,1.0,buy\n").await.unwrap();
        file.write_all(b"2024-01-01T00:00:01Z,50001.0,2.0,sell\n").await.unwrap();
        
        let source = DataSource::CustomCSV {
            file_path,
            delimiter: ',',
            has_header: true,
            timestamp_col: 0,
            price_col: 1,
            quantity_col: 2,
            side_col: 3,
        };
        
        let config = PlaybackConfig {
            speed_multiplier: 10.0,  // 10x speed
            warmup_period_sec: 0,
            ..Default::default()
        };
        
        let engine = PlaybackEngine::new(config).unwrap();
        let sequence = engine.load_data(source).await.unwrap();
        
        assert_eq!(sequence.total_events, 2);
        
        // Run playback with dummy strategy
        let strategy = Box::new(DummyStrategy);
        let result = engine.start_playback(sequence, Some(strategy)).await.unwrap();
        
        assert_eq!(result.events_processed, 2);
    }
}