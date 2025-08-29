// Latency Arbitrage Detection System
use rust_decimal::prelude::ToPrimitive;
// Team: Jordan (Performance) + Casey (Exchange) + Quinn (Risk)
// CRITICAL: Detect and prevent being front-run by faster traders
// References:
// - "The High-Frequency Trading Arms Race" - Budish et al. (2015)
// - "Latency Arbitrage in Fragmented Markets" - Wah & Wellman (2016)
// - "Flash Boys" - Michael Lewis (2014)

use std::sync::Arc;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, Duration};
use tracing::{error, warn, debug};
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use statrs::statistics::Statistics;

/// Latency arbitrage event
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct LatencyArbitrageEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: ArbitrageType,
    pub our_order_id: String,
    pub symbol: String,
    pub our_side: OrderSide,
    pub our_price: Decimal,
    pub our_quantity: Decimal,
    pub market_price_before: Decimal,
    pub market_price_at_execution: Decimal,
    pub market_price_after: Decimal,
    pub time_to_execution_ms: i64,
    pub adverse_price_move_bps: i32,
    pub likely_arbitrageur: Option<String>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum ArbitrageType {
    FrontRunning,       // Someone trades ahead of us
    BackRunning,        // Someone trades immediately after us
    Sandwiching,        // We're caught between two trades
    QuoteFading,        // Quotes disappear when we try to hit them
    LatencyShading,     // Prices systematically worse by the time we execute
    PhantomLiquidity,   // Liquidity vanishes when accessed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}

/// Latency measurement for different operations
#[derive(Debug, Clone, Default)]
/// TODO: Add docs
pub struct LatencyMetrics {
    pub data_feed_latency_ms: f64,      // Time from exchange to us
    pub decision_latency_ms: f64,       // Our processing time
    pub order_send_latency_ms: f64,     // Time to send order
    pub execution_latency_ms: f64,      // Time to get fill
    pub total_latency_ms: f64,          // End-to-end
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

/// Arbitrageur profile
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ArbitrageurProfile {
    pub identifier: String,
    pub detectioncount: u64,
    pub front_runningcount: u64,
    pub back_runningcount: u64,
    pub sandwichcount: u64,
    pub avg_speed_advantage_ms: f64,
    pub estimated_profit_taken: Decimal,
    pub confidence_score: f64,
    pub first_detected: DateTime<Utc>,
    pub last_detected: DateTime<Utc>,
}

/// Latency Arbitrage Detector
/// Jordan: "Every microsecond counts - we need to know when we're being beaten"
/// TODO: Add docs
pub struct LatencyArbitrageDetector {
    /// Recent order executions for analysis
    recent_executions: Arc<RwLock<VecDeque<ExecutionRecord>>>,
    
    /// Market data snapshots around our orders
    market_snapshots: Arc<DashMap<String, VecDeque<MarketSnapshot>>>,
    
    /// Detected arbitrage events
    arbitrage_events: Arc<RwLock<Vec<LatencyArbitrageEvent>>>,
    
    /// Arbitrageur profiles
    arbitrageurs: Arc<DashMap<String, ArbitrageurProfile>>,
    
    /// Our latency metrics
    our_latency: Arc<RwLock<LatencyMetrics>>,
    
    /// Detection thresholds
    config: DetectionConfig,
    
    /// Statistics
    total_orders_analyzed: Arc<AtomicU64>,
    arbitrage_detected: Arc<AtomicU64>,
    estimated_loss: Arc<RwLock<Decimal>>,
    false_positive_rate: Arc<RwLock<f64>>,
}

/// Execution record for our orders
#[derive(Debug, Clone)]
struct ExecutionRecord {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub intended_price: Decimal,
    pub executed_price: Decimal,
    pub quantity: Decimal,
    pub order_sent_time: DateTime<Utc>,
    pub execution_time: DateTime<Utc>,
    pub market_snapshot_before: Option<MarketSnapshot>,
    pub market_snapshot_after: Option<MarketSnapshot>,
}

/// Market snapshot at a point in time
#[derive(Debug, Clone)]
struct MarketSnapshot {
    pub timestamp: DateTime<Utc>,
    pub best_bid: Decimal,
    pub best_ask: Decimal,
    pub mid_price: Decimal,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
    pub recent_trades: Vec<TradeEvent>,
}

/// Trade event in the market
#[derive(Debug, Clone)]
struct TradeEvent {
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub quantity: Decimal,
    pub aggressor_side: OrderSide,
    pub participant_id: Option<String>,
}

/// Detection configuration
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct DetectionConfig {
    pub min_adverse_move_bps: i32,         // Min move to consider arbitrage (e.g., 2)
    pub max_execution_time_ms: i64,        // Max time for normal execution (e.g., 100)
    pub front_run_window_ms: i64,          // Window before our order (e.g., 50)
    pub back_run_window_ms: i64,           // Window after our order (e.g., 50)
    pub confidence_threshold: f64,         // Min confidence to flag (e.g., 0.7)
    pub sandwich_detection_enabled: bool,   // Detect sandwich attacks
    pub phantom_liquidity_threshold: f64,   // % of liquidity that vanishes
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_adverse_move_bps: 2,           // 2 basis points
            max_execution_time_ms: 100,        // 100ms expected execution
            front_run_window_ms: 50,           // 50ms before our order
            back_run_window_ms: 50,            // 50ms after our order
            confidence_threshold: 0.7,         // 70% confidence
            sandwich_detection_enabled: true,
            phantom_liquidity_threshold: 0.5,   // 50% vanishes
        }
    }
}

impl LatencyArbitrageDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self {
            recent_executions: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            market_snapshots: Arc::new(DashMap::new()),
            arbitrage_events: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            arbitrageurs: Arc::new(DashMap::new()),
            our_latency: Arc::new(RwLock::new(LatencyMetrics::default())),
            config,
            total_orders_analyzed: Arc::new(AtomicU64::new(0)),
            arbitrage_detected: Arc::new(AtomicU64::new(0)),
            estimated_loss: Arc::new(RwLock::new(Decimal::ZERO)),
            false_positive_rate: Arc::new(RwLock::new(0.05)), // Estimated 5%
        }
    }
    
    /// Record our order execution for analysis
    pub fn record_execution(
        &self,
        order_id: String,
        symbol: String,
        side: OrderSide,
        intended_price: Decimal,
        executed_price: Decimal,
        quantity: Decimal,
        order_sent_time: DateTime<Utc>,
        execution_time: DateTime<Utc>,
    ) {
        let _execution = ExecutionRecord {
            order_id: order_id.clone(),
            symbol: symbol.clone(),
            side,
            intended_price,
            executed_price,
            quantity,
            order_sent_time,
            execution_time,
            market_snapshot_before: self.get_snapshot_at(&symbol, order_sent_time),
            market_snapshot_after: self.get_snapshot_at(&symbol, execution_time),
        };
        
        // Store execution
        self.recent_executions.write().push_back(execution.clone());
        
        // Analyze for arbitrage
        self.analyze_execution(execution);
        
        // Update statistics
        self.total_orders_analyzed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update market snapshot
    pub fn update_market_snapshot(&self, symbol: &str, snapshot: MarketSnapshot) {
        let mut snapshots = self.market_snapshots
            .entry(symbol.to_string())
            .or_insert_with(|| VecDeque::with_capacity(1000));
        
        snapshots.push_back(snapshot);
        
        // Keep only last 10 seconds of snapshots
        let cutoff = Utc::now() - Duration::seconds(10);
        while let Some(front) = snapshots.front() {
            if front.timestamp < cutoff {
                snapshots.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get market snapshot at specific time
    fn get_snapshot_at(&self, symbol: &str, timestamp: DateTime<Utc>) -> Option<MarketSnapshot> {
        self.market_snapshots.get(symbol).and_then(|snapshots| {
            // Find closest snapshot to timestamp
            snapshots.iter()
                .min_by_key(|s| (s.timestamp - timestamp).num_milliseconds().abs())
                .cloned()
        })
    }
    
    /// Analyze execution for latency arbitrage
    fn analyze_execution(&self, execution: ExecutionRecord) {
        let _execution_time_ms = (execution.execution_time - execution.order_sent_time)
            .num_milliseconds();
        
        // Check for slow execution (potential arbitrage opportunity)
        if execution_time_ms > self.config.max_execution_time_ms {
            debug!(
                "Slow execution detected: {}ms for order {}",
                execution_time_ms, execution.order_id
            );
        }
        
        // Calculate adverse price movement
        let _adverse_move_bps = match execution.side {
            OrderSide::Buy => {
                ((execution.executed_price - execution.intended_price) / 
                 execution.intended_price * dec!(10000)).round_dp(0).to_i32().unwrap_or(0)
            }
            OrderSide::Sell => {
                ((execution.intended_price - execution.executed_price) / 
                 execution.intended_price * dec!(10000)).round_dp(0).to_i32().unwrap_or(0)
            }
        };
        
        if adverse_move_bps < self.config.min_adverse_move_bps {
            return; // No significant adverse movement
        }
        
        // Detect different types of arbitrage
        let mut arbitrage_type = None;
        let mut confidence = 0.0;
        let mut likely_arbitrageur = None;
        
        // Check for front-running
        if let Some(snapshot_before) = &execution.market_snapshot_before {
            if self.detect_front_running(&execution, snapshot_before) {
                arbitrage_type = Some(ArbitrageType::FrontRunning);
                confidence += 0.4;
                
                // Try to identify the front-runner
                if let Some(trader) = self.identify_front_runner(_snapshot_before, &execution) {
                    likely_arbitrageur = Some(trader);
                    confidence += 0.2;
                }
            }
        }
        
        // Check for quote fading
        if self.detect_quote_fading(&execution) {
            arbitrage_type = Some(ArbitrageType::QuoteFading);
            confidence += 0.3;
        }
        
        // Check for sandwich attack
        if self.config.sandwich_detection_enabled {
            if let (Some(before), Some(after)) = 
                (&execution.market_snapshot_before, &execution.market_snapshot_after) {
                if self.detect_sandwich(_before, after, &execution) {
                    arbitrage_type = Some(ArbitrageType::Sandwiching);
                    confidence = 0.9; // High confidence for sandwich
                }
            }
        }
        
        // Check for phantom liquidity
        if self.detect_phantom_liquidity(&execution) {
            arbitrage_type = Some(ArbitrageType::PhantomLiquidity);
            confidence += 0.3;
        }
        
        // Default to latency shading if we have adverse movement but no specific pattern
        if arbitrage_type.is_none() && adverse_move_bps >= self.config.min_adverse_move_bps {
            arbitrage_type = Some(ArbitrageType::LatencyShading);
            confidence = 0.5;
        }
        
        // Record event if confidence exceeds threshold
        if let Some(arb_type) = arbitrage_type {
            if confidence >= self.config.confidence_threshold {
                let event = LatencyArbitrageEvent {
                    timestamp: execution.execution_time,
                    event_type: arb_type,
                    our_order_id: execution.order_id.clone(),
                    symbol: execution.symbol.clone(),
                    our_side: execution.side,
                    our_price: execution.executed_price,
                    our_quantity: execution.quantity,
                    market_price_before: execution.market_snapshot_before
                        .as_ref()
                        .map(|s| s.mid_price)
                        .unwrap_or(execution.intended_price),
                    market_price_at_execution: execution.executed_price,
                    market_price_after: execution.market_snapshot_after
                        .as_ref()
                        .map(|s| s.mid_price)
                        .unwrap_or(execution.executed_price),
                    time_to_execution_ms: execution_time_ms,
                    adverse_price_move_bps: adverse_move_bps,
                    likely_arbitrageur: likely_arbitrageur.clone(),
                    confidence_score: confidence,
                };
                
                warn!(
                    "Latency arbitrage detected: {:?} on {} (confidence: {:.1}%, loss: {} bps)",
                    arb_type, execution.symbol, confidence * 100.0, adverse_move_bps
                );
                
                // Store event
                self.arbitrage_events.write().push(event);
                self.arbitrage_detected.fetch_add(1, Ordering::Relaxed);
                
                // Update arbitrageur profile if identified
                if let Some(arb_id) = likely_arbitrageur {
                    self.update_arbitrageur_profile(&arb_id, arb_type);
                }
                
                // Estimate loss
                let loss = execution.quantity * execution.executed_price * 
                          Decimal::from(adverse_move_bps) / dec!(10000);
                *self.estimated_loss.write() += loss;
            }
        }
    }
    
    /// Detect front-running pattern
    fn detect_front_running(&self, execution: &ExecutionRecord, snapshot: &MarketSnapshot) -> bool {
        // Look for trades just before our order that moved the price adversely
        let _window_start = execution.order_sent_time - 
                          Duration::milliseconds(self.config.front_run_window_ms);
        
        let suspicious_trades: Vec<_> = snapshot.recent_trades.iter()
            .filter(|t| t.timestamp >= window_start && t.timestamp < execution.order_sent_time)
            .filter(|t| {
                // Same direction as our order = potential front-runner
                match execution.side {
                    OrderSide::Buy => t.aggressor_side == OrderSide::Buy && 
                                     t.price >= execution.intended_price,
                    OrderSide::Sell => t.aggressor_side == OrderSide::Sell && 
                                      t.price <= execution.intended_price,
                }
            })
            .collect();
        
        !suspicious_trades.is_empty()
    }
    
    /// Detect quote fading (liquidity disappearing)
    fn detect_quote_fading(&self, execution: &ExecutionRecord) -> bool {
        if let Some(snapshot) = &execution.market_snapshot_before {
            let _expected_liquidity = match execution.side {
                OrderSide::Buy => snapshot.ask_size,
                OrderSide::Sell => snapshot.bid_size,
            };
            
            // Check if we got significantly less than expected
            execution.quantity < expected_liquidity * 
                                Decimal::from_f64_retain(1.0 - self.config.phantom_liquidity_threshold)
                                .unwrap_or(dec!(0.5))
        } else {
            false
        }
    }
    
    /// Detect sandwich attack
    fn detect_sandwich(
        &self,
        before: &MarketSnapshot,
        after: &MarketSnapshot,
        execution: &ExecutionRecord,
    ) -> bool {
        // Look for trade before and after ours in opposite directions
        let _window = Duration::milliseconds(self.config.back_run_window_ms);
        
        let _trades_before = before.recent_trades.iter()
            .filter(|t| (execution.order_sent_time - t.timestamp) < window)
            .any(|t| t.aggressor_side == execution.side);
        
        let _trades_after = after.recent_trades.iter()
            .filter(|t| (t.timestamp - execution.execution_time) < window)
            .any(|t| t.aggressor_side != execution.side); // Opposite direction
        
        trades_before && trades_after
    }
    
    /// Detect phantom liquidity
    fn detect_phantom_liquidity(&self, execution: &ExecutionRecord) -> bool {
        if let Some(snapshot) = &execution.market_snapshot_before {
            let _shown_liquidity = match execution.side {
                OrderSide::Buy => snapshot.ask_size,
                OrderSide::Sell => snapshot.bid_size,
            };
            
            // Liquidity vanished when we tried to access it
            execution.quantity < shown_liquidity * dec!(0.1) // Got less than 10% of shown
        } else {
            false
        }
    }
    
    /// Try to identify the front-runner
    fn identify_front_runner(&self, snapshot: &MarketSnapshot, execution: &ExecutionRecord) -> Option<String> {
        // Find the most likely front-runner based on timing and size
        snapshot.recent_trades.iter()
            .filter(|t| {
                let _time_diff = (execution.order_sent_time - t.timestamp).num_milliseconds();
                time_diff > 0 && time_diff < self.config.front_run_window_ms
            })
            .filter(|t| t.aggressor_side == execution.side)
            .max_by_key(|t| (t.quantity.to_f64().unwrap_or(0.0) * 1000.0) as i64)
            .and_then(|t| t.participant_id.clone())
    }
    
    /// Update arbitrageur profile
    fn update_arbitrageur_profile(&self, identifier: &str, arb_type: ArbitrageType) {
        let mut profile = self.arbitrageurs
            .entry(identifier.to_string())
            .or_insert_with(|| ArbitrageurProfile {
                identifier: identifier.to_string(),
                detectioncount: 0,
                front_runningcount: 0,
                back_runningcount: 0,
                sandwichcount: 0,
                avg_speed_advantage_ms: 0.0,
                estimated_profit_taken: Decimal::ZERO,
                confidence_score: 0.5,
                first_detected: Utc::now(),
                last_detected: Utc::now(),
            });
        
        profile.detectioncount += 1;
        profile.last_detected = Utc::now();
        
        match arb_type {
            ArbitrageType::FrontRunning => profile.front_runningcount += 1,
            ArbitrageType::BackRunning => profile.back_runningcount += 1,
            ArbitrageType::Sandwiching => profile.sandwichcount += 1,
            _ => {}
        }
        
        // Update confidence based on detection frequency
        profile.confidence_score = (profile.detectioncount as f64 / 10.0).min(0.95);
        
        if profile.detectioncount > 5 {
            error!(
                "Frequent arbitrageur detected: {} ({} detections)",
                identifier, profile.detectioncount
            );
        }
    }
    
    /// Update our latency metrics
    pub fn update_latency_metrics(&self, metrics: LatencyMetrics) {
        *self.our_latency.write() = metrics;
    }
    
    /// Get protection recommendations
    pub fn get_protection_recommendations(&self) -> ArbitrageProtection {
        let _events = self.arbitrage_events.read();
        let _recent_events = events.iter()
            .filter(|e| e.timestamp > Utc::now() - Duration::hours(1))
            .count();
        
        let _arbitrage_rate = recent_events as f64 / 
                           self.total_orders_analyzed.load(Ordering::Relaxed).max(1) as f64;
        
        if arbitrage_rate > 0.1 {
            // High arbitrage rate - need aggressive protection
            ArbitrageProtection {
                strategy: "AGGRESSIVE_PROTECTION".to_string(),
                recommendations: vec![
                    "Use randomized order timing".to_string(),
                    "Split orders across multiple venues".to_string(),
                    "Use dark pools when available".to_string(),
                    "Implement order hiding techniques".to_string(),
                    "Add artificial latency variance".to_string(),
                ],
                estimated_improvement: "30-50% reduction in arbitrage".to_string(),
                urgency: "HIGH".to_string(),
            }
        } else if arbitrage_rate > 0.05 {
            // Moderate arbitrage rate
            ArbitrageProtection {
                strategy: "MODERATE_PROTECTION".to_string(),
                recommendations: vec![
                    "Use iceberg orders".to_string(),
                    "Randomize order sizes".to_string(),
                    "Monitor for known arbitrageurs".to_string(),
                    "Avoid predictable patterns".to_string(),
                ],
                estimated_improvement: "20-30% reduction in arbitrage".to_string(),
                urgency: "MEDIUM".to_string(),
            }
        } else {
            // Low arbitrage rate
            ArbitrageProtection {
                strategy: "STANDARD_PROTECTION".to_string(),
                recommendations: vec![
                    "Continue monitoring".to_string(),
                    "Maintain current practices".to_string(),
                    "Review periodically".to_string(),
                ],
                estimated_improvement: "Current protection adequate".to_string(),
                urgency: "LOW".to_string(),
            }
        }
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> LatencyArbitrageStats {
        let _total = self.total_orders_analyzed.load(Ordering::Relaxed);
        let _detected = self.arbitrage_detected.load(Ordering::Relaxed);
        
        LatencyArbitrageStats {
            total_orders_analyzed: total,
            arbitrage_events_detected: detected,
            arbitrage_rate: detected as f64 / total.max(1) as f64,
            estimated_total_loss: *self.estimated_loss.read(),
            avg_adverse_move_bps: self.calculate_avg_adverse_move(),
            our_avg_latency_ms: self.our_latency.read().total_latency_ms,
            known_arbitrageurs: self.arbitrageurs.len(),
            false_positive_rate: *self.false_positive_rate.read(),
        }
    }
    
    /// Calculate average adverse move
    fn calculate_avg_adverse_move(&self) -> f64 {
        let _events = self.arbitrage_events.read();
        if events.is_empty() {
            return 0.0;
        }
        
        let sum: i32 = events.iter().map(|e| e.adverse_price_move_bps).sum();
        sum as f64 / events.len() as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ArbitrageProtection {
    pub strategy: String,
    pub recommendations: Vec<String>,
    pub estimated_improvement: String,
    pub urgency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct LatencyArbitrageStats {
    pub total_orders_analyzed: u64,
    pub arbitrage_events_detected: u64,
    pub arbitrage_rate: f64,
    pub estimated_total_loss: Decimal,
    pub avg_adverse_move_bps: f64,
    pub our_avg_latency_ms: f64,
    pub known_arbitrageurs: usize,
    pub false_positive_rate: f64,
}

// ============================================================================
// TESTS - Jordan & Casey validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_front_running_detection() {
        let detector = LatencyArbitrageDetector::new(DetectionConfig::default());
        
        // Create market snapshot with suspicious trade
        let mut snapshot = MarketSnapshot {
            timestamp: Utc::now(),
            best_bid: dec!(49990),
            best_ask: dec!(50010),
            mid_price: dec!(50000),
            bid_size: dec!(10),
            ask_size: dec!(10),
            recent_trades: vec![],
        };
        
        // Add a front-running trade
        snapshot.recent_trades.push(TradeEvent {
            timestamp: Utc::now() - Duration::milliseconds(30),
            price: dec!(50005),
            quantity: dec!(1.0),
            aggressor_side: OrderSide::Buy,
            participant_id: Some("FAST_TRADER".to_string()),
        });
        
        // Update market snapshot
        detector.update_market_snapshot("BTCUSDT", snapshot);
        
        // Record our execution with adverse price
        detector.record_execution(
            "ORDER_1".to_string(),
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(50000),  // Intended
            dec!(50010),  // Executed (worse)
            dec!(0.5),
            Utc::now() - Duration::milliseconds(100),
            Utc::now(),
        );
        
        // Should detect arbitrage
        let _stats = detector.get_statistics();
        assert!(stats.arbitrage_events_detected > 0);
    }
    
    #[test]
    fn test_protection_recommendations() {
        let detector = LatencyArbitrageDetector::new(DetectionConfig::default());
        
        // Get initial recommendations (no data)
        let _protection = detector.get_protection_recommendations();
        assert_eq!(protection.urgency, "LOW");
        
        // Simulate high arbitrage rate and verify recommendations change
        // (Would need to add many events to test properly)
    }
}