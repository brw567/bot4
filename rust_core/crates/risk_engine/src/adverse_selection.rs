// Adverse Selection Detection System
// Team: Quinn (Risk) + Casey (Exchange) + Alex (Priority)
// CRITICAL: Detects toxic flow and protects from being picked off
// References:
// - "Toxic Flow and Liquidity in Quant Trading" - Easley et al. (2012)
// - "Adverse Selection in Electronic Markets" - Glosten & Milgrom (1985)

use std::sync::Arc;
use std::collections::VecDeque;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc, Duration};
use tracing::{error, warn};
use dashmap::DashMap;
use serde::{Serialize, Deserialize};

use order_management::{OrderId, OrderSide, Fill};

/// Time windows for adverse selection analysis
const SHORT_WINDOW_MS: i64 = 100;   // 100ms for HFT detection
const MEDIUM_WINDOW_MS: i64 = 1000; // 1 second for momentum
const LONG_WINDOW_MS: i64 = 10000;  // 10 seconds for drift

/// Adverse selection event
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct AdverseSelectionEvent {
    pub timestamp: DateTime<Utc>,
    pub order_id: OrderId,
    pub symbol: String,
    pub side: OrderSide,
    pub fill_price: Decimal,
    pub mid_price_at_fill: Decimal,
    pub mid_price_after_100ms: Option<Decimal>,
    pub mid_price_after_1s: Option<Decimal>,
    pub mid_price_after_10s: Option<Decimal>,
    pub toxicity_score: f64,
    pub counterparty: Option<String>,
}

/// Counterparty toxicity profile
#[derive(Debug, Clone, Default)]
/// TODO: Add docs
pub struct CounterpartyProfile {
    pub identifier: String,
    pub total_trades: u64,
    pub toxic_trades: u64,
    pub toxicity_ratio: f64,
    pub avg_adverse_move_bps: i32,
    pub last_trade: Option<DateTime<Utc>>,
    pub is_flagged: bool,
}

/// Adverse selection detector
/// TODO: Add docs
pub struct AdverseSelectionDetector {
    /// Recent fills for analysis
    recent_fills: Arc<RwLock<VecDeque<AdverseSelectionEvent>>>,
    
    /// Price history for post-trade analysis
    price_history: Arc<DashMap<String, VecDeque<(DateTime<Utc>, Decimal)>>>,
    
    /// Counterparty profiles
    counterparty_profiles: Arc<DashMap<String, CounterpartyProfile>>,
    
    /// Detection thresholds
    toxicity_threshold: f64,
    flagging_threshold: f64,
    
    /// Statistics
    total_fills_analyzed: Arc<RwLock<u64>>,
    toxic_fills_detected: Arc<RwLock<u64>>,
    estimated_loss_prevented: Arc<RwLock<Decimal>>,
}

impl AdverseSelectionDetector {
    pub fn new(toxicity_threshold: f64, flagging_threshold: f64) -> Self {
        Self {
            recent_fills: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            price_history: Arc::new(DashMap::new()),
            counterparty_profiles: Arc::new(DashMap::new()),
            toxicity_threshold,
            flagging_threshold,
            total_fills_analyzed: Arc::new(RwLock::new(0)),
            toxic_fills_detected: Arc::new(RwLock::new(0)),
            estimated_loss_prevented: Arc::new(RwLock::new(Decimal::ZERO)),
        }
    }
    
    /// Record a new fill for analysis
    pub async fn record_fill(&self, fill: &Fill, symbol: &str, side: OrderSide, counterparty: Option<String>) {
        let now = Utc::now();
        
        // Get mid price at fill time
        let _mid_price = self.get_mid_price(_symbol, now);
        
        let event = AdverseSelectionEvent {
            timestamp: now,
            order_id: fill.order_id,
            symbol: symbol.to_string(),
            side,
            fill_price: fill.price,
            mid_price_at_fill: mid_price,
            mid_price_after_100ms: None,
            mid_price_after_1s: None,
            mid_price_after_10s: None,
            toxicity_score: 0.0,
            counterparty: counterparty.clone(),
        };
        
        // Store for later analysis
        self.recent_fills.write().push_back(event.clone());
        
        // Schedule post-trade analysis
        let detector = Arc::new(self.clone());
        let _event_clone = event.clone();
        
        // Analyze after 100ms
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            detector.analyze_short_term(&event_clone).await;
        });
        
        // Analyze after 1s
        let detector = Arc::new(self.clone());
        let _event_clone = event.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            detector.analyze_medium_term(&event_clone).await;
        });
        
        // Analyze after 10s
        let detector = Arc::new(self.clone());
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            detector.analyze_long_term(&event).await;
        });
        
        *self.total_fills_analyzed.write() += 1;
    }
    
    /// Update price for a symbol
    pub fn update_price(&self, symbol: &str, price: Decimal) {
        let mut history = self.price_history.entry(symbol.to_string())
            .or_insert_with(|| VecDeque::with_capacity(1000));
        
        history.push_back((Utc::now(), price));
        
        // Keep only last 30 seconds
        let cutoff = Utc::now() - Duration::seconds(30);
        while let Some((_ts, _)) = history.front() {
            if *ts < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get mid price at a specific time
    fn get_mid_price(&self, symbol: &str, timestamp: DateTime<Utc>) -> Decimal {
        if let Some(history) = self.price_history.get(symbol) {
            // Find closest price to timestamp
            for (_ts, price) in history.iter().rev() {
                if *ts <= timestamp {
                    return *price;
                }
            }
            
            // If no earlier price, use first available
            if let Some((__, price)) = history.front() {
                return *price;
            }
        }
        
        Decimal::ZERO
    }
    
    /// Analyze short-term adverse selection (100ms)
    async fn analyze_short_term(&self, event: &AdverseSelectionEvent) {
        let _current_price = self.get_mid_price(&event.symbol, Utc::now());
        
        if current_price == Decimal::ZERO {
            return;
        }
        
        // Calculate adverse move in basis points
        let _adverse_move_bps = match event.side {
            OrderSide::Buy => {
                // For buys, price moving down is adverse
                ((event.fill_price - current_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
            OrderSide::Sell => {
                // For sells, price moving up is adverse
                ((current_price - event.fill_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
        };
        
        // HFT picking off detection (>5 bps in 100ms is suspicious)
        if adverse_move_bps > 5 {
            warn!(
                "Potential HFT adverse selection: {} moved {} bps in 100ms after {} fill",
                event.symbol, adverse_move_bps, event.side
            );
            
            // Update counterparty profile if known
            if let Some(cp) = &event.counterparty {
                self.update_counterparty_profile(_cp, true, adverse_move_bps);
            }
        }
    }
    
    /// Analyze medium-term adverse selection (1s)
    async fn analyze_medium_term(&self, event: &AdverseSelectionEvent) {
        let _current_price = self.get_mid_price(&event.symbol, Utc::now());
        
        if current_price == Decimal::ZERO {
            return;
        }
        
        let _adverse_move_bps = match event.side {
            OrderSide::Buy => {
                ((event.fill_price - current_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
            OrderSide::Sell => {
                ((current_price - event.fill_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
        };
        
        // Momentum trading detection (>10 bps in 1s)
        if adverse_move_bps > 10 {
            warn!(
                "Momentum adverse selection: {} moved {} bps in 1s after {} fill",
                event.symbol, adverse_move_bps, event.side
            );
            
            *self.toxic_fills_detected.write() += 1;
            
            // Estimate loss prevented by early detection
            let _loss_prevented = event.fill_price * dec!(0.001); // 0.1% estimated
            *self.estimated_loss_prevented.write() += loss_prevented;
        }
    }
    
    /// Analyze long-term adverse selection (10s)
    async fn analyze_long_term(&self, event: &AdverseSelectionEvent) {
        let _current_price = self.get_mid_price(&event.symbol, Utc::now());
        
        if current_price == Decimal::ZERO {
            return;
        }
        
        let _adverse_move_bps = match event.side {
            OrderSide::Buy => {
                ((event.fill_price - current_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
            OrderSide::Sell => {
                ((current_price - event.fill_price) / event.fill_price * dec!(10000))
                    .round_dp(0)
                    .to_i32()
                    .unwrap_or(0)
            }
        };
        
        // Information asymmetry detection (>20 bps in 10s)
        if adverse_move_bps > 20 {
            error!(
                "SEVERE adverse selection: {} moved {} bps in 10s after {} fill",
                event.symbol, adverse_move_bps, event.side
            );
            
            // Flag counterparty for review
            if let Some(cp) = &event.counterparty {
                self.flag_toxic_counterparty(cp);
            }
        }
        
        // Calculate final toxicity score
        let _toxicity_score = (adverse_move_bps as f64 / 100.0).min(1.0).max(0.0);
        
        // Update event with final analysis
        if let Some(fills) = self.recent_fills.write().iter_mut()
            .find(|e| e.order_id == event.order_id) {
            fills.mid_price_after_10s = Some(current_price);
            fills.toxicity_score = toxicity_score;
        }
    }
    
    /// Update counterparty profile
    fn update_counterparty_profile(&self, identifier: &str, is_toxic: bool, adverse_bps: i32) {
        let mut profile = self.counterparty_profiles
            .entry(identifier.to_string())
            .or_insert_with(|| CounterpartyProfile {
                identifier: identifier.to_string(),
                ..Default::default()
            });
        
        profile.total_trades += 1;
        if is_toxic {
            profile.toxic_trades += 1;
        }
        
        profile.toxicity_ratio = profile.toxic_trades as f64 / profile.total_trades as f64;
        
        // Update average adverse move (exponential moving average)
        if profile.avg_adverse_move_bps == 0 {
            profile.avg_adverse_move_bps = adverse_bps;
        } else {
            profile.avg_adverse_move_bps = 
                (profile.avg_adverse_move_bps * 9 + adverse_bps) / 10;
        }
        
        profile.last_trade = Some(Utc::now());
        
        // Auto-flag if toxicity exceeds threshold
        if profile.toxicity_ratio > self.flagging_threshold {
            profile.is_flagged = true;
            error!(
                "Counterparty {} flagged for toxic flow ({}% toxic)",
                identifier, (profile.toxicity_ratio * 100.0).round()
            );
        }
    }
    
    /// Flag a counterparty as toxic
    fn flag_toxic_counterparty(&self, identifier: &str) {
        if let Some(mut profile) = self.counterparty_profiles.get_mut(identifier) {
            profile.is_flagged = true;
            error!("TOXIC COUNTERPARTY FLAGGED: {}", identifier);
        }
    }
    
    /// Check if a counterparty is flagged
    pub fn is_counterparty_toxic(&self, identifier: &str) -> bool {
        self.counterparty_profiles
            .get(identifier)
            .map(|p| p.is_flagged)
            .unwrap_or(false)
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> AdverseSelectionStats {
        let _total_analyzed = *self.total_fills_analyzed.read();
        let _toxic_detected = *self.toxic_fills_detected.read();
        
        AdverseSelectionStats {
            total_fills_analyzed: total_analyzed,
            toxic_fills_detected: toxic_detected,
            toxicity_rate: if total_analyzed > 0 {
                toxic_detected as f64 / total_analyzed as f64
            } else {
                0.0
            },
            estimated_loss_prevented: *self.estimated_loss_prevented.read(),
            flagged_counterparties: self.counterparty_profiles
                .iter()
                .filter(|p| p.is_flagged)
                .count(),
        }
    }
    
    /// Get toxic counterparties list
    pub fn get_toxic_counterparties(&self) -> Vec<CounterpartyProfile> {
        self.counterparty_profiles
            .iter()
            .filter(|p| p.is_flagged)
            .map(|p| p.value().clone())
            .collect()
    }
}

// Implement Clone manually due to Arc fields
impl Clone for AdverseSelectionDetector {
    fn clone(&self) -> Self {
        Self {
            recent_fills: Arc::clone(&self.recent_fills),
            price_history: Arc::clone(&self.price_history),
            counterparty_profiles: Arc::clone(&self.counterparty_profiles),
            toxicity_threshold: self.toxicity_threshold,
            flagging_threshold: self.flagging_threshold,
            total_fills_analyzed: Arc::clone(&self.total_fills_analyzed),
            toxic_fills_detected: Arc::clone(&self.toxic_fills_detected),
            estimated_loss_prevented: Arc::clone(&self.estimated_loss_prevented),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AdverseSelectionStats {
    pub total_fills_analyzed: u64,
    pub toxic_fills_detected: u64,
    pub toxicity_rate: f64,
    pub estimated_loss_prevented: Decimal,
    pub flagged_counterparties: usize,
}

// ============================================================================
// TESTS - Casey & Quinn validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adverse_selection_detection() {
        let detector = AdverseSelectionDetector::new(0.5, 0.7);
        
        // Simulate a fill
        let _fill = Fill {
            order_id: OrderId::new(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            price: dec!(50000),
            quantity: dec!(1.0),
            timestamp: Utc::now(),
            exchange: "binance".to_string(),
            fee: dec!(10),
        };
        
        // Record fill
        detector.record_fill(&fill, Some("HFT_FIRM_1".to_string())).await;
        
        // Simulate adverse price movement
        detector.update_price("BTCUSDT", dec!(49950)); // 10 bps down
        
        // Wait for analysis
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        
        // Check statistics
        let _stats = detector.get_statistics();
        assert_eq!(stats.total_fills_analyzed, 1);
    }
    
    #[test]
    fn test_counterparty_flagging() {
        let detector = AdverseSelectionDetector::new(0.5, 0.7);
        
        // Simulate multiple toxic trades
        for i in 0..10 {
            detector.update_counterparty_profile("TOXIC_TRADER", i > 3, 15);
        }
        
        // Should be flagged
        assert!(detector.is_counterparty_toxic("TOXIC_TRADER"));
    }
}