// Risk Monitoring and Metrics
// Real-time tracking of portfolio risk metrics

use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn, error};

/// Risk metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    // P&L metrics
    pub total_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub weekly_pnl: Decimal,
    pub monthly_pnl: Decimal,
    
    // Exposure metrics
    pub total_exposure: Decimal,
    pub long_exposure: Decimal,
    pub short_exposure: Decimal,
    pub net_exposure: Decimal,
    
    // Risk metrics
    pub current_drawdown: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub win_rate: f64,
    
    // Position metrics
    pub open_positions: usize,
    pub avg_position_size: Decimal,
    pub largest_position: Decimal,
    
    // Limits usage
    pub daily_loss_used_pct: Decimal,
    pub exposure_used_pct: Decimal,
    pub drawdown_used_pct: Decimal,
    
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            total_pnl: Decimal::ZERO,
            daily_pnl: Decimal::ZERO,
            weekly_pnl: Decimal::ZERO,
            monthly_pnl: Decimal::ZERO,
            total_exposure: Decimal::ZERO,
            long_exposure: Decimal::ZERO,
            short_exposure: Decimal::ZERO,
            net_exposure: Decimal::ZERO,
            current_drawdown: Decimal::ZERO,
            max_drawdown: Decimal::ZERO,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            win_rate: 0.0,
            open_positions: 0,
            avg_position_size: Decimal::ZERO,
            largest_position: Decimal::ZERO,
            daily_loss_used_pct: Decimal::ZERO,
            exposure_used_pct: Decimal::ZERO,
            drawdown_used_pct: Decimal::ZERO,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Drawdown tracker
pub struct DrawdownTracker {
    peak_value: Arc<RwLock<Decimal>>,
    current_value: Arc<RwLock<Decimal>>,
    max_drawdown: Arc<RwLock<Decimal>>,
    current_drawdown: Arc<RwLock<Decimal>>,
    drawdown_start: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    drawdown_history: Arc<RwLock<VecDeque<DrawdownEvent>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownEvent {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub peak_value: Decimal,
    pub trough_value: Decimal,
    pub drawdown_pct: Decimal,
    pub duration_hours: Option<f64>,
}

impl DrawdownTracker {
    pub fn new(initial_value: Decimal) -> Self {
        Self {
            peak_value: Arc::new(RwLock::new(initial_value)),
            current_value: Arc::new(RwLock::new(initial_value)),
            max_drawdown: Arc::new(RwLock::new(Decimal::ZERO)),
            current_drawdown: Arc::new(RwLock::new(Decimal::ZERO)),
            drawdown_start: Arc::new(RwLock::new(None)),
            drawdown_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }
    
    /// Update portfolio value and track drawdown
    pub fn update_value(&self, new_value: Decimal) {
        let mut current = self.current_value.write();
        let mut peak = self.peak_value.write();
        
        *current = new_value;
        
        // Update peak if new high
        if new_value > *peak {
            // Drawdown ended
            if let Some(start_time) = *self.drawdown_start.read() {
                let event = DrawdownEvent {
                    start_time,
                    end_time: Some(chrono::Utc::now()),
                    peak_value: *peak,
                    trough_value: *current,
                    drawdown_pct: *self.current_drawdown.read(),
                    duration_hours: Some(
                        chrono::Utc::now()
                            .signed_duration_since(start_time)
                            .num_seconds() as f64 / 3600.0
                    ),
                };
                
                self.drawdown_history.write().push_back(event);
                if self.drawdown_history.read().len() > 100 {
                    self.drawdown_history.write().pop_front();
                }
            }
            
            *peak = new_value;
            *self.drawdown_start.write() = None;
            *self.current_drawdown.write() = Decimal::ZERO;
        } else {
            // Calculate drawdown
            let drawdown = (*peak - new_value) / *peak * dec!(100);
            *self.current_drawdown.write() = drawdown;
            
            // Track max drawdown
            if drawdown > *self.max_drawdown.read() {
                *self.max_drawdown.write() = drawdown;
                
                // Alert on significant drawdown
                if drawdown > dec!(10) {
                    error!("SIGNIFICANT DRAWDOWN: {}%", drawdown);
                }
            }
            
            // Mark drawdown start
            if self.drawdown_start.read().is_none() {
                *self.drawdown_start.write() = Some(chrono::Utc::now());
            }
        }
    }
    
    pub fn get_current_drawdown(&self) -> Decimal {
        *self.current_drawdown.read()
    }
    
    pub fn get_max_drawdown(&self) -> Decimal {
        *self.max_drawdown.read()
    }
    
    pub fn is_in_drawdown(&self) -> bool {
        self.drawdown_start.read().is_some()
    }
    
    pub fn get_drawdown_duration(&self) -> Option<chrono::Duration> {
        self.drawdown_start.read().map(|start| {
            chrono::Utc::now().signed_duration_since(start)
        })
    }
}

/// Main risk monitor
pub struct RiskMonitor {
    metrics: Arc<RwLock<RiskMetrics>>,
    drawdown_tracker: Arc<DrawdownTracker>,
    position_pnls: Arc<DashMap<String, Decimal>>,
    trades_won: Arc<AtomicU64>,
    trades_lost: Arc<AtomicU64>,
    alerts: Arc<RwLock<Vec<RiskAlert>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    pub severity: AlertSeverity,
    pub message: String,
    pub metric: String,
    pub value: Decimal,
    pub threshold: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl RiskMonitor {
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(RiskMetrics::default())),
            drawdown_tracker: Arc::new(DrawdownTracker::new(initial_capital)),
            position_pnls: Arc::new(DashMap::new()),
            trades_won: Arc::new(AtomicU64::new(0)),
            trades_lost: Arc::new(AtomicU64::new(0)),
            alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Update position P&L
    pub fn update_position_pnl(&self, symbol: String, pnl: Decimal) {
        self.position_pnls.insert(symbol, pnl);
        self.recalculate_metrics();
    }
    
    /// Record trade result
    pub fn record_trade(&self, pnl: Decimal) {
        if pnl > Decimal::ZERO {
            self.trades_won.fetch_add(1, Ordering::Relaxed);
        } else if pnl < Decimal::ZERO {
            self.trades_lost.fetch_add(1, Ordering::Relaxed);
        }
        self.recalculate_metrics();
    }
    
    /// Recalculate all metrics
    fn recalculate_metrics(&self) {
        let mut metrics = self.metrics.write();
        
        // Calculate total P&L
        metrics.total_pnl = self.position_pnls
            .iter()
            .map(|entry| *entry.value())
            .sum();
        
        // Update drawdown
        let portfolio_value = dec!(100000) + metrics.total_pnl; // Assuming 100k initial
        self.drawdown_tracker.update_value(portfolio_value);
        metrics.current_drawdown = self.drawdown_tracker.get_current_drawdown();
        metrics.max_drawdown = self.drawdown_tracker.get_max_drawdown();
        
        // Calculate win rate
        let wins = self.trades_won.load(Ordering::Relaxed) as f64;
        let losses = self.trades_lost.load(Ordering::Relaxed) as f64;
        let total = wins + losses;
        metrics.win_rate = if total > 0.0 { wins / total } else { 0.0 };
        
        metrics.timestamp = chrono::Utc::now();
    }
    
    /// Check risk thresholds and generate alerts
    pub fn check_thresholds(&self, limits: &crate::limits::RiskLimits) {
        let metrics = self.metrics.read();
        let mut alerts = self.alerts.write();
        
        // Check daily loss
        if let Some(daily_limit) = limits.loss_limits.daily_loss_limit {
            if metrics.daily_pnl < -daily_limit {
                alerts.push(RiskAlert {
                    severity: AlertSeverity::Critical,
                    message: "Daily loss limit exceeded!".to_string(),
                    metric: "daily_pnl".to_string(),
                    value: metrics.daily_pnl,
                    threshold: -daily_limit,
                    timestamp: chrono::Utc::now(),
                });
                error!("DAILY LOSS LIMIT BREACHED: {} / {}", metrics.daily_pnl, daily_limit);
            }
        }
        
        // Check drawdown
        let max_dd_pct = limits.loss_limits.max_drawdown_pct * dec!(100);
        if metrics.current_drawdown > max_dd_pct {
            alerts.push(RiskAlert {
                severity: AlertSeverity::Emergency,
                message: "Maximum drawdown exceeded!".to_string(),
                metric: "drawdown".to_string(),
                value: metrics.current_drawdown,
                threshold: max_dd_pct,
                timestamp: chrono::Utc::now(),
            });
            error!("MAX DRAWDOWN BREACHED: {}% / {}%", metrics.current_drawdown, max_dd_pct);
        }
        
        // Check exposure
        if metrics.total_exposure > limits.position_limits.max_total_exposure {
            alerts.push(RiskAlert {
                severity: AlertSeverity::Warning,
                message: "Total exposure limit exceeded".to_string(),
                metric: "exposure".to_string(),
                value: metrics.total_exposure,
                threshold: limits.position_limits.max_total_exposure,
                timestamp: chrono::Utc::now(),
            });
        }
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> RiskMetrics {
        self.metrics.read().clone()
    }
    
    /// Get recent alerts
    pub fn get_alerts(&self, severity_filter: Option<AlertSeverity>) -> Vec<RiskAlert> {
        let alerts = self.alerts.read();
        if let Some(severity) = severity_filter {
            alerts.iter()
                .filter(|a| a.severity == severity)
                .cloned()
                .collect()
        } else {
            alerts.clone()
        }
    }
    
    /// Clear old alerts
    pub fn clear_old_alerts(&self, hours: i64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(hours);
        self.alerts.write().retain(|a| a.timestamp > cutoff);
    }
    
    /// Get risk report
    pub fn generate_report(&self) -> RiskReport {
        let metrics = self.get_metrics();
        let alerts = self.get_alerts(None);
        
        RiskReport {
            metrics,
            alerts,
            drawdown_active: self.drawdown_tracker.is_in_drawdown(),
            drawdown_duration: self.drawdown_tracker.get_drawdown_duration(),
            generated_at: chrono::Utc::now(),
        }
    }
}

/// Risk report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReport {
    pub metrics: RiskMetrics,
    pub alerts: Vec<RiskAlert>,
    pub drawdown_active: bool,
    pub drawdown_duration: Option<chrono::Duration>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl RiskReport {
    /// Get risk score (0-100, higher is riskier)
    pub fn risk_score(&self) -> u8 {
        let mut score = 0u8;
        
        // Drawdown contribution (0-40 points)
        let dd_score = (self.metrics.current_drawdown / dec!(0.15) * dec!(40))
            .to_u8()
            .unwrap_or(40)
            .min(40);
        score += dd_score;
        
        // Loss contribution (0-30 points)
        if self.metrics.daily_pnl < Decimal::ZERO {
            let loss_score = ((-self.metrics.daily_pnl / dec!(2000)) * dec!(30))
                .to_u8()
                .unwrap_or(30)
                .min(30);
            score += loss_score;
        }
        
        // Win rate contribution (0-20 points)
        let wr_score = ((1.0 - self.metrics.win_rate) * 20.0) as u8;
        score += wr_score;
        
        // Alert contribution (0-10 points)
        let critical_alerts = self.alerts.iter()
            .filter(|a| matches!(a.severity, AlertSeverity::Critical | AlertSeverity::Emergency))
            .count() as u8;
        score += (critical_alerts * 2).min(10);
        
        score.min(100)
    }
}

use rust_decimal::prelude::ToPrimitive;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_drawdown_tracking() {
        let tracker = DrawdownTracker::new(dec!(100000));
        
        // No drawdown initially
        assert_eq!(tracker.get_current_drawdown(), Decimal::ZERO);
        assert!(!tracker.is_in_drawdown());
        
        // Enter drawdown
        tracker.update_value(dec!(95000));
        assert_eq!(tracker.get_current_drawdown(), dec!(5)); // 5%
        assert!(tracker.is_in_drawdown());
        
        // Deeper drawdown
        tracker.update_value(dec!(90000));
        assert_eq!(tracker.get_current_drawdown(), dec!(10)); // 10%
        assert_eq!(tracker.get_max_drawdown(), dec!(10));
        
        // Recovery
        tracker.update_value(dec!(100000));
        assert_eq!(tracker.get_current_drawdown(), Decimal::ZERO);
        assert!(!tracker.is_in_drawdown());
        assert_eq!(tracker.get_max_drawdown(), dec!(10)); // Max preserved
    }
    
    #[test]
    fn test_risk_monitor() {
        let monitor = RiskMonitor::new(dec!(100000));
        
        // Update some positions
        monitor.update_position_pnl("BTC".to_string(), dec!(1000));
        monitor.update_position_pnl("ETH".to_string(), dec!(-500));
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_pnl, dec!(500));
        
        // Record trades
        monitor.record_trade(dec!(100));
        monitor.record_trade(dec!(-50));
        monitor.record_trade(dec!(75));
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.win_rate, 2.0 / 3.0); // 2 wins, 1 loss
    }
}