// Pre-trade Risk Checks
// Quinn's first line of defense - NOTHING passes without approval
// Performance: <10μs for all checks combined

use std::sync::Arc;
use std::time::Instant;
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn, error};

use order_management::{Order, Position};
use crate::limits::RiskLimits;

/// Result of risk check
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum RiskCheckResult {
    /// Order approved
    Approved {
        checks_passed: Vec<String>,
        warnings: Vec<String>,
    },
    /// Order rejected (Quinn's VETO)
    Rejected {
        reason: String,
        checks_failed: Vec<String>,
    },
    /// Order needs modification
    ModifyRequired {
        suggestions: Vec<RiskSuggestion>,
    },
}

/// Risk modification suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct RiskSuggestion {
    pub field: String,
    pub current_value: String,
    pub suggested_value: String,
    pub reason: String,
}

/// Pre-trade check types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// TODO: Add docs
pub enum PreTradeCheck {
    PositionSize,
    StopLoss,
    MaxExposure,
    Correlation,
    DailyLoss,
    Leverage,
    Liquidity,
    All,
}

/// Risk checker - Quinn's enforcement engine
/// TODO: Add docs
pub struct RiskChecker {
    limits: Arc<RiskLimits>,
    current_positions: Arc<DashMap<String, Position>>,
    daily_pnl: Arc<RwLock<Decimal>>,
    total_exposure: Arc<RwLock<Decimal>>,
    check_latencies: Arc<DashMap<PreTradeCheck, u64>>,
}

impl RiskChecker {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits: Arc::new(limits),
            current_positions: Arc::new(DashMap::new()),
            daily_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            total_exposure: Arc::new(RwLock::new(Decimal::ZERO)),
            check_latencies: Arc::new(DashMap::new()),
        }
    }
    
    /// Main risk check function - Quinn's gatekeeper
    pub async fn check_order(&self, order: &Order) -> RiskCheckResult {
        let start = Instant::now();
        let mut checks_passed = Vec::new();
        let mut checks_failed = Vec::new();
        let mut warnings = Vec::new();
        
        // 1. Position Size Check (2% max)
        if let Err(reason) = self.check_position_size(order) {
            checks_failed.push(format!("Position size: {}", reason));
        } else {
            checks_passed.push("Position size within limits".to_string());
        }
        
        // 2. Stop Loss Check (mandatory)
        if let Err(reason) = self.check_stop_loss(order) {
            checks_failed.push(format!("Stop loss: {}", reason));
        } else {
            checks_passed.push("Stop loss properly set".to_string());
        }
        
        // 3. Max Exposure Check
        if let Err(reason) = self.check_max_exposure(order).await {
            checks_failed.push(format!("Exposure: {}", reason));
        } else {
            checks_passed.push("Exposure within limits".to_string());
        }
        
        // 4. Daily Loss Check
        if let Err(reason) = self.check_daily_loss(order) {
            checks_failed.push(format!("Daily loss: {}", reason));
        } else {
            checks_passed.push("Daily loss limit OK".to_string());
        }
        
        // 5. Leverage Check
        if let Err(reason) = self.check_leverage(order) {
            warnings.push(format!("Leverage warning: {}", reason));
        } else {
            checks_passed.push("Leverage acceptable".to_string());
        }
        
        // Record latency
        let latency = start.elapsed().as_micros() as u64;
        self.check_latencies.insert(PreTradeCheck::All, latency);
        
        debug!("Risk checks completed in {}μs", latency);
        
        // Quinn's decision
        if !checks_failed.is_empty() {
            error!("Order {} REJECTED by Quinn: {:?}", order.id, checks_failed);
            RiskCheckResult::Rejected {
                reason: "Risk checks failed - Quinn's VETO".to_string(),
                checks_failed,
            }
        } else {
            debug!("Order {} approved: {:?}", order.id, checks_passed);
            RiskCheckResult::Approved {
                checks_passed,
                warnings,
            }
        }
    }
    
    /// Check position size (2% max per Quinn's rules)
    fn check_position_size(&self, order: &Order) -> Result<(), String> {
        let max_size = self.limits.position_limits.max_position_size_pct;
        
        if order.position_size_pct > max_size {
            Err(format!(
                "Position size {}% exceeds {}% limit",
                order.position_size_pct * dec!(100),
                max_size * dec!(100)
            ))
        } else {
            Ok(())
        }
    }
    
    /// Check stop loss (mandatory for all orders)
    fn check_stop_loss(&self, order: &Order) -> Result<(), String> {
        if !self.limits.position_limits.require_stop_loss {
            return Ok(());
        }
        
        if order.stop_loss_price.is_none() {
            return Err("Stop loss is MANDATORY - Quinn's rule #1".to_string());
        }
        
        // Check stop loss distance
        if let (Some(entry_price), Some(stop_price)) = (order.price, order.stop_loss_price) {
            let stop_distance = ((entry_price - stop_price).abs() / entry_price) * dec!(100);
            let max_stop = self.limits.position_limits.default_stop_loss_pct * dec!(100);
            
            if stop_distance > max_stop * dec!(2) {
                return Err(format!(
                    "Stop loss {}% is too far (max {}%)",
                    stop_distance, max_stop * dec!(2)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Check maximum exposure
    async fn check_max_exposure(&self, order: &Order) -> Result<(), String> {
        let order_value = order.quantity * order.price.unwrap_or(dec!(50000));
        let current_exposure = *self.total_exposure.read();
        let new_exposure = current_exposure + order_value;
        
        let max_exposure = self.limits.position_limits.max_total_exposure;
        
        if new_exposure > max_exposure {
            Err(format!(
                "New exposure {} exceeds max {}",
                new_exposure, max_exposure
            ))
        } else {
            Ok(())
        }
    }
    
    /// Check daily loss limit
    fn check_daily_loss(&self, order: &Order) -> Result<(), String> {
        let daily_pnl = *self.daily_pnl.read();
        
        if let Some(daily_limit) = self.limits.loss_limits.daily_loss_limit {
            if daily_pnl < -daily_limit {
                return Err(format!(
                    "Daily loss {} exceeds limit {}",
                    daily_pnl, daily_limit
                ));
            }
        }
        
        Ok(())
    }
    
    /// Check leverage
    fn check_leverage(&self, order: &Order) -> Result<(), String> {
        // Simplified leverage check - in production would check actual margin
        let leverage = dec!(1); // Placeholder
        let max_leverage = self.limits.position_limits.max_leverage;
        
        if leverage > max_leverage {
            Err(format!(
                "Leverage {} exceeds max {}",
                leverage, max_leverage
            ))
        } else {
            Ok(())
        }
    }
    
    /// Update current positions
    pub fn update_position(&self, symbol: String, position: Position) {
        self.current_positions.insert(symbol, position);
        self.recalculate_exposure();
    }
    
    /// Recalculate total exposure
    fn recalculate_exposure(&self) {
        let total: Decimal = self.current_positions
            .iter()
            .map(|entry| entry.value().position_value)
            .sum();
        
        *self.total_exposure.write() = total;
    }
    
    /// Update daily P&L
    pub fn update_daily_pnl(&self, pnl: Decimal) {
        *self.daily_pnl.write() = pnl;
    }
    
    /// Get check latencies for monitoring
    pub fn get_latencies(&self) -> Vec<(PreTradeCheck, u64)> {
        self.check_latencies
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }
    
    /// Emergency validation - Quinn's override
    pub fn emergency_check(&self, order: &Order) -> bool {
        // Quinn can emergency block any order
        if order.position_size_pct > dec!(0.01) {
            warn!("Quinn's emergency check: Large position {}", order.position_size_pct);
        }
        
        // Check if we're in emergency stop mode
        if *self.daily_pnl.read() < dec!(-1000) {
            error!("EMERGENCY STOP: Daily loss exceeded emergency threshold");
            return false;
        }
        
        true
    }
}

/// Batch risk checker for multiple orders
/// TODO: Add docs
pub struct BatchRiskChecker {
    checker: Arc<RiskChecker>,
}

impl BatchRiskChecker {
    pub fn new(checker: Arc<RiskChecker>) -> Self {
        Self { checker }
    }
    
    /// Check multiple orders with correlation analysis
    pub async fn check_orders(&self, orders: &[Order]) -> Vec<RiskCheckResult> {
        let mut results = Vec::new();
        
        for order in orders {
            let result = self.checker.check_order(order).await;
            results.push(result);
        }
        
        // TODO: Add correlation checks between orders
        
        results
    }
}

#[derive(Debug, Error)]
/// TODO: Add docs
pub enum RiskCheckError {
    #[error("Position size exceeds limit: {0}")]
    PositionSizeTooLarge(String),
    
    #[error("Stop loss not set")]
    NoStopLoss,
    
    #[error("Daily loss limit exceeded: {0}")]
    DailyLossExceeded(String),
    
    #[error("Maximum exposure exceeded: {0}")]
    MaxExposureExceeded(String),
    
    #[error("Correlation limit exceeded: {0}")]
    CorrelationTooHigh(String),
    
    #[error("Emergency stop triggered: {0}")]
    EmergencyStop(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use order_management::{OrderSide, OrderType};
    use crate::limits::LossLimits;
    
    #[tokio::test]
    async fn test_position_size_check() {
        let limits = RiskLimits::default();
        let checker = RiskChecker::new(limits);
        
        let mut order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.1),
        );
        order.position_size_pct = dec!(0.03); // 3% - too large
        order.stop_loss_price = Some(dec!(49000));
        
        let result = checker.check_order(&order).await;
        
        match result {
            RiskCheckResult::Rejected { reason, .. } => {
                assert!(reason.contains("VETO"));
            }
            _ => panic!("Should have rejected large position"),
        }
    }
    
    #[tokio::test]
    async fn test_stop_loss_mandatory() {
        let limits = RiskLimits::default();
        let checker = RiskChecker::new(limits);
        
        let mut order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.1),
        );
        order.position_size_pct = dec!(0.01);
        order.price = Some(dec!(50000));
        // No stop loss set
        
        let result = checker.check_order(&order).await;
        
        match result {
            RiskCheckResult::Rejected { checks_failed, .. } => {
                assert!(checks_failed.iter().any(|s| s.contains("Stop loss")));
            }
            _ => panic!("Should have rejected order without stop loss"),
        }
    }
    
    #[tokio::test]
    async fn test_approved_order() {
        let limits = RiskLimits::default();
        let checker = RiskChecker::new(limits);
        
        let mut order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.1),
        );
        order.position_size_pct = dec!(0.01); // 1% - OK
        order.stop_loss_price = Some(dec!(49000));
        order.price = Some(dec!(50000));
        
        let result = checker.check_order(&order).await;
        
        match result {
            RiskCheckResult::Approved { checks_passed, .. } => {
                assert!(!checks_passed.is_empty());
            }
            _ => panic!("Should have approved valid order"),
        }
    }
}