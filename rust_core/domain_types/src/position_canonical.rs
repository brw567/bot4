//! # CANONICAL POSITION TYPE - UNIFIED BY TEAM
//! Consolidates 10 Position struct duplicates into ONE
//! Team: All 9 agents contributed requirements
//! Date: 2025-08-28

use crate::{Price, Quantity, Symbol, Exchange};
use crate::order_enhanced::{OrderId, OrderSide};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Unique position identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositionId(pub Uuid);

impl PositionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// CANONICAL POSITION - All Requirements Unified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    // ======= IDENTITY =======
    pub id: PositionId,
    pub symbol: Symbol,
    pub exchange: Exchange,
    pub side: OrderSide,
    
    // ======= QUANTITY & PRICING =======
    pub quantity: Quantity,
    pub entry_price: Price,
    pub current_price: Price,
    pub average_entry: Price,  // For scaled entries
    
    // ======= P&L TRACKING (Cameron) =======
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_pct: Decimal,
    pub realized_pnl: Decimal,
    pub total_pnl: Decimal,
    pub commission_paid: Decimal,
    pub funding_paid: Decimal,  // Drew: "Critical for perpetuals"
    
    // ======= RISK MANAGEMENT (Cameron) =======
    pub stop_loss: Option<Price>,
    pub take_profit: Option<Price>,
    pub max_drawdown: Decimal,
    pub max_profit: Decimal,
    pub risk_score: f64,
    pub kelly_fraction: Decimal,
    pub value_at_risk: Decimal,
    pub liquidation_price: Option<Price>,  // Critical!
    
    // ======= ML/STRATEGY (Blake) =======
    pub strategy_id: Option<String>,
    pub ml_confidence: Option<f64>,
    pub predicted_move: Option<f64>,
    pub feature_snapshot: Option<Vec<f64>>,
    pub entry_signals: Vec<String>,
    
    // ======= TIME TRACKING (Ellis) =======
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub holding_period_seconds: u64,
    pub time_in_profit_pct: f64,
    
    // ======= ORDER TRACKING (Drew) =======
    pub opening_orders: Vec<OrderId>,
    pub closing_orders: Vec<OrderId>,
    pub adjustment_orders: Vec<OrderId>,
    
    // ======= SAFETY (Skyler) =======
    pub kill_switch: Arc<AtomicBool>,
    pub risk_limits_checked: bool,
    pub audit_trail: Vec<AuditEntry>,
    
    // ======= PERFORMANCE (Ellis) =======
    pub last_update_latency_ns: u64,
    pub total_updates: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub agent: String,
    pub details: String,
}

impl Position {
    /// Cameron: "Calculate current risk metrics"
    pub fn update_risk_metrics(&mut self) {
        // Update unrealized P&L
        let price_diff = match self.side {
            OrderSide::Buy => self.current_price - self.entry_price,
            OrderSide::Sell => self.entry_price - self.current_price,
        };
        
        self.unrealized_pnl = price_diff.inner() * self.quantity.inner();
        self.unrealized_pnl_pct = if self.entry_price.inner() > Decimal::ZERO {
            (self.unrealized_pnl / (self.entry_price.inner() * self.quantity.inner())) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };
        
        // Track extremes
        if self.unrealized_pnl < -self.max_drawdown {
            self.max_drawdown = -self.unrealized_pnl;
        }
        if self.unrealized_pnl > self.max_profit {
            self.max_profit = self.unrealized_pnl;
        }
        
        self.total_pnl = self.unrealized_pnl + self.realized_pnl - self.commission_paid - self.funding_paid;
    }
    
    /// Skyler: "Emergency stop check"
    pub fn should_emergency_close(&self) -> bool {
        self.kill_switch.load(Ordering::SeqCst) ||
        self.max_drawdown > Decimal::from(1000) ||  // $1000 max loss
        self.risk_score > 90.0
    }
    
    /// Blake: "ML prediction quality"
    pub fn has_strong_ml_signal(&self) -> bool {
        self.ml_confidence.unwrap_or(0.0) > 0.8
    }
    
    /// Drew: "Check if approaching liquidation"
    pub fn liquidation_risk(&self) -> f64 {
        if let Some(liq_price) = self.liquidation_price {
            let distance = ((self.current_price - liq_price).inner() / self.current_price.inner()).abs();
            1.0 - distance.to_f64().unwrap_or(0.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Quinn: "Full validation"
    pub fn validate(&self) -> Result<(), String> {
        if self.quantity.inner() <= Decimal::ZERO {
            return Err("Invalid quantity".into());
        }
        if self.entry_price.inner() <= Decimal::ZERO {
            return Err("Invalid entry price".into());
        }
        Ok(())
    }
}

// MORGAN: "Builder for testing"
pub struct PositionBuilder {
    position: Position,
}

impl PositionBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            position: Position {
                id: PositionId::new(),
                symbol: Symbol::from(symbol),
                exchange: Exchange::Binance,
                side: OrderSide::Buy,
                quantity: Quantity::zero(),
                entry_price: Price::zero(),
                current_price: Price::zero(),
                average_entry: Price::zero(),
                unrealized_pnl: Decimal::ZERO,
                unrealized_pnl_pct: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                total_pnl: Decimal::ZERO,
                commission_paid: Decimal::ZERO,
                funding_paid: Decimal::ZERO,
                stop_loss: None,
                take_profit: None,
                max_drawdown: Decimal::ZERO,
                max_profit: Decimal::ZERO,
                risk_score: 0.0,
                kelly_fraction: Decimal::from_str_exact("0.1").expect("SAFETY: Add proper error handling"),
                value_at_risk: Decimal::ZERO,
                liquidation_price: None,
                strategy_id: None,
                ml_confidence: None,
                predicted_move: None,
                feature_snapshot: None,
                entry_signals: Vec::new(),
                opened_at: Utc::now(),
                updated_at: Utc::now(),
                closed_at: None,
                holding_period_seconds: 0,
                time_in_profit_pct: 0.0,
                opening_orders: Vec::new(),
                closing_orders: Vec::new(),
                adjustment_orders: Vec::new(),
                kill_switch: Arc::new(AtomicBool::new(false)),
                risk_limits_checked: false,
                audit_trail: Vec::new(),
                last_update_latency_ns: 0,
                total_updates: 0,
            }
        }
    }
    
    pub fn quantity(mut self, qty: Quantity) -> Self {
        self.position.quantity = qty;
        self
    }
    
    pub fn entry_price(mut self, price: Price) -> Self {
        self.position.entry_price = price;
        self.position.current_price = price;
        self.position.average_entry = price;
        self
    }
    
    pub fn build(self) -> Position {
        self.position
    }
}