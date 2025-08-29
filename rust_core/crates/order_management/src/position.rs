//! Module uses canonical Position type from domain_types
//! Cameron: "Single source of truth for Position struct"

pub use domain_types::position_canonical::{
    Position, PositionId, PositionSide, PositionStatus,
    PositionError, PositionUpdate
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type PositionResult<T> = Result<T, PositionError>;

// Position Management and P&L Calculation
// Tracks open positions and calculates real-time P&L

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::order::{Order, OrderId, OrderSide};

/// Position identifier

impl Default for PositionId {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for PositionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Position tracking
    pub id: PositionId,
    pub symbol: String,
    pub side: OrderSide,
    
    // Quantity tracking
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    
    // P&L tracking
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_pct: Decimal,
    pub realized_pnl: Decimal,
    pub total_commission: Decimal,
    
    // Risk management
    pub stop_loss_price: Option<Decimal>,
    pub take_profit_price: Option<Decimal>,
    pub max_drawdown: Decimal,
    pub position_value: Decimal,
    
    // Metadata
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub strategy_id: Option<String>,
    
    // Order tracking
    pub opening_orders: Vec<OrderId>,
    pub closing_orders: Vec<OrderId>,
}

impl Position {
    pub fn new(symbol: String, side: OrderSide, quantity: Decimal, entry_price: Decimal) -> Self {
        let position_value = quantity * entry_price;
        Self {
            id: PositionId::new(),
            symbol,
            side,
            quantity,
            entry_price,
            current_price: entry_price,
            unrealized_pnl: Decimal::ZERO,
            unrealized_pnl_pct: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            stop_loss_price: None,
            take_profit_price: None,
            max_drawdown: Decimal::ZERO,
            position_value,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
            closed_at: None,
            strategy_id: None,
            opening_orders: Vec::new(),
            closing_orders: Vec::new(),
        }
    }
    
    /// Update position with new market price
    pub fn update_price(&mut self, new_price: Decimal) {
        self.current_price = new_price;
        self.position_value = self.quantity * new_price;
        
        // Calculate unrealized P&L
        let price_diff = match self.side {
            OrderSide::Buy => new_price - self.entry_price,
            OrderSide::Sell => self.entry_price - new_price,
        };
        
        self.unrealized_pnl = price_diff * self.quantity;
        self.unrealized_pnl_pct = if self.entry_price.is_zero() {
            Decimal::ZERO
        } else {
            (price_diff / self.entry_price) * dec!(100)
        };
        
        // Track max drawdown
        if self.unrealized_pnl < self.max_drawdown {
            self.max_drawdown = self.unrealized_pnl;
        }
        
        self.updated_at = Utc::now();
    }
    
    /// Check if position hit stop loss
    pub fn is_stop_loss_hit(&self) -> bool {
        if let Some(stop_price) = self.stop_loss_price {
            match self.side {
                OrderSide::Buy => self.current_price <= stop_price,
                OrderSide::Sell => self.current_price >= stop_price,
            }
        } else {
            false
        }
    }
    
    /// Check if position hit take profit
    pub fn is_take_profit_hit(&self) -> bool {
        if let Some(tp_price) = self.take_profit_price {
            match self.side {
                OrderSide::Buy => self.current_price >= tp_price,
                OrderSide::Sell => self.current_price <= tp_price,
            }
        } else {
            false
        }
    }
    
    /// Calculate position risk/reward ratio
    pub fn risk_reward_ratio(&self) -> Option<Decimal> {
        match (self.stop_loss_price, self.take_profit_price) {
            (Some(sl), Some(tp)) => {
                let risk = (self.entry_price - sl).abs();
                let reward = (tp - self.entry_price).abs();
                
                if risk.is_zero() {
                    None
                } else {
                    Some(reward / risk)
                }
            }
            _ => None,
        }
    }
    
    /// Check if position is profitable
    pub fn is_profitable(&self) -> bool {
        self.unrealized_pnl > Decimal::ZERO
    }
    
    /// Get total P&L including commissions
    pub fn total_pnl(&self) -> Decimal {
        self.unrealized_pnl + self.realized_pnl - self.total_commission
    }
}

/// Position manager for tracking all positions
/// TODO: Add docs
pub struct PositionManager {
    positions: Arc<DashMap<PositionId, Arc<RwLock<Position>>>>,
    symbol_positions: Arc<DashMap<String, Vec<PositionId>>>,
    pnl_calculator: Arc<PnLCalculator>,
}

impl Default for PositionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionManager {
    pub fn new() -> Self {
        Self {
            positions: Arc::new(DashMap::new()),
            symbol_positions: Arc::new(DashMap::new()),
            pnl_calculator: Arc::new(PnLCalculator::new()),
        }
    }
    
    /// Open a new position from an order
    pub fn open_position(&self, order: &Order) -> PositionId {
        let mut position = Position::new(
            order.symbol.clone(),
            order.side,
            order.filled_quantity,
            order.average_fill_price.unwrap_or(order.price.unwrap_or(Decimal::ZERO)),
        );
        
        position.stop_loss_price = order.stop_loss_price;
        position.take_profit_price = order.take_profit_price;
        position.strategy_id = order.strategy_id.clone();
        position.opening_orders.push(order.id);
        position.total_commission = order.commission;
        
        let position_id = position.id;
        
        // Store position
        self.positions.insert(position_id, Arc::new(RwLock::new(position)));
        
        // Track by symbol
        self.symbol_positions
            .entry(order.symbol.clone())
            .or_default()
            .push(position_id);
        
        info!(
            "Opened position {} for {} {} {} @ {}",
            position_id,
            order.side,
            order.filled_quantity,
            order.symbol,
            order.average_fill_price.unwrap_or_default()
        );
        
        position_id
    }
    
    /// Update existing position with additional fill
    pub fn add_to_position(
        &self,
        position_id: PositionId,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> Result<(), String> {
        if let Some(position_ref) = self.positions.get(&position_id) {
            let mut position = position_ref.write();
            
            // Calculate new average entry price
            let total_value = position.entry_price * position.quantity + price * quantity;
            let new_quantity = position.quantity + quantity;
            position.entry_price = total_value / new_quantity;
            position.quantity = new_quantity;
            position.total_commission += commission;
            position.updated_at = Utc::now();
            
            // Recalculate P&L
            let current_price = position.current_price;
            position.update_price(current_price);
            
            let new_avg_price = position.entry_price;
            info!(
                "Added to position {}: {} @ {} (new avg: {})",
                position_id, quantity, price, new_avg_price
            );
            
            Ok(())
        } else {
            Err(format!("Position {} not found", position_id))
        }
    }
    
    /// Reduce position size
    pub fn reduce_position(
        &self,
        position_id: PositionId,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> Result<Decimal, String> {
        if let Some(position_ref) = self.positions.get(&position_id) {
            let mut position = position_ref.write();
            
            if quantity > position.quantity {
                return Err(format!(
                    "Cannot reduce position by {} when size is {}",
                    quantity, position.quantity
                ));
            }
            
            // Calculate realized P&L for the reduced portion
            let price_diff = match position.side {
                OrderSide::Buy => price - position.entry_price,
                OrderSide::Sell => position.entry_price - price,
            };
            let realized = price_diff * quantity - commission;
            
            position.quantity -= quantity;
            position.realized_pnl += realized;
            position.total_commission += commission;
            position.updated_at = Utc::now();
            
            // Close position if fully reduced
            if position.quantity.is_zero() {
                position.closed_at = Some(Utc::now());
                info!("Position {} closed with P&L: {}", position_id, position.total_pnl());
            }
            
            Ok(realized)
        } else {
            Err(format!("Position {} not found", position_id))
        }
    }
    
    /// Update all positions with new market prices
    pub fn update_market_prices(&self, symbol: &str, price: Decimal) {
        if let Some(position_ids) = self.symbol_positions.get(symbol) {
            for position_id in position_ids.iter() {
                if let Some(position_ref) = self.positions.get(position_id) {
                    let mut position = position_ref.write();
                    if position.closed_at.is_none() {
                        position.update_price(price);
                        
                        // Check stop loss and take profit
                        if position.is_stop_loss_hit() {
                            warn!("Position {} hit stop loss at {}", position_id, price);
                        }
                        if position.is_take_profit_hit() {
                            info!("Position {} hit take profit at {}", position_id, price);
                        }
                    }
                }
            }
        }
    }
    
    /// Get position by ID
    pub fn get_position(&self, position_id: PositionId) -> Option<Position> {
        self.positions.get(&position_id).map(|p| p.read().clone())
    }
    
    /// Get all open positions
    pub fn get_open_positions(&self) -> Vec<Position> {
        self.positions
            .iter()
            .filter_map(|entry| {
                let position = entry.value().read();
                if position.closed_at.is_none() {
                    Some(position.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get total unrealized P&L
    pub fn total_unrealized_pnl(&self) -> Decimal {
        self.get_open_positions()
            .iter()
            .map(|p| p.unrealized_pnl)
            .sum()
    }
    
    /// Get total exposure
    pub fn total_exposure(&self) -> Decimal {
        self.get_open_positions()
            .iter()
            .map(|p| p.position_value)
            .sum()
    }
}

/// P&L Calculator for various scenarios
/// TODO: Add docs
pub struct PnLCalculator {
    commission_rate: Decimal,
}

impl Default for PnLCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PnLCalculator {
    pub fn new() -> Self {
        Self {
            commission_rate: dec!(0.001), // 0.1% default
        }
    }
    
    /// Calculate P&L for a trade
    pub fn calculate_pnl(
        &self,
        side: OrderSide,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        let price_diff = match side {
            OrderSide::Buy => exit_price - entry_price,
            OrderSide::Sell => entry_price - exit_price,
        };
        
        let gross_pnl = price_diff * quantity;
        let commission = (entry_price * quantity + exit_price * quantity) * self.commission_rate;
        
        gross_pnl - commission
    }
    
    /// Calculate break-even price
    pub fn break_even_price(
        &self,
        side: OrderSide,
        entry_price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        // BUGFIX: Commission should consider quantity for accurate break-even
        // Commission = price * quantity * rate (paid on both entry and exit)
        let total_value = entry_price * quantity;
        let commission_per_unit = (total_value * self.commission_rate * dec!(2)) / quantity;
        let commission_adjustment = commission_per_unit;
        
        match side {
            OrderSide::Buy => entry_price + commission_adjustment,
            OrderSide::Sell => entry_price - commission_adjustment,
        }
    }
    
    /// Calculate position size for risk amount
    pub fn position_size_for_risk(
        &self,
        account_balance: Decimal,
        risk_percentage: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
    ) -> Decimal {
        let risk_amount = account_balance * risk_percentage;
        let price_difference = (entry_price - stop_loss_price).abs();
        
        if price_difference.is_zero() {
            Decimal::ZERO
        } else {
            risk_amount / price_difference
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_position_pnl() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(0.1),
            dec!(50000),
        );
        
        // Price goes up
        position.update_price(dec!(51000));
        assert_eq!(position.unrealized_pnl, dec!(100));
        assert_eq!(position.unrealized_pnl_pct, dec!(2));
        assert!(position.is_profitable());
        
        // Price goes down
        position.update_price(dec!(49000));
        assert_eq!(position.unrealized_pnl, dec!(-100));
        assert_eq!(position.unrealized_pnl_pct, dec!(-2));
        assert!(!position.is_profitable());
    }
    
    #[test]
    fn test_stop_loss_take_profit() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(0.1),
            dec!(50000),
        );
        
        position.stop_loss_price = Some(dec!(49000));
        position.take_profit_price = Some(dec!(52000));
        
        // Normal price
        position.update_price(dec!(50500));
        assert!(!position.is_stop_loss_hit());
        assert!(!position.is_take_profit_hit());
        
        // Stop loss hit
        position.update_price(dec!(48999));
        assert!(position.is_stop_loss_hit());
        assert!(!position.is_take_profit_hit());
        
        // Take profit hit
        position.update_price(dec!(52001));
        assert!(!position.is_stop_loss_hit());
        assert!(position.is_take_profit_hit());
    }
    
    #[test]
    fn test_risk_reward_ratio() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(0.1),
            dec!(50000),
        );
        
        position.stop_loss_price = Some(dec!(49000));
        position.take_profit_price = Some(dec!(53000));
        
        let rr = position.risk_reward_ratio().unwrap();
        assert_eq!(rr, dec!(3)); // Risk 1000, Reward 3000
    }
}
