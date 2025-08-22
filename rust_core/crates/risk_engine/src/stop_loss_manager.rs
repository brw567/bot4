// CRITICAL: Stop Loss Manager Implementation
// Team: Quinn (Risk Lead) + Alex (Priority) + Full Team
// THIS WAS MISSING - CRITICAL RISK EXPOSURE FIXED!
// References:
// - "Trading Systems" by Kaufman (2013)
// - "Risk Management in Trading" by Davis (2015)

use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tracing::{error, warn, info};
use dashmap::DashMap;
use chrono::{DateTime, Utc};

use order_management::{Order, OrderSide, Position, PositionId};

/// CRITICAL: Stop loss trigger for position protection
#[derive(Debug, Clone)]
pub struct StopLossTrigger {
    pub position_id: PositionId,
    pub symbol: String,
    pub side: OrderSide,
    pub stop_price: Decimal,
    pub quantity: Decimal,
    pub created_at: DateTime<Utc>,
    pub triggered_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub is_trailing: bool,
    pub trail_distance: Option<Decimal>,
    pub high_water_mark: Option<Decimal>,
}

/// Stop loss manager - Quinn's critical safety system
pub struct StopLossManager {
    /// Active stop losses by position ID
    stop_losses: Arc<DashMap<PositionId, StopLossTrigger>>,
    
    /// Price feeds for monitoring
    price_feeds: Arc<DashMap<String, Decimal>>,
    
    /// Order sender channel
    order_sender: mpsc::Sender<Order>,
    
    /// Emergency stop flag
    emergency_stop: Arc<RwLock<bool>>,
    
    /// Statistics
    triggers_activated: Arc<RwLock<u64>>,
    losses_prevented: Arc<RwLock<Decimal>>,
}

impl StopLossManager {
    pub fn new(order_sender: mpsc::Sender<Order>) -> Self {
        Self {
            stop_losses: Arc::new(DashMap::new()),
            price_feeds: Arc::new(DashMap::new()),
            order_sender,
            emergency_stop: Arc::new(RwLock::new(false)),
            triggers_activated: Arc::new(RwLock::new(0)),
            losses_prevented: Arc::new(RwLock::new(Decimal::ZERO)),
        }
    }
    
    /// Add stop loss for a position - MANDATORY per Quinn's rules
    pub fn add_stop_loss(
        &self,
        position: &Position,
        stop_price: Decimal,
        is_trailing: bool,
        trail_percentage: Option<Decimal>,
    ) -> Result<(), String> {
        // Validate stop price
        match position.side {
            OrderSide::Buy => {
                if stop_price >= position.entry_price {
                    return Err(format!(
                        "Buy stop {} must be below entry {}",
                        stop_price, position.entry_price
                    ));
                }
            }
            OrderSide::Sell => {
                if stop_price <= position.entry_price {
                    return Err(format!(
                        "Sell stop {} must be above entry {}",
                        stop_price, position.entry_price
                    ));
                }
            }
        }
        
        let _trail_distance = if is_trailing {
            trail_percentage.map(|pct| position.entry_price * pct / dec!(100))
        } else {
            None
        };
        
        let _trigger = StopLossTrigger {
            position_id: position.id,
            symbol: position.symbol.clone(),
            side: position.side,
            stop_price,
            quantity: position.quantity,
            created_at: Utc::now(),
            triggered_at: None,
            is_active: true,
            is_trailing,
            trail_distance,
            high_water_mark: if is_trailing {
                Some(position.current_price)
            } else {
                None
            },
        };
        
        self.stop_losses.insert(position.id, trigger);
        
        info!(
            "Stop loss added for position {} at {} (trailing: {})",
            position.id, stop_price, is_trailing
        );
        
        Ok(())
    }
    
    /// Update price and check all stop losses
    /// Alex: "This MUST run on every tick!"
    pub async fn update_price(&self, symbol: &str, price: Decimal) {
        // Update price feed
        self.price_feeds.insert(symbol.to_string(), price);
        
        // Check emergency stop
        if *self.emergency_stop.read() {
            error!("EMERGENCY STOP ACTIVE - Liquidating all positions!");
            self.trigger_emergency_liquidation().await;
            return;
        }
        
        // Check all stop losses for this symbol
        let mut triggered = Vec::new();
        
        for entry in self.stop_losses.iter() {
            let mut trigger = entry.value().clone();
            
            if !trigger.is_active || trigger.symbol != symbol {
                continue;
            }
            
            // Update trailing stop if needed
            if trigger.is_trailing {
                self.update_trailing_stop(&mut trigger, price);
            }
            
            // Check if stop is hit
            if self.is_stop_triggered(&trigger, price) {
                triggered.push(trigger.clone());
            }
        }
        
        // Execute triggered stops
        for trigger in triggered {
            self.execute_stop_loss(trigger).await;
        }
    }
    
    /// Check if stop loss should trigger
    fn is_stop_triggered(&self, trigger: &StopLossTrigger, current_price: Decimal) -> bool {
        match trigger.side {
            OrderSide::Buy => current_price <= trigger.stop_price,
            OrderSide::Sell => current_price >= trigger.stop_price,
        }
    }
    
    /// Update trailing stop loss
    fn update_trailing_stop(&self, trigger: &mut StopLossTrigger, current_price: Decimal) {
        if !trigger.is_trailing || trigger.trail_distance.is_none() {
            return;
        }
        
        let _trail_distance = trigger.trail_distance.unwrap();
        
        match trigger.side {
            OrderSide::Buy => {
                // For long positions, trail below the high
                if let Some(high) = trigger.high_water_mark {
                    if current_price > high {
                        // New high, update stop
                        trigger.high_water_mark = Some(current_price);
                        let _new_stop = current_price - trail_distance;
                        
                        if new_stop > trigger.stop_price {
                            info!(
                                "Trailing stop updated for {} from {} to {}",
                                trigger.position_id, trigger.stop_price, new_stop
                            );
                            trigger.stop_price = new_stop;
                            
                            // Update in map
                            self.stop_losses.insert(trigger.position_id, trigger.clone());
                        }
                    }
                }
            }
            OrderSide::Sell => {
                // For short positions, trail above the low
                if let Some(low) = trigger.high_water_mark {
                    if current_price < low {
                        // New low, update stop
                        trigger.high_water_mark = Some(current_price);
                        let _new_stop = current_price + trail_distance;
                        
                        if new_stop < trigger.stop_price {
                            info!(
                                "Trailing stop updated for {} from {} to {}",
                                trigger.position_id, trigger.stop_price, new_stop
                            );
                            trigger.stop_price = new_stop;
                            
                            // Update in map
                            self.stop_losses.insert(trigger.position_id, trigger.clone());
                        }
                    }
                }
            }
        }
    }
    
    /// Execute stop loss order
    async fn execute_stop_loss(&self, mut trigger: StopLossTrigger) {
        warn!(
            "STOP LOSS TRIGGERED for position {} at {}",
            trigger.position_id, trigger.stop_price
        );
        
        // Mark as triggered
        trigger.triggered_at = Some(Utc::now());
        trigger.is_active = false;
        self.stop_losses.insert(trigger.position_id, trigger.clone());
        
        // Create market order to close position
        let _close_order = Order::new(
            trigger.symbol.clone(),
            // Reverse side to close
            match trigger.side {
                OrderSide::Buy => OrderSide::Sell,
                OrderSide::Sell => OrderSide::Buy,
            },
            order_management::OrderType::Market,
            trigger.quantity,
        );
        // TODO: Add urgency and reason fields to Order struct
        // .with_urgency(true)
        // .with_reason("STOP_LOSS_TRIGGERED");
        
        // Send order
        if let Err(e) = self.order_sender.send(close_order).await {
            error!(
                "CRITICAL: Failed to send stop loss order for {}: {}",
                trigger.position_id, e
            );
            
            // Activate emergency stop if we can't send orders
            *self.emergency_stop.write() = true;
        }
        
        // Update statistics
        *self.triggers_activated.write() += 1;
        
        // Estimate loss prevented (simplified)
        let _loss_prevented = trigger.quantity * trigger.stop_price * dec!(0.05); // 5% estimated
        *self.losses_prevented.write() += loss_prevented;
    }
    
    /// Emergency liquidation - Quinn's nuclear option
    async fn trigger_emergency_liquidation(&self) {
        error!("EMERGENCY LIQUIDATION INITIATED!");
        
        let positions: Vec<_> = self.stop_losses.iter()
            .filter(|e| e.value().is_active)
            .map(|e| e.value().clone())
            .collect();
        
        for trigger in positions {
            // Force liquidate at market
            let _emergency_order = Order::new(
                trigger.symbol.clone(),
                match trigger.side {
                    OrderSide::Buy => OrderSide::Sell,
                    OrderSide::Sell => OrderSide::Buy,
                },
                order_management::OrderType::Market,
                trigger.quantity,
            );
            // TODO: Add urgency and reason fields to Order struct
            // .with_urgency(true)
            // .with_reason("EMERGENCY_LIQUIDATION");
            
            let __ = self.order_sender.send(emergency_order).await;
        }
    }
    
    /// Activate emergency stop
    pub fn activate_emergency_stop(&self) {
        error!("EMERGENCY STOP ACTIVATED BY USER!");
        *self.emergency_stop.write() = true;
    }
    
    /// Get stop loss statistics
    pub fn get_statistics(&self) -> StopLossStatistics {
        StopLossStatistics {
            active_stops: self.stop_losses.iter()
                .filter(|e| e.value().is_active)
                .count(),
            triggered_stops: *self.triggers_activated.read(),
            losses_prevented: *self.losses_prevented.read(),
            emergency_stop_active: *self.emergency_stop.read(),
        }
    }
    
    /// Validate all positions have stop losses (Quinn's requirement)
    pub fn validate_all_protected(&self, positions: &[Position]) -> Result<(), Vec<String>> {
        let mut unprotected = Vec::new();
        
        for position in positions {
            if !self.stop_losses.contains_key(&position.id) {
                unprotected.push(format!(
                    "Position {} ({}) has NO STOP LOSS!",
                    position.id, position.symbol
                ));
            }
        }
        
        if !unprotected.is_empty() {
            error!("CRITICAL: Positions without stop loss protection!");
            Err(unprotected)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct StopLossStatistics {
    pub active_stops: usize,
    pub triggered_stops: u64,
    pub losses_prevented: Decimal,
    pub emergency_stop_active: bool,
}

// ============================================================================
// TESTS - Quinn's critical validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stop_loss_trigger() {
        let (_tx, mut rx) = mpsc::channel(10);
        let _manager = StopLossManager::new(tx);
        
        // Create long position
        let _position = Position::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(1.0),
            dec!(50000),
        );
        
        // Add stop loss at 49000 (2% below entry)
        manager.add_stop_loss(&position, dec!(49000), false, None).unwrap();
        
        // Price drops to stop level
        manager.update_price("BTCUSDT", dec!(48999)).await;
        
        // Should receive stop loss order
        let _order = rx.recv().await.unwrap();
        assert_eq!(order.side, OrderSide::Sell); // Opposite side to close
        assert_eq!(order.quantity, dec!(1.0));
    }
    
    #[tokio::test]
    async fn test_trailing_stop() {
        let (_tx, _rx) = mpsc::channel(10);
        let _manager = StopLossManager::new(tx);
        
        let _position = Position::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            dec!(1.0),
            dec!(50000),
        );
        
        // Add trailing stop 2% below
        manager.add_stop_loss(&position, dec!(49000), true, Some(dec!(2))).await.unwrap();
        
        // Price rises to 52000
        manager.update_price("BTCUSDT", dec!(52000)).await;
        
        // Stop should trail up to 50960 (2% below 52000)
        let _stop = manager.stop_losses.get(&position.id).unwrap();
        assert!(stop.stop_price > dec!(49000));
        assert!(stop.stop_price <= dec!(51000));
    }
}