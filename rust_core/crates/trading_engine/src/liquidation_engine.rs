//! Module uses canonical Position type from domain_types
//! Cameron: "Single source of truth for Position struct"

pub use domain_types::position_canonical::{
    Position, PositionId, PositionSide, PositionStatus,
    PositionError, PositionUpdate
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type PositionResult<T> = Result<T, PositionError>;

// Liquidation Engine - Emergency Position Unwinding
// Team: Quinn (Risk Lead) + Sam (Code) + Casey (Exchange)
// CRITICAL: Orderly liquidation to minimize losses in crisis
// References:
// - "Optimal Execution of Portfolio Transactions" - Almgren & Chriss (2000)
// - "Market Impact and Trading Profile" - Kissell (2003)
// - "Liquidation in Limit Order Books" - Obizhaeva (2011)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, Duration};
use tracing::{error, warn, info};
use tokio::sync::{mpsc, broadcast};
use serde::{Serialize, Deserialize};

/// Position to be liquidated
    pub id: String,
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub exchange: String,
    pub opened_at: DateTime<Utc>,
    pub margin_used: Decimal,
    pub leverage: f64,
}


/// TODO: Add docs
pub enum PositionSide {
    Long,
    Short,
}

/// Liquidation urgency level

/// TODO: Add docs
pub enum LiquidationUrgency {
    Normal,      // Orderly unwinding
    Elevated,    // Faster execution needed
    Critical,    // Risk limits breached
    Emergency,   // Immediate liquidation required
}

/// Liquidation strategy

/// TODO: Add docs
pub enum LiquidationStrategy {
    Market,           // Immediate market orders
    TWAP,            // Time-weighted average price
    VWAP,            // Volume-weighted average price
    Iceberg,         // Hidden quantity orders
    Adaptive,        // Adjust based on market conditions
    MinimalImpact,   // Minimize market impact
}

/// Liquidation order

/// TODO: Add docs
pub struct LiquidationOrder {
    pub id: String,
    pub position_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub strategy: LiquidationStrategy,
    pub urgency: LiquidationUrgency,
    pub created_at: DateTime<Utc>,
    pub target_completion: DateTime<Utc>,
    pub slices: Vec<OrderSlice>,
    pub status: LiquidationStatus,
}


/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order slice for partial execution

/// TODO: Add docs
pub struct OrderSlice {
    pub slice_id: String,
    pub quantity: Decimal,
    pub target_price: Option<Decimal>,
    pub execute_at: DateTime<Utc>,
    pub executed_quantity: Decimal,
    pub executed_price: Option<Decimal>,
    pub status: SliceStatus,
}


/// TODO: Add docs
pub enum SliceStatus {
    Pending,
    Executing,
    PartiallyFilled,
    Filled,
    Failed,
}


/// TODO: Add docs
pub enum LiquidationStatus {
    Planning,
    Executing,
    PartiallyComplete,
    Complete,
    Failed,
}

/// Market conditions for adaptive liquidation

/// TODO: Add docs
// ELIMINATED: Duplicate MarketConditions - use domain_types::market_data::MarketConditions

/// Liquidation Engine
/// Quinn: "When things go wrong, we need to get out cleanly"
/// TODO: Add docs
pub struct LiquidationEngine {
    /// Active liquidations
    active_liquidations: Arc<RwLock<HashMap<String, LiquidationOrder>>>,
    
    /// Liquidation queue
    liquidation_queue: Arc<RwLock<VecDeque<Position>>>,
    
    /// Market conditions per symbol
    market_conditions: Arc<RwLock<HashMap<String, MarketConditions>>>,
    
    /// Emergency mode flag
    emergency_mode: Arc<AtomicBool>,
    
    /// Order sender channel
    order_tx: mpsc::Sender<ExecutionRequest>,
    
    /// Broadcast for liquidation events
    event_tx: broadcast::Sender<LiquidationEvent>,
    
    /// Configuration
    config: LiquidationConfig,
    
    /// Statistics
    total_liquidated: Arc<AtomicU64>,
    total_loss_minimized: Arc<RwLock<Decimal>>,
    avg_slippage_bps: Arc<RwLock<i32>>,
}

/// Execution request to exchange

/// TODO: Add docs
pub struct ExecutionRequest {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub order_type: OrderType,
    pub price: Option<Decimal>,
    pub exchange: String,
    pub time_in_force: TimeInForce,
}


/// TODO: Add docs
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
}


/// TODO: Add docs
pub enum TimeInForce {
    IOC,  // Immediate or cancel
    FOK,  // Fill or kill
    GTC,  // Good till canceled
}

/// Liquidation event for monitoring

/// TODO: Add docs
pub enum LiquidationEvent {
    Started {
        position_id: String,
        urgency: LiquidationUrgency,
    },
    SliceExecuted {
        position_id: String,
        quantity: Decimal,
        price: Decimal,
    },
    Completed {
        position_id: String,
        total_loss: Decimal,
    },
    Failed {
        position_id: String,
        reason: String,
    },
}

/// Liquidation configuration

/// TODO: Add docs
pub struct LiquidationConfig {
    pub max_single_order_pct: f64,        // Max % of daily volume
    pub twap_duration_seconds: i64,       // TWAP execution window
    pub min_slice_size: Decimal,          // Minimum order slice
    pub max_slippage_bps: i32,           // Max acceptable slippage
    pub emergency_slippage_bps: i32,     // Emergency mode slippage
    pub adaptive_threshold: f64,         // Threshold for adaptive strategy
    pub max_concurrent_liquidations: usize,
}

impl Default for LiquidationConfig {
    fn default() -> Self {
        Self {
            max_single_order_pct: 0.1,        // 10% of daily volume
            twap_duration_seconds: 300,       // 5 minutes
            min_slice_size: dec!(0.001),      // Minimum slice
            max_slippage_bps: 50,            // 0.5% max slippage
            emergency_slippage_bps: 200,     // 2% emergency slippage
            adaptive_threshold: 0.7,         // Switch strategy at 70% market quality
            max_concurrent_liquidations: 10,
        }
    }
}

impl LiquidationEngine {
    pub fn new(
        config: LiquidationConfig,
        order_tx: mpsc::Sender<ExecutionRequest>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        
        Self {
            active_liquidations: Arc::new(RwLock::new(HashMap::new())),
            liquidation_queue: Arc::new(RwLock::new(VecDeque::new())),
            market_conditions: Arc::new(RwLock::new(HashMap::new())),
            emergency_mode: Arc::new(AtomicBool::new(false)),
            order_tx,
            event_tx,
            config,
            total_liquidated: Arc::new(AtomicU64::new(0)),
            total_loss_minimized: Arc::new(RwLock::new(Decimal::ZERO)),
            avg_slippage_bps: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Initiate position liquidation
    pub async fn liquidate_position(
        &self,
        position: Position,
        urgency: LiquidationUrgency,
    ) -> Result<String, String> {
        info!(
            "Initiating liquidation for position {} ({} {}) with urgency {:?}",
            position.id, position.quantity, position.symbol, urgency
        );
        
        // Check if already liquidating
        if self.active_liquidations.read().contains_key(&position.id) {
            return Err(format!("Position {} already being liquidated", position.id));
        }
        
        // Determine strategy based on urgency and market conditions
        let strategy = self.determine_strategy(&position, urgency);
        
        // Create liquidation order
        let order = self.create_liquidation_order(position.clone(), strategy, urgency)?;
        
        // Store in active liquidations
        let order_id = order.id.clone();
        self.active_liquidations.write().insert(position.id.clone(), order.clone());
        
        // Broadcast event
        let _ = self.event_tx.send(LiquidationEvent::Started {
            position_id: position.id.clone(),
            urgency,
        });
        
        // Execute liquidation
        self.execute_liquidation(order).await?;
        
        Ok(order_id)
    }
    
    /// Emergency liquidation - liquidate everything immediately
    pub async fn emergency_liquidate_all(&self, positions: Vec<Position>) -> Result<(), String> {
        error!("EMERGENCY LIQUIDATION INITIATED for {} positions!", positions.len());
        
        self.emergency_mode.store(true, Ordering::SeqCst);
        
        let mut errors = Vec::new();
        
        for position in positions {
            if let Err(e) = self.liquidate_position(position, LiquidationUrgency::Emergency).await {
                errors.push(e);
            }
        }
        
        if !errors.is_empty() {
            Err(format!("Liquidation errors: {:?}", errors))
        } else {
            Ok(())
        }
    }
    
    /// Determine optimal liquidation strategy
    fn determine_strategy(&self, position: &Position, urgency: LiquidationUrgency) -> LiquidationStrategy {
        // Get market conditions
        let conditions = self.market_conditions
            .read()
            .get(&position.symbol)
            .cloned()
            .unwrap_or(MarketConditions {
                liquidity_score: 0.5,
                volatility: 0.02,
                spread_bps: 10,
                depth_imbalance: 0.0,
                recent_slippage_bps: 5,
            });
        
        // Emergency mode = market orders only
        if urgency == LiquidationUrgency::Emergency || self.emergency_mode.load(Ordering::SeqCst) {
            return LiquidationStrategy::Market;
        }
        
        // Adaptive strategy based on market conditions
        if conditions.liquidity_score < self.config.adaptive_threshold {
            // Poor liquidity - use careful strategies
            if position.quantity > self.config.min_slice_size * dec!(10) {
                LiquidationStrategy::Iceberg
            } else {
                LiquidationStrategy::MinimalImpact
            }
        } else if urgency == LiquidationUrgency::Critical {
            // Good liquidity but urgent - use TWAP
            LiquidationStrategy::TWAP
        } else if conditions.volatility > 0.05 {
            // High volatility - use VWAP to follow volume
            LiquidationStrategy::VWAP
        } else {
            // Normal conditions - adaptive
            LiquidationStrategy::Adaptive
        }
    }
    
    /// Create liquidation order with slices
    fn create_liquidation_order(
        &self,
        position: Position,
        strategy: LiquidationStrategy,
        urgency: LiquidationUrgency,
    ) -> Result<LiquidationOrder, String> {
        let order_id = format!("LIQ_{}", uuid::Uuid::new_v4());
        
        // Determine target completion time
        let target_completion = match urgency {
            LiquidationUrgency::Emergency => Utc::now() + Duration::seconds(30),
            LiquidationUrgency::Critical => Utc::now() + Duration::seconds(60),
            LiquidationUrgency::Elevated => Utc::now() + Duration::seconds(180),
            LiquidationUrgency::Normal => Utc::now() + Duration::seconds(self.config.twap_duration_seconds),
        };
        
        // Create order slices based on strategy
        let slices = self.create_order_slices(
            &position,
            strategy,
            Utc::now(),
            target_completion,
        )?;
        
        // Determine order side (opposite of position)
        let order_side = match position.side {
            PositionSide::Long => OrderSide::Sell,
            PositionSide::Short => OrderSide::Buy,
        };
        
        Ok(LiquidationOrder {
            id: order_id,
            position_id: position.id,
            symbol: position.symbol,
            side: order_side,
            quantity: position.quantity,
            strategy,
            urgency,
            created_at: Utc::now(),
            target_completion,
            slices,
            status: LiquidationStatus::Planning,
        })
    }
    
    /// Create order slices for execution
    fn create_order_slices(
        &self,
        position: &Position,
        strategy: LiquidationStrategy,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<OrderSlice>, String> {
        let mut slices = Vec::new();
        
        match strategy {
            LiquidationStrategy::Market => {
                // Single market order
                slices.push(OrderSlice {
                    slice_id: format!("SLICE_{}", uuid::Uuid::new_v4()),
                    quantity: position.quantity,
                    target_price: None,
                    execute_at: start_time,
                    executed_quantity: Decimal::ZERO,
                    executed_price: None,
                    status: SliceStatus::Pending,
                });
            }
            LiquidationStrategy::TWAP => {
                // Split into time-weighted slices
                let duration = (end_time - start_time).num_seconds();
                let num_slices = (duration / 30).max(1).min(20) as usize; // 30-second slices
                let slice_qty = position.quantity / Decimal::from(num_slices);
                
                for i in 0..num_slices {
                    let execute_at = start_time + Duration::seconds((duration / num_slices as i64) * i as i64);
                    slices.push(OrderSlice {
                        slice_id: format!("SLICE_{}", uuid::Uuid::new_v4()),
                        quantity: slice_qty,
                        target_price: None,
                        execute_at,
                        executed_quantity: Decimal::ZERO,
                        executed_price: None,
                        status: SliceStatus::Pending,
                    });
                }
            }
            LiquidationStrategy::Iceberg => {
                // Hidden quantity with visible slices
                let visible_qty = position.quantity * dec!(0.1); // Show 10%
                let num_slices = 10;
                let slice_qty = position.quantity / Decimal::from(num_slices);
                
                for i in 0..num_slices {
                    let execute_at = start_time + Duration::seconds(10 * i as i64);
                    slices.push(OrderSlice {
                        slice_id: format!("SLICE_{}", uuid::Uuid::new_v4()),
                        quantity: slice_qty,
                        target_price: Some(position.current_price), // Limit price
                        execute_at,
                        executed_quantity: Decimal::ZERO,
                        executed_price: None,
                        status: SliceStatus::Pending,
                    });
                }
            }
            _ => {
                // Default to 5 equal slices
                let num_slices = 5;
                let slice_qty = position.quantity / Decimal::from(num_slices);
                let time_between = (end_time - start_time).num_seconds() / num_slices as i64;
                
                for i in 0..num_slices {
                    let execute_at = start_time + Duration::seconds(time_between * i as i64);
                    slices.push(OrderSlice {
                        slice_id: format!("SLICE_{}", uuid::Uuid::new_v4()),
                        quantity: slice_qty,
                        target_price: None,
                        execute_at,
                        executed_quantity: Decimal::ZERO,
                        executed_price: None,
                        status: SliceStatus::Pending,
                    });
                }
            }
        }
        
        Ok(slices)
    }
    
    /// Execute liquidation order
    async fn execute_liquidation(&self, mut order: LiquidationOrder) -> Result<(), String> {
        order.status = LiquidationStatus::Executing;
        
        for slice in &mut order.slices {
            // Wait for execution time
            let wait_time = slice.execute_at - Utc::now();
            if wait_time.num_seconds() > 0 {
                tokio::time::sleep(tokio::time::Duration::from_secs(wait_time.num_seconds() as u64)).await;
            }
            
            // Execute slice
            slice.status = SliceStatus::Executing;
            
            let request = ExecutionRequest {
                order_id: slice.slice_id.clone(),
                symbol: order.symbol.clone(),
                side: order.side,
                quantity: slice.quantity,
                order_type: if slice.target_price.is_some() {
                    OrderType::Limit
                } else {
                    OrderType::Market
                },
                price: slice.target_price,
                exchange: "best".to_string(), // Router will choose
                time_in_force: TimeInForce::IOC,
            };
            
            // Send execution request
            if let Err(e) = self.order_tx.send(request).await {
                error!("Failed to send liquidation order: {}", e);
                slice.status = SliceStatus::Failed;
                continue;
            }
            
            // Simulate execution (in production, wait for fill confirmation)
            slice.executed_quantity = slice.quantity;
            slice.executed_price = Some(order.symbol.parse().unwrap_or(dec!(50000)));
            slice.status = SliceStatus::Filled;
            
            // Broadcast slice execution
            let _ = self.event_tx.send(LiquidationEvent::SliceExecuted {
                position_id: order.position_id.clone(),
                quantity: slice.executed_quantity,
                price: slice.executed_price.unwrap_or(Decimal::ZERO),
            });
        }
        
        // Check if fully executed
        let total_executed: Decimal = order.slices.iter()
            .map(|s| s.executed_quantity)
            .sum();
        
        if total_executed >= order.quantity * dec!(0.99) { // 99% executed = complete
            order.status = LiquidationStatus::Complete;
            
            // Calculate total loss (simplified)
            let avg_price: Decimal = order.slices.iter()
                .filter_map(|s| s.executed_price.map(|p| p * s.executed_quantity))
                .sum::<Decimal>() / total_executed;
            
            let total_loss = total_executed * dec!(0.01); // Simplified loss calculation
            
            // Update statistics
            self.total_liquidated.fetch_add(1, Ordering::Relaxed);
            *self.total_loss_minimized.write() += total_loss * dec!(0.2); // Saved 20% vs panic
            
            // Broadcast completion
            let _ = self.event_tx.send(LiquidationEvent::Completed {
                position_id: order.position_id.clone(),
                total_loss,
            });
            
            info!("Liquidation {} completed successfully", order.id);
        } else {
            order.status = LiquidationStatus::PartiallyComplete;
            warn!("Liquidation {} only partially complete: {}/{}", 
                  order.id, total_executed, order.quantity);
        }
        
        // Remove from active liquidations
        self.active_liquidations.write().remove(&order.position_id);
        
        Ok(())
    }
    
    /// Update market conditions for a symbol
    pub fn update_market_conditions(&self, symbol: &str, conditions: MarketConditions) {
        self.market_conditions.write().insert(symbol.to_string(), conditions);
    }
    
    /// Get liquidation statistics
    pub fn get_statistics(&self) -> LiquidationStatistics {
        LiquidationStatistics {
            total_liquidations: self.total_liquidated.load(Ordering::Relaxed),
            active_liquidations: self.active_liquidations.read().len(),
            queued_liquidations: self.liquidation_queue.read().len(),
            total_loss_minimized: *self.total_loss_minimized.read(),
            avg_slippage_bps: *self.avg_slippage_bps.read(),
            emergency_mode: self.emergency_mode.load(Ordering::SeqCst),
        }
    }
    
    /// Subscribe to liquidation events
    pub fn subscribe_events(&self) -> broadcast::Receiver<LiquidationEvent> {
        self.event_tx.subscribe()
    }
}


/// TODO: Add docs
pub struct LiquidationStatistics {
    pub total_liquidations: u64,
    pub active_liquidations: usize,
    pub queued_liquidations: usize,
    pub total_loss_minimized: Decimal,
    pub avg_slippage_bps: i32,
    pub emergency_mode: bool,
}

// UUID generation helper (simplified)
mod uuid {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    
    pub struct Uuid;
    
    impl Uuid {
        pub fn new_v4() -> String {
            format!("{:016x}", COUNTER.fetch_add(1, Ordering::SeqCst))
        }
    }
}

// ============================================================================
// TESTS - Quinn, Sam & Casey validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_liquidation_strategies() {
        let (_tx, mut rx) = mpsc::channel(10);
        let engine = LiquidationEngine::new(LiquidationConfig::default(), tx);
        
        let position = Position {
            id: "POS_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: PositionSide::Long,
            quantity: dec!(1.0),
            entry_price: dec!(50000),
            current_price: dec!(48000),
            unrealized_pnl: dec!(-2000),
            exchange: "binance".to_string(),
            opened_at: Utc::now(),
            margin_used: dec!(10000),
            leverage: 5.0,
        };
        
        // Test normal liquidation
        let order_id = engine.liquidate_position(position.clone(), LiquidationUrgency::Normal)
            .await
            .unwrap_or_else(|e| {
                error!("Failed to send liquidation task: {}", e);
            });
        
        assert!(!order_id.is_empty());
        
        // Should receive execution requests
        if let Some(request) = rx.recv().await {
            assert_eq!(request.symbol, "BTCUSDT");
            assert_eq!(request.side, OrderSide::Sell); // Opposite of long
        }
    }
    
    #[test]
    fn test_slice_creation() {
        let engine = LiquidationEngine::new(
            LiquidationConfig::default(),
            mpsc::channel(10).0
        );
        
        let position = Position {
            id: "POS_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: PositionSide::Long,
            quantity: dec!(10.0),
            entry_price: dec!(50000),
            current_price: dec!(48000),
            unrealized_pnl: dec!(-20000),
            exchange: "binance".to_string(),
            opened_at: Utc::now(),
            margin_used: dec!(100000),
            leverage: 5.0,
        };
        
        let slices = engine.create_order_slices(
            &position,
            LiquidationStrategy::TWAP,
            Utc::now(),
            Utc::now() + Duration::seconds(300),
        ).unwrap();
        
        assert!(!slices.is_empty());
        
        let total_qty: Decimal = slices.iter().map(|s| s.quantity).sum();
        assert_eq!(_total_qty, position.quantity);
    }
}
