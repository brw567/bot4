//! Partial Fill Management System
//! Team: Full 8-Agent ULTRATHINK Collaboration
//! Research Applied: Market microstructure, Kyle (1985), Almgren-Chriss (2001)
//! Purpose: Handle realistic order execution with partial fills

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Order fill status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FillStatus {
    /// Order submitted but not yet acknowledged
    Pending,
    /// Order acknowledged by exchange
    Open,
    /// Order partially filled
    PartiallyFilled {
        filled_qty: Decimal,
        remaining_qty: Decimal,
        avg_price: Decimal,
        last_fill_time: DateTime<Utc>,
    },
    /// Order completely filled
    Filled {
        total_qty: Decimal,
        avg_price: Decimal,
        completion_time: DateTime<Utc>,
    },
    /// Order cancelled with potential partial fill
    Cancelled {
        filled_qty: Decimal,
        remaining_qty: Decimal,
        avg_price: Option<Decimal>,
    },
    /// Order rejected by exchange
    Rejected {
        reason: String,
    },
}

/// Individual fill record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRecord {
    pub fill_id: String,
    pub order_id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub fee: Decimal,
    pub timestamp: DateTime<Utc>,
    pub liquidity_type: LiquidityType,  // Maker or Taker
    pub trade_id: String,  // Exchange trade ID
}

/// Liquidity type for fee calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityType {
    Maker,  // Provided liquidity (lower fees)
    Taker,  // Removed liquidity (higher fees)
}

/// Order side
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Enhanced order tracking with partial fill support
#[derive(Debug, Clone)]
pub struct OrderTracker {
    pub order_id: String,
    pub client_order_id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub original_quantity: Decimal,
    pub limit_price: Option<Decimal>,
    pub status: FillStatus,
    pub fills: Vec<FillRecord>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    // Execution metrics
    pub total_filled: Decimal,
    pub weighted_avg_price: Decimal,
    pub total_fees: Decimal,
    pub slippage: Decimal,  // Difference from expected price
    
    // Time metrics
    pub first_fill_time: Option<DateTime<Utc>>,
    pub last_fill_time: Option<DateTime<Utc>>,
    pub time_to_fill_ms: Option<u64>,
    
    // Market impact estimation (Kyle's Lambda)
    pub estimated_impact: Decimal,
    pub realized_impact: Decimal,
}

/// Order type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
    IcebergOrder {
        visible_quantity: Decimal,
        total_quantity: Decimal,
    },
    TWAP {
        duration_ms: u64,
        slice_count: usize,
    },
    VWAP {
        duration_ms: u64,
        participation_rate: Decimal,
    },
}

/// Partial Fill Manager - Core system for handling partial executions
pub struct PartialFillManager {
    /// Active orders being tracked
    active_orders: Arc<RwLock<HashMap<String, OrderTracker>>>,
    
    /// Historical fills for analysis
    fill_history: Arc<RwLock<Vec<FillRecord>>>,
    
    /// Exchange-specific fee structures
    fee_schedules: Arc<RwLock<HashMap<String, FeeSchedule>>>,
    
    /// Market impact model
    impact_model: Arc<MarketImpactModel>,
    
    /// Execution analytics
    analytics: Arc<RwLock<ExecutionAnalytics>>,
}

impl PartialFillManager {
    /// Create new partial fill manager
    pub fn new() -> Self {
        Self {
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            fill_history: Arc::new(RwLock::new(Vec::new())),
            fee_schedules: Arc::new(RwLock::new(Self::default_fee_schedules())),
            impact_model: Arc::new(MarketImpactModel::new()),
            analytics: Arc::new(RwLock::new(ExecutionAnalytics::new())),
        }
    }
    
    /// Process incoming fill from exchange
    pub async fn process_fill(&self, fill: FillRecord) -> Result<(), String> {
        let mut orders = self.active_orders.write().await;
        
        // Find the order
        let order = orders.get_mut(&fill.order_id)
            .ok_or_else(|| format!("Order {} not found", fill.order_id))?;
        
        // Update order with new fill
        order.fills.push(fill.clone());
        order.total_filled += fill.quantity;
        order.total_fees += fill.fee;
        
        // Update weighted average price
        if order.total_filled > Decimal::ZERO {
            let weight = fill.quantity / order.total_filled;
            order.weighted_avg_price = order.weighted_avg_price * (Decimal::ONE - weight) 
                + fill.price * weight;
        }
        
        // Update timing
        if order.first_fill_time.is_none() {
            order.first_fill_time = Some(fill.timestamp);
        }
        order.last_fill_time = Some(fill.timestamp);
        
        // Update status
        if order.total_filled >= order.original_quantity {
            order.status = FillStatus::Filled {
                total_qty: order.total_filled,
                avg_price: order.weighted_avg_price,
                completion_time: fill.timestamp,
            };
            
            // Calculate time to fill
            if let Some(first_fill) = order.first_fill_time {
                order.time_to_fill_ms = Some(
                    (fill.timestamp - first_fill).num_milliseconds() as u64
                );
            }
        } else {
            order.status = FillStatus::PartiallyFilled {
                filled_qty: order.total_filled,
                remaining_qty: order.original_quantity - order.total_filled,
                avg_price: order.weighted_avg_price,
                last_fill_time: fill.timestamp,
            };
        }
        
        // Calculate slippage
        if let Some(limit_price) = order.limit_price {
            order.slippage = match order.side {
                OrderSide::Buy => order.weighted_avg_price - limit_price,
                OrderSide::Sell => limit_price - order.weighted_avg_price,
            };
        }
        
        // Update market impact
        order.realized_impact = self.impact_model.calculate_realized_impact(
            &order.fills,
            order.side.clone(),
        ).await;
        
        // Store in history
        self.fill_history.write().await.push(fill);
        
        // Update analytics
        self.analytics.write().await.update(order);
        
        Ok(())
    }
    
    /// Submit new order for tracking
    pub async fn submit_order(&self, order: OrderRequest) -> Result<String, String> {
        let order_id = format!("ORD_{}", uuid::Uuid::new_v4());
        let now = Utc::now();
        
        // Estimate market impact
        let estimated_impact = self.impact_model.estimate_impact(
            order.quantity,
            order.symbol.clone(),
            order.side.clone(),
        ).await;
        
        let tracker = OrderTracker {
            order_id: order_id.clone(),
            client_order_id: order.client_order_id,
            exchange: order.exchange,
            symbol: order.symbol,
            side: order.side,
            order_type: order.order_type,
            original_quantity: order.quantity,
            limit_price: order.limit_price,
            status: FillStatus::Pending,
            fills: Vec::new(),
            created_at: now,
            updated_at: now,
            total_filled: Decimal::ZERO,
            weighted_avg_price: Decimal::ZERO,
            total_fees: Decimal::ZERO,
            slippage: Decimal::ZERO,
            first_fill_time: None,
            last_fill_time: None,
            time_to_fill_ms: None,
            estimated_impact,
            realized_impact: Decimal::ZERO,
        };
        
        self.active_orders.write().await.insert(order_id.clone(), tracker);
        
        Ok(order_id)
    }
    
    /// Cancel order (may have partial fills)
    pub async fn cancel_order(&self, order_id: &str) -> Result<FillStatus, String> {
        let mut orders = self.active_orders.write().await;
        
        let order = orders.get_mut(order_id)
            .ok_or_else(|| format!("Order {} not found", order_id))?;
        
        // Update status to cancelled
        let avg_price = if order.total_filled > Decimal::ZERO {
            Some(order.weighted_avg_price)
        } else {
            None
        };
        
        order.status = FillStatus::Cancelled {
            filled_qty: order.total_filled,
            remaining_qty: order.original_quantity - order.total_filled,
            avg_price,
        };
        
        Ok(order.status.clone())
    }
    
    /// Get order status with all fill details
    pub async fn get_order_status(&self, order_id: &str) -> Option<OrderTracker> {
        self.active_orders.read().await.get(order_id).cloned()
    }
    
    /// Calculate optimal slice size for large orders (Almgren-Chriss)
    pub async fn calculate_optimal_slicing(
        &self,
        total_quantity: Decimal,
        symbol: String,
        urgency: f64,  // 0 = patient, 1 = urgent
    ) -> Vec<Decimal> {
        // Almgren-Chriss optimal execution trajectory
        let risk_aversion = 1.0 - urgency;
        let time_horizon = 3600.0; // 1 hour default
        let volatility = 0.02; // 2% hourly volatility estimate
        
        // Number of slices based on urgency
        let num_slices = ((1.0 - urgency) * 20.0 + 5.0) as usize;
        
        let mut slices = Vec::new();
        let base_size = total_quantity / Decimal::from(num_slices);
        
        for i in 0..num_slices {
            // Exponential decay for patient execution
            let time_factor = (i as f64 / num_slices as f64);
            let decay = (-risk_aversion * time_factor).exp();
            let slice_size = base_size * Decimal::from_f64(decay).unwrap_or(Decimal::ONE);
            slices.push(slice_size);
        }
        
        // Normalize to ensure sum equals total
        let sum: Decimal = slices.iter().sum();
        if sum > Decimal::ZERO {
            for slice in &mut slices {
                *slice = *slice * total_quantity / sum;
            }
        }
        
        slices
    }
    
    /// Default fee schedules for major exchanges
    fn default_fee_schedules() -> HashMap<String, FeeSchedule> {
        let mut schedules = HashMap::new();
        
        schedules.insert("binance".to_string(), FeeSchedule {
            maker_fee: Decimal::from_str("0.001").unwrap(),  // 0.1%
            taker_fee: Decimal::from_str("0.001").unwrap(),  // 0.1%
            volume_discounts: vec![
                (Decimal::from(1000000), Decimal::from_str("0.0009").unwrap()),
                (Decimal::from(5000000), Decimal::from_str("0.0008").unwrap()),
            ],
        });
        
        schedules.insert("coinbase".to_string(), FeeSchedule {
            maker_fee: Decimal::from_str("0.004").unwrap(),  // 0.4%
            taker_fee: Decimal::from_str("0.006").unwrap(),  // 0.6%
            volume_discounts: vec![
                (Decimal::from(10000000), Decimal::from_str("0.0035").unwrap()),
                (Decimal::from(50000000), Decimal::from_str("0.0025").unwrap()),
            ],
        });
        
        schedules
    }
    
    /// Get execution analytics
    pub async fn get_analytics(&self) -> ExecutionAnalytics {
        self.analytics.read().await.clone()
    }
}

/// Fee schedule for an exchange
#[derive(Debug, Clone)]
pub struct FeeSchedule {
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
    pub volume_discounts: Vec<(Decimal, Decimal)>,  // (volume_threshold, fee_rate)
}

/// Market impact model (Kyle's Lambda + Almgren-Chriss)
pub struct MarketImpactModel {
    /// Historical impact coefficients by symbol
    lambda_estimates: Arc<RwLock<HashMap<String, f64>>>,
}

impl MarketImpactModel {
    pub fn new() -> Self {
        Self {
            lambda_estimates: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Estimate market impact before execution
    pub async fn estimate_impact(
        &self,
        quantity: Decimal,
        symbol: String,
        side: OrderSide,
    ) -> Decimal {
        // Kyle's lambda model: Impact = λ * √(Volume)
        let lambda = self.lambda_estimates.read().await
            .get(&symbol)
            .copied()
            .unwrap_or(0.001);  // Default 10 bps per √(100% ADV)
        
        let volume_fraction = quantity.to_f64().unwrap_or(0.0) / 1000000.0;  // Normalize
        let impact = lambda * volume_fraction.sqrt();
        
        Decimal::from_f64(impact).unwrap_or(Decimal::ZERO)
    }
    
    /// Calculate realized impact from fills
    pub async fn calculate_realized_impact(
        &self,
        fills: &[FillRecord],
        side: OrderSide,
    ) -> Decimal {
        if fills.is_empty() {
            return Decimal::ZERO;
        }
        
        // Calculate VWAP
        let total_value: Decimal = fills.iter()
            .map(|f| f.price * f.quantity)
            .sum();
        let total_quantity: Decimal = fills.iter()
            .map(|f| f.quantity)
            .sum();
        
        if total_quantity == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        let vwap = total_value / total_quantity;
        
        // Compare to initial price
        let initial_price = fills.first().unwrap().price;
        
        match side {
            OrderSide::Buy => (vwap - initial_price) / initial_price,
            OrderSide::Sell => (initial_price - vwap) / initial_price,
        }
    }
}

/// Execution analytics
#[derive(Debug, Clone)]
pub struct ExecutionAnalytics {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub partial_fills: usize,
    pub cancelled_orders: usize,
    pub avg_fill_rate: f64,
    pub avg_time_to_fill_ms: f64,
    pub total_slippage: Decimal,
    pub total_fees: Decimal,
    pub avg_fills_per_order: f64,
}

impl ExecutionAnalytics {
    pub fn new() -> Self {
        Self {
            total_orders: 0,
            filled_orders: 0,
            partial_fills: 0,
            cancelled_orders: 0,
            avg_fill_rate: 0.0,
            avg_time_to_fill_ms: 0.0,
            total_slippage: Decimal::ZERO,
            total_fees: Decimal::ZERO,
            avg_fills_per_order: 0.0,
        }
    }
    
    pub fn update(&mut self, order: &OrderTracker) {
        self.total_orders += 1;
        
        match &order.status {
            FillStatus::Filled { .. } => self.filled_orders += 1,
            FillStatus::PartiallyFilled { .. } => self.partial_fills += 1,
            FillStatus::Cancelled { .. } => self.cancelled_orders += 1,
            _ => {}
        }
        
        self.total_slippage += order.slippage.abs();
        self.total_fees += order.total_fees;
        
        if !order.fills.is_empty() {
            self.avg_fills_per_order = 
                (self.avg_fills_per_order * (self.total_orders - 1) as f64 
                + order.fills.len() as f64) / self.total_orders as f64;
        }
        
        if let Some(time_ms) = order.time_to_fill_ms {
            self.avg_time_to_fill_ms = 
                (self.avg_time_to_fill_ms * (self.filled_orders - 1) as f64 
                + time_ms as f64) / self.filled_orders as f64;
        }
        
        // Calculate fill rate
        let fill_rate = (order.total_filled / order.original_quantity)
            .to_f64()
            .unwrap_or(0.0);
        self.avg_fill_rate = 
            (self.avg_fill_rate * (self.total_orders - 1) as f64 + fill_rate) 
            / self.total_orders as f64;
    }
}

/// Order request for submission
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub client_order_id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub limit_price: Option<Decimal>,
}

// External dependencies placeholder
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> Self { Self }
    }
    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "uuid")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_partial_fill_tracking() {
        let manager = PartialFillManager::new();
        
        // Submit order
        let order_req = OrderRequest {
            client_order_id: "CLIENT_001".to_string(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: Decimal::from(10),
            limit_price: Some(Decimal::from(50000)),
        };
        
        let order_id = manager.submit_order(order_req).await.unwrap();
        
        // Process partial fill
        let fill1 = FillRecord {
            fill_id: "FILL_001".to_string(),
            order_id: order_id.clone(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(3),
            price: Decimal::from(49995),
            fee: Decimal::from_str("14.9985").unwrap(),
            timestamp: Utc::now(),
            liquidity_type: LiquidityType::Taker,
            trade_id: "TRADE_001".to_string(),
        };
        
        manager.process_fill(fill1).await.unwrap();
        
        // Check status
        let status = manager.get_order_status(&order_id).await.unwrap();
        match status.status {
            FillStatus::PartiallyFilled { filled_qty, remaining_qty, .. } => {
                assert_eq!(filled_qty, Decimal::from(3));
                assert_eq!(remaining_qty, Decimal::from(7));
            }
            _ => panic!("Expected PartiallyFilled status"),
        }
        
        // Process second fill
        let fill2 = FillRecord {
            fill_id: "FILL_002".to_string(),
            order_id: order_id.clone(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(7),
            price: Decimal::from(49998),
            fee: Decimal::from_str("34.9986").unwrap(),
            timestamp: Utc::now(),
            liquidity_type: LiquidityType::Maker,
            trade_id: "TRADE_002".to_string(),
        };
        
        manager.process_fill(fill2).await.unwrap();
        
        // Check filled status
        let status = manager.get_order_status(&order_id).await.unwrap();
        match status.status {
            FillStatus::Filled { total_qty, avg_price, .. } => {
                assert_eq!(total_qty, Decimal::from(10));
                // VWAP = (3*49995 + 7*49998) / 10 = 49997.1
                assert!((avg_price - Decimal::from_str("49997.1").unwrap()).abs() 
                    < Decimal::from_str("0.1").unwrap());
            }
            _ => panic!("Expected Filled status"),
        }
    }
    
    #[tokio::test]
    async fn test_optimal_slicing() {
        let manager = PartialFillManager::new();
        
        // Test patient execution (low urgency)
        let slices = manager.calculate_optimal_slicing(
            Decimal::from(1000),
            "BTC/USDT".to_string(),
            0.2,  // Low urgency
        ).await;
        
        assert!(slices.len() > 15);  // Should have many slices
        
        // Test urgent execution
        let urgent_slices = manager.calculate_optimal_slicing(
            Decimal::from(1000),
            "BTC/USDT".to_string(),
            0.9,  // High urgency
        ).await;
        
        assert!(urgent_slices.len() < 10);  // Should have fewer slices
        
        // Verify sum equals total
        let sum: Decimal = slices.iter().sum();
        assert!((sum - Decimal::from(1000)).abs() < Decimal::from_str("0.01").unwrap());
    }
}

// Team: Full 8-Agent ULTRATHINK Collaboration
// Alex: "Partial fills are THE REALITY of trading!"
// Jordan: "Track EVERY fill for perfect reconciliation!"
// Morgan: "Feed this data to ML for execution optimization!"
// Taylor: "Slippage analysis drives strategy refinement!"
// Riley: "Time-to-fill metrics reveal market conditions!"
// Casey: "Fee optimization through maker/taker analysis!"
// Drew: "Market impact minimization through smart slicing!"
// Avery: "ZERO trades lost to poor execution!"