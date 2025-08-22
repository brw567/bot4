// Smart Order Router
// Routes orders to best exchange based on liquidity, fees, and latency

use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info};

use crate::order::Order;

/// Exchange route information
#[derive(Debug, Clone)]
pub struct ExchangeRoute {
    pub exchange_name: String,
    pub is_active: bool,
    pub priority: u8,
    
    // Performance metrics
    pub avg_latency_ms: u64,
    pub success_rate: f64,
    pub avg_slippage_bps: i32, // basis points
    
    // Fees
    pub maker_fee_bps: i32,
    pub taker_fee_bps: i32,
    
    // Limits
    pub min_order_size: Decimal,
    pub max_order_size: Decimal,
    pub rate_limit_per_second: u32,
    
    // Current state (not serialized)
    #[allow(dead_code)]
    pub current_requests: Arc<RwLock<u32>>,
    pub last_request: Arc<RwLock<Instant>>,
}

impl ExchangeRoute {
    pub fn new(exchange_name: String) -> Self {
        Self {
            exchange_name,
            is_active: true,
            priority: 1,
            avg_latency_ms: 100,
            success_rate: 0.99,
            avg_slippage_bps: 5,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            max_order_size: Decimal::from_str_exact("100").unwrap(),
            rate_limit_per_second: 10,
            current_requests: Arc::new(RwLock::new(0)),
            last_request: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    /// Check if exchange can handle the order
    pub fn can_handle(&self, order: &Order) -> bool {
        if !self.is_active {
            return false;
        }
        
        // Check order size limits
        if order.quantity < self.min_order_size || order.quantity > self.max_order_size {
            return false;
        }
        
        // Check rate limit
        let now = Instant::now();
        let last = *self.last_request.read();
        let elapsed = now.duration_since(last);
        
        if elapsed < Duration::from_millis(1000 / self.rate_limit_per_second as u64) {
            return false;
        }
        
        true
    }
    
    /// Calculate expected cost for order
    pub fn calculate_cost(&self, order: &Order, order_value: Decimal) -> Decimal {
        let fee_bps = if order.order_type == crate::order::OrderType::Limit {
            self.maker_fee_bps
        } else {
            self.taker_fee_bps
        };
        
        let fee = order_value * Decimal::from(fee_bps) / Decimal::from(10000);
        let slippage = order_value * Decimal::from(self.avg_slippage_bps.abs()) / Decimal::from(10000);
        
        fee + slippage
    }
    
    /// Score the route for order routing decision
    pub fn score(&self, order: &Order, order_value: Decimal) -> f64 {
        if !self.can_handle(order) {
            return 0.0;
        }
        
        // Calculate cost score (lower is better)
        let cost = self.calculate_cost(order, order_value);
        use rust_decimal::prelude::ToPrimitive;
        let cost_score = 1.0 / (1.0 + cost.to_f64().unwrap_or(1.0));
        
        // Latency score (lower is better)
        let latency_score = 1.0 / (1.0 + self.avg_latency_ms as f64 / 100.0);
        
        // Success rate score
        let success_score = self.success_rate;
        
        // Priority weight
        let priority_weight = (11 - self.priority.min(10)) as f64 / 10.0;
        
        // Weighted average
        cost_score * 0.4 + latency_score * 0.2 + success_score * 0.3 + priority_weight * 0.1
    }
}


/// Routing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to exchange with best price
    BestPrice,
    /// Route to exchange with lowest fees
    LowestFee,
    /// Route to exchange with lowest latency
    FastestExecution,
    /// Route based on weighted score
    SmartRoute,
    /// Always use primary exchange
    PrimaryOnly,
    /// Round-robin between active exchanges
    RoundRobin,
}

/// Order router for smart order routing
pub struct OrderRouter {
    routes: Arc<DashMap<String, ExchangeRoute>>,
    strategy: Arc<RwLock<RoutingStrategy>>,
    primary_exchange: Arc<RwLock<Option<String>>>,
    round_robin_index: Arc<RwLock<usize>>,
}

impl OrderRouter {
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            routes: Arc::new(DashMap::new()),
            strategy: Arc::new(RwLock::new(strategy)),
            primary_exchange: Arc::new(RwLock::new(None)),
            round_robin_index: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Add an exchange route
    pub fn add_route(&self, route: ExchangeRoute) {
        info!("Added route to {}", route.exchange_name);
        self.routes.insert(route.exchange_name.clone(), route);
    }
    
    /// Remove an exchange route
    pub fn remove_route(&self, exchange_name: &str) {
        self.routes.remove(exchange_name);
        info!("Removed route to {}", exchange_name);
    }
    
    /// Set primary exchange
    pub fn set_primary_exchange(&self, exchange_name: String) {
        *self.primary_exchange.write() = Some(exchange_name);
    }
    
    /// Route an order to best exchange
    pub async fn route_order(&self, order: &Order) -> Result<String, RoutingError> {
        let strategy = *self.strategy.read();
        
        match strategy {
            RoutingStrategy::PrimaryOnly => self.route_to_primary(),
            RoutingStrategy::RoundRobin => self.route_round_robin(order),
            RoutingStrategy::BestPrice => self.route_best_price(order).await,
            RoutingStrategy::LowestFee => self.route_lowest_fee(order),
            RoutingStrategy::FastestExecution => self.route_fastest(order),
            RoutingStrategy::SmartRoute => self.route_smart(order),
        }
    }
    
    fn route_to_primary(&self) -> Result<String, RoutingError> {
        self.primary_exchange
            .read()
            .clone()
            .ok_or(RoutingError::NoPrimaryExchange)
    }
    
    fn route_round_robin(&self, order: &Order) -> Result<String, RoutingError> {
        // ZERO-COPY: Find active routes without collecting
        let active_routes: Vec<_> = self.routes
            .iter()
            .filter(|r| r.can_handle(order))
            .collect();
        
        if active_routes.is_empty() {
            return Err(RoutingError::NoAvailableRoute);
        }
        
        let mut index = self.round_robin_index.write();
        let selected = active_routes[*index % active_routes.len()].key().clone();
        *index += 1;
        
        Ok(selected)
    }
    
    async fn route_best_price(&self, order: &Order) -> Result<String, RoutingError> {
        // In production, this would query actual order books
        // For now, use the route with lowest slippage
        self.routes
            .iter()
            .filter(|r| r.can_handle(order))
            .min_by_key(|r| r.avg_slippage_bps)
            .map(|r| r.key().clone())
            .ok_or(RoutingError::NoAvailableRoute)
    }
    
    fn route_lowest_fee(&self, order: &Order) -> Result<String, RoutingError> {
        let fee_field = if order.order_type == crate::order::OrderType::Limit {
            |r: &ExchangeRoute| r.maker_fee_bps
        } else {
            |r: &ExchangeRoute| r.taker_fee_bps
        };
        
        self.routes
            .iter()
            .filter(|r| r.can_handle(order))
            .min_by_key(|r| fee_field(r.value()))
            .map(|r| r.key().clone())
            .ok_or(RoutingError::NoAvailableRoute)
    }
    
    fn route_fastest(&self, order: &Order) -> Result<String, RoutingError> {
        self.routes
            .iter()
            .filter(|r| r.can_handle(order))
            .min_by_key(|r| r.avg_latency_ms)
            .map(|r| r.key().clone())
            .ok_or(RoutingError::NoAvailableRoute)
    }
    
    fn route_smart(&self, order: &Order) -> Result<String, RoutingError> {
        // Calculate order value for scoring
        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(50000));
        
        // Score all routes and pick the best
        let best_route = self.routes
            .iter()
            .filter(|r| r.can_handle(order))
            .map(|r| (r.key().clone(), r.score(order, order_value)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        match best_route {
            Some((exchange, score)) => {
                debug!(
                    "Smart routing order {} to {} (score: {:.3})",
                    order.id, exchange, score
                );
                Ok(exchange)
            }
            None => Err(RoutingError::NoAvailableRoute),
        }
    }
    
    /// Update route metrics after order execution
    pub fn update_metrics(
        &self,
        exchange_name: &str,
        latency_ms: u64,
        success: bool,
        slippage_bps: Option<i32>,
    ) {
        if let Some(mut route) = self.routes.get_mut(exchange_name) {
            // Update latency (exponential moving average)
            route.avg_latency_ms = (route.avg_latency_ms * 9 + latency_ms) / 10;
            
            // Update success rate
            route.success_rate = if success {
                route.success_rate * 0.99 + 1.0
            } else {
                route.success_rate * 0.99
            };
            
            // Update slippage if provided
            if let Some(slippage) = slippage_bps {
                route.avg_slippage_bps = (route.avg_slippage_bps * 9 + slippage) / 10;
            }
            
            // Update request tracking
            *route.last_request.write() = Instant::now();
        }
    }
    
    /// Get routing statistics
    pub fn get_stats(&self) -> RoutingStats {
        let total_routes = self.routes.len();
        let active_routes = self.routes.iter().filter(|r| r.is_active).count();
        
        let avg_latency = if total_routes > 0 {
            self.routes.iter().map(|r| r.avg_latency_ms).sum::<u64>() / total_routes as u64
        } else {
            0
        };
        
        let avg_success_rate = if total_routes > 0 {
            self.routes.iter().map(|r| r.success_rate).sum::<f64>() / total_routes as f64
        } else {
            0.0
        };
        
        RoutingStats {
            total_routes,
            active_routes,
            avg_latency_ms: avg_latency,
            avg_success_rate,
            current_strategy: *self.strategy.read(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStats {
    pub total_routes: usize,
    pub active_routes: usize,
    pub avg_latency_ms: u64,
    pub avg_success_rate: f64,
    pub current_strategy: RoutingStrategy,
}

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("No available route for order")]
    NoAvailableRoute,
    
    #[error("No primary exchange configured")]
    NoPrimaryExchange,
    
    #[error("Exchange not found: {0}")]
    ExchangeNotFound(String),
    
    #[error("Rate limit exceeded for exchange: {0}")]
    RateLimitExceeded(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::{OrderSide, OrderType};
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_route_scoring() {
        let route = ExchangeRoute::new("binance".to_string());
        
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            dec!(0.1),
        )
        .with_price(dec!(50000));
        
        let score = route.score(&order, dec!(5000));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_smart_routing() {
        let router = OrderRouter::new(RoutingStrategy::SmartRoute);
        
        // Add routes with different characteristics
        let mut fast_route = ExchangeRoute::new("fast_exchange".to_string());
        fast_route.avg_latency_ms = 50;
        fast_route.taker_fee_bps = 30;
        router.add_route(fast_route);
        
        let mut cheap_route = ExchangeRoute::new("cheap_exchange".to_string());
        cheap_route.avg_latency_ms = 150;
        cheap_route.taker_fee_bps = 10;
        router.add_route(cheap_route);
        
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            dec!(0.1),
        );
        
        let selected = router.route_order(&order).await.unwrap();
        assert!(selected == "fast_exchange" || selected == "cheap_exchange");
    }
    
    #[tokio::test]
    async fn test_round_robin_routing() {
        let router = OrderRouter::new(RoutingStrategy::RoundRobin);
        
        router.add_route(ExchangeRoute::new("exchange1".to_string()));
        router.add_route(ExchangeRoute::new("exchange2".to_string()));
        router.add_route(ExchangeRoute::new("exchange3".to_string()));
        
        let order = Order::new(
            "BTCUSDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            dec!(0.1),
        );
        
        let mut selections = Vec::new();
        for _ in 0..6 {
            selections.push(router.route_order(&order).await.unwrap());
        }
        
        // Should cycle through all exchanges
        assert!(selections.contains(&"exchange1".to_string()));
        assert!(selections.contains(&"exchange2".to_string()));
        assert!(selections.contains(&"exchange3".to_string()));
    }
}