//! # Risk Layer Abstractions (Layer 2)
//!
//! Abstractions that risk layer provides for higher layers to use.
//! This fixes the violation where risk_engine was importing from order_management.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use domain_types::{Price, Quantity};
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::AbstractionResult;

/// Risk assessment for any tradeable entity
#[async_trait]
pub trait RiskAssessment: Send + Sync {
    /// Assess risk for a position
    async fn assess_position_risk(
        &self,
        symbol: &str,
        quantity: Quantity,
        entry_price: Price,
    ) -> AbstractionResult<RiskMetrics>;
    
    /// Check if trade is allowed
    async fn is_trade_allowed(
        &self,
        symbol: &str,
        quantity: Quantity,
        price: Price,
    ) -> AbstractionResult<bool>;
    
    /// Get current risk limits
    async fn get_risk_limits(&self) -> AbstractionResult<RiskLimits>;
    
    /// Calculate position size using Kelly criterion
    async fn calculate_position_size(
        &self,
        win_probability: f64,
        win_loss_ratio: f64,
        max_risk: f64,
    ) -> AbstractionResult<f64>;
}

/// Risk metrics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk
    pub var_95: Decimal,
    /// Conditional VaR
    pub cvar_95: Decimal,
    /// Maximum drawdown
    pub max_drawdown: Decimal,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Beta
    pub beta: f64,
    /// Risk score (0-100)
    pub risk_score: f64,
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size as % of portfolio
    pub max_position_size_pct: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Maximum drawdown allowed
    pub max_drawdown_pct: f64,
    /// Maximum correlation between positions
    pub max_correlation: f64,
    /// Daily loss limit
    pub daily_loss_limit: Decimal,
}

/// Stop loss management abstraction
#[async_trait]
pub trait StopLossManager: Send + Sync {
    /// Calculate stop loss price
    async fn calculate_stop_loss(
        &self,
        entry_price: Price,
        atr: f64,
        risk_pct: f64,
    ) -> AbstractionResult<Price>;
    
    /// Update trailing stop
    async fn update_trailing_stop(
        &self,
        current_price: Price,
        current_stop: Price,
        trail_pct: f64,
    ) -> AbstractionResult<Price>;
    
    /// Check if stop triggered
    async fn is_stop_triggered(
        &self,
        current_price: Price,
        stop_price: Price,
        is_long: bool,
    ) -> bool;
}

/// Portfolio risk management
#[async_trait]
pub trait PortfolioRiskManager: Send + Sync {
    /// Calculate portfolio VaR
    async fn calculate_portfolio_var(
        &self,
        confidence: f64,
        horizon_days: u32,
    ) -> AbstractionResult<Decimal>;
    
    /// Get correlation matrix
    async fn get_correlation_matrix(&self) -> AbstractionResult<Vec<Vec<f64>>>;
    
    /// Optimize portfolio weights
    async fn optimize_weights(
        &self,
        target_return: f64,
        risk_tolerance: f64,
    ) -> AbstractionResult<Vec<f64>>;
    
    /// Stress test portfolio
    async fn stress_test(
        &self,
        scenarios: Vec<StressScenario>,
    ) -> AbstractionResult<Vec<StressTestResult>>;
}

/// Stress test scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    /// Scenario name
    pub name: String,
    /// Market shock percentage
    pub market_shock: f64,
    /// Volatility multiplier
    pub volatility_multiplier: f64,
    /// Correlation breakdown
    pub correlation_breakdown: bool,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario name
    pub scenario: String,
    /// Portfolio loss
    pub portfolio_loss: Decimal,
    /// Worst position
    pub worst_position: String,
    /// Worst position loss
    pub worst_position_loss: Decimal,
    /// Margin call triggered
    pub margin_call: bool,
}

/// Circuit breaker for risk management
#[async_trait]
pub trait RiskCircuitBreaker: Send + Sync {
    /// Check if circuit should trip
    async fn should_trip(
        &self,
        current_loss: Decimal,
        loss_velocity: f64,
    ) -> bool;
    
    /// Trip the circuit
    async fn trip(&self, reason: String) -> AbstractionResult<()>;
    
    /// Reset the circuit
    async fn reset(&self) -> AbstractionResult<()>;
    
    /// Get cooldown period
    fn cooldown_ms(&self) -> u64;
}

/// Adverse selection detector
#[async_trait]
pub trait AdverseSelectionDetector: Send + Sync {
    /// Detect adverse selection
    async fn detect(
        &self,
        symbol: &str,
        fill_price: Price,
        mid_price: Price,
        size: Quantity,
    ) -> AbstractionResult<AdverseSelectionMetrics>;
    
    /// Get historical adverse selection
    async fn get_historical(
        &self,
        symbol: &str,
        days: u32,
    ) -> AbstractionResult<Vec<AdverseSelectionMetrics>>;
}

/// Adverse selection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdverseSelectionMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Slippage in basis points
    pub slippage_bps: f64,
    /// Price impact
    pub price_impact: f64,
    /// Toxicity score (0-100)
    pub toxicity_score: f64,
    /// Is toxic
    pub is_toxic: bool,
}