//!
//! ## Design Principles
//! - Fail fast with clear error messages
//! - Composable validators
//! - Performance-aware (caching, lazy evaluation)
//! - Extensible for exchange-specific rules

use crate::{Order, OrderType, OrderSide, Price, Quantity, Trade, Candle, OrderBook};
use rust_decimal::Decimal;
use thiserror::Error;
use std::collections::HashMap;

/// Validation errors with detailed context
#[derive(Debug, Error, Clone)]
/// TODO: Add docs
pub enum ValidationError {
    #[error("Price validation failed: {0}")]
    InvalidPrice(String),
    
    #[error("Quantity validation failed: {0}")]
    InvalidQuantity(String),
    
    #[error("Order validation failed: {0}")]
    InvalidOrder(String),
    
    #[error("Trade validation failed: {0}")]
    InvalidTrade(String),
    
    #[error("Candle validation failed: {0}")]
    InvalidCandle(String),
    
    #[error("Market data validation failed: {0}")]
    InvalidMarketData(String),
    
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    
    #[error("Exchange rule violation: {0}")]
    ExchangeRuleViolation(String),
    
    #[error("Multiple validation errors: {0:?}")]
    Multiple(Vec<ValidationError>),
}

/// Validation result type
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Trait for validatable types
pub trait Validatable {
    /// Validates the type according to business rules
    fn validate(&self) -> ValidationResult<()>;
    
    /// Validates with context (exchange rules, limits, etc.)
    fn validate_with_context(&self, context: &ValidationContext) -> ValidationResult<()> {
        // Default implementation just calls basic validate
        self.validate()
    }
}

/// Validation context with exchange rules and limits
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ValidationContext {
    /// Exchange-specific rules
    pub exchange_rules: HashMap<String, ExchangeRules>,
    /// Risk limits
    pub risk_limits: RiskLimits,
    /// Market conditions
    pub market_conditions: MarketConditions,
}

/// Exchange-specific trading rules
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ExchangeRules {
    /// Exchange name
    pub exchange: String,
    /// Minimum order size by symbol
    pub min_order_size: HashMap<String, Quantity>,
    /// Maximum order size by symbol
    pub max_order_size: HashMap<String, Quantity>,
    /// Tick size by symbol
    pub tick_size: HashMap<String, Price>,
    /// Lot size by symbol
    pub lot_size: HashMap<String, Quantity>,
    /// Allowed order types
    pub allowed_order_types: Vec<OrderType>,
    /// Maximum leverage
    pub max_leverage: Decimal,
    /// Maker fee rate
    pub maker_fee: Decimal,
    /// Taker fee rate
    pub taker_fee: Decimal,
}

/// Risk management limits
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RiskLimits {
    /// Maximum position size as % of portfolio
    pub max_position_pct: Decimal,
    /// Maximum loss per trade
    pub max_loss_per_trade: Decimal,
    /// Maximum daily loss
    pub max_daily_loss: Decimal,
    /// Maximum correlation between positions
    pub max_correlation: Decimal,
    /// Required stop loss
    pub require_stop_loss: bool,
    /// Maximum leverage allowed
    pub max_leverage: Decimal,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_pct: Decimal::from_str_exact("0.02").expect("SAFETY: Add proper error handling"), // 2%
            max_loss_per_trade: Decimal::from(1000),
            max_daily_loss: Decimal::from(5000),
            max_correlation: Decimal::from_str_exact("0.7").expect("SAFETY: Add proper error handling"),
            require_stop_loss: true,
            max_leverage: Decimal::from(3),
        }
    }
}

/// Current market conditions for validation
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: Duplicate MarketConditions - use domain_types::market_data::MarketConditions

/// Main validator struct
/// TODO: Add docs
pub struct Validator {
    context: ValidationContext,
    cache: HashMap<String, ValidationResult<()>>,
}

impl Validator {
    /// Creates a new validator with context
    pub fn new(context: ValidationContext) -> Self {
        Self {
            context,
            cache: HashMap::new(),
        }
    }
    
    /// Validates an order
    pub fn validate_order(&mut self, order: &Order) -> ValidationResult<()> {
        let mut errors = Vec::new();
        
        // Check basic order fields
        if order.quantity.is_zero() {
            errors.push(ValidationError::InvalidOrder(
                "Order quantity cannot be zero".to_string()
            ));
        }
        
        // Check price for limit orders
        if order.order_type.requires_price() && order.price.is_none() {
            errors.push(ValidationError::InvalidOrder(
                format!("Order type {:?} requires price", order.order_type)
            ));
        }
        
        // Check stop price for stop orders
        if order.order_type.requires_stop_price() && order.stop_price.is_none() {
            errors.push(ValidationError::InvalidOrder(
                format!("Order type {:?} requires stop price", order.order_type)
            ));
        }
        
        // Validate stop loss and take profit
        if let Some(stop_loss) = order.stop_loss {
            if let Some(price) = order.price {
                match order.side {
                    OrderSide::Buy => {
                        if stop_loss >= price {
                            errors.push(ValidationError::InvalidOrder(
                                "Buy stop loss must be below entry price".to_string()
                            ));
                        }
                    }
                    OrderSide::Sell => {
                        if stop_loss <= price {
                            errors.push(ValidationError::InvalidOrder(
                                "Sell stop loss must be above entry price".to_string()
                            ));
                        }
                    }
                }
            }
        }
        
        // Check exchange rules if available
        if let Some(exchange) = &order.exchange {
            if let Some(rules) = self.context.exchange_rules.get(exchange) {
                // Check min/max order size
                if let Some(min_size) = rules.min_order_size.get(&order.symbol) {
                    if order.quantity < *min_size {
                        errors.push(ValidationError::ExchangeRuleViolation(
                            format!("Order quantity {} below minimum {}", order.quantity, min_size)
                        ));
                    }
                }
                
                if let Some(max_size) = rules.max_order_size.get(&order.symbol) {
                    if order.quantity > *max_size {
                        errors.push(ValidationError::ExchangeRuleViolation(
                            format!("Order quantity {} exceeds maximum {}", order.quantity, max_size)
                        ));
                    }
                }
                
                // Check tick size
                if let Some(price) = order.price {
                    if let Some(tick_size) = rules.tick_size.get(&order.symbol) {
                        let remainder = price.as_decimal() % tick_size.as_decimal();
                        if !remainder.is_zero() {
                            errors.push(ValidationError::ExchangeRuleViolation(
                                format!("Price {} not aligned with tick size {}", price, tick_size)
                            ));
                        }
                    }
                }
                
                // Check lot size
                if let Some(lot_size) = rules.lot_size.get(&order.symbol) {
                    if let Err(e) = order.quantity.validate_lot_size(*lot_size) {
                        errors.push(ValidationError::ExchangeRuleViolation(
                            format!("Quantity not aligned with lot size: {}", e)
                        ));
                    }
                }
                
                // Check allowed order types
                if !rules.allowed_order_types.contains(&order.order_type) {
                    errors.push(ValidationError::ExchangeRuleViolation(
                        format!("Order type {:?} not allowed on {}", order.order_type, exchange)
                    ));
                }
            }
        }
        
        // Check risk limits
        if self.context.risk_limits.require_stop_loss && order.stop_loss.is_none() {
            errors.push(ValidationError::RiskLimitExceeded(
                "Stop loss required but not set".to_string()
            ));
        }
        
        if let Some(position_pct) = order.position_size_pct {
            if position_pct > self.context.risk_limits.max_position_pct {
                errors.push(ValidationError::RiskLimitExceeded(
                    format!("Position size {}% exceeds maximum {}%",
                        position_pct * Decimal::from(100),
                        self.context.risk_limits.max_position_pct * Decimal::from(100))
                ));
            }
        }
        
        // Check market conditions
        if self.context.market_conditions.high_volatility {
            if order.order_type == OrderType::Market && order.max_slippage_bps.is_none() {
                errors.push(ValidationError::InvalidOrder(
                    "Market order requires max_slippage_bps during high volatility".to_string()
                ));
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.into_iter().next().expect("SAFETY: Add proper error handling"))
        } else {
            Err(ValidationError::Multiple(errors))
        }
    }
    
    /// Validates a trade
    pub fn validate_trade(&self, trade: &Trade) -> ValidationResult<()> {
        let mut errors = Vec::new();
        
        // Check price and quantity
        if trade.price.is_zero() {
            errors.push(ValidationError::InvalidTrade(
                "Trade price cannot be zero".to_string()
            ));
        }
        
        if trade.quantity.is_zero() {
            errors.push(ValidationError::InvalidTrade(
                "Trade quantity cannot be zero".to_string()
            ));
        }
        
        // Check commission
        if trade.commission < Decimal::ZERO {
            errors.push(ValidationError::InvalidTrade(
                "Commission cannot be negative".to_string()
            ));
        }
        
        // Check value calculation
        let expected_value = trade.price.as_decimal() * trade.quantity.as_decimal();
        if (trade.value - expected_value).abs() > Decimal::from_str_exact("0.01").expect("SAFETY: Add proper error handling") {
            errors.push(ValidationError::InvalidTrade(
                format!("Trade value {} doesn't match price * quantity {}", 
                    trade.value, expected_value)
            ));
        }
        
        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.into_iter().next().expect("SAFETY: Add proper error handling"))
        } else {
            Err(ValidationError::Multiple(errors))
        }
    }
    
    /// Validates a candle
    pub fn validate_candle(&self, candle: &Candle) -> ValidationResult<()> {
        let mut errors = Vec::new();
        
        // Check OHLC relationships
        if candle.high < candle.low {
            errors.push(ValidationError::InvalidCandle(
                format!("High {} cannot be less than low {}", candle.high, candle.low)
            ));
        }
        
        if candle.high < candle.open || candle.high < candle.close {
            errors.push(ValidationError::InvalidCandle(
                "High must be >= max(open, close)".to_string()
            ));
        }
        
        if candle.low > candle.open || candle.low > candle.close {
            errors.push(ValidationError::InvalidCandle(
                "Low must be <= min(open, close)".to_string()
            ));
        }
        
        // Check volume
        if candle.volume.as_decimal() < Decimal::ZERO {
            errors.push(ValidationError::InvalidCandle(
                "Volume cannot be negative".to_string()
            ));
        }
        
        // Check time consistency
        if candle.close_time <= candle.open_time {
            errors.push(ValidationError::InvalidCandle(
                "Close time must be after open time".to_string()
            ));
        }
        
        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.into_iter().next().expect("SAFETY: Add proper error handling"))
        } else {
            Err(ValidationError::Multiple(errors))
        }
    }
    
    /// Validates order book
    pub fn validate_order_book(&self, book: &OrderBook) -> ValidationResult<()> {
        let mut errors = Vec::new();
        
        // Check bid/ask relationship
        if let (Some(best_bid), Some(best_ask)) = (book.best_bid(), book.best_ask()) {
            if best_bid >= best_ask {
                errors.push(ValidationError::InvalidMarketData(
                    format!("Best bid {} >= best ask {}", best_bid, best_ask)
                ));
            }
        }
        
        // Check for crossed levels within each side
        let bid_prices: Vec<_> = book.bids.top_levels(10).iter().map(|l| l.price).collect();
        for i in 1..bid_prices.len() {
            if bid_prices[i] > bid_prices[i-1] {
                errors.push(ValidationError::InvalidMarketData(
                    "Bid prices not in descending order".to_string()
                ));
                break;
            }
        }
        
        let ask_prices: Vec<_> = book.asks.top_levels(10).iter().map(|l| l.price).collect();
        for i in 1..ask_prices.len() {
            if ask_prices[i] < ask_prices[i-1] {
                errors.push(ValidationError::InvalidMarketData(
                    "Ask prices not in ascending order".to_string()
                ));
                break;
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.into_iter().next().expect("SAFETY: Add proper error handling"))
        } else {
            Err(ValidationError::Multiple(errors))
        }
    }
}

// Implement Validatable for core types
impl Validatable for Order {
    fn validate(&self) -> ValidationResult<()> {
        let context = ValidationContext {
            exchange_rules: HashMap::new(),
            risk_limits: RiskLimits::default(),
            market_conditions: MarketConditions {
                high_volatility: false,
                low_liquidity: false,
                near_close: false,
                current_drawdown: Decimal::ZERO,
            },
        };
        
        let mut validator = Validator::new(context);
        validator.validate_order(self)
    }
}

impl Validatable for Trade {
    fn validate(&self) -> ValidationResult<()> {
        let context = ValidationContext {
            exchange_rules: HashMap::new(),
            risk_limits: RiskLimits::default(),
            market_conditions: MarketConditions {
                high_volatility: false,
                low_liquidity: false,
                near_close: false,
                current_drawdown: Decimal::ZERO,
            },
        };
        
        let validator = Validator::new(context);
        validator.validate_trade(self)
    }
}

impl Validatable for Candle {
    fn validate(&self) -> ValidationResult<()> {
        let context = ValidationContext {
            exchange_rules: HashMap::new(),
            risk_limits: RiskLimits::default(),
            market_conditions: MarketConditions {
                high_volatility: false,
                low_liquidity: false,
                near_close: false,
                current_drawdown: Decimal::ZERO,
            },
        };
        
        let validator = Validator::new(context);
        validator.validate_candle(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use crate::{OrderSide, OrderId, TradeSide, TradeId, TradeRole, CandleInterval};
    use chrono::Utc;
    
    #[test]
    fn test_order_validation_passes() {
        let order = Order::market(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            Quantity::new(dec!(0.1)).expect("SAFETY: Add proper error handling"),
        );
        
        assert!(order.validate().is_ok());
    }
    
    #[test]
    fn test_order_validation_fails_zero_quantity() {
        let mut order = Order::market(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            Quantity::zero(),
        );
        order.quantity = Quantity::zero();
        
        let result = order.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("zero"));
    }
    
    #[test]
    fn test_trade_validation() {
        let trade = Trade::new(
            OrderId::new(),
            "ETH/USDT".to_string(),
            TradeSide::Buy,
            Price::new(dec!(3000)).expect("SAFETY: Add proper error handling"),
            Quantity::new(dec!(1)).expect("SAFETY: Add proper error handling"),
            TradeRole::Taker,
            "binance".to_string(),
        );
        
        let validator = Validator::new(ValidationContext {
            exchange_rules: HashMap::new(),
            risk_limits: RiskLimits::default(),
            market_conditions: MarketConditions {
                high_volatility: false,
                low_liquidity: false,
                near_close: false,
                current_drawdown: Decimal::ZERO,
            },
        });
        
        assert!(validator.validate_trade(&trade).is_ok());
    }
    
    #[test]
    fn test_candle_validation() {
        let candle = Candle::new(
            Price::new(dec!(100)).expect("SAFETY: Add proper error handling"),
            Price::new(dec!(110)).expect("SAFETY: Add proper error handling"),
            Price::new(dec!(95)).expect("SAFETY: Add proper error handling"),
            Price::new(dec!(105)).expect("SAFETY: Add proper error handling"),
            Quantity::new(dec!(1000)).expect("SAFETY: Add proper error handling"),
            Utc::now(),
            CandleInterval::Minute1,
        ).expect("SAFETY: Add proper error handling");
        
        assert!(candle.validate().is_ok());
    }
}