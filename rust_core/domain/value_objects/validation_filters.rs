// Value Object: Validation Filters
// Exchange-specific validation rules for orders
// Addresses Sophia's #5 critical feedback on validation filters
// Owner: Casey | Reviewer: Sam

use anyhow::{Result, bail};
use crate::domain::entities::{Order, OrderType, OrderSide};
use crate::domain::value_objects::{Price, Quantity, Symbol};

/// Price filter for order validation
#[derive(Debug, Clone)]
pub struct PriceFilter {
    /// Minimum allowed price
    pub min_price: f64,
    /// Maximum allowed price
    pub max_price: f64,
    /// Tick size (price must be multiple of this)
    pub tick_size: f64,
}

impl PriceFilter {
    /// Validate a price against the filter
    pub fn validate(&self, price: f64) -> Result<()> {
        // Check min/max bounds
        if price < self.min_price {
            bail!(
                "Price {} below minimum {} (error code: -1013)",
                price, self.min_price
            );
        }
        
        if price > self.max_price {
            bail!(
                "Price {} above maximum {} (error code: -1013)",
                price, self.max_price
            );
        }
        
        // Check tick size
        let remainder = (price / self.tick_size) % 1.0;
        if remainder.abs() > 1e-8 {
            bail!(
                "Price {} not a multiple of tick size {} (error code: -1013)",
                price, self.tick_size
            );
        }
        
        Ok(())
    }
}

impl Default for PriceFilter {
    fn default() -> Self {
        Self {
            min_price: 0.01,
            max_price: 1_000_000.0,
            tick_size: 0.01,
        }
    }
}

/// Lot size filter for quantity validation
#[derive(Debug, Clone)]
pub struct LotSizeFilter {
    /// Minimum order quantity
    pub min_qty: f64,
    /// Maximum order quantity
    pub max_qty: f64,
    /// Minimum quantity increment
    pub step_size: f64,
    /// Minimum market order quantity (can be different)
    pub market_min_qty: Option<f64>,
    /// Maximum market order quantity
    pub market_max_qty: Option<f64>,
}

impl LotSizeFilter {
    /// Validate a quantity against the filter
    pub fn validate(&self, quantity: f64, is_market: bool) -> Result<()> {
        let min = if is_market {
            self.market_min_qty.unwrap_or(self.min_qty)
        } else {
            self.min_qty
        };
        
        let max = if is_market {
            self.market_max_qty.unwrap_or(self.max_qty)
        } else {
            self.max_qty
        };
        
        // Check bounds
        if quantity < min {
            bail!(
                "Quantity {} below minimum {} (error code: -1013)",
                quantity, min
            );
        }
        
        if quantity > max {
            bail!(
                "Quantity {} above maximum {} (error code: -1013)",
                quantity, max
            );
        }
        
        // Check step size
        let remainder = (quantity / self.step_size) % 1.0;
        if remainder.abs() > 1e-8 {
            bail!(
                "Quantity {} not a multiple of step size {} (error code: -1013)",
                quantity, self.step_size
            );
        }
        
        Ok(())
    }
}

impl Default for LotSizeFilter {
    fn default() -> Self {
        Self {
            min_qty: 0.00001,
            max_qty: 10000.0,
            step_size: 0.00001,
            market_min_qty: Some(0.0001),
            market_max_qty: Some(1000.0),
        }
    }
}

/// Notional filter (minimum order value)
#[derive(Debug, Clone)]
pub struct NotionalFilter {
    /// Minimum notional value (price * quantity)
    pub min_notional: f64,
    /// Apply to market orders
    pub apply_to_market: bool,
    /// Average price for market order estimation
    pub avg_price_minutes: u32,
}

impl NotionalFilter {
    /// Validate notional value
    pub fn validate(&self, price: f64, quantity: f64, is_market: bool) -> Result<()> {
        if is_market && !self.apply_to_market {
            return Ok(());
        }
        
        let notional = price * quantity;
        
        if notional < self.min_notional {
            bail!(
                "Notional value {} below minimum {} (error code: -1013)",
                notional, self.min_notional
            );
        }
        
        Ok(())
    }
}

impl Default for NotionalFilter {
    fn default() -> Self {
        Self {
            min_notional: 10.0, // $10 minimum
            apply_to_market: true,
            avg_price_minutes: 5,
        }
    }
}

/// Percent price filter (protects against fat finger trades)
#[derive(Debug, Clone)]
pub struct PercentPriceFilter {
    /// Maximum percentage above average price
    pub up_percent: f64,
    /// Maximum percentage below average price
    pub down_percent: f64,
    /// Time window for average price (minutes)
    pub avg_price_minutes: u32,
}

impl PercentPriceFilter {
    /// Validate price against recent average
    pub fn validate(&self, price: f64, avg_price: f64, side: OrderSide) -> Result<()> {
        let percent_diff = ((price - avg_price) / avg_price) * 100.0;
        
        match side {
            OrderSide::Buy => {
                // Buy orders shouldn't be too high above market
                if percent_diff > self.up_percent {
                    bail!(
                        "Buy price {}% above average exceeds {}% limit (error code: -1013)",
                        percent_diff, self.up_percent
                    );
                }
            }
            OrderSide::Sell => {
                // Sell orders shouldn't be too far below market
                if percent_diff < -self.down_percent {
                    bail!(
                        "Sell price {}% below average exceeds {}% limit (error code: -1013)",
                        percent_diff.abs(), self.down_percent
                    );
                }
            }
        }
        
        Ok(())
    }
}

impl Default for PercentPriceFilter {
    fn default() -> Self {
        Self {
            up_percent: 10.0,   // 10% above
            down_percent: 10.0, // 10% below
            avg_price_minutes: 5,
        }
    }
}

/// Ice berg order filter
#[derive(Debug, Clone)]
pub struct IcebergFilter {
    /// Maximum number of iceberg parts
    pub max_parts: u32,
    /// Minimum quantity per iceberg part
    pub min_qty_per_part: f64,
}

/// Maximum orders filter
#[derive(Debug, Clone)]
pub struct MaxOrdersFilter {
    /// Maximum number of open orders per symbol
    pub max_open_orders: u32,
    /// Maximum algorithmic orders (stop, OCO, etc.)
    pub max_algo_orders: u32,
}

/// Comprehensive validation filters for a trading pair
#[derive(Debug, Clone)]
pub struct ValidationFilters {
    pub symbol: Symbol,
    pub price_filter: PriceFilter,
    pub lot_size_filter: LotSizeFilter,
    pub notional_filter: NotionalFilter,
    pub percent_price_filter: Option<PercentPriceFilter>,
    pub iceberg_filter: Option<IcebergFilter>,
    pub max_orders_filter: MaxOrdersFilter,
    pub is_trading_enabled: bool,
    pub is_spot_trading_allowed: bool,
    pub is_margin_trading_allowed: bool,
}

impl ValidationFilters {
    /// Create filters for BTC/USDT (example)
    pub fn btc_usdt() -> Self {
        Self {
            symbol: Symbol::new("BTC/USDT").unwrap(),
            price_filter: PriceFilter {
                min_price: 1.0,
                max_price: 1_000_000.0,
                tick_size: 0.01,
            },
            lot_size_filter: LotSizeFilter {
                min_qty: 0.00001,
                max_qty: 9000.0,
                step_size: 0.00001,
                market_min_qty: Some(0.00001),
                market_max_qty: Some(100.0),
            },
            notional_filter: NotionalFilter {
                min_notional: 10.0,
                apply_to_market: true,
                avg_price_minutes: 5,
            },
            percent_price_filter: Some(PercentPriceFilter::default()),
            iceberg_filter: Some(IcebergFilter {
                max_parts: 10,
                min_qty_per_part: 0.001,
            }),
            max_orders_filter: MaxOrdersFilter {
                max_open_orders: 200,
                max_algo_orders: 10,
            },
            is_trading_enabled: true,
            is_spot_trading_allowed: true,
            is_margin_trading_allowed: false,
        }
    }
    
    /// Validate an order against all filters
    pub fn validate_order(&self, order: &Order, avg_price: Option<f64>) -> Result<()> {
        // Check if trading is enabled
        if !self.is_trading_enabled {
            bail!("Trading is disabled for {}", self.symbol);
        }
        
        // Validate quantity
        let is_market = order.order_type() == OrderType::Market;
        self.lot_size_filter.validate(order.quantity().value(), is_market)?;
        
        // Validate price (for limit orders)
        if let Some(price) = order.price() {
            self.price_filter.validate(price.value())?;
            
            // Validate notional
            self.notional_filter.validate(
                price.value(),
                order.quantity().value(),
                is_market
            )?;
            
            // Validate percent price (if configured and avg_price provided)
            if let (Some(filter), Some(avg)) = (&self.percent_price_filter, avg_price) {
                filter.validate(price.value(), avg, order.side())?;
            }
        } else if is_market {
            // For market orders, use average price if available
            if let Some(avg) = avg_price {
                self.notional_filter.validate(
                    avg,
                    order.quantity().value(),
                    true
                )?;
            }
        }
        
        Ok(())
    }
    
    /// Check if an order would cross the spread (for post-only validation)
    pub fn would_cross_spread(
        &self,
        order: &Order,
        best_bid: f64,
        best_ask: f64,
    ) -> bool {
        if let Some(price) = order.price() {
            match order.side() {
                OrderSide::Buy => price.value() >= best_ask,
                OrderSide::Sell => price.value() <= best_bid,
            }
        } else {
            // Market orders always cross
            true
        }
    }
}

/// Exchange-specific error codes
#[derive(Debug, Clone)]
pub enum ExchangeErrorCode {
    FilterFailure { code: i32 },        // -1013
    TooManyOrders { code: i32 },        // -1015
    InvalidTimestamp { code: i32 },     // -1021
    PostOnlyWouldCross { code: i32 },   // -2010
    UnknownOrder { code: i32 },         // -2011
    DuplicateOrder { code: i32 },       // -2026
}

impl ExchangeErrorCode {
    pub fn to_string(&self) -> String {
        match self {
            Self::FilterFailure { code } => format!("Filter failure ({})", code),
            Self::TooManyOrders { code } => format!("Too many orders ({})", code),
            Self::InvalidTimestamp { code } => format!("Invalid timestamp ({})", code),
            Self::PostOnlyWouldCross { code } => format!("Post-only would cross ({})", code),
            Self::UnknownOrder { code } => format!("Unknown order ({})", code),
            Self::DuplicateOrder { code } => format!("Duplicate order ({})", code),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_price_filter() {
        let filter = PriceFilter {
            min_price: 10.0,
            max_price: 100.0,
            tick_size: 0.1,
        };
        
        // Valid price
        assert!(filter.validate(50.0).is_ok());
        
        // Below minimum
        assert!(filter.validate(5.0).is_err());
        
        // Above maximum
        assert!(filter.validate(150.0).is_err());
        
        // Invalid tick size
        assert!(filter.validate(50.05).is_err());
        assert!(filter.validate(50.1).is_ok());
    }
    
    #[test]
    fn test_lot_size_filter() {
        let filter = LotSizeFilter {
            min_qty: 0.01,
            max_qty: 100.0,
            step_size: 0.01,
            market_min_qty: Some(0.1),
            market_max_qty: Some(10.0),
        };
        
        // Valid limit order quantity
        assert!(filter.validate(1.0, false).is_ok());
        
        // Valid market order quantity
        assert!(filter.validate(1.0, true).is_ok());
        
        // Below limit minimum
        assert!(filter.validate(0.005, false).is_err());
        
        // Below market minimum
        assert!(filter.validate(0.05, true).is_err());
        
        // Invalid step size
        assert!(filter.validate(1.005, false).is_err());
    }
    
    #[test]
    fn test_notional_filter() {
        let filter = NotionalFilter {
            min_notional: 10.0,
            apply_to_market: true,
            avg_price_minutes: 5,
        };
        
        // Valid notional
        assert!(filter.validate(100.0, 1.0, false).is_ok()); // $100
        
        // Below minimum
        assert!(filter.validate(5.0, 1.0, false).is_err()); // $5
        
        // Market order with apply_to_market = true
        assert!(filter.validate(5.0, 1.0, true).is_err());
    }
    
    #[test]
    fn test_percent_price_filter() {
        let filter = PercentPriceFilter {
            up_percent: 5.0,
            down_percent: 5.0,
            avg_price_minutes: 5,
        };
        
        let avg_price = 100.0;
        
        // Valid buy price (within 5% above)
        assert!(filter.validate(104.0, avg_price, OrderSide::Buy).is_ok());
        
        // Invalid buy price (>5% above)
        assert!(filter.validate(106.0, avg_price, OrderSide::Buy).is_err());
        
        // Valid sell price (within 5% below)
        assert!(filter.validate(96.0, avg_price, OrderSide::Sell).is_ok());
        
        // Invalid sell price (>5% below)
        assert!(filter.validate(94.0, avg_price, OrderSide::Sell).is_err());
    }
    
    #[test]
    fn test_comprehensive_validation() {
        let filters = ValidationFilters::btc_usdt();
        
        // Create a test order
        let order = Order::limit(
            Symbol::new("BTC/USDT").unwrap(),
            OrderSide::Buy,
            Price::new(50000.0).unwrap(),
            Quantity::new(0.001).unwrap(),
            crate::domain::entities::TimeInForce::GTC,
        );
        
        // Should pass validation
        assert!(filters.validate_order(&order, Some(50000.0)).is_ok());
        
        // Test would_cross_spread
        assert!(!filters.would_cross_spread(&order, 49999.0, 50001.0));
        assert!(filters.would_cross_spread(&order, 49999.0, 50000.0)); // Would cross
    }
}