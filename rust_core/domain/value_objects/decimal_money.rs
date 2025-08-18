// Decimal Arithmetic for Money Operations
// Owner: Quinn | Reviewer: Sam (Code Quality)
// Pre-Production Requirement #3 from Sophia
// Target: Zero rounding errors in financial calculations

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Money type using decimal arithmetic for perfect precision
/// Sophia's requirement: No floating point errors in money calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Money {
    amount: Decimal,
    currency: Currency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Currency {
    USD,
    BTC,
    ETH,
    USDT,
    USDC,
}

impl Money {
    /// Create new Money with exact decimal representation
    pub fn new(amount: Decimal, currency: Currency) -> Self {
        Self { amount, currency }
    }
    
    /// Create from string to avoid float conversion
    pub fn from_str(s: &str, currency: Currency) -> Result<Self, MoneyError> {
        let amount = Decimal::from_str(s)
            .map_err(|e| MoneyError::ParseError(e.to_string()))?;
        Ok(Self::new(amount, currency))
    }
    
    /// Create from integer cents/satoshis
    pub fn from_minor_units(units: i64, currency: Currency) -> Self {
        let scale = currency.decimal_places();
        let amount = Decimal::new(units, scale);
        Self::new(amount, currency)
    }
    
    /// Get amount in minor units (cents/satoshis)
    pub fn to_minor_units(&self) -> i64 {
        let scale = Decimal::new(1, self.currency.decimal_places());
        (self.amount / scale).to_i64().unwrap_or(0)
    }
    
    /// Round to currency precision
    pub fn round(&self) -> Self {
        let places = self.currency.decimal_places();
        Self::new(self.amount.round_dp(places), self.currency)
    }
    
    /// Check if amount is zero
    pub fn is_zero(&self) -> bool {
        self.amount.is_zero()
    }
    
    /// Check if amount is positive
    pub fn is_positive(&self) -> bool {
        self.amount.is_sign_positive() && !self.amount.is_zero()
    }
    
    /// Check if amount is negative
    pub fn is_negative(&self) -> bool {
        self.amount.is_sign_negative()
    }
    
    /// Get absolute value
    pub fn abs(&self) -> Self {
        Self::new(self.amount.abs(), self.currency)
    }
    
    /// Calculate percentage
    pub fn percentage(&self, percent: Decimal) -> Self {
        let result = self.amount * percent / Decimal::new(100, 0);
        Self::new(result, self.currency)
    }
    
    /// Apply fee with exact decimal calculation
    pub fn apply_fee(&self, fee_bps: i32) -> (Money, Money) {
        let fee_decimal = Decimal::new(fee_bps as i64, 4); // Basis points = 0.01%
        let fee_amount = self.amount * fee_decimal;
        let net_amount = self.amount - fee_amount;
        
        (
            Self::new(net_amount, self.currency),
            Self::new(fee_amount, self.currency),
        )
    }
    
    /// Convert to another currency with exact rate
    pub fn convert_to(&self, target: Currency, rate: Decimal) -> Money {
        let converted = self.amount * rate;
        Money::new(converted, target)
    }
}

impl Currency {
    /// Get decimal places for currency
    pub fn decimal_places(&self) -> u32 {
        match self {
            Currency::USD | Currency::USDT | Currency::USDC => 2,
            Currency::BTC => 8,
            Currency::ETH => 18,
        }
    }
    
    /// Get minimum tradeable amount
    pub fn min_amount(&self) -> Decimal {
        match self {
            Currency::USD | Currency::USDT | Currency::USDC => Decimal::new(1, 2),  // $0.01
            Currency::BTC => Decimal::new(1, 8),   // 0.00000001 BTC (1 satoshi)
            Currency::ETH => Decimal::new(1, 18),  // 1 wei
        }
    }
}

// Arithmetic operations
impl Add for Money {
    type Output = Result<Money, MoneyError>;
    
    fn add(self, other: Money) -> Self::Output {
        if self.currency != other.currency {
            return Err(MoneyError::CurrencyMismatch);
        }
        Ok(Money::new(self.amount + other.amount, self.currency))
    }
}

impl Sub for Money {
    type Output = Result<Money, MoneyError>;
    
    fn sub(self, other: Money) -> Self::Output {
        if self.currency != other.currency {
            return Err(MoneyError::CurrencyMismatch);
        }
        Ok(Money::new(self.amount - other.amount, self.currency))
    }
}

impl Mul<Decimal> for Money {
    type Output = Money;
    
    fn mul(self, scalar: Decimal) -> Money {
        Money::new(self.amount * scalar, self.currency)
    }
}

impl Div<Decimal> for Money {
    type Output = Money;
    
    fn div(self, scalar: Decimal) -> Money {
        Money::new(self.amount / scalar, self.currency)
    }
}

impl Neg for Money {
    type Output = Money;
    
    fn neg(self) -> Money {
        Money::new(-self.amount, self.currency)
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let places = self.currency.decimal_places();
        let rounded = self.amount.round_dp(places);
        write!(f, "{} {:?}", rounded, self.currency)
    }
}

/// Price type for order book entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Price {
    value: Decimal,
    pair: TradingPair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TradingPair {
    base: Currency,
    quote: Currency,
}

impl Price {
    pub fn new(value: Decimal, pair: TradingPair) -> Self {
        Self { value, pair }
    }
    
    pub fn from_str(s: &str, pair: TradingPair) -> Result<Self, MoneyError> {
        let value = Decimal::from_str(s)
            .map_err(|e| MoneyError::ParseError(e.to_string()))?;
        Ok(Self::new(value, pair))
    }
    
    /// Calculate total value for given quantity
    pub fn calculate_value(&self, quantity: Decimal) -> Money {
        let total = self.value * quantity;
        Money::new(total, self.pair.quote)
    }
    
    /// Apply price improvement
    pub fn improve_by_bps(&self, bps: i32) -> Price {
        let improvement = Decimal::new(bps as i64, 4);
        let improved_value = if bps > 0 {
            self.value * (Decimal::ONE + improvement)
        } else {
            self.value * (Decimal::ONE - improvement.abs())
        };
        Price::new(improved_value, self.pair)
    }
    
    /// Check if price is at tick size
    pub fn is_at_tick(&self, tick_size: Decimal) -> bool {
        (self.value % tick_size).is_zero()
    }
    
    /// Round to tick size
    pub fn round_to_tick(&self, tick_size: Decimal) -> Price {
        let rounded = (self.value / tick_size).round() * tick_size;
        Price::new(rounded, self.pair)
    }
}

/// Quantity type for order sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Quantity {
    value: Decimal,
    currency: Currency,
}

impl Quantity {
    pub fn new(value: Decimal, currency: Currency) -> Self {
        Self { value, currency }
    }
    
    pub fn from_str(s: &str, currency: Currency) -> Result<Self, MoneyError> {
        let value = Decimal::from_str(s)
            .map_err(|e| MoneyError::ParseError(e.to_string()))?;
        Ok(Self::new(value, currency))
    }
    
    /// Check if quantity meets lot size
    pub fn is_valid_lot(&self, lot_size: Decimal) -> bool {
        (self.value % lot_size).is_zero()
    }
    
    /// Round to lot size
    pub fn round_to_lot(&self, lot_size: Decimal) -> Quantity {
        let rounded = (self.value / lot_size).floor() * lot_size;
        Quantity::new(rounded, self.currency)
    }
    
    /// Split into fills
    pub fn split(&self, num_fills: usize) -> Vec<Quantity> {
        if num_fills == 0 {
            return vec![];
        }
        
        let fill_size = self.value / Decimal::new(num_fills as i64, 0);
        let mut fills = vec![Quantity::new(fill_size, self.currency); num_fills - 1];
        
        // Last fill gets the remainder to avoid rounding errors
        let total_filled = fill_size * Decimal::new((num_fills - 1) as i64, 0);
        let remainder = self.value - total_filled;
        fills.push(Quantity::new(remainder, self.currency));
        
        fills
    }
}

/// Portfolio value with multi-currency support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    balances: HashMap<Currency, Money>,
    base_currency: Currency,
}

impl Portfolio {
    pub fn new(base_currency: Currency) -> Self {
        Self {
            balances: HashMap::new(),
            base_currency,
        }
    }
    
    /// Add balance
    pub fn add_balance(&mut self, money: Money) -> Result<(), MoneyError> {
        let balance = self.balances.entry(money.currency).or_insert_with(|| {
            Money::new(Decimal::ZERO, money.currency)
        });
        
        *balance = (*balance + money)?;
        Ok(())
    }
    
    /// Get balance for currency
    pub fn get_balance(&self, currency: Currency) -> Money {
        self.balances.get(&currency)
            .copied()
            .unwrap_or_else(|| Money::new(Decimal::ZERO, currency))
    }
    
    /// Calculate total value in base currency
    pub fn total_value(&self, rates: &HashMap<(Currency, Currency), Decimal>) -> Money {
        let mut total = Decimal::ZERO;
        
        for (currency, balance) in &self.balances {
            if *currency == self.base_currency {
                total += balance.amount;
            } else {
                // Look up conversion rate
                if let Some(rate) = rates.get(&(*currency, self.base_currency)) {
                    total += balance.amount * rate;
                }
            }
        }
        
        Money::new(total, self.base_currency)
    }
    
    /// Check if sufficient balance for withdrawal
    pub fn has_sufficient_balance(&self, required: Money) -> bool {
        self.get_balance(required.currency) >= required
    }
}

#[derive(Debug, Clone)]
pub enum MoneyError {
    CurrencyMismatch,
    InsufficientBalance,
    ParseError(String),
    InvalidAmount,
}

// ============================================================================
// MIGRATION HELPERS
// ============================================================================

/// Convert legacy f64 prices to decimal
pub fn migrate_price_f64_to_decimal(price: f64, pair: TradingPair) -> Price {
    // Use string conversion to avoid float precision issues
    let value = Decimal::from_str(&format!("{:.8}", price))
        .unwrap_or_else(|_| Decimal::from_f64(price).unwrap());
    Price::new(value, pair)
}

/// Convert legacy f64 amounts to Money
pub fn migrate_amount_f64_to_money(amount: f64, currency: Currency) -> Money {
    let decimal = Decimal::from_str(&format!("{:.8}", amount))
        .unwrap_or_else(|_| Decimal::from_f64(amount).unwrap());
    Money::new(decimal, currency)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exact_decimal_arithmetic() {
        // Test that 0.1 + 0.2 = 0.3 (fails with f64)
        let a = Money::from_str("0.1", Currency::USD).unwrap();
        let b = Money::from_str("0.2", Currency::USD).unwrap();
        let c = (a + b).unwrap();
        
        assert_eq!(c, Money::from_str("0.3", Currency::USD).unwrap());
    }
    
    #[test]
    fn test_fee_calculation_exact() {
        // Test fee calculation with basis points
        let amount = Money::from_str("1000.00", Currency::USD).unwrap();
        let (net, fee) = amount.apply_fee(25); // 0.25%
        
        assert_eq!(net, Money::from_str("997.50", Currency::USD).unwrap());
        assert_eq!(fee, Money::from_str("2.50", Currency::USD).unwrap());
    }
    
    #[test]
    fn test_btc_satoshi_precision() {
        // Test Bitcoin satoshi precision (8 decimal places)
        let btc = Money::from_minor_units(100_000_000, Currency::BTC); // 1 BTC
        assert_eq!(btc, Money::from_str("1.00000000", Currency::BTC).unwrap());
        
        let satoshi = Money::from_minor_units(1, Currency::BTC); // 1 satoshi
        assert_eq!(satoshi, Money::from_str("0.00000001", Currency::BTC).unwrap());
    }
    
    #[test]
    fn test_quantity_splitting_exact() {
        // Test that splitting preserves total quantity exactly
        let qty = Quantity::from_str("10.0", Currency::BTC).unwrap();
        let fills = qty.split(3);
        
        let total: Decimal = fills.iter().map(|f| f.value).sum();
        assert_eq!(total, qty.value);
    }
    
    #[test]
    fn test_price_tick_rounding() {
        let price = Price::from_str("50123.456", TradingPair {
            base: Currency::BTC,
            quote: Currency::USD,
        }).unwrap();
        
        let tick_size = Decimal::from_str("0.01").unwrap();
        let rounded = price.round_to_tick(tick_size);
        
        assert_eq!(rounded.value, Decimal::from_str("50123.46").unwrap());
    }
}

// Benefits of decimal arithmetic:
// - No floating point errors (0.1 + 0.2 = 0.3)
// - Exact fee calculations
// - Proper satoshi/wei precision
// - Consistent rounding behavior
// - Serialization without precision loss