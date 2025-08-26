//! # Phantom Type Wrappers for Currency Safety
//! 
//! Combines Price and Quantity with Currency phantom types to prevent
//! mixing different currencies at compile time.
//!
//! ## Zero-Cost Abstraction
//! The phantom type parameter adds no runtime overhead - it exists only at compile time.
//!
//! ## Example
//! ```compile_fail
//! let btc_price: TypedPrice<BTC> = TypedPrice::new(50000.0)?;
//! let usd_price: TypedPrice<USD> = TypedPrice::new(100.0)?;
//! let sum = btc_price.add(usd_price); // Compile error! Cannot add BTC to USD
//! ```

use crate::{Currency, Price, PriceError, Quantity, QuantityError};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Div};

/// Type-safe price with currency information
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedPrice<C: Currency> {
    price: Price,
    #[serde(skip)]
    _currency: PhantomData<C>,
}

impl<C: Currency> TypedPrice<C> {
    /// Creates a new typed price
    pub fn new(value: impl Into<Decimal>) -> Result<Self, PriceError> {
        Ok(Self {
            price: Price::new(value)?,
            _currency: PhantomData,
        })
    }
    
    /// Creates a typed price from an untyped price
    pub fn from_price(price: Price) -> Self {
        Self {
            price,
            _currency: PhantomData,
        }
    }
    
    /// Gets the underlying price
    pub const fn as_price(&self) -> Price {
        self.price
    }
    
    /// Gets the decimal value
    pub fn as_decimal(&self) -> Decimal {
        self.price.as_decimal()
    }
    
    /// Gets the currency code
    pub fn currency_code() -> &'static str {
        C::CODE
    }
    
    /// Formats with currency symbol
    pub fn format_with_symbol(&self) -> String {
        format!("{}{}", C::SYMBOL, self.price)
    }
    
    /// Zero price in this currency
    pub fn zero() -> Self {
        Self {
            price: Price::zero(),
            _currency: PhantomData,
        }
    }
    
    /// Adds another price of the same currency
    pub fn add(&self, other: TypedPrice<C>) -> Result<Self, PriceError> {
        Ok(Self {
            price: self.price.add(other.price)?,
            _currency: PhantomData,
        })
    }
    
    /// Subtracts another price of the same currency
    pub fn subtract(&self, other: TypedPrice<C>) -> Result<Self, PriceError> {
        Ok(Self {
            price: self.price.subtract(other.price)?,
            _currency: PhantomData,
        })
    }
    
    /// Multiplies by a scalar (maintains currency)
    pub fn multiply(&self, scalar: impl Into<Decimal>) -> Result<Self, PriceError> {
        Ok(Self {
            price: self.price.multiply(scalar)?,
            _currency: PhantomData,
        })
    }
    
    /// Divides by a scalar (maintains currency)
    pub fn divide(&self, scalar: impl Into<Decimal>) -> Result<Self, PriceError> {
        Ok(Self {
            price: self.price.divide(scalar)?,
            _currency: PhantomData,
        })
    }
}

impl<C: Currency> fmt::Display for TypedPrice<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.price, C::CODE)
    }
}

/// Type-safe quantity with currency/asset information
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedQuantity<C: Currency> {
    quantity: Quantity,
    #[serde(skip)]
    _currency: PhantomData<C>,
}

impl<C: Currency> TypedQuantity<C> {
    /// Creates a new typed quantity
    pub fn new(value: impl Into<Decimal>) -> Result<Self, QuantityError> {
        Ok(Self {
            quantity: Quantity::new(value)?,
            _currency: PhantomData,
        })
    }
    
    /// Creates from an untyped quantity
    pub fn from_quantity(quantity: Quantity) -> Self {
        Self {
            quantity,
            _currency: PhantomData,
        }
    }
    
    /// Gets the underlying quantity
    pub const fn as_quantity(&self) -> Quantity {
        self.quantity
    }
    
    /// Gets the decimal value
    pub fn as_decimal(&self) -> Decimal {
        self.quantity.as_decimal()
    }
    
    /// Gets the currency/asset code
    pub fn asset_code() -> &'static str {
        C::CODE
    }
    
    /// Zero quantity in this currency
    pub fn zero() -> Self {
        Self {
            quantity: Quantity::zero(),
            _currency: PhantomData,
        }
    }
    
    /// Adds another quantity of the same currency
    pub fn add(&self, other: TypedQuantity<C>) -> Result<Self, QuantityError> {
        Ok(Self {
            quantity: self.quantity.add(other.quantity)?,
            _currency: PhantomData,
        })
    }
    
    /// Subtracts another quantity of the same currency
    pub fn subtract(&self, other: TypedQuantity<C>) -> Result<Self, QuantityError> {
        Ok(Self {
            quantity: self.quantity.subtract(other.quantity)?,
            _currency: PhantomData,
        })
    }
    
    /// Calculates total value (quantity * price)
    pub fn value(&self, price: TypedPrice<C>) -> Result<TypedPrice<C>, PriceError> {
        let value = self.quantity.as_decimal() * price.as_decimal();
        TypedPrice::new(value)
    }
}

impl<C: Currency> fmt::Display for TypedQuantity<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.quantity, C::CODE)
    }
}

/// Type-safe money amount (combines quantity and currency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Money<C: Currency> {
    amount: Decimal,
    #[serde(skip)]
    _currency: PhantomData<C>,
}

impl<C: Currency> Money<C> {
    /// Creates new money amount
    pub fn new(amount: impl Into<Decimal>) -> Self {
        Self {
            amount: amount.into(),
            _currency: PhantomData,
        }
    }
    
    /// Zero money
    pub fn zero() -> Self {
        Self {
            amount: Decimal::ZERO,
            _currency: PhantomData,
        }
    }
    
    /// Gets the amount
    pub const fn amount(&self) -> Decimal {
        self.amount
    }
    
    /// Formats with currency symbol
    pub fn format_with_symbol(&self) -> String {
        format!("{}{:.prec$}", C::SYMBOL, self.amount, prec = C::DECIMALS as usize)
    }
    
    /// Adds another money amount of the same currency
    pub fn add(&self, other: Money<C>) -> Money<C> {
        Self {
            amount: self.amount + other.amount,
            _currency: PhantomData,
        }
    }
    
    /// Subtracts another money amount of the same currency
    pub fn subtract(&self, other: Money<C>) -> Option<Money<C>> {
        if other.amount > self.amount {
            return None;
        }
        
        Some(Self {
            amount: self.amount - other.amount,
            _currency: PhantomData,
        })
    }
}

impl<C: Currency> fmt::Display for Money<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:.prec$} {}", 
            C::SYMBOL, 
            self.amount, 
            C::CODE,
            prec = C::DECIMALS as usize)
    }
}

/// Exchange rate between two currencies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExchangeRate<From: Currency, To: Currency> {
    rate: Decimal,
    #[serde(skip)]
    _from: PhantomData<From>,
    #[serde(skip)]
    _to: PhantomData<To>,
}

impl<From: Currency, To: Currency> ExchangeRate<From, To> {
    /// Creates a new exchange rate
    pub fn new(rate: Decimal) -> Self {
        Self {
            rate,
            _from: PhantomData,
            _to: PhantomData,
        }
    }
    
    /// Converts a price from one currency to another
    pub fn convert_price(&self, price: TypedPrice<From>) -> TypedPrice<To> {
        let converted_value = price.as_decimal() * self.rate;
        TypedPrice::from_price(Price::new(converted_value).unwrap_or(Price::zero()))
    }
    
    /// Converts a quantity value from one currency to another
    pub fn convert_value(&self, quantity: TypedQuantity<From>, price: TypedPrice<From>) -> Money<To> {
        let value = quantity.as_decimal() * price.as_decimal() * self.rate;
        Money::new(value)
    }
    
    /// Gets the inverse exchange rate
    pub fn inverse(&self) -> ExchangeRate<To, From> {
        ExchangeRate {
            rate: Decimal::ONE / self.rate,
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::{BTC, USD, USDT};
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_typed_price_operations() {
        let btc_price1 = TypedPrice::<BTC>::new(dec!(50000)).unwrap();
        let btc_price2 = TypedPrice::<BTC>::new(dec!(10000)).unwrap();
        
        let sum = btc_price1.add(btc_price2).unwrap();
        assert_eq!(sum.as_decimal(), dec!(60000));
        
        // This would not compile:
        // let usd_price = TypedPrice::<USD>::new(dec!(100)).unwrap();
        // let invalid_sum = btc_price1.add(usd_price); // Compile error!
    }
    
    #[test]
    fn test_typed_quantity_value() {
        let btc_qty = TypedQuantity::<BTC>::new(dec!(0.5)).unwrap();
        let btc_price = TypedPrice::<BTC>::new(dec!(50000)).unwrap();
        
        let value = btc_qty.value(btc_price).unwrap();
        assert_eq!(value.as_decimal(), dec!(25000));
    }
    
    #[test]
    fn test_exchange_rate_conversion() {
        let btc_price = TypedPrice::<BTC>::new(dec!(50000)).unwrap();
        let btc_to_usd = ExchangeRate::<BTC, USD>::new(dec!(50000));
        
        let usd_price = btc_to_usd.convert_price(btc_price);
        assert_eq!(usd_price.as_decimal(), dec!(2500000000)); // 50000 * 50000
    }
    
    #[test]
    fn test_money_formatting() {
        let usd_amount = Money::<USD>::new(dec!(1234.56));
        assert_eq!(usd_amount.format_with_symbol(), "$1234.56");
        assert_eq!(usd_amount.to_string(), "$1234.56 USD");
    }
}