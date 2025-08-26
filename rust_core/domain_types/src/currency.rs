//! # Currency Phantom Types for Compile-Time Safety
//! 
//! Prevents mixing of different currencies at compile time using Rust's type system.
//! Zero runtime overhead - all checks happen at compilation.
//!
//! ## Design Pattern
//! Phantom types carry currency information in the type system without runtime cost.
//! This prevents bugs like adding USD prices to BTC prices.
//!
//! ## External Research Applied
//! - "Making Illegal States Unrepresentable" - Yaron Minsky
//! - "Parse, Don't Validate" - Alexis King  
//! - Phantom Types in Haskell/Rust

use std::marker::PhantomData;

/// Marker trait for currencies
pub trait Currency: Send + Sync + 'static {
    /// Currency code (e.g., "USD", "BTC")
    const CODE: &'static str;
    
    /// Currency symbol (e.g., "$", "₿")
    const SYMBOL: &'static str;
    
    /// Number of decimal places for display
    const DECIMALS: u32;
    
    /// Is this a fiat currency?
    const IS_FIAT: bool;
    
    /// Is this a stablecoin?
    const IS_STABLE: bool;
}

/// US Dollar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct USD;

impl Currency for USD {
    const CODE: &'static str = "USD";
    const SYMBOL: &'static str = "$";
    const DECIMALS: u32 = 2;
    const IS_FIAT: bool = true;
    const IS_STABLE: bool = true;
}

/// Bitcoin
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BTC;

impl Currency for BTC {
    const CODE: &'static str = "BTC";
    const SYMBOL: &'static str = "₿";
    const DECIMALS: u32 = 8;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = false;
}

/// Ethereum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ETH;

impl Currency for ETH {
    const CODE: &'static str = "ETH";
    const SYMBOL: &'static str = "Ξ";
    const DECIMALS: u32 = 18;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = false;
}

/// Tether (USDT)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct USDT;

impl Currency for USDT {
    const CODE: &'static str = "USDT";
    const SYMBOL: &'static str = "₮";
    const DECIMALS: u32 = 6;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = true;
}

/// USDC
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct USDC;

impl Currency for USDC {
    const CODE: &'static str = "USDC";
    const SYMBOL: &'static str = "$";
    const DECIMALS: u32 = 6;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = true;
}

/// Euro
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EUR;

impl Currency for EUR {
    const CODE: &'static str = "EUR";
    const SYMBOL: &'static str = "€";
    const DECIMALS: u32 = 2;
    const IS_FIAT: bool = true;
    const IS_STABLE: bool = true;
}

/// Japanese Yen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JPY;

impl Currency for JPY {
    const CODE: &'static str = "JPY";
    const SYMBOL: &'static str = "¥";
    const DECIMALS: u32 = 0; // No decimal places for JPY
    const IS_FIAT: bool = true;
    const IS_STABLE: bool = true;
}

/// Binance Coin
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BNB;

impl Currency for BNB {
    const CODE: &'static str = "BNB";
    const SYMBOL: &'static str = "BNB";
    const DECIMALS: u32 = 8;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = false;
}

/// Solana
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SOL;

impl Currency for SOL {
    const CODE: &'static str = "SOL";
    const SYMBOL: &'static str = "◎";
    const DECIMALS: u32 = 9;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = false;
}

/// XRP (Ripple)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XRP;

impl Currency for XRP {
    const CODE: &'static str = "XRP";
    const SYMBOL: &'static str = "XRP";
    const DECIMALS: u32 = 6;
    const IS_FIAT: bool = false;
    const IS_STABLE: bool = false;
}

/// Trading pair representation with compile-time safety
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TradingPair<Base: Currency, Quote: Currency> {
    _base: PhantomData<Base>,
    _quote: PhantomData<Quote>,
}

impl<Base: Currency, Quote: Currency> TradingPair<Base, Quote> {
    /// Creates a new trading pair
    pub const fn new() -> Self {
        Self {
            _base: PhantomData,
            _quote: PhantomData,
        }
    }
    
    /// Gets the symbol representation (e.g., "BTC/USDT")
    pub fn symbol() -> String {
        format!("{}/{}", Base::CODE, Quote::CODE)
    }
    
    /// Gets the base currency code
    pub fn base_currency() -> &'static str {
        Base::CODE
    }
    
    /// Gets the quote currency code
    pub fn quote_currency() -> &'static str {
        Quote::CODE
    }
    
    /// Checks if this is a stablecoin pair
    pub fn is_stable_pair() -> bool {
        Base::IS_STABLE || Quote::IS_STABLE
    }
    
    /// Checks if this is a crypto-to-crypto pair
    pub fn is_crypto_pair() -> bool {
        !Base::IS_FIAT && !Quote::IS_FIAT
    }
    
    /// Checks if this is a fiat pair
    pub fn is_fiat_pair() -> bool {
        Base::IS_FIAT || Quote::IS_FIAT
    }
}

/// Common trading pairs
pub type BTC_USDT = TradingPair<BTC, USDT>;
pub type ETH_USDT = TradingPair<ETH, USDT>;
pub type BTC_USD = TradingPair<BTC, USD>;
pub type ETH_USD = TradingPair<ETH, USD>;
pub type ETH_BTC = TradingPair<ETH, BTC>;
pub type BNB_USDT = TradingPair<BNB, USDT>;
pub type SOL_USDT = TradingPair<SOL, USDT>;
pub type XRP_USDT = TradingPair<XRP, USDT>;

/// Currency conversion rates (for reference/calculation)
pub trait CurrencyConverter {
    /// Convert amount from one currency to another
    fn convert<From: Currency, To: Currency>(
        amount: rust_decimal::Decimal,
        rate: rust_decimal::Decimal,
    ) -> rust_decimal::Decimal {
        amount * rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_currency_properties() {
        assert_eq!(USD::CODE, "USD");
        assert_eq!(USD::SYMBOL, "$");
        assert_eq!(USD::DECIMALS, 2);
        assert!(USD::IS_FIAT);
        assert!(USD::IS_STABLE);
        
        assert_eq!(BTC::CODE, "BTC");
        assert_eq!(BTC::DECIMALS, 8);
        assert!(!BTC::IS_FIAT);
        assert!(!BTC::IS_STABLE);
    }
    
    #[test]
    fn test_trading_pair() {
        type BTCUSD = TradingPair<BTC, USD>;
        
        assert_eq!(BTCUSD::symbol(), "BTC/USD");
        assert_eq!(BTCUSD::base_currency(), "BTC");
        assert_eq!(BTCUSD::quote_currency(), "USD");
        assert!(BTCUSD::is_fiat_pair());
        assert!(!BTCUSD::is_crypto_pair());
    }
    
    #[test]
    fn test_stable_pair_detection() {
        type BTCUSDT = TradingPair<BTC, USDT>;
        type ETHBTC = TradingPair<ETH, BTC>;
        
        assert!(BTCUSDT::is_stable_pair());
        assert!(!ETHBTC::is_stable_pair());
    }
}