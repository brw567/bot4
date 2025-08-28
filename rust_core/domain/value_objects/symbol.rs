// Domain Value Object: Symbol
// Immutable, no identity, represents a trading pair symbol
// Owner: Sam | Reviewer: Casey

use std::fmt;
use anyhow::{Result, bail};

/// Symbol value object - immutable representation of a trading pair
/// 
/// # Invariants
/// - Symbol must follow format: BASE/QUOTE (e.g., BTC/USDT)
/// - Both base and quote must be valid asset codes
/// - Symbol is case-insensitive but stored in uppercase
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    base: String,
    quote: String,
    raw: String,
}

impl Symbol {
    /// Create a new Symbol with validation
    /// 
    /// # Arguments
    /// * `value` - The symbol string (e.g., "BTC/USDT" or "ETH-USD")
    /// 
    /// # Returns
    /// * `Ok(Symbol)` if valid
    /// * `Err` if symbol format is invalid
    /// 
    /// # Example
    /// ```
    /// let symbol = Symbol::new("BTC/USDT")?;
    /// assert_eq!(symbol.base(), "BTC");
    /// assert_eq!(symbol.quote(), "USDT");
    /// ```
    pub fn new(value: &str) -> Result<Self> {
        // Normalize the symbol
        let normalized = value.to_uppercase();
        
        // Try different separators
        let parts: Vec<&str> = if normalized.contains('/') {
            normalized.split('/').collect()
        } else if normalized.contains('-') {
            normalized.split('-').collect()
        } else if normalized.contains('_') {
            normalized.split('_').collect()
        } else {
            // Try to split common patterns like BTCUSDT
            Self::split_combined(&normalized)?
        };
        
        if parts.len() != 2 {
            bail!("Invalid symbol format: {}", value);
        }
        
        let base = parts[0].to_string();
        let quote = parts[1].to_string();
        
        // Validate base and quote
        if base.is_empty() || quote.is_empty() {
            bail!("Symbol parts cannot be empty: {}", value);
        }
        
        if !Self::is_valid_asset_code(&base) {
            bail!("Invalid base asset: {}", base);
        }
        
        if !Self::is_valid_asset_code(&quote) {
            bail!("Invalid quote asset: {}", quote);
        }
        
        Ok(Symbol {
            base,
            quote,
            raw: format!("{}/{}", parts[0], parts[1]),
        })
    }
    
    /// Create a Symbol from base and quote assets
    pub fn from_parts(base: &str, quote: &str) -> Result<Self> {
        let symbol_str = format!("{}/{}", base, quote);
        Self::new(&symbol_str)
    }
    
    /// Get the base asset
    #[inline]
    pub fn base(&self) -> &str {
        &self.base
    }
    
    /// Get the quote asset
    #[inline]
    pub fn quote(&self) -> &str {
        &self.quote
    }
    
    /// Get the normalized symbol string
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.raw
    }
    
    /// Convert to exchange-specific format
    pub fn to_exchange_format(&self, exchange: &str) -> String {
        match exchange.to_lowercase().as_str() {
            "binance" => format!("{}{}", self.base, self.quote), // BTCUSDT
            "kraken" => format!("{}/{}", self.base, self.quote),  // BTC/USDT
            "coinbase" => format!("{}-{}", self.base, self.quote), // BTC-USDT
            _ => self.raw.clone(),
        }
    }
    
    /// Check if this is a stablecoin pair
    pub fn is_stablecoin_pair(&self) -> bool {
        const STABLECOINS: &[&str] = &["USDT", "USDC", "BUSD", "DAI", "TUSD", "UST"];
        STABLECOINS.contains(&self.quote.as_str()) || STABLECOINS.contains(&self.base.as_str())
    }
    
    /// Check if this is a fiat pair
    pub fn is_fiat_pair(&self) -> bool {
        const FIAT: &[&str] = &["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"];
        FIAT.contains(&self.quote.as_str())
    }
    
    /// Get the inverse symbol (swap base and quote)
    pub fn inverse(&self) -> Symbol {
        Symbol {
            base: self.quote.clone(),
            quote: self.base.clone(),
            raw: format!("{}/{}", self.quote, self.base),
        }
    }
    
    // Helper methods
    
    fn is_valid_asset_code(code: &str) -> bool {
        // Asset code should be 2-10 characters, alphanumeric
        code.len() >= 2 && 
        code.len() <= 10 && 
        code.chars().all(|c| c.is_ascii_alphanumeric())
    }
    
    fn split_combined(combined: &str) -> Result<Vec<&str>> {
        // Common patterns for combined symbols
        const COMMON_QUOTES: &[&str] = &["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB", "USD", "EUR"];
        
        for quote in COMMON_QUOTES {
            if combined.ends_with(quote) {
                let base = &combined[..combined.len() - quote.len()];
                if !base.is_empty() {
                    return Ok(vec![base, quote]);
                }
            }
        }
        
        bail!("Cannot parse combined symbol: {}", combined);
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn should_create_symbol_with_slash() {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "BTC");
        assert_eq!(symbol.quote(), "USDT");
        assert_eq!(symbol.as_str(), "BTC/USDT");
    }
    
    #[test]
    fn should_create_symbol_with_dash() {
        let symbol = Symbol::new("ETH-USD").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "ETH");
        assert_eq!(symbol.quote(), "USD");
    }
    
    #[test]
    fn should_create_symbol_with_underscore() {
        let symbol = Symbol::new("XRP_USDC").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "XRP");
        assert_eq!(symbol.quote(), "USDC");
    }
    
    #[test]
    fn should_parse_combined_symbol() {
        let symbol = Symbol::new("BTCUSDT").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "BTC");
        assert_eq!(symbol.quote(), "USDT");
    }
    
    #[test]
    fn should_normalize_to_uppercase() {
        let symbol = Symbol::new("btc/usdt").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "BTC");
        assert_eq!(symbol.quote(), "USDT");
    }
    
    #[test]
    fn should_create_from_parts() {
        let symbol = Symbol::from_parts("ETH", "BTC").expect("SAFETY: Add proper error handling");
        assert_eq!(symbol.base(), "ETH");
        assert_eq!(symbol.quote(), "BTC");
    }
    
    #[test]
    fn should_reject_empty_parts() {
        let result = Symbol::new("/USDT");
        assert!(result.is_err());
        
        let result = Symbol::new("BTC/");
        assert!(result.is_err());
    }
    
    #[test]
    fn should_reject_invalid_format() {
        let result = Symbol::new("INVALID");
        assert!(result.is_err());
    }
    
    #[test]
    fn should_convert_to_exchange_formats() {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        
        assert_eq!(symbol.to_exchange_format("binance"), "BTCUSDT");
        assert_eq!(symbol.to_exchange_format("kraken"), "BTC/USDT");
        assert_eq!(symbol.to_exchange_format("coinbase"), "BTC-USDT");
    }
    
    #[test]
    fn should_identify_stablecoin_pairs() {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        assert!(symbol.is_stablecoin_pair());
        
        let symbol = Symbol::new("ETH/BTC").expect("SAFETY: Add proper error handling");
        assert!(!symbol.is_stablecoin_pair());
    }
    
    #[test]
    fn should_identify_fiat_pairs() {
        let symbol = Symbol::new("BTC/USD").expect("SAFETY: Add proper error handling");
        assert!(symbol.is_fiat_pair());
        
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        assert!(!symbol.is_fiat_pair());
    }
    
    #[test]
    fn should_create_inverse_symbol() {
        let symbol = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        let inverse = symbol.inverse();
        
        assert_eq!(inverse.base(), "USDT");
        assert_eq!(inverse.quote(), "BTC");
        assert_eq!(inverse.as_str(), "USDT/BTC");
    }
    
    #[test]
    fn should_implement_equality() {
        let s1 = Symbol::new("BTC/USDT").expect("SAFETY: Add proper error handling");
        let s2 = Symbol::new("btc/usdt").expect("SAFETY: Add proper error handling");
        let s3 = Symbol::new("ETH/USDT").expect("SAFETY: Add proper error handling");
        
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }
}