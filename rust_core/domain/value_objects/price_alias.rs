//! Price type alias to canonical implementation
pub use domain_types::price::{Price, PriceError};

// Re-export for backward compatibility
pub type LegacyPrice = Price;
