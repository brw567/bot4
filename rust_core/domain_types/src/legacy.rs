//! # Legacy Type Compatibility Layer
//! 
//! Provides type aliases and wrapper types for backwards compatibility
//! during the migration from 158 duplicate types to canonical types.
//!
//! ## Migration Timeline
//! 1. Phase 1: Add this compatibility layer
//! 2. Phase 2: Update imports to use these aliases
//! 3. Phase 3: Gradually replace with canonical types
//! 4. Phase 4: Remove this module entirely
//!
//! ## Deprecation Strategy
//! All types in this module are marked deprecated to encourage migration

#![allow(deprecated)]

use crate::{Order as CanonicalOrder, Price as CanonicalPrice, Quantity as CanonicalQuantity};
use crate::{Trade as CanonicalTrade, Candle as CanonicalCandle};
use crate::{OrderBook as CanonicalOrderBook, Ticker as CanonicalTicker};

/// Legacy type alias for gradual migration
#[deprecated(since = "1.0.0", note = "Use domain_types::Order instead")]
pub type LegacyOrder = CanonicalOrder;

#[deprecated(since = "1.0.0", note = "Use domain_types::Price instead")]
pub type LegacyPrice = CanonicalPrice;

#[deprecated(since = "1.0.0", note = "Use domain_types::Quantity instead")]
pub type LegacyQuantity = CanonicalQuantity;

#[deprecated(since = "1.0.0", note = "Use domain_types::Trade instead")]
pub type LegacyTrade = CanonicalTrade;

#[deprecated(since = "1.0.0", note = "Use domain_types::Candle instead")]
pub type LegacyCandle = CanonicalCandle;

#[deprecated(since = "1.0.0", note = "Use domain_types::OrderBook instead")]
pub type LegacyOrderBook = CanonicalOrderBook;

#[deprecated(since = "1.0.0", note = "Use domain_types::Ticker instead")]
pub type LegacyTicker = CanonicalTicker;