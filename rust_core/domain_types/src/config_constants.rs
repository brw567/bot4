//! # CONFIGURATION CONSTANTS - No More Hardcoding
//! Team: "All magic numbers must be configurable"

use once_cell::sync::Lazy;
use std::env;

/// Risk management constants
/// TODO: Add docs
pub struct RiskConfig {
    pub max_position_pct: f64,
    pub max_daily_loss_pct: f64,
    pub max_leverage: f64,
    pub position_limit: u64,
    pub order_size_limit: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_pct: env::var("MAX_POSITION_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.02),
            max_daily_loss_pct: env::var("MAX_DAILY_LOSS_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.02),
            max_leverage: env::var("MAX_LEVERAGE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3.0),
            position_limit: env::var("POSITION_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100_000),
            order_size_limit: env::var("ORDER_SIZE_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
        }
    }
}

pub static RISK_CONFIG: Lazy<RiskConfig> = Lazy::new(RiskConfig::default);

/// Time window constants
/// TODO: Add docs
pub struct TimeConfig {
    pub default_window_seconds: u64,
    pub day_seconds: u64,
    pub cache_ttl_seconds: u64,
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            default_window_seconds: env::var("DEFAULT_WINDOW_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
            day_seconds: 86400,
            cache_ttl_seconds: env::var("CACHE_TTL_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
        }
    }
}

pub static TIME_CONFIG: Lazy<TimeConfig> = Lazy::new(TimeConfig::default);

// Team: "Configuration over hardcoding!"
