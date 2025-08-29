// ML EXTENSIONS - Adding missing fields and methods to existing structs
// Team: Morgan (ML Lead) + Full Team
// DEEP DIVE: Complete the ML infrastructure gaps

use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use crate::unified_types::SignalAction;

/// Extended fields for MLFeedbackSystem
/// TODO: Add docs
pub struct MLFeedbackSystemExt {
    pub predictions: VecDeque<PredictionRecord>,
    pub platt_a: Option<f64>,
    pub platt_b: Option<f64>,
    pub isotonic_mapping: Vec<(f64, f64)>,
    pub last_recalibration: DateTime<Utc>,
}

/// Extended fields for SHAPCalculator
/// TODO: Add docs
pub struct SHAPCalculatorExt {
    pub feature_names: Vec<String>,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub baseline_value: Option<f64>,
    pub shap_history: VecDeque<SHAPRecord>,
}

/// Extended fields for MarketAnalytics
/// TODO: Add docs
pub struct MarketAnalyticsExt {
    pub price_history: Vec<f64>,
    pub volume_history: Vec<f64>,
    pub obv_values: VecDeque<f64>,
    pub macd_values: VecDeque<f64>,
    pub rsi_values: VecDeque<f64>,
}

/// Prediction record for ML feedback
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PredictionRecord {
    pub timestamp: DateTime<Utc>,
    pub action: SignalAction,
    pub confidence: f64,
    pub actual_outcome: Option<f64>,
}

/// SHAP value record
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct SHAPRecord {
    pub timestamp: DateTime<Utc>,
    pub values: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Order level for order book
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: pub struct OrderLevel {
// ELIMINATED:     pub price: Decimal,
// ELIMINATED:     pub size: Decimal,
// ELIMINATED: }

/// Enhanced order book structure
#[derive(Debug, Clone, Default)]
/// TODO: Add docs
pub struct EnhancedOrderBook {
    pub bids: Vec<OrderLevel>,
    pub asks: Vec<OrderLevel>,
    pub last_update: DateTime<Utc>,
    pub exchange: String,
}

// Implement Default for extensions
impl Default for MLFeedbackSystemExt {
    fn default() -> Self {
        Self {
            predictions: VecDeque::new(),
            platt_a: Some(1.0),
            platt_b: Some(0.0),
            isotonic_mapping: Vec::new(),
            last_recalibration: Utc::now(),
        }
    }
}

impl Default for SHAPCalculatorExt {
    fn default() -> Self {
        Self {
            feature_names: Vec::new(),
            feature_means: Vec::new(),
            feature_weights: Vec::new(),
            baseline_value: Some(0.5),
            shap_history: VecDeque::new(),
        }
    }
}

impl Default for MarketAnalyticsExt {
    fn default() -> Self {
        Self {
            price_history: Vec::new(),
            volume_history: Vec::new(),
            obv_values: VecDeque::new(),
            macd_values: VecDeque::new(),
            rsi_values: VecDeque::new(),
        }
    }
}