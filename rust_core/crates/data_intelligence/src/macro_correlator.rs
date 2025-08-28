use mathematical_ops::correlation::calculate_correlation;
// MACRO ECONOMIC CORRELATOR - DEEP DIVE IMPLEMENTATION  
// Team: FULL TEAM - NO SIMPLIFICATIONS!
// Alex: "Correlate EVERY macro indicator with crypto movements!"
// Quinn: "Risk-adjusted correlation with regime detection"

use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MacroError {
    #[error("Data fetch error: {0}")]
    DataError(String),
    
    #[error("Correlation calculation failed: {0}")]
    CorrelationError(String),
}

pub type Result<T> = std::result::Result<T, MacroError>;

#[derive(Debug, Clone)]
pub struct MacroConfig {
    pub indicators: Vec<String>,
    pub correlation_window_days: u32,
    pub enable_regime_detection: bool,
}

impl Default for MacroConfig {
    fn default() -> Self {
        Self {
            indicators: vec![
                "DXY".to_string(),
                "VIX".to_string(),
                "GOLD".to_string(),
                "TNX".to_string(),  // 10-year yield
            ],
            correlation_window_days: 30,
            enable_regime_detection: true,
        }
    }
}

/// Macro Economic Correlator - tracks correlations with traditional markets
pub struct MacroEconomicCorrelator {
    config: MacroConfig,
    correlation_matrix: Arc<RwLock<HashMap<String, f64>>>,
}

impl MacroEconomicCorrelator {
    pub async fn new(config: MacroConfig) -> Result<Self> {
        Ok(Self {
            config,
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Calculate correlation between crypto and macro indicator
    pub use mathematical_ops::unified_calculations::calculate_correlation; // fn calculate_correlation(&self, crypto_data: &[f64], macro_data: &[f64]) -> f64 {
        if crypto_data.len() != macro_data.len() || crypto_data.is_empty() {
            return 0.0;
        }
        
        let n = crypto_data.len() as f64;
        let mean_crypto = crypto_data.iter().sum::<f64>() / n;
        let mean_macro = macro_data.iter().sum::<f64>() / n;
        
        let mut cov = 0.0;
        let mut var_crypto = 0.0;
        let mut var_macro = 0.0;
        
        for i in 0..crypto_data.len() {
            let diff_crypto = crypto_data[i] - mean_crypto;
            let diff_macro = macro_data[i] - mean_macro;
            cov += diff_crypto * diff_macro;
            var_crypto += diff_crypto * diff_crypto;
            var_macro += diff_macro * diff_macro;
        }
        
        if var_crypto > 0.0 && var_macro > 0.0 {
            cov / (var_crypto.sqrt() * var_macro.sqrt())
        } else {
            0.0
        }
    }
}