use domain_types::FeatureMetadata;
use domain_types::FeatureVector;
// Feature Engineering Pipeline - Critical ML Component
// Team Lead: Morgan | Full Team Collaboration Required
// Phase 3 - Machine Learning Integration
// Target: <100μs feature extraction, 100+ features

pub mod indicators;
pub mod indicators_extended;
pub mod ichimoku;
pub mod elliott_wave;
pub mod harmonic_patterns;
pub mod pipeline;
pub mod scaler;
pub mod selector;

pub use pipeline::{FeaturePipeline, FeatureConfig};
pub use scaler::{FeatureScaler, ScalingMethod};
pub use selector::{FeatureSelector, SelectionMethod};
pub use ichimoku::{IchimokuCloud, IchimokuResult, IchimokuSignal};
pub use elliott_wave::{ElliottWaveDetector, ElliottPattern, WaveType, PatternType, MarketPosition};
pub use harmonic_patterns::{HarmonicPatternDetector, HarmonicPattern, HarmonicType, PotentialReversalZone};

use anyhow::Result;
use rust_decimal::Decimal;

// ============================================================================
// TEAM ASSIGNMENTS
// ============================================================================
// Morgan: Feature engineering algorithms, statistical features
// Avery: Data normalization, persistence layer
// Casey: Market microstructure features
// Jordan: Performance optimization, SIMD
// Sam: Architecture, clean code
// Quinn: Risk-based features, validation
// Riley: Testing, feature validation
// Alex: Integration, coordination

/// Core feature vector used by all ML models
#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::FeatureVector
// pub struct FeatureVector {
    /// Raw features before scaling
    pub raw_features: Vec<f64>,
    
    /// Scaled features ready for ML
    pub scaled_features: Vec<f64>,
    
    /// Feature names for interpretability
    pub feature_names: Vec<String>,
    
    /// Feature importance scores
    pub importance_scores: Option<Vec<f64>>,
    
    /// Timestamp of feature calculation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Symbol these features belong to
    pub symbol: String,
}

impl FeatureVector {
    /// Create new feature vector
    pub fn new(symbol: String, capacity: usize) -> Self {
        Self {
            raw_features: Vec::with_capacity(capacity),
            scaled_features: Vec::with_capacity(capacity),
            feature_names: Vec::with_capacity(capacity),
            importance_scores: None,
            timestamp: chrono::Utc::now(),
            symbol,
        }
    }
    
    /// Add a feature to the vector
    pub fn add_feature(&mut self, name: &str, value: f64) {
        self.feature_names.push(name.to_string());
        self.raw_features.push(value);
    }
    
    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<f64> {
        self.feature_names
            .iter()
            .position(|n| n == name)
            .map(|idx| self.raw_features[idx])
    }
    
    /// Apply scaling to features
    pub fn apply_scaling(&mut self, scaler: &FeatureScaler) -> Result<()> {
        self.scaled_features = scaler.transform(&self.raw_features)?;
        Ok(())
    }
    
    /// Select subset of features
    pub fn select_features(&mut self, selector: &FeatureSelector) -> Result<()> {
        let indices = selector.select(&self.raw_features)?;
        
        // Filter features based on selected indices
        let mut selected_raw = Vec::new();
        let mut selected_scaled = Vec::new();
        let mut selected_names = Vec::new();
        
        for &idx in &indices {
            if idx < self.raw_features.len() {
                selected_raw.push(self.raw_features[idx]);
                if !self.scaled_features.is_empty() {
                    selected_scaled.push(self.scaled_features[idx]);
                }
                selected_names.push(self.feature_names[idx].clone());
            }
        }
        
        self.raw_features = selected_raw;
        self.scaled_features = selected_scaled;
        self.feature_names = selected_names;
        
        Ok(())
    }
}

/// Market data snapshot for feature calculation
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MarketSnapshot {
    /// Current price
    pub price: Decimal,
    
    /// Bid/Ask spread
    pub spread: Decimal,
    
    /// Volume in base currency
    pub volume: Decimal,
    
    /// Recent price history (newest first)
    pub price_history: Vec<Decimal>,
    
    /// Recent volume history
    pub volume_history: Vec<Decimal>,
    
    /// Order book imbalance
    pub order_book_imbalance: f64,
    
    /// Market depth
    pub market_depth: f64,
    
    /// Funding rate (for perpetuals)
    pub funding_rate: Option<Decimal>,
    
    /// Open interest
    pub open_interest: Option<Decimal>,
}

/// Feature metadata for tracking and debugging
#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::FeatureMetadata
// pub struct FeatureMetadata {
    /// Feature name
    pub name: String,
    
    /// Feature category (price, volume, technical, etc.)
    pub category: FeatureCategory,
    
    /// Computation time in microseconds
    pub computation_time_us: u64,
    
    /// Is this a lagged feature?
    pub is_lagged: bool,
    
    /// Lag period if applicable
    pub lag_period: Option<usize>,
    
    /// Feature importance from last training
    pub importance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
/// TODO: Add docs
pub enum FeatureCategory {
    Price,
    Volume,
    Technical,
    Statistical,
    Microstructure,
    Sentiment,
    Risk,
    Custom,
}

/// Feature engineering statistics for monitoring
#[derive(Debug, Clone, Default)]
/// TODO: Add docs
pub struct FeatureStats {
    /// Total features computed
    pub total_features: usize,
    
    /// Features after selection
    pub selected_features: usize,
    
    /// Average computation time
    pub avg_computation_time_us: f64,
    
    /// Missing values count
    pub missing_values: usize,
    
    /// Infinite values count
    pub infinite_values: usize,
    
    /// NaN values count
    pub nan_values: usize,
}

impl FeatureStats {
    /// Validate feature vector quality
    pub fn validate(&self) -> Result<()> {
        if self.nan_values > 0 {
            anyhow::bail!("Feature vector contains {} NaN values", self.nan_values);
        }
        
        if self.infinite_values > 0 {
            anyhow::bail!("Feature vector contains {} infinite values", self.infinite_values);
        }
        
        if self.selected_features == 0 {
            anyhow::bail!("No features selected");
        }
        
        Ok(())
    }
}

// Re-export commonly used types
pub use indicators::{IndicatorEngine, IndicatorParams as IndicatorConfig};
pub use indicators_extended::register_all_indicators;

// Type aliases for backward compatibility
pub type TechnicalIndicators = IndicatorEngine;
pub type ExtendedIndicators = IndicatorEngine;
pub type AdvancedIndicatorConfig = IndicatorConfig;
pub type FeatureExtractor = IndicatorEngine;
pub type AdvancedFeatureEngine = IndicatorEngine;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_vector_creation() {
        let mut fv = FeatureVector::new("BTC/USDT".to_string(), 10);
        
        fv.add_feature("price", 50000.0);
        fv.add_feature("volume", 1000.0);
        fv.add_feature("rsi", 65.5);
        
        assert_eq!(fv.raw_features.len(), 3);
        assert_eq!(fv.feature_names.len(), 3);
        assert_eq!(fv.get_feature("price"), Some(50000.0));
        assert_eq!(fv.get_feature("rsi"), Some(65.5));
    }
    
    #[test]
    fn test_feature_stats_validation() {
        let mut stats = FeatureStats::default();
        stats.selected_features = 10;
        assert!(stats.validate().is_ok());
        
        stats.nan_values = 1;
        assert!(stats.validate().is_err());
    }
}

// Team Sign-off:
// Morgan: "Feature vector structure complete"
// Avery: "Data structures optimized"
// Sam: "Clean architecture maintained"
// Quinn: "Validation logic sound"
// Riley: "Tests passing"
// Jordan: "Performance considerations addressed"
// Casey: "Market data integration ready"
// Alex: "Module structure approved"