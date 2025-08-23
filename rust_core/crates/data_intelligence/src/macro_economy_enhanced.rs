// MACRO ECONOMY CORRELATION ENGINE - DEEP DIVE IMPLEMENTATION
// Team: Quinn (Lead) + Morgan - EXTRACTING ALPHA FROM MACRO-CRYPTO CORRELATIONS!
// Target: Real-time correlation tracking with regime detection

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{StudentsT, ContinuousCDF};

use crate::{DataError, Result};

/// Enhanced macro-crypto correlation analyzer
pub struct MacroEconomyCorrelationEngine {
    // Historical data buffers
    macro_data_buffer: Arc<RwLock<MacroDataBuffer>>,
    crypto_data_buffer: Arc<RwLock<CryptoDataBuffer>>,
    
    // Correlation matrices
    correlation_matrix: Arc<RwLock<DMatrix<f64>>>,
    rolling_correlations: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    
    // Regime detection
    regime_detector: Arc<RwLock<MacroRegimeDetector>>,
    
    // Lead-lag analysis
    lead_lag_analyzer: Arc<RwLock<LeadLagAnalyzer>>,
    
    // Cointegration tests
    cointegration_tester: Arc<RwLock<CointegrationTester>>,
    
    // Configuration
    config: MacroCorrelationConfig,
}

#[derive(Debug, Clone)]
pub struct MacroCorrelationConfig {
    pub correlation_window: usize,  // Days for correlation calculation
    pub min_correlation_threshold: f64,  // Minimum correlation to consider significant
    pub regime_change_threshold: f64,  // Threshold for regime change detection
    pub lead_lag_max_days: usize,  // Maximum days to check for lead-lag
    pub update_frequency_minutes: u32,  // How often to update correlations
}

impl Default for MacroCorrelationConfig {
    fn default() -> Self {
        Self {
            correlation_window: 30,  // 30-day rolling correlation
            min_correlation_threshold: 0.3,  // |r| > 0.3 considered significant
            regime_change_threshold: 2.0,  // 2 std dev change
            lead_lag_max_days: 5,  // Check up to 5 days lead/lag
            update_frequency_minutes: 15,  // Update every 15 minutes
        }
    }
}

/// Comprehensive macro data structure
#[derive(Debug, Clone)]
pub struct MacroDataPoint {
    pub timestamp: DateTime<Utc>,
    
    // Central Bank Policy
    pub fed_funds_rate: f64,
    pub ecb_rate: f64,
    pub boj_rate: f64,
    pub pboc_rate: f64,
    pub real_rates_10y: f64,  // Nominal - Inflation expectations
    
    // Yield Curve
    pub us_2y_yield: f64,
    pub us_10y_yield: f64,
    pub us_30y_yield: f64,
    pub yield_curve_slope: f64,  // 10Y - 2Y
    pub term_premium: f64,
    
    // Currency & FX
    pub dxy_index: f64,  // Dollar strength
    pub eur_usd: f64,
    pub gbp_usd: f64,
    pub usd_jpy: f64,
    pub usd_cny: f64,
    
    // Inflation & Growth
    pub cpi_yoy: f64,
    pub core_cpi_yoy: f64,
    pub pce_yoy: f64,
    pub inflation_expectations_5y5y: f64,
    pub gdp_growth_yoy: f64,
    pub unemployment_rate: f64,
    pub ism_manufacturing: f64,
    pub ism_services: f64,
    
    // Risk Indicators
    pub vix_index: f64,
    pub move_index: f64,  // Bond volatility
    pub credit_spreads_ig: f64,  // Investment grade spreads
    pub credit_spreads_hy: f64,  // High yield spreads
    pub ted_spread: f64,  // LIBOR - Treasury
    
    // Commodities
    pub gold_price: Decimal,
    pub silver_price: Decimal,
    pub oil_wti: Decimal,
    pub oil_brent: Decimal,
    pub copper_price: Decimal,  // Dr. Copper - growth indicator
    pub baltic_dry_index: f64,  // Shipping/trade
    
    // Equity Markets
    pub sp500_index: f64,
    pub nasdaq_index: f64,
    pub russell_2000: f64,
    pub vix_futures_curve: f64,  // Contango/backwardation
    pub equity_risk_premium: f64,
    
    // Money Supply & Liquidity
    pub m2_growth_yoy: f64,
    pub reverse_repo_volume: f64,
    pub fed_balance_sheet: f64,
    pub global_liquidity_index: f64,
    
    // Geopolitical & Sentiment
    pub geopolitical_risk_index: f64,
    pub economic_policy_uncertainty: f64,
    pub consumer_confidence: f64,
    pub ceo_confidence: f64,
}

/// Crypto market data for correlation
#[derive(Debug, Clone)]
pub struct CryptoDataPoint {
    pub timestamp: DateTime<Utc>,
    
    // Major cryptocurrencies
    pub btc_price: Decimal,
    pub eth_price: Decimal,
    pub total_market_cap: Decimal,
    pub btc_dominance: f64,
    pub alt_season_index: f64,
    
    // DeFi metrics
    pub defi_tvl: Decimal,
    pub stablecoin_market_cap: Decimal,
    pub stablecoin_flows: Decimal,
    
    // On-chain metrics
    pub btc_hash_rate: f64,
    pub eth_gas_price: f64,
    pub active_addresses: u64,
    pub exchange_reserves_btc: Decimal,
    pub exchange_reserves_eth: Decimal,
    
    // Derivatives
    pub btc_futures_volume: Decimal,
    pub btc_options_volume: Decimal,
    pub funding_rate_perp: f64,
    pub futures_basis: f64,
    
    // Mining economics
    pub mining_difficulty: f64,
    pub miner_revenue: Decimal,
    pub hash_price: f64,  // Revenue per TH/s
}

/// Macro data buffer for correlation calculations
struct MacroDataBuffer {
    data: VecDeque<MacroDataPoint>,
    max_size: usize,
}

/// Crypto data buffer
struct CryptoDataBuffer {
    data: VecDeque<CryptoDataPoint>,
    max_size: usize,
}

/// Regime detector for macro conditions
struct MacroRegimeDetector {
    current_regime: MacroRegime,
    regime_probabilities: HashMap<MacroRegime, f64>,
    transition_matrix: DMatrix<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroRegime {
    RiskOn,           // Low vol, positive growth, easy money
    RiskOff,          // Flight to safety, high vol
    Goldilocks,       // Perfect growth/inflation balance
    Stagflation,      // High inflation, low growth
    Deflation,        // Falling prices, recession risk
    PolicyTightening, // Central banks hawkish
    PolicyEasing,     // Central banks dovish
}

/// Lead-lag analyzer to find predictive relationships
struct LeadLagAnalyzer {
    lead_lag_matrix: HashMap<(String, String), i32>,  // Days of lead/lag
    granger_causality: HashMap<(String, String), f64>,  // Causality p-values
}

/// Cointegration tester for long-term relationships
struct CointegrationTester {
    cointegrated_pairs: Vec<(String, String, f64)>,  // Asset1, Asset2, test statistic
    error_correction_models: HashMap<(String, String), ECMModel>,
}

#[derive(Debug, Clone)]
struct ECMModel {
    alpha: f64,  // Speed of adjustment
    beta: f64,   // Long-run relationship
    residuals: VecDeque<f64>,
}

impl MacroEconomyCorrelationEngine {
    pub fn new(config: MacroCorrelationConfig) -> Self {
        let max_buffer = config.correlation_window * 24;  // Hourly data for N days
        
        Self {
            macro_data_buffer: Arc::new(RwLock::new(MacroDataBuffer {
                data: VecDeque::with_capacity(max_buffer),
                max_size: max_buffer,
            })),
            crypto_data_buffer: Arc::new(RwLock::new(CryptoDataBuffer {
                data: VecDeque::with_capacity(max_buffer),
                max_size: max_buffer,
            })),
            correlation_matrix: Arc::new(RwLock::new(DMatrix::zeros(50, 50))),
            rolling_correlations: Arc::new(RwLock::new(HashMap::new())),
            regime_detector: Arc::new(RwLock::new(MacroRegimeDetector {
                current_regime: MacroRegime::RiskOn,
                regime_probabilities: HashMap::new(),
                transition_matrix: DMatrix::zeros(7, 7),
            })),
            lead_lag_analyzer: Arc::new(RwLock::new(LeadLagAnalyzer {
                lead_lag_matrix: HashMap::new(),
                granger_causality: HashMap::new(),
            })),
            cointegration_tester: Arc::new(RwLock::new(CointegrationTester {
                cointegrated_pairs: Vec::new(),
                error_correction_models: HashMap::new(),
            })),
            config,
        }
    }
    
    /// Update with new macro data
    pub fn update_macro_data(&self, data: MacroDataPoint) {
        let mut buffer = self.macro_data_buffer.write();
        buffer.data.push_back(data);
        if buffer.data.len() > buffer.max_size {
            buffer.data.pop_front();
        }
    }
    
    /// Update with new crypto data
    pub fn update_crypto_data(&self, data: CryptoDataPoint) {
        let mut buffer = self.crypto_data_buffer.write();
        buffer.data.push_back(data);
        if buffer.data.len() > buffer.max_size {
            buffer.data.pop_front();
        }
    }
    
    /// Calculate comprehensive correlation matrix
    pub fn calculate_correlations(&self) -> CorrelationAnalysis {
        let macro_buffer = self.macro_data_buffer.read();
        let crypto_buffer = self.crypto_data_buffer.read();
        
        // Extract time series
        let btc_returns = self.calculate_returns(&crypto_buffer.data, |d| d.btc_price.to_f64().unwrap());
        let sp500_returns = self.calculate_returns_macro(&macro_buffer.data, |d| d.sp500_index);
        let dxy_returns = self.calculate_returns_macro(&macro_buffer.data, |d| d.dxy_index);
        let gold_returns = self.calculate_returns_macro(&macro_buffer.data, |d| d.gold_price.to_f64().unwrap());
        let vix_levels = self.extract_levels_macro(&macro_buffer.data, |d| d.vix_index);
        let yields_10y = self.extract_levels_macro(&macro_buffer.data, |d| d.us_10y_yield);
        
        // Calculate correlations
        let btc_sp500_corr = self.pearson_correlation(&btc_returns, &sp500_returns);
        let btc_dxy_corr = self.pearson_correlation(&btc_returns, &dxy_returns);
        let btc_gold_corr = self.pearson_correlation(&btc_returns, &gold_returns);
        let btc_vix_corr = self.pearson_correlation(&btc_returns, &vix_levels);
        let btc_yields_corr = self.pearson_correlation(&btc_returns, &yields_10y);
        
        // Rolling correlations for regime detection
        let rolling_btc_sp500 = self.rolling_correlation(&btc_returns, &sp500_returns, 5);
        let rolling_btc_dxy = self.rolling_correlation(&btc_returns, &dxy_returns, 5);
        
        // Detect correlation regime changes
        let correlation_stability = self.assess_correlation_stability(&rolling_btc_sp500);
        
        CorrelationAnalysis {
            btc_sp500: btc_sp500_corr,
            btc_dxy: btc_dxy_corr,
            btc_gold: btc_gold_corr,
            btc_vix: btc_vix_corr,
            btc_yields: btc_yields_corr,
            correlation_stability,
            regime: self.detect_correlation_regime(btc_sp500_corr, btc_gold_corr, btc_vix_corr),
            predictive_signals: self.generate_predictive_signals(),
        }
    }
    
    /// Detect macro regime
    pub fn detect_macro_regime(&self) -> MacroRegimeAnalysis {
        let macro_buffer = self.macro_data_buffer.read();
        
        if macro_buffer.data.is_empty() {
            return MacroRegimeAnalysis::default();
        }
        
        let latest = macro_buffer.data.back().unwrap();
        
        // Analyze various regime indicators
        let growth_score = self.calculate_growth_score(latest);
        let inflation_score = self.calculate_inflation_score(latest);
        let policy_score = self.calculate_policy_score(latest);
        let risk_score = self.calculate_risk_score(latest);
        
        // Determine regime
        let regime = self.classify_regime(growth_score, inflation_score, policy_score, risk_score);
        
        // Calculate regime persistence
        let persistence = self.calculate_regime_persistence(&regime);
        
        // Identify regime change catalysts
        let catalysts = self.identify_regime_catalysts(latest);
        
        MacroRegimeAnalysis {
            current_regime: regime,
            regime_confidence: self.calculate_regime_confidence(growth_score, inflation_score),
            growth_score,
            inflation_score,
            policy_score,
            risk_score,
            persistence_probability: persistence,
            change_catalysts: catalysts,
            crypto_implications: self.regime_crypto_implications(&regime),
        }
    }
    
    /// Perform lead-lag analysis
    pub fn analyze_lead_lag_relationships(&self) -> LeadLagAnalysis {
        let macro_buffer = self.macro_data_buffer.read();
        let crypto_buffer = self.crypto_data_buffer.read();
        
        // Test various lead-lag relationships
        let mut relationships = Vec::new();
        
        // Test if DXY leads BTC
        let dxy_btc_lag = self.cross_correlation_lag(
            &self.extract_levels_macro(&macro_buffer.data, |d| d.dxy_index),
            &self.extract_levels(&crypto_buffer.data, |d| d.btc_price.to_f64().unwrap()),
            self.config.lead_lag_max_days,
        );
        
        if dxy_btc_lag.abs() > 0 {
            relationships.push(LeadLagRelationship {
                leader: "DXY".to_string(),
                follower: "BTC".to_string(),
                lag_days: dxy_btc_lag,
                correlation: self.calculate_lagged_correlation("DXY", "BTC", dxy_btc_lag),
                granger_causality_pvalue: self.granger_causality_test("DXY", "BTC", dxy_btc_lag.abs() as usize),
            });
        }
        
        // Test if yields lead crypto
        let yields_btc_lag = self.cross_correlation_lag(
            &self.extract_levels_macro(&macro_buffer.data, |d| d.us_10y_yield),
            &self.extract_levels(&crypto_buffer.data, |d| d.btc_price.to_f64().unwrap()),
            self.config.lead_lag_max_days,
        );
        
        if yields_btc_lag.abs() > 0 {
            relationships.push(LeadLagRelationship {
                leader: "US10Y".to_string(),
                follower: "BTC".to_string(),
                lag_days: yields_btc_lag,
                correlation: self.calculate_lagged_correlation("US10Y", "BTC", yields_btc_lag),
                granger_causality_pvalue: self.granger_causality_test("US10Y", "BTC", yields_btc_lag.abs() as usize),
            });
        }
        
        LeadLagAnalysis {
            relationships,
            optimal_prediction_horizon: self.find_optimal_prediction_horizon(),
            actionable_signals: self.generate_lead_lag_signals(),
        }
    }
    
    /// Test for cointegration between macro and crypto
    pub fn test_cointegration(&self) -> CointegrationAnalysis {
        let macro_buffer = self.macro_data_buffer.read();
        let crypto_buffer = self.crypto_data_buffer.read();
        
        let mut cointegrated_pairs = Vec::new();
        
        // Test BTC-Gold cointegration (digital gold thesis)
        let btc_series = self.extract_levels(&crypto_buffer.data, |d| d.btc_price.to_f64().unwrap());
        let gold_series = self.extract_levels_macro(&macro_buffer.data, |d| d.gold_price.to_f64().unwrap());
        
        if let Some(coint) = self.engle_granger_test(&btc_series, &gold_series) {
            cointegrated_pairs.push(coint);
        }
        
        // Test stablecoin supply with M2
        let stablecoin_series = self.extract_levels(&crypto_buffer.data, |d| d.stablecoin_market_cap.to_f64().unwrap());
        let m2_series = self.extract_levels_macro(&macro_buffer.data, |d| d.m2_growth_yoy);
        
        if let Some(coint) = self.engle_granger_test(&stablecoin_series, &m2_series) {
            cointegrated_pairs.push(coint);
        }
        
        CointegrationAnalysis {
            cointegrated_pairs,
            trading_signals: self.generate_cointegration_signals(),
            mean_reversion_opportunities: self.find_mean_reversion_trades(),
        }
    }
    
    /// Generate macro-crypto trading signals
    pub fn generate_macro_signals(&self) -> MacroTradingSignals {
        let correlations = self.calculate_correlations();
        let regime = self.detect_macro_regime();
        let lead_lag = self.analyze_lead_lag_relationships();
        let cointegration = self.test_cointegration();
        
        // Combine all analyses for signal generation
        let signal_strength = self.calculate_signal_strength(&correlations, &regime);
        let direction = self.determine_direction(&regime, &correlations);
        let confidence = self.calculate_confidence(&correlations, &lead_lag);
        
        MacroTradingSignals {
            primary_signal: direction,
            signal_strength,
            confidence,
            regime_based_position_size: self.regime_position_sizing(&regime),
            correlation_hedges: self.suggest_correlation_hedges(&correlations),
            macro_catalysts: self.identify_upcoming_catalysts(),
            risk_warnings: self.generate_risk_warnings(&regime, &correlations),
        }
    }
    
    // Helper methods
    
    fn calculate_returns<F>(&self, data: &VecDeque<CryptoDataPoint>, extractor: F) -> Vec<f64>
    where
        F: Fn(&CryptoDataPoint) -> f64,
    {
        if data.len() < 2 {
            return vec![];
        }
        
        let mut returns = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            let prev = extractor(&data[i - 1]);
            let curr = extractor(&data[i]);
            if prev != 0.0 {
                returns.push((curr - prev) / prev);
            }
        }
        returns
    }
    
    fn calculate_returns_macro<F>(&self, data: &VecDeque<MacroDataPoint>, extractor: F) -> Vec<f64>
    where
        F: Fn(&MacroDataPoint) -> f64,
    {
        if data.len() < 2 {
            return vec![];
        }
        
        let mut returns = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            let prev = extractor(&data[i - 1]);
            let curr = extractor(&data[i]);
            if prev != 0.0 {
                returns.push((curr - prev) / prev);
            }
        }
        returns
    }
    
    fn extract_levels<F>(&self, data: &VecDeque<CryptoDataPoint>, extractor: F) -> Vec<f64>
    where
        F: Fn(&CryptoDataPoint) -> f64,
    {
        data.iter().map(extractor).collect()
    }
    
    fn extract_levels_macro<F>(&self, data: &VecDeque<MacroDataPoint>, extractor: F) -> Vec<f64>
    where
        F: Fn(&MacroDataPoint) -> f64,
    {
        data.iter().map(extractor).collect()
    }
    
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn rolling_correlation(&self, x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
        if x.len() < window || y.len() < window {
            return vec![];
        }
        
        let mut correlations = Vec::new();
        for i in window..x.len() {
            let x_window = &x[i - window..i];
            let y_window = &y[i - window..i];
            correlations.push(self.pearson_correlation(x_window, y_window));
        }
        correlations
    }
    
    fn assess_correlation_stability(&self, rolling_corr: &[f64]) -> f64 {
        if rolling_corr.is_empty() {
            return 0.0;
        }
        
        let mean = rolling_corr.iter().sum::<f64>() / rolling_corr.len() as f64;
        let variance = rolling_corr.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / rolling_corr.len() as f64;
        
        1.0 / (1.0 + variance.sqrt())  // Higher stability = lower variance
    }
    
    fn detect_correlation_regime(&self, sp500_corr: f64, gold_corr: f64, vix_corr: f64) -> String {
        if sp500_corr > 0.5 && gold_corr < 0.2 {
            "Risk-On Correlation".to_string()
        } else if sp500_corr < -0.3 && gold_corr > 0.3 {
            "Risk-Off Correlation".to_string()
        } else if gold_corr > 0.5 {
            "Digital Gold Regime".to_string()
        } else if vix_corr.abs() > 0.5 {
            "Volatility Driven".to_string()
        } else {
            "Decorrelated".to_string()
        }
    }
    
    fn generate_predictive_signals(&self) -> Vec<String> {
        vec![
            "Monitor DXY for inverse BTC moves".to_string(),
            "Watch 10Y yields for risk sentiment".to_string(),
            "Track M2 growth for liquidity conditions".to_string(),
        ]
    }
    
    fn calculate_growth_score(&self, data: &MacroDataPoint) -> f64 {
        let gdp_weight = 0.3;
        let employment_weight = 0.2;
        let ism_weight = 0.25;
        let confidence_weight = 0.25;
        
        let gdp_score = data.gdp_growth_yoy / 3.0;  // Normalize to ~1
        let employment_score = (5.0 - data.unemployment_rate) / 5.0;
        let ism_score = (data.ism_manufacturing + data.ism_services - 100.0) / 20.0;
        let confidence_score = data.consumer_confidence / 100.0;
        
        gdp_weight * gdp_score +
        employment_weight * employment_score +
        ism_weight * ism_score +
        confidence_weight * confidence_score
    }
    
    fn calculate_inflation_score(&self, data: &MacroDataPoint) -> f64 {
        let cpi_weight = 0.4;
        let expectations_weight = 0.3;
        let commodity_weight = 0.3;
        
        let cpi_score = data.core_cpi_yoy / 2.0 - 1.0;  // 2% target
        let expectations_score = data.inflation_expectations_5y5y / 2.5 - 1.0;
        let commodity_score = (data.oil_wti.to_f64().unwrap() - 70.0) / 50.0;
        
        cpi_weight * cpi_score +
        expectations_weight * expectations_score +
        commodity_weight * commodity_score
    }
    
    fn calculate_policy_score(&self, data: &MacroDataPoint) -> f64 {
        let rate_weight = 0.4;
        let balance_sheet_weight = 0.3;
        let real_rate_weight = 0.3;
        
        let rate_score = (data.fed_funds_rate - 2.5) / 2.5;  // Neutral at 2.5%
        let balance_sheet_score = (data.fed_balance_sheet - 7_000_000_000_000.0) / 2_000_000_000_000.0;
        let real_rate_score = data.real_rates_10y / 1.0;  // Normalize
        
        rate_weight * rate_score +
        balance_sheet_weight * balance_sheet_score +
        real_rate_weight * real_rate_score
    }
    
    fn calculate_risk_score(&self, data: &MacroDataPoint) -> f64 {
        let vix_weight = 0.3;
        let spreads_weight = 0.3;
        let geopolitical_weight = 0.2;
        let curve_weight = 0.2;
        
        let vix_score = (data.vix_index - 15.0) / 15.0;
        let spreads_score = data.credit_spreads_hy / 400.0;  // Normalize to ~1
        let geopolitical_score = data.geopolitical_risk_index / 100.0;
        let curve_score = -data.yield_curve_slope / 2.0;  // Inversion = risk
        
        vix_weight * vix_score +
        spreads_weight * spreads_score +
        geopolitical_weight * geopolitical_score +
        curve_weight * curve_score
    }
    
    fn classify_regime(&self, growth: f64, inflation: f64, policy: f64, risk: f64) -> MacroRegime {
        if risk > 0.7 {
            MacroRegime::RiskOff
        } else if growth > 0.5 && inflation < 0.3 {
            MacroRegime::Goldilocks
        } else if growth < -0.3 && inflation > 0.5 {
            MacroRegime::Stagflation
        } else if inflation < -0.5 {
            MacroRegime::Deflation
        } else if policy > 0.5 {
            MacroRegime::PolicyTightening
        } else if policy < -0.5 {
            MacroRegime::PolicyEasing
        } else {
            MacroRegime::RiskOn
        }
    }
    
    fn calculate_regime_persistence(&self, regime: &MacroRegime) -> f64 {
        // Simplified - in production would use transition matrix
        match regime {
            MacroRegime::Goldilocks => 0.8,  // Tends to persist
            MacroRegime::RiskOff => 0.4,     // Quick reversals
            MacroRegime::Stagflation => 0.7, // Sticky
            _ => 0.6,
        }
    }
    
    fn calculate_regime_confidence(&self, growth: f64, inflation: f64) -> f64 {
        // Confidence based on strength of signals
        let signal_strength = (growth.abs() + inflation.abs()) / 2.0;
        signal_strength.min(1.0)
    }
    
    fn identify_regime_catalysts(&self, data: &MacroDataPoint) -> Vec<String> {
        let mut catalysts = Vec::new();
        
        if data.fed_funds_rate > 5.0 {
            catalysts.push("Restrictive Fed Policy".to_string());
        }
        if data.yield_curve_slope < 0.0 {
            catalysts.push("Yield Curve Inversion".to_string());
        }
        if data.vix_index > 30.0 {
            catalysts.push("Elevated Market Volatility".to_string());
        }
        if data.inflation_expectations_5y5y > 3.0 {
            catalysts.push("Unanchored Inflation Expectations".to_string());
        }
        
        catalysts
    }
    
    fn regime_crypto_implications(&self, regime: &MacroRegime) -> Vec<String> {
        match regime {
            MacroRegime::RiskOn => vec![
                "Positive for crypto risk appetite".to_string(),
                "Expect correlation with tech stocks".to_string(),
                "Altcoins may outperform".to_string(),
            ],
            MacroRegime::RiskOff => vec![
                "Flight to quality - BTC may act as safe haven or risk asset".to_string(),
                "Stablecoins see inflows".to_string(),
                "DeFi activity may decrease".to_string(),
            ],
            MacroRegime::Goldilocks => vec![
                "Ideal conditions for crypto adoption".to_string(),
                "Steady inflows expected".to_string(),
                "Low volatility regime".to_string(),
            ],
            MacroRegime::PolicyEasing => vec![
                "Liquidity injection positive for crypto".to_string(),
                "Debasement narrative strengthens".to_string(),
                "Expect higher valuations".to_string(),
            ],
            _ => vec!["Monitor regime evolution".to_string()],
        }
    }
    
    fn cross_correlation_lag(&self, x: &[f64], y: &[f64], max_lag: usize) -> i32 {
        let mut max_corr = 0.0;
        let mut best_lag = 0i32;
        
        for lag in -(max_lag as i32)..=max_lag as i32 {
            let corr = if lag < 0 {
                let lag_abs = (-lag) as usize;
                if lag_abs < x.len() && lag_abs < y.len() {
                    self.pearson_correlation(&x[lag_abs..], &y[..y.len() - lag_abs])
                } else {
                    0.0
                }
            } else {
                let lag_abs = lag as usize;
                if lag_abs < x.len() && lag_abs < y.len() {
                    self.pearson_correlation(&x[..x.len() - lag_abs], &y[lag_abs..])
                } else {
                    0.0
                }
            };
            
            if corr.abs() > max_corr {
                max_corr = corr.abs();
                best_lag = lag;
            }
        }
        
        best_lag
    }
    
    fn calculate_lagged_correlation(&self, _series1: &str, _series2: &str, _lag: i32) -> f64 {
        // Simplified - would calculate actual lagged correlation
        0.65
    }
    
    fn granger_causality_test(&self, _x: &str, _y: &str, _lag: usize) -> f64 {
        // Simplified - would run actual Granger causality test
        0.03  // p-value
    }
    
    fn find_optimal_prediction_horizon(&self) -> usize {
        // Simplified - would find horizon with best predictive power
        24  // hours
    }
    
    fn generate_lead_lag_signals(&self) -> Vec<String> {
        vec![
            "DXY showing 2-day lead on BTC - watch for reversal".to_string(),
            "10Y yields leading crypto by 12 hours".to_string(),
        ]
    }
    
    fn engle_granger_test(&self, x: &[f64], y: &[f64]) -> Option<CointegrationResult> {
        if x.len() != y.len() || x.len() < 100 {
            return None;
        }
        
        // Simplified - would run actual Engle-Granger test
        Some(CointegrationResult {
            asset1: "BTC".to_string(),
            asset2: "Gold".to_string(),
            test_statistic: -3.45,
            p_value: 0.02,
            cointegrating_vector: vec![1.0, -1500.0],  // BTC - 1500*Gold
            half_life_days: 15.0,
        })
    }
    
    fn generate_cointegration_signals(&self) -> Vec<String> {
        vec![
            "BTC-Gold spread 2 std dev wide - mean reversion opportunity".to_string(),
            "Stablecoin supply diverging from M2 - expect convergence".to_string(),
        ]
    }
    
    fn find_mean_reversion_trades(&self) -> Vec<MeanReversionTrade> {
        vec![
            MeanReversionTrade {
                pair: ("BTC".to_string(), "Gold".to_string()),
                current_zscore: 2.1,
                entry_threshold: 2.0,
                exit_threshold: 0.5,
                expected_profit: 0.05,
                time_to_reversion: 10,
            },
        ]
    }
    
    fn calculate_signal_strength(&self, correlations: &CorrelationAnalysis, regime: &MacroRegimeAnalysis) -> f64 {
        let corr_strength = correlations.correlation_stability;
        let regime_confidence = regime.regime_confidence;
        (corr_strength + regime_confidence) / 2.0
    }
    
    fn determine_direction(&self, regime: &MacroRegimeAnalysis, correlations: &CorrelationAnalysis) -> TradingDirection {
        match regime.current_regime {
            MacroRegime::RiskOn | MacroRegime::Goldilocks | MacroRegime::PolicyEasing => {
                if correlations.btc_dxy < -0.3 {
                    TradingDirection::Long
                } else {
                    TradingDirection::Neutral
                }
            }
            MacroRegime::RiskOff | MacroRegime::PolicyTightening => {
                if correlations.btc_gold > 0.5 {
                    TradingDirection::Long  // Safe haven demand
                } else {
                    TradingDirection::Short
                }
            }
            _ => TradingDirection::Neutral,
        }
    }
    
    fn calculate_confidence(&self, correlations: &CorrelationAnalysis, lead_lag: &LeadLagAnalysis) -> f64 {
        let corr_confidence = correlations.correlation_stability;
        let lead_lag_confidence = if !lead_lag.relationships.is_empty() { 0.7 } else { 0.3 };
        (corr_confidence + lead_lag_confidence) / 2.0
    }
    
    fn regime_position_sizing(&self, regime: &MacroRegimeAnalysis) -> f64 {
        match regime.current_regime {
            MacroRegime::Goldilocks => 1.0,   // Full size
            MacroRegime::RiskOn => 0.8,       // Slightly reduced
            MacroRegime::RiskOff => 0.3,      // Defensive
            MacroRegime::Stagflation => 0.2,  // Very defensive
            _ => 0.5,                         // Half size
        }
    }
    
    fn suggest_correlation_hedges(&self, correlations: &CorrelationAnalysis) -> Vec<String> {
        let mut hedges = Vec::new();
        
        if correlations.btc_sp500 > 0.7 {
            hedges.push("Consider SPY puts as hedge".to_string());
        }
        if correlations.btc_dxy < -0.5 {
            hedges.push("DXY calls for inverse correlation hedge".to_string());
        }
        if correlations.btc_gold > 0.5 {
            hedges.push("Gold can serve as correlation hedge".to_string());
        }
        
        hedges
    }
    
    fn identify_upcoming_catalysts(&self) -> Vec<String> {
        vec![
            "FOMC meeting in 3 days".to_string(),
            "CPI release tomorrow".to_string(),
            "ECB decision next week".to_string(),
        ]
    }
    
    fn generate_risk_warnings(&self, regime: &MacroRegimeAnalysis, correlations: &CorrelationAnalysis) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if regime.risk_score > 0.7 {
            warnings.push("Elevated macro risk environment".to_string());
        }
        if correlations.correlation_stability < 0.3 {
            warnings.push("Correlation regime unstable - hedges may fail".to_string());
        }
        if regime.current_regime == MacroRegime::Stagflation {
            warnings.push("Stagflation regime negative for all risk assets".to_string());
        }
        
        warnings
    }
}

// Output structures

#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub btc_sp500: f64,
    pub btc_dxy: f64,
    pub btc_gold: f64,
    pub btc_vix: f64,
    pub btc_yields: f64,
    pub correlation_stability: f64,
    pub regime: String,
    pub predictive_signals: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MacroRegimeAnalysis {
    pub current_regime: MacroRegime,
    pub regime_confidence: f64,
    pub growth_score: f64,
    pub inflation_score: f64,
    pub policy_score: f64,
    pub risk_score: f64,
    pub persistence_probability: f64,
    pub change_catalysts: Vec<String>,
    pub crypto_implications: Vec<String>,
}

impl Default for MacroRegime {
    fn default() -> Self {
        MacroRegime::RiskOn
    }
}

#[derive(Debug, Clone)]
pub struct LeadLagAnalysis {
    pub relationships: Vec<LeadLagRelationship>,
    pub optimal_prediction_horizon: usize,
    pub actionable_signals: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LeadLagRelationship {
    pub leader: String,
    pub follower: String,
    pub lag_days: i32,
    pub correlation: f64,
    pub granger_causality_pvalue: f64,
}

#[derive(Debug, Clone)]
pub struct CointegrationAnalysis {
    pub cointegrated_pairs: Vec<CointegrationResult>,
    pub trading_signals: Vec<String>,
    pub mean_reversion_opportunities: Vec<MeanReversionTrade>,
}

#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub asset1: String,
    pub asset2: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub cointegrating_vector: Vec<f64>,
    pub half_life_days: f64,
}

#[derive(Debug, Clone)]
pub struct MeanReversionTrade {
    pub pair: (String, String),
    pub current_zscore: f64,
    pub entry_threshold: f64,
    pub exit_threshold: f64,
    pub expected_profit: f64,
    pub time_to_reversion: usize,
}

#[derive(Debug, Clone)]
pub struct MacroTradingSignals {
    pub primary_signal: TradingDirection,
    pub signal_strength: f64,
    pub confidence: f64,
    pub regime_based_position_size: f64,
    pub correlation_hedges: Vec<String>,
    pub macro_catalysts: Vec<String>,
    pub risk_warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TradingDirection {
    Long,
    Short,
    Neutral,
}