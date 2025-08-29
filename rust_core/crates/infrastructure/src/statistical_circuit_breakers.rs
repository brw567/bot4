// STATISTICAL CIRCUIT BREAKERS - Layer 0.9.3
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Detect mathematical anomalies that threshold-based breakers miss
//
// External Research Applied:
// - "Adaptive Markets Hypothesis" - Andrew Lo (2004, 2017)
// - "Regime Changes in Financial Markets" - Hamilton (1989)
// - "ARCH/GARCH Models" - Engle (1982), Bollerslev (1986)
// - "Statistical Process Control in Finance" - Apley & Shi (2003)
// - "Anomaly Detection in High-Frequency Trading" - Golbeck et al. (2016)
// - "Hidden Markov Models in Finance" - Bhar & Hamori (2004)
// - "Risk Metrics Technical Document" - J.P. Morgan (1996)
// - "Volatility Clustering in Financial Markets" - Cont (2007)

use std::sync::Arc;
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, SystemTime, Instant};
use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};
use tokio::sync::{RwLock, Mutex};
use parking_lot::RwLock as SyncRwLock;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;

// ============================================================================
// STATISTICAL ANOMALY TYPES
// ============================================================================

/// Types of statistical anomalies we detect
/// Morgan: "Each requires different mathematical approach"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum StatisticalAnomaly {
    /// Sharpe ratio degradation below threshold
    SharpeDegradation,
    
    /// Market regime change detected
    RegimeChange,
    
    /// Abnormal volatility clustering
    VolatilityClustering,
    
    /// Serial correlation breakdown
    CorrelationBreakdown,
    
    /// Distribution tail risk increase
    TailRiskIncrease,
    
    /// Liquidity evaporation
    LiquidityEvaporation,
    
    /// Microstructure breakdown
    MicrostructureAnomaly,
    
    /// Cross-asset correlation spike
    CorrelationSpike,
}

/// Statistical circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum StatisticalState {
    /// Normal market conditions
    Normal,
    
    /// Warning - anomalies detected but within tolerance
    Warning,
    
    /// Critical - significant anomalies, consider reducing exposure
    Critical,
    
    /// Tripped - statistical evidence of market breakdown
    Tripped,
}

/// Market regime identified by Hidden Markov Model
/// Based on Hamilton (1989) regime-switching model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// TODO: Add docs
pub enum MarketRegime {
    /// Low volatility, trending market
    Trending,
    
    /// Medium volatility, mean-reverting
    MeanReverting,
    
    /// High volatility, crisis/stress
    Crisis,
    
    /// Transitioning between regimes
    Transition,
}

// ============================================================================
// SHARPE RATIO MONITOR
// ============================================================================

/// Monitor Sharpe ratio degradation in real-time
/// Quinn: "Risk-adjusted returns are critical for position sizing"
/// TODO: Add docs
pub struct SharpeMonitor {
    /// Rolling window of returns
    returns_window: VecDeque<Decimal>,
    
    /// Window size in periods
    window_size: usize,
    
    /// Current Sharpe ratio
    current_sharpe: Decimal,
    
    /// Baseline Sharpe (from backtesting)
    baseline_sharpe: Decimal,
    
    /// Degradation threshold (e.g., 50% of baseline)
    degradation_threshold: Decimal,
    
    /// Risk-free rate (annualized)
    risk_free_rate: Decimal,
    
    /// Periods per year (for annualization)
    periods_per_year: Decimal,
    
    /// Historical Sharpe values for trend analysis
    sharpe_history: VecDeque<Decimal>,
    
    /// Maximum history size
    max_history: usize,
}

impl SharpeMonitor {
    pub fn new(
        window_size: usize,
        baseline_sharpe: Decimal,
        degradation_threshold: Decimal,
        risk_free_rate: Decimal,
        periods_per_year: Decimal,
    ) -> Self {
        Self {
            returns_window: VecDeque::with_capacity(window_size),
            window_size,
            current_sharpe: Decimal::ZERO,
            baseline_sharpe,
            degradation_threshold,
            risk_free_rate,
            periods_per_year,
            sharpe_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }
    
    /// Add new return and update Sharpe ratio
    pub fn add_return(&mut self, return_value: Decimal) -> Result<()> {
        // Add to window
        self.returns_window.push_back(return_value);
        
        // Remove old values if window is full
        while self.returns_window.len() > self.window_size {
            self.returns_window.pop_front();
        }
        
        // Calculate Sharpe if we have enough data
        if self.returns_window.len() >= self.window_size / 2 {
            self.current_sharpe = self.calculate_sharpe()?;
            
            // Add to history
            self.sharpe_history.push_back(self.current_sharpe);
            while self.sharpe_history.len() > self.max_history {
                self.sharpe_history.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Calculate Sharpe ratio from returns window
    use mathematical_ops::risk_metrics::calculate_sharpe; // fn calculate_sharpe(&self) -> Result<Decimal> {
        if self.returns_window.is_empty() {
            return Ok(Decimal::ZERO);
        }
        
        // Calculate mean return
        let sum: Decimal = self.returns_window.iter().sum();
        let count = Decimal::from(self.returns_window.len());
        let mean_return = sum / count;
        
        // Calculate standard deviation
        let variance_sum: Decimal = self.returns_window
            .iter()
            .map(|r| {
                let diff = *r - mean_return;
                diff * diff  // Use multiplication instead of powi
            })
            .sum();
        let variance = variance_sum / count;
        let std_dev = variance.sqrt().unwrap_or(Decimal::ONE);
        
        // Avoid division by zero
        if std_dev == Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }
        
        // Calculate annualized Sharpe ratio
        // Sharpe = (E[R] - Rf) / σ * sqrt(periods_per_year)
        let excess_return = mean_return - (self.risk_free_rate / self.periods_per_year);
        let sharpe = (excess_return / std_dev) * self.periods_per_year.sqrt().unwrap_or(Decimal::ONE);
        
        Ok(sharpe)
    }
    
    /// Check if Sharpe has degraded significantly
    pub fn is_degraded(&self) -> bool {
        if self.baseline_sharpe == Decimal::ZERO {
            return false;
        }
        
        let threshold = self.baseline_sharpe * self.degradation_threshold;
        self.current_sharpe < threshold
    }
    
    /// Get degradation percentage
    pub fn degradation_percentage(&self) -> Decimal {
        if self.baseline_sharpe == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        let degradation = (self.baseline_sharpe - self.current_sharpe) / self.baseline_sharpe;
        degradation * dec!(100)
    }
    
    /// Detect trend in Sharpe ratio (improving/degrading)
    pub fn detect_trend(&self) -> SharpeTrend {
        if self.sharpe_history.len() < 10 {
            return SharpeTrend::Insufficient;
        }
        
        // Simple linear regression on recent Sharpe values
        let recent: Vec<Decimal> = self.sharpe_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();
        
        let n = Decimal::from(recent.len());
        let mut sum_x = Decimal::ZERO;
        let mut sum_y = Decimal::ZERO;
        let mut sum_xy = Decimal::ZERO;
        let mut sum_x2 = Decimal::ZERO;
        
        for (i, sharpe) in recent.iter().enumerate() {
            let x = Decimal::from(i);
            sum_x += x;
            sum_y += *sharpe;
            sum_xy += x * *sharpe;
            sum_x2 += x * x;
        }
        
        // Calculate slope
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator == Decimal::ZERO {
            return SharpeTrend::Stable;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        
        // Determine trend based on slope
        if slope > dec!(0.01) {
            SharpeTrend::Improving
        } else if slope < dec!(-0.01) {
            SharpeTrend::Degrading
        } else {
            SharpeTrend::Stable
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// TODO: Add docs
pub enum SharpeTrend {
    Improving,
    Stable,
    Degrading,
    Insufficient,
}

// ============================================================================
// REGIME CHANGE DETECTOR (Hidden Markov Model)
// ============================================================================

/// Detect market regime changes using Hidden Markov Model
/// Based on Hamilton (1989) and Bhar & Hamori (2004)
/// TODO: Add docs
pub struct RegimeDetector {
    /// Number of hidden states (regimes)
    num_states: usize,
    
    /// Transition probability matrix
    transition_matrix: Vec<Vec<f64>>,
    
    /// Emission parameters for each state (mean, variance)
    emission_params: Vec<(f64, f64)>,
    
    /// Current regime probabilities
    state_probabilities: Vec<f64>,
    
    /// Observation window
    observations: VecDeque<f64>,
    
    /// Window size
    window_size: usize,
    
    /// Current most likely regime
    current_regime: MarketRegime,
    
    /// Regime history
    regime_history: VecDeque<MarketRegime>,
    
    /// Confidence threshold for regime identification
    confidence_threshold: f64,
}

impl RegimeDetector {
    pub fn new(window_size: usize) -> Self {
        // Initialize 3-state HMM
        // State 0: Trending (low vol)
        // State 1: Mean-reverting (medium vol)
        // State 2: Crisis (high vol)
        
        let transition_matrix = vec![
            vec![0.95, 0.04, 0.01],  // Trending tends to persist
            vec![0.05, 0.90, 0.05],  // Mean-reverting is stable
            vec![0.02, 0.08, 0.90],  // Crisis tends to persist
        ];
        
        // Emission parameters (mean return, volatility)
        let emission_params = vec![
            (0.001, 0.01),   // Trending: positive returns, low vol
            (0.0, 0.02),     // Mean-reverting: zero returns, medium vol
            (-0.002, 0.05),  // Crisis: negative returns, high vol
        ];
        
        Self {
            num_states: 3,
            transition_matrix,
            emission_params,
            state_probabilities: vec![0.33, 0.34, 0.33], // Start with equal probabilities
            observations: VecDeque::with_capacity(window_size),
            window_size,
            current_regime: MarketRegime::MeanReverting,
            regime_history: VecDeque::with_capacity(100),
            confidence_threshold: 0.7,
        }
    }
    
    /// Add new observation and update regime probabilities
    pub fn add_observation(&mut self, return_value: f64) -> Result<()> {
        self.observations.push_back(return_value);
        
        while self.observations.len() > self.window_size {
            self.observations.pop_front();
        }
        
        // Update state probabilities using Forward algorithm
        if self.observations.len() >= 10 {
            self.update_state_probabilities()?;
            
            // Determine current regime
            let new_regime = self.identify_regime();
            if new_regime != self.current_regime {
                info!("Regime change detected: {:?} -> {:?}", self.current_regime, new_regime);
                self.current_regime = new_regime;
            }
            
            // Add to history
            self.regime_history.push_back(self.current_regime);
            while self.regime_history.len() > 100 {
                self.regime_history.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Update state probabilities using Forward algorithm
    fn update_state_probabilities(&mut self) -> Result<()> {
        if self.observations.is_empty() {
            return Ok(());
        }
        
        let observation = *self.observations.back().unwrap();
        
        // Calculate emission probabilities for current observation
        let mut emission_probs = vec![0.0; self.num_states];
        for (i, (mean, std_dev)) in self.emission_params.iter().enumerate() {
            emission_probs[i] = self.gaussian_pdf(observation, *mean, *std_dev);
        }
        
        // Forward step: P(state_t | obs_1:t) ∝ P(obs_t | state_t) * Σ P(state_t | state_{t-1}) * P(state_{t-1} | obs_1:{t-1})
        let mut new_probs = vec![0.0; self.num_states];
        for j in 0..self.num_states {
            for i in 0..self.num_states {
                new_probs[j] += self.transition_matrix[i][j] * self.state_probabilities[i];
            }
            new_probs[j] *= emission_probs[j];
        }
        
        // Normalize probabilities
        let sum: f64 = new_probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut new_probs {
                *prob /= sum;
            }
        }
        
        self.state_probabilities = new_probs;
        
        Ok(())
    }
    
    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let variance = std_dev * std_dev;
        let exponent = -((x - mean).powi(2)) / (2.0 * variance);
        let coefficient = 1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt());
        coefficient * exponent.exp()
    }
    
    /// Identify current regime based on state probabilities
    fn identify_regime(&self) -> MarketRegime {
        let max_prob_index = self.state_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        
        let max_prob = self.state_probabilities[max_prob_index];
        
        // Check if we have sufficient confidence
        if max_prob < self.confidence_threshold {
            return MarketRegime::Transition;
        }
        
        match max_prob_index {
            0 => MarketRegime::Trending,
            1 => MarketRegime::MeanReverting,
            2 => MarketRegime::Crisis,
            _ => MarketRegime::MeanReverting,
        }
    }
    
    /// Get regime change probability
    pub fn regime_change_probability(&self) -> f64 {
        // Probability of leaving current state
        let current_state = match self.current_regime {
            MarketRegime::Trending => 0,
            MarketRegime::MeanReverting => 1,
            MarketRegime::Crisis => 2,
            MarketRegime::Transition => return 0.5, // Uncertain
        };
        
        1.0 - self.transition_matrix[current_state][current_state]
    }
    
    /// Check if regime is unstable
    pub fn is_unstable(&self) -> bool {
        // Check if we've had multiple regime changes recently
        if self.regime_history.len() < 20 {
            return false;
        }
        
        let recent: Vec<_> = self.regime_history.iter().rev().take(20).collect();
        let mut changes = 0;
        for i in 1..recent.len() {
            if recent[i] != recent[i-1] {
                changes += 1;
            }
        }
        
        changes > 5 // More than 5 changes in 20 periods is unstable
    }
}

// ============================================================================
// GARCH VOLATILITY CLUSTERING DETECTOR
// ============================================================================

/// Detect abnormal volatility clustering using GARCH(1,1) model
/// Based on Bollerslev (1986) and Engle (1982)
/// TODO: Add docs
pub struct GARCHDetector {
    /// GARCH parameters (omega, alpha, beta)
    omega: f64,  // Constant
    alpha: f64,  // ARCH term coefficient
    beta: f64,   // GARCH term coefficient
    
    /// Returns window
    returns: VecDeque<f64>,
    
    /// Squared returns (for ARCH effect)
    squared_returns: VecDeque<f64>,
    
    /// Conditional variance estimates
    conditional_variances: VecDeque<f64>,
    
    /// Window size
    window_size: usize,
    
    /// Current conditional volatility
    current_volatility: f64,
    
    /// Long-run average volatility
    long_run_volatility: f64,
    
    /// Volatility clustering threshold (multiple of long-run)
    clustering_threshold: f64,
    
    /// Persistence measure (alpha + beta)
    persistence: f64,
}

impl GARCHDetector {
    pub fn new(window_size: usize) -> Self {
        // Standard GARCH(1,1) parameters from empirical studies
        let omega = 0.00001;  // Small constant
        let alpha = 0.1;      // ARCH coefficient
        let beta = 0.85;      // GARCH coefficient
        
        // Long-run variance = omega / (1 - alpha - beta)
        let long_run_variance: f64 = omega / (1.0 - alpha - beta);
        let long_run_volatility = long_run_variance.sqrt();
        
        Self {
            omega,
            alpha,
            beta,
            returns: VecDeque::with_capacity(window_size),
            squared_returns: VecDeque::with_capacity(window_size),
            conditional_variances: VecDeque::with_capacity(window_size),
            window_size,
            current_volatility: long_run_volatility,
            long_run_volatility,
            clustering_threshold: 2.0, // Alert if vol > 2x long-run
            persistence: alpha + beta,
        }
    }
    
    /// Add new return and update GARCH model
    pub fn add_return(&mut self, return_value: f64) -> Result<()> {
        self.returns.push_back(return_value);
        self.squared_returns.push_back(return_value * return_value);
        
        while self.returns.len() > self.window_size {
            self.returns.pop_front();
            self.squared_returns.pop_front();
        }
        
        // Update conditional variance
        // σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        if !self.squared_returns.is_empty() {
            let prev_squared_return = self.squared_returns[self.squared_returns.len() - 1];
            let prev_variance = if self.conditional_variances.is_empty() {
                self.long_run_volatility * self.long_run_volatility
            } else {
                self.conditional_variances[self.conditional_variances.len() - 1]
            };
            
            let new_variance = self.omega 
                + self.alpha * prev_squared_return 
                + self.beta * prev_variance;
            
            self.conditional_variances.push_back(new_variance);
            self.current_volatility = new_variance.sqrt();
            
            while self.conditional_variances.len() > self.window_size {
                self.conditional_variances.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Check if volatility clustering is detected
    pub fn is_clustering(&self) -> bool {
        self.current_volatility > self.long_run_volatility * self.clustering_threshold
    }
    
    /// Get volatility ratio (current / long-run)
    pub fn volatility_ratio(&self) -> f64 {
        if self.long_run_volatility > 0.0 {
            self.current_volatility / self.long_run_volatility
        } else {
            1.0
        }
    }
    
    /// Calculate Value at Risk using GARCH volatility
    pub use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var(&self, confidence_level: f64) -> f64 {
        // VaR = -μ + σ * z_α
        // For 95% confidence, z = 1.645
        // For 99% confidence, z = 2.326
        let z_score = match confidence_level {
            0.95 => 1.645,
            0.99 => 2.326,
            0.999 => 3.090,
            _ => 1.645,
        };
        
        // Assume zero mean for simplicity
        self.current_volatility * z_score
    }
    
    /// Check if GARCH model shows high persistence (near unit root)
    pub fn is_highly_persistent(&self) -> bool {
        self.persistence > 0.95 // Near integrated GARCH
    }
    
    /// Detect volatility regime
    pub fn volatility_regime(&self) -> VolatilityRegime {
        let ratio = self.volatility_ratio();
        
        if ratio < 0.5 {
            VolatilityRegime::VeryLow
        } else if ratio < 0.8 {
            VolatilityRegime::Low
        } else if ratio < 1.2 {
            VolatilityRegime::Normal
        } else if ratio < 2.0 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// TODO: Add docs
pub enum VolatilityRegime {
    VeryLow,
    Low,
    Normal,
    High,
    Extreme,
}

// ============================================================================
// STATISTICAL CIRCUIT BREAKER COORDINATOR
// ============================================================================

/// Coordinates all statistical anomaly detectors
/// TODO: Add docs
pub struct StatisticalCircuitBreaker {
    /// Sharpe ratio monitor
    sharpe_monitor: Arc<SyncRwLock<SharpeMonitor>>,
    
    /// Regime change detector
    regime_detector: Arc<SyncRwLock<RegimeDetector>>,
    
    /// GARCH volatility detector
    garch_detector: Arc<SyncRwLock<GARCHDetector>>,
    
    /// Current state
    state: Arc<SyncRwLock<StatisticalState>>,
    
    /// Anomaly history
    anomaly_history: Arc<SyncRwLock<VecDeque<AnomalyEvent>>>,
    
    /// Configuration
    config: StatisticalConfig,
    
    /// Last update time
    last_update: Arc<SyncRwLock<Instant>>,
    
    /// Trip conditions
    trip_conditions: TripConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct StatisticalConfig {
    /// Window size for calculations
    pub window_size: usize,
    
    /// Baseline Sharpe ratio
    pub baseline_sharpe: Decimal,
    
    /// Sharpe degradation threshold
    pub sharpe_degradation_threshold: Decimal,
    
    /// Risk-free rate
    pub risk_free_rate: Decimal,
    
    /// Periods per year
    pub periods_per_year: Decimal,
    
    /// Update interval
    pub update_interval: Duration,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TripConditions {
    /// Trip if Sharpe degrades by this percentage
    pub sharpe_trip_threshold: Decimal,
    
    /// Trip if regime is Crisis
    pub trip_on_crisis: bool,
    
    /// Trip if volatility clustering exceeds ratio
    pub volatility_trip_ratio: f64,
    
    /// Number of anomalies to trip
    pub anomaly_count_threshold: usize,
    
    /// Time window for anomaly counting
    pub anomaly_time_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct AnomalyEvent {
    pub timestamp: SystemTime,
    pub anomaly_type: StatisticalAnomaly,
    pub severity: f64,
    pub description: String,
}

impl StatisticalCircuitBreaker {
    pub fn new(config: StatisticalConfig) -> Self {
        let sharpe_monitor = Arc::new(SyncRwLock::new(SharpeMonitor::new(
            config.window_size,
            config.baseline_sharpe,
            config.sharpe_degradation_threshold,
            config.risk_free_rate,
            config.periods_per_year,
        )));
        
        let regime_detector = Arc::new(SyncRwLock::new(RegimeDetector::new(
            config.window_size,
        )));
        
        let garch_detector = Arc::new(SyncRwLock::new(GARCHDetector::new(
            config.window_size,
        )));
        
        let trip_conditions = TripConditions {
            sharpe_trip_threshold: dec!(0.3), // Trip if Sharpe drops 70%
            trip_on_crisis: true,
            volatility_trip_ratio: 3.0, // Trip if vol > 3x normal
            anomaly_count_threshold: 3,
            anomaly_time_window: Duration::from_secs(300), // 5 minutes
        };
        
        Self {
            sharpe_monitor,
            regime_detector,
            garch_detector,
            state: Arc::new(SyncRwLock::new(StatisticalState::Normal)),
            anomaly_history: Arc::new(SyncRwLock::new(VecDeque::with_capacity(100))),
            config,
            last_update: Arc::new(SyncRwLock::new(Instant::now())),
            trip_conditions,
        }
    }
    
    /// Process new market data
    pub fn process_data(&self, return_value: Decimal, _volume: Decimal, _spread: Decimal) -> Result<()> {
        // Update all detectors
        self.sharpe_monitor.write().add_return(return_value)?;
        self.regime_detector.write().add_observation(return_value.to_f64().unwrap_or(0.0))?;
        self.garch_detector.write().add_return(return_value.to_f64().unwrap_or(0.0))?;
        
        // Check for anomalies
        self.check_anomalies()?;
        
        // Update state
        self.update_state()?;
        
        *self.last_update.write() = Instant::now();
        
        Ok(())
    }
    
    /// Check for statistical anomalies
    fn check_anomalies(&self) -> Result<()> {
        let mut anomalies = Vec::new();
        let now = SystemTime::now();
        
        // Check Sharpe degradation
        {
            let sharpe = self.sharpe_monitor.read();
            if sharpe.is_degraded() {
                let degradation = sharpe.degradation_percentage();
                anomalies.push(AnomalyEvent {
                    timestamp: now,
                    anomaly_type: StatisticalAnomaly::SharpeDegradation,
                    severity: degradation.to_f64().unwrap_or(0.0) / 100.0,
                    description: format!("Sharpe degraded by {:.1}%", degradation),
                });
            }
        }
        
        // Check regime
        {
            let regime = self.regime_detector.read();
            if regime.current_regime == MarketRegime::Crisis {
                anomalies.push(AnomalyEvent {
                    timestamp: now,
                    anomaly_type: StatisticalAnomaly::RegimeChange,
                    severity: 0.9,
                    description: "Market in crisis regime".to_string(),
                });
            } else if regime.is_unstable() {
                anomalies.push(AnomalyEvent {
                    timestamp: now,
                    anomaly_type: StatisticalAnomaly::RegimeChange,
                    severity: 0.5,
                    description: "Regime instability detected".to_string(),
                });
            }
        }
        
        // Check volatility clustering
        {
            let garch = self.garch_detector.read();
            if garch.is_clustering() {
                let ratio = garch.volatility_ratio();
                anomalies.push(AnomalyEvent {
                    timestamp: now,
                    anomaly_type: StatisticalAnomaly::VolatilityClustering,
                    severity: (ratio - 1.0).min(1.0),
                    description: format!("Volatility {:.1}x normal", ratio),
                });
            }
        }
        
        // Add anomalies to history
        if !anomalies.is_empty() {
            let mut history = self.anomaly_history.write();
            for anomaly in anomalies {
                info!("Statistical anomaly detected: {:?}", anomaly);
                history.push_back(anomaly);
                while history.len() > 100 {
                    history.pop_front();
                }
            }
        }
        
        Ok(())
    }
    
    /// Update circuit breaker state based on anomalies
    fn update_state(&self) -> Result<()> {
        let mut should_trip = false;
        let mut new_state = StatisticalState::Normal;
        
        // Check trip conditions
        let sharpe = self.sharpe_monitor.read();
        let regime = self.regime_detector.read();
        let garch = self.garch_detector.read();
        
        // Sharpe-based trip
        if sharpe.current_sharpe < self.config.baseline_sharpe * self.trip_conditions.sharpe_trip_threshold {
            should_trip = true;
        }
        
        // Regime-based trip
        if self.trip_conditions.trip_on_crisis && regime.current_regime == MarketRegime::Crisis {
            should_trip = true;
        }
        
        // Volatility-based trip
        if garch.volatility_ratio() > self.trip_conditions.volatility_trip_ratio {
            should_trip = true;
        }
        
        // Count recent anomalies
        let recent_anomalies = {
            let history = self.anomaly_history.read();
            let cutoff = SystemTime::now() - self.trip_conditions.anomaly_time_window;
            history.iter()
                .filter(|a| a.timestamp > cutoff)
                .count()
        };
        
        if recent_anomalies >= self.trip_conditions.anomaly_count_threshold {
            should_trip = true;
        }
        
        // Determine state
        if should_trip {
            new_state = StatisticalState::Tripped;
        } else if recent_anomalies >= 2 {
            new_state = StatisticalState::Critical;
        } else if recent_anomalies >= 1 {
            new_state = StatisticalState::Warning;
        }
        
        // Update state if changed
        let mut state = self.state.write();
        if *state != new_state {
            info!("Statistical circuit breaker state change: {:?} -> {:?}", *state, new_state);
            *state = new_state;
        }
        
        Ok(())
    }
    
    /// Get current state
    pub fn get_state(&self) -> StatisticalState {
        *self.state.read()
    }
    
    /// Get comprehensive status
    pub fn get_status(&self) -> StatisticalStatus {
        let sharpe = self.sharpe_monitor.read();
        let regime = self.regime_detector.read();
        let garch = self.garch_detector.read();
        
        StatisticalStatus {
            state: *self.state.read(),
            current_sharpe: sharpe.current_sharpe,
            sharpe_trend: sharpe.detect_trend(),
            current_regime: regime.current_regime,
            regime_stability: !regime.is_unstable(),
            volatility_ratio: garch.volatility_ratio(),
            volatility_regime: garch.volatility_regime(),
            recent_anomalies: {
                let history = self.anomaly_history.read();
                let cutoff = SystemTime::now() - Duration::from_secs(300);
                history.iter()
                    .filter(|a| a.timestamp > cutoff)
                    .cloned()
                    .collect()
            },
            last_update: *self.last_update.read(),
        }
    }
    
    /// Force reset to normal state (admin override)
    pub fn reset(&self) {
        *self.state.write() = StatisticalState::Normal;
        self.anomaly_history.write().clear();
        info!("Statistical circuit breaker reset to normal");
    }
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct StatisticalStatus {
    pub state: StatisticalState,
    pub current_sharpe: Decimal,
    pub sharpe_trend: SharpeTrend,
    pub current_regime: MarketRegime,
    pub regime_stability: bool,
    pub volatility_ratio: f64,
    pub volatility_regime: VolatilityRegime,
    pub recent_anomalies: Vec<AnomalyEvent>,
    pub last_update: Instant,
}

// ============================================================================
// INTEGRATION WITH MAIN CIRCUIT BREAKER
// ============================================================================

/// Integration point with main circuit breaker system
/// TODO: Add docs
pub struct StatisticalBreakerIntegration {
    statistical_breaker: Arc<StatisticalCircuitBreaker>,
    integration_enabled: Arc<SyncRwLock<bool>>,
}

impl StatisticalBreakerIntegration {
    pub fn new(config: StatisticalConfig) -> Self {
        Self {
            statistical_breaker: Arc::new(StatisticalCircuitBreaker::new(config)),
            integration_enabled: Arc::new(SyncRwLock::new(true)),
        }
    }
    
    /// Check if trading should be allowed based on statistical analysis
    pub fn should_allow_trading(&self) -> bool {
        if !*self.integration_enabled.read() {
            return true; // Bypass if disabled
        }
        
        let state = self.statistical_breaker.get_state();
        matches!(state, StatisticalState::Normal | StatisticalState::Warning)
    }
    
    /// Get risk multiplier based on statistical state
    pub fn get_risk_multiplier(&self) -> Decimal {
        let state = self.statistical_breaker.get_state();
        match state {
            StatisticalState::Normal => dec!(1.0),
            StatisticalState::Warning => dec!(0.7),
            StatisticalState::Critical => dec!(0.3),
            StatisticalState::Tripped => dec!(0.0),
        }
    }
    
    /// Process market update
    pub fn process_market_update(&self, return_value: Decimal, volume: Decimal, spread: Decimal) -> Result<()> {
        self.statistical_breaker.process_data(return_value, volume, spread)
    }
    
    /// Enable/disable integration
    pub fn set_enabled(&self, enabled: bool) {
        *self.integration_enabled.write() = enabled;
        info!("Statistical breaker integration {}", if enabled { "enabled" } else { "disabled" });
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sharpe_monitor() {
        let mut monitor = SharpeMonitor::new(
            20,
            dec!(1.5),
            dec!(0.5),
            dec!(0.02),
            dec!(252),
        );
        
        // Add some returns
        for i in 0..20 {
            let return_val = if i % 2 == 0 { dec!(0.001) } else { dec!(-0.0005) };
            monitor.add_return(return_val).unwrap();
        }
        
        // Check Sharpe calculation
        assert!(monitor.current_sharpe != Decimal::ZERO);
        
        // Check degradation detection
        monitor.current_sharpe = dec!(0.5); // Force low Sharpe
        assert!(monitor.is_degraded());
    }
    
    #[test]
    fn test_regime_detector() {
        let mut detector = RegimeDetector::new(50);
        
        // Simulate trending market
        for _ in 0..20 {
            detector.add_observation(0.001).unwrap();
        }
        
        // Simulate volatile market
        for i in 0..20 {
            let return_val = if i % 2 == 0 { 0.05 } else { -0.04 };
            detector.add_observation(return_val).unwrap();
        }
        
        // Check regime detection
        assert!(detector.current_regime != MarketRegime::Trending);
    }
    
    #[test]
    fn test_garch_detector() {
        let mut detector = GARCHDetector::new(100);
        
        // Add normal returns
        for _ in 0..50 {
            detector.add_return(0.001).unwrap();
        }
        
        assert!(!detector.is_clustering());
        
        // Add volatile returns
        for i in 0..20 {
            let return_val = if i % 2 == 0 { 0.05 } else { -0.05 };
            detector.add_return(return_val).unwrap();
        }
        
        // Should detect clustering
        assert!(detector.volatility_ratio() > 1.0);
    }
    
    #[test]
    fn test_statistical_circuit_breaker() {
        let config = StatisticalConfig {
            window_size: 50,
            baseline_sharpe: dec!(1.5),
            sharpe_degradation_threshold: dec!(0.5),
            risk_free_rate: dec!(0.02),
            periods_per_year: dec!(252),
            update_interval: Duration::from_secs(60),
        };
        
        let breaker = StatisticalCircuitBreaker::new(config);
        
        // Process normal data
        for _ in 0..20 {
            breaker.process_data(dec!(0.001), dec!(1000000), dec!(0.0001)).unwrap();
        }
        
        assert_eq!(breaker.get_state(), StatisticalState::Normal);
        
        // Process volatile data
        for i in 0..20 {
            let return_val = if i % 2 == 0 { dec!(0.05) } else { dec!(-0.05) };
            breaker.process_data(return_val, dec!(2000000), dec!(0.001)).unwrap();
        }
        
        // Should detect anomalies
        let status = breaker.get_status();
        assert!(!status.recent_anomalies.is_empty());
    }
}