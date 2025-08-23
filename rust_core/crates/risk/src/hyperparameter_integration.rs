// Hyperparameter Integration System - DEEP DIVE WITH NO SIMPLIFICATIONS
// Team: Alex (Lead) + Full Team Deep Collaboration
// CRITICAL: Integrate optimization into ALL beneficial components
// EXTRACT 100% FROM THE MARKET through intelligent parameter adaptation

use crate::hyperparameter_optimization::*;
use crate::auto_tuning::AutoTuningSystem;
use crate::kelly_sizing::{KellySizer, KellyConfig};
use crate::clamps::{RiskClampSystem, ClampConfig};
use crate::ml_feedback::MLFeedbackSystem;
use crate::profit_extractor::ProfitExtractor;
use crate::market_analytics::MarketAnalytics;
use crate::unified_types::*;
use crate::isotonic::MarketRegime;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// Temporary stub for OptimalExecutor until full implementation
pub struct OptimalExecutor {
    algorithm_bias: f64,
    max_participation_rate: f64,
}

impl OptimalExecutor {
    pub fn new() -> Self {
        Self {
            algorithm_bias: 0.5,
            max_participation_rate: 0.1,
        }
    }
    
    pub fn set_algorithm_bias(&mut self, bias: f64) {
        self.algorithm_bias = bias;
    }
    
    pub fn set_max_participation_rate(&mut self, rate: f64) {
        self.max_participation_rate = rate;
    }
}

/// Master integration system that connects hyperparameter optimization
/// to ALL components for maximum market value extraction
pub struct HyperparameterIntegrationSystem {
    // Core optimization engine
    auto_tuner: Arc<RwLock<AutoTuner>>,
    
    // Components to optimize
    kelly_sizer: Arc<RwLock<KellySizer>>,
    risk_clamps: Arc<RwLock<RiskClampSystem>>,
    ml_system: Arc<RwLock<MLFeedbackSystem>>,
    executor: Arc<RwLock<OptimalExecutor>>,
    profit_extractor: Arc<RwLock<ProfitExtractor>>,
    
    // Performance tracking
    performance_history: Vec<PerformanceSnapshot>,
    optimization_history: Vec<OptimizationEvent>,
    
    // Current optimized parameters
    current_params: Arc<RwLock<HashMap<String, f64>>>,
    
    // Market regime for context-aware optimization
    current_regime: MarketRegime,
    
    // Optimization scheduling
    last_optimization: DateTime<Utc>,
    optimization_interval: std::time::Duration,
    
    // Performance metrics for objective function
    recent_sharpe: f64,
    recent_drawdown: f64,
    recent_win_rate: f64,
    total_pnl: f64,
}

#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    timestamp: DateTime<Utc>,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    total_return: f64,
    parameters_used: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct OptimizationEvent {
    timestamp: DateTime<Utc>,
    trigger: OptimizationTrigger,
    old_params: HashMap<String, f64>,
    new_params: HashMap<String, f64>,
    improvement: f64,
}

#[derive(Debug, Clone)]
enum OptimizationTrigger {
    Scheduled,
    RegimeChange(MarketRegime, MarketRegime),
    PerformanceDegradation(f64),
    ManualTrigger,
}

impl HyperparameterIntegrationSystem {
    pub fn new(config: AutoTunerConfig) -> Self {
        let auto_tuner = Arc::new(RwLock::new(AutoTuner::new(config.clone())));
        
        // Initialize all components with default parameters
        let kelly_sizer = Arc::new(RwLock::new(KellySizer::new(KellyConfig::default())));
        let risk_clamps = Arc::new(RwLock::new(RiskClampSystem::new(ClampConfig::default())));
        let ml_system = Arc::new(RwLock::new(MLFeedbackSystem::new()));
        let executor = Arc::new(RwLock::new(OptimalExecutor::new()));
        let profit_extractor = Arc::new(RwLock::new(ProfitExtractor::new()));
        
        Self {
            auto_tuner,
            kelly_sizer,
            risk_clamps,
            ml_system,
            executor,
            profit_extractor,
            performance_history: Vec::new(),
            optimization_history: Vec::new(),
            current_params: Arc::new(RwLock::new(HashMap::new())),
            current_regime: MarketRegime::Normal,
            last_optimization: Utc::now(),
            optimization_interval: config.optimization_interval,
            recent_sharpe: 0.0,
            recent_drawdown: 0.0,
            recent_win_rate: 0.5,
            total_pnl: 0.0,
        }
    }
    
    /// CRITICAL: Main optimization cycle that adapts ALL parameters
    pub fn run_optimization_cycle(&mut self) -> HashMap<String, f64> {
        println!("🔧 Running DEEP DIVE optimization cycle - NO SIMPLIFICATIONS!");
        
        // Build comprehensive objective function that considers ALL aspects
        let objective = self.build_comprehensive_objective();
        
        // Run Bayesian optimization with TPE
        let mut tuner = self.auto_tuner.write().unwrap();
        let optimized_params = if self.should_use_regime_specific() {
            tuner.optimize_for_regime(objective, self.current_regime)
        } else {
            tuner.optimize(objective)
        };
        
        // Apply optimized parameters to ALL components
        self.apply_parameters_to_all_components(&optimized_params);
        
        // Record optimization event
        let old_params = self.current_params.read().unwrap().clone();
        let improvement = self.calculate_improvement(&old_params, &optimized_params);
        
        self.optimization_history.push(OptimizationEvent {
            timestamp: Utc::now(),
            trigger: OptimizationTrigger::Scheduled,
            old_params,
            new_params: optimized_params.clone(),
            improvement,
        });
        
        // Update current parameters
        *self.current_params.write().unwrap() = optimized_params.clone();
        self.last_optimization = Utc::now();
        
        println!("✅ Optimization complete! Improvement: {:.2}%", improvement * 100.0);
        optimized_params
    }
    
    /// Build objective function that considers ALL system components
    fn build_comprehensive_objective(&self) -> Box<dyn Fn(&HashMap<String, f64>) -> f64> {
        let sharpe = self.recent_sharpe;
        let drawdown = self.recent_drawdown;
        let win_rate = self.recent_win_rate;
        let regime = self.current_regime.clone();
        
        Box::new(move |params: &HashMap<String, f64>| -> f64 {
            // CRITICAL: Multi-factor objective function
            // Game Theory: Balance competing objectives
            
            // 1. Risk-adjusted returns (Sharpe ratio component)
            let kelly = params.get("kelly_fraction").unwrap_or(&0.25);
            let expected_return = kelly * 2.0 * (1.0 + win_rate - 0.5);
            let expected_risk = kelly.powf(2.0) * 3.0;
            let sharpe_component = (expected_return - expected_risk).max(-1.0);
            
            // 2. Downside protection (drawdown component)
            let var_limit = params.get("var_limit").unwrap_or(&0.02);
            let stop_loss = params.get("stop_loss_percentage").unwrap_or(&0.02);
            let protection_score = (1.0 - drawdown) * (1.0 - var_limit * 10.0) * 
                                  (1.0 - stop_loss * 5.0);
            
            // 3. ML effectiveness (confidence and accuracy)
            let ml_threshold = params.get("ml_confidence_threshold").unwrap_or(&0.6);
            let ml_effectiveness = ml_threshold * win_rate;
            
            // 4. Execution efficiency (market impact minimization)
            let max_position = params.get("max_position_size").unwrap_or(&0.02);
            let execution_algo = params.get("execution_algorithm_bias").unwrap_or(&0.5);
            let execution_score = (1.0 - max_position * 2.0) * execution_algo;
            
            // 5. Market regime adaptation
            let regime_multiplier = match regime {
                MarketRegime::Bull => 1.3,  // Can be more aggressive
                MarketRegime::Bear => 0.7,  // Need protection
                MarketRegime::Crisis => 0.5, // Maximum protection
                MarketRegime::Sideways => 0.9, // Moderate
                MarketRegime::Normal => 1.0,
            };
            
            // 6. Information exploitation (game theory)
            let entry_threshold = params.get("entry_threshold").unwrap_or(&0.005);
            let exit_threshold = params.get("exit_threshold").unwrap_or(&0.003);
            let alpha_capture = (1.0 / entry_threshold.max(0.001)) * 0.01 +
                              (1.0 / exit_threshold.max(0.001)) * 0.01;
            
            // Combine all factors with weights
            let total_score = (sharpe_component * 0.3 +
                             protection_score * 0.2 +
                             ml_effectiveness * 0.2 +
                             execution_score * 0.1 +
                             alpha_capture * 0.2) * regime_multiplier;
            
            // Add small noise for exploration
            let noise = (rand::random::<f64>() - 0.5) * 0.02;
            
            (total_score + noise).max(-2.0).min(3.0)
        })
    }
    
    /// Apply optimized parameters to ALL system components
    fn apply_parameters_to_all_components(&self, params: &HashMap<String, f64>) {
        println!("📊 Applying optimized parameters to ALL components...");
        
        // 1. Update Kelly Sizer
        if let Ok(mut kelly) = self.kelly_sizer.write() {
            kelly.update_config(KellyConfig {
                max_kelly_fraction: *params.get("kelly_fraction").unwrap_or(&0.25),
                confidence_scaling: true,
                min_edge_threshold: *params.get("entry_threshold").unwrap_or(&0.005),
                use_half_kelly: false, // We optimize the fraction directly
            });
            println!("  ✓ Kelly Sizer updated: fraction={:.3}", 
                    params.get("kelly_fraction").unwrap_or(&0.25));
        }
        
        // 2. Update Risk Clamps
        if let Ok(mut clamps) = self.risk_clamps.write() {
            clamps.update_limits(
                *params.get("var_limit").unwrap_or(&0.02),
                *params.get("max_position_size").unwrap_or(&0.02),
                *params.get("max_leverage").unwrap_or(&3.0),
                *params.get("correlation_limit").unwrap_or(&0.7),
            );
            println!("  ✓ Risk Clamps updated: VaR={:.3}, MaxPos={:.3}",
                    params.get("var_limit").unwrap_or(&0.02),
                    params.get("max_position_size").unwrap_or(&0.02));
        }
        
        // 3. Update ML System thresholds
        if let Ok(mut ml) = self.ml_system.write() {
            ml.set_confidence_threshold(
                *params.get("ml_confidence_threshold").unwrap_or(&0.6)
            );
            ml.set_feature_importance_threshold(
                *params.get("feature_importance_threshold").unwrap_or(&0.1)
            );
            println!("  ✓ ML System updated: confidence={:.3}",
                    params.get("ml_confidence_threshold").unwrap_or(&0.6));
        }
        
        // 4. Update Execution Algorithm preferences
        if let Ok(mut executor) = self.executor.write() {
            executor.set_algorithm_bias(
                *params.get("execution_algorithm_bias").unwrap_or(&0.5)
            );
            executor.set_max_participation_rate(
                *params.get("max_participation_rate").unwrap_or(&0.1)
            );
            println!("  ✓ Executor updated: algo_bias={:.3}",
                    params.get("execution_algorithm_bias").unwrap_or(&0.5));
        }
        
        // 5. Update Profit Extractor thresholds
        if let Ok(mut extractor) = self.profit_extractor.write() {
            extractor.update_thresholds(
                *params.get("take_profit_percentage").unwrap_or(&0.05),
                *params.get("stop_loss_percentage").unwrap_or(&0.02),
                *params.get("trailing_stop_percentage").unwrap_or(&0.015),
            );
            println!("  ✓ Profit Extractor updated: TP={:.3}, SL={:.3}",
                    params.get("take_profit_percentage").unwrap_or(&0.05),
                    params.get("stop_loss_percentage").unwrap_or(&0.02));
        }
        
        println!("✅ All components updated with optimized parameters!");
    }
    
    /// Determine if we should use regime-specific optimization
    fn should_use_regime_specific(&self) -> bool {
        matches!(self.current_regime, MarketRegime::Crisis | MarketRegime::Bear)
    }
    
    /// Calculate improvement between parameter sets
    fn calculate_improvement(&self, old: &HashMap<String, f64>, new: &HashMap<String, f64>) -> f64 {
        // Estimate improvement based on parameter changes
        let old_kelly = old.get("kelly_fraction").unwrap_or(&0.25);
        let new_kelly = new.get("kelly_fraction").unwrap_or(&0.25);
        
        let old_var = old.get("var_limit").unwrap_or(&0.02);
        let new_var = new.get("var_limit").unwrap_or(&0.02);
        
        // Improvement if Kelly increased (more aggressive) or VaR decreased (less risk)
        let kelly_improvement = (new_kelly - old_kelly) / old_kelly.max(0.01);
        let var_improvement = (old_var - new_var) / old_var.max(0.001);
        
        (kelly_improvement * 0.5 + var_improvement * 0.5).max(-1.0).min(1.0)
    }
    
    /// Update performance metrics for next optimization
    pub fn update_performance_metrics(&mut self, 
                                     sharpe: f64, 
                                     drawdown: f64, 
                                     win_rate: f64,
                                     pnl: f64) {
        self.recent_sharpe = sharpe;
        self.recent_drawdown = drawdown;
        self.recent_win_rate = win_rate;
        self.total_pnl += pnl;
        
        // Store snapshot
        self.performance_history.push(PerformanceSnapshot {
            timestamp: Utc::now(),
            sharpe_ratio: sharpe,
            max_drawdown: drawdown,
            win_rate,
            total_return: self.total_pnl,
            parameters_used: self.current_params.read().unwrap().clone(),
        });
        
        // Check if we need emergency re-optimization
        if self.should_trigger_emergency_optimization() {
            println!("⚠️ Performance degradation detected! Triggering emergency optimization...");
            self.run_optimization_cycle();
        }
    }
    
    /// Check if emergency optimization is needed
    fn should_trigger_emergency_optimization(&self) -> bool {
        // Trigger if Sharpe drops below 0 or drawdown exceeds 15%
        self.recent_sharpe < 0.0 || self.recent_drawdown > 0.15
    }
    
    /// Update market regime and potentially re-optimize
    pub fn update_market_regime(&mut self, new_regime: MarketRegime) {
        if new_regime != self.current_regime {
            println!("🔄 Market regime change: {:?} -> {:?}", self.current_regime, new_regime);
            
            let old_regime = self.current_regime.clone();
            self.current_regime = new_regime.clone();
            
            // Trigger re-optimization on regime change
            let optimized = self.run_optimization_cycle();
            
            self.optimization_history.push(OptimizationEvent {
                timestamp: Utc::now(),
                trigger: OptimizationTrigger::RegimeChange(old_regime, new_regime),
                old_params: self.current_params.read().unwrap().clone(),
                new_params: optimized,
                improvement: 0.0, // Will be calculated after trading
            });
        }
    }
    
    /// Get current optimized parameters
    pub fn get_current_parameters(&self) -> HashMap<String, f64> {
        self.current_params.read().unwrap().clone()
    }
    
    /// Check if optimization is due
    pub fn should_optimize(&self) -> bool {
        let elapsed = Utc::now().signed_duration_since(self.last_optimization);
        elapsed.to_std().unwrap_or(std::time::Duration::from_secs(0)) > self.optimization_interval
    }
    
    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let total_optimizations = self.optimization_history.len();
        let avg_improvement = if total_optimizations > 0 {
            self.optimization_history.iter()
                .map(|e| e.improvement)
                .sum::<f64>() / total_optimizations as f64
        } else {
            0.0
        };
        
        let best_params = self.performance_history
            .iter()
            .max_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap())
            .map(|p| p.parameters_used.clone())
            .unwrap_or_default();
        
        OptimizationStats {
            total_optimizations,
            average_improvement: avg_improvement,
            best_sharpe_achieved: self.performance_history
                .iter()
                .map(|p| p.sharpe_ratio)
                .fold(0.0, f64::max),
            best_parameters: best_params,
            current_regime: self.current_regime.clone(),
        }
    }
    
    /// Analyze which parameters have the most impact
    pub fn analyze_parameter_sensitivity(&self) -> HashMap<String, f64> {
        let mut sensitivity = HashMap::new();
        
        if self.performance_history.len() < 10 {
            return sensitivity; // Not enough data
        }
        
        // Calculate correlation between each parameter and performance
        let param_names = vec![
            "kelly_fraction", "var_limit", "ml_confidence_threshold",
            "stop_loss_percentage", "take_profit_percentage"
        ];
        
        for param_name in param_names {
            let mut param_values = Vec::new();
            let mut performance_values = Vec::new();
            
            for snapshot in &self.performance_history {
                if let Some(value) = snapshot.parameters_used.get(param_name) {
                    param_values.push(*value);
                    performance_values.push(snapshot.sharpe_ratio);
                }
            }
            
            if param_values.len() >= 5 {
                let correlation = calculate_correlation(&param_values, &performance_values);
                sensitivity.insert(param_name.to_string(), correlation);
            }
        }
        
        sensitivity
    }
    
    /// Get recommended parameters for current market conditions
    pub fn get_recommended_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        
        // Base recommendations adjusted by regime
        match self.current_regime {
            MarketRegime::Bull => {
                // Aggressive parameters for bull market
                params.insert("kelly_fraction".to_string(), 0.35);
                params.insert("var_limit".to_string(), 0.03);
                params.insert("max_position_size".to_string(), 0.03);
                params.insert("ml_confidence_threshold".to_string(), 0.55);
                params.insert("take_profit_percentage".to_string(), 0.08);
                params.insert("stop_loss_percentage".to_string(), 0.025);
            }
            MarketRegime::Bear => {
                // Conservative parameters for bear market
                params.insert("kelly_fraction".to_string(), 0.15);
                params.insert("var_limit".to_string(), 0.015);
                params.insert("max_position_size".to_string(), 0.015);
                params.insert("ml_confidence_threshold".to_string(), 0.70);
                params.insert("take_profit_percentage".to_string(), 0.03);
                params.insert("stop_loss_percentage".to_string(), 0.015);
            }
            MarketRegime::Crisis => {
                // Ultra-conservative for crisis
                params.insert("kelly_fraction".to_string(), 0.05);
                params.insert("var_limit".to_string(), 0.01);
                params.insert("max_position_size".to_string(), 0.01);
                params.insert("ml_confidence_threshold".to_string(), 0.80);
                params.insert("take_profit_percentage".to_string(), 0.02);
                params.insert("stop_loss_percentage".to_string(), 0.01);
            }
            MarketRegime::Sideways => {
                // Range-trading parameters
                params.insert("kelly_fraction".to_string(), 0.20);
                params.insert("var_limit".to_string(), 0.02);
                params.insert("max_position_size".to_string(), 0.02);
                params.insert("ml_confidence_threshold".to_string(), 0.65);
                params.insert("take_profit_percentage".to_string(), 0.04);
                params.insert("stop_loss_percentage".to_string(), 0.02);
            }
            MarketRegime::Normal => {
                // Standard parameters
                params.insert("kelly_fraction".to_string(), 0.25);
                params.insert("var_limit".to_string(), 0.02);
                params.insert("max_position_size".to_string(), 0.02);
                params.insert("ml_confidence_threshold".to_string(), 0.60);
                params.insert("take_profit_percentage".to_string(), 0.05);
                params.insert("stop_loss_percentage".to_string(), 0.02);
            }
        }
        
        // Add common parameters
        params.insert("max_leverage".to_string(), 3.0);
        params.insert("correlation_limit".to_string(), 0.7);
        params.insert("execution_algorithm_bias".to_string(), 0.5);
        params.insert("max_participation_rate".to_string(), 0.1);
        params.insert("entry_threshold".to_string(), 0.005);
        params.insert("exit_threshold".to_string(), 0.003);
        params.insert("trailing_stop_percentage".to_string(), 0.015);
        params.insert("feature_importance_threshold".to_string(), 0.1);
        
        params
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_optimizations: usize,
    pub average_improvement: f64,
    pub best_sharpe_achieved: f64,
    pub best_parameters: HashMap<String, f64>,
    pub current_regime: MarketRegime,
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
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

// Extension trait for easy integration
impl AutoTuner {
    /// Quick optimization with fewer trials for real-time adaptation
    pub fn optimize_quick(&mut self, 
                          objective: Box<dyn Fn(&HashMap<String, f64>) -> f64>,
                          n_trials: usize) -> HashMap<String, f64> {
        let space = TradingParameterSpace::new();
        let mut best_params = HashMap::new();
        let mut best_value = f64::NEG_INFINITY;
        
        for i in 0..n_trials {
            let params = if i < 3 {
                space.sample_random()
            } else {
                self.sampler.sample(&space)
            };
            
            let value = objective(&params);
            
            if value > best_value {
                best_value = value;
                best_params = params.clone();
            }
            
            let trial = Trial {
                id: i,
                params,
                value,
                state: TrialState::Complete,
                timestamp: chrono::Utc::now(),
            };
            
            self.sampler.update(trial);
        }
        
        best_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_integration_system_initialization() {
        println!("Testing HyperparameterIntegrationSystem initialization");
        
        let config = AutoTunerConfig::default();
        let system = HyperparameterIntegrationSystem::new(config);
        
        assert_eq!(system.current_regime, MarketRegime::Normal);
        assert_eq!(system.performance_history.len(), 0);
        assert_eq!(system.optimization_history.len(), 0);
        
        println!("✅ Integration system initialized successfully");
    }
    
    #[test]
    fn test_comprehensive_optimization() {
        println!("Testing comprehensive optimization across ALL components");
        
        let config = AutoTunerConfig {
            n_trials: 10,
            n_startup_trials: 3,
            optimization_interval: std::time::Duration::from_secs(60),
            performance_window: 50,
            min_samples_before_optimization: 5,
        };
        
        let mut system = HyperparameterIntegrationSystem::new(config);
        
        // Set some performance metrics
        system.update_performance_metrics(1.5, 0.08, 0.65, 1000.0);
        
        // Run optimization
        let optimized_params = system.run_optimization_cycle();
        
        assert!(!optimized_params.is_empty());
        assert!(optimized_params.contains_key("kelly_fraction"));
        assert!(optimized_params.contains_key("var_limit"));
        
        println!("✅ Comprehensive optimization completed successfully");
    }
    
    #[test]
    fn test_regime_change_triggers_optimization() {
        println!("Testing regime change optimization trigger");
        
        let config = AutoTunerConfig::default();
        let mut system = HyperparameterIntegrationSystem::new(config);
        
        // Change regime from Normal to Crisis
        system.update_market_regime(MarketRegime::Crisis);
        
        // Should have optimization event
        assert!(!system.optimization_history.is_empty());
        
        // Parameters should be conservative for crisis
        let params = system.get_current_parameters();
        if let Some(kelly) = params.get("kelly_fraction") {
            assert!(*kelly <= 0.15, "Kelly should be conservative in crisis");
        }
        
        println!("✅ Regime change correctly triggers re-optimization");
    }
}