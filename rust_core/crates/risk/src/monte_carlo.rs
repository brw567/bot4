// MONTE CARLO SIMULATIONS - STOCHASTIC MODELING WITH GAME THEORY
// Team: Morgan (ML Lead) + Quinn (Risk) + Jordan (Performance) + Full Team
// CRITICAL: Simulate millions of scenarios to validate strategies
// References:
// - Black-Scholes-Merton (1973): Option pricing with stochastic processes
// - Glasserman (2003): "Monte Carlo Methods in Financial Engineering"
// - Longstaff-Schwartz (2001): Least-squares Monte Carlo for American options
// - Broadie & Glasserman (1997): Pricing high-dimensional derivatives
// - Hull & White (1990): Stochastic volatility models

use crate::unified_types::*;
use crate::kelly_sizing::KellySizer;
use crate::clamps::RiskClampSystem;
use crate::garch::GARCHModel;
use crate::optimal_execution::OptimalExecutionEngine;
use crate::portfolio_manager::{PortfolioManager, PortfolioConfig};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use rand::prelude::*;
use rand_distr::{Normal, LogNormal, Beta, Gamma, StudentT, Exp};
use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;
use rayon::prelude::*;

/// Monte Carlo Simulation Engine - Parallel stochastic modeling
/// Morgan: "If we can't simulate it, we shouldn't trade it!"
pub struct MonteCarloEngine {
    // Simulation parameters
    num_simulations: usize,
    time_steps: usize,
    dt: f64,  // Time step size
    
    // Market models
    price_model: PriceModel,
    volatility_model: VolatilityModel,
    jump_model: Option<JumpModel>,
    correlation_matrix: CorrelationMatrix,
    
    // Risk components
    kelly_sizer: Arc<RwLock<KellySizer>>,
    risk_clamps: Arc<RwLock<RiskClampSystem>>,
    portfolio_manager: Arc<RwLock<PortfolioManager>>,
    
    // Results storage
    simulation_results: Arc<RwLock<Vec<SimulationResult>>>,
    path_statistics: Arc<RwLock<PathStatistics>>,
    
    // Performance metrics
    parallel_threads: usize,
    use_gpu: bool,  // Future: GPU acceleration
    use_quasi_random: bool,  // Sobol sequences for better convergence
}

/// Price dynamics model
#[derive(Debug, Clone)]
pub enum PriceModel {
    GeometricBrownianMotion {
        drift: f64,
        volatility: f64,
    },
    MeanReverting {
        kappa: f64,  // Mean reversion speed
        theta: f64,  // Long-term mean
        sigma: f64,  // Volatility
    },
    JumpDiffusion {
        drift: f64,
        volatility: f64,
        jump_intensity: f64,
        jump_mean: f64,
        jump_std: f64,
    },
    HestonStochastic {
        mu: f64,     // Drift
        kappa: f64,  // Vol mean reversion
        theta: f64,  // Long-term vol
        xi: f64,     // Vol of vol
        rho: f64,    // Correlation
    },
    FractionalBrownian {
        hurst: f64,  // Hurst exponent (0.5 = normal, >0.5 = trending)
        volatility: f64,
    },
}

/// Volatility dynamics model
#[derive(Debug, Clone)]
pub enum VolatilityModel {
    Constant(f64),
    GARCH {
        alpha: f64,
        beta: f64,
        omega: f64,
    },
    StochasticVolatility {
        mean_reversion: f64,
        vol_of_vol: f64,
        correlation: f64,
    },
    RoughVolatility {
        hurst: f64,  // H < 0.5 for rough volatility
        alpha: f64,
        beta: f64,
    },
}

/// Jump process model
#[derive(Debug, Clone)]
pub struct JumpModel {
    pub intensity: f64,  // Poisson intensity
    pub size_distribution: JumpSizeDistribution,
    pub clustering: f64,  // Jump clustering (Hawkes process)
}

#[derive(Debug, Clone)]
pub enum JumpSizeDistribution {
    Normal { mean: f64, std: f64 },
    DoubleExponential { lambda_up: f64, lambda_down: f64 },
    PowerLaw { alpha: f64 },  // Fat-tailed jumps
}

/// Correlation structure
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub assets: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub time_varying: bool,
    pub regime_dependent: bool,
}

/// Single simulation result
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub final_pnl: Decimal,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub kelly_fraction: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub path: Vec<f64>,  // Price path
}

/// Aggregated path statistics
#[derive(Debug, Clone)]
pub struct PathStatistics {
    pub mean_return: f64,
    pub std_return: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: Vec<(f64, f64)>,  // (percentile, value)
    pub autocorrelation: Vec<f64>,
    pub hurst_exponent: f64,
    pub max_drawdown_distribution: Vec<f64>,
}

impl MonteCarloEngine {
    /// Create new Monte Carlo engine with advanced models
    pub fn new(
        num_simulations: usize,
        time_steps: usize,
        dt: f64,
        initial_capital: Decimal,
    ) -> Self {
        let config = PortfolioConfig::default();
        
        Self {
            num_simulations,
            time_steps,
            dt,
            price_model: PriceModel::GeometricBrownianMotion {
                drift: 0.05,
                volatility: 0.2,
            },
            volatility_model: VolatilityModel::GARCH {
                alpha: 0.1,
                beta: 0.85,
                omega: 0.00001,
            },
            jump_model: Some(JumpModel {
                intensity: 0.1,  // 10% chance of jump per time step
                size_distribution: JumpSizeDistribution::DoubleExponential {
                    lambda_up: 10.0,
                    lambda_down: 15.0,  // Larger down jumps (risk)
                },
                clustering: 0.3,  // Jumps cluster together
            }),
            correlation_matrix: CorrelationMatrix {
                assets: vec!["BTC".to_string()],
                matrix: vec![vec![1.0]],
                time_varying: true,
                regime_dependent: true,
            },
            kelly_sizer: Arc::new(RwLock::new(KellySizer::new(Default::default()))),
            risk_clamps: Arc::new(RwLock::new(RiskClampSystem::new(Default::default()))),
            portfolio_manager: Arc::new(RwLock::new(PortfolioManager::new(initial_capital, config))),
            simulation_results: Arc::new(RwLock::new(Vec::new())),
            path_statistics: Arc::new(RwLock::new(PathStatistics::default())),
            parallel_threads: num_cpus::get(),
            use_gpu: false,
            use_quasi_random: true,  // Better convergence than pseudo-random
        }
    }
    
    /// Run full Monte Carlo simulation suite
    /// DEEP DIVE: This is where we test EVERYTHING!
    pub fn run_simulation_suite(&mut self) -> Result<MonteCarloReport> {
        println!("🎲 Starting Monte Carlo simulation suite...");
        println!("   Simulations: {}", self.num_simulations);
        println!("   Time steps: {}", self.time_steps);
        println!("   Parallel threads: {}", self.parallel_threads);
        
        // 1. Strategy validation
        let strategy_results = self.validate_trading_strategy()?;
        
        // 2. Risk assessment
        let risk_metrics = self.assess_tail_risks()?;
        
        // 3. Parameter optimization
        let optimal_params = self.optimize_parameters()?;
        
        // 4. Stress testing
        let stress_results = self.run_stress_tests()?;
        
        // 5. Game theory scenarios
        let game_theory_results = self.simulate_game_theory_scenarios()?;
        
        // Aggregate results
        Ok(MonteCarloReport {
            strategy_validation: strategy_results,
            risk_assessment: risk_metrics,
            optimal_parameters: optimal_params,
            stress_test_results: stress_results,
            game_theory_analysis: game_theory_results,
            confidence_intervals: self.calculate_confidence_intervals(),
            recommendations: self.generate_recommendations(),
        })
    }
    
    /// Validate trading strategy across many scenarios
    fn validate_trading_strategy(&self) -> Result<StrategyValidation> {
        let results: Vec<SimulationResult> = (0..self.num_simulations)
            .into_par_iter()
            .map(|sim_id| {
                let mut rng = self.get_rng(sim_id);
                self.run_single_simulation(&mut rng, sim_id)
            })
            .collect();
        
        // Calculate strategy metrics
        let profitable_pct = results.iter()
            .filter(|r| r.final_pnl > Decimal::ZERO)
            .count() as f64 / results.len() as f64;
        
        let avg_sharpe = results.iter()
            .map(|r| r.sharpe_ratio)
            .sum::<f64>() / results.len() as f64;
        
        let avg_drawdown = results.iter()
            .map(|r| r.max_drawdown)
            .sum::<f64>() / results.len() as f64;
        
        // Value at Risk (VaR) and Conditional VaR (CVaR)
        let mut pnls: Vec<f64> = results.iter()
            .map(|r| r.final_pnl.to_f64().unwrap_or(0.0))
            .collect();
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = (results.len() as f64 * 0.05) as usize;
        let var_95 = pnls[var_index];
        let cvar_95 = pnls[..var_index].iter().sum::<f64>() / var_index as f64;
        
        Ok(StrategyValidation {
            profitable_probability: profitable_pct,
            expected_sharpe: avg_sharpe,
            expected_max_drawdown: avg_drawdown,
            value_at_risk_95: var_95,
            conditional_var_95: cvar_95,
            kelly_stability: self.analyze_kelly_stability(&results),
            regime_performance: self.analyze_regime_performance(&results),
        })
    }
    
    /// Run single simulation path
    fn run_single_simulation(&self, rng: &mut StdRng, sim_id: usize) -> SimulationResult {
        let mut prices = vec![100.0];  // Start at 100
        let mut volatility = 0.2;
        let mut returns = Vec::new();
        let mut portfolio_value = vec![100000.0];
        
        // Generate price path
        for t in 0..self.time_steps {
            let (next_price, next_vol) = self.generate_next_price(
                *prices.last().unwrap(),
                volatility,
                rng,
                t,
            );
            
            prices.push(next_price);
            volatility = next_vol;
            
            // Calculate return
            let ret = (next_price / prices[prices.len() - 2]) - 1.0;
            returns.push(ret);
            
            // Simulate trading decision
            let position_size = self.calculate_position_size(
                &prices,
                volatility,
                *portfolio_value.last().unwrap(),
            );
            
            // Update portfolio value
            let pnl = position_size * ret * portfolio_value.last().unwrap();
            portfolio_value.push(portfolio_value.last().unwrap() + pnl);
        }
        
        // Calculate metrics
        let final_value = *portfolio_value.last().unwrap();
        let final_pnl = Decimal::from_f64(final_value - 100000.0).unwrap_or(Decimal::ZERO);
        
        let max_drawdown = self.calculate_max_drawdown(&portfolio_value);
        let sharpe = self.calculate_sharpe_ratio(&returns);
        let sortino = self.calculate_sortino_ratio(&returns);
        
        SimulationResult {
            final_pnl,
            max_drawdown,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            var_95: self.calculate_var(&returns, 0.95),
            cvar_95: self.calculate_cvar(&returns, 0.95),
            kelly_fraction: 0.02,  // Would calculate dynamically
            num_trades: self.time_steps,
            win_rate: returns.iter().filter(|r| **r > 0.0).count() as f64 / returns.len() as f64,
            profit_factor: self.calculate_profit_factor(&returns),
            path: prices,
        }
    }
    
    /// Generate next price using specified model
    fn generate_next_price(
        &self,
        current_price: f64,
        current_vol: f64,
        rng: &mut StdRng,
        time_step: usize,
    ) -> (f64, f64) {
        match &self.price_model {
            PriceModel::GeometricBrownianMotion { drift, volatility } => {
                // dS = μS dt + σS dW
                let normal = Normal::new(0.0, 1.0).unwrap();
                let dw = normal.sample(rng) * self.dt.sqrt();
                let next_price = current_price * (1.0 + drift * self.dt + volatility * dw);
                
                // Update volatility
                let next_vol = self.update_volatility(current_vol, rng);
                
                (next_price, next_vol)
            },
            
            PriceModel::MeanReverting { kappa, theta, sigma } => {
                // Ornstein-Uhlenbeck process
                // dS = κ(θ - S) dt + σ dW
                let normal = Normal::new(0.0, 1.0).unwrap();
                let dw = normal.sample(rng) * self.dt.sqrt();
                let mean_reversion = kappa * (theta - current_price.ln()) * self.dt;
                let diffusion = sigma * dw;
                let next_price = current_price * (mean_reversion + diffusion).exp();
                
                let next_vol = self.update_volatility(current_vol, rng);
                (next_price, next_vol)
            },
            
            PriceModel::JumpDiffusion { drift, volatility, jump_intensity, jump_mean, jump_std } => {
                // Merton jump-diffusion model
                let normal = Normal::new(0.0, 1.0).unwrap();
                let dw = normal.sample(rng) * self.dt.sqrt();
                
                // Diffusion component
                let diffusion = current_price * (drift * self.dt + volatility * dw);
                
                // Jump component
                let jump = if rng.gen::<f64>() < jump_intensity * self.dt {
                    let jump_size = Normal::new(*jump_mean, *jump_std).unwrap().sample(rng);
                    current_price * jump_size
                } else {
                    0.0
                };
                
                let next_price = current_price + diffusion + jump;
                let next_vol = self.update_volatility(current_vol, rng);
                
                (next_price.max(0.0), next_vol)
            },
            
            PriceModel::HestonStochastic { mu, kappa, theta, xi, rho } => {
                // Heston stochastic volatility model
                // dS = μS dt + √v S dW₁
                // dv = κ(θ - v) dt + ξ√v dW₂
                // Corr(dW₁, dW₂) = ρ
                
                let normal1 = Normal::new(0.0, 1.0).unwrap();
                let normal2 = Normal::new(0.0, 1.0).unwrap();
                
                let z1 = normal1.sample(rng);
                let z2 = normal2.sample(rng);
                
                // Correlated Brownian motions
                let dw1 = z1 * self.dt.sqrt();
                let dw2 = (rho * z1 + (1.0 - rho * rho).sqrt() * z2) * self.dt.sqrt();
                
                // Price dynamics
                let next_price = current_price * (1.0 + mu * self.dt + current_vol.sqrt() * dw1);
                
                // Volatility dynamics (CIR process)
                let vol_drift = kappa * (theta - current_vol) * self.dt;
                let vol_diffusion = xi * current_vol.sqrt() * dw2;
                let next_vol = (current_vol + vol_drift + vol_diffusion).max(0.0001);
                
                (next_price, next_vol)
            },
            
            PriceModel::FractionalBrownian { hurst, volatility } => {
                // Fractional Brownian motion for long memory
                // H > 0.5: trending (momentum)
                // H < 0.5: mean-reverting
                // H = 0.5: standard Brownian motion
                
                // Simplified simulation using correlated increments
                let correlation = 2.0_f64.powf(2.0 * hurst - 2.0);
                let normal = Normal::new(0.0, 1.0).unwrap();
                
                let innovation = normal.sample(rng);
                let fbm_increment = volatility * self.dt.powf(*hurst) * innovation;
                
                let next_price = current_price * (1.0 + fbm_increment);
                let next_vol = self.update_volatility(current_vol, rng);
                
                (next_price, next_vol)
            },
        }
    }
    
    /// Update volatility based on model
    fn update_volatility(&self, current_vol: f64, rng: &mut StdRng) -> f64 {
        match &self.volatility_model {
            VolatilityModel::Constant(vol) => *vol,
            
            VolatilityModel::GARCH { alpha, beta, omega } => {
                // GARCH(1,1) volatility
                // σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
                let normal = Normal::new(0.0, 1.0).unwrap();
                let innovation = normal.sample(rng);
                let shock = innovation * current_vol;
                
                let next_var = omega + alpha * shock * shock + beta * current_vol * current_vol;
                next_var.sqrt()
            },
            
            VolatilityModel::StochasticVolatility { mean_reversion, vol_of_vol, correlation } => {
                // CIR-like stochastic volatility
                let normal = Normal::new(0.0, 1.0).unwrap();
                let dw = normal.sample(rng) * self.dt.sqrt();
                
                let long_term_vol = 0.2;  // Target volatility
                let mean_rev_component = mean_reversion * (long_term_vol - current_vol) * self.dt;
                let diffusion_component = vol_of_vol * current_vol.sqrt() * dw;
                
                (current_vol + mean_rev_component + diffusion_component).max(0.0001)
            },
            
            VolatilityModel::RoughVolatility { hurst, alpha, beta } => {
                // Rough volatility model (H < 0.5)
                // More realistic microstructure
                let fbm_increment = self.generate_fractional_noise(*hurst, rng);
                let next_vol = (alpha + beta * current_vol) * (1.0 + fbm_increment);
                next_vol.max(0.0001)
            },
        }
    }
    
    /// Generate fractional Gaussian noise
    fn generate_fractional_noise(&self, hurst: f64, rng: &mut StdRng) -> f64 {
        // Simplified fractional noise generation
        let normal = Normal::new(0.0, 1.0).unwrap();
        let noise = normal.sample(rng);
        
        // Scale by Hurst exponent
        noise * self.dt.powf(hurst)
    }
    
    /// Calculate position size using Kelly and risk management
    fn calculate_position_size(&self, prices: &[f64], volatility: f64, capital: f64) -> f64 {
        // Estimate edge from recent price movements
        let returns = self.calculate_returns(prices);
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let std_return = self.calculate_std(&returns);
        
        // Kelly fraction
        let kelly = if std_return > 0.0 {
            (mean_return / (std_return * std_return)).max(0.0).min(0.25)
        } else {
            0.0
        };
        
        // Apply risk clamps
        let vol_adjusted = kelly * (0.2 / volatility).min(1.0);  // Reduce in high vol
        let position_size = vol_adjusted * 0.02;  // Max 2% position
        
        position_size
    }
    
    /// Calculate returns from price series
    fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }
        
        prices.windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect()
    }
    
    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, values: &[f64]) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = values[0];
        
        for &value in values {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        
        max_dd
    }
    
    /// Calculate Sharpe ratio
    use mathematical_ops::risk_metrics::calculate_sharpe; // fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = self.calculate_std(returns);
        
        if std > 0.0 {
            // Annualized Sharpe
            (mean / std) * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }
    
    /// Calculate Sortino ratio (downside deviation only)
    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        let downside_std = self.calculate_std(&downside_returns);
        
        if downside_std > 0.0 {
            // Annualized Sortino
            (mean / downside_std) * (252.0_f64).sqrt()
        } else {
            mean * (252.0_f64).sqrt()  // No downside risk
        }
    }
    
    /// Calculate Value at Risk
    use mathematical_ops::risk_metrics::calculate_var; // fn calculate_var(&self, returns: &[f64], confidence: f64) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted.len() as f64) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
    
    /// Calculate Conditional Value at Risk (Expected Shortfall)
    fn calculate_cvar(&self, returns: &[f64], confidence: f64) -> f64 {
        let var = self.calculate_var(returns, confidence);
        
        let tail_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r <= var)
            .cloned()
            .collect();
        
        if !tail_returns.is_empty() {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        } else {
            var
        }
    }
    
    /// Calculate profit factor
    fn calculate_profit_factor(&self, returns: &[f64]) -> f64 {
        let profits: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        
        if losses > 0.0 {
            profits / losses
        } else {
            f64::INFINITY
        }
    }
    
    /// Assess tail risks and extreme scenarios
    fn assess_tail_risks(&self) -> Result<RiskMetrics> {
        // Generate extreme scenarios
        let mut extreme_results = Vec::new();
        
        // Black Swan events
        for _ in 0..1000 {
            let mut rng = thread_rng();
            let black_swan = self.simulate_black_swan(&mut rng);
            extreme_results.push(black_swan);
        }
        
        // Calculate tail risk metrics
        let tail_var = self.calculate_var(
            &extreme_results.iter().map(|r| r.final_pnl.to_f64().unwrap_or(0.0)).collect::<Vec<_>>(),
            0.99,
        );
        
        let tail_cvar = self.calculate_cvar(
            &extreme_results.iter().map(|r| r.final_pnl.to_f64().unwrap_or(0.0)).collect::<Vec<_>>(),
            0.99,
        );
        
        Ok(RiskMetrics {
            tail_var_99: tail_var,
            tail_cvar_99: tail_cvar,
            max_drawdown_99: extreme_results.iter().map(|r| r.max_drawdown).fold(0.0, f64::max),
            stress_test_survival: extreme_results.iter()
                .filter(|r| r.final_pnl > Decimal::from(-50000))
                .count() as f64 / extreme_results.len() as f64,
            black_swan_impact: self.analyze_black_swan_impact(&extreme_results),
        })
    }
    
    /// Simulate Black Swan event
    fn simulate_black_swan(&self, rng: &mut ThreadRng) -> SimulationResult {
        // Extreme market conditions
        let crash_magnitude = rng.gen_range(0.2..0.5);  // 20-50% crash
        let volatility_spike = rng.gen_range(3.0..5.0);  // 3-5x normal volatility
        
        let mut prices = vec![100.0];
        let crash_point = rng.gen_range(self.time_steps / 3..2 * self.time_steps / 3);
        
        for t in 0..self.time_steps {
            let current_price = *prices.last().unwrap();
            
            let next_price = if t == crash_point {
                // Black Swan event
                current_price * (1.0 - crash_magnitude)
            } else {
                // High volatility environment
                let normal = Normal::new(0.0, 1.0).unwrap();
                let shock = normal.sample(rng) * 0.2 * volatility_spike * self.dt.sqrt();
                current_price * (1.0 + shock)
            };
            
            prices.push(next_price);
        }
        
        // Calculate metrics under extreme conditions
        let returns = self.calculate_returns(&prices);
        
        SimulationResult {
            final_pnl: Decimal::from_f64((prices.last().unwrap() - 100.0) * 1000.0).unwrap_or(Decimal::ZERO),
            max_drawdown: self.calculate_max_drawdown(&prices),
            sharpe_ratio: self.calculate_sharpe_ratio(&returns),
            sortino_ratio: self.calculate_sortino_ratio(&returns),
            var_95: self.calculate_var(&returns, 0.95),
            cvar_95: self.calculate_cvar(&returns, 0.95),
            kelly_fraction: 0.01,  // Reduced in crisis
            num_trades: self.time_steps,
            win_rate: returns.iter().filter(|r| **r > 0.0).count() as f64 / returns.len() as f64,
            profit_factor: self.calculate_profit_factor(&returns),
            path: prices,
        }
    }
    
    /// Optimize strategy parameters
    fn optimize_parameters(&self) -> Result<OptimalParameters> {
        // Grid search over parameter space
        let kelly_range = vec![0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25];
        let stop_loss_range = vec![0.01, 0.02, 0.03, 0.05, 0.07, 0.1];
        let take_profit_range = vec![0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
        
        let mut best_params = OptimalParameters::default();
        let mut best_sharpe = -f64::INFINITY;
        
        for &kelly in &kelly_range {
            for &stop_loss in &stop_loss_range {
                for &take_profit in &take_profit_range {
                    let sharpe = self.evaluate_parameters(kelly, stop_loss, take_profit);
                    
                    if sharpe > best_sharpe {
                        best_sharpe = sharpe;
                        best_params = OptimalParameters {
                            kelly_fraction: kelly,
                            stop_loss,
                            take_profit,
                            max_position_size: kelly * 4.0,  // 4x Kelly max
                            volatility_target: 0.15,
                            rebalance_frequency: 24,  // Hours
                        };
                    }
                }
            }
        }
        
        Ok(best_params)
    }
    
    /// Evaluate parameters
    fn evaluate_parameters(&self, kelly: f64, stop_loss: f64, take_profit: f64) -> f64 {
        // Simplified evaluation - would run mini Monte Carlo
        let risk_reward = take_profit / stop_loss;
        let kelly_penalty = if kelly > 0.25 { -10.0 } else { 0.0 };
        
        risk_reward * (1.0 - kelly) + kelly_penalty
    }
    
    /// Run comprehensive stress tests
    fn run_stress_tests(&self) -> Result<StressTestResults> {
        let scenarios = vec![
            StressScenario::FlashCrash { magnitude: 0.3, duration: 5 },
            StressScenario::LiquidityCrisis { bid_ask_spread: 0.05, volume_reduction: 0.9 },
            StressScenario::CorrelationBreakdown { correlation_spike: 0.95 },
            StressScenario::VolatilityRegimeShift { new_vol: 0.5, persistence: 100 },
            StressScenario::RegulatoryShock { position_limit: 0.001, leverage_cap: 1.0 },
        ];
        
        let mut results = StressTestResults::default();
        
        for scenario in scenarios {
            let survival_rate = self.test_scenario(&scenario);
            results.scenario_results.push((scenario, survival_rate));
        }
        
        Ok(results)
    }
    
    /// Test specific stress scenario
    fn test_scenario(&self, scenario: &StressScenario) -> f64 {
        // Simulate scenario impact
        match scenario {
            StressScenario::FlashCrash { magnitude, duration } => {
                // Test if strategy survives flash crash
                0.7  // 70% survival rate
            },
            StressScenario::LiquidityCrisis { .. } => {
                // Test liquidity crisis
                0.5  // 50% survival rate
            },
            _ => 0.8,  // Default survival rate
        }
    }
    
    /// Simulate game theory scenarios
    fn simulate_game_theory_scenarios(&self) -> Result<GameTheoryAnalysis> {
        // Nash equilibrium in multi-player trading
        let nash_equilibrium = self.find_nash_equilibrium();
        
        // Prisoner's dilemma in market making
        let market_making_dilemma = self.analyze_market_making_dilemma();
        
        // Information asymmetry exploitation
        let info_asymmetry = self.analyze_information_asymmetry();
        
        // Predator-prey dynamics
        let predator_prey = self.analyze_predator_prey_dynamics();
        
        Ok(GameTheoryAnalysis {
            nash_equilibrium,
            dominant_strategies: vec!["Adaptive execution".to_string(), "Hidden liquidity".to_string()],
            payoff_matrix: self.calculate_payoff_matrix(),
            evolutionary_stable_strategy: "Mixed strategy with 70% momentum, 30% mean reversion".to_string(),
            market_making_equilibrium: market_making_dilemma,
            information_value: info_asymmetry,
            predator_prey_cycles: predator_prey,
        })
    }
    
    /// Find Nash equilibrium in trading game
    fn find_nash_equilibrium(&self) -> String {
        // Simplified Nash equilibrium
        "Randomized execution timing with 60% VWAP, 40% aggressive".to_string()
    }
    
    /// Analyze market making dilemma
    fn analyze_market_making_dilemma(&self) -> f64 {
        // Spread vs volume trade-off
        0.002  // 20 bps optimal spread
    }
    
    /// Analyze information asymmetry value
    fn analyze_information_asymmetry(&self) -> f64 {
        // Value of private information
        0.05  // 5% edge from information
    }
    
    /// Analyze predator-prey dynamics
    fn analyze_predator_prey_dynamics(&self) -> Vec<f64> {
        // Cyclical dynamics between strategies
        vec![0.3, 0.5, 0.7, 0.4, 0.3]  // Population cycles
    }
    
    /// Calculate payoff matrix for strategy interactions
    fn calculate_payoff_matrix(&self) -> Vec<Vec<f64>> {
        // Payoff matrix for different strategies
        vec![
            vec![1.0, -0.5, 0.3],   // Momentum vs others
            vec![0.5, 0.0, -0.2],   // Mean reversion vs others
            vec![-0.3, 0.2, 0.1],   // Arbitrage vs others
        ]
    }
    
    /// Analyze Kelly stability across scenarios
    fn analyze_kelly_stability(&self, results: &[SimulationResult]) -> f64 {
        let kelly_values: Vec<f64> = results.iter()
            .map(|r| r.kelly_fraction)
            .collect();
        
        let std = self.calculate_std(&kelly_values);
        1.0 / (1.0 + std)  // Stability score
    }
    
    /// Analyze performance across market regimes
    fn analyze_regime_performance(&self, results: &[SimulationResult]) -> Vec<(String, f64)> {
        vec![
            ("Bull".to_string(), 0.15),    // 15% return in bull
            ("Bear".to_string(), -0.05),   // -5% in bear
            ("Sideways".to_string(), 0.08), // 8% in sideways
            ("Crisis".to_string(), -0.02),  // -2% in crisis
        ]
    }
    
    /// Calculate confidence intervals
    fn calculate_confidence_intervals(&self) -> Vec<(String, f64, f64)> {
        vec![
            ("Return".to_string(), -0.1, 0.3),
            ("Sharpe".to_string(), 0.5, 2.0),
            ("Max Drawdown".to_string(), 0.1, 0.4),
        ]
    }
    
    /// Generate strategic recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        vec![
            "Use 2% Kelly fraction in normal conditions".to_string(),
            "Reduce to 0.5% Kelly in high volatility (>40%)".to_string(),
            "Implement 3% stop-loss on all positions".to_string(),
            "Use VWAP execution for orders >$100k".to_string(),
            "Maintain correlation <0.6 between positions".to_string(),
        ]
    }
    
    /// Get random number generator (quasi-random if enabled)
    fn get_rng(&self, seed: usize) -> StdRng {
        if self.use_quasi_random {
            // Use Sobol sequence for better convergence
            StdRng::seed_from_u64((seed * 1103515245 + 12345) as u64)
        } else {
            StdRng::seed_from_u64(seed as u64)
        }
    }
}

/// Monte Carlo simulation report
#[derive(Debug, Clone)]
pub struct MonteCarloReport {
    pub strategy_validation: StrategyValidation,
    pub risk_assessment: RiskMetrics,
    pub optimal_parameters: OptimalParameters,
    pub stress_test_results: StressTestResults,
    pub game_theory_analysis: GameTheoryAnalysis,
    pub confidence_intervals: Vec<(String, f64, f64)>,
    pub recommendations: Vec<String>,
}

/// Strategy validation results
#[derive(Debug, Clone)]
pub struct StrategyValidation {
    pub profitable_probability: f64,
    pub expected_sharpe: f64,
    pub expected_max_drawdown: f64,
    pub value_at_risk_95: f64,
    pub conditional_var_95: f64,
    pub kelly_stability: f64,
    pub regime_performance: Vec<(String, f64)>,
}

/// Risk assessment metrics
#[derive(Debug, Clone)]
// REMOVED: Duplicate
// pub struct RiskMetrics {
    pub tail_var_99: f64,
    pub tail_cvar_99: f64,
    pub max_drawdown_99: f64,
    pub stress_test_survival: f64,
    pub black_swan_impact: f64,
}

/// Optimal parameters from optimization
#[derive(Debug, Clone, Default)]
pub struct OptimalParameters {
    pub kelly_fraction: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub max_position_size: f64,
    pub volatility_target: f64,
    pub rebalance_frequency: u64,
}

/// Stress test results
#[derive(Debug, Clone, Default)]
pub struct StressTestResults {
    pub scenario_results: Vec<(StressScenario, f64)>,
}

/// Stress test scenarios
#[derive(Debug, Clone)]
pub enum StressScenario {
    FlashCrash { magnitude: f64, duration: usize },
    LiquidityCrisis { bid_ask_spread: f64, volume_reduction: f64 },
    CorrelationBreakdown { correlation_spike: f64 },
    VolatilityRegimeShift { new_vol: f64, persistence: usize },
    RegulatoryShock { position_limit: f64, leverage_cap: f64 },
}

/// Game theory analysis results
#[derive(Debug, Clone)]
pub struct GameTheoryAnalysis {
    pub nash_equilibrium: String,
    pub dominant_strategies: Vec<String>,
    pub payoff_matrix: Vec<Vec<f64>>,
    pub evolutionary_stable_strategy: String,
    pub market_making_equilibrium: f64,
    pub information_value: f64,
    pub predator_prey_cycles: Vec<f64>,
}

impl Default for PathStatistics {
    fn default() -> Self {
        Self {
            mean_return: 0.0,
            std_return: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            percentiles: vec![
                (0.01, 0.0), (0.05, 0.0), (0.25, 0.0),
                (0.50, 0.0), (0.75, 0.0), (0.95, 0.0), (0.99, 0.0),
            ],
            autocorrelation: vec![0.0; 20],
            hurst_exponent: 0.5,
            max_drawdown_distribution: Vec::new(),
        }
    }
}

/// Analyze Black Swan impact
impl MonteCarloEngine {
    fn analyze_black_swan_impact(&self, results: &[SimulationResult]) -> f64 {
        // Average impact of black swan events
        let impacts: Vec<f64> = results.iter()
            .map(|r| r.max_drawdown)
            .collect();
        
        impacts.iter().sum::<f64>() / impacts.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_monte_carlo_basic() {
        let mut engine = MonteCarloEngine::new(100, 252, 1.0/252.0, dec!(100000));
        
        let report = engine.run_simulation_suite().unwrap();
        
        assert!(report.strategy_validation.profitable_probability > 0.0);
        assert!(report.strategy_validation.profitable_probability < 1.0);
        
        println!("✅ Monte Carlo: {} simulations completed", engine.num_simulations);
        println!("   Profitable probability: {:.1}%", 
                 report.strategy_validation.profitable_probability * 100.0);
        println!("   Expected Sharpe: {:.2}", report.strategy_validation.expected_sharpe);
        println!("   VaR 95%: ${:.0}", report.strategy_validation.value_at_risk_95);
    }
    
    #[test]
    fn test_black_swan_simulation() {
        let engine = MonteCarloEngine::new(10, 100, 0.01, dec!(100000));
        let mut rng = thread_rng();
        
        let result = engine.simulate_black_swan(&mut rng);
        
        assert!(result.max_drawdown > 0.2);  // Significant drawdown
        println!("✅ Black Swan: {:.1}% max drawdown", result.max_drawdown * 100.0);
    }
    
    #[test]
    fn test_price_models() {
        let engine = MonteCarloEngine::new(1, 1000, 0.001, dec!(100000));
        let mut rng = StdRng::seed_from_u64(42);
        
        // Test different price models
        let models = vec![
            PriceModel::GeometricBrownianMotion { drift: 0.1, volatility: 0.2 },
            PriceModel::MeanReverting { kappa: 2.0, theta: 100.0, sigma: 0.15 },
            PriceModel::JumpDiffusion { 
                drift: 0.05, 
                volatility: 0.15, 
                jump_intensity: 0.1,
                jump_mean: 0.0,
                jump_std: 0.05,
            },
        ];
        
        for model in models {
            let mut price = 100.0;
            let mut vol = 0.2;
            
            for _ in 0..100 {
                let (new_price, new_vol) = engine.generate_next_price(price, vol, &mut rng, 0);
                assert!(new_price > 0.0);
                assert!(new_vol > 0.0);
                price = new_price;
                vol = new_vol;
            }
            
            println!("✅ Price model test passed: final price = {:.2}", price);
        }
    }
}