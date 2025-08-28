//! Bot4 RiskQuant Agent
//! Advanced risk management, portfolio optimization, and Kelly criterion

use anyhow::Result;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use rmcp::{
    server::{Server, ServerBuilder, ToolHandler},
    transport::DockerTransport,
    types::{Tool, ToolCall, ToolResult, Resource, Prompt},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use statrs::distribution::{Normal, ContinuousCDF};
use ndarray::{Array1, Array2};

mod portfolio;
mod kelly;
mod var;
mod risk_metrics;

use portfolio::PortfolioOptimizer;
use kelly::KellyCalculator;
use var::VaRCalculator;
use risk_metrics::RiskMetrics;

/// RiskQuant agent implementation
struct RiskQuantAgent {
    redis: ConnectionManager,
    portfolio_optimizer: PortfolioOptimizer,
    kelly_calculator: KellyCalculator,
    var_calculator: VaRCalculator,
    risk_metrics: RiskMetrics,
    max_position_size: Decimal,
    max_portfolio_risk: Decimal,
    max_correlation: Decimal,
}

impl RiskQuantAgent {
    async fn new() -> Result<Self> {
        // Connect to Redis
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let client = redis::Client::open(redis_url)?;
        let redis = ConnectionManager::new(client).await?;
        
        // Risk parameters from environment
        let max_position_size = std::env::var("MAX_POSITION_SIZE")
            .unwrap_or_else(|_| "0.02".to_string())
            .parse::<Decimal>()?;
        
        let max_portfolio_risk = std::env::var("MAX_PORTFOLIO_RISK")
            .unwrap_or_else(|_| "0.15".to_string())
            .parse::<Decimal>()?;
        
        let max_correlation = std::env::var("MAX_CORRELATION")
            .unwrap_or_else(|_| "0.7".to_string())
            .parse::<Decimal>()?;
        
        Ok(Self {
            redis,
            portfolio_optimizer: PortfolioOptimizer::new(),
            kelly_calculator: KellyCalculator::new(),
            var_calculator: VaRCalculator::new(),
            risk_metrics: RiskMetrics::new(),
            max_position_size,
            max_portfolio_risk,
            max_correlation,
        })
    }
    
    /// Calculate Kelly criterion for position sizing
    async fn calculate_kelly(&self, win_prob: f64, win_return: f64, loss_return: f64) -> Result<ToolResult> {
        info!("Calculating Kelly criterion: win_prob={}, win_return={}, loss_return={}", 
              win_prob, win_return, loss_return);
        
        let kelly_fraction = self.kelly_calculator.calculate(win_prob, win_return, loss_return)?;
        let fractional_kelly = kelly_fraction * 0.25; // Conservative 1/4 Kelly
        
        // Apply position size limits
        let final_size = Decimal::from_f64_retain(fractional_kelly)
            .unwrap_or(dec!(0))
            .min(self.max_position_size);
        
        Ok(ToolResult::Success(serde_json::json!({
            "full_kelly": kelly_fraction,
            "fractional_kelly": fractional_kelly,
            "final_position_size": final_size,
            "max_allowed": self.max_position_size,
            "recommendation": if final_size > dec!(0) { "TRADE" } else { "SKIP" }
        })))
    }
    
    /// Calculate Value at Risk (VaR)
    async fn calculate_var(&self, returns: Vec<f64>, confidence: f64, horizon: u32) -> Result<ToolResult> {
        info!("Calculating VaR: {} returns, confidence={}, horizon={} days", 
              returns.len(), confidence, horizon);
        
        let var_result = self.var_calculator.calculate(&returns, confidence, horizon)?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "var_amount": var_result.var_amount,
            "confidence_level": confidence,
            "horizon_days": horizon,
            "expected_shortfall": var_result.expected_shortfall,
            "max_loss": var_result.max_loss,
            "violation": var_result.var_amount > self.max_portfolio_risk
        })))
    }
    
    /// Optimize portfolio weights
    async fn optimize_portfolio(&self, assets: Vec<String>, returns: Vec<Vec<f64>>, target_return: f64) -> Result<ToolResult> {
        info!("Optimizing portfolio: {} assets, target_return={}", assets.len(), target_return);
        
        let weights = self.portfolio_optimizer.optimize(&assets, &returns, target_return)?;
        
        // Check correlation constraints
        let correlations = self.portfolio_optimizer.calculate_correlations(&returns)?;
        let max_corr = correlations.iter().flatten()
            .filter(|&&c| c < 1.0) // Exclude self-correlation
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        
        let correlation_violation = Decimal::from_f64_retain(max_corr)
            .unwrap_or(dec!(0)) > self.max_correlation;
        
        Ok(ToolResult::Success(serde_json::json!({
            "assets": assets,
            "optimal_weights": weights,
            "target_return": target_return,
            "max_correlation": max_corr,
            "correlation_violation": correlation_violation,
            "sharpe_ratio": self.portfolio_optimizer.calculate_sharpe(&returns, &weights)?
        })))
    }
    
    /// Calculate portfolio risk metrics
    async fn calculate_risk_metrics(&self, positions: Vec<serde_json::Value>) -> Result<ToolResult> {
        info!("Calculating risk metrics for {} positions", positions.len());
        
        let metrics = self.risk_metrics.calculate(&positions)?;
        
        // Check risk violations
        let violations = vec![
            ("position_size", metrics.max_position > self.max_position_size),
            ("portfolio_risk", metrics.portfolio_risk > self.max_portfolio_risk),
            ("correlation", metrics.max_correlation > self.max_correlation),
            ("leverage", metrics.leverage > dec!(3)),
        ];
        
        Ok(ToolResult::Success(serde_json::json!({
            "metrics": metrics,
            "violations": violations.into_iter()
                .filter(|(_, v)| *v)
                .map(|(name, _)| name)
                .collect::<Vec<_>>(),
            "risk_score": metrics.calculate_risk_score(),
            "recommendation": if violations.iter().any(|(_, v)| *v) { 
                "REDUCE_RISK" 
            } else { 
                "ACCEPTABLE" 
            }
        })))
    }
    
    /// Monte Carlo simulation for risk assessment
    async fn monte_carlo_simulation(&self, initial_value: f64, drift: f64, volatility: f64, 
                                    time_horizon: f64, num_simulations: u32) -> Result<ToolResult> {
        info!("Running Monte Carlo simulation: {} paths, horizon={}", num_simulations, time_horizon);
        
        let mut final_values = Vec::with_capacity(num_simulations as usize);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        for _ in 0..num_simulations {
            let random_shock = normal.sample(&mut rand::thread_rng());
            let final_value = initial_value * (
                (drift - volatility.powi(2) / 2.0) * time_horizon + 
                volatility * time_horizon.sqrt() * random_shock
            ).exp();
            final_values.push(final_value);
        }
        
        // Calculate statistics
        let mean = final_values.iter().sum::<f64>() / final_values.len() as f64;
        let p5 = self.var_calculator.percentile(&final_values, 0.05)?;
        let p95 = self.var_calculator.percentile(&final_values, 0.95)?;
        
        Ok(ToolResult::Success(serde_json::json!({
            "initial_value": initial_value,
            "mean_outcome": mean,
            "percentile_5": p5,
            "percentile_95": p95,
            "probability_of_profit": final_values.iter().filter(|&&v| v > initial_value).count() as f64 
                / final_values.len() as f64,
            "max_drawdown": (initial_value - final_values.iter().cloned().fold(f64::INFINITY, f64::min)) 
                / initial_value,
            "num_simulations": num_simulations
        })))
    }
}

#[async_trait]
impl ToolHandler for RiskQuantAgent {
    async fn handle_tool_call(&self, tool_call: ToolCall) -> ToolResult {
        match tool_call.name.as_str() {
            "calculate_kelly" => {
                let win_prob = tool_call.arguments["win_probability"].as_f64().unwrap_or(0.5);
                let win_return = tool_call.arguments["win_return"].as_f64().unwrap_or(0.0);
                let loss_return = tool_call.arguments["loss_return"].as_f64().unwrap_or(0.0);
                self.calculate_kelly(win_prob, win_return, loss_return).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to calculate Kelly: {}", e))
                })
            }
            "calculate_var" => {
                let returns: Vec<f64> = tool_call.arguments["returns"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                    .unwrap_or_default();
                let confidence = tool_call.arguments["confidence"].as_f64().unwrap_or(0.95);
                let horizon = tool_call.arguments["horizon"].as_u64().unwrap_or(1) as u32;
                self.calculate_var(returns, confidence, horizon).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to calculate VaR: {}", e))
                })
            }
            "optimize_portfolio" => {
                let assets: Vec<String> = tool_call.arguments["assets"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let returns: Vec<Vec<f64>> = tool_call.arguments["returns"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| {
                        v.as_array().map(|inner| inner.iter().filter_map(|x| x.as_f64()).collect())
                    }).collect())
                    .unwrap_or_default();
                let target_return = tool_call.arguments["target_return"].as_f64().unwrap_or(0.0);
                self.optimize_portfolio(assets, returns, target_return).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to optimize portfolio: {}", e))
                })
            }
            "calculate_risk_metrics" => {
                let positions = tool_call.arguments["positions"].as_array()
                    .map(|arr| arr.to_vec())
                    .unwrap_or_default();
                self.calculate_risk_metrics(positions).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to calculate risk metrics: {}", e))
                })
            }
            "monte_carlo_simulation" => {
                let initial = tool_call.arguments["initial_value"].as_f64().unwrap_or(1000.0);
                let drift = tool_call.arguments["drift"].as_f64().unwrap_or(0.05);
                let volatility = tool_call.arguments["volatility"].as_f64().unwrap_or(0.2);
                let horizon = tool_call.arguments["time_horizon"].as_f64().unwrap_or(1.0);
                let sims = tool_call.arguments["num_simulations"].as_u64().unwrap_or(10000) as u32;
                self.monte_carlo_simulation(initial, drift, volatility, horizon, sims).await.unwrap_or_else(|e| {
                    ToolResult::Error(format!("Failed to run Monte Carlo: {}", e))
                })
            }
            _ => ToolResult::Error(format!("Unknown tool: {}", tool_call.name))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .init();
    
    info!("Starting Bot4 RiskQuant Agent v1.0");
    
    // Create agent
    let agent = RiskQuantAgent::new().await?;
    
    // Define tools
    let tools = vec![
        Tool {
            name: "calculate_kelly".to_string(),
            description: "Calculate Kelly criterion for optimal position sizing".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "win_probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "win_return": {"type": "number"},
                    "loss_return": {"type": "number"}
                },
                "required": ["win_probability", "win_return", "loss_return"]
            }),
        },
        Tool {
            name: "calculate_var".to_string(),
            description: "Calculate Value at Risk (VaR) and Expected Shortfall".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"}},
                    "confidence": {"type": "number", "default": 0.95},
                    "horizon": {"type": "integer", "default": 1}
                },
                "required": ["returns"]
            }),
        },
        Tool {
            name: "optimize_portfolio".to_string(),
            description: "Optimize portfolio weights using Markowitz optimization".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "assets": {"type": "array", "items": {"type": "string"}},
                    "returns": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "target_return": {"type": "number"}
                },
                "required": ["assets", "returns", "target_return"]
            }),
        },
        Tool {
            name: "calculate_risk_metrics".to_string(),
            description: "Calculate comprehensive portfolio risk metrics".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "positions": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["positions"]
            }),
        },
        Tool {
            name: "monte_carlo_simulation".to_string(),
            description: "Run Monte Carlo simulation for risk assessment".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "initial_value": {"type": "number"},
                    "drift": {"type": "number"},
                    "volatility": {"type": "number"},
                    "time_horizon": {"type": "number"},
                    "num_simulations": {"type": "integer", "default": 10000}
                },
                "required": ["initial_value", "drift", "volatility", "time_horizon"]
            }),
        },
    ];
    
    // Build and run MCP server
    let server = ServerBuilder::new("riskquant-agent", "1.0.0")
        .with_tools(tools)
        .with_tool_handler(agent)
        .build()?;
    
    // Use Docker transport
    let transport = DockerTransport::new()?;
    server.run(transport).await?;
    
    Ok(())
}