use domain_types::TrainingResult;
//! # REINFORCEMENT LEARNING FRAMEWORK - Learn from Market Interaction
//! Blake (ML Lead) + Full Team Collaboration
//! 
//! External Research Applied:
//! - "Deep Reinforcement Learning for Trading" - Deng et al. (2017)
//! - "Multi-Agent Reinforcement Learning in Finance" - Ganesh et al. (2019)
//! - "PPO for Portfolio Management" - Jiang et al. (2017)
//! - "Trading via Image Classification" - Sezer & Ozbayoglu (2018)

pub mod dqn;
pub mod ppo;
pub mod replay_buffer;
pub mod environment;
pub mod reward_shaping;
pub mod multi_agent;

use std::sync::Arc;
use parking_lot::RwLock;
use ndarray::{Array1, Array2};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// RL Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RLConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Discount factor (gamma)
    pub discount_factor: f64,
    
    /// Exploration rate (epsilon for epsilon-greedy)
    pub exploration_rate: f64,
    
    /// Minimum exploration rate
    pub min_exploration_rate: f64,
    
    /// Exploration decay rate
    pub exploration_decay: f64,
    
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    
    /// Batch size for training
    pub batch_size: usize,
    
    /// Update target network every N steps
    pub target_update_frequency: u32,
    
    /// Maximum episode length
    pub max_episode_steps: u32,
    
    /// Reward scaling factor
    pub reward_scale: f64,
    
    /// Enable prioritized experience replay
    pub prioritized_replay: bool,
    
    /// PPO clip range
    pub ppo_clip_range: f64,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.0001,
            discount_factor: 0.99,
            exploration_rate: 1.0,
            min_exploration_rate: 0.01,
            exploration_decay: 0.995,
            replay_buffer_size: 100_000,
            batch_size: 32,
            target_update_frequency: 1000,
            max_episode_steps: 1000,
            reward_scale: 1.0,
            prioritized_replay: true,
            ppo_clip_range: 0.2,
        }
    }
}

/// Trading environment state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct State {
    /// Market features
    pub market_features: Array1<f64>,
    
    /// Portfolio state
    pub portfolio_state: PortfolioState,
    
    /// Technical indicators
    pub indicators: Array1<f64>,
    
    /// Order book state
    pub orderbook_features: Array1<f64>,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Current step in episode
    pub step: u32,
}

/// Portfolio state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PortfolioState {
    /// Current cash balance
    pub cash: f64,
    
    /// Current positions (symbol -> quantity)
    pub positions: HashMap<String, f64>,
    
    /// Total portfolio value
    pub total_value: f64,
    
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    
    /// Realized PnL
    pub realized_pnl: f64,
    
    /// Current leverage
    pub leverage: f64,
    
    /// Open orders
    pub open_orders: u32,
}

/// Trading action
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum Action {
    /// Hold position
    Hold,
    
    /// Market buy
    Buy { size: f64 },
    
    /// Market sell
    Sell { size: f64 },
    
    /// Limit buy
    LimitBuy { price: f64, size: f64 },
    
    /// Limit sell
    LimitSell { price: f64, size: f64 },
    
    /// Close position
    Close { symbol: String },
    
    /// Close all positions
    CloseAll,
    
    /// Adjust position
    AdjustPosition { symbol: String, target_size: f64 },
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct Experience {
    pub state: State,
    pub action: Action,
    pub reward: f64,
    pub next_state: State,
    pub done: bool,
    pub info: ExperienceInfo,
}

/// Additional experience information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ExperienceInfo {
    pub timestamp: DateTime<Utc>,
    pub episode: u32,
    pub step: u32,
    pub td_error: Option<f64>,
    pub value_estimate: Option<f64>,
    pub action_probabilities: Option<Vec<f64>>,
}

/// Base RL Agent trait
#[async_trait::async_trait]
pub trait RLAgent: Send + Sync {
    /// Select action given state
    async fn act(&self, state: &State) -> Action;
    
    /// Update agent with experience
    async fn learn(&mut self, experience: Experience);
    
    /// Train on batch of experiences
    async fn train_batch(&mut self, batch: Vec<Experience>) -> TrainingResult;
    
    /// Save model checkpoint
    async fn save_checkpoint(&self, path: &str) -> Result<(), RLError>;
    
    /// Load model checkpoint
    async fn load_checkpoint(&mut self, path: &str) -> Result<(), RLError>;
    
    /// Get current performance metrics
    fn get_metrics(&self) -> AgentMetrics;
    
    /// Reset for new episode
    fn reset(&mut self);
}

/// Training result
#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::TrainingResult
// pub struct TrainingResult {
    pub loss: f64,
    pub value_loss: Option<f64>,
    pub policy_loss: Option<f64>,
    pub entropy_loss: Option<f64>,
    pub td_errors: Option<Vec<f64>>,
    pub gradient_norm: Option<f64>,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct AgentMetrics {
    pub total_episodes: u32,
    pub total_steps: u64,
    pub average_reward: f64,
    pub average_episode_length: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub exploration_rate: f64,
    pub learning_rate: f64,
}

/// Reward function trait
pub trait RewardFunction: Send + Sync {
    /// Calculate reward for transition
    fn calculate(
        &self,
        state: &State,
        action: &Action,
        next_state: &State,
        info: &TransitionInfo,
    ) -> f64;
}

/// Transition information for reward calculation
#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TransitionInfo {
    pub pnl: f64,
    pub commission: f64,
    pub slippage: f64,
    pub market_impact: f64,
    pub risk_penalty: f64,
}

/// Standard reward functions
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct SharpeReward {
    window_size: usize,
    returns: Vec<f64>,
    risk_free_rate: f64,
}

impl RewardFunction for SharpeReward {
    fn calculate(
        &self,
        _state: &State,
        _action: &Action,
        next_state: &State,
        info: &TransitionInfo,
    ) -> f64 {
        let return_val = info.pnl - info.commission - info.slippage;
        
        // Update returns window
        let returns = &self.returns;
        if returns.len() < 2 {
            return return_val;
        }
        
        // Calculate Sharpe ratio
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return return_val;
        }
        
        let sharpe = (mean - self.risk_free_rate) / std_dev * (252.0_f64).sqrt();
        
        // Reward is weighted combination
        0.7 * return_val + 0.3 * sharpe
    }
}

/// Risk-adjusted reward
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RiskAdjustedReward {
    risk_aversion: f64,
    max_drawdown_penalty: f64,
}

impl RewardFunction for RiskAdjustedReward {
    fn calculate(
        &self,
        state: &State,
        _action: &Action,
        next_state: &State,
        info: &TransitionInfo,
    ) -> f64 {
        let return_val = info.pnl - info.commission - info.slippage;
        
        // Calculate drawdown
        let drawdown = if next_state.portfolio_state.total_value < state.portfolio_state.total_value {
            (state.portfolio_state.total_value - next_state.portfolio_state.total_value) 
                / state.portfolio_state.total_value
        } else {
            0.0
        };
        
        // Risk-adjusted reward
        return_val 
            - self.risk_aversion * info.risk_penalty
            - self.max_drawdown_penalty * drawdown
    }
}

/// Multi-objective reward combining multiple metrics
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct MultiObjectiveReward {
    weights: HashMap<String, f64>,
}

impl MultiObjectiveReward {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("pnl".to_string(), 0.4);
        weights.insert("sharpe".to_string(), 0.2);
        weights.insert("sortino".to_string(), 0.1);
        weights.insert("calmar".to_string(), 0.1);
        weights.insert("risk".to_string(), -0.1);
        weights.insert("cost".to_string(), -0.1);
        
        Self { weights }
    }
}

impl RewardFunction for MultiObjectiveReward {
    fn calculate(
        &self,
        _state: &State,
        _action: &Action,
        _next_state: &State,
        info: &TransitionInfo,
    ) -> f64 {
        let pnl = info.pnl;
        let costs = info.commission + info.slippage + info.market_impact;
        
        // Calculate components
        let pnl_component = self.weights.get("pnl").unwrap_or(&0.0) * pnl;
        let cost_component = self.weights.get("cost").unwrap_or(&0.0) * costs;
        let risk_component = self.weights.get("risk").unwrap_or(&0.0) * info.risk_penalty;
        
        pnl_component + cost_component + risk_component
    }
}

/// RL Error types
#[derive(Debug, thiserror::Error)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum RLError {
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Environment error: {0}")]
    EnvironmentError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

use std::collections::HashMap;

// Blake: "RL will enable our system to learn optimal strategies from experience!"