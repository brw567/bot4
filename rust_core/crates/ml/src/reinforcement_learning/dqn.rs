//! # DEEP Q-NETWORK (DQN) - Discrete Action RL
//! Blake: "DQN for high-frequency trading decisions"
//!
//! Based on:
//! - "Human-level control through deep reinforcement learning" - Mnih et al. (2015)
//! - "Rainbow: Combining Improvements in Deep RL" - Hessel et al. (2018)

use super::*;
use ndarray::{Array1, Array2, Axis};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use rand::Rng;
use std::collections::VecDeque;

/// Deep Q-Network Agent
pub struct DQNAgent {
    /// Q-network (online)
    q_network: QNetwork,
    
    /// Target Q-network
    target_network: QNetwork,
    
    /// Optimizer
    optimizer: Optimizer,
    
    /// Configuration
    config: RLConfig,
    
    /// Experience replay buffer
    replay_buffer: PrioritizedReplayBuffer,
    
    /// Current exploration rate
    exploration_rate: f64,
    
    /// Training step counter
    training_steps: u64,
    
    /// Metrics
    metrics: AgentMetrics,
    
    /// Device (CPU/GPU)
    device: Device,
}

/// Q-Network architecture
struct QNetwork {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    fc3: candle_nn::Linear,
    advantage: candle_nn::Linear,
    value: candle_nn::Linear,
    var_map: VarMap,
}

impl QNetwork {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        vs: VarBuilder,
    ) -> candle_core::Result<Self> {
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
        
        // Dueling DQN architecture
        let fc1 = candle_nn::linear(input_dim, hidden_dim, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, hidden_dim, vs.pp("fc2"))?;
        let fc3 = candle_nn::linear(hidden_dim, hidden_dim, vs.pp("fc3"))?;
        
        // Advantage stream
        let advantage = candle_nn::linear(hidden_dim, output_dim, vs.pp("advantage"))?;
        
        // Value stream
        let value = candle_nn::linear(hidden_dim, 1, vs.pp("value"))?;
        
        Ok(Self {
            fc1,
            fc2,
            fc3,
            advantage,
            value,
            var_map,
        })
    }
    
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Shared layers
        let h1 = self.fc1.forward(x)?.relu()?;
        let h2 = self.fc2.forward(&h1)?.relu()?;
        let h3 = self.fc3.forward(&h2)?.relu()?;
        
        // Dueling streams
        let advantage = self.advantage.forward(&h3)?;
        let value = self.value.forward(&h3)?;
        
        // Q = V(s) + A(s,a) - mean(A(s,a))
        let advantage_mean = advantage.mean_keepdim(1)?;
        let q_values = value.broadcast_add(&advantage.broadcast_sub(&advantage_mean)?)?;
        
        Ok(q_values)
    }
    
    fn copy_weights_from(&mut self, other: &QNetwork) {
        self.var_map.data().lock().unwrap().clone_from(&other.var_map.data().lock().unwrap());
    }
}

/// Prioritized Experience Replay Buffer
pub struct PrioritizedReplayBuffer {
    buffer: VecDeque<Experience>,
    priorities: VecDeque<f64>,
    capacity: usize,
    alpha: f64,  // Priority exponent
    beta: f64,   // Importance sampling exponent
    epsilon: f64, // Small constant for numerical stability
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            alpha: 0.6,
            beta: 0.4,
            epsilon: 1e-6,
        }
    }
    
    pub fn push(&mut self, experience: Experience, td_error: f64) {
        let priority = (td_error.abs() + self.epsilon).powf(self.alpha);
        
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }
        
        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
    }
    
    pub fn sample(&self, batch_size: usize) -> Vec<(Experience, f64, usize)> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        
        let total_priority: f64 = self.priorities.iter().sum();
        let mut rng = rand::thread_rng();
        let mut samples = Vec::new();
        
        for _ in 0..batch_size.min(self.buffer.len()) {
            let random_val = rng.gen::<f64>() * total_priority;
            let mut cumsum = 0.0;
            
            for (idx, priority) in self.priorities.iter().enumerate() {
                cumsum += priority;
                if cumsum > random_val {
                    // Calculate importance sampling weight
                    let prob = priority / total_priority;
                    let weight = (1.0 / (self.buffer.len() as f64 * prob))
                        .powf(self.beta)
                        .min(1.0);
                    
                    samples.push((self.buffer[idx].clone(), weight, idx));
                    break;
                }
            }
        }
        
        samples
    }
    
    pub fn update_priorities(&mut self, indices: Vec<usize>, td_errors: Vec<f64>) {
        for (idx, td_error) in indices.iter().zip(td_errors.iter()) {
            if *idx < self.priorities.len() {
                self.priorities[*idx] = (td_error.abs() + self.epsilon).powf(self.alpha);
            }
        }
    }
    
    pub fn anneal_beta(&mut self, step: u64, total_steps: u64) {
        // Linearly anneal beta from 0.4 to 1.0
        self.beta = 0.4 + (1.0 - 0.4) * (step as f64 / total_steps as f64);
    }
}

impl DQNAgent {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        config: RLConfig,
    ) -> Result<Self, RLError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Create networks
        let q_network = QNetwork::new(input_dim, 512, output_dim, vs.clone())
            .map_err(|e| RLError::ModelError(format!("Failed to create Q-network: {}", e)))?;
        
        let target_network = QNetwork::new(input_dim, 512, output_dim, vs)
            .map_err(|e| RLError::ModelError(format!("Failed to create target network: {}", e)))?;
        
        // Create optimizer
        let optimizer = candle_nn::AdamW::new(
            var_map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: config.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?;
        
        Ok(Self {
            q_network,
            target_network,
            optimizer,
            config,
            replay_buffer: PrioritizedReplayBuffer::new(config.replay_buffer_size),
            exploration_rate: config.exploration_rate,
            training_steps: 0,
            metrics: AgentMetrics::default(),
            device,
        })
    }
    
    /// Epsilon-greedy action selection
    pub fn select_action(&self, state: &State, training: bool) -> Action {
        let mut rng = rand::thread_rng();
        
        // Exploration
        if training && rng.gen::<f64>() < self.exploration_rate {
            // Random action
            let action_idx = rng.gen_range(0..7);
            self.idx_to_action(action_idx)
        } else {
            // Exploitation - choose best action
            let state_tensor = self.state_to_tensor(state).unwrap();
            let q_values = self.q_network.forward(&state_tensor).unwrap();
            let q_values_vec = q_values.to_vec1::<f32>().unwrap();
            
            let best_action_idx = q_values_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            
            self.idx_to_action(best_action_idx)
        }
    }
    
    /// Train on batch of experiences
    pub fn train_step(&mut self, batch: Vec<(Experience, f64, usize)>) -> TrainingResult {
        if batch.is_empty() {
            return TrainingResult {
                loss: 0.0,
                value_loss: None,
                policy_loss: None,
                entropy_loss: None,
                td_errors: None,
                gradient_norm: None,
            };
        }
        
        let batch_size = batch.len();
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut dones = Vec::new();
        let mut weights = Vec::new();
        let mut indices = Vec::new();
        
        for (exp, weight, idx) in batch {
            states.push(self.state_to_tensor(&exp.state).unwrap());
            actions.push(self.action_to_idx(&exp.action));
            rewards.push(exp.reward);
            next_states.push(self.state_to_tensor(&exp.next_state).unwrap());
            dones.push(exp.done as i64 as f32);
            weights.push(weight);
            indices.push(idx);
        }
        
        // Stack tensors
        let states = Tensor::stack(&states, 0).unwrap();
        let actions = Tensor::new(actions, &self.device).unwrap();
        let rewards = Tensor::new(rewards, &self.device).unwrap();
        let next_states = Tensor::stack(&next_states, 0).unwrap();
        let dones = Tensor::new(dones, &self.device).unwrap();
        let weights = Tensor::new(weights, &self.device).unwrap();
        
        // Current Q values
        let current_q = self.q_network.forward(&states).unwrap();
        let current_q_selected = current_q.gather(&actions.unsqueeze(1)?, 1)?.squeeze(1)?;
        
        // Double DQN: use online network to select action, target network to evaluate
        let next_q_online = self.q_network.forward(&next_states).unwrap();
        let next_actions = next_q_online.argmax_keepdim(1)?;
        
        let next_q_target = self.target_network.forward(&next_states).unwrap();
        let next_q_selected = next_q_target.gather(&next_actions, 1)?.squeeze(1)?;
        
        // TD target
        let gamma = Tensor::new(&[self.config.discount_factor as f32], &self.device)?;
        let target = rewards.broadcast_add(
            &gamma.broadcast_mul(&next_q_selected.broadcast_mul(&(1.0 - dones))?)?
        )?;
        
        // TD error for priority update
        let td_errors = (&current_q_selected - &target)?;
        let td_errors_vec = td_errors.to_vec1::<f32>().unwrap()
            .into_iter()
            .map(|e| e as f64)
            .collect::<Vec<_>>();
        
        // Weighted Huber loss
        let loss = td_errors.abs()?.smooth_l1_loss(&target, 1.0)?;
        let weighted_loss = loss.broadcast_mul(&weights)?.mean_all()?;
        
        // Backpropagation
        self.optimizer.backward_step(&weighted_loss)?;
        
        // Update priorities
        self.replay_buffer.update_priorities(indices, td_errors_vec.clone());
        
        // Update exploration rate
        self.exploration_rate = (self.exploration_rate * self.config.exploration_decay)
            .max(self.config.min_exploration_rate);
        
        // Update target network
        self.training_steps += 1;
        if self.training_steps % self.config.target_update_frequency as u64 == 0 {
            self.update_target_network();
        }
        
        TrainingResult {
            loss: weighted_loss.to_scalar::<f32>().unwrap() as f64,
            value_loss: None,
            policy_loss: None,
            entropy_loss: None,
            td_errors: Some(td_errors_vec),
            gradient_norm: None,
        }
    }
    
    fn update_target_network(&mut self) {
        self.target_network.copy_weights_from(&self.q_network);
    }
    
    fn state_to_tensor(&self, state: &State) -> candle_core::Result<Tensor> {
        let features = [
            state.market_features.as_slice().unwrap(),
            state.indicators.as_slice().unwrap(),
            state.orderbook_features.as_slice().unwrap(),
        ].concat();
        
        Tensor::new(features.as_slice(), &self.device)
    }
    
    fn action_to_idx(&self, action: &Action) -> u32 {
        match action {
            Action::Hold => 0,
            Action::Buy { .. } => 1,
            Action::Sell { .. } => 2,
            Action::LimitBuy { .. } => 3,
            Action::LimitSell { .. } => 4,
            Action::Close { .. } => 5,
            Action::CloseAll => 6,
            _ => 0,
        }
    }
    
    fn idx_to_action(&self, idx: usize) -> Action {
        match idx {
            0 => Action::Hold,
            1 => Action::Buy { size: 0.01 },  // Default size
            2 => Action::Sell { size: 0.01 },
            3 => Action::LimitBuy { price: 0.0, size: 0.01 },
            4 => Action::LimitSell { price: 0.0, size: 0.01 },
            5 => Action::Close { symbol: "BTC".to_string() },
            6 => Action::CloseAll,
            _ => Action::Hold,
        }
    }
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            total_episodes: 0,
            total_steps: 0,
            average_reward: 0.0,
            average_episode_length: 0.0,
            win_rate: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            exploration_rate: 1.0,
            learning_rate: 0.0001,
        }
    }
}

// Blake: "DQN with prioritized replay and double Q-learning for stable training!"