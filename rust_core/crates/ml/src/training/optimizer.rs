// Optimizer Module - Advanced Optimization Algorithms
// Team Lead: Morgan (Optimization Algorithms)
// Contributors: Jordan (Performance), Sam (Architecture)
// Date: January 18, 2025
// NO SIMPLIFICATIONS - FULL IMPLEMENTATION

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// OPTIMIZER TRAIT - Sam's Clean Architecture
// ============================================================================

pub trait Optimizer: Send + Sync {
    /// Update parameters based on gradients
    fn step(
        &mut self,
        params: &mut Array1<f64>,
        gradients: &Array1<f64>,
        iteration: usize,
    ) -> Result<()>;
    
    /// Reset optimizer state
    fn reset(&mut self);
    
    /// Get current learning rate
    fn get_learning_rate(&self, iteration: usize) -> f64;
}

// ============================================================================
// ADAM OPTIMIZER - Morgan's Implementation
// ============================================================================

/// Adam optimizer with momentum and adaptive learning
pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    m: Option<Array1<f64>>, // First moment
    v: Option<Array1<f64>>, // Second moment
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: None,
            v: None,
        }
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamOptimizer {
    fn step(
        &mut self,
        params: &mut Array1<f64>,
        gradients: &Array1<f64>,
        iteration: usize,
    ) -> Result<()> {
        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(Array1::zeros(params.len()));
            self.v = Some(Array1::zeros(params.len()));
        }
        
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        
        // Update biased moments
        *m = &*m * self.beta1 + gradients * (1.0 - self.beta1);
        *v = &*v * self.beta2 + gradients.mapv(|g| g * g) * (1.0 - self.beta2);
        
        // Compute bias-corrected moments
        let t = iteration as f64 + 1.0;
        let m_hat = m / (1.0 - self.beta1.powf(t));
        let v_hat = v / (1.0 - self.beta2.powf(t));
        
        // Apply weight decay if specified (AdamW variant)
        if self.weight_decay > 0.0 {
            *params = &*params * (1.0 - self.learning_rate * self.weight_decay);
        }
        
        // Update parameters
        *params = &*params - self.learning_rate * m_hat / (v_hat.mapv(f64::sqrt) + self.epsilon);
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.m = None;
        self.v = None;
    }
    
    fn get_learning_rate(&self, _iteration: usize) -> f64 {
        self.learning_rate
    }
}

// ============================================================================
// SGD WITH MOMENTUM - Jordan's Fast Implementation
// ============================================================================

pub struct SGDOptimizer {
    learning_rate: f64,
    momentum: f64,
    nesterov: bool,
    velocity: Option<Array1<f64>>,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterov: false,
            velocity: None,
        }
    }
    
    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }
}

impl Optimizer for SGDOptimizer {
    fn step(
        &mut self,
        params: &mut Array1<f64>,
        gradients: &Array1<f64>,
        iteration: usize,
    ) -> Result<()> {
        // Initialize velocity if needed
        if self.velocity.is_none() {
            self.velocity = Some(Array1::zeros(params.len()));
        }
        
        let velocity = self.velocity.as_mut().unwrap();
        
        if self.nesterov {
            // Nesterov accelerated gradient
            let prev_velocity = velocity.clone();
            *velocity = &*velocity * self.momentum - gradients * self.learning_rate;
            *params = &*params + &*velocity * (1.0 + self.momentum) - prev_velocity * self.momentum;
        } else {
            // Standard momentum
            *velocity = &*velocity * self.momentum - gradients * self.learning_rate;
            *params = &*params + velocity;
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.velocity = None;
    }
    
    fn get_learning_rate(&self, _iteration: usize) -> f64 {
        self.learning_rate
    }
}

// ============================================================================
// RMSPROP OPTIMIZER - Morgan's Adaptive Learning
// ============================================================================

pub struct RMSpropOptimizer {
    learning_rate: f64,
    decay_rate: f64,
    epsilon: f64,
    momentum: f64,
    cache: Option<Array1<f64>>,
    velocity: Option<Array1<f64>>,
}

impl RMSpropOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            decay_rate: 0.9,
            epsilon: 1e-8,
            momentum: 0.0,
            cache: None,
            velocity: None,
        }
    }
    
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for RMSpropOptimizer {
    fn step(
        &mut self,
        params: &mut Array1<f64>,
        gradients: &Array1<f64>,
        iteration: usize,
    ) -> Result<()> {
        // Initialize cache
        if self.cache.is_none() {
            self.cache = Some(Array1::zeros(params.len()));
        }
        
        let cache = self.cache.as_mut().unwrap();
        
        // Update cache with exponential moving average
        *cache = &*cache * self.decay_rate + gradients.mapv(|g| g * g) * (1.0 - self.decay_rate);
        
        // Compute update
        let update = gradients / (cache.mapv(f64::sqrt) + self.epsilon) * self.learning_rate;
        
        if self.momentum > 0.0 {
            // Apply momentum if specified
            if self.velocity.is_none() {
                self.velocity = Some(Array1::zeros(params.len()));
            }
            let velocity = self.velocity.as_mut().unwrap();
            *velocity = &*velocity * self.momentum + update * (1.0 - self.momentum);
            *params = &*params - velocity;
        } else {
            *params = &*params - update;
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.cache = None;
        self.velocity = None;
    }
    
    fn get_learning_rate(&self, _iteration: usize) -> f64 {
        self.learning_rate
    }
}

// ============================================================================
// LEARNING RATE SCHEDULERS - Morgan's Scheduling
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduler {
    Constant,
    StepDecay { step_size: usize, gamma: f64 },
    ExponentialDecay { gamma: f64 },
    CosineAnnealing { t_max: usize },
    WarmupCosine { warmup_steps: usize, t_max: usize },
}

impl LRScheduler {
    /// Calculate learning rate multiplier for given iteration
    pub fn get_multiplier(&self, iteration: usize) -> f64 {
        match self {
            LRScheduler::Constant => 1.0,
            
            LRScheduler::StepDecay { step_size, gamma } => {
                gamma.powi((iteration / step_size) as i32)
            }
            
            LRScheduler::ExponentialDecay { gamma } => {
                gamma.powi(iteration as i32)
            }
            
            LRScheduler::CosineAnnealing { t_max } => {
                let progress = (iteration % t_max) as f64 / *t_max as f64;
                0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            
            LRScheduler::WarmupCosine { warmup_steps, t_max } => {
                if iteration < *warmup_steps {
                    // Linear warmup
                    iteration as f64 / *warmup_steps as f64
                } else {
                    // Cosine annealing after warmup
                    let progress = ((iteration - warmup_steps) % (t_max - warmup_steps)) as f64
                        / (*t_max - warmup_steps) as f64;
                    0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZER FACTORY - Sam's Pattern
// ============================================================================

pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create optimizer from configuration
    pub fn create(
        optimizer_type: &str,
        learning_rate: f64,
        params: &HashMap<String, f64>,
    ) -> Result<Box<dyn Optimizer>> {
        match optimizer_type {
            "adam" => {
                let mut opt = AdamOptimizer::new(learning_rate);
                if let Some(&wd) = params.get("weight_decay") {
                    opt = opt.with_weight_decay(wd);
                }
                Ok(Box::new(opt))
            }
            
            "sgd" => {
                let momentum = params.get("momentum").copied().unwrap_or(0.9);
                let mut opt = SGDOptimizer::new(learning_rate, momentum);
                if params.get("nesterov").copied().unwrap_or(0.0) > 0.5 {
                    opt = opt.with_nesterov();
                }
                Ok(Box::new(opt))
            }
            
            "rmsprop" => {
                let mut opt = RMSpropOptimizer::new(learning_rate);
                if let Some(&momentum) = params.get("momentum") {
                    opt = opt.with_momentum(momentum);
                }
                Ok(Box::new(opt))
            }
            
            _ => Err(anyhow::anyhow!("Unknown optimizer type: {}", optimizer_type))
        }
    }
}

// ============================================================================
// TESTS - Riley's Validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adam_optimizer() {
        let mut opt = AdamOptimizer::new(0.001);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        
        opt.step(&mut params, &gradients, 0).unwrap();
        
        // Parameters should decrease (negative gradients)
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }
    
    #[test]
    fn test_sgd_momentum() {
        let mut opt = SGDOptimizer::new(0.01, 0.9);
        let mut params = Array1::from_vec(vec![1.0, 1.0]);
        let gradients = Array1::from_vec(vec![1.0, 1.0]);
        
        // First step
        opt.step(&mut params, &gradients, 0).unwrap();
        let first_params = params.clone();
        
        // Second step with same gradients - momentum should increase step
        opt.step(&mut params, &gradients, 1).unwrap();
        let step_size = (&first_params - &params).mapv(f64::abs);
        
        assert!(step_size[0] > 0.01); // Larger than base learning rate due to momentum
    }
    
    #[test]
    fn test_learning_rate_schedulers() {
        // Test step decay
        let scheduler = LRScheduler::StepDecay { step_size: 10, gamma: 0.5 };
        assert_eq!(scheduler.get_multiplier(0), 1.0);
        assert_eq!(scheduler.get_multiplier(10), 0.5);
        assert_eq!(scheduler.get_multiplier(20), 0.25);
        
        // Test cosine annealing
        let scheduler = LRScheduler::CosineAnnealing { t_max: 100 };
        assert!((scheduler.get_multiplier(0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_multiplier(50) - 0.0).abs() < 1e-6);
        assert!((scheduler.get_multiplier(100) - 1.0).abs() < 1e-6);
    }
}

// ============================================================================
// TEAM SIGN-OFF
// ============================================================================
// Morgan: "Advanced optimizers with adaptive learning implemented"
// Jordan: "Performance optimized with efficient array operations"
// Sam: "Clean factory pattern and trait design"