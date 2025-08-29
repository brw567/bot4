// SAGA Pattern Implementation - Orchestration & Choreography
// Task 2.3: FULL SAGA implementation with compensation logic
// Team: Casey (Exchange) + Morgan (ML) + Quinn (Risk)
// References:
// - Garcia-Molina & Salem (1987): "Sagas"
// - Richardson (2018): "Microservices Patterns"
// - Temporal.io SAGA best practices

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use uuid::Uuid;
use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// SAGA execution state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum SagaState {
    /// Saga not started
    Pending,
    /// Saga is executing forward
    Running,
    /// Saga completed successfully
    Completed,
    /// Saga failed and compensating
    Compensating,
    /// Saga fully compensated
    Compensated,
    /// Saga in inconsistent state
    Aborted { reason: String },
}

/// Step execution result
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum StepResult {
    /// Step succeeded, continue to next
    Success(serde_json::Value),
    /// Step failed, trigger compensation
    Failure(String),
    /// Step requires retry
    Retry,
    /// Step skipped (conditional)
    Skipped,
}

/// SAGA step definition
#[async_trait]
pub trait SagaStep: Send + Sync {
    /// Step name for logging
    fn name(&self) -> &str;
    
    /// Execute forward action
    async fn execute(&self, context: &SagaContext) -> Result<StepResult>;
    
    /// Execute compensating action
    async fn compensate(&self, context: &SagaContext) -> Result<()>;
    
    /// Check if step is idempotent
    fn is_idempotent(&self) -> bool {
        false
    }
    
    /// Maximum retry attempts
    fn max_retries(&self) -> u32 {
        3
    }
    
    /// Retry delay strategy
    fn retry_delay(&self, attempt: u32) -> Duration {
        // Exponential backoff: 100ms, 200ms, 400ms, ...
        Duration::from_millis(100 * (1 << attempt))
    }
}

/// SAGA execution context
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct SagaContext {
    /// Unique saga ID
    pub saga_id: Uuid,
    /// Transaction ID this saga belongs to
    pub transaction_id: Uuid,
    /// Shared state between steps
    pub state: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// Execution history
    pub history: Arc<RwLock<Vec<ExecutionRecord>>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct ExecutionRecord {
    pub step_name: String,
    pub started_at: u64,
    pub completed_at: Option<u64>,
    pub result: String,
    pub error: Option<String>,
    pub retries: u32,
}

impl SagaContext {
    pub fn new(transaction_id: Uuid) -> Self {
        Self {
            saga_id: Uuid::new_v4(),
            transaction_id,
            state: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            metadata: HashMap::new(),
        }
    }
    
    /// Store value in context
    pub fn set<T: Serialize>(&self, key: &str, value: T) -> Result<()> {
        let json_value = serde_json::to_value(value)?;
        self.state.write().insert(key.to_string(), json_value);
        Ok(())
    }
    
    /// Retrieve value from context
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let state = self.state.read();
        match state.get(key) {
            Some(value) => {
                let deserialized = serde_json::from_value(value.clone())?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }
    
    /// Record step execution
    fn record_execution(&self, step_name: &str, result: &str, error: Option<String>, retries: u32) {
        let record = ExecutionRecord {
            step_name: step_name.to_string(),
            started_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64,
            completed_at: Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64),
            result: result.to_string(),
            error,
            retries,
        };
        
        self.history.write().push(record);
    }
}

/// SAGA definition
/// TODO: Add docs
pub struct Saga {
    /// Unique saga ID
    pub id: Uuid,
    /// Saga name
    pub name: String,
    /// Steps in order
    pub steps: Vec<Box<dyn SagaStep + Send + Sync>>,
    /// Current step index
    pub current_step: usize,
    /// Saga state
    pub state: SagaState,
    /// Execution context
    pub context: SagaContext,
}

impl Saga {
    pub fn new(name: String, transaction_id: Uuid) -> Self {
        let id = Uuid::new_v4();
        Self {
            id,
            name,
            steps: Vec::new(),
            current_step: 0,
            state: SagaState::Pending,
            context: SagaContext::new(transaction_id),
        }
    }
    
    /// Add step to saga
    pub fn add_step(mut self, step: Box<dyn SagaStep + Send + Sync>) -> Self {
        self.steps.push(step);
        self
    }
    
    /// Execute saga forward
    pub async fn execute(&mut self) -> Result<()> {
        self.state = SagaState::Running;
        
        while self.current_step < self.steps.len() {
            let mut retries = 0;
            
            // Execute step with proper borrow scope
            let step_result = {
                let step = &self.steps[self.current_step];
                let step_name = step.name().to_string();
                let max_retries = step.max_retries();
                
                let mut result = None;
                
                loop {
                    match step.execute(&self.context).await {
                        Ok(StepResult::Success(data)) => {
                            // Record success
                            self.context.record_execution(
                                &step_name,
                                "Success",
                                None,
                                retries
                            );
                            
                            // Store result in context
                            self.context.set(&format!("{}_result", step_name), &data)?;
                            
                            result = Some(Ok(()));
                            break;
                        }
                        Ok(StepResult::Failure(error)) => {
                            // Record failure
                            self.context.record_execution(
                                &step_name,
                                "Failed",
                                Some(error.clone()),
                                retries
                            );
                            
                            result = Some(Err(anyhow::anyhow!("Saga failed: {}", error)));
                            break;
                        }
                        Ok(StepResult::Retry) => {
                            retries += 1;
                            if retries > max_retries {
                                result = Some(Err(anyhow::anyhow!("Max retries exceeded for step {}", step_name)));
                                break;
                            }
                            
                            // Wait before retry
                            tokio::time::sleep(step.retry_delay(retries)).await;
                        }
                        Ok(StepResult::Skipped) => {
                            // Record skip
                            self.context.record_execution(
                                &step_name,
                                "Skipped",
                                None,
                                retries
                            );
                            
                            result = Some(Ok(()));
                            break;
                        }
                        Err(e) => {
                            // Unexpected error
                            self.context.record_execution(
                                &step_name,
                                "Error",
                                Some(e.to_string()),
                                retries
                            );
                            
                            result = Some(Err(e));
                            break;
                        }
                    }
                }
                
                result.unwrap()
            };
            
            // Handle result outside of borrow scope
            match step_result {
                Ok(()) => {
                    // Move to next step
                    self.current_step += 1;
                }
                Err(e) => {
                    // Trigger compensation
                    self.state = SagaState::Compensating;
                    self.compensate().await?;
                    return Err(e);
                }
            }
        }
        
        self.state = SagaState::Completed;
        Ok(())
    }
    
    /// Execute compensation (reverse order)
    pub async fn compensate(&mut self) -> Result<()> {
        // Compensate in reverse order from current step
        for i in (0..self.current_step).rev() {
            let step = &self.steps[i];
            let step_name = step.name().to_string();
            
            match step.compensate(&self.context).await {
                Ok(()) => {
                    self.context.record_execution(
                        &format!("{}_compensate", step_name),
                        "Success",
                        None,
                        0
                    );
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    self.context.record_execution(
                        &format!("{}_compensate", step_name),
                        "Failed",
                        Some(error_msg.clone()),
                        0
                    );
                    
                    // Compensation failed - inconsistent state
                    self.state = SagaState::Aborted {
                        reason: format!("Compensation failed at {}: {}", step_name, error_msg),
                    };
                    return Err(e);
                }
            }
        }
        
        self.state = SagaState::Compensated;
        Ok(())
    }
}

/// SAGA Orchestrator - manages multiple sagas
/// TODO: Add docs
pub struct SagaOrchestrator {
    /// Active sagas
    sagas: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Saga>>>>>,
    /// Event channel for choreography
    event_tx: mpsc::UnboundedSender<SagaEvent>,
    /// Metrics
    metrics: Arc<SagaMetrics>,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum SagaEvent {
    Started(Uuid),
    StepCompleted(Uuid, String),
    StepFailed(Uuid, String, String),
    Completed(Uuid),
    Compensated(Uuid),
    Aborted(Uuid, String),
}

#[derive(Debug, Default)]
struct SagaMetrics {
    total_sagas: AtomicU64,
    successful_sagas: AtomicU64,
    failed_sagas: AtomicU64,
    compensated_sagas: AtomicU64,
    average_duration_ms: AtomicU64,
}

use std::sync::atomic::{AtomicU64, Ordering};

impl SagaOrchestrator {
    pub fn new() -> Self {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let metrics = Arc::new(SagaMetrics::default());
        
        // Spawn event processor for choreography
        let event_metrics = metrics.clone();
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                match event {
                    SagaEvent::Completed(_) => {
                        event_metrics.successful_sagas.fetch_add(1, Ordering::Relaxed);
                    }
                    SagaEvent::Compensated(_) => {
                        event_metrics.compensated_sagas.fetch_add(1, Ordering::Relaxed);
                    }
                    SagaEvent::Aborted(_, _) => {
                        event_metrics.failed_sagas.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        });
        
        Self {
            sagas: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            metrics,
        }
    }
    
    /// Start a new saga
    pub async fn start_saga(&self, saga: Saga) -> Result<Uuid> {
        let saga_id = saga.id;
        let saga_arc = Arc::new(RwLock::new(saga));
        
        // Store saga
        self.sagas.write().insert(saga_id, saga_arc.clone());
        
        // Send start event
        let _ = self.event_tx.send(SagaEvent::Started(saga_id));
        
        // Increment metrics
        self.metrics.total_sagas.fetch_add(1, Ordering::Relaxed);
        
        // For now, execute synchronously to avoid Send issues
        // TODO: Refactor to make Saga fully Send + Sync
        let event_tx = self.event_tx.clone();
        let start = std::time::Instant::now();
        
        {
            let mut saga_mut = saga_arc.write();
            match saga_mut.execute().await {
                Ok(()) => {
                    let _ = event_tx.send(SagaEvent::Completed(saga_id));
                }
                Err(e) => {
                    match &saga_mut.state {
                        SagaState::Compensated => {
                            let _ = event_tx.send(SagaEvent::Compensated(saga_id));
                        }
                        SagaState::Aborted { reason } => {
                            let _ = event_tx.send(SagaEvent::Aborted(saga_id, reason.clone()));
                        }
                        _ => {
                            let _ = event_tx.send(SagaEvent::Aborted(saga_id, e.to_string()));
                        }
                    }
                }
            }
        }
        
        // Update duration metric
        let duration = start.elapsed().as_millis() as u64;
        let current = self.metrics.average_duration_ms.load(Ordering::Relaxed);
        self.metrics.average_duration_ms.store((current + duration) / 2, Ordering::Relaxed);
        
        Ok(saga_id)
    }
    
    /// Get saga status
    pub fn get_saga_status(&self, saga_id: Uuid) -> Option<SagaState> {
        let sagas = self.sagas.read();
        sagas.get(&saga_id).map(|saga| saga.read().state.clone())
    }
    
    /// Cancel saga (trigger compensation)
    pub async fn cancel_saga(&self, saga_id: Uuid) -> Result<()> {
        let saga_arc = {
            let sagas = self.sagas.read();
            sagas.get(&saga_id).cloned()
        };
        
        if let Some(saga_arc) = saga_arc {
            let mut saga = saga_arc.write();
            if !matches!(saga.state, SagaState::Completed | SagaState::Compensated) {
                saga.state = SagaState::Compensating;
                saga.compensate().await?;
            }
        }
        
        Ok(())
    }
}

// Example SAGA steps for trading operations

/// Order placement step
/// TODO: Add docs
pub struct PlaceOrderStep {
    symbol: String,
    quantity: rust_decimal::Decimal,
    price: Option<rust_decimal::Decimal>,
}

#[async_trait]
impl SagaStep for PlaceOrderStep {
    fn name(&self) -> &str {
        "PlaceOrder"
    }
    
    async fn execute(&self, context: &SagaContext) -> Result<StepResult> {
        // Simulate order placement
        let order_id = Uuid::new_v4();
        
        // Store order ID in context
        context.set("order_id", order_id)?;
        
        // Return success with order details
        Ok(StepResult::Success(serde_json::json!({
            "order_id": order_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "status": "placed"
        })))
    }
    
    async fn compensate(&self, context: &SagaContext) -> Result<()> {
        // Cancel the order
        if let Some(order_id) = context.get::<Uuid>("order_id")? {
            // Simulate order cancellation
            println!("Cancelling order {}", order_id);
        }
        Ok(())
    }
    
    fn is_idempotent(&self) -> bool {
        true // Order placement with same ID is idempotent
    }
}

/// Risk check step
/// TODO: Add docs
pub struct RiskCheckStep {
    max_position_size: f64,
    max_leverage: f64,
}

#[async_trait]
impl SagaStep for RiskCheckStep {
    fn name(&self) -> &str {
        "RiskCheck"
    }
    
    async fn execute(&self, _context: &SagaContext) -> Result<StepResult> {
        // Simulate risk check
        let position_size = 0.01; // Example
        let leverage = 1.5; // Example
        
        if position_size > self.max_position_size || leverage > self.max_leverage {
            return Ok(StepResult::Failure("Risk limits exceeded".to_string()));
        }
        
        Ok(StepResult::Success(serde_json::json!({
            "position_size": position_size,
            "leverage": leverage,
            "approved": true
        })))
    }
    
    async fn compensate(&self, _context: &SagaContext) -> Result<()> {
        // No compensation needed for risk check
        Ok(())
    }
}

/// Balance update step
/// TODO: Add docs
pub struct UpdateBalanceStep {
    account_id: Uuid,
    amount: rust_decimal::Decimal,
}

#[async_trait]
impl SagaStep for UpdateBalanceStep {
    fn name(&self) -> &str {
        "UpdateBalance"
    }
    
    async fn execute(&self, context: &SagaContext) -> Result<StepResult> {
        // Simulate balance update
        let previous_balance = rust_decimal::Decimal::from(10000);
        let new_balance = previous_balance - self.amount;
        
        // Store previous balance for compensation
        context.set("previous_balance", previous_balance)?;
        
        Ok(StepResult::Success(serde_json::json!({
            "account_id": self.account_id,
            "previous_balance": previous_balance,
            "new_balance": new_balance
        })))
    }
    
    async fn compensate(&self, context: &SagaContext) -> Result<()> {
        // Restore previous balance
        if let Some(previous_balance) = context.get::<rust_decimal::Decimal>("previous_balance")? {
            println!("Restoring balance to {}", previous_balance);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;
    
    #[tokio::test]
    async fn test_saga_execution() {
        let mut saga = Saga::new("TestTradeSaga".to_string(), Uuid::new_v4());
        
        saga = saga
            .add_step(Box::new(RiskCheckStep {
                max_position_size: 0.02,
                max_leverage: 3.0,
            }))
            .add_step(Box::new(PlaceOrderStep {
                symbol: "BTC/USDT".to_string(),
                quantity: Decimal::from(1),
                price: Some(Decimal::from(50000)),
            }))
            .add_step(Box::new(UpdateBalanceStep {
                account_id: Uuid::new_v4(),
                amount: Decimal::from(50000),
            }));
        
        let result = saga.execute().await;
        assert!(result.is_ok());
        assert_eq!(saga.state, SagaState::Completed);
    }
    
    #[tokio::test]
    async fn test_saga_compensation() {
        let mut saga = Saga::new("FailingSaga".to_string(), Uuid::new_v4());
        
        saga = saga
            .add_step(Box::new(PlaceOrderStep {
                symbol: "BTC/USDT".to_string(),
                quantity: Decimal::from(1),
                price: Some(Decimal::from(50000)),
            }))
            .add_step(Box::new(RiskCheckStep {
                max_position_size: 0.0001, // Will fail
                max_leverage: 0.1,
            }));
        
        let result = saga.execute().await;
        assert!(result.is_err());
        assert_eq!(saga.state, SagaState::Compensated);
    }
}