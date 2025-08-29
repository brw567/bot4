// Transaction Rollback System - SAGA Pattern Implementation
// Task 2: FULL Implementation with NO SIMPLIFICATIONS
// Team: Sam (Architecture) + Casey (Exchange) + Quinn (Risk) + Jordan (Performance)
// References:
// - Garcia-Molina & Salem (1987): "Sagas" 
// - Richardson (2018): "Microservices Patterns"
// - Martin Fowler: "Patterns of Distributed Systems"

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use uuid::Uuid;
use rust_decimal::Decimal;
use tokio::sync::mpsc;
use std::collections::{HashMap, VecDeque};

pub mod wal;
pub mod saga;
pub mod compensator;
pub mod retry;

// tests module is in tests.rs file
#[cfg(test)]
mod tests;

pub use wal::WriteAheadLog;
pub use saga::{Saga, SagaStep, SagaState};
pub use compensator::CompensatingTransaction;
pub use retry::{RetryPolicy, RetryManager, CircuitBreaker, CircuitBreakerConfig};

/// Transaction types in our trading system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum TransactionType {
    OrderPlacement {
        order_id: Uuid,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    },
    PositionUpdate {
        position_id: Uuid,
        symbol: String,
        delta_quantity: Decimal,
        new_average_price: Decimal,
    },
    RiskCheck {
        check_id: Uuid,
        portfolio_heat: f64,
        var_limit: f64,
        leverage: f64,
    },
    BalanceUpdate {
        account_id: Uuid,
        currency: String,
        delta: Decimal,
        reason: String,
    },
    MarginRequirement {
        position_id: Uuid,
        required_margin: Decimal,
        current_margin: Decimal,
    },
    FeeDeduction {
        transaction_id: Uuid,
        amount: Decimal,
        fee_type: FeeType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum FeeType {
    Maker,
    Taker,
    Funding,
    Withdrawal,
}

/// Transaction status following SAGA pattern
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// TODO: Add docs
pub enum TransactionStatus {
    /// Transaction has been initiated but not yet started
    Pending,
    /// Transaction is currently being processed
    InProgress,
    /// Transaction completed successfully
    Committed,
    /// Transaction failed and needs compensation
    Failed { error: String },
    /// Compensating transaction is in progress
    Compensating,
    /// Transaction has been fully compensated/rolled back
    Compensated,
    /// Transaction is in an inconsistent state (requires manual intervention)
    Inconsistent { reason: String },
}

/// Represents a single transaction in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Transaction {
    /// Unique identifier for idempotency
    pub id: Uuid,
    /// Idempotency key from client (prevents duplicates)
    pub idempotency_key: Option<String>,
    /// Type of transaction
    pub transaction_type: TransactionType,
    /// Current status
    pub status: TransactionStatus,
    /// Timestamp when created (microseconds since epoch)
    pub created_at: u64,
    /// Timestamp when last updated
    pub updated_at: u64,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Parent transaction ID (for nested transactions)
    pub parent_id: Option<Uuid>,
    /// Child transaction IDs
    pub children: Vec<Uuid>,
    /// Metadata for debugging and audit
    pub metadata: HashMap<String, String>,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(transaction_type: TransactionType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        Self {
            id: Uuid::new_v4(),
            idempotency_key: None,
            transaction_type,
            status: TransactionStatus::Pending,
            created_at: now,
            updated_at: now,
            retry_count: 0,
            parent_id: None,
            children: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create with idempotency key
    pub fn with_idempotency_key(mut self, key: String) -> Self {
        self.idempotency_key = Some(key);
        self
    }
    
    /// Update transaction status
    pub fn update_status(&mut self, status: TransactionStatus) {
        self.status = status;
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
    }
    
    /// Check if transaction needs retry
    pub fn needs_retry(&self) -> bool {
        matches!(self.status, TransactionStatus::Failed { .. }) 
            && self.retry_count < 5 // Max 5 retries
    }
    
    /// Check if transaction is terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            TransactionStatus::Committed | 
            TransactionStatus::Compensated |
            TransactionStatus::Inconsistent { .. }
        )
    }
}

/// Transaction Manager - Orchestrates SAGA pattern
/// TODO: Add docs
pub struct TransactionManager {
    /// Write-Ahead Log for durability
    wal: Arc<WriteAheadLog>,
    /// Active transactions
    transactions: Arc<RwLock<HashMap<Uuid, Transaction>>>,
    /// Idempotency cache (key -> transaction_id)
    idempotency_cache: Arc<RwLock<HashMap<String, Uuid>>>,
    /// Saga orchestrator
    saga_orchestrator: Arc<saga::SagaOrchestrator>,
    /// Retry manager
    retry_manager: Arc<retry::RetryManager>,
    /// Transaction event channel
    event_tx: mpsc::UnboundedSender<TransactionEvent>,
    /// Metrics
    metrics: Arc<RwLock<TransactionMetrics>>,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum TransactionEvent {
    Started(Uuid),
    Committed(Uuid),
    Failed(Uuid, String),
    Compensating(Uuid),
    Compensated(Uuid),
}

#[derive(Debug, Default)]
/// TODO: Add docs
pub struct TransactionMetrics {
    pub total_transactions: u64,
    pub successful_transactions: u64,
    pub failed_transactions: u64,
    pub compensated_transactions: u64,
    pub retry_attempts: u64,
    pub average_latency_us: u64,
}

impl TransactionManager {
    /// Create new transaction manager
    pub async fn new(wal_path: &str) -> Result<Self> {
        let wal = Arc::new(WriteAheadLog::new(wal_path).await?);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        
        // Spawn event processor
        let metrics = Arc::new(RwLock::new(TransactionMetrics::default()));
        let metrics_clone = metrics.clone();
        
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                let mut m = metrics_clone.write();
                match event {
                    TransactionEvent::Started(_) => {
                        // Game Theory: Track all transaction attempts for audit
                        // Trading: Essential for order flow analysis
                        m.total_transactions += 1;
                    }
                    TransactionEvent::Committed(_) => {
                        m.successful_transactions += 1;
                    }
                    TransactionEvent::Failed(_, _) => {
                        m.failed_transactions += 1;
                    }
                    TransactionEvent::Compensated(_) => {
                        m.compensated_transactions += 1;
                    }
                    TransactionEvent::Compensating(_) => {
                        // Track compensation attempts separately
                    }
                }
            }
        });
        
        Ok(Self {
            wal,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            idempotency_cache: Arc::new(RwLock::new(HashMap::new())),
            saga_orchestrator: Arc::new(saga::SagaOrchestrator::new()),
            retry_manager: Arc::new(retry::RetryManager::new()),
            event_tx,
            metrics,
        })
    }
    
    /// Begin a new transaction (with idempotency check)
    pub async fn begin_transaction(
        &self,
        transaction: Transaction,
    ) -> Result<Uuid> {
        // Check idempotency
        if let Some(ref key) = transaction.idempotency_key {
            let cache = self.idempotency_cache.read();
            if let Some(&existing_id) = cache.get(key) {
                // Return existing transaction ID (idempotent)
                return Ok(existing_id);
            }
        }
        
        let id = transaction.id;
        
        // Write to WAL first (durability)
        self.wal.append(&transaction).await?;
        
        // Store in memory
        {
            let mut txns = self.transactions.write();
            txns.insert(id, transaction.clone());
        }
        
        // Update idempotency cache
        if let Some(ref key) = transaction.idempotency_key {
            let mut cache = self.idempotency_cache.write();
            cache.insert(key.clone(), id);
        }
        
        // Send event
        let _ = self.event_tx.send(TransactionEvent::Started(id));
        
        Ok(id)
    }
    
    /// Commit a transaction
    pub async fn commit_transaction(&self, id: Uuid) -> Result<()> {
        let mut transaction = {
            let txns = self.transactions.read();
            txns.get(&id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?
                .clone()
        };
        
        // Update status
        transaction.update_status(TransactionStatus::Committed);
        
        // Write to WAL
        self.wal.append(&transaction).await?;
        
        // Update in memory
        {
            let mut txns = self.transactions.write();
            txns.insert(id, transaction);
        }
        
        // Send event
        let _ = self.event_tx.send(TransactionEvent::Committed(id));
        
        Ok(())
    }
    
    /// Rollback a transaction (trigger compensation)
    pub async fn rollback_transaction(
        &self,
        id: Uuid,
        error: String,
    ) -> Result<()> {
        let mut transaction = {
            let txns = self.transactions.read();
            txns.get(&id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?
                .clone()
        };
        
        // Update status to failed
        transaction.update_status(TransactionStatus::Failed { error: error.clone() });
        
        // Write to WAL
        self.wal.append(&transaction).await?;
        
        // Trigger compensation
        self.trigger_compensation(id).await?;
        
        // Send event
        let _ = self.event_tx.send(TransactionEvent::Failed(id, error));
        
        Ok(())
    }
    
    /// Trigger compensating transactions
    async fn trigger_compensation(&self, id: Uuid) -> Result<()> {
        let transaction = {
            let txns = self.transactions.read();
            txns.get(&id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?
                .clone()
        };
        
        // Create compensating transaction based on type
        let compensator = compensator::create_compensator(&transaction)?;
        
        // Execute compensation
        compensator.execute().await?;
        
        // Update status
        let mut transaction = transaction;
        transaction.update_status(TransactionStatus::Compensated);
        
        // Write to WAL
        self.wal.append(&transaction).await?;
        
        // Update in memory
        {
            let mut txns = self.transactions.write();
            txns.insert(id, transaction);
        }
        
        // Send event
        let _ = self.event_tx.send(TransactionEvent::Compensated(id));
        
        Ok(())
    }
    
    /// Execute transaction with retry logic
    pub async fn execute_with_retry<F, T>(
        &self,
        transaction: Transaction,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
        T: Send + 'static,
    {
        let id = self.begin_transaction(transaction).await?;
        
        // Execute with retry
        match self.retry_manager.execute_with_retry(operation).await {
            Ok(result) => {
                self.commit_transaction(id).await?;
                Ok(result)
            }
            Err(e) => {
                self.rollback_transaction(id, e.to_string()).await?;
                Err(e)
            }
        }
    }
    
    /// Get transaction by ID
    pub fn get_transaction(&self, id: Uuid) -> Option<Transaction> {
        let txns = self.transactions.read();
        txns.get(&id).cloned()
    }
    
    /// Get metrics
    pub fn get_metrics(&self) -> TransactionMetrics {
        let m = self.metrics.read();
        TransactionMetrics {
            total_transactions: m.total_transactions,
            successful_transactions: m.successful_transactions,
            failed_transactions: m.failed_transactions,
            compensated_transactions: m.compensated_transactions,
            retry_attempts: m.retry_attempts,
            average_latency_us: m.average_latency_us,
        }
    }
    
    /// Recover from WAL on startup
    pub async fn recover(&self) -> Result<()> {
        let entries = self.wal.recover().await?;
        
        for entry in entries {
            if let Ok(transaction) = serde_json::from_slice::<Transaction>(&entry) {
                // Replay transaction
                let mut txns = self.transactions.write();
                txns.insert(transaction.id, transaction.clone());
                
                // Update idempotency cache
                if let Some(ref key) = transaction.idempotency_key {
                    let mut cache = self.idempotency_cache.write();
                    cache.insert(key.clone(), transaction.id);
                }
                
                // Check if needs compensation
                if !transaction.is_terminal() && transaction.needs_retry() {
                    // Schedule for retry
                    let _ = self.retry_manager.schedule_retry(transaction.id);
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_transaction_creation() {
        let tx = Transaction::new(TransactionType::OrderPlacement {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
        });
        
        assert_eq!(tx.status, TransactionStatus::Pending);
        assert_eq!(tx.retry_count, 0);
        assert!(tx.children.is_empty());
    }
    
    #[tokio::test]
    async fn test_idempotency() {
        let manager = TransactionManager::new("/tmp/test_wal")
            .await
            .unwrap();
        
        let tx1 = Transaction::new(TransactionType::BalanceUpdate {
            account_id: Uuid::new_v4(),
            currency: "USDT".to_string(),
            delta: Decimal::from(1000),
            reason: "Deposit".to_string(),
        }).with_idempotency_key("test_key_123".to_string());
        
        let id1 = manager.begin_transaction(tx1.clone()).await.unwrap();
        
        // Same idempotency key should return same ID
        let tx2 = Transaction::new(TransactionType::BalanceUpdate {
            account_id: Uuid::new_v4(),
            currency: "USDT".to_string(),
            delta: Decimal::from(2000), // Different amount
            reason: "Deposit".to_string(),
        }).with_idempotency_key("test_key_123".to_string());
        
        let id2 = manager.begin_transaction(tx2).await.unwrap();
        
        assert_eq!(id1, id2); // Should be same transaction ID
    }
}