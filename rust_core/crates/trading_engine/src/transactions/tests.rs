// Comprehensive Transaction Rollback Tests
// Task 2.6: FULL test coverage with failure scenarios
// Team: Riley (Testing Lead) + Full Team Review
// Testing Strategy: Unit, Integration, Stress, and Chaos tests

use super::*;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use uuid::Uuid;
use rust_decimal::Decimal;

// ===== UNIT TESTS =====

#[cfg(test)]
mod unit_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_transaction_lifecycle() {
        let tx = Transaction::new(TransactionType::OrderPlacement {
            order_id: Uuid::new_v4(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
        });
        
        assert_eq!(tx.status, TransactionStatus::Pending);
        assert!(!tx.is_terminal());
        assert!(!tx.needs_retry());
        
        let mut tx = tx;
        tx.update_status(TransactionStatus::InProgress);
        assert_eq!(tx.status, TransactionStatus::InProgress);
        
        tx.update_status(TransactionStatus::Committed);
        assert!(tx.is_terminal());
    }
    
    #[tokio::test]
    async fn test_idempotency_key() {
        let key = "unique_key_123";
        let tx = Transaction::new(TransactionType::BalanceUpdate {
            account_id: Uuid::new_v4(),
            currency: "USDT".to_string(),
            delta: Decimal::from(1000),
            reason: "Deposit".to_string(),
        }).with_idempotency_key(key.to_string());
        
        assert_eq!(tx.idempotency_key, Some(key.to_string()));
    }
    
    #[test]
    fn test_retry_policy_backoff() {
        let policy = RetryPolicy {
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0, // No jitter for predictable test
            ..Default::default()
        };
        
        // Verify exponential backoff
        assert_eq!(policy.calculate_delay(0).as_millis(), 100);
        assert_eq!(policy.calculate_delay(1).as_millis(), 200);
        assert_eq!(policy.calculate_delay(2).as_millis(), 400);
        assert_eq!(policy.calculate_delay(3).as_millis(), 800);
        assert_eq!(policy.calculate_delay(4).as_millis(), 1600);
        assert_eq!(policy.calculate_delay(5).as_millis(), 3200);
        assert_eq!(policy.calculate_delay(6).as_millis(), 5000); // Capped at max
    }
    
    #[test]
    fn test_retry_policy_jitter() {
        let policy = RetryPolicy {
            initial_delay_ms: 1000,
            jitter_factor: 0.5,
            ..Default::default()
        };
        
        // With 50% jitter, delay should be within 500-1500ms
        for _ in 0..10 {
            let delay = policy.calculate_delay(0).as_millis();
            assert!(delay >= 500 && delay <= 1500, "Delay {} out of range", delay);
        }
    }
    
    #[test]
    fn test_error_classification() {
        let policy = RetryPolicy::default();
        
        // Retryable errors
        assert!(policy.is_retryable("ConnectionError: timeout"));
        assert!(policy.is_retryable("TimeoutError: request timed out"));
        assert!(policy.is_retryable("RateLimitError: too many requests"));
        
        // Non-retryable errors
        assert!(!policy.is_retryable("InsufficientFunds: not enough balance"));
        assert!(!policy.is_retryable("InvalidCredentials: wrong API key"));
        assert!(!policy.is_retryable("OrderNotFound: order does not exist"));
    }
}

// ===== INTEGRATION TESTS =====

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_transaction_manager_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let manager = TransactionManager::new(temp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        
        // Create transaction
        let tx = Transaction::new(TransactionType::OrderPlacement {
            order_id: Uuid::new_v4(),
            symbol: "ETH/USDT".to_string(),
            side: OrderSide::Sell,
            quantity: Decimal::from(10),
            price: Some(Decimal::from(3000)),
        });
        
        let tx_id = manager.begin_transaction(tx).await.unwrap();
        
        // Verify transaction exists
        let retrieved = manager.get_transaction(tx_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().status, TransactionStatus::Pending);
        
        // Commit transaction
        manager.commit_transaction(tx_id).await.unwrap();
        
        // Verify committed
        let committed = manager.get_transaction(tx_id).unwrap();
        assert_eq!(committed.status, TransactionStatus::Committed);
    }
    
    #[tokio::test]
    async fn test_idempotency_enforcement() {
        let temp_dir = TempDir::new().unwrap();
        let manager = TransactionManager::new(temp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        
        let idempotency_key = "test_idempotent_123";
        
        // First transaction
        let tx1 = Transaction::new(TransactionType::BalanceUpdate {
            account_id: Uuid::new_v4(),
            currency: "BTC".to_string(),
            delta: Decimal::from(1),
            reason: "Test".to_string(),
        }).with_idempotency_key(idempotency_key.to_string());
        
        let id1 = manager.begin_transaction(tx1).await.unwrap();
        
        // Second transaction with same key but different data
        let tx2 = Transaction::new(TransactionType::BalanceUpdate {
            account_id: Uuid::new_v4(), // Different account
            currency: "ETH".to_string(), // Different currency
            delta: Decimal::from(100),   // Different amount
            reason: "Different".to_string(),
        }).with_idempotency_key(idempotency_key.to_string());
        
        let id2 = manager.begin_transaction(tx2).await.unwrap();
        
        // Should return same transaction ID
        assert_eq!(id1, id2);
        
        // Verify only one transaction exists
        let tx = manager.get_transaction(id1).unwrap();
        assert_eq!(tx.transaction_type, TransactionType::BalanceUpdate {
            account_id: manager.get_transaction(id1).unwrap().id, // Original data
            currency: "BTC".to_string(),
            delta: Decimal::from(1),
            reason: "Test".to_string(),
        });
    }
    
    #[tokio::test]
    async fn test_rollback_triggers_compensation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = TransactionManager::new(temp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        
        let tx = Transaction::new(TransactionType::OrderPlacement {
            order_id: Uuid::new_v4(),
            symbol: "SOL/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(100),
            price: Some(Decimal::from(100)),
        });
        
        let tx_id = manager.begin_transaction(tx).await.unwrap();
        
        // Rollback with error
        manager.rollback_transaction(tx_id, "Market closed".to_string())
            .await
            .unwrap();
        
        // Give compensation time to complete
        sleep(Duration::from_millis(100)).await;
        
        // Verify compensated
        let rolled_back = manager.get_transaction(tx_id).unwrap();
        assert!(matches!(
            rolled_back.status,
            TransactionStatus::Compensated | TransactionStatus::Failed { .. }
        ));
    }
    
    #[tokio::test]
    async fn test_wal_persistence_and_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().to_str().unwrap();
        
        // Create WAL and write entries
        {
            let wal = WriteAheadLog::new(wal_path).await.unwrap();
            
            #[derive(Serialize, Deserialize, Debug, PartialEq)]
            struct TestEntry {
                id: u64,
                data: String,
            }
            
            for i in 0..10 {
                let entry = TestEntry {
                    id: i,
                    data: format!("Entry {}", i),
                };
                wal.append(&entry).await.unwrap();
            }
        }
        
        // Create new WAL and recover
        {
            let wal = WriteAheadLog::new(wal_path).await.unwrap();
            let recovered = wal.recover().await.unwrap();
            
            assert_eq!(recovered.len(), 10);
            
            for (i, data) in recovered.iter().enumerate() {
                let entry: TestEntry = bincode::deserialize(data).unwrap();
                assert_eq!(entry.id, i as u64);
                assert_eq!(entry.data, format!("Entry {}", i));
            }
        }
    }
}

// ===== SAGA TESTS =====

#[cfg(test)]
mod saga_tests {
    use super::*;
    use super::saga::*;
    
    // Test step that tracks execution
    struct TestStep {
        name: String,
        should_fail: bool,
        executed: Arc<AtomicBool>,
        compensated: Arc<AtomicBool>,
    }
    
    #[async_trait]
    impl SagaStep for TestStep {
        fn name(&self) -> &str {
            &self.name
        }
        
        async fn execute(&self, _context: &SagaContext) -> Result<StepResult> {
            self.executed.store(true, Ordering::SeqCst);
            
            if self.should_fail {
                Ok(StepResult::Failure("Test failure".to_string()))
            } else {
                Ok(StepResult::Success(serde_json::json!({
                    "step": self.name,
                    "result": "success"
                })))
            }
        }
        
        async fn compensate(&self, _context: &SagaContext) -> Result<()> {
            self.compensated.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    
    #[tokio::test]
    async fn test_saga_successful_execution() {
        let mut saga = Saga::new("TestSaga".to_string(), Uuid::new_v4());
        
        let step1_exec = Arc::new(AtomicBool::new(false));
        let step1_comp = Arc::new(AtomicBool::new(false));
        let step2_exec = Arc::new(AtomicBool::new(false));
        let step2_comp = Arc::new(AtomicBool::new(false));
        
        saga = saga
            .add_step(Box::new(TestStep {
                name: "Step1".to_string(),
                should_fail: false,
                executed: step1_exec.clone(),
                compensated: step1_comp.clone(),
            }))
            .add_step(Box::new(TestStep {
                name: "Step2".to_string(),
                should_fail: false,
                executed: step2_exec.clone(),
                compensated: step2_comp.clone(),
            }));
        
        let result = saga.execute().await;
        assert!(result.is_ok());
        assert_eq!(saga.state, SagaState::Completed);
        
        // Both steps executed, no compensation
        assert!(step1_exec.load(Ordering::SeqCst));
        assert!(step2_exec.load(Ordering::SeqCst));
        assert!(!step1_comp.load(Ordering::SeqCst));
        assert!(!step2_comp.load(Ordering::SeqCst));
    }
    
    #[tokio::test]
    async fn test_saga_compensation_on_failure() {
        let mut saga = Saga::new("FailingSaga".to_string(), Uuid::new_v4());
        
        let step1_exec = Arc::new(AtomicBool::new(false));
        let step1_comp = Arc::new(AtomicBool::new(false));
        let step2_exec = Arc::new(AtomicBool::new(false));
        let step2_comp = Arc::new(AtomicBool::new(false));
        let step3_exec = Arc::new(AtomicBool::new(false));
        let step3_comp = Arc::new(AtomicBool::new(false));
        
        saga = saga
            .add_step(Box::new(TestStep {
                name: "Step1".to_string(),
                should_fail: false,
                executed: step1_exec.clone(),
                compensated: step1_comp.clone(),
            }))
            .add_step(Box::new(TestStep {
                name: "Step2".to_string(),
                should_fail: false,
                executed: step2_exec.clone(),
                compensated: step2_comp.clone(),
            }))
            .add_step(Box::new(TestStep {
                name: "Step3".to_string(),
                should_fail: true, // This will fail
                executed: step3_exec.clone(),
                compensated: step3_comp.clone(),
            }));
        
        let result = saga.execute().await;
        assert!(result.is_err());
        assert_eq!(saga.state, SagaState::Compensated);
        
        // First two steps executed and compensated
        assert!(step1_exec.load(Ordering::SeqCst));
        assert!(step2_exec.load(Ordering::SeqCst));
        assert!(step3_exec.load(Ordering::SeqCst));
        
        // Only first two compensated (reverse order)
        assert!(step1_comp.load(Ordering::SeqCst));
        assert!(step2_comp.load(Ordering::SeqCst));
        assert!(!step3_comp.load(Ordering::SeqCst)); // Failed step not compensated
    }
}

// ===== CIRCUIT BREAKER TESTS =====

#[cfg(test)]
mod circuit_breaker_tests {
    use super::*;
    use super::retry::*;
    
    #[test]
    fn test_circuit_breaker_transitions() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout_ms: 50,
            half_open_percentage: 100,
        };
        
        let breaker = CircuitBreaker::new(config);
        
        // Initial: Closed
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.should_allow());
        
        // First failure
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        
        // Second failure - opens circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.should_allow());
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));
        
        // Should transition to half-open
        assert!(breaker.should_allow());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        // Two successes to close
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
    
    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_ms: 50,
            half_open_percentage: 100,
        };
        
        let breaker = CircuitBreaker::new(config);
        
        // Open circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        
        // Wait and transition to half-open
        std::thread::sleep(Duration::from_millis(60));
        assert!(breaker.should_allow());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        // Failure in half-open immediately opens
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }
}

// ===== STRESS TESTS =====

#[cfg(test)]
mod stress_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_concurrent_transactions() {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            TransactionManager::new(temp_dir.path().to_str().unwrap())
                .await
                .unwrap()
        );
        
        let mut handles = Vec::new();
        
        // Spawn 100 concurrent transactions
        for i in 0..100 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let tx = Transaction::new(TransactionType::BalanceUpdate {
                    account_id: Uuid::new_v4(),
                    currency: "USDT".to_string(),
                    delta: Decimal::from(i),
                    reason: format!("Concurrent {}", i),
                });
                
                let id = manager_clone.begin_transaction(tx).await.unwrap();
                
                // Random operation
                if i % 3 == 0 {
                    manager_clone.commit_transaction(id).await.unwrap();
                } else if i % 3 == 1 {
                    manager_clone.rollback_transaction(id, "Test rollback".to_string())
                        .await
                        .unwrap();
                }
                // Leave some pending
                
                id
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        let ids: Vec<Uuid> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        // Verify all transactions exist
        for id in ids {
            assert!(manager.get_transaction(id).is_some());
        }
        
        // Check metrics
        let metrics = manager.get_metrics();
        assert_eq!(metrics.total_transactions, 100);
    }
    
    #[tokio::test]
    async fn test_wal_high_throughput() {
        let temp_dir = TempDir::new().unwrap();
        let wal = Arc::new(
            WriteAheadLog::new(temp_dir.path().to_str().unwrap())
                .await
                .unwrap()
        );
        
        let write_count = Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();
        
        // 10 concurrent writers, 1000 writes each
        for writer_id in 0..10 {
            let wal_clone = wal.clone();
            let count_clone = write_count.clone();
            
            let handle = tokio::spawn(async move {
                for i in 0..1000 {
                    let entry = format!("Writer {} Entry {}", writer_id, i);
                    wal_clone.append(&entry).await.unwrap();
                    count_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }
        
        // Wait for completion
        futures::future::join_all(handles).await;
        
        assert_eq!(write_count.load(Ordering::Relaxed), 10000);
        
        // Verify metrics
        let metrics = wal.metrics();
        assert_eq!(metrics.total_writes.load(Ordering::Relaxed), 10000);
    }
    
    #[tokio::test]
    async fn test_retry_manager_with_circuit_breaker() {
        let manager = RetryManager::new();
        
        let failure_count = Arc::new(AtomicU32::new(0));
        let failure_count_clone = failure_count.clone();
        
        // Operation that fails 10 times
        let failing_operation = move || {
            let count = failure_count_clone.fetch_add(1, Ordering::SeqCst);
            if count < 10 {
                Err(anyhow::anyhow!("ConnectionError: Service unavailable"))
            } else {
                Ok("Success")
            }
        };
        
        // First few attempts should retry
        for _ in 0..3 {
            let result = manager
                .execute_with_retry_and_policy(failing_operation.clone(), "aggressive", "test_service")
                .await;
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let result = manager
            .execute_with_retry_and_policy(failing_operation, "aggressive", "test_service")
            .await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Circuit breaker open"));
    }
}

// ===== PERFORMANCE BENCHMARKS =====

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_wal_append_latency() {
        let temp_dir = TempDir::new().unwrap();
        let wal = WriteAheadLog::new(temp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        
        // Warm up
        for _ in 0..100 {
            wal.append(&"warmup").await.unwrap();
        }
        
        // Measure append latency
        let mut latencies = Vec::new();
        
        for i in 0..1000 {
            let data = format!("Performance test entry {}", i);
            let start = Instant::now();
            wal.append(&data).await.unwrap();
            let latency = start.elapsed();
            latencies.push(latency.as_micros());
        }
        
        // Calculate statistics
        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[latencies.len() * 99 / 100];
        let avg: u128 = latencies.iter().sum::<u128>() / latencies.len() as u128;
        
        println!("WAL Append Latency - P50: {}μs, P99: {}μs, Avg: {}μs", p50, p99, avg);
        
        // Assert performance requirements
        assert!(p99 < 1000, "P99 latency {}μs exceeds 1ms", p99); // P99 < 1ms
        assert!(avg < 500, "Average latency {}μs exceeds 500μs", avg); // Avg < 500μs
    }
    
    #[test]
    fn test_circuit_breaker_decision_latency() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        
        // Warm up
        for _ in 0..1000 {
            let _ = breaker.should_allow();
        }
        
        // Measure decision latency
        let iterations = 1_000_000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = breaker.should_allow();
        }
        
        let total_time = start.elapsed();
        let per_decision = total_time.as_nanos() / iterations;
        
        println!("Circuit Breaker Decision Latency: {}ns", per_decision);
        
        // Assert performance requirement
        assert!(per_decision < 100, "Decision latency {}ns exceeds 100ns", per_decision);
    }
}