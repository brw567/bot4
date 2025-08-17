// Exchange Outage Recovery Tests
// Nexus Requirement: <5s recovery target
// Tests circuit breaker behavior during exchange outages

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::time::sleep;

use crate::infrastructure::circuit_breaker::{GlobalCircuitBreaker, CircuitState, SystemClock};
use crate::websocket::client::WebSocketClient;
use crate::risk_engine::emergency::{KillSwitch, EmergencyStop, TripCondition};

/// Test recovery from complete exchange outage
#[tokio::test]
async fn test_exchange_outage_recovery_under_5s() {
    let start = Instant::now();
    let clock = Arc::new(SystemClock);
    let circuit_breaker = Arc::new(GlobalCircuitBreaker::new(clock.clone()));
    let kill_switch = Arc::new(KillSwitch::new(Some(Duration::from_secs(10))));
    
    // Simulate exchange outage
    simulate_exchange_outage(&circuit_breaker, "Binance").await;
    
    // Verify circuit breaker opened
    assert_eq!(
        circuit_breaker.component_state("Binance"),
        Some(CircuitState::Open)
    );
    
    // Start recovery
    let recovery_start = Instant::now();
    
    // Wait for half-open transition (should be <5s)
    while circuit_breaker.component_state("Binance") == Some(CircuitState::Open) {
        if recovery_start.elapsed() > Duration::from_secs(5) {
            panic!("Recovery took longer than 5s target!");
        }
        sleep(Duration::from_millis(100)).await;
    }
    
    // Verify we're in half-open state
    assert_eq!(
        circuit_breaker.component_state("Binance"),
        Some(CircuitState::HalfOpen)
    );
    
    // Simulate successful reconnection
    simulate_successful_reconnection(&circuit_breaker, "Binance").await;
    
    // Verify circuit closed
    assert_eq!(
        circuit_breaker.component_state("Binance"),
        Some(CircuitState::Closed)
    );
    
    let total_recovery = recovery_start.elapsed();
    println!("Exchange recovery time: {:?}", total_recovery);
    assert!(total_recovery < Duration::from_secs(5), "Recovery must be under 5s");
}

/// Test cascading exchange failures
#[tokio::test]
async fn test_cascading_exchange_failures() {
    let clock = Arc::new(SystemClock);
    let circuit_breaker = Arc::new(GlobalCircuitBreaker::new(clock.clone()));
    let emergency_stop = Arc::new(EmergencyStop::new(Arc::new(KillSwitch::new(None))));
    
    // Simulate multiple exchange failures
    let exchanges = vec!["Binance", "Kraken", "Coinbase"];
    
    for exchange in &exchanges {
        simulate_exchange_outage(&circuit_breaker, exchange).await;
    }
    
    // Check if emergency stop triggered
    emergency_stop.check_conditions();
    
    // Verify global circuit breaker tripped
    let metrics = circuit_breaker.metrics();
    assert!(metrics.open_count >= 3);
    
    // Test recovery sequence
    for exchange in &exchanges {
        simulate_successful_reconnection(&circuit_breaker, exchange).await;
    }
    
    // Verify all recovered
    for exchange in &exchanges {
        assert_eq!(
            circuit_breaker.component_state(exchange),
            Some(CircuitState::Closed)
        );
    }
}

/// Test WebSocket auto-reconnection during outage
#[tokio::test]
async fn test_websocket_auto_reconnect() {
    let reconnect_count = Arc::new(AtomicU64::new(0));
    let connected = Arc::new(AtomicBool::new(false));
    
    // Simulate WebSocket with auto-reconnect
    let ws_client = WebSocketClient::new("wss://stream.binance.com:9443/ws");
    
    // Configure exponential backoff
    ws_client.set_reconnect_strategy(vec![
        Duration::from_millis(100),
        Duration::from_millis(200),
        Duration::from_millis(400),
        Duration::from_millis(800),
        Duration::from_millis(1600),
    ]);
    
    // Simulate connection failure
    simulate_connection_failure(&ws_client).await;
    
    // Start reconnection attempts
    let reconnect_start = Instant::now();
    
    while !connected.load(Ordering::Acquire) {
        if reconnect_start.elapsed() > Duration::from_secs(5) {
            panic!("Reconnection exceeded 5s limit");
        }
        
        // Try to reconnect
        if ws_client.try_reconnect().await.is_ok() {
            connected.store(true, Ordering::Release);
            reconnect_count.fetch_add(1, Ordering::Relaxed);
        }
        
        sleep(Duration::from_millis(100)).await;
    }
    
    let reconnect_time = reconnect_start.elapsed();
    let attempts = reconnect_count.load(Ordering::Relaxed);
    
    println!("Reconnection successful after {} attempts in {:?}", attempts, reconnect_time);
    assert!(reconnect_time < Duration::from_secs(5));
}

/// Test kill switch activation during prolonged outage
#[tokio::test]
async fn test_kill_switch_on_prolonged_outage() {
    let kill_switch = Arc::new(KillSwitch::new(None));
    let emergency_stop = Arc::new(EmergencyStop::new(kill_switch.clone()));
    
    // Add condition for exchange outage
    emergency_stop.add_condition(
        "exchange_health".to_string(),
        || {
            // Simulate unhealthy exchange
            Some(TripCondition::ExchangeIssue {
                exchange: "Binance".to_string(),
                issue: "Connection timeout".to_string(),
            })
        },
    );
    
    // Check conditions (should trigger kill switch)
    emergency_stop.check_conditions();
    
    // Verify kill switch activated
    assert!(kill_switch.is_active());
    
    // Verify reason
    if let Some(TripCondition::ExchangeIssue { exchange, issue }) = kill_switch.get_trigger_reason() {
        assert_eq!(exchange, "Binance");
        assert_eq!(issue, "Connection timeout");
    } else {
        panic!("Expected ExchangeIssue trigger");
    }
}

/// Test recovery plan execution
#[tokio::test]
async fn test_recovery_plan_execution() {
    use crate::risk_engine::emergency::{RecoveryPlan, RecoveryAction};
    
    // Test standard recovery
    let standard_plan = RecoveryPlan::standard();
    assert_eq!(standard_plan.steps.len(), 3);
    assert!(standard_plan.requires_manual_approval);
    assert_eq!(standard_plan.estimated_time, Duration::from_secs(300));
    
    // Test aggressive recovery
    let aggressive_plan = RecoveryPlan::aggressive();
    assert_eq!(aggressive_plan.steps.len(), 3);
    assert!(!aggressive_plan.requires_manual_approval);
    assert_eq!(aggressive_plan.estimated_time, Duration::from_secs(30));
    
    // Simulate plan execution
    for step in &aggressive_plan.steps {
        match &step.action {
            RecoveryAction::CancelAllOrders => {
                println!("Cancelling all orders...");
                sleep(Duration::from_millis(100)).await;
            }
            RecoveryAction::CloseAllPositions => {
                println!("Closing all positions...");
                sleep(Duration::from_millis(200)).await;
            }
            RecoveryAction::ManualIntervention { instructions } => {
                println!("Manual intervention: {}", instructions);
            }
            _ => {}
        }
    }
}

/// Test partial exchange functionality during degraded service
#[tokio::test]
async fn test_degraded_service_handling() {
    let clock = Arc::new(SystemClock);
    let circuit_breaker = Arc::new(GlobalCircuitBreaker::new(clock.clone()));
    
    // Simulate degraded service (50% failure rate)
    for i in 0..10 {
        if i % 2 == 0 {
            simulate_failed_request(&circuit_breaker, "Binance").await;
        } else {
            simulate_successful_request(&circuit_breaker, "Binance").await;
        }
    }
    
    // Circuit should be in half-open or open state
    let state = circuit_breaker.component_state("Binance");
    assert!(
        state == Some(CircuitState::Open) || state == Some(CircuitState::HalfOpen),
        "Circuit should detect degraded service"
    );
}

// Helper functions

async fn simulate_exchange_outage(breaker: &GlobalCircuitBreaker, exchange: &str) {
    // Simulate multiple failures to trip the breaker
    for _ in 0..5 {
        match breaker.acquire(exchange) {
            Permit::Allowed(guard) => {
                guard.record(Outcome::Failure);
            }
            _ => break,
        }
    }
}

async fn simulate_successful_reconnection(breaker: &GlobalCircuitBreaker, exchange: &str) {
    // Simulate successful requests to close the circuit
    for _ in 0..3 {
        if let Permit::Allowed(guard) = breaker.acquire(exchange) {
            guard.record(Outcome::Success);
        }
        sleep(Duration::from_millis(10)).await;
    }
}

async fn simulate_connection_failure(client: &WebSocketClient) {
    client.disconnect().await;
}

async fn simulate_failed_request(breaker: &GlobalCircuitBreaker, exchange: &str) {
    if let Permit::Allowed(guard) = breaker.acquire(exchange) {
        guard.record(Outcome::Failure);
    }
}

async fn simulate_successful_request(breaker: &GlobalCircuitBreaker, exchange: &str) {
    if let Permit::Allowed(guard) = breaker.acquire(exchange) {
        guard.record(Outcome::Success);
    }
}

/// Performance test for recovery time
#[tokio::test]
async fn bench_recovery_performance() {
    let mut recovery_times = Vec::new();
    
    for _ in 0..100 {
        let start = Instant::now();
        
        // Simulate outage and recovery
        let clock = Arc::new(SystemClock);
        let breaker = Arc::new(GlobalCircuitBreaker::new(clock));
        
        simulate_exchange_outage(&breaker, "test").await;
        sleep(Duration::from_millis(100)).await;
        simulate_successful_reconnection(&breaker, "test").await;
        
        recovery_times.push(start.elapsed());
    }
    
    // Calculate statistics
    recovery_times.sort();
    let p50 = recovery_times[50];
    let p95 = recovery_times[95];
    let p99 = recovery_times[99];
    
    println!("Recovery Time Distribution:");
    println!("  p50: {:?}", p50);
    println!("  p95: {:?}", p95);
    println!("  p99: {:?}", p99);
    
    assert!(p99 < Duration::from_secs(5), "p99 recovery must be under 5s");
}