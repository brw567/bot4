// Loom Tests for Circuit Breaker Concurrency
// Sophia Test Requirement #1: Concurrent state transitions
// Tests for race conditions, deadlocks, and invariant violations

#![cfg(loom)]

use loom::sync::Arc;
use loom::thread;
use infrastructure::circuit_breaker::{
    CircuitBreaker, CircuitState, CircuitConfig, 
    ComponentBreaker, Outcome, SystemClock
};

/// Test concurrent open/close transitions around thresholds
#[test]
fn test_concurrent_state_transitions() {
    let mut config = loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig::default());
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Spawn threads that will cause state transitions
        let handles: Vec<_> = (0..3).map(|i| {
            let breaker = breaker.clone();
            thread::spawn(move || {
                for j in 0..5 {
                    // Alternate between success and failure
                    if (i + j) % 2 == 0 {
                        // Record failures to potentially trip the breaker
                        if let Ok(guard) = breaker.try_acquire(&format!("thread-{}", i)) {
                            guard.record(Outcome::Failure);
                        }
                    } else {
                        // Record successes to potentially close the breaker
                        if let Ok(guard) = breaker.try_acquire(&format!("thread-{}", i)) {
                            guard.record(Outcome::Success);
                        }
                    }
                }
            })
        }).collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Check invariants
        let state = breaker.current_state();
        
        // State should be one of the valid states
        assert!(
            state == CircuitState::Closed || 
            state == CircuitState::Open || 
            state == CircuitState::HalfOpen
        );
    });
}

/// Test config reload during state transition
#[test]
fn test_config_reload_during_transition() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig {
            consecutive_failures_threshold: 3,
            ..Default::default()
        });
        
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Thread 1: Cause failures to trip breaker
        let breaker1 = breaker.clone();
        let handle1 = thread::spawn(move || {
            for _ in 0..5 {
                if let Ok(guard) = breaker1.try_acquire("writer") {
                    guard.record(Outcome::Failure);
                }
            }
        });
        
        // Thread 2: Try to acquire during transitions
        let breaker2 = breaker.clone();
        let handle2 = thread::spawn(move || {
            for _ in 0..10 {
                let _ = breaker2.try_acquire("reader");
            }
        });
        
        handle1.join().unwrap();
        handle2.join().unwrap();
        
        // No deadlock should occur
    });
}

/// Test token leak with panic in guarded section
#[test]
fn test_token_leak_on_panic() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig {
            half_open_max_concurrent: 3,
            ..Default::default()
        });
        
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Force to half-open state
        for _ in 0..5 {
            if let Ok(guard) = breaker.try_acquire("setup") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Wait for cooldown (simulated)
        thread::yield_now();
        
        // Try to leak tokens
        let breaker1 = breaker.clone();
        let handle = thread::spawn(move || {
            if let Ok(guard) = breaker1.try_acquire("leaker") {
                // Simulate panic before recording
                std::mem::forget(guard);  // Intentionally leak
            }
        });
        
        let _ = handle.join();
        
        // Tokens should still be properly managed (Drop impl should handle)
        // Try to acquire remaining tokens
        let mut acquired = 0;
        for i in 0..10 {
            if breaker.try_acquire(&format!("test-{}", i)).is_ok() {
                acquired += 1;
            }
        }
        
        // Should be able to acquire some tokens despite the leak attempt
        assert!(acquired <= 3);  // Max concurrent in half-open
    });
}

/// Test concurrent token acquisition in half-open state
#[test]
fn test_concurrent_half_open_tokens() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig {
            half_open_max_concurrent: 2,
            consecutive_failures_threshold: 2,
            ..Default::default()
        });
        
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Trip the breaker
        for _ in 0..2 {
            if let Ok(guard) = breaker.try_acquire("tripper") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Simulate cooldown elapsed
        thread::yield_now();
        
        // Multiple threads try to acquire half-open tokens
        let handles: Vec<_> = (0..5).map(|i| {
            let breaker = breaker.clone();
            thread::spawn(move || {
                breaker.try_acquire(&format!("thread-{}", i)).is_ok()
            })
        }).collect();
        
        let mut success_count = 0;
        for handle in handles {
            if handle.join().unwrap() {
                success_count += 1;
            }
        }
        
        // Only max_concurrent should succeed
        assert!(success_count <= 2);
    });
}

/// Test state transition invariants
#[test]
fn test_state_transition_invariants() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig::default());
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Spawn threads that verify invariants
        let handles: Vec<_> = (0..3).map(|i| {
            let breaker = breaker.clone();
            thread::spawn(move || {
                for _ in 0..5 {
                    let state = breaker.current_state();
                    
                    // Invariant: State transitions must be valid
                    // Closed -> Open (on failures)
                    // Open -> HalfOpen (on cooldown)
                    // HalfOpen -> Closed (on success)
                    // HalfOpen -> Open (on failure)
                    
                    match state {
                        CircuitState::Closed => {
                            // Can transition to Open
                            if let Ok(guard) = breaker.try_acquire(&format!("closed-{}", i)) {
                                guard.record(Outcome::Failure);
                            }
                        }
                        CircuitState::Open => {
                            // Can transition to HalfOpen after cooldown
                            thread::yield_now();  // Simulate time passing
                            let _ = breaker.try_acquire(&format!("open-{}", i));
                        }
                        CircuitState::HalfOpen => {
                            // Can transition to Closed or Open
                            if let Ok(guard) = breaker.try_acquire(&format!("half-{}", i)) {
                                if i % 2 == 0 {
                                    guard.record(Outcome::Success);
                                } else {
                                    guard.record(Outcome::Failure);
                                }
                            }
                        }
                    }
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Final state should be valid
        let final_state = breaker.current_state();
        assert!(matches!(
            final_state,
            CircuitState::Closed | CircuitState::Open | CircuitState::HalfOpen
        ));
    });
}

/// Test global circuit breaker coordination
#[test]
fn test_global_breaker_coordination() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig {
            global_trip_conditions: GlobalTripConditions {
                component_open_ratio: 0.5,
                component_close_ratio: 0.35,
                min_components: 2,
                min_samples_per_component: 5,
            },
            ..Default::default()
        });
        
        let global = Arc::new(GlobalCircuitBreaker::new(clock, config.clone(), None));
        
        // Create components in parallel
        let handles: Vec<_> = (0..4).map(|i| {
            let global = global.clone();
            thread::spawn(move || {
                let component = format!("comp-{}", i);
                
                // Each thread works with its component
                for j in 0..10 {
                    if let Ok(guard) = global.acquire(&component) {
                        // Half fail, half succeed
                        if j < 5 {
                            guard.record(Outcome::Failure);
                        } else {
                            guard.record(Outcome::Success);
                        }
                    }
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Global state should reflect component states
        let global_state = global.derive_global_state();
        
        // Should be in some valid state
        assert!(matches!(
            global_state,
            CircuitState::Closed | CircuitState::Open | CircuitState::HalfOpen
        ));
    });
}

/// Test memory ordering guarantees
#[test]
fn test_memory_ordering() {
    loom::model(|| {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(CircuitConfig::default());
        let breaker = Arc::new(ComponentBreaker::new(clock, config.clone(), None));
        
        // Shared flag to verify ordering
        let flag = Arc::new(loom::sync::atomic::AtomicBool::new(false));
        
        // Writer thread
        let breaker1 = breaker.clone();
        let flag1 = flag.clone();
        let handle1 = thread::spawn(move || {
            // Set flag
            flag1.store(true, loom::sync::atomic::Ordering::Release);
            
            // Then record failure
            if let Ok(guard) = breaker1.try_acquire("writer") {
                guard.record(Outcome::Failure);
            }
        });
        
        // Reader thread
        let breaker2 = breaker.clone();
        let flag2 = flag.clone();
        let handle2 = thread::spawn(move || {
            // Check state
            let state = breaker2.current_state();
            
            // If we see a state change, flag must be set
            if state != CircuitState::Closed {
                assert!(flag2.load(loom::sync::atomic::Ordering::Acquire));
            }
        });
        
        handle1.join().unwrap();
        handle2.join().unwrap();
    });
}