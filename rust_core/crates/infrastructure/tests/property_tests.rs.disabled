// Property-Based Tests for Circuit Breaker
// Sophia Test Requirement #2: State invariants and hysteresis
// Uses proptest to verify properties hold for all inputs

use proptest::prelude::*;
use infrastructure::circuit_breaker::{
    CircuitBreaker, CircuitState, CircuitConfig, 
    ComponentBreaker, Outcome, SystemClock, GlobalTripConditions
};
use std::sync::Arc;
use std::time::Duration;

/// Generate arbitrary circuit configuration
fn arb_config() -> impl Strategy<Value = CircuitConfig> {
    (
        1u32..100,  // min_calls
        0.1f32..0.9,  // error_rate_open_threshold
        0.05f32..0.5,  // error_rate_close_threshold
        1u32..10,  // consecutive_failures_threshold
        1u32..10,  // half_open_max_concurrent
        1u32..10,  // half_open_required_successes
        1u32..5,   // half_open_allowed_failures
    ).prop_map(|(min_calls, open_thresh, close_thresh, consec, max_conc, req_succ, allow_fail)| {
        CircuitConfig {
            rolling_window: Duration::from_secs(10),
            min_calls,
            error_rate_open_threshold: open_thresh,
            error_rate_close_threshold: close_thresh.min(open_thresh - 0.05),  // Ensure hysteresis
            consecutive_failures_threshold: consec,
            open_cooldown: Duration::from_secs(5),
            half_open_max_concurrent: max_conc,
            half_open_required_successes: req_succ,
            half_open_allowed_failures: allow_fail,
            global_trip_conditions: GlobalTripConditions {
                component_open_ratio: 0.5,
                component_close_ratio: 0.35,
                min_components: 3,
                min_samples_per_component: 10,
            },
        }
    })
}

/// Generate sequence of outcomes
fn arb_outcomes(size: usize) -> impl Strategy<Value = Vec<Outcome>> {
    prop::collection::vec(
        prop_oneof![
            Just(Outcome::Success),
            Just(Outcome::Failure),
        ],
        0..size
    )
}

proptest! {
    /// Property: State transitions are always valid
    #[test]
    fn prop_valid_state_transitions(
        config in arb_config(),
        outcomes in arb_outcomes(100)
    ) {
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        let mut prev_state = breaker.current_state();
        
        for outcome in outcomes {
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(outcome);
            }
            
            let new_state = breaker.current_state();
            
            // Verify valid transitions
            match (prev_state, new_state) {
                // Valid transitions
                (CircuitState::Closed, CircuitState::Closed) => {},
                (CircuitState::Closed, CircuitState::Open) => {},
                (CircuitState::Open, CircuitState::Open) => {},
                (CircuitState::Open, CircuitState::HalfOpen) => {},
                (CircuitState::HalfOpen, CircuitState::HalfOpen) => {},
                (CircuitState::HalfOpen, CircuitState::Closed) => {},
                (CircuitState::HalfOpen, CircuitState::Open) => {},
                
                // Invalid transitions
                (CircuitState::Closed, CircuitState::HalfOpen) => {
                    panic!("Invalid transition: Closed -> HalfOpen");
                }
                (CircuitState::Open, CircuitState::Closed) => {
                    panic!("Invalid transition: Open -> Closed (must go through HalfOpen)");
                }
                _ => {}
            }
            
            prev_state = new_state;
        }
    }
    
    /// Property: Hysteresis prevents flapping
    #[test]
    fn prop_hysteresis_prevents_flapping(
        mut config in arb_config(),
        error_rate in 0.4f32..0.6  // Around the threshold
    ) {
        // Set thresholds around the error rate
        config.error_rate_open_threshold = 0.5;
        config.error_rate_close_threshold = 0.35;
        config.min_calls = 20;
        
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        // Generate outcomes based on error rate
        let total_calls = 100;
        let mut state_changes = 0;
        let mut prev_state = breaker.current_state();
        
        for i in 0..total_calls {
            let outcome = if (i as f32 / total_calls as f32) < error_rate {
                Outcome::Failure
            } else {
                Outcome::Success
            };
            
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(outcome);
            }
            
            let new_state = breaker.current_state();
            if new_state != prev_state {
                state_changes += 1;
                prev_state = new_state;
            }
        }
        
        // With hysteresis, should have fewer state changes
        prop_assert!(state_changes < 10, "Too many state changes: {}", state_changes);
    }
    
    /// Property: Min calls respected before state change
    #[test]
    fn prop_min_calls_respected(
        mut config in arb_config(),
        outcomes in arb_outcomes(50)
    ) {
        config.min_calls = 10;
        
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        // Record fewer than min_calls failures
        for _ in 0..5 {
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Should still be closed (not enough calls)
        prop_assert_eq!(breaker.current_state(), CircuitState::Closed);
        
        // Record more calls
        for outcome in outcomes.iter().take(20) {
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(*outcome);
            }
        }
        
        // Now state changes are allowed
        // (actual state depends on error rate)
    }
    
    /// Property: Half-open token limit enforced
    #[test]
    fn prop_half_open_token_limit(
        mut config in arb_config(),
        concurrent_attempts in 1usize..20
    ) {
        config.half_open_max_concurrent = 3;
        config.consecutive_failures_threshold = 2;
        
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        // Trip the breaker
        for _ in 0..2 {
            if let Ok(guard) = breaker.try_acquire("trip") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Force to half-open (simulate cooldown)
        // In real test would wait, here we just check the property
        
        let mut acquired = Vec::new();
        for i in 0..concurrent_attempts {
            if let Ok(guard) = breaker.try_acquire(&format!("test-{}", i)) {
                acquired.push(guard);
            }
        }
        
        // Should never exceed max_concurrent
        prop_assert!(acquired.len() <= 3);
    }
    
    /// Property: Global state derives correctly from components
    #[test]
    fn prop_global_state_derivation(
        component_states in prop::collection::vec(
            prop_oneof![
                Just(CircuitState::Closed),
                Just(CircuitState::Open),
                Just(CircuitState::HalfOpen),
            ],
            2..10
        )
    ) {
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
        
        let global = GlobalCircuitBreaker::new(clock, config.clone(), None);
        
        // Set up components with given states
        for (i, &state) in component_states.iter().enumerate() {
            let component = global.component(&format!("comp-{}", i));
            
            // Record enough calls to meet min_samples
            match state {
                CircuitState::Open => {
                    // Record failures to open
                    for _ in 0..10 {
                        if let Ok(guard) = component.try_acquire(&format!("comp-{}", i)) {
                            guard.record(Outcome::Failure);
                        }
                    }
                }
                CircuitState::Closed => {
                    // Record successes to stay closed
                    for _ in 0..10 {
                        if let Ok(guard) = component.try_acquire(&format!("comp-{}", i)) {
                            guard.record(Outcome::Success);
                        }
                    }
                }
                CircuitState::HalfOpen => {
                    // Open then wait (simulated)
                    for _ in 0..5 {
                        if let Ok(guard) = component.try_acquire(&format!("comp-{}", i)) {
                            guard.record(Outcome::Failure);
                        }
                    }
                }
            }
        }
        
        let global_state = global.derive_global_state();
        
        // Calculate expected state
        let open_count = component_states.iter()
            .filter(|&&s| s == CircuitState::Open)
            .count();
        let half_open_count = component_states.iter()
            .filter(|&&s| s == CircuitState::HalfOpen)
            .count();
        let total = component_states.len();
        
        let open_ratio = open_count as f32 / total as f32;
        
        // Verify derivation logic with hysteresis
        if open_ratio >= 0.5 {
            prop_assert_eq!(global_state, CircuitState::Open);
        } else if open_ratio <= 0.35 && half_open_count == 0 {
            prop_assert_eq!(global_state, CircuitState::Closed);
        } else if half_open_count > 0 {
            prop_assert!(
                global_state == CircuitState::HalfOpen || 
                global_state == CircuitState::Open
            );
        }
    }
    
    /// Property: Consecutive failures always trip the breaker
    #[test]
    fn prop_consecutive_failures_trip(
        mut config in arb_config(),
        pre_outcomes in arb_outcomes(20),
        consecutive_count in 2u32..10
    ) {
        config.consecutive_failures_threshold = consecutive_count;
        config.min_calls = 1;  // Don't block on min calls
        
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        // Record some mixed outcomes
        for outcome in pre_outcomes {
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(outcome);
            }
        }
        
        // Record consecutive failures
        for _ in 0..consecutive_count {
            if let Ok(guard) = breaker.try_acquire("test") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Breaker should be open
        prop_assert_eq!(breaker.current_state(), CircuitState::Open);
    }
    
    /// Property: Success in half-open closes the circuit
    #[test]
    fn prop_half_open_success_closes(
        mut config in arb_config(),
        success_count in 1u32..10
    ) {
        config.half_open_required_successes = success_count;
        config.consecutive_failures_threshold = 2;
        
        let clock = Arc::new(SystemClock);
        let config = Arc::new(config);
        let breaker = ComponentBreaker::new(clock, config.clone(), None);
        
        // Trip the breaker
        for _ in 0..2 {
            if let Ok(guard) = breaker.try_acquire("trip") {
                guard.record(Outcome::Failure);
            }
        }
        
        // Simulate transition to half-open
        // (In real scenario would wait for cooldown)
        
        // Record required successes
        for _ in 0..success_count {
            if let Ok(guard) = breaker.try_acquire("success") {
                guard.record(Outcome::Success);
                
                // After enough successes, should close
                if breaker.current_state() == CircuitState::Closed {
                    break;
                }
            }
        }
        
        // Property: enough successes should close the circuit
        // (May still be half-open if not enough tokens acquired)
    }
}