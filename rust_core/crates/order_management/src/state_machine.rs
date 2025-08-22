// Order State Machine
// Implements atomic state transitions with validation
// Ensures orders can't get stuck in invalid states

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString, FromRepr};
use thiserror::Error;
use tracing::{debug, warn, error};

use crate::order::OrderId;

/// Order states in the lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString, FromRepr)]
#[repr(u8)]
pub enum OrderState {
    /// Order created but not yet submitted
    Created = 0,
    /// Order validated and ready for submission
    Validated = 1,
    /// Order submitted to exchange
    Submitted = 2,
    /// Order acknowledged by exchange
    Acknowledged = 3,
    /// Order partially filled
    PartiallyFilled = 4,
    /// Order completely filled
    Filled = 5,
    /// Order cancelled
    Cancelled = 6,
    /// Order rejected by exchange
    Rejected = 7,
    /// Order expired
    Expired = 8,
    /// Order failed due to error
    Failed = 9,
}

impl OrderState {
    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::Rejected
                | OrderState::Expired
                | OrderState::Failed
        )
    }
    
    /// Check if order is active (can still be filled)
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            OrderState::Submitted | OrderState::Acknowledged | OrderState::PartiallyFilled
        )
    }
    
    /// Check if transition to another state is valid
    pub fn can_transition_to(&self, next: OrderState) -> bool {
        match self {
            OrderState::Created => matches!(
                next,
                OrderState::Validated | OrderState::Failed
            ),
            OrderState::Validated => matches!(
                next,
                OrderState::Submitted | OrderState::Failed | OrderState::Cancelled
            ),
            OrderState::Submitted => matches!(
                next,
                OrderState::Acknowledged
                    | OrderState::PartiallyFilled
                    | OrderState::Filled
                    | OrderState::Rejected
                    | OrderState::Failed
            ),
            OrderState::Acknowledged => matches!(
                next,
                OrderState::PartiallyFilled
                    | OrderState::Filled
                    | OrderState::Cancelled
                    | OrderState::Expired
                    | OrderState::Failed
            ),
            OrderState::PartiallyFilled => matches!(
                next,
                OrderState::Filled | OrderState::Cancelled | OrderState::Expired
            ),
            // Terminal states can't transition
            OrderState::Filled
            | OrderState::Cancelled
            | OrderState::Rejected
            | OrderState::Expired
            | OrderState::Failed => false,
        }
    }
}

/// State transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub order_id: OrderId,
    pub from_state: OrderState,
    pub to_state: OrderState,
    pub timestamp: DateTime<Utc>,
    pub reason: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Order state machine with atomic transitions
pub struct OrderStateMachine {
    order_id: OrderId,
    current_state: Arc<AtomicU8>,
    state_history: Arc<RwLock<Vec<StateTransition>>>,
    created_at: DateTime<Utc>,
    updated_at: Arc<RwLock<DateTime<Utc>>>,
}

impl OrderStateMachine {
    pub fn new(order_id: OrderId) -> Self {
        Self {
            order_id,
            current_state: Arc::new(AtomicU8::new(OrderState::Created as u8)),
            state_history: Arc::new(RwLock::new(Vec::new())),
            created_at: Utc::now(),
            updated_at: Arc::new(RwLock::new(Utc::now())),
        }
    }
    
    /// Get current state
    pub fn current_state(&self) -> OrderState {
        let state_value = self.current_state.load(Ordering::SeqCst);
        OrderState::from_repr(state_value).unwrap_or(OrderState::Failed)
    }
    
    /// Attempt state transition
    pub fn transition_to(
        &self,
        new_state: OrderState,
        reason: Option<String>,
    ) -> Result<StateTransition, StateTransitionError> {
        let current = self.current_state();
        
        // Check if transition is valid
        if !current.can_transition_to(new_state) {
            return Err(StateTransitionError::InvalidTransition {
                from: current,
                to: new_state,
            });
        }
        
        // Check if already in terminal state
        if current.is_terminal() {
            return Err(StateTransitionError::AlreadyTerminal(current));
        }
        
        // Perform atomic transition
        let result = self.current_state.compare_exchange(
            current as u8,
            new_state as u8,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        
        match result {
            Ok(_) => {
                let transition = StateTransition {
                    order_id: self.order_id,
                    from_state: current,
                    to_state: new_state,
                    timestamp: Utc::now(),
                    reason,
                    metadata: None,
                };
                
                // Record transition in history
                self.state_history.write().push(transition.clone());
                *self.updated_at.write() = Utc::now();
                
                debug!(
                    "Order {} transitioned from {} to {}",
                    self.order_id, current, new_state
                );
                
                Ok(transition)
            }
            Err(actual) => {
                // Another thread changed the state
                let actual_state = OrderState::from_repr(actual).unwrap_or(OrderState::Failed);
                warn!(
                    "Order {} state changed by another thread: expected {}, got {}",
                    self.order_id, current, actual_state
                );
                Err(StateTransitionError::ConcurrentModification {
                    expected: current,
                    actual: actual_state,
                })
            }
        }
    }
    
    /// Force transition (use carefully, only for error recovery)
    pub fn force_transition(&self, new_state: OrderState, reason: String) {
        let current = self.current_state();
        self.current_state.store(new_state as u8, Ordering::SeqCst);
        
        let transition = StateTransition {
            order_id: self.order_id,
            from_state: current,
            to_state: new_state,
            timestamp: Utc::now(),
            reason: Some(format!("FORCED: {}", reason)),
            metadata: None,
        };
        
        self.state_history.write().push(transition);
        *self.updated_at.write() = Utc::now();
        
        error!(
            "Order {} force transitioned from {} to {}: {}",
            self.order_id, current, new_state, reason
        );
    }
    
    /// Check if order is in terminal state
    pub fn is_terminal(&self) -> bool {
        self.current_state().is_terminal()
    }
    
    /// Check if order is active
    pub fn is_active(&self) -> bool {
        self.current_state().is_active()
    }
    
    /// Get state history
    pub fn history(&self) -> Vec<StateTransition> {
        self.state_history.read().clone()
    }
    
    /// Get time in current state
    pub fn time_in_current_state(&self) -> chrono::Duration {
        let updated = *self.updated_at.read();
        Utc::now() - updated
    }
    
    /// Process order lifecycle events
    pub fn process_event(&self, event: OrderEvent) -> Result<StateTransition, StateTransitionError> {
        match event {
            OrderEvent::Validate => {
                self.transition_to(OrderState::Validated, Some("Order validated".to_string()))
            }
            OrderEvent::Submit => {
                self.transition_to(OrderState::Submitted, Some("Order submitted to exchange".to_string()))
            }
            OrderEvent::Acknowledge => {
                self.transition_to(OrderState::Acknowledged, Some("Order acknowledged by exchange".to_string()))
            }
            OrderEvent::PartialFill { quantity, price } => {
                self.transition_to(
                    OrderState::PartiallyFilled,
                    Some(format!("Partially filled: {} @ {}", quantity, price)),
                )
            }
            OrderEvent::Fill { quantity, price } => {
                self.transition_to(
                    OrderState::Filled,
                    Some(format!("Filled: {} @ {}", quantity, price)),
                )
            }
            OrderEvent::Cancel => {
                self.transition_to(OrderState::Cancelled, Some("Order cancelled".to_string()))
            }
            OrderEvent::Reject { reason } => {
                self.transition_to(OrderState::Rejected, Some(format!("Rejected: {}", reason)))
            }
            OrderEvent::Expire => {
                self.transition_to(OrderState::Expired, Some("Order expired".to_string()))
            }
            OrderEvent::Fail { error } => {
                self.transition_to(OrderState::Failed, Some(format!("Failed: {}", error)))
            }
        }
    }
}

/// Order lifecycle events
#[derive(Debug, Clone)]
pub enum OrderEvent {
    Validate,
    Submit,
    Acknowledge,
    PartialFill { quantity: rust_decimal::Decimal, price: rust_decimal::Decimal },
    Fill { quantity: rust_decimal::Decimal, price: rust_decimal::Decimal },
    Cancel,
    Reject { reason: String },
    Expire,
    Fail { error: String },
}

#[derive(Debug, Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from} to {to}")]
    InvalidTransition { from: OrderState, to: OrderState },
    
    #[error("Order already in terminal state: {0}")]
    AlreadyTerminal(OrderState),
    
    #[error("Concurrent modification: expected {expected}, actual {actual}")]
    ConcurrentModification {
        expected: OrderState,
        actual: OrderState,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_transitions() {
        let sm = OrderStateMachine::new(OrderId::new());
        
        // Created -> Validated
        assert!(sm.transition_to(OrderState::Validated, None).is_ok());
        assert_eq!(sm.current_state(), OrderState::Validated);
        
        // Validated -> Submitted
        assert!(sm.transition_to(OrderState::Submitted, None).is_ok());
        assert_eq!(sm.current_state(), OrderState::Submitted);
        
        // Submitted -> Acknowledged
        assert!(sm.transition_to(OrderState::Acknowledged, None).is_ok());
        assert_eq!(sm.current_state(), OrderState::Acknowledged);
        
        // Acknowledged -> Filled
        assert!(sm.transition_to(OrderState::Filled, None).is_ok());
        assert_eq!(sm.current_state(), OrderState::Filled);
        
        // Terminal state - no more transitions
        assert!(sm.transition_to(OrderState::Cancelled, None).is_err());
    }
    
    #[test]
    fn test_invalid_transitions() {
        let sm = OrderStateMachine::new(OrderId::new());
        
        // Can't go directly from Created to Filled
        assert!(sm.transition_to(OrderState::Filled, None).is_err());
        
        // Can't go backwards
        sm.transition_to(OrderState::Validated, None).unwrap();
        assert!(sm.transition_to(OrderState::Created, None).is_err());
    }
    
    #[test]
    fn test_state_history() {
        let sm = OrderStateMachine::new(OrderId::new());
        
        sm.transition_to(OrderState::Validated, Some("Test".to_string())).unwrap();
        sm.transition_to(OrderState::Submitted, None).unwrap();
        sm.transition_to(OrderState::Acknowledged, None).unwrap();
        
        let history = sm.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].from_state, OrderState::Created);
        assert_eq!(history[0].to_state, OrderState::Validated);
        assert_eq!(history[2].to_state, OrderState::Acknowledged);
    }
    
    #[test]
    fn test_process_events() {
        let sm = OrderStateMachine::new(OrderId::new());
        
        assert!(sm.process_event(OrderEvent::Validate).is_ok());
        assert!(sm.process_event(OrderEvent::Submit).is_ok());
        assert!(sm.process_event(OrderEvent::Acknowledge).is_ok());
        
        use rust_decimal_macros::dec;
        assert!(sm.process_event(OrderEvent::Fill {
            quantity: dec!(1.0),
            price: dec!(50000),
        }).is_ok());
        
        assert_eq!(sm.current_state(), OrderState::Filled);
        assert!(sm.is_terminal());
    }
}