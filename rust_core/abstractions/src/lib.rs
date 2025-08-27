//! # Cross-Layer Abstractions
//!
//! Provides trait abstractions to enable proper dependency inversion
//! between layers. Lower layers define traits that higher layers implement.
//!
//! ## Design Principles
//! - Dependency Inversion Principle (DIP) 
//! - Interface Segregation Principle (ISP)
//! - Abstractions should not depend on details
//! - Details should depend on abstractions
//!
//! ## External Research Applied
//! - "Dependency Injection Principles" (Mark Seemann)
//! - "Clean Code" (Robert C. Martin)
//! - "Inversion of Control Containers" (Martin Fowler)
//! - Rust trait objects for runtime polymorphism

#![warn(missing_docs)]

pub mod risk;
pub mod orders;
pub mod ml;
pub mod data;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use uuid::Uuid;

/// Result type for abstraction operations
pub type AbstractionResult<T> = Result<T, AbstractionError>;

/// Errors that can occur in abstraction layer
#[derive(Debug, thiserror::Error)]
pub enum AbstractionError {
    /// Layer violation detected
    #[error("Layer violation: {0}")]
    LayerViolation(String),
    
    /// Invalid dependency
    #[error("Invalid dependency: {0}")]
    InvalidDependency(String),
    
    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    /// Generic error
    #[error("Abstraction error: {0}")]
    Generic(String),
}

/// Base trait for all layer components
#[async_trait]
pub trait LayerComponent: Send + Sync {
    /// Get component identifier
    fn id(&self) -> Uuid;
    
    /// Get component layer number
    fn layer(&self) -> u8;
    
    /// Initialize component
    async fn initialize(&mut self) -> AbstractionResult<()>;
    
    /// Shutdown component
    async fn shutdown(&mut self) -> AbstractionResult<()>;
    
    /// Health check
    async fn health_check(&self) -> AbstractionResult<bool>;
}

/// Event bus abstraction for cross-layer communication
#[async_trait]
pub trait EventBus: Send + Sync {
    /// Event type
    type Event: Send + Sync;
    
    /// Publish event
    async fn publish(&self, event: Self::Event) -> AbstractionResult<()>;
    
    /// Subscribe to events
    async fn subscribe<F>(&self, handler: F) -> AbstractionResult<Uuid>
    where
        F: Fn(Self::Event) -> AbstractionResult<()> + Send + Sync + 'static;
    
    /// Unsubscribe from events
    async fn unsubscribe(&self, subscription_id: Uuid) -> AbstractionResult<()>;
}

/// Metrics collector abstraction
#[async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Record a metric value
    async fn record(&self, name: &str, value: f64, tags: Vec<(&str, &str)>);
    
    /// Increment a counter
    async fn increment(&self, name: &str, tags: Vec<(&str, &str)>);
    
    /// Record a timing
    async fn timing(&self, name: &str, duration_ms: u64, tags: Vec<(&str, &str)>);
}

/// Configuration provider abstraction
#[async_trait]
pub trait ConfigProvider: Send + Sync {
    /// Get configuration value
    async fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> AbstractionResult<T>;
    
    /// Set configuration value
    async fn set<T: Serialize>(&self, key: &str, value: T) -> AbstractionResult<()>;
    
    /// Watch for configuration changes
    async fn watch<F>(&self, key: &str, callback: F) -> AbstractionResult<Uuid>
    where
        F: Fn(serde_json::Value) -> AbstractionResult<()> + Send + Sync + 'static;
}

/// Circuit breaker abstraction
#[async_trait]
pub trait CircuitBreaker: Send + Sync {
    /// Check if circuit is open
    fn is_open(&self) -> bool;
    
    /// Record success
    fn record_success(&self);
    
    /// Record failure
    fn record_failure(&self);
    
    /// Get current state
    fn state(&self) -> CircuitState;
    
    /// Reset circuit
    fn reset(&self);
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (blocking calls)
    Open,
    /// Circuit is half-open (testing)
    HalfOpen,
}

/// Repository pattern for data access
#[async_trait]
pub trait Repository<T, ID>: Send + Sync 
where
    T: Send + Sync,
    ID: Send + Sync,
{
    /// Find by ID
    async fn find_by_id(&self, id: ID) -> AbstractionResult<Option<T>>;
    
    /// Find all
    async fn find_all(&self) -> AbstractionResult<Vec<T>>;
    
    /// Save entity
    async fn save(&self, entity: T) -> AbstractionResult<ID>;
    
    /// Update entity
    async fn update(&self, id: ID, entity: T) -> AbstractionResult<()>;
    
    /// Delete entity
    async fn delete(&self, id: ID) -> AbstractionResult<()>;
}

/// Unit of Work pattern for transactions
#[async_trait]
pub trait UnitOfWork: Send + Sync {
    /// Begin transaction
    async fn begin(&mut self) -> AbstractionResult<()>;
    
    /// Commit transaction
    async fn commit(&mut self) -> AbstractionResult<()>;
    
    /// Rollback transaction
    async fn rollback(&mut self) -> AbstractionResult<()>;
    
    /// Check if in transaction
    fn in_transaction(&self) -> bool;
}

/// Factory pattern for object creation
#[async_trait]
pub trait Factory<T>: Send + Sync 
where
    T: Send + Sync,
{
    /// Create new instance
    async fn create(&self) -> AbstractionResult<T>;
    
    /// Create with configuration
    async fn create_with_config(&self, config: serde_json::Value) -> AbstractionResult<T>;
}

/// Observer pattern for event notification
#[async_trait]
pub trait Observer<T>: Send + Sync 
where
    T: Send + Sync,
{
    /// Handle notification
    async fn notify(&self, event: T) -> AbstractionResult<()>;
}

/// Subject for observer pattern
#[async_trait]
pub trait Subject<T>: Send + Sync 
where
    T: Send + Sync + Clone,
{
    /// Attach observer
    async fn attach(&mut self, observer: Arc<dyn Observer<T>>) -> AbstractionResult<Uuid>;
    
    /// Detach observer
    async fn detach(&mut self, id: Uuid) -> AbstractionResult<()>;
    
    /// Notify all observers
    async fn notify_all(&self, event: T) -> AbstractionResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_abstraction_error() {
        let err = AbstractionError::LayerViolation("test".to_string());
        assert_eq!(err.to_string(), "Layer violation: test");
    }
    
    #[tokio::test]
    async fn test_circuit_state() {
        let state = CircuitState::Open;
        assert_eq!(state, CircuitState::Open);
    }
}