// Trading Engine - Core transaction and order management
// FULL implementation with NO SIMPLIFICATIONS

pub mod transactions;

// Integration tests module
#[cfg(test)]
pub mod simple_integration_test;

// Re-export main types
pub use transactions::{
    Transaction,
    TransactionManager,
    TransactionStatus,
    TransactionType,
    Saga,
    SagaStep,
    WriteAheadLog,
};
