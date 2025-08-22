// Trading Engine - Core transaction and order management
// FULL implementation with NO SIMPLIFICATIONS

pub mod transactions;

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
