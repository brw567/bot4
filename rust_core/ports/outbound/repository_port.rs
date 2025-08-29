// Port: Repository Interface
// Defines the contract for data persistence
// Owner: Avery | Reviewer: Sam

use async_trait::async_trait;
use anyhow::Result;
use chrono::{DateTime, Utc};

use crate::domain::entities::{Order, OrderId, OrderStatus};
use crate::domain::value_objects::Symbol;

/// Generic repository trait for any entity
#[async_trait]
pub trait Repository<T, ID>: Send + Sync 
where
    T: Send + Sync,
    ID: Send + Sync,
{
    /// Save an entity
    async fn save(&self, entity: &T) -> Result<()>;
    
    /// Find an entity by ID
    async fn find_by_id(&self, id: &ID) -> Result<Option<T>>;
    
    /// Find all entities
    async fn find_all(&self) -> Result<Vec<T>>;
    
    /// Update an entity
    async fn update(&self, entity: &T) -> Result<()>;
    
    /// Delete an entity by ID
    async fn delete(&self, id: &ID) -> Result<()>;
    
    /// Check if entity exists
    async fn exists(&self, id: &ID) -> Result<bool>;
    
    /// Count all entities
    async fn count(&self) -> Result<usize>;
}

/// Specific repository for Orders with additional query methods
#[async_trait]
pub trait OrderRepository: Repository<Order, OrderId> {
    /// Find orders by status
    async fn find_by_status(&self, status: OrderStatus) -> Result<Vec<Order>>;
    
    /// Find orders by symbol
    async fn find_by_symbol(&self, symbol: &Symbol) -> Result<Vec<Order>>;
    
    /// Find orders by symbol and status
    async fn find_by_symbol_and_status(
        &self, 
        symbol: &Symbol, 
        status: OrderStatus
    ) -> Result<Vec<Order>>;
    
    /// Find active orders (pending, open, partially filled)
    async fn find_active(&self) -> Result<Vec<Order>>;
    
    /// Find orders created within a time range
    async fn find_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Order>>;
    
    /// Find the most recent orders
    async fn find_recent(&self, limit: usize) -> Result<Vec<Order>>;
    
    /// Get total volume traded
    async fn get_total_volume(&self) -> Result<f64>;
    
    /// Get order statistics
    async fn get_statistics(&self) -> Result<OrderStatistics>;
}

/// Order statistics
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct OrderStatistics {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
    pub total_volume: f64,
    pub average_fill_time: Option<Duration>,
}

use std::time::Duration;

/// Unit of Work pattern for transactional operations
#[async_trait]
pub trait UnitOfWork: Send + Sync {
    /// Begin a transaction
    async fn begin(&mut self) -> Result<()>;
    
    /// Commit the transaction
    async fn commit(&mut self) -> Result<()>;
    
    /// Rollback the transaction
    async fn rollback(&mut self) -> Result<()>;
    
    /// Get order repository within this transaction
    fn orders(&self) -> &dyn OrderRepository;
}