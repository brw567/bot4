// Adapter: PostgreSQL Order Repository Implementation
// Implements the OrderRepository port for PostgreSQL persistence
// Owner: Avery | Reviewer: Sam
// Implements Repository Pattern following SOLID principles

use async_trait::async_trait;
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use std::sync::Arc;
use std::time::Duration;

use crate::domain::entities::{Order, OrderId, OrderStatus, OrderSide, OrderType};
use crate::domain::value_objects::{Symbol, Price, Quantity};
use crate::ports::outbound::repository_port::{Repository, OrderRepository, OrderStatistics};
use crate::dto::database::OrderDto;

/// PostgreSQL implementation of OrderRepository
/// Follows Hexagonal Architecture - this is an adapter
/// TODO: Add docs
pub struct PostgresOrderRepository {
    pool: Arc<PgPool>,
    table_name: String,
}

impl PostgresOrderRepository {
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self {
            pool,
            table_name: "orders".to_string(),
        }
    }
    
    /// Convert domain Order to database DTO
    fn to_dto(&self, order: &Order) -> OrderDto {
        OrderDto {
            id: order.id().to_string(),
            symbol: order.symbol().to_string(),
            side: order.side().to_string(),
            order_type: order.order_type().to_string(),
            quantity: order.quantity().value(),
            price: order.price().map(|p| p.value()),
            status: order.status().to_string(),
            client_order_id: order.client_order_id().cloned(),
            exchange_order_id: order.exchange_order_id().cloned(),
            filled_quantity: order.filled_quantity(),
            average_fill_price: order.average_fill_price(),
            created_at: order.created_at(),
            updated_at: order.updated_at(),
            metadata: serde_json::to_value(order.metadata()).ok(),
        }
    }
    
    /// Convert database DTO to domain Order
    fn from_dto(&self, dto: OrderDto) -> Result<Order> {
        let id = OrderId::parse(&dto.id)?;
        let symbol = Symbol::new(&dto.symbol)?;
        let side = OrderSide::from_str(&dto.side)?;
        let order_type = OrderType::from_str(&dto.order_type)?;
        let quantity = Quantity::new(dto.quantity)?;
        let price = dto.price.map(|p| Price::new(p)).transpose()?;
        let status = OrderStatus::from_str(&dto.status)?;
        
        // Reconstruct order using builder pattern
        let mut builder = Order::builder()
            .id(id)
            .symbol(symbol)
            .side(side)
            .order_type(order_type)
            .quantity(quantity)
            .status(status)
            .created_at(dto.created_at)
            .updated_at(dto.updated_at);
            
        if let Some(p) = price {
            builder = builder.price(p);
        }
        
        if let Some(cid) = dto.client_order_id {
            builder = builder.client_order_id(cid);
        }
        
        if let Some(eid) = dto.exchange_order_id {
            builder = builder.exchange_order_id(eid);
        }
        
        if dto.filled_quantity > 0.0 {
            builder = builder.filled_quantity(dto.filled_quantity);
        }
        
        if let Some(afp) = dto.average_fill_price {
            builder = builder.average_fill_price(afp);
        }
        
        Ok(builder.build()?)
    }
}

#[async_trait]
impl Repository<Order, OrderId> for PostgresOrderRepository {
    async fn save(&self, entity: &Order) -> Result<()> {
        let dto = self.to_dto(entity);
        
        let query = format!(
            r#"
            INSERT INTO {} (
                id, symbol, side, order_type, quantity, price, status,
                client_order_id, exchange_order_id, filled_quantity,
                average_fill_price, created_at, updated_at, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (id) DO NOTHING
            "#,
            self.table_name
        );
        
        sqlx::query(&query)
            .bind(&dto.id)
            .bind(&dto.symbol)
            .bind(&dto.side)
            .bind(&dto.order_type)
            .bind(dto.quantity)
            .bind(dto.price)
            .bind(&dto.status)
            .bind(&dto.client_order_id)
            .bind(&dto.exchange_order_id)
            .bind(dto.filled_quantity)
            .bind(dto.average_fill_price)
            .bind(dto.created_at)
            .bind(dto.updated_at)
            .bind(&dto.metadata)
            .execute(self.pool.as_ref())
            .await
            .context("Failed to save order to database")?;
            
        Ok(())
    }
    
    async fn find_by_id(&self, id: &OrderId) -> Result<Option<Order>> {
        let query = format!(
            "SELECT * FROM {} WHERE id = $1",
            self.table_name
        );
        
        let row = sqlx::query(&query)
            .bind(id.to_string())
            .fetch_optional(self.pool.as_ref())
            .await
            .context("Failed to find order by id")?;
            
        match row {
            Some(row) => {
                let dto = OrderDto::from_row(row)?;
                Ok(Some(self.from_dto(dto)?))
            },
            None => Ok(None),
        }
    }
    
    async fn find_all(&self) -> Result<Vec<Order>> {
        let query = format!(
            "SELECT * FROM {} ORDER BY created_at DESC",
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to fetch all orders")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn update(&self, entity: &Order) -> Result<()> {
        let dto = self.to_dto(entity);
        
        let query = format!(
            r#"
            UPDATE {}
            SET symbol = $2, side = $3, order_type = $4, quantity = $5,
                price = $6, status = $7, client_order_id = $8,
                exchange_order_id = $9, filled_quantity = $10,
                average_fill_price = $11, updated_at = $12, metadata = $13
            WHERE id = $1
            "#,
            self.table_name
        );
        
        let result = sqlx::query(&query)
            .bind(&dto.id)
            .bind(&dto.symbol)
            .bind(&dto.side)
            .bind(&dto.order_type)
            .bind(dto.quantity)
            .bind(dto.price)
            .bind(&dto.status)
            .bind(&dto.client_order_id)
            .bind(&dto.exchange_order_id)
            .bind(dto.filled_quantity)
            .bind(dto.average_fill_price)
            .bind(dto.updated_at)
            .bind(&dto.metadata)
            .execute(self.pool.as_ref())
            .await
            .context("Failed to update order")?;
            
        if result.rows_affected() == 0 {
            anyhow::bail!("Order not found for update");
        }
        
        Ok(())
    }
    
    async fn delete(&self, id: &OrderId) -> Result<()> {
        let query = format!(
            "DELETE FROM {} WHERE id = $1",
            self.table_name
        );
        
        let result = sqlx::query(&query)
            .bind(id.to_string())
            .execute(self.pool.as_ref())
            .await
            .context("Failed to delete order")?;
            
        if result.rows_affected() == 0 {
            anyhow::bail!("Order not found for deletion");
        }
        
        Ok(())
    }
    
    async fn exists(&self, id: &OrderId) -> Result<bool> {
        let query = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE id = $1)",
            self.table_name
        );
        
        let exists: bool = sqlx::query_scalar(&query)
            .bind(id.to_string())
            .fetch_one(self.pool.as_ref())
            .await
            .context("Failed to check order existence")?;
            
        Ok(exists)
    }
    
    async fn count(&self) -> Result<usize> {
        let query = format!(
            "SELECT COUNT(*) FROM {}",
            self.table_name
        );
        
        let count: i64 = sqlx::query_scalar(&query)
            .fetch_one(self.pool.as_ref())
            .await
            .context("Failed to count orders")?;
            
        Ok(count as usize)
    }
}

#[async_trait]
impl OrderRepository for PostgresOrderRepository {
    async fn find_by_status(&self, status: OrderStatus) -> Result<Vec<Order>> {
        let query = format!(
            "SELECT * FROM {} WHERE status = $1 ORDER BY created_at DESC",
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(status.to_string())
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find orders by status")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn find_by_symbol(&self, symbol: &Symbol) -> Result<Vec<Order>> {
        let query = format!(
            "SELECT * FROM {} WHERE symbol = $1 ORDER BY created_at DESC",
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(symbol.to_string())
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find orders by symbol")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn find_by_symbol_and_status(
        &self,
        symbol: &Symbol,
        status: OrderStatus,
    ) -> Result<Vec<Order>> {
        let query = format!(
            "SELECT * FROM {} WHERE symbol = $1 AND status = $2 ORDER BY created_at DESC",
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(symbol.to_string())
            .bind(status.to_string())
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find orders by symbol and status")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn find_active(&self) -> Result<Vec<Order>> {
        let query = format!(
            r#"
            SELECT * FROM {}
            WHERE status IN ('pending', 'open', 'partially_filled')
            ORDER BY created_at DESC
            "#,
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find active orders")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn find_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Order>> {
        let query = format!(
            r#"
            SELECT * FROM {}
            WHERE created_at >= $1 AND created_at <= $2
            ORDER BY created_at DESC
            "#,
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(start)
            .bind(end)
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find orders by date range")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn find_recent(&self, limit: usize) -> Result<Vec<Order>> {
        let query = format!(
            "SELECT * FROM {} ORDER BY created_at DESC LIMIT $1",
            self.table_name
        );
        
        let rows = sqlx::query(&query)
            .bind(limit as i64)
            .fetch_all(self.pool.as_ref())
            .await
            .context("Failed to find recent orders")?;
            
        let mut orders = Vec::new();
        for row in rows {
            let dto = OrderDto::from_row(row)?;
            orders.push(self.from_dto(dto)?);
        }
        
        Ok(orders)
    }
    
    async fn get_total_volume(&self) -> Result<f64> {
        let query = format!(
            r#"
            SELECT COALESCE(SUM(filled_quantity * average_fill_price), 0.0) as total_volume
            FROM {}
            WHERE status = 'filled'
            "#,
            self.table_name
        );
        
        let volume: f64 = sqlx::query_scalar(&query)
            .fetch_one(self.pool.as_ref())
            .await
            .context("Failed to get total volume")?;
            
        Ok(volume)
    }
    
    async fn get_statistics(&self) -> Result<OrderStatistics> {
        let query = format!(
            r#"
            SELECT 
                COUNT(*) as total_orders,
                COUNT(*) FILTER (WHERE status = 'filled') as filled_orders,
                COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_orders,
                COUNT(*) FILTER (WHERE status = 'rejected') as rejected_orders,
                COALESCE(SUM(filled_quantity * average_fill_price) FILTER (WHERE status = 'filled'), 0.0) as total_volume,
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) FILTER (WHERE status = 'filled') as avg_fill_seconds
            FROM {}
            "#,
            self.table_name
        );
        
        let row = sqlx::query(&query)
            .fetch_one(self.pool.as_ref())
            .await
            .context("Failed to get order statistics")?;
            
        let total_orders: i64 = row.get("total_orders");
        let filled_orders: i64 = row.get("filled_orders");
        let cancelled_orders: i64 = row.get("cancelled_orders");
        let rejected_orders: i64 = row.get("rejected_orders");
        let total_volume: f64 = row.get("total_volume");
        let avg_fill_seconds: Option<f64> = row.get("avg_fill_seconds");
        
        Ok(OrderStatistics {
            total_orders: total_orders as usize,
            filled_orders: filled_orders as usize,
            cancelled_orders: cancelled_orders as usize,
            rejected_orders: rejected_orders as usize,
            total_volume,
            average_fill_time: avg_fill_seconds.map(|s| Duration::from_secs_f64(s)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Integration tests would go here
    // Following our standards, we'd test against a real test database
    // not mocks
}