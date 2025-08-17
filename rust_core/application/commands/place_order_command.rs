// Command: Place Order
// Implements command pattern for order placement
// Owner: Sam | Reviewer: Quinn

use async_trait::async_trait;
use anyhow::{Result, Context};
use std::sync::Arc;
use chrono::Utc;

use crate::domain::entities::{Order, OrderId};
use crate::domain::events::OrderEvent;
use crate::ports::outbound::exchange_port::ExchangePort;
use crate::ports::outbound::repository_port::OrderRepository;

/// Command trait - all commands must implement this
#[async_trait]
pub trait Command: Send + Sync {
    type Output;
    
    /// Execute the command
    async fn execute(&self) -> Result<Self::Output>;
    
    /// Validate the command before execution
    async fn validate(&self) -> Result<()> {
        Ok(())
    }
    
    /// Compensate (undo) the command if needed
    async fn compensate(&self) -> Result<()> {
        Ok(())
    }
}

/// Place order command - encapsulates order placement logic
pub struct PlaceOrderCommand {
    order: Order,
    exchange: Arc<dyn ExchangePort>,
    repository: Arc<dyn OrderRepository>,
    risk_checker: Arc<dyn RiskChecker>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl PlaceOrderCommand {
    pub fn new(
        order: Order,
        exchange: Arc<dyn ExchangePort>,
        repository: Arc<dyn OrderRepository>,
        risk_checker: Arc<dyn RiskChecker>,
        event_publisher: Arc<dyn EventPublisher>,
    ) -> Self {
        Self {
            order,
            exchange,
            repository,
            risk_checker,
            event_publisher,
        }
    }
}

#[async_trait]
impl Command for PlaceOrderCommand {
    type Output = (OrderId, Option<String>); // (order_id, exchange_order_id)
    
    async fn validate(&self) -> Result<()> {
        // Validate order state
        if !self.order.status().is_draft() {
            anyhow::bail!("Can only place draft orders");
        }
        
        // Check risk limits
        self.risk_checker.check_order(&self.order)
            .await
            .context("Risk check failed")?;
        
        // Check sufficient balance
        let balances = self.exchange.get_balances()
            .await
            .context("Failed to get balances")?;
        
        let required_asset = match self.order.side() {
            crate::domain::entities::OrderSide::Buy => self.order.symbol().quote(),
            crate::domain::entities::OrderSide::Sell => self.order.symbol().base(),
        };
        
        let required_amount = self.order.quantity().value() * 
            self.order.price().map(|p| p.value()).unwrap_or(50000.0); // Use market price for market orders
        
        if let Some(balance) = balances.get(required_asset) {
            if balance.free.value() < required_amount {
                anyhow::bail!(
                    "Insufficient balance: required {}, available {}", 
                    required_amount, 
                    balance.free.value()
                );
            }
        } else {
            anyhow::bail!("No balance for asset {}", required_asset);
        }
        
        Ok(())
    }
    
    async fn execute(&self) -> Result<Self::Output> {
        // Step 1: Validate
        self.validate().await?;
        
        // Step 2: Submit order (domain event)
        let (order, submit_event) = self.order.clone().submit()
            .context("Failed to submit order")?;
        
        // Step 3: Save to repository (pending state)
        self.repository.save(&order)
            .await
            .context("Failed to save order")?;
        
        // Step 4: Publish submit event
        self.event_publisher.publish(submit_event.clone())
            .await
            .context("Failed to publish submit event")?;
        
        // Step 5: Place on exchange
        let exchange_order_id = self.exchange.place_order(&order)
            .await
            .context("Failed to place order on exchange")?;
        
        // Step 6: Confirm order (domain event)
        let (confirmed_order, confirm_event) = order.confirm(exchange_order_id.clone())
            .context("Failed to confirm order")?;
        
        // Step 7: Update repository
        self.repository.update(&confirmed_order)
            .await
            .context("Failed to update order")?;
        
        // Step 8: Publish confirm event
        self.event_publisher.publish(confirm_event)
            .await
            .context("Failed to publish confirm event")?;
        
        Ok((confirmed_order.id().clone(), Some(exchange_order_id)))
    }
    
    async fn compensate(&self) -> Result<()> {
        // If order was placed, try to cancel it
        if let Ok(status) = self.exchange.get_order_status(self.order.id()).await {
            if status.is_active() {
                self.exchange.cancel_order(self.order.id())
                    .await
                    .context("Failed to cancel order during compensation")?;
            }
        }
        
        // Mark order as cancelled in repository
        if let Ok(Some(mut order)) = self.repository.find_by_id(self.order.id()).await {
            let (cancelled_order, cancel_event) = order.cancel("Command compensation".to_string())
                .context("Failed to cancel order")?;
            
            self.repository.update(&cancelled_order)
                .await
                .context("Failed to update cancelled order")?;
            
            self.event_publisher.publish(cancel_event)
                .await
                .context("Failed to publish cancel event")?;
        }
        
        Ok(())
    }
}

/// Cancel order command
pub struct CancelOrderCommand {
    order_id: OrderId,
    reason: String,
    exchange: Arc<dyn ExchangePort>,
    repository: Arc<dyn OrderRepository>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl CancelOrderCommand {
    pub fn new(
        order_id: OrderId,
        reason: String,
        exchange: Arc<dyn ExchangePort>,
        repository: Arc<dyn OrderRepository>,
        event_publisher: Arc<dyn EventPublisher>,
    ) -> Self {
        Self {
            order_id,
            reason,
            exchange,
            repository,
            event_publisher,
        }
    }
}

#[async_trait]
impl Command for CancelOrderCommand {
    type Output = ();
    
    async fn validate(&self) -> Result<()> {
        // Check order exists
        let order = self.repository.find_by_id(&self.order_id)
            .await
            .context("Failed to find order")?
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        
        // Check if cancellable
        if !order.can_cancel() {
            anyhow::bail!("Order cannot be cancelled in status {:?}", order.status());
        }
        
        Ok(())
    }
    
    async fn execute(&self) -> Result<Self::Output> {
        // Validate
        self.validate().await?;
        
        // Get order from repository
        let order = self.repository.find_by_id(&self.order_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        
        // Cancel on exchange first
        self.exchange.cancel_order(&self.order_id)
            .await
            .context("Failed to cancel on exchange")?;
        
        // Update domain model
        let (cancelled_order, cancel_event) = order.cancel(self.reason.clone())
            .context("Failed to cancel order")?;
        
        // Update repository
        self.repository.update(&cancelled_order)
            .await
            .context("Failed to update order")?;
        
        // Publish event
        self.event_publisher.publish(cancel_event)
            .await
            .context("Failed to publish cancel event")?;
        
        Ok(())
    }
}

/// Batch order command - place multiple orders
pub struct BatchOrderCommand {
    orders: Vec<Order>,
    stop_on_error: bool,
    exchange: Arc<dyn ExchangePort>,
    repository: Arc<dyn OrderRepository>,
    risk_checker: Arc<dyn RiskChecker>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl BatchOrderCommand {
    pub fn new(
        orders: Vec<Order>,
        stop_on_error: bool,
        exchange: Arc<dyn ExchangePort>,
        repository: Arc<dyn OrderRepository>,
        risk_checker: Arc<dyn RiskChecker>,
        event_publisher: Arc<dyn EventPublisher>,
    ) -> Self {
        Self {
            orders,
            stop_on_error,
            exchange,
            repository,
            risk_checker,
            event_publisher,
        }
    }
}

#[async_trait]
impl Command for BatchOrderCommand {
    type Output = Vec<Result<(OrderId, Option<String>)>>;
    
    async fn execute(&self) -> Result<Self::Output> {
        let mut results = Vec::new();
        
        for order in &self.orders {
            let command = PlaceOrderCommand::new(
                order.clone(),
                self.exchange.clone(),
                self.repository.clone(),
                self.risk_checker.clone(),
                self.event_publisher.clone(),
            );
            
            let result = command.execute().await;
            
            if result.is_err() && self.stop_on_error {
                // Compensate successful orders if stop_on_error is true
                for success_result in &results {
                    if let Ok((order_id, _)) = success_result {
                        let cancel_command = CancelOrderCommand::new(
                            order_id.clone(),
                            "Batch cancelled due to error".to_string(),
                            self.exchange.clone(),
                            self.repository.clone(),
                            self.event_publisher.clone(),
                        );
                        
                        let _ = cancel_command.execute().await;
                    }
                }
                
                results.push(result);
                break;
            }
            
            results.push(result);
        }
        
        Ok(results)
    }
}

// Trait definitions for dependencies

/// Risk checker trait
#[async_trait]
pub trait RiskChecker: Send + Sync {
    async fn check_order(&self, order: &Order) -> Result<()>;
}

/// Event publisher trait
#[async_trait]
pub trait EventPublisher: Send + Sync {
    async fn publish(&self, event: OrderEvent) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock implementations would go here for testing
    // Following our standards, we'd create proper test doubles
    // not simple mocks
}