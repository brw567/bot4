// Domain Events: Order-related events
// Events that occur during order lifecycle
// Owner: Sam | Reviewer: Alex

use chrono::{DateTime, Utc};
use crate::domain::entities::OrderId;
use crate::domain::value_objects::{Price, Quantity, Symbol};
use crate::domain::entities::OrderSide;

/// Domain events for order lifecycle
/// These events are emitted when order state changes
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderEvent {
    /// Order has been submitted for execution
    Submitted {
        order_id: OrderId,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        price: Option<Price>,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has been confirmed by the exchange
    Confirmed {
        order_id: OrderId,
        exchange_order_id: String,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has been filled (partially or completely)
    Filled {
        order_id: OrderId,
        filled_quantity: Quantity,
        fill_price: Price,
        remaining_quantity: Quantity,
        is_complete: bool,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has been cancelled
    Cancelled {
        order_id: OrderId,
        reason: String,
        remaining_quantity: Quantity,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has been rejected
    Rejected {
        order_id: OrderId,
        reason: String,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has expired
    Expired {
        order_id: OrderId,
        timestamp: DateTime<Utc>,
    },
    
    /// Order has been modified
    Modified {
        order_id: OrderId,
        old_price: Option<Price>,
        new_price: Option<Price>,
        old_quantity: Quantity,
        new_quantity: Quantity,
        timestamp: DateTime<Utc>,
    },
}

impl OrderEvent {
    /// Get the order ID associated with this event
    pub fn order_id(&self) -> &OrderId {
        match self {
            OrderEvent::Submitted { order_id, .. } => order_id,
            OrderEvent::Confirmed { order_id, .. } => order_id,
            OrderEvent::Filled { order_id, .. } => order_id,
            OrderEvent::Cancelled { order_id, .. } => order_id,
            OrderEvent::Rejected { order_id, .. } => order_id,
            OrderEvent::Expired { order_id, .. } => order_id,
            OrderEvent::Modified { order_id, .. } => order_id,
        }
    }
    
    /// Get the timestamp of this event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            OrderEvent::Submitted { timestamp, .. } => *timestamp,
            OrderEvent::Confirmed { timestamp, .. } => *timestamp,
            OrderEvent::Filled { timestamp, .. } => *timestamp,
            OrderEvent::Cancelled { timestamp, .. } => *timestamp,
            OrderEvent::Rejected { timestamp, .. } => *timestamp,
            OrderEvent::Expired { timestamp, .. } => *timestamp,
            OrderEvent::Modified { timestamp, .. } => *timestamp,
        }
    }
    
    /// Get the event type as a string
    pub fn event_type(&self) -> &str {
        match self {
            OrderEvent::Submitted { .. } => "Submitted",
            OrderEvent::Confirmed { .. } => "Confirmed",
            OrderEvent::Filled { .. } => "Filled",
            OrderEvent::Cancelled { .. } => "Cancelled",
            OrderEvent::Rejected { .. } => "Rejected",
            OrderEvent::Expired { .. } => "Expired",
            OrderEvent::Modified { .. } => "Modified",
        }
    }
    
    /// Check if this is a terminal event (ends order lifecycle)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderEvent::Filled { is_complete: true, .. } |
            OrderEvent::Cancelled { .. } |
            OrderEvent::Rejected { .. } |
            OrderEvent::Expired { .. }
        )
    }
}