// Order Management System for Bot4 Trading Platform
// Phase: 1.4 - Order Management
// Performance target: <100μs internal processing, <100ms total with exchange

pub mod order;
pub mod state_machine;
pub mod manager;
pub mod position;
pub mod router;

pub use order::{Order, OrderId, OrderType, OrderSide, TimeInForce};
pub use state_machine::{OrderState, OrderStateMachine, StateTransition};
pub use manager::{OrderManager, OrderManagerConfig};
pub use position::{Position, PositionManager, PnLCalculator};
pub use router::{OrderRouter, ExchangeRoute, RoutingStrategy};