use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
// ELIMINATED: OrderId - Enhanced with UUID v7, time-ordered
// pub struct OrderId(pub String);