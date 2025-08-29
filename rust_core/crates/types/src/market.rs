use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Price(pub Decimal);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Quantity(pub Decimal);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Symbol(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Exchange(pub String);