use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Price(pub Decimal);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantity(pub Decimal);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange(pub String);