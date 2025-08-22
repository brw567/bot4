use rust_decimal::Decimal;
use rust_decimal::prelude::*;

fn main() {
    let d = Decimal::from(42);
    // Try different methods
    println!("Methods available:");
    // to_f64() requires ToPrimitive trait
}
