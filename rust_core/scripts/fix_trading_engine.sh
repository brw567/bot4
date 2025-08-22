#!/bin/bash

echo "Fixing trading_engine crate errors..."

# Fix fee_optimization.rs
sed -i 's/let (_fee_structure, tier)/let (fee_structure, tier)/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/self.fee_cache.insert(key.clone(), _fee_structure.clone())/self.fee_cache.insert(key.clone(), fee_structure.clone())/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/_fee_structure.maker_fee/_fee_structure.taker_fee/fee_structure.maker_fee * fee_structure.taker_fee/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/for (_exchange_name, fee_structure) in/for (exchange_name, fee_structure) in/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/for (_exchange, fee_structure) in/for (exchange, fee_structure) in/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/ExchangeScore { exchange: exchange.clone()/ExchangeScore { exchange: exchange.clone()/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/score: score }/score: score }/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/for (_ex, (total, individual))/for (ex, (total, individual))/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/exchange: ex.clone()/exchange: ex.clone()/g' crates/trading_engine/src/fee_optimization.rs
sed -i 's/total_cost: total/total_cost: total/g' crates/trading_engine/src/fee_optimization.rs

# Fix circuit_breaker.rs channel declarations
sed -i 's/let (_event_tx, event_rx)/let (event_tx, event_rx)/g' crates/trading_engine/src/circuit_breaker.rs

# Fix risk_engine.rs
sed -i 's/RiskValidationResult::Rejected(_position)/RiskValidationResult::Rejected(position)/g' crates/trading_engine/src/risk_engine.rs

echo "Trading engine fixes applied"