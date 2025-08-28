//! Smart order routing across multiple exchanges

use anyhow::{Result, bail};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct RoutingResult {
    pub splits: Vec<OrderSplit>,
    pub estimated_average_price: Decimal,
    pub estimated_slippage: Decimal,
    pub best_exchange: String,
    pub execution_plan: Vec<ExecutionStep>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderSplit {
    pub exchange: String,
    pub quantity: Decimal,
    pub percentage: Decimal,
    pub estimated_price: Decimal,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step: u32,
    pub exchange: String,
    pub action: String,
    pub quantity: Decimal,
    pub timing: String,
}

#[derive(Debug)]
struct ExchangeLiquidity {
    exchange: String,
    available_quantity: Decimal,
    price_levels: Vec<(Decimal, Decimal)>, // (price, quantity)
    spread: Decimal,
    fee_rate: Decimal,
}

pub struct SmartRouter;

impl SmartRouter {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn route_order(&self, symbol: &str, side: &str, quantity: Decimal, 
                             slippage_tolerance: Decimal) -> Result<RoutingResult> {
        // Get liquidity from all exchanges
        let liquidity = self.get_aggregated_liquidity(symbol, side).await?;
        
        // Calculate optimal routing
        let splits = self.calculate_optimal_splits(&liquidity, quantity, slippage_tolerance)?;
        
        // Calculate estimated execution price
        let estimated_price = self.calculate_weighted_price(&splits);
        
        // Calculate estimated slippage
        let mid_price = self.get_mid_price(&liquidity);
        let estimated_slippage = ((estimated_price - mid_price) / mid_price).abs();
        
        // Determine best single exchange
        let best_exchange = self.find_best_exchange(&liquidity, quantity);
        
        // Create execution plan
        let execution_plan = self.create_execution_plan(&splits, side);
        
        Ok(RoutingResult {
            splits,
            estimated_average_price: estimated_price,
            estimated_slippage,
            best_exchange,
            execution_plan,
        })
    }
    
    async fn get_aggregated_liquidity(&self, symbol: &str, side: &str) -> Result<Vec<ExchangeLiquidity>> {
        // In production, would fetch real orderbook data from exchanges
        // Simulating liquidity data for demonstration
        
        let mut liquidity = Vec::new();
        
        // Binance liquidity (typically highest)
        liquidity.push(ExchangeLiquidity {
            exchange: "binance".to_string(),
            available_quantity: dec!(100),
            price_levels: vec![
                (dec!(50000), dec!(5)),
                (dec!(50010), dec!(10)),
                (dec!(50020), dec!(15)),
                (dec!(50030), dec!(20)),
                (dec!(50040), dec!(30)),
            ],
            spread: dec!(0.0001),
            fee_rate: dec!(0.001),
        });
        
        // Kraken liquidity
        liquidity.push(ExchangeLiquidity {
            exchange: "kraken".to_string(),
            available_quantity: dec!(50),
            price_levels: vec![
                (dec!(50005), dec!(3)),
                (dec!(50015), dec!(7)),
                (dec!(50025), dec!(10)),
                (dec!(50035), dec!(15)),
                (dec!(50045), dec!(15)),
            ],
            spread: dec!(0.0002),
            fee_rate: dec!(0.0026),
        });
        
        // Coinbase liquidity
        liquidity.push(ExchangeLiquidity {
            exchange: "coinbase".to_string(),
            available_quantity: dec!(40),
            price_levels: vec![
                (dec!(50002), dec!(2)),
                (dec!(50012), dec!(5)),
                (dec!(50022), dec!(8)),
                (dec!(50032), dec!(12)),
                (dec!(50042), dec!(13)),
            ],
            spread: dec!(0.00015),
            fee_rate: dec!(0.006),
        });
        
        Ok(liquidity)
    }
    
    fn calculate_optimal_splits(&self, liquidity: &[ExchangeLiquidity], 
                                total_quantity: Decimal, 
                                slippage_tolerance: Decimal) -> Result<Vec<OrderSplit>> {
        let mut splits = Vec::new();
        let mut remaining = total_quantity;
        
        // Sort exchanges by best execution price
        let mut sorted_exchanges: Vec<_> = liquidity.iter()
            .map(|ex| {
                let avg_price = self.calculate_average_price(&ex.price_levels, total_quantity);
                let total_cost = avg_price * (dec!(1) + ex.fee_rate);
                (ex, total_cost)
            })
            .collect();
        
        sorted_exchanges.sort_by_key(|(_ex, cost)| (*cost * dec!(10000)).to_i64_with_trounding(0).unwrap_or(0));
        
        // Allocate orders to exchanges
        for (exchange_liq, _cost) in sorted_exchanges {
            if remaining <= dec!(0) {
                break;
            }
            
            // Calculate how much we can fill on this exchange
            let available = exchange_liq.available_quantity.min(remaining);
            
            if available > dec!(0) {
                // Calculate execution price for this quantity
                let exec_price = self.calculate_average_price(&exchange_liq.price_levels, available);
                
                // Check if slippage is acceptable
                let base_price = exchange_liq.price_levels[0].0;
                let slippage = ((exec_price - base_price) / base_price).abs();
                
                if slippage <= slippage_tolerance {
                    let percentage = (available / total_quantity) * dec!(100);
                    
                    splits.push(OrderSplit {
                        exchange: exchange_liq.exchange.clone(),
                        quantity: available,
                        percentage,
                        estimated_price: exec_price,
                        reason: self.get_routing_reason(&exchange_liq.exchange, slippage),
                    });
                    
                    remaining -= available;
                }
            }
        }
        
        // If we couldn't route all quantity, add warning
        if remaining > dec!(0) {
            warn!("Could not route full quantity. Remaining: {}", remaining);
        }
        
        Ok(splits)
    }
    
    fn calculate_average_price(&self, price_levels: &[(Decimal, Decimal)], quantity: Decimal) -> Decimal {
        let mut total_cost = dec!(0);
        let mut filled = dec!(0);
        
        for (price, available) in price_levels {
            let fill_qty = available.min(quantity - filled);
            total_cost += price * fill_qty;
            filled += fill_qty;
            
            if filled >= quantity {
                break;
            }
        }
        
        if filled > dec!(0) {
            total_cost / filled
        } else {
            price_levels.first().map(|(p, _)| *p).unwrap_or(dec!(0))
        }
    }
    
    fn calculate_weighted_price(&self, splits: &[OrderSplit]) -> Decimal {
        let mut total_cost = dec!(0);
        let mut total_quantity = dec!(0);
        
        for split in splits {
            total_cost += split.estimated_price * split.quantity;
            total_quantity += split.quantity;
        }
        
        if total_quantity > dec!(0) {
            total_cost / total_quantity
        } else {
            dec!(0)
        }
    }
    
    fn get_mid_price(&self, liquidity: &[ExchangeLiquidity]) -> Decimal {
        // Calculate volume-weighted mid price across exchanges
        let mut weighted_price = dec!(0);
        let mut total_weight = dec!(0);
        
        for exchange_liq in liquidity {
            if let Some((best_price, _)) = exchange_liq.price_levels.first() {
                let weight = exchange_liq.available_quantity;
                weighted_price += best_price * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > dec!(0) {
            weighted_price / total_weight
        } else {
            dec!(50000) // Default fallback
        }
    }
    
    fn find_best_exchange(&self, liquidity: &[ExchangeLiquidity], quantity: Decimal) -> String {
        let mut best_exchange = String::new();
        let mut best_cost = Decimal::MAX;
        
        for exchange_liq in liquidity {
            if exchange_liq.available_quantity >= quantity {
                let avg_price = self.calculate_average_price(&exchange_liq.price_levels, quantity);
                let total_cost = avg_price * (dec!(1) + exchange_liq.fee_rate);
                
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_exchange = exchange_liq.exchange.clone();
                }
            }
        }
        
        if best_exchange.is_empty() {
            // If no single exchange can handle full order, pick the one with most liquidity
            liquidity.iter()
                .max_by_key(|ex| (ex.available_quantity * dec!(1000)).to_i64_with_trounding(0).unwrap_or(0))
                .map(|ex| ex.exchange.clone())
                .unwrap_or_else(|| "binance".to_string())
        } else {
            best_exchange
        }
    }
    
    fn create_execution_plan(&self, splits: &[OrderSplit], side: &str) -> Vec<ExecutionStep> {
        let mut plan = Vec::new();
        let mut step = 1;
        
        // Phase 1: Place passive orders on exchanges with better prices
        for split in splits.iter().filter(|s| s.reason.contains("price")) {
            plan.push(ExecutionStep {
                step,
                exchange: split.exchange.clone(),
                action: format!("Place {} limit order", side),
                quantity: split.quantity,
                timing: "Immediate".to_string(),
            });
            step += 1;
        }
        
        // Phase 2: Place aggressive orders on liquid exchanges
        for split in splits.iter().filter(|s| s.reason.contains("liquidity")) {
            plan.push(ExecutionStep {
                step,
                exchange: split.exchange.clone(),
                action: format!("Place {} market order", side),
                quantity: split.quantity,
                timing: "After 100ms".to_string(),
            });
            step += 1;
        }
        
        // Phase 3: Fill remaining on any exchange
        for split in splits.iter().filter(|s| !s.reason.contains("price") && !s.reason.contains("liquidity")) {
            plan.push(ExecutionStep {
                step,
                exchange: split.exchange.clone(),
                action: format!("Place {} order", side),
                quantity: split.quantity,
                timing: "After 200ms".to_string(),
            });
            step += 1;
        }
        
        plan
    }
    
    fn get_routing_reason(&self, exchange: &str, slippage: Decimal) -> String {
        if slippage < dec!(0.0001) {
            format!("Best price on {}", exchange)
        } else if slippage < dec!(0.0005) {
            format!("Good liquidity on {}", exchange)
        } else {
            format!("Available liquidity on {}", exchange)
        }
    }
    
    pub async fn estimate_impact(&self, symbol: &str, side: &str, quantity: Decimal) -> Result<Decimal> {
        // Estimate market impact of large order
        let liquidity = self.get_aggregated_liquidity(symbol, side).await?;
        
        let total_available: Decimal = liquidity.iter()
            .map(|ex| ex.available_quantity)
            .sum();
        
        // Simple impact model: impact increases with size relative to available liquidity
        let size_ratio = quantity / total_available.max(dec!(1));
        let impact = size_ratio * dec!(0.01); // 1% impact for order equal to total liquidity
        
        Ok(impact)
    }
    
    pub async fn find_arbitrage(&self, symbol: &str) -> Result<Option<ArbitrageOpportunity>> {
        let buy_liquidity = self.get_aggregated_liquidity(symbol, "buy").await?;
        let sell_liquidity = self.get_aggregated_liquidity(symbol, "sell").await?;
        
        // Find best bid and ask across exchanges
        let mut best_bid = dec!(0);
        let mut best_bid_exchange = String::new();
        let mut best_ask = Decimal::MAX;
        let mut best_ask_exchange = String::new();
        
        for exchange in &sell_liquidity {
            if let Some((price, _)) = exchange.price_levels.first() {
                if *price > best_bid {
                    best_bid = *price;
                    best_bid_exchange = exchange.exchange.clone();
                }
            }
        }
        
        for exchange in &buy_liquidity {
            if let Some((price, _)) = exchange.price_levels.first() {
                if *price < best_ask {
                    best_ask = *price;
                    best_ask_exchange = exchange.exchange.clone();
                }
            }
        }
        
        // Check for arbitrage opportunity
        if best_bid > best_ask {
            let spread = best_bid - best_ask;
            let spread_percentage = (spread / best_ask) * dec!(100);
            
            // Account for fees
            let buy_fee = buy_liquidity.iter()
                .find(|ex| ex.exchange == best_ask_exchange)
                .map(|ex| ex.fee_rate)
                .unwrap_or(dec!(0.001));
            
            let sell_fee = sell_liquidity.iter()
                .find(|ex| ex.exchange == best_bid_exchange)
                .map(|ex| ex.fee_rate)
                .unwrap_or(dec!(0.001));
            
            let net_profit = spread_percentage - (buy_fee + sell_fee) * dec!(100);
            
            if net_profit > dec!(0) {
                return Ok(Some(ArbitrageOpportunity {
                    buy_exchange: best_ask_exchange,
                    sell_exchange: best_bid_exchange,
                    buy_price: best_ask,
                    sell_price: best_bid,
                    spread: spread_percentage,
                    net_profit_percentage: net_profit,
                }));
            }
        }
        
        Ok(None)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub buy_price: Decimal,
    pub sell_price: Decimal,
    pub spread: Decimal,
    pub net_profit_percentage: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_smart_routing() {
        let router = SmartRouter::new();
        
        let result = router.route_order(
            "BTC/USDT",
            "buy",
            dec!(10),
            dec!(0.001)
        ).await.unwrap();
        
        assert!(!result.splits.is_empty());
        assert!(result.estimated_average_price > dec!(0));
        assert!(result.estimated_slippage >= dec!(0));
        assert!(!result.best_exchange.is_empty());
    }
    
    #[tokio::test]
    async fn test_market_impact() {
        let router = SmartRouter::new();
        
        let small_impact = router.estimate_impact("BTC/USDT", "buy", dec!(1)).await.unwrap();
        let large_impact = router.estimate_impact("BTC/USDT", "buy", dec!(100)).await.unwrap();
        
        assert!(large_impact > small_impact);
    }
}