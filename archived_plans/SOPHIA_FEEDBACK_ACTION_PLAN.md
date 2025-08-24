# Sophia's Phase 2 Feedback - Action Plan
## Exchange Simulator Improvements

---

## Executive Summary

Sophia gave us **93/100** - CONDITIONAL PASS! Her feedback identifies critical production readiness gaps that we'll address immediately. Top priorities: idempotency, OCO correctness, and fee models.

---

## 🔴 Critical Issues to Fix (Week 1)

### 1. Idempotency & Deduplication (CRITICAL) ✅ COMPLETE
**Issue**: No client_order_id deduplication can cause double orders
**Solution**: IMPLEMENTED in `/home/hamster/bot4/rust_core/adapters/outbound/exchanges/idempotency_manager.rs`
```rust
pub struct IdempotencyManager {
    // Track client_order_id -> exchange_order_id mapping
    order_cache: DashMap<String, String>,
    // Expiry after 24 hours
    ttl: Duration,
}

impl ExchangeSimulator {
    pub async fn place_order_idempotent(
        &self,
        order: &Order,
        client_order_id: String,
    ) -> Result<String> {
        // Check for duplicate
        if let Some(existing) = self.idempotency_mgr.get(&client_order_id) {
            return Ok(existing.clone()); // Return existing order_id
        }
        
        // Place new order
        let exchange_id = self.place_order_internal(order).await?;
        
        // Store mapping
        self.idempotency_mgr.insert(client_order_id, exchange_id.clone());
        
        Ok(exchange_id)
    }
}
```

### 2. OCO Edge Case Semantics (HIGH) ✅ COMPLETE
**Issue**: Incorrect OCO cancellation logic
**Solution**: IMPLEMENTED in `/home/hamster/bot4/rust_core/domain/entities/oco_order.rs`
```rust
pub struct OcoOrder {
    limit_order: Order,
    stop_order: Order,
    // Rules for execution
    trigger_cancels_sibling: bool,
    partial_fill_cancels_sibling: bool,
    priority: OcoPriority,
}

pub enum OcoPriority {
    LimitFirst,    // If both trigger, execute limit
    StopFirst,     // If both trigger, execute stop
    Timestamp,     // First triggered wins
}

impl OcoExecutor {
    pub async fn handle_trigger(&mut self, triggered_leg: &Order) {
        // Cancel sibling immediately on trigger
        self.cancel_sibling(triggered_leg).await?;
        
        // Handle partial fill scenario
        if triggered_leg.is_partially_filled() {
            self.handle_partial_oco(triggered_leg).await?;
        }
    }
}
```

### 3. Fee Model Implementation (HIGH) ✅ COMPLETE
**Issue**: No fees applied to fills
**Solution**: IMPLEMENTED in `/home/hamster/bot4/rust_core/domain/value_objects/fee.rs`
```rust
pub struct FeeModel {
    maker_fee_bps: i32,  // Can be negative (rebate)
    taker_fee_bps: i32,
    min_fee: Option<f64>,
}

impl FeeCalculator {
    pub fn calculate_fee(&self, fill: &Fill) -> Fee {
        let is_maker = fill.is_post_only || !fill.crossed_spread;
        
        let fee_bps = if is_maker {
            self.maker_fee_bps
        } else {
            self.taker_fee_bps
        };
        
        let fee_amount = fill.quantity * fill.price * (fee_bps as f64 / 10000.0);
        
        Fee {
            amount: fee_amount.max(self.min_fee.unwrap_or(0.0)),
            currency: fill.quote_currency.clone(),
            is_rebate: fee_bps < 0,
        }
    }
}
```

### 4. Timestamp Skew & Server Time (HIGH)
**Issue**: No timestamp validation
**Solution**:
```rust
pub struct TimeValidator {
    server_time: AtomicU64,
    max_recv_window: Duration,  // Default 5000ms
    max_clock_drift: Duration,  // Default 1000ms
}

impl TimeValidator {
    pub fn validate_request(&self, client_timestamp: u64) -> Result<()> {
        let server_time = self.server_time.load(Ordering::Relaxed);
        let drift = (server_time as i64 - client_timestamp as i64).abs();
        
        if drift > self.max_clock_drift.as_millis() as i64 {
            return Err(ExchangeError::TimestampOutOfRange {
                code: -1021,
                server_time,
                client_time: client_timestamp,
            });
        }
        
        Ok(())
    }
}
```

### 5. Validation Filters (HIGH)
**Issue**: Missing exchange-specific filters
**Solution**:
```rust
pub struct ValidationFilters {
    price_filter: PriceFilter {
        min_price: f64,
        max_price: f64,
        tick_size: f64,
    },
    lot_size: LotSizeFilter {
        min_qty: f64,
        max_qty: f64,
        step_size: f64,
    },
    min_notional: f64,
    max_orders: usize,
}

impl OrderValidator {
    pub fn validate(&self, order: &Order) -> Result<()> {
        // Price filter
        if order.price % self.price_filter.tick_size != 0.0 {
            return Err(ExchangeError::PriceFilter { code: -1013 });
        }
        
        // Lot size
        if order.quantity % self.lot_size.step_size != 0.0 {
            return Err(ExchangeError::LotSize { code: -1013 });
        }
        
        // Min notional
        if order.price * order.quantity < self.min_notional {
            return Err(ExchangeError::MinNotional { code: -1013 });
        }
        
        // Post-only crossing check
        if order.is_post_only && self.would_cross_spread(order) {
            return Err(ExchangeError::PostOnlyWouldCross { code: -2010 });
        }
        
        Ok(())
    }
}
```

---

## 🟡 Medium Priority Improvements (Week 2)

### 6. Per-Symbol Actor Loops (Determinism)
**Issue**: Shared RwLock causes contention and non-determinism
**Solution**:
```rust
pub struct SymbolActor {
    symbol: Symbol,
    order_book: OrderBook,
    orders: HashMap<OrderId, Order>,
    receiver: mpsc::Receiver<SymbolCommand>,
}

impl SymbolActor {
    pub async fn run(mut self) {
        // Single-threaded event loop per symbol
        while let Some(cmd) = self.receiver.recv().await {
            match cmd {
                SymbolCommand::PlaceOrder(order) => {
                    self.process_order(order).await;
                }
                SymbolCommand::CancelOrder(id) => {
                    self.cancel_order(id).await;
                }
                // Deterministic processing - no races
            }
        }
    }
}

pub struct ExchangeSimulator {
    // One actor per symbol
    symbol_actors: HashMap<Symbol, mpsc::Sender<SymbolCommand>>,
}
```

### 7. Property-Based Testing
```rust
#[proptest]
fn test_time_in_force_semantics(
    order: Order,
    tif: TimeInForce,
    market_state: MarketState,
) {
    // Property: IOC orders never remain open
    if tif == TimeInForce::IOC {
        assert!(!order.is_open_after_execution());
    }
    
    // Property: FOK is all or nothing
    if tif == TimeInForce::FOK {
        assert!(order.is_fully_filled() || order.is_rejected());
    }
}

#[proptest]
fn test_idempotency(
    order: Order,
    client_order_id: String,
) {
    // Property: Same client_order_id always returns same exchange_order_id
    let result1 = simulator.place_order(&order, &client_order_id);
    let result2 = simulator.place_order(&order, &client_order_id);
    assert_eq!(result1, result2);
}
```

---

## 🟢 Additional Enhancements (Week 3)

### Exchange-Specific Profiles
```rust
pub enum ExchangeProfile {
    Binance {
        latency: LatencyProfile {
            rest: 10..30,  // ms
            websocket: 5..15,
        },
        rate_limits: RateLimits {
            weight_per_minute: 1200,
            orders_per_second: 10,
        },
    },
    Coinbase {
        latency: LatencyProfile {
            rest: 20..50,
            websocket: 10..25,
        },
        rate_limits: RateLimits {
            requests_per_second: 15,
        },
    },
}
```

### Extended Error Taxonomy
```rust
pub enum ExchangeError {
    PriceFilter { code: -1013 },
    TooManyOrders { code: -1015 },
    TimestampOutOfRange { code: -1021 },
    InvalidPriceQty { code: -2010 },
    UnknownOrder { code: -2011 },
    InvalidApiKey { code: -2015 },
    MissingParameter { code: -1102 },
    ServerError { code: 500, retry_after: Duration },
    NetworkTimeout { code: 504 },
    DuplicateClientOrderId { code: -2026 },
}
```

### Market Impact Models
```rust
pub enum MarketImpact {
    Linear { coefficient: f64 },
    SquareRoot { coefficient: f64 },  // Sophia's recommendation
    OrderBookDriven { walk_depth: bool },
}

impl MarketImpact {
    pub fn calculate_slippage(&self, order: &Order, book: &OrderBook) -> f64 {
        match self {
            MarketImpact::SquareRoot { coefficient } => {
                coefficient * (order.quantity / book.total_liquidity()).sqrt()
            }
            MarketImpact::OrderBookDriven { .. } => {
                // Walk the book level by level
                self.walk_order_book(order, book)
            }
        }
    }
}
```

---

## Test Coverage Expansion

### New Test Scenarios Required:
1. **Idempotency**: Same client_order_id returns same result
2. **OCO Simultaneous**: Both legs trigger at once
3. **Post-Only Rejection**: Order would cross spread
4. **Reduce-Only**: Cannot increase position
5. **FOK Semantics**: All or nothing execution
6. **IOC Remainder**: Partial fill + cancel rest
7. **Amend Race**: Modify while filling
8. **Rate Limit Burst**: Burst + retry with backoff
9. **Timestamp Skew**: -1021 error scenarios
10. **Fee Calculations**: Maker/taker with rebates

---

## Metrics to Add

```rust
pub struct SimulatorMetrics {
    // Sophia's requested metrics
    validations_total: Counter,
    rejects_by_reason: HashMap<String, Counter>,
    fees_collected: Gauge,
    slippage_realized: Histogram,
    slippage_expected: Histogram,
    idempotent_hits: Counter,
    timestamp_skew_errors: Counter,
    
    // Per-symbol metrics
    orders_per_symbol: HashMap<Symbol, Counter>,
    volume_per_symbol: HashMap<Symbol, Gauge>,
}
```

---

## Timeline

### Week 1 (Critical):
- Mon-Tue: Idempotency + Client Order ID
- Wed: OCO edge cases
- Thu: Fee model
- Fri: Timestamp validation + Filters

### Week 2 (Medium):
- Mon-Tue: Per-symbol actors
- Wed-Thu: Property tests
- Fri: Integration testing

### Week 3 (Enhancements):
- Mon: Exchange profiles
- Tue: Error taxonomy
- Wed: Market impact
- Thu-Fri: Final testing

---

## Success Criteria

Sophia will mark **APPROVED** when:
1. ✅ Idempotency fully implemented
2. ✅ OCO semantics correct
3. ✅ Fees applied to all fills
4. ✅ Timestamp validation active
5. ✅ All filters enforced
6. ✅ Deterministic execution
7. ✅ Property tests passing

---

*With these fixes, we'll achieve production-grade status!*