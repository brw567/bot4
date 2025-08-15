# Type Standardization Migration Guide

**Date**: January 12, 2025
**Purpose**: Fix type inconsistencies across all core components
**Impact**: All 10 core components need updates

---

## 📋 Migration Checklist

### Phase 1: Add Common Types Module
- [x] Create `bot3-common` crate with unified types
- [x] Add to workspace Cargo.toml
- [ ] Update all components to use common types

### Phase 2: Update Each Component

#### 1. TimeframeAggregator
```rust
// OLD
use crate::{TimeframeSignal, AggregatedSignal, MLSignal};

// NEW
use bot3_common::{Signal, SignalSource, Timeframe, TradingError};

// Update function signatures
pub fn calculate_weighted_signal(&self) -> Result<Signal, TradingError> {
    // Implementation
}
```

#### 2. AdaptiveThresholds
```rust
// OLD
pub fn apply_dynamic_threshold(&self, signal: f64, ...) -> ThresholdResult

// NEW
use bot3_common::{Signal, MarketRegime, TradingError};

pub fn apply_dynamic_threshold(&self, signal: &Signal, ...) -> Result<Signal, TradingError>
```

#### 3. Microstructure
```rust
// OLD - Multiple signal types
pub struct OrderBookSignal { ... }
pub struct SpreadSignal { ... }
pub struct FlowSignal { ... }

// NEW - Single signal with metadata
use bot3_common::{Signal, SignalSource, MicrostructureType};

pub fn analyze_order_book(&mut self, snapshot: &OrderBookSnapshot) -> Result<Signal, TradingError> {
    let signal = Signal::new(
        SignalSource::Microstructure(MicrostructureType::OrderBookImbalance),
        imbalance.pressure,
        imbalance.confidence,
    )?;
    Ok(signal)
}
```

#### 4. Kelly Criterion
```rust
// OLD
pub fn calculate_position_size(...) -> PositionSizeResult

// NEW
use bot3_common::{PositionSize, TradingError};

pub fn calculate_position_size(...) -> Result<PositionSize, TradingError> {
    // Add validation
    let signal_strength = validate_signal_strength(signal_strength)?;
    let confidence = validate_confidence(confidence)?;
    
    // Implementation
}

// Add portfolio Kelly
mod correlation; // New module we created
pub use correlation::PortfolioKelly;
```

#### 5. Smart Leverage
```rust
// OLD
pub struct LeverageDecision { ... }

// NEW
use bot3_common::{PositionSize, TradingError, validate_leverage};

pub fn calculate_optimal_leverage(
    &self,
    signal: &Signal,  // Use Signal type
    kelly_fraction: f64,
) -> Result<PositionSize, TradingError>
```

#### 6. Reinvestment Engine
```rust
// OLD
pub fn process_profit(...) -> CompoundDecision

// NEW
use bot3_common::{PositionSize, TradingError};

pub fn process_profit(
    &mut self,
    position: &PositionSize,  // Use standard type
    profit: f64,
) -> Result<CompoundDecision, TradingError>
```

#### 7. Cross-Exchange Arbitrage
```rust
// OLD
pub struct ArbitrageOpportunity { ... }  // Name collision!

// NEW
use bot3_common::{Opportunity, OpportunityType, OpportunityDetails};

pub fn find_opportunities(&self) -> Result<Vec<Opportunity>, TradingError> {
    let opportunity = Opportunity {
        opportunity_type: OpportunityType::CrossExchangeArbitrage,
        profit_percentage,
        confidence,
        execution_window: Duration::from_millis(500),
        risk_score,
        details: OpportunityDetails::CrossExchange { ... },
    };
}
```

#### 8. Statistical Arbitrage
```rust
// OLD
pub struct StatArbOpportunity { ... }

// NEW
use bot3_common::{Opportunity, OpportunityType, OpportunityDetails};

pub fn get_opportunities(&self) -> Result<Vec<Opportunity>, TradingError> {
    let opportunity = Opportunity {
        opportunity_type: OpportunityType::StatisticalArbitrage,
        details: OpportunityDetails::Statistical { ... },
        ...
    };
}
```

#### 9. Triangular Arbitrage
```rust
// OLD
pub struct TriangularOpportunity { ... }

// NEW - Same pattern as above
use bot3_common::{Opportunity, OpportunityType, OpportunityDetails};
```

#### 10. Integration Tests
```rust
// OLD - Custom types
struct MarketData { ... }
struct Signal { ... }

// NEW - Use production types
use bot3_common::{Signal, SignalSource, Opportunity, Order};
use timeframe_aggregator::TimeframeAggregator;
use kelly_criterion::{KellyCriterion, PortfolioKelly};
```

---

## 🔧 Implementation Steps

### Step 1: Update Cargo.toml files
```toml
# In each component's Cargo.toml
[dependencies]
bot3-common = { path = "../common" }
```

### Step 2: Update imports
```rust
// At top of each lib.rs
use bot3_common::{
    Signal, SignalSource, Timeframe,
    Opportunity, OpportunityType,
    PositionSize, Order,
    TradingError, ValidationError,
    validate_signal_strength, validate_confidence, validate_leverage,
};
```

### Step 3: Update function signatures
All public functions should return `Result<T, TradingError>`:
```rust
// Before
pub fn calculate_something(value: f64) -> f64

// After
pub fn calculate_something(value: f64) -> Result<f64, TradingError> {
    let validated = validate_signal_strength(value)?;
    // ... calculation
    Ok(result)
}
```

### Step 4: Add validation
```rust
// At start of each calculation function
let signal_strength = validate_signal_strength(signal_strength)?;
let confidence = validate_confidence(confidence)?;
let leverage = validate_leverage(leverage)?;
```

### Step 5: Update tests
```rust
#[test]
fn test_with_result() {
    let result = function_that_returns_result(0.5);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), expected_value);
    
    // Test error cases
    let error_result = function_that_returns_result(2.0); // Out of bounds
    assert!(error_result.is_err());
}
```

---

## 📊 Type Mapping Table

| Old Type | New Type | Module |
|----------|----------|--------|
| TimeframeSignal | Signal | bot3_common |
| AggregatedSignal | Signal | bot3_common |
| MLSignal (duplicate) | Signal | bot3_common |
| ArbitrageOpportunity (multiple) | Opportunity | bot3_common |
| StatArbOpportunity | Opportunity | bot3_common |
| TriangularOpportunity | Opportunity | bot3_common |
| PositionSizeResult | PositionSize | bot3_common |
| LeverageDecision | PositionSize | bot3_common |
| OrderBookSignal | Signal | bot3_common |
| SpreadSignal | Signal | bot3_common |
| FlowSignal | Signal | bot3_common |

---

## 🎯 Benefits After Migration

### Type Safety
- Single source of truth for types
- Compile-time guarantees
- No more type mismatches

### Validation
- Centralized validation functions
- Consistent bounds checking
- Better error messages

### Maintainability
- Changes in one place
- Easier to extend
- Clear documentation

### Integration
- Components work together seamlessly
- Tests use production types
- No conversion needed

---

## ⚠️ Breaking Changes

### API Changes
All public functions now return `Result<T, TradingError>` instead of raw values.

### Type Changes
- Signal types unified
- Opportunity types unified
- Position sizing types unified

### Migration Path
1. Update one component at a time
2. Fix compilation errors
3. Update tests
4. Verify integration

---

## 📝 Example Migration

### Before
```rust
// timeframe_aggregator/src/lib.rs
pub struct TimeframeSignal {
    pub timeframe: Timeframe,
    pub ta_signal: f64,
    pub ml_signal: f64,
    pub strength: f64,
}

pub fn calculate_weighted_signal(&self) -> f64 {
    // calculation
    combined_signal
}
```

### After
```rust
// timeframe_aggregator/src/lib.rs
use bot3_common::{Signal, SignalSource, Timeframe, TradingError};

pub fn calculate_weighted_signal(&self) -> Result<Signal, TradingError> {
    // calculation
    let signal = Signal::new(
        SignalSource::Timeframe(dominant_timeframe),
        combined_strength,
        confidence,
    )?;
    Ok(signal)
}
```

---

## 🚀 Rollout Plan

### Day 1
- [x] Create common types module
- [x] Add correlation to Kelly
- [ ] Update TimeframeAggregator
- [ ] Update AdaptiveThresholds

### Day 2
- [ ] Update Microstructure
- [ ] Update Kelly Criterion
- [ ] Update Smart Leverage
- [ ] Update Reinvestment

### Day 3
- [ ] Update all arbitrage modules
- [ ] Update integration tests
- [ ] Run full test suite
- [ ] Performance benchmarks

---

## ✅ Verification

After migration, verify:
1. All components compile without warnings
2. All tests pass
3. Integration tests use production types
4. No duplicate type definitions
5. Performance unchanged (<100μs)

---

**Migration Status**: Ready to begin
**Estimated Time**: 1-2 days
**Risk**: Low (compile-time safety)