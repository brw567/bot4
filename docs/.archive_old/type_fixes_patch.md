# Type Standardization Patches

## Quick Reference for Fixing Each Component

### 1. ✅ TimeframeAggregator (COMPLETED)
- Uses `bot3_common::Signal` instead of custom types
- Returns `Result<Signal, TradingError>` from all functions
- Validates all inputs with common validators

### 2. AdaptiveThresholds
```rust
// Add to Cargo.toml
bot3-common = { path = "../common" }

// Replace in lib.rs
use bot3_common::{
    Signal, SignalSource, MarketRegime, TradingError, ValidationError,
    validate_signal_strength, validate_confidence,
};

// Change function signature
pub fn apply_dynamic_threshold(
    &self,
    signal: &Signal,  // Changed from f64
    volatility: f64,
    regime: MarketRegime,
) -> Result<Signal, TradingError> {  // Returns Result<Signal>
    // Validate inputs
    let volatility = validate_confidence(volatility)?;
    
    // Apply threshold logic
    let adjusted_strength = signal.strength * adjustment_factor;
    
    // Return new Signal
    Signal::new(
        signal.source.clone(),
        adjusted_strength,
        signal.confidence * confidence_factor,
    )
}
```

### 3. Microstructure
```rust
// Remove these duplicate types
- pub struct OrderBookSignal { ... }
- pub struct SpreadSignal { ... }  
- pub struct FlowSignal { ... }

// Replace with
use bot3_common::{Signal, SignalSource, MicrostructureType, TradingError};

// Change all analyzer functions to return Signal
pub fn analyze_order_book(&mut self, snapshot: &OrderBookSnapshot) -> Result<Signal, TradingError> {
    // ... calculation ...
    Signal::new(
        SignalSource::Microstructure(MicrostructureType::OrderBookImbalance),
        imbalance.pressure,
        imbalance.confidence,
    )
}
```

### 4. ✅ Kelly Criterion (COMPLETED)
- Uses `bot3_common::PositionSize`
- Returns `Result<PositionSize, TradingError>`
- Includes correlation module

### 5. SmartLeverage
```rust
// Remove
- pub struct LeverageDecision { ... }

// Add
use bot3_common::{Signal, PositionSize, TradingError, validate_leverage};

// Change function
pub fn calculate_optimal_leverage(
    &self,
    signal: &Signal,  // Use Signal type
    kelly_fraction: f64,
) -> Result<PositionSize, TradingError> {
    let leverage = calculate_leverage_from_kelly(kelly_fraction);
    let leverage = validate_leverage(leverage)?;
    
    Ok(PositionSize {
        size: signal.strength * self.capital * kelly_fraction,
        kelly_fraction,
        leverage,
        risk_adjusted_size: /* calculated */,
        confidence: signal.confidence,
    })
}
```

### 6. ReinvestmentEngine
```rust
use bot3_common::{PositionSize, TradingError};

pub fn process_profit(
    &mut self,
    position: &PositionSize,  // Use standard type
    profit: f64,
) -> Result<CompoundDecision, TradingError> {
    // Validation
    if profit < 0.0 {
        return Err(TradingError::InvalidParameters("Negative profit".to_string()));
    }
    
    // Calculate reinvestment
    let reinvest_amount = profit * self.reinvestment_rate;
    
    Ok(CompoundDecision {
        reinvest_amount,
        withdraw_amount: profit - reinvest_amount,
        new_capital: self.capital + reinvest_amount,
        compound_rate: self.reinvestment_rate,
    })
}
```

### 7. CrossExchangeArbitrage
```rust
// Remove
- pub struct ArbitrageOpportunity { ... }  // Name collision!

// Add
use bot3_common::{Opportunity, OpportunityType, OpportunityDetails, TradingError};

// Change function
pub fn find_opportunities(&self) -> Result<Vec<Opportunity>, TradingError> {
    let mut opportunities = Vec::new();
    
    // ... find opportunities ...
    
    opportunities.push(Opportunity {
        opportunity_type: OpportunityType::CrossExchangeArbitrage,
        profit_percentage,
        confidence,
        execution_window: Duration::from_millis(500),
        risk_score,
        details: OpportunityDetails::CrossExchange {
            symbol: symbol.clone(),
            buy_exchange: buy_exchange.clone(),
            sell_exchange: sell_exchange.clone(),
            buy_price,
            sell_price,
        },
    });
    
    Ok(opportunities)
}
```

### 8. StatisticalArbitrage
```rust
// Remove
- pub struct StatArbOpportunity { ... }

// Add same as CrossExchange but use
opportunity_type: OpportunityType::StatisticalArbitrage,
details: OpportunityDetails::Statistical {
    pair1: pair1.clone(),
    pair2: pair2.clone(),
    z_score,
    half_life,
    hedge_ratio,
},
```

### 9. TriangularArbitrage
```rust
// Remove
- pub struct TriangularOpportunity { ... }

// Add same pattern
opportunity_type: OpportunityType::TriangularArbitrage,
details: OpportunityDetails::Triangular {
    path: path.clone(),
    rates: rates.clone(),
    exchanges: exchanges.clone(),
},
```

### 10. Integration Tests
```rust
// Remove all custom types
- struct MarketData { ... }
- struct Signal { ... }
- struct Position { ... }

// Use production types
use bot3_common::{
    Signal, SignalSource, Timeframe,
    Opportunity, OpportunityType,
    PositionSize, Order,
    TradingError,
};

// Update all test functions to use real types
#[test]
fn test_signal_processing_integration() {
    let signal = Signal::new(
        SignalSource::Timeframe(Timeframe::H1),
        0.7,
        0.8,
    ).unwrap();
    
    // Use real aggregator
    let aggregator = timeframe_aggregator::TimeframeAggregator::new();
    let result = aggregator.calculate_weighted_signal().unwrap();
    
    assert!(result.confidence > 0.6);
}
```

## Common Patterns to Apply

### Pattern 1: Add Result Types
```rust
// Before
pub fn calculate(x: f64) -> f64

// After
pub fn calculate(x: f64) -> Result<f64, TradingError>
```

### Pattern 2: Validate Inputs
```rust
use bot3_common::{validate_signal_strength, validate_confidence, validate_leverage};

// At start of functions
let signal = validate_signal_strength(signal)?;
let confidence = validate_confidence(confidence)?;
```

### Pattern 3: Use Common Types
```rust
// Instead of custom Signal/Opportunity/Position types
use bot3_common::{Signal, Opportunity, PositionSize};
```

### Pattern 4: Error Handling
```rust
// Replace panic/unwrap with proper errors
if condition_failed {
    return Err(TradingError::CalculationError("Reason".to_string()));
}
```

## Validation Checklist

After applying fixes to each component:

- [ ] Compiles without warnings
- [ ] All functions return Result
- [ ] Uses bot3_common types
- [ ] No duplicate type definitions
- [ ] Input validation added
- [ ] Tests updated to use real types
- [ ] Documentation updated

## Build Commands

```bash
# Check individual component
cd crates/component_name
cargo check

# Check entire workspace
cd /home/hamster/bot4/rust_core
cargo check --all

# Run tests
cargo test --all

# Fix formatting
cargo fmt --all
```