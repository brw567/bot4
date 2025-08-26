# 📋 DEDUPLICATION TASKS MASTER LIST
## Complete Actionable Task List with File Locations
## Team: Full 8-Member Execution
## Generated: August 26, 2025

---

# 🔴 IMMEDIATE DEDUPLICATION TARGETS

## Priority 1: Most Critical Duplications (44 instances!)

### 🎯 TASK 1: Consolidate Order Struct (44 → 1)
**Impact**: Highest - 44 duplicate definitions
**Files with Order structs to consolidate**:
```
1. rust_core/domain/entities/order.rs:15
2. rust_core/domain/value_objects/order_value.rs:14
3. rust_core/ports/outbound/exchange_port.rs:14
4. rust_core/dto/response/order_response.rs:13
5. rust_core/dto/database/order_dto.rs:15
6. rust_core/crates/data_intelligence/src/lib.rs:290
7. rust_core/crates/websocket/src/zero_copy.rs:195
8. rust_core/crates/websocket/src/message.rs:68
9. rust_core/crates/risk/src/manipulation_detection.rs:34
10. rust_core/crates/risk/src/decision_orchestrator.rs:803
... [34 more locations - see ARCHITECTURE_DEEP_DIVE_ANALYSIS.md for full list]
```

**Action Steps**:
```bash
# Step 1: Create canonical types crate
cd /home/hamster/bot4/rust_core/crates
cargo new domain_types --lib

# Step 2: Copy the canonical Order implementation
cp /home/hamster/bot4/REFACTORING_IMPLEMENTATION_GUIDE.md domain_types/ORDER_TEMPLATE.md

# Step 3: Implement canonical Order in domain_types/src/order.rs
# Step 4: Run migration script (see below)
```

---

## Priority 2: Mathematical Functions (13 instances)

### 🎯 TASK 2: Consolidate calculate_correlation (13 → 1)
**Files to consolidate**:
```
1. rust_core/crates/data_intelligence/src/macro_economy_enhanced.rs:266
2. rust_core/crates/data_intelligence/src/macro_correlator.rs:61
3. rust_core/crates/data_intelligence/src/overfitting_prevention.rs:806
4. rust_core/crates/risk/src/hyperparameter_integration.rs:571
5. rust_core/crates/risk/src/kyle_lambda_validation.rs:354
6. rust_core/crates/risk/src/hyperparameter_optimization.rs:1800
7. rust_core/crates/risk_engine/src/correlation.rs:146
8. rust_core/crates/risk_engine/src/correlation_avx512.rs:207
9. rust_core/crates/risk_engine/src/correlation_portable.rs:561
10. rust_core/crates/risk_engine/src/correlation_simd.rs:176
11. rust_core/crates/analysis/src/statistical_tests.rs:221
12. rust_core/crates/ml/src/validation/purged_cv.rs:286
13. rust_core/crates/ml/src/feature_engine/selector.rs:289
```

**Action**:
```bash
# Create mathematical operations library
cd /home/hamster/bot4/rust_core/crates
cargo new mathematical_ops --lib

# Move best implementation to mathematical_ops/src/correlation.rs
# Update all 13 imports to use mathematical_ops::calculate_correlation
```

### 🎯 TASK 3: Consolidate calculate_var (8 → 1)
**Files to consolidate**:
```
1. rust_core/crates/risk/src/monte_carlo.rs:604
2. rust_core/crates/risk/src/garch.rs:526
3. rust_core/crates/ml/src/garch.rs:233
4. rust_core/crates/ml/src/training/metrics.rs:348
5. rust_core/crates/ml/src/feature_engine/selector.rs:330
6. rust_core/crates/ml/src/models/garch.rs:122
7. rust_core/crates/infrastructure/src/circuit_breaker_layer_integration.rs:421
8. rust_core/crates/infrastructure/src/statistical_circuit_breakers.rs:593
```

---

## Priority 3: Technical Indicators (4 instances each)

### 🎯 TASK 4: Consolidate calculate_ema (4 → 1)
**Files to consolidate**:
```
1. rust_core/crates/risk/src/market_analytics.rs:466
2. rust_core/crates/risk/src/ta_accuracy_audit.rs:159
3. rust_core/crates/risk/src/ml_complete_impl.rs:748
4. rust_core/crates/infrastructure/src/simd_avx512.rs:46
```

### 🎯 TASK 5: Consolidate calculate_rsi (4 → 1)
**Files to consolidate**:
```
1. rust_core/crates/risk/src/market_analytics.rs:505
2. rust_core/crates/risk/src/ml_complete_impl.rs:40
3. rust_core/crates/ml/src/feature_engine/indicators.rs:473
4. rust_core/crates/infrastructure/src/simd_avx512.rs:85
```

---

## Priority 4: Trading Operations (6-8 instances each)

### 🎯 TASK 6: Consolidate process_event (6 → 1)
**Files to consolidate**:
```
1. rust_core/crates/data_intelligence/src/stablecoin_tracker.rs:760
2. rust_core/crates/order_management/src/state_machine.rs:252
3. rust_core/crates/risk_engine/src/market_maker_detection.rs:158
4. rust_core/crates/infrastructure/src/circuit_breaker_layer_integration.rs:788
5. rust_core/crates/data_ingestion/src/aggregators/timescale_aggregator.rs:732
6. rust_core/crates/data_ingestion/src/replay/playback_engine.rs:500
```

### 🎯 TASK 7: Consolidate place_order (6 → 1)
**Files to consolidate**:
```
1. rust_core/ports/outbound/exchange_port.rs:65
2. rust_core/adapters/outbound/exchanges/symbol_actor.rs:291
3. rust_core/adapters/outbound/exchanges/binance_real.rs:492
4. rust_core/adapters/outbound/exchanges/exchange_simulator.rs:430
5. rust_core/adapters/outbound/exchanges/exchange_adapter_trait.rs:289
6. rust_core/adapters/inbound/rest/api_server.rs:220
```

### 🎯 TASK 8: Consolidate cancel_order (8 → 1)
**Files to consolidate**:
```
1. rust_core/ports/outbound/exchange_port.rs:68
2. rust_core/crates/order_management/src/manager.rs:245
3. rust_core/crates/trading_engine/src/orders/oco.rs:155
4. rust_core/crates/trading_engine/src/transactions/compensator.rs:428
5. rust_core/adapters/outbound/exchanges/symbol_actor.rs:312
6. rust_core/adapters/outbound/exchanges/binance_real.rs:591
7. rust_core/adapters/outbound/exchanges/exchange_simulator.rs:553
8. rust_core/adapters/outbound/exchanges/exchange_adapter_trait.rs:380
```

---

## Priority 5: Type Definitions (14 instances)

### 🎯 TASK 9: Consolidate Price Type (14 → 1)
**Files to consolidate**:
```
1. rust_core/crates/websocket/src/message.rs:76 (PriceLevel)
2. rust_core/crates/data_ingestion/src/replay/microburst_detector.rs:92 (PriceJump)
3. rust_core/crates/feature_store/src/market_microstructure.rs:813 (PriceBar)
4. rust_core/crates/types/src/market.rs:5 (Price with Decimal)
5. rust_core/domain/value_objects/decimal_money.rs:184 (Price struct)
6. rust_core/crates/risk/src/market_analytics.rs:41 (PriceHistory)
7. rust_core/domain/value_objects/validation_filters.rs:12 (PriceFilter)
8. rust_core/domain/value_objects/price.rs:15 (Price with f64)
9. rust_core/crates/risk/src/unified_types.rs:21 (Price with Decimal)
10. rust_core/crates/data_ingestion/src/event_driven/adaptive_sampler.rs:125 (PriceSample)
11. rust_core/crates/risk/src/order_book_analytics.rs:72 (PriceLevel)
12. rust_core/crates/infrastructure/src/panic_conditions.rs:614 (PriceDivergenceMonitor)
13. rust_core/crates/infrastructure/src/panic_conditions.rs:626 (PricePoint)
14. rust_core/crates/ml/src/feature_engine/harmonic_patterns.rs:45 (PricePoint)
```

---

# 🛠️ AUTOMATED DEDUPLICATION SCRIPTS

## Script 1: Find and Replace Order Types
```python
#!/usr/bin/env python3
# scripts/deduplicate_orders.py

import os
import re
from pathlib import Path

# Map of old Order structs to canonical
ORDER_MAPPINGS = {
    'domain::entities::Order': 'domain_types::order::Order',
    'dto::database::Order': 'domain_types::order::Order',
    'types::Order': 'domain_types::order::Order',
    # Add all 44 mappings
}

def replace_order_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    modified = False
    for old, new in ORDER_MAPPINGS.items():
        if old in content:
            content = content.replace(f'use {old};', f'use {new};')
            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")

# Execute
for rust_file in Path('/home/hamster/bot4/rust_core').rglob('*.rs'):
    replace_order_imports(rust_file)
```

## Script 2: Consolidate Mathematical Functions
```bash
#!/bin/bash
# scripts/consolidate_math.sh

# Create consolidated math library
cd /home/hamster/bot4/rust_core/crates
cargo new mathematical_ops --lib

# Copy best implementations
echo "Copying best calculate_correlation from correlation_avx512.rs..."
cp rust_core/crates/risk_engine/src/correlation_avx512.rs \
   mathematical_ops/src/correlation.rs

echo "Copying best calculate_var from monte_carlo.rs..."
cp rust_core/crates/risk/src/monte_carlo.rs \
   mathematical_ops/src/var.rs

# Update all imports
find . -name "*.rs" -exec sed -i \
  's/use.*calculate_correlation.*/use mathematical_ops::correlation::calculate_correlation;/g' {} \;

find . -name "*.rs" -exec sed -i \
  's/use.*calculate_var.*/use mathematical_ops::var::calculate_var;/g' {} \;
```

---

# 📊 TRACKING PROGRESS

## Deduplication Scorecard
| Task | Duplicates | Target | Status | Owner |
|------|-----------|---------|--------|-------|
| Order Struct | 44 | 1 | ⏳ | Sam |
| calculate_correlation | 13 | 1 | ⏳ | Morgan |
| calculate_var | 8 | 1 | ⏳ | Morgan |
| place_order | 6 | 1 | ⏳ | Casey |
| cancel_order | 8 | 1 | ⏳ | Casey |
| process_event | 6 | 1 | ⏳ | Avery |
| Price Type | 14 | 1 | ⏳ | Sam |
| calculate_ema | 4 | 1 | ⏳ | Jordan |
| calculate_rsi | 4 | 1 | ⏳ | Jordan |
| get_balance | 6 | 1 | ⏳ | Quinn |
| update_position | 5 | 1 | ⏳ | Quinn |
| validate_order | 4 | 1 | ⏳ | Riley |
| calculate_kelly | 2 | 1 | ⏳ | Morgan |
| calculate_volatility | 3 | 1 | ⏳ | Morgan |
| Trade Struct | 18 | 1 | ⏳ | Sam |
| Candle Struct | 6 | 1 | ⏳ | Avery |
| MarketData Struct | 6 | 1 | ⏳ | Avery |
| RiskLimits Struct | 3 | 1 | ⏳ | Quinn |

## Total: 158 duplicates → 18 canonical implementations

---

# ⚡ EXECUTION ORDER

## Phase 1: Types (Week 1)
1. Create `domain_types` crate
2. Consolidate Order (44 → 1)
3. Consolidate Price (14 → 1)
4. Consolidate Trade (18 → 1)
5. Consolidate other structs

## Phase 2: Functions (Week 2)
1. Create `mathematical_ops` crate
2. Consolidate math functions (31 → 5)
3. Create `indicators` crate
4. Consolidate TA indicators (11 → 4)

## Phase 3: Operations (Week 3)
1. Create `event_bus` crate
2. Consolidate process_event (6 → 1)
3. Consolidate order operations (20 → 3)

## Phase 4: Cleanup (Week 4)
1. Remove all old implementations
2. Update all imports
3. Verify no compilation errors
4. Run all 7,809 tests
5. Performance validation

---

# ✅ VERIFICATION COMMANDS

```bash
# Count remaining duplicates
./scripts/check_duplicates.sh

# Verify compilation
cd /home/hamster/bot4/rust_core
cargo build --all

# Run all tests
cargo test --all

# Check for unused dependencies
cargo udeps

# Measure binary size reduction
ls -lh target/release/bot4-trading
```

---

# 🚀 START IMMEDIATELY!

**The deduplication tasks are clearly defined with exact file locations.**
**Begin with Task 1: Consolidating 44 Order structs into 1.**