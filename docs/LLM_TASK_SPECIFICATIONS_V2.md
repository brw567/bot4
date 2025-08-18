# Bot4 LLM-Optimized Task Specifications V2
## CRITICAL UPDATES FROM EXTERNAL REVIEWS
## Version: 2.0 | Date: 2025-01-18 | Reviews: Sophia + Nexus

---

## 🔴 CRITICAL CHANGES - MUST READ FIRST

```yaml
breaking_changes:
  1_new_blocking_phase:
    phase: 3.3 (Safety Controls)
    priority: BLOCKS_ALL_TRADING
    reason: Sophia mandates safety before any live system
    
  2_grok_architecture:
    requirement: ASYNC_ONLY
    never: In hot path or decision loop
    always: Background enrichment with cache
    
  3_minimum_capital:
    old: $2,000 (impossible)
    new: $10,000 (realistic)
    reason: Costs require 1.4% monthly at $10k
    
  4_risk_models:
    old: Historical VaR
    new: GARCH-VaR with Student-t
    reason: 20-30% tail underestimation fix
    
  5_ml_validation:
    old: train_test_split
    new: TimeSeriesSplit with purge
    reason: Prevents future leakage
```

---

## 📋 REVISED PHASE 3 ATOMIC TASKS

### PHASE 3.3: SAFETY CONTROLS (NEW - HIGHEST PRIORITY)

```yaml
task_id: TASK_3.3.1
task_name: Implement Hardware Kill Switch
parent_phase: 3.3
dependencies: []  # No dependencies - FIRST PRIORITY
owner: Sam
estimated_hours: 8

specification:
  inputs:
    required:
      gpio_pin: BCM_17
      interrupt_type: FALLING_EDGE
  outputs:
    deliverables:
      kill_switch: GPIO interrupt handler
      status_leds: Red/Yellow/Green indicators

implementation:
  steps:
    1. Setup GPIO interrupt on BCM_17
    2. Implement immediate halt handler
    3. Add status LED control (BCM_22,23,24)
    4. Add tamper detection
    5. Test emergency stop scenarios

success_criteria:
  - Kill switch halts all trading <1ms
  - Status LEDs reflect system state
  - Tamper detection triggers alerts
  - Audit log captures all events

tests:
  - Test GPIO interrupt triggering
  - Verify <1ms halt time
  - Test LED state transitions
  - Verify audit logging
```

```yaml
task_id: TASK_3.3.2
task_name: Software Control Modes
parent_phase: 3.3
dependencies: [TASK_3.3.1]
owner: Riley
estimated_hours: 12

specification:
  inputs:
    required:
      modes: [Normal, Paused, Reduced, Emergency]
  outputs:
    deliverables:
      control_system: Multi-mode trading controller
      audit_system: Tamper-proof logging

implementation:
  steps:
    1. Create TradingMode enum
    2. Implement SafetyController with atomic mode switching
    3. Add audit logging for all transitions
    4. Implement graduated emergency actions
    5. Create read-only dashboard views

success_criteria:
  - Mode switches are atomic
  - Every transition is logged
  - No manual trading possible
  - Dashboard is read-only

tests:
  - Test all mode transitions
  - Verify audit trail integrity
  - Test emergency liquidation
  - Verify no manual overrides
```

### PHASE 3.4: PERFORMANCE INFRASTRUCTURE (REVISED)

```yaml
task_id: TASK_3.4.1
task_name: MiMalloc Global Allocator
parent_phase: 3.4
dependencies: [TASK_3.3.2]  # Safety first
owner: Jordan
estimated_hours: 4

specification:
  inputs:
    required:
      allocator: mimalloc v0.1
  outputs:
    deliverables:
      global_allocator: 7ns allocation latency

implementation:
  steps:
    1. Add mimalloc to Cargo.toml
    2. Set #[global_allocator]
    3. Benchmark allocation latency
    4. Verify across all components
    5. Document performance gains

success_criteria:
  - Allocation latency <10ns
  - No memory leaks
  - All tests pass with new allocator

tests:
  - Benchmark allocation speed
  - Memory leak detection
  - Stress test with 1M allocations
```

```yaml
task_id: TASK_3.4.2
task_name: Object Pool Implementation
parent_phase: 3.4
dependencies: [TASK_3.4.1]
owner: Jordan
estimated_hours: 8

specification:
  inputs:
    required:
      pools:
        orders: 1_000_000
        ticks: 10_000_000
        signals: 100_000
  outputs:
    deliverables:
      object_pools: Lock-free pre-allocated pools

implementation:
  steps:
    1. Create generic ObjectPool<T>
    2. Pre-allocate 1M orders
    3. Pre-allocate 10M ticks
    4. Pre-allocate 100K signals
    5. Implement lock-free acquire/release

success_criteria:
  - Zero allocations in hot path
  - Lock-free operations
  - <100ns acquire/release

tests:
  - Verify pool sizes
  - Test concurrent access
  - Benchmark acquire/release
  - Test pool exhaustion handling
```

### PHASE 3.5: ENHANCED MODELS & RISK (EXPANDED)

```yaml
task_id: TASK_3.5.1
task_name: GARCH-ARIMA Implementation
parent_phase: 3.5
dependencies: [TASK_3.4.2]
owner: Morgan
estimated_hours: 12

specification:
  inputs:
    required:
      model: GARCH(1,1) + ARIMA(2,1,2)
      distribution: Student-t(df=4)
  outputs:
    deliverables:
      garch_arima: Fat-tail aware forecasting

implementation:
  steps:
    1. Replace basic ARIMA with GARCH-ARIMA
    2. Implement Student-t distribution
    3. Handle volatility clustering
    4. Add conditional volatility forecasting
    5. Validate 15-25% RMSE improvement

success_criteria:
  - Handles kurtosis >3
  - 15-25% RMSE improvement
  - Proper volatility clustering
  - Fat tail modeling

tests:
  - Test on crypto data (kurtosis >3)
  - Verify RMSE improvement
  - Test volatility forecasting
  - Validate Student-t fit
```

```yaml
task_id: TASK_3.5.2
task_name: GARCH-VaR Integration
parent_phase: 3.5
dependencies: [TASK_3.5.1]
owner: Quinn
estimated_hours: 8

specification:
  inputs:
    required:
      confidence: 0.99
      horizon: 1 day
      limit: 0.02 (2% daily)
  outputs:
    deliverables:
      garch_var: Proper tail risk estimation

implementation:
  steps:
    1. Implement GARCHVaR struct
    2. Add conditional volatility forecast
    3. Use Student-t quantiles
    4. Add CVaR calculation
    5. Fix 20-30% underestimation

success_criteria:
  - VaR captures fat tails
  - 20-30% better than historical
  - CVaR implemented
  - Hard limit enforcement

tests:
  - Backtest VaR violations
  - Compare with historical VaR
  - Test on crisis periods
  - Verify limit enforcement
```

```yaml
task_id: TASK_3.5.3
task_name: TimeSeriesSplit CV
parent_phase: 3.5
dependencies: [TASK_3.5.1]
owner: Morgan
estimated_hours: 6

specification:
  inputs:
    required:
      n_splits: 10
      gap: 1 week
      test_size: 1 month
  outputs:
    deliverables:
      cv_pipeline: No future leakage validation

implementation:
  steps:
    1. Replace train_test_split
    2. Implement TimeSeriesSplit
    3. Add purge for overlaps
    4. Add embargo for leakage
    5. Validate <10% generalization error

success_criteria:
  - No future leakage
  - Proper time ordering
  - <10% generalization gap
  - Purge and embargo working

tests:
  - Test for future leakage
  - Verify time ordering
  - Test purge mechanism
  - Validate embargo period
```

```yaml
task_id: TASK_3.5.4
task_name: Partial Fill Manager
parent_phase: 3.5
dependencies: [TASK_3.5.2]
owner: Casey
estimated_hours: 10

specification:
  inputs:
    required:
      tracking: Weighted average entry
      repricing: Dynamic stop/target adjustment
  outputs:
    deliverables:
      fill_manager: Partial-fill aware system

implementation:
  steps:
    1. Track all partial fills
    2. Calculate weighted average entry
    3. Reprice stops after each fill
    4. Implement OCO management
    5. Exchange-specific handling

success_criteria:
  - Accurate weighted averages
  - Stops reprice correctly
  - OCO orders work
  - All exchanges supported

tests:
  - Test partial fill scenarios
  - Verify weighted averages
  - Test stop repricing
  - Test OCO logic
```

### PHASE 3.6: GROK INTEGRATION (ASYNC ONLY)

```yaml
task_id: TASK_3.6.1
task_name: Async Grok Enrichment Service
parent_phase: 3.6
dependencies: [TASK_3.5.4]
owner: Casey + Avery
estimated_hours: 8

specification:
  inputs:
    required:
      pattern: Background enrichment only
      cache_ttl: 5 minutes
      budget: $20/month (80,000 requests)
  outputs:
    deliverables:
      grok_service: Async-only enrichment

implementation:
  steps:
    1. Create background tokio task
    2. Implement 3-tier cache
    3. Add ROI tracking
    4. Never block trading path
    5. Cache-only access from trading

success_criteria:
  - Zero blocking of trades
  - Cache hit rate >85%
  - ROI tracking working
  - Budget adherence

tests:
  - Verify async operation
  - Test cache hit rates
  - Verify no blocking
  - Test budget limits
```

## 📊 CRITICAL PERFORMANCE TARGETS (REVISED)

```yaml
performance_requirements:
  latency:
    decision: <1μs p99 (149ns current OK)
    allocation: <10ns (with MiMalloc)
    ml_inference: <200μs
    risk_check: <10μs
    
  throughput:
    internal: 500k ops/sec (not 1M)
    orders: 5-10k/sec
    risk_checks: 100k/sec
    
  accuracy:
    sharpe: 1.5-2.0 realistic
    win_rate: >55%
    max_drawdown: <15-20%
```

## 🚨 VALIDATION REQUIREMENTS

```yaml
before_any_task:
  - Check dependencies complete
  - Verify test environment ready
  - Review success criteria
  
after_each_task:
  - Run all specified tests
  - Update task status
  - Document actual metrics
  - Update dependent tasks
  
before_live_trading:
  - ALL Phase 3.3 tasks complete (safety)
  - ALL Phase 3.4 tasks complete (performance)
  - ALL Phase 3.5 tasks complete (risk/ML)
  - 60-90 days paper trading
  - Positive after-cost metrics
```

## ✅ TASK STATUS TRACKING

```yaml
phase_3_3_safety:        # HIGHEST PRIORITY
  TASK_3.3.1: pending    # Hardware kill switch
  TASK_3.3.2: pending    # Software control modes
  
phase_3_4_performance:   # CRITICAL
  TASK_3.4.1: pending    # MiMalloc
  TASK_3.4.2: pending    # Object pools
  
phase_3_5_models:        # ESSENTIAL
  TASK_3.5.1: pending    # GARCH-ARIMA
  TASK_3.5.2: pending    # GARCH-VaR
  TASK_3.5.3: pending    # TimeSeriesSplit
  TASK_3.5.4: pending    # Partial fills
  
phase_3_6_grok:          # MEDIUM
  TASK_3.6.1: pending    # Async enrichment
```

---

**This document supersedes previous task specifications where conflicts exist.**
**External review requirements have been fully integrated.**
**Tasks must be completed in dependency order.**