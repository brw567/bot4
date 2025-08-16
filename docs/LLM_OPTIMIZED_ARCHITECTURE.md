# Bot4 LLM-Optimized Architecture Document
## Version 1.0 - Designed for AI Agent Implementation
## Format: Structured for Claude, ChatGPT, Grok, and other LLMs

---

## 📋 DOCUMENT USAGE INSTRUCTIONS FOR LLMs

```yaml
document_type: technical_specification
optimization_for: llm_agents
format: structured_with_contracts
parsing_hints:
  - Look for CONTRACT blocks for interfaces
  - Check REQUIREMENTS before implementation
  - Follow EXAMPLE blocks exactly
  - Use TEST_SPEC for validation
  - Report METRICS after implementation
```

---

## 🎯 COMPONENT SPECIFICATION TEMPLATE

Each component MUST be documented with this structure:

```yaml
component_id: UNIQUE_ID
component_name: NAME
owner: AGENT_NAME
dependencies: [LIST_OF_COMPONENTS]
phase: PHASE_NUMBER

contract:
  inputs:
    - name: TYPE # description
  outputs:
    - name: TYPE # description
  errors:
    - ERROR_TYPE # when thrown
  
requirements:
  functional:
    - REQUIREMENT_1
  performance:
    - latency: <VALUE
    - throughput: >VALUE
  quality:
    - test_coverage: >95%
    - documentation: complete

implementation_spec:
  language: Rust
  patterns: [PATTERN_LIST]
  restrictions:
    - NO_UNWRAP
    - NO_PANIC
    - NO_TODO

test_spec:
  unit_tests:
    - test_name: expected_behavior
  integration_tests:
    - test_name: expected_behavior
  benchmarks:
    - metric: target_value

example:
  ```rust
  // Actual code example
  ```
```

---

## 🏗️ COMPLETE COMPONENT SPECIFICATIONS

### COMPONENT: RiskManager

```yaml
component_id: RISK_001
component_name: RiskManager
owner: Quinn
dependencies: []
phase: 2

contract:
  inputs:
    - position: Position # Position to validate
    - market_data: MarketData # Current market state
    - portfolio: Portfolio # Current portfolio state
  outputs:
    - validation: RiskValidation # Approval or rejection with reasons
  errors:
    - RiskLimitExceeded # When limits breached
    - InsufficientMargin # When margin too low
    - CorrelationTooHigh # When correlation >0.7

requirements:
  functional:
    - MUST validate position size <2% of portfolio
    - MUST check correlation <0.7 with existing positions
    - MUST verify margin requirements
    - MUST calculate liquidation price
    - MUST enforce stop loss
  performance:
    - latency: <100ns
    - throughput: >1M checks/sec
  quality:
    - test_coverage: 100% # Critical component
    - documentation: complete
    - no_fake_implementations: true

implementation_spec:
  language: Rust
  patterns: [Builder, Strategy, CircuitBreaker]
  restrictions:
    - NEVER approve if circuit breaker tripped
    - NEVER allow position without stop loss
    - NEVER exceed position limits
    - ALWAYS log risk decisions

test_spec:
  unit_tests:
    - test_position_size_limit: rejects >2%
    - test_correlation_check: rejects >0.7
    - test_margin_validation: requires 110%
    - test_liquidation_calc: accurate to 0.01%
  integration_tests:
    - test_with_portfolio: validates portfolio risk
    - test_with_market_data: uses real-time data
  benchmarks:
    - latency: <100ns
    - throughput: >1M/sec

example:
  ```rust
  pub struct RiskManager {
      limits: RiskLimits,
      portfolio: Arc<RwLock<Portfolio>>,
      circuit_breaker: CircuitBreaker,
  }
  
  impl RiskManager {
      pub fn validate_position(&self, position: &Position) -> Result<RiskValidation> {
          // Check circuit breaker first
          if self.circuit_breaker.is_tripped() {
              return Err(RiskError::CircuitBreakerTripped);
          }
          
          // Validate position size
          let portfolio_value = self.portfolio.read().unwrap().total_value();
          let position_value = position.size * position.price;
          let position_percentage = position_value / portfolio_value;
          
          if position_percentage > 0.02 {
              return Ok(RiskValidation::Rejected {
                  reason: "Position exceeds 2% limit".to_string(),
                  suggested_size: portfolio_value * 0.02 / position.price,
              });
          }
          
          // Check correlation
          let correlation = self.calculate_correlation(position);
          if correlation > 0.7 {
              return Ok(RiskValidation::Rejected {
                  reason: "Correlation too high".to_string(),
                  suggested_size: 0.0,
              });
          }
          
          // Validate margin
          let margin_required = self.calculate_margin(position);
          let margin_available = self.portfolio.read().unwrap().available_margin();
          
          if margin_available < margin_required * 1.1 {
              return Ok(RiskValidation::Rejected {
                  reason: "Insufficient margin".to_string(),
                  suggested_size: margin_available / (margin_required * 1.1),
              });
          }
          
          Ok(RiskValidation::Approved {
              position_id: position.id,
              risk_score: self.calculate_risk_score(position),
              stop_loss: self.calculate_stop_loss(position),
              take_profit: self.calculate_take_profit(position),
          })
      }
  }
  ```

validation_criteria:
  - ALL tests must pass
  - Latency must be <100ns
  - No unwrap() calls
  - No panic!() calls
  - 100% branch coverage
```

---

### COMPONENT: FeeManager

```yaml
component_id: FEE_001
component_name: FeeManager
owner: Casey
dependencies: [RISK_001]
phase: 5

contract:
  inputs:
    - order: Order # Order to calculate fees for
    - exchange: ExchangeId # Target exchange
    - vip_tier: VipTier # Current VIP status
  outputs:
    - fee_calculation: FeeCalculation # Complete fee breakdown
  errors:
    - ExchangeNotSupported # Unknown exchange
    - FeeDataStale # Fee data too old

requirements:
  functional:
    - MUST track real-time fee structures
    - MUST optimize maker vs taker
    - MUST consider VIP tiers
    - MUST include funding rates
    - MUST predict slippage
  performance:
    - latency: <50ns
    - accuracy: >99.9%
  quality:
    - test_coverage: >95%
    - documentation: complete

implementation_spec:
  language: Rust
  patterns: [Strategy, Cache, Observer]
  restrictions:
    - CACHE fee data for performance
    - UPDATE fees every 60 seconds
    - NEVER use stale data >5 minutes

test_spec:
  unit_tests:
    - test_binance_fees: validates Binance fee calc
    - test_kraken_fees: validates Kraken fee calc
    - test_vip_tiers: applies discounts correctly
    - test_funding_rates: includes funding costs
  integration_tests:
    - test_with_real_exchanges: live fee validation
  benchmarks:
    - latency: <50ns
    - cache_hit_rate: >99%

example:
  ```rust
  pub struct FeeManager {
      fee_cache: Arc<RwLock<HashMap<ExchangeId, FeeStructure>>>,
      funding_tracker: FundingRateTracker,
      slippage_model: SlippageModel,
      last_update: Arc<RwLock<HashMap<ExchangeId, Instant>>>,
  }
  
  impl FeeManager {
      pub fn calculate_total_cost(&self, order: &Order) -> Result<FeeCalculation> {
          // Get current fee structure
          let fee_structure = self.get_fee_structure(order.exchange)?;
          
          // Check data freshness
          if self.is_stale(order.exchange) {
              self.update_fees(order.exchange).await?;
          }
          
          // Base fee calculation
          let base_fee = match order.order_type {
              OrderType::Limit => fee_structure.maker_fee,
              OrderType::Market => fee_structure.taker_fee,
          };
          
          // Apply VIP discount
          let vip_discount = self.get_vip_discount(order.exchange, order.vip_tier);
          let discounted_fee = base_fee * (1.0 - vip_discount);
          
          // Calculate funding if applicable
          let funding_cost = if order.is_perpetual {
              self.funding_tracker.calculate_cost(order)
          } else {
              0.0
          };
          
          // Estimate slippage
          let slippage = self.slippage_model.estimate(order);
          
          Ok(FeeCalculation {
              exchange_fee: discounted_fee * order.value,
              funding_cost,
              estimated_slippage: slippage,
              network_fee: self.get_network_fee(order.asset),
              total_cost: discounted_fee * order.value + funding_cost + slippage,
              break_even_price: self.calculate_breakeven(order, total_cost),
          })
      }
  }
  ```

validation_criteria:
  - Fee accuracy within 0.01%
  - Latency <50ns
  - Cache working properly
  - All exchanges supported
```

---

## 📊 TASK SPECIFICATION TEMPLATE

Each task MUST be documented with:

```yaml
task_id: TASK_X.Y.Z
task_name: NAME
phase: PHASE_NUMBER
owner: AGENT_NAME
dependencies: [TASK_IDS]
estimated_hours: NUMBER

inputs:
  required:
    - name: TYPE # source
  optional:
    - name: TYPE # default_value

outputs:
  deliverables:
    - name: TYPE # destination
  artifacts:
    - name: FORMAT # location

implementation_steps:
  1. STEP_DESCRIPTION
     ```rust
     // Code if applicable
     ```
  2. STEP_DESCRIPTION

success_criteria:
  - CRITERION_1: measurable_target
  - CRITERION_2: measurable_target

test_requirements:
  - TEST_TYPE: what_to_test

example_usage:
  ```rust
  // How to use the deliverable
  ```
```

---

## 🔄 DATA FLOW SPECIFICATIONS

```yaml
flow_id: FLOW_001
flow_name: Market Data Ingestion
components: [WEBSOCKET, VALIDATOR, PROCESSOR, STORAGE]

pipeline:
  - step: WebSocket receives data
    input: ExchangeWebSocket
    output: RawMarketData
    latency: <10μs
    
  - step: Validator checks quality
    input: RawMarketData
    output: ValidatedData | ValidationError
    latency: <50μs
    
  - step: Processor extracts features
    input: ValidatedData
    output: MarketFeatures
    latency: <200μs
    
  - step: Storage persists data
    input: MarketFeatures
    output: StorageConfirmation
    latency: <100μs

total_latency: <360μs
error_handling: circuit_breaker
monitoring: prometheus_metrics
```

---

## 🎮 LLM INTERACTION PROTOCOLS

### For Implementation Tasks

```yaml
llm_instruction_format:
  task: IMPLEMENT_COMPONENT
  component_id: COMPONENT_ID
  requirements: 
    - Use CONTRACT section for interface
    - Follow IMPLEMENTATION_SPEC exactly
    - Include all TEST_SPEC tests
    - Meet PERFORMANCE requirements
  
  deliverables:
    - Complete Rust implementation
    - Unit tests with >95% coverage
    - Integration tests
    - Performance benchmarks
    - Documentation
  
  validation:
    - Run: cargo test --all
    - Run: cargo bench
    - Run: cargo clippy -- -D warnings
    - Check: No TODO, unwrap, panic
```

### For Review Tasks

```yaml
llm_instruction_format:
  task: REVIEW_COMPONENT
  component_id: COMPONENT_ID
  checklist:
    - Contract compliance
    - Requirement satisfaction
    - Performance targets met
    - Test coverage adequate
    - Documentation complete
    - No fake implementations
  
  output_format:
    compliant: true/false
    issues: [LIST]
    suggestions: [LIST]
```

---

## 📐 ARCHITECTURAL PATTERNS

### Pattern: Circuit Breaker

```yaml
pattern_id: PATTERN_001
pattern_name: Circuit Breaker
use_cases: [All external calls, Risk management]

specification:
  states: [Closed, Open, HalfOpen]
  transitions:
    - from: Closed, to: Open, condition: failure_count > threshold
    - from: Open, to: HalfOpen, condition: timeout_expired
    - from: HalfOpen, to: Closed, condition: success
    - from: HalfOpen, to: Open, condition: failure

implementation:
  ```rust
  pub struct CircuitBreaker {
      state: Arc<RwLock<State>>,
      failure_count: Arc<AtomicU32>,
      threshold: u32,
      timeout: Duration,
      last_failure: Arc<RwLock<Instant>>,
  }
  
  impl CircuitBreaker {
      pub fn call<F, T>(&self, f: F) -> Result<T>
      where
          F: FnOnce() -> Result<T>
      {
          match *self.state.read().unwrap() {
              State::Open => {
                  if self.should_attempt_reset() {
                      *self.state.write().unwrap() = State::HalfOpen;
                  } else {
                      return Err(CircuitBreakerError::Open);
                  }
              }
              State::HalfOpen => {
                  // Attempt single call
              }
              State::Closed => {
                  // Normal operation
              }
          }
          
          // Execute and track
          match f() {
              Ok(result) => {
                  self.on_success();
                  Ok(result)
              }
              Err(e) => {
                  self.on_failure();
                  Err(e)
              }
          }
      }
  }
  ```
```

---

## 🔍 VALIDATION RULES FOR LLMS

```yaml
validation_rules:
  code_quality:
    - NO todo!() macro
    - NO unimplemented!() macro
    - NO panic!() except in tests
    - NO unwrap() except with justification
    - NO println!() in production code
    
  performance:
    - ALL functions < specified latency
    - Memory usage < specified limit
    - No unbounded allocations
    - Use SIMD where specified
    
  testing:
    - Unit test coverage > 95%
    - All edge cases tested
    - Integration tests pass
    - Benchmarks meet targets
    
  documentation:
    - Every public function documented
    - Examples provided
    - Error conditions described
    - Performance characteristics noted
```

---

## 📝 UPDATE PROTOCOL

```yaml
update_protocol:
  when: After each component implementation
  what:
    - Update component status
    - Add actual performance metrics
    - Document any deviations
    - Update dependencies
  
  format:
    ```yaml
    component_id: COMPONENT_ID
    status: IMPLEMENTED
    actual_metrics:
      latency: MEASURED_VALUE
      throughput: MEASURED_VALUE
    deviations:
      - DEVIATION: justification
    ```
```

---

## 🎯 SUCCESS METRICS FOR LLM IMPLEMENTATION

```yaml
success_metrics:
  per_component:
    - Contract satisfied: 100%
    - Tests passing: 100%
    - Performance met: 100%
    - Documentation complete: 100%
    
  per_phase:
    - All components implemented
    - Integration tests passing
    - Performance targets met
    - No fake implementations
    
  overall:
    - System functional: true
    - Performance targets: met
    - Test coverage: >95%
    - Documentation: complete
```

---

*This document is optimized for LLM parsing and implementation.*
*Use structured sections for precise implementation.*
*Update after each component completion.*