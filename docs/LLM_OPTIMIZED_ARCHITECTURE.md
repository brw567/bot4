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

## 📊 IMPLEMENTATION STATUS

### Phase 1: Core Infrastructure ✅ COMPLETE (2025-08-17)

```yaml
completed_components:
  - component_id: INFRA_001
    name: CircuitBreaker
    status: COMPLETE
    location: rust_core/crates/infrastructure/src/circuit_breaker.rs
    validated_by: [Sophia/ChatGPT - APPROVED]
    features:
      - Lock-free with AtomicU64 (no RwLock in hot paths)
      - Global state derivation from components
      - Half-Open token limiting with CAS
      - Sliding window mechanics
      - Panic-safe event callbacks
    performance:
      - latency: <1μs overhead
      - state_transitions: atomic
    
  - component_id: DB_001
    name: DatabaseSchema
    status: COMPLETE
    location: sql/001_core_schema.sql
    features: [TimescaleDB, risk_constraints, mandatory_stop_loss, 2%_position_limits]
    tables: 11
    hypertables: 4
    
  - component_id: WS_001
    name: WebSocketInfrastructure
    status: COMPLETE
    location: rust_core/crates/websocket/
    validated_by: [Nexus/Grok - VERIFIED]
    performance:
      - latency: p99 @ 0.95ms
      - throughput: 12000_msg_per_sec_sustained
      - auto_reconnect: exponential_backoff
    
  - component_id: ORDER_001
    name: OrderManagementSystem
    status: COMPLETE
    location: rust_core/crates/order_management/
    validated_by: [Sophia/ChatGPT - APPROVED]
    features:
      - Atomic state machine (no invalid states)
      - Smart order routing (BestPrice, LowestFee, SmartRoute)
      - Position tracking with real-time P&L
    performance:
      - processing: p99 @ 98μs
      - state_transitions: lock-free
      - throughput: 10000_orders_per_sec_burst
    
  - component_id: RISK_001
    name: RiskEngine
    status: COMPLETE
    location: rust_core/crates/risk_engine/
    validated_by: [Sophia/ChatGPT - APPROVED, Nexus/Grok - VERIFIED]
    features:
      - Pre-trade checks with parallel validation
      - Mandatory stop-loss enforcement
      - Position limits (2% max)
      - Correlation analysis (0.7 max)
      - Drawdown tracking (15% max)
      - Emergency kill switch
    performance:
      - pre_trade_checks: p99 @ 10μs
      - throughput: 120000_checks_per_sec
      - contention_reduction: 5-10x_via_atomics
    
  - component_id: BENCH_001
    name: PerformanceBenchmarks
    status: COMPLETE
    location: rust_core/benches/
    files:
      - risk_engine_bench.rs
      - order_management_bench.rs
    features:
      - Criterion with 100k+ samples
      - Hardware counter collection (perf stat)
      - Automatic latency assertions
      - CI artifact generation
    
  - component_id: CI_001
    name: CIPipeline
    status: COMPLETE
    location: .github/workflows/ci.yml
    gates:
      - no_fakes_validation
      - coverage_95_percent
      - performance_targets
      - security_audit
      - clippy_warnings_denied

validation_results:
  sophia_chatgpt:
    verdict: APPROVED
    date: 2025-08-17
    quote: "green light to merge PR #6"
    fixes_applied: 9_critical_issues_resolved
    
  nexus_grok:
    verdict: VERIFIED
    date: 2025-08-17
    quote: "internal latencies and throughputs substantiated"
    performance_validated: all_targets_met

phase_1_metrics:
  code_lines: ~5000
  test_coverage: 95%_ready
  compilation_warnings: 0
  security_issues: 0
  performance_regressions: 0
  technical_debt: minimal
  
  - component_id: ORDER_001
    name: OrderManagementSystem
    status: COMPLETE
    location: rust_core/crates/order_management/
    features: [atomic_state_machine, smart_routing, position_tracking]
    
  - component_id: RISK_001
    name: RiskEngine
    status: COMPLETE
    location: rust_core/crates/risk_engine/
    features: [pre_trade_checks, correlation_analysis, kill_switch]
    performance: <10μs_checks
```

## 🏗️ COMPLETE COMPONENT SPECIFICATIONS

### ✅ PHASE 1 COMPLETED COMPONENTS (2025-08-17)

### COMPONENT: CircuitBreaker (INFRA_001) ✅

```yaml
component_id: INFRA_001
component_name: CircuitBreaker
owner: Sam
dependencies: []
phase: 1
status: COMPLETE
validated_by: Sophia/ChatGPT - APPROVED

contract:
  inputs:
    - component: String # Component name to protect
    - operation: Future<T> # Operation to execute
  outputs:
    - result: Result<T, CircuitError> # Protected execution result
  errors:
    - CircuitError::Open # Circuit is open
    - CircuitError::HalfOpenExhausted # Half-open tokens exhausted
    - CircuitError::GlobalOpen # Global breaker tripped
    - CircuitError::MinCallsNotMet # Statistical confidence not met

requirements:
  functional:
    - Lock-free operation with AtomicU64
    - Global state derived from components
    - Half-Open token limiting with CAS
    - Sliding window error tracking
    - Panic-safe event callbacks
  performance:
    - latency: <1μs overhead
    - state_transitions: atomic
    - memory: zero allocations in hot path
  quality:
    - test_coverage: >95%
    - no_rwlock_in_hot_path: true
    - cache_padded_atomics: true

implementation_achieved:
  - Replaced all RwLock with AtomicU64
  - CAS-based Half-Open token acquisition
  - RAII CallGuard for automatic outcome recording
  - Clock trait for testability
  - ArcSwap for hot configuration reloading
```

### COMPONENT: RiskEngine (RISK_001) ✅

```yaml
component_id: RISK_001
component_name: RiskEngine
owner: Quinn
dependencies: [INFRA_001]
phase: 1
status: COMPLETE
validated_by: [Sophia/ChatGPT - APPROVED, Nexus/Grok - VERIFIED]

contract:
  inputs:
    - order: Order # Order to validate
    - portfolio: Portfolio # Current portfolio state
  outputs:
    - result: RiskCheckResult # Approval or rejection
  errors:
    - PositionLimitExceeded # >2% position
    - StopLossRequired # No stop loss set
    - CorrelationTooHigh # >0.7 correlation
    - DrawdownExceeded # >15% drawdown

requirements:
  functional:
    - 2% max position size (Quinn's rule)
    - Mandatory stop-loss enforcement
    - 0.7 max correlation between positions
    - 15% max drawdown with kill switch
    - Emergency stop with recovery plans
  performance:
    - pre_trade_checks: <10μs (p99)
    - throughput: >100,000 checks/sec
    - parallel_validation: true
  quality:
    - benchmarks: 100k+ samples
    - perf_stat_validation: true

implementation_achieved:
  - p99 latency: 10μs (validated)
  - Throughput: 120,000 checks/sec
  - Lock-free with 5-10x contention reduction
  - Kill switch with multiple trip conditions
  - Recovery plans (standard and aggressive)
```

### COMPONENT: OrderManagement (ORDER_001) ✅

```yaml
component_id: ORDER_001
component_name: OrderManagementSystem
owner: Casey
dependencies: [RISK_001]
phase: 1
status: COMPLETE
validated_by: Sophia/ChatGPT - APPROVED

contract:
  inputs:
    - order: Order # Order to process
    - strategy: RoutingStrategy # How to route
  outputs:
    - order_id: OrderId # Unique identifier
    - route: ExchangeRoute # Selected exchange
  errors:
    - InvalidOrderState # State transition invalid
    - RoutingFailed # No available route

requirements:
  functional:
    - Atomic state machine (no invalid states)
    - Smart order routing strategies
    - Real-time P&L tracking
    - Position management
  performance:
    - processing: <100μs (p99)
    - throughput: >10,000 orders/sec burst
    - state_transitions: lock-free
  quality:
    - no_invalid_states: guaranteed
    - complete_lifecycle: tracked

implementation_achieved:
  - p99 latency: 98μs (validated)
  - 10,000 orders/sec burst confirmed
  - Lock-free state transitions with AtomicU8
  - Smart routing: BestPrice, LowestFee, SmartRoute
```

### COMPONENT: WebSocketInfrastructure (WS_001) ✅

```yaml
component_id: WS_001
component_name: WebSocketInfrastructure
owner: Jordan
dependencies: []
phase: 1
status: COMPLETE
validated_by: Nexus/Grok - VERIFIED

requirements:
  functional:
    - Auto-reconnection with exponential backoff
    - Message routing and type safety
    - Connection pooling with load balancing
  performance:
    - latency: <1ms (p99)
    - throughput: >10,000 msg/sec
    - reconnection: <5s
  quality:
    - zero_message_loss: true
    - ordered_delivery: guaranteed

implementation_achieved:
  - p99 latency: 0.95ms
  - Throughput: 12,000 msg/sec sustained
  - Auto-reconnect with backoff implemented
  - Connection pooling operational
```

### COMPONENT: DatabaseSchema (DB_001) ✅

```yaml
component_id: DB_001
component_name: DatabaseSchema
owner: Avery
dependencies: []
phase: 1
status: COMPLETE

implementation:
  - 11 core tables with constraints
  - TimescaleDB hypertables for time-series
  - Mandatory stop-loss enforcement
  - 2% position size limits in schema
  - Risk constraints at database level
```

---

### COMPONENT: RiskManager (Phase 2 Planning)

```yaml
component_id: RISK_002
component_name: RiskManager_V2
owner: Quinn
dependencies: [RISK_001]
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
    - latency: <1ms
    - throughput: >10K checks/sec
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
    - latency: <1ms
    - throughput: >10K/sec

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

### COMPONENT: RegimeDetectionSystem

```yaml
component_id: REGIME_001
component_name: RegimeDetectionSystem
owner: Morgan
dependencies: [DATA_001]
phase: 3.5

contract:
  inputs:
    - market_data: MarketData # Current market conditions
    - sentiment_data: SentimentData # Fear/Greed, social sentiment
    - onchain_data: OnChainMetrics # Blockchain metrics
  outputs:
    - regime: MarketRegime # Detected market regime
    - confidence: f64 # Detection confidence (0.0-1.0)
    - transition_signal: Option<TransitionSignal> # Regime change signal
  errors:
    - InsufficientData # Not enough data for detection
    - LowConfidence # Confidence below threshold
    - ModelFailure # One or more models failed

requirements:
  functional:
    - MUST use 5-model consensus (HMM, LSTM, XGBoost, Microstructure, OnChain)
    - MUST achieve >90% accuracy on historical data
    - MUST require 3+ models agreement for regime change
    - MUST detect all 5 regime types (BullEuphoria, BullNormal, Choppy, Bear, BlackSwan)
    - MUST provide transition signals with lead time
  performance:
    - latency: <1 second  # ML consensus requirement
    - accuracy: >90%
    - false_positive_rate: <5%
  quality:
    - test_coverage: >95%
    - backtested_years: >5
    - model_validation: cross-validated

implementation_spec:
  language: Rust
  patterns: [Strategy, Observer, Consensus]
  ml_frameworks: [ONNX, Candle]
  restrictions:
    - NEVER change regime without consensus
    - NEVER switch during high volatility
    - LIMIT switches to 1 per 4 hours
    - ALWAYS log regime changes

test_spec:
  unit_tests:
    - test_hmm_detection: validates HMM model
    - test_lstm_classifier: validates LSTM accuracy
    - test_consensus_mechanism: requires 3+ agreement
    - test_all_regimes: detects each regime type
  integration_tests:
    - test_with_historical_data: >90% accuracy on 5 years
    - test_regime_transitions: smooth transitions
  benchmarks:
    - detection_latency: <1 second
    - accuracy: >90%

example:
  ```rust
  pub struct RegimeDetectionSystem {
      hmm_detector: HiddenMarkovModel,
      lstm_classifier: LSTMRegimeClassifier,
      xgboost_detector: XGBoostDetector,
      microstructure_analyzer: MicrostructureAnalyzer,
      onchain_analyzer: OnChainAnalyzer,
      consensus_threshold: f64,  // 0.75
      min_models_agreement: usize,  // 3
  }
  
  impl RegimeDetectionSystem {
      pub fn detect_regime(&self, data: &MarketData) -> Result<RegimeDetection> {
          // Collect predictions from all models
          let predictions = vec![
              (self.hmm_detector.predict(data)?, 0.25),
              (self.lstm_classifier.predict(data)?, 0.30),
              (self.xgboost_detector.predict(data)?, 0.20),
              (self.microstructure_analyzer.analyze(data)?, 0.15),
              (self.onchain_analyzer.analyze(data)?, 0.10),
          ];
          
          // Calculate weighted consensus
          let consensus = self.calculate_weighted_consensus(&predictions)?;
          
          // Validate confidence threshold
          if consensus.confidence < self.consensus_threshold {
              return Err(RegimeError::LowConfidence(consensus.confidence));
          }
          
          // Check minimum agreement
          let agreeing_models = predictions.iter()
              .filter(|(pred, _)| pred.regime == consensus.regime)
              .count();
          
          if agreeing_models < self.min_models_agreement {
              return Err(RegimeError::InsufficientConsensus);
          }
          
          Ok(RegimeDetection {
              regime: consensus.regime,
              confidence: consensus.confidence,
              transition_signal: self.check_transition(&consensus.regime)?,
          })
      }
  }
  ```

validation_criteria:
  - Consensus from 3+ models
  - Confidence >75%
  - Historical accuracy >90%
  - No rapid switching
```

---

### COMPONENT: EmotionFreeValidator

```yaml
component_id: EMOTION_001
component_name: EmotionFreeValidator
owner: Quinn
dependencies: [RISK_001, REGIME_001]
phase: 3.5

contract:
  inputs:
    - signal: TradingSignal # Signal to validate
    - context: TradingContext # Current market context
    - statistics: SignalStatistics # Historical performance
  outputs:
    - decision: ValidationDecision # Approve/Reject with reasoning
    - metrics: ValidationMetrics # Statistical measures
  errors:
    - EmotionalBiasDetected # Emotional pattern found
    - StatisticallyInsignificant # p-value > 0.05
    - NegativeExpectedValue # EV <= 0
    - InsufficientSharpe # Sharpe < 2.0

requirements:
  functional:
    - MUST validate statistical significance (p < 0.05)
    - MUST calculate expected value (EV > 0)
    - MUST verify Sharpe ratio (> 2.0)
    - MUST check confidence level (> 75%)
    - MUST detect emotional biases
    - MUST block all non-mathematical decisions
  performance:
    - latency: <100ms  # Validation layer
    - validation_rate: 100%
  quality:
    - test_coverage: 100% # Critical component
    - false_negative_rate: 0% # Never allow emotional trades

implementation_spec:
  language: Rust
  patterns: [Validator, ChainOfResponsibility]
  restrictions:
    - NEVER approve emotional decisions
    - NEVER skip validation
    - ALWAYS require mathematical proof
    - ALWAYS log rejections

test_spec:
  unit_tests:
    - test_statistical_significance: rejects p > 0.05
    - test_expected_value: rejects EV <= 0
    - test_sharpe_validation: rejects Sharpe < 2.0
    - test_confidence_check: rejects < 75%
    - test_bias_detection: catches all bias types
  integration_tests:
    - test_with_historical_signals: 0% emotional trades
    - test_with_regime_context: adapts to regime
  benchmarks:
    - validation_latency: <100ms
    - rejection_accuracy: 100%

example:
  ```rust
  pub struct EmotionFreeValidator {
      significance_threshold: f64,  // 0.05
      min_expected_value: f64,      // 0.0
      min_sharpe_ratio: f64,        // 2.0
      min_confidence: f64,          // 0.75
      bias_detector: BiasDetector,
      statistical_validator: StatisticalValidator,
  }
  
  impl EmotionFreeValidator {
      pub fn validate(&self, signal: &TradingSignal) -> Result<ValidationDecision> {
          // First check for emotional biases
          if let Some(bias) = self.bias_detector.detect(signal) {
              return Ok(ValidationDecision::Reject {
                  reason: format!("Emotional bias detected: {:?}", bias),
                  bias_type: Some(bias),
              });
          }
          
          // Statistical significance test
          let p_value = self.statistical_validator.calculate_p_value(signal);
          if p_value > self.significance_threshold {
              return Ok(ValidationDecision::Reject {
                  reason: format!("Not statistically significant: p={:.4}", p_value),
                  bias_type: None,
              });
          }
          
          // Expected value calculation
          let expected_value = self.calculate_expected_value(signal);
          if expected_value <= self.min_expected_value {
              return Ok(ValidationDecision::Reject {
                  reason: format!("Negative expected value: {:.4}", expected_value),
                  bias_type: None,
              });
          }
          
          // Sharpe ratio validation
          let sharpe = self.calculate_sharpe_ratio(signal);
          if sharpe < self.min_sharpe_ratio {
              return Ok(ValidationDecision::Reject {
                  reason: format!("Insufficient Sharpe ratio: {:.2}", sharpe),
                  bias_type: None,
              });
          }
          
          // Confidence check
          if signal.confidence < self.min_confidence {
              return Ok(ValidationDecision::Reject {
                  reason: format!("Low confidence: {:.2}%", signal.confidence * 100.0),
                  bias_type: None,
              });
          }
          
          Ok(ValidationDecision::Approve {
              signal_id: signal.id,
              expected_value,
              sharpe_ratio: sharpe,
              p_value,
              reasoning: "All mathematical criteria satisfied".to_string(),
          })
      }
  }
  ```

validation_criteria:
  - Zero emotional trades
  - 100% mathematical validation
  - All biases blocked
  - Statistical rigor enforced
```

---

### COMPONENT: PsychologicalBiasBlocker

```yaml
component_id: BIAS_001
component_name: PsychologicalBiasBlocker
owner: Morgan
dependencies: [EMOTION_001]
phase: 3.5

contract:
  inputs:
    - context: TradingContext # Current trading state
    - history: TradingHistory # Recent trading history
    - market_sentiment: MarketSentiment # Social/fear indicators
  outputs:
    - biases: Vec<BiasDetection> # Detected biases
    - recommendations: Vec<BiasRecommendation> # Corrective actions
  errors:
    - AnalysisError # Cannot analyze context

requirements:
  functional:
    - MUST detect FOMO (fear of missing out)
    - MUST detect revenge trading patterns
    - MUST detect overconfidence after wins
    - MUST detect loss aversion behavior
    - MUST detect confirmation bias
    - MUST provide corrective actions
  performance:
    - latency: <50ms  # Bias detection
    - detection_rate: >99%
  quality:
    - test_coverage: >95%
    - false_positive_rate: <1%

implementation_spec:
  language: Rust
  patterns: [Strategy, Observer]
  restrictions:
    - ALWAYS block detected biases
    - NEVER allow override
    - ENFORCE cooldown periods
    - LOG all detections

test_spec:
  unit_tests:
    - test_fomo_detection: catches chase trades
    - test_revenge_trading: detects immediate re-entry
    - test_overconfidence: catches position oversizing
    - test_loss_aversion: detects stop loss avoidance
    - test_confirmation_bias: requires multiple sources
  integration_tests:
    - test_with_trading_history: detects patterns
    - test_with_market_sentiment: uses fear/greed
  benchmarks:
    - detection_latency: <50ms
    - accuracy: >99%

example:
  ```rust
  pub struct PsychologicalBiasBlocker {
      fomo_detector: FOMODetector,
      revenge_detector: RevengeTradingDetector,
      overconfidence_detector: OverconfidenceDetector,
      loss_aversion_detector: LossAversionDetector,
      confirmation_bias_detector: ConfirmationBiasDetector,
  }
  
  impl PsychologicalBiasBlocker {
      pub fn detect_biases(&self, context: &TradingContext) -> Vec<BiasDetection> {
          let mut biases = Vec::new();
          
          // FOMO Detection
          if self.fomo_detector.detect(context) {
              biases.push(BiasDetection::FOMO {
                  trigger: "Rapid price rise with high sentiment".to_string(),
                  action: BiasAction::BlockTrade,
                  cooldown: Duration::hours(1),
              });
          }
          
          // Revenge Trading
          if self.revenge_detector.detect(context) {
              biases.push(BiasDetection::RevengeTrade {
                  trigger: "Recent loss followed by immediate re-entry".to_string(),
                  action: BiasAction::EnforceCooldown,
                  cooldown: Duration::hours(4),
              });
          }
          
          // Overconfidence - ENHANCED DETECTION
          if self.overconfidence_detector.detect(context) {
              // Detect based on multiple factors:
              // 1. Win streak > 5 consecutive wins
              // 2. Position sizes increasing > 50% from baseline
              // 3. Ignoring risk warnings
              // 4. Reduced stop loss distances
              let win_streak = context.consecutive_wins;
              let position_growth = context.avg_position_size / context.baseline_size;
              let risk_overrides = context.risk_warning_overrides;
              
              if win_streak > 5 || position_growth > 1.5 || risk_overrides > 0 {
                  biases.push(BiasDetection::Overconfidence {
                      trigger: format!("Win streak: {}, Size growth: {:.1}x, Overrides: {}", 
                                     win_streak, position_growth, risk_overrides),
                      action: BiasAction::CapPositionSize,
                      max_size: context.portfolio_value * 0.01,  // Cap at 1% max
                      cooldown: Duration::hours(4),  // Force 4-hour cooldown
                  });
              }
          }
          
          // Loss Aversion
          if self.loss_aversion_detector.detect(context) {
              biases.push(BiasDetection::LossAversion {
                  trigger: "Holding losing position beyond stop".to_string(),
                  action: BiasAction::ForceStopLoss,
              });
          }
          
          // Confirmation Bias
          if self.confirmation_bias_detector.detect(context) {
              biases.push(BiasDetection::ConfirmationBias {
                  trigger: "Single signal source".to_string(),
                  action: BiasAction::RequireMultipleConfirmations,
                  min_confirmations: 3,
              });
          }
          
          biases
      }
  }
  ```

validation_criteria:
  - All bias types detected
  - Corrective actions enforced
  - No override possible
  - Cooldowns respected
```

---

### COMPONENT: RegimeStrategyAllocator

```yaml
component_id: ALLOCATOR_001
component_name: RegimeStrategyAllocator
owner: Sam
dependencies: [REGIME_001, EMOTION_001]
phase: 3.5

contract:
  inputs:
    - regime: MarketRegime # Current market regime
    - portfolio: Portfolio # Current portfolio state
    - available_strategies: Vec<Strategy> # Available strategies
  outputs:
    - allocation: StrategyAllocation # Weighted strategy mix
    - target_returns: Range<f64> # Expected returns for regime
  errors:
    - InvalidRegime # Unknown regime type
    - InsufficientStrategies # Not enough strategies available

requirements:
  functional:
    - MUST allocate strategies per regime type
    - MUST achieve regime-specific return targets
    - MUST adapt to regime transitions
    - MUST maintain diversification
    - MUST respect risk limits
  performance:
    - allocation_time: <100ms  # Strategy selection
    - rebalance_time: <1 second
  quality:
    - test_coverage: >95%
    - backtested_returns: match targets ±10%

implementation_spec:
  language: Rust
  patterns: [Strategy, Factory]
  restrictions:
    - NEVER exceed regime risk limits
    - ALWAYS maintain diversification
    - GRADUALLY transition allocations
    - LOG all allocation changes

test_spec:
  unit_tests:
    - test_bull_euphoria_allocation: 30-50% monthly target
    - test_bull_normal_allocation: 15-25% monthly target
    - test_choppy_allocation: 8-15% monthly target
    - test_bear_allocation: 5-10% monthly target
    - test_black_swan_allocation: capital preservation
  integration_tests:
    - test_regime_transitions: smooth rebalancing
    - test_with_portfolio: respects constraints
  benchmarks:
    - allocation_speed: <100ms
    - target_accuracy: ±10%

example:
  ```rust
  pub struct RegimeStrategyAllocator {
      regime_allocations: HashMap<MarketRegime, StrategyWeights>,
      transition_manager: TransitionManager,
  }
  
  impl RegimeStrategyAllocator {
      pub fn get_allocation(&self, regime: &MarketRegime) -> StrategyAllocation {
          match regime {
              MarketRegime::BullEuphoria { .. } => {
                  StrategyAllocation {
                      leveraged_momentum: 0.40,
                      breakout_trading: 0.30,
                      launchpad_sniping: 0.20,
                      memecoin_rotation: 0.10,
                      target_monthly_return: 0.30..0.50,
                      max_leverage: 5.0,
                      risk_multiplier: 1.5,
                  }
              },
              MarketRegime::BullNormal { .. } => {
                  StrategyAllocation {
                      trend_following: 0.35,
                      swing_trading: 0.30,
                      defi_yield: 0.20,
                      arbitrage: 0.15,
                      target_monthly_return: 0.15..0.25,
                      max_leverage: 3.0,
                      risk_multiplier: 1.0,
                  }
              },
              MarketRegime::Choppy { .. } => {
                  StrategyAllocation {
                      market_making: 0.35,
                      mean_reversion: 0.30,
                      arbitrage: 0.25,
                      funding_rates: 0.10,
                      target_monthly_return: 0.08..0.15,
                      max_leverage: 2.0,
                      risk_multiplier: 0.7,
                  }
              },
              MarketRegime::Bear { .. } => {
                  StrategyAllocation {
                      short_selling: 0.30,
                      stable_farming: 0.30,
                      arbitrage_only: 0.30,
                      cash_reserve: 0.10,
                      target_monthly_return: 0.05..0.10,
                      max_leverage: 1.0,
                      risk_multiplier: 0.5,
                  }
              },
              MarketRegime::BlackSwan { .. } => {
                  StrategyAllocation {
                      emergency_hedge: 0.50,
                      stable_coins: 0.40,
                      gold_tokens: 0.10,
                      target_monthly_return: -0.05..0.00,
                      max_leverage: 0.0,
                      risk_multiplier: 0.1,
                  }
              },
          }
      }
  }
  ```

validation_criteria:
  - Correct allocation per regime
  - Target returns achievable
  - Risk limits respected
  - Smooth transitions
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
    - latency: <1ms  # Fee calculation
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
    - latency: <1ms
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
  - Latency <1ms
  - Cache working properly
  - All exchanges supported
```

---

### COMPONENT: DataPipeline

```yaml
component_id: DATA_001
component_name: DataPipeline
owner: Avery
dependencies: []
phase: 3

contract:
  inputs:
    - exchange_feeds: Vec<ExchangeFeed> # Multiple exchange connections
    - validation_rules: ValidationConfig # Data quality rules
  outputs:
    - market_data: ValidatedMarketData # Clean, validated data
    - data_quality: DataQualityMetrics # Quality scores
  errors:
    - InvalidData # Data fails validation
    - StaleData # Data too old
    - MissingData # Required fields missing

requirements:
  functional:
    - MUST validate all incoming data
    - MUST handle multiple exchange formats
    - MUST detect and filter outliers
    - MUST maintain data continuity
    - MUST provide backfill capability
  performance:
    - latency: <10ms
    - throughput: >100k messages/sec
    - reliability: 99.99%
  quality:
    - test_coverage: >95%
    - documentation: complete
    - circuit_breakers: mandatory

implementation_spec:
  language: Rust
  patterns: [Pipeline, Validator, CircuitBreaker]
  restrictions:
    - ALWAYS validate before forwarding
    - NEVER pass invalid data
    - ALWAYS maintain audit trail
    - ENFORCE circuit breakers

test_spec:
  unit_tests:
    - test_validation: rejects invalid data
    - test_outlier_detection: filters spikes
    - test_format_normalization: handles all exchanges
    - test_backfill: fills gaps correctly
  integration_tests:
    - test_with_live_feeds: handles real data
    - test_failover: switches feeds on failure
  benchmarks:
    - latency: <10ms
    - throughput: >100k/sec

example:
  ```rust
  pub struct DataPipeline {
      validators: Vec<Box<dyn DataValidator>>,
      normalizers: HashMap<ExchangeId, Box<dyn Normalizer>>,
      circuit_breaker: CircuitBreaker,
      quality_tracker: QualityTracker,
  }
  
  impl DataPipeline {
      pub async fn process(&self, raw_data: RawMarketData) -> Result<ValidatedMarketData> {
          // Circuit breaker check
          if self.circuit_breaker.is_tripped() {
              return Err(DataError::CircuitBreakerTripped);
          }
          
          // Normalize exchange-specific format
          let normalized = self.normalizers
              .get(&raw_data.exchange)
              .ok_or(DataError::UnknownExchange)?
              .normalize(&raw_data)?;
          
          // Run all validators
          for validator in &self.validators {
              validator.validate(&normalized)?;
          }
          
          // Check data freshness
          if normalized.timestamp.elapsed() > Duration::from_millis(500) {
              return Err(DataError::StaleData);
          }
          
          // Update quality metrics
          self.quality_tracker.record_success();
          
          Ok(ValidatedMarketData {
              data: normalized,
              quality_score: self.quality_tracker.current_score(),
              validated_at: Instant::now(),
          })
      }
  }
  ```

validation_criteria:
  - All data validated
  - No invalid data passes
  - Circuit breakers work
  - Quality metrics accurate
```

---

### COMPONENT: ExecutionEngine

```yaml
component_id: EXEC_001
component_name: ExecutionEngine
owner: Casey
dependencies: [RISK_001, EMOTION_001]
phase: 10

contract:
  inputs:
    - signal: ValidatedSignal # From EmotionFreeValidator
    - market_data: MarketData # Current market state
    - risk_approval: RiskApproval # From RiskManager
  outputs:
    - order_result: OrderResult # Success or failure
    - execution_metrics: ExecutionMetrics # Slippage, fees, etc
  errors:
    - ExecutionFailed # Order failed
    - RiskRejected # Risk check failed
    - ExchangeError # Exchange issue

requirements:
  functional:
    - MUST validate risk approval
    - MUST optimize execution strategy
    - MUST handle partial fills
    - MUST track slippage
    - MUST retry on failure
  performance:
    - latency: <100μs
    - success_rate: >99%
    - slippage: <0.1%
  quality:
    - test_coverage: >95%
    - documentation: complete
    - circuit_breakers: mandatory

implementation_spec:
  language: Rust
  patterns: [Strategy, CircuitBreaker, Retry]
  restrictions:
    - NEVER execute without risk approval
    - NEVER exceed position limits
    - ALWAYS track execution metrics
    - ENFORCE stop losses

test_spec:
  unit_tests:
    - test_risk_validation: rejects without approval
    - test_execution_strategies: optimizes correctly
    - test_partial_fills: handles properly
    - test_retry_logic: retries on failure
  integration_tests:
    - test_with_exchanges: executes on all exchanges
    - test_failover: switches exchanges on failure
  benchmarks:
    - latency: <100μs
    - success_rate: >99%

example:
  ```rust
  pub struct ExecutionEngine {
      exchange_connectors: HashMap<ExchangeId, Box<dyn ExchangeConnector>>,
      execution_strategies: Vec<Box<dyn ExecutionStrategy>>,
      circuit_breaker: CircuitBreaker,
      retry_policy: RetryPolicy,
  }
  
  impl ExecutionEngine {
      pub async fn execute(&self, signal: &ValidatedSignal) -> Result<OrderResult> {
          // Verify risk approval
          if !signal.risk_approval.is_approved() {
              return Err(ExecutionError::RiskRejected);
          }
          
          // Check circuit breaker
          if self.circuit_breaker.is_tripped() {
              return Err(ExecutionError::CircuitBreakerTripped);
          }
          
          // Select optimal execution strategy
          let strategy = self.select_strategy(signal)?;
          
          // Execute with retries
          let result = self.retry_policy.execute_with_retry(|| async {
              let exchange = self.select_exchange(signal)?;
              let connector = self.exchange_connectors
                  .get(&exchange)
                  .ok_or(ExecutionError::ExchangeNotConfigured)?;
              
              // Place order
              let order = strategy.create_order(signal)?;
              let result = connector.place_order(order).await?;
              
              // Track metrics
              let metrics = ExecutionMetrics {
                  slippage: self.calculate_slippage(&result),
                  fees: result.fees,
                  latency: result.latency,
              };
              
              Ok(OrderResult {
                  order_id: result.order_id,
                  status: result.status,
                  metrics,
              })
          }).await?;
          
          Ok(result)
      }
  }
  ```

validation_criteria:
  - Risk approval enforced
  - Execution optimized
  - Metrics tracked
  - Retries work
```

---

### COMPONENT: PortfolioManager

```yaml
component_id: PORTFOLIO_001
component_name: PortfolioManager
owner: Quinn
dependencies: [RISK_001, ALLOCATOR_001]
phase: 9

contract:
  inputs:
    - positions: Vec<Position> # Current positions
    - market_data: MarketData # Market state
    - regime: MarketRegime # Current regime
  outputs:
    - portfolio_state: PortfolioState # Current state
    - rebalance_signals: Vec<RebalanceSignal> # Needed adjustments
  errors:
    - PortfolioError # Calculation error
    - DataError # Missing data

requirements:
  functional:
    - MUST track all positions
    - MUST calculate real-time P&L
    - MUST monitor correlations
    - MUST suggest rebalancing
    - MUST enforce diversification
  performance:
    - latency: <100ms
    - accuracy: 99.99%
  quality:
    - test_coverage: >95%
    - documentation: complete

implementation_spec:
  language: Rust
  patterns: [Observer, Strategy]
  restrictions:
    - ALWAYS maintain accurate state
    - NEVER exceed risk limits
    - ENFORCE diversification rules
    - TRACK all changes

test_spec:
  unit_tests:
    - test_pnl_calculation: accurate to 0.01%
    - test_correlation_matrix: updates correctly
    - test_rebalancing: suggests optimal changes
    - test_diversification: enforces limits
  integration_tests:
    - test_with_live_data: handles real portfolios
    - test_regime_changes: adapts correctly
  benchmarks:
    - latency: <100ms
    - accuracy: 99.99%

example:
  ```rust
  pub struct PortfolioManager {
      positions: Arc<RwLock<HashMap<String, Position>>>,
      correlation_matrix: Arc<RwLock<CorrelationMatrix>>,
      risk_limits: RiskLimits,
      regime_allocator: RegimeStrategyAllocator,
  }
  
  impl PortfolioManager {
      pub fn analyze(&self) -> Result<PortfolioState> {
          let positions = self.positions.read().unwrap();
          
          // Calculate total value and P&L
          let total_value = positions.values()
              .map(|p| p.current_value())
              .sum();
          
          let total_pnl = positions.values()
              .map(|p| p.unrealized_pnl())
              .sum();
          
          // Update correlation matrix
          let correlations = self.calculate_correlations(&positions)?;
          *self.correlation_matrix.write().unwrap() = correlations;
          
          // Check risk limits
          let risk_metrics = RiskMetrics {
              position_concentration: self.max_position_concentration(&positions),
              correlation_risk: correlations.max_correlation(),
              leverage: self.calculate_leverage(&positions),
          };
          
          // Generate rebalance signals if needed
          let rebalance_signals = if risk_metrics.exceeds_limits(&self.risk_limits) {
              self.generate_rebalance_signals(&positions, &risk_metrics)?
          } else {
              Vec::new()
          };
          
          Ok(PortfolioState {
              total_value,
              total_pnl,
              positions: positions.clone(),
              correlations,
              risk_metrics,
              rebalance_signals,
          })
      }
  }
  ```

validation_criteria:
  - Accurate P&L tracking
  - Correlation monitoring
  - Risk limit enforcement
  - Rebalancing logic correct
```

---

### COMPONENT: GlobalCircuitBreaker

```yaml
component_id: CIRCUIT_001
component_name: GlobalCircuitBreaker
owner: Jordan
dependencies: []
phase: 1

contract:
  inputs:
    - component_id: String # Component making the call
    - call_type: CallType # External API, Database, etc
  outputs:
    - permission: CircuitPermission # Allow or Deny
    - state: CircuitState # Current breaker state
  errors:
    - CircuitOpen # Circuit is open
    - TooManyFailures # Threshold exceeded

requirements:
  functional:
    - MUST track failures per component
    - MUST auto-reset after cooldown
    - MUST support manual override
    - MUST cascade failures upstream
    - MUST log all state changes
  performance:
    - latency: <100μs  # Permission check
    - memory: <10MB
  quality:
    - test_coverage: 100%  # Critical component
    - documentation: complete

implementation_spec:
  language: Rust
  patterns: [Singleton, Observer, State]
  restrictions:
    - NEVER allow calls when open
    - ALWAYS track metrics
    - ENFORCE cooldown periods
    - CASCADE failures

test_spec:
  unit_tests:
    - test_failure_tracking: counts correctly
    - test_auto_reset: resets after cooldown
    - test_cascade: propagates upstream
    - test_manual_override: allows force reset
  integration_tests:
    - test_with_components: protects all calls
    - test_under_load: handles high volume
  benchmarks:
    - latency: <100μs
    - memory: <10MB

example:
  ```rust
  pub struct GlobalCircuitBreaker {
      breakers: Arc<DashMap<String, ComponentBreaker>>,
      global_state: Arc<RwLock<CircuitState>>,
      config: CircuitConfig,
  }
  
  pub struct ComponentBreaker {
      state: CircuitState,
      failure_count: AtomicU32,
      success_count: AtomicU32,
      last_failure: Instant,
      last_attempt: Instant,
  }
  
  #[derive(Debug, Clone)]
  pub enum CircuitState {
      Closed,     // Normal operation
      Open,       // Blocking calls
      HalfOpen,   // Testing recovery
  }
  
  impl GlobalCircuitBreaker {
      pub fn check_permission(&self, component_id: &str) -> CircuitPermission {
          // Check global state first
          if *self.global_state.read().unwrap() == CircuitState::Open {
              return CircuitPermission::Denied("Global circuit open");
          }
          
          // Get or create component breaker
          let breaker = self.breakers.entry(component_id.to_string())
              .or_insert_with(|| ComponentBreaker::new(&self.config));
          
          match breaker.state {
              CircuitState::Closed => {
                  CircuitPermission::Allowed
              }
              CircuitState::Open => {
                  // Check if cooldown expired
                  if breaker.last_failure.elapsed() > self.config.cooldown {
                      breaker.state = CircuitState::HalfOpen;
                      CircuitPermission::Allowed  // Allow one test
                  } else {
                      CircuitPermission::Denied("Circuit open")
                  }
              }
              CircuitState::HalfOpen => {
                  // Allow limited calls for testing
                  CircuitPermission::Allowed
              }
          }
      }
      
      pub fn record_success(&self, component_id: &str) {
          if let Some(mut breaker) = self.breakers.get_mut(component_id) {
              breaker.success_count.fetch_add(1, Ordering::Relaxed);
              
              // Reset if in half-open state
              if breaker.state == CircuitState::HalfOpen {
                  breaker.state = CircuitState::Closed;
                  breaker.failure_count.store(0, Ordering::Relaxed);
              }
          }
      }
      
      pub fn record_failure(&self, component_id: &str) {
          if let Some(mut breaker) = self.breakers.get_mut(component_id) {
              let failures = breaker.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
              breaker.last_failure = Instant::now();
              
              // Trip circuit if threshold exceeded
              if failures >= self.config.failure_threshold {
                  breaker.state = CircuitState::Open;
                  
                  // Cascade check
                  self.check_cascade(component_id);
              }
          }
      }
  }
  ```

validation_criteria:
  - All external calls protected
  - Automatic recovery works
  - Cascading implemented
  - Metrics tracked
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