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
    - latency: <1 second
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
    - latency: <100ms
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
    - latency: <50ms
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
          
          // Overconfidence
          if self.overconfidence_detector.detect(context) {
              biases.push(BiasDetection::Overconfidence {
                  trigger: "Win streak with increasing position sizes".to_string(),
                  action: BiasAction::CapPositionSize,
                  max_size: context.portfolio_value * 0.01,  // 1% max
              });
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
    - allocation_time: <100ms
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