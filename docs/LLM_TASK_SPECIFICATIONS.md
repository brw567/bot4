# Bot4 LLM-Optimized Task Specifications
## Complete Task Details for AI Agent Implementation
## Version 1.0 - Structured for Autonomous Execution

---

## 🤖 LLM PARSING INSTRUCTIONS

```yaml
document_purpose: task_execution_guide
target_agents: [Claude, ChatGPT, Grok, Other_LLMs]
format: structured_specifications
execution_mode: autonomous

instructions:
  1. Find task by TASK_ID
  2. Check DEPENDENCIES first
  3. Follow IMPLEMENTATION_STEPS exactly
  4. Validate against SUCCESS_CRITERIA
  5. Run all TESTS specified
  6. Update STATUS when complete
```

---

## 📋 TASK SPECIFICATION FORMAT

```yaml
task_template:
  task_id: STRING # Unique identifier
  parent_phase: NUMBER # Phase this belongs to
  dependencies: [TASK_IDS] # Must complete first
  
  specification:
    inputs: MAP # What you receive
    outputs: MAP # What you produce
    constraints: LIST # Restrictions
    
  implementation:
    steps: LIST # Ordered steps
    code_samples: LIST # Examples
    patterns: LIST # Design patterns
    
  validation:
    tests: LIST # Required tests
    metrics: MAP # Performance targets
    checklist: LIST # Completion criteria
```

---

## 🔧 PHASE 0: FOUNDATION TASKS

### TASK 0.1.1: Development Environment Configuration

```yaml
task_id: TASK_0.1.1
parent_phase: 0
dependencies: []
owner: Jordan
estimated_hours: 4

specification:
  inputs:
    host_system: Linux Ubuntu 20.04+
    available_resources:
      cpu: 8+ cores
      ram: 32GB+
      disk: 500GB+
  
  outputs:
    development_environment:
      rust: 1.70+
      docker: 24.0+
      postgres: 15+
      redis: 7+
      monitoring_stack: [prometheus, grafana, jaeger, loki]
  
  constraints:
    - Use local development only (/home/hamster/bot4)
    - No remote servers
    - All services dockerized

implementation:
  steps:
    - step: Install Rust
      command: |
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source $HOME/.cargo/env
        rustup component add rustfmt clippy rust-analyzer
      validation: rustc --version
      
    - step: Install Docker
      command: |
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
      validation: docker --version
      
    - step: Create docker-compose.yml
      content: |
        version: '3.8'
        services:
          postgres:
            image: postgres:15-alpine
            environment:
              POSTGRES_PASSWORD: bot4pass
              POSTGRES_DB: bot4_trading
            ports:
              - "5432:5432"
            volumes:
              - postgres_data:/var/lib/postgresql/data
              
          timescaledb:
            image: timescale/timescaledb:latest-pg15
            environment:
              POSTGRES_PASSWORD: bot4pass
            ports:
              - "5433:5432"
              
          redis:
            image: redis:7-alpine
            ports:
              - "6379:6379"
            command: redis-server --requirepass bot4redis
            
          prometheus:
            image: prom/prometheus:latest
            ports:
              - "9090:9090"
            volumes:
              - ./prometheus.yml:/etc/prometheus/prometheus.yml
              
          grafana:
            image: grafana/grafana:latest
            ports:
              - "3001:3000"
            environment:
              GF_SECURITY_ADMIN_PASSWORD: bot4grafana
      
    - step: Initialize databases
      command: |
        docker-compose up -d postgres timescaledb redis
        sleep 10
        PGPASSWORD=bot4pass psql -h localhost -U postgres -d bot4_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
      validation: docker-compose ps

validation:
  tests:
    - test: Rust installed
      command: rustc --version
      expected: version 1.70 or higher
      
    - test: Docker running
      command: docker ps
      expected: CONTAINER ID present
      
    - test: Databases accessible
      command: PGPASSWORD=bot4pass psql -h localhost -U postgres -d bot4_trading -c "SELECT 1"
      expected: "1"
      
  checklist:
    - [x] Rust toolchain installed
    - [x] Docker & Docker Compose installed
    - [x] All containers running
    - [x] Databases initialized
    - [x] Monitoring stack operational

success_criteria:
  - All services running
  - No port conflicts
  - All validations pass
  - Can compile Rust code
  - Can connect to databases
```

---

### TASK 0.2.1: Rust Workspace Structure

```yaml
task_id: TASK_0.2.1
parent_phase: 0
dependencies: [TASK_0.1.1]
owner: Sam
estimated_hours: 2

specification:
  inputs:
    project_root: /home/hamster/bot4
    architecture_doc: MASTER_ARCHITECTURE_V2.md
  
  outputs:
    workspace_structure:
      cargo_toml: workspace configuration
      crate_structure: modular crates
      directory_layout: organized folders
  
  constraints:
    - Pure Rust (no Python)
    - Modular architecture
    - Clear separation of concerns

implementation:
  steps:
    - step: Create workspace Cargo.toml
      file: /home/hamster/bot4/rust_core/Cargo.toml
      content: |
        [workspace]
        members = [
            "crates/common",
            "crates/infrastructure",
            "crates/risk",
            "crates/data",
            "crates/exchanges",
            "crates/analysis",
            "crates/strategies",
            "crates/execution",
            "crates/monitoring",
            "crates/trading-engine",
        ]
        resolver = "2"
        
        [workspace.package]
        version = "0.1.0"
        edition = "2021"
        authors = ["Bot4 Team"]
        license = "Proprietary"
        
        [workspace.dependencies]
        tokio = { version = "1.35", features = ["full"] }
        serde = { version = "1.0", features = ["derive"] }
        serde_json = "1.0"
        anyhow = "1.0"
        thiserror = "1.0"
        tracing = "0.1"
        chrono = { version = "0.4", features = ["serde"] }
        rust_decimal = "1.33"
        
    - step: Create crate structure
      commands: |
        cd /home/hamster/bot4/rust_core
        cargo new --lib crates/common
        cargo new --lib crates/infrastructure
        cargo new --lib crates/risk
        cargo new --lib crates/data
        cargo new --lib crates/exchanges
        cargo new --lib crates/analysis
        cargo new --lib crates/strategies
        cargo new --lib crates/execution
        cargo new --lib crates/monitoring
        cargo new crates/trading-engine
        
    - step: Create common types
      file: /home/hamster/bot4/rust_core/crates/common/src/lib.rs
      content: |
        //! Common types and traits for Bot4 trading platform
        
        use chrono::{DateTime, Utc};
        use rust_decimal::Decimal;
        use serde::{Deserialize, Serialize};
        
        /// Unique identifier for orders
        pub type OrderId = uuid::Uuid;
        
        /// Unique identifier for positions
        pub type PositionId = uuid::Uuid;
        
        /// Trading side
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum Side {
            Buy,
            Sell,
        }
        
        /// Order type
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum OrderType {
            Market,
            Limit,
            StopLoss,
            TakeProfit,
        }
        
        /// Market data point
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MarketData {
            pub symbol: String,
            pub exchange: String,
            pub bid: Decimal,
            pub ask: Decimal,
            pub last: Decimal,
            pub volume: Decimal,
            pub timestamp: DateTime<Utc>,
        }
        
        /// Trading signal
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Signal {
            pub symbol: String,
            pub side: Side,
            pub confidence: f64,
            pub size: Decimal,
            pub entry: Decimal,
            pub stop_loss: Decimal,
            pub take_profit: Decimal,
            pub metadata: serde_json::Value,
        }

validation:
  tests:
    - test: Workspace compiles
      command: cd /home/hamster/bot4/rust_core && cargo build --all
      expected: Compilation successful
      
    - test: Tests pass
      command: cd /home/hamster/bot4/rust_core && cargo test --all
      expected: All tests pass
      
  checklist:
    - [x] Workspace Cargo.toml created
    - [x] All crates initialized
    - [x] Common types defined
    - [x] Dependencies specified
    - [x] Compilation successful

success_criteria:
  - Clean workspace structure
  - All crates compile
  - No dependency conflicts
  - Clear module separation
```

---

## 🔧 PHASE 1: INFRASTRUCTURE TASKS

### TASK 1.1.1: Event Bus Implementation

```yaml
task_id: TASK_1.1.1
parent_phase: 1
dependencies: [TASK_0.2.1]
owner: Jordan
estimated_hours: 8

specification:
  inputs:
    performance_requirement:
      latency: <50ns
      throughput: >1M events/sec
    patterns: [Observer, Pub-Sub]
  
  outputs:
    event_bus:
      type: lock-free, async
      features: [routing, filtering, replay]
  
  constraints:
    - Zero-copy where possible
    - No dynamic allocation in hot path
    - Thread-safe

implementation:
  steps:
    - step: Define event types
      file: crates/infrastructure/src/events.rs
      content: |
        use chrono::{DateTime, Utc};
        use serde::{Deserialize, Serialize};
        
        /// Base event trait
        pub trait Event: Send + Sync + 'static {
            fn event_type(&self) -> &str;
            fn timestamp(&self) -> DateTime<Utc>;
            fn correlation_id(&self) -> uuid::Uuid;
        }
        
        /// Market data event
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MarketDataEvent {
            pub event_type: String,
            pub timestamp: DateTime<Utc>,
            pub correlation_id: uuid::Uuid,
            pub data: common::MarketData,
        }
        
        /// Order event
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct OrderEvent {
            pub event_type: String,
            pub timestamp: DateTime<Utc>,
            pub correlation_id: uuid::Uuid,
            pub order_id: common::OrderId,
            pub status: OrderStatus,
        }
        
    - step: Implement event bus
      file: crates/infrastructure/src/event_bus.rs
      content: |
        use std::sync::Arc;
        use tokio::sync::broadcast;
        use crossbeam::channel;
        
        pub struct EventBus {
            sender: broadcast::Sender<Arc<dyn Event>>,
            receiver: broadcast::Receiver<Arc<dyn Event>>,
            capacity: usize,
        }
        
        impl EventBus {
            pub fn new(capacity: usize) -> Self {
                let (sender, receiver) = broadcast::channel(capacity);
                Self { sender, receiver, capacity }
            }
            
            pub async fn publish(&self, event: Arc<dyn Event>) -> Result<()> {
                self.sender.send(event)?;
                Ok(())
            }
            
            pub async fn subscribe(&self) -> EventSubscriber {
                EventSubscriber {
                    receiver: self.sender.subscribe(),
                }
            }
        }
        
    - step: Add performance optimizations
      content: |
        // Use SIMD for filtering
        // Implement zero-copy serialization
        // Add memory pool for events

validation:
  tests:
    - test: Latency benchmark
      command: cargo bench --package infrastructure --bench event_bus
      expected: p99 latency <50ns
      
    - test: Throughput benchmark
      command: cargo bench --package infrastructure --bench throughput
      expected: >1M events/sec
      
    - test: Thread safety
      command: cargo test --package infrastructure thread_safety
      expected: All tests pass
      
  metrics:
    latency_ns: <50
    throughput_per_sec: >1000000
    memory_usage_mb: <100

success_criteria:
  - Meets latency requirements
  - Meets throughput requirements
  - Thread-safe implementation
  - Zero-copy optimization working
  - All tests passing
```

---

## 🔧 PHASE 2: RISK MANAGEMENT TASKS

### TASK 2.1.1: Risk Manager Core

```yaml
task_id: TASK_2.1.1
parent_phase: 2
dependencies: [TASK_1.1.1]
owner: Quinn
estimated_hours: 12

specification:
  inputs:
    risk_limits:
      max_position_size: 0.02  # 2% of portfolio
      max_leverage: 3.0
      max_drawdown: 0.15  # 15%
      max_correlation: 0.7
    market_data: MarketData
    portfolio: Portfolio
  
  outputs:
    risk_validation: RiskValidation
    risk_metrics: RiskMetrics
  
  constraints:
    - NEVER exceed position limits
    - ALWAYS require stop loss
    - Circuit breaker mandatory

implementation:
  steps:
    - step: Define risk types
      file: crates/risk/src/types.rs
      content: |
        use rust_decimal::Decimal;
        
        #[derive(Debug, Clone)]
        pub struct RiskLimits {
            pub max_position_size: Decimal,
            pub max_leverage: Decimal,
            pub max_drawdown: Decimal,
            pub max_correlation: f64,
            pub require_stop_loss: bool,
        }
        
        #[derive(Debug, Clone)]
        pub enum RiskValidation {
            Approved {
                position_id: PositionId,
                risk_score: f64,
                stop_loss: Decimal,
                take_profit: Option<Decimal>,
            },
            Rejected {
                reason: String,
                suggested_size: Decimal,
            },
        }
        
    - step: Implement risk manager
      file: crates/risk/src/manager.rs
      content: |
        pub struct RiskManager {
            limits: RiskLimits,
            portfolio: Arc<RwLock<Portfolio>>,
            circuit_breaker: CircuitBreaker,
            margin_manager: MarginManager,
            liquidation_manager: LiquidationManager,
        }
        
        impl RiskManager {
            pub fn validate_position(&self, position: &Position) -> Result<RiskValidation> {
                // Circuit breaker check
                if self.circuit_breaker.is_tripped() {
                    return Err(RiskError::CircuitBreakerTripped);
                }
                
                // Position size check
                let portfolio_value = self.portfolio.read().unwrap().total_value();
                let position_percentage = position.value / portfolio_value;
                
                if position_percentage > self.limits.max_position_size {
                    return Ok(RiskValidation::Rejected {
                        reason: format!("Position exceeds {}% limit", 
                                      self.limits.max_position_size * 100.0),
                        suggested_size: self.calculate_safe_size(position),
                    });
                }
                
                // Correlation check
                let correlation = self.calculate_correlation(position);
                if correlation > self.limits.max_correlation {
                    return Ok(RiskValidation::Rejected {
                        reason: format!("Correlation {} exceeds limit", correlation),
                        suggested_size: Decimal::ZERO,
                    });
                }
                
                // Margin check
                let margin_result = self.margin_manager.validate(position)?;
                if !margin_result.sufficient {
                    return Ok(RiskValidation::Rejected {
                        reason: "Insufficient margin".to_string(),
                        suggested_size: margin_result.max_position_size,
                    });
                }
                
                // Liquidation check
                let liquidation_price = self.liquidation_manager.calculate(position);
                let distance_to_liquidation = 
                    ((position.entry_price - liquidation_price) / position.entry_price).abs();
                
                if distance_to_liquidation < 0.05 {  // <5% from liquidation
                    return Ok(RiskValidation::Rejected {
                        reason: "Too close to liquidation".to_string(),
                        suggested_size: self.calculate_safe_size(position),
                    });
                }
                
                // Stop loss requirement
                if self.limits.require_stop_loss && position.stop_loss.is_none() {
                    return Ok(RiskValidation::Rejected {
                        reason: "Stop loss required".to_string(),
                        suggested_size: position.size,
                    });
                }
                
                Ok(RiskValidation::Approved {
                    position_id: position.id,
                    risk_score: self.calculate_risk_score(position),
                    stop_loss: position.stop_loss.unwrap_or_else(|| 
                        self.calculate_stop_loss(position)),
                    take_profit: self.calculate_take_profit(position),
                })
            }
        }
        
    - step: Implement liquidation prevention
      file: crates/risk/src/liquidation.rs
      content: |
        pub struct LiquidationManager {
            maintenance_margin_rate: Decimal,
            emergency_reducer: EmergencyReducer,
        }
        
        impl LiquidationManager {
            pub fn calculate_liquidation_price(&self, position: &Position) -> Decimal {
                let initial_margin = position.size * position.entry_price / position.leverage;
                let maintenance_margin = position.size * position.entry_price 
                                       * self.maintenance_margin_rate;
                
                match position.side {
                    Side::Long => {
                        position.entry_price * (Decimal::ONE - 
                            (initial_margin - maintenance_margin) / position.size)
                    }
                    Side::Short => {
                        position.entry_price * (Decimal::ONE + 
                            (initial_margin - maintenance_margin) / position.size)
                    }
                }
            }
            
            pub fn monitor_positions(&self, positions: &[Position]) -> Vec<RiskAction> {
                let mut actions = Vec::new();
                
                for position in positions {
                    let liq_price = self.calculate_liquidation_price(position);
                    let current_price = self.get_current_price(&position.symbol);
                    let distance = ((current_price - liq_price) / current_price).abs();
                    
                    if distance < 0.03 {  // <3% from liquidation
                        actions.push(RiskAction::EmergencyClose(position.id));
                    } else if distance < 0.05 {  // <5%
                        actions.push(RiskAction::ReducePosition(position.id, 0.5));
                    } else if distance < 0.10 {  // <10%
                        actions.push(RiskAction::AddMargin(position.id));
                    }
                }
                
                actions
            }
        }

validation:
  tests:
    - test: Position size validation
      input: position with 3% of portfolio
      expected: Rejected with reason
      
    - test: Correlation check
      input: position with 0.8 correlation
      expected: Rejected
      
    - test: Liquidation prevention
      input: position 4% from liquidation
      expected: EmergencyClose action
      
    - test: Stop loss enforcement
      input: position without stop loss
      expected: Rejected when required
      
    - test: Circuit breaker
      input: tripped circuit breaker
      expected: Error returned
      
  benchmarks:
    - validation_latency: <100ns
    - throughput: >1M validations/sec
    
  checklist:
    - [x] All risk checks implemented
    - [x] Liquidation prevention working
    - [x] Circuit breaker integrated
    - [x] Performance targets met
    - [x] 100% test coverage

success_criteria:
  - Never approves risky positions
  - Prevents liquidations
  - Meets performance targets
  - All tests passing
  - No fake implementations
```

---

## 📊 TASK DEPENDENCY GRAPH

```yaml
dependency_graph:
  phase_0:
    - TASK_0.1.1 → TASK_0.1.2 → TASK_0.1.3
    - TASK_0.2.1 → TASK_0.2.2 → TASK_0.2.3
    - TASK_0.3.1 → TASK_0.3.2 → TASK_0.3.3
    
  phase_1:
    requires: [phase_0]
    - TASK_1.1.1 → TASK_1.1.2 → TASK_1.1.3
    - TASK_1.2.1 → TASK_1.2.2 → TASK_1.2.3
    
  phase_2:
    requires: [phase_1]
    - TASK_2.1.1 → TASK_2.1.2 → TASK_2.1.3
    - TASK_2.2.1 → TASK_2.2.2 → TASK_2.2.3
    - TASK_2.3.1 → TASK_2.3.2 → TASK_2.3.3
    
  critical_path:
    - phase_0 → phase_1 → phase_2 → phase_3 → phase_4 → phase_5
```

---

## 🤖 LLM EXECUTION PROTOCOL

```yaml
execution_protocol:
  for_each_task:
    1. verify_dependencies:
        check: all dependencies.status == 'completed'
        action: wait if not ready
        
    2. prepare_environment:
        load: inputs from dependencies
        setup: required resources
        
    3. execute_implementation:
        follow: implementation.steps
        use: code_samples as templates
        apply: patterns specified
        
    4. validate_results:
        run: all validation.tests
        check: metrics meet targets
        verify: checklist items
        
    5. update_status:
        set: status = 'completed'
        record: actual_metrics
        note: any deviations

  error_handling:
    on_test_failure:
      - Fix implementation
      - Re-run tests
      - Document issue
      
    on_performance_miss:
      - Optimize code
      - Try SIMD/parallel
      - Document if impossible
      
    on_dependency_fail:
      - Wait for dependency
      - Report blockage
      - Work on parallel tasks
```

---

## 📈 PROGRESS TRACKING

```yaml
progress_tracking:
  per_task:
    status: [not_started, in_progress, completed, blocked]
    metrics:
      estimated_hours: NUMBER
      actual_hours: NUMBER
      test_coverage: PERCENTAGE
      performance_met: BOOLEAN
      
  per_phase:
    total_tasks: NUMBER
    completed_tasks: NUMBER
    blocked_tasks: NUMBER
    completion_percentage: PERCENTAGE
    
  overall:
    total_phases: 13
    completed_phases: NUMBER
    current_phase: NUMBER
    estimated_completion: DATE
```

---

## ✅ VALIDATION CHECKLIST FOR LLMS

```yaml
validation_checklist:
  before_marking_complete:
    code_quality:
      - [ ] No TODO or unimplemented
      - [ ] No unwrap without justification
      - [ ] No panic in production code
      - [ ] All error cases handled
      
    testing:
      - [ ] Unit tests >95% coverage
      - [ ] Integration tests pass
      - [ ] Performance benchmarks met
      - [ ] Edge cases tested
      
    documentation:
      - [ ] Public APIs documented
      - [ ] Examples provided
      - [ ] README updated
      - [ ] Changelog entry added
      
    performance:
      - [ ] Latency requirements met
      - [ ] Throughput requirements met
      - [ ] Memory usage acceptable
      - [ ] No memory leaks
```

---

*This document provides complete task specifications for LLM implementation.*
*Each task includes all necessary details for autonomous execution.*
*Update task status after completion.*