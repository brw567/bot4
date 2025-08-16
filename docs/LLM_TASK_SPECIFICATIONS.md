# Bot4 LLM-Optimized Task Specifications
## Complete Atomic Task Breakdown for AI Agent Implementation
## Designed for Claude and Other LLMs - Single Context Window Execution

---

## 🤖 LLM PARSING INSTRUCTIONS

```yaml
document_purpose: atomic_task_execution_guide
target_agents: [Claude, ChatGPT, Grok, Other_LLMs]
format: atomic_structured_specifications
execution_mode: single_context_autonomous

instructions:
  1. Each task is ATOMIC - completable in one context window
  2. Find task by TASK_ID (e.g., TASK_2.1.1)
  3. Check DEPENDENCIES - must be complete first
  4. Follow IMPLEMENTATION steps exactly
  5. Validate against SUCCESS_CRITERIA
  6. Run all TESTS specified
  7. Update STATUS when complete

atomic_requirements:
  max_hours: 12  # Single task max duration
  self_contained: true  # All info in task spec
  clear_deliverable: true  # Explicit output
  testable: true  # Verification criteria included
```

---

## 📋 ATOMIC TASK SPECIFICATION TEMPLATE

```yaml
task_template:
  task_id: TASK_X.Y.Z  # Unique identifier
  task_name: STRING  # Descriptive name
  parent_phase: NUMBER  # Phase this belongs to
  dependencies: [TASK_IDS]  # Must complete first
  owner: AGENT_NAME  # Responsible agent
  estimated_hours: NUMBER  # 1-12 hours max
  
  specification:
    inputs:
      required: MAP  # Required inputs
      optional: MAP  # Optional inputs
    outputs:
      deliverables: MAP  # What to produce
      artifacts: LIST  # Files/components created
    constraints:
      - CONSTRAINT_1  # Hard requirements
      - CONSTRAINT_2
    
  implementation:
    steps:
      - step: NAME
        action: DESCRIPTION
        code: |
          # Actual code to write
    validation:
      - CHECK_1  # How to verify
      - CHECK_2
    
  success_criteria:
    functional:
      - CRITERION_1: measurable_target
    performance:
      - latency: <VALUE
      - throughput: >VALUE
    quality:
      - test_coverage: >95%
      - no_todos: true
      
  test_spec:
    unit_tests:
      - test_name: expected_outcome
    integration_tests:
      - test_name: expected_outcome
    benchmarks:
      - metric: target_value
```

---

## 🔧 PHASE 0: FOUNDATION & PLANNING

### TASK 0.1.1: Development Environment Setup

```yaml
task_id: TASK_0.1.1
task_name: Setup Rust Development Environment
parent_phase: 0
dependencies: []
owner: Jordan
estimated_hours: 4

specification:
  inputs:
    required:
      host_os: Linux Ubuntu 20.04+
      disk_space: 500GB+
      ram: 32GB+
    optional:
      gpu: NVIDIA with CUDA support
  outputs:
    deliverables:
      rust_toolchain: 1.75+ installed
      docker: 24.0+ configured
      databases: PostgreSQL 15+, Redis 7+
    artifacts:
      - ~/.cargo/config.toml
      - docker-compose.yml
      - .env.development
  constraints:
    - Local development only (/home/hamster/bot4)
    - No cloud deployments
    - All services containerized

implementation:
  steps:
    - step: Install Rust
      action: Setup Rust toolchain with required components
      code: |
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source $HOME/.cargo/env
        rustup component add rustfmt clippy rust-analyzer
        rustup target add x86_64-unknown-linux-gnu
        
    - step: Configure Docker
      action: Install Docker and docker-compose
      code: |
        sudo apt-get update
        sudo apt-get install docker.io docker-compose
        sudo usermod -aG docker $USER
        
    - step: Create docker-compose.yml
      action: Define all required services
      code: |
        version: '3.8'
        services:
          postgres:
            image: timescale/timescaledb:latest-pg15
            environment:
              POSTGRES_USER: bot3user
              POSTGRES_PASSWORD: bot3pass
              POSTGRES_DB: bot3trading
            ports:
              - "5432:5432"
            volumes:
              - pgdata:/var/lib/postgresql/data
              
          redis:
            image: redis:7-alpine
            ports:
              - "6379:6379"
            command: redis-server --appendonly yes
            
          prometheus:
            image: prom/prometheus:latest
            ports:
              - "9090:9090"
            volumes:
              - ./prometheus.yml:/etc/prometheus/prometheus.yml
              
  validation:
    - rustc --version shows 1.75+
    - docker --version shows 24.0+
    - docker-compose up -d starts all services
    - psql connection successful

success_criteria:
  functional:
    - all_tools_installed: true
    - services_running: true
  performance:
    - startup_time: <30s
  quality:
    - documentation_complete: true

test_spec:
  unit_tests:
    - test_rust_compile: cargo new test && cd test && cargo build
    - test_docker_run: docker run hello-world
  integration_tests:
    - test_db_connection: psql -U bot3user -h localhost -d bot3trading -c "SELECT 1"
  benchmarks:
    - service_startup: <30s
```

---

### TASK 0.2.1: Rust Workspace Structure

```yaml
task_id: TASK_0.2.1
task_name: Create Rust Workspace Structure
parent_phase: 0
dependencies: [TASK_0.1.1]
owner: Sam
estimated_hours: 3

specification:
  inputs:
    required:
      project_root: /home/hamster/bot4
      architecture_doc: MASTER_ARCHITECTURE.md
    optional:
      template_repo: github.com/example/rust-template
  outputs:
    deliverables:
      workspace_structure: Complete Rust workspace
      cargo_toml: Workspace configuration
      crate_structure: All crates initialized
    artifacts:
      - Cargo.toml (workspace)
      - rust_core/Cargo.toml
      - rust_core/crates/*/Cargo.toml
  constraints:
    - Pure Rust implementation
    - No Python dependencies
    - Workspace-based organization

implementation:
  steps:
    - step: Create workspace Cargo.toml
      action: Define workspace with all member crates
      code: |
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
        
        [workspace.dependencies]
        tokio = { version = "1.35", features = ["full"] }
        serde = { version = "1.0", features = ["derive"] }
        anyhow = "1.0"
        thiserror = "1.0"
        
    - step: Create crate structure
      action: Initialize each crate with proper structure
      code: |
        for crate in common infrastructure risk data exchanges \
                     analysis strategies execution monitoring trading-engine; do
          cargo new --lib rust_core/crates/$crate
          echo "pub mod $crate;" > rust_core/crates/$crate/src/lib.rs
        done
        
    - step: Setup common types
      action: Define shared types in common crate
      code: |
        // rust_core/crates/common/src/lib.rs
        pub mod types;
        pub mod errors;
        pub mod constants;
        
        // rust_core/crates/common/src/types.rs
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Price(pub f64);
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Volume(pub f64);
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct OrderId(pub String);

  validation:
    - cargo build --workspace compiles successfully
    - All crates present in rust_core/crates/
    - Workspace dependencies resolved

success_criteria:
  functional:
    - workspace_compiles: true
    - all_crates_created: true
  performance:
    - build_time: <60s
  quality:
    - structure_follows_architecture: true

test_spec:
  unit_tests:
    - test_workspace_build: cargo build --workspace
    - test_workspace_test: cargo test --workspace
  integration_tests:
    - test_crate_dependencies: cargo tree
  benchmarks:
    - clean_build_time: <60s
```

---

## 🔧 PHASE 1: CORE INFRASTRUCTURE

### TASK 1.1.1: Event Bus Implementation

```yaml
task_id: TASK_1.1.1
task_name: Implement High-Performance Event Bus
parent_phase: 1
dependencies: [TASK_0.2.1]
owner: Jordan
estimated_hours: 8

specification:
  inputs:
    required:
      event_types: Vec<EventType>
      performance_target: <50ns latency
    optional:
      buffer_size: 10000
  outputs:
    deliverables:
      event_bus: Complete implementation
      benchmarks: Performance proof
    artifacts:
      - crates/infrastructure/src/event_bus.rs
      - crates/infrastructure/benches/event_bench.rs
  constraints:
    - Lock-free implementation
    - Zero-copy where possible
    - <50ns latency requirement

implementation:
  steps:
    - step: Define event types
      action: Create event enum and traits
      code: |
        use std::sync::Arc;
        use crossbeam::channel::{unbounded, Sender, Receiver};
        
        #[derive(Debug, Clone)]
        pub enum Event {
            MarketData(MarketDataEvent),
            Order(OrderEvent),
            Risk(RiskEvent),
            System(SystemEvent),
        }
        
        pub trait EventHandler: Send + Sync {
            fn handle(&self, event: &Event) -> Result<()>;
            fn event_types(&self) -> Vec<EventType>;
        }
        
    - step: Implement event bus
      action: Create high-performance event distribution
      code: |
        pub struct EventBus {
            senders: Arc<DashMap<EventType, Vec<Sender<Event>>>>,
            handlers: Arc<DashMap<String, Box<dyn EventHandler>>>,
        }
        
        impl EventBus {
            pub fn publish(&self, event: Event) -> Result<()> {
                let event_type = event.event_type();
                
                if let Some(senders) = self.senders.get(&event_type) {
                    for sender in senders.iter() {
                        sender.send(event.clone())?;
                    }
                }
                
                Ok(())
            }
            
            pub fn subscribe(&self, handler_id: String, handler: Box<dyn EventHandler>) {
                let (tx, rx) = unbounded();
                
                for event_type in handler.event_types() {
                    self.senders.entry(event_type)
                        .or_insert_with(Vec::new)
                        .push(tx.clone());
                }
                
                self.handlers.insert(handler_id, handler);
                self.spawn_handler(rx);
            }
        }
        
    - step: Add benchmarks
      action: Prove <50ns latency
      code: |
        #[bench]
        fn bench_event_publish(b: &mut Bencher) {
            let bus = EventBus::new();
            let event = Event::System(SystemEvent::Heartbeat);
            
            b.iter(|| {
                bus.publish(event.clone()).unwrap();
            });
        }

  validation:
    - Event publishing works
    - Subscription mechanism works
    - Benchmarks show <50ns latency
    - No memory leaks

success_criteria:
  functional:
    - publish_subscribe_works: true
    - all_event_types_supported: true
  performance:
    - latency: <50ns
    - throughput: >1M events/sec
  quality:
    - test_coverage: >95%
    - no_unwrap: true

test_spec:
  unit_tests:
    - test_publish: publishes event successfully
    - test_subscribe: receives published events
    - test_multiple_subscribers: all receive events
  integration_tests:
    - test_high_load: handles 1M events/sec
  benchmarks:
    - publish_latency: <50ns
    - throughput: >1M/sec
```

---

### TASK 1.2.1: State Management System

```yaml
task_id: TASK_1.2.1
task_name: Implement State Store with Snapshots
parent_phase: 1
dependencies: [TASK_1.1.1]
owner: Jordan
estimated_hours: 10

specification:
  inputs:
    required:
      state_types: [Portfolio, Positions, Orders, Risk]
      persistence: PostgreSQL with TimescaleDB
    optional:
      snapshot_interval: 60 seconds
  outputs:
    deliverables:
      state_store: Thread-safe state management
      snapshot_system: Automatic snapshots
      recovery: State recovery from snapshots
    artifacts:
      - crates/infrastructure/src/state.rs
      - crates/infrastructure/src/snapshot.rs
  constraints:
    - Thread-safe access
    - Consistent snapshots
    - Fast recovery <5s

implementation:
  steps:
    - step: Define state types
      action: Create state structures
      code: |
        use std::sync::Arc;
        use tokio::sync::RwLock;
        use serde::{Serialize, Deserialize};
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct TradingState {
            pub portfolio: Portfolio,
            pub positions: HashMap<String, Position>,
            pub orders: HashMap<OrderId, Order>,
            pub risk_metrics: RiskMetrics,
            pub last_updated: DateTime<Utc>,
        }
        
        pub struct StateStore {
            current: Arc<RwLock<TradingState>>,
            snapshots: Arc<RwLock<VecDeque<StateSnapshot>>>,
            persistence: Box<dyn StatePersistence>,
        }
        
    - step: Implement state operations
      action: CRUD operations for state
      code: |
        impl StateStore {
            pub async fn update_position(&self, position: Position) -> Result<()> {
                let mut state = self.current.write().await;
                state.positions.insert(position.symbol.clone(), position);
                state.last_updated = Utc::now();
                
                self.publish_update(StateUpdate::Position).await?;
                Ok(())
            }
            
            pub async fn get_portfolio(&self) -> Portfolio {
                self.current.read().await.portfolio.clone()
            }
            
            pub async fn create_snapshot(&self) -> Result<StateSnapshot> {
                let state = self.current.read().await;
                let snapshot = StateSnapshot {
                    state: state.clone(),
                    timestamp: Utc::now(),
                    version: self.next_version(),
                };
                
                self.persistence.save_snapshot(&snapshot).await?;
                self.snapshots.write().await.push_back(snapshot.clone());
                
                Ok(snapshot)
            }
        }
        
    - step: Implement recovery
      action: State recovery from snapshots
      code: |
        impl StateStore {
            pub async fn recover_from_snapshot(&self, version: Option<u64>) -> Result<()> {
                let snapshot = match version {
                    Some(v) => self.persistence.load_snapshot(v).await?,
                    None => self.persistence.load_latest_snapshot().await?,
                };
                
                *self.current.write().await = snapshot.state;
                
                // Replay events since snapshot
                let events = self.persistence.load_events_since(snapshot.timestamp).await?;
                for event in events {
                    self.apply_event(event).await?;
                }
                
                Ok(())
            }
        }

  validation:
    - State updates work correctly
    - Snapshots created successfully
    - Recovery completes in <5s
    - Thread-safe operations

success_criteria:
  functional:
    - crud_operations: working
    - snapshot_creation: automatic
    - recovery: <5s
  performance:
    - update_latency: <1ms
    - snapshot_time: <100ms
  quality:
    - thread_safe: true
    - persistence_reliable: true

test_spec:
  unit_tests:
    - test_state_update: updates reflected correctly
    - test_snapshot: creates valid snapshot
    - test_recovery: recovers to correct state
  integration_tests:
    - test_concurrent_access: no race conditions
    - test_persistence: survives restart
  benchmarks:
    - update_latency: <1ms
    - recovery_time: <5s
```

---

## 🔧 PHASE 2: RISK MANAGEMENT SYSTEM

### TASK 2.1.1: Core Risk Manager

```yaml
task_id: TASK_2.1.1
task_name: Implement Risk Management Core
parent_phase: 2
dependencies: [TASK_1.2.1]
owner: Quinn
estimated_hours: 12

specification:
  inputs:
    required:
      risk_limits: RiskLimits config
      portfolio: Portfolio state
      market_data: Real-time prices
    optional:
      override_config: Emergency overrides
  outputs:
    deliverables:
      risk_manager: Complete risk system
      risk_metrics: Real-time calculations
      circuit_breakers: Protection system
    artifacts:
      - crates/risk/src/manager.rs
      - crates/risk/src/circuit_breaker.rs
      - crates/risk/src/metrics.rs
  constraints:
    - Position size <2% portfolio
    - Correlation <0.7 between positions
    - Mandatory stop loss
    - <100ns validation time

implementation:
  steps:
    - step: Define risk structures
      action: Create risk types and limits
      code: |
        #[derive(Debug, Clone)]
        pub struct RiskLimits {
            pub max_position_pct: f64,  // 0.02 (2%)
            pub max_correlation: f64,   // 0.7
            pub max_leverage: f64,       // 3.0
            pub max_drawdown: f64,       // 0.15 (15%)
            pub require_stop_loss: bool, // true
            pub min_liquidity_ratio: f64, // 1.5
        }
        
        #[derive(Debug)]
        pub struct RiskManager {
            limits: RiskLimits,
            portfolio: Arc<RwLock<Portfolio>>,
            circuit_breaker: CircuitBreaker,
            metrics_calculator: MetricsCalculator,
        }
        
    - step: Implement validation
      action: Position validation logic
      code: |
        impl RiskManager {
            pub async fn validate_position(&self, position: &Position) -> Result<RiskValidation> {
                // Check circuit breaker first
                if self.circuit_breaker.is_tripped() {
                    return Ok(RiskValidation::Rejected {
                        reason: "Circuit breaker tripped".into(),
                        suggestion: None,
                    });
                }
                
                let portfolio = self.portfolio.read().await;
                
                // Position size check
                let position_value = position.size * position.entry_price;
                let portfolio_value = portfolio.total_value();
                let position_pct = position_value / portfolio_value;
                
                if position_pct > self.limits.max_position_pct {
                    return Ok(RiskValidation::Rejected {
                        reason: format!("Position size {:.2}% exceeds limit", position_pct * 100.0),
                        suggestion: Some(position.size * self.limits.max_position_pct / position_pct),
                    });
                }
                
                // Correlation check
                let correlation = self.calculate_correlation(position, &portfolio).await?;
                if correlation > self.limits.max_correlation {
                    return Ok(RiskValidation::Rejected {
                        reason: format!("Correlation {:.2} too high", correlation),
                        suggestion: None,
                    });
                }
                
                // Stop loss check
                if position.stop_loss.is_none() && self.limits.require_stop_loss {
                    return Ok(RiskValidation::Rejected {
                        reason: "Stop loss required".into(),
                        suggestion: None,
                    });
                }
                
                // Margin check
                let margin_required = self.calculate_margin(position);
                let margin_available = portfolio.available_margin();
                
                if margin_available < margin_required * 1.1 {
                    return Ok(RiskValidation::Rejected {
                        reason: "Insufficient margin".into(),
                        suggestion: Some(position.size * margin_available / (margin_required * 1.1)),
                    });
                }
                
                Ok(RiskValidation::Approved {
                    risk_score: self.calculate_risk_score(position),
                    adjusted_size: position.size,
                    stop_loss: position.stop_loss.unwrap_or_else(|| self.calculate_stop_loss(position)),
                })
            }
        }
        
    - step: Implement circuit breaker
      action: Multi-level protection system
      code: |
        pub struct CircuitBreaker {
            state: Arc<AtomicU8>,  // 0=closed, 1=half-open, 2=open
            consecutive_losses: Arc<AtomicU32>,
            total_loss_pct: Arc<AtomicF64>,
            last_reset: Arc<RwLock<Instant>>,
            config: CircuitBreakerConfig,
        }
        
        impl CircuitBreaker {
            pub fn record_trade(&self, pnl: f64, portfolio_value: f64) {
                if pnl < 0.0 {
                    self.consecutive_losses.fetch_add(1, Ordering::Relaxed);
                    
                    let loss_pct = pnl.abs() / portfolio_value;
                    let total = self.total_loss_pct.fetch_add(loss_pct, Ordering::Relaxed);
                    
                    // Check trip conditions
                    if self.consecutive_losses.load(Ordering::Relaxed) >= 3 ||
                       total + loss_pct > 0.05 {
                        self.trip();
                    }
                } else {
                    self.consecutive_losses.store(0, Ordering::Relaxed);
                }
            }
            
            pub fn trip(&self) {
                self.state.store(2, Ordering::Relaxed);
                warn!("Circuit breaker TRIPPED!");
            }
        }

  validation:
    - Risk validation works correctly
    - Circuit breaker trips on losses
    - All limits enforced
    - Performance meets requirements

success_criteria:
  functional:
    - position_validation: complete
    - circuit_breaker: functional
    - risk_metrics: accurate
  performance:
    - validation_time: <100ns
    - calculation_time: <1ms
  quality:
    - test_coverage: >95%
    - no_panics: true

test_spec:
  unit_tests:
    - test_position_size_limit: rejects >2%
    - test_correlation_limit: rejects >0.7
    - test_stop_loss_required: enforces stops
    - test_circuit_breaker_trip: trips on losses
  integration_tests:
    - test_with_portfolio: validates against real portfolio
    - test_concurrent_validation: thread-safe
  benchmarks:
    - validation_latency: <100ns
    - throughput: >10K validations/sec
```

---

### TASK 2.2.1: Liquidation Prevention System

```yaml
task_id: TASK_2.2.1
task_name: Implement Liquidation Prevention
parent_phase: 2
dependencies: [TASK_2.1.1]
owner: Quinn
estimated_hours: 8

specification:
  inputs:
    required:
      positions: Active positions
      market_prices: Real-time prices
      margin_requirements: Exchange rules
    optional:
      emergency_actions: Allowed actions
  outputs:
    deliverables:
      liquidation_monitor: Real-time monitoring
      prevention_actions: Automatic adjustments
      alerts: Warning system
    artifacts:
      - crates/risk/src/liquidation.rs
      - crates/risk/src/margin_monitor.rs
  constraints:
    - Monitor every 100ms
    - Act before 90% margin
    - Preserve capital priority

implementation:
  steps:
    - step: Liquidation monitor
      action: Real-time margin monitoring
      code: |
        pub struct LiquidationMonitor {
            positions: Arc<RwLock<HashMap<String, Position>>>,
            margin_calculator: MarginCalculator,
            action_executor: Box<dyn ActionExecutor>,
            alert_threshold: f64,  // 0.8 (80% of margin)
            action_threshold: f64,  // 0.9 (90% of margin)
        }
        
        impl LiquidationMonitor {
            pub async fn monitor_loop(&self) {
                let mut interval = tokio::time::interval(Duration::from_millis(100));
                
                loop {
                    interval.tick().await;
                    
                    let positions = self.positions.read().await;
                    for (symbol, position) in positions.iter() {
                        if let Err(e) = self.check_position(position).await {
                            error!("Failed to check position {}: {}", symbol, e);
                        }
                    }
                }
            }
            
            async fn check_position(&self, position: &Position) -> Result<()> {
                let margin_used = self.margin_calculator.calculate_used(position)?;
                let margin_available = self.margin_calculator.calculate_available(position)?;
                let margin_ratio = margin_used / margin_available;
                
                if margin_ratio > self.action_threshold {
                    self.take_prevention_action(position).await?;
                } else if margin_ratio > self.alert_threshold {
                    self.send_alert(position, margin_ratio).await?;
                }
                
                Ok(())
            }
        }
        
    - step: Prevention actions
      action: Automatic position adjustments
      code: |
        impl LiquidationMonitor {
            async fn take_prevention_action(&self, position: &Position) -> Result<()> {
                // Priority 1: Reduce position size
                let reduction_amount = position.size * 0.2;  // Reduce by 20%
                
                let order = Order {
                    symbol: position.symbol.clone(),
                    side: if position.side == Side::Long { Side::Sell } else { Side::Buy },
                    size: reduction_amount,
                    order_type: OrderType::Market,
                    time_in_force: TimeInForce::IOC,
                };
                
                self.action_executor.execute_order(order).await?;
                
                // Priority 2: Add margin if possible
                if let Ok(available_funds) = self.check_available_funds().await {
                    if available_funds > 0.0 {
                        self.add_margin(position, available_funds * 0.5).await?;
                    }
                }
                
                // Priority 3: Close other positions if critical
                if self.is_critical(position).await? {
                    self.close_lowest_performing_positions(2).await?;
                }
                
                Ok(())
            }
        }

  validation:
    - Monitoring works continuously
    - Actions taken before liquidation
    - Capital preserved

success_criteria:
  functional:
    - monitoring_active: true
    - prevention_effective: true
    - alerts_sent: true
  performance:
    - check_interval: 100ms
    - action_time: <1s
  quality:
    - no_liquidations: true
    - capital_preserved: true

test_spec:
  unit_tests:
    - test_margin_calculation: accurate calculation
    - test_threshold_detection: triggers at 90%
    - test_prevention_action: reduces position
  integration_tests:
    - test_live_monitoring: continuous operation
    - test_action_execution: orders placed
  benchmarks:
    - monitoring_latency: <10ms
    - action_latency: <1s
```

---

## 🎯 PHASE 3.5: EMOTION-FREE TRADING SYSTEM

### TASK 3.5.1.1: Hidden Markov Model for Regime Detection

```yaml
task_id: TASK_3.5.1.1
task_name: Implement HMM Regime Detector
parent_phase: 3.5
dependencies: [TASK_2.1.1]
owner: Morgan
estimated_hours: 8

specification:
  inputs:
    required:
      price_data: Vec<Candle>
      volume_data: Vec<f64>
      volatility: f64
    optional:
      training_data: Historical 2 years
  outputs:
    deliverables:
      hmm_model: Trained HMM
      regime_detector: Detection function
      accuracy_report: Validation results
    artifacts:
      - crates/analysis/src/regime/hmm.rs
      - models/hmm_regime.onnx
  constraints:
    - 5 hidden states required
    - >85% accuracy on test data
    - <100ms inference time

implementation:
  steps:
    - step: Define HMM structure
      action: Create Hidden Markov Model
      code: |
        use nalgebra::{DMatrix, DVector};
        
        pub struct HiddenMarkovModel {
            n_states: usize,  // 5 states
            n_observations: usize,
            transition_matrix: DMatrix<f64>,
            emission_matrix: DMatrix<f64>,
            initial_probs: DVector<f64>,
        }
        
        #[derive(Debug, Clone, PartialEq)]
        pub enum HiddenState {
            StrongBull,
            Bull,
            Neutral,
            Bear,
            StrongBear,
        }
        
        impl HiddenMarkovModel {
            pub fn new(n_states: usize, n_observations: usize) -> Self {
                Self {
                    n_states,
                    n_observations,
                    transition_matrix: DMatrix::from_element(n_states, n_states, 1.0 / n_states as f64),
                    emission_matrix: DMatrix::from_element(n_states, n_observations, 1.0 / n_observations as f64),
                    initial_probs: DVector::from_element(n_states, 1.0 / n_states as f64),
                }
            }
        }
        
    - step: Implement Baum-Welch training
      action: Train HMM parameters
      code: |
        impl HiddenMarkovModel {
            pub fn train(&mut self, observations: &[Vec<usize>], max_iter: usize) -> f64 {
                let mut log_likelihood = f64::NEG_INFINITY;
                
                for _ in 0..max_iter {
                    let mut new_transition = DMatrix::zeros(self.n_states, self.n_states);
                    let mut new_emission = DMatrix::zeros(self.n_states, self.n_observations);
                    let mut new_initial = DVector::zeros(self.n_states);
                    
                    let mut total_ll = 0.0;
                    
                    for seq in observations {
                        let (alpha, ll) = self.forward(seq);
                        let beta = self.backward(seq);
                        
                        total_ll += ll;
                        
                        // Update parameters using alpha and beta
                        self.update_parameters(&alpha, &beta, seq, 
                                              &mut new_transition, 
                                              &mut new_emission, 
                                              &mut new_initial);
                    }
                    
                    // Normalize and update
                    self.transition_matrix = new_transition;
                    self.emission_matrix = new_emission;
                    self.initial_probs = new_initial;
                    
                    if (total_ll - log_likelihood).abs() < 1e-6 {
                        break;
                    }
                    log_likelihood = total_ll;
                }
                
                log_likelihood
            }
        }
        
    - step: Implement Viterbi decoder
      action: Find most likely state sequence
      code: |
        impl HiddenMarkovModel {
            pub fn viterbi(&self, observations: &[usize]) -> Vec<HiddenState> {
                let t = observations.len();
                let mut delta = DMatrix::zeros(self.n_states, t);
                let mut psi = DMatrix::zeros(self.n_states, t);
                
                // Initialize
                for i in 0..self.n_states {
                    delta[(i, 0)] = self.initial_probs[i] * 
                                    self.emission_matrix[(i, observations[0])];
                }
                
                // Recursion
                for t_idx in 1..t {
                    for j in 0..self.n_states {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_state = 0;
                        
                        for i in 0..self.n_states {
                            let val = delta[(i, t_idx-1)] * 
                                     self.transition_matrix[(i, j)] * 
                                     self.emission_matrix[(j, observations[t_idx])];
                            if val > max_val {
                                max_val = val;
                                max_state = i;
                            }
                        }
                        
                        delta[(j, t_idx)] = max_val;
                        psi[(j, t_idx)] = max_state as f64;
                    }
                }
                
                // Backtrack
                let mut states = vec![0; t];
                states[t-1] = delta.column(t-1).argmax().0;
                
                for t_idx in (0..t-1).rev() {
                    states[t_idx] = psi[(states[t_idx+1], t_idx+1)] as usize;
                }
                
                states.into_iter()
                      .map(|s| self.map_to_regime(s))
                      .collect()
            }
            
            fn map_to_regime(&self, state: usize) -> HiddenState {
                match state {
                    0 => HiddenState::StrongBull,
                    1 => HiddenState::Bull,
                    2 => HiddenState::Neutral,
                    3 => HiddenState::Bear,
                    4 => HiddenState::StrongBear,
                    _ => HiddenState::Neutral,
                }
            }
        }

  validation:
    - HMM trains successfully
    - Viterbi decoder works
    - Accuracy >85% on test set
    - Inference <100ms

success_criteria:
  functional:
    - training_complete: true
    - inference_working: true
    - regime_detection: accurate
  performance:
    - training_time: <5 minutes
    - inference_time: <100ms
    - accuracy: >0.85
  quality:
    - numerically_stable: true
    - test_coverage: >90%

test_spec:
  unit_tests:
    - test_forward_backward: algorithms correct
    - test_viterbi: finds optimal path
    - test_training: converges properly
  integration_tests:
    - test_with_real_data: >85% accuracy
    - test_regime_transitions: detects changes
  benchmarks:
    - inference_latency: <100ms
    - training_time: <5 minutes
```

---

### TASK 3.5.1.2: LSTM Regime Classifier

```yaml
task_id: TASK_3.5.1.2
task_name: Implement LSTM for Regime Classification
parent_phase: 3.5
dependencies: [TASK_3.5.1.1]
owner: Morgan
estimated_hours: 12

specification:
  inputs:
    required:
      sequence_length: 50  # 50 candles lookback
      features: [OHLCV, RSI, MACD, Volume, Volatility]
      training_data: 2 years historical
    optional:
      pretrained_model: Path to existing model
  outputs:
    deliverables:
      lstm_model: Trained LSTM
      feature_extractor: Feature pipeline
      predictions: Regime classifications
    artifacts:
      - crates/analysis/src/regime/lstm.rs
      - models/lstm_regime.onnx
  constraints:
    - 2 LSTM layers, 128 units each
    - Dropout 0.2 for regularization
    - >90% validation accuracy

implementation:
  steps:
    - step: Feature extraction
      action: Prepare sequences for LSTM
      code: |
        use candle::{Tensor, Device};
        use ta::{indicators::*, DataItem};
        
        pub struct FeatureExtractor {
            sequence_length: usize,
            feature_count: usize,
        }
        
        impl FeatureExtractor {
            pub fn extract_features(&self, candles: &[Candle]) -> Result<Tensor> {
                let mut features = Vec::new();
                
                for window in candles.windows(self.sequence_length) {
                    let mut window_features = Vec::new();
                    
                    // Price features
                    let prices: Vec<f64> = window.iter().map(|c| c.close).collect();
                    let returns = calculate_returns(&prices);
                    let log_returns = calculate_log_returns(&prices);
                    
                    // Technical indicators
                    let rsi = RSI::new(14).calculate(&prices)?;
                    let macd = MACD::new(12, 26, 9).calculate(&prices)?;
                    let bb = BollingerBands::new(20, 2.0).calculate(&prices)?;
                    
                    // Volume features
                    let volumes: Vec<f64> = window.iter().map(|c| c.volume).collect();
                    let volume_ma = SMA::new(10).calculate(&volumes)?;
                    let volume_ratio = volumes.last() / volume_ma.last();
                    
                    // Volatility
                    let volatility = calculate_volatility(&returns, 20);
                    
                    // Combine features
                    window_features.extend(&returns);
                    window_features.extend(&rsi);
                    window_features.extend(&macd.signal);
                    window_features.push(volume_ratio);
                    window_features.push(volatility);
                    
                    features.push(window_features);
                }
                
                Tensor::from_vec(features.concat(), 
                               &[features.len(), self.sequence_length, self.feature_count],
                               &Device::Cpu)
            }
        }
        
    - step: Build LSTM model
      action: Create 2-layer LSTM architecture
      code: |
        use candle_nn::{Module, lstm, linear, dropout, seq};
        
        pub struct LSTMRegimeClassifier {
            lstm1: lstm::LSTM,
            dropout1: Dropout,
            lstm2: lstm::LSTM,
            dropout2: Dropout,
            fc: Linear,
            softmax: Softmax,
        }
        
        impl LSTMRegimeClassifier {
            pub fn new(vs: &nn::Path, input_size: usize, hidden_size: usize, n_classes: usize) -> Self {
                Self {
                    lstm1: lstm(vs / "lstm1", input_size, hidden_size, Default::default()),
                    dropout1: dropout(0.2),
                    lstm2: lstm(vs / "lstm2", hidden_size, hidden_size, Default::default()),
                    dropout2: dropout(0.2),
                    fc: linear(vs / "fc", hidden_size, n_classes, Default::default()),
                    softmax: softmax(-1),
                }
            }
            
            pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
                let (lstm1_out, _) = self.lstm1.forward(x)?;
                let dropout1_out = if train {
                    self.dropout1.forward(&lstm1_out)?
                } else {
                    lstm1_out
                };
                
                let (lstm2_out, _) = self.lstm2.forward(&dropout1_out)?;
                let dropout2_out = if train {
                    self.dropout2.forward(&lstm2_out)?
                } else {
                    lstm2_out
                };
                
                // Take last timestep
                let last_hidden = dropout2_out.narrow(1, 
                                                      dropout2_out.dim(1)? - 1, 
                                                      1)?;
                
                let fc_out = self.fc.forward(&last_hidden.squeeze(1)?)?;
                self.softmax.forward(&fc_out)
            }
        }
        
    - step: Training loop
      action: Train with early stopping
      code: |
        pub fn train_lstm(
            model: &mut LSTMRegimeClassifier,
            train_data: &DataLoader,
            val_data: &DataLoader,
            epochs: usize,
        ) -> Result<TrainingHistory> {
            let mut optimizer = Adam::new(&model.parameters(), 1e-3)?;
            let mut best_val_acc = 0.0;
            let mut patience_counter = 0;
            let patience = 10;
            
            let mut history = TrainingHistory::new();
            
            for epoch in 0..epochs {
                // Training
                let mut train_loss = 0.0;
                let mut train_correct = 0;
                let mut train_total = 0;
                
                for (batch_x, batch_y) in train_data.iter() {
                    optimizer.zero_grad();
                    
                    let predictions = model.forward(&batch_x, true)?;
                    let loss = cross_entropy(&predictions, &batch_y)?;
                    
                    loss.backward()?;
                    optimizer.step()?;
                    
                    train_loss += loss.to_scalar::<f32>()?;
                    train_correct += count_correct(&predictions, &batch_y)?;
                    train_total += batch_y.dim(0)?;
                }
                
                // Validation
                let val_acc = validate(model, val_data)?;
                
                history.add_epoch(train_loss / train_data.len() as f32,
                                 train_correct as f32 / train_total as f32,
                                 val_acc);
                
                // Early stopping
                if val_acc > best_val_acc {
                    best_val_acc = val_acc;
                    patience_counter = 0;
                    model.save("best_model.safetensors")?;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        println!("Early stopping at epoch {}", epoch);
                        break;
                    }
                }
            }
            
            Ok(history)
        }

  validation:
    - LSTM trains successfully
    - Features extracted correctly
    - Accuracy >90% on validation
    - Early stopping works

success_criteria:
  functional:
    - model_trains: true
    - predictions_accurate: true
    - features_valid: true
  performance:
    - training_time: <30 minutes
    - inference_time: <200ms
    - accuracy: >0.90
  quality:
    - no_overfitting: true
    - stable_training: true

test_spec:
  unit_tests:
    - test_feature_extraction: correct features
    - test_model_forward: output shape correct
    - test_training_step: loss decreases
  integration_tests:
    - test_full_training: converges properly
    - test_validation: >90% accuracy
  benchmarks:
    - inference_latency: <200ms
    - batch_processing: >100 samples/sec
```

---

### TASK 3.5.2.1: Emotion-Free Validator

```yaml
task_id: TASK_3.5.2.1
task_name: Implement Mathematical Decision Validator
parent_phase: 3.5
dependencies: [TASK_3.5.1.2]
owner: Quinn
estimated_hours: 10

specification:
  inputs:
    required:
      signal: TradingSignal
      historical_data: Performance statistics
      market_context: Current regime and volatility
    optional:
      override: Emergency override flag
  outputs:
    deliverables:
      validator: Complete validation system
      metrics: Statistical calculations
      decision: Accept/Reject with reasoning
    artifacts:
      - crates/analysis/src/emotion_free/validator.rs
      - crates/analysis/src/emotion_free/statistics.rs
  constraints:
    - p-value < 0.05 required
    - Expected value > 0 required
    - Sharpe ratio > 2.0 required
    - Confidence > 75% required

implementation:
  steps:
    - step: Statistical significance test
      action: Implement p-value calculation
      code: |
        use statrs::distribution::{StudentsT, ContinuousCDF};
        use statrs::statistics::Statistics;
        
        pub struct StatisticalValidator {
            significance_level: f64,  // 0.05
            min_sample_size: usize,   // 30
        }
        
        impl StatisticalValidator {
            pub fn calculate_p_value(&self, 
                                    signal_return: f64,
                                    historical_returns: &[f64]) -> f64 {
                if historical_returns.len() < self.min_sample_size {
                    return 1.0;  // Not enough data
                }
                
                let mean = historical_returns.mean();
                let std_dev = historical_returns.std_dev();
                let n = historical_returns.len() as f64;
                let std_error = std_dev / n.sqrt();
                
                // Calculate t-statistic
                let t_stat = (signal_return - mean) / std_error;
                
                // Get p-value from t-distribution
                let df = n - 1.0;
                let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
                
                // Two-tailed test
                2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
            }
            
            pub fn is_significant(&self, p_value: f64) -> bool {
                p_value < self.significance_level
            }
        }
        
    - step: Expected value calculation
      action: Calculate mathematical expectation
      code: |
        pub struct ExpectedValueCalculator {
            risk_free_rate: f64,  // 0.04 (4% annual)
        }
        
        impl ExpectedValueCalculator {
            pub fn calculate_ev(&self, signal: &TradingSignal) -> f64 {
                // Kelly Criterion based EV
                let win_prob = signal.win_probability;
                let loss_prob = 1.0 - win_prob;
                
                let avg_win = signal.average_win;
                let avg_loss = signal.average_loss;
                
                // Basic EV
                let raw_ev = (win_prob * avg_win) - (loss_prob * avg_loss);
                
                // Adjust for fees and slippage
                let fee_adjustment = signal.expected_fees;
                let slippage_adjustment = signal.expected_slippage;
                
                // Adjust for opportunity cost
                let holding_period = signal.expected_holding_period;
                let opportunity_cost = self.risk_free_rate * holding_period / 365.0;
                
                raw_ev - fee_adjustment - slippage_adjustment - opportunity_cost
            }
            
            pub fn calculate_kelly_fraction(&self, signal: &TradingSignal) -> f64 {
                let p = signal.win_probability;
                let q = 1.0 - p;
                let b = signal.average_win / signal.average_loss;
                
                // Kelly formula: f = (p*b - q) / b
                let kelly = (p * b - q) / b;
                
                // Apply Kelly reduction (25% of full Kelly)
                (kelly * 0.25).max(0.0).min(0.1)  // Cap at 10% of capital
            }
        }
        
    - step: Sharpe ratio validation
      action: Risk-adjusted return calculation
      code: |
        pub struct SharpeCalculator {
            risk_free_rate: f64,
            min_sharpe: f64,  // 2.0
        }
        
        impl SharpeCalculator {
            pub fn calculate_sharpe(&self, 
                                   returns: &[f64],
                                   period: TradingPeriod) -> f64 {
                if returns.is_empty() {
                    return 0.0;
                }
                
                let mean_return = returns.mean();
                let std_dev = returns.std_dev();
                
                if std_dev == 0.0 {
                    return 0.0;
                }
                
                // Annualize based on period
                let periods_per_year = match period {
                    TradingPeriod::Minute => 525600.0,
                    TradingPeriod::Hour => 8760.0,
                    TradingPeriod::Day => 365.0,
                };
                
                let annual_return = mean_return * periods_per_year;
                let annual_vol = std_dev * periods_per_year.sqrt();
                
                (annual_return - self.risk_free_rate) / annual_vol
            }
            
            pub fn validate_sharpe(&self, sharpe: f64) -> bool {
                sharpe >= self.min_sharpe
            }
        }
        
    - step: Main validation logic
      action: Combine all validators
      code: |
        pub struct EmotionFreeValidator {
            statistical: StatisticalValidator,
            ev_calculator: ExpectedValueCalculator,
            sharpe_calculator: SharpeCalculator,
            min_confidence: f64,  // 0.75
            bias_detector: BiasDetector,
        }
        
        impl EmotionFreeValidator {
            pub fn validate(&self, signal: &TradingSignal) -> ValidationDecision {
                // Check for emotional biases first
                if let Some(bias) = self.bias_detector.detect(signal) {
                    return ValidationDecision::Reject {
                        reason: format!("Emotional bias detected: {:?}", bias),
                        suggestion: "Wait for objective signal".into(),
                    };
                }
                
                // Statistical significance
                let p_value = self.statistical.calculate_p_value(
                    signal.expected_return,
                    &signal.historical_returns
                );
                
                if !self.statistical.is_significant(p_value) {
                    return ValidationDecision::Reject {
                        reason: format!("Not statistically significant (p={:.4})", p_value),
                        suggestion: "Need more data or stronger signal".into(),
                    };
                }
                
                // Expected value
                let ev = self.ev_calculator.calculate_ev(signal);
                if ev <= 0.0 {
                    return ValidationDecision::Reject {
                        reason: format!("Negative expected value: {:.4}", ev),
                        suggestion: "Signal not profitable after costs".into(),
                    };
                }
                
                // Sharpe ratio
                let sharpe = self.sharpe_calculator.calculate_sharpe(
                    &signal.historical_returns,
                    signal.period
                );
                
                if !self.sharpe_calculator.validate_sharpe(sharpe) {
                    return ValidationDecision::Reject {
                        reason: format!("Insufficient Sharpe ratio: {:.2}", sharpe),
                        suggestion: "Risk-adjusted returns too low".into(),
                    };
                }
                
                // Confidence check
                if signal.confidence < self.min_confidence {
                    return ValidationDecision::Reject {
                        reason: format!("Low confidence: {:.1}%", signal.confidence * 100.0),
                        suggestion: "Wait for higher confidence signal".into(),
                    };
                }
                
                // All checks passed
                ValidationDecision::Approve {
                    signal_id: signal.id.clone(),
                    expected_value: ev,
                    sharpe_ratio: sharpe,
                    p_value,
                    position_size: self.ev_calculator.calculate_kelly_fraction(signal),
                    reasoning: "All mathematical criteria satisfied".into(),
                }
            }
        }

  validation:
    - Statistical tests work correctly
    - EV calculation accurate
    - Sharpe ratio correct
    - Integration works

success_criteria:
  functional:
    - all_validators_work: true
    - decisions_consistent: true
    - math_correct: true
  performance:
    - validation_time: <100ms
    - batch_processing: >1000/sec
  quality:
    - no_emotional_trades: true
    - test_coverage: >95%

test_spec:
  unit_tests:
    - test_p_value: correct calculation
    - test_expected_value: includes all costs
    - test_sharpe: properly annualized
    - test_rejection: rejects bad signals
  integration_tests:
    - test_full_validation: complete pipeline
    - test_edge_cases: handles extremes
  benchmarks:
    - validation_latency: <100ms
    - throughput: >1000 validations/sec
```

---

## 🔧 PHASE 4: EXCHANGE INTEGRATION

### TASK 4.1.1: Binance Connector

```yaml
task_id: TASK_4.1.1
task_name: Implement Binance Exchange Connector
parent_phase: 4
dependencies: [TASK_1.1.1, TASK_2.1.1]
owner: Casey
estimated_hours: 10

specification:
  inputs:
    required:
      api_credentials: API key and secret
      testnet: Boolean for testnet/mainnet
      rate_limits: Binance specific limits
    optional:
      proxy_config: Proxy settings
  outputs:
    deliverables:
      connector: Complete Binance integration
      websocket: Real-time data streams
      rest_api: REST API implementation
    artifacts:
      - crates/exchanges/src/binance/mod.rs
      - crates/exchanges/src/binance/websocket.rs
      - crates/exchanges/src/binance/rest.rs
  constraints:
    - Rate limiting compliance
    - Automatic reconnection
    - <100ms order latency

implementation:
  steps:
    - step: REST API client
      action: Implement Binance REST API
      code: |
        use reqwest::{Client, Response};
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        pub struct BinanceRestClient {
            client: Client,
            api_key: String,
            api_secret: String,
            base_url: String,
            rate_limiter: RateLimiter,
        }
        
        impl BinanceRestClient {
            pub async fn place_order(&self, order: &Order) -> Result<OrderResponse> {
                let mut params = HashMap::new();
                params.insert("symbol", order.symbol.clone());
                params.insert("side", order.side.to_string());
                params.insert("type", order.order_type.to_string());
                params.insert("quantity", order.quantity.to_string());
                
                if let Some(price) = order.price {
                    params.insert("price", price.to_string());
                }
                
                params.insert("timestamp", Utc::now().timestamp_millis().to_string());
                
                // Sign request
                let query_string = self.build_query_string(&params);
                let signature = self.sign(&query_string);
                params.insert("signature", signature);
                
                // Rate limiting
                self.rate_limiter.acquire().await?;
                
                let response = self.client
                    .post(&format!("{}/api/v3/order", self.base_url))
                    .headers(self.build_headers())
                    .query(&params)
                    .send()
                    .await?;
                
                self.handle_response(response).await
            }
            
            fn sign(&self, data: &str) -> String {
                let mut mac = Hmac::<Sha256>::new_from_slice(self.api_secret.as_bytes())
                    .expect("HMAC can take key of any size");
                mac.update(data.as_bytes());
                hex::encode(mac.finalize().into_bytes())
            }
        }
        
    - step: WebSocket streams
      action: Real-time market data
      code: |
        use tokio_tungstenite::{connect_async, WebSocketStream};
        use futures_util::{SinkExt, StreamExt};
        
        pub struct BinanceWebSocket {
            streams: Vec<String>,
            ws: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
            event_sender: Sender<MarketEvent>,
            reconnect_interval: Duration,
        }
        
        impl BinanceWebSocket {
            pub async fn connect(&mut self) -> Result<()> {
                let url = self.build_ws_url();
                let (ws_stream, _) = connect_async(url).await?;
                self.ws = Some(ws_stream);
                
                self.spawn_message_handler();
                Ok(())
            }
            
            fn spawn_message_handler(&self) {
                let ws = self.ws.clone();
                let sender = self.event_sender.clone();
                
                tokio::spawn(async move {
                    while let Some(msg) = ws.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                if let Ok(event) = self.parse_message(&text) {
                                    let _ = sender.send(event).await;
                                }
                            }
                            Ok(Message::Close(_)) => {
                                self.handle_disconnect().await;
                            }
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                self.reconnect().await;
                            }
                            _ => {}
                        }
                    }
                });
            }
            
            async fn reconnect(&mut self) -> Result<()> {
                let mut attempts = 0;
                loop {
                    tokio::time::sleep(self.reconnect_interval).await;
                    
                    match self.connect().await {
                        Ok(_) => {
                            info!("Reconnected successfully");
                            return Ok(());
                        }
                        Err(e) => {
                            attempts += 1;
                            error!("Reconnect attempt {} failed: {}", attempts, e);
                            if attempts > 10 {
                                return Err(anyhow!("Max reconnection attempts exceeded"));
                            }
                        }
                    }
                }
            }
        }
        
    - step: Rate limiting
      action: Comply with Binance limits
      code: |
        use governor::{Quota, RateLimiter as Gov};
        
        pub struct RateLimiter {
            weight_limiter: Gov<NotKeyed, InMemoryState, DefaultClock>,
            order_limiter: Gov<NotKeyed, InMemoryState, DefaultClock>,
        }
        
        impl RateLimiter {
            pub fn new() -> Self {
                // Binance limits: 1200 weight per minute
                let weight_quota = Quota::per_minute(nonzero!(1200u32));
                
                // 10 orders per second, 100 per minute
                let order_quota = Quota::per_second(nonzero!(10u32))
                    .allow_burst(nonzero!(20u32));
                
                Self {
                    weight_limiter: Gov::new(weight_quota),
                    order_limiter: Gov::new(order_quota),
                }
            }
            
            pub async fn acquire(&self, weight: u32) -> Result<()> {
                for _ in 0..weight {
                    self.weight_limiter.until_ready().await;
                }
                Ok(())
            }
        }

  validation:
    - REST API works
    - WebSocket streams data
    - Rate limiting enforced
    - Reconnection works

success_criteria:
  functional:
    - orders_placed: successfully
    - market_data_received: real-time
    - rate_limits_respected: true
  performance:
    - order_latency: <100ms
    - websocket_latency: <10ms
  quality:
    - error_handling: comprehensive
    - test_coverage: >90%

test_spec:
  unit_tests:
    - test_signature: correct HMAC
    - test_rate_limiter: enforces limits
    - test_message_parsing: handles all types
  integration_tests:
    - test_place_order: testnet order works
    - test_websocket: receives real data
    - test_reconnection: auto-reconnects
  benchmarks:
    - order_latency: <100ms
    - message_processing: <1ms
```

---

## 🔧 PHASE 5: COST MANAGEMENT SYSTEM

### TASK 5.1.1: Fee Calculator

```yaml
task_id: TASK_5.1.1
task_name: Implement Comprehensive Fee Calculator
parent_phase: 5
dependencies: [TASK_4.1.1]
owner: Casey
estimated_hours: 8

specification:
  inputs:
    required:
      exchange: Exchange identifier
      order_type: Market/Limit
      vip_level: User's VIP tier
      volume_30d: 30-day trading volume
    optional:
      rebate_program: Special rebates
  outputs:
    deliverables:
      fee_calculator: Complete fee system
      fee_optimizer: Minimize fees
      reports: Fee analysis
    artifacts:
      - crates/execution/src/fees/calculator.rs
      - crates/execution/src/fees/optimizer.rs
  constraints:
    - Real-time fee updates
    - All exchanges supported
    - <50ns calculation time

implementation:
  steps:
    - step: Fee structure definition
      action: Define fee tiers for all exchanges
      code: |
        use rust_decimal::Decimal;
        
        #[derive(Debug, Clone)]
        pub struct FeeStructure {
            pub exchange: Exchange,
            pub maker_fee: Decimal,
            pub taker_fee: Decimal,
            pub vip_tiers: Vec<VipTier>,
        }
        
        #[derive(Debug, Clone)]
        pub struct VipTier {
            pub level: u8,
            pub volume_requirement: Decimal,
            pub maker_discount: Decimal,
            pub taker_discount: Decimal,
        }
        
        pub struct FeeCalculator {
            structures: HashMap<Exchange, FeeStructure>,
            user_tiers: HashMap<Exchange, u8>,
            cache: Arc<DashMap<FeeKey, FeeResult>>,
        }
        
        impl FeeCalculator {
            pub fn calculate_fee(&self, order: &Order) -> FeeResult {
                // Check cache first
                let key = FeeKey::from_order(order);
                if let Some(cached) = self.cache.get(&key) {
                    return cached.clone();
                }
                
                let structure = self.structures.get(&order.exchange)
                    .ok_or(FeeError::UnknownExchange)?;
                
                let base_fee = match order.order_type {
                    OrderType::Market => structure.taker_fee,
                    OrderType::Limit => structure.maker_fee,
                    _ => structure.taker_fee,
                };
                
                // Apply VIP discount
                let vip_level = self.user_tiers.get(&order.exchange).unwrap_or(&0);
                let discount = self.get_vip_discount(structure, *vip_level, order.order_type);
                
                let final_fee = base_fee * (Decimal::ONE - discount);
                let fee_amount = order.quantity * order.price * final_fee;
                
                let result = FeeResult {
                    fee_rate: final_fee,
                    fee_amount,
                    currency: order.quote_currency.clone(),
                    breakdown: FeeBreakdown {
                        base_fee,
                        vip_discount: discount,
                        volume_discount: Decimal::ZERO,
                        special_rebate: Decimal::ZERO,
                    },
                };
                
                // Cache result
                self.cache.insert(key, result.clone());
                result
            }
        }
        
    - step: Fee optimization
      action: Minimize trading costs
      code: |
        pub struct FeeOptimizer {
            calculator: FeeCalculator,
            volume_tracker: VolumeTracker,
        }
        
        impl FeeOptimizer {
            pub fn optimize_order_routing(&self, 
                                         order: &Order, 
                                         exchanges: &[Exchange]) -> RoutingDecision {
                let mut best_exchange = None;
                let mut best_fee = Decimal::MAX;
                
                for exchange in exchanges {
                    let mut test_order = order.clone();
                    test_order.exchange = *exchange;
                    
                    let fee_result = self.calculator.calculate_fee(&test_order);
                    let total_cost = fee_result.fee_amount + 
                                    self.estimate_slippage(&test_order);
                    
                    if total_cost < best_fee {
                        best_fee = total_cost;
                        best_exchange = Some(*exchange);
                    }
                }
                
                RoutingDecision {
                    exchange: best_exchange.unwrap(),
                    expected_fee: best_fee,
                    routing_reason: "Lowest total cost".into(),
                }
            }
            
            pub fn suggest_order_type(&self, order: &Order) -> OrderType {
                let limit_fee = self.calculator.calculate_fee_for_type(order, OrderType::Limit);
                let market_fee = self.calculator.calculate_fee_for_type(order, OrderType::Market);
                
                let slippage = self.estimate_slippage(order);
                
                // If limit order saves more than slippage cost, use limit
                if limit_fee.fee_amount + slippage < market_fee.fee_amount {
                    OrderType::Limit
                } else {
                    OrderType::Market
                }
            }
        }

  validation:
    - Fee calculations accurate
    - Optimization works
    - Caching effective

success_criteria:
  functional:
    - all_exchanges_supported: true
    - fee_accuracy: 100%
    - optimization_effective: true
  performance:
    - calculation_time: <50ns
    - cache_hit_rate: >90%
  quality:
    - test_coverage: >95%
    - documentation: complete

test_spec:
  unit_tests:
    - test_fee_calculation: accurate for all exchanges
    - test_vip_discounts: correctly applied
    - test_optimization: finds best route
  integration_tests:
    - test_with_real_orders: matches exchange fees
    - test_caching: improves performance
  benchmarks:
    - calculation_latency: <50ns
    - optimization_time: <1ms
```

---

## 📊 TASK DEPENDENCY GRAPH

```yaml
dependency_graph:
  phase_0:
    - TASK_0.1.1: []  # Environment setup
    - TASK_0.2.1: [TASK_0.1.1]  # Workspace structure
    
  phase_1:
    - TASK_1.1.1: [TASK_0.2.1]  # Event bus
    - TASK_1.2.1: [TASK_1.1.1]  # State management
    
  phase_2:
    - TASK_2.1.1: [TASK_1.2.1]  # Risk manager
    - TASK_2.2.1: [TASK_2.1.1]  # Liquidation prevention
    
  phase_3_5:
    - TASK_3.5.1.1: [TASK_2.1.1]  # HMM regime
    - TASK_3.5.1.2: [TASK_3.5.1.1]  # LSTM regime
    - TASK_3.5.2.1: [TASK_3.5.1.2]  # Emotion-free validator
    
  phase_4:
    - TASK_4.1.1: [TASK_1.1.1, TASK_2.1.1]  # Binance connector
    
  phase_5:
    - TASK_5.1.1: [TASK_4.1.1]  # Fee calculator
    
  critical_path:
    - TASK_0.1.1 -> TASK_0.2.1 -> TASK_1.1.1 -> TASK_1.2.1 -> TASK_2.1.1
```

---

## 🤖 LLM EXECUTION PROTOCOL

```yaml
execution_protocol:
  1_task_selection:
    - Find next task with satisfied dependencies
    - Verify estimated_hours fits context window
    - Load task specification
    
  2_implementation:
    - Follow implementation steps exactly
    - Write actual code (no placeholders)
    - Include all error handling
    
  3_validation:
    - Run unit tests specified
    - Check success criteria
    - Verify performance targets
    
  4_completion:
    - Update task status to "completed"
    - Document actual metrics
    - Report any deviations
    
  rules:
    - NEVER use todo!() or unimplemented!()
    - ALWAYS include error handling
    - MUST meet performance targets
    - MUST have >95% test coverage
```

---

## 📈 PROGRESS TRACKING

```yaml
progress_tracking:
  total_phases: 14  # 0-13
  total_tasks: 234  # All atomic tasks
  
  phase_status:
    phase_0: in_progress
    phase_1: pending
    phase_2: pending
    phase_3: pending
    phase_3_5: pending  # Emotion-free (NEW)
    phase_4: pending
    phase_5: pending
    phase_6: pending
    phase_7: pending
    phase_8: pending
    phase_9: pending
    phase_10: pending
    phase_11: pending
    phase_12: pending
    phase_13: pending
    
  metrics:
    tasks_completed: 0
    tasks_in_progress: 2
    tasks_blocked: 0
    completion_percentage: 0%
    
  critical_additions:
    emotion_free_trading: Added as Phase 3.5
    fee_management: Phase 5 (critical for profitability)
    atomic_breakdown: All tasks <12 hours for LLM execution
```

---

## ✅ VALIDATION CHECKLIST FOR LLMS

```yaml
validation_checklist:
  before_marking_complete:
    code_quality:
      - [ ] No TODO, unimplemented!() or panic!()
      - [ ] All unwrap() justified or removed
      - [ ] Error handling comprehensive
      - [ ] Performance targets met
      
    testing:
      - [ ] Unit tests >95% coverage
      - [ ] Integration tests pass
      - [ ] Benchmarks meet targets
      - [ ] Edge cases handled
      
    documentation:
      - [ ] Public APIs documented
      - [ ] Examples provided
      - [ ] Architecture updated
      - [ ] Metrics recorded
      
    atomic_requirements:
      - [ ] Task completable in one session
      - [ ] All dependencies satisfied
      - [ ] Deliverables produced
      - [ ] Success criteria met
```

---

*This document provides complete atomic task specifications for LLM implementation.*
*Every task is self-contained and executable within a single context window.*
*NO versioned documents - this is the single source of truth for task specifications.*