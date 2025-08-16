# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔴 ALEX'S MANDATORY INSTRUCTIONS - ABSOLUTE REQUIREMENTS

### 1. PROJECT GOAL
Develop a new generation, mixed **TA + ML + Grok xAI** fully autonomous trading bot that is:
- **Fully self-adjustable** - Adapts to market conditions automatically
- **Auto-tunable** - Optimizes parameters without human intervention
- **Full market auto-adaptation** - Learns and evolves with market changes
- **Zero human intervention** - Extracts maximum value from cryptocurrency exchanges autonomously

### 2. ZERO TOLERANCE POLICY - NO EXCEPTIONS!!!
**ABSOLUTE REQUIREMENTS - VIOLATION = IMMEDIATE REJECTION:**
- ❌ **NO fake implementations**
- ❌ **NO incomplete code**
- ❌ **NO empty functions**
- ❌ **NO leftover TODOs**
- ❌ **NO simplifications**
- ❌ **NO hardcoded values**
- **THIS IS MANDATORY - MUST DO!!!**

### 3. MANDATORY GITHUB PR WORKFLOW - EXTERNAL QA REVIEW
**EVERY SUB-TASK REQUIRES A SEPARATE PR:**
1. **ONE SUB-TASK = ONE PR** - Granular review by external QA team
2. **BEFORE starting ANY task**: Read PROJECT_MANAGEMENT_TASK_LIST_V5.md AND ARCHITECTURE.md
3. **Create feature branch**: `git checkout -b task-X.Y.Z-description`
4. **Implement with 100% quality** - No shortcuts, no fakes
5. **Create PR to GitHub** - Detailed description with validation checklist
6. **EXTERNAL QA REVIEW** - Wait for approval before proceeding
7. **Update docs IMMEDIATELY**: After QA approval, update V5 and ARCHITECTURE.md
8. **NO WORK WITHOUT PR APPROVAL** - This is MANDATORY!

### 4. 100% VALIDATION REQUIREMENTS
**EACH TASK MUST HAVE:**
- ✅ **100% test coverage** - FULL coverage, no exceptions
- ✅ **100% implemented** - Complete functionality
- ✅ **100% integrated** - Fully connected to system
- ✅ **ZERO shortcuts** - No compromises
- ✅ **100% tested** - All edge cases covered
- ✅ **Fully integrated into logic** - Working with all components
- **THIS IS A MUST!!!!**

### 5. GITHUB REPOSITORY
**Repository**: git@github.com:brw567/bot4.git
**Branch Strategy**: 
- `main` - Production ready code only
- `develop` - Integration branch
- `task-*` - Individual sub-task branches
**External QA**: Every PR reviewed by external quality assurance team

## 🚨 CRITICAL: Quality Gates & Enforcement

### Mandatory Before ANY Development
```bash
# Install git hooks - REQUIRED
cd /home/hamster/bot4
cp .git-hooks/* .git/hooks/
chmod +x .git/hooks/*
```

### Task Completion Verification
```bash
# MUST pass before marking ANY task complete
./scripts/verify_completion.sh

# Validate no fake implementations in Rust
python scripts/validate_no_fakes_rust.py

# Run all tests
cd rust_core && cargo test --all

# Check formatting and linting
cargo fmt --check
cargo clippy -- -D warnings
```

## 🎯 Project Overview

Bot4 is a next-generation, fully autonomous cryptocurrency trading platform combining **TA + ML + Grok xAI** for maximum market value extraction. Key characteristics:
- **Hybrid Intelligence**: Technical Analysis + Machine Learning + Grok xAI integration
- **Fully Autonomous**: Zero human intervention required
- **Self-Adjustable**: Auto-tunes and adapts to market conditions
- **Pure Rust implementation** - Zero Python in production
- **<50ns decision latency** target
- **200-300% APY target** in bull markets
- **Local development only** at `/home/hamster/bot4/`
- **Multi-agent development** with 8 specialized AI personas
- **ZERO TOLERANCE** for fake implementations, incomplete code, or shortcuts

## 📋 Task Management System

### Primary Documents (Read These First)
1. **PROJECT_MANAGEMENT_TASK_LIST_V5.md** - Master task list with 1,250+ tasks
2. **ARCHITECTURE.md** - Complete technical specification (2,267 lines)
3. **DEVELOPMENT_RULES.md** - Mandatory compliance rules

### Task Workflow - MANDATORY PROCESS
```bash
# ALEX'S MANDATORY WORKFLOW - EVERY TASK MUST:
1. READ PROJECT_MANAGEMENT_TASK_LIST_V5.md AND ARCHITECTURE.md FIRST!!!
2. Reference a task ID from V5 (e.g., "Task 8.3.2")
3. Start with grooming session if >2 hours work
4. Implement with 100% functionality - NO SHORTCUTS
5. Achieve 100% test coverage - NO EXCEPTIONS
6. Validate 100% integration - MUST WORK WITH SYSTEM
7. Update PROJECT_MANAGEMENT_TASK_LIST_V5.md IMMEDIATELY after EACH sub-task
8. Update ARCHITECTURE.md with FULL implementation details
9. Run ./scripts/verify_completion.sh - MUST PASS 100%
10. NO TASK IS COMPLETE WITHOUT 100% VALIDATION!!!
```

## 🏗️ Build & Development Commands

### Rust Development
```bash
# Build the project
cd /home/hamster/bot4/rust_core
cargo build --release

# Run tests with coverage
cargo test --all
cargo tarpaulin --out Html  # If installed

# Run benchmarks
cargo bench

# Format and lint
cargo fmt
cargo clippy -- -D warnings

# Check for security issues
cargo audit
```

### Database Setup
```bash
# PostgreSQL setup (bot3* naming is intentional - legacy compatibility)
sudo -u postgres psql
CREATE USER bot3user WITH PASSWORD 'bot3pass';
CREATE DATABASE bot3trading OWNER bot3user;
\q

# Initialize schema
PGPASSWORD=bot3pass psql -U bot3user -h localhost -d bot3trading -f /home/hamster/bot4/sql/init_schema.sql

# TimescaleDB extensions
PGPASSWORD=bot3pass psql -U bot3user -h localhost -d bot3trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

### Docker Operations
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-engine

# Restart a service
docker-compose restart trading-engine

# Full rebuild
docker-compose build --no-cache
docker-compose up -d
```

## 🏛️ Architecture Overview

### Project Structure
```
/home/hamster/bot4/
├── rust_core/           # Pure Rust implementation (PRIMARY)
│   ├── src/            # Main application code
│   └── crates/         # Component crates (to be migrated)
├── rust_core_old_epic7/ # Legacy Rust code (being migrated)
│   └── crates/         # 50+ specialized crates
├── frontend/           # React TypeScript UI
├── scripts/            # Validation and automation
└── sql/               # Database schemas
```

### Core Components (Rust)

#### Trading Engine (`rust_core/crates/trading_engine/`)
- Order management with <100μs execution
- Position tracking with real-time P&L
- Risk engine with circuit breakers
- WebSocket bridge for real-time data

#### Risk Management (`rust_core/crates/risk/`)
- Position limits (2% max per trade)
- Stop-loss enforcement (mandatory)
- Max drawdown monitoring (15% limit)
- Correlation analysis (<0.7 between positions)

#### ML Pipeline (`rust_core/crates/ml/`)
- Feature engineering pipeline
- Model versioning system
- Real-time inference (<50ns)
- Backtesting integration

#### Exchange Integration (`rust_core/crates/exchanges/`)
- Binance, Kraken, Coinbase connectors
- WebSocket stream management
- Order routing with smart execution
- Rate limiting and reconnection logic

## 🔧 Development Patterns

### Rust Code Standards
```rust
// EVERY component must include:
pub struct ComponentName {
    // Circuit breaker is MANDATORY
    circuit_breaker: CircuitBreaker,
    kill_switch: AtomicBool,
    max_impact: f64,  // Position/risk limit
}

// Performance documentation required
/// Phase: 2.3 - Trading Engine
/// Performance: <100μs order submission
/// Dependencies: risk_engine, exchange_connector
impl ComponentName {
    // Real implementations only - no todo!() or unimplemented!()
}
```

### Testing Requirements - 100% COVERAGE MANDATORY
```rust
#[cfg(test)]
mod tests {
    // ALEX'S MANDATE: 100% TEST COVERAGE - NO EXCEPTIONS!!!
    // ✅ Unit tests: REQUIRED for ALL functions (public AND private)
    // ✅ Integration tests: REQUIRED for ALL components
    // ✅ Performance tests: REQUIRED for ALL paths
    // ✅ Edge cases: ALL must be tested
    // ✅ Error paths: ALL must be tested
    // ✅ NO MOCKS - Real data only
    // ✅ NO SHORTCUTS - Full validation
    
    #[test]
    fn test_real_functionality() {
        // MUST use real calculations
        // MUST test all branches
        // MUST test all edge cases
        // MUST validate all outputs
        // 100% COVERAGE OR TASK FAILS!!!
    }
}
```

## 🚫 Common Pitfalls to Avoid

### Never Do This
```rust
// ❌ FORBIDDEN - Fake implementations
fn calculate_atr() -> f64 {
    price * 0.02  // REJECTED by validation
}

// ❌ FORBIDDEN - Placeholder returns
fn get_signal() -> Signal {
    todo!()  // Build will FAIL
}

// ❌ FORBIDDEN - Mock data in production
let mock_price = 50000.0;  // Will be caught
```

### Always Do This
```rust
// ✅ CORRECT - Real implementation
fn calculate_atr(candles: &[Candle]) -> f64 {
    // Actual ATR calculation
    technical_indicators::atr(candles, 14)
}

// ✅ CORRECT - Complete error handling
fn place_order(order: Order) -> Result<OrderId> {
    // Real exchange API call with retry logic
    exchange.place_order_with_retry(order, 3)
}
```

## 🎭 Multi-Agent System

The project uses 8 virtual agents with specific roles:

1. **Alex** - Team Lead: Coordinates all work, breaks deadlocks
2. **Morgan** - ML Specialist: ML models, zero tolerance for overfitting
3. **Sam** - Code Quality: Rust lead, VETO on fake code
4. **Quinn** - Risk Manager: VETO on uncapped risk
5. **Jordan** - Performance: <50ns latency enforcement
6. **Casey** - Exchange Integration: Order accuracy, rate limits
7. **Riley** - Testing: 95%+ coverage requirement
8. **Avery** - Data Engineer: TimescaleDB optimization

### Conflict Resolution
- Max 3 debate rounds before Alex decides
- Quinn has absolute veto on risk matters
- Sam has absolute veto on fake implementations
- Data-driven decisions when possible

## 🔍 Quality Enforcement Scripts

### Pre-commit Validation
```bash
# Automatic checks on every commit:
scripts/validate_no_fakes.py      # Detects fake implementations
scripts/validate_no_fakes_rust.py  # Rust-specific validation
cargo fmt --check                  # Format verification
cargo clippy -- -D warnings        # Linting
```

### Performance Monitoring
```bash
# Check latency targets
cargo bench --bench trading_bench

# Monitor memory usage
valgrind --leak-check=full target/release/bot4-trading

# Profile CPU usage
perf record -g target/release/bot4-trading
perf report
```

## 📊 Current Status & Priorities

### Active Phase: Phase 0 - Foundation Setup
Current focus areas from PROJECT_MANAGEMENT_TASK_LIST_V5.md:
- Task 0.1-0.5: Environment and tooling setup ✅
- Task 1.x: Core infrastructure (in progress)
- Task 2.x: Trading engine implementation (next)

### Migration in Progress
Moving from `rust_core_old_epic7/` to `rust_core/`:
- Consolidating 50+ crates into organized workspace
- Removing all Python dependencies
- Implementing missing core components

## 🚀 Quick Development Workflow

```bash
# 1. Start your day
cd /home/hamster/bot4
git pull
./scripts/verify_completion.sh  # Ensure clean state

# 2. Pick a task from V5
grep "TODO" PROJECT_MANAGEMENT_TASK_LIST_V5.md | head -20

# 3. Create feature branch
git checkout -b task-X.Y.Z-description

# 4. Implement with TDD
cd rust_core
cargo test --test specific_test -- --nocapture  # Write test first
# Implement feature
cargo test --all  # Verify

# 5. Validate before commit
./scripts/verify_completion.sh
cargo fmt
cargo clippy -- -D warnings

# 6. Update documentation
# Update PROJECT_MANAGEMENT_TASK_LIST_V5.md - mark task complete
# Update ARCHITECTURE.md - add implementation details

# 7. Commit with task reference
git add -A
git commit -m "Task X.Y.Z: Description of implementation"
```

## 🔐 Security & Risk Management

### Environment Variables
```bash
# Required in .env (never commit!)
DATABASE_URL=postgresql://bot3user:bot3pass@localhost:5432/bot3trading
REDIS_URL=redis://localhost:6379/0
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET=your_testnet_secret

# Risk limits (enforced by Quinn)
MAX_POSITION_SIZE=0.02
MAX_LEVERAGE=3
REQUIRE_STOP_LOSS=true
MAX_DRAWDOWN=0.15
```

### Circuit Breakers
Every component must implement circuit breakers:
- Trip on 3 consecutive errors
- Automatic reset after cooldown
- Manual kill switch for emergencies
- Cascade protection (upstream trips propagate)

## 📈 Performance Targets

Critical metrics that must be maintained:
- **Decision Latency**: <50ns (use SIMD where applicable)
- **Order Submission**: <100μs including network
- **Throughput**: 10,000+ orders/second capability
- **Memory**: No unbounded growth, <1GB steady state
- **Test Coverage**: **100% MANDATORY** - Alex's requirement, NO EXCEPTIONS!
- **Integration**: 100% working with all components
- **Validation**: 100% of functionality verified

## ✅ ALEX'S VALIDATION CHECKLIST - EVERY TASK

Before ANY task can be considered complete:
- [ ] **100% Implemented** - No TODOs, no empty functions, no placeholders
- [ ] **100% Tested** - Full test coverage, all edge cases
- [ ] **100% Integrated** - Works with entire system
- [ ] **Zero Shortcuts** - No simplifications or hardcoded values
- [ ] **Zero Fakes** - All implementations are real and working
- [ ] **Docs Updated** - V5 and ARCHITECTURE.md updated IMMEDIATELY
- [ ] **Verification Passed** - ./scripts/verify_completion.sh SUCCESS
- [ ] **Review Complete** - Validated by team member
- [ ] **NO COMPROMISES** - This is MANDATORY!!!

## 🧪 Testing Strategy

### Test Hierarchy
1. **Unit Tests**: Every public function
2. **Integration Tests**: Component interactions
3. **Performance Tests**: Latency benchmarks
4. **Backtesting**: Strategy validation with real data
5. **Shadow Mode**: 24-hour parallel run before production

### Running Specific Test Suites
```bash
# Unit tests only
cargo test --lib

# Integration tests
cargo test --test '*'

# Specific component
cargo test -p trading_engine

# With output
cargo test -- --nocapture

# Benchmarks
cargo bench --bench trading_bench
```

## 🎯 Remember

1. **Build it right the first time** - No shortcuts, ever
2. **Every line must be real** - No fake implementations
3. **Task tracking is mandatory** - Always reference V5 task IDs
4. **Documentation in same commit** - Code without docs doesn't exist
5. **Local development only** - Never deploy to remote servers
6. **95% test coverage minimum** - Non-negotiable
7. **Performance over features** - Maintain <50ns latency
8. **Risk management first** - Quinn has veto power
9. **Multi-agent consensus** - Use grooming sessions for complex tasks
10. **Continuous validation** - Run verify_completion.sh frequently