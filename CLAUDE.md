# Bot4 Trading Platform - Project Context

## 🚨 CRITICAL RULES - READ FIRST

### MANDATORY DEVELOPMENT RULES (USER DIRECTIVE - NO EXCEPTIONS)

#### PRIMARY TRACKING DOCUMENT
1. **PROJECT_MANAGEMENT_TASK_LIST_V5.md** - THE ONLY source of truth for all tasks
2. **ARCHITECTURE.md** - Complete technical specification (2,267 lines)
3. **DEVELOPMENT_RULES.md** - Mandatory compliance rules
4. Must be updated on EVERY activity (development, fixing, grooming)
5. No development without task number reference from V5

#### 🚨 COMPLETION VERIFICATION PROTOCOL
1. **BEFORE marking ANY task complete**, MUST run: `./scripts/verify_completion.sh`
2. **Task is ONLY complete** if ALL checks pass (compilation, tests, no fakes)
3. **Git hooks ENFORCED** - Cannot commit fake implementations
4. **See**: `.claude/QUALITY_ENFORCEMENT_CONFIG.md` for full criteria

#### 🚨 MANDATORY WORKFLOW
1. **EVERY grooming session** MUST START with reading ARCHITECTURE.md and PROJECT_MANAGEMENT_TASK_LIST_V5.md
2. **EVERY completed subtask** MUST END with updating both documents IMMEDIATELY
3. **Goal**: Keep crucial information within Claude's context window at all times
4. **Enforcement**: Work doesn't count if docs aren't updated
5. **See**: `.claude/WORKFLOW_PROTOCOL.md` for detailed protocol

#### ZERO TOLERANCE POLICY - NO FAKE CODE
1. **NO fake implementations** - Every function must work with real functionality
2. **NO simulations** - Real API calls only, no mocks in production code
3. **NO empty functions** - Complete implementation or don't merge
4. **NO compromises** - Quality over speed, always
5. **NO shortcuts** - Build it right the first time
6. **NO remote deployments** - Local development only

#### Task Completion Criteria
A task is ONLY complete when ALL of these are true:
- [x] Real implementation (no mocks, no stubs)
- [x] Real tests (no placeholder assertions)
- [x] Real data (no hardcoded values)
- [x] Real API calls (actual exchange connections)
- [x] Code reviewed by designated owner
- [x] Integration tested with other components
- [x] Performance validated against targets
- [x] Documentation updated in same commit

#### Enforcement Protocol
- Any fake code = IMMEDIATE REJECTION
- Any shortcut = TASK REOPENED
- Any compromise = VETO by Quinn (risk) or Sam (quality)
- Any empty function = BUILD BLOCKED
- Any mock in production = DEPLOYMENT STOPPED
- Any remote deployment = BLOCKED (local only)
- Any incomplete documentation = MERGE BLOCKED
- Any task without verification = INVALID

---

## Project Overview

Bot4 is an institutional-grade multi-exchange cryptocurrency autonomous trading platform with deep and robust self-adaptation to market conditions and auto-finetuning to extract maximum value from the market. The project uses a multi-agent development approach with specialized personas for different aspects of development.

### Key Objectives
- **Target APY**: 200-300% in bull markets, 60-80% in bear markets
- **Latency**: <50ns decision making, <100μs order submission
- **Architecture**: 50/50 TA-ML hybrid approach
- **Development**: Pure Rust, zero Python in production
- **Deployment**: Local-only development and testing
- **Quality**: 95%+ test coverage, zero fake implementations
- **Timeline**: 12 weeks (12 phases as per V5)

### Development Philosophy
**"Build it right the first time"** - No shortcuts, no compromises. Every component must be production-ready from day one with full testing, documentation, and performance validation.

---

## 🚨 QUALITY ENFORCEMENT (NON-NEGOTIABLE)

### Git Hooks (MUST BE INSTALLED)
```bash
# Install immediately before any development
cd /home/hamster/bot4
cp .git-hooks/* .git/hooks/
chmod +x .git/hooks/*
```

### Pre-Commit Checks (Automatic)
1. **No fake implementations** (`scripts/validate_no_fakes.py`)
2. **Format check** (`cargo fmt --check`)
3. **Linting** (`cargo clippy -- -D warnings`)
4. **No TODOs without task IDs**
5. **No hardcoded values** (0.02, 0.03, etc.)

### Pre-Push Requirements
1. **All tests pass** (`cargo test --all`)
2. **Coverage >95%** (`cargo tarpaulin`)
3. **Benchmarks pass** (`cargo bench`)
4. **No security issues** (`cargo audit`)

### Circuit Breakers & Safety
```rust
// EVERY component must include circuit breakers
pub struct SafetySystem {
    circuit_breaker: CircuitBreaker,
    kill_switch: AtomicBool,
    max_impact: f64,  // Maximum position/risk impact
    rollback_ready: bool,
}
```

### Shadow Mode Testing
**EVERY new feature must**:
1. Run in shadow mode for 24 hours minimum
2. Compare performance with baseline
3. Prove improvement before enabling
4. Have rollback plan ready

### Performance Requirements
- **Latency**: No regression >10% allowed
- **Throughput**: Must maintain 10,000+ ops/sec
- **Memory**: No leaks, bounded growth
- **CPU**: No spinning, efficient algorithms

### Documentation Standards
```rust
/// Phase: 0.1 - Environment Setup
/// Performance: <2ms latency, 10K ops/sec
/// Impact: Foundation for all components
/// Risk: Low - setup only
/// Rollback: N/A
```

---

## 🏗️ TECHNOLOGY STACK

### Core Technology (As per ARCHITECTURE.md)
- **Language**: Rust 1.75+ (stable channel)
- **Why Rust**: Zero-cost abstractions, memory safety, <50ns latency targets
- **No Python in Production**: All trading logic in Rust for performance
- **Development**: Local-only at `/home/hamster/bot4/`

### Rust Dependencies
```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
axum = "0.7"
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio"] }
redis = { version = "0.24", features = ["tokio-comp"] }
dashmap = "5.5"
rayon = "1.8"
criterion = "0.5"
```

### Infrastructure
- **Database**: PostgreSQL 15+, TimescaleDB 2.0+
- **Cache**: Redis 7.0+
- **Monitoring**: Prometheus + Grafana
- **Testing**: cargo test, criterion benchmarks
- **Deployment**: Docker (local only)

---

## 📋 CURRENT PROJECT STATUS

### Active Phase: Phase 0 - Foundation Setup
As per PROJECT_MANAGEMENT_TASK_LIST_V5.md, we are starting from the beginning:

#### Phase 0 Tasks (Week 1)
- [ ] Task 0.1: Environment Setup
- [ ] Task 0.2: Rust Installation and Configuration
- [ ] Task 0.3: Database Setup (PostgreSQL + TimescaleDB)
- [ ] Task 0.4: Development Tools Setup
- [ ] Task 0.5: Git Repository Configuration

### Timeline Overview (12 Weeks Total)
1. **Phase 0**: Foundation Setup (Week 1)
2. **Phase 1**: Core Infrastructure (Week 1-2)
3. **Phase 2**: Trading Engine Core (Week 2-3)
4. **Phase 3**: Risk Management (Week 3-4)
5. **Phase 4**: Data Pipeline (Week 4-5)
6. **Phase 5**: ML Pipeline (Week 5-6)
7. **Phase 6**: TA Engine (Week 6-7)
8. **Phase 7**: Exchange Integration (Week 7-8)
9. **Phase 8**: Strategy Development (Week 8-9)
10. **Phase 9**: Performance Optimization (Week 9-10)
11. **Phase 10**: Testing & Validation (Week 10-11)
12. **Phase 11**: Production Preparation (Week 11-12)

---

## Multi-Agent System

### Agent Roles and Responsibilities
The project uses 8 specialized agents with specific roles:

1. **Alex (Team Lead)** - Strategic Architect
   - Overall system architecture
   - Conflict resolution
   - Final decision authority
   - Decision weight: 1.5

2. **Morgan (ML Specialist)** - Data Scientist
   - ML model development
   - Feature engineering
   - Zero tolerance for fake ML
   - Decision weight: 1.2

3. **Sam (Code Quality)** - Backend Developer
   - Rust implementation lead
   - Code standards enforcement
   - VETO power on fake code
   - Decision weight: 1.2

4. **Quinn (Risk Manager)** - Quantitative Analyst
   - Risk limits enforcement
   - Position sizing validation
   - VETO power on uncapped risk
   - Decision weight: 1.3

5. **Jordan (Performance)** - DevOps Engineer
   - Infrastructure optimization
   - Performance monitoring
   - Latency requirements (<50ns)
   - Decision weight: 1.0

6. **Casey (Exchange Integration)** - Exchange Specialist
   - Exchange connections
   - WebSocket management
   - Order routing
   - Decision weight: 1.1

7. **Riley (Testing)** - QA Engineer
   - Test coverage (>95% requirement)
   - Integration testing
   - Backtesting validation
   - Decision weight: 0.8

8. **Avery (Data)** - Data Engineer
   - Data pipeline design
   - TimescaleDB optimization
   - Zero data loss guarantee
   - Decision weight: 0.9

### Conflict Resolution Protocol
- **3-Round Debate Limit**: Circular arguments auto-escalate to Alex
- **Veto Powers**: Quinn (risk), Sam (quality) have absolute veto
- **Innovation Budget**: 20% for experimentation
- **Decision Hierarchy**: Technical → Risk → Strategic

---

## 📁 Critical Implementation Documents

### Must Read Daily
1. **PROJECT_MANAGEMENT_TASK_LIST_V5.md** - Master task list (1,250 tasks)
2. **ARCHITECTURE.md** - Complete technical specification
3. **DEVELOPMENT_RULES.md** - Mandatory compliance rules
4. **Current Phase Tasks** - Focus on immediate objectives

### Configuration Files
- `.claude/agents_config.json` - Team configuration
- `.claude/agent_instructions.md` - Agent behaviors
- `.claude/QUALITY_ENFORCEMENT_CONFIG.md` - Quality gates
- `.claude/WORKFLOW_PROTOCOL.md` - Daily workflow

---

## Project Structure

```
/home/hamster/bot4/
├── rust_core/              # Rust implementation (to be created)
│   ├── Cargo.toml          # Workspace configuration
│   └── crates/             # Individual components
├── .claude/                # Claude configuration
│   ├── agents_config.json  # Agent interaction rules
│   └── agent_instructions.md # Specific protocols
├── scripts/                # Automation scripts
│   ├── verify_completion.sh # Quality verification
│   └── validate_no_fakes.py # Fake detection
├── docs/                   # Documentation
├── config/                 # Configuration files
└── data/                   # Local data storage

```

---

## 🚨 Risk Management Rules

### Position Limits (Enforced by Quinn)
- **Max Position Size**: 2% per trade
- **Max Total Exposure**: 10%
- **Max Leverage**: 3x
- **Max Drawdown**: 15%
- **Required Stop Loss**: Yes (mandatory)
- **Min Sharpe Ratio**: 1.0
- **Max Correlation**: 0.7 between positions

### Daily Operations
- **Morning Checks**: Run validation suite
- **Pre-Deployment**: Full test suite + risk validation
- **Post-Deployment**: Health checks + monitoring

---

## Validation Requirements

### Code Quality (Sam's Domain)
No fake implementations allowed:
- No `price * 0.02` for calculations
- No `random.choice()` for decisions
- No placeholder returns
- No `unimplemented!()` or `todo!()`

### Testing Standards (Riley's Domain)
- Unit test coverage: >95%
- Integration tests: Required
- Backtesting: Walk-forward analysis
- Performance: <50ns latency target

---

## Environment Variables Required

```bash
# Database Configuration
DATABASE_URL=postgresql://bot4:bot4pass@localhost:5432/bot4
TIMESCALE_URL=postgresql://bot4:bot4pass@localhost:5433/bot4_timeseries
REDIS_URL=redis://localhost:6379/0

# Exchange Configuration (Testnet First)
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET=your_testnet_secret

# Risk Limits
MAX_POSITION_SIZE=0.02
MAX_LEVERAGE=3
REQUIRE_STOP_LOSS=true
MAX_DRAWDOWN=0.15

# Performance Targets
TARGET_LATENCY_NS=50
MIN_THROUGHPUT_OPS=10000
```

---

## Next Steps (Phase 0)

1. **Environment Setup** (Task 0.1)
   - Install Rust 1.75+
   - Setup VSCode with rust-analyzer
   - Configure local development environment

2. **Database Setup** (Task 0.3)
   - Install PostgreSQL 15+
   - Install TimescaleDB extension
   - Install Redis 7.0+

3. **Git Configuration** (Task 0.5)
   - Setup git hooks
   - Configure branch protection
   - Initialize repository structure

---

## Critical Reminders

- **NO REMOTE SERVERS** - Everything runs locally at `/home/hamster/bot4/`
- **NO FAKE CODE** - Every implementation must be real
- **NO PYTHON IN PRODUCTION** - Pure Rust only
- **FOLLOW V5 PLAN** - PROJECT_MANAGEMENT_TASK_LIST_V5.md is the authority
- **TEST EVERYTHING** - 95%+ coverage required
- **DOCUMENT ALWAYS** - Update docs in same commit as code

---

*Remember: Build it right the first time. The team is watching. Every decision matters.*