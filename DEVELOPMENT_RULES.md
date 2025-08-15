# Bot4 Development Rules - MANDATORY COMPLIANCE

## 🚨 ABSOLUTE DEVELOPMENT RULES (NO EXCEPTIONS)

### 1. LOCAL DEVELOPMENT ONLY
- **ALL development happens on local machine** at `/home/hamster/bot4/`
- **NO remote servers** - Better control and immediate feedback
- **NO cloud deployments** - Direct code access required
- **NO SSH deployments** - Everything runs locally
- **Benefit**: Instant debugging, full visibility, no network delays

### 2. ZERO FAKE IMPLEMENTATIONS
```yaml
prohibited_patterns:
  - unimplemented!()
  - todo!()
  - panic!("not implemented")
  - // TODO (without task ID)
  - return Ok(()) # Empty implementation
  - mock_data
  - fake_response
  - dummy_value
  - price * 0.02 # Hardcoded calculations
```

### 3. MANDATORY VERIFICATION
Every task completion MUST:
1. Run `./scripts/verify_completion.sh`
2. Pass ALL checks (100% required)
3. Update PROJECT_MANAGEMENT_TASK_LIST_V5.md
4. Update ARCHITECTURE.md if architecture changed
5. Commit with detailed message

### 4. DEVELOPMENT WORKFLOW

#### Daily Routine (MANDATORY)
```yaml
09:00:
  - Read ARCHITECTURE.md
  - Read PROJECT_MANAGEMENT_TASK_LIST_V5.md
  - Team standup (15 min)
  - Select tasks for day

10:00-17:00:
  - TDD approach (test first)
  - Real implementations only
  - Continuous verification
  - Update docs in same commit

17:00:
  - Run full test suite
  - Update task statuses
  - Commit all work
  - Plan tomorrow

18:00:
  - Backup everything locally
  - Generate progress report
```

### 5. QUALITY GATES

#### Pre-Commit (Automatic)
```bash
#!/bin/bash
# .git/hooks/pre-commit
cargo fmt --check || exit 1
cargo clippy -- -D warnings || exit 1
cargo test || exit 1
./scripts/validate_no_fakes.py || exit 1
./scripts/check_risk_limits.py || exit 1
```

#### Pre-Merge Requirements
- Test coverage >= 95%
- Zero compilation warnings
- Zero clippy warnings
- Performance benchmarks passing
- Documentation complete
- 2 code reviews minimum

### 6. TASK MANAGEMENT

#### Task Selection
1. ONLY work on tasks from PROJECT_MANAGEMENT_TASK_LIST_V5.md
2. Tasks must be assigned by Alex (team lead)
3. Cannot skip task numbers
4. Must complete all subtasks

#### Task Completion
A task is ONLY complete when:
- [x] 100% implemented (no partial work)
- [x] 100% tested (all edge cases)
- [x] 100% documented (inline + external)
- [x] 100% reviewed (by domain expert)
- [x] 100% verified (script passes)

### 7. DOCUMENTATION REQUIREMENTS

#### Every Function Must Have:
```rust
/// Task: [Task ID from PROJECT_MANAGEMENT_TASK_LIST_V5.md]
/// Purpose: [Clear description]
/// Performance: [Latency and throughput targets]
/// Risk: [Risk considerations]
/// Author: [Agent name]
/// Tests: [Test file location]
pub fn function_name() -> Result<()> {
    // REAL implementation
}
```

#### Every File Must Have:
```rust
//! Module: [Module name]
//! Task: [Task ID]
//! Architecture: [Link to ARCHITECTURE.md section]
//! Owner: [Agent responsible]
//! Status: [Development/Testing/Production]
```

### 8. TESTING REQUIREMENTS

#### Test Pyramid
```
         /\
        /E2E\      10% - Full system tests
       /------\
      /Integr.\    30% - Component tests  
     /----------\
    /   Unit     \ 60% - Function tests
   /--------------\
```

#### Test Standards
- NO mocked external services in integration tests
- NO hardcoded test data
- NO skipped tests
- NO flaky tests
- REAL data from test exchanges

### 9. PERFORMANCE REQUIREMENTS

#### Every Component Must:
- Benchmark before optimization
- Meet latency targets (<100μs)
- Maintain throughput (>10K ops/sec)
- Profile memory usage
- Document performance characteristics

#### Benchmark Template:
```rust
#[bench]
fn bench_component(b: &mut Bencher) {
    b.iter(|| {
        // Measure real operation
        let result = component.process();
        assert!(result.latency_ns < 100_000);
    });
}
```

### 10. RISK MANAGEMENT

#### Every Trading Operation Must:
1. Pre-trade risk check
2. Position size validation
3. Stop-loss enforcement
4. Drawdown monitoring
5. Circuit breaker integration

#### Risk Wrapper Pattern:
```rust
pub async fn execute_trade(signal: Signal) -> Result<Order> {
    // MANDATORY risk checks
    risk_manager.pre_trade_check(&signal)?;
    
    let order = create_order(signal)?;
    
    risk_manager.validate_order(&order)?;
    
    let result = exchange.execute(order).await?;
    
    risk_manager.post_trade_check(&result)?;
    
    Ok(result)
}
```

### 11. AGENT RESPONSIBILITIES

#### Task Assignment Rules:
- **Alex**: Assigns all tasks, coordinates team
- **Sam**: Trading engine, TA, strategies
- **Morgan**: ML pipeline, models, AI
- **Quinn**: Risk management, compliance
- **Jordan**: Infrastructure, performance
- **Casey**: Exchange integration, orders
- **Riley**: Testing, frontend, UX
- **Avery**: Data pipeline, storage

#### Review Requirements:
- Code owner must review
- Domain expert must approve
- Alex final sign-off
- Quinn veto power on risk

### 12. PROHIBITED PRACTICES

#### NEVER DO:
- ❌ Deploy to remote servers
- ❌ Use mock data in production
- ❌ Skip tests to save time
- ❌ Merge without verification
- ❌ Hardcode credentials
- ❌ Ignore performance regression
- ❌ Leave TODOs without task IDs
- ❌ Implement partial features
- ❌ Copy-paste without understanding
- ❌ Assume instead of testing

### 13. ENFORCEMENT

#### Automatic Rejection If:
- Verification script fails
- Tests don't pass
- Coverage < 95%
- Performance regresses
- Documentation missing
- Fake implementation detected

#### Immediate Escalation If:
- Remote deployment attempted
- Production mock detected
- Risk limits exceeded
- Circuit breaker triggered
- Data loss occurs

### 14. LOCAL ENVIRONMENT

#### Required Local Services:
```yaml
services:
  postgresql: 15+
  timescaledb: latest
  redis: 7+
  prometheus: latest
  grafana: latest
  docker: 24+
  rust: 1.75+
```

#### Local Directories:
```
/home/hamster/bot4/
├── rust_core/      # Main application
├── data/           # Local data storage
├── logs/           # Application logs
├── backups/        # Local backups
├── scripts/        # Automation
└── config/         # Configuration
```

### 15. SUCCESS METRICS

#### Daily Metrics:
- Tasks completed: Target 5+
- Tests written: Target 20+
- Coverage maintained: >95%
- Performance stable: No regression
- Zero fake implementations

#### Weekly Metrics:
- Phase milestones met
- Integration tests passing
- Benchmarks improving
- Documentation current
- No technical debt

---

## ENFORCEMENT STATEMENT

**These rules are MANDATORY and NON-NEGOTIABLE. Any deviation will result in immediate rejection of work. Quality over speed. Build it right the first time.**

**Remember:**
- Local development gives us full control
- Real implementations only
- Test everything
- Document everything
- Verify everything

**The goal:** Build a trading platform that achieves 200-300% APY with ZERO human intervention, ZERO fake code, and 100% reliability.

---

Last Updated: 2025-01-14
Version: 1.0
Status: ENFORCED