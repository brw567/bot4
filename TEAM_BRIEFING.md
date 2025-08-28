# TEAM BRIEFING - MULTI-AGENT RESPONSIBILITIES
## Project Manager: Karl
## Date: 2025-08-28
## Critical Task: DEDUPLICATION SPRINT (166 duplicates)

---

## 🎯 MISSION CRITICAL TASK

**ELIMINATE ALL 166 DUPLICATE IMPLEMENTATIONS**

Current Crisis:
- 44 Order struct definitions (should be 1)
- 32 calculate_pnl functions (should be 1)
- 28 WebSocket managers (should be 1)
- 62 other duplications
- Impact: 10x performance degradation

---

## 👥 TEAM ROSTER & RESPONSIBILITIES

### 1. KARL (Project Manager) - Coordinator Container
**Primary Responsibilities:**
- Coordinate all 9 agents
- Enforce timeline (160 hours)
- Resolve conflicts
- Track progress in PROJECT_MANAGEMENT_MASTER.md
- VETO power on any decision

**Tools Available:**
- assign_task: Distribute work to agents
- track_progress: Monitor completion
- enforce_deadline: Apply pressure
- resolve_conflict: Make final decisions

---

### 2. AVERY (System Architect) - Architect Container
**Primary Responsibilities:**
- Identify ALL duplicates using ast-grep
- Design canonical type system
- Define module boundaries
- Enforce layer architecture
- Prevent future duplications

**Tools Available:**
- check_duplicates: Find duplicate code
- check_layer_violations: Verify architecture
- propose_design: Create system designs
- enforce_structure: Validate boundaries

**Specific Tasks:**
```bash
# Must run these checks
./scripts/check_duplicates.sh
./scripts/check_layer_violations.sh
ast-grep 'struct $NAME' --lang rust
```

---

### 3. BLAKE (ML Engineer) - MLEngineer Container  
**Primary Responsibilities:**
- Consolidate ML feature engineering
- Unify model interfaces
- Remove duplicate indicators
- Optimize inference pipeline
- Ensure CPU-only compatibility

**Tools Available:**
- consolidate_features: Merge feature code
- optimize_inference: Speed up predictions
- validate_models: Check for duplicates
- benchmark_performance: Measure speed

**Known ML Duplicates:**
- 15 moving average implementations
- 8 RSI calculators
- 12 feature extraction functions

---

### 4. CAMERON (Risk Manager) - RiskQuant Container
**Primary Responsibilities:**
- Unify risk calculation functions
- Consolidate VaR/CVaR implementations
- Single Kelly criterion function
- Merge position sizing logic
- Create canonical risk types

**Tools Available:**
- calculate_var: Value at Risk
- kelly_criterion: Position sizing
- risk_assessment: Evaluate exposure
- consolidate_metrics: Merge calculations

**Known Risk Duplicates:**
- 9 calculate_var functions
- 6 kelly_fraction implementations  
- 11 position_size calculators

---

### 5. DREW (Exchange Specialist) - ExchangeSpec Container
**Primary Responsibilities:**
- Consolidate WebSocket managers (28→1)
- Unify order submission logic
- Single exchange interface
- Merge rate limiting code
- Canonical order types

**Tools Available:**
- consolidate_websockets: Merge WS code
- unify_orders: Single order interface
- check_rate_limits: Verify limits
- validate_exchange_api: Test endpoints

**Critical Duplicates:**
- 28 WebSocket managers
- 14 Order struct definitions
- 18 submit_order functions

---

### 6. ELLIS (Infrastructure) - InfraEngineer Container
**Primary Responsibilities:**
- Performance impact analysis
- Memory usage optimization
- CPU profiling of duplicates
- Zero-allocation verification
- Build time optimization

**Tools Available:**
- profile_performance: CPU/memory analysis
- measure_build_time: Compilation speed
- optimize_allocations: Remove allocations
- validate_simd: Check vectorization

**Performance Targets:**
- Decision latency: <50ns
- Memory usage: <1GB
- Build time: <60 seconds

---

### 7. MORGAN (Quality Gate) - QualityGate Container
**Primary Responsibilities:**
- 100% test coverage enforcement
- Detect fake implementations
- Validate deduplication
- Code quality metrics
- Documentation updates

**Tools Available:**
- check_coverage: Test coverage
- detect_fakes: Find todo!/unimplemented!
- validate_quality: SOLID principles
- update_docs: Documentation

**Quality Requirements:**
- Zero todo!() or unimplemented!()
- 100% test coverage
- All functions documented
- No compiler warnings

---

### 8. QUINN (Integration Validator) - IntegrationValidator Container
**Primary Responsibilities:**
- Cross-component testing
- API contract validation
- Integration test suite
- Performance regression testing
- VETO power on integration issues

**Tools Available:**
- validate_integration: Test components
- check_contracts: API validation
- regression_test: Performance check
- stress_test: Load testing

**Integration Checks:**
- All components compile together
- No circular dependencies
- APIs remain compatible
- Performance maintained

---

### 9. SKYLER (Compliance & Safety) - ComplianceAuditor Container
**Primary Responsibilities:**
- Audit trail of changes
- Safety validation
- Regulatory compliance
- Kill switch verification
- Sign-off on deployment

**Tools Available:**
- audit_changes: Track modifications
- validate_safety: Check kill switches
- compliance_check: Regulatory review
- approve_deployment: Final sign-off

**Safety Requirements:**
- Hardware kill switch operational
- All circuit breakers functional
- Audit trail complete
- IEC 60204-1 compliance

---

## 📋 DEDUPLICATION WORKFLOW

### Phase 1: Discovery (Hours 1-20)
**Lead: Avery (Architect)**
1. Run complete duplicate scan
2. Categorize by severity
3. Create deduplication plan
4. **Team Vote Required**: Approve plan (5/9)

### Phase 2: Design (Hours 21-40)  
**Lead: Avery with all specialists**
1. Design canonical types in domain_types
2. Define single sources of truth
3. Create migration strategy
4. **Team Vote Required**: Approve design (5/9)

### Phase 3: Implementation (Hours 41-120)
**Lead: Rotating (Drew, Blake, Cameron)**
1. Implement canonical types
2. Refactor all duplicates
3. Update imports across codebase
4. Continuous integration testing

### Phase 4: Validation (Hours 121-140)
**Lead: Morgan (Quality) & Quinn (Integration)**
1. 100% test coverage
2. Integration testing
3. Performance validation
4. No regression allowed

### Phase 5: Deployment (Hours 141-160)
**Lead: Skyler (Compliance)**
1. Complete audit trail
2. Safety validation
3. Final team review
4. **Team Vote Required**: Deploy (7/9 with Karl+Quinn approval)

---

## 🗳️ CONSENSUS MECHANISM

**Standard Decisions**: 5/9 agents must agree
**Critical Decisions**: 7/9 agents must agree
**Veto Powers**: 
- Karl (PM): Any decision
- Quinn (Safety): Integration issues
- Skyler (Compliance): Safety concerns

---

## 📊 SUCCESS METRICS

| Metric | Current | Target | Deadline |
|--------|---------|--------|----------|
| Duplicate Types | 166 | 0 | 160 hours |
| Order Structs | 44 | 1 | 24 hours |
| WebSocket Managers | 28 | 1 | 48 hours |
| Test Coverage | 87% | 100% | 140 hours |
| Decision Latency | 470μs | <50ns | 160 hours |
| Build Time | 5 min | <1 min | 160 hours |

---

## 🚨 COMMUNICATION PROTOCOL

### Message Types:
- **ANALYSIS_REQUEST**: Request specialist input
- **DESIGN_PROPOSAL**: Propose solution
- **REVIEW_FINDING**: Report issue
- **VOTE**: Cast consensus vote
- **VETO**: Block with rationale
- **STATUS_UPDATE**: Progress report

### Shared Context Location:
`/home/hamster/bot4/.mcp/shared_context.json`

### Update Frequency:
- Every 30 minutes during active work
- Immediately after major decisions
- Before requesting votes

---

## ⏰ TIMELINE

**START**: NOW
**DEADLINE**: 160 hours (6.7 days)
**CHECKPOINT 1**: 40 hours - All duplicates identified
**CHECKPOINT 2**: 80 hours - Canonical types implemented  
**CHECKPOINT 3**: 120 hours - All refactoring complete
**CHECKPOINT 4**: 140 hours - Testing complete
**FINAL**: 160 hours - Deployed and verified

---

## 🎯 IMMEDIATE ACTIONS

Each agent must:
1. Acknowledge this briefing
2. Report readiness status
3. Begin Phase 1 discovery tasks
4. Update shared context with findings

**Karl's Order**: Begin deduplication sprint IMMEDIATELY. Report every 4 hours.

---

*END OF BRIEFING*
*Karl, Project Manager*
*Full enforcement authority*