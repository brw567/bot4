# Team Review Protocol - MANDATORY for Task Completion

## ⚠️ CRITICAL: No Task is Complete Without Full Team Sign-off

### Review Requirements (ALL MUST PASS)

#### 1. Code Quality Review (Sam - Lead)
- [ ] No `todo!()`, `unimplemented!()`, or fake implementations
- [ ] All error handling implemented (no `.unwrap()` in production)
- [ ] Memory safety verified (no leaks, proper lifetimes)
- [ ] Performance targets met (<1μs decision, <10μs risk)
- [ ] Code follows Rust best practices

#### 2. Risk Review (Quinn - Lead)
- [ ] All operations have circuit breakers
- [ ] Position limits enforced (2% max)
- [ ] Stop-loss mandatory and verified
- [ ] Drawdown protection active (15% max)
- [ ] Kill switches accessible

#### 3. Test Coverage Review (Riley - Lead)
- [ ] Unit tests >95% coverage
- [ ] Integration tests complete
- [ ] Performance benchmarks passing
- [ ] Edge cases covered
- [ ] Stress tests under high load

#### 4. Mathematical Validation (Morgan - Lead)
- [ ] Algorithms mathematically correct
- [ ] No overfitting in ML models
- [ ] Statistical significance verified
- [ ] Backtesting results validated
- [ ] Emotion-free logic confirmed

#### 5. Performance Review (Jordan - Lead)
- [ ] Latency targets achieved
- [ ] Zero allocations in hot path
- [ ] Memory usage bounded
- [ ] Thread contention minimized
- [ ] SIMD optimizations where applicable

#### 6. Integration Review (Casey - Lead)
- [ ] Exchange API compliance
- [ ] Rate limiting implemented
- [ ] Reconnection logic tested
- [ ] Order accuracy verified
- [ ] Partial fill handling correct

#### 7. Data Pipeline Review (Avery - Lead)
- [ ] Data integrity maintained
- [ ] TimescaleDB optimizations applied
- [ ] Monitoring metrics exposed
- [ ] Logging comprehensive
- [ ] Tracing spans correct

#### 8. Documentation Review (Alex - Lead)
- [ ] PROJECT_MANAGEMENT_MASTER.md updated
- [ ] LLM_OPTIMIZED_ARCHITECTURE.md synchronized
- [ ] LLM_TASK_SPECIFICATIONS.md current
- [ ] Code comments adequate
- [ ] API documentation complete

### Review Process

1. **Developer completes implementation**
2. **Self-review against checklist**
3. **Submit for team review with evidence**
4. **Each team member reviews their domain**
5. **Minimum 6/8 approvals required**
6. **Quinn and Sam have VETO power**
7. **Alex breaks ties if needed**

### Sign-off Template

```yaml
task_id: TASK_X.Y.Z
task_name: <name>
developer: <agent_name>
completion_date: <date>

reviews:
  sam_code_quality: APPROVED/REJECTED - <reason>
  quinn_risk: APPROVED/REJECTED - <reason>
  riley_testing: APPROVED/REJECTED - <reason>
  morgan_math: APPROVED/REJECTED - <reason>
  jordan_performance: APPROVED/REJECTED - <reason>
  casey_integration: APPROVED/REJECTED - <reason>
  avery_data: APPROVED/REJECTED - <reason>
  alex_documentation: APPROVED/REJECTED - <reason>

final_status: APPROVED/REJECTED
merged_to_main: YES/NO
```

### Rejection Handling

If ANY critical review fails:
1. Document specific issues
2. Create fix tasks
3. Re-implement
4. Full review cycle repeats
5. No shortcuts allowed

### Remember

> "Build it right the first time" - No rushing, no shortcuts, no compromises.

---

*This protocol is MANDATORY. Violations will result in immediate rollback.*