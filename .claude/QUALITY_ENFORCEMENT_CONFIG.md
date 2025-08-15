# Claude Quality Enforcement Configuration
# MANDATORY - Zero Tolerance for Fake Implementations
# Created: January 14, 2025
# Purpose: Prevent false completion claims and ensure REAL implementations

## 🚨 CRITICAL ENFORCEMENT RULES

### 1. COMPLETION CRITERIA (NO EXCEPTIONS)

A task is ONLY marked complete when ALL of these are true:

```yaml
completion_requirements:
  code:
    - COMPILES without errors
    - ZERO unimplemented!() calls
    - ZERO todo!() macros
    - ZERO panic!("not implemented")
    - NO hardcoded test data
    - NO mock responses
    
  testing:
    - Tests COMPILE
    - Tests PASS (100% success rate)
    - Coverage >80%
    - Integration tests with REAL services
    - NO mocked external dependencies
    
  integration:
    - Connects to REAL APIs (testnet acceptable)
    - Handles REAL data
    - Produces REAL outputs
    - Error handling for ALL edge cases
    - Logging and monitoring integrated
    
  documentation:
    - Code documented
    - Architecture updated
    - Examples provided and WORKING
    - API documentation complete
```

### 2. AUTOMATIC VERIFICATION CHECKS

Before marking ANY task complete, Claude MUST run:

```bash
# Compilation Check
cargo build --all-features 2>&1 | grep -E "error|failed"
# If ANY errors -> TASK INCOMPLETE

# Test Check  
cargo test --all 2>&1 | grep -E "FAILED|error"
# If ANY failures -> TASK INCOMPLETE

# Fake Detection
grep -r "unimplemented!\|todo!\|panic!" src/ --include="*.rs"
# If ANY found -> TASK INCOMPLETE

# Mock Detection
grep -r "mock\|fake\|dummy\|test_" src/ --include="*.rs" | grep -v "tests/"
# If found in production code -> TASK INCOMPLETE

# API Connection Check
grep -r "localhost\|127.0.0.1\|testnet" src/ --include="*.rs"
# Must use REAL endpoints or explicit testnet
```

### 3. TASK BREAKDOWN REQUIREMENTS

Every task MUST be broken down into:

```yaml
task_structure:
  planning:
    - Architecture design document
    - Interface definitions
    - Test specifications
    
  implementation:
    - Core functionality (NO SHORTCUTS)
    - Error handling (COMPLETE)
    - Logging (COMPREHENSIVE)
    - Metrics (PRODUCTION-READY)
    
  testing:
    - Unit tests (>80% coverage)
    - Integration tests (REAL services)
    - Performance tests (meet targets)
    - Failure scenario tests
    
  validation:
    - Code review checklist
    - Functionality verification
    - Production readiness assessment
```

### 4. PROGRESS TRACKING HONESTY

```yaml
progress_reporting:
  statuses:
    NOT_STARTED: "No code written"
    IN_PROGRESS: "Actively developing, tests failing"
    TESTING: "Code complete, testing in progress"
    REVIEW: "Tests passing, pending review"
    COMPLETE: "ALL criteria met, production ready"
    
  forbidden_practices:
    - Marking complete with failing tests
    - Claiming percentages without verification
    - Counting lines of code as progress
    - Including boilerplate in completion metrics
```

### 5. AGENT-SPECIFIC ENFORCEMENT

```yaml
agent_responsibilities:
  sam:
    - MUST run fake detection before ANY completion
    - VETO power on incomplete code
    - Weekly audit of all "completed" tasks
    
  quinn:
    - MUST verify risk controls actually work
    - Test with REAL market scenarios
    - No theoretical implementations
    
  casey:
    - MUST connect to REAL exchange APIs
    - Verify order placement with testnet
    - No mock responses in production
    
  riley:
    - MUST ensure >80% test coverage
    - All tests MUST pass
    - No disabled or skipped tests
    
  morgan:
    - ML models MUST use real data
    - No random number generators
    - Validation on out-of-sample data
```

### 6. DAILY VERIFICATION RITUAL

```bash
#!/bin/bash
# Run EVERY morning before work

echo "=== DAILY INTEGRITY CHECK ==="

# Check compilation
echo "Checking compilation..."
cargo build --all 2>&1 | grep -c "error"
# Must be 0

# Check tests
echo "Running all tests..."
cargo test --all 2>&1 | grep "test result"
# Must show 100% passed

# Check for fakes
echo "Scanning for fake implementations..."
./scripts/validate_no_fakes.py
# Must return clean

# Check claimed vs actual
echo "Verifying claimed completions..."
for task in $(grep "✅ COMPLETED" PROJECT_MANAGEMENT_TASK_LIST_V4.md); do
  # Verify each completed task still works
done

echo "=== INTEGRITY CHECK COMPLETE ==="
```

### 7. RESET PROTOCOL

If starting fresh or resetting:

```yaml
reset_requirements:
  before_reset:
    - Document ALL lessons learned
    - Archive working code
    - List specific failures
    
  clean_start:
    - NO grand claims
    - NO future promises  
    - Start with ONE working feature
    - Build incrementally
    - Test EVERYTHING
    
  first_milestone:
    - ONE real API connection
    - ONE successful trade on testnet
    - ONE strategy with backtesting
    - Then expand from there
```

### 8. FORBIDDEN PHRASES

Claude must NEVER use these phrases:

```
FORBIDDEN:
- "This is production-ready" (unless verified)
- "100% complete" (without test proof)
- "Fully implemented" (without integration proof)
- "Works perfectly" (without error handling)
- "No issues found" (without comprehensive testing)
- "Ready to deploy" (without production validation)

REQUIRED:
- "Compiles with warnings"
- "Tests pass but coverage at 60%"
- "Connects to testnet only"
- "Partial implementation"
- "Needs production validation"
```

### 9. ENFORCEMENT MECHANISMS

```yaml
enforcement:
  automatic_blocks:
    - Git pre-commit hooks checking for fakes
    - CI/CD pipeline with quality gates
    - Automated reversion of false completions
    
  manual_reviews:
    - Daily team standup with proof
    - Weekly audit of completions
    - Monthly full system validation
    
  consequences:
    - False completion = immediate reversion
    - Pattern of false claims = full reset
    - Incomplete work = blocked from new tasks
```

### 10. TRUTH TRACKING METRICS

```yaml
truth_metrics:
  daily_tracking:
    - Lines that compile: X
    - Tests that pass: Y/Z  
    - Real API calls made: N
    - Actual trades executed: M
    
  weekly_reporting:
    - Working features: [list]
    - Broken features: [list]
    - Fake implementations found: N
    - Technical debt items: [list]
    
  monthly_assessment:
    - Can it trade real money? YES/NO
    - Would you trust it with $10k? YES/NO
    - Production readiness: 0-100% (with proof)
```

## 🎯 IMPLEMENTATION CHECKLIST

Before implementing this configuration:

- [ ] Install git hooks for fake detection
- [ ] Setup automated testing pipeline
- [ ] Create verification scripts
- [ ] Document truth tracking process
- [ ] Train all agents on new standards
- [ ] Create rollback procedures
- [ ] Establish audit schedule

## 📝 COMMITMENT STATEMENT

By using this configuration, we commit to:

1. **NEVER** mark incomplete work as done
2. **ALWAYS** verify before claiming completion
3. **IMMEDIATELY** report issues and blockers
4. **HONESTLY** track actual progress
5. **CONTINUOUSLY** improve quality standards

---

**Remember**: It's better to have 10% WORKING than 100% claimed but broken.

**Motto**: "Ship REAL code or ship nothing."