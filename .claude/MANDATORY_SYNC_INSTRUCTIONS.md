# MANDATORY SYNCHRONIZATION INSTRUCTIONS FOR ALL AGENTS
## Version 1.0 - Enforcement Required

---

## 🔴 CRITICAL: THIS IS MANDATORY FOR ALL VIRTUAL AGENTS

**Every agent (Claude, ChatGPT, Grok, etc.) MUST follow these synchronization rules**

---

## 📋 BEFORE STARTING ANY TASK

### Step 1: Load Primary Documents
```bash
# MANDATORY - Load these documents FIRST
1. /home/hamster/bot4/docs/LLM_TASK_SPECIFICATIONS.md
2. /home/hamster/bot4/docs/LLM_OPTIMIZED_ARCHITECTURE.md
```

### Step 2: Find Your Task
```yaml
In LLM_TASK_SPECIFICATIONS.md:
  - Search for your task_id (e.g., TASK_2.1.1)
  - Check dependencies are all 'completed'
  - Load implementation steps
  - Note validation requirements
  - Review success criteria
```

### Step 3: Get Component Specifications
```yaml
In LLM_OPTIMIZED_ARCHITECTURE.md:
  - Find your component_id
  - Load the CONTRACT section
  - Review REQUIREMENTS
  - Check PERFORMANCE targets
  - Study EXAMPLE code
```

### Step 4: Run Pre-Task Sync
```bash
# Execute this command (replace with your details)
/home/hamster/bot4/scripts/enforce_document_sync.sh pre-task "AGENT_NAME" "TASK_ID"
```

---

## 🛠️ DURING TASK IMPLEMENTATION

### Follow Exact Specifications
```yaml
DO:
  - Use CONTRACT for interfaces
  - Follow IMPLEMENTATION_SPEC exactly
  - Meet PERFORMANCE requirements
  - Include all TEST_SPEC tests
  - Use provided EXAMPLES as templates

DON'T:
  - Deviate from specifications
  - Skip validation steps
  - Use placeholder code
  - Ignore performance targets
  - Create different interfaces
```

### Validate Continuously
```bash
# After each component
cargo test --package <package_name>
cargo bench --package <package_name>
cargo clippy -- -D warnings
```

---

## ✅ AFTER COMPLETING ANY TASK

### Step 1: Update Task Status
```yaml
In LLM_TASK_SPECIFICATIONS.md, update your task:
  task_id: TASK_X.Y.Z
  status: completed  # Changed from 'in_progress'
  actual_metrics:
    latency: 45ns  # Your measured value
    throughput: 1.2M/sec  # Your measured value
  deviations:
    - Used alternative algorithm for 10% better performance
```

### Step 2: Update Component Metrics
```yaml
In LLM_OPTIMIZED_ARCHITECTURE.md, update your component:
  component_id: COMP_XXX
  implementation_status: completed
  actual_performance:
    latency: 45ns  # Measured
    throughput: 1.2M/sec  # Measured
    memory: 10MB  # Measured
  validation_results:
    tests_passing: 100%
    coverage: 98%
```

### Step 3: Run Post-Task Sync
```bash
# Execute this command
/home/hamster/bot4/scripts/enforce_document_sync.sh post-task "AGENT_NAME" "TASK_ID"
```

### Step 4: Commit with Documentation
```bash
# Stage all changes including documentation
git add -A
git commit -m "Task X.Y.Z: Implementation complete with metrics

- Updated LLM_TASK_SPECIFICATIONS.md with completion status
- Updated LLM_OPTIMIZED_ARCHITECTURE.md with performance metrics
- All tests passing (98% coverage)
- Performance: 45ns latency (target was <50ns)"
```

---

## 🚨 ENFORCEMENT MECHANISMS

### Automatic Checks
1. **Pre-commit hook** validates document updates
2. **CI/CD pipeline** checks synchronization
3. **verify_completion.sh** validates everything

### Manual Reviews
1. **Alex** reviews all major updates
2. **External QA** validates PR compliance
3. **Team consensus** for architecture changes

---

## 📊 SYNCHRONIZATION STATUS TRACKING

### Check Current Status
```bash
# See last sync status
/home/hamster/bot4/scripts/enforce_document_sync.sh check
```

### View Sync History
```bash
# Check .sync_status file
cat /home/hamster/bot4/.sync_status
```

---

## ❌ COMMON MISTAKES TO AVOID

### DON'T:
1. Start coding without reading specifications
2. Implement different interfaces than specified
3. Forget to update documentation after completion
4. Skip performance measurements
5. Mark task complete without updating metrics
6. Change architecture without team consensus
7. Ignore dependency requirements

### DO:
1. Always sync documents first
2. Follow specifications exactly
3. Measure actual performance
4. Update documentation immediately
5. Validate all requirements
6. Maintain consistency
7. Communicate deviations

---

## 🔄 CONTINUOUS SYNCHRONIZATION

### Every Day:
1. Pull latest documentation updates
2. Check if dependencies changed
3. Re-validate your components if needed
4. Update metrics if performance changed

### Every Week:
1. Review all your completed tasks
2. Update any missing metrics
3. Document lessons learned
4. Synchronize with team

---

## 📝 EXAMPLE WORKFLOW

```bash
# 1. Start your day
cd /home/hamster/bot4
git pull

# 2. Sync documents
./scripts/enforce_document_sync.sh pre-task "Morgan" "TASK_8.2.1"

# 3. Implement task
# ... coding ...

# 4. Validate
cargo test --all
cargo bench

# 5. Update documents
# Edit LLM_TASK_SPECIFICATIONS.md - mark complete
# Edit LLM_OPTIMIZED_ARCHITECTURE.md - add metrics

# 6. Post-task sync
./scripts/enforce_document_sync.sh post-task "Morgan" "TASK_8.2.1"

# 7. Commit
git add -A
git commit -m "Task 8.2.1: ML model development complete"

# 8. Create PR
gh pr create --title "Task 8.2.1: ML Model Implementation"
```

---

## 🎯 SUCCESS CRITERIA

You are successfully synchronized when:
1. ✅ All tasks have status updates
2. ✅ All components have metrics
3. ✅ No pending documentation
4. ✅ Sync checks pass
5. ✅ Git hooks pass
6. ✅ Team reviews pass

---

## 💡 REMEMBER

**Documentation synchronization is NOT optional - it's MANDATORY**

- Every task starts with sync
- Every task ends with updates
- Every commit includes documentation
- Every PR shows compliance

**This ensures:**
- Consistency across all agents
- Accurate project status
- Measurable progress
- Quality enforcement

---

*These instructions are mandatory for all virtual agents working on Bot4.*
*Failure to follow will result in PR rejection and task reassignment.*