#!/bin/bash
# Task Completion Verification Script
# MUST pass ALL checks before marking any task complete

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "   TASK COMPLETION VERIFICATION SYSTEM   "
echo "=========================================="

PASSED=0
FAILED=0

# Function to check and report
check() {
    local name=$1
    local cmd=$2
    echo -n "Checking $name... "
    
    if eval $cmd > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
        return 1
    fi
}

# 1. Compilation Check
echo -e "\n${YELLOW}=== COMPILATION CHECKS ===${NC}"
check "Rust compilation" "cd /home/hamster/bot4/rust_core && cargo check --all 2>&1 | grep -v warning"
check "No compilation errors" "! (cd /home/hamster/bot4/rust_core && cargo check --all 2>&1 | grep -E 'error\[|error:')"

# 2. Test Execution
echo -e "\n${YELLOW}=== TEST VERIFICATION ===${NC}"
check "All tests compile" "cd /home/hamster/bot4/rust_core && cargo test --all --no-run"
check "All tests pass" "cd /home/hamster/bot4/rust_core && cargo test --all --quiet"

# 3. Fake Implementation Detection
echo -e "\n${YELLOW}=== FAKE DETECTION ===${NC}"
check "No unimplemented!()" "! grep -r 'unimplemented!' --include='*.rs' /home/hamster/bot4/rust_core/src/ /home/hamster/bot4/rust_core/crates/ 2>/dev/null | grep -v '/tests/'"
check "No todo!()" "! grep -r 'todo!' --include='*.rs' /home/hamster/bot4/rust_core/src/ /home/hamster/bot4/rust_core/crates/ 2>/dev/null | grep -v '/tests/'"
check "No panic!()" "! grep -r 'panic!' --include='*.rs' /home/hamster/bot4/rust_core/src/ /home/hamster/bot4/rust_core/crates/ 2>/dev/null | grep -v -E '/tests/|expected'"

# 4. Mock Detection in Production
echo -e "\n${YELLOW}=== MOCK DETECTION ===${NC}"
check "No mock in production" "! grep -r 'mock\|Mock\|MOCK' --include='*.rs' src/ crates/ 2>/dev/null | grep -v -E '/tests/|/test_|_test\.rs'"
check "No fake in production" "! grep -r 'fake\|Fake\|FAKE' --include='*.rs' src/ crates/ 2>/dev/null | grep -v -E '/tests/|/test_|_test\.rs'"
check "No dummy data" "! grep -r 'dummy\|Dummy\|DUMMY' --include='*.rs' src/ crates/ 2>/dev/null | grep -v -E '/tests/|/test_|_test\.rs'"

# 5. Hardcoded Values Check
echo -e "\n${YELLOW}=== HARDCODED VALUES ===${NC}"
check "No price * 0.02" "! grep -r 'price.*\*.*0\.0[0-9]' --include='*.rs' src/ crates/ 2>/dev/null | grep -v -E '/tests/|comment'"
check "No hardcoded IPs" "! grep -r '127\.0\.0\.1\|localhost' --include='*.rs' src/ crates/ 2>/dev/null | grep -v -E '/tests/|/examples/'"

# 6. API Connection Verification
echo -e "\n${YELLOW}=== API CONNECTIVITY ===${NC}"
check "Exchange configs exist" "ls crates/exchanges/*/src/lib.rs 2>/dev/null | wc -l | grep -v '^0$'"
check "WebSocket handlers present" "grep -r 'WebSocket\|websocket' --include='*.rs' crates/ | wc -l | grep -v '^0$'"

# 7. Documentation Check
echo -e "\n${YELLOW}=== DOCUMENTATION ===${NC}"
check "README exists" "test -f README.md"
check "Architecture documented" "test -f ARCHITECTURE.md"
check "Task list updated" "test -f PROJECT_MANAGEMENT_TASK_LIST_V4.md"

# 8. Code Coverage (if available)
echo -e "\n${YELLOW}=== CODE COVERAGE ===${NC}"
if command -v cargo-tarpaulin &> /dev/null; then
    check "Coverage > 60%" "cargo tarpaulin --print-summary 2>/dev/null | grep -E '[6-9][0-9]\.[0-9]+%|100\.'"
else
    echo -e "${YELLOW}Skipping: cargo-tarpaulin not installed${NC}"
fi

# 9. Performance Benchmarks
echo -e "\n${YELLOW}=== PERFORMANCE ===${NC}"
check "Benchmarks compile" "cargo bench --no-run 2>/dev/null"

# Final Report
echo -e "\n=========================================="
echo -e "           VERIFICATION REPORT            "
echo -e "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓✓✓ ALL CHECKS PASSED ✓✓✓${NC}"
    echo -e "${GREEN}Task can be marked as COMPLETE${NC}"
    exit 0
else
    echo -e "\n${RED}✗✗✗ VERIFICATION FAILED ✗✗✗${NC}"
    echo -e "${RED}$FAILED checks failed. Task is NOT complete.${NC}"
    echo -e "${YELLOW}Fix all issues before marking complete.${NC}"
    exit 1
fi