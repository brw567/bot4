#!/bin/bash
# MANDATORY DOCUMENTATION ENFORCEMENT SYSTEM
# Project Manager Authority - Zero Tolerance Policy
# NO TASK COMPLETION WITHOUT DOCUMENTATION UPDATES

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}========================================${NC}"
echo -e "${PURPLE}   DOCUMENTATION ENFORCEMENT SYSTEM${NC}"
echo -e "${PURPLE}   Project Manager Authority v3.0${NC}"
echo -e "${PURPLE}========================================${NC}"

# Function to check if file was modified
check_file_modified() {
    local file=$1
    if git diff HEAD --name-only | grep -q "$file"; then
        echo -e "${GREEN}✓${NC} $file has been modified"
        return 0
    else
        echo -e "${RED}✗${NC} $file NOT modified - BLOCKING COMMIT"
        return 1
    fi
}

# Function to check for research citations
check_research_citations() {
    local file=$1
    local min_citations=$2
    
    # Count citations (looking for patterns like [1], (Author, Year), arXiv:, doi:)
    local citation_count=$(grep -E '\[[0-9]+\]|\([A-Z][a-z]+.*[0-9]{4}\)|arXiv:|doi:|http.*paper|http.*arxiv' "$file" | wc -l)
    
    if [ "$citation_count" -ge "$min_citations" ]; then
        echo -e "${GREEN}✓${NC} Found $citation_count research citations (minimum: $min_citations)"
        return 0
    else
        echo -e "${RED}✗${NC} Only $citation_count citations found (minimum required: $min_citations)"
        return 1
    fi
}

# Function to check for forbidden patterns
check_forbidden_patterns() {
    echo -e "\n${YELLOW}Checking for forbidden patterns...${NC}"
    
    local violations=0
    
    # Check for TODO/unimplemented
    if grep -r "todo!\|unimplemented!\|panic!(\"not implemented" --include="*.rs" --include="*.py" rust_core/ src/ 2>/dev/null; then
        echo -e "${RED}✗ CRITICAL: Fake implementations detected!${NC}"
        violations=$((violations + 1))
    fi
    
    # Check for placeholder comments
    if grep -r "PLACEHOLDER\|STUB\|FAKE\|DUMMY" --include="*.rs" --include="*.py" rust_core/ src/ 2>/dev/null; then
        echo -e "${RED}✗ CRITICAL: Placeholder code detected!${NC}"
        violations=$((violations + 1))
    fi
    
    if [ "$violations" -eq 0 ]; then
        echo -e "${GREEN}✓ No forbidden patterns found${NC}"
        return 0
    else
        echo -e "${RED}BLOCKED: $violations violation(s) found${NC}"
        return 1
    fi
}

# Function to check test coverage
check_test_coverage() {
    echo -e "\n${YELLOW}Checking test coverage...${NC}"
    
    if command -v cargo &> /dev/null; then
        # Run tarpaulin for Rust projects
        if [ -f "Cargo.toml" ]; then
            echo "Running test coverage analysis..."
            coverage_output=$(cargo tarpaulin --print-summary 2>/dev/null || echo "Coverage: 0%")
            coverage=$(echo "$coverage_output" | grep "Coverage" | sed 's/.*Coverage: \([0-9.]*\)%.*/\1/')
            
            if [ "$(echo "$coverage >= 100" | bc)" -eq 1 ]; then
                echo -e "${GREEN}✓ Test coverage: ${coverage}%${NC}"
                return 0
            else
                echo -e "${RED}✗ Test coverage only ${coverage}% (100% required)${NC}"
                return 1
            fi
        fi
    fi
    
    echo -e "${YELLOW}⚠ Could not verify test coverage${NC}"
    return 0
}

# Main enforcement checks
ERRORS=0

echo -e "\n${BLUE}=== MANDATORY DOCUMENTATION CHECKS ===${NC}"

# 1. Check Architecture Documentation
echo -e "\n${YELLOW}1. Architecture Documentation${NC}"
if ! check_file_modified "docs/LLM_OPTIMIZED_ARCHITECTURE.md"; then
    ERRORS=$((ERRORS + 1))
fi

# 2. Check Project Management Documentation
echo -e "\n${YELLOW}2. Project Management Documentation${NC}"
if ! check_file_modified "PROJECT_MANAGEMENT_MASTER.md"; then
    ERRORS=$((ERRORS + 1))
fi

# 3. Check Shared Context
echo -e "\n${YELLOW}3. Shared Context Update${NC}"
if ! check_file_modified ".mcp/shared_context.json"; then
    # Create if doesn't exist
    if [ ! -f ".mcp/shared_context.json" ]; then
        mkdir -p .mcp
        cat > .mcp/shared_context.json << 'EOF'
{
  "last_update": "$(date -Iseconds)",
  "current_task": "undefined",
  "agents_involved": [],
  "research_citations": [],
  "quality_metrics": {
    "test_coverage": 0,
    "research_count": 0
  }
}
EOF
        echo -e "${YELLOW}Created shared_context.json - please update it${NC}"
    fi
    ERRORS=$((ERRORS + 1))
fi

# 4. Check Research Citations
echo -e "\n${YELLOW}4. Research Citations Check${NC}"
MIN_CITATIONS=3
for file in docs/*.md README.md; do
    if [ -f "$file" ] && git diff HEAD --name-only | grep -q "$file"; then
        if ! check_research_citations "$file" $MIN_CITATIONS; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

# 5. Check for Forbidden Patterns
check_forbidden_patterns || ERRORS=$((ERRORS + 1))

# 6. Check Test Coverage
check_test_coverage || ERRORS=$((ERRORS + 1))

# 7. Check Agent Learnings
echo -e "\n${YELLOW}7. Agent Learnings Documentation${NC}"
if [ ! -f "docs/AGENT_LEARNINGS.md" ]; then
    cat > docs/AGENT_LEARNINGS.md << 'EOF'
# AGENT LEARNINGS DOCUMENTATION

## Latest Task: [Task Name]
**Date**: $(date -I)
**Agents Involved**: [List agents]

### What Worked Well
- [Success point 1]
- [Success point 2]

### What Failed and Why
- [Failure point 1 with root cause]
- [Failure point 2 with root cause]

### External Resources That Helped
1. [Resource 1 with link]
2. [Resource 2 with link]
3. [Resource 3 with link]

### Patterns to Replicate
- [Pattern 1]
- [Pattern 2]

### Anti-Patterns to Avoid
- [Anti-pattern 1]
- [Anti-pattern 2]
EOF
    echo -e "${YELLOW}Created AGENT_LEARNINGS.md template - please update it${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 8. Generate Documentation Report
echo -e "\n${BLUE}=== DOCUMENTATION COMPLIANCE REPORT ===${NC}"
cat > .mcp/documentation_report.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "checks_performed": 8,
  "errors_found": $ERRORS,
  "compliance": $([ $ERRORS -eq 0 ] && echo "true" || echo "false"),
  "details": {
    "architecture_updated": $(check_file_modified "docs/LLM_OPTIMIZED_ARCHITECTURE.md" && echo "true" || echo "false"),
    "project_mgmt_updated": $(check_file_modified "PROJECT_MANAGEMENT_MASTER.md" && echo "true" || echo "false"),
    "shared_context_updated": $(check_file_modified ".mcp/shared_context.json" && echo "true" || echo "false"),
    "test_coverage": "$(cargo tarpaulin --print-summary 2>/dev/null | grep Coverage | sed 's/.*Coverage: \([0-9.]*\)%.*/\1/' || echo "0")",
    "research_citations": "$MIN_CITATIONS"
  }
}
EOF

# Final verdict
echo -e "\n${PURPLE}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL DOCUMENTATION CHECKS PASSED${NC}"
    echo -e "${GREEN}Task completion approved by Project Manager${NC}"
    exit 0
else
    echo -e "${RED}❌ DOCUMENTATION COMPLIANCE FAILED${NC}"
    echo -e "${RED}Found $ERRORS violation(s)${NC}"
    echo -e "${RED}${NC}"
    echo -e "${RED}PROJECT MANAGER VERDICT: TASK NOT COMPLETE${NC}"
    echo -e "${RED}${NC}"
    echo -e "${YELLOW}Required Actions:${NC}"
    echo -e "  1. Update all documentation files"
    echo -e "  2. Add minimum $MIN_CITATIONS research citations"
    echo -e "  3. Achieve 100% test coverage"
    echo -e "  4. Remove all fake implementations"
    echo -e "  5. Update shared context with findings"
    echo -e "${PURPLE}========================================${NC}"
    exit 1
fi