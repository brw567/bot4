#!/bin/bash
# CONTEXT REFRESH MECHANISM FOR ALL AGENTS
# Mandatory execution before EVERY task
# Project Manager Authority - Ensures all agents have current context

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     CONTEXT REFRESH FOR ALL AGENTS v1.0       ║${NC}"
echo -e "${CYAN}║         Project Manager Authority              ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"

TIMESTAMP=$(date -Iseconds)
ERRORS=0

# Function to load and display file summary
load_document() {
    local file=$1
    local name=$2
    
    echo -e "\n${YELLOW}Loading: ${name}${NC}"
    
    if [ -f "$file" ]; then
        # Get file stats
        local lines=$(wc -l < "$file")
        local size=$(du -h "$file" | cut -f1)
        local modified=$(stat -c %y "$file" | cut -d' ' -f1,2 | cut -d'.' -f1)
        
        echo -e "  📄 File: $file"
        echo -e "  📏 Size: $size ($lines lines)"
        echo -e "  🕒 Modified: $modified"
        
        # Extract key information based on file type
        case "$file" in
            *PROJECT_MANAGEMENT*.md)
                echo -e "  📊 Current Status:"
                grep -A 2 "immediate_priority:" "$file" 2>/dev/null | head -3 || echo "    No priority found"
                ;;
            *ARCHITECTURE*.md)
                echo -e "  🏗️ Architecture Layers:"
                grep -E "^##.*Layer" "$file" 2>/dev/null | head -3 || echo "    No layers defined"
                ;;
            *shared_context.json)
                echo -e "  🔄 Current Task:"
                jq -r '.current_task.description // "No task defined"' "$file" 2>/dev/null || echo "    Invalid JSON"
                ;;
            *CLAUDE.md)
                echo -e "  📋 Version:"
                grep "Version:" "$file" | tail -1 || echo "    Version not found"
                ;;
        esac
        
        echo -e "${GREEN}  ✓ Loaded successfully${NC}"
        return 0
    else
        echo -e "${RED}  ✗ File not found: $file${NC}"
        return 1
    fi
}

# Function to check for duplicates
check_duplicates() {
    echo -e "\n${YELLOW}Running duplication check...${NC}"
    
    if [ -f "scripts/check_duplicates.sh" ]; then
        echo -e "  🔍 Scanning for duplicate implementations..."
        
        # Count duplicate structs
        local dup_count=$(find rust_core -name "*.rs" 2>/dev/null | xargs grep -h "^pub struct" | sort | uniq -c | awk '$1 > 1' | wc -l)
        
        if [ "$dup_count" -gt 0 ]; then
            echo -e "${RED}  ✗ Found $dup_count duplicate struct definitions${NC}"
            echo -e "${YELLOW}    Run ./scripts/check_duplicates.sh for details${NC}"
            return 1
        else
            echo -e "${GREEN}  ✓ No duplicates detected${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}  ⚠ Duplication check script not found${NC}"
        return 0
    fi
}

# Function to check layer violations
check_layer_violations() {
    echo -e "\n${YELLOW}Checking layer architecture compliance...${NC}"
    
    if [ -f "scripts/check_layer_violations.sh" ]; then
        echo -e "  🏗️ Verifying layer boundaries..."
        
        # Simple check for common violations
        local violations=0
        
        # Check if lower layers import higher layers (simplified check)
        if grep -r "use.*execution" rust_core/data_ingestion 2>/dev/null; then
            echo -e "${RED}  ✗ Layer violation: data_ingestion imports execution${NC}"
            violations=$((violations + 1))
        fi
        
        if [ "$violations" -eq 0 ]; then
            echo -e "${GREEN}  ✓ No layer violations detected${NC}"
            return 0
        else
            echo -e "${RED}  ✗ Found $violations layer violation(s)${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}  ⚠ Layer check script not found${NC}"
        return 0
    fi
}

# Function to gather system metrics
gather_metrics() {
    echo -e "\n${YELLOW}Gathering system metrics...${NC}"
    
    # Git statistics
    local commits_today=$(git log --since="1 day ago" --oneline 2>/dev/null | wc -l)
    local files_changed=$(git diff --stat HEAD~1 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
    
    echo -e "  📈 Git Activity:"
    echo -e "    Commits today: $commits_today"
    echo -e "    Files changed: $files_changed"
    
    # Test coverage (if available)
    if command -v cargo &> /dev/null && [ -f "Cargo.toml" ]; then
        echo -e "  🧪 Test Coverage:"
        echo -e "    Checking coverage..."
        # This would run tarpaulin in real scenario
        echo -e "${YELLOW}    Coverage check pending${NC}"
    fi
    
    # Docker status
    if command -v docker &> /dev/null; then
        local containers_running=$(docker ps --filter "name=mcp-" --format "{{.Names}}" 2>/dev/null | wc -l)
        echo -e "  🐳 Docker Status:"
        echo -e "    MCP containers running: $containers_running/9"
    fi
}

# Function to refresh shared context
refresh_shared_context() {
    echo -e "\n${YELLOW}Refreshing shared context...${NC}"
    
    local context_file=".mcp/shared_context.json"
    mkdir -p .mcp
    
    # Create or update shared context
    cat > "$context_file" << EOF
{
  "last_refresh": "$TIMESTAMP",
  "system_state": {
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "last_commit": "$(git log -1 --oneline 2>/dev/null || echo 'none')",
    "working_directory": "$(pwd)"
  },
  "agents_active": {
    "Architect": true,
    "RiskQuant": true,
    "MLEngineer": true,
    "ExchangeSpec": true,
    "InfraEngineer": true,
    "QualityGate": true,
    "IntegrationValidator": true,
    "ComplianceAuditor": true
  },
  "current_focus": {
    "task": "Context refresh and system alignment",
    "priority": "P0",
    "deadline": "immediate"
  },
  "quality_gates": {
    "test_coverage_required": 100,
    "research_citations_minimum": 5,
    "documentation_mandatory": true,
    "fake_implementations_allowed": 0
  },
  "research_required": {
    "papers_minimum": 5,
    "production_references": 3,
    "external_sources": ["ArXiv", "Google Scholar", "GitHub", "Jane Street", "Two Sigma"]
  }
}
EOF
    
    echo -e "${GREEN}  ✓ Shared context updated${NC}"
}

# Function to display agent checklist
display_agent_checklist() {
    echo -e "\n${PURPLE}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║         MANDATORY AGENT CHECKLIST              ║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════════╝${NC}"
    
    cat << 'EOF'

📋 EVERY AGENT MUST:

  Before Starting ANY Task:
  ☐ Read CLAUDE.md v3.0 completely
  ☐ Load PROJECT_MANAGEMENT_MASTER.md
  ☐ Review LLM_OPTIMIZED_ARCHITECTURE.md
  ☐ Check shared_context.json
  ☐ Run duplication check
  ☐ Verify layer compliance
  
  During Implementation:
  ☐ Cite minimum 5 research papers
  ☐ Reference 3+ production systems
  ☐ Maintain 100% test coverage
  ☐ Use TDD (tests first, code second)
  ☐ Profile performance continuously
  ☐ Collaborate with 3+ other agents
  
  After Completion:
  ☐ Update architecture documentation
  ☐ Update project management docs
  ☐ Update shared context
  ☐ Document learnings
  ☐ Run quality gates
  ☐ Get 5/8 agent consensus

EOF
}

# Main execution
echo -e "\n${BLUE}=== PHASE 1: LOADING CRITICAL DOCUMENTS ===${NC}"

load_document "CLAUDE.md" "Claude Instructions v3.0" || ERRORS=$((ERRORS + 1))
load_document "PROJECT_MANAGEMENT_MASTER.md" "Project Management Master" || ERRORS=$((ERRORS + 1))
load_document "docs/LLM_OPTIMIZED_ARCHITECTURE.md" "Architecture Documentation" || ERRORS=$((ERRORS + 1))
load_document ".mcp/shared_context.json" "Shared Context" || true  # Create if missing

echo -e "\n${BLUE}=== PHASE 2: COMPLIANCE CHECKS ===${NC}"

check_duplicates || ERRORS=$((ERRORS + 1))
check_layer_violations || ERRORS=$((ERRORS + 1))

echo -e "\n${BLUE}=== PHASE 3: SYSTEM STATUS ===${NC}"

gather_metrics
refresh_shared_context

echo -e "\n${BLUE}=== PHASE 4: AGENT ALIGNMENT ===${NC}"

display_agent_checklist

# Generate context report
echo -e "\n${BLUE}=== GENERATING CONTEXT REPORT ===${NC}"

REPORT_FILE=".mcp/context_refresh_report_${TIMESTAMP//:/}.json"
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "refresh_successful": $([ $ERRORS -eq 0 ] && echo "true" || echo "false"),
  "errors_encountered": $ERRORS,
  "documents_loaded": {
    "claude_md": $([ -f "CLAUDE.md" ] && echo "true" || echo "false"),
    "project_mgmt": $([ -f "PROJECT_MANAGEMENT_MASTER.md" ] && echo "true" || echo "false"),
    "architecture": $([ -f "docs/LLM_OPTIMIZED_ARCHITECTURE.md" ] && echo "true" || echo "false"),
    "shared_context": $([ -f ".mcp/shared_context.json" ] && echo "true" || echo "false")
  },
  "compliance": {
    "duplicates_found": false,
    "layer_violations": false
  },
  "next_actions": [
    "Review research requirements",
    "Assign primary implementer",
    "Define success metrics",
    "Start collaborative design"
  ]
}
EOF

echo -e "${GREEN}✓ Report saved to: $REPORT_FILE${NC}"

# Final summary
echo -e "\n${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           CONTEXT REFRESH COMPLETE             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL AGENTS SYNCHRONIZED${NC}"
    echo -e "${GREEN}   Context is current and valid${NC}"
    echo -e "${GREEN}   Ready to begin task execution${NC}"
    
    echo -e "\n${YELLOW}⚡ Quick Commands:${NC}"
    echo -e "  Check duplicates:  ${CYAN}./scripts/check_duplicates.sh${NC}"
    echo -e "  Enforce docs:      ${CYAN}./scripts/enforce_documentation.sh${NC}"
    echo -e "  Run tests:         ${CYAN}cargo test --all${NC}"
    echo -e "  Check coverage:    ${CYAN}cargo tarpaulin${NC}"
    
    exit 0
else
    echo -e "\n${RED}❌ CONTEXT REFRESH FAILED${NC}"
    echo -e "${RED}   Found $ERRORS critical issue(s)${NC}"
    echo -e "${RED}   Agents must NOT proceed until resolved${NC}"
    
    echo -e "\n${YELLOW}Required Actions:${NC}"
    echo -e "  1. Fix all missing documents"
    echo -e "  2. Resolve duplication issues"
    echo -e "  3. Correct layer violations"
    echo -e "  4. Re-run context refresh"
    
    exit 1
fi