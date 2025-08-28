#!/bin/bash
# MCP Shared Context Loader
# Loads and displays the current shared context for agents

set -euo pipefail

CONTEXT_FILE="/home/hamster/bot4/.mcp/shared_context.json"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}📋 Loading Bot4 Shared Context${NC}"
echo "----------------------------------------"

# Check if context file exists
if [ ! -f "$CONTEXT_FILE" ]; then
    echo -e "${RED}Error: Shared context not found at $CONTEXT_FILE${NC}"
    exit 1
fi

# Parse and display key information
if command -v jq &> /dev/null; then
    # Use jq for pretty parsing if available
    echo -e "\n${YELLOW}Current Task:${NC}"
    jq -r '.current_task | "ID: \(.id)\nDescription: \(.description)\nPhase: \(.phase)\nStatus: \(.status)"' "$CONTEXT_FILE"
    
    echo -e "\n${YELLOW}Sprint Goals:${NC}"
    jq -r '.current_sprint.goals[]' "$CONTEXT_FILE" | while read -r goal; do
        echo "• $goal"
    done
    
    echo -e "\n${YELLOW}Metrics:${NC}"
    jq -r '.metrics | "Test Coverage: \(.test_coverage_percent)%\nDuplication: \(.duplication_percent)%\nLatency: \(.latency_us)μs\nMemory: \(.memory_mb)MB"' "$CONTEXT_FILE"
    
    echo -e "\n${YELLOW}Remaining Duplications:${NC}"
    jq -r '.discovered_issues.duplications | "Total: \(.remaining) (resolved \(.resolved) of \(.total))"' "$CONTEXT_FILE"
    
    echo -e "\n${YELLOW}Agent Status:${NC}"
    jq -r '.agent_status | to_entries[] | "\(.key): \(.value.status)"' "$CONTEXT_FILE"
    
    echo -e "\n${YELLOW}Next Actions:${NC}"
    jq -r '.next_actions[] | "• \(.action) (owner: \(.owner), deadline: \(.deadline))"' "$CONTEXT_FILE"
    
else
    # Fallback to basic display
    echo -e "${YELLOW}Context loaded (install jq for formatted output)${NC}"
    cat "$CONTEXT_FILE"
fi

# Export context path for other scripts
export BOT4_SHARED_CONTEXT="$CONTEXT_FILE"

echo -e "\n----------------------------------------"
echo -e "${GREEN}✅ Context loaded successfully${NC}"
echo -e "Export: BOT4_SHARED_CONTEXT=$CONTEXT_FILE"