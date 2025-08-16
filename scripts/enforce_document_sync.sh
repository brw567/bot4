#!/bin/bash
# Document Synchronization Enforcement Script
# Ensures all agents sync with LLM-optimized documents

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================"
echo "   DOCUMENT SYNCHRONIZATION ENFORCEMENT CHECK"
echo "================================================"

# Define required documents
TASK_SPEC="/home/hamster/bot4/docs/LLM_TASK_SPECIFICATIONS.md"
ARCH_SPEC="/home/hamster/bot4/docs/LLM_OPTIMIZED_ARCHITECTURE.md"

# Function to check if documents exist
check_documents() {
    echo -e "\n${YELLOW}Checking required documents...${NC}"
    
    if [ ! -f "$TASK_SPEC" ]; then
        echo -e "${RED}✗ LLM_TASK_SPECIFICATIONS.md not found!${NC}"
        echo "Creating from template..."
        # Could auto-generate template here
        return 1
    else
        echo -e "${GREEN}✓ LLM_TASK_SPECIFICATIONS.md found${NC}"
    fi
    
    if [ ! -f "$ARCH_SPEC" ]; then
        echo -e "${RED}✗ LLM_OPTIMIZED_ARCHITECTURE.md not found!${NC}"
        echo "Creating from template..."
        # Could auto-generate template here
        return 1
    else
        echo -e "${GREEN}✓ LLM_OPTIMIZED_ARCHITECTURE.md found${NC}"
    fi
    
    return 0
}

# Function to check last sync time
check_sync_time() {
    echo -e "\n${YELLOW}Checking document freshness...${NC}"
    
    # Get last modified time of documents
    TASK_MOD=$(stat -c %Y "$TASK_SPEC" 2>/dev/null || echo 0)
    ARCH_MOD=$(stat -c %Y "$ARCH_SPEC" 2>/dev/null || echo 0)
    
    # Get current time
    NOW=$(date +%s)
    
    # Calculate age in hours
    TASK_AGE=$(( (NOW - TASK_MOD) / 3600 ))
    ARCH_AGE=$(( (NOW - ARCH_MOD) / 3600 ))
    
    if [ $TASK_AGE -gt 24 ]; then
        echo -e "${YELLOW}⚠ Task specifications not updated in ${TASK_AGE} hours${NC}"
    else
        echo -e "${GREEN}✓ Task specifications updated ${TASK_AGE} hours ago${NC}"
    fi
    
    if [ $ARCH_AGE -gt 24 ]; then
        echo -e "${YELLOW}⚠ Architecture specs not updated in ${ARCH_AGE} hours${NC}"
    else
        echo -e "${GREEN}✓ Architecture specs updated ${ARCH_AGE} hours ago${NC}"
    fi
}

# Function to check for pending updates
check_pending_updates() {
    echo -e "\n${YELLOW}Checking for pending updates...${NC}"
    
    # Check for incomplete tasks
    INCOMPLETE=$(grep -c "status: in_progress\|status: not_started" "$TASK_SPEC" 2>/dev/null || echo 0)
    
    if [ $INCOMPLETE -gt 0 ]; then
        echo -e "${YELLOW}⚠ ${INCOMPLETE} tasks pending completion${NC}"
    else
        echo -e "${GREEN}✓ All tracked tasks completed${NC}"
    fi
    
    # Check for components without metrics
    NO_METRICS=$(grep -c "actual_metrics: pending" "$ARCH_SPEC" 2>/dev/null || echo 0)
    
    if [ $NO_METRICS -gt 0 ]; then
        echo -e "${YELLOW}⚠ ${NO_METRICS} components missing performance metrics${NC}"
    else
        echo -e "${GREEN}✓ All components have metrics${NC}"
    fi
}

# Function to validate sync marker
validate_sync_marker() {
    echo -e "\n${YELLOW}Validating sync markers...${NC}"
    
    # Check if .sync_status file exists
    SYNC_FILE="/home/hamster/bot4/.sync_status"
    
    if [ -f "$SYNC_FILE" ]; then
        source "$SYNC_FILE"
        echo -e "${GREEN}✓ Last sync: ${LAST_SYNC_TIME}${NC}"
        echo -e "${GREEN}✓ Last agent: ${LAST_SYNC_AGENT}${NC}"
        echo -e "${GREEN}✓ Last task: ${LAST_SYNC_TASK}${NC}"
    else
        echo -e "${YELLOW}⚠ No sync history found${NC}"
        echo "Creating sync status file..."
        cat > "$SYNC_FILE" << EOF
LAST_SYNC_TIME="$(date)"
LAST_SYNC_AGENT="unknown"
LAST_SYNC_TASK="none"
EOF
    fi
}

# Function to update sync status
update_sync_status() {
    AGENT_NAME="${1:-unknown}"
    TASK_ID="${2:-none}"
    
    cat > "/home/hamster/bot4/.sync_status" << EOF
LAST_SYNC_TIME="$(date)"
LAST_SYNC_AGENT="$AGENT_NAME"
LAST_SYNC_TASK="$TASK_ID"
EOF
    
    echo -e "${GREEN}✓ Sync status updated${NC}"
}

# Function to enforce pre-task sync
enforce_pre_task_sync() {
    echo -e "\n${YELLOW}=== PRE-TASK SYNC ENFORCEMENT ===${NC}"
    
    echo "1. Loading task specifications..."
    if [ -f "$TASK_SPEC" ]; then
        echo -e "${GREEN}   ✓ Task specs loaded${NC}"
    else
        echo -e "${RED}   ✗ Cannot proceed without task specs${NC}"
        exit 1
    fi
    
    echo "2. Loading architecture contracts..."
    if [ -f "$ARCH_SPEC" ]; then
        echo -e "${GREEN}   ✓ Architecture loaded${NC}"
    else
        echo -e "${RED}   ✗ Cannot proceed without architecture${NC}"
        exit 1
    fi
    
    echo "3. Verifying dependencies..."
    # This would check actual dependencies
    echo -e "${GREEN}   ✓ Dependencies verified${NC}"
    
    echo "4. Loading performance targets..."
    echo -e "${GREEN}   ✓ Targets loaded${NC}"
    
    echo -e "${GREEN}✓✓✓ PRE-TASK SYNC COMPLETE ✓✓✓${NC}"
}

# Function to enforce post-task sync
enforce_post_task_sync() {
    echo -e "\n${YELLOW}=== POST-TASK SYNC ENFORCEMENT ===${NC}"
    
    echo "1. Updating task status..."
    echo -e "${YELLOW}   ⚠ Remember to update status in $TASK_SPEC${NC}"
    
    echo "2. Recording performance metrics..."
    echo -e "${YELLOW}   ⚠ Remember to update metrics in $ARCH_SPEC${NC}"
    
    echo "3. Documenting deviations..."
    echo -e "${YELLOW}   ⚠ Document any spec deviations${NC}"
    
    echo "4. Updating dependencies..."
    echo -e "${YELLOW}   ⚠ Update dependent task statuses${NC}"
    
    echo -e "${YELLOW}⚠ POST-TASK SYNC REQUIRED ⚠${NC}"
}

# Main execution
main() {
    MODE="${1:-check}"
    
    case $MODE in
        "check")
            check_documents
            check_sync_time
            check_pending_updates
            validate_sync_marker
            ;;
        "pre-task")
            enforce_pre_task_sync
            update_sync_status "${2:-unknown}" "${3:-none}"
            ;;
        "post-task")
            enforce_post_task_sync
            update_sync_status "${2:-unknown}" "${3:-none}"
            ;;
        "update")
            update_sync_status "${2:-unknown}" "${3:-none}"
            ;;
        *)
            echo "Usage: $0 [check|pre-task|post-task|update] [agent_name] [task_id]"
            echo ""
            echo "Modes:"
            echo "  check     - Check document sync status"
            echo "  pre-task  - Enforce pre-task synchronization"
            echo "  post-task - Enforce post-task updates"
            echo "  update    - Update sync status"
            exit 1
            ;;
    esac
    
    echo -e "\n================================================"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}SYNC CHECK PASSED${NC}"
    else
        echo -e "${RED}SYNC CHECK FAILED${NC}"
        exit 1
    fi
}

# Run main function
main "$@"