#!/bin/bash

# Bot4 Integration Verification Script
# Ensures all components are properly connected
# Date: August 24, 2025

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔗 Bot4 Integration Verification"
echo "================================"

QUIET_MODE=false
if [[ "$1" == "--quiet" ]]; then
    QUIET_MODE=true
fi

ERRORS=0
WARNINGS=0

# Function to check a connection
check_connection() {
    local name=$1
    local check_cmd=$2
    
    if ! $QUIET_MODE; then
        echo -n "Checking $name... "
    fi
    
    if eval "$check_cmd" >/dev/null 2>&1; then
        if ! $QUIET_MODE; then
            echo -e "${GREEN}✅${NC}"
        fi
        return 0
    else
        if ! $QUIET_MODE; then
            echo -e "${RED}❌${NC}"
        fi
        ((ERRORS++))
        return 1
    fi
}

# Function to check a warning condition
check_warning() {
    local name=$1
    local check_cmd=$2
    
    if ! $QUIET_MODE; then
        echo -n "Checking $name... "
    fi
    
    if eval "$check_cmd" >/dev/null 2>&1; then
        if ! $QUIET_MODE; then
            echo -e "${GREEN}✅${NC}"
        fi
        return 0
    else
        if ! $QUIET_MODE; then
            echo -e "${YELLOW}⚠️${NC}"
        fi
        ((WARNINGS++))
        return 1
    fi
}

# 1. Check Rust compilation
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}1. Code Compilation${NC}"
fi
check_connection "Rust workspace compilation" "cd /home/hamster/bot4/rust_core && cargo check --quiet"

# 2. Check Database Connections
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}2. Database Connections${NC}"
fi
check_connection "PostgreSQL connection" "PGPASSWORD=bot3pass psql -U bot3user -h localhost -d bot3trading -c 'SELECT 1'"
check_connection "Redis connection" "redis-cli ping"

# 3. Check Crate Dependencies
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}3. Crate Integration${NC}"
fi
check_connection "Risk crate integration" "cd /home/hamster/bot4/rust_core && cargo tree -p risk --quiet"
check_connection "Data crate integration" "cd /home/hamster/bot4/rust_core && cargo tree -p data --quiet"
check_warning "ML crate integration" "cd /home/hamster/bot4/rust_core && cargo tree -p ml --quiet"

# 4. Check Critical Files
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}4. Critical Files${NC}"
fi
check_connection "PROJECT_MANAGEMENT_MASTER.md exists" "test -f /home/hamster/bot4/PROJECT_MANAGEMENT_MASTER.md"
check_connection "ARCHITECTURE.md exists" "test -f /home/hamster/bot4/ARCHITECTURE.md"
check_connection "CLAUDE.md exists" "test -f /home/hamster/bot4/CLAUDE.md"

# 5. Check Data Flow Components
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}5. Data Pipeline Components${NC}"
fi
check_warning "Feature store tables" "PGPASSWORD=bot3pass psql -U bot3user -h localhost -d bot3trading -c 'SELECT 1 FROM information_schema.tables WHERE table_name = '\''feature_store'\'' '"
check_warning "TimescaleDB extension" "PGPASSWORD=bot3pass psql -U bot3user -h localhost -d bot3trading -c 'SELECT 1 FROM pg_extension WHERE extname = '\''timescaledb'\'' '"

# 6. Check Risk Integration
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}6. Risk Management Integration${NC}"
fi
check_connection "Risk engine tests" "cd /home/hamster/bot4/rust_core && cargo test -p risk --lib --quiet"
check_warning "Circuit breaker implementation" "grep -r 'CircuitBreaker' /home/hamster/bot4/rust_core/crates/risk/src/"

# 7. Check Layer Dependencies
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}7. Layer Architecture${NC}"
fi
LAYER_0_COMPLETE=$(grep "Layer 0.*Complete" /home/hamster/bot4/PROJECT_MANAGEMENT_MASTER.md | grep -o "[0-9]*%" | grep -o "[0-9]*")
if [ "$LAYER_0_COMPLETE" -lt 100 ] 2>/dev/null; then
    if ! $QUIET_MODE; then
        echo -e "Layer 0 (Safety): ${YELLOW}${LAYER_0_COMPLETE}% - BLOCKER${NC}"
    fi
    ((WARNINGS++))
else
    if ! $QUIET_MODE; then
        echo -e "Layer 0 (Safety): ${GREEN}Complete${NC}"
    fi
fi

# 8. Check Documentation Sync
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}8. Documentation Sync${NC}"
fi
check_warning "LLM_TASK_SPECIFICATIONS.md synced" "test -f /home/hamster/bot4/docs/LLM_TASK_SPECIFICATIONS.md"
check_warning "LLM_OPTIMIZED_ARCHITECTURE.md synced" "test -f /home/hamster/bot4/docs/LLM_OPTIMIZED_ARCHITECTURE.md"

# Summary
if ! $QUIET_MODE; then
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Integration Verification Complete"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✅ All integration points verified successfully!${NC}"
        exit 0
    elif [ $ERRORS -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Verification complete with $WARNINGS warnings${NC}"
        exit 0
    else
        echo -e "${RED}❌ Verification failed with $ERRORS errors and $WARNINGS warnings${NC}"
        echo "Please fix integration issues before proceeding!"
        exit 1
    fi
else
    # Quiet mode - just exit with appropriate code
    if [ $ERRORS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
fi