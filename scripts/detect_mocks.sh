#!/bin/bash
# Mock Implementation Detection Script
# CRITICAL: Run before ANY production deployment
# Owner: Alex | Team: Full Squad

set -euo pipefail

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "================================================"
echo "Mock Implementation Detection Script"
echo "Date: $(date)"
echo "================================================"
echo ""

# Track if any mocks found
MOCKS_FOUND=0
CRITICAL_MOCKS=0

# Check for mock warnings in code
echo "Checking for MOCK implementations..."
echo ""

# Critical mock patterns
echo -e "${RED}=== CRITICAL MOCK CHECKS ===${NC}"

# Check for order placement mocks
if grep -r "MOCK_BINANCE_" rust_core/ --include="*.rs" | grep -v test | grep -v "^Binary" > /dev/null 2>&1; then
    echo -e "${RED}✗ CRITICAL: Found mock order placement!${NC}"
    grep -r "MOCK_BINANCE_" rust_core/ --include="*.rs" | grep -v test | head -5
    CRITICAL_MOCKS=$((CRITICAL_MOCKS + 1))
fi

# Check for mock balance returns
if grep -r "USING MOCK BALANCES" rust_core/ --include="*.rs" | grep -v test > /dev/null 2>&1; then
    echo -e "${RED}✗ CRITICAL: Found mock balance retrieval!${NC}"
    grep -r "USING MOCK BALANCES" rust_core/ --include="*.rs" | grep -v test | head -5
    CRITICAL_MOCKS=$((CRITICAL_MOCKS + 1))
fi

# Check for mock order operations
if grep -r "USING MOCK ORDER" rust_core/ --include="*.rs" | grep -v test > /dev/null 2>&1; then
    echo -e "${RED}✗ CRITICAL: Found mock order operations!${NC}"
    grep -r "USING MOCK ORDER" rust_core/ --include="*.rs" | grep -v test | head -5
    CRITICAL_MOCKS=$((CRITICAL_MOCKS + 1))
fi

echo ""
echo -e "${YELLOW}=== HIGH PRIORITY MOCK CHECKS ===${NC}"

# Check for TODO phase 8 tasks
if grep -r "TODO: \[PHASE 8" rust_core/ --include="*.rs" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Found Phase 8 TODOs (Exchange Integration):${NC}"
    grep -r "TODO: \[PHASE 8" rust_core/ --include="*.rs" | wc -l | xargs echo "  Count:"
    MOCKS_FOUND=$((MOCKS_FOUND + 1))
fi

# Check for mock WebSocket
if grep -r "MOCK WebSocket" rust_core/ --include="*.rs" | grep -v test > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Found mock WebSocket implementation${NC}"
    MOCKS_FOUND=$((MOCKS_FOUND + 1))
fi

# Check for temporary implementations
if grep -r "TEMPORARY MOCK" rust_core/ --include="*.rs" | grep -v test > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Found temporary mock implementations:${NC}"
    grep -r "TEMPORARY MOCK" rust_core/ --include="*.rs" | grep -v test | wc -l | xargs echo "  Count:"
    MOCKS_FOUND=$((MOCKS_FOUND + 1))
fi

echo ""
echo "================================================"
echo "DETECTION SUMMARY"
echo "================================================"

if [ $CRITICAL_MOCKS -gt 0 ]; then
    echo -e "${RED}✗ CRITICAL MOCKS FOUND: $CRITICAL_MOCKS${NC}"
    echo -e "${RED}DO NOT DEPLOY TO PRODUCTION!${NC}"
    echo -e "${RED}The system will NOT execute real trades!${NC}"
    echo ""
    echo "Critical mocks that MUST be replaced:"
    echo "1. Order placement (p8-exchange-3)"
    echo "2. Balance retrieval (p8-exchange-5)"
    echo "3. Order cancellation (p8-exchange-4)"
    echo ""
    echo "See CRITICAL_MOCK_IMPLEMENTATIONS_TRACKER.md for details"
    exit 1
elif [ $MOCKS_FOUND -gt 0 ]; then
    echo -e "${YELLOW}⚠ Non-critical mocks found: $MOCKS_FOUND${NC}"
    echo "These should be replaced before production but are not blocking."
    echo ""
    echo "Run with -v flag for verbose output"
    exit 0
else
    echo -e "${GREEN}✓ No mock implementations detected${NC}"
    echo "System appears ready for production deployment"
    echo ""
    echo "Note: This script may not catch all mocks."
    echo "Always review CRITICAL_MOCK_IMPLEMENTATIONS_TRACKER.md"
    exit 0
fi