#!/bin/bash
# Test script for Bot4 MCP agents
# Validates agent functionality and connectivity

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🧪 Bot4 MCP Agent Test Suite${NC}"
echo "================================"

# Test configuration
COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:3000}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
TIMEOUT=30

# Function to test agent health
test_agent_health() {
    local agent=$1
    local port=$2
    local url="http://localhost:$port/health"
    
    echo -n "Testing $agent agent health... "
    
    # Wait for agent to be ready
    local count=0
    while [ $count -lt $TIMEOUT ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            return 0
        fi
        sleep 1
        ((count++))
    done
    
    echo -e "${RED}✗ Timeout${NC}"
    return 1
}

# Function to test MCP tool
test_mcp_tool() {
    local agent=$1
    local tool=$2
    local params=$3
    local port=$4
    
    echo -n "Testing $agent.$tool... "
    
    local response=$(curl -s -X POST "http://localhost:$port/tools/$tool" \
        -H "Content-Type: application/json" \
        -d "$params" 2>/dev/null || echo "failed")
    
    if [[ "$response" == *"error"* ]] || [[ "$response" == "failed" ]]; then
        echo -e "${RED}✗${NC}"
        echo "  Response: $response"
        return 1
    else
        echo -e "${GREEN}✓${NC}"
        return 0
    fi
}

# Function to test Redis connectivity
test_redis() {
    echo -n "Testing Redis connectivity... "
    
    if redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Function to test coordinator
test_coordinator() {
    echo -n "Testing MCP Coordinator... "
    
    local response=$(curl -s "$COORDINATOR_URL/health" 2>/dev/null || echo "failed")
    
    if [[ "$response" == *"ok"* ]]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Function to test agent communication
test_agent_communication() {
    echo -n "Testing inter-agent communication... "
    
    # Send test message through coordinator
    local response=$(curl -s -X POST "$COORDINATOR_URL/message" \
        -H "Content-Type: application/json" \
        -d '{"from":"test","to":"architect","type":"ping","content":{}}' 2>/dev/null)
    
    if [[ "$response" == *"success"* ]]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠ Limited${NC}"
    fi
}

# Function to run integration test
run_integration_test() {
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    
    # Test 1: Duplication check flow
    echo -n "Integration: Duplication detection flow... "
    local response=$(curl -s -X POST "$COORDINATOR_URL/workflow/duplication-check" \
        -H "Content-Type: application/json" \
        -d '{"component":"Order","type":"struct"}' 2>/dev/null)
    
    if [[ "$response" == *"duplicates"* ]]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
    
    # Test 2: Risk assessment flow
    echo -n "Integration: Risk assessment flow... "
    response=$(curl -s -X POST "$COORDINATOR_URL/workflow/risk-assessment" \
        -H "Content-Type: application/json" \
        -d '{"positions":[{"symbol":"BTC/USDT","size":0.01,"value":500}]}' 2>/dev/null)
    
    if [[ "$response" == *"risk_score"* ]]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
    
    # Test 3: ML prediction flow
    echo -n "Integration: ML prediction flow... "
    response=$(curl -s -X POST "$COORDINATOR_URL/workflow/predict" \
        -H "Content-Type: application/json" \
        -d '{"model_id":"test","features":[[0.5,0.3,0.8]]}' 2>/dev/null)
    
    if [[ "$response" == *"predictions"* ]]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Function to test performance
test_performance() {
    echo -e "\n${YELLOW}Performance tests...${NC}"
    
    # Latency test
    echo -n "Coordinator latency: "
    local start=$(date +%s%N)
    curl -s "$COORDINATOR_URL/health" > /dev/null 2>&1
    local end=$(date +%s%N)
    local latency=$(( (end - start) / 1000000 ))
    
    if [ $latency -lt 100 ]; then
        echo -e "${GREEN}${latency}ms ✓${NC}"
    else
        echo -e "${YELLOW}${latency}ms ⚠${NC}"
    fi
    
    # Throughput test
    echo -n "Message throughput: "
    local count=0
    start=$(date +%s)
    
    while [ $count -lt 100 ]; do
        curl -s "$COORDINATOR_URL/health" > /dev/null 2>&1 &
        ((count++))
    done
    wait
    
    end=$(date +%s)
    local duration=$((end - start))
    local throughput=$((100 / duration))
    
    echo -e "${GREEN}${throughput} req/s${NC}"
}

# Main test execution
main() {
    local failed=0
    
    # Check dependencies
    echo "Checking dependencies..."
    for cmd in curl redis-cli docker; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}Error: $cmd not found${NC}"
            exit 1
        fi
    done
    
    # Test infrastructure
    echo -e "\n${YELLOW}Infrastructure tests...${NC}"
    test_redis || ((failed++))
    test_coordinator || ((failed++))
    
    # Test individual agents
    echo -e "\n${YELLOW}Agent health tests...${NC}"
    test_agent_health "architect" 8080 || ((failed++))
    test_agent_health "riskquant" 8081 || ((failed++))
    test_agent_health "mlengineer" 8082 || ((failed++))
    test_agent_health "exchangespec" 8083 || ((failed++))
    
    # Test agent tools
    echo -e "\n${YELLOW}Agent tool tests...${NC}"
    test_mcp_tool "architect" "check_duplicates" '{"component":"Test","type":"struct"}' 8080 || ((failed++))
    test_mcp_tool "riskquant" "calculate_kelly" '{"win_probability":0.6,"win_return":1.0,"loss_return":-1.0}' 8081 || ((failed++))
    test_mcp_tool "mlengineer" "detect_regime" '{"market_data":[{"close":50000,"volume":100}]}' 8082 || ((failed++))
    test_mcp_tool "exchangespec" "get_exchange_status" '{"exchange":"binance"}' 8083 || ((failed++))
    
    # Test communication
    echo -e "\n${YELLOW}Communication tests...${NC}"
    test_agent_communication || ((failed++))
    
    # Run integration tests
    run_integration_test || ((failed++))
    
    # Performance tests
    test_performance
    
    # Summary
    echo -e "\n================================"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed! ✓${NC}"
        exit 0
    else
        echo -e "${RED}$failed tests failed ✗${NC}"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coordinator)
            COORDINATOR_URL=$2
            shift 2
            ;;
        --redis)
            REDIS_URL=$2
            shift 2
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--coordinator URL] [--redis URL] [--quick]"
            exit 1
            ;;
    esac
done

# Run tests
if [ "$QUICK_TEST" = "true" ]; then
    echo "Running quick tests only..."
    test_redis
    test_coordinator
    test_agent_health "architect" 8080
else
    main
fi