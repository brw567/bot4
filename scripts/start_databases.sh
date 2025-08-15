#!/bin/bash
# Bot4 Database & Monitoring Stack Startup Script
# Task 0.1.2 & 0.1.3: Setup databases and monitoring
# V5 Compliant - 100% Implementation

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "   BOT4 V5 - DATABASE & MONITORING SETUP"
echo "=========================================="

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    echo -n "Checking $service... "
    
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓ RUNNING${NC}"
        return 0
    else
        echo -e "${RED}✗ NOT RUNNING${NC}"
        return 1
    fi
}

# Start all services
echo -e "\n${YELLOW}Starting Docker services...${NC}"
docker-compose -f docker-compose-v5.yml up -d

# Wait for services to initialize
echo -e "\n${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check all services
echo -e "\n${YELLOW}=== SERVICE STATUS ===${NC}"
FAILED=0

check_service "PostgreSQL" 5432 || ((FAILED++))
check_service "TimescaleDB" 5433 || ((FAILED++))
check_service "Redis" 6379 || ((FAILED++))
check_service "Prometheus" 9090 || ((FAILED++))
check_service "Grafana" 3001 || ((FAILED++))
check_service "Loki" 3100 || ((FAILED++))
check_service "Jaeger" 16686 || ((FAILED++))

# Test database connections
echo -e "\n${YELLOW}=== DATABASE CONNECTIVITY ===${NC}"

# Test PostgreSQL
echo -n "Testing PostgreSQL connection... "
if PGPASSWORD=bot4pass_secure_2025 psql -h localhost -U bot4user -d bot4_trading -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ CONNECTED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test Redis
echo -n "Testing Redis connection... "
if redis-cli -a bot4redis_secure_2025 ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ CONNECTED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Report
echo -e "\n=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ ALL SERVICES RUNNING ✓✓✓${NC}"
    echo -e "${GREEN}Task 0.1.2 & 0.1.3 COMPLETE${NC}"
    echo ""
    echo "Access points:"
    echo "  PostgreSQL:  localhost:5432"
    echo "  TimescaleDB: localhost:5433"
    echo "  Redis:       localhost:6379"
    echo "  Prometheus:  http://localhost:9090"
    echo "  Grafana:     http://localhost:3001 (admin/bot4grafana_secure_2025)"
    echo "  Jaeger:      http://localhost:16686"
    exit 0
else
    echo -e "${RED}✗✗✗ $FAILED SERVICES FAILED ✗✗✗${NC}"
    echo -e "${YELLOW}Check docker-compose logs for details${NC}"
    exit 1
fi