#!/bin/bash
# Bot4 MCP Deployment Script
# Orchestrates the deployment of all MCP agents and infrastructure

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Bot4 MCP Deployment System         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Compose file: $COMPOSE_FILE"
echo "Log level: $LOG_LEVEL"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    local missing=0
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker not installed${NC}"
        ((missing++))
    else
        echo -e "${GREEN}✓ Docker $(docker --version | cut -d' ' -f3)${NC}"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}✗ Docker Compose not installed${NC}"
        ((missing++))
    else
        echo -e "${GREEN}✓ Docker Compose $(docker-compose --version | cut -d' ' -f4)${NC}"
    fi
    
    # Check Redis CLI (optional but useful)
    if command -v redis-cli &> /dev/null; then
        echo -e "${GREEN}✓ Redis CLI available${NC}"
    else
        echo -e "${YELLOW}⚠ Redis CLI not found (optional)${NC}"
    fi
    
    if [ $missing -gt 0 ]; then
        echo -e "${RED}Missing prerequisites. Please install required tools.${NC}"
        exit 1
    fi
    
    echo ""
}

# Function to build images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    if [ -f "./build-agents.sh" ]; then
        ./build-agents.sh
    else
        echo -e "${RED}Build script not found${NC}"
        exit 1
    fi
    
    echo ""
}

# Function to start infrastructure
start_infrastructure() {
    echo -e "${YELLOW}Starting infrastructure services...${NC}"
    
    # Start Redis and PostgreSQL first
    docker-compose -f "$COMPOSE_FILE" up -d redis postgres
    
    # Wait for Redis
    echo -n "Waiting for Redis... "
    local count=0
    while [ $count -lt 30 ]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 1
        ((count++))
    done
    
    if [ $count -eq 30 ]; then
        echo -e "${RED}✗ Timeout${NC}"
        return 1
    fi
    
    # Wait for PostgreSQL
    echo -n "Waiting for PostgreSQL... "
    count=0
    while [ $count -lt 30 ]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U bot3user > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 1
        ((count++))
    done
    
    if [ $count -eq 30 ]; then
        echo -e "${RED}✗ Timeout${NC}"
        return 1
    fi
    
    echo ""
}

# Function to start coordinator
start_coordinator() {
    echo -e "${YELLOW}Starting MCP Coordinator...${NC}"
    
    docker-compose -f "$COMPOSE_FILE" up -d mcp-coordinator
    
    # Wait for coordinator to be ready
    echo -n "Waiting for Coordinator... "
    local count=0
    while [ $count -lt 30 ]; do
        if curl -s "http://localhost:3000/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 1
        ((count++))
    done
    
    if [ $count -eq 30 ]; then
        echo -e "${RED}✗ Timeout${NC}"
        return 1
    fi
    
    echo ""
}

# Function to start agents
start_agents() {
    echo -e "${YELLOW}Starting MCP Agents...${NC}"
    
    local agents=(
        "architect-agent"
        "riskquant-agent"
        "mlengineer-agent"
        "exchangespec-agent"
    )
    
    for agent in "${agents[@]}"; do
        echo -n "Starting $agent... "
        docker-compose -f "$COMPOSE_FILE" up -d "$agent"
        echo -e "${GREEN}✓${NC}"
    done
    
    # Wait for all agents to be healthy
    echo -n "Waiting for agents to be healthy... "
    sleep 5  # Give agents time to initialize
    
    local all_healthy=true
    for agent in "${agents[@]}"; do
        local health=$(docker inspect --format='{{.State.Health.Status}}' "$(docker-compose -f "$COMPOSE_FILE" ps -q "$agent")" 2>/dev/null || echo "unknown")
        if [ "$health" != "healthy" ]; then
            all_healthy=false
            echo -e "${YELLOW}⚠ $agent not healthy yet${NC}"
        fi
    done
    
    if $all_healthy; then
        echo -e "${GREEN}✓ All agents healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Some agents still initializing${NC}"
    fi
    
    echo ""
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running system tests...${NC}"
    
    if [ -f "./test-agents.sh" ]; then
        ./test-agents.sh --quick
    else
        echo -e "${YELLOW}Test script not found, skipping tests${NC}"
    fi
    
    echo ""
}

# Function to show status
show_status() {
    echo -e "${YELLOW}System Status:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo -e "${YELLOW}Agent Endpoints:${NC}"
    echo "  Coordinator:  http://localhost:3000"
    echo "  Architect:    http://localhost:8080"
    echo "  RiskQuant:    http://localhost:8081"
    echo "  MLEngineer:   http://localhost:8082"
    echo "  ExchangeSpec: http://localhost:8083"
    echo ""
    echo -e "${YELLOW}Infrastructure:${NC}"
    echo "  Redis:        redis://localhost:6379"
    echo "  PostgreSQL:   postgresql://bot3user:bot3pass@localhost:5432/bot3trading"
    echo ""
}

# Function to tail logs
tail_logs() {
    echo -e "${YELLOW}Tailing logs (Ctrl+C to stop)...${NC}"
    docker-compose -f "$COMPOSE_FILE" logs -f --tail=50
}

# Function to stop all services
stop_all() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main deployment function
deploy() {
    check_prerequisites
    
    # Build images if requested
    if [ "$BUILD" = "true" ]; then
        build_images
    fi
    
    # Start services
    start_infrastructure
    start_coordinator
    start_agents
    
    # Run tests if requested
    if [ "$TEST" = "true" ]; then
        run_tests
    fi
    
    # Show status
    show_status
    
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Deployment Complete! ✓             ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    
    # Tail logs if requested
    if [ "$FOLLOW_LOGS" = "true" ]; then
        tail_logs
    fi
}

# Parse command line arguments
COMMAND="${1:-deploy}"

case "$COMMAND" in
    deploy)
        BUILD=false
        TEST=false
        FOLLOW_LOGS=false
        
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                --build)
                    BUILD=true
                    shift
                    ;;
                --test)
                    TEST=true
                    shift
                    ;;
                --follow)
                    FOLLOW_LOGS=true
                    shift
                    ;;
                --env)
                    ENVIRONMENT=$2
                    shift 2
                    ;;
                *)
                    echo "Unknown option: $1"
                    exit 1
                    ;;
            esac
        done
        
        deploy
        ;;
    
    start)
        start_infrastructure
        start_coordinator
        start_agents
        show_status
        ;;
    
    stop)
        stop_all
        ;;
    
    restart)
        stop_all
        sleep 2
        deploy
        ;;
    
    status)
        show_status
        ;;
    
    logs)
        tail_logs
        ;;
    
    test)
        run_tests
        ;;
    
    clean)
        cleanup
        ;;
    
    build)
        build_images
        ;;
    
    *)
        echo "Bot4 MCP Deployment Script"
        echo ""
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  deploy    Deploy the complete system (default)"
        echo "  start     Start all services"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  status    Show system status"
        echo "  logs      Tail service logs"
        echo "  test      Run system tests"
        echo "  clean     Clean up everything"
        echo "  build     Build Docker images"
        echo ""
        echo "Deploy Options:"
        echo "  --build   Build images before deploying"
        echo "  --test    Run tests after deployment"
        echo "  --follow  Follow logs after deployment"
        echo "  --env     Set environment (development/production)"
        echo ""
        echo "Examples:"
        echo "  $0 deploy --build --test"
        echo "  $0 restart"
        echo "  $0 logs"
        exit 0
        ;;
esac