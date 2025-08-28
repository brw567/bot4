#!/bin/bash
# Quick deployment script for Bot4 MCP system

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Bot4 MCP Deployment${NC}"
echo "========================"

case "$1" in
    start)
        echo "Starting MCP services..."
        docker-compose up -d
        echo -e "${GREEN}✓ Services started${NC}"
        echo "View logs: docker-compose logs -f"
        ;;
    
    stop)
        echo "Stopping MCP services..."
        docker-compose down
        echo -e "${GREEN}✓ Services stopped${NC}"
        ;;
    
    build)
        echo "Building MCP services..."
        ./build-agents.sh --build
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    status)
        echo "MCP Service Status:"
        docker-compose ps
        echo ""
        echo "Health checks:"
        for service in coordinator architect-agent riskquant-agent mlengineer-agent exchangespec-agent; do
            container="bot4-${service//-agent/}"
            if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container.*healthy"; then
                echo -e "  ${GREEN}✓${NC} $service: healthy"
            elif docker ps --format "table {{.Names}}" | grep -q "$container"; then
                echo -e "  ${YELLOW}⚠${NC} $service: running (not healthy yet)"
            else
                echo -e "  ${RED}✗${NC} $service: not running"
            fi
        done
        ;;
    
    logs)
        shift
        docker-compose logs -f "$@"
        ;;
    
    test)
        echo "Testing MCP endpoints..."
        echo -n "  Coordinator: "
        curl -s http://localhost:8000/health && echo " OK" || echo " FAILED"
        echo -n "  Architect: "
        curl -s http://localhost:8080/health && echo " OK" || echo " FAILED"
        echo -n "  RiskQuant: "
        curl -s http://localhost:8081/health && echo " OK" || echo " FAILED"
        echo -n "  MLEngineer: "
        curl -s http://localhost:8082/health && echo " OK" || echo " FAILED"
        echo -n "  ExchangeSpec: "
        curl -s http://localhost:8083/health && echo " OK" || echo " FAILED"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|build|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start  - Start all MCP services"
        echo "  stop   - Stop all MCP services"
        echo "  build  - Build Docker images"
        echo "  status - Show service status"
        echo "  logs   - Show service logs (optionally specify service)"
        echo "  test   - Test MCP endpoints"
        exit 1
        ;;
esac