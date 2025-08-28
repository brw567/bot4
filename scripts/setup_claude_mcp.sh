#!/bin/bash
# Setup script for Claude CLI MCP integration with Bot4 agents
# This script configures Claude to use the Docker-based MCP servers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BOT4_ROOT="/home/hamster/bot4"
MCP_DIR="$BOT4_ROOT/mcp"

echo -e "${BLUE}🤖 Claude CLI MCP Integration Setup${NC}"
echo "======================================"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    local missing=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("Docker")
    else
        echo -e "${GREEN}✓ Docker $(docker --version | awk '{print $3}')${NC}"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing+=("Docker Compose")
    else
        echo -e "${GREEN}✓ Docker Compose $(docker-compose --version | awk '{print $3}')${NC}"
    fi
    
    # Check Claude CLI
    if ! command -v claude &> /dev/null; then
        echo -e "${YELLOW}⚠ Claude CLI not found in PATH${NC}"
        echo "  You may need to configure Claude Code manually"
    else
        echo -e "${GREEN}✓ Claude CLI found${NC}"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing prerequisites: ${missing[*]}${NC}"
        exit 1
    fi
}

# Function to build Docker images
build_images() {
    echo -e "\n${YELLOW}Building Docker images...${NC}"
    
    cd "$MCP_DIR"
    
    # Fix coordinator Cargo.toml
    if [ -f "coordinator/Cargo.toml" ]; then
        echo "Fixing coordinator Cargo.toml..."
        # Remove the benchmark reference that's causing issues
        sed -i '/\[\[bench\]\]/,/^$/d' coordinator/Cargo.toml 2>/dev/null || true
    fi
    
    # Build images
    if [ -x "./build-agents.sh" ]; then
        echo "Running build script..."
        ./build-agents.sh --build
    else
        echo -e "${RED}Build script not found or not executable${NC}"
        exit 1
    fi
}

# Function to start Docker services
start_services() {
    echo -e "\n${YELLOW}Starting Docker services...${NC}"
    
    cd "$MCP_DIR"
    
    # Stop any existing services
    docker-compose down 2>/dev/null || true
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps | grep -q "healthy"; then
            echo -e "${GREEN}✓ Services are healthy${NC}"
            break
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "\n${YELLOW}⚠ Some services may not be fully ready${NC}"
    fi
}

# Function to configure Claude CLI
configure_claude() {
    echo -e "\n${YELLOW}Configuring Claude CLI...${NC}"
    
    # Create Claude config directory if it doesn't exist
    CLAUDE_CONFIG_DIR="$HOME/.config/claude-code"
    mkdir -p "$CLAUDE_CONFIG_DIR"
    
    # Copy MCP configuration
    if [ -f "$BOT4_ROOT/claude-mcp-config.json" ]; then
        cp "$BOT4_ROOT/claude-mcp-config.json" "$CLAUDE_CONFIG_DIR/mcp.json"
        echo -e "${GREEN}✓ MCP configuration copied to Claude config${NC}"
    else
        echo -e "${RED}MCP configuration file not found${NC}"
        exit 1
    fi
    
    # Create a startup script for Claude
    cat > "$BOT4_ROOT/start-claude-with-mcp.sh" << 'EOF'
#!/bin/bash
# Start Claude with MCP servers

BOT4_ROOT="/home/hamster/bot4"
MCP_DIR="$BOT4_ROOT/mcp"

echo "Starting Bot4 MCP services..."
cd "$MCP_DIR"
docker-compose up -d

echo "Waiting for services..."
sleep 5

echo "Starting Claude CLI..."
claude "$@"
EOF
    
    chmod +x "$BOT4_ROOT/start-claude-with-mcp.sh"
    echo -e "${GREEN}✓ Created Claude startup script${NC}"
}

# Function to test MCP connectivity
test_connectivity() {
    echo -e "\n${YELLOW}Testing MCP connectivity...${NC}"
    
    # Test coordinator health
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✓ Coordinator is responding${NC}"
    else
        echo -e "${RED}✗ Coordinator not responding${NC}"
    fi
    
    # Test Redis
    if docker exec bot4-redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Redis is responding${NC}"
    else
        echo -e "${RED}✗ Redis not responding${NC}"
    fi
    
    # Test agent endpoints
    for port in 8080 8081 8082 8083; do
        agent_name=""
        case $port in
            8080) agent_name="Architect" ;;
            8081) agent_name="RiskQuant" ;;
            8082) agent_name="MLEngineer" ;;
            8083) agent_name="ExchangeSpec" ;;
        esac
        
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $agent_name agent is responding${NC}"
        else
            echo -e "${YELLOW}⚠ $agent_name agent not responding yet${NC}"
        fi
    done
}

# Function to create example usage
create_examples() {
    echo -e "\n${YELLOW}Creating example usage scripts...${NC}"
    
    cat > "$BOT4_ROOT/examples/test-mcp-agents.md" << 'EOF'
# Testing Bot4 MCP Agents

## 1. Architect Agent - Check for duplicates
```
claude --mcp bot4-architect
> Check for duplicate implementations of Order struct
```

## 2. RiskQuant Agent - Calculate position size
```
claude --mcp bot4-riskquant  
> Calculate Kelly criterion for win_prob=0.6, win_return=1.0, loss_return=-1.0
```

## 3. MLEngineer Agent - Feature engineering
```
claude --mcp bot4-mlengineer
> Extract features from the last 1000 candles for BTC/USDT
```

## 4. ExchangeSpec Agent - Check rate limits
```
claude --mcp bot4-exchangespec
> What are the current rate limits for Binance?
```

## Multi-Agent Collaboration
```
claude --mcp bot4-coordinator
> Orchestrate Task 1.6.5: Testing Infrastructure Consolidation
```
EOF
    
    echo -e "${GREEN}✓ Created example usage guide${NC}"
}

# Main setup process
main() {
    echo -e "\n${BLUE}Starting Claude MCP integration setup...${NC}"
    
    # Check prerequisites
    check_prerequisites
    
    # Build Docker images
    build_images
    
    # Start services
    start_services
    
    # Configure Claude CLI
    configure_claude
    
    # Test connectivity
    test_connectivity
    
    # Create examples
    mkdir -p "$BOT4_ROOT/examples"
    create_examples
    
    echo -e "\n======================================"
    echo -e "${GREEN}✅ Claude MCP Integration Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start Claude with MCP: ./start-claude-with-mcp.sh"
    echo "2. Or manually: cd mcp && docker-compose up -d"
    echo "3. Access coordinator API: http://localhost:8000"
    echo "4. View logs: docker-compose logs -f [service-name]"
    echo ""
    echo "Available MCP servers:"
    echo "  • bot4-coordinator (port 8000)"
    echo "  • bot4-architect (port 8080)"
    echo "  • bot4-riskquant (port 8081)"
    echo "  • bot4-mlengineer (port 8082)"
    echo "  • bot4-exchangespec (port 8083)"
    echo ""
    echo "See examples/test-mcp-agents.md for usage examples"
    echo ""
    echo -e "${BLUE}Ready for multi-agent collaboration!${NC}"
}

# Run main setup
main "$@"