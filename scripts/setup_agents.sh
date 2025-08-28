#!/bin/bash
# Bot4 Multi-Agent System Setup Script
# Installs and configures all 8 MCP agents

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BOT4_ROOT="/home/hamster/bot4"
AGENTS_DIR="$BOT4_ROOT/agents"

echo -e "${BLUE}🤖 Bot4 Multi-Agent System Setup${NC}"
echo "Installing 8 specialized agents..."
echo "----------------------------------------"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    local missing=()
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        missing+=("Node.js")
    else
        echo -e "${GREEN}✓ Node.js $(node --version)${NC}"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing+=("Python 3")
    else
        echo -e "${GREEN}✓ Python $(python3 --version)${NC}"
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        missing+=("Redis")
    else
        echo -e "${GREEN}✓ Redis installed${NC}"
    fi
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        missing+=("Rust/Cargo")
    else
        echo -e "${GREEN}✓ Rust $(rustc --version)${NC}"
    fi
    
    # Check ripgrep
    if ! command -v rg &> /dev/null; then
        missing+=("ripgrep")
    else
        echo -e "${GREEN}✓ ripgrep installed${NC}"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing prerequisites: ${missing[*]}${NC}"
        echo "Please install missing components and try again"
        exit 1
    fi
}

# Function to setup TypeScript agents
setup_typescript_agent() {
    local agent_name=$1
    local agent_dir="$AGENTS_DIR/$agent_name"
    
    echo -e "\n${YELLOW}Setting up $agent_name agent...${NC}"
    
    if [ -d "$agent_dir" ]; then
        cd "$agent_dir"
        
        # Install dependencies
        if [ -f "package.json" ]; then
            echo "Installing dependencies..."
            npm install --silent
            
            # Build TypeScript
            if [ -f "tsconfig.json" ]; then
                echo "Building TypeScript..."
                npm run build
            fi
            
            echo -e "${GREEN}✓ $agent_name agent ready${NC}"
        else
            echo -e "${YELLOW}⚠ No package.json found for $agent_name${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Agent directory not found: $agent_dir${NC}"
    fi
}

# Function to setup Python agents
setup_python_agent() {
    local agent_name=$1
    local agent_dir="$AGENTS_DIR/$agent_name"
    
    echo -e "\n${YELLOW}Setting up $agent_name agent (Python)...${NC}"
    
    # Create agent directory if it doesn't exist
    mkdir -p "$agent_dir"
    
    # Create virtual environment
    if [ ! -d "$agent_dir/venv" ]; then
        python3 -m venv "$agent_dir/venv"
    fi
    
    # Install Python MCP SDK
    source "$agent_dir/venv/bin/activate"
    pip install -q mcp redis
    deactivate
    
    echo -e "${GREEN}✓ $agent_name agent environment ready${NC}"
}

# Function to start Redis if not running
start_redis() {
    echo -e "\n${YELLOW}Checking Redis...${NC}"
    
    if ! redis-cli ping &> /dev/null; then
        echo "Starting Redis server..."
        redis-server --daemonize yes
        sleep 2
    fi
    
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${RED}❌ Failed to start Redis${NC}"
        exit 1
    fi
}

# Function to initialize shared context
init_shared_context() {
    echo -e "\n${YELLOW}Initializing shared context...${NC}"
    
    # Ensure .mcp directory exists
    mkdir -p "$BOT4_ROOT/.mcp"
    
    # Check if shared context exists
    if [ -f "$BOT4_ROOT/.mcp/shared_context.json" ]; then
        echo -e "${GREEN}✓ Shared context exists${NC}"
    else
        echo -e "${RED}❌ Shared context not found${NC}"
        exit 1
    fi
    
    # Make load script executable
    chmod +x "$BOT4_ROOT/.mcp/load_context.sh"
}

# Function to create agent launcher
create_launcher() {
    local launcher_script="$BOT4_ROOT/scripts/launch_agents.sh"
    
    echo -e "\n${YELLOW}Creating agent launcher...${NC}"
    
    cat > "$launcher_script" << 'EOF'
#!/bin/bash
# Launch all Bot4 agents

BOT4_ROOT="/home/hamster/bot4"
AGENTS_DIR="$BOT4_ROOT/agents"
LOGS_DIR="$BOT4_ROOT/.mcp/logs"

mkdir -p "$LOGS_DIR"

echo "Starting Bot4 Multi-Agent System..."

# Start TypeScript agents
for agent in architect exchange_spec infra_engineer quality_gate integration_validator; do
    if [ -d "$AGENTS_DIR/$agent/dist" ]; then
        echo "Starting $agent..."
        nohup node "$AGENTS_DIR/$agent/dist/index.js" > "$LOGS_DIR/$agent.log" 2>&1 &
        echo "$!" > "$LOGS_DIR/$agent.pid"
    fi
done

# Start Python agents
for agent in risk_quant ml_engineer compliance_auditor; do
    if [ -d "$AGENTS_DIR/$agent" ]; then
        echo "Starting $agent..."
        source "$AGENTS_DIR/$agent/venv/bin/activate"
        nohup python3 -m "bot4.agents.$agent" > "$LOGS_DIR/$agent.log" 2>&1 &
        echo "$!" > "$LOGS_DIR/$agent.pid"
        deactivate
    fi
done

echo "All agents started. Check logs in $LOGS_DIR"
EOF
    
    chmod +x "$launcher_script"
    echo -e "${GREEN}✓ Launcher created: $launcher_script${NC}"
}

# Function to create stop script
create_stop_script() {
    local stop_script="$BOT4_ROOT/scripts/stop_agents.sh"
    
    cat > "$stop_script" << 'EOF'
#!/bin/bash
# Stop all Bot4 agents

LOGS_DIR="/home/hamster/bot4/.mcp/logs"

echo "Stopping Bot4 agents..."

for pidfile in "$LOGS_DIR"/*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        agent=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $agent (PID: $pid)"
            kill "$pid"
        fi
        rm "$pidfile"
    fi
done

echo "All agents stopped"
EOF
    
    chmod +x "$stop_script"
    echo -e "${GREEN}✓ Stop script created: $stop_script${NC}"
}

# Main setup
main() {
    echo -e "\n${BLUE}Starting multi-agent system setup...${NC}"
    
    # Check prerequisites
    check_prerequisites
    
    # Start Redis
    start_redis
    
    # Initialize shared context
    init_shared_context
    
    # Setup TypeScript agents
    setup_typescript_agent "architect"
    # setup_typescript_agent "exchange_spec"
    # setup_typescript_agent "infra_engineer"
    # setup_typescript_agent "quality_gate"
    # setup_typescript_agent "integration_validator"
    
    # Setup Python agents
    setup_python_agent "risk_quant"
    setup_python_agent "ml_engineer"
    setup_python_agent "compliance_auditor"
    
    # Create launcher and stop scripts
    create_launcher
    create_stop_script
    
    echo -e "\n----------------------------------------"
    echo -e "${GREEN}✅ Multi-Agent System Setup Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start agents: ./scripts/launch_agents.sh"
    echo "2. Load context: source .mcp/load_context.sh"
    echo "3. Check status: cat .mcp/shared_context.json | jq .agent_status"
    echo "4. Stop agents: ./scripts/stop_agents.sh"
    echo ""
    echo "Validation scripts:"
    echo "• Check duplicates: ./scripts/check_duplicates.sh <component>"
    echo "• Check layers: ./scripts/check_layer_violations.sh"
    echo ""
    echo -e "${BLUE}Ready for collaborative development!${NC}"
}

# Run main
main