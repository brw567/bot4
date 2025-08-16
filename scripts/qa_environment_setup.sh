#!/bin/bash
# Bot4 QA Environment Setup Script
# Purpose: Automated setup for external QA team testing
# Requirements: Ubuntu 20.04+ or similar Linux distribution

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo "   Bot4 Trading Platform - QA Environment Setup"
echo "================================================"

# Function to check command existence
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 installed${NC}"
        return 0
    fi
}

# Function to install if missing
install_if_missing() {
    local cmd=$1
    local install_cmd=$2
    
    if ! check_command $cmd; then
        echo -e "${YELLOW}Installing $cmd...${NC}"
        eval $install_cmd
    fi
}

# 1. System Update
echo -e "\n${BLUE}Step 1: System Update${NC}"
sudo apt-get update -qq

# 2. Install Core Dependencies
echo -e "\n${BLUE}Step 2: Installing Core Dependencies${NC}"

# Git
install_if_missing "git" "sudo apt-get install -y git"

# Curl
install_if_missing "curl" "sudo apt-get install -y curl"

# Build essentials
echo "Installing build tools..."
sudo apt-get install -y build-essential pkg-config libssl-dev

# 3. Install Rust
echo -e "\n${BLUE}Step 3: Rust Installation${NC}"
if ! check_command "rustc"; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Verify Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo "Rust version: $RUST_VERSION"

# Install Rust tools
echo "Installing Rust development tools..."
rustup component add rustfmt clippy rust-analyzer
cargo install cargo-tarpaulin cargo-audit cargo-watch

# 4. Install Docker & Docker Compose
echo -e "\n${BLUE}Step 4: Docker Installation${NC}"
if ! check_command "docker"; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

if ! check_command "docker-compose"; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 5. Install Database Clients
echo -e "\n${BLUE}Step 5: Database Client Tools${NC}"
sudo apt-get install -y postgresql-client redis-tools

# 6. Install Python for validation scripts
echo -e "\n${BLUE}Step 6: Python for Validation Scripts${NC}"
install_if_missing "python3" "sudo apt-get install -y python3 python3-pip"
pip3 install --user pytest pylint black

# 7. Install GitHub CLI
echo -e "\n${BLUE}Step 7: GitHub CLI${NC}"
if ! check_command "gh"; then
    echo "Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh
fi

# 8. Clone Repository (if not already cloned)
echo -e "\n${BLUE}Step 8: Repository Setup${NC}"
if [ ! -d "bot4" ]; then
    echo "Cloning Bot4 repository..."
    git clone git@github.com:brw567/bot4.git
    cd bot4
else
    echo "Repository already exists"
    cd bot4
    git fetch origin
fi

# 9. Install Git Hooks
echo -e "\n${BLUE}Step 9: Git Hooks Installation${NC}"
if [ -d ".git/hooks" ]; then
    echo "Installing quality enforcement hooks..."
    cp .git-hooks/* .git/hooks/ 2>/dev/null || echo "No git hooks to copy"
    chmod +x .git/hooks/* 2>/dev/null || true
fi

# 10. Build Rust Project
echo -e "\n${BLUE}Step 10: Building Rust Project${NC}"
cd rust_core
cargo build --all
cargo test --all

# 11. Start Docker Services
echo -e "\n${BLUE}Step 11: Starting Docker Services${NC}"
cd ..
if [ -f "docker-compose-v5.yml" ]; then
    echo "Starting database and monitoring services..."
    docker-compose -f docker-compose-v5.yml up -d
    
    echo "Waiting for services to start..."
    sleep 10
    
    # Verify services
    docker-compose -f docker-compose-v5.yml ps
fi

# 12. Run Validation Scripts
echo -e "\n${BLUE}Step 12: Running Validation Checks${NC}"
echo "Checking for fake implementations..."
python3 scripts/validate_no_fakes.py || true
python3 scripts/validate_no_fakes_rust.py || true

echo "Running verification script..."
./scripts/verify_completion.sh || true

# 13. Environment Variables Setup
echo -e "\n${BLUE}Step 13: Environment Configuration${NC}"
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cat > .env.example << 'EOF'
# Bot4 Trading Platform - Environment Variables
# Copy to .env and update with your values

# Database Configuration
DATABASE_URL=postgresql://bot4user:bot4pass_secure_2025@localhost:5432/bot4_trading
TIMESCALE_URL=postgresql://bot4user:bot4pass_secure_2025@localhost:5433/bot4_timeseries
REDIS_URL=redis://:bot4redis_secure_2025@localhost:6379/0

# Exchange Configuration (Testnet)
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_SECRET=your_testnet_secret_here

# Risk Limits
MAX_POSITION_SIZE=0.02
MAX_LEVERAGE=3
REQUIRE_STOP_LOSS=true
MAX_DRAWDOWN=0.15

# Performance Targets
TARGET_LATENCY_NS=50
MIN_THROUGHPUT_OPS=10000

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3001
GRAFANA_USER=bot4admin
GRAFANA_PASSWORD=bot4grafana_secure_2025
EOF
    echo -e "${YELLOW}Please copy .env.example to .env and update with your values${NC}"
fi

# 14. Final System Check
echo -e "\n${BLUE}Step 14: System Verification${NC}"
echo "================================================"
echo "Checking all components..."

PASSED=0
FAILED=0

# Check Rust
check_command "rustc" && ((PASSED++)) || ((FAILED++))
check_command "cargo" && ((PASSED++)) || ((FAILED++))

# Check Docker
check_command "docker" && ((PASSED++)) || ((FAILED++))
check_command "docker-compose" && ((PASSED++)) || ((FAILED++))

# Check Database clients
check_command "psql" && ((PASSED++)) || ((FAILED++))
check_command "redis-cli" && ((PASSED++)) || ((FAILED++))

# Check Python
check_command "python3" && ((PASSED++)) || ((FAILED++))

# Check GitHub CLI
check_command "gh" && ((PASSED++)) || ((FAILED++))

# Report
echo -e "\n================================================"
echo "           QA ENVIRONMENT SETUP COMPLETE"
echo "================================================"
echo -e "Components Passed: ${GREEN}$PASSED${NC}"
echo -e "Components Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ Environment ready for QA testing!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review code in your preferred IDE"
    echo "2. Check out PR branches: git checkout <branch-name>"
    echo "3. Run tests: cargo test --all"
    echo "4. Validate no fakes: python3 scripts/validate_no_fakes_rust.py"
    echo "5. Access monitoring:"
    echo "   - Prometheus: http://localhost:9090"
    echo "   - Grafana: http://localhost:3001"
    echo ""
    echo "QA Validation Focus:"
    echo "- NO fake implementations (todo!(), unimplemented!())"
    echo "- NO placeholders or mock data"
    echo "- NO hardcoded values"
    echo "- 100% real functionality"
    echo "- 100% test coverage"
    exit 0
else
    echo -e "\n${RED}⚠️  Some components failed to install${NC}"
    echo "Please check the errors above and fix manually"
    exit 1
fi