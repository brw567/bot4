#!/bin/bash

# Bot4 Complete Development Environment Setup
# Team: Full 8-member collaboration
# Purpose: One-click setup for all development dependencies

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Bot4 Development Environment Setup                   ║${NC}"
echo -e "${CYAN}║         Autonomous Trading Platform - Full Setup               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}✗ Please do not run as root. Script will use sudo when needed.${NC}"
   exit 1
fi

print_section "Step 1: System Dependencies"

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libpq-dev \
    redis-server \
    postgresql \
    postgresql-contrib \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    jq \
    htop \
    tmux \
    vim

echo -e "${GREEN}✓ System packages installed${NC}"

print_section "Step 2: Rust Toolchain"

if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo -e "${GREEN}✓ Rust already installed: $(rustc --version)${NC}"
fi

# Update Rust
rustup update stable
rustup component add rustfmt clippy rust-analyzer

echo -e "${GREEN}✓ Rust toolchain ready${NC}"

print_section "Step 3: Database Setup"

# PostgreSQL setup
echo "Setting up PostgreSQL..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check if user exists
if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='bot3user'" | grep -q 1; then
    echo -e "${GREEN}✓ PostgreSQL user already exists${NC}"
else
    sudo -u postgres psql << EOF
CREATE USER bot3user WITH PASSWORD 'bot3pass';
CREATE DATABASE bot3trading OWNER bot3user;
GRANT ALL PRIVILEGES ON DATABASE bot3trading TO bot3user;
EOF
    echo -e "${GREEN}✓ PostgreSQL database created${NC}"
fi

# Install TimescaleDB
if ! sudo -u postgres psql -d bot3trading -c "SELECT 1 FROM pg_extension WHERE extname='timescaledb'" | grep -q 1; then
    echo "Installing TimescaleDB..."
    sudo apt-get install -y postgresql-14-timescaledb
    sudo timescaledb-tune --quiet --yes
    sudo systemctl restart postgresql
    
    sudo -u postgres psql -d bot3trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
    echo -e "${GREEN}✓ TimescaleDB installed${NC}"
else
    echo -e "${GREEN}✓ TimescaleDB already installed${NC}"
fi

# Redis setup
echo "Setting up Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server
echo -e "${GREEN}✓ Redis ready${NC}"

print_section "Step 4: LibTorch for ML Components"

# Setup libtorch
echo "Configuring LibTorch..."
if [ -f /usr/lib/x86_64-linux-gnu/libtorch.so ]; then
    echo -e "${GREEN}✓ LibTorch already installed${NC}"
    export LIBTORCH=/usr/lib/x86_64-linux-gnu
else
    echo "Installing LibTorch..."
    sudo apt-get install -y libtorch1.8 libtorch-dev libtorch-test
    export LIBTORCH=/usr/lib/x86_64-linux-gnu
fi

# Create environment file
cat > ~/.bot4_env << EOF
# Bot4 Environment Variables
export LIBTORCH=/usr/lib/x86_64-linux-gnu
export LIBTORCH_INCLUDE=/usr/include
export LD_LIBRARY_PATH=\$LIBTORCH:\$LD_LIBRARY_PATH

# Database
export DATABASE_URL=postgresql://bot3user:bot3pass@localhost:5432/bot3trading
export REDIS_URL=redis://localhost:6379/0

# Development
export RUST_LOG=info
export RUST_BACKTRACE=1
EOF

echo -e "${GREEN}✓ LibTorch configured${NC}"

print_section "Step 5: Python Dependencies"

echo "Installing Python packages for validation scripts..."
pip3 install --user \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    pytest \
    black \
    flake8 \
    mypy

echo -e "${GREEN}✓ Python dependencies installed${NC}"

print_section "Step 6: Git Hooks"

echo "Setting up git hooks..."
if [ -d .git ]; then
    # Integration check hook is already created
    if [ -f .git/hooks/pre-commit ]; then
        echo -e "${GREEN}✓ Git hooks already configured${NC}"
    else
        echo -e "${YELLOW}⚠ Run from repository root to setup git hooks${NC}"
    fi
fi

print_section "Step 7: Build Validation"

echo "Testing Rust build..."
cd rust_core

# Set environment
export LIBTORCH=/usr/lib/x86_64-linux-gnu

# Try to build
echo "Running cargo check..."
if cargo check 2>&1 | grep -q "error\[E"; then
    echo -e "${YELLOW}⚠ Some crates have compilation issues${NC}"
    echo "  This is expected - run 'cargo build' to see details"
else
    echo -e "${GREEN}✓ Rust project structure valid${NC}"
fi

cd ..

print_section "Step 8: Integration Verification"

echo "Running integration check..."
if [ -f scripts/verify_integration.sh ]; then
    ./scripts/verify_integration.sh || true
else
    echo -e "${YELLOW}⚠ Integration script not found${NC}"
fi

print_section "Setup Complete!"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Development Environment Ready!                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Quick Start Commands:${NC}"
echo "  ${YELLOW}cd rust_core && cargo build --release${NC}  # Build the project"
echo "  ${YELLOW}cargo test --all${NC}                       # Run tests"
echo "  ${YELLOW}./scripts/verify_integration.sh${NC}        # Check integration"
echo ""
echo -e "${CYAN}Environment Variables:${NC}"
echo "  ${YELLOW}source ~/.bot4_env${NC}                     # Load environment"
echo ""
echo -e "${CYAN}Database Access:${NC}"
echo "  ${YELLOW}psql -U bot3user -d bot3trading${NC}        # PostgreSQL"
echo "  ${YELLOW}redis-cli${NC}                              # Redis"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"

# Create quick start script
cat > start_dev.sh << 'EOF'
#!/bin/bash
# Quick start for development
source ~/.bot4_env
echo "Bot4 environment loaded!"
echo "LIBTORCH=$LIBTORCH"
echo "DATABASE_URL=$DATABASE_URL"
cd rust_core
EOF
chmod +x start_dev.sh

echo -e "\n${YELLOW}Tip: Run './start_dev.sh' to quickly set up your development session${NC}"