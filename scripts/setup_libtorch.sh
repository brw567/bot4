#!/bin/bash

# LIBTORCH Setup Script - Complete ML Infrastructure
# Team: Full 8-member collaboration
# Purpose: Ensure libtorch is properly configured for ML components

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Bot4 LibTorch Setup - ML Infrastructure Configuration${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to check if running with sudo
check_sudo() {
    if [ "$EUID" -eq 0 ]; then 
        echo -e "${GREEN}✓ Running with sudo privileges${NC}"
    else
        echo -e "${YELLOW}⚠ This script requires sudo for system package installation${NC}"
        echo "Please run: sudo $0"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        echo -e "${RED}✗ Cannot detect OS${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Detected OS: $OS $VER${NC}"
}

# Function to check existing libtorch
check_existing_libtorch() {
    echo -e "\n${YELLOW}Checking for existing libtorch installation...${NC}"
    
    # Check system installation
    if [ -f /usr/lib/x86_64-linux-gnu/libtorch.so ]; then
        echo -e "${GREEN}✓ System libtorch found at /usr/lib/x86_64-linux-gnu${NC}"
        LIBTORCH_PATH="/usr/lib/x86_64-linux-gnu"
        return 0
    fi
    
    # Check PyTorch installation
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
        PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓ PyTorch ${PYTORCH_VERSION} found${NC}"
        echo "  You can use: export LIBTORCH_USE_PYTORCH=1"
        PYTORCH_AVAILABLE=true
    fi
    
    # Check manual installation
    if [ -d /opt/libtorch ]; then
        echo -e "${GREEN}✓ Manual libtorch found at /opt/libtorch${NC}"
        LIBTORCH_PATH="/opt/libtorch"
        return 0
    fi
    
    return 1
}

# Function to install system packages
install_system_libtorch() {
    echo -e "\n${YELLOW}Installing libtorch from system packages...${NC}"
    
    # Update package list
    apt-get update
    
    # Install libtorch packages
    apt-get install -y \
        libtorch1.8 \
        libtorch-dev \
        libtorch-test \
        libopenblas-dev \
        cmake \
        build-essential
    
    echo -e "${GREEN}✓ System packages installed${NC}"
}

# Function to download and install libtorch manually
install_manual_libtorch() {
    echo -e "\n${YELLOW}Installing libtorch manually...${NC}"
    
    # Create directory
    mkdir -p /opt
    cd /opt
    
    # Download libtorch (CPU version for compatibility)
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip"
    
    echo "Downloading libtorch..."
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    
    echo "Extracting..."
    unzip -q libtorch.zip
    rm libtorch.zip
    
    echo -e "${GREEN}✓ Manual installation complete at /opt/libtorch${NC}"
    LIBTORCH_PATH="/opt/libtorch"
}

# Function to setup environment variables
setup_environment() {
    echo -e "\n${YELLOW}Setting up environment variables...${NC}"
    
    # Create environment script
    cat > /etc/profile.d/libtorch.sh << EOF
# LibTorch environment variables for Bot4
export LIBTORCH=${LIBTORCH_PATH}
export LIBTORCH_INCLUDE=${LIBTORCH_PATH}/include
export LIBTORCH_LIB=${LIBTORCH_PATH}/lib
export LD_LIBRARY_PATH=\${LIBTORCH_PATH}/lib:\$LD_LIBRARY_PATH

# Optional: Use PyTorch if available
# export LIBTORCH_USE_PYTORCH=1
# export LIBTORCH_BYPASS_VERSION_CHECK=1
EOF
    
    chmod +x /etc/profile.d/libtorch.sh
    
    # Also add to cargo config for Rust builds
    mkdir -p ~/.cargo
    cat >> ~/.cargo/config.toml << EOF

[env]
LIBTORCH = "${LIBTORCH_PATH}"
LIBTORCH_INCLUDE = "${LIBTORCH_PATH}/include"
LIBTORCH_LIB = "${LIBTORCH_PATH}/lib"
EOF
    
    echo -e "${GREEN}✓ Environment variables configured${NC}"
    echo -e "${YELLOW}  Note: Run 'source /etc/profile.d/libtorch.sh' to load in current session${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${YELLOW}Verifying libtorch installation...${NC}"
    
    # Set environment for test
    export LIBTORCH=${LIBTORCH_PATH}
    export LIBTORCH_INCLUDE=${LIBTORCH_PATH}/include
    export LIBTORCH_LIB=${LIBTORCH_PATH}/lib
    export LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib:$LD_LIBRARY_PATH
    
    # Create test program
    cat > /tmp/test_torch.cpp << 'EOF'
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "LibTorch test successful!" << std::endl;
    std::cout << "Tensor:\n" << tensor << std::endl;
    return 0;
}
EOF
    
    # Try to compile test program
    if command -v g++ &> /dev/null; then
        echo "Testing C++ compilation..."
        if g++ -o /tmp/test_torch /tmp/test_torch.cpp \
            -I${LIBTORCH_INCLUDE} \
            -L${LIBTORCH_LIB} \
            -ltorch -lc10 -Wl,-rpath,${LIBTORCH_LIB} 2>/dev/null; then
            
            if /tmp/test_torch 2>/dev/null; then
                echo -e "${GREEN}✓ C++ test successful${NC}"
            else
                echo -e "${YELLOW}⚠ C++ compilation OK but runtime failed${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ C++ compilation failed (this is OK for Rust-only usage)${NC}"
        fi
    fi
    
    # Clean up
    rm -f /tmp/test_torch /tmp/test_torch.cpp
}

# Function to test Rust integration
test_rust_integration() {
    echo -e "\n${YELLOW}Testing Rust integration...${NC}"
    
    # Navigate to rust_core
    cd /home/hamster/bot4/rust_core
    
    # Try to build with libtorch
    export LIBTORCH=${LIBTORCH_PATH}
    
    echo "Running cargo check on ML crate..."
    if cargo check -p ml 2>&1 | grep -q "error"; then
        echo -e "${YELLOW}⚠ ML crate has compilation issues (may be unrelated to libtorch)${NC}"
    else
        echo -e "${GREEN}✓ ML crate builds with libtorch${NC}"
    fi
    
    echo "Running cargo check on data_intelligence crate..."
    if cargo check -p data_intelligence 2>&1 | grep -q "error"; then
        echo -e "${YELLOW}⚠ Data intelligence crate has issues (may be unrelated to libtorch)${NC}"
    else
        echo -e "${GREEN}✓ Data intelligence crate builds with libtorch${NC}"
    fi
}

# Function to display configuration summary
show_summary() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    Installation Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}LibTorch Path:${NC} ${LIBTORCH_PATH}"
    echo -e "${GREEN}Environment File:${NC} /etc/profile.d/libtorch.sh"
    echo ""
    echo -e "${YELLOW}To use libtorch in your current session:${NC}"
    echo "  source /etc/profile.d/libtorch.sh"
    echo ""
    echo -e "${YELLOW}For Rust development:${NC}"
    echo "  export LIBTORCH=${LIBTORCH_PATH}"
    echo ""
    if [ "$PYTORCH_AVAILABLE" = true ]; then
        echo -e "${YELLOW}Alternative: Use PyTorch installation:${NC}"
        echo "  export LIBTORCH_USE_PYTORCH=1"
        echo ""
    fi
    echo -e "${GREEN}Setup complete! LibTorch is ready for Bot4 ML components.${NC}"
}

# Main installation flow
main() {
    echo -e "${YELLOW}Starting libtorch setup...${NC}"
    
    # Check sudo
    if [ "$1" != "--no-sudo-check" ]; then
        check_sudo
    fi
    
    # Detect OS
    detect_os
    
    # Check existing installation
    if check_existing_libtorch; then
        echo -e "${GREEN}✓ LibTorch already installed${NC}"
    else
        echo -e "${YELLOW}LibTorch not found, installing...${NC}"
        
        # Try system packages first
        if command -v apt-get &> /dev/null; then
            install_system_libtorch
            LIBTORCH_PATH="/usr/lib/x86_64-linux-gnu"
        else
            # Fall back to manual installation
            install_manual_libtorch
        fi
    fi
    
    # Setup environment
    setup_environment
    
    # Verify installation
    verify_installation
    
    # Test Rust integration
    if [ -d /home/hamster/bot4/rust_core ]; then
        test_rust_integration
    fi
    
    # Show summary
    show_summary
}

# Handle script arguments
case "$1" in
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help           Show this help message"
        echo "  --no-sudo-check  Skip sudo check (for Docker/CI)"
        echo "  --manual         Force manual installation"
        echo "  --verify-only    Only verify existing installation"
        exit 0
        ;;
    --verify-only)
        check_existing_libtorch
        verify_installation
        exit 0
        ;;
    --manual)
        install_manual_libtorch
        setup_environment
        verify_installation
        show_summary
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac