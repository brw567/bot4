#!/bin/bash
# Install Playwright for browser automation on Ubuntu 22.04

echo "🚀 Installing Playwright for LLM browser automation"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version

# Install pip if not present
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Installing pip3...${NC}"
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install Playwright
echo -e "${YELLOW}Installing Playwright...${NC}"
pip3 install --user playwright

# Add local pip bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}✓ Added ~/.local/bin to PATH${NC}"
fi

# Install Chromium browser for Playwright
echo -e "${YELLOW}Installing Chromium for Playwright...${NC}"
$HOME/.local/bin/playwright install chromium

# Install system dependencies for headless Chrome on Ubuntu
echo -e "${YELLOW}Installing system dependencies...${NC}"
$HOME/.local/bin/playwright install-deps chromium

echo -e "${GREEN}✅ Playwright installation complete!${NC}"

# Test installation
echo -e "${YELLOW}Testing Playwright installation...${NC}"
python3 -c "from playwright.sync_api import sync_playwright; print('✅ Playwright imported successfully')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Playwright is ready to use!${NC}"
else
    echo -e "${RED}❌ Playwright import failed. Please check installation.${NC}"
    exit 1
fi

echo ""
echo "📋 Next steps:"
echo "1. Run: python3 scripts/llm_browser_bridge.py setup chatgpt"
echo "2. Run: python3 scripts/llm_browser_bridge.py setup grok"
echo "3. Then: python3 scripts/llm_browser_bridge.py process all"