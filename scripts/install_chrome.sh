#!/bin/bash
# Install Google Chrome and playwright-stealth for Ubuntu 22.04

echo "🌐 Installing Google Chrome and stealth packages"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Install Google Chrome
echo -e "${YELLOW}Installing Google Chrome...${NC}"

# Add Google Chrome repository
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

# Update and install Chrome
sudo apt-get update
sudo apt-get install -y google-chrome-stable

echo -e "${GREEN}✓ Google Chrome installed${NC}"

# Install additional Python packages for stealth
echo -e "${YELLOW}Installing stealth packages...${NC}"
pip3 install --user playwright-stealth pyppeteer-stealth undetected-chromedriver

# Also install Selenium as backup option
pip3 install --user selenium webdriver-manager

echo -e "${GREEN}✓ Stealth packages installed${NC}"

# Check Chrome version
echo -e "${YELLOW}Chrome version:${NC}"
google-chrome --version

echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "Next: Use the stealth client instead of the regular one"