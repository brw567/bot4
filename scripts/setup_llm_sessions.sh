#!/bin/bash
# Setup LLM sessions with Chrome remote debugging

echo "🚀 LLM Session Setup Script"
echo "=========================="
echo ""
echo "This script will help you set up ChatGPT and Grok sessions."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to setup ChatGPT
setup_chatgpt() {
    echo -e "${YELLOW}Setting up ChatGPT...${NC}"
    
    # Launch Chrome with debugging
    echo "1. Opening Chrome with debugging enabled..."
    DISPLAY=:0 google-chrome \
        --remote-debugging-port=9222 \
        --user-data-dir=/home/hamster/bot4/browser_sessions/chrome-chatgpt \
        "https://chat.openai.com" &
    
    CHROME_PID=$!
    sleep 5
    
    echo ""
    echo -e "${GREEN}Chrome opened with ChatGPT${NC}"
    echo "2. Please log in to ChatGPT in the browser window"
    echo "3. After logging in, press Enter here..."
    read -p ""
    
    # Now capture the session using Playwright
    echo "Capturing session..."
    DISPLAY=:0 python3 -c "
import asyncio
from playwright.async_api import async_playwright
import json
from pathlib import Path

async def capture():
    async with async_playwright() as p:
        # Connect to existing Chrome
        browser = await p.chromium.connect_over_cdp('http://localhost:9222')
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
            # Save the state
            state = await context.storage_state()
            output = Path('/home/hamster/bot4/browser_sessions/chatgpt_session.json')
            output.write_text(json.dumps(state, indent=2))
            print(f'✅ Session saved to {output}')
        await browser.close()

asyncio.run(capture())
" 2>/dev/null
    
    # Kill Chrome
    kill $CHROME_PID 2>/dev/null
    
    echo -e "${GREEN}✅ ChatGPT setup complete!${NC}"
}

# Function to setup Grok
setup_grok() {
    echo -e "${YELLOW}Setting up Grok...${NC}"
    
    # Launch Chrome with debugging
    echo "1. Opening Chrome with debugging enabled..."
    DISPLAY=:0 google-chrome \
        --remote-debugging-port=9222 \
        --user-data-dir=/home/hamster/bot4/browser_sessions/chrome-grok \
        "https://grok.x.ai" &
    
    CHROME_PID=$!
    sleep 5
    
    echo ""
    echo -e "${GREEN}Chrome opened with Grok${NC}"
    echo "2. Please log in to Grok in the browser window"
    echo "3. After logging in, press Enter here..."
    read -p ""
    
    # Capture session
    echo "Capturing session..."
    DISPLAY=:0 python3 -c "
import asyncio
from playwright.async_api import async_playwright
import json
from pathlib import Path

async def capture():
    async with async_playwright() as p:
        # Connect to existing Chrome
        browser = await p.chromium.connect_over_cdp('http://localhost:9222')
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
            # Save the state
            state = await context.storage_state()
            output = Path('/home/hamster/bot4/browser_sessions/grok_session.json')
            output.write_text(json.dumps(state, indent=2))
            print(f'✅ Session saved to {output}')
        await browser.close()

asyncio.run(capture())
" 2>/dev/null
    
    # Kill Chrome
    kill $CHROME_PID 2>/dev/null
    
    echo -e "${GREEN}✅ Grok setup complete!${NC}"
}

# Main menu
echo "What would you like to set up?"
echo "1) ChatGPT"
echo "2) Grok"
echo "3) Both"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        setup_chatgpt
        ;;
    2)
        setup_grok
        ;;
    3)
        setup_chatgpt
        echo ""
        setup_grok
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "You can now use:"
echo "  python3 scripts/llm_browser_bridge.py test chatgpt"
echo "  python3 scripts/llm_browser_bridge.py test grok"
echo "  python3 scripts/llm_browser_bridge.py process all"