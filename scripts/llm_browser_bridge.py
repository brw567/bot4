#!/usr/bin/env python3
"""
Browser automation bridge for ChatGPT and Grok interaction
Requires: pip install playwright asyncio
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Check if playwright is installed
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not installed. Install with: pip install playwright")
    print("   Then run: playwright install chromium")

class LLMBrowserBridge:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.cookies_path = self.base_path / "browser_sessions"
        self.cookies_path.mkdir(exist_ok=True)
        
        self.chatgpt_cookies = self.cookies_path / "chatgpt_cookies.json"
        self.grok_cookies = self.cookies_path / "grok_cookies.json"
        
        self.browser = None
        self.context = None
        
    async def setup_browser(self, headless: bool = False):
        """Initialize browser with saved session"""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available")
            return False
            
        playwright = await async_playwright().start()
        
        # Use headless=False for first login, then True for automation
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        return True
    
    async def save_session_interactive(self, platform: str):
        """Interactive session to save cookies"""
        print(f"\n🔐 Setting up {platform} session...")
        print("1. A browser window will open")
        print("2. Please log in to your account")
        print("3. Once logged in, press Enter here")
        
        await self.setup_browser(headless=False)
        self.context = await self.browser.new_context()
        page = await self.context.new_page()
        
        if platform == "chatgpt":
            await page.goto("https://chat.openai.com")
            print("🌐 Opening ChatGPT... Please log in.")
        elif platform == "grok":
            await page.goto("https://grok.x.ai")
            print("🌐 Opening Grok... Please log in.")
        
        # Wait for user to log in
        input("\n✅ Press Enter after you've logged in successfully...")
        
        # Save cookies
        cookies = await self.context.cookies()
        cookies_file = self.chatgpt_cookies if platform == "chatgpt" else self.grok_cookies
        
        with open(cookies_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"✅ Session saved to {cookies_file}")
        
        await self.browser.close()
        return True
    
    async def load_session(self, platform: str):
        """Load saved session cookies"""
        cookies_file = self.chatgpt_cookies if platform == "chatgpt" else self.grok_cookies
        
        if not cookies_file.exists():
            print(f"❌ No saved session for {platform}")
            print(f"   Run: python3 {__file__} setup {platform}")
            return False
        
        with open(cookies_file, 'r') as f:
            cookies = json.load(f)
        
        await self.context.add_cookies(cookies)
        return True
    
    async def send_to_chatgpt(self, message: str) -> Optional[str]:
        """Send message to ChatGPT and get response"""
        if not PLAYWRIGHT_AVAILABLE:
            return None
            
        try:
            await self.setup_browser(headless=True)
            self.context = await self.browser.new_context()
            
            # Load saved session
            if not await self.load_session("chatgpt"):
                return None
            
            page = await self.context.new_page()
            await page.goto("https://chat.openai.com")
            
            # Wait for page to load
            await page.wait_for_timeout(3000)
            
            # Find and click new chat button if needed
            try:
                await page.click('button[aria-label="New chat"]', timeout=5000)
            except:
                pass  # Might already be in new chat
            
            # Find the input textarea
            await page.wait_for_selector('textarea[placeholder*="Message"]', timeout=10000)
            
            # Type the message
            await page.fill('textarea[placeholder*="Message"]', message)
            
            # Send the message
            await page.keyboard.press('Enter')
            
            # Wait for response (with timeout)
            print("⏳ Waiting for ChatGPT response...")
            await page.wait_for_timeout(5000)  # Initial wait
            
            # Wait for the response to complete (check for stop generating button to disappear)
            try:
                await page.wait_for_selector('button[aria-label*="Stop"]', state='hidden', timeout=30000)
            except:
                pass  # Response might be quick
            
            # Extract the last response
            responses = await page.query_selector_all('.markdown.prose')
            if responses:
                last_response = responses[-1]
                response_text = await last_response.inner_text()
                
                await self.browser.close()
                return response_text
            
            await self.browser.close()
            return None
            
        except Exception as e:
            print(f"❌ Error interacting with ChatGPT: {e}")
            if self.browser:
                await self.browser.close()
            return None
    
    async def send_to_grok(self, message: str) -> Optional[str]:
        """Send message to Grok and get response"""
        if not PLAYWRIGHT_AVAILABLE:
            return None
            
        try:
            await self.setup_browser(headless=True)
            self.context = await self.browser.new_context()
            
            # Load saved session
            if not await self.load_session("grok"):
                return None
            
            page = await self.context.new_page()
            await page.goto("https://grok.x.ai")
            
            # Wait for page to load
            await page.wait_for_timeout(3000)
            
            # Find the input field (Grok's selector might vary)
            await page.wait_for_selector('textarea', timeout=10000)
            
            # Type the message
            await page.fill('textarea', message)
            
            # Send the message
            await page.keyboard.press('Enter')
            
            # Wait for response
            print("⏳ Waiting for Grok response...")
            await page.wait_for_timeout(5000)
            
            # Extract response (selectors might need adjustment)
            responses = await page.query_selector_all('.message-content')
            if responses:
                last_response = responses[-1]
                response_text = await last_response.inner_text()
                
                await self.browser.close()
                return response_text
            
            await self.browser.close()
            return None
            
        except Exception as e:
            print(f"❌ Error interacting with Grok: {e}")
            if self.browser:
                await self.browser.close()
            return None
    
    async def process_review_request(self, platform: str):
        """Process pending review requests"""
        if platform == "chatgpt":
            pending_dir = self.base_path / "chatgpt_reviews" / "pending"
            completed_dir = self.base_path / "chatgpt_reviews" / "completed"
            send_func = self.send_to_chatgpt
        else:
            pending_dir = self.base_path / "grok_reviews" / "pending"
            completed_dir = self.base_path / "grok_reviews" / "completed"
            send_func = self.send_to_grok
        
        # Find pending reviews
        pending_files = list(pending_dir.glob("*.md"))
        
        if not pending_files:
            print(f"📭 No pending reviews for {platform}")
            return
        
        for file in pending_files:
            print(f"\n📄 Processing: {file.name}")
            
            # Read the request
            request = file.read_text()
            
            # Send to LLM
            print(f"📤 Sending to {platform}...")
            response = await send_func(request)
            
            if response:
                # Save response
                response_file = completed_dir / f"{file.stem}_response.md"
                response_file.write_text(response)
                print(f"✅ Response saved to {response_file}")
                
                # Move request to completed
                file.rename(completed_dir / f"{file.stem}_sent.md")
            else:
                print(f"❌ Failed to get response from {platform}")

async def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("""
Usage: python3 llm_browser_bridge.py [command] [args]

Commands:
  setup chatgpt    - Save ChatGPT session (interactive)
  setup grok       - Save Grok session (interactive)
  test chatgpt     - Test ChatGPT connection
  test grok        - Test Grok connection
  process chatgpt  - Process pending ChatGPT reviews
  process grok     - Process pending Grok reviews
  process all      - Process all pending reviews
        """)
        sys.exit(1)
    
    bridge = LLMBrowserBridge()
    command = sys.argv[1]
    
    if command == "setup" and len(sys.argv) > 2:
        platform = sys.argv[2]
        await bridge.save_session_interactive(platform)
        
    elif command == "test" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            response = await bridge.send_to_chatgpt("Hello! Please respond with: 'ChatGPT connection successful'")
        else:
            response = await bridge.send_to_grok("Hello! Please respond with: 'Grok connection successful'")
        
        if response:
            print(f"✅ {platform} test successful!")
            print(f"Response: {response[:200]}...")
        else:
            print(f"❌ {platform} test failed")
    
    elif command == "process":
        if len(sys.argv) > 2:
            target = sys.argv[2]
            if target == "all":
                await bridge.process_review_request("chatgpt")
                await bridge.process_review_request("grok")
            else:
                await bridge.process_review_request(target)
        else:
            print("Specify: chatgpt, grok, or all")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    if not PLAYWRIGHT_AVAILABLE:
        print("\n📦 Installation required:")
        print("1. pip install playwright")
        print("2. playwright install chromium")
        print("3. playwright install-deps  # if on Linux")
        sys.exit(1)
    
    asyncio.run(main())