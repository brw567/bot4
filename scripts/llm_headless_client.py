#!/usr/bin/env python3
"""
Headless Chromium client for ChatGPT and Grok
Optimized for Ubuntu 22.04 server environments
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# For initial setup without playwright
def check_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        print("❌ Playwright not installed!")
        print("\nTo install, run:")
        print("  chmod +x scripts/install_playwright.sh")
        print("  ./scripts/install_playwright.sh")
        return False

if check_playwright():
    from playwright.sync_api import sync_playwright

class HeadlessLLMClient:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.sessions_path = self.base_path / "browser_sessions"
        self.sessions_path.mkdir(exist_ok=True)
        
        # Session storage files
        self.chatgpt_session = self.sessions_path / "chatgpt_auth.json"
        self.grok_session = self.sessions_path / "grok_auth.json"
        
    def setup_chatgpt_session(self):
        """
        One-time setup for ChatGPT session
        This needs to be run with display (not headless) for initial login
        """
        print("\n🔐 ChatGPT Session Setup")
        print("=" * 50)
        print("This will open a browser window for you to log in.")
        print("After logging in, the session will be saved for headless use.")
        print("=" * 50)
        
        with sync_playwright() as p:
            # Launch with GUI for login
            browser = p.chromium.launch(
                headless=False,
                args=['--no-sandbox']
            )
            
            context = browser.new_context()
            page = context.new_page()
            
            print("\n🌐 Opening ChatGPT...")
            page.goto("https://chat.openai.com")
            
            print("\n⏳ Please log in to ChatGPT in the browser window")
            print("   After logging in successfully, press Enter here...")
            input()
            
            # Save storage state (cookies, localStorage, etc.)
            storage = context.storage_state()
            with open(self.chatgpt_session, 'w') as f:
                json.dump(storage, f, indent=2)
            
            browser.close()
            
        print("✅ ChatGPT session saved successfully!")
        return True
    
    def setup_grok_session(self):
        """
        One-time setup for Grok session
        This needs to be run with display (not headless) for initial login
        """
        print("\n🔐 Grok Session Setup")
        print("=" * 50)
        print("This will open a browser window for you to log in.")
        print("After logging in, the session will be saved for headless use.")
        print("=" * 50)
        
        with sync_playwright() as p:
            # Launch with GUI for login
            browser = p.chromium.launch(
                headless=False,
                args=['--no-sandbox']
            )
            
            context = browser.new_context()
            page = context.new_page()
            
            print("\n🌐 Opening Grok...")
            page.goto("https://grok.x.ai")
            
            print("\n⏳ Please log in to Grok in the browser window")
            print("   After logging in successfully, press Enter here...")
            input()
            
            # Save storage state
            storage = context.storage_state()
            with open(self.grok_session, 'w') as f:
                json.dump(storage, f, indent=2)
            
            browser.close()
            
        print("✅ Grok session saved successfully!")
        return True
    
    def send_to_chatgpt_headless(self, message: str) -> Optional[str]:
        """
        Send message to ChatGPT using headless browser
        """
        if not self.chatgpt_session.exists():
            print("❌ No ChatGPT session found. Run setup first:")
            print("   python3 scripts/llm_headless_client.py setup chatgpt")
            return None
        
        print("🤖 Connecting to ChatGPT (headless)...")
        
        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(
                headless=True,  # Headless mode!
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-gpu',
                    '--disable-gpu',
                    '--window-size=1920,1080'
                ]
            )
            
            # Load saved session
            with open(self.chatgpt_session, 'r') as f:
                storage_state = json.load(f)
            
            context = browser.new_context(
                storage_state=storage_state,
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = context.new_page()
            
            try:
                print("📡 Loading ChatGPT...")
                page.goto("https://chat.openai.com", wait_until='networkidle')
                
                # Wait a bit for full load
                page.wait_for_timeout(3000)
                
                # Try to start new chat
                try:
                    page.click('a[href="/"]', timeout=3000)
                except:
                    pass  # Already in new chat
                
                # Find message input
                print("💬 Sending message...")
                textarea = page.wait_for_selector('textarea[placeholder*="Message"]', timeout=10000)
                textarea.fill(message)
                
                # Send message
                page.keyboard.press('Enter')
                
                # Wait for response to start
                print("⏳ Waiting for response...")
                page.wait_for_timeout(3000)
                
                # Wait for response to complete (look for specific indicators)
                max_wait = 30
                for i in range(max_wait):
                    # Check if still generating
                    generating = page.query_selector('button[aria-label*="Stop"]')
                    if not generating:
                        break
                    time.sleep(1)
                    if i % 5 == 0:
                        print(f"   Still waiting... ({i}s)")
                
                # Get the response
                response_elements = page.query_selector_all('[data-message-author-role="assistant"]')
                
                if response_elements:
                    # Get the last assistant message
                    last_response = response_elements[-1]
                    response_text = last_response.inner_text()
                    
                    browser.close()
                    print("✅ Response received!")
                    return response_text
                else:
                    print("❌ No response found")
                    browser.close()
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                browser.close()
                return None
    
    def send_to_grok_headless(self, message: str) -> Optional[str]:
        """
        Send message to Grok using headless browser
        """
        if not self.grok_session.exists():
            print("❌ No Grok session found. Run setup first:")
            print("   python3 scripts/llm_headless_client.py setup grok")
            return None
        
        print("🤖 Connecting to Grok (headless)...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--no-gpu'
                ]
            )
            
            # Load saved session
            with open(self.grok_session, 'r') as f:
                storage_state = json.load(f)
            
            context = browser.new_context(storage_state=storage_state)
            page = context.new_page()
            
            try:
                print("📡 Loading Grok...")
                page.goto("https://grok.x.ai", wait_until='networkidle')
                page.wait_for_timeout(3000)
                
                # Find and fill message input
                print("💬 Sending message...")
                textarea = page.wait_for_selector('textarea', timeout=10000)
                textarea.fill(message)
                
                # Send message
                page.keyboard.press('Enter')
                
                # Wait for response
                print("⏳ Waiting for response...")
                page.wait_for_timeout(5000)
                
                # Get response (adjust selector as needed)
                response_elements = page.query_selector_all('.message-content')
                
                if response_elements and len(response_elements) > 1:
                    last_response = response_elements[-1]
                    response_text = last_response.inner_text()
                    
                    browser.close()
                    print("✅ Response received!")
                    return response_text
                else:
                    print("❌ No response found")
                    browser.close()
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                browser.close()
                return None
    
    def process_reviews(self):
        """
        Process all pending reviews automatically
        """
        print("\n🔄 Processing pending reviews...")
        
        # Process ChatGPT reviews
        chatgpt_pending = self.base_path / "chatgpt_reviews" / "pending"
        for review_file in chatgpt_pending.glob("*.md"):
            print(f"\n📄 Processing ChatGPT review: {review_file.name}")
            
            content = review_file.read_text()
            response = self.send_to_chatgpt_headless(content)
            
            if response:
                # Save response
                completed_dir = self.base_path / "chatgpt_reviews" / "completed"
                response_file = completed_dir / f"{review_file.stem}_response.md"
                response_file.write_text(f"# Sophia's Response\n\n{response}")
                
                # Move request to processed
                review_file.rename(completed_dir / f"{review_file.stem}_processed.md")
                print(f"✅ Saved response to {response_file.name}")
        
        # Process Grok reviews
        grok_pending = self.base_path / "grok_reviews" / "pending"
        for review_file in grok_pending.glob("*.md"):
            print(f"\n📄 Processing Grok review: {review_file.name}")
            
            content = review_file.read_text()
            response = self.send_to_grok_headless(content)
            
            if response:
                # Save response
                completed_dir = self.base_path / "grok_reviews" / "completed"
                response_file = completed_dir / f"{review_file.stem}_response.md"
                response_file.write_text(f"# Nexus's Response\n\n{response}")
                
                # Move request to processed
                review_file.rename(completed_dir / f"{review_file.stem}_processed.md")
                print(f"✅ Saved response to {response_file.name}")
        
        print("\n✅ Review processing complete!")

def main():
    if len(sys.argv) < 2:
        print("""
Headless LLM Client for ChatGPT and Grok
=========================================

Usage: python3 llm_headless_client.py [command]

Commands:
  setup chatgpt    - Set up ChatGPT session (requires display)
  setup grok       - Set up Grok session (requires display)
  test chatgpt     - Test ChatGPT connection (headless)
  test grok        - Test Grok connection (headless)
  process          - Process all pending reviews (headless)
  
First time setup:
  1. Run: ./scripts/install_playwright.sh
  2. Run: python3 scripts/llm_headless_client.py setup chatgpt
  3. Run: python3 scripts/llm_headless_client.py setup grok
  
Then you can use headless mode:
  python3 scripts/llm_headless_client.py process
        """)
        sys.exit(1)
    
    client = HeadlessLLMClient()
    command = sys.argv[1]
    
    if command == "setup" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            client.setup_chatgpt_session()
        elif platform == "grok":
            client.setup_grok_session()
        else:
            print(f"Unknown platform: {platform}")
    
    elif command == "test" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            response = client.send_to_chatgpt_headless("Hello! Reply with 'Connection successful'")
            if response:
                print(f"\n✅ ChatGPT test successful!")
                print(f"Response: {response[:200]}")
        elif platform == "grok":
            response = client.send_to_grok_headless("Hello! Reply with 'Connection successful'")
            if response:
                print(f"\n✅ Grok test successful!")
                print(f"Response: {response[:200]}")
    
    elif command == "process":
        client.process_reviews()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    if not check_playwright():
        sys.exit(1)
    main()