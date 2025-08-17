#!/usr/bin/env python3
"""
LLM client using saved Chrome profiles
Automates interaction with ChatGPT and Grok
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
import subprocess

class ProfileBasedLLMClient:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.chatgpt_profile = self.base_path / "browser_sessions" / "chrome-chatgpt"
        self.grok_profile = self.base_path / "browser_sessions" / "chrome-grok"
        
        # Review directories
        self.chatgpt_pending = self.base_path / "chatgpt_reviews" / "pending"
        self.chatgpt_completed = self.base_path / "chatgpt_reviews" / "completed"
        self.grok_pending = self.base_path / "grok_reviews" / "pending"
        self.grok_completed = self.base_path / "grok_reviews" / "completed"
        
        # Ensure directories exist
        self.chatgpt_completed.mkdir(parents=True, exist_ok=True)
        self.grok_completed.mkdir(parents=True, exist_ok=True)
    
    async def send_to_chatgpt(self, message: str, headless: bool = True):
        """Send message to ChatGPT using saved Chrome profile"""
        print(f"🤖 Sending to ChatGPT (headless={headless})...")
        
        async with async_playwright() as p:
            # Launch Chrome with saved profile
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(self.chatgpt_profile),
                headless=headless,
                channel="chrome",
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage'
                ]
            )
            
            try:
                # Go to ChatGPT
                page = await browser.new_page()
                await page.goto("https://chat.openai.com", wait_until="networkidle")
                
                # Wait for page to load
                await page.wait_for_timeout(3000)
                
                # Try to start new chat
                try:
                    new_chat = await page.wait_for_selector('a[href="/"]', timeout=3000)
                    await new_chat.click()
                    await page.wait_for_timeout(1000)
                except:
                    pass  # Already in new chat
                
                # Find message input
                print("💬 Typing message...")
                textarea = await page.wait_for_selector('textarea[placeholder*="Message"]', timeout=10000)
                await textarea.fill(message)
                
                # Send message
                await page.keyboard.press('Enter')
                
                # Wait for response to start
                print("⏳ Waiting for response...")
                await page.wait_for_timeout(3000)
                
                # Wait for response to complete
                max_wait = 60
                for i in range(max_wait):
                    # Check if still generating
                    generating = await page.query_selector('button[aria-label*="Stop"]')
                    if not generating:
                        break
                    await asyncio.sleep(1)
                    if i % 5 == 0:
                        print(f"   Still waiting... ({i}s)")
                
                # Get the response
                await page.wait_for_timeout(2000)
                responses = await page.query_selector_all('[data-message-author-role="assistant"]')
                
                if responses:
                    # Get the last assistant message
                    last_response = responses[-1]
                    response_text = await last_response.inner_text()
                    
                    await browser.close()
                    print("✅ Response received!")
                    return response_text
                else:
                    print("❌ No response found")
                    await browser.close()
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                await browser.close()
                return None
    
    async def send_to_grok(self, message: str, headless: bool = True):
        """Send message to Grok using saved Chrome profile"""
        print(f"🧠 Sending to Grok (headless={headless})...")
        
        async with async_playwright() as p:
            # Launch Chrome with saved profile
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(self.grok_profile),
                headless=headless,
                channel="chrome",
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            
            try:
                # Go to Grok
                page = await browser.new_page()
                await page.goto("https://grok.x.ai", wait_until="networkidle")
                
                # Wait for page to load
                await page.wait_for_timeout(3000)
                
                # Find message input (Grok has different selector)
                print("💬 Typing message...")
                textarea = await page.wait_for_selector('textarea', timeout=10000)
                await textarea.fill(message)
                
                # Send message
                await page.keyboard.press('Enter')
                
                # Wait for response
                print("⏳ Waiting for response...")
                await page.wait_for_timeout(5000)
                
                # Wait more for completion
                max_wait = 30
                for i in range(max_wait):
                    await asyncio.sleep(1)
                    if i % 5 == 0:
                        print(f"   Still waiting... ({i}s)")
                    
                    # Check if response appeared
                    responses = await page.query_selector_all('.prose')
                    if len(responses) > 1:  # More than just the prompt
                        break
                
                # Get response
                await page.wait_for_timeout(2000)
                responses = await page.query_selector_all('.prose')
                
                if responses and len(responses) > 1:
                    # Get the last message (skip the user's message)
                    last_response = responses[-1]
                    response_text = await last_response.inner_text()
                    
                    await browser.close()
                    print("✅ Response received!")
                    return response_text
                else:
                    print("❌ No response found")
                    await browser.close()
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                await browser.close()
                return None
    
    async def process_chatgpt_reviews(self):
        """Process all pending ChatGPT reviews"""
        print("\n📋 Processing ChatGPT reviews...")
        
        pending_files = list(self.chatgpt_pending.glob("*.md"))
        if not pending_files:
            print("  No pending ChatGPT reviews")
            return
        
        for review_file in pending_files:
            print(f"\n📄 Processing: {review_file.name}")
            
            # Read the review request
            content = review_file.read_text()
            
            # Send to ChatGPT
            response = await self.send_to_chatgpt(content)
            
            if response:
                # Save response
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                response_file = self.chatgpt_completed / f"sophia_response_{timestamp}.md"
                
                response_content = f"""# Sophia's Response
## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## Original Request: {review_file.name}

---

{response}

---

*Response captured by automated system*
"""
                response_file.write_text(response_content)
                print(f"✅ Response saved to: {response_file.name}")
                
                # Move original request to completed
                completed_request = self.chatgpt_completed / f"processed_{review_file.name}"
                review_file.rename(completed_request)
                print(f"✅ Request moved to completed")
            else:
                print(f"❌ Failed to get response for {review_file.name}")
    
    async def process_grok_reviews(self):
        """Process all pending Grok reviews"""
        print("\n📋 Processing Grok reviews...")
        
        pending_files = list(self.grok_pending.glob("*.md"))
        if not pending_files:
            print("  No pending Grok reviews")
            return
        
        for review_file in pending_files:
            print(f"\n📄 Processing: {review_file.name}")
            
            # Read the review request
            content = review_file.read_text()
            
            # Send to Grok
            response = await self.send_to_grok(content)
            
            if response:
                # Save response
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                response_file = self.grok_completed / f"nexus_response_{timestamp}.md"
                
                response_content = f"""# Nexus's Response
## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## Original Request: {review_file.name}

---

{response}

---

*Response captured by automated system*
"""
                response_file.write_text(response_content)
                print(f"✅ Response saved to: {response_file.name}")
                
                # Move original request to completed
                completed_request = self.grok_completed / f"processed_{review_file.name}"
                review_file.rename(completed_request)
                print(f"✅ Request moved to completed")
            else:
                print(f"❌ Failed to get response for {review_file.name}")
    
    async def test_connections(self):
        """Test both ChatGPT and Grok connections"""
        print("\n🔍 Testing LLM Connections")
        print("=" * 50)
        
        # Test ChatGPT
        print("\n1. Testing ChatGPT...")
        response = await self.send_to_chatgpt(
            "Hello! This is a test message. Please respond with 'ChatGPT connection successful' if you receive this.",
            headless=True  # Run headless to avoid display issues
        )
        if response:
            print(f"✅ ChatGPT test successful!")
            print(f"   Response preview: {response[:100]}...")
        else:
            print("❌ ChatGPT test failed")
        
        # Test Grok
        print("\n2. Testing Grok...")
        response = await self.send_to_grok(
            "Hello! This is a test message. Please respond with 'Grok connection successful' if you receive this.",
            headless=True  # Run headless to avoid display issues
        )
        if response:
            print(f"✅ Grok test successful!")
            print(f"   Response preview: {response[:100]}...")
        else:
            print("❌ Grok test failed")

async def main():
    if len(sys.argv) < 2:
        print("""
LLM Profile Client - Automated ChatGPT & Grok Interaction
==========================================================

Usage: python3 llm_profile_client.py [command]

Commands:
  test              - Test both ChatGPT and Grok connections
  process all       - Process all pending reviews
  process chatgpt   - Process only ChatGPT reviews
  process grok      - Process only Grok reviews
  send chatgpt "message"  - Send a message to ChatGPT
  send grok "message"     - Send a message to Grok

Prerequisites:
  1. Chrome profiles must be set up with:
     python3 scripts/simple_browser_setup.py setup both
  2. You must be logged into both services
        """)
        sys.exit(1)
    
    client = ProfileBasedLLMClient()
    command = sys.argv[1]
    
    if command == "test":
        await client.test_connections()
    
    elif command == "process" and len(sys.argv) > 2:
        target = sys.argv[2]
        if target == "all":
            await client.process_chatgpt_reviews()
            await client.process_grok_reviews()
        elif target == "chatgpt":
            await client.process_chatgpt_reviews()
        elif target == "grok":
            await client.process_grok_reviews()
    
    elif command == "send" and len(sys.argv) > 3:
        platform = sys.argv[2]
        message = sys.argv[3]
        if platform == "chatgpt":
            response = await client.send_to_chatgpt(message)
            if response:
                print(f"\nResponse:\n{response}")
        elif platform == "grok":
            response = await client.send_to_grok(message)
            if response:
                print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    import os
    os.environ["DISPLAY"] = ":0"  # Ensure display is set
    asyncio.run(main())