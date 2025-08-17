#!/usr/bin/env python3
"""
Stealth browser client for ChatGPT and Grok
Uses real Chrome with anti-detection measures
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Try multiple approaches
USE_PLAYWRIGHT = False
USE_SELENIUM = False

try:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import stealth_sync
    USE_PLAYWRIGHT = True
    print("✓ Using Playwright with stealth")
except ImportError:
    print("⚠ Playwright not available, trying Selenium...")

if not USE_PLAYWRIGHT:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import undetected_chromedriver as uc
        USE_SELENIUM = True
        print("✓ Using Selenium with undetected-chromedriver")
    except ImportError:
        print("❌ Neither Playwright nor Selenium available!")
        print("\nInstall with:")
        print("  ./scripts/install_chrome.sh")
        sys.exit(1)

class StealthLLMClient:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.sessions_path = self.base_path / "browser_sessions"
        self.sessions_path.mkdir(exist_ok=True)
        
        self.chatgpt_cookies = self.sessions_path / "chatgpt_cookies.json"
        self.grok_cookies = self.sessions_path / "grok_cookies.json"
    
    def setup_chatgpt_selenium(self):
        """Setup ChatGPT using Selenium with undetected Chrome"""
        print("\n🔐 ChatGPT Setup (Selenium Stealth Mode)")
        print("=" * 50)
        
        # Use undetected-chromedriver to bypass detection
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Remove problematic options for compatibility
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('useAutomationExtension', False)
        
        # Create driver with undetected-chromedriver
        driver = uc.Chrome(options=options, version_main=None, use_subprocess=True)
        
        # Additional stealth
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        print("🌐 Opening ChatGPT...")
        driver.get("https://chat.openai.com")
        
        print("\n⏳ Please log in to ChatGPT")
        print("   After logging in, press Enter here...")
        input()
        
        # Save cookies
        cookies = driver.get_cookies()
        with open(self.chatgpt_cookies, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        driver.quit()
        print("✅ ChatGPT session saved!")
        return True
    
    def setup_grok_selenium(self):
        """Setup Grok using Selenium with undetected Chrome"""
        print("\n🔐 Grok Setup (Selenium Stealth Mode)")
        print("=" * 50)
        
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        driver = uc.Chrome(options=options, version_main=None, use_subprocess=True)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("🌐 Opening Grok...")
        driver.get("https://grok.x.ai")
        
        print("\n⏳ Please log in to Grok")
        print("   After logging in, press Enter here...")
        input()
        
        # Save cookies
        cookies = driver.get_cookies()
        with open(self.grok_cookies, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        driver.quit()
        print("✅ Grok session saved!")
        return True
    
    def setup_chatgpt_playwright(self):
        """Setup ChatGPT using Playwright with stealth"""
        print("\n🔐 ChatGPT Setup (Playwright Stealth Mode)")
        print("=" * 50)
        
        with sync_playwright() as p:
            # Use real Chrome instead of Chromium
            browser = p.chromium.launch(
                headless=False,
                channel="chrome",  # Use real Chrome
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=site-per-process',
                    '--disable-web-security',
                ]
            )
            
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            page = context.new_page()
            
            # Apply stealth techniques
            stealth_sync(page)
            
            print("🌐 Opening ChatGPT...")
            page.goto("https://chat.openai.com")
            
            print("\n⏳ Please log in to ChatGPT")
            print("   After logging in, press Enter here...")
            input()
            
            # Save session
            storage = context.storage_state()
            with open(self.chatgpt_cookies, 'w') as f:
                json.dump(storage, f, indent=2)
            
            browser.close()
            
        print("✅ ChatGPT session saved!")
        return True
    
    def send_to_chatgpt_headless(self, message: str):
        """Send message to ChatGPT using saved session"""
        if not self.chatgpt_cookies.exists():
            print("❌ No session found. Run setup first.")
            return None
        
        if USE_SELENIUM:
            return self._send_chatgpt_selenium(message)
        else:
            return self._send_chatgpt_playwright(message)
    
    def _send_chatgpt_selenium(self, message: str):
        """Send using Selenium"""
        print("🤖 Sending to ChatGPT via Selenium...")
        
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")  # New headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        
        driver = uc.Chrome(options=options, version_main=None)
        
        try:
            # Load cookies
            driver.get("https://chat.openai.com")
            with open(self.chatgpt_cookies, 'r') as f:
                cookies = json.load(f)
            for cookie in cookies:
                if 'sameSite' in cookie:
                    del cookie['sameSite']
                driver.add_cookie(cookie)
            
            # Refresh to apply cookies
            driver.refresh()
            time.sleep(3)
            
            # Find and fill textarea
            wait = WebDriverWait(driver, 10)
            textarea = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[placeholder*='Message']"))
            )
            
            textarea.send_keys(message)
            textarea.send_keys(Keys.RETURN)
            
            # Wait for response
            print("⏳ Waiting for response...")
            time.sleep(10)  # Initial wait
            
            # Get response
            response_elements = driver.find_elements(By.CSS_SELECTOR, "[data-message-author-role='assistant']")
            if response_elements:
                response = response_elements[-1].text
                driver.quit()
                return response
            
            driver.quit()
            return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            driver.quit()
            return None
    
    def _send_chatgpt_playwright(self, message: str):
        """Send using Playwright"""
        print("🤖 Sending to ChatGPT via Playwright...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                channel="chrome",
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Load saved session
            with open(self.chatgpt_cookies, 'r') as f:
                storage = json.load(f)
            
            context = browser.new_context(storage_state=storage)
            page = context.new_page()
            stealth_sync(page)
            
            try:
                page.goto("https://chat.openai.com")
                page.wait_for_timeout(3000)
                
                # Send message
                textarea = page.wait_for_selector("textarea[placeholder*='Message']", timeout=10000)
                textarea.fill(message)
                page.keyboard.press("Enter")
                
                # Wait for response
                print("⏳ Waiting for response...")
                page.wait_for_timeout(10000)
                
                # Get response
                responses = page.query_selector_all("[data-message-author-role='assistant']")
                if responses:
                    response = responses[-1].inner_text()
                    browser.close()
                    return response
                
                browser.close()
                return None
                
            except Exception as e:
                print(f"❌ Error: {e}")
                browser.close()
                return None
    
    def process_reviews(self):
        """Process all pending reviews"""
        print("\n🔄 Processing pending reviews...")
        
        # Process ChatGPT reviews
        chatgpt_pending = self.base_path / "chatgpt_reviews" / "pending"
        for review_file in chatgpt_pending.glob("*.md"):
            print(f"\n📄 Processing: {review_file.name}")
            
            content = review_file.read_text()
            response = self.send_to_chatgpt_headless(content)
            
            if response:
                # Save response
                completed_dir = self.base_path / "chatgpt_reviews" / "completed"
                completed_dir.mkdir(exist_ok=True)
                response_file = completed_dir / f"{review_file.stem}_response.md"
                response_file.write_text(f"# Sophia's Response\n\n{response}")
                print(f"✅ Response saved to {response_file.name}")
                
                # Move original
                review_file.rename(completed_dir / f"{review_file.stem}_processed.md")

def main():
    if len(sys.argv) < 2:
        print("""
Stealth LLM Client (Anti-Detection)
====================================

Usage: python3 llm_stealth_client.py [command]

Commands:
  setup chatgpt    - Set up ChatGPT session
  setup grok       - Set up Grok session
  test chatgpt     - Test ChatGPT connection
  test grok        - Test Grok connection
  process          - Process all pending reviews

First install Chrome:
  chmod +x scripts/install_chrome.sh
  ./scripts/install_chrome.sh
        """)
        sys.exit(1)
    
    client = StealthLLMClient()
    command = sys.argv[1]
    
    if command == "setup" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            if USE_SELENIUM:
                client.setup_chatgpt_selenium()
            else:
                client.setup_chatgpt_playwright()
        elif platform == "grok":
            if USE_SELENIUM:
                client.setup_grok_selenium()
            else:
                print("Grok Playwright setup not implemented yet")
    
    elif command == "test" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            response = client.send_to_chatgpt_headless("Hello! Reply with 'Connection successful'")
            if response:
                print(f"\n✅ Test successful!")
                print(f"Response: {response[:200]}")
    
    elif command == "process":
        client.process_reviews()

if __name__ == "__main__":
    main()