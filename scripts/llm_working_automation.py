#!/usr/bin/env python3
"""
Working LLM automation with correct ChromeDriver
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class WorkingLLMAutomation:
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
    
    def create_driver(self, profile_dir: Path, headless: bool = True):
        """Create Chrome driver with profile"""
        options = Options()
        options.add_argument(f"--user-data-dir={profile_dir}")
        
        if headless:
            options.add_argument("--headless=new")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        
        # Use webdriver-manager to get correct driver
        service = Service(ChromeDriverManager().install())
        
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    
    def send_to_chatgpt(self, message: str, headless: bool = True):
        """Send message to ChatGPT"""
        print(f"\n🤖 Sending to ChatGPT (headless={headless})...")
        
        driver = None
        try:
            driver = self.create_driver(self.chatgpt_profile, headless)
            
            # Go to ChatGPT
            print("   Loading ChatGPT...")
            driver.get("https://chat.openai.com")
            time.sleep(5)
            
            # Check if logged in by looking for the textarea
            try:
                # Try to start new chat
                try:
                    new_chat = driver.find_element(By.CSS_SELECTOR, 'a[href="/"]')
                    new_chat.click()
                    time.sleep(2)
                except:
                    pass  # Already in new chat or button not found
                
                # Find and fill textarea
                print("   Typing message...")
                textarea = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'textarea[placeholder*="Message"]'))
                )
                textarea.clear()
                textarea.send_keys(message)
                textarea.send_keys(Keys.RETURN)
                
                # Wait for response
                print("   Waiting for response...")
                time.sleep(5)  # Initial wait
                
                # Wait for response to complete
                max_wait = 60
                last_text = ""
                stable_count = 0
                
                for i in range(max_wait):
                    time.sleep(2)
                    
                    # Get all assistant messages
                    responses = driver.find_elements(By.CSS_SELECTOR, '[data-message-author-role="assistant"]')
                    
                    if responses:
                        current_text = responses[-1].text
                        
                        # Check if text has stabilized (response complete)
                        if current_text == last_text:
                            stable_count += 1
                            if stable_count >= 2:  # Text unchanged for 4 seconds
                                print("   Response complete!")
                                driver.quit()
                                return current_text
                        else:
                            stable_count = 0
                            last_text = current_text
                    
                    if i % 5 == 0 and i > 0:
                        print(f"   Still waiting... ({i*2}s)")
                
                # Return whatever we have
                if responses:
                    response_text = responses[-1].text
                    driver.quit()
                    return response_text
                
            except Exception as e:
                print(f"   ❌ Not logged in or error: {e}")
                print("   Please run: python3 scripts/simple_browser_setup.py setup chatgpt")
                
            driver.quit()
            return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if driver:
                driver.quit()
            return None
    
    def send_to_grok(self, message: str, headless: bool = True):
        """Send message to Grok"""
        print(f"\n🧠 Sending to Grok (headless={headless})...")
        
        driver = None
        try:
            driver = self.create_driver(self.grok_profile, headless)
            
            # Go to Grok
            print("   Loading Grok...")
            driver.get("https://grok.x.ai")
            time.sleep(5)
            
            # Find and fill textarea
            try:
                print("   Typing message...")
                # Grok might have different selector
                textarea = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'textarea'))
                )
                textarea.clear()
                textarea.send_keys(message)
                textarea.send_keys(Keys.RETURN)
                
                # Wait for response
                print("   Waiting for response...")
                time.sleep(10)  # Grok might be slower
                
                # Look for response elements
                # Adjust selector based on Grok's actual structure
                responses = driver.find_elements(By.CLASS_NAME, 'prose')
                
                if len(responses) > 1:
                    response_text = responses[-1].text
                    driver.quit()
                    return response_text
                    
            except Exception as e:
                print(f"   ❌ Not logged in or error: {e}")
                print("   Please run: python3 scripts/simple_browser_setup.py setup grok")
            
            driver.quit()
            return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if driver:
                driver.quit()
            return None
    
    def test_connections(self):
        """Test both connections"""
        print("\n" + "="*60)
        print("🔍 Testing LLM Connections")
        print("="*60)
        
        # Test ChatGPT
        print("\n1. Testing ChatGPT...")
        response = self.send_to_chatgpt(
            "Hello! This is a test. Please respond with 'ChatGPT connection successful'.",
            headless=True
        )
        
        if response:
            print(f"✅ ChatGPT test successful!")
            print(f"   Response preview: {response[:200]}...")
        else:
            print("❌ ChatGPT test failed")
        
        # Test Grok
        print("\n2. Testing Grok...")
        response = self.send_to_grok(
            "Hello! This is a test. Please respond with 'Grok connection successful'.",
            headless=True
        )
        
        if response:
            print(f"✅ Grok test successful!")
            print(f"   Response preview: {response[:200]}...")
        else:
            print("❌ Grok test failed")
        
        print("\n" + "="*60)
    
    def process_all_reviews(self):
        """Process all pending reviews"""
        print("\n📋 Processing All Pending Reviews")
        print("="*60)
        
        # Process ChatGPT reviews
        chatgpt_files = list(self.chatgpt_pending.glob("*.md"))
        if chatgpt_files:
            print(f"\n📄 Found {len(chatgpt_files)} ChatGPT review(s)")
            for file in chatgpt_files:
                print(f"\n   Processing: {file.name}")
                content = file.read_text()
                
                response = self.send_to_chatgpt(content)
                
                if response:
                    # Save response
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    response_file = self.chatgpt_completed / f"sophia_response_{timestamp}.md"
                    
                    full_response = f"""# Sophia's Response
## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## Original Request: {file.name}

---

{response}

---

*Response captured by automated system*
"""
                    response_file.write_text(full_response)
                    print(f"   ✅ Response saved to: {response_file.name}")
                    
                    # Move original to completed
                    completed = self.chatgpt_completed / f"processed_{file.name}"
                    file.rename(completed)
                else:
                    print(f"   ❌ Failed to get response")
        else:
            print("\n✅ No pending ChatGPT reviews")
        
        # Process Grok reviews
        grok_files = list(self.grok_pending.glob("*.md"))
        if grok_files:
            print(f"\n📄 Found {len(grok_files)} Grok review(s)")
            for file in grok_files:
                print(f"\n   Processing: {file.name}")
                content = file.read_text()
                
                response = self.send_to_grok(content)
                
                if response:
                    # Save response
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    response_file = self.grok_completed / f"nexus_response_{timestamp}.md"
                    
                    full_response = f"""# Nexus's Response
## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## Original Request: {file.name}

---

{response}

---

*Response captured by automated system*
"""
                    response_file.write_text(full_response)
                    print(f"   ✅ Response saved to: {response_file.name}")
                    
                    # Move original to completed
                    completed = self.grok_completed / f"processed_{file.name}"
                    file.rename(completed)
                else:
                    print(f"   ❌ Failed to get response")
        else:
            print("\n✅ No pending Grok reviews")
        
        print("\n" + "="*60)
        print("✅ Review processing complete!")

def main():
    if len(sys.argv) < 2:
        print("""
Working LLM Automation - Fixed ChromeDriver Version
====================================================

Usage: python3 llm_working_automation.py [command]

Commands:
  test         - Test both ChatGPT and Grok connections
  process      - Process all pending reviews
  chatgpt "message"  - Send a message to ChatGPT
  grok "message"     - Send a message to Grok

Prerequisites:
  1. Chrome profiles must be set up with logged-in sessions:
     python3 scripts/simple_browser_setup.py setup both
     
This version uses the correct ChromeDriver for your Chrome version.
        """)
        sys.exit(1)
    
    automation = WorkingLLMAutomation()
    command = sys.argv[1]
    
    if command == "test":
        automation.test_connections()
    
    elif command == "process":
        automation.process_all_reviews()
    
    elif command == "chatgpt" and len(sys.argv) > 2:
        message = " ".join(sys.argv[2:])
        response = automation.send_to_chatgpt(message)
        if response:
            print(f"\n📬 Response:\n{response}")
    
    elif command == "grok" and len(sys.argv) > 2:
        message = " ".join(sys.argv[2:])
        response = automation.send_to_grok(message)
        if response:
            print(f"\n📬 Response:\n{response}")

if __name__ == "__main__":
    main()