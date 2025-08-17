#!/usr/bin/env python3
"""
Fixed LLM automation using pure headless Chrome
No X11 display required - uses Chrome profiles with cookies
"""

import subprocess
import time
import json
import sys
from pathlib import Path
from datetime import datetime

class HeadlessLLMAutomation:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.chatgpt_profile = self.base_path / "browser_sessions" / "chrome-chatgpt"
        self.grok_profile = self.base_path / "browser_sessions" / "chrome-grok"
        
        # Create a virtual display for headless operation
        self.virtual_display = None
        
    def start_virtual_display(self):
        """Start Xvfb virtual display"""
        print("🖥️  Starting virtual display...")
        
        # Kill any existing Xvfb on display 99
        subprocess.run(["pkill", "-f", "Xvfb :99"], capture_output=True)
        time.sleep(1)
        
        # Start new Xvfb
        self.virtual_display = subprocess.Popen([
            "Xvfb", ":99",
            "-screen", "0", "1920x1080x24",
            "-ac",  # Disable access control
            "+extension", "GLX",
            "+render",
            "-noreset"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(2)
        print("✅ Virtual display started on :99")
        return ":99"
    
    def stop_virtual_display(self):
        """Stop virtual display"""
        if self.virtual_display:
            self.virtual_display.terminate()
            print("🛑 Virtual display stopped")
    
    def send_to_chatgpt_via_chrome(self, message: str):
        """Send message to ChatGPT using Chrome with saved profile"""
        print("\n🤖 Sending to ChatGPT...")
        
        # Start virtual display
        display = self.start_virtual_display()
        
        try:
            # Create Python script to run in Chrome console
            automation_script = f'''
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Setup Chrome with profile
options = Options()
options.add_argument("--user-data-dir={self.chatgpt_profile}")
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")

# Start Chrome
driver = webdriver.Chrome(options=options)

try:
    # Go to ChatGPT
    driver.get("https://chat.openai.com")
    time.sleep(5)
    
    # Try to start new chat
    try:
        new_chat = driver.find_element(By.CSS_SELECTOR, 'a[href="/"]')
        new_chat.click()
        time.sleep(2)
    except:
        pass
    
    # Find and fill textarea
    textarea = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'textarea[placeholder*="Message"]'))
    )
    textarea.send_keys("""{message}""")
    textarea.send_keys(Keys.RETURN)
    
    # Wait for response
    time.sleep(10)
    
    # Get response
    responses = driver.find_elements(By.CSS_SELECTOR, '[data-message-author-role="assistant"]')
    if responses:
        print("RESPONSE_START")
        print(responses[-1].text)
        print("RESPONSE_END")
    
finally:
    driver.quit()
'''
            
            # Save script temporarily
            script_file = self.base_path / "temp_chatgpt_script.py"
            script_file.write_text(automation_script)
            
            # Run with virtual display
            env = {
                "DISPLAY": display,
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(Path.home())
            }
            
            result = subprocess.run(
                ["python3", str(script_file)],
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )
            
            # Extract response
            if "RESPONSE_START" in result.stdout:
                start = result.stdout.index("RESPONSE_START") + len("RESPONSE_START")
                end = result.stdout.index("RESPONSE_END")
                response = result.stdout[start:end].strip()
                
                # Clean up
                script_file.unlink()
                self.stop_virtual_display()
                
                return response
            else:
                print(f"❌ No response found in output")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                
                # Clean up
                script_file.unlink()
                self.stop_virtual_display()
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.stop_virtual_display()
            return None
    
    def use_chrome_remote_debugging(self):
        """Alternative: Use Chrome DevTools Protocol"""
        print("\n🔧 Using Chrome Remote Debugging Protocol...")
        
        # Start Chrome with remote debugging
        chrome_process = subprocess.Popen([
            "google-chrome",
            "--headless=new",
            "--remote-debugging-port=9222",
            f"--user-data-dir={self.chatgpt_profile}",
            "--no-sandbox",
            "--disable-gpu",
            "https://chat.openai.com"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(5)
        
        try:
            # Use CDP to interact
            import requests
            
            # Get available tabs
            response = requests.get("http://localhost:9222/json")
            tabs = response.json()
            
            for tab in tabs:
                if "chat.openai.com" in tab.get("url", ""):
                    print(f"✅ Found ChatGPT tab: {tab['title']}")
                    
                    # Would need WebSocket connection here for full interaction
                    # For now, just confirm connection works
                    
            chrome_process.terminate()
            return True
            
        except Exception as e:
            print(f"❌ CDP Error: {e}")
            chrome_process.terminate()
            return False
    
    def process_reviews_with_api(self):
        """Process reviews using a different approach"""
        print("\n📋 Processing Reviews (Alternative Method)")
        print("=" * 50)
        
        # Read pending reviews
        chatgpt_pending = self.base_path / "chatgpt_reviews" / "pending"
        grok_pending = self.base_path / "grok_reviews" / "pending"
        
        chatgpt_files = list(chatgpt_pending.glob("*.md"))
        grok_files = list(grok_pending.glob("*.md"))
        
        if chatgpt_files or grok_files:
            print("\n📝 Creating consolidated review request...")
            
            # Create a single file with all reviews
            consolidated = self.base_path / "consolidated_reviews.md"
            
            content = "# Consolidated Review Request\n\n"
            
            if chatgpt_files:
                content += "## For ChatGPT (Sophia):\n\n"
                for file in chatgpt_files:
                    content += file.read_text() + "\n\n---\n\n"
            
            if grok_files:
                content += "## For Grok (Nexus):\n\n"
                for file in grok_files:
                    content += file.read_text() + "\n\n---\n\n"
            
            consolidated.write_text(content)
            print(f"✅ Consolidated reviews saved to: {consolidated}")
            print("\nYou can now:")
            print("1. Copy this file's content")
            print("2. Paste into ChatGPT/Grok manually")
            print("3. Or wait for full automation fix")
        else:
            print("✅ No pending reviews to process")

import os

def test_selenium_setup():
    """Test if Selenium is properly set up"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        print("✅ Selenium is installed")
        
        # Test Chrome driver
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.quit()
            print("✅ Chrome WebDriver is working")
            return True
        except Exception as e:
            print(f"❌ Chrome WebDriver error: {e}")
            print("\nTo fix, run:")
            print("  pip3 install --user selenium webdriver-manager")
            print("  python3 -m webdriver_manager.chrome")
            return False
            
    except ImportError:
        print("❌ Selenium not installed")
        print("\nTo fix, run:")
        print("  pip3 install --user selenium webdriver-manager")
        return False

def main():
    if len(sys.argv) < 2:
        print("""
Fixed LLM Automation - Headless Chrome Solution
================================================

Usage: python3 llm_automation_fixed.py [command]

Commands:
  test         - Test Selenium and Chrome setup
  cdp          - Test Chrome DevTools Protocol
  process      - Process reviews (alternative method)
  chatgpt "message"  - Send message to ChatGPT
  
This script works without X11 display issues.
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    automation = HeadlessLLMAutomation()
    
    if command == "test":
        test_selenium_setup()
    
    elif command == "cdp":
        automation.use_chrome_remote_debugging()
    
    elif command == "process":
        automation.process_reviews_with_api()
    
    elif command == "chatgpt" and len(sys.argv) > 2:
        message = sys.argv[2]
        response = automation.send_to_chatgpt_via_chrome(message)
        if response:
            print(f"\n📬 Response:\n{response}")

if __name__ == "__main__":
    main()