#!/usr/bin/env python3
"""
Capture existing Chrome browser sessions for ChatGPT and Grok
Works with already logged-in Chrome sessions
"""

import json
import os
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

class ChromeSessionCapture:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.sessions_path = self.base_path / "browser_sessions"
        self.sessions_path.mkdir(exist_ok=True)
        
        # Find Chrome profile directory
        self.chrome_profiles = [
            Path.home() / ".config/google-chrome",
            Path.home() / ".config/chromium",
            Path("/home/hamster/.config/google-chrome"),
            Path("/home/hamster/.config/chromium")
        ]
        
        self.chrome_dir = None
        for profile_dir in self.chrome_profiles:
            if profile_dir.exists():
                self.chrome_dir = profile_dir
                print(f"✓ Found Chrome profile at: {self.chrome_dir}")
                break
    
    def find_chrome_cookies(self):
        """Find and copy Chrome cookies database"""
        if not self.chrome_dir:
            print("❌ Chrome profile directory not found!")
            return None
        
        # Look for Default profile cookies
        cookies_db = self.chrome_dir / "Default" / "Cookies"
        
        if not cookies_db.exists():
            print(f"❌ Cookies database not found at {cookies_db}")
            return None
        
        # Copy cookies database (Chrome locks it)
        temp_db = self.sessions_path / "cookies_temp.db"
        shutil.copy2(cookies_db, temp_db)
        print(f"✓ Copied cookies database to {temp_db}")
        
        return temp_db
    
    def extract_cookies_for_domain(self, db_path: Path, domain: str):
        """Extract cookies for specific domain from Chrome database"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Query cookies for domain
            query = """
            SELECT name, value, host_key, path, expires_utc, is_secure, is_httponly, samesite
            FROM cookies
            WHERE host_key LIKE ?
            """
            
            cursor.execute(query, (f"%{domain}%",))
            cookies = []
            
            for row in cursor.fetchall():
                name, value, host, path, expires, secure, httponly, samesite = row
                
                # Decrypt value if needed (Chrome encrypts cookies)
                # For now, we'll use the encrypted value
                cookie = {
                    "name": name,
                    "value": value,  # This might be encrypted
                    "domain": host,
                    "path": path,
                    "secure": bool(secure),
                    "httpOnly": bool(httponly),
                    "sameSite": ["none", "lax", "strict"][samesite] if samesite < 3 else "lax"
                }
                cookies.append(cookie)
                print(f"  Found cookie: {name} for {host}")
            
            conn.close()
            return cookies
            
        except Exception as e:
            print(f"❌ Error reading cookies: {e}")
            return []
    
    def use_chrome_devtools(self):
        """Alternative: Use Chrome DevTools Protocol to get cookies"""
        print("\n🔧 Using Chrome DevTools Protocol...")
        
        try:
            # Start Chrome with debugging port
            chrome_cmd = [
                "google-chrome",
                "--remote-debugging-port=9222",
                "--user-data-dir=" + str(self.chrome_dir),
                "--headless=new"
            ]
            
            # Launch Chrome in background
            process = subprocess.Popen(
                chrome_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            import time
            time.sleep(3)  # Wait for Chrome to start
            
            # Use CDP to get cookies
            import requests
            
            # Get list of pages
            response = requests.get("http://localhost:9222/json")
            pages = response.json()
            
            for page in pages:
                if "chat.openai.com" in page.get("url", ""):
                    print(f"Found ChatGPT page: {page['url']}")
                    # Get cookies via CDP
                    ws_url = page["webSocketDebuggerUrl"]
                    # Would need websocket client here
                    
            process.terminate()
            
        except Exception as e:
            print(f"❌ DevTools method failed: {e}")
    
    def save_manual_instructions(self):
        """Save instructions for manual cookie export"""
        instructions = """
# Manual Cookie Export Instructions

Since Chrome encrypts cookies, the easiest way is to export them manually:

## For ChatGPT:
1. Open https://chat.openai.com in Chrome
2. Press F12 to open Developer Tools
3. Go to Console tab
4. Paste this code:

```javascript
// Get all cookies for current domain
const getCookies = () => {
  const cookies = [];
  document.cookie.split(';').forEach(cookie => {
    const [name, value] = cookie.trim().split('=');
    if (name && value) {
      cookies.push({
        name: name,
        value: decodeURIComponent(value),
        domain: '.chat.openai.com',
        path: '/',
        secure: true,
        httpOnly: false,
        sameSite: 'Lax'
      });
    }
  });
  console.log(JSON.stringify(cookies, null, 2));
};
getCookies();
```

5. Copy the JSON output
6. Save to: /home/hamster/bot4/browser_sessions/chatgpt_cookies.json

## For Grok:
1. Open https://grok.x.ai in Chrome
2. Repeat the same process
3. Save to: /home/hamster/bot4/browser_sessions/grok_cookies.json

## Alternative: Browser Extension
Install "EditThisCookie" extension and export cookies as JSON.
"""
        
        instructions_file = self.sessions_path / "MANUAL_EXPORT_INSTRUCTIONS.md"
        instructions_file.write_text(instructions)
        print(f"\n📝 Manual instructions saved to: {instructions_file}")
        print("\nPlease follow the instructions in that file to export cookies manually.")
    
    def test_playwright_attach(self):
        """Try to attach to existing Chrome session"""
        print("\n🔗 Attempting to attach to existing Chrome session...")
        
        # Check if Chrome is running with remote debugging
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        if "--remote-debugging-port" in result.stdout:
            print("✓ Chrome is running with debugging port")
        else:
            print("✗ Chrome not running with debugging port")
            print("\nTo enable, restart Chrome with:")
            print("  google-chrome --remote-debugging-port=9222")

def main():
    print("=" * 60)
    print("🍪 Chrome Session Capture Tool")
    print("=" * 60)
    
    capture = ChromeSessionCapture()
    
    # Try to find Chrome cookies
    cookies_db = capture.find_chrome_cookies()
    
    if cookies_db:
        print("\n📊 Extracting cookies...")
        
        # Extract ChatGPT cookies
        chatgpt_cookies = capture.extract_cookies_for_domain(cookies_db, "chat.openai.com")
        if chatgpt_cookies:
            output_file = capture.sessions_path / "chatgpt_cookies_raw.json"
            with open(output_file, 'w') as f:
                json.dump(chatgpt_cookies, f, indent=2)
            print(f"✓ Saved {len(chatgpt_cookies)} ChatGPT cookies to {output_file}")
        
        # Extract Grok cookies
        grok_cookies = capture.extract_cookies_for_domain(cookies_db, "grok.x.ai")
        if grok_cookies:
            output_file = capture.sessions_path / "grok_cookies_raw.json"
            with open(output_file, 'w') as f:
                json.dump(grok_cookies, f, indent=2)
            print(f"✓ Saved {len(grok_cookies)} Grok cookies to {output_file}")
        
        # Clean up temp database
        cookies_db.unlink()
    
    # Save manual instructions
    capture.save_manual_instructions()
    
    # Test other methods
    capture.test_playwright_attach()
    
    print("\n" + "=" * 60)
    print("⚠️  Note: Chrome encrypts cookie values in its database.")
    print("    The exported cookies may have encrypted values.")
    print("    Please use the manual export method for best results.")
    print("=" * 60)

if __name__ == "__main__":
    main()