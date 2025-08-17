#!/usr/bin/env python3
"""
Manual setup helper for ChatGPT and Grok sessions
Provides instructions for manual cookie collection
"""

import json
import sys
from pathlib import Path
from datetime import datetime

class ManualLLMSetup:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.sessions_path = self.base_path / "browser_sessions"
        self.sessions_path.mkdir(exist_ok=True)
        
        self.chatgpt_cookies = self.sessions_path / "chatgpt_cookies.json"
        self.grok_cookies = self.sessions_path / "grok_cookies.json"
    
    def setup_chatgpt_manual(self):
        """Guide user through manual ChatGPT setup"""
        print("\n" + "="*60)
        print("🔐 ChatGPT Manual Session Setup")
        print("="*60)
        print("\nSince we're on a headless server, please follow these steps:")
        print("\n1. Open ChatGPT in your LOCAL browser:")
        print("   https://chat.openai.com")
        print("\n2. Log in to your account")
        print("\n3. Open browser developer tools (F12)")
        print("\n4. Go to the 'Application' or 'Storage' tab")
        print("\n5. Find 'Cookies' → 'https://chat.openai.com'")
        print("\n6. You need these specific cookies:")
        print("   - __Secure-next-auth.session-token")
        print("   - cf_clearance (if present)")
        print("\n7. Create a JSON file with this structure:")
        print("""
{
  "cookies": [
    {
      "name": "__Secure-next-auth.session-token",
      "value": "YOUR_SESSION_TOKEN_HERE",
      "domain": ".chat.openai.com",
      "path": "/",
      "secure": true,
      "httpOnly": true,
      "sameSite": "Lax"
    }
  ]
}
""")
        print("\n8. Save this to: /home/hamster/bot4/browser_sessions/chatgpt_cookies.json")
        print("\nAlternatively, use the browser export method below...")
        
        self.show_browser_export_method()
    
    def setup_grok_manual(self):
        """Guide user through manual Grok setup"""
        print("\n" + "="*60)
        print("🔐 Grok Manual Session Setup")
        print("="*60)
        print("\nSince we're on a headless server, please follow these steps:")
        print("\n1. Open Grok in your LOCAL browser:")
        print("   https://grok.x.ai")
        print("\n2. Log in with your X/Twitter account")
        print("\n3. Open browser developer tools (F12)")
        print("\n4. Go to the 'Application' or 'Storage' tab")
        print("\n5. Find 'Cookies' → 'https://grok.x.ai'")
        print("\n6. Export the auth cookies")
        print("\n7. Save to: /home/hamster/bot4/browser_sessions/grok_cookies.json")
        
        self.show_browser_export_method()
    
    def show_browser_export_method(self):
        """Show browser console method to export cookies"""
        print("\n" + "-"*40)
        print("🌐 Browser Console Export Method:")
        print("-"*40)
        print("\nPaste this in your browser console to export all cookies:")
        print("""
// Copy this entire script to browser console:
(() => {
  const cookies = document.cookie.split(';').map(c => {
    const [name, value] = c.trim().split('=');
    return {
      name: name,
      value: decodeURIComponent(value),
      domain: window.location.hostname,
      path: '/',
      secure: true,
      httpOnly: false,
      sameSite: 'Lax'
    };
  });
  
  // Get all cookies via Chrome API if available
  if (chrome && chrome.cookies) {
    chrome.cookies.getAll({domain: window.location.hostname}, (allCookies) => {
      console.log(JSON.stringify(allCookies, null, 2));
    });
  } else {
    console.log(JSON.stringify(cookies, null, 2));
  }
})();
""")
        print("\nThen save the output to the appropriate JSON file.")
    
    def save_cookie_string(self, platform: str, cookie_string: str):
        """Save cookie string from user input"""
        try:
            # Parse the cookie string
            cookies = []
            for cookie in cookie_string.split(';'):
                cookie = cookie.strip()
                if '=' in cookie:
                    name, value = cookie.split('=', 1)
                    cookies.append({
                        "name": name.strip(),
                        "value": value.strip(),
                        "domain": f".{platform}.com" if platform == "chatgpt" else ".x.ai",
                        "path": "/",
                        "secure": True,
                        "httpOnly": True,
                        "sameSite": "Lax"
                    })
            
            # Save to file
            output_file = self.chatgpt_cookies if platform == "chatgpt" else self.grok_cookies
            with open(output_file, 'w') as f:
                json.dump(cookies, f, indent=2)
            
            print(f"✅ Cookies saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving cookies: {e}")
            return False
    
    def test_saved_session(self, platform: str):
        """Test if saved session exists"""
        cookie_file = self.chatgpt_cookies if platform == "chatgpt" else self.grok_cookies
        
        if cookie_file.exists():
            print(f"✅ Session file exists: {cookie_file}")
            with open(cookie_file, 'r') as f:
                data = json.load(f)
            print(f"   Found {len(data) if isinstance(data, list) else len(data.get('cookies', []))} cookies")
            return True
        else:
            print(f"❌ No session file found: {cookie_file}")
            return False

def main():
    if len(sys.argv) < 2:
        print("""
Manual LLM Session Setup Helper
================================

Since you're on a headless server, use these commands:

Usage: python3 llm_manual_setup.py [command]

Commands:
  setup chatgpt    - Show instructions for ChatGPT setup
  setup grok       - Show instructions for Grok setup
  test chatgpt     - Check if ChatGPT session exists
  test grok        - Check if Grok session exists
  save chatgpt "cookie_string"  - Save ChatGPT cookies
  save grok "cookie_string"     - Save Grok cookies

Alternative Solution:
--------------------
If you have access to a desktop with Chrome:
1. Install Chrome Remote Desktop
2. Connect to this server via Chrome Remote Desktop
3. Run the browser setup scripts with GUI access

Or use SSH with X11 forwarding:
  ssh -X user@server
  python3 scripts/llm_browser_bridge.py setup chatgpt
        """)
        sys.exit(1)
    
    helper = ManualLLMSetup()
    command = sys.argv[1]
    
    if command == "setup" and len(sys.argv) > 2:
        platform = sys.argv[2]
        if platform == "chatgpt":
            helper.setup_chatgpt_manual()
        elif platform == "grok":
            helper.setup_grok_manual()
    
    elif command == "test" and len(sys.argv) > 2:
        platform = sys.argv[2]
        helper.test_saved_session(platform)
    
    elif command == "save" and len(sys.argv) > 3:
        platform = sys.argv[2]
        cookie_string = sys.argv[3]
        helper.save_cookie_string(platform, cookie_string)

if __name__ == "__main__":
    main()