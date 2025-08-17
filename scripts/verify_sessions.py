#!/usr/bin/env python3
"""
Verify if ChatGPT and Grok sessions are actually saved
Opens browser visually to check login status
"""

import subprocess
import time
from pathlib import Path

def verify_chatgpt():
    """Open ChatGPT with saved profile to verify session"""
    print("🔍 Verifying ChatGPT session...")
    
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-chatgpt")
    
    if not profile_dir.exists():
        print("❌ No ChatGPT profile found")
        return False
    
    print("Opening ChatGPT in Chrome...")
    print("Please check if you're logged in.")
    print("Press Ctrl+C to close when done checking.\n")
    
    try:
        subprocess.run([
            "google-chrome",
            f"--user-data-dir={profile_dir}",
            "--no-sandbox",
            "https://chat.openai.com"
        ], env={"DISPLAY": ":0"})
    except KeyboardInterrupt:
        print("\n✅ Check complete")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def verify_grok():
    """Open Grok with saved profile to verify session"""
    print("\n🔍 Verifying Grok session...")
    
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-grok")
    
    if not profile_dir.exists():
        print("❌ No Grok profile found")
        return False
    
    print("Opening Grok in Chrome...")
    print("Please check if you're logged in.")
    print("Press Ctrl+C to close when done checking.\n")
    
    try:
        subprocess.run([
            "google-chrome",
            f"--user-data-dir={profile_dir}",
            "--no-sandbox",
            "https://grok.x.ai"
        ], env={"DISPLAY": ":0"})
    except KeyboardInterrupt:
        print("\n✅ Check complete")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    print("="*60)
    print("📋 Session Verification Tool")
    print("="*60)
    print("\nThis will open browsers to check if sessions are saved.")
    print("You need DISPLAY=:0 or X11 forwarding for this to work.\n")
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "chatgpt":
            verify_chatgpt()
        elif sys.argv[1] == "grok":
            verify_grok()
    else:
        print("Usage: python3 verify_sessions.py [chatgpt|grok]")
        print("\nOr try headless verification...")
        
        # Try headless check
        print("\n🔍 Checking cookies databases...")
        
        chatgpt_cookies = Path("/home/hamster/bot4/browser_sessions/chrome-chatgpt/Default/Cookies")
        grok_cookies = Path("/home/hamster/bot4/browser_sessions/chrome-grok/Default/Cookies")
        
        if chatgpt_cookies.exists():
            size = chatgpt_cookies.stat().st_size
            print(f"✅ ChatGPT cookies found ({size} bytes)")
            if size < 10000:
                print("   ⚠️  Cookies file seems small - may not have session")
        else:
            print("❌ No ChatGPT cookies")
        
        if grok_cookies.exists():
            size = grok_cookies.stat().st_size
            print(f"✅ Grok cookies found ({size} bytes)")
            if size < 10000:
                print("   ⚠️  Cookies file seems small - may not have session")
        else:
            print("❌ No Grok cookies")

if __name__ == "__main__":
    main()