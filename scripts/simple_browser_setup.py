#!/usr/bin/env python3
"""
Simple browser setup for ChatGPT and Grok
Works with XFCE4 and Google Chrome
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def setup_chatgpt():
    """Open ChatGPT in Chrome for manual login"""
    print("\n" + "="*60)
    print("🤖 ChatGPT Setup")
    print("="*60)
    
    # Create dedicated profile directory
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-chatgpt")
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Instructions:")
    print("1. Chrome will open with ChatGPT")
    print("2. Log in to your ChatGPT account")
    print("3. Once logged in, keep the browser open")
    print("4. Return here and press Enter")
    print("\nOpening Chrome...")
    
    # Launch Chrome with specific profile
    cmd = [
        "google-chrome",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://chat.openai.com"
    ]
    
    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    
    process = subprocess.Popen(cmd, env=env)
    
    input("\n✅ Press Enter after logging in to ChatGPT...")
    
    print("✅ ChatGPT profile saved!")
    print(f"   Profile location: {profile_dir}")
    
    # Don't kill the process, let user close it
    return profile_dir

def setup_grok():
    """Open Grok in Chrome for manual login"""
    print("\n" + "="*60)
    print("🧠 Grok Setup")
    print("="*60)
    
    # Create dedicated profile directory
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-grok")
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Instructions:")
    print("1. Chrome will open with Grok")
    print("2. Log in with your X/Twitter account")
    print("3. Once logged in, keep the browser open")
    print("4. Return here and press Enter")
    print("\nOpening Chrome...")
    
    # Launch Chrome with specific profile
    cmd = [
        "google-chrome",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://grok.x.ai"
    ]
    
    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    
    process = subprocess.Popen(cmd, env=env)
    
    input("\n✅ Press Enter after logging in to Grok...")
    
    print("✅ Grok profile saved!")
    print(f"   Profile location: {profile_dir}")
    
    return profile_dir

def test_sessions():
    """Test if sessions are saved"""
    print("\n" + "="*60)
    print("🔍 Testing Saved Sessions")
    print("="*60)
    
    chatgpt_profile = Path("/home/hamster/bot4/browser_sessions/chrome-chatgpt")
    grok_profile = Path("/home/hamster/bot4/browser_sessions/chrome-grok")
    
    if chatgpt_profile.exists():
        print("✅ ChatGPT profile exists")
        # Check for cookies
        cookies = chatgpt_profile / "Default" / "Cookies"
        if cookies.exists():
            print("   ✓ Cookies database found")
    else:
        print("❌ ChatGPT profile not found")
    
    if grok_profile.exists():
        print("✅ Grok profile exists")
        cookies = grok_profile / "Default" / "Cookies"
        if cookies.exists():
            print("   ✓ Cookies database found")
    else:
        print("❌ Grok profile not found")

def open_chatgpt_logged_in():
    """Open ChatGPT with saved profile"""
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-chatgpt")
    if not profile_dir.exists():
        print("❌ No ChatGPT profile found. Run setup first.")
        return
    
    print("Opening ChatGPT with saved session...")
    cmd = [
        "google-chrome",
        f"--user-data-dir={profile_dir}",
        "https://chat.openai.com"
    ]
    
    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    
    subprocess.Popen(cmd, env=env)
    print("✅ Chrome opened with ChatGPT session")

def open_grok_logged_in():
    """Open Grok with saved profile"""
    profile_dir = Path("/home/hamster/bot4/browser_sessions/chrome-grok")
    if not profile_dir.exists():
        print("❌ No Grok profile found. Run setup first.")
        return
    
    print("Opening Grok with saved session...")
    cmd = [
        "google-chrome",
        f"--user-data-dir={profile_dir}",
        "https://grok.x.ai"
    ]
    
    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    
    subprocess.Popen(cmd, env=env)
    print("✅ Chrome opened with Grok session")

def main():
    if len(sys.argv) < 2:
        print("""
Simple Browser Setup for LLMs
==============================

Usage: python3 simple_browser_setup.py [command]

Setup Commands:
  setup chatgpt    - Set up ChatGPT session
  setup grok       - Set up Grok session
  setup both       - Set up both sessions

Test Commands:
  test            - Check if sessions exist
  open chatgpt    - Open ChatGPT with saved session
  open grok       - Open Grok with saved session

After setup, the browser profiles will be saved and can be reused.
The sessions will persist even after reboot.
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "setup" and len(sys.argv) > 2:
        target = sys.argv[2]
        if target == "chatgpt":
            setup_chatgpt()
        elif target == "grok":
            setup_grok()
        elif target == "both":
            setup_chatgpt()
            print("\n" + "-"*40 + "\n")
            setup_grok()
    
    elif command == "test":
        test_sessions()
    
    elif command == "open" and len(sys.argv) > 2:
        target = sys.argv[2]
        if target == "chatgpt":
            open_chatgpt_logged_in()
        elif target == "grok":
            open_grok_logged_in()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()