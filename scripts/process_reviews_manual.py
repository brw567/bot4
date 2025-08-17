#!/usr/bin/env python3
"""
Manual review processing instructions
Since browser automation has display issues, this provides manual steps
"""

import json
from pathlib import Path
from datetime import datetime

def generate_manual_instructions():
    """Generate instructions for manual review processing"""
    
    base_path = Path("/home/hamster/bot4")
    
    # Check for pending reviews
    chatgpt_pending = base_path / "chatgpt_reviews" / "pending"
    grok_pending = base_path / "grok_reviews" / "pending"
    
    chatgpt_files = list(chatgpt_pending.glob("*.md"))
    grok_files = list(grok_pending.glob("*.md"))
    
    print("=" * 70)
    print("📋 MANUAL REVIEW PROCESSING INSTRUCTIONS")
    print("=" * 70)
    
    if chatgpt_files:
        print("\n🤖 ChatGPT Reviews to Process:")
        print("-" * 40)
        for file in chatgpt_files:
            print(f"\n📄 File: {file.name}")
            content = file.read_text()
            print(f"Preview:\n{content[:200]}...")
            print("\nTo process this review:")
            print("1. Open ChatGPT in your browser")
            print("2. Copy the content from the file above")
            print("3. Paste it into ChatGPT")
            print("4. Save the response to:")
            print(f"   /home/hamster/bot4/chatgpt_reviews/completed/sophia_response_{datetime.now().strftime('%Y%m%d')}.md")
    
    if grok_files:
        print("\n🧠 Grok Reviews to Process:")
        print("-" * 40)
        for file in grok_files:
            print(f"\n📄 File: {file.name}")
            content = file.read_text()
            print(f"Preview:\n{content[:200]}...")
            print("\nTo process this review:")
            print("1. Open Grok in your browser")
            print("2. Copy the content from the file above")
            print("3. Paste it into Grok")
            print("4. Save the response to:")
            print(f"   /home/hamster/bot4/grok_reviews/completed/nexus_response_{datetime.now().strftime('%Y%m%d')}.md")
    
    print("\n" + "=" * 70)
    print("💡 Alternative: Direct Browser Access")
    print("=" * 70)
    print("""
Since you have XFCE4 and Chrome set up, you can:

1. Connect to your desktop session (VNC or RDP)
2. Open Chrome manually
3. Process the reviews directly
4. Save responses to the completed folders

Or use SSH with X11 forwarding:
  ssh -X hamster@localhost
  DISPLAY=:0 google-chrome
""")

def create_team_response_template():
    """Create a template for team responses"""
    
    template = """# Team Response Integration
## Date: {date}

### Sophia (ChatGPT) - Architecture Auditor
{chatgpt_response}

### Nexus (Grok) - Performance Validator  
{grok_response}

### Team Consensus
Based on external review:
- Architecture: [APPROVED/NEEDS_REVISION]
- Performance: [VALIDATED/NEEDS_OPTIMIZATION]
- Action Items:
  1. [Item from reviews]
  2. [Item from reviews]

### Next Steps
- [ ] Implement feedback
- [ ] Update architecture docs
- [ ] Run performance tests
"""
    
    template_file = Path("/home/hamster/bot4/docs/REVIEW_RESPONSE_TEMPLATE.md")
    template_file.parent.mkdir(exist_ok=True)
    template_file.write_text(template)
    print(f"\n✅ Response template saved to: {template_file}")

if __name__ == "__main__":
    generate_manual_instructions()
    create_team_response_template()
    
    print("\n" + "=" * 70)
    print("📌 Summary")
    print("=" * 70)
    print("""
The automated browser integration is having display issues.
Please use one of these approaches:

1. Manual processing (copy/paste in browser)
2. VNC/RDP to desktop session
3. SSH with X11 forwarding
4. Wait for me to fix the headless automation

The review files are ready and waiting in:
- /home/hamster/bot4/chatgpt_reviews/pending/
- /home/hamster/bot4/grok_reviews/pending/
""")