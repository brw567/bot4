#!/bin/bash
# Daily team synchronization

echo "🔄 Running daily team sync..."
date

# Create standup
python3 /home/hamster/bot4/scripts/team_sync.py standup

# Notify user to sync with external LLMs
echo ""
echo "📋 MANUAL STEPS REQUIRED:"
echo "1. Copy contents of chatgpt_reviews/pending/ to ChatGPT"
echo "2. Copy contents of grok_reviews/pending/ to Grok"
echo "3. Get their responses and save to completed/ folders"
echo "4. Run: python3 scripts/team_sync.py collect"

# Create a notification file for the user
echo "Team sync waiting for external input: $(date)" > /home/hamster/bot4/AWAITING_EXTERNAL_INPUT.txt
