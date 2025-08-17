#!/bin/bash
# Setup script for 10-person team integration

echo "🚀 Setting up Bot4 10-Person Team Integration"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/hamster/bot4"

echo -e "${YELLOW}Creating directory structure...${NC}"

# Create all necessary directories
mkdir -p $BASE_DIR/chatgpt_reviews/{pending,completed}
mkdir -p $BASE_DIR/grok_reviews/{pending,completed}
mkdir -p $BASE_DIR/daily_standups
mkdir -p $BASE_DIR/task_reviews
mkdir -p $BASE_DIR/external_feedback
mkdir -p $BASE_DIR/team_decisions

echo -e "${GREEN}✓ Directories created${NC}"

# Make scripts executable
echo -e "${YELLOW}Setting up scripts...${NC}"
chmod +x $BASE_DIR/scripts/team_sync.py
chmod +x $BASE_DIR/scripts/daily_team_sync.sh

# Create daily sync script if it doesn't exist
if [ ! -f "$BASE_DIR/scripts/daily_team_sync.sh" ]; then
    cat > $BASE_DIR/scripts/daily_team_sync.sh << 'EOF'
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
EOF
    chmod +x $BASE_DIR/scripts/daily_team_sync.sh
fi

echo -e "${GREEN}✓ Scripts configured${NC}"

# Install Python dependencies if needed
echo -e "${YELLOW}Checking Python dependencies...${NC}"
python3 -c "from pathlib import Path; import json" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    pip3 install --user pathlib
fi
echo -e "${GREEN}✓ Dependencies ready${NC}"

# Create initial standup
echo -e "${YELLOW}Creating first daily standup...${NC}"
python3 $BASE_DIR/scripts/team_sync.py standup

echo -e "${GREEN}✓ Initial standup created${NC}"

# Create quick reference guide
cat > $BASE_DIR/TEAM_SYNC_GUIDE.md << 'EOF'
# Team Sync Quick Reference

## Daily Workflow

### Morning (9:00 UTC)
1. Run: `./scripts/daily_team_sync.sh`
2. Copy `chatgpt_reviews/pending/*.md` to ChatGPT project
3. Copy `grok_reviews/pending/*.md` to Grok project
4. Ask both: "Please review and respond to the request"

### After Responses (within 4 hours)
1. Save ChatGPT response to `chatgpt_reviews/completed/`
2. Save Grok response to `grok_reviews/completed/`
3. Run: `python3 scripts/team_sync.py collect`
4. Review consolidated feedback

## Task Review Process
```bash
# Create task review
python3 scripts/team_sync.py task TASK_1.1 "Implement circuit breaker"

# Share with external teams and get feedback
# Then collect responses
python3 scripts/team_sync.py collect
```

## Manual Sync Commands
- Create standup: `python3 scripts/team_sync.py standup`
- Collect feedback: `python3 scripts/team_sync.py collect`
- Create task review: `python3 scripts/team_sync.py task [ID] "[Description]"`

## Directory Structure
```
bot4/
├── chatgpt_reviews/
│   ├── pending/      # Requests for Sophia
│   └── completed/    # Responses from Sophia
├── grok_reviews/
│   ├── pending/      # Requests for Nexus
│   └── completed/    # Responses from Nexus
├── daily_standups/   # Daily team syncs
├── task_reviews/     # Task-specific reviews
└── external_feedback/# Consolidated feedback
```
EOF

echo -e "${GREEN}✓ Team sync guide created${NC}"

echo ""
echo "========================================="
echo -e "${GREEN}✅ Integration Setup Complete!${NC}"
echo "========================================="
echo ""
echo "📋 NEXT STEPS:"
echo "1. Review the files in chatgpt_reviews/pending/"
echo "2. Copy review request to ChatGPT (Sophia)"
echo "3. Copy validation request to Grok (Nexus)"
echo "4. Get their responses and save to completed/ folders"
echo "5. Run: python3 scripts/team_sync.py collect"
echo ""
echo "📖 See TEAM_SYNC_GUIDE.md for daily workflow"
echo ""