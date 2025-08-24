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
