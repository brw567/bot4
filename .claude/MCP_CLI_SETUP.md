# MCP Setup for Claude CLI (Code)

## Overview
Since you're using Claude Code (CLI), not the desktop application, the MCP (Model Context Protocol) configuration works differently. Claude Code has built-in support for many tools without requiring additional MCP setup.

## Available Tools in Claude Code

Claude Code CLI already provides these tools without additional configuration:

### File Operations
- **Read**: Read any file content
- **Write**: Create new files
- **Edit**: Modify existing files
- **MultiEdit**: Multiple edits in one operation
- **NotebookEdit**: Edit Jupyter notebooks

### Search & Navigation
- **Grep**: Search file contents with regex
- **Glob**: Find files by pattern
- **LS**: List directory contents

### Development
- **Bash**: Execute shell commands
- **BashOutput**: Read background process output
- **KillBash**: Terminate background processes
- **WebFetch**: Fetch and process web content
- **WebSearch**: Search the web

### GitHub Integration (via MCP)
The GitHub MCP tools are already available with prefix `mcp__github__`:
- `mcp__github__create_repository`
- `mcp__github__get_file_contents`
- `mcp__github__push_files`
- `mcp__github__create_pull_request`
- `mcp__github__create_issue`
- And many more...

## Optimizing Claude Code for Bot4

### 1. Project Structure Configuration

Create a `.claude_code` configuration file:

```bash
cat > /home/hamster/bot4/.claude_code << 'EOF'
# Claude Code Project Configuration
project_name: bot4
primary_language: python
framework: fastapi

# Key directories
source_dirs:
  - src/
  - strategies/
  - ml/

test_dirs:
  - tests/unit/
  - tests/integration/
  - tests/backtesting/

# Validation scripts
validation:
  pre_commit:
    - python scripts/validate_no_fakes.py
    - python scripts/check_risk_limits.py
  pre_push:
    - pytest tests/
    - python scripts/test/test_performance.py

# Remote deployment
deployment:
  host: 192.168.100.64
  user: hamster
  path: /home/hamster/bot4_production
  
# Agent responsibilities
agents:
  alex: "Architecture & strategy decisions"
  morgan: "ML & data science validation"
  sam: "Code quality & fake detection"
  quinn: "Risk management & limits"
  jordan: "Performance & deployment"
  casey: "Frontend & user experience"
  riley: "Testing & quality assurance"
  avery: "Security & compliance"
EOF
```

### 2. Useful Claude Code Commands

```bash
# View current working directory
/pwd

# Change working directory
/cd /home/hamster/bot4

# View active background processes
/bashes

# Clear conversation context (start fresh)
/clear

# Get help
/help
```

### 3. Workflow Optimization

#### For Development Tasks:
```bash
# 1. Start with validation
python scripts/validate_no_fakes.py
python scripts/check_risk_limits.py

# 2. Run tests before making changes
pytest tests/unit/ -v

# 3. Make your changes (Claude will help)

# 4. Validate again
./scripts/validate/validate_all.sh

# 5. Deploy when ready
./scripts/deploy/deploy_remote.sh --backup --deploy
```

#### For Remote Operations:
```bash
# Check remote status
ssh hamster@192.168.100.64 'cd /home/hamster/bot4_production && docker ps'

# View remote logs
ssh hamster@192.168.100.64 'docker logs bot4-trading --tail 50'

# Remote health check
curl http://192.168.100.64:8000/health
```

### 4. Claude Code Best Practices

#### Use Parallel Tool Calls:
When you need multiple pieces of information, Claude Code can run tools in parallel:
- Multiple Bash commands
- Multiple file reads
- Combined search operations

#### Effective Search:
```bash
# Instead of using find/grep in Bash, use Claude's tools:
# Good: Use Grep tool
# Bad: bash -c "find . -name '*.py' | xargs grep 'pattern'"
```

#### File Operations:
```bash
# Good: Use Read/Edit tools for file operations
# Bad: bash -c "cat file.py | sed 's/old/new/'"
```

### 5. Project Maintenance with Claude Code

#### Daily Tasks:
```python
# Morning validation routine
tasks = [
    "Check system health",
    "Review overnight logs", 
    "Validate no new fakes",
    "Run quick tests"
]

# Claude can help automate these:
# "Run the morning validation routine"
```

#### Weekly Tasks:
```python
# Weekly review
reviews = [
    "Full test suite",
    "Risk metrics (Quinn)",
    "Performance benchmarks (Jordan)",
    "ML model drift (Morgan)",
    "Security scan (Avery)"
]

# Ask Claude: "Run the weekly review checklist"
```

### 6. GitHub Integration

Since you have GitHub MCP tools available, you can:

```bash
# Create issues directly
# "Create a GitHub issue for the performance regression in the ML pipeline"

# Create pull requests
# "Create a PR for the risk limit updates we just made"

# Search code across repos
# "Search GitHub for similar ATR calculation implementations"
```

### 7. Environment Variables

Store deployment configurations:

```bash
# Create deployment config
cat > ~/.claude_code_env << 'EOF'
# Bot4 Deployment
export BOT3_REMOTE_HOST="192.168.100.64"
export BOT3_REMOTE_USER="hamster"
export BOT3_REMOTE_PATH="/home/hamster/bot4_production"
export BOT3_BACKUP_PATH="/home/hamster/backups"

# Load in new sessions
source ~/.claude_code_env
EOF
```

### 8. Custom Aliases for Common Tasks

```bash
# Add to ~/.bashrc or create project script
cat > /home/hamster/bot4/scripts/aliases.sh << 'EOF'
#!/bin/bash
# Bot4 project aliases

# Validation shortcuts
alias validate-all='python scripts/validate_no_fakes.py && python scripts/check_risk_limits.py'
alias sam-check='python scripts/validate_no_fakes.py'
alias quinn-check='python scripts/check_risk_limits.py'

# Testing shortcuts
alias test-unit='pytest tests/unit/ -v'
alias test-full='pytest tests/ -v'
alias test-perf='bash scripts/test/test_performance.sh'

# Deployment shortcuts
alias deploy-check='./scripts/deploy/deploy_remote.sh --check'
alias deploy-backup='./scripts/deploy/deploy_remote.sh --backup'
alias deploy-full='./scripts/deploy/deploy_remote.sh --backup --deploy'
alias deploy-status='./scripts/deploy/deploy_remote.sh --status'
alias deploy-rollback='./scripts/deploy/deploy_remote.sh --rollback'

# Remote operations
alias remote-logs='ssh hamster@192.168.100.64 "docker logs bot4-trading --tail 100"'
alias remote-status='ssh hamster@192.168.100.64 "docker ps | grep bot4"'
alias remote-health='curl http://192.168.100.64:8000/health'

echo "Bot4 aliases loaded"
EOF

# Source in current session
source scripts/aliases.sh
```

## Quick Reference Card

### Essential Commands:
```bash
# Validation
sam-check                  # Check for fakes
quinn-check               # Check risk limits
validate-all              # Run all validations

# Testing
test-unit                 # Unit tests only
test-full                 # All tests
test-perf                 # Performance benchmarks

# Deployment
deploy-check              # Pre-deployment checks
deploy-full               # Full deployment with backup
deploy-status             # Check deployment status
deploy-rollback           # Emergency rollback

# Remote
remote-logs               # View production logs
remote-status             # Check container status
remote-health             # API health check
```

### Working with Claude Code:
1. **Be specific**: "Fix the fake ATR calculation in src/indicators/atr.py"
2. **Use agents**: "As Sam, check for any fake implementations"
3. **Batch operations**: "Check all strategy files for risk violations"
4. **Validate often**: Run checks after each major change

### Emergency Procedures:
```bash
# If deployment fails
deploy-rollback

# If tests fail
git stash               # Save current changes
git checkout main       # Return to stable
pytest tests/           # Verify stable state

# If remote is down
ssh hamster@192.168.100.64
docker-compose restart
docker logs bot4-trading
```

## Success Metrics

Your Bot4 setup is complete when:
- ✅ Git hooks are working (pre-commit, pre-push)
- ✅ Validation scripts pass
- ✅ Remote deployment succeeds
- ✅ Health checks are green
- ✅ No fake implementations exist
- ✅ Risk limits are enforced

Remember: The team is watching. Build it right the first time.