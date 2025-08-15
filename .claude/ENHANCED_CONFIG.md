# Claude Enhanced Configuration Guide for Bot4

## Part 1: MCP (Model Context Protocol) Configuration

### What is MCP?
MCP allows Claude to interact with external tools and services directly, enabling features like:
- Direct database queries
- File system operations beyond current directory
- API integrations
- Remote server management

### Step-by-Step MCP Setup

#### Step 1: Install MCP Server
```bash
# On your local machine
npm install -g @anthropic/mcp-server

# Or using Docker
docker pull anthropic/mcp-server:latest
```

#### Step 2: Configure MCP Services
Create `.mcp/config.json` in your home directory:

```json
{
  "version": "1.0",
  "services": {
    "github": {
      "type": "github",
      "config": {
        "token": "${GITHUB_TOKEN}",
        "owner": "brw567",
        "repo": "bot4"
      }
    },
    "database": {
      "type": "postgresql",
      "config": {
        "host": "localhost",
        "port": 5432,
        "database": "bot4trading",
        "user": "${DB_USER}",
        "password": "${DB_PASSWORD}"
      }
    },
    "remote_docker": {
      "type": "ssh",
      "config": {
        "host": "192.168.100.64",
        "user": "hamster",
        "keyPath": "~/.ssh/id_rsa",
        "dockerEnabled": true
      }
    },
    "monitoring": {
      "type": "prometheus",
      "config": {
        "url": "http://192.168.100.64:9090",
        "grafanaUrl": "http://192.168.100.64:3000"
      }
    }
  }
}
```

#### Step 3: Enable MCP in Claude
1. Open Claude Desktop settings
2. Go to "Developer" tab
3. Enable "Model Context Protocol"
4. Add configuration path: `~/.mcp/config.json`
5. Restart Claude Desktop

#### Step 4: Test MCP Connection
```bash
# In Claude, you can now use:
"Check GitHub issues in bot4 repository"
"Query PostgreSQL: SELECT * FROM trades WHERE profit > 100"
"Deploy to remote Docker at 192.168.100.64"
"Check Prometheus metrics for last hour"
```

## Part 2: Enhanced Project Configuration

### Create `.claude/project_config.yaml`

```yaml
# Bot4 Enhanced Project Configuration
version: 2.0
project: bot4
type: crypto_trading_platform

# Project Structure Enforcement
structure:
  src:
    - core/        # Core trading logic
    - strategies/  # Trading strategies
    - indicators/  # TA indicators
    - ml/         # Machine learning
    - utils/      # Utilities
  
  scripts:
    location: scripts/
    naming: snake_case
    types:
      - deploy_*.sh     # Deployment scripts
      - fix_*.sh        # Fix scripts
      - test_*.sh       # Testing scripts
      - backup_*.sh     # Backup scripts
  
  config:
    - .env.example
    - config.yaml
    - strategies.json
  
  deployment:
    - docker/
    - kubernetes/
    - ansible/

# Git Configuration
git:
  commit_format: |
    <type>(<scope>): <subject>
    
    <body>
    
    <footer>
  
  types:
    - feat: New feature
    - fix: Bug fix
    - docs: Documentation
    - style: Formatting
    - refactor: Code restructuring
    - test: Testing
    - chore: Maintenance
    - perf: Performance
  
  branch_naming:
    - feature/<description>
    - bugfix/<issue-number>
    - hotfix/<description>
    - release/<version>
  
  protected_branches:
    - main
    - production
  
  hooks:
    pre_commit:
      - pytest tests/
      - black src/
      - flake8 src/
    
    pre_push:
      - python scripts/validate_no_fakes.py
      - python scripts/check_risk_limits.py

# Deployment Configuration
deployment:
  remote_host: 192.168.100.64
  remote_user: hamster
  remote_path: /home/hamster/bot4_production
  
  stages:
    development:
      host: localhost
      port: 8000
      database: bot4_dev
    
    staging:
      host: 192.168.100.64
      port: 8001
      database: bot4_staging
    
    production:
      host: 192.168.100.64
      port: 8000
      database: bot4_prod
  
  checklist:
    - Run all tests
    - Check risk limits
    - Validate no fake implementations
    - Backup database
    - Tag release
    - Deploy to staging
    - Run smoke tests
    - Deploy to production
    - Monitor for 1 hour

# Testing Configuration
testing:
  unit_tests:
    location: tests/unit/
    coverage_threshold: 80%
    
  integration_tests:
    location: tests/integration/
    required_services:
      - postgresql
      - redis
      - mock_exchange
  
  backtesting:
    data_path: data/historical/
    strategies: all
    metrics:
      - sharpe_ratio
      - max_drawdown
      - win_rate
      - profit_factor
  
  paper_trading:
    duration: 7_days
    initial_capital: 10000
    max_loss: 500

# Monitoring Configuration
monitoring:
  metrics:
    - latency_p99 < 100ms
    - error_rate < 0.1%
    - uptime > 99.9%
    - memory_usage < 80%
    - cpu_usage < 70%
  
  alerts:
    - profit_loss > 1000
    - drawdown > 10%
    - error_spike > 5x_normal
    - latency > 200ms
  
  dashboards:
    - trading_performance
    - system_health
    - risk_metrics
    - ml_model_performance

# Agent Specific Rules
agent_rules:
  sam:
    enforce:
      - no_fake_implementations
      - mathematical_correctness
      - backtest_required
    
  morgan:
    enforce:
      - no_overfitting
      - cross_validation
      - feature_importance
    
  quinn:
    enforce:
      - position_limits
      - stop_losses
      - risk_metrics
    
  jordan:
    enforce:
      - performance_benchmarks
      - monitoring_coverage
      - deployment_automation

# Maintenance Windows
maintenance:
  schedule:
    - sunday: 02:00-04:00 UTC
    - wednesday: 02:00-03:00 UTC
  
  tasks:
    - database_vacuum
    - log_rotation
    - cache_cleanup
    - metric_aggregation
```

## Part 3: Remote Deployment Optimization

### Create `.claude/deployment_tools.md`

```markdown
# Deployment Tools & Commands

## Quick Deploy Commands

### 1. Fast Deploy to Remote
```bash
# One-line deploy
./scripts/deploy_remote.sh --fast

# With backup
./scripts/deploy_remote.sh --backup --deploy

# Rollback
./scripts/deploy_remote.sh --rollback
```

### 2. Remote Docker Management
```bash
# Check remote status
ssh hamster@192.168.100.64 'docker ps'

# View logs
ssh hamster@192.168.100.64 'docker logs -f bot4-trading'

# Restart service
ssh hamster@192.168.100.64 'docker-compose restart trading-engine'
```

### 3. Database Operations
```bash
# Backup remote database
./scripts/backup_db.sh --remote

# Restore database
./scripts/restore_db.sh --file backup.sql

# Migration
./scripts/migrate_db.sh --version latest
```

## Automated Deployment Script

Create `scripts/deploy_remote.sh`:

```bash
#!/bin/bash
# Enhanced Remote Deployment Script

set -e

# Configuration
REMOTE_HOST="192.168.100.64"
REMOTE_USER="hamster"
REMOTE_PATH="/home/hamster/bot4_production"
BACKUP_PATH="/home/hamster/backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Pre-deployment checks
pre_deploy_checks() {
    log_info "Running pre-deployment checks..."
    
    # Run tests
    pytest tests/ || { log_error "Tests failed"; exit 1; }
    
    # Check for fake implementations
    python scripts/validate_no_fakes.py || { log_error "Fake implementations found"; exit 1; }
    
    # Check risk limits
    python scripts/check_risk_limits.py || { log_error "Risk limits violated"; exit 1; }
    
    log_info "Pre-deployment checks passed"
}

# Backup current deployment
backup_current() {
    log_info "Backing up current deployment..."
    
    ssh $REMOTE_USER@$REMOTE_HOST "
        cd $REMOTE_PATH
        docker-compose down
        tar -czf $BACKUP_PATH/bot4_$(date +%Y%m%d_%H%M%S).tar.gz .
    "
    
    log_info "Backup completed"
}

# Deploy new version
deploy() {
    log_info "Deploying new version..."
    
    # Build Docker images
    docker build -t bot4-trading:latest .
    docker save bot4-trading:latest | gzip > bot4-trading.tar.gz
    
    # Transfer to remote
    scp bot4-trading.tar.gz $REMOTE_USER@$REMOTE_HOST:/tmp/
    
    # Deploy on remote
    ssh $REMOTE_USER@$REMOTE_HOST "
        cd $REMOTE_PATH
        docker load < /tmp/bot4-trading.tar.gz
        docker-compose up -d
        rm /tmp/bot4-trading.tar.gz
    "
    
    log_info "Deployment completed"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    sleep 10
    
    # Check if services are running
    ssh $REMOTE_USER@$REMOTE_HOST "
        docker ps | grep bot4-trading || exit 1
    "
    
    # Check API health
    curl -f http://$REMOTE_HOST:8000/health || { log_error "API health check failed"; exit 1; }
    
    log_info "Health checks passed"
}

# Main execution
main() {
    case "$1" in
        --fast)
            deploy
            health_check
            ;;
        --backup)
            backup_current
            if [ "$2" == "--deploy" ]; then
                deploy
                health_check
            fi
            ;;
        --rollback)
            log_info "Rolling back to previous version..."
            ssh $REMOTE_USER@$REMOTE_HOST "
                cd $BACKUP_PATH
                latest_backup=\$(ls -t bot4_*.tar.gz | head -1)
                cd $REMOTE_PATH
                docker-compose down
                tar -xzf $BACKUP_PATH/\$latest_backup
                docker-compose up -d
            "
            ;;
        *)
            pre_deploy_checks
            backup_current
            deploy
            health_check
            log_info "Deployment successful!"
            ;;
    esac
}

main "$@"
```

## Part 4: Project Maintenance Standards

### Create `.claude/maintenance_guide.md`

```markdown
# Project Maintenance Guide

## Directory Structure Standards

### Where Things Go

```
bot4/
├── src/                 # Source code
│   ├── core/           # Core trading logic
│   ├── strategies/     # Trading strategies
│   ├── indicators/     # TA indicators
│   ├── ml/            # ML models
│   └── utils/         # Utilities
├── scripts/            # All scripts
│   ├── deploy/        # Deployment scripts
│   ├── fix/           # Fix scripts
│   ├── test/          # Test scripts
│   └── maintenance/   # Maintenance scripts
├── tests/              # All tests
│   ├── unit/
│   ├── integration/
│   └── backtesting/
├── config/             # Configuration files
├── deployment/         # Deployment configs
│   ├── docker/
│   ├── kubernetes/
│   └── ansible/
├── docs/               # Documentation
│   ├── api/
│   ├── architecture/
│   └── guides/
└── data/               # Data files
    ├── historical/
    └── models/
```

## Git Commit Standards

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Examples
```bash
# Feature
git commit -m "feat(strategy): add mean reversion strategy

- Implemented entry/exit logic
- Added risk management
- Backtested with 2 years data

Sharpe: 2.1, Max DD: 12%"

# Fix
git commit -m "fix(indicator): correct ATR calculation

- Fixed period calculation
- Added proper true range formula
- Matches ta library output now

Closes #123"

# Performance
git commit -m "perf(ml): optimize feature engineering

- Reduced computation time by 40%
- Cached frequently used features
- Parallel processing for large datasets"
```

## Fix Scripts Organization

### Location: `scripts/fix/`

```bash
scripts/fix/
├── fix_database.sh         # Database fixes
├── fix_dependencies.sh      # Dependency issues
├── fix_docker.sh           # Docker problems
├── fix_permissions.sh      # Permission issues
├── fix_websocket.sh        # WebSocket reconnection
└── fix_all.sh             # Run all fixes
```

### Template: `scripts/fix/fix_template.sh`
```bash
#!/bin/bash
# Fix Script Template

SCRIPT_NAME="fix_something"
DESCRIPTION="Fixes specific issue"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Running $SCRIPT_NAME${NC}"
echo "Description: $DESCRIPTION"

# Fix logic here
fix_issue() {
    # Implementation
    echo "Fixing..."
}

# Verification
verify_fix() {
    # Check if fixed
    echo "Verifying..."
}

# Execute
fix_issue && verify_fix && echo -e "${GREEN}Fixed!${NC}" || echo -e "${RED}Failed!${NC}"
```

## Weekly Maintenance Tasks

### Monday - Code Quality
```bash
# Sam's domain
- Review for fake implementations
- Update TA indicators
- Backtest all strategies
```

### Wednesday - Risk Review
```bash
# Quinn's domain
- Check position limits
- Review stop losses
- Analyze drawdowns
```

### Friday - Innovation Day
```bash
# Morgan's domain
- Experiment with new models
- Test new features
- Research improvements
```

### Sunday - System Maintenance
```bash
# Jordan's domain
- Database optimization
- Log rotation
- Performance review
- Security updates
```

## Automated Checks

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run tests
pytest tests/unit/ || exit 1

# Check code quality
black --check src/ || exit 1
flake8 src/ || exit 1

# Check for fake implementations
grep -r "price \* 0.02" src/ && echo "Fake ATR found!" && exit 1

# Check risk limits
python scripts/check_risk_limits.py || exit 1

echo "Pre-commit checks passed"
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: pytest
      - name: Check Fakes
        run: python scripts/validate_no_fakes.py
      - name: Backtest
        run: python scripts/run_backtest.py
```

## Performance Monitoring

### Key Metrics to Track
```python
# scripts/monitor_performance.py
METRICS = {
    'latency_p99': {'threshold': 100, 'unit': 'ms'},
    'memory_usage': {'threshold': 80, 'unit': '%'},
    'error_rate': {'threshold': 0.1, 'unit': '%'},
    'profit_daily': {'threshold': 1, 'unit': '%'},
    'sharpe_ratio': {'threshold': 2.0, 'unit': 'ratio'},
    'max_drawdown': {'threshold': 15, 'unit': '%'}
}
```
```

## Part 5: Agent-Specific Instructions

### Enhanced Agent Instructions

```yaml
# .claude/agent_enhancements.yaml

alex:
  additional_responsibilities:
    - Review all deployment decisions
    - Maintain decision_log.md
    - Weekly architecture review
    - Conflict resolution within 3 rounds

sam:
  validation_scripts:
    - scripts/validate_no_fakes.py
    - scripts/check_math_correctness.py
    - scripts/run_backtests.py
  
  auto_reject:
    - "price * 0.02"  # Fake ATR
    - "random.choice"  # Fake ML
    - "return 0"      # Placeholder

morgan:
  ml_standards:
    - train_test_split: 70/20/10
    - cross_validation: 5-fold
    - max_overfit_gap: 5%
    
  required_metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - confusion_matrix

quinn:
  risk_limits:
    - max_position: 2%
    - max_daily_loss: 5%
    - max_leverage: 3x
    - correlation_limit: 0.7
  
  veto_conditions:
    - no_stop_loss
    - uncapped_risk
    - excessive_leverage

jordan:
  deployment_checklist:
    - tests_pass: true
    - docker_builds: true
    - health_checks: true
    - rollback_ready: true
    - monitoring_active: true
  
  performance_requirements:
    - api_latency: <100ms
    - websocket_latency: <50ms
    - database_query: <10ms
    - uptime: >99.9%
```

## Part 6: Quick Reference Commands

### For Claude to Use

```bash
# Deploy to remote
"Jordan, deploy to production using scripts/deploy_remote.sh"

# Fix issues
"Jordan, run scripts/fix/fix_docker.sh on remote"

# Check metrics
"Check Prometheus metrics: latency_p99, error_rate"

# Database operations
"Query: SELECT * FROM trades WHERE date > '2024-01-01'"

# Git operations
"Commit with type 'fix' scope 'indicator'"

# Run backtests
"Sam, backtest all strategies for last 6 months"

# Check for fakes
"Sam, run scripts/validate_no_fakes.py"
```

## Part 7: MCP-Enabled Commands

With MCP configured, you can now use:

```python
# Direct database queries
"SELECT AVG(profit) FROM trades WHERE strategy='arbitrage'"

# GitHub operations  
"Create issue: 'Implement stop loss for grid strategy'"
"Check PR #45 status"

# Remote Docker management
"Deploy bot4-trading:v2.1 to 192.168.100.64"
"Check Docker logs on remote host"

# Monitoring
"Get Prometheus metrics for last hour"
"Show Grafana dashboard for trading performance"
```

## Part 8: Productivity Boosters

### 1. Aliases for Common Operations

```bash
# .claude/aliases.sh
alias deploy="./scripts/deploy_remote.sh"
alias test="pytest tests/"
alias backtest="python scripts/run_backtest.py"
alias logs="ssh hamster@192.168.100.64 'docker logs -f bot4-trading'"
alias health="curl http://192.168.100.64:8000/health"
```

### 2. Template Generators

```bash
# scripts/generate_strategy.py
python scripts/generate_strategy.py --name="momentum" --type="trend"
# Creates: src/strategies/momentum.py with boilerplate

# scripts/generate_indicator.py  
python scripts/generate_indicator.py --name="vwap" --type="volume"
# Creates: src/indicators/vwap.py with proper structure
```

### 3. Automated Testing

```bash
# scripts/test_all.sh
#!/bin/bash
pytest tests/unit/
pytest tests/integration/
python scripts/run_backtest.py --all
python scripts/validate_no_fakes.py
python scripts/check_risk_limits.py
```

### 4. Quick Status Check

```bash
# scripts/status.sh
#!/bin/bash
echo "=== System Status ==="
echo "Remote Docker:" 
ssh hamster@192.168.100.64 'docker ps | grep bot4'
echo "API Health:"
curl -s http://192.168.100.64:8000/health | jq
echo "Database:"
psql -h 192.168.100.64 -c "SELECT COUNT(*) FROM trades"
echo "Recent Profits:"
psql -h 192.168.100.64 -c "SELECT SUM(profit) FROM trades WHERE date > NOW() - INTERVAL '24 hours'"
```

## Summary

This enhanced configuration provides:

1. **MCP Integration** - Direct access to GitHub, databases, remote Docker
2. **Project Standards** - Clear structure, naming conventions, git workflow
3. **Deployment Automation** - One-command deployments with rollback
4. **Maintenance Organization** - Scripts organized by purpose
5. **Agent Enhancements** - Specific validation and enforcement rules
6. **Productivity Tools** - Templates, aliases, quick commands

With these enhancements, development will be:
- **Faster** - MCP enables direct operations
- **Safer** - Automated validation and checks
- **Cleaner** - Enforced standards and structure
- **Easier** - Templates and generators
- **Reliable** - Comprehensive testing and monitoring