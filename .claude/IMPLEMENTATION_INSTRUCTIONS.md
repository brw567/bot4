# Bot4 Implementation Instructions

## Step-by-Step Setup Guide

### 1. Enable MCP (Model Context Protocol)

#### For Claude Desktop Users:

1. **Install MCP Tools**:
```bash
# Install globally
npm install -g @anthropic/mcp-cli

# Or use npx without installation
npx @anthropic/mcp-cli init
```

2. **Configure MCP**:
```bash
# Create MCP config directory
mkdir -p ~/.config/claude/mcp

# Create configuration
cat > ~/.config/claude/mcp/config.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/home/hamster"]
    },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your_github_token_here"
      }
    }
  }
}
EOF
```

3. **Enable in Claude Desktop**:
   - Open Claude Desktop
   - Go to Settings → Developer
   - Enable "Model Context Protocol"
   - Restart Claude Desktop

4. **Verify MCP is working**:
   - Type: "List files in /home/hamster/bot4"
   - You should see file listing without using bash commands

### 2. Set Up Git Hooks

```bash
cd /home/hamster/bot4

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit checks..."

# Run unit tests
pytest tests/unit/ -q || {
    echo "❌ Unit tests failed"
    exit 1
}

# Check for fake implementations
python scripts/validate_no_fakes.py || {
    echo "❌ Fake implementations detected"
    exit 1
}

# Format check
black --check src/ || {
    echo "❌ Code formatting issues (run: black src/)"
    exit 1
}

echo "✅ Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit

# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "Running pre-push validation..."

# Full test suite
pytest tests/ || {
    echo "❌ Tests failed"
    exit 1
}

# Risk validation
python scripts/check_risk_limits.py || {
    echo "❌ Risk limits exceeded"
    exit 1
}

echo "✅ Pre-push validation passed"
EOF

chmod +x .git/hooks/pre-push
```

### 3. Configure Remote Deployment

```bash
# Set up SSH key for remote host
ssh-copy-id hamster@192.168.100.64

# Create deployment environment
cat > .env.deploy << 'EOF'
REMOTE_HOST=192.168.100.64
REMOTE_USER=hamster
REMOTE_PATH=/home/hamster/bot4_production
BACKUP_PATH=/home/hamster/backups
DOCKER_REGISTRY=hamster/bot4
EOF

# Make deployment script executable
chmod +x scripts/deploy_remote.sh

# Test deployment connection
./scripts/deploy_remote.sh --status
```

### 4. Set Up Project Structure

```bash
# Create all required directories
mkdir -p {scripts/{deploy,fix,test,validate,backup,maintenance},tests/{unit,integration,backtesting},docs/{api,architecture,guides},data/{historical,models}}

# Move scripts to correct locations
mv scripts/validate_no_fakes.py scripts/validate/
mv scripts/check_risk_limits.py scripts/validate/
mv scripts/deploy_remote.sh scripts/deploy/

# Create script index
cat > scripts/README.md << 'EOF'
# Scripts Directory Structure

## Categories:
- `deploy/` - Deployment scripts
- `fix/` - Fix and recovery scripts  
- `test/` - Testing scripts
- `validate/` - Validation scripts
- `backup/` - Backup scripts
- `maintenance/` - Maintenance scripts

## Naming Convention:
- Deploy scripts: `deploy_*.sh`
- Fix scripts: `fix_*.sh`
- Test scripts: `test_*.sh`
- Validation: `validate_*.py`
- Backup: `backup_*.sh`
EOF
```

### 5. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Additional development tools
pip install pre-commit black flake8 mypy pytest-cov

# Frontend dependencies
cd frontend
npm install
cd ..

# Pre-commit framework
pre-commit install
```

### 6. Configure Monitoring

```bash
# Create docker-compose for monitoring
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource

volumes:
  prometheus_data:
  grafana_data:
EOF
```

### 7. Create Validation Suite

```bash
# Create comprehensive validation script
cat > scripts/validate/validate_all.sh << 'EOF'
#!/bin/bash
set -e

echo "Running complete validation suite..."

# 1. No fake implementations
python scripts/validate/validate_no_fakes.py

# 2. Risk limits
python scripts/validate/check_risk_limits.py

# 3. Code quality
black --check src/
flake8 src/ --max-line-length=100

# 4. Type checking
mypy src/ --ignore-missing-imports

# 5. Security scan
bandit -r src/ -ll

# 6. Dependency check
safety check

echo "✅ All validations passed!"
EOF

chmod +x scripts/validate/validate_all.sh
```

### 8. Set Up Continuous Deployment

```bash
# GitHub Actions workflow
mkdir -p .github/workflows
cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/
      
      - name: Check for fakes
        run: python scripts/validate/validate_no_fakes.py
      
      - name: Validate risk limits
        run: python scripts/validate/check_risk_limits.py

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        env:
          REMOTE_HOST: ${{ secrets.REMOTE_HOST }}
          SSH_KEY: ${{ secrets.SSH_KEY }}
        run: |
          echo "$SSH_KEY" > ssh_key
          chmod 600 ssh_key
          ssh -i ssh_key hamster@$REMOTE_HOST 'cd /home/hamster/bot4_production && git pull && ./scripts/deploy_remote.sh'
EOF
```

## Quick Reference Commands

### Daily Operations

```bash
# Morning checks
./scripts/validate/validate_all.sh
./scripts/test/test_all.sh

# Deploy to staging
./scripts/deploy/deploy_remote.sh --backup --deploy

# Monitor logs
ssh hamster@192.168.100.64 'docker logs -f bot4-trading'

# Quick status
./scripts/deploy/deploy_remote.sh --status
```

### Agent-Specific Commands

```bash
# Sam's validation
python scripts/validate/validate_no_fakes.py

# Quinn's risk check
python scripts/validate/check_risk_limits.py

# Morgan's ML validation
python scripts/test/test_ml_overfitting.py

# Jordan's performance check
./scripts/test/test_performance.sh
```

### Emergency Procedures

```bash
# Immediate rollback
./scripts/deploy/deploy_remote.sh --rollback

# Stop all trading
ssh hamster@192.168.100.64 'docker-compose stop trading-engine'

# Emergency backup
./scripts/backup/emergency_backup.sh

# Fix database issues
./scripts/fix/fix_database.sh
```

## Project Maintenance Schedule

### Daily
- [ ] Check system health
- [ ] Review error logs
- [ ] Monitor performance metrics
- [ ] Validate no new fakes

### Weekly
- [ ] Run full test suite
- [ ] Review risk metrics (Quinn)
- [ ] Backtest strategies (Sam)
- [ ] Performance optimization (Jordan)
- [ ] ML model review (Morgan)

### Monthly
- [ ] Security audit
- [ ] Dependency updates
- [ ] Strategy performance review
- [ ] Infrastructure scaling review
- [ ] Cost optimization

## Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
# Exchange Configuration
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET=your_testnet_secret
BINANCE_TESTNET=true

# Database
DATABASE_URL=postgresql://bot4user:bot4pass@localhost:5432/bot4trading
REDIS_URL=redis://localhost:6379/0

# Trading Configuration
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.02
MAX_LEVERAGE=3
USE_PAPER_TRADING=true

# Risk Management (Quinn's limits)
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.15
REQUIRE_STOP_LOSS=true

# ML Configuration (Morgan's settings)
ML_TRAIN_TEST_SPLIT=0.7,0.2,0.1
ML_MAX_OVERFIT_GAP=0.05
ML_CROSS_VALIDATION_FOLDS=5

# Performance (Jordan's requirements)
MAX_LATENCY_MS=100
MIN_UPTIME_PERCENT=99.9
HEALTH_CHECK_INTERVAL=30

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
ALERT_EMAIL=admin@bot4.com
EOF
```

## Troubleshooting Guide

### Common Issues and Fixes

1. **Fake Implementation Detected**:
```bash
# Find and fix
python scripts/validate/validate_no_fakes.py
# Shows exact location of fake code
# Fix the implementation
# Re-run validation
```

2. **Risk Limits Exceeded**:
```bash
# Check violations
python scripts/validate/check_risk_limits.py
# Quinn will show exact violations
# Adjust parameters in config
# Re-validate
```

3. **Deployment Failed**:
```bash
# Check logs
./scripts/deploy/deploy_remote.sh --status
# If health check failed
ssh hamster@192.168.100.64 'docker logs bot4-trading'
# Rollback if needed
./scripts/deploy/deploy_remote.sh --rollback
```

4. **Performance Issues**:
```bash
# Check metrics
curl http://192.168.100.64:9090/metrics | grep latency
# Profile code
python -m cProfile -o profile.stats src/main.py
# Analyze bottlenecks
python scripts/analyze_performance.py profile.stats
```

## Success Criteria

The system is considered properly configured when:

✅ All validation scripts pass  
✅ No fake implementations exist  
✅ Risk limits are enforced  
✅ Tests have >80% coverage  
✅ Deployment completes in <5 minutes  
✅ Health checks pass  
✅ Monitoring shows all green  
✅ Latency <100ms  
✅ No critical security issues  
✅ Documentation is complete  

## Contact Information

- **Repository**: https://github.com/brw567/bot4
- **Remote Host**: hamster@192.168.100.64
- **Monitoring**: http://192.168.100.64:3001 (Grafana)
- **API**: http://192.168.100.64:8000/docs

Remember: Build it right the first time. Every decision matters.