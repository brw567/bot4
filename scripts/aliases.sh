#!/bin/bash
# Bot3 project aliases

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
alias remote-logs='ssh hamster@192.168.100.64 "docker logs bot3-trading --tail 100"'
alias remote-status='ssh hamster@192.168.100.64 "docker ps | grep bot3"'
alias remote-health='curl http://192.168.100.64:8000/health'

echo "Bot3 aliases loaded"
