# Runbook: System Startup Procedure

## Pre-Flight Checklist

### 1. Infrastructure Health
```bash
# Check all services are running
docker-compose ps

# Expected output: All services "Up"
# - bot4-trading: Up
# - postgres: Up  
# - redis: Up
# - prometheus: Up
# - grafana: Up
```

### 2. Data Feed Validation
```bash
# Test exchange connectivity
./scripts/test_exchanges.sh

# Check data freshness
curl http://localhost:9090/api/v1/query?query=data_age_seconds

# Alert if any feed > 5 seconds stale
```

### 3. Risk System Check
```bash
# Load risk policies
./scripts/validate_risk_policies.sh config/risk_policies.toml

# Verify kill switches armed
curl http://localhost:8080/api/risk/status

# Expected: {"global_kill_switch": "armed", "components": "ready"}
```

## Startup Sequence

### Phase 1: Core Services (Order matters!)
```bash
# 1. Start database
docker-compose up -d postgres
sleep 5

# 2. Start cache
docker-compose up -d redis
sleep 2

# 3. Start monitoring
docker-compose up -d prometheus grafana
sleep 5
```

### Phase 2: Risk & Data
```bash
# 4. Start risk engine (must be before trading)
docker-compose up -d risk-engine

# Wait for health check
until curl -f http://localhost:8081/health; do
  echo "Waiting for risk engine..."
  sleep 1
done

# 5. Start data feeds
docker-compose up -d market-data-collector
```

### Phase 3: ML & Trading
```bash
# 6. Warm up ML models (loads into memory)
./scripts/warmup_models.sh

# 7. Start trading engine in SAFE MODE
SAFE_MODE=true docker-compose up -d trading-engine

# 8. Verify safe mode active
curl http://localhost:8080/api/status
# Expected: {"mode": "safe", "trading": "disabled"}
```

### Phase 4: Final Checks
```bash
# Run system diagnostics
./scripts/run_diagnostics.sh

# Check circuit breakers
curl http://localhost:8080/api/circuit_breakers
# All should be "closed" (operational)

# Verify data flowing
curl http://localhost:9090/api/v1/query?query=market_ticks_per_second
# Should be > 100
```

## Enable Trading

### Manual Approval Required
```bash
# After all checks pass, enable trading
curl -X POST http://localhost:8080/api/trading/enable \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"mode": "paper", "confirm": true}'

# Verify enabled
curl http://localhost:8080/api/status
# Expected: {"mode": "paper", "trading": "enabled"}
```

## Rollback Procedure

If ANY check fails:
```bash
# Immediate shutdown
docker-compose down

# Check logs
docker-compose logs --tail=100 trading-engine
docker-compose logs --tail=100 risk-engine

# Fix issue, then restart from Phase 1
```

## Monitoring URLs

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Trading API: http://localhost:8080
- Risk API: http://localhost:8081

## Alert Contacts

- On-Call: +1-XXX-XXX-XXXX
- Risk Team: risk@bot4trading.com
- Dev Team: dev@bot4trading.com
- Escalation: alex@bot4trading.com