# Runbook: Degraded Data Response

## Detection Triggers

### Automatic Detection
- Quote age > 500ms (spot)
- Book update gap > 1 second
- Price source deviation > 1%
- Missing depth levels
- Crossed quotes (bid > ask)

### Manual Detection Signs
- Unusual spread widening
- Price jumps without volume
- Frozen order book
- Repeated values (stuck feed)

## Severity Levels

### Level 1: Minor Degradation (Warning)
**Symptoms:**
- Single exchange delayed 200-500ms
- Spread widened < 2x normal
- 1 of 3 price sources offline

**Actions:**
```bash
# 1. Log incident
echo "$(date): Level 1 degradation detected" >> /var/log/bot4/incidents.log

# 2. Reduce position sizes
curl -X POST http://localhost:8080/api/risk/adjust \
  -d '{"position_multiplier": 0.75}'

# 3. Increase monitoring frequency
export HEALTH_CHECK_INTERVAL=10
```

### Level 2: Moderate Degradation (Reduce)
**Symptoms:**
- Multiple exchanges delayed
- Spread > 2x normal
- 2 of 3 price sources offline
- Book depth < $10,000

**Actions:**
```bash
# 1. Pause new positions
curl -X POST http://localhost:8080/api/trading/pause_new

# 2. Close risky positions
./scripts/close_positions.sh --risk-score ">7"

# 3. Switch to backup feeds
curl -X POST http://localhost:8080/api/data/switch_backup

# 4. Alert team
./scripts/send_alert.sh "Level 2 data degradation"
```

### Level 3: Severe Degradation (Halt)
**Symptoms:**
- All primary feeds delayed > 1 second
- No valid quotes for > 5 seconds
- Price discrepancy > 2% between venues
- Complete book failures

**Actions:**
```bash
# 1. IMMEDIATE HALT
curl -X POST http://localhost:8080/api/emergency/stop

# 2. Close ALL positions at market
./scripts/emergency_close_all.sh --confirm

# 3. Page on-call
./scripts/page_oncall.sh "CRITICAL: Data feeds failed"

# 4. Diagnostic dump
./scripts/diagnostic_dump.sh > /tmp/dump_$(date +%s).json
```

## Recovery Procedures

### Verify Data Quality
```bash
# Check all feeds
for exchange in binance coinbase kraken; do
  echo "Testing $exchange..."
  curl http://localhost:8080/api/data/test/$exchange
done

# Validate quotes
./scripts/validate_quotes.sh --threshold 0.01

# Check latencies
curl http://localhost:9090/api/v1/query?query=data_latency_p99
```

### Gradual Resumption
```bash
# 1. Start with read-only mode
curl -X POST http://localhost:8080/api/mode/readonly

# 2. Monitor for 5 minutes
sleep 300
./scripts/check_data_quality.sh

# 3. Enable paper trading
curl -X POST http://localhost:8080/api/mode/paper

# 4. Monitor for 30 minutes
sleep 1800

# 5. If stable, resume normal
curl -X POST http://localhost:8080/api/mode/normal
```

## Diagnostics Commands

### Quick Health Check
```bash
# All-in-one status
./scripts/health_check.sh --verbose

# Individual components
curl http://localhost:8080/api/data/health
curl http://localhost:8080/api/exchanges/status
curl http://localhost:8080/api/risk/status
```

### Data Feed Analysis
```bash
# Quote freshness
watch -n 1 'curl -s http://localhost:8080/api/data/age'

# Spread analysis
./scripts/analyze_spreads.sh --window 5m

# Cross-venue comparison
./scripts/compare_venues.sh --symbol BTC/USDT
```

### Historical Analysis
```sql
-- Find degradation periods
SELECT 
  timestamp,
  exchange,
  AVG(quote_age_ms) as avg_age,
  MAX(quote_age_ms) as max_age,
  COUNT(*) as samples
FROM market_data_quality
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY timestamp, exchange
HAVING MAX(quote_age_ms) > 500
ORDER BY timestamp DESC;
```

## Prevention Measures

### Monitoring Alerts
```yaml
alerts:
  - name: data_age_high
    expr: data_age_seconds > 0.5
    for: 10s
    severity: warning
    
  - name: spread_widening
    expr: spread_bps > 50
    for: 1m
    severity: critical
    
  - name: feed_disconnected
    expr: up{job="market_data"} == 0
    for: 5s
    severity: critical
```

### Redundancy Setup
- Primary: Direct exchange WebSocket
- Backup 1: REST API polling
- Backup 2: Alternative data provider
- Emergency: Cached last-known-good

## Post-Incident

### Required Actions
1. Write incident report within 4 hours
2. Update runbook with new scenarios
3. Add detection for this pattern
4. Test recovery procedure
5. Schedule post-mortem meeting

### Report Template
```markdown
## Incident Report: [DATE]

**Duration:** Start - End  
**Severity:** L1/L2/L3  
**Impact:** Positions affected, $ impact  

**Timeline:**
- HH:MM - Detection
- HH:MM - Response initiated
- HH:MM - Mitigated
- HH:MM - Resolved

**Root Cause:**
[Description]

**Action Items:**
1. [Owner] - [Action] - [Due Date]
```