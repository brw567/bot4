# Bot3 Trading Platform - Deployment Guide

## 🚀 Quick Start

```bash
# Build and deploy locally
./deploy.sh deploy-local

# Deploy to Kubernetes
./deploy.sh deploy-k8s

# View logs
./deploy.sh logs
```

---

## 📦 Deployment Package Contents

### Docker Files
- `Dockerfile` - Multi-stage production image (<100MB final size)
- `docker-compose.yml` - Local development stack
- `.dockerignore` - Optimize build context

### Kubernetes Manifests
- `k8s/deployment.yaml` - Main application deployment
- `k8s/configmap.yaml` - Configuration and secrets
- `k8s/monitoring.yaml` - Prometheus/Grafana setup
- `k8s/ingress.yaml` - External access configuration

### Monitoring
- `monitoring/prometheus.yml` - Metrics collection config
- `monitoring/grafana/dashboards/` - Pre-built dashboards
- `monitoring/alerts/` - Alert rules

### Scripts
- `deploy.sh` - Unified deployment script
- `backup.sh` - Database backup automation
- `rollback.sh` - Emergency rollback procedure

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Load Balancer                   │
│                   (Ingress)                      │
└─────────────────┬───────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼─────┐          ┌─────▼─────┐
│  Bot3     │          │  Bot3     │     ... (Auto-scaling 3-10 pods)
│  Engine   │          │  Engine   │
│  Pod 1    │          │  Pod 2    │
└─────┬─────┘          └─────┬─────┘
      │                      │
      └──────────┬───────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│ Redis │   │Postgres│   │Monitor│
│       │   │        │   │       │
└───────┘   └────────┘   └───────┘
```

---

## 🔧 Deployment Environments

### Development (Local Docker)
```bash
# Start all services
./deploy.sh deploy-local

# Access points:
- API: http://localhost:8080
- Metrics: http://localhost:9090
- Grafana: http://localhost:3000
```

### Staging (Kubernetes - Namespace: bot3-staging)
```bash
# Deploy to staging
ENVIRONMENT=staging ./deploy.sh deploy-k8s

# Run integration tests
./deploy.sh test-integration
```

### Production (Kubernetes - Namespace: bot3-prod)
```bash
# Deploy to production
ENVIRONMENT=production VERSION=v1.0.0 ./deploy.sh deploy-k8s

# Enable monitoring alerts
kubectl apply -f k8s/alerts-prod.yaml
```

---

## 📊 Resource Requirements

### Minimum Requirements (Per Pod)
- **CPU**: 2 cores
- **Memory**: 2GB RAM
- **Storage**: 10GB SSD
- **Network**: 100Mbps

### Recommended Production Setup
- **Nodes**: 3+ for HA
- **Pods**: 3-10 (auto-scaling)
- **CPU**: 4 cores per pod
- **Memory**: 4GB per pod
- **Storage**: 100GB SSD (shared)

### Database Requirements
- **PostgreSQL**: 16GB RAM, 500GB SSD
- **Redis**: 8GB RAM (in-memory)

---

## 🔐 Security Configuration

### Secrets Management
```bash
# Create secrets
kubectl create secret generic bot3-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=redis-url='redis://...' \
  -n bot3

# Create exchange API keys
kubectl create secret generic bot3-exchange-keys \
  --from-file=keys.json \
  -n bot3
```

### Network Policies
```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: bot3-network-policy
spec:
  podSelector:
    matchLabels:
      app: bot3
  policyTypes:
  - Ingress
  - Egress
```

---

## 📈 Monitoring & Observability

### Prometheus Metrics
- `bot3_orders_processed_total` - Total orders processed
- `bot3_latency_nanoseconds` - Processing latency
- `bot3_apy_current` - Current APY
- `bot3_positions_active` - Active positions
- `bot3_risk_score` - Current risk score

### Grafana Dashboards
1. **System Overview** - Overall health and performance
2. **Trading Metrics** - APY, win rate, positions
3. **Risk Dashboard** - Exposure, drawdown, limits
4. **Exchange Status** - Connectivity, latency, errors

### Alerts
- High latency (>100ms)
- Risk limit breach
- Connection failures
- Low APY (<100%)
- System errors

---

## 🔄 Deployment Process

### 1. Pre-deployment Checks
```bash
# Run tests
cargo test --all --release

# Check image builds
docker build -t bot3/trading-engine:test .

# Validate Kubernetes manifests
kubectl apply --dry-run=client -f k8s/
```

### 2. Rolling Deployment
```bash
# Update image version
kubectl set image deployment/bot3-trading-engine \
  trading-engine=bot3/trading-engine:v1.0.1 \
  -n bot3

# Monitor rollout
kubectl rollout status deployment/bot3-trading-engine -n bot3
```

### 3. Health Verification
```bash
# Check pod status
kubectl get pods -n bot3

# Check metrics
curl http://bot3-service:9090/metrics

# Run smoke tests
./deploy.sh test-smoke
```

---

## 🚨 Rollback Procedures

### Immediate Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/bot3-trading-engine -n bot3

# Verify rollback
kubectl rollout status deployment/bot3-trading-engine -n bot3
```

### Emergency Stop
```bash
# Scale to 0 (stops trading)
kubectl scale deployment/bot3-trading-engine --replicas=0 -n bot3

# Close all positions (manual intervention)
./scripts/emergency-close-positions.sh
```

---

## 📝 Configuration Management

### Environment Variables
```bash
# Required
DATABASE_URL        # PostgreSQL connection
REDIS_URL          # Redis connection
EXCHANGE_API_KEYS  # JSON with exchange credentials

# Optional
RUST_LOG           # Log level (default: info)
BOT3_MODE          # production/staging/development
MAX_POSITION_SIZE  # Risk limit (default: 0.02)
TARGET_APY         # Target APY (default: 2.0)
```

### Configuration Files
- `config/config.toml` - Main configuration
- `config/exchanges.toml` - Exchange settings
- `config/strategies.toml` - Strategy parameters
- `config/risk.toml` - Risk limits

---

## 🔧 Maintenance Operations

### Database Backup
```bash
# Automated daily backup
0 2 * * * /opt/bot3/scripts/backup.sh

# Manual backup
./scripts/backup.sh manual
```

### Log Rotation
```bash
# Configured in Vector
# Keeps 7 days of logs
# Compresses after 1 day
```

### Performance Tuning
```bash
# Adjust HPA thresholds
kubectl edit hpa bot3-trading-engine-hpa -n bot3

# Update resource limits
kubectl edit deployment bot3-trading-engine -n bot3
```

---

## 📊 Performance Benchmarks

### Expected Metrics
- **Latency P99**: <50ns
- **Throughput**: 10,000+ signals/sec
- **Memory Usage**: <2GB per pod
- **CPU Usage**: <70% average
- **Startup Time**: <10 seconds

### Load Testing
```bash
# Run load tests
./scripts/load-test.sh

# Monitor during test
watch kubectl top pods -n bot3
```

---

## 🆘 Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check events
kubectl describe pod <pod-name> -n bot3

# Check logs
kubectl logs <pod-name> -n bot3
```

#### High Memory Usage
```bash
# Check memory
kubectl top pods -n bot3

# Restart pods
kubectl rollout restart deployment/bot3-trading-engine -n bot3
```

#### Connection Issues
```bash
# Test connectivity
kubectl exec -it <pod-name> -n bot3 -- curl -v exchange-api

# Check network policies
kubectl get networkpolicies -n bot3
```

---

## 📞 Support & Contact

### Documentation
- Architecture: `/ARCHITECTURE.md`
- API Reference: `/docs/api/`
- Development: `/DEVELOPMENT.md`

### Monitoring URLs
- Grafana: https://grafana.bot3.io
- Prometheus: https://prometheus.bot3.io
- Logs: https://logs.bot3.io

### Emergency Contacts
- On-Call: Check PagerDuty
- Slack: #bot3-alerts
- Email: ops@bot3.io

---

## 📅 Deployment Checklist

### Pre-Production Checklist
- [ ] All tests passing (100%)
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Backup verified
- [ ] Rollback plan tested
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Team notified
- [ ] Maintenance window scheduled

### Post-Deployment Checklist
- [ ] All pods running
- [ ] Metrics flowing
- [ ] No error logs
- [ ] APY tracking correctly
- [ ] Risk limits enforced
- [ ] Smoke tests passed
- [ ] Performance validated
- [ ] Documentation updated
- [ ] Team notified
- [ ] Monitoring verified

---

*Last Updated: January 12, 2025*
*Version: 1.0.0*
*Status: Production Ready*