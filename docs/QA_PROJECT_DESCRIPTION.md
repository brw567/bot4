# Bot4 Trading Platform - QA Team Project Description

## 512-Character Summary

Bot4 is an institutional-grade autonomous cryptocurrency trading platform targeting 200-300% APY through revolutionary 50/50 TA-ML hybrid strategies. Built in pure Rust for <50ns latency and 10K+ ops/sec, it features comprehensive risk management, multi-exchange integration, and zero-tolerance for fake implementations. The platform operates 100% autonomously with circuit breakers, VIP tier optimization, and advanced fee management to maximize net profitability while maintaining strict 2% position limits.

---

## Extended Project Overview

### Core Technology Stack
- **Language**: 100% Rust (zero Python in production)
- **Performance**: <50 nanosecond decision latency
- **Throughput**: 10,000+ orders per second capability
- **Architecture**: Event-driven with SIMD optimizations
- **Databases**: PostgreSQL + TimescaleDB for time-series
- **Monitoring**: Prometheus, Grafana, Jaeger, Loki

### Key Features
1. **Hybrid Strategy System**: 50% Technical Analysis + 50% Machine Learning
2. **Multi-Exchange Support**: Binance, Kraken, Coinbase (more planned)
3. **Risk Management**: Circuit breakers, position limits, stop-loss enforcement
4. **Fee Optimization**: Dynamic maker/taker decisions, VIP tier management
5. **Real-time Monitoring**: Complete observability stack with alerts

### Quality Requirements
- **100% Real Code**: No TODOs, no placeholders, no fake implementations
- **95% Test Coverage**: Comprehensive unit, integration, and E2E tests
- **Performance Benchmarks**: Every critical path must meet latency targets
- **Risk Controls**: Mandatory circuit breakers in every component
- **Documentation**: 100% API documentation, updated architecture docs

### Development Methodology
- **Multi-Agent System**: 8 virtual AI personas with specialized roles
- **Granular PR Workflow**: One sub-task = one PR for external review
- **Quality Gates**: Automated validation on every commit
- **Local Development**: All work done on /home/hamster/bot4/

### Success Metrics
- **Profitability**: 200-300% APY in bull markets, 60-80% in bear
- **Reliability**: 99.99% uptime with <10 second recovery
- **Autonomy**: Zero manual interventions for 30+ days
- **Risk**: Maximum 15% drawdown, Sharpe ratio >2.0
- **Quality**: Zero fake implementations in production

### Current Status (August 16, 2025)
- **Phase 0**: Foundation setup completed
- **Phase 1-2**: Core infrastructure in progress
- **Critical Discovery**: Fee management system being added (Phase 3.5)
- **Timeline**: 12-week development cycle
- **Target**: Production ready by April 2025

### QA Focus Areas
1. **Fake Implementation Detection**: Validate no placeholders or TODOs
2. **Performance Testing**: Verify <50ns latency requirements
3. **Risk Validation**: Ensure all circuit breakers functional
4. **Fee Calculations**: Verify accuracy of fee management system
5. **Integration Testing**: Multi-exchange connectivity and failover

### Repository Structure
```
/home/hamster/bot4/
├── rust_core/          # Pure Rust implementation
├── frontend/           # React TypeScript UI
├── scripts/            # Validation and automation
├── sql/               # Database schemas
└── docs/              # Architecture and documentation
```

### PR Review Guidelines
Each PR must include:
- Task ID from PROJECT_MANAGEMENT_TASK_LIST_V5.md
- Detailed implementation explanation
- Test coverage report (must be >95%)
- Performance benchmarks where applicable
- No fake implementations (automated check)

### Contact & Resources
- **Repository**: git@github.com:brw567/bot4.git
- **Documentation**: ARCHITECTURE.md (2,267 lines)
- **Task List**: PROJECT_MANAGEMENT_TASK_LIST_V5.md
- **Setup Script**: scripts/qa_environment_setup.sh

---

## Critical Quality Gates

### Automated Checks (Every Commit)
```bash
- scripts/validate_no_fakes.py
- scripts/validate_no_fakes_rust.py
- cargo fmt --check
- cargo clippy -- -D warnings
- cargo test --all
```

### Manual Review Focus
- Verify real implementations (no mocks in production)
- Check risk controls are present and functional
- Validate fee calculations and optimizations
- Ensure performance targets are met
- Confirm documentation is complete

---

*This project represents cutting-edge algorithmic trading with institutional-grade reliability. Your QA expertise is critical to ensuring we deliver a system capable of managing millions in assets with zero tolerance for errors.*