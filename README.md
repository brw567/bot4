# Bot3 - Institutional-Grade Crypto Trading Platform

## Overview
Bot3 is a professional cryptocurrency trading system that combines advanced machine learning, comprehensive technical analysis, and multi-exchange support to deliver consistent profitability.

## Key Features
- ✅ 30+ Technical Indicators (Real implementations from RC1/RC2)
- ✅ Machine Learning Price Prediction (Advanced ML Pipeline)
- ✅ Multi-Exchange Support (Binance, Coinbase)
- ✅ Smart Order Routing
- ✅ React-based Trading Interface
- ✅ Real-time WebSocket Data Feeds
- ✅ Comprehensive Risk Management
- ✅ Advanced Backtesting Framework
- ✅ Production-Ready Architecture

## Quick Start

```bash
# 1. Setup environment
cd /home/hamster/DEV/bot3
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Start services
docker-compose up -d

# 5. Run trading bot
python src/main.py
```

## Architecture

```
bot3/
├── src/
│   ├── core/
│   │   ├── data_fetcher.py      # Market data with real TA
│   │   ├── exchange_manager.py  # Multi-exchange support
│   │   └── risk_manager.py      # Risk controls
│   ├── ml/
│   │   └── advanced_ml_pipeline.py # ML models
│   ├── strategies/               # Trading strategies
│   └── indicators/               # TA indicators
├── frontend/                     # React interface
├── backend/                      # API services
└── deployment/                   # Docker configs
```

## Recovered Components

### From RC1
- React frontend with proper build system
- Advanced ML pipeline with feature engineering
- 30+ TA indicators using ta library

### From RC2
- Complete data fetcher with real ATR calculation
- Support/resistance detection
- Market microstructure features

### From RC5 (exchange_manager_old)
- Prometheus metrics
- WebSocket streaming
- Unified Binance/Coinbase interface

## Multi-Agent System

This project uses an advanced multi-agent system where virtual team members can interact, challenge each other, and collaborate:

### Agent Roles & Capabilities
- **Alex**: Strategic Architect (Team Lead, can override all decisions)
- **Morgan**: ML/AI Specialist (Can challenge TA approaches with ML alternatives)
- **Sam**: Strategy & TA Expert (Enforces no-fake-implementations rule)
- **Quinn**: Risk Manager (Has veto power on risk-related decisions)
- **Jordan**: DevOps & Performance (Ensures production readiness)
- **Casey**: Exchange Specialist (Handles market connectivity)
- **Riley**: UI/UX Expert (Focuses on user experience)
- **Avery**: Data Pipeline Specialist (Ensures data quality)

### How Agents Interact

1. **Code Review**: Multiple agents review and challenge code
2. **Decision Making**: Weighted voting with veto powers
3. **Quality Gates**: Each agent enforces domain-specific standards
4. **Challenge Mechanism**: Agents can challenge each other's approaches

### Using the Agent System

```bash
# Single agent mode
"Morgan, implement an ML model for price prediction"

# Multi-agent review
"I need Sam, Morgan, and Quinn to review this strategy"

# Challenge mode
"Sam, implement RSI, then have Morgan challenge your approach"

# Full team meeting
"All agents: discuss the architecture for order execution"
```

### Configuration Files
- `.claude_project` - Basic persona definitions
- `.claude/agents_config.json` - Advanced interaction rules
- `.claude/AGENT_INTERACTION_GUIDE.md` - Complete usage guide
- `.claude/agent_interaction_example.py` - Working demonstration

## Status

✅ Project structure created
✅ Claude configuration with personas
✅ Working TA implementations copied
✅ React frontend copied
✅ Core components in place

## Next Steps

1. Create requirements.txt with all dependencies
2. Set up Docker configuration
3. Create main trading loop
4. Integrate all components
5. Add comprehensive testing
