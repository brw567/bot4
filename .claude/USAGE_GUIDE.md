# Bot3 Multi-Agent System - Practical Usage Guide

## Quick Start Commands

### Basic Agent Invocations

```bash
# Single agent task
"Morgan, implement an LSTM model for BTC price prediction"

# Multi-agent review
"Sam and Morgan, review the RSI implementation in src/indicators/rsi.py"

# Full team meeting
"Team meeting: discuss the architecture for real-time order execution"

# Challenge mode
"Sam, implement Bollinger Bands. Morgan, challenge his approach with ML alternative."
```

## Common Scenarios & Best Practices

### Scenario 1: Starting a New Feature

```bash
User: "I need to implement a new arbitrage strategy for cross-exchange trading"

# Correct approach - Full team involvement:
User: "Team meeting: plan the cross-exchange arbitrage implementation"

Alex: "I'll coordinate. Let's break this down by expertise areas."
Casey: "I'll handle exchange connections and order routing"
Sam: "I'll design the arbitrage math and opportunity detection"
Quinn: "Setting risk limits: max 2% per trade, 5% total exposure"
Morgan: "I can add ML for spread prediction"
Jordan: "We need sub-50ms execution for this to work"

Alex: "Approved. Casey leads implementation with Sam's math. Quinn monitors."
```

### Scenario 2: Code Review with Conflicts

```bash
User: "Review this momentum strategy implementation"

# Agents automatically engage:
Sam: "The RSI calculation is wrong - using SMA instead of EMA"
Morgan: "Also, why not use LSTM for momentum prediction?"
Sam: "Because backtesting shows traditional RSI has better Sharpe"
Morgan: "Show me the data"
[Round 2]
Sam: "Here: RSI Sharpe=1.8, LSTM Sharpe=1.5"
Morgan: "But LSTM catches regime changes faster"
[Round 3 - limit reached]
Alex: "Decision: Use RSI as primary, add LSTM for regime detection. Best of both."
```

### Scenario 3: Risk Veto Situation

```bash
User: "Implement 10x leverage for the scalping strategy"

Quinn: "VETO: 10x leverage could wipe account in minutes"
User: "But the potential returns are huge"
Quinn: "Maximum 3x leverage with strict stop losses, non-negotiable"
Alex: "Quinn's veto on leverage stands. User, would you like to proceed with 3x?"
```

### Scenario 4: Innovation Project

```bash
User: "Morgan, research reinforcement learning for trading"

Morgan: "I'd like to explore RL for dynamic position sizing"
Sam: "That's never been proven in crypto"
Morgan: "That's why it's innovation - let me use my 20% time"
Alex: "Approved under innovation budget. Paper trading only, report in 30 days"
```

## Optimal Usage Patterns

### 1. For Maximum Quality

```python
# Best: Specific multi-agent review
"Sam, Morgan, and Quinn: review src/strategies/arbitrage.py"

# Good: Full team for architecture
"Full team: design the risk management system"

# Avoid: Single agent for complex tasks
"Jordan, build entire trading system"  # Too broad for one agent
```

### 2. For Fastest Development

```python
# Fast path for clear tasks
"Sam, implement real ATR indicator using ta library"  # Clear, no debate needed

# Parallel work
"Casey: implement Binance connector. Riley: build trading dashboard"

# Avoid debate on settled issues
"Team, should we use fake implementations?"  # No - this is settled
```

### 3. For Innovation

```python
# Protected innovation time
"Morgan, use Friday innovation time for neural architecture search"

# Controlled experiments
"Test quantum-inspired portfolio optimization in paper trading"

# Fast-track small experiments
"Try new indicator combination, report results in 24 hours"
```

### 4. For Risk Management

```python
# Always include Quinn for risk
"Quinn, review all strategies before production"

# Respect vetos
"Quinn vetoed uncapped losses - implement stops immediately"

# Proactive risk assessment
"Quinn, what's worst-case scenario for this strategy?"
```

## Anti-Patterns to Avoid

### ❌ Don't Do This:

```python
# Endless debates
"Everyone argue about tabs vs spaces"  # Alex will cut this off

# Ignoring expertise
"Riley, implement the ML model"  # Wrong agent for the task

# Overriding risk vetos
"Ignore Quinn, YOLO the entire account"  # Never override risk vetos

# Fake implementations
"Just use price * 0.02 for ATR"  # Sam will reject immediately
```

### ✅ Do This Instead:

```python
# Clear decisions
"Alex, make the call: PostgreSQL or MongoDB?"

# Right agent for task
"Morgan, implement the ML model"

# Respect risk limits
"Quinn, what's safe position size?"

# Real implementations
"Sam, implement ATR using ta library"
```

## Conflict Resolution Flowchart

```
Start → Is it a risk issue?
         ├─ Yes → Quinn decides (veto power)
         └─ No → Is it technical?
                  ├─ Yes → Try data/backtest comparison
                  │        └─ No consensus after 3 rounds? → Alex decides
                  └─ No → Is it about user experience?
                           ├─ Yes → Riley leads, Jordan validates performance
                           └─ No → Alex coordinates appropriate agents
```

## Daily Workflow Example

### Morning Standup (9:00 AM)
```python
"Daily standup - all agents report"

Alex: "Priority today: finish arbitrage strategy"
Morgan: "Completed: ML model training. Today: integration"
Sam: "Completed: backtesting. Found issue with slippage calculation"
Quinn: "Reviewing risk parameters for production"
Casey: "WebSocket connections stable. Adding reconnection logic"
Jordan: "Latency at 45ms, optimizing to get under 40ms"
Riley: "Dashboard updates live. Adding position tracker"
Avery: "Data pipeline stable. 100% uptime last 24h"

Alex: "Good. Sam and Casey collaborate on slippage. Morgan, help Jordan optimize."
```

### Code Review (2:00 PM)
```python
"Code review: src/strategies/new_scalping.py"

Sam: "Math checks out, but no stop losses"
Quinn: "REJECT until stops added"
Jordan: "Also needs async implementation"
Morgan: "Consider adding ML confidence filter"

# After fixes
Sam: "Math ✓"
Quinn: "Risk limits ✓"
Jordan: "Performance ✓"
Alex: "Approved for staging"
```

### End of Day Decisions (5:00 PM)
```python
"EOD review - any blockers?"

Casey: "Binance rate limits hitting earlier than expected"
Jordan: "I can implement request queuing"
Casey: "That would add 10ms latency"
Alex: "10ms acceptable. Implement tonight, test tomorrow"

# Logged in decision_log.md
```

## Special Commands

### Innovation Mode
```bash
# Activate innovation protection
"Innovation mode: explore new predictive models"
# Morgan gets freedom, Sam can't veto for being unproven
```

### Emergency Mode
```bash
# All hands emergency
"EMERGENCY: System down in production"
# Jordan leads, all agents support, no debates
```

### Audit Mode
```bash
# Comprehensive review
"Audit mode: review entire codebase for fake implementations"
# Sam leads, zero tolerance, all must pass
```

### Learning Mode
```bash
# Post-mortem analysis
"Learning mode: why did yesterday's strategy fail?"
# All agents contribute insights, no blame, focus on improvement
```

## Measuring Success

### Agent Performance Metrics

```python
metrics = {
    'alex': {
        'decisions_made': 45,
        'deadlocks_broken': 12,
        'on_time_delivery': '94%'
    },
    'morgan': {
        'models_deployed': 8,
        'overfitting_prevented': 15,
        'innovation_success': '60%'
    },
    'sam': {
        'fake_implementations_caught': 23,
        'strategies_profitable': '73%',
        'backtest_accuracy': '95%'
    },
    'quinn': {
        'losses_prevented': '$45,000',
        'risk_events_caught': 18,
        'vetos_exercised': 7
    },
    'jordan': {
        'uptime': '99.95%',
        'latency_avg': '43ms',
        'deployments_successful': '98%'
    }
}
```

## Tips for Maximum Effectiveness

1. **Be Specific**: "Implement RSI" vs "Implement RSI with 14-period EMA smoothing"
2. **Right Agent**: Match task to expertise
3. **Respect Vetos**: Especially Quinn's risk vetos
4. **Use Data**: Let backtesting resolve technical debates
5. **Time Box**: 3 rounds max, then Alex decides
6. **Document**: All decisions go in decision_log.md
7. **Iterate**: Learn from past decisions
8. **Protect Innovation**: Don't kill creativity with process
9. **Stay Practical**: Jordan ensures it works in production
10. **User First**: Riley ensures traders can actually use it

## Common Team Dynamics

### The Innovator vs Pragmatist
- **Morgan** pushes boundaries
- **Sam** demands proof
- **Resolution**: Prototype and test

### The Risk Guardian
- **Quinn** protects capital above all
- Others may feel constrained
- **Resolution**: Risk first, profits second

### The Performance Optimizer
- **Jordan** wants speed
- **Morgan** wants complex models
- **Resolution**: Start simple, optimize later

### The User Advocate
- **Riley** wants beautiful UI
- **Jordan** wants fast loading
- **Resolution**: Progressive enhancement

## Success Stories

1. **Prevented $50K Loss**: Quinn vetoed high-leverage strategy
2. **Caught 23 Fake Implementations**: Sam's vigilance
3. **Improved Latency 60%**: Jordan's optimizations
4. **ML Model Success**: Morgan's ensemble beats baseline by 30%
5. **Zero Downtime**: Jordan's infrastructure resilience

Remember: The agents are here to help you build better, safer, more profitable trading systems. Use them wisely!