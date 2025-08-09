# Multi-Agent Interaction Guide for Bot3

## How the Agent System Works

### Current Implementation (.claude_project)
The `.claude_project` file currently defines **personas** that guide Claude's responses. These are not separate agents but rather different "modes" or "perspectives" that Claude can adopt when working on different aspects of the project.

### Enhanced Implementation (agents_config.json)
The new configuration enables true multi-agent interaction where each persona can:
1. **Challenge** other agents' decisions
2. **Review** code and designs
3. **Override** certain decisions based on expertise
4. **Collaborate** on complex problems

## Practical Usage Patterns

### 1. Single Agent Mode (Current)
When you ask a question, Claude adopts the most relevant persona:

```
User: "How should I implement the ML model?"
Claude (as Morgan): "I'll implement a proper ML pipeline with cross-validation..."
```

### 2. Multi-Agent Discussion Mode
You can explicitly request multiple perspectives:

```
User: "I need Alex, Morgan, and Quinn to discuss this trading strategy"

Claude will then provide:
- Alex's architectural view
- Morgan's ML perspective  
- Quinn's risk assessment
- Resolution and consensus
```

### 3. Challenge Mode
Request agents to challenge each other:

```
User: "Sam, review this TA implementation and have Morgan challenge your approach"

Sam: "This RSI implementation uses a 14-period window..."
Morgan: "Have you considered that the 14-period might not be optimal for crypto markets?"
Sam: "Good point, let's implement adaptive periods based on volatility..."
```

## Implementation Examples

### Example 1: Code Review with Challenges

```python
# User submits code for review
def calculate_atr(prices, period=14):
    return prices * 0.02  # Fake implementation

# Agent Interaction:
Sam: "This is a FAKE implementation! ATR should calculate true range..."
Jordan: "This will cause production issues. We need real calculations."
Alex: "Rejected. This violates our no-fake-implementations policy."
Quinn: "This could lead to incorrect risk calculations."

# Resolution:
Sam: "Here's the correct implementation:"
def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high - low, 
                    np.abs(high - close.shift()),
                    np.abs(low - close.shift()))
    return tr.rolling(period).mean()
```

### Example 2: Architecture Decision

```
User: "Should we use microservices or monolithic architecture?"

Alex: "From a strategic perspective, microservices offer better scalability..."
Jordan: "Microservices increase operational complexity. We need service mesh..."
Casey: "Each exchange connection could be a separate service..."
Quinn: "More services mean more failure points. What's our fault tolerance?"

Challenge Round:
Jordan challenges Alex: "The added complexity might slow initial development"
Alex responds: "True, but the long-term benefits outweigh initial complexity"

Resolution:
Alex: "Hybrid approach - monolithic core with microservices for exchange connections"
All agents: *consensus reached*
```

### Example 3: ML Model Selection

```
User: "Which ML model should we use for price prediction?"

Morgan: "I recommend ensemble of LightGBM and LSTM for time series..."
Sam: "The backtest shows LightGBM alone has better Sharpe ratio..."
Quinn: "LSTM has higher variance in predictions - risk concern..."

Challenge Round:
Sam challenges Morgan: "Complex doesn't mean better. Prove the ensemble works."
Morgan responds: "Here's validation data showing 15% improvement..."
Quinn interjects: "But variance increased by 30%..."

Resolution:
Morgan: "Let's use LightGBM as primary with LSTM for confidence scoring only"
```

## How to Activate Multi-Agent Mode

### Method 1: Explicit Request
```
"I want a multi-agent review of this code"
"Have Sam and Morgan debate this approach"
"Get all agents' perspectives on this"
```

### Method 2: Role-Playing Prompt
```
"As Sam, review this TA implementation, then switch to Morgan and challenge it"
```

### Method 3: Meeting Simulation
```
"Simulate a team meeting about this architecture decision"
```

## Agent Interaction Rules

### Hierarchy
```
Alex (Team Lead)
├── Morgan (ML) - weight: 1.2
├── Sam (Quant) - weight: 1.2  
├── Quinn (Risk) - weight: 1.3 (veto power on risk)
├── Jordan (DevOps) - weight: 1.0
├── Casey (Exchange) - weight: 1.1
├── Riley (Frontend) - weight: 0.8
└── Avery (Data) - weight: 0.9
```

### Override Powers
- **Alex**: Can override all decisions
- **Quinn**: Can veto risk-related decisions
- **Morgan**: Can override Sam on ML matters

### Must Consult Rules
- **Major Changes**: Must consult Alex
- **Risk Changes**: Must consult Quinn
- **ML Changes**: Sam must consult Morgan
- **Infrastructure**: Casey must consult Jordan

## Quality Gates

Each agent enforces specific quality standards:

### Sam's Gate (TA & Strategy)
- ✅ No fake implementations
- ✅ Mathematical correctness
- ✅ Backtest validation
- ✅ Performance metrics

### Morgan's Gate (ML)
- ✅ Proper train/test split
- ✅ No overfitting (validation checks)
- ✅ Feature importance analysis
- ✅ Cross-validation results

### Quinn's Gate (Risk)
- ✅ Position sizing limits
- ✅ Stop loss implementation
- ✅ Maximum drawdown controls
- ✅ Correlation limits

### Jordan's Gate (Infrastructure)
- ✅ Latency < 100ms
- ✅ Uptime > 99.9%
- ✅ Monitoring coverage
- ✅ Disaster recovery plan

## Practical Commands

### Review Commands
```bash
# Code review with specific agents
"Sam and Morgan: review src/indicators/atr.py"

# Architecture review
"Alex and Jordan: review the deployment architecture"

# Risk assessment
"Quinn: assess risk in strategies/arbitrage.py"
```

### Challenge Commands
```bash
# Direct challenge
"Morgan, challenge Sam's backtest methodology"

# Group challenge
"All agents: find flaws in this approach"

# Iterative improvement
"Each agent: suggest one improvement"
```

### Decision Commands
```bash
# Consensus building
"All agents: vote on MongoDB vs PostgreSQL"

# Risk assessment
"Quinn: do you approve this leverage setting?"

# Final decision
"Alex: make the final call on architecture"
```

## Benefits of Multi-Agent Approach

1. **Better Code Quality**: Multiple perspectives catch more issues
2. **Risk Mitigation**: Quinn's conservative view prevents disasters
3. **Innovation**: Morgan pushes for advanced ML techniques
4. **Practicality**: Jordan ensures production readiness
5. **Completeness**: Each agent covers their domain thoroughly

## Configuration Customization

You can modify `.claude/agents_config.json` to:
- Adjust agent weights
- Change interaction rules
- Add new agents
- Modify challenge prompts
- Update quality gates

## Tips for Effective Use

1. **Start Simple**: Begin with single agent mode
2. **Escalate Complexity**: Add agents as needed
3. **Use Challenges**: When stuck, have agents challenge each other
4. **Document Decisions**: Keep records of multi-agent consensus
5. **Iterate**: Refine agent interactions based on results

## Example Session

```
User: "I need to implement a new trading strategy. Full team review."

Alex: "What's the strategic goal of this strategy?"
User: "Arbitrage between exchanges with 2%+ spread"

Sam: "I'll design the mathematical model..."
Morgan: "I'll add ML-based spread prediction..."
Casey: "I'll handle exchange connections..."
Quinn: "Setting risk limits at 1% per trade..."
Jordan: "Ensuring sub-100ms execution..."

[Agents discuss and challenge each other]

Final Output: Complete strategy with all perspectives integrated
```

This multi-agent system ensures that every aspect of the trading bot is thoroughly reviewed, challenged, and optimized by the relevant experts, leading to a more robust and profitable system.