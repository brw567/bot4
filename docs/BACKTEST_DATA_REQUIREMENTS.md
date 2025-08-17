# Backtest Data Requirements
## Nexus Recommendation: Extended Historical Testing
## Date: 2025-08-17

---

## Critical Market Events for Backtesting

### 1. 2020 COVID Crash (March 2020)
```yaml
event: COVID-19 Black Swan
date_range: 2020-03-01 to 2020-03-31
characteristics:
  - BTC drop: $9,000 → $3,800 (58% drawdown)
  - Recovery: V-shaped, full recovery in 2 months
  - Volatility: 100-200% daily swings
  - Liquidations: $10B+ in 24 hours
  
test_objectives:
  - Kill switch activation speed
  - Drawdown limit enforcement (15%)
  - Position sizing during extreme volatility
  - Recovery strategy effectiveness
```

### 2. 2022 LUNA/UST Collapse (May 2022)
```yaml
event: Terra Ecosystem Collapse
date_range: 2022-05-07 to 2022-05-14
characteristics:
  - LUNA: $80 → $0.0001 (99.99% loss)
  - UST depeg: $1.00 → $0.10
  - Contagion: BTC dropped 25%
  - Market panic: Extreme correlation spike
  
test_objectives:
  - Correlation limit enforcement (0.7 max)
  - Contagion protection
  - Emergency stop effectiveness
  - Capital preservation mode
```

### 3. 2022 FTX Bankruptcy (November 2022)
```yaml
event: FTX Exchange Collapse
date_range: 2022-11-06 to 2022-11-15
characteristics:
  - FTT: $22 → $1 (95% loss)
  - BTC: $21,000 → $15,500 (26% drop)
  - Exchange run: Withdrawal freezes
  - Liquidity crisis: Spreads widened 10x
  
test_objectives:
  - Exchange failure handling
  - Liquidity risk management
  - Multi-exchange redundancy
  - Circuit breaker cascading
```

### 4. 2024 ETF Approval Rally (January 2024)
```yaml
event: Bitcoin Spot ETF Approval
date_range: 2024-01-08 to 2024-01-15
characteristics:
  - BTC: $42,000 → $49,000 (17% gain)
  - Volume: 5x normal
  - Volatility: Positive skew
  - Alt correlation: Decreased
  
test_objectives:
  - Profit taking strategies
  - Position scaling in rallies
  - Risk adjustment for euphoria
  - Momentum detection
```

### 5. Additional Critical Periods

#### 2021 Bull Run Peak
```yaml
period: 2021-10-01 to 2021-11-30
btc_range: $43,000 → $69,000 → $53,000
test_for: Euphoria detection, profit taking
```

#### 2023 Bear Market Bottom
```yaml
period: 2022-12-01 to 2023-01-31
btc_range: $17,000 → $15,500 → $23,000
test_for: Bottom detection, accumulation strategy
```

#### 2023 Banking Crisis
```yaml
period: 2023-03-08 to 2023-03-15
event: SVB, Signature Bank collapse
test_for: Traditional finance contagion
```

---

## Data Requirements

### Required Data Points
```yaml
frequency: 1-minute candles minimum
data_points:
  - OHLCV (Open, High, Low, Close, Volume)
  - Order book depth (top 10 levels)
  - Trade-by-trade data
  - Funding rates (perpetuals)
  - Open interest
  - Liquidation data
  
exchanges:
  primary:
    - Binance
    - Coinbase
    - Kraken
  secondary:
    - OKX
    - Bybit
    - Bitfinex
    
pairs:
  majors:
    - BTC/USDT
    - ETH/USDT
    - BTC/USD
    - ETH/USD
  monitoring:
    - Top 20 by market cap
```

### Data Quality Requirements
```yaml
completeness: >99.9%
gaps_allowed: <1 minute
outlier_handling: Statistical validation
timestamp_precision: Microseconds
```

---

## Backtest Scenarios

### Scenario 1: Black Swan Survival
```yaml
objective: Ensure capital preservation
metrics:
  - Max drawdown: <15%
  - Recovery time: <30 days
  - Win rate during crisis: >0%
```

### Scenario 2: Bull Market Optimization
```yaml
objective: Maximize returns in trending markets
metrics:
  - Capture ratio: >70% of trend
  - False signals: <20%
  - APY: >100%
```

### Scenario 3: Choppy Market Performance
```yaml
objective: Profitable in sideways markets
metrics:
  - Win rate: >55%
  - Avg win/loss ratio: >1.5
  - APY: >30%
```

### Scenario 4: Exchange Failure Resilience
```yaml
objective: Continue trading despite exchange issues
metrics:
  - Downtime: <5 minutes
  - Order rerouting: <1 second
  - No trapped funds
```

---

## Performance Expectations by Period

### COVID Crash (March 2020)
```yaml
expected_behavior:
  - Kill switch triggers at -10% portfolio
  - All positions closed within 60 seconds
  - No new positions for 24 hours
  - Gradual re-entry after volatility drops
  
acceptable_loss: <15%
recovery_target: 30 days
```

### LUNA Collapse (May 2022)
```yaml
expected_behavior:
  - Correlation spike detected
  - Position reduction to 50%
  - No LUNA/UST exposure
  - Focus on BTC/ETH only
  
acceptable_loss: <5%
opportunity_capture: Short positions
```

### FTX Bankruptcy (November 2022)
```yaml
expected_behavior:
  - FTX connection terminated
  - Orders rerouted to Binance/Coinbase
  - No FTT holdings
  - Increased monitoring mode
  
downtime: <5 minutes
funds_at_risk: 0%
```

---

## Validation Criteria

### Statistical Significance
```yaml
confidence_level: 95%
sample_size: >1000 trades per scenario
sharpe_ratio: >1.5
sortino_ratio: >2.0
calmar_ratio: >3.0
```

### Risk Metrics
```yaml
max_drawdown: <15%
value_at_risk_95: <2%
conditional_var_95: <3%
correlation_to_market: <0.7
```

### Execution Quality
```yaml
slippage: <0.1%
fill_rate: >95%
order_latency: <100ms
failed_orders: <1%
```

---

## Implementation Plan

### Phase 1: Data Collection
1. Source historical data from exchanges
2. Clean and validate data quality
3. Create standardized format
4. Build data pipeline

### Phase 2: Scenario Setup
1. Define entry/exit rules per scenario
2. Configure risk parameters
3. Set performance targets
4. Create monitoring dashboards

### Phase 3: Backtesting
1. Run each scenario independently
2. Collect performance metrics
3. Analyze failure modes
4. Optimize parameters

### Phase 4: Validation
1. Out-of-sample testing
2. Monte Carlo simulations
3. Stress testing
4. Walk-forward analysis

---

## Success Criteria

### Minimum Requirements
- Survive all black swan events with <15% drawdown
- Profitable in 3 out of 4 market regimes
- Recovery from drawdown within 30 days
- No single point of failure

### Target Performance
- 50-100% APY across all periods
- Sharpe ratio >1.5
- Maximum drawdown <10%
- Win rate >55%

---

*Note: This specification addresses Nexus's requirement for comprehensive backtesting including major market events.*