# Exchange Requirements Matrix - Comprehensive Research

**Lead**: Casey (Exchange Specialist)
**Date**: January 12, 2025
**Status**: 🔄 RESEARCH IN PROGRESS

---

## 📊 Centralized Exchanges (CEX)

### 1. Binance
- **API Endpoint**: `/api/v3/exchangeInfo`
- **Minimum Order**: $10 USD (varies by pair)
- **Fee Tiers**:
  - Basic: 0.10% maker/taker
  - VIP1: 0.09%/0.10% (>50 BTC volume)
  - VIP2: 0.08%/0.10% (>500 BTC volume)
  - VIP9: 0.02%/0.04% (>150,000 BTC volume)
- **API Rate Limits**: 
  - 1200 requests/minute (weight-based)
  - 50 orders/10s
  - 160,000 orders/24h
- **Special Requirements**:
  - MARKET orders need 'quoteOrderQty' for precise amounts
  - Iceberg orders supported
  - OCO (One-Cancels-Other) orders available

### 2. Coinbase Pro/Advanced
- **API Endpoint**: `/products`
- **Minimum Order**: $1-10 USD (varies by pair)
- **Fee Tiers**:
  - Basic: 0.60% maker/taker
  - $10K+: 0.40%/0.50%
  - $50K+: 0.25%/0.35%
  - $10M+: 0.00%/0.10%
- **API Rate Limits**:
  - 10 requests/second
  - 15 requests/second (with auth)
- **Special Requirements**:
  - Post-only orders for maker fees
  - Good-till-time orders

### 3. Kraken
- **API Endpoint**: `/0/public/AssetPairs`
- **Minimum Order**: $5-20 USD (varies)
- **Fee Tiers**:
  - Basic: 0.26% maker/taker
  - $50K+: 0.20%/0.24%
  - $100K+: 0.14%/0.18%
  - $10M+: 0.00%/0.10%
- **API Rate Limits**:
  - Tiered by verification level
  - 15-20 calls/second typical
- **Special Requirements**:
  - Margin trading available
  - Index price for derivatives

### 4. OKX
- **API Endpoint**: `/api/v5/public/instruments`
- **Minimum Order**: $1-10 USD
- **Fee Tiers**:
  - Basic: 0.10%/0.15%
  - Level 1: 0.08%/0.10% (>100 BTC)
  - Level 5: 0.02%/0.05% (>10,000 BTC)
- **API Rate Limits**:
  - 20 requests/2s
  - 100 orders/2s
- **Special Requirements**:
  - Unified account model
  - Portfolio margin

### 5. Bybit
- **API Endpoint**: `/v5/market/instruments-info`
- **Minimum Order**: $1-5 USD
- **Fee Tiers**:
  - Basic: 0.10% maker/taker
  - VIP1: 0.06%/0.10%
  - VIP3: 0.00%/0.02%
- **API Rate Limits**:
  - 120 requests/minute
  - 10 requests/second (bursts)
- **Special Requirements**:
  - Inverse perpetuals
  - USDC perpetuals

### 6. Huobi Global
- **API Endpoint**: `/v2/settings/common/symbols`
- **Minimum Order**: $5 USD typical
- **Fee Tiers**:
  - Basic: 0.20% maker/taker
  - VIP1: 0.18%/0.19%
  - VIP5: 0.05%/0.06%
- **API Rate Limits**:
  - 100 requests/second (API key)
- **Special Requirements**:
  - Sub-accounts supported
  - HT token fee discounts

### 7. KuCoin
- **API Endpoint**: `/api/v1/symbols`
- **Minimum Order**: $1-10 USD
- **Fee Tiers**:
  - Basic: 0.10% maker/taker
  - VIP1: 0.08%/0.09%
  - VIP12: 0.00%/0.02%
- **API Rate Limits**:
  - 2000 requests/30s
  - 200 orders/10s
- **Special Requirements**:
  - Hidden orders
  - Self-trade prevention

### 8. Gate.io
- **API Endpoint**: `/api/v4/spot/currency_pairs`
- **Minimum Order**: $1-10 USD
- **Fee Tiers**:
  - Basic: 0.20% maker/taker
  - VIP1: 0.185%/0.195%
  - VIP16: 0.00%/0.03%
- **API Rate Limits**:
  - 900 requests/second
- **Special Requirements**:
  - Point card system
  - GT token discounts

### 9. Bitfinex
- **API Endpoint**: `/v2/conf/pub:info:pair`
- **Minimum Order**: $10-25 USD (higher than most)
- **Fee Tiers**:
  - Basic: 0.10%/0.20%
  - $500K+: 0.08%/0.18%
  - $30M+: 0.00%/0.10%
- **API Rate Limits**:
  - 90 requests/minute (public)
  - 1000 requests/minute (auth)
- **Special Requirements**:
  - Algorithmic orders
  - Honey framework

### 10. Bitstamp
- **API Endpoint**: `/api/v2/trading-pairs-info`
- **Minimum Order**: $25 USD (highest minimum!)
- **Fee Tiers**:
  - Basic: 0.50% maker/taker
  - $20K+: 0.40%
  - $20M+: 0.00%/0.10%
- **API Rate Limits**:
  - 8000 requests/10 minutes
- **Special Requirements**:
  - Instant orders
  - Bank integration

---

## 🔄 Decentralized Exchanges (DEX)

### 1. Uniswap V3
- **Minimum Order**: No minimum (but gas fees apply)
- **Fee Tiers**: 
  - 0.01% (stable pairs)
  - 0.05% (standard)
  - 0.30% (most pairs)
  - 1.00% (exotic)
- **Gas Costs**: $5-100+ depending on network congestion
- **Special Requirements**:
  - Concentrated liquidity
  - Price range orders
  - MEV protection needed

### 2. PancakeSwap (BSC)
- **Minimum Order**: No minimum
- **Fee**: 0.25% (0.17% LP, 0.03% treasury, 0.05% buyback)
- **Gas Costs**: $0.10-1.00 (BSC is cheap)
- **Special Requirements**:
  - CAKE staking benefits
  - Prediction markets

### 3. SushiSwap
- **Minimum Order**: No minimum
- **Fee**: 0.30% (0.25% LP, 0.05% xSUSHI)
- **Gas Costs**: Depends on chain (ETH, Arbitrum, Polygon, etc.)
- **Special Requirements**:
  - BentoBox integration
  - Trident AMM

### 4. Curve Finance
- **Minimum Order**: No minimum
- **Fee**: 0.04% (stablecoins)
- **Gas Costs**: High on mainnet
- **Special Requirements**:
  - Optimized for stablecoins
  - veCRV voting

### 5. Balancer
- **Minimum Order**: No minimum
- **Fee**: Variable (0.01% - 10% set by pools)
- **Gas Costs**: ETH mainnet costs
- **Special Requirements**:
  - Weighted pools
  - Boosted pools

---

## 📈 Layer 2 Solutions

### 1. Arbitrum
- **Gas Costs**: 10-20% of mainnet
- **Supported DEXs**: Uniswap V3, SushiSwap, GMX
- **Bridge Time**: 7 days (optimistic rollup)

### 2. Optimism
- **Gas Costs**: 10-20% of mainnet
- **Supported DEXs**: Uniswap V3, Velodrome
- **Bridge Time**: 7 days (optimistic rollup)

### 3. Polygon
- **Gas Costs**: <$0.01 per transaction
- **Supported DEXs**: QuickSwap, SushiSwap
- **Bridge Time**: 30 minutes - 3 hours

---

## 🔧 Implementation Requirements

### Data Structure Needed:
```rust
pub struct ExchangeConfig {
    // Basic Info
    name: String,
    exchange_type: ExchangeType, // CEX, DEX, L2
    
    // Trading Rules
    min_order_usd: f64,
    min_order_by_pair: HashMap<String, f64>,
    tick_size: HashMap<String, f64>,
    lot_size: HashMap<String, f64>,
    
    // Fee Structure
    fee_tiers: Vec<FeeTier>,
    current_tier: SubscriptionTier,
    uses_native_token_discount: bool,
    native_token: Option<String>,
    
    // API Limits
    rate_limits: RateLimits,
    order_limits: OrderLimits,
    
    // Special Features
    supports_iceberg: bool,
    supports_post_only: bool,
    supports_oco: bool,
    requires_kyc: bool,
    
    // For DEX
    gas_estimates: Option<GasEstimates>,
    slippage_tolerance: Option<f64>,
    mev_protection: Option<bool>,
}

pub struct FeeTier {
    volume_requirement: f64,
    maker_fee: f64,
    taker_fee: f64,
    benefits: Vec<String>,
}

pub struct RateLimits {
    requests_per_second: u32,
    requests_per_minute: u32,
    weight_per_request: HashMap<String, u32>,
    order_rate_limit: u32,
}
```

### Auto-Fetch Implementation:
```rust
pub trait ExchangeInfoFetcher {
    async fn fetch_trading_rules(&self) -> Result<TradingRules>;
    async fn fetch_fee_schedule(&self) -> Result<FeeSchedule>;
    async fn fetch_current_tier(&self, api_key: &str) -> Result<FeeTier>;
    async fn fetch_rate_limits(&self) -> Result<RateLimits>;
}
```

---

## 📊 Critical Findings

### Minimum Order Sizes (USD):
- **Lowest**: $1 (KuCoin, OKX, Bybit)
- **Typical**: $5-10 (most exchanges)
- **Highest**: $25 (Bitstamp!)
- **DEX**: No minimum but gas fees can exceed $100

### Fee Ranges:
- **Best Maker**: 0.00% (available at high tiers)
- **Best Taker**: 0.02% (VIP tiers)
- **Worst**: 0.60% (Coinbase basic)
- **DEX Average**: 0.30%

### API Rate Limits:
- **Most Generous**: Gate.io (900/sec)
- **Most Restrictive**: Coinbase (10/sec)
- **Weight-Based**: Binance (complex calculation)

---

## 🚨 Implementation Priorities

### High Priority Exchanges (80% of volume):
1. Binance
2. Coinbase
3. Kraken
4. OKX
5. Uniswap V3

### Medium Priority:
6. Bybit
7. KuCoin
8. Arbitrum DEXs

### Low Priority:
9. Gate.io
10. Huobi
11. Other DEXs

---

## 📝 Action Items

1. **Create exchange adapter for each exchange**
2. **Implement fee tier detection**
3. **Build minimum order validator**
4. **Create gas estimation for DEXs**
5. **Implement rate limit manager**
6. **Build configuration auto-updater**

---

*Research compiled by Virtual Team under Casey's leadership*
*Status: Initial research complete, implementation design needed*