// xAI/GROK-3 MINI ENHANCED PROMPTS - DEEP DIVE IMPLEMENTATION
// Team: Morgan (Lead) + Alex - INSTITUTIONAL-GRADE INSIGHTS AT RETAIL COST!
// Target: <600ms latency with Grok-3 Mini, 186 tokens/sec output

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;

/// Enhanced prompt templates leveraging game theory and advanced trading concepts
pub struct EnhancedPromptTemplates {
    pub templates: HashMap<String, String>,
}

impl EnhancedPromptTemplates {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        // PROMPT 1: Nash Equilibrium Market Analysis
        templates.insert("nash_equilibrium_analysis".to_string(), r#"
Analyze the current market using Nash Equilibrium game theory principles.

MARKET DATA:
- Asset: {asset}
- Current Price: ${price}
- 24h Volume: ${volume}
- Order Book Imbalance: {imbalance}%
- Bid/Ask Spread: ${spread}
- Large Trader Positions (>$1M): {whale_count}
- Retail Flow Direction: {retail_flow}

GAME THEORY ANALYSIS REQUIRED:
1. NASH EQUILIBRIUM IDENTIFICATION:
   - What is the current Nash equilibrium price level where no participant benefits from unilateral deviation?
   - Are we at equilibrium or in transition?
   - Expected time to reach new equilibrium: ___

2. PLAYER STRATEGIES:
   - Market Makers: What's their optimal spread given current volatility?
   - Whales: Are they accumulating or distributing? What's their optimal strategy?
   - Retail: Following or countering institutional flow?
   - Arbitrageurs: What inefficiencies are they exploiting?

3. PAYOFF MATRIX:
   Create a 2x2 payoff matrix for:
   - Bulls vs Bears
   - Expected payoffs for each strategy combination
   - Dominant strategies if any

4. PRISONER'S DILEMMA SCENARIOS:
   - Are traders in a prisoner's dilemma (all selling causes crash but first seller wins)?
   - Coordination problems detected?
   - Trust breakdown indicators?

5. MEAN FIELD GAME DYNAMICS:
   - As N→∞ traders, what's the limiting behavior?
   - Herding effects strength (0-1): ___
   - Information cascade probability: ___

OUTPUT FORMAT:
{
  "equilibrium_price": float,
  "current_state": "equilibrium|transition|chaos",
  "dominant_strategy": "buy|sell|hold",
  "nash_deviation_profit": float,
  "coordination_failure_risk": float,
  "recommended_action": string,
  "confidence": float
}
"#.to_string());

        // PROMPT 2: Kyle's Lambda & Market Microstructure
        templates.insert("market_microstructure_analysis".to_string(), r#"
Perform advanced market microstructure analysis using Kyle's Lambda and Glosten-Milgrom models.

MICROSTRUCTURE DATA:
- Tick Size: ${tick_size}
- Average Trade Size: ${avg_trade_size}
- Trade Frequency: {trades_per_second} trades/sec
- Quote Updates: {quotes_per_second} quotes/sec
- Effective Spread: ${effective_spread}
- Realized Spread: ${realized_spread}
- Price Impact (Kyle's λ): {kyle_lambda}

DEPTH DATA:
- Level 1 Bid/Ask: {bid_1}/{ask_1} @ {bid_size_1}/{ask_size_1}
- Level 2 Bid/Ask: {bid_2}/{ask_2} @ {bid_size_2}/{ask_size_2}
- Level 5 Depth Imbalance: {depth_5_imbalance}%
- Hidden Liquidity Estimate: ${hidden_liquidity}

ANALYZE:
1. INFORMED TRADING PROBABILITY (PIN):
   - Using Easley-O'Hara PIN model
   - Estimate % of informed vs noise traders
   - Information asymmetry score (0-1): ___

2. ADVERSE SELECTION COMPONENT:
   - Glosten-Milgrom spread decomposition
   - Adverse selection cost: $___
   - Inventory holding cost: $___
   - Order processing cost: $___

3. PRICE DISCOVERY METRICS:
   - Hasbrouck information share: ___
   - Weighted price contribution: ___
   - Lead-lag relationship with other venues: ___

4. OPTIMAL EXECUTION STRATEGY:
   - Given Kyle's λ = {kyle_lambda}
   - Optimal order size to minimize impact: ___
   - Optimal execution horizon: ___ seconds
   - Use TWAP/VWAP/IS/POV? ___

5. LIQUIDITY PROVISION OPPORTUNITY:
   - Expected spread capture: $___
   - Adverse selection risk: ___
   - Optimal quote placement: Bid___ Ask___

OUTPUT FORMAT:
{
  "pin_score": float,
  "informed_trader_percentage": float,
  "kyle_lambda": float,
  "optimal_trade_size": float,
  "execution_strategy": "aggressive|passive|mixed",
  "liquidity_provision_edge": float,
  "microstructure_alpha": float
}
"#.to_string());

        // PROMPT 3: Behavioral Finance & Sentiment Extraction
        templates.insert("behavioral_sentiment_analysis".to_string(), r#"
Extract behavioral finance signals and sentiment using advanced NLP and market psychology.

SENTIMENT SOURCES:
- Twitter/X Mentions (last hour): {twitter_mentions}
- Sentiment Score: {twitter_sentiment} (-1 to 1)
- Reddit WSB Activity: {wsb_posts}/hour
- Discord/Telegram Chatter Volume: {chat_volume}
- Google Trends Score: {google_trends}
- Fear & Greed Index: {fear_greed}

KEY INFLUENCER POSTS:
{influencer_posts}

NEWS HEADLINES (last 4 hours):
{news_headlines}

BEHAVIORAL ANALYSIS REQUIRED:
1. COGNITIVE BIASES DETECTED:
   - Confirmation Bias Strength: ___ (0-1)
   - Anchoring to Price: ${anchor_price}
   - Recency Bias Impact: ___
   - Herding Behavior Score: ___
   - FOMO/FUD Level: ___

2. PROSPECT THEORY APPLICATION:
   - Reference Point: $___
   - Loss Aversion Coefficient: ___ (typically 2.25)
   - Probability Weighting Distortion: ___
   - Expected Utility vs Prospect Theory Value: ___

3. NARRATIVE ECONOMICS (Shiller):
   - Dominant Narrative: ___
   - Narrative Virality (R₀): ___
   - Counter-Narrative Strength: ___
   - Narrative Exhaustion Timeline: ___

4. REFLEXIVITY (Soros Theory):
   - Fundamental vs Perception Gap: ___
   - Self-Reinforcing Trend Strength: ___
   - Bubble/Crash Probability: ___
   - Turning Point Indicators: ___

5. ATTENTION ECONOMICS:
   - Attention Share vs Market Cap Ratio: ___
   - Attention Arbitrage Opportunity: ___
   - Peak Attention ETA: ___

OUTPUT FORMAT:
{
  "behavioral_bias_score": float,
  "sentiment_extremity": float,
  "narrative_strength": float,
  "reflexivity_gap": float,
  "contrarian_signal": boolean,
  "crowd_wisdom_reliability": float,
  "manipulation_probability": float
}
"#.to_string());

        // PROMPT 4: Cross-Asset Correlation & Regime Detection
        templates.insert("regime_correlation_analysis".to_string(), r#"
Perform cross-asset correlation analysis and market regime detection using advanced econometrics.

ASSET CORRELATIONS (30-day rolling):
- BTC-S&P500: {btc_spy_corr}
- BTC-Gold: {btc_gold_corr}
- BTC-DXY: {btc_dxy_corr}
- BTC-VIX: {btc_vix_corr}
- BTC-10Y Yield: {btc_10y_corr}

REGIME INDICATORS:
- Volatility Regime (GARCH σ): {volatility_regime}
- Correlation Regime (DCC-GARCH): {correlation_regime}
- Volume Regime: {volume_regime}
- Liquidity Regime: {liquidity_regime}

MACRO FACTORS:
- Real Rates: {real_rates}%
- Inflation Expectations: {inflation_exp}%
- Central Bank Policy: {cb_policy}
- Geopolitical Risk Index: {geopolitical_risk}

ANALYZE:
1. MARKOV REGIME SWITCHING:
   - Current Regime: Risk-On|Risk-Off|Transition
   - Regime Persistence Probability: ___
   - Expected Regime Duration: ___ days
   - Regime Change Triggers: ___

2. CORRELATION BREAKDOWN DETECTION:
   - Correlation Instability Score: ___
   - Contagion Risk Level: ___
   - Diversification Benefit Lost: ___%
   - Flight-to-Quality Probability: ___

3. FACTOR DECOMPOSITION:
   - Market Beta: ___
   - Crypto-Specific Alpha: ___
   - Macro Factor Loading: ___
   - Idiosyncratic Risk: ___%

4. TAIL DEPENDENCE (Copula):
   - Lower Tail Dependence: ___
   - Upper Tail Dependence: ___
   - Extreme Event Correlation: ___
   - Black Swan Probability: ___

5. OPTIMAL PORTFOLIO ADJUSTMENTS:
   - Increase/Decrease Crypto Allocation: ___%
   - Hedge Recommendations: ___
   - Correlation Trade Opportunities: ___

OUTPUT FORMAT:
{
  "current_regime": string,
  "regime_confidence": float,
  "regime_change_probability": float,
  "correlation_stability": float,
  "tail_risk_score": float,
  "portfolio_adjustment": object,
  "cross_asset_opportunities": array
}
"#.to_string());

        // PROMPT 5: Options Flow & Institutional Positioning
        templates.insert("options_institutional_analysis".to_string(), r#"
Analyze options flow and institutional positioning for directional insights.

OPTIONS DATA:
- Put/Call Ratio: {pc_ratio}
- IV Rank (30-day): {iv_rank}%
- IV Percentile: {iv_percentile}%
- Term Structure Slope: {term_structure}
- Skew (25δ-75δ): {skew}
- Butterfly Spread: {butterfly}

LARGE OPTIONS TRADES (>$100K):
{large_options_trades}

INSTITUTIONAL INDICATORS:
- Futures Open Interest: ${futures_oi}
- Funding Rate: {funding_rate}%
- Basis Trade: {basis}%
- CME vs Spot Premium: {cme_premium}%
- Grayscale Premium/Discount: {grayscale}%

ANALYZE:
1. DEALER POSITIONING (via Gamma):
   - Net Dealer Gamma: $___
   - Gamma Flip Point: $___
   - Charm (Gamma decay): ___
   - Vanna (IV sensitivity): ___

2. SMART MONEY FLOW:
   - Unusual Options Activity Score: ___
   - Institutional Accumulation/Distribution: ___
   - Options Flow Sentiment: ___
   - Max Pain Theory Price: $___

3. VOLATILITY ARBITRAGE:
   - Implied vs Realized Spread: ___%
   - Volatility Risk Premium: ___
   - Optimal Volatility Trade: ___
   - Dispersion Trade Opportunity: ___

4. PINNING & MANIPULATION:
   - Option Pinning Probability: ___
   - Manipulation via Gamma: ___
   - Stop Hunt Zones: $___-$___
   - Liquidation Cascade Risk: ___

5. STRATEGIC RECOMMENDATIONS:
   - Directional Bias from Flow: ___
   - Optimal Strike/Expiry: ___
   - Risk Reversal Signal: ___
   - Calendar Spread Opportunity: ___

OUTPUT FORMAT:
{
  "dealer_positioning": "long_gamma|short_gamma|neutral",
  "smart_money_direction": "bullish|bearish|neutral",
  "gamma_flip_level": float,
  "max_pain_price": float,
  "institutional_bias": float,
  "vol_arb_opportunity": float,
  "manipulation_risk": float
}
"#.to_string());

        // PROMPT 6: Machine Learning Feature Importance
        templates.insert("ml_feature_analysis".to_string(), r#"
Analyze which features are most predictive using interpretable ML techniques.

TOP 20 FEATURES BY SHAP VALUE:
{feature_importance_list}

MODEL PERFORMANCE:
- Accuracy: {accuracy}%
- Sharpe Ratio: {sharpe}
- Max Drawdown: {max_dd}%
- Win Rate: {win_rate}%
- Profit Factor: {profit_factor}

FEATURE CATEGORIES:
- Technical: {technical_features}
- Microstructure: {microstructure_features}
- Sentiment: {sentiment_features}
- On-chain: {onchain_features}
- Macro: {macro_features}

ANALYZE:
1. FEATURE INTERACTIONS:
   - Top 3 interaction effects: ___
   - Non-linear relationships: ___
   - Threshold effects: ___
   - Regime-dependent features: ___

2. TEMPORAL IMPORTANCE:
   - Features important at different time horizons
   - Short-term (1-5min): ___
   - Medium-term (1-4hr): ___
   - Long-term (1-7day): ___

3. CAUSAL INFERENCE:
   - Likely causal features: ___
   - Spurious correlations detected: ___
   - Confounding variables: ___
   - Treatment effects: ___

4. FEATURE ENGINEERING IDEAS:
   - Missing interactions: ___
   - Polynomial features needed: ___
   - Fourier features for cycles: ___
   - Embedding suggestions: ___

5. MODEL IMPROVEMENTS:
   - Ensemble recommendations: ___
   - Feature selection strategy: ___
   - Regularization adjustments: ___
   - Cross-validation strategy: ___

OUTPUT FORMAT:
{
  "critical_features": array,
  "feature_interactions": array,
  "temporal_importance": object,
  "engineering_suggestions": array,
  "model_improvements": array,
  "expected_performance_gain": float
}
"#.to_string());

        // PROMPT 7: Real-time Event Impact Assessment
        templates.insert("realtime_event_impact".to_string(), r#"
URGENT: Assess immediate market impact of breaking event.

EVENT DETAILS:
Type: {event_type}
Time Since Event: {minutes_elapsed} minutes
Description: {event_description}
Source Credibility: {credibility}/10
Confirmation Sources: {confirmations}

INITIAL MARKET REACTION:
- Price Change: {price_change}%
- Volume Spike: {volume_spike}x
- Volatility Change: {vol_change}%
- Social Mentions Spike: {social_spike}x
- Google Trends Spike: {trends_spike}

SIMILAR HISTORICAL EVENTS:
{historical_comparisons}

RAPID ASSESSMENT NEEDED:
1. IMMEDIATE IMPACT (0-1 hour):
   - Expected Price Movement: ___%
   - Volatility Expansion: ___%
   - Liquidation Cascade Risk: ___
   - Support/Resistance Levels: ___

2. SHORT-TERM (1-24 hours):
   - Trend Continuation Probability: ___
   - Mean Reversion Level: $___
   - Key Levels to Watch: ___
   - Sentiment Exhaustion Point: ___

3. INFORMATION DIFFUSION:
   - Current Awareness %: ___
   - Full Diffusion ETA: ___ hours
   - Smart vs Dumb Money Ratio: ___
   - Overreaction Probability: ___

4. GAME THEORY RESPONSE:
   - Optimal Strategy Now: ___
   - Strategy After 1hr: ___
   - Exit Strategy: ___
   - Risk Management: ___

5. FALSE SIGNAL DETECTION:
   - Manipulation Probability: ___
   - Fake News Risk: ___
   - Pump & Dump Pattern: ___
   - Verification Needed: ___

OUTPUT FORMAT (SPEED CRITICAL):
{
  "impact_magnitude": "minor|moderate|major|extreme",
  "directional_bias": "bullish|bearish|neutral",
  "action_required": "buy|sell|hold|wait",
  "confidence": float,
  "key_levels": array,
  "time_sensitivity": "immediate|urgent|moderate|low",
  "follow_up_needed": boolean
}
"#.to_string());

        Self { templates }
    }
    
    /// Get prompt with variable substitution
    pub fn get_prompt(&self, template_name: &str, variables: &HashMap<String, String>) -> String {
        let template = self.templates.get(template_name)
            .expect("Template not found");
        
        let mut prompt = template.clone();
        for (key, value) in variables {
            prompt = prompt.replace(&format!("{{{}}}", key), value);
        }
        
        prompt
    }
    
    /// Get optimal prompt based on market conditions
    pub fn select_optimal_prompt(&self, market_state: &MarketState) -> &str {
        match market_state {
            MarketState::HighVolatility => "realtime_event_impact",
            MarketState::LowLiquidity => "market_microstructure_analysis",
            MarketState::TrendingStrong => "behavioral_sentiment_analysis",
            MarketState::RangeB


ound => "nash_equilibrium_analysis",
            MarketState::RegimeChange => "regime_correlation_analysis",
            MarketState::HighOptionsVolume => "options_institutional_analysis",
            MarketState::Normal => "ml_feature_analysis",
        }
    }
}

#[derive(Debug, Clone)]
pub enum MarketState {
    HighVolatility,
    LowLiquidity,
    TrendingStrong,
    RangeBound,
    RegimeChange,
    HighOptionsVolume,
    Normal,
}

/// Grok-3 Mini configuration optimized for speed
#[derive(Debug, Clone)]
pub struct Grok3MiniConfig {
    pub model: String,
    pub reasoning_effort: ReasoningEffort,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub response_format: ResponseFormat,
    pub stream: bool,  // Enable streaming for faster perceived response
}

impl Default for Grok3MiniConfig {
    fn default() -> Self {
        Self {
            model: "grok-3-mini-fast".to_string(),  // Use fast variant
            reasoning_effort: ReasoningEffort::Low,  // Low for <600ms response
            max_tokens: 500,  // Limit for speed
            temperature: 0.3,  // Lower for consistency
            top_p: 0.9,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            response_format: ResponseFormat::Json,
            stream: true,  // Stream for faster first token
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReasoningEffort {
    Low,   // Minimal thinking for speed
    High,  // Maximum thinking for complex problems
}

#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Text,
    Json,
}

/// Prompt optimization strategies
#[derive(Debug)]
pub struct PromptOptimizer {
    pub use_few_shot: bool,
    pub use_chain_of_thought: bool,
    pub use_self_consistency: bool,
    pub parallel_prompts: usize,
}

impl PromptOptimizer {
    /// Create optimizer for real-time trading
    pub fn for_realtime_trading() -> Self {
        Self {
            use_few_shot: true,  // Include examples for consistency
            use_chain_of_thought: false,  // Skip for speed
            use_self_consistency: false,  // Skip multiple runs for speed
            parallel_prompts: 3,  // Run top 3 prompts in parallel
        }
    }
    
    /// Create optimizer for deep analysis
    pub fn for_deep_analysis() -> Self {
        Self {
            use_few_shot: true,
            use_chain_of_thought: true,
            use_self_consistency: true,
            parallel_prompts: 7,  // Run all prompts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prompt_templates() {
        let templates = EnhancedPromptTemplates::new();
        assert_eq!(templates.templates.len(), 7);
        assert!(templates.templates.contains_key("nash_equilibrium_analysis"));
    }
    
    #[test]
    fn test_variable_substitution() {
        let templates = EnhancedPromptTemplates::new();
        let mut vars = HashMap::new();
        vars.insert("asset".to_string(), "BTC/USDT".to_string());
        vars.insert("price".to_string(), "98000".to_string());
        
        let prompt = templates.get_prompt("nash_equilibrium_analysis", &vars);
        assert!(prompt.contains("BTC/USDT"));
        assert!(prompt.contains("98000"));
    }
}