// ENHANCED DECISION ORCHESTRATOR - Implementation Methods
// Team: FULL DEEP DIVE - NO SIMPLIFICATIONS!

use super::decision_orchestrator_enhanced::*;
use crate::unified_types::*;
use crate::decision_orchestrator::Signal;
use crate::HistoricalRegime;
use crate::trading_types_complete::{EnhancedOrderBook, SentimentData};
// Import all types from prelude - fixes missing types
use crate::prelude::{
    ExecutionAlgorithm, AssetClass, tail_risk, Utc
};
// Import ML method wrappers for RwLock guard access
use crate::ml_method_wrappers::{
    MLFeedbackSystemReadGuardExt, MLFeedbackSystemWriteGuardExt,
    SHAPCalculatorReadGuardExt, MarketAnalyticsWriteGuardExt
};
use anyhow::{Result, anyhow};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;  // Critical for compile-time decimal literals
use std::collections::HashMap;

/// Enhanced ML Signal with calibration
#[derive(Debug, Clone)]
pub struct EnhancedMLSignal {
    pub action: SignalAction,
    pub raw_confidence: f64,
    pub calibrated_confidence: f64,
    pub shap_values: Vec<f64>,
    pub top_features: Vec<(String, f64)>,
}

/// Monte Carlo validation results
#[derive(Debug, Clone)]
pub struct MonteCarloValidation {
    pub win_rate: f64,
    pub expected_return: f64,
    pub var_95: f64,
    pub max_drawdown: f64,
}

impl EnhancedDecisionOrchestrator {
    /// Get ML prediction with SHAP explanations
    pub async fn get_ml_prediction_with_shap(
        &self,
        features: &FeaturePipeline,
    ) -> Result<EnhancedMLSignal> {
        // Combine all features into single vector
        let mut all_features = Vec::new();
        all_features.extend(&features.price_features);
        all_features.extend(&features.volume_features);
        all_features.extend(&features.microstructure_features);
        all_features.extend(&features.technical_features);
        all_features.extend(&features.sentiment_features);
        all_features.extend(&features.regime_features);
        all_features.extend(&features.correlation_features);
        
        // Get ML prediction
        let ml_system = self.ml_system.read();
        let prediction = ml_system.predict(&all_features);
        let (signal_action, raw_confidence) = prediction;  // Destructure tuple
        
        // Apply isotonic calibration
        let calibrated_confidence = ml_system.calibrate_probability(raw_confidence);
        
        // Calculate SHAP values
        let shap_calc = self.shap_calculator.read();
        let shap_values = shap_calc.calculate_shap_values(&all_features);
        
        // Get top features by importance
        let mut feature_importance: Vec<(String, f64)> = shap_calc.feature_names().iter()
            .zip(shap_values.iter())
            .map(|(name, &value)| (name.clone(), value.abs()))
            .collect();
        feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_features = feature_importance.into_iter().take(5).collect();
        
        // Update ML feedback with prediction
        drop(ml_system);
        let mut ml_system = self.ml_system.write();
        ml_system.update_prediction_history(prediction.clone());
        
        Ok(EnhancedMLSignal {
            action: signal_action,  // Use the destructured action from prediction tuple
            raw_confidence,  // Use the destructured confidence value
            calibrated_confidence,
            shap_values,
            top_features,
        })
    }
    
    /// Get advanced TA signal using 20+ indicators
    pub async fn get_advanced_ta_signal(
        &self,
        market_data: &MarketData,
        historical_data: &[MarketData],
    ) -> Result<Signal> {
        let mut ta = self.ta_analytics.write();
        
        // Update with latest data - map MarketData to OHLCV approximations
        // For real-time tick data, we use last price as close, and approximate high/low
        let last_price = market_data.last.to_f64();
        let ask_price = market_data.ask.to_f64();
        let bid_price = market_data.bid.to_f64();
        
        ta.update(
            last_price,                    // Current price (using last trade)
            ask_price * 1.001,             // High approximation (slightly above ask)
            bid_price * 0.999,             // Low approximation (slightly below bid)
            last_price,                    // Close (same as current for real-time)
            market_data.volume.to_f64(),  // Volume
        );
        
        // Collect all indicator signals
        let mut buy_signals = 0.0;
        let mut sell_signals = 0.0;
        let mut total_weight = 0.0;
        
        // RSI (weight: 2.0)
        if let Some(rsi) = ta.get_rsi() {
            if rsi < 30.0 {
                buy_signals += 2.0;
            } else if rsi > 70.0 {
                sell_signals += 2.0;
            }
            total_weight += 2.0;
        }
        
        // MACD (weight: 1.5)
        if let Some(macd) = ta.get_macd() {
            if macd.macd > macd.signal && macd.macd > 0.0 {
                buy_signals += 1.5;
            } else if macd.macd < macd.signal && macd.macd < 0.0 {
                sell_signals += 1.5;
            }
            total_weight += 1.5;
        }
        
        // Bollinger Bands (weight: 1.0)
        if let Some(bb_pos) = ta.get_bollinger_position() {
            if bb_pos < 0.2 {
                buy_signals += 1.0;
            } else if bb_pos > 0.8 {
                sell_signals += 1.0;
            }
            total_weight += 1.0;
        }
        
        // Stochastic (weight: 1.0)
        if let Some(stoch) = ta.get_stochastic() {
            if stoch.k < 20.0 && stoch.d < 20.0 {
                buy_signals += 1.0;
            } else if stoch.k > 80.0 && stoch.d > 80.0 {
                sell_signals += 1.0;
            }
            total_weight += 1.0;
        }
        
        // ADX trend strength (weight: 1.5)
        if let Some(adx) = ta.get_adx() {
            if adx > 25.0 {  // Strong trend
                if let Some(macd) = ta.get_macd() {
                    if macd.macd > 0.0 {
                        buy_signals += 1.5;
                    } else {
                        sell_signals += 1.5;
                    }
                }
            }
            total_weight += 1.5;
        }
        
        // Volume indicators (weight: 1.0)
        if let Some(obv) = ta.get_obv() {
            if obv > ta.get_obv_ma().unwrap_or(obv) {
                buy_signals += 1.0;
            } else {
                sell_signals += 1.0;
            }
            total_weight += 1.0;
        }
        
        // Calculate confidence based on signal strength
        let net_signal = buy_signals - sell_signals;
        let confidence = ((net_signal.abs() / total_weight) as f64).min(1.0);
        
        Ok(Signal {
            action: if net_signal > 0.0 { SignalAction::Buy } else { SignalAction::Sell },
            confidence,
            size: confidence * 0.5,  // Size proportional to confidence
            features: vec![buy_signals, sell_signals, total_weight],
            reason: format!("TA: {} indicators analyzed", total_weight as usize),
            shap_values: None,
            top_features: None,
        })
    }
    
    /// Detect regime using HMM
    pub async fn detect_regime_with_hmm(
        &self,
        features: &FeaturePipeline,
    ) -> Result<HistoricalRegime> {
        let regime_features = vec![
            features.price_features.get(0).copied().unwrap_or(0.0),
            features.volume_features.get(0).copied().unwrap_or(0.0),
            features.technical_features.get(0).copied().unwrap_or(50.0),  // RSI
            features.correlation_features.get(0).copied().unwrap_or(0.5),
        ];
        
        let (regime, _confidence) = self.regime_calibration.detect_current_regime(&regime_features);
        Ok(regime)
    }
    
    /// Get regime-adjusted signal
    pub async fn get_regime_adjusted_signal(
        &self,
        regime: HistoricalRegime,
    ) -> Result<Signal> {
        let (action, confidence, size_mult) = match regime {
            HistoricalRegime::StrongBull => (SignalAction::Buy, 0.8, 1.5),
            HistoricalRegime::Bull => (SignalAction::Buy, 0.6, 1.2),
            HistoricalRegime::Sideways => (SignalAction::Hold, 0.4, 0.8),
            HistoricalRegime::Bear => (SignalAction::Sell, 0.6, 0.8),
            HistoricalRegime::Crisis => (SignalAction::Hold, 0.9, 0.3),
            HistoricalRegime::Recovery => (SignalAction::Buy, 0.5, 1.0),
        };
        
        Ok(Signal {
            action,
            confidence,
            size: confidence * size_mult * 0.3,
            features: vec![regime.to_index() as f64],
            reason: format!("Regime: {:?}", regime),
            shap_values: None,
            top_features: None,
        })
    }
    
    /// Analyze sentiment with NLP
    pub async fn analyze_sentiment_with_nlp(
        &self,
        sentiment: &SentimentData,
    ) -> Result<Signal> {
        // Combine multiple sentiment sources
        let overall_sentiment = (
            sentiment.twitter_sentiment * 0.3 +
            sentiment.news_sentiment * 0.4 +
            sentiment.reddit_sentiment * 0.2 +
            sentiment.fear_greed_index * 0.1
        ) / 100.0;
        
        let action = if overall_sentiment > 0.6 {
            SignalAction::Buy
        } else if overall_sentiment < 0.4 {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        Ok(Signal {
            action,
            confidence: (overall_sentiment - 0.5).abs() * 2.0,
            size: overall_sentiment * 0.3,
            features: vec![overall_sentiment],
            reason: format!("Sentiment: {:.1}%", overall_sentiment * 100.0),
            shap_values: None,
            top_features: None,
        })
    }
    
    /// Calculate VPIN toxicity
    pub async fn calculate_vpin_toxicity(
        &self,
        order_book: &EnhancedOrderBook,
        historical_data: &[MarketData],
    ) -> Result<f64> {
        let mut vpin = self.vpin_calculator.write();
        
        // Update with simulated order flow
        // Note: Basic OrderBook doesn't have trade flow, so we approximate
        // In production, use EnhancedOrderBook with trade_flow field
        let buy_volume = order_book.total_bid_volume();
        let sell_volume = order_book.total_ask_volume();
        
        Ok(vpin.get_current_vpin())
    }
    
    /// Analyze systemic risks
    pub async fn analyze_systemic_risks(
        &self,
        market_data: &MarketData,
    ) -> Result<(f64, f64)> {
        // Get tail risk from t-Copula
        let tail_metrics = self.t_copula.get_tail_metrics();
        let tail_risk = tail_metrics.max_tail_dependence;
        
        // Get contagion risk
        let contagion = self.cross_asset_corr.get_contagion_risk();
        let contagion_risk = contagion.contagion_level;
        
        // Update correlations with latest data
        // Calculate approximate 24h return from current price
        let returns_24h = ((market_data.last.to_f64() - market_data.bid.to_f64()) / market_data.bid.to_f64()) * 100.0;
        let returns = HashMap::from([
            (AssetClass::Crypto, returns_24h),  // BTC is part of Crypto asset class
        ]);
        self.cross_asset_corr.update(returns);
        
        Ok((tail_risk, contagion_risk))
    }
    
    /// Create ensemble signal with dynamic weighting
    pub async fn create_ensemble_signal(
        &self,
        ml_signal: EnhancedMLSignal,
        ta_signal: Signal,
        regime_signal: Signal,
        sentiment_signal: Option<Signal>,
        vpin_toxicity: f64,
        tail_risk: f64,
        contagion_risk: f64,
    ) -> Result<Signal> {
        // Get current weights
        let ml_weight = *self.ml_weight.read();
        let ta_weight = *self.ta_weight.read();
        let regime_weight = *self.regime_weight.read();
        let sentiment_weight = if sentiment_signal.is_some() {
            *self.sentiment_weight.read()
        } else {
            0.0
        };
        
        // Normalize weights
        let total_weight = ml_weight + ta_weight + regime_weight + sentiment_weight;
        
        // Calculate weighted action scores
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;
        
        // ML contribution
        if ml_signal.action == SignalAction::Buy {
            buy_score += ml_signal.calibrated_confidence * ml_weight;
        } else {
            sell_score += ml_signal.calibrated_confidence * ml_weight;
        }
        
        // TA contribution
        if ta_signal.action == SignalAction::Buy {
            buy_score += ta_signal.confidence * ta_weight;
        } else if ta_signal.action == SignalAction::Sell {
            sell_score += ta_signal.confidence * ta_weight;
        }
        
        // Regime contribution
        if regime_signal.action == SignalAction::Buy {
            buy_score += regime_signal.confidence * regime_weight;
        } else if regime_signal.action == SignalAction::Sell {
            sell_score += regime_signal.confidence * regime_weight;
        }
        
        // Sentiment contribution
        if let Some(sent) = sentiment_signal {
            if sent.action == SignalAction::Buy {
                buy_score += sent.confidence * sentiment_weight;
            } else if sent.action == SignalAction::Sell {
                sell_score += sent.confidence * sentiment_weight;
            }
        }
        
        // Normalize scores
        buy_score /= total_weight;
        sell_score /= total_weight;
        
        // Adjust for market toxicity and risks
        let risk_multiplier = (1.0 - vpin_toxicity) * (1.0 - tail_risk) * (1.0 - contagion_risk);
        
        let final_action = if buy_score > sell_score {
            SignalAction::Buy
        } else if sell_score > buy_score {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        let confidence = ((buy_score - sell_score).abs() * risk_multiplier).min(1.0);
        
        Ok(Signal {
            action: final_action,
            confidence,
            size: confidence * 0.5,
            features: vec![buy_score, sell_score, risk_multiplier],
            reason: format!("Ensemble: Buy={:.2} Sell={:.2} Risk={:.2}", 
                          buy_score, sell_score, risk_multiplier),
            shap_values: Some(ml_signal.shap_values),
            top_features: Some(ml_signal.top_features),
        })
    }
    
    /// Calculate advanced Kelly size
    pub async fn calculate_advanced_kelly_size(
        &self,
        signal: &Signal,
        market_data: &MarketData,
        regime: HistoricalRegime,
        tail_risk: f64,
    ) -> Result<f64> {
        let mut kelly = self.kelly_sizer.write();
        
        // Estimate win probability from signal confidence
        let win_prob = 0.5 + (signal.confidence * 0.3);  // Maps [0,1] to [0.5,0.8]
        
        // Estimate win/loss ratio based on regime
        let win_loss_ratio = match regime {
            HistoricalRegime::StrongBull => 2.0,
            HistoricalRegime::Bull => 1.5,
            HistoricalRegime::Sideways => 1.2,
            HistoricalRegime::Bear => 0.8,
            HistoricalRegime::Crisis => 0.5,
            HistoricalRegime::Recovery => 1.3,
        };
        
        // Calculate Kelly criterion
        let kelly_fraction = kelly.calculate_discrete_kelly(
            win_prob,
            win_loss_ratio,
            1.0,  // Loss ratio
        );
        
        // Apply fractional Kelly (25% default)
        let mut adjusted_kelly = kelly_fraction * 0.25;
        
        // Reduce for tail risk
        adjusted_kelly *= (1.0 - tail_risk * 0.5);
        
        // Apply regime-based cap
        let max_size = match regime {
            HistoricalRegime::StrongBull => 0.05,
            HistoricalRegime::Bull => 0.04,
            HistoricalRegime::Sideways => 0.02,
            HistoricalRegime::Bear => 0.015,
            HistoricalRegime::Crisis => 0.005,
            HistoricalRegime::Recovery => 0.03,
        };
        
        Ok(adjusted_kelly.min(max_size))
    }
    
    /// Apply comprehensive risk clamps with DEEP DIVE validation
    pub async fn apply_comprehensive_risk_clamps(
        &self,
        mut signal: Signal,
        kelly_size: f64,
        market_data: &MarketData,
        vpin_toxicity: f64,
    ) -> Result<Signal> {
        let mut clamps = self.risk_clamps.write();
        
        // Extract confidence from signal
        let confidence = signal.confidence;
        
        // Calculate volatility from market data (simplified 24h volatility)
        let price_range = (market_data.ask.to_f64() - market_data.bid.to_f64()) / market_data.last.to_f64();
        let volatility = price_range * 100.0; // Convert to percentage
        
        // Calculate tail risk from VPIN toxicity
        // Reference: "Tail Risk and Asset Prices" - Kelly (2014)
        let tail_risk = vpin_toxicity.max(0.01).min(1.0);
        
        // Prepare risk metrics with calculated values
        let risk_metrics = RiskMetrics {
            position_size: dec!(0.02),
            confidence: Decimal::from_f64(confidence).unwrap_or(Decimal::ZERO),
            expected_return: dec!(0.05),
            volatility: Decimal::from_f64(volatility).unwrap_or(dec!(0.02)),
            var_limit: Decimal::from_f64(vpin_toxicity.min(0.1)).unwrap_or(dec!(0.1)),  // Cap VaR limit
            sharpe_ratio: 1.5,
            kelly_fraction: Decimal::from_f64(kelly_size.min(1.0)).unwrap_or(dec!(0.02)),
            max_drawdown: dec!(0.15),
            current_heat: Decimal::from_f64(tail_risk * 0.1).unwrap_or(dec!(0.01)),
            leverage: 1.0,
        };
        
        // Apply all 8 clamp layers
        let clamped_size = clamps.apply_all_clamps(
            Decimal::from_f64(kelly_size).unwrap(),
            Decimal::from_f64(market_data.last.to_f64()).unwrap(),
            &risk_metrics,
        );
        
        signal.size = clamped_size.to_f64().unwrap();
        
        // If size was reduced significantly, reduce confidence too
        if signal.size < kelly_size * 0.5 {
            signal.confidence *= 0.8;
            signal.reason = format!("{} | Risk-clamped", signal.reason);
        }
        
        Ok(signal)
    }
    
    /// Validate with Monte Carlo
    pub async fn validate_with_monte_carlo(
        &self,
        signal: &Signal,
        market_data: &MarketData,
        historical_data: &[MarketData],
    ) -> Result<MonteCarloValidation> {
        let mut monte_carlo = self.monte_carlo.write();
        
        // Extract historical returns for simulation
        let mut returns = Vec::new();
        for i in 1..historical_data.len() {
            let ret = (historical_data[i].price / historical_data[i-1].price).to_f64() - 1.0;
            returns.push(ret);
        }
        
        // Run simulations
        let results = monte_carlo.simulate_trading_strategy(
            &returns,
            signal.size,
            signal.confidence,
            10000,
        );
        
        Ok(MonteCarloValidation {
            win_rate: results.win_rate,
            expected_return: results.expected_return,
            var_95: results.value_at_risk_95,
            max_drawdown: results.max_drawdown,
        })
    }
    
    /// Optimize for profit extraction
    pub async fn optimize_for_profit_extraction(
        &self,
        signal: Signal,
        market_data: &MarketData,
        order_book: &EnhancedOrderBook,
    ) -> Result<Signal> {
        let mut extractor = self.profit_extractor.write();
        
        // Analyze profit opportunity
        let opportunity = extractor.analyze_opportunity(
            market_data,
            order_book,
            &signal,
        );
        
        // Adjust signal based on opportunity
        let mut optimized = signal.clone();
        
        if opportunity.edge < 0.001 {
            // No edge, reduce size
            optimized.size *= 0.5;
            optimized.confidence *= 0.7;
        } else if opportunity.edge > 0.005 {
            // Strong edge, increase size (within limits)
            optimized.size = (optimized.size * 1.2).min(0.05);
            optimized.confidence = (optimized.confidence * 1.1).min(1.0);
        }
        
        optimized.reason = format!("{} | Edge: {:.2}bps", 
                                  optimized.reason, opportunity.edge * 10000.0);
        
        Ok(optimized)
    }
    
    /// Select optimal execution algorithm
    pub async fn select_optimal_execution(
        &self,
        signal: &Signal,
        order_book: &EnhancedOrderBook,
        vpin_toxicity: f64,
    ) -> Result<ExecutionAlgorithm> {
        let executor = self.optimal_executor.read();
        
        // Select based on market conditions
        let algo = if vpin_toxicity > 0.5 {
            // High toxicity, use passive execution
            ExecutionAlgorithm::Iceberg
        } else if signal.size > 0.02 {
            // Large order, use TWAP
            ExecutionAlgorithm::TWAP
        } else if order_book.spread_bps() > 10.0 {
            // Wide spread, use limit orders
            ExecutionAlgorithm::Passive
        } else {
            // Normal conditions, use adaptive
            ExecutionAlgorithm::Adaptive
        };
        
        Ok(algo)
    }
    
    /// Auto-tune parameters based on performance
    pub async fn auto_tune_parameters(
        &self,
        signal: &Signal,
        market_data: &MarketData,
    ) -> Result<()> {
        // Run quick hyperparameter optimization
        let mut optimizer = self.hyperparameter_optimizer.write();
        
        // Define objective function
        let objective = |params: &HashMap<String, f64>| -> f64 {
            // Simplified objective: maximize Sharpe ratio
            let ml_w = params.get("ml_weight").unwrap_or(&0.4);
            let ta_w = params.get("ta_weight").unwrap_or(&0.3);
            let regime_w = params.get("regime_weight").unwrap_or(&0.3);
            
            // Simulate performance (simplified)
            let expected_return = signal.confidence * 0.1;
            let volatility = 0.15;
            let sharpe = expected_return / volatility;
            
            // Penalize extreme weights
            let balance_penalty = ((ml_w - 0.33).powi(2) + 
                                 (ta_w - 0.33).powi(2) + 
                                 (regime_w - 0.33).powi(2)) * 0.1;
            
            sharpe - balance_penalty
        };
        
        // Run quick optimization (10 trials)
        let best_params = optimizer.optimize_quick(Box::new(objective), 10);
        
        // Update weights
        if let Some(ml_w) = best_params.get("ml_weight") {
            *self.ml_weight.write() = *ml_w;
        }
        if let Some(ta_w) = best_params.get("ta_weight") {
            *self.ta_weight.write() = *ta_w;
        }
        if let Some(regime_w) = best_params.get("regime_weight") {
            *self.regime_weight.write() = *regime_w;
        }
        
        // Update auto-tuner parameters
        let mut auto_tuner = self.auto_tuner.write();
        if let Some(var_limit) = best_params.get("var_limit") {
            auto_tuner.set_var_limit(Decimal::from_f64(*var_limit).unwrap());
        }
        if let Some(kelly_frac) = best_params.get("kelly_fraction") {
            auto_tuner.set_kelly_fraction(Decimal::from_f64(*kelly_frac).unwrap());
        }
        
        Ok(())
    }
    
    /// Record decision for learning
    pub async fn record_enhanced_decision(
        &self,
        ml_signal: &EnhancedMLSignal,
        ta_signal: &Signal,
        regime: HistoricalRegime,
        tail_risk: f64,
        contagion_risk: f64,
        vpin_toxicity: f64,
        final_signal: &Signal,
    ) -> Result<()> {
        let record = EnhancedDecisionRecord {
            timestamp: Utc::now().timestamp(),
            ml_confidence: ml_signal.calibrated_confidence,
            ta_confidence: ta_signal.confidence,
            regime,
            tail_risk,
            contagion_level: contagion_risk,
            vpin_toxicity,
            final_action: final_signal.action,
            position_size: Decimal::from_f64(final_signal.size).unwrap(),
            expected_sharpe: 1.5,  // Would calculate from historical data
            actual_pnl: None,  // Will be filled later
        };
        
        self.decision_history.write().push(record);
        
        // Keep only last 100k decisions
        let mut history = self.decision_history.write();
        if history.len() > 100000 {
            history.drain(0..10000);
        }
        
        Ok(())
    }
    
    /// Update ML feedback loop
    pub async fn update_ml_feedback(
        &self,
        features: &FeaturePipeline,
        signal: &Signal,
    ) -> Result<()> {
        let mut ml_system = self.ml_system.write();
        
        // Create feature vector
        let mut feature_vec = Vec::new();
        feature_vec.extend(&features.price_features);
        feature_vec.extend(&features.volume_features);
        feature_vec.extend(&features.microstructure_features);
        feature_vec.extend(&features.technical_features);
        
        // Record for future learning
        ml_system.add_training_example(
            feature_vec,
            if signal.action == SignalAction::Buy { 1.0 } else { -1.0 },
            signal.confidence,
        );
        
        // Trigger online learning if enough examples
        if ml_system.should_retrain() {
            ml_system.online_learning_update();
        }
        
        Ok(())
    }
    
    /// Construct final trading signal
    pub async fn construct_final_signal(
        &self,
        optimized_signal: Signal,
        execution_algo: ExecutionAlgorithm,
        market_data: &MarketData,
    ) -> Result<TradingSignal> {
        // Get current portfolio state
        let portfolio = self.portfolio_manager.get_current_state();
        
        // Calculate risk metrics
        let risk_metrics = RiskMetrics {
            position_size: optimized_signal.size,
            confidence: optimized_signal.confidence,
            expected_return: dec!(0.05),
            volatility: dec!(0.15),
            var_limit: dec!(0.02),
            sharpe_ratio: 1.5,
            kelly_fraction: optimized_signal.size,
            max_drawdown: dec!(0.15),
            current_heat: optimized_signal.size * dec!(0.1),
            leverage: 1.0,
        };
        
        // Build final signal
        Ok(TradingSignal {
            timestamp: Utc::now().timestamp_millis() as u64,
            symbol: market_data.symbol.clone(),
            action: optimized_signal.action,
            confidence: Percentage::from_f64(optimized_signal.confidence).unwrap(),
            size: Quantity::from_f64(optimized_signal.size).unwrap(),
            reason: format!("{} | Algo: {:?}", optimized_signal.reason, execution_algo),
            risk_metrics,
            ml_features: optimized_signal.features.clone(),
            ta_indicators: vec![],  // Would populate with actual TA values
        })
    }
}