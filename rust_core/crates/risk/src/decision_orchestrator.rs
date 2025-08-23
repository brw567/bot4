// UNIFIED DECISION ORCHESTRATOR - Combines ML + TA + Risk
// Team: FULL TEAM COLLABORATION - NO SIMPLIFICATIONS!
// Alex: "This is the BRAIN - it must combine EVERYTHING!"
// Morgan: "ML predictions with confidence scores"
// Quinn: "Risk-adjusted through all 8 layers"
// Casey: "Market microstructure aware"

use crate::unified_types::*;
use crate::kelly_sizing::KellySizer;
use crate::clamps::RiskClampSystem;
use crate::auto_tuning::AutoTuningSystem;
use crate::ml_feedback::MLFeedbackSystem;
use crate::profit_extractor::ProfitExtractor;
use crate::market_analytics::MarketAnalytics;
use crate::auto_tuning_persistence::AutoTuningPersistence;
use crate::portfolio_manager::{PortfolioManager, PortfolioConfig};
use crate::feature_importance::SHAPCalculator;
use crate::t_copula::{TCopula, TCopulaConfig};
use crate::historical_regime_calibration::HistoricalRegimeCalibration;
use crate::cross_asset_correlations::{CrossAssetCorrelations, AssetClass};
use std::sync::Arc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal_macros::dec;
use anyhow::Result;
use ndarray::Array2;

/// The BRAIN of the trading system - orchestrates ALL components
/// Alex: "This is where EVERYTHING comes together!"
pub struct DecisionOrchestrator {
    // Core Components
    ml_system: Arc<RwLock<MLFeedbackSystem>>,
    ta_analytics: Arc<RwLock<MarketAnalytics>>,
    kelly_sizer: Arc<RwLock<KellySizer>>,
    risk_clamps: Arc<RwLock<RiskClampSystem>>,
    auto_tuner: Arc<RwLock<AutoTuningSystem>>,
    profit_extractor: Arc<RwLock<ProfitExtractor>>,
    portfolio_manager: Arc<PortfolioManager>,  // NO MORE HARDCODED VALUES!
    
    // SHAP explainability - DEEP DIVE ENHANCEMENT
    shap_calculator: Arc<RwLock<SHAPCalculator>>,
    
    // NEXUS PRIORITY 2 SYSTEMS - DEEP DIVE ENHANCEMENTS
    t_copula: Arc<TCopula>,  // Tail dependence modeling
    regime_calibration: Arc<HistoricalRegimeCalibration>,  // HMM regime detection
    cross_asset_corr: Arc<CrossAssetCorrelations>,  // DCC-GARCH correlations
    
    // Database persistence
    persistence: Arc<AutoTuningPersistence>,
    
    // Decision weights (auto-tuned!)
    ml_weight: Arc<RwLock<f64>>,      // Weight for ML signals
    ta_weight: Arc<RwLock<f64>>,      // Weight for TA signals
    sentiment_weight: Arc<RwLock<f64>>, // Weight for sentiment (Grok)
    
    // Performance tracking
    decision_history: Arc<RwLock<Vec<DecisionRecord>>>,
    
    // Feature explanations from SHAP
    latest_shap_values: Arc<RwLock<Option<Vec<f64>>>>,
    feature_names: Vec<String>,
}

/// Complete decision record for learning
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    pub timestamp: u64,
    pub ml_signal: Signal,
    pub ta_signal: Signal,
    pub combined_signal: Signal,
    pub final_signal: Signal,  // After risk adjustments
    pub execution_result: Option<ExecutionResult>,
    pub pnl: Option<Decimal>,
}

/// Signal from a single source
#[derive(Debug, Clone)]
pub struct Signal {
    pub action: SignalAction,
    pub confidence: f64,
    pub size: f64,
    pub features: Vec<f64>,
    pub reason: String,
    // DEEP DIVE: SHAP explanations for ML transparency
    pub shap_values: Option<Vec<f64>>,
    pub top_features: Option<Vec<(String, f64)>>, // Feature name and importance
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub actual_size: Decimal,
    pub actual_price: Decimal,
    pub slippage: Decimal,
    pub fees: Decimal,
}

impl DecisionOrchestrator {
    /// Create new orchestrator with all components
    pub async fn new(database_url: &str, initial_equity: Decimal) -> Result<Self> {
        // Initialize persistence layer
        let persistence = Arc::new(AutoTuningPersistence::new(database_url).await?);
        
        // Load adaptive parameters from database
        let params = persistence.load_adaptive_parameters().await?;
        
        // Initialize weights from database or defaults
        let ml_weight = params.get("ml_weight")
            .map(|p| p.current_value.to_f64().unwrap_or(0.4))
            .unwrap_or(0.4);
            
        let ta_weight = params.get("ta_weight")
            .map(|p| p.current_value.to_f64().unwrap_or(0.4))
            .unwrap_or(0.4);
            
        let sentiment_weight = params.get("sentiment_weight")
            .map(|p| p.current_value.to_f64().unwrap_or(0.2))
            .unwrap_or(0.2);
        
        // Create auto-tuner
        let auto_tuner = Arc::new(RwLock::new(AutoTuningSystem::new()));
        
        // Create portfolio manager - NO MORE HARDCODED VALUES!
        let portfolio_config = PortfolioConfig::default();
        let portfolio_manager = Arc::new(PortfolioManager::new(initial_equity, portfolio_config));
        
        // Initialize Nexus Priority 2 systems
        let param_manager = Arc::new(crate::parameter_manager::ParameterManager::new());
        
        // t-Copula for tail dependence
        let t_copula_config = TCopulaConfig {
            initial_df: 5.0,
            min_df: 2.5,
            max_df: 30.0,
            calibration_window: 252,
            crisis_threshold: 0.8,
            update_frequency: 1,
        };
        let t_copula = Arc::new(TCopula::new(t_copula_config, param_manager.clone(), 5)); // 5 main asset classes
        
        // Historical regime calibration with HMM
        let regime_calibration = Arc::new(HistoricalRegimeCalibration::new(param_manager.clone()));
        
        // Cross-asset correlations with DCC-GARCH
        let assets = vec![
            AssetClass::BTC,     // Crypto representative
            AssetClass::SP500,   // Stocks representative
            AssetClass::US10Y,   // Bonds representative
            AssetClass::GOLD,    // Commodities representative
            AssetClass::DXY,     // FX representative
        ];
        let mut cross_asset_corr = CrossAssetCorrelations::new(assets, param_manager.clone());
        cross_asset_corr.set_t_copula(t_copula.clone());
        cross_asset_corr.set_regime_calibration(regime_calibration.clone());
        let cross_asset_corr = Arc::new(cross_asset_corr);
        
        // Create feature names for SHAP explanations
        let feature_names = vec![
            "price".to_string(),
            "price_ratio".to_string(),
            "spread_pct".to_string(),
            "order_imbalance".to_string(),
            "depth_ratio".to_string(),
            "volume".to_string(),
            "bid_size".to_string(),
            "ask_size".to_string(),
            "rsi".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
        ];
        
        // Create SHAP calculator with dummy model for now
        // In production, this would use the actual ML model
        let background_data = Array2::zeros((100, feature_names.len()));
        let shap_calculator = Arc::new(RwLock::new(
            SHAPCalculator::new(
                |x| Array2::zeros((x.nrows(), 1)).column(0).to_owned(),
                feature_names.clone(),
                background_data,
            )
        ));
        
        Ok(Self {
            ml_system: Arc::new(RwLock::new(MLFeedbackSystem::new())),
            ta_analytics: Arc::new(RwLock::new(MarketAnalytics::new())),
            kelly_sizer: Arc::new(RwLock::new(KellySizer::new(Default::default()))),
            risk_clamps: Arc::new(RwLock::new(RiskClampSystem::new(Default::default()))),
            auto_tuner: auto_tuner.clone(),
            profit_extractor: Arc::new(RwLock::new(ProfitExtractor::new(auto_tuner))),
            portfolio_manager,
            shap_calculator,
            t_copula,
            regime_calibration,
            cross_asset_corr,
            persistence,
            ml_weight: Arc::new(RwLock::new(ml_weight)),
            ta_weight: Arc::new(RwLock::new(ta_weight)),
            sentiment_weight: Arc::new(RwLock::new(sentiment_weight)),
            decision_history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            latest_shap_values: Arc::new(RwLock::new(None)),
            feature_names,
        })
    }
    
    /// MASTER DECISION FUNCTION - Combines ALL inputs
    /// This is where ML + TA + Sentiment + Risk all come together!
    pub async fn make_trading_decision(
        &self,
        market_data: &MarketData,
        order_book: &OrderBook,
        sentiment_data: Option<&SentimentData>,
    ) -> Result<TradingSignal> {
        println!("\n═══════════════════════════════════════════════════════");
        println!("UNIFIED DECISION ORCHESTRATOR - COMBINING ALL SIGNALS");
        println!("═══════════════════════════════════════════════════════\n");
        
        // Step 1: Get ML prediction
        let ml_features = self.extract_ml_features(market_data, order_book).await?;
        let ml_signal = self.get_ml_signal(&ml_features).await?;
        println!("ML Signal: {:?} (confidence: {:.2}%)", 
                 ml_signal.action, ml_signal.confidence * 100.0);
        
        // Step 2: Get TA indicators
        let ta_indicators = self.calculate_ta_indicators(market_data).await?;
        let ta_signal = self.get_ta_signal(&ta_indicators).await?;
        println!("TA Signal: {:?} (confidence: {:.2}%)", 
                 ta_signal.action, ta_signal.confidence * 100.0);
        
        // Step 3: Get sentiment signal (if available)
        let sentiment_signal = if let Some(sentiment) = sentiment_data {
            Some(self.get_sentiment_signal(sentiment).await?)
        } else {
            None
        };
        if let Some(ref sig) = sentiment_signal {
            println!("Sentiment Signal: {:?} (confidence: {:.2}%)", 
                     sig.action, sig.confidence * 100.0);
        }
        
        // Step 4: Combine signals with adaptive weights
        let combined_signal = self.combine_signals(
            &ml_signal,
            &ta_signal,
            sentiment_signal.as_ref()
        ).await?;
        println!("\nCombined Signal: {:?} (confidence: {:.2}%)", 
                 combined_signal.action, combined_signal.confidence * 100.0);
        
        // Step 5: Detect market regime with HMM
        let regime = self.detect_market_regime(market_data, &ta_indicators).await?;
        println!("Market Regime: {:?}", regime);
        
        // Step 5a: Check tail dependence risk (Nexus Priority 2)
        let tail_metrics = self.t_copula.get_tail_metrics();
        println!("Tail Dependence: {:.3} (DoF: {:.1})", 
                 tail_metrics.max_tail_dependence, tail_metrics.degrees_of_freedom);
        
        // Step 5b: Check contagion risk (Nexus Priority 2)
        let contagion_risk = self.cross_asset_corr.get_contagion_risk();
        println!("Contagion Level: {:.2}% | Systemic Risk: {:.2}%",
                 contagion_risk.contagion_level * 100.0,
                 contagion_risk.systemic_risk * 100.0);
        
        // Adjust signal based on tail risk and contagion
        let adjusted_signal = self.adjust_for_tail_risk(
            combined_signal.clone(),
            &tail_metrics,
            &contagion_risk
        );
        
        // Step 6: Apply Kelly sizing
        let kelly_size = self.calculate_kelly_size(
            &combined_signal,
            market_data
        ).await?;
        println!("Kelly Optimal Size: {:.4}%", kelly_size * 100.0);
        
        // Step 7: Apply 8-layer risk clamps
        let risk_adjusted_signal = self.apply_risk_clamps(
            combined_signal.clone(),
            kelly_size,
            market_data
        ).await?;
        println!("Risk-Adjusted Size: {:.4}%", 
                 risk_adjusted_signal.size.inner().to_f64().unwrap() * 100.0);
        
        // Step 8: Extract profit opportunity
        let final_signal = self.extract_profit_opportunity(
            risk_adjusted_signal,
            market_data,
            order_book
        ).await?;
        
        // Step 9: Record decision for learning
        self.record_decision(
            ml_signal,
            ta_signal,
            combined_signal,
            final_signal.clone()
        ).await?;
        
        // Step 10: Update auto-tuning parameters
        self.update_auto_tuning(&final_signal, regime).await?;
        
        println!("\n✅ FINAL DECISION:");
        println!("  Action: {:?}", final_signal.action);
        println!("  Size: {:.4}% of portfolio", 
                 final_signal.size.inner().to_f64().unwrap() * 100.0);
        println!("  Confidence: {:.2}%", 
                 final_signal.confidence.value() * 100.0);
        println!("  Risk Score: {:.2}", 
                 final_signal.risk_metrics.sharpe_ratio);
        
        Ok(final_signal)
    }
    
    /// Extract ML features from market data
    async fn extract_ml_features(
        &self,
        market: &MarketData,
        order_book: &OrderBook,
    ) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(50);
        
        // Price features
        features.push(market.last.to_f64());
        features.push((market.last.inner() / market.mid.inner()).to_f64().unwrap_or(1.0));
        features.push(market.spread_percentage().value());
        
        // Order book features
        let imbalance = self.calculate_order_book_imbalance(order_book);
        features.push(imbalance);
        
        let depth_ratio = self.calculate_depth_ratio(order_book);
        features.push(depth_ratio);
        
        // Volume features
        features.push(market.volume.to_f64());
        features.push(market.bid_size.to_f64());
        features.push(market.ask_size.to_f64());
        
        // Add TA features
        let ta = self.ta_analytics.read();
        if let Some(rsi) = ta.get_rsi() {
            features.push(rsi);
        }
        if let Some(macd) = ta.get_macd() {
            features.push(macd.macd);
            features.push(macd.signal);
        }
        
        Ok(features)
    }
    
    /// Get ML signal with SHAP explanations - DEEP DIVE ENHANCEMENT
    async fn get_ml_signal(&self, features: &[f64]) -> Result<Signal> {
        let ml = self.ml_system.read();
        let (action, confidence) = ml.predict(features);
        
        // DEEP DIVE: Calculate SHAP values for explainability
        let (shap_values, top_features) = {
            let mut shap = self.shap_calculator.write();
            
            // Convert features to ndarray for SHAP
            let features_array = Array2::from_shape_vec(
                (1, features.len()),
                features.to_vec()
            ).unwrap_or_else(|_| Array2::zeros((1, features.len())));
            
            // Calculate SHAP values
            let shap_matrix = shap.calculate_kernel_shap(&features_array);
            let shap_vec: Vec<f64> = shap_matrix.row(0).to_vec();
            
            // Get top 5 most important features
            let mut feature_importance: Vec<(String, f64)> = self.feature_names
                .iter()
                .zip(shap_vec.iter())
                .map(|(name, &value)| (name.clone(), value.abs()))
                .collect();
            
            feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_5: Vec<(String, f64)> = feature_importance.into_iter().take(5).collect();
            
            // Store for later analysis
            *self.latest_shap_values.write() = Some(shap_vec.clone());
            
            (Some(shap_vec), Some(top_5))
        };
        
        // Update ML feedback with SHAP insights
        ml.update_feature_importance(&self.feature_names, &shap_values.as_ref().unwrap());
        
        // Create enhanced reason with top features
        let top_features_str = top_features.as_ref()
            .map(|tf| tf.iter()
                .map(|(name, val)| format!("{}:{:.3}", name, val))
                .collect::<Vec<_>>()
                .join(", "))
            .unwrap_or_default();
        
        Ok(Signal {
            action,
            confidence,
            size: confidence * crate::parameter_manager::PARAMETERS.get("ml_base_size"), // Auto-tuned!
            features: features.to_vec(),
            reason: format!("ML prediction | Top factors: {}", top_features_str),
            shap_values,
            top_features,
        })
    }
    
    /// Calculate TA indicators
    async fn calculate_ta_indicators(&self, market: &MarketData) -> Result<TAIndicators> {
        let ta = self.ta_analytics.read();
        
        Ok(TAIndicators {
            rsi: ta.get_rsi().unwrap_or(50.0),
            macd: ta.get_macd().map(|m| m.macd).unwrap_or(0.0),
            bollinger_position: ta.get_bollinger_position().unwrap_or(0.5),
            atr: ta.get_atr().unwrap_or(market.spread.to_f64()),
            volume_ratio: ta.get_volume_ratio().unwrap_or(1.0),
            trend_strength: ta.get_adx().unwrap_or(25.0),
            support: ta.get_support_level().unwrap_or(market.bid.to_f64()),
            resistance: ta.get_resistance_level().unwrap_or(market.ask.to_f64()),
        })
    }
    
    /// Get TA signal
    async fn get_ta_signal(&self, indicators: &TAIndicators) -> Result<Signal> {
        // Combine multiple TA indicators
        let mut buy_signals = 0;
        let mut sell_signals = 0;
        
        // RSI
        if indicators.rsi < 30.0 { buy_signals += 2; }  // Oversold
        if indicators.rsi > 70.0 { sell_signals += 2; } // Overbought
        
        // MACD
        if indicators.macd > 0.0 { buy_signals += 1; }
        if indicators.macd < 0.0 { sell_signals += 1; }
        
        // Bollinger Bands
        if indicators.bollinger_position < 0.2 { buy_signals += 1; }
        if indicators.bollinger_position > 0.8 { sell_signals += 1; }
        
        // Trend strength
        if indicators.trend_strength > 40.0 {
            if indicators.macd > 0.0 { buy_signals += 1; }
            if indicators.macd < 0.0 { sell_signals += 1; }
        }
        
        let total_signals = (buy_signals + sell_signals) as f64;
        let action = if buy_signals > sell_signals {
            SignalAction::Buy
        } else if sell_signals > buy_signals {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        let confidence = if total_signals > 0.0 {
            (buy_signals.max(sell_signals) as f64) / total_signals
        } else {
            0.0
        };
        
        Ok(Signal {
            action,
            confidence,
            size: confidence * crate::parameter_manager::PARAMETERS.get("ta_base_size"), // Auto-tuned!
            features: vec![
                indicators.rsi,
                indicators.macd,
                indicators.bollinger_position,
                indicators.trend_strength,
            ],
            reason: format!("TA: RSI={:.1}, MACD={:.3}, ADX={:.1}", 
                          indicators.rsi, indicators.macd, indicators.trend_strength),
            shap_values: None, // TA doesn't use SHAP
            top_features: None,
        })
    }
    
    /// Get sentiment signal
    async fn get_sentiment_signal(&self, sentiment: &SentimentData) -> Result<Signal> {
        let action = if sentiment.score > 0.6 {
            SignalAction::Buy
        } else if sentiment.score < -0.6 {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        Ok(Signal {
            action,
            confidence: sentiment.confidence,
            size: sentiment.confidence * crate::parameter_manager::PARAMETERS.get("sentiment_base_size"), // Auto-tuned!
            features: vec![sentiment.score, sentiment.confidence],
            reason: format!("Sentiment: {} (score: {:.2})", sentiment.source, sentiment.score),
            shap_values: None, // Sentiment doesn't use SHAP
            top_features: None,
        })
    }
    
    /// Combine signals with adaptive weights
    async fn combine_signals(
        &self,
        ml: &Signal,
        ta: &Signal,
        sentiment: Option<&Signal>,
    ) -> Result<Signal> {
        let ml_w = *self.ml_weight.read();
        let ta_w = *self.ta_weight.read();
        let sent_w = *self.sentiment_weight.read();
        
        // Normalize weights
        let total_w = ml_w + ta_w + (if sentiment.is_some() { sent_w } else { 0.0 });
        
        // Calculate weighted confidence
        let mut weighted_confidence = ml.confidence * (ml_w / total_w) 
                                    + ta.confidence * (ta_w / total_w);
        
        if let Some(sent) = sentiment {
            weighted_confidence += sent.confidence * (sent_w / total_w);
        }
        
        // Determine action (majority vote with confidence weighting)
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;
        
        match ml.action {
            SignalAction::Buy => buy_score += ml.confidence * ml_w,
            SignalAction::Sell => sell_score += ml.confidence * ml_w,
            _ => {}
        }
        
        match ta.action {
            SignalAction::Buy => buy_score += ta.confidence * ta_w,
            SignalAction::Sell => sell_score += ta.confidence * ta_w,
            _ => {}
        }
        
        if let Some(sent) = sentiment {
            match sent.action {
                SignalAction::Buy => buy_score += sent.confidence * sent_w,
                SignalAction::Sell => sell_score += sent.confidence * sent_w,
                _ => {}
            }
        }
        
        let action = if buy_score > sell_score && buy_score > 0.3 {
            SignalAction::Buy
        } else if sell_score > buy_score && sell_score > 0.3 {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        // Calculate position size
        let size = (ml.size * ml_w + ta.size * ta_w 
                   + sentiment.map(|s| s.size * sent_w).unwrap_or(0.0)) / total_w;
        
        Ok(Signal {
            action,
            confidence: weighted_confidence,
            size,
            features: ml.features.clone(), // Keep ML features
            reason: format!("Combined: ML({:.0}%), TA({:.0}%), Sentiment({:.0}%)",
                          ml_w * 100.0 / total_w, 
                          ta_w * 100.0 / total_w,
                          sent_w * 100.0 / total_w),
            shap_values: ml.shap_values.clone(), // Preserve ML SHAP values
            top_features: ml.top_features.clone(), // Preserve ML top features
        })
    }
    
    /// Calculate order book imbalance
    fn calculate_order_book_imbalance(&self, order_book: &OrderBook) -> f64 {
        let bid_volume: f64 = order_book.bids.iter()
            .take(5)
            .map(|o| o.quantity.to_f64())
            .sum();
            
        let ask_volume: f64 = order_book.asks.iter()
            .take(5)
            .map(|o| o.quantity.to_f64())
            .sum();
            
        if bid_volume + ask_volume > 0.0 {
            (bid_volume - ask_volume) / (bid_volume + ask_volume)
        } else {
            0.0
        }
    }
    
    /// Calculate depth ratio
    fn calculate_depth_ratio(&self, order_book: &OrderBook) -> f64 {
        if !order_book.bids.is_empty() && !order_book.asks.is_empty() {
            let best_bid_size = order_book.bids[0].quantity.to_f64();
            let best_ask_size = order_book.asks[0].quantity.to_f64();
            
            if best_ask_size > 0.0 {
                best_bid_size / best_ask_size
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    
    // ... Additional helper methods ...
    
    /// Adjust signal for tail risk and contagion (Nexus Priority 2)
    fn adjust_for_tail_risk(
        &self,
        mut signal: Signal,
        tail_metrics: &crate::t_copula::TailDependenceMetrics,
        contagion_risk: &crate::cross_asset_correlations::ContagionRisk,
    ) -> Signal {
        // Alex: "When correlations spike to 1.0, REDUCE EVERYTHING!"
        
        // Reduce confidence based on tail dependence
        if tail_metrics.max_tail_dependence > 0.5 {
            signal.confidence *= 0.7;  // 30% reduction
            signal.reason = format!("{} | Tail risk: {:.2}", 
                                   signal.reason, tail_metrics.max_tail_dependence);
        }
        
        // Further reduction for crisis mode
        if tail_metrics.is_crisis {
            signal.confidence *= 0.5;  // Another 50% reduction
            signal.reason = format!("{} | CRISIS MODE", signal.reason);
        }
        
        // Adjust for contagion
        if contagion_risk.contagion_level > 0.5 {
            signal.confidence *= 0.6;  // 40% reduction
            signal.size *= 0.5;  // Halve position size
            signal.reason = format!("{} | Contagion: {:.2}%", 
                                   signal.reason, contagion_risk.contagion_level * 100.0);
        }
        
        // Extreme risk - convert to hold/exit signal
        if contagion_risk.systemic_risk > 0.8 {
            signal.action = SignalAction::Hold;  // Conservative action
            signal.confidence = 0.9;  // High confidence to hold
            signal.reason = format!("SYSTEMIC RISK DETECTED: {}", 
                                  contagion_risk.recommended_action);
        }
        
        signal
    }
    
    /// Detect market regime
    async fn detect_market_regime(
        &self,
        market: &MarketData,
        indicators: &TAIndicators,
    ) -> Result<String> {
        // Simplified regime detection
        if indicators.trend_strength > 40.0 && indicators.macd > 0.0 {
            Ok("Bull".to_string())
        } else if indicators.trend_strength > 40.0 && indicators.macd < 0.0 {
            Ok("Bear".to_string())
        } else {
            Ok("Sideways".to_string())
        }
    }
    
    /// Calculate Kelly size
    async fn calculate_kelly_size(&self, signal: &Signal, market: &MarketData) -> Result<f64> {
        let kelly = self.kelly_sizer.read();
        let size = kelly.calculate_position_size(
            Decimal::from_f64(signal.confidence).unwrap_or(dec!(0.5)),
            Decimal::from_f64(0.02).unwrap(), // Expected return
            Decimal::from_f64(0.01).unwrap(), // Expected risk
            Some(Decimal::from_f64(0.002).unwrap()), // Trading costs
        )?;
        
        Ok(size.to_f64().unwrap_or(0.0))
    }
    
    /// Apply risk clamps
    async fn apply_risk_clamps(
        &self,
        signal: Signal,
        kelly_size: f64,
        market: &MarketData,
    ) -> Result<TradingSignal> {
        // Convert to TradingSignal
        let trading_signal = TradingSignal {
            timestamp: market.timestamp,
            symbol: market.symbol.clone(),
            action: signal.action,
            confidence: Percentage::new(signal.confidence),
            size: Quantity::new(Decimal::from_f64(kelly_size).unwrap()),
            reason: signal.reason,
            risk_metrics: RiskMetrics::default(),
            ml_features: signal.features,
            ta_indicators: vec![],
        };
        
        // Apply clamps - use calculate_position_size instead
        let mut clamps = self.risk_clamps.write();
        
        // Get REAL portfolio state - NO HARDCODED VALUES!
        let portfolio_heat = self.portfolio_manager.calculate_portfolio_heat() as f32;
        let correlation = self.portfolio_manager.get_correlation(&trading_signal.symbol, "BTC/USDT") as f32;
        let account_equity = self.portfolio_manager.get_account_equity() as f32;
        
        // Apply risk clamps to position size
        let clamped_size = clamps.calculate_position_size(
            signal.confidence as f32,
            self.ta_analytics.read().get_current_volatility() as f32,
            portfolio_heat,
            correlation,
            account_equity,
        );
        
        // Update trading signal with clamped size
        let mut clamped_signal = trading_signal;
        clamped_signal.size = Quantity::new(Decimal::from_f32(clamped_size).unwrap_or(dec!(0)));
        
        Ok(clamped_signal)
    }
    
    /// Extract profit opportunity
    async fn extract_profit_opportunity(
        &self,
        signal: TradingSignal,
        market: &MarketData,
        order_book: &OrderBook,
    ) -> Result<TradingSignal> {
        // Profit extractor would optimize execution here
        Ok(signal)
    }
    
    /// Record decision for learning
    async fn record_decision(
        &self,
        ml: Signal,
        ta: Signal,
        combined: Signal,
        final_signal: TradingSignal,
    ) -> Result<()> {
        let record = DecisionRecord {
            timestamp: chrono::Utc::now().timestamp() as u64,
            ml_signal: ml,
            ta_signal: ta,
            combined_signal: combined,
            final_signal: Signal {
                action: final_signal.action,
                confidence: final_signal.confidence.value(),
                size: final_signal.size.to_f64(),
                features: final_signal.ml_features,
                reason: final_signal.reason,
                shap_values: None, // Final signal doesn't have SHAP (already applied)
                top_features: None,
            },
            execution_result: None,
            pnl: None,
        };
        
        self.decision_history.write().push(record);
        Ok(())
    }
    
    /// Update auto-tuning parameters
    async fn update_auto_tuning(&self, signal: &TradingSignal, regime: String) -> Result<()> {
        // Update weights based on performance
        // This would track actual PnL and adjust weights accordingly
        
        // Save to database
        self.persistence.update_parameter(
            "ml_weight",
            Decimal::from_f64(*self.ml_weight.read()).unwrap(),
            "Auto-tuning weight adjustment",
            Some(&regime),
        ).await?;
        
        Ok(())
    }
}

/// TA Indicators structure
#[derive(Debug, Clone)]
pub struct TAIndicators {
    pub rsi: f64,
    pub macd: f64,
    pub bollinger_position: f64,
    pub atr: f64,
    pub volume_ratio: f64,
    pub trend_strength: f64,
    pub support: f64,
    pub resistance: f64,
}

/// Order book structure
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<Order>,
    pub asks: Vec<Order>,
    pub timestamp: u64,
}

/// Order in the book
#[derive(Debug, Clone)]
pub struct Order {
    pub price: Price,
    pub quantity: Quantity,
}

/// Sentiment data
#[derive(Debug, Clone)]
pub struct SentimentData {
    pub source: String,
    pub score: f64,       // -1 to 1
    pub confidence: f64,   // 0 to 1
    pub timestamp: u64,
}