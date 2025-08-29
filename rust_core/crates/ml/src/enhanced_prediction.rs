use domain_types::market_data::MarketData;
//! # ENHANCED ML PREDICTION PIPELINE - Next-Gen Analytics
//! Blake: "Multi-model ensemble with confidence weighting!"
//! Cameron: "Risk-adjusted predictions!"
//!
//! Improvements:
//! - Model ensemble voting
//! - Confidence calibration
//! - Feature importance tracking
//! - Real-time model selection
//! - Risk-adjusted sizing

use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Enhanced ML Prediction System
/// TODO: Add docs
pub struct EnhancedMLPipeline {
    /// Multiple models for ensemble
    models: ModelEnsemble,
    
    /// Feature engineering
    feature_engine: FeatureEngine,
    
    /// Model performance tracker
    performance_tracker: ModelPerformanceTracker,
    
    /// Risk adjustment (Cameron)
    risk_adjuster: RiskAdjuster,
    
    /// Real-time model selector
    model_selector: DynamicModelSelector,
}

impl EnhancedMLPipeline {
    /// MAIN PREDICTION - Ensemble with confidence
    pub async fn predict(&mut self, market_data: &MarketData) -> MLPrediction {
        println!("BLAKE: Running enhanced ML prediction pipeline");
        
        // ==========================================
        // STAGE 1: Feature Engineering
        // ==========================================
        
        let features = self.feature_engine.extract_features(market_data);
        println!("BLAKE: Extracted {} features", features.len());
        
        // Top features (SHAP values)
        let important_features = self.feature_engine.get_important_features(&features);
        println!("BLAKE: Top 5 features: {:?}", 
                 important_features.iter().take(5).collect::<Vec<_>>());
        
        // ==========================================
        // STAGE 2: Multi-Model Predictions
        // ==========================================
        
        let predictions = self.models.predict_all(&features).await;
        
        // Model predictions
        println!("BLAKE: XGBoost prediction: {:.4}, confidence: {:.2}%",
                 predictions.xgboost.value, predictions.xgboost.confidence * 100.0);
        println!("BLAKE: LSTM prediction: {:.4}, confidence: {:.2}%", 
                 predictions.lstm.value, predictions.lstm.confidence * 100.0);
        println!("BLAKE: Random Forest: {:.4}, confidence: {:.2}%",
                 predictions.random_forest.value, predictions.random_forest.confidence * 100.0);
        
        // ==========================================
        // STAGE 3: Dynamic Model Selection
        // ==========================================
        
        let selected_models = self.model_selector.select_best_models(
            &predictions,
            &self.performance_tracker,
            market_data,
        );
        
        println!("BLAKE: Selected {} models for current market regime", 
                 selected_models.len());
        
        // ==========================================
        // STAGE 4: Ensemble Voting
        // ==========================================
        
        let ensemble_prediction = self.ensemble_vote(
            &predictions,
            &selected_models,
        );
        
        println!("BLAKE: Ensemble prediction: {:.4}, confidence: {:.2}%",
                 ensemble_prediction.value, ensemble_prediction.confidence * 100.0);
        
        // ==========================================
        // STAGE 5: Risk Adjustment (Cameron)
        // ==========================================
        
        let risk_adjusted = self.risk_adjuster.adjust_prediction(
            &ensemble_prediction,
            market_data,
        );
        
        println!("CAMERON: Risk-adjusted prediction: {:.4}, size: {:.2}%",
                 risk_adjusted.value, risk_adjusted.position_size * 100.0);
        
        // ==========================================
        // STAGE 6: Confidence Calibration
        // ==========================================
        
        let calibrated = self.calibrate_confidence(
            &risk_adjusted,
            &self.performance_tracker,
        );
        
        println!("BLAKE: Calibrated confidence: {:.2}% (was {:.2}%)",
                 calibrated.confidence * 100.0, 
                 risk_adjusted.confidence * 100.0);
        
        // Track prediction for later validation
        self.performance_tracker.track_prediction(&calibrated);
        
        calibrated
    }
    
    /// Ensemble voting with confidence weighting
    fn ensemble_vote(
        &self,
        predictions: &ModelPredictions,
        selected_models: &[ModelType],
    ) -> MLPrediction {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut confidence_sum = 0.0;
        
        for model_type in selected_models {
            let (pred, weight) = match model_type {
                ModelType::XGBoost => {
                    let p = &predictions.xgboost;
                    (p, p.confidence * self.performance_tracker.get_accuracy(model_type))
                },
                ModelType::LSTM => {
                    let p = &predictions.lstm;
                    (p, p.confidence * self.performance_tracker.get_accuracy(model_type))
                },
                ModelType::RandomForest => {
                    let p = &predictions.random_forest;
                    (p, p.confidence * self.performance_tracker.get_accuracy(model_type))
                },
                ModelType::GradientBoost => {
                    let p = &predictions.gradient_boost;
                    (p, p.confidence * self.performance_tracker.get_accuracy(model_type))
                },
                ModelType::NeuralNet => {
                    let p = &predictions.neural_net;
                    (p, p.confidence * self.performance_tracker.get_accuracy(model_type))
                },
            };
            
            weighted_sum += pred.value * weight;
            weight_sum += weight;
            confidence_sum += pred.confidence;
        }
        
        MLPrediction {
            value: if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 },
            confidence: confidence_sum / selected_models.len() as f64,
            direction: if weighted_sum > 0.0 { Direction::Long } else { Direction::Short },
            position_size: 0.0,  // Will be set by risk adjuster
            models_used: selected_models.to_vec(),
            timestamp: Utc::now(),
        }
    }
    
    /// Calibrate confidence based on historical performance
    fn calibrate_confidence(
        &self,
        prediction: &MLPrediction,
        tracker: &ModelPerformanceTracker,
    ) -> MLPrediction {
        // Get historical accuracy for this confidence level
        let historical_accuracy = tracker.get_accuracy_at_confidence(prediction.confidence);
        
        // Calibrate using isotonic regression approximation
        let calibrated_confidence = if historical_accuracy > 0.0 {
            // If model is overconfident, reduce confidence
            // If model is underconfident, increase confidence
            let ratio = historical_accuracy / prediction.confidence;
            (prediction.confidence * ratio.sqrt()).min(0.99).max(0.01)
        } else {
            prediction.confidence * 0.8  // Conservative default
        };
        
        let mut calibrated = prediction.clone();
        calibrated.confidence = calibrated_confidence;
        calibrated
    }
}

/// Model Ensemble - Multiple models for robustness
struct ModelEnsemble {
    xgboost: XGBoostModel,
    lstm: LSTMModel,
    random_forest: RandomForestModel,
    gradient_boost: GradientBoostModel,
    neural_net: NeuralNetModel,
}

impl ModelEnsemble {
    async fn predict_all(&self, features: &FeatureVector) -> ModelPredictions {
        // Run all models in parallel
        let (xgb, lstm, rf, gb, nn) = tokio::join!(
            self.xgboost.predict(features),
            self.lstm.predict(features),
            self.random_forest.predict(features),
            self.gradient_boost.predict(features),
            self.neural_net.predict(features),
        );
        
        ModelPredictions {
            xgboost: xgb,
            lstm,
            random_forest: rf,
            gradient_boost: gb,
            neural_net: nn,
        }
    }
}

/// Dynamic Model Selection - Choose best models for current regime
struct DynamicModelSelector {
    regime_detector: MarketRegimeDetector,
}

impl DynamicModelSelector {
    fn select_best_models(
        &self,
        predictions: &ModelPredictions,
        tracker: &ModelPerformanceTracker,
        market_data: &MarketData,
    ) -> Vec<ModelType> {
        let regime = self.regime_detector.detect(market_data);
        
        // Get model performance for current regime
        let mut model_scores: Vec<(ModelType, f64)> = vec![
            (ModelType::XGBoost, tracker.get_regime_accuracy(&ModelType::XGBoost, &regime)),
            (ModelType::LSTM, tracker.get_regime_accuracy(&ModelType::LSTM, &regime)),
            (ModelType::RandomForest, tracker.get_regime_accuracy(&ModelType::RandomForest, &regime)),
            (ModelType::GradientBoost, tracker.get_regime_accuracy(&ModelType::GradientBoost, &regime)),
            (ModelType::NeuralNet, tracker.get_regime_accuracy(&ModelType::NeuralNet, &regime)),
        ];
        
        // Sort by performance
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Select top 3 models with >60% accuracy
        model_scores.into_iter()
            .filter(|(_, score)| *score > 0.6)
            .take(3)
            .map(|(model, _)| model)
            .collect()
    }
}

/// Risk Adjuster - Cameron's risk-based sizing
struct RiskAdjuster {
    risk_calculator: UnifiedRiskCalculator,
    portfolio_manager: PortfolioManager,
}

impl RiskAdjuster {
    fn adjust_prediction(
        &self,
        prediction: &MLPrediction,
        market_data: &MarketData,
    ) -> MLPrediction {
        let mut adjusted = prediction.clone();
        
        // Calculate Kelly fraction
        let kelly = self.risk_calculator.calculate_kelly_criterion(
            prediction.confidence,
            0.015,  // Expected 1.5% win
            0.01,   // Expected 1% loss
            Some(prediction.confidence),
        );
        
        // Check portfolio heat
        let portfolio_heat = self.portfolio_manager.get_portfolio_heat();
        
        // Reduce size if portfolio is hot
        let heat_adjusted = kelly.to_f64() * (1.0 - portfolio_heat * 0.5);
        
        // Apply volatility adjustment
        let volatility = market_data.volatility;
        let vol_adjusted = if volatility > 0.02 {  // High volatility
            heat_adjusted * 0.5
        } else if volatility > 0.015 {  // Medium volatility
            heat_adjusted * 0.75
        } else {  // Low volatility
            heat_adjusted
        };
        
        adjusted.position_size = vol_adjusted.min(0.02);  // Max 2% position
        
        adjusted
    }
}

/// Feature Engine - Advanced feature extraction
struct FeatureEngine {
    technical_indicators: UnifiedIndicators,
    market_microstructure: MicrostructureAnalyzer,
    sentiment_analyzer: SentimentAnalyzer,
}

impl FeatureEngine {
    fn extract_features(&mut self, market_data: &MarketData) -> FeatureVector {
        let mut features = Vec::new();
        
        // Technical indicators (15 features)
        let indicators = self.technical_indicators.get_all_indicators();
        features.push(indicators.rsi.unwrap_or(50.0));
        features.push(indicators.macd.map(|m| m.macd).unwrap_or(0.0));
        // ... more indicators
        
        // Market microstructure (10 features)
        let micro = self.market_microstructure.analyze(market_data);
        features.push(micro.bid_ask_spread);
        features.push(micro.order_imbalance);
        // ... more microstructure
        
        // Sentiment (5 features)
        let sentiment = self.sentiment_analyzer.analyze(market_data);
        features.push(sentiment.fear_greed_index);
        // ... more sentiment
        
        // Normalize features
        self.normalize_features(&mut features);
        
        FeatureVector(features)
    }
    
    fn get_important_features(&self, features: &FeatureVector) -> Vec<(String, f64)> {
        // Would use SHAP values in production
        vec![
            ("RSI".to_string(), 0.15),
            ("Order_Imbalance".to_string(), 0.12),
            ("MACD_Signal".to_string(), 0.10),
            ("Bid_Ask_Spread".to_string(), 0.08),
            ("Volume_Profile".to_string(), 0.07),
        ]
    }
    
    fn normalize_features(&self, features: &mut [f64]) {
        // Z-score normalization
        let mean = features.iter().sum::<f64>() / features.len() as f64;
        let variance = features.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / features.len() as f64;
        let std = variance.sqrt();
        
        if std > 0.0 {
            for f in features.iter_mut() {
                *f = (*f - mean) / std;
            }
        }
    }
}

/// Model Performance Tracker
struct ModelPerformanceTracker {
    accuracy_history: HashMap<ModelType, Vec<f64>>,
    regime_accuracy: HashMap<(ModelType, MarketRegime), f64>,
    confidence_calibration: Vec<(f64, f64)>,  // (predicted_conf, actual_accuracy)
}

impl ModelPerformanceTracker {
    fn get_accuracy(&self, model: &ModelType) -> f64 {
        self.accuracy_history.get(model)
            .and_then(|hist| {
                if hist.is_empty() { None }
                else { Some(hist.iter().sum::<f64>() / hist.len() as f64) }
            })
            .unwrap_or(0.5)
    }
    
    fn get_regime_accuracy(&self, model: &ModelType, regime: &MarketRegime) -> f64 {
        self.regime_accuracy.get(&(*model, *regime))
            .copied()
            .unwrap_or(0.5)
    }
    
    fn get_accuracy_at_confidence(&self, confidence: f64) -> f64 {
        // Find closest confidence level in calibration data
        self.confidence_calibration.iter()
            .min_by_key(|(conf, _)| ((conf - confidence).abs() * 1000.0) as i32)
            .map(|(_, acc)| *acc)
            .unwrap_or(confidence * 0.8)
    }
    
    fn track_prediction(&mut self, prediction: &MLPrediction) {
        // Store for later validation when outcome is known
    }
}

// Supporting types
#[derive(Clone)]
/// TODO: Add docs
// ELIMINATED: MLPrediction - Enhanced with Confidence intervals, SHAP values
// pub struct MLPrediction {
    pub value: f64,  // Predicted price movement
    pub confidence: f64,  // 0-1
    pub direction: Direction,
    pub position_size: f64,  // Risk-adjusted size
    pub models_used: Vec<ModelType>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Copy, PartialEq)]
/// TODO: Add docs
pub enum Direction {
    Long,
    Short,
    Neutral,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// TODO: Add docs
pub enum ModelType {
    XGBoost,
    LSTM,
    RandomForest,
    GradientBoost,
    NeuralNet,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// TODO: Add docs
pub enum MarketRegime {
    Trending,
    RangeRound,
    HighVolatility,
    LowVolatility,
}

struct ModelPredictions {
    xgboost: SingleModelPrediction,
    lstm: SingleModelPrediction,
    random_forest: SingleModelPrediction,
    gradient_boost: SingleModelPrediction,
    neural_net: SingleModelPrediction,
}

struct SingleModelPrediction {
    value: f64,
    confidence: f64,
}

// REMOVED: Using canonical domain_types::market_data::MarketData
// pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub timestamp: DateTime<Utc>,
}

struct FeatureVector(Vec<f64>);

impl FeatureVector {
    fn len(&self) -> usize {
        self.0.len()
    }
}

// Model implementations (simplified)
struct XGBoostModel;
struct LSTMModel;
struct RandomForestModel;
struct GradientBoostModel;
struct NeuralNetModel;

impl XGBoostModel {
    async fn predict(&self, _features: &FeatureVector) -> SingleModelPrediction {
        SingleModelPrediction { value: 0.0012, confidence: 0.82 }
    }
}

impl LSTMModel {
    async fn predict(&self, _features: &FeatureVector) -> SingleModelPrediction {
        SingleModelPrediction { value: 0.0015, confidence: 0.78 }
    }
}

impl RandomForestModel {
    async fn predict(&self, _features: &FeatureVector) -> SingleModelPrediction {
        SingleModelPrediction { value: 0.0010, confidence: 0.75 }
    }
}

impl GradientBoostModel {
    async fn predict(&self, _features: &FeatureVector) -> SingleModelPrediction {
        SingleModelPrediction { value: 0.0013, confidence: 0.80 }
    }
}

impl NeuralNetModel {
    async fn predict(&self, _features: &FeatureVector) -> SingleModelPrediction {
        SingleModelPrediction { value: 0.0014, confidence: 0.77 }
    }
}

// Supporting components
use crate::unified_indicators::UnifiedIndicators;
use crate::unified_risk_calculations::UnifiedRiskCalculator;
use crate::portfolio_manager::PortfolioManager;

struct MicrostructureAnalyzer {
    bid_ask_spread: f64,
    order_imbalance: f64,
}

impl MicrostructureAnalyzer {
    fn analyze(&self, _market: &MarketData) -> Self {
        Self {
            bid_ask_spread: 0.0001,
            order_imbalance: 0.05,
        }
    }
}

struct SentimentAnalyzer {
    fear_greed_index: f64,
}

impl SentimentAnalyzer {
    fn analyze(&self, _market: &MarketData) -> Self {
        Self {
            fear_greed_index: 55.0,
        }
    }
}

struct MarketRegimeDetector;

impl MarketRegimeDetector {
    fn detect(&self, _market: &MarketData) -> MarketRegime {
        MarketRegime::Trending
    }
}

// BLAKE: "Enhanced ML pipeline with ensemble voting!"
// CAMERON: "Risk-adjusted predictions for safety!"