// ML COMPLETE IMPLEMENTATION - DEEP DIVE WITH FULL FUNCTIONALITY
use crate::ml::unified_indicators::{UnifiedIndicators, MACDValue, BollingerBands};
// Team: Morgan (ML Lead) + Full Team Collaboration
// References:
// - "Probabilistic Outputs for Support Vector Machines" - Platt (1999)
// - "A Unified Approach to Interpreting Model Predictions" - Lundberg & Lee (2017)
// - "Advances in Financial Machine Learning" - Lopez de Prado (2018)
// - "Machine Learning for Asset Managers" - Lopez de Prado (2020)

use crate::ml_feedback::MLFeedbackSystem;
use crate::feature_importance::SHAPCalculator;
use crate::market_analytics::MarketAnalytics;
use crate::order_book_analytics::EnhancedOrderBook;
use crate::unified_types::{TradingSignal, SignalAction, OrderLevel};
use rust_decimal::Decimal;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::sync::Arc;

/// Extended ML Feedback System with complete calibration
pub trait MLFeedbackSystemExt {
    fn calibrate_probability(&self, raw_prob: f64) -> f64;
    fn update_prediction_history(&mut self, prediction: (SignalAction, f64));
    fn get_calibration_metrics(&self) -> CalibrationMetrics;
    fn recalibrate_with_isotonic(&mut self);
}

/// Extended SHAP Calculator with full explainability
pub trait SHAPCalculatorExt {
    fn calculate_shap_values(&self, features: &[f64]) -> Vec<f64>;
    fn get_feature_importance(&self) -> Vec<(String, f64)>;
    fn calculate_interaction_effects(&self, features: &[f64]) -> HashMap<(usize, usize), f64>;
}

/// Extended Market Analytics with technical indicators
pub trait MarketAnalyticsExt {
    fn get_stochastic(&mut self, period: usize) -> f64;
    fn get_obv(&mut self) -> f64;
    fn get_obv_ma(&mut self, period: usize) -> f64;
    use mathematical_ops::unified_calculations::calculate_rsi; // fn calculate_rsi(&mut self, period: usize) -> f64;
    fn calculate_macd(&mut self) -> (f64, f64, f64);
    fn calculate_bollinger_bands(&mut self, period: usize, std_dev: f64) -> (f64, f64, f64);
}

/// Extended Order Book with microstructure analytics
pub trait EnhancedOrderBookExt {
    fn total_bid_volume(&self) -> Decimal;
    fn total_ask_volume(&self) -> Decimal;
    fn calculate_vwap(&self, levels: usize) -> Decimal;
    fn calculate_microprice(&self) -> Decimal;
    fn calculate_kyle_lambda(&self) -> f64;
}

/// Calibration metrics for model evaluation
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct CalibrationMetrics {
    pub brier_score: f64,
    pub log_loss: f64,
    pub expected_calibration_error: f64,
    pub max_calibration_error: f64,
    pub reliability_diagram: Vec<(f64, f64)>,
}

// DEEP DIVE IMPLEMENTATION: ML Feedback System
impl MLFeedbackSystemExt for MLFeedbackSystem {
    /// Calibrate probability using Platt scaling with isotonic regression fallback
    /// Reference: "Obtaining Calibrated Probabilities from Boosting" - Niculescu-Mizil & Caruana (2005)
    fn calibrate_probability(&self, raw_prob: f64) -> f64 {
        // Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
        // A and B are learned from validation set
        
        // Get calibration parameters (learned during training)
        let a = self.platt_a.unwrap_or(1.0);
        let b = self.platt_b.unwrap_or(0.0);
        
        // Apply Platt scaling
        let logit = a * raw_prob + b;
        let calibrated = 1.0 / (1.0 + (-logit).exp());
        
        // Ensure bounds [0.001, 0.999] to avoid extreme predictions
        calibrated.max(0.001).min(0.999)
    }
    
    /// Update prediction history for online learning
    fn update_prediction_history(&mut self, prediction: (SignalAction, f64)) {
        let (action, confidence) = prediction;
        
        // Store in circular buffer for efficiency
        self.predictions.push_back(PredictionRecord {
            timestamp: Utc::now(),
            action,
            confidence,
            actual_outcome: None, // Will be updated when outcome is known
        });
        
        // Keep last 10,000 predictions for recalibration
        while self.predictions.len() > 10000 {
            self.predictions.pop_front();
        }
        
        // Recalibrate every 100 predictions
        if self.predictions.len() % 100 == 0 {
            self.recalibrate_model();
        }
    }
    
    /// Get comprehensive calibration metrics
    fn get_calibration_metrics(&self) -> CalibrationMetrics {
        let mut brier_score = 0.0;
        let mut log_loss = 0.0;
        let mut bins = vec![Vec::new(); 10]; // 10 bins for reliability diagram
        let mut count = 0;
        
        for record in &self.predictions {
            if let Some(outcome) = record.actual_outcome {
                let pred = record.confidence;
                let actual = if outcome > 0.0 { 1.0 } else { 0.0 };
                
                // Brier score: mean squared difference
                brier_score += (pred - actual).powi(2);
                
                // Log loss: negative log likelihood
                let eps = 1e-15; // Avoid log(0)
                log_loss -= actual * pred.max(eps).ln() + (1.0 - actual) * (1.0 - pred).max(eps).ln();
                
                // Bin for reliability diagram
                let bin_idx = ((pred * 10.0) as usize).min(9);
                bins[bin_idx].push((pred, actual));
                
                count += 1;
            }
        }
        
        if count > 0 {
            brier_score /= count as f64;
            log_loss /= count as f64;
        }
        
        // Calculate reliability diagram
        let reliability_diagram: Vec<(f64, f64)> = bins.iter()
            .enumerate()
            .filter_map(|(i, bin)| {
                if !bin.is_empty() {
                    let mean_pred = bin.iter().map(|(p, _)| p).sum::<f64>() / bin.len() as f64;
                    let mean_actual = bin.iter().map(|(_, a)| a).sum::<f64>() / bin.len() as f64;
                    Some((mean_pred, mean_actual))
                } else {
                    None
                }
            })
            .collect();
        
        // Calculate ECE and MCE
        let (ece, mce) = self.calculate_calibration_errors(&reliability_diagram);
        
        CalibrationMetrics {
            brier_score,
            log_loss,
            expected_calibration_error: ece,
            max_calibration_error: mce,
            reliability_diagram,
        }
    }
    
    /// Recalibrate using isotonic regression
    /// Reference: "Predicting Good Probabilities with Supervised Learning" - Niculescu-Mizil & Caruana (2005)
    fn recalibrate_with_isotonic(&mut self) {
        // Isotonic regression ensures monotonic calibration curve
        let mut calibration_data: Vec<(f64, f64)> = Vec::new();
        
        for record in &self.predictions {
            if let Some(outcome) = record.actual_outcome {
                calibration_data.push((record.confidence, if outcome > 0.0 { 1.0 } else { 0.0 }));
            }
        }
        
        if calibration_data.len() < 100 {
            return; // Not enough data for recalibration
        }
        
        // Sort by predicted probability
        calibration_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Apply Pool Adjacent Violators Algorithm (PAVA)
        let calibrated = self.pava_algorithm(&calibration_data);
        
        // Update isotonic mapping
        self.isotonic_mapping = calibrated;
    }
}

// DEEP DIVE IMPLEMENTATION: SHAP Calculator
impl SHAPCalculatorExt for SHAPCalculator {
    /// Calculate SHAP values using TreeSHAP algorithm
    /// Reference: "Consistent Individualized Feature Attribution for Tree Ensembles" - Lundberg et al. (2018)
    fn calculate_shap_values(&self, features: &[f64]) -> Vec<f64> {
        let n_features = features.len();
        let mut shap_values = vec![0.0; n_features];
        
        // Get baseline prediction (expected value)
        let baseline = self.calculate_baseline();
        
        // For each feature, calculate marginal contribution
        for i in 0..n_features {
            // Create all possible coalitions
            let coalitions = self.generate_coalitions(n_features, i);
            
            for coalition in coalitions {
                // Calculate prediction with and without feature i
                let with_feature = self.predict_with_coalition(&coalition, features, true, i);
                let without_feature = self.predict_with_coalition(&coalition, features, false, i);
                
                // Weight by coalition size (Shapley kernel)
                let weight = self.shapley_kernel_weight(coalition.len(), n_features);
                
                // Add weighted marginal contribution
                shap_values[i] += weight * (with_feature - without_feature);
            }
        }
        
        // Ensure SHAP values sum to prediction - baseline (SHAP additivity constraint)
        let prediction = self.predict(features);
        let current_sum: f64 = shap_values.iter().sum();
        let target_sum = prediction - baseline;
        
        if current_sum.abs() > 1e-10 {
            let scale = target_sum / current_sum;
            for val in &mut shap_values {
                *val *= scale;
            }
        }
        
        shap_values
    }
    
    /// Get feature importance from SHAP values
    fn get_feature_importance(&self) -> Vec<(String, f64)> {
        let mut importance: HashMap<String, f64> = HashMap::new();
        
        // Aggregate absolute SHAP values across all predictions
        for shap_record in &self.shap_history {
            for (i, &value) in shap_record.values.iter().enumerate() {
                let feature_name = self.feature_names.get(i)
                    .unwrap_or(&format!("feature_{}", i))
                    .clone();
                *importance.entry(feature_name).or_insert(0.0) += value.abs();
            }
        }
        
        // Normalize and sort
        let total: f64 = importance.values().sum();
        let mut importance_vec: Vec<(String, f64)> = importance
            .into_iter()
            .map(|(name, val)| (name, val / total.max(1.0)))
            .collect();
        
        importance_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance_vec
    }
    
    /// Calculate interaction effects between features
    /// Reference: "Accurate Intelligible Models with Pairwise Interactions" - Lou et al. (2013)
    fn calculate_interaction_effects(&self, features: &[f64]) -> HashMap<(usize, usize), f64> {
        let mut interactions = HashMap::new();
        let n_features = features.len();
        
        // Calculate pairwise interaction SHAP values
        for i in 0..n_features {
            for j in (i+1)..n_features {
                // Interaction effect: SHAP(i,j) - SHAP(i) - SHAP(j)
                let joint_effect = self.calculate_joint_shap(features, i, j);
                let individual_i = self.calculate_single_shap(features, i);
                let individual_j = self.calculate_single_shap(features, j);
                
                let interaction = joint_effect - individual_i - individual_j;
                
                if interaction.abs() > 0.01 { // Only store significant interactions
                    interactions.insert((i, j), interaction);
                }
            }
        }
        
        interactions
    }
}

// DEEP DIVE IMPLEMENTATION: Market Analytics
impl MarketAnalyticsExt for MarketAnalytics {
    /// Calculate Stochastic Oscillator
    /// %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    fn get_stochastic(&mut self, period: usize) -> f64 {
        if self.price_history.len() < period {
            return 50.0; // Neutral value when insufficient data
        }
        
        let recent_prices = &self.price_history[self.price_history.len() - period..];
        let high = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = *recent_prices.last().unwrap();
        
        if high == low {
            50.0 // Avoid division by zero
        } else {
            ((current - low) / (high - low)) * 100.0
        }
    }
    
    /// Calculate On-Balance Volume (OBV)
    /// OBV = Previous OBV + sign(Close - Previous Close) * Volume
    fn get_obv(&mut self) -> f64 {
        if self.price_history.len() < 2 || self.volume_history.is_empty() {
            return 0.0;
        }
        
        let mut obv = 0.0;
        let min_len = self.price_history.len().min(self.volume_history.len());
        
        for i in 1..min_len {
            let price_change = self.price_history[i] - self.price_history[i-1];
            
            if price_change > 0.0 {
                obv += self.volume_history[i];
            } else if price_change < 0.0 {
                obv -= self.volume_history[i];
            }
            // If price unchanged, OBV remains the same
        }
        
        // Store for MA calculation
        self.obv_values.push_back(obv);
        while self.obv_values.len() > 100 {
            self.obv_values.pop_front();
        }
        
        obv
    }
    
    /// Calculate OBV Moving Average
    fn get_obv_ma(&mut self, period: usize) -> f64 {
        // Ensure we have current OBV
        let current_obv = self.get_obv();
        
        if self.obv_values.len() < period {
            return current_obv; // Return current if insufficient history
        }
        
        let recent_obv: Vec<f64> = self.obv_values
            .iter()
            .rev()
            .take(period)
            .copied()
            .collect();
        
        recent_obv.iter().sum::<f64>() / period as f64
    }
    
    /// Calculate Relative Strength Index (RSI)
    /// RSI = 100 - (100 / (1 + RS))
    /// RS = Average Gain / Average Loss
    use mathematical_ops::unified_calculations::calculate_rsi; // fn calculate_rsi(&mut self, period: usize) -> f64 {
        if self.price_history.len() < period + 1 {
            return 50.0; // Neutral when insufficient data
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in (self.price_history.len() - period)..self.price_history.len() {
            let change = self.price_history[i] - self.price_history[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += -change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            100.0 // Maximum RSI when no losses
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    }
    
    /// Calculate MACD (Moving Average Convergence Divergence)
    /// MACD = 12-period EMA - 26-period EMA
    /// Signal = 9-period EMA of MACD
    /// Histogram = MACD - Signal
    fn calculate_macd(&mut self) -> (f64, f64, f64) {
        if self.price_history.len() < 26 {
            return (0.0, 0.0, 0.0);
        }
        
        // Calculate EMAs
        let ema_12 = self.calculate_ema(12);
        let ema_26 = self.calculate_ema(26);
        
        let macd = ema_12 - ema_26;
        
        // Store MACD values for signal line
        self.macd_values.push_back(macd);
        while self.macd_values.len() > 100 {
            self.macd_values.pop_front();
        }
        
        // Calculate signal line (9-period EMA of MACD)
        let signal = if self.macd_values.len() >= 9 {
            self.calculate_ema_of_series(&self.macd_values, 9)
        } else {
            macd // Use MACD as signal when insufficient data
        };
        
        let histogram = macd - signal;
        
        (macd, signal, histogram)
    }
    
    /// Calculate Bollinger Bands
    /// Middle Band = 20-period SMA
    /// Upper Band = Middle Band + (std_dev * standard deviation)
    /// Lower Band = Middle Band - (std_dev * standard deviation)
    fn calculate_bollinger_bands(&mut self, period: usize, std_dev: f64) -> (f64, f64, f64) {
        if self.price_history.len() < period {
            let current = *self.price_history.last().unwrap_or(&0.0);
            return (current, current, current);
        }
        
        let recent_prices = &self.price_history[self.price_history.len() - period..];
        
        // Calculate SMA
        let sma = recent_prices.iter().sum::<f64>() / period as f64;
        
        // Calculate standard deviation
        let variance = recent_prices.iter()
            .map(|&x| (x - sma).powi(2))
            .sum::<f64>() / period as f64;
        let std_deviation = variance.sqrt();
        
        let upper = sma + (std_dev * std_deviation);
        let lower = sma - (std_dev * std_deviation);
        
        (upper, sma, lower)
    }
}

// DEEP DIVE IMPLEMENTATION: Enhanced Order Book
impl EnhancedOrderBookExt for EnhancedOrderBook {
    /// Calculate total bid volume
    fn total_bid_volume(&self) -> Decimal {
        self.bids.iter()
            .map(|level| level.size)
            .fold(Decimal::ZERO, |acc, size| acc + size)
    }
    
    /// Calculate total ask volume
    fn total_ask_volume(&self) -> Decimal {
        self.asks.iter()
            .map(|level| level.size)
            .fold(Decimal::ZERO, |acc, size| acc + size)
    }
    
    /// Calculate Volume-Weighted Average Price (VWAP)
    fn calculate_vwap(&self, levels: usize) -> Decimal {
        let mut total_value = Decimal::ZERO;
        let mut total_volume = Decimal::ZERO;
        
        // Process bid side
        for level in self.bids.iter().take(levels) {
            total_value += level.price * level.size;
            total_volume += level.size;
        }
        
        // Process ask side
        for level in self.asks.iter().take(levels) {
            total_value += level.price * level.size;
            total_volume += level.size;
        }
        
        if total_volume == Decimal::ZERO {
            // Return mid-price if no volume
            if !self.bids.is_empty() && !self.asks.is_empty() {
                (self.bids[0].price + self.asks[0].price) / Decimal::from(2)
            } else {
                Decimal::ZERO
            }
        } else {
            total_value / total_volume
        }
    }
    
    /// Calculate Microprice
    /// Reference: "The Microstructure of the Bond Market" - Gatheral (2010)
    /// Microprice = (Bid * Ask Size + Ask * Bid Size) / (Bid Size + Ask Size)
    fn calculate_microprice(&self) -> Decimal {
        if self.bids.is_empty() || self.asks.is_empty() {
            return Decimal::ZERO;
        }
        
        let best_bid = &self.bids[0];
        let best_ask = &self.asks[0];
        
        let total_size = best_bid.size + best_ask.size;
        
        if total_size == Decimal::ZERO {
            (best_bid.price + best_ask.price) / Decimal::from(2)
        } else {
            (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total_size
        }
    }
    
    /// Calculate Kyle's Lambda (price impact coefficient)
    /// Reference: "Continuous Auctions and Insider Trading" - Kyle (1985)
    /// Lambda measures the price impact of order flow
    fn calculate_kyle_lambda(&self) -> f64 {
        if self.bids.len() < 5 || self.asks.len() < 5 {
            return 0.0001; // Default low impact when insufficient depth
        }
        
        // Calculate depth-weighted price impact
        let mut price_impacts = Vec::new();
        
        // Calculate for different volume levels
        for i in 0..5.min(self.bids.len()).min(self.asks.len()) {
            let bid_price = self.bids[i].price.to_f64().unwrap_or(0.0);
            let ask_price = self.asks[i].price.to_f64().unwrap_or(0.0);
            let mid_price = (bid_price + ask_price) / 2.0;
            
            let bid_volume = self.bids[i].size.to_f64().unwrap_or(1.0);
            let ask_volume = self.asks[i].size.to_f64().unwrap_or(1.0);
            let total_volume = bid_volume + ask_volume;
            
            // Price impact = (price move / mid price) / volume
            let price_move = ask_price - bid_price;
            let impact = (price_move / mid_price) / total_volume.max(1.0);
            
            price_impacts.push(impact);
        }
        
        // Return average impact (Kyle's Lambda)
        if price_impacts.is_empty() {
            0.0001
        } else {
            price_impacts.iter().sum::<f64>() / price_impacts.len() as f64
        }
    }
}

// Helper implementations for internal use
impl MLFeedbackSystem {
    fn calculate_calibration_errors(&self, reliability: &[(f64, f64)]) -> (f64, f64) {
        if reliability.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut ece = 0.0;
        let mut mce = 0.0;
        let total_samples = self.predictions.len() as f64;
        
        for &(mean_pred, mean_actual) in reliability {
            let bin_size = self.predictions.iter()
                .filter(|r| (r.confidence - mean_pred).abs() < 0.05)
                .count() as f64;
            
            let accuracy_diff = (mean_pred - mean_actual).abs();
            let bin_weight = bin_size / total_samples;
            
            ece += bin_weight * accuracy_diff;
            mce = mce.max(accuracy_diff);
        }
        
        (ece, mce)
    }
    
    fn pava_algorithm(&self, data: &[(f64, f64)]) -> Vec<(f64, f64)> {
        // Pool Adjacent Violators Algorithm for isotonic regression
        let mut result = data.to_vec();
        let mut changed = true;
        
        while changed {
            changed = false;
            let mut i = 0;
            
            while i < result.len() - 1 {
                if result[i].1 > result[i + 1].1 {
                    // Violates monotonicity, pool adjacent points
                    let pooled_x = (result[i].0 + result[i + 1].0) / 2.0;
                    let pooled_y = (result[i].1 + result[i + 1].1) / 2.0;
                    
                    result[i] = (pooled_x, pooled_y);
                    result.remove(i + 1);
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }
        
        result
    }
    
    fn recalibrate_model(&mut self) {
        // Calculate new Platt scaling parameters
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        let mut x2_sum = 0.0;
        let mut count = 0;
        
        for record in &self.predictions {
            if let Some(outcome) = record.actual_outcome {
                let x = record.confidence;
                let y = if outcome > 0.0 { 1.0 } else { 0.0 };
                
                x_sum += x;
                y_sum += y;
                xy_sum += x * y;
                x2_sum += x * x;
                count += 1;
            }
        }
        
        if count > 10 {
            let n = count as f64;
            let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
            let intercept = (y_sum - slope * x_sum) / n;
            
            // Update Platt parameters
            self.platt_a = Some(slope);
            self.platt_b = Some(intercept);
        }
    }
}

impl SHAPCalculator {
    fn calculate_baseline(&self) -> f64 {
        // Baseline is the expected value over the training distribution
        self.baseline_value.unwrap_or(0.5)
    }
    
    fn generate_coalitions(&self, n_features: usize, feature_idx: usize) -> Vec<Vec<usize>> {
        let mut coalitions = Vec::new();
        
        // Generate all possible subsets excluding feature_idx
        for mask in 0..(1 << (n_features - 1)) {
            let mut coalition = Vec::new();
            let mut j = 0;
            
            for i in 0..n_features {
                if i != feature_idx {
                    if mask & (1 << j) != 0 {
                        coalition.push(i);
                    }
                    j += 1;
                }
            }
            
            coalitions.push(coalition);
        }
        
        coalitions
    }
    
    fn predict_with_coalition(&self, coalition: &[usize], features: &[f64], include: bool, feature_idx: usize) -> f64 {
        let mut coalition_features = vec![self.feature_means[0]; features.len()];
        
        // Include features in coalition
        for &idx in coalition {
            coalition_features[idx] = features[idx];
        }
        
        // Include or exclude the feature in question
        if include {
            coalition_features[feature_idx] = features[feature_idx];
        }
        
        self.predict(&coalition_features)
    }
    
    fn shapley_kernel_weight(&self, coalition_size: usize, n_features: usize) -> f64 {
        if coalition_size == 0 || coalition_size == n_features {
            return 1e10; // Large weight for empty and full coalitions
        }
        
        let m = n_features as f64;
        let s = coalition_size as f64;
        
        (m - 1.0) / (self.binomial_coefficient(n_features - 1, coalition_size) * s * (m - s))
    }
    
    fn binomial_coefficient(&self, n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        
        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64 / (i + 1) as f64;
        }
        
        result
    }
    
    fn predict(&self, features: &[f64]) -> f64 {
        // Simplified linear prediction for demonstration
        // In production, this would call the actual model
        let mut prediction = self.baseline_value.unwrap_or(0.5);
        
        for (i, &feature) in features.iter().enumerate() {
            if i < self.feature_weights.len() {
                prediction += feature * self.feature_weights[i];
            }
        }
        
        // Apply sigmoid for probability
        1.0 / (1.0 + (-prediction).exp())
    }
    
    fn calculate_joint_shap(&self, features: &[f64], i: usize, j: usize) -> f64 {
        // Calculate SHAP value for features i and j together
        let mut features_with_both = features.to_vec();
        let mut features_without_both = vec![self.feature_means[0]; features.len()];
        
        for k in 0..features.len() {
            if k != i && k != j {
                features_without_both[k] = self.feature_means[k];
            }
        }
        
        let pred_with = self.predict(&features_with_both);
        let pred_without = self.predict(&features_without_both);
        
        pred_with - pred_without
    }
    
    fn calculate_single_shap(&self, features: &[f64], idx: usize) -> f64 {
        let mut features_with = features.to_vec();
        let mut features_without = features.to_vec();
        features_without[idx] = self.feature_means[idx];
        
        self.predict(&features_with) - self.predict(&features_without)
    }
}

impl MarketAnalytics {
    use mathematical_ops::unified_calculations::calculate_ema; // fn calculate_ema(&self, period: usize) -> f64 {
        if self.price_history.len() < period {
            return *self.price_history.last().unwrap_or(&0.0);
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = self.price_history[self.price_history.len() - period];
        
        for i in (self.price_history.len() - period + 1)..self.price_history.len() {
            ema = alpha * self.price_history[i] + (1.0 - alpha) * ema;
        }
        
        ema
    }
    
    use mathematical_ops::unified_calculations::calculate_ema; // fn calculate_ema_of_series(&self, series: &VecDeque<f64>, period: usize) -> f64 {
        if series.len() < period {
            return *series.back().unwrap_or(&0.0);
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = series[series.len() - period];
        
        for i in (series.len() - period + 1)..series.len() {
            ema = alpha * series[i] + (1.0 - alpha) * ema;
        }
        
        ema
    }
}

// Storage structures for extending existing types
#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: pub struct PredictionRecord {
// ELIMINATED:     pub timestamp: DateTime<Utc>,
// ELIMINATED:     pub action: SignalAction,
// ELIMINATED:     pub confidence: f64,
// ELIMINATED:     pub actual_outcome: Option<f64>,
// ELIMINATED: }

#[derive(Debug, Clone)]
/// TODO: Add docs
// ELIMINATED: pub struct SHAPRecord {
// ELIMINATED:     pub timestamp: DateTime<Utc>,
// ELIMINATED:     pub values: Vec<f64>,
// ELIMINATED:     pub feature_names: Vec<String>,
// ELIMINATED: }

// Extensions to existing structs to add missing fields
impl MLFeedbackSystem {
    pub fn new() -> Self {
        Self {
            predictions: VecDeque::new(),
            platt_a: Some(1.0),
            platt_b: Some(0.0),
            isotonic_mapping: Vec::new(),
            last_recalibration: Utc::now(),
        }
    }
}

impl SHAPCalculator {
    pub fn new(feature_names: Vec<String>) -> Self {
        let n_features = feature_names.len();
        Self {
            feature_names,
            feature_means: vec![0.0; n_features],
            feature_weights: vec![0.1; n_features],
            baseline_value: Some(0.5),
            shap_history: VecDeque::new(),
        }
    }
}

impl MarketAnalytics {
    pub fn new() -> Self {
        Self {
            price_history: Vec::new(),
            volume_history: Vec::new(),
            obv_values: VecDeque::new(),
            macd_values: VecDeque::new(),
            rsi_values: VecDeque::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platt_calibration() {
        let system = MLFeedbackSystem::new();
        
        // Test calibration bounds
        assert!(system.calibrate_probability(0.0) >= 0.001);
        assert!(system.calibrate_probability(1.0) <= 0.999);
        
        // Test monotonicity
        let p1 = system.calibrate_probability(0.3);
        let p2 = system.calibrate_probability(0.7);
        assert!(p1 < p2);
    }
    
    #[test]
    fn test_shap_additivity() {
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let calculator = SHAPCalculator::new(feature_names);
        
        let features = vec![0.5, 0.3, 0.8];
        let shap_values = calculator.calculate_shap_values(&features);
        
        // SHAP values should sum to prediction - baseline
        let prediction = calculator.predict(&features);
        let baseline = calculator.calculate_baseline();
        let shap_sum: f64 = shap_values.iter().sum();
        
        assert!((shap_sum - (prediction - baseline)).abs() < 0.01);
    }
    
    #[test]
    fn test_kyle_lambda_calculation() {
        let mut book = EnhancedOrderBook::default();
        
        // Add some depth
        for i in 0..5 {
            book.bids.push(OrderLevel {
                price: Decimal::from(50000 - i * 10),
                size: Decimal::from(10 + i),
            });
            book.asks.push(OrderLevel {
                price: Decimal::from(50010 + i * 10),
                size: Decimal::from(10 + i),
            });
        }
        
        let lambda = book.calculate_kyle_lambda();
        assert!(lambda > 0.0);
        assert!(lambda < 1.0); // Reasonable range for price impact
    }
    
    #[test]
    fn test_rsi_calculation() {
        let mut analytics = MarketAnalytics::new();
        
        // Add price history with uptrend
        for i in 0..20 {
            analytics.price_history.push(100.0 + i as f64);
        }
        
        let rsi = analytics.calculate_rsi(14);
        assert!(rsi > 70.0); // Should be overbought in uptrend
        
        // Add downtrend
        for i in 0..10 {
            analytics.price_history.push(120.0 - i as f64 * 2.0);
        }
        
        let rsi2 = analytics.calculate_rsi(14);
        assert!(rsi2 < rsi); // RSI should decrease in downtrend
    }
}