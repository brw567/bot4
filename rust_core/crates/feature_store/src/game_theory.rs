// Game Theory Feature Engineering - Nash Equilibrium & Market Dynamics
// DEEP DIVE: Advanced game-theoretic features for market microstructure
//
// External Research Applied:
// - "Algorithmic and High-Frequency Trading" - Cartea et al. (2015)
// - "Market Microstructure Theory" - O'Hara (1995)
// - "Game Theory and Economic Modelling" - Kreps (1990)
// - "The Microstructure Approach to Exchange Rates" - Lyons (2001)
// - Kyle's Lambda model for price impact
// - Glosten-Milgrom bid-ask spread model

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, instrument};
use parking_lot::RwLock;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Normal, ContinuousCDF};

use crate::{FeatureUpdate, FeatureValue};

/// Game theory feature configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GameTheoryConfig {
    pub enable_nash_equilibrium: bool,
    pub enable_kyle_lambda: bool,
    pub enable_glosten_milgrom: bool,
    pub enable_prisoner_dilemma: bool,
    pub enable_stackelberg: bool,
    pub history_window_ms: i64,
    pub min_players: usize,
    pub max_iterations: usize,
}

impl Default for GameTheoryConfig {
    fn default() -> Self {
        Self {
            enable_nash_equilibrium: true,
            enable_kyle_lambda: true,
            enable_glosten_milgrom: true,
            enable_prisoner_dilemma: true,
            enable_stackelberg: true,
            history_window_ms: 5000, // 5 seconds of microstructure
            min_players: 2,
            max_iterations: 100,
        }
    }
}

/// Game theory feature calculator
pub struct GameTheoryCalculator {
    config: GameTheoryConfig,
    
    // Market state tracking
    order_book_history: Arc<RwLock<VecDeque<OrderBookState>>>,
    trade_history: Arc<RwLock<VecDeque<TradeEvent>>>,
    player_actions: Arc<RwLock<HashMap<String, PlayerHistory>>>,
    
    // Computed equilibria
    nash_equilibria: Arc<RwLock<HashMap<String, NashEquilibrium>>>,
    kyle_lambda: Arc<RwLock<f64>>,
    glosten_milgrom_spread: Arc<RwLock<f64>>,
}

impl GameTheoryCalculator {
    pub fn new(config: GameTheoryConfig) -> Self {
        Self {
            config,
            order_book_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            trade_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            player_actions: Arc::new(RwLock::new(HashMap::new())),
            nash_equilibria: Arc::new(RwLock::new(HashMap::new())),
            kyle_lambda: Arc::new(RwLock::new(0.0)),
            glosten_milgrom_spread: Arc::new(RwLock::new(0.0)),
        }
    }
    
    /// Calculate all game theory features
    #[instrument(skip(self))]
    pub async fn calculate_features(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        
        // Nash Equilibrium features
        if self.config.enable_nash_equilibrium {
            features.extend(self.calculate_nash_equilibrium(symbol, timestamp).await?);
        }
        
        // Kyle's Lambda (price impact)
        if self.config.enable_kyle_lambda {
            features.push(self.calculate_kyle_lambda(symbol, timestamp).await?);
        }
        
        // Glosten-Milgrom spread model
        if self.config.enable_glosten_milgrom {
            features.push(self.calculate_glosten_milgrom(symbol, timestamp).await?);
        }
        
        // Prisoner's Dilemma (market maker competition)
        if self.config.enable_prisoner_dilemma {
            features.extend(self.calculate_prisoner_dilemma(symbol, timestamp).await?);
        }
        
        // Stackelberg game (leader-follower dynamics)
        if self.config.enable_stackelberg {
            features.extend(self.calculate_stackelberg(symbol, timestamp).await?);
        }
        
        Ok(features)
    }
    
    /// Calculate Nash Equilibrium for current market state
    async fn calculate_nash_equilibrium(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        
        // Get recent order book states
        let states = self.get_recent_states(timestamp);
        if states.len() < self.config.min_players {
            return Ok(features);
        }
        
        // Construct payoff matrix
        let payoff_matrix = self.construct_payoff_matrix(&states)?;
        
        // Find mixed strategy Nash equilibrium using Lemke-Howson algorithm
        let equilibrium = self.lemke_howson_algorithm(&payoff_matrix)?;
        
        // Store equilibrium
        self.nash_equilibria.write().insert(
            symbol.to_string(),
            equilibrium.clone(),
        );
        
        // Generate features
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "nash_bid_probability".to_string(),
            value: FeatureValue::Float(equilibrium.bid_probability),
            timestamp,
            metadata: None,
        });
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "nash_ask_probability".to_string(),
            value: FeatureValue::Float(equilibrium.ask_probability),
            timestamp,
            metadata: None,
        });
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "nash_expected_payoff".to_string(),
            value: FeatureValue::Float(equilibrium.expected_payoff),
            timestamp,
            metadata: None,
        });
        
        Ok(features)
    }
    
    /// Calculate Kyle's Lambda (price impact coefficient)
    async fn calculate_kyle_lambda(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        // Kyle's model: ΔP = λ * Q
        // where λ is price impact per unit of volume
        
        let trades = self.trade_history.read();
        let recent_trades: Vec<_> = trades.iter()
            .filter(|t| {
                timestamp.signed_duration_since(t.timestamp).num_milliseconds() 
                    < self.config.history_window_ms
            })
            .collect();
        
        if recent_trades.len() < 10 {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "kyle_lambda".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Calculate price changes and volumes
        let mut price_impacts = Vec::new();
        let mut volumes = Vec::new();
        
        for window in recent_trades.windows(2) {
            let price_change = (window[1].price - window[0].price).abs();
            let volume = window[1].volume;
            
            if volume > 0.0 {
                price_impacts.push(price_change);
                volumes.push(volume);
            }
        }
        
        // Linear regression: price_impact = λ * volume
        let lambda = if !volumes.is_empty() {
            let mean_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
            let mean_impact = price_impacts.iter().sum::<f64>() / price_impacts.len() as f64;
            
            let numerator: f64 = volumes.iter().zip(price_impacts.iter())
                .map(|(v, p)| (v - mean_volume) * (p - mean_impact))
                .sum();
            
            let denominator: f64 = volumes.iter()
                .map(|v| (v - mean_volume).powi(2))
                .sum();
            
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        *self.kyle_lambda.write() = lambda;
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "kyle_lambda".to_string(),
            value: FeatureValue::Float(lambda),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate Glosten-Milgrom bid-ask spread
    async fn calculate_glosten_milgrom(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FeatureUpdate> {
        // GM model: spread based on probability of informed trading
        
        let states = self.get_recent_states(timestamp);
        if states.is_empty() {
            return Ok(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "glosten_milgrom_spread".to_string(),
                value: FeatureValue::Float(0.0),
                timestamp,
                metadata: None,
            });
        }
        
        // Estimate probability of informed trading (PIN)
        let trades = self.trade_history.read();
        let recent_trades: Vec<_> = trades.iter()
            .filter(|t| {
                timestamp.signed_duration_since(t.timestamp).num_milliseconds() 
                    < self.config.history_window_ms
            })
            .collect();
        
        // Calculate order imbalance as proxy for informed trading
        let buy_volume: f64 = recent_trades.iter()
            .filter(|t| t.is_buy)
            .map(|t| t.volume)
            .sum();
            
        let sell_volume: f64 = recent_trades.iter()
            .filter(|t| !t.is_buy)
            .map(|t| t.volume)
            .sum();
        
        let total_volume = buy_volume + sell_volume;
        let order_imbalance = if total_volume > 0.0 {
            (buy_volume - sell_volume).abs() / total_volume
        } else {
            0.0
        };
        
        // PIN estimate (simplified)
        let pin = order_imbalance.min(1.0);
        
        // Calculate theoretical spread
        // Spread = 2 * (V * PIN) / (1 - PIN)
        // where V is asset value uncertainty
        let value_uncertainty = self.estimate_value_uncertainty(&states);
        
        let spread = if pin < 0.99 {
            2.0 * value_uncertainty * pin / (1.0 - pin)
        } else {
            2.0 * value_uncertainty
        };
        
        *self.glosten_milgrom_spread.write() = spread;
        
        Ok(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "glosten_milgrom_spread".to_string(),
            value: FeatureValue::Float(spread),
            timestamp,
            metadata: None,
        })
    }
    
    /// Calculate Prisoner's Dilemma features (market maker competition)
    async fn calculate_prisoner_dilemma(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        
        // Model market makers as players in repeated prisoner's dilemma
        // Cooperate = wide spreads (higher profit)
        // Defect = tight spreads (competitive)
        
        let states = self.get_recent_states(timestamp);
        if states.len() < 2 {
            return Ok(features);
        }
        
        // Calculate average spread over time
        let spreads: Vec<f64> = states.iter()
            .map(|s| s.ask_price - s.bid_price)
            .collect();
        
        let mean_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let spread_variance = spreads.iter()
            .map(|s| (s - mean_spread).powi(2))
            .sum::<f64>() / spreads.len() as f64;
        
        // High variance suggests defection (competition)
        // Low variance suggests cooperation (collusion)
        let cooperation_index = 1.0 / (1.0 + spread_variance.sqrt());
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "market_cooperation_index".to_string(),
            value: FeatureValue::Float(cooperation_index),
            timestamp,
            metadata: None,
        });
        
        // Calculate optimal defection threshold
        let defection_threshold = mean_spread * (1.0 - cooperation_index);
        
        features.push(FeatureUpdate {
            entity_id: symbol.to_string(),
            feature_id: "optimal_defection_spread".to_string(),
            value: FeatureValue::Float(defection_threshold),
            timestamp,
            metadata: None,
        });
        
        Ok(features)
    }
    
    /// Calculate Stackelberg game features (leader-follower)
    async fn calculate_stackelberg(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<FeatureUpdate>> {
        let mut features = Vec::new();
        
        // Identify market leader (largest volume provider)
        let players = self.player_actions.read();
        if players.is_empty() {
            return Ok(features);
        }
        
        // Find leader by volume
        let leader = players.iter()
            .max_by(|a, b| {
                a.1.total_volume.partial_cmp(&b.1.total_volume).unwrap()
            })
            .map(|(id, _)| id.clone());
        
        if let Some(leader_id) = leader {
            // Calculate Stackelberg equilibrium
            // Leader moves first, followers respond optimally
            
            let leader_data = &players[&leader_id];
            let leader_aggression = leader_data.aggression_score();
            
            // Followers' best response
            let follower_response: f64 = players.iter()
                .filter(|(id, _)| *id != &leader_id)
                .map(|(_, p)| p.aggression_score())
                .sum::<f64>() / (players.len() - 1) as f64;
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "stackelberg_leader_aggression".to_string(),
                value: FeatureValue::Float(leader_aggression),
                timestamp,
                metadata: None,
            });
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "stackelberg_follower_response".to_string(),
                value: FeatureValue::Float(follower_response),
                timestamp,
                metadata: None,
            });
            
            // Market inefficiency from Stackelberg dynamics
            let inefficiency = (leader_aggression - follower_response).abs();
            
            features.push(FeatureUpdate {
                entity_id: symbol.to_string(),
                feature_id: "stackelberg_inefficiency".to_string(),
                value: FeatureValue::Float(inefficiency),
                timestamp,
                metadata: None,
            });
        }
        
        Ok(features)
    }
    
    /// Update order book state
    pub fn update_order_book(&self, state: OrderBookState) {
        let mut history = self.order_book_history.write();
        history.push_back(state);
        
        // Keep only recent history
        let cutoff = Utc::now() - Duration::milliseconds(self.config.history_window_ms * 2);
        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Update trade event
    pub fn update_trade(&self, trade: TradeEvent) {
        // Track player actions
        let mut players = self.player_actions.write();
        let player = players.entry(trade.player_id.clone())
            .or_insert_with(PlayerHistory::new);
        
        player.add_trade(&trade);
        
        // Add to history
        let mut history = self.trade_history.write();
        history.push_back(trade);
        
        // Cleanup old history
        let cutoff = Utc::now() - Duration::milliseconds(self.config.history_window_ms * 2);
        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get recent order book states
    fn get_recent_states(&self, timestamp: DateTime<Utc>) -> Vec<OrderBookState> {
        let history = self.order_book_history.read();
        let cutoff = timestamp - Duration::milliseconds(self.config.history_window_ms);
        
        history.iter()
            .filter(|s| s.timestamp >= cutoff && s.timestamp <= timestamp)
            .cloned()
            .collect()
    }
    
    /// Construct payoff matrix for game theory analysis
    fn construct_payoff_matrix(&self, states: &[OrderBookState]) -> Result<DMatrix<f64>> {
        // Simplified 2x2 game: Bid vs Ask strategies
        let mut payoffs = DMatrix::zeros(2, 2);
        
        for state in states {
            let spread = state.ask_price - state.bid_price;
            let mid_price = (state.ask_price + state.bid_price) / 2.0;
            
            // Payoff for (Bid, Bid) - both compete on bid side
            payoffs[(0, 0)] += -spread / 2.0;
            
            // Payoff for (Bid, Ask) - one bids, one asks
            payoffs[(0, 1)] += spread / 4.0;
            
            // Payoff for (Ask, Bid) - one asks, one bids
            payoffs[(1, 0)] += spread / 4.0;
            
            // Payoff for (Ask, Ask) - both compete on ask side
            payoffs[(1, 1)] += -spread / 2.0;
        }
        
        // Normalize by number of observations
        if !states.is_empty() {
            payoffs /= states.len() as f64;
        }
        
        Ok(payoffs)
    }
    
    /// Lemke-Howson algorithm for finding Nash equilibrium
    fn lemke_howson_algorithm(&self, payoff_matrix: &DMatrix<f64>) -> Result<NashEquilibrium> {
        // Simplified version - finds mixed strategy Nash equilibrium
        // For 2x2 game, can solve analytically
        
        let a = payoff_matrix[(0, 0)];
        let b = payoff_matrix[(0, 1)];
        let c = payoff_matrix[(1, 0)];
        let d = payoff_matrix[(1, 1)];
        
        // Mixed strategy probabilities
        let denominator = a - b - c + d;
        
        let bid_probability = if denominator.abs() > 1e-10 {
            (d - b) / denominator
        } else {
            0.5 // Default to equal probability
        };
        
        let ask_probability = 1.0 - bid_probability;
        
        // Expected payoff
        let expected_payoff = bid_probability * (a * bid_probability + b * ask_probability)
            + ask_probability * (c * bid_probability + d * ask_probability);
        
        Ok(NashEquilibrium {
            bid_probability: bid_probability.max(0.0).min(1.0),
            ask_probability: ask_probability.max(0.0).min(1.0),
            expected_payoff,
        })
    }
    
    /// Estimate value uncertainty for Glosten-Milgrom model
    fn estimate_value_uncertainty(&self, states: &[OrderBookState]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        
        let mid_prices: Vec<f64> = states.iter()
            .map(|s| (s.ask_price + s.bid_price) / 2.0)
            .collect();
        
        let mean = mid_prices.iter().sum::<f64>() / mid_prices.len() as f64;
        
        let variance = mid_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / mid_prices.len() as f64;
        
        variance.sqrt()
    }
}

/// Order book state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookState {
    pub timestamp: DateTime<Utc>,
    pub bid_price: f64,
    pub bid_volume: f64,
    pub ask_price: f64,
    pub ask_volume: f64,
    pub depth_imbalance: f64,
}

/// Trade event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    pub timestamp: DateTime<Utc>,
    pub player_id: String,
    pub price: f64,
    pub volume: f64,
    pub is_buy: bool,
}

/// Player history tracking
#[derive(Debug, Clone)]
struct PlayerHistory {
    total_volume: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: usize,
    last_action: Option<DateTime<Utc>>,
}

impl PlayerHistory {
    fn new() -> Self {
        Self {
            total_volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            trade_count: 0,
            last_action: None,
        }
    }
    
    fn add_trade(&mut self, trade: &TradeEvent) {
        self.total_volume += trade.volume;
        if trade.is_buy {
            self.buy_volume += trade.volume;
        } else {
            self.sell_volume += trade.volume;
        }
        self.trade_count += 1;
        self.last_action = Some(trade.timestamp);
    }
    
    fn aggression_score(&self) -> f64 {
        // Higher score = more aggressive trading
        if self.total_volume > 0.0 {
            let imbalance = (self.buy_volume - self.sell_volume).abs() / self.total_volume;
            let frequency = self.trade_count as f64 / self.total_volume.max(1.0);
            imbalance * frequency
        } else {
            0.0
        }
    }
}

/// Nash equilibrium result
#[derive(Debug, Clone)]
struct NashEquilibrium {
    bid_probability: f64,
    ask_probability: f64,
    expected_payoff: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_nash_equilibrium() {
        let config = GameTheoryConfig::default();
        let calculator = GameTheoryCalculator::new(config);
        
        // Add some order book states
        for i in 0..10 {
            calculator.update_order_book(OrderBookState {
                timestamp: Utc::now() - Duration::milliseconds(i * 100),
                bid_price: 100.0 - i as f64 * 0.01,
                bid_volume: 1000.0,
                ask_price: 100.0 + i as f64 * 0.01,
                ask_volume: 1000.0,
                depth_imbalance: 0.0,
            });
        }
        
        let features = calculator.calculate_nash_equilibrium("BTC", Utc::now()).await.unwrap();
        assert!(!features.is_empty());
    }
    
    #[tokio::test]
    async fn test_kyle_lambda() {
        let config = GameTheoryConfig::default();
        let calculator = GameTheoryCalculator::new(config);
        
        // Add trades with varying volumes and prices
        for i in 0..20 {
            calculator.update_trade(TradeEvent {
                timestamp: Utc::now() - Duration::milliseconds(i * 50),
                player_id: format!("player_{}", i % 3),
                price: 100.0 + (i as f64 * 0.1),
                volume: 100.0 * (1.0 + i as f64 * 0.1),
                is_buy: i % 2 == 0,
            });
        }
        
        let feature = calculator.calculate_kyle_lambda("BTC", Utc::now()).await.unwrap();
        
        if let FeatureValue::Float(lambda) = feature.value {
            assert!(lambda >= 0.0); // Lambda should be non-negative
        }
    }
}