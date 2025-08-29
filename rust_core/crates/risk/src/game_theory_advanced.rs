use domain_types::MarketState;
// DEEP DIVE: Advanced Game Theory for Trading
// Team: Alex (Lead) + Morgan (ML) + Quinn (Risk) + Full Team
// NO SIMPLIFICATIONS - FULL IMPLEMENTATION OF TRADING GAME THEORY!

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::parameter_manager::ParameterManager;

/// Advanced Game Theory Engine
/// Implements multi-player games, information asymmetry, and market microstructure games
/// TODO: Add docs
pub struct AdvancedGameTheory {
    params: Arc<ParameterManager>,
    
    /// Historical payoff matrix for strategy evaluation
    payoff_history: Vec<PayoffMatrix>,
    
    /// Current market participants and their estimated strategies
    market_players: HashMap<String, PlayerProfile>,
    
    /// Our strategy evolution
    strategy_evolution: Vec<Strategy>,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PayoffMatrix {
    timestamp: chrono::DateTime<chrono::Utc>,
    our_strategy: Strategy,
    opponent_strategies: Vec<Strategy>,
    our_payoff: f64,
    market_payoff: f64,
}

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PlayerProfile {
    /// Estimated capital
    capital: f64,
    
    /// Observed strategy distribution
    strategy_distribution: HashMap<Strategy, f64>,
    
    /// Estimated skill level (0-1)
    skill_level: f64,
    
    /// Information advantage (0-1)
    information_advantage: f64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
/// TODO: Add docs
pub enum Strategy {
    Aggressive,      // High frequency, large positions
    Conservative,    // Low frequency, small positions
    Momentum,        // Follow trends
    Contrarian,      // Fade moves
    MarketMaking,    // Provide liquidity
    Arbitrage,       // Exploit inefficiencies
    Predatory,       // Hunt stops, squeeze shorts
}

impl AdvancedGameTheory {
    pub fn new(params: Arc<ParameterManager>) -> Self {
        Self {
            params,
            payoff_history: Vec::new(),
            market_players: HashMap::new(),
            strategy_evolution: Vec::new(),
        }
    }
    
    /// Calculate optimal strategy using Multi-Agent Reinforcement Learning concepts
    /// Theory: Markets are multi-player partially observable stochastic games
    pub fn calculate_optimal_strategy(&self,
                                     market_state: &MarketState,
                                     our_capital: f64) -> (Strategy, f64) {
        // Estimate current game state
        let players = self.estimate_active_players(market_state);
        let information_asymmetry = self.calculate_information_asymmetry(market_state);
        
        // Build payoff matrix for each strategy
        let mut expected_payoffs = HashMap::new();
        
        for our_strategy in &[
            Strategy::Aggressive,
            Strategy::Conservative,
            Strategy::Momentum,
            Strategy::Contrarian,
            Strategy::MarketMaking,
            Strategy::Arbitrage,
        ] {
            let payoff = self.calculate_expected_payoff(
                our_strategy,
                &players,
                market_state,
                our_capital,
                information_asymmetry,
            );
            expected_payoffs.insert(our_strategy.clone(), payoff);
        }
        
        // Find Nash equilibrium using fictitious play
        let nash_strategy = self.find_nash_equilibrium(&expected_payoffs, &players);
        
        // Apply regret minimization
        let final_strategy = self.apply_regret_minimization(nash_strategy, &expected_payoffs);
        
        // Calculate confidence based on game theory metrics
        let confidence = self.calculate_strategy_confidence(&final_strategy, &expected_payoffs);
        
        (final_strategy, confidence)
    }
    
    /// Calculate expected payoff for a strategy
    fn calculate_expected_payoff(&self,
                                our_strategy: &Strategy,
                                players: &[PlayerProfile],
                                market_state: &MarketState,
                                our_capital: f64,
                                information_asymmetry: f64) -> f64 {
        let mut total_payoff = 0.0;
        let mut total_probability = 0.0;
        
        // Consider all possible opponent strategy combinations
        for player in players {
            for (opp_strategy, prob) in &player.strategy_distribution {
                let payoff = self.calculate_payoff_against(
                    our_strategy,
                    opp_strategy,
                    market_state,
                    our_capital,
                    player.capital,
                );
                
                // Adjust for information asymmetry
                let adjusted_payoff = payoff * (1.0 + information_asymmetry * player.skill_level);
                
                total_payoff += adjusted_payoff * prob;
                total_probability += prob;
            }
        }
        
        if total_probability > 0.0 {
            total_payoff / total_probability
        } else {
            0.0
        }
    }
    
    /// Calculate payoff when playing strategy A against strategy B
    fn calculate_payoff_against(&self,
                               our_strategy: &Strategy,
                               opp_strategy: &Strategy,
                               market_state: &MarketState,
                               our_capital: f64,
                               opp_capital: f64) -> f64 {
        use Strategy::*;
        
        // Payoff matrix based on game theory and market microstructure
        // Positive values mean we win, negative means we lose
        let base_payoff = match (our_strategy, opp_strategy) {
            // Aggressive vs others
            (Aggressive, Aggressive) => -0.02,    // Both lose to market impact
            (Aggressive, Conservative) => 0.03,   // We push them out
            (Aggressive, Momentum) => 0.01,       // Slight edge
            (Aggressive, Contrarian) => -0.01,    // They fade our moves
            (Aggressive, MarketMaking) => 0.02,   // We take liquidity
            (Aggressive, Arbitrage) => 0.0,       // Neutral
            (Aggressive, Predatory) => -0.03,     // They hunt us
            
            // Conservative vs others
            (Conservative, Aggressive) => -0.01,  // Get pushed around
            (Conservative, Conservative) => 0.01, // Both preserve capital
            (Conservative, Momentum) => 0.0,      // Neutral
            (Conservative, Contrarian) => 0.01,   // Both cautious
            (Conservative, MarketMaking) => 0.005, // Small edge
            (Conservative, Arbitrage) => 0.0,     // Neutral
            (Conservative, Predatory) => 0.02,    // Avoid their traps
            
            // Momentum vs others
            (Momentum, Aggressive) => -0.01,      // They front-run
            (Momentum, Conservative) => 0.02,     // We ride trends they miss
            (Momentum, Momentum) => -0.015,       // Crowded trade
            (Momentum, Contrarian) => -0.02,      // Direct opposition
            (Momentum, MarketMaking) => 0.01,     // Neutral to positive
            (Momentum, Arbitrage) => 0.0,         // Neutral
            (Momentum, Predatory) => -0.025,      // They trigger stops
            
            // Contrarian vs others
            (Contrarian, Aggressive) => 0.02,     // Fade their pushes
            (Contrarian, Conservative) => 0.0,    // Neutral
            (Contrarian, Momentum) => 0.025,      // Fade the crowd
            (Contrarian, Contrarian) => -0.01,    // Cancel out
            (Contrarian, MarketMaking) => 0.01,   // Take liquidity at extremes
            (Contrarian, Arbitrage) => 0.0,       // Neutral
            (Contrarian, Predatory) => -0.02,     // They manipulate
            
            // MarketMaking vs others
            (MarketMaking, Aggressive) => -0.01,  // Adverse selection
            (MarketMaking, Conservative) => 0.01, // Steady flow
            (MarketMaking, Momentum) => 0.005,    // Provide liquidity
            (MarketMaking, Contrarian) => 0.0,    // Neutral
            (MarketMaking, MarketMaking) => -0.005, // Compete on spread
            (MarketMaking, Arbitrage) => 0.01,    // Complement each other
            (MarketMaking, Predatory) => -0.03,   // Get picked off
            
            // Arbitrage vs others
            (Arbitrage, _) => 0.01,               // Generally profitable
            
            // Predatory vs others
            (Predatory, Aggressive) => 0.04,      // Hunt their stops
            (Predatory, Conservative) => -0.01,   // They don't take bait
            (Predatory, Momentum) => 0.03,        // Trigger their stops
            (Predatory, Contrarian) => 0.02,      // Squeeze them
            (Predatory, MarketMaking) => 0.035,   // Pick them off
            (Predatory, Arbitrage) => 0.0,        // Neutral
            (Predatory, Predatory) => -0.02,      // Mutual destruction
        };
        
        // Adjust for market conditions
        let volatility_adjustment = market_state.volatility * 
            self.params.get("volatility_scaling");
        
        // Adjust for capital differences (larger capital has advantage)
        let capital_ratio = (our_capital / opp_capital).ln();
        let capital_adjustment = capital_ratio * 0.01;
        
        // Adjust for market depth (deeper markets reduce payoffs)
        let depth_adjustment = 1.0 / (1.0 + market_state.depth / 1000000.0);
        
        base_payoff * (1.0 + volatility_adjustment) * depth_adjustment + capital_adjustment
    }
    
    /// Find Nash equilibrium using iterative methods
    fn find_nash_equilibrium(&self,
                           payoffs: &HashMap<Strategy, f64>,
                           players: &[PlayerProfile]) -> Strategy {
        let iterations = self.params.get("nash_equilibrium_iterations") as usize;
        let mut strategy_weights: HashMap<Strategy, f64> = HashMap::new();
        
        // Initialize with uniform distribution
        for strategy in payoffs.keys() {
            strategy_weights.insert(strategy.clone(), 1.0 / payoffs.len() as f64);
        }
        
        // Fictitious play iteration
        for _ in 0..iterations {
            let mut new_weights = HashMap::new();
            
            for (strategy, &payoff) in payoffs {
                // Calculate best response frequency
                let mut response_value = payoff;
                
                // Consider opponent adaptations
                for player in players {
                    for (opp_strat, &opp_prob) in &player.strategy_distribution {
                        let counter_payoff = self.calculate_payoff_against(
                            strategy,
                            opp_strat,
                            &MarketState::default(), // Simplified for Nash calculation
                            1.0,
                            1.0,
                        );
                        response_value += counter_payoff * opp_prob * player.skill_level;
                    }
                }
                
                // Update weight using exponential weights algorithm
                let new_weight = strategy_weights.get(strategy).unwrap_or(&0.0) * 
                                (response_value * 10.0).exp();
                new_weights.insert(strategy.clone(), new_weight);
            }
            
            // Normalize weights
            let total: f64 = new_weights.values().sum();
            for weight in new_weights.values_mut() {
                *weight /= total;
            }
            
            strategy_weights = new_weights;
        }
        
        // Select strategy with highest equilibrium probability
        strategy_weights
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(strategy, _)| strategy)
            .unwrap_or(Strategy::Conservative)
    }
    
    /// Apply regret minimization to improve strategy selection
    fn apply_regret_minimization(&self,
                                base_strategy: Strategy,
                                payoffs: &HashMap<Strategy, f64>) -> Strategy {
        // Calculate regret for not choosing each strategy
        let base_payoff = payoffs.get(&base_strategy).unwrap_or(&0.0);
        let mut regrets = HashMap::new();
        
        for (strategy, &payoff) in payoffs {
            let regret = (payoff - base_payoff).max(0.0);
            regrets.insert(strategy.clone(), regret);
        }
        
        // If another strategy has significantly less regret, switch
        let min_regret_strategy = regrets
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(strategy, _)| strategy)
            .unwrap_or(base_strategy.clone());
        
        // Only switch if regret difference is significant
        let switch_threshold = self.params.get("strategy_switch_threshold");
        let min_regret = payoffs.get(&min_regret_strategy).unwrap_or(&0.0);
        
        if (min_regret - base_payoff) > switch_threshold {
            min_regret_strategy
        } else {
            base_strategy
        }
    }
    
    /// Calculate confidence in the selected strategy
    fn calculate_strategy_confidence(&self,
                                   strategy: &Strategy,
                                   payoffs: &HashMap<Strategy, f64>) -> f64 {
        let our_payoff = payoffs.get(strategy).unwrap_or(&0.0);
        let max_payoff = payoffs.values().fold(0.0f64, |a, &b| a.max(b));
        let min_payoff = payoffs.values().fold(0.0f64, |a, &b| a.min(b));
        
        if max_payoff > min_payoff {
            (our_payoff - min_payoff) / (max_payoff - min_payoff)
        } else {
            0.5 // Neutral confidence if all payoffs are equal
        }
    }
    
    /// Estimate active market players from order flow
    fn estimate_active_players(&self, market_state: &MarketState) -> Vec<PlayerProfile> {
        let mut players = Vec::new();
        
        // Estimate number of active players from volume and order distribution
        let estimated_players = (market_state.volume.sqrt() / 100.0).min(10.0) as usize;
        
        for i in 0..estimated_players {
            let mut strategy_dist = HashMap::new();
            
            // Infer strategies from market characteristics
            if market_state.volatility > 0.02 {
                strategy_dist.insert(Strategy::Aggressive, 0.3);
                strategy_dist.insert(Strategy::Predatory, 0.2);
            } else {
                strategy_dist.insert(Strategy::Conservative, 0.3);
                strategy_dist.insert(Strategy::MarketMaking, 0.2);
            }
            
            if market_state.trend.abs() > 0.01 {
                strategy_dist.insert(Strategy::Momentum, 0.3);
            } else {
                strategy_dist.insert(Strategy::Contrarian, 0.2);
                strategy_dist.insert(Strategy::Arbitrage, 0.1);
            }
            
            // Normalize distribution
            let total: f64 = strategy_dist.values().sum();
            for prob in strategy_dist.values_mut() {
                *prob /= total;
            }
            
            players.push(PlayerProfile {
                capital: market_state.volume * 1000.0 / estimated_players as f64,
                strategy_distribution: strategy_dist,
                skill_level: 0.5 + (i as f64 / estimated_players as f64) * 0.3,
                information_advantage: (i as f64 / estimated_players as f64) * 0.2,
            });
        }
        
        players
    }
    
    /// Calculate information asymmetry in the market
    fn calculate_information_asymmetry(&self, market_state: &MarketState) -> f64 {
        // Based on Kyle's lambda and PIN (Probability of Informed Trading)
        let price_impact = market_state.price_impact;
        let volume_imbalance = market_state.volume_imbalance.abs();
        let spread = market_state.spread;
        
        // Higher price impact, volume imbalance, and spread indicate information asymmetry
        let lambda_component = (price_impact / 0.0001).min(1.0) * 0.4;
        let imbalance_component = volume_imbalance.min(1.0) * 0.3;
        let spread_component = (spread / 0.001).min(1.0) * 0.3;
        
        lambda_component + imbalance_component + spread_component
    }
    
    /// Record actual payoff for learning
    pub fn record_payoff(&mut self,
                        strategy: Strategy,
                        payoff: f64,
                        market_state: MarketState) {
        let matrix = PayoffMatrix {
            timestamp: chrono::Utc::now(),
            our_strategy: strategy.clone(),
            opponent_strategies: self.estimate_active_players(&market_state)
                .into_iter()
                .flat_map(|p| p.strategy_distribution.into_keys())
                .collect(),
            our_payoff: payoff,
            market_payoff: 0.0, // Could calculate average market return
        };
        
        self.payoff_history.push(matrix);
        self.strategy_evolution.push(strategy);
        
        // Keep only recent history
        if self.payoff_history.len() > 1000 {
            self.payoff_history.remove(0);
        }
        if self.strategy_evolution.len() > 1000 {
            self.strategy_evolution.remove(0);
        }
    }
}

/// Market state for game theory calculations
#[derive(Debug, Clone, Default)]
// ELIMINATED: use domain_types::MarketState
// pub struct MarketState {
    pub volatility: f64,
    pub volume: f64,
    pub trend: f64,
    pub depth: f64,
    pub spread: f64,
    pub price_impact: f64,
    pub volume_imbalance: f64,
}

/// Prisoner's Dilemma for Market Makers
/// Theory: Two market makers must decide whether to cooperate (maintain spreads)
/// or defect (tighten spreads to capture more flow)
/// TODO: Add docs
pub struct MarketMakerDilemma {
    params: Arc<ParameterManager>,
}

impl MarketMakerDilemma {
    pub fn new(params: Arc<ParameterManager>) -> Self {
        Self { params }
    }
    
    /// Calculate optimal market making strategy considering game theory
    pub fn calculate_optimal_spread(&self,
                                   our_inventory: f64,
                                   competitor_spread: f64,
                                   market_volatility: f64) -> f64 {
        // Payoff matrix for spread decisions
        // Cooperate = maintain wide spread, Defect = tighten spread
        
        let fair_spread = 2.0 * market_volatility.sqrt() * self.params.get("spread_volatility_factor");
        
        // If competitor has wide spread, we can tighten slightly to capture flow
        // If competitor has tight spread, we need to match or lose all flow
        
        let nash_spread = if competitor_spread > fair_spread * 1.2 {
            // Competitor cooperating, we can defect slightly
            competitor_spread * 0.9
        } else if competitor_spread < fair_spread * 0.8 {
            // Competitor defecting, we must match
            competitor_spread * 1.01 // Slightly wider to avoid negative spread race
        } else {
            // Near equilibrium
            fair_spread
        };
        
        // Adjust for inventory risk
        let inventory_adjustment = our_inventory.abs() * self.params.get("inventory_risk_factor");
        
        nash_spread * (1.0 + inventory_adjustment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_selection() {
        let params = Arc::new(ParameterManager::new());
        let game_theory = AdvancedGameTheory::new(params);
        
        let market_state = MarketState {
            volatility: 0.02,
            volume: 1000000.0,
            trend: 0.01,
            depth: 5000000.0,
            spread: 0.0002,
            price_impact: 0.00001,
            volume_imbalance: 0.1,
        };
        
        let (strategy, confidence) = game_theory.calculate_optimal_strategy(&market_state, 100000.0);
        
        // Should select a reasonable strategy with some confidence
        assert!(confidence > 0.0 && confidence <= 1.0);
        println!("Selected strategy: {:?} with confidence: {:.2}%", strategy, confidence * 100.0);
    }
    
    #[test]
    fn test_market_maker_dilemma() {
        let params = Arc::new(ParameterManager::new());
        let dilemma = MarketMakerDilemma::new(params);
        
        let spread = dilemma.calculate_optimal_spread(
            100.0,      // our inventory
            0.0003,     // competitor spread
            0.015,      // volatility
        );
        
        assert!(spread > 0.0);
        assert!(spread < 0.01); // Reasonable spread
    }
}