"""
Kelly Criterion implementation for optimal position sizing in statistical arbitrage.
Includes fractional Kelly and risk constraints.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion calculator for position sizing.
    
    The Kelly formula: f = (p*b - q) / b
    Where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = odds (ratio of win to loss amounts)
    """
    
    def __init__(self, fraction: float = 0.25, max_position_pct: float = 0.02):
        """
        Initialize Kelly calculator.
        
        Args:
            fraction: Fraction of Kelly to use (0.25 = 25% of full Kelly)
            max_position_pct: Maximum position as percentage of bankroll
        """
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.min_edge = 0.01  # Minimum edge required to trade
        
    def calculate_position_size(self, win_rate: float, avg_win: float,
                              avg_loss: float, bankroll: float,
                              confidence: float = 1.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            bankroll: Total available capital
            confidence: Confidence in the edge (0-1)
            
        Returns:
            Suggested position size in currency units
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win rate: {win_rate}")
            return 0
        
        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid win/loss amounts: {avg_win}/{avg_loss}")
            return 0
        
        # Calculate Kelly percentage
        kelly_pct = self._calculate_kelly_percentage(
            win_rate, avg_win, avg_loss
        )
        
        # Apply confidence adjustment
        kelly_pct *= confidence
        
        # Apply fractional Kelly
        kelly_pct *= self.fraction
        
        # Apply maximum position limit
        kelly_pct = min(kelly_pct, self.max_position_pct)
        
        # Calculate position size
        position_size = kelly_pct * bankroll
        
        return position_size
    
    def _calculate_kelly_percentage(self, win_rate: float, 
                                  avg_win: float, avg_loss: float) -> float:
        """
        Calculate raw Kelly percentage.
        
        Args:
            win_rate: Probability of winning (p)
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly percentage (0-1)
        """
        # Calculate odds (b)
        odds = avg_win / avg_loss
        
        # Kelly formula: f = (p*b - q) / b
        q = 1 - win_rate
        kelly = (win_rate * odds - q) / odds
        
        # Ensure non-negative
        kelly = max(0, kelly)
        
        # Cap at 100% (no leverage)
        kelly = min(1.0, kelly)
        
        return kelly
    
    def calculate_kelly_with_multiple_outcomes(self, 
                                             outcomes: Dict[float, float]) -> float:
        """
        Calculate Kelly for multiple possible outcomes.
        
        Args:
            outcomes: Dict of {return: probability} for each outcome
            
        Returns:
            Kelly percentage
        """
        # Ensure probabilities sum to 1
        total_prob = sum(outcomes.values())
        if abs(total_prob - 1.0) > 0.001:
            logger.warning(f"Probabilities sum to {total_prob}, normalizing")
            outcomes = {k: v/total_prob for k, v in outcomes.items()}
        
        # Calculate expected value
        expected_return = sum(ret * prob for ret, prob in outcomes.items())
        
        # Calculate variance
        variance = sum(prob * (ret - expected_return)**2 
                      for ret, prob in outcomes.items())
        
        # Kelly for multiple outcomes: f = μ / σ²
        if variance > 0:
            kelly = expected_return / variance
        else:
            kelly = 0
        
        # Apply constraints
        kelly = max(0, min(1.0, kelly))
        
        return kelly
    
    def calculate_kelly_with_correlation(self, positions: list,
                                       correlation_matrix: np.ndarray) -> Dict[int, float]:
        """
        Calculate Kelly sizes for correlated positions.
        
        Args:
            positions: List of position dictionaries with win_rate, avg_win, avg_loss
            correlation_matrix: Correlation matrix between positions
            
        Returns:
            Dict of position index to Kelly percentage
        """
        n_positions = len(positions)
        
        # Calculate individual Kelly percentages
        individual_kellys = []
        for pos in positions:
            kelly = self._calculate_kelly_percentage(
                pos['win_rate'], 
                pos['avg_win'], 
                pos['avg_loss']
            )
            individual_kellys.append(kelly)
        
        # Adjust for correlation
        if n_positions == 1:
            return {0: individual_kellys[0]}
        
        # Simple correlation adjustment (reduces size for correlated positions)
        adjusted_kellys = {}
        
        for i in range(n_positions):
            # Calculate average correlation with other positions
            avg_correlation = 0
            count = 0
            
            for j in range(n_positions):
                if i != j:
                    avg_correlation += abs(correlation_matrix[i, j])
                    count += 1
            
            if count > 0:
                avg_correlation /= count
            
            # Reduce Kelly size based on correlation
            # High correlation = smaller position
            correlation_penalty = 1 - (avg_correlation * 0.5)
            adjusted_kellys[i] = individual_kellys[i] * correlation_penalty
        
        return adjusted_kellys
    
    def calculate_portfolio_heat(self, positions: list, 
                               correlations: np.ndarray) -> float:
        """
        Calculate total portfolio heat (risk).
        
        Args:
            positions: List of position sizes
            correlations: Correlation matrix
            
        Returns:
            Portfolio heat as percentage
        """
        n = len(positions)
        if n == 0:
            return 0
        
        # Convert to numpy array
        positions = np.array(positions)
        
        # Portfolio variance considering correlations
        portfolio_variance = 0
        
        for i in range(n):
            for j in range(n):
                portfolio_variance += (positions[i] * positions[j] * 
                                     correlations[i, j])
        
        # Portfolio heat is square root of variance
        portfolio_heat = np.sqrt(portfolio_variance)
        
        return portfolio_heat
    
    def calculate_optimal_leverage(self, sharpe_ratio: float,
                                 return_frequency: str = 'daily') -> float:
        """
        Calculate optimal leverage based on Sharpe ratio.
        
        Kelly leverage = Sharpe² / 2 (for Gaussian returns)
        
        Args:
            sharpe_ratio: Annualized Sharpe ratio
            return_frequency: 'daily', 'hourly', etc.
            
        Returns:
            Optimal leverage multiplier
        """
        # Convert Sharpe to appropriate frequency
        if return_frequency == 'daily':
            sharpe_daily = sharpe_ratio / np.sqrt(252)
        elif return_frequency == 'hourly':
            sharpe_daily = sharpe_ratio / np.sqrt(252 * 24)
        else:
            sharpe_daily = sharpe_ratio / np.sqrt(252)
        
        # Kelly leverage formula
        optimal_leverage = (sharpe_daily ** 2) / 2
        
        # Apply reasonable limits
        optimal_leverage = max(0.5, min(3.0, optimal_leverage))
        
        return optimal_leverage
    
    def get_sizing_recommendation(self, pair_stats: Dict[str, Any],
                                market_conditions: Dict[str, Any],
                                portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive sizing recommendation considering all factors.
        
        Args:
            pair_stats: Statistics for the trading pair
            market_conditions: Current market volatility, liquidity, etc.
            portfolio_state: Current portfolio exposure and risk
            
        Returns:
            Sizing recommendation with explanation
        """
        # Base Kelly calculation
        base_kelly = self._calculate_kelly_percentage(
            pair_stats['win_rate'],
            pair_stats['avg_win'],
            pair_stats['avg_loss']
        )
        
        # Adjustments
        adjustments = {}
        
        # 1. Confidence adjustment
        confidence = pair_stats.get('confidence', 1.0)
        adjustments['confidence'] = confidence
        
        # 2. Volatility adjustment
        current_vol = market_conditions.get('volatility', 1.0)
        normal_vol = market_conditions.get('normal_volatility', 1.0)
        vol_ratio = current_vol / normal_vol if normal_vol > 0 else 1.0
        vol_adjustment = 1 / np.sqrt(vol_ratio)  # Reduce size in high vol
        adjustments['volatility'] = vol_adjustment
        
        # 3. Liquidity adjustment
        liquidity_score = market_conditions.get('liquidity_score', 1.0)
        adjustments['liquidity'] = min(1.0, liquidity_score)
        
        # 4. Portfolio heat adjustment
        current_heat = portfolio_state.get('heat', 0)
        max_heat = portfolio_state.get('max_heat', 1.0)
        heat_remaining = max(0, (max_heat - current_heat) / max_heat)
        adjustments['portfolio_heat'] = heat_remaining
        
        # 5. Correlation adjustment
        avg_correlation = portfolio_state.get('avg_correlation', 0)
        adjustments['correlation'] = 1 - (avg_correlation * 0.5)
        
        # Calculate final size
        final_kelly = base_kelly
        for adj_name, adj_value in adjustments.items():
            final_kelly *= adj_value
        
        # Apply fractional Kelly
        final_kelly *= self.fraction
        
        # Apply limits
        final_kelly = min(final_kelly, self.max_position_pct)
        
        return {
            'base_kelly': base_kelly,
            'adjustments': adjustments,
            'final_kelly': final_kelly,
            'recommended_size_pct': final_kelly * 100,
            'explanation': self._generate_explanation(
                base_kelly, adjustments, final_kelly
            )
        }
    
    def _generate_explanation(self, base_kelly: float, 
                            adjustments: Dict[str, float],
                            final_kelly: float) -> str:
        """Generate human-readable explanation of sizing decision."""
        explanation = f"Base Kelly: {base_kelly:.1%}\n"
        
        for adj_name, adj_value in adjustments.items():
            impact = (adj_value - 1) * 100
            explanation += f"{adj_name.title()}: {impact:+.1f}%\n"
        
        explanation += f"Final size: {final_kelly:.1%} of capital"
        
        return explanation