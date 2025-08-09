"""
Dynamic Strategy Matrix Selector

This module implements dynamic strategy selection based on market conditions,
volatility regimes, and trigger conditions to optimize win rate to 80%.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from sympy import sympify, Symbol
from sympy.parsing.sympy_parser import parse_expr

logger = logging.getLogger(__name__)

# Strategy matrices configuration
STRATEGY_MATRICES = {
    'spot': {
        'low_volatility': {
            'base': ['Grid', 'RSI'],
            'triggers': {
                'oi_change > 0.15': ['Heatmap'],
                'funding_rate > 0.01': ['DeltaNeutral']
            }
        },
        'medium_volatility': {
            'base': ['EMA', 'RSI', 'FVG'],
            'triggers': {
                'drawdown > 0.02': ['EMA']  # Fallback to safe strategy
            }
        },
        'high_volatility': {
            'base': ['MEV', 'FVG'],
            'triggers': {
                'mempool_density > 0.8': ['Arbitrage']
            }
        }
    },
    'futures': {
        'low_volatility': {
            'base': ['EMA', 'RSI'],
            'triggers': {
                'oi_change > 0.15': ['Heatmap']
            }
        },
        'medium_volatility': {
            'base': ['Bollinger', 'Heatmap'],
            'triggers': {
                'funding_rate > 0.01': ['DeltaNeutral']
            }
        },
        'high_volatility': {
            'base': ['Bollinger', 'Heatmap', 'FVG'],
            'triggers': {
                'drawdown > 0.02': ['EMA', 'RSI']  # Conservative fallback
            }
        }
    }
}

# Strategy optimization priorities
STRATEGY_PRIORITIES = {
    'Grid': {'risk': 'low', 'profit': 'medium', 'complexity': 'low'},
    'RSI': {'risk': 'low', 'profit': 'medium', 'complexity': 'low'},
    'EMA': {'risk': 'low', 'profit': 'low', 'complexity': 'low'},
    'Bollinger': {'risk': 'medium', 'profit': 'medium', 'complexity': 'medium'},
    'Heatmap': {'risk': 'medium', 'profit': 'high', 'complexity': 'high'},
    'FVG': {'risk': 'high', 'profit': 'high', 'complexity': 'medium'},
    'MEV': {'risk': 'high', 'profit': 'very_high', 'complexity': 'very_high'},
    'Arbitrage': {'risk': 'medium', 'profit': 'high', 'complexity': 'medium'},
    'DeltaNeutral': {'risk': 'very_low', 'profit': 'medium', 'complexity': 'medium'}
}


class DynamicStrategySelector:
    """
    Dynamically selects optimal trading strategies based on:
    - Pair type (spot/futures)
    - Current volatility regime
    - Active market triggers
    - Risk/reward optimization
    """
    
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.015,      # < 1.5% daily volatility
            'medium': 0.035,   # 1.5% - 3.5% daily volatility
            'high': float('inf')  # > 3.5% daily volatility
        }
        
        # Safe symbols for condition evaluation
        self.safe_symbols = {
            'oi_change': Symbol('oi_change'),
            'funding_rate': Symbol('funding_rate'),
            'drawdown': Symbol('drawdown'),
            'mempool_density': Symbol('mempool_density'),
            'volatility': Symbol('volatility'),
            'volume_ratio': Symbol('volume_ratio'),
            'momentum': Symbol('momentum')
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.regime_history = []
        
    async def select_strategies(self, pair_config: Dict) -> List[str]:
        """
        Select optimal strategies for a trading pair.
        
        Args:
            pair_config: Configuration including:
                - pair: Trading pair symbol
                - type: 'spot' or 'futures'
                - volatility: Current volatility
                - metrics: Dict of market metrics
                
        Returns:
            List of selected strategy names
        """
        try:
            pair_type = pair_config.get('type', 'spot')
            volatility = pair_config.get('volatility', 0.02)
            metrics = pair_config.get('metrics', {})
            
            # Determine volatility regime
            volatility_regime = self.get_volatility_regime(volatility)
            
            # Log regime detection
            logger.info(f"Pair: {pair_config.get('pair')} | Type: {pair_type} | "
                       f"Volatility: {volatility:.3f} | Regime: {volatility_regime}")
            
            # Get base strategies for regime
            if pair_type not in STRATEGY_MATRICES:
                logger.warning(f"Unknown pair type: {pair_type}, defaulting to spot")
                pair_type = 'spot'
            
            matrix = STRATEGY_MATRICES[pair_type].get(volatility_regime, {})
            base_strategies = matrix.get('base', ['EMA', 'RSI']).copy()
            
            # Apply dynamic triggers
            triggers = matrix.get('triggers', {})
            triggered_strategies = []
            
            for condition, strategies in triggers.items():
                if self.evaluate_condition(condition, metrics):
                    logger.info(f"Trigger activated: {condition} -> Adding {strategies}")
                    triggered_strategies.extend(strategies)
            
            # Combine and optimize strategies
            all_strategies = base_strategies + triggered_strategies
            optimized_strategies = self.optimize_combination(
                all_strategies, 
                pair_config.get('risk_tolerance', 'medium')
            )
            
            # Record selection for performance tracking
            self._record_selection(
                pair_config.get('pair'),
                volatility_regime,
                optimized_strategies,
                metrics
            )
            
            return optimized_strategies
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            # Return safe default strategies
            return ['EMA', 'RSI']
    
    def get_volatility_regime(self, volatility: float) -> str:
        """
        Determine volatility regime from current volatility.
        
        Args:
            volatility: Current volatility (0-1 scale)
            
        Returns:
            'low_volatility', 'medium_volatility', or 'high_volatility'
        """
        if volatility < self.volatility_thresholds['low']:
            return 'low_volatility'
        elif volatility < self.volatility_thresholds['medium']:
            return 'medium_volatility'
        else:
            return 'high_volatility'
    
    def evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """
        Safely evaluate trigger conditions using sympy.
        
        Args:
            condition: Condition string (e.g., 'oi_change > 0.15')
            metrics: Current market metrics
            
        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Replace Python keywords with sympy equivalents
            condition_safe = condition.replace(' and ', ' & ').replace(' or ', ' | ')
            
            # Parse condition safely
            expr = parse_expr(
                condition_safe,
                local_dict=self.safe_symbols,
                transformations='all'
            )
            
            # Substitute values
            substitutions = {}
            for var_name, symbol in self.safe_symbols.items():
                if var_name in metrics:
                    substitutions[symbol] = metrics[var_name]
                else:
                    # Default values for missing metrics
                    substitutions[symbol] = 0.0
            
            # Evaluate condition
            result = expr.subs(substitutions)
            
            # Convert to boolean
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Condition evaluation failed for '{condition}': {e}")
            return False
    
    def optimize_combination(
        self, 
        strategies: List[str], 
        risk_tolerance: str = 'medium'
    ) -> List[str]:
        """
        Optimize strategy combination based on risk/reward profile.
        
        Args:
            strategies: List of candidate strategies
            risk_tolerance: 'low', 'medium', or 'high'
            
        Returns:
            Optimized list of strategies
        """
        # Remove duplicates while preserving order
        unique_strategies = []
        seen = set()
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        # Score strategies based on risk tolerance
        risk_weights = {
            'low': {'risk': -2.0, 'profit': 1.0, 'complexity': -1.0},
            'medium': {'risk': -1.0, 'profit': 2.0, 'complexity': -0.5},
            'high': {'risk': 0.5, 'profit': 3.0, 'complexity': 0.0}
        }
        
        weights = risk_weights.get(risk_tolerance, risk_weights['medium'])
        
        # Calculate scores
        strategy_scores = []
        for strategy in unique_strategies:
            if strategy not in STRATEGY_PRIORITIES:
                logger.warning(f"Unknown strategy: {strategy}")
                continue
            
            props = STRATEGY_PRIORITIES[strategy]
            
            # Convert qualitative to quantitative
            risk_score = {
                'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
            }.get(props['risk'], 2)
            
            profit_score = {
                'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
            }.get(props['profit'], 2)
            
            complexity_score = {
                'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
            }.get(props['complexity'], 2)
            
            # Calculate weighted score
            score = (
                weights['risk'] * risk_score +
                weights['profit'] * profit_score +
                weights['complexity'] * complexity_score
            )
            
            strategy_scores.append((strategy, score))
        
        # Sort by score (descending) and take top strategies
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of strategies based on risk tolerance
        max_strategies = {
            'low': 2,
            'medium': 3,
            'high': 4
        }.get(risk_tolerance, 3)
        
        optimized = [s[0] for s in strategy_scores[:max_strategies]]
        
        # Ensure at least one safe strategy
        if risk_tolerance in ['low', 'medium']:
            safe_strategies = ['EMA', 'RSI', 'Grid']
            if not any(s in optimized for s in safe_strategies):
                optimized.append('RSI')
        
        logger.info(f"Optimized strategies: {optimized} (from {unique_strategies})")
        
        return optimized
    
    def _record_selection(
        self, 
        pair: str, 
        regime: str, 
        strategies: List[str],
        metrics: Dict[str, float]
    ) -> None:
        """Record strategy selection for performance tracking."""
        self.regime_history.append({
            'timestamp': datetime.now(),
            'pair': pair,
            'regime': regime,
            'strategies': strategies,
            'metrics': metrics.copy()
        })
        
        # Keep only last 1000 records
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
    
    def update_strategy_performance(
        self, 
        pair: str, 
        strategy: str, 
        profit: float
    ) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            pair: Trading pair
            strategy: Strategy name
            profit: Profit/loss from trade
        """
        key = f"{pair}_{strategy}"
        
        if key not in self.strategy_performance:
            self.strategy_performance[key] = {
                'trades': 0,
                'total_profit': 0.0,
                'wins': 0,
                'losses': 0
            }
        
        perf = self.strategy_performance[key]
        perf['trades'] += 1
        perf['total_profit'] += profit
        
        if profit > 0:
            perf['wins'] += 1
        else:
            perf['losses'] += 1
        
        # Calculate win rate
        perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection and strategy selection."""
        if not self.regime_history:
            return {}
        
        # Count regime occurrences
        regime_counts = {}
        strategy_counts = {}
        
        for record in self.regime_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            for strategy in record['strategies']:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Calculate regime transitions
        transitions = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i]['regime'] != self.regime_history[i-1]['regime']:
                transitions += 1
        
        return {
            'total_selections': len(self.regime_history),
            'regime_counts': regime_counts,
            'strategy_counts': strategy_counts,
            'regime_transitions': transitions,
            'avg_strategies_per_selection': np.mean([
                len(r['strategies']) for r in self.regime_history
            ])
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        summary = {}
        
        for key, perf in self.strategy_performance.items():
            pair, strategy = key.rsplit('_', 1)
            
            if strategy not in summary:
                summary[strategy] = {
                    'total_trades': 0,
                    'total_profit': 0.0,
                    'avg_win_rate': 0.0,
                    'pairs_traded': set()
                }
            
            summary[strategy]['total_trades'] += perf['trades']
            summary[strategy]['total_profit'] += perf['total_profit']
            summary[strategy]['pairs_traded'].add(pair)
        
        # Calculate average win rates
        for strategy in summary:
            strategy_perfs = [
                perf for key, perf in self.strategy_performance.items()
                if key.endswith(f'_{strategy}')
            ]
            
            if strategy_perfs:
                win_rates = [p['win_rate'] for p in strategy_perfs if p['trades'] > 0]
                if win_rates:
                    summary[strategy]['avg_win_rate'] = np.mean(win_rates)
            
            # Convert set to list for JSON serialization
            summary[strategy]['pairs_traded'] = list(summary[strategy]['pairs_traded'])
        
        return summary


# Global instance
_selector_instance = None

def get_strategy_selector() -> DynamicStrategySelector:
    """Get global strategy selector instance."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = DynamicStrategySelector()
    return _selector_instance