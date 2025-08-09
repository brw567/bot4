"""
Statistical Arbitrage Engine with Z-score trading signals and Kelly Criterion sizing.
Implements the complete stat arb trading cycle from pair discovery to execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

from .cointegration_engine import CointegrationEngine, PairCandidate
from .kelly_criterion import KellyCriterion

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG_SPREAD = "long_spread"    # Buy asset1, sell asset2
    SHORT_SPREAD = "short_spread"  # Sell asset1, buy asset2
    EXIT = "exit"                   # Close position
    HOLD = "hold"                   # No action
    

@dataclass
class TradingSignal:
    """Statistical arbitrage trading signal"""
    pair: Tuple[str, str]
    signal_type: SignalType
    z_score: float
    spread_value: float
    hedge_ratio: float
    confidence: float
    suggested_size: float  # Kelly criterion size
    entry_price1: float
    entry_price2: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class StatArbPosition:
    """Active statistical arbitrage position"""
    pair: Tuple[str, str]
    entry_z_score: float
    current_z_score: float
    entry_spread: float
    current_spread: float
    hedge_ratio: float
    position_size: float
    entry_time: datetime
    pnl: float
    pnl_percent: float
    days_held: int
    max_z_score: float  # Track maximum deviation


class StatArbEngine:
    """
    Complete statistical arbitrage trading engine.
    Handles pair discovery, signal generation, and position management.
    """
    
    def __init__(self, 
                 lookback_days: int = 60,
                 entry_z_score: float = 2.0,
                 exit_z_score: float = 0.0,
                 stop_z_score: float = 3.5,
                 max_pairs: int = 10,
                 kelly_fraction: float = 0.25):
        """
        Initialize statistical arbitrage engine.
        
        Args:
            lookback_days: Days of history for analysis
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            stop_z_score: Stop loss z-score
            max_pairs: Maximum number of pairs to trade
            kelly_fraction: Fraction of Kelly criterion to use
        """
        self.lookback_days = lookback_days
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_z_score = stop_z_score
        self.max_pairs = max_pairs
        self.kelly_fraction = kelly_fraction
        
        # Initialize components
        self.cointegration_engine = CointegrationEngine(lookback_days)
        self.kelly_calculator = KellyCriterion(fraction=kelly_fraction)
        
        # State tracking
        self.active_pairs: List[PairCandidate] = []
        self.positions: Dict[Tuple[str, str], StatArbPosition] = {}
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
        
    async def initialize(self, historical_prices: Dict[str, pd.Series]):
        """
        Initialize engine with historical data and discover pairs.
        
        Args:
            historical_prices: Historical price data for universe
        """
        logger.info("Initializing statistical arbitrage engine...")
        
        # Discover cointegrated pairs
        self.active_pairs = self.cointegration_engine.find_cointegrated_pairs(
            historical_prices,
            min_half_life=5,
            max_half_life=20
        )
        
        # Keep only top pairs
        self.active_pairs = self.active_pairs[:self.max_pairs]
        
        logger.info(f"Found {len(self.active_pairs)} tradeable pairs")
        
        # Calculate historical performance for Kelly sizing
        for pair in self.active_pairs:
            self._calculate_pair_statistics(pair, historical_prices)
    
    def scan_opportunities(self, current_prices: Dict[str, float],
                         min_confidence: float = 0.7) -> List[TradingSignal]:
        """
        Scan for trading opportunities in active pairs.
        
        Args:
            current_prices: Current market prices
            min_confidence: Minimum confidence for signals
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for pair_candidate in self.active_pairs:
            symbol1, symbol2 = pair_candidate.symbols
            
            if symbol1 not in current_prices or symbol2 not in current_prices:
                continue
            
            # Calculate current spread and z-score
            price1 = current_prices[symbol1]
            price2 = current_prices[symbol2]
            
            spread, z_score = self._calculate_spread_metrics(
                price1, price2, pair_candidate
            )
            
            # Check if pair is already in position
            if pair_candidate.symbols in self.positions:
                signal = self._check_exit_signal(
                    pair_candidate, z_score, spread, price1, price2
                )
            else:
                signal = self._check_entry_signal(
                    pair_candidate, z_score, spread, price1, price2
                )
            
            if signal and signal.confidence >= min_confidence:
                signals.append(signal)
                self.signal_history.append(signal)
        
        return signals
    
    def _calculate_spread_metrics(self, price1: float, price2: float,
                                pair: PairCandidate) -> Tuple[float, float]:
        """
        Calculate spread and z-score for a pair.
        
        Args:
            price1: Current price of asset 1
            price2: Current price of asset 2
            pair: Pair candidate with statistics
            
        Returns:
            Tuple of (spread, z_score)
        """
        # Get stored statistics (would be from database in production)
        hedge_ratio = getattr(pair, 'hedge_ratio', 1.0)
        spread_mean = getattr(pair, 'spread_mean', 0.0)
        spread_std = getattr(pair, 'spread_std', 1.0)
        
        # Calculate spread
        spread = price1 - hedge_ratio * price2
        
        # Calculate z-score
        z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        return spread, z_score
    
    def _check_entry_signal(self, pair: PairCandidate, z_score: float,
                          spread: float, price1: float, price2: float) -> Optional[TradingSignal]:
        """
        Check for entry signals.
        
        Args:
            pair: Pair candidate
            z_score: Current z-score
            spread: Current spread value
            price1: Current price of asset 1
            price2: Current price of asset 2
            
        Returns:
            Trading signal if conditions met
        """
        signal_type = None
        
        if z_score >= self.entry_z_score:
            signal_type = SignalType.SHORT_SPREAD
        elif z_score <= -self.entry_z_score:
            signal_type = SignalType.LONG_SPREAD
        
        if signal_type:
            # Calculate Kelly size
            kelly_size = self.kelly_calculator.calculate_position_size(
                win_rate=pair.win_rate,
                avg_win=getattr(pair, 'avg_win', 0.02),
                avg_loss=getattr(pair, 'avg_loss', 0.01),
                bankroll=1.0  # Normalized
            )
            
            # Calculate confidence based on z-score extremity and pair quality
            confidence = min(1.0, (abs(z_score) - self.entry_z_score) / 2 + 
                           pair.cointegration_score * 0.5)
            
            return TradingSignal(
                pair=pair.symbols,
                signal_type=signal_type,
                z_score=z_score,
                spread_value=spread,
                hedge_ratio=getattr(pair, 'hedge_ratio', 1.0),
                confidence=confidence,
                suggested_size=kelly_size,
                entry_price1=price1,
                entry_price2=price2,
                timestamp=datetime.now(),
                metadata={
                    'half_life': pair.half_life,
                    'sharpe_ratio': pair.sharpe_ratio,
                    'cointegration_score': pair.cointegration_score
                }
            )
        
        return None
    
    def _check_exit_signal(self, pair: PairCandidate, z_score: float,
                         spread: float, price1: float, price2: float) -> Optional[TradingSignal]:
        """
        Check for exit signals on existing positions.
        
        Args:
            pair: Pair candidate
            z_score: Current z-score
            spread: Current spread value
            price1: Current price of asset 1
            price2: Current price of asset 2
            
        Returns:
            Trading signal if exit conditions met
        """
        position = self.positions.get(pair.symbols)
        if not position:
            return None
        
        # Update position metrics
        position.current_z_score = z_score
        position.current_spread = spread
        position.max_z_score = max(abs(z_score), position.max_z_score)
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Mean reversion exit
        if abs(z_score) <= self.exit_z_score:
            should_exit = True
            exit_reason = "mean_reversion"
        
        # Stop loss
        elif abs(z_score) >= self.stop_z_score:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Time-based exit (if held too long)
        elif position.days_held > pair.half_life * 3:
            should_exit = True
            exit_reason = "time_stop"
        
        # Profit target (optional)
        elif position.pnl_percent > 0.05:  # 5% profit
            should_exit = True
            exit_reason = "profit_target"
        
        if should_exit:
            return TradingSignal(
                pair=pair.symbols,
                signal_type=SignalType.EXIT,
                z_score=z_score,
                spread_value=spread,
                hedge_ratio=position.hedge_ratio,
                confidence=0.9,  # High confidence for exits
                suggested_size=position.position_size,  # Exit full position
                entry_price1=price1,
                entry_price2=price2,
                timestamp=datetime.now(),
                metadata={
                    'exit_reason': exit_reason,
                    'pnl': position.pnl,
                    'pnl_percent': position.pnl_percent,
                    'days_held': position.days_held
                }
            )
        
        return None
    
    def open_position(self, signal: TradingSignal, actual_size: float):
        """
        Record opening of a new position.
        
        Args:
            signal: Trading signal that triggered the position
            actual_size: Actual position size taken
        """
        position = StatArbPosition(
            pair=signal.pair,
            entry_z_score=signal.z_score,
            current_z_score=signal.z_score,
            entry_spread=signal.spread_value,
            current_spread=signal.spread_value,
            hedge_ratio=signal.hedge_ratio,
            position_size=actual_size,
            entry_time=signal.timestamp,
            pnl=0,
            pnl_percent=0,
            days_held=0,
            max_z_score=abs(signal.z_score)
        )
        
        self.positions[signal.pair] = position
        self.performance_metrics['total_trades'] += 1
        
        logger.info(f"Opened position in {signal.pair} at z-score {signal.z_score:.2f}")
    
    def close_position(self, signal: TradingSignal, exit_prices: Dict[str, float]):
        """
        Record closing of a position and calculate P&L.
        
        Args:
            signal: Exit signal
            exit_prices: Exit prices for both assets
        """
        position = self.positions.get(signal.pair)
        if not position:
            return
        
        # Calculate P&L (simplified - in practice would use actual fill prices)
        symbol1, symbol2 = signal.pair
        
        if position.entry_z_score > 0:  # Was short spread
            pnl_pct = (position.entry_spread - signal.spread_value) / abs(position.entry_spread)
        else:  # Was long spread
            pnl_pct = (signal.spread_value - position.entry_spread) / abs(position.entry_spread)
        
        position.pnl_percent = pnl_pct
        position.pnl = pnl_pct * position.position_size
        
        # Update performance metrics
        self.performance_metrics['total_pnl'] += position.pnl
        if position.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        
        # Remove position
        del self.positions[signal.pair]
        
        logger.info(f"Closed position in {signal.pair} with {pnl_pct:.2%} P&L")
    
    def calculate_kelly_size(self, pair_stats: Dict[str, float],
                           capital: float, max_position_pct: float = 0.02) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            pair_stats: Historical statistics for the pair
            capital: Total available capital
            max_position_pct: Maximum position as percent of capital
            
        Returns:
            Suggested position size in currency
        """
        kelly_pct = self.kelly_calculator.calculate_position_size(
            win_rate=pair_stats.get('win_rate', 0.5),
            avg_win=pair_stats.get('avg_win', 0.02),
            avg_loss=pair_stats.get('avg_loss', 0.01),
            bankroll=capital
        )
        
        # Apply maximum position limit
        max_size = capital * max_position_pct
        suggested_size = min(kelly_pct * capital, max_size)
        
        return suggested_size
    
    def get_active_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of all active positions.
        
        Returns:
            Dictionary with position summaries
        """
        summary = {
            'total_positions': len(self.positions),
            'total_exposure': sum(p.position_size for p in self.positions.values()),
            'total_pnl': sum(p.pnl for p in self.positions.values()),
            'positions': []
        }
        
        for pair, position in self.positions.items():
            summary['positions'].append({
                'pair': pair,
                'z_score': position.current_z_score,
                'pnl_percent': position.pnl_percent,
                'days_held': position.days_held,
                'size': position.position_size
            })
        
        return summary
    
    def update_pair_statistics(self, price_history: Dict[str, pd.Series]):
        """
        Update statistics for all active pairs.
        
        Args:
            price_history: Recent price history
        """
        updated_pairs = []
        
        for pair in self.active_pairs:
            updated = self.cointegration_engine.update_pair_metrics(
                pair, price_history
            )
            if updated:
                updated_pairs.append(updated)
        
        self.active_pairs = updated_pairs[:self.max_pairs]
    
    def _calculate_pair_statistics(self, pair: PairCandidate,
                                 historical_prices: Dict[str, pd.Series]):
        """
        Calculate historical statistics for Kelly sizing.
        
        Args:
            pair: Pair candidate
            historical_prices: Historical price data
        """
        symbol1, symbol2 = pair.symbols
        
        if symbol1 not in historical_prices or symbol2 not in historical_prices:
            return
        
        # This would typically involve a more sophisticated backtest
        # For now, using the metrics from pair evaluation
        setattr(pair, 'win_rate', pair.win_rate)
        setattr(pair, 'avg_win', max(pair.avg_profit, 0.01))
        setattr(pair, 'avg_loss', 0.01)  # Conservative estimate
        setattr(pair, 'hedge_ratio', 1.0)  # Would be calculated properly
        setattr(pair, 'spread_mean', 0.0)
        setattr(pair, 'spread_std', 1.0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance metrics and statistics
        """
        total_trades = self.performance_metrics['total_trades']
        
        if total_trades > 0:
            win_rate = self.performance_metrics['winning_trades'] / total_trades
            avg_pnl = self.performance_metrics['total_pnl'] / total_trades
        else:
            win_rate = 0
            avg_pnl = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': self.performance_metrics['total_pnl'],
            'average_pnl': avg_pnl,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'active_pairs': len(self.active_pairs),
            'active_positions': len(self.positions)
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from signal history."""
        if len(self.signal_history) < 2:
            return 0
        
        # Extract returns from closed positions
        returns = []
        for signal in self.signal_history:
            if signal.signal_type == SignalType.EXIT:
                returns.append(signal.metadata.get('pnl_percent', 0))
        
        if len(returns) < 2:
            return 0
        
        return np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from P&L history."""
        if not self.signal_history:
            return 0
        
        # Build cumulative P&L curve
        cumulative_pnl = []
        current_pnl = 0
        
        for signal in self.signal_history:
            if signal.signal_type == SignalType.EXIT:
                current_pnl += signal.metadata.get('pnl', 0)
                cumulative_pnl.append(current_pnl)
        
        if not cumulative_pnl:
            return 0
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        
        return abs(np.min(drawdown))