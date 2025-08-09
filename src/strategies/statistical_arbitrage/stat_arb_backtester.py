"""
Backtesting framework for statistical arbitrage strategies.
Provides comprehensive performance analysis and walk-forward optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .stat_arb_engine import StatArbEngine, SignalType
from .cointegration_engine import CointegrationEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    profit_factor: float
    calmar_ratio: float
    daily_returns: pd.Series
    trade_history: List[Dict[str, Any]]
    equity_curve: pd.Series
    parameters: Dict[str, Any]


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    in_sample_results: List[BacktestResult]
    out_sample_results: List[BacktestResult]
    best_parameters: List[Dict[str, Any]]
    stability_score: float
    overfitting_score: float


class StatArbBacktester:
    """
    Comprehensive backtesting framework for statistical arbitrage.
    Supports walk-forward analysis and Monte Carlo simulations.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
            slippage: Slippage as percentage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def backtest(self, price_data: Dict[str, pd.DataFrame],
                entry_z: float = 2.0,
                exit_z: float = 0.0,
                stop_z: float = 3.5,
                lookback_days: int = 60,
                max_pairs: int = 10,
                kelly_fraction: float = 0.25) -> BacktestResult:
        """
        Run a single backtest with given parameters.
        
        Args:
            price_data: Historical price data for all symbols
            entry_z: Entry z-score threshold
            exit_z: Exit z-score threshold
            stop_z: Stop loss z-score
            lookback_days: Lookback period for cointegration
            max_pairs: Maximum number of pairs to trade
            kelly_fraction: Fraction of Kelly criterion to use
            
        Returns:
            Backtest results
        """
        # Initialize engine
        engine = StatArbEngine(
            lookback_days=lookback_days,
            entry_z_score=entry_z,
            exit_z_score=exit_z,
            stop_z_score=stop_z,
            max_pairs=max_pairs,
            kelly_fraction=kelly_fraction
        )
        
        # Prepare data
        all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
        
        # Initialize tracking variables
        capital = self.initial_capital
        positions = {}
        trade_history = []
        daily_returns = []
        equity_curve = [capital]
        
        # Warmup period
        warmup_end = lookback_days
        
        # Main backtest loop
        for i in range(warmup_end, len(all_dates)):
            current_date = all_dates[i]
            
            # Get historical data up to current date
            historical_data = {}
            current_prices = {}
            
            for symbol, df in price_data.items():
                if current_date in df.index:
                    # Historical data for analysis
                    hist_end_idx = df.index.get_loc(current_date)
                    hist_start_idx = max(0, hist_end_idx - lookback_days)
                    historical_data[symbol] = df.iloc[hist_start_idx:hist_end_idx]['close']
                    
                    # Current price
                    current_prices[symbol] = df.loc[current_date, 'close']
            
            # Initialize engine with pairs on first iteration after warmup
            if i == warmup_end:
                asyncio.run(engine.initialize(historical_data))
            
            # Update pair statistics periodically
            if i % 20 == 0:  # Every 20 days
                engine.update_pair_statistics(historical_data)
            
            # Get trading signals
            signals = engine.scan_opportunities(current_prices)
            
            # Process signals
            for signal in signals:
                if signal.signal_type in [SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD]:
                    # Open position
                    position = self._open_position(
                        signal, capital, positions, engine, current_date
                    )
                    if position:
                        trade_history.append(position)
                        
                elif signal.signal_type == SignalType.EXIT:
                    # Close position
                    closed_trade = self._close_position(
                        signal, positions, current_prices, current_date
                    )
                    if closed_trade:
                        capital += closed_trade['pnl']
                        trade_history.append(closed_trade)
            
            # Update position values and calculate daily P&L
            daily_pnl = self._update_positions(positions, current_prices, historical_data)
            
            # Calculate daily return
            prev_equity = equity_curve[-1]
            current_equity = capital + sum(p['current_value'] for p in positions.values())
            daily_return = (current_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
            equity_curve.append(current_equity)
        
        # Create results
        daily_returns_series = pd.Series(
            daily_returns, 
            index=all_dates[warmup_end:]
        )
        
        equity_series = pd.Series(
            equity_curve[1:],  # Exclude initial capital
            index=all_dates[warmup_end:]
        )
        
        return self._calculate_metrics(
            daily_returns_series,
            trade_history,
            equity_series,
            {
                'entry_z': entry_z,
                'exit_z': exit_z,
                'stop_z': stop_z,
                'lookback_days': lookback_days,
                'max_pairs': max_pairs,
                'kelly_fraction': kelly_fraction
            }
        )
    
    def walk_forward_analysis(self, price_data: Dict[str, pd.DataFrame],
                            param_grid: Dict[str, List[Any]],
                            in_sample_periods: int = 252,
                            out_sample_periods: int = 63,
                            n_windows: int = 4) -> WalkForwardResult:
        """
        Perform walk-forward analysis to test parameter stability.
        
        Args:
            price_data: Historical price data
            param_grid: Parameter grid for optimization
            in_sample_periods: Days for in-sample optimization
            out_sample_periods: Days for out-of-sample testing
            n_windows: Number of walk-forward windows
            
        Returns:
            Walk-forward analysis results
        """
        # Get date range
        all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
        total_periods = len(all_dates)
        
        # Calculate window size
        window_size = in_sample_periods + out_sample_periods
        step_size = (total_periods - window_size) // (n_windows - 1)
        
        in_sample_results = []
        out_sample_results = []
        best_parameters = []
        
        # Perform walk-forward analysis
        for window in range(n_windows):
            start_idx = window * step_size
            in_sample_end = start_idx + in_sample_periods
            out_sample_end = in_sample_end + out_sample_periods
            
            if out_sample_end > total_periods:
                break
            
            # Split data
            in_sample_data = {}
            out_sample_data = {}
            
            for symbol, df in price_data.items():
                in_sample_data[symbol] = df.iloc[start_idx:in_sample_end]
                out_sample_data[symbol] = df.iloc[in_sample_end:out_sample_end]
            
            # Optimize on in-sample data
            best_params, best_result = self._optimize_parameters(
                in_sample_data, param_grid
            )
            in_sample_results.append(best_result)
            best_parameters.append(best_params)
            
            # Test on out-of-sample data
            oos_result = self.backtest(out_sample_data, **best_params)
            out_sample_results.append(oos_result)
        
        # Calculate stability metrics
        stability_score = self._calculate_stability_score(
            in_sample_results, out_sample_results
        )
        
        overfitting_score = self._calculate_overfitting_score(
            in_sample_results, out_sample_results
        )
        
        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_sample_results=out_sample_results,
            best_parameters=best_parameters,
            stability_score=stability_score,
            overfitting_score=overfitting_score
        )
    
    def monte_carlo_simulation(self, backtest_result: BacktestResult,
                             n_simulations: int = 1000,
                             confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            backtest_result: Original backtest result
            n_simulations: Number of simulations to run
            confidence_levels: Confidence levels for statistics
            
        Returns:
            Monte Carlo simulation results
        """
        trade_returns = []
        
        # Extract trade returns from history
        for trade in backtest_result.trade_history:
            if 'return' in trade:
                trade_returns.append(trade['return'])
        
        if not trade_returns:
            return {}
        
        trade_returns = np.array(trade_returns)
        n_trades = len(trade_returns)
        
        # Run simulations
        simulation_results = []
        
        for _ in range(n_simulations):
            # Bootstrap sample trades
            sampled_returns = np.random.choice(
                trade_returns, size=n_trades, replace=True
            )
            
            # Calculate metrics for this simulation
            total_return = np.prod(1 + sampled_returns) - 1
            sharpe = np.sqrt(252) * np.mean(sampled_returns) / np.std(sampled_returns)
            
            # Calculate drawdown
            equity = np.cumprod(1 + sampled_returns)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_dd = np.min(drawdown)
            
            simulation_results.append({
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })
        
        # Calculate statistics
        results_df = pd.DataFrame(simulation_results)
        
        stats = {}
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            stats[metric] = {
                'mean': results_df[metric].mean(),
                'std': results_df[metric].std(),
                'percentiles': {}
            }
            
            for cl in confidence_levels:
                stats[metric]['percentiles'][f'{cl:.0%}'] = (
                    results_df[metric].quantile(cl)
                )
        
        return {
            'statistics': stats,
            'simulations': results_df,
            'probability_of_loss': (results_df['total_return'] < 0).mean(),
            'expected_shortfall': results_df['total_return'][
                results_df['total_return'] < results_df['total_return'].quantile(0.05)
            ].mean()
        }
    
    def _open_position(self, signal, capital, positions, engine, current_date):
        """Open a new position based on signal."""
        # Calculate position size
        available_capital = capital * 0.9  # Keep 10% cash reserve
        position_size = signal.suggested_size * available_capital
        
        # Apply transaction costs
        cost = position_size * self.transaction_cost
        
        # Create position record
        position = {
            'pair': signal.pair,
            'entry_date': current_date,
            'entry_z_score': signal.z_score,
            'entry_spread': signal.spread_value,
            'hedge_ratio': signal.hedge_ratio,
            'position_size': position_size - cost,
            'direction': signal.signal_type,
            'entry_prices': {
                signal.pair[0]: signal.entry_price1,
                signal.pair[1]: signal.entry_price2
            }
        }
        
        positions[signal.pair] = position
        engine.open_position(signal, position_size)
        
        return position
    
    def _close_position(self, signal, positions, current_prices, current_date):
        """Close an existing position."""
        if signal.pair not in positions:
            return None
        
        position = positions[signal.pair]
        symbol1, symbol2 = signal.pair
        
        # Calculate P&L
        entry_spread = position['entry_spread']
        exit_spread = signal.spread_value
        
        if position['direction'] == SignalType.SHORT_SPREAD:
            spread_return = (entry_spread - exit_spread) / abs(entry_spread)
        else:
            spread_return = (exit_spread - entry_spread) / abs(entry_spread)
        
        # Apply slippage and transaction costs
        spread_return -= self.slippage + self.transaction_cost
        
        pnl = position['position_size'] * spread_return
        
        # Create trade record
        trade = {
            'pair': signal.pair,
            'entry_date': position['entry_date'],
            'exit_date': current_date,
            'duration_days': (current_date - position['entry_date']).days,
            'entry_z_score': position['entry_z_score'],
            'exit_z_score': signal.z_score,
            'return': spread_return,
            'pnl': pnl,
            'exit_reason': signal.metadata.get('exit_reason', 'unknown')
        }
        
        # Remove position
        del positions[signal.pair]
        
        return trade
    
    def _update_positions(self, positions, current_prices, historical_data):
        """Update position values with current prices."""
        total_pnl = 0
        
        for pair, position in positions.items():
            symbol1, symbol2 = pair
            
            if symbol1 in current_prices and symbol2 in current_prices:
                # Calculate current spread
                current_spread = (current_prices[symbol1] - 
                                position['hedge_ratio'] * current_prices[symbol2])
                
                # Calculate unrealized P&L
                entry_spread = position['entry_spread']
                
                if position['direction'] == SignalType.SHORT_SPREAD:
                    spread_return = (entry_spread - current_spread) / abs(entry_spread)
                else:
                    spread_return = (current_spread - entry_spread) / abs(entry_spread)
                
                position['current_value'] = position['position_size'] * (1 + spread_return)
                position['unrealized_pnl'] = position['position_size'] * spread_return
                
                total_pnl += position['unrealized_pnl']
        
        return total_pnl
    
    def _calculate_metrics(self, daily_returns, trade_history, 
                         equity_curve, parameters):
        """Calculate comprehensive backtest metrics."""
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # Sharpe ratio
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if trade_history:
            winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trade_history if t.get('pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / len(trade_history)
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1
            
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else np.inf
            
            durations = [t.get('duration_days', 0) for t in trade_history if 'duration_days' in t]
            avg_duration = np.mean(durations) if durations else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_duration = 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trade_history),
            avg_trade_duration=avg_duration,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            daily_returns=daily_returns,
            trade_history=trade_history,
            equity_curve=equity_curve,
            parameters=parameters
        )
    
    def _optimize_parameters(self, price_data, param_grid):
        """Optimize parameters using grid search."""
        best_sharpe = -np.inf
        best_params = None
        best_result = None
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                result = self.backtest(price_data, **params)
                
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")
                continue
        
        return best_params, best_result
    
    def _calculate_stability_score(self, in_sample_results, out_sample_results):
        """Calculate parameter stability score."""
        if not in_sample_results or not out_sample_results:
            return 0
        
        # Compare Sharpe ratios
        is_sharpes = [r.sharpe_ratio for r in in_sample_results]
        oos_sharpes = [r.sharpe_ratio for r in out_sample_results]
        
        # Calculate correlation
        if len(is_sharpes) > 1:
            correlation = np.corrcoef(is_sharpes, oos_sharpes)[0, 1]
        else:
            correlation = 0
        
        # Calculate consistency
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        
        if avg_is > 0:
            consistency = min(1.0, avg_oos / avg_is)
        else:
            consistency = 0
        
        # Combined score
        stability = (correlation + consistency) / 2
        
        return max(0, min(1, stability))
    
    def _calculate_overfitting_score(self, in_sample_results, out_sample_results):
        """Calculate overfitting score (0 = no overfitting, 1 = severe)."""
        if not in_sample_results or not out_sample_results:
            return 0
        
        # Compare performance degradation
        is_returns = [r.total_return for r in in_sample_results]
        oos_returns = [r.total_return for r in out_sample_results]
        
        avg_is = np.mean(is_returns)
        avg_oos = np.mean(oos_returns)
        
        if avg_is > 0:
            degradation = max(0, (avg_is - avg_oos) / avg_is)
        else:
            degradation = 0
        
        return min(1, degradation)


# Import asyncio here to avoid circular imports
import asyncio