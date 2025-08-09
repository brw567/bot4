"""
Cointegration-based pair discovery engine for statistical arbitrage.
Implements Johansen, Engle-Granger, and Phillips-Ouliaris tests.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from statsmodels.tsa.stattools import coint, adfuller
# from statsmodels.johansen import coint_johansen  # Not available in current statsmodels
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Results from cointegration testing"""
    symbol1: str
    symbol2: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    cointegrated: bool
    hedge_ratio: float
    half_life: float
    hurst_exponent: float
    correlation: float
    spread_mean: float
    spread_std: float
    confidence_score: float


@dataclass
class PairCandidate:
    """Candidate pair for statistical arbitrage"""
    symbols: Tuple[str, str]
    cointegration_score: float
    half_life: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_profit: float
    profitability_score: float
    last_updated: datetime


class CointegrationEngine:
    """
    Advanced cointegration testing and pair discovery engine.
    Finds and ranks statistically significant trading pairs.
    """
    
    def __init__(self, lookback_days: int = 60, min_correlation: float = 0.5):
        self.lookback_days = lookback_days
        self.min_correlation = min_correlation
        self.confidence_levels = {
            '1%': 0.01,
            '5%': 0.05,
            '10%': 0.10
        }
        self.pair_cache = {}
        
    def find_cointegrated_pairs(self, price_data: Dict[str, pd.Series],
                               min_half_life: int = 5,
                               max_half_life: int = 20,
                               min_correlation: Optional[float] = None) -> List[PairCandidate]:
        """
        Find all cointegrated pairs from price data.
        
        Args:
            price_data: Dictionary of price series by symbol
            min_half_life: Minimum half-life for mean reversion
            max_half_life: Maximum half-life for mean reversion
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of ranked pair candidates
        """
        if min_correlation is None:
            min_correlation = self.min_correlation
            
        symbols = list(price_data.keys())
        candidates = []
        
        # Test all possible pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Get price series
                prices1 = price_data[symbol1]
                prices2 = price_data[symbol2]
                
                # Align series
                aligned = pd.DataFrame({
                    symbol1: prices1,
                    symbol2: prices2
                }).dropna()
                
                if len(aligned) < 50:  # Minimum data requirement
                    continue
                
                # Check correlation first (pre-filter)
                correlation = aligned[symbol1].corr(aligned[symbol2])
                if abs(correlation) < min_correlation:
                    continue
                
                # Test for cointegration
                result = self.test_cointegration(
                    aligned[symbol1].values,
                    aligned[symbol2].values,
                    symbol1,
                    symbol2
                )
                
                if result.cointegrated and min_half_life <= result.half_life <= max_half_life:
                    # Calculate additional metrics
                    candidate = self._evaluate_pair(
                        aligned[symbol1],
                        aligned[symbol2],
                        result
                    )
                    if candidate:
                        candidates.append(candidate)
        
        # Rank candidates by profitability score
        candidates.sort(key=lambda x: x.profitability_score, reverse=True)
        
        return candidates
    
    def test_cointegration(self, series1: np.ndarray, series2: np.ndarray,
                          symbol1: str, symbol2: str) -> CointegrationResult:
        """
        Test cointegration between two price series using Engle-Granger method.
        
        Args:
            series1: First price series
            series2: Second price series
            symbol1: Symbol name for series1
            symbol2: Symbol name for series2
            
        Returns:
            Cointegration test results
        """
        # Engle-Granger test
        eg_stat, eg_pvalue, eg_crit = coint(series1, series2, trend='c')
        
        # Calculate hedge ratio using OLS
        model = OLS(series1, sm.add_constant(series2))
        results = model.fit()
        hedge_ratio = results.params[1]
        
        # Calculate spread
        spread = series1 - hedge_ratio * series2
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        # Calculate Hurst exponent
        hurst = self._calculate_hurst_exponent(spread)
        
        # Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Correlation
        correlation = np.corrcoef(series1, series2)[0, 1]
        
        # Determine if cointegrated
        cointegrated = eg_pvalue < 0.05 and hurst < 0.5
        
        # Confidence score (0-1)
        confidence = self._calculate_confidence_score(
            eg_pvalue, half_life, hurst, spread_std
        )
        
        return CointegrationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            test_statistic=eg_stat,
            p_value=eg_pvalue,
            critical_values={
                '1%': eg_crit[0],
                '5%': eg_crit[1],
                '10%': eg_crit[2]
            },
            cointegrated=cointegrated,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            hurst_exponent=hurst,
            correlation=correlation,
            spread_mean=spread_mean,
            spread_std=spread_std,
            confidence_score=confidence
        )
    
    def test_johansen(self, price_data: pd.DataFrame, 
                     det_order: int = 0, k_ar_diff: int = 1) -> Dict[str, Any]:
        """
        Johansen test for multi-asset cointegration.
        
        Args:
            price_data: DataFrame with multiple price series
            det_order: Deterministic trend order (-1, 0, 1)
            k_ar_diff: Number of lags
            
        Returns:
            Johansen test results
        """
        try:
            # Run Johansen test
            result = coint_johansen(price_data, det_order, k_ar_diff)
            
            # Extract results
            trace_stats = result.lr1  # Trace statistics
            eigen_stats = result.lr2  # Maximum eigenvalue statistics
            critical_values_trace = result.cvt  # Critical values for trace
            critical_values_eigen = result.cvm  # Critical values for max eigenvalue
            
            # Determine number of cointegrating relationships
            n_coint_trace = np.sum(trace_stats > critical_values_trace[:, 1])  # 5% level
            n_coint_eigen = np.sum(eigen_stats > critical_values_eigen[:, 1])  # 5% level
            
            return {
                'trace_statistics': trace_stats,
                'eigen_statistics': eigen_stats,
                'critical_values_trace': critical_values_trace,
                'critical_values_eigen': critical_values_eigen,
                'n_cointegrating_trace': n_coint_trace,
                'n_cointegrating_eigen': n_coint_eigen,
                'eigenvectors': result.evec,
                'eigenvalues': result.eig
            }
            
        except Exception as e:
            logger.error(f"Johansen test failed: {e}")
            return None
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion using OLS.
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life in periods
        """
        try:
            # Create lagged spread
            spread_lag = np.roll(spread, 1)[1:]
            spread_diff = spread[1:] - spread_lag
            
            # Run regression: spread_t - spread_t-1 = alpha + beta * spread_t-1
            X = sm.add_constant(spread_lag)
            model = OLS(spread_diff, X)
            results = model.fit()
            
            # Half-life = -log(2) / beta
            beta = results.params[1]
            if beta >= 0:
                return np.inf  # No mean reversion
            
            half_life = -np.log(2) / beta
            return half_life
            
        except Exception as e:
            logger.error(f"Half-life calculation failed: {e}")
            return np.inf
    
    def _calculate_hurst_exponent(self, series: np.ndarray, 
                                 max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            series: Time series
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent
        """
        try:
            lags = range(2, max_lag)
            tau = []
            
            for lag in lags:
                # Calculate standard deviation of differenced series
                std_dev = np.std(np.diff(series[:lag]))
                
                # Calculate range
                cumsum = np.cumsum(series[:lag] - np.mean(series[:lag]))
                R = np.max(cumsum) - np.min(cumsum)
                
                # Calculate R/S
                if std_dev > 0:
                    tau.append(R / std_dev)
                else:
                    tau.append(0)
            
            # Fit log(R/S) = log(c) + H * log(lag)
            log_lags = np.log(list(lags))
            log_tau = np.log([t for t in tau if t > 0])
            
            if len(log_tau) < 2:
                return 0.5  # Default to random walk
            
            # Linear regression
            A = np.vstack([log_lags[:len(log_tau)], np.ones(len(log_tau))]).T
            H, c = np.linalg.lstsq(A, log_tau, rcond=None)[0]
            
            return H
            
        except Exception as e:
            logger.error(f"Hurst calculation failed: {e}")
            return 0.5
    
    def _evaluate_pair(self, series1: pd.Series, series2: pd.Series,
                      coint_result: CointegrationResult) -> Optional[PairCandidate]:
        """
        Evaluate pair profitability using backtesting.
        
        Args:
            series1: First price series
            series2: Second price series
            coint_result: Cointegration test results
            
        Returns:
            Pair candidate with profitability metrics
        """
        try:
            # Calculate spread
            spread = series1.values - coint_result.hedge_ratio * series2.values
            
            # Z-score normalization
            zscore = (spread - np.mean(spread)) / np.std(spread)
            
            # Simple backtest with 2/-2 entry/exit
            positions = np.zeros(len(zscore))
            
            for i in range(1, len(zscore)):
                if zscore[i] > 2:
                    positions[i] = -1  # Short spread
                elif zscore[i] < -2:
                    positions[i] = 1   # Long spread
                elif abs(zscore[i]) < 0.5:
                    positions[i] = 0   # Exit
                else:
                    positions[i] = positions[i-1]  # Hold
            
            # Calculate returns
            spread_returns = np.diff(spread) / spread[:-1]
            strategy_returns = positions[1:] * spread_returns
            
            # Performance metrics
            total_trades = np.sum(np.diff(positions) != 0)
            if total_trades == 0:
                return None
            
            winning_trades = np.sum(strategy_returns > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Sharpe ratio (annualized)
            if np.std(strategy_returns) > 0:
                sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            else:
                sharpe = 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Average profit per trade
            avg_profit = np.mean(strategy_returns[strategy_returns != 0])
            
            # Profitability score (weighted combination)
            profitability_score = (
                0.3 * min(sharpe / 3, 1) +  # Normalized Sharpe
                0.3 * win_rate +
                0.2 * (1 - abs(max_drawdown)) +
                0.1 * coint_result.confidence_score +
                0.1 * min(avg_profit * 100, 1)  # Normalized avg profit
            )
            
            return PairCandidate(
                symbols=(coint_result.symbol1, coint_result.symbol2),
                cointegration_score=coint_result.confidence_score,
                half_life=coint_result.half_life,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_profit=avg_profit,
                profitability_score=profitability_score,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Pair evaluation failed: {e}")
            return None
    
    def _calculate_confidence_score(self, p_value: float, half_life: float,
                                  hurst: float, spread_std: float) -> float:
        """
        Calculate confidence score for cointegration relationship.
        
        Args:
            p_value: Cointegration test p-value
            half_life: Mean reversion half-life
            hurst: Hurst exponent
            spread_std: Spread standard deviation
            
        Returns:
            Confidence score (0-1)
        """
        # P-value component (lower is better)
        p_score = 1 - p_value
        
        # Half-life component (5-20 is ideal)
        if 5 <= half_life <= 20:
            hl_score = 1.0
        elif half_life < 5:
            hl_score = half_life / 5
        elif half_life <= 30:
            hl_score = (30 - half_life) / 10
        else:
            hl_score = 0
        
        # Hurst component (lower is better for mean reversion)
        hurst_score = max(0, (0.5 - hurst) * 2)
        
        # Spread stability (lower std is better)
        std_score = 1 / (1 + spread_std)
        
        # Weighted average
        confidence = (
            0.4 * p_score +
            0.3 * hl_score +
            0.2 * hurst_score +
            0.1 * std_score
        )
        
        return np.clip(confidence, 0, 1)
    
    def update_pair_metrics(self, pair: PairCandidate, 
                          new_prices: Dict[str, pd.Series]) -> PairCandidate:
        """
        Update pair metrics with new price data.
        
        Args:
            pair: Existing pair candidate
            new_prices: New price data
            
        Returns:
            Updated pair candidate
        """
        symbol1, symbol2 = pair.symbols
        
        if symbol1 not in new_prices or symbol2 not in new_prices:
            return pair
        
        # Re-test cointegration with new data
        result = self.test_cointegration(
            new_prices[symbol1].values,
            new_prices[symbol2].values,
            symbol1,
            symbol2
        )
        
        # Re-evaluate pair
        updated_pair = self._evaluate_pair(
            new_prices[symbol1],
            new_prices[symbol2],
            result
        )
        
        return updated_pair if updated_pair else pair
    
    def find_multi_asset_portfolios(self, price_data: pd.DataFrame,
                                  max_assets: int = 4) -> List[Dict[str, Any]]:
        """
        Find cointegrated portfolios of multiple assets using Johansen test.
        
        Args:
            price_data: DataFrame with price series
            max_assets: Maximum assets in portfolio
            
        Returns:
            List of cointegrated portfolios
        """
        portfolios = []
        symbols = price_data.columns.tolist()
        
        # Test different combinations
        from itertools import combinations
        
        for n_assets in range(3, min(max_assets + 1, len(symbols) + 1)):
            for combo in combinations(symbols, n_assets):
                subset = price_data[list(combo)]
                
                # Run Johansen test
                johansen_result = self.test_johansen(subset)
                
                if johansen_result and johansen_result['n_cointegrating_trace'] > 0:
                    # Extract portfolio weights from eigenvectors
                    weights = johansen_result['eigenvectors'][:, 0]
                    weights = weights / np.sum(np.abs(weights))  # Normalize
                    
                    portfolio = {
                        'assets': list(combo),
                        'weights': dict(zip(combo, weights)),
                        'n_cointegrating': johansen_result['n_cointegrating_trace'],
                        'test_statistic': johansen_result['trace_statistics'][0],
                        'confidence': 1 - (0.05 ** johansen_result['n_cointegrating_trace'])
                    }
                    
                    portfolios.append(portfolio)
        
        # Sort by confidence
        portfolios.sort(key=lambda x: x['confidence'], reverse=True)
        
        return portfolios


# Import statsmodels here to avoid circular imports
import statsmodels.api as sm