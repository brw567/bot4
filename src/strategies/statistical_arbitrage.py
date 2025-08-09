"""
Statistical Arbitrage Strategy using cointegration and mean reversion.
Identifies and trades statistically related pairs for market-neutral profits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
import logging

# Statistical testing
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# Kalman filter for dynamic hedge ratios
from filterpy.kalman import KalmanFilter


logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """Cointegrated pair information"""
    symbol1: str
    symbol2: str
    cointegration_pvalue: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float
    last_updated: datetime
    exchange: str


@dataclass
class StatArbSignal:
    """Statistical arbitrage trading signal"""
    pair: TradingPair
    action: str  # 'LONG_PAIR1_SHORT_PAIR2', 'SHORT_PAIR1_LONG_PAIR2', 'CLOSE'
    current_zscore: float
    expected_profit: float
    confidence: float
    entry_prices: Dict[str, float]


class CointegrationTester:
    """Tests for cointegration between price series"""
    
    @staticmethod
    def test_stationarity(series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series: Price series to test
            significance: Significance level
            
        Returns:
            Tuple of (is_stationary, p_value)
        """
        try:
            result = adfuller(series.dropna())
            return result[1] < significance, result[1]
        except Exception as e:
            logger.error(f"Stationarity test failed: {e}")
            return False, 1.0
    
    @staticmethod
    def find_cointegration(series1: pd.Series, series2: pd.Series, 
                          significance: float = 0.05) -> Tuple[bool, float, float]:
        """
        Test for cointegration between two series using Engle-Granger method.
        
        Args:
            series1: First price series
            series2: Second price series
            significance: Significance level
            
        Returns:
            Tuple of (is_cointegrated, p_value, hedge_ratio)
        """
        try:
            # Run cointegration test
            score, pvalue, _ = coint(series1, series2)
            
            # Calculate hedge ratio using OLS
            model = sm.OLS(series1, sm.add_constant(series2))
            results = model.fit()
            hedge_ratio = results.params[1]
            
            return pvalue < significance, pvalue, hedge_ratio
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return False, 1.0, 0.0
    
    @staticmethod
    def johansen_test(price_matrix: pd.DataFrame, significance: float = 0.05) -> Dict:
        """
        Johansen test for cointegration of multiple series.
        
        Args:
            price_matrix: DataFrame with price series as columns
            significance: Significance level
            
        Returns:
            Dictionary with test results
        """
        try:
            result = coint_johansen(price_matrix.values, det_order=0, k_ar_diff=1)
            
            # Check trace statistic
            trace_stat = result.lr1
            trace_crit = result.cvt[:, 1]  # 5% critical values
            
            n_coint = sum(trace_stat > trace_crit)
            
            return {
                'n_cointegrating': n_coint,
                'trace_stats': trace_stat,
                'critical_values': trace_crit,
                'eigenvectors': result.evec
            }
        except Exception as e:
            logger.error(f"Johansen test failed: {e}")
            return {'n_cointegrating': 0}


class DynamicHedgeRatioCalculator:
    """Calculate dynamic hedge ratios using Kalman filter"""
    
    def __init__(self, initial_hedge: float = 1.0):
        self.kf = self._initialize_kalman_filter(initial_hedge)
        self.hedge_history = []
        
    def _initialize_kalman_filter(self, initial_hedge: float) -> KalmanFilter:
        """Initialize Kalman filter for hedge ratio estimation"""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        
        # State transition matrix
        kf.F = np.array([[1.]])
        
        # Measurement function
        kf.H = np.array([[1.]])
        
        # Measurement noise
        kf.R = 0.01
        
        # Process noise
        kf.Q = 0.0001
        
        # Initial state
        kf.x = np.array([[initial_hedge]])
        
        # Initial covariance
        kf.P = np.array([[1.]])
        
        return kf
    
    def update(self, price1: float, price2: float) -> float:
        """
        Update hedge ratio with new price observation.
        
        Args:
            price1: Price of first asset
            price2: Price of second asset
            
        Returns:
            Updated hedge ratio
        """
        # Calculate instantaneous hedge ratio
        if len(self.hedge_history) > 0 and price2 != 0:
            instant_hedge = price1 / price2
        else:
            instant_hedge = 1.0
            
        # Kalman filter update
        self.kf.predict()
        self.kf.update(instant_hedge)
        
        hedge_ratio = float(self.kf.x[0])
        self.hedge_history.append(hedge_ratio)
        
        # Keep only recent history
        if len(self.hedge_history) > 1000:
            self.hedge_history = self.hedge_history[-1000:]
            
        return hedge_ratio


class StatisticalArbitrage:
    """Main statistical arbitrage strategy implementation"""
    
    def __init__(self, 
                 lookback_period: int = 1000,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5,
                 stop_zscore: float = 3.5,
                 min_half_life: int = 10,
                 max_half_life: int = 100,
                 position_size_pct: float = 0.1):
        
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.position_size_pct = position_size_pct
        
        self.coint_tester = CointegrationTester()
        self.pairs = {}  # symbol -> TradingPair
        self.positions = {}  # symbol -> position info
        self.hedge_calculators = {}  # symbol -> DynamicHedgeRatioCalculator
        
    async def find_cointegrated_pairs(self, 
                                    price_data: Dict[str, pd.Series],
                                    exchange: str) -> List[TradingPair]:
        """
        Find cointegrated pairs from price data.
        
        Args:
            price_data: Dictionary of symbol -> price series
            exchange: Exchange name
            
        Returns:
            List of cointegrated trading pairs
        """
        pairs = []
        symbols = list(price_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                series1, series2 = price_data[symbol1], price_data[symbol2]
                
                # Ensure same length
                min_len = min(len(series1), len(series2))
                if min_len < self.lookback_period:
                    continue
                    
                series1 = series1[-min_len:]
                series2 = series2[-min_len:]
                
                # Test for cointegration
                is_coint, pvalue, hedge_ratio = self.coint_tester.find_cointegration(
                    series1, series2
                )
                
                if is_coint:
                    # Calculate spread statistics
                    spread = series1 - hedge_ratio * series2
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    
                    # Calculate half-life of mean reversion
                    half_life = self._calculate_half_life(spread)
                    
                    if self.min_half_life <= half_life <= self.max_half_life:
                        pair = TradingPair(
                            symbol1=symbol1,
                            symbol2=symbol2,
                            cointegration_pvalue=pvalue,
                            hedge_ratio=hedge_ratio,
                            spread_mean=spread_mean,
                            spread_std=spread_std,
                            half_life=half_life,
                            last_updated=datetime.now(),
                            exchange=exchange
                        )
                        pairs.append(pair)
                        
                        # Initialize dynamic hedge calculator
                        key = f"{symbol1}-{symbol2}"
                        self.hedge_calculators[key] = DynamicHedgeRatioCalculator(hedge_ratio)
                        
        return pairs
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process"""
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            
            # Remove NaN values
            mask = ~(spread_lag.isna() | spread_diff.isna())
            spread_lag = spread_lag[mask]
            spread_diff = spread_diff[mask]
            
            if len(spread_lag) < 50:
                return 50.0  # Default if insufficient data
            
            # OLS regression
            model = sm.OLS(spread_diff, sm.add_constant(spread_lag))
            results = model.fit()
            
            # Half-life calculation
            if results.params[1] >= 0:
                return 50.0  # No mean reversion
                
            half_life = -np.log(2) / results.params[1]
            
            return float(np.clip(half_life, 1, 200))
            
        except Exception as e:
            logger.error(f"Half-life calculation failed: {e}")
            return 50.0
    
    def generate_signals(self, 
                        current_prices: Dict[str, float],
                        pairs: List[TradingPair]) -> List[StatArbSignal]:
        """
        Generate trading signals for cointegrated pairs.
        
        Args:
            current_prices: Current prices for all symbols
            pairs: List of cointegrated pairs to check
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for pair in pairs:
            if pair.symbol1 not in current_prices or pair.symbol2 not in current_prices:
                continue
                
            price1 = current_prices[pair.symbol1]
            price2 = current_prices[pair.symbol2]
            
            # Update dynamic hedge ratio
            key = f"{pair.symbol1}-{pair.symbol2}"
            if key in self.hedge_calculators:
                hedge_ratio = self.hedge_calculators[key].update(price1, price2)
            else:
                hedge_ratio = pair.hedge_ratio
            
            # Calculate current spread and z-score
            spread = price1 - hedge_ratio * price2
            zscore = (spread - pair.spread_mean) / pair.spread_std if pair.spread_std > 0 else 0
            
            # Check if we have an existing position
            has_position = key in self.positions
            
            # Generate signals
            signal = None
            
            if not has_position:
                # Entry signals
                if zscore > self.entry_zscore:
                    # Spread is too high, short pair1 and long pair2
                    signal = StatArbSignal(
                        pair=pair,
                        action='SHORT_PAIR1_LONG_PAIR2',
                        current_zscore=zscore,
                        expected_profit=self._estimate_profit(zscore, pair.half_life),
                        confidence=self._calculate_confidence(pair),
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
                elif zscore < -self.entry_zscore:
                    # Spread is too low, long pair1 and short pair2
                    signal = StatArbSignal(
                        pair=pair,
                        action='LONG_PAIR1_SHORT_PAIR2',
                        current_zscore=zscore,
                        expected_profit=self._estimate_profit(-zscore, pair.half_life),
                        confidence=self._calculate_confidence(pair),
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
            else:
                # Exit signals for existing positions
                position = self.positions[key]
                
                # Check for exit conditions
                if abs(zscore) < self.exit_zscore:
                    # Mean reversion achieved, close position
                    signal = StatArbSignal(
                        pair=pair,
                        action='CLOSE',
                        current_zscore=zscore,
                        expected_profit=0,
                        confidence=1.0,
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
                elif abs(zscore) > self.stop_zscore:
                    # Stop loss triggered
                    signal = StatArbSignal(
                        pair=pair,
                        action='CLOSE',
                        current_zscore=zscore,
                        expected_profit=-0.02,  # Expected 2% loss
                        confidence=1.0,
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
                elif position['direction'] == 'SHORT_PAIR1' and zscore < -self.exit_zscore:
                    # Wrong direction, close
                    signal = StatArbSignal(
                        pair=pair,
                        action='CLOSE',
                        current_zscore=zscore,
                        expected_profit=0,
                        confidence=0.5,
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
                elif position['direction'] == 'LONG_PAIR1' and zscore > self.exit_zscore:
                    # Wrong direction, close
                    signal = StatArbSignal(
                        pair=pair,
                        action='CLOSE',
                        current_zscore=zscore,
                        expected_profit=0,
                        confidence=0.5,
                        entry_prices={pair.symbol1: price1, pair.symbol2: price2}
                    )
            
            if signal:
                signals.append(signal)
                
        return signals
    
    def _estimate_profit(self, zscore: float, half_life: float) -> float:
        """Estimate expected profit based on z-score and half-life"""
        # Expected z-score reduction over half-life period
        expected_reduction = abs(zscore) * 0.5
        
        # Convert to percentage return
        # Assuming 1 std dev = 1% price movement
        expected_return = expected_reduction * 0.01
        
        # Adjust for time (faster mean reversion = higher annualized return)
        time_factor = 365 / (half_life * 2)  # Annualize based on half-life
        
        return expected_return * min(time_factor, 10)  # Cap at 10x
    
    def _calculate_confidence(self, pair: TradingPair) -> float:
        """Calculate confidence in the trading pair"""
        # Factors affecting confidence:
        # 1. Cointegration p-value (lower is better)
        # 2. Half-life within optimal range
        # 3. Time since last update
        
        # P-value score (0-1, higher is better)
        pvalue_score = 1 - pair.cointegration_pvalue
        
        # Half-life score (optimal around 20-50 days)
        optimal_half_life = 35
        half_life_score = 1 - min(abs(pair.half_life - optimal_half_life) / optimal_half_life, 1)
        
        # Freshness score (depreciate over time)
        hours_old = (datetime.now() - pair.last_updated).total_seconds() / 3600
        freshness_score = max(0, 1 - hours_old / (24 * 7))  # Depreciate over a week
        
        # Combined confidence
        confidence = (pvalue_score * 0.4 + half_life_score * 0.4 + freshness_score * 0.2)
        
        return float(np.clip(confidence, 0, 1))
    
    def update_position(self, signal: StatArbSignal, executed_prices: Dict[str, float]):
        """Update position tracking after trade execution"""
        key = f"{signal.pair.symbol1}-{signal.pair.symbol2}"
        
        if signal.action == 'CLOSE':
            # Remove position
            if key in self.positions:
                del self.positions[key]
        else:
            # Add or update position
            self.positions[key] = {
                'pair': signal.pair,
                'direction': signal.action.split('_')[0] + '_PAIR1',
                'entry_prices': executed_prices,
                'entry_zscore': signal.current_zscore,
                'entry_time': datetime.now()
            }
    
    async def backtest(self, 
                      price_data: Dict[str, pd.DataFrame],
                      initial_capital: float = 100000) -> Dict:
        """
        Backtest the statistical arbitrage strategy.
        
        Args:
            price_data: Historical price data for all symbols
            initial_capital: Starting capital
            
        Returns:
            Backtest results including performance metrics
        """
        # Implementation for backtesting
        # This would simulate the strategy over historical data
        pass