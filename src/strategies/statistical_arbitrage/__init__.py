"""
Statistical Arbitrage Strategy Module.
Provides cointegration-based pair trading with optimal position sizing.
"""

from .cointegration_engine import (
    CointegrationEngine,
    CointegrationResult,
    PairCandidate
)

from .stat_arb_engine import (
    StatArbEngine,
    TradingSignal,
    SignalType,
    StatArbPosition
)

from .kelly_criterion import KellyCriterion

from .stat_arb_backtester import (
    StatArbBacktester,
    BacktestResult,
    WalkForwardResult
)

from .real_time_monitor import (
import logging

    RealTimeMonitor,
    PairHealth,
    PairStatus,
    MonitorAlert,
    AlertSeverity
)

logger = logging.getLogger(__name__)

__all__ = [
    # Cointegration
    'CointegrationEngine',
    'CointegrationResult', 
    'PairCandidate',
    
    # Statistical Arbitrage
    'StatArbEngine',
    'TradingSignal',
    'SignalType',
    'StatArbPosition',
    
    # Position Sizing
    'KellyCriterion',
    
    # Backtesting
    'StatArbBacktester',
    'BacktestResult',
    'WalkForwardResult',
    
    # Monitoring
    'RealTimeMonitor',
    'PairHealth',
    'PairStatus',
    'MonitorAlert',
    'AlertSeverity'
]