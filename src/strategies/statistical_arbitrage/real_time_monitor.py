"""
Real-time monitoring system for statistical arbitrage pairs.
Tracks pair health, generates alerts, and manages pair lifecycle.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from .stat_arb_engine import StatArbEngine, TradingSignal, SignalType
from .cointegration_engine import PairCandidate

logger = logging.getLogger(__name__)


class PairStatus(Enum):
    """Status of a trading pair"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    BROKEN = "broken"
    RECOVERING = "recovering"
    INACTIVE = "inactive"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PairHealth:
    """Health metrics for a trading pair"""
    pair: tuple
    status: PairStatus
    current_p_value: float
    rolling_correlation: float
    spread_std_ratio: float  # Current std / historical std
    z_score: float
    last_trade_days_ago: int
    recent_win_rate: float
    health_score: float  # 0-1 composite score


@dataclass
class MonitorAlert:
    """Alert from monitoring system"""
    timestamp: datetime
    severity: AlertSeverity
    pair: Optional[tuple]
    message: str
    action_required: bool
    metadata: Dict[str, Any]


class RealTimeMonitor:
    """
    Real-time monitoring system for statistical arbitrage pairs.
    Monitors pair health, detects breakdowns, and manages lifecycle.
    """
    
    def __init__(self, 
                 stat_arb_engine: StatArbEngine,
                 health_check_interval: int = 300,  # 5 minutes
                 retest_interval: int = 3600,  # 1 hour
                 max_inactive_days: int = 7):
        """
        Initialize real-time monitor.
        
        Args:
            stat_arb_engine: Statistical arbitrage engine instance
            health_check_interval: Seconds between health checks
            retest_interval: Seconds between cointegration retests
            max_inactive_days: Days before marking pair inactive
        """
        self.engine = stat_arb_engine
        self.health_check_interval = health_check_interval
        self.retest_interval = retest_interval
        self.max_inactive_days = max_inactive_days
        
        # Monitoring state
        self.pair_health: Dict[tuple, PairHealth] = {}
        self.alerts: List[MonitorAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Tracking
        self.last_retest: Dict[tuple, datetime] = {}
        self.trade_history: Dict[tuple, List[datetime]] = {}
        self.degraded_pairs: Set[tuple] = set()
        
        # Tasks
        self.monitor_task = None
        self.is_running = False
        
    async def start_monitoring(self, price_callback: Callable,
                             initial_pairs: Optional[List[PairCandidate]] = None):
        """
        Start real-time monitoring.
        
        Args:
            price_callback: Async function to get current prices
            initial_pairs: Initial pairs to monitor
        """
        self.is_running = True
        self.price_callback = price_callback
        
        # Initialize pairs
        if initial_pairs:
            for pair in initial_pairs:
                self._initialize_pair_tracking(pair)
        
        # Start monitoring tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Real-time pair monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time pair monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Get current prices
                prices = await self.price_callback()
                
                # Check health of all pairs
                await self._check_all_pairs_health(prices)
                
                # Retest cointegration for pairs needing it
                await self._retest_cointegration(prices)
                
                # Check for inactive pairs
                self._check_inactive_pairs()
                
                # Generate summary alert if needed
                self._generate_summary_alert()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await self._create_alert(
                    AlertSeverity.WARNING,
                    None,
                    f"Monitoring error: {str(e)}",
                    False
                )
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _check_all_pairs_health(self, current_prices: Dict[str, float]):
        """Check health metrics for all active pairs."""
        for pair_candidate in self.engine.active_pairs:
            pair = pair_candidate.symbols
            
            if pair[0] not in current_prices or pair[1] not in current_prices:
                continue
            
            # Calculate current metrics
            health = await self._calculate_pair_health(
                pair_candidate, current_prices
            )
            
            # Store health
            self.pair_health[pair] = health
            
            # Check for status changes
            await self._check_status_change(pair, health)
    
    async def _calculate_pair_health(self, pair_candidate: PairCandidate,
                                   current_prices: Dict[str, float]) -> PairHealth:
        """Calculate comprehensive health metrics for a pair."""
        pair = pair_candidate.symbols
        symbol1, symbol2 = pair
        
        # Get recent price history (would be from database in production)
        price_history = await self._get_recent_price_history(pair, days=20)
        
        if not price_history:
            return self._create_default_health(pair)
        
        # Calculate current spread and z-score
        price1 = current_prices[symbol1]
        price2 = current_prices[symbol2]
        spread, z_score = self.engine._calculate_spread_metrics(
            price1, price2, pair_candidate
        )
        
        # Calculate rolling correlation
        if len(price_history[symbol1]) > 10:
            rolling_corr = price_history[symbol1].tail(20).corr(
                price_history[symbol2].tail(20)
            )
        else:
            rolling_corr = pair_candidate.correlation
        
        # Calculate spread stability
        historical_spreads = (price_history[symbol1] - 
                            pair_candidate.hedge_ratio * price_history[symbol2])
        current_std = historical_spreads.tail(20).std()
        historical_std = historical_spreads.std()
        
        spread_std_ratio = current_std / historical_std if historical_std > 0 else 1
        
        # Get recent trading activity
        last_trade_days = self._get_days_since_last_trade(pair)
        recent_win_rate = self._calculate_recent_win_rate(pair)
        
        # Re-test cointegration with recent data
        if len(price_history[symbol1]) > 50:
            result = self.engine.cointegration_engine.test_cointegration(
                price_history[symbol1].values,
                price_history[symbol2].values,
                symbol1, symbol2
            )
            current_p_value = result.p_value
        else:
            current_p_value = pair_candidate.cointegration_score
        
        # Calculate composite health score
        health_score = self._calculate_health_score(
            current_p_value, rolling_corr, spread_std_ratio,
            abs(z_score), last_trade_days, recent_win_rate
        )
        
        # Determine status
        status = self._determine_pair_status(
            health_score, current_p_value, rolling_corr, spread_std_ratio
        )
        
        return PairHealth(
            pair=pair,
            status=status,
            current_p_value=current_p_value,
            rolling_correlation=rolling_corr,
            spread_std_ratio=spread_std_ratio,
            z_score=z_score,
            last_trade_days_ago=last_trade_days,
            recent_win_rate=recent_win_rate,
            health_score=health_score
        )
    
    def _calculate_health_score(self, p_value: float, correlation: float,
                              std_ratio: float, abs_z_score: float,
                              days_inactive: int, win_rate: float) -> float:
        """Calculate composite health score (0-1)."""
        # P-value score (lower is better)
        p_score = max(0, 1 - p_value * 10)  # 0.1 p-value = 0 score
        
        # Correlation score (higher is better)
        corr_score = abs(correlation)
        
        # Stability score (ratio close to 1 is better)
        stability_score = 1 / (1 + abs(std_ratio - 1))
        
        # Activity score (recent activity is better)
        activity_score = 1 / (1 + days_inactive / 7)
        
        # Performance score
        perf_score = win_rate
        
        # Z-score penalty (extreme values indicate stressed pair)
        z_penalty = 1 / (1 + max(0, abs_z_score - 3))
        
        # Weighted average
        health_score = (
            0.25 * p_score +
            0.20 * corr_score +
            0.20 * stability_score +
            0.15 * activity_score +
            0.15 * perf_score +
            0.05 * z_penalty
        )
        
        return np.clip(health_score, 0, 1)
    
    def _determine_pair_status(self, health_score: float, p_value: float,
                             correlation: float, std_ratio: float) -> PairStatus:
        """Determine pair status based on metrics."""
        # Critical failures
        if p_value > 0.1 or abs(correlation) < 0.3:
            return PairStatus.BROKEN
        
        # Check if recovering
        if self.pair_health.get(tuple) and self.pair_health[tuple].status == PairStatus.BROKEN:
            if health_score > 0.6:
                return PairStatus.RECOVERING
        
        # Normal classification
        if health_score >= 0.7:
            return PairStatus.ACTIVE
        elif health_score >= 0.5:
            return PairStatus.DEGRADED
        else:
            return PairStatus.BROKEN
    
    async def _check_status_change(self, pair: tuple, new_health: PairHealth):
        """Check for status changes and generate alerts."""
        old_health = self.pair_health.get(pair)
        
        if not old_health:
            return
        
        if old_health.status != new_health.status:
            # Status changed
            if new_health.status == PairStatus.BROKEN:
                await self._create_alert(
                    AlertSeverity.CRITICAL,
                    pair,
                    f"Pair {pair} cointegration broken (p-value: {new_health.current_p_value:.3f})",
                    True,
                    {'old_status': old_health.status.value,
                     'new_status': new_health.status.value,
                     'health_score': new_health.health_score}
                )
                self.degraded_pairs.add(pair)
                
            elif new_health.status == PairStatus.DEGRADED:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    pair,
                    f"Pair {pair} showing signs of degradation",
                    False,
                    {'health_score': new_health.health_score,
                     'correlation': new_health.rolling_correlation}
                )
                self.degraded_pairs.add(pair)
                
            elif new_health.status == PairStatus.RECOVERING:
                await self._create_alert(
                    AlertSeverity.INFO,
                    pair,
                    f"Pair {pair} showing signs of recovery",
                    False,
                    {'health_score': new_health.health_score}
                )
                
            elif new_health.status == PairStatus.ACTIVE and old_health.status != PairStatus.ACTIVE:
                await self._create_alert(
                    AlertSeverity.INFO,
                    pair,
                    f"Pair {pair} fully recovered",
                    False,
                    {'health_score': new_health.health_score}
                )
                self.degraded_pairs.discard(pair)
    
    async def _retest_cointegration(self, current_prices: Dict[str, float]):
        """Retest cointegration for pairs that need it."""
        now = datetime.now()
        
        for pair in self.degraded_pairs:
            last_test = self.last_retest.get(pair, datetime.min)
            
            if (now - last_test).seconds >= self.retest_interval:
                # Get extended price history
                price_history = await self._get_recent_price_history(pair, days=60)
                
                if price_history and len(price_history[pair[0]]) > 50:
                    # Retest cointegration
                    result = self.engine.cointegration_engine.test_cointegration(
                        price_history[pair[0]].values,
                        price_history[pair[1]].values,
                        pair[0], pair[1]
                    )
                    
                    self.last_retest[pair] = now
                    
                    # Check if recovered
                    if result.cointegrated and result.p_value < 0.05:
                        await self._create_alert(
                            AlertSeverity.INFO,
                            pair,
                            f"Pair {pair} passed cointegration retest",
                            False,
                            {'p_value': result.p_value,
                             'half_life': result.half_life}
                        )
    
    def _check_inactive_pairs(self):
        """Check for pairs with no recent trading activity."""
        for pair in self.engine.active_pairs:
            days_inactive = self._get_days_since_last_trade(pair.symbols)
            
            if days_inactive > self.max_inactive_days:
                health = self.pair_health.get(pair.symbols)
                if health and health.status == PairStatus.ACTIVE:
                    # Update status
                    health.status = PairStatus.INACTIVE
                    
                    asyncio.create_task(self._create_alert(
                        AlertSeverity.WARNING,
                        pair.symbols,
                        f"Pair {pair.symbols} inactive for {days_inactive} days",
                        False,
                        {'days_inactive': days_inactive}
                    ))
    
    def _generate_summary_alert(self):
        """Generate summary alert if multiple pairs are degraded."""
        n_degraded = len(self.degraded_pairs)
        n_total = len(self.engine.active_pairs)
        
        if n_degraded > n_total * 0.3:  # More than 30% degraded
            asyncio.create_task(self._create_alert(
                AlertSeverity.WARNING,
                None,
                f"{n_degraded}/{n_total} pairs degraded - market regime may have changed",
                True,
                {'degraded_pairs': list(self.degraded_pairs)}
            ))
    
    async def _create_alert(self, severity: AlertSeverity, pair: Optional[tuple],
                          message: str, action_required: bool,
                          metadata: Optional[Dict] = None):
        """Create and dispatch alert."""
        alert = MonitorAlert(
            timestamp=datetime.now(),
            severity=severity,
            pair=pair,
            message=message,
            action_required=action_required,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Keep last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Dispatch to callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        # Pair health summary
        health_summary = {
            'active': 0,
            'degraded': 0,
            'broken': 0,
            'recovering': 0,
            'inactive': 0
        }
        
        pair_details = []
        
        for pair, health in self.pair_health.items():
            health_summary[health.status.value] += 1
            
            pair_details.append({
                'pair': f"{pair[0]}-{pair[1]}",
                'status': health.status.value,
                'health_score': health.health_score,
                'p_value': health.current_p_value,
                'correlation': health.rolling_correlation,
                'z_score': health.z_score,
                'last_trade_days': health.last_trade_days_ago,
                'win_rate': health.recent_win_rate
            })
        
        # Recent alerts
        recent_alerts = []
        for alert in self.alerts[-20:]:  # Last 20 alerts
            recent_alerts.append({
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'pair': f"{alert.pair[0]}-{alert.pair[1]}" if alert.pair else None,
                'message': alert.message,
                'action_required': alert.action_required
            })
        
        return {
            'health_summary': health_summary,
            'pair_details': sorted(pair_details, 
                                 key=lambda x: x['health_score'], 
                                 reverse=True),
            'recent_alerts': recent_alerts,
            'total_pairs': len(self.engine.active_pairs),
            'last_update': datetime.now().isoformat()
        }
    
    # Helper methods
    
    def _initialize_pair_tracking(self, pair: PairCandidate):
        """Initialize tracking for a new pair."""
        self.last_retest[pair.symbols] = datetime.now()
        self.trade_history[pair.symbols] = []
        
    def _get_days_since_last_trade(self, pair: tuple) -> int:
        """Get days since last trade for a pair."""
        if pair not in self.trade_history or not self.trade_history[pair]:
            return 999  # Never traded
        
        last_trade = max(self.trade_history[pair])
        return (datetime.now() - last_trade).days
    
    def _calculate_recent_win_rate(self, pair: tuple, days: int = 30) -> float:
        """Calculate win rate for recent period."""
        # This would query actual trade history
        # For now, return a placeholder
        return 0.5
    
    async def _get_recent_price_history(self, pair: tuple, 
                                      days: int) -> Optional[Dict[str, pd.Series]]:
        """Get recent price history for a pair."""
        # This would fetch from database or price service
        # For now, return None
        return None
    
    def _create_default_health(self, pair: tuple) -> PairHealth:
        """Create default health metrics when data unavailable."""
        return PairHealth(
            pair=pair,
            status=PairStatus.INACTIVE,
            current_p_value=1.0,
            rolling_correlation=0,
            spread_std_ratio=1.0,
            z_score=0,
            last_trade_days_ago=999,
            recent_win_rate=0,
            health_score=0
        )
    
    def update_trade_history(self, pair: tuple, timestamp: datetime):
        """Update trade history for a pair."""
        if pair not in self.trade_history:
            self.trade_history[pair] = []
        
        self.trade_history[pair].append(timestamp)