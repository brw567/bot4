"""
Integrated Strategy Selector with Hybrid Regime Detection

This module extends the dynamic strategy selector to integrate with:
- Hybrid regime detector for market context
- Per-pair strategy switching and tracking
- Comprehensive logging system
- Strategy change notifications
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json

from strategies.strategy_selector import DynamicStrategySelector, get_strategy_selector
from core.hybrid_regime_detector import HybridRegimeDetector, get_hybrid_regime_detector

logger = logging.getLogger(__name__)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class IntegratedStrategySelector(DynamicStrategySelector):
    """
    Enhanced strategy selector that integrates with hybrid regime detection
    and provides per-pair strategy management.
    """
    
    def __init__(self):
        super().__init__()
        
        # Hybrid regime detector
        self.regime_detector = get_hybrid_regime_detector()
        
        # Per-pair strategy tracking
        self.active_strategies = {}  # pair -> list of strategies
        self.strategy_history = defaultdict(list)  # pair -> list of historical selections
        self.last_regime = {}  # pair -> last known regime
        self.strategy_performance = defaultdict(lambda: defaultdict(float))  # pair -> strategy -> metrics
        
        # Logging configuration
        self.enable_detailed_logging = True
        self.log_file = "logs/strategy_changes.log"
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Setup dedicated file logging for strategy changes."""
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Create a separate logger for strategy changes
            self.strategy_logger = logging.getLogger('strategy_changes')
            self.strategy_logger.addHandler(file_handler)
            self.strategy_logger.setLevel(logging.INFO)
            
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    async def select_strategies_with_regime(
        self, 
        pair: str,
        pair_type: str = 'spot'
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select strategies based on both volatility and hybrid regime.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            pair_type: 'spot' or 'futures'
            
        Returns:
            Tuple of (selected_strategies, context_data)
        """
        try:
            # Get current hybrid regime
            regime_data = await self.regime_detector.get_current_regime(pair)
            
            # Get market volatility
            volatility = await self._calculate_current_volatility(pair)
            
            # Prepare metrics including regime information
            metrics = {
                'volatility': volatility,
                'hybrid_score': regime_data.get('hybrid_score', 0),
                'local_score': regime_data.get('local_score', 0),
                'sentiment_score': regime_data.get('sentiment_score', 0.5),
                'regime': regime_data.get('regime', 'neutral'),
                'leverage': regime_data.get('leverage', 1.5)
            }
            
            # Add additional market metrics
            market_metrics = await self._get_market_metrics(pair)
            metrics.update(market_metrics)
            
            # Adjust risk tolerance based on regime
            risk_tolerance = self._determine_risk_tolerance(regime_data)
            
            # Prepare configuration for strategy selection
            pair_config = {
                'pair': pair,
                'type': pair_type,
                'volatility': volatility,
                'metrics': metrics,
                'risk_tolerance': risk_tolerance
            }
            
            # Select strategies using parent class method
            selected_strategies = await self.select_strategies(pair_config)
            
            # Check if strategies changed
            previous_strategies = self.active_strategies.get(pair, [])
            if selected_strategies != previous_strategies:
                await self._handle_strategy_change(
                    pair, previous_strategies, selected_strategies, 
                    regime_data, metrics
                )
            
            # Update active strategies
            self.active_strategies[pair] = selected_strategies
            self.last_regime[pair] = regime_data.get('regime', 'neutral')
            
            # Prepare context data
            context = {
                'regime': regime_data,
                'volatility': volatility,
                'risk_tolerance': risk_tolerance,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            return selected_strategies, context
            
        except Exception as e:
            logger.error(f"Strategy selection failed for {pair}: {e}")
            # Return safe defaults
            return ['EMA', 'RSI'], {'error': str(e)}
    
    async def _calculate_current_volatility(self, pair: str) -> float:
        """Calculate current volatility for a pair."""
        try:
            from utils.binance_utils import get_binance_client
            client = get_binance_client()
            
            # Get recent OHLCV data
            ohlcv = client.fetch_ohlcv(pair, '5m', limit=288)  # 24 hours
            
            if len(ohlcv) < 50:
                return 0.02  # Default volatility
            
            # Calculate returns
            closes = [candle[4] for candle in ohlcv]
            returns = []
            
            for i in range(1, len(closes)):
                if closes[i-1] != 0:
                    ret = (closes[i] - closes[i-1]) / closes[i-1]
                    returns.append(ret)
            
            # Calculate volatility
            import numpy as np
            volatility = np.std(returns) if returns else 0.02
            
            return max(0.001, min(1.0, volatility))
            
        except Exception as e:
            logger.warning(f"Volatility calculation failed for {pair}: {e}")
            return 0.02
    
    async def _get_market_metrics(self, pair: str) -> Dict[str, float]:
        """Get additional market metrics for strategy selection."""
        try:
            from utils.binance_utils import get_binance_client
            client = get_binance_client()
            
            # Get ticker data
            ticker = client.fetch_ticker(pair)
            
            # Calculate metrics
            metrics = {
                'volume_ratio': 1.0,  # Placeholder
                'momentum': (ticker.get('last', 0) - ticker.get('open', 0)) / ticker.get('open', 1) if ticker.get('open') else 0,
                'spread': abs(ticker.get('bid', 0) - ticker.get('ask', 0)) / ticker.get('last', 1) if ticker.get('last') else 0,
            }
            
            # Get order book imbalance
            try:
                orderbook = client.fetch_order_book(pair, limit=20)
                total_bid = sum([b[1] for b in orderbook['bids'][:10]])
                total_ask = sum([a[1] for a in orderbook['asks'][:10]])
                
                if total_bid + total_ask > 0:
                    metrics['book_imbalance'] = (total_bid - total_ask) / (total_bid + total_ask)
                else:
                    metrics['book_imbalance'] = 0
            except:
                metrics['book_imbalance'] = 0
            
            # Placeholder values for other metrics
            metrics['oi_change'] = 0.1
            metrics['funding_rate'] = 0.005
            metrics['drawdown'] = 0.01
            metrics['mempool_density'] = 0.5
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Market metrics failed for {pair}: {e}")
            return {
                'volume_ratio': 1.0,
                'momentum': 0,
                'spread': 0.001,
                'book_imbalance': 0,
                'oi_change': 0.1,
                'funding_rate': 0.005,
                'drawdown': 0.01,
                'mempool_density': 0.5
            }
    
    def _determine_risk_tolerance(self, regime_data: Dict) -> str:
        """Determine risk tolerance based on regime data."""
        regime = regime_data.get('regime', 'neutral')
        hybrid_score = regime_data.get('hybrid_score', 0)
        
        # Bull regime: can take more risk
        if regime == 'bull' and hybrid_score > 0.7:
            return 'high'
        elif regime == 'bull':
            return 'medium'
        
        # Bear regime: conservative
        elif regime == 'bear':
            return 'low'
        
        # Neutral: balanced approach
        else:
            if abs(hybrid_score) < 0.3:
                return 'medium'
            else:
                return 'low'
    
    async def _handle_strategy_change(
        self,
        pair: str,
        old_strategies: List[str],
        new_strategies: List[str],
        regime_data: Dict,
        metrics: Dict
    ) -> None:
        """Handle strategy change with comprehensive logging."""
        change_data = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'old_strategies': old_strategies,
            'new_strategies': new_strategies,
            'regime': regime_data.get('regime', 'unknown'),
            'hybrid_score': regime_data.get('hybrid_score', 0),
            'volatility': metrics.get('volatility', 0),
            'risk_tolerance': self._determine_risk_tolerance(regime_data),
            'triggers': self._identify_active_triggers(metrics)
        }
        
        # Log to file
        self.strategy_logger.info(f"STRATEGY_CHANGE: {json.dumps(change_data)}")
        
        # Log to console with formatting
        logger.info(
            f"📊 Strategy Change for {pair}:\n"
            f"   Old: {', '.join(old_strategies)}\n"
            f"   New: {', '.join(new_strategies)}\n"
            f"   Regime: {change_data['regime']} (score: {change_data['hybrid_score']:.3f})\n"
            f"   Volatility: {change_data['volatility']:.3f}\n"
            f"   Active Triggers: {', '.join(change_data['triggers']) if change_data['triggers'] else 'None'}"
        )
        
        # Record in history
        self.strategy_history[pair].append(change_data)
        
        # Keep only last 1000 records per pair
        if len(self.strategy_history[pair]) > 1000:
            self.strategy_history[pair] = self.strategy_history[pair][-1000:]
    
    def _identify_active_triggers(self, metrics: Dict) -> List[str]:
        """Identify which triggers are currently active."""
        active_triggers = []
        
        if metrics.get('oi_change', 0) > 0.15:
            active_triggers.append('oi_surge')
        if metrics.get('funding_rate', 0) > 0.01:
            active_triggers.append('high_funding')
        if metrics.get('drawdown', 0) > 0.02:
            active_triggers.append('drawdown_protection')
        if metrics.get('mempool_density', 0) > 0.8:
            active_triggers.append('mev_opportunity')
        if metrics.get('volatility', 0) > 0.05:
            active_triggers.append('high_volatility')
        if abs(metrics.get('book_imbalance', 0)) > 0.3:
            active_triggers.append('orderbook_imbalance')
        
        return active_triggers
    
    async def switch_strategies_for_pair(
        self,
        pair: str,
        forced_strategies: Optional[List[str]] = None,
        reason: str = "manual_override"
    ) -> bool:
        """
        Switch strategies for a specific pair.
        
        Args:
            pair: Trading pair
            forced_strategies: Optional list of strategies to force
            reason: Reason for the switch
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            old_strategies = self.active_strategies.get(pair, [])
            
            if forced_strategies:
                # Manual override
                new_strategies = forced_strategies
                regime_data = {'regime': 'manual', 'hybrid_score': 0}
                metrics = {'reason': reason}
            else:
                # Automatic selection
                new_strategies, context = await self.select_strategies_with_regime(pair)
                regime_data = context.get('regime', {})
                metrics = context.get('metrics', {})
            
            # Update active strategies
            self.active_strategies[pair] = new_strategies
            
            # Log the change
            if old_strategies != new_strategies:
                await self._handle_strategy_change(
                    pair, old_strategies, new_strategies,
                    regime_data, metrics
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Strategy switch failed for {pair}: {e}")
            return False
    
    def get_active_strategies(self, pair: Optional[str] = None) -> Dict[str, List[str]]:
        """Get currently active strategies."""
        if pair:
            return {pair: self.active_strategies.get(pair, [])}
        return dict(self.active_strategies)
    
    def get_strategy_history(self, pair: str, hours: int = 24) -> List[Dict]:
        """Get strategy change history for a pair."""
        history = self.strategy_history.get(pair, [])
        
        if hours > 0:
            cutoff = datetime.now() - timedelta(hours=hours)
            history = [
                record for record in history
                if datetime.fromisoformat(record['timestamp']) > cutoff
            ]
        
        return history
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        stats = {
            'active_pairs': len(self.active_strategies),
            'total_changes': sum(len(hist) for hist in self.strategy_history.values()),
            'regime_distribution': defaultdict(int),
            'strategy_usage': defaultdict(int),
            'average_strategies_per_pair': 0
        }
        
        # Count regime distribution
        for pair, regime in self.last_regime.items():
            stats['regime_distribution'][regime] += 1
        
        # Count strategy usage
        total_strategies = 0
        for strategies in self.active_strategies.values():
            total_strategies += len(strategies)
            for strategy in strategies:
                stats['strategy_usage'][strategy] += 1
        
        # Calculate average
        if stats['active_pairs'] > 0:
            stats['average_strategies_per_pair'] = total_strategies / stats['active_pairs']
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['regime_distribution'] = dict(stats['regime_distribution'])
        stats['strategy_usage'] = dict(stats['strategy_usage'])
        
        return stats
    
    async def periodic_strategy_review(self, interval_minutes: int = 60):
        """Periodically review and update strategies for all pairs."""
        while True:
            try:
                logger.info("Starting periodic strategy review...")
                
                for pair in list(self.active_strategies.keys()):
                    try:
                        # Re-evaluate strategies
                        new_strategies, context = await self.select_strategies_with_regime(pair)
                        
                        # Log review
                        logger.debug(f"Reviewed {pair}: {new_strategies}")
                        
                        # Small delay between pairs
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Review failed for {pair}: {e}")
                
                # Log statistics
                stats = self.get_strategy_statistics()
                logger.info(f"Strategy review complete. Stats: {json.dumps(stats, indent=2)}")
                
            except Exception as e:
                logger.error(f"Periodic review error: {e}")
            
            # Wait for next review
            await asyncio.sleep(interval_minutes * 60)


# Global instance
_integrated_selector = None

def get_integrated_selector() -> IntegratedStrategySelector:
    """Get global integrated strategy selector instance."""
    global _integrated_selector
    if _integrated_selector is None:
        _integrated_selector = IntegratedStrategySelector()
    return _integrated_selector