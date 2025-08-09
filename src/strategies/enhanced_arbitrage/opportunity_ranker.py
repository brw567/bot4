"""
Real-time opportunity ranking system for arbitrage and statistical arbitrage.
Ranks opportunities by profitability, risk, and execution probability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
import logging
import json

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of trading opportunities"""
    CROSS_EXCHANGE_ARB = "cross_exchange_arbitrage"
    STATISTICAL_ARB = "statistical_arbitrage"
    TRIANGULAR_ARB = "triangular_arbitrage"
    MULTI_HOP_ARB = "multi_hop_arbitrage"


@dataclass
class TradingOpportunity:
    """Base class for all trading opportunities"""
    id: str
    type: OpportunityType
    expected_profit_pct: float
    expected_profit_usd: float
    required_capital: float
    confidence_score: float  # 0-1
    time_sensitivity: float  # 0-1, higher means more time sensitive
    risk_score: float  # 0-1, lower is better
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Execution details
    estimated_execution_time: float = 0  # seconds
    success_probability: float = 0.95
    
    # For heap comparison (max heap using negative score)
    def __lt__(self, other):
        return self.get_composite_score() > other.get_composite_score()
    
    def get_composite_score(self) -> float:
        """Calculate composite score for ranking"""
        # Override in subclasses for specific scoring
        return (self.expected_profit_pct * self.confidence_score * 
                self.success_probability / (1 + self.risk_score))


@dataclass
class ArbitrageOpportunity(TradingOpportunity):
    """Cross-exchange arbitrage opportunity"""
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    max_size: float
    spread_pct: float
    fee_adjusted_profit_pct: float
    transfer_time: Optional[float] = None  # minutes
    
    def get_composite_score(self) -> float:
        """Arbitrage-specific scoring"""
        # Prioritize fee-adjusted profit and quick execution
        time_factor = 1 / (1 + self.estimated_execution_time / 60)  # Prefer faster
        size_factor = min(1, self.max_size / 10000)  # Normalize by $10k
        
        return (self.fee_adjusted_profit_pct * 
                self.confidence_score * 
                self.success_probability * 
                time_factor * 
                size_factor / 
                (1 + self.risk_score))


@dataclass 
class StatArbOpportunity(TradingOpportunity):
    """Statistical arbitrage opportunity"""
    pair: Tuple[str, str]
    z_score: float
    half_life: float
    signal_type: str  # 'long_spread', 'short_spread'
    hedge_ratio: float
    cointegration_pvalue: float
    kelly_size_pct: float
    
    def get_composite_score(self) -> float:
        """Stat arb-specific scoring"""
        # Prioritize extreme z-scores and good cointegration
        z_score_factor = min(abs(self.z_score) / 3, 1)  # Normalize by z=3
        coint_factor = 1 - self.cointegration_pvalue  # Lower p-value is better
        kelly_factor = self.kelly_size_pct / 2  # Normalize by 2% position
        
        return (self.expected_profit_pct * 
                z_score_factor * 
                coint_factor * 
                kelly_factor * 
                self.confidence_score / 
                (1 + self.risk_score))


class OpportunityRanker:
    """
    Ranks and manages trading opportunities in real-time.
    Maintains a priority queue of opportunities.
    """
    
    def __init__(self, 
                 max_opportunities: int = 100,
                 stale_threshold_seconds: int = 300):
        """
        Initialize the ranker.
        
        Args:
            max_opportunities: Maximum opportunities to track
            stale_threshold_seconds: Time before opportunity is considered stale
        """
        self.max_opportunities = max_opportunities
        self.stale_threshold = timedelta(seconds=stale_threshold_seconds)
        
        # Priority queue of opportunities (max heap)
        self.opportunities: List[TradingOpportunity] = []
        
        # Quick lookup by ID
        self.opportunity_map: Dict[str, TradingOpportunity] = {}
        
        # Scoring weights (configurable)
        self.weights = {
            'profit': 0.3,
            'confidence': 0.25,
            'risk': 0.2,
            'size': 0.15,
            'time': 0.1
        }
        
        # Performance tracking
        self.stats = {
            'total_evaluated': 0,
            'total_accepted': 0,
            'total_executed': 0,
            'avg_score': 0
        }
    
    def add_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """
        Add a new opportunity to the ranking.
        
        Args:
            opportunity: Trading opportunity to add
            
        Returns:
            True if added, False if rejected
        """
        # Check if already exists
        if opportunity.id in self.opportunity_map:
            self.update_opportunity(opportunity)
            return True
        
        # Calculate enhanced score
        score = self._calculate_enhanced_score(opportunity)
        
        # Add to heap
        heapq.heappush(self.opportunities, opportunity)
        self.opportunity_map[opportunity.id] = opportunity
        
        # Maintain size limit
        if len(self.opportunities) > self.max_opportunities:
            removed = heapq.heappop(self.opportunities)
            del self.opportunity_map[removed.id]
        
        # Update stats
        self.stats['total_evaluated'] += 1
        self.stats['total_accepted'] += 1
        
        return True
    
    def update_opportunity(self, opportunity: TradingOpportunity):
        """Update an existing opportunity"""
        if opportunity.id in self.opportunity_map:
            # Remove old version
            old = self.opportunity_map[opportunity.id]
            self.opportunities.remove(old)
            heapq.heapify(self.opportunities)
            
            # Add new version
            heapq.heappush(self.opportunities, opportunity)
            self.opportunity_map[opportunity.id] = opportunity
    
    def get_top_opportunities(self, n: int = 10, 
                            opportunity_type: Optional[OpportunityType] = None) -> List[TradingOpportunity]:
        """
        Get top N opportunities, optionally filtered by type.
        
        Args:
            n: Number of opportunities to return
            opportunity_type: Filter by type (optional)
            
        Returns:
            List of top opportunities
        """
        # Clean stale opportunities first
        self._remove_stale_opportunities()
        
        # Get all opportunities sorted by score
        all_opps = sorted(self.opportunities, reverse=True)
        
        # Filter by type if specified
        if opportunity_type:
            filtered = [opp for opp in all_opps if opp.type == opportunity_type]
        else:
            filtered = all_opps
        
        return filtered[:n]
    
    def _calculate_enhanced_score(self, opp: TradingOpportunity) -> float:
        """Calculate enhanced score with multiple factors"""
        # Base scores
        profit_score = self._normalize_profit(opp.expected_profit_pct)
        risk_adj_score = (1 - opp.risk_score)
        time_score = self._calculate_time_score(opp)
        size_score = self._normalize_size(opp.required_capital)
        
        # Type-specific adjustments
        type_multiplier = self._get_type_multiplier(opp)
        
        # Weighted combination
        score = (
            self.weights['profit'] * profit_score +
            self.weights['confidence'] * opp.confidence_score +
            self.weights['risk'] * risk_adj_score +
            self.weights['size'] * size_score +
            self.weights['time'] * time_score
        ) * type_multiplier
        
        return score
    
    def _normalize_profit(self, profit_pct: float) -> float:
        """Normalize profit percentage to 0-1 scale"""
        # Assume 5% is excellent, use sigmoid
        return 1 / (1 + np.exp(-profit_pct / 2))
    
    def _normalize_size(self, capital: float) -> float:
        """Normalize capital requirement to favor optimal sizes"""
        # Prefer opportunities between $1k-$10k
        if capital < 1000:
            return capital / 1000
        elif capital <= 10000:
            return 1.0
        else:
            return 10000 / capital
    
    def _calculate_time_score(self, opp: TradingOpportunity) -> float:
        """Calculate time-based score"""
        # Decay based on age
        age = (datetime.now() - opp.timestamp).total_seconds()
        age_factor = np.exp(-age / 300)  # 5-minute half-life
        
        # Urgency factor
        urgency = opp.time_sensitivity
        
        # Execution speed factor
        if opp.estimated_execution_time > 0:
            speed_factor = 1 / (1 + opp.estimated_execution_time / 10)
        else:
            speed_factor = 1
        
        return age_factor * (1 + urgency) * speed_factor / 2
    
    def _get_type_multiplier(self, opp: TradingOpportunity) -> float:
        """Get multiplier based on opportunity type"""
        multipliers = {
            OpportunityType.CROSS_EXCHANGE_ARB: 1.2,  # Prefer simple arbitrage
            OpportunityType.STATISTICAL_ARB: 1.0,
            OpportunityType.TRIANGULAR_ARB: 0.9,
            OpportunityType.MULTI_HOP_ARB: 0.8  # More complex, lower priority
        }
        return multipliers.get(opp.type, 1.0)
    
    def _remove_stale_opportunities(self):
        """Remove opportunities older than threshold"""
        now = datetime.now()
        
        # Find stale opportunities
        stale_ids = []
        for opp in self.opportunities:
            if now - opp.timestamp > self.stale_threshold:
                stale_ids.append(opp.id)
        
        # Remove them
        for opp_id in stale_ids:
            if opp_id in self.opportunity_map:
                opp = self.opportunity_map[opp_id]
                self.opportunities.remove(opp)
                del self.opportunity_map[opp_id]
        
        # Re-heapify if needed
        if stale_ids:
            heapq.heapify(self.opportunities)
    
    def mark_executed(self, opportunity_id: str, success: bool = True):
        """Mark an opportunity as executed"""
        if opportunity_id in self.opportunity_map:
            self.stats['total_executed'] += 1
            
            # Remove from active opportunities
            opp = self.opportunity_map[opportunity_id]
            self.opportunities.remove(opp)
            del self.opportunity_map[opportunity_id]
            heapq.heapify(self.opportunities)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ranking statistics"""
        if not self.opportunities:
            return self.stats
        
        # Calculate current stats
        scores = [opp.get_composite_score() for opp in self.opportunities]
        profits = [opp.expected_profit_pct for opp in self.opportunities]
        
        current_stats = {
            **self.stats,
            'active_opportunities': len(self.opportunities),
            'avg_score': np.mean(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'avg_profit_pct': np.mean(profits) if profits else 0,
            'opportunity_types': self._count_by_type()
        }
        
        return current_stats
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count opportunities by type"""
        counts = {}
        for opp in self.opportunities:
            type_name = opp.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def get_opportunity_details(self, opportunity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an opportunity"""
        opp = self.opportunity_map.get(opportunity_id)
        if not opp:
            return None
        
        details = {
            'id': opp.id,
            'type': opp.type.value,
            'score': opp.get_composite_score(),
            'expected_profit_pct': opp.expected_profit_pct,
            'expected_profit_usd': opp.expected_profit_usd,
            'required_capital': opp.required_capital,
            'confidence': opp.confidence_score,
            'risk': opp.risk_score,
            'age_seconds': (datetime.now() - opp.timestamp).total_seconds(),
            'time_sensitivity': opp.time_sensitivity,
            'success_probability': opp.success_probability
        }
        
        # Add type-specific details
        if isinstance(opp, ArbitrageOpportunity):
            details.update({
                'buy_exchange': opp.buy_exchange,
                'sell_exchange': opp.sell_exchange,
                'symbol': opp.symbol,
                'spread_pct': opp.spread_pct,
                'fee_adjusted_profit_pct': opp.fee_adjusted_profit_pct
            })
        elif isinstance(opp, StatArbOpportunity):
            details.update({
                'pair': opp.pair,
                'z_score': opp.z_score,
                'signal_type': opp.signal_type,
                'half_life': opp.half_life,
                'kelly_size_pct': opp.kelly_size_pct
            })
        
        return details
    
    def export_opportunities(self, format: str = 'json') -> Union[str, pd.DataFrame]:
        """Export current opportunities in specified format"""
        opps_data = []
        
        for opp in sorted(self.opportunities, reverse=True):
            opps_data.append(self.get_opportunity_details(opp.id))
        
        if format == 'json':
            import json
            return json.dumps(opps_data, indent=2, default=str)
        elif format == 'dataframe':
            return pd.DataFrame(opps_data)
        else:
            return str(opps_data)