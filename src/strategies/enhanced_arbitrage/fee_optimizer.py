"""
Fee optimization system for minimizing trading costs across exchanges.
Includes dynamic fee tiers, VIP discounts, and optimal routing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeeStructure:
    """Fee structure for an exchange"""
    exchange: str
    maker_fee: float  # Base maker fee
    taker_fee: float  # Base taker fee
    vip_level: int = 0
    volume_30d: float = 0
    bnb_discount: float = 0  # Binance specific
    fee_currency: str = "USD"  # Currency fees are paid in
    
    # Tiered fee structure
    volume_tiers: List[Tuple[float, float, float]] = None  # [(volume, maker, taker)]
    
    def get_effective_fees(self) -> Tuple[float, float]:
        """Get effective fees considering VIP level and discounts"""
        maker = self.maker_fee
        taker = self.taker_fee
        
        # Apply volume-based discounts
        if self.volume_tiers and self.volume_30d > 0:
            for volume, tier_maker, tier_taker in reversed(self.volume_tiers):
                if self.volume_30d >= volume:
                    maker = tier_maker
                    taker = tier_taker
                    break
        
        # Apply exchange-specific discounts
        if self.exchange == "binance" and self.bnb_discount > 0:
            maker *= (1 - self.bnb_discount)
            taker *= (1 - self.bnb_discount)
        
        return maker, taker


class FeeOptimizer:
    """
    Optimizes trading fees across multiple exchanges.
    Considers fee tiers, discounts, and optimal routing.
    """
    
    def __init__(self):
        self.fee_structures: Dict[str, FeeStructure] = {}
        self.cached_routes: Dict[str, Any] = {}
        self.volume_history: Dict[str, List[float]] = {}
        
        # Default fee structures
        self._initialize_default_fees()
    
    def _initialize_default_fees(self):
        """Initialize default fee structures for major exchanges"""
        # Binance fee tiers (30d volume in USD, maker fee, taker fee)
        binance_tiers = [
            (0, 0.001, 0.001),
            (1_000_000, 0.0009, 0.001),
            (5_000_000, 0.0008, 0.001),
            (10_000_000, 0.0007, 0.001),
            (50_000_000, 0.0006, 0.0009),
            (100_000_000, 0.0004, 0.0007),
        ]
        
        self.fee_structures['binance'] = FeeStructure(
            exchange='binance',
            maker_fee=0.001,
            taker_fee=0.001,
            volume_tiers=binance_tiers,
            bnb_discount=0.25  # 25% discount when using BNB
        )
        
        # KuCoin fee tiers
        kucoin_tiers = [
            (0, 0.001, 0.001),
            (50_000, 0.0009, 0.001),
            (200_000, 0.0008, 0.001),
            (500_000, 0.0007, 0.0009),
            (1_000_000, 0.0006, 0.0008),
            (2_000_000, 0.0005, 0.0007),
        ]
        
        self.fee_structures['kucoin'] = FeeStructure(
            exchange='kucoin',
            maker_fee=0.001,
            taker_fee=0.001,
            volume_tiers=kucoin_tiers
        )
        
        # Bybit fee structure
        self.fee_structures['bybit'] = FeeStructure(
            exchange='bybit',
            maker_fee=0.001,
            taker_fee=0.001
        )
        
        # OKX fee structure
        self.fee_structures['okx'] = FeeStructure(
            exchange='okx',
            maker_fee=0.0008,
            taker_fee=0.001
        )
        
        # Coinbase fee structure
        coinbase_tiers = [
            (0, 0.004, 0.006),  # 0.4% maker, 0.6% taker
            (10_000, 0.0035, 0.0055),
            (50_000, 0.0025, 0.0045),
            (100_000, 0.0015, 0.0035),
            (1_000_000, 0.001, 0.0025),
            (10_000_000, 0.0008, 0.002),
        ]
        
        self.fee_structures['coinbase'] = FeeStructure(
            exchange='coinbase',
            maker_fee=0.004,
            taker_fee=0.006,
            volume_tiers=coinbase_tiers
        )
        
        # Kraken fee structure
        kraken_tiers = [
            (0, 0.0016, 0.0026),  # 0.16% maker, 0.26% taker
            (50_000, 0.0014, 0.0024),
            (100_000, 0.0012, 0.0022),
            (250_000, 0.001, 0.002),
            (500_000, 0.0008, 0.0018),
            (1_000_000, 0.0006, 0.0016),
            (2_500_000, 0.0004, 0.0014),
            (5_000_000, 0.0002, 0.0012),
            (10_000_000, 0.0, 0.001),  # 0% maker fee!
        ]
        
        self.fee_structures['kraken'] = FeeStructure(
            exchange='kraken',
            maker_fee=0.0016,
            taker_fee=0.0026,
            volume_tiers=kraken_tiers
        )
    
    def update_volume_data(self, exchange: str, volume_30d: float):
        """Update 30-day volume for an exchange"""
        if exchange in self.fee_structures:
            self.fee_structures[exchange].volume_30d = volume_30d
            logger.info(f"Updated {exchange} 30d volume: ${volume_30d:,.2f}")
    
    def calculate_trade_cost(self, exchange: str, trade_value: float,
                           is_maker: bool = False) -> float:
        """
        Calculate the cost of a trade including fees.
        
        Args:
            exchange: Exchange name
            trade_value: Value of the trade in USD
            is_maker: Whether this is a maker order
            
        Returns:
            Total fee cost in USD
        """
        if exchange not in self.fee_structures:
            logger.warning(f"Unknown exchange {exchange}, using default fees")
            return trade_value * 0.001  # Default 0.1%
        
        fee_struct = self.fee_structures[exchange]
        maker_fee, taker_fee = fee_struct.get_effective_fees()
        
        fee_rate = maker_fee if is_maker else taker_fee
        return trade_value * fee_rate
    
    def optimize_arbitrage_route(self, 
                               buy_exchanges: List[str],
                               sell_exchanges: List[str],
                               trade_value: float,
                               use_maker_orders: bool = True) -> Dict[str, Any]:
        """
        Find the optimal exchange pair for arbitrage considering fees.
        
        Args:
            buy_exchanges: List of exchanges where we can buy
            sell_exchanges: List of exchanges where we can sell
            trade_value: Value of the trade
            use_maker_orders: Whether to use maker orders
            
        Returns:
            Optimal routing information
        """
        best_route = None
        min_total_fee = float('inf')
        
        for buy_ex in buy_exchanges:
            for sell_ex in sell_exchanges:
                # Calculate fees for both legs
                buy_fee = self.calculate_trade_cost(
                    buy_ex, trade_value, use_maker_orders
                )
                sell_fee = self.calculate_trade_cost(
                    sell_ex, trade_value, use_maker_orders
                )
                
                total_fee = buy_fee + sell_fee
                
                if total_fee < min_total_fee:
                    min_total_fee = total_fee
                    best_route = {
                        'buy_exchange': buy_ex,
                        'sell_exchange': sell_ex,
                        'total_fee': total_fee,
                        'buy_fee': buy_fee,
                        'sell_fee': sell_fee,
                        'fee_percentage': (total_fee / trade_value) * 100
                    }
        
        return best_route
    
    def calculate_break_even_spread(self, exchange1: str, exchange2: str,
                                  trade_value: float,
                                  use_maker_orders: bool = True) -> float:
        """
        Calculate minimum spread needed to break even after fees.
        
        Args:
            exchange1: First exchange
            exchange2: Second exchange
            trade_value: Trade value
            use_maker_orders: Whether using maker orders
            
        Returns:
            Break-even spread percentage
        """
        fee1 = self.calculate_trade_cost(exchange1, trade_value, use_maker_orders)
        fee2 = self.calculate_trade_cost(exchange2, trade_value, use_maker_orders)
        
        total_fee_pct = ((fee1 + fee2) / trade_value) * 100
        
        # Add small buffer for other costs
        return total_fee_pct * 1.1  # 10% buffer
    
    def optimize_order_type(self, exchange: str, urgency: float,
                          spread_bps: float) -> str:
        """
        Determine optimal order type based on urgency and spread.
        
        Args:
            exchange: Exchange name
            urgency: Urgency score (0-1)
            spread_bps: Current spread in basis points
            
        Returns:
            'maker' or 'taker'
        """
        if urgency > 0.8:
            return 'taker'  # High urgency, use market order
        
        # Get fee difference
        fee_struct = self.fee_structures.get(exchange)
        if not fee_struct:
            return 'maker'  # Default to maker
        
        maker_fee, taker_fee = fee_struct.get_effective_fees()
        fee_diff_bps = (taker_fee - maker_fee) * 10000
        
        # If spread is wide enough to compensate for fee difference
        if spread_bps > fee_diff_bps * 2:
            return 'maker'
        else:
            return 'taker'
    
    def calculate_multi_hop_fees(self, path: List[Tuple[str, str, str]],
                               values: List[float]) -> Dict[str, Any]:
        """
        Calculate fees for multi-hop arbitrage.
        
        Args:
            path: List of (exchange, action, symbol) tuples
            values: List of trade values for each hop
            
        Returns:
            Fee breakdown for the path
        """
        total_fees = 0
        hop_details = []
        
        for i, (exchange, action, symbol) in enumerate(path):
            is_maker = True  # Assume maker for multi-hop
            fee = self.calculate_trade_cost(
                exchange, values[i], is_maker
            )
            
            total_fees += fee
            hop_details.append({
                'hop': i + 1,
                'exchange': exchange,
                'action': action,
                'symbol': symbol,
                'value': values[i],
                'fee': fee,
                'fee_pct': (fee / values[i]) * 100
            })
        
        return {
            'total_fees': total_fees,
            'average_fee_pct': (total_fees / sum(values)) * 100,
            'hop_details': hop_details,
            'path': path
        }
    
    def suggest_volume_optimization(self, current_volumes: Dict[str, float],
                                  capital: float) -> Dict[str, Any]:
        """
        Suggest how to distribute volume to optimize fee tiers.
        
        Args:
            current_volumes: Current 30d volumes by exchange
            capital: Available capital
            
        Returns:
            Volume distribution suggestions
        """
        suggestions = {}
        
        for exchange, fee_struct in self.fee_structures.items():
            if not fee_struct.volume_tiers:
                continue
            
            current_vol = current_volumes.get(exchange, 0)
            current_tier_idx = 0
            
            # Find current tier
            for i, (vol_threshold, _, _) in enumerate(fee_struct.volume_tiers):
                if current_vol >= vol_threshold:
                    current_tier_idx = i
            
            # Check if close to next tier
            if current_tier_idx < len(fee_struct.volume_tiers) - 1:
                next_tier_vol = fee_struct.volume_tiers[current_tier_idx + 1][0]
                volume_needed = next_tier_vol - current_vol
                
                if volume_needed < capital * 30:  # Can reach in 30 days
                    _, current_maker, current_taker = fee_struct.volume_tiers[current_tier_idx]
                    _, next_maker, next_taker = fee_struct.volume_tiers[current_tier_idx + 1]
                    
                    fee_savings = ((current_maker - next_maker) + 
                                 (current_taker - next_taker)) / 2
                    
                    suggestions[exchange] = {
                        'current_tier': current_tier_idx,
                        'next_tier': current_tier_idx + 1,
                        'volume_needed': volume_needed,
                        'days_to_reach': volume_needed / capital,
                        'fee_savings_bps': fee_savings * 10000,
                        'monthly_savings': capital * 30 * fee_savings
                    }
        
        return suggestions
    
    def calculate_effective_spread(self, gross_spread: float,
                                 buy_exchange: str, sell_exchange: str,
                                 trade_value: float,
                                 is_maker: bool = True) -> float:
        """
        Calculate effective spread after fees.
        
        Args:
            gross_spread: Spread before fees (percentage)
            buy_exchange: Exchange to buy on
            sell_exchange: Exchange to sell on
            trade_value: Trade value
            is_maker: Whether using maker orders
            
        Returns:
            Net spread after fees (percentage)
        """
        buy_fee = self.calculate_trade_cost(buy_exchange, trade_value, is_maker)
        sell_fee = self.calculate_trade_cost(sell_exchange, trade_value, is_maker)
        
        total_fee_pct = ((buy_fee + sell_fee) / trade_value) * 100
        
        return gross_spread - total_fee_pct