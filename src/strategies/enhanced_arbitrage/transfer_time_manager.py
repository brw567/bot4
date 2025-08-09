"""
Transfer time management for cross-exchange arbitrage.
Tracks and predicts transfer times, manages transfer risks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TransferRoute:
    """Information about a transfer route between exchanges"""
    from_exchange: str
    to_exchange: str
    asset: str
    network: str  # e.g., 'ERC20', 'TRC20', 'BEP20', 'SOL'
    avg_time_minutes: float
    std_dev_minutes: float
    min_confirmations: int
    withdrawal_fee: float
    is_suspended: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.98  # Historical success rate
    
    def get_estimated_time(self, confidence: float = 0.95) -> float:
        """Get estimated transfer time with confidence interval"""
        # Use normal distribution for time estimate
        if confidence == 0.95:
            z_score = 1.96
        elif confidence == 0.99:
            z_score = 2.58
        else:
            z_score = 1.96
        
        return self.avg_time_minutes + (z_score * self.std_dev_minutes)


@dataclass
class TransferRecord:
    """Record of an actual transfer"""
    transfer_id: str
    route: TransferRoute
    initiated_at: datetime
    completed_at: Optional[datetime]
    amount: float
    status: str  # 'pending', 'completed', 'failed'
    actual_time_minutes: Optional[float] = None


class TransferTimeManager:
    """
    Manages transfer times and risks for cross-exchange arbitrage.
    Tracks historical data and provides predictions.
    """
    
    def __init__(self):
        self.routes: Dict[str, TransferRoute] = {}
        self.transfer_history: List[TransferRecord] = []
        self.pending_transfers: Dict[str, TransferRecord] = {}
        
        # Initialize with default transfer times
        self._initialize_default_routes()
        
        # Network congestion factors
        self.network_congestion: Dict[str, float] = {
            'ERC20': 1.0,  # 1.0 = normal, >1.0 = congested
            'BEP20': 1.0,
            'TRC20': 1.0,
            'SOL': 1.0,
            'MATIC': 1.0
        }
    
    def _initialize_default_routes(self):
        """Initialize default transfer routes and times"""
        # Common routes with historical averages
        default_routes = [
            # Binance to KuCoin
            ('binance', 'kucoin', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            ('binance', 'kucoin', 'USDT', 'ERC20', 15, 5, 12, 5.0),
            ('binance', 'kucoin', 'BTC', 'BTC', 30, 10, 2, 0.0002),
            
            # KuCoin to Binance
            ('kucoin', 'binance', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            ('kucoin', 'binance', 'USDT', 'ERC20', 15, 5, 12, 5.0),
            
            # Binance to Bybit
            ('binance', 'bybit', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            ('binance', 'bybit', 'ETH', 'ERC20', 12, 4, 12, 0.005),
            
            # Bybit to Binance
            ('bybit', 'binance', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            
            # OKX routes
            ('okx', 'binance', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            ('binance', 'okx', 'USDT', 'TRC20', 5, 2, 1, 1.0),
            
            # Coinbase routes (slower, more confirmations)
            ('coinbase', 'binance', 'USDT', 'ERC20', 20, 5, 12, 5.0),
            ('coinbase', 'binance', 'BTC', 'BTC', 60, 15, 3, 0.0005),
            ('binance', 'coinbase', 'USDT', 'ERC20', 20, 5, 12, 5.0),
            ('coinbase', 'kraken', 'USD', 'WIRE', 1440, 120, 1, 25.0),  # Wire transfer
            
            # Kraken routes
            ('kraken', 'binance', 'USDT', 'ERC20', 15, 5, 12, 5.0),
            ('kraken', 'binance', 'EUR', 'SEPA', 720, 60, 1, 0.0),  # SEPA transfer
            ('binance', 'kraken', 'USDT', 'ERC20', 15, 5, 12, 5.0),
            ('kraken', 'coinbase', 'BTC', 'BTC', 45, 10, 3, 0.0002),
        ]
        
        for route_data in default_routes:
            from_ex, to_ex, asset, network, avg_time, std_dev, confirms, fee = route_data
            
            route_key = f"{from_ex}:{to_ex}:{asset}:{network}"
            self.routes[route_key] = TransferRoute(
                from_exchange=from_ex,
                to_exchange=to_ex,
                asset=asset,
                network=network,
                avg_time_minutes=avg_time,
                std_dev_minutes=std_dev,
                min_confirmations=confirms,
                withdrawal_fee=fee
            )
    
    def get_transfer_time(self, from_exchange: str, to_exchange: str,
                         asset: str, network: Optional[str] = None,
                         confidence: float = 0.95) -> Optional[float]:
        """
        Get estimated transfer time between exchanges.
        
        Args:
            from_exchange: Source exchange
            to_exchange: Destination exchange
            asset: Asset to transfer
            network: Network to use (optional)
            confidence: Confidence level for estimate
            
        Returns:
            Estimated time in minutes, or None if route not found
        """
        if network:
            route_key = f"{from_exchange}:{to_exchange}:{asset}:{network}"
            route = self.routes.get(route_key)
            if route and not route.is_suspended:
                # Apply network congestion factor
                congestion = self.network_congestion.get(network, 1.0)
                base_time = route.get_estimated_time(confidence)
                return base_time * congestion
        else:
            # Find best network
            best_time = float('inf')
            for key, route in self.routes.items():
                if (route.from_exchange == from_exchange and
                    route.to_exchange == to_exchange and
                    route.asset == asset and
                    not route.is_suspended):
                    
                    congestion = self.network_congestion.get(route.network, 1.0)
                    time = route.get_estimated_time(confidence) * congestion
                    if time < best_time:
                        best_time = time
            
            return best_time if best_time < float('inf') else None
    
    def get_fastest_route(self, from_exchange: str, to_exchange: str,
                         asset: str) -> Optional[TransferRoute]:
        """Get the fastest available route for a transfer"""
        fastest_route = None
        min_time = float('inf')
        
        for key, route in self.routes.items():
            if (route.from_exchange == from_exchange and
                route.to_exchange == to_exchange and
                route.asset == asset and
                not route.is_suspended):
                
                time = route.avg_time_minutes
                if time < min_time:
                    min_time = time
                    fastest_route = route
        
        return fastest_route
    
    def calculate_transfer_risk(self, route: TransferRoute,
                              amount_usd: float,
                              time_window: float) -> Dict[str, float]:
        """
        Calculate risk metrics for a transfer.
        
        Args:
            route: Transfer route
            amount_usd: Transfer amount in USD
            time_window: Available time window in minutes
            
        Returns:
            Risk metrics
        """
        # Probability of completing within time window
        from scipy import stats
        
        z_score = (time_window - route.avg_time_minutes) / route.std_dev_minutes
        prob_on_time = stats.norm.cdf(z_score)
        
        # Expected value considering success rate
        expected_value = amount_usd * route.success_rate
        
        # Value at risk (95% confidence)
        var_95 = amount_usd * (1 - route.success_rate) * 0.95
        
        # Time risk score (0-1, higher is riskier)
        time_risk = 1 - prob_on_time
        
        # Network risk based on congestion
        network_risk = max(0, self.network_congestion.get(route.network, 1.0) - 1)
        
        # Combined risk score
        combined_risk = (time_risk * 0.5 + 
                        network_risk * 0.3 + 
                        (1 - route.success_rate) * 0.2)
        
        return {
            'probability_on_time': prob_on_time,
            'expected_value': expected_value,
            'value_at_risk_95': var_95,
            'time_risk_score': time_risk,
            'network_risk_score': network_risk,
            'combined_risk_score': combined_risk,
            'withdrawal_fee': route.withdrawal_fee
        }
    
    def update_network_congestion(self, network: str, congestion_factor: float):
        """Update network congestion factor"""
        self.network_congestion[network] = congestion_factor
        logger.info(f"Updated {network} congestion factor to {congestion_factor:.2f}")
    
    def record_transfer(self, transfer_id: str, from_exchange: str,
                       to_exchange: str, asset: str, network: str,
                       amount: float) -> TransferRecord:
        """Record a new transfer"""
        route_key = f"{from_exchange}:{to_exchange}:{asset}:{network}"
        route = self.routes.get(route_key)
        
        if not route:
            # Create new route with default values
            route = TransferRoute(
                from_exchange=from_exchange,
                to_exchange=to_exchange,
                asset=asset,
                network=network,
                avg_time_minutes=30,  # Default
                std_dev_minutes=10,
                min_confirmations=12,
                withdrawal_fee=0
            )
            self.routes[route_key] = route
        
        record = TransferRecord(
            transfer_id=transfer_id,
            route=route,
            initiated_at=datetime.now(),
            completed_at=None,
            amount=amount,
            status='pending'
        )
        
        self.pending_transfers[transfer_id] = record
        return record
    
    def complete_transfer(self, transfer_id: str, success: bool = True):
        """Mark a transfer as completed and update statistics"""
        if transfer_id not in self.pending_transfers:
            logger.warning(f"Transfer {transfer_id} not found")
            return
        
        record = self.pending_transfers[transfer_id]
        record.completed_at = datetime.now()
        record.status = 'completed' if success else 'failed'
        
        if success and record.initiated_at:
            # Calculate actual time
            actual_time = (record.completed_at - record.initiated_at).total_seconds() / 60
            record.actual_time_minutes = actual_time
            
            # Update route statistics
            self._update_route_statistics(record.route, actual_time)
        
        # Move to history
        self.transfer_history.append(record)
        del self.pending_transfers[transfer_id]
    
    def _update_route_statistics(self, route: TransferRoute, actual_time: float):
        """Update route statistics with new data point"""
        # Simple exponential moving average update
        alpha = 0.1  # Learning rate
        
        # Update average time
        route.avg_time_minutes = (1 - alpha) * route.avg_time_minutes + alpha * actual_time
        
        # Update standard deviation (simplified)
        deviation = abs(actual_time - route.avg_time_minutes)
        route.std_dev_minutes = (1 - alpha) * route.std_dev_minutes + alpha * deviation
        
        route.last_updated = datetime.now()
    
    def get_arbitrage_time_window(self, from_exchange: str, to_exchange: str,
                                 asset: str, price_volatility: float) -> Dict[str, Any]:
        """
        Calculate safe time window for arbitrage considering transfer time.
        
        Args:
            from_exchange: Source exchange
            to_exchange: Destination exchange  
            asset: Asset to transfer
            price_volatility: Hourly volatility percentage
            
        Returns:
            Time window analysis
        """
        route = self.get_fastest_route(from_exchange, to_exchange, asset)
        
        if not route:
            return {'viable': False, 'reason': 'No route found'}
        
        # Get 99% confidence transfer time
        max_transfer_time = route.get_estimated_time(0.99)
        
        # Calculate price drift risk
        hours = max_transfer_time / 60
        expected_drift = price_volatility * np.sqrt(hours) * 2  # 2 std devs
        
        # Viable if drift is less than typical spread minus fees
        typical_spread = 0.3  # 0.3% typical arbitrage spread
        fee_cost = route.withdrawal_fee / 100  # Rough estimate
        
        net_opportunity = typical_spread - fee_cost - expected_drift
        
        return {
            'viable': net_opportunity > 0.1,  # Need at least 0.1% profit
            'transfer_time_99': max_transfer_time,
            'expected_drift_pct': expected_drift,
            'net_opportunity_pct': net_opportunity,
            'route': {
                'network': route.network,
                'avg_time': route.avg_time_minutes,
                'withdrawal_fee': route.withdrawal_fee
            }
        }
    
    def optimize_transfer_routing(self, from_exchange: str, to_exchange: str,
                                asset: str, amount_usd: float,
                                max_time_minutes: float) -> Optional[Dict[str, Any]]:
        """
        Find optimal transfer route considering time and cost.
        
        Args:
            from_exchange: Source exchange
            to_exchange: Destination exchange
            asset: Asset to transfer
            amount_usd: Transfer amount in USD
            max_time_minutes: Maximum acceptable time
            
        Returns:
            Optimal route information
        """
        candidates = []
        
        for key, route in self.routes.items():
            if (route.from_exchange == from_exchange and
                route.to_exchange == to_exchange and
                route.asset == asset and
                not route.is_suspended):
                
                # Check if meets time constraint
                time_99 = route.get_estimated_time(0.99)
                if time_99 <= max_time_minutes:
                    # Calculate total cost (fee as percentage)
                    if asset == 'USDT':
                        fee_pct = route.withdrawal_fee / amount_usd
                    else:
                        fee_pct = 0.001  # Estimate 0.1% for other assets
                    
                    # Score based on speed and cost
                    speed_score = 1 - (route.avg_time_minutes / max_time_minutes)
                    cost_score = 1 - fee_pct
                    
                    # Combined score (weighted)
                    score = 0.7 * speed_score + 0.3 * cost_score
                    
                    candidates.append({
                        'route': route,
                        'score': score,
                        'estimated_time': route.avg_time_minutes,
                        'time_99': time_99,
                        'fee': route.withdrawal_fee,
                        'fee_pct': fee_pct * 100
                    })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        return {
            'network': best['route'].network,
            'estimated_time': best['estimated_time'],
            'max_time_99': best['time_99'],
            'withdrawal_fee': best['fee'],
            'fee_percentage': best['fee_pct'],
            'score': best['score'],
            'alternatives': candidates[1:3] if len(candidates) > 1 else []
        }
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of transfer statistics"""
        completed = [r for r in self.transfer_history if r.status == 'completed']
        
        if not completed:
            return {'total_transfers': 0}
        
        # Group by route
        route_stats = defaultdict(list)
        for record in completed:
            if record.actual_time_minutes:
                key = f"{record.route.from_exchange}->{record.route.to_exchange} ({record.route.asset})"
                route_stats[key].append(record.actual_time_minutes)
        
        # Calculate statistics
        summary = {
            'total_transfers': len(self.transfer_history),
            'completed_transfers': len(completed),
            'success_rate': len(completed) / len(self.transfer_history),
            'pending_transfers': len(self.pending_transfers),
            'route_statistics': {}
        }
        
        for route, times in route_stats.items():
            summary['route_statistics'][route] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'std_dev': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return summary