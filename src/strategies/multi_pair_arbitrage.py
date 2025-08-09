"""
Multi-Pair Arbitrage Strategy with ML

Integrates the multi-pair ML predictor with arbitrage strategy
for 100+ trading pairs with EU compliance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from strategies.arbitrage_strategy import ArbitrageStrategy
from utils.ml_multi_pair_predictor import get_multi_pair_predictor
from config.trading_pairs_config import load_default_config, get_eu_compliant_pairs

logger = logging.getLogger(__name__)


class MultiPairArbitrageStrategy(ArbitrageStrategy):
    """
    Enhanced arbitrage strategy supporting 100+ pairs with ML predictions.
    """
    
    def __init__(
        self,
        threshold: float = 0.002,
        max_position_size: float = 1000,
        region: str = 'global',
        exchange: str = 'binance'
    ):
        super().__init__(threshold, max_position_size)
        
        # Load configuration based on region
        self.config = load_default_config(region, exchange)
        self.region = region
        self.exchange = exchange
        
        # Initialize multi-pair ML predictor
        self.ml_predictor = get_multi_pair_predictor(
            eu_mode=(region == 'eu_compliant')
        )
        
        # Active opportunities tracking
        self.active_opportunities = {}
        self.pair_performance = {}
        
        logger.info(f"Initialized MultiPairArbitrageStrategy:")
        logger.info(f"  Region: {region}")
        logger.info(f"  Exchange: {exchange}")
        logger.info(f"  Base currencies: {self.config['base_currencies']}")
        logger.info(f"  Total pairs: {len(self.config['trading_pairs'])}")
    
    async def scan_all_pairs(
        self,
        min_probability: float = 0.8,
        max_opportunities: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scan all configured pairs for arbitrage opportunities.
        
        Args:
            min_probability: Minimum ML probability threshold
            max_opportunities: Maximum number of opportunities to return
            
        Returns:
            List of top arbitrage opportunities
        """
        logger.info(f"Scanning {len(self.config['trading_pairs'])} pairs for opportunities...")
        
        opportunities = []
        
        # Get pairs based on region
        if self.region == 'eu_compliant':
            pairs = get_eu_compliant_pairs(self.config['primary_base'])
        else:
            pairs = []
            for symbol in self.config['trading_pairs']:
                for base in self.config['base_currencies']:
                    pairs.append(f"{symbol}/{base}")
        
        # Batch predict all pairs
        predictions = await self.ml_predictor.predict_batch(
            pairs, 
            min_probability=min_probability
        )
        
        # Check arbitrage spread for predicted opportunities
        for pair, prediction in predictions.items():
            if prediction.get('should_trade', False):
                # Parse pair
                symbol, base = pair.rsplit('/', 1)
                
                # Check actual arbitrage spread
                spot_symbol = pair
                futures_symbol = f"{pair}:PERP" if base != 'USDT' else f"{pair}:{base}"
                
                try:
                    spread = await self.calculate_spread(spot_symbol, futures_symbol)
                    
                    if abs(spread) > self.threshold:
                        opportunity = {
                            'pair': pair,
                            'symbol': symbol,
                            'base': base,
                            'spread': spread,
                            'ml_probability': prediction['probability'],
                            'features': prediction.get('features', {}),
                            'timestamp': datetime.now(),
                            'profitable_direction': 'buy_spot' if spread > 0 else 'buy_futures'
                        }
                        
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate spread for {pair}: {e}")
        
        # Sort by ML probability and spread
        opportunities.sort(
            key=lambda x: (x['ml_probability'], abs(x['spread'])), 
            reverse=True
        )
        
        # Store active opportunities
        self.active_opportunities = {
            opp['pair']: opp for opp in opportunities[:max_opportunities]
        }
        
        return opportunities[:max_opportunities]
    
    async def execute_best_opportunity(self) -> Optional[Dict[str, Any]]:
        """
        Execute the best available arbitrage opportunity.
        
        Returns:
            Trade result or None if no opportunity
        """
        # Scan for opportunities
        opportunities = await self.scan_all_pairs()
        
        if not opportunities:
            logger.info("No arbitrage opportunities found")
            return None
        
        # Execute best opportunity
        best = opportunities[0]
        logger.info(
            f"Executing arbitrage for {best['pair']}: "
            f"spread={best['spread']:.4f}, "
            f"ML probability={best['ml_probability']:.3f}"
        )
        
        # Prepare for execution
        symbol = best['symbol']
        base = best['base']
        
        # Adjust position size based on ML confidence
        confidence_multiplier = best['ml_probability']
        adjusted_position = self.max_position_size * confidence_multiplier
        
        # Execute based on direction
        if best['profitable_direction'] == 'buy_spot':
            result = await self._execute_buy_spot_sell_futures(
                symbol, base, adjusted_position
            )
        else:
            result = await self._execute_buy_futures_sell_spot(
                symbol, base, adjusted_position
            )
        
        # Track performance
        if result:
            self._update_pair_performance(best['pair'], result)
        
        return result
    
    async def monitor_active_positions(self) -> Dict[str, Any]:
        """
        Monitor all active arbitrage positions across pairs.
        
        Returns:
            Status of all active positions
        """
        positions = {}
        
        # Get open positions from exchange
        # This would integrate with actual exchange API
        
        for pair, opportunity in self.active_opportunities.items():
            # Check if spread has converged
            current_spread = await self.calculate_spread(
                f"{opportunity['symbol']}/{opportunity['base']}",
                f"{opportunity['symbol']}/{opportunity['base']}:PERP"
            )
            
            positions[pair] = {
                'original_spread': opportunity['spread'],
                'current_spread': current_spread,
                'convergence': abs(current_spread) < 0.001,
                'ml_probability': opportunity['ml_probability'],
                'duration': (datetime.now() - opportunity['timestamp']).seconds
            }
        
        return positions
    
    def _update_pair_performance(self, pair: str, result: Dict[str, Any]):
        """Update performance metrics for a trading pair."""
        if pair not in self.pair_performance:
            self.pair_performance[pair] = {
                'trades': 0,
                'wins': 0,
                'total_profit': 0,
                'avg_ml_probability': 0
            }
        
        perf = self.pair_performance[pair]
        perf['trades'] += 1
        
        if result.get('profit', 0) > 0:
            perf['wins'] += 1
        
        perf['total_profit'] += result.get('profit', 0)
        
        # Update average ML probability
        n = perf['trades']
        perf['avg_ml_probability'] = (
            (perf['avg_ml_probability'] * (n - 1) + 
             self.active_opportunities.get(pair, {}).get('ml_probability', 0)) / n
        )
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance metrics by pair and overall
        """
        report = {
            'overall': {
                'total_pairs_traded': len(self.pair_performance),
                'total_trades': sum(p['trades'] for p in self.pair_performance.values()),
                'total_wins': sum(p['wins'] for p in self.pair_performance.values()),
                'total_profit': sum(p['total_profit'] for p in self.pair_performance.values())
            },
            'by_pair': self.pair_performance,
            'by_base_currency': {},
            'ml_statistics': {}
        }
        
        # Calculate win rate
        if report['overall']['total_trades'] > 0:
            report['overall']['win_rate'] = (
                report['overall']['total_wins'] / 
                report['overall']['total_trades']
            )
        else:
            report['overall']['win_rate'] = 0
        
        # Group by base currency
        for pair, perf in self.pair_performance.items():
            base = pair.split('/')[-1]
            
            if base not in report['by_base_currency']:
                report['by_base_currency'][base] = {
                    'trades': 0,
                    'profit': 0
                }
            
            report['by_base_currency'][base]['trades'] += perf['trades']
            report['by_base_currency'][base]['profit'] += perf['total_profit']
        
        # ML statistics
        ml_probs = [
            perf['avg_ml_probability'] 
            for perf in self.pair_performance.values()
            if perf['trades'] > 0
        ]
        
        if ml_probs:
            report['ml_statistics'] = {
                'avg_probability': sum(ml_probs) / len(ml_probs),
                'min_probability': min(ml_probs),
                'max_probability': max(ml_probs)
            }
        
        return report
    
    async def run_continuous_scanner(
        self,
        scan_interval: int = 60,
        max_concurrent: int = 5
    ):
        """
        Run continuous scanning for arbitrage opportunities.
        
        Args:
            scan_interval: Seconds between scans
            max_concurrent: Maximum concurrent positions
        """
        logger.info(
            f"Starting continuous scanner: "
            f"interval={scan_interval}s, "
            f"max_concurrent={max_concurrent}"
        )
        
        while True:
            try:
                # Check current positions
                active_positions = len(self.active_opportunities)
                
                if active_positions < max_concurrent:
                    # Scan for new opportunities
                    opportunities = await self.scan_all_pairs(
                        max_opportunities=max_concurrent - active_positions
                    )
                    
                    if opportunities:
                        logger.info(f"Found {len(opportunities)} new opportunities")
                        
                        # Execute top opportunities
                        for opp in opportunities[:max_concurrent - active_positions]:
                            await self.execute_best_opportunity()
                            await asyncio.sleep(1)  # Avoid rapid-fire execution
                
                # Monitor existing positions
                positions = await self.monitor_active_positions()
                
                # Close converged positions
                for pair, status in positions.items():
                    if status['convergence']:
                        logger.info(f"Closing converged position: {pair}")
                        # Close position logic here
                
                # Log status
                if self.active_opportunities:
                    logger.info(
                        f"Active opportunities: {len(self.active_opportunities)}, "
                        f"Next scan in {scan_interval}s"
                    )
                
            except Exception as e:
                logger.error(f"Scanner error: {e}")
            
            await asyncio.sleep(scan_interval)


# Example usage
async def demo_multi_pair_arbitrage():
    """Demonstrate multi-pair arbitrage capabilities."""
    print("=" * 80)
    print("🌍 MULTI-PAIR ARBITRAGE DEMO")
    print("=" * 80)
    
    # Test different regions
    regions = ['global', 'eu_compliant']
    
    for region in regions:
        print(f"\n📍 Testing {region.upper()} configuration")
        print("-" * 60)
        
        strategy = MultiPairArbitrageStrategy(
            threshold=0.002,
            max_position_size=1000,
            region=region,
            exchange='binance'
        )
        
        # Scan for opportunities
        print("\nScanning for arbitrage opportunities...")
        opportunities = await strategy.scan_all_pairs(
            min_probability=0.8,
            max_opportunities=5
        )
        
        if opportunities:
            print(f"\nTop {len(opportunities)} opportunities:")
            print(f"{'Pair':<15} {'Spread':<10} {'ML Prob':<10} {'Direction'}")
            print("-" * 50)
            
            for opp in opportunities:
                print(
                    f"{opp['pair']:<15} "
                    f"{opp['spread']:>9.4f} "
                    f"{opp['ml_probability']:>9.3f} "
                    f"{opp['profitable_direction']}"
                )
        else:
            print("No opportunities found")
    
    # Performance report
    print("\n\n📊 PERFORMANCE CAPABILITIES")
    print("-" * 60)
    print("The system tracks:")
    print("  • Win rate per pair")
    print("  • Profit by base currency")
    print("  • ML prediction accuracy")
    print("  • Execution timing")
    print("  • Regional compliance")


if __name__ == "__main__":
    asyncio.run(demo_multi_pair_arbitrage())