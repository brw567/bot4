"""
Example usage of the integrated arbitrage system.
Demonstrates initialization, configuration, and monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from strategies.enhanced_arbitrage.integrated_arbitrage_system import IntegratedArbitrageSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockExchangeAdapter:
    """Mock exchange adapter for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        self.orderbooks = {}
        self.tickers = {}
    
    async def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Mock orderbook data"""
        # Simulate different prices across exchanges
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        spread_factor = hash(self.name + symbol) % 100 / 10000  # 0-1% spread
        
        return {
            'bids': [
                {'price': base_price * (1 - spread_factor), 'size': 1.0},
                {'price': base_price * (1 - spread_factor - 0.0001), 'size': 2.0}
            ],
            'asks': [
                {'price': base_price * (1 + spread_factor), 'size': 1.0},
                {'price': base_price * (1 + spread_factor + 0.0001), 'size': 2.0}
            ]
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Mock ticker data"""
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        return {
            'last': base_price,
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'volume': 1000000
        }


async def main():
    """
    Demonstrate the integrated arbitrage system achieving target metrics:
    - Win Rate: 84-85%
    - Sharpe Ratio: 4.0+
    - Max Drawdown: <3%
    - Execution Slippage: -50%
    - Arbitrage Capture: 5-10 opportunities/day
    """
    
    # Initialize mock exchanges
    exchanges = {
        'binance': MockExchangeAdapter('binance'),
        'kucoin': MockExchangeAdapter('kucoin'),
        'bybit': MockExchangeAdapter('bybit'),
        'okx': MockExchangeAdapter('okx')
    }
    
    # Create integrated system with conservative parameters for high win rate
    system = IntegratedArbitrageSystem(
        exchanges=exchanges,
        initial_capital=100000,  # $100k starting capital
        target_win_rate=0.85,    # 85% target win rate
        max_drawdown=0.03        # 3% max drawdown
    )
    
    logger.info("=== Integrated Arbitrage System Demo ===")
    logger.info(f"Initial Capital: $100,000")
    logger.info(f"Target Win Rate: 85%")
    logger.info(f"Max Drawdown: 3%")
    logger.info(f"Target Sharpe: 4.0+")
    
    # Initialize the system
    await system.initialize()
    
    # Configure for target performance
    logger.info("\n=== Configuration for Target Performance ===")
    
    # 1. Fee optimization setup
    logger.info("Setting up fee optimization...")
    system.fee_optimizer.update_volume_data('binance', 5_000_000)  # $5M volume for lower fees
    system.fee_optimizer.update_volume_data('kucoin', 1_000_000)   # $1M volume
    
    # 2. Transfer time configuration
    logger.info("Configuring transfer times...")
    system.transfer_manager.update_network_congestion('ERC20', 1.2)  # 20% congestion
    system.transfer_manager.update_network_congestion('TRC20', 1.0)  # Normal
    
    # 3. Statistical arbitrage pairs
    logger.info("Setting up statistical arbitrage pairs...")
    # In real usage, this would be done through historical data analysis
    # system.stat_arb_engine.add_cointegrated_pair(...)
    
    # 4. Position sizing constraints (already configured in __init__)
    logger.info("Position constraints configured:")
    logger.info(f"  - Max position size: 10% of capital ($10,000)")
    logger.info(f"  - Max gross exposure: 150%")
    logger.info(f"  - Daily VaR limit: 2%")
    logger.info(f"  - Min Sharpe requirement: 2.0")
    
    # Run the system for demonstration
    logger.info("\n=== Starting Trading System ===")
    
    # Create monitoring task
    async def monitor_performance():
        """Monitor and log performance metrics"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            status = system.get_system_status()
            
            logger.info("\n=== Performance Update ===")
            logger.info(f"Win Rate: {status['performance']['win_rate']}")
            logger.info(f"Sharpe Ratio: {status['performance']['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {status['performance']['max_drawdown']}")
            logger.info(f"Total P&L: ${status['performance']['total_pnl']:,.2f}")
            logger.info(f"Total Trades: {status['performance']['total_trades']}")
            logger.info(f"Active Opportunities: {status['opportunities']['active_opportunities']}")
            
            # Check if meeting targets
            perf = system.performance
            if perf.total_trades > 20:  # After sufficient trades
                if perf.win_rate >= 0.84:
                    logger.info("✓ Win rate target achieved!")
                if perf.sharpe_ratio >= 4.0:
                    logger.info("✓ Sharpe ratio target achieved!")
                if perf.max_drawdown <= 0.03:
                    logger.info("✓ Drawdown control maintained!")
    
    # Simulate some opportunities for demonstration
    async def simulate_opportunities():
        """Simulate trading opportunities"""
        await asyncio.sleep(5)  # Wait for system to start
        
        # Simulate different types of opportunities
        opportunity_count = 0
        
        while opportunity_count < 10:  # Simulate 10 opportunities
            # Create cross-exchange arbitrage opportunity
            from strategies.enhanced_arbitrage.opportunity_ranker import (
                ArbitrageOpportunity, OpportunityType
            )
            
            opp = ArbitrageOpportunity(
                id=f"demo_arb_{opportunity_count}",
                type=OpportunityType.CROSS_EXCHANGE_ARB,
                expected_profit_pct=0.25,  # 0.25% profit
                expected_profit_usd=25,
                required_capital=10000,
                confidence_score=0.95,
                time_sensitivity=0.8,
                risk_score=0.1,
                buy_exchange='binance',
                sell_exchange='kucoin',
                symbol='BTC/USDT',
                buy_price=50000,
                sell_price=50125,
                max_size=0.2,
                spread_pct=0.25,
                fee_adjusted_profit_pct=0.15,  # After fees
                estimated_execution_time=5,
                success_probability=0.95
            )
            
            system.opportunity_ranker.add_opportunity(opp)
            logger.info(f"Added opportunity: {opp.id} - Expected profit: {opp.expected_profit_pct}%")
            
            opportunity_count += 1
            await asyncio.sleep(10)  # Wait 10 seconds between opportunities
    
    # Run all tasks
    tasks = [
        system.run(),
        monitor_performance(),
        simulate_opportunities()
    ]
    
    try:
        # Run for 5 minutes demonstration
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300
        )
    except asyncio.TimeoutError:
        logger.info("\n=== Demo completed ===")
        
        # Final status
        final_status = system.get_system_status()
        logger.info("\n=== Final Performance Summary ===")
        logger.info(f"Win Rate: {final_status['performance']['win_rate']}")
        logger.info(f"Sharpe Ratio: {final_status['performance']['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {final_status['performance']['max_drawdown']}")
        logger.info(f"Total P&L: ${final_status['performance']['total_pnl']:,.2f}")
        logger.info(f"Total Trades: {final_status['performance']['total_trades']}")


def demonstrate_optimization():
    """
    Demonstrate how the system achieves target metrics through optimization.
    """
    logger.info("\n=== Optimization Strategy for Target Metrics ===")
    
    # 1. Win Rate Optimization (84-85%)
    logger.info("\n1. Win Rate Optimization:")
    logger.info("   - Conservative entry thresholds (z-score > 2.5)")
    logger.info("   - High confidence requirements (>0.95)")
    logger.info("   - Strict risk management (stop at z-score 3.5)")
    logger.info("   - Fee-optimized routing")
    
    # 2. Sharpe Ratio Optimization (4.0+)
    logger.info("\n2. Sharpe Ratio Optimization:")
    logger.info("   - Mean-variance portfolio optimization")
    logger.info("   - Minimum Sharpe requirement of 2.0 per position")
    logger.info("   - Correlation-based position sizing")
    logger.info("   - Fractional Kelly sizing (20% of full Kelly)")
    
    # 3. Drawdown Control (<3%)
    logger.info("\n3. Drawdown Control:")
    logger.info("   - Max 10% per position")
    logger.info("   - 2% daily VaR limit")
    logger.info("   - Dynamic position reduction at 80% of drawdown limit")
    logger.info("   - Automatic loss position closing")
    
    # 4. Execution Quality
    logger.info("\n4. Execution Quality:")
    logger.info("   - Smart order routing based on urgency")
    logger.info("   - Adaptive execution (VWAP for stat arb)")
    logger.info("   - Aggressive execution for time-sensitive arbitrage")
    logger.info("   - Real-time slippage monitoring")
    
    # 5. Opportunity Capture (5-10/day)
    logger.info("\n5. Opportunity Capture:")
    logger.info("   - 1-second scan interval for cross-exchange")
    logger.info("   - 5-second scan for statistical arbitrage")
    logger.info("   - Priority queue ranking system")
    logger.info("   - Multi-factor opportunity scoring")


if __name__ == "__main__":
    # Show optimization strategy
    demonstrate_optimization()
    
    # Run the system
    asyncio.run(main())