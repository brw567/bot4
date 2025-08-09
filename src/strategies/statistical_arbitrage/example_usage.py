"""
Example usage of the Statistical Arbitrage Engine.
Demonstrates pair discovery, backtesting, and live trading simulation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List

from strategies.statistical_arbitrage import (
    StatArbEngine,
    CointegrationEngine,
    StatArbBacktester,
    RealTimeMonitor,
    SignalType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Generate sample price data for demonstration.
    Creates correlated pairs that exhibit mean reversion.
    """
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    
    # Generate base price series
    np.random.seed(42)
    
    # Market factor (common trend)
    market = np.cumsum(np.random.normal(0.0005, 0.01, len(dates)))
    
    # Generate correlated pairs
    data = {}
    
    # Pair 1: BTC and ETH (highly correlated)
    btc_noise = np.cumsum(np.random.normal(0, 0.005, len(dates)))
    eth_noise = np.cumsum(np.random.normal(0, 0.005, len(dates)))
    
    data['BTC/USDT'] = pd.DataFrame({
        'close': 40000 * np.exp(market + btc_noise),
        'volume': np.random.uniform(1000, 2000, len(dates))
    }, index=dates)
    
    data['ETH/USDT'] = pd.DataFrame({
        'close': 3000 * np.exp(market * 0.8 + eth_noise),
        'volume': np.random.uniform(5000, 10000, len(dates))
    }, index=dates)
    
    # Pair 2: BNB and SOL (moderately correlated)
    bnb_noise = np.cumsum(np.random.normal(0, 0.007, len(dates)))
    sol_noise = np.cumsum(np.random.normal(0, 0.008, len(dates)))
    
    data['BNB/USDT'] = pd.DataFrame({
        'close': 300 * np.exp(market * 0.6 + bnb_noise),
        'volume': np.random.uniform(2000, 4000, len(dates))
    }, index=dates)
    
    data['SOL/USDT'] = pd.DataFrame({
        'close': 100 * np.exp(market * 0.6 + sol_noise),
        'volume': np.random.uniform(3000, 6000, len(dates))
    }, index=dates)
    
    # Add mean-reverting spread
    for i in range(1, len(dates)):
        # Add mean reversion to BTC/ETH pair
        spread = data['BTC/USDT']['close'].iloc[i] / data['ETH/USDT']['close'].iloc[i]
        target_ratio = 13.33  # Target BTC/ETH ratio
        
        if spread > target_ratio * 1.02:
            data['BTC/USDT']['close'].iloc[i] *= 0.998
            data['ETH/USDT']['close'].iloc[i] *= 1.002
        elif spread < target_ratio * 0.98:
            data['BTC/USDT']['close'].iloc[i] *= 1.002
            data['ETH/USDT']['close'].iloc[i] *= 0.998
    
    return data


async def example_pair_discovery():
    """Demonstrate cointegration pair discovery."""
    print("\n" + "="*60)
    print("STATISTICAL ARBITRAGE - PAIR DISCOVERY")
    print("="*60)
    
    # Generate sample data
    price_data = await generate_sample_data()
    
    # Extract close prices for cointegration testing
    close_prices = {symbol: df['close'] for symbol, df in price_data.items()}
    
    # Initialize cointegration engine
    coint_engine = CointegrationEngine(lookback_days=60)
    
    # Find cointegrated pairs
    print("\nSearching for cointegrated pairs...")
    pairs = coint_engine.find_cointegrated_pairs(
        close_prices,
        min_half_life=5,
        max_half_life=20,
        min_correlation=0.5
    )
    
    # Display results
    print(f"\nFound {len(pairs)} cointegrated pairs:")
    print("-" * 80)
    print(f"{'Pair':<20} {'P-Value':<10} {'Half-Life':<10} {'Sharpe':<10} {'Score':<10}")
    print("-" * 80)
    
    for pair in pairs[:5]:  # Show top 5
        pair_str = f"{pair.symbols[0]}-{pair.symbols[1]}"
        print(f"{pair_str:<20} {pair.cointegration_score:.4f}    "
              f"{pair.half_life:>8.1f}  {pair.sharpe_ratio:>8.2f}  "
              f"{pair.profitability_score:>8.3f}")
    
    return pairs, price_data


async def example_backtesting(pairs: List, price_data: Dict[str, pd.DataFrame]):
    """Demonstrate backtesting of statistical arbitrage strategy."""
    print("\n" + "="*60)
    print("BACKTESTING STATISTICAL ARBITRAGE")
    print("="*60)
    
    # Initialize backtester
    backtester = StatArbBacktester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Test different parameter combinations
    param_sets = [
        {"entry_z": 2.0, "exit_z": 0.0, "kelly_fraction": 0.25},
        {"entry_z": 2.5, "exit_z": 0.5, "kelly_fraction": 0.20},
        {"entry_z": 1.5, "exit_z": 0.0, "kelly_fraction": 0.30},
    ]
    
    print("\nTesting different parameter combinations:")
    print("-" * 100)
    print(f"{'Entry Z':<10} {'Exit Z':<10} {'Kelly':<10} {'Return':<10} "
          f"{'Sharpe':<10} {'MaxDD':<10} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 100)
    
    best_result = None
    best_sharpe = -np.inf
    
    for params in param_sets:
        result = backtester.backtest(
            price_data,
            entry_z=params["entry_z"],
            exit_z=params["exit_z"],
            kelly_fraction=params["kelly_fraction"],
            lookback_days=60,
            max_pairs=3
        )
        
        print(f"{params['entry_z']:<10.1f} {params['exit_z']:<10.1f} "
              f"{params['kelly_fraction']:<10.2f} {result.total_return:>9.1%} "
              f"{result.sharpe_ratio:>9.2f} {result.max_drawdown:>9.1%} "
              f"{result.total_trades:>9} {result.win_rate:>9.1%}")
        
        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_result = result
    
    # Display best result details
    print(f"\nBest parameters: {best_result.parameters}")
    print(f"Total return: {best_result.total_return:.1%}")
    print(f"Sharpe ratio: {best_result.sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {best_result.max_drawdown:.1%}")
    print(f"Win rate: {best_result.win_rate:.1%}")
    
    # Show sample trades
    print("\nSample trades from backtest:")
    print("-" * 80)
    print(f"{'Pair':<20} {'Entry Date':<12} {'Exit Date':<12} "
          f"{'Duration':<10} {'Return':<10}")
    print("-" * 80)
    
    for trade in best_result.trade_history[:5]:
        if 'exit_date' in trade:
            pair_str = f"{trade['pair'][0]}-{trade['pair'][1]}"
            print(f"{pair_str:<20} {trade['entry_date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['exit_date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['duration_days']:>9}d {trade['return']:>9.2%}")
    
    return best_result


async def example_live_trading_simulation():
    """Demonstrate live trading simulation with real-time monitoring."""
    print("\n" + "="*60)
    print("LIVE TRADING SIMULATION")
    print("="*60)
    
    # Generate initial data
    price_data = await generate_sample_data()
    close_prices = {symbol: df['close'] for symbol, df in price_data.items()}
    
    # Initialize engine
    engine = StatArbEngine(
        lookback_days=60,
        entry_z_score=2.0,
        exit_z_score=0.0,
        stop_z_score=3.5,
        max_pairs=3,
        kelly_fraction=0.25
    )
    
    # Initialize with historical data
    historical_prices = {symbol: series[:-50] for symbol, series in close_prices.items()}
    await engine.initialize(historical_prices)
    
    # Initialize real-time monitor
    monitor = RealTimeMonitor(engine)
    
    # Simulate live trading for last 50 days
    live_data = {symbol: series[-50:] for symbol, series in close_prices.items()}
    
    print("\nSimulating live trading...")
    print("-" * 80)
    
    total_pnl = 0
    capital = 100000
    
    for i in range(len(live_data['BTC/USDT'])):
        # Get current prices
        current_prices = {
            symbol: float(series.iloc[i]) 
            for symbol, series in live_data.items()
        }
        
        current_date = live_data['BTC/USDT'].index[i]
        
        # Scan for opportunities
        signals = engine.scan_opportunities(current_prices)
        
        # Process signals
        for signal in signals:
            if signal.signal_type == SignalType.LONG_SPREAD:
                print(f"\n{current_date.strftime('%Y-%m-%d')} - "
                      f"LONG {signal.pair[0]}-{signal.pair[1]} "
                      f"@ z-score {signal.z_score:.2f}")
                print(f"  Suggested size: ${signal.suggested_size * capital:.0f}")
                
                # Simulate opening position
                engine.open_position(signal, signal.suggested_size * capital)
                
            elif signal.signal_type == SignalType.SHORT_SPREAD:
                print(f"\n{current_date.strftime('%Y-%m-%d')} - "
                      f"SHORT {signal.pair[0]}-{signal.pair[1]} "
                      f"@ z-score {signal.z_score:.2f}")
                print(f"  Suggested size: ${signal.suggested_size * capital:.0f}")
                
                engine.open_position(signal, signal.suggested_size * capital)
                
            elif signal.signal_type == SignalType.EXIT:
                exit_reason = signal.metadata.get('exit_reason', 'unknown')
                pnl = signal.metadata.get('pnl', 0)
                pnl_pct = signal.metadata.get('pnl_percent', 0)
                
                print(f"\n{current_date.strftime('%Y-%m-%d')} - "
                      f"EXIT {signal.pair[0]}-{signal.pair[1]} "
                      f"@ z-score {signal.z_score:.2f}")
                print(f"  Reason: {exit_reason}")
                print(f"  P&L: ${pnl:.2f} ({pnl_pct:.2%})")
                
                total_pnl += pnl
                capital += pnl
                
                engine.close_position(signal, current_prices)
    
    # Final summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    performance = engine.get_performance_report()
    
    print(f"\nTotal trades: {performance['total_trades']}")
    print(f"Win rate: {performance['win_rate']:.1%}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Return: {(capital / 100000 - 1):.1%}")
    print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    
    # Show active positions
    positions = engine.get_active_positions_summary()
    if positions['total_positions'] > 0:
        print(f"\nActive positions: {positions['total_positions']}")
        for pos in positions['positions']:
            print(f"  {pos['pair'][0]}-{pos['pair'][1]}: "
                  f"z-score {pos['z_score']:.2f}, "
                  f"P&L {pos['pnl_percent']:.2%}")


async def example_walk_forward():
    """Demonstrate walk-forward analysis."""
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS")
    print("="*60)
    
    # Generate data
    price_data = await generate_sample_data()
    
    # Initialize backtester
    backtester = StatArbBacktester()
    
    # Define parameter grid
    param_grid = {
        'entry_z': [1.5, 2.0, 2.5],
        'exit_z': [0.0, 0.5],
        'kelly_fraction': [0.20, 0.25, 0.30]
    }
    
    print("\nPerforming walk-forward analysis...")
    print("This tests parameter stability across multiple time periods.")
    
    # Run walk-forward analysis
    wf_result = backtester.walk_forward_analysis(
        price_data,
        param_grid,
        in_sample_periods=252,  # 1 year
        out_sample_periods=63,  # 3 months
        n_windows=3
    )
    
    print("\nWalk-Forward Results:")
    print("-" * 60)
    print(f"Stability score: {wf_result.stability_score:.2f}")
    print(f"Overfitting score: {wf_result.overfitting_score:.2f}")
    
    print("\nIn-sample vs Out-of-sample performance:")
    for i, (is_res, oos_res) in enumerate(zip(
        wf_result.in_sample_results, 
        wf_result.out_sample_results
    )):
        print(f"\nWindow {i+1}:")
        print(f"  In-sample Sharpe: {is_res.sharpe_ratio:.2f}")
        print(f"  Out-sample Sharpe: {oos_res.sharpe_ratio:.2f}")
        print(f"  Best params: {wf_result.best_parameters[i]}")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("STATISTICAL ARBITRAGE ENGINE - DEMONSTRATION")
    print("="*60)
    print("\nThis example demonstrates:")
    print("1. Cointegration pair discovery")
    print("2. Backtesting with parameter optimization") 
    print("3. Live trading simulation")
    print("4. Walk-forward analysis")
    
    # Run examples
    pairs, price_data = await example_pair_discovery()
    
    if pairs:
        await example_backtesting(pairs, price_data)
        await example_live_trading_simulation()
        await example_walk_forward()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())