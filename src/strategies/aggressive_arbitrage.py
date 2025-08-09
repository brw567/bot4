"""
Aggressive Arbitrage Strategy for 30%+ Monthly Returns
Single exchange (Binance) optimized for maximum profitability
"""
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import ccxt
from dataclasses import dataclass
import logging

# Define base class here for simplicity
class StrategyBase:
    """Base strategy class"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange = None
        
    async def scan_opportunities(self) -> List[Any]:
        raise NotImplementedError


@dataclass
class AggressiveOpportunity:
    """Enhanced opportunity with aggressive parameters"""
    pair: str
    exchange: str
    spread: float
    size: float
    confidence: float
    expected_profit: float
    execution_priority: int
    strategy_type: str
    entry_price: float
    target_price: float
    stop_loss: float
    leverage: float = 1.0
    

class AggressiveArbitrageStrategy(StrategyBase):
    """
    Aggressive arbitrage implementation targeting 30%+ monthly returns
    Uses multiple sub-strategies and aggressive position sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_spread = 0.0008  # 0.08% minimum
        self.confidence_threshold = 0.65  # Lower threshold
        self.max_position_pct = 0.05  # 5% max position
        self.use_leverage = True
        self.max_leverage = 3.0
        self.parallel_positions = 20
        self.capital_usage = 0.8  # Use 80% of capital
        
        # Sub-strategies
        self.strategies = {
            'spot_futures': self._spot_futures_arbitrage,
            'statistical': self._statistical_arbitrage,
            'momentum': self._momentum_scalping,
            'mean_reversion': self._mean_reversion,
            'triangular': self._triangular_arbitrage
        }
        
        # Performance tracking
        self.daily_pnl = 0
        self.daily_trades = 0
        self.winning_trades = 0
        self.current_drawdown = 0
        self.max_daily_loss = -0.07  # -7% circuit breaker
        
    async def scan_opportunities(self) -> List[AggressiveOpportunity]:
        """Scan all strategies for opportunities"""
        all_opportunities = []
        
        # Run all strategies in parallel
        tasks = []
        for strategy_name, strategy_func in self.strategies.items():
            tasks.append(strategy_func())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and rank opportunities
        for opportunities in results:
            if isinstance(opportunities, list):
                all_opportunities.extend(opportunities)
        
        # Rank by expected profit and confidence
        ranked = self._rank_opportunities(all_opportunities)
        
        # Apply position limits
        return self._apply_position_limits(ranked)
    
    async def _spot_futures_arbitrage(self) -> List[AggressiveOpportunity]:
        """Enhanced spot-futures arbitrage"""
        opportunities = []
        
        # Get top volume pairs
        pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        
        for symbol in pairs:
            try:
                # Get spot and futures prices
                spot_ticker = await self.exchange.fetch_ticker(symbol)
                futures_symbol = symbol.replace('USDT', 'USDTPERP')
                futures_ticker = await self.exchange.fetch_ticker(futures_symbol)
                
                spot_price = spot_ticker['last']
                futures_price = futures_ticker['last']
                
                # Calculate spread
                spread = abs(futures_price - spot_price) / spot_price
                
                if spread > self.min_spread:
                    # Get ML confidence
                    ml_confidence = await self._get_ml_confidence(symbol, spread)
                    
                    if ml_confidence > self.confidence_threshold:
                        # Calculate optimal size with leverage
                        size = self._calculate_aggressive_size(spread, ml_confidence)
                        
                        opp = AggressiveOpportunity(
                            pair=symbol,
                            exchange='binance',
                            spread=spread,
                            size=size,
                            confidence=ml_confidence,
                            expected_profit=spread * size * spot_price,
                            execution_priority=1,  # Highest priority
                            strategy_type='spot_futures',
                            entry_price=spot_price,
                            target_price=futures_price,
                            stop_loss=spot_price * 0.995,  # 0.5% stop
                            leverage=2.0  # Use 2x leverage
                        )
                        opportunities.append(opp)
                        
            except Exception as e:
                logging.error(f"Error in spot-futures arb for {symbol}: {e}")
        
        return opportunities
    
    async def _statistical_arbitrage(self) -> List[AggressiveOpportunity]:
        """Pairs trading with aggressive parameters"""
        opportunities = []
        
        # Cointegrated pairs
        pairs = [
            ('BTCUSDT', 'ETHUSDT', 0.067),  # Historical ratio
            ('BNBUSDT', 'ETHUSDT', 0.15),
            ('SOLUSDT', 'ADAUSDT', 50.0)
        ]
        
        for pair1, pair2, mean_ratio in pairs:
            try:
                ticker1 = await self.exchange.fetch_ticker(pair1)
                ticker2 = await self.exchange.fetch_ticker(pair2)
                
                current_ratio = ticker1['last'] / ticker2['last']
                z_score = (current_ratio - mean_ratio) / (mean_ratio * 0.02)  # 2% std dev
                
                if abs(z_score) > 1.5:  # Aggressive threshold
                    # Mean reversion opportunity
                    confidence = min(0.95, 0.65 + abs(z_score) * 0.1)
                    
                    if confidence > self.confidence_threshold:
                        size = self._calculate_aggressive_size(abs(z_score) * 0.01, confidence)
                        
                        opp = AggressiveOpportunity(
                            pair=f"{pair1}/{pair2}",
                            exchange='binance',
                            spread=abs(z_score) * 0.01,
                            size=size,
                            confidence=confidence,
                            expected_profit=abs(z_score) * 0.01 * size * ticker1['last'],
                            execution_priority=2,
                            strategy_type='statistical',
                            entry_price=current_ratio,
                            target_price=mean_ratio,
                            stop_loss=current_ratio * (1.02 if z_score > 0 else 0.98),
                            leverage=1.5
                        )
                        opportunities.append(opp)
                        
            except Exception as e:
                logging.error(f"Error in statistical arb: {e}")
        
        return opportunities
    
    async def _momentum_scalping(self) -> List[AggressiveOpportunity]:
        """Aggressive momentum-based scalping"""
        opportunities = []
        
        # High volume pairs for scalping
        pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in pairs:
            try:
                # Get recent candles
                candles = await self.exchange.fetch_ohlcv(symbol, '1m', limit=20)
                
                if len(candles) >= 20:
                    closes = [c[4] for c in candles]
                    volumes = [c[5] for c in candles]
                    
                    # Calculate momentum indicators
                    momentum = (closes[-1] - closes[-5]) / closes[-5]  # 5-min momentum
                    volume_surge = volumes[-1] / np.mean(volumes[:-1])
                    
                    if abs(momentum) > 0.002 and volume_surge > 2.0:
                        # Strong momentum with volume
                        confidence = min(0.85, 0.65 + volume_surge * 0.05)
                        
                        if confidence > self.confidence_threshold:
                            size = self._calculate_aggressive_size(abs(momentum), confidence)
                            
                            opp = AggressiveOpportunity(
                                pair=symbol,
                                exchange='binance',
                                spread=abs(momentum),
                                size=size,
                                confidence=confidence,
                                expected_profit=abs(momentum) * size * closes[-1],
                                execution_priority=3,
                                strategy_type='momentum',
                                entry_price=closes[-1],
                                target_price=closes[-1] * (1.003 if momentum > 0 else 0.997),
                                stop_loss=closes[-1] * (0.997 if momentum > 0 else 1.003),
                                leverage=2.5
                            )
                            opportunities.append(opp)
                            
            except Exception as e:
                logging.error(f"Error in momentum scalping: {e}")
        
        return opportunities
    
    async def _mean_reversion(self) -> List[AggressiveOpportunity]:
        """Mean reversion on oversold/overbought conditions"""
        opportunities = []
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']:
            try:
                # Get hourly candles for RSI
                candles = await self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
                
                if len(candles) >= 14:
                    closes = np.array([c[4] for c in candles])
                    
                    # Calculate RSI
                    rsi = self._calculate_rsi(closes, period=14)
                    
                    # Bollinger Bands
                    sma = np.mean(closes[-20:])
                    std = np.std(closes[-20:])
                    upper_band = sma + 2 * std
                    lower_band = sma - 2 * std
                    
                    current_price = closes[-1]
                    
                    # Check for extreme conditions
                    if rsi < 25 or current_price < lower_band:
                        # Oversold
                        confidence = 0.70 + (30 - rsi) * 0.01
                        spread = (sma - current_price) / current_price
                        
                        if confidence > self.confidence_threshold and spread > 0.001:
                            size = self._calculate_aggressive_size(spread, confidence)
                            
                            opp = AggressiveOpportunity(
                                pair=symbol,
                                exchange='binance',
                                spread=spread,
                                size=size,
                                confidence=confidence,
                                expected_profit=spread * size * current_price,
                                execution_priority=4,
                                strategy_type='mean_reversion',
                                entry_price=current_price,
                                target_price=sma,
                                stop_loss=current_price * 0.98,
                                leverage=1.0  # No leverage for mean reversion
                            )
                            opportunities.append(opp)
                            
                    elif rsi > 75 or current_price > upper_band:
                        # Overbought - short opportunity
                        confidence = 0.70 + (rsi - 70) * 0.01
                        spread = (current_price - sma) / current_price
                        
                        if confidence > self.confidence_threshold and spread > 0.001:
                            size = self._calculate_aggressive_size(spread, confidence)
                            
                            opp = AggressiveOpportunity(
                                pair=symbol,
                                exchange='binance',
                                spread=spread,
                                size=size,
                                confidence=confidence,
                                expected_profit=spread * size * current_price,
                                execution_priority=4,
                                strategy_type='mean_reversion_short',
                                entry_price=current_price,
                                target_price=sma,
                                stop_loss=current_price * 1.02,
                                leverage=1.0
                            )
                            opportunities.append(opp)
                            
            except Exception as e:
                logging.error(f"Error in mean reversion: {e}")
        
        return opportunities
    
    async def _triangular_arbitrage(self) -> List[AggressiveOpportunity]:
        """Aggressive triangular arbitrage"""
        opportunities = []
        
        # Triangle paths
        triangles = [
            ('BTCUSDT', 'ETHBTC', 'ETHUSDT'),
            ('BNBUSDT', 'BNBBTC', 'BTCUSDT'),
            ('ETHUSDT', 'BNBETH', 'BNBUSDT')
        ]
        
        for path in triangles:
            try:
                # Get all three prices
                tickers = await asyncio.gather(
                    self.exchange.fetch_ticker(path[0]),
                    self.exchange.fetch_ticker(path[1]),
                    self.exchange.fetch_ticker(path[2])
                )
                
                # Calculate arbitrage
                # Path: USD -> A -> B -> USD
                rate1 = 1 / tickers[0]['last']  # USD to A
                rate2 = tickers[1]['last']       # A to B
                rate3 = tickers[2]['last']       # B to USD
                
                final_rate = rate1 * rate2 * rate3
                profit = final_rate - 1
                
                if profit > self.min_spread:
                    confidence = min(0.90, 0.70 + profit * 50)
                    
                    if confidence > self.confidence_threshold:
                        size = self._calculate_aggressive_size(profit, confidence)
                        
                        opp = AggressiveOpportunity(
                            pair=f"{path[0]}-{path[1]}-{path[2]}",
                            exchange='binance',
                            spread=profit,
                            size=size,
                            confidence=confidence,
                            expected_profit=profit * size * tickers[0]['last'],
                            execution_priority=2,  # High priority
                            strategy_type='triangular',
                            entry_price=1.0,
                            target_price=final_rate,
                            stop_loss=0.995,  # 0.5% stop
                            leverage=2.0
                        )
                        opportunities.append(opp)
                        
            except Exception as e:
                logging.error(f"Error in triangular arb: {e}")
        
        return opportunities
    
    def _calculate_aggressive_size(self, spread: float, confidence: float) -> float:
        """Calculate position size with aggressive parameters"""
        base_size = self.capital * self.max_position_pct
        
        # Spread multiplier (bigger spread = bigger position)
        spread_mult = min(2.0, 1 + spread * 50)
        
        # Confidence multiplier
        conf_mult = confidence ** 0.5  # Square root for smoother scaling
        
        # Win rate multiplier
        if self.daily_trades > 10:
            win_rate = self.winning_trades / self.daily_trades
            if win_rate > 0.85:
                win_mult = 1.3
            elif win_rate > 0.75:
                win_mult = 1.1
            elif win_rate < 0.60:
                win_mult = 0.7
            else:
                win_mult = 1.0
        else:
            win_mult = 1.0
        
        # Drawdown adjustment
        if self.current_drawdown < -0.03:
            dd_mult = 0.5  # Cut size in half if down 3%
        elif self.current_drawdown > 0.02:
            dd_mult = 1.2  # Increase size if up 2%
        else:
            dd_mult = 1.0
        
        final_size = base_size * spread_mult * conf_mult * win_mult * dd_mult
        
        # Apply capital usage limit
        available_capital = self.capital * self.capital_usage - self._get_current_exposure()
        
        return min(final_size, available_capital)
    
    def _rank_opportunities(self, opportunities: List[AggressiveOpportunity]) -> List[AggressiveOpportunity]:
        """Rank opportunities by expected profit and priority"""
        # Sort by execution priority first, then expected profit
        return sorted(opportunities, 
                     key=lambda x: (-x.execution_priority, -x.expected_profit))
    
    def _apply_position_limits(self, opportunities: List[AggressiveOpportunity]) -> List[AggressiveOpportunity]:
        """Apply position and risk limits"""
        selected = []
        total_exposure = self._get_current_exposure()
        position_count = len(self.active_positions)
        
        for opp in opportunities:
            # Check position count
            if position_count >= self.parallel_positions:
                break
            
            # Check exposure
            if total_exposure + opp.size > self.capital * self.capital_usage:
                continue
            
            # Check daily loss limit
            if self.daily_pnl < self.max_daily_loss * self.capital:
                logging.warning("Daily loss limit reached, stopping trading")
                break
            
            selected.append(opp)
            total_exposure += opp.size
            position_count += 1
        
        return selected
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_current_exposure(self) -> float:
        """Get total current exposure"""
        return sum(pos.size for pos in self.active_positions.values())
    
    async def _get_ml_confidence(self, symbol: str, spread: float) -> float:
        """Get ML model confidence (simplified)"""
        # In production, this would call your ML predictor
        # For now, return confidence based on spread
        base_confidence = 0.65
        spread_bonus = min(0.25, spread * 50)
        return base_confidence + spread_bonus
    
    async def execute_opportunity(self, opp: AggressiveOpportunity) -> Dict[str, Any]:
        """Execute aggressive trading opportunity"""
        try:
            # Use market orders for speed
            if opp.strategy_type in ['spot_futures', 'triangular']:
                # Execute both legs simultaneously
                orders = await self._execute_arbitrage_legs(opp)
            else:
                # Single order execution
                order = await self._execute_single_order(opp)
                orders = [order]
            
            # Update tracking
            self.daily_trades += 1
            
            return {
                'status': 'success',
                'orders': orders,
                'opportunity': opp
            }
            
        except Exception as e:
            logging.error(f"Execution error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'opportunity': opp
            }
    
    async def _execute_arbitrage_legs(self, opp: AggressiveOpportunity) -> List[Dict]:
        """Execute multi-leg arbitrage trades"""
        if opp.strategy_type == 'spot_futures':
            # Buy spot, sell futures
            spot_order = self.exchange.create_market_buy_order(
                opp.pair, opp.size / opp.entry_price
            )
            
            futures_symbol = opp.pair.replace('USDT', 'USDTPERP')
            futures_order = self.exchange.create_market_sell_order(
                futures_symbol, opp.size / opp.target_price
            )
            
            return [spot_order, futures_order]
            
        elif opp.strategy_type == 'triangular':
            # Execute three legs of triangular arbitrage
            # Implementation depends on specific path
            pass
        
        return []
    
    async def _execute_single_order(self, opp: AggressiveOpportunity) -> Dict:
        """Execute single order with aggressive parameters"""
        side = 'buy' if opp.strategy_type != 'mean_reversion_short' else 'sell'
        
        # Use leverage if specified
        if opp.leverage > 1:
            await self.exchange.set_leverage(opp.leverage, opp.pair)
        
        # Market order for immediate execution
        order = await self.exchange.create_market_order(
            opp.pair,
            side,
            opp.size / opp.entry_price
        )
        
        # Set stop loss and take profit
        await self._set_stop_take_profit(order, opp)
        
        return order
    
    async def _set_stop_take_profit(self, order: Dict, opp: AggressiveOpportunity):
        """Set stop loss and take profit orders"""
        if order['status'] == 'closed':
            # Set stop loss
            stop_side = 'sell' if order['side'] == 'buy' else 'buy'
            await self.exchange.create_order(
                opp.pair,
                'stop_loss_limit',
                stop_side,
                order['amount'],
                opp.stop_loss,
                {'stopPrice': opp.stop_loss}
            )
            
            # Set take profit
            await self.exchange.create_order(
                opp.pair,
                'take_profit_limit',
                stop_side,
                order['amount'],
                opp.target_price,
                {'stopPrice': opp.target_price}
            )
    
    def update_daily_metrics(self, trade_result: Dict[str, Any]):
        """Update daily performance metrics"""
        if trade_result['status'] == 'success':
            # Calculate P&L
            pnl = trade_result.get('pnl', 0)
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Update drawdown
            if self.daily_pnl < self.current_drawdown:
                self.current_drawdown = self.daily_pnl
    
    def reset_daily_metrics(self):
        """Reset metrics at start of new day"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.winning_trades = 0
        self.current_drawdown = 0